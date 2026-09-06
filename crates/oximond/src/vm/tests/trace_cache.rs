//! Broker integration and the trace cache.
//!
//! These tests drive a tracing VM with the broker's synchronous adapter, so
//! install points are deterministic: a request finishes on the poll the VM
//! makes at its next loop-header consult, optionally after a set number of
//! empty polls. They compare results with an isolated interpreter and read
//! the trace counters, because equal results alone cannot tell a trace from
//! a slow path.

use super::trace_record::*;
use super::*;
use crate::vm::jit::trace::cache::CacheLimits;
use crate::vm::jit::trace::cache::MAX_NEGATIVE_SITES;
use crate::vm::jit::trace::policy::{RejectCause, SiteState};
use crate::vm::jit::trace::{TraceExitKind, TraceStatistics};
use crate::vm::stack::CallStack;

/// A tracing Lua 5.1 VM whose trace compiles run on the synchronous adapter,
/// finishing a request after `delay` polls.
fn vm_with(delay: u32) -> Vm {
    let mut vm =
        Vm::with_profile_and_execution_mode(CompatibilityProfile::Lua51, ExecutionMode::Tracing);
    vm.use_delayed_synchronous_jit_compilation(delay);
    vm
}

fn stats_of(vm: &Vm) -> TraceStatistics {
    vm.jit_statistics().trace
}

/// Steps `frames` to completion.
fn finish(vm: &mut Vm, frames: &mut CallStack) -> Vec<Value> {
    for _ in 0..5_000_000 {
        if let Some(results) = vm.step(frames).unwrap() {
            return results;
        }
    }
    panic!("the program does not finish");
}

/// Steps until `done` holds. Panics after `limit` steps.
fn step_until(vm: &mut Vm, frames: &mut CallStack, limit: usize, done: impl Fn(&Vm) -> bool) {
    for _ in 0..limit {
        if done(vm) {
            return;
        }
        assert!(
            vm.step(frames).unwrap().is_none(),
            "the program ended early: {:?}",
            stats_of(vm)
        );
    }
    panic!("condition not reached: {:?}", stats_of(vm));
}

/// One numeric-for sum loop after another, each over its own registers
/// (five per loop). Loop `k` has its header at pc `7k + 5`. The result is
/// the last loop's sum.
fn sequential_sums(limits: &[f64]) -> Chunk {
    let mut constants = vec![Constant::Number(1.0), Constant::Number(0.0)];
    constants.extend(limits.iter().map(|limit| Constant::Number(*limit)));
    let mut code = Vec::new();
    for (index, _) in limits.iter().enumerate() {
        let start = 7 * index;
        let base = u8::try_from(5 * index).unwrap();
        code.push(load(base, 0));
        code.push(load(base + 1, u16::try_from(2 + index).unwrap()));
        code.push(load(base + 2, 0));
        code.push(load(base + 4, 1));
        code.push(for_prep(base, start + 4, start + 6));
        code.push(binary(OpCode::Add, base + 4, base + 4, base + 3));
        code.push(for_loop(base, start + 6, start + 5));
    }
    code.push(Instruction::return_(
        u8::try_from(5 * (limits.len() - 1) + 4).unwrap(),
    ));
    Chunk::new(u8::try_from(5 * limits.len()).unwrap(), constants, code)
}

fn header(index: usize) -> usize {
    7 * index + 5
}

// ------------------------------------------------------------------ broker

#[test]
fn a_closed_trace_installs_at_a_later_safepoint_while_the_interpreter_keeps_running() {
    let chunk = Arc::new(sum_program(50_000.0));
    let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
    let mut vm = vm_with(25);
    let mut frames = frames_for(&vm, &chunk, None);

    step_until(&mut vm, &mut frames, 100_000, |vm| {
        stats_of(vm).compile_submissions == 1
    });
    let stats = stats_of(&vm);
    assert_eq!(stats.traces_recorded, 1);
    assert_eq!(stats.traces_compiled, 0, "the request has not finished");
    assert_eq!(vm.jit.trace.pending_compiles(), 1);
    assert_eq!(
        vm.jit.trace.site_state_any(5),
        Some(SiteState::Compiling),
        "the site waits for the broker"
    );
    assert!(
        vm.jit.trace.cache().reserved_meta() > 0,
        "the IR is reserved"
    );

    // The adapter needs 25 polls and the interpreter polls once per
    // iteration, so ten more iterations leave the request outstanding.
    for _ in 0..20 {
        assert!(vm.step(&mut frames).unwrap().is_none());
    }
    assert_eq!(stats_of(&vm).traces_compiled, 0);
    assert_eq!(stats_of(&vm).native_entries, 0);

    assert_eq!(finish(&mut vm, &mut frames), expected);
    let stats = stats_of(&vm);
    assert_eq!(stats.traces_compiled, 1, "{stats:?}");
    assert!(stats.native_entries >= 1, "{stats:?}");
    // The interpreter ran the loop until the install: at least the 25 polls
    // the compile needed, so not every iteration was native.
    assert!(stats.iterations <= 50_000 - 25, "{stats:?}");
    assert!(stats.iterations > 49_000, "{stats:?}");
    assert_eq!(vm.jit.trace.pending_compiles(), 0);
    assert_eq!(
        vm.jit.trace.cache().reserved_meta(),
        0,
        "the reservation ended"
    );
    assert!(vm.jit.trace.cache().code_bytes() >= 4096);
    assert!(vm.jit.trace.cache().meta_bytes() > 0);
}

#[test]
fn compiles_install_in_submission_order() {
    // Loop 0 is short, so its request is still outstanding when loop 1
    // records and submits its own. Each request needs 100 polls.
    let chunk = Arc::new(sequential_sums(&[90.0, 2_000.0]));
    let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
    let mut vm = vm_with(100);
    let mut frames = frames_for(&vm, &chunk, None);
    step_until(&mut vm, &mut frames, 200_000, |vm| {
        stats_of(vm).compile_submissions == 2
    });
    assert_eq!(stats_of(&vm).traces_compiled, 0, "both are outstanding");
    assert_eq!(vm.jit.trace.pending_compiles(), 2);
    assert_eq!(finish(&mut vm, &mut frames), expected);

    let stats = stats_of(&vm);
    assert_eq!(stats.traces_compiled, 2, "{stats:?}");
    let first = vm.jit.trace.installed_index_at_pc(header(0)).unwrap();
    let second = vm.jit.trace.installed_index_at_pc(header(1)).unwrap();
    assert!(
        vm.jit.trace.installed_serial(first) < vm.jit.trace.installed_serial(second),
        "the broker finishes requests in submission order"
    );
}

#[test]
fn identical_chunks_closing_back_to_back_each_get_their_trace() {
    // The broker folds the second request into the first because the IR has
    // the same content. Both headers must still install.
    let first = Arc::new(sum_program(50_000.0));
    let second = Arc::new(sum_program(50_000.0));
    let expected = interpreted(CompatibilityProfile::Lua51, &first);
    let mut vm = vm_with(300);
    let mut frames = frames_for(&vm, &first, None);
    step_until(&mut vm, &mut frames, 100_000, |vm| {
        stats_of(vm).compile_submissions == 1
    });
    let mut other = frames_for(&vm, &second, None);
    step_until(&mut vm, &mut other, 100_000, |vm| {
        stats_of(vm).compile_submissions == 2
    });
    assert_eq!(stats_of(&vm).traces_compiled, 0, "both are outstanding");
    assert_eq!(finish(&mut vm, &mut other), expected);
    assert_eq!(finish(&mut vm, &mut frames), expected);
    let stats = stats_of(&vm);
    assert_eq!(stats.traces_compiled, 2, "{stats:?}");
    for chunk in [&first, &second] {
        assert!(
            matches!(
                vm.jit.trace.site_state(chunk, 5),
                Some(SiteState::Installed { .. })
            ),
            "{:?}",
            vm.jit.trace.site_state(chunk, 5)
        );
    }
}

#[test]
fn replacing_the_broker_forgets_outstanding_trace_compiles() {
    let chunk = Arc::new(sum_program(50_000.0));
    let mut vm = vm_with(100_000);
    let mut frames = frames_for(&vm, &chunk, None);
    step_until(&mut vm, &mut frames, 100_000, |vm| {
        stats_of(vm).compile_submissions == 1
    });
    assert_eq!(vm.jit.trace.pending_compiles(), 1);
    vm.use_synchronous_jit_compilation();
    assert_eq!(vm.jit.trace.pending_compiles(), 0);
    assert_eq!(vm.jit.trace.cache().reserved_meta(), 0);
}

#[test]
fn a_compile_finished_after_a_flush_never_installs() {
    let chunk = Arc::new(sum_program(60_000.0));
    let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
    let mut vm = vm_with(40);
    let mut frames = frames_for(&vm, &chunk, None);
    step_until(&mut vm, &mut frames, 100_000, |vm| {
        stats_of(vm).compile_submissions == 1
    });
    let epoch = vm.jit.trace.epoch();
    vm.runtime_mut().request_jit_flush();
    vm.service_jit_flush();
    assert_eq!(vm.jit.trace.epoch(), epoch + 1);
    assert_eq!(vm.jit.trace.pending_compiles(), 0);
    assert_eq!(vm.jit.trace.cache().reserved_meta(), 0);
    assert_eq!(stats_of(&vm).stale_completions, 1);

    // The worker still finishes the old request. Its result is discarded, and
    // the loop records and installs a new trace under the new epoch.
    assert_eq!(finish(&mut vm, &mut frames), expected);
    let stats = stats_of(&vm);
    assert!(stats.traces_recorded >= 2, "{stats:?}");
    assert_eq!(
        stats.traces_compiled,
        stats.traces_recorded - 1,
        "exactly the flushed request is lost: {stats:?}"
    );
    assert!(stats.native_entries >= 1, "{stats:?}");
}

#[test]
fn a_stale_epoch_is_rejected_even_when_the_broker_returns_the_result() {
    // Complete a request the policy no longer knows: the result is stale.
    let chunk = Arc::new(sum_program(60_000.0));
    let mut vm = vm_with(0);
    vm.jit.trace.set_native_compilation(false);
    let mut frames = frames_for(&vm, &chunk, None);
    step_until(&mut vm, &mut frames, 100_000, |vm| {
        !vm.jit.trace.queued().is_empty()
    });
    let ir = std::sync::Arc::new(vm.jit.trace.queued()[0].ir.clone());
    let mut broker = JitBroker::synchronous(4);
    let id = broker
        .submit_trace(ir, OptimizationTier::Baseline, 7)
        .unwrap();
    let done = broker.poll_trace().unwrap();
    let mut stats = TraceStatistics::default();
    vm.jit
        .trace
        .complete_compile(&mut stats, id, done.epoch(), done.into_output());
    assert_eq!(stats.stale_completions, 1);
    assert_eq!(stats.traces_compiled, 0);
}

#[test]
fn a_full_broker_queue_backs_the_site_off_without_spending_retries() {
    let chunk = Arc::new(sum_program(40_000.0));
    let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
    let mut vm = vm_with(0);
    // A one-slot broker already holding another request refuses the trace.
    vm.jit.broker = Some(JitBroker::synchronous_with_delay(1, 1_000_000));
    let other = {
        let (probe, _) = record(CompatibilityProfile::Lua51, &sum_program(300.0));
        Arc::new(only_trace(&probe).clone())
    };
    vm.jit
        .broker
        .as_mut()
        .unwrap()
        .submit_trace(other, OptimizationTier::Baseline, u64::MAX)
        .unwrap();
    let mut frames = frames_for(&vm, &chunk, None);
    assert_eq!(finish(&mut vm, &mut frames), expected);
    let stats = stats_of(&vm);
    assert!(stats.compile_refusals >= 1, "{stats:?}");
    assert_eq!(stats.compile_submissions, 0, "{stats:?}");
    assert_eq!(stats.traces_compiled, 0);
    // A refusal is not the trace's fault, so the site is not rejected.
    assert!(matches!(
        vm.jit.trace.site_state_any(5),
        Some(SiteState::Cooldown { .. })
    ));
}

#[test]
fn a_full_broker_queue_holds_off_new_recordings_until_a_compile_finishes() {
    let chunk = Arc::new(sum_program(40_000.0));
    let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
    let mut vm = vm_with(0);
    // A one-slot broker whose request never finishes refuses every trace.
    vm.jit.broker = Some(JitBroker::synchronous_with_delay(1, u32::MAX));
    let other = {
        let (probe, _) = record(CompatibilityProfile::Lua51, &sum_program(300.0));
        Arc::new(only_trace(&probe).clone())
    };
    vm.jit
        .broker
        .as_mut()
        .unwrap()
        .submit_trace(other, OptimizationTier::Baseline, u64::MAX)
        .unwrap();
    let mut frames = frames_for(&vm, &chunk, None);
    assert_eq!(finish(&mut vm, &mut frames), expected);
    let stats = stats_of(&vm);
    // Without the wait, every hot period after a refusal records again:
    // about 600 recordings over this loop.
    assert!(stats.compile_refusals >= 1, "{stats:?}");
    assert!(stats.recording_starts <= 4, "{stats:?}");
}

// ------------------------------------------------------------------- caps

#[test]
fn the_trace_count_cap_evicts_the_least_recently_entered_trace() {
    let chunk = Arc::new(sequential_sums(&[3_000.0, 3_000.0, 3_000.0]));
    let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
    let mut vm = vm_with(0);
    vm.jit.trace.cache_mut().set_limits(CacheLimits {
        traces: 2,
        ..CacheLimits::default()
    });
    let mut frames = frames_for(&vm, &chunk, None);
    assert_eq!(finish(&mut vm, &mut frames), expected);
    let stats = stats_of(&vm);
    assert_eq!(stats.traces_compiled, 3, "{stats:?}");
    assert_eq!(stats.cache_evictions, 1, "{stats:?}");
    assert_eq!(vm.jit.trace.installed_count(), 2);
    // Loop 0 was entered first and not since, so it is the one evicted. Its
    // header counts again from cold.
    assert_eq!(
        vm.jit.trace.site_state_any(header(0)),
        Some(SiteState::Cold { backedges: 0 })
    );
    assert!(vm.jit.trace.installed_index_at_pc(header(1)).is_some());
    assert!(vm.jit.trace.installed_index_at_pc(header(2)).is_some());
}

#[test]
fn the_executable_byte_cap_makes_room_or_refuses() {
    let chunk = Arc::new(sequential_sums(&[3_000.0, 3_000.0]));
    let expected = interpreted(CompatibilityProfile::Lua51, &chunk);

    // Room for one trace's pages: the second install evicts the first.
    let mut vm = vm_with(0);
    vm.jit.trace.cache_mut().set_limits(CacheLimits {
        code_bytes: 4096,
        ..CacheLimits::default()
    });
    let mut frames = frames_for(&vm, &chunk, None);
    assert_eq!(finish(&mut vm, &mut frames), expected);
    let stats = stats_of(&vm);
    assert_eq!(stats.traces_compiled, 2, "{stats:?}");
    assert_eq!(stats.cache_evictions, 1, "{stats:?}");
    assert!(vm.jit.trace.cache().code_bytes() <= 4096);

    // Less than one page: no trace fits. The loops still run, interpreted.
    let mut vm = vm_with(0);
    vm.jit.trace.cache_mut().set_limits(CacheLimits {
        code_bytes: 100,
        ..CacheLimits::default()
    });
    let mut frames = frames_for(&vm, &chunk, None);
    assert_eq!(finish(&mut vm, &mut frames), expected);
    let stats = stats_of(&vm);
    assert_eq!(stats.traces_compiled, 0, "{stats:?}");
    assert!(stats.cache_rejections >= 1, "{stats:?}");
    assert_eq!(stats.native_entries, 0);
    assert_eq!(vm.jit.trace.installed_count(), 0);
    assert_eq!(vm.jit.trace.cache().code_bytes(), 0);
}

#[test]
fn the_metadata_cap_stops_requests_before_the_broker_sees_them() {
    let chunk = Arc::new(sum_program(20_000.0));
    let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
    let mut vm = vm_with(0);
    vm.jit.trace.cache_mut().set_limits(CacheLimits {
        meta_bytes: 64,
        ..CacheLimits::default()
    });
    let mut frames = frames_for(&vm, &chunk, None);
    assert_eq!(finish(&mut vm, &mut frames), expected);
    let stats = stats_of(&vm);
    assert!(stats.traces_recorded >= 1, "{stats:?}");
    assert!(stats.cache_rejections >= 1, "{stats:?}");
    assert_eq!(stats.compile_submissions, 0, "nothing reaches the broker");
    assert_eq!(vm.jit.trace.cache().reserved_meta(), 0);
    assert_eq!(stats.native_entries, 0);
}

#[test]
fn metadata_admission_evicts_cold_traces_to_make_room() {
    let chunk = Arc::new(sequential_sums(&[3_000.0, 3_000.0]));
    let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
    // Measure one trace's metadata, then allow exactly that.
    let mut probe = vm_with(0);
    let mut frames = frames_for(&probe, &chunk, None);
    finish(&mut probe, &mut frames);
    let one = probe.jit.trace.cache().meta_bytes() / 2;
    assert!(one > 0);

    let mut vm = vm_with(0);
    vm.jit.trace.cache_mut().set_limits(CacheLimits {
        meta_bytes: one + one / 2,
        ..CacheLimits::default()
    });
    let mut frames = frames_for(&vm, &chunk, None);
    assert_eq!(finish(&mut vm, &mut frames), expected);
    let stats = stats_of(&vm);
    assert_eq!(stats.traces_compiled, 2, "{stats:?}");
    assert_eq!(stats.cache_evictions, 1, "{stats:?}");
    assert!(vm.jit.trace.cache().meta_bytes() <= one + one / 2);
}

#[test]
fn a_refused_request_evicts_nothing_and_a_full_queue_backs_off_briefly() {
    let chunk = Arc::new(sequential_sums(&[3_000.0, 3_000.0]));
    let mut probe = vm_with(0);
    let mut frames = frames_for(&probe, &chunk, None);
    finish(&mut probe, &mut frames);
    let one = probe.jit.trace.cache().meta_bytes() / 2;

    let mut vm = vm_with(0);
    vm.jit.trace.cache_mut().set_limits(CacheLimits {
        meta_bytes: one + one / 2,
        ..CacheLimits::default()
    });
    let mut frames = frames_for(&vm, &chunk, None);
    step_until(&mut vm, &mut frames, 200_000, |vm| {
        vm.jit.trace.installed_index_at_pc(5).is_some()
    });
    // From here the broker has no room, so the second loop's request is
    // refused. Admission must not have evicted the first trace for it.
    let mut broker = crate::jit::JitBroker::synchronous_with_delay(1, u32::MAX);
    let first = vm.jit.trace.installed_index_at_pc(5).unwrap();
    let filler = Arc::new(vm.jit.trace.installed_ir(first).unwrap().clone());
    broker
        .submit_trace(filler, crate::jit::OptimizationTier::Optimized, 0)
        .unwrap();
    vm.jit.broker = Some(broker);
    finish(&mut vm, &mut frames);
    let stats = stats_of(&vm);
    assert!(stats.compile_refusals >= 1, "{stats:?}");
    assert_eq!(stats.cache_evictions, 0, "{stats:?}");
    assert_eq!(vm.jit.trace.cache().live(), 1, "{stats:?}");
    assert!(vm.jit.trace.installed_index_at_pc(5).is_some());
    assert_eq!(vm.jit.trace.cache().reserved_meta(), 0);
    // Only the queue was full, so the header waits one hot period.
    assert!(
        matches!(
            vm.jit.trace.site_state_any(12),
            Some(SiteState::Cooldown { remaining }) if remaining <= crate::vm::jit::trace::native::QUEUE_FULL_COOLDOWN
        ),
        "{:?}",
        vm.jit.trace.site_state_any(12)
    );
}

#[test]
fn the_negative_cache_forgets_its_oldest_rejection() {
    let mut policy = crate::vm::jit::trace::TracePolicy::new(ExecutionMode::Tracing);
    policy.sites_mut().set_negative_cap(3);
    let chunk = Arc::new(sum_program(10.0));
    let mut dropped = 0;
    for pc in 10..16 {
        // Make the site, then reject it.
        policy.sites_mut().site_mut(&chunk, pc).unwrap();
        dropped += policy
            .sites_mut()
            .reject(&chunk, pc, RejectCause::Blacklisted);
    }
    assert_eq!(dropped, 3);
    assert_eq!(policy.sites_mut().rejected_count(), 3);
    assert_eq!(policy.sites_mut().len(), 3);
    // The newest three survive.
    for pc in 13..16 {
        assert_eq!(policy.site_state(&chunk, pc), Some(SiteState::Rejected));
    }
    for pc in 10..13 {
        assert_eq!(policy.site_state(&chunk, pc), None);
    }
    // Re-rejecting refreshes a site's age.
    policy
        .sites_mut()
        .reject(&chunk, 13, RejectCause::Blacklisted);
    policy.sites_mut().site_mut(&chunk, 20).unwrap();
    policy
        .sites_mut()
        .reject(&chunk, 20, RejectCause::Blacklisted);
    assert_eq!(policy.site_state(&chunk, 13), Some(SiteState::Rejected));
    assert_eq!(policy.site_state(&chunk, 14), None);
    const { assert!(MAX_NEGATIVE_SITES >= 64) };
}

/// A header given up by a path that aborts no recording (its traces
/// retired too often) reports that path, and an abort met in a callee keeps
/// the callee's chunk for the report.
#[test]
fn a_rejection_without_an_abort_reports_its_cause() {
    let mut policy = crate::vm::jit::trace::TracePolicy::new(ExecutionMode::Tracing);
    let chunk = Arc::new(sum_program(10.0));
    let callee = Arc::new(sum_program(20.0));
    policy.sites_mut().site_mut(&chunk, 12).unwrap();
    let coverage = policy.coverage(&chunk, 12);
    assert!(coverage.abort.is_none() && coverage.rejected_by.is_none());
    policy.sites_mut().reject(&chunk, 12, RejectCause::Retired);
    let coverage = policy.coverage(&chunk, 12);
    assert_eq!(coverage.rejected_by, Some(RejectCause::Retired));
    assert!(coverage.abort.is_none());
    // An abort with an instruction in a callee's frame.
    let site = policy.sites_mut().site_mut(&chunk, 13).unwrap();
    site.note_abort_in(TraceAbortReason::UnsupportedOperand, Some(3), Some(&callee));
    let coverage = policy.coverage(&chunk, 13);
    let abort = coverage.abort.expect("abort");
    assert_eq!(abort.reason, TraceAbortReason::UnsupportedOperand);
    let (met, pc) = abort.at.expect("an instruction");
    assert!(Arc::ptr_eq(&met, &callee) && pc == 3);
    // An abort with no instruction (a refused compile) still names its
    // reason.
    let site = policy.sites_mut().site_mut(&chunk, 14).unwrap();
    site.note_abort(TraceAbortReason::CompileRejected, None);
    policy.sites_mut().reject(&chunk, 14, RejectCause::Aborts);
    let coverage = policy.coverage(&chunk, 14);
    assert_eq!(coverage.rejected_by, Some(RejectCause::Aborts));
    let abort = coverage.abort.expect("abort");
    assert_eq!(abort.reason, TraceAbortReason::CompileRejected);
    assert!(abort.at.is_none());
}

/// The memo of a rejected function entry pins its chunk, so the address
/// cannot pass to another chunk while the memo names it, and it goes when
/// the site does.
#[test]
fn a_remembered_rejected_function_entry_pins_its_chunk_until_its_site_goes() {
    use crate::vm::jit::trace::policy::FUNCTION_ENTRY_PC;
    let mut policy = crate::vm::jit::trace::TracePolicy::new(ExecutionMode::Tracing);
    policy.sites_mut().set_negative_cap(1);
    let chunk = Arc::new(sum_program(10.0));
    let address = Arc::as_ptr(&chunk) as usize;
    // A site that is not rejected is not remembered.
    policy
        .sites_mut()
        .site_mut(&chunk, FUNCTION_ENTRY_PC)
        .unwrap();
    policy.remember_rejected_entry(Arc::as_ptr(&chunk));
    assert!(!policy.entry_rejected(address));
    policy
        .sites_mut()
        .reject(&chunk, FUNCTION_ENTRY_PC, RejectCause::Blacklisted);
    policy.remember_rejected_entry(Arc::as_ptr(&chunk));
    assert!(policy.entry_rejected(address));
    // The chunk dies; no new chunk can take its address while the memo
    // holds it.
    drop(chunk);
    let others: Vec<Arc<Chunk>> = (0..64).map(|_| Arc::new(sum_program(10.0))).collect();
    assert!(
        others
            .iter()
            .all(|other| Arc::as_ptr(other) as usize != address)
    );
    assert!(policy.entry_rejected(address));
    // A newer rejection evicts the site past the cap, and the memo with it.
    let other = &others[0];
    policy.sites_mut().site_mut(other, 3).unwrap();
    policy
        .sites_mut()
        .reject(other, 3, RejectCause::Blacklisted);
    assert!(!policy.entry_rejected(address));
}

#[test]
fn a_flush_while_a_trace_runs_defers_freeing_its_code() {
    let chunk = Arc::new(sum_program(10_000.0));
    let mut vm = vm_with(0);
    let mut frames = frames_for(&vm, &chunk, None);
    step_until(&mut vm, &mut frames, 100_000, |vm| {
        vm.jit.trace.installed_index_at_pc(5).is_some()
    });
    let index = vm.jit.trace.installed_index_at_pc(5).unwrap();
    let code = vm.jit.trace.cache().code_bytes();
    assert!(code > 0);

    // Pin the trace as a native run does, then flush.
    let mut stats = TraceStatistics::default();
    vm.jit.trace.cache_mut().begin_run(index);
    vm.jit.trace.flush(&mut stats);
    assert!(vm.jit.trace.cache().clear_deferred());
    // The table is empty at once, so a later install cannot join the batch
    // that is about to be dropped. The code lives on until the run returns.
    assert!(vm.jit.trace.installed_tier(index).is_none());
    assert_eq!(vm.jit.trace.cache().retired_len(), 1);
    assert_eq!(vm.jit.trace.site_state_any(5), None, "no new entries");
    // The outermost run returning frees it.
    let freed = vm.jit.trace.cache_mut().end_run();
    assert_eq!(freed.len(), 1);
    drop(freed);
    assert_eq!(vm.jit.trace.cache().retired_len(), 0);
    assert_eq!(vm.jit.trace.cache().code_bytes(), 0);
    assert_eq!(vm.jit.trace.cache().live(), 0);
}

// ----------------------------------------------------------------- versions

/// `for i = 1, limit do if r6 then acc = acc + i end end`, header pc 5.
/// `r6` is never written, so its type at entry picks the recorded path:
/// nil skips the add, a float takes it.
fn condition_program(limit: f64) -> Chunk {
    Chunk::new(
        7,
        vec![
            Constant::Number(1.0),
            Constant::Number(limit),
            Constant::Number(0.0),
        ],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 0),
            load(4, 2),
            for_prep(0, 4, 7),
            jump_if_false(6, 5, 7),
            binary(OpCode::Add, 4, 4, 3),
            for_loop(0, 7, 5),
            Instruction::return_(4),
        ],
    )
}

/// Runs `condition_program` in the interpreter, setting r6 to `value` the
/// first time the frame equals the state `at`. Returns the results.
fn condition_reference(chunk: &Arc<Chunk>, at: &State, value: &Value) -> Vec<Value> {
    let mut vm = interpreter_vm();
    let mut frames = frames_for(&vm, chunk, None);
    for _ in 0..5_000_000 {
        if capture(&frames) == *at {
            frames.last_mut().unwrap().registers[6] = value.clone();
        }
        if let Some(results) = vm.step(&mut frames).unwrap() {
            return results;
        }
    }
    panic!("reference does not finish");
}

#[test]
fn entry_type_mismatches_record_a_second_version_and_the_cap_holds() {
    let chunk = Arc::new(condition_program(200_000.0));
    let mut vm = vm_with(0);
    let mut frames = frames_for(&vm, &chunk, None);
    step_until(&mut vm, &mut frames, 200_000, |vm| {
        vm.jit.trace.installed_index_at_pc(5).is_some()
    });
    let nil_version = vm.jit.trace.installed_index_at_pc(5).unwrap();
    assert_eq!(vm.jit.trace.versions_at_pc(5), vec![nil_version]);

    // From here r6 holds a float: the nil version no longer matches.
    let patched = capture(&frames);
    frames.last_mut().unwrap().registers[6] = Value::Number(1.0);
    let stats_before = stats_of(&vm);
    step_until(&mut vm, &mut frames, 200_000, |vm| {
        vm.jit.trace.versions_at_pc(5).len() == 2
    });
    let stats = stats_of(&vm);
    assert_eq!(stats.version_installs, 1, "{stats:?}");
    assert!(
        stats.exits_of(TraceExitKind::EntryMismatch)
            > stats_before.exits_of(TraceExitKind::EntryMismatch),
        "{stats:?}"
    );
    let versions = vm.jit.trace.versions_at_pc(5);
    assert_ne!(versions[0], nil_version, "the newest version is first");
    assert_eq!(versions[1], nil_version, "the old version stays installed");

    // The float version runs natively, and the whole program agrees with an
    // interpreter that sets r6 at the same state.
    let results = finish(&mut vm, &mut frames);
    let expected = condition_reference(&chunk, &patched, &Value::Number(1.0));
    assert_eq!(results, expected);
    let stats = stats_of(&vm);
    assert!(stats.iterations > 150_000, "{stats:?}");

    // A frame with the old types still takes the older version: re-enter
    // both versions on fresh frames.
    let mut fresh = frames_for(&vm, &chunk, None);
    {
        let mut frame = fresh.at_mut(0);
        frame.pc = 5;
        frame.last_interpreted_pc = Some(7).into();
        for (register, value) in frame.registers.iter_mut().enumerate() {
            *value = match register {
                1 => Value::Number(10_000.0),
                2..=4 => Value::Number(0.0),
                0 => Value::Number(1.0),
                _ => Value::Nil,
            };
        }
    }
    let entries = stats_of(&vm).native_entries;
    let run = {
        let jit = &mut vm.jit;
        jit.trace.enter(
            &mut jit.statistics.trace,
            versions[0],
            &mut fresh,
            0,
            100,
            &mut vm.runtime,
        )
    };
    assert!(matches!(run, crate::vm::jit::trace::TraceRun::Ran { .. }));
    assert_eq!(
        stats_of(&vm).native_entries,
        entries + 1,
        "the nil frame skipped the float version and ran the nil one"
    );
}

#[test]
fn a_header_keeps_at_most_the_capped_number_of_versions() {
    let chunk = Arc::new(condition_program(400_000.0));
    let mut vm = vm_with(0);
    vm.jit.trace.cache_mut().set_limits(CacheLimits {
        versions: 2,
        ..CacheLimits::default()
    });
    let mut frames = frames_for(&vm, &chunk, None);
    step_until(&mut vm, &mut frames, 200_000, |vm| {
        vm.jit.trace.installed_index_at_pc(5).is_some()
    });
    // Alternate the entry types: nil is version one. A float adds version
    // two. A boolean would need a third, which the cap refuses.
    frames.last_mut().unwrap().registers[6] = Value::Number(1.0);
    step_until(&mut vm, &mut frames, 200_000, |vm| {
        vm.jit.trace.versions_at_pc(5).len() == 2
    });
    frames.last_mut().unwrap().registers[6] = Value::Boolean(true);
    // Enough steps for the mismatch streak and a would-be recording. The
    // header never holds a third version: a boolean version may only appear
    // after a mismatching version retired and freed a place.
    let mut max = 0;
    for _ in 0..30_000 {
        if vm.step(&mut frames).unwrap().is_some() {
            break;
        }
        max = max.max(vm.jit.trace.versions_at_pc(5).len());
        assert!(vm.jit.trace.installed_count() <= 2);
    }
    assert_eq!(max, 2, "{:?}", stats_of(&vm));
    assert!(stats_of(&vm).exits_of(TraceExitKind::EntryMismatch) > 4);
}

/// Runs `chunk` ([`condition_program`]) once on a fresh frame with r6 set
/// to `condition`, checks its sum, and returns the iterations native code ran.
fn run_condition(vm: &mut Vm, chunk: &Arc<Chunk>, limit: f64, condition: &Value) -> u64 {
    let before = stats_of(vm).iterations;
    let mut frames = frames_for(vm, chunk, None);
    frames.last_mut().unwrap().registers[6] = condition.clone();
    let results = finish(vm, &mut frames);
    let expected = if matches!(condition, Value::Nil) {
        0.0
    } else {
        limit * (limit + 1.0) / 2.0
    };
    assert_eq!(results, vec![Value::Number(expected)], "{condition:?}");
    stats_of(vm).iterations - before
}

/// A version retired beside another closes the header to new versions.
/// Once the last version retires too, the next attempt opens it again: its
/// one version asks for a second when the entry types change, as the first
/// attempt's did. Mineclonia's `fill_box_rotated` (mineshaft.lua:299) takes
/// an rng table from corridors and nil from staircases, and with the header
/// left closed every later attempt kept one version, which the other
/// signature's entries retired by idle mismatches until the site was
/// rejected.
#[test]
fn the_last_version_retiring_opens_the_header_to_versions_again() {
    // Each run is one entry of the loop, so the versions close their
    // productivity windows and pay: a retired version that never paid is
    // not compiled again.
    let limit = 300.0;
    let chunk = Arc::new(condition_program(limit));
    let mut vm = vm_with(0);
    // Not a number: a register that holds nil or a number is taken untyped
    // by one version.
    let truthy = Value::Boolean(true);
    for _ in 0..100 {
        run_condition(&mut vm, &chunk, limit, &Value::Nil);
    }
    for _ in 0..100 {
        run_condition(&mut vm, &chunk, limit, &truthy);
    }
    let versions = vm.jit.trace.versions_at_pc(5);
    assert_eq!(versions.len(), 2, "{:?}", stats_of(&vm));
    {
        let jit = &mut vm.jit;
        jit.trace
            .retire_for_test(&mut jit.statistics.trace, versions[0]);
    }
    assert_eq!(vm.jit.trace.versions_at_pc(5), vec![versions[1]]);
    assert_eq!(
        vm.jit.trace.versions_closed_at_pc(5),
        Some(true),
        "a version retired beside another closes the header"
    );
    {
        let jit = &mut vm.jit;
        jit.trace
            .retire_for_test(&mut jit.statistics.trace, versions[1]);
    }
    assert!(
        matches!(
            vm.jit.trace.site_state_any(5),
            Some(SiteState::Cooldown { .. })
        ),
        "{:?}",
        vm.jit.trace.site_state_any(5)
    );
    assert_eq!(vm.jit.trace.versions_closed_at_pc(5), Some(false));

    // The second attempt records under the boolean, then gets a version for
    // nil when the entries change back, and both run natively.
    for _ in 0..100 {
        run_condition(&mut vm, &chunk, limit, &truthy);
    }
    assert_eq!(
        vm.jit.trace.versions_at_pc(5).len(),
        1,
        "{:?}",
        stats_of(&vm)
    );
    let installs = stats_of(&vm).version_installs;
    for _ in 0..100 {
        run_condition(&mut vm, &chunk, limit, &Value::Nil);
    }
    assert_eq!(
        vm.jit.trace.versions_at_pc(5).len(),
        2,
        "{:?}",
        stats_of(&vm)
    );
    assert_eq!(stats_of(&vm).version_installs, installs + 1);
    assert!(run_condition(&mut vm, &chunk, limit, &Value::Nil) > 200);
    assert!(run_condition(&mut vm, &chunk, limit, &truthy) > 200);
}

// --------------------------------------------------------------- jit control

#[test]
fn jit_off_stops_trace_entries_and_on_resumes_them() {
    // Long enough that native code returns for a poll before the loop ends.
    let chunk = Arc::new(sum_program(700_000.0));
    let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
    let mut vm = vm_with(0);
    let mut frames = frames_for(&vm, &chunk, None);
    step_until(&mut vm, &mut frames, 200_000, |vm| {
        stats_of(vm).native_entries >= 1
    });
    // The run ended at a poll. Switch off.
    vm.runtime_mut().set_jit_enabled(false);
    let entries = stats_of(&vm).native_entries;
    for _ in 0..3_000 {
        assert!(vm.step(&mut frames).unwrap().is_none());
    }
    assert_eq!(
        stats_of(&vm).native_entries,
        entries,
        "off means no entries"
    );
    assert_eq!(vm.jit.trace.installed_count(), 1, "the trace stays cached");
    vm.runtime_mut().set_jit_enabled(true);
    assert_eq!(finish(&mut vm, &mut frames), expected);
    assert!(stats_of(&vm).native_entries > entries, "on resumes entries");
}

#[test]
fn jit_off_aborts_a_recording_and_a_flush_clears_every_trace_state() {
    let chunk = Arc::new(sum_program(30_000.0));
    let mut vm = vm_with(0);
    vm.jit.trace.set_native_compilation(false);
    let mut frames = frames_for(&vm, &chunk, None);
    step_until(&mut vm, &mut frames, 200_000, |vm| vm.jit.trace.recording());
    vm.runtime_mut().set_jit_enabled(false);
    for _ in 0..20 {
        if vm.step(&mut frames).unwrap().is_some() {
            break;
        }
    }
    assert!(!vm.jit.trace.recording(), "switching the JIT off aborts it");
    assert_eq!(
        stats_of(&vm).aborts(crate::vm::TraceAbortReason::Flushed),
        1,
        "{:?}",
        stats_of(&vm)
    );
    vm.runtime_mut().set_jit_enabled(true);
    vm.runtime_mut().request_jit_flush();
    vm.service_jit_flush();
    assert_eq!(vm.jit.trace.site_state_any(5), None);
    assert!(vm.jit.trace.queued().is_empty());
}
