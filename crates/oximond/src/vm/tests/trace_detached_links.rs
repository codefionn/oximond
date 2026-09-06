//! A tree that links through the return of its function into the loop that
//! calls it, with the link compiled as a function of its own (see "Sides
//! compiled apart" in `vm::jit::trace`): the side that returns into the caller
//! reads the caller's registers from the run's extra inputs, writes the ones
//! it changes into the run's input slots where it links back, and the VM
//! writes them into the interpreter's frames when the run leaves.
//!
//! The programs run from Lua source and are stepped by hand, so that a flush
//! lands between two steps while a side compiles. Every case compares with an
//! isolated interpreter.

use super::trace_native::stats_of;
use super::*;
use crate::vm::jit::trace::TraceExitKind;

/// A float profile, an integer profile and the one with both kinds of
/// numbers; the other profiles are covered by the integration tests.
const PROFILES: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::LuaJit21,
];

/// A function with a nest, and a driver that calls it.
pub(super) fn program(driver: &str) -> String {
    format!(
        "local function sweep(n, m)
             local acc = 0
             for i = 1, n do
                 for j = i, m do
                     acc = acc + (i * j) % 7
                 end
             end
             return acc
         end
         {driver}"
    )
}

const PLAIN_DRIVER: &str = "
    local total = 0
    for k = 1, CALLS do
        total = (total + sweep(8 + k % 3, 40)) % 1000003
    end
    return total";

fn with_calls(driver: &str, calls: u64) -> String {
    driver.replace("CALLS", &calls.to_string())
}

/// What a stepped run did.
pub(super) struct Run {
    pub(super) result: Result<Vec<Value>, String>,
    pub(super) stats: TraceStatistics,
    pub(super) lua: crate::Lua,
}

/// Runs `source` one step at a time, handing the VM to `act` before each
/// step. A compile result installs after `delay` safepoint polls, when given.
pub(super) fn stepped(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
    delay: Option<u32>,
    mut act: impl FnMut(usize, &mut Vm, &crate::vm::stack::CallStack),
) -> Run {
    let mut lua = crate::Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    let chunk = lua.compile("=(links)", source).unwrap();
    let vm = lua.vm_mut();
    match delay {
        Some(delay) => vm.use_delayed_synchronous_jit_compilation(delay),
        None => vm.use_synchronous_jit_compilation(),
    }
    let mut frames = vm.stack_for(&chunk).unwrap();
    let mut steps = 0;
    let result = loop {
        steps += 1;
        act(steps, vm, &frames);
        match vm.step(&mut frames) {
            Ok(Some(results)) => break Ok(results),
            Ok(None) => {}
            Err(error) => break Err(format!("{error:?}")),
        }
    };
    let stats = stats_of(vm);
    Run { result, stats, lua }
}

pub(super) fn interpreted(profile: CompatibilityProfile, source: &str) -> Run {
    stepped(
        profile,
        ExecutionMode::Interpreter,
        source,
        None,
        |_, _, _| {},
    )
}

pub(super) fn traced(profile: CompatibilityProfile, source: &str) -> Run {
    stepped(profile, ExecutionMode::Tracing, source, None, |_, _, _| {})
}

#[test]
fn the_link_into_the_calling_loop_is_a_function_of_its_own() {
    crate::jit::debug_env::sides_apart_for_this_test();
    for profile in PROFILES {
        let source = with_calls(&program(PLAIN_DRIVER), 6_000);
        let expected = interpreted(profile, &source);
        let run = traced(profile, &source);
        assert_eq!(run.result, expected.result, "{profile:?}");
        let stats = &run.stats;
        // The tree was compiled once, and the side that returns into the
        // loop beside it, without compiling the tree again.
        assert_eq!(stats.traces_compiled, 1, "{profile:?}: {stats:?}");
        assert!(stats.side_compiles >= 1, "{profile:?}: {stats:?}");
        assert_eq!(stats.link_recompiles, 0, "{profile:?}: {stats:?}");
        // After it the calls run in the loop around them: native code is
        // entered for the calls before the side landed and for no others.
        assert!(
            stats.native_entries * 2 < 6_000,
            "{profile:?}: {} entries for 6000 calls: {stats:?}",
            stats.native_entries
        );
        let vm = run.lua.vm();
        let index = vm.jit.trace.linked_trace_index().expect("a linked tree");
        let compiled = vm.jit.trace.installed_compiled(index).unwrap();
        assert!(compiled.attached_sides() >= 1, "{profile:?}: {stats:?}");
        // The loop's registers the side changes ride in the run's input
        // slots, and the frame gets them when the run leaves.
        assert!(!compiled.writebacks().is_empty(), "{profile:?}");
        assert!(!compiled.extras().is_empty(), "{profile:?}");
    }
}

#[test]
fn the_tree_stays_as_it_was_compiled_while_the_link_is_added() {
    crate::jit::debug_env::sides_apart_for_this_test();
    // The tree is not compiled again for the link: its code, its entry
    // contract and its pinned closures are the ones of the first compile.
    let profile = CompatibilityProfile::Lua51;
    let source = with_calls(&program(PLAIN_DRIVER), 4_000);
    let mut before = None;
    let run = stepped(
        profile,
        ExecutionMode::Tracing,
        &source,
        None,
        |_, vm, _| {
            if before.is_none() {
                for index in 0..8 {
                    if let Some(compiled) = vm.jit.trace.installed_compiled(index)
                        && !compiled.is_staged()
                    {
                        before = Some((
                            index,
                            compiled.entry_contract().to_vec(),
                            compiled.lowered_ir().pinned_closures.clone(),
                            compiled.clif_insts_total(),
                        ));
                        break;
                    }
                }
            }
        },
    );
    let (index, contract, pins, clif) = before.expect("a tree was installed");
    let vm = run.lua.vm();
    let compiled = vm.jit.trace.installed_compiled(index).unwrap();
    assert!(compiled.attached_sides() >= 1, "{:?}", run.stats);
    assert_eq!(compiled.entry_contract(), &contract[..]);
    assert_eq!(compiled.lowered_ir().pinned_closures, pins);
    // The tree's own instructions did not change; the sides came on top.
    assert!(compiled.clif_insts_total() > clif);
    assert_eq!(run.stats.traces_compiled, 1, "{:?}", run.stats);
}

#[test]
fn a_flush_while_the_side_that_returns_compiles_drops_it() {
    crate::jit::debug_env::sides_apart_for_this_test();
    for profile in PROFILES {
        let source = with_calls(&program(PLAIN_DRIVER), 6_000);
        let expected = interpreted(profile, &source);
        for delay in [3_u32, 20, 81, 90, 130, 400] {
            let mut flushed = false;
            let mut before = None;
            let run = stepped(
                profile,
                ExecutionMode::Tracing,
                &source,
                Some(delay),
                |_, vm, _| {
                    if !flushed && vm.jit.trace.side_compile_pending() {
                        before = Some(stats_of(vm));
                        flushed = true;
                        vm.runtime_mut().request_jit_flush();
                    }
                },
            );
            assert_eq!(run.result, expected.result, "{profile:?} delay {delay}");
            assert!(
                flushed,
                "{profile:?} delay {delay}: no side compile: {:?}",
                run.stats
            );
            let before = before.unwrap();
            // The side that finished found no tree to join; the program
            // recorded its tree and sides again.
            assert!(
                run.stats.stale_completions > before.stale_completions,
                "{profile:?} delay {delay}: {:?}",
                run.stats
            );
        }
    }
}

#[test]
fn a_tree_at_the_peak_tier_takes_the_side_that_returns_with_it() {
    crate::jit::debug_env::sides_apart_for_this_test();
    // The nest runs past the iterations that earn the peak tier with the side
    // that returns into the loop attached: the peak compile holds it inline,
    // so the link survives the recompile and the results stay the same. The
    // policy keeps a tree MIR compiled off the tier, so the test asks for it.
    crate::jit::debug_env::peak_for_this_test();
    let profile = CompatibilityProfile::Lua51;
    let source = with_calls(&program(PLAIN_DRIVER), 12_000);
    let expected = interpreted(profile, &source);
    let run = traced(profile, &source);
    assert_eq!(run.result, expected.result);
    let stats = &run.stats;
    assert!(stats.side_compiles >= 1, "{stats:?}");
    let vm = run.lua.vm();
    let index = vm.jit.trace.linked_trace_index().expect("a linked tree");
    assert_eq!(
        vm.jit.trace.installed_tier(index),
        Some(crate::jit::OptimizationTier::Peak),
        "{stats:?}"
    );
    let compiled = vm.jit.trace.installed_compiled(index).unwrap();
    assert_eq!(compiled.attached_sides(), 0, "{stats:?}");
    assert!(stats.native_entries * 3 < 12_000, "{stats:?}");
}

#[test]
fn an_exit_inside_the_caller_frame_leaves_the_caller_as_the_interpreter_has_it() {
    crate::jit::debug_env::sides_apart_for_this_test();
    // The code after the call stores, branches and keeps state in several
    // registers. A rare branch of it takes the exit inside the caller's
    // frame: the run has changed the loop's registers by then, and the frame
    // must hold them.
    for profile in PROFILES {
        let source = program(
            "local total, marks, flag, label, count = 0, 0, true, 'x', 0
             for k = 1, 9000 do
                 local value = sweep(8 + k % 3, 40)
                 total = (total + value) % 1000003
                 flag = not flag
                 count = count + 1
                 if k % 1500 == 0 then marks = marks + k; label = label .. 'y' end
             end
             return total, marks, flag, label, count",
        );
        let expected = interpreted(profile, &source);
        let run = traced(profile, &source);
        assert_eq!(run.result, expected.result, "{profile:?}");
        let stats = &run.stats;
        assert!(stats.side_compiles >= 1, "{profile:?}: {stats:?}");
        // The rare branch left through the side's own exits.
        assert!(stats.side_member_exits >= 1, "{profile:?}: {stats:?}");
        assert!(
            stats.exits_of(TraceExitKind::GuardFail) >= 1,
            "{profile:?}: {stats:?}"
        );
    }
}

#[test]
fn an_outer_side_compiled_apart_hands_the_outer_loops_registers_to_the_run() {
    // The tree of the inner loop has run long enough to have had its burst of
    // sides when a branch of the code after the inner loop starts to be taken
    // now and then. The side that continues it runs through the outer loop, so
    // it writes the outer loop's counters and `flag`, which the root does not
    // carry: the side stores them in the run's input slots at the link, and
    // the run writes them into the frame when it leaves.
    // The integer profile records no second outer side for this program.
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        let source = "local acc, marks, flag = 0, 0, true
             for a = 1, 6000 do
                 for b = 1, 60 do
                     acc = (acc + a * b) % 1000003
                 end
                 flag = not flag
                 if a > 2000 and a % 50 == 0 then
                     marks = marks + a
                 end
             end
             return acc, marks, flag";
        let expected = interpreted(profile, source);
        let run = traced(profile, source);
        assert_eq!(run.result, expected.result, "{profile:?}");
        let stats = &run.stats;
        assert!(stats.side_compiles >= 1, "{profile:?}: {stats:?}");
        assert!(stats.outer_sides >= 2, "{profile:?}: {stats:?}");
        // The root was compiled once and the sides beside it.
        assert_eq!(stats.link_recompiles, 0, "{profile:?}: {stats:?}");
        let vm = run.lua.vm();
        let compiled = (0..8)
            .filter_map(|index| vm.jit.trace.installed_compiled(index))
            .find(|compiled| compiled.attached_sides() > 0)
            .expect("a tree with a side compiled apart");
        assert!(!compiled.writebacks().is_empty(), "{profile:?}");
    }
}

#[test]
fn a_stencil_that_reaches_the_peak_tier_keeps_its_range_versions() {
    // Two million native iterations earn the peak tier. The stencil runs
    // 2.4 million of them over arrays it swaps each pass, and the peak
    // compile keeps the loop's fast version (the root's range plan) as the
    // optimized compile has it. MIR compiles this tree and its code would
    // stay, so the test asks for the tier.
    crate::jit::debug_env::peak_for_this_test();
    let source = r"
        local n = 20000
        local a, b = {}, {}
        for i = 1, n do a[i] = i % 97; b[i] = 0 end
        local total = 0
        for pass = 1, 120 do
            for i = 2, n - 1 do
                b[i] = (a[i - 1] + a[i] + a[i + 1]) % 1009
            end
            a, b = b, a
            total = total + a[10000]
        end
        return total, a[1], a[n]
    ";
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        let expected = interpreted(profile, source);
        let run = traced(profile, source);
        assert_eq!(run.result, expected.result, "{profile:?}");
        let vm = run.lua.vm();
        let mut peak_versioned = false;
        for index in 0..vm.jit.trace.installed_count() {
            let index = index as u32;
            let compiled = vm.jit.trace.installed_compiled(index).unwrap();
            if vm.jit.trace.installed_tier(index) == Some(crate::jit::OptimizationTier::Peak) {
                peak_versioned |= compiled.versioned_loop();
            }
        }
        assert!(peak_versioned, "{profile:?}: {:?}", run.stats);
    }
}
