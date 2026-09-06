//! Native trace execution in the VM.
//!
//! The programs are the hand-assembled chunks of `trace_record`. Every case
//! compares native execution with an isolated interpreter: a VM that only
//! interprets and single-steps. A restored state must be a state that
//! interpreter passes through, and running on from it must finish as the
//! interpreter does. Counters then prove native code really ran, because
//! equal output alone cannot tell a trace from a slow path.

use super::trace_record::*;
use super::*;
use crate::jit::OptimizationTier;
use crate::jit_ir::trace::{TraceIr, TraceScalar};
use crate::vm::jit::trace::TraceStatistics;
use crate::vm::jit::trace::deopt::scalar_to_value;
use crate::vm::jit::trace::native::TRACE_PRODUCTIVITY_WINDOW;
use crate::vm::jit::trace::policy::{MAX_RECORDING_ATTEMPTS, SiteState};
use crate::vm::jit::trace::{TraceExitKind, TraceGuardKind, TraceRun};
use crate::vm::stack::CallStack;
use std::collections::BTreeSet;

pub(super) fn native_vm(profile: CompatibilityProfile) -> Vm {
    let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
    // Traces compile through the broker. The synchronous adapter finishes one
    // request per poll, so an install happens at the next loop-header consult.
    vm.use_synchronous_jit_compilation();
    vm
}

/// [`native_vm`] with every branch on a boolean kept as a guard. The tests
/// of branch guards and their sides build on small two-arm branches whose
/// arms a value select would otherwise run both of, leaving no guard.
pub(super) fn guarded_vm(profile: CompatibilityProfile) -> Vm {
    let mut vm = native_vm(profile);
    vm.jit.trace.set_value_selects(false);
    vm
}

pub(super) fn stats_of(vm: &Vm) -> TraceStatistics {
    vm.jit_statistics().trace
}

/// A VM, a chunk and the index of a compiled trace for the loop at `header`.
pub(super) struct Installed {
    pub(super) vm: Vm,
    pub(super) chunk: Arc<Chunk>,
    pub(super) index: u32,
    header: usize,
    /// Whether `enter` clears the trace's unproductive-run count after each
    /// run, so tests that force many exits do not retire it.
    forgive: bool,
}

impl Installed {
    fn new(chunk: Chunk, header: usize) -> Self {
        Self::with_profile(CompatibilityProfile::Lua51, chunk, header)
    }

    pub(super) fn with_profile(profile: CompatibilityProfile, chunk: Chunk, header: usize) -> Self {
        Self::with_profile_at(
            profile,
            crate::jit::OptimizationTier::Optimized,
            chunk,
            header,
        )
    }

    /// A trace compiled at `tier`. Tests that compare every guard with the
    /// reference evaluator of the recorded IR need `Baseline`, which keeps
    /// every guard where the recording put it.
    pub(super) fn with_profile_at(
        profile: CompatibilityProfile,
        tier: crate::jit::OptimizationTier,
        chunk: Chunk,
        header: usize,
    ) -> Self {
        let chunk = Arc::new(chunk);
        // The guards of two-arm branches are among the exits under test.
        let mut vm = guarded_vm(profile);
        vm.jit.trace.set_compile_tier(tier);
        let mut frames = frames_for(&vm, &chunk, None);
        for _ in 0..500_000 {
            if let Some(index) = vm.jit.trace.installed_index_at_pc(header) {
                return Self {
                    vm,
                    chunk,
                    index,
                    header,
                    forgive: true,
                };
            }
            if vm.step(&mut frames).unwrap().is_some() {
                break;
            }
        }
        panic!(
            "no trace was installed at pc {header} under {profile:?}: {:?}",
            stats_of(&vm)
        );
    }

    pub(super) fn ir(&self) -> TraceIr {
        self.vm
            .jit
            .trace
            .installed_ir(self.index)
            .expect("the trace is installed")
            .clone()
    }

    /// Enters the trace on a fresh frame in `state` through the policy.
    pub(super) fn enter(&mut self, state: &State, fuel: i64) -> (TraceRun, State) {
        let mut frames = frames_for(&self.vm, &self.chunk, Some(state));
        let run = {
            let jit = &mut self.vm.jit;
            jit.trace.enter(
                &mut jit.statistics.trace,
                self.index,
                &mut frames,
                0,
                fuel,
                &mut self.vm.runtime,
            )
        };
        if self.forgive {
            self.vm.jit.trace.forgive(self.index);
        }
        (run, capture(&frames))
    }
}

/// Steps an interpreter from `start` until it is in state `target`, giving
/// up after `limit` steps. A crafted state can send the interpreter into a
/// runtime error, for instance arithmetic on a register the trace never
/// reads. That also ends the search.
fn reaches(chunk: &Arc<Chunk>, start: &State, target: &State, limit: usize) -> bool {
    reaches_with(CompatibilityProfile::Lua51, chunk, start, target, limit)
}

/// [`reaches`] under another profile.
pub(super) fn reaches_with(
    profile: CompatibilityProfile,
    chunk: &Arc<Chunk>,
    start: &State,
    target: &State,
    limit: usize,
) -> bool {
    let mut vm = interpreter_vm_with(profile);
    let mut frames = frames_for(&vm, chunk, Some(start));
    let live_in = crate::jit_ir::observable_registers(chunk);
    for _ in 0..limit {
        if capture(&frames).same_as_with(target, &live_in) {
            return true;
        }
        match vm.step(&mut frames) {
            Ok(None) => {}
            Ok(Some(_)) | Err(_) => return false,
        }
    }
    false
}

/// Programs that record a native trace, with their loop header.
fn native_programs() -> Vec<(&'static str, Chunk, usize)> {
    vec![
        ("sum", sum_program(300.0), 5),
        ("alternating", alternating_program(300.0), 8),
        ("descending", descending_program(300.0), 5),
        ("while", while_program(), 5),
        ("mixed ops", mixed_ops_program(300.0), 8),
        ("float semantics", float_semantics_program(300.0), 9),
        ("early return", early_return_program(300.0, 100.0), 8),
        ("backward alternative", branch_to_header_program(300.0), 5),
    ]
}

/// `i` counts to `limit`. For `i < 100` the body adds it. After that the
/// body jumps straight back to the header. Recording starts around `i = 64`,
/// so the recorded path falls through the second test, whose alternative
/// successor is the loop header itself.
///
/// Header pc 5. The branch at pc 9 is the one under test.
fn branch_to_header_program(limit: f64) -> Chunk {
    Chunk::new(
        7,
        vec![
            Constant::Number(0.0),
            Constant::Number(limit),
            Constant::Number(1.0),
            Constant::Number(100.0),
        ],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 2),
            load(3, 3),
            load(4, 0),
            binary(OpCode::Add, 0, 0, 2),
            binary(OpCode::LessThan, 6, 0, 1),
            jump_if_false(6, 7, 13),
            binary(OpCode::LessThan, 6, 0, 3),
            jump_if_false(6, 9, 5),
            binary(OpCode::Add, 4, 4, 0),
            jump_to(11, 5),
            Instruction::load_nil(5),
            Instruction::return_(4),
        ],
    )
}

// ------------------------------------------------------- end to end

#[test]
fn native_traces_compute_what_the_interpreter_computes() {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        for (name, chunk) in numeric_programs() {
            let expected = interpreted(profile, &chunk);
            let mut vm = native_vm(profile);
            let actual = vm.execute(&chunk).unwrap();
            assert_eq!(actual, expected, "{name} {profile:?}");
        }
    }
}

#[test]
fn counters_prove_native_iterations_happened() {
    for (name, chunk, _) in native_programs() {
        let mut vm = native_vm(CompatibilityProfile::Lua51);
        vm.execute(&chunk).unwrap();
        let stats = stats_of(&vm);
        assert!(stats.traces_compiled >= 1, "{name}: {stats:?}");
        assert!(stats.native_entries >= 1, "{name}: {stats:?}");
        assert!(stats.iterations >= 1, "{name}: {stats:?}");
        // Every entry that ran code left through exactly one exit.
        let exits: u64 = stats.exits.iter().sum();
        assert_eq!(exits, stats.native_entries, "{name}: {stats:?}");
    }
    // Loops the recorder cannot follow stay entirely in the interpreter.
    for (name, chunk) in [
        ("unsupported", unsupported_program(300.0)),
        ("rotating", rotating_program(300.0)),
    ] {
        let mut vm = native_vm(CompatibilityProfile::Lua51);
        vm.execute(&chunk).unwrap();
        let stats = stats_of(&vm);
        assert_eq!(stats.traces_compiled, 0, "{name}: {stats:?}");
        assert_eq!(stats.native_entries, 0, "{name}: {stats:?}");
        assert_eq!(stats.iterations, 0, "{name}: {stats:?}");
    }
}

#[test]
fn a_hot_sum_runs_almost_entirely_in_native_code() {
    let mut vm = native_vm(CompatibilityProfile::Lua51);
    let results = vm.execute(&sum_program(100_000.0)).unwrap();
    assert_eq!(results, vec![Value::Number(5_000_050_000.0)]);
    let stats = stats_of(&vm);
    assert_eq!(stats.native_entries, 1, "{stats:?}");
    assert_eq!(stats.exits_of(TraceExitKind::LoopComplete), 1, "{stats:?}");
    // Recording covers the first 64 iterations and a couple more.
    assert!(stats.iterations > 99_900, "{stats:?}");
}

#[test]
fn optimizer_counters_are_summed_into_the_trace_statistics() {
    let mut vm = native_vm(CompatibilityProfile::Lua51);
    vm.execute(&sum_program(1_000.0)).unwrap();
    let stats = stats_of(&vm);
    assert_eq!(stats.traces_compiled, 1, "{stats:?}");
    assert!(stats.optimizer.total() > 0, "{stats:?}");
    assert_eq!(stats.optimizer.loads_forwarded, 0, "{stats:?}");
    assert_eq!(stats.optimizer.loads_merged, 0, "{stats:?}");
}

#[test]
fn the_interpreter_never_runs_traces_and_every_profile_does() {
    let chunk = sum_program(1_000.0);
    for profile in CompatibilityProfile::ALL {
        let mut vm = native_vm(profile);
        let results = vm.execute(&chunk).unwrap();
        assert_eq!(results, interpreted(profile, &chunk), "{profile:?}");
        let stats = stats_of(&vm);
        assert_eq!(stats.traces_compiled, 1, "{profile:?}: {stats:?}");
        assert!(stats.iterations > 900, "{profile:?}: {stats:?}");
    }
    let mut interpreter = Vm::with_profile_and_execution_mode(
        CompatibilityProfile::Lua51,
        ExecutionMode::Interpreter,
    );
    interpreter.execute(&chunk).unwrap();
    assert_eq!(stats_of(&interpreter), TraceStatistics::default());
}

#[test]
fn a_long_loop_polls_and_resumes_exactly() {
    // The steady budget is 262144 loop-backs, so this loop polls twice.
    let mut vm = native_vm(CompatibilityProfile::Lua51);
    let results = vm.execute(&sum_program(700_000.0)).unwrap();
    assert_eq!(results, vec![Value::Number(245_000_350_000.0)]);
    let stats = stats_of(&vm);
    assert!(stats.exits_of(TraceExitKind::Poll) >= 2, "{stats:?}");
    assert_eq!(stats.exits_of(TraceExitKind::LoopComplete), 1, "{stats:?}");
    assert!(stats.iterations > 699_000, "{stats:?}");
    // A poll re-enters: entries = polls + the final run.
    assert_eq!(
        stats.native_entries,
        stats.exits_of(TraceExitKind::Poll) + 1,
        "{stats:?}"
    );
}

#[test]
fn a_trace_installs_once_at_the_optimized_tier() {
    let mut installed = Installed::new(sum_program(20_000.0), 5);
    assert_eq!(
        installed.vm.jit.trace.installed_tier(installed.index),
        Some(OptimizationTier::Optimized)
    );
    let run = drive(&installed.chunk, None);
    let visits = header_visits(&run, 5);
    let start = &run.states[visits[70]];
    let (ran, restored) = installed.enter(start, 500);
    assert!(matches!(ran, TraceRun::Ran { poll: true, .. }));
    assert_eq!(restored, run.states[visits[70 + 500]]);
    // A long run asks for no recompile.
    assert!(!installed.vm.jit.trace.has_compile_work());
    let (_, again) = installed.enter(&restored, 700);
    assert_eq!(again, run.states[visits[70 + 500 + 700]]);
}

// ---------------------------------------------- forced guard failures

/// States the trace may be entered with, derived from a real header state by
/// changing one register at a time.
pub(super) fn variants(base: &State) -> Vec<State> {
    let mut out = vec![base.clone()];
    for register in 0..base.registers.len() {
        let mut push = |scalar: TraceScalar| {
            let mut state = base.clone();
            state.registers[register] = scalar;
            out.push(state);
        };
        match base.registers[register] {
            TraceScalar::Float(x) => {
                for y in [
                    -x,
                    x + 0.5,
                    x * 2.0 + 1.0,
                    0.0,
                    -0.0,
                    1.0,
                    f64::NAN,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                ] {
                    push(TraceScalar::Float(y));
                }
                push(TraceScalar::Nil);
                push(TraceScalar::Boolean(true));
            }
            TraceScalar::Boolean(flag) => {
                push(TraceScalar::Boolean(!flag));
                push(TraceScalar::Float(1.0));
            }
            TraceScalar::Nil => {
                push(TraceScalar::Float(2.0));
                push(TraceScalar::Boolean(false));
            }
            TraceScalar::Integer(x) => {
                for y in [
                    x.wrapping_add(1),
                    x.wrapping_sub(1),
                    x.wrapping_neg(),
                    0,
                    1,
                    -1,
                    i64::MIN,
                    i64::MAX,
                    i64::MAX - 1,
                    9_007_199_254_740_993,
                ] {
                    push(TraceScalar::Integer(y));
                }
                // The same number as a float is another entry type.
                push(TraceScalar::Float(x as f64));
                push(TraceScalar::Nil);
            }
            // No program here holds an opaque register or a 64-bit integer at
            // its header.
            TraceScalar::Int64(_)
            | TraceScalar::UInt64(_)
            | TraceScalar::Table(_)
            | TraceScalar::Closure(_)
            | TraceScalar::Thread(_)
            | TraceScalar::Userdata(_)
            | TraceScalar::StringRef(_)
            | TraceScalar::String(_) => {}
        }
    }
    out
}

#[test]
fn every_guard_fails_into_a_state_the_isolated_interpreter_passes_through() {
    for (name, chunk, header) in native_programs() {
        let mut installed = Installed::with_profile_at(
            CompatibilityProfile::Lua51,
            crate::jit::OptimizationTier::Baseline,
            chunk,
            header,
        );
        let ir = installed.ir();
        let run = drive(&installed.chunk, None);
        let visits = header_visits(&run, header);
        assert!(visits.len() > 80, "{name}: the loop runs long enough");
        let mut failed_guards = BTreeSet::new();
        let mut mismatches = 0;
        let mut polls = 0;
        for &visit in visits.iter().skip(66).step_by(9).take(14) {
            for state in variants(&run.states[visit]) {
                let (ran, restored) = installed.enter(&state, 600);
                match evaluate(&ir, &state, 600) {
                    Evaluation::EntryMismatch => {
                        mismatches += 1;
                        assert_eq!(ran, TraceRun::NotEntered, "{name}");
                        assert_eq!(restored, state, "{name}: a mismatch changes nothing");
                    }
                    Evaluation::Guard {
                        guard,
                        iteration,
                        restored: expected,
                    } => {
                        failed_guards.insert(guard);
                        let TraceRun::Ran { poll, .. } = ran else {
                            panic!("{name}: guard {guard} did not run natively");
                        };
                        assert!(!poll, "{name}");
                        assert_eq!(
                            restored, expected,
                            "{name}: guard {guard} in iteration {iteration} from {state:?}"
                        );
                        assert!(
                            reaches(&installed.chunk, &state, &restored, 20_000),
                            "{name}: guard {guard} restored {restored:?}, which the \
                             interpreter never reaches from {state:?}"
                        );
                    }
                    Evaluation::Iterations { registers } => {
                        polls += 1;
                        let TraceRun::Ran { poll, .. } = ran else {
                            panic!("{name}: a non-ending loop ran natively");
                        };
                        assert!(poll, "{name}");
                        for (register, scalar) in registers {
                            assert_eq!(restored.registers[register], scalar, "{name} r{register}");
                        }
                        assert!(
                            reaches(&installed.chunk, &state, &restored, 20_000),
                            "{name}: poll state {restored:?}"
                        );
                    }
                }
            }
        }
        // The branch at pc 14 of "mixed ops" tests `not nil`, a value the body
        // builds from constants, so no entry state can make it fail.
        let tautology: &[usize] = if name == "mixed ops" { &[0] } else { &[] };
        let guards: BTreeSet<usize> = (0..ir.guards.len())
            .filter(|guard| !tautology.contains(guard))
            .collect();
        assert_eq!(
            failed_guards, guards,
            "{name}: every guard must fail at least once (polls {polls})"
        );
        assert!(mismatches > 0, "{name}: entry mismatches are exercised");
        let stats = stats_of(&installed.vm);
        assert!(stats.exits_of(TraceExitKind::EntryMismatch) > 0, "{name}");
        assert!(
            stats.exits_of(TraceExitKind::GuardFail) + stats.exits_of(TraceExitKind::LoopComplete)
                > 0,
            "{name}: {stats:?}"
        );
        // Side exits are the guard failures that do not end the loop.
        assert_eq!(
            stats.side_exits_of(TraceGuardKind::Branch),
            stats.exits_of(TraceExitKind::GuardFail),
            "{name}: {stats:?}"
        );
    }
}

#[test]
fn running_on_from_an_exit_finishes_as_the_interpreter_does() {
    for (name, chunk, header) in native_programs() {
        let mut installed = Installed::new(chunk, header);
        let run = drive(&installed.chunk, None);
        let visits = header_visits(&run, header);
        for &visit in visits.iter().skip(66).step_by(11).take(8) {
            for fuel in [1_i64, 3, 1_000] {
                let (ran, restored) = installed.enter(&run.states[visit], fuel);
                assert!(matches!(ran, TraceRun::Ran { .. }), "{name}");
                let index = run.states[visit..]
                    .iter()
                    .position(|state| state.same_as(&restored, &installed.chunk))
                    .map_or_else(
                        || panic!("{name}: {restored:?} is not an interpreter state"),
                        |found| found + visit,
                    );
                // The tail from the restored state takes exactly the steps
                // the uninterrupted run has left, and ends the same way.
                let tail = drive(&installed.chunk, Some(&restored));
                assert_eq!(tail.results, run.results, "{name}");
                assert_eq!(tail.states.len(), run.states.len() - index, "{name}");
            }
        }
    }
}

#[test]
fn a_poll_publishes_the_state_of_the_interpreter_between_iterations() {
    for (name, chunk, header) in native_programs() {
        let mut installed = Installed::new(chunk, header);
        let run = drive(&installed.chunk, None);
        let visits = header_visits(&run, header);
        for (position, fuel) in [(70_usize, 1_i64), (70, 2), (72, 3), (75, 5)] {
            let start = &run.states[visits[position]];
            let (ran, restored) = installed.enter(start, fuel);
            match ran {
                // The trace left earlier through a guard: the state must be
                // one the interpreter reaches.
                TraceRun::Failed | TraceRun::Returned { .. } => {
                    unreachable!("these loop traces build no virtual table")
                }
                TraceRun::Ran { poll: false, .. } => {
                    assert!(
                        run.states[visits[position]..]
                            .iter()
                            .any(|state| state.same_as(&restored, &installed.chunk)),
                        "{name}: {restored:?}"
                    );
                }
                TraceRun::Ran { poll: true, .. } => {
                    // Each iteration of the loop is one header visit, unless
                    // the loop has several backward edges per iteration.
                    let expected = visits[position..]
                        .iter()
                        .map(|&visit| &run.states[visit])
                        .nth(usize::try_from(fuel).unwrap());
                    assert!(
                        expected.is_some_and(|state| state.same_as(&restored, &installed.chunk)),
                        "{name} fuel {fuel}: {restored:?} vs {expected:?}"
                    );
                }
                TraceRun::NotEntered => panic!("{name}: the header state must enter"),
            }
        }
    }
}

// --------------------------------------------- dispatch after an exit

#[test]
fn a_dispatch_that_exits_a_trace_runs_the_resume_instruction_once() {
    let mut installed = Installed::new(branch_to_header_program(300.0), 5);
    let run = drive(&installed.chunk, None);
    // The state before the iteration that makes `i` 100: i is 99.
    let start_index = run
        .states
        .iter()
        .position(|state| {
            state.pc == 5
                && state.last.is_some_and(|last| last >= 5)
                && state.registers[0] == TraceScalar::Float(99.0)
        })
        .expect("the interpreter reaches i = 99 at the header");
    let start = &run.states[start_index];
    let mut frames = frames_for(&installed.vm, &installed.chunk, Some(start));
    let entries_before = stats_of(&installed.vm).native_entries;
    // One dispatch enters the trace at the header, the branch guard fails at
    // i = 100, and the same dispatch interprets the resume instruction (the
    // header) once instead of re-entering at the state the trace just left.
    assert!(installed.vm.step(&mut frames).unwrap().is_none());
    let after = stats_of(&installed.vm);
    assert_eq!(after.native_entries, entries_before + 1);
    assert_eq!(after.exits_of(TraceExitKind::GuardFail), 1);
    let restored = capture(&frames);
    let restored_index = run
        .states
        .iter()
        .position(|state| state.same_as(&restored, &installed.chunk))
        .expect("the dispatch ends on a state the interpreter passes through");
    assert!(restored_index > start_index);
    // The header instruction ran once past the exit state (i = 100).
    assert_eq!(restored.registers[0], TraceScalar::Float(101.0));
}

#[test]
fn a_guard_exit_never_repeats_or_skips_a_guest_instruction() {
    // Step a tracing tracing VM through whole programs. Every exit, entry
    // mismatch and poll in the run must leave the program on the
    // interpreter's path, and the final results must match.
    for (name, chunk, _) in native_programs() {
        let chunk = Arc::new(chunk);
        let reference = drive(&chunk, None);
        let mut vm = native_vm(CompatibilityProfile::Lua51);
        let mut frames = frames_for(&vm, &chunk, None);
        let mut seen = 0_usize;
        let results = loop {
            if let Some(results) = vm.step(&mut frames).unwrap() {
                break scalars(&results);
            }
            // Every state the stepped VM is in must lie ahead of the last one
            // on the interpreter's path.
            let state = capture(&frames);
            let found = reference.states[seen..]
                .iter()
                .position(|candidate| candidate.same_as(&state, &chunk))
                .unwrap_or_else(|| panic!("{name}: {state:?} is off the interpreter's path"));
            seen += found;
        };
        assert_eq!(results, reference.results, "{name}");
        assert!(stats_of(&vm).native_entries > 0, "{name}");
    }
}

// ------------------------------------------------- entry refusals

#[test]
fn a_frame_with_captured_slots_enters_and_its_cells_follow_the_registers() {
    let mut installed = Installed::new(sum_program(1_000.0), 5);
    let run = drive(&installed.chunk, None);
    let visits = header_visits(&run, 5);
    let state = &run.states[visits[80]];
    let mut frames = frames_for(&installed.vm, &installed.chunk, Some(state));
    // Register 4, the running sum, is a captured local: its open cell is the
    // register, and the trace enters the frame all the same.
    let sum = frames.last().unwrap().registers[4].clone();
    let cell = installed.vm.capture_slot(&mut frames, 0, 4).unwrap();
    let entries = stats_of(&installed.vm).native_entries;
    let ran = installed
        .vm
        .enter_native_trace(&mut frames, 0, installed.index)
        .unwrap();
    assert!(ran.is_some(), "the trace entered the captured frame");
    assert_eq!(stats_of(&installed.vm).native_entries, entries + 1);
    let register = frames.last().unwrap().registers[4].clone();
    assert_ne!(register, sum, "the sum advanced");
    assert_eq!(
        installed.vm.runtime.cell_get(cell).unwrap(),
        register,
        "the cell reads the register the exit restored"
    );
    installed.vm.close_upvalues(&mut frames, 0);
    // The same run on an uncaptured frame ends in the same state.
    let (_, expected) = installed.enter(state, 1_000_000);
    assert_eq!(capture(&frames).registers, expected.registers);
}

#[test]
fn entry_type_mismatches_leave_the_frame_alone_and_eventually_retire_the_trace() {
    let mut installed = Installed::new(sum_program(1_000.0), 5);
    installed.forgive = false;
    let run = drive(&installed.chunk, None);
    let visits = header_visits(&run, 5);
    let mut state = run.states[visits[80]].clone();
    // The step register holds nil: the trace needs a float there. The
    // code's own check turns the first entry away, a whole entry's worth of
    // mismatch. From then on the frame's register tags refuse the entry
    // before anything is marshalled, so each mismatch weighs a fraction.
    state.registers[2] = TraceScalar::Nil;
    let weight = crate::vm::jit::trace::native::REFUSED_MISMATCH_WEIGHT;
    let limit = crate::vm::jit::trace::native::TRACE_IDLE_MISMATCH_LIMIT * weight - weight + 1;
    for attempt in 0..limit - 1 {
        let (ran, restored) = installed.enter(&state, 100);
        assert_eq!(ran, TraceRun::NotEntered, "attempt {attempt}");
        assert_eq!(restored, state);
    }
    assert!(installed.vm.jit.trace.installed_index_at_pc(5).is_some());
    let (ran, _) = installed.enter(&state, 100);
    assert_eq!(ran, TraceRun::NotEntered);
    // A run of mismatching entries with no matching one between retires the
    // trace and backs the site off.
    assert_eq!(installed.vm.jit.trace.installed_index_at_pc(5), None);
    assert_eq!(installed.vm.jit.trace.installed_count(), 0);
    let trace_stats = stats_of(&installed.vm);
    assert_eq!(
        trace_stats.exits_of(TraceExitKind::EntryMismatch),
        u64::from(limit),
        "{trace_stats:?}"
    );
    assert_eq!(trace_stats.iterations, 0, "no trace operation ran");
    assert!(matches!(
        installed.vm.jit.trace.site_state_any(5),
        Some(SiteState::Cooldown { .. })
    ));
}

/// A burst of entries that the frame's register tags refuse, several times
/// longer than `TRACE_IDLE_MISMATCH_LIMIT` (the region nests of Mineclonia's
/// `shape.lua`, whose outer locals are nil on some calls), leaves the trace
/// installed: the next entry of the types it was recorded for runs natively.
#[test]
fn a_burst_of_refused_entries_keeps_a_trace_that_pays() {
    let mut installed = Installed::new(sum_program(1_000.0), 5);
    installed.forgive = false;
    let run = drive(&installed.chunk, None);
    let visits = header_visits(&run, 5);
    let good = run.states[visits[80]].clone();
    let mut refused = good.clone();
    refused.registers[2] = TraceScalar::Nil;
    let (ran, _) = installed.enter(&good, 100);
    assert_ne!(ran, TraceRun::NotEntered);
    let burst = 4 * crate::vm::jit::trace::native::TRACE_IDLE_MISMATCH_LIMIT;
    for attempt in 0..burst {
        let (ran, restored) = installed.enter(&refused, 100);
        assert_eq!(ran, TraceRun::NotEntered, "attempt {attempt}");
        assert_eq!(restored, refused, "a refused entry leaves the frame alone");
    }
    assert_eq!(
        installed.vm.jit.trace.installed_index_at_pc(5),
        Some(installed.index)
    );
    let iterations = stats_of(&installed.vm).iterations;
    let (ran, _) = installed.enter(&good, 100);
    assert_ne!(ran, TraceRun::NotEntered);
    assert!(stats_of(&installed.vm).iterations > iterations);
}

#[test]
fn the_fast_restore_plan_matches_the_generic_restore_of_every_exit() {
    for (name, chunk, header) in native_programs() {
        let mut installed = Installed::new(chunk, header);
        let run = drive(&installed.chunk, None);
        let visits = header_visits(&run, header);
        let base = &run.states[visits[70]];
        let mut seed = 0x1234_5678_9abc_def1_u64;
        for exit in 0..installed.vm.jit.trace.exit_count(installed.index) {
            for round in 0..4 {
                // Arbitrary bits in every slot: float payloads, integers and
                // booleans decode the same way through both paths.
                let area: Vec<i64> = (0..64)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        if round == 0 { 0 } else { seed.cast_signed() }
                    })
                    .collect();
                let mut planned = frames_for(&installed.vm, &installed.chunk, Some(base));
                let mut generic = frames_for(&installed.vm, &installed.chunk, Some(base));
                installed.vm.jit.trace.plan_restore(
                    installed.index,
                    exit,
                    &area,
                    &mut planned.at_mut(0),
                    &mut installed.vm.runtime,
                );
                let generic_writes = installed
                    .vm
                    .jit
                    .trace
                    .generic_restore(
                        installed.index,
                        exit,
                        &area,
                        &mut generic.at_mut(0),
                        &mut installed.vm.runtime,
                    )
                    .expect("the generic restore succeeds");
                assert_eq!(
                    format!(
                        "{:?}",
                        installed
                            .vm
                            .jit
                            .trace
                            .plan_upvalues(installed.index, exit, &area)
                    ),
                    format!("{generic_writes:?}"),
                    "{name} exit {exit} round {round}: cell writes"
                );
                assert_eq!(
                    capture_bits(&planned),
                    capture_bits(&generic),
                    "{name} exit {exit} round {round}"
                );
            }
        }
    }
}

/// `capture`, but keeping every register's exact bits even for NaN.
fn capture_bits(frames: &CallStack) -> (usize, Option<usize>, Vec<String>) {
    let frame = frames.last().unwrap();
    (
        frame.pc,
        frame.last_interpreted_pc.get(),
        frame
            .registers
            .iter()
            .map(|value| match value {
                Value::Number(number) => format!("n{:016x}", number.to_bits()),
                other => format!("{other:?}"),
            })
            .collect(),
    )
}

#[test]
fn a_trace_that_keeps_exiting_at_once_is_retired_and_the_program_stays_correct() {
    // The alternating branch fails its guard every other iteration, so every
    // run does one loop-back at most. Its other arm cannot be traced, so no
    // side trace ever covers it.
    let chunk = alternating_unsupported_program(2_000.0);
    let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
    let mut vm = native_vm(CompatibilityProfile::Lua51);
    assert_eq!(vm.execute(&chunk).unwrap(), expected);
    let stats = stats_of(&vm);
    assert!(
        stats.recording_starts > 1,
        "retired traces record again: {stats:?}"
    );
    assert!(stats.traces_compiled >= 1, "{stats:?}");
    // The failing arm was tried as a side trace and refused, once: the
    // recorder cannot follow it, so no second attempt would do better.
    assert_eq!(stats.side_recordings, 1, "{stats:?}");
    assert_eq!(stats.side_aborts, 1, "{stats:?}");
    assert_eq!(stats.side_installs, 0, "{stats:?}");
    assert!(stats.exits_of(TraceExitKind::GuardFail) > 0, "{stats:?}");
    // Entries stop once the retry budget is spent: far fewer than the 2000
    // iterations the loop runs.
    assert!(stats.native_entries < 200, "{stats:?}");
}

#[test]
fn a_heavy_body_is_held_to_the_loop_backs_its_runs_did() {
    // Every run is credited one loop-back, also one that left before the
    // first. A heavy body's bar is one per entry, so that credit alone
    // passed every window: shape-469's driver loop left four frames deep in
    // each of 38000 entries and kept its tree. The heavy bar counts what the
    // runs did, without the credit of a run that left inside an inlined
    // call before its first loop-back.
    let mut installed = Installed::new(sum_program(20_000.0), 5);
    let index = installed.index;
    let policy = &mut installed.vm.jit.trace;
    assert!(policy.heavy_window_retires(index, 0));
    assert!(!policy.heavy_window_retires(index, 1));
    assert!(!policy.heavy_window_retires(index, 2));
}

#[test]
fn a_branch_that_flips_often_hands_the_loop_back_after_one_window() {
    // `if i % 7 < 3 then ... else ... end`: the recorded side holds for three
    // iterations of seven, so a trace averages well under one loop-back per
    // entry. Without the productivity check this loop pays an entry and exit
    // for almost every iteration.
    let chunk = branch_period_unsupported_program(30_000.0);
    let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
    let mut vm = native_vm(CompatibilityProfile::Lua51);
    assert_eq!(vm.execute(&chunk).unwrap(), expected);
    let stats = stats_of(&vm);
    let windows = u64::from(MAX_RECORDING_ATTEMPTS);
    assert!(
        stats.native_entries <= windows * u64::from(TRACE_PRODUCTIVITY_WINDOW),
        "{stats:?}"
    );
    assert!(
        stats.native_entries >= u64::from(TRACE_PRODUCTIVITY_WINDOW),
        "{stats:?}"
    );
    assert_eq!(vm.jit.trace.installed_count(), 0);
    assert_eq!(
        vm.jit.trace.site_state_any(9),
        Some(SiteState::Rejected),
        "{stats:?}"
    );
}

#[test]
fn a_loop_of_a_few_iterations_per_entry_runs_on_through_its_outer_loop() {
    // The inner loop runs three iterations per entry, which an entry and exit
    // cost more than they save. Its exit continues into a side trace that
    // runs the rest of the outer iteration and links back to the inner loop,
    // so the trace is entered a few times and the outer loop runs inside it.
    let chunk = nested_program(600.0);
    let mut vm = native_vm(CompatibilityProfile::Lua51);
    assert_eq!(vm.execute(&chunk).unwrap(), vec![Value::Number(3_600.0)]);
    let stats = stats_of(&vm);
    assert_eq!(stats.outer_sides, 1, "{stats:?}");
    assert_eq!(stats.covered_loops, 1, "{stats:?}");
    assert!(stats.native_entries > 0, "{stats:?}");
    assert!(stats.native_entries < 40, "{stats:?}");
    assert!(stats.side_entries > 400, "{stats:?}");
    assert_eq!(vm.jit.trace.installed_count(), 1);
}

#[test]
fn a_loop_of_a_few_iterations_per_entry_is_not_worth_a_trace_when_its_outer_loop_cannot_record() {
    // The same loops with a read of a missing global after the inner loop,
    // which no side trace can contain. Every outer iteration enters and
    // leaves the trace, which costs more than it saves, so it retires.
    let chunk = nested_unsupported_program(600.0);
    let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
    let mut vm = native_vm(CompatibilityProfile::Lua51);
    assert_eq!(vm.execute(&chunk).unwrap(), expected);
    let stats = stats_of(&vm);
    assert_eq!(stats.outer_sides, 0, "{stats:?}");
    assert!(stats.native_entries > 0, "{stats:?}");
    assert!(stats.native_entries < 600, "{stats:?}");
    assert_eq!(vm.jit.trace.installed_count(), 0);
}

/// `nested_program` with a read of a global that is not there after the inner
/// loop. The inner loop's header is at pc 9.
pub(super) fn nested_unsupported_program(outer: f64) -> Chunk {
    Chunk::new(
        10,
        vec![
            Constant::Number(1.0),
            Constant::Number(outer),
            Constant::Number(3.0),
            Constant::Number(0.0),
            Constant::String("missing".into()),
        ],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 0),
            load(8, 3),
            for_prep(0, 4, 12),
            load(4, 0),
            load(5, 2),
            load(6, 0),
            for_prep(4, 8, 10),
            binary(OpCode::Add, 8, 8, 7),
            for_loop(4, 10, 9),
            Instruction::get_global(9, 4),
            for_loop(0, 12, 5),
            Instruction::return_(8),
        ],
    )
}

/// `for i = 1, limit do if i % 7 < 3 then yes = yes + i else _ = missing; no
/// = no + i end end`, with the loop header at pc 9. The `else` arm reads a
/// global that is not there, which the recorder refuses. A side trace
/// for that arm cannot be recorded, so the loop keeps its unproductive root.
pub(super) fn branch_period_unsupported_program(limit: f64) -> Chunk {
    Chunk::new(
        12,
        vec![
            Constant::Number(1.0),
            Constant::Number(limit),
            Constant::Number(0.0),
            Constant::Number(7.0),
            Constant::Number(3.0),
            Constant::String("missing".into()),
        ],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 0),
            load(4, 2),
            load(5, 2),
            load(6, 3),
            load(7, 4),
            load(11, 2),
            for_prep(0, 8, 16),
            binary(OpCode::Modulo, 8, 3, 6),
            binary(OpCode::LessThan, 9, 8, 7),
            jump_if_false(9, 10, 13),
            binary(OpCode::Add, 4, 4, 3),
            jump_to(12, 15),
            Instruction::get_global(10, 5),
            binary(OpCode::Add, 5, 5, 3),
            for_loop(0, 16, 9),
            Instruction::return_(4),
        ],
    )
}

#[test]
fn flushing_drops_compiled_traces() {
    let mut installed = Installed::new(sum_program(1_000.0), 5);
    assert_eq!(installed.vm.jit.trace.installed_count(), 1);
    let mut stats = TraceStatistics::default();
    installed.vm.jit.trace.flush(&mut stats);
    assert_eq!(installed.vm.jit.trace.installed_count(), 0);
    assert_eq!(
        installed
            .vm
            .jit
            .trace
            .installed_index_at_pc(installed.header),
        None
    );
    // Entering a flushed index does nothing.
    let run = drive(&installed.chunk, None);
    let visits = header_visits(&run, 5);
    let (ran, restored) = installed.enter(&run.states[visits[80]], 10);
    assert_eq!(ran, TraceRun::NotEntered);
    assert_eq!(restored, run.states[visits[80]]);
}

#[test]
fn restored_registers_keep_their_exact_float_bits() {
    // NaN payloads, signed zero and infinities survive the exit area.
    let mut installed = Installed::new(float_semantics_program(300.0), 9);
    let run = drive(&installed.chunk, None);
    let visits = header_visits(&run, 9);
    let ir = installed.ir();
    let mut state = run.states[visits[80]].clone();
    state.registers[3] = TraceScalar::Float(-0.0);
    let (_, restored) = installed.enter(&state, 600);
    let Evaluation::Guard {
        restored: expected, ..
    } = evaluate(&ir, &state, 600)
    else {
        panic!("the loop ends through a guard");
    };
    for (native, reference) in restored.registers.iter().zip(&expected.registers) {
        // `TraceScalar` equality compares float bits.
        assert_eq!(native, reference);
    }
    // And the VM value for a scalar is the number with those bits.
    let nan = f64::from_bits(0x7ff8_dead_beef_0001);
    let Value::Number(round_trip) = scalar_to_value(TraceScalar::Float(nan)) else {
        panic!("a float scalar is a number");
    };
    assert_eq!(round_trip.to_bits(), nan.to_bits());
}

#[test]
fn only_the_frame_the_trace_was_built_for_enters() {
    let installed = Installed::new(sum_program(1_000.0), 5);
    let other = Arc::new(sum_program(1_000.0));
    let run = drive(&installed.chunk, None);
    let visits = header_visits(&run, 5);
    let state = &run.states[visits[80]];
    let mut installed = installed;
    // A different chunk (even with identical code) is not the trace's chunk.
    let mut frames = frames_for(&installed.vm, &other, Some(state));
    assert_eq!(
        installed
            .vm
            .enter_native_trace(&mut frames, 0, installed.index)
            .unwrap(),
        None
    );
    // Neither is the right chunk at another pc.
    let mut moved = state.clone();
    moved.pc = 6;
    let mut frames = frames_for(&installed.vm, &installed.chunk, Some(&moved));
    assert_eq!(
        installed
            .vm
            .enter_native_trace(&mut frames, 0, installed.index)
            .unwrap(),
        None
    );
    let _ = CallStack::from(Vec::new());
}

// ------------------------------------------------- optimized tier

impl Installed {
    /// Enters the trace from `start` until the broker has replaced its
    /// baseline code with optimized code. False when it never promotes:
    /// a trace that leaves within one iteration completes no loop-back.
    pub(super) fn try_promote(&mut self, start: &State, fuel: i64) -> bool {
        for _ in 0..40 {
            if self.vm.jit.trace.installed_tier(self.index) == Some(OptimizationTier::Optimized) {
                return true;
            }
            self.enter(start, fuel);
            self.vm.service_trace_compilations();
        }
        self.vm.jit.trace.installed_tier(self.index) == Some(OptimizationTier::Optimized)
    }

    fn promote(&mut self, start: &State, fuel: i64) {
        assert!(
            self.try_promote(start, fuel),
            "the trace was never promoted: {:?}",
            self.vm.jit_statistics().trace
        );
    }
}

/// A for loop with invariant work in its body. Header pc 7, latch pc 13.
///
/// ```text
/// r4 acc, r5 a, r6 b           (a and b are never written in the loop)
/// r7 = a * b                   invariant
/// r8 = r7 + step               invariant
/// r9 = a < b                   invariant condition
/// if r9 then acc = acc + r8 end
/// acc = acc + index
/// ```
fn invariant_program(limit: f64) -> Chunk {
    Chunk::new(
        10,
        vec![
            Constant::Number(1.0),
            Constant::Number(limit),
            Constant::Number(0.0),
            Constant::Number(2.5),
            Constant::Number(4.0),
        ],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 0),
            load(4, 2),
            load(5, 3),
            load(6, 4),
            for_prep(0, 6, 13),
            binary(OpCode::Multiply, 7, 5, 6),
            binary(OpCode::Add, 8, 7, 2),
            binary(OpCode::LessThan, 9, 5, 6),
            jump_if_false(9, 10, 12),
            binary(OpCode::Add, 4, 4, 8),
            binary(OpCode::Add, 4, 4, 3),
            for_loop(0, 13, 7),
            Instruction::return_(4),
        ],
    )
}

#[test]
fn promoted_traces_hoist_invariants_and_still_run_natively() {
    let mut installed = Installed::new(invariant_program(1_000.0), 7);
    let run = drive(&installed.chunk, None);
    let visits = header_visits(&run, 7);
    let start = run.states[visits[70]].clone();
    installed.promote(&start, 200);
    let lowered = installed
        .vm
        .jit
        .trace
        .installed_lowered_ir(installed.index)
        .unwrap();
    lowered.verify().unwrap();
    assert!(
        !lowered.preheader.is_empty(),
        "invariant work moved out of the loop: {lowered:#?}"
    );
    assert!(
        lowered.body.len() < installed.ir().body.len(),
        "the steady-state loop is smaller than the recorded iteration"
    );
    let before = stats_of(&installed.vm).iterations;
    let (ran, restored) = installed.enter(&start, 300);
    assert!(matches!(ran, TraceRun::Ran { poll: true, .. }));
    assert!(
        restored.same_as(&run.states[visits[70 + 300]], &installed.chunk),
        "{restored:?}"
    );
    assert_eq!(stats_of(&installed.vm).iterations, before + 300);
    // The whole program still finishes the same way from the exit.
    assert_eq!(
        drive(&installed.chunk, Some(&restored)).results,
        run.results
    );
}

#[test]
fn a_hoisted_guard_exit_is_an_interpreter_state_and_the_program_finishes_the_same() {
    let mut installed = Installed::new(invariant_program(1_000.0), 7);
    let run = drive(&installed.chunk, None);
    let visits = header_visits(&run, 7);
    let start = run.states[visits[70]].clone();
    installed.promote(&start, 200);
    let lowered = installed
        .vm
        .jit
        .trace
        .installed_lowered_ir(installed.index)
        .unwrap();
    assert!(
        lowered
            .preheader
            .iter()
            .any(|instruction| instruction.op.guard().is_some()),
        "the branch on a < b moved into the preheader"
    );
    let mut hoisted_exits = 0;
    for (a, b) in [
        (2.5, 4.0),
        (4.0, 2.5),
        (3.0, 3.0),
        (f64::NAN, 1.0),
        (1.0, f64::NAN),
        (f64::NEG_INFINITY, f64::INFINITY),
        (-0.0, 0.0),
        (1e308, 1e308),
    ] {
        let mut state = start.clone();
        state.registers[5] = TraceScalar::Float(a);
        state.registers[6] = TraceScalar::Float(b);
        let expected = drive(&installed.chunk, Some(&state)).results;
        let iterations = stats_of(&installed.vm).iterations;
        let (ran, restored) = installed.enter(&state, 150);
        let TraceRun::Ran { poll, .. } = ran else {
            panic!("a < b with {a:?}, {b:?} did not enter");
        };
        // The guard fails in the preheader exactly when `a < b` is false.
        let passes = a < b;
        assert_eq!(poll, passes, "{a:?} < {b:?}");
        if !passes {
            hoisted_exits += 1;
            assert_eq!(
                stats_of(&installed.vm).iterations,
                iterations,
                "no iteration ran before the hoisted guard failed"
            );
            assert_eq!(restored.registers, state.registers, "{a:?} {b:?}");
            assert_eq!(restored.pc, 7, "the header");
        }
        assert!(
            reaches(&installed.chunk, &state, &restored, 50_000),
            "{a:?} {b:?}: {restored:?} is not an interpreter state"
        );
        assert_eq!(
            drive(&installed.chunk, Some(&restored)).results,
            expected,
            "{a:?} {b:?}"
        );
    }
    assert!(hoisted_exits >= 6);
}

/// The guard the reference evaluator fails in `ir` from `state`, and the
/// interpreter state it restores. `None` when the run ends another way.
pub(super) fn reference_exit(ir: &TraceIr, state: &State) -> Option<(usize, State)> {
    use crate::jit_ir::trace::reference::{Outcome, evaluate_forced};
    let evaluation = evaluate_forced(ir, &state.registers, &[], 600, None, |_, _| None);
    let Outcome::GuardFailed { guard, restored } = evaluation.outcome else {
        return None;
    };
    Some((
        guard.index(),
        State {
            pc: restored.pc,
            last: restored.last_interpreted_pc,
            registers: restored.registers,
        },
    ))
}

#[test]
fn promoted_traces_exit_into_interpreter_states_and_finish_like_the_interpreter() {
    let mut programs = native_programs();
    programs.push(("invariant", invariant_program(300.0), 7));
    let mut promoted_programs = 0;
    for (name, chunk, header) in programs {
        let mut installed = Installed::new(chunk, header);
        let run = drive(&installed.chunk, None);
        let visits = header_visits(&run, header);
        assert!(visits.len() > 80, "{name}");
        let promoted = visits
            .iter()
            .skip(66)
            .take(12)
            .any(|visit| installed.try_promote(&run.states[*visit], 40));
        if !promoted {
            // Traces that leave within one iteration never get hot enough.
            continue;
        }
        promoted_programs += 1;
        let lowered = installed
            .vm
            .jit
            .trace
            .installed_lowered_ir(installed.index)
            .unwrap()
            .clone();
        lowered.verify().unwrap();
        let mut failed_guards = BTreeSet::new();
        let mut exits = 0;
        for &visit in visits.iter().skip(66).step_by(9).take(14) {
            for state in variants(&run.states[visit]) {
                let (ran, restored) = installed.enter(&state, 600);
                match ran {
                    TraceRun::NotEntered => {
                        assert_eq!(restored, state, "{name}: a mismatch changes nothing");
                    }
                    TraceRun::Failed | TraceRun::Returned { .. } => {
                        unreachable!("these loop traces build no virtual table")
                    }
                    TraceRun::Ran { .. } => {
                        exits += 1;
                        assert!(
                            reaches(&installed.chunk, &state, &restored, 20_000),
                            "{name}: {restored:?} is not an interpreter state from {state:?}"
                        );
                        // The reference evaluation of the lowered trace, which
                        // includes the preheader, names the exit and the state.
                        // A run the fuel ends has no single expected state.
                        let reference = reference_exit(&lowered, &state);
                        if let Some((guard, expected)) = reference {
                            failed_guards.insert(guard);
                            assert_eq!(
                                restored, expected,
                                "{name}: lowered guard {guard} from {state:?}"
                            );
                        }
                    }
                }
            }
        }
        assert!(exits > 20, "{name}: {exits} native exits");
        // Every guard of the lowered trace, hoisted ones included, fails at
        // least once. The optimizer already folded away the always-true branch
        // that the source trace of "mixed ops" keeps.
        let guards: BTreeSet<usize> = (0..lowered.guards.len()).collect();
        assert_eq!(
            failed_guards, guards,
            "{name}: every lowered guard must fail at least once"
        );
        // Running on from a poll state ends where the interpreter ends.
        let (ran, restored) = installed.enter(&run.states[visits[70]], 25);
        assert!(matches!(ran, TraceRun::Ran { .. }), "{name}");
        assert_eq!(
            drive(&installed.chunk, Some(&restored)).results,
            run.results,
            "{name}"
        );
    }
    assert!(
        promoted_programs >= 4,
        "{promoted_programs} programs promoted"
    );
}

#[test]
fn optimized_tracing_runs_match_the_interpreter_end_to_end() {
    let mut programs: Vec<(&str, Chunk)> = native_programs()
        .into_iter()
        .map(|(name, chunk, _)| (name, chunk))
        .collect();
    programs.push(("invariant", invariant_program(300_000.0)));
    programs.push(("invariant short", invariant_program(500.0)));
    for (name, chunk) in programs {
        let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
        let mut vm = native_vm(CompatibilityProfile::Lua51);
        assert_eq!(vm.execute(&chunk).unwrap(), expected, "{name}");
    }
}

#[test]
fn a_trace_that_is_mostly_pattern_calls_is_helper_bound() {
    use crate::jit_ir::trace::{
        FrameId, GuardId, HelperId, TraceBuilder, TraceFrameInfo, TraceId, TraceOp, TraceSite,
        TraceTerminator, TraceType,
    };
    use crate::vm::jit::trace::native::is_helper_bound;
    // A loop body of `arithmetic` constants and one call of each helper.
    let body = |helpers: &[HelperId], arithmetic: usize| {
        let mut b = TraceBuilder::new(
            TraceId(0),
            TraceFrameInfo {
                register_count: 4,
                code_len: 8,
                inlined: None,
                activation: None,
            },
        );
        let site = TraceSite {
            frame: FrameId(0),
            pc: 1,
        };
        for _ in 0..arithmetic {
            b.emit(
                site,
                TraceOp::Constant(TraceScalar::Integer(1)),
                Some(TraceType::Integer),
            );
        }
        for helper in helpers {
            b.emit(
                site,
                TraceOp::CallHelper {
                    helper: *helper,
                    args: Vec::new(),
                    effect: None,
                    guard: GuardId(0),
                },
                None,
            );
        }
        b.finish(TraceTerminator::LoopBack { args: Vec::new() })
    };
    let find = HelperId::string_find_pattern(1, 1);
    let gsub = HelperId::STRING_GSUB;
    // A body with no call, and one with a call lost among a hundred
    // operations, is the optimizing tier's.
    assert!(!is_helper_bound(&body(&[], 40)));
    assert!(!is_helper_bound(&body(&[find], 100)));
    // The calls are most of the work of the iteration.
    assert!(is_helper_bound(&body(&[find], 30)));
    assert!(is_helper_bound(&body(&[find, gsub, find], 100)));
    // A call that is not a pattern call is not.
    assert!(!is_helper_bound(&body(
        &[HelperId::TOSTRING, HelperId::CONCAT],
        10
    )));
}
