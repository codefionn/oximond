//! The trace recorder, driven through the real interpreter loops.
//!
//! The programs are hand-assembled chunks. Each test runs one under the trace
//! engine, then checks the recorded `TraceIr` against the interpreter. A small
//! evaluator runs the trace from a real loop-header state, and compares its
//! loop-carried registers with later header states of an interpreter-only run.
//! A guard that fails must restore a state that the interpreter itself passes
//! through, and running on from that state must finish with the same result in
//! the number of steps that remained.

use super::*;
use crate::jit_ir::trace::{
    LoopParamInit, RegisterEntry, SnapshotPhase, TraceId, TraceIr, TraceOp, TraceScalar,
    TraceTerminator, TraceType,
};
use crate::vm::jit::trace::deopt::{scalar_to_value, value_to_scalar};
use crate::vm::jit::trace::policy::{MAX_SAME_ABORTS, SiteState, TRACE_HOT_THRESHOLD};
use crate::vm::jit::trace::recorder::{HookPoint, HookSource, Recorder, Step};
use crate::vm::jit::trace::{
    LoopHeader, TraceAbortReason, TraceDispatch, TracePolicy, TraceStatistics,
};
use crate::vm::stack::CallStack;

const ALL_PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

/// A tracing VM whose closed traces wait in the queue instead of compiling, so
/// these tests see the recorder's output. `trace_native.rs` runs the same
/// loops natively.
fn traced_vm(profile: CompatibilityProfile) -> Vm {
    let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
    vm.jit.trace.set_native_compilation(false);
    vm
}

fn trace_stats(vm: &Vm) -> TraceStatistics {
    vm.jit_statistics().trace
}

pub(super) fn scalars(values: &[Value]) -> Vec<TraceScalar> {
    values
        .iter()
        .map(|value| value_to_scalar(value).expect("scalar result"))
        .collect()
}

/// Runs `chunk` in a VM that only interprets.
pub(super) fn interpreted(profile: CompatibilityProfile, chunk: &Chunk) -> Vec<Value> {
    Vm::with_profile_and_execution_mode(profile, ExecutionMode::Interpreter)
        .execute(chunk)
        .unwrap()
}

// ---------------------------------------------------------------- assembly

pub(super) fn load(destination: u8, constant: u16) -> Instruction {
    Instruction::load_constant(destination, constant)
}

pub(super) fn binary(opcode: OpCode, destination: u8, left: u8, right: u8) -> Instruction {
    Instruction::binary(opcode, destination, left, right)
}

pub(super) fn jump(offset: i32) -> Instruction {
    Instruction::aj(OpCode::Jump, offset).unwrap()
}

/// An offset from the instruction at `from` to the one at `to`.
pub(super) fn offset(from: usize, to: usize) -> i16 {
    i16::try_from(to as i64 - from as i64 - 1).unwrap()
}

pub(super) fn jump_to(from: usize, to: usize) -> Instruction {
    jump(i32::from(offset(from, to)))
}

pub(super) fn jump_if_false(condition: u8, from: usize, to: usize) -> Instruction {
    Instruction::jump_if_false(condition, offset(from, to))
}

pub(super) fn for_prep(base: u8, from: usize, latch: usize) -> Instruction {
    Instruction::numeric_for_prep(base, offset(from, latch))
}

pub(super) fn for_loop(base: u8, from: usize, header: usize) -> Instruction {
    Instruction::numeric_for_loop(base, offset(from, header))
}

// --------------------------------------------------------------- programs

/// Sums 1..=limit with the visible index. Header pc 5, latch pc 6.
pub(super) fn sum_program(limit: f64) -> Chunk {
    numeric_sum_chunk(1.0, limit, 1.0)
}

/// Adds the index when it is even and subtracts one when it is odd.
/// Header pc 8, latch pc 14. The recorded path depends on the parity of the
/// iteration that recording starts on.
pub(super) fn alternating_program(limit: f64) -> Chunk {
    Chunk::new(
        10,
        vec![
            Constant::Number(1.0),
            Constant::Number(limit),
            Constant::Number(0.0),
            Constant::Number(2.0),
        ],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 0),
            load(4, 2),
            load(5, 3),
            load(7, 2),
            load(9, 0),
            for_prep(0, 7, 14),
            binary(OpCode::Modulo, 6, 3, 5),
            binary(OpCode::Equal, 8, 6, 7),
            jump_if_false(8, 10, 13),
            binary(OpCode::Add, 4, 4, 3),
            jump_to(12, 14),
            binary(OpCode::Subtract, 4, 4, 9),
            for_loop(0, 14, 8),
            Instruction::return_(4),
        ],
    )
}

/// `alternating_program` whose odd iterations read a global that is not there,
/// which the recorder refuses. No trace can contain that arm, so a trace of
/// the even arm fails its branch every other iteration for good. Header pc 8,
/// latch pc 15.
pub(super) fn alternating_unsupported_program(limit: f64) -> Chunk {
    Chunk::new(
        10,
        vec![
            Constant::Number(1.0),
            Constant::Number(limit),
            Constant::Number(0.0),
            Constant::Number(2.0),
            Constant::String("missing".into()),
        ],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 0),
            load(4, 2),
            load(5, 3),
            load(7, 2),
            load(9, 0),
            for_prep(0, 7, 15),
            binary(OpCode::Modulo, 6, 3, 5),
            binary(OpCode::Equal, 8, 6, 7),
            jump_if_false(8, 10, 13),
            binary(OpCode::Add, 4, 4, 3),
            jump_to(12, 15),
            Instruction::get_global(6, 4),
            binary(OpCode::Subtract, 4, 4, 9),
            for_loop(0, 15, 8),
            Instruction::return_(4),
        ],
    )
}

/// Counts down from `start` to 1 in steps of -1, summing the index.
pub(super) fn descending_program(start: f64) -> Chunk {
    numeric_sum_chunk(start, 1.0, -1.0)
}

/// `x = 1; while x < 1e12 do x = x * 1.2 + 1; count = count + 1 end`.
/// Header pc 5, closing `Jump` at pc 10.
pub(super) fn while_program() -> Chunk {
    Chunk::new(
        7,
        vec![
            Constant::Number(1.0),
            Constant::Number(1e12),
            Constant::Number(1.2),
            Constant::Number(0.0),
        ],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 2),
            load(3, 0),
            load(5, 3),
            binary(OpCode::LessThan, 4, 0, 1),
            jump_if_false(4, 6, 11),
            binary(OpCode::Multiply, 6, 0, 2),
            binary(OpCode::Add, 0, 6, 3),
            binary(OpCode::Add, 5, 5, 3),
            jump_to(10, 5),
            Instruction::return_(5),
        ],
    )
}

/// Negate, Move, `LoadNil`, Not, `ClearRange` and a boolean branch in one body.
/// Header pc 8.
pub(super) fn mixed_ops_program(limit: f64) -> Chunk {
    Chunk::new(
        12,
        vec![
            Constant::Number(1.0),
            Constant::Number(limit),
            Constant::Number(0.0),
            Constant::Boolean(true),
        ],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 0),
            load(9, 2),
            load(10, 3),
            Instruction::load_nil(11),
            Instruction::load_nil(11),
            for_prep(0, 7, 16),
            Instruction::unary(OpCode::Negate, 4, 3),
            Instruction::move_(5, 4),
            Instruction::load_nil(6),
            Instruction::unary(OpCode::Not, 7, 6),
            // r8 = r5 + r3 is always 0.0, then r9 accumulates the index.
            binary(OpCode::Add, 8, 5, 3),
            binary(OpCode::Add, 9, 9, 3),
            jump_if_false(7, 14, 16),
            Instruction::clear_range(6, 2),
            for_loop(0, 16, 8),
            Instruction::return_(9),
        ],
    )
}

/// Division by zero, floor modulo, and a power in the body.
/// Header pc 9.
pub(super) fn float_semantics_program(limit: f64) -> Chunk {
    Chunk::new(
        12,
        vec![
            Constant::Number(1.0),
            Constant::Number(limit),
            Constant::Number(0.0),
            Constant::Number(3.0),
            Constant::Number(2.0),
            Constant::Number(-0.0),
        ],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 0),
            load(4, 2),
            load(5, 3),
            load(6, 4),
            load(7, 2),
            load(8, 5),
            for_prep(0, 8, 17),
            binary(OpCode::Divide, 9, 3, 7),
            binary(OpCode::Modulo, 10, 3, 5),
            binary(OpCode::Power, 11, 3, 6),
            binary(OpCode::Add, 4, 4, 10),
            binary(OpCode::Add, 4, 4, 11),
            binary(OpCode::Subtract, 8, 9, 9),
            binary(OpCode::Add, 7, 7, 7),
            binary(OpCode::Add, 7, 7, 7),
            for_loop(0, 17, 9),
            Instruction::return_(4),
        ],
    )
}

/// An outer loop of `outer` iterations around an inner loop of 3. The inner
/// header is pc 9 and the outer header pc 5.
pub(super) fn nested_program(outer: f64) -> Chunk {
    Chunk::new(
        9,
        vec![
            Constant::Number(1.0),
            Constant::Number(outer),
            Constant::Number(3.0),
            Constant::Number(0.0),
        ],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 0),
            load(8, 3),
            for_prep(0, 4, 11),
            load(4, 0),
            load(5, 2),
            load(6, 0),
            for_prep(4, 8, 10),
            binary(OpCode::Add, 8, 8, 7),
            for_loop(4, 10, 9),
            for_loop(0, 11, 5),
            Instruction::return_(8),
        ],
    )
}

/// Sums the index and returns early once it passes `after`. Header pc 8.
pub(super) fn early_return_program(limit: f64, after: f64) -> Chunk {
    Chunk::new(
        9,
        vec![
            Constant::Number(1.0),
            Constant::Number(limit),
            Constant::Number(0.0),
            Constant::Number(after),
        ],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 0),
            load(4, 2),
            load(5, 3),
            Instruction::load_nil(6),
            Instruction::load_nil(6),
            for_prep(0, 7, 12),
            binary(OpCode::LessThan, 6, 5, 3),
            jump_if_false(6, 9, 11),
            Instruction::return_(4),
            binary(OpCode::Add, 4, 4, 3),
            for_loop(0, 12, 8),
            Instruction::return_(4),
        ],
    )
}

/// Rotates three registers of different types through each other, so the
/// type of r5, r6 and r7 changes every iteration. Header pc 8.
pub(super) fn rotating_program(limit: f64) -> Chunk {
    Chunk::new(
        9,
        vec![
            Constant::Number(1.0),
            Constant::Number(limit),
            Constant::Number(0.0),
            Constant::Boolean(true),
        ],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 0),
            load(4, 2),
            load(5, 0),
            Instruction::load_nil(6),
            load(7, 3),
            for_prep(0, 7, 13),
            Instruction::move_(8, 5),
            Instruction::move_(5, 6),
            Instruction::move_(6, 7),
            Instruction::move_(7, 8),
            binary(OpCode::Add, 4, 4, 3),
            for_loop(0, 13, 8),
            Instruction::return_(4),
        ],
    )
}

/// A loop whose body reads a global that is not there. The read would run
/// `__index`, which a trace cannot repeat, so the recorder refuses it.
pub(super) fn unsupported_program(limit: f64) -> Chunk {
    Chunk::new(
        6,
        vec![
            Constant::Number(1.0),
            Constant::Number(limit),
            Constant::Number(0.0),
            Constant::String("missing".into()),
        ],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 0),
            load(4, 2),
            for_prep(0, 4, 7),
            Instruction::get_global(5, 3),
            binary(OpCode::Add, 4, 4, 3),
            for_loop(0, 7, 5),
            Instruction::return_(4),
        ],
    )
}

pub(super) fn numeric_programs() -> Vec<(&'static str, Chunk)> {
    vec![
        ("sum", sum_program(300.0)),
        ("alternating", alternating_program(300.0)),
        ("descending", descending_program(300.0)),
        ("while", while_program()),
        ("mixed ops", mixed_ops_program(300.0)),
        ("float semantics", float_semantics_program(300.0)),
        ("nested", nested_program(300.0)),
        ("early return", early_return_program(300.0, 100.0)),
        ("rotating", rotating_program(300.0)),
        ("unsupported", unsupported_program(300.0)),
    ]
}

// ------------------------------------------------------------- reference

#[derive(Clone, Debug, PartialEq)]
pub(super) struct State {
    pub(super) pc: usize,
    pub(super) last: Option<usize>,
    pub(super) registers: Vec<TraceScalar>,
}

impl State {
    /// Whether the two states are the same to the interpreter: equal except
    /// in registers nothing reads from `pc` on. A trace restores a register
    /// that only its snapshots mention, and that is dead at the resume pc,
    /// to a placeholder, since the interpreter never looks at it again.
    pub(super) fn same_as(&self, other: &Self, chunk: &Chunk) -> bool {
        self.same_as_with(other, &crate::jit_ir::observable_registers(chunk))
    }

    /// [`Self::same_as`] with the chunk's liveness computed once by the
    /// caller.
    pub(super) fn same_as_with(
        &self,
        other: &Self,
        live_in: &[crate::jit_ir::RegisterSet],
    ) -> bool {
        if self.pc != other.pc
            || self.last != other.last
            || self.registers.len() != other.registers.len()
        {
            return false;
        }
        if self.registers == other.registers {
            return true;
        }
        let live = live_in.get(self.pc);
        self.registers
            .iter()
            .zip(&other.registers)
            .enumerate()
            .all(|(register, (a, b))| {
                a == b
                    || live.is_some_and(|set| {
                        u8::try_from(register).is_ok_and(|register| {
                            !crate::jit_ir::register_set_contains(set, register)
                        })
                    })
            })
    }
}

pub(super) struct Run {
    /// `states[k]` is the state before step `k`.
    pub(super) states: Vec<State>,
    pub(super) results: Vec<TraceScalar>,
}

/// An interpreter-only VM that executes exactly one guest instruction per
/// step. A hook that never fires keeps the dispatch loop from fusing an `Add`
/// with the numeric-for latch behind it, which would hide the state between
/// them.
pub(super) fn interpreter_vm() -> Vm {
    interpreter_vm_with(CompatibilityProfile::Lua51)
}

/// [`interpreter_vm`] for another profile.
pub(super) fn interpreter_vm_with(profile: CompatibilityProfile) -> Vm {
    let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Interpreter);
    vm.debug_hooks.insert(
        None,
        crate::vm::DebugHook {
            function: Value::Nil,
            mask: crate::vm::DebugHookMask::default(),
            count: 0,
            remaining: 0,
            running: false,
        },
    );
    vm.refresh_return_watch();
    vm
}

pub(super) fn frames_for(vm: &Vm, chunk: &Arc<Chunk>, state: Option<&State>) -> CallStack {
    let mut frames = CallStack::from(vec![Frame::new(
        Arc::clone(chunk),
        Vec::new(),
        Vec::new(),
        None,
        None,
        vm.runtime().global_environment(),
        None,
    )]);
    if let Some(state) = state {
        let mut frame = frames.last_mut().unwrap();
        frame.pc = state.pc;
        frame.last_interpreted_pc = state.last.into();
        for (register, scalar) in frame.registers.iter_mut().zip(&state.registers) {
            *register = scalar_to_value(*scalar);
        }
    }
    frames
}

pub(super) fn capture(frames: &CallStack) -> State {
    let frame = frames.last().expect("a live frame");
    State {
        pc: frame.pc,
        last: frame.last_interpreted_pc.get(),
        registers: frame
            .registers
            .iter()
            .map(|value| value_to_scalar(value).expect("test programs use scalars"))
            .collect(),
    }
}

/// Steps to completion from `start` (or the beginning).
pub(super) fn drive(chunk: &Arc<Chunk>, start: Option<&State>) -> Run {
    drive_with(CompatibilityProfile::Lua51, chunk, start)
}

/// [`drive`] under another profile.
pub(super) fn drive_with(
    profile: CompatibilityProfile,
    chunk: &Arc<Chunk>,
    start: Option<&State>,
) -> Run {
    let mut vm = interpreter_vm_with(profile);
    let mut frames = frames_for(&vm, chunk, start);
    let mut states = Vec::new();
    loop {
        states.push(capture(&frames));
        assert!(states.len() < 200_000, "program does not terminate");
        if let Some(results) = vm.step(&mut frames).unwrap() {
            return Run {
                states,
                results: scalars(&results),
            };
        }
    }
}

// ------------------------------------------------------------- evaluator

#[derive(Debug)]
pub(super) enum Evaluation {
    /// An entry input has the wrong type.
    EntryMismatch,
    /// A guard failed in iteration `iteration` (0 is the first). `restored`
    /// is what its snapshot gives the interpreter.
    Guard {
        guard: usize,
        iteration: usize,
        restored: State,
    },
    /// The trace ran every requested iteration. `registers` has the value of
    /// each loop-carried register.
    Iterations {
        registers: Vec<(usize, TraceScalar)>,
    },
}

/// Runs `ir` from `entry` for at most `max_iterations` loop iterations.
pub(super) fn evaluate(ir: &TraceIr, entry: &State, max_iterations: usize) -> Evaluation {
    let mut values: Vec<Option<TraceScalar>> = vec![None; ir.values.len()];
    for input in &ir.entry.inputs {
        let scalar = entry.registers[usize::from(input.register)];
        if scalar.ty() != input.ty {
            return Evaluation::EntryMismatch;
        }
        values[input.value.index()] = Some(scalar);
    }
    for param in &ir.loop_params {
        values[param.value.index()] = Some(match param.init {
            LoopParamInit::Input(value) => values[value.index()].unwrap(),
            LoopParamInit::Const(scalar) => scalar,
        });
    }
    let get = |values: &[Option<TraceScalar>], id: crate::jit_ir::trace::ValueId| {
        values[id.index()].expect("a defined value")
    };
    for iteration in 0..max_iterations {
        for instruction in &ir.body {
            let result = match &instruction.op {
                TraceOp::Constant(scalar) => Some(*scalar),
                TraceOp::Move { source } => Some(get(&values, *source)),
                TraceOp::Negate { source } => {
                    Some(crate::jit_ir::trace::negate(get(&values, *source)).expect("a number"))
                }
                TraceOp::IntToFloat { source } => {
                    Some(crate::jit_ir::trace::to_float(get(&values, *source)).expect("an integer"))
                }
                TraceOp::Arithmetic { op, left, right } => Some(
                    crate::jit_ir::trace::arithmetic(
                        *op,
                        get(&values, *left),
                        get(&values, *right),
                    )
                    .expect("legal arithmetic"),
                ),
                TraceOp::Compare {
                    comparison,
                    left,
                    right,
                } => Some(TraceScalar::Boolean(
                    crate::jit_ir::trace::compare(
                        *comparison,
                        get(&values, *left),
                        get(&values, *right),
                    )
                    .expect("legal comparison"),
                )),
                TraceOp::Truthy { source } => {
                    Some(TraceScalar::Boolean(get(&values, *source).is_truthy()))
                }
                TraceOp::Not { source } => {
                    Some(TraceScalar::Boolean(!get(&values, *source).is_truthy()))
                }
                TraceOp::Select {
                    condition,
                    if_true,
                    if_false,
                } => {
                    let TraceScalar::Boolean(flag) = get(&values, *condition) else {
                        panic!("select on a non-boolean")
                    };
                    Some(get(&values, if flag { *if_true } else { *if_false }))
                }
                TraceOp::GuardBranch {
                    condition,
                    expected,
                    guard,
                } => {
                    let TraceScalar::Boolean(actual) = get(&values, *condition) else {
                        panic!("branch guard on a non-boolean")
                    };
                    if actual != *expected {
                        let info = ir.guards[guard.index()];
                        let snapshot = &ir.snapshots[info.snapshot.index()];
                        let resume = snapshot.resume_point();
                        let registers = snapshot.frames[0]
                            .registers
                            .iter()
                            .enumerate()
                            .map(|(register, entry_kind)| match entry_kind {
                                RegisterEntry::Value(value) => get(&values, *value),
                                RegisterEntry::Const(scalar) => *scalar,
                                RegisterEntry::Unchanged => entry.registers[register],
                                RegisterEntry::Root(_) => panic!("no roots in P2"),
                                RegisterEntry::Virtual(_) | RegisterEntry::Slice(_) => {
                                    panic!("no virtual tables in P2")
                                }
                            })
                            .collect();
                        return Evaluation::Guard {
                            guard: guard.index(),
                            iteration,
                            restored: State {
                                pc: resume.pc,
                                last: resume.last_interpreted_pc,
                                registers,
                            },
                        };
                    }
                    None
                }
                other => panic!("the recorder never emits {other:?}"),
            };
            if let (Some(id), Some(result)) = (instruction.result, result) {
                values[id.index()] = Some(result);
            }
        }
        let TraceTerminator::LoopBack { args } = &ir.terminator else {
            panic!("the recorder closes loops")
        };
        let next: Vec<_> = args.iter().map(|arg| get(&values, *arg)).collect();
        for (param, value) in ir.loop_params.iter().zip(next) {
            values[param.value.index()] = Some(value);
        }
    }
    let registers = ir
        .loop_params
        .iter()
        .map(|param| {
            let LoopParamInit::Input(input) = param.init else {
                panic!("the recorder initialises parameters from entry inputs")
            };
            let register = ir
                .entry
                .inputs
                .iter()
                .find(|candidate| candidate.value == input)
                .map(|candidate| usize::from(candidate.register))
                .unwrap();
            (register, values[param.value.index()].unwrap())
        })
        .collect();
    Evaluation::Iterations { registers }
}

// ----------------------------------------------------------------- checks

/// Runs `chunk` under the trace engine and returns the VM and the results.
pub(super) fn record(profile: CompatibilityProfile, chunk: &Chunk) -> (Vm, Vec<Value>) {
    let mut vm = traced_vm(profile);
    let results = vm.execute(chunk).unwrap();
    (vm, results)
}

pub(super) fn only_trace(vm: &Vm) -> &TraceIr {
    let queued = vm.jit.trace.queued();
    assert_eq!(queued.len(), 1, "{:?}", vm.jit_statistics().trace);
    &queued[0].ir
}

/// Header visits of the reference run: states at the header reached by a
/// backward step.
pub(super) fn header_visits(run: &Run, header: usize) -> Vec<usize> {
    run.states
        .iter()
        .enumerate()
        .filter(|(_, state)| state.pc == header && state.last.is_some_and(|last| last >= header))
        .map(|(index, _)| index)
        .collect()
}

/// Evaluates `ir` from several header visits and compares what it computes
/// with the interpreter. Returns how many guard failures it exercised.
fn check_against_interpreter(chunk: &Arc<Chunk>, ir: &TraceIr, header: usize) -> usize {
    let run = drive(chunk, None);
    let visits = header_visits(&run, header);
    assert!(visits.len() > 70, "the loop runs long enough to test");
    let mut failures = 0;
    for (position, &start) in visits.iter().enumerate().skip(2).step_by(7) {
        let entry = &run.states[start];
        // Run the trace for as many iterations as the interpreter has
        // header visits left, one at a time, to compare each boundary.
        let remaining = visits.len() - position - 1;
        for iterations in 0..=remaining.min(6) {
            match evaluate(ir, entry, iterations) {
                Evaluation::EntryMismatch => break,
                Evaluation::Iterations { registers } => {
                    let expected = &run.states[visits[position + iterations]];
                    for (register, scalar) in registers {
                        assert_eq!(
                            scalar, expected.registers[register],
                            "r{register} after {iterations} iterations from visit {position}"
                        );
                    }
                }
                Evaluation::Guard {
                    guard,
                    iteration,
                    restored,
                } => {
                    failures += 1;
                    assert_restores_to_interpreter_state(chunk, &run, start, &restored, || {
                        format!("guard {guard} in iteration {iteration} from visit {position}")
                    });
                    break;
                }
            }
        }
    }
    failures
}

/// `restored` must be a state the interpreter passes through at or after
/// `start`, and running on from it must finish as the reference does.
pub(super) fn assert_restores_to_interpreter_state(
    chunk: &Arc<Chunk>,
    run: &Run,
    start: usize,
    restored: &State,
    context: impl Fn() -> String,
) {
    let found = run.states[start..]
        .iter()
        .position(|state| state.same_as(restored, chunk))
        .map(|index| index + start);
    let Some(index) = found else {
        panic!(
            "{}: restored state {restored:?} is not an interpreter state",
            context()
        );
    };
    let tail = drive(chunk, Some(restored));
    assert_eq!(tail.results, run.results, "{}", context());
    assert_eq!(tail.states.len(), run.states.len() - index, "{}", context());
}

// ------------------------------------------------------------------ tests

#[test]
fn hot_sum_loop_records_one_closed_trace_with_the_expected_shape() {
    let (vm, results) = record(CompatibilityProfile::Lua51, &sum_program(1000.0));
    assert_eq!(results, vec![Value::Number(500_500.0)]);
    let stats = trace_stats(&vm);
    assert_eq!(stats.recording_starts, 1, "{stats:?}");
    assert_eq!(stats.traces_recorded, 1, "{stats:?}");
    assert_eq!(stats.total_aborts(), 0, "{stats:?}");
    // The body Add and the latch.
    assert_eq!(stats.recorded_instructions, 2, "{stats:?}");
    let queued = &vm.jit.trace.queued()[0];
    assert_eq!(queued.header_pc, 5);
    assert_eq!(queued.guest_instructions, 2);
    let ir = &queued.ir;
    ir.verify().unwrap();
    // Add, then the latch: zero, sign compare, sign guard, advance, bound
    // compare, bound guard.
    assert_eq!(ir.body.len(), 7);
    assert_eq!(ir.guards.len(), 2);
    // Two guard snapshots and the poll snapshot.
    assert_eq!(ir.snapshots.len(), 3);
    let poll = &ir.snapshots[ir
        .poll_snapshot
        .expect("a looping trace is pollable")
        .index()];
    assert_eq!(
        poll.phase,
        SnapshotPhase::After {
            pc: 6,
            successor: 5
        }
    );
    // r0 control, r3 visible index and r4 sum ride in loop parameters. The
    // limit, step and the loaded zero constant never change.
    let carried: Vec<_> = ir.loop_params.iter().map(|param| param.ty).collect();
    assert_eq!(carried, vec![TraceType::Float; 3]);
    assert_eq!(ir.entry.inputs.len(), 5);
    assert_eq!(
        ir.written_registers
            .iter()
            .map(|(_, register)| *register)
            .collect::<Vec<_>>(),
        vec![0, 3, 4]
    );
    assert!(matches!(ir.terminator, TraceTerminator::LoopBack { .. }));
    // The queue holds the trace and the site is Queued.
    // The chunk died with the run, so look the site up by pc.
    assert_eq!(vm.jit.trace.site_state_any(5), Some(SiteState::Queued));
}

#[test]
fn the_recorded_trace_computes_what_the_interpreter_computes() {
    for (name, chunk, header) in [
        ("sum", sum_program(300.0), 5),
        ("alternating", alternating_program(300.0), 8),
        ("descending", descending_program(300.0), 5),
        ("while", while_program(), 5),
        ("mixed ops", mixed_ops_program(300.0), 8),
        ("float semantics", float_semantics_program(300.0), 9),
    ] {
        let (vm, results) = record(CompatibilityProfile::Lua51, &chunk);
        assert_eq!(
            scalars(&results),
            drive(&Arc::new(chunk.clone()), None).results,
            "{name}"
        );
        let queued = vm.jit.trace.queued();
        let trace = queued
            .iter()
            .find(|trace| trace.header_pc == header)
            .unwrap_or_else(|| {
                panic!(
                    "{name}: no trace at header {header}: {:?}",
                    trace_stats(&vm)
                )
            });
        trace.ir.verify().unwrap();
        let failures = check_against_interpreter(&Arc::new(chunk.clone()), &trace.ir, header);
        println!(
            "{name}: {} body ops, {failures} guard failures",
            trace.ir.body.len()
        );
    }
}

#[test]
fn the_alternating_branch_fails_its_guard_and_restores_the_other_path() {
    let chunk = Arc::new(alternating_program(300.0));
    // The arms are pure: with value selects the branch would have no guard.
    let mut vm = traced_vm(CompatibilityProfile::Lua51);
    vm.jit.trace.set_value_selects(false);
    vm.execute(&chunk).unwrap();
    let ir = only_trace(&vm);
    // Recording starts before iteration 64 (even), so the parity guard
    // expects `i % 2 == 0`.
    let guards: Vec<_> = ir
        .body
        .iter()
        .filter_map(|instruction| match instruction.op {
            TraceOp::GuardBranch { expected, .. } => Some((instruction.site.pc, expected)),
            _ => None,
        })
        .collect();
    assert_eq!(guards[0], (10, true));
    let failures = check_against_interpreter(&chunk, ir, 8);
    assert!(
        failures >= 5,
        "odd iterations fail the parity guard: {failures}"
    );
}

#[test]
fn forced_guard_failures_restore_interpreter_states() {
    // The sign guard of the latch: a step of -1 entering a trace recorded
    // with +1 fails it before the latch runs, so the interpreter executes
    // the latch itself.
    let chunk = Arc::new(sum_program(300.0));
    let (vm, _) = record(CompatibilityProfile::Lua51, &chunk);
    let ir = only_trace(&vm);
    let run = drive(&chunk, None);
    let visits = header_visits(&run, 5);
    let mut entry = run.states[visits[80]].clone();
    entry.registers[2] = TraceScalar::Float(-1.0);
    let Evaluation::Guard {
        guard, restored, ..
    } = evaluate(ir, &entry, 5)
    else {
        panic!("the sign guard must fail")
    };
    assert_eq!(guard, 0);
    assert_eq!(restored.pc, 6, "Before the latch");
    assert_eq!(restored.last, Some(5));
    // The state is a real interpreter state of the run that starts there.
    let mutated = drive(&chunk, Some(&entry));
    assert_restores_to_interpreter_state(&chunk, &mutated, 0, &restored, || {
        "sign guard".to_string()
    });

    // The bound guard: the header state has control 81, so a limit of 82
    // lets one iteration finish and stops the next. The After snapshot has advanced the control
    // register and left the visible index alone.
    let mut entry = run.states[visits[80]].clone();
    entry.registers[1] = TraceScalar::Float(82.0);
    let Evaluation::Guard {
        guard,
        iteration,
        restored,
    } = evaluate(ir, &entry, 5)
    else {
        panic!("the bound guard must fail")
    };
    assert_eq!(guard, 1);
    assert_eq!(iteration, 1);
    assert_eq!(restored.pc, 7, "past the latch");
    assert_eq!(restored.last, Some(6));
    let mutated = drive(&chunk, Some(&entry));
    assert_restores_to_interpreter_state(&chunk, &mutated, 0, &restored, || {
        "bound guard".to_string()
    });
}

#[test]
fn zero_and_one_iteration_loops_never_start_a_recording() {
    for limit in [0.0, 1.0, 5.0, f64::from(TRACE_HOT_THRESHOLD) - 1.0] {
        let (vm, results) = record(CompatibilityProfile::Lua51, &sum_program(limit));
        assert_eq!(
            results,
            interpreted(CompatibilityProfile::Lua51, &sum_program(limit)),
            "limit {limit}"
        );
        let stats = trace_stats(&vm);
        assert_eq!(stats.recording_starts, 0, "limit {limit}: {stats:?}");
        assert!(vm.jit.trace.queued().is_empty());
    }
}

#[test]
fn a_loop_that_ends_while_recording_aborts_and_keeps_its_result() {
    // 64 iterations: recording starts before the last one, whose latch falls
    // through. Nothing closes the loop.
    let limit = f64::from(TRACE_HOT_THRESHOLD);
    let (vm, results) = record(CompatibilityProfile::Lua51, &sum_program(limit));
    assert_eq!(
        results,
        interpreted(CompatibilityProfile::Lua51, &sum_program(limit))
    );
    let stats = trace_stats(&vm);
    assert_eq!(stats.recording_starts, 1, "{stats:?}");
    assert_eq!(
        stats.aborts(TraceAbortReason::LoopNotClosed),
        1,
        "{stats:?}"
    );
    assert_eq!(stats.traces_recorded, 0);
    assert!(!vm.jit.trace.recording());

    // One more iteration is enough to close.
    let (vm, _) = record(CompatibilityProfile::Lua51, &sum_program(limit + 1.0));
    assert_eq!(trace_stats(&vm).traces_recorded, 1);
}

#[test]
fn returning_while_recording_aborts_as_a_retry_not_a_rejection() {
    // Recording starts before iteration 64. The index passes 63 inside it,
    // so the recorded iteration reaches the return.
    let chunk = early_return_program(300.0, 63.0);
    let (vm, results) = record(CompatibilityProfile::Lua51, &chunk);
    assert_eq!(results, interpreted(CompatibilityProfile::Lua51, &chunk));
    let stats = trace_stats(&vm);
    assert_eq!(
        stats.aborts(TraceAbortReason::LoopNotClosed),
        1,
        "{stats:?}"
    );
    assert_eq!(stats.aborts(TraceAbortReason::UnsupportedOpcode), 0);
    assert_eq!(stats.traces_recorded, 0);
    assert_eq!(stats.recording_starts, 1);
    // The abort depends on the iteration, so the site is cooling down.
    let site = vm.jit.trace.site_state_any(8);
    assert!(matches!(site, Some(SiteState::Cooldown { .. })), "{site:?}");
}

/// `for j = 1, limit do i = 0; while i < 4 do i = i + 1 end; s = s + i end`.
/// The inner header is pc 8 and its closing `Jump` is pc 11. Each outer
/// iteration takes four inner backedges, so the 64th lands on the loop test
/// that exits.
fn inner_while_program(limit: f64) -> Chunk {
    Chunk::new(
        9,
        vec![
            Constant::Number(1.0),
            Constant::Number(limit),
            Constant::Number(0.0),
            Constant::Number(4.0),
        ],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 0),
            load(4, 2),
            load(7, 3),
            load(8, 0),
            for_prep(0, 6, 13),
            load(5, 2),
            binary(OpCode::LessThan, 6, 5, 7),
            jump_if_false(6, 9, 12),
            binary(OpCode::Add, 5, 5, 8),
            jump_to(11, 8),
            binary(OpCode::Add, 4, 4, 5),
            for_loop(0, 13, 7),
            Instruction::return_(4),
        ],
    )
}

#[test]
fn recording_that_starts_on_a_loop_exit_retries_and_closes_later() {
    let chunk = inner_while_program(300.0);
    let (vm, results) = record(CompatibilityProfile::Lua51, &chunk);
    assert_eq!(results, interpreted(CompatibilityProfile::Lua51, &chunk));
    assert_eq!(results, vec![Value::Number(1200.0)]);
    let stats = trace_stats(&vm);
    // The pass that leaves is dropped without an abort or an attempt, and
    // the next backedge records the loop.
    assert_eq!(
        stats.aborts(TraceAbortReason::LoopNotClosed),
        0,
        "{stats:?}"
    );
    assert_eq!(
        stats.aborts(TraceAbortReason::UnsupportedOpcode),
        0,
        "{stats:?}"
    );
    assert!(
        vm.jit
            .trace
            .queued()
            .iter()
            .any(|queued| queued.header_pc == 8),
        "the inner loop closes a trace after the retry: {stats:?}"
    );
}

/// A sum loop whose body skips the add when `r5` is falsy. `r5` is set before
/// the loop by `load(5, ...)`. Header pc 6, latch pc 8.
fn flag_program(flag: Instruction) -> Chunk {
    Chunk::new(
        6,
        vec![
            Constant::Number(1.0),
            Constant::Number(300.0),
            Constant::Number(0.0),
        ],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 0),
            load(4, 2),
            flag,
            for_prep(0, 5, 8),
            jump_if_false(5, 6, 8),
            binary(OpCode::Add, 4, 4, 3),
            for_loop(0, 8, 6),
            Instruction::return_(4),
        ],
    )
}

#[test]
fn an_unguarded_branch_condition_joins_the_entry_contract() {
    for (name, flag, expected) in [
        ("nil", Instruction::load_nil(5), 0.0),
        ("float", load(5, 0), 45_150.0),
    ] {
        let chunk = flag_program(flag);
        let (vm, results) = record(CompatibilityProfile::Lua51, &chunk);
        assert_eq!(results, vec![Value::Number(expected)], "{name}");
        assert_eq!(results, interpreted(CompatibilityProfile::Lua51, &chunk));
        let ir = only_trace(&vm);
        ir.verify().unwrap();
        let condition = ir
            .entry
            .inputs
            .iter()
            .find(|input| input.register == 5)
            .unwrap_or_else(|| panic!("{name}: r5 missing from {:?}", ir.entry.inputs));
        let expected_type = if name == "nil" {
            TraceType::Nil
        } else {
            TraceType::Float
        };
        assert_eq!(condition.ty, expected_type, "{name}");
    }
}

#[test]
fn an_early_return_after_the_trace_closed_changes_nothing() {
    let chunk = early_return_program(300.0, 100.0);
    let (vm, results) = record(CompatibilityProfile::Lua51, &chunk);
    assert_eq!(results, interpreted(CompatibilityProfile::Lua51, &chunk));
    let stats = trace_stats(&vm);
    assert_eq!(stats.traces_recorded, 1, "{stats:?}");
    assert_eq!(stats.total_aborts(), 0, "{stats:?}");
    // The branch on `i < 100` is a boolean branch, so it carries a guard.
    let ir = only_trace(&vm);
    assert!(ir.guards.len() >= 3);
}

#[test]
fn unsupported_operations_abort_with_a_reason_and_leave_results_alone() {
    let chunk = unsupported_program(300.0);
    let (vm, results) = record(CompatibilityProfile::Lua51, &chunk);
    assert_eq!(results, interpreted(CompatibilityProfile::Lua51, &chunk));
    let stats = trace_stats(&vm);
    // An unsupported operation retries within the budget instead of
    // rejecting the header on the first sighting.
    assert!(stats.recording_starts >= 1, "{stats:?}");
    assert_eq!(
        stats.aborts(TraceAbortReason::UnsupportedOperand),
        stats.recording_starts,
        "{stats:?}"
    );
    assert_eq!(stats.traces_recorded, 0);
    assert!(vm.jit.trace.queued().is_empty());
}

#[test]
fn nested_loops_record_the_inner_loop_and_the_outer_loop_unrolling_it() {
    let chunk = nested_program(300.0);
    let (vm, results) = record(CompatibilityProfile::Lua51, &chunk);
    assert_eq!(results, interpreted(CompatibilityProfile::Lua51, &chunk));
    let stats = trace_stats(&vm);
    let queued = vm.jit.trace.queued();
    let mut headers: Vec<usize> = queued.iter().map(|trace| trace.header_pc).collect();
    headers.sort_unstable();
    // The inner loop has 3 trips, few enough to unroll into the outer loop's
    // trace, so the outer loop records instead of giving up on the inner
    // latch.
    assert_eq!(headers, vec![5, 9], "{stats:?}");
    // The first recording of the outer loop gives up on the inner latch, as
    // it always did; the next one unrolls.
    assert_eq!(stats.aborts(TraceAbortReason::InnerLoop), 1, "{stats:?}");
    let chunk = Arc::new(chunk);
    for trace in queued {
        let failures = check_against_interpreter(&chunk, &trace.ir, trace.header_pc);
        let _ = failures;
    }
}

#[test]
fn loop_that_ends_inside_the_outer_loop_keeps_the_inner_trace_valid() {
    // Nothing else may disturb the interpreter: the exact result of the
    // nested loop is the sum of 1..=3 for every outer iteration.
    let chunk = nested_program(100.0);
    let (_, results) = record(CompatibilityProfile::Lua51, &chunk);
    assert_eq!(results, vec![Value::Number(600.0)]);
}

#[test]
fn recording_runs_on_the_fast_loop_and_matches_the_general_loop() {
    for (name, chunk) in numeric_programs() {
        let (fast, fast_results) = record(CompatibilityProfile::Lua51, &chunk);
        let mut general = traced_vm(CompatibilityProfile::Lua51);
        // Profiling keeps the fast loop out, so only the general loop hooks.
        general.set_interpreter_profiling(true);
        let general_results = general.execute(&chunk).unwrap();
        assert_eq!(fast_results, general_results, "{name}");
        let (fast_stats, general_stats) = (trace_stats(&fast), trace_stats(&general));
        if fast_stats.recorded_instructions > 0 {
            assert!(fast_stats.fast_path_records > 0, "{name}: {fast_stats:?}");
            assert_eq!(fast_stats.general_path_records, 0, "{name}: {fast_stats:?}");
            assert!(
                general_stats.general_path_records > 0,
                "{name}: {general_stats:?}"
            );
            assert_eq!(
                general_stats.fast_path_records, 0,
                "{name}: {general_stats:?}"
            );
        }
        // Same bytecode, same trace, whichever loop observed it.
        assert_eq!(
            fast_stats.recording_aborts, general_stats.recording_aborts,
            "{name}"
        );
        let fast_traces: Vec<_> = fast.jit.trace.queued().iter().map(|t| &t.ir).collect();
        let general_traces: Vec<_> = general.jit.trace.queued().iter().map(|t| &t.ir).collect();
        assert_eq!(fast_traces, general_traces, "{name}");
    }
}

#[test]
fn recording_never_changes_program_results_in_any_profile() {
    for profile in ALL_PROFILES {
        for (name, chunk) in numeric_programs() {
            let expected = interpreted(profile, &chunk);
            let (_, actual) = record(profile, &chunk);
            assert_eq!(actual, expected, "{name} {profile:?}");
        }
    }
}

#[test]
fn every_profile_records_a_numeric_loop() {
    for profile in ALL_PROFILES {
        let (vm, _) = record(profile, &sum_program(300.0));
        let stats = trace_stats(&vm);
        assert_eq!(stats.recording_starts, 1, "{profile:?}: {stats:?}");
        assert_eq!(stats.traces_recorded, 1, "{profile:?}: {stats:?}");
        assert_eq!(stats.total_aborts(), 0, "{profile:?}: {stats:?}");
    }
}

#[test]
fn the_policy_state_machine_counts_cools_down_and_gives_up() {
    let chunk = Arc::new(sum_program(10.0));
    let mut policy = TracePolicy::new(ExecutionMode::Tracing);
    let mut stats = TraceStatistics::default();
    let header = |profile| LoopHeader {
        chunk: &chunk,
        depth: 1,
        pc: 5,
        latch: 6,
        profile,
        coroutine: None,
        captured: &[],
        captured_base: 0,
        closure: None,
        function: false,
    };
    for _ in 0..TRACE_HOT_THRESHOLD - 1 {
        policy.at_loop_header(&mut stats, &header(CompatibilityProfile::Lua51));
    }
    assert!(!policy.recording());
    assert_eq!(
        policy.site_state(&chunk, 5),
        Some(SiteState::Cold {
            backedges: TRACE_HOT_THRESHOLD - 1
        })
    );
    policy.at_loop_header(&mut stats, &header(CompatibilityProfile::Lua51));
    assert!(policy.recording());
    assert_eq!(policy.site_state(&chunk, 5), Some(SiteState::Recording));
    assert_eq!(stats.recording_starts, 1);

    // A transient abort cools down, and the cooldown ends in a new recording.
    policy.abort_recording(&mut stats, TraceAbortReason::Hook);
    assert!(!policy.recording());
    assert!(matches!(
        policy.site_state(&chunk, 5),
        Some(SiteState::Cooldown { .. })
    ));
    let mut consults = 0;
    while !policy.recording() {
        policy.at_loop_header(&mut stats, &header(CompatibilityProfile::Lua51));
        consults += 1;
        assert!(consults < 1000);
    }
    assert!(consults > 1, "the cooldown costs consults");
    assert_eq!(stats.recording_starts, 2);

    // Four transient aborts reject the site for good.
    for _ in 0..3 {
        policy.abort_recording(&mut stats, TraceAbortReason::Hook);
        while !policy.recording() && policy.site_state(&chunk, 5) != Some(SiteState::Rejected) {
            policy.at_loop_header(&mut stats, &header(CompatibilityProfile::Lua51));
        }
    }
    assert_eq!(policy.site_state(&chunk, 5), Some(SiteState::Rejected));
    assert_eq!(stats.aborts(TraceAbortReason::Hook), 4);
    let starts_before = stats.recording_starts;
    for _ in 0..1000 {
        policy.at_loop_header(&mut stats, &header(CompatibilityProfile::Lua51));
    }
    assert_eq!(
        stats.recording_starts, starts_before,
        "rejected sites never record"
    );
}

#[test]
fn flush_aborts_a_recording_and_drops_sites_and_traces() {
    let chunk = Arc::new(sum_program(10.0));
    let mut policy = TracePolicy::new(ExecutionMode::Tracing);
    let mut stats = TraceStatistics::default();
    let header = LoopHeader {
        chunk: &chunk,
        depth: 1,
        pc: 5,
        latch: 6,
        profile: CompatibilityProfile::Lua51,
        coroutine: None,
        captured: &[],
        captured_base: 0,
        closure: None,
        function: false,
    };
    for _ in 0..TRACE_HOT_THRESHOLD {
        policy.at_loop_header(&mut stats, &header);
    }
    assert!(policy.recording());
    policy.flush(&mut stats);
    assert!(!policy.recording());
    assert_eq!(stats.aborts(TraceAbortReason::Flushed), 1);
    assert_eq!(policy.site_state(&chunk, 5), None);

    // Through the VM: a flush request clears queued traces too.
    let mut vm = traced_vm(CompatibilityProfile::Lua51);
    vm.execute(&sum_program(300.0)).unwrap();
    assert_eq!(vm.jit.trace.queued().len(), 1);
    vm.runtime_mut().request_jit_flush();
    vm.service_jit_flush();
    assert!(vm.jit.trace.queued().is_empty());
    assert_eq!(trace_stats(&vm).traces_recorded, 1, "statistics survive");
}

// ------------------------------------------------------- recorder hooks

fn hook<'a>(
    chunk: &'a Arc<Chunk>,
    pc: usize,
    registers: &'a [Value],
    depth: usize,
) -> HookPoint<'a> {
    HookPoint {
        chunk,
        depth,
        top_level: depth == 1,
        pc,
        registers,
        open: &[],
        coroutine: None,
        hooks_active: false,
        source: HookSource::General,
        surroundings: None,
    }
}

#[test]
fn a_repeated_hook_for_an_unexecuted_instruction_records_once() {
    let chunk = Arc::new(sum_program(100.0));
    let mut recorder = Recorder::new(
        Arc::clone(&chunk),
        1,
        5,
        6,
        None,
        CompatibilityProfile::Lua51,
    );
    let registers = vec![
        Value::Number(3.0),
        Value::Number(100.0),
        Value::Number(1.0),
        Value::Number(3.0),
        Value::Number(6.0),
    ];
    let id = TraceId(0);
    assert!(matches!(
        recorder.step(&hook(&chunk, 5, &registers, 1), id),
        Step::Recorded
    ));
    // The general loop reaching the instruction the fast loop declined.
    assert!(matches!(
        recorder.step(&hook(&chunk, 5, &registers, 1), id),
        Step::Skipped
    ));
    assert!(matches!(
        recorder.step(&hook(&chunk, 5, &registers, 1), id),
        Step::Skipped
    ));
    // A different pc than the predicted successor is a divergence.
    assert!(matches!(
        recorder.step(&hook(&chunk, 7, &registers, 1), id),
        Step::Abort(TraceAbortReason::Divergence)
    ));
}

#[test]
fn the_recorder_aborts_on_depth_hooks_and_a_drifted_register_model() {
    let chunk = Arc::new(sum_program(100.0));
    let registers = vec![
        Value::Number(3.0),
        Value::Number(100.0),
        Value::Number(1.0),
        Value::Number(3.0),
        Value::Number(6.0),
    ];
    let id = TraceId(0);

    let mut recorder = Recorder::new(
        Arc::clone(&chunk),
        1,
        5,
        6,
        None,
        CompatibilityProfile::Lua51,
    );
    assert!(matches!(
        recorder.step(&hook(&chunk, 5, &registers, 2), id),
        Step::Abort(TraceAbortReason::DepthChange)
    ));

    let mut recorder = Recorder::new(
        Arc::clone(&chunk),
        1,
        5,
        6,
        None,
        CompatibilityProfile::Lua51,
    );
    let mut at = hook(&chunk, 5, &registers, 1);
    at.hooks_active = true;
    assert!(matches!(
        recorder.step(&at, id),
        Step::Abort(TraceAbortReason::Hook)
    ));

    // The first hook must be the header.
    let mut recorder = Recorder::new(
        Arc::clone(&chunk),
        1,
        5,
        6,
        None,
        CompatibilityProfile::Lua51,
    );
    assert!(matches!(
        recorder.step(&hook(&chunk, 6, &registers, 1), id),
        Step::Abort(TraceAbortReason::Divergence)
    ));

    // The recorder believes r1 is the float it saw at the header. A hook that
    // shows it as nil contradicts that as soon as the latch reads it.
    let mut recorder = Recorder::new(
        Arc::clone(&chunk),
        1,
        5,
        6,
        None,
        CompatibilityProfile::Lua51,
    );
    recorder.step(&hook(&chunk, 5, &registers, 1), id);
    let mut drifted = registers.clone();
    drifted[1] = Value::Nil;
    assert!(matches!(
        recorder.step(&hook(&chunk, 6, &drifted, 1), id),
        Step::Abort(TraceAbortReason::Divergence)
    ));
}

#[test]
fn the_recorder_stops_at_its_instruction_limit() {
    // A body past the instruction limit.
    let body = crate::vm::jit::trace::recorder::MAX_TRACE_OPS + 100;
    let mut code = vec![
        load(0, 0),
        load(1, 1),
        load(2, 0),
        load(4, 2),
        for_prep(0, 4, 5 + body),
    ];
    for _ in 0..body {
        code.push(binary(OpCode::Add, 4, 4, 3));
    }
    code.push(for_loop(0, 5 + body, 5));
    code.push(Instruction::return_(4));
    let chunk = Chunk::new(
        5,
        vec![
            Constant::Number(1.0),
            Constant::Number(200.0),
            Constant::Number(0.0),
        ],
        code,
    );
    let (vm, results) = record(CompatibilityProfile::Lua51, &chunk);
    assert_eq!(results, interpreted(CompatibilityProfile::Lua51, &chunk));
    let stats = trace_stats(&vm);
    assert!(
        stats.aborts(TraceAbortReason::LimitExceeded) >= 1,
        "{stats:?}"
    );
    assert_eq!(stats.traces_recorded, 0);
}

#[test]
fn the_snapshot_limit_leaves_room_for_the_for_loop_latch() {
    // One fewer branch guard than the limit uses that many snapshots. The
    // latch adds two more, which would close a trace one over the limit.
    let guards = crate::vm::jit::trace::recorder::MAX_TRACE_SNAPSHOTS - 1;
    let header = 7;
    let latch = header + guards;
    let mut code = vec![
        load(0, 0),
        load(1, 1),
        load(2, 0),
        load(4, 2),
        Instruction::load_nil(6),
        Instruction::unary(OpCode::Not, 5, 6),
        for_prep(0, 6, latch),
    ];
    for i in 0..guards {
        code.push(jump_if_false(5, header + i, header + 1 + i));
    }
    code.push(for_loop(0, latch, header));
    code.push(Instruction::return_(4));
    let chunk = Chunk::new(
        7,
        vec![
            Constant::Number(1.0),
            Constant::Number(200.0),
            Constant::Number(0.0),
        ],
        code,
    );
    let (vm, results) = record(CompatibilityProfile::Lua51, &chunk);
    assert_eq!(results, interpreted(CompatibilityProfile::Lua51, &chunk));
    let stats = trace_stats(&vm);
    assert!(
        stats.aborts(TraceAbortReason::LimitExceeded) >= 1,
        "{stats:?}"
    );
    assert_eq!(stats.traces_recorded, 0);
}

#[test]
fn a_register_that_changes_type_every_iteration_is_retried_then_rejected() {
    let chunk = rotating_program(300.0);
    let (vm, results) = record(CompatibilityProfile::Lua51, &chunk);
    assert_eq!(results, interpreted(CompatibilityProfile::Lua51, &chunk));
    let stats = trace_stats(&vm);
    assert_eq!(stats.traces_recorded, 0, "{stats:?}");
    // One attempt per retry, until the same abort at the same instruction
    // has come back often enough to say every attempt meets it.
    assert_eq!(
        stats.aborts(TraceAbortReason::TypeUnstable),
        u64::from(MAX_SAME_ABORTS),
        "{stats:?}"
    );
    assert_eq!(
        stats.recording_starts,
        u64::from(MAX_SAME_ABORTS),
        "{stats:?}"
    );
}

#[test]
fn a_register_that_settles_after_the_first_iteration_records_without_a_retry() {
    // r5 is nil when the loop starts and a float from the first iteration
    // on. Recording starts at iteration 64, so it never sees the nil.
    let chunk = Chunk::new(
        6,
        vec![
            Constant::Number(1.0),
            Constant::Number(300.0),
            Constant::Number(0.0),
        ],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 0),
            load(4, 2),
            for_prep(0, 4, 7),
            binary(OpCode::Add, 5, 3, 3),
            binary(OpCode::Add, 4, 4, 5),
            for_loop(0, 7, 5),
            Instruction::return_(4),
        ],
    );
    let (vm, results) = record(CompatibilityProfile::Lua51, &chunk);
    assert_eq!(results, interpreted(CompatibilityProfile::Lua51, &chunk));
    let stats = trace_stats(&vm);
    assert_eq!(stats.traces_recorded, 1, "{stats:?}");
    assert_eq!(stats.total_aborts(), 0, "{stats:?}");
}

#[test]
fn every_profile_creates_a_site() {
    for profile in ALL_PROFILES {
        let chunk = Arc::new(sum_program(300.0));
        let mut policy = TracePolicy::new(ExecutionMode::Tracing);
        let mut stats = TraceStatistics::default();
        let header = LoopHeader {
            chunk: &chunk,
            depth: 1,
            pc: 5,
            latch: 6,
            profile,
            coroutine: None,
            captured: &[],
            captured_base: 0,
            closure: None,
            function: false,
        };
        assert_eq!(
            policy.at_loop_header(&mut stats, &header),
            TraceDispatch::Interpret,
            "{profile:?}"
        );
        assert!(policy.site_state(&chunk, 5).is_some(), "{profile:?}");
    }
}

#[test]
fn skipped_backedges_count_toward_a_cold_sites_hotness() {
    let chunk = Arc::new(sum_program(300.0));
    let mut policy = TracePolicy::new(ExecutionMode::Tracing);
    let mut stats = TraceStatistics::default();
    let header = LoopHeader {
        chunk: &chunk,
        depth: 1,
        pc: 5,
        latch: 6,
        profile: CompatibilityProfile::Lua51,
        coroutine: None,
        captured: &[],
        captured_base: 0,
        closure: None,
        function: false,
    };
    policy.at_loop_header(&mut stats, &header);
    policy.credit_skipped_backedges(&chunk, 5, TRACE_HOT_THRESHOLD);
    policy.at_loop_header(&mut stats, &header);
    assert!(policy.recording(), "a credited window makes the site hot");
}

/// The helpers `ir` calls, in order.
fn called_helpers(ir: &TraceIr) -> Vec<crate::jit_ir::trace::HelperId> {
    ir.preheader
        .iter()
        .chain(&ir.body)
        .filter_map(|instruction| match &instruction.op {
            TraceOp::CallHelper { helper, .. } => Some(*helper),
            _ => None,
        })
        .collect()
}

#[test]
fn string_equality_records_a_byte_comparison_or_folds() {
    use crate::jit_ir::trace::HelperId;
    // `==` against a string constant (`TestEqualK`) and between two
    // registers (`Equal`): a table read gives the string, so only the bytes
    // tell. Two root constants fold, and a number is never a string.
    for (source, compares) in [
        (
            "local p = 'a' local t = {p .. 'b', 'x'} local n = 0 \
             for i = 1, 300 do local s = t[i % 2 + 1] if s == 'ab' then n = n + 1 end end \
             return n",
            true,
        ),
        (
            "local p = 'a' local t = {p .. 'b', 'x'} local n = 0 \
             for i = 1, 300 do local s, u = t[i % 2 + 1], t[1] local e = s == u \
               if e then n = n + 1 end end return n",
            true,
        ),
        (
            "local n = 0 \
             for i = 1, 300 do local s, u = 'ab', 'ab' if s == u then n = n + 1 end \
               if s == 'cd' then n = n + 10 end end return n",
            false,
        ),
        (
            "local t = {1, 2} local n = 0 \
             for i = 1, 300 do local s = t[i % 2 + 1] if s == 'ab' then n = n + 1 end end \
             return n",
            false,
        ),
    ] {
        for profile in ALL_PROFILES {
            let chunk = crate::compiler::compile(source).unwrap();
            let (vm, results) = record(profile, &chunk);
            assert_eq!(
                scalars(&results),
                scalars(&interpreted(profile, &chunk)),
                "{profile:?} {source}"
            );
            let stats = trace_stats(&vm);
            assert_eq!(stats.total_aborts(), 0, "{profile:?} {source} {stats:?}");
            let ir = only_trace(&vm);
            ir.verify().unwrap();
            let helpers = called_helpers(ir);
            assert_eq!(
                helpers.contains(&HelperId::STRING_EQUAL),
                compares,
                "{profile:?} {source} {helpers:?}"
            );
        }
    }
}
