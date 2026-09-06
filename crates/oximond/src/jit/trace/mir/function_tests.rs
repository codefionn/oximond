//! Function traces through the MIR backend, against Cranelift: the
//! activation, its self calls, its tail calls, each type of result, and the
//! exits an activation takes inside nested ones (a depth cap, the fuel, a
//! guard without a side), which every activation around it unwinds.
//!
//! Each case compiles a hand-built tree with each backend, checks that the
//! MIR backend compiled it, and compares what runs observe: the outcome, the
//! control block and every word of the exit area.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    reason = "small test numbers"
)]

use super::super::compile::CompiledTrace;
use super::super::exit::{TraceControl, TraceInputSlot, TraceOutcome};
use super::tests::{LAST_FALLBACK, TIERS, compile};
use crate::jit_ir::Comparison;
use crate::jit_ir::trace::{
    ArithmeticOp, ExternalSlot, FrameId, FunctionTrace, RegisterEntry, Snapshot, SnapshotFrame,
    SnapshotPhase, TraceBuilder, TraceFrameInfo, TraceId, TraceIr, TraceOp, TraceScalar, TraceSide,
    TraceSite, TraceTerminator, TraceType, ValueId,
};

fn site(pc: usize) -> TraceSite {
    TraceSite {
        frame: FrameId(0),
        pc,
    }
}

fn frame(registers: u16) -> TraceFrameInfo {
    TraceFrameInfo {
        register_count: registers,
        code_len: 11,
        inlined: None,
        activation: None,
    }
}

fn snapshot(phase: SnapshotPhase, last: Option<usize>, registers: Vec<RegisterEntry>) -> Snapshot {
    Snapshot::new(
        phase,
        0,
        vec![SnapshotFrame {
            frame: FrameId(0),
            last_interpreted_pc: last,
            registers: registers.into(),
        }],
    )
}

fn emit(b: &mut TraceBuilder, pc: usize, op: TraceOp, ty: TraceType) -> ValueId {
    b.emit(site(pc), op, Some(ty)).unwrap()
}

fn number(ty: TraceType, value: i64) -> TraceScalar {
    if ty == TraceType::Integer {
        TraceScalar::Integer(value)
    } else {
        TraceScalar::Float(value as f64)
    }
}

fn constant(b: &mut TraceBuilder, ty: TraceType, pc: usize, value: i64) -> ValueId {
    emit(b, pc, TraceOp::Constant(number(ty, value)), ty)
}

fn arithmetic(
    b: &mut TraceBuilder,
    ty: TraceType,
    pc: usize,
    op: ArithmeticOp,
    (left, right): (ValueId, ValueId),
) -> ValueId {
    emit(b, pc, TraceOp::Arithmetic { op, left, right }, ty)
}

fn less_than(b: &mut TraceBuilder, pc: usize, left: ValueId, right: ValueId) -> ValueId {
    emit(
        b,
        pc,
        TraceOp::Compare {
            comparison: Comparison::LessThan,
            left,
            right,
        },
        TraceType::Boolean,
    )
}

fn branch(b: &mut TraceBuilder, condition: ValueId, base: crate::jit_ir::trace::SnapshotId) {
    let guard = b.guard(site(1), base);
    b.emit(
        site(1),
        TraceOp::GuardBranch {
            condition,
            expected: false,
            guard,
        },
        None,
    );
}

// ---- fib ----------------------------------------------------------------------

/// `fib` over numbers of type `ty`, as `function_tests.rs` of the trace
/// module has it: the base case `n < 2` is a branch guard, which the side
/// `return n` continues when `side` is set and which exits otherwise.
fn fib(ty: TraceType, side: bool) -> TraceIr {
    use RegisterEntry::{Const, Unchanged, Value};
    let nil = Const(TraceScalar::Nil);
    let function = FunctionTrace {
        parameters: vec![ty],
        result: Some(ty),
        callee_registers: vec![4],
    };
    let mut b = TraceBuilder::new(TraceId(7), frame(5));
    b.set_external(vec![ExternalSlot::Upvalue(0)]);
    b.set_function(function.clone());
    for register in 1..4 {
        b.mark_written(FrameId(0), register);
    }
    let n = b.input(0, ty);
    let callee = b.input(4, TraceType::Closure);
    let two = constant(&mut b, ty, 0, 2);
    let small = less_than(&mut b, 0, n, two);
    let base = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 1,
            successor: 2,
        },
        Some(1),
        vec![Value(n), Value(small), nil, nil, Unchanged],
    ));
    branch(&mut b, small, base);
    let one = constant(&mut b, ty, 4, 1);
    let first_argument = arithmetic(&mut b, ty, 4, ArithmeticOp::Subtract, (n, one));
    let before_first = b.snapshot(snapshot(
        SnapshotPhase::Before { pc: 5 },
        Some(4),
        vec![
            Value(n),
            Value(callee),
            Value(first_argument),
            nil,
            Unchanged,
        ],
    ));
    let guard = b.guard(site(5), before_first);
    let first = emit(
        &mut b,
        5,
        TraceOp::SelfCall {
            args: vec![first_argument],
            guard,
        },
        ty,
    );
    let two = constant(&mut b, ty, 7, 2);
    let second_argument = arithmetic(&mut b, ty, 7, ArithmeticOp::Subtract, (n, two));
    let before_second = b.snapshot(snapshot(
        SnapshotPhase::Before { pc: 8 },
        Some(7),
        vec![
            Value(n),
            Value(first),
            Value(callee),
            Value(second_argument),
            Unchanged,
        ],
    ));
    let guard = b.guard(site(8), before_second);
    let second = emit(
        &mut b,
        8,
        TraceOp::SelfCall {
            args: vec![second_argument],
            guard,
        },
        ty,
    );
    let sum = arithmetic(&mut b, ty, 9, ArithmeticOp::Add, (first, second));
    let entry = b.snapshot(snapshot(
        SnapshotPhase::Before { pc: 0 },
        None,
        vec![Value(n), nil, nil, nil, Unchanged],
    ));
    b.set_poll_snapshot(entry);
    let mut ir = b.finish(TraceTerminator::Return { value: Some(sum) });
    if side {
        let mut b = TraceBuilder::new(TraceId(8), frame(5));
        b.set_external(vec![ExternalSlot::Upvalue(0)]);
        b.set_function(function);
        let n = b.input(0, ty);
        ir.sides.push(TraceSide {
            parent: None,
            guard: 0,
            ir: Box::new(b.finish(TraceTerminator::Return { value: Some(n) })),
        });
    }
    ir
}

// ---- a value passed down and returned ---------------------------------------

/// `f(n, x)`: `if n < 1 then return x end; return f(n - 1, x)`, with `x` of
/// type `ty` (with `ty` `None`, `f(n)` returns nothing). With `cell`, the
/// window also has a cell of that type (register 5), which the snapshots
/// name in register 3: an exit copies it from the input slots.
fn passthrough(ty: Option<TraceType>, cell: Option<TraceType>) -> TraceIr {
    use RegisterEntry::{Const, Unchanged, Value};
    let nil = Const(TraceScalar::Nil);
    let mut parameters = vec![TraceType::Float];
    parameters.extend(ty);
    let function = FunctionTrace {
        parameters,
        result: ty,
        callee_registers: vec![4],
    };
    let external = if cell.is_some() {
        vec![ExternalSlot::Upvalue(0), ExternalSlot::Upvalue(1)]
    } else {
        vec![ExternalSlot::Upvalue(0)]
    };
    let registers = 5 + u16::from(cell.is_some());
    let mut b = TraceBuilder::new(TraceId(9), frame(registers));
    b.set_external(external.clone());
    b.set_function(function.clone());
    for register in 2..4 {
        b.mark_written(FrameId(0), register);
    }
    let n = b.input(0, TraceType::Float);
    let x = ty.map(|ty| b.input(1, ty));
    let held = |x: Option<ValueId>| x.map_or(nil, Value);
    let callee = b.input(4, TraceType::Closure);
    let kept = cell.map(|ty| b.input(5, ty));
    let mut window = vec![Unchanged];
    if kept.is_some() {
        window.push(Unchanged);
    }
    let one = constant(&mut b, TraceType::Float, 0, 1);
    let small = less_than(&mut b, 0, n, one);
    let mut registers = vec![Value(n), held(x), Value(small), held(kept)];
    registers.extend(window.iter().copied());
    let base = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 1,
            successor: 2,
        },
        Some(1),
        registers,
    ));
    branch(&mut b, small, base);
    let next = arithmetic(
        &mut b,
        TraceType::Float,
        4,
        ArithmeticOp::Subtract,
        (n, one),
    );
    let mut registers = vec![Value(n), held(x), Value(next), held(kept)];
    registers.extend(window.iter().copied());
    registers[4] = Value(callee);
    let before = b.snapshot(snapshot(
        SnapshotPhase::Before { pc: 5 },
        Some(4),
        registers,
    ));
    let guard = b.guard(site(5), before);
    let mut args = vec![next];
    args.extend(x);
    let returned = b.emit(site(5), TraceOp::SelfCall { args, guard }, ty);
    let mut registers = vec![Value(n), held(x), nil, nil];
    registers.extend(window.iter().copied());
    let entry = b.snapshot(snapshot(SnapshotPhase::Before { pc: 0 }, None, registers));
    b.set_poll_snapshot(entry);
    let mut ir = b.finish(TraceTerminator::Return { value: returned });
    let mut b = TraceBuilder::new(TraceId(10), frame(5 + u16::from(cell.is_some())));
    b.set_external(external);
    b.set_function(function);
    b.input(0, TraceType::Float);
    let x = ty.map(|ty| b.input(1, ty));
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(b.finish(TraceTerminator::Return { value: x })),
    });
    ir
}

// ---- a tail call --------------------------------------------------------------

/// `g(n, acc)`: `if n < 1 then return acc end; return g(n - 1, acc + n)`,
/// whose tail call starts the activation over (register 3 counts them).
fn tail_sum() -> TraceIr {
    use RegisterEntry::{Const, Value};
    let nil = Const(TraceScalar::Nil);
    let function = FunctionTrace {
        parameters: vec![TraceType::Float, TraceType::Float],
        result: Some(TraceType::Float),
        callee_registers: Vec::new(),
    };
    let mut b = TraceBuilder::new(TraceId(11), frame(4));
    b.set_external(vec![ExternalSlot::TailCalls]);
    b.set_function(function.clone());
    b.mark_written(FrameId(0), 2);
    let n = b.input(0, TraceType::Float);
    let acc = b.input(1, TraceType::Float);
    let count = b.input(3, TraceType::Integer);
    let one = constant(&mut b, TraceType::Float, 0, 1);
    let small = less_than(&mut b, 0, n, one);
    let base = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 1,
            successor: 2,
        },
        Some(1),
        vec![Value(n), Value(acc), Value(small), Value(count)],
    ));
    branch(&mut b, small, base);
    let next = arithmetic(
        &mut b,
        TraceType::Float,
        4,
        ArithmeticOp::Subtract,
        (n, one),
    );
    let sum = arithmetic(&mut b, TraceType::Float, 4, ArithmeticOp::Add, (acc, n));
    let entry = b.snapshot(snapshot(
        SnapshotPhase::Before { pc: 0 },
        None,
        vec![Value(n), Value(acc), nil, Value(count)],
    ));
    b.set_poll_snapshot(entry);
    let mut ir = b.finish(TraceTerminator::TailCall {
        args: vec![next, sum],
        pc: 5,
    });
    let mut b = TraceBuilder::new(TraceId(12), frame(4));
    b.set_external(vec![ExternalSlot::TailCalls]);
    b.set_function(function);
    b.input(0, TraceType::Float);
    let acc = b.input(1, TraceType::Float);
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(b.finish(TraceTerminator::Return { value: Some(acc) })),
    });
    ir
}

// ---- running ------------------------------------------------------------------

/// One run: the input slots, the depth cap and the fuel.
struct Start {
    inputs: Vec<TraceInputSlot>,
    cap: i64,
    fuel: i64,
}

fn start(inputs: Vec<TraceInputSlot>, cap: i64, fuel: i64) -> Start {
    Start { inputs, cap, fuel }
}

#[derive(Debug, PartialEq)]
struct Observed {
    outcome: TraceOutcome,
    control: TraceControl,
    area: Vec<i64>,
}

fn run(trace: &CompiledTrace, start: &Start) -> Observed {
    let stride = trace.activation_stride().expect("a function trace");
    let slots = stride * (usize::try_from(start.cap).unwrap() + 1);
    let mut area = vec![0x5a5a_5a5a_i64; slots];
    let mut control = TraceControl::new(start.fuel);
    control.depth_cap = start.cap;
    let outcome = trace.run(&start.inputs, &mut control, &mut area);
    // Scratch, and the addresses of this run's buffers.
    control.scratch = 0;
    control.inputs = 0;
    control.area = 0;
    Observed {
        outcome,
        control,
        area,
    }
}

/// Compiles `ir` with both backends at both tiers, requires the MIR backend
/// to have compiled it, and compares every run. Returns the MIR runs of the
/// last tier.
fn same_everywhere(name: &str, ir: &TraceIr, starts: &[Start]) -> Vec<Observed> {
    ir.verify()
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    let mut last = Vec::new();
    for tier in TIERS {
        LAST_FALLBACK.with(|fallback| fallback.set(None));
        let mir = compile(ir, tier, true);
        assert!(
            mir.mir_address().is_some(),
            "{name}: the MIR backend compiled it (refused: {:?})",
            LAST_FALLBACK.with(std::cell::Cell::get)
        );
        let cranelift = compile(ir, tier, false);
        assert!(
            cranelift.mir_address().is_none(),
            "{name}: Cranelift compiled it"
        );
        assert_eq!(mir.exits(), cranelift.exits(), "{name}: same exits");
        assert_eq!(mir.activation_stride(), cranelift.activation_stride());
        last.clear();
        for (k, start) in starts.iter().enumerate() {
            let (a, b) = (run(&mir, start), run(&cranelift, start));
            assert_eq!(a, b, "{name} at {tier:?}, start {k}");
            last.push(a);
        }
    }
    last
}

fn float_starts(n: &[f64], caps: &[i64], fuels: &[i64]) -> Vec<Start> {
    let mut starts = Vec::new();
    for &n in n {
        for &cap in caps {
            for &fuel in fuels {
                starts.push(start(
                    vec![TraceInputSlot::float(n), TraceInputSlot::closure(0)],
                    cap,
                    fuel,
                ));
            }
        }
    }
    starts
}

// ---- cases --------------------------------------------------------------------

#[test]
fn recursion_returns_like_cranelift() {
    let starts = float_starts(&[0.0, 1.0, 2.0, 5.0, 15.0], &[100], &[1 << 40]);
    let runs = same_everywhere("fib", &fib(TraceType::Float, true), &starts);
    for observed in &runs {
        assert_eq!(observed.outcome, TraceOutcome::Returned);
    }
    // fib(15)
    assert_eq!(f64::from_bits(runs[4].area[0].cast_unsigned()), 610.0);
    let starts: Vec<Start> = [0, 1, 3, 10, 20]
        .into_iter()
        .map(|n| {
            start(
                vec![TraceInputSlot::integer(n), TraceInputSlot::closure(0)],
                100,
                1 << 40,
            )
        })
        .collect();
    let runs = same_everywhere("integer fib", &fib(TraceType::Integer, true), &starts);
    assert_eq!(runs[4].area[0], 6765);
}

#[test]
fn exits_inside_nested_activations_unwind_like_cranelift() {
    // The depth cap leaves at the activation it stops, the fuel at the one
    // that runs out of it; every activation around it unwinds.
    let starts = float_starts(
        &[6.0, 10.0],
        &[0, 1, 2, 3, 5, 100],
        &[1, 2, 3, 5, 17, 40, 1 << 40],
    );
    let runs = same_everywhere("fib exits", &fib(TraceType::Float, true), &starts);
    assert!(runs.iter().any(
        |observed| matches!(observed.outcome, TraceOutcome::Exit { .. })
            && observed.control.depth > 1
    ));
    // Without the side, the base case's guard leaves from the deepest
    // activation.
    let starts = float_starts(&[1.0, 4.0, 9.0], &[2, 100], &[1 << 40]);
    let runs = same_everywhere("fib guard exits", &fib(TraceType::Float, false), &starts);
    assert!(
        runs.iter()
            .all(|observed| observed.outcome != TraceOutcome::Returned)
    );
}

#[test]
fn entry_mismatches_like_cranelift() {
    let starts = vec![
        start(
            vec![TraceInputSlot::integer(3), TraceInputSlot::closure(0)],
            10,
            100,
        ),
        start(
            vec![TraceInputSlot::float(3.0), TraceInputSlot::integer(0)],
            10,
            100,
        ),
    ];
    let runs = same_everywhere("fib mismatch", &fib(TraceType::Float, true), &starts);
    assert_eq!(runs[0].outcome, TraceOutcome::EntryMismatch { input: 0 });
    assert_eq!(runs[1].outcome, TraceOutcome::EntryMismatch { input: 1 });
}

#[test]
fn each_type_of_result_returns_like_cranelift() {
    let values = [
        (Some(TraceType::Float), TraceInputSlot::float(-2.5)),
        (Some(TraceType::Integer), TraceInputSlot::integer(-77)),
        (Some(TraceType::Boolean), TraceInputSlot::boolean(true)),
        (Some(TraceType::Boolean), TraceInputSlot::boolean(false)),
        (Some(TraceType::Table), TraceInputSlot::table(3)),
        (Some(TraceType::Nil), TraceInputSlot::nil()),
        (None, TraceInputSlot::nil()),
    ];
    for (ty, value) in values {
        let ir = passthrough(ty, None);
        let mut starts = Vec::new();
        for (n, cap, fuel) in [(0.0, 10, 100), (4.0, 10, 100), (4.0, 2, 100), (6.0, 10, 3)] {
            let mut inputs = vec![TraceInputSlot::float(n)];
            if ty.is_some() {
                inputs.push(value);
            }
            inputs.push(TraceInputSlot::closure(0));
            starts.push(start(inputs, cap, fuel));
        }
        let name = format!("passthrough {ty:?}");
        let runs = same_everywhere(&name, &ir, &starts);
        assert_eq!(runs[1].outcome, TraceOutcome::Returned, "{name}");
    }
}

#[test]
fn exits_copy_cells_like_cranelift() {
    let cells = [
        (TraceType::Float, TraceInputSlot::float(0.75)),
        (TraceType::Integer, TraceInputSlot::integer(12)),
        (TraceType::Boolean, TraceInputSlot::boolean(true)),
        (TraceType::Table, TraceInputSlot::table(9)),
    ];
    for (ty, cell) in cells {
        let ir = passthrough(Some(TraceType::Integer), Some(ty));
        let mut starts = Vec::new();
        for (n, cap, fuel) in [(5.0, 10, 100), (5.0, 2, 100), (6.0, 10, 3)] {
            starts.push(start(
                vec![
                    TraceInputSlot::float(n),
                    TraceInputSlot::integer(5),
                    TraceInputSlot::closure(0),
                    cell,
                ],
                cap,
                fuel,
            ));
        }
        same_everywhere(&format!("cell {ty:?}"), &ir, &starts);
    }
}

#[test]
fn tail_calls_start_over_like_cranelift() {
    let mut starts = Vec::new();
    for (n, count, fuel) in [(0.0, 0, 100), (10.0, 0, 100), (10.0, 4, 100), (30.0, 0, 7)] {
        starts.push(start(
            vec![
                TraceInputSlot::float(n),
                TraceInputSlot::float(0.5),
                TraceInputSlot::integer(count),
            ],
            10,
            fuel,
        ));
    }
    let runs = same_everywhere("tail sum", &tail_sum(), &starts);
    assert_eq!(runs[1].outcome, TraceOutcome::Returned);
    assert_eq!(f64::from_bits(runs[1].area[0].cast_unsigned()), 55.5);
    assert!(matches!(runs[3].outcome, TraceOutcome::Exit { .. }));
}
