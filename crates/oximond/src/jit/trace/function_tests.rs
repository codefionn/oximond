//! Function traces: an activation that calls itself natively.
//!
//! The tree is `fib` as the recorder sees it:
//!
//! ```text
//! pc0  r1 = n < 2
//! pc1  JumpIfFalse r1 -> pc3       (recorded: taken, the recursive path)
//! pc2  return n                     (the side trace's path)
//! pc3  r1 = fib                     (upvalue 0, window slot r4)
//! pc4  r2 = n - 1
//! pc5  r1 = r1(r2)
//! pc6  r2 = fib
//! pc7  r3 = n - 2
//! pc8  r2 = r2(r3)
//! pc9  r1 = r1 + r2
//! pc10 return r1
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    reason = "small test numbers"
)]

use super::*;
use crate::jit::OptimizationTier;
use crate::jit_ir::Comparison;
use crate::jit_ir::trace::{
    ArithmeticOp, ExternalSlot, FrameId, FunctionTrace, RegisterEntry, Snapshot, SnapshotFrame,
    SnapshotPhase, TraceBuilder, TraceFrameInfo, TraceId, TraceIr, TraceOp, TraceScalar, TraceSide,
    TraceSite, TraceTerminator, TraceType, ValueId,
};

const TIERS: [OptimizationTier; 2] = [OptimizationTier::Baseline, OptimizationTier::Optimized];

fn site(pc: usize) -> TraceSite {
    TraceSite {
        frame: FrameId(0),
        pc,
    }
}

fn frame() -> TraceFrameInfo {
    TraceFrameInfo {
        register_count: 5,
        code_len: 11,
        inlined: None,
        activation: None,
    }
}

fn function(ty: TraceType) -> FunctionTrace {
    FunctionTrace {
        parameters: vec![ty],
        result: Some(ty),
        callee_registers: vec![4],
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

/// A number of type `ty`.
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

fn subtract(
    b: &mut TraceBuilder,
    ty: TraceType,
    pc: usize,
    left: ValueId,
    right: ValueId,
) -> ValueId {
    emit(
        b,
        pc,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Subtract,
            left,
            right,
        },
        ty,
    )
}

fn root() -> TraceIr {
    root_of(TraceType::Float)
}

/// The recursive path, with the base case as its one branch guard, over
/// numbers of type `ty`.
fn root_of(ty: TraceType) -> TraceIr {
    use RegisterEntry::{Const, Unchanged, Value};
    let nil = Const(TraceScalar::Nil);
    let mut b = TraceBuilder::new(TraceId(7), frame());
    b.set_external(vec![ExternalSlot::Upvalue(0)]);
    b.set_function(function(ty));
    for register in 1..4 {
        b.mark_written(FrameId(0), register);
    }
    let n = b.input(0, ty);
    let callee = b.input(4, TraceType::Closure);
    let two = constant(&mut b, ty, 0, 2);
    let small = emit(
        &mut b,
        0,
        TraceOp::Compare {
            comparison: Comparison::LessThan,
            left: n,
            right: two,
        },
        TraceType::Boolean,
    );
    let base = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 1,
            successor: 2,
        },
        Some(1),
        vec![Value(n), Value(small), nil, nil, Unchanged],
    ));
    let guard = b.guard(site(1), base);
    b.emit(
        site(1),
        TraceOp::GuardBranch {
            condition: small,
            expected: false,
            guard,
        },
        None,
    );
    let one = constant(&mut b, ty, 4, 1);
    let first_argument = subtract(&mut b, ty, 4, n, one);
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
    let second_argument = subtract(&mut b, ty, 7, n, two);
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
    let sum = emit(
        &mut b,
        9,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: first,
            right: second,
        },
        ty,
    );
    let entry = b.snapshot(snapshot(
        SnapshotPhase::Before { pc: 0 },
        None,
        vec![Value(n), nil, nil, nil, Unchanged],
    ));
    b.set_poll_snapshot(entry);
    b.finish(TraceTerminator::Return { value: Some(sum) })
}

/// The base case, continuing the root's branch guard: `return n`.
fn base_case(ty: TraceType) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(8), frame());
    b.set_external(vec![ExternalSlot::Upvalue(0)]);
    b.set_function(function(ty));
    let n = b.input(0, ty);
    b.finish(TraceTerminator::Return { value: Some(n) })
}

fn tree() -> TraceIr {
    tree_of(TraceType::Float)
}

fn tree_of(ty: TraceType) -> TraceIr {
    let mut ir = root_of(ty);
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(base_case(ty)),
    });
    ir
}

fn fib(n: u32) -> f64 {
    let (mut a, mut b) = (0.0, 1.0);
    for _ in 0..n {
        (a, b) = (b, a + b);
    }
    a
}

fn inputs(n: f64) -> [TraceInputSlot; 2] {
    [TraceInputSlot::float(n), TraceInputSlot::closure(0)]
}

fn run(
    trace: &CompiledTrace,
    n: f64,
    cap: i64,
    fuel: i64,
) -> (TraceOutcome, TraceControl, Vec<i64>) {
    let stride = trace.activation_stride().expect("a function trace");
    let mut area = vec![0_i64; stride * (usize::try_from(cap).unwrap() + 1)];
    let mut control = TraceControl::new(fuel);
    control.depth_cap = cap;
    let outcome = trace.run(&inputs(n), &mut control, &mut area);
    (outcome, control, area)
}

#[test]
fn the_tree_verifies() {
    tree().verify().unwrap();
    root().verify().unwrap();
}

#[test]
fn recursion_runs_natively_and_returns() {
    for tier in TIERS {
        let trace = compile_trace(&tree(), tier).unwrap();
        for n in [0_u32, 1, 2, 5, 20] {
            let (outcome, control, area) = run(&trace, f64::from(n), 100, 1 << 40);
            assert_eq!(outcome, TraceOutcome::Returned, "{tier:?} fib({n})");
            assert_eq!(
                f64::from_bits(area[0].cast_unsigned()),
                fib(n),
                "{tier:?} fib({n})"
            );
            assert_eq!(control.depth, 0);
            // One unit of fuel per activation.
            let calls = if n < 2 { 1 } else { 2 * fib(n + 1) as i64 - 1 };
            assert_eq!((1_i64 << 40) - control.fuel, calls, "{tier:?} fib({n})");
        }
    }
}

#[test]
fn an_integer_recursion_returns_its_integer() {
    // An integer status, fuel and result would not fit the return
    // registers: the status carries the fuel.
    for tier in TIERS {
        let trace = compile_trace(&tree_of(TraceType::Integer), tier).unwrap();
        let stride = trace.activation_stride().unwrap();
        let mut area = vec![0_i64; stride * 101];
        let mut control = TraceControl::new(1 << 40);
        control.depth_cap = 100;
        let outcome = trace.run(
            &[TraceInputSlot::integer(24), TraceInputSlot::closure(0)],
            &mut control,
            &mut area,
        );
        assert_eq!(outcome, TraceOutcome::Returned, "{tier:?}");
        assert_eq!(area[0], fib(24) as i64, "{tier:?}");
    }
}

#[test]
fn the_deepest_activation_leaves_before_its_call_and_the_others_unwind() {
    for tier in TIERS {
        let trace = compile_trace(&tree(), tier).unwrap();
        let stride = trace.activation_stride().unwrap();
        let (outcome, control, area) = run(&trace, 10.0, 3, 1 << 40);
        let TraceOutcome::Exit { index } = outcome else {
            panic!("{tier:?}: {outcome:?}");
        };
        // fib(10) calls fib(9), fib(8), fib(7) on its way down; fib(7) is at
        // the cap and leaves before its first self call.
        assert_eq!(control.depth, 3);
        let exit = &trace.exits()[index];
        assert_eq!(exit.class, TraceExitClass::GuardFail);
        assert_eq!(exit.guard, Some(TraceGuardClass::Depth));
        let deepest = &area[3 * stride..4 * stride];
        let snapshot = &trace.lowered_member(exit.member).snapshots[exit.snapshot.index()];
        let RegisterEntry::Value(n) = snapshot.frames[0].registers[0] else {
            panic!("n is a value");
        };
        assert_eq!(exit.scalar(deepest, n), Some(TraceScalar::Float(7.0)));
        for depth in 0..3 {
            let own = &area[depth * stride..(depth + 1) * stride];
            let unwind = usize::try_from(own[stride - 1]).unwrap();
            let exit = &trace.exits()[unwind];
            assert_eq!(exit.class, TraceExitClass::Unwind);
            let snapshot = &trace.lowered_member(exit.member).snapshots[exit.snapshot.index()];
            assert_eq!(snapshot.phase, SnapshotPhase::Before { pc: 5 });
            let RegisterEntry::Value(n) = snapshot.frames[0].registers[0] else {
                panic!("n is a value");
            };
            assert_eq!(
                exit.scalar(own, n),
                Some(TraceScalar::Float(10.0 - depth as f64))
            );
        }
    }
}

#[test]
fn an_activation_polls_when_the_fuel_runs_out() {
    let trace = compile_trace(&tree(), OptimizationTier::Optimized).unwrap();
    let (outcome, control, _) = run(&trace, 10.0, 100, 5);
    let TraceOutcome::Exit { index } = outcome else {
        panic!("{outcome:?}");
    };
    assert_eq!(trace.exits()[index].class, TraceExitClass::Poll);
    // Activations 0 to 3 ran (fib 10, 9, 8, 7); the fifth (fib 6) polled.
    assert_eq!(control.depth, 4);
    assert_eq!(control.fuel, 0);
}

#[test]
fn a_wrong_entry_type_mismatches() {
    let trace = compile_trace(&tree(), OptimizationTier::Optimized).unwrap();
    let stride = trace.activation_stride().unwrap();
    let mut area = vec![0_i64; stride];
    let mut control = TraceControl::new(100);
    let outcome = trace.run(
        &[TraceInputSlot::integer(3), TraceInputSlot::closure(0)],
        &mut control,
        &mut area,
    );
    assert_eq!(outcome, TraceOutcome::EntryMismatch { input: 0 });
}

#[test]
fn a_self_call_must_pass_the_entry_types() {
    let mut ir = root();
    let one = ir
        .body
        .iter()
        .position(|instruction| matches!(instruction.op, TraceOp::SelfCall { .. }))
        .unwrap();
    // Pass the comparison (a boolean) instead of the float.
    if let TraceOp::SelfCall { args, .. } = &mut ir.body[one].op {
        args[0] = ValueId(3);
    }
    assert!(ir.verify().is_err());
}
