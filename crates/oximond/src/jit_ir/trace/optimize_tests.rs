#![allow(
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::many_single_char_names
)]

use super::super::ir::*;
use super::super::reference::{Evaluation, ForcedExit, Outcome, evaluate, evaluate_forced};
use super::super::snapshot::*;
use super::super::verify::TraceVerifyError;
use super::{
    Draft, OptimizeLevel, OptimizeOptions, OptimizeStats, TreeWrites, eliminate_copies,
    eliminate_dead_code, eliminate_guards, fold_constants, hoist_invariants, number_values,
    optimize_pipeline, optimize_trace, optimize_with,
};
use crate::compat::FloatModulo;
use crate::jit_ir::Comparison;
use std::collections::HashSet;

fn site(pc: usize) -> TraceSite {
    TraceSite {
        frame: FrameId(0),
        pc,
    }
}

fn root(registers: u16) -> TraceFrameInfo {
    TraceFrameInfo {
        register_count: registers,
        code_len: 512,
        inlined: None,
        activation: None,
    }
}

fn snap(
    phase: SnapshotPhase,
    effects: u32,
    last: Option<usize>,
    registers: Vec<RegisterEntry>,
) -> Snapshot {
    Snapshot::new(
        phase,
        effects,
        vec![SnapshotFrame {
            frame: FrameId(0),
            last_interpreted_pc: last,
            registers: registers.into(),
        }],
    )
}

fn f(value: f64) -> TraceScalar {
    TraceScalar::Float(value)
}

fn int(value: i64) -> TraceScalar {
    TraceScalar::Integer(value)
}

fn emit(b: &mut TraceBuilder, pc: usize, op: TraceOp, ty: TraceType) -> ValueId {
    b.emit(site(pc), op, Some(ty)).unwrap()
}

fn konst(b: &mut TraceBuilder, pc: usize, scalar: TraceScalar) -> ValueId {
    emit(b, pc, TraceOp::Constant(scalar), scalar.ty())
}

fn arith(b: &mut TraceBuilder, pc: usize, op: ArithmeticOp, l: ValueId, r: ValueId) -> ValueId {
    let ty = b.value_type_for_tests(l);
    emit(
        b,
        pc,
        TraceOp::Arithmetic {
            op,
            left: l,
            right: r,
        },
        ty,
    )
}

/// A straight-line trace that exits through one snapshot holding `results`.
/// Registers are `results.len()` wide, and the exit snapshot names each.
fn straight(build: impl FnOnce(&mut TraceBuilder) -> Vec<ValueId>) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(1), root(1));
    let results = build(&mut b);
    let mut ir = b.finish(TraceTerminator::Exit {
        snapshot: SnapshotId(0),
    });
    ir.frames[0].register_count = u16::try_from(results.len()).unwrap();
    let snapshot = snap(
        SnapshotPhase::Before { pc: 400 },
        ir.effect_count(),
        None,
        results
            .iter()
            .map(|value| RegisterEntry::Value(*value))
            .collect(),
    );
    ir.snapshots.push(Snapshot {
        id: SnapshotId(0),
        ..snapshot
    });
    for index in 0..results.len() {
        ir.written_registers
            .push((FrameId(0), u16::try_from(index).unwrap()));
    }
    ir.verify().unwrap();
    ir
}

fn exit_registers(ir: &TraceIr) -> Vec<TraceScalar> {
    let evaluation = evaluate(ir, &[], &[], 1, |_, _| None);
    let Outcome::Exited(restored) = evaluation.outcome else {
        panic!("expected an exit, got {:?}", evaluation.outcome)
    };
    restored.registers
}

impl TraceBuilder {
    fn value_type_for_tests(&self, value: ValueId) -> TraceType {
        self.peek_type(value)
    }
}

fn run_pass(
    ir: &TraceIr,
    pass: fn(&TraceIr, &mut OptimizeStats) -> Option<TraceIr>,
) -> (TraceIr, OptimizeStats) {
    let mut stats = OptimizeStats::default();
    // `None` means the pass changed nothing, which is the input itself.
    let out = pass(ir, &mut stats).unwrap_or_else(|| ir.clone());
    out.verify().unwrap();
    (out, stats)
}

fn count(ir: &TraceIr, predicate: impl Fn(&TraceOp) -> bool) -> usize {
    ir.preheader
        .iter()
        .chain(&ir.body)
        .filter(|instruction| predicate(&instruction.op))
        .count()
}

// ---- copy elimination ------------------------------------------------------

/// [`hoist_invariants`] as a single trace runs it.
fn hoist_with(ir: &TraceIr, hoist_guards: bool, stats: &mut OptimizeStats) -> TraceIr {
    let options = OptimizeOptions {
        hoist_guards,
        ..OptimizeOptions::for_level(OptimizeLevel::Optimized)
    };
    hoist_invariants(ir, &options, &TreeWrites::of_members([ir]), stats)
        .unwrap_or_else(|| ir.clone())
}

#[test]
fn copy_elimination_forwards_sources_through_chains_and_snapshots() {
    let ir = straight(|b| {
        let one = konst(b, 1, f(1.0));
        let a = emit(b, 2, TraceOp::Move { source: one }, TraceType::Float);
        let c = emit(b, 3, TraceOp::Move { source: a }, TraceType::Float);
        let sum = arith(b, 4, ArithmeticOp::Add, c, a);
        vec![sum, c]
    });
    let (out, stats) = run_pass(&ir, eliminate_copies);
    assert_eq!(stats.copies_removed, 2);
    assert_eq!(count(&out, |op| matches!(op, TraceOp::Move { .. })), 0);
    assert_eq!(exit_registers(&out), exit_registers(&ir));
}

// ---- constant folding ------------------------------------------------------

#[test]
fn folding_evaluates_float_operations_with_exact_lua_semantics() {
    let values = [5.5, -2.0, 0.0, 2.0, 10.0, f64::INFINITY, f64::NAN, -0.0];
    let ir = straight(|b| {
        let c: Vec<ValueId> = values.iter().map(|v| konst(b, 1, f(*v))).collect();
        let mut out = Vec::new();
        for op in [
            ArithmeticOp::Add,
            ArithmeticOp::Subtract,
            ArithmeticOp::Multiply,
            ArithmeticOp::Divide,
            ArithmeticOp::Modulo,
            ArithmeticOp::Power,
            ArithmeticOp::ModuloWith(FloatModulo::Floor),
            ArithmeticOp::ModuloWith(FloatModulo::Fmod53),
            ArithmeticOp::ModuloWith(FloatModulo::Fmod54),
            ArithmeticOp::PowerSquare,
        ] {
            for i in 0..values.len() {
                for j in 0..values.len() {
                    out.push(arith(b, 2, op, c[i], c[j]));
                }
            }
        }
        for source in &c {
            out.push(emit(
                b,
                3,
                TraceOp::Negate { source: *source },
                TraceType::Float,
            ));
        }
        out
    });
    let (out, stats) = run_pass(&ir, fold_constants);
    assert!(stats.folded > 300);
    assert_eq!(
        count(&out, |op| matches!(
            op,
            TraceOp::Arithmetic { .. } | TraceOp::Negate { .. }
        )),
        0
    );
    // Bit-exact: NaN payloads and signed zero included. TraceScalar equality
    // compares float bits.
    assert_eq!(exit_registers(&out), exit_registers(&ir));
}

#[test]
fn folding_keeps_spot_values_of_the_corner_cases() {
    let ir = straight(|b| {
        let a = konst(b, 1, f(5.5));
        let c = konst(b, 1, f(-2.0));
        let zero = konst(b, 1, f(0.0));
        let nzero = konst(b, 1, f(-0.0));
        let inf = konst(b, 1, f(f64::INFINITY));
        vec![
            arith(b, 2, ArithmeticOp::Modulo, a, c),
            arith(b, 2, ArithmeticOp::Modulo, c, a),
            arith(b, 2, ArithmeticOp::Modulo, a, zero),
            arith(b, 2, ArithmeticOp::Modulo, inf, a),
            arith(b, 2, ArithmeticOp::Divide, a, zero),
            arith(b, 2, ArithmeticOp::Divide, a, nzero),
            arith(b, 2, ArithmeticOp::Divide, zero, zero),
            arith(b, 2, ArithmeticOp::Subtract, inf, inf),
            arith(b, 2, ArithmeticOp::Add, nzero, nzero),
            arith(b, 2, ArithmeticOp::Add, nzero, zero),
            emit(b, 3, TraceOp::Negate { source: zero }, TraceType::Float),
        ]
    });
    let (out, _) = run_pass(&ir, fold_constants);
    let registers = exit_registers(&out);
    assert_eq!(registers[0], f(-0.5));
    assert_eq!(registers[1], f(3.5));
    assert!(matches!(registers[2], TraceScalar::Float(v) if v.is_nan()));
    assert!(matches!(registers[3], TraceScalar::Float(v) if v.is_nan()));
    assert_eq!(registers[4], f(f64::INFINITY));
    assert_eq!(registers[5], f(f64::NEG_INFINITY));
    assert!(matches!(registers[6], TraceScalar::Float(v) if v.is_nan()));
    assert!(matches!(registers[7], TraceScalar::Float(v) if v.is_nan()));
    assert_eq!(registers[8], f(-0.0));
    assert_eq!(registers[9], f(0.0));
    assert_eq!(registers[10], f(-0.0));
    assert_ne!(registers[10], f(0.0), "signed zero survives folding");
    assert_eq!(registers, exit_registers(&ir));
}

/// The values stock Lua prints (lua5.1, lua5.4, luajit; see the corpus entries
/// float-modulo-* and float-power-lua54), folded at compile time.
#[test]
fn folding_float_modulo_and_square_follow_each_profile() {
    let quotient = 4.0 / -2.25;
    let cube = 483.0_f64.powf(3.0);
    let ir = straight(|b| {
        let a = konst(b, 1, f(-583.0));
        let d = konst(b, 1, f(3.001));
        let big = konst(b, 1, f(1e10));
        let q = konst(b, 1, f(quotient));
        let x = konst(b, 1, f(cube));
        let two = konst(b, 1, f(2.0));
        let inf = konst(b, 1, f(f64::INFINITY));
        let five = konst(b, 1, f(5.0));
        let nzero = konst(b, 1, f(-0.0));
        let three = konst(b, 1, f(3.0));
        let ninf = konst(b, 1, f(f64::NEG_INFINITY));
        let with = |kind| ArithmeticOp::ModuloWith(kind);
        vec![
            arith(b, 2, with(FloatModulo::Floor), a, d),
            arith(b, 2, with(FloatModulo::Floor), big, q),
            arith(b, 2, ArithmeticOp::Modulo, a, d),
            arith(b, 2, ArithmeticOp::Modulo, big, q),
            arith(b, 2, with(FloatModulo::Fmod54), a, d),
            arith(b, 2, with(FloatModulo::Fmod54), big, q),
            arith(b, 2, with(FloatModulo::Fmod53), big, q),
            arith(b, 2, ArithmeticOp::PowerSquare, x, two),
            arith(b, 2, ArithmeticOp::Power, x, two),
            arith(b, 2, with(FloatModulo::Fmod54), five, inf),
            arith(b, 2, with(FloatModulo::Fmod54), nzero, three),
            arith(b, 2, with(FloatModulo::Fmod54), five, ninf),
            arith(b, 2, with(FloatModulo::Floor), five, inf),
        ]
    });
    let (out, _) = run_pass(&ir, fold_constants);
    assert_eq!(
        count(&out, |op| matches!(op, TraceOp::Arithmetic { .. })),
        0
    );
    let registers = exit_registers(&out);
    assert_eq!(registers[0], f(2.194_999_999_999_936_3));
    assert_eq!(registers[1], f(0.0));
    assert_eq!(registers[2], f(2.194_999_999_999_936_3));
    assert_eq!(registers[3], f(0.0));
    assert_eq!(registers[4], f(2.194_999_999_999_978_5));
    assert_eq!(registers[5], f(-1.777_777_222_666_265_4));
    assert_eq!(registers[6], f(-1.777_777_222_666_265_4));
    assert_eq!(registers[7], f(12_696_463_968_316_568.0));
    // The C library's pow, as stock Lua calls it: glibc rounds this tie of
    // 12_696_463_968_316_569 up to ..570, the Windows C runtime to ..568.
    assert_eq!(registers[8], f(483.0_f64.powf(3.0).powf(2.0)));
    assert_eq!(registers[9], f(5.0));
    assert_eq!(registers[10], f(-0.0));
    assert_eq!(registers[11], f(f64::NEG_INFINITY));
    assert!(matches!(registers[12], TraceScalar::Float(v) if v.is_nan()));
}

#[test]
fn folding_compares_nan_and_signed_zero_like_the_interpreter() {
    use TraceScalar::Boolean as B;
    let ir = straight(|b| {
        let nan = konst(b, 1, f(f64::NAN));
        let zero = konst(b, 1, f(0.0));
        let nzero = konst(b, 1, f(-0.0));
        let mut out = Vec::new();
        for (cmp, l, r) in [
            (Comparison::Equal, nan, nan),
            (Comparison::Equal, zero, nzero),
            (Comparison::LessThan, nzero, zero),
            (Comparison::LessEqual, nzero, zero),
            (Comparison::LessThan, nan, zero),
            (Comparison::LessEqual, nan, nan),
        ] {
            out.push(emit(
                b,
                2,
                TraceOp::Compare {
                    comparison: cmp,
                    left: l,
                    right: r,
                },
                TraceType::Boolean,
            ));
        }
        out
    });
    let (out, stats) = run_pass(&ir, fold_constants);
    assert_eq!(stats.folded, 6);
    assert_eq!(
        exit_registers(&out),
        [B(false), B(true), B(false), B(true), B(false), B(false)]
    );
}

#[test]
fn folding_wraps_integers_and_applies_integer_identities_only() {
    let ir = straight(|b| {
        let max = konst(b, 1, int(i64::MAX));
        let one = konst(b, 1, int(1));
        let zero = konst(b, 1, int(0));
        let min = konst(b, 1, int(i64::MIN));
        let fzero = konst(b, 1, f(0.0));
        vec![
            arith(b, 2, ArithmeticOp::Add, max, one),
            arith(b, 2, ArithmeticOp::Multiply, min, max),
            emit(b, 2, TraceOp::Negate { source: min }, TraceType::Integer),
            arith(b, 2, ArithmeticOp::Subtract, zero, one),
            fzero,
        ]
    });
    let (out, _) = run_pass(&ir, fold_constants);
    let registers = exit_registers(&out);
    assert_eq!(registers[0], int(i64::MIN));
    assert_eq!(registers[2], int(i64::MIN));
    assert_eq!(registers[3], int(-1));
    assert_eq!(registers, exit_registers(&ir));

    // x + 0, x * 1, x - x, x * 0 fold for integers. x + 0.0 does not fold
    // for floats, because -0.0 + 0.0 is +0.0.
    let mut b = TraceBuilder::new(TraceId(1), root(2));
    let xi = b.input(0, TraceType::Integer);
    let xf = b.input(1, TraceType::Float);
    let zero = konst(&mut b, 1, int(0));
    let one = konst(&mut b, 1, int(1));
    let fzero = konst(&mut b, 1, f(0.0));
    let a = arith(&mut b, 2, ArithmeticOp::Add, xi, zero);
    let c = arith(&mut b, 2, ArithmeticOp::Multiply, one, a);
    let d = arith(&mut b, 2, ArithmeticOp::Subtract, c, c);
    let e = arith(&mut b, 2, ArithmeticOp::Add, xf, fzero);
    b.mark_written(FrameId(0), 0);
    b.mark_written(FrameId(0), 1);
    let mut ir = b.finish(TraceTerminator::Exit {
        snapshot: SnapshotId(0),
    });
    ir.snapshots.push(Snapshot {
        id: SnapshotId(0),
        ..snap(
            SnapshotPhase::Before { pc: 400 },
            0,
            None,
            vec![RegisterEntry::Value(c), RegisterEntry::Value(e)],
        )
    });
    ir.verify().unwrap();
    let (out, _) = run_pass(&ir, fold_constants);
    let out = run_pass(&out, eliminate_copies).0;
    let out = run_pass(&out, eliminate_dead_code).0;
    // `c` aliases the integer input and `d` is gone. The float add stays.
    assert_eq!(
        count(&out, |op| matches!(
            op,
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                ..
            }
        )),
        1
    );
    let outcome = evaluate(&out, &[int(7), f(-0.0)], &[], 1, |_, _| None).outcome;
    let Outcome::Exited(restored) = outcome else {
        panic!()
    };
    assert_eq!(restored.registers, [int(7), f(0.0)]);
    let _ = d;
}

#[test]
fn folding_evaluates_integer_division_shifts_conversions_and_mixed_compares() {
    let ir = straight(|b| {
        let min = konst(b, 1, int(i64::MIN));
        let max = konst(b, 1, int(i64::MAX));
        let minus_one = konst(b, 1, int(-1));
        let seven = konst(b, 1, int(7));
        let minus_three = konst(b, 1, int(-3));
        let sixty_four = konst(b, 1, int(64));
        let two_53 = konst(b, 1, int(9_007_199_254_740_993));
        let float_two_53 = konst(b, 1, f(9_007_199_254_740_992.0));
        let float_two_63 = konst(b, 1, f(9_223_372_036_854_775_808.0));
        let to_float = emit(
            b,
            2,
            TraceOp::IntToFloat { source: two_53 },
            TraceType::Float,
        );
        let less = |b: &mut TraceBuilder, comparison, left, right| {
            emit(
                b,
                2,
                TraceOp::Compare {
                    comparison,
                    left,
                    right,
                },
                TraceType::Boolean,
            )
        };
        vec![
            arith(b, 2, ArithmeticOp::FloorDivide, min, minus_one),
            arith(b, 2, ArithmeticOp::Modulo, min, minus_one),
            arith(b, 2, ArithmeticOp::FloorDivide, seven, minus_three),
            arith(b, 2, ArithmeticOp::Modulo, seven, minus_three),
            arith(b, 2, ArithmeticOp::ShiftLeft, seven, sixty_four),
            arith(b, 2, ArithmeticOp::ShiftRight, min, sixty_four),
            arith(b, 2, ArithmeticOp::BitXor, max, min),
            to_float,
            less(b, Comparison::LessThan, float_two_53, two_53),
            less(b, Comparison::Equal, float_two_53, two_53),
            less(b, Comparison::LessThan, max, float_two_63),
            less(b, Comparison::Equal, to_float, two_53),
        ]
    });
    let (out, stats) = run_pass(&ir, fold_constants);
    // Every operation folded to a constant, and none of them rounds.
    assert!(stats.folded >= 12, "{stats:?}");
    let registers = exit_registers(&out);
    assert_eq!(registers, exit_registers(&ir));
    assert_eq!(registers[0], int(i64::MIN));
    assert_eq!(registers[1], int(0));
    assert_eq!(registers[2], int(-3));
    assert_eq!(registers[3], int(-2));
    assert_eq!(registers[4], int(0));
    assert_eq!(registers[5], int(0));
    assert_eq!(registers[6], int(-1));
    assert_eq!(registers[7], f(9_007_199_254_740_992.0));
    assert_eq!(registers[8], TraceScalar::Boolean(true));
    assert_eq!(registers[9], TraceScalar::Boolean(false));
    assert_eq!(registers[10], TraceScalar::Boolean(true));
    assert_eq!(registers[11], TraceScalar::Boolean(false));
}

#[test]
fn value_numbering_merges_integer_conversions_and_commutes_bitwise_operations() {
    let mut b = TraceBuilder::new(TraceId(1), root(4));
    let x = b.input(0, TraceType::Integer);
    let y = b.input(1, TraceType::Integer);
    let first = emit(
        &mut b,
        1,
        TraceOp::IntToFloat { source: x },
        TraceType::Float,
    );
    let second = emit(
        &mut b,
        2,
        TraceOp::IntToFloat { source: x },
        TraceType::Float,
    );
    let and = arith(&mut b, 3, ArithmeticOp::BitAnd, x, y);
    let swapped = arith(&mut b, 4, ArithmeticOp::BitAnd, y, x);
    // Shifts do not commute.
    let left = arith(&mut b, 5, ArithmeticOp::ShiftLeft, x, y);
    let right = arith(&mut b, 6, ArithmeticOp::ShiftLeft, y, x);
    for register in 0..4 {
        b.mark_written(FrameId(0), register);
    }
    let mut ir = b.finish(TraceTerminator::Exit {
        snapshot: SnapshotId(0),
    });
    ir.snapshots.push(Snapshot {
        id: SnapshotId(0),
        ..snap(
            SnapshotPhase::Before { pc: 400 },
            0,
            None,
            vec![
                RegisterEntry::Value(first),
                RegisterEntry::Value(second),
                RegisterEntry::Value(and),
                RegisterEntry::Value(swapped),
            ],
        )
    });
    // `left` and `right` are dead but must not merge with each other.
    let _ = (left, right);
    ir.verify().unwrap();
    let (out, stats) = run_pass(&ir, number_values);
    assert_eq!(stats.merged, 2, "{stats:?}");
    assert_eq!(
        count(&out, |op| matches!(op, TraceOp::IntToFloat { .. })),
        1
    );
    assert_eq!(
        count(&out, |op| matches!(
            op,
            TraceOp::Arithmetic {
                op: ArithmeticOp::BitAnd,
                ..
            }
        )),
        1
    );
    assert_eq!(
        count(&out, |op| matches!(
            op,
            TraceOp::Arithmetic {
                op: ArithmeticOp::ShiftLeft,
                ..
            }
        )),
        2
    );
}

#[test]
fn the_verifier_enforces_the_integer_and_float_typing_rules() {
    type Case<'a> = (&'a str, &'a dyn Fn(&mut TraceBuilder, ValueId, ValueId));
    let build = |build: &dyn Fn(&mut TraceBuilder, ValueId, ValueId)| {
        let mut b = TraceBuilder::new(TraceId(1), root(2));
        let integer = b.input(0, TraceType::Integer);
        let float = b.input(1, TraceType::Float);
        build(&mut b, integer, float);
        let mut ir = b.finish(TraceTerminator::Exit {
            snapshot: SnapshotId(0),
        });
        ir.snapshots.push(Snapshot {
            id: SnapshotId(0),
            ..snap(
                SnapshotPhase::Before { pc: 400 },
                0,
                None,
                vec![RegisterEntry::Unchanged, RegisterEntry::Unchanged],
            )
        });
        ir.verify()
    };
    // Legal: conversion, mixed ordering and equality, float floor division,
    // integer floor division, modulo and shifts.
    build(&|b, integer, float| {
        emit(
            b,
            1,
            TraceOp::IntToFloat { source: integer },
            TraceType::Float,
        );
        for comparison in [
            Comparison::Equal,
            Comparison::LessThan,
            Comparison::LessEqual,
        ] {
            for (left, right) in [(integer, float), (float, integer)] {
                emit(
                    b,
                    1,
                    TraceOp::Compare {
                        comparison,
                        left,
                        right,
                    },
                    TraceType::Boolean,
                );
            }
        }
        arith(b, 1, ArithmeticOp::FloorDivide, float, float);
        for op in [
            ArithmeticOp::FloorDivide,
            ArithmeticOp::Modulo,
            ArithmeticOp::ShiftLeft,
            ArithmeticOp::ShiftRight,
            ArithmeticOp::BitAnd,
            ArithmeticOp::BitOr,
            ArithmeticOp::BitXor,
        ] {
            arith(b, 1, op, integer, integer);
        }
    })
    .unwrap();
    // Illegal: each of these must be refused.
    let cases: [Case; 6] = [
        ("integer true division", &|b, integer, _| {
            emit(
                b,
                1,
                TraceOp::Arithmetic {
                    op: ArithmeticOp::Divide,
                    left: integer,
                    right: integer,
                },
                TraceType::Integer,
            );
        }),
        ("integer power", &|b, integer, _| {
            emit(
                b,
                1,
                TraceOp::Arithmetic {
                    op: ArithmeticOp::Power,
                    left: integer,
                    right: integer,
                },
                TraceType::Integer,
            );
        }),
        ("float bitwise and", &|b, _, float| {
            emit(
                b,
                1,
                TraceOp::Arithmetic {
                    op: ArithmeticOp::BitAnd,
                    left: float,
                    right: float,
                },
                TraceType::Float,
            );
        }),
        ("float shift", &|b, _, float| {
            emit(
                b,
                1,
                TraceOp::Arithmetic {
                    op: ArithmeticOp::ShiftLeft,
                    left: float,
                    right: float,
                },
                TraceType::Float,
            );
        }),
        (
            "mixed arithmetic without a conversion",
            &|b, integer, float| {
                emit(
                    b,
                    1,
                    TraceOp::Arithmetic {
                        op: ArithmeticOp::Add,
                        left: integer,
                        right: float,
                    },
                    TraceType::Float,
                );
            },
        ),
        ("converting a float", &|b, _, float| {
            emit(
                b,
                1,
                TraceOp::IntToFloat { source: float },
                TraceType::Float,
            );
        }),
    ];
    for (name, case) in cases {
        assert!(
            matches!(build(case), Err(TraceVerifyError::TypeMismatch { .. })),
            "{name}"
        );
    }
}

#[test]
fn folding_turns_typed_truthiness_into_constants() {
    use TraceScalar::Boolean as B;
    let mut b = TraceBuilder::new(TraceId(1), root(3));
    let n = b.input(0, TraceType::Float);
    let flag = b.input(1, TraceType::Boolean);
    let nil = b.input(2, TraceType::Nil);
    let t1 = emit(&mut b, 1, TraceOp::Truthy { source: n }, TraceType::Boolean);
    let t2 = emit(
        &mut b,
        1,
        TraceOp::Truthy { source: nil },
        TraceType::Boolean,
    );
    let t3 = emit(
        &mut b,
        1,
        TraceOp::Truthy { source: flag },
        TraceType::Boolean,
    );
    let t4 = emit(&mut b, 1, TraceOp::Not { source: nil }, TraceType::Boolean);
    for r in 0..3 {
        b.mark_written(FrameId(0), r);
    }
    let mut ir = b.finish(TraceTerminator::Exit {
        snapshot: SnapshotId(0),
    });
    ir.snapshots.push(Snapshot {
        id: SnapshotId(0),
        ..snap(
            SnapshotPhase::Before { pc: 400 },
            0,
            None,
            vec![
                RegisterEntry::Value(t1),
                RegisterEntry::Value(t2),
                RegisterEntry::Value(t3),
            ],
        )
    });
    // t4 is dead; folding still handles it.
    let _ = t4;
    ir.verify().unwrap();
    let (out, _) = run_pass(&ir, fold_constants);
    let (out, _) = run_pass(&out, eliminate_dead_code);
    assert_eq!(count(&out, |op| matches!(op, TraceOp::Truthy { .. })), 0);
    let Outcome::Exited(restored) = evaluate(
        &out,
        &[f(f64::NAN), TraceScalar::Boolean(false), TraceScalar::Nil],
        &[],
        1,
        |_, _| None,
    )
    .outcome
    else {
        panic!()
    };
    assert_eq!(restored.registers, [B(true), B(false), B(false)]);
}

// ---- guards ----------------------------------------------------------------

/// pc 1: read, guard float; pc 2: second guard on the same value; pc 3: a
/// guard on the guard's result.
fn guarded_read() -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(1), root(2));
    let x = b.input(0, TraceType::Float);
    let y = b.input(1, TraceType::Float);
    b.mark_written(FrameId(0), 1);
    let raw = emit(
        &mut b,
        1,
        TraceOp::Read {
            label: 1,
            operands: vec![x],
            result: TraceType::Any,
        },
        TraceType::Any,
    );
    let guard = |b: &mut TraceBuilder, pc: usize, source: ValueId, ty: TraceType| {
        let s = b.snapshot(snap(
            SnapshotPhase::Before { pc },
            0,
            Some(0),
            vec![RegisterEntry::Unchanged, RegisterEntry::Value(y)],
        ));
        let g = b.guard(site(pc), s);
        emit(
            b,
            pc,
            TraceOp::GuardType {
                source,
                expected: ty,
                guard: g,
            },
            ty,
        )
    };
    let g1 = guard(&mut b, 2, raw, TraceType::Float);
    let g2 = guard(&mut b, 3, raw, TraceType::Float);
    let g3 = guard(&mut b, 4, g1, TraceType::Float);
    let sum = arith(&mut b, 5, ArithmeticOp::Add, g2, g3);
    let mut ir = b.finish(TraceTerminator::Exit {
        snapshot: SnapshotId(3),
    });
    ir.snapshots.push(Snapshot {
        id: SnapshotId(3),
        ..snap(
            SnapshotPhase::Before { pc: 400 },
            0,
            None,
            vec![RegisterEntry::Unchanged, RegisterEntry::Value(sum)],
        )
    });
    ir.verify().unwrap();
    ir
}

#[test]
fn redundant_type_guards_are_removed_and_their_snapshots_with_them() {
    let ir = guarded_read();
    assert_eq!(ir.guards.len(), 3);
    let (out, stats) = run_pass(&ir, eliminate_guards);
    assert_eq!(stats.guards_removed, 2);
    assert_eq!(out.guards.len(), 1);
    assert_eq!(out.guards[0].site.pc, 2, "the first guard is the one kept");
    assert_eq!(out.snapshots.len(), 2, "one guard snapshot and the exit");
    let run = |ir: &TraceIr, value| {
        evaluate(ir, &[f(0.0), f(0.0)], &[], 1, move |_, _| Some(value)).outcome
    };
    assert_eq!(run(&ir, f(2.0)), run(&out, f(2.0)));
    assert_eq!(run(&ir, int(2)), run(&out, int(2)));
}

#[test]
fn a_guard_on_a_typed_value_is_removed() {
    let mut b = TraceBuilder::new(TraceId(1), root(1));
    let x = b.input(0, TraceType::Float);
    b.mark_written(FrameId(0), 0);
    let s = b.snapshot(snap(
        SnapshotPhase::Before { pc: 1 },
        0,
        None,
        vec![RegisterEntry::Value(x)],
    ));
    let g = b.guard(site(1), s);
    let y = emit(
        &mut b,
        1,
        TraceOp::GuardType {
            source: x,
            expected: TraceType::Float,
            guard: g,
        },
        TraceType::Float,
    );
    let mut ir = b.finish(TraceTerminator::Exit {
        snapshot: SnapshotId(1),
    });
    ir.snapshots.push(Snapshot {
        id: SnapshotId(1),
        ..snap(
            SnapshotPhase::Before { pc: 400 },
            0,
            None,
            vec![RegisterEntry::Value(y)],
        )
    });
    ir.verify().unwrap();
    let (out, stats) = run_pass(&ir, eliminate_guards);
    assert_eq!(stats.guards_removed, 1);
    assert_eq!(out.guards, [] as [crate::jit_ir::trace::GuardInfo; 0]);
}

fn branch_trace(conditions: &[(bool, bool)]) -> TraceIr {
    // Registers: r0 float input. Each entry is (use the constant condition,
    // expected). A shared comparison is used for the others.
    let mut b = TraceBuilder::new(TraceId(1), root(1));
    let x = b.input(0, TraceType::Float);
    let zero = konst(&mut b, 1, f(0.0));
    let shared = emit(
        &mut b,
        1,
        TraceOp::Compare {
            comparison: Comparison::LessThan,
            left: zero,
            right: x,
        },
        TraceType::Boolean,
    );
    let truth = konst(&mut b, 1, TraceScalar::Boolean(true));
    for (index, (constant, expected)) in conditions.iter().enumerate() {
        let pc = 2 + index;
        let s = b.snapshot(snap(
            SnapshotPhase::Before { pc },
            0,
            Some(0),
            vec![RegisterEntry::Unchanged],
        ));
        let g = b.guard(site(pc), s);
        b.emit(
            site(pc),
            TraceOp::GuardBranch {
                condition: if *constant { truth } else { shared },
                expected: *expected,
                guard: g,
            },
            None,
        );
    }
    let exit = b.snapshot(snap(
        SnapshotPhase::Before { pc: 400 },
        0,
        None,
        vec![RegisterEntry::Unchanged],
    ));
    let ir = b.finish(TraceTerminator::Exit { snapshot: exit });
    ir.verify().unwrap();
    ir
}

#[test]
fn branch_guards_on_constants_and_repeats_are_removed() {
    // constant true expected true: gone. repeat of the shared condition: one
    // kept. constant true expected false: always fails, kept.
    let ir = branch_trace(&[(true, true), (false, true), (false, true), (true, false)]);
    let (out, stats) = run_pass(&ir, eliminate_guards);
    assert_eq!(stats.guards_removed, 2);
    let pcs: Vec<usize> = out.guards.iter().map(|g| g.site.pc).collect();
    assert_eq!(pcs, [3, 5]);
    // The opposite repeat is kept: it can never pass after the first.
    let ir = branch_trace(&[(false, true), (false, false)]);
    let (out, stats) = run_pass(&ir, eliminate_guards);
    assert_eq!(stats.guards_removed, 0);
    assert_eq!(out.guards.len(), 2);
    for x in [1.0, -1.0] {
        assert_eq!(
            evaluate(&ir, &[f(x)], &[], 1, |_, _| None).outcome,
            evaluate(&out, &[f(x)], &[], 1, |_, _| None).outcome
        );
    }
}

// ---- value numbering -------------------------------------------------------

#[test]
fn value_numbering_merges_pure_operations_by_exact_bits() {
    let mut b = TraceBuilder::new(TraceId(1), root(4));
    let x = b.input(0, TraceType::Float);
    let y = b.input(1, TraceType::Float);
    let i = b.input(2, TraceType::Integer);
    let j = b.input(3, TraceType::Integer);
    let a1 = arith(&mut b, 1, ArithmeticOp::Multiply, x, y);
    let a2 = arith(&mut b, 2, ArithmeticOp::Multiply, x, y);
    let swapped = arith(&mut b, 3, ArithmeticOp::Multiply, y, x);
    let c0 = konst(&mut b, 4, f(0.0));
    let n0 = konst(&mut b, 4, f(-0.0));
    let c0b = konst(&mut b, 5, f(0.0));
    let nan1 = konst(&mut b, 6, f(f64::from_bits(0x7ff8_0000_0000_0001)));
    let nan2 = konst(&mut b, 6, f(f64::from_bits(0x7ff8_0000_0000_0002)));
    let ij = arith(&mut b, 7, ArithmeticOp::Add, i, j);
    let ji = arith(&mut b, 8, ArithmeticOp::Add, j, i);
    let eq1 = emit(
        &mut b,
        9,
        TraceOp::Compare {
            comparison: Comparison::Equal,
            left: x,
            right: y,
        },
        TraceType::Boolean,
    );
    let eq2 = emit(
        &mut b,
        9,
        TraceOp::Compare {
            comparison: Comparison::Equal,
            left: y,
            right: x,
        },
        TraceType::Boolean,
    );
    let read1 = emit(
        &mut b,
        10,
        TraceOp::Read {
            label: 1,
            operands: vec![x],
            result: TraceType::Float,
        },
        TraceType::Float,
    );
    let read2 = emit(
        &mut b,
        11,
        TraceOp::Read {
            label: 1,
            operands: vec![x],
            result: TraceType::Float,
        },
        TraceType::Float,
    );
    let all = [
        a1, a2, swapped, c0, n0, c0b, nan1, nan2, ij, ji, eq1, eq2, read1, read2,
    ];
    let mut ir = b.finish(TraceTerminator::Exit {
        snapshot: SnapshotId(0),
    });
    ir.frames[0].register_count = 4;
    ir.snapshots.push(Snapshot {
        id: SnapshotId(0),
        ..snap(
            SnapshotPhase::Before { pc: 400 },
            0,
            None,
            vec![RegisterEntry::Unchanged; 4],
        )
    });
    // Keep every value alive through the exit by mapping to registers is not
    // possible (4 registers), so check merge counts on the instruction mix.
    let _ = all;
    ir.verify().unwrap();
    let (out, stats) = run_pass(&ir, number_values);
    // a2, c0b, ji, eq2 merge. swapped (float operands) does not. -0.0 and
    // 0.0 stay apart, so do the NaN payloads, and reads never merge.
    assert_eq!(stats.merged, 4);
    assert_eq!(count(&out, |op| matches!(op, TraceOp::Read { .. })), 2);
    assert_eq!(
        count(&out, |op| matches!(
            op,
            TraceOp::Constant(TraceScalar::Float(_))
        )),
        4
    );
}

// ---- dead code -------------------------------------------------------------

#[test]
fn dead_code_keeps_values_only_a_snapshot_reads() {
    // The guard's snapshot reads `kept` (a loop-carried-looking value), and
    // nothing else does. It must stay, while a truly dead value goes.
    let mut b = TraceBuilder::new(TraceId(1), root(2));
    let x = b.input(0, TraceType::Float);
    let x1 = b.input(1, TraceType::Float);
    b.mark_written(FrameId(0), 0);
    b.mark_written(FrameId(0), 1);
    let kept = arith(&mut b, 1, ArithmeticOp::Add, x, x);
    let dead = arith(&mut b, 1, ArithmeticOp::Multiply, x, x);
    let dead_const = konst(&mut b, 1, f(3.0));
    let snap_const = konst(&mut b, 1, f(9.0));
    let cond = emit(
        &mut b,
        2,
        TraceOp::Compare {
            comparison: Comparison::LessThan,
            left: x,
            right: x,
        },
        TraceType::Boolean,
    );
    let s = b.snapshot(snap(
        SnapshotPhase::Before { pc: 3 },
        0,
        Some(2),
        vec![RegisterEntry::Value(kept), RegisterEntry::Value(snap_const)],
    ));
    let g = b.guard(site(3), s);
    b.emit(
        site(3),
        TraceOp::GuardBranch {
            condition: cond,
            expected: true,
            guard: g,
        },
        None,
    );
    let exit = b.snapshot(snap(
        SnapshotPhase::Before { pc: 400 },
        0,
        None,
        vec![RegisterEntry::Value(x), RegisterEntry::Value(x1)],
    ));
    let ir = b.finish(TraceTerminator::Exit { snapshot: exit });
    ir.verify().unwrap();
    let _ = (dead, dead_const);
    let (out, stats) = run_pass(&ir, eliminate_dead_code);
    assert_eq!(
        count(&out, |op| matches!(op, TraceOp::Arithmetic { .. })),
        1
    );
    assert_eq!(
        count(&out, |op| matches!(op, TraceOp::Constant(_))),
        0,
        "constants a snapshot names ride in the snapshot"
    );
    assert!(stats.dead_removed >= 3);
    let Outcome::GuardFailed { restored, .. } =
        evaluate(&out, &[f(2.0), f(0.0)], &[], 1, |_, _| None).outcome
    else {
        panic!()
    };
    assert_eq!(restored.registers, [f(4.0), f(9.0)]);
}

#[test]
fn dead_code_keeps_reads_effects_and_guards() {
    let ir = guarded_read();
    let (out, _) = run_pass(&ir, eliminate_dead_code);
    assert_eq!(count(&out, |op| matches!(op, TraceOp::Read { .. })), 1);
    assert_eq!(out.guards.len(), 3);

    let mut b = TraceBuilder::new(TraceId(1), root(1));
    let x = b.input(0, TraceType::Float);
    b.mark_written(FrameId(0), 0);
    let e = b.next_effect();
    b.emit(
        site(1),
        TraceOp::Effect {
            effect: e,
            label: 7,
            operands: vec![x],
            result: Some(TraceType::Float),
        },
        Some(TraceType::Float),
    );
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 1,
            successor: 2,
        },
        1,
        Some(1),
        vec![RegisterEntry::Unchanged],
    ));
    let mut ir = b.finish(TraceTerminator::Exit { snapshot: exit });
    ir.written_registers.clear();
    ir.verify().unwrap();
    let (out, stats) = run_pass(&ir, eliminate_dead_code);
    assert_eq!(stats.dead_removed, 0);
    assert_eq!(count(&out, |op| matches!(op, TraceOp::Effect { .. })), 1);
}

// ---- loops and hoisting ----------------------------------------------------

/// A counting loop. r0 is a loop-carried sum, r1 and r2 are invariant inputs.
///
/// ```text
/// pc 1: t = r1 * r2          invariant
/// pc 2: u = t + 1.0          invariant
/// pc 3: sum = sum + u        carried
/// pc 4: c = r1 < r2          invariant condition
/// pc 5: guard c
/// pc 6: s = sum + sum        carried
/// ```
fn invariant_loop() -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(1), root(3));
    b.mark_written(FrameId(0), 0);
    let sum_in = b.input(0, TraceType::Float);
    let r1 = b.input(1, TraceType::Float);
    let r2 = b.input(2, TraceType::Float);
    let sum = b.loop_param(TraceType::Float, LoopParamInit::Input(sum_in));
    let t = arith(&mut b, 1, ArithmeticOp::Multiply, r1, r2);
    let one = konst(&mut b, 2, f(1.0));
    let u = arith(&mut b, 2, ArithmeticOp::Add, t, one);
    let sum2 = arith(&mut b, 3, ArithmeticOp::Add, sum, u);
    let c = emit(
        &mut b,
        4,
        TraceOp::Compare {
            comparison: Comparison::LessThan,
            left: r1,
            right: r2,
        },
        TraceType::Boolean,
    );
    let s = b.snapshot(snap(
        SnapshotPhase::Before { pc: 5 },
        0,
        Some(4),
        vec![
            RegisterEntry::Value(sum2),
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
        ],
    ));
    let g = b.guard(site(5), s);
    b.emit(
        site(5),
        TraceOp::GuardBranch {
            condition: c,
            expected: true,
            guard: g,
        },
        None,
    );
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 6,
            successor: 1,
        },
        0,
        Some(6),
        vec![
            RegisterEntry::Value(sum2),
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
        ],
    ));
    b.set_poll_snapshot(poll);
    let ir = b.finish(TraceTerminator::LoopBack { args: vec![sum2] });
    ir.verify().unwrap();
    ir
}

#[test]
fn invariant_operations_and_guards_move_into_the_preheader() {
    let ir = invariant_loop();
    let mut stats = OptimizeStats::default();
    let out = hoist_with(&ir, true, &mut stats);
    out.verify().unwrap();
    assert_eq!(stats.hoisted, 3, "multiply, add, compare");
    assert_eq!(stats.guards_hoisted, 1);
    assert_eq!(out.guards.len(), 1);
    assert_eq!(out.preheader.len(), 5, "constant 1.0 moves with its user");
    // The steady-state loop keeps only the carried add.
    assert_eq!(out.body.len(), 1);
    assert!(matches!(
        out.body[0].op,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            ..
        }
    ));
    assert_eq!(out.snapshots.len(), 2, "entry-state snapshot and poll");
    out.verify().unwrap();

    // Without guard hoisting the guard stays in the body.
    let mut stats = OptimizeStats::default();
    let out = hoist_with(&ir, false, &mut stats);
    assert_eq!(stats.guards_hoisted, 0);
    assert_eq!(
        count(&out, |op| matches!(op, TraceOp::GuardBranch { .. })),
        1
    );
    assert_eq!(out.body.len(), 2);
}

#[test]
fn a_guard_a_side_trace_continues_stays_in_the_body() {
    // The invariant guard would fail once, before the loop, where no side
    // trace starts, and a link brings the loop new values for what it tests.
    let ir = invariant_loop();
    let origin = ir.guards[0].origin;
    let options = OptimizeOptions {
        hoist_guards: true,
        ..OptimizeOptions::for_level(OptimizeLevel::Optimized)
    };
    let writes = TreeWrites::of_members([&ir]).with_continued_guards([(0, origin)]);
    let mut stats = OptimizeStats::default();
    let out = hoist_invariants(&ir, &options, &writes, &mut stats).unwrap_or_else(|| ir.clone());
    out.verify().unwrap();
    assert_eq!(stats.guards_hoisted, 0);
    assert_eq!(
        count(&out, |op| matches!(op, TraceOp::GuardBranch { .. })),
        1
    );
    assert!(
        out.body
            .iter()
            .any(|instruction| matches!(instruction.op, TraceOp::GuardBranch { .. })),
        "the guard is in the body"
    );
    // A guard of another origin, or of another member, is hoisted as before.
    for guards in [[(0, origin + 7)], [(1, origin)]] {
        let writes = TreeWrites::of_members([&ir]).with_continued_guards(guards);
        let mut stats = OptimizeStats::default();
        hoist_invariants(&ir, &options, &writes, &mut stats);
        assert_eq!(stats.guards_hoisted, 1);
    }
}

#[test]
fn a_hoisted_guard_of_a_loop_side_names_the_state_the_side_starts_from() {
    // A loop side in a tree starts from its parent's native state, which
    // the interpreter's registers do not hold: its hoisted guard has to
    // restore every input. Leaving them `Unchanged`, as a root may, sent a
    // Mineclonia `region_op` nest back to the interpreter with the previous
    // run's locals (shape.lua:359, a boolean in `ly_dom`).
    let ir = invariant_loop();
    let options = OptimizeOptions {
        hoist_guards: true,
        ..OptimizeOptions::for_level(OptimizeLevel::Optimized)
    };
    let hoisted_snapshot = |member: usize| {
        let writes = TreeWrites::of_members([&ir]).for_member(member);
        let mut stats = OptimizeStats::default();
        let out = hoist_invariants(&ir, &options, &writes, &mut stats).expect("it hoists");
        out.verify().unwrap();
        assert_eq!(stats.guards_hoisted, 1);
        let guard = out
            .preheader
            .iter()
            .find_map(|instruction| match instruction.op {
                TraceOp::GuardBranch { guard, .. } => Some(guard),
                _ => None,
            })
            .expect("the guard is in the preheader");
        let snapshot = out.guards[guard.index()].snapshot;
        out.snapshots[snapshot.index()].frames[0].registers.clone()
    };
    // The root's entry state is the interpreter's.
    assert_eq!(hoisted_snapshot(0), vec![RegisterEntry::Unchanged; 3]);
    // A side's is not.
    let inputs: Vec<RegisterEntry> = ir
        .entry
        .inputs
        .iter()
        .map(|input| RegisterEntry::Value(input.value))
        .collect();
    assert_eq!(hoisted_snapshot(1), inputs);
}

#[test]
fn a_side_that_starts_inside_calls_hoists_no_guard() {
    let mut ir = invariant_loop();
    ir.seed_frames = 1;
    let options = OptimizeOptions {
        hoist_guards: true,
        ..OptimizeOptions::for_level(OptimizeLevel::Optimized)
    };
    let writes = TreeWrites::of_members([&ir]).for_member(1);
    let mut stats = OptimizeStats::default();
    hoist_invariants(&ir, &options, &writes, &mut stats);
    assert_eq!(stats.guards_hoisted, 0);
}

#[test]
fn loop_carried_values_and_reads_are_not_hoisted() {
    let mut b = TraceBuilder::new(TraceId(1), root(2));
    b.mark_written(FrameId(0), 0);
    let sum_in = b.input(0, TraceType::Float);
    let r1 = b.input(1, TraceType::Float);
    let sum = b.loop_param(TraceType::Float, LoopParamInit::Input(sum_in));
    // Reads an invariant operand, but it is a heap read.
    let load = emit(
        &mut b,
        1,
        TraceOp::Read {
            label: 1,
            operands: vec![r1],
            result: TraceType::Float,
        },
        TraceType::Float,
    );
    let from_load = arith(&mut b, 2, ArithmeticOp::Multiply, load, r1);
    let from_sum = arith(&mut b, 3, ArithmeticOp::Multiply, sum, r1);
    let total = arith(&mut b, 4, ArithmeticOp::Add, from_load, from_sum);
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 4,
            successor: 1,
        },
        0,
        Some(4),
        vec![RegisterEntry::Value(total), RegisterEntry::Unchanged],
    ));
    b.set_poll_snapshot(poll);
    let ir = b.finish(TraceTerminator::LoopBack { args: vec![total] });
    ir.verify().unwrap();
    let mut stats = OptimizeStats::default();
    let out = hoist_with(&ir, true, &mut stats);
    assert_eq!(stats.hoisted, 0);
    assert_eq!(
        out.preheader,
        [] as [crate::jit_ir::trace::TraceInstruction; 0]
    );
    assert_eq!(out, ir);
}

#[test]
fn a_failing_hoisted_guard_restores_the_entry_state_before_any_iteration() {
    let ir = invariant_loop();
    let (out, _) = optimize_trace(&ir, OptimizeLevel::Optimized);
    assert_ne!(
        out.preheader,
        [] as [crate::jit_ir::trace::TraceInstruction; 0]
    );
    // r1 >= r2: the guard fails. The original runs the first iteration up to
    // the guard first, then exits at pc 5. The hoisted guard exits at the
    // header with the registers untouched.
    let entry = [f(10.0), f(3.0), f(2.0)];
    let before = evaluate(&ir, &entry, &[], 5, |_, _| None);
    let after = evaluate(&out, &entry, &[], 5, |_, _| None);
    let Outcome::GuardFailed {
        restored: original, ..
    } = before.outcome
    else {
        panic!()
    };
    assert_eq!(original.pc, 5);
    assert_eq!(original.registers[0], f(10.0 + 7.0));
    let Outcome::GuardFailed { restored, guard } = after.outcome else {
        panic!()
    };
    assert_eq!(guard, GuardId(0));
    assert_eq!(restored.pc, 1, "the loop header");
    assert_eq!(restored.registers, entry);
    assert_eq!(restored.last_interpreted_pc, Some(6));
    assert_eq!(after.iterations, 0);
    assert_eq!(
        after.effects,
        [] as [crate::jit_ir::trace::reference::EffectRecord; 0]
    );

    // When the guard passes, both agree for any number of iterations.
    let entry = [f(10.0), f(2.0), f(3.0)];
    for iterations in 1..6 {
        let a = evaluate(&ir, &entry, &[], iterations, |_, _| None);
        let b = evaluate(&out, &entry, &[], iterations, |_, _| None);
        assert_eq!(a.outcome, b.outcome);
        assert_eq!(a.iterations, b.iterations);
    }
}

#[test]
fn verifier_rejects_a_preheader_guard_with_a_state_dependent_snapshot() {
    let ir = invariant_loop();
    let (mut out, _) = optimize_trace(&ir, OptimizeLevel::Optimized);
    // Point the hoisted guard at a snapshot that is not the entry state.
    let snapshot = &mut out.snapshots[out.guards[0].snapshot.index()];
    snapshot.frames[0]
        .registers
        .set(1, RegisterEntry::Const(f(1.0)));
    assert!(out.verify().is_err());
}

#[test]
fn verifier_rejects_a_preheader_guard_snapshot_that_resumes_mid_body() {
    let ir = invariant_loop();
    let (mut out, _) = optimize_trace(&ir, OptimizeLevel::Optimized);
    assert!(out.verify().is_ok());
    let snapshot = &mut out.snapshots[out.guards[0].snapshot.index()];
    snapshot.phase = SnapshotPhase::Before { pc: 3 };
    assert!(out.verify().is_err(), "wrong pc");
    let (mut out, _) = optimize_trace(&ir, OptimizeLevel::Optimized);
    let snapshot = &mut out.snapshots[out.guards[0].snapshot.index()];
    snapshot.frames[0].last_interpreted_pc = None;
    assert!(out.verify().is_err(), "wrong last interpreted pc");
}

#[test]
fn verifier_rejects_effects_in_the_preheader() {
    let ir = invariant_loop();
    let (mut out, _) = optimize_trace(&ir, OptimizeLevel::Optimized);
    out.preheader.push(TraceInstruction {
        site: site(1),
        result: None,
        op: TraceOp::Effect {
            effect: EffectId(0),
            label: 1,
            operands: vec![],
            result: None,
        },
    });
    assert!(out.verify().is_err());
}

/// An unrolled inner loop runs one guest instruction twice, with the check of
/// another pc between the runs. Dead code removes that check, the two effects
/// at one pc then touch, and the full verifier reads them as one guest step:
/// the second guard's `Before` snapshot looks like a replay of the first
/// effect. The optimizer's result is right, so it is kept, and it passes the
/// verifier for optimized traces. (Invariant code motion and value numbering
/// do the same with a hoisted or merged constant; profile load of such a
/// trace used to panic in debug builds.)
#[test]
fn an_optimized_trace_may_put_two_runs_of_one_effectful_pc_side_by_side() {
    let mut b = TraceBuilder::new(TraceId(1), root(2));
    let x = b.input(0, TraceType::Float);
    b.mark_written(FrameId(0), 0);
    let first = b.next_effect();
    b.emit(
        site(3),
        TraceOp::Effect {
            effect: first,
            label: 1,
            operands: vec![x],
            result: None,
        },
        None,
    );
    // The boundary of the next execution of pc 3, unused.
    konst(&mut b, 4, f(1.0));
    let cond = emit(
        &mut b,
        3,
        TraceOp::Compare {
            comparison: Comparison::LessThan,
            left: x,
            right: x,
        },
        TraceType::Boolean,
    );
    let before = b.snapshot(snap(
        SnapshotPhase::Before { pc: 3 },
        1,
        Some(2),
        vec![RegisterEntry::Unchanged, RegisterEntry::Unchanged],
    ));
    let guard = b.guard(site(3), before);
    b.emit(
        site(3),
        TraceOp::GuardBranch {
            condition: cond,
            expected: false,
            guard,
        },
        None,
    );
    let second = b.next_effect();
    b.emit(
        site(3),
        TraceOp::Effect {
            effect: second,
            label: 1,
            operands: vec![x],
            result: None,
        },
        None,
    );
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 3,
            successor: 4,
        },
        2,
        Some(3),
        vec![RegisterEntry::Unchanged, RegisterEntry::Unchanged],
    ));
    let mut ir = b.finish(TraceTerminator::Exit { snapshot: exit });
    ir.written_registers.clear();
    ir.verify().expect("the recorded trace has its boundary");

    let (out, stats) = optimize_trace(&ir, OptimizeLevel::Optimized);
    assert_eq!(stats.dead_removed, 1, "the boundary constant is dead");
    assert_eq!(count(&out, |op| matches!(op, TraceOp::Constant(_))), 0);
    out.verify_optimized().unwrap();
    assert!(
        matches!(
            out.verify(),
            Err(TraceVerifyError::EffectReplay {
                instruction: Some(_),
                ..
            })
        ),
        "the full verifier cannot tell the two executions apart"
    );
    // The other effect checks still hold for it: a guard that moves across an
    // effect has the wrong effect position.
    let mut moved = out.clone();
    moved.snapshots[0].effect_position = 0;
    assert!(moved.verify_optimized().is_err());
}

// ---- pipeline --------------------------------------------------------------

#[test]
fn optimized_tier_cleans_a_recorded_looking_trace() {
    // Moves, a repeated constant, a repeated product and an unused value.
    let mut b = TraceBuilder::new(TraceId(1), root(2));
    b.mark_written(FrameId(0), 0);
    b.mark_written(FrameId(0), 1);
    let sum_in = b.input(0, TraceType::Float);
    let x_in = b.input(1, TraceType::Float);
    let sum = b.loop_param(TraceType::Float, LoopParamInit::Input(sum_in));
    let x = b.loop_param(TraceType::Float, LoopParamInit::Input(x_in));
    let m = emit(&mut b, 1, TraceOp::Move { source: x }, TraceType::Float);
    let k1 = konst(&mut b, 2, f(2.0));
    let k2 = konst(&mut b, 3, f(2.0));
    let p1 = arith(&mut b, 4, ArithmeticOp::Multiply, m, k1);
    let p2 = arith(&mut b, 5, ArithmeticOp::Multiply, x, k2);
    let _unused = arith(&mut b, 6, ArithmeticOp::Divide, p1, p2);
    let total = arith(&mut b, 7, ArithmeticOp::Add, sum, p2);
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 7,
            successor: 1,
        },
        0,
        Some(7),
        vec![RegisterEntry::Value(total), RegisterEntry::Value(x)],
    ));
    b.set_poll_snapshot(poll);
    let ir = b.finish(TraceTerminator::LoopBack {
        args: vec![total, x],
    });
    ir.verify().unwrap();
    let (out, stats) = optimize_trace(&ir, OptimizeLevel::Optimized);
    out.verify().unwrap();
    assert!(stats.copies_removed >= 1 && stats.merged >= 2 && stats.dead_removed >= 1);
    assert_eq!(
        count(&out, |op| matches!(op, TraceOp::Arithmetic { .. })),
        2
    );
    for iterations in 1..5 {
        let entry = [f(1.5), f(-4.25)];
        let a = evaluate(&ir, &entry, &[], iterations, |_, _| None);
        let b = evaluate(&out, &entry, &[], iterations, |_, _| None);
        assert_eq!(a.outcome, b.outcome);
    }
    // Baseline keeps folds and merges out of reach but still verifies.
    let (baseline, baseline_stats) = optimize_trace(&ir, OptimizeLevel::Baseline);
    baseline.verify().unwrap();
    assert_eq!((baseline_stats.merged, baseline_stats.folded), (0, 0));
    assert!(baseline_stats.dead_removed >= 2 && baseline_stats.copies_removed == 1);
}

#[test]
fn optimizing_twice_changes_nothing_more() {
    let ir = invariant_loop();
    let (once, _) = optimize_trace(&ir, OptimizeLevel::Optimized);
    let (twice, stats) = optimize_trace(&once, OptimizeLevel::Optimized);
    assert_eq!(once, twice);
    assert_eq!(stats.total(), 0);
}

// ---- generated traces ------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() >> 33).unwrap() % bound
    }

    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }
}

const SPECIAL_FLOATS: [f64; 14] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    2.0,
    -2.5,
    3.0,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    1e308,
    1e-310,
    -7.25,
];
const SPECIAL_INTS: [i64; 12] = [
    0,
    1,
    -1,
    2,
    7,
    i64::MAX,
    i64::MIN,
    1 << 40,
    64,
    -64,
    9_007_199_254_740_993,
    -3,
];
const REGISTERS: usize = 6;
const READ_ANY: u32 = 1;
const READ_FLOAT: u32 = 2;
const EFFECT_NONE: u32 = 3;
const EFFECT_FLOAT: u32 = 4;

fn random_scalar(rng: &mut Rng, ty: TraceType) -> TraceScalar {
    match ty {
        TraceType::Float => f(rng.pick(&SPECIAL_FLOATS)),
        TraceType::Integer => int(rng.pick(&SPECIAL_INTS)),
        TraceType::Boolean => TraceScalar::Boolean(rng.chance(50)),
        TraceType::Table => TraceScalar::Table(rng.next() % 3),
        TraceType::Closure => TraceScalar::Closure(rng.next() % 3),
        TraceType::Thread => TraceScalar::Thread(rng.next() % 3),
        TraceType::Userdata => TraceScalar::Userdata(rng.next() % 3),
        TraceType::String => TraceScalar::String(rng.next() % 3),
        TraceType::Int64 => TraceScalar::Int64(rng.pick(&SPECIAL_INTS)),
        TraceType::UInt64 => TraceScalar::UInt64(rng.pick(&SPECIAL_INTS).cast_unsigned()),
        // A borrow is never generated: only a table read makes one.
        TraceType::Nil | TraceType::Any | TraceType::StringRef => TraceScalar::Nil,
    }
}

struct Generated {
    ir: TraceIr,
    types: Vec<TraceType>,
}

/// Builds a random, verified trace. It mimics the recorder: every register
/// is an entry input, written registers are loop parameters, and every guard
/// snapshot maps each register to its current value.
fn generate(seed: u64) -> Generated {
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    for _ in 0..4 {
        rng.next();
    }
    let looping = rng.chance(80);
    let types: Vec<TraceType> = (0..REGISTERS)
        .map(|_| match rng.below(10) {
            0 => TraceType::Integer,
            1 => TraceType::Boolean,
            2 if rng.chance(30) => TraceType::Nil,
            2 | 3 => TraceType::Integer,
            _ => TraceType::Float,
        })
        .collect();
    let mut b = TraceBuilder::new(
        TraceId(1),
        TraceFrameInfo {
            register_count: u16::try_from(REGISTERS).unwrap(),
            code_len: 512,
            inlined: None,
            activation: None,
        },
    );
    let written: Vec<bool> = (0..REGISTERS).map(|_| rng.chance(60)).collect();
    for (register, w) in written.iter().enumerate() {
        if *w {
            b.mark_written(FrameId(0), u16::try_from(register).unwrap());
        }
    }
    let inputs: Vec<ValueId> = types
        .iter()
        .enumerate()
        .map(|(register, ty)| b.input(u16::try_from(register).unwrap(), *ty))
        .collect();
    let mut current = inputs.clone();
    let mut params = Vec::new();
    if looping {
        for register in 0..REGISTERS {
            if written[register] {
                let param = b.loop_param(types[register], LoopParamInit::Input(inputs[register]));
                current[register] = param;
                params.push(register);
            }
        }
    }
    let mut pool: Vec<(ValueId, TraceType)> = Vec::new();
    let mut conditions: Vec<ValueId> = Vec::new();
    let mut anys: Vec<ValueId> = Vec::new();
    let mut pc = 0_usize;
    let mut last_pc = 0_usize;
    let steps = 6 + rng.below(36);

    macro_rules! values {
        () => {{
            let mut all: Vec<(ValueId, TraceType)> = pool.clone();
            for register in 0..REGISTERS {
                all.push((current[register], types[register]));
            }
            all
        }};
    }
    let guard_snapshot =
        |b: &mut TraceBuilder, rng: &mut Rng, current: &[ValueId], pc: usize, last: usize| {
            let registers = (0..REGISTERS)
                .map(|register| {
                    if !written[register] && rng.chance(50) {
                        RegisterEntry::Unchanged
                    } else if types[register] == TraceType::Nil && rng.chance(50) {
                        RegisterEntry::Const(TraceScalar::Nil)
                    } else {
                        RegisterEntry::Value(current[register])
                    }
                })
                .collect();
            b.snapshot(snap(
                SnapshotPhase::Before { pc },
                b.effect_position(),
                Some(last),
                registers,
            ))
        };

    for _ in 0..steps {
        pc += 1;
        let all = values!();
        let numeric: Vec<(ValueId, TraceType)> = all
            .iter()
            .copied()
            .filter(|(_, ty)| matches!(ty, TraceType::Float | TraceType::Integer))
            .collect();
        let step_pc = pc;
        match rng.below(14) {
            0 | 1 => {
                let ty = rng.pick(&[
                    TraceType::Float,
                    TraceType::Float,
                    TraceType::Float,
                    TraceType::Integer,
                    TraceType::Boolean,
                ]);
                let scalar = random_scalar(&mut rng, ty);
                let v = emit(&mut b, step_pc, TraceOp::Constant(scalar), ty);
                pool.push((v, ty));
            }
            2..=5 if !numeric.is_empty() => {
                let (l, ty) = rng.pick(&numeric);
                let same: Vec<_> = numeric.iter().filter(|(_, t)| *t == ty).collect();
                let (r, _) = **same.get(rng.below(same.len())).unwrap();
                let op = if ty == TraceType::Float {
                    rng.pick(&[
                        ArithmeticOp::Add,
                        ArithmeticOp::Subtract,
                        ArithmeticOp::Multiply,
                        ArithmeticOp::Divide,
                        ArithmeticOp::Modulo,
                        ArithmeticOp::Power,
                        ArithmeticOp::FloorDivide,
                        ArithmeticOp::Add,
                        ArithmeticOp::Multiply,
                    ])
                } else {
                    rng.pick(&[
                        ArithmeticOp::Add,
                        ArithmeticOp::Subtract,
                        ArithmeticOp::Multiply,
                        ArithmeticOp::Modulo,
                        ArithmeticOp::FloorDivide,
                        ArithmeticOp::BitAnd,
                        ArithmeticOp::BitOr,
                        ArithmeticOp::BitXor,
                        ArithmeticOp::ShiftLeft,
                        ArithmeticOp::ShiftRight,
                    ])
                };
                let v = emit(
                    &mut b,
                    step_pc,
                    TraceOp::Arithmetic {
                        op,
                        left: l,
                        right: r,
                    },
                    ty,
                );
                pool.push((v, ty));
            }
            6 if !numeric.is_empty() => {
                let (s, ty) = rng.pick(&numeric);
                if ty == TraceType::Integer && rng.chance(50) {
                    let v = emit(
                        &mut b,
                        step_pc,
                        TraceOp::IntToFloat { source: s },
                        TraceType::Float,
                    );
                    pool.push((v, TraceType::Float));
                } else {
                    let v = emit(&mut b, step_pc, TraceOp::Negate { source: s }, ty);
                    pool.push((v, ty));
                }
            }
            7 | 8 => {
                let (l, ty) = rng.pick(&all);
                let same: Vec<_> = all.iter().filter(|(_, t)| *t == ty).collect();
                let mut r = same.get(rng.below(same.len())).unwrap().0;
                // An integer against a float compares exactly.
                if matches!(ty, TraceType::Float | TraceType::Integer) && rng.chance(30) {
                    r = rng.pick(&numeric).0;
                }
                let comparison = if matches!(ty, TraceType::Float | TraceType::Integer) {
                    rng.pick(&[
                        Comparison::Equal,
                        Comparison::LessThan,
                        Comparison::LessEqual,
                    ])
                } else {
                    Comparison::Equal
                };
                let v = emit(
                    &mut b,
                    step_pc,
                    TraceOp::Compare {
                        comparison,
                        left: l,
                        right: r,
                    },
                    TraceType::Boolean,
                );
                pool.push((v, TraceType::Boolean));
                conditions.push(v);
            }
            9 => {
                let (s, _) = rng.pick(&all);
                let op = if rng.chance(50) {
                    TraceOp::Truthy { source: s }
                } else {
                    TraceOp::Not { source: s }
                };
                let v = emit(&mut b, step_pc, op, TraceType::Boolean);
                pool.push((v, TraceType::Boolean));
                conditions.push(v);
            }
            10 => {
                let (s, ty) = rng.pick(&all);
                let v = emit(&mut b, step_pc, TraceOp::Move { source: s }, ty);
                pool.push((v, ty));
            }
            11 => {
                // A read, then (usually) a guard that narrows it.
                let operands: Vec<ValueId> = (0..rng.below(3)).map(|_| rng.pick(&all).0).collect();
                if rng.chance(50) {
                    let v = emit(
                        &mut b,
                        step_pc,
                        TraceOp::Read {
                            label: READ_FLOAT,
                            operands,
                            result: TraceType::Float,
                        },
                        TraceType::Float,
                    );
                    pool.push((v, TraceType::Float));
                } else {
                    let v = emit(
                        &mut b,
                        step_pc,
                        TraceOp::Read {
                            label: READ_ANY,
                            operands,
                            result: TraceType::Any,
                        },
                        TraceType::Any,
                    );
                    anys.push(v);
                }
            }
            12 if !anys.is_empty() || !conditions.is_empty() => {
                // A guard. Type guards reuse earlier Any values often, so
                // repeats occur. Branch guards reuse earlier conditions.
                let use_type_guard = !anys.is_empty() && (conditions.is_empty() || rng.chance(60));
                let snapshot = guard_snapshot(&mut b, &mut rng, &current, step_pc, last_pc);
                let guard = b.guard(site(step_pc), snapshot);
                if use_type_guard {
                    let source = rng.pick(&anys);
                    let expected = rng.pick(&[
                        TraceType::Float,
                        TraceType::Float,
                        TraceType::Float,
                        TraceType::Integer,
                    ]);
                    let v = emit(
                        &mut b,
                        step_pc,
                        TraceOp::GuardType {
                            source,
                            expected,
                            guard,
                        },
                        expected,
                    );
                    pool.push((v, expected));
                } else {
                    let condition = rng.pick(&conditions);
                    b.emit(
                        site(step_pc),
                        TraceOp::GuardBranch {
                            condition,
                            expected: rng.chance(50),
                            guard,
                        },
                        None,
                    );
                }
            }
            13 => {
                let operands: Vec<ValueId> = (0..rng.below(3)).map(|_| rng.pick(&all).0).collect();
                let effect = b.next_effect();
                let with_result = rng.chance(30);
                let v = b.emit(
                    site(step_pc),
                    TraceOp::Effect {
                        effect,
                        label: if with_result {
                            EFFECT_FLOAT
                        } else {
                            EFFECT_NONE
                        },
                        operands,
                        result: with_result.then_some(TraceType::Float),
                    },
                    with_result.then_some(TraceType::Float),
                );
                if let Some(v) = v {
                    pool.push((v, TraceType::Float));
                }
            }
            _ => {
                // Assign a pool value to a written register of its type.
                let (v, ty) = rng.pick(&all);
                let targets: Vec<usize> = (0..REGISTERS)
                    .filter(|r| written[*r] && types[*r] == ty)
                    .collect();
                if !targets.is_empty() {
                    current[targets[rng.below(targets.len())]] = v;
                }
            }
        }
        last_pc = step_pc;
        // Sometimes assign what was just computed.
        if rng.chance(35)
            && let Some((v, ty)) = pool.last().copied()
        {
            let targets: Vec<usize> = (0..REGISTERS)
                .filter(|r| written[*r] && types[*r] == ty)
                .collect();
            if !targets.is_empty() {
                current[targets[rng.below(targets.len())]] = v;
            }
        }
    }
    pc += 1;
    let close = pc;
    let final_registers = |rng: &mut Rng, current: &[ValueId]| -> Vec<RegisterEntry> {
        (0..REGISTERS)
            .map(|register| {
                if !written[register] && rng.chance(50) {
                    RegisterEntry::Unchanged
                } else {
                    RegisterEntry::Value(current[register])
                }
            })
            .collect()
    };
    let ir = if looping {
        let poll = b.snapshot(snap(
            SnapshotPhase::After {
                pc: close,
                successor: 0,
            },
            b.effect_position(),
            Some(close),
            final_registers(&mut rng, &current)
                .into_iter()
                .enumerate()
                .map(|(register, entry)| {
                    if written[register] {
                        RegisterEntry::Value(current[register])
                    } else {
                        entry
                    }
                })
                .collect(),
        ));
        b.set_poll_snapshot(poll);
        let args = params.iter().map(|r| current[*r]).collect();
        b.finish(TraceTerminator::LoopBack { args })
    } else {
        let exit = b.snapshot(snap(
            SnapshotPhase::After {
                pc: close,
                successor: close + 1,
            },
            b.effect_position(),
            Some(close),
            final_registers(&mut rng, &current),
        ));
        b.finish(TraceTerminator::Exit { snapshot: exit })
    };
    ir.verify().unwrap_or_else(|error| {
        panic!("seed {seed}: generated an invalid trace: {error}\n{ir:#?}")
    });
    Generated { ir, types }
}

fn entry_state(rng: &mut Rng, types: &[TraceType]) -> Vec<TraceScalar> {
    types.iter().map(|ty| random_scalar(rng, *ty)).collect()
}

/// A deterministic heap: results depend on the label and the operands only,
/// so two traces that issue the same reads see the same values.
fn heap(record: &super::super::reference::EffectRecord, _committed: u32) -> Option<TraceScalar> {
    let mut hash = u64::from(record.label).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    for operand in &record.operands {
        let bits = match operand {
            TraceScalar::Nil => 1,
            TraceScalar::Boolean(flag) => 2 + u64::from(*flag),
            TraceScalar::Integer(value) => value.cast_unsigned(),
            TraceScalar::Float(value) => value.to_bits(),
            TraceScalar::Int64(value) => value.cast_unsigned() ^ 0x0069_3634,
            TraceScalar::UInt64(value) => *value ^ 0x0075_3634,
            TraceScalar::Table(lane) => 0x0074_6162_6c65 ^ lane,
            TraceScalar::Closure(lane) => 0x0063_6c6f_7375 ^ lane,
            TraceScalar::Thread(lane) => 0x0074_6872_6561 ^ lane,
            TraceScalar::Userdata(lane) => 0x0075_7365_7264 ^ lane,
            TraceScalar::String(lane) => 0x0073_7472_696e ^ lane,
            TraceScalar::StringRef(bits) => 0x0073_7472_7265 ^ bits,
        };
        hash = (hash ^ bits)
            .wrapping_mul(0x2545_f491_4f6c_dd1d)
            .rotate_left(17);
    }
    Some(match record.label {
        READ_ANY => match hash % 7 {
            0 => int(i64::try_from(hash >> 40).unwrap()),
            1 => TraceScalar::Nil,
            _ => f(SPECIAL_FLOATS[usize::try_from(hash >> 8).unwrap() % SPECIAL_FLOATS.len()]),
        },
        _ => f(SPECIAL_FLOATS[usize::try_from(hash >> 8).unwrap() % SPECIAL_FLOATS.len()]),
    })
}

/// An outcome with the failing guard named by its site, since passes
/// renumber guards.
fn keyed(ir: &TraceIr, outcome: &Outcome) -> (Option<TraceSite>, Outcome) {
    match outcome {
        Outcome::GuardFailed { guard, restored } => (
            Some(ir.guards[guard.index()].site),
            Outcome::GuardFailed {
                guard: GuardId(0),
                restored: restored.clone(),
            },
        ),
        other => (None, other.clone()),
    }
}

fn same(label: &str, original: (&TraceIr, &Evaluation), out: (&TraceIr, &Evaluation)) {
    assert_eq!(
        keyed(original.0, &original.1.outcome),
        keyed(out.0, &out.1.outcome),
        "{label}: outcome"
    );
    assert_eq!(original.1.effects, out.1.effects, "{label}: effects");
    assert_eq!(
        original.1.iterations, out.1.iterations,
        "{label}: iterations"
    );
}

const SEEDS: u64 = 1500;

#[test]
fn generated_traces_verify_and_exercise_every_construct() {
    let mut loops = 0;
    let mut guards = 0;
    let mut effects = 0;
    let mut reads = 0;
    for seed in 0..SEEDS {
        let ir = generate(seed).ir;
        loops += usize::from(matches!(ir.terminator, TraceTerminator::LoopBack { .. }));
        guards += ir.guards.len();
        effects += ir.effect_count() as usize;
        reads += count(&ir, |op| matches!(op, TraceOp::Read { .. }));
    }
    assert!(loops > 1000 && guards > 1500 && effects > 1500 && reads > 1500);
}

fn optimized_variants(ir: &TraceIr) -> Vec<(&'static str, TraceIr, OptimizeStats)> {
    [
        (
            "baseline",
            OptimizeOptions::for_level(OptimizeLevel::Baseline),
        ),
        (
            "optimized without guard hoisting",
            OptimizeOptions {
                level: OptimizeLevel::Optimized,
                hoist_guards: false,
                memory: true,
            },
        ),
        (
            "optimized without memory passes",
            OptimizeOptions {
                hoist_guards: false,
                memory: false,
                ..OptimizeOptions::for_level(OptimizeLevel::Optimized)
            },
        ),
    ]
    .into_iter()
    .map(|(name, options)| {
        let (out, stats) = optimize_with(ir, options);
        (name, out, stats)
    })
    .collect()
}

#[test]
fn optimized_traces_evaluate_like_the_originals() {
    let mut changed = 0;
    for seed in 0..SEEDS {
        let generated = generate(seed);
        let mut rng = Rng(seed ^ 0xabcd_ef01 | 1);
        for (name, out, stats) in optimized_variants(&generated.ir) {
            changed += usize::from(stats.total() > 0);
            out.verify()
                .unwrap_or_else(|e| panic!("seed {seed} {name}: {e}"));
            for _ in 0..6 {
                let entry = entry_state(&mut rng, &generated.types);
                let label = format!("seed {seed} {name} entry {entry:?}");
                let a = evaluate(&generated.ir, &entry, &[], 4, heap);
                let b = evaluate(&out, &entry, &[], 4, heap);
                same(&label, (&generated.ir, &a), (&out, &b));
            }
        }
    }
    assert!(changed > SEEDS as usize, "the optimizer actually ran");
}

#[test]
fn optimized_traces_match_under_a_forced_exit_at_every_guard() {
    let mut forced = 0_usize;
    for seed in 0..SEEDS {
        let generated = generate(seed);
        let mut rng = Rng(seed ^ 0x1357_9bdf | 1);
        let mut variants = optimized_variants(&generated.ir);
        let (hoisting, stats) = optimize_trace(&generated.ir, OptimizeLevel::Optimized);
        variants.push(("optimized", hoisting, stats));
        for (name, out, _) in variants {
            // Every guard the optimized trace still has, and the original's
            // guards at the same sites, fail at the first iterations. A guard
            // hoisted into the preheader exits with the entry state and runs
            // once, so it has no counterpart at later iterations. It is
            // checked by `guard_hoisting_only_moves_exits_to_the_entry_state`.
            let hoisted: Vec<usize> = out
                .guards
                .iter()
                .take(
                    out.preheader
                        .iter()
                        .filter(|instruction| instruction.op.guard().is_some())
                        .count(),
                )
                .map(|g| g.site.pc)
                .collect();
            let sites: Vec<usize> = out
                .guards
                .iter()
                .map(|g| g.site.pc)
                .filter(|pc| !hoisted.contains(pc))
                .collect();
            let entry = entry_state(&mut rng, &generated.types);
            for pc in sites {
                for iteration in 0..3 {
                    let force = Some(ForcedExit {
                        frame: FrameId(0),
                        pc,
                        guard: None,
                        iteration,
                    });
                    let label = format!("seed {seed} {name} forced pc {pc} iteration {iteration}");
                    let a = evaluate_forced(&generated.ir, &entry, &[], 5, force, heap);
                    let b = evaluate_forced(&out, &entry, &[], 5, force, heap);
                    if let Outcome::GuardFailed { guard, restored } = &b.outcome
                        && guard.index() < hoisted.len()
                    {
                        // A hoisted guard failed on the data. The state is
                        // the entry state, and the original left in its
                        // first iteration too.
                        assert_eq!(restored.registers, entry, "{label}");
                        assert_eq!(b.iterations, 0, "{label}");
                        assert!(
                            matches!(a.outcome, Outcome::GuardFailed { .. }) && a.iterations == 0,
                            "{label}: original {:?}",
                            a.outcome
                        );
                        continue;
                    }
                    same(&label, (&generated.ir, &a), (&out, &b));
                    forced += 1;
                }
            }
        }
    }
    assert!(forced > 3000, "forced {forced} exits");
}

#[test]
fn guard_hoisting_only_moves_exits_to_the_entry_state() {
    let mut hoisted_runs = 0_usize;
    let mut hoisted_traces = 0_usize;
    for seed in 0..SEEDS * 2 {
        let generated = generate(seed);
        let (out, stats) = optimize_trace(&generated.ir, OptimizeLevel::Optimized);
        out.verify().unwrap();
        if stats.guards_hoisted == 0 {
            continue;
        }
        hoisted_traces += 1;
        let preheader_guards = out
            .preheader
            .iter()
            .filter(|instruction| instruction.op.guard().is_some())
            .count();
        let mut rng = Rng(seed ^ 0x2468_ace0 | 1);
        for _ in 0..8 {
            let entry = entry_state(&mut rng, &generated.types);
            let a = evaluate(&generated.ir, &entry, &[], 4, heap);
            let b = evaluate(&out, &entry, &[], 4, heap);
            match &b.outcome {
                Outcome::GuardFailed { guard, restored } if guard.index() < preheader_guards => {
                    hoisted_runs += 1;
                    // The interpreter restarts the iteration from the entry
                    // state.
                    assert_eq!(restored.registers, entry, "seed {seed}");
                    assert_eq!(restored.effect_position, 0);
                    assert_eq!(
                        b.effects,
                        [] as [crate::jit_ir::trace::reference::EffectRecord; 0]
                    );
                    assert_eq!(b.iterations, 0);
                    // The original must also have left in its first
                    // iteration, since the condition cannot change.
                    assert!(
                        matches!(a.outcome, Outcome::GuardFailed { .. }) && a.iterations == 0,
                        "seed {seed}: original {:?}",
                        a.outcome
                    );
                }
                _ => same(
                    &format!("seed {seed} entry {entry:?}"),
                    (&generated.ir, &a),
                    (&out, &b),
                ),
            }
        }
    }
    assert!(
        hoisted_traces > 20,
        "{hoisted_traces} traces hoisted a guard"
    );
    assert!(
        hoisted_runs > 20,
        "{hoisted_runs} runs failed a hoisted guard"
    );
}

#[test]
fn a_trace_that_keeps_its_branch_guards_hoists_none_and_agrees() {
    // A root of a short loop in a nest keeps its branch guards in the loop,
    // where its sides start (`TraceIr::keep_branch_guards`); everything
    // else is optimized as before.
    let mut kept_traces = 0_usize;
    for seed in 0..SEEDS * 2 {
        let generated = generate(seed);
        let (hoisting, _) = optimize_trace(&generated.ir, OptimizeLevel::Optimized);
        let hoisted_branches = hoisting
            .preheader
            .iter()
            .filter(|instruction| matches!(instruction.op, TraceOp::GuardBranch { .. }))
            .count();
        if hoisted_branches == 0 {
            continue;
        }
        kept_traces += 1;
        let mut ir = generated.ir.clone();
        ir.keep_branch_guards = true;
        let (out, _) = optimize_trace(&ir, OptimizeLevel::Optimized);
        out.verify().unwrap();
        assert!(out.keep_branch_guards, "seed {seed}");
        assert!(
            !out.preheader
                .iter()
                .any(|instruction| matches!(instruction.op, TraceOp::GuardBranch { .. })),
            "seed {seed}"
        );
        let mut rng = Rng(seed ^ 0x1357_9bdf | 1);
        for _ in 0..8 {
            let entry = entry_state(&mut rng, &generated.types);
            let a = evaluate(&generated.ir, &entry, &[], 4, heap);
            let b = evaluate(&out, &entry, &[], 4, heap);
            same(
                &format!("seed {seed} entry {entry:?}"),
                (&generated.ir, &a),
                (&out, &b),
            );
        }
    }
    assert!(
        kept_traces > 10,
        "{kept_traces} traces hoisted a branch guard"
    );
}

#[test]
fn entry_mismatch_is_unchanged_by_optimization() {
    for seed in 0..200 {
        let generated = generate(seed);
        let (out, _) = optimize_trace(&generated.ir, OptimizeLevel::Optimized);
        let mut entry: Vec<TraceScalar> = generated
            .types
            .iter()
            .map(|ty| random_scalar(&mut Rng(seed | 1), *ty))
            .collect();
        entry[0] = if matches!(generated.types[0], TraceType::Float) {
            int(1)
        } else {
            f(1.0)
        };
        let a = evaluate(&generated.ir, &entry, &[], 3, heap);
        let b = evaluate(&out, &entry, &[], 3, heap);
        assert_eq!(a.outcome, Outcome::EntryMismatch);
        assert_eq!(b.outcome, Outcome::EntryMismatch);
    }
}

#[test]
fn the_optimizer_leaves_unverifiable_input_alone() {
    let mut ir = invariant_loop();
    ir.guards.clear();
    let (out, stats) = optimize_trace(&ir, OptimizeLevel::Optimized);
    assert_eq!(out, ir);
    assert_eq!(stats.total(), 0);
}

#[test]
fn passes_never_reuse_a_snapshot_id_or_leave_gaps() {
    let mut seen = HashSet::new();
    for seed in 0..300 {
        let ir = generate(seed).ir;
        let (out, _) = optimize_trace(&ir, OptimizeLevel::Optimized);
        for (index, snapshot) in out.snapshots.iter().enumerate() {
            assert_eq!(snapshot.id.index(), index);
            seen.insert((seed, index));
        }
        let guard_count = count(&out, |op| op.guard().is_some());
        assert_eq!(guard_count, out.guards.len());
    }
    assert!(!seen.is_empty());
}

// ------------------------------------------------------- inlined calls

/// `sum = sum + f(x)` with `f` inlined: root registers `r0` the sum, `r1` the
/// callee, `r2` the argument. The callee (frame 1, pinned closure 0) triples
/// its argument and guards that the product is positive. The identity guard
/// and the callee's guard depend only on invariant inputs.
fn inlined_call_loop() -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(1), root(3));
    b.set_pinned_closures(vec![0x00c0_ffee]);
    let callee = b.add_frame(TraceFrameInfo {
        register_count: 2,
        code_len: 512,
        inlined: Some(InlinedFrame {
            closure: 0,
            parent: FrameId(0),
            call_pc: 1,
            result_base: 2,
            results: 1,
            reexec: false,
            tail_calls: 0,
            tail_site: None,
            protected: Protected::No,
            entry: Entry::Call,
            dynamic: false,
        }),
        activation: None,
    });
    b.mark_written(FrameId(0), 0);
    let sum_in = b.input(0, TraceType::Float);
    let function = b.input(1, TraceType::Closure);
    let x = b.input(2, TraceType::Float);
    let sum = b.loop_param(TraceType::Float, LoopParamInit::Input(sum_in));
    let pinned = emit(
        &mut b,
        1,
        TraceOp::Constant(TraceScalar::Closure(0)),
        TraceType::Closure,
    );
    let same = emit(
        &mut b,
        1,
        TraceOp::Compare {
            comparison: Comparison::Equal,
            left: function,
            right: pinned,
        },
        TraceType::Boolean,
    );
    let state = |sum: ValueId| {
        vec![
            RegisterEntry::Value(sum),
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
        ]
    };
    let before = b.snapshot(snap(
        SnapshotPhase::Before { pc: 1 },
        0,
        Some(5),
        state(sum),
    ));
    let g0 = b.guard(site(1), before);
    b.emit(
        site(1),
        TraceOp::GuardBranch {
            condition: same,
            expected: true,
            guard: g0,
        },
        None,
    );
    let inside = |pc| TraceSite { frame: callee, pc };
    let argument = emit(&mut b, 1, TraceOp::Move { source: x }, TraceType::Float);
    let three = b
        .emit(inside(0), TraceOp::Constant(f(3.0)), Some(TraceType::Float))
        .unwrap();
    let product = b
        .emit(
            inside(0),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Multiply,
                left: argument,
                right: three,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let zero = b
        .emit(inside(1), TraceOp::Constant(f(0.0)), Some(TraceType::Float))
        .unwrap();
    let positive = b
        .emit(
            inside(1),
            TraceOp::Compare {
                comparison: Comparison::LessThan,
                left: zero,
                right: product,
            },
            Some(TraceType::Boolean),
        )
        .unwrap();
    let in_callee = b.snapshot(Snapshot::new(
        SnapshotPhase::After {
            pc: 1,
            successor: 2,
        },
        0,
        vec![
            SnapshotFrame {
                frame: FrameId(0),
                last_interpreted_pc: Some(1),
                registers: state(sum).into(),
            },
            SnapshotFrame {
                frame: callee,
                last_interpreted_pc: Some(1),
                registers: vec![
                    RegisterEntry::Value(argument),
                    RegisterEntry::Value(product),
                ]
                .into(),
            },
        ],
    ));
    let g1 = b.guard(inside(1), in_callee);
    b.emit(
        inside(1),
        TraceOp::GuardBranch {
            condition: positive,
            expected: true,
            guard: g1,
        },
        None,
    );
    let total = arith(&mut b, 2, ArithmeticOp::Add, sum, product);
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 2,
            successor: 1,
        },
        0,
        Some(2),
        state(total),
    ));
    b.set_poll_snapshot(poll);
    let ir = b.finish(TraceTerminator::LoopBack { args: vec![total] });
    ir.verify().unwrap();
    ir
}

#[test]
fn an_invariant_callee_guard_and_its_callee_leave_the_loop() {
    let ir = inlined_call_loop();
    let (out, stats) = optimize_trace(&ir, OptimizeLevel::Optimized);
    out.verify().unwrap();
    // What the trace specialised on and the frames it names survive.
    assert_eq!(out.pinned_closures, ir.pinned_closures);
    assert_eq!(out.frames, ir.frames);
    assert!(stats.guards_hoisted >= 2, "{stats:?}");
    // Both guards moved to the preheader and exit through the entry state,
    // which describes the root frame only.
    assert_eq!(out.guards.len(), 2);
    for guard in &out.guards {
        let snapshot = &out.snapshots[guard.snapshot.index()];
        assert_eq!(snapshot.frames.len(), 1);
        assert!(matches!(snapshot.phase, SnapshotPhase::Before { .. }));
    }
    // The steady state is the add alone. The pinned lane stays a constant
    // and no snapshot names it, since a snapshot cannot hold one.
    assert_eq!(out.body.len(), 1, "{:#?}", out.body);
    assert!(
        out.preheader.iter().any(|instruction| matches!(
            instruction.op,
            TraceOp::Constant(TraceScalar::Closure(0))
        ))
    );
    for snapshot in &out.snapshots {
        for frame in &snapshot.frames {
            assert!(frame.registers.iter().all(|entry| !matches!(
                entry,
                RegisterEntry::Const(scalar) if scalar.ty().is_opaque()
            )));
        }
    }
}

#[test]
fn optimized_inlined_loops_agree_with_the_originals_and_restore_every_frame() {
    let ir = inlined_call_loop();
    let (hoisted, _) = optimize_trace(&ir, OptimizeLevel::Optimized);
    let (kept, _) = optimize_with(
        &ir,
        OptimizeOptions {
            level: OptimizeLevel::Optimized,
            hoist_guards: false,
            memory: true,
        },
    );
    kept.verify().unwrap();
    for (function, x) in [
        (TraceScalar::Closure(0), 2.0),
        (TraceScalar::Closure(0), -2.0),
        (TraceScalar::Closure(0), f64::NAN),
        (TraceScalar::Closure(1), 2.0),
    ] {
        let entry = [f(10.0), function, f(x)];
        for iterations in 1..5 {
            let original = evaluate(&ir, &entry, &[], iterations, |_, _| None);
            let plain = evaluate(&kept, &entry, &[], iterations, |_, _| None);
            // Without hoisting every exit is the original's, outer frames
            // included.
            assert_eq!(original.outcome, plain.outcome, "{function:?} {x}");
            assert_eq!(original.iterations, plain.iterations);
            let moved = evaluate(&hoisted, &entry, &[], iterations, |_, _| None);
            match (&original.outcome, &moved.outcome) {
                (Outcome::IterationLimit { .. }, outcome) => {
                    assert_eq!(&original.outcome, outcome, "{function:?} {x}");
                }
                (Outcome::GuardFailed { .. }, Outcome::GuardFailed { restored, .. }) => {
                    // A hoisted guard leaves before the iteration, with the
                    // root frame as it entered.
                    assert_eq!(restored.registers, entry);
                    assert_eq!(
                        restored.outer,
                        [] as [crate::jit_ir::trace::reference::RestoredFrame; 0]
                    );
                    assert_eq!(moved.iterations, 0);
                }
                other => panic!("{other:?}"),
            }
        }
    }
    // A forced failure of each guard of the unhoisted trace restores the same
    // frames in the original and the optimized one.
    for guard in 0..ir.guards.len() {
        let site = ir.guards[guard].site;
        let forced = ForcedExit {
            frame: site.frame,
            pc: site.pc,
            guard: None,
            iteration: 2,
        };
        let entry = [f(10.0), TraceScalar::Closure(0), f(2.0)];
        let a = evaluate_forced(&ir, &entry, &[], 10, Some(forced), |_, _| None);
        let b = evaluate_forced(&kept, &entry, &[], 10, Some(forced), |_, _| None);
        let (Outcome::GuardFailed { restored: ra, .. }, Outcome::GuardFailed { restored: rb, .. }) =
            (&a.outcome, &b.outcome)
        else {
            panic!("{:?} {:?}", a.outcome, b.outcome)
        };
        assert_eq!(ra, rb, "guard {guard}");
        assert_eq!(ra.outer.len(), usize::from(guard == 1));
    }
}

// ---- native calls ----------------------------------------------------------

extern "C" fn halve(value: f64) -> f64 {
    value / 2.0
}

extern "C" fn third(value: f64) -> f64 {
    value / 3.0
}

extern "C" fn smaller(left: f64, right: f64) -> f64 {
    left.min(right)
}

fn native_unary(
    b: &mut TraceBuilder,
    pc: usize,
    function: extern "C" fn(f64) -> f64,
    source: ValueId,
) -> ValueId {
    emit(
        b,
        pc,
        TraceOp::NativeUnary {
            function: NativeUnary(function),
            source,
        },
        TraceType::Float,
    )
}

#[test]
fn folding_evaluates_native_calls_on_constants() {
    let ir = straight(|b| {
        let x = konst(b, 0, f(9.0));
        let y = konst(b, 1, f(4.0));
        let halved = native_unary(b, 2, halve, x);
        let least = emit(
            b,
            3,
            TraceOp::NativeBinary {
                function: NativeBinary(smaller),
                left: x,
                right: y,
            },
            TraceType::Float,
        );
        vec![halved, least]
    });
    let (out, _) = run_pass(&ir, fold_constants);
    assert_eq!(
        count(&out, |op| matches!(op, TraceOp::NativeUnary { .. })),
        0
    );
    assert_eq!(
        count(&out, |op| matches!(op, TraceOp::NativeBinary { .. })),
        0
    );
    assert_eq!(exit_registers(&out), vec![f(4.5), f(4.0)]);
}

#[test]
fn value_numbering_merges_a_native_call_only_with_the_same_function() {
    let mut b = TraceBuilder::new(TraceId(1), root(4));
    let x = b.input(0, TraceType::Float);
    let first = native_unary(&mut b, 1, halve, x);
    let again = native_unary(&mut b, 2, halve, x);
    let other = native_unary(&mut b, 3, third, x);
    let pair = emit(
        &mut b,
        4,
        TraceOp::NativeBinary {
            function: NativeBinary(smaller),
            left: x,
            right: first,
        },
        TraceType::Float,
    );
    for register in 0..4 {
        b.mark_written(FrameId(0), register);
    }
    let mut ir = b.finish(TraceTerminator::Exit {
        snapshot: SnapshotId(0),
    });
    ir.snapshots.push(Snapshot {
        id: SnapshotId(0),
        ..snap(
            SnapshotPhase::Before { pc: 400 },
            0,
            None,
            vec![
                RegisterEntry::Value(first),
                RegisterEntry::Value(again),
                RegisterEntry::Value(other),
                RegisterEntry::Value(pair),
            ],
        )
    });
    ir.verify().unwrap();
    let (out, stats) = run_pass(&ir, number_values);
    assert_eq!(stats.merged, 1, "{stats:?}");
    assert_eq!(
        count(&out, |op| matches!(op, TraceOp::NativeUnary { .. })),
        2
    );
}

#[test]
fn a_dead_native_call_is_removed_and_the_verifier_wants_floats() {
    let ir = straight(|b| {
        let x = konst(b, 0, f(2.0));
        let _unused = native_unary(b, 1, halve, x);
        vec![x]
    });
    let (out, _) = run_pass(&ir, eliminate_dead_code);
    assert_eq!(
        count(&out, |op| matches!(op, TraceOp::NativeUnary { .. })),
        0
    );

    let mut b = TraceBuilder::new(TraceId(1), root(2));
    b.mark_written(FrameId(0), 1);
    let n = b.input(0, TraceType::Integer);
    let result = b.emit(
        site(1),
        TraceOp::NativeUnary {
            function: NativeUnary(halve),
            source: n,
        },
        Some(TraceType::Float),
    );
    // The builder takes the op, the verifier refuses it.
    let mut ir = b.finish(TraceTerminator::Exit {
        snapshot: SnapshotId(0),
    });
    ir.snapshots.push(Snapshot {
        id: SnapshotId(0),
        ..snap(
            SnapshotPhase::Before { pc: 400 },
            0,
            None,
            vec![
                RegisterEntry::Value(n),
                RegisterEntry::Value(result.unwrap()),
            ],
        )
    });
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::TypeMismatch { .. })
    ));
}

#[test]
fn the_memory_flag_follows_the_level() {
    assert!(OptimizeOptions::for_level(OptimizeLevel::Optimized).memory);
    assert!(!OptimizeOptions::for_level(OptimizeLevel::Baseline).memory);
}

#[test]
fn opt_flags_switch_passes_off_but_never_past_the_tier() {
    use crate::compat::{TracingOptFlags, TracingOptLevel};
    let level_0 = TracingOptLevel::O0.flags();
    assert_eq!(
        OptimizeOptions::for_level_and_flags(OptimizeLevel::Optimized, level_0),
        OptimizeOptions::for_level(OptimizeLevel::Optimized)
    );
    let no_memory = OptimizeOptions::for_level_and_flags(
        OptimizeLevel::Optimized,
        level_0.without(TracingOptFlags::MEMORY),
    );
    assert!(!no_memory.memory);
    assert!(no_memory.hoist_guards);
    let baseline = OptimizeOptions::for_level_and_flags(OptimizeLevel::Baseline, level_0);
    assert_eq!(
        baseline,
        OptimizeOptions::for_level(OptimizeLevel::Baseline)
    );
}

#[test]
fn load_counters_start_at_zero_and_are_summed() {
    let mut total = OptimizeStats::default();
    total.accumulate(OptimizeStats {
        loads_forwarded: 2,
        loads_merged: 1,
        ..OptimizeStats::default()
    });
    total.accumulate(OptimizeStats {
        loads_forwarded: 3,
        ..OptimizeStats::default()
    });
    assert_eq!((total.loads_forwarded, total.loads_merged), (5, 1));
    assert_eq!(total.total(), 6);
    let (_, stats) = optimize_trace(&generate(0).ir, OptimizeLevel::Optimized);
    assert_eq!((stats.loads_forwarded, stats.loads_merged), (0, 0));
}

// ---- memory option ---------------------------------------------------------

/// The optimized pipeline as it ran before the memory pass existed.
pub(in crate::jit_ir::trace) fn pipeline_without_memory(ir: &TraceIr) -> TraceIr {
    let mut current = ir.clone();
    for _ in 0..4 {
        let mut changes = 0;
        for pass in [
            eliminate_copies,
            fold_constants,
            eliminate_guards,
            number_values,
            eliminate_dead_code,
        ] {
            let (next, stats) = run_pass(&current, pass);
            changes += stats.total();
            current = next;
        }
        if changes == 0 {
            break;
        }
    }
    run_pass(&current, |ir, stats| {
        let options = OptimizeOptions {
            memory: false,
            ..OptimizeOptions::for_level(OptimizeLevel::Optimized)
        };
        hoist_invariants(ir, &options, &TreeWrites::of_members([ir]), stats)
    })
    .0
}

#[test]
fn with_the_memory_option_off_the_optimized_ir_is_the_old_pipelines() {
    use super::super::table_tests::{Mem, array_binding, mineclonia_case};
    let mut m = Mem::new(1, 0, vec![array_binding(0)]);
    let t = m.tables[0];
    let k = m.konst(int(1));
    let first = m.load(0, t, k, TraceType::Float);
    m.out(first);
    let second = m.load(0, t, k, TraceType::Float);
    m.out(second);
    let repeated = m.finish();
    for ir in [
        mineclonia_case(),
        repeated,
        invariant_loop(),
        guarded_read(),
    ] {
        let options = OptimizeOptions {
            memory: false,
            ..OptimizeOptions::for_level(OptimizeLevel::Optimized)
        };
        let (off, stats) = optimize_with(&ir, options);
        assert_eq!((stats.loads_forwarded, stats.loads_merged), (0, 0));
        assert_eq!(off, pipeline_without_memory(&ir));
        let (on, _) = optimize_with(&ir, OptimizeOptions::for_level(OptimizeLevel::Optimized));
        on.verify().unwrap();
    }
}

// ---- passes that change nothing ---------------------------------------------

/// A trace whose result depends on an input, so no pass can remove or fold
/// anything in it.
fn nothing_to_optimize() -> TraceIr {
    straight(|b| {
        let x = b.input(0, TraceType::Float);
        let one = konst(b, 1, f(1.0));
        vec![arith(b, 2, ArithmeticOp::Add, x, one)]
    })
}

#[test]
fn a_trace_with_nothing_to_optimize_comes_back_equal() {
    let ir = nothing_to_optimize();
    let (out, stats) = optimize_trace(&ir, OptimizeLevel::Optimized);
    assert_eq!(stats.total(), 0);
    assert_eq!(out, ir);
    let (out, stats) = optimize_trace(&ir, OptimizeLevel::Baseline);
    assert_eq!(stats.total(), 0);
    assert_eq!(out, ir);
}

#[test]
fn a_pass_that_changes_nothing_returns_none_with_zero_stats() {
    let ir = nothing_to_optimize();
    let passes: [fn(&TraceIr, &mut OptimizeStats) -> Option<TraceIr>; 5] = [
        eliminate_copies,
        fold_constants,
        eliminate_guards,
        number_values,
        eliminate_dead_code,
    ];
    for pass in passes {
        let mut stats = OptimizeStats::default();
        assert!(pass(&ir, &mut stats).is_none());
        assert_eq!(stats.total(), 0);
    }
}

/// The pipeline that rebuilds after every pass, as it once did.
fn optimize_always_rebuild(ir: &TraceIr, options: OptimizeOptions) -> (TraceIr, OptimizeStats) {
    optimize_pipeline(ir, options, &TreeWrites::of_members([ir]), true, true)
}

fn all_option_sets() -> [OptimizeOptions; 4] {
    [
        OptimizeOptions::for_level(OptimizeLevel::Baseline),
        OptimizeOptions::for_level(OptimizeLevel::Optimized),
        OptimizeOptions {
            hoist_guards: false,
            ..OptimizeOptions::for_level(OptimizeLevel::Optimized)
        },
        OptimizeOptions {
            memory: false,
            ..OptimizeOptions::for_level(OptimizeLevel::Optimized)
        },
    ]
}

#[test]
fn skipping_rebuilds_of_unchanged_passes_gives_the_always_rebuild_result() {
    let mut fixtures = vec![
        nothing_to_optimize(),
        guarded_read(),
        invariant_loop(),
        inlined_call_loop(),
        branch_trace(&[(true, true), (false, true), (true, false)]),
    ];
    fixtures.extend((0..300).map(|seed| generate(seed).ir));
    for (index, ir) in fixtures.iter().enumerate() {
        for options in all_option_sets() {
            let (fast, fast_stats) = optimize_with(ir, options);
            let (slow, mut slow_stats) = optimize_always_rebuild(ir, options);
            assert_eq!(fast, slow, "fixture {index}, {options:?}");
            // A loop whose body emptied is padded with one dead constant by
            // every rebuild, and the next round removes and counts it again.
            // The single draft is padded once, so it counts fewer.
            let padded = matches!(
                fast.body.as_slice(),
                [TraceInstruction {
                    op: TraceOp::Constant(TraceScalar::Boolean(false)),
                    ..
                }]
            );
            if padded {
                assert!(fast_stats.dead_removed <= slow_stats.dead_removed);
                slow_stats.dead_removed = fast_stats.dead_removed;
            }
            assert_eq!(fast_stats, slow_stats, "fixture {index}");
        }
    }
}

#[test]
fn finishing_a_draft_twice_changes_nothing() {
    for ir in [guarded_read(), invariant_loop(), inlined_call_loop()] {
        let once = Draft::new(&ir).finish();
        assert_eq!(Draft::new(&once).finish(), once);
    }
}

#[test]
fn a_draft_borrows_snapshots_until_a_pass_writes_one() {
    let ir = guarded_read();
    let draft = Draft::new(&ir);
    assert!(
        matches!(draft.snapshots, std::borrow::Cow::Borrowed(_)),
        "a draft copies no snapshot until a pass edits one"
    );
    // Finishing clones only the snapshots the trace still references.
    assert_eq!(draft.finish(), Draft::new(&ir).finish());
}

#[test]
fn number_values_pre_check_skips_traces_without_repeated_keys() {
    let ir = nothing_to_optimize();
    let mut stats = OptimizeStats::default();
    assert!(number_values(&ir, &mut stats).is_none());
    assert_eq!(stats.total(), 0);

    // A repeated pure operation is still merged, so the pre-check is not
    // skipping real work.
    let repeated = straight(|b| {
        let x = b.input(0, TraceType::Float);
        let first = arith(b, 1, ArithmeticOp::Add, x, x);
        let second = arith(b, 2, ArithmeticOp::Add, x, x);
        vec![first, second]
    });
    let mut stats = OptimizeStats::default();
    let merged = number_values(&repeated, &mut stats).expect("a repeat is merged");
    assert_eq!(stats.merged, 1);
    merged.verify().unwrap();
}

/// A trace over a boolean input `r0` and a float input `r1` that exits
/// through one snapshot holding `results`.
fn value_select_trace(
    build: impl FnOnce(&mut TraceBuilder, ValueId, ValueId) -> Vec<ValueId>,
) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(1), root(2));
    let condition = b.input(0, TraceType::Boolean);
    let x = b.input(1, TraceType::Float);
    let results = build(&mut b, condition, x);
    let written = u16::try_from(results.len()).unwrap();
    let width = written.max(2);
    for register in 0..written {
        b.mark_written(FrameId(0), register);
    }
    let mut ir = b.finish(TraceTerminator::Exit {
        snapshot: SnapshotId(0),
    });
    ir.frames[0].register_count = width;
    let mut registers: Vec<RegisterEntry> = results
        .iter()
        .map(|value| RegisterEntry::Value(*value))
        .collect();
    registers.resize(usize::from(width), RegisterEntry::Unchanged);
    ir.snapshots.push(Snapshot {
        id: SnapshotId(0),
        ..snap(SnapshotPhase::Before { pc: 400 }, 0, None, registers)
    });
    ir.verify().unwrap();
    ir
}

fn select(
    b: &mut TraceBuilder,
    condition: ValueId,
    if_true: ValueId,
    if_false: ValueId,
) -> ValueId {
    let ty = b.value_type_for_tests(if_true);
    emit(
        b,
        5,
        TraceOp::Select {
            condition,
            if_true,
            if_false,
        },
        ty,
    )
}

#[test]
fn a_value_select_picks_the_arm_of_its_condition_in_the_reference() {
    let ir = value_select_trace(|b, condition, x| {
        let one = konst(b, 1, f(1.0));
        let doubled = arith(b, 2, ArithmeticOp::Multiply, x, x);
        vec![select(b, condition, one, doubled)]
    });
    for (flag, expected) in [(true, 1.0), (false, 6.25)] {
        let outcome = evaluate(
            &ir,
            &[TraceScalar::Boolean(flag), f(2.5)],
            &[],
            1,
            |_, _| None,
        )
        .outcome;
        let Outcome::Exited(restored) = outcome else {
            panic!("expected an exit, got {outcome:?}")
        };
        assert_eq!(restored.registers[0], f(expected), "{flag}");
    }
}

#[test]
fn a_value_select_with_a_known_condition_or_equal_arms_folds_to_one_arm() {
    let ir = value_select_trace(|b, condition, x| {
        let yes = konst(b, 1, TraceScalar::Boolean(true));
        let no = konst(b, 1, TraceScalar::Boolean(false));
        let one = konst(b, 1, f(1.0));
        vec![
            select(b, yes, one, x),
            select(b, no, one, x),
            select(b, condition, x, x),
            select(b, condition, one, x),
        ]
    });
    let (out, stats) = run_pass(&ir, fold_constants);
    assert_eq!(stats.folded, 3, "{stats:?}");
    assert_eq!(count(&out, |op| matches!(op, TraceOp::Select { .. })), 1);
    for flag in [true, false] {
        let inputs = [TraceScalar::Boolean(flag), f(2.5)];
        let before = evaluate(&ir, &inputs, &[], 1, |_, _| None).outcome;
        let after = evaluate(&out, &inputs, &[], 1, |_, _| None).outcome;
        assert_eq!(format!("{before:?}"), format!("{after:?}"));
    }
}

#[test]
fn value_numbering_merges_equal_value_selects_only() {
    let ir = value_select_trace(|b, condition, x| {
        let one = konst(b, 1, f(1.0));
        let first = select(b, condition, one, x);
        let second = select(b, condition, one, x);
        // The arms swapped are another value.
        let swapped = select(b, condition, x, one);
        vec![first, second, swapped]
    });
    let (out, stats) = run_pass(&ir, number_values);
    assert_eq!(stats.merged, 1, "{stats:?}");
    assert_eq!(count(&out, |op| matches!(op, TraceOp::Select { .. })), 2);
}

#[test]
fn a_value_select_needs_a_boolean_condition_and_arms_of_one_number_type() {
    let verify = |condition_ty: TraceType, arm: TraceScalar| {
        let mut b = TraceBuilder::new(TraceId(1), root(2));
        let condition = b.input(0, condition_ty);
        let x = b.input(1, TraceType::Float);
        let other = konst(&mut b, 1, arm);
        let picked = b
            .emit(
                site(2),
                TraceOp::Select {
                    condition,
                    if_true: x,
                    if_false: other,
                },
                Some(TraceType::Float),
            )
            .unwrap();
        b.mark_written(FrameId(0), 0);
        let mut ir = b.finish(TraceTerminator::Exit {
            snapshot: SnapshotId(0),
        });
        ir.snapshots.push(Snapshot {
            id: SnapshotId(0),
            ..snap(
                SnapshotPhase::Before { pc: 400 },
                0,
                None,
                vec![RegisterEntry::Value(picked), RegisterEntry::Unchanged],
            )
        });
        ir.verify()
    };
    assert!(verify(TraceType::Boolean, f(1.0)).is_ok());
    // A float arm against an integer arm.
    assert!(verify(TraceType::Boolean, int(1)).is_err());
    // A float condition.
    assert!(verify(TraceType::Float, f(1.0)).is_err());
}
