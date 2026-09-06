//! Links that go straight to a loop header (stage 8, links), against
//! Cranelift: a side of a range-versioned root that steps the induction as
//! the root does goes back to the fast loop past the range checks, and its
//! copy for the guarded version still goes through them.

use super::super::compile::compile_trace;
use super::super::table_tests::{Tables, emit, mir_matches_cranelift, root, site, snap};
use super::tests::{TIERS, compile, float, run_hosted};
use crate::jit::OptimizationTier;
use crate::jit_ir::trace::range::tests::LoopSpec;
use crate::jit_ir::trace::{
    ArithmeticOp, FrameId, RegisterEntry, SnapshotPhase, TraceBuilder, TraceId, TraceIr, TraceOp,
    TraceScalar, TraceSide, TraceTerminator, TraceType,
};
use crate::runtime::Value;

/// The counted loop of `spec` with a side on the first read's guard that
/// adds 1000 to the sum, steps the control by the spec's step (the register
/// the root steps by) and, past the bound guard the root has, links back
/// with the stepped value for both the control and the visible variable.
/// With `bounded` false the side has no bound guard, so its link cannot
/// skip the checks.
fn counted_loop_with_a_stepping_side(spec: &LoopSpec, bounded: bool) -> TraceIr {
    let ty = if spec.float {
        TraceType::Float
    } else {
        TraceType::Integer
    };
    let number = |value: f64| {
        if spec.float {
            TraceScalar::Float(value)
        } else {
            TraceScalar::Integer(value as i64)
        }
    };
    let mut ir = crate::jit_ir::trace::range::tests::counted_loop(spec);
    ir.header_live = vec![0, 1, 2];
    let mut s = TraceBuilder::new(TraceId(8), root(7));
    s.set_bindings(ir.bindings.clone());
    for register in 0..3 {
        s.mark_written(FrameId(0), register);
    }
    let control = s.input(0, ty);
    let visible = s.input(1, ty);
    let sum = s.input(2, TraceType::Float);
    let limit = s.input(3, ty);
    let thousand = emit(
        &mut s,
        3,
        TraceOp::Constant(TraceScalar::Float(1000.0)),
        TraceType::Float,
    );
    let sum2 = emit(
        &mut s,
        3,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: sum,
            right: thousand,
        },
        TraceType::Float,
    );
    let step = emit(
        &mut s,
        3,
        TraceOp::Constant(number(spec.step.expect("a constant step"))),
        ty,
    );
    let next = emit(
        &mut s,
        3,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: control,
            right: step,
        },
        ty,
    );
    let registers = |control, visible| {
        vec![
            RegisterEntry::Value(control),
            RegisterEntry::Value(visible),
            RegisterEntry::Value(sum2),
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
        ]
    };
    if bounded {
        let (left, right) = if spec.next_on_left {
            (next, limit)
        } else {
            (limit, next)
        };
        let more = emit(
            &mut s,
            3,
            TraceOp::Compare {
                comparison: spec.comparison,
                left,
                right,
            },
            TraceType::Boolean,
        );
        let out = s.snapshot(snap(
            SnapshotPhase::After {
                pc: 3,
                successor: 20,
            },
            0,
            Some(3),
            registers(next, visible),
        ));
        let guard = s.guard(site(3), out);
        s.emit(
            site(3),
            TraceOp::GuardBranch {
                condition: more,
                expected: spec.expected,
                guard,
            },
            None,
        );
    }
    let link = s.snapshot(snap(
        SnapshotPhase::After {
            pc: 3,
            successor: 1,
        },
        0,
        Some(3),
        registers(next, next),
    ));
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(s.finish(TraceTerminator::Link { snapshot: link })),
    });
    ir
}

/// Starts and limits inside, at the edges and outside the arrays, over
/// arrays with holes (an integer where the loop wants a float, so the side
/// is taken from the fast loop and from the guarded one), as the reference
/// test of sides makes them.
fn cases(float: bool) -> Vec<([TraceScalar; 7], Tables)> {
    let number = |value: f64| {
        if float {
            TraceScalar::Float(value)
        } else {
            TraceScalar::Integer(value as i64)
        }
    };
    let array = |holes: &[usize]| -> Vec<Value> {
        let mut slots = vec![Value::Number(-100.0), Value::Number(-200.0)];
        slots.extend((0..9).map(|index| {
            if holes.contains(&index) {
                Value::Integer(5)
            } else {
                Value::Number(index as f64 + 0.25)
            }
        }));
        slots.extend([Value::Number(100.0), Value::Number(200.0)]);
        slots
    };
    let mut cases = Vec::new();
    for holes in [
        &[][..],
        &[3][..],
        &[0, 4, 7][..],
        &[8][..],
        &[1, 2, 3, 4, 5, 6][..],
    ] {
        for (start, limit) in [
            (1.0, 9.0),
            (2.0, 8.0),
            (9.0, 1.0),
            (1.0, 12.0),
            (0.0, 5.0),
            (3.0, 3.0),
        ] {
            let registers = [
                number(start),
                number(start),
                TraceScalar::Float(0.0),
                number(limit),
                number(1.0),
                TraceScalar::Table(0),
                TraceScalar::Table(1),
            ];
            let tables = Tables {
                arrays: vec![array(holes), array(&[])],
                fields: Vec::new(),
                lengths: Vec::new(),
                padding: 2,
            };
            cases.push((registers, tables));
        }
    }
    cases
}

fn fast_links_lowered(ir: &TraceIr, tier: OptimizationTier) -> u32 {
    links_lowered(ir, tier).0
}

/// The links lowered straight to the fast loop and to the guarded loop.
fn links_lowered(ir: &TraceIr, tier: OptimizationTier) -> (u32, u32) {
    use crate::jit::debug_env::FORCE_BACKEND;
    crate::jit::trace::mir::FAST_LINKS.with(|count| count.set(0));
    crate::jit::trace::mir::GUARDED_LINKS.with(|count| count.set(0));
    FORCE_BACKEND.with(|force| force.set(Some(true)));
    let compiled = compile_trace(ir, tier);
    FORCE_BACKEND.with(|force| force.set(None));
    compiled.unwrap();
    (
        crate::jit::trace::mir::FAST_LINKS.with(std::cell::Cell::get),
        crate::jit::trace::mir::GUARDED_LINKS.with(std::cell::Cell::get),
    )
}

/// A link that steps the induction as the root does and passes the root's
/// bound goes to the fast loop's own `Join`, and runs like Cranelift's: the
/// same exits, snapshots, iterations and memory, from starts the checks pass
/// and fail, with the side taken from the fast loop and from the guarded one,
/// for every fuel.
#[test]
fn a_link_that_steps_like_the_root_goes_to_the_fast_loop_like_cranelift() {
    let mut direct = 0;
    for float in [false, true] {
        let upward = LoopSpec::upward(float);
        for spec in [
            upward.clone(),
            LoopSpec {
                step: Some(2.0),
                reads: vec![-1, 0, 1],
                write: Some(1),
                write_same_table: true,
                ..upward.clone()
            },
            LoopSpec {
                step: Some(-1.0),
                next_on_left: false,
                key_on_control: true,
                ..upward
            },
        ] {
            let ir = counted_loop_with_a_stepping_side(&spec, true);
            ir.verify().unwrap();
            let label = format!("{spec:?}");
            let cases = cases(float);
            for tier in [OptimizationTier::Baseline, OptimizationTier::Optimized] {
                let lowered = fast_links_lowered(&ir, tier);
                if tier == OptimizationTier::Baseline {
                    assert_eq!(lowered, 0, "{label}: no versions at the baseline tier");
                } else {
                    assert_eq!(lowered, 1, "{label}: the link skips the checks");
                    direct += 1;
                }
                mir_matches_cranelift(&ir, tier, &cases, &[100_000, 40, 11, 7, 4, 2], &label);
            }
        }
    }
    assert_eq!(direct, 6);
}

/// A side with no bound guard may step past the end of the array, so its
/// link runs the checks again: nothing is lowered straight to the fast loop,
/// and the code still agrees with Cranelift.
#[test]
fn a_link_without_the_bound_guard_runs_the_checks_again_like_cranelift() {
    for float in [false, true] {
        let spec = LoopSpec::upward(float);
        let ir = counted_loop_with_a_stepping_side(&spec, false);
        ir.verify().unwrap();
        let label = format!("{spec:?}");
        assert_eq!(fast_links_lowered(&ir, OptimizationTier::Optimized), 0);
        mir_matches_cranelift(
            &ir,
            OptimizationTier::Optimized,
            &cases(float),
            &[100_000, 11, 4],
            &label,
        );
    }
}

/// A loop side whose bound is a host call its preheader makes, and a side of
/// that loop that rejoins it more than once. The loop passes no parameter on
/// unchanged, so Cranelift runs the preheader once, before the loop's header,
/// and a rejoin jumps to the header: the call is made once per entry of the
/// loop however often the side rejoins.
#[test]
fn a_loop_side_runs_its_hoisted_call_once_however_often_it_is_rejoined() {
    let mut most = 0;
    for (limit, cap, wrap) in [
        (100.0, 400.0, 30.0),
        (10.0, 90.0, 40.0),
        (64.0, 1000.0, 9.0),
    ] {
        let ir =
            super::super::tree_tests::loop_side::linked_tree_with(limit, cap, wrap, false, true);
        ir.verify().unwrap();
        for tier in TIERS {
            let mir = compile(&ir, tier, true);
            let cranelift = compile(&ir, tier, false);
            assert!(
                mir.mir_address().is_some(),
                "{tier:?}: MIR compiled the tree"
            );
            assert_eq!(mir.exits(), cranelift.exits(), "{tier:?}");
            for start in [1.0, 2.0, 3.0, 4.0, 7.0, 15.0, 39.0, 41.0, 90.5, -3.0, 0.0] {
                for fuel in [1, 2, 3, 7, 40, 900, 1_000_000] {
                    let state = [float(10.0), float(20.0), float(start), TraceScalar::Nil];
                    let (a, calls_a) = run_hosted(&mir, &state, fuel);
                    let (b, calls_b) = run_hosted(&cranelift, &state, fuel);
                    assert_eq!(a, b, "{tier:?} from {state:?}, fuel {fuel}");
                    assert_eq!(calls_a, calls_b, "host calls at {tier:?} from {state:?}");
                    if tier == OptimizationTier::Optimized {
                        most = most.max(calls_a);
                    }
                }
            }
        }
    }
    assert_eq!(
        most, 1,
        "the call is made once per run, however often the side rejoins"
    );
}

/// A link of a side entered from the guarded loop, which changes only what
/// that loop carries, goes to the guarded loop's own `Join`, past the entry
/// values and the checks. The loops start at and past the edges of the
/// arrays, so the checks fail and the guarded loop takes the side, and the
/// code agrees with Cranelift's on exits, iterations and memory. A side with
/// no bound guard has no fast link but still has the guarded one.
#[test]
fn a_link_from_the_guarded_loop_goes_to_its_join_like_cranelift() {
    let mut direct = 0;
    for float in [false, true] {
        for bounded in [true, false] {
            let spec = LoopSpec::upward(float);
            let ir = counted_loop_with_a_stepping_side(&spec, bounded);
            ir.verify().unwrap();
            let label = format!("{spec:?} bounded {bounded}");
            let (fast, guarded) = links_lowered(&ir, OptimizationTier::Optimized);
            assert_eq!(fast, u32::from(bounded), "{label}");
            assert_eq!(guarded, 1, "{label}: the link stays in the guarded loop");
            direct += guarded;
            mir_matches_cranelift(
                &ir,
                OptimizationTier::Optimized,
                &cases(float),
                &[100_000, 40, 11, 7, 4, 2],
                &label,
            );
        }
    }
    assert_eq!(direct, 4);
}

/// The trees of the two tests above, counted (the root's iteration costs 5,
/// the side's path 8): the budget goes through the range checks, the fast
/// loop's `Join`, the guarded loop and its `Join` as the fuel does, and each
/// link pays its own cost, as Cranelift's code does.
#[test]
fn a_counted_versioned_tree_pays_its_budget_like_cranelift() {
    for float in [false, true] {
        for bounded in [true, false] {
            let spec = LoopSpec::upward(float);
            let ir = super::tests::counted::counted(
                counted_loop_with_a_stepping_side(&spec, bounded),
                5,
                8,
            );
            ir.verify().unwrap();
            let label = format!("counted {spec:?} bounded {bounded}");
            let (fast, guarded) = links_lowered(&ir, OptimizationTier::Optimized);
            assert_eq!(fast, u32::from(bounded), "{label}");
            assert_eq!(guarded, 1, "{label}");
            for tier in [OptimizationTier::Baseline, OptimizationTier::Optimized] {
                mir_matches_cranelift(
                    &ir,
                    tier,
                    &cases(float),
                    &[100_000, 40, 11, 7, 4, 2],
                    &label,
                );
            }
        }
    }
}
