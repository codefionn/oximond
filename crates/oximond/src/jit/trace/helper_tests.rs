//! Direct helper calls ([`TraceOp::CallHelper`]) against the reference
//! evaluator: typed register arguments, a status that leaves through the
//! guard, and a helper with two results.

use super::*;
use crate::jit::OptimizationTier;
use crate::jit_ir::trace::reference::{Outcome, TraceHeap, evaluate_with_heap};
use crate::jit_ir::trace::{
    FrameId, HelperEffects, HelperId, RegisterEntry, Snapshot, SnapshotFrame, SnapshotPhase,
    TableBinding, TraceBuilder, TraceFrameInfo, TraceId, TraceIr, TraceOp, TraceScalar, TraceSite,
    TraceTerminator, TraceType, integer_floor_divide, integer_floor_modulo,
};
use std::sync::atomic::Ordering;

const TIERS: [OptimizationTier; 2] = [OptimizationTier::Baseline, OptimizationTier::Optimized];

fn site(pc: usize) -> TraceSite {
    TraceSite {
        frame: FrameId(0),
        pc,
    }
}

fn snap(phase: SnapshotPhase, effects: u32, registers: Vec<RegisterEntry>) -> Snapshot {
    Snapshot::new(
        phase,
        effects,
        vec![SnapshotFrame {
            frame: FrameId(0),
            last_interpreted_pc: Some(0),
            registers: registers.into(),
        }],
    )
}

/// The reference's view of the helpers: what the native functions compute.
struct Helpers;

impl TraceHeap for Helpers {
    fn entry_accepts(&mut self, _: &[TableBinding]) -> bool {
        true
    }
    fn array_get(&mut self, _: u32, _: TraceScalar) -> Option<TraceScalar> {
        unreachable!()
    }
    fn array_set(&mut self, _: u32, _: TraceScalar, _: TraceScalar) -> bool {
        unreachable!()
    }
    fn field_get(&mut self, _: u32) -> TraceScalar {
        unreachable!()
    }
    fn field_set(&mut self, _: u32, _: TraceScalar) {
        unreachable!()
    }
    fn length(&mut self, _: u32) -> TraceScalar {
        unreachable!()
    }
    fn helper_call(&mut self, helper: HelperId, args: &[TraceScalar]) -> Option<Vec<TraceScalar>> {
        match (helper, args) {
            (HelperId::TEST_DIVMOD, [TraceScalar::Integer(left), TraceScalar::Integer(right)]) => {
                (*right != 0 && !(*left == i64::MIN && *right == -1)).then(|| {
                    vec![
                        TraceScalar::Integer(integer_floor_divide(*left, *right)),
                        TraceScalar::Integer(integer_floor_modulo(*left, *right)),
                    ]
                })
            }
            (HelperId::TEST_WRITE, [TraceScalar::Integer(value)]) => {
                Some(vec![TraceScalar::Integer(*value)])
            }
            _ => None,
        }
    }
}

/// `r2, r3 = divmod(r0, r1)` once, leaving through the call's guard when the
/// helper misses.
pub(in crate::jit::trace) fn divmod_trace() -> TraceIr {
    let mut b = TraceBuilder::new(
        TraceId(0),
        TraceFrameInfo {
            register_count: 4,
            code_len: 32,
            inlined: None,
            activation: None,
        },
    );
    b.mark_written(FrameId(0), 2);
    b.mark_written(FrameId(0), 3);
    let left = b.input(0, TraceType::Integer);
    let right = b.input(1, TraceType::Integer);
    let before = b.snapshot(snap(
        SnapshotPhase::Before { pc: 0 },
        0,
        vec![
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
            RegisterEntry::Const(TraceScalar::Nil),
            RegisterEntry::Const(TraceScalar::Nil),
        ],
    ));
    let guard = b.guard(site(0), before);
    let quotient = b
        .emit(
            site(0),
            TraceOp::CallHelper {
                helper: HelperId::TEST_DIVMOD,
                args: vec![left, right],
                effect: None,
                guard,
            },
            Some(TraceType::Integer),
        )
        .unwrap();
    let remainder = b
        .emit(
            site(0),
            TraceOp::HelperResult {
                call: quotient,
                index: 1,
            },
            Some(TraceType::Integer),
        )
        .unwrap();
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 0,
            successor: 1,
        },
        0,
        vec![
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
            RegisterEntry::Value(quotient),
            RegisterEntry::Value(remainder),
        ],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

#[test]
fn a_helper_with_two_results_matches_the_reference_and_leaves_on_a_miss() {
    let values = [0, 1, -1, 2, -2, 5, -5, 7, -7, 1 << 40, i64::MAX, i64::MIN];
    let ir = divmod_trace();
    let (mut exits, mut passes) = (0, 0);
    for tier in TIERS {
        let trace = compile_trace(&ir, tier).unwrap();
        // The helper takes no context, so a plain run is enough.
        assert!(!trace.needs_tables());
        assert!(trace.calls_host());
        for left in values {
            for right in values {
                let registers = [
                    TraceScalar::Integer(left),
                    TraceScalar::Integer(right),
                    TraceScalar::Nil,
                    TraceScalar::Nil,
                ];
                let inputs: Vec<TraceInputSlot> = registers[..2]
                    .iter()
                    .map(|scalar| TraceInputSlot::from_scalar(*scalar))
                    .collect();
                let mut area = vec![0_i64; trace.exit_area_len()];
                let mut control = TraceControl::new(100);
                let TraceOutcome::Exit { index } = trace.run(&inputs, &mut control, &mut area)
                else {
                    panic!("the trace refused entry");
                };
                let descriptor = &trace.exits()[index];
                let snapshot = &trace.lowered_ir().snapshots[descriptor.snapshot.index()];
                let state: Vec<TraceScalar> = snapshot.frames[0]
                    .registers
                    .iter()
                    .enumerate()
                    .map(|(register, entry)| match entry {
                        RegisterEntry::Value(value) => descriptor.scalar(&area, *value).unwrap(),
                        RegisterEntry::Const(scalar) => *scalar,
                        RegisterEntry::Unchanged => registers[register],
                        _ => unreachable!(),
                    })
                    .collect();
                let evaluation =
                    evaluate_with_heap(&ir, &registers, &[], 100, None, &mut Helpers, |_, _| None);
                let (expected, guard) = match evaluation.outcome {
                    Outcome::GuardFailed { guard, restored } => (restored, Some(guard)),
                    Outcome::Exited(restored) => (restored, None),
                    other => panic!("unexpected reference outcome {other:?}"),
                };
                let label = format!("{left} divmod {right} at {tier:?}");
                assert_eq!(descriptor.origin, guard.map(|guard| guard.0), "{label}");
                let words = |scalars: &[TraceScalar]| -> Vec<String> {
                    scalars.iter().map(|scalar| format!("{scalar:?}")).collect()
                };
                assert_eq!(words(&state), words(&expected.registers), "{label}");
                exits += usize::from(guard.is_some());
                passes += usize::from(guard.is_none());
            }
        }
    }
    // A zero divisor and `MIN // -1` miss; the rest answer.
    assert!(exits > 20 && passes > 100, "{exits} exits, {passes} passes");
}

/// `r1 = write(r0)`: a helper that writes commits an effect of its own.
fn write_trace() -> TraceIr {
    let mut b = TraceBuilder::new(
        TraceId(0),
        TraceFrameInfo {
            register_count: 2,
            code_len: 32,
            inlined: None,
            activation: None,
        },
    );
    b.mark_written(FrameId(0), 1);
    let value = b.input(0, TraceType::Integer);
    let before = b.snapshot(snap(
        SnapshotPhase::Before { pc: 0 },
        0,
        vec![
            RegisterEntry::Unchanged,
            RegisterEntry::Const(TraceScalar::Nil),
        ],
    ));
    let guard = b.guard(site(0), before);
    assert!(
        HelperId::TEST_WRITE
            .effects()
            .contains(HelperEffects::WRITES_HEAP)
    );
    let effect = Some(b.next_effect());
    let answer = b
        .emit(
            site(0),
            TraceOp::CallHelper {
                helper: HelperId::TEST_WRITE,
                args: vec![value],
                effect,
                guard,
            },
            Some(TraceType::Integer),
        )
        .unwrap();
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 0,
            successor: 1,
        },
        1,
        vec![RegisterEntry::Unchanged, RegisterEntry::Value(answer)],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

#[test]
fn a_helper_that_writes_runs_once_with_its_register_arguments() {
    let ir = write_trace();
    for tier in TIERS {
        let trace = compile_trace(&ir, tier).unwrap();
        crate::jit::trace::helpers::TEST_WRITTEN.store(0, Ordering::SeqCst);
        let inputs = [TraceInputSlot::integer(41)];
        let mut area = vec![0_i64; trace.exit_area_len()];
        let mut control = TraceControl::new(100);
        let TraceOutcome::Exit { index } = trace.run(&inputs, &mut control, &mut area) else {
            panic!("the trace refused entry");
        };
        let descriptor = &trace.exits()[index];
        let snapshot = &trace.lowered_ir().snapshots[descriptor.snapshot.index()];
        let RegisterEntry::Value(answer) = snapshot.frames[0].registers[1] else {
            panic!("the answer is a value");
        };
        assert_eq!(
            descriptor.scalar(&area, answer),
            Some(TraceScalar::Integer(41)),
            "{tier:?}"
        );
        assert_eq!(
            crate::jit::trace::helpers::TEST_WRITTEN.load(Ordering::SeqCst),
            41,
            "{tier:?}"
        );
    }
}
