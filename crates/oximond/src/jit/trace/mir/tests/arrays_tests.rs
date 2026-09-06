//! Stage 8, arrays: the array step (`ArrayNext`) and the loads that follow
//! it, MIR against Cranelift.

use super::*;
use crate::jit::trace::exit::TraceBinding;
use crate::jit_ir::trace::{BindingKind, TableBinding};
use crate::runtime::Value;

/// `r2 = next index after r0` along the array part of the table in r1, then
/// with `load` the read `r3 = r1[r2]` of the type `load` (the slot the step
/// found), from the start of the array when `start`. The step gives a number
/// of type `result`.
fn step_trace(key: TraceType, start: bool, result: TraceType, load: Option<TraceType>) -> TraceIr {
    use RegisterEntry::{Const, Unchanged, Value as Val};
    let mut b = TraceBuilder::new(TraceId(0), frame(4));
    b.mark_written(FrameId(0), 2);
    b.mark_written(FrameId(0), 3);
    let k = b.input(0, key);
    let table = b.input(1, TraceType::Table);
    b.set_bindings(vec![TableBinding {
        register: 1,
        kind: BindingKind::Array { first: None },
    }]);
    let nil = || Const(TraceScalar::Nil);
    let before = |b: &mut TraceBuilder, pc, registers| {
        let snapshot = b.snapshot(snapshot(SnapshotPhase::Before { pc }, pc, registers));
        b.guard(site(pc), snapshot)
    };
    let guard = before(&mut b, 0, vec![Val(k), Unchanged, nil(), nil()]);
    let next = b
        .emit(
            site(0),
            TraceOp::ArrayNext {
                binding: 0,
                table,
                key: k,
                start,
                ty: result,
                guard,
            },
            Some(result),
        )
        .unwrap();
    let loaded = load.map(|expected| {
        let guard = before(&mut b, 1, vec![Val(k), Unchanged, Val(next), nil()]);
        b.emit(
            site(1),
            TraceOp::ArrayLoad {
                binding: 0,
                table,
                key: next,
                expected,
                past_end: false,
                guard,
            },
            Some(expected),
        )
        .unwrap()
    });
    let last = usize::from(loaded.is_some());
    let exit = b.snapshot(snapshot(
        after(last),
        last,
        vec![Val(k), Unchanged, Val(next), loaded.map_or_else(nil, Val)],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

/// Both backends' code for `ir`, after checking that MIR compiled its own.
fn compile_both(ir: &TraceIr, tier: OptimizationTier) -> (CompiledTrace, CompiledTrace) {
    ir.verify().unwrap();
    let mir = compile(ir, tier, true);
    let cranelift = compile(ir, tier, false);
    assert!(mir.mir_address().is_some(), "compiled by MIR");
    assert!(cranelift.mir_address().is_none());
    assert_eq!(mir.exits(), cranelift.exits());
    (mir, cranelift)
}

fn observe_step(trace: &CompiledTrace, slots: &[Value], key: TraceScalar) -> (Observed, String) {
    let mut slots = slots.to_vec();
    let registers = [
        key,
        TraceScalar::Table(0),
        TraceScalar::Nil,
        TraceScalar::Nil,
    ];
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| match registers[usize::from(*register)] {
            TraceScalar::Table(lane) => TraceInputSlot::table(lane),
            other => TraceInputSlot::from_scalar(other),
        })
        .collect();
    let bindings = [TraceBinding::raw_array(
        slots.len(),
        slots.as_mut_ptr() as usize,
    )];
    let mut area = vec![0x5a5a_i64; trace.exit_area_len()];
    let mut control = TraceControl::new(50);
    let outcome = trace.run_bound(&inputs, &bindings, &mut control, &mut area);
    (
        Observed {
            outcome,
            control,
            area,
        },
        format!("{slots:?}"),
    )
}

fn hole_patterns() -> Vec<Vec<Value>> {
    let nil = || Value::Nil;
    let some = Value::Integer;
    let mut long = vec![nil(); 40];
    long[0] = some(1);
    long[39] = some(40);
    vec![
        vec![],
        vec![nil()],
        vec![some(1)],
        vec![some(1), some(2), some(3), some(4), some(5)],
        vec![nil(), nil(), some(3), nil(), some(5)],
        vec![some(1), some(2), nil(), nil(), nil()],
        vec![nil(), nil(), nil(), nil(), nil(), nil()],
        vec![
            Value::Number(0.5),
            nil(),
            Value::Boolean(false),
            nil(),
            some(5),
        ],
        long,
    ]
}

/// Every key, hole pattern and result type: the step finds the slot after the
/// key (past a run of holes, through the scan), leaves through its guard at a
/// key that names no value or when no value follows, and an `ArrayLoad` of
/// the step's key reads the slot the step found (or leaves through its own
/// guard on the type), all as Cranelift does.
#[test]
fn array_steps_match_cranelift_for_every_key_and_pattern_of_holes() {
    let integer_keys: Vec<TraceScalar> = [-1, 0, 1, 2, 3, 4, 5, 6, 9, 40, 41, i64::MAX, i64::MIN]
        .into_iter()
        .map(TraceScalar::Integer)
        .collect();
    let float_keys: Vec<TraceScalar> = [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        9.0,
        40.0,
        41.0,
        1.5,
        -1.0,
        f64::NAN,
        1e300,
    ]
    .into_iter()
    .map(float)
    .collect();
    let (mut exits, mut passes) = (0, 0);
    for tier in TIERS {
        for load in [
            None,
            Some(TraceType::Integer),
            Some(TraceType::Float),
            Some(TraceType::Boolean),
        ] {
            for result in [TraceType::Integer, TraceType::Float] {
                for (key_ty, keys) in [
                    (TraceType::Integer, &integer_keys),
                    (TraceType::Float, &float_keys),
                ] {
                    let (mir, cranelift) =
                        compile_both(&step_trace(key_ty, false, result, load), tier);
                    for slots in hole_patterns() {
                        for key in keys {
                            let a = observe_step(&mir, &slots, *key);
                            let b = observe_step(&cranelift, &slots, *key);
                            assert_eq!(
                                a, b,
                                "{key:?} in {slots:?} as {result:?} then {load:?} at {tier:?}"
                            );
                            match a.0.outcome {
                                TraceOutcome::Exit { index: 0 } => exits += 1,
                                TraceOutcome::Exit { .. } => passes += 1,
                                other => panic!("{other:?}"),
                            }
                        }
                    }
                }
                // The start of a traversal ignores the key.
                let (mir, cranelift) =
                    compile_both(&step_trace(TraceType::Integer, true, result, load), tier);
                for slots in hole_patterns() {
                    let key = TraceScalar::Integer(0);
                    let a = observe_step(&mir, &slots, key);
                    let b = observe_step(&cranelift, &slots, key);
                    assert_eq!(a, b, "start of {slots:?} as {result:?} then {load:?}");
                    match a.0.outcome {
                        TraceOutcome::Exit { index: 0 } => exits += 1,
                        _ => passes += 1,
                    }
                }
            }
        }
    }
    assert!(exits > 100 && passes > 50, "{exits} exits, {passes} passes");
}
