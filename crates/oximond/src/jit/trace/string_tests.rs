//! Native reads of the bytes of a bound string against the reference
//! evaluator.
//!
//! The VM hands native code the address and length of a string's bytes. These
//! tests do the same over real strings, run the compiled trace at both tiers
//! and require the same exit and the same restored registers as the reference
//! evaluator, which reads the same bytes.

#![allow(clippy::too_many_lines, clippy::float_cmp)]

use super::*;
use crate::jit::OptimizationTier;
use crate::jit_ir::trace::reference::{Outcome, TraceHeap, evaluate_with_heap};
use crate::jit_ir::trace::{
    BindingKind, FrameId, RegisterEntry, Snapshot, SnapshotFrame, SnapshotPhase, TableBinding,
    TraceBuilder, TraceFrameInfo, TraceId, TraceIr, TraceOp, TraceScalar, TraceSite,
    TraceTerminator, TraceType,
};
use crate::runtime::{NativeBytes, Value};

const TIERS: [OptimizationTier; 2] = [OptimizationTier::Baseline, OptimizationTier::Optimized];

fn site(pc: usize) -> TraceSite {
    TraceSite {
        frame: FrameId(0),
        pc,
    }
}

fn snap(phase: SnapshotPhase, last: Option<usize>, registers: Vec<RegisterEntry>) -> Snapshot {
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

/// The reference's view of the string: its bytes.
struct Bytes {
    bytes: Vec<u8>,
    number: TraceType,
}

impl Bytes {
    fn number(&self, value: usize) -> TraceScalar {
        if self.number == TraceType::Float {
            TraceScalar::Float(value as f64)
        } else {
            TraceScalar::Integer(value as i64)
        }
    }
}

impl TraceHeap for Bytes {
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
    fn string_byte(&mut self, _: u32, key: TraceScalar) -> Option<TraceScalar> {
        let index = match key {
            TraceScalar::Integer(index) => index,
            TraceScalar::Float(value) if value.fract() == 0.0 && value.abs() < 1e15 => value as i64,
            _ => return None,
        };
        let slot = usize::try_from(index.checked_sub(1)?).ok()?;
        self.bytes
            .get(slot)
            .map(|byte| self.number(usize::from(*byte)))
    }
    fn string_length(&mut self, _: u32) -> TraceScalar {
        self.number(self.bytes.len())
    }
}

/// `r2 = byte(r1, r0)` and `r3 = #r1`, once. `number` is the profile's number
/// type, which the key has as well.
fn byte_trace(number: TraceType) -> TraceIr {
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
    let key = b.input(0, number);
    let text = b.input(1, TraceType::String);
    b.set_bindings(vec![TableBinding {
        register: 1,
        kind: BindingKind::Bytes { ty: number },
    }]);
    let before = b.snapshot(snap(
        SnapshotPhase::Before { pc: 0 },
        None,
        vec![
            RegisterEntry::Value(key),
            RegisterEntry::Unchanged,
            RegisterEntry::Const(TraceScalar::Nil),
            RegisterEntry::Const(TraceScalar::Nil),
        ],
    ));
    let guard = b.guard(site(0), before);
    let length = b
        .emit(
            site(0),
            TraceOp::StringLength {
                binding: 0,
                string: text,
            },
            Some(number),
        )
        .unwrap();
    let byte = b
        .emit(
            site(0),
            TraceOp::StringByte {
                binding: 0,
                string: text,
                key,
                guard,
            },
            Some(number),
        )
        .unwrap();
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 0,
            successor: 1,
        },
        Some(0),
        vec![
            RegisterEntry::Value(key),
            RegisterEntry::Unchanged,
            RegisterEntry::Value(byte),
            RegisterEntry::Value(length),
        ],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

fn float_keys() -> Vec<TraceScalar> {
    let mut keys = vec![
        0.0,
        -0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        -1.0,
        1.5,
        2.999_999_999_999_999_6,
        0.999_999_999_999_999_9,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        9_223_372_036_854_775_808.0,
        4_503_599_627_370_497.0,
        1e308,
        5e-324,
    ];
    keys.extend((0..4).map(|offset| f64::from(offset) + 0.5));
    keys.into_iter().map(TraceScalar::Float).collect()
}

fn integer_keys() -> Vec<TraceScalar> {
    [0, 1, 2, 3, 4, 5, 6, 7, -1, i64::MIN, i64::MAX, 1 << 40]
        .into_iter()
        .map(TraceScalar::Integer)
        .collect()
}

#[test]
fn bytes_and_lengths_match_the_reference_for_every_key_and_string() {
    let strings: [&[u8]; 4] = [b"", b"a", b"hello", &[0xff, 0x00, 0x80, 0x7f, b'!', 0x01]];
    let mut exits = 0;
    let mut passes = 0;
    for tier in TIERS {
        for number in [TraceType::Float, TraceType::Integer] {
            let keys = if number == TraceType::Float {
                float_keys()
            } else {
                integer_keys()
            };
            let ir = byte_trace(number);
            let trace = compile_trace(&ir, tier).unwrap();
            for text in strings {
                let Value::String(string) = Value::from(text) else {
                    unreachable!()
                };
                for key in &keys {
                    let registers = [
                        *key,
                        TraceScalar::String(0),
                        TraceScalar::Nil,
                        TraceScalar::Nil,
                    ];
                    let inputs: Vec<TraceInputSlot> = trace
                        .entry_contract()
                        .iter()
                        .map(|(register, _)| {
                            TraceInputSlot::from_scalar(registers[usize::from(*register)])
                        })
                        .collect();
                    let bindings = [TraceBinding::bytes(NativeBytes::of(&string))];
                    let mut area = vec![0_i64; trace.exit_area_len()];
                    let mut control = TraceControl::new(100);
                    let outcome = trace.run_bound(&inputs, &bindings, &mut control, &mut area);
                    let TraceOutcome::Exit { index } = outcome else {
                        panic!("the trace refused entry");
                    };
                    let descriptor = &trace.exits()[index];
                    let lowered = trace.lowered_ir();
                    let snapshot = &lowered.snapshots[descriptor.snapshot.index()];
                    let state: Vec<TraceScalar> = snapshot.frames[0]
                        .registers
                        .iter()
                        .enumerate()
                        .map(|(register, entry)| match entry {
                            RegisterEntry::Value(value) => {
                                descriptor.scalar(&area, *value).unwrap_or_else(|| {
                                    // An exit leaves out a register's own entry input when the
                                    // frame keeps it: the restore leaves the register alone.
                                    assert!(
                                        lowered.entry.inputs.iter().any(|input| {
                                            usize::from(input.register) == register
                                                && input.value == *value
                                        }),
                                        "the exit spilled every snapshot value"
                                    );
                                    registers[register]
                                })
                            }
                            RegisterEntry::Const(scalar) => *scalar,
                            RegisterEntry::Unchanged => registers[register],
                            RegisterEntry::Root(_) => unreachable!(),
                            RegisterEntry::Virtual(_) | RegisterEntry::Slice(_) => {
                                unreachable!("no virtual tables here")
                            }
                        })
                        .collect();
                    let mut model = Bytes {
                        bytes: text.to_vec(),
                        number,
                    };
                    let evaluation =
                        evaluate_with_heap(&ir, &registers, &[], 100, None, &mut model, |_, _| {
                            None
                        });
                    let (expected, guard) = match evaluation.outcome {
                        Outcome::GuardFailed { guard, restored } => (restored, Some(guard)),
                        Outcome::Exited(restored) => (restored, None),
                        other => panic!("unexpected reference outcome {other:?}"),
                    };
                    let label = format!("{number:?} key {key:?} in {text:?} at {tier:?}");
                    assert_eq!(descriptor.origin, guard.map(|guard| guard.0), "{label}");
                    assert_eq!(snapshot.resume_point().pc, expected.pc, "{label}");
                    let bits = |scalars: &[TraceScalar]| -> Vec<String> {
                        scalars.iter().map(|scalar| format!("{scalar:?}")).collect()
                    };
                    assert_eq!(bits(&state), bits(&expected.registers), "{label}");
                    exits += usize::from(guard.is_some());
                    passes += usize::from(guard.is_none());
                }
            }
        }
    }
    assert!(exits > 100 && passes > 20, "{exits} exits, {passes} passes");
}

#[test]
fn a_string_input_needs_a_string_register() {
    let ir = byte_trace(TraceType::Float);
    let trace = compile_trace(&ir, OptimizationTier::Optimized).unwrap();
    let Value::String(string) = Value::from("text") else {
        unreachable!()
    };
    let bindings = [TraceBinding::bytes(NativeBytes::of(&string))];
    let mut area = vec![0_i64; trace.exit_area_len()];
    let mut control = TraceControl::new(100);
    // A table lane where the string should be does not enter.
    let inputs = [TraceInputSlot::float(1.0), TraceInputSlot::table(0)];
    assert_eq!(
        trace.run_bound(&inputs, &bindings, &mut control, &mut area),
        TraceOutcome::EntryMismatch { input: 1 }
    );
    let inputs = [TraceInputSlot::float(1.0), TraceInputSlot::string(0)];
    assert!(matches!(
        trace.run_bound(&inputs, &bindings, &mut control, &mut area),
        TraceOutcome::Exit { .. }
    ));
}
