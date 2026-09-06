//! Native traces against the symbolic reference evaluator.
//!
//! Each case compiles a hand-built trace at both tiers, runs it natively and
//! compares what the exit restores with the reference evaluator, bit for bit.

use super::*;
use crate::compat::FloatModulo;
use crate::jit::OptimizationTier;
use crate::jit_ir::Comparison;
use crate::jit_ir::trace::reference::{Outcome, evaluate};
use crate::jit_ir::trace::{
    ArithmeticOp, Entry, FrameId, InlinedFrame, LoopParamInit, NativeBinary, NativeUnary,
    NumericOp, Protected, RegisterEntry, Snapshot, SnapshotFrame, SnapshotPhase, TraceBuilder,
    TraceFrameInfo, TraceId, TraceIr, TraceOp, TraceScalar, TraceSite, TraceTerminator, TraceType,
    ValueId,
};

const TIERS: [OptimizationTier; 2] = [OptimizationTier::Baseline, OptimizationTier::Optimized];

fn site(pc: usize) -> TraceSite {
    TraceSite {
        frame: FrameId(0),
        pc,
    }
}

fn root(registers: u16) -> TraceFrameInfo {
    TraceFrameInfo {
        register_count: registers,
        code_len: 16,
        inlined: None,
        activation: None,
    }
}

fn snapshot(phase: SnapshotPhase, last: usize, registers: Vec<RegisterEntry>) -> Snapshot {
    Snapshot::new(
        phase,
        0,
        vec![SnapshotFrame {
            frame: FrameId(0),
            last_interpreted_pc: Some(last),
            registers: registers.into(),
        }],
    )
}

fn f(value: f64) -> TraceScalar {
    TraceScalar::Float(value)
}

fn slot(scalar: TraceScalar) -> TraceInputSlot {
    TraceInputSlot::from_scalar(scalar)
}

/// Reads a restored register back from an exit.
fn restored(
    trace: &CompiledTrace,
    _ir: &TraceIr,
    index: usize,
    area: &[i64],
    registers: &[TraceScalar],
) -> Vec<TraceScalar> {
    let descriptor = &trace.exits()[index];
    let snapshot = &trace.lowered_ir().snapshots[descriptor.snapshot.index()];
    snapshot.frames[0]
        .registers
        .iter()
        .enumerate()
        .map(|(register, entry)| match entry {
            RegisterEntry::Value(value) => descriptor.scalar(area, *value).unwrap_or_else(|| {
                // An exit leaves out a register's own entry input when the
                // frame keeps it: the restore leaves the register alone.
                assert!(
                    trace.lowered_ir().entry.inputs.iter().any(|input| {
                        usize::from(input.register) == register && input.value == *value
                    }),
                    "the exit spilled every snapshot value"
                );
                registers[register]
            }),
            RegisterEntry::Const(scalar) => *scalar,
            RegisterEntry::Unchanged => registers[register],
            RegisterEntry::Root(_) => unreachable!("no roots in these traces"),
            RegisterEntry::Virtual(_) | RegisterEntry::Slice(_) => {
                unreachable!("no virtual tables here")
            }
        })
        .collect()
}

/// A trace that computes `r2 = op(r0, r1)` once and exits.
fn binary_trace(op: TraceOp, input_ty: TraceType, result_ty: TraceType) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), root(3));
    b.mark_written(FrameId(0), 2);
    let left = b.input(0, input_ty);
    let right = b.input(1, input_ty);
    let op = match op {
        TraceOp::Arithmetic { op, .. } => TraceOp::Arithmetic { op, left, right },
        TraceOp::Compare { comparison, .. } => TraceOp::Compare {
            comparison,
            left,
            right,
        },
        other => other,
    };
    let result = b.emit(site(0), op, Some(result_ty)).unwrap();
    let exit = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 0,
            successor: 1,
        },
        0,
        vec![
            RegisterEntry::Value(left),
            RegisterEntry::Value(right),
            RegisterEntry::Value(result),
        ],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

fn arithmetic(op: ArithmeticOp) -> TraceOp {
    TraceOp::Arithmetic {
        op,
        left: ValueId(0),
        right: ValueId(0),
    }
}

fn compare(comparison: Comparison) -> TraceOp {
    TraceOp::Compare {
        comparison,
        left: ValueId(0),
        right: ValueId(0),
    }
}

fn run_once(
    trace: &CompiledTrace,
    ir: &TraceIr,
    registers: &[TraceScalar],
) -> (TraceOutcome, Vec<TraceScalar>) {
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| slot(registers[usize::from(*register)]))
        .collect();
    let mut area = vec![0_i64; trace.exit_area_len()];
    let mut control = TraceControl::new(1_000);
    let outcome = trace.run(&inputs, &mut control, &mut area);
    let state = match outcome {
        TraceOutcome::Exit { index } => restored(trace, ir, index, &area, registers),
        TraceOutcome::EntryMismatch { .. }
        | TraceOutcome::StoreEntryMissing
        | TraceOutcome::Returned => Vec::new(),
    };
    (outcome, state)
}

const CORNERS: [f64; 27] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    2.5,
    -2.5,
    3.0,
    -7.0,
    0.5,
    1e308,
    5e-324,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    9_007_199_254_740_993.0,
    9_007_199_254_740_991.0,
    -9_007_199_254_740_991.0,
    2.0,
    -6.0,
    6.0,
    7.0,
    -7.0,
    0.5,
    -583.0,
    3.001,
    1e10,
    -1.777_777_777_777_777_7,
];

/// Which float `%` and `^` forms call a helper from native code. The count is
/// of `call` instructions in the CLIF function (a cold block's call counts).
#[test]
fn float_modulo_forms_call_a_helper_only_off_the_inline_path() {
    // Asserts what Cranelift emits.
    crate::jit::debug_env::cranelift_for_this_test();
    let calls = |op: ArithmeticOp| {
        let ir = binary_trace(arithmetic(op), TraceType::Float, TraceType::Float);
        let trace = compile_trace(&ir, OptimizationTier::Optimized).unwrap();
        trace.call_count()
    };
    let baseline = calls(ArithmeticOp::Add);
    // Lua 5.1, 5.2 and LuaJIT: inline floor formula, no call.
    assert_eq!(calls(ArithmeticOp::Modulo), baseline);
    assert_eq!(
        calls(ArithmeticOp::ModuloWith(FloatModulo::Floor)),
        baseline
    );
    // Lua 5.3 and later: the exact paths are inline, one call (in a cold
    // block) for the rest, with or without an FMA path.
    for form in [FloatModulo::Fmod53, FloatModulo::Fmod54] {
        assert_eq!(calls(ArithmeticOp::ModuloWith(form)), baseline + 1);
    }
    // `a ^ 2` is a product inline; other exponents call `pow` once.
    assert_eq!(calls(ArithmeticOp::PowerSquare), baseline + 1);
}

/// The non-integral `%` of Lua 5.3 and later uses an inline `fma` exactly
/// when the target has one, and no other form does.
#[test]
fn float_fmod_uses_fma_only_when_the_isa_has_it() {
    // Asserts what Cranelift emits.
    crate::jit::debug_env::cranelift_for_this_test();
    let fmas = |op: ArithmeticOp| {
        let ir = binary_trace(arithmetic(op), TraceType::Float, TraceType::Float);
        compile_trace(&ir, OptimizationTier::Optimized)
            .unwrap()
            .fma_count()
    };
    #[cfg(target_arch = "x86_64")]
    let host_fma =
        std::arch::is_x86_feature_detected!("fma") && std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(not(target_arch = "x86_64"))]
    let host_fma = cfg!(target_arch = "aarch64");
    for form in [FloatModulo::Fmod53, FloatModulo::Fmod54] {
        // One `fma` per fused remainder; an off-by-one quotient takes the
        // helper instead of a second `fma`.
        assert_eq!(
            fmas(ArithmeticOp::ModuloWith(form)),
            usize::from(host_fma),
            "{form:?}"
        );
    }
    for op in [
        ArithmeticOp::Modulo,
        ArithmeticOp::ModuloWith(FloatModulo::Floor),
        ArithmeticOp::PowerSquare,
        ArithmeticOp::Add,
    ] {
        assert_eq!(fmas(op), 0);
    }
}

/// Native `%` is bit-exact with the interpreter's (libm `fmod` plus the
/// sign fix) on random operands of mixed magnitude, through whichever of the
/// integer, `fma` and helper paths each pair takes.
#[test]
fn float_fmod_native_is_bit_exact_on_random_operands() {
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut next = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let float = move |next: &mut dyn FnMut() -> u64| -> f64 {
        let bits = next();
        match bits & 7 {
            0 => f64::from_bits(next()),
            1 => ((next() >> 11) as i64 - (1 << 52)) as f64,
            2 => (next() % 2001) as f64 - 1000.0,
            3 => (9_007_199_254_740_992.0 + (next() % 9) as f64 - 4.0)
                .copysign(f64::from_bits(next())),
            4 => f64::from_bits(next() & !(0x7ff << 52)) * 1e-300,
            _ => {
                let mantissa = next() & ((1 << 52) - 1);
                let exponent = 1023 - 60 + (next() % 120);
                f64::from_bits((next() & (1 << 63)) | (exponent << 52) | mantissa)
            }
        }
    };
    for form in [FloatModulo::Fmod53, FloatModulo::Fmod54] {
        let ir = binary_trace(
            arithmetic(ArithmeticOp::ModuloWith(form)),
            TraceType::Float,
            TraceType::Float,
        );
        let trace = compile_trace(&ir, OptimizationTier::Optimized).unwrap();
        for _ in 0..20_000 {
            let a = float(&mut next);
            let b = if next() & 1 == 0 {
                float(&mut next)
            } else {
                a / (1.0 + (next() % 1000) as f64) * f64::from_bits(next() | 1)
            };
            // One pair in four puts the quotient next to an integer, where
            // the truncated quotient can be one too large.
            let a = match next() & 7 {
                0 => ((next() % 4001) as f64 - 2000.0) * b,
                1 => (((next() % 4001) as f64 - 2000.0) * b).next_up(),
                2 => (((next() % 4001) as f64 - 2000.0) * b).next_down(),
                _ => a,
            };
            let registers = [f(a), f(b), TraceScalar::Nil];
            let (outcome, native) = run_once(&trace, &ir, &registers);
            assert!(matches!(outcome, TraceOutcome::Exit { .. }));
            let TraceScalar::Float(got) = native[2] else {
                panic!("a float result");
            };
            let expected = form.apply(a, b);
            assert!(
                got.to_bits() == expected.to_bits() || (got.is_nan() && expected.is_nan()),
                "{form:?} {a:e} % {b:e}: native {got:e}, expected {expected:e}"
            );
        }
    }
}

#[test]
fn float_operations_match_the_reference_on_corner_values() {
    let operations: Vec<(&str, TraceOp, TraceType)> = vec![
        ("add", arithmetic(ArithmeticOp::Add), TraceType::Float),
        ("sub", arithmetic(ArithmeticOp::Subtract), TraceType::Float),
        ("mul", arithmetic(ArithmeticOp::Multiply), TraceType::Float),
        ("div", arithmetic(ArithmeticOp::Divide), TraceType::Float),
        ("mod", arithmetic(ArithmeticOp::Modulo), TraceType::Float),
        ("pow", arithmetic(ArithmeticOp::Power), TraceType::Float),
        (
            "mod_floor",
            arithmetic(ArithmeticOp::ModuloWith(FloatModulo::Floor)),
            TraceType::Float,
        ),
        (
            "mod_fmod53",
            arithmetic(ArithmeticOp::ModuloWith(FloatModulo::Fmod53)),
            TraceType::Float,
        ),
        (
            "mod_fmod54",
            arithmetic(ArithmeticOp::ModuloWith(FloatModulo::Fmod54)),
            TraceType::Float,
        ),
        (
            "pow_square",
            arithmetic(ArithmeticOp::PowerSquare),
            TraceType::Float,
        ),
        ("eq", compare(Comparison::Equal), TraceType::Boolean),
        ("lt", compare(Comparison::LessThan), TraceType::Boolean),
        ("le", compare(Comparison::LessEqual), TraceType::Boolean),
    ];
    for tier in TIERS {
        for (name, op, result_ty) in &operations {
            let ir = binary_trace(op.clone(), TraceType::Float, *result_ty);
            let trace = compile_trace(&ir, tier).unwrap();
            for &a in &CORNERS {
                for &b in &CORNERS {
                    let registers = [f(a), f(b), TraceScalar::Nil];
                    let (outcome, native) = run_once(&trace, &ir, &registers);
                    assert!(matches!(outcome, TraceOutcome::Exit { .. }));
                    let reference = evaluate(&ir, &registers, &[], 1, |_, _| None);
                    let Outcome::Exited(expected) = reference.outcome else {
                        panic!("the reference exits");
                    };
                    assert_eq!(
                        native, expected.registers,
                        "{name}({a:?}, {b:?}) at {tier:?}"
                    );
                }
            }
        }
    }
}

extern "C" fn sine(value: f64) -> f64 {
    value.sin()
}

extern "C" fn larger(left: f64, right: f64) -> f64 {
    if left > right { left } else { right }
}

/// A pure native function of one float, or two, runs like the reference
/// calls it, at both tiers and with a chain in front of it and behind it.
#[test]
fn native_calls_match_the_reference_on_corner_values() {
    for tier in TIERS {
        // r2 = larger(sine(r0 + r1), r1) - r0
        let mut b = TraceBuilder::new(TraceId(0), root(3));
        b.mark_written(FrameId(0), 2);
        let left = b.input(0, TraceType::Float);
        let right = b.input(1, TraceType::Float);
        let sum = b
            .emit(
                site(0),
                TraceOp::Arithmetic {
                    op: ArithmeticOp::Add,
                    left,
                    right,
                },
                Some(TraceType::Float),
            )
            .unwrap();
        let wave = b
            .emit(
                site(1),
                TraceOp::NativeUnary {
                    function: NativeUnary(sine),
                    source: sum,
                },
                Some(TraceType::Float),
            )
            .unwrap();
        let top = b
            .emit(
                site(2),
                TraceOp::NativeBinary {
                    function: NativeBinary(larger),
                    left: wave,
                    right,
                },
                Some(TraceType::Float),
            )
            .unwrap();
        let result = b
            .emit(
                site(3),
                TraceOp::Arithmetic {
                    op: ArithmeticOp::Subtract,
                    left: top,
                    right: left,
                },
                Some(TraceType::Float),
            )
            .unwrap();
        let exit = b.snapshot(snapshot(
            SnapshotPhase::After {
                pc: 3,
                successor: 4,
            },
            3,
            vec![
                RegisterEntry::Value(left),
                RegisterEntry::Value(right),
                RegisterEntry::Value(result),
            ],
        ));
        let ir = b.finish(TraceTerminator::Exit { snapshot: exit });
        let trace = compile_trace(&ir, tier).unwrap();
        for &a in &CORNERS {
            for &c in &CORNERS {
                let registers = [f(a), f(c), TraceScalar::Nil];
                let (outcome, native) = run_once(&trace, &ir, &registers);
                assert!(matches!(outcome, TraceOutcome::Exit { .. }));
                let Outcome::Exited(expected) =
                    evaluate(&ir, &registers, &[], 1, |_, _| None).outcome
                else {
                    panic!("the reference exits");
                };
                assert_eq!(native, expected.registers, "({a:?}, {c:?}) at {tier:?}");
            }
        }
    }
}

#[test]
fn negate_truthy_and_not_match_the_reference() {
    for tier in TIERS {
        for (ty, values) in [
            (TraceType::Float, vec![f(0.0), f(-0.0), f(f64::NAN), f(2.0)]),
            (
                TraceType::Boolean,
                vec![TraceScalar::Boolean(false), TraceScalar::Boolean(true)],
            ),
            (TraceType::Nil, vec![TraceScalar::Nil]),
        ] {
            for case in 0..3 {
                let mut b = TraceBuilder::new(TraceId(0), root(2));
                b.mark_written(FrameId(0), 1);
                let source = b.input(0, ty);
                let (op, result_ty) = match case {
                    0 if ty == TraceType::Float => (TraceOp::Negate { source }, TraceType::Float),
                    0 | 1 => (TraceOp::Truthy { source }, TraceType::Boolean),
                    _ => (TraceOp::Not { source }, TraceType::Boolean),
                };
                let result = b.emit(site(0), op, Some(result_ty)).unwrap();
                let exit = b.snapshot(snapshot(
                    SnapshotPhase::After {
                        pc: 0,
                        successor: 1,
                    },
                    0,
                    vec![RegisterEntry::Value(source), RegisterEntry::Value(result)],
                ));
                let ir = b.finish(TraceTerminator::Exit { snapshot: exit });
                let trace = compile_trace(&ir, tier).unwrap();
                for &value in &values {
                    let registers = [value, TraceScalar::Nil];
                    let (_, native) = run_once(&trace, &ir, &registers);
                    let Outcome::Exited(expected) =
                        evaluate(&ir, &registers, &[], 1, |_, _| None).outcome
                    else {
                        panic!("the reference exits");
                    };
                    assert_eq!(native, expected.registers, "{ty:?} case {case} {value:?}");
                }
            }
        }
    }
}

#[test]
fn integer_operations_wrap_like_the_reference() {
    let values = [0, 1, -1, i64::MAX, i64::MIN, 12_345_678_901, -7];
    for tier in TIERS {
        for op in [
            arithmetic(ArithmeticOp::Add),
            arithmetic(ArithmeticOp::Subtract),
            arithmetic(ArithmeticOp::Multiply),
            compare(Comparison::Equal),
            compare(Comparison::LessThan),
            compare(Comparison::LessEqual),
        ] {
            let result_ty = if matches!(op, TraceOp::Arithmetic { .. }) {
                TraceType::Integer
            } else {
                TraceType::Boolean
            };
            let ir = binary_trace(op, TraceType::Integer, result_ty);
            let trace = compile_trace(&ir, tier).unwrap();
            for &a in &values {
                for &b in &values {
                    let registers = [
                        TraceScalar::Integer(a),
                        TraceScalar::Integer(b),
                        TraceScalar::Nil,
                    ];
                    let (_, native) = run_once(&trace, &ir, &registers);
                    let Outcome::Exited(expected) =
                        evaluate(&ir, &registers, &[], 1, |_, _| None).outcome
                    else {
                        panic!("the reference exits");
                    };
                    assert_eq!(native, expected.registers, "{a} op {b}");
                }
            }
        }
    }
}

#[test]
fn entry_checks_every_input_type_before_running_anything() {
    let ir = binary_trace(
        arithmetic(ArithmeticOp::Add),
        TraceType::Float,
        TraceType::Float,
    );
    for tier in TIERS {
        let trace = compile_trace(&ir, tier).unwrap();
        assert_eq!(
            trace.entry_contract(),
            &[(0, TraceType::Float), (1, TraceType::Float)]
        );
        let mut area = vec![0_i64; trace.exit_area_len()];
        let cases = [
            (
                [slot(f(1.0)), TraceInputSlot::other()],
                TraceOutcome::EntryMismatch { input: 1 },
            ),
            (
                [TraceInputSlot::nil(), slot(f(1.0))],
                TraceOutcome::EntryMismatch { input: 0 },
            ),
            (
                [TraceInputSlot::integer(1), slot(f(1.0))],
                TraceOutcome::EntryMismatch { input: 0 },
            ),
            (
                [slot(f(1.0)), TraceInputSlot::boolean(true)],
                TraceOutcome::EntryMismatch { input: 1 },
            ),
        ];
        for (inputs, expected) in cases {
            // A mismatch leaves the control block and exit area untouched.
            area.fill(-1);
            let mut control = TraceControl::new(77);
            assert_eq!(trace.run(&inputs, &mut control, &mut area), expected);
            assert_eq!(control.fuel, 77);
            assert!(area.iter().all(|word| *word == -1));
        }
    }
}

/// `x = x + 1` until `x > limit`: r0 x, r1 limit. The header is pc 0, the
/// closing instruction pc 1.
fn counting_loop() -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), root(2));
    b.mark_written(FrameId(0), 0);
    let x_in = b.input(0, TraceType::Float);
    let limit = b.input(1, TraceType::Float);
    let x = b.loop_param(TraceType::Float, LoopParamInit::Input(x_in));
    let one = b
        .emit(site(0), TraceOp::Constant(f(1.0)), Some(TraceType::Float))
        .unwrap();
    let next = b
        .emit(
            site(0),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: x,
                right: one,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let keep_going = b
        .emit(
            site(1),
            TraceOp::Compare {
                comparison: Comparison::LessEqual,
                left: next,
                right: limit,
            },
            Some(TraceType::Boolean),
        )
        .unwrap();
    // Leaving: the loop ended, resume after the latch with x advanced.
    let leave = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 1,
            successor: 2,
        },
        1,
        vec![RegisterEntry::Value(next), RegisterEntry::Unchanged],
    ));
    let guard = b.guard(site(1), leave);
    b.emit(
        site(1),
        TraceOp::GuardBranch {
            condition: keep_going,
            expected: true,
            guard,
        },
        None,
    );
    let poll = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 1,
            successor: 0,
        },
        1,
        vec![RegisterEntry::Value(next), RegisterEntry::Unchanged],
    ));
    b.set_poll_snapshot(poll);
    b.finish(TraceTerminator::LoopBack { args: vec![next] })
}

#[test]
fn a_loop_runs_natively_and_leaves_through_its_guard() {
    let ir = counting_loop();
    ir.verify().unwrap();
    for tier in TIERS {
        let trace = compile_trace(&ir, tier).unwrap();
        assert_eq!(trace.tier(), tier);
        assert!(trace.code_size() > 0);
        let classes: Vec<_> = trace.exits().iter().map(|exit| exit.class).collect();
        assert_eq!(
            classes,
            [TraceExitClass::LoopComplete, TraceExitClass::Poll]
        );
        let registers = [f(0.0), f(10.0)];
        let mut area = vec![0; trace.exit_area_len()];
        let mut control = TraceControl::new(1_000);
        let inputs: Vec<_> = registers.iter().map(|r| slot(*r)).collect();
        let outcome = trace.run(&inputs, &mut control, &mut area);
        assert_eq!(outcome, TraceOutcome::Exit { index: 0 });
        // Ten iterations succeed, the eleventh fails the guard: x was
        // advanced to 11 before the test.
        assert_eq!(control.fuel, 1_000 - 10);
        assert_eq!(
            restored(&trace, &ir, 0, &area, &registers),
            vec![f(11.0), f(10.0)]
        );
    }
}

#[test]
fn the_poll_exit_publishes_the_state_between_iterations() {
    let ir = counting_loop();
    for tier in TIERS {
        let trace = compile_trace(&ir, tier).unwrap();
        let registers = [f(0.0), f(1.0e9)];
        let inputs: Vec<_> = registers.iter().map(|r| slot(*r)).collect();
        let mut area = vec![0; trace.exit_area_len()];
        for fuel in [1_i64, 2, 5, 64] {
            let mut control = TraceControl::new(fuel);
            let outcome = trace.run(&inputs, &mut control, &mut area);
            assert_eq!(outcome, TraceOutcome::Exit { index: 1 }, "fuel {fuel}");
            // Each loop-back costs one unit; the poll fires when none is left.
            assert_eq!(control.fuel, 0);
            // `fuel` iterations completed, so x advanced that many times.
            assert_eq!(
                restored(&trace, &ir, 1, &area, &registers),
                vec![f(fuel as f64), f(1.0e9)],
                "fuel {fuel} at {tier:?}"
            );
        }
        // Resuming from the published state continues exactly.
        let resumed = [f(5.0), f(1.0e9)];
        let inputs: Vec<_> = resumed.iter().map(|r| slot(*r)).collect();
        let mut control = TraceControl::new(3);
        trace.run(&inputs, &mut control, &mut area);
        assert_eq!(
            restored(&trace, &ir, 1, &area, &resumed),
            vec![f(8.0), f(1.0e9)]
        );
    }
}

/// [`counting_loop`] as a counted trace: an iteration runs four counted guest
/// instructions, and the guard leaves after three of them.
fn counted_counting_loop() -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), root(2));
    b.set_counted(true);
    b.mark_written(FrameId(0), 0);
    let x_in = b.input(0, TraceType::Float);
    let limit = b.input(1, TraceType::Float);
    let x = b.loop_param(TraceType::Float, LoopParamInit::Input(x_in));
    let one = b
        .emit(site(0), TraceOp::Constant(f(1.0)), Some(TraceType::Float))
        .unwrap();
    let next = b
        .emit(
            site(0),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: x,
                right: one,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let keep_going = b
        .emit(
            site(1),
            TraceOp::Compare {
                comparison: Comparison::LessEqual,
                left: next,
                right: limit,
            },
            Some(TraceType::Boolean),
        )
        .unwrap();
    let leave = b.snapshot(
        snapshot(
            SnapshotPhase::After {
                pc: 1,
                successor: 2,
            },
            1,
            vec![RegisterEntry::Value(next), RegisterEntry::Unchanged],
        )
        .with_counted(3),
    );
    let guard = b.guard(site(1), leave);
    b.emit(
        site(1),
        TraceOp::GuardBranch {
            condition: keep_going,
            expected: true,
            guard,
        },
        None,
    );
    let poll = b.snapshot(
        snapshot(
            SnapshotPhase::After {
                pc: 1,
                successor: 0,
            },
            1,
            vec![RegisterEntry::Value(next), RegisterEntry::Unchanged],
        )
        .with_counted(4),
    );
    b.set_poll_snapshot(poll);
    b.finish(TraceTerminator::LoopBack { args: vec![next] })
}

#[test]
fn a_counted_loop_pays_for_each_iteration_it_completes() {
    let ir = counted_counting_loop();
    ir.verify().unwrap();
    for tier in TIERS {
        let trace = compile_trace(&ir, tier).unwrap();
        assert!(trace.counted());
        assert_eq!(trace.max_counted(), 4);
        let counted: Vec<_> = trace.exits().iter().map(|exit| exit.counted).collect();
        assert_eq!(counted, [3, 4]);
        let registers = [f(0.0), f(10.0)];
        let inputs: Vec<_> = registers.iter().map(|r| slot(*r)).collect();
        let mut area = vec![0; trace.exit_area_len()];
        // Ten iterations complete, and the eleventh leaves after three of its
        // four instructions: the exit reports the budget the eleventh began
        // with, and its descriptor the three it ran.
        let mut control = TraceControl::new(1_000);
        control.budget = 1_000;
        let outcome = trace.run(&inputs, &mut control, &mut area);
        assert_eq!(outcome, TraceOutcome::Exit { index: 0 });
        assert_eq!(control.budget, 1_000 - 10 * 4);
        // The poll leaves between iterations, after `fuel` of them.
        let registers = [f(0.0), f(1.0e9)];
        let inputs: Vec<_> = registers.iter().map(|r| slot(*r)).collect();
        for fuel in [1_i64, 2, 5, 64] {
            let mut control = TraceControl::new(fuel);
            control.budget = 1_000_000;
            let outcome = trace.run(&inputs, &mut control, &mut area);
            assert_eq!(outcome, TraceOutcome::Exit { index: 1 });
            let left = control.budget - i64::from(trace.exits()[1].counted);
            assert_eq!(left, 1_000_000 - fuel * 4, "fuel {fuel} at {tier:?}");
        }
    }
}

#[test]
fn a_trace_that_is_not_counted_leaves_the_budget_alone() {
    let ir = counting_loop();
    let trace = compile_trace(&ir, OptimizationTier::Optimized).unwrap();
    assert!(!trace.counted());
    let registers = [f(0.0), f(10.0)];
    let inputs: Vec<_> = registers.iter().map(|r| slot(*r)).collect();
    let mut area = vec![0; trace.exit_area_len()];
    let mut control = TraceControl::new(1_000);
    control.budget = 77;
    trace.run(&inputs, &mut control, &mut area);
    assert_eq!(control.budget, 77);
}

#[test]
fn the_loop_matches_the_reference_for_many_limits() {
    let ir = counting_loop();
    for tier in TIERS {
        let trace = compile_trace(&ir, tier).unwrap();
        for start in [-3.0, 0.0, 0.5, 7.0] {
            for limit in [-5.0, 0.0, 1.0, 2.5, 100.0, f64::NAN, -f64::INFINITY] {
                let registers = [f(start), f(limit)];
                let (outcome, native) = run_once(&trace, &ir, &registers);
                assert_eq!(outcome, TraceOutcome::Exit { index: 0 });
                let Outcome::GuardFailed {
                    restored: expected, ..
                } = evaluate(&ir, &registers, &[], 10_000, |_, _| None).outcome
                else {
                    panic!("the reference leaves through the guard");
                };
                assert_eq!(native, expected.registers, "{start} .. {limit}");
            }
        }
    }
}

/// `a, b = b, a` each iteration while a counter counts down: the loop-back is
/// a parallel assignment. r0 a, r1 b, r2 counter.
fn swap_loop() -> TraceIr {
    swap_loop_of(TraceType::Float)
}

/// [`swap_loop`] over values of type `ty`.
fn swap_loop_of(ty: TraceType) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), root(3));
    for register in 0..3 {
        b.mark_written(FrameId(0), register);
    }
    let a_in = b.input(0, ty);
    let b_in = b.input(1, ty);
    let n_in = b.input(2, TraceType::Float);
    let a = b.loop_param(ty, LoopParamInit::Input(a_in));
    let bb = b.loop_param(ty, LoopParamInit::Input(b_in));
    let n = b.loop_param(TraceType::Float, LoopParamInit::Input(n_in));
    let one = b
        .emit(site(0), TraceOp::Constant(f(1.0)), Some(TraceType::Float))
        .unwrap();
    let next = b
        .emit(
            site(0),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Subtract,
                left: n,
                right: one,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let zero = b
        .emit(site(1), TraceOp::Constant(f(0.0)), Some(TraceType::Float))
        .unwrap();
    let more = b
        .emit(
            site(1),
            TraceOp::Compare {
                comparison: Comparison::LessThan,
                left: zero,
                right: next,
            },
            Some(TraceType::Boolean),
        )
        .unwrap();
    // Leaving after the swap of this iteration.
    let leave = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 1,
            successor: 2,
        },
        1,
        vec![
            RegisterEntry::Value(bb),
            RegisterEntry::Value(a),
            RegisterEntry::Value(next),
        ],
    ));
    let guard = b.guard(site(1), leave);
    b.emit(
        site(1),
        TraceOp::GuardBranch {
            condition: more,
            expected: true,
            guard,
        },
        None,
    );
    let poll = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 1,
            successor: 0,
        },
        1,
        vec![
            RegisterEntry::Value(bb),
            RegisterEntry::Value(a),
            RegisterEntry::Value(next),
        ],
    ));
    b.set_poll_snapshot(poll);
    b.finish(TraceTerminator::LoopBack {
        args: vec![bb, a, next],
    })
}

#[test]
fn a_loop_carried_swap_is_a_parallel_assignment() {
    let ir = swap_loop();
    ir.verify().unwrap();
    for tier in TIERS {
        let trace = compile_trace(&ir, tier).unwrap();
        for count in [1.0, 2.0, 3.0, 8.0, 101.0] {
            let registers = [f(1.5), f(-2.5), f(count)];
            let (outcome, native) = run_once(&trace, &ir, &registers);
            assert_eq!(outcome, TraceOutcome::Exit { index: 0 });
            let Outcome::GuardFailed {
                restored: expected, ..
            } = evaluate(&ir, &registers, &[], 10_000, |_, _| None).outcome
            else {
                panic!("the reference leaves through the guard");
            };
            assert_eq!(native, expected.registers, "{count} iterations");
            // Odd and even counts swap or restore the pair.
            let swapped = (count as u64) % 2 == 1;
            let pair = if swapped {
                [f(-2.5), f(1.5)]
            } else {
                [f(1.5), f(-2.5)]
            };
            assert_eq!(&native[..2], &pair, "{count}");
        }
    }
}

#[test]
fn loop_parameters_with_a_constant_start_begin_at_the_constant() {
    let mut b = TraceBuilder::new(TraceId(0), root(2));
    b.mark_written(FrameId(0), 0);
    let flag_in = b.input(1, TraceType::Boolean);
    let c = b.loop_param(TraceType::Float, LoopParamInit::Const(f(40.0)));
    let one = b
        .emit(site(0), TraceOp::Constant(f(1.0)), Some(TraceType::Float))
        .unwrap();
    let next = b
        .emit(
            site(0),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: c,
                right: one,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let guard_snapshot = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 1,
            successor: 2,
        },
        1,
        vec![RegisterEntry::Value(next), RegisterEntry::Unchanged],
    ));
    let guard = b.guard(site(1), guard_snapshot);
    b.emit(
        site(1),
        TraceOp::GuardBranch {
            condition: flag_in,
            expected: true,
            guard,
        },
        None,
    );
    let poll = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 1,
            successor: 0,
        },
        1,
        vec![RegisterEntry::Value(next), RegisterEntry::Unchanged],
    ));
    b.set_poll_snapshot(poll);
    let ir = b.finish(TraceTerminator::LoopBack { args: vec![next] });
    ir.verify().unwrap();
    let trace = compile_trace(&ir, OptimizationTier::Baseline).unwrap();
    let registers = [TraceScalar::Nil, TraceScalar::Boolean(true)];
    // Only register 1 is an entry input.
    assert_eq!(trace.entry_contract(), &[(1, TraceType::Boolean)]);
    let inputs = [slot(TraceScalar::Boolean(true))];
    let mut area = vec![0; trace.exit_area_len()];
    let mut control = TraceControl::new(3);
    let outcome = trace.run(&inputs, &mut control, &mut area);
    assert_eq!(outcome, TraceOutcome::Exit { index: 1 });
    assert_eq!(restored(&trace, &ir, 1, &area, &registers)[0], f(43.0));
}

#[test]
fn guard_exits_record_their_class() {
    let ir = counting_loop();
    let trace = compile_trace(&ir, OptimizationTier::Baseline).unwrap();
    assert_eq!(trace.exits()[0].guard, Some(TraceGuardClass::Branch));
    assert_eq!(trace.exits()[1].guard, None);
}

#[test]
fn unsupported_and_invalid_traces_are_rejected() {
    // An effect has no native lowering yet.
    let mut b = TraceBuilder::new(TraceId(0), root(1));
    let input = b.input(0, TraceType::Float);
    let effect = b.next_effect();
    b.emit(
        site(0),
        TraceOp::Effect {
            effect,
            label: 1,
            operands: vec![input],
            result: None,
        },
        None,
    );
    let exit = b.snapshot(Snapshot::new(
        SnapshotPhase::After {
            pc: 0,
            successor: 1,
        },
        1,
        vec![SnapshotFrame {
            frame: FrameId(0),
            last_interpreted_pc: Some(0),
            registers: vec![RegisterEntry::Value(input)].into(),
        }],
    ));
    let ir = b.finish(TraceTerminator::Exit { snapshot: exit });
    ir.verify().unwrap();
    assert!(matches!(
        compile_trace(&ir, OptimizationTier::Baseline),
        Err(TraceCompileError::Unsupported { what: "effect", .. })
    ));

    // A loop with no poll snapshot cannot be stopped between iterations.
    let mut ir = counting_loop();
    ir.snapshots.pop();
    ir.poll_snapshot = None;
    ir.verify().unwrap();
    assert_eq!(
        compile_trace(&ir, OptimizationTier::Baseline).unwrap_err(),
        TraceCompileError::MissingPollSnapshot
    );

    // An unverifiable trace never reaches the backend.
    let mut ir = counting_loop();
    ir.loop_params.clear();
    assert!(matches!(
        compile_trace(&ir, OptimizationTier::Baseline),
        Err(TraceCompileError::Invalid(_))
    ));
}

#[test]
fn compiled_traces_move_between_threads() {
    let ir = counting_loop();
    let trace = compile_trace(&ir, OptimizationTier::Baseline).unwrap();
    let handle = std::thread::spawn(move || {
        let registers = [f(0.0), f(3.0)];
        let inputs: Vec<_> = registers.iter().map(|r| slot(*r)).collect();
        let mut area = vec![0; trace.exit_area_len()];
        let mut control = TraceControl::new(100);
        trace.run(&inputs, &mut control, &mut area)
    });
    assert_eq!(handle.join().unwrap(), TraceOutcome::Exit { index: 0 });
}

// ---- optimizer ------------------------------------------------------------

/// A loop with invariant work: r0 sum (carried), r1 and r2 invariant inputs.
///
/// ```text
/// pc 1: t = r1 * r2; u = t + 1.0     invariant
/// pc 3: sum = sum + u                carried
/// pc 4: c = r1 < r2                  invariant condition
/// pc 5: guard c
/// ```
fn invariant_loop() -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), root(3));
    b.mark_written(FrameId(0), 0);
    let sum_in = b.input(0, TraceType::Float);
    let r1 = b.input(1, TraceType::Float);
    let r2 = b.input(2, TraceType::Float);
    let sum = b.loop_param(TraceType::Float, LoopParamInit::Input(sum_in));
    let mut emit = |pc, op, ty| b.emit(site(pc), op, Some(ty)).unwrap();
    let t = emit(
        1,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Multiply,
            left: r1,
            right: r2,
        },
        TraceType::Float,
    );
    let one = emit(2, TraceOp::Constant(f(1.0)), TraceType::Float);
    let u = emit(
        2,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: t,
            right: one,
        },
        TraceType::Float,
    );
    let next = emit(
        3,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: sum,
            right: u,
        },
        TraceType::Float,
    );
    let c = emit(
        4,
        TraceOp::Compare {
            comparison: Comparison::LessThan,
            left: r1,
            right: r2,
        },
        TraceType::Boolean,
    );
    let regs = |value| {
        vec![
            RegisterEntry::Value(value),
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
        ]
    };
    let fail = b.snapshot(snapshot(SnapshotPhase::Before { pc: 5 }, 4, regs(next)));
    let guard = b.guard(site(5), fail);
    b.emit(
        site(5),
        TraceOp::GuardBranch {
            condition: c,
            expected: true,
            guard,
        },
        None,
    );
    let poll = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 6,
            successor: 1,
        },
        6,
        regs(next),
    ));
    b.set_poll_snapshot(poll);
    b.finish(TraceTerminator::LoopBack { args: vec![next] })
}

#[test]
fn hoisted_invariant_code_runs_once_and_the_loop_stays_exact() {
    let ir = invariant_loop();
    ir.verify().unwrap();
    let baseline = compile_trace(&ir, OptimizationTier::Baseline).unwrap();
    assert_eq!(
        baseline.lowered_ir().preheader,
        [] as [crate::jit_ir::trace::TraceInstruction; 0]
    );
    let optimized = compile_trace(&ir, OptimizationTier::Optimized).unwrap();
    assert_eq!(optimized.lowered_ir().preheader.len(), 5);
    assert_eq!(optimized.lowered_ir().body.len(), 1);

    for trace in [&baseline, &optimized] {
        // The guard passes: r1 < r2. Each iteration adds r1 * r2 + 1.
        for (sum0, r1, r2) in [
            (0.0, 2.0, 3.0),
            (-0.0, 0.5, f64::INFINITY),
            (1.5, -1.0, 4.0),
        ] {
            let registers = [f(sum0), f(r1), f(r2)];
            let inputs: Vec<_> = registers.iter().map(|r| slot(*r)).collect();
            let mut area = vec![0; trace.exit_area_len()];
            for fuel in [1_i64, 3, 17] {
                let mut control = TraceControl::new(fuel);
                let outcome = trace.run(&inputs, &mut control, &mut area);
                let poll = trace.exits().len() - 1;
                assert_eq!(outcome, TraceOutcome::Exit { index: poll });
                let mut expected = sum0;
                for _ in 0..fuel {
                    expected += r1 * r2 + 1.0;
                }
                let state = restored(trace, &ir, poll, &area, &registers);
                assert_eq!(state[0], f(expected), "fuel {fuel} {sum0} {r1} {r2}");
            }
        }
    }
}

#[test]
fn a_failing_hoisted_guard_leaves_before_any_iteration_with_the_entry_state() {
    let ir = invariant_loop();
    let baseline = compile_trace(&ir, OptimizationTier::Baseline).unwrap();
    let optimized = compile_trace(&ir, OptimizationTier::Optimized).unwrap();
    // r1 >= r2 fails the guard. NaN fails the ordered comparison too.
    for (r1, r2) in [(3.0, 2.0), (2.0, 2.0), (f64::NAN, 1.0), (1.0, f64::NAN)] {
        let registers = [f(10.0), f(r1), f(r2)];
        let inputs: Vec<_> = registers.iter().map(|r| slot(*r)).collect();

        let mut area = vec![0; baseline.exit_area_len()];
        let mut control = TraceControl::new(100);
        let outcome = baseline.run(&inputs, &mut control, &mut area);
        assert_eq!(outcome, TraceOutcome::Exit { index: 0 });
        // The baseline runs the first iteration up to the guard.
        let after_one = restored(&baseline, &ir, 0, &area, &registers);
        assert_eq!(after_one[0], f(10.0 + (r1 * r2 + 1.0)));

        let mut area = vec![0; optimized.exit_area_len()];
        let mut control = TraceControl::new(100);
        let outcome = optimized.run(&inputs, &mut control, &mut area);
        assert_eq!(outcome, TraceOutcome::Exit { index: 0 });
        assert_eq!(control.fuel, 100, "no iteration ran");
        let descriptor = &optimized.exits()[0];
        assert_eq!(descriptor.class, TraceExitClass::GuardFail);
        assert!(
            descriptor.slots.is_empty(),
            "the entry state needs no spill"
        );
        let state = restored(&optimized, &ir, 0, &area, &registers);
        assert_eq!(state, registers.to_vec());
        let snapshot = &optimized.lowered_ir().snapshots[descriptor.snapshot.index()];
        assert_eq!(snapshot.resume_point().pc, 1, "the loop header");
    }
}

#[test]
fn folded_constants_match_unfolded_native_code_and_the_reference_bit_for_bit() {
    let values = [
        0.0,
        -0.0,
        1.0,
        -2.5,
        3.0,
        0.5,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        1e308,
    ];
    let mut b = TraceBuilder::new(TraceId(0), root(1));
    let constants: Vec<ValueId> = values
        .iter()
        .map(|v| {
            b.emit(site(0), TraceOp::Constant(f(*v)), Some(TraceType::Float))
                .unwrap()
        })
        .collect();
    let mut results = Vec::new();
    for op in [
        ArithmeticOp::Add,
        ArithmeticOp::Subtract,
        ArithmeticOp::Multiply,
        ArithmeticOp::Divide,
        ArithmeticOp::Modulo,
        ArithmeticOp::Power,
    ] {
        for left in &constants {
            for right in &constants {
                results.push(
                    b.emit(
                        site(1),
                        TraceOp::Arithmetic {
                            op,
                            left: *left,
                            right: *right,
                        },
                        Some(TraceType::Float),
                    )
                    .unwrap(),
                );
            }
        }
    }
    for source in &constants {
        results.push(
            b.emit(
                site(2),
                TraceOp::Negate { source: *source },
                Some(TraceType::Float),
            )
            .unwrap(),
        );
    }
    for comparison in [
        Comparison::Equal,
        Comparison::LessThan,
        Comparison::LessEqual,
    ] {
        for left in &constants {
            for right in &constants {
                results.push(
                    b.emit(
                        site(3),
                        TraceOp::Compare {
                            comparison,
                            left: *left,
                            right: *right,
                        },
                        Some(TraceType::Boolean),
                    )
                    .unwrap(),
                );
            }
        }
    }
    let mut ir = b.finish(TraceTerminator::Exit {
        snapshot: crate::jit_ir::trace::SnapshotId(0),
    });
    ir.frames[0].register_count = u16::try_from(results.len()).unwrap();
    for register in 0..results.len() {
        ir.written_registers
            .push((FrameId(0), u16::try_from(register).unwrap()));
    }
    ir.snapshots.push(Snapshot {
        id: crate::jit_ir::trace::SnapshotId(0),
        ..snapshot(
            SnapshotPhase::Before { pc: 4 },
            0,
            results.iter().map(|v| RegisterEntry::Value(*v)).collect(),
        )
    });
    ir.verify().unwrap();

    let registers = vec![TraceScalar::Nil; results.len()];
    let reference = {
        let Outcome::Exited(restored) = evaluate(&ir, &registers, &[], 1, |_, _| None).outcome
        else {
            panic!("the reference exits");
        };
        restored.registers
    };
    let baseline = compile_trace(&ir, OptimizationTier::Baseline).unwrap();
    let optimized = compile_trace(&ir, OptimizationTier::Optimized).unwrap();
    assert!(
        optimized.lowered_ir().body.is_empty()
            || optimized
                .lowered_ir()
                .body
                .iter()
                .all(|i| matches!(i.op, TraceOp::Constant(_))),
        "everything folded away"
    );
    // The optimized exit spills nothing: the snapshot carries constants.
    assert_eq!(optimized.exit_area_len(), 0);
    for trace in [&baseline, &optimized] {
        let (_, native) = run_once(trace, &ir, &registers);
        assert_eq!(native, reference);
    }
}

/// A small deterministic generator for the seeded test below.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % bound as u64).unwrap()
    }
}

/// A numeric loop over r0 and r1 (carried) and r2 and r3 (never written, so
/// loop invariant). Operands come from a pool that holds the parameters, the
/// invariant inputs, constants and earlier results, which lets the optimizer
/// hoist arithmetic and branch guards into a preheader. Guard snapshots name
/// values from the pool, hoisted ones included.
fn generated_numeric_loop(seed: u64) -> TraceIr {
    generated_loop(seed, false)
}

/// [`generated_numeric_loop`], optionally with calls of pure native functions
/// among its operations.
fn generated_loop(seed: u64, natives: bool) -> TraceIr {
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut b = TraceBuilder::new(TraceId(0), root(4));
    b.mark_written(FrameId(0), 0);
    b.mark_written(FrameId(0), 1);
    let in0 = b.input(0, TraceType::Float);
    let in1 = b.input(1, TraceType::Float);
    let in2 = b.input(2, TraceType::Float);
    let in3 = b.input(3, TraceType::Float);
    let p0 = b.loop_param(TraceType::Float, LoopParamInit::Input(in0));
    let p1 = b.loop_param(TraceType::Float, LoopParamInit::Input(in1));
    let mut pool = vec![p0, p1, in2, in3, in2, in3];
    let steps = 4 + rng.below(6);
    let mut pc = 0;
    let mut guards = 0;
    for _ in 0..steps {
        pc += 1;
        let pick = |rng: &mut Rng, pool: &[ValueId]| pool[rng.below(pool.len())];
        match rng.below(if natives { 10 } else { 8 }) {
            // A pure native function of one float or two.
            8 | 9 => {
                let (left, right) = (pick(&mut rng, &pool), pick(&mut rng, &pool));
                let op = if rng.below(2) == 0 {
                    TraceOp::NativeUnary {
                        function: NativeUnary(sine),
                        source: left,
                    }
                } else {
                    TraceOp::NativeBinary {
                        function: NativeBinary(larger),
                        left,
                        right,
                    }
                };
                let value = b.emit(site(pc), op, Some(TraceType::Float)).unwrap();
                pool.push(value);
            }
            0 => {
                let scalar = f(CORNERS[rng.below(CORNERS.len())]);
                let value = b
                    .emit(site(pc), TraceOp::Constant(scalar), Some(TraceType::Float))
                    .unwrap();
                pool.push(value);
            }
            1 => {
                let source = pick(&mut rng, &pool);
                let value = b
                    .emit(site(pc), TraceOp::Negate { source }, Some(TraceType::Float))
                    .unwrap();
                pool.push(value);
            }
            2 | 3 if guards < 3 => {
                guards += 1;
                let comparison = [
                    Comparison::Equal,
                    Comparison::LessThan,
                    Comparison::LessEqual,
                ][rng.below(3)];
                let (left, right) = (pick(&mut rng, &pool), pick(&mut rng, &pool));
                let condition = b
                    .emit(
                        site(pc),
                        TraceOp::Compare {
                            comparison,
                            left,
                            right,
                        },
                        Some(TraceType::Boolean),
                    )
                    .unwrap();
                let (r0, r1) = (pick(&mut rng, &pool), pick(&mut rng, &pool));
                let fail = b.snapshot(snapshot(
                    SnapshotPhase::Before { pc },
                    pc - 1,
                    vec![
                        RegisterEntry::Value(r0),
                        RegisterEntry::Value(r1),
                        RegisterEntry::Unchanged,
                        RegisterEntry::Unchanged,
                    ],
                ));
                let guard = b.guard(site(pc), fail);
                b.emit(
                    site(pc),
                    TraceOp::GuardBranch {
                        condition,
                        expected: rng.below(4) != 0,
                        guard,
                    },
                    None,
                );
            }
            _ => {
                let op = [
                    ArithmeticOp::Add,
                    ArithmeticOp::Subtract,
                    ArithmeticOp::Multiply,
                    ArithmeticOp::Divide,
                    ArithmeticOp::Modulo,
                    ArithmeticOp::Power,
                ][rng.below(6)];
                let (left, right) = (pick(&mut rng, &pool), pick(&mut rng, &pool));
                let value = b
                    .emit(
                        site(pc),
                        TraceOp::Arithmetic { op, left, right },
                        Some(TraceType::Float),
                    )
                    .unwrap();
                pool.push(value);
            }
        }
    }
    let last = pool.len() - 1;
    let (next0, next1) = (pool[last], pool[last.saturating_sub(1 + rng.below(3))]);
    let poll_snapshot = b.snapshot(snapshot(
        SnapshotPhase::After { pc, successor: 1 },
        pc,
        vec![
            RegisterEntry::Value(next0),
            RegisterEntry::Value(next1),
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
        ],
    ));
    b.set_poll_snapshot(poll_snapshot);
    b.finish(TraceTerminator::LoopBack {
        args: vec![next0, next1],
    })
}

/// The native exits of seeded numeric loops at both tiers equal the reference
/// evaluation of the lowered trace. At the Optimized tier that trace has a
/// preheader, so a failing hoisted guard and snapshots that read hoisted
/// values are covered.
#[test]
fn seeded_numeric_loops_exit_like_the_reference_at_both_tiers() {
    let mut compared = 0_usize;
    let mut hoisted_exits = 0_usize;
    let mut preheaders = 0_usize;
    for seed in 0..400_u64 {
        let ir = generated_numeric_loop(seed);
        if ir.verify().is_err() {
            continue;
        }
        let mut rng = Rng(seed ^ 0x5555_aaaa | 1);
        for tier in TIERS {
            let trace = compile_trace(&ir, tier).unwrap();
            let lowered = trace.lowered_ir();
            let hoisted = lowered
                .preheader
                .iter()
                .filter(|instruction| instruction.op.guard().is_some())
                .count();
            preheaders += usize::from(!lowered.preheader.is_empty());
            for _ in 0..24 {
                let registers: Vec<TraceScalar> = (0..4)
                    .map(|_| f(CORNERS[rng.below(CORNERS.len())]))
                    .collect();
                let (outcome, native) = run_once(&trace, lowered, &registers);
                let evaluation = evaluate(lowered, &registers, &[], 64, |_, _| None);
                match evaluation.outcome {
                    Outcome::GuardFailed { guard, restored } => {
                        assert_eq!(
                            outcome,
                            TraceOutcome::Exit {
                                index: guard.index()
                            },
                            "seed {seed} {tier:?} {registers:?}"
                        );
                        assert_eq!(
                            native, restored.registers,
                            "seed {seed} {tier:?} guard {guard:?} from {registers:?}"
                        );
                        compared += 1;
                        if tier == OptimizationTier::Optimized {
                            hoisted_exits += usize::from(guard.index() < hoisted);
                        }
                    }
                    // The loop ran its iterations out. The poll exit has no
                    // single reference state, and the fuel is large.
                    Outcome::IterationLimit { .. } => {}
                    other => panic!("seed {seed}: unexpected {other:?}"),
                }
            }
        }
    }
    assert!(compared > 2000, "{compared} exits compared");
    assert!(
        preheaders > 100,
        "{preheaders} lowered traces had a preheader"
    );
    assert!(hoisted_exits > 50, "{hoisted_exits} hoisted guard exits");
}

/// Two floats are the same result when their bits agree or both are NaN: the
/// sign of a NaN depends on how the hardware or the compiler orders the
/// operands of an operation, which no Lua program can see.
fn same_floats(native: &[TraceScalar], expected: &[TraceScalar]) -> bool {
    native.len() == expected.len()
        && native.iter().zip(expected).all(|pair| match pair {
            (TraceScalar::Float(a), TraceScalar::Float(b)) => {
                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
            }
            (a, b) => a == b,
        })
}

/// Seeded loops that call native functions among their operations exit like
/// the reference at both tiers: the call's result, the guards behind it and
/// the values the snapshots restore.
#[test]
fn seeded_loops_with_native_calls_exit_like_the_reference_at_both_tiers() {
    let mut compared = 0_usize;
    let mut with_calls = 0_usize;
    for seed in 0..400_u64 {
        let ir = generated_loop(seed, true);
        if ir.verify().is_err() {
            continue;
        }
        let mut rng = Rng(seed ^ 0x5555_aaaa | 1);
        for tier in TIERS {
            let trace = compile_trace(&ir, tier).unwrap();
            let lowered = trace.lowered_ir();
            with_calls += usize::from(lowered.body.iter().chain(&lowered.preheader).any(|i| {
                matches!(
                    i.op,
                    TraceOp::NativeUnary { .. } | TraceOp::NativeBinary { .. }
                )
            }));
            for _ in 0..24 {
                let registers: Vec<TraceScalar> = (0..4)
                    .map(|_| f(CORNERS[rng.below(CORNERS.len())]))
                    .collect();
                let (outcome, native) = run_once(&trace, lowered, &registers);
                let evaluation = evaluate(lowered, &registers, &[], 64, |_, _| None);
                if let Outcome::GuardFailed { guard, restored } = evaluation.outcome {
                    assert_eq!(
                        outcome,
                        TraceOutcome::Exit {
                            index: guard.index()
                        },
                        "seed {seed} {tier:?} {registers:?}"
                    );
                    assert!(
                        same_floats(&native, &restored.registers),
                        "seed {seed} {tier:?} guard {guard:?} from {registers:?}: \
                         {native:?} against {:?}",
                        restored.registers
                    );
                    compared += 1;
                }
            }
        }
    }
    assert!(compared > 1500, "{compared} exits compared");
    assert!(with_calls > 200, "{with_calls} lowered traces kept a call");
}

// ------------------------------------------------------ integer semantics

const INTEGER_CORNERS: [i64; 21] = [
    0,
    1,
    -1,
    2,
    -2,
    7,
    -7,
    63,
    64,
    65,
    -63,
    -64,
    -65,
    i64::MAX,
    i64::MIN,
    i64::MAX - 1,
    i64::MIN + 1,
    9_007_199_254_740_992,
    9_007_199_254_740_993,
    -9_007_199_254_740_993,
    12_345_678_901,
];

/// Floats that sit on the edge of the integer range or of exact conversion.
const EDGE_FLOATS: [f64; 14] = [
    0.0,
    -0.0,
    0.5,
    -0.5,
    1.0,
    -1.0,
    9_007_199_254_740_992.0,
    9_007_199_254_740_994.0,
    9_223_372_036_854_775_808.0,
    -9_223_372_036_854_775_808.0,
    1e19,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
];

fn i(value: i64) -> TraceScalar {
    TraceScalar::Integer(value)
}

/// `r2 = op(r0, r1)` with typed inputs, once.
fn typed_binary_trace(
    op: impl FnOnce(ValueId, ValueId) -> TraceOp,
    left_ty: TraceType,
    right_ty: TraceType,
    result_ty: TraceType,
) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), root(3));
    b.mark_written(FrameId(0), 2);
    let left = b.input(0, left_ty);
    let right = b.input(1, right_ty);
    let result = b.emit(site(0), op(left, right), Some(result_ty)).unwrap();
    let exit = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 0,
            successor: 1,
        },
        0,
        vec![
            RegisterEntry::Value(left),
            RegisterEntry::Value(right),
            RegisterEntry::Value(result),
        ],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

fn assert_matches_reference(
    ir: &TraceIr,
    registers: &[TraceScalar],
    tiers: &[OptimizationTier],
    what: &str,
) {
    for &tier in tiers {
        let trace = compile_trace(ir, tier).unwrap();
        let (outcome, native) = run_once(&trace, ir, registers);
        assert!(matches!(outcome, TraceOutcome::Exit { .. }), "{what}");
        let Outcome::Exited(expected) = evaluate(ir, registers, &[], 1, |_, _| None).outcome else {
            panic!("the reference exits");
        };
        assert_eq!(
            native, expected.registers,
            "{what} at {tier:?} from {registers:?}"
        );
    }
}

#[test]
fn integer_division_modulo_and_shifts_match_the_reference_on_corner_values() {
    use ArithmeticOp::{BitAnd, BitOr, BitXor, FloorDivide, Modulo, ShiftLeft, ShiftRight};
    for op in [
        FloorDivide,
        Modulo,
        BitAnd,
        BitOr,
        BitXor,
        ShiftLeft,
        ShiftRight,
    ] {
        let ir = typed_binary_trace(
            |left, right| TraceOp::Arithmetic { op, left, right },
            TraceType::Integer,
            TraceType::Integer,
            TraceType::Integer,
        );
        for &a in &INTEGER_CORNERS {
            for &b in &INTEGER_CORNERS {
                assert_matches_reference(
                    &ir,
                    &[i(a), i(b), TraceScalar::Nil],
                    &TIERS,
                    &format!("{op:?}({a}, {b})"),
                );
            }
        }
    }
}

/// A constant positive divisor takes a mask, a shift or a multiplication
/// instead of the general sequence, and must give the same floor results,
/// negative dividends and the ends of the range included.
#[test]
fn integer_division_and_modulo_by_a_constant_match_the_reference() {
    let divisors = [
        1_i64,
        2,
        3,
        7,
        8,
        32,
        97,
        1000,
        1 << 31,
        4_294_967_296,
        1_000_000_007,
        i64::MAX,
        i64::MAX - 1,
        1 << 62,
        -1,
        -7,
        -32,
        i64::MIN,
        0,
    ];
    for op in [ArithmeticOp::Modulo, ArithmeticOp::FloorDivide] {
        for &divisor in &divisors {
            let mut b = TraceBuilder::new(TraceId(0), root(3));
            b.mark_written(FrameId(0), 2);
            let left = b.input(0, TraceType::Integer);
            let right = b
                .emit(
                    site(0),
                    TraceOp::Constant(TraceScalar::Integer(divisor)),
                    Some(TraceType::Integer),
                )
                .unwrap();
            let result = b
                .emit(
                    site(0),
                    TraceOp::Arithmetic { op, left, right },
                    Some(TraceType::Integer),
                )
                .unwrap();
            let exit = b.snapshot(snapshot(
                SnapshotPhase::After {
                    pc: 0,
                    successor: 1,
                },
                0,
                vec![
                    RegisterEntry::Value(left),
                    RegisterEntry::Const(TraceScalar::Nil),
                    RegisterEntry::Value(result),
                ],
            ));
            let ir = b.finish(TraceTerminator::Exit { snapshot: exit });
            for &a in INTEGER_CORNERS.iter().chain(&[
                divisor,
                divisor.wrapping_sub(1),
                divisor.wrapping_add(1),
                divisor.wrapping_neg(),
                divisor.wrapping_mul(3),
                divisor.wrapping_mul(-3).wrapping_add(5),
            ]) {
                assert_matches_reference(
                    &ir,
                    &[i(a), TraceScalar::Nil, TraceScalar::Nil],
                    &TIERS,
                    &format!("{op:?}({a}, const {divisor})"),
                );
            }
        }
    }
}

#[test]
fn the_shared_integer_semantics_are_lua_semantics() {
    use crate::jit_ir::trace::{integer_floor_divide, integer_floor_modulo, integer_shift};
    // Floor division and modulo against the definition: the quotient rounds
    // toward negative infinity and `a == (a // b) * b + a % b`.
    for &a in &INTEGER_CORNERS {
        for &b in &INTEGER_CORNERS {
            if b == 0 {
                assert_eq!(integer_floor_divide(a, b), 0);
                assert_eq!(integer_floor_modulo(a, b), 0);
                continue;
            }
            let quotient = integer_floor_divide(a, b);
            let remainder = integer_floor_modulo(a, b);
            assert_eq!(
                quotient.wrapping_mul(b).wrapping_add(remainder),
                a,
                "{a} // {b}"
            );
            if b != -1 {
                // Floor division is Euclidean division of the operands
                // made positive in the divisor.
                let (n, d) = (i128::from(a), i128::from(b));
                let exact = if d > 0 {
                    n.div_euclid(d)
                } else {
                    (-n).div_euclid(-d)
                };
                assert_eq!(i128::from(quotient), exact, "{a} // {b}");
                // The remainder takes the divisor's sign.
                assert!(
                    remainder == 0 || (remainder < 0) == (b < 0),
                    "{a} % {b} = {remainder}"
                );
            }
        }
    }
    assert_eq!(integer_floor_divide(i64::MIN, -1), i64::MIN);
    assert_eq!(integer_floor_modulo(i64::MIN, -1), 0);
    assert_eq!(integer_shift(1, 63), i64::MIN);
    assert_eq!(integer_shift(1, 64), 0);
    assert_eq!(integer_shift(-1, -1), i64::MAX);
    assert_eq!(integer_shift(-1, -64), 0);
    assert_eq!(integer_shift(5, -1), 2);
}

#[test]
fn float_floor_division_and_integer_conversion_match_the_reference() {
    for tier in TIERS {
        let ir = typed_binary_trace(
            |left, right| TraceOp::Arithmetic {
                op: ArithmeticOp::FloorDivide,
                left,
                right,
            },
            TraceType::Float,
            TraceType::Float,
            TraceType::Float,
        );
        for &a in CORNERS.iter().chain(&EDGE_FLOATS) {
            for &b in CORNERS.iter().chain(&EDGE_FLOATS) {
                assert_matches_reference(
                    &ir,
                    &[f(a), f(b), TraceScalar::Nil],
                    &[tier],
                    &format!("{a:?} // {b:?}"),
                );
            }
        }
    }
    let convert = typed_binary_trace(
        |left, _| TraceOp::IntToFloat { source: left },
        TraceType::Integer,
        TraceType::Integer,
        TraceType::Float,
    );
    for &a in &INTEGER_CORNERS {
        assert_matches_reference(
            &convert,
            &[i(a), i(0), TraceScalar::Nil],
            &TIERS,
            &format!("{a} as float"),
        );
    }
}

#[test]
fn numeric_operations_match_the_reference() {
    for op in [
        NumericOp::Floor,
        NumericOp::Ceil,
        NumericOp::Abs,
        NumericOp::FloatToInteger,
        NumericOp::TrickBits,
    ] {
        let result_ty = if matches!(op, NumericOp::FloatToInteger | NumericOp::TrickBits) {
            TraceType::Integer
        } else {
            TraceType::Float
        };
        let ir = typed_unary_trace(
            |source| TraceOp::Numeric { op, source },
            TraceType::Float,
            result_ty,
        );
        for &a in CORNERS.iter().chain(&EDGE_FLOATS) {
            assert_matches_reference(
                &ir,
                &[f(a), TraceScalar::Nil],
                &TIERS,
                &format!("{op:?} {a:?}"),
            );
        }
    }
    let abs = typed_unary_trace(
        |source| TraceOp::Numeric {
            op: NumericOp::Abs,
            source,
        },
        TraceType::Integer,
        TraceType::Integer,
    );
    for &a in &INTEGER_CORNERS {
        assert_matches_reference(&abs, &[i(a), TraceScalar::Nil], &TIERS, &format!("abs {a}"));
    }
}

#[test]
fn the_rounding_trick_keeps_the_low_word_of_a_rounded_float() {
    use crate::jit_ir::trace::trick_bits;
    for (value, word) in [
        (0.0, 0),
        (-0.0, 0),
        (0.5, 0),
        (1.5, 2),
        (2.5, 2),
        (-0.5, 0),
        (-1.5, 0xffff_fffe),
        (-2.5, 0xffff_fffe),
        (4_294_967_295.5, 0),
        (4_294_967_296.5, 0),
        (2_147_483_647.5, 0x8000_0000),
        (f64::NAN, 0),
        (f64::INFINITY, 0),
        (f64::NEG_INFINITY, 0),
        (3_002_399_751_580_330.0, 0x5555_5555),
    ] {
        assert_eq!(trick_bits(value), word, "{value:?}");
    }
    let ir = typed_unary_trace(
        |source| TraceOp::Numeric {
            op: NumericOp::TrickBits,
            source,
        },
        TraceType::Float,
        TraceType::Integer,
    );
    for value in [
        0.5,
        1.5,
        -1.5,
        2_147_483_647.5,
        4_294_967_296.5,
        f64::NAN,
        1e300,
    ] {
        assert_matches_reference(
            &ir,
            &[f(value), TraceScalar::Nil],
            &TIERS,
            &format!("trick {value:?}"),
        );
    }
}

#[test]
fn comparing_an_integer_with_a_float_is_exact() {
    use crate::jit_ir::trace::{compare, integer_float_equal};
    for comparison in [
        Comparison::Equal,
        Comparison::LessThan,
        Comparison::LessEqual,
    ] {
        for (left_ty, right_ty) in [
            (TraceType::Integer, TraceType::Float),
            (TraceType::Float, TraceType::Integer),
        ] {
            let ir = typed_binary_trace(
                |left, right| TraceOp::Compare {
                    comparison,
                    left,
                    right,
                },
                left_ty,
                right_ty,
                TraceType::Boolean,
            );
            for &integer in &INTEGER_CORNERS {
                for &float in CORNERS.iter().chain(&EDGE_FLOATS) {
                    let registers = if left_ty == TraceType::Integer {
                        [i(integer), f(float), TraceScalar::Nil]
                    } else {
                        [f(float), i(integer), TraceScalar::Nil]
                    };
                    assert_matches_reference(
                        &ir,
                        &registers,
                        &TIERS,
                        &format!("{comparison:?} {left_ty:?} {right_ty:?}"),
                    );
                }
            }
        }
    }
    // The semantics the backend and the folder share are exact, not a
    // conversion: 2^53 + 1 is above 2^53 as a float, every integer is below
    // 2^63 as a float, and an integer equals only the float of its value.
    let two_53 = 9_007_199_254_740_992_i64;
    assert_eq!(
        compare(Comparison::LessThan, f(two_53 as f64), i(two_53 + 1)),
        Some(true)
    );
    assert_eq!(
        compare(Comparison::Equal, f(two_53 as f64), i(two_53 + 1)),
        Some(false)
    );
    assert_eq!(
        compare(
            Comparison::LessThan,
            i(i64::MAX),
            f(9_223_372_036_854_775_808.0)
        ),
        Some(true)
    );
    assert_eq!(
        compare(
            Comparison::LessEqual,
            f(-9_223_372_036_854_775_808.0),
            i(i64::MIN)
        ),
        Some(true)
    );
    assert!(integer_float_equal(i64::MIN, -9_223_372_036_854_775_808.0));
    assert!(!integer_float_equal(i64::MAX, 9_223_372_036_854_775_808.0));
    assert!(integer_float_equal(0, -0.0));
    assert_eq!(
        compare(Comparison::LessThan, i(0), f(f64::NAN)),
        Some(false)
    );
    assert_eq!(
        compare(Comparison::LessEqual, f(f64::NAN), i(0)),
        Some(false)
    );
}

/// A float against an integer constant takes the folded float compare when
/// the constant is exactly representable, and the exact branch-free form
/// otherwise. Both agree with the reference on both sides.
#[test]
fn comparing_a_float_with_an_integer_constant_is_exact() {
    let constants = [
        0_i64,
        1,
        -1,
        32,
        (1 << 53) - 1,
        1 << 53,
        (1 << 53) + 1,
        -(1 << 53),
        -(1 << 53) - 1,
        i64::MAX,
        i64::MIN,
        i64::MIN + 1,
    ];
    for comparison in [
        Comparison::Equal,
        Comparison::LessThan,
        Comparison::LessEqual,
    ] {
        for constant_left in [false, true] {
            for &constant in &constants {
                let mut b = TraceBuilder::new(TraceId(0), root(3));
                b.mark_written(FrameId(0), 2);
                let float = b.input(0, TraceType::Float);
                let integer = b
                    .emit(
                        site(0),
                        TraceOp::Constant(TraceScalar::Integer(constant)),
                        Some(TraceType::Integer),
                    )
                    .unwrap();
                let (left, right) = if constant_left {
                    (integer, float)
                } else {
                    (float, integer)
                };
                let result = b
                    .emit(
                        site(0),
                        TraceOp::Compare {
                            comparison,
                            left,
                            right,
                        },
                        Some(TraceType::Boolean),
                    )
                    .unwrap();
                let exit = b.snapshot(snapshot(
                    SnapshotPhase::After {
                        pc: 0,
                        successor: 1,
                    },
                    0,
                    vec![
                        RegisterEntry::Value(float),
                        RegisterEntry::Const(TraceScalar::Nil),
                        RegisterEntry::Value(result),
                    ],
                ));
                let ir = b.finish(TraceTerminator::Exit { snapshot: exit });
                for &value in CORNERS.iter().chain(&EDGE_FLOATS).chain(&[
                    constant as f64,
                    (constant as f64).next_up(),
                    (constant as f64).next_down(),
                    constant as f64 + 0.5,
                ]) {
                    assert_matches_reference(
                        &ir,
                        &[f(value), TraceScalar::Nil, TraceScalar::Nil],
                        &TIERS,
                        &format!("{comparison:?} {value:e} const {constant} left {constant_left}"),
                    );
                }
            }
        }
    }
}

/// A loop over integer r0 and r1 (carried), integer r2 and float r3 (never
/// written). The body mixes every integer and float operation, with and
/// without the divisor guard the recorder always emits, so native code is
/// checked on its own for zero divisors as well.
fn generated_integer_loop(seed: u64) -> TraceIr {
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut b = TraceBuilder::new(TraceId(0), root(4));
    b.mark_written(FrameId(0), 0);
    b.mark_written(FrameId(0), 1);
    let in0 = b.input(0, TraceType::Integer);
    let in1 = b.input(1, TraceType::Integer);
    let in2 = b.input(2, TraceType::Integer);
    let in3 = b.input(3, TraceType::Float);
    let p0 = b.loop_param(TraceType::Integer, LoopParamInit::Input(in0));
    let p1 = b.loop_param(TraceType::Integer, LoopParamInit::Input(in1));
    let mut ints = vec![p0, p1, in2, in2];
    let mut floats = vec![in3];
    let steps = 5 + rng.below(8);
    let mut pc = 0;
    let mut guards = 0;
    for _ in 0..steps {
        pc += 1;
        let pick = |rng: &mut Rng, pool: &[ValueId]| pool[rng.below(pool.len())];
        match rng.below(11) {
            0 => {
                let value = b
                    .emit(
                        site(pc),
                        TraceOp::Constant(i(INTEGER_CORNERS[rng.below(INTEGER_CORNERS.len())])),
                        Some(TraceType::Integer),
                    )
                    .unwrap();
                ints.push(value);
            }
            1 => {
                let source = pick(&mut rng, &ints);
                let value = b
                    .emit(
                        site(pc),
                        TraceOp::IntToFloat { source },
                        Some(TraceType::Float),
                    )
                    .unwrap();
                floats.push(value);
            }
            2 => {
                let source = pick(&mut rng, &ints);
                let value = b
                    .emit(
                        site(pc),
                        TraceOp::Negate { source },
                        Some(TraceType::Integer),
                    )
                    .unwrap();
                ints.push(value);
            }
            3 | 4 if guards < 3 => {
                guards += 1;
                let comparison = [
                    Comparison::Equal,
                    Comparison::LessThan,
                    Comparison::LessEqual,
                ][rng.below(3)];
                let (left, right) = match rng.below(4) {
                    0 => (pick(&mut rng, &ints), pick(&mut rng, &ints)),
                    1 => (pick(&mut rng, &floats), pick(&mut rng, &floats)),
                    2 => (pick(&mut rng, &ints), pick(&mut rng, &floats)),
                    _ => (pick(&mut rng, &floats), pick(&mut rng, &ints)),
                };
                let condition = b
                    .emit(
                        site(pc),
                        TraceOp::Compare {
                            comparison,
                            left,
                            right,
                        },
                        Some(TraceType::Boolean),
                    )
                    .unwrap();
                let (r0, r1) = (pick(&mut rng, &ints), pick(&mut rng, &ints));
                let fail = b.snapshot(snapshot(
                    SnapshotPhase::Before { pc },
                    pc - 1,
                    vec![
                        RegisterEntry::Value(r0),
                        RegisterEntry::Value(r1),
                        RegisterEntry::Unchanged,
                        RegisterEntry::Unchanged,
                    ],
                ));
                let guard = b.guard(site(pc), fail);
                b.emit(
                    site(pc),
                    TraceOp::GuardBranch {
                        condition,
                        expected: rng.below(4) != 0,
                        guard,
                    },
                    None,
                );
            }
            5 | 6 => {
                let op = [
                    ArithmeticOp::Add,
                    ArithmeticOp::Subtract,
                    ArithmeticOp::Multiply,
                    ArithmeticOp::Divide,
                    ArithmeticOp::Modulo,
                    ArithmeticOp::Power,
                    ArithmeticOp::FloorDivide,
                ][rng.below(7)];
                let (left, right) = (pick(&mut rng, &floats), pick(&mut rng, &floats));
                let value = b
                    .emit(
                        site(pc),
                        TraceOp::Arithmetic { op, left, right },
                        Some(TraceType::Float),
                    )
                    .unwrap();
                floats.push(value);
            }
            _ => {
                let op = [
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
                ][rng.below(10)];
                let (left, right) = (pick(&mut rng, &ints), pick(&mut rng, &ints));
                let value = b
                    .emit(
                        site(pc),
                        TraceOp::Arithmetic { op, left, right },
                        Some(TraceType::Integer),
                    )
                    .unwrap();
                ints.push(value);
            }
        }
    }
    let last = ints.len() - 1;
    let (next0, next1) = (ints[last], ints[last.saturating_sub(1 + rng.below(3))]);
    let poll = b.snapshot(snapshot(
        SnapshotPhase::After { pc, successor: 1 },
        pc,
        vec![
            RegisterEntry::Value(next0),
            RegisterEntry::Value(next1),
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
        ],
    ));
    b.set_poll_snapshot(poll);
    b.finish(TraceTerminator::LoopBack {
        args: vec![next0, next1],
    })
}

/// Native exits of seeded integer loops equal the reference evaluation of the
/// lowered trace at both tiers, over the integer extremes and the floats at
/// the edge of the integer range. The loops use every integer operation, so
/// this also checks that no operand pair can trap native code.
#[test]
fn seeded_integer_loops_exit_like_the_reference_at_both_tiers() {
    let mut compared = 0_usize;
    for seed in 0..500_u64 {
        let ir = generated_integer_loop(seed);
        if ir.verify().is_err() {
            continue;
        }
        let mut rng = Rng(seed ^ 0x3333_cccc | 1);
        for tier in TIERS {
            let trace = compile_trace(&ir, tier).unwrap();
            let lowered = trace.lowered_ir();
            for _ in 0..24 {
                let registers = [
                    i(INTEGER_CORNERS[rng.below(INTEGER_CORNERS.len())]),
                    i(INTEGER_CORNERS[rng.below(INTEGER_CORNERS.len())]),
                    i(INTEGER_CORNERS[rng.below(INTEGER_CORNERS.len())]),
                    f(EDGE_FLOATS[rng.below(EDGE_FLOATS.len())]),
                ];
                let (outcome, native) = run_once(&trace, lowered, &registers);
                let evaluation = evaluate(lowered, &registers, &[], 64, |_, _| None);
                match evaluation.outcome {
                    Outcome::GuardFailed { guard, restored } => {
                        assert_eq!(
                            outcome,
                            TraceOutcome::Exit {
                                index: guard.index()
                            },
                            "seed {seed} {tier:?} {registers:?}"
                        );
                        assert_eq!(
                            native, restored.registers,
                            "seed {seed} {tier:?} guard {guard:?} from {registers:?}"
                        );
                        compared += 1;
                    }
                    Outcome::IterationLimit { .. } => {}
                    other => panic!("seed {seed}: unexpected {other:?}"),
                }
            }
        }
    }
    assert!(compared > 1500, "{compared} exits compared");
}

// ------------------------------------------------------- opaque lanes

fn table(lane: u64) -> TraceScalar {
    TraceScalar::Table(lane)
}

fn closure(lane: u64) -> TraceScalar {
    TraceScalar::Closure(lane)
}

/// r0 and r1 tables, r2 a closure: `r3 = r0 == r1`, `r4 = not r0`,
/// `r5 = r2 == r2`, `r6 = truthy r2`, and the lanes themselves come back.
fn opaque_trace() -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), root(7));
    for register in 3..7 {
        b.mark_written(FrameId(0), register);
    }
    let first = b.input(0, TraceType::Table);
    let second = b.input(1, TraceType::Table);
    let function = b.input(2, TraceType::Closure);
    let same = b
        .emit(
            site(0),
            TraceOp::Compare {
                comparison: Comparison::Equal,
                left: first,
                right: second,
            },
            Some(TraceType::Boolean),
        )
        .unwrap();
    let not = b
        .emit(
            site(0),
            TraceOp::Not { source: first },
            Some(TraceType::Boolean),
        )
        .unwrap();
    let itself = b
        .emit(
            site(0),
            TraceOp::Compare {
                comparison: Comparison::Equal,
                left: function,
                right: function,
            },
            Some(TraceType::Boolean),
        )
        .unwrap();
    let truthy = b
        .emit(
            site(0),
            TraceOp::Truthy { source: function },
            Some(TraceType::Boolean),
        )
        .unwrap();
    let moved = b
        .emit(
            site(0),
            TraceOp::Move { source: second },
            Some(TraceType::Table),
        )
        .unwrap();
    let exit = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 0,
            successor: 1,
        },
        0,
        vec![
            RegisterEntry::Value(first),
            RegisterEntry::Value(moved),
            RegisterEntry::Value(function),
            RegisterEntry::Value(same),
            RegisterEntry::Value(not),
            RegisterEntry::Value(itself),
            RegisterEntry::Value(truthy),
        ],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

#[test]
fn opaque_lanes_compare_by_identity_and_come_back_unchanged() {
    let ir = opaque_trace();
    ir.verify().unwrap();
    for tier in TIERS {
        let trace = compile_trace(&ir, tier).unwrap();
        for (a, b) in [
            (0, 0),
            (0, 1),
            (7, 7),
            (1_000_003, 5),
            (u64::MAX >> 1, u64::MAX >> 1),
        ] {
            let registers = [
                table(a),
                table(b),
                closure(3),
                TraceScalar::Nil,
                TraceScalar::Nil,
                TraceScalar::Nil,
                TraceScalar::Nil,
            ];
            let (outcome, native) = run_once(&trace, &ir, &registers);
            assert!(matches!(outcome, TraceOutcome::Exit { .. }));
            let reference = evaluate(&ir, &registers, &[], 1, |_, _| None);
            let Outcome::Exited(expected) = reference.outcome else {
                panic!("the reference exits");
            };
            assert_eq!(native, expected.registers, "({a}, {b}) at {tier:?}");
            assert_eq!(native[0], table(a));
            assert_eq!(native[1], table(b));
            assert_eq!(native[2], closure(3));
            assert_eq!(native[3], TraceScalar::Boolean(a == b));
            assert_eq!(native[4], TraceScalar::Boolean(false), "a table is truthy");
            assert_eq!(native[5], TraceScalar::Boolean(true));
            assert_eq!(native[6], TraceScalar::Boolean(true));
        }
    }
}

#[test]
fn a_table_does_not_enter_where_a_closure_is_required_and_the_reverse() {
    let ir = opaque_trace();
    for tier in TIERS {
        let trace = compile_trace(&ir, tier).unwrap();
        for (first, second, function) in [
            (closure(1), table(1), closure(1)),
            (table(1), closure(1), closure(1)),
            (table(1), table(1), table(1)),
            (TraceScalar::Nil, table(1), closure(1)),
            (f(1.0), table(1), closure(1)),
        ] {
            let registers = [
                first,
                second,
                function,
                TraceScalar::Nil,
                TraceScalar::Nil,
                TraceScalar::Nil,
                TraceScalar::Nil,
            ];
            let (outcome, _) = run_once(&trace, &ir, &registers);
            assert!(
                matches!(outcome, TraceOutcome::EntryMismatch { .. }),
                "{registers:?} at {tier:?}"
            );
        }
    }
}

#[test]
fn a_loop_carried_swap_of_opaque_values_is_a_parallel_assignment() {
    for ty in [TraceType::Table, TraceType::Closure] {
        let ir = swap_loop_of(ty);
        ir.verify().unwrap();
        let lane = |value: u64| {
            if ty == TraceType::Table {
                table(value)
            } else {
                closure(value)
            }
        };
        for tier in TIERS {
            let trace = compile_trace(&ir, tier).unwrap();
            for count in [1.0, 2.0, 3.0, 8.0, 101.0] {
                let registers = [lane(4), lane(9), f(count)];
                let (outcome, native) = run_once(&trace, &ir, &registers);
                assert_eq!(outcome, TraceOutcome::Exit { index: 0 });
                let Outcome::GuardFailed {
                    restored: expected, ..
                } = evaluate(&ir, &registers, &[], 10_000, |_, _| None).outcome
                else {
                    panic!("the reference leaves through the guard");
                };
                assert_eq!(native, expected.registers, "{ty:?} {count} iterations");
                let pair = if (count as u64) % 2 == 1 {
                    [lane(9), lane(4)]
                } else {
                    [lane(4), lane(9)]
                };
                assert_eq!(&native[..2], &pair, "{ty:?} {count}");
            }
        }
    }
}

// ------------------------------------------------------------ inlined frames

/// A call of the pinned closure 0 from root pc 2, inlined. The callee, frame
/// 1, doubles its argument (`r0 * 2` at its pc 0) and guards that the product
/// is positive (its pc 1). The root keeps the result in `r3`.
///
/// Guards: 0 compares the callee with the pinned lane (`Before` the call),
/// 1 is inside the callee (`After`, two frames).
fn inlined_call_trace() -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), root(5));
    b.set_pinned_closures(vec![0xfeed]);
    let callee = b.add_frame(TraceFrameInfo {
        register_count: 3,
        code_len: 16,
        inlined: Some(InlinedFrame {
            closure: 0,
            parent: FrameId(0),
            call_pc: 2,
            result_base: 3,
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
    b.mark_written(FrameId(0), 3);
    let function = b.input(0, TraceType::Closure);
    let x = b.input(1, TraceType::Float);
    // The trace writes r3 at the end, so before that r3 cannot stay in the
    // interpreter: it restores to nil, which the test registers' r3 never is.
    let unchanged = || {
        let mut registers = vec![RegisterEntry::Unchanged; 5];
        registers[3] = RegisterEntry::Const(TraceScalar::Nil);
        registers
    };
    let pinned = b
        .emit(
            site(2),
            TraceOp::Constant(TraceScalar::Closure(0)),
            Some(TraceType::Closure),
        )
        .unwrap();
    let same = b
        .emit(
            site(2),
            TraceOp::Compare {
                comparison: Comparison::Equal,
                left: function,
                right: pinned,
            },
            Some(TraceType::Boolean),
        )
        .unwrap();
    let before_call = b.snapshot(snapshot(SnapshotPhase::Before { pc: 2 }, 1, unchanged()));
    let g0 = b.guard(site(2), before_call);
    b.emit(
        site(2),
        TraceOp::GuardBranch {
            condition: same,
            expected: true,
            guard: g0,
        },
        None,
    );
    let in_callee = |pc| TraceSite { frame: callee, pc };
    let argument = b
        .emit(site(2), TraceOp::Move { source: x }, Some(TraceType::Float))
        .unwrap();
    let two = b
        .emit(
            in_callee(0),
            TraceOp::Constant(f(2.0)),
            Some(TraceType::Float),
        )
        .unwrap();
    let product = b
        .emit(
            in_callee(0),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Multiply,
                left: argument,
                right: two,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let zero = b
        .emit(
            in_callee(1),
            TraceOp::Constant(f(0.0)),
            Some(TraceType::Float),
        )
        .unwrap();
    let positive = b
        .emit(
            in_callee(1),
            TraceOp::Compare {
                comparison: Comparison::LessThan,
                left: zero,
                right: product,
            },
            Some(TraceType::Boolean),
        )
        .unwrap();
    let inside = b.snapshot(Snapshot::new(
        SnapshotPhase::After {
            pc: 1,
            successor: 2,
        },
        0,
        vec![
            SnapshotFrame {
                frame: FrameId(0),
                last_interpreted_pc: Some(2),
                registers: unchanged().into(),
            },
            SnapshotFrame {
                frame: callee,
                last_interpreted_pc: Some(1),
                registers: vec![
                    RegisterEntry::Value(argument),
                    RegisterEntry::Value(product),
                    RegisterEntry::Const(TraceScalar::Nil),
                ]
                .into(),
            },
        ],
    ));
    let g1 = b.guard(in_callee(1), inside);
    b.emit(
        in_callee(1),
        TraceOp::GuardBranch {
            condition: positive,
            expected: true,
            guard: g1,
        },
        None,
    );
    let result = b
        .emit(
            site(3),
            TraceOp::Move { source: product },
            Some(TraceType::Float),
        )
        .unwrap();
    let mut done = unchanged();
    done[3] = RegisterEntry::Value(result);
    let exit = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 3,
            successor: 4,
        },
        3,
        done,
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

/// Every frame an exit restores, outermost first.
fn restored_frames(
    trace: &CompiledTrace,
    index: usize,
    area: &[i64],
    registers: &[TraceScalar],
) -> Vec<Vec<TraceScalar>> {
    let descriptor = &trace.exits()[index];
    let snapshot = &trace.lowered_ir().snapshots[descriptor.snapshot.index()];
    snapshot
        .frames
        .iter()
        .map(|frame| {
            frame
                .registers
                .iter()
                .enumerate()
                .map(|(register, entry)| match entry {
                    RegisterEntry::Value(value) => descriptor
                        .scalar(area, *value)
                        .expect("the exit spilled every snapshot value"),
                    RegisterEntry::Const(scalar) => *scalar,
                    RegisterEntry::Unchanged => registers[register],
                    RegisterEntry::Root(_) => unreachable!("no roots in this trace"),
                    RegisterEntry::Virtual(_) | RegisterEntry::Slice(_) => {
                        unreachable!("no virtual tables here")
                    }
                })
                .collect()
        })
        .collect()
}

#[test]
fn an_inlined_call_exits_with_every_frame_restored_like_the_reference() {
    let ir = inlined_call_trace();
    ir.verify().unwrap();
    for tier in TIERS {
        let trace = compile_trace(&ir, tier).unwrap();
        for (function, x) in [
            (closure(0), 3.0),
            (closure(0), -1.5),
            (closure(0), 0.0),
            (closure(0), f64::NAN),
            // Another closure is refused before the call.
            (closure(1), 3.0),
            (closure(9), -4.0),
        ] {
            let registers = [function, f(x), f(7.0), f(8.0), f(9.0)];
            let inputs: Vec<TraceInputSlot> = trace
                .entry_contract()
                .iter()
                .map(|(register, _)| slot(registers[usize::from(*register)]))
                .collect();
            let mut area = vec![0_i64; trace.exit_area_len()];
            let mut control = TraceControl::new(100);
            let outcome = trace.run(&inputs, &mut control, &mut area);
            let TraceOutcome::Exit { index } = outcome else {
                panic!("entry mismatch for {function:?} {x}");
            };
            let native = restored_frames(&trace, index, &area, &registers);
            let reference = evaluate(&ir, &registers, &[], 1, |_, _| None);
            let (expected_outer, expected_inner, pc, last) = match reference.outcome {
                Outcome::GuardFailed { restored, .. } | Outcome::Exited(restored) => (
                    restored.outer,
                    restored.registers,
                    restored.pc,
                    restored.last_interpreted_pc,
                ),
                other => panic!("unexpected outcome {other:?}"),
            };
            assert_eq!(
                native.last().unwrap(),
                &expected_inner,
                "{function:?} {x} {tier:?}"
            );
            assert_eq!(
                native.len(),
                expected_outer.len() + 1,
                "{function:?} {x} {tier:?}"
            );
            for (frame, outer) in native.iter().zip(&expected_outer) {
                assert_eq!(frame, &outer.registers, "{function:?} {x} {tier:?}");
            }
            // Which exit it was.
            let descriptor = &trace.exits()[index];
            let snapshot = &trace.lowered_ir().snapshots[descriptor.snapshot.index()];
            assert_eq!(snapshot.resume_point().pc, pc);
            assert_eq!(snapshot.resume_point().last_interpreted_pc, last);
            let callee_is_pinned = function == closure(0);
            if !callee_is_pinned {
                assert_eq!(native.len(), 1, "refused before the call");
                assert_eq!(descriptor.guard, Some(TraceGuardClass::Callee));
            } else if x > 0.0 {
                assert_eq!(native.len(), 1, "ran to the end");
                assert_eq!(native[0][3], f(x * 2.0));
            } else {
                assert_eq!(native.len(), 2, "failed inside the callee");
                assert_eq!(native[1][1], f(x * 2.0), "{x}");
                assert_eq!(descriptor.guard, Some(TraceGuardClass::Branch));
            }
        }
    }
}

const WIDE_CORNERS: [u64; 14] = [
    0,
    1,
    2,
    3,
    63,
    64,
    65,
    0xffff_ffff,
    0x1_0000_0000,
    0x7fff_ffff_ffff_ffff,
    0x8000_0000_0000_0000,
    0x8000_0000_0000_0001,
    0xffff_ffff_ffff_fffe,
    0xffff_ffff_ffff_ffff,
];

fn wide(ty: TraceType, bits: u64) -> TraceScalar {
    if ty == TraceType::Int64 {
        TraceScalar::Int64(bits.cast_signed())
    } else {
        TraceScalar::UInt64(bits)
    }
}

const WIDE_TYPES: [TraceType; 2] = [TraceType::Int64, TraceType::UInt64];

#[test]
fn wide_arithmetic_matches_the_reference_and_the_runtime_on_corner_values() {
    use crate::runtime::Value;
    use crate::runtime::cint64::{CIntArith, arith};
    use ArithmeticOp::{
        Add, ArithmeticShiftRight, BitAnd, BitOr, BitXor, Divide, Modulo, Multiply, RotateLeft,
        RotateRight, ShiftLeft, ShiftRight, Subtract,
    };
    for ty in WIDE_TYPES {
        for op in [
            Add,
            Subtract,
            Multiply,
            Divide,
            Modulo,
            BitAnd,
            BitOr,
            BitXor,
            ShiftLeft,
            ShiftRight,
            ArithmeticShiftRight,
            RotateLeft,
            RotateRight,
        ] {
            let ir = typed_binary_trace(
                |left, right| TraceOp::Arithmetic { op, left, right },
                ty,
                ty,
                ty,
            );
            for &a in &WIDE_CORNERS {
                for &b in &WIDE_CORNERS {
                    let registers = [wide(ty, a), wide(ty, b), TraceScalar::Nil];
                    assert_matches_reference(&ir, &registers, &TIERS, &format!("{op:?} {ty:?}"));
                    let runtime = match op {
                        Add => Some(CIntArith::Add),
                        Subtract => Some(CIntArith::Subtract),
                        Multiply => Some(CIntArith::Multiply),
                        Divide => Some(CIntArith::Divide),
                        Modulo => Some(CIntArith::Modulo),
                        _ => None,
                    };
                    if let Some(runtime) = runtime {
                        let to_value = |scalar: TraceScalar| match scalar {
                            TraceScalar::Int64(value) => Value::Int64(value),
                            TraceScalar::UInt64(value) => Value::UInt64(value),
                            _ => unreachable!(),
                        };
                        let expected =
                            arith(runtime, &to_value(registers[0]), &to_value(registers[1]))
                                .unwrap();
                        let produced =
                            crate::jit_ir::trace::arithmetic(op, registers[0], registers[1])
                                .unwrap();
                        assert_eq!(to_value(produced), expected, "{op:?} {ty:?} {a:#x} {b:#x}");
                    }
                }
            }
        }
    }
}

#[test]
fn wide_comparisons_order_signed_and_unsigned_bits_differently() {
    for ty in WIDE_TYPES {
        for comparison in [
            Comparison::Equal,
            Comparison::LessThan,
            Comparison::LessEqual,
        ] {
            let ir = typed_binary_trace(
                |left, right| TraceOp::Compare {
                    comparison,
                    left,
                    right,
                },
                ty,
                ty,
                TraceType::Boolean,
            );
            for &a in &WIDE_CORNERS {
                for &b in &WIDE_CORNERS {
                    let registers = [wide(ty, a), wide(ty, b), TraceScalar::Nil];
                    assert_matches_reference(
                        &ir,
                        &registers,
                        &TIERS,
                        &format!("{comparison:?} {ty:?} {a:#x} {b:#x}"),
                    );
                    let expected = if ty == TraceType::Int64 {
                        let (a, b) = (a.cast_signed(), b.cast_signed());
                        match comparison {
                            Comparison::Equal => a == b,
                            Comparison::LessThan => a < b,
                            Comparison::LessEqual => a <= b,
                        }
                    } else {
                        match comparison {
                            Comparison::Equal => a == b,
                            Comparison::LessThan => a < b,
                            Comparison::LessEqual => a <= b,
                        }
                    };
                    assert_eq!(
                        crate::jit_ir::trace::compare(comparison, registers[0], registers[1]),
                        Some(expected)
                    );
                }
            }
        }
    }
}

const WIDE_FLOAT_CORNERS: [f64; 24] = [
    0.0,
    -0.0,
    0.5,
    -0.5,
    1.5,
    2.5,
    3.5,
    -1.5,
    -2.7,
    4_294_967_295.5,
    4_294_967_296.5,
    2_251_799_813_685_248.0,
    9_007_199_254_740_993.0,
    9_223_372_036_854_774_784.0,
    9_223_372_036_854_775_808.0,
    -9_223_372_036_854_775_808.0,
    -9_223_372_036_854_777_856.0,
    1.8e19,
    18_446_744_073_709_551_616.0,
    1e19,
    -1e19,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
];

#[test]
fn float_to_wide_conversions_match_the_reference_and_the_runtime() {
    use crate::jit_ir::trace::WideConversion;
    use crate::runtime::cint64::{number_to_i64, number_to_u64};
    for target in WIDE_TYPES {
        for conversion in [WideConversion::Truncate, WideConversion::Bits32] {
            if conversion == WideConversion::Bits32 && target != TraceType::Int64 {
                continue;
            }
            let ir = typed_unary_trace(
                |source| TraceOp::ToWide {
                    source,
                    target,
                    conversion,
                },
                TraceType::Float,
                target,
            );
            for &number in &WIDE_FLOAT_CORNERS {
                assert_matches_reference(
                    &ir,
                    &[f(number), TraceScalar::Nil],
                    &TIERS,
                    &format!("{conversion:?} {target:?} {number:e}"),
                );
                if conversion == WideConversion::Truncate {
                    let expected = if target == TraceType::Int64 {
                        TraceScalar::Int64(number_to_i64(number))
                    } else {
                        TraceScalar::UInt64(number_to_u64(number))
                    };
                    assert_eq!(
                        crate::jit_ir::trace::to_wide(f(number), target, conversion),
                        Some(expected),
                        "{target:?} {number:e}"
                    );
                }
            }
        }
    }
    // The other cdata keeps its bits.
    for (from, to) in [
        (TraceType::Int64, TraceType::UInt64),
        (TraceType::UInt64, TraceType::Int64),
    ] {
        let ir = typed_unary_trace(
            |source| TraceOp::ToWide {
                source,
                target: to,
                conversion: WideConversion::Truncate,
            },
            from,
            to,
        );
        for &bits in &WIDE_CORNERS {
            assert_matches_reference(
                &ir,
                &[wide(from, bits), TraceScalar::Nil],
                &TIERS,
                &format!("{from:?} to {to:?}"),
            );
        }
    }
}

#[test]
fn wide_negation_and_conversion_to_float_match_the_reference() {
    for ty in WIDE_TYPES {
        let negate = typed_unary_trace(|source| TraceOp::Negate { source }, ty, ty);
        let to_float = typed_unary_trace(
            |source| TraceOp::IntToFloat { source },
            ty,
            TraceType::Float,
        );
        for &bits in &WIDE_CORNERS {
            let registers = [wide(ty, bits), TraceScalar::Nil];
            assert_matches_reference(&negate, &registers, &TIERS, &format!("negate {ty:?}"));
            assert_matches_reference(&to_float, &registers, &TIERS, &format!("float {ty:?}"));
            let TraceScalar::Float(converted) =
                crate::jit_ir::trace::to_float(registers[0]).unwrap()
            else {
                panic!("a float");
            };
            let expected = if ty == TraceType::Int64 {
                bits.cast_signed() as f64
            } else {
                bits as f64
            };
            assert_eq!(converted.to_bits(), expected.to_bits(), "{ty:?} {bits:#x}");
        }
    }
}

/// `r1 = op(r0)` with a typed input, once.
fn typed_unary_trace(
    op: impl FnOnce(ValueId) -> TraceOp,
    input_ty: TraceType,
    result_ty: TraceType,
) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), root(2));
    b.mark_written(FrameId(0), 1);
    let source = b.input(0, input_ty);
    let result = b.emit(site(0), op(source), Some(result_ty)).unwrap();
    let exit = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 0,
            successor: 1,
        },
        0,
        vec![RegisterEntry::Value(source), RegisterEntry::Value(result)],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

/// The optimizer folds an operation on constants with the same semantics the
/// native code has: the folded trace and the reference agree.
#[test]
fn wide_operations_on_constants_fold_like_they_run() {
    use ArithmeticOp::{Add, ArithmeticShiftRight, Divide, Modulo, Multiply, RotateRight};
    for ty in WIDE_TYPES {
        for op in [
            Add,
            Multiply,
            Divide,
            Modulo,
            ArithmeticShiftRight,
            RotateRight,
        ] {
            for &a in &WIDE_CORNERS {
                for &b in &[0_u64, 1, 7, 0xffff_ffff_ffff_ffff] {
                    let mut builder = TraceBuilder::new(TraceId(0), root(3));
                    builder.mark_written(FrameId(0), 2);
                    let left = builder
                        .emit(site(0), TraceOp::Constant(wide(ty, a)), Some(ty))
                        .unwrap();
                    let right = builder
                        .emit(site(0), TraceOp::Constant(wide(ty, b)), Some(ty))
                        .unwrap();
                    let result = builder
                        .emit(site(0), TraceOp::Arithmetic { op, left, right }, Some(ty))
                        .unwrap();
                    let exit = builder.snapshot(snapshot(
                        SnapshotPhase::After {
                            pc: 0,
                            successor: 1,
                        },
                        0,
                        vec![
                            RegisterEntry::Value(left),
                            RegisterEntry::Value(right),
                            RegisterEntry::Value(result),
                        ],
                    ));
                    let ir = builder.finish(TraceTerminator::Exit { snapshot: exit });
                    assert_matches_reference(
                        &ir,
                        &[TraceScalar::Nil, TraceScalar::Nil, TraceScalar::Nil],
                        &TIERS,
                        &format!("{op:?} {ty:?} {a:#x} {b:#x}"),
                    );
                }
            }
        }
    }
}

/// A phantom table op counts one table and its footprint's bytes in the
/// control block, once per execution, and nothing else.
#[test]
fn phantom_table_ops_count_tables_and_bytes_in_the_control_block() {
    for tier in TIERS {
        for loops in [1_i64, 5] {
            // A loop that runs two constructors per iteration.
            let mut b = TraceBuilder::new(TraceId(0), root(2));
            b.mark_written(FrameId(0), 1);
            let x = b.input(0, TraceType::Float);
            let sum = b.loop_param(TraceType::Float, LoopParamInit::Input(x));
            b.emit(site(0), TraceOp::PhantomTable { entries: 3 }, None);
            let one = b
                .emit(site(1), TraceOp::Constant(f(1.0)), Some(TraceType::Float))
                .unwrap();
            b.emit(site(2), TraceOp::PhantomTable { entries: 8 }, None);
            let next = b
                .emit(
                    site(3),
                    TraceOp::Arithmetic {
                        op: ArithmeticOp::Add,
                        left: sum,
                        right: one,
                    },
                    Some(TraceType::Float),
                )
                .unwrap();
            let poll = b.snapshot(snapshot(
                SnapshotPhase::After {
                    pc: 3,
                    successor: 0,
                },
                3,
                vec![RegisterEntry::Value(next), RegisterEntry::Value(next)],
            ));
            b.set_poll_snapshot(poll);
            let ir = b.finish(TraceTerminator::LoopBack { args: vec![next] });
            ir.verify().unwrap();
            let trace = compile_trace(&ir, tier).unwrap();
            let inputs = [slot(f(0.0))];
            let mut area = vec![0; trace.exit_area_len()];
            let mut control = TraceControl::new(loops);
            let outcome = trace.run(&inputs, &mut control, &mut area);
            assert!(matches!(outcome, TraceOutcome::Exit { .. }), "{tier:?}");
            // The poll leaves when the fuel is spent: `loops` iterations ran.
            let ran = loops;
            assert_eq!(control.elided_tables, 2 * ran, "{tier:?} {loops}");
            let per_iteration = crate::runtime::Runtime::elided_footprint(3)
                + crate::runtime::Runtime::elided_footprint(8);
            assert_eq!(
                control.elided_bytes,
                i64::try_from(per_iteration).unwrap() * ran,
                "{tier:?} {loops}"
            );
        }
    }
}

/// A phantom charge is a count the run owes, so it never sits in the
/// preheader, which runs once.
#[test]
fn a_phantom_table_in_the_preheader_is_refused() {
    let mut b = TraceBuilder::new(TraceId(0), root(2));
    b.mark_written(FrameId(0), 1);
    let x = b.input(0, TraceType::Float);
    let sum = b.loop_param(TraceType::Float, LoopParamInit::Input(x));
    let mut ir = {
        let next = b
            .emit(
                site(0),
                TraceOp::Arithmetic {
                    op: ArithmeticOp::Add,
                    left: sum,
                    right: sum,
                },
                Some(TraceType::Float),
            )
            .unwrap();
        let poll = b.snapshot(snapshot(
            SnapshotPhase::After {
                pc: 0,
                successor: 0,
            },
            0,
            vec![RegisterEntry::Value(next), RegisterEntry::Value(next)],
        ));
        b.set_poll_snapshot(poll);
        b.finish(TraceTerminator::LoopBack { args: vec![next] })
    };
    ir.verify().unwrap();
    ir.preheader.push(crate::jit_ir::trace::TraceInstruction {
        site: site(0),
        result: None,
        op: TraceOp::PhantomTable { entries: 1 },
    });
    assert!(ir.verify().is_err());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn the_forwarding_stubs_are_compiled_once_per_process() {
    // Every trace imports the same copy of a stub that only forwards to a
    // helper, so two compiles resolve to one address, also across threads.
    use super::emit::shared_stubs;
    let stubs = shared_stubs();
    let admit = stubs.admit_table().unwrap();
    let set = stubs.table_set().unwrap();
    let reserve = stubs.reserve_array().unwrap();
    let get = stubs.hash_get().unwrap();
    let helper = stubs.helper(5).unwrap();
    let mut all = [admit, set, reserve, get, helper];
    all.sort_unstable();
    assert!(all.windows(2).all(|pair| pair[0] != pair[1]), "{all:?}");
    assert!(all.iter().all(|address| *address != 0));
    let again = std::thread::spawn(|| {
        let stubs = shared_stubs();
        [
            stubs.admit_table().unwrap(),
            stubs.table_set().unwrap(),
            stubs.reserve_array().unwrap(),
            stubs.hash_get().unwrap(),
            stubs.helper(5).unwrap(),
        ]
    })
    .join()
    .unwrap();
    assert_eq!(again, [admit, set, reserve, get, helper]);
    assert_ne!(stubs.helper(6).unwrap(), helper);
    assert_eq!(stubs.helper(6).unwrap(), stubs.helper(6).unwrap());
}

/// A trace that computes `r3 = r0 and r1 or r2` as a value select of a
/// boolean `r0` (or, with `compared`, of `r1 < r2`) and exits.
fn value_select_trace(arm_ty: TraceType, compared: bool) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), root(4));
    b.mark_written(FrameId(0), 3);
    let flag = b.input(0, TraceType::Boolean);
    let if_true = b.input(1, arm_ty);
    let if_false = b.input(2, arm_ty);
    let condition = if compared {
        b.emit(
            site(0),
            TraceOp::Compare {
                comparison: Comparison::LessThan,
                left: if_true,
                right: if_false,
            },
            Some(TraceType::Boolean),
        )
        .unwrap()
    } else {
        flag
    };
    let result = b
        .emit(
            site(1),
            TraceOp::Select {
                condition,
                if_true,
                if_false,
            },
            Some(arm_ty),
        )
        .unwrap();
    let exit = b.snapshot(snapshot(
        SnapshotPhase::Before { pc: 2 },
        1,
        vec![
            RegisterEntry::Value(flag),
            RegisterEntry::Value(if_true),
            RegisterEntry::Value(if_false),
            RegisterEntry::Value(result),
        ],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

#[test]
fn value_selects_pick_like_the_reference_at_both_tiers() {
    let floats: Vec<TraceScalar> = CORNERS.iter().copied().map(f).collect();
    let integers: Vec<TraceScalar> = [0, 1, -1, 7, i64::MAX, i64::MIN]
        .into_iter()
        .map(TraceScalar::Integer)
        .collect();
    let booleans = vec![TraceScalar::Boolean(false), TraceScalar::Boolean(true)];
    let cases = [
        (TraceType::Float, false, &floats),
        (TraceType::Float, true, &floats),
        (TraceType::Integer, false, &integers),
        (TraceType::Integer, true, &integers),
        (TraceType::Boolean, false, &booleans),
    ];
    for tier in TIERS {
        for (arm_ty, compared, values) in cases {
            let ir = value_select_trace(arm_ty, compared);
            ir.verify().unwrap();
            let trace = compile_trace(&ir, tier).unwrap();
            for flag in [false, true] {
                for &a in values {
                    for &b in values {
                        let registers = [TraceScalar::Boolean(flag), a, b, TraceScalar::Nil];
                        let (outcome, native) = run_once(&trace, &ir, &registers);
                        assert!(matches!(outcome, TraceOutcome::Exit { .. }));
                        let reference = evaluate(&ir, &registers, &[], 1, |_, _| None);
                        let Outcome::Exited(expected) = reference.outcome else {
                            panic!("the reference exits");
                        };
                        // Bit for bit: a NaN arm keeps its payload.
                        assert_eq!(
                            format!("{native:?}"),
                            format!("{:?}", expected.registers),
                            "select({flag}, {a:?}, {b:?}) {arm_ty:?} compared {compared} at {tier:?}"
                        );
                    }
                }
            }
        }
    }
}
