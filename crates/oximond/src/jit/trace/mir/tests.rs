//! The MIR backend against Cranelift, tree for tree.
//!
//! Each case compiles one hand-built trace with each backend (the backend is
//! picked per thread, see `debug_env::FORCE_BACKEND`), checks that the MIR
//! backend really compiled it, runs both from many states and compares
//! everything a run can observe: the outcome, the whole control block (fuel
//! among it) and every word of the exit area.

use super::super::compile::{
    CompiledTrace, TraceCompileOptions, compile_trace, compile_trace_with,
};
use super::super::exit::{TraceControl, TraceInputSlot, TraceOutcome};
use super::support::with_target;
use crate::compat::FloatModulo;
use crate::jit::OptimizationTier;
use crate::jit::debug_env::FORCE_BACKEND;
use crate::jit_ir::Comparison;
use crate::jit_ir::trace::{
    ArithmeticOp, FrameId, LoopParamInit, NumericOp, RegisterEntry, Snapshot, SnapshotFrame,
    SnapshotPhase, TraceBuilder, TraceFrameInfo, TraceId, TraceIr, TraceOp, TraceScalar, TraceSide,
    TraceSite, TraceTerminator, TraceType, ValueId,
};
use oximond_mir::x64::X64;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

// ---- allocation counting ------------------------------------------------------

struct Counting;

thread_local! {
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
    /// What the last MIR compile on this thread allocated in the backend.
    pub(super) static LAST_COMPILE_ALLOCATIONS: Cell<u64> = const { Cell::new(u64::MAX) };
    /// What the allocator reported for the last MIR compile on this thread.
    pub(super) static LAST_STATS: Cell<Option<oximond_mir::ra::Stats>> = const { Cell::new(None) };
    /// Why the last tree this thread handed to Cranelift was refused. The
    /// global counters are shared by every test thread, so cases check this.
    pub(super) static LAST_FALLBACK: Cell<Option<&'static str>> = const { Cell::new(None) };
    /// Trees with a call of a host function (`HelperId::vm_call`) the MIR
    /// backend compiled on this thread.
    pub(super) static VM_CALL_TREES: Cell<u64> = const { Cell::new(0) };
}

pub(super) fn thread_allocations() -> u64 {
    ALLOCATIONS.with(Cell::get)
}

// SAFETY: forwards every call to `System` unchanged; it only counts.
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|count| count.set(count.get() + 1));
        // SAFETY: the caller's contract is `System`'s.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: as above.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.with(|count| count.set(count.get() + 1));
        // SAFETY: as above.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

// ---- builders -----------------------------------------------------------------

pub(super) const OPTIMIZED: OptimizationTier = OptimizationTier::Optimized;
pub(super) const TIERS: [OptimizationTier; 2] =
    [OptimizationTier::Baseline, OptimizationTier::Optimized];

fn site(pc: usize) -> TraceSite {
    TraceSite {
        frame: FrameId(0),
        pc,
    }
}

fn frame(registers: u16) -> TraceFrameInfo {
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

fn after(pc: usize) -> SnapshotPhase {
    SnapshotPhase::After {
        pc,
        successor: pc + 1,
    }
}

pub(super) fn float(value: f64) -> TraceScalar {
    TraceScalar::Float(value)
}

/// A trace that applies `op` to inputs of the given types and exits, with the
/// inputs and the result in registers 0.. and the result's type `result`.
fn op_trace(
    inputs: &[TraceType],
    result: TraceType,
    make: impl FnOnce(&[ValueId]) -> TraceOp,
) -> TraceIr {
    let count = inputs.len() as u16;
    let mut b = TraceBuilder::new(TraceId(0), frame(count + 1));
    b.mark_written(FrameId(0), count);
    let values: Vec<ValueId> = inputs
        .iter()
        .enumerate()
        .map(|(register, ty)| b.input(register as u16, *ty))
        .collect();
    let out = b.emit(site(0), make(&values), Some(result)).unwrap();
    let mut registers: Vec<RegisterEntry> =
        values.iter().map(|v| RegisterEntry::Value(*v)).collect();
    registers.push(RegisterEntry::Value(out));
    let exit = b.snapshot(snapshot(after(0), 0, registers));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

/// Everything a run can observe.
#[derive(Debug, PartialEq)]
pub(super) struct Observed {
    outcome: TraceOutcome,
    control: TraceControl,
    area: Vec<i64>,
}

pub(super) fn compile(ir: &TraceIr, tier: OptimizationTier, mir: bool) -> CompiledTrace {
    FORCE_BACKEND.with(|force| force.set(Some(mir)));
    let compiled = compile_trace(ir, tier);
    FORCE_BACKEND.with(|force| force.set(None));
    compiled.unwrap_or_else(|error| panic!("compile (mir {mir}) at {tier:?}: {error}"))
}

fn run(trace: &CompiledTrace, registers: &[TraceScalar], fuel: i64) -> Observed {
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| TraceInputSlot::from_scalar(registers[usize::from(*register)]))
        .collect();
    let mut area = vec![0x5a5a_5a5a_i64; trace.exit_area_len()];
    let mut control = TraceControl::new(fuel);
    let outcome = trace.run(&inputs, &mut control, &mut area);
    // Scratch the MIR backend uses, not a result.
    control.scratch = 0;
    Observed {
        outcome,
        control,
        area,
    }
}

/// Compiles `ir` with both backends at both tiers, requires the MIR backend
/// to have made its code, and compares every run in `states`.
pub(super) fn same_everywhere(name: &str, ir: &TraceIr, states: &[Vec<TraceScalar>], fuel: i64) {
    ir.verify()
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    for tier in TIERS {
        let mir = compile(ir, tier, true);
        let cranelift = compile(ir, tier, false);
        assert!(
            mir.mir_address().is_some(),
            "{name}: the MIR backend compiled it"
        );
        assert!(
            cranelift.mir_address().is_none(),
            "{name}: Cranelift compiled it"
        );
        assert_eq!(
            mir.exits(),
            cranelift.exits(),
            "{name}: same exit descriptors"
        );
        assert_eq!(mir.exit_area_len(), cranelift.exit_area_len(), "{name}");
        for state in states {
            let (a, b) = (run(&mir, state, fuel), run(&cranelift, state, fuel));
            assert_eq!(a, b, "{name} at {tier:?} from {state:?}");
        }
    }
}

const FLOATS: [f64; 22] = [
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
    -9_007_199_254_740_991.0,
    2.0,
    6.0,
    7.0,
    -583.0,
    3.001,
    -1.777_777_777_777_777_7,
];

const INTEGERS: [i64; 14] = [
    0,
    1,
    -1,
    2,
    -2,
    7,
    -7,
    255,
    1 << 40,
    -(1 << 40),
    i64::MAX,
    i64::MIN,
    i64::MAX - 1,
    i64::MIN + 1,
];

fn float_pairs() -> Vec<Vec<TraceScalar>> {
    FLOATS
        .iter()
        .flat_map(|a| {
            FLOATS
                .iter()
                .map(move |b| vec![float(*a), float(*b), TraceScalar::Nil])
        })
        .collect()
}

fn integer_pairs() -> Vec<Vec<TraceScalar>> {
    INTEGERS
        .iter()
        .flat_map(|a| {
            INTEGERS.iter().map(move |b| {
                vec![
                    TraceScalar::Integer(*a),
                    TraceScalar::Integer(*b),
                    TraceScalar::Nil,
                ]
            })
        })
        .collect()
}

// ---- arithmetic and conversions -------------------------------------------------

#[test]
fn float_arithmetic_matches_cranelift() {
    for (name, op) in [
        ("add", ArithmeticOp::Add),
        ("subtract", ArithmeticOp::Subtract),
        ("multiply", ArithmeticOp::Multiply),
        ("divide", ArithmeticOp::Divide),
        ("modulo", ArithmeticOp::Modulo),
        (
            "modulo with floor",
            ArithmeticOp::ModuloWith(FloatModulo::Floor),
        ),
        ("floor divide", ArithmeticOp::FloorDivide),
    ] {
        let ir = op_trace(&[TraceType::Float; 2], TraceType::Float, |v| {
            TraceOp::Arithmetic {
                op,
                left: v[0],
                right: v[1],
            }
        });
        same_everywhere(name, &ir, &float_pairs(), 100);
    }
}

/// Operations that call a C function (the `pow` helper, the compat modulo,
/// the 32-bit number conversion) give what Cranelift's code gives for every
/// pair, with the call made by the MIR call op.
#[test]
fn calls_to_helper_functions_match_cranelift() {
    for (name, op) in [
        ("power", ArithmeticOp::Power),
        ("power square", ArithmeticOp::PowerSquare),
        ("fmod 5.3", ArithmeticOp::ModuloWith(FloatModulo::Fmod53)),
        ("fmod 5.4", ArithmeticOp::ModuloWith(FloatModulo::Fmod54)),
    ] {
        let ir = op_trace(&[TraceType::Float; 2], TraceType::Float, |v| {
            TraceOp::Arithmetic {
                op,
                left: v[0],
                right: v[1],
            }
        });
        same_everywhere(name, &ir, &float_pairs(), 100);
    }
    let floats: Vec<Vec<TraceScalar>> = FLOATS
        .iter()
        .chain(&[4_294_967_296.0, -4_294_967_295.0, 1e10, 3.5e9])
        .map(|x| vec![float(*x), TraceScalar::Nil])
        .collect();
    let ir = op_trace(&[TraceType::Float], TraceType::Int64, |v| TraceOp::ToWide {
        source: v[0],
        target: TraceType::Int64,
        conversion: crate::jit_ir::trace::WideConversion::Bits32,
    });
    same_everywhere("32-bit number bits", &ir, &floats, 100);
    // A call has live values on both sides of it: the loop keeps its float
    // accumulator across the call of `pow`.
    let ir = tree_with_a_pow_loop();
    let states: Vec<Vec<TraceScalar>> = [1.0, 2.0, 7.5, 50.0]
        .iter()
        .map(|start| vec![float(0.0), float(*start), TraceScalar::Nil])
        .collect();
    for fuel in [2, 5, 10_000] {
        same_everywhere(
            &format!("loop with a call, fuel {fuel}"),
            &ir,
            &states,
            fuel,
        );
    }
}

/// The targets the lowering has forms for: what the CPU has (the fused
/// remainder where it has FMA), AVX without FMA when the CPU has more, and
/// SSE2 alone.
fn targets() -> Vec<X64> {
    let detected = X64::detect();
    let mut targets = vec![detected];
    if detected.fma {
        targets.push(X64 {
            avx: true,
            fma: false,
            bmi2: detected.bmi2,
        });
    }
    if detected.avx {
        targets.push(X64::sse2());
    }
    targets
}

/// Operand pairs of the remainder tests: integral, fractional, huge, tiny,
/// and the ones that leave the in-line path.
fn remainder_pairs() -> Vec<Vec<TraceScalar>> {
    let mut pairs = float_pairs();
    for (a, b) in [
        (7.0, 3.0),
        (-7.0, 3.0),
        (7.0, -3.0),
        (-7.0, -3.0),
        (6.0, 3.0),
        (-6.0, 3.0),
        (6.0, -3.0),
        (-0.0, 3.0),
        (9_007_199_254_740_991.0, 2.0),
        (-9_007_199_254_740_991.0, 9_007_199_254_740_990.0),
        (9_007_199_254_740_992.0, 3.0),
        (3.0, 9_007_199_254_740_992.0),
        (7.5, 2.0),
        (7.0, 2.5),
        (-0.75, 0.25),
        (1e300, 7.0e290),
        (5e-324, 3.0),
    ] {
        pairs.push(vec![float(a), float(b), TraceScalar::Nil]);
    }
    pairs
}

/// The remainders of Lua 5.3 and later give what Cranelift's give for every
/// pair, on every form of the lowering (fused, integral without FMA).
#[test]
fn float_remainders_match_cranelift_on_every_target() {
    for target in targets() {
        with_target(target, || {
            for (name, op) in [
                ("fmod 5.3", ArithmeticOp::ModuloWith(FloatModulo::Fmod53)),
                ("fmod 5.4", ArithmeticOp::ModuloWith(FloatModulo::Fmod54)),
            ] {
                let ir = op_trace(&[TraceType::Float; 2], TraceType::Float, |v| {
                    TraceOp::Arithmetic {
                        op,
                        left: v[0],
                        right: v[1],
                    }
                });
                same_everywhere(
                    &format!("{name}, fma {}, avx {}", target.fma, target.avx),
                    &ir,
                    &remainder_pairs(),
                    100,
                );
            }
        });
    }
}

/// Whether both operands are integral, below 2^53 in magnitude, and the
/// divisor is not zero: the pairs the form without FMA takes in line.
fn integral_pair(a: f64, b: f64) -> bool {
    let whole = |x: f64| x.abs() < 9_007_199_254_740_992.0 && x.fract() == 0.0;
    whole(a) && whole(b) && b != 0.0
}

/// The remainders of Lua 5.3 and later answer in line and call the compat
/// function otherwise. With FMA that is every pair of finite operands with a
/// quotient below 2^53 (a zero or infinite divisor, NaN, an infinite dividend
/// and a huge quotient call), as Cranelift's fused form has it. Without FMA
/// it is two integral operands below 2^53 and a divisor that is not zero, as
/// Cranelift's integral form has it.
#[test]
fn float_remainders_call_the_helper_only_off_the_fast_path() {
    for target in targets() {
        with_target(target, || float_remainder_calls(target.fma));
    }
}

fn float_remainder_calls(fused: bool) {
    use super::lower::modulo::HELPER_CALLS;
    let calls_for = |op: ArithmeticOp, a: f64, b: f64| {
        let ir = op_trace(&[TraceType::Float; 2], TraceType::Float, |v| {
            TraceOp::Arithmetic {
                op,
                left: v[0],
                right: v[1],
            }
        });
        let mir = compile(&ir, OPTIMIZED, true);
        let before = HELPER_CALLS.with(Cell::get);
        run(&mir, &[float(a), float(b), TraceScalar::Nil], 100);
        HELPER_CALLS.with(Cell::get) - before
    };
    for op in [
        ArithmeticOp::ModuloWith(FloatModulo::Fmod53),
        ArithmeticOp::ModuloWith(FloatModulo::Fmod54),
    ] {
        for (a, b) in [
            (7.0, 3.0),
            (-7.0, 3.0),
            (7.0, -3.0),
            (-7.0, -3.0),
            (7.5, 2.25),
            (-0.75, 0.25),
            (6.0, 3.0),
            (-6.0, 3.0),
            (0.0, 5.0),
            (-0.0, 5.0),
            (9_007_199_254_740_991.0, 2.0),
            (9_007_199_254_740_992.0, 3.0),
            (1e300, 7.0e290),
            (5e-324, 3.0),
            (3.0, 1e300),
            (-3.0, 1e300),
        ] {
            let calls = calls_for(op, a, b);
            let in_line = fused || integral_pair(a, b);
            assert_eq!(calls, u64::from(!in_line), "{op:?} {a} % {b}, fma {fused}");
        }
        for (a, b) in [
            (7.0, 0.0),
            (7.0, -0.0),
            (f64::NAN, 3.0),
            (3.0, f64::NAN),
            (f64::INFINITY, 3.0),
            (f64::NEG_INFINITY, 3.0),
            (3.0, f64::INFINITY),
            (-3.0, f64::NEG_INFINITY),
            (1e300, 3.0),
            (9_007_199_254_740_993.0, 1.0),
        ] {
            // The fused form may fail both of its tests and call twice.
            let calls = calls_for(op, a, b);
            assert!(
                calls == 1 || (fused && calls == 2),
                "{op:?} {a} % {b}, fma {fused}: {calls} calls"
            );
        }
    }
}

/// `NumberModulo` of two numbers held as floats: the integer floor modulo when
/// the flag says both are integers (of magnitude at most 2^53, the divisor not
/// zero), the profile's float modulo otherwise.
#[test]
fn number_modulo_matches_cranelift() {
    let integers: [f64; 14] = [
        0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
        7.0,
        -7.0,
        255.0,
        1_099_511_627_776.0,
        -1_099_511_627_776.0,
        9_007_199_254_740_992.0,
        -9_007_199_254_740_992.0,
        3.0,
        -3.0,
    ];
    let mut states: Vec<Vec<TraceScalar>> = Vec::new();
    for a in integers {
        for b in integers.iter().filter(|b| **b != 0.0) {
            states.push(vec![float(a), float(*b), TraceScalar::Boolean(true)]);
        }
    }
    for (a, b) in float_pairs()
        .iter()
        .map(|state| (state[0], state[1]))
        .filter(|(_, b)| !matches!(b, TraceScalar::Float(b) if *b == 0.0))
    {
        states.push(vec![a, b, TraceScalar::Boolean(false)]);
    }
    for (name, float_op) in [
        ("floor", ArithmeticOp::Modulo),
        ("floor form", ArithmeticOp::ModuloWith(FloatModulo::Floor)),
        ("fmod 5.3", ArithmeticOp::ModuloWith(FloatModulo::Fmod53)),
        ("fmod 5.4", ArithmeticOp::ModuloWith(FloatModulo::Fmod54)),
    ] {
        let mut b = TraceBuilder::new(TraceId(0), frame(4));
        b.mark_written(FrameId(0), 3);
        let left = b.input(0, TraceType::Float);
        let right = b.input(1, TraceType::Float);
        let integer = b.input(2, TraceType::Boolean);
        let out = b
            .emit(
                site(0),
                TraceOp::NumberModulo {
                    float: float_op,
                    left,
                    right,
                    integer,
                },
                Some(TraceType::Float),
            )
            .unwrap();
        let registers = vec![
            RegisterEntry::Value(left),
            RegisterEntry::Value(right),
            RegisterEntry::Value(integer),
            RegisterEntry::Value(out),
        ];
        let exit = b.snapshot(snapshot(after(0), 0, registers));
        let ir = b.finish(TraceTerminator::Exit { snapshot: exit });
        for target in targets() {
            with_target(target, || {
                same_everywhere(
                    &format!("number modulo ({name}, fma {})", target.fma),
                    &ir,
                    &states,
                    100,
                );
            });
        }
    }
}

/// `no += i ^ 1.5; i += 1` until `i > 100`.
fn tree_with_a_pow_loop() -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), frame(3));
    for register in [0, 1] {
        b.mark_written(FrameId(0), register);
    }
    let no_in = b.input(0, TraceType::Float);
    let i_in = b.input(1, TraceType::Float);
    let no = b.loop_param(TraceType::Float, LoopParamInit::Input(no_in));
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let f = |b: &mut TraceBuilder, pc, op, ty| b.emit(site(pc), op, Some(ty)).unwrap();
    let power = f(&mut b, 0, TraceOp::Constant(float(1.5)), TraceType::Float);
    let raised = f(
        &mut b,
        0,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Power,
            left: i,
            right: power,
        },
        TraceType::Float,
    );
    let no2 = f(
        &mut b,
        1,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: no,
            right: raised,
        },
        TraceType::Float,
    );
    let one = f(&mut b, 2, TraceOp::Constant(float(1.0)), TraceType::Float);
    let i2 = f(
        &mut b,
        2,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: i,
            right: one,
        },
        TraceType::Float,
    );
    let top = f(&mut b, 2, TraceOp::Constant(float(100.0)), TraceType::Float);
    let more = f(
        &mut b,
        2,
        TraceOp::Compare {
            comparison: Comparison::LessEqual,
            left: i2,
            right: top,
        },
        TraceType::Boolean,
    );
    let regs = |no: ValueId, i: ValueId| {
        vec![
            RegisterEntry::Value(no),
            RegisterEntry::Value(i),
            RegisterEntry::Const(TraceScalar::Nil),
        ]
    };
    let snap = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 2,
            successor: 3,
        },
        2,
        regs(no2, i2),
    ));
    let guard = b.guard(site(2), snap);
    b.emit(
        site(2),
        TraceOp::GuardBranch {
            condition: more,
            expected: true,
            guard,
        },
        None,
    );
    let poll = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 2,
            successor: 0,
        },
        2,
        regs(no2, i2),
    ));
    b.set_poll_snapshot(poll);
    let mut ir = b.finish(TraceTerminator::LoopBack {
        args: vec![no2, i2],
    });
    ir.header_live = vec![0, 1];
    ir
}

/// `no += i; i += 1` until `i > 100`, and four values only the guard's and
/// the poll exit's snapshots keep: `c1 = no * no`, `c2 = -c1`, `c3 = c2 % i`
/// and (with `shared`) `c4 = c1 + 1.0`, which the loop-back takes too.
fn tree_with_cold_values(shared: bool) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), frame(7));
    for register in [0, 1] {
        b.mark_written(FrameId(0), register);
    }
    let no_in = b.input(0, TraceType::Float);
    let i_in = b.input(1, TraceType::Float);
    let no = b.loop_param(TraceType::Float, LoopParamInit::Input(no_in));
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let f = |b: &mut TraceBuilder, pc, op, ty| b.emit(site(pc), op, Some(ty)).unwrap();
    let arithmetic = |op, left, right| TraceOp::Arithmetic { op, left, right };
    let no2 = f(
        &mut b,
        0,
        arithmetic(ArithmeticOp::Add, no, i),
        TraceType::Float,
    );
    let one = f(&mut b, 1, TraceOp::Constant(float(1.0)), TraceType::Float);
    let i2 = f(
        &mut b,
        1,
        arithmetic(ArithmeticOp::Add, i, one),
        TraceType::Float,
    );
    let c1 = f(
        &mut b,
        2,
        arithmetic(ArithmeticOp::Multiply, no2, no2),
        TraceType::Float,
    );
    let c2 = f(&mut b, 2, TraceOp::Negate { source: c1 }, TraceType::Float);
    let c3 = f(
        &mut b,
        2,
        arithmetic(ArithmeticOp::Modulo, c2, i2),
        TraceType::Float,
    );
    let c4 = f(
        &mut b,
        2,
        arithmetic(ArithmeticOp::Add, c1, one),
        TraceType::Float,
    );
    let top = f(&mut b, 3, TraceOp::Constant(float(100.0)), TraceType::Float);
    let more = f(
        &mut b,
        3,
        TraceOp::Compare {
            comparison: Comparison::LessEqual,
            left: i2,
            right: top,
        },
        TraceType::Boolean,
    );
    let regs = |values: [ValueId; 6]| {
        let mut regs: Vec<RegisterEntry> = values.into_iter().map(RegisterEntry::Value).collect();
        regs.push(RegisterEntry::Const(TraceScalar::Nil));
        regs
    };
    let state = [no2, i2, c1, c2, c3, c4];
    let snap = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 3,
            successor: 4,
        },
        3,
        regs(state),
    ));
    let guard = b.guard(site(3), snap);
    b.emit(
        site(3),
        TraceOp::GuardBranch {
            condition: more,
            expected: true,
            guard,
        },
        None,
    );
    // With `shared` the back edge takes `c4`, so it stays in the loop.
    let carried = if shared { c4 } else { no2 };
    let poll = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 3,
            successor: 0,
        },
        3,
        regs([carried, i2, c1, c2, c3, c4]),
    ));
    b.set_poll_snapshot(poll);
    let mut ir = b.finish(TraceTerminator::LoopBack {
        args: vec![carried, i2],
    });
    ir.header_live = vec![0, 1];
    ir
}

/// A value only exits read is made in their stubs, from the operands the
/// snapshots keep alive, and gives what Cranelift's rematerialized code
/// gives; the loop does not compute it unless the back edge takes it too.
#[test]
fn values_only_exits_read_match_cranelift() {
    let states: Vec<Vec<TraceScalar>> = [(0.0, 1.0), (1.5, 98.0), (-2.5, 3.0), (7.0, 0.5)]
        .iter()
        .map(|(no, i)| vec![float(*no), float(*i), TraceScalar::Nil])
        .collect();
    for shared in [false, true] {
        let ir = tree_with_cold_values(shared);
        for fuel in [1, 2, 6, 10_000] {
            same_everywhere(
                &format!("cold values (shared {shared}), fuel {fuel}"),
                &ir,
                &states,
                fuel,
            );
        }
        // The stubs of the two exits compute the four values, seven MIR
        // operations each (the modulo is four). When the back edge takes the
        // sum, the product it reads stays in the loop too.
        compile(&ir, OPTIMIZED, true);
        let compiled = LAST_STATS.with(Cell::get).expect("a MIR compile");
        let (stubs, ops) = if shared { (2, 10) } else { (2, 14) };
        assert_eq!(
            (compiled.cold_stubs, compiled.cold_ops),
            (stubs, ops),
            "shared {shared}: {compiled:?}"
        );
    }
}

#[test]
fn integer_arithmetic_matches_cranelift() {
    for (name, op) in [
        ("add", ArithmeticOp::Add),
        ("subtract", ArithmeticOp::Subtract),
        ("multiply", ArithmeticOp::Multiply),
        ("and", ArithmeticOp::BitAnd),
        ("or", ArithmeticOp::BitOr),
        ("xor", ArithmeticOp::BitXor),
    ] {
        let ir = op_trace(&[TraceType::Integer; 2], TraceType::Integer, |v| {
            TraceOp::Arithmetic {
                op,
                left: v[0],
                right: v[1],
            }
        });
        same_everywhere(name, &ir, &integer_pairs(), 100);
    }
}

/// One unary case: its name, the operand and result type, and the op it
/// lowers.
type UnaryCase<'a> = (&'a str, TraceType, TraceType, fn(ValueId) -> TraceOp);

#[test]
fn unary_operations_and_conversions_match_cranelift() {
    let floats: Vec<Vec<TraceScalar>> = FLOATS
        .iter()
        .map(|x| vec![float(*x), TraceScalar::Nil])
        .collect();
    let integers: Vec<Vec<TraceScalar>> = INTEGERS
        .iter()
        .map(|x| vec![TraceScalar::Integer(*x), TraceScalar::Nil])
        .collect();
    let cases: [UnaryCase<'_>; 6] = [
        (
            "negate float",
            TraceType::Float,
            TraceType::Float,
            |source| TraceOp::Negate { source },
        ),
        ("floor", TraceType::Float, TraceType::Float, |source| {
            TraceOp::Numeric {
                op: NumericOp::Floor,
                source,
            }
        }),
        ("ceil", TraceType::Float, TraceType::Float, |source| {
            TraceOp::Numeric {
                op: NumericOp::Ceil,
                source,
            }
        }),
        ("abs", TraceType::Float, TraceType::Float, |source| {
            TraceOp::Numeric {
                op: NumericOp::Abs,
                source,
            }
        }),
        (
            "negate integer",
            TraceType::Integer,
            TraceType::Integer,
            |source| TraceOp::Negate { source },
        ),
        (
            "integer to float",
            TraceType::Integer,
            TraceType::Float,
            |source| TraceOp::IntToFloat { source },
        ),
    ];
    for (name, input, output, make) in cases {
        let ir = op_trace(&[input], output, |v| make(v[0]));
        let states = if input == TraceType::Float {
            &floats
        } else {
            &integers
        };
        same_everywhere(name, &ir, states, 100);
    }
}

// ---- division, shifts and 64-bit integers ----------------------------------------------

/// `r = x OP constant`, with the constant an entry-independent value.
fn constant_trace(op: ArithmeticOp, ty: TraceType, constant: TraceScalar) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), frame(3));
    b.mark_written(FrameId(0), 2);
    let x = b.input(0, ty);
    let k = b
        .emit(site(0), TraceOp::Constant(constant), Some(constant.ty()))
        .unwrap();
    let out = b
        .emit(
            site(0),
            TraceOp::Arithmetic {
                op,
                left: x,
                right: k,
            },
            Some(ty),
        )
        .unwrap();
    let exit = b.snapshot(snapshot(
        after(0),
        0,
        vec![
            RegisterEntry::Value(x),
            RegisterEntry::Unchanged,
            RegisterEntry::Value(out),
        ],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

fn integer_states() -> Vec<Vec<TraceScalar>> {
    INTEGERS
        .iter()
        .map(|x| vec![TraceScalar::Integer(*x), TraceScalar::Nil, TraceScalar::Nil])
        .collect()
}

#[test]
fn integer_division_and_modulo_match_cranelift() {
    for (name, op) in [
        ("modulo", ArithmeticOp::Modulo),
        ("floor divide", ArithmeticOp::FloorDivide),
    ] {
        // By a value: zero, minus one, the extremes and the rest.
        let ir = op_trace(&[TraceType::Integer; 2], TraceType::Integer, |v| {
            TraceOp::Arithmetic {
                op,
                left: v[0],
                right: v[1],
            }
        });
        same_everywhere(&format!("{name} by a value"), &ir, &integer_pairs(), 100);
        // By constants: powers of two, other positive ones, negative ones.
        for divisor in [
            1_i64,
            2,
            3,
            4,
            7,
            8,
            10,
            1000,
            1 << 40,
            (1 << 40) + 1,
            -3,
            -8,
            i64::MAX,
        ] {
            let ir = constant_trace(op, TraceType::Integer, TraceScalar::Integer(divisor));
            same_everywhere(&format!("{name} by {divisor}"), &ir, &integer_states(), 100);
        }
    }
}

#[test]
fn shifts_by_a_constant_match_cranelift() {
    for (name, op) in [
        ("left", ArithmeticOp::ShiftLeft),
        ("right", ArithmeticOp::ShiftRight),
    ] {
        for distance in [
            i64::MIN,
            -100,
            -64,
            -63,
            -5,
            -1,
            0,
            1,
            5,
            62,
            63,
            64,
            65,
            1000,
            i64::MAX,
        ] {
            let ir = constant_trace(op, TraceType::Integer, TraceScalar::Integer(distance));
            same_everywhere(
                &format!("shift {name} by {distance}"),
                &ir,
                &integer_states(),
                100,
            );
        }
    }
}

fn wide_states(ty: TraceType, pairs: bool) -> Vec<Vec<TraceScalar>> {
    let make = |bits: i64| match ty {
        TraceType::Int64 => TraceScalar::Int64(bits),
        _ => TraceScalar::UInt64(bits.cast_unsigned()),
    };
    if pairs {
        INTEGERS
            .iter()
            .flat_map(|a| {
                INTEGERS
                    .iter()
                    .map(move |b| vec![make(*a), make(*b), TraceScalar::Nil])
            })
            .collect()
    } else {
        INTEGERS
            .iter()
            .map(|a| vec![make(*a), TraceScalar::Nil, TraceScalar::Nil])
            .collect()
    }
}

#[test]
fn sixty_four_bit_integers_match_cranelift() {
    for ty in [TraceType::Int64, TraceType::UInt64] {
        for (name, op) in [
            ("add", ArithmeticOp::Add),
            ("subtract", ArithmeticOp::Subtract),
            ("multiply", ArithmeticOp::Multiply),
            ("and", ArithmeticOp::BitAnd),
            ("or", ArithmeticOp::BitOr),
            ("xor", ArithmeticOp::BitXor),
            ("divide", ArithmeticOp::Divide),
            ("modulo", ArithmeticOp::Modulo),
        ] {
            let ir = op_trace(&[ty; 2], ty, |v| TraceOp::Arithmetic {
                op,
                left: v[0],
                right: v[1],
            });
            same_everywhere(&format!("{ty:?} {name}"), &ir, &wide_states(ty, true), 100);
        }
        for comparison in [
            Comparison::Equal,
            Comparison::LessThan,
            Comparison::LessEqual,
        ] {
            for expected in [true, false] {
                let ir = guard_trace(ty, comparison, expected);
                let name = format!("{ty:?} {comparison:?} expected {expected}");
                same_everywhere(&name, &ir, &wide_states(ty, true), 100);
            }
        }
        for (name, op) in [
            ("shift left", ArithmeticOp::ShiftLeft),
            ("shift right", ArithmeticOp::ShiftRight),
            ("arithmetic shift right", ArithmeticOp::ArithmeticShiftRight),
            ("rotate left", ArithmeticOp::RotateLeft),
            ("rotate right", ArithmeticOp::RotateRight),
        ] {
            for count in [0_i64, 1, 13, 63, 64, 65, 200] {
                let constant = match ty {
                    TraceType::Int64 => TraceScalar::Int64(count),
                    _ => TraceScalar::UInt64(count.cast_unsigned()),
                };
                let ir = constant_trace(op, ty, constant);
                same_everywhere(
                    &format!("{ty:?} {name} by {count}"),
                    &ir,
                    &wide_states(ty, false),
                    100,
                );
            }
        }
        // Negation, truthiness and a conversion to a float.
        let ir = op_trace(&[ty], ty, |v| TraceOp::Negate { source: v[0] });
        same_everywhere(&format!("{ty:?} negate"), &ir, &wide_states(ty, false), 100);
    }
    for ty in [TraceType::Int64, TraceType::UInt64] {
        let ir = op_trace(&[ty], TraceType::Float, |v| TraceOp::IntToFloat {
            source: v[0],
        });
        let mut states = wide_states(ty, false);
        // Values around 2^63 and 2^64 where the rounding matters.
        for bits in [
            0x8000_0000_0000_0001_u64,
            0x8000_0000_0000_0400,
            0x8000_0000_0000_0401,
            0xffff_ffff_ffff_fbff,
            0xffff_ffff_ffff_fc00,
            0xffff_ffff_ffff_ffff,
            0x7fff_ffff_ffff_fe00,
        ] {
            let scalar = if ty == TraceType::Int64 {
                TraceScalar::Int64(bits.cast_signed())
            } else {
                TraceScalar::UInt64(bits)
            };
            states.push(vec![scalar, TraceScalar::Nil, TraceScalar::Nil]);
        }
        same_everywhere(&format!("{ty:?} to float"), &ir, &states, 100);
    }
}

#[test]
fn conversions_to_a_64_bit_integer_match_cranelift() {
    use crate::jit_ir::trace::WideConversion;
    let floats: Vec<Vec<TraceScalar>> = FLOATS
        .iter()
        .chain(&[
            9.2e18,
            -9.3e18,
            1.8e19,
            1.9e19,
            2.0e19,
            4.0e19,
            9_223_372_036_854_775_807.0,
        ])
        .map(|x| vec![float(*x), TraceScalar::Nil])
        .collect();
    for target in [TraceType::Int64, TraceType::UInt64] {
        let ir = op_trace(&[TraceType::Float], target, |v| TraceOp::ToWide {
            source: v[0],
            target,
            conversion: WideConversion::Truncate,
        });
        same_everywhere(&format!("float to {target:?}"), &ir, &floats, 100);
    }
    // Between the two signednesses the bits stay as they are.
    for (from, to) in [
        (TraceType::Int64, TraceType::UInt64),
        (TraceType::UInt64, TraceType::Int64),
    ] {
        let ir = op_trace(&[from], to, |v| TraceOp::ToWide {
            source: v[0],
            target: to,
            conversion: WideConversion::Truncate,
        });
        same_everywhere(
            &format!("{from:?} to {to:?}"),
            &ir,
            &wide_states(from, false),
            100,
        );
    }
}

// ---- comparisons, guards, selects ------------------------------------------------

/// `if not (a CMP b) == expected then leave`, then the terminal exit.
fn guard_trace(ty: TraceType, comparison: Comparison, expected: bool) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), frame(3));
    let left = b.input(0, ty);
    let right = b.input(1, ty);
    let condition = b
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
    let leave = b.snapshot(snapshot(
        after(0),
        0,
        vec![
            RegisterEntry::Value(left),
            RegisterEntry::Value(right),
            RegisterEntry::Value(condition),
        ],
    ));
    let guard = b.guard(site(0), leave);
    b.emit(
        site(0),
        TraceOp::GuardBranch {
            condition,
            expected,
            guard,
        },
        None,
    );
    let end = b.snapshot(snapshot(
        after(1),
        1,
        vec![
            RegisterEntry::Value(left),
            RegisterEntry::Value(right),
            RegisterEntry::Const(TraceScalar::Nil),
        ],
    ));
    b.finish(TraceTerminator::Exit { snapshot: end })
}

#[test]
fn compares_and_guard_branches_match_cranelift() {
    for (ty, states) in [
        (TraceType::Float, float_pairs()),
        (TraceType::Integer, integer_pairs()),
    ] {
        for comparison in [
            Comparison::Equal,
            Comparison::LessThan,
            Comparison::LessEqual,
        ] {
            for expected in [true, false] {
                let ir = guard_trace(ty, comparison, expected);
                let name = format!("{ty:?} {comparison:?} expected {expected}");
                same_everywhere(&name, &ir, &states, 100);
            }
        }
    }
}

#[test]
#[allow(
    clippy::many_single_char_names,
    reason = "the letters are the ones the comment names"
)]
fn selects_truthiness_and_booleans_match_cranelift() {
    // c = a < b, d = a == b; the results are a select between the inputs, the
    // negation and truthiness of c, c and d, and c itself as a stored value.
    let mut b = TraceBuilder::new(TraceId(0), frame(8));
    for register in 2..8 {
        b.mark_written(FrameId(0), register);
    }
    let x = b.input(0, TraceType::Integer);
    let y = b.input(1, TraceType::Integer);
    let mut emit = |op, ty| b.emit(site(0), op, Some(ty)).unwrap();
    let c = emit(
        TraceOp::Compare {
            comparison: Comparison::LessThan,
            left: x,
            right: y,
        },
        TraceType::Boolean,
    );
    let d = emit(
        TraceOp::Compare {
            comparison: Comparison::Equal,
            left: x,
            right: y,
        },
        TraceType::Boolean,
    );
    let picked = emit(
        TraceOp::Select {
            condition: c,
            if_true: x,
            if_false: y,
        },
        TraceType::Integer,
    );
    let not = emit(TraceOp::Not { source: c }, TraceType::Boolean);
    let truthy = emit(TraceOp::Truthy { source: d }, TraceType::Boolean);
    let both = emit(TraceOp::BoolAnd { left: c, right: d }, TraceType::Boolean);
    let exit_snapshot = b.snapshot(snapshot(
        after(0),
        0,
        vec![
            RegisterEntry::Value(x),
            RegisterEntry::Value(y),
            RegisterEntry::Value(picked),
            RegisterEntry::Value(not),
            RegisterEntry::Value(truthy),
            RegisterEntry::Value(both),
            RegisterEntry::Value(c),
            RegisterEntry::Value(d),
        ],
    ));
    let ir = b.finish(TraceTerminator::Exit {
        snapshot: exit_snapshot,
    });
    same_everywhere("select and booleans", &ir, &integer_pairs(), 100);
}

#[test]
fn a_float_comparison_stored_as_a_value_matches_cranelift() {
    for comparison in [
        Comparison::Equal,
        Comparison::LessThan,
        Comparison::LessEqual,
    ] {
        let ir = op_trace(&[TraceType::Float; 2], TraceType::Boolean, |v| {
            TraceOp::Compare {
                comparison,
                left: v[0],
                right: v[1],
            }
        });
        same_everywhere(
            &format!("{comparison:?} as a value"),
            &ir,
            &float_pairs(),
            100,
        );
    }
}

#[test]
fn selects_between_floats_match_cranelift() {
    // r = (x < y) ? x : y on floats, and the same on a float compare of
    // integers converted, as `math.min` and `a and b or c` record.
    for comparison in [
        Comparison::Equal,
        Comparison::LessThan,
        Comparison::LessEqual,
    ] {
        let mut b = TraceBuilder::new(TraceId(0), frame(4));
        b.mark_written(FrameId(0), 2);
        b.mark_written(FrameId(0), 3);
        let x = b.input(0, TraceType::Float);
        let y = b.input(1, TraceType::Float);
        let c = b
            .emit(
                site(0),
                TraceOp::Compare {
                    comparison,
                    left: x,
                    right: y,
                },
                Some(TraceType::Boolean),
            )
            .unwrap();
        let picked = b
            .emit(
                site(0),
                TraceOp::Select {
                    condition: c,
                    if_true: x,
                    if_false: y,
                },
                Some(TraceType::Float),
            )
            .unwrap();
        let negated = b
            .emit(
                site(0),
                TraceOp::Not { source: c },
                Some(TraceType::Boolean),
            )
            .unwrap();
        let other = b
            .emit(
                site(0),
                TraceOp::Select {
                    condition: negated,
                    if_true: y,
                    if_false: x,
                },
                Some(TraceType::Float),
            )
            .unwrap();
        let exit = b.snapshot(snapshot(
            after(0),
            0,
            vec![
                RegisterEntry::Value(x),
                RegisterEntry::Value(y),
                RegisterEntry::Value(picked),
                RegisterEntry::Value(other),
            ],
        ));
        let ir = b.finish(TraceTerminator::Exit { snapshot: exit });
        same_everywhere(
            &format!("float select on {comparison:?}"),
            &ir,
            &float_pairs(),
            100,
        );
    }
}

// ---- loops, fuel, entry checks ----------------------------------------------------

/// `x = x + 1` until `x > limit`: r0 x, r1 limit.
fn counting_loop() -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), frame(2));
    b.mark_written(FrameId(0), 0);
    let x_in = b.input(0, TraceType::Float);
    let limit = b.input(1, TraceType::Float);
    let x = b.loop_param(TraceType::Float, LoopParamInit::Input(x_in));
    let one = b
        .emit(
            site(0),
            TraceOp::Constant(float(1.0)),
            Some(TraceType::Float),
        )
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
    let leave = b.snapshot(snapshot(
        after(1),
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
fn a_loop_spends_fuel_and_leaves_through_its_poll_exit_like_cranelift() {
    let ir = counting_loop();
    let states: Vec<Vec<TraceScalar>> = [
        (0.0, 10.0),
        (0.0, 1.0e9),
        (5.0, 3.0),
        (f64::NAN, 4.0),
        (-3.0, 2.0),
    ]
    .iter()
    .map(|(x, limit)| vec![float(*x), float(*limit)])
    .collect();
    for fuel in [1, 2, 3, 5, 11, 12, 64, 1_000] {
        same_everywhere(
            &format!("counting loop with fuel {fuel}"),
            &ir,
            &states,
            fuel,
        );
    }
}

#[test]
fn entry_mismatches_run_nothing_and_leave_the_buffers_alone() {
    let ir = op_trace(
        &[TraceType::Float, TraceType::Integer],
        TraceType::Float,
        |v| TraceOp::IntToFloat { source: v[1] },
    );
    let mir = compile(&ir, OPTIMIZED, true);
    assert!(mir.mir_address().is_some());
    let cases = [
        [TraceInputSlot::integer(1), TraceInputSlot::integer(2)],
        [
            TraceInputSlot::from_scalar(float(1.0)),
            TraceInputSlot::from_scalar(float(1.0)),
        ],
        [TraceInputSlot::nil(), TraceInputSlot::integer(2)],
        [
            TraceInputSlot::from_scalar(float(1.0)),
            TraceInputSlot::boolean(true),
        ],
        [TraceInputSlot::other(), TraceInputSlot::other()],
    ];
    for inputs in cases {
        // The run reports the first input whose tag differs.
        let first = mir
            .entry_contract()
            .iter()
            .zip(&inputs)
            .position(|((_, ty), slot)| super::super::exit::entry_tag(*ty) != Some(slot.tag()))
            .expect("a case that mismatches");
        let mut area = vec![-1_i64; mir.exit_area_len()];
        let mut control = TraceControl::new(77);
        assert_eq!(
            mir.run(&inputs, &mut control, &mut area),
            TraceOutcome::EntryMismatch { input: first }
        );
        assert_eq!(control.fuel, 77);
        assert!(area.iter().all(|word| *word == -1));
    }
}

// ---- trees --------------------------------------------------------------------------

/// A root loop (`r2 % 7 < 3` guards, `no += i`) whose guard continues into a
/// side that leaves through an exit of its own (a `break`).
#[allow(
    clippy::many_single_char_names,
    reason = "the letters are the registers of the loop the doc comment spells"
)]
fn tree_with_a_leaving_side() -> TraceIr {
    let limit = 100.0;
    let mut b = TraceBuilder::new(TraceId(0), frame(4));
    for register in [1, 2, 3] {
        b.mark_written(FrameId(0), register);
    }
    let no_in = b.input(1, TraceType::Float);
    let i_in = b.input(2, TraceType::Float);
    let no = b.loop_param(TraceType::Float, LoopParamInit::Input(no_in));
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let f = |b: &mut TraceBuilder, pc, op, ty| b.emit(site(pc), op, Some(ty)).unwrap();
    let seven = f(&mut b, 0, TraceOp::Constant(float(7.0)), TraceType::Float);
    let rem = f(
        &mut b,
        0,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Modulo,
            left: i,
            right: seven,
        },
        TraceType::Float,
    );
    let three = f(&mut b, 1, TraceOp::Constant(float(3.0)), TraceType::Float);
    let c = f(
        &mut b,
        1,
        TraceOp::Compare {
            comparison: Comparison::LessThan,
            left: rem,
            right: three,
        },
        TraceType::Boolean,
    );
    let snap = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 2,
            successor: 3,
        },
        2,
        vec![
            RegisterEntry::Unchanged,
            RegisterEntry::Value(no),
            RegisterEntry::Value(i),
            RegisterEntry::Value(c),
        ],
    ));
    let guard = b.guard(site(2), snap);
    b.emit(
        site(2),
        TraceOp::GuardBranch {
            condition: c,
            expected: false,
            guard,
        },
        None,
    );
    let no2 = f(
        &mut b,
        5,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: no,
            right: i,
        },
        TraceType::Float,
    );
    let one = f(&mut b, 6, TraceOp::Constant(float(1.0)), TraceType::Float);
    let i2 = f(
        &mut b,
        6,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: i,
            right: one,
        },
        TraceType::Float,
    );
    let top = f(&mut b, 6, TraceOp::Constant(float(limit)), TraceType::Float);
    let more = f(
        &mut b,
        6,
        TraceOp::Compare {
            comparison: Comparison::LessEqual,
            left: i2,
            right: top,
        },
        TraceType::Boolean,
    );
    let regs = |no: ValueId, i: ValueId, c: ValueId| {
        vec![
            RegisterEntry::Unchanged,
            RegisterEntry::Value(no),
            RegisterEntry::Value(i),
            RegisterEntry::Value(c),
        ]
    };
    let snap = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 6,
            successor: 7,
        },
        6,
        regs(no2, i2, c),
    ));
    let guard = b.guard(site(6), snap);
    b.emit(
        site(6),
        TraceOp::GuardBranch {
            condition: more,
            expected: true,
            guard,
        },
        None,
    );
    let poll = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        6,
        regs(no2, i2, c),
    ));
    b.set_poll_snapshot(poll);
    let mut ir = b.finish(TraceTerminator::LoopBack {
        args: vec![no2, i2],
    });
    ir.header_live = vec![0, 1, 2];
    // The side: `yes = yes + i`, then leave the loop.
    let mut s = TraceBuilder::new(TraceId(1), frame(4));
    s.mark_written(FrameId(0), 0);
    let yes = s.input(0, TraceType::Float);
    let no = s.input(1, TraceType::Float);
    let i = s.input(2, TraceType::Float);
    let yes2 = s
        .emit(
            site(3),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: yes,
                right: i,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let out = s.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 4,
            successor: 12,
        },
        4,
        vec![
            RegisterEntry::Value(yes2),
            RegisterEntry::Value(no),
            RegisterEntry::Value(i),
            RegisterEntry::Const(TraceScalar::Nil),
        ],
    ));
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(s.finish(TraceTerminator::Exit { snapshot: out })),
    });
    ir
}

#[test]
fn a_side_that_ends_in_an_exit_of_its_own_runs_through_mir_like_cranelift() {
    let ir = tree_with_a_leaving_side();
    let states: Vec<Vec<TraceScalar>> = [1.0, 2.0, 3.0, 4.0, 8.0, 50.5, 99.0, 100.0, f64::NAN]
        .iter()
        .flat_map(|start| {
            [0.0, 5.0].map(|yes| vec![float(yes), float(6.0), float(*start), TraceScalar::Nil])
        })
        .collect();
    for fuel in [2, 3, 10_000] {
        same_everywhere(
            &format!("tree with a leaving side, fuel {fuel}"),
            &ir,
            &states,
            fuel,
        );
    }
    let mir = compile(&ir, OPTIMIZED, true);
    assert_eq!(mir.member_count(), 2, "the side is part of the tree");
}

/// The tree of `tree_with_a_leaving_side`, with a side that does one step of
/// the loop itself (`no += i`, `i += 1`) and links back to the root's loop.
fn tree_with_a_rejoining_side() -> TraceIr {
    let mut ir = tree_with_a_leaving_side();
    let mut s = TraceBuilder::new(TraceId(1), frame(4));
    for register in [1, 2] {
        s.mark_written(FrameId(0), register);
    }
    let no = s.input(1, TraceType::Float);
    let i = s.input(2, TraceType::Float);
    let no2 = s
        .emit(
            site(3),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: no,
                right: i,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let one = s
        .emit(
            site(3),
            TraceOp::Constant(float(1.0)),
            Some(TraceType::Float),
        )
        .unwrap();
    let i2 = s
        .emit(
            site(3),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: i,
                right: one,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let regs = |no: ValueId, i: ValueId| {
        vec![
            RegisterEntry::Unchanged,
            RegisterEntry::Value(no),
            RegisterEntry::Value(i),
            RegisterEntry::Const(TraceScalar::Nil),
        ]
    };
    let link = s.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 4,
            successor: 0,
        },
        4,
        regs(no2, i2),
    ));
    *ir.sides[0].ir = s.finish(TraceTerminator::Link { snapshot: link });
    ir
}

#[test]
fn a_side_that_rejoins_the_loop_runs_through_mir_like_cranelift() {
    let ir = tree_with_a_rejoining_side();
    let states: Vec<Vec<TraceScalar>> = [1.0, 2.0, 3.0, 4.0, 8.0, 50.5, 99.0, 100.0, f64::NAN]
        .iter()
        .flat_map(|start| {
            [0.0, 5.0].map(|yes| vec![float(yes), float(6.0), float(*start), TraceScalar::Nil])
        })
        .collect();
    for fuel in [1, 2, 3, 4, 7, 10, 10_000] {
        same_everywhere(
            &format!("tree with a rejoining side, fuel {fuel}"),
            &ir,
            &states,
            fuel,
        );
    }
    let mir = compile(&ir, OPTIMIZED, true);
    assert_eq!(mir.member_count(), 2, "the side is part of the tree");
}

/// A root loop with a parameter that starts from a constant (register 3,
/// which no entry reads), a value the optimizer hoists (`limit + 1`, an
/// invariant of the entry's `limit`), and a side on the loop's branch that
/// does a step of its own and links back. The loop changes every parameter,
/// so the `Join` of the sides sits after the preheader.
#[allow(
    clippy::many_single_char_names,
    reason = "the letters are the registers of the loop the doc comment spells"
)]
fn tree_with_a_constant_start() -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), frame(4));
    for register in [1, 2, 3] {
        b.mark_written(FrameId(0), register);
    }
    let limit = b.input(0, TraceType::Float);
    let no_in = b.input(1, TraceType::Float);
    let i_in = b.input(2, TraceType::Float);
    let no = b.loop_param(TraceType::Float, LoopParamInit::Input(no_in));
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let c = b.loop_param(TraceType::Float, LoopParamInit::Const(float(40.0)));
    let f = |b: &mut TraceBuilder, pc, op, ty| b.emit(site(pc), op, Some(ty)).unwrap();
    let one = f(&mut b, 0, TraceOp::Constant(float(1.0)), TraceType::Float);
    let top = f(
        &mut b,
        0,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: limit,
            right: one,
        },
        TraceType::Float,
    );
    let seven = f(&mut b, 0, TraceOp::Constant(float(7.0)), TraceType::Float);
    let rem = f(
        &mut b,
        0,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Modulo,
            left: i,
            right: seven,
        },
        TraceType::Float,
    );
    let three = f(&mut b, 1, TraceOp::Constant(float(3.0)), TraceType::Float);
    let cond = f(
        &mut b,
        1,
        TraceOp::Compare {
            comparison: Comparison::LessThan,
            left: rem,
            right: three,
        },
        TraceType::Boolean,
    );
    let regs = |no: ValueId, i: ValueId, c: ValueId| {
        vec![
            RegisterEntry::Unchanged,
            RegisterEntry::Value(no),
            RegisterEntry::Value(i),
            RegisterEntry::Value(c),
        ]
    };
    let snap = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 2,
            successor: 3,
        },
        2,
        regs(no, i, c),
    ));
    let guard = b.guard(site(2), snap);
    b.emit(
        site(2),
        TraceOp::GuardBranch {
            condition: cond,
            expected: false,
            guard,
        },
        None,
    );
    let step = |b: &mut TraceBuilder, pc, left, right| {
        f(
            b,
            pc,
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left,
                right,
            },
            TraceType::Float,
        )
    };
    let no2 = step(&mut b, 5, no, i);
    let c2 = step(&mut b, 5, c, one);
    let i2 = step(&mut b, 6, i, one);
    let more = f(
        &mut b,
        6,
        TraceOp::Compare {
            comparison: Comparison::LessEqual,
            left: i2,
            right: top,
        },
        TraceType::Boolean,
    );
    let exit = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 6,
            successor: 7,
        },
        6,
        regs(no2, i2, c2),
    ));
    let guard = b.guard(site(6), exit);
    b.emit(
        site(6),
        TraceOp::GuardBranch {
            condition: more,
            expected: true,
            guard,
        },
        None,
    );
    let poll = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        6,
        regs(no2, i2, c2),
    ));
    b.set_poll_snapshot(poll);
    let mut ir = b.finish(TraceTerminator::LoopBack {
        args: vec![no2, i2, c2],
    });
    ir.loop_params[2].register = Some(3);
    ir.header_live = vec![0, 1, 2, 3];
    // The side: `no += i; c += 100; i += 1`, then the loop again.
    let mut s = TraceBuilder::new(TraceId(1), frame(4));
    for register in [1, 2, 3] {
        s.mark_written(FrameId(0), register);
    }
    let no = s.input(1, TraceType::Float);
    let i = s.input(2, TraceType::Float);
    let c = s.input(3, TraceType::Float);
    let hundred = f(&mut s, 3, TraceOp::Constant(float(100.0)), TraceType::Float);
    let one = f(&mut s, 3, TraceOp::Constant(float(1.0)), TraceType::Float);
    let no2 = step(&mut s, 3, no, i);
    let c2 = step(&mut s, 3, c, hundred);
    let i2 = step(&mut s, 3, i, one);
    let link = s.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 4,
            successor: 0,
        },
        4,
        regs(no2, i2, c2),
    ));
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(s.finish(TraceTerminator::Link { snapshot: link })),
    });
    ir
}

#[test]
fn a_side_that_rejoins_a_loop_with_a_constant_start_runs_through_mir_like_cranelift() {
    let ir = tree_with_a_constant_start();
    let states: Vec<Vec<TraceScalar>> = [20.0, 40.0, 99.0, f64::NAN, -3.0]
        .iter()
        .flat_map(|limit| {
            [1.0, 2.0, 3.0, 4.0, 8.0, 50.5, 99.0]
                .map(|start| vec![float(*limit), float(6.0), float(start), TraceScalar::Nil])
        })
        .collect();
    for fuel in [1, 2, 3, 4, 7, 10, 10_000] {
        same_everywhere(
            &format!("tree with a constant start, fuel {fuel}"),
            &ir,
            &states,
            fuel,
        );
    }
    let mir = compile(&ir, OPTIMIZED, true);
    assert_eq!(mir.member_count(), 2, "the side is part of the tree");
    assert!(
        !mir.lowered_ir().preheader.is_empty(),
        "limit + 1 is hoisted, and the join sits after what the preheader made"
    );
}

#[test]
fn a_hoisted_check_on_a_register_only_a_side_writes_runs_through_mir_like_cranelift() {
    // `yes` is a parameter the loop passes on, which only the side changes:
    // the preheader, which holds a check on it, runs again after every link
    // and the join sits before it.
    let ir = super::super::tree_tests::tree_with_a_check_on_yes(400.0);
    let states: Vec<Vec<TraceScalar>> = [0.0, 10.0, 30.0, 47.0, 49.0, 50.0, 80.0]
        .iter()
        .flat_map(|yes| {
            [1.0, 2.0, 4.0, 9.0, 11.0]
                .map(|start| vec![float(*yes), float(3.0), float(start), TraceScalar::Nil])
        })
        .collect();
    for fuel in [1, 2, 5, 9, 40, 100_000] {
        same_everywhere(
            &format!("check on a register a side writes, fuel {fuel}"),
            &ir,
            &states,
            fuel,
        );
    }
    let mir = compile(&ir, OPTIMIZED, true);
    assert!(
        mir.lowered_ir()
            .preheader
            .iter()
            .any(|instruction| instruction.op.guard().is_some()),
        "the check sits in the preheader"
    );
}

// ---- a call into the host hoisted above the loop of a root with rejoining sides -----

/// A root whose preheader holds a call into the host, on the entry input
/// `yes` (r0), and a side that does a step of the loop itself and links back.
/// The loop ends when its counter passes what the call answered.
///
/// With `side_writes_yes` the side adds the counter to `yes` too, which makes
/// `yes` a loop parameter that the loop passes on unchanged: the call depends
/// on a parameter only a link changes, so Cranelift runs it again at its outer
/// header after every link, and the `Join` of the sides sits before the
/// preheader. Without it every parameter changes in the loop, the call runs
/// once and a link jumps past it to the loop header.
fn tree_with_a_hoisted_call(side_writes_yes: bool) -> TraceIr {
    use crate::jit_ir::trace::HostFunction;
    let mut b = TraceBuilder::new(TraceId(0), frame(4));
    for register in [1, 2] {
        b.mark_written(FrameId(0), register);
    }
    let yes = b.input(0, TraceType::Float);
    let no_in = b.input(1, TraceType::Float);
    let i_in = b.input(2, TraceType::Float);
    let no = b.loop_param(TraceType::Float, LoopParamInit::Input(no_in));
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let regs = |no: ValueId, i: ValueId| {
        vec![
            RegisterEntry::Unchanged,
            RegisterEntry::Value(no),
            RegisterEntry::Value(i),
            RegisterEntry::Const(TraceScalar::Nil),
        ]
    };
    let f = |b: &mut TraceBuilder, pc, op, ty| b.emit(site(pc), op, Some(ty)).unwrap();
    let before = b.snapshot(snapshot(SnapshotPhase::Before { pc: 0 }, 0, regs(no, i)));
    let call_guard = b.guard(site(0), before);
    let top = f(
        &mut b,
        0,
        TraceOp::HostCall {
            function: HostFunction::Native(7),
            args: vec![yes],
            result: TraceType::Float,
            guard: call_guard,
        },
        TraceType::Float,
    );
    let seven = f(&mut b, 0, TraceOp::Constant(float(7.0)), TraceType::Float);
    let rem = f(
        &mut b,
        0,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Modulo,
            left: i,
            right: seven,
        },
        TraceType::Float,
    );
    let three = f(&mut b, 1, TraceOp::Constant(float(3.0)), TraceType::Float);
    let cond = f(
        &mut b,
        1,
        TraceOp::Compare {
            comparison: Comparison::LessThan,
            left: rem,
            right: three,
        },
        TraceType::Boolean,
    );
    let snap = b.snapshot(snapshot(after_pc(2, 3), 2, regs(no, i)));
    let guard = b.guard(site(2), snap);
    b.emit(
        site(2),
        TraceOp::GuardBranch {
            condition: cond,
            expected: false,
            guard,
        },
        None,
    );
    let step = |b: &mut TraceBuilder, pc, left, right| {
        f(
            b,
            pc,
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left,
                right,
            },
            TraceType::Float,
        )
    };
    let one = f(&mut b, 5, TraceOp::Constant(float(1.0)), TraceType::Float);
    let no2 = step(&mut b, 5, no, i);
    let i2 = step(&mut b, 6, i, one);
    let more = f(
        &mut b,
        6,
        TraceOp::Compare {
            comparison: Comparison::LessEqual,
            left: i2,
            right: top,
        },
        TraceType::Boolean,
    );
    let exit = b.snapshot(snapshot(after_pc(6, 7), 6, regs(no2, i2)));
    let guard = b.guard(site(6), exit);
    b.emit(
        site(6),
        TraceOp::GuardBranch {
            condition: more,
            expected: true,
            guard,
        },
        None,
    );
    let poll = b.snapshot(snapshot(after_pc(6, 0), 6, regs(no2, i2)));
    b.set_poll_snapshot(poll);
    let mut ir = b.finish(TraceTerminator::LoopBack {
        args: vec![no2, i2],
    });
    ir.header_live = vec![0, 1, 2];
    // The side: `no += i; i += 1` (and `yes += i`), then the loop again. It
    // continues the root's branch, which is the second guard.
    let mut s = TraceBuilder::new(TraceId(1), frame(4));
    for register in [0, 1, 2] {
        if register != 0 || side_writes_yes {
            s.mark_written(FrameId(0), register);
        }
    }
    let yes = s.input(0, TraceType::Float);
    let no = s.input(1, TraceType::Float);
    let i = s.input(2, TraceType::Float);
    let one = f(&mut s, 3, TraceOp::Constant(float(1.0)), TraceType::Float);
    let no2 = step(&mut s, 3, no, i);
    let yes2 = step(&mut s, 3, yes, i);
    let i2 = step(&mut s, 3, i, one);
    let link = s.snapshot(snapshot(
        after_pc(4, 0),
        4,
        vec![
            if side_writes_yes {
                RegisterEntry::Value(yes2)
            } else {
                RegisterEntry::Unchanged
            },
            RegisterEntry::Value(no2),
            RegisterEntry::Value(i2),
            RegisterEntry::Const(TraceScalar::Nil),
        ],
    ));
    ir.sides.push(TraceSide {
        parent: None,
        guard: 1,
        ir: Box::new(s.finish(TraceTerminator::Link { snapshot: link })),
    });
    ir
}

fn after_pc(pc: usize, successor: usize) -> SnapshotPhase {
    SnapshotPhase::After { pc, successor }
}

/// A host that answers `Native(7)` with twice its argument, and counts.
#[derive(Default)]
struct CountingHost {
    calls: u32,
}

impl super::super::exit::TraceTables for CountingHost {
    fn lane(&mut self, _: &crate::runtime::Value) -> Option<usize> {
        None
    }

    fn row(&mut self, _: usize, row: &mut [usize]) {
        row.fill(0);
    }

    fn host_call(
        &mut self,
        function: crate::jit_ir::trace::HostFunction,
        args: &[TraceInputSlot],
        out: &mut TraceInputSlot,
        _: &mut [u32],
    ) -> bool {
        if function != crate::jit_ir::trace::HostFunction::Native(7) {
            return false;
        }
        self.calls += 1;
        *out = TraceInputSlot::from_scalar(float(f64::from_bits(args[0].bits as u64) * 2.0));
        true
    }
}

pub(super) fn run_hosted(
    trace: &CompiledTrace,
    registers: &[TraceScalar],
    fuel: i64,
) -> (Observed, u32) {
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| TraceInputSlot::from_scalar(registers[usize::from(*register)]))
        .collect();
    let mut area = vec![0x5a5a_5a5a_i64; trace.exit_area_len()];
    let mut control = TraceControl::new(fuel);
    let mut space = super::super::exit::TraceTableSpace::new();
    let mut host = CountingHost::default();
    let outcome =
        trace.run_with_tables(&inputs, &[], &mut control, &mut area, &mut space, &mut host);
    (
        Observed {
            outcome,
            control,
            area,
        },
        host.calls,
    )
}

/// Runs the tree over many states and fuels under both backends and compares
/// the outcome, the control block, the exit area and how often the host was
/// called. Returns the most calls one run of the optimized code made.
fn same_host_calls(name: &str, ir: &TraceIr, yes: &[f64]) -> u32 {
    ir.verify()
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    let mut most = 0;
    for tier in TIERS {
        let mir = compile(ir, tier, true);
        let cranelift = compile(ir, tier, false);
        assert!(mir.mir_address().is_some(), "{name}: MIR compiled it");
        assert!(cranelift.mir_address().is_none(), "{name}");
        assert_eq!(
            mir.member_count(),
            2,
            "{name}: the side is part of the tree"
        );
        assert_eq!(mir.exits(), cranelift.exits(), "{name}: same exits");
        if tier == OPTIMIZED {
            for trace in [&mir, &cranelift] {
                assert!(
                    trace
                        .lowered_ir()
                        .preheader
                        .iter()
                        .any(|instruction| matches!(instruction.op, TraceOp::HostCall { .. })),
                    "{name}: the call is hoisted"
                );
            }
        }
        for &yes in yes {
            for start in [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 50.5, 99.0] {
                for fuel in [1, 2, 3, 5, 8, 20, 10_000] {
                    let state = [float(yes), float(6.0), float(start), TraceScalar::Nil];
                    let (a, calls_a) = run_hosted(&mir, &state, fuel);
                    let (b, calls_b) = run_hosted(&cranelift, &state, fuel);
                    assert_eq!(a, b, "{name} at {tier:?} from {state:?}, fuel {fuel}");
                    assert_eq!(
                        calls_a, calls_b,
                        "{name}: host calls at {tier:?} from {state:?}, fuel {fuel}"
                    );
                    if tier == OPTIMIZED {
                        most = most.max(calls_a);
                    }
                }
            }
        }
    }
    most
}

#[test]
fn a_call_hoisted_over_a_parameter_a_side_changes_runs_again_after_each_link_like_cranelift() {
    // The join sits before the preheader: the call runs at the start and
    // again after every link.
    let ir = tree_with_a_hoisted_call(true);
    let most = same_host_calls("call on a parameter a side writes", &ir, &[4.0, 10.0, 30.0]);
    assert!(most > 3, "links ran the call again ({most} calls)");
}

#[test]
fn a_call_hoisted_over_an_entry_input_runs_once_however_often_a_side_links_like_cranelift() {
    // Every parameter changes in the loop: the join sits after the preheader,
    // and a link goes to the loop header without the call.
    let ir = tree_with_a_hoisted_call(false);
    let most = same_host_calls("call on an entry input", &ir, &[4.0, 10.0, 30.0]);
    assert_eq!(most, 1, "the call runs once per run");
}

// ---- the length of a string held by lane ---------------------------------------------

/// A host with a length table for the lanes the run starts with, and a VM
/// answer for the root lanes (which the table does not hold): their length is
/// `7 * (lane - ROOT_STRING_LANE_BASE) + 2`, in the type the trace expects, or
/// a boolean when `wrong_tag` is set.
struct LengthHost {
    calls: u32,
    wrong_tag: bool,
    integer: bool,
}

impl super::super::exit::TraceTables for LengthHost {
    fn lane(&mut self, _: &crate::runtime::Value) -> Option<usize> {
        None
    }

    fn row(&mut self, _: usize, row: &mut [usize]) {
        row.fill(0);
    }

    fn string_lengths(&mut self, lengths: &mut [u32]) {
        for (lane, length) in lengths.iter_mut().enumerate().take(64) {
            *length = 3 * lane as u32 + 1;
        }
    }

    fn host_call(
        &mut self,
        function: crate::jit_ir::trace::HostFunction,
        args: &[TraceInputSlot],
        out: &mut TraceInputSlot,
        _: &mut [u32],
    ) -> bool {
        use crate::jit_ir::trace::{HostFunction, ROOT_STRING_LANE_BASE};
        if function != HostFunction::StringLength {
            return false;
        }
        self.calls += 1;
        let lane = args[0].bits as u64;
        let length = 7 * (lane - ROOT_STRING_LANE_BASE) + 2;
        *out = if self.wrong_tag {
            TraceInputSlot::from_scalar(TraceScalar::Boolean(true))
        } else if self.integer {
            TraceInputSlot::from_scalar(TraceScalar::Integer(length as i64))
        } else {
            TraceInputSlot::from_scalar(float(length as f64))
        };
        true
    }
}

/// `r1 = #r0` as a host call of `StringLength` on a string held by lane, with
/// a result of type `result`, then an exit.
fn string_length_trace(result: TraceType) -> TraceIr {
    use crate::jit_ir::trace::HostFunction;
    let mut b = TraceBuilder::new(TraceId(0), frame(2));
    b.mark_written(FrameId(0), 1);
    let text = b.input(0, TraceType::String);
    let before = b.snapshot(snapshot(
        SnapshotPhase::Before { pc: 0 },
        0,
        vec![
            RegisterEntry::Value(text),
            RegisterEntry::Const(TraceScalar::Nil),
        ],
    ));
    let guard = b.guard(site(0), before);
    let length = b
        .emit(
            site(0),
            TraceOp::HostCall {
                function: HostFunction::StringLength,
                args: vec![text],
                result,
                guard,
            },
            Some(result),
        )
        .unwrap();
    let exit = b.snapshot(snapshot(
        after(0),
        0,
        vec![RegisterEntry::Value(text), RegisterEntry::Value(length)],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

/// `sum += #text` over `n` rounds, the length asked for in every round: the
/// skip and its join inside a loop.
fn string_length_loop() -> TraceIr {
    use crate::jit_ir::trace::{HostFunction, LoopParamInit};
    let mut b = TraceBuilder::new(TraceId(0), frame(4));
    for register in [1, 2] {
        b.mark_written(FrameId(0), register);
    }
    let text = b.input(0, TraceType::String);
    let sum_in = b.input(1, TraceType::Float);
    let i_in = b.input(2, TraceType::Float);
    let n = b.input(3, TraceType::Float);
    let sum = b.loop_param(TraceType::Float, LoopParamInit::Input(sum_in));
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let regs = |sum: ValueId, i: ValueId| {
        vec![
            RegisterEntry::Value(text),
            RegisterEntry::Value(sum),
            RegisterEntry::Value(i),
            RegisterEntry::Value(n),
        ]
    };
    let f = |b: &mut TraceBuilder, pc, op, ty| b.emit(site(pc), op, Some(ty)).unwrap();
    let before = b.snapshot(snapshot(SnapshotPhase::Before { pc: 0 }, 0, regs(sum, i)));
    let call_guard = b.guard(site(0), before);
    let length = f(
        &mut b,
        0,
        TraceOp::HostCall {
            function: HostFunction::StringLength,
            args: vec![text],
            result: TraceType::Float,
            guard: call_guard,
        },
        TraceType::Float,
    );
    let sum2 = f(
        &mut b,
        1,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: sum,
            right: length,
        },
        TraceType::Float,
    );
    let one = f(&mut b, 2, TraceOp::Constant(float(1.0)), TraceType::Float);
    let i2 = f(
        &mut b,
        2,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: i,
            right: one,
        },
        TraceType::Float,
    );
    let more = f(
        &mut b,
        3,
        TraceOp::Compare {
            comparison: Comparison::LessEqual,
            left: i2,
            right: n,
        },
        TraceType::Boolean,
    );
    let exit = b.snapshot(snapshot(after_pc(3, 4), 3, regs(sum2, i2)));
    let guard = b.guard(site(3), exit);
    b.emit(
        site(3),
        TraceOp::GuardBranch {
            condition: more,
            expected: true,
            guard,
        },
        None,
    );
    let poll = b.snapshot(snapshot(after_pc(3, 0), 3, regs(sum2, i2)));
    b.set_poll_snapshot(poll);
    let mut ir = b.finish(TraceTerminator::LoopBack {
        args: vec![sum2, i2],
    });
    ir.header_live = vec![0, 1, 2, 3];
    ir
}

fn run_lengths(
    trace: &CompiledTrace,
    registers: &[TraceScalar],
    host: &mut LengthHost,
    fuel: i64,
) -> Observed {
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| TraceInputSlot::from_scalar(registers[usize::from(*register)]))
        .collect();
    let mut area = vec![0x5a5a_5a5a_i64; trace.exit_area_len()];
    let mut control = TraceControl::new(fuel);
    let mut space = super::super::exit::TraceTableSpace::new();
    let outcome = trace.run_with_tables(&inputs, &[], &mut control, &mut area, &mut space, host);
    control.scratch = 0;
    Observed {
        outcome,
        control,
        area,
    }
}

/// A lane the run made reads its length from the table in line, a root lane
/// goes to the VM, and the answer's tag is guarded: the outcome, the control
/// block, the exit area and the number of calls of the host equal Cranelift's.
#[test]
fn the_length_of_a_lane_string_is_read_in_line_like_cranelift() {
    use crate::jit_ir::trace::ROOT_STRING_LANE_BASE;
    let lanes = [
        0,
        1,
        5,
        63,
        ROOT_STRING_LANE_BASE,
        ROOT_STRING_LANE_BASE + 3,
        ROOT_STRING_LANE_BASE + 40,
    ];
    let (mut in_line, mut in_vm, mut left) = (0, 0, 0);
    for tier in TIERS {
        for result in [TraceType::Float, TraceType::Integer] {
            let ir = string_length_trace(result);
            ir.verify().unwrap();
            let mir = compile(&ir, tier, true);
            let cranelift = compile(&ir, tier, false);
            assert!(mir.mir_address().is_some(), "compiled by MIR");
            assert!(cranelift.mir_address().is_none());
            assert_eq!(mir.exits(), cranelift.exits());
            for lane in lanes {
                for wrong_tag in [false, true] {
                    let state = [TraceScalar::String(lane), TraceScalar::Nil];
                    let host = |integer| LengthHost {
                        calls: 0,
                        wrong_tag,
                        integer,
                    };
                    let integer = result == TraceType::Integer;
                    let (mut a_host, mut b_host) = (host(integer), host(integer));
                    let a = run_lengths(&mir, &state, &mut a_host, 10);
                    let b = run_lengths(&cranelift, &state, &mut b_host, 10);
                    assert_eq!(a, b, "lane {lane} {result:?} at {tier:?}");
                    assert_eq!(a_host.calls, b_host.calls, "calls, lane {lane}");
                    match (a.outcome, lane >= ROOT_STRING_LANE_BASE) {
                        (TraceOutcome::Exit { index: 0 }, true) => left += 1,
                        (TraceOutcome::Exit { .. }, true) => in_vm += 1,
                        (_, false) => {
                            assert_eq!(a_host.calls, 0, "a lane of the table is read in line");
                            in_line += 1;
                        }
                        (outcome, _) => panic!("{outcome:?}"),
                    }
                }
            }
        }
    }
    assert!(
        in_line > 0 && in_vm > 0 && left > 0,
        "{in_line} in line, {in_vm} in the VM, {left} left by the guard"
    );
}

#[test]
fn the_length_of_a_lane_string_in_a_loop_matches_cranelift() {
    use crate::jit_ir::trace::ROOT_STRING_LANE_BASE;
    let ir = string_length_loop();
    ir.verify().unwrap();
    for tier in TIERS {
        let mir = compile(&ir, tier, true);
        let cranelift = compile(&ir, tier, false);
        assert!(mir.mir_address().is_some(), "compiled by MIR");
        assert!(cranelift.mir_address().is_none());
        for lane in [2, 17, ROOT_STRING_LANE_BASE + 1] {
            for (rounds, fuel) in [(1.0, 100), (5.0, 100), (50.0, 100), (50.0, 7)] {
                let state = [
                    TraceScalar::String(lane),
                    float(0.0),
                    float(0.0),
                    float(rounds),
                ];
                let mut a_host = LengthHost {
                    calls: 0,
                    wrong_tag: false,
                    integer: false,
                };
                let mut b_host = LengthHost {
                    calls: 0,
                    wrong_tag: false,
                    integer: false,
                };
                let a = run_lengths(&mir, &state, &mut a_host, fuel);
                let b = run_lengths(&cranelift, &state, &mut b_host, fuel);
                assert_eq!(
                    a, b,
                    "lane {lane}, {rounds} rounds, fuel {fuel} at {tier:?}"
                );
                assert_eq!(a_host.calls, b_host.calls, "host calls, lane {lane}");
            }
        }
    }
}

// ---- backend selection ---------------------------------------------------------------

/// A tree whose root's exit continues in a loop side (the loop runs in place),
/// and the same tree with a side of that loop that rejoins it.
fn loop_side_states() -> Vec<Vec<TraceScalar>> {
    [1.0, 2.0, 3.0, 4.0, 7.0, 15.0, 39.0, 41.0, 90.5, -3.0, 0.0]
        .iter()
        .map(|start| vec![float(10.0), float(20.0), float(*start), TraceScalar::Nil])
        .collect()
}

#[test]
fn a_loop_side_entered_by_an_exit_runs_through_mir_like_cranelift() {
    for fuel in [1_000_000, 500, 7, 1] {
        for (limit, cap) in [(100.0, 40.0), (10.0, 5.0), (200.0, 1_000.0)] {
            let ir = super::super::tree_tests::loop_side::tree(limit, cap);
            same_everywhere("loop side", &ir, &loop_side_states(), fuel);
        }
    }
}

#[test]
fn a_nested_side_that_rejoins_the_loop_side_runs_through_mir_like_cranelift() {
    for fuel in [1_000_000, 900, 40, 7, 2] {
        for (limit, cap, wrap) in [
            (100.0, 400.0, 500.0),
            (10.0, 90.0, 40.0),
            (64.0, 40.0, 30.0),
        ] {
            let ir = super::super::tree_tests::loop_side::linked_tree(limit, cap, wrap, false);
            same_everywhere(
                "loop side with a rejoining side",
                &ir,
                &loop_side_states(),
                fuel,
            );
        }
    }
}

#[test]
fn a_helper_call_with_two_results_and_a_guard_matches_cranelift() {
    // `r2, r3 = divmod(r0, r1)`: the helper misses for a zero divisor and for
    // the one quotient that overflows, and the call's guard leaves.
    let ir = super::super::helper_tests::divmod_trace();
    let states: Vec<Vec<TraceScalar>> = integer_pairs()
        .into_iter()
        .map(|state| state[..2].to_vec())
        .map(|mut state| {
            state.resize(4, TraceScalar::Nil);
            state
        })
        .collect();
    same_everywhere("divmod helper", &ir, &states, 10);
}

#[test]
fn the_peak_tier_and_unsupported_operations_use_cranelift() {
    let ir = op_trace(&[TraceType::Float; 2], TraceType::Float, |v| {
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: v[0],
            right: v[1],
        }
    });
    FORCE_BACKEND.with(|force| force.set(Some(true)));
    let peak = compile_trace(&ir, OptimizationTier::Peak).unwrap();
    FORCE_BACKEND.with(|force| force.set(None));
    assert!(peak.mir_address().is_none(), "the peak tier is Cranelift's");
    // Default: no MIR code.
    FORCE_BACKEND.with(|force| force.set(Some(false)));
    let default = compile_trace(&ir, OPTIMIZED).unwrap();
    FORCE_BACKEND.with(|force| force.set(None));
    assert!(default.mir_address().is_none());
    // An integer compared with a float calls a helper in Cranelift's code,
    // so the tree goes to Cranelift for that tree.
    let mixed = op_trace(
        &[TraceType::Integer, TraceType::Float],
        TraceType::Boolean,
        |v| TraceOp::Compare {
            comparison: Comparison::LessThan,
            left: v[0],
            right: v[1],
        },
    );
    FORCE_BACKEND.with(|force| force.set(Some(true)));
    LAST_FALLBACK.with(|last| last.set(None));
    let trace = compile_trace(&mixed, OPTIMIZED).unwrap();
    let reason = LAST_FALLBACK.with(Cell::get);
    FORCE_BACKEND.with(|force| force.set(None));
    assert!(trace.mir_address().is_none());
    assert_eq!(reason, Some("comparison of an integer with a float"));
    // The shared counters see it too (other threads add to them, so only
    // that the reason is there).
    assert!(
        super::counters()
            .reasons
            .contains_key("comparison of an integer with a float")
    );
    // And it still runs right.
    let observed = run(
        &trace,
        &[TraceScalar::Integer(2), float(10.0), TraceScalar::Nil],
        10,
    );
    assert!(matches!(observed.outcome, TraceOutcome::Exit { index: 0 }));
}

// ---- memory ----------------------------------------------------------------------------

#[test]
fn a_steady_state_compile_allocates_nothing_in_the_backend() {
    let ir = tree_with_a_leaving_side();
    let guards = guard_trace(TraceType::Float, Comparison::LessThan, true);
    FORCE_BACKEND.with(|force| force.set(Some(true)));
    // A versioned loop allocates in every compile: `RangePlan::of` builds the
    // plan's vectors and the backend's versioned lowering builds the checks,
    // shifts and stepped copies. This measures the unversioned lowering.
    let mut options = TraceCompileOptions::default();
    options.opt_flags = options
        .opt_flags
        .without(crate::compat::TracingOptFlags::RANGE_PROOFS);
    let mut kept = Vec::new();
    for ir in [&ir, &guards, &ir, &guards, &ir, &guards] {
        // The first compiles size the arenas.
        kept.push(compile_trace_with(ir, OPTIMIZED, options).unwrap());
    }
    for ir in [&ir, &guards, &ir, &guards] {
        let trace = compile_trace_with(ir, OPTIMIZED, options).unwrap();
        assert!(trace.mir_address().is_some());
        let allocations = LAST_COMPILE_ALLOCATIONS.with(Cell::get);
        assert_eq!(
            allocations, 0,
            "allocations inside the MIR backend for one tree"
        );
        kept.push(trace);
    }
    FORCE_BACKEND.with(|force| force.set(None));
}

#[test]
fn a_steady_state_compile_of_the_string_probe_allocates_nothing_in_the_backend() {
    // The lookup of a string key walks the probe sequence in nested skips, one
    // trace per type it expects (each shape of answer), at both tiers.
    let traces: Vec<TraceIr> = [
        TraceType::Integer,
        TraceType::Float,
        TraceType::Boolean,
        TraceType::Nil,
    ]
    .into_iter()
    .map(super::super::hash_tests::trace)
    .collect();
    crate::jit::debug_env::mir_probe_for_this_test();
    FORCE_BACKEND.with(|force| force.set(Some(true)));
    let mut options = TraceCompileOptions::default();
    options.opt_flags = options
        .opt_flags
        .without(crate::compat::TracingOptFlags::RANGE_PROOFS);
    let mut kept = Vec::new();
    for _ in 0..2 {
        for ir in &traces {
            for tier in TIERS {
                kept.push(compile_trace_with(ir, tier, options).unwrap());
            }
        }
    }
    for ir in &traces {
        for tier in TIERS {
            let trace = compile_trace_with(ir, tier, options).unwrap();
            assert!(trace.mir_address().is_some(), "{tier:?}");
            let allocations = LAST_COMPILE_ALLOCATIONS.with(Cell::get);
            assert_eq!(
                allocations, 0,
                "allocations inside the MIR backend for a probe, {tier:?}"
            );
            kept.push(trace);
        }
    }
    FORCE_BACKEND.with(|force| force.set(None));
}

#[test]
fn code_lives_in_the_one_region_of_the_process() {
    use oximond_mir::exec::CodeRegion;
    let region = CodeRegion::global().expect("the region maps");
    let ir = guard_trace(TraceType::Integer, Comparison::LessThan, true);
    let traces: Vec<CompiledTrace> = (0..20).map(|_| compile(&ir, OPTIMIZED, true)).collect();
    let mut addresses: Vec<usize> = traces
        .iter()
        .map(|trace| trace.mir_address().expect("MIR code"))
        .collect();
    assert!(addresses.iter().all(|address| region.contains(*address)));
    addresses.sort_unstable();
    addresses.dedup();
    assert_eq!(
        addresses.len(),
        traces.len(),
        "each tree has a block of its own"
    );
    // Blocks go back to the region when the code is dropped; the carving
    // itself is tested in `oximond-mir/tests/region.rs`, where the region is
    // private to the test.
}

#[test]
fn the_options_count_links_without_changing_the_exits() {
    // `count_links` only adds to the control block's link counter at a side.
    let ir = tree_with_a_leaving_side();
    let plain = compile(&ir, OPTIMIZED, true);
    assert!(plain.mir_address().is_some());
    FORCE_BACKEND.with(|force| force.set(Some(true)));
    let counting = super::super::compile::compile_trace_with(
        &ir,
        OPTIMIZED,
        TraceCompileOptions {
            count_links: true,
            ..TraceCompileOptions::default()
        },
    )
    .unwrap();
    let reference = super::super::compile::compile_trace_with(
        &ir,
        OPTIMIZED,
        TraceCompileOptions {
            count_links: true,
            ..TraceCompileOptions::default()
        },
    );
    FORCE_BACKEND.with(|force| force.set(None));
    drop(reference);
    assert!(counting.mir_address().is_some());
    assert_eq!(counting.exits(), plain.exits());
    let cranelift = {
        FORCE_BACKEND.with(|force| force.set(Some(false)));
        let trace = super::super::compile::compile_trace_with(
            &ir,
            OPTIMIZED,
            TraceCompileOptions {
                count_links: true,
                ..TraceCompileOptions::default()
            },
        )
        .unwrap();
        FORCE_BACKEND.with(|force| force.set(None));
        trace
    };
    for start in [1.0, 2.0, 5.0, 8.0] {
        let state = [float(5.0), float(6.0), float(start), TraceScalar::Nil];
        assert_eq!(
            run(&counting, &state, 1_000),
            run(&cranelift, &state, 1_000),
            "{start}"
        );
    }
}

// ---- rows: fields of tables met by lane, and the metatable guard ---------------------------

mod rows {
    use super::*;
    use crate::jit::trace::exit::{TraceBinding, TraceTableSpace, TraceTables};
    use crate::jit_ir::trace::meta_word;
    use crate::jit_ir::trace::{BindingKind, META_FIELD_KEY, MetaPin, TableBinding, TableField};
    use crate::runtime::Value;

    const METATABLE: u64 = 7;

    /// Slots of three fields per table, and the metatable word each row holds.
    #[derive(Clone)]
    struct World {
        tables: Vec<Vec<Value>>,
        meta: Vec<u64>,
        /// Rows whose field `n` is absent (the row word is zero).
        missing: Vec<(usize, usize)>,
    }

    struct Dispatcher<'a>(&'a mut World);

    impl TraceTables for Dispatcher<'_> {
        fn lane(&mut self, _: &Value) -> Option<usize> {
            None
        }

        fn row(&mut self, lane: usize, row: &mut [usize]) {
            let world = &mut *self.0;
            for field in 0..3 {
                row[field] = if world.missing.contains(&(lane, field)) {
                    0
                } else {
                    std::ptr::from_mut(&mut world.tables[lane][field]) as usize
                };
            }
            row[3] = world.meta[lane] as usize;
        }
    }

    #[allow(
        clippy::many_single_char_names,
        reason = "the letters are the row and key of the table the trace reads"
    )]
    fn trace() -> TraceIr {
        use RegisterEntry::{Const, Unchanged, Value as Val};
        let nil = Const(TraceScalar::Nil);
        let mut b = TraceBuilder::new(TraceId(0), frame(5));
        for register in [2, 3, 4] {
            b.mark_written(FrameId(0), register);
        }
        let table = b.input(0, TraceType::Table);
        let k = b.input(1, TraceType::Float);
        b.set_bindings(vec![TableBinding {
            register: 0,
            kind: BindingKind::MetaClass {
                metatable: METATABLE,
                pins: vec![MetaPin {
                    event: b"__index"[..].into(),
                    handler: 0,
                }]
                .into(),
                bare: 0,
            },
        }]);
        b.set_fields(vec![
            TableField {
                key: b"x"[..].into(),
                ty: TraceType::Float,
            },
            TableField {
                key: b"count"[..].into(),
                ty: TraceType::Integer,
            },
            TableField {
                key: b"flag"[..].into(),
                ty: TraceType::Boolean,
            },
            TableField {
                key: META_FIELD_KEY.into(),
                ty: TraceType::Integer,
            },
        ]);
        let mut effects = 0;
        let state = |b: &mut TraceBuilder, pc, extra: Vec<RegisterEntry>, effects: u32| {
            let mut registers = vec![Val(table), Val(k)];
            registers.extend(extra);
            registers.resize(5, nil);
            let snapshot = b.snapshot(snapshot_of(
                SnapshotPhase::Before { pc },
                effects,
                registers,
            ));
            b.guard(site(pc), snapshot)
        };
        let g = state(&mut b, 0, vec![], effects);
        b.emit(
            site(0),
            TraceOp::MetaGuard {
                table,
                binding: 0,
                guard: g,
            },
            None,
        );
        let g = state(&mut b, 1, vec![], effects);
        let x = b
            .emit(
                site(1),
                TraceOp::TableFieldLoad {
                    field: 0,
                    table,
                    guard: g,
                },
                Some(TraceType::Float),
            )
            .unwrap();
        let y = b
            .emit(
                site(2),
                TraceOp::Arithmetic {
                    op: ArithmeticOp::Add,
                    left: x,
                    right: k,
                },
                Some(TraceType::Float),
            )
            .unwrap();
        let g = state(&mut b, 3, vec![Val(y)], effects);
        let effect = b.next_effect();
        effects += 1;
        b.emit(
            site(3),
            TraceOp::TableFieldStore {
                field: 0,
                table,
                value: y,
                effect,
                guard: g,
            },
            None,
        );
        let g = state(&mut b, 4, vec![Val(y)], effects);
        let count = b
            .emit(
                site(4),
                TraceOp::TableFieldLoad {
                    field: 1,
                    table,
                    guard: g,
                },
                Some(TraceType::Integer),
            )
            .unwrap();
        let one = b
            .emit(
                site(5),
                TraceOp::Constant(TraceScalar::Integer(1)),
                Some(TraceType::Integer),
            )
            .unwrap();
        let count2 = b
            .emit(
                site(5),
                TraceOp::Arithmetic {
                    op: ArithmeticOp::Add,
                    left: count,
                    right: one,
                },
                Some(TraceType::Integer),
            )
            .unwrap();
        let g = state(&mut b, 6, vec![Val(y), Val(count2)], effects);
        let effect = b.next_effect();
        effects += 1;
        b.emit(
            site(6),
            TraceOp::TableFieldStore {
                field: 1,
                table,
                value: count2,
                effect,
                guard: g,
            },
            None,
        );
        let g = state(&mut b, 7, vec![Val(y), Val(count2)], effects);
        let flag = b
            .emit(
                site(7),
                TraceOp::TableFieldLoad {
                    field: 2,
                    table,
                    guard: g,
                },
                Some(TraceType::Boolean),
            )
            .unwrap();
        let flipped = b
            .emit(
                site(8),
                TraceOp::Not { source: flag },
                Some(TraceType::Boolean),
            )
            .unwrap();
        let g = state(&mut b, 9, vec![Val(y), Val(count2), Val(flag)], effects);
        let effect = b.next_effect();
        effects += 1;
        b.emit(
            site(9),
            TraceOp::TableFieldStore {
                field: 2,
                table,
                value: flipped,
                effect,
                guard: g,
            },
            None,
        );
        let _ = Unchanged;
        let exit = b.snapshot(snapshot_of(
            after(9),
            effects,
            vec![Val(table), Val(k), Val(y), Val(count2), Val(flipped)],
        ));
        b.finish(TraceTerminator::Exit { snapshot: exit })
    }

    fn snapshot_of(phase: SnapshotPhase, effects: u32, registers: Vec<RegisterEntry>) -> Snapshot {
        Snapshot::new(
            phase,
            effects,
            vec![SnapshotFrame {
                frame: FrameId(0),
                last_interpreted_pc: None,
                registers: registers.into(),
            }],
        )
    }

    fn observe(trace: &CompiledTrace, world: &World, lane: u64, k: f64) -> (Observed, Vec<String>) {
        let mut world = world.clone();
        let inputs = [
            TraceInputSlot::table(lane),
            TraceInputSlot::from_scalar(float(k)),
        ];
        let bindings = [TraceBinding::callee()];
        let mut area = vec![0x5a5a_i64; trace.exit_area_len()];
        let mut control = TraceControl::new(50);
        let mut space = TraceTableSpace::new();
        let outcome = trace.run_with_tables(
            &inputs,
            &bindings,
            &mut control,
            &mut area,
            &mut space,
            &mut Dispatcher(&mut world),
        );
        let memory = world
            .tables
            .iter()
            .map(|slots| format!("{slots:?}"))
            .collect();
        (
            Observed {
                outcome,
                control,
                area,
            },
            memory,
        )
    }

    #[test]
    fn field_loads_stores_and_the_metatable_guard_match_cranelift() {
        let ir = trace();
        ir.verify().unwrap();
        let identity = meta_word(METATABLE);
        let fresh = || World {
            tables: (0..3)
                .map(|n| {
                    vec![
                        Value::Number(f64::from(n) * 0.5),
                        Value::Integer(i64::from(n)),
                        Value::Boolean(n % 2 == 0),
                    ]
                })
                .collect(),
            meta: vec![identity, identity, identity ^ 1],
            missing: Vec::new(),
        };
        let mut worlds = vec![("plain", fresh(), vec![0, 1, 2])];
        let mut absent = fresh();
        absent.missing = vec![(1, 1)];
        worlds.push(("absent count", absent, vec![1]));
        let mut absent = fresh();
        absent.missing = vec![(0, 0), (2, 2)];
        worlds.push(("absent x and flag", absent, vec![0, 2]));
        for tier in TIERS {
            let mir = compile(&ir, tier, true);
            let cranelift = compile(&ir, tier, false);
            assert!(mir.mir_address().is_some(), "compiled by MIR");
            assert!(cranelift.mir_address().is_none());
            for (name, world, lanes) in &worlds {
                for &lane in lanes {
                    for k in [0.0, 1.5, -2.0, f64::NAN] {
                        let a = observe(&mir, world, lane, k);
                        let b = observe(&cranelift, world, lane, k);
                        assert_eq!(a, b, "{name} lane {lane} k {k} at {tier:?}");
                    }
                }
            }
        }
        // The metatable word of lane 2 differs, so its guard leaves.
        let mir = compile(&ir, OPTIMIZED, true);
        let (seen, _) = observe(&mir, &worlds[0].1, 2, 1.0);
        assert!(
            matches!(seen.outcome, TraceOutcome::Exit { index: 0 }),
            "{seen:?}"
        );
    }
}

/// A type guard on a typed value changes nothing (entry has checked the
/// type), under either backend. The recorder does not emit one from Lua
/// today, so the case is built by hand.
#[test]
fn a_type_guard_on_a_typed_value_matches_cranelift() {
    let mut b = TraceBuilder::new(TraceId(0), frame(2));
    b.mark_written(FrameId(0), 1);
    let x = b.input(0, TraceType::Float);
    let before = b.snapshot(snapshot(
        SnapshotPhase::Before { pc: 0 },
        0,
        vec![
            RegisterEntry::Value(x),
            RegisterEntry::Const(TraceScalar::Nil),
        ],
    ));
    let guard = b.guard(site(0), before);
    let guarded = b
        .emit(
            site(0),
            TraceOp::GuardType {
                source: x,
                expected: TraceType::Float,
                guard,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let exit = b.snapshot(snapshot(
        after(0),
        0,
        vec![RegisterEntry::Value(x), RegisterEntry::Value(guarded)],
    ));
    let ir = b.finish(TraceTerminator::Exit { snapshot: exit });
    ir.verify().unwrap();
    let states: Vec<Vec<TraceScalar>> = [0.0, -1.5, f64::NAN, 1e300]
        .iter()
        .map(|v| vec![float(*v), TraceScalar::Nil])
        .collect();
    same_everywhere("type guard", &ir, &states, 100);
}

// ---- memory: array slots, table borders and string bytes ---------------------------------

mod memory {
    use super::*;
    use crate::jit::trace::exit::TraceBinding;
    use crate::jit_ir::trace::{BindingKind, TableBinding};
    use crate::runtime::{NativeBytes, Value};
    use RegisterEntry::{Const, Value as Val};

    const NIL: RegisterEntry = Const(TraceScalar::Nil);

    fn state(effects: u32, registers: Vec<RegisterEntry>) -> impl FnOnce(PhaseAt) -> Snapshot {
        move |at| {
            let mut registers = registers;
            registers.resize(4, NIL);
            Snapshot::new(
                at.0,
                effects,
                vec![SnapshotFrame {
                    frame: FrameId(0),
                    last_interpreted_pc: None,
                    registers: registers.into(),
                }],
            )
        }
    }

    struct PhaseAt(SnapshotPhase);

    /// `r2 = t[r1] + 1`, `t[r1] = r2`, `r3 = #t`, over one array binding and
    /// one length binding. `key_ty` and `slot_ty` are the key's type and the
    /// type the load expects.
    fn array_trace(key_ty: TraceType, slot_ty: TraceType) -> TraceIr {
        let mut b = TraceBuilder::new(TraceId(0), frame(4));
        b.mark_written(FrameId(0), 2);
        b.mark_written(FrameId(0), 3);
        let table = b.input(0, TraceType::Table);
        let key = b.input(1, key_ty);
        b.set_bindings(vec![
            TableBinding {
                register: 0,
                kind: BindingKind::Array { first: None },
            },
            TableBinding {
                register: 0,
                kind: BindingKind::Length {
                    ty: TraceType::Integer,
                    raw: false,
                },
            },
        ]);
        let before = |b: &mut TraceBuilder, pc, effects, extra: Vec<RegisterEntry>| {
            let mut registers = vec![Val(table), Val(key)];
            registers.extend(extra);
            let snapshot = state(effects, registers)(PhaseAt(SnapshotPhase::Before { pc }));
            let snapshot = b.snapshot(snapshot);
            b.guard(site(pc), snapshot)
        };
        let guard = before(&mut b, 0, 0, vec![]);
        let loaded = b
            .emit(
                site(0),
                TraceOp::ArrayLoad {
                    binding: 0,
                    table,
                    key,
                    expected: slot_ty,
                    past_end: false,
                    guard,
                },
                Some(slot_ty),
            )
            .unwrap();
        let one = if slot_ty == TraceType::Float {
            TraceScalar::Float(1.0)
        } else {
            TraceScalar::Integer(1)
        };
        let one = b
            .emit(site(1), TraceOp::Constant(one), Some(slot_ty))
            .unwrap();
        let sum = b
            .emit(
                site(1),
                TraceOp::Arithmetic {
                    op: ArithmeticOp::Add,
                    left: loaded,
                    right: one,
                },
                Some(slot_ty),
            )
            .unwrap();
        let guard = before(&mut b, 2, 0, vec![Val(sum)]);
        let effect = b.next_effect();
        b.emit(
            site(2),
            TraceOp::ArrayStore {
                binding: 0,
                table,
                key,
                value: sum,
                effect,
                guard,
            },
            None,
        );
        let length = b
            .emit(
                site(3),
                TraceOp::TableLength { binding: 1, table },
                Some(TraceType::Integer),
            )
            .unwrap();
        let exit = b.snapshot(state(1, vec![Val(table), Val(key), Val(sum), Val(length)])(
            PhaseAt(after(3)),
        ));
        b.finish(TraceTerminator::Exit { snapshot: exit })
    }

    fn observe_array(
        trace: &CompiledTrace,
        slots: &[Value],
        key: TraceScalar,
    ) -> (Observed, String) {
        let mut slots = slots.to_vec();
        let inputs = [TraceInputSlot::table(0), TraceInputSlot::from_scalar(key)];
        let bindings = [
            TraceBinding::raw_array(slots.len(), slots.as_mut_ptr() as usize),
            TraceBinding::integer_length(slots.len() as i64),
        ];
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

    #[test]
    fn array_loads_stores_and_table_lengths_match_cranelift() {
        let float_slots = vec![
            Value::Number(0.5),
            Value::Number(-2.0),
            Value::Integer(7),
            Value::Nil,
            Value::Boolean(true),
            Value::Number(f64::NAN),
        ];
        let integer_slots = vec![
            Value::Integer(10),
            Value::Integer(-3),
            Value::Number(1.5),
            Value::Nil,
            Value::Integer(i64::MAX),
        ];
        let float_keys = [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            1.5,
            -1.0,
            f64::NAN,
            1e300,
        ];
        let integer_keys = [0, 1, 2, 3, 4, 5, 6, i64::MIN, i64::MAX, -1];
        let mut passes = 0;
        let mut exits = 0;
        for tier in TIERS {
            for (key_ty, slot_ty, slots) in [
                (TraceType::Float, TraceType::Float, &float_slots),
                (TraceType::Integer, TraceType::Integer, &integer_slots),
                (TraceType::Float, TraceType::Integer, &integer_slots),
            ] {
                let ir = array_trace(key_ty, slot_ty);
                ir.verify().unwrap();
                let mir = compile(&ir, tier, true);
                let cranelift = compile(&ir, tier, false);
                assert!(mir.mir_address().is_some(), "compiled by MIR");
                assert!(cranelift.mir_address().is_none());
                let keys: Vec<TraceScalar> = if key_ty == TraceType::Float {
                    float_keys.iter().map(|k| float(*k)).collect()
                } else {
                    integer_keys
                        .iter()
                        .map(|k| TraceScalar::Integer(*k))
                        .collect()
                };
                for key in keys {
                    let a = observe_array(&mir, slots, key);
                    let b = observe_array(&cranelift, slots, key);
                    assert_eq!(a, b, "{key:?} {key_ty:?}/{slot_ty:?} at {tier:?}");
                    match a.0.outcome {
                        TraceOutcome::Exit { index: 0 } => {}
                        TraceOutcome::Exit { .. } => exits += 1,
                        other => panic!("{other:?}"),
                    }
                    passes += 1;
                }
            }
        }
        // Some keys store and leave through the last exit, others fail a
        // bounds or tag guard on the load: both must have happened.
        assert!(
            exits > 0 && passes > exits,
            "{exits} exits in {passes} runs"
        );
    }

    /// `r2 = byte(r1, r0)` and `r3 = #r1`, with the key of type `number`.
    fn byte_trace(number: TraceType) -> TraceIr {
        let mut b = TraceBuilder::new(TraceId(0), frame(4));
        b.mark_written(FrameId(0), 2);
        b.mark_written(FrameId(0), 3);
        let key = b.input(0, number);
        let text = b.input(1, TraceType::String);
        b.set_bindings(vec![TableBinding {
            register: 1,
            kind: BindingKind::Bytes { ty: number },
        }]);
        let before = b.snapshot(state(0, vec![Val(key), RegisterEntry::Unchanged])(PhaseAt(
            SnapshotPhase::Before { pc: 0 },
        )));
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
        let exit = b.snapshot(state(
            0,
            vec![Val(key), RegisterEntry::Unchanged, Val(byte), Val(length)],
        )(PhaseAt(after(0))));
        b.finish(TraceTerminator::Exit { snapshot: exit })
    }

    #[test]
    fn string_bytes_and_lengths_match_cranelift() {
        let strings: [&[u8]; 4] = [b"", b"a", b"hello", &[0xff, 0x00, 0x80, 0x7f, b'!']];
        let mut exits = 0;
        let mut passes = 0;
        for tier in TIERS {
            for number in [TraceType::Float, TraceType::Integer] {
                let ir = byte_trace(number);
                ir.verify().unwrap();
                let mir = compile(&ir, tier, true);
                let cranelift = compile(&ir, tier, false);
                assert!(mir.mir_address().is_some(), "compiled by MIR");
                assert!(cranelift.mir_address().is_none());
                let keys: Vec<TraceScalar> = if number == TraceType::Float {
                    [0.0, 1.0, 2.0, 5.0, 6.0, -1.0, 1.5, f64::NAN, 1e300]
                        .iter()
                        .map(|k| float(*k))
                        .collect()
                } else {
                    [0, 1, 2, 5, 6, -1, i64::MIN, i64::MAX]
                        .iter()
                        .map(|k| TraceScalar::Integer(*k))
                        .collect()
                };
                for text in strings {
                    let Value::String(string) = Value::from(text) else {
                        unreachable!()
                    };
                    for key in &keys {
                        let observe = |trace: &CompiledTrace| {
                            let inputs = [
                                TraceInputSlot::from_scalar(*key),
                                TraceInputSlot::from_scalar(TraceScalar::String(0)),
                            ];
                            let bindings = [TraceBinding::bytes(NativeBytes::of(&string))];
                            let mut area = vec![0x5a5a_i64; trace.exit_area_len()];
                            let mut control = TraceControl::new(50);
                            let outcome =
                                trace.run_bound(&inputs, &bindings, &mut control, &mut area);
                            Observed {
                                outcome,
                                control,
                                area,
                            }
                        };
                        let a = observe(&mir);
                        let b = observe(&cranelift);
                        assert_eq!(a, b, "{key:?} in {text:?} at {tier:?}");
                        match a.outcome {
                            TraceOutcome::Exit { index: 0 } => exits += 1,
                            TraceOutcome::Exit { .. } => passes += 1,
                            other => panic!("{other:?}"),
                        }
                    }
                }
            }
        }
        assert!(
            exits > 0 && passes > 0,
            "{exits} guard exits, {passes} passes"
        );
    }
}

/// A strict header promises that a set moves no storage: Cranelift appends,
/// pops and overwrites in place and leaves through the guard otherwise, never
/// calling the VM. The backend does the same in place (see `tests/arrays.rs`
/// for the runs against Cranelift), so a tree with a strict header or a
/// header that is not strict stays on MIR, with no fallback.
#[test]
fn an_array_set_under_a_header_is_compiled_by_mir() {
    use crate::jit::trace::store_tests::set_case;
    use crate::jit_ir::trace::BindingKind;
    for strict in [true, false] {
        let ir = set_case(
            Some(BindingKind::Header { strict }),
            TraceScalar::Integer(3),
        );
        FORCE_BACKEND.with(|force| force.set(Some(true)));
        LAST_FALLBACK.with(|last| last.set(None));
        let trace = compile_trace(&ir, OPTIMIZED).unwrap();
        let reason = LAST_FALLBACK.with(Cell::get);
        FORCE_BACKEND.with(|force| force.set(None));
        assert!(trace.mir_address().is_some(), "strict {strict}");
        assert_eq!(reason, None, "strict {strict}");
    }
}

mod arrays;
mod arrays_tests;
pub(super) mod counted;
mod vmcall;

// ---- closure rows: prototype checks and cells read and written in line ---------------------

mod closures {
    use super::*;
    use crate::jit::trace::closure_cell_tests::{
        Closure, Dispatcher, closure, trace, trace_pinned_at,
    };
    use crate::jit::trace::exit::TraceTableSpace;
    use crate::jit_ir::trace::HelperId;
    use crate::runtime::Value;

    /// Runs `ir` over `closures` with the closure of lane 1 as the callee,
    /// and returns what the run left: the observation and the cells.
    fn observe(
        compiled: &CompiledTrace,
        closures: &[(usize, Vec<Value>)],
    ) -> (Observed, Vec<Vec<Value>>) {
        observe_callee(compiled, closures, 1)
    }

    fn observe_callee(
        compiled: &CompiledTrace,
        closures: &[(usize, Vec<Value>)],
        callee: u64,
    ) -> (Observed, Vec<Vec<Value>>) {
        let mut tables = Dispatcher {
            closures: closures
                .iter()
                .map(|(prototype, cells)| closure(*prototype, cells.clone()))
                .collect::<Vec<Closure>>(),
        };
        let mut space = TraceTableSpace::new();
        let inputs = [TraceInputSlot::closure(callee)];
        let mut area = vec![0x5a5a_5a5a_i64; compiled.exit_area_len()];
        let mut control = TraceControl::new(100);
        let outcome = compiled.run_with_tables(
            &inputs,
            &[],
            &mut control,
            &mut area,
            &mut space,
            &mut tables,
        );
        let cells = tables.closures.iter().map(|c| c.cells.clone()).collect();
        (
            Observed {
                outcome,
                control,
                area,
            },
            cells,
        )
    }

    fn same(name: &str, read: HelperId, worlds: &[Vec<(usize, Vec<Value>)>]) {
        same_pinned(name, &trace(read), 1, worlds);
    }

    /// Both backends over each world, with the closure of lane `callee` as
    /// the callee of `ir`.
    fn same_pinned(name: &str, ir: &TraceIr, callee: u64, worlds: &[Vec<(usize, Vec<Value>)>]) {
        ir.verify().unwrap();
        for tier in TIERS {
            let mir = compile(ir, tier, true);
            let cranelift = compile(ir, tier, false);
            assert!(
                mir.mir_address().is_some(),
                "{name}: MIR compiled it ({:?})",
                LAST_FALLBACK.with(Cell::get)
            );
            assert!(cranelift.mir_address().is_none(), "{name}");
            assert_eq!(mir.exits(), cranelift.exits(), "{name}");
            assert_eq!(mir.row_stride(), cranelift.row_stride(), "{name}");
            for world in worlds {
                assert_eq!(
                    observe_callee(&mir, world, callee),
                    observe_callee(&cranelift, world, callee),
                    "{name} at {tier:?} over {world:?}"
                );
            }
        }
    }

    #[test]
    fn a_float_cell_is_read_and_written_in_line() {
        same(
            "float cell",
            HelperId::CELL_GET_FLOAT,
            &[vec![
                (0xaa, vec![Value::Number(1.0)]),
                (0xaa, vec![Value::Number(10.0), Value::Number(7.0)]),
            ]],
        );
    }

    #[test]
    fn an_integer_cell_is_read_and_written_in_line() {
        same(
            "integer cell",
            HelperId::CELL_GET_INTEGER,
            &[vec![
                (0xaa, vec![Value::Integer(1)]),
                (0xaa, vec![Value::Integer(-5)]),
            ]],
        );
    }

    #[test]
    fn a_failed_prototype_check_a_cell_of_another_type_or_a_missing_cell_leave_alike() {
        same(
            "leaving",
            HelperId::CELL_GET_FLOAT,
            &[
                // Another prototype.
                vec![
                    (0xaa, vec![Value::Number(1.0)]),
                    (0xbb, vec![Value::Number(3.0)]),
                ],
                // A lane that is no Lua closure.
                vec![(0xaa, vec![Value::Number(1.0)]), (0, vec![])],
                // Neither lane is one: equal prototype words of zero are no
                // match.
                vec![(0, vec![]), (0, vec![])],
                // A cell of another type.
                vec![
                    (0xaa, vec![Value::Number(1.0)]),
                    (0xaa, vec![Value::Integer(3)]),
                ],
                // The closure has no such cell.
                vec![(0xaa, vec![Value::Number(1.0)]), (0xaa, vec![])],
            ],
        );
    }

    /// The pinned closure sits in lane 2 and the callee in lane 1, so the
    /// pinned row is read at a non-zero displacement of the rows pointer.
    #[test]
    fn the_prototype_check_reads_the_pinned_row_of_a_lane_that_is_not_the_first() {
        let cell = || vec![Value::Number(1.0)];
        let worlds = [
            // The callee runs the pinned prototype, lane 0 another one.
            vec![(0xbb, cell()), (0xaa, cell()), (0xaa, cell())],
            // The callee runs the prototype of lane 0, the pinned one another.
            vec![(0xaa, cell()), (0xaa, cell()), (0xbb, cell())],
            // Another prototype throughout.
            vec![(0xaa, cell()), (0xbb, cell()), (0xcc, cell())],
            // Zero prototypes at the callee and at the pinned lane.
            vec![(0xaa, cell()), (0, vec![]), (0, vec![])],
            // Zero at the callee only.
            vec![(0xaa, cell()), (0, vec![]), (0xaa, cell())],
        ];
        same_pinned(
            "pinned lane 2",
            &trace_pinned_at(HelperId::CELL_GET_FLOAT, 2),
            1,
            &worlds,
        );
        // The first world passes the check and the second does not.
        let ir = trace_pinned_at(HelperId::CELL_GET_FLOAT, 2);
        let reference = compile(&ir, OptimizationTier::Optimized, false);
        let (hit, _) = observe_callee(&reference, &worlds[0], 1);
        let (miss, _) = observe_callee(&reference, &worlds[1], 1);
        assert_ne!(hit.outcome, miss.outcome);
    }

    #[test]
    fn a_cell_written_by_one_run_is_read_by_the_next_run_in_line() {
        // The test dispatcher gives each closure its own cells, so sharing
        // between two closures is not built here (the mir_backend.rs snippet
        // with an inc and get pair covers it end to end). This test reruns the
        // trace over the cells the previous run left.
        let ir = trace(HelperId::CELL_GET_FLOAT);
        for tier in TIERS {
            let mir = compile(&ir, tier, true);
            let cranelift = compile(&ir, tier, false);
            let mut world = vec![
                (0xaa, vec![Value::Number(1.0)]),
                (0xaa, vec![Value::Number(10.0)]),
            ];
            for _ in 0..3 {
                let (a, cells_a) = observe(&mir, &world);
                let (b, cells_b) = observe(&cranelift, &world);
                assert_eq!(a, b, "{tier:?}");
                assert_eq!(cells_a, cells_b, "{tier:?}");
                for (entry, cells) in world.iter_mut().zip(cells_a) {
                    entry.1 = cells;
                }
            }
            assert_eq!(world[1].1, vec![Value::Number(13.0)]);
        }
    }

    // ---- a table held by a cell, named by lane in the closure's row ----

    /// Closures whose rows name the lane of the table each cell holds, over a
    /// bank of table values.
    struct Named {
        closures: Vec<Closure>,
        /// Per closure and cell: the lane plus one the row names (zero for none).
        cell_lanes: Vec<Vec<usize>>,
        bank: Vec<Value>,
    }

    impl crate::jit::trace::exit::TraceTables for Named {
        fn lane(&mut self, _: &Value) -> Option<usize> {
            None
        }

        fn row(&mut self, lane: usize, row: &mut [usize]) {
            use crate::jit::trace::exit::CLOSURE_ROW_CELLS;
            row.fill(0);
            let closure = &mut self.closures[lane];
            row[0] = closure.prototype;
            for (position, cell) in closure.cells.iter_mut().enumerate() {
                row[1 + position] = std::ptr::from_mut(cell) as usize;
                row[1 + CLOSURE_ROW_CELLS + position] = self.cell_lanes[lane][position];
            }
        }

        fn closure_lanes(&mut self, lanes: &mut Vec<usize>) {
            lanes.extend(0..self.closures.len());
        }

        fn bank_address(&mut self) -> usize {
            if self.bank.is_empty() {
                0
            } else {
                self.bank.as_ptr() as usize
            }
        }
    }

    fn snapshot_of(phase: SnapshotPhase, effects: u32, registers: Vec<RegisterEntry>) -> Snapshot {
        Snapshot::new(
            phase,
            effects,
            vec![SnapshotFrame {
                frame: FrameId(0),
                last_interpreted_pc: None,
                registers: registers.into(),
            }],
        )
    }

    fn table_cell_trace() -> TraceIr {
        use RegisterEntry::{Const, Unchanged, Value as Val};
        let mut b = TraceBuilder::new(TraceId(0), frame(2));
        b.mark_written(FrameId(0), 1);
        let callee = b.input(0, TraceType::Closure);
        let index = b
            .emit(
                site(0),
                TraceOp::Constant(TraceScalar::Integer(0)),
                Some(TraceType::Integer),
            )
            .unwrap();
        let before = b.snapshot(snapshot_of(
            SnapshotPhase::Before { pc: 0 },
            0,
            vec![Unchanged, Const(TraceScalar::Nil)],
        ));
        let guard = b.guard(site(0), before);
        let effect = Some(b.next_effect());
        let table = b
            .emit(
                site(0),
                TraceOp::CallHelper {
                    helper: HelperId::CELL_GET_TABLE,
                    args: vec![callee, index],
                    effect,
                    guard,
                },
                Some(TraceType::Table),
            )
            .unwrap();
        let exit = b.snapshot(snapshot_of(
            SnapshotPhase::After {
                pc: 0,
                successor: 1,
            },
            1,
            vec![Unchanged, Val(table)],
        ));
        b.finish(TraceTerminator::Exit { snapshot: exit })
    }

    fn observe_named(compiled: &CompiledTrace, world: &mut Named) -> Observed {
        let mut space = TraceTableSpace::new();
        let inputs = [TraceInputSlot::closure(0)];
        let mut area = vec![0x5a5a_5a5a_i64; compiled.exit_area_len()];
        let mut control = TraceControl::new(100);
        let outcome =
            compiled.run_with_tables(&inputs, &[], &mut control, &mut area, &mut space, world);
        Observed {
            outcome,
            control,
            area,
        }
    }

    /// A named world of [`Named`] the tests build on call.
    type World<'a> = (&'a str, Box<dyn Fn() -> Named>);

    #[test]
    fn a_table_cell_is_read_in_line_for_a_hit_a_changed_cell_no_lane_or_no_bank() {
        use crate::runtime::TableHandle;
        let table = |n: u64| Value::Table(TableHandle::from_native_identity(n));
        let world = |cell: Value, named: usize, bank: Vec<Value>, prototype: usize| Named {
            closures: vec![closure(prototype, vec![cell])],
            cell_lanes: vec![vec![named]],
            bank,
        };
        let worlds: Vec<World<'_>> = vec![
            (
                "hit",
                Box::new(move || world(table(5), 2, vec![Value::Nil, table(5)], 1)),
            ),
            (
                "hit at lane zero",
                Box::new(move || world(table(5), 1, vec![table(5)], 1)),
            ),
            (
                "changed",
                Box::new(move || world(table(6), 2, vec![Value::Nil, table(5)], 1)),
            ),
            (
                "no lane",
                Box::new(move || world(table(5), 0, vec![table(5)], 1)),
            ),
            (
                "empty bank",
                Box::new(move || world(table(5), 1, vec![], 1)),
            ),
            (
                "not a table",
                Box::new(move || world(Value::Integer(5), 1, vec![table(5)], 1)),
            ),
            (
                "no cell",
                Box::new(move || Named {
                    closures: vec![closure(1, vec![])],
                    cell_lanes: vec![vec![]],
                    bank: vec![table(5)],
                }),
            ),
        ];
        let ir = table_cell_trace();
        ir.verify().unwrap();
        for tier in TIERS {
            let mir = compile(&ir, tier, true);
            let cranelift = compile(&ir, tier, false);
            assert!(
                mir.mir_address().is_some(),
                "MIR compiled it ({:?})",
                LAST_FALLBACK.with(Cell::get)
            );
            assert!(cranelift.mir_address().is_none());
            assert_eq!(mir.exits(), cranelift.exits());
            assert_eq!(mir.row_stride(), cranelift.row_stride());
            assert_eq!(
                mir.row_stride(),
                crate::jit::trace::exit::CLOSURE_ROW_TABLE_WORDS
            );
            for (name, make) in &worlds {
                let (mut a, mut b) = (make(), make());
                assert_eq!(
                    observe_named(&mir, &mut a),
                    observe_named(&cranelift, &mut b),
                    "{name} at {tier:?}"
                );
            }
        }
    }
}

// ---- numbers: conversions and registers of either subtype -----------------------------------

#[path = "../number_tests.rs"]
mod number_tests;
