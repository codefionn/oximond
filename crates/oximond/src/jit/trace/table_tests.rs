//! Native table access against the reference evaluator.
//!
//! The VM resolves table storage before a run and hands native code the
//! addresses. These tests do the same with real memory (`Vec<Value>`), run the
//! compiled trace at both tiers, run the reference evaluator over a copy of
//! the same data, and require the same exit, the same restored registers and
//! the same memory afterwards, bit for bit.

#![allow(
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]

use super::*;
use crate::jit::OptimizationTier;
use crate::jit_ir::Comparison;
use crate::jit_ir::trace::reference::{Outcome, TraceHeap, evaluate_with_heap};
use crate::jit_ir::trace::{
    ArithmeticOp, BindingKind, FrameId, GuardId, LoopParamInit, RegisterEntry, Snapshot,
    SnapshotFrame, SnapshotPhase, TableBinding, TraceBuilder, TraceFrameInfo, TraceId, TraceIr,
    TraceOp, TraceScalar, TraceSite, TraceTerminator, TraceType,
};
use crate::runtime::Value;

const TIERS: [OptimizationTier; 3] = [
    OptimizationTier::Baseline,
    OptimizationTier::Optimized,
    OptimizationTier::Peak,
];

pub(super) fn site(pc: usize) -> TraceSite {
    TraceSite {
        frame: FrameId(0),
        pc,
    }
}

pub(super) fn root(registers: u16) -> TraceFrameInfo {
    TraceFrameInfo {
        register_count: registers,
        code_len: 32,
        inlined: None,
        activation: None,
    }
}

pub(super) fn snap(
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

pub(super) fn emit(
    b: &mut TraceBuilder,
    pc: usize,
    op: TraceOp,
    ty: TraceType,
) -> crate::jit_ir::trace::ValueId {
    b.emit(site(pc), op, Some(ty)).unwrap()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(bound).unwrap()).unwrap()
    }
}

/// What a table scalar (a slot of the model) is as a trace scalar, if native
/// code can hold it.
fn scalar_of(value: &Value) -> Option<TraceScalar> {
    match value {
        Value::Number(number) => Some(TraceScalar::Float(*number)),
        Value::Integer(integer) => Some(TraceScalar::Integer(*integer)),
        Value::Boolean(flag) => Some(TraceScalar::Boolean(*flag)),
        _ => None,
    }
}

fn payload(scalar: TraceScalar) -> Value {
    match scalar {
        TraceScalar::Float(number) => Value::Number(number),
        TraceScalar::Integer(integer) => Value::Integer(integer),
        TraceScalar::Boolean(flag) => Value::Boolean(flag),
        other => panic!("{other:?} has no slot payload"),
    }
}

/// A canonical form for comparing memory: NaN payloads and signed zero
/// included.
fn canonical(values: &[Value]) -> Vec<String> {
    values
        .iter()
        .map(|value| match value {
            Value::Number(number) => format!("n{:016x}", number.to_bits()),
            Value::Integer(integer) => format!("i{integer}"),
            Value::Boolean(flag) => format!("b{flag}"),
            Value::Nil => "nil".to_owned(),
            other => format!("other:{}", other.type_name()),
        })
        .collect()
}

/// The reference's view of the same tables the native code writes: its own
/// copy of each array and field slot.
#[derive(Clone)]
pub(super) struct Tables {
    pub(super) arrays: Vec<Vec<Value>>,
    pub(super) fields: Vec<Value>,
    pub(super) lengths: Vec<TraceScalar>,
    /// Slots before and after each array that belong to no array: native
    /// code that reads or writes one gives another result than the model.
    pub(super) padding: usize,
}

impl Tables {
    fn slot(&self, binding: u32, key: TraceScalar) -> Option<usize> {
        let index = match key {
            TraceScalar::Integer(index) => index,
            TraceScalar::Float(value) if value as i64 as f64 == value => value as i64,
            _ => return None,
        };
        let slot = usize::try_from(index.checked_sub(1)?).ok()?;
        (slot < self.arrays[binding as usize].len() - 2 * self.padding)
            .then_some(slot + self.padding)
    }
}

impl TraceHeap for Tables {
    fn entry_accepts(&mut self, _: &[TableBinding]) -> bool {
        true
    }
    fn array_get(&mut self, binding: u32, key: TraceScalar) -> Option<TraceScalar> {
        let slot = self.slot(binding, key)?;
        scalar_of(&self.arrays[binding as usize][slot])
    }
    fn array_next(&mut self, binding: u32, key: TraceScalar, start: bool) -> Option<i64> {
        assert_eq!(self.padding, 0, "steps run over arrays without padding");
        let from = if start {
            0
        } else {
            let slot = self.slot(binding, key)?;
            if matches!(self.arrays[binding as usize][slot], Value::Nil) {
                return None;
            }
            slot + 1
        };
        self.arrays[binding as usize][from..]
            .iter()
            .position(|value| !matches!(value, Value::Nil))
            .map(|offset| i64::try_from(from + offset + 1).unwrap())
    }
    fn array_set(&mut self, binding: u32, key: TraceScalar, value: TraceScalar) -> bool {
        let Some(slot) = self.slot(binding, key) else {
            return false;
        };
        let held = &mut self.arrays[binding as usize][slot];
        if scalar_of(held).map(TraceScalar::ty) != Some(value.ty()) {
            return false;
        }
        *held = payload(value);
        true
    }
    fn field_get(&mut self, binding: u32) -> TraceScalar {
        scalar_of(&self.fields[binding as usize]).expect("a bound field has a native type")
    }
    fn field_set(&mut self, binding: u32, value: TraceScalar) {
        self.fields[binding as usize] = payload(value);
    }
    fn length(&mut self, binding: u32) -> TraceScalar {
        self.lengths[binding as usize]
    }
}

/// The input words of every binding of `ir`, over `tables`' memory.
fn binding_words(ir: &TraceIr, tables: &mut Tables) -> Vec<TraceBinding> {
    ir.bindings
        .iter()
        .enumerate()
        .map(|(index, binding)| match &binding.kind {
            BindingKind::Array { .. } => TraceBinding::raw_array(
                tables.arrays[index].len() - 2 * tables.padding,
                tables.arrays[index][tables.padding..].as_mut_ptr() as usize,
            ),
            BindingKind::Field { .. } => {
                TraceBinding::raw_field(std::ptr::from_mut(&mut tables.fields[index]) as usize)
            }
            BindingKind::Callee { .. }
            | BindingKind::StringField { .. }
            | BindingKind::MetaClass { .. }
            | BindingKind::NoMetatable
            | BindingKind::Hash => TraceBinding::callee(),
            BindingKind::Bytes { .. } => panic!("these tests bind no string"),
            BindingKind::Header { .. } => panic!("these tests bind no header"),
            BindingKind::Length { .. } => match tables.lengths[index] {
                TraceScalar::Float(length) => TraceBinding::float_length(length),
                TraceScalar::Integer(length) => TraceBinding::integer_length(length),
                other => panic!("a length is a number: {other:?}"),
            },
        })
        .collect()
}

/// Runs `ir` natively and under the reference over copies of `tables`, and
/// compares everything observable. Returns whether the run left through a
/// guard.
fn agree(
    ir: &TraceIr,
    tier: OptimizationTier,
    registers: &[TraceScalar],
    tables: &Tables,
    label: &str,
) -> bool {
    let trace = compile_trace(ir, tier).unwrap();
    assert_eq!(trace.bindings(), &ir.bindings[..]);
    let mut native_tables = tables.clone();
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| TraceInputSlot::from_scalar(registers[usize::from(*register)]))
        .collect();
    let bindings = binding_words(ir, &mut native_tables);
    let mut area = vec![0_i64; trace.exit_area_len()];
    let mut control = TraceControl::new(100_000);
    let outcome = trace.run_bound(&inputs, &bindings, &mut control, &mut area);

    let mut model = tables.clone();
    let evaluation = evaluate_with_heap(ir, registers, &[], 100_000, None, &mut model, |_, _| None);
    let lowered = trace.lowered_ir();
    let TraceOutcome::Exit { index } = outcome else {
        panic!("{label}: the trace refused entry");
    };
    let descriptor = &trace.exits()[index];
    let snapshot = &lowered.snapshots[descriptor.snapshot.index()];
    let state: Vec<TraceScalar> = snapshot.frames[0]
        .registers
        .iter()
        .enumerate()
        .map(|(register, entry)| match entry {
            RegisterEntry::Value(value) => descriptor.scalar(&area, *value).unwrap_or_else(|| {
                // An exit leaves out a register's own entry input when the
                // frame keeps it: the restore leaves the register alone.
                assert!(
                    lowered.entry.inputs.iter().any(|input| {
                        usize::from(input.register) == register && input.value == *value
                    }),
                    "the exit spilled every snapshot value"
                );
                registers[register]
            }),
            RegisterEntry::Const(scalar) => *scalar,
            RegisterEntry::Unchanged => registers[register],
            RegisterEntry::Root(_) => unreachable!("no roots here"),
            RegisterEntry::Virtual(_) | RegisterEntry::Slice(_) => {
                unreachable!("no virtual tables here")
            }
        })
        .collect();
    let (expected, guard) = match evaluation.outcome {
        Outcome::GuardFailed { guard, restored } => (restored, Some(guard)),
        Outcome::Exited(restored) => (restored, None),
        other => panic!("{label}: unexpected reference outcome {other:?}"),
    };
    assert_eq!(
        descriptor.origin,
        guard.map(|guard| guard.0),
        "{label} {tier:?}: which guard left"
    );
    assert_eq!(snapshot.resume_point().pc, expected.pc, "{label} {tier:?}");
    assert_eq!(
        snapshot.resume_point().last_interpreted_pc,
        expected.last_interpreted_pc,
        "{label} {tier:?}"
    );
    assert_eq!(
        snapshot.effect_position, expected.effect_position,
        "{label} {tier:?}"
    );
    // Compare registers by bits: NaN payloads and signed zero must survive.
    let bits = |scalars: &[TraceScalar]| -> Vec<String> {
        scalars
            .iter()
            .map(|scalar| match scalar {
                TraceScalar::Float(number) => format!("f{:016x}", number.to_bits()),
                other => format!("{other:?}"),
            })
            .collect()
    };
    assert_eq!(bits(&state), bits(&expected.registers), "{label} {tier:?}");
    for (array, (native, reference)) in native_tables.arrays.iter().zip(&model.arrays).enumerate() {
        assert_eq!(
            canonical(native),
            canonical(reference),
            "{label} {tier:?}: array {array} after the run"
        );
    }
    assert_eq!(
        canonical(&native_tables.fields),
        canonical(&model.fields),
        "{label} {tier:?}: fields after the run"
    );
    guard.is_some()
}

const KEYS: [TraceScalar; 25] = [
    TraceScalar::Float(1.0),
    TraceScalar::Float(2.0),
    TraceScalar::Float(3.0),
    TraceScalar::Float(4.0),
    TraceScalar::Float(5.0),
    TraceScalar::Float(0.0),
    TraceScalar::Float(-0.0),
    TraceScalar::Float(-1.0),
    TraceScalar::Float(1.5),
    TraceScalar::Float(f64::NAN),
    TraceScalar::Float(f64::INFINITY),
    TraceScalar::Float(f64::NEG_INFINITY),
    TraceScalar::Float(9_223_372_036_854_775_808.0),
    TraceScalar::Float(1e308),
    TraceScalar::Float(4_503_599_627_370_496.0),
    TraceScalar::Float(4_503_599_627_370_497.0),
    TraceScalar::Float(0.999_999_999_999_999_9),
    TraceScalar::Float(1.000_000_000_000_000_2),
    TraceScalar::Float(2.999_999_999_999_999_6),
    TraceScalar::Float(5e-324),
    TraceScalar::Float(-1e308),
    TraceScalar::Integer(1),
    TraceScalar::Integer(3),
    TraceScalar::Integer(0),
    TraceScalar::Integer(i64::MIN),
];

/// Four slots of every native type, a nil, and another type.
fn slots() -> Vec<Value> {
    vec![
        Value::Number(1.5),
        Value::Integer(7),
        Value::Boolean(true),
        Value::Nil,
        Value::from("text"),
    ]
}

/// `r2 = t[r0]`, once, with the load's expected type `expected`.
fn load_trace(key: TraceType, expected: TraceType) -> TraceIr {
    use crate::jit_ir::trace::ValueId;
    let mut b = TraceBuilder::new(TraceId(0), root(4));
    b.mark_written(FrameId(0), 2);
    let k = b.input(0, key);
    let table = b.input(1, TraceType::Table);
    b.set_bindings(vec![TableBinding {
        register: 1,
        kind: BindingKind::Array { first: None },
    }]);
    let before = b.snapshot(snap(
        SnapshotPhase::Before { pc: 0 },
        0,
        None,
        vec![
            RegisterEntry::Value(k),
            RegisterEntry::Unchanged,
            RegisterEntry::Const(TraceScalar::Nil),
            RegisterEntry::Const(TraceScalar::Nil),
        ],
    ));
    let guard = b.guard(site(0), before);
    let loaded: ValueId = emit(
        &mut b,
        0,
        TraceOp::ArrayLoad {
            binding: 0,
            table,
            key: k,
            expected,
            past_end: false,
            guard,
        },
        expected,
    );
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 0,
            successor: 1,
        },
        0,
        Some(0),
        vec![
            RegisterEntry::Value(k),
            RegisterEntry::Unchanged,
            RegisterEntry::Value(loaded),
            RegisterEntry::Const(TraceScalar::Nil),
        ],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

/// `r2 = next index after r0` along the array part, once: the step of
/// [`TraceOp::ArrayNext`], from the start of the array when `start`, giving a
/// number of type `result`.
fn next_trace(key: TraceType, start: bool, result: TraceType) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), root(4));
    b.mark_written(FrameId(0), 2);
    let k = b.input(0, key);
    let table = b.input(1, TraceType::Table);
    b.set_bindings(vec![TableBinding {
        register: 1,
        kind: BindingKind::Array { first: None },
    }]);
    let before = b.snapshot(snap(
        SnapshotPhase::Before { pc: 0 },
        0,
        None,
        vec![
            RegisterEntry::Value(k),
            RegisterEntry::Unchanged,
            RegisterEntry::Const(TraceScalar::Nil),
            RegisterEntry::Const(TraceScalar::Nil),
        ],
    ));
    let guard = b.guard(site(0), before);
    let next = emit(
        &mut b,
        0,
        TraceOp::ArrayNext {
            binding: 0,
            table,
            key: k,
            start,
            ty: result,
            guard,
        },
        result,
    );
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 0,
            successor: 1,
        },
        0,
        Some(0),
        vec![
            RegisterEntry::Value(k),
            RegisterEntry::Unchanged,
            RegisterEntry::Value(next),
            RegisterEntry::Const(TraceScalar::Nil),
        ],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

#[test]
fn array_steps_match_the_reference_for_every_key_and_pattern_of_holes() {
    let nil = || Value::Nil;
    let some = |i: i64| Value::Integer(i);
    let patterns: Vec<Vec<Value>> = vec![
        vec![],
        vec![nil()],
        vec![some(1)],
        vec![some(1), some(2), some(3), some(4), some(5)],
        vec![nil(), nil(), some(3), nil(), some(5)],
        vec![
            some(1),
            nil(),
            nil(),
            nil(),
            nil(),
            nil(),
            nil(),
            nil(),
            some(9),
        ],
        vec![some(1), some(2), nil(), nil(), nil()],
        vec![nil(), nil(), nil(), nil(), nil(), nil()],
        vec![
            Value::Number(0.5),
            nil(),
            Value::Boolean(false),
            nil(),
            some(5),
        ],
    ];
    let mut exits = 0;
    let mut passes = 0;
    for tier in TIERS {
        for slots in &patterns {
            for result in [TraceType::Integer, TraceType::Float] {
                for key in KEYS {
                    let ir = next_trace(key.ty(), false, result);
                    let tables = tables_of(slots.clone());
                    let registers = [
                        key,
                        TraceScalar::Table(0),
                        TraceScalar::Nil,
                        TraceScalar::Nil,
                    ];
                    let failed = agree(
                        &ir,
                        tier,
                        &registers,
                        &tables,
                        &format!("{key:?} in {slots:?} as {result:?}"),
                    );
                    exits += usize::from(failed);
                    passes += usize::from(!failed);
                }
                // The start of a traversal ignores the key.
                let ir = next_trace(TraceType::Integer, true, result);
                let tables = tables_of(slots.clone());
                let registers = [
                    TraceScalar::Integer(0),
                    TraceScalar::Table(0),
                    TraceScalar::Nil,
                    TraceScalar::Nil,
                ];
                let failed = agree(
                    &ir,
                    tier,
                    &registers,
                    &tables,
                    &format!("start of {slots:?}"),
                );
                exits += usize::from(failed);
                passes += usize::from(!failed);
            }
        }
    }
    assert!(exits > 100 && passes > 20, "{exits} exits, {passes} passes");
}

/// `t[r0] = r2`, once, where r2 has type `value`.
fn store_trace(key: TraceType, value: TraceType) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), root(4));
    let k = b.input(0, key);
    let table = b.input(1, TraceType::Table);
    let stored = b.input(2, value);
    b.set_bindings(vec![TableBinding {
        register: 1,
        kind: BindingKind::Array { first: None },
    }]);
    let registers = vec![
        RegisterEntry::Value(k),
        RegisterEntry::Unchanged,
        RegisterEntry::Value(stored),
        RegisterEntry::Const(TraceScalar::Nil),
    ];
    let before = b.snapshot(snap(
        SnapshotPhase::Before { pc: 0 },
        b.effect_position(),
        None,
        registers.clone(),
    ));
    let guard = b.guard(site(0), before);
    let effect = b.next_effect();
    b.emit(
        site(0),
        TraceOp::ArrayStore {
            binding: 0,
            table,
            key: k,
            value: stored,
            effect,
            guard,
        },
        None,
    );
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 0,
            successor: 1,
        },
        b.effect_position(),
        Some(0),
        registers,
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

fn tables_of(values: Vec<Value>) -> Tables {
    Tables {
        arrays: vec![values],
        fields: Vec::new(),
        lengths: Vec::new(),
        padding: 0,
    }
}

#[test]
fn array_loads_match_the_reference_for_every_key_and_slot_type() {
    let mut exits = 0;
    let mut passes = 0;
    for tier in TIERS {
        for expected in [TraceType::Float, TraceType::Integer, TraceType::Boolean] {
            for key in KEYS {
                let ir = load_trace(key.ty(), expected);
                let tables = tables_of(slots());
                let registers = [
                    key,
                    TraceScalar::Table(0),
                    TraceScalar::Nil,
                    TraceScalar::Nil,
                ];
                let failed = agree(
                    &ir,
                    tier,
                    &registers,
                    &tables,
                    &format!("{key:?} as {expected:?}"),
                );
                exits += usize::from(failed);
                passes += usize::from(!failed);
            }
        }
    }
    // Both outcomes must have happened many times.
    assert!(exits > 80 && passes > 5, "{exits} exits, {passes} passes");
}

#[test]
fn array_stores_match_the_reference_for_every_key_slot_and_value_type() {
    let values = [
        TraceScalar::Float(-0.0),
        TraceScalar::Float(f64::NAN),
        TraceScalar::Integer(i64::MIN),
        TraceScalar::Boolean(false),
    ];
    let mut committed = 0;
    let mut refused = 0;
    for tier in TIERS {
        for value in values {
            for key in KEYS {
                let ir = store_trace(key.ty(), value.ty());
                let tables = tables_of(slots());
                let registers = [key, TraceScalar::Table(0), value, TraceScalar::Nil];
                let failed = agree(
                    &ir,
                    tier,
                    &registers,
                    &tables,
                    &format!("{key:?} = {value:?}"),
                );
                committed += usize::from(!failed);
                refused += usize::from(failed);
            }
        }
    }
    assert!(committed >= 8 && refused > 100, "{committed} and {refused}");
}

/// The worked case over real memory:
///
/// ```lua
/// while true do
///   local x = t[i]         -- pc 1, guard before it
///   t[i] = x * 2 + 1       -- pc 2, guard before it, one effect
///   if x > 0 then          -- pc 3/4, guard after the store
///     sum = sum + x
///   end
///   i = i + 1              -- pc 6
/// end
/// ```
///
/// Registers: r0 i, r1 sum, r2 x, r3 condition, r4 the table.
fn loop_trace(key_type: TraceType) -> TraceIr {
    use RegisterEntry::{Unchanged, Value};
    let float = key_type == TraceType::Float;
    let constant = |v: f64| {
        if float {
            TraceScalar::Float(v)
        } else {
            TraceScalar::Integer(v as i64)
        }
    };
    let mut b = TraceBuilder::new(TraceId(1), root(5));
    for register in 0..4 {
        b.mark_written(FrameId(0), register);
    }
    let i_in = b.input(0, key_type);
    let sum_in = b.input(1, TraceType::Float);
    let x_in = b.input(2, TraceType::Float);
    let cond_in = b.input(3, TraceType::Boolean);
    let table = b.input(4, TraceType::Table);
    b.set_bindings(vec![TableBinding {
        register: 4,
        kind: BindingKind::Array { first: None },
    }]);
    let i = b.loop_param(key_type, LoopParamInit::Input(i_in));
    let sum = b.loop_param(TraceType::Float, LoopParamInit::Input(sum_in));
    let x_prev = b.loop_param(TraceType::Float, LoopParamInit::Input(x_in));
    let cond_prev = b.loop_param(TraceType::Boolean, LoopParamInit::Input(cond_in));

    let s0 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 1 },
        0,
        Some(6),
        vec![
            Value(i),
            Value(sum),
            Value(x_prev),
            Value(cond_prev),
            Unchanged,
        ],
    ));
    let g0 = b.guard(site(1), s0);
    let x = emit(
        &mut b,
        1,
        TraceOp::ArrayLoad {
            binding: 0,
            table,
            key: i,
            expected: TraceType::Float,
            past_end: false,
            guard: g0,
        },
        TraceType::Float,
    );
    let two = emit(
        &mut b,
        2,
        TraceOp::Constant(TraceScalar::Float(2.0)),
        TraceType::Float,
    );
    let one = emit(
        &mut b,
        2,
        TraceOp::Constant(TraceScalar::Float(1.0)),
        TraceType::Float,
    );
    let doubled = emit(
        &mut b,
        2,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Multiply,
            left: x,
            right: two,
        },
        TraceType::Float,
    );
    let y = emit(
        &mut b,
        2,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: doubled,
            right: one,
        },
        TraceType::Float,
    );
    let s1 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 2 },
        b.effect_position(),
        Some(1),
        vec![Value(i), Value(sum), Value(x), Value(cond_prev), Unchanged],
    ));
    let g1 = b.guard(site(2), s1);
    let effect = b.next_effect();
    b.emit(
        site(2),
        TraceOp::ArrayStore {
            binding: 0,
            table,
            key: i,
            value: y,
            effect,
            guard: g1,
        },
        None,
    );
    let zero = emit(
        &mut b,
        3,
        TraceOp::Constant(TraceScalar::Float(0.0)),
        TraceType::Float,
    );
    let cond = emit(
        &mut b,
        3,
        TraceOp::Compare {
            comparison: Comparison::LessThan,
            left: zero,
            right: x,
        },
        TraceType::Boolean,
    );
    let s2 = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 4,
            successor: 9,
        },
        b.effect_position(),
        None,
        vec![Value(i), Value(sum), Value(x), Value(cond), Unchanged],
    ));
    let g2 = b.guard(site(4), s2);
    b.emit(
        site(4),
        TraceOp::GuardBranch {
            condition: cond,
            expected: true,
            guard: g2,
        },
        None,
    );
    let sum_next = emit(
        &mut b,
        5,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: sum,
            right: x,
        },
        TraceType::Float,
    );
    let step = emit(&mut b, 6, TraceOp::Constant(constant(1.0)), key_type);
    let i_next = emit(
        &mut b,
        6,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: i,
            right: step,
        },
        key_type,
    );
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 6,
            successor: 1,
        },
        b.effect_position(),
        Some(6),
        vec![
            Value(i_next),
            Value(sum_next),
            Value(x),
            Value(cond),
            Unchanged,
        ],
    ));
    b.set_poll_snapshot(poll);
    b.finish(TraceTerminator::LoopBack {
        args: vec![i_next, sum_next, x, cond],
    })
}

#[test]
fn a_loop_with_loads_stores_and_a_late_branch_exit_matches_the_reference() {
    let mut after_a_store = 0;
    let mut before_the_store = 0;
    let mut before_the_load = 0;
    for key_type in [TraceType::Float, TraceType::Integer] {
        let ir = loop_trace(key_type);
        ir.verify().unwrap();
        for seed in 0..300_u64 {
            let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
            let length = rng.below(9);
            let values: Vec<Value> = (0..length)
                .map(|_| match rng.below(12) {
                    0 => Value::Nil,
                    1 => Value::Integer(5),
                    2 => Value::Boolean(true),
                    3 => Value::from("s"),
                    4 => Value::Number(-0.0),
                    5 => Value::Number(f64::NAN),
                    6 => Value::Number(-3.5),
                    n => Value::Number(f64::from(u8::try_from(n).unwrap()) - 4.5),
                })
                .collect();
            let start = match rng.below(4) {
                0 => 0,
                1 => 2,
                _ => 1,
            };
            let first = if key_type == TraceType::Float {
                TraceScalar::Float(f64::from(start))
            } else {
                TraceScalar::Integer(i64::from(start))
            };
            let registers = [
                first,
                TraceScalar::Float(0.0),
                TraceScalar::Float(0.0),
                TraceScalar::Boolean(false),
                TraceScalar::Table(0),
            ];
            let tables = tables_of(values);
            for tier in TIERS {
                let trace = compile_trace(&ir, tier).unwrap();
                let label = format!("{key_type:?} seed {seed}");
                let failed = agree(&ir, tier, &registers, &tables, &label);
                assert!(failed, "{label}: the loop only ends in a guard");
                // Which kind of guard it was.
                let mut model = tables.clone();
                let evaluation =
                    evaluate_with_heap(&ir, &registers, &[], 100_000, None, &mut model, |_, _| {
                        None
                    });
                if let Outcome::GuardFailed { guard, .. } = evaluation.outcome {
                    match guard {
                        GuardId(0) => before_the_load += 1,
                        GuardId(1) => before_the_store += 1,
                        _ => after_a_store += 1,
                    }
                }
                drop(trace);
            }
        }
    }
    assert!(
        after_a_store > 100,
        "{after_a_store} {before_the_store} {before_the_load}"
    );
    // The load just proved the slot holds a float, so the store of a float
    // into it cannot fail on its own. Forced exits cover that guard.
    assert_eq!(before_the_store, 0);
    assert!(before_the_load > 100, "{before_the_load}");
}

#[test]
fn field_and_length_bindings_match_the_reference() {
    use RegisterEntry::{Unchanged, Value as Val};
    // `p.x = p.x + p.v * dt; n = #p` repeated until the guard on the
    // length fails.
    let float_length = |ty: TraceType| {
        let mut b = TraceBuilder::new(TraceId(2), root(4));
        b.mark_written(FrameId(0), 0);
        let i_in = b.input(0, TraceType::Float);
        let dt = b.input(1, TraceType::Float);
        let table = b.input(2, TraceType::Table);
        b.set_bindings(vec![
            TableBinding {
                register: 2,
                kind: BindingKind::Field {
                    key: b"x"[..].into(),
                    ty: TraceType::Float,
                    depth: 0,
                },
            },
            TableBinding {
                register: 2,
                kind: BindingKind::Field {
                    key: b"v"[..].into(),
                    ty: TraceType::Float,
                    depth: 0,
                },
            },
            TableBinding {
                register: 2,
                kind: BindingKind::Length { ty, raw: false },
            },
            TableBinding {
                register: 2,
                kind: BindingKind::Field {
                    key: b"flag"[..].into(),
                    ty: TraceType::Boolean,
                    depth: 0,
                },
            },
        ]);
        let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
        let x = emit(
            &mut b,
            1,
            TraceOp::FieldLoad { binding: 0, table },
            TraceType::Float,
        );
        let v = emit(
            &mut b,
            2,
            TraceOp::FieldLoad { binding: 1, table },
            TraceType::Float,
        );
        let step = emit(
            &mut b,
            3,
            TraceOp::Arithmetic {
                op: ArithmeticOp::Multiply,
                left: v,
                right: dt,
            },
            TraceType::Float,
        );
        let next = emit(
            &mut b,
            4,
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: x,
                right: step,
            },
            TraceType::Float,
        );
        let effect = b.next_effect();
        b.emit(
            site(5),
            TraceOp::FieldStore {
                binding: 0,
                table,
                value: next,
                effect,
            },
            None,
        );
        let flag = emit(
            &mut b,
            5,
            TraceOp::FieldLoad { binding: 3, table },
            TraceType::Boolean,
        );
        let not_flag = emit(&mut b, 5, TraceOp::Not { source: flag }, TraceType::Boolean);
        let effect = b.next_effect();
        b.emit(
            site(5),
            TraceOp::FieldStore {
                binding: 3,
                table,
                value: not_flag,
                effect,
            },
            None,
        );
        let length = emit(&mut b, 6, TraceOp::TableLength { binding: 2, table }, ty);
        let length = if ty == TraceType::Integer {
            emit(
                &mut b,
                6,
                TraceOp::IntToFloat { source: length },
                TraceType::Float,
            )
        } else {
            length
        };
        let one = emit(
            &mut b,
            7,
            TraceOp::Constant(TraceScalar::Float(1.0)),
            TraceType::Float,
        );
        let i_next = emit(
            &mut b,
            7,
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: i,
                right: one,
            },
            TraceType::Float,
        );
        let more = emit(
            &mut b,
            8,
            TraceOp::Compare {
                comparison: Comparison::LessEqual,
                left: i_next,
                right: length,
            },
            TraceType::Boolean,
        );
        let exit = b.snapshot(snap(
            SnapshotPhase::After {
                pc: 9,
                successor: 10,
            },
            b.effect_position(),
            Some(9),
            vec![Val(i_next), Unchanged, Unchanged, Unchanged],
        ));
        let guard = b.guard(site(9), exit);
        b.emit(
            site(9),
            TraceOp::GuardBranch {
                condition: more,
                expected: true,
                guard,
            },
            None,
        );
        let poll = b.snapshot(snap(
            SnapshotPhase::After {
                pc: 9,
                successor: 1,
            },
            b.effect_position(),
            Some(9),
            vec![Val(i_next), Unchanged, Unchanged, Unchanged],
        ));
        b.set_poll_snapshot(poll);
        b.finish(TraceTerminator::LoopBack { args: vec![i_next] })
    };
    for ty in [TraceType::Float, TraceType::Integer] {
        let ir = float_length(ty);
        ir.verify().unwrap();
        for length in [0_i64, 1, 4, 9] {
            let tables = Tables {
                padding: 0,
                arrays: vec![Vec::new(); 4],
                fields: vec![
                    Value::Number(0.25),
                    Value::Number(-1.5),
                    Value::Nil,
                    Value::Boolean(false),
                ],
                lengths: vec![
                    TraceScalar::Nil,
                    TraceScalar::Nil,
                    if ty == TraceType::Float {
                        TraceScalar::Float(length as f64)
                    } else {
                        TraceScalar::Integer(length)
                    },
                    TraceScalar::Nil,
                ],
            };
            let registers = [
                TraceScalar::Float(1.0),
                TraceScalar::Float(0.5),
                TraceScalar::Table(0),
                TraceScalar::Nil,
            ];
            for tier in TIERS {
                assert!(agree(
                    &ir,
                    tier,
                    &registers,
                    &tables,
                    &format!("{ty:?} len {length}")
                ));
            }
        }
    }
}

#[test]
fn a_forced_array_guard_takes_its_exit_stub_before_touching_memory() {
    let ir = loop_trace(TraceType::Float);
    let tables = tables_of(vec![Value::Number(3.0), Value::Number(4.0)]);
    let registers = [
        TraceScalar::Float(1.0),
        TraceScalar::Float(0.0),
        TraceScalar::Float(0.0),
        TraceScalar::Boolean(false),
        TraceScalar::Table(0),
    ];
    for tier in TIERS {
        for guard in [0, 1] {
            let trace = compile_trace_forcing_guard(&ir, tier, guard).unwrap();
            let mut memory = tables.clone();
            let inputs: Vec<TraceInputSlot> = trace
                .entry_contract()
                .iter()
                .map(|(register, _)| TraceInputSlot::from_scalar(registers[usize::from(*register)]))
                .collect();
            let bindings = binding_words(&ir, &mut memory);
            let mut area = vec![0_i64; trace.exit_area_len()];
            let mut control = TraceControl::new(100);
            let outcome = trace.run_bound(&inputs, &bindings, &mut control, &mut area);
            assert_eq!(outcome, TraceOutcome::Exit { index: guard }, "{tier:?}");
            assert_eq!(
                trace.exits()[guard].guard,
                Some(TraceGuardClass::Array),
                "{tier:?}"
            );
            // The forced guard leaves in the first iteration, before the
            // store it guards: the memory is as it was.
            assert_eq!(
                canonical(&memory.arrays[0]),
                canonical(&tables.arrays[0]),
                "{tier:?} guard {guard}"
            );
        }
    }
}

#[test]
#[should_panic(expected = "bindings")]
fn run_demands_a_slot_for_every_binding() {
    // Native code would read the binding words past the end of the slice,
    // and plain `run` has no way to supply them.
    let ir = loop_trace(TraceType::Float);
    let trace = compile_trace(&ir, OptimizationTier::Baseline).unwrap();
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|_| TraceInputSlot::nil())
        .collect();
    let mut area = vec![0_i64; trace.exit_area_len()];
    trace.run(&inputs, &mut TraceControl::new(1), &mut area);
}

#[test]
#[should_panic(expected = "bindings")]
fn run_bound_refuses_a_binding_of_the_wrong_kind() {
    // The loop trace binds an array. A length word there would be read as an
    // address, so the kind is checked before native code runs.
    let ir = loop_trace(TraceType::Float);
    let trace = compile_trace(&ir, OptimizationTier::Baseline).unwrap();
    assert!(trace.bindings().iter().any(TableBinding::is_array));
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|_| TraceInputSlot::nil())
        .collect();
    let bindings = vec![TraceBinding::integer_length(1); trace.bindings().len()];
    let mut area = vec![0_i64; trace.exit_area_len()];
    trace.run_bound(&inputs, &bindings, &mut TraceControl::new(1), &mut area);
}

/// The loops of the range-proof tests: upward and downward, with constant
/// and register steps, shifted keys and stores.
fn range_specs(float: bool) -> Vec<crate::jit_ir::trace::range::tests::LoopSpec> {
    use crate::jit_ir::trace::range::tests::LoopSpec;
    let upward = LoopSpec::upward(float);
    let downward = LoopSpec {
        step: Some(-1.0),
        next_on_left: false,
        ..upward.clone()
    };
    vec![
        upward.clone(),
        LoopSpec {
            step: Some(2.0),
            reads: vec![-1, 0, 1],
            write: Some(1),
            write_same_table: true,
            ..upward.clone()
        },
        LoopSpec {
            step: None,
            comparison: Comparison::LessThan,
            ..upward.clone()
        },
        LoopSpec {
            step: None,
            subtract: true,
            reads: vec![0, 2],
            write: Some(-1),
            ..downward.clone()
        },
        LoopSpec {
            key_on_control: true,
            ..downward
        },
        // The store reads back what the load proved.
        LoopSpec {
            reads: vec![0],
            write: Some(0),
            write_same_table: true,
            ..upward.clone()
        },
        // Three offsets of one array, and a store to the other: each
        // offset is its own displacement from one row.
        LoopSpec {
            reads: vec![-1, 0, 1],
            write: Some(0),
            ..upward.clone()
        },
        // Keys shifted by the step register, which the loop never
        // changes: `a[i + r4]` and `a[i - r4]`.
        LoopSpec {
            shifted: Some(false),
            ..upward.clone()
        },
        LoopSpec {
            shifted: Some(true),
            reads: vec![],
            ..upward
        },
    ]
}

/// The registers and the tables of one run of a range-proof loop: `seed`
/// picks the arrays, the start, the limit and the step.
fn range_case(float: bool, seed: u64) -> ([TraceScalar; 7], Tables) {
    let number = |value: f64| {
        if float {
            TraceScalar::Float(value)
        } else {
            TraceScalar::Integer(value as i64)
        }
    };
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    // Two float slots on either side that no key may reach.
    let array = |rng: &mut Rng, length: usize| -> Vec<Value> {
        let mut slots = vec![Value::Number(-100.0), Value::Number(-200.0)];
        slots.extend((0..length).map(|index| match rng.below(40) {
            0 => Value::Nil,
            1 => Value::Integer(5),
            2 => Value::Boolean(true),
            _ => Value::Number(f64::from(u32::try_from(index).unwrap()) + 0.25),
        }));
        slots.extend([Value::Number(100.0), Value::Number(200.0)]);
        slots
    };
    let fixed = seed < 13;
    let length = if fixed { 8 } else { rng.below(12) };
    let a = array(&mut rng, length);
    let other = if !fixed && rng.below(4) == 0 {
        rng.below(12)
    } else {
        length
    };
    let b = array(&mut rng, other);
    let pick = |rng: &mut Rng, choices: &[f64]| choices[rng.below(choices.len())];
    let n = length as f64;
    let starts = [1.0, 1.0, 2.0, 3.0, n - 1.0, n, n + 1.0, 0.0, -1.0, 40.0];
    let start = if fixed { 1.0 } else { pick(&mut rng, &starts) };
    let visible = if !fixed && rng.below(6) == 0 {
        pick(&mut rng, &starts)
    } else {
        start
    };
    let mut limits = vec![
        n,
        n,
        n - 1.0,
        n - 2.0,
        n + 1.0,
        2.0 * n,
        1.0,
        -1.0,
        9_007_199_254_740_992.0,
        -9_007_199_254_740_992.0,
    ];
    if float {
        limits.extend([f64::NAN, f64::INFINITY, 1e300, n - 0.5]);
    } else {
        // `i64::MAX` and `i64::MIN` (the cast saturates), and the ends of
        // what the fast loop compares as a scaled integer.
        limits.extend([1e19, -1e19, 1_099_511_627_776.0, 1_099_511_627_777.0]);
    }
    let limit = if fixed {
        n - 3.0
    } else {
        pick(&mut rng, &limits)
    };
    let mut steps = vec![1.0, 1.0, 2.0, 3.0, -1.0, -2.0, 1_099_511_627_776.0];
    if float {
        steps.extend([0.5, -0.5, 1.5, -1.5, f64::NAN, f64::INFINITY]);
    } else {
        // Steps whose sum with the variable wraps, and the ends of the
        // steps a proof takes.
        steps.extend([
            1e19,
            -1e19,
            2_147_483_648.0,
            2_147_483_649.0,
            -2_147_483_649.0,
        ]);
    }
    // The first seeds take each step once, over a whole array.
    let step = if fixed {
        steps[usize::try_from(seed).unwrap() % steps.len()]
    } else {
        pick(&mut rng, &steps)
    };
    // A fractional start keeps the slow version.
    let (start, visible) = match rng.below(8) {
        0 if float && !fixed => (start + 0.5, visible + 0.5),
        1 if float && !fixed => (start + 0.5, visible),
        _ => (start, visible),
    };
    let registers = [
        number(start),
        number(visible),
        TraceScalar::Float(0.0),
        number(limit),
        number(step),
        TraceScalar::Table(0),
        TraceScalar::Table(1),
    ];
    let tables = Tables {
        arrays: vec![a, b],
        fields: Vec::new(),
        lengths: Vec::new(),
        padding: 2,
    };
    (registers, tables)
}

/// Counted loops whose keys a range plan proves, against the reference: the
/// fast version (no bounds checks, integer copies of float inductions) and
/// the slow one must both leave where the reference leaves, with the same
/// state and memory. The data covers starts and limits inside, at the edges
/// of and outside the arrays, fractional and NaN numbers, steps of either
/// sign, and arrays with holes and slots of other types.
#[test]
fn counted_loops_with_range_proofs_match_the_reference() {
    use crate::jit_ir::trace::RangePlan;
    use crate::jit_ir::trace::range::tests::counted_loop;
    let mut proven = 0;
    let mut runs = 0;
    for float in [false, true] {
        for spec in range_specs(float) {
            let ir = counted_loop(&spec);
            ir.verify().unwrap();
            assert!(RangePlan::of(&ir, false).is_some(), "{spec:?}");
            proven += 1;
            for seed in 0..80_u64 {
                let (registers, tables) = range_case(float, seed);
                for tier in TIERS {
                    let label = format!("{spec:?} seed {seed} {registers:?}");
                    agree(&ir, tier, &registers, &tables, &label);
                    runs += 1;
                }
            }
        }
    }
    assert_eq!(proven, 18);
    assert!(runs > 1500, "{runs}");
}
// ---- the range-proven loop version under the MIR backend ---------------------------------

/// Everything a run of `trace` over `tables` shows: the outcome, the control
/// block, the exit area and the memory of the arrays.
fn observe(
    trace: &CompiledTrace,
    ir: &TraceIr,
    registers: &[TraceScalar],
    tables: &Tables,
    fuel: i64,
) -> (TraceOutcome, TraceControl, Vec<i64>, Vec<Vec<String>>) {
    let mut memory = tables.clone();
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| TraceInputSlot::from_scalar(registers[usize::from(*register)]))
        .collect();
    let bindings = binding_words(ir, &mut memory);
    let mut area = vec![0x5a5a_5a5a_i64; trace.exit_area_len()];
    let mut control = TraceControl::new(fuel);
    // A counted tree pays out of it; one that is not leaves it alone.
    control.budget = 1 << 20;
    let outcome = trace.run_bound(&inputs, &bindings, &mut control, &mut area);
    let arrays = memory.arrays.iter().map(|array| canonical(array)).collect();
    (outcome, control, area, arrays)
}

/// Compiles `ir` with each backend and runs both over the same data: they
/// must leave through the same exit with the same state and memory. Returns
/// whether the MIR code runs its loop in the range-proven version, which
/// must be what Cranelift decides.
pub(super) fn mir_matches_cranelift(
    ir: &TraceIr,
    tier: OptimizationTier,
    cases: &[([TraceScalar; 7], Tables)],
    fuels: &[i64],
    label: &str,
) -> bool {
    mir_runs_like_cranelift(ir, tier, cases, fuels, label, true)
}

/// [`mir_matches_cranelift`]; with `same_version` false the two backends may
/// differ in whether they version the loop (the MIR backend proves shapes
/// Cranelift does not).
fn mir_runs_like_cranelift(
    ir: &TraceIr,
    tier: OptimizationTier,
    cases: &[([TraceScalar; 7], Tables)],
    fuels: &[i64],
    label: &str,
    same_version: bool,
) -> bool {
    use crate::jit::debug_env::FORCE_BACKEND;
    let compile_with = |mir: bool| {
        FORCE_BACKEND.with(|force| force.set(Some(mir)));
        let compiled = compile_trace(ir, tier);
        FORCE_BACKEND.with(|force| force.set(None));
        compiled.unwrap()
    };
    let (mir, cranelift) = (compile_with(true), compile_with(false));
    assert!(mir.mir_address().is_some(), "{label} {tier:?}: MIR code");
    assert!(cranelift.mir_address().is_none(), "{label} {tier:?}");
    if same_version {
        assert_eq!(
            mir.versioned_loop(),
            cranelift.versioned_loop(),
            "{label} {tier:?}: the same loop is versioned"
        );
    }
    assert_eq!(mir.exits(), cranelift.exits(), "{label} {tier:?}");
    for (registers, tables) in cases {
        for &fuel in fuels {
            assert_eq!(
                observe(&mir, ir, registers, tables, fuel),
                observe(&cranelift, ir, registers, tables, fuel),
                "{label} {tier:?} fuel {fuel} from {registers:?}"
            );
        }
    }
    mir.versioned_loop()
}

/// Every loop of the range-proof tests, integer and float, over the data of
/// the reference test: starts inside, at the edges and outside the arrays
/// (the proof holds, or fails at entry), limits that end the loop in the
/// middle (a guard leaves inside the versioned copy), negative and offset
/// keys, a fuel that runs out in the loop, and float keys (the integer
/// copies of the fast version).
#[test]
fn range_proven_loops_run_through_mir_like_cranelift() {
    use crate::jit_ir::trace::range::tests::counted_loop;
    let (mut versioned, mut plain) = (0, 0);
    for float in [false, true] {
        for spec in range_specs(float) {
            let ir = counted_loop(&spec);
            ir.verify().unwrap();
            let cases: Vec<_> = (0..40).map(|seed| range_case(float, seed)).collect();
            for tier in TIERS {
                if tier == OptimizationTier::Peak {
                    continue;
                }
                let label = format!("{spec:?}");
                if mir_matches_cranelift(&ir, tier, &cases, &[100_000, 5], &label) {
                    assert_eq!(tier, OptimizationTier::Optimized, "{label}");
                    versioned += 1;
                } else {
                    plain += 1;
                }
            }
        }
    }
    assert_eq!(
        versioned, 18,
        "every loop is versioned at the optimized tier"
    );
    assert_eq!(plain, 18, "and none at the baseline tier");
}

/// Loops no guard bounds (the bound moves, or the guard is an equality), as
/// `ipairs` and `while t[i] do` record: the MIR backend runs them in a fast
/// version that compares each key with the array's length and proves the
/// rest once, where Cranelift keeps every check. Over the data of the range
/// tests (starts inside, at the edges and outside the arrays, keys that run
/// off the end in the middle of a run, a fuel that runs out) both leave
/// through the same exit with the same state and memory.
#[test]
fn open_loops_run_through_mir_like_cranelift() {
    use crate::jit_ir::trace::range::tests::{LoopSpec, counted_loop};
    let mut versioned = 0;
    for float in [false, true] {
        let upward = LoopSpec::upward(float);
        let specs = [
            LoopSpec {
                bound_varies: true,
                write: None,
                ..upward.clone()
            },
            LoopSpec {
                bound_varies: true,
                reads: vec![-1, 0, 2],
                step: Some(2.0),
                ..upward.clone()
            },
            LoopSpec {
                comparison: Comparison::Equal,
                reads: vec![0, 1],
                write: None,
                ..upward.clone()
            },
            LoopSpec {
                comparison: Comparison::Equal,
                key_on_control: true,
                step: Some(3.0),
                ..upward.clone()
            },
            // Not open: the step is a register, or points down.
            LoopSpec {
                bound_varies: true,
                step: None,
                ..upward.clone()
            },
            LoopSpec {
                bound_varies: true,
                step: Some(-1.0),
                ..upward
            },
        ];
        let cases: Vec<_> = (0..40).map(|seed| range_case(float, seed)).collect();
        for (at, spec) in specs.iter().enumerate() {
            let ir = counted_loop(spec);
            ir.verify().unwrap();
            let label = format!("{spec:?}");
            assert!(!mir_runs_like_cranelift(
                &ir,
                OptimizationTier::Baseline,
                &cases,
                &[100_000, 5],
                &label,
                false
            ));
            let fast = mir_runs_like_cranelift(
                &ir,
                OptimizationTier::Optimized,
                &cases,
                &[100_000, 40, 7, 5, 2, 1],
                &label,
                false,
            );
            assert_eq!(fast, at < 4, "{label}");
            versioned += u32::from(fast);
        }
    }
    assert_eq!(versioned, 8);
}

/// A fast loop whose induction ends on one compare: the loop closes on the
/// nearest of its bound, the end of its arrays and its fuel. A bound past
/// the arrays no longer keeps the run out of the fast version: it stops at
/// the last key inside them and goes on in the guarded loop, which leaves
/// where the reference leaves. Every fuel from one iteration up leaves at the
/// poll with the state Cranelift's code has there. A loop whose bound moves
/// keeps its guard in the body. Integer inductions close the same way, on
/// bounds up to both ends of an `i64`.
#[test]
fn a_loop_closes_on_one_compare_and_goes_on_past_its_arrays() {
    use crate::jit::debug_env::FORCE_BACKEND;
    use crate::jit_ir::trace::range::tests::{LoopSpec, counted_loop};
    let folded = |ir: &TraceIr| {
        crate::jit::trace::mir::FOLDED_LOOPS.with(|count| count.set(0));
        FORCE_BACKEND.with(|force| force.set(Some(true)));
        let compiled = compile_trace(ir, OptimizationTier::Optimized);
        FORCE_BACKEND.with(|force| force.set(None));
        compiled.unwrap();
        crate::jit::trace::mir::FOLDED_LOOPS.with(std::cell::Cell::get)
    };
    let array = |length: usize| -> Vec<Value> {
        let mut slots = vec![Value::Number(-100.0), Value::Number(-200.0)];
        slots.extend((0..length).map(|index| Value::Number(index as f64 + 0.25)));
        slots.extend([Value::Number(100.0), Value::Number(200.0)]);
        slots
    };
    let fuels: Vec<i64> = (1..=12).chain([100_000]).collect();
    for float in [true, false] {
        let number = |value: f64| {
            if float {
                TraceScalar::Float(value)
            } else {
                TraceScalar::Integer(value as i64)
            }
        };
        let moving = LoopSpec {
            bound_varies: true,
            ..LoopSpec::upward(float)
        };
        assert_eq!(folded(&counted_loop(&moving)), 0, "{moving:?}");
        let mut limits = vec![4.0, 4.5, 8.0, 9.0, 20.0, 1e9, -3.0, 0.0];
        if !float {
            // Both ends of an `i64`, and the largest bound the fast loop
            // takes with the one after it.
            limits.extend([1e19, -1e19, 1_099_511_627_776.0, 1_099_511_627_777.0]);
        }
        let mut closed = 0;
        for spec in range_specs(float) {
            let ir = counted_loop(&spec);
            closed += folded(&ir);
            let upward = !spec.subtract && spec.step.is_none_or(|step| step > 0.0);
            let mut cases = Vec::new();
            // Arrays of 8 and of 5 slots, and bounds inside them, at their
            // end, past it, between two keys and far outside.
            for (first, second) in [(8, 8), (8, 5), (5, 8)] {
                for &limit in &limits {
                    for step in [1.0, 2.0, -1.0] {
                        let start = if upward { 2.0 } else { 6.0 };
                        let registers = [
                            number(start),
                            number(start),
                            TraceScalar::Float(0.0),
                            number(limit),
                            number(step),
                            TraceScalar::Table(0),
                            TraceScalar::Table(1),
                        ];
                        let tables = Tables {
                            arrays: vec![array(first), array(second)],
                            fields: Vec::new(),
                            lengths: Vec::new(),
                            padding: 2,
                        };
                        let label = format!("{spec:?} {registers:?} arrays {first} {second}");
                        agree(
                            &ir,
                            OptimizationTier::Optimized,
                            &registers,
                            &tables,
                            &label,
                        );
                        cases.push((registers, tables));
                    }
                }
            }
            let label = format!("{spec:?}");
            assert!(mir_matches_cranelift(
                &ir,
                OptimizationTier::Optimized,
                &cases,
                &fuels,
                &label
            ));
        }
        assert!(
            closed >= 6,
            "{closed} of the loops close on one compare (float: {float})"
        );
    }
}

/// The folded loop over every trip count from none to twelve: the bound lies
/// `trips` keys past the start, inside arrays of 16 slots, so the run ends at
/// its bound, and against arrays of 6 slots it ends at their end first. Each
/// runs on every fuel from 1 to 14: below the length of the array the limit
/// is also held to the copy the fuel reaches, from the length up the fuel
/// outlasts the run and the limit takes nothing of it. A start that is not
/// integral, or outside the arrays, takes the guarded loop. Integer loops run
/// the same counts, from starts near both ends of an `i64` too (the sum with
/// the step wraps there), and both kinds run on steps of zero and of the
/// wrong sign, which only the fuel ends.
#[test]
fn a_folded_loop_runs_every_trip_count_on_every_fuel() {
    use crate::jit_ir::trace::range::tests::counted_loop;
    let array = |length: usize| -> Vec<Value> {
        let mut slots = vec![Value::Number(-100.0), Value::Number(-200.0)];
        slots.extend((0..length).map(|index| Value::Number(index as f64 + 0.25)));
        slots.extend([Value::Number(100.0), Value::Number(200.0)]);
        slots
    };
    let fuels: Vec<i64> = (1..=14).chain([20, 21, 100_000]).collect();
    for float in [true, false] {
        let number = |value: f64| {
            if float {
                TraceScalar::Float(value)
            } else {
                TraceScalar::Integer(value as i64)
            }
        };
        let starts: &[f64] = if float {
            &[3.0, 3.5, -40.0, 1e12]
        } else {
            &[
                3.0,
                -40.0,
                1e12,
                9_223_372_036_854_775_000.0,
                -9_223_372_036_854_775_000.0,
            ]
        };
        for spec in range_specs(float) {
            let ir = counted_loop(&spec);
            let upward = !spec.subtract && spec.step.is_none_or(|step| step > 0.0);
            let mut cases = Vec::new();
            let mut stuck = Vec::new();
            for length in [16, 6] {
                let tables = Tables {
                    arrays: vec![array(length), array(length)],
                    fields: Vec::new(),
                    lengths: Vec::new(),
                    padding: 2,
                };
                for trips in 0..=12 {
                    for &start in starts {
                        let (first, limit) = if upward {
                            (start, start + f64::from(trips) - 1.0)
                        } else {
                            (start + 12.0, start + 13.0 - f64::from(trips))
                        };
                        let registers = [
                            number(first),
                            number(first),
                            TraceScalar::Float(0.0),
                            number(limit),
                            number(if upward { 1.0 } else { -1.0 }),
                            TraceScalar::Table(0),
                            TraceScalar::Table(1),
                        ];
                        let label = format!("{spec:?} {registers:?} arrays of {length}");
                        agree(
                            &ir,
                            OptimizationTier::Optimized,
                            &registers,
                            &tables,
                            &label,
                        );
                        cases.push((registers, tables.clone()));
                    }
                }
                // A step of zero, and one that moves away from the bound: a
                // loop with a register step never ends by itself.
                for step in [0.0, if upward { -1.0 } else { 1.0 }] {
                    let registers = [
                        number(3.0),
                        number(3.0),
                        TraceScalar::Float(0.0),
                        number(if upward { 5.0 } else { 1.0 }),
                        number(step),
                        TraceScalar::Table(0),
                        TraceScalar::Table(1),
                    ];
                    stuck.push((registers, tables.clone()));
                }
            }
            let label = format!("{spec:?}");
            assert!(mir_matches_cranelift(
                &ir,
                OptimizationTier::Optimized,
                &cases,
                &fuels,
                &label
            ));
            assert!(mir_matches_cranelift(
                &ir,
                OptimizationTier::Optimized,
                &stuck,
                &fuels[..14],
                &label
            ));
        }
    }
}

/// Loops that test first (`while i <= n do ... i = i + s end`): the MIR
/// backend proves the keys of the accesses after the guard and runs the
/// loop without their bounds checks, on the integer copy of the induction.
/// Cranelift keeps every check. Both must leave where the reference leaves,
/// over the data of the other range tests (starts, bounds and steps inside,
/// at the edges of and outside the arrays, of either sign, fractional and
/// NaN, near both ends of an `i64`), over every trip count from none to
/// twelve, and on every fuel.
#[test]
fn loops_that_test_first_match_the_reference_and_cranelift() {
    use crate::jit_ir::trace::range::tests::{LoopSpec, counted_loop};
    use crate::jit_ir::trace::range::{RangeOptions, RangePlan};
    let array = |length: usize| -> Vec<Value> {
        let mut slots = vec![Value::Number(-100.0), Value::Number(-200.0)];
        slots.extend((0..length).map(|index| Value::Number(index as f64 + 0.25)));
        slots.extend([Value::Number(100.0), Value::Number(200.0)]);
        slots
    };
    let fuels: Vec<i64> = (1..=14).chain([100_000]).collect();
    let mut versioned = 0;
    for float in [true, false] {
        let number = |value: f64| {
            if float {
                TraceScalar::Float(value)
            } else {
                TraceScalar::Integer(value as i64)
            }
        };
        for spec in range_specs(float) {
            let spec = LoopSpec {
                tests_first: true,
                ..spec
            };
            let ir = counted_loop(&spec);
            ir.verify().unwrap();
            assert_eq!(RangePlan::of(&ir, false), None, "{spec:?}");
            assert!(
                RangePlan::with(
                    &ir,
                    RangeOptions {
                        tests_first: true,
                        ..RangeOptions::default()
                    }
                )
                .is_some(),
                "{spec:?}"
            );
            let upward = !spec.subtract && spec.step.is_none_or(|step| step > 0.0);
            let mut cases: Vec<_> = (0..60).map(|seed| range_case(float, seed)).collect();
            for (registers, tables) in &cases {
                for tier in TIERS {
                    let label = format!("{spec:?} {registers:?}");
                    agree(&ir, tier, registers, tables, &label);
                }
            }
            for length in [16, 6] {
                let tables = Tables {
                    arrays: vec![array(length), array(length)],
                    fields: Vec::new(),
                    lengths: Vec::new(),
                    padding: 2,
                };
                for trips in 0..=12 {
                    for start in [3.0, -40.0, 1e12] {
                        let (first, limit) = if upward {
                            (start, start + f64::from(trips) - 1.0)
                        } else {
                            (start + 12.0, start + 13.0 - f64::from(trips))
                        };
                        let registers = [
                            number(first),
                            number(first),
                            TraceScalar::Float(0.0),
                            number(limit),
                            number(if upward { 1.0 } else { -1.0 }),
                            TraceScalar::Table(0),
                            TraceScalar::Table(1),
                        ];
                        let label = format!("{spec:?} {registers:?} arrays of {length}");
                        agree(
                            &ir,
                            OptimizationTier::Optimized,
                            &registers,
                            &tables,
                            &label,
                        );
                        cases.push((registers, tables.clone()));
                    }
                }
                // Steps of zero and of the wrong sign, which the fuel ends.
                for step in [0.0, if upward { -1.0 } else { 1.0 }] {
                    cases.push((
                        [
                            number(3.0),
                            number(3.0),
                            TraceScalar::Float(0.0),
                            number(if upward { 5.0 } else { 1.0 }),
                            number(step),
                            TraceScalar::Table(0),
                            TraceScalar::Table(1),
                        ],
                        tables.clone(),
                    ));
                }
            }
            let label = format!("{spec:?}");
            if mir_runs_like_cranelift(
                &ir,
                OptimizationTier::Optimized,
                &cases,
                &fuels,
                &label,
                false,
            ) {
                versioned += 1;
            }
        }
    }
    assert_eq!(versioned, 18, "the MIR backend versions every such loop");
}

/// The loop of `spec` with a side on the guard of its first read, which the
/// slot's type fails: `rejoin` adds 1000 to the sum, steps both variables
/// and goes back to the loop (so the fast version is entered again, through
/// the checks), otherwise it adds 1000 and leaves.
fn counted_loop_with_a_side(spec: &LoopSpecOf, rejoin: bool) -> TraceIr {
    use crate::jit_ir::trace::TraceSide;
    use crate::jit_ir::trace::range::tests::counted_loop;
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
    let mut ir = counted_loop(spec);
    ir.header_live = vec![0, 1, 2];
    let mut s = TraceBuilder::new(TraceId(8), root(7));
    s.set_bindings(ir.bindings.clone());
    for register in 0..3 {
        s.mark_written(FrameId(0), register);
    }
    let control = s.input(0, ty);
    let visible = s.input(1, ty);
    let sum = s.input(2, TraceType::Float);
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
    let registers = |control: crate::jit_ir::trace::ValueId,
                     visible: crate::jit_ir::trace::ValueId| {
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
    let terminator = if rejoin {
        let one = emit(&mut s, 3, TraceOp::Constant(number(1.0)), ty);
        let step = |s: &mut TraceBuilder, value: crate::jit_ir::trace::ValueId| {
            emit(
                s,
                3,
                TraceOp::Arithmetic {
                    op: ArithmeticOp::Add,
                    left: value,
                    right: one,
                },
                ty,
            )
        };
        let (control2, visible2) = (step(&mut s, control), step(&mut s, visible));
        let link = s.snapshot(snap(
            SnapshotPhase::After {
                pc: 3,
                successor: 1,
            },
            0,
            Some(3),
            registers(control2, visible2),
        ));
        TraceTerminator::Link { snapshot: link }
    } else {
        let out = s.snapshot(snap(
            SnapshotPhase::After {
                pc: 3,
                successor: 20,
            },
            0,
            Some(3),
            registers(control, visible),
        ));
        TraceTerminator::Exit { snapshot: out }
    };
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(s.finish(terminator)),
    });
    ir
}

type LoopSpecOf = crate::jit_ir::trace::range::tests::LoopSpec;

/// A loop version with a side on a guard of the loop: the side leaves, or
/// rejoins the loop, which enters a version again through the checks. The
/// arrays hold integers where the loop wants floats, so the side is taken
/// from inside the fast version and from the guarded one.
#[test]
fn range_proven_loops_with_sides_run_through_mir_like_cranelift() {
    let mut taken = 0;
    for float in [false, true] {
        let upward = LoopSpecOf::upward(float);
        for spec in [
            upward.clone(),
            LoopSpecOf {
                step: Some(2.0),
                reads: vec![-1, 0, 1],
                write: Some(1),
                write_same_table: true,
                ..upward.clone()
            },
            LoopSpecOf {
                step: Some(-1.0),
                next_on_left: false,
                key_on_control: true,
                ..upward
            },
        ] {
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
            for holes in [&[][..], &[3][..], &[0, 4, 7][..], &[8][..]] {
                for (start, limit) in [(1.0, 9.0), (2.0, 8.0), (9.0, 1.0), (1.0, 12.0), (0.0, 5.0)]
                {
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
            for rejoin in [false, true] {
                let ir = counted_loop_with_a_side(&spec, rejoin);
                ir.verify().unwrap();
                let label = format!("{spec:?} rejoin {rejoin}");
                for tier in [OptimizationTier::Baseline, OptimizationTier::Optimized] {
                    let versioned =
                        mir_matches_cranelift(&ir, tier, &cases, &[100_000, 11, 4], &label);
                    taken += usize::from(versioned);
                }
            }
        }
    }
    assert_eq!(taken, 12, "every tree is versioned at the optimized tier");
}

/// `r2 = t[r0]` as a number of either subtype (`NumberLoad`), with its
/// `NumberTag` in `r3`, once.
fn number_load_trace() -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), root(7));
    b.mark_written(FrameId(0), 2);
    b.mark_written(FrameId(0), 3);
    let k = b.input(0, TraceType::Integer);
    let table = b.input(1, TraceType::Table);
    b.set_bindings(vec![TableBinding {
        register: 1,
        kind: BindingKind::Array { first: None },
    }]);
    let before = b.snapshot(snap(
        SnapshotPhase::Before { pc: 0 },
        0,
        None,
        vec![
            RegisterEntry::Value(k),
            RegisterEntry::Unchanged,
            RegisterEntry::Const(TraceScalar::Nil),
            RegisterEntry::Const(TraceScalar::Nil),
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
        ],
    ));
    let guard = b.guard(site(0), before);
    let loaded = emit(
        &mut b,
        0,
        TraceOp::NumberLoad {
            binding: 0,
            table,
            key: k,
            guard,
        },
        TraceType::Float,
    );
    let tag = emit(
        &mut b,
        0,
        TraceOp::NumberTag { load: loaded },
        TraceType::Boolean,
    );
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 0,
            successor: 1,
        },
        0,
        Some(0),
        vec![
            RegisterEntry::Value(k),
            RegisterEntry::Unchanged,
            RegisterEntry::Value(loaded),
            RegisterEntry::Value(tag),
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
        ],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

/// A number of either subtype is read as the reference reads it, and MIR's
/// code (the float on the straight path, the integer's checks in a skip)
/// leaves where Cranelift's leaves: floats, integers at and past 2^53 either
/// way, other types and keys outside the array.
#[test]
fn number_loads_of_either_subtype_match_the_reference_and_cranelift() {
    let edge = 1_i64 << 53;
    let values = vec![
        Value::Number(1.5),
        Value::Integer(7),
        Value::Integer(edge),
        Value::Integer(edge + 1),
        Value::Integer(-edge),
        Value::Integer(-edge - 1),
        Value::Integer(i64::MAX),
        Value::Number(f64::NAN),
        Value::Number(-0.0),
        Value::Boolean(true),
        Value::Nil,
        Value::from("text"),
        Value::Integer(-3),
    ];
    let ir = number_load_trace();
    ir.verify().unwrap();
    let len = i64::try_from(values.len()).unwrap();
    let (mut passes, mut exits) = (0, 0);
    let mut cases = Vec::new();
    for key in -1..=len + 2 {
        let registers = [
            TraceScalar::Integer(key),
            TraceScalar::Table(0),
            TraceScalar::Nil,
            TraceScalar::Nil,
            TraceScalar::Nil,
            TraceScalar::Nil,
            TraceScalar::Nil,
        ];
        let tables = tables_of(values.clone());
        for tier in [OptimizationTier::Baseline, OptimizationTier::Optimized] {
            let failed = agree(&ir, tier, &registers, &tables, &format!("key {key}"));
            exits += usize::from(failed);
            passes += usize::from(!failed);
        }
        cases.push((registers, tables));
    }
    for tier in [OptimizationTier::Baseline, OptimizationTier::Optimized] {
        mir_matches_cranelift(&ir, tier, &cases, &[10], "number load");
    }
    assert!(
        passes >= 10 && exits >= 10,
        "{passes} passes, {exits} exits"
    );
}
