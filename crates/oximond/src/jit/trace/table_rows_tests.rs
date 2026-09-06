//! Tables a trace meets while it runs: loaded out of an array, then read and
//! written by field through a lane and its row of slot addresses.
//!
//! The tests play the VM. Tables are `Vec<Value>` slot lists the test owns,
//! an array holds `Value::Table` handles (identity = index of the table), and
//! the dispatcher gives out lanes in the order tables are met and fills rows
//! with the addresses of the slots. The compiled trace runs at both tiers,
//! the reference evaluator runs over a copy of the same data, and the exit,
//! the restored registers and the memory afterwards must agree.

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
    ArithmeticOp, BindingKind, FrameId, LoopParamInit, RegisterEntry, Snapshot, SnapshotFrame,
    SnapshotPhase, TableBinding, TableField, TraceBuilder, TraceFrameInfo, TraceId, TraceIr,
    TraceOp, TraceScalar, TraceSite, TraceTerminator, TraceType, ValueId,
};
use crate::runtime::{TableHandle, Value};

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
            last_interpreted_pc: None,
            registers: registers.into(),
        }],
    )
}

fn emit(b: &mut TraceBuilder, pc: usize, op: TraceOp, ty: TraceType) -> ValueId {
    b.emit(site(pc), op, Some(ty)).unwrap()
}

fn handle(identity: u64) -> Value {
    Value::Table(TableHandle::from_native_identity(identity))
}

/// The identity of the table a slot holds.
fn identity_of(value: &Value) -> Option<u64> {
    match value {
        Value::Table(table) => Some(table.native_identity()),
        _ => None,
    }
}

/// Table 0 is the array; table `n` is the one the array's `n`th slot names.
#[derive(Clone)]
struct World {
    array: Vec<Value>,
    /// One slot per field of the trace for each table, `Nil` when absent.
    tables: Vec<Vec<Value>>,
    /// The fields of the trace, for the type each row slot must hold.
    kinds: Vec<TraceType>,
}

fn holds(ty: TraceType, value: &Value) -> bool {
    matches!(
        (ty, value),
        (TraceType::Float, Value::Number(_))
            | (TraceType::Integer, Value::Integer(_))
            | (TraceType::Boolean, Value::Boolean(_))
    )
}

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

fn canonical(values: &[Value]) -> Vec<String> {
    values
        .iter()
        .map(|value| match value {
            Value::Number(number) => format!("n{:016x}", number.to_bits()),
            Value::Integer(integer) => format!("i{integer}"),
            Value::Boolean(flag) => format!("b{flag}"),
            Value::Nil => "nil".to_owned(),
            Value::Table(table) => format!("t{}", table.native_identity()),
            other => format!("other:{}", other.type_name()),
        })
        .collect()
}

impl World {
    fn slot(&self, key: TraceScalar) -> Option<usize> {
        let index = match key {
            TraceScalar::Integer(index) => index,
            TraceScalar::Float(value) if value as i64 as f64 == value => value as i64,
            _ => return None,
        };
        let slot = usize::try_from(index.checked_sub(1)?).ok()?;
        (slot < self.array.len()).then_some(slot)
    }
}

/// The reference's model: a lane is the table's own identity.
impl TraceHeap for World {
    fn entry_accepts(&mut self, _: &[TableBinding]) -> bool {
        true
    }
    fn array_get(&mut self, _: u32, key: TraceScalar) -> Option<TraceScalar> {
        let slot = self.slot(key)?;
        match &self.array[slot] {
            Value::Table(table) => Some(TraceScalar::Table(table.native_identity())),
            other => scalar_of(other),
        }
    }
    fn array_set(&mut self, _: u32, _: TraceScalar, _: TraceScalar) -> bool {
        unreachable!("these traces do not write arrays")
    }
    fn field_get(&mut self, _: u32) -> TraceScalar {
        unreachable!("these traces bind no fields")
    }
    fn field_set(&mut self, _: u32, _: TraceScalar) {
        unreachable!("these traces bind no fields")
    }
    fn length(&mut self, _: u32) -> TraceScalar {
        unreachable!("these traces bind no length")
    }
    fn table_field_get(&mut self, table: u64, field: u32) -> Option<TraceScalar> {
        let value = self.tables.get(usize::try_from(table).ok()?)?[field as usize].clone();
        holds(self.kinds[field as usize], &value)
            .then(|| scalar_of(&value))
            .flatten()
    }
    fn table_field_set(&mut self, table: u64, field: u32, value: TraceScalar) -> bool {
        let kind = self.kinds[field as usize];
        let Some(row) = self.tables.get_mut(usize::try_from(table).unwrap()) else {
            return false;
        };
        let held = &mut row[field as usize];
        if !holds(kind, held) || scalar_of(held).map(TraceScalar::ty) != Some(value.ty()) {
            return false;
        }
        *held = payload(value);
        true
    }
}

/// The VM's side of a run: lanes in the order tables are met, rows over the
/// slots of the world's memory.
struct Dispatcher<'a> {
    world: &'a mut World,
    /// The identity of the table of each lane.
    lanes: Vec<u64>,
    /// Lanes the run may hand out, to test a full bank.
    capacity: usize,
    rows_filled: usize,
}

impl TraceTables for Dispatcher<'_> {
    fn lane(&mut self, value: &Value) -> Option<usize> {
        let identity = identity_of(value)?;
        if let Some(lane) = self.lanes.iter().position(|held| *held == identity) {
            return Some(lane);
        }
        if self.lanes.len() >= self.capacity {
            return None;
        }
        self.lanes.push(identity);
        Some(self.lanes.len() - 1)
    }

    fn row(&mut self, lane: usize, row: &mut [usize]) {
        self.rows_filled += 1;
        let identity = usize::try_from(self.lanes[lane]).unwrap();
        // The row's last word is its presence mask, which the space seals.
        for (field, address) in row.iter_mut().enumerate().take(self.world.kinds.len()) {
            *address = match self.world.tables.get_mut(identity) {
                Some(slots) if holds(self.world.kinds[field], &slots[field]) => {
                    std::ptr::from_mut(&mut slots[field]) as usize
                }
                _ => 0,
            };
        }
    }
}

const X: usize = 0;
const COUNT: usize = 1;

fn fields() -> Vec<TableField> {
    vec![
        TableField {
            key: b"x"[..].into(),
            ty: TraceType::Float,
        },
        TableField {
            key: b"count"[..].into(),
            ty: TraceType::Integer,
        },
    ]
}

/// Registers: r0 index (float), r1 the array, r2 limit, r3 the table, r4 x,
/// r5 the new x, r6 the count.
///
/// ```lua
/// repeat
///   t = bodies[i]        -- pc 0
///   x = t.x              -- pc 1
///   y = x + 1            -- pc 2
///   t.x = y              -- pc 3
///   c = t.count          -- pc 4
///   t.count = c + 1      -- pc 5
///   i = i + 1; until i > n   -- pc 6, 7
/// ```
// The single letters follow the names in the listing above.
#[allow(clippy::many_single_char_names)]
fn loop_trace() -> TraceIr {
    use RegisterEntry::{Const, Unchanged, Value as Val};
    let nil = Const(TraceScalar::Nil);
    let mut b = TraceBuilder::new(
        TraceId(0),
        TraceFrameInfo {
            register_count: 7,
            code_len: 32,
            inlined: None,
            activation: None,
        },
    );
    for register in [0, 3, 4, 5, 6] {
        b.mark_written(FrameId(0), register);
    }
    let i_in = b.input(0, TraceType::Float);
    let bodies = b.input(1, TraceType::Table);
    let limit = b.input(2, TraceType::Float);
    b.set_bindings(vec![TableBinding {
        register: 1,
        kind: BindingKind::Array { first: None },
    }]);
    b.set_fields(fields());
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));

    let s0 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 0 },
        0,
        vec![Val(i), Unchanged, Unchanged, nil, nil, nil, nil],
    ));
    let g0 = b.guard(site(0), s0);
    let t = emit(
        &mut b,
        0,
        TraceOp::ArrayLoad {
            binding: 0,
            table: bodies,
            key: i,
            expected: TraceType::Table,
            past_end: false,
            guard: g0,
        },
        TraceType::Table,
    );
    let s1 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 1 },
        0,
        vec![Val(i), Unchanged, Unchanged, Val(t), nil, nil, nil],
    ));
    let g1 = b.guard(site(1), s1);
    let x = emit(
        &mut b,
        1,
        TraceOp::TableFieldLoad {
            field: X as u32,
            table: t,
            guard: g1,
        },
        TraceType::Float,
    );
    let one = emit(
        &mut b,
        2,
        TraceOp::Constant(TraceScalar::Float(1.0)),
        TraceType::Float,
    );
    let y = emit(
        &mut b,
        2,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: x,
            right: one,
        },
        TraceType::Float,
    );
    let s2 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 3 },
        0,
        vec![Val(i), Unchanged, Unchanged, Val(t), Val(x), Val(y), nil],
    ));
    let g2 = b.guard(site(3), s2);
    let e0 = b.next_effect();
    b.emit(
        site(3),
        TraceOp::TableFieldStore {
            field: X as u32,
            table: t,
            value: y,
            effect: e0,
            guard: g2,
        },
        None,
    );
    let s3 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 4 },
        1,
        vec![Val(i), Unchanged, Unchanged, Val(t), Val(x), Val(y), nil],
    ));
    let g3 = b.guard(site(4), s3);
    let count = emit(
        &mut b,
        4,
        TraceOp::TableFieldLoad {
            field: COUNT as u32,
            table: t,
            guard: g3,
        },
        TraceType::Integer,
    );
    let one_int = emit(
        &mut b,
        5,
        TraceOp::Constant(TraceScalar::Integer(1)),
        TraceType::Integer,
    );
    let next_count = emit(
        &mut b,
        5,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: count,
            right: one_int,
        },
        TraceType::Integer,
    );
    let s4 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 5 },
        1,
        vec![
            Val(i),
            Unchanged,
            Unchanged,
            Val(t),
            Val(x),
            Val(y),
            Val(count),
        ],
    ));
    let g4 = b.guard(site(5), s4);
    let e1 = b.next_effect();
    b.emit(
        site(5),
        TraceOp::TableFieldStore {
            field: COUNT as u32,
            table: t,
            value: next_count,
            effect: e1,
            guard: g4,
        },
        None,
    );
    let step = emit(
        &mut b,
        6,
        TraceOp::Constant(TraceScalar::Float(1.0)),
        TraceType::Float,
    );
    let i_next = emit(
        &mut b,
        6,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: i,
            right: step,
        },
        TraceType::Float,
    );
    let more = emit(
        &mut b,
        7,
        TraceOp::Compare {
            comparison: Comparison::LessEqual,
            left: i_next,
            right: limit,
        },
        TraceType::Boolean,
    );
    let registers_after = vec![
        Val(i_next),
        Unchanged,
        Unchanged,
        Val(t),
        Val(x),
        Val(y),
        Val(count),
    ];
    let s5 = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 7,
            successor: 8,
        },
        2,
        registers_after.clone(),
    ));
    let g5 = b.guard(site(7), s5);
    b.emit(
        site(7),
        TraceOp::GuardBranch {
            condition: more,
            expected: true,
            guard: g5,
        },
        None,
    );
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 7,
            successor: 0,
        },
        2,
        registers_after,
    ));
    b.set_poll_snapshot(poll);
    b.finish(TraceTerminator::LoopBack { args: vec![i_next] })
}

fn row(x: Value, count: Value) -> Vec<Value> {
    vec![x, count]
}

/// A world of `n` tables with every field, the array naming them in order.
fn world_of(n: usize) -> World {
    World {
        array: (1..=n as u64).map(handle).collect(),
        tables: std::iter::once(row(Value::Nil, Value::Nil))
            .chain((0..n).map(|k| row(Value::Number(k as f64 * 0.5), Value::Integer(k as i64))))
            .collect(),
        kinds: fields().iter().map(|field| field.ty).collect(),
    }
}

fn registers(start: f64, limit: f64) -> Vec<TraceScalar> {
    let nothing = TraceScalar::Nil;
    vec![
        TraceScalar::Float(start),
        TraceScalar::Table(0),
        TraceScalar::Float(limit),
        nothing,
        nothing,
        nothing,
        nothing,
    ]
}

/// What a run did, in the terms both sides share.
struct Observed {
    guard: Option<u32>,
    registers: Vec<String>,
    tables: Vec<Vec<String>>,
    effects: u32,
}

fn run_native(
    ir: &TraceIr,
    tier: OptimizationTier,
    registers: &[TraceScalar],
    world: &World,
    capacity: usize,
) -> Observed {
    let trace = compile_trace(ir, tier).unwrap();
    assert!(trace.needs_tables());
    assert_eq!(trace.fields(), &ir.fields[..]);
    let mut world = world.clone();
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| TraceInputSlot::from_scalar(registers[usize::from(*register)]))
        .collect();
    let bindings = [TraceBinding::raw_array(
        world.array.len(),
        world.array.as_mut_ptr() as usize,
    )];
    let mut area = vec![0_i64; trace.exit_area_len()];
    let mut control = TraceControl::new(100_000);
    let mut space = TraceTableSpace::new();
    let mut dispatcher = Dispatcher {
        world: &mut world,
        lanes: vec![0],
        capacity,
        rows_filled: 0,
    };
    let outcome = trace.run_with_tables(
        &inputs,
        &bindings,
        &mut control,
        &mut area,
        &mut space,
        &mut dispatcher,
    );
    let lanes = dispatcher.lanes.clone();
    let TraceOutcome::Exit { index } = outcome else {
        panic!("the trace refused entry");
    };
    let descriptor = &trace.exits()[index];
    let lowered = trace.lowered_ir();
    let snapshot = &lowered.snapshots[descriptor.snapshot.index()];
    let state: Vec<String> = snapshot.frames[0]
        .registers
        .iter()
        .enumerate()
        .map(|(register, entry)| {
            let scalar = match entry {
                RegisterEntry::Value(value) => descriptor.scalar(&area, *value).unwrap(),
                RegisterEntry::Const(scalar) => *scalar,
                RegisterEntry::Unchanged => registers[register],
                RegisterEntry::Root(_) => unreachable!("no roots here"),
                RegisterEntry::Virtual(_) | RegisterEntry::Slice(_) => {
                    unreachable!("no virtual tables here")
                }
            };
            match scalar {
                TraceScalar::Float(number) => format!("f{:016x}", number.to_bits()),
                // A lane names the table it was given for.
                TraceScalar::Table(lane) => format!("Table({})", lanes[lane as usize]),
                other => format!("{other:?}"),
            }
        })
        .collect();
    Observed {
        guard: descriptor.origin,
        registers: state,
        tables: world.tables.iter().map(|slots| canonical(slots)).collect(),
        effects: snapshot.effect_position,
    }
}

fn run_reference(ir: &TraceIr, registers: &[TraceScalar], world: &World) -> Observed {
    let mut model = world.clone();
    let evaluation = evaluate_with_heap(ir, registers, &[], 100_000, None, &mut model, |_, _| None);
    let (restored, guard) = match evaluation.outcome {
        Outcome::GuardFailed { guard, restored } => (restored, Some(guard.0)),
        Outcome::Exited(restored) => (restored, None),
        other => panic!("unexpected reference outcome {other:?}"),
    };
    Observed {
        guard,
        registers: restored
            .registers
            .iter()
            .map(|scalar| match scalar {
                TraceScalar::Float(number) => format!("f{:016x}", number.to_bits()),
                other => format!("{other:?}"),
            })
            .collect(),
        tables: model.tables.iter().map(|slots| canonical(slots)).collect(),
        effects: restored.effect_position,
    }
}

fn agree(world: &World, start: f64, limit: f64, label: &str) -> Option<u32> {
    agree_trace(&loop_trace(), world, start, limit, label)
}

fn agree_trace(ir: &TraceIr, world: &World, start: f64, limit: f64, label: &str) -> Option<u32> {
    ir.verify().unwrap();
    let registers = registers(start, limit);
    let reference = run_reference(ir, &registers, world);
    for tier in TIERS {
        let native = run_native(ir, tier, &registers, world, 256);
        assert_eq!(native.guard, reference.guard, "{label} {tier:?}: guard");
        assert_eq!(
            native.registers, reference.registers,
            "{label} {tier:?}: registers"
        );
        assert_eq!(native.tables, reference.tables, "{label} {tier:?}: tables");
        assert_eq!(
            native.effects, reference.effects,
            "{label} {tier:?}: effects"
        );
    }
    reference.guard
}

#[test]
fn a_loop_over_an_array_of_tables_updates_every_field_like_the_reference() {
    let world = world_of(5);
    // The loop ends when the index passes the limit, through the branch.
    assert_eq!(agree(&world, 1.0, 5.0, "full"), Some(5));
    assert_eq!(agree(&world, 2.0, 3.0, "middle"), Some(5));
    // Past the end of the array the load leaves before it.
    assert_eq!(agree(&world, 1.0, 9.0, "past the end"), Some(0));
    assert_eq!(agree(&world, 0.0, 3.0, "from zero"), Some(0));
}

#[test]
fn one_table_met_twice_has_one_lane_and_one_row() {
    let mut world = world_of(3);
    // Slots 1 and 3 name the same table: both updates land in it.
    world.array[2] = handle(1);
    assert_eq!(agree(&world, 1.0, 3.0, "aliased"), Some(5));
    let ir = loop_trace();
    let registers = registers(1.0, 3.0);
    let native = run_native(&ir, OptimizationTier::Optimized, &registers, &world, 256);
    // Table 1 was updated twice: 0.0 + 1 + 1.
    assert_eq!(
        native.tables[1],
        canonical(&[Value::Number(2.0), Value::Integer(2)])
    );
}

#[test]
fn a_table_without_the_field_or_with_another_type_leaves_before_the_access() {
    for (label, tweak, guard) in [
        (
            "missing x",
            Box::new(|world: &mut World| world.tables[2][X] = Value::Nil)
                as Box<dyn Fn(&mut World)>,
            1,
        ),
        (
            "x is an integer",
            Box::new(|world: &mut World| world.tables[2][X] = Value::Integer(4)),
            1,
        ),
        (
            "count is a string",
            Box::new(|world: &mut World| world.tables[2][COUNT] = Value::from("n")),
            3,
        ),
        (
            "an element is not a table",
            Box::new(|world: &mut World| world.array[1] = Value::Number(1.0)),
            0,
        ),
        (
            "an element is nil",
            Box::new(|world: &mut World| world.array[1] = Value::Nil),
            0,
        ),
    ] {
        let mut world = world_of(4);
        tweak(&mut world);
        assert_eq!(agree(&world, 1.0, 4.0, label), Some(guard), "{label}");
    }
}

#[test]
fn a_run_with_no_lane_to_give_leaves_at_the_load() {
    let world = world_of(8);
    let ir = loop_trace();
    let registers = registers(1.0, 8.0);
    // The array itself holds lane 0, so a capacity of 4 gives three tables
    // lanes and the fourth load leaves.
    for tier in TIERS {
        let native = run_native(&ir, tier, &registers, &world, 4);
        assert_eq!(native.guard, Some(0), "{tier:?}");
        for (k, table) in native.tables.iter().enumerate().skip(1) {
            let touched = k <= 3;
            let expected = canonical(&[
                Value::Number((k - 1) as f64 * 0.5 + f64::from(u8::from(touched))),
                Value::Integer((k - 1) as i64 + i64::from(touched)),
            ]);
            assert_eq!(*table, expected, "{tier:?}: table {k}");
        }
    }
}

#[test]
fn rows_are_filled_once_per_table_and_run() {
    let mut world = world_of(3);
    let ir = loop_trace();
    let registers = registers(1.0, 3.0);
    let trace = compile_trace(&ir, OptimizationTier::Optimized).unwrap();
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| TraceInputSlot::from_scalar(registers[usize::from(*register)]))
        .collect();
    let bindings = [TraceBinding::raw_array(
        world.array.len(),
        world.array.as_mut_ptr() as usize,
    )];
    let mut area = vec![0_i64; trace.exit_area_len()];
    let mut space = TraceTableSpace::new();
    for _ in 0..2 {
        let mut dispatcher = Dispatcher {
            world: &mut world,
            lanes: vec![0],
            capacity: 256,
            rows_filled: 0,
        };
        trace.run_with_tables(
            &inputs,
            &bindings,
            &mut TraceControl::new(100_000),
            &mut area,
            &mut space,
            &mut dispatcher,
        );
        // Three tables. The array's own lane (an entry input) is only bound:
        // no operation reads a field of it, so it gets no row.
        assert_eq!(dispatcher.rows_filled, 3);
    }
}

#[test]
#[should_panic(expected = "run_with_tables")]
fn a_trace_that_reads_tables_by_lane_refuses_to_run_without_them() {
    let ir = loop_trace();
    let trace = compile_trace(&ir, OptimizationTier::Optimized).unwrap();
    let mut world = world_of(2);
    let inputs = vec![TraceInputSlot::nil(); 3];
    let bindings = [TraceBinding::raw_array(
        world.array.len(),
        world.array.as_mut_ptr() as usize,
    )];
    let mut area = vec![0_i64; trace.exit_area_len()];
    trace.run_bound(&inputs, &bindings, &mut TraceControl::new(1), &mut area);
}

#[test]
fn an_entry_table_past_the_lane_space_is_an_entry_mismatch() {
    let mut world = world_of(2);
    let ir = loop_trace();
    let trace = compile_trace(&ir, OptimizationTier::Optimized).unwrap();
    let mut registers = registers(1.0, 2.0);
    registers[1] = TraceScalar::Table(MAX_TRACE_TABLE_LANES as u64);
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| TraceInputSlot::from_scalar(registers[usize::from(*register)]))
        .collect();
    let bindings = [TraceBinding::raw_array(
        world.array.len(),
        world.array.as_mut_ptr() as usize,
    )];
    let mut area = vec![0_i64; trace.exit_area_len()];
    let mut dispatcher = Dispatcher {
        world: &mut world,
        lanes: vec![0],
        capacity: 256,
        rows_filled: 0,
    };
    let outcome = trace.run_with_tables(
        &inputs,
        &bindings,
        &mut TraceControl::new(10),
        &mut area,
        &mut TraceTableSpace::new(),
        &mut dispatcher,
    );
    assert!(matches!(outcome, TraceOutcome::EntryMismatch { .. }));
}

#[test]
fn the_presence_mask_names_each_present_word() {
    // The last word of a row is the mask, bit `i` for word `i`.
    let mut row = vec![0usize; 6];
    seal_row(&mut row);
    assert_eq!(row[5], 0);
    row[0] = 8;
    row[3] = 16;
    row[4] = 0;
    seal_row(&mut row);
    assert_eq!(row[5], 0b1001);
    // A masked row of fields keeps one word for the mask, and so does a
    // closure's. A row without one is as long as its words.
    assert_eq!(row_stride_of(3, 0, true), 4);
    assert_eq!(row_stride_of(2, 5, true), 6);
    assert_eq!(row_stride_of(2, 5, false), 5);
    assert_eq!(row_stride_of(0, 0, true), 0);
}

// ---- groups of fields under one guard ----------------------------------------

/// A loop that loads `count` float fields of the table in each array slot and
/// adds them up, so the loads of one table sit in one stretch and share a
/// guard on the row's presence mask. Registers as in [`loop_trace`], with the
/// sum in r4.
#[allow(clippy::many_single_char_names)]
fn group_trace(count: usize) -> TraceIr {
    use RegisterEntry::{Const, Unchanged, Value as Val};
    let nil = Const(TraceScalar::Nil);
    let mut b = TraceBuilder::new(
        TraceId(0),
        TraceFrameInfo {
            register_count: 7,
            code_len: 1024,
            inlined: None,
            activation: None,
        },
    );
    for register in [0, 3, 4] {
        b.mark_written(FrameId(0), register);
    }
    let i_in = b.input(0, TraceType::Float);
    let bodies = b.input(1, TraceType::Table);
    let limit = b.input(2, TraceType::Float);
    b.set_bindings(vec![TableBinding {
        register: 1,
        kind: BindingKind::Array { first: None },
    }]);
    b.set_fields(group_fields(count));
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let s0 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 0 },
        0,
        vec![Val(i), Unchanged, Unchanged, nil, nil, nil, nil],
    ));
    let g0 = b.guard(site(0), s0);
    let t = emit(
        &mut b,
        0,
        TraceOp::ArrayLoad {
            binding: 0,
            table: bodies,
            key: i,
            expected: TraceType::Table,
            past_end: false,
            guard: g0,
        },
        TraceType::Table,
    );
    let mut sum: Option<ValueId> = None;
    for field in 0..count {
        let pc = 1 + field;
        let state = b.snapshot(snap(
            SnapshotPhase::Before { pc },
            0,
            vec![Val(i), Unchanged, Unchanged, Val(t), nil, nil, nil],
        ));
        let guard = b.guard(site(pc), state);
        let loaded = emit(
            &mut b,
            pc,
            TraceOp::TableFieldLoad {
                field: field as u32,
                table: t,
                guard,
            },
            TraceType::Float,
        );
        sum = Some(match sum {
            None => loaded,
            Some(so_far) => emit(
                &mut b,
                pc,
                TraceOp::Arithmetic {
                    op: ArithmeticOp::Add,
                    left: so_far,
                    right: loaded,
                },
                TraceType::Float,
            ),
        });
    }
    let sum = sum.unwrap();
    let step = emit(
        &mut b,
        600,
        TraceOp::Constant(TraceScalar::Float(1.0)),
        TraceType::Float,
    );
    let i_next = emit(
        &mut b,
        600,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: i,
            right: step,
        },
        TraceType::Float,
    );
    let more = emit(
        &mut b,
        601,
        TraceOp::Compare {
            comparison: Comparison::LessEqual,
            left: i_next,
            right: limit,
        },
        TraceType::Boolean,
    );
    let after = vec![
        Val(i_next),
        Unchanged,
        Unchanged,
        Val(t),
        Val(sum),
        nil,
        nil,
    ];
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 601,
            successor: 602,
        },
        0,
        after.clone(),
    ));
    let guard = b.guard(site(601), exit);
    b.emit(
        site(601),
        TraceOp::GuardBranch {
            condition: more,
            expected: true,
            guard,
        },
        None,
    );
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 601,
            successor: 0,
        },
        0,
        after,
    ));
    b.set_poll_snapshot(poll);
    b.finish(TraceTerminator::LoopBack { args: vec![i_next] })
}

fn group_fields(count: usize) -> Vec<TableField> {
    (0..count)
        .map(|field| TableField {
            key: format!("f{field}").into_bytes().into(),
            ty: TraceType::Float,
        })
        .collect()
}

fn group_world(tables: usize, count: usize) -> World {
    World {
        array: (1..=tables as u64).map(handle).collect(),
        tables: std::iter::once(vec![Value::Nil; count])
            .chain((0..tables).map(|k| {
                (0..count)
                    .map(|field| Value::Number((k * 100 + field) as f64))
                    .collect()
            }))
            .collect(),
        kinds: vec![TraceType::Float; count],
    }
}

/// The index of the guard of the first field load: guard 0 is the array's.
const FIRST_FIELD_GUARD: u32 = 1;

#[test]
fn a_group_of_present_fields_runs_like_the_reference() {
    for count in [2, 3, 12, 63, 64, 66] {
        let world = group_world(4, count);
        let ir = group_trace(count);
        let exit = agree_trace(&ir, &world, 1.0, 4.0, &format!("{count} fields"));
        // The loop leaves through its branch, the last guard.
        assert_eq!(exit, Some(count as u32 + 1), "{count} fields");
    }
}

#[test]
fn a_table_that_lacks_one_field_of_a_group_leaves_at_the_groups_first_access() {
    // The reference leaves at the load of the missing field. Native code
    // checks the group at its first access, so it leaves there, in the state
    // before the group, with nothing written.
    for (count, missing) in [
        (3, 1),
        (3, 2),
        (12, 11),
        (64, 63),
        // Fields past the mask's 64 bits take the and-of-compares fallback.
        (66, 64),
        (66, 65),
        (66, 3),
    ] {
        let mut world = group_world(4, count);
        world.tables[2][missing] = Value::Nil;
        let ir = group_trace(count);
        ir.verify().unwrap();
        let registers = registers(1.0, 4.0);
        let reference = run_reference(&ir, &registers, &world);
        assert_eq!(reference.guard, Some(1 + missing as u32));
        for tier in TIERS {
            let native = run_native(&ir, tier, &registers, &world, 256);
            assert_eq!(
                native.guard,
                Some(FIRST_FIELD_GUARD),
                "{count} fields, {missing} missing, {tier:?}"
            );
            // The same table, the same index: the state before the group.
            assert_eq!(native.registers[0], reference.registers[0]);
            assert_eq!(native.registers[3], reference.registers[3]);
            assert_eq!(native.tables, reference.tables);
        }
    }
}

#[test]
fn a_field_of_another_type_leaves_a_group_too() {
    let mut world = group_world(3, 5);
    world.tables[1][4] = Value::Integer(7);
    let ir = group_trace(5);
    let registers = registers(1.0, 3.0);
    for tier in TIERS {
        let native = run_native(&ir, tier, &registers, &world, 256);
        assert_eq!(native.guard, Some(FIRST_FIELD_GUARD), "{tier:?}");
    }
}

#[test]
#[allow(unsafe_code)]
fn a_claimed_row_comes_back_sealed() {
    let mut space = TraceTableSpace::new();
    space.begin(row_stride_of(3, 0, true), 4, true);
    let claimed = space.claim_row(2, |row| {
        row[0] = 8;
        row[2] = 16;
    });
    assert!(claimed);
    // A lane with a row is not claimed again, and one past the space never.
    assert!(!space.claim_row(2, |_| unreachable!("the row is filled")));
    assert!(!space.claim_row(4, |_| unreachable!("past the space")));
    // SAFETY: the space holds four rows of four words and `rows_address` is
    // the first.
    let mask = unsafe { *(space.rows_address() as *const usize).add(2 * 4 + 3) };
    assert_eq!(mask, 0b101);
    #[cfg(debug_assertions)]
    space.assert_sealed();
}

/// A table a lane array view reads: its words at the offsets
/// `NativeTableLayout` names, and the slots of its array part, now and after
/// a `test.resize`.
struct LaneTable {
    words: Vec<usize>,
    slots: Vec<Value>,
    spare: Vec<Value>,
}

impl LaneTable {
    fn new(slots: Vec<Value>, spare: Vec<Value>) -> Self {
        use crate::jit::trace::helpers::TEST_TABLE_WORDS;
        let layout = crate::runtime::NativeTableLayout::get().unwrap();
        let word = |offset: i32| usize::try_from(offset).unwrap() / size_of::<usize>();
        assert!(word(layout.array_pointer).max(word(layout.array_length)) < TEST_TABLE_WORDS - 2);
        let mut table = Self {
            words: vec![0; TEST_TABLE_WORDS],
            slots,
            spare,
        };
        table.words[word(layout.array_pointer)] = table.slots.as_mut_ptr() as usize;
        table.words[word(layout.array_length)] = table.slots.len();
        table.words[TEST_TABLE_WORDS - 2] = table.spare.as_mut_ptr() as usize;
        table.words[TEST_TABLE_WORDS - 1] = table.spare.len();
        table
    }
}

/// Gives lane 0 to the one table and fills its row with the table's address.
struct LaneDispatcher {
    address: usize,
}

impl TraceTables for LaneDispatcher {
    fn lane(&mut self, value: &Value) -> Option<usize> {
        (identity_of(value) == Some(0)).then_some(0)
    }

    fn row(&mut self, lane: usize, row: &mut [usize]) {
        assert_eq!(lane, 0);
        row[0] = self.address;
    }
}

/// Registers: r0 the table, r1 the key (an integer), r2 the table's address
/// for `test.resize`, r3 and r4 the two reads.
///
/// ```text
/// pc 0  r3 = t[k]           (LaneArrayLoad of a float)
/// pc 1  test.resize(addr)   (when `resize`)
/// pc 2  r4 = t[k]
/// ```
fn lane_trace(resize: bool) -> TraceIr {
    use RegisterEntry::{Const, Unchanged, Value as Val};
    let nil = Const(TraceScalar::Nil);
    let mut b = TraceBuilder::new(
        TraceId(0),
        TraceFrameInfo {
            register_count: 5,
            code_len: 8,
            inlined: None,
            activation: None,
        },
    );
    for register in [3, 4] {
        b.mark_written(FrameId(0), register);
    }
    let table = b.input(0, TraceType::Table);
    let key = b.input(1, TraceType::Integer);
    let address = b.input(2, TraceType::Integer);
    b.set_fields(vec![TableField {
        key: crate::jit_ir::trace::TABLE_ADDRESS_KEY.into(),
        ty: TraceType::Integer,
    }]);
    let s0 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 0 },
        0,
        vec![Unchanged, Unchanged, Unchanged, nil, nil],
    ));
    let g0 = b.guard(site(0), s0);
    let first = emit(
        &mut b,
        0,
        TraceOp::LaneArrayLoad {
            table,
            field: 0,
            key,
            expected: TraceType::Float,
            guard: g0,
        },
        TraceType::Float,
    );
    let mut effects = 0;
    if resize {
        let s1 = b.snapshot(snap(
            SnapshotPhase::Before { pc: 1 },
            0,
            vec![Unchanged, Unchanged, Unchanged, Val(first), nil],
        ));
        let g1 = b.guard(site(1), s1);
        let effect = Some(b.next_effect());
        emit(
            &mut b,
            1,
            TraceOp::CallHelper {
                helper: crate::jit_ir::trace::HelperId::TEST_RESIZE,
                args: vec![address],
                effect,
                guard: g1,
            },
            TraceType::Integer,
        );
        effects = 1;
    }
    let s2 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 2 },
        effects,
        vec![Unchanged, Unchanged, Unchanged, Val(first), nil],
    ));
    let g2 = b.guard(site(2), s2);
    let second = emit(
        &mut b,
        2,
        TraceOp::LaneArrayLoad {
            table,
            field: 0,
            key,
            expected: TraceType::Float,
            guard: g2,
        },
        TraceType::Float,
    );
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 2,
            successor: 3,
        },
        effects,
        vec![Unchanged, Unchanged, Unchanged, Val(first), Val(second)],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

/// Runs `lane_trace(resize)` on `table` with key `key`: the guard that left
/// (`None` for the exit after the reads) and the two reads, as far as the
/// exit names them.
fn run_lane(
    table: &mut LaneTable,
    key: i64,
    resize: bool,
    tier: OptimizationTier,
) -> (Option<u32>, Vec<Option<TraceScalar>>) {
    let ir = lane_trace(resize);
    ir.verify().unwrap();
    let trace = compile_trace(&ir, tier).unwrap();
    assert!(trace.needs_tables());
    assert!(trace.lane_array_loads() > 0);
    let address = table.words.as_mut_ptr() as usize;
    let registers = [
        TraceScalar::Table(0),
        TraceScalar::Integer(key),
        TraceScalar::Integer(i64::try_from(address).unwrap()),
    ];
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| TraceInputSlot::from_scalar(registers[usize::from(*register)]))
        .collect();
    let mut area = vec![0_i64; trace.exit_area_len()];
    let mut control = TraceControl::new(100);
    let mut space = TraceTableSpace::new();
    let mut dispatcher = LaneDispatcher { address };
    let outcome = trace.run_with_tables(
        &inputs,
        &[],
        &mut control,
        &mut area,
        &mut space,
        &mut dispatcher,
    );
    let TraceOutcome::Exit { index } = outcome else {
        panic!("the trace refused entry");
    };
    let descriptor = &trace.exits()[index];
    let snapshot = &trace.lowered_ir().snapshots[descriptor.snapshot.index()];
    let reads = snapshot.frames[0]
        .registers
        .range(3, usize::MAX)
        .map(|entry| match entry {
            RegisterEntry::Value(value) => descriptor.scalar(&area, *value),
            _ => None,
        })
        .collect();
    (descriptor.origin, reads)
}

fn floats(values: &[f64]) -> Vec<Value> {
    values.iter().map(|&value| Value::Number(value)).collect()
}

#[test]
fn a_lane_array_load_reads_the_slot_through_the_tables_address() {
    for tier in TIERS {
        let mut table = LaneTable::new(floats(&[1.5, 2.5, 3.5]), Vec::new());
        let (guard, reads) = run_lane(&mut table, 2, false, tier);
        assert_eq!(guard, None, "{tier:?}");
        assert_eq!(
            reads,
            [Some(TraceScalar::Float(2.5)), Some(TraceScalar::Float(2.5))],
            "{tier:?}"
        );
    }
}

#[test]
fn a_lane_array_load_misses_on_a_bad_tag_or_a_key_outside_the_array() {
    for tier in TIERS {
        for (label, slots, key) in [
            ("integer", vec![Value::Integer(4), Value::Number(1.0)], 1),
            ("string", vec![Value::from("s"), Value::Number(1.0)], 1),
            ("nil", vec![Value::Nil, Value::Number(1.0)], 1),
            ("past the end", floats(&[1.0, 2.0]), 3),
            ("zero", floats(&[1.0, 2.0]), 0),
        ] {
            let mut table = LaneTable::new(slots, Vec::new());
            let (guard, _) = run_lane(&mut table, key, false, tier);
            assert_eq!(guard, Some(0), "{tier:?} {label}");
        }
    }
}

#[test]
fn a_lane_array_load_sees_the_array_a_rewriting_helper_moved() {
    for tier in TIERS {
        let mut table = LaneTable::new(floats(&[1.0, 2.0]), floats(&[10.0, 20.0, 30.0]));
        let (guard, reads) = run_lane(&mut table, 2, true, tier);
        assert_eq!(guard, None, "{tier:?}");
        assert_eq!(
            reads,
            [
                Some(TraceScalar::Float(2.0)),
                Some(TraceScalar::Float(20.0))
            ],
            "{tier:?}"
        );
        // A key past the old array but inside the new one reads after the
        // move, and the first read left before it.
        let mut table = LaneTable::new(floats(&[1.0, 2.0]), floats(&[10.0, 20.0, 30.0]));
        let (guard, _) = run_lane(&mut table, 3, true, tier);
        assert_eq!(guard, Some(0), "{tier:?}");
    }
}

/// Registers: r0 the table, r1 the key, r2 the value a store writes, r3 and r4
/// the reads. A load reads `t[k]` twice (the second through the words the
/// first kept), a store writes `r2` and reads the slot back.
fn lane_access_trace(key_ty: TraceType, ty: TraceType, store: bool) -> TraceIr {
    use RegisterEntry::{Const, Unchanged, Value as Val};
    let nil = Const(TraceScalar::Nil);
    let mut b = TraceBuilder::new(
        TraceId(0),
        TraceFrameInfo {
            register_count: 5,
            code_len: 8,
            inlined: None,
            activation: None,
        },
    );
    for register in [3, 4] {
        b.mark_written(FrameId(0), register);
    }
    let table = b.input(0, TraceType::Table);
    let key = b.input(1, key_ty);
    let value = store.then(|| b.input(2, ty));
    b.set_fields(vec![TableField {
        key: crate::jit_ir::trace::TABLE_ADDRESS_KEY.into(),
        ty: TraceType::Integer,
    }]);
    let held = [Unchanged, Unchanged, Unchanged];
    let regs = |first: RegisterEntry, second: RegisterEntry| {
        let mut regs = held.to_vec();
        regs.extend([first, second]);
        regs
    };
    let s0 = b.snapshot(snap(SnapshotPhase::Before { pc: 0 }, 0, regs(nil, nil)));
    let g0 = b.guard(site(0), s0);
    let mut effects = 0;
    let first = if let Some(value) = value {
        let effect = b.next_effect();
        b.emit(
            site(0),
            TraceOp::LaneArrayStore {
                table,
                field: 0,
                key,
                value,
                effect,
                guard: g0,
            },
            None,
        );
        effects = 1;
        None
    } else {
        Some(emit(
            &mut b,
            0,
            TraceOp::LaneArrayLoad {
                table,
                field: 0,
                key,
                expected: ty,
                guard: g0,
            },
            ty,
        ))
    };
    let s1 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 1 },
        effects,
        regs(first.map_or(nil, Val), nil),
    ));
    let g1 = b.guard(site(1), s1);
    let second = emit(
        &mut b,
        1,
        TraceOp::LaneArrayLoad {
            table,
            field: 0,
            key,
            expected: ty,
            guard: g1,
        },
        ty,
    );
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 1,
            successor: 2,
        },
        effects,
        regs(first.map_or(nil, Val), Val(second)),
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

/// What a run of a lane access leaves: the outcome, the exit area, the fuel
/// and the table's slots.
fn observe_lane(
    trace: &CompiledTrace,
    (slots, address_known): (&[Value], bool),
    (key, value): (TraceScalar, TraceScalar),
) -> (TraceOutcome, Vec<i64>, i64, String) {
    let mut table = LaneTable::new(slots.to_vec(), Vec::new());
    let address = table.words.as_mut_ptr() as usize;
    let registers = [TraceScalar::Table(0), key, value];
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| TraceInputSlot::from_scalar(registers[usize::from(*register)]))
        .collect();
    let mut area = vec![0x5a5a_i64; trace.exit_area_len()];
    let mut control = TraceControl::new(100);
    let mut space = TraceTableSpace::new();
    let mut dispatcher = LaneDispatcher {
        address: if address_known { address } else { 0 },
    };
    let outcome = trace.run_with_tables(
        &inputs,
        &[],
        &mut control,
        &mut area,
        &mut space,
        &mut dispatcher,
    );
    (outcome, area, control.fuel, format!("{:?}", table.slots))
}

/// The lane array loads and stores of both backends leave through the same
/// exits with the same values and memory, for every element type and key
/// type, slots of every type, keys inside and outside the array, a row
/// without the table's address, and a table in an array slot (lane 0 is the
/// one table the dispatcher knows).
#[test]
fn lane_array_loads_and_stores_match_cranelift() {
    use crate::jit::debug_env::FORCE_BACKEND;
    let compile_with = |ir: &TraceIr, tier, mir: bool| {
        FORCE_BACKEND.with(|force| force.set(Some(mir)));
        let compiled = compile_trace(ir, tier);
        FORCE_BACKEND.with(|force| force.set(None));
        compiled.unwrap()
    };
    let tables: [Vec<Value>; 7] = [
        floats(&[1.5, -2.0, 3.5]),
        vec![
            Value::Integer(10),
            Value::Integer(-3),
            Value::Integer(i64::MAX),
        ],
        vec![
            Value::Boolean(true),
            Value::Boolean(false),
            Value::Boolean(true),
        ],
        vec![
            Value::Number(0.5),
            Value::Integer(7),
            Value::Boolean(true),
            Value::Nil,
            Value::from("s"),
        ],
        vec![handle(0), Value::Number(1.0), handle(1)],
        Vec::new(),
        floats(&[9.0]),
    ];
    let int_keys = [0, 1, 2, 3, 4, 5, 6, -1, i64::MIN, i64::MAX];
    let float_keys = [
        0.0,
        -0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        1.5,
        f64::NAN,
        1e300,
        -1.0,
    ];
    let mut hits = 0;
    let mut misses = 0;
    for tier in TIERS {
        for store in [false, true] {
            for ty in [
                TraceType::Float,
                TraceType::Integer,
                TraceType::Boolean,
                TraceType::Table,
            ] {
                if store && ty == TraceType::Table {
                    continue;
                }
                for key_ty in [TraceType::Integer, TraceType::Float] {
                    let ir = lane_access_trace(key_ty, ty, store);
                    ir.verify().unwrap();
                    let mir = compile_with(&ir, tier, true);
                    let cranelift = compile_with(&ir, tier, false);
                    assert!(mir.mir_address().is_some(), "{ty:?} {key_ty:?}: MIR code");
                    assert!(cranelift.mir_address().is_none());
                    assert_eq!(mir.exits(), cranelift.exits());
                    let keys: Vec<TraceScalar> = if key_ty == TraceType::Float {
                        float_keys.iter().map(|k| TraceScalar::Float(*k)).collect()
                    } else {
                        int_keys.iter().map(|k| TraceScalar::Integer(*k)).collect()
                    };
                    let value = match ty {
                        TraceType::Float => TraceScalar::Float(42.5),
                        TraceType::Integer => TraceScalar::Integer(-9),
                        TraceType::Boolean => TraceScalar::Boolean(false),
                        _ => TraceScalar::Nil,
                    };
                    for slots in &tables {
                        for known in [true, false] {
                            for &key in &keys {
                                let case = ((slots.as_slice(), known), (key, value));
                                let (a, b) = (
                                    observe_lane(&mir, case.0, case.1),
                                    observe_lane(&cranelift, case.0, case.1),
                                );
                                assert_eq!(
                                    a, b,
                                    "{tier:?} store {store} {ty:?} key {key:?} in {slots:?} (address {known})"
                                );
                                match a.0 {
                                    TraceOutcome::Exit { index: 0 | 1 } => misses += 1,
                                    TraceOutcome::Exit { .. } => hits += 1,
                                    other => panic!("{other:?}"),
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(hits > 40 && misses > 100, "{hits} hits, {misses} misses");
}

/// Lane 0 is a record whose field `inner` holds a table; lane 1 is that
/// table, whose array the lane read goes through: the shape of
/// `region.solids[idx]`.
struct Nested {
    /// The slot of field `inner` of lane 0, or none for a row without it.
    inner: Option<usize>,
    /// The words of the inner table.
    words: usize,
}

impl TraceTables for Nested {
    fn lane(&mut self, value: &Value) -> Option<usize> {
        match identity_of(value) {
            Some(0) => Some(0),
            Some(1) => Some(1),
            _ => None,
        }
    }

    fn row(&mut self, lane: usize, row: &mut [usize]) {
        match lane {
            0 => row[0] = self.inner.unwrap_or(0),
            1 => row[1] = self.words,
            _ => {}
        }
    }
}

/// `r2 = t.inner` (a field that holds a table) and `r3 = r2[k]` (a lane read
/// of a float).
fn nested_trace() -> TraceIr {
    use RegisterEntry::{Const, Unchanged, Value as Val};
    let nil = Const(TraceScalar::Nil);
    let mut b = TraceBuilder::new(
        TraceId(0),
        TraceFrameInfo {
            register_count: 4,
            code_len: 8,
            inlined: None,
            activation: None,
        },
    );
    for register in [2, 3] {
        b.mark_written(FrameId(0), register);
    }
    let table = b.input(0, TraceType::Table);
    let key = b.input(1, TraceType::Integer);
    b.set_fields(vec![
        TableField {
            key: b"inner"[..].into(),
            ty: TraceType::Table,
        },
        TableField {
            key: crate::jit_ir::trace::TABLE_ADDRESS_KEY.into(),
            ty: TraceType::Integer,
        },
    ]);
    let s0 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 0 },
        0,
        vec![Unchanged, Unchanged, nil, nil],
    ));
    let g0 = b.guard(site(0), s0);
    let inner = emit(
        &mut b,
        0,
        TraceOp::TableFieldLoad {
            field: 0,
            table,
            guard: g0,
        },
        TraceType::Table,
    );
    let s1 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 1 },
        0,
        vec![Unchanged, Unchanged, Val(inner), nil],
    ));
    let g1 = b.guard(site(1), s1);
    let read = emit(
        &mut b,
        1,
        TraceOp::LaneArrayLoad {
            table: inner,
            field: 1,
            key,
            expected: TraceType::Float,
            guard: g1,
        },
        TraceType::Float,
    );
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 1,
            successor: 2,
        },
        0,
        vec![Unchanged, Unchanged, Val(inner), Val(read)],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

/// A field that holds a table gives the table's lane (`slot_access` of a
/// table slot in Cranelift): the load leaves when the slot holds another
/// type, a table the VM gives no lane to, or the row has no such field, and
/// the lane read that follows leaves when the key is outside the array.
#[test]
fn a_field_that_holds_a_table_is_read_by_lane_like_cranelift() {
    use crate::jit::debug_env::FORCE_BACKEND;
    let ir = nested_trace();
    ir.verify().unwrap();
    let compile_with = |tier, mir: bool| {
        FORCE_BACKEND.with(|force| force.set(Some(mir)));
        let compiled = compile_trace(&ir, tier);
        FORCE_BACKEND.with(|force| force.set(None));
        compiled.unwrap()
    };
    let observe = |trace: &CompiledTrace, (held, row): (&Value, bool), key: i64| {
        let slot = [held.clone()];
        let mut inner = LaneTable::new(floats(&[1.5, 2.5, 3.5]), Vec::new());
        let mut host = Nested {
            inner: row.then_some(slot.as_ptr() as usize),
            words: inner.words.as_mut_ptr() as usize,
        };
        let registers = [TraceScalar::Table(0), TraceScalar::Integer(key)];
        let inputs: Vec<TraceInputSlot> = trace
            .entry_contract()
            .iter()
            .map(|(register, _)| TraceInputSlot::from_scalar(registers[usize::from(*register)]))
            .collect();
        let mut area = vec![0x5a5a_i64; trace.exit_area_len()];
        let mut control = TraceControl::new(100);
        let mut space = TraceTableSpace::new();
        let outcome =
            trace.run_with_tables(&inputs, &[], &mut control, &mut area, &mut space, &mut host);
        (outcome, area, control.fuel)
    };
    let held = [
        handle(1),
        handle(5),
        Value::Number(1.0),
        Value::Nil,
        Value::from("s"),
    ];
    let mut reads = 0;
    let mut misses = 0;
    for tier in TIERS {
        let (mir, cranelift) = (compile_with(tier, true), compile_with(tier, false));
        assert!(mir.mir_address().is_some(), "{tier:?}: MIR code");
        assert!(cranelift.mir_address().is_none());
        assert_eq!(mir.exits(), cranelift.exits());
        for value in &held {
            for row in [true, false] {
                for key in [0, 1, 2, 3, 4, -1] {
                    let (a, b) = (
                        observe(&mir, (value, row), key),
                        observe(&cranelift, (value, row), key),
                    );
                    assert_eq!(a, b, "{tier:?} {value:?} row {row} key {key}");
                    match a.0 {
                        TraceOutcome::Exit { index: 0 | 1 } => misses += 1,
                        TraceOutcome::Exit { .. } => reads += 1,
                        other => panic!("{other:?}"),
                    }
                }
            }
        }
    }
    assert!(reads >= 6 && misses >= 40, "{reads} reads, {misses} misses");
}
