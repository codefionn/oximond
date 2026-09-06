//! [`TraceOp::MetaGuard`]: a loop over an array of tables whose every table
//! must have the pinned metatable. The row of a table holds the metatable's
//! word in the field [`META_FIELD_KEY`]; the guard is one load and one compare.
//!
//! The tests play the VM as `table_rows_tests` does: tables are the test's
//! words, the array names them by identity, and the dispatcher gives lanes
//! out in the order tables are met. The compiled trace runs at both tiers,
//! the reference evaluator runs over a copy of the same data, and the exit
//! and the restored registers must agree.

#![allow(clippy::too_many_lines, clippy::cast_possible_truncation)]

use super::*;
use crate::jit::OptimizationTier;
use crate::jit_ir::Comparison;
use crate::jit_ir::trace::reference::{Outcome, TraceHeap, evaluate_with_heap};
use crate::jit_ir::trace::{
    ArithmeticOp, BindingKind, FrameId, LoopParamInit, META_FIELD_KEY, META_PIN_ABSENT, MetaPin,
    RegisterEntry, Snapshot, SnapshotFrame, SnapshotPhase, TableBinding, TableField, TraceBuilder,
    TraceFrameInfo, TraceId, TraceIr, TraceOp, TraceScalar, TraceSite, TraceTerminator, TraceType,
    meta_word,
};
use crate::runtime::{TableHandle, Value};

const TIERS: [OptimizationTier; 2] = [OptimizationTier::Baseline, OptimizationTier::Optimized];

/// The identity of the metatable the trace pins.
const PINNED: u64 = 7;

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

/// Table 0 is the array; table `n` is the one the array's `n`th slot names.
#[derive(Clone)]
struct World {
    array: Vec<Value>,
    /// The word of each table's row: zero for a table without a metatable.
    words: Vec<u64>,
}

impl World {
    /// `n` tables, every one with the pinned metatable.
    fn of(n: usize) -> Self {
        Self {
            array: (1..=n as u64)
                .map(|identity| Value::Table(TableHandle::from_native_identity(identity)))
                .collect(),
            words: std::iter::once(0)
                .chain((0..n).map(|_| meta_word(PINNED)))
                .collect(),
        }
    }
}

impl TraceHeap for World {
    fn entry_accepts(&mut self, _: &[TableBinding]) -> bool {
        true
    }
    fn array_get(&mut self, _: u32, key: TraceScalar) -> Option<TraceScalar> {
        let TraceScalar::Float(index) = key else {
            return None;
        };
        let slot = usize::try_from((index as i64).checked_sub(1)?).ok()?;
        match self.array.get(slot)? {
            Value::Table(table) => Some(TraceScalar::Table(table.native_identity())),
            _ => None,
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
        assert_eq!(field, 0, "the metatable word is the only field");
        let word = *self.words.get(usize::try_from(table).ok()?)?;
        Some(TraceScalar::Integer(word.cast_signed()))
    }
    fn table_field_set(&mut self, _: u64, _: u32, _: TraceScalar) -> bool {
        unreachable!("these traces write no fields")
    }
}

struct Dispatcher<'a> {
    world: &'a mut World,
    lanes: Vec<u64>,
    rows_filled: usize,
}

impl TraceTables for Dispatcher<'_> {
    fn lane(&mut self, value: &Value) -> Option<usize> {
        let Value::Table(table) = value else {
            return None;
        };
        let identity = table.native_identity();
        if let Some(lane) = self.lanes.iter().position(|held| *held == identity) {
            return Some(lane);
        }
        self.lanes.push(identity);
        Some(self.lanes.len() - 1)
    }

    fn row(&mut self, lane: usize, row: &mut [usize]) {
        self.rows_filled += 1;
        let identity = usize::try_from(self.lanes[lane]).unwrap();
        // The row's one word is the metatable's word itself.
        row[0] = self.world.words.get(identity).copied().unwrap_or(0) as usize;
    }
}

/// Registers: r0 index (float), r1 the array, r2 limit, r3 the table.
///
/// ```lua
/// repeat
///   t = objects[i]     -- pc 0
///   t.k                -- pc 1: a read that needs the class
///   i = i + 1; until i > n
/// ```
fn loop_trace(metatable: u64) -> TraceIr {
    use RegisterEntry::{Const, Unchanged, Value as Val};
    let nil = Const(TraceScalar::Nil);
    let mut b = TraceBuilder::new(
        TraceId(0),
        TraceFrameInfo {
            register_count: 4,
            code_len: 16,
            inlined: None,
            activation: None,
        },
    );
    for register in [0, 3] {
        b.mark_written(FrameId(0), register);
    }
    let i_in = b.input(0, TraceType::Float);
    let objects = b.input(1, TraceType::Table);
    let limit = b.input(2, TraceType::Float);
    b.set_bindings(vec![
        TableBinding {
            register: 1,
            kind: BindingKind::Array { first: None },
        },
        TableBinding {
            register: 1,
            kind: BindingKind::MetaClass {
                metatable,
                pins: vec![MetaPin {
                    event: b"__index"[..].into(),
                    handler: 3,
                }]
                .into(),
                bare: 0,
            },
        },
    ]);
    b.set_fields(vec![TableField {
        key: META_FIELD_KEY.into(),
        ty: TraceType::Integer,
    }]);
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let s0 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 0 },
        0,
        vec![Val(i), Unchanged, Unchanged, nil],
    ));
    let g0 = b.guard(site(0), s0);
    let t = b
        .emit(
            site(0),
            TraceOp::ArrayLoad {
                binding: 0,
                table: objects,
                key: i,
                expected: TraceType::Table,
                past_end: false,
                guard: g0,
            },
            Some(TraceType::Table),
        )
        .unwrap();
    let s1 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 1 },
        0,
        vec![Val(i), Unchanged, Unchanged, Val(t)],
    ));
    let g1 = b.guard(site(1), s1);
    b.emit(
        site(1),
        TraceOp::MetaGuard {
            table: t,
            binding: 1,
            guard: g1,
        },
        None,
    );
    let step = b
        .emit(
            site(2),
            TraceOp::Constant(TraceScalar::Float(1.0)),
            Some(TraceType::Float),
        )
        .unwrap();
    let i_next = b
        .emit(
            site(2),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: i,
                right: step,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let more = b
        .emit(
            site(3),
            TraceOp::Compare {
                comparison: Comparison::LessEqual,
                left: i_next,
                right: limit,
            },
            Some(TraceType::Boolean),
        )
        .unwrap();
    let after = vec![Val(i_next), Unchanged, Unchanged, Val(t)];
    let s2 = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 3,
            successor: 4,
        },
        0,
        after.clone(),
    ));
    let g2 = b.guard(site(3), s2);
    b.emit(
        site(3),
        TraceOp::GuardBranch {
            condition: more,
            expected: true,
            guard: g2,
        },
        None,
    );
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 3,
            successor: 0,
        },
        0,
        after,
    ));
    b.set_poll_snapshot(poll);
    b.finish(TraceTerminator::LoopBack { args: vec![i_next] })
}

fn registers(start: f64, limit: f64) -> Vec<TraceScalar> {
    vec![
        TraceScalar::Float(start),
        TraceScalar::Table(0),
        TraceScalar::Float(limit),
        TraceScalar::Nil,
    ]
}

/// What a run did, in the terms both sides share.
#[derive(Debug, PartialEq)]
struct Observed {
    guard: Option<u32>,
    index: String,
}

fn run_native(
    ir: &TraceIr,
    tier: OptimizationTier,
    registers: &[TraceScalar],
    world: &World,
) -> Observed {
    let trace = compile_trace(ir, tier).unwrap();
    assert!(trace.needs_tables());
    let mut world = world.clone();
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| TraceInputSlot::from_scalar(registers[usize::from(*register)]))
        .collect();
    let bindings = [
        TraceBinding::raw_array(world.array.len(), world.array.as_mut_ptr() as usize),
        TraceBinding::callee(),
    ];
    let mut area = vec![0_i64; trace.exit_area_len()];
    let mut dispatcher = Dispatcher {
        world: &mut world,
        lanes: vec![0],
        rows_filled: 0,
    };
    let outcome = trace.run_with_tables(
        &inputs,
        &bindings,
        &mut TraceControl::new(100_000),
        &mut area,
        &mut TraceTableSpace::new(),
        &mut dispatcher,
    );
    let TraceOutcome::Exit { index } = outcome else {
        panic!("the trace refused entry");
    };
    let descriptor = &trace.exits()[index];
    let snapshot = &trace.lowered_ir().snapshots[descriptor.snapshot.index()];
    let state = match snapshot.frames[0].registers[0] {
        RegisterEntry::Value(value) => descriptor.scalar(&area, value).unwrap(),
        RegisterEntry::Const(scalar) => scalar,
        RegisterEntry::Unchanged => registers[0],
        _ => unreachable!("the index is a value"),
    };
    Observed {
        guard: descriptor.origin,
        index: format!("{state:?}"),
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
        index: format!("{:?}", restored.registers[0]),
    }
}

fn agree(metatable: u64, world: &World, start: f64, limit: f64, label: &str) -> Option<u32> {
    let ir = loop_trace(metatable);
    ir.verify().unwrap();
    let registers = registers(start, limit);
    let reference = run_reference(&ir, &registers, world);
    for tier in TIERS {
        let native = run_native(&ir, tier, &registers, world);
        assert_eq!(native, reference, "{label} {tier:?}");
    }
    reference.guard
}

#[test]
fn tables_of_the_pinned_metatable_pass() {
    let world = World::of(5);
    // The loop ends through its branch (guard 2).
    assert_eq!(agree(PINNED, &world, 1.0, 5.0, "all"), Some(2));
}

#[test]
fn a_table_of_another_metatable_leaves_at_the_guard() {
    let mut world = World::of(5);
    world.words[3] = meta_word(PINNED + 1);
    assert_eq!(agree(PINNED, &world, 1.0, 5.0, "another"), Some(1));
}

#[test]
fn a_table_without_a_metatable_leaves_at_the_guard() {
    let mut world = World::of(5);
    world.words[2] = 0;
    assert_eq!(agree(PINNED, &world, 1.0, 5.0, "none"), Some(1));
}

#[test]
fn the_metatable_of_identity_zero_is_not_a_table_without_one() {
    // The first table a runtime makes has identity zero. A row that names no
    // metatable holds zero, and the word of identity zero is one.
    let mut world = World::of(4);
    world
        .words
        .iter_mut()
        .skip(1)
        .for_each(|word| *word = meta_word(0));
    assert_eq!(agree(0, &world, 1.0, 4.0, "identity zero"), Some(2));
    world.words[2] = 0;
    assert_eq!(
        agree(0, &world, 1.0, 4.0, "identity zero, one bare"),
        Some(1)
    );
}

#[test]
fn the_absent_pin_names_no_closure() {
    // A closure's identity is an index and a generation: the identity that
    // stands for an absent event is none of them.
    assert_eq!(META_PIN_ABSENT, u64::MAX);
    assert_ne!(meta_word(u64::MAX - 1), 0);
}

#[test]
fn a_metatable_guard_needs_the_metatable_field() {
    let mut ir = loop_trace(PINNED);
    ir.fields.clear();
    assert!(ir.verify().is_err());
}

#[test]
fn a_metatable_guard_names_a_class_binding() {
    let mut ir = loop_trace(PINNED);
    for instruction in &mut ir.body {
        if let TraceOp::MetaGuard { binding, .. } = &mut instruction.op {
            *binding = 0;
        }
    }
    assert!(ir.verify().is_err());
}

#[test]
fn one_table_met_twice_has_one_row() {
    let mut world = World::of(3);
    world.array[2] = world.array[0].clone();
    let ir = loop_trace(PINNED);
    let registers = registers(1.0, 3.0);
    let trace = compile_trace(&ir, OptimizationTier::Optimized).unwrap();
    let mut model = world.clone();
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| TraceInputSlot::from_scalar(registers[usize::from(*register)]))
        .collect();
    let bindings = [
        TraceBinding::raw_array(model.array.len(), model.array.as_mut_ptr() as usize),
        TraceBinding::callee(),
    ];
    let mut area = vec![0_i64; trace.exit_area_len()];
    let mut dispatcher = Dispatcher {
        world: &mut model,
        lanes: vec![0],
        rows_filled: 0,
    };
    trace.run_with_tables(
        &inputs,
        &bindings,
        &mut TraceControl::new(100_000),
        &mut area,
        &mut TraceTableSpace::new(),
        &mut dispatcher,
    );
    // Two tables. The array's own lane (an entry input) is only bound: no
    // operation reads a field of it, so it gets no row.
    assert_eq!(dispatcher.rows_filled, 2);
}
