//! Table operations in the trace IR: the verifier's rules, the reference
//! evaluator's semantics over a heap model, and the optimizer's promise never
//! to move, merge or drop a table access.
//!
//! The worked recovery case of the plan runs here with real table operations.

#![allow(
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]

use super::helper::{HelperEffects, HelperId, HelperType, MAX_RESUME_VALUES};
use super::ir::*;
use super::optimize::{
    OptimizeLevel, OptimizeOptions, OptimizeStats, TreeWrites, optimize_trace, optimize_with,
    optimize_with_tree,
};
use super::reference::{
    Evaluation, ForcedExit, Outcome, TABLE_STORE_LABEL, TraceHeap, evaluate_with_heap,
};
use super::snapshot::*;
use super::verify::TraceVerifyError;
use crate::jit_ir::Comparison;

fn site(pc: usize) -> TraceSite {
    TraceSite {
        frame: FrameId(0),
        pc,
    }
}

fn root(registers: u16) -> TraceFrameInfo {
    TraceFrameInfo {
        register_count: registers,
        code_len: 32,
        inlined: None,
        activation: None,
    }
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

/// The table slots a binding resolves, as a model: one array per array
/// binding, one value per field binding, one length per length binding.
#[derive(Clone, Debug)]
pub(super) struct Model {
    arrays: Vec<Vec<TraceScalar>>,
    fields: Vec<TraceScalar>,
    lengths: Vec<TraceScalar>,
    /// The fields of run-time tables: for each lane, one value per entry of
    /// [`TraceIr::fields`], `Nil` for a field the table lacks.
    rows: Vec<Vec<TraceScalar>>,
    accept: bool,
    /// Keys a table set is refused for, as the VM refuses a key that may
    /// need `__newindex`.
    refused: Vec<TraceScalar>,
    /// The storage each binding resolves, when bindings share it, as two
    /// registers holding one table or two tables reaching one field through
    /// `__index` do. Empty means each binding has its own storage.
    alias: Vec<usize>,
}

impl Model {
    fn store(&self, binding: u32) -> usize {
        self.alias
            .get(binding as usize)
            .copied()
            .unwrap_or(binding as usize)
    }

    fn array(values: Vec<TraceScalar>) -> Self {
        Self {
            arrays: vec![values],
            fields: Vec::new(),
            lengths: Vec::new(),
            rows: Vec::new(),
            accept: true,
            refused: Vec::new(),
            alias: Vec::new(),
        }
    }

    /// The zero-based slot a key names, if it is integral and inside the
    /// array.
    fn slot(&self, binding: u32, key: TraceScalar) -> Option<usize> {
        let index = match key {
            TraceScalar::Integer(index) => index,
            TraceScalar::Float(value) if value as i64 as f64 == value => value as i64,
            _ => return None,
        };
        let slot = usize::try_from(index.checked_sub(1)?).ok()?;
        (slot < self.arrays[self.store(binding)].len()).then_some(slot)
    }
}

impl TraceHeap for Model {
    fn entry_accepts(&mut self, _: &[TableBinding]) -> bool {
        self.accept
    }
    fn array_get(&mut self, binding: u32, key: TraceScalar) -> Option<TraceScalar> {
        let slot = self.slot(binding, key)?;
        let value = self.arrays[self.store(binding)][slot];
        (value != TraceScalar::Nil).then_some(value)
    }
    fn array_set(&mut self, binding: u32, key: TraceScalar, value: TraceScalar) -> bool {
        let Some(slot) = self.slot(binding, key) else {
            return false;
        };
        let store = self.store(binding);
        let held = &mut self.arrays[store][slot];
        if held.ty() != value.ty() {
            return false;
        }
        *held = value;
        true
    }
    fn helper_call(&mut self, helper: HelperId, args: &[TraceScalar]) -> Option<Vec<TraceScalar>> {
        match (helper, args) {
            (HelperId::TEST_DIVMOD, [TraceScalar::Integer(a), TraceScalar::Integer(b)]) => {
                (*b != 0).then(|| vec![int(a.div_euclid(*b)), int(a.rem_euclid(*b))])
            }
            (HelperId::TEST_READ, [TraceScalar::Integer(a)]) => Some(vec![int(*a)]),
            (HelperId::TEST_DRAW, [TraceScalar::Integer(a)]) => Some(vec![int(a + 1)]),
            // Writes its argument into the first slot of the first array.
            (HelperId::TEST_WRITE, [TraceScalar::Integer(a)]) => {
                self.arrays[0][0] = f(*a as f64);
                Some(vec![int(*a)])
            }
            _ => None,
        }
    }
    fn field_get(&mut self, binding: u32) -> TraceScalar {
        self.fields[self.store(binding)]
    }
    fn field_set(&mut self, binding: u32, value: TraceScalar) {
        let store = self.store(binding);
        self.fields[store] = value;
    }
    fn length(&mut self, binding: u32) -> TraceScalar {
        self.lengths[binding as usize]
    }
    fn table_field_get(&mut self, table: u64, field: u32) -> Option<TraceScalar> {
        let value = self.rows[table as usize][field as usize];
        (value != TraceScalar::Nil).then_some(value)
    }
    fn table_field_set(&mut self, table: u64, field: u32, value: TraceScalar) -> bool {
        let held = &mut self.rows[table as usize][field as usize];
        if held.ty() != value.ty() {
            return false;
        }
        *held = value;
        true
    }
    fn array_hole(&mut self, binding: u32, key: TraceScalar) -> bool {
        self.slot(binding, key)
            .is_some_and(|slot| self.arrays[self.store(binding)][slot] == TraceScalar::Nil)
    }
    /// A raw set of an integral key of the array part: it overwrites any
    /// slot, or appends at the end. The model refuses every other key, and
    /// the keys of `refused`.
    fn table_set(&mut self, binding: u32, key: TraceScalar, value: TraceScalar) -> bool {
        if self.refused.contains(&key) {
            return false;
        }
        if let Some(slot) = self.slot(binding, key) {
            let store = self.store(binding);
            self.arrays[store][slot] = value;
            return true;
        }
        let store = self.store(binding);
        let array = &mut self.arrays[store];
        let next = f(array.len() as f64 + 1.0);
        if key == next || key == int(array.len() as i64 + 1) {
            array.push(value);
            return true;
        }
        false
    }
}

fn run(ir: &TraceIr, model: &mut Model, registers: &[TraceScalar], max: u32) -> Evaluation {
    run_forced(ir, model, registers, max, None)
}

fn run_forced(
    ir: &TraceIr,
    model: &mut Model,
    registers: &[TraceScalar],
    max: u32,
    forced: Option<ForcedExit>,
) -> Evaluation {
    evaluate_with_heap(ir, registers, &[], max, forced, model, |_, _| None)
}

/// The worked case over a real array binding:
///
/// ```lua
/// while ... do
///   local x = t[i]      -- pc 1: ArrayLoad, guard leaves before it
///   t[i] = x + 1        -- pc 2: ArrayStore, guard leaves before it
///   if x > 0 then ...   -- pc 3/4: branch guard, fails after the store
///     sum = sum + x
///   i = i + 1
/// end
/// ```
///
/// Registers: r0 i, r1 sum, r2 x, r3 condition, r4 the table, which the trace
/// never writes.
fn table_case() -> TraceIr {
    use RegisterEntry::{Unchanged, Value};
    let mut b = TraceBuilder::new(TraceId(1), root(5));
    for register in 0..4 {
        b.mark_written(FrameId(0), register);
    }
    let i_in = b.input(0, TraceType::Float);
    let sum_in = b.input(1, TraceType::Float);
    let x_in = b.input(2, TraceType::Float);
    let cond_in = b.input(3, TraceType::Boolean);
    let table = b.input(4, TraceType::Table);
    b.set_bindings(vec![TableBinding {
        register: 4,
        kind: BindingKind::Array { first: None },
    }]);
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let sum = b.loop_param(TraceType::Float, LoopParamInit::Input(sum_in));
    let x_prev = b.loop_param(TraceType::Float, LoopParamInit::Input(x_in));
    let cond_prev = b.loop_param(TraceType::Boolean, LoopParamInit::Input(cond_in));

    let s0 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 1 },
        b.effect_position(),
        Some(0),
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

    let one = emit(&mut b, 2, TraceOp::Constant(f(1.0)), TraceType::Float);
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

    let zero = emit(&mut b, 3, TraceOp::Constant(f(0.0)), TraceType::Float);
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
    let i_next = emit(
        &mut b,
        6,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: i,
            right: one,
        },
        TraceType::Float,
    );
    b.finish(TraceTerminator::LoopBack {
        args: vec![i_next, sum_next, x, cond],
    })
}

/// Entry registers of [`table_case`]. The table register is any table
/// scalar: the model resolves storage, the register only has to be a table.
fn entry(i: f64, sum: f64) -> [TraceScalar; 5] {
    [
        f(i),
        f(sum),
        f(0.0),
        TraceScalar::Boolean(false),
        TraceScalar::Table(0),
    ]
}

fn stores(evaluation: &Evaluation) -> Vec<Vec<TraceScalar>> {
    evaluation
        .effects
        .iter()
        .map(|record| {
            assert_eq!(record.label, TABLE_STORE_LABEL);
            record.operands.clone()
        })
        .collect()
}

#[test]
fn the_table_case_verifies() {
    table_case().verify().unwrap();
}

#[test]
fn a_branch_guard_after_a_committed_array_store_keeps_the_store() {
    let ir = table_case();
    let mut model = Model::array(vec![f(3.0), f(5.0), f(-2.0), f(4.0)]);
    let evaluation = run(&ir, &mut model, &entry(1.0, 0.0), 10);
    // Iterations at keys 1 and 2 pass. The one at key 3 reads -2, stores
    // -1, and then fails the branch guard.
    let Outcome::GuardFailed { guard, restored } = &evaluation.outcome else {
        panic!("expected a guard failure: {:?}", evaluation.outcome)
    };
    assert_eq!(*guard, GuardId(2));
    assert_eq!(restored.pc, 9);
    assert_eq!(restored.last_interpreted_pc, Some(4));
    assert_eq!(restored.effect_position, 1);
    assert_eq!(
        restored.registers[..4],
        [f(3.0), f(8.0), f(-2.0), TraceScalar::Boolean(false)]
    );
    // Three stores, each once, the last one before the failing guard.
    assert_eq!(
        stores(&evaluation),
        vec![
            vec![f(1.0), f(4.0)],
            vec![f(2.0), f(6.0)],
            vec![f(3.0), f(-1.0)],
        ]
    );
    // The interpreter resumes after the branch, so it never stores again.
    assert_eq!(model.arrays[0], vec![f(4.0), f(6.0), f(-1.0), f(4.0)]);
}

#[test]
fn every_way_an_array_access_can_fail_leaves_before_the_instruction() {
    let ir = table_case();
    // A key past the end, a nil slot, another type, a fractional key, a
    // key below one, and NaN.
    let cases: [(Vec<TraceScalar>, f64); 6] = [
        (vec![f(1.0)], 2.0),
        (vec![f(1.0), TraceScalar::Nil], 2.0),
        (vec![f(1.0), int(7)], 2.0),
        (vec![f(1.0), f(2.0)], 1.5),
        (vec![f(1.0), f(2.0)], 0.0),
        (vec![f(1.0), f(2.0)], f64::NAN),
    ];
    for (values, key) in cases {
        let mut model = Model::array(values.clone());
        let evaluation = run(&ir, &mut model, &entry(key, 0.0), 10);
        let Outcome::GuardFailed { guard, restored } = &evaluation.outcome else {
            panic!("{values:?} at {key}: expected a guard failure")
        };
        assert_eq!(*guard, GuardId(0), "{values:?} at {key}");
        assert_eq!(restored.pc, 1);
        assert_eq!(restored.effect_position, 0);
        assert_eq!(evaluation.effects.len(), 0);
        assert_eq!(model.arrays[0], values, "a failed read changes nothing");
    }
}

#[test]
fn a_store_that_would_change_the_slots_type_is_not_committed() {
    // Reading and storing a float into a slot that holds an integer: the
    // load fails first, so build the case with a store guard of its own.
    let ir = table_case();
    let mut model = Model::array(vec![f(2.0)]);
    // Force the store's guard: the load passed, the store did not.
    let forced = ForcedExit {
        frame: FrameId(0),
        pc: 2,
        guard: Some(GuardId(1)),
        iteration: 0,
    };
    let evaluation = run_forced(&ir, &mut model, &entry(1.0, 0.0), 10, Some(forced));
    let Outcome::GuardFailed { guard, restored } = &evaluation.outcome else {
        panic!("expected the forced guard to fail")
    };
    assert_eq!(*guard, GuardId(1));
    assert_eq!(restored.pc, 2);
    assert_eq!(restored.effect_position, 0);
    assert_eq!(evaluation.effects.len(), 0);
    assert_eq!(model.arrays[0], vec![f(2.0)]);
}

#[test]
fn an_entry_the_model_refuses_runs_nothing() {
    let ir = table_case();
    let mut model = Model::array(vec![f(1.0)]);
    model.accept = false;
    let evaluation = run(&ir, &mut model, &entry(1.0, 0.0), 10);
    assert_eq!(evaluation.outcome, Outcome::EntryMismatch);
    assert_eq!(evaluation.effects.len(), 0);
}

fn expect_error(mutate: impl FnOnce(&mut TraceIr), check: impl FnOnce(&TraceVerifyError) -> bool) {
    let mut ir = table_case();
    mutate(&mut ir);
    let error = ir.verify().expect_err("the mutation must not verify");
    assert!(check(&error), "unexpected error: {error}");
}

fn is_type_error(error: &TraceVerifyError) -> bool {
    matches!(error, TraceVerifyError::TypeMismatch { .. })
}

#[test]
fn the_verifier_rejects_a_table_operand_that_is_not_the_bound_input() {
    // The key is not a table.
    expect_error(
        |ir| {
            if let TraceOp::ArrayLoad { table, key, .. } = &mut ir.body[0].op {
                *table = *key;
            }
        },
        is_type_error,
    );
    // The binding names another register than the input the operand is.
    expect_error(|ir| ir.bindings[0].register = 3, |_| true);
    // A binding for a missing entry input.
    expect_error(
        |ir| {
            ir.bindings.push(TableBinding {
                register: 0,
                kind: BindingKind::Array { first: None },
            });
        },
        |error| matches!(error, TraceVerifyError::InvalidEntry { .. }),
    );
}

#[test]
fn the_verifier_rejects_a_binding_of_the_wrong_kind_and_bad_types() {
    expect_error(
        |ir| {
            ir.bindings[0].kind = BindingKind::Length {
                ty: TraceType::Float,
                raw: false,
            };
        },
        is_type_error,
    );
    // A store of a value that is not a native slot type.
    expect_error(
        |ir| {
            if let TraceOp::ArrayStore { value, .. } = &mut ir.body[3].op {
                // r4's table input
                *value = ValueId(4);
            }
        },
        is_type_error,
    );
    // An array key that is not a number.
    expect_error(
        |ir| {
            if let TraceOp::ArrayLoad { key, .. } = &mut ir.body[0].op {
                *key = ValueId(4);
            }
        },
        is_type_error,
    );
}

/// One `ArrayNext` over the array bound to register 1, giving a number of
/// type `ty` from the key in register 0 (of type `key`).
fn array_next_trace(key: TraceType, ty: TraceType, start: bool) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), root(4));
    b.mark_written(FrameId(0), 2);
    let k = b.input(0, key);
    let table = b.input(1, TraceType::Table);
    b.set_bindings(vec![TableBinding {
        register: 1,
        kind: BindingKind::Array { first: None },
    }]);
    let registers = |next: Option<ValueId>| {
        vec![
            RegisterEntry::Value(k),
            RegisterEntry::Unchanged,
            next.map_or(RegisterEntry::Const(TraceScalar::Nil), RegisterEntry::Value),
            RegisterEntry::Const(TraceScalar::Nil),
        ]
    };
    let before = b.snapshot(snap(
        SnapshotPhase::Before { pc: 0 },
        0,
        None,
        registers(None),
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
            ty,
            guard,
        },
        ty,
    );
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 0,
            successor: 1,
        },
        0,
        Some(0),
        registers(Some(next)),
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

#[test]
fn the_verifier_checks_an_array_step() {
    for key in [TraceType::Integer, TraceType::Float] {
        for ty in [TraceType::Integer, TraceType::Float] {
            array_next_trace(key, ty, false).verify().unwrap();
        }
    }
    // The step gives a number, from a number.
    for ty in [TraceType::Boolean, TraceType::Table, TraceType::Nil] {
        let error = array_next_trace(TraceType::Integer, ty, false)
            .verify()
            .unwrap_err();
        assert!(is_type_error(&error), "{ty:?}: {error:?}");
    }
    let error = array_next_trace(TraceType::Boolean, TraceType::Integer, false)
        .verify()
        .unwrap_err();
    assert!(is_type_error(&error), "{error:?}");
    // A step is a heap read: the optimizer keeps it though nothing reads its
    // result.
    let ir = array_next_trace(TraceType::Integer, TraceType::Integer, false);
    ir.verify().unwrap();
    let (optimized, _) = optimize_trace(&ir, OptimizeLevel::Optimized);
    assert!(
        optimized
            .body
            .iter()
            .any(|instruction| matches!(instruction.op, TraceOp::ArrayNext { .. }))
    );
}

#[test]
fn the_verifier_rejects_an_array_hint_for_a_type_with_no_payload() {
    // A table has none either, but an array of tables is read as such (see
    // table_rows_tests), so its first slot may name the version.
    for hint in [TraceType::Nil, TraceType::Closure] {
        expect_error(
            |ir| ir.bindings[0].kind = BindingKind::Array { first: Some(hint) },
            |error| matches!(error, TraceVerifyError::InvalidEntry { .. }),
        );
    }
    let mut ir = table_case();
    ir.bindings[0].kind = BindingKind::Array {
        first: Some(TraceType::Float),
    };
    ir.verify().unwrap();
}

#[test]
fn the_verifier_rejects_table_operations_in_the_preheader_and_bindings_with_sides() {
    expect_error(
        |ir| {
            let load = ir.body.remove(0);
            ir.preheader.push(load);
        },
        |_| true,
    );
    let mut ir = table_case();
    ir.sides.push(TraceSide {
        parent: None,
        guard: 2,
        ir: Box::new(table_case()),
    });
    assert!(ir.verify().is_err());
}

#[test]
fn a_store_numbers_its_effect_and_counts_in_snapshots() {
    let mut ir = table_case();
    assert_eq!(ir.effect_count(), 1);
    // The branch guard's snapshot after the store must count it.
    ir.snapshots[2].effect_position = 0;
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::EffectOrder { .. })
    ));
    // The store's own guard must not count it.
    let mut ir = table_case();
    ir.snapshots[1].effect_position = 1;
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::EffectOrder { .. })
    ));
}

/// A field binding trace: `p.x = p.x + p.v * dt` with `dt` a loop-invariant
/// input and the table `p` in r2.
fn field_case() -> TraceIr {
    use RegisterEntry::{Unchanged, Value};
    let mut b = TraceBuilder::new(TraceId(3), root(4));
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
            kind: BindingKind::Length {
                ty: TraceType::Float,
                raw: false,
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
    let len = emit(
        &mut b,
        6,
        TraceOp::TableLength { binding: 2, table },
        TraceType::Float,
    );
    let one = emit(&mut b, 7, TraceOp::Constant(f(1.0)), TraceType::Float);
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
            right: len,
        },
        TraceType::Boolean,
    );
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 9,
            successor: 1,
        },
        b.effect_position(),
        Some(9),
        vec![Value(i_next), Unchanged, Unchanged, Unchanged],
    ));
    b.set_poll_snapshot(poll);
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 9,
            successor: 10,
        },
        b.effect_position(),
        Some(9),
        vec![Value(i_next), Unchanged, Unchanged, Unchanged],
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
    b.finish(TraceTerminator::LoopBack { args: vec![i_next] })
}

#[test]
fn field_stores_and_loads_follow_program_order() {
    let ir = field_case();
    ir.verify().unwrap();
    let mut model = Model {
        arrays: Vec::new(),
        fields: vec![f(1.0), f(0.5), f(0.0)],
        lengths: vec![TraceScalar::Nil, TraceScalar::Nil, f(3.0)],
        rows: Vec::new(),
        accept: true,
        refused: Vec::new(),
        alias: Vec::new(),
    };
    let registers = [f(1.0), f(2.0), TraceScalar::Table(0), TraceScalar::Nil];
    let evaluation = run(&ir, &mut model, &registers, 100);
    // Three iterations (i = 1, 2, 3), each x = x + v * 2 with x read after
    // the previous store.
    assert!(matches!(evaluation.outcome, Outcome::GuardFailed { .. }));
    assert_eq!(
        stores(&evaluation),
        vec![vec![f(2.0)], vec![f(3.0)], vec![f(4.0)]]
    );
    assert_eq!(model.fields[0], f(4.0));
}

#[test]
fn a_dead_field_load_and_a_dead_length_go_but_stores_and_array_reads_stay() {
    // A field read and a length read cannot fail and change nothing, so
    // unused ones are dead code.
    use RegisterEntry::{Unchanged, Value};
    let mut b = TraceBuilder::new(TraceId(6), root(3));
    b.mark_written(FrameId(0), 0);
    let i_in = b.input(0, TraceType::Float);
    let table = b.input(1, TraceType::Table);
    b.set_bindings(vec![
        TableBinding {
            register: 1,
            kind: BindingKind::Field {
                key: b"x"[..].into(),
                ty: TraceType::Float,
                depth: 0,
            },
        },
        TableBinding {
            register: 1,
            kind: BindingKind::Length {
                ty: TraceType::Float,
                raw: false,
            },
        },
    ]);
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    emit(
        &mut b,
        1,
        TraceOp::FieldLoad { binding: 0, table },
        TraceType::Float,
    );
    emit(
        &mut b,
        2,
        TraceOp::TableLength { binding: 1, table },
        TraceType::Float,
    );
    let one = emit(&mut b, 3, TraceOp::Constant(f(1.0)), TraceType::Float);
    let i_next = emit(
        &mut b,
        3,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: i,
            right: one,
        },
        TraceType::Float,
    );
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 4,
            successor: 1,
        },
        0,
        Some(4),
        vec![Value(i_next), Unchanged, Unchanged],
    ));
    b.set_poll_snapshot(poll);
    let ir = b.finish(TraceTerminator::LoopBack { args: vec![i_next] });
    ir.verify().unwrap();
    for level in [OptimizeLevel::Baseline, OptimizeLevel::Optimized] {
        let (optimized, _) = optimize_trace(&ir, level);
        assert!(
            optimized.body.iter().all(|i| !i.op.touches_heap()),
            "{level:?}: {:?}",
            optimized.body
        );
    }
    // An array read nothing uses still guards the access, and stays, and
    // so does the store.
    for level in [OptimizeLevel::Baseline, OptimizeLevel::Optimized] {
        let (optimized, _) = optimize_trace(&table_case(), level);
        let loads = optimized
            .body
            .iter()
            .filter(|i| matches!(i.op, TraceOp::ArrayLoad { .. }))
            .count();
        let stored = optimized
            .body
            .iter()
            .filter(|i| matches!(i.op, TraceOp::ArrayStore { .. }))
            .count();
        assert_eq!((loads, stored), (1, 1), "{level:?}");
    }
}

#[test]
fn table_operations_are_never_merged_or_hoisted() {
    // Two loads of one field around a store of it must stay three
    // operations in order, unless the memory pass forwards the stored value
    // to the second load (the store stays).
    use RegisterEntry::{Unchanged, Value};
    let mut b = TraceBuilder::new(TraceId(4), root(4));
    b.mark_written(FrameId(0), 0);
    let i_in = b.input(0, TraceType::Float);
    let table = b.input(1, TraceType::Table);
    b.set_bindings(vec![TableBinding {
        register: 1,
        kind: BindingKind::Field {
            key: b"x"[..].into(),
            ty: TraceType::Float,
            depth: 0,
        },
    }]);
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let first = emit(
        &mut b,
        1,
        TraceOp::FieldLoad { binding: 0, table },
        TraceType::Float,
    );
    let one = emit(&mut b, 2, TraceOp::Constant(f(1.0)), TraceType::Float);
    let bumped = emit(
        &mut b,
        2,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: first,
            right: one,
        },
        TraceType::Float,
    );
    let effect = b.next_effect();
    b.emit(
        site(3),
        TraceOp::FieldStore {
            binding: 0,
            table,
            value: bumped,
            effect,
        },
        None,
    );
    let second = emit(
        &mut b,
        4,
        TraceOp::FieldLoad { binding: 0, table },
        TraceType::Float,
    );
    let i_next = emit(
        &mut b,
        5,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: i,
            right: second,
        },
        TraceType::Float,
    );
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 6,
            successor: 1,
        },
        b.effect_position(),
        Some(6),
        vec![Value(i_next), Unchanged, Unchanged, Unchanged],
    ));
    b.set_poll_snapshot(poll);
    let ir = b.finish(TraceTerminator::LoopBack { args: vec![i_next] });
    ir.verify().unwrap();
    for (level, memory) in [
        (OptimizeLevel::Baseline, false),
        (OptimizeLevel::Optimized, false),
        (OptimizeLevel::Optimized, true),
    ] {
        let (optimized, _) = optimize_with(
            &ir,
            OptimizeOptions {
                memory,
                ..OptimizeOptions::for_level(level)
            },
        );
        let order: Vec<&'static str> = optimized
            .body
            .iter()
            .filter_map(|i| match i.op {
                TraceOp::FieldLoad { .. } => Some("load"),
                TraceOp::FieldStore { .. } => Some("store"),
                _ => None,
            })
            .collect();
        if memory {
            assert_eq!(order, ["load", "store"], "{level:?}");
        } else {
            assert_eq!(order, ["load", "store", "load"], "{level:?}");
        }
        assert_eq!(optimized.preheader.len(), 0, "{level:?}");
        let run = |ir: &TraceIr| {
            let mut model = Model {
                arrays: Vec::new(),
                fields: vec![f(10.0)],
                lengths: vec![TraceScalar::Nil],
                rows: Vec::new(),
                accept: true,
                refused: Vec::new(),
                alias: Vec::new(),
            };
            let registers = [
                f(0.0),
                TraceScalar::Table(0),
                TraceScalar::Nil,
                TraceScalar::Nil,
            ];
            let evaluation = run_forced(ir, &mut model, &registers, 4, None);
            (stores(&evaluation), model.fields[0], evaluation.outcome)
        };
        assert_eq!(run(&ir), run(&optimized), "{level:?}");
    }
}

#[test]
fn optimized_table_traces_match_the_original_under_every_forced_exit() {
    let ir = table_case();
    let heaps = [
        vec![f(3.0), f(5.0), f(-2.0), f(4.0)],
        vec![f(3.0), f(5.0), f(2.0), f(4.0), f(1.0)],
        vec![f(1.0), TraceScalar::Nil, f(1.0)],
        vec![f(-1.0)],
    ];
    for level in [OptimizeLevel::Baseline, OptimizeLevel::Optimized] {
        let (optimized, _) = optimize_trace(&ir, level);
        optimized.verify().unwrap();
        for values in &heaps {
            for forced_guard in [None, Some((1, 0)), Some((2, 1)), Some((4, 2))] {
                for iteration in 0..3 {
                    let forced = forced_guard.map(|(pc, guard)| ForcedExit {
                        frame: FrameId(0),
                        pc,
                        guard: Some(GuardId(guard)),
                        iteration,
                    });
                    let mut original_model = Model::array(values.clone());
                    let mut optimized_model = original_model.clone();
                    let original =
                        run_forced(&ir, &mut original_model, &entry(1.0, 0.0), 6, forced);
                    let after = run_forced(
                        &optimized,
                        &mut optimized_model,
                        &entry(1.0, 0.0),
                        6,
                        forced,
                    );
                    assert_eq!(original.outcome, after.outcome, "{level:?} {values:?}");
                    assert_eq!(original.effects, after.effects, "{level:?} {values:?}");
                    assert_eq!(
                        original_model.arrays, optimized_model.arrays,
                        "{level:?} {values:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_global_slot_may_hold_a_table_next_to_stores_but_not_a_number() {
    // A table global is never rebound by a payload store. A number global
    // can change under one (through the environment table), so the trace
    // that holds it would read a stale value.
    let mut ir = table_case();
    ir.external = vec![ExternalSlot::Global(0)];
    // The table is the last window register, the global's.
    ir.verify().unwrap();

    let mut b = TraceBuilder::new(TraceId(5), root(2));
    b.set_external(vec![ExternalSlot::Global(0)]);
    let table = b.input(0, TraceType::Table);
    let global = b.input(1, TraceType::Float);
    b.set_bindings(vec![TableBinding {
        register: 0,
        kind: BindingKind::Field {
            key: b"x"[..].into(),
            ty: TraceType::Float,
            depth: 0,
        },
    }]);
    let effect = b.next_effect();
    b.emit(
        site(1),
        TraceOp::FieldStore {
            binding: 0,
            table,
            value: global,
            effect,
        },
        None,
    );
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 2,
            successor: 1,
        },
        b.effect_position(),
        Some(2),
        vec![RegisterEntry::Unchanged, RegisterEntry::Unchanged],
    ));
    b.set_poll_snapshot(poll);
    let ir = b.finish(TraceTerminator::LoopBack { args: vec![] });
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::InvalidEntry { .. })
    ));
}

#[test]
fn a_callee_binding_needs_a_pinned_lane_and_a_table_entry_input() {
    let callee = |lane| TableBinding {
        register: 4,
        kind: BindingKind::Callee {
            key: b"sin"[..].into(),
            lane,
            depth: 0,
        },
    };
    // A binding for a callable the trace pinned verifies.
    let mut ir = table_case();
    ir.pinned_closures = vec![pinned_native_identity(3)];
    ir.bindings.push(callee(0));
    ir.verify().unwrap();
    // A lane past the pinned list does not.
    let mut ir = table_case();
    ir.pinned_closures = vec![pinned_native_identity(3)];
    ir.bindings.push(callee(1));
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::InvalidEntry { .. })
    ));
    // No operation can read a callee binding as storage.
    let mut ir = table_case();
    ir.pinned_closures = vec![pinned_native_identity(3)];
    ir.bindings[0] = callee(0);
    assert!(ir.verify().is_err());
    // The identities round-trip, and a closure never reads as a native.
    assert_eq!(pinned_native_index(pinned_native_identity(0)), Some(0));
    assert_eq!(
        pinned_native_index(pinned_native_identity(u32::MAX)),
        Some(u32::MAX)
    );
    assert_eq!(pinned_native_index(0x1234_0000_0042), None);
}

#[test]
fn a_store_needs_the_bound_tables_own_field_and_hops_are_bounded() {
    // A read may go through `__index` hops, a store may not: it would land
    // in the bound table, not in the fallback the field was found in.
    let mut ir = field_case();
    ir.verify().unwrap();
    if let BindingKind::Field { depth, .. } = &mut ir.bindings[0].kind {
        *depth = 1;
    }
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::TypeMismatch { .. })
    ));
    // The loaded field (binding 1) may sit behind hops, up to the limit.
    let mut ir = field_case();
    if let BindingKind::Field { depth, .. } = &mut ir.bindings[1].kind {
        *depth = MAX_INDEX_HOPS;
    }
    ir.verify().unwrap();
    if let BindingKind::Field { depth, .. } = &mut ir.bindings[1].kind {
        *depth = MAX_INDEX_HOPS + 1;
    }
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::InvalidEntry { .. })
    ));
}

/// `t[i] = i; i = i + 1`, looping for ever over a table set
/// ([`TraceOp::ArraySet`]).
fn set_case() -> TraceIr {
    use RegisterEntry::{Unchanged, Value};
    let mut b = TraceBuilder::new(TraceId(2), root(2));
    b.mark_written(FrameId(0), 0);
    let i_in = b.input(0, TraceType::Float);
    let table = b.input(1, TraceType::Table);
    b.set_bindings(vec![TableBinding {
        register: 1,
        kind: BindingKind::Array { first: None },
    }]);
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let s0 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 1 },
        b.effect_position(),
        Some(0),
        vec![Value(i), Unchanged],
    ));
    let g0 = b.guard(site(1), s0);
    let effect = b.next_effect();
    b.emit(
        site(1),
        TraceOp::ArraySet {
            binding: 0,
            table,
            key: i,
            value: i,
            effect,
            guard: g0,
        },
        None,
    );
    let one = emit(&mut b, 2, TraceOp::Constant(f(1.0)), TraceType::Float);
    let i_next = emit(
        &mut b,
        2,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: i,
            right: one,
        },
        TraceType::Float,
    );
    b.finish(TraceTerminator::LoopBack { args: vec![i_next] })
}

#[test]
fn a_table_set_appends_and_leaves_before_the_store_the_vm_refuses() {
    let ir = set_case();
    ir.verify().unwrap();
    let mut model = Model::array(vec![f(9.0)]);
    let evaluation = run(&ir, &mut model, &[f(1.0), TraceScalar::Table(0)], 4);
    assert_eq!(
        evaluation.outcome,
        Outcome::IterationLimit { iterations: 4 }
    );
    // Key 1 overwrites the 9, keys 2 to 4 append.
    assert_eq!(model.arrays[0], vec![f(1.0), f(2.0), f(3.0), f(4.0)]);
    // The VM refuses the third key: the guard leaves before that store, and
    // the two before it stay committed, once.
    let mut model = Model::array(Vec::new());
    model.refused.push(f(3.0));
    let evaluation = run(&ir, &mut model, &[f(1.0), TraceScalar::Table(0)], 10);
    let Outcome::GuardFailed { guard, restored } = &evaluation.outcome else {
        panic!(
            "expected the refused set to leave: {:?}",
            evaluation.outcome
        )
    };
    assert_eq!(*guard, GuardId(0));
    assert_eq!(restored.pc, 1);
    assert_eq!(restored.registers[0], f(3.0));
    assert_eq!(
        stores(&evaluation),
        vec![vec![f(1.0), f(1.0)], vec![f(2.0), f(2.0)]]
    );
    assert_eq!(model.arrays[0], vec![f(1.0), f(2.0)]);
}

#[test]
fn a_table_set_moves_storage_no_field_or_lane_access_may_hold() {
    // A field binding beside the set: a resize could move the field's slot
    // if the set named that table, so entry refuses a run in which it does.
    // The set is the only operation that moves storage, so the trace holds.
    let mut ir = set_case();
    ir.bindings.push(TableBinding {
        register: 1,
        kind: BindingKind::Field {
            key: b"x".as_slice().into(),
            ty: TraceType::Float,
            depth: 0,
        },
    });
    ir.verify().unwrap();
    assert!(ir.sets_tables());
    assert!(ir.only_array_sets_move_storage());
    assert!(ir.set_moves_storage(0));
    assert!(ir.storage_a_set_would_move().is_none());
    // A length binding is another matter: the VM rewrites its word after a
    // set, and native code reads it again.
    let mut ir = set_case();
    ir.bindings.push(TableBinding {
        register: 1,
        kind: BindingKind::Length {
            ty: TraceType::Float,
            raw: false,
        },
    });
    assert!(ir.storage_a_set_would_move().is_none());
    // A set may store a table (the VM's store entry owns the write barrier).
    let mut ir = set_case();
    let table = ir.entry.inputs[1].value;
    if let TraceOp::ArraySet { value, .. } = &mut ir.body[0].op {
        *value = table;
    }
    assert!(!matches!(
        ir.verify(),
        Err(TraceVerifyError::TypeMismatch { instruction: 0, .. })
    ));
}

#[test]
fn a_nil_read_needs_a_table_without_a_metatable() {
    let mut ir = table_case();
    if let TraceOp::ArrayLoad { expected, .. } = &mut ir.body[0].op {
        *expected = TraceType::Nil;
    }
    let result = ir.body[0].result.unwrap();
    ir.values[result.index()].ty = TraceType::Nil;
    // Without the binding that keeps `__index` away, nil is no value of the
    // access. (The trace's later arithmetic on it is not what fails first.)
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::TypeMismatch { instruction: 0, .. })
    ));
    ir.bindings.push(TableBinding {
        register: 4,
        kind: BindingKind::NoMetatable,
    });
    // The read itself now verifies: the next failure is the addition of nil.
    assert!(!matches!(
        ir.verify(),
        Err(TraceVerifyError::TypeMismatch { instruction: 0, .. })
    ));
    // The reference reads a hole as nil, and leaves at a present value.
    let mut model = Model::array(vec![TraceScalar::Nil, f(2.0)]);
    assert!(model.array_hole(0, f(1.0)));
    assert!(!model.array_hole(0, f(2.0)));
    assert!(!model.array_hole(0, f(3.0)));
}

// ---- memory forwarding -----------------------------------------------------

/// Straight-line traces of table accesses for the forwarding pass. Registers:
/// the tables first, then the integer key inputs, then six output registers
/// that every snapshot restores from the values named with [`Mem::out`].
pub(super) struct Mem {
    b: TraceBuilder,
    pc: usize,
    outs: Vec<ValueId>,
    first_out: u16,
    registers: u16,
    pub(super) tables: Vec<ValueId>,
    pub(super) keys: Vec<ValueId>,
}

const OUTS: u16 = 6;

impl Mem {
    pub(super) fn new(tables: u16, keys: u16, bindings: Vec<TableBinding>) -> Self {
        let registers = tables + keys + OUTS;
        let mut b = TraceBuilder::new(
            TraceId(9),
            TraceFrameInfo {
                register_count: registers,
                code_len: 512,
                inlined: None,
                activation: None,
            },
        );
        for out in 0..OUTS {
            b.mark_written(FrameId(0), tables + keys + out);
        }
        let table_values = (0..tables).map(|r| b.input(r, TraceType::Table)).collect();
        let key_values = (0..keys)
            .map(|r| b.input(tables + r, TraceType::Integer))
            .collect();
        b.set_bindings(bindings);
        Self {
            b,
            pc: 0,
            outs: Vec::new(),
            first_out: tables + keys,
            registers,
            tables: table_values,
            keys: key_values,
        }
    }

    fn registers(&self) -> Vec<RegisterEntry> {
        (0..self.registers)
            .map(
                |register| match register.checked_sub(self.first_out).map(usize::from) {
                    None => RegisterEntry::Unchanged,
                    Some(out) => self
                        .outs
                        .get(out)
                        .map_or(RegisterEntry::Const(TraceScalar::Nil), |value| {
                            RegisterEntry::Value(*value)
                        }),
                },
            )
            .collect()
    }

    /// Makes `value` the next output register, which later snapshots and the
    /// final exit restore.
    pub(super) fn out(&mut self, value: ValueId) {
        self.outs.push(value);
    }

    fn guard(&mut self) -> GuardId {
        self.pc += 1;
        let snapshot = self.b.snapshot(snap(
            SnapshotPhase::Before { pc: self.pc },
            self.b.effect_position(),
            Some(self.pc - 1),
            self.registers(),
        ));
        self.b.guard(site(self.pc), snapshot)
    }

    pub(super) fn konst(&mut self, scalar: TraceScalar) -> ValueId {
        self.pc += 1;
        emit(&mut self.b, self.pc, TraceOp::Constant(scalar), scalar.ty())
    }

    pub(super) fn load(
        &mut self,
        binding: u32,
        table: ValueId,
        key: ValueId,
        expected: TraceType,
    ) -> ValueId {
        let guard = self.guard();
        emit(
            &mut self.b,
            self.pc,
            TraceOp::ArrayLoad {
                binding,
                table,
                key,
                expected,
                past_end: false,
                guard,
            },
            expected,
        )
    }

    pub(super) fn store(&mut self, binding: u32, table: ValueId, key: ValueId, value: ValueId) {
        let guard = self.guard();
        let effect = self.b.next_effect();
        self.b.emit(
            site(self.pc),
            TraceOp::ArrayStore {
                binding,
                table,
                key,
                value,
                effect,
                guard,
            },
            None,
        );
    }

    pub(super) fn set(&mut self, binding: u32, table: ValueId, key: ValueId, value: ValueId) {
        let guard = self.guard();
        let effect = self.b.next_effect();
        self.b.emit(
            site(self.pc),
            TraceOp::ArraySet {
                binding,
                table,
                key,
                value,
                effect,
                guard,
            },
            None,
        );
    }

    /// Calls `helper` on `args`. Returns its first result.
    pub(super) fn helper(&mut self, helper: HelperId, args: &[ValueId]) -> Option<ValueId> {
        let guard = self.guard();
        let info = helper.info().unwrap();
        let effect = info.effects.commits_effect().then(|| self.b.next_effect());
        self.b.emit(
            site(self.pc),
            TraceOp::CallHelper {
                helper,
                args: args.to_vec(),
                effect,
                guard,
            },
            info.results.first().map(|result| result.trace_type()),
        )
    }

    /// Reads the integer field `n` of the table `table` from its lane's row.
    fn lane_field(&mut self, table: ValueId) -> ValueId {
        self.b.set_fields(vec![TableField {
            key: b"n"[..].into(),
            ty: TraceType::Integer,
        }]);
        let guard = self.guard();
        emit(
            &mut self.b,
            self.pc,
            TraceOp::TableFieldLoad {
                field: 0,
                table,
                guard,
            },
            TraceType::Integer,
        )
    }

    /// Guards that the metatable of `table` is the class `binding` pins.
    pub(super) fn meta_guard(&mut self, binding: u32, table: ValueId) {
        let guard = self.guard();
        self.b.emit(
            site(self.pc),
            TraceOp::MetaGuard {
                table,
                binding,
                guard,
            },
            None,
        );
    }

    pub(super) fn field_load(&mut self, binding: u32, table: ValueId, ty: TraceType) -> ValueId {
        self.pc += 1;
        emit(
            &mut self.b,
            self.pc,
            TraceOp::FieldLoad { binding, table },
            ty,
        )
    }

    pub(super) fn field_store(&mut self, binding: u32, table: ValueId, value: ValueId) {
        self.pc += 1;
        let effect = self.b.next_effect();
        self.b.emit(
            site(self.pc),
            TraceOp::FieldStore {
                binding,
                table,
                value,
                effect,
            },
            None,
        );
    }

    /// `0.0 < value` as a branch guard that must hold. Its snapshot resumes
    /// after the branch and restores the outputs named so far.
    pub(super) fn branch_if_positive(&mut self, value: ValueId) {
        let zero = self.konst(f(0.0));
        self.pc += 1;
        let cond = emit(
            &mut self.b,
            self.pc,
            TraceOp::Compare {
                comparison: Comparison::LessThan,
                left: zero,
                right: value,
            },
            TraceType::Boolean,
        );
        self.pc += 1;
        let snapshot = self.b.snapshot(snap(
            SnapshotPhase::After {
                pc: self.pc,
                successor: self.pc + 1,
            },
            self.b.effect_position(),
            None,
            self.registers(),
        ));
        let guard = self.b.guard(site(self.pc), snapshot);
        self.b.emit(
            site(self.pc),
            TraceOp::GuardBranch {
                condition: cond,
                expected: true,
                guard,
            },
            None,
        );
    }

    pub(super) fn finish(self) -> TraceIr {
        let ir = self.finish_unverified();
        ir.verify().unwrap();
        ir
    }

    /// Closes the trace without verifying it, for a test of what the
    /// verifier refuses.
    pub(super) fn finish_unverified(mut self) -> TraceIr {
        let exit = self.b.snapshot(snap(
            SnapshotPhase::Before { pc: 400 },
            self.b.effect_position(),
            None,
            self.registers(),
        ));
        self.b.finish(TraceTerminator::Exit { snapshot: exit })
    }
}

pub(super) fn array_binding(register: u16) -> TableBinding {
    TableBinding {
        register,
        kind: BindingKind::Array { first: None },
    }
}

pub(super) fn field_binding(register: u16, key: &[u8]) -> TableBinding {
    TableBinding {
        register,
        kind: BindingKind::Field {
            key: key.into(),
            ty: TraceType::Float,
            depth: 0,
        },
    }
}

pub(super) fn memory_model(
    arrays: Vec<Vec<TraceScalar>>,
    fields: Vec<TraceScalar>,
    alias: Vec<usize>,
) -> Model {
    Model {
        arrays,
        fields,
        lengths: Vec::new(),
        rows: Vec::new(),
        accept: true,
        refused: Vec::new(),
        alias,
    }
}

fn memory_on() -> OptimizeOptions {
    OptimizeOptions::for_level(OptimizeLevel::Optimized)
}

fn memory_off() -> OptimizeOptions {
    OptimizeOptions {
        memory: false,
        ..memory_on()
    }
}

fn loads(ir: &TraceIr) -> usize {
    ir.body
        .iter()
        .filter(|i| matches!(i.op, TraceOp::ArrayLoad { .. } | TraceOp::FieldLoad { .. }))
        .count()
}

fn stores_in(ir: &TraceIr) -> usize {
    ir.body
        .iter()
        .filter(|i| i.op.committed_effect().is_some())
        .count()
}

/// Optimizes `ir` with the memory pass and checks it against the reference
/// evaluator: same outcome, same effects and same final heap, with no forced
/// exit and with each guard the optimized trace kept forced to fail.
fn check_forwarding(
    ir: &TraceIr,
    heaps: &[Model],
    registers: &[TraceScalar],
) -> (TraceIr, OptimizeStats) {
    let (optimized, stats) = optimize_with(ir, memory_on());
    optimized.verify().unwrap();
    let mut forced_exits: Vec<Option<ForcedExit>> = vec![None];
    for guard in &optimized.guards {
        forced_exits.push(Some(ForcedExit {
            frame: guard.site.frame,
            pc: guard.site.pc,
            guard: None,
            iteration: 0,
        }));
    }
    for heap in heaps {
        for forced in &forced_exits {
            let mut original_heap = heap.clone();
            let mut optimized_heap = heap.clone();
            let before = run_forced(ir, &mut original_heap, registers, 1, *forced);
            let after = run_forced(&optimized, &mut optimized_heap, registers, 1, *forced);
            // Guard ids are renumbered when loads and their guards go, so
            // compare the restored state, which names the guard's site.
            let strip = |outcome: &Outcome| match outcome {
                Outcome::GuardFailed { restored, .. } => Outcome::GuardFailed {
                    guard: GuardId(0),
                    restored: restored.clone(),
                },
                other => other.clone(),
            };
            assert_eq!(
                strip(&before.outcome),
                strip(&after.outcome),
                "{forced:?} {heap:?}"
            );
            assert_eq!(before.effects, after.effects, "{forced:?} {heap:?}");
            assert_eq!(original_heap.arrays, optimized_heap.arrays, "{forced:?}");
            assert_eq!(original_heap.fields, optimized_heap.fields, "{forced:?}");
        }
    }
    (optimized, stats)
}

fn floats(values: &[f64]) -> Vec<TraceScalar> {
    values.iter().map(|value| f(*value)).collect()
}

fn one_table_regs(extra: &[TraceScalar]) -> Vec<TraceScalar> {
    let mut registers = vec![TraceScalar::Table(0)];
    registers.extend_from_slice(extra);
    registers.resize(1 + extra.len() + usize::from(OUTS), TraceScalar::Nil);
    registers
}

/// The mineclonia pattern: two stores to constant keys, then both reloaded.
pub(super) fn mineclonia_case() -> TraceIr {
    let mut m = Mem::new(1, 0, vec![array_binding(0)]);
    let t = m.tables[0];
    let (k1, k2) = (m.konst(int(1)), m.konst(int(2)));
    let (v1, v2) = (m.konst(f(5.0)), m.konst(f(6.0)));
    m.store(0, t, k1, v1);
    m.store(0, t, k2, v2);
    let a = m.load(0, t, k1, TraceType::Float);
    m.out(a);
    let b = m.load(0, t, k2, TraceType::Float);
    m.out(b);
    m.finish()
}

#[test]
fn a_store_answers_a_later_load_of_its_key_and_both_stores_stay() {
    let ir = mineclonia_case();
    let heaps = [
        memory_model(vec![floats(&[1.0, 2.0])], vec![], vec![]),
        memory_model(vec![floats(&[1.0])], vec![], vec![]),
    ];
    let (optimized, stats) = check_forwarding(&ir, &heaps, &one_table_regs(&[]));
    assert_eq!(loads(&optimized), 0);
    assert_eq!(stores_in(&optimized), 2);
    assert_eq!(stats.loads_forwarded, 2);
    assert_eq!(stats.loads_merged, 0);
    // The loads' guards go with them, the stores' guards stay.
    assert_eq!(optimized.guards.len(), 2);
}

#[test]
fn a_repeated_load_merges_across_a_store_to_another_constant_key() {
    let mut m = Mem::new(1, 0, vec![array_binding(0)]);
    let t = m.tables[0];
    let (k1, k2) = (m.konst(int(1)), m.konst(int(2)));
    let v = m.konst(f(5.0));
    let first = m.load(0, t, k1, TraceType::Float);
    m.out(first);
    m.store(0, t, k2, v);
    let second = m.load(0, t, k1, TraceType::Float);
    m.out(second);
    let ir = m.finish();
    let heaps = [memory_model(vec![floats(&[1.0, 2.0])], vec![], vec![])];
    let (optimized, stats) = check_forwarding(&ir, &heaps, &one_table_regs(&[]));
    assert_eq!(loads(&optimized), 1);
    assert_eq!(stores_in(&optimized), 1);
    assert_eq!((stats.loads_forwarded, stats.loads_merged), (0, 1));
}

#[test]
fn a_store_to_a_key_that_may_be_equal_kills_the_fact() {
    let mut m = Mem::new(1, 1, vec![array_binding(0)]);
    let (t, k) = (m.tables[0], m.keys[0]);
    let k1 = m.konst(int(1));
    let (v, w) = (m.konst(f(5.0)), m.konst(f(6.0)));
    m.store(0, t, k1, v);
    m.store(0, t, k, w);
    let reloaded = m.load(0, t, k1, TraceType::Float);
    m.out(reloaded);
    let ir = m.finish();
    let heaps = [memory_model(vec![floats(&[1.0, 2.0])], vec![], vec![])];
    for key in [1, 2] {
        let (optimized, stats) = check_forwarding(&ir, &heaps, &one_table_regs(&[int(key)]));
        assert_eq!(loads(&optimized), 1, "key {key}");
        assert_eq!(stats.loads_forwarded, 0);
    }
}

/// Stores 5.0 to `first`, stores 6.0 to `second`, then reloads `first`.
/// Returns the optimized trace and its counters, after checking equivalence.
fn two_key_case(first: TraceScalar, second: TraceScalar) -> (TraceIr, OptimizeStats) {
    let mut m = Mem::new(1, 0, vec![array_binding(0)]);
    let t = m.tables[0];
    let (ka, kb) = (m.konst(first), m.konst(second));
    let (v, w) = (m.konst(f(5.0)), m.konst(f(6.0)));
    m.store(0, t, ka, v);
    m.store(0, t, kb, w);
    let reloaded = m.load(0, t, ka, TraceType::Float);
    m.out(reloaded);
    let ir = m.finish();
    let heaps = [
        memory_model(vec![floats(&[1.0, 2.0, 3.0])], vec![], vec![]),
        memory_model(vec![floats(&[1.0])], vec![], vec![]),
    ];
    check_forwarding(&ir, &heaps, &one_table_regs(&[]))
}

#[test]
fn a_store_to_the_same_number_as_float_and_integer_answers_the_load() {
    // a[1] = 5; a[1.0] = 6; reload a[1]: one key, so the last store answers.
    let (optimized, stats) = two_key_case(int(1), f(1.0));
    assert_eq!(loads(&optimized), 0);
    assert_eq!(stats.loads_forwarded, 1);
    // And the other way round.
    let (optimized, stats) = two_key_case(f(1.0), int(1));
    assert_eq!(loads(&optimized), 0);
    assert_eq!(stats.loads_forwarded, 1);
}

#[test]
fn a_float_key_store_keeps_the_fact_of_another_float_key() {
    let (optimized, stats) = two_key_case(f(1.0), f(2.0));
    assert_eq!(loads(&optimized), 0);
    assert_eq!(stats.loads_forwarded, 1);
}

#[test]
fn an_integer_key_store_keeps_the_fact_of_a_float_key() {
    let (optimized, stats) = two_key_case(f(1.0), int(2));
    assert_eq!(loads(&optimized), 0);
    assert_eq!(stats.loads_forwarded, 1);
    let (optimized, stats) = two_key_case(int(1), f(2.0));
    assert_eq!(loads(&optimized), 0);
    assert_eq!(stats.loads_forwarded, 1);
}

#[test]
fn a_fractional_float_key_differs_from_an_integer_key() {
    let (optimized, stats) = two_key_case(int(1), f(1.5));
    assert_eq!(loads(&optimized), 0);
    assert_eq!(stats.loads_forwarded, 1);
}

/// Stores 5.0 to `store_key`, then loads `load_key`. The load is answered
/// only when the optimizer judges both constants to be one key, so a
/// regression in key identity leaves the load in place and fails the caller.
fn assert_one_key(store_key: TraceScalar, load_key: TraceScalar) {
    let mut m = Mem::new(1, 0, vec![array_binding(0)]);
    let t = m.tables[0];
    let (ks, kl) = (m.konst(store_key), m.konst(load_key));
    let v = m.konst(f(5.0));
    m.store(0, t, ks, v);
    let a = m.load(0, t, kl, TraceType::Float);
    m.out(a);
    let ir = m.finish();
    let heaps = [memory_model(vec![floats(&[1.0, 2.0])], vec![], vec![])];
    let (optimized, stats) = check_forwarding(&ir, &heaps, &one_table_regs(&[]));
    assert_eq!(loads(&optimized), 0, "{store_key:?} {load_key:?}");
    assert_eq!(stats.loads_forwarded, 1, "{store_key:?} {load_key:?}");
}

#[test]
fn negative_zero_and_zero_are_one_key() {
    // Key 0 is outside the model array, so the evaluator cannot tell the
    // difference; the counters can.
    assert_one_key(f(0.0), f(-0.0));
    assert_one_key(f(-0.0), f(0.0));
    assert_one_key(int(0), f(-0.0));
    assert_one_key(f(-0.0), int(0));
}

#[test]
fn a_nan_key_store_kills_the_fact() {
    let (optimized, stats) = two_key_case(int(1), f(f64::NAN));
    assert_eq!(loads(&optimized), 1);
    assert_eq!(stats.loads_forwarded, 0);
}

#[test]
fn integers_beyond_two_to_the_53_compare_exactly() {
    let big = 1_i64 << 53;
    // 2^53 + 1 is not 2^53 although the cast to f64 rounds to it.
    let (optimized, stats) = two_key_case(f(big as f64), int(big + 1));
    assert_eq!(loads(&optimized), 0);
    assert_eq!(stats.loads_forwarded, 1);
    let (optimized, stats) = two_key_case(int(big + 1), f(big as f64));
    assert_eq!(loads(&optimized), 0);
    assert_eq!(stats.loads_forwarded, 1);
    // Equal values still share a key. The keys are outside the model array,
    // so only the counters of a store followed by a load show it.
    assert_one_key(int(big), f(big as f64));
    assert_one_key(f(big as f64), int(big));
    assert_one_key(int(i64::MIN), f(-9_223_372_036_854_775_808.0));
    assert_one_key(f(-9_223_372_036_854_775_808.0), int(i64::MIN));
    // 2^63 is not i64::MAX.
    let (optimized, stats) = two_key_case(int(i64::MAX), f(9_223_372_036_854_775_808.0));
    assert_eq!(loads(&optimized), 0);
    assert_eq!(stats.loads_forwarded, 1);
}

#[test]
fn a_float_key_load_after_an_integer_key_store_hits() {
    let mut m = Mem::new(1, 0, vec![array_binding(0)]);
    let t = m.tables[0];
    let (k, kf) = (m.konst(int(1)), m.konst(f(1.0)));
    let v = m.konst(f(5.0));
    m.store(0, t, k, v);
    let a = m.load(0, t, kf, TraceType::Float);
    m.out(a);
    let ir = m.finish();
    let heaps = [memory_model(vec![floats(&[1.0, 2.0])], vec![], vec![])];
    let (optimized, stats) = check_forwarding(&ir, &heaps, &one_table_regs(&[]));
    assert_eq!(loads(&optimized), 0);
    assert_eq!(stats.loads_forwarded, 1);
}

/// The lua51 mineclonia pattern: the keys are float constants.
#[test]
fn float_keyed_mineclonia_pattern_forwards_both_reloads() {
    let mut m = Mem::new(1, 0, vec![array_binding(0)]);
    let t = m.tables[0];
    let (k1, k2) = (m.konst(f(1.0)), m.konst(f(2.0)));
    let (v1, v2) = (m.konst(f(5.0)), m.konst(f(6.0)));
    m.store(0, t, k1, v1);
    m.store(0, t, k2, v2);
    let a = m.load(0, t, k1, TraceType::Float);
    m.out(a);
    let b = m.load(0, t, k2, TraceType::Float);
    m.out(b);
    let ir = m.finish();
    let heaps = [
        memory_model(vec![floats(&[1.0, 2.0])], vec![], vec![]),
        memory_model(vec![floats(&[1.0])], vec![], vec![]),
    ];
    let (optimized, stats) = check_forwarding(&ir, &heaps, &one_table_regs(&[]));
    assert_eq!(loads(&optimized), 0);
    assert_eq!(stores_in(&optimized), 2);
    assert_eq!(stats.loads_forwarded, 2);
}

#[test]
fn two_table_values_that_name_one_table_never_share_a_fact() {
    // Registers 0 and 1 hold the same table: the model gives both bindings
    // one storage.
    let mut m = Mem::new(2, 0, vec![array_binding(0), array_binding(1)]);
    let (t0, t1) = (m.tables[0], m.tables[1]);
    let k = m.konst(int(1));
    let (v, w) = (m.konst(f(5.0)), m.konst(f(6.0)));
    m.store(0, t0, k, v);
    // Reads the slot the store through the other value wrote.
    let through_alias = m.load(1, t1, k, TraceType::Float);
    m.out(through_alias);
    // Answered by the store through t0 ...
    let same = m.load(0, t0, k, TraceType::Float);
    m.out(same);
    // ... until the alias overwrites the slot.
    m.store(1, t1, k, w);
    let stale = m.load(0, t0, k, TraceType::Float);
    m.out(stale);
    let ir = m.finish();
    let heaps = [memory_model(
        vec![floats(&[1.0, 2.0]), vec![]],
        vec![],
        vec![0, 0],
    )];
    let registers = [
        TraceScalar::Table(0),
        TraceScalar::Table(0),
        TraceScalar::Nil,
        TraceScalar::Nil,
        TraceScalar::Nil,
        TraceScalar::Nil,
        TraceScalar::Nil,
        TraceScalar::Nil,
    ];
    let (optimized, stats) = check_forwarding(&ir, &heaps, &registers);
    assert_eq!(
        loads(&optimized),
        2,
        "the alias read and the stale read stay"
    );
    assert_eq!(stats.loads_forwarded, 1);
    let mut model = heaps[0].clone();
    let evaluation = run(&optimized, &mut model, &registers, 1);
    let Outcome::Exited(restored) = evaluation.outcome else {
        panic!("expected an exit")
    };
    assert_eq!(restored.registers[2..5], [f(5.0), f(5.0), f(6.0)]);
}

#[test]
fn two_bindings_that_reach_one_field_never_share_a_fact() {
    // Two registers reach field x through one __index holder: one storage.
    let mut m = Mem::new(2, 0, vec![field_binding(0, b"x"), field_binding(1, b"x")]);
    let (t0, t1) = (m.tables[0], m.tables[1]);
    let v = m.konst(f(7.0));
    let before = m.field_load(0, t0, TraceType::Float);
    m.out(before);
    m.field_store(1, t1, v);
    let after = m.field_load(0, t0, TraceType::Float);
    m.out(after);
    // The store through t1 dropped the binding 0 fact, so `after` reloads.
    // The load through t1 itself still has the stored value, so it is
    // forwarded.
    let own = m.field_load(1, t1, TraceType::Float);
    m.out(own);
    let ir = m.finish();
    let heaps = [memory_model(vec![], floats(&[1.0, 1.0]), vec![0, 0])];
    let registers = [
        TraceScalar::Table(0),
        TraceScalar::Table(0),
        TraceScalar::Nil,
        TraceScalar::Nil,
        TraceScalar::Nil,
        TraceScalar::Nil,
        TraceScalar::Nil,
        TraceScalar::Nil,
    ];
    let (optimized, stats) = check_forwarding(&ir, &heaps, &registers);
    assert_eq!(
        loads(&optimized),
        2,
        "the first and the post-store load stay"
    );
    assert_eq!(stats.loads_forwarded, 1);
    let mut model = heaps[0].clone();
    let evaluation = run(&optimized, &mut model, &registers, 1);
    let Outcome::Exited(restored) = evaluation.outcome else {
        panic!("expected an exit")
    };
    assert_eq!(restored.registers[2..5], [f(1.0), f(7.0), f(7.0)]);
}

#[test]
fn a_field_store_forwards_to_its_own_field_and_keeps_other_fields_facts() {
    let mut m = Mem::new(
        1,
        0,
        vec![
            field_binding(0, b"x"),
            field_binding(0, b"y"),
            array_binding(0),
        ],
    );
    let t = m.tables[0];
    let k1 = m.konst(int(1));
    let v = m.konst(f(7.0));
    let x1 = m.field_load(0, t, TraceType::Float);
    m.out(x1);
    let a1 = m.load(2, t, k1, TraceType::Float);
    m.out(a1);
    m.field_store(1, t, v);
    let x2 = m.field_load(0, t, TraceType::Float);
    m.out(x2);
    let a2 = m.load(2, t, k1, TraceType::Float);
    m.out(a2);
    let y = m.field_load(1, t, TraceType::Float);
    m.out(y);
    let ir = m.finish();
    let heaps = [memory_model(
        vec![vec![], vec![], floats(&[4.0, 5.0])],
        floats(&[1.0, 2.0, 0.0]),
        vec![],
    )];
    let (optimized, stats) = check_forwarding(&ir, &heaps, &one_table_regs(&[]));
    // x merged across the store of y, y forwarded from its store, the array
    // load repeats because a field store drops every array fact.
    let field_loads = optimized
        .body
        .iter()
        .filter(|i| matches!(i.op, TraceOp::FieldLoad { .. }))
        .count();
    assert_eq!(field_loads, 1);
    assert_eq!(loads(&optimized) - field_loads, 2);
    assert_eq!((stats.loads_forwarded, stats.loads_merged), (1, 1));
}

#[test]
fn an_array_set_forgets_every_fact() {
    let mut m = Mem::new(1, 0, vec![array_binding(0)]);
    let t = m.tables[0];
    let (k1, k2) = (m.konst(int(1)), m.konst(int(2)));
    let (v, w) = (m.konst(f(5.0)), m.konst(f(6.0)));
    m.store(0, t, k1, v);
    m.set(0, t, k2, w);
    let reloaded = m.load(0, t, k1, TraceType::Float);
    m.out(reloaded);
    let ir = m.finish();
    let heaps = [
        memory_model(vec![floats(&[1.0, 2.0])], vec![], vec![]),
        memory_model(vec![floats(&[1.0])], vec![], vec![]),
    ];
    let (optimized, stats) = check_forwarding(&ir, &heaps, &one_table_regs(&[]));
    assert_eq!(loads(&optimized), 1);
    assert_eq!(stats.loads_forwarded, 0);
}

#[test]
fn an_effect_forgets_every_fact() {
    let mut m = Mem::new(1, 0, vec![array_binding(0)]);
    let t = m.tables[0];
    let k1 = m.konst(int(1));
    let v = m.konst(f(5.0));
    m.store(0, t, k1, v);
    m.pc += 1;
    let effect = m.b.next_effect();
    m.b.emit(
        site(m.pc),
        TraceOp::Effect {
            effect,
            label: 1,
            operands: vec![],
            result: None,
        },
        None,
    );
    let reloaded = m.load(0, t, k1, TraceType::Float);
    m.out(reloaded);
    let ir = m.finish();
    let (optimized, stats) = optimize_with(&ir, memory_on());
    assert_eq!(loads(&optimized), 1);
    assert_eq!(stats.loads_forwarded, 0);
}

#[test]
fn a_store_never_answers_a_load_of_another_type_or_of_nil_or_table() {
    // The slot holds an integer; the store writes an integer.
    let mut m = Mem::new(1, 0, vec![array_binding(0)]);
    let t = m.tables[0];
    let k1 = m.konst(int(1));
    let v = m.konst(int(9));
    m.store(0, t, k1, v);
    let same = m.load(0, t, k1, TraceType::Integer);
    m.out(same);
    let ir = m.finish();
    let heaps = [memory_model(vec![vec![int(1)]], vec![], vec![])];
    let (optimized, stats) = check_forwarding(&ir, &heaps, &one_table_regs(&[]));
    assert_eq!((loads(&optimized), stats.loads_forwarded), (0, 1));

    // A load that expects another type stays, and fails its own guard.
    let mut m = Mem::new(1, 0, vec![array_binding(0)]);
    let t = m.tables[0];
    let k1 = m.konst(int(1));
    let v = m.konst(f(9.0));
    m.store(0, t, k1, v);
    let other = m.load(0, t, k1, TraceType::Integer);
    m.out(other);
    let ir = m.finish();
    let heaps = [memory_model(vec![floats(&[1.0])], vec![], vec![])];
    let (optimized, stats) = check_forwarding(&ir, &heaps, &one_table_regs(&[]));
    assert_eq!((loads(&optimized), stats.loads_forwarded), (1, 0));

    // A load that expects nil stays.
    let mut m = Mem::new(
        1,
        0,
        vec![
            array_binding(0),
            TableBinding {
                register: 0,
                kind: BindingKind::NoMetatable,
            },
        ],
    );
    let t = m.tables[0];
    let k1 = m.konst(int(1));
    let v = m.konst(f(9.0));
    m.store(0, t, k1, v);
    let nil = m.load(0, t, k1, TraceType::Nil);
    m.out(nil);
    let ir = m.finish();
    let heaps = [memory_model(vec![floats(&[1.0]), vec![]], vec![], vec![])];
    let (optimized, stats) = check_forwarding(&ir, &heaps, &one_table_regs(&[]));
    assert_eq!((loads(&optimized), stats.loads_forwarded), (1, 0));
    // The verifier refuses to store a table into an array slot, so a table
    // load has no store to be answered by.
}

#[test]
fn a_guard_exit_after_a_committed_store_restores_the_forwarded_value() {
    for stored in [3.0, -3.0] {
        let mut m = Mem::new(1, 0, vec![array_binding(0)]);
        let t = m.tables[0];
        let k1 = m.konst(int(1));
        let v = m.konst(f(stored));
        m.store(0, t, k1, v);
        let loaded = m.load(0, t, k1, TraceType::Float);
        m.out(loaded);
        // Fails when the stored value is not positive; it resumes after the
        // branch, the store committed and restored from the forwarded value.
        m.branch_if_positive(loaded);
        let ir = m.finish();
        let heaps = [memory_model(vec![floats(&[1.0, 2.0])], vec![], vec![])];
        let (optimized, stats) = check_forwarding(&ir, &heaps, &one_table_regs(&[]));
        assert_eq!(stats.loads_forwarded, 1);
        assert_eq!(loads(&optimized), 0);
        let mut model = heaps[0].clone();
        let evaluation = run(&optimized, &mut model, &one_table_regs(&[]), 1);
        assert_eq!(evaluation.effects.len(), 1, "the store commits once");
        if stored < 0.0 {
            let Outcome::GuardFailed { restored, .. } = evaluation.outcome else {
                panic!("the branch guard fails")
            };
            assert_eq!(restored.registers[1], f(stored));
            assert_eq!(restored.effect_position, 1);
        }
    }
}

#[test]
fn facts_do_not_cross_the_loop_back_edge() {
    // The table case loads before it stores: nothing to forward, and the
    // load of the next iteration is never answered by the store before it.
    let ir = table_case();
    let (optimized, stats) = optimize_with(&ir, memory_on());
    assert_eq!(stats.loads_forwarded + stats.loads_merged, 0);
    assert_eq!(
        optimized
            .body
            .iter()
            .filter(|i| matches!(i.op, TraceOp::ArrayLoad { .. }))
            .count(),
        1
    );
}

#[test]
fn the_memory_flag_off_keeps_every_load() {
    let ir = mineclonia_case();
    let (optimized, stats) = optimize_with(&ir, memory_off());
    assert_eq!(loads(&optimized), 2);
    assert_eq!((stats.loads_forwarded, stats.loads_merged), (0, 0));
}

// Hoisting of table reads out of a loop body, proved by the tree's writes.

/// A loop over one table at register 1 with a float accumulator in register
/// 0. Bindings: 0 the field `x`, 1 the field `y`, 2 the array.
struct Lp {
    b: TraceBuilder,
    pc: usize,
    table: ValueId,
    acc: ValueId,
    sum: Option<ValueId>,
    /// The index of the array binding.
    array: u32,
}

impl Lp {
    fn new() -> Self {
        Self::with(
            vec![
                field_binding(1, b"x"),
                field_binding(1, b"y"),
                array_binding(1),
            ],
            2,
        )
    }

    /// A loop with the array binding only, which a table set may share a
    /// trace with.
    fn arrays_only() -> Self {
        Self::with(vec![array_binding(1)], 0)
    }

    fn with(bindings: Vec<TableBinding>, array: u32) -> Self {
        let mut b = TraceBuilder::new(TraceId(7), root(2));
        b.mark_written(FrameId(0), 0);
        let acc_in = b.input(0, TraceType::Float);
        let table = b.input(1, TraceType::Table);
        b.set_bindings(bindings);
        let acc = b.loop_param(TraceType::Float, LoopParamInit::Input(acc_in));
        Self {
            b,
            pc: 0,
            table,
            acc,
            sum: None,
            array,
        }
    }

    fn registers(&self) -> Vec<RegisterEntry> {
        vec![RegisterEntry::Value(self.acc), RegisterEntry::Unchanged]
    }

    fn konst(&mut self, scalar: TraceScalar) -> ValueId {
        self.pc += 1;
        emit(&mut self.b, self.pc, TraceOp::Constant(scalar), scalar.ty())
    }

    fn guard(&mut self) -> GuardId {
        self.pc += 1;
        let snapshot = self.b.snapshot(snap(
            SnapshotPhase::Before { pc: self.pc },
            self.b.effect_position(),
            Some(self.pc - 1),
            self.registers(),
        ));
        self.b.guard(site(self.pc), snapshot)
    }

    fn field_load(&mut self, binding: u32) -> ValueId {
        self.pc += 1;
        let table = self.table;
        emit(
            &mut self.b,
            self.pc,
            TraceOp::FieldLoad { binding, table },
            TraceType::Float,
        )
    }

    fn field_store(&mut self, binding: u32, value: ValueId) {
        self.pc += 1;
        let effect = self.b.next_effect();
        let table = self.table;
        self.b.emit(
            site(self.pc),
            TraceOp::FieldStore {
                binding,
                table,
                value,
                effect,
            },
            None,
        );
    }

    fn array_load(&mut self, key: ValueId) -> ValueId {
        let guard = self.guard();
        let table = self.table;
        emit(
            &mut self.b,
            self.pc,
            TraceOp::ArrayLoad {
                binding: self.array,
                table,
                key,
                expected: TraceType::Float,
                past_end: false,
                guard,
            },
            TraceType::Float,
        )
    }

    fn array_write(&mut self, key: ValueId, value: ValueId, set: bool) {
        let guard = self.guard();
        let effect = self.b.next_effect();
        let table = self.table;
        let binding = self.array;
        let op = if set {
            TraceOp::ArraySet {
                binding,
                table,
                key,
                value,
                effect,
                guard,
            }
        } else {
            TraceOp::ArrayStore {
                binding,
                table,
                key,
                value,
                effect,
                guard,
            }
        };
        self.b.emit(site(self.pc), op, None);
    }

    /// Calls `helper` on `args`. Returns its first result.
    fn helper(&mut self, helper: HelperId, args: &[ValueId]) -> Option<ValueId> {
        let guard = self.guard();
        let info = helper.info().unwrap();
        let effect = info.effects.commits_effect().then(|| self.b.next_effect());
        self.b.emit(
            site(self.pc),
            TraceOp::CallHelper {
                helper,
                args: args.to_vec(),
                effect,
                guard,
            },
            info.results.first().map(|result| result.trace_type()),
        )
    }

    fn effect(&mut self) {
        self.pc += 1;
        let effect = self.b.next_effect();
        self.b.emit(
            site(self.pc),
            TraceOp::Effect {
                effect,
                label: 1,
                operands: vec![],
                result: None,
            },
            None,
        );
    }

    /// Adds `value` to the running sum the loop hands back.
    fn add(&mut self, value: ValueId) {
        self.pc += 1;
        let left = self.sum.unwrap_or(self.acc);
        self.sum = Some(emit(
            &mut self.b,
            self.pc,
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left,
                right: value,
            },
            TraceType::Float,
        ));
    }

    fn finish(mut self) -> TraceIr {
        let next = self.sum.unwrap_or(self.acc);
        self.pc += 1;
        let poll = self.b.snapshot(snap(
            SnapshotPhase::After {
                pc: self.pc,
                successor: 1,
            },
            self.b.effect_position(),
            Some(self.pc),
            vec![RegisterEntry::Value(next), RegisterEntry::Unchanged],
        ));
        self.b.set_poll_snapshot(poll);
        let ir = self
            .b
            .finish(TraceTerminator::LoopBack { args: vec![next] });
        ir.verify().unwrap();
        ir
    }
}

/// A one-trace tree member that stores field `key` (binding 0) once.
fn storing_member(key: &[u8]) -> TraceIr {
    let mut m = Mem::new(1, 0, vec![field_binding(0, key)]);
    let t = m.tables[0];
    let v = m.konst(f(1.0));
    m.field_store(0, t, v);
    m.finish()
}

fn loop_model(array: &[f64]) -> Model {
    memory_model(
        vec![vec![], vec![], floats(array)],
        floats(&[2.0, 3.0]),
        vec![],
    )
}

fn loop_regs() -> [TraceScalar; 2] {
    [f(1.0), TraceScalar::Table(0)]
}

fn preheader_reads(ir: &TraceIr) -> usize {
    ir.preheader.iter().filter(|i| i.op.touches_heap()).count()
}

/// Runs the original and the optimized loop for three iterations with the
/// same heap and checks they end in the same state.
fn assert_same_loop(ir: &TraceIr, optimized: &TraceIr, array: &[f64]) {
    optimized.verify().unwrap();
    let mut before_heap = loop_model(array);
    let mut after_heap = loop_model(array);
    let before = run(ir, &mut before_heap, &loop_regs(), 3);
    let after = run(optimized, &mut after_heap, &loop_regs(), 3);
    assert_eq!(before.outcome, after.outcome);
    assert_eq!(before.effects, after.effects);
    assert_eq!(before_heap.arrays, after_heap.arrays);
    assert_eq!(before_heap.fields, after_heap.fields);
}

fn field_loop(stored: Option<u32>) -> TraceIr {
    let mut l = Lp::new();
    let x = l.field_load(0);
    l.add(x);
    if let Some(binding) = stored {
        let one = l.konst(f(1.0));
        l.field_store(binding, one);
    }
    l.finish()
}

#[test]
fn a_field_load_nothing_stores_moves_into_the_preheader() {
    let ir = field_loop(None);
    let (optimized, stats) = optimize_with(&ir, memory_on());
    assert_eq!(stats.loads_hoisted, 1);
    assert_eq!(preheader_reads(&optimized), 1);
    assert_eq!(loads(&optimized), 0);
    assert!(matches!(
        optimized.preheader[0].op,
        TraceOp::FieldLoad { .. }
    ));
    assert_same_loop(&ir, &optimized, &[]);
    // The same trace with the flag off keeps the old IR.
    let (old, stats) = optimize_with(&ir, memory_off());
    assert_eq!(stats.loads_hoisted, 0);
    assert_eq!(
        old.preheader,
        [] as [crate::jit_ir::trace::TraceInstruction; 0]
    );
    assert_eq!(loads(&old), 1);
    assert_eq!(old, super::optimize::tests::pipeline_without_memory(&ir));
}

#[test]
fn a_field_load_stays_when_the_root_stores_the_same_key() {
    let ir = field_loop(Some(0));
    let (optimized, stats) = optimize_with(&ir, memory_on());
    assert_eq!(stats.loads_hoisted, 0);
    assert_eq!(preheader_reads(&optimized), 0);
    assert_eq!(loads(&optimized), 1);
    assert_same_loop(&ir, &optimized, &[]);
}

#[test]
fn a_store_of_another_key_does_not_block_a_field_load() {
    let ir = field_loop(Some(1));
    let (optimized, stats) = optimize_with(&ir, memory_on());
    assert_eq!(stats.loads_hoisted, 1);
    assert_eq!(preheader_reads(&optimized), 1);
    assert_eq!(stores_in(&optimized), 1);
    assert_same_loop(&ir, &optimized, &[]);
}

#[test]
fn a_side_member_that_stores_the_key_blocks_the_hoist() {
    let ir = field_loop(None);
    let same = TreeWrites::of_members([&ir, &storing_member(b"x")]);
    let (optimized, stats) = optimize_with_tree(&ir, memory_on(), &same);
    assert_eq!(stats.loads_hoisted, 0);
    assert_eq!(loads(&optimized), 1);
    assert_eq!(
        optimized.preheader,
        [] as [crate::jit_ir::trace::TraceInstruction; 0]
    );
    // A side that stores another key does not.
    let other = TreeWrites::of_members([&ir, &storing_member(b"y")]);
    let (optimized, stats) = optimize_with_tree(&ir, memory_on(), &other);
    assert_eq!(stats.loads_hoisted, 1);
    assert_eq!(preheader_reads(&optimized), 1);
}

/// Tree members that write anywhere: one with an effect, one with an array
/// set.
fn unknown_writers() -> [TraceIr; 2] {
    let mut with_effect = Lp::new();
    with_effect.effect();
    let mut with_set = Lp::arrays_only();
    let (k, v) = (with_set.konst(int(1)), with_set.konst(f(1.0)));
    with_set.array_write(k, v, true);
    [with_effect.finish(), with_set.finish()]
}

#[test]
fn an_effect_or_array_set_anywhere_in_the_tree_blocks_every_hoist() {
    let ir = field_loop(None);
    for blocker in &unknown_writers() {
        let writes = TreeWrites::of_members([&ir, blocker]);
        assert!(writes.has_unknown_write());
        let (optimized, stats) = optimize_with_tree(&ir, memory_on(), &writes);
        assert_eq!(stats.loads_hoisted, 0);
        assert_eq!(
            optimized.preheader,
            [] as [crate::jit_ir::trace::TraceInstruction; 0]
        );
    }
    // In the loop itself, too.
    let mut l = Lp::new();
    let x = l.field_load(0);
    l.add(x);
    l.effect();
    let ir = l.finish();
    let (optimized, stats) = optimize_with(&ir, memory_on());
    assert_eq!(stats.loads_hoisted, 0);
    assert_eq!(loads(&optimized), 1);
}

/// A member reads of the tree's writes only what its optimization depends
/// on (`TreeWrites::read_by`): writes that read alike optimize alike, and a
/// write that changes the result reads differently.
#[test]
fn a_member_reads_only_the_tree_writes_its_optimization_depends_on() {
    let optimize = |ir: &TraceIr, writes: &TreeWrites| optimize_with_tree(ir, memory_on(), writes);
    let ir = field_loop(None);
    let alone = TreeWrites::of_members([&ir]);
    let stores = |key: &[u8]| TreeWrites::of_members([&ir, &storing_member(key)]);
    let (x, y, z) = (stores(b"x"), stores(b"v"), stores(b"w"));
    // Stores of keys the loop does not read look alike to it.
    assert_eq!(y.read_by(&ir), z.read_by(&ir));
    assert_eq!(optimize(&ir, &y), optimize(&ir, &z));
    // A store of the key it reads keeps the load in the loop.
    assert_ne!(x.read_by(&ir), y.read_by(&ir));
    assert_ne!(optimize(&ir, &x).0, optimize(&ir, &y).0);
    // Any field store keeps array reads in the loop, and an unknown write
    // keeps every read there.
    assert_ne!(alone.read_by(&ir), y.read_by(&ir));
    for blocker in &unknown_writers() {
        let writes = TreeWrites::of_members([&ir, blocker]);
        assert_ne!(writes.read_by(&ir), alone.read_by(&ir));
    }

    // A member without a loop moves nothing out of it, so no write of the
    // tree matters to it.
    let side = storing_member(b"x");
    assert!(!matches!(side.terminator, TraceTerminator::LoopBack { .. }));
    let own = TreeWrites::of_members([&side]).for_member(1);
    for writes in [&alone, &x, &y] {
        let writes = writes.for_member(1);
        assert_eq!(writes.read_by(&side), own.read_by(&side));
        assert_eq!(optimize(&side, &writes), optimize(&side, &own));
    }
    // A guard of its own that a side continues does.
    let continued = x.clone().with_continued_guards([(1, 0)]).for_member(1);
    assert_ne!(continued.read_by(&side), own.read_by(&side));
    // So does a guard of the loop.
    let continued = alone.clone().with_continued_guards([(0, 0)]);
    assert_ne!(continued.read_by(&ir), alone.read_by(&ir));
}

fn array_loop(load_key: i64, store_key: Option<i64>) -> TraceIr {
    let mut l = Lp::new();
    let k = l.konst(int(load_key));
    let a = l.array_load(k);
    l.add(a);
    if let Some(store_key) = store_key {
        let sk = l.konst(int(store_key));
        let v = l.konst(f(9.0));
        l.array_write(sk, v, false);
    }
    l.finish()
}

fn array_loads_in(ir: &TraceIr, part: &[TraceInstruction]) -> usize {
    let _ = ir;
    part.iter()
        .filter(|i| matches!(i.op, TraceOp::ArrayLoad { .. }))
        .count()
}

#[test]
fn an_array_load_with_a_constant_key_hoists_unless_a_store_may_equal_it() {
    let ir = array_loop(1, None);
    let (optimized, stats) = optimize_with(&ir, memory_on());
    assert_eq!(stats.loads_hoisted, 1);
    assert_eq!(array_loads_in(&optimized, &optimized.preheader), 1);
    assert_eq!(array_loads_in(&optimized, &optimized.body), 0);
    assert_eq!(optimized.guards.len(), 1);
    assert_same_loop(&ir, &optimized, &[4.0, 5.0]);

    // A store to key 2 does not block key 1.
    let ir = array_loop(1, Some(2));
    let (optimized, stats) = optimize_with(&ir, memory_on());
    assert_eq!(stats.loads_hoisted, 1);
    assert_same_loop(&ir, &optimized, &[4.0, 5.0]);

    // A store to key 1 does.
    let ir = array_loop(1, Some(1));
    let (optimized, stats) = optimize_with(&ir, memory_on());
    assert_eq!(stats.loads_hoisted, 0);
    assert_eq!(array_loads_in(&optimized, &optimized.body), 1);
    assert_same_loop(&ir, &optimized, &[4.0, 5.0]);

    // So does a store with an unknown key, even next to a constant one.
    let mut l = Lp::new();
    let k = l.konst(int(1));
    let a = l.array_load(k);
    l.add(a);
    let unknown = l.field_load(1);
    l.add(unknown);
    let ir = l.finish();
    let mut writer = Mem::new(1, 1, vec![array_binding(0)]);
    let (t, key) = (writer.tables[0], writer.keys[0]);
    let v = writer.konst(f(1.0));
    writer.store(0, t, key, v);
    let writer = writer.finish();
    let writes = TreeWrites::of_members([&ir, &writer]);
    let (_, stats) = optimize_with_tree(&ir, memory_on(), &writes);
    assert_eq!(stats.loads_hoisted, 1, "only the field load moves");
}

#[test]
fn an_array_set_or_effect_anywhere_in_the_tree_blocks_an_array_load_hoist() {
    let ir = array_loop(1, None);
    for blocker in &unknown_writers() {
        let writes = TreeWrites::of_members([&ir, blocker]);
        let (optimized, stats) = optimize_with_tree(&ir, memory_on(), &writes);
        assert_eq!(stats.loads_hoisted, 0);
        assert_eq!(array_loads_in(&optimized, &optimized.body), 1);
        assert_eq!(array_loads_in(&optimized, &optimized.preheader), 0);
        assert_same_loop(&ir, &optimized, &[4.0, 5.0]);
    }
}

#[test]
fn an_array_load_expecting_nil_is_never_hoisted() {
    let mut l = Lp::with(
        vec![
            array_binding(1),
            TableBinding {
                register: 1,
                kind: BindingKind::NoMetatable,
            },
        ],
        0,
    );
    let k = l.konst(int(1));
    let guard = l.guard();
    let table = l.table;
    let binding = l.array;
    emit(
        &mut l.b,
        l.pc,
        TraceOp::ArrayLoad {
            binding,
            table,
            key: k,
            expected: TraceType::Nil,
            past_end: false,
            guard,
        },
        TraceType::Nil,
    );
    let ir = l.finish();
    let (optimized, stats) = optimize_with(&ir, memory_on());
    assert_eq!(stats.loads_hoisted, 0);
    assert_eq!(array_loads_in(&optimized, &optimized.body), 1);
    assert_eq!(array_loads_in(&optimized, &optimized.preheader), 0);
}

#[test]
fn hoisting_array_loads_needs_the_guard_option_and_the_memory_option() {
    let ir = array_loop(1, None);
    for options in [
        memory_off(),
        OptimizeOptions {
            hoist_guards: false,
            ..memory_on()
        },
    ] {
        let (optimized, stats) = optimize_with(&ir, options);
        assert_eq!(stats.loads_hoisted, 0);
        assert!(optimized.preheader.iter().all(|i| !i.op.touches_heap()));
        assert_eq!(array_loads_in(&optimized, &optimized.body), 1);
    }
    // A field load needs no guard, so the guard option does not matter.
    let ir = field_loop(None);
    let options = OptimizeOptions {
        hoist_guards: false,
        ..memory_on()
    };
    assert_eq!(optimize_with(&ir, options).1.loads_hoisted, 1);
}

#[test]
fn a_hoisted_array_load_that_fails_at_entry_restores_the_entry_state() {
    let ir = array_loop(1, None);
    let (optimized, _) = optimize_with(&ir, memory_on());
    // Slot 1 is missing: the guard fails before any iteration.
    let mut heap = loop_model(&[]);
    let evaluation = run(&optimized, &mut heap, &loop_regs(), 3);
    let Outcome::GuardFailed { restored, .. } = evaluation.outcome else {
        panic!("the hoisted guard fails")
    };
    assert_eq!(evaluation.iterations, 0);
    assert_eq!(
        evaluation.effects,
        [] as [crate::jit_ir::trace::reference::EffectRecord; 0]
    );
    assert_eq!(restored.pc, 1);
    assert_eq!(restored.registers, loop_regs().to_vec());
    assert_eq!(restored.effect_position, 0);
}

/// A loop that sums `t.x` where `t` is the entry table, read as a runtime
/// table field (fields 0 `x` and 1 `y`). `stored` names a field the loop
/// writes.
fn table_field_loop(stored: Option<u32>) -> TraceIr {
    let mut l = Lp::arrays_only();
    l.b.set_fields(vec![
        TableField {
            key: b"x"[..].into(),
            ty: TraceType::Float,
        },
        TableField {
            key: b"y"[..].into(),
            ty: TraceType::Float,
        },
    ]);
    let guard = l.guard();
    let table = l.table;
    let x = emit(
        &mut l.b,
        l.pc,
        TraceOp::TableFieldLoad {
            field: 0,
            table,
            guard,
        },
        TraceType::Float,
    );
    l.add(x);
    if let Some(field) = stored {
        let one = l.konst(f(1.0));
        let guard = l.guard();
        let effect = l.b.next_effect();
        l.b.emit(
            site(l.pc),
            TraceOp::TableFieldStore {
                field,
                table,
                value: one,
                effect,
                guard,
            },
            None,
        );
    }
    l.finish()
}

/// A tree member that stores table field `key` (field 0).
fn table_field_storing_member(key: &[u8]) -> TraceIr {
    let mut m = Mem::new(1, 0, vec![]);
    m.b.set_fields(vec![TableField {
        key: key.into(),
        ty: TraceType::Float,
    }]);
    let t = m.tables[0];
    let v = m.konst(f(1.0));
    let guard = m.guard();
    let effect = m.b.next_effect();
    m.b.emit(
        site(m.pc),
        TraceOp::TableFieldStore {
            field: 0,
            table: t,
            value: v,
            effect,
            guard,
        },
        None,
    );
    m.finish()
}

fn row_model(x: TraceScalar) -> Model {
    let mut model = loop_model(&[]);
    model.rows = vec![vec![x, f(7.0)]];
    model
}

fn table_field_loads_in(part: &[TraceInstruction]) -> usize {
    part.iter()
        .filter(|i| matches!(i.op, TraceOp::TableFieldLoad { .. }))
        .count()
}

#[test]
fn a_runtime_table_field_load_hoists_unless_a_member_stores_its_key() {
    let ir = table_field_loop(None);
    let (optimized, stats) = optimize_with(&ir, memory_on());
    assert_eq!(stats.loads_hoisted, 1);
    assert_eq!(table_field_loads_in(&optimized.preheader), 1);
    assert_eq!(table_field_loads_in(&optimized.body), 0);
    optimized.verify().unwrap();
    let run_both = |ir: &TraceIr, optimized: &TraceIr| {
        let mut before = row_model(f(2.0));
        let mut after = row_model(f(2.0));
        let a = run(ir, &mut before, &loop_regs(), 3);
        let b = run(optimized, &mut after, &loop_regs(), 3);
        assert_eq!(a.outcome, b.outcome);
        assert_eq!(a.effects, b.effects);
        assert_eq!(before.rows, after.rows);
    };
    run_both(&ir, &optimized);

    // With the flag off the IR stays as it was.
    let (old, stats) = optimize_with(&ir, memory_off());
    assert_eq!(stats.loads_hoisted, 0);
    assert_eq!(table_field_loads_in(&old.body), 1);

    // Without the guard option a guarded read stays put.
    let no_guards = OptimizeOptions {
        hoist_guards: false,
        ..memory_on()
    };
    let (kept, stats) = optimize_with(&ir, no_guards);
    assert_eq!(stats.loads_hoisted, 0);
    assert_eq!(table_field_loads_in(&kept.body), 1);

    // A store of the same key in the root blocks it; another key does not.
    let same = table_field_loop(Some(0));
    let (optimized, stats) = optimize_with(&same, memory_on());
    assert_eq!(stats.loads_hoisted, 0);
    assert_eq!(table_field_loads_in(&optimized.body), 1);
    assert!(optimized.preheader.iter().all(|i| !i.op.touches_heap()));
    run_both(&same, &optimized);
    let other = table_field_loop(Some(1));
    let (optimized, stats) = optimize_with(&other, memory_on());
    assert_eq!(stats.loads_hoisted, 1);
    assert_eq!(table_field_loads_in(&optimized.preheader), 1);
    run_both(&other, &optimized);

    // A side member that stores the key, as a table field or a bound field,
    // blocks it; one that stores another key does not.
    for blocker in [table_field_storing_member(b"x"), storing_member(b"x")] {
        let writes = TreeWrites::of_members([&ir, &blocker]);
        let (optimized, stats) = optimize_with_tree(&ir, memory_on(), &writes);
        assert_eq!(stats.loads_hoisted, 0);
        assert_eq!(table_field_loads_in(&optimized.body), 1);
    }
    for harmless in [table_field_storing_member(b"y"), storing_member(b"y")] {
        let writes = TreeWrites::of_members([&ir, &harmless]);
        let (optimized, stats) = optimize_with_tree(&ir, memory_on(), &writes);
        assert_eq!(stats.loads_hoisted, 1);
        assert_eq!(table_field_loads_in(&optimized.preheader), 1);
    }
}

#[test]
fn a_hoisted_table_field_load_that_fails_at_entry_restores_the_entry_state() {
    let (optimized, _) = optimize_with(&table_field_loop(None), memory_on());
    assert_eq!(table_field_loads_in(&optimized.preheader), 1);
    // The field is nil: the guard fails before any iteration.
    let mut heap = row_model(TraceScalar::Nil);
    let evaluation = run(&optimized, &mut heap, &loop_regs(), 3);
    let Outcome::GuardFailed { restored, .. } = evaluation.outcome else {
        panic!("the hoisted guard fails")
    };
    assert_eq!(evaluation.iterations, 0);
    assert_eq!(
        evaluation.effects,
        [] as [crate::jit_ir::trace::reference::EffectRecord; 0]
    );
    assert_eq!(restored.registers, loop_regs().to_vec());
    assert_eq!(restored.effect_position, 0);
}

#[test]
fn the_verifier_accepts_table_reads_in_a_preheader_and_nothing_else_that_touches_the_heap() {
    let (hoisted, _) = optimize_with(&array_loop(1, None), memory_on());
    hoisted.verify().unwrap();
    let (hoisted_field, _) = optimize_with(&field_loop(None), memory_on());
    hoisted_field.verify().unwrap();
    let (hoisted_row, _) = optimize_with(&table_field_loop(None), memory_on());
    assert_eq!(table_field_loads_in(&hoisted_row.preheader), 1);
    hoisted_row.verify().unwrap();

    // Table reads are fine in a preheader.
    let (base, _) = optimize_with(&field_loop(None), memory_on());
    let table = base.entry.inputs[1].value;
    let acc = base.entry.inputs[0].value;
    let site = base.body[0].site;
    let with_preheader = |op, result| with_preheader_op(&base, site, op, result);
    let mut length = with_preheader(
        TraceOp::TableLength { binding: 3, table },
        Some(TraceType::Integer),
    );
    length.bindings.push(TableBinding {
        register: 1,
        kind: BindingKind::Length {
            ty: TraceType::Integer,
            raw: false,
        },
    });
    length.verify().unwrap();

    // Anything else that touches the heap is refused, and the refusal comes
    // from the preheader rule, not from some other broken invariant.
    let effect = EffectId(0);
    let refused = [
        (
            TraceOp::FieldStore {
                binding: 0,
                table,
                value: acc,
                effect,
            },
            None,
        ),
        (
            TraceOp::Effect {
                effect,
                label: 1,
                operands: vec![],
                result: None,
            },
            None,
        ),
        (
            TraceOp::Read {
                label: 1,
                operands: vec![table],
                result: TraceType::Float,
            },
            Some(TraceType::Float),
        ),
    ];
    for (op, result) in refused {
        let ir = with_preheader(op.clone(), result);
        let err = ir.verify().unwrap_err();
        assert!(
            matches!(
                err,
                TraceVerifyError::EffectOrder {
                    detail: "the preheader holds no stores or effects",
                    ..
                }
            ),
            "{op:?}: {err:?}"
        );
    }
    // ArrayStore and ArraySet carry a guard and a key as well. A table set
    // may not share a trace with a field or length binding, so it gets a
    // trace of its own.
    for set in [false, true] {
        let mut l = if set { Lp::arrays_only() } else { Lp::new() };
        let binding = l.array;
        let k = l.konst(int(1));
        let a = l.array_load(k);
        l.add(a);
        let (loop_ir, _) = optimize_with(&l.finish(), memory_on());
        let table = loop_ir.entry.inputs[1].value;
        let acc = loop_ir.entry.inputs[0].value;
        let key = loop_ir
            .preheader
            .iter()
            .find_map(|i| matches!(i.op, TraceOp::Constant(_)).then_some(i.result))
            .flatten()
            .expect("the key constant moved to the preheader");
        let guard = GuardId(0);
        let op = if set {
            TraceOp::ArraySet {
                binding,
                table,
                key,
                value: acc,
                effect,
                guard,
            }
        } else {
            TraceOp::ArrayStore {
                binding,
                table,
                key,
                value: acc,
                effect,
                guard,
            }
        };
        let site = loop_ir.body[0].site;
        let err = with_preheader_op(&loop_ir, site, op.clone(), None)
            .verify()
            .unwrap_err();
        assert!(
            matches!(
                err,
                TraceVerifyError::EffectOrder {
                    detail: "the preheader holds no stores or effects",
                    ..
                }
            ),
            "{op:?}: {err:?}"
        );
    }
}

/// `base` with one more instruction at the end of its preheader.
fn with_preheader_op(
    base: &TraceIr,
    site: TraceSite,
    op: TraceOp,
    result: Option<TraceType>,
) -> TraceIr {
    let mut ir = base.clone();
    let result = result.map(|ty| {
        let value = ValueId(ir.values.len() as u32);
        ir.values.push(ValueInfo {
            ty,
            def: ValueDef::Preheader(ir.preheader.len()),
        });
        value
    });
    ir.preheader.push(TraceInstruction { site, result, op });
    ir
}

// ---- helpers and the memory passes -----------------------------------------

/// Loads slot 1, calls `helper` on a constant, loads slot 1 again. Returns
/// the optimized trace and its counters after checking it against the
/// reference evaluator over a heap whose first slot a writing helper changes.
fn load_around_helper(helper: HelperId) -> (TraceIr, OptimizeStats) {
    let mut m = Mem::new(1, 0, vec![array_binding(0)]);
    let t = m.tables[0];
    let k1 = m.konst(int(1));
    let first = m.load(0, t, k1, TraceType::Float);
    m.out(first);
    let argument = m.konst(int(7));
    let args: Vec<ValueId> = match helper {
        HelperId::TEST_DIVMOD => vec![argument, m.konst(int(2))],
        _ => vec![argument],
    };
    m.helper(helper, &args).unwrap();
    let second = m.load(0, t, k1, TraceType::Float);
    m.out(second);
    let ir = m.finish();
    let heaps = [memory_model(vec![floats(&[1.0, 2.0])], vec![], vec![])];
    check_forwarding(&ir, &heaps, &one_table_regs(&[]))
}

#[test]
fn a_pure_helper_is_no_barrier_for_a_cached_table_read() {
    let (optimized, stats) = load_around_helper(HelperId::TEST_DIVMOD);
    assert_eq!(loads(&optimized), 1);
    assert_eq!((stats.loads_forwarded, stats.loads_merged), (0, 1));
    assert_eq!(optimized.helper_calls(), [HelperId::TEST_DIVMOD]);
}

#[test]
fn a_reading_helper_is_no_barrier_for_a_cached_table_read() {
    let (optimized, stats) = load_around_helper(HelperId::TEST_READ);
    assert_eq!(loads(&optimized), 1);
    assert_eq!((stats.loads_forwarded, stats.loads_merged), (0, 1));
}

#[test]
fn a_writing_helper_invalidates_cached_table_reads() {
    let (optimized, stats) = load_around_helper(HelperId::TEST_WRITE);
    assert_eq!(loads(&optimized), 2);
    assert_eq!((stats.loads_forwarded, stats.loads_merged), (0, 0));
    // The helper commits an effect of its own.
    assert_eq!(stores_in(&optimized), 1);
}

#[test]
fn a_private_state_helper_keeps_cached_table_reads_but_stays_in_order() {
    let (optimized, stats) = load_around_helper(HelperId::TEST_DRAW);
    // The second load is answered by the first, across the draw...
    assert_eq!(loads(&optimized), 1);
    assert_eq!((stats.loads_forwarded, stats.loads_merged), (0, 1));
    // ... and the draw is neither dropped nor given up as an effect.
    assert_eq!(optimized.helper_calls(), [HelperId::TEST_DRAW]);
    assert_eq!(stores_in(&optimized), 1);
    let effects = HelperId::TEST_DRAW.effects();
    assert!(effects.commits_effect());
    assert!(!effects.contains(HelperEffects::WRITES_HEAP));
    assert!(!effects.moves_storage());
}

#[test]
fn a_store_answers_a_load_across_a_private_state_helper() {
    let mut m = Mem::new(1, 0, vec![array_binding(0)]);
    let t = m.tables[0];
    let (k1, v) = (m.konst(int(1)), m.konst(f(5.0)));
    m.store(0, t, k1, v);
    let a = m.konst(int(9));
    m.helper(HelperId::TEST_DRAW, &[a]).unwrap();
    let reloaded = m.load(0, t, k1, TraceType::Float);
    m.out(reloaded);
    let ir = m.finish();
    let heaps = [memory_model(vec![floats(&[1.0, 2.0])], vec![], vec![])];
    let (optimized, stats) = check_forwarding(&ir, &heaps, &one_table_regs(&[]));
    assert_eq!(loads(&optimized), 0);
    assert_eq!(stats.loads_forwarded, 1);
    assert_eq!(optimized.helper_calls(), [HelperId::TEST_DRAW]);
}

#[test]
fn a_writing_helper_stops_a_store_from_answering_a_later_load() {
    let mut m = Mem::new(1, 0, vec![array_binding(0)]);
    let t = m.tables[0];
    let (k1, v) = (m.konst(int(1)), m.konst(f(5.0)));
    m.store(0, t, k1, v);
    let argument = m.konst(int(9));
    m.helper(HelperId::TEST_WRITE, &[argument]).unwrap();
    let reloaded = m.load(0, t, k1, TraceType::Float);
    m.out(reloaded);
    let ir = m.finish();
    let heaps = [memory_model(vec![floats(&[1.0, 2.0])], vec![], vec![])];
    let (optimized, stats) = check_forwarding(&ir, &heaps, &one_table_regs(&[]));
    assert_eq!(loads(&optimized), 1);
    assert_eq!(stats.loads_forwarded, 0);
    assert_eq!(stores_in(&optimized), 2);
}

#[test]
fn a_store_answers_a_load_across_a_pure_helper() {
    let mut m = Mem::new(1, 0, vec![array_binding(0)]);
    let t = m.tables[0];
    let (k1, v) = (m.konst(int(1)), m.konst(f(5.0)));
    m.store(0, t, k1, v);
    let (a, b) = (m.konst(int(9)), m.konst(int(4)));
    m.helper(HelperId::TEST_DIVMOD, &[a, b]).unwrap();
    let reloaded = m.load(0, t, k1, TraceType::Float);
    m.out(reloaded);
    let ir = m.finish();
    let heaps = [memory_model(vec![floats(&[1.0, 2.0])], vec![], vec![])];
    let (optimized, stats) = check_forwarding(&ir, &heaps, &one_table_regs(&[]));
    assert_eq!(loads(&optimized), 0);
    assert_eq!(stats.loads_forwarded, 1);
}

#[test]
fn a_helper_with_two_results_gives_both_to_the_evaluator() {
    let mut m = Mem::new(1, 0, vec![array_binding(0)]);
    let (a, b) = (m.konst(int(17)), m.konst(int(5)));
    let quotient = m.helper(HelperId::TEST_DIVMOD, &[a, b]).unwrap();
    // The remainder is the call's second result.
    m.pc += 1;
    let remainder = emit(
        &mut m.b,
        m.pc,
        TraceOp::HelperResult {
            call: quotient,
            index: 1,
        },
        TraceType::Integer,
    );
    m.out(quotient);
    m.out(remainder);
    let ir = m.finish();
    let heaps = [memory_model(vec![floats(&[1.0])], vec![], vec![])];
    let (optimized, _) = check_forwarding(&ir, &heaps, &one_table_regs(&[]));
    assert_eq!(optimized.helper_calls(), [HelperId::TEST_DIVMOD]);
    let run = run_forced(&ir, &mut heaps[0].clone(), &one_table_regs(&[]), 1, None);
    let Outcome::Exited(restored) = run.outcome else {
        panic!("{:?}", run.outcome);
    };
    assert_eq!(restored.registers[1], int(3));
    assert_eq!(restored.registers[2], int(2));
}

#[test]
fn the_verifier_checks_helper_calls() {
    let build = |helper: HelperId, args: usize, bad_effect: bool| {
        let mut m = Mem::new(1, 0, vec![array_binding(0)]);
        let args: Vec<ValueId> = (0..args).map(|_| m.konst(int(1))).collect();
        let guard = m.guard();
        let effect =
            (bad_effect || helper == HelperId::TEST_WRITE || helper == HelperId::TEST_DRAW)
                .then(|| m.b.next_effect());
        m.b.emit(
            site(m.pc),
            TraceOp::CallHelper {
                helper,
                args,
                effect,
                guard,
            },
            helper
                .info()
                .and_then(|info| info.results.first())
                .map(|r| r.trace_type()),
        );
        let exit = m.b.snapshot(snap(
            SnapshotPhase::Before { pc: 400 },
            m.b.effect_position(),
            None,
            m.registers(),
        ));
        m.b.finish(TraceTerminator::Exit { snapshot: exit })
    };
    assert!(build(HelperId::TEST_DIVMOD, 2, false).verify().is_ok());
    assert!(build(HelperId::TEST_WRITE, 1, false).verify().is_ok());
    // Too few arguments.
    assert!(matches!(
        build(HelperId::TEST_DIVMOD, 1, false).verify(),
        Err(TraceVerifyError::TypeMismatch { .. })
    ));
    // An effect on a helper that does not write.
    assert!(matches!(
        build(HelperId::TEST_READ, 1, true).verify(),
        Err(TraceVerifyError::TypeMismatch { .. })
    ));
    // An id the table lacks.
    assert!(matches!(
        build(HelperId(999), 1, false).verify(),
        Err(TraceVerifyError::TypeMismatch { .. })
    ));
}

#[test]
fn the_verifier_refuses_a_metatable_write_beside_a_metatable_fact() {
    let build = |kind: Option<BindingKind>, write: bool| {
        let mut bindings = vec![array_binding(0)];
        if let Some(kind) = kind {
            bindings.push(TableBinding { register: 0, kind });
        }
        let mut m = Mem::new(1, 0, bindings);
        if write {
            let table = m.tables[0];
            let metatable = m.konst(TraceScalar::Nil);
            m.helper(HelperId::SETMETATABLE, &[table, metatable]);
        }
        let exit = m.b.snapshot(snap(
            SnapshotPhase::Before { pc: 400 },
            m.b.effect_position(),
            None,
            m.registers(),
        ));
        m.b.finish(TraceTerminator::Exit { snapshot: exit })
    };
    // The write alone, a fact alone, and a field the table holds itself.
    assert!(build(None, true).verify().is_ok());
    assert!(
        build(Some(BindingKind::NoMetatable), false)
            .verify()
            .is_ok()
    );
    let own_field = || BindingKind::Field {
        key: b"x"[..].into(),
        ty: TraceType::Float,
        depth: 0,
    };
    assert!(build(Some(own_field()), true).verify().is_ok());
    // A fact about a metatable beside the write.
    for kind in [
        BindingKind::NoMetatable,
        BindingKind::Length {
            ty: TraceType::Integer,
            raw: false,
        },
        BindingKind::Field {
            key: b"x"[..].into(),
            ty: TraceType::Float,
            depth: 1,
        },
        BindingKind::Callee {
            key: b"f"[..].into(),
            lane: 0,
            depth: 1,
        },
    ] {
        let ir = build(Some(kind.clone()), true);
        assert!(ir.stale_metatable_fact().is_some(), "{kind:?}");
        assert!(
            matches!(ir.verify(), Err(TraceVerifyError::InvalidEntry { .. })),
            "{kind:?}"
        );
    }
}

#[test]
fn the_verifier_accepts_a_nil_read_past_the_end_only_with_the_header_and_no_metatable() {
    let build = |extra: &[BindingKind], past_end: bool, expected: TraceType| {
        let mut bindings = vec![array_binding(0)];
        bindings.extend(extra.iter().map(|kind| TableBinding {
            register: 0,
            kind: kind.clone(),
        }));
        let mut m = Mem::new(1, 1, bindings);
        let table = m.tables[0];
        let key = m.konst(TraceScalar::Float(5000.0));
        let guard = m.guard();
        m.b.emit(
            site(m.pc),
            TraceOp::ArrayLoad {
                binding: 0,
                table,
                key,
                expected,
                past_end,
                guard,
            },
            Some(expected),
        );
        let exit = m.b.snapshot(snap(
            SnapshotPhase::Before { pc: 400 },
            m.b.effect_position(),
            None,
            m.registers(),
        ));
        m.b.finish(TraceTerminator::Exit { snapshot: exit })
    };
    let plain = BindingKind::NoMetatable;
    let header = BindingKind::Header { strict: false };
    let nil = TraceType::Nil;
    assert!(
        build(&[plain.clone(), header.clone()], true, nil)
            .verify()
            .is_ok()
    );
    // Without the header nothing says the hash part is empty.
    assert!(
        build(std::slice::from_ref(&plain), true, nil)
            .verify()
            .is_err()
    );
    // Without the fact about the metatable the nil is not the access's value.
    assert!(
        build(std::slice::from_ref(&header), true, nil)
            .verify()
            .is_err()
    );
    // Only a nil read goes past the end.
    assert!(
        build(&[plain, header], true, TraceType::Float)
            .verify()
            .is_err()
    );
}

#[test]
fn a_metatable_write_moves_no_storage() {
    let effects = HelperId::SETMETATABLE.effects();
    assert!(effects.contains(HelperEffects::WRITES_HEAP));
    assert!(effects.contains(HelperEffects::WRITES_METATABLE));
    assert!(!effects.moves_storage());
    assert!(HelperId::TEST_WRITE.effects().moves_storage());
    // A trace that holds a field binding may call it; a table set may not.
    let mut m = Mem::new(1, 0, vec![field_binding(0, b"x")]);
    let table = m.tables[0];
    let metatable = m.konst(TraceScalar::Nil);
    m.helper(HelperId::SETMETATABLE, &[table, metatable]);
    let ir = m.finish();
    assert!(!ir.sets_tables());
    assert!(ir.stale_metatable_fact().is_none());
}

#[test]
fn an_array_write_moves_arrays_but_no_hash_storage() {
    let effects = HelperId::TABLE_INSERT.effects();
    assert!(effects.contains(HelperEffects::WRITES_HEAP));
    assert!(effects.contains(HelperEffects::REWRITES_ARRAYS));
    assert!(effects.moves_storage());
    assert!(!effects.moves_hash_storage());
    // A read of the table the same helper family answers moves nothing.
    assert!(!HelperId::TABLE_REMOVE_NOTHING.effects().moves_storage());
    // Another writing helper may move anything.
    assert!(HelperId::TEST_WRITE.effects().moves_hash_storage());

    // A field binding may sit beside the array write, and a length binding
    // too: the VM rewrites the border's word after the call.
    for binding in [
        field_binding(0, b"x"),
        TableBinding {
            register: 0,
            kind: BindingKind::Length {
                ty: TraceType::Integer,
                raw: false,
            },
        },
    ] {
        let mut m = Mem::new(1, 0, vec![binding]);
        let table = m.tables[0];
        let value = m.konst(TraceScalar::Integer(1));
        m.helper(HelperId::TABLE_INSERT, &[table, value]);
        let ir = m.finish();
        // (`finish` verified it.)
        assert!(ir.sets_tables());
        assert!(!ir.moves_hash_storage());
    }

    // The write beside a general table write is the table write's problem.
    let mut m = Mem::new(1, 0, vec![field_binding(0, b"x")]);
    let table = m.tables[0];
    let value = m.konst(TraceScalar::Integer(1));
    m.helper(HelperId::TABLE_INSERT, &[table, value]);
    let one = m.konst(TraceScalar::Integer(1));
    m.helper(HelperId::TEST_WRITE, &[one]);
    let ir = m.finish_unverified();
    assert!(ir.moves_hash_storage());
    assert!(ir.storage_a_set_would_move().is_some());
    assert!(ir.verify().is_err());
}

/// The `table.set_string` helper on the first table, with a constant string
/// key and a constant value.
fn set_string(m: &mut Mem, value: TraceScalar) {
    let table = m.tables[0];
    // (Any scalar types as the key: the IR checks the types, not the bytes.)
    let key = m.konst(TraceScalar::Nil);
    let value = m.konst(value);
    m.helper(HelperId::SET_STRING, &[table, key, value]);
}

#[test]
fn a_store_of_a_string_key_moves_hash_storage_but_not_arrays() {
    let effects = HelperId::SET_STRING.effects();
    assert!(effects.contains(HelperEffects::WRITES_HEAP));
    assert!(effects.moves_hash_storage());
    assert!(!effects.contains(HelperEffects::REWRITES_ARRAYS));
    // A trace that stores a string key beside an array binding is fine, and
    // says it moves hash storage, so a native hash step stays off.
    let mut m = Mem::new(1, 0, vec![array_binding(0)]);
    set_string(&mut m, int(1));
    let ir = m.finish();
    assert!(ir.moves_hash_storage());
    assert!(ir.storage_a_set_would_move().is_none());
    // Beside a bound field, a rehash would move the field's slot.
    let mut m = Mem::new(1, 0, vec![array_binding(0), field_binding(0, b"x")]);
    set_string(&mut m, int(1));
    let ir = m.finish_unverified();
    assert!(ir.moves_hash_storage());
    assert!(ir.storage_a_set_would_move().is_some());
}

#[test]
fn an_overwrite_in_place_keeps_every_binding_and_a_delete_in_place_refuses_fields() {
    // The overwrite keeps the entries, their types and the layout: a field
    // binding may sit beside it, and the tree's native hash step stays.
    let mut m = Mem::new(1, 0, vec![field_binding(0, b"x")]);
    let table = m.tables[0];
    let (key, value) = (m.konst(TraceScalar::Nil), m.konst(int(1)));
    m.helper(HelperId::SET_PRESENT, &[table, key, value]);
    let ir = m.finish();
    assert!(ir.verify().is_ok());
    assert!(!ir.moves_hash_storage() && !ir.deletes_keys());
    // The delete leaves entries where they are but empties one, which a
    // field binding typed for it would read as nil.
    let mut m = Mem::new(1, 0, vec![field_binding(0, b"x")]);
    let table = m.tables[0];
    let key = m.konst(TraceScalar::Nil);
    m.helper(HelperId::DELETE_STRING, &[table, key]);
    let ir = m.finish_unverified();
    assert!(!ir.moves_hash_storage() && ir.deletes_keys());
    assert!(ir.storage_a_set_would_move().is_some());
    assert!(ir.verify().is_err());
    // Beside an array binding it is fine, and keeps the native hash step.
    let mut m = Mem::new(1, 0, vec![array_binding(0)]);
    let table = m.tables[0];
    let key = m.konst(TraceScalar::Nil);
    m.helper(HelperId::DELETE_STRING, &[table, key]);
    let ir = m.finish();
    assert!(ir.verify().is_ok() && !ir.moves_hash_storage());
}

#[test]
fn a_store_into_a_deleted_key_refuses_fields_like_the_delete_and_keeps_hash_storage() {
    // The store fills a deleted entry where it stands: no entry moves, so the
    // tree's string indexes stay, and a field binding typed for a nil it
    // turns into a value is refused as for the delete.
    let mut m = Mem::new(1, 0, vec![field_binding(0, b"x")]);
    let table = m.tables[0];
    let (key, value) = (m.konst(TraceScalar::Nil), m.konst(int(1)));
    m.helper(HelperId::SET_REVIVE, &[table, key, value]);
    let ir = m.finish_unverified();
    assert!(!ir.moves_hash_storage() && ir.deletes_keys());
    assert!(ir.storage_a_set_would_move().is_some());
    assert!(ir.verify().is_err());
    let mut m = Mem::new(1, 0, vec![array_binding(0)]);
    let table = m.tables[0];
    let (key, value) = (m.konst(TraceScalar::Nil), m.konst(int(1)));
    m.helper(HelperId::SET_REVIVE, &[table, key, value]);
    let ir = m.finish();
    assert!(ir.verify().is_ok() && !ir.moves_hash_storage());
}

#[test]
fn the_stores_native_code_makes_in_place_use_the_string_index_and_the_others_do_not() {
    for helper in [
        HelperId::SET_PRESENT,
        HelperId::SET_REVIVE,
        HelperId::DELETE_STRING,
    ] {
        let mut m = Mem::new(1, 0, vec![array_binding(0)]);
        let table = m.tables[0];
        let key = m.konst(TraceScalar::Nil);
        if helper == HelperId::DELETE_STRING {
            m.helper(helper, &[table, key]);
        } else {
            let value = m.konst(int(1));
            m.helper(helper, &[table, key, value]);
        }
        let ir = m.finish();
        assert!(ir.uses_string_index(), "{helper:?}");
        assert!(!ir.reads_hash(), "{helper:?}");
    }
    let mut m = Mem::new(1, 0, vec![array_binding(0)]);
    set_string(&mut m, int(1));
    assert!(!m.finish().uses_string_index());
}

#[test]
fn a_store_of_a_string_key_invalidates_the_array_reads_around_it() {
    // A traversal that updates the table it walks reads a slot, stores a
    // string key (which may clear the entry the step reads next) and reads
    // the slot again: the second read cannot be answered by the first.
    let mut m = Mem::new(1, 1, vec![array_binding(0)]);
    let (t, k) = (m.tables[0], m.keys[0]);
    let first = m.load(0, t, k, TraceType::Float);
    m.out(first);
    set_string(&mut m, int(1));
    let second = m.load(0, t, k, TraceType::Float);
    m.out(second);
    let ir = m.finish();
    let (optimized, stats) = optimize_with(&ir, memory_on());
    assert_eq!(loads(&optimized), 2);
    assert_eq!((stats.loads_forwarded, stats.loads_merged), (0, 0));
    assert_eq!(optimized.helper_calls(), [HelperId::SET_STRING]);
}

#[test]
fn a_store_of_a_string_key_in_a_loop_keeps_the_array_reads_in_the_body() {
    // The body reads slot 1 and stores a string key: the read cannot leave
    // the loop, since the store may change what the next iteration sees.
    let mut l = Lp::with(vec![array_binding(1)], 0);
    let (table, key) = (l.table, l.konst(int(1)));
    l.pc += 1;
    let guard = l.guard();
    let loaded = emit(
        &mut l.b,
        l.pc,
        TraceOp::ArrayLoad {
            binding: 0,
            table,
            key,
            expected: TraceType::Float,
            past_end: false,
            guard,
        },
        TraceType::Float,
    );
    l.add(loaded);
    let name = l.konst(TraceScalar::Nil);
    let value = l.konst(int(1));
    let guard = l.guard();
    let effect = l.b.next_effect();
    l.b.emit(
        site(l.pc),
        TraceOp::CallHelper {
            helper: HelperId::SET_STRING,
            args: vec![table, name, value],
            effect: Some(effect),
            guard,
        },
        None,
    );
    let ir = l.finish();
    let (optimized, stats) = optimize_with(&ir, memory_on());
    assert_eq!(stats.loads_hoisted, 0);
    assert!(
        optimized
            .body
            .iter()
            .any(|i| matches!(i.op, TraceOp::ArrayLoad { .. }))
    );
}

/// A loop that adds the border of its table, with an insert into the table
/// first when `insert` says so.
fn length_loop(insert: bool) -> TraceIr {
    let mut l = Lp::with(
        vec![TableBinding {
            register: 1,
            kind: BindingKind::Length {
                ty: TraceType::Float,
                raw: false,
            },
        }],
        0,
    );
    if insert {
        let value = l.konst(f(1.0));
        let guard = l.guard();
        let effect = l.b.next_effect();
        let table = l.table;
        l.b.emit(
            site(l.pc),
            TraceOp::CallHelper {
                helper: HelperId::TABLE_INSERT,
                args: vec![table, value],
                effect: Some(effect),
                guard,
            },
            None,
        );
    }
    l.pc += 1;
    let table = l.table;
    let length = emit(
        &mut l.b,
        l.pc,
        TraceOp::TableLength { binding: 0, table },
        TraceType::Float,
    );
    l.add(length);
    l.finish()
}

#[test]
fn a_border_moves_out_of_the_loop_unless_the_loop_inserts() {
    let ir = length_loop(false);
    assert_eq!(optimize_with(&ir, memory_on()).1.loads_hoisted, 1);
    // The insert changes the border, so every iteration reads it again.
    let ir = length_loop(true);
    let (optimized, stats) = optimize_with(&ir, memory_on());
    assert_eq!(stats.loads_hoisted, 0);
    assert!(
        optimized
            .body
            .iter()
            .any(|i| matches!(i.op, TraceOp::TableLength { .. }))
    );
    assert!(optimized.sets_tables());
    assert!(!optimized.moves_hash_storage());
}

/// A loop that makes a table every iteration, fills it and sums a field.
fn table_making_loop() -> TraceIr {
    let mut l = Lp::new();
    // Every argument is a constant: nothing about the call varies.
    let length = l.konst(int(2));
    let table = l.helper(HelperId::TABLE_NEW, &[length]).unwrap();
    let (key, value) = (l.konst(int(1)), l.konst(f(1.0)));
    l.helper(HelperId::TABLE_INIT, &[table, key, value]);
    let x = l.field_load(0);
    l.add(x);
    l.finish()
}

fn helper_calls(part: &[TraceInstruction], wanted: HelperId) -> usize {
    part.iter()
        .filter(|i| matches!(&i.op, TraceOp::CallHelper { helper, .. } if *helper == wanted))
        .count()
}

#[test]
fn a_table_made_in_a_loop_is_made_by_the_body_every_iteration() {
    let ir = table_making_loop();
    assert!(ir.makes_tables());
    // The call has constant arguments, which would move it into the
    // preheader as any helper that writes nothing: one table for the whole
    // run. It makes a new table each time, so it stays, with its fill, and
    // the trace is not one that elides the allocation.
    for level in [OptimizeLevel::Baseline, OptimizeLevel::Optimized] {
        let (optimized, _) = optimize_trace(&ir, level);
        optimized.verify().unwrap();
        assert_eq!(
            helper_calls(&optimized.body, HelperId::TABLE_NEW),
            1,
            "{level:?}"
        );
        assert_eq!(
            helper_calls(&optimized.body, HelperId::TABLE_INIT),
            1,
            "{level:?}"
        );
        assert!(
            optimized
                .preheader
                .iter()
                .all(|i| !matches!(&i.op, TraceOp::CallHelper { .. }))
        );
        assert!(optimized.makes_tables(), "{level:?}");
    }
    // A loop that makes none says so.
    assert!(!field_loop(None).makes_tables());
}

#[test]
fn the_verifier_refuses_a_table_made_in_the_preheader() {
    let mut ir = table_making_loop();
    let at = ir
        .body
        .iter()
        .position(|i| {
            matches!(&i.op, TraceOp::CallHelper { helper, .. } if *helper == HelperId::TABLE_NEW)
        })
        .unwrap();
    // The call moves with the constants before it, and its guard gets the
    // entry state a preheader guard needs, so that only the placement is
    // wrong: the preheader runs once, and one table would serve every
    // iteration.
    let latch = match ir.snapshots[ir.poll_snapshot.unwrap().index()].phase {
        SnapshotPhase::After { pc, .. } => pc,
        phase @ SnapshotPhase::Before { .. } => panic!("{phase:?}"),
    };
    let moved: Vec<_> = ir.body.drain(..=at).collect();
    let TraceOp::CallHelper { guard, .. } = moved.last().unwrap().op else {
        panic!("the call comes last")
    };
    let snapshot = &mut ir.snapshots[ir.guards[guard.index()].snapshot.index()];
    snapshot.phase = SnapshotPhase::Before { pc: 1 };
    snapshot.frames[0].last_interpreted_pc = Some(latch);
    snapshot.frames[0].registers = vec![RegisterEntry::Unchanged; 2].into();
    ir.preheader.extend(moved);
    // The value table says where each value is defined.
    for (position, instruction) in ir.preheader.iter().enumerate() {
        if let Some(value) = instruction.result {
            ir.values[value.index()].def = ValueDef::Preheader(position);
        }
    }
    for (position, instruction) in ir.body.iter().enumerate() {
        if let Some(value) = instruction.result {
            ir.values[value.index()].def = ValueDef::Instruction(position);
        }
    }
    match ir.verify() {
        Err(TraceVerifyError::EffectOrder { detail, .. }) => {
            assert_eq!(detail, "the preheader runs once, so it makes no table");
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn making_and_filling_a_table_keep_the_cached_reads_and_bindings_of_the_loop() {
    // Neither writes a table anyone else can name, so a field binding and a
    // forwarded read stay valid beside them.
    let ir = table_making_loop();
    assert!(!ir.moves_hash_storage() && !ir.deletes_keys());
    let (optimized, stats) = optimize_with(&ir, memory_on());
    // The field the loop reads is read once, before the loop.
    assert_eq!(stats.loads_hoisted, 1);
    assert_eq!(preheader_reads(&optimized), 1);
}

// ---- coroutines -------------------------------------------------------------

/// The helper of a resume that gave one integer.
fn resume_one_integer() -> HelperId {
    HelperId::coroutine_resume(1, 0)
}

#[test]
fn an_array_read_gives_a_coroutine_that_a_resume_helper_takes() {
    // `value = resume (threads[k], k)`: the read leaves a lane of the run's
    // bank, typed as a coroutine, and the helper takes it as a value.
    let mut m = Mem::new(1, 1, vec![array_binding(0)]);
    let (threads, key) = (m.tables[0], m.keys[0]);
    let thread = m.load(0, threads, key, TraceType::Thread);
    let value = m.helper(resume_one_integer(), &[thread, key]).unwrap();
    m.out(thread);
    m.out(value);
    let ir = m.finish();
    assert!(ir.body.iter().any(|instruction| matches!(
        instruction.op,
        TraceOp::ArrayLoad {
            expected: TraceType::Thread,
            ..
        }
    )));
    assert_eq!(ir.value_type(thread), Some(TraceType::Thread));
    assert_eq!(ir.value_type(value), Some(TraceType::Integer));
    assert_eq!(ir.helper_calls(), [resume_one_integer()]);
    assert!(ir.resumes());
    assert_eq!(ir.resume_conflict(), None);
    // Both reads are of the array part: no lane table, no field binding.
    assert!(!ir.moves_hash_storage() || ir.storage_a_set_would_move().is_none());
}

#[test]
fn a_resume_helper_writes_the_heap_without_promising_to_keep_it() {
    let info = resume_one_integer().info().unwrap();
    assert_eq!(info.name, "coroutine.resume");
    assert_eq!(info.results, [HelperType::Integer]);
    assert!(info.context && info.required == 1);
    let effects = info.effects;
    // The thread runs any Lua code: it commits an effect in order, moves
    // storage and ends borrows, and may miss after it ran.
    assert!(effects.resumes() && effects.commits_effect() && effects.moves_storage());
    assert!(effects.ends_borrows() && effects.moves_hash_storage());
    assert!(effects.contains(HelperEffects::MAY_EXIT));
    assert!(!effects.deletes_keys());
    // The call checks the facts bindings take from metatables itself.
    assert!(!effects.contains(HelperEffects::WRITES_METATABLE));
    // Every shape lists its values in place: the kinds in base 3.
    let mut next = HelperId::COROUTINE_RESUME_BASE;
    for values in 0..=MAX_RESUME_VALUES {
        for kinds in 0..3_usize.pow(u32::try_from(values).unwrap()) {
            let id = HelperId::coroutine_resume(values, kinds);
            assert_eq!(id.0, next);
            next += 1;
            let info = id.info().unwrap();
            assert_eq!(info.results.len(), values);
            let mut digits = kinds;
            for result in info.results {
                assert_eq!(
                    *result,
                    [HelperType::Integer, HelperType::Float, HelperType::Boolean][digits % 3]
                );
                digits /= 3;
            }
            assert!(!id.is_search_outcome());
        }
    }
}

#[test]
fn a_resume_beside_what_a_thread_could_invalidate_is_refused() {
    let case = |bindings: Vec<TableBinding>, build: &dyn Fn(&mut Mem)| {
        let mut m = Mem::new(2, 1, bindings);
        let (threads, key) = (m.tables[0], m.keys[0]);
        let thread = m.load(0, threads, key, TraceType::Thread);
        m.helper(resume_one_integer(), &[thread, key]);
        build(&mut m);
        m.finish_unverified()
    };
    // The bare case verifies.
    let plain = case(vec![array_binding(0), array_binding(1)], &|_| {});
    plain.verify().unwrap();
    // A field binding addresses storage a rehash moves.
    let field = case(vec![array_binding(0), field_binding(1, b"x")], &|_| {});
    assert!(field.resume_conflict().is_some());
    assert!(matches!(
        field.verify(),
        Err(TraceVerifyError::InvalidEntry { .. })
    ));
    // A table met by lane has a row of slot addresses.
    let lane = case(vec![array_binding(0), array_binding(1)], &|m| {
        let (other, key) = (m.tables[1], m.keys[0]);
        let loaded = m.load(1, other, key, TraceType::Table);
        m.out(loaded);
    });
    assert!(matches!(
        lane.verify(),
        Err(TraceVerifyError::InvalidEntry { .. })
    ));
    // A table set may resize an array the thread reads, and a helper that
    // writes has bindings of its own to rewrite.
    let set = case(vec![array_binding(0), array_binding(1)], &|m| {
        let (other, key) = (m.tables[1], m.keys[0]);
        let value = m.konst(f(1.0));
        m.set(1, other, key, value);
    });
    assert!(set.resume_conflict().is_some());
    assert!(set.verify().is_err());
    let write = case(vec![array_binding(0), array_binding(1)], &|m| {
        let other = m.tables[1];
        let (name, value) = (m.konst(TraceScalar::Nil), m.konst(int(1)));
        m.helper(HelperId::SET_STRING, &[other, name, value]);
    });
    assert!(write.resume_conflict().is_some());
    // An overwrite in place goes to the table's own memory, which the thread
    // reads as it is.
    let store = case(vec![array_binding(0), array_binding(1)], &|m| {
        let (other, key) = (m.tables[1], m.keys[0]);
        let value = m.konst(f(1.0));
        m.store(1, other, key, value);
    });
    assert_eq!(store.resume_conflict(), None);
    // A metatable fact the thread could change is not one the call checks.
    let plain_table = case(vec![array_binding(0), array_binding(1)], &|_| {});
    assert_eq!(plain_table.resume_conflict(), None);
    let header = case(
        vec![
            array_binding(0),
            TableBinding {
                register: 1,
                kind: BindingKind::Header { strict: false },
            },
        ],
        &|_| {},
    );
    assert!(header.resume_conflict().is_some());
}

#[test]
fn a_resume_ends_the_reads_around_it() {
    // Slot 1 of the array is read before and after the call: the thread may
    // have stored there, so the second read is a read of its own.
    let mut m = Mem::new(2, 1, vec![array_binding(0), array_binding(1)]);
    let (numbers, threads, key) = (m.tables[0], m.tables[1], m.keys[0]);
    let first = m.load(0, numbers, key, TraceType::Float);
    m.out(first);
    let thread = m.load(1, threads, key, TraceType::Thread);
    let value = m.helper(resume_one_integer(), &[thread, key]).unwrap();
    m.out(value);
    let second = m.load(0, numbers, key, TraceType::Float);
    m.out(second);
    let ir = m.finish();
    let (optimized, stats) = optimize_with(&ir, memory_on());
    optimized.verify().unwrap();
    assert_eq!((stats.loads_forwarded, stats.loads_merged), (0, 0));
    assert_eq!(
        optimized
            .body
            .iter()
            .filter(|i| matches!(
                i.op,
                TraceOp::ArrayLoad {
                    expected: TraceType::Float,
                    ..
                }
            ))
            .count(),
        2
    );
    assert_eq!(optimized.helper_calls(), [resume_one_integer()]);
    // The words of the array bindings are read again after the call.
    assert!(optimized.sets_tables());
    assert!(optimized.moves_hash_storage());
}

#[test]
fn a_host_call_ends_the_reads_around_it_and_is_never_merged() {
    // The called code may store to any slot: the second read is a read of
    // its own, and two calls with no arguments are two calls.
    let call = HelperId::vm_call(1, 0);
    let mut m = Mem::new(1, 1, vec![array_binding(0)]);
    let (numbers, key) = (m.tables[0], m.keys[0]);
    let first = m.load(0, numbers, key, TraceType::Float);
    m.out(first);
    let one = m.helper(call, &[]).unwrap();
    m.out(one);
    let second = m.load(0, numbers, key, TraceType::Float);
    m.out(second);
    let two = m.helper(call, &[]).unwrap();
    m.out(two);
    let ir = m.finish();
    for level in [OptimizeLevel::Baseline, OptimizeLevel::Optimized] {
        let (optimized, stats) = optimize_trace(&ir, level);
        optimized.verify().unwrap();
        assert_eq!(
            (stats.loads_forwarded, stats.loads_merged),
            (0, 0),
            "{level:?}"
        );
        assert_eq!(optimized.helper_calls(), [call, call], "{level:?}");
        assert!(optimized.preheader.iter().all(|i| !i.op.touches_heap()));
        // The called code may move field slots, so the plain check counts
        // the call, but the run reads the bindings and the rows again after
        // it, so outside the calls nothing moves a field slot.
        assert!(optimized.moves_hash_storage());
        assert!(!optimized.moves_hash_storage_outside_vm_calls());
        assert_eq!(optimized.resume_conflict(), None, "{level:?}");
    }
}

#[test]
fn a_resume_is_never_merged_hoisted_or_dropped() {
    // Two calls with the same arguments run the thread twice, and a call
    // nothing uses still ran it.
    let mut m = Mem::new(1, 1, vec![array_binding(0)]);
    let (threads, key) = (m.tables[0], m.keys[0]);
    let thread = m.load(0, threads, key, TraceType::Thread);
    let first = m.helper(resume_one_integer(), &[thread, key]).unwrap();
    m.helper(resume_one_integer(), &[thread, key]).unwrap();
    m.helper(HelperId::coroutine_resume(0, 0), &[thread]);
    m.out(first);
    let ir = m.finish();
    for level in [OptimizeLevel::Baseline, OptimizeLevel::Optimized] {
        let (optimized, _) = optimize_trace(&ir, level);
        optimized.verify().unwrap();
        assert_eq!(
            optimized.helper_calls(),
            [
                resume_one_integer(),
                resume_one_integer(),
                HelperId::coroutine_resume(0, 0)
            ],
            "{level:?}"
        );
        assert!(optimized.preheader.iter().all(|i| !i.op.touches_heap()));
    }
}

#[test]
fn a_loop_over_threads_keeps_the_read_and_the_call_in_its_body() {
    // The array read of the next thread cannot leave the loop: the call
    // before it may have replaced the slot.
    let mut l = Lp::with(vec![array_binding(1)], 0);
    let (table, key) = (l.table, l.konst(int(1)));
    let guard = l.guard();
    let thread = emit(
        &mut l.b,
        l.pc,
        TraceOp::ArrayLoad {
            binding: 0,
            table,
            key,
            expected: TraceType::Thread,
            past_end: false,
            guard,
        },
        TraceType::Thread,
    );
    let value = l.helper(resume_one_integer(), &[thread, key]).unwrap();
    let as_float = emit(
        &mut l.b,
        l.pc,
        TraceOp::IntToFloat { source: value },
        TraceType::Float,
    );
    l.add(as_float);
    let ir = l.finish();
    let (optimized, stats) = optimize_with(&ir, memory_on());
    assert_eq!(stats.loads_hoisted, 0);
    assert!(optimized.body.iter().any(|i| matches!(
        i.op,
        TraceOp::ArrayLoad {
            expected: TraceType::Thread,
            ..
        }
    )));
    assert_eq!(optimized.helper_calls(), [resume_one_integer()]);
}

#[test]
fn a_store_before_a_resume_stays_for_the_thread_to_see() {
    // The same slot is stored twice with the call between: the first store is
    // not dead, since the thread may read the slot.
    let mut m = Mem::new(2, 1, vec![array_binding(0), array_binding(1)]);
    let (numbers, threads, key) = (m.tables[0], m.tables[1], m.keys[0]);
    let first = m.konst(f(1.0));
    let second = m.konst(f(2.0));
    m.store(0, numbers, key, first);
    let thread = m.load(1, threads, key, TraceType::Thread);
    m.helper(HelperId::coroutine_resume(0, 0), &[thread]);
    m.store(0, numbers, key, second);
    let ir = m.finish();
    for level in [OptimizeLevel::Baseline, OptimizeLevel::Optimized] {
        let (optimized, _) = optimize_trace(&ir, level);
        optimized.verify().unwrap();
        assert_eq!(stores_in(&optimized), 3, "{level:?}");
    }
}

/// The class of metatable 7, which names closure 3 for `__index`.
fn class_binding(register: u16) -> TableBinding {
    TableBinding {
        register,
        kind: BindingKind::MetaClass {
            metatable: 7,
            pins: vec![MetaPin {
                event: b"__index"[..].into(),
                handler: 3,
            }]
            .into(),
            bare: 0,
        },
    }
}

fn meta_field() -> TableField {
    TableField {
        key: META_FIELD_KEY.into(),
        ty: TraceType::Integer,
    }
}

/// A trace of `guards` metatable guards of table 0 with the `calls` helper
/// calls between them, whose first call comes after the first guard.
fn guarded(guards: usize, setmetatable_between: bool) -> TraceIr {
    let mut m = Mem::new(1, 0, vec![array_binding(0), class_binding(0)]);
    m.b.set_fields(vec![meta_field()]);
    let table = m.tables[0];
    for guard in 0..guards {
        if guard > 0 && setmetatable_between {
            let metatable = m.konst(TraceScalar::Nil);
            m.helper(HelperId::SETMETATABLE, &[table, metatable]);
        }
        m.meta_guard(1, table);
    }
    m.finish()
}

fn meta_guards_in(ir: &TraceIr) -> usize {
    ir.preheader
        .iter()
        .chain(&ir.body)
        .filter(|instruction| matches!(instruction.op, TraceOp::MetaGuard { .. }))
        .count()
}

#[test]
fn a_repeated_class_guard_of_one_table_is_one_guard() {
    let ir = guarded(3, false);
    assert_eq!(meta_guards_in(&ir), 3);
    for level in [OptimizeLevel::Baseline, OptimizeLevel::Optimized] {
        let (optimized, stats) = optimize_trace(&ir, level);
        optimized.verify().unwrap();
        assert_eq!(meta_guards_in(&optimized), 1, "{level:?}");
        assert_eq!(stats.guards_removed, 2, "{level:?}");
    }
}

#[test]
fn a_metatable_write_ends_what_a_class_guard_knew() {
    let ir = guarded(3, true);
    // The class binding is no fact a metatable write makes stale: the guard
    // reads the row, which the write makes again.
    assert!(ir.stale_metatable_fact().is_none());
    for level in [OptimizeLevel::Baseline, OptimizeLevel::Optimized] {
        let (optimized, _) = optimize_trace(&ir, level);
        optimized.verify().unwrap();
        assert_eq!(meta_guards_in(&optimized), 3, "{level:?}");
    }
}

#[test]
fn the_verifier_wants_a_class_binding_and_the_metatable_field_for_a_class_guard() {
    assert!(guarded(1, false).verify().is_ok());
    // No field for the row's word.
    let mut ir = guarded(1, false);
    ir.fields.clear();
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::TypeMismatch { .. })
    ));
    // A binding of another kind.
    let mut ir = guarded(1, false);
    for instruction in &mut ir.body {
        if let TraceOp::MetaGuard { binding, .. } = &mut instruction.op {
            *binding = 0;
        }
    }
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::TypeMismatch { .. })
    ));
    // A class that pins nothing, or a field that is no event.
    for pins in [
        Vec::new(),
        vec![MetaPin {
            event: b"index"[..].into(),
            handler: 1,
        }],
    ] {
        let mut ir = guarded(1, false);
        ir.bindings[1].kind = BindingKind::MetaClass {
            metatable: 7,
            pins: pins.into(),
            bare: 0,
        };
        assert!(
            matches!(ir.verify(), Err(TraceVerifyError::InvalidEntry { .. })),
            "{:?}",
            ir.bindings[1]
        );
    }
}

#[test]
fn a_nil_lookup_is_the_guard_of_an_absent_key_only_beside_a_class() {
    let build = |class: bool| {
        let mut bindings = vec![
            TableBinding {
                register: 0,
                kind: BindingKind::Hash,
            },
            array_binding(1),
        ];
        if class {
            bindings.push(class_binding(0));
        }
        let mut m = Mem::new(2, 0, bindings);
        let (table, keys) = (m.tables[0], m.tables[1]);
        let index = m.konst(TraceScalar::Float(1.0));
        let key = m.load(1, keys, index, TraceType::StringRef);
        let guard = m.guard();
        m.b.emit(
            site(m.pc),
            TraceOp::HashLoad {
                binding: 0,
                table,
                key,
                expected: TraceType::Nil,
                guard,
            },
            Some(TraceType::Nil),
        );
        m.finish_unverified()
    };
    // A nil is the access's value only without a metatable.
    assert!(build(false).verify().is_err());
    assert!(build(true).verify().is_ok());
}

#[test]
fn the_reference_leaves_at_a_class_guard_the_table_fails() {
    struct Words(u64);
    impl TraceHeap for Words {
        fn entry_accepts(&mut self, _: &[TableBinding]) -> bool {
            true
        }
        fn array_get(&mut self, _: u32, _: TraceScalar) -> Option<TraceScalar> {
            None
        }
        fn array_set(&mut self, _: u32, _: TraceScalar, _: TraceScalar) -> bool {
            false
        }
        fn field_get(&mut self, _: u32) -> TraceScalar {
            TraceScalar::Nil
        }
        fn field_set(&mut self, _: u32, _: TraceScalar) {}
        fn length(&mut self, _: u32) -> TraceScalar {
            TraceScalar::Nil
        }
        fn table_field_get(&mut self, _: u64, field: u32) -> Option<TraceScalar> {
            assert_eq!(field, 0);
            Some(TraceScalar::Integer(self.0.cast_signed()))
        }
    }
    let ir = guarded(1, false);
    let registers = [TraceScalar::Table(0)];
    for (word, passes) in [(meta_word(7), true), (meta_word(8), false), (0, false)] {
        let evaluation = evaluate_with_heap(
            &ir,
            &registers,
            &[],
            1000,
            None,
            &mut Words(word),
            |_, _| None,
        );
        assert_eq!(
            matches!(evaluation.outcome, Outcome::GuardFailed { .. }),
            !passes,
            "word {word}"
        );
    }
}

#[test]
fn lookup_calls_on_a_lane_table_sit_beside_a_store_that_moves_hash_storage() {
    // The lookups of `TABLE_GET_*` read the VM's table on each call and bind
    // no address, so a rehash cannot strand them.
    for read in [
        HelperId::TABLE_GET_INTEGER,
        HelperId::TABLE_GET_TABLE,
        HelperId::TABLE_GET_STRING,
        HelperId::TABLE_GET_USERDATA,
    ] {
        let mut m = Mem::new(1, 0, vec![array_binding(0)]);
        let table = m.tables[0];
        let (key, walk) = (m.konst(int(1)), m.konst(int(0)));
        m.helper(read, &[table, key, walk]);
        set_string(&mut m, int(1));
        let ir = m.finish();
        assert!(ir.moves_hash_storage(), "{read:?}");
        assert!(ir.storage_a_set_would_move().is_none(), "{read:?}");
    }
    // The length of a lane table is such a call too.
    let mut m = Mem::new(1, 0, vec![array_binding(0)]);
    let table = m.tables[0];
    let raw = m.konst(int(1));
    m.helper(HelperId::TABLE_LENGTH, &[table, raw]);
    set_string(&mut m, int(1));
    let ir = m.finish();
    assert!(ir.storage_a_set_would_move().is_none());
}

#[test]
fn a_row_read_of_a_lane_table_is_refused_beside_a_store_that_moves_hash_storage() {
    // The row of a table met by lane holds addresses a rehash moves.
    let mut m = Mem::new(1, 0, vec![array_binding(0)]);
    let table = m.tables[0];
    m.lane_field(table);
    set_string(&mut m, int(1));
    let ir = m.finish_unverified();
    assert!(ir.moves_hash_storage());
    assert_eq!(
        ir.storage_a_set_would_move(),
        Some("a table set beside a table met by lane")
    );
    assert!(ir.verify().is_err());
}

#[test]
fn a_store_through_a_number_key_into_a_lane_table_moves_storage() {
    let effects = HelperId::TABLE_SET_INDEX.effects();
    assert!(effects.contains(HelperEffects::WRITES_HEAP));
    assert!(effects.moves_storage() && effects.moves_hash_storage());
    assert!(effects.ends_borrows());
    assert!(!effects.contains(HelperEffects::REWRITES_ARRAYS));
    // Array words are read again after it, so an array binding may sit beside
    // it; a bound field may not, and neither may the row of a lane.
    let build = |bindings: Vec<TableBinding>, row: bool| {
        let mut m = Mem::new(1, 0, bindings);
        let table = m.tables[0];
        if row {
            m.lane_field(table);
        }
        let (key, value, raw) = (m.konst(int(1)), m.konst(int(7)), m.konst(int(0)));
        m.helper(HelperId::TABLE_SET_INDEX, &[table, key, value, raw]);
        m
    };
    let ir = build(vec![array_binding(0)], false).finish();
    assert!(ir.sets_tables() && ir.moves_hash_storage());
    assert!(ir.storage_a_set_would_move().is_none());
    let ir = build(vec![array_binding(0), field_binding(0, b"x")], false).finish_unverified();
    assert!(ir.storage_a_set_would_move().is_some());
    assert!(ir.verify().is_err());
    let ir = build(vec![array_binding(0)], true).finish_unverified();
    assert!(ir.storage_a_set_would_move().is_some());
    assert!(ir.verify().is_err());
}
