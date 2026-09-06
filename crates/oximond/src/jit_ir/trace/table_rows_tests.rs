//! Field operations on tables a trace meets while it runs
//! ([`TraceOp::TableFieldLoad`] and [`TraceOp::TableFieldStore`]), and array
//! reads that produce tables: the verifier's rules, the reference evaluator's
//! semantics over a heap model, the optimizer's promise never to move, merge
//! or drop one, and how a tree shares the field list.

#![allow(clippy::too_many_lines, clippy::float_cmp)]

use super::ir::*;
use super::optimize::{OptimizeLevel, optimize_trace};
use super::reference::{
    Evaluation, ForcedExit, Outcome, TABLE_STORE_LABEL, TraceHeap, evaluate_with_heap,
};
use super::snapshot::*;
use super::tree::{TreeError, plan_tree};
use super::verify::TraceVerifyError;
use crate::jit_ir::Comparison;

fn site(pc: usize) -> TraceSite {
    TraceSite {
        frame: FrameId(0),
        pc,
    }
}

fn f(value: f64) -> TraceScalar {
    TraceScalar::Float(value)
}

fn emit(b: &mut TraceBuilder, pc: usize, op: TraceOp, ty: TraceType) -> ValueId {
    b.emit(site(pc), op, Some(ty)).unwrap()
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

const X: u32 = 0;
const N: u32 = 1;

fn fields() -> Vec<TableField> {
    vec![
        TableField {
            key: b"x"[..].into(),
            ty: TraceType::Float,
        },
        TableField {
            key: b"n"[..].into(),
            ty: TraceType::Integer,
        },
    ]
}

/// Table 0 is the array; slot `k` of it names table `k + 1`. A table's row
/// has one value per field, `Nil` when it lacks the field.
#[derive(Clone, Debug)]
struct World {
    array: Vec<TraceScalar>,
    rows: Vec<Vec<TraceScalar>>,
}

impl TraceHeap for World {
    fn entry_accepts(&mut self, _: &[TableBinding]) -> bool {
        true
    }
    fn array_get(&mut self, _: u32, key: TraceScalar) -> Option<TraceScalar> {
        let TraceScalar::Float(key) = key else {
            return None;
        };
        let slot = usize::try_from(key as i64 - 1).ok()?;
        self.array.get(slot).copied()
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
    fn table_field_get(&mut self, table: u64, field: u32) -> Option<TraceScalar> {
        let value = self.rows.get(table as usize)?[field as usize];
        (value != TraceScalar::Nil).then_some(value)
    }
    fn table_field_set(&mut self, table: u64, field: u32, value: TraceScalar) -> bool {
        let Some(row) = self.rows.get_mut(table as usize) else {
            return false;
        };
        let held = &mut row[field as usize];
        if *held == TraceScalar::Nil || held.ty() != value.ty() {
            return false;
        }
        *held = value;
        true
    }
}

fn world(count: usize) -> World {
    World {
        array: (1..=count as u64).map(TraceScalar::Table).collect(),
        rows: std::iter::once(vec![TraceScalar::Nil, TraceScalar::Nil])
            .chain((0..count).map(|k| vec![f(k as f64 * 0.5), TraceScalar::Integer(k as i64)]))
            .collect(),
    }
}

/// Registers: r0 index, r1 the array, r2 limit, r3 the table, r4 x, r5 n.
///
/// ```lua
/// repeat
///   t = bodies[i]        -- pc 0
///   x = t.x              -- pc 1
///   t.x = x + 1          -- pc 2
///   n = t.n              -- pc 3
///   t.n = n + 1          -- pc 4
///   i = i + 1            -- pc 5
/// until i > limit        -- pc 6
/// ```
// The single letters follow the names in the listing above.
#[allow(clippy::many_single_char_names)]
fn rows_case() -> TraceIr {
    use RegisterEntry::{Const, Unchanged, Value};
    let nil = Const(TraceScalar::Nil);
    let mut b = TraceBuilder::new(
        TraceId(1),
        TraceFrameInfo {
            register_count: 6,
            code_len: 32,
            inlined: None,
            activation: None,
        },
    );
    for register in [0, 3, 4, 5] {
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
        vec![Value(i), Unchanged, Unchanged, nil, nil, nil],
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
        vec![Value(i), Unchanged, Unchanged, Value(t), nil, nil],
    ));
    let g1 = b.guard(site(1), s1);
    let x = emit(
        &mut b,
        1,
        TraceOp::TableFieldLoad {
            field: X,
            table: t,
            guard: g1,
        },
        TraceType::Float,
    );
    let one = emit(&mut b, 2, TraceOp::Constant(f(1.0)), TraceType::Float);
    let bumped = emit(
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
        SnapshotPhase::Before { pc: 2 },
        0,
        vec![Value(i), Unchanged, Unchanged, Value(t), Value(x), nil],
    ));
    let g2 = b.guard(site(2), s2);
    let e0 = b.next_effect();
    b.emit(
        site(2),
        TraceOp::TableFieldStore {
            field: X,
            table: t,
            value: bumped,
            effect: e0,
            guard: g2,
        },
        None,
    );
    let s3 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 3 },
        1,
        vec![Value(i), Unchanged, Unchanged, Value(t), Value(x), nil],
    ));
    let g3 = b.guard(site(3), s3);
    let n = emit(
        &mut b,
        3,
        TraceOp::TableFieldLoad {
            field: N,
            table: t,
            guard: g3,
        },
        TraceType::Integer,
    );
    let one_int = emit(
        &mut b,
        4,
        TraceOp::Constant(TraceScalar::Integer(1)),
        TraceType::Integer,
    );
    let n_next = emit(
        &mut b,
        4,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: n,
            right: one_int,
        },
        TraceType::Integer,
    );
    let s4 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 4 },
        1,
        vec![Value(i), Unchanged, Unchanged, Value(t), Value(x), Value(n)],
    ));
    let g4 = b.guard(site(4), s4);
    let e1 = b.next_effect();
    b.emit(
        site(4),
        TraceOp::TableFieldStore {
            field: N,
            table: t,
            value: n_next,
            effect: e1,
            guard: g4,
        },
        None,
    );
    let i_next = emit(
        &mut b,
        5,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: i,
            right: one,
        },
        TraceType::Float,
    );
    let more = emit(
        &mut b,
        6,
        TraceOp::Compare {
            comparison: Comparison::LessEqual,
            left: i_next,
            right: limit,
        },
        TraceType::Boolean,
    );
    let after = vec![
        Value(i_next),
        Unchanged,
        Unchanged,
        Value(t),
        Value(x),
        Value(n),
    ];
    let s5 = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 6,
            successor: 7,
        },
        2,
        after.clone(),
    ));
    let g5 = b.guard(site(6), s5);
    b.emit(
        site(6),
        TraceOp::GuardBranch {
            condition: more,
            expected: true,
            guard: g5,
        },
        None,
    );
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        2,
        after,
    ));
    b.set_poll_snapshot(poll);
    b.finish(TraceTerminator::LoopBack { args: vec![i_next] })
}

fn registers(start: f64, limit: f64) -> Vec<TraceScalar> {
    vec![
        f(start),
        TraceScalar::Table(0),
        f(limit),
        TraceScalar::Nil,
        TraceScalar::Nil,
        TraceScalar::Nil,
    ]
}

fn run(ir: &TraceIr, world: &mut World, start: f64, limit: f64) -> Evaluation {
    run_forced(ir, world, start, limit, None)
}

fn run_forced(
    ir: &TraceIr,
    world: &mut World,
    start: f64,
    limit: f64,
    forced: Option<ForcedExit>,
) -> Evaluation {
    evaluate_with_heap(
        ir,
        &registers(start, limit),
        &[],
        100,
        forced,
        world,
        |_, _| None,
    )
}

#[test]
fn the_rows_case_verifies_and_updates_each_table_once() {
    let ir = rows_case();
    ir.verify().unwrap();
    assert_eq!(ir.effect_count(), 2);
    let mut world = world(4);
    let evaluation = run(&ir, &mut world, 1.0, 3.0);
    // The branch fails after the third table's stores.
    assert!(
        matches!(evaluation.outcome, Outcome::GuardFailed { guard, .. } if guard == GuardId(5))
    );
    assert_eq!(evaluation.effects.len(), 6);
    assert!(
        evaluation
            .effects
            .iter()
            .all(|record| record.label == TABLE_STORE_LABEL)
    );
    assert_eq!(world.rows[1], [f(1.0), TraceScalar::Integer(1)]);
    assert_eq!(world.rows[3], [f(2.0), TraceScalar::Integer(3)]);
    // The fourth table is untouched.
    assert_eq!(world.rows[4], [f(1.5), TraceScalar::Integer(3)]);
}

#[test]
fn a_missing_field_or_another_type_leaves_before_the_access() {
    let ir = rows_case();
    for (label, tweak, expected) in [
        (
            "no x",
            Box::new(|w: &mut World| w.rows[2][X as usize] = TraceScalar::Nil)
                as Box<dyn Fn(&mut World)>,
            1,
        ),
        (
            "x is an integer",
            Box::new(|w: &mut World| w.rows[2][X as usize] = TraceScalar::Integer(1)),
            // The load reads it as the wrong type.
            1,
        ),
        (
            "no n",
            Box::new(|w: &mut World| w.rows[2][N as usize] = TraceScalar::Nil),
            3,
        ),
        (
            "slot is nil",
            Box::new(|w: &mut World| w.array[1] = TraceScalar::Nil),
            0,
        ),
    ] {
        let mut world = world(3);
        tweak(&mut world);
        let evaluation = run(&ir, &mut world, 1.0, 3.0);
        assert!(
            matches!(evaluation.outcome, Outcome::GuardFailed { guard, .. } if guard == GuardId(expected)),
            "{label}: {:?}",
            evaluation.outcome
        );
        // The first table was updated by the iteration before it.
        assert_eq!(world.rows[1], [f(1.0), TraceScalar::Integer(1)], "{label}");
    }
}

fn expect_error(mutate: impl FnOnce(&mut TraceIr), check: impl FnOnce(&TraceVerifyError) -> bool) {
    let mut ir = rows_case();
    mutate(&mut ir);
    let error = ir.verify().expect_err("the mutation must not verify");
    assert!(check(&error), "unexpected error: {error}");
}

fn is_type_error(error: &TraceVerifyError) -> bool {
    matches!(error, TraceVerifyError::TypeMismatch { .. })
}

#[test]
fn the_verifier_rejects_malformed_field_operations() {
    // A field the trace does not list.
    expect_error(
        |ir| {
            if let TraceOp::TableFieldLoad { field, .. } = &mut ir.body[1].op {
                *field = 7;
            }
        },
        is_type_error,
    );
    // The receiver is not a table.
    expect_error(
        |ir| {
            if let TraceOp::TableFieldLoad { table, .. } = &mut ir.body[1].op {
                // r2, the limit.
                *table = ValueId(2);
            }
        },
        is_type_error,
    );
    // A store of the other type.
    expect_error(
        |ir| {
            if let TraceOp::TableFieldStore { field, .. } = &mut ir.body[4].op {
                *field = N;
            }
        },
        is_type_error,
    );
    // A field twice, and a field with no payload.
    expect_error(
        |ir| {
            let first = ir.fields[0].clone();
            ir.fields.push(first);
        },
        |error| matches!(error, TraceVerifyError::InvalidEntry { .. }),
    );
    for ty in [TraceType::String, TraceType::Closure, TraceType::Any] {
        expect_error(
            |ir| {
                ir.fields.push(TableField {
                    key: b"other"[..].into(),
                    ty,
                });
            },
            |error| matches!(error, TraceVerifyError::InvalidEntry { .. }),
        );
    }
}

#[test]
fn a_field_of_a_table_or_nil_is_read_but_never_stored() {
    // The read gives a lane or nil, so the field is listed beside the others.
    for ty in [TraceType::Table, TraceType::Nil] {
        let mut ir = rows_case();
        ir.fields.push(TableField {
            key: b"other"[..].into(),
            ty,
        });
        ir.verify().expect("a table or nil field verifies");
        // A store through it would change the slot's type or the table's
        // keys.
        expect_error(
            |ir| {
                ir.fields.push(TableField {
                    key: b"other"[..].into(),
                    ty,
                });
                let last = u32::try_from(ir.fields.len() - 1).unwrap();
                if let TraceOp::TableFieldStore { field, .. } = &mut ir.body[4].op {
                    *field = last;
                }
            },
            is_type_error,
        );
    }
}

#[test]
fn an_array_read_yields_a_table_but_not_a_closure() {
    rows_case().verify().unwrap();
    expect_error(
        |ir| {
            if let TraceOp::ArrayLoad { expected, .. } = &mut ir.body[0].op {
                *expected = TraceType::Closure;
            }
            ir.values[3].ty = TraceType::Closure;
        },
        |_| true,
    );
    // A table is a legal hint for the type of slot 1.
    let mut ir = rows_case();
    ir.bindings[0].kind = BindingKind::Array {
        first: Some(TraceType::Table),
    };
    ir.verify().unwrap();
}

#[test]
fn field_operations_are_never_merged_hoisted_or_dropped() {
    let ir = rows_case();
    for level in [OptimizeLevel::Baseline, OptimizeLevel::Optimized] {
        let (optimized, _) = optimize_trace(&ir, level);
        optimized.verify().unwrap();
        assert!(optimized.preheader.iter().all(|i| !i.op.touches_heap()));
        let order: Vec<&'static str> = optimized
            .body
            .iter()
            .filter_map(|instruction| match instruction.op {
                TraceOp::ArrayLoad { .. } => Some("array"),
                TraceOp::TableFieldLoad { field, .. } if field == X => Some("load x"),
                TraceOp::TableFieldLoad { .. } => Some("load n"),
                TraceOp::TableFieldStore { field, .. } if field == X => Some("store x"),
                TraceOp::TableFieldStore { .. } => Some("store n"),
                _ => None,
            })
            .collect();
        assert_eq!(
            order,
            ["array", "load x", "store x", "load n", "store n"],
            "{level:?}"
        );
        assert_eq!(optimized.fields, ir.fields);
        assert_eq!(optimized.effect_count(), 2);
    }
}

#[test]
fn a_load_nothing_reads_still_leaves_through_its_guard() {
    // The result of the load of `n` is only needed by the store. Dropping the
    // store's operand must not drop the load: it may leave.
    let mut ir = rows_case();
    // Make the load's value unused: store the constant 1 instead.
    let one_int = ValueId(9);
    assert_eq!(ir.value_type(one_int), Some(TraceType::Integer));
    if let TraceOp::TableFieldStore { value, .. } = &mut ir.body[8].op {
        *value = one_int;
    }
    ir.verify().unwrap();
    for level in [OptimizeLevel::Baseline, OptimizeLevel::Optimized] {
        let (optimized, _) = optimize_trace(&ir, level);
        let loads = optimized
            .body
            .iter()
            .filter(|i| matches!(i.op, TraceOp::TableFieldLoad { .. }))
            .count();
        assert_eq!(loads, 2, "{level:?}");
    }
}

#[test]
fn optimized_row_traces_match_the_original_under_every_forced_exit() {
    let ir = rows_case();
    let worlds = [world(4), {
        let mut w = world(4);
        w.rows[2][X as usize] = TraceScalar::Nil;
        w
    }];
    for level in [OptimizeLevel::Baseline, OptimizeLevel::Optimized] {
        let (optimized, _) = optimize_trace(&ir, level);
        for base in &worlds {
            for forced_guard in [None, Some((0, 0)), Some((1, 1)), Some((2, 2)), Some((4, 4))] {
                for iteration in 0..3 {
                    let forced = forced_guard.map(|(pc, guard)| ForcedExit {
                        frame: FrameId(0),
                        pc,
                        guard: Some(GuardId(guard)),
                        iteration,
                    });
                    let (mut a, mut b) = (base.clone(), base.clone());
                    let original = run_forced(&ir, &mut a, 1.0, 4.0, forced);
                    let after = run_forced(&optimized, &mut b, 1.0, 4.0, forced);
                    assert_eq!(original.outcome, after.outcome, "{level:?} {forced:?}");
                    assert_eq!(original.effects, after.effects, "{level:?} {forced:?}");
                    assert_eq!(a.rows, b.rows, "{level:?} {forced:?}");
                }
            }
        }
    }
}

// ------------------------------------------------------------------- trees

// The single letters follow the names in the case listings (`x = t.x`).
#[allow(clippy::many_single_char_names)]
fn side_of(fields: Vec<TableField>) -> TraceIr {
    // A side that continues the branch guard of the case and links back: it
    // does the next iteration's loads of x only.
    use RegisterEntry::{Const, Unchanged, Value};
    let mut b = TraceBuilder::new(
        TraceId(2),
        TraceFrameInfo {
            register_count: 6,
            code_len: 32,
            inlined: None,
            activation: None,
        },
    );
    for register in [0, 3, 4, 5] {
        b.mark_written(FrameId(0), register);
    }
    let i = b.input(0, TraceType::Float);
    let bodies = b.input(1, TraceType::Table);
    let t = b.input(3, TraceType::Table);
    let x_in = b.input(4, TraceType::Float);
    let n_in = b.input(5, TraceType::Integer);
    let _ = (bodies, x_in, n_in);
    b.set_bindings(vec![TableBinding {
        register: 1,
        kind: BindingKind::Array { first: None },
    }]);
    let has_x = !fields.is_empty();
    b.set_fields(fields);
    let nil = Const(TraceScalar::Nil);
    let registers = if has_x {
        let s = b.snapshot(snap(
            SnapshotPhase::Before { pc: 1 },
            0,
            vec![Value(i), Unchanged, Unchanged, Value(t), nil, nil],
        ));
        let g = b.guard(site(1), s);
        let x = emit(
            &mut b,
            1,
            TraceOp::TableFieldLoad {
                field: 0,
                table: t,
                guard: g,
            },
            TraceType::Float,
        );
        vec![Value(i), Unchanged, Unchanged, Value(t), Value(x), nil]
    } else {
        vec![Value(i), Unchanged, Unchanged, Value(t), nil, nil]
    };
    let link = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        0,
        registers,
    ));
    b.finish(TraceTerminator::Link { snapshot: link })
}

#[test]
fn a_tree_shares_the_longest_field_list_with_every_member() {
    let mut longer = fields();
    longer.push(TableField {
        key: b"y"[..].into(),
        ty: TraceType::Boolean,
    });
    // The side names a field the root does not.
    let mut ir = rows_case();
    ir.header_live = vec![0, 1, 2, 3, 4, 5];
    ir.sides.push(TraceSide {
        parent: None,
        guard: 5,
        ir: Box::new(side_of(longer.clone())),
    });
    let planned = plan_tree(&ir).unwrap();
    assert_eq!(planned.len(), 2);
    for member in &planned {
        assert_eq!(member.ir.fields, longer);
    }
    // A side recorded before the root's list grew lists a prefix of it.
    let mut ir = rows_case();
    ir.header_live = vec![0, 1, 2, 3, 4, 5];
    ir.sides.push(TraceSide {
        parent: None,
        guard: 5,
        ir: Box::new(side_of(fields()[..1].to_vec())),
    });
    let planned = plan_tree(&ir).unwrap();
    assert!(planned.iter().all(|member| member.ir.fields == fields()));
}

#[test]
fn a_side_recorded_against_another_field_list_is_refused() {
    let mut ir = rows_case();
    ir.header_live = vec![0, 1, 2, 3, 4, 5];
    let other = vec![TableField {
        key: b"z"[..].into(),
        ty: TraceType::Float,
    }];
    ir.sides.push(TraceSide {
        parent: None,
        guard: 5,
        ir: Box::new(side_of(other)),
    });
    assert!(ir.verify().is_err());
    assert!(matches!(
        plan_tree(&ir),
        Err(TreeError::BindingMismatch { side: 0 })
    ));
}
