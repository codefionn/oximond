//! Range plans of counted loops, and the loop the backend's tests run.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::struct_excessive_bools,
    reason = "small test constants and a builder of one-letter values"
)]

use super::*;
use crate::jit_ir::trace::{
    FrameId, LoopParamInit, RegisterEntry, Snapshot, SnapshotFrame, SnapshotPhase, TableBinding,
    TraceBuilder, TraceFrameInfo, TraceId, TraceSite,
};

/// The shape of a numeric `for` loop as the recorder records it, with knobs
/// for everything a range proof looks at.
///
/// Registers: 0 the control, 1 the visible variable (both loop parameters
/// that take `next`), 2 a float sum, 3 the limit, 4 the step, 5 the table
/// `a` (binding 0), 6 the table `b` (binding 1). The body reads `a[k + r]`
/// for each read offset `r` and adds the elements to the sum, then may store
/// the sum into `b[k + w]` (or `a[k + w]`), steps, and leaves unless the
/// guard on `next` and the limit passes. `k` is the visible variable, or the
/// control with `key_on_control`.
#[derive(Clone, Debug)]
pub(crate) struct LoopSpec {
    pub float: bool,
    /// A constant step, or `None` for the step register.
    pub step: Option<f64>,
    pub subtract: bool,
    pub comparison: Comparison,
    pub next_on_left: bool,
    pub expected: bool,
    /// The guard compares with the sum, which changes, not the limit.
    pub bound_varies: bool,
    pub reads: Vec<i64>,
    pub write: Option<i64>,
    pub write_same_table: bool,
    pub key_on_control: bool,
    /// The first read is `a[k * 2]` instead.
    pub multiplied_key: bool,
    /// A last read `a[k + r4]` (`a[k - r4]` with `true`), shifted by the
    /// step register, an invariant.
    pub shifted: Option<bool>,
    /// The loop tests first, as a `while` loop does: the guard stands at the
    /// top of the body and compares the control itself with the limit.
    pub tests_first: bool,
}

impl LoopSpec {
    /// `for i = start, limit do sum = sum + a[i] + a[i + 1]; b[i] = sum end`.
    pub(crate) fn upward(float: bool) -> Self {
        Self {
            float,
            step: Some(1.0),
            subtract: false,
            comparison: Comparison::LessEqual,
            next_on_left: true,
            expected: true,
            bound_varies: false,
            reads: vec![0, 1],
            write: Some(0),
            write_same_table: false,
            key_on_control: false,
            multiplied_key: false,
            shifted: None,
            tests_first: false,
        }
    }
}

fn emit(b: &mut TraceBuilder, pc: usize, op: TraceOp, ty: TraceType) -> ValueId {
    b.emit(site(pc), op, Some(ty)).unwrap()
}

fn site(pc: usize) -> TraceSite {
    TraceSite {
        frame: FrameId(0),
        pc,
    }
}

/// The trace `spec` describes.
pub(crate) fn counted_loop(spec: &LoopSpec) -> TraceIr {
    use RegisterEntry::{Unchanged, Value};
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
    let mut b = TraceBuilder::new(
        TraceId(7),
        TraceFrameInfo {
            register_count: 7,
            code_len: 32,
            inlined: None,
            activation: None,
        },
    );
    for register in 0..3 {
        b.mark_written(FrameId(0), register);
    }
    let control_in = b.input(0, ty);
    let visible_in = b.input(1, ty);
    let sum_in = b.input(2, TraceType::Float);
    let limit = b.input(3, ty);
    let step_in = b.input(4, ty);
    let a = b.input(5, TraceType::Table);
    let t = b.input(6, TraceType::Table);
    b.set_bindings(vec![
        TableBinding {
            register: 5,
            kind: BindingKind::Array { first: None },
        },
        TableBinding {
            register: 6,
            kind: BindingKind::Array { first: None },
        },
    ]);
    let control = b.loop_param(ty, LoopParamInit::Input(control_in));
    let visible = b.loop_param(ty, LoopParamInit::Input(visible_in));
    let sum = b.loop_param(TraceType::Float, LoopParamInit::Input(sum_in));
    let base = if spec.key_on_control {
        control
    } else {
        visible
    };
    let snapshot = |b: &mut TraceBuilder, pc: usize, last: usize, sum: ValueId| {
        let snapshot = Snapshot::new(
            SnapshotPhase::Before { pc },
            b.effect_position(),
            vec![SnapshotFrame {
                frame: FrameId(0),
                last_interpreted_pc: Some(last),
                registers: vec![
                    Value(control),
                    Value(visible),
                    Value(sum),
                    Unchanged,
                    Unchanged,
                    Unchanged,
                    Unchanged,
                ]
                .into(),
            }],
        );
        b.snapshot(snapshot)
    };
    let keyed = |b: &mut TraceBuilder, pc: usize, offset: i64| {
        if offset == 0 {
            return base;
        }
        let constant = emit(b, pc, TraceOp::Constant(number(offset.abs() as f64)), ty);
        let op = if offset > 0 {
            ArithmeticOp::Add
        } else {
            ArithmeticOp::Subtract
        };
        emit(
            b,
            pc,
            TraceOp::Arithmetic {
                op,
                left: base,
                right: constant,
            },
            ty,
        )
    };
    let mut total = sum;
    let mut pc = 1;
    if spec.tests_first {
        let (left, right) = if spec.next_on_left {
            (control, limit)
        } else {
            (limit, control)
        };
        let more = emit(
            &mut b,
            0,
            TraceOp::Compare {
                comparison: spec.comparison,
                left,
                right,
            },
            TraceType::Boolean,
        );
        let exit = Snapshot::new(
            SnapshotPhase::After {
                pc: 0,
                successor: 30,
            },
            b.effect_position(),
            vec![SnapshotFrame {
                frame: FrameId(0),
                last_interpreted_pc: Some(0),
                registers: vec![
                    Value(control),
                    Value(visible),
                    Value(sum),
                    Unchanged,
                    Unchanged,
                    Unchanged,
                    Unchanged,
                ]
                .into(),
            }],
        );
        let exit = b.snapshot(exit);
        let guard = b.guard(site(0), exit);
        b.emit(
            site(0),
            TraceOp::GuardBranch {
                condition: more,
                expected: spec.expected,
                guard,
            },
            None,
        );
    }
    let mut reads: Vec<Option<i64>> = spec.reads.iter().copied().map(Some).collect();
    if spec.shifted.is_some() {
        reads.push(None);
    }
    for (index, offset) in reads.iter().enumerate() {
        let key = if let (None, Some(negated)) = (offset, spec.shifted) {
            emit(
                &mut b,
                pc,
                TraceOp::Arithmetic {
                    op: if negated {
                        ArithmeticOp::Subtract
                    } else {
                        ArithmeticOp::Add
                    },
                    left: base,
                    right: step_in,
                },
                ty,
            )
        } else if index == 0 && spec.multiplied_key {
            let two = emit(&mut b, pc, TraceOp::Constant(number(2.0)), ty);
            emit(
                &mut b,
                pc,
                TraceOp::Arithmetic {
                    op: ArithmeticOp::Multiply,
                    left: base,
                    right: two,
                },
                ty,
            )
        } else {
            keyed(&mut b, pc, offset.expect("a constant offset"))
        };
        let s = snapshot(&mut b, pc, pc.saturating_sub(1).max(1), total);
        let guard = b.guard(site(pc), s);
        let x = emit(
            &mut b,
            pc,
            TraceOp::ArrayLoad {
                binding: 0,
                table: a,
                key,
                expected: TraceType::Float,
                past_end: false,
                guard,
            },
            TraceType::Float,
        );
        total = emit(
            &mut b,
            pc,
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: total,
                right: x,
            },
            TraceType::Float,
        );
        pc += 1;
    }
    if let Some(offset) = spec.write {
        let key = keyed(&mut b, pc, offset);
        let s = snapshot(&mut b, pc, pc - 1, total);
        let guard = b.guard(site(pc), s);
        let effect = b.next_effect();
        let (binding, table) = if spec.write_same_table {
            (0, a)
        } else {
            (1, t)
        };
        b.emit(
            site(pc),
            TraceOp::ArrayStore {
                binding,
                table,
                key,
                value: total,
                effect,
                guard,
            },
            None,
        );
        pc += 1;
    }
    let step = match spec.step {
        Some(step) => emit(&mut b, pc, TraceOp::Constant(number(step)), ty),
        None => step_in,
    };
    let next = emit(
        &mut b,
        pc,
        TraceOp::Arithmetic {
            op: if spec.subtract {
                ArithmeticOp::Subtract
            } else {
                ArithmeticOp::Add
            },
            left: control,
            right: step,
        },
        ty,
    );
    if spec.tests_first {
        let poll = Snapshot::new(
            SnapshotPhase::After { pc, successor: 0 },
            b.effect_position(),
            vec![SnapshotFrame {
                frame: FrameId(0),
                last_interpreted_pc: Some(pc),
                registers: vec![
                    Value(next),
                    Value(next),
                    Value(total),
                    Unchanged,
                    Unchanged,
                    Unchanged,
                    Unchanged,
                ]
                .into(),
            }],
        );
        let poll = b.snapshot(poll);
        b.set_poll_snapshot(poll);
        return b.finish(TraceTerminator::LoopBack {
            args: vec![next, next, total],
        });
    }
    // A bound that changes: the sum, as a number of the loop's type.
    let bound = if spec.bound_varies {
        if spec.float { total } else { next }
    } else {
        limit
    };
    let (left, right) = if spec.next_on_left {
        (next, bound)
    } else {
        (bound, next)
    };
    let more = emit(
        &mut b,
        pc,
        TraceOp::Compare {
            comparison: spec.comparison,
            left,
            right,
        },
        TraceType::Boolean,
    );
    let exit = Snapshot::new(
        SnapshotPhase::After {
            pc,
            successor: pc + 1,
        },
        b.effect_position(),
        vec![SnapshotFrame {
            frame: FrameId(0),
            last_interpreted_pc: Some(pc - 1),
            registers: vec![
                Value(next),
                Value(visible),
                Value(total),
                Unchanged,
                Unchanged,
                Unchanged,
                Unchanged,
            ]
            .into(),
        }],
    );
    let exit = b.snapshot(exit);
    let guard = b.guard(site(pc), exit);
    b.emit(
        site(pc),
        TraceOp::GuardBranch {
            condition: more,
            expected: spec.expected,
            guard,
        },
        None,
    );
    let poll = Snapshot::new(
        SnapshotPhase::After { pc, successor: 1 },
        b.effect_position(),
        vec![SnapshotFrame {
            frame: FrameId(0),
            last_interpreted_pc: Some(pc),
            registers: vec![
                Value(next),
                Value(next),
                Value(total),
                Unchanged,
                Unchanged,
                Unchanged,
                Unchanged,
            ]
            .into(),
        }],
    );
    let poll = b.snapshot(poll);
    b.set_poll_snapshot(poll);
    b.finish(TraceTerminator::LoopBack {
        args: vec![next, next, total],
    })
}

fn plan(spec: &LoopSpec) -> Option<RangePlan> {
    let ir = counted_loop(spec);
    ir.verify().unwrap();
    RangePlan::of(&ir, false)
}

#[test]
fn a_numeric_for_loop_proves_every_offset_key() {
    for float in [false, true] {
        let spec = LoopSpec::upward(float);
        let ir = counted_loop(&spec);
        let plan = RangePlan::of(&ir, false).expect("a counted loop");
        assert_eq!(plan.inductions.len(), 1);
        let induction = &plan.inductions[0];
        // The control is the root, and the keys use the visible variable.
        assert_eq!(induction.root, 0);
        assert_eq!(induction.params, vec![0, 1]);
        assert!(induction.upward);
        assert_eq!(induction.float, float);
        assert_eq!(induction.step, RangeStep::Constant(1));
        assert_eq!(
            plan.checks,
            vec![
                RangeCheck {
                    binding: 0,
                    induction: 0,
                    low: 0,
                    high: 1,
                    shift: None,
                },
                RangeCheck {
                    binding: 1,
                    induction: 0,
                    low: 0,
                    high: 0,
                    shift: None,
                },
            ]
        );
        assert_eq!(plan.keys.len(), 3);
        assert!(plan.keys.iter().all(|key| key.param == 1));
    }
}

#[test]
fn the_guard_may_name_the_bound_first_or_expect_false() {
    // `limit >= next` passing, and `next > limit` failing, both keep the
    // loop below the limit.
    for (comparison, next_on_left, expected) in [
        (Comparison::LessEqual, false, false),
        (Comparison::LessThan, true, true),
        (Comparison::LessThan, false, false),
    ] {
        let spec = LoopSpec {
            comparison,
            next_on_left,
            expected,
            ..LoopSpec::upward(true)
        };
        let upward = next_on_left == expected;
        if upward {
            let plan = plan(&spec).expect("an upward guard and step");
            assert!(plan.inductions[0].upward);
        } else {
            assert_eq!(plan(&spec), None, "{spec:?}");
        }
    }
}

#[test]
fn a_downward_loop_needs_a_negative_step() {
    let down = LoopSpec {
        step: Some(-1.0),
        next_on_left: false,
        ..LoopSpec::upward(false)
    };
    let plan_down = plan(&down).expect("limit <= next with a negative step");
    assert!(!plan_down.inductions[0].upward);
    assert_eq!(plan_down.inductions[0].step, RangeStep::Constant(-1));
    let subtracted = LoopSpec {
        step: Some(2.0),
        subtract: true,
        ..down.clone()
    };
    assert_eq!(
        plan(&subtracted).expect("a subtracted step").inductions[0].step,
        RangeStep::Constant(-2)
    );
    // A positive step that a downward guard bounds from below proves nothing.
    let mismatched = LoopSpec {
        step: Some(1.0),
        ..down
    };
    assert_eq!(plan(&mismatched), None);
}

#[test]
fn an_invariant_step_is_left_to_the_backend() {
    for subtract in [false, true] {
        let spec = LoopSpec {
            step: None,
            subtract,
            ..LoopSpec::upward(true)
        };
        let plan = plan(&spec).expect("an invariant step");
        assert!(matches!(
            plan.inductions[0].step,
            RangeStep::Invariant { negated, .. } if negated == subtract
        ));
    }
}

#[test]
fn loops_without_a_proof_have_no_plan() {
    let cases = [
        // A float step that is not integral.
        LoopSpec {
            step: Some(0.5),
            ..LoopSpec::upward(true)
        },
        // A zero step.
        LoopSpec {
            step: Some(0.0),
            ..LoopSpec::upward(false)
        },
        // A step too large to keep the sums exact.
        LoopSpec {
            step: Some(2.0_f64.powi(40)),
            ..LoopSpec::upward(true)
        },
        // A bound the loop changes.
        LoopSpec {
            bound_varies: true,
            ..LoopSpec::upward(true)
        },
        // An equality is no bound.
        LoopSpec {
            comparison: Comparison::Equal,
            ..LoopSpec::upward(false)
        },
        // No key the induction forms.
        LoopSpec {
            reads: vec![0],
            write: None,
            multiplied_key: true,
            ..LoopSpec::upward(false)
        },
    ];
    for spec in cases {
        assert_eq!(plan(&spec), None, "{spec:?}");
    }
}

#[test]
fn only_affine_keys_with_small_offsets_are_proven() {
    let spec = LoopSpec {
        reads: vec![0, -3, 1 << 40],
        multiplied_key: false,
        write: Some(2),
        write_same_table: true,
        key_on_control: true,
        ..LoopSpec::upward(false)
    };
    let plan = plan(&spec).unwrap();
    // `a[i + 2^40]` stays checked; the store shares the read's binding.
    assert_eq!(plan.keys.len(), 3);
    assert_eq!(plan.inductions[0].params, vec![0]);
    assert_eq!(
        plan.checks,
        vec![RangeCheck {
            binding: 0,
            induction: 0,
            low: -3,
            high: 2,
            shift: None,
        }]
    );
    let multiplied = LoopSpec {
        multiplied_key: true,
        ..LoopSpec::upward(true)
    };
    let plan = super::RangePlan::of(&counted_loop(&multiplied), false).unwrap();
    assert_eq!(plan.checks[0].low, 1, "only `a[i + 1]` of binding 0");
}

#[test]
fn a_loop_that_tests_first_is_proven_for_a_backend_that_asks() {
    let options = RangeOptions {
        tests_first: true,
        ..RangeOptions::default()
    };
    for float in [false, true] {
        let spec = LoopSpec {
            tests_first: true,
            ..LoopSpec::upward(float)
        };
        let ir = counted_loop(&spec);
        ir.verify().unwrap();
        assert_eq!(RangePlan::of(&ir, false), None, "{spec:?}");
        let plan = RangePlan::with(&ir, options).expect("a loop that tests first");
        assert_eq!(plan.inductions.len(), 1);
        // The guard is the second instruction of the body, after its compare.
        assert_eq!(plan.inductions[0].tested, Some(1));
        assert!(plan.inductions[0].upward);
        assert_eq!(plan.keys.len(), 3, "two reads and the store");
        assert_eq!(
            RangePlan::with(
                &ir,
                RangeOptions {
                    resizes: true,
                    ..options
                }
            ),
            None
        );
        // An access keyed on the induction before the guard may see a value
        // past the bound: nothing of the induction is proven.
        let mut early = ir.clone();
        let read = early
            .body
            .iter()
            .position(|instruction| matches!(instruction.op, TraceOp::ArrayLoad { .. }))
            .unwrap();
        let moved = early.body.remove(read);
        early.body.insert(0, moved);
        assert_eq!(RangePlan::with(&early, options), None, "{spec:?}");
        // A downward loop that tests first.
        let down = LoopSpec {
            step: Some(-1.0),
            next_on_left: false,
            tests_first: true,
            ..LoopSpec::upward(float)
        };
        let plan = RangePlan::with(&counted_loop(&down), options).expect("a downward loop");
        assert!(!plan.inductions[0].upward);
    }
}

/// `spec`'s loop with its store turned into a raw set of the sum (a float),
/// or of `value` when given, through a header binding on the table's
/// register when `header`.
/// A loop no guard bounds (its bound moves, or its guard is an equality)
/// is open for a backend that asks: a constant step of at least one, loads
/// proven with no bound, stores left alone, and nothing for a step that is a
/// register or points down.
#[test]
fn a_loop_without_a_bound_is_open_for_a_backend_that_asks() {
    let open = RangeOptions {
        open: true,
        sets: true,
        tests_first: true,
        ..RangeOptions::default()
    };
    for float in [false, true] {
        let unbounded = [
            LoopSpec {
                bound_varies: true,
                ..LoopSpec::upward(float)
            },
            LoopSpec {
                comparison: Comparison::Equal,
                step: Some(3.0),
                ..LoopSpec::upward(float)
            },
        ];
        for spec in unbounded {
            let ir = counted_loop(&spec);
            ir.verify().unwrap();
            assert_eq!(RangePlan::of(&ir, false), None, "{spec:?}");
            let plan = RangePlan::with(&ir, open).expect("an open induction");
            assert_eq!(plan.inductions.len(), 1, "{spec:?}");
            let induction = &plan.inductions[0];
            assert_eq!(induction.bound, None, "{spec:?}");
            assert!(induction.upward && induction.tested.is_none(), "{spec:?}");
            assert_eq!(induction.float, float);
            assert!(matches!(induction.step, RangeStep::Constant(step) if step >= 1));
            // The loads of `a` are proven, the store to `b` is not.
            assert_eq!(plan.keys.len(), spec.reads.len(), "{spec:?}");
            assert!(plan.keys.iter().all(|key| key.binding == 0), "{spec:?}");
            // No link of a side goes straight to its fast loop.
            assert!(!plan.inductions.iter().any(|held| held.bound.is_some()));
        }
        let closed = [
            // A step in a register.
            LoopSpec {
                bound_varies: true,
                step: None,
                ..LoopSpec::upward(float)
            },
            // A step that points down.
            LoopSpec {
                bound_varies: true,
                step: Some(-1.0),
                ..LoopSpec::upward(float)
            },
            // Keys shifted by a register.
            LoopSpec {
                bound_varies: true,
                shifted: Some(false),
                reads: vec![],
                write: None,
                ..LoopSpec::upward(float)
            },
        ];
        for spec in closed {
            let ir = counted_loop(&spec);
            ir.verify().unwrap();
            assert_eq!(RangePlan::with(&ir, open), None, "{spec:?}");
        }
        // A bounded loop stays bounded.
        let ir = counted_loop(&LoopSpec::upward(float));
        let plan = RangePlan::with(&ir, open).expect("a counted loop");
        assert!(plan.inductions[0].bound.is_some());
    }
}

/// A set keyed on an open induction is not proven, and costs the root its
/// plan as any set without a proof does.
#[test]
fn a_set_on_an_open_induction_has_no_plan() {
    let options = RangeOptions {
        open: true,
        sets: true,
        ..RangeOptions::default()
    };
    for float in [false, true] {
        let bounded = loop_with_a_set(&LoopSpec::upward(float), true, false);
        assert!(RangePlan::with(&bounded, options).is_some());
        let spec = LoopSpec {
            bound_varies: true,
            ..LoopSpec::upward(float)
        };
        let ir = loop_with_a_set(&spec, true, false);
        ir.verify().unwrap();
        assert_eq!(RangePlan::with(&ir, options), None, "{spec:?}");
    }
}

fn loop_with_a_set(spec: &LoopSpec, header: bool, table_value: bool) -> TraceIr {
    let mut ir = counted_loop(spec);
    let store = ir
        .body
        .iter()
        .position(|instruction| matches!(instruction.op, TraceOp::ArrayStore { .. }))
        .expect("the loop stores");
    let TraceOp::ArrayStore {
        binding,
        table,
        key,
        value,
        effect,
        guard,
    } = ir.body[store].op
    else {
        unreachable!()
    };
    ir.body[store].op = TraceOp::ArraySet {
        binding,
        table,
        key,
        value: if table_value { table } else { value },
        effect,
        guard,
    };
    if header {
        let register = ir.bindings[binding as usize].register;
        ir.bindings.push(TableBinding {
            register,
            kind: BindingKind::Header { strict: false },
        });
    }
    ir
}

#[test]
fn a_set_with_a_proven_key_keeps_the_plan_of_a_backend_that_stores_it_in_place() {
    let options = RangeOptions {
        sets: true,
        ..RangeOptions::default()
    };
    let spec = LoopSpec::upward(true);
    let ir = loop_with_a_set(&spec, true, false);
    // The set may resize for a backend that does not ask.
    assert!(ir.sets_tables());
    assert!(!resizes_beside_sets(&ir));
    assert_eq!(RangePlan::of(&ir, ir.sets_tables()), None);
    let plan = RangePlan::with(&ir, options).expect("a proven set");
    let TraceOp::ArraySet { binding, key, .. } = ir
        .body
        .iter()
        .find(|instruction| matches!(instruction.op, TraceOp::ArraySet { .. }))
        .unwrap()
        .op
    else {
        unreachable!()
    };
    assert!(plan.key(binding, key).is_some());
    // No header to store through, a value the backend does not store in
    // place, or a key that is no sum of the induction: no plan at all.
    assert_eq!(
        RangePlan::with(&loop_with_a_set(&spec, false, false), options),
        None
    );
    assert_eq!(
        RangePlan::with(&loop_with_a_set(&spec, true, true), options),
        None
    );
    let mut odd = ir.clone();
    for instruction in &mut odd.body {
        if let TraceOp::ArraySet { key, value, .. } = &mut instruction.op {
            // The sum is a float the loop changes.
            *key = *value;
        }
    }
    assert_eq!(RangePlan::with(&odd, options), None);
    // A set in a loop that tests first, with both options.
    let first = LoopSpec {
        tests_first: true,
        ..LoopSpec::upward(false)
    };
    let both = RangeOptions {
        sets: true,
        tests_first: true,
        ..RangeOptions::default()
    };
    assert!(RangePlan::with(&loop_with_a_set(&first, true, false), both).is_some());
    assert_eq!(
        RangePlan::with(&loop_with_a_set(&first, true, false), options),
        None
    );
}

#[test]
fn a_tree_that_resizes_arrays_has_no_plan() {
    let ir = counted_loop(&LoopSpec::upward(false));
    assert!(RangePlan::of(&ir, false).is_some());
    assert_eq!(RangePlan::of(&ir, true), None);
}

#[test]
fn an_invariant_shift_gets_a_check_of_its_own() {
    for negated in [false, true] {
        let spec = LoopSpec {
            shifted: Some(negated),
            ..LoopSpec::upward(true)
        };
        let plan = plan(&spec).expect("a shifted key");
        let shift = RangeShift {
            value: ValueId(4),
            negated,
        };
        assert_eq!(
            plan.checks
                .iter()
                .filter(|check| check.shift == Some(shift))
                .count(),
            1,
            "{plan:?}"
        );
        assert_eq!(plan.keys.len(), 4);
    }
}
