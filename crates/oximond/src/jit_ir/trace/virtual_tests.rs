//! Virtual tables in snapshots: the verifier's rules for recipes, what the
//! reference evaluator materializes, what the optimizer keeps, and the sides
//! a snapshot with a virtual table refuses.

use super::ir::*;
use super::reference::{Outcome, Restored, RestoredTable, VIRTUAL_LANE_BASE, evaluate};
use super::snapshot::*;
use super::verify::TraceVerifyError;
use super::{OptimizeLevel, TreeError, optimize_trace, plan_tree};
use crate::jit_ir::Comparison;
use std::sync::Arc;

fn site(pc: usize) -> TraceSite {
    TraceSite {
        frame: FrameId(0),
        pc,
    }
}

fn f(value: f64) -> TraceScalar {
    TraceScalar::Float(value)
}

fn key(text: &str) -> Arc<[u8]> {
    Arc::from(text.as_bytes())
}

fn nil() -> RegisterEntry {
    RegisterEntry::Const(TraceScalar::Nil)
}

fn recipe(array: Vec<RegisterEntry>, fields: Vec<(VirtualKey, RegisterEntry)>) -> VirtualTable {
    VirtualTable {
        alloc: VirtualAlloc::Plain,
        array,
        fields,
        phantom_entries: None,
        metatable: None,
    }
}

/// The values a case offers its recipes.
struct Values {
    /// A float input (r0).
    x: ValueId,
    /// `x + x`, a float.
    twice: ValueId,
    /// An integer constant, 7.
    seven: ValueId,
}

/// A trace that computes a few values and leaves through an exit whose
/// snapshot has five registers: `r0` holds `x`, and `make` says the rest and
/// the recipes.
fn exit_trace(make: impl FnOnce(&Values) -> (Vec<RegisterEntry>, Vec<VirtualTable>)) -> TraceIr {
    let mut b = TraceBuilder::new(
        TraceId(3),
        TraceFrameInfo {
            register_count: 5,
            code_len: 16,
            inlined: None,
            activation: None,
        },
    );
    for register in 1..5 {
        b.mark_written(FrameId(0), register);
    }
    let x = b.input(0, TraceType::Float);
    let twice = b
        .emit(
            site(1),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: x,
                right: x,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let seven = b
        .emit(
            site(1),
            TraceOp::Constant(TraceScalar::Integer(7)),
            Some(TraceType::Integer),
        )
        .unwrap();
    let (registers, virtuals) = make(&Values { x, twice, seven });
    let exit = b.snapshot(
        Snapshot::new(
            SnapshotPhase::After {
                pc: 1,
                successor: 2,
            },
            0,
            vec![SnapshotFrame {
                frame: FrameId(0),
                last_interpreted_pc: Some(1),
                registers: registers.into(),
            }],
        )
        .with_virtuals(virtuals),
    );
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

fn exited(ir: &TraceIr, x: f64) -> Restored {
    let registers = vec![
        f(x),
        TraceScalar::Nil,
        TraceScalar::Nil,
        TraceScalar::Nil,
        TraceScalar::Nil,
    ];
    match evaluate(ir, &registers, &[], 4, |_, _| None).outcome {
        Outcome::Exited(restored) => restored,
        other => panic!("expected an exit: {other:?}"),
    }
}

fn lane(id: u32) -> TraceScalar {
    TraceScalar::Table(VIRTUAL_LANE_BASE + u64::from(id))
}

fn rejection(ir: &TraceIr) -> &'static str {
    match ir.verify() {
        Err(TraceVerifyError::InvalidSnapshot { detail, .. }) => detail,
        other => panic!("expected an invalid snapshot: {other:?}"),
    }
}

#[test]
fn an_exit_builds_each_named_table_once_with_its_contents() {
    use RegisterEntry::{Const, Value, Virtual};
    let ir = exit_trace(|v| {
        let table = VirtualTable {
            alloc: VirtualAlloc::Shaped {
                keys: vec![key("left"), key("right")],
            },
            array: vec![Value(v.twice), nil(), Const(TraceScalar::Integer(3))],
            fields: vec![
                (VirtualKey::String(key("left")), Value(v.seven)),
                (VirtualKey::String(key("right")), nil()),
                (VirtualKey::Integer(40), Const(TraceScalar::Boolean(true))),
                (VirtualKey::Float(0.5), Value(v.x)),
            ],
            phantom_entries: Some(7),
            metatable: None,
        };
        (
            vec![
                Value(v.x),
                Virtual(VirtualId(0)),
                Virtual(VirtualId(0)),
                Virtual(VirtualId(2)),
                Const(TraceScalar::Nil),
            ],
            vec![
                table,
                recipe(vec![Value(v.x)], vec![]), // named by nothing
                recipe(vec![], vec![]),
            ],
        )
    });
    ir.verify().unwrap();
    let restored = exited(&ir, 1.5);
    // Two registers name one table, so they hold one lane.
    assert_eq!(restored.registers[1], lane(0));
    assert_eq!(restored.registers[1], restored.registers[2]);
    assert_eq!(restored.registers[3], lane(2));
    assert_ne!(restored.registers[1], restored.registers[3]);
    // Only the tables a register names are built, in id order.
    let built: Vec<_> = restored.tables.iter().map(|(id, _)| id.0).collect();
    assert_eq!(built, vec![0, 2]);
    assert_eq!(
        restored.tables[0].1,
        RestoredTable {
            alloc: VirtualAlloc::Shaped {
                keys: vec![key("left"), key("right")],
            },
            // The hole stays a nil for the exit to skip.
            array: vec![f(3.0), TraceScalar::Nil, TraceScalar::Integer(3)],
            fields: vec![
                (VirtualKey::String(key("left")), TraceScalar::Integer(7)),
                (VirtualKey::String(key("right")), TraceScalar::Nil),
                (VirtualKey::Integer(40), TraceScalar::Boolean(true)),
                (VirtualKey::Float(0.5), f(1.5)),
            ],
            metatable: None,
        }
    );
    assert_eq!(restored.tables[1].1.alloc, VirtualAlloc::Plain);
    assert_eq!(
        restored.tables[1].1.array,
        [] as [crate::jit_ir::trace::TraceScalar; 0]
    );
}

#[test]
fn a_recipe_may_hold_a_table_built_inside_it() {
    use RegisterEntry::{Value, Virtual};
    // `{x, {x + x}}`: one register names the outer table, which holds the
    // inner one. The inner table is built because the outer one names it,
    // though no register does.
    let ir = exit_trace(|v| {
        (
            vec![Value(v.x), Virtual(VirtualId(0)), nil(), nil(), nil()],
            vec![
                recipe(
                    vec![Value(v.x), Virtual(VirtualId(1))],
                    vec![(VirtualKey::String(key("inner")), Virtual(VirtualId(1)))],
                ),
                recipe(vec![Value(v.twice)], vec![]),
            ],
        )
    });
    ir.verify().unwrap();
    let restored = exited(&ir, 1.5);
    assert_eq!(restored.registers[1], lane(0));
    let built: Vec<_> = restored.tables.iter().map(|(id, _)| id.0).collect();
    assert_eq!(built, vec![0, 1]);
    // The outer table stores the inner one's lane, twice.
    assert_eq!(restored.tables[0].1.array, vec![f(1.5), lane(1)]);
    assert_eq!(
        restored.tables[0].1.fields,
        vec![(VirtualKey::String(key("inner")), lane(1))]
    );
    assert_eq!(restored.tables[1].1.array, vec![f(3.0)]);
}

#[test]
fn a_recipe_may_hold_root_constants() {
    use RegisterEntry::{Root, Value, Virtual};
    let mut ir = exit_trace(|v| {
        (
            vec![Value(v.x), Virtual(VirtualId(0)), nil(), nil(), nil()],
            vec![recipe(
                vec![Root(0)],
                vec![(VirtualKey::String(key("name")), Root(1))],
            )],
        )
    });
    ir.root_slot_count = 2;
    ir.verify().unwrap();
    let registers = vec![f(1.0); 5];
    let roots = [TraceScalar::String(10), TraceScalar::String(11)];
    let Outcome::Exited(restored) = evaluate(&ir, &registers, &roots, 4, |_, _| None).outcome
    else {
        panic!("expected an exit")
    };
    assert_eq!(restored.tables[0].1.array, vec![TraceScalar::String(10)]);
    assert_eq!(
        restored.tables[0].1.fields,
        vec![(VirtualKey::String(key("name")), TraceScalar::String(11))]
    );
}

#[test]
fn the_verifier_checks_ids_entries_and_values_of_recipes() {
    use RegisterEntry::{Unchanged, Value, Virtual};
    let case = |registers: fn(&Values) -> Vec<RegisterEntry>,
                virtuals: fn(&Values) -> Vec<VirtualTable>| {
        exit_trace(|v| (registers(v), virtuals(v)))
    };
    let named = |v: &Values| vec![Value(v.x), Virtual(VirtualId(0)), nil(), nil(), nil()];

    // A virtual entry needs a recipe.
    let ir = case(named, |_| vec![]);
    assert_eq!(rejection(&ir), "a virtual entry names no recipe");

    // A table inside a table names another recipe of the snapshot, never its
    // own.
    let ir = case(named, |_| vec![recipe(vec![Virtual(VirtualId(0))], vec![])]);
    assert_eq!(rejection(&ir), "a virtual table cannot hold itself");
    let ir = case(named, |_| {
        vec![recipe(
            vec![],
            vec![(VirtualKey::Integer(9), Virtual(VirtualId(1)))],
        )]
    });
    assert_eq!(rejection(&ir), "a virtual entry names no recipe");

    // A recipe has no interpreter register to keep.
    let ir = case(named, |_| vec![recipe(vec![Unchanged], vec![])]);
    assert_eq!(
        rejection(&ir),
        "a virtual table cannot hold an unchanged register"
    );

    // A recipe value must exist at the snapshot.
    let ir = case(named, |_| vec![recipe(vec![Value(ValueId(99))], vec![])]);
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::UnknownValue(ValueId(99)))
    ));

    // A recipe root must exist.
    let ir = case(named, |_| {
        vec![recipe(vec![RegisterEntry::Root(0)], vec![])]
    });
    assert_eq!(rejection(&ir), "root slot is out of range");

    // An opaque constant cannot be rebuilt.
    let ir = case(named, |_| {
        vec![recipe(
            vec![RegisterEntry::Const(TraceScalar::Table(3))],
            vec![],
        )]
    });
    assert_eq!(
        rejection(&ir),
        "an opaque value cannot be a snapshot constant"
    );
}

#[test]
fn a_recipe_value_must_be_defined_before_the_snapshot() {
    use RegisterEntry::{Value, Virtual};
    // A guard snapshot placed before `twice` is computed cannot name it in a
    // recipe, as it cannot in a register.
    let mut b = TraceBuilder::new(
        TraceId(4),
        TraceFrameInfo {
            register_count: 3,
            code_len: 16,
            inlined: None,
            activation: None,
        },
    );
    b.mark_written(FrameId(0), 1);
    let x = b.input(0, TraceType::Float);
    let cond = b
        .emit(
            site(1),
            TraceOp::Compare {
                comparison: Comparison::LessThan,
                left: x,
                right: x,
            },
            Some(TraceType::Boolean),
        )
        .unwrap();
    // The value the recipe wants arrives only after the guard.
    let twice_id = ValueId(2); // x, cond, then this
    let snapshot = b.snapshot(
        Snapshot::new(
            SnapshotPhase::Before { pc: 1 },
            0,
            vec![SnapshotFrame {
                frame: FrameId(0),
                last_interpreted_pc: Some(0),
                registers: vec![Value(x), Virtual(VirtualId(0)), nil()].into(),
            }],
        )
        .with_virtuals(vec![recipe(vec![Value(twice_id)], vec![])]),
    );
    let guard = b.guard(site(1), snapshot);
    b.emit(
        site(1),
        TraceOp::GuardBranch {
            condition: cond,
            expected: false,
            guard,
        },
        None,
    );
    let twice = b
        .emit(
            site(2),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: x,
                right: x,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    assert_eq!(twice, twice_id);
    let exit = b.snapshot(Snapshot::new(
        SnapshotPhase::After {
            pc: 2,
            successor: 3,
        },
        0,
        vec![SnapshotFrame {
            frame: FrameId(0),
            last_interpreted_pc: Some(2),
            registers: vec![Value(x), Value(twice), nil()].into(),
        }],
    ));
    let ir = b.finish(TraceTerminator::Exit { snapshot: exit });
    assert!(
        matches!(
            ir.verify(),
            Err(TraceVerifyError::SnapshotValueUndefined { .. })
        ),
        "{:?}",
        ir.verify()
    );
}

#[test]
fn the_verifier_checks_allocation_and_keys() {
    use RegisterEntry::{Value, Virtual};
    let named = |v: &Values| vec![Value(v.x), Virtual(VirtualId(0)), nil(), nil(), nil()];
    let shaped = |keys: Vec<&str>| VirtualTable {
        alloc: VirtualAlloc::Shaped {
            keys: keys.into_iter().map(key).collect(),
        },
        array: vec![],
        fields: vec![],
        phantom_entries: None,
        metatable: None,
    };
    let nine: Vec<String> = (0..9).map(|i| format!("k{i}")).collect();
    let eight: Vec<String> = (0..8).map(|i| format!("k{i}")).collect();
    let cases: Vec<(VirtualTable, Option<&str>)> = vec![
        (shaped(vec!["a", "b"]), None),
        (shaped(eight.iter().map(String::as_str).collect()), None),
        (
            shaped(nine.iter().map(String::as_str).collect()),
            Some("a shaped virtual table needs one to eight keys"),
        ),
        (
            shaped(vec![]),
            Some("a shaped virtual table needs one to eight keys"),
        ),
        (
            shaped(vec!["a", "a"]),
            Some("a virtual table shape repeats a key"),
        ),
        // Keys inside the array part are positions, not fields.
        (
            recipe(vec![nil(), nil()], vec![(VirtualKey::Integer(2), nil())]),
            Some("a virtual field key lies in the array part"),
        ),
        (
            recipe(vec![nil(), nil()], vec![(VirtualKey::Integer(3), nil())]),
            None,
        ),
        (recipe(vec![], vec![(VirtualKey::Integer(0), nil())]), None),
        (recipe(vec![], vec![(VirtualKey::Integer(-1), nil())]), None),
        (
            recipe(vec![], vec![(VirtualKey::Float(f64::NAN), nil())]),
            Some("a virtual field key is NaN"),
        ),
        (
            recipe(vec![], vec![(VirtualKey::Float(2.0), nil())]),
            Some("an integral float key must be an integer key"),
        ),
        (
            recipe(vec![], vec![(VirtualKey::Float(-0.0), nil())]),
            Some("an integral float key must be an integer key"),
        ),
        (
            recipe(vec![], vec![(VirtualKey::Float(1e300), nil())]),
            None,
        ),
        (
            recipe(vec![], vec![(VirtualKey::Float(f64::INFINITY), nil())]),
            None,
        ),
        (
            recipe(
                vec![],
                vec![
                    (VirtualKey::Float(0.5), nil()),
                    (VirtualKey::Float(0.5), nil()),
                ],
            ),
            Some("a virtual table repeats a field key"),
        ),
        (
            recipe(
                vec![],
                vec![
                    (VirtualKey::String(key("a")), nil()),
                    (VirtualKey::String(key("a")), nil()),
                ],
            ),
            Some("a virtual table repeats a field key"),
        ),
        (
            recipe(
                vec![],
                vec![
                    (VirtualKey::Integer(5), nil()),
                    (VirtualKey::Integer(5), nil()),
                ],
            ),
            Some("a virtual table repeats a field key"),
        ),
    ];
    for (index, (table, expected)) in cases.into_iter().enumerate() {
        let ir = exit_trace(|v| (named(v), vec![table]));
        match expected {
            None => ir
                .verify()
                .unwrap_or_else(|error| panic!("case {index}: {error}")),
            Some(detail) => assert_eq!(rejection(&ir), detail, "case {index}"),
        }
    }
}

#[test]
fn a_cell_or_global_slot_cannot_hold_a_virtual_table() {
    use RegisterEntry::{Value, Virtual};
    // r4 is a window slot beyond the four guest registers.
    let mut ir = exit_trace(|v| {
        (
            vec![Value(v.x), nil(), nil(), nil(), Virtual(VirtualId(0))],
            vec![recipe(vec![], vec![])],
        )
    });
    ir.external = vec![ExternalSlot::Upvalue(0)];
    assert_eq!(
        rejection(&ir),
        "a virtual table cannot sit in a cell or global slot"
    );
}

#[test]
fn the_optimizer_folds_and_keeps_the_values_of_recipes() {
    use RegisterEntry::{Value, Virtual};
    for level in [OptimizeLevel::Baseline, OptimizeLevel::Optimized] {
        let ir = exit_trace(|v| {
            (
                vec![Value(v.x), Virtual(VirtualId(0)), nil(), nil(), nil()],
                vec![recipe(
                    // `twice` is used by nothing but the recipe: it must stay.
                    vec![Value(v.twice), Value(v.seven)],
                    vec![(VirtualKey::Integer(9), Value(v.seven))],
                )],
            )
        });
        ir.verify().unwrap();
        let (optimized, _) = optimize_trace(&ir, level);
        optimized.verify().unwrap();
        // The constant folded into the recipe, the computed value did not.
        let table = &optimized.snapshots[0].virtuals[0];
        assert_eq!(
            table.array[1],
            RegisterEntry::Const(TraceScalar::Integer(7))
        );
        assert_eq!(
            table.fields[0].1,
            RegisterEntry::Const(TraceScalar::Integer(7))
        );
        let RegisterEntry::Value(twice) = table.array[0] else {
            panic!("the computed value stays a value: {table:?}")
        };
        assert!(
            optimized
                .body
                .iter()
                .any(|instruction| instruction.result == Some(twice)),
            "{level:?}: dead code removal dropped a value only a recipe names"
        );
        // The same exit builds the same tables before and after.
        let before = exited(&ir, 2.0);
        let after = exited(&optimized, 2.0);
        assert_eq!(before.tables, after.tables);
        assert_eq!(before.registers, after.registers);
    }
}

#[test]
fn a_recipe_nothing_names_still_keeps_its_values_alive() {
    use RegisterEntry::Value;
    let ir = exit_trace(|v| {
        (
            vec![Value(v.x), nil(), nil(), nil(), nil()],
            vec![recipe(vec![Value(v.twice)], vec![])],
        )
    });
    ir.verify().unwrap();
    let (optimized, _) = optimize_trace(&ir, OptimizeLevel::Optimized);
    optimized.verify().unwrap();
}

#[test]
fn a_snapshot_with_virtual_entries_reports_them() {
    use RegisterEntry::{Value, Virtual};
    let ir = exit_trace(|v| {
        (
            vec![Value(v.x), Virtual(VirtualId(0)), nil(), nil(), nil()],
            vec![recipe(vec![], vec![])],
        )
    });
    assert!(ir.snapshots[0].has_virtual_entry());
    let plain = exit_trace(|v| (vec![Value(v.x), nil(), nil(), nil(), nil()], vec![]));
    assert!(!plain.snapshots[0].has_virtual_entry());
}

// ------------------------------------------------------------------- trees

fn frame4() -> TraceFrameInfo {
    TraceFrameInfo {
        register_count: 4,
        code_len: 16,
        inlined: None,
        activation: None,
    }
}

/// The loop of `tree_tests`, the root's first guard (`pc2`) restoring a
/// virtual table in r3 when `virtual_in_root` is set, and a side continuing
/// that guard.
// The single letters follow the IR's operand names (`i`, `no`, `c`).
#[allow(clippy::many_single_char_names)]
fn tree(virtual_in_root: bool) -> TraceIr {
    use RegisterEntry::{Const, Unchanged, Value, Virtual};
    let snap = |phase, last, registers| {
        Snapshot::new(
            phase,
            0,
            vec![SnapshotFrame {
                frame: FrameId(0),
                last_interpreted_pc: Some(last),
                registers,
            }],
        )
    };
    let mut b = TraceBuilder::new(TraceId(0), frame4());
    for register in [1, 2, 3] {
        b.mark_written(FrameId(0), register);
    }
    let no_in = b.input(1, TraceType::Float);
    let i_in = b.input(2, TraceType::Float);
    let no = b.loop_param(TraceType::Float, LoopParamInit::Input(no_in));
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let seven = b
        .emit(site(0), TraceOp::Constant(f(7.0)), Some(TraceType::Float))
        .unwrap();
    let rem = b
        .emit(
            site(0),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Modulo,
                left: i,
                right: seven,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let three = b
        .emit(site(1), TraceOp::Constant(f(3.0)), Some(TraceType::Float))
        .unwrap();
    let c = b
        .emit(
            site(1),
            TraceOp::Compare {
                comparison: Comparison::LessThan,
                left: rem,
                right: three,
            },
            Some(TraceType::Boolean),
        )
        .unwrap();
    let held = if virtual_in_root {
        Virtual(VirtualId(0))
    } else {
        Value(c)
    };
    let mut first = snap(
        SnapshotPhase::After {
            pc: 2,
            successor: 3,
        },
        2,
        vec![Unchanged, Value(no), Value(i), held].into(),
    );
    if virtual_in_root {
        first = first.with_virtuals(vec![recipe(vec![Value(no)], vec![])]);
    }
    let first = b.snapshot(first);
    let g = b.guard(site(2), first);
    b.emit(
        site(2),
        TraceOp::GuardBranch {
            condition: c,
            expected: false,
            guard: g,
        },
        None,
    );
    let no2 = b
        .emit(
            site(5),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: no,
                right: i,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let one = b
        .emit(site(6), TraceOp::Constant(f(1.0)), Some(TraceType::Float))
        .unwrap();
    let i2 = b
        .emit(
            site(6),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: i,
                right: one,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let top = b
        .emit(site(6), TraceOp::Constant(f(10.0)), Some(TraceType::Float))
        .unwrap();
    let more = b
        .emit(
            site(6),
            TraceOp::Compare {
                comparison: Comparison::LessEqual,
                left: i2,
                right: top,
            },
            Some(TraceType::Boolean),
        )
        .unwrap();
    let after = vec![Unchanged, Value(no2), Value(i2), Value(c)];
    let s = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 6,
            successor: 7,
        },
        6,
        after.clone().into(),
    ));
    let g = b.guard(site(6), s);
    b.emit(
        site(6),
        TraceOp::GuardBranch {
            condition: more,
            expected: true,
            guard: g,
        },
        None,
    );
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        6,
        after.into(),
    ));
    b.set_poll_snapshot(poll);
    let mut root = b.finish(TraceTerminator::LoopBack {
        args: vec![no2, i2],
    });
    root.header_live = vec![0, 1, 2];

    // The side along the other path.
    let mut b = TraceBuilder::new(TraceId(1), frame4());
    for register in [0, 2] {
        b.mark_written(FrameId(0), register);
    }
    let yes = b.input(0, TraceType::Float);
    let no = b.input(1, TraceType::Float);
    let i = b.input(2, TraceType::Float);
    let yes2 = b
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
    let one = b
        .emit(site(6), TraceOp::Constant(f(1.0)), Some(TraceType::Float))
        .unwrap();
    let i2 = b
        .emit(
            site(6),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: i,
                right: one,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let top = b
        .emit(site(6), TraceOp::Constant(f(10.0)), Some(TraceType::Float))
        .unwrap();
    let more = b
        .emit(
            site(6),
            TraceOp::Compare {
                comparison: Comparison::LessEqual,
                left: i2,
                right: top,
            },
            Some(TraceType::Boolean),
        )
        .unwrap();
    let state = vec![Value(yes2), Value(no), Value(i2), Const(TraceScalar::Nil)];
    let s = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 6,
            successor: 7,
        },
        6,
        state.clone().into(),
    ));
    let g = b.guard(site(6), s);
    b.emit(
        site(6),
        TraceOp::GuardBranch {
            condition: more,
            expected: true,
            guard: g,
        },
        None,
    );
    let link = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        6,
        state.into(),
    ));
    let side = b.finish(TraceTerminator::Link { snapshot: link });
    root.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(side),
    });
    root
}

#[test]
fn a_side_of_a_guard_that_restores_a_virtual_table_is_refused() {
    let plain = tree(false);
    plain.verify().unwrap();
    plan_tree(&plain).unwrap();

    let carried = tree(true);
    carried.verify().unwrap();
    assert_eq!(
        plan_tree(&carried).unwrap_err(),
        TreeError::VirtualCarried { side: 0 }
    );
    // Without the side, the root alone plans and optimizes with its virtual
    // table.
    let mut alone = carried.clone();
    alone.sides.clear();
    let members = plan_tree(&alone).unwrap();
    assert_eq!(members.len(), 1);
    let (optimized, _) = optimize_trace(&members[0].ir, OptimizeLevel::Optimized);
    optimized.verify().unwrap();
    assert!(optimized.snapshots.iter().any(Snapshot::has_virtual_entry));
}
