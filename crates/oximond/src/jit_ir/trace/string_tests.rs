//! String operations in the trace IR: the verifier's rules, and the
//! optimizer's promise never to merge or drop a byte read that has a guard.

use super::helper::HelperId;
use super::ir::*;
use super::optimize::{OptimizeLevel, optimize_trace};
use super::snapshot::*;

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

/// Reads byte `r0` of the string `r1` twice and its length once.
fn trace() -> TraceIr {
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
    let key = b.input(0, TraceType::Float);
    let text = b.input(1, TraceType::String);
    b.set_bindings(vec![TableBinding {
        register: 1,
        kind: BindingKind::Bytes {
            ty: TraceType::Float,
        },
    }]);
    let registers = || {
        vec![
            RegisterEntry::Value(key),
            RegisterEntry::Unchanged,
            RegisterEntry::Const(TraceScalar::Nil),
            RegisterEntry::Const(TraceScalar::Nil),
        ]
    };
    let before = b.snapshot(snap(SnapshotPhase::Before { pc: 0 }, None, registers()));
    let first = b.guard(site(0), before);
    let second = b.guard(site(0), before);
    let length = b
        .emit(
            site(0),
            TraceOp::StringLength {
                binding: 0,
                string: text,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let a = b
        .emit(
            site(0),
            TraceOp::StringByte {
                binding: 0,
                string: text,
                key,
                guard: first,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    b.emit(
        site(0),
        TraceOp::StringByte {
            binding: 0,
            string: text,
            key,
            guard: second,
        },
        Some(TraceType::Float),
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
            RegisterEntry::Value(a),
            RegisterEntry::Value(length),
        ],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

fn rejects(mutate: impl FnOnce(&mut TraceIr), detail: &str) {
    let mut ir = trace();
    ir.verify().expect("the unmutated trace verifies");
    mutate(&mut ir);
    let error = ir.verify().expect_err("the mutated trace must not verify");
    assert!(
        error.to_string().contains(detail),
        "unexpected error: {error}"
    );
}

#[test]
fn the_verifier_checks_the_string_a_binding_resolves() {
    rejects(
        |ir| ir.bindings[0].register = 0,
        "a table binding names no table entry input",
    );
    // The binding's register holds a table, not a string.
    rejects(
        |ir| {
            ir.entry.inputs[1].ty = TraceType::Table;
            ir.values[ir.entry.inputs[1].value.index()].ty = TraceType::Table;
        },
        "a table binding names no table entry input",
    );
    rejects(
        |ir| {
            ir.bindings[0].kind = BindingKind::Bytes {
                ty: TraceType::Boolean,
            }
        },
        "a bytes binding needs a number type",
    );
    rejects(
        |ir| ir.bindings[0].kind = BindingKind::Array { first: None },
        "a table binding names no table entry input",
    );
}

#[test]
fn a_side_may_list_a_binding_of_the_root_it_never_uses_over_a_register_without_a_table() {
    // A side lists the root's bindings first. One whose register holds no
    // table where the side starts (a temporary of the root's iteration) is
    // fine while no operation of the side names it: only the root's entry
    // resolves bindings, and a member writes the register, so the plan
    // demotes the binding or refuses the tree.
    let mut ir = trace();
    ir.bindings.push(TableBinding {
        register: 0,
        kind: BindingKind::Array { first: None },
    });
    ir.verify()
        .expect("an unused binding over a float verifies in a side");
    // Named by an operation, it still needs its table.
    rejects(
        |ir| {
            ir.bindings.insert(
                0,
                TableBinding {
                    register: 0,
                    kind: BindingKind::Array { first: None },
                },
            );
        },
        "a table binding names no table entry input",
    );
}

#[test]
fn byte_reads_are_neither_merged_nor_dropped() {
    let ir = trace();
    let (optimized, _) = optimize_trace(&ir, OptimizeLevel::Optimized);
    optimized.verify().unwrap();
    let reads = |ir: &TraceIr| {
        ir.preheader
            .iter()
            .chain(&ir.body)
            .filter(|instruction| matches!(instruction.op, TraceOp::StringByte { .. }))
            .count()
    };
    // Each carries a guard of its own: the second may only be dropped by a
    // proof the first covers it, which the optimizer does not make.
    assert_eq!(reads(&optimized), 2);
}

/// `#(s .. x)`: the concatenation of the string in r1 and the number in r0,
/// then its length. With `keep`, the exit also names the string, so another
/// reader needs it.
fn concat_length(keep: bool) -> TraceIr {
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
    let number = b.input(0, TraceType::Float);
    let text = b.input(1, TraceType::String);
    let before = b.snapshot(snap(
        SnapshotPhase::Before { pc: 0 },
        None,
        vec![
            RegisterEntry::Value(number),
            RegisterEntry::Value(text),
            RegisterEntry::Const(TraceScalar::Nil),
            RegisterEntry::Const(TraceScalar::Nil),
        ],
    ));
    let made = b.guard(site(0), before);
    let joined = b
        .emit(
            site(0),
            TraceOp::CallHelper {
                helper: HelperId::CONCAT,
                args: vec![text, number],
                effect: None,
                guard: made,
            },
            Some(TraceType::String),
        )
        .unwrap();
    // The length's own snapshot names the string it measures, as the
    // recorder's does.
    let measuring = b.snapshot(snap(
        SnapshotPhase::Before { pc: 1 },
        Some(0),
        vec![
            RegisterEntry::Value(number),
            RegisterEntry::Value(text),
            RegisterEntry::Value(joined),
            RegisterEntry::Const(TraceScalar::Nil),
        ],
    ));
    let measured = b.guard(site(1), measuring);
    let length = b
        .emit(
            site(1),
            TraceOp::HostCall {
                function: HostFunction::StringLength,
                args: vec![joined],
                result: TraceType::Float,
                guard: measured,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 1,
            successor: 2,
        },
        Some(1),
        vec![
            RegisterEntry::Value(number),
            RegisterEntry::Value(text),
            RegisterEntry::Value(length),
            if keep {
                RegisterEntry::Value(joined)
            } else {
                RegisterEntry::Const(TraceScalar::Nil)
            },
        ],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

fn host_functions(ir: &TraceIr) -> Vec<HostFunction> {
    ir.body
        .iter()
        .filter_map(|instruction| match instruction.op {
            TraceOp::HostCall { function, .. } => Some(function),
            _ => None,
        })
        .collect()
}

fn concats(ir: &TraceIr) -> usize {
    ir.body
        .iter()
        .filter(|instruction| {
            matches!(
                instruction.op,
                TraceOp::CallHelper {
                    helper: HelperId::CONCAT,
                    ..
                }
            )
        })
        .count()
}

#[test]
fn the_length_of_a_concatenation_nothing_else_reads_makes_no_string() {
    for level in [OptimizeLevel::Baseline, OptimizeLevel::Optimized] {
        let ir = concat_length(false);
        ir.verify().expect("the trace verifies");
        let (optimized, _) = optimize_trace(&ir, level);
        optimized.verify().unwrap();
        assert_eq!(concats(&optimized), 0, "{level:?}");
        assert_eq!(
            host_functions(&optimized),
            [HostFunction::ConcatLength],
            "{level:?}"
        );
        // The call leaves through the guard the concatenation had, which
        // replays both instructions.
        assert_eq!(optimized.guards.len(), 1);
        assert_eq!(optimized.guards[0].site, site(0));
    }
}

#[test]
fn a_concatenation_another_instruction_reads_keeps_its_string() {
    for level in [OptimizeLevel::Baseline, OptimizeLevel::Optimized] {
        let ir = concat_length(true);
        let (optimized, _) = optimize_trace(&ir, level);
        optimized.verify().unwrap();
        assert_eq!(concats(&optimized), 1, "{level:?}");
        assert_eq!(
            host_functions(&optimized),
            [HostFunction::StringLength],
            "{level:?}"
        );
    }
}

#[test]
fn the_verifier_checks_a_concatenation_length_call() {
    let (mut ir, _) = optimize_trace(&concat_length(false), OptimizeLevel::Optimized);
    ir.verify().unwrap();
    let TraceOp::HostCall { args, .. } = &mut ir.body[0].op else {
        panic!("the call is first");
    };
    args.truncate(1);
    let error = ir.verify().expect_err("a concatenation of one operand");
    assert!(error.to_string().contains("wrong number"), "{error}");
}

/// `local slice = string.sub (r2, r0, r1)` then a byte read of `r2` at `r0`
/// whose guard restores `slice`, then an exit. With `reader` something reads
/// the substring: its length.
fn substring(reader: bool) -> TraceIr {
    let mut b = TraceBuilder::new(
        TraceId(0),
        TraceFrameInfo {
            register_count: 5,
            code_len: 32,
            inlined: None,
            activation: None,
        },
    );
    b.mark_written(FrameId(0), 3);
    b.mark_written(FrameId(0), 4);
    let first = b.input(0, TraceType::Float);
    let last = b.input(1, TraceType::Float);
    let text = b.input(2, TraceType::String);
    b.set_bindings(vec![TableBinding {
        register: 2,
        kind: BindingKind::Bytes {
            ty: TraceType::Float,
        },
    }]);
    let nil = || RegisterEntry::Const(TraceScalar::Nil);
    let unchanged = || {
        vec![
            RegisterEntry::Value(first),
            RegisterEntry::Value(last),
            RegisterEntry::Value(text),
        ]
    };
    let before_sub = b.snapshot(snap(
        SnapshotPhase::Before { pc: 0 },
        None,
        [unchanged(), vec![nil(), nil()]].concat(),
    ));
    let made = b.guard(site(0), before_sub);
    let slice = b
        .emit(
            site(0),
            TraceOp::CallHelper {
                helper: HelperId::STRING_SLICE,
                args: vec![text, first, last],
                effect: None,
                guard: made,
            },
            Some(TraceType::String),
        )
        .unwrap();
    let before_byte = b.snapshot(snap(
        SnapshotPhase::Before { pc: 1 },
        Some(0),
        [unchanged(), vec![RegisterEntry::Value(slice), nil()]].concat(),
    ));
    let read = b.guard(site(1), before_byte);
    let byte = b
        .emit(
            site(1),
            TraceOp::StringByte {
                binding: 0,
                string: text,
                key: first,
                guard: read,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let length = reader.then(|| {
        let measured = b.snapshot(snap(
            SnapshotPhase::Before { pc: 2 },
            Some(1),
            [unchanged(), vec![RegisterEntry::Value(slice), nil()]].concat(),
        ));
        let measured = b.guard(site(2), measured);
        b.emit(
            site(2),
            TraceOp::HostCall {
                function: HostFunction::StringLength,
                args: vec![slice],
                result: TraceType::Float,
                guard: measured,
            },
            Some(TraceType::Float),
        )
        .unwrap()
    });
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 2,
            successor: 3,
        },
        Some(2),
        [
            unchanged(),
            vec![
                RegisterEntry::Value(slice),
                RegisterEntry::Value(length.unwrap_or(byte)),
            ],
        ]
        .concat(),
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

fn slice_calls(ir: &TraceIr) -> usize {
    ir.body
        .iter()
        .filter(|instruction| {
            matches!(
                instruction.op,
                TraceOp::CallHelper {
                    helper: HelperId::STRING_SLICE,
                    ..
                }
            )
        })
        .count()
}

fn slice_entries(ir: &TraceIr) -> usize {
    ir.snapshots
        .iter()
        .flat_map(|snapshot| &snapshot.frames)
        .flat_map(|frame| &frame.registers)
        .filter(|entry| matches!(entry, RegisterEntry::Slice(_)))
        .count()
}

#[test]
fn a_substring_only_snapshots_read_is_never_made() {
    for level in [OptimizeLevel::Baseline, OptimizeLevel::Optimized] {
        let ir = substring(false);
        ir.verify().expect("the trace verifies");
        let (optimized, stats) = optimize_trace(&ir, level);
        optimized.verify().unwrap();
        assert_eq!(slice_calls(&optimized), 0, "{level:?}");
        assert_eq!(stats.folded, 1, "{level:?}");
        // The guard of the byte read and the exit each name it, with the
        // recipe: the source and the two indices.
        assert_eq!(slice_entries(&optimized), 2, "{level:?}");
        for snapshot in &optimized.snapshots {
            for recipe in &snapshot.slices {
                assert!(matches!(recipe.string, RegisterEntry::Value(_)));
                assert!(matches!(recipe.first, RegisterEntry::Value(_)));
                assert!(matches!(recipe.last, RegisterEntry::Value(_)));
            }
        }
        // The call's own guard went with it.
        assert_eq!(optimized.guards.len(), 1, "{level:?}");
    }
}

#[test]
fn a_substring_an_operation_reads_is_made() {
    for level in [OptimizeLevel::Baseline, OptimizeLevel::Optimized] {
        let (optimized, _) = optimize_trace(&substring(true), level);
        optimized.verify().unwrap();
        assert_eq!(slice_calls(&optimized), 1, "{level:?}");
        assert_eq!(slice_entries(&optimized), 0, "{level:?}");
    }
}

#[test]
fn a_substring_a_side_trace_continues_from_is_made() {
    use super::optimize::{OptimizeOptions, TreeWrites, optimize_with_tree};
    let ir = substring(false);
    // The guard of the byte read is the one a side continues.
    let origin = ir.guards[1].origin;
    let writes = TreeWrites::of_members([&ir]).with_continued_guards([(0, origin)]);
    let (optimized, _) = optimize_with_tree(
        &ir,
        OptimizeOptions::for_level(OptimizeLevel::Optimized),
        &writes,
    );
    optimized.verify().unwrap();
    assert_eq!(slice_calls(&optimized), 1);
    assert_eq!(slice_entries(&optimized), 0);
}

#[test]
fn the_verifier_checks_a_substring_recipe() {
    let (ir, _) = optimize_trace(&substring(false), OptimizeLevel::Optimized);
    ir.verify().expect("the optimized trace verifies");
    let broken = |mutate: &dyn Fn(&mut TraceIr), detail: &str| {
        let mut ir = ir.clone();
        mutate(&mut ir);
        let error = ir.verify().expect_err("the mutated trace must not verify");
        assert!(
            error.to_string().contains(detail),
            "unexpected error: {error}"
        );
    };
    let with_slices = |ir: &mut TraceIr, change: &dyn Fn(&mut SliceRecipe)| {
        for snapshot in &mut ir.snapshots {
            snapshot.slices.iter_mut().for_each(change);
        }
    };
    broken(
        &|ir| {
            for snapshot in &mut ir.snapshots {
                snapshot.slices.clear();
            }
        },
        "names no recipe",
    );
    broken(
        &|ir| with_slices(ir, &|recipe| recipe.string = recipe.first),
        "made from a string value",
    );
    broken(
        &|ir| {
            with_slices(ir, &|recipe| {
                recipe.last = RegisterEntry::Const(TraceScalar::Nil);
            });
        },
        "number indices",
    );
}
