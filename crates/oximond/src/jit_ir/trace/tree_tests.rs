//! Trees of traces: planning, verification and the optimizer's guard origins.
//!
//! The loop is the one `oximond::jit`'s tree tests compile:
//!
//! ```text
//! pc0  r3 = r2 % 7
//! pc1  r3 = r3 < 3
//! pc2  JumpIfFalse r3 -> pc5        (the root's recorded path takes the jump)
//! pc3  r0 = r0 + r2                 (the side's path)
//! pc4  Jump -> pc6
//! pc5  r1 = r1 + r2                 (the root's path)
//! pc6  r2 = r2 + 1; r2 <= limit loops back to pc0
//! ```

use super::ir::*;
use super::reference::{Outcome, evaluate, evaluate_tree};
use super::snapshot::*;
use super::{
    OptimizeLevel, TreeError, optimize_trace, plan_tree, verify_grown_tree, verify_side_fit,
};
use crate::jit_ir::Comparison;
use crate::jit_ir::trace::HelperId;

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

fn snap(phase: SnapshotPhase, last: usize, registers: Vec<RegisterEntry>) -> Snapshot {
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

fn constant(b: &mut TraceBuilder, pc: usize, value: f64) -> ValueId {
    b.emit(
        site(pc),
        TraceOp::Constant(f(value)),
        Some(TraceType::Float),
    )
    .unwrap()
}

fn arithmetic(
    b: &mut TraceBuilder,
    pc: usize,
    op: ArithmeticOp,
    left: ValueId,
    right: ValueId,
) -> ValueId {
    b.emit(
        site(pc),
        TraceOp::Arithmetic { op, left, right },
        Some(TraceType::Float),
    )
    .unwrap()
}

fn compare(
    b: &mut TraceBuilder,
    pc: usize,
    comparison: Comparison,
    left: ValueId,
    right: ValueId,
) -> ValueId {
    b.emit(
        site(pc),
        TraceOp::Compare {
            comparison,
            left,
            right,
        },
        Some(TraceType::Boolean),
    )
    .unwrap()
}

fn guard(
    b: &mut TraceBuilder,
    pc: usize,
    condition: ValueId,
    expected: bool,
    phase: SnapshotPhase,
    registers: Vec<RegisterEntry>,
) {
    let snapshot = b.snapshot(snap(phase, pc, registers));
    let id = b.guard(site(pc), snapshot);
    b.emit(
        site(pc),
        TraceOp::GuardBranch {
            condition,
            expected,
            guard: id,
        },
        None,
    );
}

fn root_loop(limit: f64) -> TraceIr {
    root_loop_with(limit, false)
}

/// [`root_loop`], taking r0 (which it never reads) as an entry input of
/// type `Any` when `untyped`: a register that holds nil on some entries and
/// a number on others (see `Recorder::with_lazy`).
fn root_loop_with(limit: f64, untyped: bool) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), frame(4));
    for register in [1, 2, 3] {
        b.mark_written(FrameId(0), register);
    }
    if untyped {
        b.input(0, TraceType::Any);
    }
    let no_in = b.input(1, TraceType::Float);
    let i_in = b.input(2, TraceType::Float);
    let no = b.loop_param(TraceType::Float, LoopParamInit::Input(no_in));
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let seven = constant(&mut b, 0, 7.0);
    let rem = arithmetic(&mut b, 0, ArithmeticOp::Modulo, i, seven);
    let three = constant(&mut b, 1, 3.0);
    let c = compare(&mut b, 1, Comparison::LessThan, rem, three);
    guard(
        &mut b,
        2,
        c,
        false,
        SnapshotPhase::After {
            pc: 2,
            successor: 3,
        },
        vec![
            RegisterEntry::Unchanged,
            RegisterEntry::Value(no),
            RegisterEntry::Value(i),
            RegisterEntry::Value(c),
        ],
    );
    let no2 = arithmetic(&mut b, 5, ArithmeticOp::Add, no, i);
    let one = constant(&mut b, 6, 1.0);
    let i2 = arithmetic(&mut b, 6, ArithmeticOp::Add, i, one);
    let top = constant(&mut b, 6, limit);
    let more = compare(&mut b, 6, Comparison::LessEqual, i2, top);
    let after = vec![
        RegisterEntry::Unchanged,
        RegisterEntry::Value(no2),
        RegisterEntry::Value(i2),
        RegisterEntry::Value(c),
    ];
    guard(
        &mut b,
        6,
        more,
        true,
        SnapshotPhase::After {
            pc: 6,
            successor: 7,
        },
        after.clone(),
    );
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        6,
        after,
    ));
    b.set_poll_snapshot(poll);
    b.finish(TraceTerminator::LoopBack {
        args: vec![no2, i2],
    })
}

/// The side along the other path. `yes_ty` is the type of the value it
/// writes into r0.
fn then_side(limit: f64) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(1), frame(4));
    for register in [0, 2] {
        b.mark_written(FrameId(0), register);
    }
    let yes = b.input(0, TraceType::Float);
    let no = b.input(1, TraceType::Float);
    let i = b.input(2, TraceType::Float);
    let yes2 = arithmetic(&mut b, 3, ArithmeticOp::Add, yes, i);
    let one = constant(&mut b, 6, 1.0);
    let i2 = arithmetic(&mut b, 6, ArithmeticOp::Add, i, one);
    let top = constant(&mut b, 6, limit);
    let more = compare(&mut b, 6, Comparison::LessEqual, i2, top);
    let state = vec![
        RegisterEntry::Value(yes2),
        RegisterEntry::Value(no),
        RegisterEntry::Value(i2),
        RegisterEntry::Const(TraceScalar::Nil),
    ];
    guard(
        &mut b,
        6,
        more,
        true,
        SnapshotPhase::After {
            pc: 6,
            successor: 7,
        },
        state.clone(),
    );
    let link = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        6,
        state,
    ));
    b.finish(TraceTerminator::Link { snapshot: link })
}

fn tree(limit: f64) -> TraceIr {
    let mut ir = root_loop(limit);
    ir.header_live = vec![0, 1, 2];
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(then_side(limit)),
    });
    ir
}

#[test]
fn a_trace_without_sides_plans_as_itself() {
    let ir = root_loop(10.0);
    let members = plan_tree(&ir).unwrap();
    assert_eq!(members.len(), 1);
    assert_eq!(members[0].ir, ir);
    assert_eq!(members[0].parent, None);
}

#[test]
fn a_planned_tree_verifies_member_by_member() {
    let ir = tree(10.0);
    ir.verify().unwrap();
    for member in plan_tree(&ir).unwrap() {
        member.ir.verify().unwrap();
        assert_eq!(member.ir.sides, [] as [crate::jit_ir::trace::TraceSide; 0]);
    }
}

#[test]
fn widening_a_read_only_register_makes_it_a_parameter() {
    // The root reads r0 as an invariant: `no = no + r0`. The side writes r0,
    // so after the side links back the root must read the new value.
    let mut b = TraceBuilder::new(TraceId(0), frame(4));
    for register in [1, 2, 3] {
        b.mark_written(FrameId(0), register);
    }
    let yes_in = b.input(0, TraceType::Float);
    let no_in = b.input(1, TraceType::Float);
    let i_in = b.input(2, TraceType::Float);
    let no = b.loop_param(TraceType::Float, LoopParamInit::Input(no_in));
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let seven = constant(&mut b, 0, 7.0);
    let rem = arithmetic(&mut b, 0, ArithmeticOp::Modulo, i, seven);
    let three = constant(&mut b, 1, 3.0);
    let c = compare(&mut b, 1, Comparison::LessThan, rem, three);
    guard(
        &mut b,
        2,
        c,
        false,
        SnapshotPhase::After {
            pc: 2,
            successor: 3,
        },
        vec![
            RegisterEntry::Unchanged,
            RegisterEntry::Value(no),
            RegisterEntry::Value(i),
            RegisterEntry::Value(c),
        ],
    );
    let no2 = arithmetic(&mut b, 5, ArithmeticOp::Add, no, yes_in);
    let one = constant(&mut b, 6, 1.0);
    let i2 = arithmetic(&mut b, 6, ArithmeticOp::Add, i, one);
    let top = constant(&mut b, 6, 40.0);
    let more = compare(&mut b, 6, Comparison::LessEqual, i2, top);
    let after = vec![
        RegisterEntry::Unchanged,
        RegisterEntry::Value(no2),
        RegisterEntry::Value(i2),
        RegisterEntry::Value(c),
    ];
    guard(
        &mut b,
        6,
        more,
        true,
        SnapshotPhase::After {
            pc: 6,
            successor: 7,
        },
        after.clone(),
    );
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        6,
        after,
    ));
    b.set_poll_snapshot(poll);
    let mut ir = b.finish(TraceTerminator::LoopBack {
        args: vec![no2, i2],
    });
    ir.header_live = vec![0, 1, 2];
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(then_side(40.0)),
    });
    ir.verify().unwrap();

    let members = plan_tree(&ir).unwrap();
    let root = &members[0].ir;
    // r0 was an input and is now also a parameter. Nothing reads the entry
    // value any more: the operation reads the parameter.
    let yes_param = root
        .loop_params
        .iter()
        .find(|param| param.register == Some(0))
        .expect("r0 is a loop parameter");
    let add = root
        .body
        .iter()
        .find_map(|instruction| match instruction.op {
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left,
                right,
            } if instruction.site.pc == 5 => Some((left, right)),
            _ => None,
        })
        .unwrap();
    assert_eq!(add.1, yes_param.value);
    root.verify().unwrap();

    // The tree computes what the interpreter does: with r0 = 1.0 the root
    // path adds r0, whose value the side advances by `i`.
    let registers = [f(1.0), f(0.0), f(1.0), TraceScalar::Nil];
    let evaluation = evaluate_tree(&ir, &registers, &[], 1_000);
    let Outcome::GuardFailed { restored, .. } = evaluation.outcome else {
        panic!("{:?}", evaluation.outcome);
    };
    // Whatever it computed, a widened root evaluates to the same state when it
    // runs the tree's work itself: compare against the root evaluated alone
    // from the same registers while no side runs (the first iteration takes
    // the side, so start where only the root path runs).
    let registers = [f(1.0), f(0.0), f(3.0), TraceScalar::Nil];
    let alone = evaluate(root, &registers, &[], 1_000, |_, _| None);
    let Outcome::GuardFailed {
        restored: alone, ..
    } = alone.outcome
    else {
        panic!("the widened root runs and leaves");
    };
    assert_eq!(alone.registers[0], f(1.0), "nothing changed r0 in the root");
    assert_eq!(restored.registers.len(), 4);
}

/// The root reads r0 as an invariant (`no = no + r0`) and never writes it.
/// The guard at pc 2 sits behind an instruction that writes r0 on the other
/// path only (as a `ForLoop` recorded at the loop's end steps its variable
/// when it goes on), so its snapshot alone holds the new r0. The side
/// neither writes r0 nor changes it, and links back with the value it
/// started with: as its input when `reads_r0`, else `Unchanged`.
fn snapshot_only_write_tree(reads_r0: bool) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), frame(4));
    for register in [1, 2, 3] {
        b.mark_written(FrameId(0), register);
    }
    let yes_in = b.input(0, TraceType::Float);
    let no_in = b.input(1, TraceType::Float);
    let i_in = b.input(2, TraceType::Float);
    let no = b.loop_param(TraceType::Float, LoopParamInit::Input(no_in));
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let seven = constant(&mut b, 0, 7.0);
    let rem = arithmetic(&mut b, 0, ArithmeticOp::Modulo, i, seven);
    let three = constant(&mut b, 1, 3.0);
    let c = compare(&mut b, 1, Comparison::LessThan, rem, three);
    let one = constant(&mut b, 2, 1.0);
    let stepped = arithmetic(&mut b, 2, ArithmeticOp::Add, yes_in, one);
    guard(
        &mut b,
        2,
        c,
        false,
        SnapshotPhase::After {
            pc: 2,
            successor: 3,
        },
        vec![
            RegisterEntry::Value(stepped),
            RegisterEntry::Value(no),
            RegisterEntry::Value(i),
            RegisterEntry::Value(c),
        ],
    );
    let no2 = arithmetic(&mut b, 5, ArithmeticOp::Add, no, yes_in);
    let i2 = arithmetic(&mut b, 6, ArithmeticOp::Add, i, one);
    let top = constant(&mut b, 6, 40.0);
    let more = compare(&mut b, 6, Comparison::LessEqual, i2, top);
    let after = vec![
        RegisterEntry::Unchanged,
        RegisterEntry::Value(no2),
        RegisterEntry::Value(i2),
        RegisterEntry::Value(c),
    ];
    guard(
        &mut b,
        6,
        more,
        true,
        SnapshotPhase::After {
            pc: 6,
            successor: 7,
        },
        after.clone(),
    );
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        6,
        after,
    ));
    b.set_poll_snapshot(poll);
    let mut ir = b.finish(TraceTerminator::LoopBack {
        args: vec![no2, i2],
    });
    ir.header_live = vec![0, 1, 2];

    // The side steps `i` and links, with r0 as it found it.
    let mut b = TraceBuilder::new(TraceId(1), frame(4));
    b.mark_written(FrameId(0), 2);
    let yes = if reads_r0 {
        RegisterEntry::Value(b.input(0, TraceType::Float))
    } else {
        RegisterEntry::Unchanged
    };
    let no = b.input(1, TraceType::Float);
    let i = b.input(2, TraceType::Float);
    let one = constant(&mut b, 6, 1.0);
    let i2 = arithmetic(&mut b, 6, ArithmeticOp::Add, i, one);
    let top = constant(&mut b, 6, 40.0);
    let more = compare(&mut b, 6, Comparison::LessEqual, i2, top);
    let state = vec![
        yes,
        RegisterEntry::Value(no),
        RegisterEntry::Value(i2),
        RegisterEntry::Const(TraceScalar::Nil),
    ];
    guard(
        &mut b,
        6,
        more,
        true,
        SnapshotPhase::After {
            pc: 6,
            successor: 7,
        },
        state.clone(),
    );
    let link = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        6,
        state,
    ));
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(b.finish(TraceTerminator::Link { snapshot: link })),
    });
    ir.verify().unwrap();
    ir
}

/// The root of [`snapshot_only_write_tree`] reads r0 through the loop
/// parameter its link carries.
fn assert_r0_carried(members: &[super::tree::TreeMember]) {
    let root = &members[0].ir;
    let carried = root
        .loop_params
        .iter()
        .find(|param| param.register == Some(0))
        .expect("r0 is a loop parameter");
    let add = root
        .body
        .iter()
        .find_map(|instruction| match instruction.op {
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left,
                right,
            } if instruction.site.pc == 5 => Some((left, right)),
            _ => None,
        })
        .unwrap();
    assert_eq!(add.1, carried.value);
    root.verify().unwrap();
}

#[test]
fn a_register_only_a_guards_snapshot_writes_is_carried_through_the_link() {
    let members = plan_tree(&snapshot_only_write_tree(true)).unwrap();
    assert_r0_carried(&members);
}

#[test]
fn a_side_that_never_touches_a_guard_committed_register_links_with_it() {
    // The side has no input for r0, and its link names r0 `Unchanged`: the
    // value the root's guard committed. The plan gives the side that value
    // as an input, typed as the guard names it, and links with it.
    let members = plan_tree(&snapshot_only_write_tree(false)).unwrap();
    assert_r0_carried(&members);
    let side = &members[1].ir;
    let input = side
        .entry
        .inputs
        .iter()
        .find(|input| input.register == 0)
        .expect("the side takes r0");
    assert_eq!(input.ty, TraceType::Float);
    let TraceTerminator::Link { snapshot } = side.terminator else {
        panic!("the side links");
    };
    assert_eq!(
        side.snapshots[snapshot.index()].frames[0].registers[0],
        RegisterEntry::Value(input.value)
    );
    side.verify().unwrap();
}

/// [`tree`] with the root's guard failing inside a call the root inlined
/// (frame 1 in its snapshot, whose register 0 is `callee`), and a side that
/// starts inside that call: it adds the callee's register 0, a frame input of
/// type `ty`, into r0.
fn tree_inside_a_call(callee: RegisterEntry, ty: TraceType) -> TraceIr {
    let mut ir = tree(10.0);
    let snapshot = ir.guards[0].snapshot;
    ir.snapshots[snapshot.index()].frames.push(SnapshotFrame {
        frame: FrameId(1),
        last_interpreted_pc: Some(2),
        registers: vec![callee, RegisterEntry::Unchanged].into(),
    });
    let mut b = TraceBuilder::new(TraceId(1), frame(4));
    b.set_seed_frames(1);
    for register in [0, 2] {
        b.mark_written(FrameId(0), register);
    }
    let yes = b.input(0, TraceType::Float);
    let no = b.input(1, TraceType::Float);
    let i = b.input(2, TraceType::Float);
    let x = b.frame_input(FrameId(1), 0, ty);
    let yes2 = arithmetic(&mut b, 3, ArithmeticOp::Add, yes, x);
    let one = constant(&mut b, 6, 1.0);
    let i2 = arithmetic(&mut b, 6, ArithmeticOp::Add, i, one);
    let state = vec![
        RegisterEntry::Value(yes2),
        RegisterEntry::Value(no),
        RegisterEntry::Value(i2),
        RegisterEntry::Const(TraceScalar::Nil),
    ];
    let link = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        6,
        state,
    ));
    *ir.sides[0].ir = b.finish(TraceTerminator::Link { snapshot: link });
    ir
}

#[test]
fn plan_tree_binds_frame_inputs_from_snapshot_frames_above_the_root_and_refuses_an_untyped_entry() {
    // The callee's register holds the root's float `i` where the guard
    // fails: the side's frame input binds to it.
    let ir = tree(10.0);
    let i = ir.loop_params[1].value;
    let members = plan_tree(&tree_inside_a_call(
        RegisterEntry::Value(i),
        TraceType::Float,
    ))
    .unwrap();
    assert_eq!(members.len(), 2);
    assert_eq!(members[1].ir.seed_frames, 1);
    assert_eq!(
        members[1].ir.frame_inputs,
        [FrameInput {
            frame: FrameId(1),
            register: 0,
            ty: TraceType::Float,
            value: members[1].ir.frame_inputs[0].value,
        }]
    );

    // A frame input of another type than the parent's value.
    assert_eq!(
        plan_tree(&tree_inside_a_call(
            RegisterEntry::Value(i),
            TraceType::Integer
        )),
        Err(TreeError::Unbound {
            side: 0,
            register: 0
        })
    );
    // A callee register the snapshot leaves as the interpreter holds it: the
    // frame does not exist in the interpreter, so nothing types it.
    assert_eq!(
        plan_tree(&tree_inside_a_call(
            RegisterEntry::Unchanged,
            TraceType::Float
        )),
        Err(TreeError::Unbound {
            side: 0,
            register: 0
        })
    );
    // A side that seeds a frame the guard's snapshot does not have.
    let mut ir = tree_inside_a_call(RegisterEntry::Value(i), TraceType::Float);
    ir.sides[0].ir.seed_frames = 2;
    assert_eq!(plan_tree(&ir), Err(TreeError::NoSuchGuard { side: 0 }));
}

#[test]
fn planning_refuses_what_cannot_link() {
    // Already optimized.
    let mut optimized = optimize_trace(&tree(10.0), OptimizeLevel::Optimized).0;
    assert!(
        optimized
            .preheader
            .iter()
            .all(|instruction| instruction.op.guard().is_none())
    );
    optimized.preheader.push(TraceInstruction {
        site: site(0),
        result: None,
        op: TraceOp::Constant(f(0.0)),
    });
    assert_eq!(plan_tree(&optimized), Err(TreeError::NotSource));

    // A side that continues a guard the root does not have.
    let mut ir = tree(10.0);
    ir.sides[0].guard = 9;
    assert_eq!(plan_tree(&ir), Err(TreeError::NoSuchGuard { side: 0 }));

    // A side input typed differently than the parent's value.
    let mut ir = tree(10.0);
    let side = &mut ir.sides[0].ir;
    side.entry.inputs[2].ty = TraceType::Integer;
    let value = side.entry.inputs[2].value;
    side.values[value.index()].ty = TraceType::Integer;
    assert!(matches!(
        plan_tree(&ir),
        Err(TreeError::Unbound {
            side: 0,
            register: 2
        })
    ));

    // The side hands the loop an integer where the root carries a float.
    let mut ir = tree(10.0);
    let TraceTerminator::Link { snapshot: link } = ir.sides[0].ir.terminator else {
        unreachable!()
    };
    ir.sides[0].ir.snapshots[link.index()].frames[0]
        .registers
        .set(2, RegisterEntry::Const(TraceScalar::Integer(0)));
    assert_eq!(
        plan_tree(&ir),
        Err(TreeError::LinkValue {
            side: 0,
            register: 2
        })
    );
}

/// The side reads r1 nowhere, only its snapshots restore it: a side whose
/// recording saw another type there (a nil-or-number local that was nil in
/// the run it watched) takes the number its parent hands over. A slot the
/// side reads still refuses to bind.
#[test]
fn a_side_that_only_passes_a_number_on_takes_its_parents_type() {
    fn retyped(ty: TraceType) -> TraceIr {
        let mut ir = tree(10.0);
        let side = &mut ir.sides[0].ir;
        assert_eq!(side.entry.inputs[1].register, 1);
        side.entry.inputs[1].ty = ty;
        let value = side.entry.inputs[1].value;
        side.values[value.index()].ty = ty;
        ir
    }

    let members = plan_tree(&retyped(TraceType::Nil)).unwrap();
    let side = &members[1].ir;
    assert_eq!(side.entry.inputs[1].ty, TraceType::Float);
    let value = side.entry.inputs[1].value;
    assert_eq!(side.values[value.index()].ty, TraceType::Float);
    for member in &members {
        member.ir.verify().unwrap();
    }
    assert_eq!(
        plan_tree(&retyped(TraceType::Integer)).map(|members| members.len()),
        Ok(2)
    );

    // The side reads the register: its type is the one it computed with.
    let mut ir = retyped(TraceType::Integer);
    let side = &mut ir.sides[0].ir;
    let read = side.entry.inputs[1].value;
    let add = side
        .body
        .iter()
        .position(|instruction| matches!(instruction.op, TraceOp::Arithmetic { .. }))
        .unwrap();
    if let TraceOp::Arithmetic { left, .. } = &mut side.body[add].op {
        *left = read;
    }
    assert!(matches!(
        plan_tree(&ir),
        Err(TreeError::Unbound {
            side: 0,
            register: 1
        })
    ));
}

/// A side input no snapshot names either takes its parent's type whatever
/// the two types are: the recording took it for a temporary it never
/// touched (the register of a table binding the plan demotes), which held
/// what the interpreter last left there. One a snapshot restores keeps the
/// type the side saw unless a number is involved.
#[test]
fn a_side_input_nothing_names_takes_its_parents_type() {
    // The root's exit clears r3, a dead temporary; the side's recording
    // found a table there.
    fn stale(named: bool) -> TraceIr {
        let mut ir = tree(10.0);
        for snapshot in &mut ir.snapshots {
            snapshot.frames[0]
                .registers
                .set(3, RegisterEntry::Const(TraceScalar::Nil));
        }
        let side = &mut ir.sides[0].ir;
        let position = side.entry.inputs.len();
        side.values.push(ValueInfo {
            ty: TraceType::Table,
            def: ValueDef::Input(position),
        });
        let value = ValueId(u32::try_from(side.values.len() - 1).unwrap());
        side.entry.inputs.push(EntryInput {
            register: 3,
            ty: TraceType::Table,
            value,
        });
        if named {
            side.snapshots[0].frames[0]
                .registers
                .set(3, RegisterEntry::Value(value));
        }
        ir
    }

    let members = plan_tree(&stale(false)).unwrap();
    let side = &members[1].ir;
    let input = side
        .entry
        .inputs
        .iter()
        .find(|input| input.register == 3)
        .unwrap();
    assert_eq!(input.ty, TraceType::Nil);
    assert_eq!(side.values[input.value.index()].ty, TraceType::Nil);
    for member in &members {
        member.ir.verify().unwrap();
    }
    // A snapshot restores it: the side keeps the type it saw, and does not
    // bind.
    assert!(matches!(
        plan_tree(&stale(true)),
        Err(TreeError::Unbound {
            side: 0,
            register: 3
        })
    ));
}

/// A side along the other path that writes `written` into r0 without
/// reading it, and links back.
fn writing_side(limit: f64, written: Option<RegisterEntry>) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(1), frame(4));
    for register in [0, 2] {
        b.mark_written(FrameId(0), register);
    }
    let no = b.input(1, TraceType::Float);
    let i = b.input(2, TraceType::Float);
    let one = constant(&mut b, 6, 1.0);
    let i2 = arithmetic(&mut b, 6, ArithmeticOp::Add, i, one);
    let top = constant(&mut b, 6, limit);
    let more = compare(&mut b, 6, Comparison::LessEqual, i2, top);
    let state = vec![
        written.unwrap_or(RegisterEntry::Value(i2)),
        RegisterEntry::Value(no),
        RegisterEntry::Value(i2),
        RegisterEntry::Const(TraceScalar::Nil),
    ];
    guard(
        &mut b,
        6,
        more,
        true,
        SnapshotPhase::After {
            pc: 6,
            successor: 7,
        },
        state.clone(),
    );
    let link = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        6,
        state,
    ));
    b.finish(TraceTerminator::Link { snapshot: link })
}

fn untyped_tree(written: Option<RegisterEntry>) -> TraceIr {
    let mut ir = root_loop_with(10.0, true);
    ir.header_live = vec![0, 1, 2];
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(writing_side(10.0, written)),
    });
    ir
}

/// A side that writes a register the root takes untyped hands it over
/// through the root's input slot: the root keeps the input of type `Any`
/// and carries no loop parameter for it, whichever of nil or a number the
/// side writes, and the slot is one the run leaves for the VM.
#[test]
fn a_linking_side_writes_a_register_the_root_takes_untyped_through_its_slot() {
    for written in [None, Some(RegisterEntry::Const(TraceScalar::Nil))] {
        let members = plan_tree(&untyped_tree(written)).unwrap();
        let root = &members[0].ir;
        assert!(
            root.loop_params
                .iter()
                .all(|param| param.register != Some(0)),
            "{written:?}"
        );
        assert_eq!(root.entry.inputs[0].register, 0);
        assert_eq!(root.entry.inputs[0].ty, TraceType::Any);
        for member in &members {
            member.ir.verify().unwrap();
        }
        let lowered: Vec<TraceIr> = members.into_iter().map(|member| member.ir).collect();
        assert_eq!(super::untyped_link_writes(&lowered), vec![(0, 0)]);
    }
    // A root that takes no register untyped has no such slot.
    let members = plan_tree(&tree(10.0)).unwrap();
    let lowered: Vec<TraceIr> = members.into_iter().map(|member| member.ir).collect();
    assert_eq!(super::untyped_link_writes(&lowered), Vec::new());
}

/// A value with no tag of its own (a string) has no slot form, and a side
/// compiled apart links through the run, which writes no such slot: both
/// refuse to link.
#[test]
fn a_side_cannot_hand_an_untyped_register_a_string_or_link_apart() {
    assert_eq!(
        plan_tree(&untyped_tree(Some(RegisterEntry::Root(0)))).map(|members| members.len()),
        Err(TreeError::LinkValue {
            side: 0,
            register: 0
        })
    );
    assert_eq!(
        super::plan_tree_with(&untyped_tree(None), &[], &[0]).map(|members| members.len()),
        Err(TreeError::LinkValue {
            side: 0,
            register: 0
        })
    );
    assert_eq!(
        super::apart_link_writes(&untyped_tree(None), 0),
        Err(TreeError::LinkValue {
            side: 0,
            register: 0
        })
    );
}

/// A side that takes a register untyped where the root reads nothing (the
/// root was recorded down a path that never read it) and a side that
/// writes it: the root takes the register untyped too, so both bind.
#[test]
fn a_root_takes_a_register_untyped_that_one_side_reads_untyped_and_another_writes() {
    let mut ir = root_loop(10.0);
    ir.header_live = vec![0, 1, 2];
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(writing_side(10.0, None)),
    });
    let plain = plan_tree(&ir).unwrap();
    assert_eq!(plain[0].ir.loop_params.len(), 3, "r0 is carried typed");
    let mut reader = writing_side(10.0, None);
    reader
        .written_registers
        .retain(|(_, register)| *register != 0);
    let any = reader.values.len();
    reader.values.push(ValueInfo {
        ty: TraceType::Any,
        def: ValueDef::Input(reader.entry.inputs.len()),
    });
    reader.entry.inputs.push(EntryInput {
        register: 0,
        ty: TraceType::Any,
        value: ValueId(u32::try_from(any).unwrap()),
    });
    for snapshot in &mut reader.snapshots {
        snapshot.frames[0]
            .registers
            .set(0, RegisterEntry::Unchanged);
    }
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(reader),
    });
    let members = plan_tree(&ir).unwrap();
    let root = &members[0].ir;
    assert_eq!(root.loop_params.len(), 2);
    assert!(
        root.entry
            .inputs
            .iter()
            .any(|input| input.register == 0 && input.ty == TraceType::Any)
    );
    for member in &members {
        member.ir.verify().unwrap();
    }
}

#[test]
fn two_sides_that_write_a_register_with_different_types_conflict() {
    let mut ir = tree(10.0);
    // A second side along the same guard path writing r0 as an integer.
    let mut other = then_side(10.0);
    let TraceTerminator::Link { snapshot: link } = other.terminator else {
        unreachable!()
    };
    other.snapshots[link.index()].frames[0]
        .registers
        .set(0, RegisterEntry::Const(TraceScalar::Integer(1)));
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(other),
    });
    assert_eq!(plan_tree(&ir), Err(TreeError::TypeConflict { register: 0 }));
}

#[test]
fn a_side_cannot_precede_its_parent_or_have_sides_of_its_own() {
    let mut ir = tree(10.0);
    ir.sides[0].parent = Some(0);
    assert!(ir.verify().is_err());

    let mut ir = tree(10.0);
    let nested = ir.sides[0].clone();
    ir.sides[0].ir.sides.push(nested);
    assert!(ir.verify().is_err());
}

#[test]
fn a_side_shares_the_roots_window() {
    let mut ir = tree(10.0);
    ir.sides[0].ir.frames[0].register_count = 5;
    // The side's own snapshots no longer match its frame either; either
    // failure refuses it.
    assert!(ir.verify().is_err());

    let mut ir = tree(10.0);
    ir.sides[0].ir.root_slot_count = 3;
    assert!(ir.verify().is_err());
}

#[test]
fn a_side_may_grow_the_roots_bank_of_strings() {
    use std::sync::Arc;
    let bank = |ir: &mut TraceIr, texts: &[&str]| {
        ir.root_constants = texts
            .iter()
            .map(|text| Arc::from(text.as_bytes()))
            .collect();
        ir.root_slot_count = u32::try_from(texts.len()).unwrap();
    };
    // The side names a string the root does not: the tree has to hold it.
    let mut ir = tree(10.0);
    bank(&mut ir.sides[0].ir, &["a", "b"]);
    assert!(ir.verify().is_err());
    bank(&mut ir, &["a", "b"]);
    ir.verify().unwrap();
    // A side that knows fewer of them is a prefix of the bank.
    bank(&mut ir.sides[0].ir, &["a"]);
    ir.verify().unwrap();
    // Another string in the same slot is another bank.
    bank(&mut ir.sides[0].ir, &["c"]);
    assert!(ir.verify().is_err());
}

#[test]
fn a_linking_trace_has_no_loop_and_no_preheader() {
    let mut side = then_side(10.0);
    side.verify().unwrap();
    side.loop_params.push(LoopParam {
        value: ValueId(0),
        ty: TraceType::Float,
        init: LoopParamInit::Const(f(0.0)),
        register: None,
    });
    assert!(side.verify().is_err());
}

#[test]
fn a_link_snapshot_resumes_after_an_instruction_in_the_root_frame() {
    let mut side = then_side(10.0);
    let TraceTerminator::Link { snapshot: link } = side.terminator else {
        unreachable!()
    };
    side.snapshots[link.index()].phase = SnapshotPhase::Before { pc: 0 };
    assert!(side.verify().is_err());
}

#[test]
fn a_side_must_continue_a_guard_of_its_parent() {
    // Origin 1 is the loop's latch guard, a guard of the body, so a side may
    // continue it. An origin the parent has no guard for cannot be.
    let mut ir = tree(10.0);
    ir.sides[0].guard = 1;
    ir.verify().unwrap();
    ir.sides[0].guard = 2;
    assert!(ir.verify().is_err());
}

#[test]
fn guards_keep_their_origin_through_optimization() {
    let ir = root_loop(10.0);
    assert_eq!(
        ir.guards.iter().map(|g| g.origin).collect::<Vec<_>>(),
        vec![0, 1]
    );
    for level in [OptimizeLevel::Baseline, OptimizeLevel::Optimized] {
        let (optimized, _) = optimize_trace(&ir, level);
        let origins: Vec<u32> = optimized.guards.iter().map(|g| g.origin).collect();
        assert_eq!(origins, vec![0, 1], "{level:?}");
    }
}

#[test]
fn an_optimizer_that_removes_a_guard_leaves_the_others_their_origin() {
    // `1 < 2` is constant, so folding removes the first guard. The second
    // keeps origin 1 although its id is now 0.
    let mut b = TraceBuilder::new(TraceId(0), frame(4));
    b.mark_written(FrameId(0), 2);
    let i_in = b.input(2, TraceType::Float);
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let one = constant(&mut b, 0, 1.0);
    let two = constant(&mut b, 0, 2.0);
    let always = compare(&mut b, 1, Comparison::LessThan, one, two);
    guard(
        &mut b,
        2,
        always,
        true,
        SnapshotPhase::After {
            pc: 2,
            successor: 3,
        },
        vec![
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
            RegisterEntry::Value(i),
            RegisterEntry::Value(always),
        ],
    );
    let i2 = arithmetic(&mut b, 6, ArithmeticOp::Add, i, one);
    let top = constant(&mut b, 6, 10.0);
    let more = compare(&mut b, 6, Comparison::LessEqual, i2, top);
    let after = vec![
        RegisterEntry::Unchanged,
        RegisterEntry::Unchanged,
        RegisterEntry::Value(i2),
        RegisterEntry::Value(always),
    ];
    guard(
        &mut b,
        6,
        more,
        true,
        SnapshotPhase::After {
            pc: 6,
            successor: 7,
        },
        after.clone(),
    );
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        6,
        after,
    ));
    b.set_poll_snapshot(poll);
    let ir = b.finish(TraceTerminator::LoopBack { args: vec![i2] });
    let (optimized, stats) = optimize_trace(&ir, OptimizeLevel::Optimized);
    assert!(stats.guards_removed >= 1);
    assert_eq!(optimized.guards.len(), 1);
    assert_eq!(optimized.guards[0].origin, 1);
}

#[test]
fn a_linking_side_survives_every_optimizer_level() {
    for level in [OptimizeLevel::Baseline, OptimizeLevel::Optimized] {
        let (optimized, _) = optimize_trace(&then_side(10.0), level);
        assert!(matches!(optimized.terminator, TraceTerminator::Link { .. }));
        optimized.verify().unwrap();
        assert!(optimized.preheader.is_empty(), "a side has no loop");
    }
}

#[test]
fn the_tree_evaluator_follows_a_side_and_rejoins_the_loop() {
    let ir = tree(20.0);
    let registers = [f(0.0), f(0.0), f(1.0), TraceScalar::Nil];
    let evaluation = evaluate_tree(&ir, &registers, &[], 10_000);
    assert!(evaluation.side_entries > 0);
    let Outcome::GuardFailed { restored, .. } = evaluation.outcome else {
        panic!("{:?}", evaluation.outcome);
    };
    // The loop ends past i = 20. `yes` took the i with i % 7 < 3, `no` the
    // others.
    let (mut yes, mut no) = (0.0, 0.0);
    for i in 1..=20 {
        if f64::from(i % 7) < 3.0 {
            yes += f64::from(i);
        } else {
            no += f64::from(i);
        }
    }
    assert_eq!(restored.registers[0], f(yes));
    assert_eq!(restored.registers[1], f(no));
    assert_eq!(restored.registers[2], f(21.0));
}

#[test]
fn a_register_the_root_names_in_its_poll_is_carried_with_that_value() {
    // The root writes r0 without a loop parameter for it: its guards and its
    // poll name `no2` for r0 (the key variable of a `next` loop the body
    // never reads is such a register). The side writes r0 too. The root's
    // parameter for r0 takes the loop-back value the poll names, so the
    // widened root still verifies and an optimizer may run on it.
    let mut b = TraceBuilder::new(TraceId(0), frame(4));
    for register in [0, 1, 2, 3] {
        b.mark_written(FrameId(0), register);
    }
    let no_in = b.input(1, TraceType::Float);
    let i_in = b.input(2, TraceType::Float);
    let no = b.loop_param(TraceType::Float, LoopParamInit::Input(no_in));
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let seven = constant(&mut b, 0, 7.0);
    let rem = arithmetic(&mut b, 0, ArithmeticOp::Modulo, i, seven);
    let three = constant(&mut b, 1, 3.0);
    let c = compare(&mut b, 1, Comparison::LessThan, rem, three);
    guard(
        &mut b,
        2,
        c,
        false,
        SnapshotPhase::After {
            pc: 2,
            successor: 3,
        },
        vec![
            RegisterEntry::Value(no),
            RegisterEntry::Value(no),
            RegisterEntry::Value(i),
            RegisterEntry::Value(c),
        ],
    );
    let no2 = arithmetic(&mut b, 5, ArithmeticOp::Add, no, i);
    let one = constant(&mut b, 6, 1.0);
    let i2 = arithmetic(&mut b, 6, ArithmeticOp::Add, i, one);
    let top = constant(&mut b, 6, 40.0);
    let more = compare(&mut b, 6, Comparison::LessEqual, i2, top);
    let after = vec![
        RegisterEntry::Value(no2),
        RegisterEntry::Value(no2),
        RegisterEntry::Value(i2),
        RegisterEntry::Value(c),
    ];
    guard(
        &mut b,
        6,
        more,
        true,
        SnapshotPhase::After {
            pc: 6,
            successor: 7,
        },
        after.clone(),
    );
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        6,
        after,
    ));
    b.set_poll_snapshot(poll);
    let mut ir = b.finish(TraceTerminator::LoopBack {
        args: vec![no2, i2],
    });
    ir.header_live = vec![0, 1, 2];
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(then_side(40.0)),
    });
    ir.verify().unwrap();

    let members = plan_tree(&ir).unwrap();
    let root = &members[0].ir;
    let position = root
        .loop_params
        .iter()
        .position(|param| param.register == Some(0))
        .expect("r0 is a loop parameter");
    let TraceTerminator::LoopBack { args } = &root.terminator else {
        panic!("the root loops");
    };
    assert_ne!(args[position], root.loop_params[position].value);
    root.verify().unwrap();
}

#[test]
fn verify_side_fit_refuses_a_side_with_another_window_as_verify_does() {
    let ir = tree(10.0);
    assert!(verify_side_fit(&ir, 0, true).is_ok());
    assert!(verify_side_fit(&ir, 0, false).is_ok());
    assert!(verify_side_fit(&ir, 1, false).is_err());

    // The side's own IR is not checked without `check_ir`, but its fit with
    // the tree is, and that alone refuses the side just as `verify` does.
    let mut ir = tree(10.0);
    ir.sides[0].ir.root_slot_count = 3;
    assert!(verify_side_fit(&ir, 0, false).is_err());
    assert!(verify_side_fit(&ir, 0, true).is_err());
    assert!(ir.verify().is_err());
}

#[test]
fn verify_grown_tree_refuses_pins_that_leave_an_earlier_side_behind_as_verify_does() {
    // The tree pins [1] and its side [1, 2]. A new side pins [1, 3], so the
    // tree's pins grow to [1, 3] and the first side is a prefix of neither.
    let mut ir = tree(10.0);
    ir.pinned_closures = vec![1];
    ir.sides[0].ir.pinned_closures = vec![1, 2];
    ir.verify().unwrap();

    let mut other = then_side(10.0);
    other.pinned_closures = vec![1, 3];
    let mut grown = ir.clone();
    grown.pinned_closures.clone_from(&other.pinned_closures);
    grown.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(other),
    });
    assert!(grown.verify().is_err());
    assert!(verify_grown_tree(&grown, 1).is_err());
    // The fit of the new side alone cannot see it.
    assert!(verify_side_fit(&grown, 1, false).is_ok());

    // A side that extends the pins keeps the earlier one a prefix.
    let mut extending = then_side(10.0);
    extending.pinned_closures = vec![1, 2, 3];
    let mut grown = ir.clone();
    grown.pinned_closures.clone_from(&extending.pinned_closures);
    grown.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(extending),
    });
    assert_eq!(grown.verify().is_ok(), verify_grown_tree(&grown, 1).is_ok());
}

#[test]
fn verify_grown_tree_agrees_with_verify_on_a_plain_tree() {
    let ir = tree(10.0);
    assert!(verify_grown_tree(&ir, 0).is_ok());
    assert!(verify_grown_tree(&ir, 1).is_err());
    let mut ir = tree(10.0);
    ir.sides[0].ir.root_slot_count = 3;
    assert!(ir.verify().is_err());
    assert!(verify_grown_tree(&ir, 0).is_err());
    // More root constants than slots is a root-level bank check.
    let mut ir = tree(10.0);
    ir.root_constants.push(std::sync::Arc::from(&b"x"[..]));
    assert!(ir.verify().is_err());
    assert!(verify_grown_tree(&ir, 0).is_err());
}

/// A call of `helper` that only the tree-wide predicates look at.
fn helper_call(helper: HelperId) -> TraceInstruction {
    TraceInstruction {
        site: site(3),
        result: None,
        op: TraceOp::CallHelper {
            helper,
            args: Vec::new(),
            effect: None,
            guard: GuardId(0),
        },
    }
}

/// `ir` with a second, plain side added at index 1. The new side is the one
/// `verify_grown_tree` is asked about; `edit` changes the earlier side, whose
/// own IR the check does not walk.
fn grown_by_a_plain_side(mut ir: TraceIr, edit: impl FnOnce(&mut TraceIr)) -> TraceIr {
    edit(&mut ir.sides[0].ir);
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(then_side(10.0)),
    });
    ir
}

#[test]
fn verify_grown_tree_runs_the_tree_wide_checks_that_read_the_sides() {
    // The root reads rows by lane and a side stores a string key, which can
    // rehash and move the rows.
    let mut ir = tree(10.0);
    ir.body.push(TraceInstruction {
        site: site(3),
        result: None,
        op: TraceOp::TableFieldLoad {
            field: 0,
            table: ValueId(0),
            guard: GuardId(0),
        },
    });
    let grown = grown_by_a_plain_side(ir, |side| {
        side.body.push(helper_call(HelperId::SET_STRING));
    });
    assert!(grown.storage_a_set_would_move().is_some());
    assert!(verify_grown_tree(&grown, 1).is_err());
    assert_eq!(
        verify_grown_tree(&grown, 1).is_err(),
        grown.verify().is_err()
    );

    // The root relies on a metatable fact and a side replaces a metatable.
    let mut ir = tree(10.0);
    ir.bindings.push(TableBinding {
        register: 0,
        kind: BindingKind::NoMetatable,
    });
    assert!(ir.stale_metatable_fact().is_none());
    let grown = grown_by_a_plain_side(ir, |side| {
        side.body.push(helper_call(HelperId::SETMETATABLE));
    });
    assert!(grown.stale_metatable_fact().is_some());
    assert!(verify_grown_tree(&grown, 1).is_err());
    assert_eq!(
        verify_grown_tree(&grown, 1).is_err(),
        grown.verify().is_err()
    );

    // A side resumes a coroutine next to a root with a hash binding.
    let mut ir = tree(10.0);
    ir.bindings.push(TableBinding {
        register: 0,
        kind: BindingKind::Hash,
    });
    assert!(ir.resume_conflict().is_none());
    let grown = grown_by_a_plain_side(ir, |side| {
        side.body
            .push(helper_call(HelperId::coroutine_resume(0, 0)));
    });
    assert!(grown.resume_conflict().is_some());
    assert!(verify_grown_tree(&grown, 1).is_err());
}

/// A root that reads `t[i]` from the table in r0 through `use_table`, and a
/// side from its loop test that writes r0 (with the same table) and links
/// back. Registers: r0 the table, r1 the sum, r2 the index.
fn table_tree(
    binding: BindingKind,
    use_table: impl Fn(&mut TraceBuilder, ValueId, [ValueId; 2]) -> ValueId,
) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), frame(3));
    for register in [1, 2] {
        b.mark_written(FrameId(0), register);
    }
    let table = b.input(0, TraceType::Table);
    let sum_in = b.input(1, TraceType::Float);
    let i_in = b.input(2, TraceType::Float);
    let bindings = vec![TableBinding {
        register: 0,
        kind: binding,
    }];
    b.set_bindings(bindings.clone());
    let sum = b.loop_param(TraceType::Float, LoopParamInit::Input(sum_in));
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let x = use_table(&mut b, table, [sum, i]);
    let sum2 = arithmetic(&mut b, 1, ArithmeticOp::Add, sum, x);
    let one = constant(&mut b, 1, 1.0);
    let i2 = arithmetic(&mut b, 1, ArithmeticOp::Add, i, one);
    let top = constant(&mut b, 1, 10.0);
    let more = compare(&mut b, 1, Comparison::LessEqual, i2, top);
    let after = vec![
        RegisterEntry::Unchanged,
        RegisterEntry::Value(sum2),
        RegisterEntry::Value(i2),
    ];
    guard(
        &mut b,
        1,
        more,
        true,
        SnapshotPhase::After {
            pc: 1,
            successor: 2,
        },
        after.clone(),
    );
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 1,
            successor: 0,
        },
        1,
        after,
    ));
    b.set_poll_snapshot(poll);
    let mut ir = b.finish(TraceTerminator::LoopBack {
        args: vec![sum2, i2],
    });

    // The side starts over: r0 gets the table again, r2 restarts at 1.
    let mut b = TraceBuilder::new(TraceId(1), frame(3));
    for register in [0, 2] {
        b.mark_written(FrameId(0), register);
    }
    b.set_bindings(bindings);
    let table = b.input(0, TraceType::Table);
    let sum = b.input(1, TraceType::Float);
    let start = constant(&mut b, 1, 1.0);
    let link = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 1,
            successor: 0,
        },
        1,
        vec![
            RegisterEntry::Value(table),
            RegisterEntry::Value(sum),
            RegisterEntry::Value(start),
        ],
    ));
    let side = b.finish(TraceTerminator::Link { snapshot: link });
    ir.header_live = vec![0, 1, 2];
    // The loop test is the root's last guard.
    let guard = ir.guards.last().unwrap().origin;
    ir.sides.push(TraceSide {
        parent: None,
        guard,
        ir: Box::new(side),
    });
    ir
}

fn array_read(b: &mut TraceBuilder, table: ValueId, [sum, i]: [ValueId; 2]) -> ValueId {
    let snapshot = b.snapshot(snap(
        SnapshotPhase::Before { pc: 0 },
        0,
        vec![
            RegisterEntry::Unchanged,
            RegisterEntry::Value(sum),
            RegisterEntry::Value(i),
        ],
    ));
    let id = b.guard(site(0), snapshot);
    b.emit(
        site(0),
        TraceOp::ArrayLoad {
            binding: 0,
            table,
            key: i,
            expected: TraceType::Float,
            past_end: false,
            guard: id,
        },
        Some(TraceType::Float),
    )
    .unwrap()
}

#[test]
fn a_side_that_writes_a_bound_array_register_demotes_the_binding_to_lane_reads() {
    let ir = table_tree(BindingKind::Array { first: None }, array_read);
    ir.verify().unwrap();
    let members = plan_tree(&ir).unwrap();
    assert_eq!(members[0].demoted, 1);
    for member in &members {
        member.ir.verify().unwrap();
        assert_eq!(member.ir.bindings, []);
        assert!(
            member
                .ir
                .fields
                .iter()
                .any(|field| field.key.as_ref() == TABLE_ADDRESS_KEY)
        );
    }
    let root = &members[0].ir;
    assert!(
        root.body
            .iter()
            .any(|instruction| matches!(instruction.op, TraceOp::LaneArrayLoad { .. }))
    );
    assert!(
        !root
            .body
            .iter()
            .any(|instruction| matches!(instruction.op, TraceOp::ArrayLoad { .. }))
    );
}

#[test]
fn a_side_that_writes_a_bound_field_register_guards_the_table_at_its_link() {
    let ir = table_tree(
        BindingKind::Field {
            key: b"x"[..].into(),
            ty: TraceType::Float,
            depth: 0,
        },
        |b, table, _| {
            b.emit(
                site(0),
                TraceOp::FieldLoad { binding: 0, table },
                Some(TraceType::Float),
            )
            .unwrap()
        },
    );
    ir.verify().unwrap();
    let members = plan_tree(&ir).unwrap();
    // Nothing is demoted, and the root does not carry r0: it goes on
    // reading its entry input.
    assert_eq!(members[0].demoted, 0);
    assert!(
        members[0]
            .ir
            .loop_params
            .iter()
            .all(|param| param.register != Some(0))
    );
    for member in &members {
        member.ir.verify().unwrap();
        assert_eq!(member.ir.bindings.len(), 1);
    }
    // The side leaves through its link snapshot unless r0 is the entry
    // table: a guard after its body, with an origin of its own.
    let side = &members[1].ir;
    let TraceTerminator::Link { snapshot } = side.terminator else {
        panic!("the side links");
    };
    let last = side.body.last().unwrap();
    let TraceOp::GuardBranch {
        expected: true,
        guard,
        ..
    } = last.op
    else {
        panic!("the side ends in a guard: {:?}", last.op);
    };
    assert_eq!(side.guards[guard.index()].snapshot, snapshot);
    let origins: Vec<u32> = side.guards.iter().map(|info| info.origin).collect();
    assert_eq!(
        origins
            .iter()
            .filter(|&&origin| origin == side.guards[guard.index()].origin)
            .count(),
        1
    );
}

#[test]
fn a_side_that_writes_a_callee_register_is_still_refused() {
    let mut ir = table_tree(BindingKind::Array { first: None }, array_read);
    ir.pinned_closures = vec![7];
    ir.bindings.push(TableBinding {
        register: 0,
        kind: BindingKind::Callee {
            key: b"f"[..].into(),
            lane: 0,
            depth: 0,
        },
    });
    assert_eq!(
        plan_tree(&ir).unwrap_err(),
        TreeError::TableWritten { register: 0 }
    );
}
