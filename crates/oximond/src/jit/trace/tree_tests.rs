//! Trace trees: a root loop and side traces compiled as one native function,
//! against the reference evaluator that runs them as the interpreter would.
//!
//! The root is the loop
//!
//! ```text
//! pc0  r3 = r2 % 7
//! pc1  r3 = r3 < 3
//! pc2  JumpIfFalse r3 -> pc5        (recorded: the jump is taken)
//! pc3  r0 = r0 + r2                 (the side trace's path)
//! pc4  Jump -> pc6
//! pc5  r1 = r1 + r2                 (the root's path)
//! pc6  r2 = r2 + 1; r2 <= 100 loops back to pc0
//! ```
//!
//! Registers: r0 `yes`, r1 `no`, r2 the counter, r3 a temporary. The root
//! never touches `yes`, so the tree has to widen it (see
//! `oximond::jit_ir::trace::plan_tree`).

use super::*;
use crate::jit::OptimizationTier;
use crate::jit_ir::Comparison;
use crate::jit_ir::trace::reference::{Outcome, evaluate_tree};
use crate::jit_ir::trace::{
    ArithmeticOp, ExternalSlot, FrameId, LoopParamInit, RegisterEntry, Snapshot, SnapshotFrame,
    SnapshotId, SnapshotPhase, TraceBuilder, TraceFrameInfo, TraceId, TraceIr, TraceOp,
    TraceScalar, TraceSide, TraceSite, TraceTerminator, TraceType, ValueId,
};

const TIERS: [OptimizationTier; 2] = [OptimizationTier::Baseline, OptimizationTier::Optimized];

fn site(pc: usize) -> TraceSite {
    TraceSite {
        frame: FrameId(0),
        pc,
    }
}

fn frame() -> TraceFrameInfo {
    TraceFrameInfo {
        register_count: 4,
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

fn f(value: f64) -> TraceScalar {
    TraceScalar::Float(value)
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

fn constant(b: &mut TraceBuilder, pc: usize, value: f64) -> ValueId {
    b.emit(
        site(pc),
        TraceOp::Constant(f(value)),
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
    after: SnapshotPhase,
    registers: Vec<RegisterEntry>,
) {
    let snap = b.snapshot(snapshot(after, pc, registers));
    let id = b.guard(site(pc), snap);
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

/// The root: the `else` path, `no = no + i`.
fn root_loop(limit: f64) -> TraceIr {
    root_loop_with(limit, false)
}

/// [`root_loop`], and with `string` the guard that enters the side holds a
/// root string in r3.
fn root_loop_with(limit: f64, string: bool) -> TraceIr {
    root_loop_full(limit, string, false)
}

/// [`root_loop_with`], and with `reads_yes` the root takes `yes` (r0) as an
/// entry input it never uses, as a root does whose iteration reads a register
/// a side writes.
fn root_loop_full(limit: f64, string: bool, reads_yes: bool) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), frame());
    for register in [1, 2, 3] {
        b.mark_written(FrameId(0), register);
    }
    if reads_yes {
        b.input(0, TraceType::Float);
    }
    let no_in = b.input(1, TraceType::Float);
    let i_in = b.input(2, TraceType::Float);
    let no = b.loop_param(TraceType::Float, LoopParamInit::Input(no_in));
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let seven = constant(&mut b, 0, 7.0);
    let rem = arithmetic(&mut b, 0, ArithmeticOp::Modulo, i, seven);
    let three = constant(&mut b, 1, 3.0);
    let c = compare(&mut b, 1, Comparison::LessThan, rem, three);
    // The branch at pc2 jumps over the `then` part. The failure is the other
    // direction: it falls into pc3.
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
            if string {
                RegisterEntry::Root(0)
            } else {
                RegisterEntry::Value(c)
            },
        ],
    );
    let no2 = arithmetic(&mut b, 5, ArithmeticOp::Add, no, i);
    let one = constant(&mut b, 6, 1.0);
    let i2 = arithmetic(&mut b, 6, ArithmeticOp::Add, i, one);
    let top = constant(&mut b, 6, limit);
    let more = compare(&mut b, 6, Comparison::LessEqual, i2, top);
    guard(
        &mut b,
        6,
        more,
        true,
        SnapshotPhase::After {
            pc: 6,
            successor: 7,
        },
        vec![
            RegisterEntry::Unchanged,
            RegisterEntry::Value(no2),
            RegisterEntry::Value(i2),
            RegisterEntry::Value(c),
        ],
    );
    let poll = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        6,
        vec![
            RegisterEntry::Unchanged,
            RegisterEntry::Value(no2),
            RegisterEntry::Value(i2),
            RegisterEntry::Value(c),
        ],
    ));
    b.set_poll_snapshot(poll);
    let mut ir = b.finish(TraceTerminator::LoopBack {
        args: vec![no2, i2],
    });
    if string {
        ir.root_constants = vec![std::sync::Arc::from(&b"word"[..])];
        ir.root_slot_count = 1;
    }
    ir
}

/// The side: the `then` path, `yes = yes + i`, rejoining the loop.
fn then_side(limit: f64) -> TraceIr {
    then_side_with(limit, false)
}

/// [`then_side`], and with `string` it takes the string r3 holds as an
/// input and keeps it in the registers it leaves.
fn then_side_with(limit: f64, string: bool) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(1), frame());
    for register in [0, 2] {
        b.mark_written(FrameId(0), register);
    }
    let yes = b.input(0, TraceType::Float);
    let no = b.input(1, TraceType::Float);
    let i = b.input(2, TraceType::Float);
    let word = string.then(|| b.input(3, TraceType::String));
    let last = word.map_or(RegisterEntry::Const(TraceScalar::Nil), RegisterEntry::Value);
    let yes2 = arithmetic(&mut b, 3, ArithmeticOp::Add, yes, i);
    let one = constant(&mut b, 6, 1.0);
    let i2 = arithmetic(&mut b, 6, ArithmeticOp::Add, i, one);
    let top = constant(&mut b, 6, limit);
    let more = compare(&mut b, 6, Comparison::LessEqual, i2, top);
    guard(
        &mut b,
        6,
        more,
        true,
        SnapshotPhase::After {
            pc: 6,
            successor: 7,
        },
        vec![
            RegisterEntry::Value(yes2),
            RegisterEntry::Value(no),
            RegisterEntry::Value(i2),
            last,
        ],
    );
    let link = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        6,
        vec![
            RegisterEntry::Value(yes2),
            RegisterEntry::Value(no),
            RegisterEntry::Value(i2),
            last,
        ],
    ));
    let mut ir = b.finish(TraceTerminator::Link { snapshot: link });
    if string {
        ir.root_constants = vec![std::sync::Arc::from(&b"word"[..])];
        ir.root_slot_count = 1;
    }
    ir
}

fn tree(limit: f64) -> TraceIr {
    tree_with(limit, false)
}

fn tree_with(limit: f64, string: bool) -> TraceIr {
    let mut ir = root_loop_with(limit, string);
    ir.header_live = vec![0, 1, 2];
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(then_side_with(limit, string)),
    });
    ir
}

/// The state after native code left through exit `index`.
fn restored(
    trace: &CompiledTrace,
    index: usize,
    area: &[i64],
    registers: &[TraceScalar],
) -> (usize, Vec<TraceScalar>) {
    let descriptor = &trace.exits()[index];
    let ir = trace.lowered_member(descriptor.member);
    let snapshot = &ir.snapshots[descriptor.snapshot.index()];
    let state = snapshot.frames[0]
        .registers
        .iter()
        .enumerate()
        .map(|(register, entry)| match entry {
            RegisterEntry::Value(value) => descriptor.scalar(area, *value).unwrap_or_else(|| {
                // An exit leaves out a register's own entry input when the
                // frame keeps it: the restore leaves the register alone.
                assert!(
                    ir.entry.inputs.iter().any(|input| {
                        usize::from(input.register) == register && input.value == *value
                    }),
                    "the exit spilled every snapshot value"
                );
                registers[register]
            }),
            RegisterEntry::Const(scalar) => *scalar,
            RegisterEntry::Unchanged => registers[register],
            RegisterEntry::Root(_) => unreachable!("no roots in these traces"),
            RegisterEntry::Virtual(_) | RegisterEntry::Slice(_) => {
                unreachable!("no virtual tables here")
            }
        })
        .collect();
    (snapshot.resume_point().pc, state)
}

/// Runs the compiled tree from `registers` with `fuel`, returning the exit
/// index, the state it restores, and the fuel left.
fn run(
    trace: &CompiledTrace,
    registers: &[TraceScalar],
    fuel: i64,
) -> (TraceOutcome, usize, Vec<TraceScalar>, i64) {
    // The entry inputs, then the registers sides compiled apart read.
    let extras = trace.extras().iter().map(|(source, _)| match source {
        crate::jit::ExtraSource::Register(register) => *register,
        crate::jit::ExtraSource::Caller(_) | crate::jit::ExtraSource::External(_) => {
            unreachable!("these trees have no caller")
        }
    });
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| *register)
        .chain(extras)
        .map(|register| TraceInputSlot::from_scalar(registers[usize::from(register)]))
        .collect();
    let mut area = vec![0_i64; trace.exit_area_len()];
    let mut control = TraceControl::new(fuel);
    let outcome = trace.run(&inputs, &mut control, &mut area);
    match outcome {
        TraceOutcome::Exit { index } => {
            let (pc, state) = restored(trace, index, &area, registers);
            (outcome, pc, state, control.fuel)
        }
        TraceOutcome::EntryMismatch { .. }
        | TraceOutcome::StoreEntryMissing
        | TraceOutcome::Returned => (outcome, 0, Vec::new(), control.fuel),
    }
}

fn reference(ir: &TraceIr, registers: &[TraceScalar]) -> (usize, Vec<TraceScalar>, u32) {
    let evaluation = evaluate_tree(ir, registers, &[], 100_000);
    let restored = match evaluation.outcome {
        Outcome::GuardFailed { restored, .. } | Outcome::Exited(restored) => restored,
        other => panic!("the reference leaves the loop, not {other:?}"),
    };
    (restored.pc, restored.registers, evaluation.side_entries)
}

#[test]
fn the_widened_root_carries_a_register_only_a_side_writes() {
    let members = crate::jit_ir::trace::plan_tree(&tree(100.0)).unwrap();
    assert_eq!(members.len(), 2);
    let root = &members[0].ir;
    // `yes` (r0) became an input and a loop parameter, and no snapshot of
    // the root calls it unchanged any more.
    assert!(root.entry.inputs.iter().any(|input| input.register == 0));
    assert_eq!(
        root.loop_params
            .iter()
            .map(|param| param.register)
            .collect::<Vec<_>>(),
        vec![Some(1), Some(2), Some(0)]
    );
    for snapshot in &root.snapshots {
        assert_ne!(
            snapshot.frames[0].registers[0],
            RegisterEntry::Unchanged,
            "snapshot {} still trusts the interpreter register of r0",
            snapshot.id.0
        );
    }
    root.verify().unwrap();
    assert_eq!(members[1].parent, Some((0, 0)));
}

#[test]
fn a_tree_with_a_side_matches_the_interpreter_like_reference() {
    let ir = tree(100.0);
    for tier in TIERS {
        let trace = compile_trace(&ir, tier).unwrap();
        assert_eq!(trace.member_count(), 2, "{tier:?}");
        assert_eq!(trace.side_members(), &[Some(1)]);
        // The root asks for yes (r0) as well now.
        assert!(trace.entry_contract().iter().any(|(r, _)| *r == 0));
        for start in [1.0, 2.0, 3.0, 4.0, 7.0, 50.5, 99.0] {
            let registers = [f(10.0), f(20.0), f(start), TraceScalar::Nil];
            let (outcome, pc, state, _) = run(&trace, &registers, 100_000);
            assert!(matches!(outcome, TraceOutcome::Exit { .. }));
            let (expected_pc, expected, _) = reference(&ir, &registers);
            assert_eq!(pc, expected_pc, "start {start} at {tier:?}");
            assert_eq!(state, expected, "start {start} at {tier:?}");
        }
    }
}

#[test]
fn guards_with_a_side_run_it_natively_and_leave_only_through_the_loop_end() {
    let ir = tree(1_000.0);
    let trace = compile_trace(&ir, OptimizationTier::Optimized).unwrap();
    let registers = [f(0.0), f(0.0), f(1.0), TraceScalar::Nil];
    let (outcome, pc, state, left) = run(&trace, &registers, 1_000_000);
    let TraceOutcome::Exit { index } = outcome else {
        panic!("the loop exits");
    };
    // A thousand iterations and exactly one exit: the loop's own end. The
    // hundreds of alternations in between never left native code.
    assert_eq!(trace.exits()[index].class, TraceExitClass::LoopComplete);
    assert_eq!(pc, 7);
    assert_eq!(
        1_000_000 - left,
        999,
        "one loop-back per finished iteration"
    );
    let (_, expected, side_entries) = reference(&ir, &registers);
    assert_eq!(state, expected);
    assert!(side_entries > 100, "the side ran {side_entries} times");
}

#[test]
fn a_root_without_its_side_exits_at_the_branch() {
    // The same loop without the side leaves natively at every `then` iteration.
    let mut ir = tree(100.0);
    ir.sides.clear();
    let trace = compile_trace(&ir, OptimizationTier::Optimized).unwrap();
    assert_eq!(trace.member_count(), 1);
    let registers = [f(0.0), f(0.0), f(1.0), TraceScalar::Nil];
    let (outcome, pc, _, _) = run(&trace, &registers, 100_000);
    let TraceOutcome::Exit { index } = outcome else {
        panic!("it exits");
    };
    assert_eq!(trace.exits()[index].class, TraceExitClass::GuardFail);
    // i = 1: 1 % 7 = 1 < 3, so the branch is not taken.
    assert_eq!(pc, 3);
}

#[test]
fn fuel_counts_links_and_a_poll_exit_resumes_to_the_same_result() {
    let ir = tree(500.0);
    for tier in TIERS {
        let trace = compile_trace(&ir, tier).unwrap();
        let registers = [f(1.0), f(2.0), f(1.0), TraceScalar::Nil];
        let (_, expected_pc, expected, _) = {
            let (pc, state, entries) = reference(&ir, &registers);
            (0, pc, state, entries)
        };
        // Poll every 7 loop-backs and links, and re-enter with the state.
        let mut state = registers.to_vec();
        let mut polls = 0;
        let mut total = 0;
        let final_pc = loop {
            let (outcome, pc, restored, left) = run(&trace, &state, 7);
            let TraceOutcome::Exit { index } = outcome else {
                panic!("entry types hold");
            };
            total += 7 - left;
            state = restored;
            if trace.exits()[index].class == TraceExitClass::Poll {
                assert_eq!(left, 0, "a poll exit has spent all of its fuel");
                polls += 1;
                continue;
            }
            break pc;
        };
        assert_eq!(final_pc, expected_pc, "{tier:?}");
        assert_eq!(state, expected, "{tier:?}");
        // One unit per iteration, whichever trace ran it.
        assert_eq!(total, 499, "{tier:?}");
        assert!(polls >= 70, "{tier:?}: {polls} polls");
    }
}

#[test]
fn a_poll_inside_a_side_publishes_the_state_the_loop_header_sees() {
    // A fuel of 1 spends itself at the first loop-back, wherever it is. With
    // i = 1 the iteration runs the side (1 % 7 < 3), so the poll is the
    // side's.
    let ir = tree(100.0);
    let trace = compile_trace(&ir, OptimizationTier::Optimized).unwrap();
    let registers = [f(5.0), f(6.0), f(1.0), TraceScalar::Nil];
    let (outcome, pc, state, left) = run(&trace, &registers, 1);
    let TraceOutcome::Exit { index } = outcome else {
        panic!("it exits");
    };
    let descriptor = &trace.exits()[index];
    assert_eq!(descriptor.class, TraceExitClass::Poll);
    assert_eq!(descriptor.member, 1, "the side's own poll");
    assert_eq!(left, 0);
    assert_eq!(pc, 0);
    // yes = 5 + 1, no unchanged, i = 2.
    assert_eq!(state[0], f(6.0));
    assert_eq!(state[1], f(6.0));
    assert_eq!(state[2], f(2.0));
}

#[test]
fn a_guard_of_a_side_exits_with_the_sides_snapshot() {
    let ir = tree(3.5);
    for tier in TIERS {
        let trace = compile_trace(&ir, tier).unwrap();
        // i = 3: 3 % 7 = 3, not < 3, so the root path runs: no += 3, i = 4,
        // and the latch ends the loop in the root.
        let registers = [f(0.0), f(0.0), f(3.0), TraceScalar::Nil];
        let (outcome, pc, state, _) = run(&trace, &registers, 1_000);
        let TraceOutcome::Exit { index } = outcome else {
            panic!("it exits");
        };
        assert_eq!(trace.exits()[index].member, 0, "{tier:?}");
        assert_eq!(pc, 7);
        assert_eq!(state[1], f(3.0));
        // i = 2: the side runs, yes += 2, i = 3, and 3 <= 3.5 loops back into
        // the root, which ends the loop at i = 3 -> 4.
        let registers = [f(0.0), f(0.0), f(2.0), TraceScalar::Nil];
        let (_, pc, state, _) = run(&trace, &registers, 1_000);
        assert_eq!(pc, 7);
        assert_eq!(state[0], f(2.0));
        assert_eq!(state[1], f(3.0));
        assert_eq!(state[2], f(4.0));
        // i = 2.9: the side runs and its own latch ends the loop, since
        // 3.9 > 3.5.
        let registers = [f(0.0), f(0.0), f(2.9), TraceScalar::Nil];
        let (outcome, pc, state, _) = run(&trace, &registers, 1_000);
        let TraceOutcome::Exit { index } = outcome else {
            panic!("it exits");
        };
        assert_eq!(trace.exits()[index].member, 1, "{tier:?}");
        assert_eq!(pc, 7);
        assert_eq!(state[0], f(2.9));
        assert_eq!(state[2], f(2.9 + 1.0));
    }
}

#[test]
fn forcing_a_guard_of_the_side_takes_the_native_exit_stub() {
    let ir = tree(100.0);
    let trace = compile_trace(&ir, OptimizationTier::Baseline).unwrap();
    let side = trace.lowered_member(1);
    assert_ne!(side.guards, [] as [crate::jit_ir::trace::GuardInfo; 0]);
    let forced = compile_trace_forcing_member_guard(&ir, OptimizationTier::Baseline, 1, 0).unwrap();
    // i = 1 takes the side; its latch guard is forced to fail at once.
    let registers = [f(5.0), f(6.0), f(1.0), TraceScalar::Nil];
    let (outcome, pc, state, _) = run(&forced, &registers, 100);
    let TraceOutcome::Exit { index } = outcome else {
        panic!("it exits");
    };
    let descriptor = &forced.exits()[index];
    assert_eq!(descriptor.member, 1);
    assert_eq!(descriptor.origin, Some(0));
    assert_eq!(pc, 7);
    assert_eq!(state[0], f(6.0));
    assert_eq!(state[2], f(2.0));
}

#[test]
fn entry_types_cover_what_the_side_needs() {
    // A tree whose side reads a register the root never reads: the root
    // takes it as an entry input, so a non-number there does not enter.
    let ir = tree(100.0);
    let trace = compile_trace(&ir, OptimizationTier::Optimized).unwrap();
    let mut area = vec![0_i64; trace.exit_area_len()];
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| {
            if *register == 0 {
                TraceInputSlot::nil()
            } else {
                TraceInputSlot::float(1.0)
            }
        })
        .collect();
    let mut control = TraceControl::new(10);
    assert!(matches!(
        trace.run(&inputs, &mut control, &mut area),
        TraceOutcome::EntryMismatch { .. }
    ));
}

#[test]
fn a_side_takes_a_root_string_its_parent_holds() {
    // The guard that enters the side holds the string as a root constant,
    // which the side reads as an owned string input: the lane id of the slot.
    let ir = tree_with(100.0, true);
    for tier in TIERS {
        compile_trace(&ir, tier).unwrap();
        let registers = [f(5.0), f(6.0), f(1.0), TraceScalar::Nil];
        // i = 1 takes the side; its first latch guard is forced to fail.
        let forced = compile_trace_forcing_member_guard(&ir, tier, 1, 0).unwrap();
        let (outcome, _, state, _) = run(&forced, &registers, 100);
        let TraceOutcome::Exit { index } = outcome else {
            panic!("it exits");
        };
        assert_eq!(forced.exits()[index].member, 1, "{tier:?}");
        assert_eq!(
            state[3],
            TraceScalar::String(crate::jit_ir::trace::ROOT_STRING_LANE_BASE),
            "{tier:?}"
        );
    }
}

#[test]
fn a_side_whose_final_type_disagrees_with_the_loop_is_refused() {
    // The side leaves r1 (a loop parameter of the root) as nil.
    let mut ir = tree(100.0);
    let side = &mut ir.sides[0].ir;
    let TraceTerminator::Link { snapshot: link } = side.terminator else {
        unreachable!()
    };
    side.snapshots[link.index()].frames[0]
        .registers
        .set(1, RegisterEntry::Const(TraceScalar::Nil));
    assert!(matches!(
        crate::jit_ir::trace::plan_tree(&ir),
        Err(crate::jit_ir::trace::TreeError::LinkValue {
            side: 0,
            register: 1
        })
    ));
    assert!(matches!(
        compile_trace(&ir, OptimizationTier::Baseline),
        Err(TraceCompileError::Tree(_))
    ));
}

#[test]
fn a_side_for_a_guard_the_optimizer_removed_is_left_out() {
    // Make the root's branch guard pass for certain: its condition is the
    // constant `true`. Folding removes the guard and with it the side's
    // entrance, which the compiled code reports.
    let mut b = TraceBuilder::new(TraceId(0), frame());
    b.mark_written(FrameId(0), 2);
    let i_in = b.input(2, TraceType::Float);
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let one = constant(&mut b, 0, 1.0);
    let two = constant(&mut b, 0, 2.0);
    let c = compare(&mut b, 1, Comparison::LessThan, one, two);
    guard(
        &mut b,
        2,
        c,
        true,
        SnapshotPhase::After {
            pc: 2,
            successor: 3,
        },
        vec![
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
            RegisterEntry::Value(i),
            RegisterEntry::Value(c),
        ],
    );
    let i2 = arithmetic(&mut b, 6, ArithmeticOp::Add, i, one);
    let top = constant(&mut b, 6, 10.0);
    let more = compare(&mut b, 6, Comparison::LessEqual, i2, top);
    guard(
        &mut b,
        6,
        more,
        true,
        SnapshotPhase::After {
            pc: 6,
            successor: 7,
        },
        vec![
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
            RegisterEntry::Value(i2),
            RegisterEntry::Value(c),
        ],
    );
    let poll = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        6,
        vec![
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
            RegisterEntry::Value(i2),
            RegisterEntry::Value(c),
        ],
    ));
    b.set_poll_snapshot(poll);
    let mut ir = b.finish(TraceTerminator::LoopBack { args: vec![i2] });
    ir.header_live = vec![2];
    // A side that links back from the guard at origin 0.
    let mut s = TraceBuilder::new(TraceId(1), frame());
    s.mark_written(FrameId(0), 2);
    let side_i = s.input(2, TraceType::Float);
    let side_one = constant(&mut s, 3, 1.0);
    let side_i2 = arithmetic(&mut s, 3, ArithmeticOp::Add, side_i, side_one);
    let link = s.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        6,
        vec![
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
            RegisterEntry::Value(side_i2),
            RegisterEntry::Const(TraceScalar::Nil),
        ],
    ));
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(s.finish(TraceTerminator::Link { snapshot: link })),
    });
    ir.verify().unwrap();
    let baseline = compile_trace(&ir, OptimizationTier::Baseline).unwrap();
    assert_eq!(
        baseline.side_members(),
        &[Some(1)],
        "baseline keeps the guard"
    );
    let optimized = compile_trace(&ir, OptimizationTier::Optimized).unwrap();
    assert_eq!(optimized.side_members(), &[None]);
    assert_eq!(optimized.member_count(), 1);
}

/// A tiny deterministic generator for the property test below.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
}

#[test]
fn trees_with_random_start_states_match_the_reference_at_both_tiers() {
    let mut rng = Lcg(0x5eed_1234);
    for limit in [10.0, 64.0, 200.0] {
        let ir = tree(limit);
        for tier in TIERS {
            let trace = compile_trace(&ir, tier).unwrap();
            for _ in 0..60 {
                let pick = |rng: &mut Lcg| match rng.next() % 6 {
                    0 => 0.0,
                    1 => -3.5,
                    2 => 1e15,
                    _ => (rng.next() % 1_000) as f64 / 8.0,
                };
                let registers = [
                    f(pick(&mut rng)),
                    f(pick(&mut rng)),
                    f((rng.next() % 300) as f64 / 2.0 - 5.0),
                    TraceScalar::Nil,
                ];
                let (outcome, pc, state, _) = run(&trace, &registers, 1_000_000);
                assert!(matches!(outcome, TraceOutcome::Exit { .. }));
                let (expected_pc, expected, _) = reference(&ir, &registers);
                assert_eq!(pc, expected_pc, "{registers:?} limit {limit} {tier:?}");
                assert_eq!(state, expected, "{registers:?} limit {limit} {tier:?}");
            }
        }
    }
}

// ------------------------------------------------------------- three arms

/// A three-way branch, `r = i % 3; if r < 1 then a += i elseif r < 2 then
/// b += i else c += i end`, over nine registers:
///
/// ```text
/// r0 a  r1 b  r2 i  r3 compare  r4 r  r5 c  r6 1.0  r7 2.0  r8 3.0
/// ```
///
/// The root records arm A. The first side (from the root's guard) records
/// arm B, and a side of that side (from its guard) records arm C. The root
/// never reads r7, which the first side does.
mod three_way {
    use super::*;

    const REGISTERS: u16 = 9;
    const LATCH: usize = 11;

    fn window() -> TraceFrameInfo {
        TraceFrameInfo {
            register_count: REGISTERS,
            code_len: 16,
            inlined: None,
            activation: None,
        }
    }

    fn nil() -> RegisterEntry {
        RegisterEntry::Const(TraceScalar::Nil)
    }

    /// The loop's latch, `i += 1; if i <= limit`.
    fn latch_ops(b: &mut TraceBuilder, i: ValueId, limit: f64) -> (ValueId, ValueId) {
        let one = constant(b, LATCH, 1.0);
        let next = arithmetic(b, LATCH, ArithmeticOp::Add, i, one);
        let top = constant(b, LATCH, limit);
        let more = compare(b, LATCH, Comparison::LessEqual, next, top);
        (next, more)
    }

    fn latch_guard(b: &mut TraceBuilder, more: ValueId, state: Vec<RegisterEntry>) {
        guard(
            b,
            LATCH,
            more,
            true,
            SnapshotPhase::After {
                pc: LATCH,
                successor: 12,
            },
            state,
        );
    }

    fn close(b: &mut TraceBuilder, state: Vec<RegisterEntry>) -> SnapshotId {
        b.snapshot(snapshot(
            SnapshotPhase::After {
                pc: LATCH,
                successor: 0,
            },
            LATCH,
            state,
        ))
    }

    pub(super) fn root(limit: f64) -> TraceIr {
        root_with(limit, false)
    }

    /// The root. With `invariant_branch` its first branch tests two
    /// invariant registers, which the optimizer hoists out of the loop.
    pub(super) fn root_with(limit: f64, invariant_branch: bool) -> TraceIr {
        let mut b = TraceBuilder::new(TraceId(0), window());
        for register in [0, 2, 3, 4] {
            b.mark_written(FrameId(0), register);
        }
        let a_in = b.input(0, TraceType::Float);
        let i_in = b.input(2, TraceType::Float);
        let one_in = b.input(6, TraceType::Float);
        let three_in = b.input(8, TraceType::Float);
        let a = b.loop_param(TraceType::Float, LoopParamInit::Input(a_in));
        let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
        let rem = arithmetic(&mut b, 0, ArithmeticOp::Modulo, i, three_in);
        let c0 = if invariant_branch {
            compare(&mut b, 1, Comparison::LessThan, one_in, three_in)
        } else {
            compare(&mut b, 1, Comparison::LessThan, rem, one_in)
        };
        // The jump at pc2 is not taken: the guard holds while c0 is true.
        guard(
            &mut b,
            2,
            c0,
            true,
            SnapshotPhase::After {
                pc: 2,
                successor: 5,
            },
            vec![
                RegisterEntry::Value(a),
                RegisterEntry::Unchanged,
                RegisterEntry::Value(i),
                RegisterEntry::Value(c0),
                RegisterEntry::Value(rem),
                RegisterEntry::Unchanged,
                RegisterEntry::Unchanged,
                RegisterEntry::Unchanged,
                RegisterEntry::Unchanged,
            ],
        );
        let a2 = arithmetic(&mut b, 3, ArithmeticOp::Add, a, i);
        let (i2, more) = latch_ops(&mut b, i, limit);
        let state = vec![
            RegisterEntry::Value(a2),
            RegisterEntry::Unchanged,
            RegisterEntry::Value(i2),
            nil(),
            nil(),
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
        ];
        latch_guard(&mut b, more, state.clone());
        let poll = close(&mut b, state);
        b.set_poll_snapshot(poll);
        b.finish(TraceTerminator::LoopBack { args: vec![a2, i2] })
    }

    /// Every live register the root leaves alone is an input of the side,
    /// like the recorder makes them.
    struct Inputs {
        a: ValueId,
        b: ValueId,
        i: ValueId,
        r: ValueId,
        c: ValueId,
        one: ValueId,
        two: ValueId,
        three: ValueId,
    }

    fn side_inputs(b: &mut TraceBuilder) -> Inputs {
        Inputs {
            a: b.input(0, TraceType::Float),
            b: b.input(1, TraceType::Float),
            i: b.input(2, TraceType::Float),
            r: b.input(4, TraceType::Float),
            c: b.input(5, TraceType::Float),
            one: b.input(6, TraceType::Float),
            two: b.input(7, TraceType::Float),
            three: b.input(8, TraceType::Float),
        }
    }

    // The single letters name the trace's registers, as the IR text dumps do.
    #[allow(clippy::many_single_char_names)]
    fn state(
        a: ValueId,
        b: ValueId,
        i: ValueId,
        compare: Option<ValueId>,
        r: Option<ValueId>,
        c: ValueId,
        invariants: (ValueId, ValueId, ValueId),
    ) -> Vec<RegisterEntry> {
        vec![
            RegisterEntry::Value(a),
            RegisterEntry::Value(b),
            RegisterEntry::Value(i),
            compare.map_or_else(nil, RegisterEntry::Value),
            r.map_or_else(nil, RegisterEntry::Value),
            RegisterEntry::Value(c),
            RegisterEntry::Value(invariants.0),
            RegisterEntry::Value(invariants.1),
            RegisterEntry::Value(invariants.2),
        ]
    }

    /// Arm B, from the root's guard at pc2: `b += i`.
    pub(super) fn arm_b(limit: f64) -> TraceIr {
        let mut t = TraceBuilder::new(TraceId(1), window());
        for register in [1, 2, 3] {
            t.mark_written(FrameId(0), register);
        }
        let v = side_inputs(&mut t);
        let invariants = (v.one, v.two, v.three);
        let c1 = compare(&mut t, 5, Comparison::LessThan, v.r, v.two);
        // Falls through into arm B while c1 holds. Failing goes to arm C.
        guard(
            &mut t,
            6,
            c1,
            true,
            SnapshotPhase::After {
                pc: 6,
                successor: 9,
            },
            state(v.a, v.b, v.i, Some(c1), Some(v.r), v.c, invariants),
        );
        let b2 = arithmetic(&mut t, 7, ArithmeticOp::Add, v.b, v.i);
        let (i2, more) = latch_ops(&mut t, v.i, limit);
        let state = state(v.a, b2, i2, None, None, v.c, invariants);
        latch_guard(&mut t, more, state.clone());
        let link = close(&mut t, state);
        t.finish(TraceTerminator::Link { snapshot: link })
    }

    /// Arm C, from arm B's guard at pc6: `c += i`.
    pub(super) fn arm_c(limit: f64) -> TraceIr {
        let mut t = TraceBuilder::new(TraceId(2), window());
        for register in [2, 5] {
            t.mark_written(FrameId(0), register);
        }
        let v = side_inputs(&mut t);
        let invariants = (v.one, v.two, v.three);
        let c2 = arithmetic(&mut t, 9, ArithmeticOp::Add, v.c, v.i);
        let (i2, more) = latch_ops(&mut t, v.i, limit);
        let state = state(v.a, v.b, i2, None, None, c2, invariants);
        latch_guard(&mut t, more, state.clone());
        let link = close(&mut t, state);
        t.finish(TraceTerminator::Link { snapshot: link })
    }

    pub(super) fn tree(limit: f64) -> TraceIr {
        tree_of(root(limit), limit)
    }

    pub(super) fn tree_of(mut ir: TraceIr, limit: f64) -> TraceIr {
        ir.header_live = vec![0, 1, 2, 5, 6, 7, 8];
        ir.sides.push(TraceSide {
            parent: None,
            guard: 0,
            ir: Box::new(arm_b(limit)),
        });
        ir.sides.push(TraceSide {
            parent: Some(0),
            guard: 0,
            ir: Box::new(arm_c(limit)),
        });
        ir
    }

    pub(super) fn registers(a: f64, b: f64, i: f64, c: f64) -> Vec<TraceScalar> {
        vec![
            f(a),
            f(b),
            f(i),
            TraceScalar::Nil,
            TraceScalar::Nil,
            f(c),
            f(1.0),
            f(2.0),
            f(3.0),
        ]
    }
}

#[test]
fn a_side_of_a_side_carries_three_arms_natively() {
    let ir = three_way::tree(60.0);
    ir.verify().unwrap();
    for tier in TIERS {
        let trace = compile_trace(&ir, tier).unwrap();
        assert_eq!(trace.member_count(), 3, "{tier:?}");
        assert_eq!(trace.side_members(), &[Some(1), Some(2)]);
        // The root widened: it reads the registers both sides need.
        let entry: Vec<u16> = trace.entry_contract().iter().map(|(r, _)| *r).collect();
        for register in [0, 1, 2, 5, 6, 7, 8] {
            assert!(entry.contains(&register), "r{register} in {entry:?}");
        }
        let registers = three_way::registers(0.0, 0.0, 1.0, 0.0);
        let (outcome, pc, state, left) = run(&trace, &registers, 1_000_000);
        let TraceOutcome::Exit { index } = outcome else {
            panic!("it exits");
        };
        assert_eq!(trace.exits()[index].class, TraceExitClass::LoopComplete);
        assert_eq!(pc, 12);
        assert_eq!(1_000_000 - left, 59, "{tier:?}");
        let (expected_pc, expected, entries) = reference(&ir, &registers);
        assert_eq!(pc, expected_pc);
        assert_eq!(state, expected, "{tier:?}");
        // 60 iterations in three arms: all but the first (the root's) came
        // through a side, and a third of them through a side of a side.
        assert!(entries >= 38, "{entries}");
    }
}

#[test]
fn a_side_of_a_side_matches_the_reference_from_many_states() {
    let ir = three_way::tree(30.0);
    let mut rng = Lcg(0xabcd_ef01);
    for tier in TIERS {
        let trace = compile_trace(&ir, tier).unwrap();
        for _ in 0..80 {
            let registers = three_way::registers(
                (rng.next() % 50) as f64 / 4.0,
                (rng.next() % 50) as f64 / 4.0,
                (rng.next() % 60) as f64 - 5.0,
                (rng.next() % 50) as f64 / 4.0,
            );
            let (outcome, pc, state, _) = run(&trace, &registers, 1_000_000);
            assert!(matches!(outcome, TraceOutcome::Exit { .. }));
            let (expected_pc, expected, _) = reference(&ir, &registers);
            assert_eq!(pc, expected_pc, "{registers:?} {tier:?}");
            assert_eq!(state, expected, "{registers:?} {tier:?}");
        }
    }
}

#[test]
fn a_guard_a_side_continues_is_not_hoisted_out_of_the_loop() {
    // The root's first branch tests invariants only, so it would fail once, at
    // entry, before the loop and never inside it. A side continues it, and a
    // link brings the loop a new value for what the test reads, so the guard
    // stays in the body at both tiers: the side is attached and the loop
    // takes it when the guard fails.
    let ir = three_way::tree_of(three_way::root_with(30.0, true), 30.0);
    ir.verify().unwrap();
    for tier in [OptimizationTier::Baseline, OptimizationTier::Optimized] {
        let compiled = compile_trace(&ir, tier).unwrap();
        assert_eq!(compiled.side_members(), &[Some(1), Some(2)], "{tier:?}");
        assert_eq!(compiled.member_count(), 3, "{tier:?}");
        for start in [1.0, 5.0, 2.0] {
            let registers = three_way::registers(0.0, 0.0, start, 0.0);
            let (outcome, pc, state, _) = run(&compiled, &registers, 100_000);
            assert!(matches!(outcome, TraceOutcome::Exit { .. }));
            let (expected_pc, expected, _) = reference(&ir, &registers);
            assert_eq!(pc, expected_pc, "{start} {tier:?}");
            assert_eq!(state, expected, "{start} {tier:?}");
        }
    }
}

#[test]
fn links_are_counted_only_when_asked() {
    let ir = tree(1_000.0);
    let registers = [f(0.0), f(0.0), f(1.0), TraceScalar::Nil];
    let inputs = |trace: &CompiledTrace| -> Vec<TraceInputSlot> {
        trace
            .entry_contract()
            .iter()
            .map(|(register, _)| TraceInputSlot::from_scalar(registers[usize::from(*register)]))
            .collect()
    };
    let plain = compile_trace(&ir, OptimizationTier::Baseline).unwrap();
    let counting = compile_trace_with(
        &ir,
        OptimizationTier::Baseline,
        TraceCompileOptions {
            count_links: true,
            ..TraceCompileOptions::default()
        },
    )
    .unwrap();
    for (trace, counted) in [(&plain, false), (&counting, true)] {
        let mut area = vec![0_i64; trace.exit_area_len()];
        let mut control = TraceControl::new(1_000_000);
        let outcome = trace.run(&inputs(trace), &mut control, &mut area);
        assert!(matches!(outcome, TraceOutcome::Exit { .. }));
        if counted {
            // i % 7 < 3 holds for 3 of every 7 iterations. The root path
            // runs when it fails, so the side runs for the rest.
            let (_, _, side_entries) = reference(&ir, &registers);
            assert_eq!(control.links, i64::from(side_entries));
            assert!(control.links > 300);
        } else {
            assert_eq!(control.links, 0);
        }
    }
    // Counting changes nothing else.
    assert_eq!(plain.exits(), counting.exits());
}

#[test]
fn a_side_that_leaves_the_loop_exits_through_its_own_terminator() {
    // The side ends in an exit instead of a link: `then` is a `break`.
    let limit = 100.0;
    let mut ir = root_loop(limit);
    ir.header_live = vec![0, 1, 2];
    let mut b = TraceBuilder::new(TraceId(1), frame());
    b.mark_written(FrameId(0), 0);
    let yes = b.input(0, TraceType::Float);
    let no = b.input(1, TraceType::Float);
    let i = b.input(2, TraceType::Float);
    let yes2 = arithmetic(&mut b, 3, ArithmeticOp::Add, yes, i);
    let out = b.snapshot(snapshot(
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
        ir: Box::new(b.finish(TraceTerminator::Exit { snapshot: out })),
    });
    ir.verify().unwrap();
    for tier in TIERS {
        let trace = compile_trace(&ir, tier).unwrap();
        assert_eq!(trace.member_count(), 2);
        for start in [1.0, 2.0, 4.0, 8.0] {
            let registers = [f(5.0), f(6.0), f(start), TraceScalar::Nil];
            let (outcome, pc, state, _) = run(&trace, &registers, 10_000);
            let TraceOutcome::Exit { index } = outcome else {
                panic!("it exits");
            };
            let (expected_pc, expected, _) = reference(&ir, &registers);
            assert_eq!(pc, expected_pc, "{start} {tier:?}");
            assert_eq!(state, expected, "{start} {tier:?}");
            // The first i with i % 7 < 3 ends the loop through the side.
            if trace.exits()[index].member == 1 {
                assert_eq!(pc, 12);
            }
        }
    }
}

// -------------------------------------------------------- window growth

/// The `then` side of the first tree, reading an upvalue cell the root never
/// names: `yes = yes + cell`. The cell is window register 4.
fn then_side_with_cell(limit: f64, written: bool) -> TraceIr {
    let window = TraceFrameInfo {
        register_count: 5,
        code_len: 16,
        inlined: None,
        activation: None,
    };
    let mut b = TraceBuilder::new(TraceId(1), window);
    b.set_external(vec![ExternalSlot::Upvalue(0)]);
    for register in [0, 2] {
        b.mark_written(FrameId(0), register);
    }
    if written {
        b.mark_written(FrameId(0), 4);
    }
    let yes = b.input(0, TraceType::Float);
    let no = b.input(1, TraceType::Float);
    let i = b.input(2, TraceType::Float);
    let cell = b.input(4, TraceType::Float);
    let yes2 = arithmetic(&mut b, 3, ArithmeticOp::Add, yes, cell);
    // `cell = cell + 1` when the side writes it.
    let one = constant(&mut b, 3, 1.0);
    let cell2 = if written {
        arithmetic(&mut b, 3, ArithmeticOp::Add, cell, one)
    } else {
        cell
    };
    let next = arithmetic(&mut b, 6, ArithmeticOp::Add, i, one);
    let top = constant(&mut b, 6, limit);
    let more = compare(&mut b, 6, Comparison::LessEqual, next, top);
    let state = vec![
        RegisterEntry::Value(yes2),
        RegisterEntry::Value(no),
        RegisterEntry::Value(next),
        RegisterEntry::Const(TraceScalar::Nil),
        RegisterEntry::Value(cell2),
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
    let link = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        6,
        state,
    ));
    b.finish(TraceTerminator::Link { snapshot: link })
}

fn tree_with_cell(limit: f64, written: bool) -> TraceIr {
    let mut ir = root_loop(limit);
    ir.header_live = vec![0, 1, 2];
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(then_side_with_cell(limit, written)),
    });
    ir
}

fn five(a: f64, b: f64, i: f64, cell: f64) -> Vec<TraceScalar> {
    vec![f(a), f(b), f(i), TraceScalar::Nil, f(cell)]
}

#[test]
fn a_side_that_names_a_cell_grows_the_window_of_the_whole_tree() {
    for written in [false, true] {
        let ir = tree_with_cell(40.0, written);
        ir.verify().unwrap();
        let members = crate::jit_ir::trace::plan_tree(&ir).unwrap();
        // Every member now has the five-register window and the same cell.
        for member in &members {
            assert_eq!(member.ir.frames[0].register_count, 5, "{written}");
            assert_eq!(member.ir.external, vec![ExternalSlot::Upvalue(0)]);
            member.ir.verify().unwrap();
        }
        // The root reads the cell at entry (and carries it when it is written).
        assert!(
            members[0]
                .ir
                .entry
                .inputs
                .iter()
                .any(|input| input.register == 4),
            "{written}"
        );
        let carried = members[0]
            .ir
            .loop_params
            .iter()
            .any(|param| param.register == Some(4));
        assert_eq!(carried, written);
        for tier in TIERS {
            let trace = compile_trace(&ir, tier).unwrap();
            assert_eq!(trace.member_count(), 2);
            for start in [1.0, 2.0, 3.0, 5.0, 9.0] {
                let registers = five(1.0, 2.0, start, 0.5);
                let (outcome, pc, state, _) = run(&trace, &registers, 100_000);
                assert!(matches!(outcome, TraceOutcome::Exit { .. }));
                let evaluation = evaluate_tree(&ir, &registers, &[], 100_000);
                let restored = match evaluation.outcome {
                    Outcome::GuardFailed { restored, .. } | Outcome::Exited(restored) => restored,
                    other => panic!("{other:?}"),
                };
                assert_eq!(pc, restored.pc, "{start} {tier:?} {written}");
                assert_eq!(&state[..4], &restored.registers[..4], "{start} {tier:?}");
                // The reference leaves the cell where a root exit says
                // `Unchanged`: the interpreter register, stale after the
                // side wrote it. The native tree has the cell's value.
                if written {
                    let (_, side_entries) = (restored.pc, evaluation.side_entries);
                    assert_eq!(
                        state[4],
                        f(0.5 + f64::from(side_entries)),
                        "{start} {tier:?}"
                    );
                } else {
                    assert_eq!(state[4], f(0.5));
                }
            }
        }
    }
}

#[test]
fn two_sides_may_extend_the_window_differently_if_they_agree_on_its_start() {
    let mut ir = tree_with_cell(40.0, false);
    // A second side with a window that disagrees about slot 0.
    let mut other = then_side_with_cell(40.0, false);
    other.external = vec![ExternalSlot::Upvalue(3)];
    ir.sides.push(TraceSide {
        parent: None,
        guard: 1,
        ir: Box::new(other),
    });
    assert_eq!(
        crate::jit_ir::trace::plan_tree(&ir),
        Err(crate::jit_ir::trace::TreeError::WindowMismatch { side: 1 })
    );
}

#[test]
fn sides_extend_the_pinned_callables_and_must_agree_on_their_start() {
    let native = crate::jit_ir::trace::pinned_native_identity(7);
    // A side that pins a native the root does not: the tree gets it, behind
    // the root's lanes, and every member lists it.
    let mut ir = tree_with_cell(40.0, false);
    ir.pinned_closures = vec![0xabc];
    ir.sides[0].ir.pinned_closures = vec![0xabc, native];
    let members = crate::jit_ir::trace::plan_tree(&ir).unwrap();
    assert!(
        members
            .iter()
            .all(|member| member.ir.pinned_closures == vec![0xabc, native])
    );
    // Two sides that put different callables at one lane: the later one was
    // recorded against another tree.
    let mut ir = tree_with_cell(40.0, false);
    ir.pinned_closures = vec![0xabc];
    ir.sides[0].ir.pinned_closures = vec![0xabc, native];
    let mut other = then_side_with_cell(40.0, false);
    other.pinned_closures = vec![0xabc, crate::jit_ir::trace::pinned_native_identity(8)];
    ir.sides.push(TraceSide {
        parent: None,
        guard: 1,
        ir: Box::new(other),
    });
    assert_eq!(
        crate::jit_ir::trace::plan_tree(&ir),
        Err(crate::jit_ir::trace::TreeError::PinMismatch { side: 1 })
    );
}

#[test]
fn sides_compiled_apart_must_agree_on_the_lanes_they_pin_beyond_the_tree() {
    // Two sides compiled apart that pin different callables behind the root's
    // lanes: their pins stay out of the tree, but the run pins one list for
    // both, so the second would read the first's callable at its lane.
    let mut ir = tree_with_cell(40.0, false);
    ir.pinned_closures = vec![0xabc];
    ir.sides[0].ir.pinned_closures = vec![0xabc, crate::jit_ir::trace::pinned_native_identity(7)];
    let mut other = then_side_with_cell(40.0, false);
    other.pinned_closures = vec![0xabc, crate::jit_ir::trace::pinned_native_identity(8)];
    ir.sides.push(TraceSide {
        parent: None,
        guard: 1,
        ir: Box::new(other),
    });
    assert_eq!(
        crate::jit_ir::trace::plan_tree_with(&ir, &[], &[0, 1]),
        Err(crate::jit_ir::trace::TreeError::PinMismatch { side: 1 })
    );
    // One side apart that pins past the root stays compatible with an inline
    // side that pins only the root's lanes.
    ir.sides[1].ir.pinned_closures = vec![0xabc];
    assert!(crate::jit_ir::trace::plan_tree_with(&ir, &[], &[0]).is_ok());
}

// ------------------------------------------------------- forced failures

/// Every guard of every member of `ir`, as (side, origin, pc).
fn tree_guards(ir: &TraceIr) -> Vec<(Option<usize>, u32, usize)> {
    let mut all: Vec<(Option<usize>, u32, usize)> = ir
        .guards
        .iter()
        .map(|info| (None, info.origin, info.site.pc))
        .collect();
    for (index, side) in ir.sides.iter().enumerate() {
        all.extend(
            side.ir
                .guards
                .iter()
                .map(|info| (Some(index), info.origin, info.site.pc)),
        );
    }
    all
}

#[test]
fn forcing_any_guard_of_a_tree_exits_like_the_reference_forcing_it() {
    use crate::jit_ir::trace::reference::{TreeGuard, evaluate_tree_forced};
    let trees = [
        ("two arms", tree(60.0), vec![0.0, 0.0, 1.0, 0.0]),
        ("three arms", three_way::tree(60.0), Vec::new()),
    ];
    let mut forced_exits = 0;
    let mut forced_links = 0;
    for (name, ir, _) in &trees {
        for tier in TIERS {
            let plain = compile_trace(ir, tier).unwrap();
            for (side, origin, _) in tree_guards(ir) {
                // Where the guard lives in the compiled code, if it does.
                let member = match side {
                    None => Some(0),
                    Some(index) => plain.side_members()[index],
                };
                let Some(member) = member else { continue };
                let Some(guard) = plain
                    .lowered_member(member)
                    .guards
                    .iter()
                    .position(|info| info.origin == origin)
                else {
                    continue;
                };
                let forced = compile_trace_forcing_member_guard(ir, tier, member, guard).unwrap();
                for start in 1..9 {
                    let registers = if ir.frames[0].register_count == 9 {
                        three_way::registers(0.5, 1.5, f64::from(start), 2.5)
                    } else {
                        vec![f(0.5), f(1.5), f(f64::from(start)), TraceScalar::Nil]
                    };
                    let (outcome, pc, state, _) = run(&forced, &registers, 100_000);
                    let TraceOutcome::Exit { .. } = outcome else {
                        panic!("{name}: {outcome:?}");
                    };
                    let evaluation = evaluate_tree_forced(
                        ir,
                        &registers,
                        &[],
                        100_000,
                        Some(TreeGuard { side, origin }),
                    );
                    let restored = match evaluation.outcome {
                        Outcome::GuardFailed { restored, .. } | Outcome::Exited(restored) => {
                            restored
                        }
                        other => panic!("{name}: {other:?}"),
                    };
                    assert_eq!(
                        pc, restored.pc,
                        "{name} {tier:?} forcing {side:?}/{origin} from {start}"
                    );
                    assert_eq!(
                        state, restored.registers,
                        "{name} {tier:?} forcing {side:?}/{origin} from {start}"
                    );
                    forced_exits += 1;
                    if evaluation.side_entries > 0 {
                        forced_links += 1;
                    }
                }
            }
        }
    }
    assert!(forced_exits > 100, "{forced_exits}");
    assert!(forced_links > 30, "{forced_links}");
}

// ------------------------------------------- invariants of a widened root

/// The root of `tree`, preceded by a check that depends on `yes` (r0) alone:
/// `yes * 2 < 100`. The root reads `yes` but never writes it, so recorded
/// alone it is an invariant entry input and the check moves to the
/// preheader. The side writes it, which makes it a loop parameter the root
/// passes on unchanged, and the check has to run again whenever the side
/// links back with a new `yes`.
fn root_with_a_check_on_yes(limit: f64) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), frame());
    for register in [1, 2, 3] {
        b.mark_written(FrameId(0), register);
    }
    let yes = b.input(0, TraceType::Float);
    let no_in = b.input(1, TraceType::Float);
    let i_in = b.input(2, TraceType::Float);
    let no = b.loop_param(TraceType::Float, LoopParamInit::Input(no_in));
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let two = constant(&mut b, 0, 2.0);
    let doubled = arithmetic(&mut b, 0, ArithmeticOp::Multiply, yes, two);
    let cap = constant(&mut b, 0, 100.0);
    let fits = compare(&mut b, 0, Comparison::LessThan, doubled, cap);
    // Leaves before the iteration, with the registers as they are.
    guard(
        &mut b,
        0,
        fits,
        true,
        SnapshotPhase::Before { pc: 0 },
        vec![
            RegisterEntry::Unchanged,
            RegisterEntry::Value(no),
            RegisterEntry::Value(i),
            RegisterEntry::Const(TraceScalar::Nil),
        ],
    );
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
    let poll = b.snapshot(snapshot(
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

pub(in crate::jit::trace) fn tree_with_a_check_on_yes(limit: f64) -> TraceIr {
    let mut ir = root_with_a_check_on_yes(limit);
    ir.header_live = vec![0, 1, 2];
    // The side is recorded on its own, so it holds none of the root's guards.
    // Its first guard continues the root's branch, which has origin 1 here.
    ir.sides.push(TraceSide {
        parent: None,
        guard: 1,
        ir: Box::new(then_side(limit)),
    });
    ir
}

#[test]
fn work_on_a_register_only_a_side_writes_runs_again_after_every_link() {
    let ir = tree_with_a_check_on_yes(400.0);
    let members = crate::jit_ir::trace::plan_tree(&ir).unwrap();
    let root = &members[0].ir;
    // `yes` is a loop parameter now, which the loop-back passes on as it is.
    assert_ne!(
        crate::jit_ir::trace::pass_through_params(root),
        [] as [usize; 0]
    );
    for tier in TIERS {
        let trace = compile_trace(&ir, tier).unwrap();
        assert_eq!(trace.member_count(), 2, "{tier:?}");
        if tier == OptimizationTier::Optimized {
            // The check depends on a pass-through parameter alone: it moved
            // out of the loop.
            let lowered = trace.lowered_ir();
            assert!(
                lowered
                    .preheader
                    .iter()
                    .any(|instruction| instruction.op.guard().is_some()),
                "the check on yes sits in the preheader"
            );
        }
        // Each pass through the side adds `i` to `yes`, so the check fails
        // some links after the start, at the header, before the iteration.
        for yes in [0.0, 10.0, 30.0, 47.0, 49.0, 50.0, 80.0] {
            for start in [1.0, 2.0, 4.0, 9.0, 11.0] {
                let registers = [f(yes), f(3.0), f(start), TraceScalar::Nil];
                let (outcome, pc, state, _) = run(&trace, &registers, 100_000);
                assert!(matches!(outcome, TraceOutcome::Exit { .. }));
                let (expected_pc, expected, _) = reference(&ir, &registers);
                assert_eq!(pc, expected_pc, "{tier:?} yes {yes} from {start}");
                assert_eq!(state, expected, "{tier:?} yes {yes} from {start}");
            }
        }
    }
}

#[test]
fn a_hoisted_check_leaves_through_the_state_the_link_brought() {
    // Start where the check holds and let the side drive `yes` past the cap.
    // The exit is the header's, with `yes` as the side left it: not the
    // interpreter's register, which still holds the value from before.
    let ir = tree_with_a_check_on_yes(400.0);
    let trace = compile_trace(&ir, OptimizationTier::Optimized).unwrap();
    let registers = [f(40.0), f(0.0), f(1.0), TraceScalar::Nil];
    let (outcome, pc, state, _) = run(&trace, &registers, 100_000);
    let TraceOutcome::Exit { index } = outcome else {
        panic!("it exits");
    };
    let descriptor = &trace.exits()[index];
    assert_eq!(descriptor.class, TraceExitClass::GuardFail);
    assert_eq!(pc, 0, "before the iteration, at the loop header");
    let TraceScalar::Float(yes) = state[0] else {
        panic!("{state:?}");
    };
    assert!(yes >= 50.0, "yes grew through the side: {state:?}");
    let (expected_pc, expected, _) = reference(&ir, &registers);
    assert_eq!((pc, state), (expected_pc, expected));
}

// -------------------------------------------------------------- loop sides

/// A side that ends in front of a loop of its own, and the loop as a side:
///
/// ```text
/// pc3   r0 = r0 + r2                 (the root's side, as `then_side`)
/// pc6   ... leaves in front of pc10, the loop's header
/// pc10  r0 = r0 + r2                 (the loop side)
/// pc11  r2 = r2 + 1; r2 <= cap loops back to pc10
/// ```
///
/// The root's `then` arm never links back: it runs the second loop to its
/// end, and the tree leaves through the loop's closing guard.
pub(in crate::jit::trace) mod loop_side {
    use super::*;
    use crate::jit_ir::trace::TERMINAL_GUARD;

    const HEADER: usize = 10;
    const LATCH: usize = 11;

    /// The `then` arm, which leaves in front of the loop at `HEADER`.
    fn leaving_side(limit: f64) -> TraceIr {
        let mut b = TraceBuilder::new(TraceId(1), frame());
        b.mark_written(FrameId(0), 0);
        let yes = b.input(0, TraceType::Float);
        let no = b.input(1, TraceType::Float);
        let i = b.input(2, TraceType::Float);
        let yes2 = arithmetic(&mut b, 3, ArithmeticOp::Add, yes, i);
        // A guard of its own, which the loop's trip cannot reach.
        let top = constant(&mut b, 6, limit);
        let below = compare(&mut b, 6, Comparison::LessThan, i, top);
        guard(
            &mut b,
            6,
            below,
            true,
            SnapshotPhase::After {
                pc: 6,
                successor: 9,
            },
            vec![
                RegisterEntry::Value(yes2),
                RegisterEntry::Value(no),
                RegisterEntry::Value(i),
                RegisterEntry::Const(TraceScalar::Nil),
            ],
        );
        let exit = b.snapshot(snapshot(
            SnapshotPhase::After {
                pc: LATCH + 1,
                successor: HEADER,
            },
            LATCH + 1,
            vec![
                RegisterEntry::Value(yes2),
                RegisterEntry::Value(no),
                RegisterEntry::Value(i),
                RegisterEntry::Const(TraceScalar::Nil),
            ],
        ));
        b.finish(TraceTerminator::Exit { snapshot: exit })
    }

    /// The loop: `yes = yes + i; i = i + 1` while `i <= cap`.
    fn the_loop(cap: f64) -> TraceIr {
        let mut b = TraceBuilder::new(TraceId(2), frame());
        for register in [0, 2] {
            b.mark_written(FrameId(0), register);
        }
        let yes_in = b.input(0, TraceType::Float);
        let no = b.input(1, TraceType::Float);
        let i_in = b.input(2, TraceType::Float);
        let yes = b.loop_param(TraceType::Float, LoopParamInit::Input(yes_in));
        let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
        let yes2 = arithmetic(&mut b, HEADER, ArithmeticOp::Add, yes, i);
        let one = constant(&mut b, LATCH, 1.0);
        let i2 = arithmetic(&mut b, LATCH, ArithmeticOp::Add, i, one);
        let top = constant(&mut b, LATCH, cap);
        let more = compare(&mut b, LATCH, Comparison::LessEqual, i2, top);
        let state = vec![
            RegisterEntry::Value(yes2),
            RegisterEntry::Value(no),
            RegisterEntry::Value(i2),
            RegisterEntry::Const(TraceScalar::Nil),
        ];
        guard(
            &mut b,
            LATCH,
            more,
            true,
            SnapshotPhase::After {
                pc: LATCH,
                successor: LATCH + 1,
            },
            state.clone(),
        );
        let poll = b.snapshot(snapshot(
            SnapshotPhase::After {
                pc: LATCH,
                successor: HEADER,
            },
            LATCH,
            state,
        ));
        b.set_poll_snapshot(poll);
        b.finish(TraceTerminator::LoopBack {
            args: vec![yes2, i2],
        })
    }

    pub(in crate::jit::trace) fn tree(limit: f64, cap: f64) -> TraceIr {
        let mut ir = root_loop(limit);
        ir.header_live = vec![0, 1, 2];
        ir.sides.push(TraceSide {
            parent: None,
            guard: 0,
            ir: Box::new(leaving_side(limit)),
        });
        ir.sides.push(TraceSide {
            parent: Some(0),
            guard: TERMINAL_GUARD,
            ir: Box::new(the_loop(cap)),
        });
        ir
    }

    #[test]
    fn the_tree_plans_and_every_member_verifies() {
        let ir = tree(100.0, 40.0);
        ir.verify().unwrap();
        let members = crate::jit_ir::trace::plan_tree(&ir).unwrap();
        assert_eq!(members.len(), 3);
        assert_eq!(members[2].parent, Some((1, TERMINAL_GUARD)));
        for member in &members {
            member.ir.verify().unwrap();
        }
        // The loop keeps its own parameters: the root's are not widened by it.
        assert_eq!(members[2].ir.loop_params.len(), 2);
    }

    #[test]
    fn a_loop_side_that_continues_nothing_is_refused() {
        // Not a terminal exit: the guard names none of the parent's.
        let mut ir = tree(100.0, 40.0);
        ir.sides[1].guard = 0;
        assert!(ir.verify().is_err());
        // The parent does not end in an exit.
        let mut ir = tree(100.0, 40.0);
        ir.sides[1].parent = None;
        assert!(ir.verify().is_err());
        // Two loops for one exit.
        let mut ir = tree(100.0, 40.0);
        let again = ir.sides[1].clone();
        ir.sides.push(again);
        assert!(ir.verify().is_err());
        // A counted tree has none.
        let mut ir = tree(100.0, 40.0);
        ir.sides[1].ir.counted = true;
        assert!(ir.verify().is_err());
    }

    #[test]
    fn the_compiled_tree_runs_the_loop_in_place_and_matches_the_reference() {
        let ir = tree(100.0, 40.0);
        for tier in TIERS {
            let trace = compile_trace(&ir, tier).unwrap();
            assert_eq!(trace.member_count(), 3, "{tier:?}");
            assert_eq!(trace.side_members(), &[Some(1), Some(2)], "{tier:?}");
            let polls = trace
                .exits()
                .iter()
                .filter(|exit| exit.class == TraceExitClass::Poll && exit.member == 2)
                .count();
            assert_eq!(polls, 1, "the loop polls once: {tier:?}");
            for start in [1.0, 2.0, 3.0, 4.0, 7.0, 15.0, 39.0, 41.0, 90.5] {
                let registers = [f(10.0), f(20.0), f(start), TraceScalar::Nil];
                let (outcome, pc, state, _) = run(&trace, &registers, 100_000);
                assert!(matches!(outcome, TraceOutcome::Exit { .. }), "{outcome:?}");
                let (expected_pc, expected, _) = reference(&ir, &registers);
                assert_eq!(pc, expected_pc, "start {start} at {tier:?}");
                assert_eq!(state, expected, "start {start} at {tier:?}");
            }
        }
    }

    #[test]
    fn random_start_states_match_the_reference_at_both_tiers() {
        let mut rng = Lcg(0x100b_51de);
        for (limit, cap) in [(10.0, 5.0), (64.0, 40.0), (200.0, 1_000.0)] {
            let ir = tree(limit, cap);
            for tier in TIERS {
                let trace = compile_trace(&ir, tier).unwrap();
                for _ in 0..60 {
                    let registers = [
                        f((rng.next() % 1_000) as f64 / 8.0),
                        f((rng.next() % 1_000) as f64 / 8.0),
                        f((rng.next() % 300) as f64 / 2.0 - 5.0),
                        TraceScalar::Nil,
                    ];
                    let (outcome, pc, state, _) = run(&trace, &registers, 1_000_000);
                    assert!(matches!(outcome, TraceOutcome::Exit { .. }));
                    let (expected_pc, expected, _) = reference(&ir, &registers);
                    assert_eq!(pc, expected_pc, "{registers:?} {limit}/{cap} {tier:?}");
                    assert_eq!(state, expected, "{registers:?} {limit}/{cap} {tier:?}");
                }
            }
        }
    }

    #[test]
    fn the_loop_spends_the_fuel_of_the_run_and_polls_when_it_is_out() {
        let ir = tree(100.0, 100_000.0);
        let trace = compile_trace(&ir, OptimizationTier::Optimized).unwrap();
        // Start where the root's guard sends the run into the arm: i % 7 < 3.
        let registers = [f(0.0), f(0.0), f(7.0), TraceScalar::Nil];
        let (outcome, pc, state, left) = run(&trace, &registers, 500);
        let TraceOutcome::Exit { index } = outcome else {
            panic!("the run leaves");
        };
        assert_eq!(trace.exits()[index].class, TraceExitClass::Poll);
        assert_eq!(trace.exits()[index].member, 2);
        assert_eq!(left, 0, "the loop used the run's fuel");
        assert_eq!(pc, HEADER, "the poll resumes at the loop's header");
        // The state is the one the interpreter has after those trips.
        assert_eq!(state[1], f(0.0));
        let TraceScalar::Float(i) = state[2] else {
            panic!("{state:?}");
        };
        assert!(i > 400.0, "the counter advanced {i}");
    }

    #[test]
    fn a_forced_guard_of_the_loop_leaves_through_its_own_exit() {
        let ir = tree(100.0, 1_000.0);
        // The loop's closing guard is the first (and only) guard of member 2.
        let trace =
            compile_trace_forcing_member_guard(&ir, OptimizationTier::Baseline, 2, 0).unwrap();
        let registers = [f(0.0), f(0.0), f(7.0), TraceScalar::Nil];
        let (outcome, pc, state, _) = run(&trace, &registers, 100_000);
        let TraceOutcome::Exit { index } = outcome else {
            panic!("the run leaves");
        };
        assert_eq!(trace.exits()[index].member, 2);
        assert_eq!(pc, LATCH + 1);
        let TraceScalar::Float(yes) = state[0] else {
            panic!("{state:?}");
        };
        assert_eq!(yes, 14.0, "one trip: yes = 7 + 7 from the arm and the loop");
    }

    /// The loop of [`the_loop`] with a branch at its header: `yes2 < wrap`
    /// stays in the loop's own code, the other arm is a side. With `call` the
    /// loop's bound is the host's answer for the register the loop never
    /// writes, a call the optimizer hoists into the loop's preheader.
    fn branching_loop(cap: f64, wrap: f64, call: bool) -> TraceIr {
        let mut b = TraceBuilder::new(TraceId(2), frame());
        for register in [0, 2] {
            b.mark_written(FrameId(0), register);
        }
        let yes_in = b.input(0, TraceType::Float);
        let no = b.input(1, TraceType::Float);
        let i_in = b.input(2, TraceType::Float);
        let yes = b.loop_param(TraceType::Float, LoopParamInit::Input(yes_in));
        let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
        let yes2 = arithmetic(&mut b, HEADER, ArithmeticOp::Add, yes, i);
        let bound = constant(&mut b, HEADER, wrap);
        let small = compare(&mut b, HEADER, Comparison::LessThan, yes2, bound);
        guard(
            &mut b,
            HEADER,
            small,
            true,
            SnapshotPhase::After {
                pc: HEADER,
                successor: LATCH,
            },
            vec![
                RegisterEntry::Value(yes2),
                RegisterEntry::Value(no),
                RegisterEntry::Value(i),
                RegisterEntry::Const(TraceScalar::Nil),
            ],
        );
        let one = constant(&mut b, LATCH, 1.0);
        let i2 = arithmetic(&mut b, LATCH, ArithmeticOp::Add, i, one);
        let top = if call {
            let before = b.snapshot(snapshot(
                SnapshotPhase::Before { pc: HEADER },
                HEADER,
                vec![
                    RegisterEntry::Value(yes),
                    RegisterEntry::Value(no),
                    RegisterEntry::Value(i),
                    RegisterEntry::Const(TraceScalar::Nil),
                ],
            ));
            let call_guard = b.guard(
                TraceSite {
                    frame: FrameId(0),
                    pc: HEADER,
                },
                before,
            );
            b.emit(
                TraceSite {
                    frame: FrameId(0),
                    pc: HEADER,
                },
                TraceOp::HostCall {
                    function: crate::jit_ir::trace::HostFunction::Native(7),
                    args: vec![no],
                    result: TraceType::Float,
                    guard: call_guard,
                },
                Some(TraceType::Float),
            )
            .unwrap()
        } else {
            constant(&mut b, LATCH, cap)
        };
        let more = compare(&mut b, LATCH, Comparison::LessEqual, i2, top);
        let state = vec![
            RegisterEntry::Value(yes2),
            RegisterEntry::Value(no),
            RegisterEntry::Value(i2),
            RegisterEntry::Const(TraceScalar::Nil),
        ];
        guard(
            &mut b,
            LATCH,
            more,
            true,
            SnapshotPhase::After {
                pc: LATCH,
                successor: LATCH + 1,
            },
            state.clone(),
        );
        let poll = b.snapshot(snapshot(
            SnapshotPhase::After {
                pc: LATCH,
                successor: HEADER,
            },
            LATCH,
            state,
        ));
        b.set_poll_snapshot(poll);
        b.finish(TraceTerminator::LoopBack {
            args: vec![yes2, i2],
        })
    }

    /// The side of the branch: `yes = yes - wrap`, then the latch, and back
    /// into the loop at its header. With `writes_no` it also sets `no`,
    /// which the loop reads from its entry state without carrying it.
    fn wrapping_side(cap: f64, wrap: f64, writes_no: bool) -> TraceIr {
        let mut b = TraceBuilder::new(TraceId(3), frame());
        for register in [0, 2] {
            b.mark_written(FrameId(0), register);
        }
        let yes = b.input(0, TraceType::Float);
        let mut no = b.input(1, TraceType::Float);
        let i = b.input(2, TraceType::Float);
        let bound = constant(&mut b, HEADER, wrap);
        let yes2 = arithmetic(&mut b, HEADER, ArithmeticOp::Subtract, yes, bound);
        if writes_no {
            b.mark_written(FrameId(0), 1);
            no = arithmetic(&mut b, HEADER, ArithmeticOp::Add, no, bound);
        }
        let one = constant(&mut b, LATCH, 1.0);
        let i2 = arithmetic(&mut b, LATCH, ArithmeticOp::Add, i, one);
        let top = constant(&mut b, LATCH, cap);
        let more = compare(&mut b, LATCH, Comparison::LessEqual, i2, top);
        let state = vec![
            RegisterEntry::Value(yes2),
            RegisterEntry::Value(no),
            RegisterEntry::Value(i2),
            RegisterEntry::Const(TraceScalar::Nil),
        ];
        guard(
            &mut b,
            LATCH,
            more,
            true,
            SnapshotPhase::After {
                pc: LATCH,
                successor: LATCH + 1,
            },
            state.clone(),
        );
        let link = b.snapshot(snapshot(
            SnapshotPhase::After {
                pc: LATCH,
                successor: HEADER,
            },
            LATCH,
            state,
        ));
        b.finish(TraceTerminator::LinkLoop {
            side: 1,
            snapshot: link,
        })
    }

    pub(in crate::jit::trace) fn linked_tree(
        limit: f64,
        cap: f64,
        wrap: f64,
        writes_no: bool,
    ) -> TraceIr {
        linked_tree_with(limit, cap, wrap, writes_no, false)
    }

    /// As [`linked_tree`]; with `call` the loop's bound is a host call its
    /// preheader makes (see `branching_loop`).
    pub(in crate::jit::trace) fn linked_tree_with(
        limit: f64,
        cap: f64,
        wrap: f64,
        writes_no: bool,
        call: bool,
    ) -> TraceIr {
        let mut ir = root_loop(limit);
        ir.header_live = vec![0, 1, 2];
        ir.sides.push(TraceSide {
            parent: None,
            guard: 0,
            ir: Box::new(leaving_side(limit)),
        });
        ir.sides.push(TraceSide {
            parent: Some(0),
            guard: TERMINAL_GUARD,
            ir: Box::new(branching_loop(cap, wrap, call)),
        });
        ir.sides.push(TraceSide {
            parent: Some(1),
            guard: 0,
            ir: Box::new(wrapping_side(cap, wrap, writes_no)),
        });
        ir
    }

    #[test]
    fn a_side_of_the_loop_rejoins_it_and_matches_the_reference() {
        let ir = linked_tree(100.0, 400.0, 500.0, false);
        ir.verify().unwrap();
        let members = crate::jit_ir::trace::plan_tree(&ir).unwrap();
        assert_eq!(members.len(), 4);
        for member in &members {
            member.ir.verify().unwrap();
        }
        for tier in TIERS {
            let trace = compile_trace(&ir, tier).unwrap();
            assert_eq!(trace.loop_links(), 1, "{tier:?}");
            for start in [1.0, 2.0, 3.0, 7.0, 15.0, 39.0, 90.5] {
                let registers = [f(10.0), f(20.0), f(start), TraceScalar::Nil];
                let (outcome, pc, state, _) = run(&trace, &registers, 1_000_000);
                assert!(matches!(outcome, TraceOutcome::Exit { .. }), "{outcome:?}");
                let (expected_pc, expected, _) = reference(&ir, &registers);
                assert_eq!(pc, expected_pc, "start {start} at {tier:?}");
                assert_eq!(state, expected, "start {start} at {tier:?}");
            }
        }
    }

    #[test]
    fn a_link_that_writes_what_the_loop_reads_from_its_entry_is_refused() {
        let ir = linked_tree(100.0, 400.0, 500.0, true);
        let refused = crate::jit_ir::trace::plan_tree(&ir);
        assert!(
            matches!(
                refused,
                Err(crate::jit_ir::trace::TreeError::LinkValue { register: 1, .. })
            ),
            "{refused:?}"
        );
    }

    #[test]
    fn a_link_to_a_side_that_is_no_loop_is_refused() {
        // Side 0 is the arm that leaves in front of the loop, not the loop.
        let mut ir = linked_tree(100.0, 400.0, 500.0, false);
        let TraceTerminator::LinkLoop { side, .. } = &mut ir.sides[2].ir.terminator else {
            panic!("the side links into the loop");
        };
        *side = 0;
        assert!(ir.verify().is_err() || crate::jit_ir::trace::plan_tree(&ir).is_err());
    }
}

// Sides compiled apart from their tree, called through the cell of the exit
// they continue.

/// The tree whose root reads `yes`, so a side that writes it widens nothing
/// when it is compiled apart and leaves at the loop header.
fn tree_reading_yes(limit: f64) -> TraceIr {
    let mut ir = root_loop_full(limit, false, true);
    ir.header_live = vec![0, 1, 2];
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(then_side(limit)),
    });
    ir
}

/// The root of [`tree_reading_yes`] compiled alone, and the exit its branch
/// guard leaves through.
fn lone_root(limit: f64, tier: OptimizationTier) -> (CompiledTrace, usize) {
    let mut ir = tree_reading_yes(limit);
    ir.sides.clear();
    let trace = compile_trace(&ir, tier).unwrap();
    let exit = trace
        .exits()
        .iter()
        .position(|exit| exit.origin == Some(0) && exit.class == TraceExitClass::GuardFail)
        .expect("the branch guard has an exit");
    (trace, exit)
}

/// Runs `trace` from `registers`, entering it again at each poll exit with
/// the state it restores, as the VM does, until it leaves for another reason.
fn run_through_polls(
    trace: &CompiledTrace,
    registers: &[TraceScalar],
) -> (usize, Vec<TraceScalar>, u32) {
    let mut state = registers.to_vec();
    let mut polls = 0;
    loop {
        let (outcome, pc, restored, _) = run(trace, &state, 1_000_000);
        let TraceOutcome::Exit { index } = outcome else {
            panic!("entry types hold: {outcome:?}");
        };
        state = restored;
        if trace.exits()[index].class == TraceExitClass::Poll {
            polls += 1;
            continue;
        }
        return (pc, state, polls);
    }
}

#[test]
fn a_side_compiled_apart_leaves_at_the_loop_header_and_matches_the_tree() {
    let ir = tree_reading_yes(100.0);
    for tier in TIERS {
        let (mut trace, exit) = lone_root(100.0, tier);
        let own = trace.exits().len();
        let context = trace.side_context(exit, 1).expect("the exit takes a side");
        let side = compile_side(
            &ir,
            &[0],
            &[0],
            0,
            true,
            &context,
            tier,
            TraceCompileOptions::default(),
        )
        .unwrap();
        trace.attach(side, 0).unwrap();
        assert_eq!(trace.attached_sides(), 1);
        assert_eq!(trace.member_count(), 2);
        assert_eq!(trace.side_members(), &[Some(1)]);
        assert!(trace.exits().len() > own);
        assert!(!trace.can_attach_at(exit), "the exit has its side");
        let mut polls_seen = false;
        for start in [1.0, 2.0, 3.0, 4.0, 7.0, 50.5, 99.0] {
            let registers = [f(10.0), f(20.0), f(start), TraceScalar::Nil];
            let (pc, state, polls) = run_through_polls(&trace, &registers);
            let (expected_pc, expected, _) = reference(&ir, &registers);
            assert_eq!(pc, expected_pc, "start {start} at {tier:?}");
            assert_eq!(state, expected, "start {start} at {tier:?}");
            polls_seen |= polls > 0;
        }
        assert!(polls_seen, "the side leaves at the loop header at {tier:?}");
    }
}

#[test]
fn clif_instructions_of_a_side_compiled_apart_are_fewer_than_the_tree_that_holds_it() {
    // Asserts what Cranelift emits.
    crate::jit::debug_env::cranelift_for_this_test();
    let ir = tree_reading_yes(100.0);
    let whole = compile_trace(&ir, OptimizationTier::Optimized).unwrap();
    let (trace, exit) = lone_root(100.0, OptimizationTier::Optimized);
    let context = trace.side_context(exit, 1).unwrap();
    let side = compile_side(
        &ir,
        &[0],
        &[0],
        0,
        true,
        &context,
        OptimizationTier::Optimized,
        TraceCompileOptions::default(),
    )
    .unwrap();
    assert!(side.clif_insts() > 0);
    assert!(
        side.clif_insts() < whole.clif_insts(),
        "{} side against {} for the tree",
        side.clif_insts(),
        whole.clif_insts()
    );
}

/// A side that adds to `no` instead of `yes`: it writes only registers the
/// root already carries, so it links back to the loop without widening it.
fn no_side(limit: f64) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(1), frame());
    for register in [1, 2] {
        b.mark_written(FrameId(0), register);
    }
    let yes = b.input(0, TraceType::Float);
    let no = b.input(1, TraceType::Float);
    let i = b.input(2, TraceType::Float);
    let no2 = arithmetic(&mut b, 3, ArithmeticOp::Add, no, i);
    let one = constant(&mut b, 6, 1.0);
    let i2 = arithmetic(&mut b, 6, ArithmeticOp::Add, i, one);
    let top = constant(&mut b, 6, limit);
    let more = compare(&mut b, 6, Comparison::LessEqual, i2, top);
    let registers = || {
        vec![
            RegisterEntry::Value(yes),
            RegisterEntry::Value(no2),
            RegisterEntry::Value(i2),
            RegisterEntry::Const(TraceScalar::Nil),
        ]
    };
    guard(
        &mut b,
        6,
        more,
        true,
        SnapshotPhase::After {
            pc: 6,
            successor: 7,
        },
        registers(),
    );
    let link = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        6,
        registers(),
    ));
    b.finish(TraceTerminator::Link { snapshot: link })
}

fn tree_with_no_side(limit: f64) -> TraceIr {
    let mut ir = root_loop_full(limit, false, true);
    ir.header_live = vec![0, 1, 2];
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(no_side(limit)),
    });
    ir
}

fn attach_no_side(
    limit: f64,
    tier: OptimizationTier,
    count_links: bool,
) -> (CompiledTrace, TraceIr) {
    let ir = tree_with_no_side(limit);
    let mut lone = ir.clone();
    lone.sides.clear();
    let options = TraceCompileOptions {
        count_links,
        ..TraceCompileOptions::default()
    };
    let mut trace = compile_trace_with(&lone, tier, options).unwrap();
    let exit = trace
        .exits()
        .iter()
        .position(|exit| exit.origin == Some(0) && exit.class == TraceExitClass::GuardFail)
        .expect("the branch guard has an exit");
    let context = trace.side_context(exit, 3).expect("the exit takes a side");
    let side = compile_side(&ir, &[], &[0], 0, false, &context, tier, options).unwrap();
    trace.attach(side, 0).unwrap();
    (trace, ir)
}

#[test]
fn a_side_compiled_apart_links_back_to_the_loop_natively() {
    for tier in TIERS {
        let (trace, ir) = attach_no_side(1_000.0, tier, false);
        let registers = [f(0.0), f(0.0), f(1.0), TraceScalar::Nil];
        let (outcome, pc, state, left) = run(&trace, &registers, 1_000_000);
        let TraceOutcome::Exit { index } = outcome else {
            panic!("the loop exits");
        };
        // The hundreds of alternations never left native code: the only exit
        // is the loop's own end, in the root.
        assert_eq!(
            trace.exits()[index].class,
            TraceExitClass::LoopComplete,
            "{tier:?}"
        );
        assert_eq!(trace.exits()[index].member, 0, "{tier:?}");
        assert_eq!(
            1_000_000 - left,
            999,
            "one unit of fuel per iteration at {tier:?}"
        );
        let (expected_pc, expected, side_entries) = reference(&ir, &registers);
        assert_eq!((pc, state), (expected_pc, expected), "{tier:?}");
        assert!(side_entries > 100);
    }
}

#[test]
fn a_side_compiled_apart_matches_the_reference_from_many_states() {
    for tier in TIERS {
        let (trace, ir) = attach_no_side(40.0, tier, false);
        for start in [1.0, 2.0, 3.0, 7.0, 8.0, 15.5, 39.0, 40.0] {
            for fuel in [1, 2, 3, 100] {
                let registers = [f(10.0), f(20.0), f(start), TraceScalar::Nil];
                // Poll every `fuel` iterations and enter again.
                let mut state = registers.to_vec();
                let (pc, state) = loop {
                    let (outcome, pc, restored, _) = run(&trace, &state, fuel);
                    let TraceOutcome::Exit { index } = outcome else {
                        panic!("entry types hold");
                    };
                    state = restored;
                    if trace.exits()[index].class != TraceExitClass::Poll {
                        break (pc, state);
                    }
                };
                let (expected_pc, expected, _) = reference(&ir, &registers);
                assert_eq!(
                    (pc, state),
                    (expected_pc, expected),
                    "start {start} fuel {fuel} {tier:?}"
                );
            }
        }
    }
}

#[test]
fn a_poll_inside_a_side_compiled_apart_has_the_sides_exit() {
    let (trace, _) = attach_no_side(100.0, OptimizationTier::Optimized, false);
    let registers = [f(5.0), f(6.0), f(1.0), TraceScalar::Nil];
    // i = 1 takes the side, and a fuel of 1 is spent at its link.
    let (outcome, pc, state, left) = run(&trace, &registers, 1);
    let TraceOutcome::Exit { index } = outcome else {
        panic!("it exits");
    };
    let descriptor = &trace.exits()[index];
    assert_eq!(descriptor.class, TraceExitClass::Poll);
    assert_eq!(descriptor.member, 1);
    assert_eq!(left, 0);
    assert_eq!(pc, 0);
    // no = 6 + 1, i = 2.
    assert_eq!(state[1], f(7.0));
    assert_eq!(state[2], f(2.0));
}

#[test]
fn the_calls_of_a_side_compiled_apart_are_counted() {
    // The run counts them whether or not the tree's code counts the jumps into
    // the sides it holds.
    for count in [false, true] {
        let (trace, _) = attach_no_side(100.0, OptimizationTier::Optimized, count);
        let registers = [f(0.0), f(0.0), f(1.0), TraceScalar::Nil];
        let inputs: Vec<TraceInputSlot> = trace
            .entry_contract()
            .iter()
            .map(|(register, _)| TraceInputSlot::from_scalar(registers[usize::from(*register)]))
            .collect();
        let mut area = vec![0_i64; trace.exit_area_len()];
        let mut control = TraceControl::new(1_000);
        trace.run(&inputs, &mut control, &mut area);
        // Native code counts the calls it makes itself apart.
        let calls = control.child_calls + control.native_calls;
        assert!(calls > 10, "{count}");
        assert_eq!(control.links, calls, "{count}");
    }
}

/// The root of [`tree`] with a fifth register, `cell`, that every iteration
/// passes on unchanged, as the root of a nest does for the registers of the
/// loops around it. Such a root links natively.
fn root_loop_passing_cell(limit: f64) -> TraceIr {
    let window = TraceFrameInfo {
        register_count: 5,
        code_len: 16,
        inlined: None,
        activation: None,
    };
    let mut b = TraceBuilder::new(TraceId(0), window);
    for register in [1, 2, 3] {
        b.mark_written(FrameId(0), register);
    }
    let no_in = b.input(1, TraceType::Float);
    let i_in = b.input(2, TraceType::Float);
    let cell_in = b.input(4, TraceType::Float);
    let no = b.loop_param(TraceType::Float, LoopParamInit::Input(no_in));
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let cell = b.loop_param(TraceType::Float, LoopParamInit::Input(cell_in));
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
            RegisterEntry::Value(cell),
        ],
    );
    let no2 = arithmetic(&mut b, 5, ArithmeticOp::Add, no, i);
    let one = constant(&mut b, 6, 1.0);
    let i2 = arithmetic(&mut b, 6, ArithmeticOp::Add, i, one);
    let top = constant(&mut b, 6, limit);
    let more = compare(&mut b, 6, Comparison::LessEqual, i2, top);
    let state = vec![
        RegisterEntry::Unchanged,
        RegisterEntry::Value(no2),
        RegisterEntry::Value(i2),
        RegisterEntry::Value(c),
        RegisterEntry::Value(cell),
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
    let poll = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        6,
        state,
    ));
    b.set_poll_snapshot(poll);
    b.finish(TraceTerminator::LoopBack {
        args: vec![no2, i2, cell],
    })
}

#[test]
fn a_side_that_declines_is_called_once_and_counted_once() {
    // The side was recorded for a float in `yes` (r0), which the root never
    // reads, and the run holds nil: the side declines, the guard's exit is
    // the one the run leaves through, and the side is called once, whether
    // native code or the run calls it.
    let tier = OptimizationTier::Optimized;
    for count in [false, true] {
        let options = TraceCompileOptions {
            count_links: count,
            ..TraceCompileOptions::default()
        };
        let mut ir = root_loop_passing_cell(100.0);
        ir.header_live = vec![0, 1, 2, 4];
        ir.sides.push(TraceSide {
            parent: None,
            guard: 0,
            ir: Box::new(then_side_with_cell(100.0, false)),
        });
        let mut lone = ir.clone();
        lone.sides.clear();
        let mut trace = compile_trace_with(&lone, tier, options).unwrap();
        assert!(trace.links_natively(), "{count}");
        let exit = trace
            .exits()
            .iter()
            .position(|exit| exit.origin == Some(0) && exit.class == TraceExitClass::GuardFail)
            .expect("the branch guard has an exit");
        let context = trace.side_context(exit, 2).unwrap();
        let side = compile_side(&ir, &[0], &[0], 0, true, &context, tier, options).unwrap();
        trace.attach(side, 0).unwrap();
        let registers = [TraceScalar::Nil, f(20.0), f(1.0), TraceScalar::Nil, f(0.5)];
        let extras = trace.extras().iter().map(|(source, _)| match source {
            crate::jit::ExtraSource::Register(register) => *register,
            _ => unreachable!("these trees have no caller"),
        });
        let inputs: Vec<TraceInputSlot> = trace
            .entry_contract()
            .iter()
            .map(|(register, _)| *register)
            .chain(extras)
            .map(|register| TraceInputSlot::from_scalar(registers[usize::from(register)]))
            .collect();
        let mut area = vec![0_i64; trace.exit_area_len()];
        let mut control = TraceControl::new(1_000);
        let outcome = trace.run(&inputs, &mut control, &mut area);
        let TraceOutcome::Exit { index } = outcome else {
            panic!("it exits: {outcome:?}");
        };
        assert_eq!(
            trace.exits()[index].class,
            TraceExitClass::GuardFail,
            "{count}"
        );
        assert_eq!(control.native_calls, 1, "{count}");
        assert_eq!(control.child_calls, 0, "{count}");
        assert_eq!(control.links, 1, "{count}");
    }
}

/// The tree of [`root_loop_passing_cell`] with a side that always links back
/// (it accepts the float in r0), as a lone root plus the side compiled apart.
fn attach_cell_side(limit: f64, tier: OptimizationTier) -> (CompiledTrace, TraceIr) {
    let options = TraceCompileOptions::default();
    let mut ir = root_loop_passing_cell(limit);
    ir.header_live = vec![0, 1, 2, 4];
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(then_side_with_cell(limit, false)),
    });
    let mut lone = ir.clone();
    lone.sides.clear();
    let mut trace = compile_trace_with(&lone, tier, options).unwrap();
    assert!(trace.links_natively(), "{tier:?}");
    let exit = trace
        .exits()
        .iter()
        .position(|exit| exit.origin == Some(0) && exit.class == TraceExitClass::GuardFail)
        .expect("the branch guard has an exit");
    let context = trace.side_context(exit, 2).unwrap();
    let side = compile_side(&ir, &[0], &[0], 0, true, &context, tier, options).unwrap();
    trace.attach(side, 0).unwrap();
    (trace, ir)
}

/// Enters `trace` with `registers` and `fuel`, returning the control block
/// too, which `run` does not.
fn run_counting(
    trace: &CompiledTrace,
    registers: &[TraceScalar],
    fuel: i64,
) -> (TraceOutcome, usize, Vec<TraceScalar>, TraceControl) {
    let extras = trace.extras().iter().map(|(source, _)| match source {
        crate::jit::ExtraSource::Register(register) => *register,
        _ => unreachable!("these trees have no caller"),
    });
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| *register)
        .chain(extras)
        .map(|register| TraceInputSlot::from_scalar(registers[usize::from(register)]))
        .collect();
    let mut area = vec![0_i64; trace.exit_area_len()];
    let mut control = TraceControl::new(fuel);
    let outcome = trace.run(&inputs, &mut control, &mut area);
    let TraceOutcome::Exit { index } = outcome else {
        panic!("it exits: {outcome:?}");
    };
    let (pc, state) = restored(trace, index, &area, registers);
    (outcome, pc, state, control)
}

#[test]
fn fuel_is_polled_on_the_native_link_path() {
    // A root that links natively and a side that always links back keep one
    // run inside native code; only the fuel can end it, and it must end it
    // after at most `fuel` calls.
    let tier = OptimizationTier::Optimized;
    let (endless, _) = attach_cell_side(1e9, tier);
    let registers = [f(0.0), f(20.0), f(1.0), TraceScalar::Nil, f(0.5)];
    for fuel in [1, 2, 3, 64] {
        let (outcome, _, _, control) = run_counting(&endless, &registers, fuel);
        let TraceOutcome::Exit { index } = outcome else {
            unreachable!()
        };
        assert_eq!(endless.exits()[index].class, TraceExitClass::Poll, "{fuel}");
        assert!(
            control.native_calls > 0,
            "{fuel}: the side was called natively"
        );
        assert!(
            control.native_calls <= fuel,
            "{fuel}: {}",
            control.native_calls
        );
    }
    // Entering again at each poll reaches the reference state.
    let (trace, ir) = attach_cell_side(40.0, tier);
    for fuel in [1, 2, 3, 64] {
        let mut state = registers.to_vec();
        let (pc, state) = loop {
            let (outcome, pc, restored, _) = run_counting(&trace, &state, fuel);
            let TraceOutcome::Exit { index } = outcome else {
                unreachable!()
            };
            state = restored;
            if trace.exits()[index].class != TraceExitClass::Poll {
                break (pc, state);
            }
        };
        let (expected_pc, expected, _) = reference(&ir, &registers);
        assert_eq!((pc, state), (expected_pc, expected), "fuel {fuel}");
    }
}

#[test]
fn a_side_compiled_apart_reads_a_register_the_root_does_not_from_an_extra_input() {
    // The side adds to `yes` (r0), which the root never reads: a tree
    // compiled together takes it as an entry input, and a side compiled apart
    // from a root that is already compiled reads it where the run keeps it.
    let ir = tree(100.0);
    for tier in TIERS {
        let mut lone = ir.clone();
        lone.sides.clear();
        let mut trace = compile_trace(&lone, tier).unwrap();
        assert_eq!(trace.extras().len(), 0);
        let exit = trace
            .exits()
            .iter()
            .position(|exit| exit.origin == Some(0) && exit.class == TraceExitClass::GuardFail)
            .expect("the branch guard has an exit");
        let context = trace.side_context(exit, 2).unwrap();
        let side = compile_side(
            &ir,
            &[0],
            &[0],
            0,
            true,
            &context,
            tier,
            TraceCompileOptions::default(),
        )
        .unwrap();
        trace.attach(side, 0).unwrap();
        assert_eq!(
            trace.extras(),
            &[(crate::jit::ExtraSource::Register(0), TraceType::Float)],
            "{tier:?}"
        );
        assert!(
            !trace
                .entry_contract()
                .iter()
                .any(|(register, _)| *register == 0)
        );
        // The side runs when the register holds a float, and the tree
        // gives the interpreter's answer.
        for start in [1.0, 2.0, 3.0, 8.0, 15.0, 50.5] {
            let registers = [f(10.0), f(20.0), f(start), TraceScalar::Nil];
            let (pc, state, polls) = run_through_polls(&trace, &registers);
            let (expected_pc, expected, _) = reference(&ir, &registers);
            assert_eq!(
                (pc, state),
                (expected_pc, expected),
                "start {start} {tier:?}"
            );
            assert!(polls > 0 || (3.0..7.0).contains(&start) || start > 50.0);
        }
        // Another type than the side was recorded for: it declines, and the
        // guard's exit is taken as if the side did not exist. The branch
        // is taken at i = 1, where `yes` would be read.
        let registers = [TraceScalar::Nil, f(20.0), f(1.0), TraceScalar::Nil];
        let (outcome, pc, _, _) = run(&trace, &registers, 1_000);
        let TraceOutcome::Exit { index } = outcome else {
            panic!("it exits");
        };
        let descriptor = &trace.exits()[index];
        assert_eq!(descriptor.member, 0, "{tier:?}");
        assert_eq!(descriptor.class, TraceExitClass::GuardFail, "{tier:?}");
        assert_eq!(pc, 3, "{tier:?}");
    }
}

#[test]
fn a_side_cannot_be_attached_twice_to_an_exit_or_out_of_order() {
    let ir = tree_reading_yes(100.0);
    let tier = OptimizationTier::Optimized;
    let (mut trace, exit) = lone_root(100.0, tier);
    let context = trace.side_context(exit, 1).unwrap();
    let compile = || {
        compile_side(
            &ir,
            &[0],
            &[0],
            0,
            true,
            &context,
            tier,
            TraceCompileOptions::default(),
        )
        .unwrap()
    };
    // A side that is not the next of the tree's sides.
    assert_eq!(trace.attach(compile(), 3), Err(AttachError::Order));
    assert_eq!(trace.attached_sides(), 0);
    assert_eq!(trace.member_count(), 1);
    // One that is taken out before it is published leaves the tree as it
    // was, and its exit takes no side.
    let exits = trace.exits().len();
    trace.prepare_attach(compile(), 0).unwrap();
    assert!(!trace.can_attach_at(exit) || trace.exits().len() > exits);
    trace.cancel_attached();
    assert_eq!(trace.exits().len(), exits);
    assert_eq!(trace.member_count(), 1);
    assert_eq!(trace.attached_sides(), 0);
    assert!(trace.can_attach_at(exit));
    trace.attach(compile(), 0).unwrap();
    assert!(!trace.can_attach_at(exit));
    assert_eq!(trace.attach(compile(), 1), Err(AttachError::NoCell));
    assert!(trace.side_context(exit, 5).is_none());
}

#[test]
fn a_tree_compiled_from_its_plan_matches_one_that_plans_again() {
    let ir = tree(100.0);
    let planned = crate::jit_ir::trace::plan_tree(&ir).unwrap();
    for tier in TIERS {
        let fresh = compile_trace_with(&ir, tier, TraceCompileOptions::default()).unwrap();
        let from_plan =
            compile_trace_with_plan(&ir, tier, TraceCompileOptions::default(), Some(&planned))
                .unwrap();
        assert_eq!(from_plan.member_count(), fresh.member_count(), "{tier:?}");
        assert_eq!(from_plan.side_members(), fresh.side_members(), "{tier:?}");
        assert_eq!(from_plan.exits().len(), fresh.exits().len(), "{tier:?}");
        for start in [1.0, 7.0, 99.0] {
            let registers = [f(10.0), f(20.0), f(start), TraceScalar::Nil];
            let (_, pc, state, _) = run(&fresh, &registers, 100_000);
            let (_, from_pc, from_state, _) = run(&from_plan, &registers, 100_000);
            assert_eq!(
                (from_pc, from_state),
                (pc, state),
                "start {start} at {tier:?}"
            );
        }
    }
}

#[test]
fn a_side_compiled_from_its_plan_matches_one_that_plans_again() {
    for tier in TIERS {
        let ir = tree_with_no_side(40.0);
        let mut lone = ir.clone();
        lone.sides.clear();
        let options = TraceCompileOptions::default();
        let planned = crate::jit_ir::trace::plan_tree_with(&ir, &[], &[0]).unwrap();
        let mut traces = Vec::new();
        for plan in [None, Some(planned.as_slice())] {
            let mut trace = compile_trace_with(&lone, tier, options).unwrap();
            let exit = trace
                .exits()
                .iter()
                .position(|exit| exit.origin == Some(0) && exit.class == TraceExitClass::GuardFail)
                .expect("the branch guard has an exit");
            let context = trace.side_context(exit, 3).expect("the exit takes a side");
            let side =
                compile_side_with_plan(&ir, &[], &[0], 0, false, &context, tier, options, plan)
                    .unwrap();
            trace.attach(side, 0).unwrap();
            traces.push(trace);
        }
        let (fresh, from_plan) = (&traces[0], &traces[1]);
        assert_eq!(from_plan.exits().len(), fresh.exits().len(), "{tier:?}");
        assert_eq!(from_plan.side_members(), fresh.side_members(), "{tier:?}");
        for start in [1.0, 2.0, 7.0, 39.0] {
            let registers = [f(10.0), f(20.0), f(start), TraceScalar::Nil];
            let expected = run(fresh, &registers, 100_000);
            let actual = run(from_plan, &registers, 100_000);
            assert_eq!(
                (actual.1, actual.2),
                (expected.1, expected.2),
                "start {start} at {tier:?}"
            );
        }
    }
}

#[test]
fn census_counts_the_ops_guards_and_pressure_of_each_member() {
    use super::census::{member_census, op_name};
    let ir = tree_reading_yes(100.0);
    for member in std::iter::once(&ir).chain(ir.sides.iter().map(|side| &*side.ir)) {
        let census = member_census(member);
        assert_eq!(census.preheader, member.preheader.len());
        assert_eq!(census.body, member.body.len());
        let total: u64 = census.ops.values().sum();
        assert_eq!(total as usize, member.preheader.len() + member.body.len());
        let guards = member
            .preheader
            .iter()
            .chain(&member.body)
            .filter(|instruction| instruction.op.guard().is_some())
            .count();
        assert_eq!(census.guards.len(), guards);
        for guard in &census.guards {
            let [window, value, constant, unchanged, root, virt, slice, ..] = *guard;
            assert_eq!(window, value + constant + unchanged + root + virt + slice);
        }
        assert!(census.peak_gpr + census.peak_fpr > 0);
    }
    let root = member_census(&ir);
    assert!(
        root.ops.contains_key("GuardBranch") || root.ops.keys().any(|k| k.starts_with("Guard"))
    );
    assert_eq!(
        op_name(&ir.body[0].op)
            .chars()
            .next()
            .map(char::is_alphabetic),
        Some(true)
    );
}

#[test]
fn census_clif_accounting_excludes_nested_lowerings_and_reports_json() {
    use super::census::{begin, enter, exit, report};
    let ir = tree_reading_yes(100.0);
    let op = &ir.body[0].op;
    begin();
    enter();
    enter();
    exit(op, 3);
    exit(op, 10);
    let line = report(
        &[&ir, &ir.sides[0].ir],
        OptimizationTier::Baseline,
        42,
        false,
    );
    let name = super::census::op_name(op);
    // Two lowerings of one kind: the inner has 3, the outer 10 - 3 = 7.
    assert!(line.contains(&format!("\"{name}\":[2,10]")), "{line}");
    assert!(line.contains("\"clif_total\":42"), "{line}");
    assert!(line.contains("\"tier\":\"Baseline\""), "{line}");
    assert_eq!(line.matches('{').count(), line.matches('}').count());
    assert_eq!(line.matches('[').count(), line.matches(']').count());
}

/// Two loop parameters and one passed through unchanged, an invariant input,
/// and a guard whose snapshot names a parameter that is dead after its last
/// read.
#[allow(
    clippy::many_single_char_names,
    reason = "the letters are the loop parameters and inputs the doc comment names"
)]
fn census_loop(loops: bool) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), frame());
    let inv = b.input(0, TraceType::Float);
    let a_in = b.input(1, TraceType::Float);
    let b_in = b.input(2, TraceType::Float);
    let p_in = b.input(3, TraceType::Float);
    let a = b.loop_param(TraceType::Float, LoopParamInit::Input(a_in));
    let bb = b.loop_param(TraceType::Float, LoopParamInit::Input(b_in));
    let p = b.loop_param(TraceType::Float, LoopParamInit::Input(p_in));
    let s = arithmetic(&mut b, 0, ArithmeticOp::Add, a, bb);
    let c = compare(&mut b, 1, Comparison::LessThan, s, inv);
    guard(
        &mut b,
        2,
        c,
        true,
        SnapshotPhase::After {
            pc: 2,
            successor: 3,
        },
        vec![
            RegisterEntry::Value(a),
            RegisterEntry::Value(s),
            RegisterEntry::Value(bb),
            RegisterEntry::Value(p),
        ],
    );
    let n = arithmetic(&mut b, 3, ArithmeticOp::Add, s, bb);
    b.finish(if loops {
        TraceTerminator::LoopBack {
            args: vec![n, s, p],
        }
    } else {
        TraceTerminator::Return { value: None }
    })
}

#[test]
fn census_pins_pressure_and_values_live_across_a_guard() {
    use super::census::member_census;
    // Looping. Live at position 0: a, b, p, inv, s (5 floats); a dies at 0.
    // Across the guard at 2: b (read at 3), p and s (back-edge arguments),
    // inv (read each iteration). The snapshot names a, s, b, p, and a is
    // the only one no real use keeps.
    let looping = member_census(&census_loop(true));
    assert_eq!((looping.peak_gpr, looping.peak_fpr), (1, 5));
    assert_eq!(looping.guards.len(), 1);
    assert_eq!(looping.guards[0][7..], [4, 4, 1]);
    // Returning. Nothing is carried: inv ends at its read, p is never read
    // and n is unused, so only b and s cross the guard, and a and p are weak.
    let leaving = member_census(&census_loop(false));
    assert_eq!((leaving.peak_gpr, leaving.peak_fpr), (1, 4));
    assert_eq!(leaving.guards[0][7..], [4, 2, 2]);
}

/// A tree compiled again takes the optimized members that did not change from
/// its last compile, and optimizes afresh what changed: the level, the
/// flags, what the tree writes, or the member itself.
#[test]
fn a_tree_compiled_again_reuses_only_the_members_that_did_not_change() {
    use super::member_cache::OptimizedMembers;
    use crate::compat::TracingOptFlags;
    use crate::jit_ir::trace::{
        OptimizeLevel, OptimizeStats, TreeMember, TreeWrites, optimize_tree_member, plan_tree,
    };
    use std::sync::Arc;

    let flags = TracingOptFlags::LEVEL_1;
    let writes_of = |members: &[TreeMember]| {
        TreeWrites::of_members(members.iter().map(|member| &member.ir))
            .with_continued_guards(members.iter().filter_map(|member| member.parent))
    };
    let fresh = |members: &[TreeMember], level: OptimizeLevel| -> Vec<Arc<TraceIr>> {
        let writes = writes_of(members);
        members
            .iter()
            .enumerate()
            .map(|(index, member)| {
                Arc::new(
                    optimize_tree_member(&member.ir, level, flags, &writes.for_member(index)).0,
                )
            })
            .collect()
    };
    let level = OptimizeLevel::Optimized;
    let members = plan_tree(&tree(100.0)).unwrap();
    let writes = writes_of(&members);
    let mut cache = OptimizedMembers::new();
    let mut stats = OptimizeStats::default();
    // Runs the cache on `members` and returns its result and the hits and
    // misses it counted.
    let mut step = |cache: &mut OptimizedMembers,
                    members: &[TreeMember],
                    level: OptimizeLevel,
                    flags: TracingOptFlags| {
        let before = cache.counts();
        let result = cache.optimize_tree(
            TraceId(0),
            members,
            level,
            flags,
            &writes_of(members),
            &mut stats,
        );
        let after = cache.counts();
        (result, (after.0 - before.0, after.1 - before.1))
    };
    let (first, counts) = step(&mut cache, &members, level, flags);
    assert_eq!(first, fresh(&members, level));
    assert_eq!(counts, (0, 2));

    // The same tree takes both members (debug builds check each against a
    // fresh optimization).
    let (again, counts) = step(&mut cache, &members, level, flags);
    assert_eq!(counts, (2, 0), "the unchanged members are optimized again");
    assert_eq!(again, first);

    // Another level misses, and so do other flags.
    let (baseline, counts) = step(&mut cache, &members, OptimizeLevel::Baseline, flags);
    assert_eq!(counts, (0, 2));
    assert_eq!(baseline, fresh(&members, OptimizeLevel::Baseline));
    let (_, counts) = step(
        &mut cache,
        &members,
        OptimizeLevel::Baseline,
        TracingOptFlags::NONE,
    );
    assert_eq!(counts, (0, 2));

    // A changed side misses, the unchanged root hits.
    step(&mut cache, &members, level, flags);
    let mut changed = members.clone();
    changed[1].ir.id = TraceId(5);
    let (grown, counts) = step(&mut cache, &changed, level, flags);
    assert_eq!(counts, (1, 1));
    assert_eq!(grown, fresh(&changed, level));

    // A side that moved to another position in the plan hits.
    step(&mut cache, &members, level, flags);
    let mut copy = members[1].clone();
    copy.ir.id = TraceId(6);
    let shifted = vec![members[0].clone(), copy, members[1].clone()];
    let (moved, counts) = step(&mut cache, &shifted, level, flags);
    assert_eq!(counts, (2, 1), "the moved side is optimized again");
    assert_eq!(moved, fresh(&shifted, level));

    // A guard a side continues now, which the root's loop kept in its body
    // anyway: the root takes its earlier result, and that is what
    // optimizing it again makes.
    let alone = TreeWrites::of_members(members.iter().map(|member| &member.ir));
    let mut cache = OptimizedMembers::new();
    cache.optimize_tree(TraceId(0), &members, level, flags, &alone, &mut stats);
    let before = cache.counts();
    let kept = cache.optimize_tree(TraceId(0), &members, level, flags, &writes, &mut stats);
    assert_eq!(cache.counts().0, before.0 + 2, "both members are kept");
    assert_eq!(kept, fresh(&members, level));

    // A guard that hoisting moved out of the loop, which a side continues
    // now: the guard has to stay in the body, so the root is optimized
    // again.
    let members = plan_tree(&three_way::tree_of(three_way::root_with(30.0, true), 30.0)).unwrap();
    let alone = TreeWrites::of_members(members.iter().map(|member| &member.ir));
    let mut cache = OptimizedMembers::new();
    let first = cache.optimize_tree(TraceId(0), &members, level, flags, &alone, &mut stats);
    let hoisted = |ir: &TraceIr| {
        ir.preheader.iter().any(|instruction| {
            instruction
                .op
                .guard()
                .is_some_and(|guard| ir.guards[guard.index()].origin == 0)
        })
    };
    assert!(
        hoisted(&first[0]),
        "the invariant branch moves out of the loop"
    );
    let writes = writes_of(&members);
    let before = cache.counts();
    let again = cache.optimize_tree(TraceId(0), &members, level, flags, &writes, &mut stats);
    assert_eq!(
        cache.counts().1,
        before.1 + 1,
        "the root is optimized again"
    );
    assert!(!hoisted(&again[0]));
    assert_eq!(again, fresh(&members, level));
    // At the Baseline level nothing moves out of the loop, so the root is
    // kept.
    let level = OptimizeLevel::Baseline;
    cache.optimize_tree(TraceId(0), &members, level, flags, &alone, &mut stats);
    let before = cache.counts();
    let baseline = cache.optimize_tree(TraceId(0), &members, level, flags, &writes, &mut stats);
    assert_eq!(cache.counts().0, before.0 + members.len() as u64);
    assert_eq!(baseline, fresh(&members, level));
}

/// What [`compiled_shape`] returns.
type CompiledShape = (
    Vec<TraceIr>,
    Vec<ExitDescriptor>,
    Vec<(usize, Vec<TraceScalar>)>,
);

/// Compiles `ir` at `tier` and returns what the compile made of each member,
/// its exits, and the result of a run from a few start states.
fn compiled_shape(ir: &TraceIr, tier: OptimizationTier) -> CompiledShape {
    let trace = compile_trace(ir, tier).unwrap();
    let members = (0..trace.member_count())
        .map(|member| trace.lowered_member(member).clone())
        .collect();
    let runs = [1.0, 4.0, 50.5]
        .into_iter()
        .map(|start| {
            let registers = [f(10.0), f(20.0), f(start), TraceScalar::Nil];
            let (_, pc, state, _) = run(&trace, &registers, 100_000);
            (pc, state)
        })
        .collect();
    (members, trace.exits().to_vec(), runs)
}

/// A tree compiled again takes its members from the compile thread's cache,
/// and the code is what a cold compile makes. A member whose input changed
/// is optimized again, and the tree is again what a cold compile makes.
#[test]
fn a_tree_compiled_from_cached_members_equals_a_cold_compile() {
    use super::member_cache::{clear, counts};
    let ir = tree(100.0);
    for tier in TIERS {
        clear();
        let cold = compiled_shape(&ir, tier);
        assert_eq!(counts(), (0, 2), "{tier:?}");
        let warm = compiled_shape(&ir, tier);
        assert_eq!(
            counts(),
            (2, 2),
            "{tier:?}: both members come from the cache"
        );
        assert_eq!(warm, cold, "{tier:?}");
    }
    // The tree at another tier optimizes every member again.
    clear();
    compiled_shape(&ir, OptimizationTier::Baseline);
    let optimized = compiled_shape(&ir, OptimizationTier::Optimized);
    assert_eq!(counts(), (0, 4));
    clear();
    assert_eq!(optimized, compiled_shape(&ir, OptimizationTier::Optimized));

    // A side that changed is optimized again, the root is not.
    let mut changed = tree(100.0);
    *changed.sides[0].ir = then_side(50.0);
    for tier in TIERS {
        clear();
        compiled_shape(&ir, tier);
        let warm = compiled_shape(&changed, tier);
        assert_eq!(counts(), (1, 3), "{tier:?}");
        clear();
        assert_eq!(warm, compiled_shape(&changed, tier), "{tier:?}");
    }
}

/// A root that reads `yes` (r0) and never writes it: `no = no + yes` while
/// the counter runs to `limit`. Its exits name r0 as the entry input.
fn root_reading_yes(limit: f64) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), frame());
    for register in [1, 2] {
        b.mark_written(FrameId(0), register);
    }
    let yes = b.input(0, TraceType::Float);
    let no_in = b.input(1, TraceType::Float);
    let i_in = b.input(2, TraceType::Float);
    let no = b.loop_param(TraceType::Float, LoopParamInit::Input(no_in));
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let no2 = arithmetic(&mut b, 0, ArithmeticOp::Add, no, yes);
    let one = constant(&mut b, 1, 1.0);
    let i2 = arithmetic(&mut b, 1, ArithmeticOp::Add, i, one);
    let top = constant(&mut b, 1, limit);
    let more = compare(&mut b, 1, Comparison::LessEqual, i2, top);
    let registers = vec![
        RegisterEntry::Value(yes),
        RegisterEntry::Value(no2),
        RegisterEntry::Value(i2),
        RegisterEntry::Const(TraceScalar::Nil),
    ];
    let after = |successor| SnapshotPhase::After { pc: 1, successor };
    guard(&mut b, 1, more, true, after(2), registers.clone());
    let poll = b.snapshot(snapshot(after(0), 1, registers));
    b.set_poll_snapshot(poll);
    b.finish(TraceTerminator::LoopBack {
        args: vec![no2, i2],
    })
}

#[test]
fn an_exit_leaves_out_a_register_the_frame_keeps() {
    let ir = root_reading_yes(30.0);
    for tier in TIERS {
        let trace = compile_trace(&ir, tier).unwrap();
        let mut left_out = 0;
        for descriptor in trace.exits() {
            let lowered = trace.lowered_member(descriptor.member);
            let snapshot = &lowered.snapshots[descriptor.snapshot.index()];
            let RegisterEntry::Value(yes) = snapshot.frames[0].registers[0] else {
                continue;
            };
            assert!(
                lowered
                    .entry
                    .inputs
                    .iter()
                    .any(|input| input.register == 0 && input.value == yes),
                "{tier:?}: r0 is the entry input"
            );
            assert!(
                descriptor.slots.iter().all(|slot| slot.value != yes),
                "{tier:?}: exit {descriptor:?} stores r0, which the frame keeps"
            );
            left_out += 1;
        }
        assert!(left_out > 0, "{tier:?}: no exit names r0");
        for start in [1.0, 12.5, 29.0, 31.0] {
            let registers = [f(2.5), f(-4.0), f(start), TraceScalar::Nil];
            let (outcome, pc, state, _) = run(&trace, &registers, 100_000);
            assert!(matches!(outcome, TraceOutcome::Exit { .. }));
            let (expected_pc, expected, _) = reference(&ir, &registers);
            assert_eq!(pc, expected_pc, "start {start} at {tier:?}");
            assert_eq!(state, expected, "start {start} at {tier:?}");
        }
    }
}

/// A tree whose root never writes `yes` (r0), but whose first guard commits
/// `yes + 100` to r0 in its snapshot alone, as a guard behind an instruction
/// that writes a register only on the other path does. The side that starts
/// there steps the counter and exits naming r0 as its own entry input, which
/// is the value the guard committed, not the frame's.
fn guard_committed_yes(limit: f64) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), frame());
    for register in [1, 2, 3] {
        b.mark_written(FrameId(0), register);
    }
    let yes = b.input(0, TraceType::Float);
    let no_in = b.input(1, TraceType::Float);
    let i_in = b.input(2, TraceType::Float);
    let no = b.loop_param(TraceType::Float, LoopParamInit::Input(no_in));
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let hundred = constant(&mut b, 0, 100.0);
    let committed = arithmetic(&mut b, 0, ArithmeticOp::Add, yes, hundred);
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
            RegisterEntry::Value(committed),
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
    let state = vec![
        RegisterEntry::Value(yes),
        RegisterEntry::Value(no2),
        RegisterEntry::Value(i2),
        RegisterEntry::Value(c),
    ];
    let after = |successor| SnapshotPhase::After { pc: 6, successor };
    guard(&mut b, 6, more, true, after(7), state.clone());
    let poll = b.snapshot(snapshot(after(0), 6, state));
    b.set_poll_snapshot(poll);
    let mut ir = b.finish(TraceTerminator::LoopBack {
        args: vec![no2, i2],
    });

    let mut b = TraceBuilder::new(TraceId(1), frame());
    b.mark_written(FrameId(0), 2);
    let side_yes = b.input(0, TraceType::Float);
    let side_no = b.input(1, TraceType::Float);
    let side_i = b.input(2, TraceType::Float);
    let one = constant(&mut b, 3, 1.0);
    let side_i2 = arithmetic(&mut b, 3, ArithmeticOp::Add, side_i, one);
    let exit = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 3,
            successor: 4,
        },
        3,
        vec![
            RegisterEntry::Value(side_yes),
            RegisterEntry::Value(side_no),
            RegisterEntry::Value(side_i2),
            RegisterEntry::Const(TraceScalar::Nil),
        ],
    ));
    let side = b.finish(TraceTerminator::Exit { snapshot: exit });
    ir.header_live = vec![0, 1, 2];
    ir.sides.push(TraceSide {
        parent: None,
        guard: 0,
        ir: Box::new(side),
    });
    ir
}

#[test]
fn a_side_exit_restores_a_register_its_parents_guard_committed() {
    let ir = guard_committed_yes(40.0);
    for tier in TIERS {
        let trace = compile_trace(&ir, tier).unwrap();
        assert_eq!(trace.member_count(), 2, "{tier:?}");
        for start in [1.0, 2.0, 3.0, 4.0, 7.0, 12.5] {
            let registers = [f(2.5), f(-4.0), f(start), TraceScalar::Nil];
            let (outcome, pc, state, _) = run(&trace, &registers, 100_000);
            assert!(matches!(outcome, TraceOutcome::Exit { .. }));
            let (expected_pc, expected, _) = reference(&ir, &registers);
            assert_eq!(pc, expected_pc, "start {start} at {tier:?}");
            assert_eq!(state, expected, "start {start} at {tier:?}");
        }
    }
}
