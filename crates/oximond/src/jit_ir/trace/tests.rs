#![allow(clippy::similar_names, clippy::too_many_lines)]

use super::ir::*;
use super::reference::{EffectRecord, ForcedExit, Outcome, Restored, evaluate, evaluate_forced};
use super::snapshot::*;
use super::verify::TraceVerifyError;
use crate::jit_ir::Comparison;

const LOAD: u32 = 1;
const STORE: u32 = 2;

fn site(pc: usize) -> TraceSite {
    TraceSite {
        frame: FrameId(0),
        pc,
    }
}

fn root(registers: u16) -> TraceFrameInfo {
    TraceFrameInfo {
        register_count: registers,
        code_len: 16,
        inlined: None,
        activation: None,
    }
}

/// An inlined call of pinned closure `closure` from the root frame's
/// instruction `call_pc`, with `registers` registers.
fn inlined(registers: u16, closure: u32, call_pc: usize) -> TraceFrameInfo {
    TraceFrameInfo {
        inlined: Some(InlinedFrame {
            closure,
            parent: FrameId(0),
            call_pc,
            result_base: 0,
            results: 1,
            reexec: false,
            tail_calls: 0,
            tail_site: None,
            protected: Protected::No,
            entry: Entry::Call,
            dynamic: false,
        }),
        activation: None,
        ..root(registers)
    }
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

fn f(value: f64) -> TraceScalar {
    TraceScalar::Float(value)
}

fn emit(builder: &mut TraceBuilder, pc: usize, op: TraceOp, ty: TraceType) -> ValueId {
    builder.emit(site(pc), op, Some(ty)).unwrap()
}

/// The worked recovery case from the plan:
///
/// ```lua
/// while ... do
///   local x = t[i]      -- pc 1
///   t[i] = x + 1        -- pc 2 (store committed)
///   if x > 0 then ...   -- pc 3 compare, pc 4 branch, else at pc 9
///     sum = sum + x
///   i = i + 1
/// end
/// ```
///
/// Registers: r0 i, r1 sum, r2 x, r3 branch condition. The trace writes all
/// four every iteration, so x and cond are loop parameters too. The trace is
/// entered with r2 a float and r3 a boolean, so a guard failing in a later
/// iteration restores the previous iteration's values exactly.
fn worked_case() -> TraceIr {
    use RegisterEntry::Value;
    let mut b = TraceBuilder::new(TraceId(1), root(4));
    for register in 0..4 {
        b.mark_written(FrameId(0), register);
    }
    let i_in = b.input(0, TraceType::Float);
    let sum_in = b.input(1, TraceType::Float);
    let x_in = b.input(2, TraceType::Float);
    let cond_in = b.input(3, TraceType::Boolean);
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let sum = b.loop_param(TraceType::Float, LoopParamInit::Input(sum_in));
    let x_prev = b.loop_param(TraceType::Float, LoopParamInit::Input(x_in));
    let cond_prev = b.loop_param(TraceType::Boolean, LoopParamInit::Input(cond_in));

    // pc 1: load, then guard the slot is a number. Exits before any effect
    // of this iteration, so the interpreter evaluates the original load.
    let raw = emit(
        &mut b,
        1,
        TraceOp::Read {
            label: LOAD,
            operands: vec![i],
            result: TraceType::Any,
        },
        TraceType::Any,
    );
    let s0 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 1 },
        b.effect_position(),
        Some(0),
        vec![Value(i), Value(sum), Value(x_prev), Value(cond_prev)],
    ));
    let g0 = b.guard(site(1), s0);
    let x = emit(
        &mut b,
        1,
        TraceOp::GuardType {
            source: raw,
            expected: TraceType::Float,
            guard: g0,
        },
        TraceType::Float,
    );

    // pc 2: t[i] = x + 1 commits one effect.
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
    let effect = b.next_effect();
    b.emit(
        site(2),
        TraceOp::Effect {
            effect,
            label: STORE,
            operands: vec![i, y],
            result: None,
        },
        None,
    );

    // pc 3/4: x > 0. The branch guard fails after the store. Its snapshot
    // is the branch post-state with the else successor, and one committed
    // effect.
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
    let s1 = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 4,
            successor: 9,
        },
        b.effect_position(),
        None,
        vec![Value(i), Value(sum), Value(x), Value(cond)],
    ));
    let g1 = b.guard(site(4), s1);
    b.emit(
        site(4),
        TraceOp::GuardBranch {
            condition: cond,
            expected: true,
            guard: g1,
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

/// Entry registers for the worked case.
fn worked_entry(i: f64, sum: f64, x: f64, cond: bool) -> [TraceScalar; 4] {
    [f(i), f(sum), f(x), TraceScalar::Boolean(cond)]
}

/// A fake heap table for the symbolic load and store.
struct Heap(Vec<TraceScalar>);

impl Heap {
    fn run(
        &mut self,
        ir: &TraceIr,
        registers: &[TraceScalar],
        iterations: u32,
    ) -> super::reference::Evaluation {
        // The evaluator takes one closure for reads and effect results. Stores
        // return no value, and the heap applies them after the run.
        let table = self.0.clone();
        let evaluation = evaluate(ir, registers, &[], iterations, |record, _| {
            assert_eq!(record.label, LOAD);
            let TraceScalar::Float(index) = record.operands[0] else {
                panic!("index is a float")
            };
            // The test heap is 0-based.
            Some(table[index as usize])
        });
        for record in &evaluation.effects {
            assert_eq!(record.label, STORE);
            let [TraceScalar::Float(index), value] = record.operands[..] else {
                panic!("store operands")
            };
            self.0[index as usize] = value;
        }
        evaluation
    }
}

#[test]
fn trace_ir_is_owned_and_send() {
    fn assert_send<T: Send + 'static>() {}
    assert_send::<TraceIr>();
    assert_send::<Snapshot>();
}

#[test]
fn worked_case_verifies() {
    worked_case().verify().unwrap();
}

#[test]
fn guard_after_a_committed_store_resumes_after_it_and_never_replays_it() {
    let ir = worked_case();
    let mut heap = Heap(vec![f(3.0), f(5.0), f(-2.0), f(4.0)]);
    let evaluation = heap.run(&ir, &worked_entry(0.0, 0.0, 0.0, false), 10);
    // Iterations 0 and 1 pass, iteration 2 reads -2 and fails the branch
    // guard after its store.
    let Outcome::GuardFailed { guard, restored } = &evaluation.outcome else {
        panic!("expected a guard failure: {:?}", evaluation.outcome)
    };
    assert_eq!(*guard, GuardId(1));
    assert_eq!(
        restored,
        &Restored {
            pc: 9,
            last_interpreted_pc: Some(4),
            registers: vec![f(2.0), f(8.0), f(-2.0), TraceScalar::Boolean(false)],
            effect_position: 1,
            outer: Vec::new(),
            tables: Vec::new(),
        }
    );
    // Three stores, each exactly once, including the one before the failure.
    let stores: Vec<_> = evaluation
        .effects
        .iter()
        .map(|record: &EffectRecord| (record.effect, record.operands.clone()))
        .collect();
    assert_eq!(
        stores,
        vec![
            (0, vec![f(0.0), f(4.0)]),
            (0, vec![f(1.0), f(6.0)]),
            (0, vec![f(2.0), f(-1.0)]),
        ]
    );
    assert_eq!(heap.0, vec![f(4.0), f(6.0), f(-1.0), f(4.0)]);
    // Restarting at the loop header would read the new -1 and store again.
    // The restored state resumes at the else branch, so the slot holds -1.
    assert_eq!(heap.0[2], f(-1.0));
}

#[test]
fn slot_type_guard_exits_before_the_read_and_the_store() {
    let ir = worked_case();
    let mut heap = Heap(vec![f(3.0), TraceScalar::Nil, f(1.0)]);
    let evaluation = heap.run(&ir, &worked_entry(0.0, 0.0, 0.0, false), 10);
    let Outcome::GuardFailed { guard, restored } = &evaluation.outcome else {
        panic!("expected a guard failure")
    };
    assert_eq!(*guard, GuardId(0));
    // Before phase: the interpreter executes pc 1 itself.
    assert_eq!(restored.pc, 1);
    assert_eq!(restored.last_interpreted_pc, Some(0));
    assert_eq!(restored.effect_position, 0);
    // Only iteration 0's store committed. Iteration 1 stored nothing.
    assert_eq!(evaluation.effects.len(), 1);
    assert_eq!(heap.0[1], TraceScalar::Nil);
    // r2 and r3 hold iteration 0's x and cond, which the interpreter would
    // hold at this boundary. The entry values 0.0 and false are stale.
    assert_eq!(
        restored.registers,
        vec![f(1.0), f(3.0), f(3.0), TraceScalar::Boolean(true)]
    );
}

#[test]
fn verifier_rejects_unchanged_for_a_written_register() {
    expect_error(
        |ir| {
            ir.snapshots[0].frames[0]
                .registers
                .set(2, RegisterEntry::Unchanged)
        },
        |error| {
            matches!(
                error,
                TraceVerifyError::UnchangedWritten {
                    snapshot: 0,
                    register: 2,
                    ..
                }
            )
        },
    );
    // The same map is legal once the trace does not write r2.
    let mut ir = worked_case();
    ir.snapshots[0].frames[0]
        .registers
        .set(2, RegisterEntry::Unchanged);
    ir.written_registers.retain(|&(_, register)| register != 2);
    ir.verify().unwrap();
}

#[test]
fn entry_mismatch_runs_nothing() {
    let ir = worked_case();
    let mut heap = Heap(vec![f(3.0)]);
    let evaluation = heap.run(
        &ir,
        &[
            TraceScalar::Integer(0),
            f(0.0),
            f(0.0),
            TraceScalar::Boolean(false),
        ],
        10,
    );
    assert_eq!(evaluation.outcome, Outcome::EntryMismatch);
    assert_eq!(
        evaluation.effects,
        [] as [crate::jit_ir::trace::reference::EffectRecord; 0]
    );
}

#[test]
fn branch_guard_effect_position_must_count_the_store() {
    let mut ir = worked_case();
    ir.snapshots[1].effect_position = 0;
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::EffectOrder {
            instruction: Some(_),
            ..
        })
    ));
}

/// Loop-carried swap: `a, b = b, a` on every iteration.
// The single letters name the swap's operands, as the listing above does.
#[allow(clippy::many_single_char_names)]
fn swap_loop() -> TraceIr {
    use RegisterEntry::{Const, Value};
    let mut b = TraceBuilder::new(TraceId(2), root(4));
    let a_in = b.input(0, TraceType::Float);
    let b_in = b.input(1, TraceType::Float);
    let a = b.loop_param(TraceType::Float, LoopParamInit::Input(a_in));
    let bb = b.loop_param(TraceType::Float, LoopParamInit::Input(b_in));
    let n = b.loop_param(
        TraceType::Integer,
        LoopParamInit::Const(TraceScalar::Integer(0)),
    );
    let one = emit(
        &mut b,
        1,
        TraceOp::Constant(TraceScalar::Integer(1)),
        TraceType::Integer,
    );
    let limit = emit(
        &mut b,
        1,
        TraceOp::Constant(TraceScalar::Integer(2)),
        TraceType::Integer,
    );
    let n_next = emit(
        &mut b,
        1,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: n,
            right: one,
        },
        TraceType::Integer,
    );
    // The swap goes through temporary r2. The snapshot is after the third
    // move: r0 holds the old b, r1 the old a, r2 the old a.
    let cond = emit(
        &mut b,
        4,
        TraceOp::Compare {
            comparison: Comparison::LessEqual,
            left: n_next,
            right: limit,
        },
        TraceType::Boolean,
    );
    let s = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 5,
            successor: 8,
        },
        0,
        None,
        vec![
            Value(bb),
            Value(a),
            Value(a),
            Const(TraceScalar::Integer(0)),
        ],
    ));
    let g = b.guard(site(5), s);
    b.emit(
        site(5),
        TraceOp::GuardBranch {
            condition: cond,
            expected: true,
            guard: g,
        },
        None,
    );
    // The next iteration sees a = old b and b = old a. Both arguments read
    // the current parameters before either is assigned.
    b.finish(TraceTerminator::LoopBack {
        args: vec![bb, a, n_next],
    })
}

#[test]
fn loop_carried_swap_restores_the_swapped_registers() {
    let ir = swap_loop();
    ir.verify().unwrap();
    let evaluation = evaluate(
        &ir,
        &[f(1.0), f(2.0), TraceScalar::Nil, TraceScalar::Nil],
        &[],
        10,
        |_, _| None,
    );
    let Outcome::GuardFailed { restored, .. } = evaluation.outcome else {
        panic!("expected a guard failure")
    };
    // Two completed iterations swap twice, so the third starts with (1, 2).
    // Its own swap leaves r0 = 2, r1 = 1.
    assert_eq!(evaluation.iterations, 2);
    assert_eq!(
        restored.registers,
        vec![f(2.0), f(1.0), f(1.0), TraceScalar::Integer(0)]
    );
    assert_eq!(restored.pc, 8);
}

#[test]
fn loop_swap_with_an_odd_failing_iteration_shows_the_other_order() {
    let mut ir = swap_loop();
    // Shrink the limit so the second iteration's guard fails.
    for instruction in &mut ir.body {
        if let TraceOp::Constant(TraceScalar::Integer(2)) = instruction.op {
            instruction.op = TraceOp::Constant(TraceScalar::Integer(1));
        }
    }
    let evaluation = evaluate(
        &ir,
        &[f(1.0), f(2.0), TraceScalar::Nil, TraceScalar::Nil],
        &[],
        10,
        |_, _| None,
    );
    let Outcome::GuardFailed { restored, .. } = evaluation.outcome else {
        panic!("expected a guard failure")
    };
    assert_eq!(evaluation.iterations, 1);
    // One swap done, the second pass swaps back: a = 2, b = 1 before it.
    assert_eq!(restored.registers[..2], [f(1.0), f(2.0)]);
}

#[test]
fn register_live_only_on_the_cold_branch_is_restored() {
    use RegisterEntry::{Unchanged, Value};
    // r0 x, r1 cold-only product. The hot path never reads r1 again.
    let mut b = TraceBuilder::new(TraceId(3), root(3));
    let x_in = b.input(0, TraceType::Float);
    let x = b.loop_param(TraceType::Float, LoopParamInit::Input(x_in));
    let two = emit(&mut b, 1, TraceOp::Constant(f(2.0)), TraceType::Float);
    let cold = emit(
        &mut b,
        1,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Multiply,
            left: x,
            right: two,
        },
        TraceType::Float,
    );
    let limit = emit(&mut b, 2, TraceOp::Constant(f(10.0)), TraceType::Float);
    let cond = emit(
        &mut b,
        2,
        TraceOp::Compare {
            comparison: Comparison::LessThan,
            left: x,
            right: limit,
        },
        TraceType::Boolean,
    );
    let s = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 3,
            successor: 12,
        },
        0,
        None,
        vec![Value(x), Value(cold), Unchanged],
    ));
    let g = b.guard(site(3), s);
    b.emit(
        site(3),
        TraceOp::GuardBranch {
            condition: cond,
            expected: true,
            guard: g,
        },
        None,
    );
    let x_next = emit(
        &mut b,
        4,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: x,
            right: two,
        },
        TraceType::Float,
    );
    let ir = b.finish(TraceTerminator::LoopBack { args: vec![x_next] });
    ir.verify().unwrap();
    let evaluation = evaluate(&ir, &[f(7.0), f(-1.0), f(-5.0)], &[], 10, |_, _| None);
    let Outcome::GuardFailed { restored, .. } = evaluation.outcome else {
        panic!("expected a guard failure")
    };
    // x runs 7, 9, 11. At 11 the guard fails with the product 22 live in r1
    // and the untouched r2 intact.
    assert_eq!(restored.registers, vec![f(11.0), f(22.0), f(-5.0)]);
}

#[test]
fn before_and_after_phases_set_pc_and_last_interpreted_pc() {
    let before = snap(SnapshotPhase::Before { pc: 6 }, 0, Some(5), vec![]);
    assert_eq!(
        before.resume_point(),
        ResumePoint {
            pc: 6,
            last_interpreted_pc: Some(5)
        }
    );
    let first = snap(SnapshotPhase::Before { pc: 0 }, 0, None, vec![]);
    assert_eq!(first.resume_point().last_interpreted_pc, None);
    let after = snap(
        SnapshotPhase::After {
            pc: 6,
            successor: 7,
        },
        0,
        Some(1),
        vec![],
    );
    assert_eq!(
        after.resume_point(),
        ResumePoint {
            pc: 7,
            last_interpreted_pc: Some(6)
        }
    );
}

#[test]
fn lua_float_semantics_match_the_vm() {
    use RegisterEntry::Value;
    let mut b = TraceBuilder::new(TraceId(4), root(7));
    let mut results = Vec::new();
    let constant = |b: &mut TraceBuilder, value: f64| {
        emit(b, 0, TraceOp::Constant(f(value)), TraceType::Float)
    };
    let a = constant(&mut b, 5.5);
    let c = constant(&mut b, -2.0);
    let zero = constant(&mut b, 0.0);
    let base = constant(&mut b, 2.0);
    let ten = constant(&mut b, 10.0);
    for (op, left, right) in [
        (ArithmeticOp::Modulo, a, c),
        (ArithmeticOp::Power, base, ten),
        (ArithmeticOp::Divide, a, zero),
        (ArithmeticOp::Divide, zero, zero),
        (ArithmeticOp::Modulo, c, ten),
    ] {
        results.push(emit(
            &mut b,
            1,
            TraceOp::Arithmetic { op, left, right },
            TraceType::Float,
        ));
    }
    let negative_zero = emit(
        &mut b,
        2,
        TraceOp::Negate { source: zero },
        TraceType::Float,
    );
    results.push(negative_zero);
    let max = emit(
        &mut b,
        2,
        TraceOp::Constant(TraceScalar::Integer(i64::MAX)),
        TraceType::Integer,
    );
    let one = emit(
        &mut b,
        2,
        TraceOp::Constant(TraceScalar::Integer(1)),
        TraceType::Integer,
    );
    let wrapped = emit(
        &mut b,
        2,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: max,
            right: one,
        },
        TraceType::Integer,
    );
    results.push(wrapped);
    let exit = b.snapshot(snap(
        SnapshotPhase::Before { pc: 3 },
        0,
        None,
        results.iter().map(|value| Value(*value)).collect(),
    ));
    let ir = b.finish(TraceTerminator::Exit { snapshot: exit });
    ir.verify().unwrap();
    let evaluation = evaluate(&ir, &[TraceScalar::Nil; 7], &[], 1, |_, _| None);
    let Outcome::Exited(restored) = evaluation.outcome else {
        panic!("expected an exit")
    };
    assert_eq!(restored.registers[0], f(-0.5));
    assert_eq!(restored.registers[1], f(1024.0));
    assert_eq!(restored.registers[2], f(f64::INFINITY));
    assert!(matches!(restored.registers[3], TraceScalar::Float(v) if v.is_nan()));
    assert_eq!(restored.registers[4], f(8.0));
    assert_eq!(restored.registers[5], f(-0.0));
    assert_ne!(restored.registers[5], f(0.0), "signed zero survives");
    assert_eq!(restored.registers[6], TraceScalar::Integer(i64::MIN));
}

#[test]
fn const_and_root_entries_restore() {
    use RegisterEntry::{Const, Root};
    let mut b = TraceBuilder::new(TraceId(5), root(3));
    b.set_root_slot_count(1);
    let exit = b.snapshot(snap(
        SnapshotPhase::Before { pc: 0 },
        0,
        None,
        vec![
            Const(TraceScalar::Integer(7)),
            Root(0),
            Const(TraceScalar::Boolean(true)),
        ],
    ));
    let ir = b.finish(TraceTerminator::Exit { snapshot: exit });
    ir.verify().unwrap();
    let evaluation = evaluate(&ir, &[TraceScalar::Nil; 3], &[f(9.0)], 1, |_, _| None);
    let Outcome::Exited(restored) = evaluation.outcome else {
        panic!("expected an exit")
    };
    assert_eq!(
        restored.registers,
        vec![TraceScalar::Integer(7), f(9.0), TraceScalar::Boolean(true)]
    );
}

fn expect_error(mutate: impl FnOnce(&mut TraceIr), check: impl FnOnce(&TraceVerifyError) -> bool) {
    let mut ir = worked_case();
    mutate(&mut ir);
    let error = ir
        .verify()
        .expect_err("mutated trace must fail verification");
    assert!(check(&error), "unexpected error: {error}");
}

#[test]
fn verifier_rejects_use_before_definition() {
    expect_error(
        |ir| ir.body.swap(1, 2),
        |error| {
            matches!(
                error,
                TraceVerifyError::UseBeforeDefinition { .. }
                    | TraceVerifyError::ValueTableMismatch(_)
            )
        },
    );
}

#[test]
fn verifier_rejects_wrong_operand_types() {
    expect_error(
        |ir| {
            // Add a float to an integer constant.
            let index = ir
                .body
                .iter()
                .position(|i| matches!(i.op, TraceOp::Constant(_)))
                .unwrap();
            ir.body[index].op = TraceOp::Constant(TraceScalar::Integer(1));
            ir.values[ir.body[index].result.unwrap().index()].ty = TraceType::Integer;
        },
        |error| matches!(error, TraceVerifyError::TypeMismatch { .. }),
    );
}

#[test]
fn verifier_rejects_incomplete_guard_snapshots() {
    expect_error(
        |ir| {
            let registers = &mut ir.snapshots[0].frames[0].registers;
            *registers = registers.resized(registers.len() - 1, RegisterEntry::Unchanged, None);
        },
        |error| matches!(error, TraceVerifyError::InvalidSnapshot { .. }),
    );
}

#[test]
fn verifier_rejects_snapshot_values_defined_after_the_guard() {
    expect_error(
        |ir| {
            // The branch guard's result value x2 is defined after guard 0.
            let later = ir.snapshots[1].frames[0].registers[3];
            ir.snapshots[0].frames[0].registers.set(3, later);
        },
        |error| matches!(error, TraceVerifyError::SnapshotValueUndefined { .. }),
    );
}

#[test]
fn verifier_rejects_boundaries_outside_the_chunk() {
    expect_error(
        |ir| {
            ir.snapshots[1].phase = SnapshotPhase::After {
                pc: 4,
                successor: 99,
            }
        },
        |error| matches!(error, TraceVerifyError::InvalidSnapshot { .. }),
    );
    expect_error(
        |ir| ir.body[0].site.pc = 99,
        |error| matches!(error, TraceVerifyError::InvalidSite { .. }),
    );
}

#[test]
fn verifier_rejects_bad_loop_arguments() {
    expect_error(
        |ir| {
            if let TraceTerminator::LoopBack { args } = &mut ir.terminator {
                args.pop();
            }
        },
        |error| {
            matches!(
                error,
                TraceVerifyError::LoopArity {
                    expected: 4,
                    actual: 3
                }
            )
        },
    );
    expect_error(
        |ir| {
            if let TraceTerminator::LoopBack { args } = &mut ir.terminator {
                // A boolean where a float is carried.
                args[0] = ir
                    .body
                    .iter()
                    .find_map(|i| match i.op {
                        TraceOp::Compare { .. } => i.result,
                        _ => None,
                    })
                    .unwrap();
            }
        },
        |error| matches!(error, TraceVerifyError::LoopArgumentType { index: 0 }),
    );
}

#[test]
fn verifier_rejects_effect_ordering_errors() {
    expect_error(
        |ir| {
            for instruction in &mut ir.body {
                if let TraceOp::Effect { effect, .. } = &mut instruction.op {
                    *effect = EffectId(3);
                }
            }
        },
        |error| matches!(error, TraceVerifyError::EffectOrder { .. }),
    );
}

/// A store at pc 2 whose result is guarded at the same pc.
fn replay_trace(phase: SnapshotPhase) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(4), root(2));
    let i_in = b.input(0, TraceType::Float);
    let effect = b.next_effect();
    let result = b
        .emit(
            site(2),
            TraceOp::Effect {
                effect,
                label: STORE,
                operands: vec![i_in],
                result: Some(TraceType::Any),
            },
            Some(TraceType::Any),
        )
        .unwrap();
    let s = b.snapshot(snap(
        phase,
        b.effect_position(),
        Some(1),
        vec![RegisterEntry::Value(i_in), RegisterEntry::Const(f(0.0))],
    ));
    let g = b.guard(site(2), s);
    b.emit(
        site(2),
        TraceOp::GuardType {
            source: result,
            expected: TraceType::Float,
            guard: g,
        },
        Some(TraceType::Float),
    );
    b.finish(TraceTerminator::Exit {
        snapshot: SnapshotId(0),
    })
}

#[test]
fn verifier_rejects_before_snapshot_that_replays_a_committed_effect() {
    let ir = replay_trace(SnapshotPhase::Before { pc: 2 });
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::EffectReplay {
            snapshot: 0,
            instruction: Some(1)
        })
    ));
    // After the committed instruction the same guard is legal, and the
    // evaluator resumes after the store without repeating it.
    let ir = replay_trace(SnapshotPhase::After {
        pc: 2,
        successor: 3,
    });
    ir.verify().unwrap();
    let evaluation = evaluate(&ir, &[f(1.0), f(0.0)], &[], 1, |_, _| {
        Some(TraceScalar::Nil)
    });
    let Outcome::GuardFailed { restored, .. } = evaluation.outcome else {
        panic!("expected a guard failure")
    };
    assert_eq!(restored.pc, 3);
    assert_eq!(restored.effect_position, 1);
    assert_eq!(evaluation.effects.len(), 1);
}

#[test]
fn verifier_rejects_the_exit_snapshot_that_replays_a_committed_effect() {
    let mut ir = replay_trace(SnapshotPhase::After {
        pc: 2,
        successor: 3,
    });
    ir.snapshots[0].phase = SnapshotPhase::Before { pc: 2 };
    ir.body.pop();
    ir.guards.pop();
    ir.values.pop();
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::EffectReplay {
            instruction: None,
            ..
        })
    ));
}

#[test]
fn verifier_rejects_guard_snapshots_for_another_location() {
    expect_error(
        |ir| ir.snapshots[0].phase = SnapshotPhase::Before { pc: 11 },
        |error| matches!(error, TraceVerifyError::GuardLocation { instruction: 1 }),
    );
    expect_error(
        |ir| {
            ir.snapshots[1].phase = SnapshotPhase::After {
                pc: 6,
                successor: 9,
            }
        },
        |error| matches!(error, TraceVerifyError::GuardLocation { .. }),
    );
}

#[test]
fn verifier_rejects_unbounded_frame_depth() {
    expect_error(
        |ir| {
            let frame = ir.snapshots[0].frames[0].clone();
            ir.snapshots[0].frames = vec![frame; MAX_TRACE_FRAME_DEPTH + 1];
        },
        |error| matches!(error, TraceVerifyError::FrameDepth { .. }),
    );
}

#[test]
fn verifier_rejects_any_values_and_bad_roots_in_snapshots() {
    expect_error(
        |ir| {
            let raw = ir
                .body
                .iter()
                .find_map(|i| match i.op {
                    TraceOp::Read { .. } => i.result,
                    _ => None,
                })
                .unwrap();
            ir.snapshots[0].frames[0]
                .registers
                .set(2, RegisterEntry::Value(raw));
        },
        |error| matches!(error, TraceVerifyError::InvalidSnapshot { .. }),
    );
    expect_error(
        |ir| {
            ir.snapshots[0].frames[0]
                .registers
                .set(2, RegisterEntry::Root(0))
        },
        |error| matches!(error, TraceVerifyError::InvalidSnapshot { .. }),
    );
}

#[test]
fn verifier_rejects_duplicate_entry_registers() {
    expect_error(
        |ir| ir.entry.inputs[1].register = 0,
        |error| matches!(error, TraceVerifyError::InvalidEntry { .. }),
    );
}

/// A guard at pc 4 followed by the store of the same guest instruction.
fn guard_before_own_store(phase: SnapshotPhase) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(5), root(2));
    let k = b.input(0, TraceType::Float);
    let zero = emit(&mut b, 4, TraceOp::Constant(f(0.0)), TraceType::Float);
    let cond = emit(
        &mut b,
        4,
        TraceOp::Compare {
            comparison: Comparison::LessThan,
            left: zero,
            right: k,
        },
        TraceType::Boolean,
    );
    let s = b.snapshot(snap(
        phase,
        b.effect_position(),
        Some(3),
        vec![RegisterEntry::Value(k), RegisterEntry::Const(f(0.0))],
    ));
    let g = b.guard(site(4), s);
    b.emit(
        site(4),
        TraceOp::GuardBranch {
            condition: cond,
            expected: true,
            guard: g,
        },
        None,
    );
    let effect = b.next_effect();
    b.emit(
        site(4),
        TraceOp::Effect {
            effect,
            label: STORE,
            operands: vec![k],
            result: None,
        },
        None,
    );
    // The exit is after the store, so it is legal in both variants.
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 4,
            successor: 5,
        },
        b.effect_position(),
        Some(3),
        vec![RegisterEntry::Value(k), RegisterEntry::Const(f(0.0))],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

#[test]
fn verifier_rejects_after_snapshot_that_skips_its_own_pending_effect() {
    // The store at pc 4 has not run when the guard fails. Resuming after pc 4
    // would lose it.
    let ir = guard_before_own_store(SnapshotPhase::After {
        pc: 4,
        successor: 5,
    });
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::EffectSkipped {
            snapshot: 0,
            instruction: Some(2)
        })
    ));
    // Resuming before pc 4 executes the store exactly once.
    guard_before_own_store(SnapshotPhase::Before { pc: 4 })
        .verify()
        .unwrap();
}

#[test]
fn verifier_rejects_unreferenced_snapshots_with_bad_values() {
    let mut ir = worked_case();
    let raw = ir
        .body
        .iter()
        .find_map(|i| match i.op {
            TraceOp::Read { .. } => i.result,
            _ => None,
        })
        .unwrap();
    let registers = ir.snapshots[0].frames[0].registers.clone();
    let mut extra = ir.snapshots[0].clone();
    extra.id = SnapshotId(u32::try_from(ir.snapshots.len()).unwrap());
    extra.frames[0].registers = registers;
    extra.frames[0].registers.set(2, RegisterEntry::Value(raw));
    ir.snapshots.push(extra);
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::InvalidSnapshot { .. })
    ));
}

#[test]
fn verifier_rejects_unchanged_in_inlined_frames() {
    expect_error(
        |ir| {
            ir.pinned_closures = vec![9];
            let call_pc = ir.snapshots[0].frames[0].last_interpreted_pc.unwrap();
            ir.frames.push(inlined(2, 0, call_pc));
            let registers = vec![RegisterEntry::Unchanged; 2];
            let root_frame = ir.snapshots[0].frames[0].clone();
            ir.snapshots[0].frames = vec![
                root_frame,
                SnapshotFrame {
                    frame: FrameId(1),
                    last_interpreted_pc: None,
                    registers: registers.into(),
                },
            ];
        },
        |error| matches!(error, TraceVerifyError::InvalidSnapshot { detail, .. } if detail.contains("root frame")),
    );
}

/// Frame-0 read at pc 2, an inlined callee effect at frame 1, then a frame-0
/// guard at pc 2. `guard_first` puts the guard before the callee instead.
fn inlined_effect_trace(phase: SnapshotPhase, guard_first: bool) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(5), root(2));
    b.set_pinned_closures(vec![9]);
    let callee = b.add_frame(inlined(1, 0, 2));
    let i_in = b.input(0, TraceType::Float);
    let call = emit(
        &mut b,
        2,
        TraceOp::Read {
            label: LOAD,
            operands: vec![i_in],
            result: TraceType::Any,
        },
        TraceType::Any,
    );
    let callee_effect = |b: &mut TraceBuilder| {
        let effect = b.next_effect();
        b.emit(
            TraceSite {
                frame: callee,
                pc: 0,
            },
            TraceOp::Effect {
                effect,
                label: STORE,
                operands: vec![i_in],
                result: None,
            },
            None,
        );
    };
    if !guard_first {
        callee_effect(&mut b);
    }
    let effects = b.effect_position();
    let s = b.snapshot(snap(
        phase,
        effects,
        Some(1),
        vec![RegisterEntry::Value(i_in), RegisterEntry::Const(f(0.0))],
    ));
    let g = b.guard(site(2), s);
    b.emit(
        site(2),
        TraceOp::GuardType {
            source: call,
            expected: TraceType::Float,
            guard: g,
        },
        Some(TraceType::Float),
    );
    if guard_first {
        callee_effect(&mut b);
    }
    b.finish(TraceTerminator::Exit {
        snapshot: SnapshotId(0),
    })
}

#[test]
fn verifier_sees_inlined_callee_effects_inside_a_guest_step() {
    // Before{2} after the callee stored: the interpreter would re-run the call.
    let ir = inlined_effect_trace(SnapshotPhase::Before { pc: 2 }, false);
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::EffectReplay { .. })
    ));
    // After{2} before the callee stored would skip the pending store.
    let ir = inlined_effect_trace(
        SnapshotPhase::After {
            pc: 2,
            successor: 3,
        },
        true,
    );
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::EffectSkipped { .. })
    ));
}

#[test]
fn verifier_accepts_consistent_inlined_snapshots() {
    inlined_effect_trace(
        SnapshotPhase::After {
            pc: 2,
            successor: 3,
        },
        false,
    )
    .verify()
    .unwrap();
}

#[test]
fn verifier_rejects_unreferenced_snapshots_that_replay_an_effect() {
    let mut ir = replay_trace(SnapshotPhase::After {
        pc: 2,
        successor: 3,
    });
    let mut extra = ir.snapshots[0].clone();
    extra.id = SnapshotId(u32::try_from(ir.snapshots.len()).unwrap());
    extra.phase = SnapshotPhase::Before { pc: 2 };
    ir.snapshots.push(extra);
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::InvalidSnapshot { detail, .. }) if detail.contains("not referenced")
    ));
}

// ------------------------------------------------------------ poll snapshots

/// `swap_loop` with a poll snapshot: the swapped registers, resumed at the
/// header.
fn pollable_swap_loop() -> TraceIr {
    let mut ir = swap_loop();
    let mut poll = ir.snapshots[0].clone();
    poll.id = SnapshotId(u32::try_from(ir.snapshots.len()).unwrap());
    poll.phase = SnapshotPhase::After {
        pc: 5,
        successor: 1,
    };
    ir.poll_snapshot = Some(poll.id);
    ir.snapshots.push(poll);
    ir
}

#[test]
fn a_poll_snapshot_that_resumes_after_the_closing_instruction_verifies() {
    let ir = pollable_swap_loop();
    ir.verify().unwrap();
    let poll = &ir.snapshots[ir.poll_snapshot.unwrap().index()];
    // The poll resumes at the header and records the closing pc, which is
    // what tracing backedge detection reads.
    let resume = poll.resume_point();
    assert_eq!(resume.pc, 1);
    assert_eq!(resume.last_interpreted_pc, Some(5));
}

#[test]
fn verifier_rejects_malformed_poll_snapshots() {
    let mut before = pollable_swap_loop();
    let poll = before.poll_snapshot.unwrap().index();
    before.snapshots[poll].phase = SnapshotPhase::Before { pc: 1 };
    assert!(matches!(
        before.verify(),
        Err(TraceVerifyError::InvalidSnapshot { detail, .. }) if detail.contains("after the closing")
    ));

    let mut missing = pollable_swap_loop();
    missing.poll_snapshot = Some(SnapshotId(99));
    assert!(missing.verify().is_err());

    // The snapshot a poll names must not lose a committed effect.
    let mut ir = worked_case();
    let mut poll = ir.snapshots[0].clone();
    poll.id = SnapshotId(u32::try_from(ir.snapshots.len()).unwrap());
    poll.phase = SnapshotPhase::After {
        pc: 5,
        successor: 1,
    };
    poll.effect_position = 0;
    ir.poll_snapshot = Some(poll.id);
    ir.snapshots.push(poll);
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::EffectOrder {
            instruction: None,
            ..
        })
    ));

    // A poll register must hold its loop parameter's loop-back argument. The
    // swap loop carries r0 = old b, so r0 = old a is the previous iteration.
    let mut stale = pollable_swap_loop();
    let poll = stale.poll_snapshot.unwrap().index();
    let previous = stale.snapshots[poll].frames[0].registers[1];
    stale.snapshots[poll].frames[0].registers.set(0, previous);
    assert!(matches!(
        stale.verify(),
        Err(TraceVerifyError::InvalidSnapshot { detail, .. }) if detail.contains("loop-back argument")
    ));

    // A trace without a body has nothing to poll between.
    let mut empty = pollable_swap_loop();
    empty.body.clear();
    empty.guards.clear();
    assert!(empty.verify().is_err());
}

#[test]
fn a_poll_snapshot_with_unreferenced_values_is_checked_like_any_other() {
    let mut ir = pollable_swap_loop();
    let poll = ir.poll_snapshot.unwrap().index();
    ir.snapshots[poll].frames[0]
        .registers
        .set(0, RegisterEntry::Value(ValueId(999)));
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::UnknownValue(_))
    ));
}

// ---------------------------------------------------------------- loop roles

#[test]
fn registers_cross_the_loop_back_by_how_the_iteration_uses_them() {
    use super::{LoopRole, RegisterUse};
    let role = |written, ops, snapshots| {
        RegisterUse {
            written,
            read_by_operations: ops,
            read_by_snapshots: snapshots,
        }
        .role()
    };
    // Written and read from the previous iteration: carried.
    assert_eq!(role(true, true, false), LoopRole::Carried);
    // A guard before the write restores the previous value: carried too.
    assert_eq!(role(true, false, true), LoopRole::Carried);
    // Never written, read by an operation: an invariant entry input.
    assert_eq!(role(false, true, false), LoopRole::Invariant);
    assert_eq!(role(false, true, true), LoopRole::Invariant);
    // Never written and only snapshotted: `Unchanged` needs no input.
    assert_eq!(role(false, false, true), LoopRole::Unused);
    // Written before it is read: a temporary.
    assert_eq!(role(true, false, false), LoopRole::Unused);
    assert!(!RegisterUse::default().needs_entry_input());
}

#[test]
fn loop_back_arguments_must_keep_their_parameters_types() {
    use super::loop_types_are_stable;
    let mut ir = swap_loop();
    assert!(loop_types_are_stable(&ir));
    // Passing the float `a` where the integer counter belongs.
    let TraceTerminator::LoopBack { args } = &mut ir.terminator else {
        panic!("a loop");
    };
    args[2] = args[0];
    assert!(!loop_types_are_stable(&ir));
    // The reference evaluator agrees the swap itself is a parallel
    // assignment: exits after one and two iterations alternate.
    let ir = swap_loop();
    let registers = [f(1.0), f(2.0), f(0.0), f(0.0)];
    let result = evaluate(&ir, &registers, &[], 10, |_, _| None);
    let Outcome::GuardFailed { restored, .. } = result.outcome else {
        panic!("the counter ends the loop");
    };
    assert_eq!(&restored.registers[..2], &[f(2.0), f(1.0)]);
}

// ------------------------------------------------- opaque values and windows

/// A trace over two table registers and a closure register that compares and
/// moves them, built the way the recorder builds it.
fn opaque_trace() -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), root(5));
    b.mark_written(FrameId(0), 2);
    b.mark_written(FrameId(0), 3);
    b.mark_written(FrameId(0), 4);
    let first = b.input(0, TraceType::Table);
    let second = b.input(1, TraceType::Table);
    let same = emit(
        &mut b,
        0,
        TraceOp::Compare {
            comparison: Comparison::Equal,
            left: first,
            right: second,
        },
        TraceType::Boolean,
    );
    let moved = emit(
        &mut b,
        1,
        TraceOp::Move { source: second },
        TraceType::Table,
    );
    let truthy = emit(
        &mut b,
        2,
        TraceOp::Truthy { source: first },
        TraceType::Boolean,
    );
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 2,
            successor: 3,
        },
        0,
        Some(2),
        vec![
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
            RegisterEntry::Value(same),
            RegisterEntry::Value(moved),
            RegisterEntry::Value(truthy),
        ],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

#[test]
fn opaque_values_move_compare_and_restore_by_lane() {
    let ir = opaque_trace();
    ir.verify().unwrap();
    let registers = [
        TraceScalar::Table(4),
        TraceScalar::Table(4),
        TraceScalar::Nil,
        TraceScalar::Nil,
        TraceScalar::Nil,
    ];
    let Outcome::Exited(restored) = evaluate(&ir, &registers, &[], 1, |_, _| None).outcome else {
        panic!("the trace exits");
    };
    assert_eq!(
        &restored.registers[2..],
        &[
            TraceScalar::Boolean(true),
            TraceScalar::Table(4),
            TraceScalar::Boolean(true)
        ]
    );
    let different = [
        TraceScalar::Table(4),
        TraceScalar::Table(5),
        TraceScalar::Nil,
        TraceScalar::Nil,
        TraceScalar::Nil,
    ];
    let Outcome::Exited(restored) = evaluate(&ir, &different, &[], 1, |_, _| None).outcome else {
        panic!("the trace exits");
    };
    assert_eq!(restored.registers[2], TraceScalar::Boolean(false));
    assert_eq!(restored.registers[3], TraceScalar::Table(5));
    // A closure in a table's place is another entry type.
    let closure = [
        TraceScalar::Closure(4),
        TraceScalar::Table(4),
        TraceScalar::Nil,
        TraceScalar::Nil,
        TraceScalar::Nil,
    ];
    assert_eq!(
        evaluate(&ir, &closure, &[], 1, |_, _| None).outcome,
        Outcome::EntryMismatch
    );
}

#[test]
fn verifier_rejects_opaque_constants_and_ordering() {
    // A lane id is only meaningful for one run, so no constant can be one.
    expect_opaque_error(
        |ir| {
            ir.body[1].op = TraceOp::Constant(TraceScalar::Table(1));
        },
        |error| matches!(error, TraceVerifyError::TypeMismatch { .. }),
    );
    // Identity has no order.
    expect_opaque_error(
        |ir| {
            ir.body[0].op = TraceOp::Compare {
                comparison: Comparison::LessThan,
                left: ValueId(0),
                right: ValueId(1),
            };
        },
        |error| matches!(error, TraceVerifyError::TypeMismatch { .. }),
    );
    // A table against a closure.
    expect_opaque_error(
        |ir| {
            ir.values[1].ty = TraceType::Closure;
            ir.entry.inputs[1].ty = TraceType::Closure;
        },
        |error| {
            matches!(
                error,
                TraceVerifyError::TypeMismatch { .. } | TraceVerifyError::ValueTableMismatch(_)
            )
        },
    );
    // Neither a snapshot constant nor an initial loop value can be opaque.
    expect_opaque_error(
        |ir| {
            ir.snapshots[0].frames[0]
                .registers
                .set(3, RegisterEntry::Const(TraceScalar::Closure(2)));
        },
        |error| matches!(error, TraceVerifyError::InvalidSnapshot { .. }),
    );
}

fn expect_opaque_error(
    mutate: impl FnOnce(&mut TraceIr),
    check: impl FnOnce(&TraceVerifyError) -> bool,
) {
    let mut ir = opaque_trace();
    mutate(&mut ir);
    let error = ir.verify().expect_err("the mutated trace is invalid");
    assert!(check(&error), "{error:?}");
}

/// A trace over a window of three guest registers, one upvalue cell and one
/// global. It adds the cell to the global each time and stores the sum back.
fn window_trace() -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), root(5));
    b.set_external(vec![ExternalSlot::Upvalue(1), ExternalSlot::Global(7)]);
    b.mark_written(FrameId(0), 0);
    b.mark_written(FrameId(0), 3);
    let cell = b.input(3, TraceType::Float);
    let global = b.input(4, TraceType::Float);
    let sum = emit(
        &mut b,
        0,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: cell,
            right: global,
        },
        TraceType::Float,
    );
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 0,
            successor: 1,
        },
        0,
        Some(0),
        vec![
            RegisterEntry::Value(sum),
            RegisterEntry::Unchanged,
            RegisterEntry::Unchanged,
            RegisterEntry::Value(sum),
            RegisterEntry::Unchanged,
        ],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

#[test]
fn a_window_with_a_cell_and_a_global_evaluates_and_verifies() {
    let ir = window_trace();
    ir.verify().unwrap();
    let registers = [
        TraceScalar::Nil,
        TraceScalar::Nil,
        TraceScalar::Nil,
        f(2.0),
        f(0.5),
    ];
    let Outcome::Exited(restored) = evaluate(&ir, &registers, &[], 1, |_, _| None).outcome else {
        panic!("the trace exits");
    };
    assert_eq!(restored.registers[0], f(2.5));
    assert_eq!(restored.registers[3], f(2.5), "the cell takes the sum");
    assert_eq!(restored.registers[4], f(0.5), "the global is left alone");
}

#[test]
fn a_global_slot_cannot_share_a_trace_with_an_effect() {
    let mut b = TraceBuilder::new(TraceId(0), root(5));
    b.set_external(vec![ExternalSlot::Upvalue(1), ExternalSlot::Global(7)]);
    let global = b.input(4, TraceType::Float);
    let effect = b.next_effect();
    b.emit(
        TraceSite {
            frame: FrameId(0),
            pc: 0,
        },
        TraceOp::Effect {
            effect,
            label: STORE,
            operands: vec![global],
            result: None,
        },
        None,
    );
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 0,
            successor: 1,
        },
        1,
        Some(0),
        vec![RegisterEntry::Unchanged; 5],
    ));
    let ir = b.finish(TraceTerminator::Exit { snapshot: exit });
    assert!(
        matches!(ir.verify(), Err(TraceVerifyError::InvalidEntry { .. })),
        "{:?}",
        ir.verify()
    );
}

#[test]
fn verifier_checks_the_external_slots() {
    let check = |mutate: &dyn Fn(&mut TraceIr), what: &str| {
        let mut ir = window_trace();
        mutate(&mut ir);
        assert!(
            matches!(ir.verify(), Err(TraceVerifyError::InvalidEntry { .. })),
            "{what}"
        );
    };
    check(
        &|ir| ir.external[1] = ExternalSlot::Upvalue(1),
        "one cell cannot have two slots",
    );
    check(
        &|ir| ir.written_registers.push((FrameId(0), 4)),
        "a global is never written",
    );
    check(
        &|ir| {
            ir.entry.inputs[1].ty = TraceType::Nil;
            ir.values[1].ty = TraceType::Nil;
        },
        "a global cannot read as nil",
    );
    check(
        &|ir| ir.external = vec![ExternalSlot::Upvalue(0); 6],
        "more slots than the window has registers",
    );
}

#[test]
fn the_optimizer_folds_truthiness_of_opaque_values_and_keeps_their_identity_tests() {
    use super::optimize::{OptimizeLevel, optimize_trace};
    let ir = opaque_trace();
    let (optimized, _) = optimize_trace(&ir, OptimizeLevel::Optimized);
    optimized.verify().unwrap();
    // `truthy table` is the constant true. The identity compare of two
    // different inputs stays.
    assert!(
        !optimized
            .body
            .iter()
            .any(|instruction| matches!(instruction.op, TraceOp::Truthy { .. }))
    );
    assert!(
        optimized
            .body
            .iter()
            .any(|instruction| matches!(instruction.op, TraceOp::Compare { .. }))
    );
    let registers = [
        TraceScalar::Table(1),
        TraceScalar::Table(2),
        TraceScalar::Nil,
        TraceScalar::Nil,
        TraceScalar::Nil,
    ];
    let expected = evaluate(&ir, &registers, &[], 1, |_, _| None).outcome;
    let actual = evaluate(&optimized, &registers, &[], 1, |_, _| None).outcome;
    assert_eq!(actual, expected);
    // A table compared with itself is equal without looking.
    let mut b = TraceBuilder::new(TraceId(0), root(3));
    b.mark_written(FrameId(0), 1);
    let table = b.input(0, TraceType::Table);
    let same = emit(
        &mut b,
        0,
        TraceOp::Compare {
            comparison: Comparison::Equal,
            left: table,
            right: table,
        },
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
            RegisterEntry::Unchanged,
            RegisterEntry::Value(same),
            RegisterEntry::Unchanged,
        ],
    ));
    let ir = b.finish(TraceTerminator::Exit { snapshot: exit });
    let (optimized, _) = optimize_trace(&ir, OptimizeLevel::Optimized);
    assert!(
        !optimized
            .body
            .iter()
            .any(|instruction| matches!(instruction.op, TraceOp::Compare { .. })),
        "{optimized:#?}"
    );
}

// -------------------------------------------------------- inlined frames

/// One call of pinned closure 0 from root pc 2, with a guard inside the
/// callee whose snapshot lists both frames. Root registers: `r0` the callee,
/// `r1` the argument. The callee has two registers.
fn call_trace() -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(7), root(2));
    b.set_pinned_closures(vec![0xabc]);
    let callee = b.add_frame(inlined(2, 0, 2));
    let function = b.input(0, TraceType::Closure);
    let x = b.input(1, TraceType::Float);
    let pinned = emit(
        &mut b,
        2,
        TraceOp::Constant(TraceScalar::Closure(0)),
        TraceType::Closure,
    );
    let same = emit(
        &mut b,
        2,
        TraceOp::Compare {
            comparison: Comparison::Equal,
            left: function,
            right: pinned,
        },
        TraceType::Boolean,
    );
    let before = b.snapshot(snap(
        SnapshotPhase::Before { pc: 2 },
        0,
        Some(1),
        vec![RegisterEntry::Unchanged; 2],
    ));
    let g0 = b.guard(site(2), before);
    b.emit(
        site(2),
        TraceOp::GuardBranch {
            condition: same,
            expected: true,
            guard: g0,
        },
        None,
    );
    let in_callee = |pc| TraceSite { frame: callee, pc };
    let copy = b
        .emit(site(2), TraceOp::Move { source: x }, Some(TraceType::Float))
        .unwrap();
    let positive = b
        .emit(
            in_callee(0),
            TraceOp::Compare {
                comparison: Comparison::LessThan,
                left: copy,
                right: copy,
            },
            Some(TraceType::Boolean),
        )
        .unwrap();
    let inside = b.snapshot(Snapshot::new(
        SnapshotPhase::After {
            pc: 0,
            successor: 1,
        },
        0,
        vec![
            SnapshotFrame {
                frame: FrameId(0),
                last_interpreted_pc: Some(2),
                registers: vec![RegisterEntry::Unchanged; 2].into(),
            },
            SnapshotFrame {
                frame: callee,
                last_interpreted_pc: Some(0),
                registers: vec![RegisterEntry::Value(copy), RegisterEntry::Const(f(0.0))].into(),
            },
        ],
    ));
    let g1 = b.guard(in_callee(0), inside);
    b.emit(
        in_callee(0),
        TraceOp::GuardBranch {
            condition: positive,
            expected: false,
            guard: g1,
        },
        None,
    );
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 3,
            successor: 4,
        },
        0,
        Some(3),
        vec![RegisterEntry::Unchanged; 2],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

fn expect_call_error(
    mutate: impl FnOnce(&mut TraceIr),
    check: impl FnOnce(&TraceVerifyError) -> bool,
) {
    let mut ir = call_trace();
    ir.verify().expect("the unmutated trace verifies");
    mutate(&mut ir);
    let error = ir
        .verify()
        .expect_err("mutated trace must fail verification");
    assert!(check(&error), "unexpected error: {error}");
}

fn invalid_entry(detail: &'static str) -> impl FnOnce(&TraceVerifyError) -> bool {
    move |error| matches!(error, TraceVerifyError::InvalidEntry { detail: found, .. } if found.contains(detail))
}

#[test]
fn verifier_accepts_a_call_the_trace_inlined() {
    let ir = call_trace();
    ir.verify().unwrap();
    // `1 < 1` is false, which the callee's guard expects, so the trace exits
    // through its terminator with the root frame alone.
    let entry = [TraceScalar::Closure(0), f(1.0)];
    let evaluation = evaluate(&ir, &entry, &[], 1, |_, _| None);
    let Outcome::Exited(restored) = evaluation.outcome else {
        panic!("{:?}", evaluation.outcome)
    };
    assert_eq!(
        restored.outer,
        [] as [crate::jit_ir::trace::reference::RestoredFrame; 0]
    );
    assert_eq!((restored.pc, restored.last_interpreted_pc), (4, Some(3)));
    // Forcing the callee's guard restores the callee, with the root stopped
    // at its call.
    let evaluation = evaluate_forced(
        &ir,
        &entry,
        &[],
        1,
        Some(ForcedExit {
            frame: FrameId(1),
            pc: 0,
            guard: None,
            iteration: 0,
        }),
        |_, _| None,
    );
    let Outcome::GuardFailed { restored, .. } = evaluation.outcome else {
        panic!("{:?}", evaluation.outcome)
    };
    assert_eq!(restored.registers, vec![f(1.0), f(0.0)]);
    assert_eq!((restored.pc, restored.last_interpreted_pc), (1, Some(0)));
    assert_eq!(restored.outer.len(), 1);
    assert_eq!(restored.outer[0].frame, FrameId(0));
    assert_eq!(restored.outer[0].last_interpreted_pc, Some(2));
    assert_eq!(restored.outer[0].registers, entry);
}

#[test]
fn verifier_rejects_malformed_inlined_frames() {
    expect_call_error(
        |ir| ir.frames[0].inlined = ir.frames[1].inlined,
        invalid_entry("root frame"),
    );
    // A snapshot that lists the frame notices first.
    expect_call_error(
        |ir| ir.frames[1].inlined = None,
        |error| matches!(error, TraceVerifyError::InvalidSnapshot { detail, .. } if detail.contains("not inlined")),
    );
    expect_call_error(
        |ir| ir.frames[1].inlined.as_mut().unwrap().closure = 3,
        invalid_entry("unpinned closure"),
    );
    // The snapshots that list the frame notice a broken chain first.
    let broken = |error: &TraceVerifyError| {
        matches!(
            error,
            TraceVerifyError::InvalidEntry { .. } | TraceVerifyError::InvalidSnapshot { .. }
        )
    };
    expect_call_error(
        |ir| ir.frames[1].inlined.as_mut().unwrap().parent = FrameId(1),
        broken,
    );
    expect_call_error(
        |ir| ir.frames[1].inlined.as_mut().unwrap().call_pc = 999,
        broken,
    );
    expect_call_error(
        |ir| {
            let call = ir.frames[1].inlined.as_mut().unwrap();
            call.result_base = 1;
            call.results = 2;
        },
        invalid_entry("past its parent"),
    );
}

#[test]
fn verifier_wants_the_closure_of_a_frame_that_runs_the_closure_of_its_call() {
    // The snapshots of such a frame have one register more than its chunk:
    // the closure the run called. A frame marked that way whose snapshots
    // lack it is refused, so no exit rebuilds it with a closure it cannot
    // name.
    expect_call_error(
        |ir| ir.frames[1].inlined.as_mut().unwrap().dynamic = true,
        |error| {
            matches!(error, TraceVerifyError::InvalidSnapshot { detail, .. }
                if detail.contains("closure of this run"))
        },
    );
}

#[test]
fn verifier_checks_the_results_a_metamethod_frame_keeps() {
    // A read or `#t` that runs a handler keeps one value; a write keeps none.
    for (entry, results) in [
        (Entry::Index, 1),
        (Entry::Len, 1),
        (Entry::NewIndex, 0),
        (Entry::Operator, 1),
        (Entry::Compare { invert: true }, 1),
        (
            Entry::Branch {
                expected: false,
                invert: true,
            },
            1,
        ),
        (
            Entry::Concat {
                first: 0,
                last: 1,
                pending: 0,
            },
            1,
        ),
    ] {
        let mut ir = call_trace();
        let call = ir.frames[1].inlined.as_mut().unwrap();
        (call.entry, call.results) = (entry, results);
        ir.verify().unwrap();
        for wrong in [0, 1, 2] {
            if wrong == results {
                continue;
            }
            let mut ir = call_trace();
            let call = ir.frames[1].inlined.as_mut().unwrap();
            (call.entry, call.results) = (entry, wrong);
            assert!(
                ir.verify()
                    .is_err_and(|error| invalid_entry("wrong number of results")(&error)),
                "{entry:?} with {wrong} results"
            );
        }
    }
    // A `tostring` handler keeps the results of its call.
    for results in [0, 1, 2] {
        let mut ir = call_trace();
        let call = ir.frames[1].inlined.as_mut().unwrap();
        (call.entry, call.results) = (Entry::ToString, results);
        ir.verify().unwrap();
    }
    // A metamethod frame is not a protected call.
    expect_call_error(
        |ir| {
            let call = ir.frames[1].inlined.as_mut().unwrap();
            (call.entry, call.protected) = (Entry::Index, Protected::Pcall);
        },
        invalid_entry("protected call"),
    );
}

#[test]
fn verifier_rejects_snapshots_that_are_not_a_chain_of_calls() {
    let invalid = |detail: &'static str| move |error: &TraceVerifyError| matches!(error, TraceVerifyError::InvalidSnapshot { detail: found, .. } if found.contains(detail));
    // The outer frame is not stopped at its call.
    expect_call_error(
        |ir| ir.snapshots[1].frames[0].last_interpreted_pc = Some(1),
        invalid("stopped at its call"),
    );
    expect_call_error(
        |ir| ir.snapshots[1].frames[0].last_interpreted_pc = None,
        invalid("stopped at its call"),
    );
    // Two frames the second of which no call of the first made.
    expect_call_error(
        |ir| {
            ir.frames.push(inlined(2, 0, 1));
            ir.frames[2].inlined.as_mut().unwrap().parent = FrameId(1);
            ir.snapshots[1].frames[1].frame = FrameId(2);
            ir.body[5].site.frame = FrameId(2);
            ir.guards[1].site.frame = FrameId(2);
        },
        |error| {
            matches!(
                error,
                TraceVerifyError::InvalidSnapshot { .. } | TraceVerifyError::InvalidSite { .. }
            )
        },
    );
    // A root frame in the second place.
    expect_call_error(
        |ir| ir.snapshots[1].frames[1].frame = FrameId(0),
        invalid("appears twice"),
    );
    // A snapshot that lists the callee without the root.
    expect_call_error(
        |ir| {
            ir.snapshots[1].frames.remove(0);
        },
        invalid("first frame is not the root"),
    );
}

#[test]
fn verifier_allows_a_closure_constant_only_for_a_pinned_lane() {
    expect_call_error(
        |ir| {
            let TraceOp::Constant(scalar) = &mut ir.body[0].op else {
                panic!()
            };
            *scalar = TraceScalar::Closure(1);
        },
        |error| matches!(error, TraceVerifyError::TypeMismatch { detail, .. } if detail.contains("pinned")),
    );
    expect_call_error(
        |ir| {
            ir.values[ir.body[0].result.unwrap().index()].ty = TraceType::Table;
            let TraceOp::Constant(scalar) = &mut ir.body[0].op else {
                panic!()
            };
            *scalar = TraceScalar::Table(0);
        },
        |error| matches!(error, TraceVerifyError::TypeMismatch { .. }),
    );
    // A snapshot cannot hold one.
    expect_call_error(
        |ir| {
            ir.snapshots[0].frames[0]
                .registers
                .set(0, RegisterEntry::Const(TraceScalar::Closure(0)));
        },
        |error| matches!(error, TraceVerifyError::InvalidSnapshot { detail, .. } if detail.contains("opaque")),
    );
}

#[test]
fn verifier_checks_the_cells_of_pinned_closures_and_their_aliases() {
    // A cell slot of a closure the trace does not pin.
    expect_call_error(
        |ir| {
            ir.frames[0].register_count = 3;
            for snapshot in &mut ir.snapshots {
                snapshot.frames[0].registers.push(RegisterEntry::Unchanged);
            }
            ir.external = vec![ExternalSlot::ClosureUpvalue {
                closure: 4,
                index: 0,
            }];
        },
        invalid_entry("unpinned closure"),
    );
    // A cell alias must name things of the trace.
    let with_slot = |ir: &mut TraceIr| {
        ir.frames[0].register_count = 3;
        for snapshot in &mut ir.snapshots {
            snapshot.frames[0].registers.push(RegisterEntry::Unchanged);
        }
        ir.external = vec![ExternalSlot::ClosureUpvalue {
            closure: 0,
            index: 0,
        }];
    };
    let slot = CellSource::ClosureUpvalue {
        closure: 0,
        index: 0,
    };
    let other = CellSource::ClosureUpvalue {
        closure: 0,
        index: 1,
    };
    let invalid_alias = |detail: &'static str| move |error: &TraceVerifyError| matches!(error, TraceVerifyError::InvalidEntry { detail: found, .. } if found.contains(detail));
    // A name that has a window slot of its own stands for its cell, and a
    // second name for it shares that slot.
    let mut ir = call_trace();
    with_slot(&mut ir);
    ir.cell_aliases = vec![CellAlias {
        cell: other,
        is: slot,
    }];
    ir.verify().unwrap();
    // Aliases of the root frame's captured locals too.
    ir.cell_aliases.push(CellAlias {
        cell: CellSource::Upvalue(3),
        is: CellSource::Captured(1),
    });
    ir.verify().unwrap();
    expect_call_error(
        |ir| {
            with_slot(ir);
            ir.cell_aliases = vec![CellAlias {
                cell: slot,
                is: other,
            }];
        },
        invalid_alias("window slot"),
    );
    expect_call_error(
        |ir| {
            with_slot(ir);
            ir.cell_aliases = vec![CellAlias {
                cell: other,
                is: other,
            }];
        },
        invalid_alias("itself"),
    );
    expect_call_error(
        |ir| {
            with_slot(ir);
            ir.cell_aliases = vec![CellAlias {
                cell: CellSource::Captured(0),
                is: slot,
            }];
        },
        invalid_alias("register or itself"),
    );
    expect_call_error(
        |ir| {
            with_slot(ir);
            ir.cell_aliases = vec![CellAlias {
                cell: other,
                is: CellSource::Captured(7),
            }];
        },
        invalid_alias("outside the trace"),
    );
    expect_call_error(
        |ir| {
            with_slot(ir);
            ir.cell_aliases = vec![
                CellAlias {
                    cell: other,
                    is: slot,
                },
                CellAlias {
                    cell: other,
                    is: CellSource::Captured(0),
                },
            ];
        },
        invalid_alias("repeats"),
    );
    expect_call_error(
        |ir| {
            with_slot(ir);
            ir.cell_aliases = vec![CellAlias {
                cell: CellSource::ClosureUpvalue {
                    closure: 5,
                    index: 0,
                },
                is: slot,
            }];
        },
        invalid_alias("outside the trace"),
    );
}

/// A read in an inlined frame at pc 1, guarded through a snapshot. With
/// `at_call` it is a `Before` snapshot of the call at pc 2 of the root frame,
/// otherwise one of the callee's own frame at pc 1. `reexec` says whether the
/// call has to start over when something in it leaves, and `store_first` puts
/// a committed effect in the callee before the guard.
fn reexec_trace(reexec: bool, store_first: bool, at_call: bool) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(5), root(2));
    b.set_pinned_closures(vec![9]);
    let mut frame = inlined(1, 0, 2);
    frame.inlined.as_mut().unwrap().reexec = reexec;
    let callee = b.add_frame(frame);
    let i_in = b.input(0, TraceType::Float);
    let in_callee = |pc| TraceSite { frame: callee, pc };
    // The call's own instruction in the root frame, which the callee's
    // instructions run inside.
    emit(&mut b, 2, TraceOp::Constant(f(1.0)), TraceType::Float);
    if store_first {
        let effect = b.next_effect();
        b.emit(
            in_callee(0),
            TraceOp::Effect {
                effect,
                label: STORE,
                operands: vec![i_in],
                result: None,
            },
            None,
        );
    }
    let read = b
        .emit(
            in_callee(1),
            TraceOp::Read {
                label: LOAD,
                operands: vec![i_in],
                result: TraceType::Any,
            },
            Some(TraceType::Any),
        )
        .unwrap();
    let effects = b.effect_position();
    let registers = vec![RegisterEntry::Value(i_in), RegisterEntry::Const(f(0.0))];
    let s = b.snapshot(if at_call {
        snap(SnapshotPhase::Before { pc: 2 }, effects, Some(1), registers)
    } else {
        Snapshot::new(
            SnapshotPhase::Before { pc: 1 },
            effects,
            vec![
                SnapshotFrame {
                    frame: FrameId(0),
                    last_interpreted_pc: Some(2),
                    registers: registers.into(),
                },
                SnapshotFrame {
                    frame: callee,
                    last_interpreted_pc: None,
                    registers: vec![RegisterEntry::Const(f(0.0))].into(),
                },
            ],
        )
    });
    let g = b.guard(in_callee(1), s);
    b.emit(
        in_callee(1),
        TraceOp::GuardType {
            source: read,
            expected: TraceType::Float,
            guard: g,
        },
        Some(TraceType::Float),
    );
    b.finish(TraceTerminator::Exit {
        snapshot: SnapshotId(0),
    })
}

#[test]
fn a_call_that_starts_over_lets_its_guards_leave_before_the_call() {
    // Any guard in a call may leave before the call instruction, in the
    // frame that made it, when the call stored nothing.
    reexec_trace(false, false, true).verify().unwrap();
    reexec_trace(true, false, true).verify().unwrap();
    // Without a store, the guard may also describe its own frame.
    reexec_trace(false, false, false).verify().unwrap();
    // A call that has to start over has no other way out.
    assert!(matches!(
        reexec_trace(true, false, false).verify(),
        Err(TraceVerifyError::GuardLocation { .. })
    ));
    // Starting over after the call stored would store twice.
    assert!(matches!(
        reexec_trace(false, true, true).verify(),
        Err(TraceVerifyError::EffectReplay { .. })
    ));
    assert!(matches!(
        reexec_trace(true, true, true).verify(),
        Err(TraceVerifyError::EffectReplay { .. })
    ));
}

/// A loop over `r0` that reads `r1`, nil on some entries and a float on
/// others: the entry takes `r1` untyped, and a type guard at the top of the
/// body narrows it. `misuse` reads the untyped input elsewhere too.
fn nil_or_number_loop(misuse: bool) -> TraceIr {
    use RegisterEntry::{Unchanged, Value};
    let mut b = TraceBuilder::new(TraceId(7), root(2));
    b.mark_written(FrameId(0), 0);
    let i_in = b.input(0, TraceType::Float);
    let lx_in = b.input(1, TraceType::Any);
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let s0 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 1 },
        b.effect_position(),
        Some(0),
        vec![Value(i), Unchanged],
    ));
    let g0 = b.guard(site(1), s0);
    let lx = emit(
        &mut b,
        1,
        TraceOp::GuardType {
            source: lx_in,
            expected: TraceType::Float,
            guard: g0,
        },
        TraceType::Float,
    );
    let next = emit(
        &mut b,
        2,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: i,
            right: if misuse { lx_in } else { lx },
        },
        TraceType::Float,
    );
    b.finish(TraceTerminator::LoopBack { args: vec![next] })
}

#[test]
fn an_untyped_entry_input_that_only_a_type_guard_reads_verifies() {
    nil_or_number_loop(false).verify().unwrap();
}

/// A side that writes a register it takes untyped (the outer side of a `y`
/// loop that computes `ly` again) keeps it `Unchanged` until the write: the
/// frame holds it, with what a link stored in its slot written back. A
/// typed input the trace writes may not stay `Unchanged`.
#[test]
fn an_untyped_entry_input_the_trace_writes_may_stay_unchanged_before_the_write() {
    let mut ir = nil_or_number_loop(false);
    ir.written_registers.push((FrameId(0), 1));
    ir.written_registers.sort_unstable();
    ir.verify().unwrap();
    // The same register typed is restored wrong by the snapshot.
    let mut ir = nil_or_number_loop(false);
    ir.written_registers.push((FrameId(0), 1));
    ir.written_registers.sort_unstable();
    ir.entry.inputs[1].ty = TraceType::Float;
    let value = ir.entry.inputs[1].value;
    ir.values[value.index()].ty = TraceType::Float;
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::UnchangedWritten { register: 1, .. })
    ));
}

#[test]
fn verifier_rejects_an_untyped_entry_input_read_past_its_guard() {
    let error = nil_or_number_loop(true)
        .verify()
        .expect_err("an untyped input read by arithmetic must fail");
    assert!(
        matches!(
            error,
            TraceVerifyError::InvalidEntry { .. } | TraceVerifyError::TypeMismatch { .. }
        ),
        "unexpected error: {error}"
    );
}

#[test]
fn verifier_rejects_an_untyped_entry_input_handed_to_the_loop() {
    let mut ir = nil_or_number_loop(false);
    let lx_in = ir.entry.inputs[1].value;
    ir.loop_params[0].init = LoopParamInit::Input(lx_in);
    assert!(ir.verify().is_err());
}

#[test]
fn verifier_rejects_an_untyped_entry_input_outside_the_frame() {
    let mut ir = nil_or_number_loop(false);
    ir.entry.inputs[1].register = 5;
    assert!(ir.verify().is_err());
}
