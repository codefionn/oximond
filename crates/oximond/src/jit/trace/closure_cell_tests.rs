//! Calls of closures whose cells native code reads and writes itself, through
//! the row of the closure's lane ([`CLOSURE_ROW_WORDS`]): the prototype check
//! of a call site and the `GetUpvalue` and `SetUpvalue` of the callee.
//!
//! The tests play the VM. A closure is a prototype name and a list of cell
//! values the test owns; the dispatcher fills a closure's row with the name
//! and the addresses of its cells. The trace runs at both tiers over closures
//! that match, that run another prototype, whose cell holds another type, that
//! have no such cell, and that are no Lua closure at all.

use super::*;
use crate::jit::OptimizationTier;
use crate::jit_ir::trace::{
    ArithmeticOp, FrameId, HelperId, RegisterEntry, Snapshot, SnapshotFrame, SnapshotPhase,
    TraceBuilder, TraceFrameInfo, TraceId, TraceIr, TraceOp, TraceScalar, TraceSite,
    TraceTerminator, TraceType,
};
use crate::runtime::Value;

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

/// One closure of the test's world: the name of its prototype (zero for a
/// lane that is no Lua closure) and its cells.
pub(super) struct Closure {
    pub(super) prototype: usize,
    pub(super) cells: Vec<Value>,
}

pub(super) struct Dispatcher {
    pub(super) closures: Vec<Closure>,
}

impl TraceTables for Dispatcher {
    fn lane(&mut self, _: &Value) -> Option<usize> {
        None
    }

    fn row(&mut self, lane: usize, row: &mut [usize]) {
        row.fill(0);
        let closure = &mut self.closures[lane];
        row[0] = closure.prototype;
        for (slot, cell) in row[1..].iter_mut().zip(&mut closure.cells) {
            *slot = std::ptr::from_mut(cell) as usize;
        }
    }

    fn closure_lanes(&mut self, lanes: &mut Vec<usize>) {
        lanes.extend(0..self.closures.len());
    }
}

/// Registers: r0 the callee, r1 the result.
///
/// ```lua
/// -- pinned: lane 0, prototype P
/// y = callee.cell[0] + 1.0    -- only if callee runs P
/// callee.cell[0] = y
/// ```
pub(super) fn trace(read: HelperId) -> TraceIr {
    trace_pinned_at(read, 0)
}

/// As [`trace`], with the pinned closure in lane `pinned` of the rows.
pub(super) fn trace_pinned_at(read: HelperId, pinned: u64) -> TraceIr {
    use RegisterEntry::{Const, Unchanged, Value as Val};
    let nil = Const(TraceScalar::Nil);
    let mut b = TraceBuilder::new(
        TraceId(0),
        TraceFrameInfo {
            register_count: 2,
            code_len: 8,
            inlined: None,
            activation: None,
        },
    );
    b.mark_written(FrameId(0), 1);
    b.set_pinned_closures((0..=pinned).map(|lane| 0xfeed + lane).collect());
    let callee = b.input(0, TraceType::Closure);
    let before = |b: &mut TraceBuilder, pc: usize, effects: u32| {
        let snapshot = b.snapshot(snap(
            SnapshotPhase::Before { pc },
            effects,
            vec![Unchanged, nil],
        ));
        b.guard(site(pc), snapshot)
    };
    let pinned = b
        .emit(
            site(0),
            TraceOp::Constant(TraceScalar::Closure(pinned)),
            Some(TraceType::Closure),
        )
        .unwrap();
    let call = before(&mut b, 0, 0);
    let same = b
        .emit(
            site(0),
            TraceOp::CallHelper {
                helper: HelperId::CLOSURE_SAME_PROTO,
                args: vec![callee, pinned],
                effect: None,
                guard: call,
            },
            Some(TraceType::Boolean),
        )
        .unwrap();
    let branch = before(&mut b, 0, 0);
    b.emit(
        site(0),
        TraceOp::GuardBranch {
            condition: same,
            expected: true,
            guard: branch,
        },
        None,
    );
    let index = b
        .emit(
            site(1),
            TraceOp::Constant(TraceScalar::Integer(0)),
            Some(TraceType::Integer),
        )
        .unwrap();
    let read_guard = before(&mut b, 1, 0);
    let first = Some(b.next_effect());
    let (value_ty, one) = match read {
        HelperId::CELL_GET_INTEGER => (TraceType::Integer, TraceScalar::Integer(1)),
        _ => (TraceType::Float, TraceScalar::Float(1.0)),
    };
    let held = b
        .emit(
            site(1),
            TraceOp::CallHelper {
                helper: read,
                args: vec![callee, index],
                effect: first,
                guard: read_guard,
            },
            Some(value_ty),
        )
        .unwrap();
    let one = b
        .emit(site(2), TraceOp::Constant(one), Some(value_ty))
        .unwrap();
    let sum = b
        .emit(
            site(2),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: held,
                right: one,
            },
            Some(value_ty),
        )
        .unwrap();
    let write_guard = before(&mut b, 3, 1);
    let second = Some(b.next_effect());
    b.emit(
        site(3),
        TraceOp::CallHelper {
            helper: HelperId::CELL_SET,
            args: vec![callee, index, sum],
            effect: second,
            guard: write_guard,
        },
        None,
    );
    let exit = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 3,
            successor: 4,
        },
        2,
        vec![Unchanged, Val(sum)],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

struct Run {
    class: Option<TraceGuardClass>,
    exited_at_the_end: bool,
    result: Option<TraceScalar>,
    stores: Vec<Value>,
}

/// Runs `ir` with the closure of lane `lane` as the callee.
fn run(ir: &TraceIr, tier: OptimizationTier, closures: Vec<Closure>, lane: u64) -> Run {
    let compiled = compile_trace(ir, tier).unwrap();
    assert!(compiled.needs_tables());
    // The rows are long enough for a closure's.
    assert_eq!(compiled.row_stride(), CLOSURE_ROW_WORDS);
    let mut tables = Dispatcher { closures };
    let mut space = TraceTableSpace::new();
    let inputs = [TraceInputSlot::closure(lane)];
    let mut area = vec![0_i64; compiled.exit_area_len()];
    let mut control = TraceControl::new(100);
    let outcome = compiled.run_with_tables(
        &inputs,
        &[],
        &mut control,
        &mut area,
        &mut space,
        &mut tables,
    );
    let TraceOutcome::Exit { index } = outcome else {
        panic!("the trace refused entry: {outcome:?}");
    };
    let descriptor = &compiled.exits()[index];
    let snapshot = &compiled.lowered_ir().snapshots[descriptor.snapshot.index()];
    let result = match snapshot.frames[0].registers[1] {
        RegisterEntry::Value(value) => descriptor.scalar(&area, value),
        _ => None,
    };
    let stores = tables.closures[lane as usize].cells.clone();
    Run {
        class: descriptor.guard,
        exited_at_the_end: descriptor.origin.is_none(),
        result,
        stores,
    }
}

pub(super) fn closure(prototype: usize, cells: Vec<Value>) -> Closure {
    Closure { prototype, cells }
}

#[test]
fn a_closure_of_the_pinned_prototype_reads_and_writes_its_own_cell() {
    for tier in TIERS {
        let ir = trace(HelperId::CELL_GET_FLOAT);
        let run = run(
            &ir,
            tier,
            vec![
                closure(0xaa, vec![Value::Number(1.0)]),
                closure(0xaa, vec![Value::Number(10.0), Value::Number(7.0)]),
            ],
            1,
        );
        assert!(run.exited_at_the_end, "{tier:?}");
        assert_eq!(run.result, Some(TraceScalar::Float(11.0)), "{tier:?}");
        assert_eq!(
            run.stores,
            vec![Value::Number(11.0), Value::Number(7.0)],
            "{tier:?}"
        );
    }
}

#[test]
fn an_integer_cell_is_read_and_written_as_an_integer() {
    for tier in TIERS {
        let ir = trace(HelperId::CELL_GET_INTEGER);
        let run = run(
            &ir,
            tier,
            vec![
                closure(0xaa, vec![Value::Integer(1)]),
                closure(0xaa, vec![Value::Integer(-5)]),
            ],
            1,
        );
        assert!(run.exited_at_the_end, "{tier:?}");
        assert_eq!(run.result, Some(TraceScalar::Integer(-4)), "{tier:?}");
        assert_eq!(run.stores, vec![Value::Integer(-4)], "{tier:?}");
    }
}

#[test]
fn another_prototype_a_native_function_or_a_cell_of_another_type_leaves_the_trace() {
    // The pinned closure is lane 0 of prototype 0xaa. Each callee fails
    // before it writes anything: the prototype check is a callee guard, the
    // cell reads are helper guards.
    let cases: [(&str, Closure, TraceGuardClass); 4] = [
        (
            "another prototype",
            closure(0xbb, vec![Value::Number(3.0)]),
            TraceGuardClass::Callee,
        ),
        (
            "no Lua closure",
            closure(0, vec![]),
            TraceGuardClass::Callee,
        ),
        (
            "a cell of another type",
            closure(0xaa, vec![Value::Integer(3)]),
            TraceGuardClass::Host,
        ),
        (
            "a closure with no such cell",
            closure(0xaa, vec![]),
            TraceGuardClass::Host,
        ),
    ];
    for tier in TIERS {
        for (label, callee, class) in &cases {
            let untouched = callee.cells.clone();
            let ir = trace(HelperId::CELL_GET_FLOAT);
            let run = run(
                &ir,
                tier,
                vec![
                    closure(0xaa, vec![Value::Number(1.0)]),
                    Closure {
                        prototype: callee.prototype,
                        cells: callee.cells.clone(),
                    },
                ],
                1,
            );
            assert!(!run.exited_at_the_end, "{label} at {tier:?}");
            assert_eq!(run.class, Some(*class), "{label} at {tier:?}");
            assert_eq!(run.stores, untouched, "{label} at {tier:?}");
        }
    }
}
