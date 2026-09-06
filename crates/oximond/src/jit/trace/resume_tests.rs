//! A loop that loads coroutines out of an array and resumes them through a
//! helper ([`HelperId::coroutine_resume`]).
//!
//! The VM implements the helper (it runs the thread); these tests install a
//! stand-in with the same calling convention and check what native code does
//! around it: the array read of a thread checks the slot's tag and gives the
//! lane the VM gave, the helper gets the thread as a tagged lane with the call's
//! other arguments and the address of the binding words it must rewrite, its
//! integer result feeds the loop, a miss leaves through the call's guard, and the code
//! reads the array words again after the call.

#![allow(unsafe_code, clippy::too_many_lines)]

use super::*;
use crate::jit::OptimizationTier;
use crate::jit_ir::Comparison;
use crate::jit_ir::trace::{
    ArithmeticOp, BindingKind, FrameId, HelperId, LoopParamInit, RegisterEntry, Snapshot,
    SnapshotFrame, SnapshotPhase, TableBinding, TraceBuilder, TraceFrameInfo, TraceId, TraceIr,
    TraceOp, TraceScalar, TraceSite, TraceTerminator, TraceType, ValueId,
};
use crate::runtime::{Runtime, ThreadHandle, Value};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

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

fn emit(b: &mut TraceBuilder, pc: usize, op: TraceOp, ty: TraceType) -> ValueId {
    b.emit(site(pc), op, Some(ty)).unwrap()
}

/// What the stand-in helper saw, in call order: the tag and bits of its first
/// argument, its argument count and whether it got the binding words.
static SEEN: Mutex<Vec<(i64, i64, i64, bool)>> = Mutex::new(Vec::new());
/// The call, if any, at which the stand-in misses (counting from one).
static MISS_AT: AtomicUsize = AtomicUsize::new(0);
/// Serializes the tests that install the stand-in.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// The stand-in: `value = argument * 2`, and a miss at the call `MISS_AT`
/// names.
#[allow(clippy::too_many_arguments)]
extern "C" fn stand_in(
    _context: usize,
    out: *mut i64,
    words: *mut TraceInputSlot,
    count: i64,
    t0: i64,
    b0: i64,
    _t1: i64,
    b1: i64,
    _t2: i64,
    _b2: i64,
    _t3: i64,
    _b3: i64,
    _t4: i64,
    _b4: i64,
) -> i64 {
    let mut seen = SEEN.lock().unwrap();
    seen.push((t0, b0, count, !words.is_null()));
    if seen.len() == MISS_AT.load(Ordering::SeqCst) {
        return 1;
    }
    // SAFETY: `out` is the control block's result words.
    unsafe { *out = b1 * 2 };
    0
}

/// Registers: r0 the index, r1 the array of threads, r2 the limit, r3 the sum,
/// r4 the thread.
///
/// ```lua
/// repeat
///   thread = threads[i]               -- pc 0
///   value = resume (thread, i)        -- pc 1 (the result is r5)
///   sum = sum + value                 -- pc 2
///   i = i + 1; until i > limit        -- pc 3
/// ```
fn resume_loop() -> TraceIr {
    use RegisterEntry::{Const, Unchanged, Value as Val};
    let nil = Const(TraceScalar::Nil);
    let mut b = TraceBuilder::new(
        TraceId(0),
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
    let i_in = b.input(0, TraceType::Integer);
    let threads = b.input(1, TraceType::Table);
    let limit = b.input(2, TraceType::Integer);
    let sum_in = b.input(3, TraceType::Integer);
    b.set_bindings(vec![TableBinding {
        register: 1,
        kind: BindingKind::Array { first: None },
    }]);
    let i = b.loop_param(TraceType::Integer, LoopParamInit::Input(i_in));
    let sum = b.loop_param(TraceType::Integer, LoopParamInit::Input(sum_in));

    let s0 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 0 },
        0,
        vec![Val(i), Unchanged, Unchanged, Val(sum), nil, nil],
    ));
    let g0 = b.guard(site(0), s0);
    let thread = emit(
        &mut b,
        0,
        TraceOp::ArrayLoad {
            binding: 0,
            table: threads,
            key: i,
            expected: TraceType::Thread,
            past_end: false,
            guard: g0,
        },
        TraceType::Thread,
    );
    let s1 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 1 },
        0,
        vec![Val(i), Unchanged, Unchanged, Val(sum), Val(thread), nil],
    ));
    let g1 = b.guard(site(1), s1);
    let effect = Some(b.next_effect());
    let value = emit(
        &mut b,
        1,
        TraceOp::CallHelper {
            helper: HelperId::coroutine_resume(1, 0),
            args: vec![thread, i],
            effect,
            guard: g1,
        },
        TraceType::Integer,
    );
    let sum_next = emit(
        &mut b,
        2,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: sum,
            right: value,
        },
        TraceType::Integer,
    );
    let one = emit(
        &mut b,
        3,
        TraceOp::Constant(TraceScalar::Integer(1)),
        TraceType::Integer,
    );
    let i_next = emit(
        &mut b,
        3,
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left: i,
            right: one,
        },
        TraceType::Integer,
    );
    let more = emit(
        &mut b,
        3,
        TraceOp::Compare {
            comparison: Comparison::LessEqual,
            left: i_next,
            right: limit,
        },
        TraceType::Boolean,
    );
    let after = vec![
        Val(i_next),
        Unchanged,
        Unchanged,
        Val(sum_next),
        Val(thread),
        Val(value),
    ];
    let s2 = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 3,
            successor: 4,
        },
        1,
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
        1,
        after,
    ));
    b.set_poll_snapshot(poll);
    b.finish(TraceTerminator::LoopBack {
        args: vec![i_next, sum_next],
    })
}

/// The VM's side of a run: the lane of each thread the code meets, in order,
/// after the one the array table has.
struct Dispatcher {
    lanes: Vec<ThreadHandle>,
}

impl TraceTables for Dispatcher {
    fn lane(&mut self, value: &Value) -> Option<usize> {
        let Value::Thread(thread) = value else {
            return None;
        };
        let at = if let Some(at) = self.lanes.iter().position(|held| held == thread) {
            at
        } else {
            self.lanes.push(*thread);
            self.lanes.len() - 1
        };
        // Lane 0 is the array's table.
        Some(at + 1)
    }

    fn row(&mut self, _: usize, row: &mut [usize]) {
        row.fill(0);
    }
}

/// What a run did.
struct Run {
    /// The guard it left through, or `None` for the loop's own exit.
    guard: Option<u32>,
    /// The registers it restored.
    state: Vec<String>,
    iterations: i64,
}

fn run(array: &mut [Value], start: i64, limit: i64, tier: OptimizationTier) -> Run {
    let ir = resume_loop();
    ir.verify().unwrap();
    let trace = compile_trace(&ir, tier).unwrap();
    assert!(trace.needs_tables() && trace.resumes());
    let registers = [
        TraceScalar::Integer(start),
        TraceScalar::Table(0),
        TraceScalar::Integer(limit),
        TraceScalar::Integer(0),
        TraceScalar::Nil,
        TraceScalar::Nil,
    ];
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| TraceInputSlot::from_scalar(registers[usize::from(*register)]))
        .collect();
    let bindings = [TraceBinding::raw_array(
        array.len(),
        array.as_mut_ptr() as usize,
    )];
    let mut area = vec![0_i64; trace.exit_area_len()];
    let mut control = TraceControl::new(100_000);
    let mut space = TraceTableSpace::new();
    let mut dispatcher = Dispatcher { lanes: Vec::new() };
    let outcome = trace.run_with_tables(
        &inputs,
        &bindings,
        &mut control,
        &mut area,
        &mut space,
        &mut dispatcher,
    );
    let TraceOutcome::Exit { index } = outcome else {
        panic!("the trace refused entry");
    };
    let descriptor = &trace.exits()[index];
    let snapshot = &trace.lowered_ir().snapshots[descriptor.snapshot.index()];
    let state = snapshot.frames[0]
        .registers
        .iter()
        .enumerate()
        .map(|(register, entry)| {
            let scalar = match entry {
                RegisterEntry::Value(value) => descriptor.scalar(&area, *value).unwrap(),
                RegisterEntry::Const(scalar) => *scalar,
                RegisterEntry::Unchanged => registers[register],
                _ => unreachable!("no roots or virtual tables here"),
            };
            format!("{scalar:?}")
        })
        .collect();
    Run {
        guard: descriptor.origin,
        state,
        iterations: 100_000 - control.fuel,
    }
}

fn threads(count: usize) -> Vec<Value> {
    let mut runtime = Runtime::default();
    (0..count)
        .map(|_| Value::Thread(runtime.create_thread().unwrap()))
        .collect()
}

#[test]
fn a_loop_over_an_array_of_coroutines_calls_the_helper_with_each_lane() {
    let _serial = ONE_AT_A_TIME.lock().unwrap();
    install_helper(
        HelperId::coroutine_resume(1, 0),
        stand_in as *const () as usize,
    );
    for tier in TIERS {
        SEEN.lock().unwrap().clear();
        MISS_AT.store(0, Ordering::SeqCst);
        let mut array = threads(3);
        let run = run(&mut array, 1, 3, tier);
        // The loop ends through its branch, after three resumes.
        assert_eq!(run.guard, Some(2), "{tier:?}");
        let seen = SEEN.lock().unwrap().clone();
        assert_eq!(seen.len(), 3, "{tier:?}");
        for (position, (tag, lane, count, words)) in seen.into_iter().enumerate() {
            // The thread is a tagged lane, the lanes follow the bank's order,
            // the call gave two arguments and the helper may rewrite words.
            assert_eq!(tag, TRACE_TAG_THREAD, "{tier:?}");
            assert_eq!(lane, i64::try_from(position).unwrap() + 1, "{tier:?}");
            assert_eq!(count, 2, "{tier:?}");
            assert!(words, "{tier:?}");
        }
        // sum = 2 + 4 + 6, and the registers hold the last thread's lane.
        assert_eq!(run.state[3], "Integer(12)", "{tier:?}");
        assert_eq!(run.state[4], "Thread(3)", "{tier:?}");
        assert_eq!(run.iterations, 3 - 1, "{tier:?}");
    }
}

#[test]
fn a_thread_slot_holding_another_type_leaves_before_the_read() {
    let _serial = ONE_AT_A_TIME.lock().unwrap();
    install_helper(
        HelperId::coroutine_resume(1, 0),
        stand_in as *const () as usize,
    );
    for tier in TIERS {
        SEEN.lock().unwrap().clear();
        MISS_AT.store(0, Ordering::SeqCst);
        let mut array = threads(3);
        array[1] = Value::Integer(42);
        let run = run(&mut array, 1, 3, tier);
        // The array read of slot 2 is the first guard: the first thread ran,
        // the number is no thread.
        assert_eq!(run.guard, Some(0), "{tier:?}");
        assert_eq!(SEEN.lock().unwrap().len(), 1, "{tier:?}");
        assert_eq!(run.state[0], "Integer(2)", "{tier:?}");
        assert_eq!(run.state[3], "Integer(2)", "{tier:?}");
    }
}

#[test]
fn a_helper_that_misses_leaves_through_the_guard_of_its_call() {
    let _serial = ONE_AT_A_TIME.lock().unwrap();
    install_helper(
        HelperId::coroutine_resume(1, 0),
        stand_in as *const () as usize,
    );
    for tier in TIERS {
        SEEN.lock().unwrap().clear();
        MISS_AT.store(2, Ordering::SeqCst);
        let mut array = threads(3);
        let run = run(&mut array, 1, 3, tier);
        // The second call missed: the guard before it restores the state at
        // its call (index 2, thread 2) with the first resume's effect.
        assert_eq!(run.guard, Some(1), "{tier:?}");
        assert_eq!(SEEN.lock().unwrap().len(), 2, "{tier:?}");
        assert_eq!(run.state[0], "Integer(2)", "{tier:?}");
        assert_eq!(run.state[3], "Integer(2)", "{tier:?}");
        assert_eq!(run.state[4], "Thread(2)", "{tier:?}");
    }
    MISS_AT.store(0, Ordering::SeqCst);
}
