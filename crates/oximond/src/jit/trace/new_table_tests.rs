//! The helpers that make and fill a table ([`HelperId::TABLE_NEW`] and
//! [`HelperId::TABLE_INIT`]), called from compiled code.
//!
//! The VM implements the two; here stand-ins count their calls and keep the
//! arguments they got, so the tests see the calling convention (the leading
//! context, the count of arguments given, the entry tag and bits of each
//! value) and what a miss does: the trace leaves through the call's guard
//! and the calls after it do not run.

use super::*;
use crate::jit::OptimizationTier;
use crate::jit_ir::trace::{
    ArithmeticOp, FrameId, HelperId, LoopParamInit, ROOT_STRING_LANE_BASE, RegisterEntry, Snapshot,
    SnapshotFrame, SnapshotPhase, TraceBuilder, TraceFrameInfo, TraceId, TraceIr, TraceOp,
    TraceScalar, TraceSite, TraceTerminator, TraceType,
};
use crate::runtime::Value;

const TIERS: [OptimizationTier; 2] = [OptimizationTier::Baseline, OptimizationTier::Optimized];

fn site(pc: usize) -> TraceSite {
    TraceSite {
        frame: FrameId(0),
        pc,
    }
}

/// What the stand-ins saw.
#[derive(Default)]
pub(super) struct Calls {
    pub(super) made: u64,
    pub(super) filled: u64,
    /// The calls of the stand-in for a store by another helper.
    pub(super) set: u64,
    /// The call that misses, counting from one, or zero for none.
    pub(super) miss_on: u64,
    /// The arguments of the last `TABLE_NEW`: the count, the array length and
    /// the first key's tag and bits.
    pub(super) last_new: (i64, i64, i64, i64),
    /// The arguments of the last `TABLE_INIT`: the table lane, then the key
    /// and the value as tags and bits.
    pub(super) last_init: [i64; 5],
    /// The lanes the stand-in handed out.
    pub(super) lanes: Vec<i64>,
    /// Whether the stand-in gives the value slots of the tables it makes, as
    /// the VM does for a table with a shape, and the slots it gave.
    pub(super) give_slots: bool,
    // Boxing pins each table's slot array: its address escapes to the trace
    // before the push, so the Vec must never move the arrays it collects.
    #[allow(clippy::vec_box)]
    pub(super) slots: Vec<Box<[Value; 3]>>,
}

#[allow(unsafe_code, clippy::too_many_arguments)]
extern "C" fn new_table(
    context: usize,
    out: *mut i64,
    count: i64,
    array_len: i64,
    tag0: i64,
    bits0: i64,
    _tag1: i64,
    _bits1: i64,
    _tag2: i64,
    _bits2: i64,
    _tag3: i64,
    _bits3: i64,
    _tag4: i64,
    _bits4: i64,
) -> i64 {
    // SAFETY: the context is the `Calls` the test keeps alive for the run.
    let calls = unsafe { &mut *(context as *mut Calls) };
    calls.made += 1;
    if calls.made == calls.miss_on {
        return 1;
    }
    calls.last_new = (count, array_len, tag0, bits0);
    let lane = 100 + i64::try_from(calls.made).unwrap();
    calls.lanes.push(lane);
    // SAFETY: `out` is the control block's result words.
    unsafe {
        *out = lane;
        if calls.give_slots {
            let mut slots = Box::new([Value::Nil, Value::Nil, Value::Nil]);
            *out.add(1) = slots.as_mut_ptr() as i64;
            calls.slots.push(slots);
        }
    }
    0
}

#[allow(unsafe_code)]
extern "C" fn fill_table(
    context: usize,
    _out: *mut i64,
    table: i64,
    key_tag: i64,
    key_bits: i64,
    value_tag: i64,
    value_bits: i64,
) -> i64 {
    // SAFETY: as in `new_table`.
    let calls = unsafe { &mut *(context as *mut Calls) };
    calls.filled += 1;
    calls.last_init = [table, key_tag, key_bits, value_tag, value_bits];
    0
}

#[allow(unsafe_code)]
extern "C" fn set_string(
    context: usize,
    _out: *mut i64,
    _table: i64,
    _key_tag: i64,
    _key_bits: i64,
    _value_tag: i64,
    _value_bits: i64,
) -> i64 {
    // SAFETY: as in `new_table`.
    let calls = unsafe { &mut *(context as *mut Calls) };
    calls.set += 1;
    0
}

/// The VM's side of a run, as far as these traces ask: the host entry that
/// carries the stand-ins' context.
struct Host {
    calls: *mut Calls,
}

extern "C" fn unused_host_call(
    _: usize,
    _: *const TraceInputSlot,
    _: *mut TraceInputSlot,
    _: i64,
    _: i64,
) {
}

extern "C" fn unused_set(
    _: usize,
    _: *mut TraceInputSlot,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    -1
}

impl TraceTables for Host {
    fn lane(&mut self, _: &Value) -> Option<usize> {
        None
    }

    fn row(&mut self, _: usize, row: &mut [usize]) {
        row.fill(0);
    }

    fn host_entry(&mut self) -> Option<HostEntry> {
        Some(HostEntry {
            function: unused_host_call as *const () as usize,
            context: self.calls as usize,
            set: unused_set as *const () as usize,
            store: 0,
        })
    }
}

/// What the loop stores into the table it makes.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum Fill {
    /// `t[1] = 2.5`
    Integer,
    /// `t.name = 2.5`
    Name,
    /// `t.name = 2.5; t.name = 3.5`
    NameTwice,
    /// `t.name = 2.5` after a store by another helper, which may move the
    /// table's slots
    HelperThenName,
    /// `t.name = nil; t.name = 2.5`: the nil store deletes the field's entry
    /// through the helper, which may move the table's slots
    NilThenName,
    /// `t.name = 7`, an integer
    NameInteger,
    /// `t.name = true`
    NameBoolean,
    /// `{2.5, 7, true}`: the call makes the three slots of the array part
    /// and each store overwrites one
    Array,
}

/// `repeat local t = {[array 3] shaped by "name"}; <fill>; i = i + 1
/// until i > limit`, with the table kept in no register: the loop makes one
/// table and fills it each iteration.
fn fill_loop_trace(fill: Fill) -> TraceIr {
    use RegisterEntry::{Unchanged, Value as Val};
    let mut b = TraceBuilder::new(
        TraceId(0),
        TraceFrameInfo {
            register_count: 2,
            code_len: 32,
            inlined: None,
            activation: None,
        },
    );
    b.mark_written(FrameId(0), 0);
    let i_in = b.input(0, TraceType::Float);
    let limit = b.input(1, TraceType::Float);
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let before = |b: &mut TraceBuilder, pc: usize, i| {
        let snapshot = b.snapshot(Snapshot::new(
            SnapshotPhase::Before { pc },
            0,
            vec![SnapshotFrame {
                frame: FrameId(0),
                last_interpreted_pc: None,
                registers: vec![Val(i), Unchanged].into(),
            }],
        ));
        b.guard(site(pc), snapshot)
    };
    let length = b
        .emit(
            site(0),
            TraceOp::Constant(TraceScalar::Integer(3)),
            Some(TraceType::Integer),
        )
        .unwrap();
    let name = b
        .emit(
            site(0),
            TraceOp::Constant(TraceScalar::String(ROOT_STRING_LANE_BASE)),
            Some(TraceType::String),
        )
        .unwrap();
    let guard = before(&mut b, 0, i);
    let table = b
        .emit(
            site(0),
            TraceOp::CallHelper {
                helper: HelperId::TABLE_NEW,
                args: if fill == Fill::Array {
                    vec![length]
                } else {
                    vec![length, name]
                },
                effect: None,
                guard,
            },
            Some(TraceType::Table),
        )
        .unwrap();
    let key = if fill == Fill::Integer {
        b.emit(
            site(1),
            TraceOp::Constant(TraceScalar::Integer(1)),
            Some(TraceType::Integer),
        )
        .unwrap()
    } else {
        name
    };
    if fill == Fill::HelperThenName {
        let other = b
            .emit(
                site(1),
                TraceOp::Constant(TraceScalar::Float(7.5)),
                Some(TraceType::Float),
            )
            .unwrap();
        let guard = before(&mut b, 1, i);
        let effect = b.next_effect();
        b.emit(
            site(1),
            TraceOp::CallHelper {
                helper: HelperId::SET_STRING,
                args: vec![table, name, other],
                effect: Some(effect),
                guard,
            },
            None,
        );
    }
    if fill == Fill::NilThenName {
        let nothing = b
            .emit(
                site(1),
                TraceOp::Constant(TraceScalar::Nil),
                Some(TraceType::Nil),
            )
            .unwrap();
        let guard = before(&mut b, 1, i);
        b.emit(
            site(1),
            TraceOp::CallHelper {
                helper: HelperId::TABLE_INIT,
                args: vec![table, key, nothing],
                effect: None,
                guard,
            },
            None,
        );
    }
    let stores: Vec<TraceScalar> = match fill {
        Fill::NameTwice => vec![TraceScalar::Float(2.5), TraceScalar::Float(3.5)],
        Fill::NameInteger => vec![TraceScalar::Integer(7)],
        Fill::NameBoolean => vec![TraceScalar::Boolean(true)],
        Fill::Array => vec![
            TraceScalar::Float(2.5),
            TraceScalar::Integer(7),
            TraceScalar::Boolean(true),
        ],
        _ => vec![TraceScalar::Float(2.5)],
    };
    for (position, stored) in stores.iter().enumerate() {
        let value = b
            .emit(site(1), TraceOp::Constant(*stored), Some(stored.ty()))
            .unwrap();
        let key = if fill == Fill::Array {
            b.emit(
                site(1),
                TraceOp::Constant(TraceScalar::Integer(i64::try_from(position + 1).unwrap())),
                Some(TraceType::Integer),
            )
            .unwrap()
        } else {
            key
        };
        // A guard after a committed effect resumes past it.
        let guard = if fill == Fill::HelperThenName {
            let snapshot = b.snapshot(Snapshot::new(
                SnapshotPhase::After {
                    pc: 1,
                    successor: 2,
                },
                1,
                vec![SnapshotFrame {
                    frame: FrameId(0),
                    last_interpreted_pc: None,
                    registers: vec![Val(i), Unchanged].into(),
                }],
            ));
            b.guard(site(1), snapshot)
        } else {
            before(&mut b, 1, i)
        };
        b.emit(
            site(1),
            TraceOp::CallHelper {
                helper: HelperId::TABLE_INIT,
                args: vec![table, key, value],
                effect: None,
                guard,
            },
            None,
        );
    }
    let one = b
        .emit(
            site(2),
            TraceOp::Constant(TraceScalar::Float(1.0)),
            Some(TraceType::Float),
        )
        .unwrap();
    let next = b
        .emit(
            site(2),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: i,
                right: one,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let more = b
        .emit(
            site(3),
            TraceOp::Compare {
                comparison: crate::jit_ir::Comparison::LessEqual,
                left: next,
                right: limit,
            },
            Some(TraceType::Boolean),
        )
        .unwrap();
    let committed = u32::from(fill == Fill::HelperThenName);
    let after = vec![Val(next), Unchanged];
    let exit = b.snapshot(Snapshot::new(
        SnapshotPhase::After {
            pc: 3,
            successor: 4,
        },
        committed,
        vec![SnapshotFrame {
            frame: FrameId(0),
            last_interpreted_pc: None,
            registers: after.clone().into(),
        }],
    ));
    let branch = b.guard(site(3), exit);
    b.emit(
        site(3),
        TraceOp::GuardBranch {
            condition: more,
            expected: true,
            guard: branch,
        },
        None,
    );
    let poll = b.snapshot(Snapshot::new(
        SnapshotPhase::After {
            pc: 3,
            successor: 0,
        },
        committed,
        vec![SnapshotFrame {
            frame: FrameId(0),
            last_interpreted_pc: None,
            registers: after.into(),
        }],
    ));
    b.set_poll_snapshot(poll);
    let mut ir = b.finish(TraceTerminator::LoopBack { args: vec![next] });
    ir.root_constants = vec![std::sync::Arc::from(&b"name"[..])];
    ir.root_slot_count = 1;
    ir.verify().unwrap();
    ir
}

/// Runs `ir` for `iterations` iterations with the stand-ins installed.
fn run(tier: OptimizationTier, iterations: u32, miss_on: u64) -> (Calls, TraceExitClass) {
    let (calls, class, _) = run_fill(tier, Fill::Integer, iterations, (miss_on, false));
    (calls, class)
}

/// [`run`] for a loop that stores `fill`, with the stand-in that makes the
/// table giving its value slots or not (`give_slots`), also returning the
/// fields the run counted as stored in place.
pub(super) fn run_fill(
    tier: OptimizationTier,
    fill: Fill,
    iterations: u32,
    (miss_on, give_slots): (u64, bool),
) -> (Calls, TraceExitClass, i64) {
    // The addresses are process-wide: the stand-ins stay in place only while
    // one run compiles, so that a trace another test compiles meanwhile gets
    // the library's helpers (the first install of those, made by a test that
    // starts a VM, would otherwise replace the stand-ins after this run chose
    // them and call the library's helper with this run's context).
    static STAND_INS: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let ir = fill_loop_trace(fill);
    let trace = {
        let _held = STAND_INS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::vm::restore_table_helpers();
        install_helper(HelperId::TABLE_NEW, new_table as *const () as usize);
        install_helper(HelperId::TABLE_INIT, fill_table as *const () as usize);
        install_helper(HelperId::SET_STRING, set_string as *const () as usize);
        let trace = compile_trace(&ir, tier).unwrap();
        crate::vm::restore_table_helpers();
        trace
    };
    assert!(
        trace.needs_tables(),
        "{tier:?}: the helpers take the context"
    );
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| match register {
            0 => TraceInputSlot::from_scalar(TraceScalar::Float(1.0)),
            _ => TraceInputSlot::from_scalar(TraceScalar::Float(f64::from(iterations))),
        })
        .collect();
    let mut calls = Box::new(Calls {
        miss_on,
        give_slots,
        ..Calls::default()
    });
    let mut host = Host {
        calls: &raw mut *calls,
    };
    let mut area = vec![0_i64; trace.exit_area_len()];
    let mut control = TraceControl::new(1_000_000);
    let mut space = TraceTableSpace::new();
    let TraceOutcome::Exit { index } =
        trace.run_with_tables(&inputs, &[], &mut control, &mut area, &mut space, &mut host)
    else {
        panic!("{tier:?}: the trace refused entry");
    };
    let class = trace.exits()[index].class;
    (*calls, class, control.initialized)
}

#[test]
fn a_loop_makes_and_fills_a_table_each_iteration_with_the_calling_convention() {
    for tier in TIERS {
        let (calls, class) = run(tier, 5, 0);
        assert_eq!(class, TraceExitClass::LoopComplete, "{tier:?}");
        // Neither call is hoisted out of the loop: one table per iteration.
        assert_eq!((calls.made, calls.filled), (5, 5), "{tier:?}");
        // Two arguments given (the length and the key), the length first,
        // then the key as an entry tag and bits: a root string.
        assert_eq!(
            calls.last_new,
            (
                2,
                3,
                TRACE_TAG_STRING,
                i64::try_from(ROOT_STRING_LANE_BASE).unwrap()
            ),
            "{tier:?}"
        );
        // The fill gets the lane of the last table, the key one as an
        // integer and the value as the bits of a float.
        assert_eq!(
            calls.last_init,
            [
                *calls.lanes.last().unwrap(),
                TRACE_TAG_INTEGER,
                1,
                TRACE_TAG_FLOAT,
                2.5_f64.to_bits().cast_signed(),
            ],
            "{tier:?}"
        );
        // Each iteration's lane is its own.
        let mut lanes = calls.lanes.clone();
        lanes.dedup();
        assert_eq!(lanes.len(), 5, "{tier:?}");
    }
}

#[test]
fn a_miss_of_the_call_that_makes_the_table_leaves_before_the_fill() {
    for tier in TIERS {
        let (calls, class) = run(tier, 10, 3);
        assert_eq!(class, TraceExitClass::GuardFail, "{tier:?}");
        // The third call missed: two tables were made and filled, the third
        // call was made and its fill never ran.
        assert_eq!((calls.made, calls.filled), (3, 2), "{tier:?}");
    }
}

#[test]
// Exact comparison is required here (NaN semantics).
#[allow(clippy::float_cmp)]
fn a_field_of_a_table_with_slots_is_stored_in_place_without_the_helper() {
    for tier in TIERS {
        let (calls, class, initialized) = run_fill(tier, Fill::Name, 5, (0, true));
        assert_eq!(class, TraceExitClass::LoopComplete, "{tier:?}");
        // The shape names "name", so the field is the one slot of the table
        // the stand-in gave: the code wrote it, and the helper never ran.
        assert_eq!((calls.made, calls.filled), (5, 0), "{tier:?}");
        assert_eq!(initialized, 5, "{tier:?}: the run counts the fields");
        assert_eq!(calls.slots.len(), 5, "{tier:?}");
        for slots in &calls.slots {
            assert!(
                matches!(slots[0], Value::Number(stored) if stored == 2.5),
                "{tier:?}: {:?}",
                slots[0]
            );
        }
    }
}

#[test]
fn a_table_without_slots_is_filled_by_the_helper() {
    for tier in TIERS {
        let (calls, class, initialized) = run_fill(tier, Fill::Name, 5, (0, false));
        assert_eq!(class, TraceExitClass::LoopComplete, "{tier:?}");
        assert_eq!(
            (calls.made, calls.filled, initialized),
            (5, 5, 0),
            "{tier:?}"
        );
    }
}

#[test]
fn a_second_store_under_the_same_key_is_the_helpers() {
    for tier in TIERS {
        let (calls, class, initialized) = run_fill(tier, Fill::NameTwice, 4, (0, true));
        assert_eq!(class, TraceExitClass::LoopComplete, "{tier:?}");
        // The first store of each table is in place and the second the
        // helper's: a slot that was filled is no slot to overwrite blindly.
        assert_eq!(
            (calls.made, calls.filled, initialized),
            (4, 4, 4),
            "{tier:?}"
        );
        assert_eq!(
            calls.last_init[4],
            3.5_f64.to_bits().cast_signed(),
            "{tier:?}"
        );
    }
}

#[test]
fn a_key_the_trace_stores_by_integer_is_the_helpers_even_with_slots() {
    for tier in TIERS {
        let (calls, _, initialized) = run_fill(tier, Fill::Integer, 3, (0, true));
        assert_eq!(
            (calls.made, calls.filled, initialized),
            (3, 3, 0),
            "{tier:?}"
        );
    }
}

#[test]
fn a_store_by_another_helper_ends_the_in_place_fills_of_the_table() {
    for tier in TIERS {
        let (calls, class, initialized) = run_fill(tier, Fill::HelperThenName, 4, (0, true));
        assert_eq!(class, TraceExitClass::LoopComplete, "{tier:?}");
        // The other helper may have grown the table's storage, so the slots
        // address from the call that made it is stale: the fill is the
        // helper's, not a write through that address.
        assert_eq!(
            (calls.made, calls.set, calls.filled, initialized),
            (4, 4, 4, 0),
            "{tier:?}"
        );
    }
}

#[test]
fn a_nil_store_by_the_helper_ends_the_in_place_fills_of_the_table() {
    for tier in TIERS {
        let (calls, class, initialized) = run_fill(tier, Fill::NilThenName, 4, (0, true));
        assert_eq!(class, TraceExitClass::LoopComplete, "{tier:?}");
        // The nil store deletes the entry under a shape key, which moves the
        // shaped storage: both stores are the helper's, none writes through
        // the slots address from the call that made the table.
        assert_eq!(
            (calls.made, calls.filled, initialized),
            (4, 8, 0),
            "{tier:?}"
        );
    }
}

#[test]
fn an_integer_and_a_boolean_field_are_stored_in_place_with_their_tags() {
    for tier in TIERS {
        for (fill, expected) in [
            (Fill::NameInteger, Value::Integer(7)),
            (Fill::NameBoolean, Value::Boolean(true)),
        ] {
            let (calls, class, initialized) = run_fill(tier, fill, 4, (0, true));
            assert_eq!(class, TraceExitClass::LoopComplete, "{tier:?}");
            assert_eq!(
                (calls.made, calls.filled, initialized),
                (4, 0, 4),
                "{tier:?}"
            );
            for slots in &calls.slots {
                assert_eq!(slots[0], expected, "{tier:?}");
            }
        }
    }
}

#[test]
// Exact comparison is required here (NaN semantics).
#[allow(clippy::float_cmp)]
fn a_short_array_is_made_with_its_slots_and_filled_in_place() {
    for tier in TIERS {
        let (calls, class, initialized) = run_fill(tier, Fill::Array, 4, (0, true));
        assert_eq!(class, TraceExitClass::LoopComplete, "{tier:?}");
        // The call asked for the prefilled array (the flag is the ninth bit of
        // the count) and the three stores are the code's.
        assert_eq!(calls.last_new.0, 1 | 1 << 8, "{tier:?}");
        assert_eq!(
            (calls.made, calls.filled, initialized),
            (4, 0, 12),
            "{tier:?}"
        );
        for slots in &calls.slots {
            assert!(matches!(slots[0], Value::Number(stored) if stored == 2.5));
            assert_eq!(slots[1], Value::Integer(7), "{tier:?}");
            assert_eq!(slots[2], Value::Boolean(true), "{tier:?}");
        }
    }
}
