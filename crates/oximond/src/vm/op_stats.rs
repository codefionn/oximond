//! VM-side helpers for [`oximond::runtime::op_stats`], compiled only with the
//! `op-stats` feature.
//!
//! The interpreter calls these from `#[cfg(feature = "op-stats")]` statements
//! so the ordinary build carries none of their code. They only read values
//! and frame records: no metamethod, no guest allocation, no interning.
//!
//! Register liveness comes from [`observable_registers`], whose entry `pc`
//! is the set of registers observable when execution resumes at `pc`
//! (before instruction `pc` runs). It keeps named locals in scope and
//! captured registers observable. The analysis is cached per chunk on this
//! thread, keyed by the chunk's address and holding a [`Weak`] reference, so
//! the address cannot be reused by another chunk while the entry exists.

use crate::bytecode::{Chunk, Instruction, OpCode};
use crate::jit_ir::{RegisterSet, observable_registers, register_set_contains};
pub(crate) use crate::runtime::op_stats::{
    Counter, add, bump, fast_break_pairs, fast_dispatch, general_dispatch,
};
use crate::runtime::op_stats::{OperandClass, SiteChange, SiteTable, operand_class, site};
use crate::runtime::{ClosureHandle, Runtime, TableHandle, Value};
use crate::vm::Frame;
use crate::vm::stack::CallStack;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Weak};

/// Call-site flag bits passed to [`site`] for [`SiteTable::Call`].
pub(crate) const CALL_FLAG_VARARG_CALLEE: u64 = 1 << 0;
pub(crate) const CALL_FLAG_CAPTURED_CALLER: u64 = 1 << 1;
pub(crate) const CALL_FLAG_CALLER_DEAD_ARGUMENT: u64 = 1 << 2;
pub(crate) const CALL_FLAG_STRING_ARGUMENT: u64 = 1 << 3;

struct LivenessCache {
    entries: FxHashMap<usize, (Weak<Chunk>, Rc<[RegisterSet]>)>,
    /// Entry count at which dead chunks are dropped next.
    prune_at: usize,
}

thread_local! {
    static LIVENESS: RefCell<LivenessCache> = RefCell::new(LivenessCache {
        entries: FxHashMap::default(),
        prune_at: 1024,
    });
}

/// The observable registers of `chunk` at every pc, computed once per chunk.
fn liveness(chunk: &Arc<Chunk>) -> Rc<[RegisterSet]> {
    let key = Arc::as_ptr(chunk) as usize;
    LIVENESS.with(|cache| {
        let mut cache = cache.borrow_mut();
        // The entry's `Weak` keeps the allocation, so a hit is this chunk.
        if let Some((_, live)) = cache.entries.get(&key) {
            return Rc::clone(live);
        }
        if cache.entries.len() >= cache.prune_at {
            cache
                .entries
                .retain(|_, (weak, _)| weak.strong_count() != 0);
            cache.prune_at = (cache.entries.len() * 2).max(1024);
        }
        let live: Rc<[RegisterSet]> = observable_registers(chunk).into();
        cache
            .entries
            .insert(key, (Arc::downgrade(chunk), Rc::clone(&live)));
        live
    })
}

fn observable(live: &[RegisterSet], pc: usize, register: usize) -> bool {
    match (live.get(pc), u8::try_from(register)) {
        (Some(set), Ok(register)) => register_set_contains(set, register),
        // Past the end of the chunk, or a register no set describes.
        _ => true,
    }
}

/// What a register slot holds, for retention counts.
#[derive(Clone, Copy, Eq, PartialEq)]
enum SlotClass {
    Nil,
    /// Numbers, booleans, 64-bit cdata and the uncollected function
    /// handles (native and builtin functions).
    Scalar,
    /// A string: its slot holds an `Arc` reference, not a GC root.
    String,
    /// A collectable object the slot keeps alive.
    Reference,
}

fn slot_class(value: &Value) -> SlotClass {
    match value {
        Value::Nil => SlotClass::Nil,
        Value::Boolean(_)
        | Value::Integer(_)
        | Value::Number(_)
        | Value::Int64(_)
        | Value::UInt64(_)
        | Value::NativeFunction(_)
        | Value::BuiltinFunction(_) => SlotClass::Scalar,
        Value::String(_) => SlotClass::String,
        Value::Table(_)
        | Value::File(_)
        | Value::Userdata(_)
        | Value::GmatchIterator(_)
        | Value::Thread(_)
        | Value::CoroutineWrapper(_)
        | Value::Closure(_) => SlotClass::Reference,
    }
}

fn operand_class_of(left: &Value, right: &Value) -> OperandClass {
    match (left, right) {
        (Value::Number(_), Value::Number(_)) => OperandClass::FloatFloat,
        (Value::Integer(_), Value::Integer(_)) => OperandClass::IntegerInteger,
        (Value::Number(_) | Value::Integer(_), Value::Number(_) | Value::Integer(_)) => {
            OperandClass::MixedNumber
        }
        _ => OperandClass::Other,
    }
}

/// Counts a fast-loop arithmetic or comparison `opcode` at `pc` of `chunk`
/// with operands `left` and `right`.
pub(crate) fn arithmetic(opcode: u8, chunk: &Chunk, pc: usize, left: &Value, right: &Value) {
    let class = operand_class_of(left, right);
    operand_class(opcode, class);
    site(
        SiteTable::Arithmetic,
        std::ptr::from_ref(chunk) as usize,
        pc,
        class as u64,
        0,
        0,
    );
}

/// [`arithmetic`] for the fast loop's out-of-line operator arm, whose
/// unary `Negate` has one operand.
pub(crate) fn slow_operator(
    instruction: Instruction,
    chunk: &Chunk,
    pc: usize,
    registers: &[Value],
) {
    let left = &registers[usize::from(instruction.b())];
    let right = if instruction.opcode() == Ok(OpCode::Negate) {
        left
    } else {
        &registers[usize::from(instruction.c())]
    };
    arithmetic(instruction.raw().to_le_bytes()[0], chunk, pc, left, right);
}

/// Counts the key class of a fast-loop `GetTable` or `SetTable`.
pub(crate) fn table_key(key: &Value, integer: Counter, float: Counter) {
    match key {
        Value::Integer(_) => bump(integer),
        Value::Number(_) => bump(float),
        _ => {}
    }
}

/// Records a fast-loop `GetField` or `SetField` on `table` at `pc`.
pub(crate) fn field(runtime: &Runtime, chunk: &Chunk, pc: usize, table: TableHandle) {
    site(
        SiteTable::Field,
        std::ptr::from_ref(chunk) as usize,
        pc,
        runtime.table_shape_identity(table),
        table.native_identity(),
        0,
    );
}

/// Counts the slots a fast-loop `ClearRange` is about to clear, by what
/// they hold.
pub(crate) fn clear_range(range: &[Value]) {
    bump(Counter::ClearRanges);
    add(Counter::ClearRangeSlots, range.len() as u64);
    for value in range {
        bump(match slot_class(value) {
            SlotClass::Nil => Counter::ClearRangeSlotsNil,
            SlotClass::Scalar => Counter::ClearRangeSlotsScalar,
            SlotClass::String => Counter::ClearRangeSlotsString,
            SlotClass::Reference => Counter::ClearRangeSlotsReference,
        });
    }
}

/// Records the fixed call `instruction` at `pc` of `caller`, which calls
/// the Lua closure `closure` with arguments in `registers`.
/// `captured_caller` says the calling frame has captured locals.
pub(crate) fn closure_call(
    runtime: &Runtime,
    caller: &Arc<Chunk>,
    pc: usize,
    instruction: Instruction,
    registers: &[Value],
    captured_caller: bool,
    closure: ClosureHandle,
) {
    let Some(parts) = runtime.closure_call_parts(closure) else {
        return;
    };
    let (start, count) = if instruction.opcode() == Ok(OpCode::Call) {
        (
            usize::from(instruction.a()) + 1,
            usize::from(instruction.b()),
        )
    } else {
        (usize::from(instruction.c()), 1)
    };
    let mut flags = 0;
    bump(Counter::CallSiteClosureCalls);
    add(Counter::CallSiteArguments, count as u64);
    if parts.chunk.is_vararg {
        bump(Counter::CallSiteVarargCallee);
        flags |= CALL_FLAG_VARARG_CALLEE;
    }
    if captured_caller {
        bump(Counter::CallSiteCapturedCaller);
        flags |= CALL_FLAG_CAPTURED_CALLER;
    }
    if let Some(arguments) = registers.get(start..start + count) {
        let live = liveness(caller);
        for (offset, value) in arguments.iter().enumerate() {
            let string = matches!(value, Value::String(_));
            if string {
                flags |= CALL_FLAG_STRING_ARGUMENT;
            }
            // The caller resumes after the call, at `pc + 1`.
            if !observable(&live, pc + 1, start + offset) {
                flags |= CALL_FLAG_CALLER_DEAD_ARGUMENT;
                bump(Counter::CallSiteCallerDeadArguments);
                if string {
                    bump(Counter::CallSiteCallerDeadStringArguments);
                }
            }
        }
    }
    let change = site(
        SiteTable::Call,
        Arc::as_ptr(caller) as usize,
        pc,
        Arc::as_ptr(parts.chunk) as usize as u64,
        closure.native_identity(),
        flags,
    );
    match change {
        SiteChange::Same => bump(Counter::CallSiteSameClosure),
        SiteChange::OnlyB => bump(Counter::CallSiteSamePrototypeOtherClosure),
        SiteChange::A => bump(Counter::CallSiteOtherPrototype),
        SiteChange::First => {}
    }
}

/// Counts arguments an activation copies from caller stack slots.
pub(crate) fn stack_arguments(arguments: &[Value]) {
    add(
        Counter::ActivationStackArgumentSlots,
        arguments.len() as u64,
    );
    add(
        Counter::ActivationStringArguments,
        arguments
            .iter()
            .filter(|value| matches!(value, Value::String(_)))
            .count() as u64,
    );
}

/// Counts one argument value an activation moves into its window.
pub(crate) fn moved_argument(value: &Value) {
    if matches!(value, Value::String(_)) {
        bump(Counter::ActivationStringArguments);
    }
}

/// Counts one pushed activation that filled `copied` parameter slots from
/// arguments and set `cleared` other window slots to nil.
pub(crate) fn activation(copied: usize, cleared: usize) {
    bump(Counter::Activations);
    add(Counter::ActivationArgumentSlots, copied as u64);
    add(Counter::ActivationClearedSlots, cleared as u64);
}

/// Counts the rooted slots of the running stack `frames` for one
/// collection, and the frame slots its frames can no longer observe.
pub(crate) fn gc_roots(frames: &CallStack) {
    bump(Counter::GcRootScans);
    let values = frames.values();
    add(Counter::GcStackSlots, values.len() as u64);
    for frame in frames {
        gc_frame(frame, values);
    }
}

/// Slot counts of one frame window at a collection.
#[derive(Debug, Default, Eq, PartialEq)]
struct FrameSlots {
    slots: u64,
    references: u64,
    dead_references: u64,
    strings: u64,
    dead_strings: u64,
    unknown_pc: bool,
}

fn gc_frame(frame: &Frame, values: &[Value]) {
    let counts = frame_slots(frame, values);
    add(Counter::GcFrameSlots, counts.slots);
    add(Counter::GcFrameReferenceSlots, counts.references);
    add(Counter::GcFrameDeadReferenceSlots, counts.dead_references);
    add(Counter::GcFrameStringSlots, counts.strings);
    add(Counter::GcFrameDeadStringSlots, counts.dead_strings);
    if counts.unknown_pc {
        bump(Counter::GcFramesUnknownPc);
    }
}

fn frame_slots(frame: &Frame, values: &[Value]) -> FrameSlots {
    let mut counts = FrameSlots::default();
    let Some(window) = values.get(frame.base as usize..frame.top as usize) else {
        return counts;
    };
    counts.slots = window.len() as u64;
    let live = liveness(&frame.chunk.as_arc());
    // The fast loop and the general loop leave `pc` differently, so a
    // register is live if it is observable at either point.
    let len = live.len();
    let known = frame.pc < len && frame.last_interpreted_pc.get().is_none_or(|pc| pc < len);
    counts.unknown_pc = !known;
    for (register, value) in window.iter().enumerate() {
        let (slots, dead_slots) = match slot_class(value) {
            SlotClass::Reference => (&mut counts.references, &mut counts.dead_references),
            SlotClass::String => (&mut counts.strings, &mut counts.dead_strings),
            SlotClass::Nil | SlotClass::Scalar => continue,
        };
        *slots += 1;
        let dead = known
            && !observable(&live, frame.pc, register)
            && frame
                .last_interpreted_pc
                .get()
                .is_none_or(|pc| !observable(&live, pc, register));
        if dead {
            *dead_slots += 1;
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::Constant;
    use crate::compat::{CompatibilityProfile, ExecutionMode};
    use crate::runtime::op_stats::get;
    use crate::vm::Vm;

    // The counters are process-wide and other tests run Lua in parallel, so
    // these only assert lower bounds on deltas, or on pure helpers.

    #[test]
    fn frame_slots_union_both_resume_points() {
        // r1 and r3 are read at pc 0 and 1; r0 and r2 never are.
        let chunk = Arc::new(Chunk::new(
            5,
            Vec::new(),
            vec![
                Instruction::move_(4, 1),
                Instruction::move_(4, 3),
                Instruction::return_(4),
            ],
        ));
        let table = Value::Table(TableHandle::from_native_identity(1));
        let frame = Frame::new(
            chunk,
            Vec::new(),
            Vec::new(),
            None,
            None,
            TableHandle::from_native_identity(2),
            None,
        );
        let mut stack = CallStack::from(vec![frame]);
        {
            let registers = stack.at_mut(0).registers;
            registers[0] = table.clone();
            registers[1] = Value::string("live");
            registers[2] = Value::string("dead");
            registers[3] = table;
        }
        let expected = FrameSlots {
            slots: 5,
            references: 2,
            dead_references: 1,
            strings: 2,
            dead_strings: 1,
            unknown_pc: false,
        };
        assert_eq!(frame_slots(&stack[0], stack.values()), expected);
        // At pc 1 alone r1 is dead, but the last executed pc still reads it.
        stack[0].pc = 1;
        stack[0].last_interpreted_pc = crate::vm::frame::LastPc::some(0);
        assert_eq!(frame_slots(&stack[0], stack.values()), expected);
        stack[0].last_interpreted_pc = crate::vm::frame::LastPc::NONE;
        assert_eq!(
            frame_slots(&stack[0], stack.values()),
            FrameSlots {
                dead_strings: 2,
                ..expected
            }
        );
        stack[0].pc = 99;
        assert_eq!(
            frame_slots(&stack[0], stack.values()),
            FrameSlots {
                dead_references: 0,
                dead_strings: 0,
                unknown_pc: true,
                ..expected
            }
        );
    }

    #[test]
    fn fast_loop_counts_calls_activations_and_clears() {
        const CALLS: u64 = 200;
        // for i = 1, CALLS do local t = i; sum = sum + f(t); clear t end
        // The global read keeps the callee off the inline plans, so every
        // call pushes an activation.
        let identity = Chunk::with_parts(
            2,
            1,
            vec![Constant::String("unset".into())],
            vec![Instruction::get_global(1, 0), Instruction::return_(0)],
            Vec::new(),
            Vec::new(),
        );
        let caller = Chunk::with_parts(
            9,
            0,
            vec![
                Constant::Number(1.0),
                Constant::Number(CALLS as f64),
                Constant::Number(0.0),
            ],
            vec![
                Instruction::load_constant(1, 0),
                Instruction::load_constant(2, 1),
                Instruction::load_constant(3, 0),
                Instruction::load_constant(7, 2),
                Instruction::closure(6, 0),
                Instruction::numeric_for_prep(1, 4),
                Instruction::move_(5, 4),
                Instruction::call_one(8, 6, 5),
                Instruction::binary(OpCode::Add, 7, 7, 8),
                Instruction::clear_range(5, 1),
                Instruction::numeric_for_loop(1, -5),
                Instruction::return_(7),
            ],
            vec![identity],
            Vec::new(),
        );
        let watched = [
            Counter::CallSiteClosureCalls,
            Counter::CallSiteArguments,
            Counter::CallSiteCallerDeadArguments,
            Counter::CallSiteSameClosure,
            Counter::Activations,
            Counter::ActivationArgumentSlots,
            Counter::ActivationStackArgumentSlots,
            Counter::ClearRanges,
            Counter::ClearRangeSlots,
            Counter::ClearRangeSlotsScalar,
            Counter::ClosuresCreated,
            Counter::FastDispatches,
        ];
        let before = watched.map(get);
        let mut vm = Vm::with_profile_and_execution_mode(
            CompatibilityProfile::Lua54,
            ExecutionMode::Interpreter,
        );
        let result = vm.execute(&caller).unwrap();
        #[allow(clippy::cast_precision_loss)]
        let sum = (CALLS * (CALLS + 1) / 2) as f64;
        assert_eq!(result, vec![Value::Number(sum)]);
        let delta = |counter: Counter| {
            get(counter) - before[watched.iter().position(|c| *c == counter).unwrap()]
        };
        for counter in [
            Counter::CallSiteClosureCalls,
            Counter::CallSiteArguments,
            Counter::CallSiteCallerDeadArguments,
            Counter::Activations,
            Counter::ActivationArgumentSlots,
            Counter::ClearRanges,
            Counter::ClearRangeSlots,
            Counter::ClearRangeSlotsScalar,
        ] {
            assert!(
                delta(counter) >= CALLS,
                "{}: {}",
                counter.name(),
                delta(counter)
            );
        }
        // The first call classifies the callee on the general path, which
        // passes its argument by value.
        assert!(delta(Counter::ActivationStackArgumentSlots) >= CALLS - 1);
        assert!(delta(Counter::CallSiteSameClosure) >= CALLS - 1);
        assert!(delta(Counter::ClosuresCreated) >= 1);
        assert!(delta(Counter::FastDispatches) >= 5 * CALLS);
    }
}
