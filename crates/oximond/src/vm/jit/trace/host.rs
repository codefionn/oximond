//! Library calls a running trace makes through the VM
//! ([`oximond::jit_ir::trace::TraceOp::HostCall`]), and the string library
//! code the helpers of [`super::helpers`] share.
//!
//! Only a call that needs the whole VM is a host call
//! ([`HostFunction`]). The calls that make strings (`string.char`, `tostring`,
//! `string.format`, `..`, `string.find`, `table.concat`) are helpers: native
//! code calls them directly and they use the functions here on the run's
//! bank.
//!
//! Native code stores the call's arguments as entry tags and bits and asks
//! [`RunTables`](super::native::RunTables) to answer. The answer runs the same
//! code the interpreter runs for the call: the callback of a native function,
//! the runtime's formatter, the runtime's concatenation. Anything unusual
//! (an error, a `Redispatch` that sends the call to the VM, an argument the
//! call cannot take, a metamethod that would run, a run out of room for
//! strings) answers `false`, and the guard leaves the trace before the
//! instruction, so the interpreter makes the call and raises whatever it
//! raises.
//!
//! The functions here are pure. They allocate strings, which the run's root
//! bank keeps alive until the run ends, and they charge the collector the
//! way the interpreter's call does, but they run no Lua code and collect
//! nothing: the collector runs at the interpreter's next safepoint, after
//! the trace has left. The room a run has for strings is bounded in lanes and
//! in bytes ([`MAX_HOST_BANK_LANES`], [`MAX_HOST_STRING_BYTES`]), so a loop
//! that builds strings leaves the trace every so many iterations and the
//! strings it built go back to the interpreter.

use crate::jit::{
    TRACE_TAG_BOOLEAN, TRACE_TAG_BORROWED_STRING, TRACE_TAG_FLOAT, TRACE_TAG_INTEGER,
    TRACE_TAG_NIL, TRACE_TAG_STRING, TRACE_TAG_TABLE, TraceInputSlot,
};
use crate::jit_ir::trace::{HostFunction, MAX_HOST_ARGS, ROOT_STRING_LANE_BASE};
use crate::runtime::{
    DirectNativeKind, NativeFunctionHandle, NativePlan, NumberFormat, Runtime, StackBuffer, Value,
    format_integer,
};
use crate::stdlib::{PreparedFormat, substring_range};
use crate::string::LuaString;
use crate::vm::builtins::next_string_conversion;
use std::mem::ManuallyDrop;
use std::sync::Arc;

pub(in crate::vm::jit::trace) mod pattern;
pub(in crate::vm::jit::trace) mod text;

/// Most lanes a run's bank may hold when a host call adds a string to it.
pub(crate) const MAX_HOST_BANK_LANES: usize = crate::jit::MAX_TRACE_STRING_LANES;

/// Most bytes of strings the host calls of one run may build.
pub(crate) const MAX_HOST_STRING_BYTES: usize = 1 << 20;

/// What the host calls of one run remember.
///
/// The strings a run makes take the lowest vacant lanes from `floor` up, so
/// the lanes made since the last release are the occupied ones in
/// `floor..top`. A release empties them (but for the few it keeps) and puts
/// `top` back at `floor`: it costs a step per string made, and nothing for the
/// bank at large.
#[derive(Debug, Default)]
pub(crate) struct HostState {
    /// Bytes of the strings built so far, less the bytes of those released.
    bytes: usize,
    /// The first lane a string built by a host call took, once there is one.
    floor: Option<usize>,
    /// The lane after the last one a string took since the last release.
    top: usize,
    /// The lanes of the strings an earlier release kept.
    carried: [u32; MAX_HOST_ARGS],
    carried_len: usize,
    /// Whether the string metatable has no `__tostring`, once looked at. No
    /// Lua code runs during a run, so the answer holds for all of it.
    plain_strings: Option<bool>,
    /// The lanes of the tables the run made since the last release
    /// ([`oximond::jit_ir::trace::HelperId::TABLE_NEW`]). Nothing names one
    /// after the loop-back, so a release drops them all.
    pub(super) made: Vec<u32>,
    /// The lanes a release emptied, which the next table the run makes takes
    /// before it grows the bank.
    pub(super) vacated: Vec<u32>,
    /// The shapes the tables of the run were made with, by the words of the
    /// keys that name them: a shape is sorted and hashed when it is built.
    /// The trace keeps them from one run to the next.
    pub(super) shapes: Vec<([i64; 10], Arc<crate::runtime::TableShape>)>,
}

/// What a table the run made weighs towards the next release, as a string of
/// a few hundred bytes does.
pub(super) const MADE_TABLE_WEIGHT: u32 = 32;

/// The value a string or table slot names: a lane of the bank, a root string,
/// or a borrowed string. `None` for any other slot.
#[allow(unsafe_code)]
pub(super) fn lane_value<'a>(
    slot: TraceInputSlot,
    bank: &'a [Value],
    roots: &'a [Value],
) -> Option<&'a Value> {
    let bits = slot.bits();
    match slot.tag() {
        TRACE_TAG_STRING => {
            let lane = bits.cast_unsigned();
            match lane.checked_sub(ROOT_STRING_LANE_BASE) {
                Some(slot) => roots.get(usize::try_from(slot).ok()?),
                None => bank.get(usize::try_from(lane).ok()?),
            }
        }
        TRACE_TAG_TABLE => bank.get(usize::try_from(bits).ok()?),
        // SAFETY: a borrowed string is the address of a slot of an array
        // part a bound table holds, which the run has not moved: the IR
        // verifier ends every borrow at the first operation that could
        // mutate, allocate, collect or re-enter (`TraceOp::ends_borrows`),
        // and a host call only reads the slot.
        TRACE_TAG_BORROWED_STRING => Some(unsafe { &*(bits as usize as *const Value) }),
        _ => None,
    }
}

/// The string an argument slot stands for: a lane of the bank, a root
/// string, or a borrowed string.
#[inline(always)]
#[allow(unsafe_code)]
pub(super) fn text<'a>(
    slot: TraceInputSlot,
    bank: &'a [Value],
    roots: &'a [Value],
) -> Option<&'a LuaString> {
    let bits = slot.bits();
    let held: &Value = if slot.tag() == TRACE_TAG_STRING {
        let lane = bits.cast_unsigned();
        if lane < ROOT_STRING_LANE_BASE {
            bank.get(lane as usize)?
        } else {
            roots.get((lane - ROOT_STRING_LANE_BASE) as usize)?
        }
    } else if slot.tag() == TRACE_TAG_BORROWED_STRING {
        // SAFETY: see `lane_value`.
        unsafe { &*(bits as usize as *const Value) }
    } else {
        return None;
    };
    match held {
        Value::String(text) => Some(text),
        _ => None,
    }
}

/// Runs `body` over the values the argument slots stand for, as a slice, and
/// gives back what it returns; `None` for a slot no host call takes.
///
/// A string or table of the bank or the roots is a bitwise copy of the value
/// held there, never dropped and never moved out: the callbacks the slice
/// goes to only borrow it, and the original outlives the call. That spares a
/// counted reference per argument and per call. A borrowed string points into
/// a table the callback reaches through the runtime, so it gets a reference of
/// its own.
///
/// Always inlined: the slot count of the table helpers is a constant, and
/// with it the copy loop and the reference bookkeeping fold away. Left to
/// the heuristics, an unrelated change elsewhere in the crate moved six
/// helper instantiations out of line, which cost `table.insert` on a trace
/// about 69 instructions per call (+6.2% on `table/insert_remove`). Callers
/// with a large body take [`with_values_apart`] instead.
#[inline(always)]
#[allow(unsafe_code)]
pub(super) fn with_values<R>(
    args: &[TraceInputSlot],
    bank: &[Value],
    roots: &[Value],
    body: impl FnOnce(&[Value]) -> R,
) -> Option<R> {
    // `string.pack` takes its format and up to five values, one more than
    // any other host call.
    const WIDEST: usize = MAX_HOST_ARGS + 1;
    if args.len() > WIDEST {
        return None;
    }
    let mut values: [ManuallyDrop<Value>; WIDEST] =
        [const { ManuallyDrop::new(Value::Nil) }; WIDEST];
    // The references a borrowed string takes, held until the body is done.
    // One is the common case and takes no allocation.
    let mut owned_first: Option<Value> = None;
    let mut owned_more: Vec<Value> = Vec::new();
    for (position, slot) in args.iter().enumerate() {
        let bits = slot.bits();
        values[position] = match slot.tag() {
            TRACE_TAG_NIL => ManuallyDrop::new(Value::Nil),
            TRACE_TAG_BOOLEAN => ManuallyDrop::new(Value::Boolean(bits != 0)),
            TRACE_TAG_INTEGER => ManuallyDrop::new(Value::Integer(bits)),
            TRACE_TAG_FLOAT => {
                ManuallyDrop::new(Value::Number(f64::from_bits(bits.cast_unsigned())))
            }
            TRACE_TAG_STRING | TRACE_TAG_TABLE => {
                let held = lane_value(*slot, bank, roots)?;
                if !matches!(held, Value::String(_) | Value::Table(_)) {
                    return None;
                }
                // SAFETY: a bitwise copy that is never dropped (see above).
                // It goes into the slot as one block: a read of the value
                // as a field-wise copy loads bytes it has just stored
                // across their edges, which the core cannot forward.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        std::ptr::from_ref(held),
                        (&raw mut values[position]).cast::<Value>(),
                        1,
                    );
                }
                continue;
            }
            TRACE_TAG_BORROWED_STRING => {
                let held = Value::String(text(*slot, bank, roots)?.clone());
                // SAFETY: a bitwise copy of the clone `owned` keeps alive.
                let copy = ManuallyDrop::new(unsafe { std::ptr::read(&raw const held) });
                if owned_first.is_none() {
                    owned_first = Some(held);
                } else {
                    owned_more.push(held);
                }
                copy
            }
            _ => return None,
        };
    }
    // SAFETY: `ManuallyDrop<Value>` has the layout of `Value`, and the first
    // `args.len()` elements are initialized.
    let slice = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<Value>(), args.len()) };
    Some(body(slice))
}

/// [`with_values`] out of line, for the host calls and pattern helpers whose
/// own body is large: inlined there it cost `string/pattern_run` 2.1% and
/// `stdlib/tonumber_parse` 1.3%.
#[inline(never)]
pub(super) fn with_values_apart<R>(
    args: &[TraceInputSlot],
    bank: &[Value],
    roots: &[Value],
    body: impl FnOnce(&[Value]) -> R,
) -> Option<R> {
    with_values(args, bank, roots, body)
}

/// Runs the callback of `handle` the way the interpreter's native call does
/// and takes its first result. `None` for an error, a `Redispatch`, or a
/// native the runtime alone cannot run.
#[inline(always)]
pub(super) fn call_plan(
    runtime: &mut Runtime,
    handle: NativeFunctionHandle,
    args: &[Value],
) -> Option<Value> {
    match runtime.native_plan(handle) {
        NativePlan::Scalar(callback)
        | NativePlan::Unary(callback, _)
        | NativePlan::Binary(callback, _) => callback(runtime, args).ok(),
        NativePlan::Paired(callback, paired) => match paired(runtime, args) {
            Ok(Some(mut pair)) => Some(pair.take(0)),
            Ok(None) => callback(runtime, args)
                .ok()
                .map(|values| values.into_iter().next().unwrap_or(Value::Nil)),
            Err(_) => None,
        },
        NativePlan::Multiple(callback) => callback(runtime, args)
            .ok()
            .map(|values| values.into_iter().next().unwrap_or(Value::Nil)),
        NativePlan::Other => None,
    }
}

/// Whether the string metatable has no `__tostring`.
fn strings_are_plain(runtime: &Runtime, state: &mut HostState) -> bool {
    *state.plain_strings.get_or_insert_with(|| {
        match runtime.raw_metatable(&Value::String(LuaString::from(&b""[..]))) {
            Ok(None) => true,
            Ok(Some(metatable)) => runtime
                .table_get_string_ref(metatable, &LuaString::from(&b"__tostring"[..]))
                .is_none(),
            Err(_) => false,
        }
    })
}

/// `tostring` of a number, as text. `None` when the number type has a
/// metatable.
fn number_text(runtime: &Runtime, value: &Value) -> Option<LuaString> {
    if runtime.number_metatable().is_some() {
        return None;
    }
    match value {
        Value::Number(number) => {
            // The cached text of a small integral float, as the interpreter's
            // fast path finds it.
            if number.to_bits() < 256.0_f64.to_bits()
                && !runtime.number_format().appends_float_suffix()
            {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "below 256"
                )]
                let digits = *number as u8;
                #[allow(clippy::float_cmp, reason = "an exact round trip tells an integer")]
                if f64::from(digits) == *number
                    && let Some(text) = runtime.small_integer_text(u64::from(digits))
                {
                    return Some(text);
                }
            }
            Some(runtime.number_format().format_string(*number))
        }
        Value::Integer(integer) => match u64::try_from(*integer)
            .ok()
            .and_then(|small| runtime.small_integer_text(small))
        {
            Some(text) => Some(text),
            None => runtime.format_value(value).ok(),
        },
        _ => None,
    }
}

/// The modern `string.format` for a call that needs no `__tostring`: every
/// `%s` takes a string, and the runtime's own formatter answers.
pub(super) fn format(
    runtime: &mut Runtime,
    state: &mut HostState,
    args: &[Value],
) -> Option<Value> {
    let handle = runtime.direct_native(DirectNativeKind::Format)?;
    let Some(Value::String(format)) = args.first() else {
        return None;
    };
    let mut format_index = 0;
    let mut argument_index = 1;
    while let Some((argument, next_format, next_argument)) =
        next_string_conversion(format.as_bytes(), format_index, argument_index)
    {
        if !matches!(args.get(argument), Some(Value::String(_)))
            || !strings_are_plain(runtime, state)
        {
            return None;
        }
        format_index = next_format;
        argument_index = next_argument;
    }
    call_plan(runtime, handle, args)
}

/// Adds the string `result` to the bank as a new lane, within the room a run
/// has for strings.
#[inline]
pub(in crate::vm::jit::trace) fn publish_string(
    bank: &mut Vec<Value>,
    state: &mut HostState,
    lengths: &mut [u32],
    result: LuaString,
    out: &mut TraceInputSlot,
) -> bool {
    publish_weighted(bank, state, lengths, result, out, 0)
}

/// [`publish_string`] for a string that weighs `shared` less than a new one.
#[inline(always)]
#[allow(unsafe_code)]
fn publish_weighted(
    bank: &mut Vec<Value>,
    state: &mut HostState,
    lengths: &mut [u32],
    result: LuaString,
    out: &mut TraceInputSlot,
    shared: u32,
) -> bool {
    // A shared text (the runtime's cached digits) is no new allocation, so
    // it weighs little. It is charged its bytes all the same, and a release
    // gives back what a string was charged, whatever the string is.
    let real = result.len();
    let weight = if shared == 0 {
        u32::try_from(real / 16)
            .unwrap_or(u32::MAX)
            .saturating_add(32)
    } else {
        shared
    };
    let Some(lane) = claim_lane(bank, state, lengths, real, weight) else {
        return false;
    };
    if lane == bank.len() {
        bank.push(Value::String(result));
    } else {
        let held = &mut bank[lane];
        debug_assert!(matches!(held, Value::Nil), "claimed lane holds {held:?}");
        // A plain assignment builds the string on the stack, drops the old
        // value and copies the 16 bytes over: the copy reads a tag byte and
        // a pointer just stored as one block, which the core cannot forward
        // (about 20 cycles a string). Writing the lane in place avoids it.
        if matches!(held, Value::Nil) {
            // SAFETY: the lane holds `Nil`, which owns nothing, so writing
            // over it drops nothing.
            unsafe { std::ptr::write(held, Value::String(result)) };
        } else {
            // `claim_lane` returns only lanes that hold `Nil`. Should that
            // break, drop what the lane holds rather than leak it.
            *held = Value::String(result);
        }
    }
    if let Some(slot) = lengths.get_mut(lane) {
        *slot = u32::try_from(real).unwrap_or(u32::MAX);
    }
    *out = TraceInputSlot::string(lane as u64);
    true
}

/// Takes the lowest vacant lane of the bank from the run's floor up for a
/// value of `real` bytes that weighs `weight` until the next release, or
/// `None` when the run is out of room. The caller stores the value.
///
/// A returned lane either is one past the end of the bank or holds
/// `Value::Nil`: the loop below skips every lane that holds anything else.
#[inline]
fn claim_lane(
    bank: &[Value],
    state: &mut HostState,
    lengths: &mut [u32],
    real: usize,
    weight: u32,
) -> Option<usize> {
    let mut lane = if state.floor.is_some() {
        state.top
    } else {
        state.floor = Some(bank.len());
        bank.len()
    };
    // The lanes a release kept, and the tables the run admitted, stay.
    while bank
        .get(lane)
        .is_some_and(|held| !matches!(held, Value::Nil))
    {
        lane += 1;
    }
    if lane >= MAX_HOST_BANK_LANES || state.bytes + real > MAX_HOST_STRING_BYTES {
        return None;
    }
    state.bytes += real;
    // What the value costs to keep until a release: its bytes and the
    // allocation that holds them (a shared text costs next to nothing).
    if let Some(pending) = lengths.get_mut(MAX_HOST_BANK_LANES) {
        *pending = pending.saturating_add(weight);
    }
    state.top = lane + 1;
    Some(lane)
}

/// Whether the run has room for strings of the byte counts `sizes`, one lane
/// each, published one after the other: what [`claim_lane`] would find for
/// each, without taking anything. A caller that publishes several strings
/// asks first, so that no publication is left behind by a miss.
pub(super) fn strings_fit(bank: &[Value], state: &HostState, sizes: &[usize]) -> bool {
    let mut lane = if state.floor.is_some() {
        state.top
    } else {
        bank.len()
    };
    let mut bytes = state.bytes;
    for real in sizes {
        while bank
            .get(lane)
            .is_some_and(|held| !matches!(held, Value::Nil))
        {
            lane += 1;
        }
        if lane >= MAX_HOST_BANK_LANES || bytes + real > MAX_HOST_STRING_BYTES {
            return false;
        }
        bytes += real;
        lane += 1;
    }
    true
}

/// [`oximond::jit_ir::trace::TraceOp::ReleaseStrings`]: drops the strings and
/// iterators the run made since the last release, and the ones it kept then,
/// except the lanes in `keep`, which stay. Each string it drops gives back
/// the bytes it was charged.
#[inline(never)]
// The `release` closure the loop body calls mutates `state`, so the loop
// cannot hold an iterator borrow of `state.carried`.
#[allow(clippy::needless_range_loop)]
fn release_strings(
    bank: &mut [Value],
    state: &mut HostState,
    lengths: &mut [u32],
    keep: &[TraceInputSlot],
) {
    if let Some(pending) = lengths.get_mut(MAX_HOST_BANK_LANES) {
        *pending = 0;
    }
    let Some(floor) = state.floor else {
        return;
    };
    let limit = bank.len();
    let end = state.top.min(limit);
    let mut next = [0_u32; MAX_HOST_ARGS];
    let mut count = 0;
    let mut release = |lane: usize| {
        // A `string.gmatch` iterator takes a lane too, and goes with the
        // strings of its iteration.
        if !matches!(bank[lane], Value::String(_) | Value::GmatchIterator(_)) {
            return;
        }
        if keep.iter().any(|slot| slot.bits() == lane as i64) {
            // What stays is what `keep` names among the lanes this run made.
            if count < MAX_HOST_ARGS {
                next[count] = lane as u32;
                count += 1;
            }
            return;
        }
        bank[lane] = Value::Nil;
        let charged = lengths.get_mut(lane).map_or(0, std::mem::take) as usize;
        state.bytes = state.bytes.saturating_sub(charged);
    };
    // The kept lanes above `end` are not in the range the loop below covers.
    for position in 0..state.carried_len {
        let lane = state.carried[position] as usize;
        if lane >= end && lane < limit {
            release(lane);
        }
    }
    for lane in floor..end {
        release(lane);
    }
    state.carried = next;
    state.carried_len = count;
    state.top = floor;
}

/// Writes the number `value` (an integer or a float, as the profile has
/// them) to `out`.
fn publish_number(value: &Value, out: &mut TraceInputSlot) -> bool {
    match value {
        Value::Integer(integer) => *out = TraceInputSlot::integer(*integer),
        Value::Number(number) => *out = TraceInputSlot::float(*number),
        _ => return false,
    }
    true
}

/// Writes the answer of a call to `out`: a scalar, or a string as a new lane.
#[inline(always)]
pub(super) fn publish(
    bank: &mut Vec<Value>,
    state: &mut HostState,
    lengths: &mut [u32],
    result: Value,
    out: &mut TraceInputSlot,
) -> bool {
    match result {
        Value::Nil => *out = TraceInputSlot::nil(),
        Value::Boolean(flag) => *out = TraceInputSlot::boolean(flag),
        Value::Integer(_) | Value::Number(_) => return publish_number(&result, out),
        Value::String(text) => return publish_string(bank, state, lengths, text, out),
        _ => return false,
    }
    true
}

/// Calls `function` on the argument slots `args` and writes the answer to
/// `out`. A string answer is a new lane of `bank`.
#[inline]
pub(crate) fn host_call(
    runtime: &mut Runtime,
    bank: &mut Vec<Value>,
    roots: &[Value],
    state: &mut HostState,
    function: HostFunction,
    args: &[TraceInputSlot],
    out: &mut TraceInputSlot,
    lengths: &mut [u32],
) -> bool {
    if function == HostFunction::ReleaseStrings {
        release_strings(bank, state, lengths, args);
        return true;
    }
    if args.is_empty() || args.len() > MAX_HOST_ARGS {
        return false;
    }
    match function {
        HostFunction::ReleaseStrings => false,
        HostFunction::MathMax | HostFunction::MathMin => {
            math_extreme(function == HostFunction::MathMax, args, out)
        }
        HostFunction::StringLength => string_length(runtime, bank, roots, args, out),
        HostFunction::ConcatLength => concat_length(runtime, bank, roots, args, out),
        HostFunction::Native(index) => {
            let handle = NativeFunctionHandle::from_index(index);
            let Some(result) = with_values_apart(args, bank, roots, |values| {
                call_plan(runtime, handle, values)
            })
            .flatten() else {
                return false;
            };
            publish(bank, state, lengths, result, out)
        }
    }
}

/// The modern `string.format` over the argument slots `args`.
#[inline(never)]
pub(super) fn format_call(
    runtime: &mut Runtime,
    bank: &mut Vec<Value>,
    roots: &[Value],
    (state, lengths): (&mut HostState, &mut [u32]),
    args: &[TraceInputSlot],
    out: &mut TraceInputSlot,
) -> bool {
    let Some(result) =
        with_values(args, bank, roots, |values| format(runtime, state, values)).flatten()
    else {
        return false;
    };
    publish(bank, state, lengths, result, out)
}

/// `string.format` with a constant format, prepared as `plan`. `native` is
/// the native of the Lua 5.1 and `LuaJIT` forms, and `None` for the modern
/// one. The prepared form formats straight into the reused buffer; whatever
/// it leaves, including a call that would raise, goes to the generic path of
/// the same helper, which then answers or misses as it does at level 0.
#[inline(never)]
pub(super) fn format_prepared_call(
    runtime: &mut Runtime,
    bank: &mut Vec<Value>,
    roots: &[Value],
    (state, lengths): (&mut HostState, &mut [u32]),
    plan: &PreparedFormat,
    native: Option<DirectNativeKind>,
    args: &[TraceInputSlot],
    out: &mut TraceInputSlot,
    used: &mut bool,
) -> bool {
    if !plan.generic_only()
        && let Some(result) = with_values(args, bank, roots, |values| {
            format_prepared(runtime, state, plan, native.is_none(), values)
        })
        .flatten()
    {
        *used = true;
        return publish(bank, state, lengths, result, out);
    }
    match native {
        None => format_call(runtime, bank, roots, (state, lengths), args, out),
        Some(kind) => through_native(runtime, bank, roots, (state, lengths), kind, args, out),
    }
}

/// The prepared form of [`format_prepared_call`], or `None` where the
/// generic path has to answer.
#[inline(always)]
fn format_prepared(
    runtime: &Runtime,
    state: &mut HostState,
    plan: &PreparedFormat,
    modern: bool,
    values: &[Value],
) -> Option<Value> {
    let numbers = runtime.number_format();
    if modern {
        // As `format`: a `%s` takes a string, and no `__tostring` is set.
        for position in plan.string_arguments() {
            if !matches!(values.get(position), Some(Value::String(_)))
                || !strings_are_plain(runtime, state)
            {
                return None;
            }
        }
    } else if !matches!(numbers, NumberFormat::Lua51 | NumberFormat::LuaJit) {
        return None;
    }
    plan.format(values, modern, numbers, runtime.nonfinite_numerals())
}

/// `math.max` or `math.min` of two integers.
#[inline(never)]
fn math_extreme(max: bool, args: &[TraceInputSlot], out: &mut TraceInputSlot) -> bool {
    let [left, right] = args else {
        return false;
    };
    if left.tag() != TRACE_TAG_INTEGER || right.tag() != TRACE_TAG_INTEGER {
        return false;
    }
    let (left, right) = (left.bits(), right.bits());
    *out = TraceInputSlot::integer(if max {
        left.max(right)
    } else {
        left.min(right)
    });
    true
}

/// `string.char` of integral numbers in `0..=255`.
#[inline(never)]
pub(super) fn string_char(
    bank: &mut Vec<Value>,
    state: &mut HostState,
    lengths: &mut [u32],
    args: &[TraceInputSlot],
    out: &mut TraceInputSlot,
) -> bool {
    let mut bytes = [0_u8; MAX_HOST_ARGS];
    for (byte, slot) in bytes.iter_mut().zip(args) {
        *byte = match slot.tag() {
            TRACE_TAG_INTEGER => match u8::try_from(slot.bits()) {
                Ok(byte) => byte,
                Err(_) => return false,
            },
            TRACE_TAG_FLOAT => {
                let number = f64::from_bits(slot.bits().cast_unsigned());
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "the round trip below rejects every inexact value"
                )]
                let byte = number as u8;
                #[allow(clippy::float_cmp, reason = "an exact round trip is the test")]
                if f64::from(byte) != number {
                    return false;
                }
                byte
            }
            _ => return false,
        };
    }
    publish_string(
        bank,
        state,
        lengths,
        LuaString::from(&bytes[..args.len()]),
        out,
    )
}

/// The byte count of a string.
#[inline(never)]
fn string_length(
    runtime: &Runtime,
    bank: &[Value],
    roots: &[Value],
    args: &[TraceInputSlot],
    out: &mut TraceInputSlot,
) -> bool {
    let [slot] = args else {
        return false;
    };
    let Some(text) = text(*slot, bank, roots) else {
        return false;
    };
    let Ok(length) = i64::try_from(text.len()) else {
        return false;
    };
    publish_number(&runtime.integer_value(length), out)
}

/// The byte count of `args[0] .. args[1] .. ...` (strings and numbers) as a
/// number of the profile, with no string made: the sum of the strings' byte
/// counts and of the digits the profile prints for each number
/// ([`NumberFormat::format`] and `format_integer`, which the interpreter's
/// concatenation uses). It misses where [`concat`] does, for an operand that
/// is neither or a result too long for a string. The collector is charged
/// what the string would have cost, as the interpreter's concatenation
/// charges it.
#[inline(never)]
fn concat_length(
    runtime: &mut Runtime,
    bank: &[Value],
    roots: &[Value],
    args: &[TraceInputSlot],
    out: &mut TraceInputSlot,
) -> bool {
    if args.len() < 2 {
        return false;
    }
    let format = runtime.number_format();
    let mut total = 0_usize;
    for slot in args {
        let part = match slot.tag() {
            TRACE_TAG_INTEGER => format_integer(slot.bits()).as_bytes().len(),
            TRACE_TAG_FLOAT => format
                .format(f64::from_bits(slot.bits().cast_unsigned()))
                .as_bytes()
                .len(),
            TRACE_TAG_STRING | TRACE_TAG_BORROWED_STRING => match text(*slot, bank, roots) {
                Some(text) => text.len(),
                None => return false,
            },
            _ => return false,
        };
        total = match total.checked_add(part) {
            Some(total) if u32::try_from(total).is_ok() => total,
            _ => return false,
        };
    }
    let Ok(length) = i64::try_from(total) else {
        return false;
    };
    if total != 0 {
        runtime.gc_charge(LuaString::allocation_size_of_length(total));
    }
    publish_number(&runtime.integer_value(length), out)
}

/// `tostring` of a number or a string.
#[inline(never)]
pub(super) fn to_string(
    runtime: &Runtime,
    bank: &mut Vec<Value>,
    roots: &[Value],
    (state, lengths): (&mut HostState, &mut [u32]),
    args: &[TraceInputSlot],
    out: &mut TraceInputSlot,
) -> bool {
    let [slot] = args else {
        return false;
    };
    match slot.tag() {
        TRACE_TAG_INTEGER | TRACE_TAG_FLOAT => {
            let value = if slot.tag() == TRACE_TAG_INTEGER {
                Value::Integer(slot.bits())
            } else {
                Value::Number(f64::from_bits(slot.bits().cast_unsigned()))
            };
            match number_text(runtime, &value) {
                Some(result) => publish_string(bank, state, lengths, result, out),
                None => false,
            }
        }
        TRACE_TAG_BOOLEAN | TRACE_TAG_NIL => {
            let value = if slot.tag() == TRACE_TAG_NIL {
                Value::Nil
            } else {
                Value::Boolean(slot.bits() != 0)
            };
            // A metatable for the type could give the value a `__tostring`.
            if !matches!(runtime.raw_metatable(&value), Ok(None)) {
                return false;
            }
            let spelled: &[u8] = match value {
                Value::Nil => b"nil",
                Value::Boolean(true) => b"true",
                _ => b"false",
            };
            publish_string(bank, state, lengths, LuaString::from(spelled), out)
        }
        TRACE_TAG_STRING | TRACE_TAG_BORROWED_STRING => {
            let Some(held) = text(*slot, bank, roots) else {
                return false;
            };
            if !strings_are_plain(runtime, state) {
                return false;
            }
            let held = held.clone();
            publish_string(bank, state, lengths, held, out)
        }
        TRACE_TAG_TABLE => {
            // The trace guarded that the metatable has no `__tostring`; a
            // change since misses and the interpreter replays the call.
            let Some(held @ Value::Table(_)) = lane_value(*slot, bank, roots) else {
                return false;
            };
            let key = LuaString::from("__tostring");
            match runtime.raw_metatable(held) {
                Ok(None) => {}
                Ok(Some(metatable)) if runtime.table_get_string_ref(metatable, &key).is_none() => {}
                _ => return false,
            }
            match runtime.format_value(held) {
                Ok(result) => publish_string(bank, state, lengths, result, out),
                Err(_) => false,
            }
        }
        _ => false,
    }
}

/// `..` over strings and numbers. With `valid_text` it misses when the result
/// holds bytes that are no UTF-8.
#[inline(never)]
pub(super) fn concat(
    runtime: &mut Runtime,
    bank: &mut Vec<Value>,
    roots: &[Value],
    (state, lengths): (&mut HostState, &mut [u32]),
    args: &[TraceInputSlot],
    out: &mut TraceInputSlot,
    valid_text: bool,
) -> bool {
    // The common case: short operands, strings and numbers, read from their
    // slots and assembled in one pass on the stack, as the runtime's own
    // concatenation does, with no `Value` made for any of them. Anything
    // else (a long result, a slot of another kind) takes the general path.
    if args.len() >= 2 {
        let format = runtime.number_format();
        let mut buffer = StackBuffer::<64>::new();
        let fits = args.iter().all(|slot| match slot.tag() {
            TRACE_TAG_INTEGER => buffer.push_integer(slot.bits()),
            TRACE_TAG_FLOAT => {
                buffer.push_number(format, f64::from_bits(slot.bits().cast_unsigned()))
            }
            TRACE_TAG_STRING | TRACE_TAG_BORROWED_STRING => {
                text(*slot, bank, roots).is_some_and(|text| buffer.push_bytes(text.as_bytes()))
            }
            _ => false,
        });
        if fits {
            if valid_text && std::str::from_utf8(buffer.as_bytes()).is_err() {
                return false;
            }
            let result = buffer.into_string();
            if !result.is_empty() {
                runtime.gc_charge(result.allocation_size());
            }
            return publish(bank, state, lengths, Value::String(result), out);
        }
    }
    let Some(result) = with_values(args, bank, roots, |values| {
        if values.len() < 2
            || !values.iter().all(|value| {
                matches!(
                    value,
                    Value::String(_) | Value::Integer(_) | Value::Number(_)
                )
            })
        {
            return None;
        }
        runtime.concatenate_many(values).ok()
    })
    .flatten() else {
        return false;
    };
    if valid_text
        && !matches!(&result, Value::String(text) if std::str::from_utf8(text.as_bytes()).is_ok())
    {
        return false;
    }
    publish(bank, state, lengths, result, out)
}

/// The answer of the native `kind` through its own callback.
#[inline(never)]
pub(super) fn through_native(
    runtime: &mut Runtime,
    bank: &mut Vec<Value>,
    roots: &[Value],
    (state, lengths): (&mut HostState, &mut [u32]),
    kind: DirectNativeKind,
    args: &[TraceInputSlot],
    out: &mut TraceInputSlot,
) -> bool {
    let Some(handle) = runtime.direct_native(kind) else {
        return false;
    };
    let Some(result) = with_values(args, bank, roots, |values| {
        call_plan(runtime, handle, values)
    })
    .flatten() else {
        return false;
    };
    publish(bank, state, lengths, result, out)
}

/// `tonumber (text [, base])`. A string with no base or an integer base is
/// converted in place, by the rules the installed native follows
/// ([`crate::stdlib::tonumber_of_text`]): the answer is a number or
/// nil, so there is no argument list to build and no lane to claim. Every
/// other call (a number, a base that is a string or out of range) goes to
/// the native.
pub(super) fn tonumber(
    runtime: &mut Runtime,
    bank: &mut Vec<Value>,
    roots: &[Value],
    kept: (&mut HostState, &mut [u32]),
    args: &[TraceInputSlot],
    out: &mut TraceInputSlot,
) -> bool {
    if runtime.direct_native(DirectNativeKind::Tonumber).is_some()
        && let Some(source) = args.first().and_then(|slot| text(*slot, bank, roots))
        && let Ok(base) = index_at(args, 1)
        && let Some(answer) = crate::stdlib::tonumber_of_text(
            runtime,
            source.as_bytes(),
            base.map(|base| base as i64),
        )
    {
        return match answer {
            Value::Nil => {
                *out = TraceInputSlot::nil();
                true
            }
            answer => publish_number(&answer, out),
        };
    }
    through_native(
        runtime,
        bank,
        roots,
        kept,
        DirectNativeKind::Tonumber,
        args,
        out,
    )
}

/// `string.rep`. The native builds the whole string before the run can refuse
/// it, so a result that cannot fit the run's room is a miss before it runs
/// (the interpreter then repeats the string, or raises its error).
pub(super) fn string_rep(
    runtime: &mut Runtime,
    bank: &mut Vec<Value>,
    roots: &[Value],
    kept: (&mut HostState, &mut [u32]),
    args: &[TraceInputSlot],
    out: &mut TraceInputSlot,
) -> bool {
    let (Some(source), Some(count)) = (
        args.first().and_then(|slot| text(*slot, bank, roots)),
        args.get(1),
    ) else {
        return false;
    };
    let times = match count.tag() {
        TRACE_TAG_INTEGER => count.bits(),
        TRACE_TAG_FLOAT => {
            let number = f64::from_bits(count.bits().cast_unsigned());
            #[allow(clippy::cast_possible_truncation, clippy::float_cmp)]
            if !(number.is_finite() && number.fract() == 0.0 && number.abs() < 9.0e15) {
                return false;
            }
            #[allow(clippy::cast_possible_truncation)]
            {
                number as i64
            }
        }
        _ => return false,
    };
    let separator = match args.get(2) {
        None => 0,
        Some(slot) if slot.tag() == TRACE_TAG_NIL => 0,
        Some(slot) => match text(*slot, bank, roots) {
            Some(separator) => separator.len(),
            None => return false,
        },
    };
    if times >= 1 {
        let fits = usize::try_from(times)
            .ok()
            .filter(|times| *times <= MAX_HOST_STRING_BYTES)
            .and_then(|times| source.len().checked_add(separator)?.checked_mul(times))
            .is_some_and(|total| total <= MAX_HOST_STRING_BYTES);
        if !fits {
            return false;
        }
    }
    through_native(
        runtime,
        bank,
        roots,
        kept,
        DirectNativeKind::StringRep,
        args,
        out,
    )
}

/// A string index argument the run holds: `Ok(None)` for one left out or nil,
/// `Ok(Some(_))` for an integer or a float that is exactly a small integer
/// (what every profile converts the same way), `Err` for any other, which the
/// native itself must judge (it may coerce a string or raise an error).
#[inline(always)]
// exact comparison is required here (NaN semantics)
#[allow(clippy::float_cmp)]
fn index_at(args: &[TraceInputSlot], at: usize) -> Result<Option<isize>, ()> {
    let Some(slot) = args.get(at) else {
        return Ok(None);
    };
    match slot.tag() {
        TRACE_TAG_NIL => Ok(None),
        TRACE_TAG_INTEGER => Ok(Some(isize::try_from(slot.bits()).unwrap_or(
            if slot.bits() < 0 {
                isize::MIN
            } else {
                isize::MAX
            },
        ))),
        TRACE_TAG_FLOAT => {
            let number = f64::from_bits(slot.bits().cast_unsigned());
            #[allow(clippy::cast_possible_truncation)]
            let small = number as i32;
            if f64::from(small) == number {
                Ok(Some(small as isize))
            } else {
                Err(())
            }
        }
        _ => Err(()),
    }
}

/// The range of `string.byte(text, i, j)` as `start..end` byte offsets,
/// `None` for a call the native must judge itself.
#[inline(always)]
fn byte_range(source: &LuaString, args: &[TraceInputSlot]) -> Option<(usize, usize)> {
    let length = isize::try_from(source.len()).ok()?;
    let raw_start = index_at(args, 1).ok()?.unwrap_or(1);
    let raw_end = index_at(args, 2).ok()?.unwrap_or(raw_start);
    Some(match substring_range(length, raw_start, raw_end) {
        Some(range) => (range.start, range.end),
        None => (0, 0),
    })
}

/// `string.sub` of a string the run holds and indices it holds, with no call
/// plan: the slice is copied once into a new string.
/// `None` for anything the native must judge, before anything is written.
#[inline]
pub(super) fn string_sub(
    bank: &mut Vec<Value>,
    roots: &[Value],
    (state, lengths): (&mut HostState, &mut [u32]),
    args: &[TraceInputSlot],
    out: &mut TraceInputSlot,
) -> Option<bool> {
    let source = text(*args.first()?, bank, roots)?;
    let length = isize::try_from(source.len()).ok()?;
    let start = index_at(args, 1).ok()??;
    // A nil end is a bad argument to the native, not a default.
    let end = match args.get(2) {
        Some(_) => index_at(args, 2).ok()??,
        None => -1,
    };
    let result = match substring_range(length, start, end) {
        Some(range) => LuaString::from(&source.as_bytes()[range]),
        None => LuaString::from(&[][..]),
    };
    Some(publish_string(bank, state, lengths, result, out))
}

/// `string.byte`: the bytes the native gives, which must be as many as
/// `bytes` holds. Writes them as integers.
pub(super) fn string_byte(
    runtime: &mut Runtime,
    bank: &[Value],
    roots: &[Value],
    args: &[TraceInputSlot],
    bytes: &mut [i64],
) -> bool {
    if let Some(source) = args.first().and_then(|slot| text(*slot, bank, roots))
        && let Some((start, end)) = byte_range(source, args)
    {
        if end - start != bytes.len() {
            return false;
        }
        for (slot, byte) in bytes.iter_mut().zip(&source.as_bytes()[start..end]) {
            *slot = i64::from(*byte);
        }
        return true;
    }
    let Some(handle) = runtime.direct_native(DirectNativeKind::StringByte) else {
        return false;
    };
    let NativePlan::Multiple(callback) = runtime.native_plan(handle) else {
        return false;
    };
    let Some(Some(values)) =
        with_values(args, bank, roots, |values| callback(runtime, values).ok())
    else {
        return false;
    };
    if values.len() != bytes.len() {
        return false;
    }
    for (slot, value) in bytes.iter_mut().zip(&values) {
        *slot = match value {
            Value::Integer(byte) => *byte,
            #[allow(clippy::cast_possible_truncation)]
            Value::Number(byte) => *byte as i64,
            _ => return false,
        };
    }
    true
}

/// `string.find`. Plain text (the flag set, or a pattern with no special
/// character) is searched here; anything else is the native's own business.
#[allow(clippy::too_many_lines)]
#[inline(never)]
pub(super) fn find(
    runtime: &mut Runtime,
    bank: &mut Vec<Value>,
    roots: &[Value],
    kept: (&mut HostState, &mut [u32]),
    args: &[TraceInputSlot],
    out: &mut TraceInputSlot,
) -> bool {
    'direct: {
        let count = args.len();
        if !(2..=4).contains(&count) {
            break 'direct;
        }
        let (Some(haystack), Some(pattern)) =
            (text(args[0], bank, roots), text(args[1], bank, roots))
        else {
            break 'direct;
        };
        let mut initial: isize = 1;
        if count >= 3 {
            let slot = args[2];
            match slot.tag() {
                TRACE_TAG_INTEGER => {
                    let Ok(small) = i32::try_from(slot.bits()) else {
                        break 'direct;
                    };
                    initial = small as isize;
                }
                TRACE_TAG_FLOAT => {
                    let number = f64::from_bits(slot.bits().cast_unsigned());
                    #[allow(clippy::cast_possible_truncation, reason = "the round trip rejects")]
                    let small = number as i32;
                    #[allow(clippy::float_cmp, reason = "an exact round trip is the test")]
                    if f64::from(small) != number {
                        break 'direct;
                    }
                    initial = small as isize;
                }
                TRACE_TAG_NIL => {}
                _ => break 'direct,
            }
        }
        let mut plain = false;
        if count == 4 {
            let slot = args[3];
            match slot.tag() {
                TRACE_TAG_BOOLEAN => plain = slot.bits() != 0,
                TRACE_TAG_NIL => {}
                _ => break 'direct,
            }
        }
        let needle = pattern.as_bytes();
        if (!plain && needle.iter().any(|byte| b"^$*+?.([%-".contains(byte)))
            || runtime.number_format() == NumberFormat::LuaJit
        {
            break 'direct;
        }
        let source = haystack.as_bytes();
        let length = source.len() as isize;
        if initial < 0 {
            initial = length + initial + 1;
        }
        if initial > length + 1 {
            // Lua 5.1 starts a search past the end at the end; 5.3 and later
            // fail it.
            if !runtime.string_init_clamps() {
                *out = TraceInputSlot::nil();
                return true;
            }
            initial = length + 1;
        }
        let start = (initial.max(1) - 1) as usize;
        let found = if needle.is_empty() {
            Some(start)
        } else {
            let tail = &source[start..];
            let rest = &needle[1..];
            memchr::memchr_iter(needle[0], tail)
                .find(|&position| tail[position + 1..].starts_with(rest))
                .map(|position| start + position)
        };
        return match found {
            None => {
                *out = TraceInputSlot::nil();
                true
            }
            Some(position) => publish_number(&runtime.integer_value(position as i64 + 1), out),
        };
    }
    through_native(
        runtime,
        bank,
        roots,
        kept,
        DirectNativeKind::Find,
        args,
        out,
    )
}

/// `table.concat` of a list of strings; anything else goes to the native.
#[inline(never)]
pub(super) fn table_concat(
    runtime: &mut Runtime,
    bank: &mut Vec<Value>,
    roots: &[Value],
    kept: (&mut HostState, &mut [u32]),
    args: &[TraceInputSlot],
    out: &mut TraceInputSlot,
) -> bool {
    let built = (|| {
        if args.len() > 2 {
            return None;
        }
        let Some(Value::Table(handle)) = lane_value(args[0], bank, roots) else {
            return None;
        };
        let separator: &[u8] = match args.get(1) {
            None => &[],
            Some(slot) => text(*slot, bank, roots)?.as_bytes(),
        };
        // A table with a metatable may need its metamethods.
        if !matches!(runtime.raw_metatable(&Value::Table(*handle)), Ok(None)) {
            return None;
        }
        let values = runtime.table_border_array(*handle).ok()??;
        if values.is_empty() {
            return None;
        }
        let mut length = separator.len() * (values.len() - 1);
        for value in values {
            let Value::String(text) = value else {
                return None;
            };
            length += text.len();
        }
        // A result that fits the inline form is gathered on the stack, and
        // its string made from it at once.
        if length <= 22 {
            let mut buffer = StackBuffer::<24>::new();
            for (position, value) in values.iter().enumerate() {
                if position != 0 {
                    buffer.push_bytes(separator);
                }
                if let Value::String(text) = value {
                    buffer.push_bytes(text.as_bytes());
                }
            }
            return Some(buffer.into_string());
        }
        let mut result = Vec::with_capacity(length);
        for (position, value) in values.iter().enumerate() {
            if position != 0 {
                result.extend_from_slice(separator);
            }
            if let Value::String(text) = value {
                result.extend_from_slice(text.as_bytes());
            }
        }
        Some(LuaString::from(result))
    })();
    match built {
        Some(result) => publish_string(bank, kept.0, kept.1, result, out),
        None => through_native(
            runtime,
            bank,
            roots,
            kept,
            DirectNativeKind::TableConcat,
            args,
            out,
        ),
    }
}

/// `tostring` of an integer or an integral float below 256, which the runtime
/// keeps the text of. Returns `false`, having changed nothing, for any other
/// argument.
#[inline]
pub(crate) fn tostring_small(
    runtime: &Runtime,
    bank: &mut Vec<Value>,
    state: &mut HostState,
    lengths: &mut [u32],
    slot: TraceInputSlot,
    out: &mut TraceInputSlot,
) -> bool {
    let digit = if slot.tag() == TRACE_TAG_INTEGER {
        match u8::try_from(slot.bits()) {
            Ok(digit) => digit,
            Err(_) => return false,
        }
    } else if slot.tag() == TRACE_TAG_FLOAT {
        let number = f64::from_bits(slot.bits().cast_unsigned());
        if number.to_bits() >= 256.0_f64.to_bits() || runtime.number_format().appends_float_suffix()
        {
            return false;
        }
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "below 256"
        )]
        let digit = number as u8;
        #[allow(clippy::float_cmp, reason = "an exact round trip tells an integer")]
        if f64::from(digit) != number {
            return false;
        }
        digit
    } else {
        return false;
    };
    if runtime.number_metatable().is_some() {
        return false;
    }
    match runtime.small_integer_text(u64::from(digit)) {
        Some(text) => publish_weighted(bank, state, lengths, text, out, 4),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Run {
        bank: Vec<Value>,
        state: HostState,
        lengths: Vec<u32>,
    }

    impl Run {
        fn new() -> Self {
            Self {
                // A lane the run started with.
                bank: vec![Value::Boolean(true)],
                state: HostState::default(),
                lengths: vec![0; MAX_HOST_BANK_LANES + 1],
            }
        }

        fn publish(&mut self, text: &[u8], shared: u32) -> usize {
            let mut out = TraceInputSlot::nil();
            assert!(publish_weighted(
                &mut self.bank,
                &mut self.state,
                &mut self.lengths,
                LuaString::from(text),
                &mut out,
                shared,
            ));
            usize::try_from(out.bits()).unwrap()
        }

        fn release(&mut self, keep: &[usize]) {
            let keep: Vec<_> = keep
                .iter()
                .map(|lane| TraceInputSlot::string(*lane as u64))
                .collect();
            release_strings(&mut self.bank, &mut self.state, &mut self.lengths, &keep);
        }
    }

    #[test]
    fn a_release_gives_back_what_shared_texts_and_new_strings_were_charged() {
        let mut run = Run::new();
        let big = vec![b'x'; 5000];
        let lanes = [
            run.publish(b"42", 4),
            run.publish(&big, 0),
            run.publish(b"7", 4),
        ];
        assert_eq!(run.state.bytes, 2 + 5000 + 1);
        run.release(&[]);
        assert_eq!(run.state.bytes, 0);
        for lane in lanes {
            assert!(matches!(run.bank[lane], Value::Nil));
        }
        // The lane the run started with is not the run's to drop.
        assert!(matches!(run.bank[0], Value::Boolean(true)));
        // A second round starts from zero and takes the same lanes again.
        assert_eq!(run.publish(b"42", 4), lanes[0]);
        assert_eq!(run.state.bytes, 2);
        run.release(&[]);
        assert_eq!(run.state.bytes, 0);
    }

    #[test]
    fn a_kept_string_keeps_its_charge_and_its_lane() {
        let mut run = Run::new();
        let kept = run.publish(b"keep", 0);
        let dropped = run.publish(b"short lived", 0);
        run.release(&[kept]);
        assert_eq!(run.state.bytes, 4);
        assert!(matches!(run.bank[kept], Value::String(_)));
        assert!(matches!(run.bank[dropped], Value::Nil));
        // New strings fill the emptied lane and never the kept one.
        assert_eq!(run.publish(b"next", 0), dropped);
        assert_eq!(run.publish(b"after", 0), dropped + 1);
        // Left out of the next release's keep, the string goes with the rest.
        run.release(&[]);
        assert_eq!(run.state.bytes, 0);
        assert!(matches!(run.bank[kept], Value::Nil));
    }

    #[test]
    fn a_loop_that_shares_texts_never_exhausts_the_budget() {
        let mut run = Run::new();
        for _ in 0..10 * MAX_HOST_BANK_LANES {
            for _ in 0..64 {
                run.publish(b"123", 4);
            }
            run.release(&[]);
            assert_eq!(run.state.bytes, 0);
        }
        assert!(run.bank.len() <= 65, "{}", run.bank.len());
    }
}
