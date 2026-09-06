//! `utf8.*` and `string.pack`, `string.unpack` and `string.packsize`: the
//! host side of their helpers.
//!
//! The helpers run the native's own code through its results probe (see
//! [`oximond::runtime::NativeResultsProbe`]), on the profile's own native, so
//! they agree with the interpreter on every profile, limit and error. A call
//! that raises an error has no answer and misses before anything is written.
//! The shape of the answer is part of the helper: one that finds the answer
//! to be of another shape (another number of results, a failure where there
//! was a count) misses, and the guard of the call takes a side trace for the
//! way the call now ends.

use super::{HostState, publish_string, text, with_values_apart};
use crate::jit::{
    TRACE_TAG_BOOLEAN, TRACE_TAG_INTEGER, TRACE_TAG_NIL, TRACE_TAG_STRING, TraceInputSlot,
};
use crate::jit_ir::trace::MAX_UNPACK_VALUES;
use crate::runtime::{
    DirectNativeKind, NativeResults, Runtime, RuntimeError, RuntimeResult, Value,
};
use crate::stdlib::ResultList;
use crate::string::utf8::{self, CodesStep, Len, Offset, Utf8Mode, codes_step};
use crate::string::{INLINE_STRING_BYTES, LuaString};

/// Every result of the native `kind` for the arguments `args`, in place: a
/// native with more results than a [`NativeResults`] holds has none.
#[inline]
fn results_of(
    runtime: &Runtime,
    (bank, roots): (&[Value], &[Value]),
    kind: DirectNativeKind,
    args: &[TraceInputSlot],
    results: &mut NativeResults,
) -> bool {
    let Some(handle) = runtime.direct_native(kind) else {
        return false;
    };
    with_values_apart(args, bank, roots, |values| {
        runtime.probe_native_results_fixed(handle, values, results)
    })
    .unwrap_or(false)
}

/// The word of an integer result.
fn integer(value: &Value) -> Option<i64> {
    match value {
        Value::Integer(integer) => Some(*integer),
        _ => None,
    }
}

/// `utf8.char` and `string.pack`: the string the native gives, as a new lane.
#[inline(never)]
pub(in crate::vm::jit::trace) fn string_result(
    runtime: &Runtime,
    bank: &mut Vec<Value>,
    roots: &[Value],
    (state, lengths): (&mut HostState, &mut [u32]),
    kind: DirectNativeKind,
    args: &[TraceInputSlot],
    out: &mut TraceInputSlot,
) -> bool {
    let mut results = NativeResults::new();
    if !results_of(runtime, (bank, roots), kind, args, &mut results) {
        return false;
    }
    let [Value::String(result)] = results.as_slice() else {
        return false;
    };
    publish_string(bank, state, lengths, result.clone(), out)
}

/// A native that gives exactly `words.len()` integers, as `utf8.codepoint`
/// gives its code points, `utf8.len` its count, `utf8.offset` its positions,
/// `string.packsize` its size and a step of `utf8.codes` its position and
/// its code point.
#[inline(never)]
pub(in crate::vm::jit::trace) fn integers(
    runtime: &Runtime,
    (bank, roots): (&[Value], &[Value]),
    kind: DirectNativeKind,
    args: &[TraceInputSlot],
    words: &mut [i64],
) -> bool {
    let mut results = NativeResults::new();
    if !results_of(runtime, (bank, roots), kind, args, &mut results) {
        return false;
    }
    if results.as_slice().len() != words.len() {
        return false;
    }
    for (word, result) in words.iter_mut().zip(results.as_slice()) {
        let Some(value) = integer(result) else {
            return false;
        };
        *word = value;
    }
    true
}

/// How a native that has nothing to give besides a failure answers.
#[derive(Clone, Copy, Eq, PartialEq)]
pub(in crate::vm::jit::trace) enum Failure {
    /// nil, and a position: `utf8.len` of an invalid string.
    NilAndPosition,
    /// nil alone: `utf8.offset` that finds no character.
    Nil,
    /// No result: the end of a `utf8.codes` loop.
    Nothing,
}

/// A native that gives the failure `failure`. Writes the position of a
/// failure that has one.
#[inline(never)]
pub(in crate::vm::jit::trace) fn failure(
    runtime: &Runtime,
    (bank, roots): (&[Value], &[Value]),
    kind: DirectNativeKind,
    args: &[TraceInputSlot],
    failure: Failure,
    words: &mut [i64],
) -> bool {
    let mut results = NativeResults::new();
    if !results_of(runtime, (bank, roots), kind, args, &mut results) {
        return false;
    }
    match (failure, results.as_slice()) {
        (Failure::NilAndPosition, [Value::Nil, position]) => {
            let Some(position) = integer(position) else {
                return false;
            };
            words[0] = position;
            true
        }
        (Failure::Nil, [Value::Nil]) | (Failure::Nothing, []) => true,
        _ => false,
    }
}

/// `string.pack`: the string the native's own code builds, as a new lane.
/// The code runs on the argument values with no list for its one result.
#[inline(never)]
pub(in crate::vm::jit::trace) fn pack(
    runtime: &Runtime,
    bank: &mut Vec<Value>,
    roots: &[Value],
    (state, lengths): (&mut HostState, &mut [u32]),
    args: &[TraceInputSlot],
    out: &mut TraceInputSlot,
) -> bool {
    if runtime
        .direct_native(DirectNativeKind::StringPack)
        .is_none()
    {
        return false;
    }
    let Some(Some(packed)) = with_values_apart(args, bank, roots, |values| {
        crate::stdlib::pack_text(runtime, values).ok()
    }) else {
        return false;
    };
    publish_string(bank, state, lengths, packed, out)
}

/// The results of a `string.unpack` call as the words of its helper: each
/// value is checked against the kind the helper stands for as it arrives,
/// and an integer or a float goes straight to its word. The strings wait in
/// `texts` until the source they were cut from is no longer borrowed from
/// the bank, which a new lane may move.
struct UnpackWords<'a> {
    words: &'a mut [i64],
    filled: usize,
    kinds: usize,
    texts: [Option<LuaString>; MAX_UNPACK_VALUES],
}

impl ResultList for UnpackWords<'_> {
    #[inline(always)]
    fn put(&mut self, value: Value) -> RuntimeResult<()> {
        // Any error here is a miss, and this one allocates no message.
        let at = self.filled;
        let Some(values) = self.words.len().checked_sub(1) else {
            return Err(RuntimeError::Redispatch);
        };
        if at > values {
            return Err(RuntimeError::Redispatch);
        }
        // The last word is the next position, an integer.
        let kind = if at == values {
            0
        } else {
            let kind = self.kinds % 3;
            self.kinds /= 3;
            kind
        };
        match (kind, value) {
            (0, Value::Integer(integer)) => self.words[at] = integer,
            (1, Value::Number(number)) => self.words[at] = number.to_bits().cast_signed(),
            (2, Value::String(text)) => match self.texts.get_mut(at) {
                Some(slot) => *slot = Some(text),
                None => return Err(RuntimeError::Redispatch),
            },
            _ => return Err(RuntimeError::Redispatch),
        }
        self.filled = at + 1;
        Ok(())
    }
}

/// `string.unpack` of a format and a source that are strings and a start
/// that is absent or an integer, by the native's own code on their bytes:
/// no argument list, no result list. `None` for any other call.
#[inline]
fn unpack_fast(
    runtime: &Runtime,
    bank: &mut Vec<Value>,
    roots: &[Value],
    (state, lengths): (&mut HostState, &mut [u32]),
    args: &[TraceInputSlot],
    kinds: usize,
    words: &mut [i64],
) -> Option<bool> {
    let clamped = runtime.unpack_clamped()?;
    runtime.direct_native(DirectNativeKind::StringUnpack)?;
    if args.len() > 3 {
        return None;
    }
    let start = integer_at(args, 2).ok()?;
    let mut sink = UnpackWords {
        words,
        filled: 0,
        kinds,
        texts: [const { None }; MAX_UNPACK_VALUES],
    };
    {
        let format = text(*args.first()?, bank, roots)?;
        let source = text(*args.get(1)?, bank, roots)?;
        if crate::stdlib::unpack_text(
            runtime,
            format.as_bytes(),
            source.as_bytes(),
            start,
            clamped,
            &mut sink,
        )
        .is_err()
        {
            return Some(false);
        }
    }
    if sink.filled != sink.words.len() {
        return Some(false);
    }
    for (word, text) in sink.words.iter_mut().zip(&mut sink.texts) {
        if let Some(text) = text.take() {
            let mut slot = TraceInputSlot::other();
            if !publish_string(bank, state, lengths, text, &mut slot) {
                return Some(false);
            }
            *word = slot.bits();
        }
    }
    Some(true)
}

/// `string.unpack`: `values` values of the kinds the base-3 digits of
/// `kinds` say (0 an integer, 1 a float, 2 a string), then the next
/// position. The strings are new lanes.
#[inline(never)]
pub(in crate::vm::jit::trace) fn unpack(
    runtime: &Runtime,
    bank: &mut Vec<Value>,
    roots: &[Value],
    (state, lengths): (&mut HostState, &mut [u32]),
    args: &[TraceInputSlot],
    (values, kinds): (usize, usize),
    words: &mut [i64],
) -> bool {
    if words.len() == values + 1
        && let Some(answered) =
            unpack_fast(runtime, bank, roots, (state, lengths), args, kinds, words)
    {
        return answered;
    }
    let mut results = NativeResults::new();
    if !results_of(
        runtime,
        (bank, roots),
        DirectNativeKind::StringUnpack,
        args,
        &mut results,
    ) {
        return false;
    }
    let results = results.as_slice();
    if results.len() != values + 1 || integer(&results[values]).is_none() {
        return false;
    }
    let mut digits = kinds;
    for result in &results[..values] {
        let right = match digits % 3 {
            0 => matches!(result, Value::Integer(_)),
            1 => matches!(result, Value::Number(_)),
            _ => matches!(result, Value::String(_)),
        };
        if !right {
            return false;
        }
        digits /= 3;
    }
    for (word, result) in words.iter_mut().zip(results) {
        *word = match result {
            Value::Integer(integer) => *integer,
            Value::Number(number) => number.to_bits().cast_signed(),
            Value::String(text) => {
                let mut slot = TraceInputSlot::other();
                if !publish_string(bank, state, lengths, text.clone(), &mut slot) {
                    return false;
                }
                slot.bits()
            }
            _ => return false,
        };
    }
    true
}

/// The check before `utf8.codes (s)` gives its iterator: the string must not
/// start with a continuation byte, which the call raises an error for.
#[inline(never)]
pub(in crate::vm::jit::trace) fn codes_check(
    (bank, roots): (&[Value], &[Value]),
    args: &[TraceInputSlot],
) -> bool {
    let [slot] = args else {
        return false;
    };
    if slot.tag() != TRACE_TAG_STRING && slot.tag() != crate::jit::TRACE_TAG_BORROWED_STRING {
        return false;
    }
    text(*slot, bank, roots).is_some_and(|source| {
        source
            .as_bytes()
            .first()
            .is_none_or(|byte| byte & 0xc0 != 0x80)
    })
}

/// A step of a `utf8.codes` loop over a string, with the position of the
/// last character an integer: the step `mode` names (0 the strict one, 1 the
/// lax one, 2 Lua 5.3's), worked out on the bytes. This is the code of the
/// native (see [`oximond::string::utf8::codes_step`]) without a call. `None`
/// for any other arguments, which the native answers.
#[inline]
pub(in crate::vm::jit::trace) fn codes_next(
    (bank, roots): (&[Value], &[Value]),
    mode: i64,
    args: &[TraceInputSlot; 2],
) -> Option<CodesStep> {
    let mode = match mode {
        0 => Utf8Mode::Strict,
        1 => Utf8Mode::Lax,
        2 => Utf8Mode::Lua53,
        _ => return None,
    };
    if args[1].tag() != TRACE_TAG_INTEGER {
        return None;
    }
    let source = text(args[0], bank, roots)?;
    let index = isize::try_from(args[1].bits()).ok()?;
    Some(codes_step(source.as_bytes(), index, mode))
}

/// An argument the fast paths below leave to the native: a float, a string.
struct Other;

/// The integer argument at `at`: `Ok(None)` for one left out or nil,
/// `Ok(Some(_))` for an integer and `Err` for any other.
fn integer_at(args: &[TraceInputSlot], at: usize) -> Result<Option<i64>, Other> {
    match args.get(at) {
        None => Ok(None),
        Some(slot) if slot.tag() == TRACE_TAG_NIL => Ok(None),
        Some(slot) if slot.tag() == TRACE_TAG_INTEGER => Ok(Some(slot.bits())),
        Some(_) => Err(Other),
    }
}

/// The mode of `utf8.len` and `utf8.codepoint` for a call whose lax flag is
/// the argument at `at`. Lua 5.3 has no flag. A flag is true for anything
/// but nil and false.
fn mode_of(runtime: &Runtime, args: &[TraceInputSlot], at: usize) -> Utf8Mode {
    if runtime
        .direct_native(DirectNativeKind::Utf8Next53)
        .is_some()
    {
        return Utf8Mode::Lua53;
    }
    match args.get(at) {
        None => Utf8Mode::Strict,
        Some(slot) if slot.tag() == TRACE_TAG_NIL => Utf8Mode::Strict,
        Some(slot) if slot.tag() == TRACE_TAG_BOOLEAN && slot.bits() == 0 => Utf8Mode::Strict,
        Some(_) => Utf8Mode::Lax,
    }
}

/// `utf8.len` of a string and integer positions, worked out on the bytes.
/// `None` for any other call, which the native answers.
#[inline]
pub(in crate::vm::jit::trace) fn len_fast(
    runtime: &Runtime,
    (bank, roots): (&[Value], &[Value]),
    args: &[TraceInputSlot],
) -> Option<Len> {
    let source = text(*args.first()?, bank, roots)?;
    let (start, end) = (integer_at(args, 1).ok()?, integer_at(args, 2).ok()?);
    utf8::len(source.as_bytes(), start, end, mode_of(runtime, args, 3))
}

/// `utf8.codepoint` of a string and integer positions: the code points in
/// `out` and how many. `None` for any other call, and for a call that gives
/// more than `out` holds.
#[inline]
pub(in crate::vm::jit::trace) fn codepoints_fast(
    runtime: &Runtime,
    (bank, roots): (&[Value], &[Value]),
    args: &[TraceInputSlot],
    out: &mut [u32],
) -> Option<usize> {
    let source = text(*args.first()?, bank, roots)?;
    let (start, end) = (integer_at(args, 1).ok()?, integer_at(args, 2).ok()?);
    utf8::codepoints(
        source.as_bytes(),
        start,
        end,
        mode_of(runtime, args, 3),
        out,
    )
}

/// `utf8.offset` of a string and integers.
#[inline]
pub(in crate::vm::jit::trace) fn offset_fast(
    (bank, roots): (&[Value], &[Value]),
    args: &[TraceInputSlot],
) -> Option<Offset> {
    let source = text(*args.first()?, bank, roots)?;
    let shift = integer_at(args, 1).ok()??;
    utf8::offset(source.as_bytes(), shift, integer_at(args, 2).ok()?)
}

/// `utf8.char` of integers: the new string as a lane. `None` for any other
/// call, which the native answers.
#[inline]
pub(in crate::vm::jit::trace) fn char_fast(
    runtime: &Runtime,
    bank: &mut Vec<Value>,
    (state, lengths): (&mut HostState, &mut [u32]),
    args: &[TraceInputSlot],
    out: &mut TraceInputSlot,
) -> Option<bool> {
    // Lua 5.3 stops at 0x10FFFF, the later profiles at 0x7FFFFFFF.
    let maximum = if runtime
        .direct_native(DirectNativeKind::Utf8Next53)
        .is_some()
    {
        0x10_ffff
    } else {
        0x7fff_ffff
    };
    // Each character is stored as a whole word and the next one starts
    // where its bytes end: the bytes past `length` stay zero.
    let mut bytes = [0_u8; 5 * 8];
    let mut length = 0;
    for slot in args {
        if slot.tag() != TRACE_TAG_INTEGER {
            return None;
        }
        let (word, count) = utf8::encode_word(slot.bits(), maximum)?;
        bytes
            .get_mut(length..length + 8)?
            .copy_from_slice(&word.to_le_bytes());
        length += count;
    }
    // A few characters fit a string's inline buffer, which the zeroed
    // bytes fill as they are.
    let made = match bytes.first_chunk::<INLINE_STRING_BYTES>() {
        Some(inline) if length <= INLINE_STRING_BYTES => {
            LuaString::from_inline_buffer(*inline, length)
        }
        _ => LuaString::from(&bytes[..length]),
    };
    Some(publish_string(bank, state, lengths, made, out))
}
