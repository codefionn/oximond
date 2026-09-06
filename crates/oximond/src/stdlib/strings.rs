//! The string and utf8 libraries: find, match, format, rep, and number parsing.

use crate::pattern::{Match as PatternMatch, Pattern};
use crate::runtime::{
    NativePair, NativeResults, NonFiniteNumerals, NumberFormat, PlainNumeral, Runtime,
    RuntimeError, RuntimeResult, Value, format_integer, scan_plain_numeral,
};
use crate::stdlib::args::{
    float_string_index, relative_index, string_index_argument, substring_range,
    usize_integer_value, usize_to_number,
};
use crate::stdlib::format::{
    FastConversion, append_format_bytes, append_format_value, append_short, format_value,
    parse_format_specification, string_ascii_case,
};
use crate::string::LuaString;
use std::borrow::Cow;
use std::cell::{Cell, RefCell};

thread_local! {
    static LAST_FORMAT_PLAN: RefCell<Option<FormatPlan>> = const { RefCell::new(None) };
    static FORMAT_BUFFER: Cell<Vec<u8>> = const { Cell::new(Vec::new()) };
}

/// The longest output buffer worth keeping between `string.format` calls.
const KEPT_FORMAT_BUFFER: usize = 1 << 16;

/// An empty output buffer that keeps its capacity from the previous call,
/// so a formatting loop grows one buffer instead of reallocating a new one
/// per call. Taking it leaves an empty one behind, which makes a nested
/// `string.format` safe.
fn take_format_buffer() -> Vec<u8> {
    let mut buffer = FORMAT_BUFFER.take();
    buffer.clear();
    buffer
}

/// Copies a finished output into its Lua string, which is allocated at its
/// exact size, and keeps the buffer for the next call.
fn finish_format_buffer(buffer: Vec<u8>) -> Value {
    let value = Value::from(&buffer[..]);
    if buffer.capacity() <= KEPT_FORMAT_BUFFER {
        FORMAT_BUFFER.set(buffer);
    }
    value
}

struct FormatPlan {
    source: LuaString,
    pointer_supported: bool,
    operations: Vec<FormatOperation>,
    /// The direct writer of each operation that has one, in step with
    /// `operations`, as in [`PreparedFormat`].
    fast: Vec<Option<FastConversion>>,
}

enum FormatOperation {
    Literal(usize, usize),
    Percent,
    Conversion(crate::stdlib::FormatSpecification),
}

use crate::string::utf8::{CodesStep, Utf8Mode, codes_step, decode as utf8_decode};

fn utf8_relative(position: isize, length: isize) -> isize {
    if position >= 0 {
        position
    } else {
        length.saturating_add(position).saturating_add(1).max(0)
    }
}

pub(crate) fn utf8_char(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    vec_of(|values| utf8_char_limit(runtime, arguments, 0x7fff_ffff, values))
}
pub(crate) fn utf8_char_lua53(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    vec_of(|values| utf8_char_limit(runtime, arguments, 0x10_ffff, values))
}
fn utf8_char_limit<R: ResultList>(
    runtime: &Runtime,
    arguments: &[Value],
    maximum: isize,
    values: &mut R,
) -> RuntimeResult<()> {
    // A few characters, the usual case, are encoded on the stack.
    let mut result = crate::stdlib::string_pack::PackBuffer::new();
    for index in 0..arguments.len() {
        let code = string_index_argument(runtime, arguments.get(index), index + 1)?;
        if !(0..=maximum).contains(&code) {
            return Err(RuntimeError::argument_error(
                index + 1,
                "value out of range",
            ));
        }
        let mut code = u32::try_from(code).expect("range checked");
        if code < 0x80 {
            result.extend_from_slice(&[u8::try_from(code).expect("ASCII")])?;
        } else {
            // At most 6 bytes: 0x7fff_ffff is 31 bits, a 1-bit prefix
            // payload and five 6-bit continuations. They fill from the end.
            let mut encoded = [0_u8; 6];
            let mut start = encoded.len();
            let mut mask = 0x3f_u32;
            while code > mask {
                start -= 1;
                encoded[start] = u8::try_from(0x80 | (code & 0x3f)).expect("continuation");
                code >>= 6;
                mask >>= 1;
            }
            let prefix = !(0x7f_u8 >> (encoded.len() - start));
            start -= 1;
            encoded[start] = prefix | u8::try_from(code).expect("prefix payload");
            result.extend_from_slice(&encoded[start..])?;
        }
    }
    values.put(Value::String(result.finish()))
}
/// The answers of the `utf8` natives for a trace to learn before the call
/// (see [`oximond::runtime::NativeResultsProbe`]). Each is the callback's own
/// code on a shared runtime.
pub(crate) fn utf8_char_probe(runtime: &Runtime, arguments: &[Value]) -> Option<Vec<Value>> {
    vec_of(|values| utf8_char_limit(runtime, arguments, 0x7fff_ffff, values)).ok()
}
pub(crate) fn utf8_char_lua53_probe(runtime: &Runtime, arguments: &[Value]) -> Option<Vec<Value>> {
    vec_of(|values| utf8_char_limit(runtime, arguments, 0x10_ffff, values)).ok()
}
pub(crate) fn utf8_codepoint_probe(runtime: &Runtime, arguments: &[Value]) -> Option<Vec<Value>> {
    vec_of(|values| utf8_codepoint_mode(runtime, arguments, utf8_lax_mode(arguments), values)).ok()
}
pub(crate) fn utf8_codepoint_lua53_probe(
    runtime: &Runtime,
    arguments: &[Value],
) -> Option<Vec<Value>> {
    vec_of(|values| utf8_codepoint_mode(runtime, arguments, Utf8Mode::Lua53, values)).ok()
}
pub(crate) fn utf8_len_probe(runtime: &Runtime, arguments: &[Value]) -> Option<Vec<Value>> {
    vec_of(|values| utf8_len_mode(runtime, arguments, utf8_lax_mode(arguments), values)).ok()
}
pub(crate) fn utf8_len_lua53_probe(runtime: &Runtime, arguments: &[Value]) -> Option<Vec<Value>> {
    vec_of(|values| utf8_len_mode(runtime, arguments, Utf8Mode::Lua53, values)).ok()
}
pub(crate) fn utf8_offset_probe(runtime: &Runtime, arguments: &[Value]) -> Option<Vec<Value>> {
    vec_of(|values| utf8_offset_with_end(runtime, arguments, false, false, values)).ok()
}
pub(crate) fn utf8_offset_lua53_probe(
    runtime: &Runtime,
    arguments: &[Value],
) -> Option<Vec<Value>> {
    vec_of(|values| utf8_offset_with_end(runtime, arguments, false, true, values)).ok()
}
pub(crate) fn utf8_offset_lua55_probe(
    runtime: &Runtime,
    arguments: &[Value],
) -> Option<Vec<Value>> {
    vec_of(|values| utf8_offset_with_end(runtime, arguments, true, false, values)).ok()
}
pub(crate) fn utf8_next_probe(runtime: &Runtime, arguments: &[Value]) -> Option<Vec<Value>> {
    utf8_next_mode(runtime, arguments, Utf8Mode::Strict).ok()
}
pub(crate) fn utf8_next_lax_probe(runtime: &Runtime, arguments: &[Value]) -> Option<Vec<Value>> {
    utf8_next_mode(runtime, arguments, Utf8Mode::Lax).ok()
}
pub(crate) fn utf8_next_lua53_probe(runtime: &Runtime, arguments: &[Value]) -> Option<Vec<Value>> {
    utf8_next_mode(runtime, arguments, Utf8Mode::Lua53).ok()
}
/// The answers of the `utf8` natives in a [`NativeResults`], for a trace to
/// learn at each pass without a list to allocate (see
/// [`oximond::runtime::NativeFixedResultsProbe`]).
pub(crate) fn utf8_char_fixed_probe(
    runtime: &Runtime,
    arguments: &[Value],
    results: &mut NativeResults,
) -> bool {
    utf8_char_limit(runtime, arguments, 0x7fff_ffff, results).is_ok()
}
pub(crate) fn utf8_char_lua53_fixed_probe(
    runtime: &Runtime,
    arguments: &[Value],
    results: &mut NativeResults,
) -> bool {
    utf8_char_limit(runtime, arguments, 0x10_ffff, results).is_ok()
}
pub(crate) fn utf8_codepoint_fixed_probe(
    runtime: &Runtime,
    arguments: &[Value],
    results: &mut NativeResults,
) -> bool {
    utf8_codepoint_mode(runtime, arguments, utf8_lax_mode(arguments), results).is_ok()
}
pub(crate) fn utf8_codepoint_lua53_fixed_probe(
    runtime: &Runtime,
    arguments: &[Value],
    results: &mut NativeResults,
) -> bool {
    utf8_codepoint_mode(runtime, arguments, Utf8Mode::Lua53, results).is_ok()
}
pub(crate) fn utf8_len_fixed_probe(
    runtime: &Runtime,
    arguments: &[Value],
    results: &mut NativeResults,
) -> bool {
    utf8_len_mode(runtime, arguments, utf8_lax_mode(arguments), results).is_ok()
}
pub(crate) fn utf8_len_lua53_fixed_probe(
    runtime: &Runtime,
    arguments: &[Value],
    results: &mut NativeResults,
) -> bool {
    utf8_len_mode(runtime, arguments, Utf8Mode::Lua53, results).is_ok()
}
pub(crate) fn utf8_offset_fixed_probe(
    runtime: &Runtime,
    arguments: &[Value],
    results: &mut NativeResults,
) -> bool {
    utf8_offset_with_end(runtime, arguments, false, false, results).is_ok()
}
pub(crate) fn utf8_offset_lua53_fixed_probe(
    runtime: &Runtime,
    arguments: &[Value],
    results: &mut NativeResults,
) -> bool {
    utf8_offset_with_end(runtime, arguments, false, true, results).is_ok()
}
pub(crate) fn utf8_offset_lua55_fixed_probe(
    runtime: &Runtime,
    arguments: &[Value],
    results: &mut NativeResults,
) -> bool {
    utf8_offset_with_end(runtime, arguments, true, false, results).is_ok()
}

/// Where a native collects its results: a `Vec` for a call, a
/// [`NativeResults`] for a probe that allocates no list.
pub(crate) trait ResultList {
    /// Adds a result. Fails when a fixed list is full: the call has no
    /// answer a probe can give.
    fn put(&mut self, value: Value) -> RuntimeResult<()>;
}

impl ResultList for Vec<Value> {
    #[inline(always)]
    fn put(&mut self, value: Value) -> RuntimeResult<()> {
        self.push(value);
        Ok(())
    }
}

impl ResultList for NativeResults {
    #[inline(always)]
    fn put(&mut self, value: Value) -> RuntimeResult<()> {
        self.push(value)
            .map_err(|_| RuntimeError::message("too many results"))
    }
}

/// The results `collect` puts in a `Vec`.
#[inline(always)]
pub(crate) fn vec_of(
    collect: impl FnOnce(&mut Vec<Value>) -> RuntimeResult<()>,
) -> RuntimeResult<Vec<Value>> {
    let mut values = Vec::new();
    collect(&mut values)?;
    Ok(values)
}

/// Adds the integer position `position` to `values` the way the profile
/// represents integers.
#[inline(always)]
pub(crate) fn put_position<R: ResultList>(
    runtime: &Runtime,
    position: usize,
    message: &'static str,
    values: &mut R,
) -> RuntimeResult<()> {
    if runtime.tagged_integers() {
        let position = i64::try_from(position).map_err(|_| RuntimeError::message(message))?;
        values.put(Value::Integer(position))
    } else {
        values.put(Value::Number(usize_to_number(position, message)?))
    }
}

/// The mode the fourth argument of `utf8.len` and `utf8.codepoint` picks.
fn utf8_lax_mode(arguments: &[Value]) -> Utf8Mode {
    if arguments.get(3).is_some_and(Value::is_truthy) {
        Utf8Mode::Lax
    } else {
        Utf8Mode::Strict
    }
}
pub(crate) fn utf8_codepoint(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    vec_of(|values| utf8_codepoint_mode(runtime, arguments, utf8_lax_mode(arguments), values))
}
pub(crate) fn utf8_codepoint_lua53(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    vec_of(|values| utf8_codepoint_mode(runtime, arguments, Utf8Mode::Lua53, values))
}
fn utf8_codepoint_mode<R: ResultList>(
    runtime: &Runtime,
    arguments: &[Value],
    mode: Utf8Mode,
    values: &mut R,
) -> RuntimeResult<()> {
    let source = string_argument_in(runtime.number_format(), arguments.first(), 1)?;
    let length = isize::try_from(source.len())
        .map_err(|_| RuntimeError::message("string slice too long"))?;
    let start = utf8_relative(
        optional_coerced_index(runtime, arguments.get(1), 2, 1)?,
        length,
    );
    let end = utf8_relative(
        optional_coerced_index(runtime, arguments.get(2), 3, start)?,
        length,
    );
    let bounds = if mode == Utf8Mode::Lua53 {
        "out of range"
    } else {
        "out of bounds"
    };
    if start < 1 {
        return Err(RuntimeError::argument_error(2, bounds));
    }
    if end > length {
        return Err(RuntimeError::argument_error(3, bounds));
    }
    if start > end {
        return Ok(());
    }
    let mut position = usize::try_from(start - 1).expect("positive");
    while position < usize::try_from(end).expect("positive interval") {
        let (next, code) = utf8_decode(&source, position, mode)
            .ok_or_else(|| RuntimeError::message("invalid UTF-8 code"))?;
        values.put(runtime.integer_value(i64::from(code)))?;
        position = next;
    }
    Ok(())
}
pub(crate) fn utf8_len(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    vec_of(|values| utf8_len_mode(runtime, arguments, utf8_lax_mode(arguments), values))
}
pub(crate) fn utf8_len_lua53(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    vec_of(|values| utf8_len_mode(runtime, arguments, Utf8Mode::Lua53, values))
}
fn utf8_len_mode<R: ResultList>(
    runtime: &Runtime,
    arguments: &[Value],
    mode: Utf8Mode,
    values: &mut R,
) -> RuntimeResult<()> {
    let source = string_argument_in(runtime.number_format(), arguments.first(), 1)?;
    let length = isize::try_from(source.len())
        .map_err(|_| RuntimeError::message("UTF-8 string is too long"))?;
    let start = utf8_relative(
        optional_coerced_index(runtime, arguments.get(1), 2, 1)?,
        length,
    );
    let end = utf8_relative(
        optional_coerced_index(runtime, arguments.get(2), 3, -1)?,
        length,
    );
    if start < 1 || start - 1 > length {
        return Err(RuntimeError::argument_error(
            2,
            if mode == Utf8Mode::Lua53 {
                "initial position out of string"
            } else {
                "initial position out of bounds"
            },
        ));
    }
    if end > length {
        return Err(RuntimeError::argument_error(
            3,
            if mode == Utf8Mode::Lua53 {
                "final position out of string"
            } else {
                "final position out of bounds"
            },
        ));
    }
    let mut position = usize::try_from(start - 1).expect("positive");
    let mut count = 0;
    while end > 0 && position < usize::try_from(end).expect("positive") {
        let Some((next, _)) = utf8_decode(&source, position, mode) else {
            values.put(Value::Nil)?;
            return put_position(runtime, position + 1, "UTF-8 position is too large", values);
        };
        position = next;
        count += 1;
    }
    values.put(runtime.integer_value(count))
}
pub(crate) fn utf8_offset(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    vec_of(|values| utf8_offset_with_end(runtime, arguments, false, false, values))
}
pub(crate) fn utf8_offset_lua53(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    vec_of(|values| utf8_offset_with_end(runtime, arguments, false, true, values))
}
pub(crate) fn utf8_offset_lua55(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    vec_of(|values| utf8_offset_with_end(runtime, arguments, true, false, values))
}
fn utf8_cont(source: &[u8], position: usize) -> bool {
    crate::string::utf8::is_continuation(source, position)
}
fn utf8_offset_with_end<R: ResultList>(
    runtime: &Runtime,
    arguments: &[Value],
    with_end: bool,
    legacy: bool,
    values: &mut R,
) -> RuntimeResult<()> {
    let source = string_argument_in(runtime.number_format(), arguments.first(), 1)?;
    let mut shift = string_index_argument(runtime, arguments.get(1), 2)?;
    let length = isize::try_from(source.len())
        .map_err(|_| RuntimeError::message("UTF-8 string is too long"))?;
    let position = utf8_relative(
        optional_coerced_index(
            runtime,
            arguments.get(2),
            3,
            if shift >= 0 { 1 } else { length + 1 },
        )?,
        length,
    );
    if position < 1 || position - 1 > length {
        return Err(RuntimeError::argument_error(
            3,
            if legacy {
                "position out of range"
            } else {
                "position out of bounds"
            },
        ));
    }
    let mut position = usize::try_from(position - 1).expect("positive");
    if shift == 0 {
        while position > 0 && utf8_cont(&source, position) {
            position -= 1;
        }
    } else {
        if utf8_cont(&source, position) {
            return Err(RuntimeError::message(
                "initial position is a continuation byte",
            ));
        }
        if shift < 0 {
            while shift < 0 && position > 0 {
                position -= 1;
                while position > 0 && utf8_cont(&source, position) {
                    position -= 1;
                }
                shift += 1;
            }
        } else {
            shift -= 1;
            while shift > 0 && position < source.len() {
                position += 1;
                while utf8_cont(&source, position) {
                    position += 1;
                }
                shift -= 1;
            }
        }
    }
    if shift != 0 {
        return values.put(Value::Nil);
    }
    put_position(runtime, position + 1, "UTF-8 position is too large", values)?;
    if with_end {
        if utf8_cont(&source, position) {
            return Err(RuntimeError::message(
                "initial position is a continuation byte",
            ));
        }
        if source.get(position).is_some_and(|byte| byte & 0x80 != 0) {
            while utf8_cont(&source, position + 1) {
                position += 1;
            }
        }
        put_position(runtime, position + 1, "UTF-8 position is too large", values)?;
    }
    Ok(())
}
pub(crate) fn utf8_codes_with_iterators(
    runtime: &mut Runtime,
    arguments: &[Value],
    strict: crate::runtime::NativeFunctionHandle,
    lax: crate::runtime::NativeFunctionHandle,
    legacy: bool,
) -> RuntimeResult<Vec<Value>> {
    let source = string_argument_in(runtime.number_format(), arguments.first(), 1)?;
    if !legacy && utf8_cont(&source, 0) {
        return Err(RuntimeError::argument_error(1, "invalid UTF-8 code"));
    }
    let iterator = if !legacy && arguments.get(1).is_some_and(Value::is_truthy) {
        lax
    } else {
        strict
    };
    Ok(vec![
        Value::NativeFunction(iterator),
        Value::bytes(source.into_owned()),
        runtime.integer_value(0),
    ])
}
pub(crate) fn utf8_next(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    utf8_next_mode(runtime, arguments, Utf8Mode::Strict)
}
pub(crate) fn utf8_next_lax(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    utf8_next_mode(runtime, arguments, Utf8Mode::Lax)
}
pub(crate) fn utf8_next_lua53(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    utf8_next_mode(runtime, arguments, Utf8Mode::Lua53)
}
fn utf8_next_mode(
    runtime: &Runtime,
    arguments: &[Value],
    mode: Utf8Mode,
) -> RuntimeResult<Vec<Value>> {
    let source = string_argument_in(runtime.number_format(), arguments.first(), 1)?;
    // lua_tointeger defaults failed conversion to zero rather than raising.
    let index = string_index_argument(runtime, arguments.get(1), 2).unwrap_or(0);
    match codes_step(&source, index, mode) {
        CodesStep::End => Ok(Vec::new()),
        CodesStep::Invalid => Err(RuntimeError::message("invalid UTF-8 code")),
        CodesStep::Character { position, code } => Ok(vec![
            usize_integer_value(runtime, position, "UTF-8 position is too large")?,
            runtime.integer_value(i64::from(code)),
        ]),
    }
}

/// `string.find` of a plain string in a string, as [`string_find`] answers
/// it, without a result vector. Declines (`None`) every other call.
///
/// # Errors
///
/// None today; the signature matches the other native entry points.
pub fn string_find_pair(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Option<NativePair>> {
    if let [
        Value::String(source),
        Value::String(pattern),
        initial,
        plain,
        ..,
    ] = arguments
        && plain.is_truthy()
        && let Some(initial) = small_index(initial, 1)
    {
        return Ok(Some(plain_find_pair(
            runtime,
            source.as_bytes(),
            pattern.as_bytes(),
            initial,
        )));
    }
    Ok(None)
}

/// Finds a literal byte sequence or a minimal pattern containing `.` wildcards.
///
/// Captures and other Lua pattern operators deliberately return an error until
/// the pattern engine can implement them correctly.
///
/// # Errors
///
/// Returns an error for invalid arguments or unsupported pattern magic.
pub fn string_find(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    if let [
        Value::String(source),
        Value::String(pattern),
        initial,
        plain,
        ..,
    ] = arguments
        && plain.is_truthy()
        && let Some(initial) = small_index(initial, 1)
    {
        return Ok(plain_find_values(
            runtime,
            source.as_bytes(),
            pattern.as_bytes(),
            initial,
        ));
    }
    let source = string_argument_in(runtime.number_format(), arguments.first(), 1)?;
    let pattern = string_argument_in(runtime.number_format(), arguments.get(1), 2)?;
    let length = isize::try_from(source.len())
        .map_err(|_| RuntimeError::message("string.find input is too long"))?;
    let initial = optional_coerced_index(runtime, arguments.get(2), 3, 1)?;
    let mut initial = relative_index(initial, length);
    if initial > length.saturating_add(1) {
        // Lua 5.1 and LuaJIT start a search past the end at the end; 5.3 and
        // later fail it.
        if !runtime.string_init_clamps() {
            return Ok(vec![Value::Nil]);
        }
        initial = length.saturating_add(1);
    }
    let initial = initial.max(1);
    let initial = usize::try_from(initial - 1)
        .map_err(|_| RuntimeError::message("string.find start index is invalid"))?;
    if arguments.get(3).is_some_and(Value::is_truthy) {
        let Some(start) = plain_find(&source, &pattern, initial) else {
            return Ok(vec![Value::Nil]);
        };
        return Ok(vec![
            usize_integer_value(runtime, start + 1, "string.find position is too large")?,
            usize_integer_value(
                runtime,
                start + pattern.len(),
                "string.find position is too large",
            )?,
        ]);
    }
    let found = Pattern::shared(&pattern, false)
        .map_err(|error| pattern_error(runtime, error))?
        .find(&source, initial)
        .map_err(|error| pattern_error(runtime, error))?;
    let Some(found) = found else {
        return Ok(vec![Value::Nil]);
    };
    let mut values = vec![
        usize_integer_value(
            runtime,
            found.start() + 1,
            "string.find position is too large",
        )?,
        usize_integer_value(runtime, found.end(), "string.find position is too large")?,
    ];
    values.extend(pattern_captures(runtime, &source, &found)?);
    Ok(values)
}

/// Returns the captures of the first Lua byte-pattern match, or the full match
/// where the pattern has no captures.
///
/// # Errors
///
/// Returns an error for invalid arguments or malformed patterns.
pub fn string_match(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let source = string_argument_in(runtime.number_format(), arguments.first(), 1)?;
    let pattern = string_argument_in(runtime.number_format(), arguments.get(1), 2)?;
    let length = isize::try_from(source.len())
        .map_err(|_| RuntimeError::message("string.match input is too long"))?;
    let initial = optional_coerced_index(runtime, arguments.get(2), 3, 1)?;
    let mut initial = relative_index(initial, length);
    if initial > length.saturating_add(1) {
        // As in `string.find`: only Lua 5.1 and LuaJIT clamp a start past the end.
        if !runtime.string_init_clamps() {
            return Ok(vec![Value::Nil]);
        }
        initial = length.saturating_add(1);
    }
    let initial = initial.max(1);
    let initial = usize::try_from(initial - 1)
        .map_err(|_| RuntimeError::message("string.match start index is invalid"))?;
    let found = Pattern::shared(&pattern, false)
        .map_err(|error| pattern_error(runtime, error))?
        .find(&source, initial)
        .map_err(|error| pattern_error(runtime, error))?;
    let Some(found) = found else {
        return Ok(vec![Value::Nil]);
    };
    let captures = pattern_captures(runtime, &source, &found)?;
    if captures.is_empty() {
        Ok(vec![Value::from(&source[found.start()..found.end()])])
    } else {
        Ok(captures)
    }
}

/// Returns a callable iterator which yields each Lua byte-pattern match.
///
/// The VM owns invocation because an iterator must keep state between ordinary
/// Lua calls. This native entry point validates the pattern and allocates the
/// runtime-owned state used by that invocation path.
///
/// # Errors
///
/// Returns an error for invalid arguments, malformed patterns, or allocation
/// failure.
pub fn string_gmatch(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let source = string_value_in(runtime.number_format(), arguments.first(), 1)?;
    let pattern = string_value_in(runtime.number_format(), arguments.get(1), 2)?;
    let handle = runtime.create_gmatch_iterator(source, pattern)?;
    let mut values = runtime.result_vec();
    values.push(Value::GmatchIterator(handle));
    Ok(values)
}

/// `string.gmatch` of Lua 5.4 and later, which takes the position to start
/// at as a third argument (negative from the end, clamped to the string).
///
/// # Errors
///
/// As [`string_gmatch`], and for an initial position that is no integer.
pub fn string_gmatch_init(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let source = string_value_in(runtime.number_format(), arguments.first(), 1)?;
    let pattern = string_value_in(runtime.number_format(), arguments.get(1), 2)?;
    let length = isize::try_from(source.len())
        .map_err(|_| RuntimeError::message("string.gmatch input is too long"))?;
    let initial = optional_coerced_index(runtime, arguments.get(2), 3, 1)?;
    // `posrelatI`, then a search that starts past the end finds nothing.
    let initial = relative_index(initial, length).max(1);
    let start = usize::try_from(initial - 1).unwrap_or(usize::MAX);
    let handle = runtime.create_gmatch_iterator(source, pattern)?;
    if start > 0 {
        // Past the end, the iterator is done.
        let end = usize::try_from(length).unwrap_or(0).saturating_add(1);
        runtime.set_gmatch_iterator_search(handle, start.min(end))?;
    }
    Ok(vec![Value::GmatchIterator(handle)])
}

/// Formats byte strings and numbers with the Lua 5.1 conversion forms used by
/// the early string conformance tests.
///
/// Floating-point conversions round exactly and tie to even, matching glibc's
/// printf in the C locale Lua uses.
///
/// # Errors
///
/// Returns an error for malformed or unsupported format directives, missing
/// arguments, invalid conversions, or a result that cannot be allocated.
#[cfg(test)]
pub fn string_format(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    string_format_inner(
        arguments,
        false,
        false,
        runtime.number_format(),
        runtime.nonfinite_numerals(),
    )
}

pub(crate) fn string_format_modern(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    string_format_inner(
        arguments,
        false,
        true,
        runtime.number_format(),
        runtime.nonfinite_numerals(),
    )
}

pub(crate) fn string_format_with_pointer(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    string_format_inner(
        arguments,
        true,
        true,
        runtime.number_format(),
        runtime.nonfinite_numerals(),
    )
}

pub(crate) fn string_format_inner(
    arguments: &[Value],
    pointer_supported: bool,
    modern_quote: bool,
    numbers: NumberFormat,
    numerals: NonFiniteNumerals,
) -> RuntimeResult<Vec<Value>> {
    string_format_value(
        arguments,
        pointer_supported,
        modern_quote,
        numbers,
        numerals,
    )
    .map(|value| vec![value])
}

/// `string.format` of Lua 5.1, which returns exactly one string.
///
/// # Errors
///
/// As the vector-returning `string.format`.
pub fn string_format_scalar(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    string_format_value(
        arguments,
        false,
        false,
        runtime.number_format(),
        runtime.nonfinite_numerals(),
    )
}

/// `LuaJIT`'s `string.format`, which returns exactly one string. A `%s`
/// argument that is not a string or number goes to the VM, which calls its
/// `__tostring` or formats it as `tostring` does.
///
/// # Errors
///
/// As [`string_format_scalar`], and `Redispatch` for a `__tostring` argument.
pub fn string_format_luajit_scalar(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Value> {
    // LuaJIT's `%s` formats any value as `tostring` does, with `__tostring`.
    // The raw formatter rejects such an argument, so the common call (only
    // strings and numbers) pays nothing; on an error, the VM converts the
    // argument and formats again.
    match string_format_scalar(runtime, arguments) {
        Err(_)
            if arguments.iter().skip(1).any(|value| {
                !matches!(
                    value,
                    Value::String(_) | Value::Integer(_) | Value::Number(_)
                ) && crate::runtime::cint64::CInt64::from_value(value).is_none()
            }) =>
        {
            Err(RuntimeError::Redispatch)
        }
        result => result,
    }
}

fn string_format_value(
    arguments: &[Value],
    pointer_supported: bool,
    modern_quote: bool,
    numbers: NumberFormat,
    numerals: NonFiniteNumerals,
) -> RuntimeResult<Value> {
    if let Some(Value::String(source)) = arguments.first() {
        let cached = LAST_FORMAT_PLAN.with(|slot| {
            let slot = slot.borrow();
            let plan = slot.as_ref()?;
            (plan.source.identity() == source.identity()
                && plan.pointer_supported == pointer_supported)
                .then(|| format_with_plan(arguments, plan, modern_quote, numbers, numerals))
        });
        if let Some(result) = cached {
            return result;
        }
    }

    let format = string_argument_in(numbers, arguments.first(), 1)?;
    let mut result = take_format_buffer();
    result.reserve(format.len());
    let mut format_index = 0;
    let mut argument_index = 1;
    while format_index < format.len() {
        if format[format_index] != b'%' {
            let literal_end = format[format_index..]
                .iter()
                .position(|byte| *byte == b'%')
                .map_or(format.len(), |offset| format_index + offset);
            append_format_bytes(&mut result, &format[format_index..literal_end])?;
            format_index = literal_end;
            continue;
        }
        let (specification, next) =
            parse_format_specification(&format, format_index + 1, pointer_supported)?;
        format_index = next;
        if specification.conversion == b'%' {
            if specification.has_modifiers() {
                return Err(RuntimeError::message(
                    "string.format does not support modifiers on %%",
                ));
            }
            append_format_bytes(&mut result, b"%")?;
            continue;
        }
        let argument = arguments
            .get(argument_index)
            .ok_or_else(|| missing_format_argument(numbers, argument_index + 1))?;
        argument_index += 1;
        let argument = float_operand(argument, &specification, numerals);
        let value = format_value(&argument, &specification, modern_quote, numbers)
            .map_err(|error| format_error_at(error, argument_index))?;
        append_format_bytes(&mut result, &value)?;
    }
    let result = finish_format_buffer(result);
    if let Some(Value::String(source)) = arguments.first() {
        // A successful call has already validated the whole format. Keep one
        // parsed format per thread for loops that repeatedly use the same one.
        let plan = parse_format_plan(source.clone(), pointer_supported)?;
        LAST_FORMAT_PLAN.with(|slot| *slot.borrow_mut() = Some(plan));
    }
    Ok(result)
}

fn parse_format_plan(source: LuaString, pointer_supported: bool) -> RuntimeResult<FormatPlan> {
    let operations = parse_format_operations(source.as_bytes(), pointer_supported)?;
    let fast = fast_conversions(&operations);
    Ok(FormatPlan {
        source,
        pointer_supported,
        operations,
        fast,
    })
}

fn parse_format_operations(
    format: &[u8],
    pointer_supported: bool,
) -> RuntimeResult<Vec<FormatOperation>> {
    let mut operations = Vec::new();
    let mut index = 0;
    while index < format.len() {
        if format[index] == b'%' {
            let (specification, next) =
                parse_format_specification(format, index + 1, pointer_supported)?;
            operations.push(if specification.conversion == b'%' {
                if specification.has_modifiers() {
                    return Err(RuntimeError::message(
                        "string.format does not support modifiers on %%",
                    ));
                }
                FormatOperation::Percent
            } else {
                FormatOperation::Conversion(specification)
            });
            index = next;
        } else {
            let end = format[index..]
                .iter()
                .position(|byte| *byte == b'%')
                .map_or(format.len(), |offset| index + offset);
            operations.push(FormatOperation::Literal(index, end));
            index = end;
        }
    }
    Ok(operations)
}

/// The direct writer of each of `operations` that has one.
fn fast_conversions(operations: &[FormatOperation]) -> Vec<Option<FastConversion>> {
    operations
        .iter()
        .map(|operation| match operation {
            FormatOperation::Conversion(specification) => FastConversion::of(specification),
            _ => None,
        })
        .collect()
}

fn format_with_plan(
    arguments: &[Value],
    plan: &FormatPlan,
    modern_quote: bool,
    numbers: NumberFormat,
    numerals: NonFiniteNumerals,
) -> RuntimeResult<Value> {
    format_operations(
        arguments,
        plan.source.as_bytes(),
        &plan.operations,
        &plan.fast,
        modern_quote,
        numbers,
        numerals,
    )
}

fn format_operations(
    arguments: &[Value],
    source: &[u8],
    operations: &[FormatOperation],
    fast: &[Option<FastConversion>],
    modern_quote: bool,
    numbers: NumberFormat,
    numerals: NonFiniteNumerals,
) -> RuntimeResult<Value> {
    let mut result = take_format_buffer();
    result.reserve(source.len());
    let mut argument_index = 1;
    for (index, operation) in operations.iter().enumerate() {
        match operation {
            FormatOperation::Literal(start, end) => {
                append_short(&mut result, &source[*start..*end])?;
            }
            FormatOperation::Percent => append_format_bytes(&mut result, b"%")?,
            FormatOperation::Conversion(specification) => {
                let argument = arguments
                    .get(argument_index)
                    .ok_or_else(|| missing_format_argument(numbers, argument_index + 1))?;
                argument_index += 1;
                if let Some(Some(conversion)) = fast.get(index)
                    && conversion.append(&mut result, argument)
                {
                    continue;
                }
                let argument = float_operand(argument, specification, numerals);
                append_format_value(&mut result, &argument, specification, modern_quote, numbers)
                    .map_err(|error| format_error_at(error, argument_index))?;
            }
        }
    }
    Ok(finish_format_buffer(result))
}

/// A constant format string of a traced `string.format`, parsed once when
/// the trace compiles. Unlike the one-entry plan cache it holds owned bytes,
/// so it is `Send + Sync` and a compiled trace can own it.
pub struct PreparedFormat {
    source: Vec<u8>,
    operations: Vec<FormatOperation>,
    /// The direct writer of each operation that has one, in step with
    /// `operations`.
    fast: Vec<Option<FastConversion>>,
    /// Whether a conversion is `%q` or `%p`, which the traced call leaves to
    /// the generic path.
    generic_only: bool,
}

impl PreparedFormat {
    /// The plan for `source`, or `None` when it does not parse (the generic
    /// path then raises its error). `%p` parses as in the profiles that
    /// support it; the profiles that do not still go through the generic
    /// path for it, so they raise as before.
    #[must_use]
    pub fn new(source: &[u8]) -> Option<Self> {
        let operations = parse_format_operations(source, true).ok()?;
        let generic_only = operations.iter().any(|operation| {
            matches!(
                operation,
                FormatOperation::Conversion(specification)
                    if matches!(specification.conversion, b'q' | b'p')
            )
        });
        let fast = fast_conversions(&operations);
        Some(Self {
            source: source.to_vec(),
            operations,
            fast,
            generic_only,
        })
    }

    /// Whether the traced call must use the generic path.
    #[must_use]
    pub fn generic_only(&self) -> bool {
        self.generic_only
    }

    /// The positions in the argument list (the format is position 0) of the
    /// `%s` conversions.
    pub fn string_arguments(&self) -> impl Iterator<Item = usize> {
        self.operations
            .iter()
            .filter_map(|operation| match operation {
                FormatOperation::Conversion(specification) => Some(specification.conversion),
                _ => None,
            })
            .enumerate()
            .filter(|(_, conversion)| *conversion == b's')
            .map(|(position, _)| position + 1)
    }

    /// Formats `arguments` (the format first) as the native of the profile
    /// does; `None` where it would raise, so the caller runs the generic
    /// path.
    #[must_use]
    pub fn format(
        &self,
        arguments: &[Value],
        modern_quote: bool,
        numbers: NumberFormat,
        numerals: NonFiniteNumerals,
    ) -> Option<Value> {
        format_operations(
            arguments,
            &self.source,
            &self.operations,
            &self.fast,
            modern_quote,
            numbers,
            numerals,
        )
        .ok()
    }
}

/// Attributes a conversion's error to argument `position`. Errors that
/// already carry the position zero are placeholders for it.
fn format_error_at(error: RuntimeError, position: usize) -> RuntimeError {
    match error {
        RuntimeError::ArgumentError {
            position: 0,
            message,
        } => RuntimeError::argument_error(position, message),
        error => error.at_argument(position),
    }
}

/// `string.format`'s error for a missing argument: `luaL_argerror`'s "no
/// value", or "value expected" as `LuaJIT`, the profile of the `LuaJit`
/// number format, words it.
fn missing_format_argument(numbers: NumberFormat, position: usize) -> RuntimeError {
    RuntimeError::argument_error(
        position,
        if numbers == NumberFormat::LuaJit {
            "value expected"
        } else {
            "no value"
        },
    )
}

/// A `%e`, `%f` or `%g` operand. These convert a string as other library
/// numbers do, so Lua 5.1 and `LuaJIT` format `"inf"` and `"nan"`.
fn float_operand<'a>(
    argument: &'a Value,
    specification: &crate::stdlib::FormatSpecification,
    numerals: NonFiniteNumerals,
) -> Cow<'a, Value> {
    if let Value::String(text) = argument
        && matches!(specification.conversion, b'e' | b'E' | b'f' | b'g' | b'G')
        && let Some(number) = numerals.parse(text.as_bytes())
    {
        Cow::Owned(Value::Number(number))
    } else {
        Cow::Borrowed(argument)
    }
}

/// Returns a byte string in reverse byte order.
///
/// # Errors
///
/// Returns an error unless the first argument is string- or number-like.
pub fn string_reverse(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let source = string_argument_in(runtime.number_format(), arguments.first(), 1)?;
    let mut reversed = source.to_vec();
    reversed.reverse();
    Ok(vec![Value::from(reversed)])
}

/// Maps ASCII lower-case bytes to upper case in the C locale.
///
/// # Errors
///
/// Returns an error unless the first argument is string- or number-like.
pub fn string_upper(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    string_ascii_case(runtime.number_format(), arguments, true)
}

/// Maps ASCII upper-case bytes to lower case in the C locale.
///
/// # Errors
///
/// Returns an error unless the first argument is string- or number-like.
pub fn string_lower(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    string_ascii_case(runtime.number_format(), arguments, false)
}

pub(crate) fn string_sub_scalar(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Value> {
    let source = string_argument_in(runtime.number_format(), arguments.first(), 1)?;
    let length = isize::try_from(source.len())
        .map_err(|_| RuntimeError::message("string.sub input is too long"))?;
    let start = string_index_argument(runtime, arguments.get(1), 2)?;
    let end = match arguments.get(2) {
        Some(_) => string_index_argument(runtime, arguments.get(2), 3)?,
        None => -1,
    };
    Ok(match substring_range(length, start, end) {
        Some(range) => Value::from(&source[range]),
        None => Value::from(""),
    })
}

/// Repeats a string a non-negative integral number of times.
///
/// # Errors
///
/// Returns an error when arguments have the wrong type, the count is negative
/// or fractional, or the resulting allocation would overflow or fail.
pub fn string_rep(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    string_rep_inner(runtime, arguments, false)
}

pub(crate) fn string_rep_with_separator(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    string_rep_inner(runtime, arguments, true)
}

pub(crate) fn string_rep_inner(
    runtime: &Runtime,
    arguments: &[Value],
    separator_supported: bool,
) -> RuntimeResult<Vec<Value>> {
    let numbers = runtime.number_format();
    let source = string_argument_in(numbers, arguments.first(), 1)?;
    let count = string_index_argument(runtime, arguments.get(1), 2)?;
    // Stock Lua returns the empty string for a count below one.
    if count < 1 {
        return Ok(vec![Value::from("")]);
    }
    let count = usize::try_from(count)
        .map_err(|_| RuntimeError::message("string.rep count is too large"))?;
    if count == 0 {
        return Ok(vec![Value::from("")]);
    }
    let separator = if separator_supported {
        match arguments.get(2) {
            None | Some(Value::Nil) => Cow::Borrowed(&[][..]),
            Some(_) => string_argument_in(numbers, arguments.get(2), 3)?,
        }
    } else {
        Cow::Borrowed(&[][..])
    };
    // Stock Lua bounds `(#s + #sep) * n` rather than the exact result, so
    // an empty string with a separator fails like it does upstream instead
    // of allocating gigabytes.
    let unit = source
        .len()
        .checked_add(separator.len())
        .ok_or_else(|| RuntimeError::message("string.rep result is too large"))?;
    if unit > i32::MAX as usize / count {
        return Err(RuntimeError::message("string.rep result is too large"));
    }
    let source_capacity = source
        .len()
        .checked_mul(count)
        .ok_or_else(|| RuntimeError::message("string.rep result is too large"))?;
    let separator_capacity = separator
        .len()
        .checked_mul(count - 1)
        .ok_or_else(|| RuntimeError::message("string.rep result is too large"))?;
    let capacity = source_capacity
        .checked_add(separator_capacity)
        .ok_or_else(|| RuntimeError::message("string.rep result is too large"))?;
    if capacity >= i32::MAX as usize {
        return Err(RuntimeError::message("string.rep result is too large"));
    }
    let mut result = Vec::new();
    result
        .try_reserve_exact(capacity)
        .map_err(|_| RuntimeError::message("string.rep result is too large"))?;
    result.extend_from_slice(&source);
    for _ in 1..count {
        result.extend_from_slice(&separator);
        result.extend_from_slice(&source);
    }
    Ok(vec![Value::from(result)])
}

/// Coerces a string-or-number argument with the default Lua 5.1 number
/// spelling. Only for arguments where a number is never meaningful, such
/// as module names; library functions use [`string_argument_in`].
pub(crate) fn string_argument(
    argument: Option<&Value>,
    position: usize,
) -> RuntimeResult<Cow<'_, [u8]>> {
    string_argument_in(NumberFormat::default(), argument, position)
}

/// Coerces a string-or-number argument, spelling numbers as `tostring`
/// does in the runtime's profile.
pub(crate) fn string_argument_in(
    numbers: NumberFormat,
    argument: Option<&Value>,
    position: usize,
) -> RuntimeResult<Cow<'_, [u8]>> {
    match argument {
        Some(Value::String(value)) => Ok(Cow::Borrowed(value.as_bytes())),
        Some(Value::Integer(value)) => Ok(Cow::Owned(format_integer(*value).to_vec())),
        Some(Value::Number(value)) => Ok(Cow::Owned(numbers.format(*value).to_vec())),
        argument => Err(RuntimeError::bad_argument(position, "string", argument)),
    }
}

/// [`string_argument_in`] as a string value: a string argument is shared, not
/// copied, and a number is spelled into a new string.
pub(crate) fn string_value_in(
    numbers: NumberFormat,
    argument: Option<&Value>,
    position: usize,
) -> RuntimeResult<LuaString> {
    match argument {
        Some(Value::String(value)) => Ok(value.clone()),
        argument => string_argument_in(numbers, argument, position)
            .map(|bytes| LuaString::from(bytes.into_owned())),
    }
}

/// A plain `string.find` of string operands whose start index is already
/// known, with the results the general path produces for them.
fn plain_find_values(
    runtime: &mut Runtime,
    source: &[u8],
    pattern: &[u8],
    initial: isize,
) -> Vec<Value> {
    let pair = plain_find_pair(runtime, source, pattern, initial);
    let mut values = runtime.result_vec();
    values.extend(pair.as_slice().iter().cloned());
    values
}

fn plain_find_pair(runtime: &Runtime, source: &[u8], pattern: &[u8], initial: isize) -> NativePair {
    // Allocations never exceed `isize::MAX` bytes.
    #[allow(clippy::cast_possible_wrap, reason = "slice lengths fit in isize")]
    let length = source.len() as isize;
    let mut initial = relative_index(initial, length);
    if initial > length.saturating_add(1) {
        // Lua 5.1 and LuaJIT start a search past the end at the end.
        if !runtime.string_init_clamps() {
            return NativePair::one(Value::Nil);
        }
        initial = length.saturating_add(1);
    }
    #[allow(clippy::cast_sign_loss, reason = "clamped to at least one above")]
    let initial = (initial.max(1) - 1) as usize;
    let Some(start) = plain_find(source, pattern, initial) else {
        return NativePair::one(Value::Nil);
    };
    let position = |position: usize| {
        if runtime.tagged_integers() {
            // Positions are bounded by a slice length, so they fit.
            #[allow(clippy::cast_possible_wrap, reason = "slice positions fit in i64")]
            Value::Integer(position as i64)
        } else {
            #[allow(clippy::cast_precision_loss, reason = "LuaNumber is f64")]
            Value::Number(position as f64)
        }
    };
    NativePair::two(position(start + 1), position(start + pattern.len()))
}

/// An absent, nil, or small integral index argument, which converts
/// exactly as [`optional_coerced_index`] would; anything else is `None`.
#[inline]
fn small_index(argument: &Value, default: isize) -> Option<isize> {
    match argument {
        Value::Nil => Some(default),
        Value::Number(number) => {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "the round trip below rejects every inexact conversion"
            )]
            let small = *number as i32;
            #[allow(clippy::float_cmp, reason = "an exact round trip is the test")]
            let exact = f64::from(small) == *number;
            exact.then_some(small as isize)
        }
        Value::Integer(integer) => i32::try_from(*integer).ok().map(|small| small as isize),
        _ => None,
    }
}

/// Optional argument `position`, a number converted to a string index.
pub(crate) fn optional_coerced_index(
    runtime: &Runtime,
    argument: Option<&Value>,
    position: usize,
    default: isize,
) -> RuntimeResult<isize> {
    let Some(value) = argument.filter(|value| !matches!(value, Value::Nil)) else {
        return Ok(default);
    };
    // Small integral indices, by far the common case, convert exactly
    // without the truncation and range checks below.
    if let Some(index) = small_index(value, default) {
        return Ok(index);
    }
    if let Value::Integer(integer) = value {
        return Ok(crate::stdlib::args::saturating_index(*integer));
    }
    let number = runtime
        .to_number(value)
        .ok_or_else(|| RuntimeError::bad_argument(position, "number", Some(value)))?;
    float_string_index(runtime, number, position)
}

#[allow(clippy::needless_pass_by_value)]
pub(crate) fn pattern_error(
    runtime: &Runtime,
    error: crate::pattern::PatternError,
) -> RuntimeError {
    RuntimeError::message(error.lua_message(runtime.legacy_pattern_errors()))
}

pub(crate) fn plain_find(source: &[u8], pattern: &[u8], initial: usize) -> Option<usize> {
    if pattern.is_empty() {
        return Some(initial);
    }
    let haystack = source.get(initial..)?;
    // Scan on the first byte and compare full candidates. Rebuilding
    // `memchr::memmem`'s ranked finder on every call costs more than the
    // search itself for the short needles Lua programs pass most often.
    let first = pattern[0];
    let rest = &pattern[1..];
    for position in memchr::memchr_iter(first, haystack) {
        if haystack[position + 1..].starts_with(rest) {
            return Some(position + initial);
        }
    }
    None
}

pub(crate) fn pattern_captures(
    runtime: &Runtime,
    source: &[u8],
    found: &PatternMatch,
) -> RuntimeResult<Vec<Value>> {
    found
        .captures()
        .iter()
        .map(|capture| {
            if capture.is_position() {
                usize_integer_value(
                    runtime,
                    capture.start() + 1,
                    "string capture position is too large",
                )
            } else {
                Ok(Value::from(&source[capture.start()..capture.end()]))
            }
        })
        .collect()
}

pub(crate) fn parse_lua_number(source: &[u8]) -> Option<f64> {
    match scan_plain_numeral(source) {
        PlainNumeral::Whole {
            negative,
            magnitude,
        } => {
            #[allow(clippy::cast_precision_loss, reason = "below 2^53, so exact")]
            let value = magnitude as f64;
            return Some(if negative { -value } else { value });
        }
        PlainNumeral::Fraction(value) => return Some(value),
        PlainNumeral::Invalid => return None,
        PlainNumeral::Other => {}
    }
    let source = trim_ascii(source);
    let numeric_body = source
        .strip_prefix(b"+")
        .or_else(|| source.strip_prefix(b"-"))
        .unwrap_or(source);
    if matches!(numeric_body.first(), Some(b'0'..=b'9' | b'.'))
        && let Ok(text) = std::str::from_utf8(source)
        && let Ok(value) = text.parse::<f64>()
    {
        return Some(value);
    }
    parse_hex_number(source)
}

pub(crate) use crate::runtime::parse_lua_integer_exact;

pub(crate) fn parse_hex_number(source: &[u8]) -> Option<f64> {
    pub(crate) const MAX_SIGNIFICAND_DIGITS: usize = 16;

    let source = trim_ascii(source);
    let (negative, source) = match source.first() {
        Some(b'-') => (true, &source[1..]),
        Some(b'+') => (false, &source[1..]),
        _ => (false, source),
    };
    let source = source
        .strip_prefix(b"0x")
        .or_else(|| source.strip_prefix(b"0X"))?;
    let (source_significand, exponent) =
        match source.iter().position(|byte| matches!(byte, b'p' | b'P')) {
            Some(index) => (
                &source[..index],
                parse_decimal_exponent(&source[index + 1..])?,
            ),
            None => (source, 0),
        };
    // Keep only the leading significand digits and account for the rest in
    // the binary exponent. Accumulating every digit first can overflow before
    // a negative `p` exponent brings the final result back into range.
    let mut leading_significand = 0.0;
    let mut significand_digits = 0_usize;
    let mut first_significant = None;
    let mut integer_digits = 0_usize;
    let mut digit_index = 0_usize;
    let mut fraction = false;
    for byte in source_significand {
        if *byte == b'.' && !fraction {
            fraction = true;
            continue;
        }
        let digit = digit_value(*byte)?;
        if digit >= 16 {
            return None;
        }
        if !fraction {
            integer_digits += 1;
        }
        if digit != 0 && first_significant.is_none() {
            first_significant = Some(digit_index);
        }
        if first_significant.is_some() && significand_digits < MAX_SIGNIFICAND_DIGITS {
            leading_significand = leading_significand * 16.0 + f64::from(digit);
            significand_digits += 1;
        }
        digit_index += 1;
    }
    if digit_index == 0 {
        return None;
    }

    let value = first_significant.map_or(0.0, |first| {
        let integer_digits = i64::try_from(integer_digits).unwrap_or(i64::MAX);
        let first = i64::try_from(first).unwrap_or(i64::MAX);
        let significand_digits = i64::try_from(significand_digits).unwrap_or(i64::MAX);
        let exponent = i64::from(exponent)
            .saturating_add(
                integer_digits
                    .saturating_sub(first)
                    .saturating_sub(1)
                    .saturating_mul(4),
            )
            .saturating_sub(significand_digits.saturating_sub(1).saturating_mul(4));
        if exponent > i64::from(i32::MAX) {
            f64::INFINITY
        } else if exponent < i64::from(i32::MIN) {
            0.0
        } else {
            let exponent = i32::try_from(exponent).expect("exponent was checked to fit i32");
            leading_significand * 2_f64.powi(exponent)
        }
    });
    Some(if negative { -value } else { value })
}

pub(crate) fn parse_decimal_exponent(source: &[u8]) -> Option<i32> {
    let text = std::str::from_utf8(source).ok()?;
    text.parse().ok()
}

pub(crate) fn parse_integer_base(source: &[u8], base: isize) -> Option<f64> {
    let source = trim_c_space(source);
    let (negative, source) = match source.first() {
        Some(b'-') => (true, &source[1..]),
        Some(b'+') => (false, &source[1..]),
        _ => (false, source),
    };
    if source.is_empty() {
        return None;
    }
    let base = u8::try_from(base).ok()?;
    let mut value = 0.0;
    for byte in source {
        let digit = digit_value(*byte)?;
        if digit >= base {
            return None;
        }
        value = value * f64::from(base) + f64::from(digit);
    }
    Some(if negative { -value } else { value })
}

/// Parses `tonumber(s, base)` as Lua 5.1 and `LuaJIT` do: C `strtoul` over the
/// string up to its first NUL, then a check that only whitespace follows.
///
/// `strtoul` skips leading whitespace (including `\v` and `\f`), takes one
/// sign, accepts a `0x` prefix in base 16 and a `0b` prefix in base 2 (glibc),
/// and saturates at `ULONG_MAX` without applying the sign. Lua 5.1 keeps the
/// unsigned result, so a negative numeral wraps to `2^64 - n`. `LuaJIT` negates
/// the converted double instead, so `-0` stays negative zero.
pub(crate) fn parse_strtoul_base(source: &[u8], base: isize, luajit: bool) -> Option<f64> {
    // Digits of the base and nothing else, too few to overflow (36^12 is
    // below 2^63): no whitespace, sign, prefix or NUL to look for.
    if (1..=12).contains(&source.len())
        && let Ok(base) = u8::try_from(base)
    {
        let mut value = 0_u64;
        let mut plain = true;
        for byte in source {
            match digit_value(*byte) {
                Some(digit) if digit < base => {
                    value = value * u64::from(base) + u64::from(digit);
                }
                _ => {
                    plain = false;
                    break;
                }
            }
        }
        if plain {
            #[allow(clippy::cast_precision_loss, reason = "as the general path converts")]
            return Some(value as f64);
        }
    }
    let source = source.split(|&byte| byte == 0).next().unwrap_or_default();
    let is_space = |byte: &u8| matches!(byte, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r');
    let start = source.iter().position(|byte| !is_space(byte))?;
    let mut rest = &source[start..];
    let negative = match rest.first() {
        Some(b'-') => {
            rest = &rest[1..];
            true
        }
        Some(b'+') => {
            rest = &rest[1..];
            false
        }
        _ => false,
    };
    let base = u8::try_from(base).ok()?;
    let digit_in_base = |byte: Option<&u8>| {
        byte.and_then(|byte| digit_value(*byte))
            .is_some_and(|digit| digit < base)
    };
    let prefix = match base {
        16 => *b"xX",
        2 => *b"bB",
        _ => [0, 0],
    };
    if rest.first() == Some(&b'0')
        && rest.get(1).is_some_and(|byte| prefix.contains(byte))
        && digit_in_base(rest.get(2))
    {
        rest = &rest[2..];
    }
    let digits = rest
        .iter()
        .take_while(|byte| digit_value(**byte).is_some_and(|digit| digit < base))
        .count();
    if digits == 0 || !rest[digits..].iter().all(is_space) {
        return None;
    }
    let mut value = 0_u64;
    let mut overflow = false;
    for byte in &rest[..digits] {
        let digit = u64::from(digit_value(*byte)?);
        if let Some(next) = value
            .checked_mul(u64::from(base))
            .and_then(|value| value.checked_add(digit))
        {
            value = next;
        } else {
            overflow = true;
            value = u64::MAX;
            break;
        }
    }
    Some(if !negative {
        value as f64
    } else if luajit {
        -(value as f64)
    } else if overflow {
        value as f64
    } else {
        value.wrapping_neg() as f64
    })
}

/// Parses `tonumber(s, base)` as Lua 5.3+ does, accumulating in a
/// `lua_Integer` that wraps on overflow.
pub(crate) fn parse_wrapping_integer_base(source: &[u8], base: isize) -> Option<i64> {
    let source = trim_c_space(source);
    let (negative, source) = match source.first() {
        Some(b'-') => (true, &source[1..]),
        Some(b'+') => (false, &source[1..]),
        _ => (false, source),
    };
    if source.is_empty() {
        return None;
    }
    let base = u8::try_from(base).ok()?;
    let mut value = 0_u64;
    for byte in source {
        let digit = digit_value(*byte)?;
        if digit >= base {
            return None;
        }
        value = value
            .wrapping_mul(u64::from(base))
            .wrapping_add(u64::from(digit));
    }
    let value = value.cast_signed();
    Some(if negative {
        value.wrapping_neg()
    } else {
        value
    })
}

/// Trims the bytes C `isspace` accepts, which include `\v` unlike
/// [`u8::is_ascii_whitespace`].
fn trim_c_space(mut bytes: &[u8]) -> &[u8] {
    let is_space = |byte: &u8| byte.is_ascii_whitespace() || *byte == 0x0b;
    while bytes.first().is_some_and(is_space) {
        bytes = &bytes[1..];
    }
    while bytes.last().is_some_and(is_space) {
        bytes = &bytes[..bytes.len() - 1];
    }
    bytes
}

pub(crate) fn trim_ascii(mut bytes: &[u8]) -> &[u8] {
    while bytes.first().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[1..];
    }
    while bytes.last().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[..bytes.len() - 1];
    }
    bytes
}

pub(crate) fn digit_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'z' => Some(byte - b'a' + 10),
        b'A'..=b'Z' => Some(byte - b'A' + 10),
        _ => None,
    }
}
