//! String byte, char, and pack/unpack builtins with pack-format parsing.

use crate::runtime::{NativeResults, Runtime, RuntimeError, RuntimeResult, Value};
use crate::stdlib::args::{number_to_isize, substring_range, usize_integer_value};
use crate::stdlib::strings::{
    ResultList, optional_coerced_index, put_position, string_argument_in, vec_of,
};
use crate::stdlib::{PackItem, PackKind};
use crate::string::LuaString;
use std::rc::Rc;

pub(crate) fn string_len_scalar(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Value> {
    let source = string_argument_in(runtime.number_format(), arguments.first(), 1)?;
    usize_integer_value(runtime, source.len(), "string.len input is too long")
}

/// Returns bytes from an inclusive, one-based string range as separate values.
///
/// # Errors
///
/// Returns an error unless the source is string- or number-like and supplied
/// indexes follow the runtime numeric coercion rule.
pub fn string_byte(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let source = string_argument_in(runtime.number_format(), arguments.first(), 1)?;
    let length = isize::try_from(source.len())
        .map_err(|_| RuntimeError::message("string.byte input is too long"))?;
    let raw_start = optional_coerced_index(runtime, arguments.get(1), 2, 1)?;
    let raw_end = optional_coerced_index(runtime, arguments.get(2), 3, raw_start)?;
    let Some(range) = substring_range(length, raw_start, raw_end) else {
        return Ok(runtime.result_vec());
    };
    let count = range.len();
    // The runtime's recycled results vector: a call of a few bytes allocates
    // nothing in a steady state.
    let mut bytes = runtime.result_vec();
    bytes
        .try_reserve_exact(count)
        .map_err(|_| RuntimeError::message("string.byte result is too large"))?;
    // One check of the profile's number representation, not one per byte.
    let slice = &source[range];
    if matches!(runtime.integer_value(0), Value::Integer(_)) {
        for byte in slice {
            bytes.push(Value::Integer(i64::from(*byte)));
        }
    } else {
        for byte in slice {
            bytes.push(Value::Number(f64::from(*byte)));
        }
    }
    Ok(bytes)
}

/// Creates a byte string from zero or more byte values.
///
/// # Errors
///
/// Returns an error unless every argument follows the runtime numeric
/// coercion rule and is an integer in the inclusive range 0 through 255.
pub fn string_char(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    string_char_with(runtime, arguments, "value out of range")
}

/// [`string_char`] with the message of Lua 5.1 and `LuaJIT` for a byte outside
/// 0 through 255.
pub(crate) fn string_char_51(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    string_char_with(runtime, arguments, "invalid value")
}

fn string_char_with(
    runtime: &mut Runtime,
    arguments: &[Value],
    out_of_range: &'static str,
) -> RuntimeResult<Vec<Value>> {
    // A few bytes, the usual case, are gathered on the stack and copied once
    // into their string rather than through a vector that is then freed.
    let mut small = [0_u8; 32];
    let mut large = Vec::new();
    let on_heap = arguments.len() > small.len();
    if on_heap {
        large
            .try_reserve_exact(arguments.len())
            .map_err(|_| RuntimeError::message("string.char result is too large"))?;
    }
    for (index, argument) in arguments.iter().enumerate() {
        let value = runtime
            .to_number(argument)
            .ok_or_else(|| RuntimeError::bad_argument(index + 1, "number", Some(argument)))?;
        let value = number_to_isize(value.trunc(), "string.char value")?;
        let value = u8::try_from(value)
            .map_err(|_| RuntimeError::argument_error(index + 1, out_of_range))?;
        if on_heap {
            large.push(value);
        } else {
            small[index] = value;
        }
    }
    let text = if on_heap {
        Value::from(large)
    } else {
        Value::from(&small[..arguments.len()])
    };
    Ok(vec![text])
}

pub(crate) fn string_pack(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    vec_of(|values| pack(runtime, arguments, values))
}

/// What `string.pack` gives, for a trace to learn before the call.
pub(crate) fn string_pack_probe(runtime: &Runtime, arguments: &[Value]) -> Option<Vec<Value>> {
    vec_of(|values| pack(runtime, arguments, values)).ok()
}

/// [`string_pack_probe`] without a list for the one result.
pub(crate) fn string_pack_fixed_probe(
    runtime: &Runtime,
    arguments: &[Value],
    results: &mut NativeResults,
) -> bool {
    pack(runtime, arguments, results).is_ok()
}

/// The bytes `string.pack` builds. A record of a few dozen bytes, which is
/// what most calls build, stays in place; a longer one moves to a `Vec`.
pub(crate) struct PackBuffer {
    small: [u8; PackBuffer::SMALL],
    len: usize,
    large: Option<Vec<u8>>,
}

impl PackBuffer {
    const SMALL: usize = 48;

    pub(crate) fn new() -> Self {
        Self {
            small: [0; Self::SMALL],
            len: 0,
            large: None,
        }
    }

    pub(crate) fn len(&self) -> usize {
        match &self.large {
            Some(large) => large.len(),
            None => self.len,
        }
    }

    /// Moves the bytes to a `Vec` with room for `more` bytes, unless they
    /// are in one already.
    fn large(&mut self, more: usize) -> RuntimeResult<&mut Vec<u8>> {
        if self.large.is_none() {
            let mut large = Vec::new();
            large
                .try_reserve(self.len.saturating_add(more).max(2 * Self::SMALL))
                .map_err(|_| RuntimeError::message("format too large"))?;
            large.extend_from_slice(&self.small[..self.len]);
            self.large = Some(large);
        }
        Ok(self.large.as_mut().expect("moved above"))
    }

    /// Appends the low `size` bytes of `word`, least significant first.
    /// `size` is at most 8 and the bytes of `word` above it are zero.
    ///
    /// The whole word is stored where there is room for it, which takes no
    /// copy loop for a width only the format knows. The bytes past the new
    /// end are then the zeros of the word, as [`Self::pad`] expects.
    #[inline]
    pub(crate) fn push_word(&mut self, word: u64, size: usize) -> RuntimeResult<()> {
        debug_assert!(size <= 8 && (size == 8 || word >> (8 * size) == 0));
        if self.large.is_none() && self.len + 8 <= Self::SMALL {
            self.small[self.len..self.len + 8].copy_from_slice(&word.to_le_bytes());
            self.len += size;
            return Ok(());
        }
        self.extend_from_slice(&word.to_le_bytes()[..size])
    }

    #[inline]
    pub(crate) fn extend_from_slice(&mut self, bytes: &[u8]) -> RuntimeResult<()> {
        if self.large.is_none() && self.len + bytes.len() <= Self::SMALL {
            self.small[self.len..self.len + bytes.len()].copy_from_slice(bytes);
            self.len += bytes.len();
            return Ok(());
        }
        self.extend_large(bytes)
    }

    #[cold]
    #[inline(never)]
    fn extend_large(&mut self, bytes: &[u8]) -> RuntimeResult<()> {
        let large = self.large(bytes.len())?;
        large
            .try_reserve(bytes.len())
            .map_err(|_| RuntimeError::message("format too large"))?;
        large.extend_from_slice(bytes);
        Ok(())
    }

    /// Appends `count` zero bytes.
    #[inline]
    pub(crate) fn pad(&mut self, count: usize) -> RuntimeResult<()> {
        if self.large.is_none() && count <= Self::SMALL - self.len {
            // The bytes past `len` are still the zeros of `new`.
            self.len += count;
            return Ok(());
        }
        self.pad_large(count)
    }

    #[cold]
    #[inline(never)]
    fn pad_large(&mut self, count: usize) -> RuntimeResult<()> {
        let large = self.large(count)?;
        large
            .try_reserve(count)
            .map_err(|_| RuntimeError::message("format too large"))?;
        large.resize(large.len() + count, 0);
        Ok(())
    }

    /// The bytes as a string. Borrowed, not consumed: moving the buffer
    /// would copy the bytes just written with wider loads than their stores.
    pub(crate) fn finish(&mut self) -> LuaString {
        match &mut self.large {
            Some(large) => LuaString::from(std::mem::take(large)),
            None => LuaString::new(&self.small[..self.len]),
        }
    }
}

#[inline]
fn pack<R: ResultList>(
    runtime: &Runtime,
    arguments: &[Value],
    results: &mut R,
) -> RuntimeResult<()> {
    results.put(Value::String(pack_text(runtime, arguments)?))
}

/// The string `string.pack (format, values...)` builds from `arguments`.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Lua's pack formats explicitly request these numeric widths.
pub(crate) fn pack_text(runtime: &Runtime, arguments: &[Value]) -> RuntimeResult<LuaString> {
    let format = string_argument_in(runtime.number_format(), arguments.first(), 1)?;
    let items = cached_pack_format(&format)?;
    let mut output = PackBuffer::new();
    let mut argument = 1;
    for item in items.iter().copied() {
        if item.align > 1 {
            pack_align(&mut output, item.align)?;
        }
        match item.kind {
            PackKind::Align => {}
            PackKind::Pad => output.pad(1)?,
            PackKind::Int { signed, size } => {
                let value = arguments.get(argument);
                if let Some(whole) = value.and_then(small_whole)
                    && whole_fits(whole, signed, size)
                {
                    push_whole(&mut output, whole, size, item.endian)?;
                } else {
                    let value = pack_number(value, argument + 1)?;
                    pack_integer(&mut output, value, signed, size, item.endian)?;
                }
                argument += 1;
            }
            PackKind::Float(size) => {
                let value = match arguments.get(argument) {
                    Some(Value::Number(value)) => *value,
                    value => pack_number(value, argument + 1)?,
                };
                argument += 1;
                let (bits, size) = if size == 4 {
                    (u64::from((value as f32).to_bits()), 4)
                } else {
                    (value.to_bits(), 8)
                };
                // The bytes of a float in the order of the item.
                let bits = if item.endian {
                    bits
                } else {
                    bits.swap_bytes() >> (64 - 8 * size)
                };
                output.push_word(bits, size)?;
            }
            PackKind::Fixed(size) => {
                let value = string_argument_in(
                    runtime.number_format(),
                    arguments.get(argument),
                    argument + 1,
                )?;
                argument += 1;
                if value.len() > size {
                    return Err(RuntimeError::message("string longer than given size"));
                }
                output.extend_from_slice(&value)?;
                output.pad(size - value.len())?;
            }
            PackKind::Zero => {
                let value = string_argument_in(
                    runtime.number_format(),
                    arguments.get(argument),
                    argument + 1,
                )?;
                argument += 1;
                if value.contains(&0) {
                    return Err(RuntimeError::message("string contains zeros"));
                }
                output.extend_from_slice(&value)?;
                output.pad(1)?;
            }
            PackKind::Sized(size) => {
                let value = string_argument_in(
                    runtime.number_format(),
                    arguments.get(argument),
                    argument + 1,
                )?;
                argument += 1;
                match i64::try_from(value.len()) {
                    Ok(length) if whole_fits(length, false, size) => {
                        push_whole(&mut output, length, size, item.endian)?;
                    }
                    _ => pack_integer(&mut output, value.len() as f64, false, size, item.endian)?,
                }
                output.extend_from_slice(&value)?;
            }
        }
    }
    if argument != arguments.len() {
        return Err(RuntimeError::message("extra arguments to 'pack'"));
    }
    Ok(output.finish())
}

pub(crate) fn string_packsize(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    packsize(runtime, arguments)
}

/// What `string.packsize` gives, for a trace to learn before the call.
pub(crate) fn string_packsize_probe(runtime: &Runtime, arguments: &[Value]) -> Option<Vec<Value>> {
    packsize(runtime, arguments).ok()
}

fn packsize(runtime: &Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let format = string_argument_in(runtime.number_format(), arguments.first(), 1)?;
    let items = cached_pack_format(&format)?;
    let mut bytes = PackBuffer::new();
    for item in items.iter().copied() {
        pack_align(&mut bytes, item.align)?;
        match item.kind {
            PackKind::Align => {}
            PackKind::Pad => bytes.pad(1)?,
            PackKind::Int { size, .. } | PackKind::Float(size) | PackKind::Fixed(size) => {
                bytes.pad(size)?;
            }
            PackKind::Zero | PackKind::Sized(_) => {
                return Err(RuntimeError::message("variable-length format"));
            }
        }
    }
    Ok(vec![usize_integer_value(
        runtime,
        bytes.len(),
        "format too large",
    )?])
}

pub(crate) fn string_unpack(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    unpack_values(runtime, arguments, false)
}

/// `string.unpack` of Lua 5.4 and 5.5, whose initial position clamps to the
/// string (`posrelatI`) where Lua 5.3's raises an error for zero and for a
/// negative one that starts before it.
pub(crate) fn string_unpack_clamped(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    unpack_values(runtime, arguments, true)
}

/// [`string_unpack_probe`] without a list for the results.
pub(crate) fn string_unpack_fixed_probe(
    runtime: &Runtime,
    arguments: &[Value],
    results: &mut NativeResults,
) -> bool {
    unpack(runtime, arguments, false, results).is_ok()
}

/// [`string_unpack_clamped_probe`] without a list for the results.
pub(crate) fn string_unpack_clamped_fixed_probe(
    runtime: &Runtime,
    arguments: &[Value],
    results: &mut NativeResults,
) -> bool {
    unpack(runtime, arguments, true, results).is_ok()
}

/// What `string.unpack` gives, for a trace to learn before the call.
pub(crate) fn string_unpack_probe(runtime: &Runtime, arguments: &[Value]) -> Option<Vec<Value>> {
    unpack_values(runtime, arguments, false).ok()
}

/// [`string_unpack_probe`] for [`string_unpack_clamped`].
pub(crate) fn string_unpack_clamped_probe(
    runtime: &Runtime,
    arguments: &[Value],
) -> Option<Vec<Value>> {
    unpack_values(runtime, arguments, true).ok()
}

/// The values `string.unpack` gives, in a `Vec`.
fn unpack_values(
    runtime: &Runtime,
    arguments: &[Value],
    clamped: bool,
) -> RuntimeResult<Vec<Value>> {
    let mut values = Vec::with_capacity(8);
    unpack(runtime, arguments, clamped, &mut values)?;
    Ok(values)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // The decoded size is checked as a non-negative integer first.
fn unpack<R: ResultList>(
    runtime: &Runtime,
    arguments: &[Value],
    clamped: bool,
    values: &mut R,
) -> RuntimeResult<()> {
    let format = string_argument_in(runtime.number_format(), arguments.first(), 1)?;
    let source = string_argument_in(runtime.number_format(), arguments.get(1), 2)?;
    // The format is parsed before the start is converted, so that a call
    // wrong in both raises the format's error.
    let items = cached_pack_format(&format)?;
    let start = optional_coerced_index(runtime, arguments.get(2), 3, 1)?;
    unpack_items(runtime, &items, &source, start, clamped, values)
}

/// `string.unpack (format, source, start)` of a format and a source that
/// are strings and a start that is absent or an integer, as the native
/// answers it: a compiled trace calls this on the bytes it holds.
#[inline]
pub(crate) fn unpack_text<R: ResultList>(
    runtime: &Runtime,
    format: &[u8],
    source: &[u8],
    start: Option<i64>,
    clamped: bool,
    values: &mut R,
) -> RuntimeResult<()> {
    let items = cached_pack_format(format)?;
    let start = match start {
        None => 1,
        Some(start) => optional_coerced_index(runtime, Some(&Value::Integer(start)), 3, 1)?,
    };
    unpack_items(runtime, &items, source, start, clamped, values)
}

/// [`unpack`] once its format is parsed and its arguments are bytes and a
/// start index.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // The decoded size is checked as a non-negative integer first.
fn unpack_items<R: ResultList>(
    runtime: &Runtime,
    items: &[PackItem],
    source: &[u8],
    start: isize,
    clamped: bool,
    values: &mut R,
) -> RuntimeResult<()> {
    let start = if start < 0 {
        isize::try_from(source.len())
            .unwrap_or(isize::MAX)
            .saturating_add(start)
            .saturating_add(1)
    } else {
        start
    };
    let start = if clamped { start.max(1) } else { start };
    if start < 1
        || usize::try_from(start - 1)
            .ok()
            .is_none_or(|start| start > source.len())
    {
        return Err(RuntimeError::message("initial position out of string"));
    }
    let mut position = usize::try_from(start - 1)
        .map_err(|_| RuntimeError::message("initial position out of string"))?;
    for item in items.iter().copied() {
        position = align_position(position, item.align);
        match item.kind {
            PackKind::Align => {}
            PackKind::Pad => {
                position = position
                    .checked_add(1)
                    .ok_or_else(|| RuntimeError::message("data too short"))?;
            }
            PackKind::Int { signed, size } if size <= 8 => {
                // At most eight bytes are one word, which is the integer
                // itself once its sign is extended.
                let raw = unpack_word(source, &mut position, size, item.endian)?;
                let unused = 64 - 8 * size as u32;
                let whole = if signed {
                    (raw << unused).cast_signed() >> unused
                } else {
                    raw.cast_signed()
                };
                values.put(if runtime.tagged_integers() {
                    Value::Integer(whole)
                } else if signed {
                    Value::Number(whole as f64)
                } else {
                    Value::Number(raw as f64)
                })?;
            }
            PackKind::Int { signed, size } if runtime.tagged_integers() => {
                values.put(Value::Integer(unpack_lua_integer(
                    source,
                    &mut position,
                    signed,
                    size,
                    item.endian,
                )?))?;
            }
            PackKind::Int { signed, size } => {
                values.put(Value::Number(unpack_integer(
                    source,
                    &mut position,
                    signed,
                    size,
                    item.endian,
                )?))?;
            }
            PackKind::Float(size) => {
                let raw = unpack_word(source, &mut position, size, item.endian)?;
                values.put(Value::Number(if size == 4 {
                    f64::from(f32::from_bits(raw as u32))
                } else {
                    f64::from_bits(raw)
                }))?;
            }
            PackKind::Fixed(size) => values.put(Value::String(LuaString::new(take_pack_bytes(
                source,
                &mut position,
                size,
            )?)))?,
            PackKind::Zero => {
                let tail = &source[position..];
                let Some(end) = tail.iter().position(|&byte| byte == 0) else {
                    return Err(RuntimeError::message("unfinished string"));
                };
                values.put(Value::String(LuaString::new(&tail[..end])))?;
                position += end + 1;
            }
            PackKind::Sized(size) if size <= 8 => {
                let length = unpack_word(source, &mut position, size, item.endian)?;
                // A length past the address space is past the data too.
                let length = usize::try_from(length).unwrap_or(usize::MAX);
                values.put(Value::String(LuaString::new(take_pack_bytes(
                    source,
                    &mut position,
                    length,
                )?)))?;
            }
            PackKind::Sized(size) => {
                let length = unpack_integer(source, &mut position, false, size, item.endian)?;
                if length < 0.0 || length.fract() != 0.0 {
                    return Err(RuntimeError::message("invalid length"));
                }
                values.put(Value::String(LuaString::new(take_pack_bytes(
                    source,
                    &mut position,
                    length as usize,
                )?)))?;
            }
        }
        if position > source.len() {
            return Err(RuntimeError::message("data too short"));
        }
    }
    put_position(runtime, position + 1, "position too large", values)
}

/// How many formats [`cached_pack_format`] remembers.
const FORMAT_CACHE: usize = 8;

/// The longest format [`cached_pack_format`] remembers.
const FORMAT_CACHE_BYTES: usize = 64;

/// A format, its items, and when a call last asked for it.
struct CachedFormat {
    format: Vec<u8>,
    items: Rc<Vec<PackItem>>,
    used: u64,
}

/// The formats [`cached_pack_format`] remembers and the count of its calls,
/// which stamps the entry each call asks for.
struct FormatCache {
    calls: u64,
    formats: Vec<CachedFormat>,
}

thread_local! {
    static PACK_FORMATS: std::cell::RefCell<FormatCache> = const {
        std::cell::RefCell::new(FormatCache {
            calls: 0,
            formats: Vec::new(),
        })
    };
}

/// [`parse_pack_format`] of a format a program uses again and again (a loop
/// that packs records calls `string.pack` with the same few formats), parsed
/// once. A format that does not parse is no entry: it raises every time.
///
/// The entries stay where they are. A loop that packs one format and
/// unpacks two others asks for a different entry on every call, and moving
/// the one it asked for to the front each time cost more than the lookup.
/// An entry carries the count of the call that last used it instead, and a
/// new format takes the place of the one unused for longest.
pub(crate) fn cached_pack_format(format: &[u8]) -> RuntimeResult<Rc<Vec<PackItem>>> {
    if format.len() > FORMAT_CACHE_BYTES {
        return parse_pack_format(format).map(Rc::new);
    }
    PACK_FORMATS.with(|cache| {
        let mut cache = cache.borrow_mut();
        let cache = &mut *cache;
        cache.calls += 1;
        let now = cache.calls;
        for known in &mut cache.formats {
            if known.format == format {
                known.used = now;
                return Ok(Rc::clone(&known.items));
            }
        }
        let items = Rc::new(parse_pack_format(format)?);
        let entry = CachedFormat {
            format: format.to_vec(),
            items: Rc::clone(&items),
            used: now,
        };
        if cache.formats.len() < FORMAT_CACHE {
            cache.formats.push(entry);
        } else if let Some(oldest) = cache.formats.iter_mut().min_by_key(|known| known.used) {
            *oldest = entry;
        }
        Ok(items)
    })
}

pub(crate) fn parse_pack_format(format: &[u8]) -> RuntimeResult<Vec<PackItem>> {
    let mut items = Vec::new();
    let mut index = 0;
    let mut little = cfg!(target_endian = "little");
    let mut max_align = 1usize;
    while index < format.len() {
        let option = format[index];
        index += 1;
        if option.is_ascii_whitespace() {
            continue;
        }
        match option {
            b'<' => little = true,
            b'>' => little = false,
            b'=' => little = cfg!(target_endian = "little"),
            b'!' => {
                let (size, next) = pack_size(format, index, None)?;
                if size == 0 || size > 16 || !size.is_power_of_two() {
                    return Err(RuntimeError::message("alignment is out of limits"));
                }
                max_align = size;
                index = next;
            }
            b'x' => items.push(PackItem {
                kind: PackKind::Pad,
                endian: little,
                align: 1,
            }),
            b'X' => {
                if index >= format.len() {
                    return Err(RuntimeError::message("invalid next option"));
                }
                let (item, next) = pack_option(format, index, little, max_align)?;
                index = next;
                items.push(PackItem {
                    kind: PackKind::Align,
                    endian: little,
                    align: item.align,
                });
            }
            _ => {
                let (item, next) = pack_option_at(option, format, index, little, max_align)?;
                index = next;
                items.push(item);
            }
        }
    }
    Ok(items)
}

pub(crate) fn pack_option(
    format: &[u8],
    index: usize,
    little: bool,
    max: usize,
) -> RuntimeResult<(PackItem, usize)> {
    pack_option_at(
        *format
            .get(index)
            .ok_or_else(|| RuntimeError::message("invalid next option"))?,
        format,
        index + 1,
        little,
        max,
    )
}

pub(crate) fn pack_option_at(
    option: u8,
    format: &[u8],
    index: usize,
    little: bool,
    max: usize,
) -> RuntimeResult<(PackItem, usize)> {
    let (kind, next) = match option {
        b'b' => (
            PackKind::Int {
                signed: true,
                size: 1,
            },
            index,
        ),
        b'B' => (
            PackKind::Int {
                signed: false,
                size: 1,
            },
            index,
        ),
        b'h' => (
            PackKind::Int {
                signed: true,
                size: 2,
            },
            index,
        ),
        b'H' => (
            PackKind::Int {
                signed: false,
                size: 2,
            },
            index,
        ),
        b'l' | b'j' => (
            PackKind::Int {
                signed: true,
                size: 8,
            },
            index,
        ),
        b'L' | b'J' | b'T' => (
            PackKind::Int {
                signed: false,
                size: 8,
            },
            index,
        ),
        b'f' => (PackKind::Float(4), index),
        b'd' | b'n' => (PackKind::Float(8), index),
        b'z' => (PackKind::Zero, index),
        b'i' | b'I' => {
            let (size, next) = pack_size(format, index, Some(4))?;
            (
                PackKind::Int {
                    signed: option == b'i',
                    size,
                },
                next,
            )
        }
        b'c' => {
            let (size, next) = pack_size(format, index, None)?;
            (PackKind::Fixed(size), next)
        }
        b's' => {
            let (size, next) = pack_size(format, index, Some(8))?;
            (PackKind::Sized(size), next)
        }
        _ => {
            return Err(RuntimeError::message(format!(
                "invalid format option '{}'",
                option as char
            )));
        }
    };
    let size = match kind {
        PackKind::Int { size, .. } | PackKind::Float(size) | PackKind::Sized(size) => size,
        PackKind::Fixed(_) | PackKind::Zero | PackKind::Pad | PackKind::Align => 1,
    };
    if matches!(kind, PackKind::Int { size: 0, .. } | PackKind::Sized(0)) {
        return Err(RuntimeError::message("size out of limits [1,16]"));
    }
    if size > 16 {
        return Err(RuntimeError::message("size out of limits [1,16]"));
    }
    Ok((
        PackItem {
            kind,
            endian: little,
            align: size.min(max),
        },
        next,
    ))
}

pub(crate) fn pack_size(
    format: &[u8],
    mut index: usize,
    default: Option<usize>,
) -> RuntimeResult<(usize, usize)> {
    let start = index;
    while index < format.len() && format[index].is_ascii_digit() {
        index += 1;
    }
    if start == index {
        return default
            .map(|v| (v, index))
            .ok_or_else(|| RuntimeError::message("missing size"));
    }
    let value = std::str::from_utf8(&format[start..index])
        .ok()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| RuntimeError::message("invalid format"))?;
    Ok((value, index))
}

pub(crate) fn pack_align(output: &mut PackBuffer, align: usize) -> RuntimeResult<()> {
    let padding = align_position(output.len(), align) - output.len();
    output.pad(padding)
}

pub(crate) fn align_position(position: usize, align: usize) -> usize {
    // Most items are not aligned, and the remainder is a division.
    if align <= 1 {
        return position;
    }
    position + (align - position % align) % align
}

/// The integer a `string.pack` argument holds, when it is one a double holds
/// exactly (at most 2^53 in magnitude): an integer, or a float without a
/// fraction. Such a value packs from its integer form, with none of the
/// float range checks of [`pack_integer`], which every other value takes.
#[inline]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "an exact round trip is the test"
)]
fn small_whole(value: &Value) -> Option<i64> {
    const LIMIT: i64 = 1 << 53;
    let whole = match value {
        Value::Integer(whole) => *whole,
        Value::Number(number) => {
            let whole = *number as i64;
            if whole as f64 != *number {
                return None;
            }
            whole
        }
        _ => return None,
    };
    (-LIMIT..=LIMIT).contains(&whole).then_some(whole)
}

/// Whether `whole` is in the range of a packed integer of `size` bytes, as
/// [`pack_integer`] checks it on the float: a signed one holds
/// `-2^(bits-1)..2^(bits-1)`, an unsigned one `0..2^bits`.
#[inline]
const fn whole_fits(whole: i64, signed: bool, size: usize) -> bool {
    if size == 0 || size > 16 {
        return false;
    }
    if size >= 8 {
        // [`small_whole`] is below 2^53 in magnitude.
        return signed || whole >= 0;
    }
    let bits = 8 * size;
    if signed {
        let half = 1_i64 << (bits - 1);
        -half <= whole && whole < half
    } else {
        0 <= whole && whole < 1_i64 << bits
    }
}

/// Appends the `size` bytes of `whole` in two's complement, in the order of
/// the item. Up to eight bytes are one word; a wider item is that word and
/// the bytes of its sign.
#[inline]
#[allow(
    clippy::cast_possible_truncation,
    reason = "a width is at most sixteen"
)]
fn push_whole(output: &mut PackBuffer, whole: i64, size: usize, little: bool) -> RuntimeResult<()> {
    let raw = whole.cast_unsigned();
    if size <= 8 {
        let unused = 64 - 8 * size as u32;
        return output.push_word(
            if little {
                // Drops the sign bytes above the width.
                raw << unused >> unused
            } else {
                // The low `size` bytes, most significant first.
                (raw << unused).swap_bytes()
            },
            size,
        );
    }
    let sign = if whole < 0 { u64::MAX } else { 0 };
    let unused = 64 - 8 * (size - 8) as u32;
    if little {
        output.push_word(raw, 8)?;
        output.push_word(sign >> unused, size - 8)
    } else {
        output.push_word(sign >> unused, size - 8)?;
        output.push_word(raw.swap_bytes(), 8)
    }
}

/// A `string.pack` number at argument `position`.
pub(crate) fn pack_number(value: Option<&Value>, position: usize) -> RuntimeResult<f64> {
    value
        .and_then(|value| Runtime::coerce_number(value, "").ok())
        .ok_or_else(|| RuntimeError::bad_argument(position, "number", value))
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)] // Format widths are capped before these deliberate two's-complement conversions.
pub(crate) fn pack_integer(
    output: &mut PackBuffer,
    value: f64,
    signed: bool,
    size: usize,
    little: bool,
) -> RuntimeResult<()> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(RuntimeError::message("integer expected"));
    }
    let bits = size
        .checked_mul(8)
        .ok_or_else(|| RuntimeError::message("size out of limits"))?;
    let min = if signed {
        -(2f64).powi((bits - 1) as i32)
    } else {
        0.0
    };
    let max = if signed {
        (2f64).powi((bits - 1) as i32) - 1.0
    } else {
        (2f64).powi(bits as i32) - 1.0
    };
    if value < min || value > max {
        return Err(RuntimeError::message("integer overflow"));
    }
    let raw = if signed {
        (value as i128) as u128
    } else {
        value as u128
    };
    // The width of a packed integer is at most 16 bytes, the size of `raw`.
    let mut bytes = [0u8; 16];
    let bytes = &mut bytes[..size];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = (raw >> (index * 8)) as u8;
    }
    if !little {
        bytes.reverse();
    }
    output.extend_from_slice(bytes)
}

pub(crate) fn take_pack_bytes<'a>(
    source: &'a [u8],
    position: &mut usize,
    size: usize,
) -> RuntimeResult<&'a [u8]> {
    let end = position
        .checked_add(size)
        .ok_or_else(|| RuntimeError::message("data too short"))?;
    let data = source
        .get(*position..end)
        .ok_or_else(|| RuntimeError::message("data too short"))?;
    *position = end;
    Ok(data)
}

/// Takes `size` bytes of `source`, at most eight, as an unsigned word in the
/// byte order of the item.
#[inline]
fn unpack_word(
    source: &[u8],
    position: &mut usize,
    size: usize,
    little: bool,
) -> RuntimeResult<u64> {
    debug_assert!(size <= 8);
    let unused = 64 - 8 * size as u32;
    // Eight bytes from the position, where the data has them, load as one
    // word; the bytes past the item are shifted out.
    if size != 0
        && let Some(wide) = source
            .get(*position..)
            .and_then(|rest| rest.first_chunk::<8>())
    {
        *position += size;
        return Ok(if little {
            u64::from_le_bytes(*wide) << unused >> unused
        } else {
            u64::from_be_bytes(*wide) >> unused
        });
    }
    let data = take_pack_bytes(source, position, size)?;
    let mut raw = 0_u64;
    if little {
        for byte in data.iter().rev() {
            raw = raw << 8 | u64::from(*byte);
        }
    } else {
        for byte in data {
            raw = raw << 8 | u64::from(*byte);
        }
    }
    Ok(raw)
}

#[allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)] // The f64-backed runtime cannot preserve wider packed integers exactly yet.
fn unpack_raw(data: &[u8], little: bool) -> u128 {
    let mut raw = 0u128;
    for (index, byte) in data.iter().enumerate() {
        let shift = if little {
            index
        } else {
            data.len() - 1 - index
        };
        raw |= u128::from(*byte) << (shift * 8);
    }
    raw
}

/// Decodes a packed integer as Lua 5.3+ does: sign-extended or wrapped into
/// a `lua_Integer`, rejecting wider values whose extra bytes carry data.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)] // Wrapping to 64 bits is the documented Lua conversion.
pub(crate) fn unpack_lua_integer(
    source: &[u8],
    position: &mut usize,
    signed: bool,
    size: usize,
    little: bool,
) -> RuntimeResult<i64> {
    let raw = unpack_raw(take_pack_bytes(source, position, size)?, little);
    if size < 8 {
        let unused = 64 - size as u32 * 8;
        let value = (raw as u64) << unused;
        return Ok(if signed {
            (value as i64) >> unused
        } else {
            (value >> unused) as i64
        });
    }
    let value = raw as u64 as i64;
    if size > 8 {
        let mask = if signed && value < 0 { u128::MAX } else { 0 };
        let high_bits = (size as u32 - 8) * 8;
        let high = (raw >> 64) & (u128::MAX >> (128 - high_bits));
        if high != mask & (u128::MAX >> (128 - high_bits)) {
            return Err(RuntimeError::message(format!(
                "{size}-byte integer does not fit into Lua Integer"
            )));
        }
    }
    Ok(value)
}

pub(crate) fn unpack_integer(
    source: &[u8],
    position: &mut usize,
    signed: bool,
    size: usize,
    little: bool,
) -> RuntimeResult<f64> {
    let raw = unpack_raw(take_pack_bytes(source, position, size)?, little);
    if signed && size < 16 && raw & (1u128 << (size * 8 - 1)) != 0 {
        Ok((raw as i128 - (1i128 << (size * 8))) as f64)
    } else {
        Ok(raw as f64)
    }
}
