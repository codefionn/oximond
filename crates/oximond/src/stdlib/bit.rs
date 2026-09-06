//! The bit library: bitwise operations and hex conversion.

use crate::runtime::cint64::{number_to_i64, number_to_u64};
use crate::runtime::{Runtime, RuntimeError, RuntimeResult, Value};
use crate::stdlib::strings::parse_lua_number;

/// Converts a Lua value to `LuaJIT`'s signed 32-bit bit-operation result.
///
/// Within +/-2^51, `BitOp` rounds numbers to the nearest integer, ties to
/// even, then retains the low 32 bits. For wider doubles this library follows
/// `LuaJIT`'s x64 conversion: add 2^52 + 2^51, then take the low IEEE-754 word.
/// The same conversion applies to numeric strings. `BitOp` does not promise
/// portable results outside its exact-input range.
///
/// # Errors
///
/// Returns an error unless its argument is a number or numeric string.
pub fn tobit(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    // A 64-bit integer keeps its low 32 bits, like any other input.
    Ok(bit_result(bit_argument(
        arguments,
        0,
        "call bit.tobit with",
    )?))
}

/// Formats a 32-bit word as lower-case hexadecimal by default. A negative
/// width selects upper-case output. As in `LuaJIT`, explicit `nil` is not an
/// omitted width and the output width is capped by the native formatter.
pub(crate) fn tohex(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let (bits, digit_count) = match wide_kind(&arguments[..arguments.len().min(1)]) {
        Some(unsigned) => (wide_argument(arguments, 0, unsigned)?, 16),
        None => (
            u64::from(bit_argument(arguments, 0, "call bit.tohex with")?),
            8,
        ),
    };
    let width = match arguments.get(1) {
        Some(value) => {
            bit_argument_value(value, "call bit.tohex with").map_err(|error| error.at_argument(2))
        }
        None => Ok(digit_count),
    }?;
    let width = i32::from_ne_bytes(width.to_ne_bytes());
    let uppercase = width < 0;
    let width = usize::try_from(i64::from(width).unsigned_abs())
        .map_err(|_| RuntimeError::message("bit.tohex width is out of range"))?;
    let width = width.min(254);
    let digit_count = digit_count as usize;
    let mut output = Vec::new();
    output
        .try_reserve_exact(width)
        .map_err(|_| RuntimeError::message("bit.tohex result is too large"))?;
    let digits = format!("{bits:0digit_count$x}");
    if width > digit_count {
        output.extend(std::iter::repeat_n(b'0', width - digit_count));
        output.extend_from_slice(digits.as_bytes());
    } else {
        output.extend_from_slice(&digits.as_bytes()[digit_count - width..]);
    }
    if uppercase {
        output.make_ascii_uppercase();
    }
    Ok(vec![Value::bytes(output)])
}

/// Computes a one-word bitwise complement.
pub(crate) fn bnot(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    match arguments.first() {
        Some(Value::UInt64(bits)) => return Ok(Value::UInt64(!bits)),
        Some(Value::Int64(bits)) => return Ok(Value::Int64(!bits)),
        _ => {}
    }
    Ok(bit_result(!bit_argument(
        arguments,
        0,
        "call bit.bnot with",
    )?))
}

/// Computes the bitwise AND of one or more words.
pub(crate) fn band(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    bit_fold(arguments, "call bit.band with", |left, right| left & right)
}

/// Computes the bitwise OR of one or more words.
pub(crate) fn bor(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    bit_fold(arguments, "call bit.bor with", |left, right| left | right)
}

/// Computes the bitwise XOR of one or more words.
pub(crate) fn bxor(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    bit_fold(arguments, "call bit.bxor with", |left, right| left ^ right)
}

/// Shifts a word left. `LuaJIT` uses only the low five bits of the count.
pub(crate) fn lshift(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    if let Some((bits, count, unsigned)) = wide_shift_arguments(arguments, "call bit.lshift with")?
    {
        return Ok(wide_result(bits.wrapping_shl(count), unsigned));
    }
    let (bits, count) = bit_shift_arguments(arguments, "call bit.lshift with")?;
    Ok(bit_result(bits.wrapping_shl(count)))
}

/// Shifts a word right with zero fill. `LuaJIT` uses only the low five bits of
/// the count.
pub(crate) fn rshift(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    if let Some((bits, count, unsigned)) = wide_shift_arguments(arguments, "call bit.rshift with")?
    {
        return Ok(wide_result(bits.wrapping_shr(count), unsigned));
    }
    let (bits, count) = bit_shift_arguments(arguments, "call bit.rshift with")?;
    Ok(bit_result(bits.wrapping_shr(count)))
}

/// Shifts a word right while copying its sign bit. `LuaJIT` uses only the low
/// five bits of the count.
pub(crate) fn arshift(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    if let Some((bits, count, unsigned)) = wide_shift_arguments(arguments, "call bit.arshift with")?
    {
        let shifted = bits.cast_signed().wrapping_shr(count).cast_unsigned();
        return Ok(wide_result(shifted, unsigned));
    }
    let (bits, count) = bit_shift_arguments(arguments, "call bit.arshift with")?;
    let signed = i32::from_ne_bytes(bits.to_ne_bytes());
    Ok(bit_result(u32::from_ne_bytes(
        signed.wrapping_shr(count).to_ne_bytes(),
    )))
}

/// Rotates a word left. `LuaJIT` uses only the low five bits of the count.
pub(crate) fn rol(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    if let Some((bits, count, unsigned)) = wide_shift_arguments(arguments, "call bit.rol with")? {
        return Ok(wide_result(bits.rotate_left(count), unsigned));
    }
    let (bits, count) = bit_shift_arguments(arguments, "call bit.rol with")?;
    Ok(bit_result(bits.rotate_left(count)))
}

/// Rotates a word right. `LuaJIT` uses only the low five bits of the count.
pub(crate) fn ror(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    if let Some((bits, count, unsigned)) = wide_shift_arguments(arguments, "call bit.ror with")? {
        return Ok(wide_result(bits.rotate_right(count), unsigned));
    }
    let (bits, count) = bit_shift_arguments(arguments, "call bit.ror with")?;
    Ok(bit_result(bits.rotate_right(count)))
}

/// Reverses the four bytes in a word.
pub(crate) fn bswap(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    if let Some(unsigned) = wide_kind(&arguments[..arguments.len().min(1)]) {
        let bits = wide_argument(arguments, 0, unsigned)?;
        return Ok(wide_result(bits.swap_bytes(), unsigned));
    }
    Ok(bit_result(
        bit_argument(arguments, 0, "call bit.bswap with")?.swap_bytes(),
    ))
}

#[inline]
pub(crate) fn bit_fold(
    arguments: &[Value],
    operation: &'static str,
    combine: fn(u64, u64) -> u64,
) -> RuntimeResult<Value> {
    if let [Value::Number(left), Value::Number(right)] = arguments {
        let (left, right) = (
            u64::from(number_bits(*left)),
            u64::from(number_bits(*right)),
        );
        return Ok(bit_result(combine(left, right) as u32));
    }
    if let [left, right] = arguments
        && let Some((left, right, unsigned)) = wide_pair(left, right)
    {
        return Ok(wide_result(combine(left, right), unsigned));
    }
    if let Some(unsigned) = wide_kind(arguments) {
        let mut result = wide_argument(arguments, 0, unsigned)?;
        for index in 1..arguments.len() {
            result = combine(result, wide_argument(arguments, index, unsigned)?);
        }
        return Ok(wide_result(result, unsigned));
    }
    let mut result = u64::from(bit_argument(arguments, 0, operation)?);
    for index in 1..arguments.len() {
        result = combine(
            result,
            u64::from(bit_argument(arguments, index, operation)?),
        );
    }
    // Every operand was zero-extended from 32 bits, so is the result.
    Ok(bit_result(result as u32))
}

pub(crate) fn bit_shift_arguments(
    arguments: &[Value],
    operation: &'static str,
) -> RuntimeResult<(u32, u32)> {
    Ok((
        bit_argument(arguments, 0, operation)?,
        bit_argument(arguments, 1, operation)? & 31,
    ))
}

pub(crate) fn bit_argument(
    arguments: &[Value],
    index: usize,
    operation: &'static str,
) -> RuntimeResult<u32> {
    match arguments.get(index) {
        Some(value) => {
            bit_argument_value(value, operation).map_err(|error| error.at_argument(index + 1))
        }
        None => Err(RuntimeError::bad_argument(index + 1, "number", None)),
    }
}

pub(crate) fn bit_argument_value(value: &Value, operation: &'static str) -> RuntimeResult<u32> {
    bit_argument_value_with_conversion(value, operation, number_bits)
}

fn bit_argument_value_with_conversion(
    value: &Value,
    operation: &'static str,
    convert: fn(f64) -> u32,
) -> RuntimeResult<u32> {
    let number = match value {
        // Modern Lua profiles represent integer literals separately. A cast
        // keeps their low 32 bits without losing precision through f64.
        // As does LuaJIT's 64-bit integer cdata.
        Value::Integer(integer) | Value::Int64(integer) => return Ok(*integer as u32),
        Value::UInt64(integer) => return Ok(*integer as u32),
        Value::Number(number) => return Ok(convert(*number)),
        Value::String(value) => {
            parse_lua_number(value.as_bytes()).ok_or(RuntimeError::TypeError {
                operation,
                expected: "number",
                found: "string",
            })?
        }
        value => {
            return Err(RuntimeError::TypeError {
                operation,
                expected: "number",
                found: value.type_name(),
            });
        }
    };
    Ok(convert(number))
}

/// `LuaJIT`'s x64 double-to-`BitOp` conversion, including wide finite numbers.
#[inline]
fn number_bits(number: f64) -> u32 {
    (number + 6_755_399_441_055_744.0).to_bits() as u32
}

/// Mathematical word normalization for Lua 5.2 `bit32`. Its unsigned words
/// keep their low bits even when `LuaJIT`'s biased double conversion differs.
fn number_bits_wide(number: f64) -> u32 {
    if number.abs() < 2_251_799_813_685_248.0 {
        return number_bits(number);
    }
    if !number.is_finite() {
        return 0;
    }
    number.round_ties_even().rem_euclid(4_294_967_296.0) as u32
}

fn number_result(bits: u32) -> f64 {
    f64::from(i32::from_ne_bytes(bits.to_ne_bytes()))
}

pub(crate) fn bit_result(bits: u32) -> Value {
    Value::Number(number_result(bits))
}

// LuaJIT 2.1 widens the operations to 64 bits when an operand is 64-bit
// integer cdata. The result is `uint64_t` when any such operand is unsigned
// and `int64_t` otherwise; numbers and numeric strings convert as C does.

/// `Some(unsigned)` when any of `values` is a 64-bit integer.
fn wide_kind(values: &[Value]) -> Option<bool> {
    let mut kind = None;
    for value in values {
        match value {
            Value::UInt64(_) => return Some(true),
            Value::Int64(_) => kind = Some(false),
            _ => {}
        }
    }
    kind
}

/// Two arguments where at least one is a 64-bit integer and the other is a
/// 64-bit integer or a number, as bits and signedness. Matches
/// `wide_kind` plus `wide_argument` for those pairs.
#[inline]
fn wide_pair(left: &Value, right: &Value) -> Option<(u64, u64, bool)> {
    Some(match (left, right) {
        (Value::UInt64(a), Value::UInt64(b)) => (*a, *b, true),
        (Value::Int64(a), Value::Int64(b)) => (a.cast_unsigned(), b.cast_unsigned(), false),
        (Value::UInt64(a), Value::Int64(b)) => (*a, b.cast_unsigned(), true),
        (Value::Int64(a), Value::UInt64(b)) => (a.cast_unsigned(), *b, true),
        (Value::UInt64(a), Value::Number(b)) => (*a, number_to_u64(*b), true),
        (Value::Number(a), Value::UInt64(b)) => (number_to_u64(*a), *b, true),
        (Value::Int64(a), Value::Number(b)) => {
            (a.cast_unsigned(), number_to_i64(*b).cast_unsigned(), false)
        }
        (Value::Number(a), Value::Int64(b)) => {
            (number_to_i64(*a).cast_unsigned(), b.cast_unsigned(), false)
        }
        _ => return None,
    })
}

fn wide_argument(arguments: &[Value], index: usize, unsigned: bool) -> RuntimeResult<u64> {
    let number = match arguments.get(index) {
        Some(Value::Int64(value) | Value::Integer(value)) => return Ok(value.cast_unsigned()),
        Some(Value::UInt64(value)) => return Ok(*value),
        Some(Value::Number(number)) => *number,
        Some(Value::String(value)) => parse_lua_number(value.as_bytes())
            .ok_or_else(|| RuntimeError::bad_argument(index + 1, "number", arguments.get(index)))?,
        value => {
            return Err(RuntimeError::bad_argument(index + 1, "number", value));
        }
    };
    Ok(if unsigned {
        number_to_u64(number)
    } else {
        number_to_i64(number).cast_unsigned()
    })
}

fn wide_result(bits: u64, unsigned: bool) -> Value {
    if unsigned {
        Value::UInt64(bits)
    } else {
        Value::Int64(bits.cast_signed())
    }
}

/// The 64-bit shift operand, count and kind when the shifted value is a
/// 64-bit integer. The count uses its low six bits.
#[inline]
fn wide_shift_arguments(
    arguments: &[Value],
    operation: &'static str,
) -> RuntimeResult<Option<(u64, u32, bool)>> {
    let (bits, unsigned) = match arguments.first() {
        Some(Value::UInt64(bits)) => (*bits, true),
        Some(Value::Int64(bits)) => (bits.cast_unsigned(), false),
        _ => return Ok(None),
    };
    let count = match arguments.get(1) {
        Some(Value::Number(count)) => number_bits(*count),
        _ => bit_argument(arguments, 1, operation)?,
    } & 63;
    Ok(Some((bits, count, unsigned)))
}

// Pure 64-bit forms for the interpreter's in-window call. Each matches its
// scalar callback called with the two arguments wherever it answers, and
// answers for a 64-bit integer operand with a number or 64-bit integer
// beside it; anything else is `None`, left to the callback.

/// The shifted value, the count's low six bits and the result's kind, as
/// [`wide_shift_arguments`] reads them without a numeric string.
#[inline]
fn wide_shift_pair(bits: &Value, count: &Value) -> Option<(u64, u32, bool)> {
    let (bits, unsigned) = match bits {
        Value::UInt64(bits) => (*bits, true),
        Value::Int64(bits) => (bits.cast_unsigned(), false),
        _ => return None,
    };
    let count = match count {
        Value::Number(count) => number_bits(*count),
        Value::Integer(count) | Value::Int64(count) => *count as u32,
        Value::UInt64(count) => *count as u32,
        _ => return None,
    } & 63;
    Some((bits, count, unsigned))
}

pub(crate) fn wide_band(left: &Value, right: &Value) -> Option<Value> {
    let (left, right, unsigned) = wide_pair(left, right)?;
    Some(wide_result(left & right, unsigned))
}

pub(crate) fn wide_bor(left: &Value, right: &Value) -> Option<Value> {
    let (left, right, unsigned) = wide_pair(left, right)?;
    Some(wide_result(left | right, unsigned))
}

pub(crate) fn wide_bxor(left: &Value, right: &Value) -> Option<Value> {
    let (left, right, unsigned) = wide_pair(left, right)?;
    Some(wide_result(left ^ right, unsigned))
}

pub(crate) fn wide_lshift(bits: &Value, count: &Value) -> Option<Value> {
    let (bits, count, unsigned) = wide_shift_pair(bits, count)?;
    Some(wide_result(bits.wrapping_shl(count), unsigned))
}

pub(crate) fn wide_rshift(bits: &Value, count: &Value) -> Option<Value> {
    let (bits, count, unsigned) = wide_shift_pair(bits, count)?;
    Some(wide_result(bits.wrapping_shr(count), unsigned))
}

pub(crate) fn wide_arshift(bits: &Value, count: &Value) -> Option<Value> {
    let (bits, count, unsigned) = wide_shift_pair(bits, count)?;
    let shifted = bits.cast_signed().wrapping_shr(count).cast_unsigned();
    Some(wide_result(shifted, unsigned))
}

pub(crate) fn wide_rol(bits: &Value, count: &Value) -> Option<Value> {
    let (bits, count, unsigned) = wide_shift_pair(bits, count)?;
    Some(wide_result(bits.rotate_left(count), unsigned))
}

pub(crate) fn wide_ror(bits: &Value, count: &Value) -> Option<Value> {
    let (bits, count, unsigned) = wide_shift_pair(bits, count)?;
    Some(wide_result(bits.rotate_right(count), unsigned))
}

// Pure numeric fast paths. Each matches its scalar callback called with
// exactly one or two `Value::Number` arguments.

pub(crate) extern "C" fn numeric_tobit(number: f64) -> f64 {
    number_result(number_bits(number))
}

pub(crate) extern "C" fn numeric_bnot(number: f64) -> f64 {
    number_result(!number_bits(number))
}

pub(crate) extern "C" fn numeric_bswap(number: f64) -> f64 {
    number_result(number_bits(number).swap_bytes())
}

pub(crate) extern "C" fn numeric_band(left: f64, right: f64) -> f64 {
    number_result(number_bits(left) & number_bits(right))
}

pub(crate) extern "C" fn numeric_bor(left: f64, right: f64) -> f64 {
    number_result(number_bits(left) | number_bits(right))
}

pub(crate) extern "C" fn numeric_bxor(left: f64, right: f64) -> f64 {
    number_result(number_bits(left) ^ number_bits(right))
}

pub(crate) extern "C" fn numeric_lshift(bits: f64, count: f64) -> f64 {
    number_result(number_bits(bits).wrapping_shl(number_bits(count) & 31))
}

pub(crate) extern "C" fn numeric_rshift(bits: f64, count: f64) -> f64 {
    number_result(number_bits(bits).wrapping_shr(number_bits(count) & 31))
}

pub(crate) extern "C" fn numeric_arshift(bits: f64, count: f64) -> f64 {
    let signed = i32::from_ne_bytes(number_bits(bits).to_ne_bytes());
    f64::from(signed.wrapping_shr(number_bits(count) & 31))
}

pub(crate) extern "C" fn numeric_rol(bits: f64, count: f64) -> f64 {
    number_result(number_bits(bits).rotate_left(number_bits(count) & 31))
}

pub(crate) extern "C" fn numeric_ror(bits: f64, count: f64) -> f64 {
    number_result(number_bits(bits).rotate_right(number_bits(count) & 31))
}

// Lua 5.2/5.3 bit32 uses unsigned words and saturating shifts. Keep these
// callbacks separate from LuaJIT BitOp, including its numeric fast paths.
pub(crate) fn bit32_word(
    runtime: &Runtime,
    arguments: &[Value],
    index: usize,
) -> RuntimeResult<u32> {
    if runtime.tagged_integers() {
        Ok(crate::stdlib::args::integer_argument(runtime, arguments.get(index), index + 1)? as u32)
    } else {
        match arguments.get(index) {
            Some(value) => {
                bit_argument_value_with_conversion(value, "call bit32 with", number_bits_wide)
                    .map_err(|error| error.at_argument(index + 1))
            }
            None => Err(RuntimeError::bad_argument(index + 1, "number", None)),
        }
    }
}

fn bit32_count(runtime: &Runtime, arguments: &[Value], index: usize) -> RuntimeResult<i64> {
    Ok(
        crate::stdlib::args::string_index_argument(runtime, arguments.get(index), index + 1)?
            as i64,
    )
}

fn bit32_shift(bits: u32, count: i64, left: bool, arithmetic: bool) -> u32 {
    let magnitude = count.unsigned_abs();
    let left = left ^ (count < 0);
    if magnitude >= 32 {
        if !left && arithmetic && count >= 0 && bits & 0x8000_0000 != 0 {
            u32::MAX
        } else {
            0
        }
    } else if left {
        bits << magnitude
    } else if arithmetic && count >= 0 {
        ((bits as i32) >> magnitude) as u32
    } else {
        bits >> magnitude
    }
}

fn bit32_field(runtime: &Runtime, arguments: &[Value], index: usize) -> RuntimeResult<(u32, u32)> {
    let field = bit32_count(runtime, arguments, index)?;
    let width = match arguments.get(index + 1) {
        None | Some(Value::Nil) => 1,
        Some(_) => bit32_count(runtime, arguments, index + 1)?,
    };
    if field < 0 {
        return Err(RuntimeError::argument_error(
            index + 1,
            "field cannot be negative",
        ));
    }
    if width <= 0 {
        return Err(RuntimeError::argument_error(
            index + 2,
            "width must be positive",
        ));
    }
    if field >= 32 || width > 32 - field {
        return Err(RuntimeError::message("trying to access non-existent bits"));
    }
    Ok((field as u32, u32::MAX >> (32 - width)))
}

pub(crate) fn bit32_operation(
    runtime: &mut Runtime,
    arguments: &[Value],
    operation: &str,
) -> RuntimeResult<Vec<Value>> {
    let bits = match operation {
        "band" | "bor" | "bxor" | "btest" => {
            let mut bits = if matches!(operation, "band" | "btest") {
                u32::MAX
            } else {
                0
            };
            for index in 0..arguments.len() {
                let word = bit32_word(runtime, arguments, index)?;
                bits = match operation {
                    "bor" => bits | word,
                    "bxor" => bits ^ word,
                    _ => bits & word,
                };
            }
            if operation == "btest" {
                return Ok(vec![Value::Boolean(bits != 0)]);
            }
            bits
        }
        "bnot" => !bit32_word(runtime, arguments, 0)?,
        "extract" => {
            let bits = bit32_word(runtime, arguments, 0)?;
            let (field, mask) = bit32_field(runtime, arguments, 1)?;
            (bits >> field) & mask
        }
        "replace" => {
            let bits = bit32_word(runtime, arguments, 0)?;
            let replacement = bit32_word(runtime, arguments, 1)?;
            let (field, mask) = bit32_field(runtime, arguments, 2)?;
            (bits & !(mask << field)) | ((replacement & mask) << field)
        }
        _ => {
            let bits = bit32_word(runtime, arguments, 0)?;
            let count = bit32_count(runtime, arguments, 1)?;
            match operation {
                "lrotate" => bits.rotate_left((count & 31) as u32),
                "rrotate" => bits.rotate_right((count & 31) as u32),
                "lshift" => bit32_shift(bits, count, true, false),
                "rshift" => bit32_shift(bits, count, false, false),
                "arshift" => bit32_shift(bits, count, false, true),
                _ => unreachable!(),
            }
        }
    };
    Ok(vec![if runtime.tagged_integers() {
        Value::Integer(i64::from(bits))
    } else {
        Value::Number(f64::from(bits))
    }])
}

#[cfg(test)]
mod tests {
    use super::{number_bits, number_bits_wide};

    #[test]
    fn bitop_and_bit32_round_ties_and_keep_their_wide_conversion_rules() {
        for (value, expected) in [
            (0.0, 0),
            (-0.0, 0),
            (0.5, 0),
            (1.5, 2),
            (2.5, 2),
            (-0.5, 0),
            (-1.5, 0xffff_fffe),
            (-2.5, 0xffff_fffe),
            (4_294_967_295.5, 0),
            (4_294_967_296.5, 0),
            (2_147_483_647.5, 0x8000_0000),
        ] {
            assert_eq!(number_bits(value), expected, "BitOp {value}");
            assert_eq!(number_bits_wide(value), expected, "bit32 {value}");
        }
        let value = 3_002_399_751_580_330.0;
        assert_eq!(number_bits(value), 0x5555_5555);
        assert_eq!(number_bits_wide(value), 0xaaaa_aaaa);
        assert_eq!(number_bits_wide(f64::INFINITY), 0);
        assert_eq!(number_bits_wide(f64::NAN), 0);
    }
}
