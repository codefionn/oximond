//! Shared argument validation and numeric conversion helpers for stdlib builtins.

use crate::runtime::{NumberFormat, Runtime, RuntimeError, RuntimeResult, TableHandle, Value};

/// Argument `position` of a library function, which must be a table.
pub(crate) fn table_argument(
    argument: Option<&Value>,
    position: usize,
) -> RuntimeResult<TableHandle> {
    match argument {
        Some(Value::Table(table)) => Ok(*table),
        argument => Err(RuntimeError::bad_argument(position, "table", argument)),
    }
}

pub(crate) fn table_list_length(
    runtime: &Runtime,
    table: TableHandle,
    operation: &'static str,
) -> RuntimeResult<isize> {
    number_to_isize(runtime.table_length(table)?, operation)
}

/// A table position at argument `position`.
pub(crate) fn table_list_position(argument: &Value, position: usize) -> RuntimeResult<isize> {
    number_to_isize(
        Runtime::coerce_number(argument, "")
            .map_err(|_| RuntimeError::bad_argument(position, "number", Some(argument)))?
            .trunc(),
        "table position",
    )
}

pub(crate) fn metatable_argument(
    runtime: &Runtime,
    argument: Option<&Value>,
    position: usize,
) -> RuntimeResult<Option<TableHandle>> {
    match argument {
        None | Some(Value::Nil) => Ok(None),
        Some(Value::Table(table)) => Ok(Some(*table)),
        argument => Err(runtime.expected_argument_error(position, "nil or table", argument)),
    }
}

pub(crate) fn validate_locale_category(category: Option<&Value>) -> RuntimeResult<()> {
    let category = match category {
        None | Some(Value::Nil) => return Ok(()),
        Some(Value::String(category)) => category.as_bytes(),
        argument => return Err(RuntimeError::bad_argument(2, "string", argument)),
    };
    if matches!(
        category,
        b"all" | b"collate" | b"ctype" | b"monetary" | b"numeric" | b"time"
    ) {
        Ok(())
    } else {
        Err(RuntimeError::argument_error(
            2,
            format!("invalid option '{}'", String::from_utf8_lossy(category)),
        ))
    }
}

/// Integer argument `position` of a library function. Numeric strings
/// convert as the runtime converts library numbers.
pub(crate) fn integer_argument(
    runtime: &Runtime,
    argument: Option<&Value>,
    position: usize,
) -> RuntimeResult<isize> {
    let value = match argument {
        Some(Value::Integer(value)) => {
            return isize::try_from(*value)
                .map_err(|_| RuntimeError::message("string index is out of range"));
        }
        Some(Value::Number(value)) => *value,
        Some(value @ Value::String(_)) => runtime
            .to_number(value)
            .ok_or_else(|| RuntimeError::bad_argument(position, "number", argument))?,
        argument => return Err(RuntimeError::bad_argument(position, "number", argument)),
    };
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(RuntimeError::message(
            "string index must be a finite integer",
        ));
    }
    value
        .to_string()
        .parse()
        .map_err(|_| RuntimeError::message("string index is out of range"))
}

/// A string position or count at argument `position`. Numeric strings
/// convert as the runtime converts library numbers.
pub(crate) fn string_index_argument(
    runtime: &Runtime,
    argument: Option<&Value>,
    position: usize,
) -> RuntimeResult<isize> {
    let value = match argument {
        Some(Value::Integer(value)) => return Ok(saturating_index(*value)),
        Some(Value::Number(value)) => *value,
        Some(value @ Value::String(_)) => runtime
            .to_number(value)
            .ok_or_else(|| RuntimeError::bad_argument(position, "number", argument))?,
        argument => return Err(RuntimeError::bad_argument(position, "number", argument)),
    };
    float_string_index(runtime, value, position)
}

/// An `i64` as an index, clamped to the pointer-sized range.
pub(crate) fn saturating_index(value: i64) -> isize {
    isize::try_from(value).unwrap_or(if value < 0 { isize::MIN } else { isize::MAX })
}

/// A float as a string index, as `luaL_checkinteger` converts it. Lua 5.3 and
/// later refuse a value with no exact integer representation. Earlier
/// versions truncate it, and a value outside the integer range becomes the
/// minimum integer, as x86 converts it. `LuaJIT` converts to a 32-bit integer
/// instead, so a value outside that range, even 2^31 or 2^53, becomes
/// `i32::MIN`.
pub(crate) fn float_string_index(
    runtime: &Runtime,
    value: f64,
    position: usize,
) -> RuntimeResult<isize> {
    const LIMIT: f64 = 9_223_372_036_854_775_808.0;
    if runtime.tagged_integers() {
        if value.fract() != 0.0 || !(-LIMIT..LIMIT).contains(&value) {
            return Err(RuntimeError::argument_error(
                position,
                "number has no integer representation",
            ));
        }
    } else if runtime.number_format() == NumberFormat::LuaJit {
        // Whether the value truncates into the i32 range, without the libm
        // `trunc` call: `as` truncates toward zero itself. NaN is outside.
        return Ok(if value > -2_147_483_649.0 && value < 2_147_483_648.0 {
            #[allow(clippy::cast_possible_truncation)] // Within the i32 range.
            {
                value as i32 as isize
            }
        } else {
            i32::MIN as isize
        });
    } else if !(-LIMIT..LIMIT).contains(&value) {
        return Ok(isize::MIN);
    }
    #[allow(clippy::cast_possible_truncation)] // Within the i64 range, truncating on purpose.
    Ok(saturating_index(value as i64))
}

pub(crate) fn optional_integer_argument(
    runtime: &Runtime,
    argument: Option<&Value>,
    position: usize,
    default: isize,
) -> RuntimeResult<isize> {
    match argument {
        None | Some(Value::Nil) => Ok(default),
        Some(value) => integer_argument(runtime, Some(value), position),
    }
}

pub(crate) fn select_index(value: &Value) -> RuntimeResult<isize> {
    match value {
        Value::Integer(number) => isize::try_from(*number)
            .map_err(|_| RuntimeError::message("select index is out of range")),
        Value::Number(number) => number_to_isize(*number, "select index"),
        value => match Runtime::coerce_number(value, "") {
            Ok(number) => number_to_isize(number, "select index"),
            Err(_) => Err(RuntimeError::bad_argument(1, "number", Some(value))),
        },
    }
}

/// Whether a finite `value` has no fractional part, as `fract() == 0.0`
/// says, without the libm `trunc` call `fract` makes: `as` truncates.
#[inline]
pub(crate) fn finite_is_integral(value: f64) -> bool {
    // From 2^52 on, every float is an integer.
    const FRACTION_LIMIT: f64 = 4_503_599_627_370_496.0;
    // Below 2^52 in magnitude the cast truncates exactly.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::float_cmp
    )]
    {
        value.abs() >= FRACTION_LIMIT || (value as i64) as f64 == value
    }
}

pub(crate) fn number_to_isize(value: f64, name: &'static str) -> RuntimeResult<isize> {
    pub(crate) const MAX_EXACT_INTEGER: f64 = 9_007_199_254_740_992.0;
    if !value.is_finite() || !finite_is_integral(value) {
        return Err(RuntimeError::message(format!(
            "{name} must be a finite integer"
        )));
    }
    let minimum = isize::MIN as f64;
    let maximum_exclusive = -(isize::MIN as f64);
    if (-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&value)
        && value >= minimum
        && value < maximum_exclusive
    {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the exact-integer and pointer-range checks make this a LuaNumber-safe cast"
        )]
        return Ok(value as isize);
    }
    value
        .to_string()
        .parse()
        .map_err(|_| RuntimeError::message(format!("{name} is out of range")))
}

#[allow(
    clippy::cast_precision_loss,
    reason = "LuaNumber is f64, so index conversion intentionally follows f64 precision"
)]
pub(crate) fn isize_to_number(value: isize) -> RuntimeResult<f64> {
    Ok(value as f64)
}

#[allow(
    clippy::cast_precision_loss,
    reason = "LuaNumber is f64, so index conversion intentionally follows f64 precision"
)]
pub(crate) fn usize_to_number(value: usize, _message: &'static str) -> RuntimeResult<f64> {
    Ok(value as f64)
}

/// Returns a length, count, or string position in the profile's integer
/// representation.
#[inline]
pub(crate) fn usize_integer_value(
    runtime: &Runtime,
    value: usize,
    message: &'static str,
) -> RuntimeResult<Value> {
    if runtime.tagged_integers() {
        return i64::try_from(value)
            .map(Value::Integer)
            .map_err(|_| RuntimeError::message(message));
    }
    Ok(Value::Number(usize_to_number(value, message)?))
}

pub(crate) fn append_print_bytes(output: &mut Vec<u8>, bytes: &[u8]) -> RuntimeResult<()> {
    output
        .try_reserve(bytes.len())
        .map_err(|_| RuntimeError::message("print output is too large"))?;
    output.extend_from_slice(bytes);
    Ok(())
}

pub(crate) fn relative_index(index: isize, length: isize) -> isize {
    if index < 0 {
        length.saturating_add(index).saturating_add(1)
    } else {
        index
    }
}

/// The bytes `string.sub (text, first, last)` and `string.byte (text, first,
/// last)` select from a string of `length` bytes, as offsets. A negative
/// index counts from the end, the range is clipped to the string, and a
/// range with nothing left is `None`.
///
/// The interpreter, the traces' helpers and the exits that rebuild a
/// substring all call this, so they cannot disagree about a clipped or
/// negative index.
pub(crate) fn substring_range(
    length: isize,
    first: isize,
    last: isize,
) -> Option<std::ops::Range<usize>> {
    let start = relative_index(first, length).max(1);
    let end = relative_index(last, length).min(length);
    if start > end {
        return None;
    }
    // `1 <= start <= end <= length`.
    Some(usize::try_from(start - 1).ok()?..usize::try_from(end).ok()?)
}
