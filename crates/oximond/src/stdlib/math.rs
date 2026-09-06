//! The math library: scalar helpers, integer conversion, and random numbers.

use crate::runtime::{FileOpenMode, LinesFormats, Runtime, RuntimeError, RuntimeResult, Value};
use crate::stdlib::base::path_from_lua_string;
use crate::stdlib::format::number_to_i64;
use crate::string::LuaString;

pub(crate) fn floor_scalar(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    if let Some(Value::Integer(value)) = arguments.first() {
        return Ok(Value::Integer(*value));
    }
    let value = number_argument(runtime, arguments, 1)?.floor();
    Ok(runtime.integral_float_value(value))
}

pub(crate) fn sin_scalar(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    unary_math_scalar(runtime, arguments, f64::sin)
}

/// Returns the floor-based remainder of its first two numeric arguments.
///
/// # Errors
///
/// Returns an error unless both arguments follow the runtime's numeric
/// coercion rule. Extra arguments are ignored.
pub fn math_mod(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let (left, right) = binary_math(runtime, arguments)?;
    Ok(vec![Value::Number(left - (left / right).floor() * right)])
}

#[allow(clippy::unnecessary_wraps)] // Native functions share one fallible callback signature.
pub(crate) fn math_type(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    Ok(vec![match arguments.first() {
        Some(Value::Integer(_)) => Value::from("integer"),
        Some(Value::Number(_)) => Value::from("float"),
        _ => Value::Nil,
    }])
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::unnecessary_wraps
)]
pub(crate) fn math_tointeger(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let Some(value) = arguments.first() else {
        return Ok(vec![Value::Nil]);
    };
    if let Value::Integer(value) = value {
        return Ok(vec![Value::Integer(*value)]);
    }
    let Ok(value) = Runtime::coerce_number(value, "call math.tointeger with") else {
        return Ok(vec![Value::Nil]);
    };
    if value.is_finite()
        && value.fract() == 0.0
        && (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&value)
    {
        Ok(vec![Value::Integer(value as i64)])
    } else {
        Ok(vec![Value::Nil])
    }
}

pub(crate) fn math_ult(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let left = integer_i64_argument(arguments.first(), 1)?;
    let right = integer_i64_argument(arguments.get(1), 2)?;
    Ok(vec![Value::Boolean(
        left.cast_unsigned() < right.cast_unsigned(),
    )])
}

/// Argument `position` as `luaL_checkinteger` reads it.
pub(crate) fn integer_i64_argument(
    argument: Option<&Value>,
    position: usize,
) -> RuntimeResult<i64> {
    if let Some(Value::Integer(value)) = argument {
        return Ok(*value);
    }
    let value = argument
        .and_then(|value| Runtime::coerce_number(value, "").ok())
        .ok_or_else(|| RuntimeError::bad_argument(position, "number", argument))?;
    if !value.is_finite()
        || value.fract() != 0.0
        || value < -9_223_372_036_854_775_808.0
        || value >= 9_223_372_036_854_775_808.0
    {
        return Err(RuntimeError::argument_error(
            position,
            "number has no integer representation",
        ));
    }
    number_to_i64(value, "math integer")
}

pub(crate) fn math_random(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let value = match arguments {
        [] => return Ok(vec![Value::Number(runtime.random_unit())]),
        [upper] => {
            let upper = random_integer(upper, 1)?;
            if upper < 1 {
                return Err(RuntimeError::message("interval is empty"));
            }
            random_bounded(runtime, 1, upper)?
        }
        [lower, upper, ..] => {
            if arguments.len() != 2 {
                return Err(RuntimeError::message(
                    "wrong number of arguments to 'random'",
                ));
            }
            let lower = random_integer(lower, 1)?;
            let upper = random_integer(upper, 2)?;
            if lower > upper {
                return Err(RuntimeError::message("interval is empty"));
            }
            random_bounded(runtime, lower, upper)?
        }
    };
    Ok(vec![runtime.integral_float_value(value)])
}

/// `LuaJIT` accepts floating bounds, reversed intervals, and extra arguments.
pub(crate) fn math_random_luajit(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    let unit = runtime.random_unit();
    let result = match arguments {
        [] => unit,
        [_] => {
            let upper = number_argument(runtime, arguments, 1)?;
            (unit * upper).floor() + 1.0
        }
        [_, _, ..] => {
            let lower = number_argument(runtime, arguments, 1)?;
            let upper = number_argument(runtime, arguments, 2)?;
            (unit * (upper - lower + 1.0)).floor() + lower
        }
    };
    Ok(vec![Value::Number(result)])
}

pub(crate) fn math_randomseed_luajit(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    if arguments.is_empty() {
        runtime.random_seed_luajit_auto();
    } else {
        runtime.random_seed_luajit(number_argument(runtime, arguments, 1)?);
    }
    Ok(Vec::new())
}

/// Lua 5.3's `math.random`: integer bounds span the whole `lua_Integer`
/// range as long as their difference fits.
pub(crate) fn math_random_lua53(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    math_random_integers(runtime, arguments, false)
}

/// Lua 5.4's `math.random`: `math.random(0)` yields all 64 random bits and
/// any ordered integer interval is accepted.
pub(crate) fn math_random_lua54(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    math_random_integers(runtime, arguments, true)
}

#[allow(clippy::cast_precision_loss)] // The Lua 5.4 unit float uses at most 53 exact bits.
fn math_random_integers(
    runtime: &mut Runtime,
    arguments: &[Value],
    lua54: bool,
) -> RuntimeResult<Vec<Value>> {
    // Lua 5.4 advances the generator once before checking arguments. Its
    // interval projection starts with that same value.
    let initial_bits = lua54.then(|| runtime.random_bits());
    let (lower, upper) = match arguments {
        [] => {
            let unit = if let Some(bits) = initial_bits {
                (bits >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
            } else {
                runtime.random_unit()
            };
            return Ok(vec![Value::Number(unit)]);
        }
        [upper] => {
            let upper = random_lua_integer(upper, 1)?;
            if lua54 && upper == 0 {
                return Ok(vec![Value::Integer(i64::from_ne_bytes(
                    initial_bits
                        .expect("Lua 5.4 drew an initial value")
                        .to_ne_bytes(),
                ))]);
            }
            (1, upper)
        }
        [lower, upper] => (random_lua_integer(lower, 1)?, random_lua_integer(upper, 2)?),
        _ => return Err(RuntimeError::message("wrong number of arguments")),
    };
    if lower > upper {
        return Err(RuntimeError::argument_error(1, "interval is empty"));
    }
    if !lua54 && upper.checked_sub(lower).is_none() {
        return Err(RuntimeError::argument_error(1, "interval too large"));
    }
    let span = upper.abs_diff(lower);
    // Rejection sampling over the smallest all-ones mask covering the span
    // keeps every value in the interval equally likely.
    let mask = u64::MAX.checked_shr(span.leading_zeros()).unwrap_or(0);
    let mut bits = initial_bits.unwrap_or_else(|| runtime.random_bits());
    let offset = loop {
        let candidate = bits & mask;
        if candidate <= span {
            break candidate;
        }
        bits = runtime.random_bits();
    };
    Ok(vec![Value::Integer(lower.wrapping_add_unsigned(offset))])
}

/// Reads a `math.random` bound the way `luaL_checkinteger` does.
fn random_lua_integer(value: &Value, position: usize) -> RuntimeResult<i64> {
    if let Value::Integer(value) = value {
        return Ok(*value);
    }
    if let Value::String(text) = value
        && let Some(value) = crate::runtime::parse_lua_integer_exact(text.as_bytes())
    {
        return Ok(value);
    }
    let value = Runtime::coerce_number(value, "")
        .map_err(|_| RuntimeError::bad_argument(position, "number", Some(value)))?;
    if value.fract() != 0.0
        || !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&value)
    {
        return Err(RuntimeError::argument_error(
            position,
            "number has no integer representation",
        ));
    }
    number_to_i64(value, "math integer")
}

pub(crate) fn math_randomseed(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    let seed = arguments
        .first()
        .and_then(|seed| Runtime::coerce_number(seed, "").ok())
        .ok_or_else(|| RuntimeError::bad_argument(1, "number", arguments.first()))?;
    runtime.random_seed(seed)?;
    Ok(Vec::new())
}

pub(crate) fn math_randomseed_modern(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    let (first, second) = if let Some(first) = arguments.first() {
        let first = random_lua_integer(first, 1)?;
        let second = match arguments.get(1) {
            None | Some(Value::Nil) => 0,
            Some(second) => random_lua_integer(second, 2)?,
        };
        runtime.random_seed_modern(first, second);
        (first, second)
    } else {
        runtime.random_seed_modern_auto()
    };
    Ok(vec![Value::Integer(first), Value::Integer(second)])
}

/// A Lua 5.1 `math.random` bound at argument `position`.
pub(crate) fn random_integer(value: &Value, position: usize) -> RuntimeResult<i64> {
    let value = Runtime::coerce_number(value, "")
        .map_err(|_| RuntimeError::bad_argument(position, "number", Some(value)))?;
    let integer_expected = || RuntimeError::argument_error(position, "integer expected");
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(integer_expected());
    }
    if value < f64::from(i32::MIN) || value > f64::from(i32::MAX) {
        return Err(integer_expected());
    }
    value
        .to_string()
        .parse::<i32>()
        .map(i64::from)
        .map_err(|_| integer_expected())
}

pub(crate) fn random_bounded(runtime: &mut Runtime, lower: i64, upper: i64) -> RuntimeResult<f64> {
    let span = upper
        .checked_sub(lower)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| RuntimeError::message("interval is too large"))?;
    let offset = (runtime.random_unit() * integer_to_number(span)?).floor();
    let result = integer_to_number(lower)? + offset;
    Ok(result)
}

pub(crate) fn integer_to_number(value: i64) -> RuntimeResult<f64> {
    value
        .to_string()
        .parse::<f64>()
        .map_err(|_| RuntimeError::message("integer is out of range"))
}

pub(crate) fn io_tmpfile(runtime: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
    Ok(vec![Value::File(runtime.create_temp_file()?)])
}

pub(crate) fn io_input(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    io_select_default(runtime, arguments, true)
}

pub(crate) fn io_output(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    io_select_default(runtime, arguments, false)
}

pub(crate) fn io_select_default(
    runtime: &mut Runtime,
    arguments: &[Value],
    input: bool,
) -> RuntimeResult<Vec<Value>> {
    let current = if input {
        runtime.default_input()
    } else {
        runtime.default_output()
    };
    let Some(value) = arguments
        .first()
        .filter(|value| !matches!(value, Value::Nil))
    else {
        return Ok(current.map_or_else(|| vec![Value::Nil], |file| vec![Value::File(file)]));
    };
    let file = match value {
        Value::File(file) => *file,
        Value::String(path) => {
            let mode = if input {
                FileOpenMode::Read
            } else {
                FileOpenMode::Write
            };
            match runtime.open_file(path_from_lua_string(path), mode) {
                Ok(file) => file,
                Err(error) => return Ok(vec![Value::Nil, Value::from(error.to_string())]),
            }
        }
        // LuaJIT reads anything else as a file name.
        value => {
            let expected = if runtime.number_format() == crate::runtime::NumberFormat::LuaJit {
                "string"
            } else {
                "FILE*"
            };
            return Err(RuntimeError::bad_argument(1, expected, Some(value)));
        }
    };
    let selected = if input {
        runtime.set_default_input(Some(file))
    } else {
        runtime.set_default_output(Some(file))
    };
    match selected {
        Ok(()) => Ok(vec![Value::File(file)]),
        Err(error) => Ok(vec![Value::Nil, Value::from(error.to_string())]),
    }
}

pub(crate) fn io_read(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let Some(file) = runtime.default_input() else {
        return Ok(vec![
            Value::Nil,
            Value::from("standard input is unavailable"),
        ]);
    };
    read_file_formats(runtime, file, arguments)
}

pub(crate) fn io_close(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let file = match arguments.first() {
        Some(Value::File(file)) => Some(*file),
        Some(value) => return Err(RuntimeError::bad_argument(1, "FILE*", Some(value))),
        None => runtime.default_output(),
    };
    let Some(file) = file else {
        return Ok(vec![Value::Boolean(true)]);
    };
    match runtime.file_close_process(file) {
        Ok(Some(status)) => Ok(process_close_result(runtime, status)),
        Ok(None) => Ok(vec![Value::Boolean(true)]),
        Err(error @ (RuntimeError::FileIo { .. } | RuntimeError::HostOutputIo { .. })) => {
            Ok(io_failure(&error))
        }
        Err(error) => Err(error),
    }
}

pub(crate) fn io_open(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let path = lua_string_argument(arguments.first(), 1)?;
    let mode = match arguments.get(1) {
        None | Some(Value::Nil) => FileOpenMode::Read,
        Some(Value::String(mode)) => match FileOpenMode::parse(mode.as_bytes()) {
            Ok(mode) => mode,
            Err(error) => return Ok(vec![Value::Nil, Value::from(error.to_string())]),
        },
        argument => return Err(RuntimeError::bad_argument(2, "string", argument)),
    };
    match runtime.open_file(path_from_lua_string(path), mode) {
        Ok(file) => Ok(vec![Value::File(file)]),
        Err(error) => Ok(io_failure(&error)),
    }
}

/// Opens a shell command as a readable or writable file.
///
/// # Errors
///
/// Returns an argument error for an invalid command or mode.
pub(crate) fn io_popen(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let command = lua_string_argument(arguments.first(), 1)?;
    let mode = match arguments.get(1) {
        None | Some(Value::Nil) => b"r".as_slice(),
        Some(Value::String(mode)) => mode.as_bytes(),
        argument => return Err(RuntimeError::bad_argument(2, "string", argument)),
    };
    let read = match mode {
        b"r" => true,
        b"w" => false,
        _ => return Err(RuntimeError::argument_error(2, "invalid mode")),
    };
    match runtime.open_process(path_from_lua_string(command), read) {
        Ok(file) => Ok(vec![Value::File(file)]),
        Err(error @ RuntimeError::FileIo { .. }) => Ok(io_failure(&error)),
        Err(error) => Err(error),
    }
}

pub(crate) fn io_lines(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let (file, close_at_eof) = match arguments.first() {
        None | Some(Value::Nil) => (
            runtime
                .default_input()
                .ok_or_else(|| RuntimeError::message("default input is unavailable"))?,
            false,
        ),
        Some(Value::String(path)) => (
            runtime.open_file(path_from_lua_string(path), FileOpenMode::Read)?,
            true,
        ),
        argument => return Err(RuntimeError::bad_argument(1, "string", argument)),
    };
    let formats = arguments.get(1..).unwrap_or_default();
    let iterator =
        runtime.create_file_lines_iterator(file, close_at_eof, formats, "io.lines", 2)?;
    if close_at_eof && runtime.lines_formats() == LinesFormats::Lua54 {
        return Ok(vec![iterator, Value::Nil, Value::Nil, Value::File(file)]);
    }
    Ok(vec![iterator])
}

pub(crate) fn io_flush(runtime: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
    let file = runtime
        .default_output()
        .ok_or_else(|| RuntimeError::message("default output is unavailable"))?;
    match runtime.file_flush(file) {
        Ok(()) => Ok(vec![Value::Boolean(true)]),
        Err(error @ (RuntimeError::FileIo { .. } | RuntimeError::HostOutputIo { .. })) => {
            Ok(io_failure(&error))
        }
        Err(error) => Err(error),
    }
}

pub(crate) fn io_type(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let Some(argument) = arguments.first() else {
        return Err(RuntimeError::argument_error(1, "value expected"));
    };
    let Value::File(file) = argument else {
        return Ok(vec![Value::Nil]);
    };
    Ok(vec![Value::from(if runtime.file_is_closed(*file)? {
        "closed file"
    } else {
        "file"
    })])
}

pub(crate) fn file_gc(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let Some(Value::File(file)) = arguments.first() else {
        return Err(RuntimeError::argument_error(
            1,
            "FILE* expected, got no value",
        ));
    };
    if !runtime.file_is_closed(*file)? {
        runtime.file_close(*file)?;
    }
    Ok(Vec::new())
}

pub(crate) fn file_tostring(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    let Some(Value::File(file)) = arguments.first() else {
        return Err(RuntimeError::argument_error(
            1,
            "FILE* expected, got no value",
        ));
    };
    Ok(vec![if runtime.file_is_closed(*file)? {
        Value::from("file (closed)")
    } else {
        Value::from(format!("file ({})", file.index()))
    }])
}

/// The file a file method takes as its receiver.
pub(crate) fn file_argument(arguments: &[Value]) -> RuntimeResult<crate::runtime::FileHandle> {
    match arguments.first() {
        Some(Value::File(file)) => Ok(*file),
        argument => Err(RuntimeError::bad_argument(1, "FILE*", argument)),
    }
}

pub(crate) fn file_write(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let file = file_argument(arguments)?;
    let values = &arguments[1..];
    // Like Lua, write the values before a bad one, then raise its error.
    let bad = values.iter().position(|value| {
        !matches!(
            value,
            Value::String(_) | Value::Number(_) | Value::Integer(_)
        )
    });
    match runtime.file_write(file, &values[..bad.unwrap_or(values.len())]) {
        Ok(()) => {}
        Err(error @ (RuntimeError::FileIo { .. } | RuntimeError::HostOutputIo { .. })) => {
            return Ok(io_failure(&error));
        }
        Err(error) => return Err(error),
    }
    match bad {
        Some(index) => Err(RuntimeError::bad_argument(
            index + 2,
            "string",
            values.get(index),
        )),
        None => Ok(vec![Value::File(file)]),
    }
}

pub(crate) fn file_seek(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let file = file_argument(arguments)?;
    let whence = arguments
        .get(1)
        .cloned()
        .unwrap_or_else(|| Value::from("cur"));
    let offset = arguments.get(2).cloned().unwrap_or(Value::Number(0.0));
    match runtime.file_seek(file, &whence, &offset) {
        Ok(position) => Ok(vec![runtime.integral_float_value(position)]),
        Err(error @ (RuntimeError::FileIo { .. } | RuntimeError::HostOutputIo { .. })) => {
            Ok(io_failure(&error))
        }
        // The whence is argument 2 and the offset argument 3.
        Err(error @ RuntimeError::TypeError { expected, .. }) => {
            Err(error.at_argument(if expected == "string" { 2 } else { 3 }))
        }
        Err(error) => Err(error),
    }
}

pub(crate) fn file_read(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let file = file_argument(arguments)?;
    runtime
        .file_read_values(file, &arguments[1..])
        .map_err(|error| error.shift_argument(1))
}

pub(crate) fn file_close(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let file = file_argument(arguments)?;
    match runtime.file_close_process(file) {
        Ok(Some(status)) => Ok(process_close_result(runtime, status)),
        Ok(None) => Ok(vec![Value::Boolean(true)]),
        Err(error @ (RuntimeError::FileIo { .. } | RuntimeError::HostOutputIo { .. })) => {
            Ok(io_failure(&error))
        }
        Err(error) => Err(error),
    }
}

pub(crate) fn file_flush(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let file = file_argument(arguments)?;
    match runtime.file_flush(file) {
        Ok(()) => Ok(vec![Value::Boolean(true)]),
        Err(error @ (RuntimeError::FileIo { .. } | RuntimeError::HostOutputIo { .. })) => {
            Ok(io_failure(&error))
        }
        Err(error) => Err(error),
    }
}

pub(crate) fn file_setvbuf(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    let file = file_argument(arguments)?;
    runtime.file_setvbuf(
        file,
        arguments.get(1).unwrap_or(&Value::Nil),
        arguments.get(2),
    )?;
    Ok(vec![Value::Boolean(true)])
}

pub(crate) fn file_lines(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let file = file_argument(arguments)?;
    let formats = arguments.get(1..).unwrap_or_default();
    Ok(vec![runtime.create_file_lines_iterator(
        file, false, formats, "lines", 1,
    )?])
}

pub(crate) fn io_failure(error: &RuntimeError) -> Vec<Value> {
    let code = match error {
        RuntimeError::FileIo { code, .. } | RuntimeError::HostOutputIo { code, .. } => *code,
        _ => 1,
    };
    vec![
        Value::Nil,
        Value::from(error.to_string()),
        Value::Number(f64::from(code)),
    ]
}

fn process_close_result(runtime: &Runtime, status: std::process::ExitStatus) -> Vec<Value> {
    if runtime.lines_formats() == LinesFormats::Ignored {
        return vec![Value::Boolean(true)];
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return vec![
                Value::Nil,
                Value::from("signal"),
                runtime.integer_value(i64::from(signal)),
            ];
        }
    }
    let success = if status.success() {
        Value::Boolean(true)
    } else {
        Value::Nil
    };
    let code = status.code().unwrap_or(-1);
    vec![
        success,
        Value::from("exit"),
        runtime.integer_value(i64::from(code)),
    ]
}

pub(crate) fn read_file_formats(
    runtime: &mut Runtime,
    file: crate::runtime::FileHandle,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    runtime.file_read_values(file, arguments)
}

/// A string argument at `position`, which must be a Lua string.
pub(crate) fn lua_string_argument(
    value: Option<&Value>,
    position: usize,
) -> RuntimeResult<&LuaString> {
    match value {
        Some(Value::String(value)) => Ok(value),
        value => Err(RuntimeError::bad_argument(position, "string", value)),
    }
}

pub(crate) fn math_abs(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    if let Some(Value::Integer(value)) = arguments.first() {
        return Ok(Value::Integer(value.wrapping_abs()));
    }
    unary_math_scalar(runtime, arguments, f64::abs)
}

pub(crate) fn math_acos(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    unary_math_scalar(runtime, arguments, f64::acos)
}

pub(crate) fn math_asin(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    unary_math_scalar(runtime, arguments, f64::asin)
}

pub(crate) fn math_atan(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    unary_math_scalar(runtime, arguments, f64::atan)
}

pub(crate) fn math_atan_modern(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    let y = number_argument(runtime, arguments, 1)?;
    let x = match arguments.get(1) {
        None | Some(Value::Nil) => 1.0,
        Some(_) => number_argument(runtime, arguments, 2)?,
    };
    Ok(Value::Number(y.atan2(x)))
}

pub(crate) fn math_atan2(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let (y, x) = binary_math(runtime, arguments)?;
    Ok(vec![Value::Number(y.atan2(x))])
}

pub(crate) fn math_ceil(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    if let Some(Value::Integer(value)) = arguments.first() {
        return Ok(vec![Value::Integer(*value)]);
    }
    let value = number_argument(runtime, arguments, 1)?.ceil();
    Ok(vec![runtime.integral_float_value(value)])
}

pub(crate) fn math_cos(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    unary_math_scalar(runtime, arguments, f64::cos)
}

pub(crate) fn math_cosh(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    unary_math_scalar(runtime, arguments, f64::cosh)
}

pub(crate) fn math_deg(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    unary_math_scalar(runtime, arguments, |value| {
        value * 180.0 / std::f64::consts::PI
    })
}

pub(crate) fn math_exp(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    unary_math_scalar(runtime, arguments, f64::exp)
}

pub(crate) fn math_fmod(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    if let [Value::Integer(left), Value::Integer(right), ..] = arguments {
        if *right == 0 {
            return Err(RuntimeError::argument_error(2, "zero"));
        }
        let result = if *left == i64::MIN && *right == -1 {
            0
        } else {
            left % right
        };
        return Ok(Value::Integer(result));
    }
    let (left, right) = binary_math(runtime, arguments)?;
    Ok(Value::Number(left % right))
}

pub(crate) fn math_frexp(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let value = number_argument(runtime, arguments, 1)?;
    let (fraction, exponent) = frexp(value);
    Ok(vec![
        Value::Number(fraction),
        Value::Number(f64::from(exponent)),
    ])
}

pub(crate) fn math_ldexp(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let (value, exponent) = binary_math(runtime, arguments)?;
    let exponent = checked_i32(exponent, "math.ldexp exponent")?;
    Ok(vec![Value::Number(scale_by_power_of_two(value, exponent))])
}

pub(crate) fn math_log(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    unary_math_scalar(runtime, arguments, f64::ln)
}

/// `math.log` of Lua 5.3 and later: the natural logarithm, or the logarithm
/// in the base of the second argument (nil counts as absent). Bases 2 and 10
/// use `log2` and `log10`, as the reference `lmathlib.c` does, so
/// `math.log(1000, 10)` is exactly 3.
pub(crate) fn math_log_with_base(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Value> {
    log_with_optional_base(runtime, arguments, numeric_log_base)
}

/// `math.log` of Lua 5.2, which special-cases only base 10.
pub(crate) fn math_log_with_base_lua52(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Value> {
    log_with_optional_base(runtime, arguments, numeric_log_base_lua52)
}

/// `LuaJIT`'s `math.log`: a base multiplies `log2(x)` by `1 / log2(base)`.
/// Unlike Lua 5.2 and later, an explicit nil base is an error.
pub(crate) fn math_log_luajit(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    let value = number_argument(runtime, arguments, 1)?;
    let result = if arguments.len() > 1 {
        numeric_log_base_luajit(value, number_argument(runtime, arguments, 2)?)
    } else {
        value.ln()
    };
    Ok(Value::Number(result))
}

fn log_with_optional_base(
    runtime: &mut Runtime,
    arguments: &[Value],
    with_base: extern "C" fn(f64, f64) -> f64,
) -> RuntimeResult<Value> {
    let value = number_argument(runtime, arguments, 1)?;
    let result = match arguments.get(1) {
        None | Some(Value::Nil) => value.ln(),
        Some(_) => with_base(value, number_argument(runtime, arguments, 2)?),
    };
    Ok(Value::Number(result))
}

/// The logarithm of `value` in `base` for Lua 5.3 to 5.5.
#[allow(
    clippy::float_cmp,
    reason = "the reference compares the base with 2 and 10 exactly"
)]
pub(crate) extern "C" fn numeric_log_base(value: f64, base: f64) -> f64 {
    if base == 2.0 {
        value.log2()
    } else if base == 10.0 {
        value.log10()
    } else {
        value.ln() / base.ln()
    }
}

/// The logarithm of `value` in `base` for Lua 5.2.
#[allow(
    clippy::float_cmp,
    reason = "the reference compares the base with 2 and 10 exactly"
)]
pub(crate) extern "C" fn numeric_log_base_lua52(value: f64, base: f64) -> f64 {
    if base == 10.0 {
        value.log10()
    } else {
        value.ln() / base.ln()
    }
}

/// The logarithm of `value` in `base` for `LuaJIT`.
pub(crate) extern "C" fn numeric_log_base_luajit(value: f64, base: f64) -> f64 {
    value.log2() * (1.0 / base.log2())
}

pub(crate) extern "C" fn numeric_log(value: f64) -> f64 {
    value.ln()
}

pub(crate) fn math_log10(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    unary_math_scalar(runtime, arguments, f64::log10)
}

pub(crate) fn math_max_scalar(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    if let [Value::Number(left), Value::Number(right)] = arguments {
        return Ok(Value::Number(numeric_max(*left, *right)));
    }
    if runtime.tagged_integers()
        && let Some(value) = tagged_extreme(arguments, numeric_less)
    {
        return Ok(value);
    }
    math_extreme(runtime, arguments, |result, value| value > result)
}

pub(crate) fn math_min_scalar(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    if runtime.tagged_integers()
        && let Some(value) = tagged_extreme(arguments, |result, value| numeric_less(value, result))
    {
        return Ok(value);
    }
    math_extreme(runtime, arguments, |result, value| value < result)
}

/// `LuaJIT`'s `math.max`: a later argument replaces the running result unless
/// it compares less, so a NaN replaces and is replaced, and equal values
/// (`0` and `-0`) take the later one.
// The NaN-aware comparison is the point.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub(crate) fn math_max_scalar_luajit(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Value> {
    math_extreme(runtime, arguments, |result, value| !(value < result))
}

/// `LuaJIT`'s `math.min`, the mirror of [`math_max_scalar_luajit`].
// The NaN-aware comparison is the point.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub(crate) fn math_min_scalar_luajit(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Value> {
    math_extreme(runtime, arguments, |result, value| !(value > result))
}

#[allow(clippy::float_cmp, reason = "a zero fraction is exactly zero")]
pub(crate) fn math_modf(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    if let Some(Value::Integer(value)) = arguments.first() {
        return Ok(vec![Value::Integer(*value), Value::Number(0.0)]);
    }
    let value = number_argument(runtime, arguments, 1)?;
    // Before Lua 5.3 the fraction is the C `modf`'s, which has the sign of
    // the argument even when it is zero (`math.modf (-3)` gives `-3, -0`).
    let signed = !runtime.tagged_integers();
    let zero = if signed { 0.0_f64.copysign(value) } else { 0.0 };
    if value.is_nan() {
        return Ok(vec![Value::Number(value), Value::Number(value)]);
    }
    if !value.is_finite() {
        return Ok(vec![Value::Number(value), Value::Number(zero)]);
    }
    let integer = value.trunc();
    let fraction = value - integer;
    Ok(vec![
        runtime.integral_float_value(integer),
        Value::Number(if signed && fraction == 0.0 {
            zero
        } else {
            fraction
        }),
    ])
}

pub(crate) fn math_pow(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    let (left, right) = binary_math(runtime, arguments)?;
    Ok(Value::Number(left.powf(right)))
}

pub(crate) extern "C" fn numeric_pow(left: f64, right: f64) -> f64 {
    left.powf(right)
}

pub(crate) fn math_rad(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    unary_math_scalar(runtime, arguments, |value| {
        value * std::f64::consts::PI / 180.0
    })
}

pub(crate) fn math_sinh(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    unary_math_scalar(runtime, arguments, f64::sinh)
}

pub(crate) fn math_sqrt(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    unary_math_scalar(runtime, arguments, f64::sqrt)
}

pub(crate) fn math_tan(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    unary_math_scalar(runtime, arguments, f64::tan)
}

pub(crate) fn math_tanh(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    unary_math_scalar(runtime, arguments, f64::tanh)
}

pub(crate) extern "C" fn numeric_acos(value: f64) -> f64 {
    value.acos()
}

pub(crate) extern "C" fn numeric_asin(value: f64) -> f64 {
    value.asin()
}

pub(crate) extern "C" fn numeric_cos(value: f64) -> f64 {
    value.cos()
}

pub(crate) extern "C" fn numeric_cosh(value: f64) -> f64 {
    value.cosh()
}

pub(crate) extern "C" fn numeric_deg(value: f64) -> f64 {
    value * 180.0 / std::f64::consts::PI
}

pub(crate) extern "C" fn numeric_exp(value: f64) -> f64 {
    value.exp()
}

pub(crate) extern "C" fn numeric_log10(value: f64) -> f64 {
    value.log10()
}

pub(crate) extern "C" fn numeric_rad(value: f64) -> f64 {
    value * std::f64::consts::PI / 180.0
}

pub(crate) extern "C" fn numeric_fmod(left: f64, right: f64) -> f64 {
    left % right
}

pub(crate) extern "C" fn numeric_sin(value: f64) -> f64 {
    value.sin()
}

pub(crate) extern "C" fn numeric_sinh(value: f64) -> f64 {
    value.sinh()
}

// Never inlined, so one definition exists and its address names the function:
// the trace emitter finds the square root by that address.
#[inline(never)]
pub(crate) extern "C" fn numeric_sqrt(value: f64) -> f64 {
    value.sqrt()
}

/// The address of [`numeric_sqrt`], which the trace emitter replaces with the
/// machine's square root instruction.
#[inline(never)]
pub(crate) fn numeric_sqrt_address() -> usize {
    numeric_sqrt as extern "C" fn(f64) -> f64 as usize
}

pub(crate) extern "C" fn numeric_tan(value: f64) -> f64 {
    value.tan()
}

pub(crate) extern "C" fn numeric_tanh(value: f64) -> f64 {
    value.tanh()
}

#[inline]
pub(crate) fn unary_math_scalar(
    runtime: &Runtime,
    arguments: &[Value],
    function: fn(f64) -> f64,
) -> RuntimeResult<Value> {
    Ok(Value::Number(function(number_argument(
        runtime, arguments, 1,
    )?)))
}

/// Argument `position` of a library function, converted to a number.
#[inline]
pub(crate) fn number_argument(
    runtime: &Runtime,
    arguments: &[Value],
    position: usize,
) -> RuntimeResult<f64> {
    let argument = arguments.get(position - 1);
    // Most arguments are numbers already; the coercion is out of line.
    if let Some(&Value::Number(number)) = argument {
        return Ok(number);
    }
    argument
        .and_then(|value| runtime.to_number(value))
        .ok_or_else(|| RuntimeError::bad_argument(position, "number", argument))
}

pub(crate) fn binary_math(runtime: &Runtime, arguments: &[Value]) -> RuntimeResult<(f64, f64)> {
    Ok((
        number_argument(runtime, arguments, 1)?,
        number_argument(runtime, arguments, 2)?,
    ))
}

pub(crate) fn math_extreme(
    runtime: &Runtime,
    arguments: &[Value],
    replaces: fn(f64, f64) -> bool,
) -> RuntimeResult<Value> {
    // Lua 5.3 and later check for any value before reading numbers.
    if arguments.is_empty() && runtime.tagged_integers() {
        return Err(RuntimeError::argument_error(1, "value expected"));
    }
    let mut result = number_argument(runtime, arguments, 1)?;
    for position in 2..=arguments.len() {
        let value = number_argument(runtime, arguments, position)?;
        if replaces(result, value) {
            result = value;
        }
    }
    Ok(Value::Number(result))
}

/// Picks the winning argument of Lua 5.3+ `math.max` or `math.min` without
/// converting it, so the result keeps its integer or float subtype. Returns
/// `None` unless every argument is already a number.
fn tagged_extreme(arguments: &[Value], replaces: fn(&Value, &Value) -> bool) -> Option<Value> {
    let (first, rest) = arguments.split_first()?;
    if !arguments
        .iter()
        .all(|value| matches!(value, Value::Integer(_) | Value::Number(_)))
    {
        return None;
    }
    let mut result = first;
    for value in rest {
        if replaces(result, value) {
            result = value;
        }
    }
    Some(result.clone())
}

/// Lua's `<` on two numbers, exact across the integer and float subtypes.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Range checks precede each conversion.
fn numeric_less(left: &Value, right: &Value) -> bool {
    const LIMIT: f64 = 9_223_372_036_854_775_808.0;
    match (left, right) {
        (Value::Integer(left), Value::Integer(right)) => left < right,
        (Value::Number(left), Value::Number(right)) => left < right,
        // For an integer i, i < f exactly when i < ceil(f).
        (Value::Integer(left), Value::Number(right)) => {
            if right.is_nan() || *right <= -LIMIT {
                false
            } else if *right >= LIMIT {
                true
            } else {
                *left < right.ceil() as i64
            }
        }
        // For an integer i, f < i exactly when floor(f) < i.
        (Value::Number(left), Value::Integer(right)) => {
            if left.is_nan() || *left >= LIMIT {
                false
            } else if *left < -LIMIT {
                true
            } else {
                (left.floor() as i64) < *right
            }
        }
        _ => false,
    }
}

pub(crate) extern "C" fn numeric_max(left: f64, right: f64) -> f64 {
    if right > left { right } else { left }
}

pub(crate) extern "C" fn numeric_min(left: f64, right: f64) -> f64 {
    if right < left { right } else { left }
}

pub(crate) extern "C" fn numeric_max_luajit(left: f64, right: f64) -> f64 {
    if right < left { left } else { right }
}

pub(crate) extern "C" fn numeric_min_luajit(left: f64, right: f64) -> f64 {
    if right > left { left } else { right }
}

pub(crate) fn checked_i32(value: f64, description: &str) -> RuntimeResult<i32> {
    if !value.is_finite()
        || value.fract() != 0.0
        || value < f64::from(i32::MIN)
        || value > f64::from(i32::MAX)
    {
        return Err(RuntimeError::message(format!(
            "{description} must be an integer"
        )));
    }
    value
        .to_string()
        .parse()
        .map_err(|_| RuntimeError::message(format!("{description} must be an integer")))
}

pub(crate) fn frexp(value: f64) -> (f64, i32) {
    if value == 0.0 || !value.is_finite() {
        return (value, 0);
    }

    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    if exponent == 0 {
        let (fraction, exponent) = frexp(value * 4_503_599_627_370_496.0);
        return (fraction, exponent - 52);
    }
    let fraction_bits = (bits & !(0x7ff_u64 << 52)) | (1022_u64 << 52);
    (f64::from_bits(fraction_bits), exponent - 1022)
}

pub(crate) fn scale_by_power_of_two(value: f64, exponent: i32) -> f64 {
    if value == 0.0 || !value.is_finite() {
        return value;
    }

    let (_, value_exponent) = frexp(value);
    let result_exponent = i64::from(value_exponent) + i64::from(exponent);
    if result_exponent > 1024 {
        return f64::INFINITY.copysign(value);
    }
    if result_exponent < -1074 {
        return 0.0_f64.copysign(value);
    }

    let mut result = value;
    let mut remaining = exponent;
    while remaining != 0 {
        let step = remaining.clamp(-512, 512);
        result *= 2.0_f64.powi(step);
        remaining -= step;
    }
    result
}
