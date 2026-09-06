//! Base, os, table, and metatable builtins.

use crate::runtime::{
    ExitRequest, NonFiniteNumerals, PlainNumeral, Runtime, RuntimeError, RuntimeResult,
    StackBuffer, TableHandle, Value, scan_plain_numeral,
};
use crate::stdlib::args::{
    append_print_bytes, isize_to_number, metatable_argument, number_to_isize,
    optional_integer_argument, select_index, table_argument, table_list_length,
    table_list_position, usize_integer_value, usize_to_number, validate_locale_category,
};
use crate::stdlib::format::append_table_concat_bytes;
use crate::stdlib::math::{integer_i64_argument, io_failure};
use crate::stdlib::strings::{
    optional_coerced_index, parse_integer_base, parse_lua_integer_exact, parse_lua_number,
    parse_strtoul_base, parse_wrapping_integer_base, string_argument, string_argument_in,
};
use crate::stdlib::{CalendarTime, TEMP_NAME_COUNTER};
use crate::string::{INLINE_STRING_BYTES, LuaString};
use chrono::{
    DateTime, Datelike, Duration, Local, LocalResult, NaiveDate, TimeZone, Timelike, Utc, Weekday,
};
use std::borrow::Cow;
#[cfg(unix)]
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
#[cfg(unix)]
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::Ordering;

/// The message of a failed `assert` that carries none.
const ASSERT_MESSAGE: &str = "assertion failed!";

/// Lua 5.3 to 5.5's `assert`: return all supplied values when the first is
/// truthy, or raise the second value unchanged, as `error` does at level 1
/// (a string gets the position of the caller).
///
/// # Errors
///
/// Returns the message when the first argument is false or `nil`.
pub fn assert(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    if arguments.is_empty() {
        return Err(RuntimeError::argument_error(1, "value expected"));
    }
    if arguments.first().is_some_and(Value::is_truthy) {
        return Ok(arguments.to_vec());
    }
    match arguments.get(1) {
        None => Err(RuntimeError::message(ASSERT_MESSAGE)),
        Some(Value::Nil) => Err(RuntimeError::RaisedNil),
        Some(value) => Err(RuntimeError::RaisedAtLevel {
            value: value.clone(),
            level: 1,
        }),
    }
}

/// Lua 5.1 and 5.2's `assert`: the message is `luaL_optstring`, so a number
/// becomes text and any other value but nil is an argument error.
///
/// # Errors
///
/// Returns the message when the first argument is false or `nil`.
pub fn assert_string_message(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    if arguments.is_empty() {
        return Err(RuntimeError::argument_error(1, "value expected"));
    }
    if arguments.first().is_some_and(Value::is_truthy) {
        return Ok(arguments.to_vec());
    }
    match arguments.get(1) {
        None | Some(Value::Nil) => Err(RuntimeError::message(ASSERT_MESSAGE)),
        Some(Value::String(message)) => Err(RuntimeError::message(message.to_lossy().into_owned())),
        Some(value @ (Value::Integer(_) | Value::Number(_))) => {
            let text = string_argument_in(runtime.number_format(), Some(value), 2)?;
            Err(RuntimeError::message(
                String::from_utf8_lossy(&text).into_owned(),
            ))
        }
        Some(value) => Err(RuntimeError::bad_argument(2, "string", Some(value))),
    }
}

/// Lua 5.2's `assert`: as Lua 5.1's, except that `luaB_assert` has no
/// `luaL_checkany`, so a call with no argument fails like a false one.
///
/// # Errors
///
/// Returns the message when the first argument is false, `nil` or missing.
pub fn assert_lua52(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    if arguments.is_empty() {
        return Err(RuntimeError::message(ASSERT_MESSAGE));
    }
    assert_string_message(runtime, arguments)
}

/// `LuaJIT`'s `assert`: a string or number message gets the caller's position,
/// any other value, nil included, is raised unchanged.
///
/// # Errors
///
/// Returns the message when the first argument is false or `nil`.
pub fn assert_luajit(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    if arguments.is_empty() {
        return Err(RuntimeError::argument_error(1, "value expected"));
    }
    if arguments.first().is_some_and(Value::is_truthy) {
        return Ok(arguments.to_vec());
    }
    match arguments.get(1) {
        None => Err(RuntimeError::message(ASSERT_MESSAGE)),
        Some(Value::String(message)) => Err(RuntimeError::message(message.to_lossy().into_owned())),
        Some(value @ (Value::Integer(_) | Value::Number(_))) => {
            let text = string_argument_in(runtime.number_format(), Some(value), 2)?;
            Err(RuntimeError::message(
                String::from_utf8_lossy(&text).into_owned(),
            ))
        }
        Some(Value::Nil) => Err(RuntimeError::RaisedNil),
        Some(value) => Err(RuntimeError::Raised(value.clone())),
    }
}

/// Raises a Lua error using its first argument as the diagnostic message.
///
/// # Errors
///
/// Always returns the requested Lua error.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn error(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    match arguments.first() {
        None | Some(Value::Nil) => Err(RuntimeError::RaisedNil),
        Some(value) => {
            let level = match arguments.get(1) {
                None | Some(Value::Nil) => 1,
                Some(Value::Integer(level)) => usize::try_from(*level).unwrap_or(0),
                Some(Value::Number(level)) if level.is_finite() => {
                    level.trunc().max(0.0).min(usize::MAX as f64) as usize
                }
                Some(Value::String(level)) => {
                    let level = parse_lua_number(level.as_bytes()).ok_or({
                        RuntimeError::TypeError {
                            operation: "use as an error level",
                            expected: "number",
                            found: "string",
                        }
                    })?;
                    level.trunc().max(0.0).min(usize::MAX as f64) as usize
                }
                Some(level) => {
                    return Err(RuntimeError::TypeError {
                        operation: "use as an error level",
                        expected: "number",
                        found: level.type_name(),
                    });
                }
            };
            let value = match value {
                Value::Integer(_) | Value::Number(_)
                    if level > 0 && runtime.error_number_positions() =>
                {
                    let text = string_argument_in(runtime.number_format(), Some(value), 1)?;
                    Value::from(text.into_owned())
                }
                value => value.clone(),
            };
            Err(RuntimeError::RaisedAtLevel { value, level })
        }
    }
}

/// Queries or selects the fixed `"C"` locale without mutating host state.
///
/// # Errors
///
/// Returns an error for an unsupported locale category.
pub fn os_setlocale(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let locale = match arguments.first() {
        None | Some(Value::Nil) => None,
        argument => Some(string_argument(argument, 1)?),
    };
    validate_locale_category(arguments.get(1))?;
    let value = match locale {
        None => Value::from("C"),
        Some(locale) if locale.as_ref() == b"C" => Value::from("C"),
        Some(_) => Value::Nil,
    };
    Ok(vec![value])
}

/// Returns process CPU time where the host exposes it.
///
/// # Errors
///
/// Returns an error when the target has no host clock. Extra arguments are ignored.
pub fn os_clock(runtime: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
    let seconds = runtime.process_cpu_seconds();
    if !seconds.is_finite() {
        return Err(RuntimeError::message(
            "host clock is unavailable on this target",
        ));
    }
    Ok(vec![Value::Number(seconds)])
}

/// Formats a Unix timestamp with Lua 5.1's fixed-C-locale date interface.
///
/// A leading `!` selects UTC. The `*t` form returns the fields consumed by
/// [`os_time`]; all other formats use the C `strftime` conversion names that
/// Lua exposes. Oximond deliberately keeps this independent of the process
/// locale, like its `os.setlocale` implementation.
///
/// # Errors
///
/// Returns a Lua argument error for a non-string format or non-numeric time.
pub fn os_date(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let format = match arguments.first() {
        None | Some(Value::Nil) => Cow::Borrowed(b"%c".as_slice()),
        argument => string_argument_in(runtime.number_format(), argument, 1)?,
    };
    let mut format = format.as_ref();
    let utc = format.first() == Some(&b'!');
    if utc {
        format = &format[1..];
    }
    let timestamp = match arguments.get(1) {
        None | Some(Value::Nil) => system_timestamp()?,
        Some(value) => lua_timestamp(value, 2)?,
    };
    let Some(calendar) = CalendarTime::at(timestamp, utc) else {
        return Ok(vec![Value::Nil]);
    };
    if format == b"*t" {
        let table = runtime.create_table()?;
        calendar.write_lua_table(runtime, table)?;
        return Ok(vec![Value::Table(table)]);
    }
    Ok(vec![Value::bytes(calendar.format(format)?)])
}

/// Returns the elapsed seconds between two Lua time values.
///
/// Lua 5.1 exposes `difftime` rather than promising that direct subtraction
/// works for every host `time_t`. Oximond stores timestamps as seconds, so
/// subtraction retains the familiar numeric result.
///
/// # Errors
///
/// Returns a type error unless both values are numeric or numeric strings.
pub fn os_difftime(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let first = lua_time_number(arguments.first(), 1)?;
    // Lua 5.1 and 5.2 default the second time to zero.
    let second = match arguments.get(1) {
        None | Some(Value::Nil) if !runtime.tagged_integers() => 0.0,
        argument => lua_time_number(argument, 2)?,
    };
    Ok(vec![Value::Number(first - second)])
}

/// Returns the current local timestamp, or converts a Lua calendar table to
/// one. Required date fields mirror Lua 5.1: `year`, `month`, and `day`.
/// `hour`, `min`, and `sec` default to noon, zero, and zero respectively.
///
/// # Errors
///
/// Returns an argument error for malformed calendar tables or values outside
/// the representable civil-date range.
#[allow(clippy::cast_precision_loss)]
pub fn os_time(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let Some(argument) = arguments.first() else {
        return Ok(vec![runtime.integer_value(system_timestamp()?)]);
    };
    if matches!(argument, Value::Nil) {
        return Ok(vec![runtime.integer_value(system_timestamp()?)]);
    }
    let table = table_argument(Some(argument), 1)?;
    // Lua 5.1 and 5.2 read the day first; Lua 5.3 and later the year.
    let (year, month, day) = if runtime.tagged_integers() {
        let year = calendar_field(runtime, table, "year", None)?;
        let month = calendar_field(runtime, table, "month", None)?;
        (year, month, calendar_field(runtime, table, "day", None)?)
    } else {
        let day = calendar_field(runtime, table, "day", None)?;
        let month = calendar_field(runtime, table, "month", None)?;
        (calendar_field(runtime, table, "year", None)?, month, day)
    };
    let hour = calendar_field(runtime, table, "hour", Some(12))?;
    let minute = calendar_field(runtime, table, "min", Some(0))?;
    let second = calendar_field(runtime, table, "sec", Some(0))?;
    let isdst = match runtime.table_get(table, &Value::from("isdst"))? {
        Value::Nil => None,
        Value::Boolean(value) => Some(value),
        _ => Some(true),
    };
    let local = normalized_local_time(year, month, day, hour, minute, second)?;
    let Some(timestamp) = resolve_local_time(local, isdst) else {
        return Ok(vec![Value::Nil]);
    };
    Ok(vec![runtime.integer_value(timestamp)])
}

#[cfg(all(target_family = "wasm", target_os = "unknown"))]
pub(crate) fn system_timestamp() -> RuntimeResult<i64> {
    Err(RuntimeError::message(
        "host clock is unavailable on this target",
    ))
}

#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
pub(crate) fn system_timestamp() -> RuntimeResult<i64> {
    let now = std::time::SystemTime::now();
    match now.duration_since(std::time::UNIX_EPOCH) {
        Ok(duration) => i64::try_from(duration.as_secs())
            .map_err(|_| RuntimeError::message("system time is out of range")),
        Err(error) => {
            let seconds = i64::try_from(error.duration().as_secs())
                .map_err(|_| RuntimeError::message("system time is out of range"))?;
            Ok(-seconds)
        }
    }
}

/// A time at argument `position`.
pub(crate) fn lua_time_number(argument: Option<&Value>, position: usize) -> RuntimeResult<f64> {
    let number = argument
        .and_then(|value| Runtime::coerce_number(value, "").ok())
        .ok_or_else(|| RuntimeError::bad_argument(position, "number", argument))?;
    if number.is_finite() {
        Ok(number)
    } else {
        Err(RuntimeError::message("time must be a finite number"))
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub(crate) fn lua_timestamp(value: &Value, position: usize) -> RuntimeResult<i64> {
    let number = lua_time_number(Some(value), position)?;
    if number < i64::MIN as f64 || number > i64::MAX as f64 {
        return Err(RuntimeError::message("time is out of range"));
    }
    Ok(number.trunc() as i64)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub(crate) fn calendar_field(
    runtime: &Runtime,
    table: TableHandle,
    name: &'static str,
    default: Option<i64>,
) -> RuntimeResult<i64> {
    let value = runtime.table_get(table, &Value::from(name))?;
    // Lua 5.1 and 5.2 treat a field that is not a number as missing.
    let number = Runtime::coerce_number(&value, "").ok();
    let Some(number) = number else {
        if matches!(value, Value::Nil) || !runtime.tagged_integers() {
            return default.ok_or_else(|| {
                RuntimeError::message(format!("field '{name}' missing in date table"))
            });
        }
        return Err(RuntimeError::message(format!(
            "field '{name}' is not an integer"
        )));
    };
    if runtime.tagged_integers() && number.fract() != 0.0 {
        return Err(RuntimeError::message(format!(
            "field '{name}' is not an integer"
        )));
    }
    if !number.is_finite() || number < i64::MIN as f64 || number > i64::MAX as f64 {
        return Err(RuntimeError::message(format!(
            "bad argument #1 to 'os.time' (field '{name}' is out of range)"
        )));
    }
    Ok(number.trunc() as i64)
}

pub(crate) fn normalized_local_time(
    year: i64,
    month: i64,
    day: i64,
    hour: i64,
    minute: i64,
    second: i64,
) -> RuntimeResult<chrono::NaiveDateTime> {
    let month_offset = month
        .checked_sub(1)
        .ok_or_else(|| RuntimeError::message("calendar month is out of range"))?;
    let normalized_year = year
        .checked_add(month_offset.div_euclid(12))
        .ok_or_else(|| RuntimeError::message("calendar year is out of range"))?;
    let normalized_month = month_offset.rem_euclid(12) + 1;
    let year = i32::try_from(normalized_year)
        .map_err(|_| RuntimeError::message("calendar year is out of range"))?;
    let month = u32::try_from(normalized_month)
        .map_err(|_| RuntimeError::message("calendar month is out of range"))?;
    let date = NaiveDate::from_ymd_opt(year, month, 1)
        .ok_or_else(|| RuntimeError::message("calendar date is out of range"))?;
    let day_offset = day
        .checked_sub(1)
        .ok_or_else(|| RuntimeError::message("calendar day is out of range"))?;
    let seconds = hour
        .checked_mul(3_600)
        .and_then(|value| {
            minute
                .checked_mul(60)
                .and_then(|minute| value.checked_add(minute))
        })
        .and_then(|value| value.checked_add(second))
        .ok_or_else(|| RuntimeError::message("calendar time is out of range"))?;
    date.and_hms_opt(0, 0, 0)
        .and_then(|time| time.checked_add_signed(Duration::days(day_offset)))
        .and_then(|time| time.checked_add_signed(Duration::seconds(seconds)))
        .ok_or_else(|| RuntimeError::message("calendar time is out of range"))
}

pub(crate) fn resolve_local_time(local: chrono::NaiveDateTime, isdst: Option<bool>) -> Option<i64> {
    match Local.from_local_datetime(&local) {
        LocalResult::Single(time) => Some(time.timestamp()),
        // The earlier occurrence at a fall-back boundary is the daylight-time
        // one on the platforms Oximond supports. This also preserves the
        // instant returned by os.date("*t") when its isdst field is retained.
        LocalResult::Ambiguous(earlier, later) => match isdst {
            Some(false) => Some(later.timestamp()),
            Some(true) | None => Some(earlier.timestamp()),
        },
        // Spring-forward local wall times do not exist. C's mktime usually
        // normalizes them, but returning nil is safer than inventing an
        // instant when the host timezone cannot represent the request.
        LocalResult::None => None,
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
impl CalendarTime {
    pub(crate) fn at(timestamp: i64, utc: bool) -> Option<Self> {
        let instant = DateTime::<Utc>::from_timestamp(timestamp, 0)?;
        if utc {
            return Some(Self::from_utc(instant));
        }
        Some(Self::from_local(instant.with_timezone(&Local)))
    }

    pub(crate) fn from_utc(time: DateTime<Utc>) -> Self {
        Self::from_parts(&time, 0, "UTC".to_owned(), false)
    }

    pub(crate) fn from_local(time: DateTime<Local>) -> Self {
        let offset = time.offset().local_minus_utc();
        let zone = time.format("%Z").to_string();
        let isdst = local_is_dst(&time, offset);
        Self::from_parts(&time, offset, zone, isdst)
    }

    pub(crate) fn from_parts<Tz: TimeZone>(
        time: &DateTime<Tz>,
        offset_seconds: i32,
        zone: String,
        isdst: bool,
    ) -> Self {
        let iso = time.iso_week();
        Self {
            year: time.year(),
            month: time.month(),
            day: time.day(),
            hour: time.hour(),
            minute: time.minute(),
            second: time.second(),
            weekday: time.weekday(),
            ordinal: time.ordinal(),
            iso_year: iso.year(),
            iso_week: iso.week(),
            offset_seconds,
            zone,
            isdst,
        }
    }

    pub(crate) fn write_lua_table(
        &self,
        runtime: &mut Runtime,
        table: TableHandle,
    ) -> RuntimeResult<()> {
        for (name, value) in [
            ("year", runtime.integer_value(i64::from(self.year))),
            ("month", runtime.integer_value(i64::from(self.month))),
            ("day", runtime.integer_value(i64::from(self.day))),
            ("hour", runtime.integer_value(i64::from(self.hour))),
            ("min", runtime.integer_value(i64::from(self.minute))),
            ("sec", runtime.integer_value(i64::from(self.second))),
            (
                "wday",
                runtime.integer_value(i64::from(self.weekday.num_days_from_sunday() + 1)),
            ),
            ("yday", runtime.integer_value(i64::from(self.ordinal))),
            ("isdst", Value::Boolean(self.isdst)),
        ] {
            runtime.table_set(table, &Value::from(name), value)?;
        }
        Ok(())
    }

    pub(crate) fn format(&self, format: &[u8]) -> RuntimeResult<Vec<u8>> {
        let mut output = Vec::with_capacity(format.len().saturating_add(16));
        let mut index = 0;
        while index < format.len() {
            if format[index] != b'%' {
                output.push(format[index]);
                index += 1;
                continue;
            }
            index += 1;
            let Some(&specifier) = format.get(index) else {
                return Err(RuntimeError::message("invalid conversion specifier '%'"));
            };
            index += 1;
            self.append_format(&mut output, specifier)?;
        }
        Ok(output)
    }

    pub(crate) fn append_format(&self, output: &mut Vec<u8>, specifier: u8) -> RuntimeResult<()> {
        const WEEKDAY_SHORT: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
        const WEEKDAY_LONG: [&str; 7] = [
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
        ];
        const MONTH_SHORT: [&str; 12] = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ];
        const MONTH_LONG: [&str; 12] = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ];
        let weekday = self.weekday.num_days_from_sunday() as usize;
        let month = (self.month - 1) as usize;
        let hour12 = match self.hour % 12 {
            0 => 12,
            value => value,
        };
        match specifier {
            b'a' => output.extend_from_slice(WEEKDAY_SHORT[weekday].as_bytes()),
            b'A' => output.extend_from_slice(WEEKDAY_LONG[weekday].as_bytes()),
            b'b' | b'h' => output.extend_from_slice(MONTH_SHORT[month].as_bytes()),
            b'B' => output.extend_from_slice(MONTH_LONG[month].as_bytes()),
            b'c' => self.append_composite(output, b"%a %b %e %H:%M:%S %Y")?,
            b'C' => append_decimal(output, self.year.div_euclid(100), 2, b'0'),
            b'd' => append_decimal(output, self.day as i32, 2, b'0'),
            b'D' | b'x' => self.append_composite(output, b"%m/%d/%y")?,
            b'e' => append_decimal(output, self.day as i32, 2, b' '),
            b'F' => self.append_composite(output, b"%Y-%m-%d")?,
            b'g' => append_decimal(output, self.iso_year.rem_euclid(100), 2, b'0'),
            b'G' => append_decimal(output, self.iso_year, 4, b'0'),
            b'H' => append_decimal(output, self.hour as i32, 2, b'0'),
            b'I' => append_decimal(output, hour12 as i32, 2, b'0'),
            b'j' => append_decimal(output, self.ordinal as i32, 3, b'0'),
            b'k' => append_decimal(output, self.hour as i32, 2, b' '),
            b'l' => append_decimal(output, hour12 as i32, 2, b' '),
            b'm' => append_decimal(output, self.month as i32, 2, b'0'),
            b'M' => append_decimal(output, self.minute as i32, 2, b'0'),
            b'n' => output.push(b'\n'),
            b'p' => output.extend_from_slice(if self.hour < 12 { b"AM" } else { b"PM" }),
            b'P' => output.extend_from_slice(if self.hour < 12 { b"am" } else { b"pm" }),
            b'r' => self.append_composite(output, b"%I:%M:%S %p")?,
            b'R' => self.append_composite(output, b"%H:%M")?,
            b'S' => append_decimal(output, self.second as i32, 2, b'0'),
            b't' => output.push(b'\t'),
            b'T' | b'X' => self.append_composite(output, b"%H:%M:%S")?,
            b'u' => append_decimal(output, self.weekday.number_from_monday() as i32, 1, b'0'),
            b'U' => append_decimal(output, self.week_number(Weekday::Sun) as i32, 2, b'0'),
            b'V' => append_decimal(output, self.iso_week as i32, 2, b'0'),
            b'w' => append_decimal(output, self.weekday.num_days_from_sunday() as i32, 1, b'0'),
            b'W' => append_decimal(output, self.week_number(Weekday::Mon) as i32, 2, b'0'),
            b'y' => append_decimal(output, self.year.rem_euclid(100), 2, b'0'),
            b'Y' => append_decimal(output, self.year, 4, b'0'),
            b'z' => append_offset(output, self.offset_seconds),
            b'Z' => output.extend_from_slice(self.zone.as_bytes()),
            b'%' => output.push(b'%'),
            _ => {
                return Err(RuntimeError::message(format!(
                    "invalid conversion specifier '%{}'",
                    specifier as char
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn append_composite(
        &self,
        output: &mut Vec<u8>,
        format: &[u8],
    ) -> RuntimeResult<()> {
        for &specifier in format {
            if specifier == b'%' {
                continue;
            }
            if matches!(specifier, b' ' | b':' | b'/' | b'-') {
                output.push(specifier);
            } else {
                self.append_format(output, specifier)?;
            }
        }
        Ok(())
    }

    pub(crate) fn week_number(&self, start: Weekday) -> u32 {
        let jan1 = NaiveDate::from_ymd_opt(self.year, 1, 1)
            .expect("a calendar year from chrono always has January 1");
        let days_to_start = (7 + i64::from(start.num_days_from_sunday())
            - i64::from(jan1.weekday().num_days_from_sunday()))
            % 7;
        let zero_based_day = i64::from(self.ordinal - 1);
        if zero_based_day < days_to_start {
            0
        } else {
            ((zero_based_day - days_to_start) / 7 + 1) as u32
        }
    }
}

pub(crate) fn local_is_dst(time: &DateTime<Local>, offset: i32) -> bool {
    // Chrono intentionally abstracts away platform-specific DST flags. A
    // standard offset is the smallest offset observed at local noon this
    // year, which is correct for ordinary positive DST shifts and still gives
    // Lua its required boolean on unusual timezone data.
    let standard = (1..=12)
        .filter_map(|month| {
            NaiveDate::from_ymd_opt(time.year(), month, 1)
                .and_then(|date| date.and_hms_opt(12, 0, 0))
                .and_then(|date| match Local.from_local_datetime(&date) {
                    LocalResult::Single(value) => Some(value.offset().local_minus_utc()),
                    LocalResult::Ambiguous(first, _) => Some(first.offset().local_minus_utc()),
                    LocalResult::None => None,
                })
        })
        .min()
        .unwrap_or(offset);
    offset > standard
}

pub(crate) fn append_decimal(output: &mut Vec<u8>, value: i32, width: usize, padding: u8) {
    let text = value.to_string();
    if text.len() < width {
        output.extend(std::iter::repeat_n(padding, width - text.len()));
    }
    output.extend_from_slice(text.as_bytes());
}

pub(crate) fn append_offset(output: &mut Vec<u8>, offset_seconds: i32) {
    let sign = if offset_seconds < 0 { b'-' } else { b'+' };
    let seconds = offset_seconds.unsigned_abs();
    output.push(sign);
    append_decimal(
        output,
        i32::try_from(seconds / 3_600).expect("UTC offset hours fit in i32"),
        2,
        b'0',
    );
    append_decimal(
        output,
        i32::try_from((seconds % 3_600) / 60).expect("UTC offset minutes fit in i32"),
        2,
        b'0',
    );
}

/// Runs one shell command using the host shell without interpolating command
/// text in Rust. The returned number follows Lua 5.1's `system`-style status
/// convention on Unix, including a nonzero no-argument shell availability
/// probe.
///
/// # Errors
///
/// Returns a type error when the optional command is neither a string nor a
/// number. A host process-launch failure returns `-1`, matching `system`.
pub fn os_execute(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let command = match arguments.first() {
        None | Some(Value::Nil) => None,
        Some(Value::String(command)) => Some(command.to_lossy().into_owned()),
        Some(Value::Number(number)) => {
            Some(String::from_utf8_lossy(&runtime.number_format().format(*number)).into_owned())
        }
        argument => return Err(RuntimeError::bad_argument(1, "string", argument)),
    };
    let status = match command {
        Some(command) => run_shell(&command),
        None => shell_available(),
    };
    Ok(vec![Value::Number(f64::from(status))])
}

/// Reads a host environment variable using Lua's string-or-number argument rule.
/// An absent or invalid name returns `nil`.
///
/// # Errors
///
/// Returns a type error when the name is not a string or number.
pub fn os_getenv(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let name = string_argument_in(runtime.number_format(), arguments.first(), 1)?;
    if name.is_empty() || name.contains(&b'=') || name.contains(&0) {
        return Ok(vec![Value::Nil]);
    }
    #[cfg(unix)]
    let value = std::env::var_os(OsString::from_vec(name.into_owned()))
        .map(|value| Value::from(value.as_bytes().to_vec()));
    #[cfg(not(unix))]
    let value = std::str::from_utf8(&name)
        .ok()
        .and_then(std::env::var_os)
        .map(|value| Value::from(value.to_string_lossy().into_owned()));
    Ok(vec![value.unwrap_or(Value::Nil)])
}

/// Lua 5.1's `os.exit([code])`: `luaL_optint` truncates a numeric status.
///
/// # Errors
///
/// Returns [`RuntimeError::Exit`] to end the program, or a type error for a
/// status that is not a number.
pub fn os_exit_lua51(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    Err(RuntimeError::Exit(ExitRequest {
        status: truncated_status(runtime, arguments.first())?,
        close: false,
    }))
}

/// `os.exit([code [, close]])` for the `lua52` and `luajit21` profiles, whose
/// `luaL_optint` and `lj_lib_optint` truncate a numeric status. `true` and
/// `false` mean `EXIT_SUCCESS` and `EXIT_FAILURE`.
///
/// # Errors
///
/// Returns [`RuntimeError::Exit`] to end the program, or a type error for a
/// status that is neither a boolean nor a number.
pub fn os_exit_truncating(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let status = match arguments.first() {
        Some(Value::Boolean(success)) => boolean_status(*success),
        status => truncated_status(runtime, status)?,
    };
    Err(exit_request(status, arguments))
}

/// `os.exit([code [, close]])` for Lua 5.3 and later, whose
/// `luaL_optinteger` rejects a float status without an integer value.
///
/// # Errors
///
/// Returns [`RuntimeError::Exit`] to end the program, or an argument error
/// for a status that is neither a boolean nor an integer.
pub fn os_exit(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let status = match arguments.first() {
        Some(Value::Boolean(success)) => boolean_status(*success),
        None | Some(Value::Nil) => 0,
        status => c_int(integer_i64_argument(status, 1)?),
    };
    Err(exit_request(status, arguments))
}

fn exit_request(status: i32, arguments: &[Value]) -> RuntimeError {
    RuntimeError::Exit(ExitRequest {
        status,
        close: arguments.get(1).is_some_and(Value::is_truthy),
    })
}

const fn boolean_status(success: bool) -> i32 {
    if success { 0 } else { 1 }
}

/// A status as `luaL_optint` reads it: absent is `EXIT_SUCCESS`, and a
/// number or numeric string converts to C `int` by truncation.
fn truncated_status(runtime: &Runtime, argument: Option<&Value>) -> RuntimeResult<i32> {
    match argument {
        None | Some(Value::Nil) => Ok(0),
        Some(Value::Integer(value)) => Ok(c_int(*value)),
        Some(value) => runtime
            .to_number(value)
            .map(truncated_c_int)
            .ok_or_else(|| RuntimeError::bad_argument(1, "number", argument)),
    }
}

/// Truncates toward zero. C leaves out-of-range results undefined; this
/// saturates to `i64` and keeps the low 32 bits.
#[allow(clippy::cast_possible_truncation)]
const fn truncated_c_int(number: f64) -> i32 {
    c_int(number as i64)
}

/// Converts to C `int` as a cast does: only the low 32 bits remain.
#[allow(clippy::cast_possible_truncation)]
const fn c_int(value: i64) -> i32 {
    value as i32
}

#[cfg(unix)]
pub(crate) fn run_shell(command: &str) -> i32 {
    let mut shell = Command::new("sh");
    shell.arg("-c").arg(command);
    crate::runtime::child_limits::restore_in_child(&mut shell);
    shell
        .status()
        .map_or(-1, std::os::unix::process::ExitStatusExt::into_raw)
}

#[cfg(not(unix))]
pub(crate) fn run_shell(command: &str) -> i32 {
    Command::new("cmd")
        .arg("/C")
        .arg(command)
        .status()
        .ok()
        .and_then(|status| status.code())
        .unwrap_or(-1)
}

#[cfg(unix)]
pub(crate) fn shell_available() -> i32 {
    i32::from(Command::new("sh").arg("-c").arg(":").status().is_ok())
}

/// Reserves and returns an empty host temporary-file path.
///
/// The path is allocated with `create_new`, so two Oximond calls cannot race
/// into the same name. Lua's `os.tmpname` API returns only a path, however,
/// so callers still own cleanup and must not treat a later path reopen as a
/// security boundary against unrelated host processes.
///
/// # Errors
///
/// Returns an error if no unique file can be reserved in the host temp
/// directory.
pub fn os_tmpname(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
    let directory = std::env::temp_dir();
    let process = std::process::id();
    for _ in 0..1024 {
        let serial = TEMP_NAME_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = directory.join(format!("oximond-{process}-{serial}"));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(_) => return Ok(vec![Value::from(path.to_string_lossy().into_owned())]),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(RuntimeError::message(format!(
                    "cannot create temp file: {error}"
                )));
            }
        }
    }
    Err(RuntimeError::message(
        "cannot reserve a unique temporary file",
    ))
}

/// Removes one filesystem entry and converts ordinary host failures into the
/// Lua `nil, message` result convention.
///
/// # Errors
///
/// Returns a type error if the path is not a Lua string.
pub fn os_remove(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let path = path_argument(arguments.first(), 1)?;
    match fs::remove_file(path) {
        Ok(()) => Ok(vec![Value::Boolean(true)]),
        Err(error) => Ok(vec![Value::Nil, Value::from(error.to_string())]),
    }
}

/// Renames one filesystem entry and converts ordinary host failures into the
/// Lua `nil, message` result convention.
///
/// # Errors
///
/// Returns a type error if either path is not a Lua string.
pub fn os_rename(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let from = path_argument(arguments.first(), 1)?;
    let to = path_argument(arguments.get(1), 2)?;
    match fs::rename(from, to) {
        Ok(()) => Ok(vec![Value::Boolean(true)]),
        Err(error) => Ok(vec![Value::Nil, Value::from(error.to_string())]),
    }
}

/// A file path at argument `position`; numbers convert to their spelling.
pub(crate) fn path_argument(value: Option<&Value>, position: usize) -> RuntimeResult<PathBuf> {
    let path = string_argument(value, position)?;
    Ok(path_from_lua_string(&LuaString::from(path.as_ref())))
}

#[cfg(unix)]
pub(crate) fn path_from_lua_string(value: &LuaString) -> PathBuf {
    use std::os::unix::ffi::OsStringExt as _;

    PathBuf::from(OsString::from_vec(value.as_bytes().to_vec()))
}

#[cfg(not(unix))]
pub(crate) fn path_from_lua_string(value: &LuaString) -> PathBuf {
    PathBuf::from(value.to_lossy().into_owned())
}

#[cfg(not(unix))]
pub(crate) fn shell_available() -> i32 {
    i32::from(Command::new("cmd").arg("/C").arg("exit 0").status().is_ok())
}

/// Returns Lua 5.1-compatible collector accounting as whole kilobytes and a
/// byte remainder.
///
/// # Errors
///
/// Returns an error only if the accounted byte total cannot become a Lua
/// number.
pub fn gcinfo(runtime: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
    let bytes = runtime.gc_count_bytes();
    Ok(vec![
        Value::Number(usize_to_number(
            bytes / 1024,
            "gcinfo byte count is too large",
        )?),
        Value::Number(usize_to_number(
            bytes % 1024,
            "gcinfo byte count is too large",
        )?),
    ])
}

/// Writes its arguments as tab-separated Lua values followed by a newline.
///
/// Strings are written as their original bytes. Tables and functions use a
/// stable identity label until the runtime has Lua-compatible address text.
///
/// # Errors
///
/// Returns an error if the runtime output buffer cannot grow.
pub fn print(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    print_arguments(runtime, arguments, false)
}

/// Lua 5.1's `print`, which writes each argument with `fputs` and so stops
/// at its first NUL byte.
///
/// # Errors
///
/// As [`print`].
pub fn print_lua51(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    print_arguments(runtime, arguments, true)
}

fn print_arguments(
    runtime: &mut Runtime,
    arguments: &[Value],
    stop_at_nul: bool,
) -> RuntimeResult<Vec<Value>> {
    let mut output = Vec::new();
    for (index, value) in arguments.iter().enumerate() {
        if index != 0 {
            append_print_bytes(&mut output, b"\t")?;
        }
        // An argument with `__tostring` needs the VM to call it. Only tables,
        // userdata and files can have one, and nothing is written until the
        // whole line is built.
        if matches!(value, Value::Table(_) | Value::Userdata(_) | Value::File(_))
            && has_tostring_metamethod(runtime, value)
        {
            return Err(RuntimeError::Redispatch);
        }
        let text = runtime.format_value(value)?;
        let mut bytes = text.as_bytes();
        if stop_at_nul && let Some(end) = bytes.iter().position(|byte| *byte == 0) {
            bytes = &bytes[..end];
        }
        append_print_bytes(&mut output, bytes)?;
    }
    append_print_bytes(&mut output, b"\n")?;
    runtime.write_output(&output)?;
    Ok(Vec::new())
}

/// Whether `value` is a table, userdata or file whose metatable has a
/// `__tostring` field.
pub(crate) fn has_tostring_metamethod(runtime: &Runtime, value: &Value) -> bool {
    static KEY: std::sync::OnceLock<LuaString> = std::sync::OnceLock::new();
    matches!(value, Value::Table(_) | Value::Userdata(_) | Value::File(_))
        && runtime
            .raw_metatable(value)
            .ok()
            .flatten()
            .is_some_and(|metatable| {
                runtime
                    .table_get_string_ref(
                        metatable,
                        KEY.get_or_init(|| LuaString::from("__tostring")),
                    )
                    .is_some()
            })
}

/// Writes string and number values to the runtime output stream without a
/// line terminator.
///
/// # Errors
///
/// Returns a Lua-style type error for unsupported values or an output-buffer
/// allocation failure.
pub fn io_write(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    // Like Lua, write the values before a bad one, then raise its error.
    let bad = arguments.iter().position(|value| {
        !matches!(
            value,
            Value::String(_) | Value::Number(_) | Value::Integer(_)
        )
    });
    if let Some(index) = bad {
        io_write(runtime, &arguments[..index])?;
        return Err(RuntimeError::bad_argument(
            index + 1,
            "string",
            arguments.get(index),
        ));
    }
    if let Some(file) = runtime.default_output() {
        return match runtime.file_write(file, arguments) {
            Ok(()) => Ok(vec![Value::File(file)]),
            Err(error @ (RuntimeError::FileIo { .. } | RuntimeError::HostOutputIo { .. })) => {
                Ok(io_failure(&error))
            }
            Err(error) => Err(error),
        };
    }
    let numbers = runtime.number_format().for_file_write();
    let mut output = Vec::new();
    for (position, value) in arguments.iter().enumerate() {
        match value {
            Value::String(value) => append_print_bytes(&mut output, value.as_bytes())?,
            Value::Number(value) => {
                append_print_bytes(&mut output, &numbers.format(*value))?;
            }
            value => {
                return Err(RuntimeError::bad_argument(
                    position + 1,
                    "string",
                    Some(value),
                ));
            }
        }
    }
    runtime.write_output(&output)?;
    Ok(vec![Value::Boolean(true)])
}

/// Selects a suffix from the variable arguments supplied after its index.
///
/// # Errors
///
/// Returns an error unless the index is `"#"` or an in-range, nonzero integer.
pub fn select(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let count = arguments.len().saturating_sub(1);
    match arguments.first() {
        Some(Value::String(index)) if index.as_bytes() == b"#" => Ok(vec![usize_integer_value(
            runtime,
            count,
            "select has too many arguments",
        )?]),
        Some(index) => {
            let index = select_index(index)?;
            let count = isize::try_from(count)
                .map_err(|_| RuntimeError::message("select has too many arguments"))?;
            let start = if index > 0 {
                index
            } else {
                count
                    .checked_add(index)
                    .and_then(|value| value.checked_add(1))
                    .ok_or_else(|| RuntimeError::argument_error(1, "index out of range"))?
            };
            let empty_suffix = count
                .checked_add(1)
                .ok_or_else(|| RuntimeError::argument_error(1, "index out of range"))?;
            // Stock Lua clips an index past the last argument: no values.
            if index > 0 && start > empty_suffix {
                return Ok(Vec::new());
            }
            if index == 0 || start < 1 {
                return Err(RuntimeError::argument_error(1, "index out of range"));
            }
            let start = usize::try_from(start - 1)
                .map_err(|_| RuntimeError::message("select index is out of range"))?;
            Ok(arguments[1 + start..].to_vec())
        }
        None => Err(RuntimeError::bad_argument(1, "number", None)),
    }
}

/// Stock Lua's value-stack limit (`LUAI_MAXSTACK`), which bounds how many
/// results `unpack` may produce before it reports "too many results".
const MAX_UNPACK_RESULTS: usize = 1_000_000;

/// Returns the inclusive numeric-key range from a table.
///
/// # Errors
///
/// Returns an error unless the first argument is a table and the optional
/// bounds are finite integers with a representable result size.
pub fn unpack(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let table = match arguments.first() {
        Some(Value::Table(table)) => *table,
        argument => return Err(RuntimeError::bad_argument(1, "table", argument)),
    };
    let start = optional_integer_argument(runtime, arguments.get(1), 2, 1)?;
    let default_end = number_to_isize(runtime.table_length(table)?, "unpack end index")?;
    let end = optional_integer_argument(runtime, arguments.get(2), 3, default_end)?;
    if start > end {
        return Ok(Vec::new());
    }
    let count = end
        .checked_sub(start)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| RuntimeError::message("unpack range is too large"))?;
    let count = usize::try_from(count)
        .ok()
        .filter(|count| *count <= MAX_UNPACK_RESULTS)
        .ok_or_else(|| RuntimeError::message("too many results to unpack"))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| RuntimeError::message("unpack range is too large"))?;
    let mut index = start;
    loop {
        values.push(runtime.table_get(table, &Value::Number(isize_to_number(index)?))?);
        if index == end {
            break;
        }
        index = index
            .checked_add(1)
            .ok_or_else(|| RuntimeError::message("unpack range is too large"))?;
    }
    Ok(values)
}

/// Returns a table containing all arguments and their exact count in `n`.
///
/// # Errors
///
/// Returns an error if the result table cannot be allocated or populated.
pub fn table_pack(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let table = runtime.create_table()?;
    for (offset, value) in arguments.iter().enumerate() {
        let index = offset
            .checked_add(1)
            .ok_or_else(|| RuntimeError::message("table.pack has too many arguments"))?;
        runtime.table_set(
            table,
            &Value::Number(usize_to_number(index, "table.pack has too many arguments")?),
            value.clone(),
        )?;
    }
    runtime.table_set(
        table,
        &Value::from("n"),
        usize_integer_value(
            runtime,
            arguments.len(),
            "table.pack has too many arguments",
        )?,
    )?;
    Ok(vec![Value::Table(table)])
}

/// Copies an inclusive integer-key range, preserving values when source and
/// destination are the same table and ranges overlap.
///
/// # Errors
///
/// Returns an error for invalid arguments, an oversized range, or a failed table access.
pub fn table_move(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let source = table_argument(arguments.first(), 1)?;
    let first = table_list_position(
        arguments
            .get(1)
            .ok_or_else(|| RuntimeError::bad_argument(2, "number", None))?,
        2,
    )?;
    let last = table_list_position(
        arguments
            .get(2)
            .ok_or_else(|| RuntimeError::bad_argument(3, "number", None))?,
        3,
    )?;
    let destination_start = table_list_position(
        arguments
            .get(3)
            .ok_or_else(|| RuntimeError::bad_argument(4, "number", None))?,
        4,
    )?;
    let destination = match arguments.get(4) {
        None | Some(Value::Nil) => source,
        Some(value) => table_argument(Some(value), 5)?,
    };
    if last < first {
        return Ok(vec![Value::Table(destination)]);
    }
    let count = last
        .checked_sub(first)
        .and_then(|count| count.checked_add(1))
        .ok_or_else(|| RuntimeError::message("table.move range is too large"))?;
    let count = usize::try_from(count)
        .map_err(|_| RuntimeError::message("table.move range is too large"))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| RuntimeError::message("table.move range is too large"))?;
    for offset in 0..count {
        let offset = isize::try_from(offset)
            .map_err(|_| RuntimeError::message("table.move range is too large"))?;
        let index = first
            .checked_add(offset)
            .ok_or_else(|| RuntimeError::message("table.move range is too large"))?;
        values.push(runtime.table_get(source, &Value::Number(isize_to_number(index)?))?);
    }
    for (offset, value) in values.into_iter().enumerate() {
        let offset = isize::try_from(offset)
            .map_err(|_| RuntimeError::message("table.move range is too large"))?;
        let index = destination_start
            .checked_add(offset)
            .ok_or_else(|| RuntimeError::message("table.move range is too large"))?;
        runtime.table_set(destination, &Value::Number(isize_to_number(index)?), value)?;
    }
    Ok(vec![Value::Table(destination)])
}

/// Allocates an empty table. Array and hash size hints are validated for Lua
/// compatibility but do not yet reserve distinct runtime storage regions.
///
/// # Errors
///
/// Returns an error for invalid size hints or if the table cannot be allocated.
pub fn table_create(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    if arguments.len() > 2 {
        return Err(RuntimeError::message(
            "wrong number of arguments to 'create'",
        ));
    }
    for (position, value) in arguments.iter().take(2).enumerate() {
        let size = Runtime::coerce_number(value, "call table.create with")?;
        if !size.is_finite() || size.fract() != 0.0 || size < 0.0 {
            return Err(RuntimeError::argument_error(
                position + 1,
                "non-negative integer expected",
            ));
        }
    }
    Ok(vec![Value::Table(runtime.create_table()?)])
}

/// Returns the current contiguous positive-integer length of a table.
///
/// # Errors
///
/// Returns an error unless the first argument is a table.
pub fn table_getn(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let table = table_argument(arguments.first(), 1)?;
    Ok(vec![Value::Number(runtime.table_length(table)?)])
}

/// Returns the greatest positive numeric key in a table, or zero when absent.
///
/// Lua 5.1's deprecated `table.maxn` examines every raw key. Negative numeric
/// keys and non-numeric keys do not lower the initial result of zero.
///
/// # Errors
///
/// Returns an error unless the first argument is a table or traversal finds a
/// stale key.
pub fn table_maxn(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let table = table_argument(arguments.first(), 1)?;
    let mut maximum = 0.0_f64;
    let mut key = Value::Nil;
    loop {
        let (next, _) = runtime.table_next(table, &key)?;
        if matches!(next, Value::Nil) {
            break;
        }
        if let Value::Number(number) = next
            && number > maximum
        {
            maximum = number;
        }
        key = next;
    }
    Ok(vec![Value::Number(maximum)])
}

/// Inserts a value into a table's contiguous list part using raw integer access.
///
/// # Errors
///
/// Returns an error for invalid arguments, an out-of-range insertion point, or
/// a table-storage failure.
pub fn table_insert(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    table_insert_with(runtime, arguments, true)
}

/// `table_insert` without the one-move shift: the element loop alone, kept as
/// the reference the tests compare the shift against.
#[cfg(test)]
pub(crate) fn table_insert_loop(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    table_insert_with(runtime, arguments, false)
}

fn table_insert_with(
    runtime: &mut Runtime,
    arguments: &[Value],
    shift: bool,
) -> RuntimeResult<Vec<Value>> {
    let table = table_argument(arguments.first(), 1)?;
    let length = table_list_length(runtime, table, "table.insert length")?;
    let (position, value) = match arguments {
        [_, value] => (
            length
                .checked_add(1)
                .ok_or_else(|| RuntimeError::message("table is too large"))?,
            value,
        ),
        [_, position, value] => (table_list_position(position, 2)?, value),
        _ => {
            return Err(RuntimeError::message(
                "wrong number of arguments to 'insert'",
            ));
        }
    };
    let maximum = length
        .checked_add(1)
        .ok_or_else(|| RuntimeError::message("table is too large"))?;
    if position < 1 || position > maximum {
        return Err(RuntimeError::argument_error(2, "position out of bounds"));
    }
    if shift && runtime.table_array_shift_insert(table, usize::try_from(position).ok(), value) {
        return Ok(Vec::new());
    }
    let mut index = maximum;
    while index > position {
        let previous = index - 1;
        let previous = runtime.table_get(table, &Value::Number(isize_to_number(previous)?))?;
        runtime.table_set(table, &Value::Number(isize_to_number(index)?), previous)?;
        index -= 1;
    }
    runtime.table_set(
        table,
        &Value::Number(isize_to_number(position)?),
        value.clone(),
    )?;
    Ok(Vec::new())
}

/// Removes and returns a value from a table's contiguous list part using raw integer access.
///
/// # Errors
///
/// Returns an error for an invalid table or position representation, or a
/// table-storage failure. Out-of-range positions return `nil`.
pub fn table_remove(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    table_remove_with(runtime, arguments, true)
}

/// `table_remove` without the one-move shift: the element loop alone, kept as
/// the reference the tests compare the shift against.
#[cfg(test)]
pub(crate) fn table_remove_loop(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    table_remove_with(runtime, arguments, false)
}

fn table_remove_with(
    runtime: &mut Runtime,
    arguments: &[Value],
    shift: bool,
) -> RuntimeResult<Vec<Value>> {
    let table = table_argument(arguments.first(), 1)?;
    let length = table_list_length(runtime, table, "table.remove length")?;
    let position = match arguments.get(1) {
        None | Some(Value::Nil) => length,
        Some(value) => table_list_position(value, 2)?,
    };
    if position < 1 || position > length {
        return Ok(vec![Value::Nil]);
    }
    if shift
        && let Some(removed) =
            runtime.table_array_shift_remove(table, usize::try_from(position).ok())
    {
        return Ok(vec![removed]);
    }
    let removed = runtime.table_get(table, &Value::Number(isize_to_number(position)?))?;
    let mut index = position;
    while index < length {
        let next = index + 1;
        let value = runtime.table_get(table, &Value::Number(isize_to_number(next)?))?;
        runtime.table_set(table, &Value::Number(isize_to_number(index)?), value)?;
        index = next;
    }
    runtime.table_set(table, &Value::Number(isize_to_number(length)?), Value::Nil)?;
    Ok(vec![removed])
}

/// A `table.concat` element at `index`, which must be a string or a number.
fn concat_element(
    numbers: crate::runtime::NumberFormat,
    value: &Value,
    index: isize,
) -> RuntimeResult<Cow<'_, [u8]>> {
    string_argument_in(numbers, Some(value), 1).map_err(|_| {
        RuntimeError::message(format!(
            "invalid value ({}) at index {index} in table for 'concat'",
            value.type_name()
        ))
    })
}

/// Appends a `table.concat` element, formatting a number in place.
fn push_concat_element(
    result: &mut Vec<u8>,
    numbers: crate::runtime::NumberFormat,
    value: &Value,
    index: isize,
) -> RuntimeResult<()> {
    match value {
        Value::String(text) => append_table_concat_bytes(result, text.as_bytes()),
        Value::Integer(number) => {
            append_table_concat_bytes(result, crate::runtime::format_integer(*number).as_bytes())
        }
        Value::Number(number) => {
            append_table_concat_bytes(result, numbers.format(*number).as_bytes())
        }
        value => append_table_concat_bytes(result, &concat_element(numbers, value, index)?),
    }
}

/// `table.concat` results up to this long are built in a stack buffer.
const SMALL_CONCAT_BYTES: usize = 64;

/// A separator up to this long is copied as one word (see
/// [`concat_values`]). It is no longer than an inline string, so the room
/// past the end of the result holds it too.
const SHORT_SEPARATOR_BYTES: usize = 8;
const _: () = assert!(SHORT_SEPARATOR_BYTES <= INLINE_STRING_BYTES);

/// `table.concat` of `values`, which are the table's elements from index
/// `start` on and none of them nil.
fn concat_values(
    numbers: crate::runtime::NumberFormat,
    separator: &[u8],
    values: &[Value],
    start: isize,
) -> RuntimeResult<Value> {
    let mut result = Vec::new();
    // A short result is gathered in one pass on the stack, numbers
    // formatted straight into the buffer, and copied once into its
    // string. The first element that does not fit, or is neither a
    // string nor a number, leaves the pass for the sized ones below.
    {
        let mut buffer = StackBuffer::<SMALL_CONCAT_BYTES>::new();
        let mut fits = true;
        for (offset, value) in values.iter().enumerate() {
            fits = (offset == 0 || buffer.push_bytes(separator))
                && match value {
                    Value::String(value) => buffer.push_bytes(value.as_bytes()),
                    Value::Integer(number) => buffer.push_integer(*number),
                    Value::Number(number) => buffer.push_number(numbers, *number),
                    _ => false,
                };
            if !fits {
                break;
            }
        }
        if fits {
            return Ok(Value::String(buffer.into_string()));
        }
    }
    if let Some(string_bytes) = values.iter().try_fold(0_usize, |length, value| {
        let Value::String(value) = value else {
            return None;
        };
        length.checked_add(value.len())
    }) && let Some(separator_bytes) = separator.len().checked_mul(values.len().saturating_sub(1))
        && let Some(capacity) = string_bytes.checked_add(separator_bytes)
    {
        // Short elements (formatted numbers, words) are the common list. An
        // inline string and a short separator are copied at the fixed size
        // of their buffers into a result with that much room past its end,
        // each over the zeros the one before it left: no copy call per
        // element for lengths only the strings know.
        if capacity / values.len().max(1) <= INLINE_STRING_BYTES
            && let Some(room) = capacity.checked_add(INLINE_STRING_BYTES)
        {
            result
                .try_reserve_exact(room)
                .map_err(|_| RuntimeError::message("table.concat result is too large"))?;
            result.resize(room, 0);
            let short_separator = (separator.len() <= SHORT_SEPARATOR_BYTES).then(|| {
                let mut word = [0_u8; SHORT_SEPARATOR_BYTES];
                word[..separator.len()].copy_from_slice(separator);
                word
            });
            let mut at = 0;
            for (offset, value) in values.iter().enumerate() {
                if offset != 0 {
                    match short_separator {
                        Some(word) => {
                            result[at..at + SHORT_SEPARATOR_BYTES].copy_from_slice(&word);
                        }
                        None => result[at..at + separator.len()].copy_from_slice(separator),
                    }
                    at += separator.len();
                }
                if let Value::String(value) = value {
                    if let Some((buffer, length)) = value.inline_buffer() {
                        result[at..at + INLINE_STRING_BYTES].copy_from_slice(buffer);
                        at += length;
                    } else {
                        let bytes = value.as_bytes();
                        result[at..at + bytes.len()].copy_from_slice(bytes);
                        at += bytes.len();
                    }
                }
            }
            debug_assert_eq!(at, capacity);
            result.truncate(capacity);
            return Ok(Value::from(result));
        }
        result
            .try_reserve_exact(capacity)
            .map_err(|_| RuntimeError::message("table.concat result is too large"))?;
        // Every element is a string and the exact capacity is reserved:
        // nothing below can fail, convert, or grow the buffer.
        for (offset, value) in values.iter().enumerate() {
            if offset != 0 {
                result.extend_from_slice(separator);
            }
            if let Value::String(value) = value {
                result.extend_from_slice(value.as_bytes());
            }
        }
        return Ok(Value::from(result));
    }
    for (offset, value) in values.iter().enumerate() {
        if offset != 0 {
            append_table_concat_bytes(&mut result, separator)?;
        }
        let index = start.saturating_add(isize::try_from(offset).unwrap_or(isize::MAX));
        push_concat_element(&mut result, numbers, value, index)?;
    }
    Ok(Value::from(result))
}

/// Concatenates an inclusive numeric-key table range using a byte separator.
///
/// # Errors
///
/// Returns an error unless the first argument is a table and each selected
/// value is a string or number. It also returns an error if the result cannot
/// reserve enough memory.
pub fn table_concat(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    let numbers = runtime.number_format();
    let table = table_argument(arguments.first(), 1)?;
    let separator = match arguments.get(1) {
        None | Some(Value::Nil) => Cow::Borrowed(&[][..]),
        Some(Value::String(text)) => Cow::Borrowed(text.as_bytes()),
        Some(value) => string_argument_in(numbers, Some(value), 2)?,
    };
    // The whole list of a table with no holes, the common call, reads the
    // array part in place without converting any index.
    if matches!(arguments.get(2), None | Some(Value::Nil))
        && matches!(arguments.get(3), None | Some(Value::Nil))
        && let Some(values) = runtime.table_border_array(table)?
        && values.iter().all(|value| !matches!(value, Value::Nil))
    {
        return concat_values(numbers, &separator, values, 1);
    }
    let start = optional_coerced_index(runtime, arguments.get(2), 3, 1)?;
    let default_end = number_to_isize(runtime.table_length(table)?, "table.concat end index")?;
    let end = optional_coerced_index(runtime, arguments.get(3), 4, default_end)?;
    if start > end {
        return Ok(Value::from(Vec::<u8>::new()));
    }

    let mut result = Vec::new();
    if let (Ok(first), Ok(last)) = (usize::try_from(start), usize::try_from(end))
        && let Some(values) = runtime.table_raw_array_range(table, first, last)?
        && values.iter().all(|value| !matches!(value, Value::Nil))
    {
        return concat_values(numbers, &separator, values, start);
    }
    let mut index = start;
    loop {
        if index != start {
            append_table_concat_bytes(&mut result, &separator)?;
        }
        let key = Value::Number(isize_to_number(index)?);
        if let Some(value) = runtime.table_get_raw_value(table, &key)? {
            let value = concat_element(numbers, value, index)?;
            append_table_concat_bytes(&mut result, &value)?;
        } else {
            let value = runtime.table_get(table, &key)?;
            let value = concat_element(numbers, &value, index)?;
            append_table_concat_bytes(&mut result, &value)?;
        }
        if index == end {
            break;
        }
        index = index
            .checked_add(1)
            .ok_or_else(|| RuntimeError::message("table.concat range is too large"))?;
    }
    Ok(Value::from(result))
}

/// Declines a call of a `table` function that Lua 5.3 and later run through
/// `__index`, `__newindex` and `__len`, so the VM runs it with metamethods.
/// A table without those metamethods keeps the raw path.
#[inline]
fn follows_metamethods(runtime: &Runtime, arguments: &[Value]) -> RuntimeResult<()> {
    if let Some(Value::Table(table)) = arguments.first()
        && runtime.table_needs_metamethods(*table)
    {
        return Err(RuntimeError::Redispatch);
    }
    Ok(())
}

pub(crate) fn table_concat_meta(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Value> {
    follows_metamethods(runtime, arguments)?;
    table_concat(runtime, arguments)
}

pub(crate) fn table_insert_meta(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    follows_metamethods(runtime, arguments)?;
    table_insert(runtime, arguments)
}

pub(crate) fn table_remove_meta(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    follows_metamethods(runtime, arguments)?;
    table_remove(runtime, arguments)
}

pub(crate) fn table_move_meta(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    follows_metamethods(runtime, arguments)?;
    // The destination table follows them too.
    if let Some(destination @ Value::Table(_)) = arguments.get(4) {
        follows_metamethods(runtime, std::slice::from_ref(destination))?;
    }
    table_move(runtime, arguments)
}

pub(crate) fn unpack_meta(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    follows_metamethods(runtime, arguments)?;
    unpack(runtime, arguments)
}

/// Returns the raw value stored under a key in a table.
///
/// # Errors
///
/// Returns an error unless the first argument is a table and the key is valid.
pub fn rawget(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    let table = table_argument(arguments.first(), 1)?;
    let key = arguments
        .get(1)
        .ok_or_else(|| RuntimeError::argument_error(2, "value expected"))?;
    runtime.table_get(table, key)
}

/// Stores a raw table value and returns the table.
///
/// Extra arguments are ignored, matching Lua 5.1's base-library call shape.
///
/// # Errors
///
/// Returns an error unless the first argument is a table, the key is valid,
/// and a value is supplied.
pub fn rawset(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    let table = table_argument(arguments.first(), 1)?;
    let key = arguments
        .get(1)
        .ok_or_else(|| RuntimeError::argument_error(2, "value expected"))?;
    let value = arguments
        .get(2)
        .ok_or_else(|| RuntimeError::argument_error(3, "value expected"))?
        .clone();
    runtime.table_set(table, key, value)?;
    Ok(Value::Table(table))
}

/// Returns the primitive length of a string or table without consulting a
/// `__len` metamethod.
///
/// # Errors
///
/// Returns an error if the first argument is not a string or table.
pub fn rawlen(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let value = match arguments.first() {
        Some(value @ (Value::Table(_) | Value::String(_))) => value,
        argument => {
            return Err(runtime.expected_argument_error(1, "table or string", argument));
        }
    };
    let length = runtime.raw_length(value)?;
    Ok(vec![runtime.integral_float_value(length)])
}

/// Emits a Lua warning, one piece per argument, through
/// [`Runtime::emit_warning`], which also handles `@on` and `@off`. Every
/// argument is checked before any piece is emitted.
///
/// # Errors
///
/// Returns an error if an argument cannot be converted to a warning string.
pub fn warn(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    if arguments.is_empty() {
        string_argument_in(runtime.number_format(), None, 1)?;
    }
    let messages = arguments
        .iter()
        .enumerate()
        .map(|(position, value)| {
            string_argument_in(runtime.number_format(), Some(value), position + 1)
        })
        .collect::<RuntimeResult<Vec<_>>>()?;
    for (position, text) in messages.iter().enumerate() {
        runtime.emit_warning(text, position + 1 < messages.len())?;
    }
    Ok(Vec::new())
}

/// Returns a value's metatable, respecting the `__metatable` protection field.
///
/// # Errors
///
/// Returns an error only when the runtime rejects a stale object handle.
pub fn getmetatable(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    let value = arguments
        .first()
        .ok_or_else(|| RuntimeError::argument_error(1, "value expected"))?;
    runtime.protected_metatable(value)
}

/// Assigns a table metatable while respecting `__metatable` protection.
///
/// # Errors
///
/// Returns an error unless its first argument is a table and its second
/// argument is a table or `nil`.
pub fn setmetatable(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    let table = table_argument(arguments.first(), 1)?;
    let metatable = metatable_argument(runtime, arguments.get(1), 2)?;
    runtime.set_table_metatable(table, metatable)?;
    Ok(Value::Table(table))
}

/// Creates LuaJIT-compatible opaque proxy userdata.
///
/// The zero-argument and `false` forms have no metatable. `true` creates a
/// fresh metatable. A userdata proxy copies its raw metatable so the two
/// values retain independent identity while sharing metamethods.
///
/// # Errors
///
/// Returns an error for an unsupported proxy argument or an allocation
/// failure in the runtime.
pub fn newproxy(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let metatable = match arguments.first() {
        None | Some(Value::Nil | Value::Boolean(false)) => None,
        Some(Value::Boolean(true)) => Some(runtime.create_table()?),
        Some(value @ Value::Userdata(_)) => runtime.raw_metatable(value)?,
        Some(_) => return Err(RuntimeError::argument_error(1, "boolean or proxy expected")),
    };
    let proxy = runtime.create_userdata()?;
    runtime.raw_set_metatable(&Value::Userdata(proxy), metatable)?;
    Ok(vec![Value::Userdata(proxy)])
}

/// Returns a value's metatable without honoring `__metatable` protection.
///
/// # Errors
///
/// Returns an error only when the runtime rejects a stale object handle.
pub fn debug_getmetatable(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let value = arguments
        .first()
        .ok_or_else(|| RuntimeError::argument_error(1, "value expected"))?;
    Ok(vec![
        runtime
            .raw_metatable(value)?
            .map_or(Value::Nil, Value::Table),
    ])
}

/// Assigns a raw metatable, including shared primitive metatables.
///
/// # Errors
///
/// Returns an error unless its second argument is a table or `nil`.
pub fn debug_setmetatable(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let value = arguments.first().cloned().unwrap_or(Value::Nil);
    let metatable = metatable_argument(runtime, arguments.get(1), 2)?;
    runtime.raw_set_metatable(&value, metatable)?;
    Ok(vec![value])
}

/// Compares two values using Lua's raw identity rules.
///
/// # Errors
///
/// Returns an error unless both values are supplied.
pub fn rawequal(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    let left = arguments
        .first()
        .ok_or_else(|| RuntimeError::argument_error(1, "value expected"))?;
    let right = arguments
        .get(1)
        .ok_or_else(|| RuntimeError::argument_error(2, "value expected"))?;
    Ok(Value::Boolean(Runtime::raw_equal(left, right)))
}

/// Returns the Lua type name of its first argument, a string shared by
/// every call, so that the interpreter stores it without a result vector or
/// an allocation.
///
/// # Errors
///
/// Raises "value expected" without an argument.
pub fn type_of(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    let value = arguments
        .first()
        .ok_or_else(|| RuntimeError::argument_error(1, "value expected"))?;
    Ok(Value::String(runtime.type_name_string(value)))
}

/// The rules of [`tonumber_scalar`].
pub(crate) const TONUMBER_RULES_MODERN: TonumberRules = TonumberRules {
    lua51: false,
    nonfinite: NonFiniteNumerals::Rejected,
};
/// The rules of [`tonumber_scalar_lua51`].
pub(crate) const TONUMBER_RULES_LUA51: TonumberRules = TonumberRules {
    lua51: true,
    nonfinite: NonFiniteNumerals::Strtod,
};
/// The rules of [`tonumber_scalar_luajit`].
pub(crate) const TONUMBER_RULES_LUAJIT: TonumberRules = TonumberRules {
    lua51: true,
    nonfinite: NonFiniteNumerals::LuaJit,
};

/// `tonumber` for Lua 5.2 and later, where a base requires a string and
/// `"inf"` and `"nan"` are not numbers.
pub(crate) fn tonumber_scalar(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    let TonumberRules { lua51, nonfinite } = TONUMBER_RULES_MODERN;
    tonumber_in(runtime, arguments, lua51, nonfinite)
}

/// `tonumber` for Lua 5.1, where base 10 converts as without a base, other
/// bases accept numbers through their string spelling, and strings convert
/// with `strtod`.
pub(crate) fn tonumber_scalar_lua51(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Value> {
    let TonumberRules { lua51, nonfinite } = TONUMBER_RULES_LUA51;
    tonumber_in(runtime, arguments, lua51, nonfinite)
}

/// `tonumber` for `LuaJIT`: Lua 5.1's base rules with `LuaJIT`'s number scanner.
pub(crate) fn tonumber_scalar_luajit(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Value> {
    let TonumberRules { lua51, nonfinite } = TONUMBER_RULES_LUAJIT;
    tonumber_in(runtime, arguments, lua51, nonfinite)
}

/// What `tonumber` of `arguments` gives in the profile whose settings are
/// `lua51` and `nonfinite`, for a trace that needs the type of the answer
/// before the call. `None` where the call raises an error.
fn tonumber_probe(
    runtime: &Runtime,
    arguments: &[Value],
    lua51: bool,
    nonfinite: NonFiniteNumerals,
) -> Option<Value> {
    tonumber_in(runtime, arguments, lua51, nonfinite).ok()
}

/// [`tonumber_probe`] for Lua 5.2 and later.
pub(crate) fn tonumber_probe_modern(runtime: &Runtime, arguments: &[Value]) -> Option<Value> {
    tonumber_probe(runtime, arguments, false, NonFiniteNumerals::Rejected)
}

/// [`tonumber_probe`] for Lua 5.1.
pub(crate) fn tonumber_probe_lua51(runtime: &Runtime, arguments: &[Value]) -> Option<Value> {
    tonumber_probe(runtime, arguments, true, NonFiniteNumerals::Strtod)
}

/// [`tonumber_probe`] for `LuaJIT`.
pub(crate) fn tonumber_probe_luajit(runtime: &Runtime, arguments: &[Value]) -> Option<Value> {
    tonumber_probe(runtime, arguments, true, NonFiniteNumerals::LuaJit)
}

#[inline]
fn tonumber_in(
    runtime: &Runtime,
    arguments: &[Value],
    lua51: bool,
    nonfinite: NonFiniteNumerals,
) -> RuntimeResult<Value> {
    // One argument that is a number or 64-bit integer already converts the
    // same way in every profile, and one string needs no look at a base.
    if let [argument] = arguments {
        match argument {
            Value::Integer(_) | Value::Number(_) => return Ok(argument.clone()),
            Value::Int64(value) => return Ok(Value::Number(*value as f64)),
            Value::UInt64(value) => return Ok(Value::Number(*value as f64)),
            Value::String(text) => return Ok(tonumber_plain(runtime, text.as_bytes(), nonfinite)),
            _ => {}
        }
    }
    tonumber_general(runtime, arguments, lua51, nonfinite)
}

/// `tonumber (text)` without a base: `nil` when `text` is no numeral.
#[inline]
fn tonumber_plain(runtime: &Runtime, text: &[u8], nonfinite: NonFiniteNumerals) -> Value {
    let tagged = runtime.tagged_integers();
    match scan_plain_numeral(text) {
        PlainNumeral::Whole {
            negative,
            magnitude,
        } => {
            if tagged {
                let value = magnitude.cast_signed();
                return Value::Integer(if negative { -value } else { value });
            }
            #[allow(clippy::cast_precision_loss, reason = "below 2^53, so exact")]
            let value = magnitude as f64;
            return Value::Number(if negative { -value } else { value });
        }
        PlainNumeral::Fraction(value) => return Value::Number(value),
        PlainNumeral::Invalid => return Value::Nil,
        PlainNumeral::Other => {}
    }
    // Lua 5.1 and 5.2 have one numeric representation. Parsing a long
    // hexadecimal numeral through the integer path would wrap it at 64 bits
    // before it can become a double.
    if tagged && let Some(value) = parse_lua_integer_exact(text) {
        return runtime.integer_value(value);
    }
    parse_lua_number(text)
        .or_else(|| nonfinite.parse(text))
        .map_or(Value::Nil, Value::Number)
}

/// `tonumber (source, base)` for a base from 2 to 36 that is not base 10
/// under Lua 5.1's rules: `nil` when `source` is no numeral in that base.
#[inline]
fn tonumber_based(
    runtime: &Runtime,
    source: &[u8],
    base: isize,
    lua51: bool,
    nonfinite: NonFiniteNumerals,
) -> Value {
    if runtime.tagged_integers() {
        return parse_wrapping_integer_base(source, base).map_or(Value::Nil, Value::Integer);
    }
    if lua51 {
        let luajit = matches!(nonfinite, NonFiniteNumerals::LuaJit);
        return parse_strtoul_base(source, base, luajit).map_or(Value::Nil, Value::Number);
    }
    parse_integer_base(source, base).map_or(Value::Nil, Value::Number)
}

/// Which `tonumber` a runtime has: Lua 5.1's base rules or the later ones,
/// and the spellings of infinity and NaN it takes. The installer records
/// them next to the native, so that a compiled trace converts a string the
/// way the native does without building its argument list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TonumberRules {
    pub(crate) lua51: bool,
    pub(crate) nonfinite: NonFiniteNumerals,
}

/// `tonumber (text [, base])` for a string and a base that is absent or an
/// integer, as the installed native answers it. `None` where the native
/// raises an error (a base out of range) or the runtime has no rules.
#[inline]
pub(crate) fn tonumber_of_text(runtime: &Runtime, text: &[u8], base: Option<i64>) -> Option<Value> {
    let TonumberRules { lua51, nonfinite } = runtime.tonumber_rules()?;
    match base {
        None => Some(tonumber_plain(runtime, text, nonfinite)),
        Some(10) if lua51 => Some(tonumber_plain(runtime, text, nonfinite)),
        Some(base @ 2..=36) => Some(tonumber_based(
            runtime,
            text,
            isize::try_from(base).ok()?,
            lua51,
            nonfinite,
        )),
        Some(_) => None,
    }
}

fn tonumber_general(
    runtime: &Runtime,
    arguments: &[Value],
    lua51: bool,
    nonfinite: NonFiniteNumerals,
) -> RuntimeResult<Value> {
    if arguments.is_empty() {
        return Err(RuntimeError::argument_error(1, "value expected"));
    }
    let base = match arguments.get(1) {
        None | Some(Value::Nil) => None,
        // A base in range, the common call, needs no conversion checks.
        Some(Value::Integer(base @ 2..=36)) => Some(*base as isize),
        #[allow(
            clippy::cast_possible_truncation,
            clippy::float_cmp,
            reason = "an exact round trip tells a small integer"
        )]
        Some(Value::Number(base)) if f64::from(*base as i8) == *base => Some(*base as isize),
        Some(base) => {
            let base = Runtime::coerce_number(base, "call tonumber with")
                .map_err(|error| error.at_argument(2))?;
            if runtime.tagged_integers() && base.fract() != 0.0 {
                return Err(RuntimeError::argument_error(
                    2,
                    "number has no integer representation",
                ));
            }
            Some(number_to_isize(base.trunc(), "tonumber base")?)
        }
    };
    let Some(base) = base.filter(|&base| !(lua51 && base == 10)) else {
        let value = match arguments.first() {
            Some(Value::Integer(value)) => return Ok(Value::Integer(*value)),
            Some(Value::Number(value)) => Some(*value),
            Some(value @ (Value::Int64(_) | Value::UInt64(_))) => {
                crate::runtime::cint64::CInt64::from_value(value)
                    .map(crate::runtime::cint64::CInt64::to_number)
            }
            Some(Value::String(value)) => {
                return Ok(tonumber_plain(runtime, value.as_bytes(), nonfinite));
            }
            _ => None,
        };
        return Ok(value.map_or(Value::Nil, Value::Number));
    };
    if !(2..=36).contains(&base) {
        return Err(RuntimeError::argument_error(2, "base out of range"));
    }
    let spelled;
    let source = match arguments.first() {
        Some(Value::String(value)) => value.as_bytes(),
        Some(Value::Number(value)) if lua51 => {
            spelled = runtime.number_format().format(*value);
            spelled.as_bytes()
        }
        argument => return Err(RuntimeError::bad_argument(1, "string", argument)),
    };
    Ok(tonumber_based(runtime, source, base, lua51, nonfinite))
}

pub(crate) extern "C" fn numeric_tonumber(value: f64) -> f64 {
    value
}
