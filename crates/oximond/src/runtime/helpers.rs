//! Free helpers for file I/O errors, buffering, and Lua number parsing/formatting.

use crate::runtime::error::{RuntimeError, RuntimeResult};
use crate::runtime::files::FileBuffering;
use crate::runtime::number_format::{NumberFormat, format_integer};
use crate::runtime::runtime::Runtime;
use crate::runtime::value::Value;
use std::fs::File;
use std::io::Write;

/// An I/O error as C's `strerror` spells it. Rust appends the OS error
/// number, `" (os error 2)"`, to the system message; Lua does not.
#[must_use]
pub fn io_error_text(error: &std::io::Error) -> String {
    #[cfg(windows)]
    if let Some((_, text)) = crt_errno(error) {
        return text.to_owned();
    }
    let text = error.to_string();
    if error.raw_os_error().is_some()
        && let Some(position) = text.rfind(" (os error ")
        && text.ends_with(')')
    {
        return text[..position].to_owned();
    }
    text
}

/// The `errno` C would have set for `error`, which Lua returns beside the
/// message.
pub(crate) fn io_error_code(error: &std::io::Error) -> Option<i32> {
    #[cfg(windows)]
    if let Some((code, _)) = crt_errno(error) {
        return Some(code);
    }
    error.raw_os_error()
}

/// Windows reports Win32 error codes and their system texts, but stock Lua
/// there reports the C runtime's `errno` and `strerror` text, so an OS error
/// of a kind C has a name for maps to those.
#[cfg(windows)]
fn crt_errno(error: &std::io::Error) -> Option<(i32, &'static str)> {
    use std::io::ErrorKind;
    error.raw_os_error()?;
    Some(match error.kind() {
        ErrorKind::NotFound => (2, "No such file or directory"),
        ErrorKind::PermissionDenied => (13, "Permission denied"),
        ErrorKind::AlreadyExists => (17, "File exists"),
        ErrorKind::NotADirectory => (20, "Not a directory"),
        ErrorKind::IsADirectory => (21, "Is a directory"),
        ErrorKind::InvalidInput => (22, "Invalid argument"),
        ErrorKind::DirectoryNotEmpty => (41, "Directory not empty"),
        _ => return None,
    })
}

pub(crate) fn file_io_error(error: &std::io::Error) -> RuntimeError {
    RuntimeError::FileIo {
        message: io_error_text(error),
        code: io_error_code(error).unwrap_or(1),
    }
}

pub(crate) fn host_output_io_error(error: &std::io::Error) -> RuntimeError {
    RuntimeError::HostOutputIo {
        message: io_error_text(error),
        code: io_error_code(error).unwrap_or(1),
    }
}

pub(crate) fn file_stream_error(message: impl Into<String>) -> RuntimeError {
    RuntimeError::FileIo {
        message: message.into(),
        code: 1,
    }
}

pub(crate) fn flush_pending(file: &mut File, pending: &mut Vec<u8>) -> RuntimeResult<()> {
    if !pending.is_empty() {
        file.write_all(pending)
            .map_err(|error| file_io_error(&error))?;
        pending.clear();
    }
    Ok(())
}

pub(crate) fn write_named_file(
    file: &mut File,
    buffering: FileBuffering,
    pending: &mut Vec<u8>,
    bytes: &[u8],
) -> RuntimeResult<()> {
    match buffering {
        FileBuffering::No => file.write_all(bytes).map_err(|error| file_io_error(&error)),
        FileBuffering::Full(limit) => {
            pending.extend_from_slice(bytes);
            if pending.len() >= limit {
                flush_pending(file, pending)?;
            }
            Ok(())
        }
        FileBuffering::Line(limit) => {
            pending.extend_from_slice(bytes);
            if pending.contains(&b'\n') || pending.len() >= limit {
                flush_pending(file, pending)?;
            }
            Ok(())
        }
    }
}

pub(crate) fn file_line_iterator_next(
    runtime: &mut Runtime,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    let Some(Value::Table(iterator)) = arguments.first() else {
        return Err(RuntimeError::message("invalid file line iterator"));
    };
    if matches!(
        runtime.table_get(*iterator, &Value::from("\0done"))?,
        Value::Boolean(true)
    ) {
        return Err(RuntimeError::message("file is already closed"));
    }
    let Value::File(file) = runtime.table_get(*iterator, &Value::from("\0file"))? else {
        return Err(RuntimeError::message("invalid file line iterator"));
    };
    let formats = match runtime.table_get(*iterator, &Value::from("\0formats"))? {
        Value::Integer(count) => usize::try_from(count).unwrap_or(0),
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // At most 250.
        Value::Number(count) => count as usize,
        _ => 0,
    };
    if formats == 0 {
        if let Some(line) = runtime.file_read_line(file)? {
            return Ok(vec![Value::String(line)]);
        }
    } else {
        let mut read_formats = Vec::with_capacity(formats);
        for index in 1..=formats {
            let key = runtime.integer_value(i64::try_from(index).unwrap_or(i64::MAX));
            read_formats.push(runtime.table_get(*iterator, &key)?);
        }
        // Like liolib's `io_readline`, a nil first value ends the loop;
        // otherwise every value read so far is returned.
        let values = runtime.file_read_values(file, &read_formats)?;
        if !matches!(values.first(), None | Some(Value::Nil)) {
            return Ok(values);
        }
    }
    let close = matches!(
        runtime.table_get(*iterator, &Value::from("\0close"))?,
        Value::Boolean(true)
    );
    if close {
        runtime.file_close(file)?;
        runtime.table_set(*iterator, &Value::from("\0done"), Value::Boolean(true))?;
    }
    Ok(Vec::new())
}

pub(crate) fn file_write_bytes(value: &Value, format: NumberFormat) -> RuntimeResult<Vec<u8>> {
    match value {
        Value::String(value) => Ok(value.as_bytes().to_vec()),
        Value::Integer(value) => Ok(format_integer(*value).to_vec()),
        Value::Number(value) => Ok(format.format(*value).to_vec()),
        value => Err(RuntimeError::TypeError {
            operation: "write",
            expected: "string",
            found: value.type_name(),
        }),
    }
}

/// What [`scan_plain_numeral`] makes of a string.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PlainNumeral {
    /// Decimal digits alone, at most 15 of them, after an optional sign.
    Whole { negative: bool, magnitude: u64 },
    /// Decimal digits around one point, at most 15 of them, with its sign.
    Fraction(f64),
    /// No decimal numeral, no hexadecimal numeral and no spelling of
    /// infinity or NaN: not a number under any profile.
    Invalid,
    /// Anything else (an exponent, a hexadecimal prefix, a long numeral,
    /// `inf`, `nan`): the full parsers decide.
    Other,
}

/// Powers of ten a double holds exactly, for [`scan_plain_numeral`].
const EXACT_POWERS_OF_TEN: [f64; 16] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
];

/// Reads the numerals most strings that become numbers hold (`"17"`,
/// `"-29"`, `"3.25"`) in one pass, and rejects in the same pass most strings
/// that are no number at all.
///
/// It agrees with the parsers it stands in front of ([`parse_lua_number`],
/// `stdlib::strings::parse_lua_number`, [`parse_lua_integer_exact`] and
/// [`NonFiniteNumerals::parse`]):
///
/// - All of them trim ASCII whitespace and take one sign.
/// - A decimal numeral must then start with a digit or a point, a
///   hexadecimal one with `0x`, and infinity or NaN with `i` or `n` in
///   either case. Any other first byte is [`PlainNumeral::Invalid`].
/// - After decimal digits and at most one point, only an exponent (`e`) can
///   continue a numeral, or the `x` of the prefix `0x`. Any other byte is
///   `Invalid`; those two are [`PlainNumeral::Other`].
/// - Up to 15 digits are below 2^53 and so exact in a double, and a power of
///   ten up to 10^15 is exact too. One division of two exact doubles rounds
///   once, so the quotient is the double nearest the numeral, which is what
///   the float parser gives.
#[inline]
pub(crate) fn scan_plain_numeral(bytes: &[u8]) -> PlainNumeral {
    let bytes = trim_ascii(bytes);
    let (negative, body) = match bytes.first() {
        Some(b'-') => (true, &bytes[1..]),
        Some(b'+') => (false, &bytes[1..]),
        _ => (false, bytes),
    };
    match body.first() {
        Some(b'0'..=b'9' | b'.') => {}
        Some(b'i' | b'I' | b'n' | b'N') => return PlainNumeral::Other,
        _ => return PlainNumeral::Invalid,
    }
    if body.len() > 16 {
        return PlainNumeral::Other;
    }
    let mut magnitude = 0_u64;
    let mut digits = 0_usize;
    let mut point = None;
    for (index, &byte) in body.iter().enumerate() {
        match byte {
            b'0'..=b'9' => {
                magnitude = magnitude * 10 + u64::from(byte - b'0');
                digits += 1;
            }
            b'.' if point.is_none() => point = Some(index),
            b'e' | b'E' => return PlainNumeral::Other,
            // The prefix of a hexadecimal numeral, and nowhere else.
            b'x' | b'X' if index == 1 && body[0] == b'0' => return PlainNumeral::Other,
            _ => return PlainNumeral::Invalid,
        }
    }
    if digits == 0 || digits > 15 {
        return PlainNumeral::Other;
    }
    let Some(point) = point else {
        return PlainNumeral::Whole {
            negative,
            magnitude,
        };
    };
    #[allow(clippy::cast_precision_loss, reason = "below 2^53, so exact")]
    let value = magnitude as f64 / EXACT_POWERS_OF_TEN[body.len() - 1 - point];
    PlainNumeral::Fraction(if negative { -value } else { value })
}

pub(crate) fn parse_lua_number(bytes: &[u8]) -> Option<f64> {
    // The capture of `%d+` that a program adds to a number needs no float
    // parser.
    match scan_plain_numeral(bytes) {
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
    let bytes = trim_ascii(bytes);
    let numeric_body = bytes
        .strip_prefix(b"+")
        .or_else(|| bytes.strip_prefix(b"-"))
        .unwrap_or(bytes);
    if matches!(numeric_body.first(), Some(b'0'..=b'9' | b'.'))
        && let Ok(text) = std::str::from_utf8(bytes)
        && let Ok(value) = text.parse::<f64>()
    {
        return Some(value);
    }

    let (negative, bytes) = match bytes.first() {
        Some(b'-') => (true, &bytes[1..]),
        Some(b'+') => (false, &bytes[1..]),
        _ => (false, bytes),
    };
    let bytes = bytes
        .strip_prefix(b"0x")
        .or_else(|| bytes.strip_prefix(b"0X"))?;
    let (significand, exponent) = match bytes.iter().position(|byte| matches!(byte, b'p' | b'P')) {
        Some(index) => (
            &bytes[..index],
            std::str::from_utf8(&bytes[index + 1..])
                .ok()?
                .parse::<i32>()
                .ok()?,
        ),
        None => (bytes, 0),
    };
    let mut value = 0.0;
    let mut digits = 0;
    let mut fraction = false;
    let mut scale = 1.0 / 16.0;
    for &byte in significand {
        if byte == b'.' && !fraction {
            fraction = true;
            continue;
        }
        let digit = match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            b'A'..=b'F' => byte - b'A' + 10,
            _ => return None,
        };
        if fraction {
            value += f64::from(digit) * scale;
            scale /= 16.0;
        } else {
            value = value * 16.0 + f64::from(digit);
        }
        digits += 1;
    }
    (digits != 0).then(|| {
        let value = value * 2_f64.powi(exponent);
        if negative { -value } else { value }
    })
}

/// Which of C's spellings of infinity and NaN a string-to-number conversion
/// accepts. Lua 5.1 converts with `strtod`, and LuaJIT's scanner accepts the
/// same words; Lua 5.2 and later reject them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonFiniteNumerals {
    /// Lua 5.2 and later: `"inf"` and `"nan"` are not numbers.
    Rejected,
    /// Lua 5.1: `strtod`'s `inf`, `infinity` and `nan`, in any case and
    /// with a sign, including a `nan(chars)` payload. The NaN keeps its sign.
    Strtod,
    /// LuaJIT: `inf`, `infinity` and `nan` in any case and with a sign.
    /// Every NaN is the canonical one.
    LuaJit,
}

impl NonFiniteNumerals {
    /// Parses `bytes`, less surrounding whitespace, as an infinity or NaN.
    #[must_use]
    pub fn parse(self, bytes: &[u8]) -> Option<f64> {
        if self == Self::Rejected {
            return None;
        }
        let bytes = trim_ascii(bytes);
        let (negative, word) = match bytes.first() {
            Some(b'-') => (true, &bytes[1..]),
            Some(b'+') => (false, &bytes[1..]),
            _ => (false, bytes),
        };
        if word.eq_ignore_ascii_case(b"inf") || word.eq_ignore_ascii_case(b"infinity") {
            return Some(if negative {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            });
        }
        let payload = word
            .get(..3)
            .filter(|nan| nan.eq_ignore_ascii_case(b"nan"))
            .map(|_| &word[3..])?;
        let payload_accepted = payload.is_empty()
            || (self == Self::Strtod
                && payload.len() >= 2
                && payload.starts_with(b"(")
                && payload.ends_with(b")")
                && payload[1..payload.len() - 1]
                    .iter()
                    .all(|byte| byte.is_ascii_alphanumeric() || *byte == b'_'));
        payload_accepted.then(|| match self {
            Self::LuaJit => f64::from_bits(0xfff8_0000_0000_0000),
            _ if negative => -f64::NAN,
            _ => f64::NAN,
        })
    }
}

/// Parses a numeric string as a Lua integer numeral, following the lexer:
/// decimal numerals that overflow `i64` are rejected (they convert as
/// floats), while hexadecimal numerals wrap around modulo 2^64.
pub fn parse_lua_integer_exact(bytes: &[u8]) -> Option<i64> {
    match scan_plain_numeral(bytes) {
        PlainNumeral::Whole {
            negative,
            magnitude,
        } => {
            let value = magnitude.cast_signed();
            return Some(if negative { -value } else { value });
        }
        // A numeral with a point is a float, and so not an integer numeral.
        PlainNumeral::Fraction(_) | PlainNumeral::Invalid => return None,
        PlainNumeral::Other => {}
    }
    let bytes = trim_ascii(bytes);
    let (negative, bytes) = match bytes.first() {
        Some(b'-') => (true, &bytes[1..]),
        Some(b'+') => (false, &bytes[1..]),
        _ => (false, bytes),
    };
    if let Some(hexadecimal) = bytes
        .strip_prefix(b"0x")
        .or_else(|| bytes.strip_prefix(b"0X"))
    {
        if hexadecimal.is_empty() {
            return None;
        }
        let mut value = 0_u64;
        for &byte in hexadecimal {
            let digit = char::from(byte).to_digit(16)?;
            value = value.wrapping_mul(16).wrapping_add(u64::from(digit));
        }
        let value = value.cast_signed();
        return Some(if negative {
            value.wrapping_neg()
        } else {
            value
        });
    }
    if !bytes.first().is_some_and(u8::is_ascii_digit) {
        return None;
    }
    let magnitude = std::str::from_utf8(bytes).ok()?.parse::<u64>().ok()?;
    if negative {
        if magnitude == 1_u64 << 63 {
            Some(i64::MIN)
        } else {
            i64::try_from(magnitude).ok().map(i64::wrapping_neg)
        }
    } else {
        i64::try_from(magnitude).ok()
    }
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

pub(crate) fn file_offset(value: &Value) -> RuntimeResult<i64> {
    let value = Runtime::coerce_number(value, "seek")?;
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(RuntimeError::message("seek offset must be an integer"));
    }
    value
        .to_string()
        .parse::<i64>()
        .map_err(|_| RuntimeError::message("seek offset is out of range"))
}

pub(crate) fn usize_to_lua_number(value: usize, message: &'static str) -> RuntimeResult<f64> {
    value
        .to_string()
        .parse::<f64>()
        .map_err(|_| RuntimeError::message(message))
}
