//! Lua number-to-string conversion for `tostring`, `print`, concatenation,
//! and the string library's implicit coercions.
//!
//! Stock Lua 5.1 through 5.4 and LuaJIT format floats with C's `"%.14g"`
//! (`LUAI_NUMFFORMAT`). Lua 5.3 and later append `".0"` when that text looks
//! like an integer, and Lua 5.5 widens the format to `"%.15g"`, falling back
//! to `"%.17g"` when fifteen digits do not read back as the same float.
//! Formatting writes into a fixed stack buffer so hot `tostring` and `..`
//! paths allocate only the resulting Lua string.

use crate::string::{INLINE_STRING_BYTES, LuaString};
use std::fmt::Write as _;
use std::ops::Deref;

/// Decimal integers below this limit share one string each per thread.
const SHARED_INTEGERS: u64 = 256;

thread_local! {
    static INTEGER_STRINGS: Vec<LuaString> = (0..SHARED_INTEGERS)
        .map(|value| LuaString::from(value.to_string()))
        .collect();
}

/// The shared string for a small non-negative integer's decimal digits.
///
/// Indexes, counters, and byte values are the numbers programs turn into
/// strings most, so `tostring(i)` of one is a reference count away instead
/// of an allocation. Strings are immutable and compare by bytes, so sharing
/// one is invisible to Lua.
#[must_use]
#[inline]
pub fn small_integer_string(value: u64) -> Option<LuaString> {
    if value >= SHARED_INTEGERS {
        return None;
    }
    #[allow(clippy::cast_possible_truncation, reason = "below 256")]
    Some(INTEGER_STRINGS.with(|strings| strings[value as usize].clone()))
}

/// How a compatibility profile spells float values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NumberFormat {
    /// Lua 5.1 and 5.2: `"%.14g"`, with NaN signs as glibc's printf spells them.
    #[default]
    Lua51,
    /// LuaJIT 2.1: `"%.14g"`, with NaN always spelled `nan`.
    LuaJit,
    /// Lua 5.3 and 5.4: `"%.14g"`, plus `".0"` on integer-looking results.
    Lua53,
    /// Lua 5.5: round-tripping `"%.15g"` or `"%.17g"`, plus `".0"`.
    Lua55,
}

impl NumberFormat {
    /// Formats a float as [`Self::format`] does, into a new string.
    ///
    /// Integral values below 10^14 (printed digit for digit by every
    /// profile, with `.0` from Lua 5.3 on) are assembled in a register
    /// rather than a text buffer.
    #[must_use]
    #[inline]
    pub fn format_string(self, value: f64) -> LuaString {
        const LIMIT: f64 = 1e14;
        let magnitude = value.abs();
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_precision_loss,
            clippy::cast_sign_loss,
            clippy::float_cmp,
            reason = "below 1e14 < 2^53, so the round trip is exact"
        )]
        if magnitude < LIMIT && (magnitude as u64) as f64 == magnitude {
            let mut digits = magnitude as u64;
            let negative = value.is_sign_negative();
            if !negative
                && !self.appends_float_suffix()
                && let Some(shared) = small_integer_string(digits)
            {
                return shared;
            }
            let mut count = 1;
            let mut scale = 10;
            while count < 14 && digits >= scale {
                count += 1;
                scale *= 10;
            }
            let digits_end = count + usize::from(negative);
            // Lua 5.3 and later mark integral floats with `.0`.
            let suffix = self.appends_float_suffix();
            let length = digits_end + if suffix { 2 } else { 0 };
            if length > 16 {
                return LuaString::new(self.format(value).as_bytes());
            }
            let mut word = if suffix {
                (u128::from(b'.') | (u128::from(b'0') << 8)) << (8 * digits_end)
            } else {
                0
            };
            let mut position = digits_end;
            loop {
                position -= 1;
                word |= u128::from(b'0' + (digits % 10) as u8) << (8 * position);
                digits /= 10;
                if digits == 0 {
                    break;
                }
            }
            if negative {
                word |= u128::from(b'-');
            }
            return LuaString::from_packed(word, length);
        }
        LuaString::new(self.format(value).as_bytes())
    }

    /// Formats a float exactly as the profile's `tostring` does.
    #[must_use]
    pub fn format(self, value: f64) -> NumberText {
        let mut text = NumberText::new();
        if let Some(spelling) = self.non_finite(value) {
            text.extend(spelling);
            return text;
        }
        let lua55 = self == Self::Lua55;
        let limit = if lua55 { 1e15 } else { 1e14 };
        let magnitude = value.abs();
        if value.is_sign_negative() {
            text.push(b'-');
        }
        // Integral values with at most `precision` digits print every digit.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_precision_loss,
            clippy::cast_sign_loss,
            clippy::float_cmp
        )] // Below 1e15 < 2^53, so the round trip is exact.
        if magnitude < limit && (magnitude as u64) as f64 == magnitude {
            push_decimal(&mut text, magnitude as u64);
        } else if let Some((digits, exponent)) = fourteen_digits(magnitude).filter(|_| !lua55) {
            let mut buffer = NumberText::new();
            push_decimal(&mut buffer, digits);
            layout_general(&mut text, buffer.as_bytes(), exponent, 14, false);
        } else {
            let mut digits = DigitBuffer::default();
            let (count, exponent, precision) = if lua55 {
                // Lua 5.5 keeps fifteen digits when they round-trip. For
                // normal floats that is exactly when the shortest round-trip
                // form has at most 15 digits, since then `%.15g` prints those
                // digits. Subnormals lack the precision for that argument.
                let (count, exponent) = if magnitude >= f64::MIN_POSITIVE {
                    digits.shortest(magnitude)
                } else {
                    digits.exact(magnitude, 15)
                };
                if count <= 15 && digits.reads_back(count, exponent, magnitude) {
                    (count, exponent, 15)
                } else {
                    let (count, exponent) = digits.exact(magnitude, 17);
                    (count, exponent, 17)
                }
            } else {
                let (count, exponent) = digits.exact(magnitude, 14);
                (count, exponent, 14)
            };
            layout_general(
                &mut text,
                &digits.bytes[..count],
                exponent,
                precision,
                false,
            );
        }
        if self.appends_float_suffix()
            && text
                .as_bytes()
                .iter()
                .all(|byte| byte.is_ascii_digit() || *byte == b'-')
        {
            text.extend(b".0");
        }
        text
    }

    /// Returns the profile's spelling of NaN or an infinity, as printf's
    /// `%g` and friends write them before case conversion.
    #[must_use]
    pub fn non_finite(self, value: f64) -> Option<&'static [u8]> {
        if value.is_nan() {
            Some(if value.is_sign_negative() && self != Self::LuaJit {
                b"-nan"
            } else {
                b"nan"
            })
        } else if value.is_infinite() {
            Some(if value.is_sign_negative() {
                b"-inf"
            } else {
                b"inf"
            })
        } else {
            None
        }
    }

    /// Returns the format `io.write` and `file:write` use. Lua 5.3 and 5.4
    /// write floats with bare `"%.14g"`, so `3.0` is written as `3`; Lua 5.5
    /// writes the `tostring` text.
    #[must_use]
    pub const fn for_file_write(self) -> Self {
        match self {
            Self::Lua53 => Self::Lua51,
            format => format,
        }
    }

    /// Reports whether integral floats print with a `".0"` suffix, so that
    /// `3.0` and the integer `3` spell differently.
    #[must_use]
    pub const fn appends_float_suffix(self) -> bool {
        matches!(self, Self::Lua53 | Self::Lua55)
    }
}

/// Formats an integer-subtype value.
#[must_use]
pub fn format_integer(value: i64) -> NumberText {
    let mut text = NumberText::new();
    if value < 0 {
        text.push(b'-');
    }
    push_decimal(&mut text, value.unsigned_abs());
    text
}

/// Appends C's `%.<precision>g` (or `%#.<precision>g`) spelling of a finite,
/// non-negative `magnitude`. Signs, padding, and case are the caller's.
pub fn push_general(output: &mut Vec<u8>, magnitude: f64, precision: usize, alternate: bool) {
    let precision = precision.max(1);
    if precision <= DigitBuffer::MAX_DIGITS {
        let mut digits = DigitBuffer::default();
        let (count, exponent) = digits.exact(magnitude, precision);
        layout_general(
            output,
            &digits.bytes[..count],
            exponent,
            precision,
            alternate,
        );
    } else {
        let formatted = format!("{magnitude:.*e}", precision - 1);
        let (digits, exponent) = split_exponential(formatted.as_bytes());
        let digits: Vec<u8> = digits.iter().copied().filter(u8::is_ascii_digit).collect();
        layout_general(output, &digits, exponent, precision, alternate);
    }
}

/// Rounds a float in `[1e-5, 1e14)` to 14 significant digits with exact
/// integer arithmetic, returning the digits as an integer in
/// `[10^13, 10^14)` and the decimal exponent of the first digit.
///
/// The float is `m * 2^-s` with `m < 2^53` and `6 <= s <= 70`, so
/// `m * 10^k` for the needed `k <= 19` fits a `u128` and dividing by `2^s`
/// yields the scaled value's integer part and exact remainder. Rounding
/// then ties to even as glibc's printf does. This covers practically every
/// non-integral number real programs print, at a fraction of the cost of
/// general decimal expansion.
fn fourteen_digits(magnitude: f64) -> Option<(u64, i32)> {
    const LOW: u64 = 10_000_000_000_000;
    const HIGH: u64 = 100_000_000_000_000;
    if !(1e-5..1e14).contains(&magnitude) {
        return None;
    }
    let bits = magnitude.to_bits();
    let mantissa = u128::from((bits & ((1 << 52) - 1)) | (1 << 52));
    #[allow(clippy::cast_possible_truncation)] // An 11-bit field.
    let biased = (bits >> 52) as i32;
    let shift = 1075 - biased;
    // floor(e * log10(2)) is the decimal exponent or one below it.
    let mut exponent = ((biased - 1023) * 78_913) >> 18;
    loop {
        let places = u32::try_from(13 - exponent).ok()?;
        let scaled = mantissa * 10_u128.checked_pow(places)?;
        let half = 1_u128 << (shift - 1);
        let remainder = scaled & ((half << 1) - 1);
        let mut digits = scaled >> shift;
        if remainder > half || (remainder == half && digits & 1 == 1) {
            digits += 1;
        }
        let digits = u64::try_from(digits).ok()?;
        if digits < LOW {
            exponent -= 1;
        } else if digits < HIGH {
            return Some((digits, exponent));
        } else if digits == HIGH {
            return Some((LOW, exponent + 1));
        } else {
            exponent += 1;
        }
    }
}

/// Writes significant digits and their decimal exponent as C's `%g` does:
/// fixed notation when `-4 <= exponent < precision`, otherwise exponential,
/// with trailing fractional zeros removed unless `alternate` (`#`) is set.
fn layout_general(
    output: &mut impl ByteSink,
    digits: &[u8],
    exponent: i32,
    precision: usize,
    alternate: bool,
) {
    // Keep at least the leading digit; zero is the digit string "0".
    let significant = if alternate {
        digits.len()
    } else {
        digits.len() - digits[1..].iter().rev().take_while(|d| **d == b'0').count()
    };
    let digits = &digits[..significant];
    let padding = if alternate {
        precision - digits.len()
    } else {
        0
    };
    let fixed = exponent >= -4 && usize::try_from(exponent).map_or(true, |e| e < precision);
    if fixed && exponent >= 0 {
        let whole = usize::try_from(exponent).unwrap_or(0) + 1;
        if digits.len() <= whole {
            output.extend(digits);
            for _ in digits.len()..whole {
                output.push(b'0');
            }
            if alternate {
                output.push(b'.');
                for _ in whole..precision {
                    output.push(b'0');
                }
            }
        } else {
            output.extend(&digits[..whole]);
            output.push(b'.');
            output.extend(&digits[whole..]);
            for _ in 0..padding {
                output.push(b'0');
            }
        }
    } else if fixed {
        output.extend(b"0.");
        for _ in 1..exponent.unsigned_abs() {
            output.push(b'0');
        }
        output.extend(digits);
        for _ in 0..padding {
            output.push(b'0');
        }
    } else {
        output.push(digits[0]);
        if digits.len() > 1 || alternate {
            output.push(b'.');
            output.extend(&digits[1..]);
            for _ in 0..padding {
                output.push(b'0');
            }
        }
        output.push(b'e');
        output.push(if exponent < 0 { b'-' } else { b'+' });
        let exponent = exponent.unsigned_abs();
        if exponent < 10 {
            output.push(b'0');
        }
        let mut text = NumberText::new();
        push_decimal(&mut text, u64::from(exponent));
        output.extend(text.as_bytes());
    }
}

/// Splits Rust's `d.ddde[-]x` exponential output into mantissa bytes and
/// the decimal exponent.
fn split_exponential(formatted: &[u8]) -> (&[u8], i32) {
    let split = formatted
        .iter()
        .position(|byte| *byte == b'e')
        .expect("Rust exponential output has an exponent");
    let (mantissa, exponent) = (&formatted[..split], &formatted[split + 1..]);
    let (negative, exponent) = match exponent.split_first() {
        Some((b'-', rest)) => (true, rest),
        _ => (false, exponent),
    };
    let magnitude = exponent
        .iter()
        .fold(0_i32, |value, digit| value * 10 + i32::from(digit - b'0'));
    (mantissa, if negative { -magnitude } else { magnitude })
}

fn push_decimal(output: &mut impl ByteSink, mut value: u64) {
    let mut digits = [0_u8; 20];
    let mut start = digits.len();
    loop {
        start -= 1;
        digits[start] = b'0' + (value % 10) as u8;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    output.extend(&digits[start..]);
}

/// Significant decimal digits produced by Rust's float formatting, which
/// rounds exactly and ties to even like glibc's printf.
#[derive(Default)]
struct DigitBuffer {
    bytes: [u8; 32],
    len: usize,
}

impl DigitBuffer {
    const MAX_DIGITS: usize = 24;

    /// Stores `precision` correctly rounded significant digits of a finite,
    /// non-negative value and returns the digit count and decimal exponent.
    fn exact(&mut self, magnitude: f64, precision: usize) -> (usize, i32) {
        self.len = 0;
        write!(self, "{magnitude:.*e}", precision - 1).expect("digit buffer fits");
        self.compact()
    }

    /// Stores the shortest round-trip digits of a finite, non-negative value.
    fn shortest(&mut self, magnitude: f64) -> (usize, i32) {
        self.len = 0;
        write!(self, "{magnitude:e}").expect("digit buffer fits");
        self.compact()
    }

    /// Reports whether the stored digits parse back to `magnitude`. Normal
    /// floats' shortest digits always do, so only subnormals pay for a parse.
    fn reads_back(&self, count: usize, exponent: i32, magnitude: f64) -> bool {
        if magnitude >= f64::MIN_POSITIVE {
            return true;
        }
        // The digits d1..dn with exponent x denote the integer d1..dn
        // times 10^(x - n + 1).
        let mut text = NumberText::new();
        text.extend(&self.bytes[..count]);
        text.push(b'e');
        let scale = i64::from(exponent) - i64::try_from(count).unwrap_or(0) + 1;
        if scale < 0 {
            text.push(b'-');
        }
        push_decimal(&mut text, scale.unsigned_abs());
        std::str::from_utf8(text.as_bytes())
            .ok()
            .and_then(|text| text.parse::<f64>().ok())
            == Some(magnitude)
    }

    /// Drops the decimal point and exponent, leaving only the digits.
    fn compact(&mut self) -> (usize, i32) {
        let (mantissa, exponent) = split_exponential(&self.bytes[..self.len]);
        let mantissa_len = mantissa.len();
        let mut count = 0;
        for index in 0..mantissa_len {
            let byte = self.bytes[index];
            if byte != b'.' {
                self.bytes[count] = byte;
                count += 1;
            }
        }
        (count, exponent)
    }
}

impl std::fmt::Write for DigitBuffer {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        let end = self.len + text.len();
        self.bytes
            .get_mut(self.len..end)
            .ok_or(std::fmt::Error)?
            .copy_from_slice(text.as_bytes());
        self.len = end;
        Ok(())
    }
}

/// A fixed-size stack buffer that strings and formatted numbers are appended
/// to in place. Every `push_*` returns false, having changed nothing, when the
/// text does not fit, so a caller can fall back to a heap buffer.
pub struct StackBuffer<const CAPACITY: usize> {
    bytes: [u8; CAPACITY],
    len: usize,
}

impl<const CAPACITY: usize> Default for StackBuffer<CAPACITY> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const CAPACITY: usize> StackBuffer<CAPACITY> {
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self {
            bytes: [0; CAPACITY],
            len: 0,
        }
    }

    #[must_use]
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }

    /// The buffer as a string. Text of at most [`INLINE_STRING_BYTES`] is
    /// handed over as the zero-padded array the inline form stores, with no
    /// repacking.
    #[must_use]
    #[inline]
    pub fn into_string(&self) -> LuaString {
        const { assert!(CAPACITY >= INLINE_STRING_BYTES) };
        if self.len <= INLINE_STRING_BYTES
            && let Ok(inline) =
                <[u8; INLINE_STRING_BYTES]>::try_from(&self.bytes[..INLINE_STRING_BYTES])
        {
            return LuaString::from_inline_buffer(inline, self.len);
        }
        LuaString::new(self.as_bytes())
    }

    #[inline]
    pub fn push_bytes(&mut self, text: &[u8]) -> bool {
        let end = self.len + text.len();
        match self.bytes.get_mut(self.len..end) {
            Some(slot) => {
                slot.copy_from_slice(text);
                self.len = end;
                true
            }
            None => false,
        }
    }

    /// Appends the decimal digits of `magnitude`, with a leading `-` and a
    /// trailing `.0` as asked, writing each digit once.
    #[inline]
    fn push_digits(&mut self, negative: bool, magnitude: u64, suffix: bool) -> bool {
        let mut count = 1;
        let mut scale = 10_u64;
        while count < 20 && magnitude >= scale {
            count += 1;
            scale = scale.wrapping_mul(10);
        }
        let total = usize::from(negative) + count + if suffix { 2 } else { 0 };
        let Some(slot) = self.bytes.get_mut(self.len..self.len + total) else {
            return false;
        };
        if negative {
            slot[0] = b'-';
        }
        let mut value = magnitude;
        let mut position = total - if suffix { 2 } else { 0 };
        loop {
            position -= 1;
            #[allow(clippy::cast_possible_truncation, reason = "a digit")]
            {
                slot[position] = b'0' + (value % 10) as u8;
            }
            value /= 10;
            if value == 0 {
                break;
            }
        }
        if suffix {
            slot[total - 2] = b'.';
            slot[total - 1] = b'0';
        }
        self.len += total;
        true
    }

    /// Appends an integer-subtype value as [`format_integer`] spells it.
    #[inline]
    pub fn push_integer(&mut self, value: i64) -> bool {
        self.push_digits(value < 0, value.unsigned_abs(), false)
    }

    /// Appends a float as [`NumberFormat::format`] spells it.
    #[inline]
    pub fn push_number(&mut self, format: NumberFormat, value: f64) -> bool {
        let limit = if format == NumberFormat::Lua55 {
            1e15
        } else {
            1e14
        };
        let magnitude = value.abs();
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_precision_loss,
            clippy::cast_sign_loss,
            clippy::float_cmp,
            reason = "below 1e15 < 2^53, so the round trip is exact"
        )]
        if magnitude < limit && (magnitude as u64) as f64 == magnitude {
            return self.push_digits(
                value.is_sign_negative(),
                magnitude as u64,
                format.appends_float_suffix(),
            );
        }
        self.push_bytes(format.format(value).as_bytes())
    }
}

/// A formatted Lua number held on the stack.
#[derive(Clone, Copy)]
pub struct NumberText {
    bytes: [u8; 32],
    len: u8,
}

impl NumberText {
    const fn new() -> Self {
        Self {
            bytes: [0; 32],
            len: 0,
        }
    }

    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..usize::from(self.len)]
    }
}

impl Deref for NumberText {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl std::fmt::Debug for NumberText {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&String::from_utf8_lossy(self.as_bytes()))
    }
}

trait ByteSink {
    fn push(&mut self, byte: u8);
    fn extend(&mut self, bytes: &[u8]);
}

impl ByteSink for NumberText {
    fn push(&mut self, byte: u8) {
        self.bytes[usize::from(self.len)] = byte;
        self.len += 1;
    }

    fn extend(&mut self, bytes: &[u8]) {
        let start = usize::from(self.len);
        self.bytes[start..start + bytes.len()].copy_from_slice(bytes);
        #[allow(clippy::cast_possible_truncation)] // The buffer holds 32 bytes.
        {
            self.len += bytes.len() as u8;
        }
    }
}

impl ByteSink for Vec<u8> {
    fn push(&mut self, byte: u8) {
        Vec::push(self, byte);
    }

    fn extend(&mut self, bytes: &[u8]) {
        self.extend_from_slice(bytes);
    }
}

/// Appends C's `%a` spelling of a finite value, as Lua 5.4's `%q` writes
/// floats: `0x1.8p+1`, `0x0p+0`, or `0x0.0000000000001p-1022`.
pub fn push_hex_float(output: &mut Vec<u8>, value: f64) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bits = value.to_bits();
    if value.is_sign_negative() {
        output.push(b'-');
    }
    let biased = ((bits >> 52) & 0x7ff) as i32;
    let mut mantissa = bits & ((1 << 52) - 1);
    let (leading, exponent) = match (biased, mantissa) {
        (0, 0) => (b'0', 0),
        (0, _) => (b'0', -1022),
        _ => (b'1', biased - 1023),
    };
    output.extend_from_slice(b"0x");
    output.push(leading);
    if mantissa != 0 {
        output.push(b'.');
        let trailing = mantissa.trailing_zeros() / 4;
        mantissa >>= 4 * trailing;
        let nibbles = 13 - trailing;
        for index in (0..nibbles).rev() {
            output.push(HEX[((mantissa >> (4 * index)) & 0xf) as usize]);
        }
    }
    output.push(b'p');
    output.push(if exponent < 0 { b'-' } else { b'+' });
    let mut text = NumberText::new();
    push_decimal(&mut text, u64::from(exponent.unsigned_abs()));
    output.extend_from_slice(text.as_bytes());
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)] // Test inputs stay far below 2^53.
mod tests {
    use super::*;

    #[test]
    fn format_string_matches_format_for_every_profile() {
        let mut values = vec![
            0.0,
            -0.0,
            0.5,
            -7.25,
            1e13,
            -1e13,
            99_999_999_999_999.0,
            -99_999_999_999_999.0,
            1e14,
            -1e14,
            123_456_789_012_345.0,
            2f64.powi(53),
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            1e100,
            f64::MIN_POSITIVE,
        ];
        let mut power = 1.0;
        for _ in 0..15 {
            values.extend([power, power - 1.0, -power, 1.0 - power, power + 0.5]);
            power *= 10.0;
        }
        for format in [
            NumberFormat::Lua51,
            NumberFormat::LuaJit,
            NumberFormat::Lua53,
            NumberFormat::Lua55,
        ] {
            for value in &values {
                assert_eq!(
                    format.format_string(*value).as_bytes(),
                    format.format(*value).as_bytes(),
                    "{format:?} {value}"
                );
            }
        }
    }

    #[test]
    fn stack_buffer_numbers_match_format_for_every_profile() {
        let mut values = vec![
            0.0,
            -0.0,
            0.5,
            -7.25,
            1e13,
            -1e13,
            99_999_999_999_999.0,
            1e14,
            -1e14,
            999_999_999_999_999.0,
            1e15,
            2f64.powi(53),
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -f64::NAN,
            1e100,
            f64::MIN_POSITIVE,
            5e-324,
            f64::MAX,
        ];
        let mut power = 1.0;
        for _ in 0..17 {
            values.extend([power, power - 1.0, -power, 1.0 - power, power + 0.5]);
            power *= 10.0;
        }
        for format in [
            NumberFormat::Lua51,
            NumberFormat::LuaJit,
            NumberFormat::Lua53,
            NumberFormat::Lua55,
        ] {
            for value in &values {
                let mut buffer = StackBuffer::<64>::new();
                assert!(buffer.push_bytes(b"<"));
                assert!(buffer.push_number(format, *value), "{format:?} {value}");
                let mut expected = b"<".to_vec();
                expected.extend_from_slice(format.format(*value).as_bytes());
                assert_eq!(buffer.as_bytes(), &expected[..], "{format:?} {value}");
            }
        }
        for value in [0, 1, -1, 9, 10, i64::MAX, i64::MIN, -9_999, 1_000_000] {
            let mut buffer = StackBuffer::<64>::new();
            assert!(buffer.push_integer(value));
            assert_eq!(
                buffer.as_bytes(),
                format_integer(value).as_bytes(),
                "{value}"
            );
        }
    }

    #[test]
    fn stack_buffer_refuses_text_that_does_not_fit_and_keeps_its_contents() {
        let mut buffer = StackBuffer::<8>::new();
        assert!(buffer.push_bytes(b"abcdef"));
        assert!(!buffer.push_bytes(b"ghi"));
        assert!(!buffer.push_integer(12345));
        assert!(!buffer.push_number(NumberFormat::Lua53, 3.0));
        assert_eq!(buffer.as_bytes(), b"abcdef");
        assert!(buffer.push_number(NumberFormat::Lua51, 3.0));
        assert_eq!(buffer.as_bytes(), b"abcdef3");
        assert!(buffer.push_bytes(b"g"));
        assert!(!buffer.push_bytes(b"h"));
    }

    #[test]
    fn stack_buffer_strings_are_exact_at_the_inline_boundary() {
        for length in [0, 1, 21, 22, 23, 40, 64] {
            let mut buffer = StackBuffer::<64>::new();
            let bytes = vec![b'q'; length];
            assert!(buffer.push_bytes(&bytes));
            assert_eq!(buffer.into_string().as_bytes(), &bytes[..], "{length}");
        }
    }

    fn text(format: NumberFormat, value: f64) -> String {
        String::from_utf8(format.format(value).to_vec()).unwrap()
    }

    #[test]
    fn profiles_match_stock_tostring() {
        // Expected strings come from lua5.1, luajit 2.1, lua5.4, and lua5.5.
        let cases: &[(f64, &str, &str, &str, &str)] = &[
            (0.1 + 0.2, "0.3", "0.3", "0.3", "0.30000000000000004"),
            (
                2f64.powi(53),
                "9.007199254741e+15",
                "9.007199254741e+15",
                "9.007199254741e+15",
                "9007199254740992.0",
            ),
            (
                1.0 / 3.0,
                "0.33333333333333",
                "0.33333333333333",
                "0.33333333333333",
                "0.33333333333333331",
            ),
            (
                100.0 / 3.0,
                "33.333333333333",
                "33.333333333333",
                "33.333333333333",
                "33.333333333333336",
            ),
            (1e15, "1e+15", "1e+15", "1e+15", "1e+15"),
            (1e14, "1e+14", "1e+14", "1e+14", "100000000000000.0"),
            (
                123_456_789_012_345.0,
                "1.2345678901234e+14",
                "1.2345678901234e+14",
                "1.2345678901234e+14",
                "123456789012345.0",
            ),
            (
                99_999_999_999_999.0,
                "99999999999999",
                "99999999999999",
                "99999999999999.0",
                "99999999999999.0",
            ),
            (
                12_345_678_901_234.5,
                "12345678901234",
                "12345678901234",
                "12345678901234.0",
                "12345678901234.5",
            ),
            (3.0, "3", "3", "3.0", "3.0"),
            (-0.0, "-0", "-0", "-0.0", "-0.0"),
            (
                2f64.powi(63),
                "9.2233720368548e+18",
                "9.2233720368548e+18",
                "9.2233720368548e+18",
                "9.2233720368547758e+18",
            ),
            (1e-4, "0.0001", "0.0001", "0.0001", "0.0001"),
            (1e-5, "1e-05", "1e-05", "1e-05", "1e-05"),
            (
                0.000_012_34,
                "1.234e-05",
                "1.234e-05",
                "1.234e-05",
                "1.234e-05",
            ),
            (
                -std::f64::consts::PI,
                "-3.1415926535898",
                "-3.1415926535898",
                "-3.1415926535898",
                "-3.1415926535897931",
            ),
            (1e100, "1e+100", "1e+100", "1e+100", "1e+100"),
            (
                5e-324,
                "4.9406564584125e-324",
                "4.9406564584125e-324",
                "4.9406564584125e-324",
                "4.94065645841247e-324",
            ),
            (f64::INFINITY, "inf", "inf", "inf", "inf"),
            (f64::NEG_INFINITY, "-inf", "-inf", "-inf", "-inf"),
            (f64::NAN, "nan", "nan", "nan", "nan"),
            (-f64::NAN, "-nan", "nan", "-nan", "-nan"),
        ];
        for (value, lua51, luajit, lua53, lua55) in cases {
            assert_eq!(text(NumberFormat::Lua51, *value), *lua51, "lua51 {value:e}");
            assert_eq!(
                text(NumberFormat::LuaJit, *value),
                *luajit,
                "luajit {value:e}"
            );
            assert_eq!(text(NumberFormat::Lua53, *value), *lua53, "lua53 {value:e}");
            assert_eq!(text(NumberFormat::Lua55, *value), *lua55, "lua55 {value:e}");
        }
        assert_eq!(text(NumberFormat::Lua53.for_file_write(), 3.0), "3");
        assert_eq!(text(NumberFormat::Lua55.for_file_write(), 3.0), "3.0");
        assert_eq!(format_integer(i64::MIN).as_bytes(), b"-9223372036854775808");
    }

    #[test]
    fn integer_rounding_matches_exact_decimal_expansion() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut checked = 0;
        for index in 0..1_000_000 {
            let value = match index % 4 {
                // Arbitrary bit patterns across the engine's range.
                0 => f64::from_bits(0x3ee0_0000_0000_0000 + next() % 0x0120_0000_0000_0000),
                // Fifteen-digit integers plus halves: exact ties.
                1 => (next() % 900_000_000_000_000 + 100_000_000_000_000) as f64 / 10.0,
                // Short decimals and values straddling powers of ten.
                2 => (next() % 10_000_000) as f64 / 10f64.powi((next() % 12) as i32),
                _ => {
                    10f64.powi((next() % 18) as i32 - 5)
                        * (1.0 + (next() % 3) as f64 * f64::EPSILON - f64::EPSILON)
                }
            };
            let Some((digits, exponent)) = fourteen_digits(value) else {
                continue;
            };
            checked += 1;
            let expected = format!("{value:.13e}");
            let (mantissa, expected_exponent) = expected.split_once('e').unwrap();
            assert_eq!(digits.to_string(), mantissa.replace('.', ""), "{value:e}");
            assert_eq!(exponent.to_string(), expected_exponent, "{value:e}");
        }
        assert!(checked > 900_000);
    }

    #[test]
    fn general_and_hex_spellings_match_printf() {
        let general = |value: f64, precision: usize, alternate: bool| {
            let mut output = Vec::new();
            push_general(&mut output, value, precision, alternate);
            String::from_utf8(output).unwrap()
        };
        assert_eq!(general(2f64.powi(53), 6, false), "9.0072e+15");
        assert_eq!(general(1_234_567.0, 3, false), "1.23e+06");
        assert_eq!(general(100_000.0, 6, false), "100000");
        assert_eq!(general(1e6, 6, false), "1e+06");
        assert_eq!(general(0.0001, 6, false), "0.0001");
        assert_eq!(general(1.5, 6, true), "1.50000");
        assert_eq!(general(100.0, 3, true), "100.");
        assert_eq!(general(2.5, 0, false), "2");
        assert_eq!(general(0.0, 6, false), "0");
        assert_eq!(general(0.1, 30, false), "0.100000000000000005551115123126");

        let hex = |value: f64| {
            let mut output = Vec::new();
            push_hex_float(&mut output, value);
            String::from_utf8(output).unwrap()
        };
        assert_eq!(hex(1.0 / 3.0), "0x1.5555555555555p-2");
        assert_eq!(hex(3.0), "0x1.8p+1");
        assert_eq!(hex(1.0), "0x1p+0");
        assert_eq!(hex(-0.0), "-0x0p+0");
        assert_eq!(hex(5e-324), "0x0.0000000000001p-1022");
        assert_eq!(hex(2f64.powi(63)), "0x1p+63");
    }
}
