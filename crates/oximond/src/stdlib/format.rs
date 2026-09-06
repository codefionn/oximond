//! Format-specification parsing and value formatting for string.format.

use crate::runtime::{
    NumberFormat, Runtime, RuntimeError, RuntimeResult, Value, push_general, push_hex_float,
};
use crate::stdlib::FormatSpecification;
use crate::stdlib::args::number_to_isize;
use crate::stdlib::strings::string_argument_in;
use std::borrow::Cow;
use std::mem::MaybeUninit;

impl FormatSpecification {
    pub(crate) const LEFT: u8 = 1;
    pub(crate) const PLUS: u8 = 1 << 1;
    pub(crate) const SPACE: u8 = 1 << 2;
    pub(crate) const ALTERNATE: u8 = 1 << 3;
    pub(crate) const ZERO: u8 = 1 << 4;

    pub(crate) fn has_flag(&self, flag: u8) -> bool {
        self.flags & flag != 0
    }

    pub(crate) fn has_modifiers(&self) -> bool {
        self.flags != 0 || self.width.is_some() || self.precision.is_some()
    }
}

pub(crate) fn parse_format_specification(
    format: &[u8],
    mut index: usize,
    pointer_supported: bool,
) -> RuntimeResult<(FormatSpecification, usize)> {
    let mut specification = FormatSpecification::default();
    while let Some(flag) = format.get(index) {
        match flag {
            b'-' => specification.flags |= FormatSpecification::LEFT,
            b'+' => specification.flags |= FormatSpecification::PLUS,
            b' ' => specification.flags |= FormatSpecification::SPACE,
            b'#' => specification.flags |= FormatSpecification::ALTERNATE,
            b'0' => specification.flags |= FormatSpecification::ZERO,
            _ => break,
        }
        index += 1;
    }
    let (width, after_width) = parse_decimal(format, index)?;
    specification.width = width;
    index = after_width;
    if format.get(index) == Some(&b'.') {
        let (precision, after_precision) = parse_decimal(format, index + 1)?;
        specification.precision = Some(precision.unwrap_or(0));
        index = after_precision;
    }
    let conversion = *format
        .get(index)
        .ok_or_else(|| RuntimeError::message("unfinished string.format directive"))?;
    if !(matches!(
        conversion,
        b'%' | b'q'
            | b's'
            | b'c'
            | b'd'
            | b'i'
            | b'u'
            | b'o'
            | b'x'
            | b'X'
            | b'e'
            | b'E'
            | b'f'
            | b'g'
            | b'G'
    ) || pointer_supported && conversion == b'p')
    {
        return Err(RuntimeError::message(format!(
            "unsupported string.format conversion %{}",
            char::from(conversion)
        )));
    }
    specification.conversion = conversion;
    Ok((specification, index + 1))
}

pub(crate) fn parse_decimal(
    format: &[u8],
    mut index: usize,
) -> RuntimeResult<(Option<usize>, usize)> {
    let start = index;
    let mut value = 0_usize;
    while let Some(digit) = format.get(index).filter(|digit| digit.is_ascii_digit()) {
        value = value
            .checked_mul(10)
            .and_then(|value| value.checked_add(usize::from(*digit - b'0')))
            .ok_or_else(|| {
                RuntimeError::message("string.format width or precision is too large")
            })?;
        index += 1;
    }
    Ok((if index == start { None } else { Some(value) }, index))
}

/// A `%s` or `%q` operand as a string. `string.format` attributes the
/// error to the operand's argument position.
fn format_string_operand(numbers: NumberFormat, argument: &Value) -> RuntimeResult<Cow<'_, [u8]>> {
    match argument {
        Value::String(value) => Ok(Cow::Borrowed(value.as_bytes())),
        Value::Integer(value) => Ok(Cow::Owned(crate::runtime::format_integer(*value).to_vec())),
        Value::Number(value) => Ok(Cow::Owned(numbers.format(*value).to_vec())),
        value => Err(RuntimeError::TypeError {
            operation: "call string.format with",
            expected: "string",
            found: value.type_name(),
        }),
    }
}

pub(crate) fn format_value(
    argument: &Value,
    specification: &FormatSpecification,
    modern_quote: bool,
    numbers: NumberFormat,
) -> RuntimeResult<Vec<u8>> {
    match specification.conversion {
        b's' => {
            let value = match crate::runtime::cint64::CInt64::from_value(argument) {
                // LuaJIT's `%s` shows 64-bit integer cdata as `tostring` does.
                Some(value) => std::borrow::Cow::Owned(value.to_text().into_bytes()),
                None => format_string_operand(numbers, argument)?,
            };
            if modern_quote && specification.has_modifiers() && value.contains(&b'\0') {
                return Err(RuntimeError::message("string.format value contains zeros"));
            }
            let length = specification
                .precision
                .map_or(value.len(), |limit| limit.min(value.len()));
            Ok(apply_format_width(
                value[..length].to_vec(),
                specification,
                0,
            ))
        }
        b'q' => {
            if specification.has_modifiers() {
                return Err(RuntimeError::message(
                    "string.format does not support modifiers on %q",
                ));
            }
            if modern_quote {
                quote_lua_value(argument)
            } else {
                let value = format_string_operand(numbers, argument)?;
                // Only LuaJIT, of the profiles here, writes 5.3-style escapes.
                Ok(quote_lua_bytes(&value, numbers == NumberFormat::Lua51))
            }
        }
        b'p' => Ok(format_pointer(argument, specification)),
        b'c' => {
            let value = Runtime::coerce_number(argument, "call string.format with")?;
            let value = number_to_isize(value.trunc(), "string.format %c value")?;
            let value = u8::try_from(value).map_err(|_| {
                RuntimeError::message("string.format %c value must be in the range 0 through 255")
            })?;
            let mut formatted = apply_format_width(vec![value], specification, 0);
            // Lua 5.1 formats `%c` into a C string and measures it with
            // `strlen`, so a NUL byte ends the conversion's output.
            if !modern_quote
                && numbers == NumberFormat::Lua51
                && let Some(end) = formatted.iter().position(|byte| *byte == 0)
            {
                formatted.truncate(end);
            }
            Ok(formatted)
        }
        b'd' | b'i' | b'u' | b'o' | b'x' | b'X' => format_integer(argument, specification, numbers),
        b'e' | b'E' | b'f' | b'g' | b'G' => format_float(argument, specification, numbers),
        _ => Err(RuntimeError::message(
            "unsupported string.format conversion",
        )),
    }
}

/// Appends the common integer forms directly into the final result. This
/// avoids allocating a digit vector, a padded vector, and then copying both.
pub(crate) fn append_format_value(
    output: &mut Vec<u8>,
    argument: &Value,
    specification: &FormatSpecification,
    modern_quote: bool,
    numbers: NumberFormat,
) -> RuntimeResult<()> {
    if matches!(
        specification.conversion,
        b'd' | b'i' | b'u' | b'o' | b'x' | b'X'
    ) && specification.precision.is_none()
    {
        return append_integer_format(output, argument, specification, numbers);
    }
    if specification.conversion == b's'
        && let Value::String(text) = argument
        && !(modern_quote && specification.has_modifiers() && text.as_bytes().contains(&0))
    {
        // A string operand is copied once, straight into the result.
        let text = text.as_bytes();
        let length = specification
            .precision
            .map_or(text.len(), |limit| limit.min(text.len()));
        let padding = specification
            .width
            .map_or(0, |width| width.saturating_sub(length));
        output
            .try_reserve(length + padding)
            .map_err(|_| RuntimeError::message("string.format result is too large"))?;
        let left = specification.has_flag(FormatSpecification::LEFT);
        let fill = if specification.has_flag(FormatSpecification::ZERO)
            && specification.precision.is_none()
        {
            b'0'
        } else {
            b' '
        };
        if !left {
            output.extend(std::iter::repeat_n(fill, padding));
        }
        output.extend_from_slice(&text[..length]);
        if left {
            output.extend(std::iter::repeat_n(b' ', padding));
        }
        return Ok(());
    }
    if specification.conversion == b'f' {
        let value = match argument {
            Value::Number(number) => *number,
            argument => Runtime::coerce_number(argument, "call string.format with")?,
        };
        if append_fixed_format(output, value, specification)? {
            return Ok(());
        }
    }
    let value = format_value(argument, specification, modern_quote, numbers)?;
    append_format_bytes(output, &value)
}

fn append_fixed_format(
    output: &mut Vec<u8>,
    value: f64,
    specification: &FormatSpecification,
) -> RuntimeResult<bool> {
    const ISCALES: [u64; 4] = [1, 10, 100, 1000];
    let precision = specification.precision.unwrap_or(6);
    let Some(scaled) = scaled_fixed(value.abs(), precision) else {
        return Ok(false);
    };
    let scale = ISCALES[precision];
    let mut digits = [0_u8; 32];
    let mut start = digits.len();
    let mut whole = scaled / scale;
    loop {
        start -= 1;
        digits[start] = b'0' + (whole % 10) as u8;
        whole /= 10;
        if whole == 0 {
            break;
        }
    }
    let sign = if value.is_sign_negative() {
        Some(b'-')
    } else if specification.has_flag(FormatSpecification::PLUS) {
        Some(b'+')
    } else if specification.has_flag(FormatSpecification::SPACE) {
        Some(b' ')
    } else {
        None
    };
    let content_len =
        usize::from(sign.is_some()) + digits.len() - start + usize::from(precision > 0) + precision;
    let width = specification.width.unwrap_or(content_len).max(content_len);
    output
        .try_reserve(width)
        .map_err(|_| RuntimeError::message("string.format result is too large"))?;
    let padding = width - content_len;
    let left = specification.has_flag(FormatSpecification::LEFT);
    let zero = specification.has_flag(FormatSpecification::ZERO)
        && specification.precision.is_none()
        && !left;
    if !left && !zero {
        output.extend(std::iter::repeat_n(b' ', padding));
    }
    if let Some(sign) = sign {
        output.push(sign);
    }
    if zero {
        output.extend(std::iter::repeat_n(b'0', padding));
    }
    output.extend_from_slice(&digits[start..]);
    if precision > 0 {
        output.push(b'.');
        let mut fraction = scaled % scale;
        let mut tail = [b'0'; 3];
        for digit in tail[..precision].iter_mut().rev() {
            *digit = b'0' + (fraction % 10) as u8;
            fraction /= 10;
        }
        output.extend_from_slice(&tail[..precision]);
    }
    if left {
        output.extend(std::iter::repeat_n(b' ', padding));
    }
    Ok(true)
}

fn append_integer_format(
    output: &mut Vec<u8>,
    argument: &Value,
    specification: &FormatSpecification,
    numbers: NumberFormat,
) -> RuntimeResult<()> {
    let number = integer_operand(argument, numbers)?;
    let signed = matches!(specification.conversion, b'd' | b'i');
    let negative = signed && number < 0;
    let mut magnitude = if signed {
        number.unsigned_abs()
    } else {
        u64::from_ne_bytes(number.to_ne_bytes())
    };
    let alphabet = if specification.conversion == b'X' {
        b"0123456789ABCDEF"
    } else {
        b"0123456789abcdef"
    };
    // Twenty-two digits hold any `u64` in octal; decimal needs twenty.
    let mut digits = [0_u8; 24];
    let mut start = digits.len();
    // Constant divisors compile to multiplies and shifts; a divisor chosen
    // at run time costs a hardware division per digit.
    match specification.conversion {
        b'o' => loop {
            start -= 1;
            digits[start] = alphabet[(magnitude & 7) as usize];
            magnitude >>= 3;
            if magnitude == 0 {
                break;
            }
        },
        b'x' | b'X' => loop {
            start -= 1;
            digits[start] = alphabet[(magnitude & 15) as usize];
            magnitude >>= 4;
            if magnitude == 0 {
                break;
            }
        },
        _ => loop {
            start -= 1;
            digits[start] = b'0' + (magnitude % 10) as u8;
            magnitude /= 10;
            if magnitude == 0 {
                break;
            }
        },
    }
    let mut prefix = [0_u8; 3];
    let mut prefix_len = 0;
    if negative {
        prefix[prefix_len] = b'-';
        prefix_len += 1;
    } else if signed && specification.has_flag(FormatSpecification::PLUS) {
        prefix[prefix_len] = b'+';
        prefix_len += 1;
    } else if signed && specification.has_flag(FormatSpecification::SPACE) {
        prefix[prefix_len] = b' ';
        prefix_len += 1;
    }
    if specification.has_flag(FormatSpecification::ALTERNATE) && number != 0 {
        match specification.conversion {
            b'o' if digits[start] != b'0' => {
                prefix[prefix_len] = b'0';
                prefix_len += 1;
            }
            b'x' | b'X' => {
                prefix[prefix_len] = b'0';
                prefix[prefix_len + 1] = specification.conversion;
                prefix_len += 2;
            }
            _ => {}
        }
    }
    let content_len = prefix_len + digits.len() - start;
    let width = specification.width.unwrap_or(content_len).max(content_len);
    output
        .try_reserve(width)
        .map_err(|_| RuntimeError::message("string.format result is too large"))?;
    let padding = width - content_len;
    let left = specification.has_flag(FormatSpecification::LEFT);
    let zero = specification.has_flag(FormatSpecification::ZERO) && !left;
    if !left && !zero {
        output.extend(std::iter::repeat_n(b' ', padding));
    }
    output.extend_from_slice(&prefix[..prefix_len]);
    if zero {
        output.extend(std::iter::repeat_n(b'0', padding));
    }
    output.extend_from_slice(&digits[start..]);
    if left {
        output.extend(std::iter::repeat_n(b' ', padding));
    }
    Ok(())
}

/// A conversion of a constant format that a plain number operand can take
/// without the general machinery: `%d`, `%i`, `%x`, `%X` and `%f` with a
/// precision up to three, and at most a width and the `-` and `0` flags.
/// A prepared format decides this once, when it parses; [`Self::append`]
/// then writes the digits straight into the result.
#[derive(Clone, Copy)]
pub(crate) struct FastConversion {
    kind: FastKind,
    width: usize,
    zero: bool,
    left: bool,
}

#[derive(Clone, Copy)]
enum FastKind {
    Decimal,
    Hex { upper: bool },
    Fixed { precision: usize },
}

impl FastConversion {
    /// The widest field the fast path pads; a wider one takes the general
    /// path.
    const MAX_WIDTH: usize = 64;

    pub(crate) fn of(specification: &FormatSpecification) -> Option<Self> {
        let allowed = FormatSpecification::LEFT | FormatSpecification::ZERO;
        if specification.flags & !allowed != 0
            || specification
                .width
                .is_some_and(|width| width > Self::MAX_WIDTH)
        {
            return None;
        }
        let kind = match (specification.conversion, specification.precision) {
            (b'd' | b'i', None) => FastKind::Decimal,
            (b'x' | b'X', None) => FastKind::Hex {
                upper: specification.conversion == b'X',
            },
            (b'f', precision) => {
                let precision = precision.unwrap_or(6);
                // The general fixed path pads a precise `%f` with spaces.
                if precision > 3
                    || (specification.has_flag(FormatSpecification::ZERO)
                        && specification.precision.is_some())
                {
                    return None;
                }
                FastKind::Fixed { precision }
            }
            _ => return None,
        };
        let left = specification.has_flag(FormatSpecification::LEFT);
        Some(Self {
            kind,
            width: specification.width.unwrap_or(0),
            zero: specification.has_flag(FormatSpecification::ZERO) && !left,
            left,
        })
    }

    /// Appends `argument` formatted, or returns `false` and appends nothing
    /// where only the general path can answer: a non-number, a number the
    /// conversion has to round or refuse, or a value `scaled_fixed` leaves
    /// to the exact backend.
    // Inline across codegen units, so `format_operations` keeps the writer in
    // its loop whatever unit the partition puts it in.
    #[inline]
    pub(crate) fn append(&self, output: &mut Vec<u8>, argument: &Value) -> bool {
        match self.kind {
            FastKind::Fixed { precision } => {
                let value = match argument {
                    Value::Number(number) => *number,
                    #[allow(clippy::cast_precision_loss)] // As `coerce_number` converts.
                    Value::Integer(integer) => *integer as f64,
                    _ => return false,
                };
                self.append_fixed(output, value, precision)
            }
            kind => {
                let number = match argument {
                    Value::Integer(integer) | Value::Int64(integer) => *integer,
                    // An integral number in range is the same integer in
                    // every profile; any other takes the general path.
                    Value::Number(number)
                        if (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0)
                            .contains(number) =>
                    {
                        #[allow(clippy::cast_possible_truncation)] // In range.
                        let integer = *number as i64;
                        #[allow(clippy::cast_precision_loss, clippy::float_cmp)]
                        // An integral number is its own round trip; no `fract` call.
                        if integer as f64 != *number {
                            return false;
                        }
                        integer
                    }
                    _ => return false,
                };
                self.append_integer(output, kind, number)
            }
        }
    }

    fn append_integer(&self, output: &mut Vec<u8>, kind: FastKind, number: i64) -> bool {
        if let FastKind::Hex { upper } = kind {
            let alphabet = if upper {
                b"0123456789ABCDEF"
            } else {
                b"0123456789abcdef"
            };
            let magnitude = number.cast_unsigned();
            let count = ((64 - (magnitude | 1).leading_zeros()) as usize).div_ceil(4);
            self.write(output, false, count, 0, |digits, _| {
                let mut magnitude = magnitude;
                for digit in digits.iter_mut().rev() {
                    digit.write(alphabet[(magnitude & 15) as usize]);
                    magnitude >>= 4;
                }
            })
        } else {
            let magnitude = number.unsigned_abs();
            self.write(
                output,
                number < 0,
                decimal_length(magnitude),
                0,
                |digits, _| write_decimal(digits, magnitude),
            )
        }
    }

    fn append_fixed(&self, output: &mut Vec<u8>, value: f64, precision: usize) -> bool {
        let Some(scaled) = scaled_fixed(value.abs(), precision) else {
            return false;
        };
        // Constant divisors compile to multiplies; see `append_integer_format`.
        let (whole, fraction) = match precision {
            0 => (scaled, 0),
            1 => (scaled / 10, scaled % 10),
            2 => (scaled / 100, scaled % 100),
            _ => (scaled / 1000, scaled % 1000),
        };
        let tail = if precision > 0 { precision + 1 } else { 0 };
        self.write(
            output,
            value.is_sign_negative(),
            decimal_length(whole),
            tail,
            |digits, tail| {
                write_decimal(digits, whole);
                if let Some((point, places)) = tail.split_first_mut() {
                    point.write(b'.');
                    write_decimal(places, fraction);
                }
            },
        )
    }

    /// Appends a field of the width straight into the spare room of
    /// `output`: the padding, a sign, `count` digits and `tail` more bytes.
    /// `fill` writes the digits and the tail, each in full.
    #[allow(unsafe_code)]
    fn write(
        &self,
        output: &mut Vec<u8>,
        negative: bool,
        count: usize,
        tail: usize,
        fill: impl FnOnce(&mut [MaybeUninit<u8>], &mut [MaybeUninit<u8>]),
    ) -> bool {
        let content = usize::from(negative) + count + tail;
        let padding = self.width.saturating_sub(content);
        let length = content + padding;
        if output.try_reserve(length).is_err() {
            return false;
        }
        let before = if self.zero || self.left { 0 } else { padding };
        let zeros = if self.zero { padding } else { 0 };
        let mut room = &mut output.spare_capacity_mut()[..length];
        let (spaces, rest) = room.split_at_mut(before);
        for byte in spaces {
            byte.write(b' ');
        }
        room = rest;
        if negative {
            room[0].write(b'-');
            room = &mut room[1..];
        }
        let (zero_run, rest) = room.split_at_mut(zeros);
        for byte in zero_run {
            byte.write(b'0');
        }
        let (digits, rest) = rest.split_at_mut(count);
        let (tail, trailing) = rest.split_at_mut(tail);
        fill(digits, tail);
        for byte in trailing {
            byte.write(b' ');
        }
        // SAFETY: the pieces above cover all `length` bytes of the room,
        // and each one was written.
        unsafe { output.set_len(output.len() + length) };
        true
    }
}

/// The number of decimal digits of `value`.
fn decimal_length(value: u64) -> usize {
    // 1233 / 4096 is just above log10(2), so `bits * 1233 >> 12` is the
    // digit count of a power of two, one low at most; the table fixes it.
    const POWERS: [u64; 20] = {
        let mut powers = [1_u64; 20];
        let mut index = 1;
        while index < 20 {
            powers[index] = powers[index - 1] * 10;
            index += 1;
        }
        powers
    };
    let value = value | 1;
    let bits = 64 - value.leading_zeros() as usize;
    let estimate = (bits * 1233) >> 12;
    estimate + usize::from(value >= POWERS[estimate])
}

/// Writes `value` into `digits`, right aligned and zero extended to their
/// length, which holds all of its digits.
fn write_decimal(digits: &mut [MaybeUninit<u8>], mut value: u64) {
    for digit in digits.iter_mut().rev() {
        digit.write(b'0' + (value % 10) as u8);
        value /= 10;
    }
}

/// `append_format_bytes` for the short pieces of a format, which it copies
/// with a pair of overlapping loads and stores instead of a call to `memcpy`.
#[allow(unsafe_code)]
#[inline]
pub(crate) fn append_short(output: &mut Vec<u8>, bytes: &[u8]) -> RuntimeResult<()> {
    let length = bytes.len();
    if length > 16 {
        return append_format_bytes(output, bytes);
    }
    output
        .try_reserve(length)
        .map_err(|_| RuntimeError::message("string.format result is too large"))?;
    let source = bytes.as_ptr();
    // SAFETY: `try_reserve` made room for `length` bytes past the end of
    // `output`, and each branch reads and writes only inside `length`
    // bytes: two pieces of equal size that overlap when `length` is not a
    // multiple of it.
    unsafe {
        let target = output.as_mut_ptr().add(output.len());
        if length >= 8 {
            let head = source.cast::<u64>().read_unaligned();
            let tail = source.add(length - 8).cast::<u64>().read_unaligned();
            target.cast::<u64>().write_unaligned(head);
            target.add(length - 8).cast::<u64>().write_unaligned(tail);
        } else if length >= 4 {
            let head = source.cast::<u32>().read_unaligned();
            let tail = source.add(length - 4).cast::<u32>().read_unaligned();
            target.cast::<u32>().write_unaligned(head);
            target.add(length - 4).cast::<u32>().write_unaligned(tail);
        } else if length >= 2 {
            let head = source.cast::<u16>().read_unaligned();
            let tail = source.add(length - 2).cast::<u16>().read_unaligned();
            target.cast::<u16>().write_unaligned(head);
            target.add(length - 2).cast::<u16>().write_unaligned(tail);
        } else if length == 1 {
            *target = *source;
        }
        output.set_len(output.len() + length);
    }
    Ok(())
}

pub(crate) fn format_pointer(argument: &Value, specification: &FormatSpecification) -> Vec<u8> {
    let value = match argument {
        Value::Nil | Value::Boolean(_) | Value::Integer(_) | Value::Number(_) => b"(null)".to_vec(),
        Value::String(value) if value.len() <= 40 => {
            let mut pointer = b"0xs".to_vec();
            for byte in value.as_bytes() {
                pointer.extend_from_slice(format!("{byte:02x}").as_bytes());
            }
            pointer
        }
        Value::String(value) => format!("0x{:x}", value.identity()).into_bytes(),
        value => {
            let mut hash = 0xcbf2_9ce4_8422_2325_u64;
            for byte in format!("{value:?}").bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            format!("0x{hash:x}").into_bytes()
        }
    };
    apply_format_width(value, specification, 0)
}

pub(crate) fn quote_lua_value(value: &Value) -> RuntimeResult<Vec<u8>> {
    match value {
        Value::String(value) => Ok(quote_lua_bytes(value.as_bytes(), false)),
        Value::Integer(value) => Ok(value.to_string().into_bytes()),
        Value::Number(value) if value.is_nan() => Ok(b"(0/0)".to_vec()),
        Value::Number(value) if value.is_infinite() && value.is_sign_negative() => {
            Ok(b"-1e9999".to_vec())
        }
        Value::Number(value) if value.is_infinite() => Ok(b"1e9999".to_vec()),
        // Lua 5.4 writes floats in C's `%a` hexadecimal so they read back exactly.
        Value::Number(value) => {
            let mut text = Vec::new();
            push_hex_float(&mut text, *value);
            Ok(text)
        }
        Value::Boolean(true) => Ok(b"true".to_vec()),
        Value::Boolean(false) => Ok(b"false".to_vec()),
        Value::Nil => Ok(b"nil".to_vec()),
        _ => Err(RuntimeError::message("value has no literal form")),
    }
}

/// An integer conversion's operand. Lua 5.3 and later refuse a number with
/// no exact integer value; earlier versions and `LuaJIT` truncate it.
fn integer_operand(argument: &Value, numbers: NumberFormat) -> RuntimeResult<i64> {
    Ok(match argument {
        Value::Integer(value) | Value::Int64(value) => *value,
        // LuaJIT formats the bits of a `uint64_t` like an `int64_t`.
        Value::UInt64(value) => value.cast_signed(),
        value => {
            let number = match value {
                Value::Number(number) => *number,
                value => Runtime::coerce_number(value, "call string.format with")?,
            };
            if matches!(numbers, NumberFormat::Lua53 | NumberFormat::Lua55) {
                if number.fract() != 0.0
                    || !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0)
                        .contains(&number)
                {
                    // The caller supplies the position.
                    return Err(RuntimeError::argument_error(
                        0,
                        "number has no integer representation",
                    ));
                }
                #[allow(clippy::cast_possible_truncation)] // Integral and within i64 bounds.
                return Ok(number as i64);
            }
            // Out of range, infinite and NaN values become the minimum, as
            // x86 converts them.
            if (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&number) {
                #[allow(clippy::cast_possible_truncation)] // Within the i64 range.
                {
                    number as i64
                }
            } else {
                i64::MIN
            }
        }
    })
}

pub(crate) fn format_integer(
    argument: &Value,
    specification: &FormatSpecification,
    numbers: NumberFormat,
) -> RuntimeResult<Vec<u8>> {
    let number = integer_operand(argument, numbers)?;
    let signed = matches!(specification.conversion, b'd' | b'i');
    let negative = signed && number < 0;
    let magnitude = if signed {
        number.unsigned_abs()
    } else {
        u64::from_ne_bytes(number.to_ne_bytes())
    };
    let (base, uppercase) = match specification.conversion {
        b'o' => (8, false),
        b'x' => (16, false),
        b'X' => (16, true),
        _ => (10, false),
    };
    let mut digits = unsigned_digits(magnitude, base, uppercase);
    if specification.precision == Some(0) && magnitude == 0 {
        digits.clear();
    }
    if let Some(precision) = specification.precision {
        while digits.len() < precision {
            digits.insert(0, b'0');
        }
    }
    let mut prefix = Vec::new();
    if negative {
        prefix.push(b'-');
    } else if signed && specification.has_flag(FormatSpecification::PLUS) {
        prefix.push(b'+');
    } else if signed && specification.has_flag(FormatSpecification::SPACE) {
        prefix.push(b' ');
    }
    if specification.has_flag(FormatSpecification::ALTERNATE) && magnitude != 0 {
        match specification.conversion {
            b'o' if !digits.starts_with(b"0") => prefix.push(b'0'),
            b'x' => prefix.extend_from_slice(b"0x"),
            b'X' => prefix.extend_from_slice(b"0X"),
            _ => {}
        }
    }
    let prefix_length = prefix.len();
    prefix.extend_from_slice(&digits);
    Ok(apply_format_width(prefix, specification, prefix_length))
}

pub(crate) fn format_float(
    argument: &Value,
    specification: &FormatSpecification,
    numbers: NumberFormat,
) -> RuntimeResult<Vec<u8>> {
    let value = Runtime::coerce_number(argument, "call string.format with")?;
    if let Some(spelling) = numbers.non_finite(value) {
        // printf pads NaN and infinities with spaces even under `0`.
        let mut specification = FormatSpecification {
            flags: specification.flags & !FormatSpecification::ZERO,
            ..*specification
        };
        // LuaJIT ignores `+` for NaN but still honors a space sign flag.
        if numbers == NumberFormat::LuaJit && value.is_nan() {
            specification.flags &= !FormatSpecification::PLUS;
        }
        let (negative, digits) = match spelling.split_first() {
            Some((b'-', digits)) => (true, digits),
            _ => (false, spelling),
        };
        return Ok(signed_float(negative, digits.to_vec(), &specification));
    }
    let negative = value.is_sign_negative();
    let magnitude = value.abs();
    let precision = specification.precision.unwrap_or(6);
    let digits = match specification.conversion {
        b'e' | b'E' => format_exponential(magnitude, precision).into_bytes(),
        b'f' => match fast_fixed_decimal(magnitude, precision) {
            Some(fast) => fast,
            None => format!("{magnitude:.precision$}").into_bytes(),
        },
        b'g' | b'G' => {
            let mut digits = Vec::new();
            push_general(
                &mut digits,
                magnitude,
                precision,
                specification.has_flag(FormatSpecification::ALTERNATE),
            );
            digits
        }
        _ => unreachable!("float conversions are checked by the parser"),
    };
    Ok(signed_float(negative, digits, specification))
}

/// Applies case, sign flags, and width to a float conversion's digits.
fn signed_float(
    negative: bool,
    mut digits: Vec<u8>,
    specification: &FormatSpecification,
) -> Vec<u8> {
    if matches!(specification.conversion, b'E' | b'G') {
        digits.make_ascii_uppercase();
    }
    let mut prefix = Vec::new();
    if negative {
        prefix.push(b'-');
    } else if specification.has_flag(FormatSpecification::PLUS) {
        prefix.push(b'+');
    } else if specification.has_flag(FormatSpecification::SPACE) {
        prefix.push(b' ');
    }
    let prefix_length = prefix.len();
    prefix.extend_from_slice(&digits);
    apply_format_width(prefix, specification, prefix_length)
}

pub(crate) fn number_to_i64(value: f64, name: &'static str) -> RuntimeResult<i64> {
    if !value.is_finite() {
        return Err(RuntimeError::message(format!("{name} must be finite")));
    }
    // Both bounds are exactly representable as f64. The upper bound is
    // exclusive because i64::MAX rounds up to 2^63 in this type.
    if value.fract() != 0.0
        || !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&value)
    {
        return Err(RuntimeError::message(format!("{name} is out of range")));
    }
    #[allow(clippy::cast_possible_truncation)] // Integral and within i64 bounds above.
    Ok(value as i64)
}

pub(crate) fn unsigned_digits(mut value: u64, base: u64, uppercase: bool) -> Vec<u8> {
    let alphabet = if uppercase {
        b"0123456789ABCDEF"
    } else {
        b"0123456789abcdef"
    };
    let mut result = Vec::new();
    loop {
        let index = usize::try_from(value % base).expect("format digit fits usize");
        result.push(alphabet[index]);
        value /= base;
        if value == 0 {
            break;
        }
    }
    result.reverse();
    result
}

/// Formats a finite non-negative `%f` value with integer arithmetic.
///
/// Returns `None` when the value is out of range, deferring to the exact
/// decimal backend. Inside range the scaled product sits far below the
/// precision where double rounding could flip the final digit, so the
/// integer-arithmetic result matches the slow path bit-for-bit while
/// avoiding decimal expansion machinery.
pub(crate) fn fast_fixed_decimal(value: f64, precision: usize) -> Option<Vec<u8>> {
    pub(crate) const ISCALES: [u64; 4] = [1, 10, 100, 1000];
    let scaled = scaled_fixed(value, precision)?;
    let iscale = ISCALES[precision];
    let mut digits = unsigned_digits(scaled / iscale, 10, false);
    if precision > 0 {
        digits.push(b'.');
        let mut fraction = scaled % iscale;
        let mut width = precision;
        let mut tail = Vec::with_capacity(precision);
        while width > 0 {
            tail.push(b'0' + (fraction % 10) as u8);
            fraction /= 10;
            width -= 1;
        }
        tail.reverse();
        digits.extend_from_slice(&tail);
    }
    Some(digits)
}

/// Rounds a non-negative `value * 10^precision` to an integer as printf's
/// `%.<precision>f` would, or returns `None` for the exact decimal backend.
///
/// Below 2^25 the scaled product carries an error under 4e-6, so rounding
/// it matches rounding the exact value unless the product sits within that
/// error of a half. printf breaks exact ties to even and float products can
/// fake or hide ties, so near-halves take the exact path too.
// The NaN-aware comparison is the point.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn scaled_fixed(value: f64, precision: usize) -> Option<u64> {
    const SCALES: [f64; 4] = [1.0, 10.0, 100.0, 1000.0];
    if precision >= SCALES.len() || !(value < 33_554_432.0) {
        return None;
    }
    let scaled = value * SCALES[precision];
    // Below 2^35 and non-negative, so the cast truncates exactly and the
    // fraction is exact; no libm `trunc` or `round` call is needed.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let whole = scaled as u64;
    #[allow(clippy::cast_precision_loss)] // Below 2^35.
    let fraction = scaled - whole as f64;
    if (fraction - 0.5).abs() < 1e-5 {
        return None;
    }
    Some(whole + u64::from(fraction > 0.5))
}

pub(crate) fn format_exponential(value: f64, precision: usize) -> String {
    let formatted = format!("{value:.precision$e}");
    let Some(exponent_index) = formatted.find('e') else {
        return formatted;
    };
    let mantissa = &formatted[..exponent_index];
    let exponent = formatted[exponent_index + 1..]
        .parse::<i32>()
        .expect("Rust exponential output has an integer exponent");
    format!("{mantissa}e{exponent:+03}")
}

/// `%q` of a string. With `legacy` (Lua 5.1 only) it escapes just `"`, `\\`,
/// newline, `\r` and NUL and keeps other control bytes raw. Otherwise (Lua
/// 5.2 and later, `LuaJIT`) every control byte is written as `\ddd`, padded to
/// three digits only when a digit follows, and a carriage return as `\13`.
pub(crate) fn quote_lua_bytes(value: &[u8], legacy: bool) -> Vec<u8> {
    let mut result = Vec::with_capacity(value.len() + 2);
    result.push(b'"');
    for (index, byte) in value.iter().enumerate() {
        match byte {
            b'\\' => result.extend_from_slice(b"\\\\"),
            b'"' => result.extend_from_slice(b"\\\""),
            b'\n' => result.extend_from_slice(b"\\\n"),
            b'\r' if legacy => result.extend_from_slice(b"\\r"),
            b'\0' if legacy => result.extend_from_slice(b"\\000"),
            byte if !legacy && (*byte < b' ' || *byte == 0x7f) => {
                if value.get(index + 1).is_some_and(u8::is_ascii_digit) {
                    result.extend_from_slice(format!("\\{byte:03}").as_bytes());
                } else {
                    result.extend_from_slice(format!("\\{byte}").as_bytes());
                }
            }
            byte => result.push(*byte),
        }
    }
    result.push(b'"');
    result
}

pub(crate) fn apply_format_width(
    mut value: Vec<u8>,
    specification: &FormatSpecification,
    zero_prefix_length: usize,
) -> Vec<u8> {
    let Some(width) = specification.width else {
        return value;
    };
    let padding = width.saturating_sub(value.len());
    if padding == 0 {
        return value;
    }
    if specification.has_flag(FormatSpecification::LEFT) {
        value.extend(std::iter::repeat_n(b' ', padding));
        return value;
    }
    let zero_padding =
        specification.has_flag(FormatSpecification::ZERO) && specification.precision.is_none();
    let fill = if zero_padding { b'0' } else { b' ' };
    let mut padded = Vec::new();
    if zero_padding && zero_prefix_length != 0 {
        padded.extend_from_slice(&value[..zero_prefix_length]);
        padded.extend(std::iter::repeat_n(fill, padding));
        padded.extend_from_slice(&value[zero_prefix_length..]);
    } else {
        padded.extend(std::iter::repeat_n(fill, padding));
        padded.extend_from_slice(&value);
    }
    padded
}

pub(crate) fn append_format_bytes(output: &mut Vec<u8>, bytes: &[u8]) -> RuntimeResult<()> {
    append_bytes_with_limit(output, bytes, "string.format result is too large")
}

pub(crate) fn append_table_concat_bytes(output: &mut Vec<u8>, bytes: &[u8]) -> RuntimeResult<()> {
    append_bytes_with_limit(output, bytes, "table.concat result is too large")
}

pub(crate) fn append_bytes_with_limit(
    output: &mut Vec<u8>,
    bytes: &[u8],
    message: &'static str,
) -> RuntimeResult<()> {
    output
        .try_reserve(bytes.len())
        .map_err(|_| RuntimeError::message(message))?;
    output.extend_from_slice(bytes);
    Ok(())
}

pub(crate) fn string_ascii_case(
    numbers: NumberFormat,
    arguments: &[Value],
    upper: bool,
) -> RuntimeResult<Vec<Value>> {
    let source = string_argument_in(numbers, arguments.first(), 1)?;
    let mut result = source.to_vec();
    for byte in &mut result {
        *byte = if upper {
            byte.to_ascii_uppercase()
        } else {
            byte.to_ascii_lowercase()
        };
    }
    Ok(vec![Value::from(result)])
}

#[cfg(test)]
mod integer_fast_path_tests {
    use super::*;

    #[test]
    fn direct_integer_append_matches_existing_formatter() {
        let values = [
            Value::Integer(0),
            Value::Integer(1),
            Value::Integer(-1),
            Value::Integer(i64::MIN),
            Value::Integer(i64::MAX),
            Value::Number(12345.9),
        ];
        for directive in [
            b"%d".as_slice(),
            b"%06d",
            b"%-8i",
            b"%+08d",
            b"% 8u",
            b"%#08x",
            b"%#8X",
            b"%#08o",
            b"%20u",
        ] {
            let (specification, _) = parse_format_specification(directive, 1, false).unwrap();
            for value in &values {
                let expected =
                    format_value(value, &specification, false, NumberFormat::Lua51).unwrap();
                let mut actual = b"head".to_vec();
                append_format_value(
                    &mut actual,
                    value,
                    &specification,
                    false,
                    NumberFormat::Lua51,
                )
                .unwrap();
                assert_eq!(&actual[4..], expected, "{directive:?}, {value:?}");
            }
        }
    }

    #[test]
    fn number_to_i64_handles_exact_bounds_without_decimal_conversion() {
        assert_eq!(
            number_to_i64(-9_223_372_036_854_775_808.0, "integer").unwrap(),
            i64::MIN
        );
        assert_eq!(
            number_to_i64(9_223_372_036_854_774_784.0, "integer").unwrap(),
            9_223_372_036_854_774_784
        );
        assert_eq!(number_to_i64(-0.0, "integer").unwrap(), 0);
        for invalid in [9_223_372_036_854_775_808.0, f64::INFINITY, f64::NAN, 1.5] {
            assert!(number_to_i64(invalid, "integer").is_err());
        }
    }

    #[test]
    fn float_conversions_round_and_spell_like_printf() {
        let format = |directive: &[u8], value: f64, numbers: NumberFormat| {
            let (specification, _) = parse_format_specification(directive, 1, false).unwrap();
            let value = Value::Number(value);
            let mut appended = Vec::new();
            append_format_value(&mut appended, &value, &specification, false, numbers).unwrap();
            assert_eq!(
                appended,
                format_value(&value, &specification, false, numbers).unwrap()
            );
            String::from_utf8(appended).unwrap()
        };
        let lua51 = NumberFormat::Lua51;
        // Exact binary ties round to even; 1.005 and 0.35 sit below a tie.
        for (directive, value, expected) in [
            (b"%.2f".as_slice(), 0.125, "0.12"),
            (b"%.0f", 0.5, "0"),
            (b"%.0f", 1.5, "2"),
            (b"%.0f", 2.5, "2"),
            (b"%.1f", 0.25, "0.2"),
            (b"%.1f", 0.35, "0.3"),
            (b"%.2f", 1.005, "1.00"),
            (b"%g", 9_007_199_254_740_992.0, "9.0072e+15"),
            (b"%.14g", 9_007_199_254_740_992.0, "9.007199254741e+15"),
            (b"%.3g", 1_234_567.0, "1.23e+06"),
            (b"%G", 1e-20, "1E-20"),
            (b"%5.1f", f64::INFINITY, "  inf"),
            (b"%05g", f64::NEG_INFINITY, " -inf"),
            (b"%+f", f64::INFINITY, "+inf"),
            (b"%E", -f64::NAN, "-NAN"),
        ] {
            assert_eq!(format(directive, value, lua51), expected, "{directive:?}");
        }
        assert_eq!(format(b"%f", -f64::NAN, NumberFormat::LuaJit), "nan");
    }

    #[test]
    fn luajit_nan_format_ignores_plus_and_preserves_space() {
        let format = |directive: &[u8], value: f64, numbers: NumberFormat| {
            let (specification, _) = parse_format_specification(directive, 1, false).unwrap();
            let value = Value::Number(value);
            let expected = format_value(&value, &specification, false, numbers).unwrap();
            let mut appended = Vec::new();
            append_format_value(&mut appended, &value, &specification, false, numbers).unwrap();
            assert_eq!(appended, expected);
            String::from_utf8(expected).unwrap()
        };
        for value in [f64::NAN, -f64::NAN] {
            for (directive, expected) in [
                (b"%+g".as_slice(), "nan"),
                (b"%+ g", " nan"),
                (b"%+010g", "       nan"),
                (b"%+G", "NAN"),
            ] {
                assert_eq!(format(directive, value, NumberFormat::LuaJit), expected);
            }
        }
        for numbers in [
            NumberFormat::Lua51,
            NumberFormat::Lua53,
            NumberFormat::Lua55,
        ] {
            assert_eq!(format(b"%+g", f64::NAN, numbers), "+nan");
            assert_eq!(format(b"%+g", -f64::NAN, numbers), "-nan");
        }
        assert_eq!(format(b"%+g", f64::INFINITY, NumberFormat::LuaJit), "+inf");
        assert_eq!(
            format(b"%+g", f64::NEG_INFINITY, NumberFormat::LuaJit),
            "-inf"
        );
        assert_eq!(format(b"%+g", 1.0, NumberFormat::LuaJit), "+1");
        for (value, plain, signed, padded) in [
            (0.0, "0", "+0", "+000000000"),
            (-0.0, "-0", "-0", "-000000000"),
        ] {
            assert_eq!(format(b"%g", value, NumberFormat::LuaJit), plain);
            assert_eq!(format(b"%+g", value, NumberFormat::LuaJit), signed);
            assert_eq!(format(b"%+ g", value, NumberFormat::LuaJit), signed);
            assert_eq!(format(b"%+010g", value, NumberFormat::LuaJit), padded);
        }
    }

    #[test]
    fn direct_fixed_append_matches_existing_formatter() {
        for directive in [b"%.3f".as_slice(), b"%08.2f", b"%-12.0f", b"%+010.3f"] {
            let (specification, _) = parse_format_specification(directive, 1, false).unwrap();
            for number in [0.0, -0.0, 1.2345, -12.5, 12345.9, 33_554_431.9] {
                let value = Value::Number(number);
                let expected =
                    format_value(&value, &specification, false, NumberFormat::Lua51).unwrap();
                let mut actual = Vec::new();
                append_format_value(
                    &mut actual,
                    &value,
                    &specification,
                    false,
                    NumberFormat::Lua51,
                )
                .unwrap();
                assert_eq!(actual, expected, "{directive:?}, {number}");
            }
        }
    }
}

#[cfg(test)]
mod fast_conversion_tests {
    use super::*;

    /// A fixed-seed generator (splitmix64), so a failure reproduces.
    struct Splitmix(u64);

    impl Splitmix {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
    }

    fn values() -> Vec<Value> {
        let mut values = vec![
            Value::Integer(0),
            Value::Integer(-1),
            Value::Integer(i64::MIN),
            Value::Integer(i64::MAX),
            Value::Int64(-42),
            Value::UInt64(u64::MAX),
            Value::Nil,
            Value::Boolean(true),
        ];
        for number in [
            0.0,
            -0.0,
            0.5,
            -0.5,
            1.5,
            2.5,
            0.0005,
            0.0015,
            0.0025,
            1.0005,
            0.125,
            0.375,
            999.9995,
            1e-320,
            5e-324,
            33_554_431.999,
            33_554_432.0,
            1e15,
            1e300,
            -1e300,
            9_223_372_036_854_775_807.0,
            9_223_372_036_854_775_808.0,
            -9_223_372_036_854_775_808.0,
            -9_223_372_036_854_777_856.0,
            12345.9,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -f64::NAN,
        ] {
            values.push(Value::Number(number));
        }
        let mut random = Splitmix(0x0123_4567_89ab_cdef);
        for round in 0..6000_u64 {
            let bits = random.next();
            values.push(match round % 6 {
                0 => Value::Integer(bits.cast_signed()),
                1 => Value::Integer((bits % 2_000_001).cast_signed() - 1_000_000),
                2 => Value::Number((bits % 4_000_000) as f64 - 2_000_000.0),
                3 => Value::Number(f64::from_bits(bits)),
                // Thousandths and their halves, where rounding flips.
                4 => Value::Number((bits % 40_000_000) as f64 / 2000.0),
                _ => Value::Number((bits % 100_000_000) as f64 / 7.0),
            });
        }
        values
    }

    #[test]
    fn fast_conversions_match_the_general_formatter_on_every_profile() {
        let mut handled = [0_usize; 3];
        let values = values();
        for directive in [
            "%d", "%i", "%x", "%X", "%.0f", "%.1f", "%.2f", "%.3f", "%6d", "%06d", "%-6d", "%-06d",
            "%8x", "%08X", "%-8x", "%10.3f", "%-10.2f", "%7.0f",
        ] {
            let (specification, _) =
                parse_format_specification(directive.as_bytes(), 1, false).unwrap();
            let fast = FastConversion::of(&specification).expect(directive);
            for numbers in [
                NumberFormat::Lua51,
                NumberFormat::LuaJit,
                NumberFormat::Lua53,
                NumberFormat::Lua55,
            ] {
                for value in &values {
                    let mut actual = b"head".to_vec();
                    if !fast.append(&mut actual, value) {
                        assert_eq!(actual, b"head", "{directive} {value:?} wrote on a miss");
                        continue;
                    }
                    let expected = format_value(value, &specification, false, numbers)
                        .unwrap_or_else(|error| {
                            panic!(
                                "{directive} {numbers:?} {value:?}: general path raised {error:?}"
                            )
                        });
                    assert_eq!(
                        &actual[4..],
                        &expected[..],
                        "{directive} {numbers:?} {value:?}"
                    );
                    handled[match specification.conversion {
                        b'f' => 2,
                        b'd' | b'i' => 0,
                        _ => 1,
                    }] += 1;
                }
            }
        }
        // Each family takes most of what it is offered.
        assert!(handled.iter().all(|&count| count > 20_000), "{handled:?}");
    }

    #[test]
    fn fast_conversions_leave_other_forms_to_the_general_path() {
        for directive in [
            "%5.2d", "%+d", "%#x", "% d", "%.3d", "%f", "%010f", "%010.2f", "%.4f", "%.5f",
            "%08.3f", "%s", "%e", "%g", "%c",
        ] {
            let (specification, _) =
                parse_format_specification(directive.as_bytes(), 1, false).unwrap();
            assert!(FastConversion::of(&specification).is_none(), "{directive}");
        }
        let (specification, _) = parse_format_specification(b"%100d", 1, false).unwrap();
        assert!(FastConversion::of(&specification).is_none());
    }
}

#[cfg(test)]
mod short_append_tests {
    use super::{append_short, decimal_length};

    #[test]
    fn decimal_length_counts_digits() {
        assert_eq!(decimal_length(0), 1);
        assert_eq!(decimal_length(u64::MAX), 20);
        for (index, expected) in (1..=20_usize).enumerate() {
            let low = 10_u64.pow(index as u32);
            assert_eq!(decimal_length(low), expected);
            if index > 0 {
                assert_eq!(decimal_length(low - 1), expected - 1);
            }
        }
    }

    #[test]
    fn short_appends_copy_every_length_exactly() {
        let source: Vec<u8> = (b'a'..=b'z').collect();
        for length in 0..=source.len() {
            let mut output = b"head".to_vec();
            append_short(&mut output, &source[..length]).unwrap();
            append_short(&mut output, &source[..length]).unwrap();
            let mut expected = b"head".to_vec();
            expected.extend_from_slice(&source[..length]);
            expected.extend_from_slice(&source[..length]);
            assert_eq!(output, expected, "{length}");
        }
    }
}
