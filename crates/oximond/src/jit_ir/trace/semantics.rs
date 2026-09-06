//! The numeric semantics every trace consumer shares.
//!
//! The reference evaluator, the optimizer's constant folder and the native
//! backend's out-of-line helpers all go through these functions, so a folded
//! constant, an interpreted trace and generated code cannot disagree about
//! NaN, signed zero, floor modulo, power or integer wrap-around.
//!
//! Native code lowers the cheap operations (`+ - * /`, comparisons, negation)
//! to machine instructions with the same IEEE-754 results, and calls
//! [`float_power`] for power. Floor modulo is spelled out in the backend with
//! the exact formula of [`float_modulo`]. Integer division, modulo and shifts
//! are spelled out too, and tested against these functions. Comparing an
//! integer with a float is exact and goes through [`compare`] natively.

#![allow(clippy::float_cmp, reason = "IEEE comparison is the semantics")]

use super::ir::{ArithmeticOp, NumericOp, TraceScalar, TraceType, WideConversion};
use crate::compat::FloatModulo;
use crate::jit_ir::Comparison;
use std::cmp::Ordering;

/// `LuaJIT`'s modulo on floats, `a - floor(a / b) * b` with the product
/// rounded. A zero divisor gives NaN, and infinities follow IEEE rules. The
/// other profiles' forms are [`FloatModulo`].
#[must_use]
pub fn float_modulo(left: f64, right: f64) -> f64 {
    FloatModulo::Floor.apply(left, right)
}

/// Lua's floor division on floats, `floor(a / b)`.
#[must_use]
pub fn float_floor_divide(left: f64, right: f64) -> f64 {
    (left / right).floor()
}

/// Lua's power on floats, C `pow`.
#[must_use]
pub fn float_power(left: f64, right: f64) -> f64 {
    left.powf(right)
}

/// Power on floats in Lua 5.4 and 5.5: `a * a` for an exponent of 2.
#[must_use]
pub fn float_power_square(left: f64, right: f64) -> f64 {
    if right == 2.0 {
        left * left
    } else {
        left.powf(right)
    }
}

/// An integer as a float, rounding to nearest.
#[must_use]
#[allow(clippy::cast_precision_loss, reason = "Lua converts integers this way")]
pub const fn integer_to_float(value: i64) -> f64 {
    value as f64
}

/// Floor division on integers. `MIN // -1` wraps to `MIN`. A zero divisor has
/// no Lua result (the interpreter raises an error before it gets here), and
/// is defined as zero so every consumer agrees on an operation that a guard
/// would have stopped.
#[must_use]
pub fn integer_floor_divide(left: i64, right: i64) -> i64 {
    if right == 0 {
        return 0;
    }
    if right == -1 {
        return left.wrapping_neg();
    }
    let quotient = left / right;
    let remainder = left % right;
    if remainder != 0 && (remainder < 0) != (right < 0) {
        quotient - 1
    } else {
        quotient
    }
}

/// Floor modulo on integers, with the sign of the divisor. `MIN % -1` is
/// zero. A zero divisor is defined as zero, like [`integer_floor_divide`].
#[must_use]
pub fn integer_floor_modulo(left: i64, right: i64) -> i64 {
    if right == 0 || right == -1 {
        return 0;
    }
    let remainder = left % right;
    if remainder != 0 && (remainder < 0) != (right < 0) {
        remainder + right
    } else {
        remainder
    }
}

/// Lua's integer shift. A positive distance shifts left, a negative one
/// shifts right without sign extension, and 64 or more places in either
/// direction clear the value.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the distance is below 64 in magnitude here"
)]
pub const fn integer_shift(value: i64, distance: i64) -> i64 {
    if distance >= 64 || distance <= -64 {
        return 0;
    }
    if distance >= 0 {
        value.wrapping_shl(distance as u32)
    } else {
        value
            .cast_unsigned()
            .wrapping_shr(distance.unsigned_abs() as u32)
            .cast_signed()
    }
}

/// A [`NumericOp`] of one scalar, or `None` when the operand has a type the
/// operation does not take (see [`NumericOp::result_type`]).
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    reason = "the conversion saturates on purpose"
)]
pub fn numeric(op: NumericOp, source: TraceScalar) -> Option<TraceScalar> {
    Some(match (op, source) {
        (NumericOp::Floor, TraceScalar::Float(value)) => TraceScalar::Float(value.floor()),
        (NumericOp::Ceil, TraceScalar::Float(value)) => TraceScalar::Float(value.ceil()),
        (NumericOp::Abs, TraceScalar::Float(value)) => TraceScalar::Float(value.abs()),
        (NumericOp::Abs, TraceScalar::Integer(value)) => TraceScalar::Integer(value.wrapping_abs()),
        (NumericOp::FloatToInteger, TraceScalar::Float(value)) => {
            TraceScalar::Integer(value as i64)
        }
        (NumericOp::TrickBits, TraceScalar::Float(value)) => {
            TraceScalar::Integer(i64::from(trick_bits(value)))
        }
        _ => return None,
    })
}

/// The low word of `number + 1.5 * 2^52`, which is `number` rounded to an
/// integer (ties to even) modulo `2^32` when its magnitude is below `2^51`.
/// This is `LuaJIT`'s x64 operand conversion for the `bit` library.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    reason = "the low word is the result"
)]
pub fn trick_bits(number: f64) -> u32 {
    (number + 6_755_399_441_055_744.0).to_bits() as u32
}

/// The order of an integer and a float by exact value, or `None` when the
/// float is NaN. No conversion rounds: `2^53 + 1` is greater than `2^53` as a
/// float, and every integer is below `2^63` as a float.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "range checks precede the conversion"
)]
pub fn integer_float_order(integer: i64, float: f64) -> Option<Ordering> {
    if float.is_nan() {
        return None;
    }
    if float >= 9_223_372_036_854_775_808.0 {
        return Some(Ordering::Less);
    }
    if float < -9_223_372_036_854_775_808.0 {
        return Some(Ordering::Greater);
    }
    let truncated = float.trunc() as i64;
    match integer.cmp(&truncated) {
        Ordering::Equal if float.fract() > 0.0 => Some(Ordering::Less),
        Ordering::Equal if float.fract() < 0.0 => Some(Ordering::Greater),
        order => Some(order),
    }
}

/// Whether an integer and a float are the same number.
#[must_use]
pub fn integer_float_equal(integer: i64, float: f64) -> bool {
    integer_float_order(integer, float) == Some(Ordering::Equal)
}

/// Binary arithmetic on two scalars of one numeric type. `None` when the
/// operand types or the operation are not legal, which a verified trace never
/// produces. Integer arithmetic wraps. See [`ArithmeticOp`] for the zero
/// divisor convention.
#[must_use]
pub fn arithmetic(op: ArithmeticOp, left: TraceScalar, right: TraceScalar) -> Option<TraceScalar> {
    Some(match (left, right) {
        (TraceScalar::Integer(a), TraceScalar::Integer(b)) => TraceScalar::Integer(match op {
            ArithmeticOp::Add => a.wrapping_add(b),
            ArithmeticOp::Subtract => a.wrapping_sub(b),
            ArithmeticOp::Multiply => a.wrapping_mul(b),
            ArithmeticOp::Modulo => integer_floor_modulo(a, b),
            ArithmeticOp::FloorDivide => integer_floor_divide(a, b),
            ArithmeticOp::BitAnd => a & b,
            ArithmeticOp::BitOr => a | b,
            ArithmeticOp::BitXor => a ^ b,
            ArithmeticOp::ShiftLeft => integer_shift(a, b),
            ArithmeticOp::ShiftRight => integer_shift(a, b.saturating_neg()),
            ArithmeticOp::Divide
            | ArithmeticOp::Power
            | ArithmeticOp::PowerSquare
            | ArithmeticOp::ModuloWith(_)
            | ArithmeticOp::ArithmeticShiftRight
            | ArithmeticOp::RotateLeft
            | ArithmeticOp::RotateRight => return None,
        }),
        (TraceScalar::Int64(a), TraceScalar::Int64(b)) => TraceScalar::Int64(
            wide_arithmetic(op, true, a.cast_unsigned(), b.cast_unsigned())?.cast_signed(),
        ),
        (TraceScalar::UInt64(a), TraceScalar::UInt64(b)) => {
            TraceScalar::UInt64(wide_arithmetic(op, false, a, b)?)
        }
        (TraceScalar::Float(a), TraceScalar::Float(b)) => TraceScalar::Float(match op {
            ArithmeticOp::Add => a + b,
            ArithmeticOp::Subtract => a - b,
            ArithmeticOp::Multiply => a * b,
            ArithmeticOp::Divide => a / b,
            ArithmeticOp::Modulo => float_modulo(a, b),
            ArithmeticOp::ModuloWith(kind) => kind.apply(a, b),
            ArithmeticOp::Power => float_power(a, b),
            ArithmeticOp::PowerSquare => float_power_square(a, b),
            ArithmeticOp::FloorDivide => float_floor_divide(a, b),
            _ => return None,
        }),
        _ => return None,
    })
}

/// Arithmetic on the bits of two 64-bit integer cdata of one signedness
/// (`signed`), as `LuaJIT` does it. `None` for an operation that has no cdata
/// form. See [`ArithmeticOp`] for the rules.
#[must_use]
pub fn wide_arithmetic(op: ArithmeticOp, signed: bool, a: u64, b: u64) -> Option<u64> {
    Some(match op {
        ArithmeticOp::Add => a.wrapping_add(b),
        ArithmeticOp::Subtract => a.wrapping_sub(b),
        ArithmeticOp::Multiply => a.wrapping_mul(b),
        ArithmeticOp::Divide if signed => a
            .cast_signed()
            .checked_div(b.cast_signed())
            .unwrap_or(i64::MIN)
            .cast_unsigned(),
        ArithmeticOp::Divide => a.checked_div(b).unwrap_or(1 << 63),
        ArithmeticOp::Modulo if signed => {
            if b == 0 {
                1 << 63
            } else {
                a.cast_signed()
                    .wrapping_rem(b.cast_signed())
                    .cast_unsigned()
            }
        }
        ArithmeticOp::Modulo => a.checked_rem(b).unwrap_or(1 << 63),
        ArithmeticOp::BitAnd => a & b,
        ArithmeticOp::BitOr => a | b,
        ArithmeticOp::BitXor => a ^ b,
        // The shift methods use the low six bits of the count.
        ArithmeticOp::ShiftLeft => a.wrapping_shl(b as u32),
        ArithmeticOp::ShiftRight => a.wrapping_shr(b as u32),
        ArithmeticOp::ArithmeticShiftRight => {
            a.cast_signed().wrapping_shr(b as u32).cast_unsigned()
        }
        ArithmeticOp::RotateLeft => a.rotate_left((b & 63) as u32),
        ArithmeticOp::RotateRight => a.rotate_right((b & 63) as u32),
        ArithmeticOp::Power
        | ArithmeticOp::PowerSquare
        | ArithmeticOp::ModuloWith(_)
        | ArithmeticOp::FloorDivide => return None,
    })
}

/// The order of two 64-bit integer cdata of one signedness.
#[must_use]
pub fn wide_compare(comparison: Comparison, signed: bool, a: u64, b: u64) -> bool {
    let order = if signed {
        a.cast_signed().cmp(&b.cast_signed())
    } else {
        a.cmp(&b)
    };
    match comparison {
        Comparison::Equal => order == Ordering::Equal,
        Comparison::LessThan => order.is_lt(),
        Comparison::LessEqual => order.is_le(),
    }
}

/// A 64-bit integer cdata as a float, rounding to nearest.
#[must_use]
#[allow(clippy::cast_precision_loss, reason = "Lua numbers are doubles")]
pub fn wide_to_float(bits: u64, signed: bool) -> f64 {
    if signed {
        bits.cast_signed() as f64
    } else {
        bits as f64
    }
}

/// The low 32 bits of `number` rounded to the nearest integer, ties to even,
/// the way the `bit` library reads an operand. Non-finite numbers have no bits
/// set.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    reason = "the low 32 bits are the result"
)]
pub fn number_bits(number: f64) -> u32 {
    if number.abs() < 2_251_799_813_685_248.0 {
        // Adding 1.5 * 2^52 rounds to an integer in the low mantissa bits.
        return (number + 6_755_399_441_055_744.0).to_bits() as u32;
    }
    if !number.is_finite() {
        return 0;
    }
    number.round_ties_even().rem_euclid(4_294_967_296.0) as u32
}

const TWO_POW_63: f64 = 9_223_372_036_854_775_808.0;
const TWO_POW_64: f64 = 18_446_744_073_709_551_616.0;

#[allow(
    clippy::cast_possible_truncation,
    reason = "the range check precedes the cast"
)]
fn float_to_i64(number: f64) -> i64 {
    if (-TWO_POW_63..TWO_POW_63).contains(&number) {
        number as i64
    } else {
        i64::MIN
    }
}

/// Converts a float or 64-bit integer to `target` as `conversion` says.
/// `None` for any combination a verified trace never has.
#[must_use]
pub fn to_wide(
    source: TraceScalar,
    target: TraceType,
    conversion: WideConversion,
) -> Option<TraceScalar> {
    let bits = match (source, conversion) {
        (TraceScalar::Float(number), WideConversion::Truncate) => match target {
            TraceType::Int64 => float_to_i64(number).cast_unsigned(),
            TraceType::UInt64 if number >= TWO_POW_63 => {
                float_to_i64(number - TWO_POW_64).cast_unsigned()
            }
            TraceType::UInt64 => float_to_i64(number).cast_unsigned(),
            _ => return None,
        },
        (TraceScalar::Float(number), WideConversion::Bits32) => u64::from(number_bits(number)),
        (TraceScalar::Int64(value), WideConversion::Truncate) => value.cast_unsigned(),
        (TraceScalar::UInt64(value), WideConversion::Truncate) => value,
        _ => return None,
    };
    match target {
        TraceType::Int64 => Some(TraceScalar::Int64(bits.cast_signed())),
        TraceType::UInt64 => Some(TraceScalar::UInt64(bits)),
        _ => None,
    }
}

/// Negation. Integer negation wraps, float negation flips the sign bit.
#[must_use]
pub fn negate(source: TraceScalar) -> Option<TraceScalar> {
    match source {
        TraceScalar::Integer(value) => Some(TraceScalar::Integer(value.wrapping_neg())),
        TraceScalar::Float(value) => Some(TraceScalar::Float(-value)),
        TraceScalar::Int64(value) => Some(TraceScalar::Int64(value.wrapping_neg())),
        TraceScalar::UInt64(value) => Some(TraceScalar::UInt64(value.wrapping_neg())),
        _ => None,
    }
}

/// The float an integer converts to. `None` for any other scalar.
#[must_use]
pub fn to_float(source: TraceScalar) -> Option<TraceScalar> {
    match source {
        TraceScalar::Integer(value) => Some(TraceScalar::Float(integer_to_float(value))),
        TraceScalar::Int64(value) => Some(TraceScalar::Float(wide_to_float(
            value.cast_unsigned(),
            true,
        ))),
        TraceScalar::UInt64(value) => Some(TraceScalar::Float(wide_to_float(value, false))),
        _ => None,
    }
}

/// Comparison of two scalars of one type, or of an integer with a float,
/// which compares exact values. Float comparisons are ordered, so any
/// comparison with NaN is false. `None` for other mixed types or ordering on
/// a non-number.
#[must_use]
pub fn compare(comparison: Comparison, left: TraceScalar, right: TraceScalar) -> Option<bool> {
    let by_order = |order: Option<Ordering>| match comparison {
        Comparison::Equal => order == Some(Ordering::Equal),
        Comparison::LessThan => order.is_some_and(Ordering::is_lt),
        Comparison::LessEqual => order.is_some_and(Ordering::is_le),
    };
    Some(match (left, right) {
        (TraceScalar::Integer(a), TraceScalar::Integer(b)) => by_order(Some(a.cmp(&b))),
        (TraceScalar::Float(a), TraceScalar::Float(b)) => by_order(a.partial_cmp(&b)),
        (TraceScalar::Integer(a), TraceScalar::Float(b)) => by_order(integer_float_order(a, b)),
        (TraceScalar::Float(a), TraceScalar::Integer(b)) => {
            by_order(integer_float_order(b, a).map(Ordering::reverse))
        }
        (TraceScalar::Int64(a), TraceScalar::Int64(b)) => {
            wide_compare(comparison, true, a.cast_unsigned(), b.cast_unsigned())
        }
        (TraceScalar::UInt64(a), TraceScalar::UInt64(b)) => wide_compare(comparison, false, a, b),
        (TraceScalar::Boolean(a), TraceScalar::Boolean(b)) if comparison == Comparison::Equal => {
            a == b
        }
        (TraceScalar::Nil, TraceScalar::Nil) if comparison == Comparison::Equal => true,
        // Opaque values are the same object exactly when their lanes agree.
        (TraceScalar::Table(a), TraceScalar::Table(b))
        | (TraceScalar::Closure(a), TraceScalar::Closure(b))
            if comparison == Comparison::Equal =>
        {
            a == b
        }
        _ => return None,
    })
}
