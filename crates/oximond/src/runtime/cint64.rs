//! `LuaJIT`'s 64-bit integer cdata (`int64_t` and `uint64_t`).
//!
//! `LuaJIT` boxes these as FFI cdata objects created by `LL`/`ULL` literals
//! and 64-bit arithmetic. Oximond keeps them immediate in [`Value::Int64`]
//! and [`Value::UInt64`], so arithmetic never allocates. The semantics
//! follow `LuaJIT`'s C-like rules: an operation involving at least one 64-bit
//! integer converts both operands to `uint64_t` when either side is
//! unsigned and to `int64_t` otherwise, then wraps modulo 2^64. Numbers
//! convert by truncation; values outside the target range and NaN become
//! `INT64_MIN`, as x86 conversion instructions do.
//!
//! Being immediate, two equal values are one object: `rawequal(1LL, 1LL)`
//! is true and equal values index the same table slot, where LuaJIT
//! compares and keys cdata by identity.

use crate::runtime::Value;

/// A 64-bit integer operand after the usual arithmetic conversions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CInt64 {
    Signed(i64),
    Unsigned(u64),
}

/// The binary arithmetic operators `LuaJIT` defines on 64-bit integers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CIntArith {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Power,
}

const TWO_POW_63: f64 = 9_223_372_036_854_775_808.0;
const TWO_POW_64: f64 = 18_446_744_073_709_551_616.0;

/// Converts a number to `int64_t` by truncation. Out-of-range values and
/// NaN become `INT64_MIN`.
#[must_use]
#[allow(clippy::cast_possible_truncation)] // The range check precedes the cast.
pub fn number_to_i64(number: f64) -> i64 {
    if (-TWO_POW_63..TWO_POW_63).contains(&number) {
        number as i64
    } else {
        i64::MIN
    }
}

/// Converts a number to `uint64_t`. Values from 2^63 wrap modulo 2^64;
/// everything else converts through `int64_t`.
#[must_use]
pub fn number_to_u64(number: f64) -> u64 {
    if number >= TWO_POW_63 {
        number_to_i64(number - TWO_POW_64).cast_unsigned()
    } else {
        number_to_i64(number).cast_unsigned()
    }
}

impl CInt64 {
    /// The 64-bit integer held by `value`, if it is one.
    #[must_use]
    pub const fn from_value(value: &Value) -> Option<Self> {
        match value {
            Value::Int64(value) => Some(Self::Signed(*value)),
            Value::UInt64(value) => Some(Self::Unsigned(*value)),
            _ => None,
        }
    }

    #[must_use]
    pub const fn into_value(self) -> Value {
        match self {
            Self::Signed(value) => Value::Int64(value),
            Self::Unsigned(value) => Value::UInt64(value),
        }
    }

    #[must_use]
    pub const fn is_unsigned(self) -> bool {
        matches!(self, Self::Unsigned(_))
    }

    /// The two's-complement bits.
    #[must_use]
    pub const fn bits(self) -> u64 {
        match self {
            Self::Signed(value) => value.cast_unsigned(),
            Self::Unsigned(value) => value,
        }
    }

    /// Reinterprets `bits` as this operand's type.
    #[must_use]
    pub const fn with_bits(self, bits: u64) -> Self {
        match self {
            Self::Signed(_) => Self::Signed(bits.cast_signed()),
            Self::Unsigned(_) => Self::Unsigned(bits),
        }
    }

    /// The value as a Lua number, as `tonumber` returns it.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // Lua numbers are doubles.
    pub fn to_number(self) -> f64 {
        match self {
            Self::Signed(value) => value as f64,
            Self::Unsigned(value) => value as f64,
        }
    }

    /// `tostring` text: the decimal value with an `LL` or `ULL` suffix.
    #[must_use]
    pub fn to_text(self) -> String {
        match self {
            Self::Signed(value) => format!("{value}LL"),
            Self::Unsigned(value) => format!("{value}ULL"),
        }
    }
}

/// Converts `value` to a 64-bit integer of kind `unsigned`. Only numbers
/// and 64-bit integers convert; strings do not coerce in arithmetic.
fn convert(value: &Value, unsigned: bool) -> Option<CInt64> {
    let bits = match value {
        Value::Int64(value) | Value::Integer(value) => value.cast_unsigned(),
        Value::UInt64(value) => *value,
        Value::Number(number) if unsigned => number_to_u64(*number),
        Value::Number(number) => number_to_i64(*number).cast_unsigned(),
        _ => return None,
    };
    Some(if unsigned {
        CInt64::Unsigned(bits)
    } else {
        CInt64::Signed(bits.cast_signed())
    })
}

/// Converts both operands to their common 64-bit type. `None` unless at
/// least one operand is a 64-bit integer and the other converts.
#[must_use]
#[inline(always)]
pub fn common_operands(left: &Value, right: &Value) -> Option<(CInt64, CInt64)> {
    // One match on the operand pair, so the common cases (two 64-bit
    // integers, or one and a number) skip the generic conversion ladder.
    // Mixed signedness promotes both sides to unsigned.
    Some(match (left, right) {
        (Value::UInt64(a), Value::UInt64(b)) => (CInt64::Unsigned(*a), CInt64::Unsigned(*b)),
        (Value::Int64(a), Value::Int64(b)) => (CInt64::Signed(*a), CInt64::Signed(*b)),
        (Value::UInt64(a), Value::Int64(b)) => {
            (CInt64::Unsigned(*a), CInt64::Unsigned(b.cast_unsigned()))
        }
        (Value::Int64(a), Value::UInt64(b)) => {
            (CInt64::Unsigned(a.cast_unsigned()), CInt64::Unsigned(*b))
        }
        (Value::UInt64(a), Value::Number(b)) => {
            (CInt64::Unsigned(*a), CInt64::Unsigned(number_to_u64(*b)))
        }
        (Value::Number(a), Value::UInt64(b)) => {
            (CInt64::Unsigned(number_to_u64(*a)), CInt64::Unsigned(*b))
        }
        (Value::Int64(a), Value::Number(b)) => {
            (CInt64::Signed(*a), CInt64::Signed(number_to_i64(*b)))
        }
        (Value::Number(a), Value::Int64(b)) => {
            (CInt64::Signed(number_to_i64(*a)), CInt64::Signed(*b))
        }
        _ => return common_operands_general(left, right),
    })
}

fn common_operands_general(left: &Value, right: &Value) -> Option<(CInt64, CInt64)> {
    let left_int = CInt64::from_value(left);
    let right_int = CInt64::from_value(right);
    if left_int.is_none() && right_int.is_none() {
        return None;
    }
    let unsigned =
        left_int.is_some_and(CInt64::is_unsigned) || right_int.is_some_and(CInt64::is_unsigned);
    Some((convert(left, unsigned)?, convert(right, unsigned)?))
}

/// Whether `value` is a 64-bit integer.
#[must_use]
pub const fn is_cint64(value: &Value) -> bool {
    matches!(value, Value::Int64(_) | Value::UInt64(_))
}

/// Performs `operation` when either operand is a 64-bit integer. `None`
/// means the operation does not apply: neither operand is a 64-bit
/// integer, or the other operand is not a number.
#[must_use]
pub fn arith(operation: CIntArith, left: &Value, right: &Value) -> Option<Value> {
    let (left, right) = common_operands(left, right)?;
    arith_converted(operation, left, right)
}

/// [`arith`] on operands already converted by [`common_operands`].
#[must_use]
pub fn arith_converted(operation: CIntArith, left: CInt64, right: CInt64) -> Option<Value> {
    let result = match (left, right) {
        (CInt64::Unsigned(a), CInt64::Unsigned(b)) => {
            CInt64::Unsigned(unsigned_arith(operation, a, b))
        }
        (CInt64::Signed(a), CInt64::Signed(b)) => CInt64::Signed(signed_arith(operation, a, b)),
        _ => unreachable!("common_operands converts both operands to one kind"),
    };
    Some(result.into_value())
}

fn unsigned_arith(operation: CIntArith, a: u64, b: u64) -> u64 {
    match operation {
        CIntArith::Add => a.wrapping_add(b),
        CIntArith::Subtract => a.wrapping_sub(b),
        CIntArith::Multiply => a.wrapping_mul(b),
        CIntArith::Divide => a.checked_div(b).unwrap_or(1 << 63),
        CIntArith::Modulo => a.checked_rem(b).unwrap_or(1 << 63),
        CIntArith::Power => unsigned_power(a, b),
    }
}

fn signed_arith(operation: CIntArith, a: i64, b: i64) -> i64 {
    match operation {
        CIntArith::Add => a.wrapping_add(b),
        CIntArith::Subtract => a.wrapping_sub(b),
        CIntArith::Multiply => a.wrapping_mul(b),
        // Division by zero and the one overflowing quotient give INT64_MIN.
        CIntArith::Divide => a.checked_div(b).unwrap_or(i64::MIN),
        CIntArith::Modulo if b == 0 => i64::MIN,
        CIntArith::Modulo => a.wrapping_rem(b),
        CIntArith::Power => signed_power(a, b),
    }
}

fn unsigned_power(mut base: u64, mut exponent: u64) -> u64 {
    let mut result: u64 = 1;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = result.wrapping_mul(base);
        }
        base = base.wrapping_mul(base);
        exponent >>= 1;
    }
    result
}

fn signed_power(base: i64, exponent: i64) -> i64 {
    if exponent >= 0 {
        return unsigned_power(base.cast_unsigned(), exponent.cast_unsigned()).cast_signed();
    }
    match base {
        0 => i64::MAX,
        1 => 1,
        -1 if exponent & 1 != 0 => -1,
        -1 => 1,
        _ => 0,
    }
}

/// Negates a 64-bit integer with wrapping.
#[must_use]
pub fn negate(value: &Value) -> Option<Value> {
    Some(match CInt64::from_value(value)? {
        CInt64::Signed(value) => Value::Int64(value.wrapping_neg()),
        CInt64::Unsigned(value) => Value::UInt64(value.wrapping_neg()),
    })
}

/// `==` when either operand is a 64-bit integer. `None` when neither is.
/// A 64-bit integer never equals a non-number.
#[must_use]
pub fn equal(left: &Value, right: &Value) -> Option<bool> {
    if !is_cint64(left) && !is_cint64(right) {
        return None;
    }
    Some(common_operands(left, right).is_some_and(|(left, right)| left.bits() == right.bits()))
}

/// `<` (`or_equal` false) or `<=` when either operand is a 64-bit integer.
/// `None` when the comparison does not apply.
#[must_use]
pub fn less(left: &Value, right: &Value, or_equal: bool) -> Option<bool> {
    let (left, right) = common_operands(left, right)?;
    let ordering = match (left, right) {
        (CInt64::Unsigned(a), CInt64::Unsigned(b)) => a.cmp(&b),
        (CInt64::Signed(a), CInt64::Signed(b)) => a.cmp(&b),
        _ => unreachable!("common_operands converts both operands to one kind"),
    };
    Some(if or_equal {
        ordering.is_le()
    } else {
        ordering.is_lt()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_signedness_promotes_to_unsigned() {
        let (a, b) = common_operands(&Value::Int64(-1), &Value::UInt64(1)).unwrap();
        assert_eq!((a, b), (CInt64::Unsigned(u64::MAX), CInt64::Unsigned(1)));
        let (a, b) = common_operands(&Value::UInt64(1), &Value::Int64(-1)).unwrap();
        assert_eq!((a, b), (CInt64::Unsigned(1), CInt64::Unsigned(u64::MAX)));
    }

    #[test]
    fn numbers_convert_to_the_integer_operands_kind() {
        let pair = common_operands(&Value::UInt64(1), &Value::Number(0.5)).unwrap();
        assert_eq!(pair, (CInt64::Unsigned(1), CInt64::Unsigned(0)));
        let pair = common_operands(&Value::Number(-2.7), &Value::Int64(1)).unwrap();
        assert_eq!(pair, (CInt64::Signed(-2), CInt64::Signed(1)));
        assert!(common_operands(&Value::Number(1.0), &Value::Number(2.0)).is_none());
        assert!(common_operands(&Value::Int64(1), &Value::Nil).is_none());
    }

    #[test]
    fn wraparound_and_zero_division() {
        let add = arith(
            CIntArith::Add,
            &Value::UInt64(u64::MAX),
            &Value::Number(1.0),
        );
        assert_eq!(add, Some(Value::UInt64(0)));
        let sub = arith(CIntArith::Subtract, &Value::UInt64(1), &Value::Number(2.0));
        assert_eq!(sub, Some(Value::UInt64(u64::MAX)));
        let mul = arith(CIntArith::Multiply, &Value::Int64(-1), &Value::Number(2.0));
        assert_eq!(mul, Some(Value::Int64(-2)));
        let div = arith(CIntArith::Divide, &Value::UInt64(1), &Value::UInt64(0));
        assert_eq!(div, Some(Value::UInt64(1 << 63)));
        let rem = arith(CIntArith::Modulo, &Value::Int64(1), &Value::Int64(0));
        assert_eq!(rem, Some(Value::Int64(i64::MIN)));
    }

    #[test]
    fn comparisons() {
        assert_eq!(
            less(&Value::UInt64(1), &Value::Number(2.0), false),
            Some(true)
        );
        assert_eq!(equal(&Value::Int64(2), &Value::Int64(2)), Some(true));
        assert_eq!(
            equal(&Value::Int64(-1), &Value::UInt64(u64::MAX)),
            Some(true)
        );
    }
}
