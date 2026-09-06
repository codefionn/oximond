//! Lua arithmetic, numeric coercion, and ordering helpers shared by the interpreter.

use crate::bytecode::{Constant, Instruction, OpCode};
use crate::compat::{CompatibilityProfile, FloatArith};
use crate::runtime::cint64::{self, CIntArith};
use crate::runtime::{
    NonFiniteNumerals, Runtime, RuntimeError, RuntimeResult, Value, parse_lua_integer_exact,
};
use crate::string::LuaString;
use crate::vm::errors::compare_type_error;
use crate::vm::{VmError, VmResult};
use std::sync::OnceLock;

pub(crate) fn usize_to_number(value: usize) -> VmResult<f64> {
    let value = i64::try_from(value)
        .map_err(|_| RuntimeError::message("vararg count exceeds Lua number range"))?;
    Ok(integer_to_number(value))
}

pub(crate) fn arithmetic_instruction(
    registers: &mut [Value],
    instruction: Instruction,
    opcode: OpCode,
    float_arith: FloatArith,
) -> VmResult<()> {
    let destination = usize::from(instruction.a());
    let left = &registers[usize::from(instruction.b())];
    match opcode {
        OpCode::Negate => {
            registers[destination] = Value::Number(-number(left, "perform arithmetic")?);
        }
        OpCode::Not => registers[destination] = Value::Boolean(!left.is_truthy()),
        OpCode::Add
        | OpCode::Subtract
        | OpCode::Multiply
        | OpCode::Divide
        | OpCode::Modulo
        | OpCode::Power => {
            let left = number(left, "perform arithmetic")?;
            let right = number(
                &registers[usize::from(instruction.c())],
                "perform arithmetic",
            )?;
            let result = match opcode {
                OpCode::Add => left + right,
                OpCode::Subtract => left - right,
                OpCode::Multiply => left * right,
                OpCode::Divide => left / right,
                OpCode::Modulo => float_arith.modulo(left, right),
                OpCode::Power => float_arith.power(left, right),
                _ => return Err(VmError::InvalidDispatch(opcode)),
            };
            registers[destination] = Value::Number(result);
        }
        _ => return Err(VmError::InvalidDispatch(opcode)),
    }
    Ok(())
}

/// The register-operand opcode that a constant-operand opcode behaves as.
pub(crate) const fn register_form(opcode: OpCode) -> Option<OpCode> {
    opcode.register_form()
}

pub(crate) fn scalar_metamethod(opcode: OpCode) -> &'static str {
    match opcode {
        OpCode::Add => "__add",
        OpCode::Subtract => "__sub",
        OpCode::Multiply => "__mul",
        OpCode::Divide => "__div",
        OpCode::FloorDivide => "__idiv",
        OpCode::Modulo => "__mod",
        OpCode::Power => "__pow",
        OpCode::BitAnd => "__band",
        OpCode::BitOr => "__bor",
        OpCode::BitXor => "__bxor",
        OpCode::ShiftLeft => "__shl",
        OpCode::ShiftRight => "__shr",
        _ => unreachable!("scalar arithmetic opcode"),
    }
}

/// Metamethod names that operator and index dispatch look up on every use.
const OPERATOR_METAMETHODS: [&str; 24] = [
    "__add",
    "__sub",
    "__mul",
    "__div",
    "__idiv",
    "__mod",
    "__pow",
    "__band",
    "__bor",
    "__bxor",
    "__shl",
    "__shr",
    "__unm",
    "__bnot",
    "__len",
    "__concat",
    "__eq",
    "__lt",
    "__le",
    "__tostring",
    "__pairs",
    "__call",
    "__index",
    "__newindex",
];

/// The numbers of the operator metamethods: positions in
/// `OPERATOR_METAMETHODS`, which dispatch looks keys up by instead of
/// comparing names.
pub(crate) mod event {
    pub(crate) const ADD: u8 = 0;
    pub(crate) const SUB: u8 = 1;
    pub(crate) const MUL: u8 = 2;
    pub(crate) const DIV: u8 = 3;
    pub(crate) const IDIV: u8 = 4;
    pub(crate) const MOD: u8 = 5;
    pub(crate) const POW: u8 = 6;
    pub(crate) const BAND: u8 = 7;
    pub(crate) const BOR: u8 = 8;
    pub(crate) const BXOR: u8 = 9;
    pub(crate) const SHL: u8 = 10;
    pub(crate) const SHR: u8 = 11;
    pub(crate) const UNM: u8 = 12;
    pub(crate) const BNOT: u8 = 13;
    pub(crate) const LEN: u8 = 14;
    pub(crate) const CONCAT: u8 = 15;
    pub(crate) const EQ: u8 = 16;
    pub(crate) const LT: u8 = 17;
    pub(crate) const LE: u8 = 18;
    pub(crate) const INDEX: u8 = 22;
    pub(crate) const NEWINDEX: u8 = 23;
}

const fn same_name(left: &str, right: &str) -> bool {
    let (left, right) = (left.as_bytes(), right.as_bytes());
    if left.len() != right.len() {
        return false;
    }
    let mut index = 0;
    while index < left.len() {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
}

const _: () = {
    let names = [
        (event::ADD, "__add"),
        (event::SUB, "__sub"),
        (event::MUL, "__mul"),
        (event::DIV, "__div"),
        (event::IDIV, "__idiv"),
        (event::MOD, "__mod"),
        (event::POW, "__pow"),
        (event::BAND, "__band"),
        (event::BOR, "__bor"),
        (event::BXOR, "__bxor"),
        (event::SHL, "__shl"),
        (event::SHR, "__shr"),
        (event::UNM, "__unm"),
        (event::BNOT, "__bnot"),
        (event::LEN, "__len"),
        (event::CONCAT, "__concat"),
        (event::EQ, "__eq"),
        (event::LT, "__lt"),
        (event::LE, "__le"),
        (event::INDEX, "__index"),
        (event::NEWINDEX, "__newindex"),
    ];
    let mut index = 0;
    while index < names.len() {
        assert!(same_name(
            OPERATOR_METAMETHODS[names[index].0 as usize],
            names[index].1
        ));
        index += 1;
    }
};

/// The event of an arithmetic or bitwise opcode: its position in
/// `OPERATOR_METAMETHODS`, as [`scalar_metamethod`] names it.
pub(crate) fn scalar_event(opcode: OpCode) -> u8 {
    match opcode {
        OpCode::Add => event::ADD,
        OpCode::Subtract => event::SUB,
        OpCode::Multiply => event::MUL,
        OpCode::Divide => event::DIV,
        OpCode::FloorDivide => event::IDIV,
        OpCode::Modulo => event::MOD,
        OpCode::Power => event::POW,
        OpCode::BitAnd => event::BAND,
        OpCode::BitOr => event::BOR,
        OpCode::BitXor => event::BXOR,
        OpCode::ShiftLeft => event::SHL,
        OpCode::ShiftRight => event::SHR,
        _ => unreachable!("scalar arithmetic opcode"),
    }
}

/// The prebuilt key of the operator metamethod numbered `event`.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn event_key(event: u8) -> &'static LuaString {
    &operator_metamethod_keys()[usize::from(event)]
}

/// The table key for an operator metamethod `name`, built once so that
/// dispatch does not allocate a key string per operation. `None` for other
/// names.
pub(crate) fn metamethod_key(name: &str) -> Option<&'static LuaString> {
    let index = OPERATOR_METAMETHODS
        .iter()
        .position(|candidate| *candidate == name)?;
    Some(&operator_metamethod_keys()[index])
}

/// The number of the operator metamethod `name` among the names the
/// trace's operator check (`HelperId::OPERATOR_HANDLER`) takes.
pub(crate) fn operator_event(name: &str) -> Option<u8> {
    let index = OPERATOR_METAMETHODS
        .iter()
        .position(|candidate| *candidate == name)?;
    u8::try_from(index).ok()
}

/// The key of the operator metamethod numbered `event` (see
/// [`operator_event`]).
pub(crate) fn operator_event_key(event: u8) -> Option<&'static LuaString> {
    operator_metamethod_keys().get(usize::from(event))
}

/// The prebuilt `__newindex` key: the last entry of `OPERATOR_METAMETHODS`.
pub(crate) fn newindex_metamethod_key() -> &'static LuaString {
    let last = OPERATOR_METAMETHODS.len() - 1;
    debug_assert_eq!(OPERATOR_METAMETHODS[last], "__newindex");
    &operator_metamethod_keys()[last]
}

/// The prebuilt `__index` key: the entry before `__newindex`.
pub(crate) fn index_metamethod_key() -> &'static LuaString {
    let index = OPERATOR_METAMETHODS.len() - 2;
    debug_assert_eq!(OPERATOR_METAMETHODS[index], "__index");
    &operator_metamethod_keys()[index]
}

/// The prebuilt `__call` key, looked up by name in `OPERATOR_METAMETHODS`.
pub(crate) fn call_metamethod_key() -> &'static LuaString {
    let index = OPERATOR_METAMETHODS
        .iter()
        .position(|candidate| *candidate == "__call")
        .expect("__call is an operator metamethod");
    &operator_metamethod_keys()[index]
}

fn operator_metamethod_keys() -> &'static [LuaString; OPERATOR_METAMETHODS.len()] {
    static KEYS: OnceLock<[LuaString; OPERATOR_METAMETHODS.len()]> = OnceLock::new();
    KEYS.get_or_init(|| OPERATOR_METAMETHODS.map(|name| crate::string::shared_key(name.as_bytes())))
}

/// The metamethod key of an arithmetic or bitwise opcode.
pub(crate) fn operator_metamethod_key(opcode: OpCode) -> Option<&'static LuaString> {
    // Positions in `OPERATOR_METAMETHODS`.
    let index = match opcode {
        OpCode::Add => 0,
        OpCode::Subtract => 1,
        OpCode::Multiply => 2,
        OpCode::Divide => 3,
        OpCode::FloorDivide => 4,
        OpCode::Modulo => 5,
        OpCode::Power => 6,
        OpCode::BitAnd => 7,
        OpCode::BitOr => 8,
        OpCode::BitXor => 9,
        OpCode::ShiftLeft => 10,
        OpCode::ShiftRight => 11,
        OpCode::Negate => 12,
        _ => return None,
    };
    Some(&operator_metamethod_keys()[index])
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Range and integrality checks precede the f64-to-i64 conversion.
pub(crate) fn coerce_lua_integer(value: &Value) -> Result<i64, RuntimeError> {
    pub(crate) const I64_EXCLUSIVE_UPPER: f64 = 9_223_372_036_854_775_808.0;
    match value {
        Value::Integer(value) => return Ok(*value),
        Value::String(text) => {
            if let Some(value) = parse_lua_integer_exact(text.as_bytes()) {
                return Ok(value);
            }
        }
        _ => {}
    }
    let number = Runtime::coerce_number(value, "perform bitwise operation")?;
    if !number.is_finite()
        || number.fract() != 0.0
        || number < i64::MIN as f64
        || number >= I64_EXCLUSIVE_UPPER
    {
        return Err(RuntimeError::message(
            "number has no integer representation",
        ));
    }
    Ok(number as i64)
}

pub(crate) fn bitwise_result(opcode: OpCode, left: i64, right: i64) -> i64 {
    match opcode {
        OpCode::BitAnd => left & right,
        OpCode::BitOr => left | right,
        OpCode::BitXor => left ^ right,
        OpCode::ShiftLeft => lua_shift(left, right),
        OpCode::ShiftRight => lua_shift(left, right.saturating_neg()),
        _ => unreachable!("bitwise binary opcode"),
    }
}

pub(crate) fn lua_shift(value: i64, distance: i64) -> i64 {
    if distance >= i64::from(i64::BITS) || distance <= -i64::from(i64::BITS) {
        return 0;
    }
    if distance >= 0 {
        value.wrapping_shl(
            u32::try_from(distance).expect("non-negative shift distance below i64::BITS"),
        )
    } else {
        value
            .cast_unsigned()
            .wrapping_shr(
                u32::try_from(distance.unsigned_abs())
                    .expect("negative shift distance magnitude below i64::BITS"),
            )
            .cast_signed()
    }
}

#[allow(clippy::float_cmp)] // Lua number equality is exact, including NaN and signed zero rules.
#[inline]
pub(crate) fn numeric_scalar_result(
    opcode: OpCode,
    left: &Value,
    right: &Value,
    float_arith: FloatArith,
) -> VmResult<Option<Value>> {
    if opcode == OpCode::Negate
        && let Value::Integer(value) = left
    {
        return Ok(Some(Value::Integer(value.wrapping_neg())));
    }
    if opcode == OpCode::BitNot
        && matches!(left, Value::Integer(_) | Value::Number(_))
        && let Ok(value) = coerce_lua_integer(left)
    {
        return Ok(Some(Value::Integer(!value)));
    }
    if let (Value::Integer(left), Value::Integer(right)) = (left, right) {
        let value = match opcode {
            OpCode::Add => Value::Integer(left.wrapping_add(*right)),
            OpCode::Subtract => Value::Integer(left.wrapping_sub(*right)),
            OpCode::Multiply => Value::Integer(left.wrapping_mul(*right)),
            OpCode::Divide => Value::Number(integer_to_float(*left) / integer_to_float(*right)),
            OpCode::FloorDivide => Value::Integer(integer_floor_div(*left, *right)?),
            OpCode::Modulo => Value::Integer(integer_floor_mod(*left, *right)?),
            OpCode::Power => {
                Value::Number(float_arith.power(integer_to_float(*left), integer_to_float(*right)))
            }
            OpCode::BitAnd
            | OpCode::BitOr
            | OpCode::BitXor
            | OpCode::ShiftLeft
            | OpCode::ShiftRight => Value::Integer(bitwise_result(opcode, *left, *right)),
            OpCode::Equal => Value::Boolean(left == right),
            OpCode::LessThan => Value::Boolean(left < right),
            OpCode::LessEqual => Value::Boolean(left <= right),
            _ => return Ok(None),
        };
        return Ok(Some(value));
    }
    if matches!(
        (left, right),
        (
            Value::Integer(_) | Value::Number(_),
            Value::Integer(_) | Value::Number(_)
        )
    ) {
        let value = match opcode {
            OpCode::Add => Value::Number(
                Runtime::coerce_number(left, "perform arithmetic")?
                    + Runtime::coerce_number(right, "perform arithmetic")?,
            ),
            OpCode::Subtract => Value::Number(
                Runtime::coerce_number(left, "perform arithmetic")?
                    - Runtime::coerce_number(right, "perform arithmetic")?,
            ),
            OpCode::Multiply => Value::Number(
                Runtime::coerce_number(left, "perform arithmetic")?
                    * Runtime::coerce_number(right, "perform arithmetic")?,
            ),
            OpCode::Divide => Value::Number(
                Runtime::coerce_number(left, "perform arithmetic")?
                    / Runtime::coerce_number(right, "perform arithmetic")?,
            ),
            OpCode::FloorDivide => Value::Number(
                (Runtime::coerce_number(left, "perform arithmetic")?
                    / Runtime::coerce_number(right, "perform arithmetic")?)
                .floor(),
            ),
            OpCode::Modulo => {
                let left = Runtime::coerce_number(left, "perform arithmetic")?;
                let right = Runtime::coerce_number(right, "perform arithmetic")?;
                Value::Number(float_arith.modulo(left, right))
            }
            OpCode::Power => Value::Number(float_arith.power(
                Runtime::coerce_number(left, "perform arithmetic")?,
                Runtime::coerce_number(right, "perform arithmetic")?,
            )),
            OpCode::BitAnd
            | OpCode::BitOr
            | OpCode::BitXor
            | OpCode::ShiftLeft
            | OpCode::ShiftRight => {
                let (Ok(left), Ok(right)) = (coerce_lua_integer(left), coerce_lua_integer(right))
                else {
                    return Ok(None);
                };
                Value::Integer(bitwise_result(opcode, left, right))
            }
            OpCode::Equal => Value::Boolean(Runtime::raw_equal(left, right)),
            OpCode::LessThan => {
                Value::Boolean(numeric_order(left, right).is_some_and(std::cmp::Ordering::is_lt))
            }
            OpCode::LessEqual => {
                Value::Boolean(numeric_order(left, right).is_some_and(std::cmp::Ordering::is_le))
            }
            _ => return Ok(None),
        };
        return Ok(Some(value));
    }
    if cint64::is_cint64(left) || cint64::is_cint64(right) {
        return Ok(cint64_scalar_result(opcode, left, right));
    }
    Ok(None)
}

/// `LuaJIT`'s 64-bit integer cdata arithmetic and comparisons. `None` leaves
/// the operation to the general path, which raises the type error.
#[cold]
fn cint64_scalar_result(opcode: OpCode, left: &Value, right: &Value) -> Option<Value> {
    let operation = match opcode {
        OpCode::Negate => return cint64::negate(left),
        OpCode::Equal => return cint64::equal(left, right).map(Value::Boolean),
        OpCode::LessThan => return cint64::less(left, right, false).map(Value::Boolean),
        OpCode::LessEqual => return cint64::less(left, right, true).map(Value::Boolean),
        OpCode::Add => CIntArith::Add,
        OpCode::Subtract => CIntArith::Subtract,
        OpCode::Multiply => CIntArith::Multiply,
        OpCode::Divide => CIntArith::Divide,
        OpCode::Modulo => CIntArith::Modulo,
        OpCode::Power => CIntArith::Power,
        _ => return None,
    };
    cint64::arith(operation, left, right)
}

pub(crate) fn integer_floor_div(left: i64, right: i64) -> RuntimeResult<i64> {
    if right == 0 {
        return Err(RuntimeError::message("attempt to divide by zero"));
    }
    if left == i64::MIN && right == -1 {
        return Ok(i64::MIN);
    }
    let quotient = left / right;
    let remainder = left % right;
    Ok(if remainder != 0 && (remainder < 0) != (right < 0) {
        quotient - 1
    } else {
        quotient
    })
}

pub(crate) fn integer_floor_mod(left: i64, right: i64) -> RuntimeResult<i64> {
    integer_floor_mod_nonzero(left, right)
        .ok_or_else(|| RuntimeError::message("attempt to perform 'n%0'"))
}

/// Lua 5.4 integer floor modulo; `None` for a zero divisor.
#[inline(always)]
pub(crate) fn integer_floor_mod_nonzero(left: i64, right: i64) -> Option<i64> {
    // Two operands in `0 .. 2^32`, the common case: one unsigned 32-bit
    // division, whose remainder needs no floor correction, and no test for
    // a divisor of -1.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    if (left | right) as u64 >> 32 == 0 {
        if right == 0 {
            return None;
        }
        return Some(i64::from(left as u32 % right as u32));
    }
    if right == 0 {
        return None;
    }
    if right == -1 {
        // Avoids the i64::MIN % -1 overflow; the result is 0 for every left.
        return Some(0);
    }
    let remainder = left % right;
    Some(if remainder != 0 && (remainder < 0) != (right < 0) {
        remainder + right
    } else {
        remainder
    })
}

#[inline(always)]
pub(crate) fn numeric_order(left: &Value, right: &Value) -> Option<std::cmp::Ordering> {
    match (left, right) {
        (Value::Integer(left), Value::Integer(right)) => Some(left.cmp(right)),
        (Value::Number(left), Value::Number(right)) => left.partial_cmp(right),
        (Value::Integer(left), Value::Number(right)) => integer_float_order(*left, *right),
        (Value::Number(left), Value::Integer(right)) => {
            integer_float_order(*right, *left).map(std::cmp::Ordering::reverse)
        }
        _ => None,
    }
}

#[allow(clippy::cast_possible_truncation)] // Bounds checks precede the integral conversion.
#[inline(always)]
pub(crate) fn integer_float_order(integer: i64, float: f64) -> Option<std::cmp::Ordering> {
    if float.is_nan() {
        return None;
    }
    if float >= 9_223_372_036_854_775_808.0 {
        return Some(std::cmp::Ordering::Less);
    }
    if float < -9_223_372_036_854_775_808.0 {
        return Some(std::cmp::Ordering::Greater);
    }
    let truncated = float.trunc() as i64;
    match integer.cmp(&truncated) {
        std::cmp::Ordering::Equal if float.fract() > 0.0 => Some(std::cmp::Ordering::Less),
        std::cmp::Ordering::Equal if float.fract() < 0.0 => Some(std::cmp::Ordering::Greater),
        order => Some(order),
    }
}

pub(crate) const fn scalar_string_like(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::Integer(_) | Value::Number(_)
    )
}

pub(crate) fn constant_value(value: &Constant, compatibility: CompatibilityProfile) -> Value {
    match value {
        Constant::Nil => Value::Nil,
        Constant::Boolean(value) => Value::Boolean(*value),
        Constant::Integer(value)
            if matches!(
                compatibility,
                CompatibilityProfile::Lua53
                    | CompatibilityProfile::Lua54
                    | CompatibilityProfile::Lua55
            ) =>
        {
            Value::Integer(*value)
        }
        Constant::Integer(value) => Value::Number(integer_to_number(*value)),
        Constant::Number(value) => Value::Number(*value),
        Constant::String(value) => Value::from(value.clone()),
        Constant::Int64(value) => Value::Int64(*value),
        Constant::UInt64(value) => Value::UInt64(*value),
    }
}

pub(crate) fn integer_to_number(value: i64) -> f64 {
    value as f64
}

/// Returns a count or string position in the profile's integer representation.
pub(crate) fn profile_usize_value(
    compatibility: CompatibilityProfile,
    value: usize,
) -> VmResult<Value> {
    let value = i64::try_from(value)
        .map_err(|_| RuntimeError::message("count exceeds Lua integer range"))?;
    Ok(profile_integer_value(compatibility, value))
}

pub(crate) fn profile_integer_value(compatibility: CompatibilityProfile, value: i64) -> Value {
    if compatibility.is_lua53_or_later() {
        Value::Integer(value)
    } else {
        Value::Number(integer_to_number(value))
    }
}

#[allow(clippy::cast_possible_truncation)] // Runtime table borders are non-negative integral f64 values.
pub(crate) fn table_length_integer(value: f64) -> i64 {
    value as i64
}

#[allow(clippy::cast_precision_loss)] // Lua arithmetic converts tagged integers to IEEE-754 here.
pub(crate) const fn integer_to_float(value: i64) -> f64 {
    value as f64
}

pub(crate) fn number(value: &Value, operation: &'static str) -> RuntimeResult<f64> {
    Runtime::coerce_number(value, operation)
}

/// The spellings of infinity and NaN that a profile's string-to-number
/// coercion accepts.
pub(crate) const fn nonfinite_numerals(compatibility: CompatibilityProfile) -> NonFiniteNumerals {
    match compatibility {
        CompatibilityProfile::Lua51 => NonFiniteNumerals::Strtod,
        CompatibilityProfile::LuaJit21 => NonFiniteNumerals::LuaJit,
        CompatibilityProfile::Lua52
        | CompatibilityProfile::Lua53
        | CompatibilityProfile::Lua54
        | CompatibilityProfile::Lua55 => NonFiniteNumerals::Rejected,
    }
}

pub(crate) fn strict_number(value: &Value, operation: &'static str) -> RuntimeResult<f64> {
    match value {
        Value::Number(value) => Ok(*value),
        value => Err(RuntimeError::TypeError {
            operation,
            expected: "number",
            found: value.type_name(),
        }),
    }
}

#[inline(always)]
pub(crate) fn ordered_compare(
    left: &Value,
    right: &Value,
    operation: &'static str,
) -> RuntimeResult<Option<std::cmp::Ordering>> {
    match (left, right) {
        (Value::Integer(_) | Value::Number(_), Value::Integer(_) | Value::Number(_)) => {
            Ok(numeric_order(left, right))
        }
        (left, right) if let Some((left, right)) = cint64::common_operands(left, right) => {
            Ok(Some(if left.is_unsigned() {
                left.bits().cmp(&right.bits())
            } else {
                (left.bits().cast_signed()).cmp(&right.bits().cast_signed())
            }))
        }
        (Value::String(left), Value::String(right)) => Ok(Some(left.cmp(right))),
        (left, right) => Err(compare_type_error(operation, left, right)),
    }
}
