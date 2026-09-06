//! Lua 5.4's string metatable arithmetic metamethods.
//!
//! Lua 5.4 no longer converts strings in the VM's arithmetic. The string
//! metatable's `__add` and friends convert both operands the way lstrlib's
//! `tonum` does and run the operation, or raise lstrlib's `trymt` error.
//! `trymt` also passes the operation to a second operand that has its own
//! metamethod; the VM does that before calling these natives, which it
//! recognizes by their [`oximond::runtime::DirectNativeKind`].

use crate::compat::CompatibilityProfile;
use crate::runtime::{Runtime, RuntimeError, RuntimeResult, Value};

#[derive(Clone, Copy)]
enum Operation {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Unm,
    FloorDiv,
}

impl Operation {
    /// The metamethod name without its `__`, as `trymt` spells it.
    const fn event(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Mod => "mod",
            Self::Pow => "pow",
            Self::Unm => "unm",
            Self::FloorDiv => "idiv",
        }
    }
}

pub(crate) fn string_add(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    arith(arguments, Operation::Add)
}

pub(crate) fn string_sub(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    arith(arguments, Operation::Sub)
}

pub(crate) fn string_mul(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    arith(arguments, Operation::Mul)
}

pub(crate) fn string_div(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    arith(arguments, Operation::Div)
}

pub(crate) fn string_mod(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    arith(arguments, Operation::Mod)
}

pub(crate) fn string_pow(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    arith(arguments, Operation::Pow)
}

pub(crate) fn string_unm(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    arith(arguments, Operation::Unm)
}

pub(crate) fn string_idiv(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    arith(arguments, Operation::FloorDiv)
}

fn arith(arguments: &[Value], operation: Operation) -> RuntimeResult<Value> {
    let left = arguments.first().unwrap_or(&Value::Nil);
    let right = arguments.get(1).unwrap_or(&Value::Nil);
    let left_number = to_number(left);
    // `tonum` pushes the converted first operand, so with a single argument
    // the second `tonum` reads that number back.
    let right_number = if arguments.len() == 1 {
        left_number.clone()
    } else {
        to_number(right)
    };
    let (Some(left_number), Some(right_number)) = (left_number, right_number) else {
        return Err(RuntimeError::message(format!(
            "attempt to {} a '{}' with a '{}'",
            operation.event(),
            left.type_name(),
            right.type_name()
        )));
    };
    // `lua_arith` raises from a C function, so its errors carry no position.
    compute(operation, &left_number, &right_number)
        .map_err(|message| RuntimeError::Raised(Value::from(message)))
}

/// lstrlib's `tonum`: a number, or a string that is a whole numeral.
fn to_number(value: &Value) -> Option<Value> {
    match value {
        Value::Integer(_) | Value::Number(_) => Some(value.clone()),
        Value::String(_) => Runtime::coerce_numeric_value(value, "perform arithmetic on").ok(),
        _ => None,
    }
}

#[allow(clippy::cast_precision_loss)] // Lua converts integers to floats here.
const fn float(value: &Value) -> f64 {
    match value {
        Value::Integer(value) => *value as f64,
        Value::Number(value) => *value,
        _ => 0.0,
    }
}

fn compute(operation: Operation, left: &Value, right: &Value) -> Result<Value, &'static str> {
    if let (Value::Integer(left), Value::Integer(right)) = (left, right) {
        let (left, right) = (*left, *right);
        match operation {
            Operation::Add => return Ok(Value::Integer(left.wrapping_add(right))),
            Operation::Sub => return Ok(Value::Integer(left.wrapping_sub(right))),
            Operation::Mul => return Ok(Value::Integer(left.wrapping_mul(right))),
            Operation::Unm => return Ok(Value::Integer(left.wrapping_neg())),
            Operation::Mod => return integer_mod(left, right).map(Value::Integer),
            Operation::FloorDiv => return integer_floor_div(left, right).map(Value::Integer),
            Operation::Div | Operation::Pow => {}
        }
    }
    if let Operation::Unm = operation
        && let Value::Integer(value) = left
    {
        return Ok(Value::Integer(value.wrapping_neg()));
    }
    let (left, right) = (float(left), float(right));
    Ok(Value::Number(match operation {
        Operation::Add => left + right,
        Operation::Sub => left - right,
        Operation::Mul => left * right,
        Operation::Div => left / right,
        Operation::Pow => CompatibilityProfile::Lua54.float_arith().power(left, right),
        Operation::Unm => -left,
        Operation::FloorDiv => (left / right).floor(),
        // `luai_nummod` and `luai_numpow` of Lua 5.4 and 5.5, which install these.
        Operation::Mod => CompatibilityProfile::Lua54
            .float_arith()
            .modulo(left, right),
    }))
}

/// `luaV_mod`.
const fn integer_mod(left: i64, right: i64) -> Result<i64, &'static str> {
    match right {
        0 => Err("attempt to perform 'n%0'"),
        -1 => Ok(0),
        _ => {
            let remainder = left % right;
            Ok(if remainder != 0 && (remainder ^ right) < 0 {
                remainder + right
            } else {
                remainder
            })
        }
    }
}

/// `luaV_idiv`.
const fn integer_floor_div(left: i64, right: i64) -> Result<i64, &'static str> {
    match right {
        0 => Err("attempt to divide by zero"),
        -1 => Ok(left.wrapping_neg()),
        _ => {
            let quotient = left / right;
            Ok(if (left ^ right) < 0 && left % right != 0 {
                quotient - 1
            } else {
                quotient
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn number(operation: Operation, left: f64, right: f64) -> f64 {
        match compute(operation, &Value::Number(left), &Value::Number(right)) {
            Ok(Value::Number(value)) => value,
            other => panic!("expected a float, got {other:?}"),
        }
    }

    /// Lua 5.4's `a ^ 2` is `a * a`, and its `%` is `fmod` with the sign of
    /// the divisor; these are the values `lua5.4` prints.
    #[test]
    fn string_metamethods_use_the_lua54_float_forms() {
        let cube = 483.0_f64.powf(3.0);
        assert_eq!(number(Operation::Pow, cube, 2.0), 12_696_463_968_316_568.0);
        assert_eq!(number(Operation::Pow, 2.0, 0.5), 2.0_f64.sqrt());
        assert_eq!(
            number(Operation::Mod, -583.0, 3.001),
            2.194_999_999_999_978_5
        );
        assert_eq!(
            number(Operation::Mod, 1e10, 4.0 / -2.25),
            -1.777_777_222_666_265_4
        );
        assert_eq!(number(Operation::Mod, -5.0, f64::INFINITY), f64::INFINITY);
        assert_eq!(number(Operation::Mod, -5.0, f64::NEG_INFINITY), -5.0);
        assert_eq!(
            number(Operation::Mod, -0.0, 3.0).to_bits(),
            (-0.0_f64).to_bits()
        );
    }
}
