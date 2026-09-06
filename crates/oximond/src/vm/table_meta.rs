//! The `table` library functions that follow metamethods in Lua 5.3 and
//! later: `concat`, `insert`, `remove`, `move` and `unpack` read and write
//! their elements with `lua_geti` and `lua_seti`, and take the length with
//! `luaL_len`, so `__index`, `__newindex` and `__len` apply.
//!
//! The natives in `oximond::stdlib` keep the raw path and decline, with
//! `RuntimeError::Redispatch`, a table whose metatable has one of those
//! metamethods. The VM then runs the call here, as a host callback, so the
//! metamethods are Lua functions run from a boundary like stock Lua's C one.

use crate::compat::CompatibilityProfile;
use crate::runtime::{DirectNativeKind, Runtime, RuntimeError, Value};
use crate::vm::arith::profile_integer_value;
use crate::vm::{Vm, VmResult};

/// Stock's `LUAI_MAXSTACK`, which bounds the results of `table.unpack`.
const MAX_UNPACK_RESULTS: u64 = 1_000_000;

pub(crate) fn token(kind: DirectNativeKind) -> Option<u64> {
    matches!(
        kind,
        DirectNativeKind::TableConcat
            | DirectNativeKind::TableInsert
            | DirectNativeKind::TableRemove
            | DirectNativeKind::TableUnpack
            | DirectNativeKind::TableMove
            | DirectNativeKind::Print
            | DirectNativeKind::StringFormatLuaJit
    )
    .then_some(kind as u64)
}

pub(crate) fn callback(vm: &mut Vm, token: u64, arguments: &[Value]) -> VmResult<Vec<Value>> {
    if token == DirectNativeKind::TableConcat as u64 {
        vm.table_concat_meta(arguments)
    } else if token == DirectNativeKind::TableInsert as u64 {
        vm.table_insert_meta(arguments)
    } else if token == DirectNativeKind::TableRemove as u64 {
        vm.table_remove_meta(arguments)
    } else if token == DirectNativeKind::TableUnpack as u64 {
        vm.table_unpack_meta(arguments)
    } else if token == DirectNativeKind::Print as u64 {
        vm.print_meta(arguments)
    } else if token == DirectNativeKind::StringFormatLuaJit as u64 {
        vm.string_format_luajit_meta(arguments)
    } else {
        vm.table_move_meta(arguments)
    }
}

/// `luaL_checkinteger` for argument `position` (counted from 1).
pub(crate) fn check_integer(arguments: &[Value], position: usize) -> Result<i64, RuntimeError> {
    let argument = arguments.get(position - 1);
    let number = match argument {
        Some(Value::Integer(value)) => return Ok(*value),
        Some(value @ (Value::Number(_) | Value::String(_))) => {
            Runtime::coerce_numeric_value(value, "").ok()
        }
        _ => None,
    };
    match number {
        Some(Value::Integer(value)) => Ok(value),
        Some(Value::Number(value)) => {
            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
            if value.fract() == 0.0
                && (-9.223_372_036_854_776e18..9.223_372_036_854_776e18).contains(&value)
            {
                Ok(value as i64)
            } else {
                Err(RuntimeError::argument_error(
                    position,
                    "number has no integer representation",
                ))
            }
        }
        _ => Err(RuntimeError::bad_argument(position, "number", argument)),
    }
}

/// `luaL_optinteger`.
fn optional_integer(
    arguments: &[Value],
    position: usize,
    default: i64,
) -> Result<i64, RuntimeError> {
    match arguments.get(position - 1) {
        None | Some(Value::Nil) => Ok(default),
        Some(_) => check_integer(arguments, position),
    }
}

impl Vm {
    pub(crate) fn element(&mut self, table: &Value, index: i64) -> VmResult<Value> {
        let key = profile_integer_value(self.compatibility, index);
        self.get_value(table.clone(), key)
    }

    pub(crate) fn set_element(&mut self, table: &Value, index: i64, value: Value) -> VmResult<()> {
        let key = profile_integer_value(self.compatibility, index);
        self.set_value(table.clone(), key, value)
    }

    /// `luaL_len`: the `#` operator, which must give an integer.
    pub(crate) fn length_with_metamethod(&mut self, table: &Value) -> VmResult<i64> {
        if let Some(handler) = self.unary_metamethod(table, crate::vm::arith::event::LEN)? {
            let result = self
                .call_value(handler, vec![table.clone()])?
                .into_iter()
                .next()
                .unwrap_or(Value::Nil);
            let integer = match &result {
                Value::Integer(value) => Some(*value),
                Value::Number(_) | Value::String(_) => {
                    check_integer(std::slice::from_ref(&result), 1).ok()
                }
                _ => None,
            };
            return integer
                .ok_or_else(|| RuntimeError::message("object length is not an integer").into());
        }
        let length = self.runtime.raw_length(table)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(length as i64)
    }

    /// `LuaJIT`'s `string.format`, whose `%s` calls `__tostring` and formats
    /// what it returned like any other `%s` argument.
    fn string_format_luajit_meta(&mut self, arguments: &[Value]) -> VmResult<Vec<Value>> {
        let mut arguments = arguments.to_vec();
        let format = crate::vm::builtins::string_format_source(&self.runtime, arguments.first())
            .unwrap_or_default();
        let (mut format_index, mut argument_index) = (0, 1);
        while let Some((argument, next_format, next_argument)) =
            crate::vm::builtins::next_string_conversion(&format, format_index, argument_index)
        {
            format_index = next_format;
            argument_index = next_argument;
            let Some(value) = arguments.get(argument).cloned() else {
                break;
            };
            if matches!(
                value,
                Value::String(_) | Value::Integer(_) | Value::Number(_)
            ) || crate::runtime::cint64::CInt64::from_value(&value).is_some()
            {
                continue;
            }
            if !self.has_tostring_metamethod(&value)? {
                arguments[argument] = Value::String(self.runtime.format_value(&value)?);
                continue;
            }
            let result = self
                .call_value(
                    Value::BuiltinFunction(crate::runtime::BuiltinFunction::ToString),
                    vec![value],
                )?
                .into_iter()
                .next()
                .unwrap_or(Value::Nil);
            arguments[argument] = Value::String(self.runtime.format_value(&result)?);
        }
        let format = self
            .runtime
            .direct_native(DirectNativeKind::StringFormatLuaJitRaw)
            .ok_or_else(|| RuntimeError::message("string.format is not installed"))?;
        Ok(self.runtime.call_native(format, &arguments)?)
    }

    /// `print` of values, some with `__tostring`.
    fn print_meta(&mut self, arguments: &[Value]) -> VmResult<Vec<Value>> {
        let mut converted = Vec::with_capacity(arguments.len());
        for value in arguments {
            if !self.has_tostring_metamethod(value)? {
                converted.push(value.clone());
                continue;
            }
            let result = self
                .call_value(
                    Value::BuiltinFunction(crate::runtime::BuiltinFunction::ToString),
                    vec![value.clone()],
                )?
                .into_iter()
                .next()
                .unwrap_or(Value::Nil);
            match result {
                Value::String(_) | Value::Integer(_) | Value::Number(_) => converted.push(result),
                // Lua 5.1 and LuaJIT let `tostring` return anything, and
                // `print` objects.
                _ => {
                    return Err(RuntimeError::message(
                        "'tostring' must return a string to 'print'",
                    )
                    .into());
                }
            }
        }
        let print = self
            .runtime
            .direct_native(DirectNativeKind::Print)
            .ok_or_else(|| RuntimeError::message("print is not installed"))?;
        Ok(self.runtime.call_native(print, &converted)?)
    }

    fn table_concat_meta(&mut self, arguments: &[Value]) -> VmResult<Vec<Value>> {
        let table = arguments.first().cloned().unwrap_or(Value::Nil);
        let separator = match arguments.get(1) {
            None | Some(Value::Nil) => Vec::new(),
            Some(value @ (Value::String(_) | Value::Integer(_) | Value::Number(_))) => {
                self.runtime.format_value(value)?.as_bytes().to_vec()
            }
            argument => return Err(RuntimeError::bad_argument(2, "string", argument).into()),
        };
        let first = optional_integer(arguments, 3, 1)?;
        // The length is taken even when the end is given.
        let length = self.length_with_metamethod(&table)?;
        let last = optional_integer(arguments, 4, length)?;
        let mut result = Vec::new();
        let mut index = first;
        while index <= last {
            let value = self.element(&table, index)?;
            if !matches!(
                value,
                Value::String(_) | Value::Integer(_) | Value::Number(_)
            ) {
                return Err(RuntimeError::message(format!(
                    "invalid value ({}) at index {index} in table for 'concat'",
                    value.type_name()
                ))
                .into());
            }
            result.extend_from_slice(self.runtime.format_value(&value)?.as_bytes());
            if index == last {
                break;
            }
            result.extend_from_slice(&separator);
            index += 1;
        }
        Ok(vec![Value::from(result)])
    }

    fn table_unpack_meta(&mut self, arguments: &[Value]) -> VmResult<Vec<Value>> {
        let table = arguments.first().cloned().unwrap_or(Value::Nil);
        let first = optional_integer(arguments, 2, 1)?;
        // Lua 5.5 takes the length even when the end is given.
        let length = (self.compatibility == CompatibilityProfile::Lua55
            || matches!(arguments.get(2), None | Some(Value::Nil)))
        .then(|| self.length_with_metamethod(&table))
        .transpose()?;
        let last = match (arguments.get(2), length) {
            (None | Some(Value::Nil), Some(length)) => length,
            _ => check_integer(arguments, 3)?,
        };
        if first > last {
            return Ok(Vec::new());
        }
        let count = last.wrapping_sub(first) as u64;
        if count >= MAX_UNPACK_RESULTS {
            return Err(RuntimeError::message("too many results to unpack").into());
        }
        let mut values = Vec::with_capacity(usize::try_from(count + 1).unwrap_or(0));
        let mut index = first;
        loop {
            values.push(self.element(&table, index)?);
            if index == last {
                break;
            }
            index += 1;
        }
        Ok(values)
    }

    fn table_insert_meta(&mut self, arguments: &[Value]) -> VmResult<Vec<Value>> {
        let table = arguments.first().cloned().unwrap_or(Value::Nil);
        let end = self.length_with_metamethod(&table)?.wrapping_add(1);
        let position = match arguments.len() {
            2 => end,
            3 => {
                let position = check_integer(arguments, 2)?;
                // `(lua_Unsigned)pos - 1u < (lua_Unsigned)e`
                if (position as u64).wrapping_sub(1) >= end as u64 {
                    return Err(RuntimeError::argument_error(2, "position out of bounds").into());
                }
                let mut index = end;
                while index > position {
                    let previous = self.element(&table, index - 1)?;
                    self.set_element(&table, index, previous)?;
                    index -= 1;
                }
                position
            }
            _ => {
                return Err(RuntimeError::message("wrong number of arguments to 'insert'").into());
            }
        };
        let value = arguments.last().cloned().unwrap_or(Value::Nil);
        self.set_element(&table, position, value)?;
        Ok(Vec::new())
    }

    fn table_remove_meta(&mut self, arguments: &[Value]) -> VmResult<Vec<Value>> {
        let table = arguments.first().cloned().unwrap_or(Value::Nil);
        let size = self.length_with_metamethod(&table)?;
        let mut position = optional_integer(arguments, 2, size)?;
        if position != size && (position as u64).wrapping_sub(1) > size as u64 {
            return Err(RuntimeError::argument_error(2, "position out of bounds").into());
        }
        let removed = self.element(&table, position)?;
        while position < size {
            let next = self.element(&table, position + 1)?;
            self.set_element(&table, position, next)?;
            position += 1;
        }
        self.set_element(&table, position, Value::Nil)?;
        Ok(vec![removed])
    }

    fn table_move_meta(&mut self, arguments: &[Value]) -> VmResult<Vec<Value>> {
        let source = arguments.first().cloned().unwrap_or(Value::Nil);
        let first = check_integer(arguments, 2)?;
        let end = check_integer(arguments, 3)?;
        let target = check_integer(arguments, 4)?;
        let destination = match arguments.get(4) {
            None | Some(Value::Nil) => source.clone(),
            Some(value @ Value::Table(_)) => value.clone(),
            argument => return Err(RuntimeError::bad_argument(5, "table", argument).into()),
        };
        if end >= first {
            if !(first > 0 || end < i64::MAX + first) {
                return Err(RuntimeError::argument_error(3, "too many elements to move").into());
            }
            let count = end - first + 1;
            if target > i64::MAX - count + 1 {
                return Err(RuntimeError::argument_error(4, "destination wrap around").into());
            }
            if target > end || target <= first || !Runtime::raw_equal(&source, &destination) {
                for offset in 0..count {
                    let value = self.element(&source, first + offset)?;
                    self.set_element(&destination, target + offset, value)?;
                }
            } else {
                for offset in (0..count).rev() {
                    let value = self.element(&source, first + offset)?;
                    self.set_element(&destination, target + offset, value)?;
                }
            }
        }
        Ok(vec![destination])
    }
}
