//! Redis's struct format (Lua struct 1.7), distinct from string.pack.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "struct wire fields deliberately wrap and convert through LuaNumber"
)]

use super::set;
use crate::runtime::{NativeCallback, Runtime, RuntimeError, RuntimeResult, TableHandle, Value};
use crate::stdlib::strings::string_argument_in;

pub(super) fn module(runtime: &mut Runtime) -> RuntimeResult<TableHandle> {
    let table = runtime.create_table()?;
    for (name, callback) in [
        ("pack", pack as NativeCallback),
        ("unpack", unpack as NativeCallback),
        ("size", size as NativeCallback),
    ] {
        let handle = runtime.register_native(callback)?;
        set(runtime, table, name, Value::NativeFunction(handle))?;
    }
    Ok(table)
}

#[derive(Clone, Copy)]
struct Item {
    option: u8,
    size: usize,
    little: bool,
    alignment: usize,
}

fn number(format: &[u8], position: &mut usize, default: usize) -> RuntimeResult<usize> {
    if !format.get(*position).is_some_and(u8::is_ascii_digit) {
        return Ok(default);
    }
    let mut value = 0usize;
    while let Some(byte) = format.get(*position).filter(|byte| byte.is_ascii_digit()) {
        value = value
            .checked_mul(10)
            .and_then(|value| value.checked_add(usize::from(*byte - b'0')))
            .filter(|value| i32::try_from(*value).is_ok())
            .ok_or_else(|| RuntimeError::message("integral size overflow"))?;
        *position += 1;
    }
    Ok(value)
}

fn parse(format: &[u8], size_only: bool) -> RuntimeResult<Vec<Item>> {
    let mut position = 0;
    let mut little = cfg!(target_endian = "little");
    let mut alignment = 1;
    let mut items = Vec::new();
    while position < format.len() && format[position] != 0 {
        let option = format[position];
        position += 1;
        let size = match option {
            b' ' => continue,
            b'<' => {
                little = true;
                continue;
            }
            b'>' => {
                little = false;
                continue;
            }
            b'!' => {
                alignment = number(format, &mut position, 8)?;
                if !alignment.is_power_of_two() {
                    return Err(RuntimeError::message(format!(
                        "alignment {alignment} is not a power of 2"
                    )));
                }
                continue;
            }
            b'b' | b'B' | b'x' => 1,
            b'h' | b'H' => 2,
            b'l' | b'L' => std::mem::size_of::<std::os::raw::c_long>(),
            b'T' => std::mem::size_of::<usize>(),
            b'f' => 4,
            b'd' => 8,
            b'i' | b'I' => {
                let size = number(format, &mut position, 4)?;
                if size > 32 {
                    return Err(RuntimeError::message(format!(
                        "integral size {size} is larger than limit of 32"
                    )));
                }
                size
            }
            b'c' => number(format, &mut position, 1)?,
            b's' => 0,
            // struct.size skips unknown alphanumeric options. pack/unpack
            // reject them. Keep this upstream quirk local to size queries.
            option if size_only && option.is_ascii_alphanumeric() => 0,
            _ => {
                return Err(RuntimeError::argument_error(
                    1,
                    format!("invalid format option '{}'", char::from(option)),
                ));
            }
        };
        items.push(Item {
            option,
            size,
            little,
            alignment,
        });
    }
    Ok(items)
}

fn padding(position: usize, item: Item) -> usize {
    if item.size == 0 || item.option == b'c' {
        return 0;
    }
    let align = item.size.min(item.alignment);
    // This deliberately matches struct's bit-mask rule for odd integer widths.
    (align - (position & (align - 1))) & (align - 1)
}

fn reserve(output: &mut Vec<u8>, length: usize) -> RuntimeResult<()> {
    output
        .try_reserve(length)
        .map_err(|_| RuntimeError::message("struct result is too large"))
}

fn pack(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let format = string_argument_in(runtime.number_format(), arguments.first(), 1)?;
    let mut output = Vec::new();
    let mut argument = 1;
    for item in parse(&format, false)? {
        let pad = padding(output.len(), item);
        reserve(&mut output, pad + item.size)?;
        output.resize(output.len() + pad, 0);
        match item.option {
            b'x' => output.push(0),
            b'c' | b's' => {
                let string = string_argument_in(
                    runtime.number_format(),
                    arguments.get(argument),
                    argument + 1,
                )?;
                argument += 1;
                let size = if item.size == 0 {
                    string.len()
                } else {
                    item.size
                };
                if string.len() < size {
                    return Err(RuntimeError::argument_error(
                        argument + 1,
                        "string too short",
                    ));
                }
                reserve(&mut output, size + usize::from(item.option == b's'))?;
                output.extend_from_slice(&string[..size]);
                if item.option == b's' {
                    output.push(0);
                }
            }
            _ => {
                let value = arguments
                    .get(argument)
                    .and_then(|value| runtime.to_number(value))
                    .ok_or_else(|| {
                        RuntimeError::bad_argument(argument + 1, "number", arguments.get(argument))
                    })?;
                argument += 1;
                let mut bytes = match item.option {
                    b'f' => (value as f32).to_le_bytes().to_vec(),
                    b'd' => value.to_le_bytes().to_vec(),
                    _ => {
                        // struct checks lua_Number even when its host Lua
                        // has tagged integers. Preserve that conversion.
                        let mut number = if value < 0.0 {
                            value as i64 as u64
                        } else {
                            value as u64
                        };
                        let mut bytes = Vec::with_capacity(item.size);
                        for _ in 0..item.size {
                            bytes.push(number as u8);
                            number >>= 8;
                        }
                        bytes
                    }
                };
                if !item.little {
                    bytes.reverse();
                }
                output.extend_from_slice(&bytes);
            }
        }
    }
    Ok(vec![Value::from(output)])
}

fn too_short() -> RuntimeError {
    RuntimeError::argument_error(2, "data string too short")
}

fn unpack(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let format = string_argument_in(runtime.number_format(), arguments.first(), 1)?;
    let input = string_argument_in(runtime.number_format(), arguments.get(1), 2)?;
    let offset = match arguments.get(2) {
        None | Some(Value::Nil) => 1.0,
        Some(value) => runtime
            .to_number(value)
            .ok_or_else(|| RuntimeError::bad_argument(3, "number", Some(value)))?,
    };
    if !offset.is_finite() || offset < 1.0 {
        return Err(RuntimeError::argument_error(
            3,
            "offset must be 1 or greater",
        ));
    }
    let mut position = (offset as usize).saturating_sub(1);
    let mut output = Vec::new();
    for item in parse(&format, false)? {
        position = position
            .checked_add(padding(position, item))
            .ok_or_else(too_short)?;
        let mut size = item.size;
        if item.option == b'c' && size == 0 {
            let previous = output
                .pop()
                .ok_or_else(|| RuntimeError::message("format 'c0' needs a previous size"))?;
            let length = runtime
                .to_number(&previous)
                .ok_or_else(|| RuntimeError::message("format 'c0' needs a previous size"))?;
            if !length.is_finite() || length < 0.0 {
                return Err(too_short());
            }
            size = length as usize;
        }
        let end = position.checked_add(size).ok_or_else(too_short)?;
        let bytes = input.get(position..end).ok_or_else(too_short)?;
        let value = match item.option {
            b'x' => {
                position = end;
                continue;
            }
            b'c' => Value::bytes(bytes.to_vec()),
            b's' => {
                let rest = input.get(position..).ok_or_else(too_short)?;
                let length = rest
                    .iter()
                    .position(|byte| *byte == 0)
                    .ok_or_else(|| RuntimeError::message("unfinished string in data"))?;
                size = length + 1;
                Value::bytes(rest[..length].to_vec())
            }
            b'f' | b'd' => {
                let mut bytes = bytes.to_vec();
                if !item.little {
                    bytes.reverse();
                }
                if item.option == b'f' {
                    Value::Number(f64::from(f32::from_le_bytes(
                        bytes.try_into().expect("float width"),
                    )))
                } else {
                    Value::Number(f64::from_le_bytes(bytes.try_into().expect("double width")))
                }
            }
            _ => {
                let mut integer = 0u64;
                if item.little {
                    for byte in bytes.iter().rev() {
                        integer = (integer << 8) | u64::from(*byte);
                    }
                } else {
                    for byte in bytes {
                        integer = (integer << 8) | u64::from(*byte);
                    }
                }
                if item.option.is_ascii_lowercase()
                    && size > 0
                    && size < 8
                    && integer & (1 << (size * 8 - 1)) != 0
                {
                    integer |= u64::MAX << (size * 8);
                }
                // struct's integer fields return lua_Number, even on newer Lua.
                Value::Number(if item.option.is_ascii_lowercase() {
                    (integer as i64) as f64
                } else {
                    integer as f64
                })
            }
        };
        output.push(value);
        position = position.checked_add(size).ok_or_else(too_short)?;
    }
    output.push(runtime.integer_value(position.checked_add(1).ok_or_else(too_short)? as i64));
    Ok(output)
}

fn size(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let format = string_argument_in(runtime.number_format(), arguments.first(), 1)?;
    let mut position = 0usize;
    for item in parse(&format, true)? {
        if item.option == b's' {
            return Err(RuntimeError::argument_error(
                1,
                "option 's' has no fixed size",
            ));
        }
        if item.option == b'c' && item.size == 0 {
            return Err(RuntimeError::argument_error(
                1,
                "option 'c0' has no fixed size",
            ));
        }
        position = position
            .checked_add(padding(position, item))
            .and_then(|position| position.checked_add(item.size))
            .ok_or_else(|| RuntimeError::message("struct size is too large"))?;
    }
    Ok(vec![runtime.integer_value(position as i64)])
}
