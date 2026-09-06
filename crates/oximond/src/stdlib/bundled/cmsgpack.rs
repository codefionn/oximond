//! Byte-compatible `MessagePack` encoding used by Redis's lua-cmsgpack 0.4.
//!
//! Unsupported Lua values and tables at the encoding depth limit become nil,
//! as in lua-cmsgpack. Strings remain arbitrary bytes. Map order follows next.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "the MessagePack wire format deliberately selects numeric widths and signedness"
)]

use super::{entries, set};
use crate::runtime::{NativeCallback, Runtime, RuntimeError, RuntimeResult, TableHandle, Value};
use crate::stdlib::strings::string_argument_in;

pub(super) fn module(runtime: &mut Runtime, safe: bool) -> RuntimeResult<TableHandle> {
    let table = runtime.create_table()?;
    for (name, callback) in [
        ("pack", pack as NativeCallback),
        ("unpack", unpack as NativeCallback),
        ("unpack_one", unpack_one as NativeCallback),
        ("unpack_limit", unpack_limit as NativeCallback),
    ] {
        let handle = runtime.register_native_closure(move |runtime, arguments| {
            match callback(runtime, arguments) {
                Err(error) if safe => Ok(vec![Value::Nil, Value::from(error.to_string())]),
                result => result,
            }
        })?;
        set(runtime, table, name, Value::NativeFunction(handle))?;
    }
    set(runtime, table, "_NAME", Value::from("cmsgpack"))?;
    set(
        runtime,
        table,
        "_VERSION",
        Value::from("lua-cmsgpack 0.4.0"),
    )?;
    set(
        runtime,
        table,
        "_COPYRIGHT",
        Value::from("Copyright (C) 2012, Salvatore Sanfilippo"),
    )?;
    set(
        runtime,
        table,
        "_DESCRIPTION",
        Value::from("MessagePack C implementation for Lua"),
    )?;
    Ok(table)
}

fn pack(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    if arguments.is_empty() {
        return Err(RuntimeError::argument_error(
            0,
            "MessagePack pack needs input.",
        ));
    }
    let mut output = Vec::new();
    for value in arguments {
        encode(runtime, value, &mut output, 0)?;
    }
    Ok(vec![Value::from(output)])
}

fn integer(output: &mut Vec<u8>, number: i64) {
    if (0..=127).contains(&number) || (-32..=-1).contains(&number) {
        output.push(number as u8);
        return;
    }
    let (tag, size) = if number >= 0 {
        if number <= 255 {
            (0xcc, 1)
        } else if number <= 65535 {
            (0xcd, 2)
        } else if number <= 0xffff_ffff {
            (0xce, 4)
        } else {
            (0xcf, 8)
        }
    } else if number >= -128 {
        (0xd0, 1)
    } else if number >= -32768 {
        (0xd1, 2)
    } else if number >= i64::from(i32::MIN) {
        (0xd2, 4)
    } else {
        (0xd3, 8)
    };
    output.push(tag);
    output.extend_from_slice(&number.to_be_bytes()[8 - size..]);
}

fn length(output: &mut Vec<u8>, len: usize, fix: u8, tags: &[u8]) -> RuntimeResult<()> {
    let len =
        u32::try_from(len).map_err(|_| RuntimeError::message("MessagePack value is too large"))?;
    if len < if fix == 0xa0 { 32 } else { 16 } {
        output.push(fix | len as u8);
    } else if tags.len() == 3 && len <= 255 {
        output.extend_from_slice(&[tags[0], len as u8]);
    } else if len <= 65535 {
        output.push(tags[tags.len() - 2]);
        output.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        output.push(tags[tags.len() - 1]);
        output.extend_from_slice(&len.to_be_bytes());
    }
    Ok(())
}

#[allow(
    clippy::float_cmp,
    reason = "a float32 is used only when it reproduces the double exactly"
)]
fn encode(
    runtime: &Runtime,
    value: &Value,
    output: &mut Vec<u8>,
    depth: usize,
) -> RuntimeResult<()> {
    match value {
        Value::Boolean(value) => output.push(if *value { 0xc3 } else { 0xc2 }),
        Value::Integer(value) => integer(output, *value),
        Value::Number(value) => {
            if value.is_finite()
                && value.fract() == 0.0
                && *value >= i64::MIN as f64
                && *value < -(i64::MIN as f64)
            {
                integer(output, *value as i64);
            } else if f64::from(*value as f32) == *value {
                output.push(0xca);
                output.extend_from_slice(&(*value as f32).to_be_bytes());
            } else {
                output.push(0xcb);
                output.extend_from_slice(&value.to_be_bytes());
            }
        }
        Value::String(value) => {
            length(output, value.len(), 0xa0, &[0xd9, 0xda, 0xdb])?;
            output.extend_from_slice(value.as_bytes());
        }
        Value::Table(table) if depth < 16 => {
            let pairs = entries(runtime, *table)?;
            let array = pairs.iter().all(|(key, _)| match key {
                Value::Integer(value) => *value > 0 && *value <= pairs.len() as i64,
                Value::Number(value) => {
                    *value > 0.0 && value.fract() == 0.0 && *value <= pairs.len() as f64
                }
                _ => false,
            });
            if array {
                length(output, pairs.len(), 0x90, &[0xdc, 0xdd])?;
                for index in 1..=pairs.len() {
                    encode(
                        runtime,
                        &runtime.table_get(*table, &runtime.integer_value(index as i64))?,
                        output,
                        depth + 1,
                    )?;
                }
            } else {
                length(output, pairs.len(), 0x80, &[0xde, 0xdf])?;
                for (key, value) in pairs {
                    encode(runtime, &key, output, depth + 1)?;
                    encode(runtime, &value, output, depth + 1)?;
                }
            }
        }
        _ => output.push(0xc0),
    }
    Ok(())
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

struct DecodeFrame {
    table: TableHandle,
    remaining: usize,
    map: bool,
    key: Option<Value>,
    index: i64,
}

impl Cursor<'_> {
    fn take(&mut self, count: usize) -> RuntimeResult<&[u8]> {
        let end = self.position.checked_add(count).ok_or_else(missing)?;
        let bytes = self.bytes.get(self.position..end).ok_or_else(missing)?;
        self.position = end;
        Ok(bytes)
    }

    fn uint(&mut self, size: usize) -> RuntimeResult<u64> {
        Ok(self
            .take(size)?
            .iter()
            .fold(0, |value, byte| (value << 8) | u64::from(*byte)))
    }

    fn decode(&mut self, runtime: &mut Runtime) -> RuntimeResult<Value> {
        let mut frames = Vec::<DecodeFrame>::new();
        loop {
            let Some(mut value) = self.item(runtime, &mut frames)? else {
                continue;
            };
            loop {
                let Some(frame) = frames.last_mut() else {
                    return Ok(value);
                };
                if frame.map && frame.key.is_none() {
                    frame.key = Some(value);
                    break;
                }
                let key = if frame.map {
                    frame.key.take().expect("map key was decoded")
                } else {
                    runtime.integer_value(frame.index)
                };
                runtime.table_set(frame.table, &key, value)?;
                frame.index += 1;
                frame.remaining -= 1;
                if frame.remaining > 0 {
                    break;
                }
                value = Value::Table(frame.table);
                frames.pop();
            }
        }
    }

    fn item(
        &mut self,
        runtime: &mut Runtime,
        frames: &mut Vec<DecodeFrame>,
    ) -> RuntimeResult<Option<Value>> {
        let tag = self.uint(1)? as u8;
        let (kind, count) = match tag {
            0x00..=0x7f => return Ok(Some(runtime.integer_value(i64::from(tag)))),
            0xe0..=0xff => return Ok(Some(runtime.integer_value(i64::from(tag as i8)))),
            0xc0 => return Ok(Some(Value::Nil)),
            0xc2 | 0xc3 => return Ok(Some(Value::Boolean(tag == 0xc3))),
            0xca => {
                return Ok(Some(Value::Number(f64::from(f32::from_bits(
                    self.uint(4)? as u32,
                )))));
            }
            0xcb => return Ok(Some(Value::Number(f64::from_bits(self.uint(8)?)))),
            0xcc..=0xcf => {
                let value = self.uint(1 << (tag - 0xcc))?;
                // lua-cmsgpack pushes uint64 through lua_Integer on a 64-bit Lua host.
                return Ok(Some(runtime.integer_value(value as i64)));
            }
            0xd0..=0xd3 => {
                let size = 1usize << (tag - 0xd0);
                let unsigned = self.uint(size)?;
                let value = ((unsigned << (64 - size * 8)) as i64) >> (64 - size * 8);
                return Ok(Some(runtime.integer_value(value)));
            }
            0xa0..=0xbf => (0, usize::from(tag & 31)),
            0xd9..=0xdb => (0, self.uint(1 << (tag - 0xd9))? as usize),
            0x90..=0x9f => (1, usize::from(tag & 15)),
            0xdc..=0xdd => (1, self.uint(if tag == 0xdc { 2 } else { 4 })? as usize),
            0x80..=0x8f => (2, usize::from(tag & 15)),
            0xde..=0xdf => (2, self.uint(if tag == 0xde { 2 } else { 4 })? as usize),
            _ => return Err(RuntimeError::message("Bad data format in input.")),
        };
        if kind == 0 {
            return Ok(Some(Value::bytes(self.take(count)?.to_vec())));
        }
        let minimum = count
            .checked_mul(if kind == 2 { 2 } else { 1 })
            .ok_or_else(missing)?;
        if minimum > self.bytes.len() - self.position {
            return Err(missing());
        }
        let table = runtime.create_table()?;
        if count == 0 {
            return Ok(Some(Value::Table(table)));
        }
        frames.push(DecodeFrame {
            table,
            remaining: count,
            map: kind == 2,
            key: None,
            index: 1,
        });
        Ok(None)
    }
}

fn missing() -> RuntimeError {
    RuntimeError::message("Missing bytes in input.")
}

fn unpack(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    unpack_full(runtime, arguments, 0, 0)
}

fn index(
    runtime: &Runtime,
    argument: Option<&Value>,
    position: usize,
    default: Option<i64>,
) -> RuntimeResult<i64> {
    match argument {
        None | Some(Value::Nil) if default.is_some() => Ok(default.unwrap_or(0)),
        Some(value) => runtime
            .to_number(value)
            .filter(|number| number.is_finite())
            .map(|number| number as i64)
            .ok_or_else(|| RuntimeError::bad_argument(position, "number", argument)),
        _ => Err(RuntimeError::bad_argument(position, "number", argument)),
    }
}

fn unpack_one(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let offset = index(runtime, arguments.get(1), 2, Some(0))?;
    unpack_full(runtime, arguments, 1, offset)
}

fn unpack_limit(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let limit = index(runtime, arguments.get(1), 2, None)?;
    let offset = index(runtime, arguments.get(2), 3, Some(0))?;
    unpack_full(runtime, arguments, limit, offset)
}

fn unpack_full(
    runtime: &mut Runtime,
    arguments: &[Value],
    limit: i64,
    offset: i64,
) -> RuntimeResult<Vec<Value>> {
    let input = string_argument_in(runtime.number_format(), arguments.first(), 1)?;
    if limit < 0 || offset < 0 {
        return Err(RuntimeError::message(format!(
            "Invalid request to unpack with offset of {offset} and limit of {}.",
            input.len()
        )));
    }
    if offset as u64 > input.len() as u64 {
        return Err(RuntimeError::message(format!(
            "Start offset {offset} greater than input length {}.",
            input.len()
        )));
    }
    let all = limit == 0 && offset == 0;
    let mut cursor = Cursor {
        bytes: &input,
        position: offset as usize,
    };
    let mut output = Vec::new();
    while cursor.position < input.len() && (all || output.len() < limit as usize) {
        output.push(cursor.decode(runtime)?);
    }
    if !all {
        output.insert(
            0,
            runtime.integer_value(if cursor.position == input.len() {
                -1
            } else {
                cursor.position as i64
            }),
        );
    }
    Ok(output)
}
