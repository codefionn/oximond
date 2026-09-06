//! Lua CJSON 2.1's byte-oriented JSON and independent per-module settings.
//!
//! Encoding and decoding use explicit work stacks so the default nesting limit
//! of 1000 does not consume the Rust call stack. As upstream, empty tables are
//! objects, numeric object keys become strings, and JSON null keeps table keys.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "CJSON uses LuaNumber conversion and integer settings bounded to INT_MAX"
)]

use super::set;
use crate::runtime::{Runtime, RuntimeError, RuntimeResult, TableHandle, Value};
use crate::stdlib::strings::string_argument_in;
use std::cell::RefCell;
use std::rc::Rc;

struct Config {
    sparse_convert: bool,
    sparse_ratio: usize,
    sparse_safe: usize,
    encode_depth: usize,
    decode_depth: usize,
    precision: usize,
    encode_invalid: u8,
    decode_invalid: bool,
    keep_buffer: bool,
    buffer: Vec<u8>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sparse_convert: false,
            sparse_ratio: 2,
            sparse_safe: 10,
            encode_depth: 1000,
            decode_depth: 1000,
            precision: 14,
            encode_invalid: 0,
            decode_invalid: true,
            keep_buffer: true,
            buffer: Vec::new(),
        }
    }
}

pub(super) fn module(runtime: &mut Runtime, null: Value, safe: bool) -> RuntimeResult<TableHandle> {
    let table = runtime.create_table()?;
    let config = Rc::new(RefCell::new(Config::default()));
    for encode in [true, false] {
        let config = Rc::clone(&config);
        let null = null.clone();
        let handle = runtime.register_native_closure(move |runtime, arguments| {
            // cjson.safe deliberately raises wrong-arity errors before its
            // protected conversion, just like the ordinary module.
            if arguments.len() != 1 {
                return Err(RuntimeError::argument_error(1, "expected 1 argument"));
            }
            let result = if encode {
                let mut config = config.borrow_mut();
                let mut buffer = std::mem::take(&mut config.buffer);
                buffer.clear();
                let result = encode_value(runtime, &arguments[0], &null, &config, &mut buffer)
                    .map(|()| vec![Value::bytes(buffer.clone())]);
                if config.keep_buffer {
                    config.buffer = buffer;
                }
                result
            } else {
                let config = config.borrow();
                string_argument_in(runtime.number_format(), arguments.first(), 1)
                    .and_then(|source| decode_value(runtime, &source, &null, &config))
                    .map(|value| vec![value])
            };
            match result {
                Err(error) if safe => Ok(vec![Value::Nil, Value::from(error.to_string())]),
                result => result,
            }
        })?;
        set(
            runtime,
            table,
            if encode { "encode" } else { "decode" },
            Value::NativeFunction(handle),
        )?;
    }
    for name in [
        "encode_sparse_array",
        "encode_max_depth",
        "decode_max_depth",
        "encode_number_precision",
        "encode_keep_buffer",
        "encode_invalid_numbers",
        "decode_invalid_numbers",
    ] {
        let config = Rc::clone(&config);
        let handle = runtime.register_native_closure(move |runtime, arguments| {
            configure(runtime, &mut config.borrow_mut(), name, arguments)
        })?;
        set(runtime, table, name, Value::NativeFunction(handle))?;
    }
    let new_null = null.clone();
    let handle = runtime.register_native_closure(move |runtime, _| {
        module(runtime, new_null.clone(), safe).map(|table| vec![Value::Table(table)])
    })?;
    set(runtime, table, "new", Value::NativeFunction(handle))?;
    set(runtime, table, "null", null)?;
    set(runtime, table, "_NAME", Value::from("cjson"))?;
    set(runtime, table, "_VERSION", Value::from("2.1.0"))?;
    Ok(table)
}

fn integer_option(
    runtime: &Runtime,
    argument: Option<&Value>,
    position: usize,
    value: &mut usize,
    minimum: usize,
    maximum: usize,
) -> RuntimeResult<()> {
    if let Some(argument) = argument.filter(|value| !matches!(value, Value::Nil)) {
        let number = runtime
            .to_number(argument)
            .ok_or_else(|| RuntimeError::bad_argument(position, "number", Some(argument)))?;
        if !number.is_finite() || number.trunc() < minimum as f64 || number.trunc() > maximum as f64
        {
            return Err(RuntimeError::argument_error(
                1,
                format!("expected integer between {minimum} and {maximum}"),
            ));
        }
        *value = number as usize;
    }
    Ok(())
}

fn enum_option(
    runtime: &Runtime,
    argument: Option<&Value>,
    position: usize,
    value: &mut u8,
    allow_null: bool,
) -> RuntimeResult<()> {
    match argument {
        None | Some(Value::Nil) => {}
        Some(Value::Boolean(boolean)) => *value = u8::from(*boolean),
        Some(argument) => {
            let string = string_argument_in(runtime.number_format(), Some(argument), position)?;
            *value = match string.as_ref() {
                b"off" => 0,
                b"on" => 1,
                b"null" if allow_null => 2,
                _ => {
                    return Err(RuntimeError::argument_error(
                        position,
                        format!("invalid option '{}'", String::from_utf8_lossy(&string)),
                    ));
                }
            };
        }
    }
    Ok(())
}

fn configure(
    runtime: &Runtime,
    config: &mut Config,
    name: &str,
    arguments: &[Value],
) -> RuntimeResult<Vec<Value>> {
    let count = if name == "encode_sparse_array" { 3 } else { 1 };
    if arguments.len() > count {
        return Err(RuntimeError::argument_error(
            count + 1,
            "found too many arguments",
        ));
    }
    match name {
        "encode_sparse_array" => {
            let mut convert = u8::from(config.sparse_convert);
            enum_option(runtime, arguments.first(), 1, &mut convert, false)?;
            config.sparse_convert = convert != 0;
            integer_option(
                runtime,
                arguments.get(1),
                2,
                &mut config.sparse_ratio,
                0,
                i32::MAX as usize,
            )?;
            integer_option(
                runtime,
                arguments.get(2),
                3,
                &mut config.sparse_safe,
                0,
                i32::MAX as usize,
            )?;
            Ok(vec![
                Value::Boolean(config.sparse_convert),
                runtime.integer_value(config.sparse_ratio as i64),
                runtime.integer_value(config.sparse_safe as i64),
            ])
        }
        "encode_max_depth" | "decode_max_depth" | "encode_number_precision" => {
            let maximum = if name == "encode_number_precision" {
                14
            } else {
                i32::MAX as usize
            };
            let value = match name {
                "encode_max_depth" => &mut config.encode_depth,
                "decode_max_depth" => &mut config.decode_depth,
                _ => &mut config.precision,
            };
            integer_option(runtime, arguments.first(), 1, value, 1, maximum)?;
            Ok(vec![runtime.integer_value(*value as i64)])
        }
        "encode_invalid_numbers" => {
            enum_option(
                runtime,
                arguments.first(),
                1,
                &mut config.encode_invalid,
                true,
            )?;
            Ok(vec![if config.encode_invalid == 2 {
                Value::from("null")
            } else {
                Value::Boolean(config.encode_invalid != 0)
            }])
        }
        _ => {
            let value = if name == "encode_keep_buffer" {
                &mut config.keep_buffer
            } else {
                &mut config.decode_invalid
            };
            let mut option = u8::from(*value);
            enum_option(runtime, arguments.first(), 1, &mut option, false)?;
            *value = option != 0;
            let result = *value;
            if !config.keep_buffer {
                config.buffer = Vec::new();
            }
            Ok(vec![Value::Boolean(result)])
        }
    }
}

fn append(output: &mut Vec<u8>, bytes: &[u8]) -> RuntimeResult<()> {
    output
        .try_reserve(bytes.len())
        .map_err(|_| RuntimeError::message("JSON result is too large"))?;
    output.extend_from_slice(bytes);
    Ok(())
}

fn quoted(output: &mut Vec<u8>, string: &[u8]) -> RuntimeResult<()> {
    const HEX: &[u8] = b"0123456789abcdef";
    append(output, b"\"")?;
    for byte in string {
        match byte {
            b'"' => append(output, b"\\\"")?,
            b'\\' => append(output, b"\\\\")?,
            b'/' => append(output, b"\\/")?,
            b'\x08' => append(output, b"\\b")?,
            b'\x0c' => append(output, b"\\f")?,
            b'\n' => append(output, b"\\n")?,
            b'\r' => append(output, b"\\r")?,
            b'\t' => append(output, b"\\t")?,
            0..=31 => append(
                output,
                &[
                    b'\\',
                    b'u',
                    b'0',
                    b'0',
                    HEX[usize::from(byte >> 4)],
                    HEX[usize::from(byte & 15)],
                ],
            )?,
            _ => append(output, &[*byte])?,
        }
    }
    append(output, b"\"")
}

fn trim_decimal(mut string: String) -> String {
    if string.contains('.') {
        while string.ends_with('0') {
            string.pop();
        }
        if string.ends_with('.') {
            string.pop();
        }
    }
    string
}

fn number_string(value: f64, precision: usize) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.into();
    }
    let scientific = format!("{:.*e}", precision - 1, value);
    let (mantissa, exponent) = scientific.split_once('e').expect("scientific notation");
    let exponent: i32 = exponent.parse().expect("scientific exponent");
    if exponent < -4 || exponent >= precision as i32 {
        format!(
            "{}e{}{:02}",
            trim_decimal(mantissa.into()),
            if exponent >= 0 { "+" } else { "-" },
            exponent.unsigned_abs()
        )
    } else {
        trim_decimal(format!(
            "{:.*}",
            (precision as i32 - 1 - exponent) as usize,
            value
        ))
    }
}

fn number(output: &mut Vec<u8>, value: f64, config: &Config) -> RuntimeResult<()> {
    if value.is_finite() {
        return append(output, number_string(value, config.precision).as_bytes());
    }
    match config.encode_invalid {
        0 => Err(RuntimeError::message(
            "Cannot serialise number: must not be NaN or Inf",
        )),
        2 => append(output, b"null"),
        _ => append(
            output,
            if value.is_nan() {
                b"nan"
            } else if value.is_sign_negative() {
                b"-inf"
            } else {
                b"inf"
            },
        ),
    }
}

enum Task {
    Value(Value, usize),
    Key(Value),
    Array {
        table: TableHandle,
        index: usize,
        maximum: usize,
        depth: usize,
    },
    Object {
        table: TableHandle,
        previous: Value,
        depth: usize,
    },
}

#[allow(
    clippy::too_many_lines,
    reason = "keep the iterative JSON encoder's transitions in one match"
)]
fn encode_value(
    runtime: &Runtime,
    value: &Value,
    null: &Value,
    config: &Config,
    output: &mut Vec<u8>,
) -> RuntimeResult<()> {
    let mut tasks = vec![Task::Value(value.clone(), 0)];
    while let Some(task) = tasks.pop() {
        match task {
            Task::Array {
                table,
                index,
                maximum,
                depth,
            } => {
                if index > maximum {
                    append(output, b"]")?;
                    continue;
                }
                if index > 1 {
                    append(output, b",")?;
                }
                tasks.push(Task::Array {
                    table,
                    index: index + 1,
                    maximum,
                    depth,
                });
                tasks.push(Task::Value(
                    runtime.table_get(table, &runtime.integer_value(index as i64))?,
                    depth,
                ));
            }
            Task::Object {
                table,
                previous,
                depth,
            } => {
                let (key, value) = runtime.table_next(table, &previous)?;
                if matches!(key, Value::Nil) {
                    append(output, b"}")?;
                    continue;
                }
                if !matches!(previous, Value::Nil) {
                    append(output, b",")?;
                }
                tasks.push(Task::Object {
                    table,
                    previous: key.clone(),
                    depth,
                });
                tasks.push(Task::Value(value, depth));
                tasks.push(Task::Key(key));
            }
            Task::Key(key) => {
                match key {
                    Value::String(value) => quoted(output, value.as_bytes())?,
                    Value::Integer(value) => {
                        append(output, b"\"")?;
                        number(output, value as f64, config)?;
                        append(output, b"\"")?;
                    }
                    Value::Number(value) => {
                        append(output, b"\"")?;
                        number(output, value, config)?;
                        append(output, b"\"")?;
                    }
                    value => {
                        return Err(RuntimeError::message(format!(
                            "Cannot serialise {}: table key must be a number or string",
                            value.type_name()
                        )));
                    }
                }
                append(output, b":")?;
            }
            Task::Value(value, depth) => match value {
                Value::Nil => append(output, b"null")?,
                value if value == *null => append(output, b"null")?,
                Value::Boolean(value) => append(output, if value { b"true" } else { b"false" })?,
                Value::String(value) => quoted(output, value.as_bytes())?,
                Value::Integer(value) => number(output, value as f64, config)?,
                Value::Number(value) => number(output, value, config)?,
                Value::Table(table) => {
                    let depth = depth + 1;
                    if depth > config.encode_depth {
                        return Err(RuntimeError::message(format!(
                            "Cannot serialise, excessive nesting ({depth})"
                        )));
                    }
                    let mut maximum = 0usize;
                    let mut count = 0usize;
                    let mut array = true;
                    let mut previous = Value::Nil;
                    loop {
                        let (key, _) = runtime.table_next(table, &previous)?;
                        if matches!(key, Value::Nil) {
                            break;
                        }
                        count += 1;
                        previous = key.clone();
                        let key = match key {
                            Value::Integer(value) => value as f64,
                            Value::Number(value) => value,
                            _ => {
                                array = false;
                                break;
                            }
                        };
                        if !key.is_finite()
                            || key < 1.0
                            || key.fract() != 0.0
                            || key > f64::from(i32::MAX)
                        {
                            array = false;
                            break;
                        }
                        maximum = maximum.max(key as usize);
                    }
                    array &= count > 0;
                    if array
                        && config.sparse_ratio > 0
                        && maximum > count.saturating_mul(config.sparse_ratio)
                        && maximum > config.sparse_safe
                    {
                        if !config.sparse_convert {
                            return Err(RuntimeError::message(
                                "Cannot serialise table: excessively sparse array",
                            ));
                        }
                        array = false;
                    }
                    if array {
                        append(output, b"[")?;
                        tasks.push(Task::Array {
                            table,
                            index: 1,
                            maximum,
                            depth,
                        });
                    } else {
                        append(output, b"{")?;
                        tasks.push(Task::Object {
                            table,
                            previous: Value::Nil,
                            depth,
                        });
                    }
                }
                value => {
                    return Err(RuntimeError::message(format!(
                        "Cannot serialise {}: type not supported",
                        value.type_name()
                    )));
                }
            },
        }
    }
    Ok(())
}

#[derive(Clone)]
enum Token {
    Object,
    ObjectEnd,
    Array,
    ArrayEnd,
    String(Vec<u8>),
    Number(f64),
    Boolean(bool),
    Null,
    Colon,
    Comma,
    End,
    Error(&'static str),
}

impl Token {
    fn name(&self) -> &'static str {
        match self {
            Self::Object => "T_OBJ_BEGIN",
            Self::ObjectEnd => "T_OBJ_END",
            Self::Array => "T_ARR_BEGIN",
            Self::ArrayEnd => "T_ARR_END",
            Self::String(_) => "T_STRING",
            Self::Number(_) => "T_NUMBER",
            Self::Boolean(_) => "T_BOOLEAN",
            Self::Null => "T_NULL",
            Self::Colon => "T_COLON",
            Self::Comma => "T_COMMA",
            Self::End => "T_END",
            Self::Error(message) => message,
        }
    }
}

struct Parser<'a> {
    source: &'a [u8],
    position: usize,
    token_start: usize,
    invalid: bool,
}

impl Parser<'_> {
    fn next(&mut self) -> Token {
        while self
            .source
            .get(self.position)
            .is_some_and(|byte| matches!(byte, b' ' | b'\t' | b'\r' | b'\n'))
        {
            self.position += 1;
        }
        self.token_start = self.position;
        let Some(&byte) = self.source.get(self.position) else {
            return Token::End;
        };
        let simple = match byte {
            0 => return Token::End,
            b'{' => Some(Token::Object),
            b'}' => Some(Token::ObjectEnd),
            b'[' => Some(Token::Array),
            b']' => Some(Token::ArrayEnd),
            b':' => Some(Token::Colon),
            b',' => Some(Token::Comma),
            _ => None,
        };
        if let Some(simple) = simple {
            self.position += 1;
            return simple;
        }
        if byte == b'"' {
            return self.string();
        }
        for (word, value) in [
            (b"true".as_slice(), Token::Boolean(true)),
            (b"false".as_slice(), Token::Boolean(false)),
            (b"null".as_slice(), Token::Null),
        ] {
            if self.source[self.position..].starts_with(word) {
                self.position += word.len();
                return value;
            }
        }
        if byte == b'-'
            || byte.is_ascii_digit()
            || (self.invalid
                && (byte == b'+'
                    || self.source[self.position..].get(..3).is_some_and(|prefix| {
                        prefix.eq_ignore_ascii_case(b"inf") || prefix.eq_ignore_ascii_case(b"nan")
                    })))
        {
            return self.numeric();
        }
        Token::Error("invalid token")
    }

    fn failure(&self, expected: &str, found: &Token) -> RuntimeError {
        RuntimeError::message(format!(
            "Expected {expected} but found {} at character {}",
            found.name(),
            self.token_start + 1
        ))
    }

    fn hex4(&self, start: usize) -> Option<u32> {
        self.source
            .get(start..start + 4)?
            .iter()
            .try_fold(0, |value, byte| {
                char::from(*byte)
                    .to_digit(16)
                    .map(|digit| (value << 4) | digit)
            })
    }

    fn string(&mut self) -> Token {
        self.position += 1;
        let mut output = Vec::new();
        loop {
            let byte = *self.source.get(self.position).unwrap_or(&0);
            if byte == 0 {
                self.token_start = self.position;
                return Token::Error("unexpected end of string");
            }
            if byte == b'"' {
                self.position += 1;
                return Token::String(output);
            }
            if byte != b'\\' {
                output.push(byte);
                self.position += 1;
                continue;
            }
            let escape = *self.source.get(self.position + 1).unwrap_or(&0);
            if escape == b'u' {
                let Some(mut codepoint) = self.hex4(self.position + 2) else {
                    self.token_start = self.position;
                    return Token::Error("invalid unicode escape code");
                };
                let mut length = 6;
                if (0xd800..=0xdfff).contains(&codepoint) {
                    let low =
                        if self.source.get(self.position + 6..self.position + 8) == Some(b"\\u") {
                            self.hex4(self.position + 8)
                        } else {
                            None
                        };
                    let Some(low) = low.filter(|low| (0xdc00..=0xdfff).contains(low)) else {
                        self.token_start = self.position;
                        return Token::Error("invalid unicode escape code");
                    };
                    if codepoint > 0xdbff {
                        self.token_start = self.position;
                        return Token::Error("invalid unicode escape code");
                    }
                    codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + low - 0xdc00;
                    length = 12;
                }
                let Some(character) = char::from_u32(codepoint) else {
                    self.token_start = self.position;
                    return Token::Error("invalid unicode escape code");
                };
                let mut utf8 = [0; 4];
                output.extend_from_slice(character.encode_utf8(&mut utf8).as_bytes());
                self.position += length;
            } else {
                let decoded = match escape {
                    b'"' | b'\\' | b'/' => escape,
                    b'b' => 8,
                    b'f' => 12,
                    b'n' => b'\n',
                    b'r' => b'\r',
                    b't' => b'\t',
                    _ => {
                        self.token_start = self.position;
                        return Token::Error("invalid escape code");
                    }
                };
                output.push(decoded);
                self.position += 2;
            }
        }
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the token scanner tracks one cursor through decimal and hexadecimal number forms"
    )]
    fn numeric(&mut self) -> Token {
        let start = self.position;
        let mut position = start;
        let negative = self.source.get(position) == Some(&b'-');
        if matches!(self.source.get(position), Some(b'-' | b'+')) {
            position += 1;
        }
        let rest = &self.source[position..];
        let invalid = self.source[start] == b'+'
            || rest.get(..3).is_some_and(|prefix| {
                prefix.eq_ignore_ascii_case(b"inf") || prefix.eq_ignore_ascii_case(b"nan")
            })
            || (rest.first() == Some(&b'0')
                && rest
                    .get(1)
                    .is_some_and(|byte| byte.is_ascii_digit() || matches!(byte, b'x' | b'X')));
        if invalid && !self.invalid {
            return Token::Error("invalid number");
        }
        for (word, value) in [
            (b"infinity".as_slice(), f64::INFINITY),
            (b"inf".as_slice(), f64::INFINITY),
            (b"nan".as_slice(), f64::NAN),
        ] {
            if rest
                .get(..word.len())
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case(word))
            {
                self.position = position + word.len();
                // strtod accepts NaN payloads containing letters, digits,
                // and underscores. CJSON exposes that when invalid numbers
                // are enabled, even though a payload is not JSON syntax.
                if value.is_nan() && self.source.get(self.position) == Some(&b'(') {
                    let mut end = self.position + 1;
                    while self
                        .source
                        .get(end)
                        .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
                    {
                        end += 1;
                    }
                    if self.source.get(end) == Some(&b')') {
                        self.position = end + 1;
                    }
                }
                return Token::Number(if negative { -value } else { value });
            }
        }
        if rest
            .get(..2)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(b"0x"))
        {
            let hex_start = position;
            position += 2;
            let mut value = 0.0;
            let mut count = 0;
            while let Some(digit) = self
                .source
                .get(position)
                .and_then(|byte| char::from(*byte).to_digit(16))
            {
                value = value * 16.0 + f64::from(digit);
                position += 1;
                count += 1;
            }
            if self.source.get(position) == Some(&b'.') {
                position += 1;
                let mut multiplier = 1.0 / 16.0;
                while let Some(digit) = self
                    .source
                    .get(position)
                    .and_then(|byte| char::from(*byte).to_digit(16))
                {
                    value += multiplier * f64::from(digit);
                    multiplier /= 16.0;
                    position += 1;
                    count += 1;
                }
            }
            if count == 0 {
                // strtod falls back to the leading zero when a hex prefix
                // has no digits, leaving its 'x' for the next token.
                self.position = hex_start + 1;
                return Token::Number(if negative { -0.0 } else { 0.0 });
            }
            if matches!(self.source.get(position), Some(b'p' | b'P')) {
                let exponent_start = position;
                position += 1;
                let digits_start = position;
                if matches!(self.source.get(position), Some(b'-' | b'+')) {
                    position += 1;
                }
                let first_digit = position;
                while self.source.get(position).is_some_and(u8::is_ascii_digit) {
                    position += 1;
                }
                if position == first_digit {
                    position = exponent_start;
                } else {
                    let exponent = std::str::from_utf8(&self.source[digits_start..position])
                        .ok()
                        .and_then(|number| number.parse::<i32>().ok())
                        .unwrap_or(if self.source[digits_start] == b'-' {
                            i32::MIN
                        } else {
                            i32::MAX
                        });
                    value *= 2.0f64.powi(exponent);
                }
            }
            self.position = position;
            return Token::Number(if negative { -value } else { value });
        }
        let digits_start = position;
        while self.source.get(position).is_some_and(u8::is_ascii_digit) {
            position += 1;
        }
        let mut count = position - digits_start;
        if self.source.get(position) == Some(&b'.') {
            position += 1;
            let digits = position;
            while self.source.get(position).is_some_and(u8::is_ascii_digit) {
                position += 1;
            }
            count += position - digits;
        }
        if count == 0 {
            return Token::Error("invalid number");
        }
        if matches!(self.source.get(position), Some(b'e' | b'E')) {
            let exponent_start = position;
            position += 1;
            if matches!(self.source.get(position), Some(b'-' | b'+')) {
                position += 1;
            }
            let digits = position;
            while self.source.get(position).is_some_and(u8::is_ascii_digit) {
                position += 1;
            }
            if position == digits {
                position = exponent_start;
            }
        }
        let number = std::str::from_utf8(&self.source[start..position])
            .ok()
            .and_then(|number| number.parse::<f64>().ok());
        match number {
            Some(number) => {
                self.position = position;
                Token::Number(number)
            }
            None => Token::Error("invalid number"),
        }
    }
}

enum Frame {
    Array {
        table: TableHandle,
        index: i64,
        need_value: bool,
    },
    Object {
        table: TableHandle,
        key: Option<Value>,
        state: u8,
    },
}

#[allow(
    clippy::too_many_lines,
    reason = "keep the iterative JSON decoder's frame transitions together"
)]
fn decode_value(
    runtime: &mut Runtime,
    source: &[u8],
    null: &Value,
    config: &Config,
) -> RuntimeResult<Value> {
    if source.len() >= 2 && (source[0] == 0 || source[1] == 0) {
        return Err(RuntimeError::message(
            "JSON parser does not support UTF-16 or UTF-32",
        ));
    }
    let mut parser = Parser {
        source,
        position: 0,
        token_start: 0,
        invalid: config.decode_invalid,
    };
    let mut frames: Vec<Frame> = Vec::new();
    let mut result = None;
    let mut token = parser.next();
    loop {
        let mut need_value = frames.is_empty() && result.is_none();
        if let Some(frame) = frames.last_mut() {
            match frame {
                Frame::Array {
                    table,
                    index,
                    need_value: want,
                } => {
                    if *want && *index == 1 && matches!(token, Token::ArrayEnd) {
                        let value = Value::Table(*table);
                        frames.pop();
                        deliver(runtime, &mut frames, &mut result, value)?;
                        token = parser.next();
                        continue;
                    }
                    if *want {
                        need_value = true;
                    } else {
                        match token {
                            Token::ArrayEnd => {
                                let value = Value::Table(*table);
                                frames.pop();
                                deliver(runtime, &mut frames, &mut result, value)?;
                            }
                            Token::Comma => *want = true,
                            _ => return Err(parser.failure("comma or array end", &token)),
                        }
                        token = parser.next();
                        continue;
                    }
                }
                Frame::Object { table, key, state } => match *state {
                    0 => {
                        if matches!(token, Token::ObjectEnd) {
                            let value = Value::Table(*table);
                            frames.pop();
                            deliver(runtime, &mut frames, &mut result, value)?;
                            token = parser.next();
                            continue;
                        }
                        let Token::String(string) = token else {
                            return Err(parser.failure("object key string", &token));
                        };
                        *key = Some(Value::bytes(string));
                        *state = 1;
                        token = parser.next();
                        continue;
                    }
                    1 => {
                        if !matches!(token, Token::Colon) {
                            return Err(parser.failure("colon", &token));
                        }
                        *state = 2;
                        token = parser.next();
                        continue;
                    }
                    2 => need_value = true,
                    3 => {
                        match token {
                            Token::ObjectEnd => {
                                let value = Value::Table(*table);
                                frames.pop();
                                deliver(runtime, &mut frames, &mut result, value)?;
                            }
                            Token::Comma => *state = 4,
                            _ => return Err(parser.failure("comma or object end", &token)),
                        }
                        token = parser.next();
                        continue;
                    }
                    _ => {
                        let Token::String(string) = token else {
                            return Err(parser.failure("object key string", &token));
                        };
                        *key = Some(Value::bytes(string));
                        *state = 1;
                        token = parser.next();
                        continue;
                    }
                },
            }
        }
        if !need_value {
            if matches!(token, Token::End) {
                return Ok(result.expect("decoded root"));
            }
            return Err(parser.failure("the end", &token));
        }
        let value = match token {
            Token::String(value) => Value::bytes(value),
            Token::Number(value) => Value::Number(value),
            Token::Boolean(value) => Value::Boolean(value),
            Token::Null => null.clone(),
            Token::Array | Token::Object => {
                let depth = frames.len() + 1;
                if depth > config.decode_depth {
                    return Err(RuntimeError::message(format!(
                        "Found too many nested data structures ({depth}) at character {}",
                        parser.position
                    )));
                }
                let table = runtime.create_table()?;
                frames.push(if matches!(token, Token::Array) {
                    Frame::Array {
                        table,
                        index: 1,
                        need_value: true,
                    }
                } else {
                    Frame::Object {
                        table,
                        key: None,
                        state: 0,
                    }
                });
                token = parser.next();
                continue;
            }
            _ => return Err(parser.failure("value", &token)),
        };
        deliver(runtime, &mut frames, &mut result, value)?;
        token = parser.next();
    }
}

fn deliver(
    runtime: &mut Runtime,
    frames: &mut [Frame],
    result: &mut Option<Value>,
    value: Value,
) -> RuntimeResult<()> {
    match frames.last_mut() {
        Some(Frame::Array {
            table,
            index,
            need_value,
        }) => {
            runtime.table_set(*table, &runtime.integer_value(*index), value)?;
            *index += 1;
            *need_value = false;
        }
        Some(Frame::Object { table, key, state }) => {
            runtime.table_set(*table, &key.take().expect("object value has key"), value)?;
            *state = 3;
        }
        None => *result = Some(value),
    }
    Ok(())
}
