//! Lua-facing controls for Oximond's JIT execution policy.

use crate::runtime::{JitOptions, Runtime, RuntimeError, RuntimeResult, Value};

/// Whether a call applies to the whole engine. `LuaJIT`'s `jit.on(func)`,
/// `jit.off(func, true)` and `jit.flush(func)` only tune how one function is
/// compiled; they never change results, so Oximond accepts and ignores them.
fn is_global(arguments: &[Value], operation: &str) -> RuntimeResult<bool> {
    match arguments.first() {
        None | Some(Value::Nil) => Ok(true),
        Some(
            Value::Closure(_)
            | Value::NativeFunction(_)
            | Value::BuiltinFunction(_)
            | Value::Boolean(true),
        ) => Ok(false),
        Some(value) => Err(RuntimeError::message(format!(
            "bad argument #1 to '{operation}' (function expected, got {})",
            value.type_name()
        ))),
    }
}

pub(crate) fn on(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    if is_global(arguments, "on")? {
        runtime.set_jit_enabled(true);
    }
    Ok(Vec::new())
}

pub(crate) fn off(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    if is_global(arguments, "off")? {
        runtime.set_jit_enabled(false);
    }
    Ok(Vec::new())
}

/// `jit.opt.start(...)` tunes `LuaJIT`'s trace compiler. The parameters
/// `maxrecord`, `maxside`, `maxsnap`, `maxtrace`, `hotloop` and `maxmcode`
/// set this VM's limits for traces recorded from then on. Every other flag
/// and parameter is accepted and ignored, as are values that are not
/// non-negative integers.
pub(crate) fn opt_start(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let mut options = JitOptions::default();
    for argument in arguments {
        let Value::String(text) = argument else {
            continue;
        };
        let bytes = text.as_bytes();
        let Some(equals) = bytes.iter().position(|&byte| byte == b'=') else {
            continue;
        };
        let Some(value) = std::str::from_utf8(&bytes[equals + 1..])
            .ok()
            .and_then(|digits| digits.trim().parse::<u32>().ok())
        else {
            continue;
        };
        match &bytes[..equals] {
            b"maxrecord" => options.max_record = Some(value),
            b"maxside" => options.max_side = Some(value),
            b"maxsnap" => options.max_snap = Some(value),
            b"maxtrace" => options.max_trace = Some(value),
            b"hotloop" => options.hot_loop = Some(value),
            b"maxmcode" => options.max_mcode = Some(value),
            _ => {}
        }
    }
    if !options.is_empty() {
        runtime.request_jit_options(options);
    }
    Ok(Vec::new())
}

pub(crate) fn status(runtime: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
    Ok(vec![Value::Boolean(runtime.jit_requested())])
}

pub(crate) fn flush(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    if is_global(arguments, "flush")? {
        runtime.request_jit_flush();
    }
    Ok(Vec::new())
}
