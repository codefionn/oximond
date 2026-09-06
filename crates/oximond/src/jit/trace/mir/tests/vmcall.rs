//! Calls of host C functions from a trace (`HelperId::vm_call`) through the
//! MIR backend. The helper writes the state of the call's guard back to the
//! frame from what the code stored in the exit area just before the call, so
//! a host function that reads its caller's locals sees what the interpreter
//! would have left there.
//!
//! Each program runs under the interpreter, then under tracing with every
//! tree on Cranelift and with the MIR backend tried first. The results, and
//! the number of calls the traces made, must be the same; the MIR run must
//! have compiled a tree that makes such a call, and the Cranelift run none.

use super::VM_CALL_TREES;
use crate::jit::debug_env::FORCE_BACKEND;
use crate::{CompatibilityProfile, ExecutionMode, Lua, Value, Vm, VmError};

fn number(value: Option<&Value>) -> Option<f64> {
    match value? {
        #[allow(clippy::cast_precision_loss)]
        Value::Integer(value) => Some(*value as f64),
        Value::Number(value) => Some(*value),
        _ => None,
    }
}

/// `cadd(a, b)`: the sum, an integer when both are.
fn cadd(_: &mut Vm, _: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    Ok(vec![match (arguments.first(), arguments.get(1)) {
        (Some(Value::Integer(a)), Some(Value::Integer(b))) => Value::Integer(a.wrapping_add(*b)),
        (a, b) => match (number(a), number(b)) {
            (Some(a), Some(b)) => Value::Number(a + b),
            _ => Value::Nil,
        },
    }])
}

/// `ctwo(x)`: `x + 1` and `x * 2`, as floats.
fn ctwo(_: &mut Vm, _: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    let x = number(arguments.first()).unwrap_or(0.0);
    Ok(vec![Value::Number(x + 1.0), Value::Number(x * 2.0)])
}

/// `cflip(i)`: whether `i` is a multiple of three.
fn cflip(_: &mut Vm, _: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    let i = number(arguments.first()).unwrap_or(0.0);
    Ok(vec![Value::Boolean(i % 3.0 == 0.0)])
}

/// `cfail(n)`: raises an error when `n` is 150, else 1.
fn cfail(_: &mut Vm, _: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    if number(arguments.first()) == Some(150.0) {
        return Err(VmError::Runtime(crate::runtime::RuntimeError::message(
            "host function failed",
        )));
    }
    Ok(vec![Value::Number(1.0)])
}

/// `clocals()`: the sum of the locals of the Lua function that called it
/// (the first level that has a local `x`), a boolean counting 1000, and how
/// many locals it has. The accumulators `s` and `count` are left out of the
/// sum, which stays exact. The locals are registers the helper wrote back to
/// the frame.
fn clocals(vm: &mut Vm, _: u64, _: &[Value]) -> Result<Vec<Value>, VmError> {
    for level in 0..4 {
        let (mut sum, mut count, mut caller) = (0.0, 0.0, false);
        for index in 1..64 {
            let Some((name, value)) = vm.host_get_local(level, index)? else {
                break;
            };
            caller |= name.as_bytes() == b"x";
            count += 1.0;
            if matches!(name.as_bytes(), b"s" | b"count") {
                continue;
            }
            match value {
                Value::Boolean(flag) => sum += f64::from(u8::from(flag)) * 1000.0,
                value => sum += number(Some(&value)).unwrap_or(0.0),
            }
        }
        if caller {
            return Ok(vec![Value::Number(sum), Value::Number(count)]);
        }
    }
    Ok(vec![Value::Number(-1.0), Value::Number(-1.0)])
}

struct Run {
    outcome: String,
    /// Calls of host functions the traces made.
    calls: u64,
    /// Trees with such a call the MIR backend compiled.
    mir_trees: u64,
}

/// Runs `source` in `mode`, with `mir` picking the backend of every tree
/// (`None`: the environment's).
fn run(mode: ExecutionMode, mir: Option<bool>, source: &str) -> Run {
    FORCE_BACKEND.with(|force| force.set(mir));
    VM_CALL_TREES.with(|trees| trees.set(0));
    let mut lua = Lua::builder()
        .profile(CompatibilityProfile::Lua54)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    lua.vm_mut().ignore_remaining_work();
    lua.vm_mut().set_vmcall_min_work(0);
    let hosts: [(&str, crate::vm::HostCallback); 5] = [
        ("cadd", cadd),
        ("ctwo", ctwo),
        ("cflip", cflip),
        ("cfail", cfail),
        ("clocals", clocals),
    ];
    for (name, callback) in hosts {
        let function = lua
            .vm_mut()
            .register_host_function(callback, 1, Vec::new())
            .unwrap();
        lua.vm_mut().runtime_mut().set_global(name, function);
    }
    let outcome = match lua.exec(source) {
        Ok(results) => format!("{results:?}"),
        Err(error) => format!("error: {error}"),
    };
    FORCE_BACKEND.with(|force| force.set(None));
    Run {
        outcome,
        calls: lua.vm().jit_statistics().host_trace_calls,
        mir_trees: VM_CALL_TREES.with(std::cell::Cell::get),
    }
}

/// Runs `source` under the interpreter, Cranelift and MIR, checks that the
/// three agree and that MIR compiled a tree with a host call that Cranelift
/// compiled too, and returns the calls the MIR run's traces made.
fn agree(name: &str, source: &str) -> u64 {
    let expected = run(ExecutionMode::Interpreter, None, source);
    let cranelift = run(ExecutionMode::Tracing, Some(false), source);
    let mir = run(ExecutionMode::Tracing, Some(true), source);
    assert_eq!(cranelift.outcome, expected.outcome, "{name}: Cranelift");
    assert_eq!(mir.outcome, expected.outcome, "{name}: MIR");
    assert_eq!(mir.calls, cranelift.calls, "{name}: calls from traces");
    assert!(
        mir.mir_trees > 0,
        "{name}: MIR compiled a tree with a host call"
    );
    assert_eq!(
        cranelift.mir_trees, 0,
        "{name}: Cranelift compiled every tree"
    );
    mir.calls
}

#[test]
fn host_calls_with_every_result_kind_run_through_mir_like_cranelift() {
    let source = "local s, t, f = 0, 1, 0.5
        for i = 1, 4000 do
            local x = i * 2 + 1
            local y = x * 3 - i + (x % 5)
            s = cadd(s, y)
            local a, b = ctwo(x)
            if cflip(i) then t = t + 1 end
            f = f + a * 0.25 - b * 0.125
            s = s + (i % 7) * 2
        end
        return s, t, f";
    let calls = agree("result kinds", source);
    assert!(calls >= 3000, "{calls} calls from traces");
}

#[test]
fn a_host_function_reads_the_locals_the_trace_wrote_back_under_mir() {
    // Integers, floats and a boolean held in registers of the frame, which
    // the trace changes every round: the host function sees them only
    // through the state stored before the call.
    let source = "local s, count = 0, 0
        for i = 1, 3000 do
            local x = i * 2 + 1
            local y = x * 0.5 + i
            local z = (i % 7) * 3
            local flag = i % 2 == 0
            local sum, n = clocals()
            s = s + sum + z + (flag and 1 or 0)
            count = n
        end
        return s, count";
    let calls = agree("locals", source);
    assert!(calls >= 1000, "{calls} calls from traces");
}

#[test]
fn a_host_error_leaves_the_mir_trace_like_cranelift() {
    let source = "local last, s = 0, 0
        local ok, message = pcall(function()
            for i = 1, 400 do
                local x = i * 3 + 0.5
                s = s + cfail(i) + x
                last = i
            end
        end)
        return ok, message, last, s";
    agree("error", source);
}
