//! A host function may have a second entry that stores its results where
//! the caller wants them. The fast loop's fixed calls use it. These calls
//! must give what the plain callback gives: the same values, nil past the
//! last one, the same errors, and arguments that survive Lua code the
//! callback runs.
#![allow(unsafe_code)]

use std::cell::Cell;

use oximond::runtime::Value;
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Vm, VmError};

thread_local! {
    static DIRECT_CALLS: Cell<u64> = const { Cell::new(0) };
}

/// `swap(a, b, f)`: calls `f` when given, then returns `b, a`. Fails on
/// the string "fail".
fn swap(vm: &mut Vm, _token: u64, args: &[Value]) -> Result<Vec<Value>, VmError> {
    if matches!(args.first(), Some(Value::String(text)) if text.as_bytes() == b"fail") {
        return Err(oximond::runtime::RuntimeError::message("swap failed").into());
    }
    if let Some(function @ Value::Closure(_)) = args.get(2) {
        vm.call_value(function.clone(), Vec::new())?;
    }
    Ok(vec![
        args.get(1).cloned().unwrap_or(Value::Nil),
        args.first().cloned().unwrap_or(Value::Nil),
    ])
}

unsafe fn swap_direct(
    vm: &mut Vm,
    token: u64,
    args: *const Value,
    count: usize,
    out: *mut Value,
    wanted: usize,
) -> Result<(), VmError> {
    DIRECT_CALLS.set(DIRECT_CALLS.get() + 1);
    // SAFETY: the caller's contract. The arguments are copied, since the
    // results may be stored over them.
    let arguments = unsafe { std::slice::from_raw_parts(args, count) }.to_vec();
    let results = swap(vm, token, &arguments)?;
    let mut results = results.into_iter();
    for index in 0..wanted {
        // SAFETY: the caller's contract.
        unsafe { *out.add(index) = results.next().unwrap_or(Value::Nil) };
    }
    Ok(())
}

fn run(direct: bool, source: &str) -> Vec<Value> {
    run_in(ExecutionMode::Interpreter, direct, source).0
}

/// The results, and the host calls traces made themselves.
fn run_in(mode: ExecutionMode, direct: bool, source: &str) -> (Vec<Value>, u64) {
    let mut lua = Lua::builder()
        .profile(CompatibilityProfile::Lua54)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let function = lua
        .vm_mut()
        .register_host_function(swap, 0, Vec::new())
        .unwrap();
    if direct {
        // SAFETY: `swap_direct` is `swap` with the results stored.
        assert!(unsafe {
            lua.vm_mut()
                .set_host_direct_callback(&function, swap_direct)
        });
    }
    lua.vm_mut().runtime_mut().set_global("swap", function);
    let results = lua.exec(source).unwrap();
    (results, lua.vm().jit_statistics().host_trace_calls)
}

fn same(source: &str) -> Vec<Value> {
    let expected = run(false, source);
    DIRECT_CALLS.set(0);
    let got = run(true, source);
    assert_eq!(got, expected, "{source}");
    assert!(DIRECT_CALLS.get() > 0, "the direct entry ran: {source}");
    got
}

#[test]
fn results_land_in_the_callers_registers() {
    let got = same(
        "local a, b, c = swap(1, 'two')
         local d = swap({}, 4)
         swap(5, 6)
         local e = {swap(7, 8)}
         return a, b, c, d, #e, (swap('x', 'y'))",
    );
    assert_eq!(got[0], Value::string("two"));
    assert_eq!(got[1], Value::Integer(1));
    assert_eq!(got[2], Value::Nil);
}

#[test]
fn errors_come_out_as_the_plain_callback_raises_them() {
    same(
        "local ok, message = pcall(function() local a, b = swap('fail', 1) return a end)
         return ok, message",
    );
}

#[test]
fn captured_arguments_survive_lua_code_the_callback_runs() {
    same(
        "local total = 0
         for i = 1, 200 do
           local x, y = i, 'k' .. i
           local function clobber() x, y = 'gone', {} end
           local a, b = swap(x, y, clobber)
           total = total + b + #a
           x, y = swap(x, y)
           assert(x ~= nil and y == 'gone')
         end
         return total",
    );
}

#[test]
fn only_a_host_function_takes_a_direct_entry() {
    let mut lua = Lua::builder()
        .profile(CompatibilityProfile::Lua54)
        .execution_mode(ExecutionMode::Interpreter)
        .build()
        .unwrap();
    assert!(!unsafe {
        lua.vm_mut()
            .set_host_direct_callback(&Value::Integer(1), swap_direct)
    });
    let function = lua
        .vm_mut()
        .register_host_function(swap, 0, Vec::new())
        .unwrap();
    assert!(unsafe {
        lua.vm_mut()
            .set_host_direct_callback(&function, swap_direct)
    });
    lua.vm_mut().runtime_mut().set_global("swap", function);
    DIRECT_CALLS.set(0);
    assert_eq!(
        lua.exec("local a, b = swap(1, 2) return a, b").unwrap(),
        vec![Value::Integer(2), Value::Integer(1)]
    );
    assert_eq!(DIRECT_CALLS.get(), 1);
}

#[test]
fn a_trace_calls_through_the_direct_entry() {
    let source = "local function sum()
           local s = 0
           for i = 1, 30000 do
             local a, b = swap(i < 99999 and i or 'no', i < 99999 and 2.5 or 'no')
             if type(a) == 'number' then s = s + a + b else s = s + b end
           end
           return s
         end
         return sum()";
    let (expected, _) = run_in(ExecutionMode::Interpreter, false, source);
    let (plain, plain_calls) = run_in(ExecutionMode::Tracing, false, source);
    DIRECT_CALLS.set(0);
    let (direct, direct_calls) = run_in(ExecutionMode::Tracing, true, source);
    assert_eq!(plain, expected);
    assert_eq!(direct, expected);
    // The traces make the calls themselves either way, and with the entry
    // each of those goes through it.
    assert!(plain_calls > 10_000, "{plain_calls} host calls from traces");
    assert!(
        direct_calls > 10_000,
        "{direct_calls} host calls from traces"
    );
    assert!(DIRECT_CALLS.get() >= direct_calls);
}

#[test]
fn a_trace_leaves_when_a_direct_call_fails_or_changes_its_results() {
    let source = "local s = 0
         local ok, message = pcall(function()
           for i = 1, 30000 do
             local a, b = swap(i < 25000 and i or 'fail', i < 20000 and 1 or 'one')
             if type(a) == 'number' then s = s + a + b else s = s + b end
           end
         end)
         return s, ok, message";
    let (expected, _) = run_in(ExecutionMode::Interpreter, false, source);
    let (direct, calls) = run_in(ExecutionMode::Tracing, true, source);
    assert_eq!(direct, expected);
    assert!(calls > 10_000, "{calls} host calls from traces");
}
