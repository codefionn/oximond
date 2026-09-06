//! Host (C API style) callbacks: the VM captures the suspended Lua stack only
//! when a registered host function is actually invoked, and everything the
//! callback can observe stays rooted and inspectable while it runs.

use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value, Vm, VmError};

fn lua(profile: CompatibilityProfile) -> Lua {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(ExecutionMode::Interpreter)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    lua
}

fn register(lua: &mut Lua, name: &str, callback: oximond::vm::HostCallback) {
    let function = lua
        .vm_mut()
        .register_host_function(callback, 0, Vec::new())
        .unwrap();
    lua.vm_mut().runtime_mut().set_global(name, function);
}

fn collect(vm: &mut Vm, _: u64, _: &[Value]) -> Result<Vec<Value>, VmError> {
    vm.collect_full_with_values(&[]);
    Ok(vec![Value::from("collected")])
}

fn debug_getinfo(vm: &mut Vm, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    let debug = vm.runtime().get_global("debug");
    let getinfo = vm.get_value(debug, Value::from("getinfo"))?;
    vm.call_value(getinfo, arguments.to_vec())
}

/// `debug.getinfo(2, options)` from inside the callback: level 1 is the C
/// function itself, level 2 its Lua caller.
fn caller_info(vm: &mut Vm, _: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    let options = arguments.first().cloned().unwrap_or(Value::from("nSl"));
    debug_getinfo(vm, &[Value::Number(2.0), options])
}

fn caller_line(vm: &mut Vm, _: u64, _: &[Value]) -> Result<Vec<Value>, VmError> {
    let frame = vm.host_stack_info(1).expect("a Lua caller is suspended");
    Ok(vec![
        Value::Number(f64::from(frame.current_line)),
        frame.name.map_or(Value::Nil, Value::String),
        Value::from(frame.name_what),
    ])
}

#[test]
fn host_metamethod_collection_keeps_suspended_lua_values_alive() {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua55] {
        let mut lua = lua(profile);
        register(&mut lua, "host_collect", collect);
        let result = lua
            .exec(
                r#"
                local weak = setmetatable({}, { __mode = "k" })
                local held = {}
                weak[held] = true
                weak[{}] = true
                local proxy = setmetatable({}, { __index = host_collect })
                local marker = proxy.anything
                local count = 0
                for _ in pairs(weak) do count = count + 1 end
                return marker, count == 1, weak[held]
            "#,
            )
            .unwrap();
        assert_eq!(
            result,
            vec![
                Value::from("collected"),
                Value::Boolean(true),
                Value::Boolean(true)
            ],
            "{profile}"
        );
    }
}

#[test]
fn host_callback_sees_its_lua_caller_line_and_name() {
    let mut lua = lua(CompatibilityProfile::Lua51);
    register(&mut lua, "where", caller_line);
    let result = lua
        .exec(
            r"
            local function inner()
                local x = 1
                return where()
            end
            local total = 0
            for i = 1, 1000 do total = total + i end
            local line, name = where()
            local inner_line, inner_name, inner_namewhat = inner()
            return line, name, inner_line, inner_name, inner_namewhat, total
        ",
        )
        .unwrap();
    // Level 1 is the suspended Lua caller: the main chunk has no name, and
    // `inner` is named by the call expression in its own caller.
    assert_eq!(
        result,
        vec![
            Value::Number(8.0),
            Value::Nil,
            Value::Number(4.0),
            Value::from("inner"),
            Value::from("local"),
            Value::Number(500_500.0),
        ]
    );
}

#[test]
fn debug_getinfo_from_host_reports_caller_and_active_lines() {
    let mut lua = lua(CompatibilityProfile::Lua51);
    register(&mut lua, "info", caller_info);
    let result = lua
        .exec(
            r#"
            local function probe()
                local t = info("SlL")
                return t.currentline, t.what, t.activelines[3], t.activelines[4]
            end
            return probe()
        "#,
        )
        .unwrap();
    assert_eq!(
        result,
        vec![
            Value::Number(3.0),
            Value::from("Lua"),
            Value::Boolean(true),
            Value::Boolean(true),
        ]
    );
}

#[test]
fn host_callbacks_nest_through_reentered_lua() {
    let mut lua = lua(CompatibilityProfile::Lua51);
    register(&mut lua, "where", caller_line);
    register(&mut lua, "host_collect", collect);
    let result = lua
        .exec(
            r"
            local keep = { value = 42 }
            local proxy = setmetatable({}, {
                __index = function(_, key)
                    host_collect()
                    local line = where()
                    return key .. line
                end,
            })
            local text = proxy.at
            return text, keep.value
        ",
        )
        .unwrap();
    assert_eq!(result, vec![Value::from("at6"), Value::Number(42.0)]);
}

fn double(_: &mut Vm, _: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    Ok(vec![match arguments.first() {
        Some(Value::Number(value)) => Value::Number(value * 2.0),
        Some(Value::Integer(value)) => Value::Integer(value * 2),
        _ => Value::Nil,
    }])
}

/// Registered host functions do not keep hot code out of native traces,
/// and native call sites that reach a host callback go through the VM so it
/// can root and describe the suspended stack.
#[test]
fn tracing_hot_loops_call_host_callbacks_through_the_vm() {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua55] {
        let mut lua = Lua::builder()
            .profile(profile)
            .execution_mode(ExecutionMode::Tracing)
            .build()
            .unwrap();
        lua.vm_mut().use_synchronous_jit_compilation();
        register(&mut lua, "double", double);
        register(&mut lua, "host_collect", collect);
        let result = lua
            .exec(
                r"
                local keep = { value = 7 }
                local total = 0
                for i = 1, 20000 do
                    total = total + double(i)
                    if i % 5000 == 0 then host_collect() end
                end
                local numeric = 0
                for i = 1, 200000 do numeric = numeric + i * 0.5 end
                return total == 400020000, keep.value == 7, numeric == 10000050000
            ",
            )
            .unwrap();
        assert_eq!(result, vec![Value::Boolean(true); 3], "{profile}");
        // The callback loop stays in the interpreter; the numeric loop after
        // it must still run as native trace code.
        assert!(
            lua.vm().jit_statistics().trace.native_entries > 0,
            "{profile}: host functions must not disable native traces"
        );
    }
}

fn churn_after_call(vm: &mut Vm, _: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    vm.call_value(arguments[0].clone(), Vec::new())?;
    // Reuse any freed capture slice with garbage.
    let mut junk: Vec<std::sync::Arc<[u64]>> = Vec::new();
    for n in 0..20000usize {
        junk.push(std::sync::Arc::from(vec![
            0xdead_beef_dead_beef_u64;
            1 + n % 12
        ]));
    }
    std::hint::black_box(&junk);
    vm.collect_full_with_values(&[]);
    Ok(Vec::new())
}

#[test]
fn upvaluejoin_from_a_host_callback_keeps_active_captures_alive() {
    // `f` borrows its capture cells while parked in the host call, and the
    // nested call replaces them via `debug.upvaluejoin`. Outside `run_stack`
    // (host coroutine) nothing else keeps the old slice alive.
    for _ in 0..2 {
        let mut lua = lua(CompatibilityProfile::Lua54);
        let values = lua
            .exec(
                r"
return function(hostcall)
  local a, b = 1, 2
  local function f()
    hostcall(function()
      local c = 100
      local function g() return c end
      debug.upvaluejoin(f, 1, g, 1)
    end)
    return a, b
  end
  return function() return f() end
end
",
            )
            .unwrap();
        let vm = lua.vm_mut();
        let host = vm
            .register_host_function(churn_after_call, 0, Vec::new())
            .unwrap();
        let entry = vm
            .call_value(values[0].clone(), vec![host])
            .unwrap()
            .remove(0);
        let thread = vm.create_host_coroutine().unwrap();
        let result = vm.start_host_coroutine(thread, entry, Vec::new()).unwrap();
        assert_eq!(result[0], Value::Boolean(true), "{result:?}");
    }
}
