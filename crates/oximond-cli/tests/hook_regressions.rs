//! Lua calls remain observable when protected entry or tail replacement is used.
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};
use std::cell::RefCell;
use std::rc::Rc;

fn trace(profile: CompatibilityProfile, mode: ExecutionMode, body: &str, entry: &str) -> String {
    let source = format!(
        r#"
        local target = assert((loadstring or load)({body:?}, "@hook-target"))
        local events = {{}}
        debug.sethook(function(event)
            local info = debug.getinfo(2, "S")
            if info and info.source == "@hook-target" then
                events[#events + 1] = event
            end
        end, "cr")
        {entry}
        debug.sethook()
        return table.concat(events, ",")
        "#
    );
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    let result = lua
        .exec(&source)
        .unwrap_or_else(|error| panic!("{profile} {mode}: {error}"));
    match &result[0] {
        Value::String(value) => value.to_lossy().into_owned(),
        other => panic!("unexpected trace {other:?}"),
    }
}

#[test]
fn protected_lua_entry_reports_its_call() {
    let body = "local function child() return 1 end; return (child())";
    for profile in CompatibilityProfile::ALL {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            for entry in [
                "local ok = pcall(target)",
                "local ok = xpcall(target, function(error) return error end)",
            ] {
                assert_eq!(
                    trace(profile, mode, body, entry),
                    "call,call,return,return",
                    "{profile} {mode} {entry}"
                );
            }
        }
    }
}

#[test]
fn tail_calls_report_profile_specific_events() {
    let body = "local function child(v) return v+1 end; local function parent(v) return child(v) end; local result=parent(1); return result";
    for profile in CompatibilityProfile::ALL {
        let expected = match profile {
            CompatibilityProfile::Lua51 => "call,call,call,return,tail return,return",
            CompatibilityProfile::LuaJit21 => "call,call,call,return,return",
            _ => "call,call,tail call,return,return",
        };
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            assert_eq!(
                trace(profile, mode, body, "local result = target()"),
                expected,
                "{profile} {mode}"
            );
        }
    }
}

#[test]
fn tail_return_hook_inspects_the_returning_loaded_chunk() {
    for (body, expected) in [
        (
            "local function child(v) return v+1 end; return child(1)",
            "call,call,return,tail return",
        ),
        (
            "local function child(v) return v+1 end; local function parent(v) return child(v) end; return parent(1)",
            "call,call,call,return,tail return,tail return",
        ),
    ] {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            assert_eq!(
                trace(
                    CompatibilityProfile::Lua51,
                    mode,
                    body,
                    "local result = target()"
                ),
                expected,
                "{mode}",
            );
        }
    }
}

#[test]
fn native_hooks_receive_each_tail_return_synchronously() {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let events = Rc::new(RefCell::new(Vec::new()));
        let captured = events.clone();
        let mut lua = Lua::builder()
            .profile(CompatibilityProfile::Lua51)
            .execution_mode(mode)
            .host_installer(move |runtime| {
                let captured = captured.clone();
                let callback = runtime.register_native_closure(move |_, arguments| {
                    if let Some(Value::String(event)) = arguments.first() {
                        captured.borrow_mut().push(event.to_lossy().into_owned());
                    }
                    Ok(Vec::new())
                })?;
                runtime.set_global("native_hook", Value::NativeFunction(callback));
                Ok(())
            })
            .build()
            .unwrap();
        lua.exec(
            r#"
            local function child(v) return v + 1 end
            local function parent(v) return child(v) end
            local function target() return parent(1) end
            debug.sethook(native_hook, "r")
            local value = target()
            debug.sethook()
            return value
            "#,
        )
        .unwrap();
        assert!(
            events.borrow().ends_with(&[
                "return".to_owned(),
                "tail return".to_owned(),
                "tail return".to_owned(),
            ]),
            "{mode}: {:?}",
            events.borrow(),
        );
    }
}

#[test]
fn count_hook_installing_line_hook_does_not_repeat_the_callers_line() {
    // Keep this source's lines and instruction sequence aligned with the
    // stock Lua 5.1 repro. Different Lua backends count different opcodes.
    let source = r#"local ev = {}
local function f()
  local x = 0
  for i=1,3 do x = x + i end
  return x
end
local function work() for i=1,20 do f() end end
debug.sethook(function() debug.sethook(function(e,l) ev[#ev+1]=l end, "l") end, "", 50)
work()
debug.sethook()
return #ev == 122, table.concat(ev, ",", 1, 12)
"#;
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut lua = Lua::builder()
            .profile(CompatibilityProfile::Lua51)
            .execution_mode(mode)
            .build()
            .unwrap();
        assert_eq!(
            lua.exec(source).unwrap(),
            vec![Value::Boolean(true), Value::from("4,5,7,3,4,4,4,4,5,7,3,4"),],
            "{mode}",
        );
    }
}

#[test]
fn count_hook_installing_line_hook_keeps_the_first_line_of_a_new_frame() {
    // This checks Oximond's count scheduling. Stock Lua stops on a
    // different opcode for this source, so its complete trace differs.
    let source = r#"local events = {}
local function target() local value = 1; value = value + 1; return value end
local function count_hook() debug.sethook(function(event, line) events[#events + 1] = line end, "l") end
debug.sethook(count_hook, "", 3)
target()
local result = 1
debug.sethook()
return table.concat(events, ",")
"#;
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut lua = Lua::builder()
            .profile(CompatibilityProfile::Lua51)
            .execution_mode(mode)
            .build()
            .unwrap();
        assert_eq!(
            lua.exec(source).unwrap(),
            vec![Value::from("2,6,7")],
            "{mode}",
        );
    }
}

#[test]
fn line_hook_installed_at_count_three_skips_the_completed_callers_line() {
    let source = r#"local events = {}
local function f() return 1 end
debug.sethook(function() debug.sethook(function(event, line) events[#events+1] = line end, "l") end, "", 3)
local value = f()
local result = value + 1
debug.sethook()
return table.concat(events, ",")
"#;
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut lua = Lua::builder()
            .profile(CompatibilityProfile::Lua51)
            .execution_mode(mode)
            .build()
            .unwrap();
        assert_eq!(
            lua.exec(source).unwrap(),
            vec![Value::from("5,6")],
            "{mode}"
        );
    }
}
