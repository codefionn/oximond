//! Closing the state runs the pending `__gc` finalizers, as `lua_close` does
//! at the end of the standalone interpreter.

use std::process::{Command, Output};

const MODES: [&str; 2] = ["interpreter", "tracing"];

fn run(args: &[&str], source: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args(args)
        .args(["-e", source])
        .output()
        .expect("run oximond")
}

fn run_modes(profile: &str, extra: &[&str], source: &str) -> Vec<Output> {
    MODES
        .iter()
        .map(|mode| {
            let compat = format!("--compat={profile}");
            let execution = format!("--execution={mode}");
            let mut args = vec![compat.as_str(), execution.as_str()];
            args.extend_from_slice(extra);
            run(&args, source)
        })
        .collect()
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

const ORDER_54: &str = "for _, n in ipairs{'a', 'b', 'c'} do \
    setmetatable({}, {__gc = function() print(n) end}) end print('end')";

#[test]
fn a_normal_end_runs_finalizers_in_reverse_order_of_marking() {
    for output in run_modes("lua54", &[], ORDER_54) {
        assert!(output.status.success(), "{output:?}");
        assert_eq!(stdout(&output), "end\nc\nb\na\n");
        assert!(output.stderr.is_empty(), "{output:?}");
    }
}

#[test]
fn lua51_finalizes_userdata_only_in_reverse_creation_order() {
    let source = "local u = newproxy(true) getmetatable(u).__gc = function() print('u1') end \
        setmetatable({}, {__gc = function() print('table') end}) \
        local v = newproxy(true) getmetatable(v).__gc = function() print('u2') end \
        print('end')";
    for output in run_modes("lua51", &[], source) {
        assert!(output.status.success(), "{output:?}");
        assert_eq!(stdout(&output), "end\nu2\nu1\n");
    }
}

#[test]
fn closing_exit_runs_finalizers_and_plain_exit_does_not() {
    let body = "setmetatable({}, {__gc = function() print('gc') end}) ";
    for output in run_modes("lua54", &[], &format!("{body} os.exit(0, true)")) {
        assert_eq!(output.status.code(), Some(0));
        assert_eq!(stdout(&output), "gc\n");
    }
    for output in run_modes("lua54", &[], &format!("{body} os.exit(0)")) {
        assert_eq!(output.status.code(), Some(0));
        assert_eq!(stdout(&output), "");
    }
}

#[test]
fn an_error_in_a_finalizer_is_ignored_unless_warnings_are_on() {
    let source = "setmetatable({}, {__gc = function() print('first') end}) \
        setmetatable({}, {__gc = function() error('boom') end}) print('end')";
    for output in run_modes("lua54", &[], source) {
        assert!(output.status.success(), "{output:?}");
        assert_eq!(stdout(&output), "end\nfirst\n");
        assert!(output.stderr.is_empty(), "{output:?}");
    }
    for output in run_modes("lua54", &["-W"], source) {
        assert!(output.status.success(), "{output:?}");
        assert_eq!(stdout(&output), "end\nfirst\n");
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        assert!(stderr.contains("error in __gc ("), "{stderr}");
        assert!(stderr.contains("boom)"), "{stderr}");
    }
    let userdata = "local u = newproxy(true) getmetatable(u).__gc = function() error('x') end \
        local v = newproxy(true) getmetatable(v).__gc = function() print('v') end";
    for output in run_modes("lua51", &[], userdata) {
        assert!(output.status.success(), "{output:?}");
        assert_eq!(stdout(&output), "v\n");
        assert!(output.stderr.is_empty(), "{output:?}");
    }
}

#[test]
fn a_finalizer_that_resurrects_its_object_runs_once() {
    let source = "setmetatable({name = 'obj'}, {__gc = function(o) keep = o print('gc', o.name) end}) \
        print('end')";
    for output in run_modes("lua54", &[], source) {
        assert!(output.status.success(), "{output:?}");
        assert_eq!(stdout(&output), "end\ngc\tobj\n");
    }
}

#[test]
fn a_failing_script_still_runs_finalizers() {
    let source = "setmetatable({}, {__gc = function() print('gc') end}) error('bad')";
    for output in run_modes("lua54", &[], source) {
        assert!(!output.status.success(), "{output:?}");
        assert_eq!(stdout(&output), "gc\n");
        assert!(String::from_utf8_lossy(&output.stderr).contains("bad"));
    }
}

#[test]
fn a_finalizer_can_choose_the_status_at_a_normal_end() {
    let source = "setmetatable({}, {__gc = function() os.exit(7, true) end})";
    for output in run_modes("lua54", &[], source) {
        assert_eq!(output.status.code(), Some(7), "{output:?}");
    }
}

#[test]
fn a_script_without_finalizers_ends_normally() {
    for output in run_modes("lua54", &[], "print('hi')") {
        assert!(output.status.success());
        assert_eq!(stdout(&output), "hi\n");
    }
}

#[test]
fn a_non_string_error_in_a_finalizer_warns_without_its_value() {
    let source = "setmetatable({}, {__gc = function() error({}) end}) \
        setmetatable({}, {__gc = function() error(12) end})";
    for output in run_modes("lua54", &["-W"], source) {
        assert!(output.status.success(), "{output:?}");
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        assert_eq!(
            stderr
                .matches("error in __gc (error object is not a string)")
                .count(),
            2,
            "{stderr}"
        );
    }
}

#[test]
fn lua51_orders_userdata_by_creation_not_by_when_gc_was_set() {
    let source = "local a = newproxy(true) local b = newproxy(true) \
        getmetatable(b).__gc = function() print('b') end \
        getmetatable(a).__gc = function() print('a') end print('e')";
    for output in run_modes("lua51", &[], source) {
        assert!(output.status.success(), "{output:?}");
        assert_eq!(stdout(&output), "e\nb\na\n");
    }
}

#[test]
fn the_error_is_reported_before_the_finalizers_run() {
    let source = "setmetatable({}, {__gc = function() print('gc') end}) error('bad')";
    for mode in MODES {
        let command = format!(
            "{} --compat=lua54 --execution={mode} -e \"$SRC\" 2>&1",
            env!("CARGO_BIN_EXE_oximond")
        );
        let output = Command::new("sh")
            .args(["-c", &command])
            .env("SRC", source)
            .output()
            .expect("run sh");
        let merged = stdout(&output);
        let error = merged.find("bad").expect("error line");
        let finalizer = merged.find("gc\n").expect("finalizer output");
        assert!(error < finalizer, "{merged}");
    }
}

#[test]
fn lua54_warns_about_an_error_in_a_collection_and_goes_on() {
    let source = "warn('@on') \
        setmetatable({}, {__gc = function() error('e1') end}) collectgarbage() print('mid') \
        setmetatable({}, {__gc = function() error({}) end}) print(pcall(collectgarbage)) \
        setmetatable({}, {__gc = 1}) collectgarbage() print('end')";
    for profile in ["lua54", "lua55"] {
        for output in run_modes(profile, &[], source) {
            assert!(output.status.success(), "{output:?}");
            assert_eq!(stdout(&output), "mid\ntrue\t0\nend\n");
            assert_eq!(
                String::from_utf8_lossy(&output.stderr),
                "Lua warning: error in __gc ((command line):1: e1)\n\
                 Lua warning: error in __gc (error object is not a string)\n\
                 Lua warning: error in __gc (attempt to call a number value (metamethod '__gc'))\n"
            );
        }
    }
}

#[test]
fn lua51_passes_an_error_in_a_collection_to_the_caller() {
    let source = "local u = newproxy(true) getmetatable(u).__gc = function() error('e1') end \
        u = nil print(pcall(collectgarbage))";
    for output in run_modes("lua51", &[], source) {
        assert!(output.status.success(), "{output:?}");
        assert_eq!(stdout(&output), "false\t(command line):1: e1\n");
    }
}

#[test]
fn warning_control_messages_follow_lauxlib() {
    let source = "warn('off') warn('x', '@on') warn('on') warn('@o', 'n') warn('@foo') \
        warn('@off', 'y') warn('@off') warn('@on', 'z') warn('@on') warn('a', 1)";
    for output in run_modes("lua54", &[], source) {
        assert!(output.status.success(), "{output:?}");
        assert_eq!(
            String::from_utf8_lossy(&output.stderr),
            "Lua warning: on\nLua warning: @on\nLua warning: @offy\nLua warning: a1\n"
        );
    }
}

#[test]
fn lua52_and_lua53_wrap_a_finalizer_error_for_the_collector() {
    let source = "setmetatable({}, {__gc = function() error('boom', 0) end}) \
        print(pcall(collectgarbage)) \
        setmetatable({}, {__gc = function() error({}) end}) \
        print(pcall(collectgarbage))";
    for profile in ["lua52", "lua53"] {
        for output in run_modes(profile, &[], source) {
            assert!(output.status.success(), "{output:?}");
            assert_eq!(
                stdout(&output),
                "false\terror in __gc metamethod (boom)\n\
                 false\terror in __gc metamethod (no message)\n",
                "{profile}: {output:?}"
            );
        }
    }
}

#[test]
fn lua54_does_not_wrap_a_finalizer_error() {
    let source = "setmetatable({}, {__gc = function() error('boom', 0) end}) \
        print(pcall(collectgarbage))";
    for output in run_modes("lua54", &[], source) {
        assert!(output.status.success(), "{output:?}");
        assert_eq!(stdout(&output), "true\t0\n", "{output:?}");
    }
}

#[test]
fn lua52_and_lua53_wrap_a_finalizer_error_past_an_unprotected_collection() {
    let nested = "setmetatable({}, {__gc = function() error('boom', 0) end}) \
        print(pcall(function() collectgarbage() end))";
    let automatic = "setmetatable({}, {__gc = function() error('alloc', 0) end}) \
        print(pcall(function() \
        for i = 1, 2000000 do local t = {i, i} end end))";
    let uncaught = "setmetatable({}, {__gc = function() error('boom', 0) end}) \
        collectgarbage()";
    for profile in ["lua52", "lua53"] {
        for output in run_modes(profile, &[], nested) {
            assert!(output.status.success(), "{output:?}");
            assert_eq!(
                stdout(&output),
                "false\terror in __gc metamethod (boom)\n",
                "{profile}: {output:?}"
            );
        }
        for output in run_modes(profile, &[], automatic) {
            assert!(output.status.success(), "{output:?}");
            assert_eq!(
                stdout(&output),
                "false\terror in __gc metamethod (alloc)\n",
                "{profile}: {output:?}"
            );
        }
        for output in run_modes(profile, &[], uncaught) {
            assert!(!output.status.success(), "{output:?}");
            let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            assert!(
                stderr.contains("error in __gc metamethod (boom)")
                    && !stderr.contains("(error in __gc"),
                "{profile}: {output:?}"
            );
        }
    }
}

#[test]
fn lua52_and_lua53_wrap_a_finalizer_error_once_across_nested_finalizers() {
    // `GCTM` wraps only a LUA_ERRRUN status; a wrapped error travels as
    // LUA_ERRGCMM and the outer finalizer passes it on unchanged.
    let nested = "setmetatable({}, {__gc = function() error('inner', 0) end}) \
        setmetatable({}, {__gc = function() collectgarbage() end}) \
        print(pcall(collectgarbage))";
    // An error raised again from Lua code is a new LUA_ERRRUN, so it wraps again.
    let rethrown = "setmetatable({}, {__gc = function() error('inner', 0) end}) \
        setmetatable({}, {__gc = function() \
        local ok, message = pcall(collectgarbage) error(message, 0) end}) \
        print(pcall(collectgarbage))";
    for profile in ["lua52", "lua53"] {
        for output in run_modes(profile, &[], nested) {
            assert!(output.status.success(), "{output:?}");
            assert_eq!(
                stdout(&output),
                "false\terror in __gc metamethod (inner)\n",
                "{profile}: {output:?}"
            );
        }
        for output in run_modes(profile, &[], rethrown) {
            assert!(output.status.success(), "{output:?}");
            assert_eq!(
                stdout(&output),
                "false\terror in __gc metamethod (error in __gc metamethod (inner))\n",
                "{profile}: {output:?}"
            );
        }
    }
}

#[test]
fn lua52_and_lua53_return_a_wrapped_finalizer_error_from_xpcall_without_its_handler() {
    // `GCTM` raises the wrapped error with `luaD_throw`, which calls no
    // message handler: xpcall returns the error as it stands.
    let source = "local called = false \
        local function handler(m) called = true return 'handled: ' .. tostring(m) end \
        setmetatable({}, {__gc = function() error('boom', 0) end}) \
        print(xpcall(function() collectgarbage() end, handler)) \
        setmetatable({}, {__gc = function() error('direct', 0) end}) \
        print(xpcall(collectgarbage, handler)) \
        print(called)";
    for profile in ["lua52", "lua53"] {
        for output in run_modes(profile, &[], source) {
            assert!(output.status.success(), "{output:?}");
            assert_eq!(
                stdout(&output),
                "false\terror in __gc metamethod (boom)\n\
                 false\terror in __gc metamethod (direct)\n\
                 false\n",
                "{profile}: {output:?}"
            );
        }
    }
}

#[test]
fn lua52_and_lua53_pass_a_memory_error_in_a_finalizer_through_unwrapped() {
    use oximond::{CompatibilityProfile, ExecutionMode, Lua, RuntimeError, Value, Vm, VmError};

    // `GCTM` wraps only a LUA_ERRRUN status.
    fn out_of_memory(_: &mut Vm, _: u64, _: &[Value]) -> Result<Vec<Value>, VmError> {
        Err(RuntimeError::memory_error().into())
    }
    let sources = [
        "setmetatable({}, {__gc = function() out_of_memory() end}) \
         return pcall(collectgarbage)",
        "setmetatable({}, {__gc = out_of_memory}) return pcall(collectgarbage)",
    ];
    for profile in [CompatibilityProfile::Lua52, CompatibilityProfile::Lua53] {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            let callback = lua
                .vm_mut()
                .register_host_function(out_of_memory, 0, Vec::new())
                .unwrap();
            lua.runtime_mut().set_global("out_of_memory", callback);
            for source in sources {
                assert_eq!(
                    lua.exec(source).unwrap(),
                    [Value::Boolean(false), Value::from("not enough memory")],
                    "{profile:?} {mode:?}: {source}"
                );
            }
        }
    }
}

/// The results of `pcall` land above the locals and move into them. While
/// `collectgarbage` runs, the locals' registers still hold temporaries of the
/// statement before, which must not keep the table alive.
#[test]
fn a_collection_under_pcall_finalizes_what_the_locals_registers_held() {
    let source = "setmetatable({}, {__gc = function() error('boom') end}) \
        local ok, err = pcall(collectgarbage) print(ok, err)";
    for profile in ["lua52", "lua53"] {
        for output in run_modes(profile, &[], source) {
            assert!(output.status.success(), "{output:?}");
            assert!(
                stdout(&output).starts_with("false\terror in __gc metamethod ("),
                "{profile}: {output:?}"
            );
        }
    }
    let source = "setmetatable({}, {__gc = function() print('gc') end}) \
        local ok, n = pcall(collectgarbage) print(ok, n)";
    for output in run_modes("lua54", &[], source) {
        assert!(output.status.success(), "{output:?}");
        assert_eq!(stdout(&output), "gc\ntrue\t0\n");
    }
}
