//! `os.exit` ends the `oximond` process with Lua's status in every profile.

use std::env;
use std::fs;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

const PROFILES: [&str; 6] = ["lua51", "lua52", "lua53", "lua54", "lua55", "luajit21"];

fn run(profile: &str, source: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args(["--compat", profile, "-e", source])
        .output()
        .expect("run oximond")
}

fn assert_exit(profile: &str, source: &str, code: i32, stdout: &str) {
    let output = run(profile, source);
    assert_eq!(
        output.status.code(),
        Some(code),
        "{profile}: {source}: {output:?}"
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        stdout,
        "{profile}: {source}"
    );
    assert!(output.stderr.is_empty(), "{profile}: {source}: {output:?}");
}

#[test]
fn os_exit_is_a_function_in_every_profile() {
    for profile in PROFILES {
        assert_exit(profile, "io.write(type(os.exit))", 0, "function");
    }
}

#[test]
fn os_exit_uses_the_numeric_status() {
    for profile in PROFILES {
        assert_exit(profile, "os.exit()", 0, "");
        assert_exit(profile, "os.exit(3)", 3, "");
        assert_exit(profile, "os.exit('4')", 4, "");
        // A process status keeps the low 8 bits, as C's `exit` does.
        assert_exit(profile, "os.exit(-1)", 255, "");
        assert_exit(profile, "os.exit(256 + 7)", 7, "");
    }
}

#[test]
fn os_exit_maps_booleans_to_success_and_failure_after_lua51() {
    for profile in ["lua52", "lua53", "lua54", "lua55", "luajit21"] {
        assert_exit(profile, "os.exit(true)", 0, "");
        assert_exit(profile, "os.exit(false)", 1, "");
    }
    let output = run("lua51", "os.exit(true)");
    assert_eq!(output.status.code(), Some(70), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("bad argument #1 to 'exit' (number expected, got boolean)"),
        "{output:?}"
    );
}

#[test]
fn os_exit_status_conversion_follows_each_profile() {
    for profile in ["lua51", "lua52", "luajit21"] {
        assert_exit(profile, "os.exit(3.7)", 3, "");
    }
    for profile in ["lua53", "lua54", "lua55"] {
        let output = run(profile, "os.exit(3.7)");
        assert_eq!(output.status.code(), Some(70), "{profile}: {output:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("number has no integer representation"),
            "{profile}: {output:?}"
        );
    }
}

#[test]
fn os_exit_flushes_buffered_output() {
    for profile in PROFILES {
        assert_exit(
            profile,
            "io.write('partial') print(' line') io.stdout:write('tail') os.exit(5)",
            5,
            "partial line\ntail",
        );
    }
}

#[test]
fn os_exit_flushes_open_files() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("current time")
        .as_nanos();
    let path = env::temp_dir().join(format!(
        "oximond-os-exit-{}-{nonce}.txt",
        std::process::id()
    ));
    let source = format!(
        "local file = assert(io.open({:?}, 'w')) file:write('kept') os.exit(0)",
        path.to_str().expect("UTF-8 temporary path")
    );
    assert_exit("lua54", &source, 0, "");
    let written = fs::read_to_string(&path);
    let _ = fs::remove_file(&path);
    assert_eq!(written.expect("file written"), "kept");
}

#[test]
fn protected_calls_and_coroutines_do_not_catch_os_exit() {
    for profile in PROFILES {
        assert_exit(profile, "pcall(os.exit, 4) print('after')", 4, "");
        assert_exit(
            profile,
            "print(pcall(function() os.exit(6) end)) print('after')",
            6,
            "",
        );
        assert_exit(
            profile,
            "xpcall(function() error('x') end, function() os.exit(7) end) print('after')",
            7,
            "",
        );
        assert_exit(
            profile,
            "coroutine.resume(coroutine.create(function() os.exit(8) end)) print('after')",
            8,
            "",
        );
    }
}

#[test]
fn closing_exit_runs_finalizers() {
    for profile in ["lua52", "lua53", "lua54", "lua55"] {
        let finalizer = "setmetatable({}, {__gc = function() print('finalized') end}) ";
        assert_exit(
            profile,
            &format!("{finalizer} os.exit(2, true)"),
            2,
            "finalized\n",
        );
        assert_exit(profile, &format!("{finalizer} os.exit(2)"), 2, "");
    }
    let userdata = "local u = newproxy(true) getmetatable(u).__gc = function() print('finalized') end _G.u = u ";
    assert_exit(
        "luajit21",
        &format!("{userdata} os.exit(true, true)"),
        0,
        "finalized\n",
    );
    assert_exit("luajit21", &format!("{userdata} os.exit(true)"), 0, "");
    // Lua 5.1 has no close argument.
    assert_exit("lua51", &format!("{userdata} os.exit(0, true)"), 0, "");
}

#[test]
fn a_finalizer_run_by_closing_exit_can_choose_the_status() {
    let source = "setmetatable({}, {__gc = function() print('first') end}) \
        setmetatable({}, {__gc = function() print('second') os.exit(11, true) end}) \
        os.exit(10, true)";
    assert_exit("lua54", source, 11, "second\nfirst\n");
    let source = "setmetatable({}, {__gc = function() print('first') end}) \
        setmetatable({}, {__gc = function() print('second') os.exit(12) end}) \
        os.exit(10, true)";
    assert_exit("lua54", source, 12, "second\n");
}

#[test]
fn closing_exit_closes_pending_variables_of_the_main_thread() {
    for profile in ["lua54", "lua55"] {
        let source = "local x <close> = setmetatable({}, {__close = function(_, e) print('x', e) end}) \
            local y <close> = setmetatable({}, {__close = function() print('y') end}) \
            os.exit(true, true)";
        assert_exit(profile, source, 0, "y\nx\tnil\n");
        let source = "local x <close> = setmetatable({}, {__close = function() print('x') end}) \
            os.exit(true)";
        assert_exit(profile, source, 0, "");
        // Only the main thread closes, as `lua_close` does.
        let source = "local co = coroutine.wrap(function() \
                local c <close> = setmetatable({}, {__close = function() print('coroutine') end}) \
                os.exit(17, true) \
            end) \
            local m <close> = setmetatable({}, {__close = function() print('main') end}) \
            co()";
        assert_exit(profile, source, 17, "main\n");
    }
}

#[test]
fn os_exit_in_a_close_handler_ends_an_error_unwind() {
    let source = "print(pcall(function() \
            local x <close> = setmetatable({}, {__close = function(_, e) print('closing', e) os.exit(16) end}) \
            error('boom', 0) \
        end)) print('after')";
    assert_exit("lua54", source, 16, "closing\tboom\n");
}

#[test]
fn os_exit_ends_tracing_execution() {
    let source = "local n = 0 \
        for i = 1, 200000 do n = n + i if i == 199999 then pcall(os.exit, 22) end end";
    for profile in PROFILES {
        let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .args(["--compat", profile, "--execution=tracing", "-e", source])
            .output()
            .expect("run oximond");
        assert_eq!(output.status.code(), Some(22), "{profile}: {output:?}");
    }
}

#[test]
fn closing_exit_in_xpcall_handler_closes_parked_targets() {
    for profile in ["lua54", "lua55"] {
        let source = r#"
            local attempts = 0
            xpcall(function()
                local target <close> = setmetatable({}, {__close = function(_, e)
                    print("target", e)
                end})
                error("original", 0)
            end, function(e)
                attempts = attempts + 1
                local attempt = attempts
                local handler <close> = setmetatable({}, {__close = function(_, e)
                    print("handler" .. attempt, e)
                end})
                if attempt == 1 then error("retry", 0) end
                os.exit(23, true)
            end)
            print("after")
        "#;
        assert_exit(
            profile,
            source,
            23,
            "handler2\tnil\nhandler1\tnil\ntarget\tnil\n",
        );
        assert_exit(profile, &source.replace("23, true", "23, false"), 23, "");
    }
}
