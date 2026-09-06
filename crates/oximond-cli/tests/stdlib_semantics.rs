//! Lua-level checks for library functions implemented by the VM.

use oximond::{CompatibilityProfile, ExecutionMode, Lua};

fn run(source: &str) {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua55] {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            if let Err(error) = lua.exec(source) {
                panic!("{profile} {mode}: {error:?}");
            }
        }
    }
}

#[test]
fn string_gsub_handles_templates_limits_positions_and_tables() {
    run(r#"
        local function check(s, p, r, n, es, en)
            local rs, rn = string.gsub(s, p, r, n)
            assert(rs == es, rs)
            assert(rn == en, tostring(rn))
        end
        check("a \t\nb", "%s+", "\n", nil, "a\nb", 1)
        check("a \255 b \t c", "%s+", "\0", 2, "a\0\255\0b \t c", 2)
        check("IDxID", "ID", "", nil, "x", 2)
        check("abc", "a+", "x", nil, "xbc", 1)
        check("ab", "", "-", nil, "-a-b-", 3)
        check("alo alo", "()[al]", "%1", nil, "12o 56o", 4)
        check("abc", "%w", "%1%0", nil, "aabbcc", 3)
        check("alo alo", "(.)", { a = "AA", l = "" }, nil, "AAo AAo", 7)
    "#);
}

#[test]
fn tostring_formats_primitives_and_rejects_missing_arguments() {
    run(r#"
        assert(tostring(nil) == "nil")
        assert(tostring(true) == "true")
        assert(tostring("\0") == "\0")
        assert(string.match(tostring({}), "^table: ") ~= nil)
        local ok, message = pcall(tostring)
        assert(not ok)
        -- Lua 5.1 cannot name a function that pcall calls; Lua 5.5 finds it in _G.
        local name = _VERSION == "Lua 5.1" and "?" or "tostring"
        assert(message == "bad argument #1 to '" .. name .. "' (value expected)", message)
    "#);
}

#[test]
fn pcall_runs_debug_builtins_that_need_the_frame_stack() {
    run(r#"
        local x = 10
        assert(pcall(debug.sethook, function() end, "", 1000))
        local ok, hook, mask, count = pcall(debug.gethook)
        assert(ok and type(hook) == "function" and mask == "" and count == 1000)
        debug.sethook()
        local ok, hook = pcall(debug.gethook)
        assert(ok and hook == nil)

        -- pcall is a C function at level 1, so level 2 is its caller.
        local ok, name, value = pcall(debug.getlocal, 2, 1)
        assert(ok and name == "x" and value == 10)
        local ok, name = pcall(debug.getlocal, 1, 1)
        assert(ok and name == nil)
        local ok, name = pcall(debug.setlocal, 2, 1, 20)
        assert(ok and name == "x" and x == 20)

        local ok, registry = pcall(debug.getregistry)
        assert(ok and registry == debug.getregistry())
        local ok, name, value = pcall(debug.getupvalue, function() return x end, 1)
        assert(ok and name == "x" and value == 20)
    "#);
}

#[test]
fn pcall_sethook_keeps_the_direct_call_hook_semantics() {
    run(r#"
        local events = {}
        local function f() return 1 end
        assert(pcall(debug.sethook, function(event) events[#events + 1] = event end, "c"))
        f()
        debug.sethook()
        assert(events[1] == "call")

        local count = 0
        assert(pcall(debug.sethook, function() count = count + 1 end, "", 100))
        for _ = 1, 1000 do end
        debug.sethook()
        assert(count > 0)

        assert(not pcall(debug.sethook, 42, "c"))

        -- An active call hook defers sethook until the hook returns.
        debug.sethook(function() end, "c")
        assert(pcall(debug.sethook))
        assert(debug.gethook() == nil)

        -- A deferred sethook still reports its errors to pcall.
        debug.sethook(function() end, "c")
        local ok = pcall(debug.sethook, 42, "c")
        local kept = debug.gethook()
        debug.sethook()
        assert(not ok and type(kept) == "function")
    "#);
}

#[test]
fn builtins_that_need_the_frame_stack_run_from_library_callbacks() {
    run(r#"
        local a, b, c = pcall(pcall, error, "e")
        assert(a == true and b == false and c == "e")
        local a, b, c = pcall(xpcall, error, function() return "handled" end)
        assert(a == true and b == false and c == "handled")
        local ok, value = xpcall(function() error("boom", 0) end, pcall)
        assert(ok == false and value == false)

        assert(xpcall(debug.gethook, print))
        assert(select(2, xpcall(debug.getregistry, print)) == debug.getregistry())
        local ok, traceback = xpcall(debug.traceback, print)
        assert(ok and traceback:find("stack traceback:", 1, true))

        assert(pcall(table.sort, {3, 1, 2}, debug.gethook))
        assert(not pcall(table.sort, {3, 1, 2}, table.sort))
        assert(not pcall(string.gsub, "ab", ".", debug.getregistry))
        -- The frame that runs the builtin reads as the C function calling it.
        local traceback = string.gsub("a", ".", debug.traceback)
        assert(traceback:find("^a\nstack traceback:\n\t%[C%]: in "))
        -- The main chunk is listed once, followed by the host's own level as
        -- stock Lua lists it.
        assert(select(2, traceback:gsub("main chunk", "")) == 1, traceback)
        local t = setmetatable({}, {__unm = debug.traceback})
        assert(-t == t)
    "#);
}

#[test]
fn error_accepts_integer_levels() {
    run(r#"
        local ok, message = pcall(function() error("plain", 0) end)
        assert(not ok and message == "plain", message)
        local function raise() error("caller", 2) end
        ok, message = pcall(function() raise() end)
        assert(not ok and message:find(":%d+: caller$"), message)
    "#);
}

#[test]
fn xpcall_handler_inspects_failed_frames() {
    for profile in [CompatibilityProfile::Lua52, CompatibilityProfile::Lua53] {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            lua.exec(
                r#"
                local function fail() error("boom") end
                local ok, value = xpcall(fail, function(e)
                    collectgarbage()
                    assert(debug.getinfo(2, "f").func == error)
                    assert(debug.getinfo(3, "fSL").func == fail)
                    assert(debug.getinfo(4, "f").func == xpcall)
                    return e
                end)
                assert(not ok and value ~= "error in error handling", value)
            "#,
            )
            .unwrap();
        }
    }
}
