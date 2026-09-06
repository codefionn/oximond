//! Globals of Lua 5.2+ chunks read and write `_ENV` with constant-key field
//! instructions, so they reach the per-site inline caches. The semantics of
//! `_ENV` must stay exactly those of the table instructions they replaced.

use oximond::Lua;
use oximond::bytecode::{Chunk, OpCode};
use oximond::compat::{CompatibilityProfile, ExecutionMode};
use oximond::compiler::{CompilerConfig, compile_with_config};
use std::fmt::Write;

const ENV_PROFILES: [CompatibilityProfile; 4] = [
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
];

const MODES: [ExecutionMode; 2] = [ExecutionMode::Interpreter, ExecutionMode::Tracing];

fn compile(source: &str, profile: CompatibilityProfile) -> Chunk {
    compile_with_config(source, CompilerConfig::new(profile)).expect("compile")
}

/// Locals for the library names the sources use, written on the first line so
/// that line numbers stay those of the source: `_ENV` is replaced in places.
const PRELUDE: &str = "local assert, pcall, load, select, setmetatable, rawget, debug, table, \
    coroutine, print = assert, pcall, load, select, setmetatable, rawget, debug, table, \
    coroutine, print ";

fn run(source: &str, profiles: &[CompatibilityProfile]) {
    let source = &format!("{PRELUDE}{source}");
    for &profile in profiles {
        for mode in MODES {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            if let Err(error) = lua.exec(source) {
                panic!("under {profile:?} in {mode:?}: {error:?}\n{source}");
            }
        }
    }
}

fn opcodes(chunk: &Chunk) -> Vec<OpCode> {
    chunk.code.iter().filter_map(|i| i.opcode().ok()).collect()
}

#[test]
fn globals_compile_to_field_instructions_on_the_environment() {
    let source = "x = 1\nlocal y = x + 1\nprint(y)\n";
    for profile in ENV_PROFILES {
        let ops = opcodes(&compile(source, profile));
        assert!(ops.contains(&OpCode::GetTabUp), "{profile:?}: {ops:?}");
        assert!(ops.contains(&OpCode::SetTabUp), "{profile:?}: {ops:?}");
        for gone in [
            OpCode::GetTable,
            OpCode::SetTable,
            OpCode::GetUpvalue,
            OpCode::LoadConstant,
        ] {
            let expected = gone == OpCode::LoadConstant;
            assert_eq!(ops.contains(&gone), expected, "{profile:?}: {ops:?}");
        }
    }
    // A local `_ENV` is a register: plain field instructions.
    let ops = opcodes(&compile(
        "local _ENV = {}\nx = 1\nreturn x\n",
        CompatibilityProfile::Lua54,
    ));
    assert!(
        ops.contains(&OpCode::SetField) && ops.contains(&OpCode::GetField),
        "{ops:?}"
    );
    assert!(!ops.contains(&OpCode::GetTabUp) && !ops.contains(&OpCode::SetTabUp));
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        let ops = opcodes(&compile(source, profile));
        assert!(ops.contains(&OpCode::GetGlobal), "{profile:?}");
        assert!(ops.contains(&OpCode::SetGlobal), "{profile:?}");
    }
}

#[test]
fn local_environment_and_parameters() {
    run(
        r"
        local function f(_ENV) return a + b end
        assert(f({a = 1, b = 2}) == 3)
        do
            local _ENV = {a = 10}
            b = 5
            assert(a == 10 and b == 5 and _ENV.b == 5)
        end
        assert(b == nil)
        local function g()
            local _ENV <const> = {v = 7}
            return function() return v end
        end
        assert(g()() == 7)
        ",
        &[CompatibilityProfile::Lua54, CompatibilityProfile::Lua55],
    );
    run(
        r"
        local function f(_ENV) return a + b end
        assert(f({a = 1, b = 2}) == 3)
        local _ENV = {a = 10}
        b = 5
        assert(a == 10 and b == 5)
        local function h() b = b + 1; return b end
        assert(h() == 6 and _ENV.b == 6)
        ",
        &ENV_PROFILES,
    );
}

#[test]
fn reassigned_environment_is_followed_by_every_site() {
    run(
        r#"
        local seen = {}
        local one, two = {x = "one"}, {x = "two"}
        local _ENV = one
        local function read() return x end
        local function write(v) x = v end
        for i = 1, 200 do
            _ENV = (i % 2 == 0) and two or one
            write(i)
            assert(read() == i)
            assert(_ENV.x == i)
        end
        assert(one.x == 199 and two.x == 200)
        "#,
        &ENV_PROFILES,
    );
    run(
        r"
        local t1, t2 = {v = 1}, {v = 2}
        local function f() return v end
        local sum = 0
        for i = 1, 100 do
            local function swap(e) _ENV = e end
            swap(i % 2 == 0 and t1 or t2)
            sum = sum + f()
        end
        ",
        &ENV_PROFILES,
    );
}

#[test]
fn multiple_assignment_with_a_changing_environment() {
    run(
        r"
        local a, b = {}, {}
        local _ENV = a
        local function set() _ENV = b end
        x, y = 1, 2
        assert(a.x == 1 and a.y == 2)
        set()
        x = 3
        assert(b.x == 3 and a.x == 1)
        ",
        &ENV_PROFILES,
    );
}

#[test]
fn multiple_assignment_stores_into_the_environment_it_started_with() {
    run(
        r#"
        local old, t = _ENV, {}
        _ENV, y = t, 2
        assert(rawget(old, "y") == 2 and t.y == nil)
        _ENV = old
        y = nil
        local t2 = {}
        y2, _ENV, y3 = 1, t2, 3
        assert(rawget(old, "y2") == 1 and rawget(old, "y3") == 3)
        assert(t2.y2 == nil and t2.y3 == nil)
        _ENV = old
        y2, y3 = nil, nil
        local t3 = {}
        local function nested()
            _ENV, z = t3, 5
            return _ENV
        end
        assert(nested() == t3)
        assert(rawget(old, "z") == 5 and t3.z == nil)
        _ENV = old
        z = nil
        local t4 = {}
        local function nested2()
            z1, _ENV, z2 = 1, t4, 2
        end
        nested2()
        assert(rawget(old, "z1") == 1 and rawget(old, "z2") == 2 and t4.z2 == nil)
        _ENV = old
        z1, z2 = nil, nil
        "#,
        &ENV_PROFILES,
    );
}

#[test]
fn tracebacks_through_global_metamethods_name_the_event() {
    run(
        r#"
        setmetatable(_G, {
            __index = function() return debug.traceback() end,
            __newindex = function() error(debug.traceback(), 0) end,
        })
        local tb = undefined_read
        assert(tb:find("in metamethod 'index'", 1, true), tb)
        local ok, message = pcall(function() undefined_write = 1 end)
        assert(message:find("in metamethod 'newindex'", 1, true), message)
        setmetatable(_G, nil)
        "#,
        &[CompatibilityProfile::Lua54, CompatibilityProfile::Lua55],
    );
}

#[test]
fn tracebacks_show_native_global_metamethods() {
    // Lua 5.5 names the level after the event, 5.4 after the function.
    for (profile, level) in [
        (CompatibilityProfile::Lua54, "[C]: in function 'error'"),
        (CompatibilityProfile::Lua55, "[C]: in metamethod 'index'"),
    ] {
        let write_level = level.replace("'index'", "'newindex'");
        run(
            &format!(
                r#"
        setmetatable(_G, {{__index = error}})
        local ok, tb = xpcall(function() return undefined_read end, debug.traceback)
        assert(not ok and tb:find("{level}", 1, true), tb)
        setmetatable(_G, {{__newindex = error}})
        local ok, tb = xpcall(function() undefined_write = 1 end, debug.traceback)
        assert(not ok and tb:find("{write_level}", 1, true), tb)
        setmetatable(_G, nil)
        "#
            ),
            &[profile],
        );
    }
}

#[test]
fn globals_table_metamethods() {
    run(
        r#"
        local log, store = {}, {}
        setmetatable(_G, {
            __index = function(_, k) log[#log + 1] = "get " .. k; return store[k] end,
            __newindex = function(_, k, v) log[#log + 1] = "set " .. k; store[k] = v end,
        })
        for i = 1, 100 do
            undefined_a = i
            assert(undefined_a == i)
        end
        assert(#log == 200, #log)
        assert(rawget(_G, "undefined_a") == nil)
        setmetatable(_G, nil)
        undefined_a = 5
        assert(rawget(_G, "undefined_a") == 5)
        setmetatable(_G, {__index = {fallback = 42}})
        for i = 1, 50 do assert(fallback == 42) end
        fallback = 1
        assert(fallback == 1)
        setmetatable(_G, nil)
        assert(fallback == 1)
        fallback = nil
        assert(fallback == nil)
        "#,
        &ENV_PROFILES,
    );
}

#[test]
fn a_global_assigned_in_a_hot_loop_follows_deletion() {
    run(
        r"
        total = 0
        for i = 1, 500 do
            total = total + i
            if i == 250 then total = nil; total = 0 end
        end
        assert(total == (500 * 501 // 2) - (250 * 251 // 2))
        ",
        &[
            CompatibilityProfile::Lua53,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::Lua55,
        ],
    );
}

#[test]
fn load_with_a_custom_environment() {
    run(
        r#"
        local env = {y = 3}
        local f = load("x = y * 2; return x", "chunk", "t", env)
        for i = 1, 100 do assert(f() == 6) end
        assert(env.x == 6 and x == nil)
        env.y = 5
        assert(f() == 10)
        local f2 = load("return z")
        assert(f2() == nil)
        z = 9
        assert(f2() == 9)
        local g = load("return a")
        assert(g() == nil)
        "#,
        &ENV_PROFILES,
    );
}

#[test]
fn debug_setupvalue_on_the_environment() {
    run(
        r#"
        local f = load("return v")
        assert(f() == nil)
        v = "global"
        assert(f() == "global")
        local name = debug.getupvalue(f, 1)
        assert(name == "_ENV")
        debug.setupvalue(f, 1, {v = "other"})
        for i = 1, 100 do assert(f() == "other") end
        debug.setupvalue(f, 1, {v = "third"})
        assert(f() == "third")
        local function reads() return v end
        local up
        for i = 1, 10 do
            local n = debug.getupvalue(reads, i)
            if n == "_ENV" then up = i end
        end
        assert(up)
        local old = select(2, debug.getupvalue(reads, up))
        debug.setupvalue(reads, up, {v = "swapped"})
        assert(reads() == "swapped")
        debug.setupvalue(reads, up, old)
        assert(reads() == "global")
        "#,
        &ENV_PROFILES,
    );
}

#[test]
fn error_messages_name_the_global() {
    run(
        r#"
        local function msg(f)
            local ok, err = pcall(f)
            assert(not ok)
            return err
        end
        assert(msg(function() missing_function() end):find("global 'missing_function'", 1, true))
        assert(msg(function() return missing_table.field end):find("global 'missing_table'", 1, true))
        assert(msg(function() missing_table.field = 1 end):find("global 'missing_table'", 1, true))
        assert(msg(function() return missing_table + 1 end):find("global 'missing_table'", 1, true))
        local _ENV = nil
        assert(select(2, pcall(function() return x end)):find("upvalue '_ENV'", 1, true)
            or select(2, pcall(function() return x end)):find("index a nil value", 1, true))
        "#,
        &ENV_PROFILES,
    );
}

#[test]
fn indexing_a_non_table_environment_fails_like_before() {
    run(
        r#"
        local function try(env)
            local f = load("return x", "=c", "t", env)
            return pcall(f)
        end
        local ok, err = try(5)
        assert(not ok and err:find("attempt to index a number value", 1, true), err)
        ok, err = try("abc")
        assert(ok and err == nil)
        local ok2, err2 = pcall(load("x = 1", "=c", "t", 5))
        assert(not ok2 and err2:find("attempt to index a number value", 1, true), err2)
        "#,
        &ENV_PROFILES,
    );
}

#[test]
fn global_function_definitions_and_declarations() {
    run(
        r"
        function f(a) return a + 1 end
        function t_f() return 1 end
        assert(f(1) == 2 and _G.f == f and t_f() == 1)
        for i = 1, 50 do
            function f(a) return a + i end
            assert(f(1) == 1 + i)
        end
        ",
        &ENV_PROFILES,
    );
    run(
        r#"
        global a, b = 1, 2
        assert(a == 1 and b == 2)
        global function gf() return a + b end
        assert(gf() == 3)
        global <const> c = 5
        a = 10
        assert(a == 10 and gf() == 12 and c == 5)
        for i = 1, 100 do a = a + 1 end
        assert(a == 110)
        assert(not load("global <const> k = 1; k = 2"))
        assert(not load("global none; undeclared_name = 1"))
        local ok = load("global x; x = 1; return x")
        assert(ok() == 1)
        "#,
        &[CompatibilityProfile::Lua55],
    );
}

#[test]
fn nested_functions_share_the_environment_upvalue() {
    run(
        r"
        local function outer()
            local function inner() counter = (counter or 0) + 1; return counter end
            return inner
        end
        local inner = outer()
        for i = 1, 100 do assert(inner() == i) end
        assert(counter == 100)
        local co = coroutine.wrap(function()
            for i = 1, 3 do shared = i; coroutine.yield(shared) end
        end)
        assert(co() == 1 and co() == 2 and co() == 3)
        ",
        &ENV_PROFILES,
    );
}

#[test]
fn line_hooks_report_the_same_lines() {
    run(
        r#"
        local lines = {}
        debug.sethook(function(_, line) lines[#lines + 1] = line end, "l")
        g1 = 1
        g2 = g1 +
            1
        local x = g2
        debug.sethook()
        assert(table.concat(lines, ",") == "4,5,7,8", table.concat(lines, ","))
        "#,
        &ENV_PROFILES,
    );
}

/// Count and line hooks see a loop over globals the same in both modes: the
/// count hook fires at the same global read or write, and a line hook (which
/// keeps traces out) reports the same lines.
#[test]
fn count_and_line_hooks_over_global_loops_agree_in_both_modes() {
    // A trace needs a long run to repay itself; traces never run under a
    // line hook, so those loops stay short.
    for (mask, count, iterations) in [
        ("", 61, 200_000),
        ("", 997, 200_000),
        ("l", 0, 3000),
        ("l", 13, 3000),
    ] {
        let source = format!(
            r#"
            local log = {{}}
            g_total, g_step = 0, 2
            debug.sethook(function(event, line)
                log[#log + 1] = event .. ":" .. (line or debug.getinfo(2, "l").currentline)
                    .. ":" .. tostring(g_total)
            end, "{mask}", {count})
            for i = 1, {iterations} do
                g_total = (g_total + i % 5 + g_step) % 10007
            end
            debug.sethook()
            return g_total, #log, table.concat(log, ";")
            "#
        );
        for profile in ENV_PROFILES {
            let (expected, _) = result_and_trace(profile, ExecutionMode::Interpreter, &source);
            let (traced, stats) = result_and_trace(profile, ExecutionMode::Tracing, &source);
            assert_eq!(expected, traced, "{profile:?} mask {mask:?} count {count}");
            if mask.is_empty() && count == 997 {
                assert!(stats.native_entries > 0, "{profile:?}: {stats:?}");
            }
        }
    }
}

#[test]
fn a_wide_constant_pool_still_reaches_globals() {
    let mut source = String::from("local t = {}\n");
    for i in 0..400 {
        writeln!(source, "t[{i}] = \"k{i}\"").expect("formatting into a String cannot fail");
    }
    for i in 0..300 {
        writeln!(source, "g{i} = {i}").expect("formatting into a String cannot fail");
    }
    source.push_str("assert(g299 == 299 and g0 == 0 and g150 == 150)\n");
    run(&source, &ENV_PROFILES);
}

fn result_and_trace(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
) -> (String, oximond::vm::TraceStatistics) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let results: Vec<oximond::Value> = lua
        .exec(source)
        .unwrap_or_else(|error| panic!("{profile:?} {mode:?}: {error}\n{source}"));
    (format!("{results:?}"), lua.vm().jit_statistics().trace)
}

/// Runs `source` in both modes, checks they agree and, when `native`, that the
/// tracing run entered native code.
fn same_in_both_modes(source: &str, native: bool) {
    for profile in ENV_PROFILES {
        let (expected, _) = result_and_trace(profile, ExecutionMode::Interpreter, source);
        let (traced, stats) = result_and_trace(profile, ExecutionMode::Tracing, source);
        assert_eq!(expected, traced, "{profile:?}\n{source}");
        if native {
            assert!(
                stats.native_entries > 0,
                "{profile:?} ran no trace\n{source}"
            );
            assert!(stats.iterations > 100, "{profile:?}: {stats:?}\n{source}");
        }
    }
}

#[test]
fn hot_loops_over_globals_run_natively() {
    same_in_both_modes(
        r"
        total, step = 0, 3
        for i = 1, 200000 do total = total + step end
        return total
        ",
        true,
    );
    same_in_both_modes(
        r"
        local function run()
            count = 0
            for i = 1, 200000 do count = count + i % 7 end
            return count
        end
        return run()
        ",
        true,
    );
}

#[test]
fn a_trace_over_globals_leaves_when_the_environment_changes() {
    // The global table gets a metatable, the key goes away, the type
    // changes, and `_ENV` itself is replaced, each in the middle of the loop.
    same_in_both_modes(
        r"
        value = 1
        local seen = 0
        for i = 1, 4000 do
            if i == 2000 then value = 2.5 end
            if i == 2500 then value = nil; setmetatable(_G, {__index = function() return 7 end}) end
            seen = seen + (value or 0)
        end
        setmetatable(_G, nil)
        return seen
        ",
        false,
    );
    same_in_both_modes(
        r#"
        local env = {x = 1}
        local function body(n)
            local sum = 0
            for i = 1, n do
                sum = sum + x
                if i == 3000 then env = {x = 10} end
            end
            return sum
        end
        local first = load("local env = ...; return function(n) local _ENV = env; local s = 0; for i = 1, n do s = s + x end return s end")
        return first(env)(5000)
        "#,
        false,
    );
    same_in_both_modes(
        r"
        local function counter()
            hits = (hits or 0) + 1
            return hits
        end
        for i = 1, 4000 do counter() end
        return hits
        ",
        false,
    );
}
