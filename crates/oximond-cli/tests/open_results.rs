//! Calls with an open result count (`f(g())`, `return g()`, `{g()}`) for
//! native and Lua `g`, with zero, one, two, and many results. The fast loop
//! keeps small result lists in the register window and runs natives and
//! plain Lua callees on them without leaving its dispatch; lists that do not
//! fit the window take the vector path. Each program checks itself with
//! `assert`, so its expectations hold for every profile and execution mode.

use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn run(source: &str) {
    for profile in PROFILES {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            lua.vm_mut().use_synchronous_jit_compilation();
            if let Err(error) = lua.exec(source) {
                panic!("{profile} {mode}: {error:?}");
            }
        }
    }
}

const PRELUDE: &str = r##"
    local unpack = table.unpack or unpack
    local values = {}
    for i = 1, 300 do values[i] = i end
    -- A Lua function with the count and the sum of what it received.
    local function summary(...)
        local n, total = select("#", ...), 0
        for i = 1, n do total = total + (select(i, ...)) end
        return n, total
    end
    local function lua_results(n) local t = {unpack(values, 1, n)} return unpack(t, 1, n) end
    local function lua_tail(n) return unpack(values, 1, n) end
    local COUNTS = {0, 1, 2, 3, 17, 40, 255, 300}
"##;

#[test]
fn open_results_feed_a_lua_callee_from_native_and_lua_producers() {
    run(&format!(
        "{PRELUDE}{}",
        r"
        for _ = 1, 3 do
            for _, n in ipairs(COUNTS) do
                local count, total = summary(unpack(values, 1, n))
                assert(count == n and total == n * (n + 1) / 2, n)
                count, total = summary(lua_results(n))
                assert(count == n and total == n * (n + 1) / 2, n)
                count, total = summary(lua_tail(n))
                assert(count == n and total == n * (n + 1) / 2, n)
                -- Fixed arguments before the open ones.
                count, total = summary(1000, unpack(values, 1, n))
                assert(count == n + 1 and total == n * (n + 1) / 2 + 1000, n)
                local first, second, third = lua_results(n)
                assert(first == (n > 0 and 1 or nil) and second == (n > 1 and 2 or nil))
                assert(third == (n > 2 and 3 or nil))
            end
        end
        "
    ));
}

#[test]
fn open_results_feed_a_native_callee() {
    run(&format!(
        "{PRELUDE}{}",
        r##"
        local select, tostring, tonumber = select, tostring, tonumber
        local max, floor, format = math.max, math.floor, string.format
        local function two() return 7, 8 end
        local function none() end
        local function nat_none() return table.insert({}, 1) end
        for _ = 1, 3 do
            for _, n in ipairs(COUNTS) do
                assert(select("#", unpack(values, 1, n)) == n, n)
                assert(select("#", lua_results(n)) == n, n)
                assert(select("#", lua_tail(n)) == n, n)
                assert(select("#", 1, lua_results(n)) == n + 1, n)
                if n > 0 then
                    assert(max(unpack(values, 1, n)) == n, n)
                    assert(max(lua_results(n)) == n, n)
                    assert(select(n, unpack(values, 1, n)) == n)
                end
            end
            assert(select("#", none()) == 0 and select("#", nat_none()) == 0)
            assert(format("%d %d", two()) == "7 8")
            assert(format("%d %d %d", floor(7.5), two()) == "7 7 8")
            assert(format("%d-%d", 5, floor(7.5)) == "5-7")
            assert(tonumber(floor(41.5)) == 41)
            assert(tostring(floor(2.5)) == "2")
            assert(format("%s|%s", "a", unpack({"b", "c"})) == "a|b")
        end
        "##
    ));
}

#[test]
fn table_constructors_take_every_open_result() {
    run(&format!(
        "{PRELUDE}{}",
        r#"
        local function none() end
        for _ = 1, 3 do
            for _, n in ipairs(COUNTS) do
                local t = {unpack(values, 1, n)}
                assert(#t == n and (n == 0 or t[n] == n), n)
                t = {lua_results(n)}
                assert(#t == n and (n == 0 or t[n] == n), n)
                t = {0, lua_tail(n)}
                assert(#t == n + 1 and t[1] == 0 and (n == 0 or t[n + 1] == n), n)
                t = {x = 1, lua_tail(n), "end"}
                assert(t.x == 1, n)
                if n > 0 then assert(t[1] == 1 and t[2] == "end", n) else assert(t[1] == nil and t[2] == "end", n) end
            end
            assert(next({none()}) == nil)
        end
        "#
    ));
}

#[test]
fn tail_calls_return_the_results_of_native_and_lua_callees() {
    run(&format!(
        "{PRELUDE}{}",
        r##"
        local function tail_native(n) return unpack(values, 1, n) end
        local function tail_lua(n) return lua_results(n) end
        local function tail_fixed(a, b) return math.max(a, b) end
        local function tail_one(x) return tonumber(x) end
        local function tail_none(t) return table.insert(t, 1) end
        local function tail_deep(n) return tail_native(n) end
        local function tail_selfless(...) return select("#", ...) end
        for _ = 1, 3 do
            for _, n in ipairs(COUNTS) do
                assert(select("#", tail_native(n)) == n, n)
                assert(select("#", tail_lua(n)) == n, n)
                assert(select("#", tail_deep(n)) == n, n)
                local count, total = summary(tail_native(n))
                assert(count == n and total == n * (n + 1) / 2, n)
                count, total = summary(tail_lua(n))
                assert(count == n and total == n * (n + 1) / 2, n)
                local last = select(n == 0 and 1 or n, tail_native(n))
                if n == 0 then assert(last == nil) else assert(last == n) end
                assert(tail_selfless(tail_native(n)) == n)
            end
            assert(tail_fixed(3, 9) == 9 and tail_one("12") == 12 and tail_one("z") == nil)
            assert(select("#", tail_none({})) == 0)
            local a, b, c = tail_native(2)
            assert(a == 1 and b == 2 and c == nil)
        end
        "##
    ));
}

#[test]
fn open_results_survive_captured_locals_and_nesting() {
    run(&format!(
        "{PRELUDE}{}",
        r##"
        local counter = 0
        local function bump() counter = counter + 1 return counter end
        local function pair() counter = counter + 1 return counter, counter * 2 end
        local function reader() return counter end
        for i = 1, 20 do
            local before = counter
            local n, total = summary(pair())
            assert(n == 2 and total == (before + 1) * 3 and reader() == before + 1)
            local x = tostring(bump())
            assert(x == tostring(before + 2) and reader() == before + 2)
            assert(math.max(bump(), pair()) >= counter)
            local nested = select("#", summary(unpack(values, 1, i)))
            assert(nested == 2)
            assert(select("#", select(1, unpack(values, 1, i))) == i)
            assert(select("#", select(2, unpack(values, 1, i))) == i - 1)
        end
        -- Results flow through a vararg function and out again.
        local function through(...) return ... end
        for _, n in ipairs(COUNTS) do
            assert(select("#", through(unpack(values, 1, n))) == n, n)
            assert(select("#", through(through(lua_results(n)))) == n, n)
        end
        "##
    ));
}

/// Outside `LuaJIT`, where a tail call names its function and gives the
/// position like any other. `LuaJIT` names '?' and gives none, as stock does.
#[test]
fn a_native_in_tail_position_reports_the_same_error_as_one_outside() {
    run(r#"
        local function strip(message) return (message:gsub("^[^:]*:%d+: ", "")) end
        local function bad_tail() return string.rep() end
        local function bad_plain() local x = string.rep() return x end
        local function bad_tostring() return tostring() end
        local function bad_nested() return bad_tail() end
        local function lines(trace)
            local count = 0
            for _ in trace:gmatch("\n") do count = count + 1 end
            return count
        end
        for _ = 1, 3 do
            local ok_tail, message_tail = pcall(bad_tail)
            local ok_plain, message_plain = pcall(bad_plain)
            assert(not ok_tail and not ok_plain)
            if jit then
                -- Stock LuaJIT names '?' and gives no position in a tail call.
                assert(message_tail == "bad argument #1 to '?' (string expected, got no value)", message_tail)
                assert(strip(message_plain) == "bad argument #1 to 'rep' (string expected, got no value)", message_plain)
                assert(message_plain ~= strip(message_plain), message_plain)
            else
                assert(strip(message_tail) == strip(message_plain), message_tail)
            end
            assert(message_tail:find("bad argument #1 to '", 1, true), message_tail)
            assert(select(2, pcall(bad_tostring)):find("bad argument #1 to '", 1, true))
            local _, trace_tail = xpcall(bad_tail, debug.traceback)
            local _, trace_plain = xpcall(bad_plain, debug.traceback)
            -- Stock LuaJIT reuses the frame of a function that tail-calls a
            -- native, so that function is missing from the trace.
            local expected = jit and lines(trace_plain) - 1 or lines(trace_plain)
            assert(lines(trace_tail) == expected, trace_tail .. "\n" .. trace_plain)
            local _, trace_nested = xpcall(bad_nested, debug.traceback)
            -- Stock Lua names no tail-called function. Lua 5.1 lists the
            -- replaced level, 5.2 and later mark it, LuaJIT keeps no trace.
            assert(jit or trace_nested:find("tail call", 1, true), trace_nested)
        end
        "#);
}

#[test]
fn open_results_cross_coroutines_protected_calls_and_hooks() {
    run(&format!(
        "{PRELUDE}{}",
        r##"
        local function count(...) return select("#", ...) end
        local co = coroutine.create(function()
            local n = count(coroutine.yield("a"))
            local m = count(coroutine.yield())
            local s = string.format("%d-%d", coroutine.yield())
            return n, m, s
        end)
        local ok, v = coroutine.resume(co)
        assert(ok and v == "a")
        ok, v = coroutine.resume(co, 1, 2, 3)
        assert(ok and v == nil)
        ok = coroutine.resume(co, 4, 5)
        assert(ok)
        local ok, n, m, s = coroutine.resume(co, 7, 8)
        assert(ok and n == 3 and m == 2 and s == "7-8")

        for _, n in ipairs(COUNTS) do
            local ok, got = pcall(count, unpack(values, 1, n))
            assert(ok and got == n, n)
            -- Lua 5.1's xpcall passes no arguments to the function.
            if not jit and _VERSION == "Lua 5.1" then
                ok, got = xpcall(function() return count(lua_results(n)) end, print)
            else
                ok, got = xpcall(count, print, lua_results(n))
            end
            assert(ok and got == n, n)
            ok, got = pcall(function() return count(lua_tail(n)) end)
            assert(ok and got == n, n)
        end

        -- An error in a native fed by open results names the same call.
        local function rep_none() return string.rep(unpack({})) end
        local ok, message = pcall(rep_none)
        assert(not ok and message:find("bad argument #1 to '", 1, true), message)
        local ok, message = pcall(function() local t = nil; return count(t.x, unpack(values, 1, 3)) end)
        assert(not ok and message:find("index", 1, true), message)

        -- A hook between the producer and the consumer changes nothing.
        local seen = 0
        debug.sethook(function() seen = seen + 1 end, "l")
        for _, n in ipairs(COUNTS) do
            assert(count(unpack(values, 1, n)) == n, n)
            assert(count(lua_results(n)) == n, n)
            assert(select("#", lua_tail(n)) == n, n)
        end
        debug.sethook()
        assert(seen > 0)
        "##
    ));
}
