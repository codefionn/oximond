//! Lua-to-Lua calls and returns that the fast interpreter loop performs
//! without leaving its dispatch. Each program checks itself with `assert`, so
//! its expectations hold for every profile and execution mode. The stack
//! depth and the traceback text are the values stock behaviour fixed before
//! the fast loop pushed activations itself.

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
            let source = source.replace("EXPECTED_DEPTH", "4095");
            if let Err(error) = lua.exec(source) {
                panic!("{profile} {mode}: {error:?}");
            }
        }
    }
}

#[test]
fn deep_recursion_overflows_at_the_same_depth_with_the_same_message() {
    run(r#"
        local depth = 0
        local function plain() depth = depth + 1; return 1 + plain() end
        local ok, message = pcall(plain)
        assert(not ok and message:find("stack overflow", 1, true), message)
        assert(depth == EXPECTED_DEPTH, "depth " .. depth)

        -- A call whose result is stored, not added, takes the same path.
        depth = 0
        local function stored(n) depth = depth + 1; local x = stored(n + 1); return x end
        ok, message = pcall(stored, 1)
        assert(not ok and message:find("stack overflow", 1, true), message)
        assert(depth == EXPECTED_DEPTH, "depth " .. depth)

        -- The stack is usable again after the overflow.
        local function down(n) if n == 0 then return 0 end return 1 + down(n - 1) end
        assert(down(1000) == 1000)
    "#);
}

#[test]
fn calls_with_fewer_and_more_arguments_than_parameters() {
    run(r#"
        local function three(a, b, c) return a, b, c end
        local x1, x2, x3, x4 = three(1)
        assert(x1 == 1 and x2 == nil and x3 == nil and x4 == nil)
        local y1, y2, y3, y4 = three(1, 2, 3, 4, 5, 6)
        assert(y1 == 1 and y2 == 2 and y3 == 3 and y4 == nil)
        local function sum(a, b) return (a or 0) + (b or 0) end
        assert(sum() == 0 and sum(1) == 1 and sum(1, 2) == 3 and sum(1, 2, 3) == 3)

        -- Leftover registers of an earlier call must not leak into the next.
        local function locals(a)
            local b, c, d
            local seen = (b == nil) and (c == nil) and (d == nil)
            b, c, d = a, a, a
            return seen
        end
        for i = 1, 5 do assert(locals(i)) end

        local function va(a, ...) return a, select('#', ...) end
        local a, n = va()
        assert(a == nil and n == 0)
        a, n = va(1, 2, 3)
        assert(a == 1 and n == 2)
        local results = {}
        for i = 1, 5 do results[#results + 1] = sum(i, i) end
        assert(table.concat(results, ",") == "2,4,6,8,10")
    "#);
}

#[test]
fn results_are_adjusted_to_the_requested_count() {
    run(r#"
        local function none() end
        local function one() return 7 end
        local function many() return 1, 2, 3 end
        local a = none(); assert(a == nil)
        local b, c = none(); assert(b == nil and c == nil)
        local d = one(); assert(d == 7)
        local e, g, h = one(); assert(e == 7 and g == nil and h == nil)
        local i = many(); assert(i == 1)
        local j, k, l, m = many(); assert(j == 1 and k == 2 and l == 3 and m == nil)
        assert(select('#', none()) == 0)
        assert(select('#', one()) == 1)
        assert(select('#', many()) == 3)
        assert((many()) == 1)
        assert(#{many(), many()} == 4)
        local function pass(...) return ... end
        assert(select('#', pass(many(), many())) == 4)

        -- Results land in the caller's registers around live values.
        local before, after = "b", "a"
        local r1, r2 = many()
        assert(before == "b" and after == "a" and r1 == 1 and r2 == 2)
        local t = {}
        t.v = one(); t[1], t[2] = many()
        assert(t.v == 7 and t[1] == 1 and t[2] == 2)
    "#);
}

#[test]
fn an_error_in_a_callee_keeps_its_traceback() {
    run(r#"
        local function inner(x)
            if x > 2 then error("boom " .. x) end
            return inner(x + 1) + 1
        end
        local function outer(x) local v = inner(x); return v end
        -- Lua 5.1's xpcall passes no arguments to the function.
        local ok, traceback = xpcall(function() return outer(0) end, debug.traceback)
        assert(not ok)
        local lines = {}
        for line in traceback:gmatch("[^\n]+") do lines[#lines + 1] = line end
        assert(lines[1]:find("boom 3", 1, true), lines[1])
        assert(lines[2] == "stack traceback:", lines[2])
        local names = {}
        for i = 3, #lines do names[#names + 1] = lines[i]:match("in (.*)$") end
        local function count(prefix)
            local n = 0
            for _, name in ipairs(names) do
                if name:find(prefix, 1, true) then n = n + 1 end
            end
            return n
        end
        -- Four activations of inner, then outer called by xpcall.
        assert(count("function 'inner'") == 4 or count("local 'inner'") == 4
            or count("upvalue 'inner'") == 4, traceback)

        local ok2, message = pcall(outer, 5)
        assert(not ok2 and message:find("boom 5", 1, true), message)

        -- Not a tail call: stock LuaJIT reuses the frame of a function that
        -- tail-calls a builtin, so `return debug.traceback(...)` omits it.
        local function level() local s = debug.traceback("here", 1); return s end
        local function caller() local s = level(); return s end
        local text = caller()
        assert(text:find("^here\nstack traceback:\n"), text)
        local first = text:match("\n\t([^\n]*)", 22)
        assert(first and first:find("level", 1, true), text)
    "#);
}

#[test]
fn lua51_generic_for_line_hooks_skip_the_body_when_iterator_is_exhausted() {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut lua = Lua::builder()
            .profile(CompatibilityProfile::Lua51)
            .execution_mode(mode)
            .build()
            .unwrap();
        lua.exec(r#"
            local function check(values, expected)
                local lines = {}
                local f = assert(loadstring("for i,v in pairs{" .. values .. "} do\n  a=i..v\nend\n"))
                debug.sethook(function(_, line) lines[#lines + 1] = line end, "l"); f(); debug.sethook()
                local actual = table.concat(lines, ",")
                assert(actual == expected, actual)
            end
            check("", "1,3")
            check("'a'", "1,2,1,3")
            check("'a','b'", "1,2,1,2,1,3")
        "#).unwrap_or_else(|error| panic!("{mode}: {error:?}"));
    }
}

#[test]
fn call_return_and_line_hooks_fire_in_order_when_set_mid_run() {
    run(r#"
        local events = {}
        local function hook(event, line)
            events[#events + 1] = event .. (line and (":" .. line) or "")
        end
        local function leaf(x) return x + 1 end
        local function mid(x) local y = leaf(x); return y * 2 end
        local total = 0
        for i = 1, 6 do
            if i == 3 then debug.sethook(hook, "crl") end
            total = total + mid(i)
            if i == 5 then debug.sethook() end
        end
        assert(total == 54)
        local joined = table.concat(events, " ")
        -- Nothing is reported before the hook is installed; after that every
        -- call is followed by its callee's lines and then by its return.
        assert(joined:find("^return"), joined)
        local calls, returns = 0, 0
        for _, event in ipairs(events) do
            if event == "call" then calls = calls + 1 end
            if event == "return" then returns = returns + 1 end
        end
        assert(calls == 7, joined)
        assert(returns == 7, joined)
        local expected = "call line:7 call line:6 return line:6 return line:7"
        local count, start = 0, 1
        while true do
            local from, to = joined:find(expected, start, true)
            if not from then break end
            count, start = count + 1, to + 1
        end
        assert(count == 3, joined)
    "#);
}

#[test]
fn open_result_calls_of_plain_and_variadic_callees() {
    run(r##"
        local unpack = table.unpack or unpack
        local load = loadstring or load
        local function two(a) return a, a + 1 end
        local function none() end
        local function many(n) local t = {} for i = 1, n do t[i] = i end return unpack(t) end
        local function variadic(...) return ... end
        local function count(...) return select("#", ...) end
        for i = 1, 300 do
            assert(count(two(i)) == 2)
            assert(count(none()) == 0)
            assert(count(variadic(i, i, i)) == 3)
            assert(count(variadic()) == 0)
            assert(count(many(i % 60)) == i % 60)
            local t = {two(i)}
            assert(#t == 2 and t[1] == i and t[2] == i + 1)
            local a, b, c = i, two(i)
            assert(a == i and b == i and c == i + 1)
            t = {i, variadic(two(i))}
            assert(#t == 3 and t[2] == i and t[3] == i + 1)
            assert(select(2, two(i)) == i + 1)
        end

        -- One call site whose callee changes between plain, variadic and
        -- native.
        local callees = {two, variadic, none, many, rawequal}
        local expected = {2, 2, 0, 3, 1}
        for i = 1, 500 do
            local which = i % 5 + 1
            local ok, n = pcall(function() return count(callees[which](3, 3)) end)
            assert(ok and n == expected[which], which)
        end

        -- A callee with more registers than the caller has left.
        local names = {}
        for i = 1, 100 do names[i] = "v" .. i end
        local wide = assert(load("local a = ... local " .. table.concat(names, ", ") .. " = a, a "
            .. "return v1, v2, v100, a"))
        for i = 1, 100 do
            local t = {count(wide(i)), wide(i)}
            assert(t[1] == 4 and t[2] == i and t[3] == i and t[4] == nil and t[5] == i)
        end

        -- The caller's line is visible from a callee entered this way.
        local function where() return debug.getinfo(2, "l").currentline, 1 end
        for i = 1, 100 do
            local t = {debug.getinfo(1, "l").currentline, where()}
            assert(t[1] == t[2] and t[3] == 1)
        end

        -- An error in the callee unwinds to the caller's handler.
        local function thrower(i) if i % 3 == 0 then error({i}) end return i, i end
        local caught = 0
        for i = 1, 300 do
            local ok, value = pcall(function() return count(thrower(i)) end)
            if ok then assert(value == 2) else caught = caught + 1; assert(value[1] == i) end
        end
        assert(caught == 100)

        -- A callee that yields resumes into the open-result call.
        local function yields() local a, b, c = coroutine.yield(1); return a, b, c end
        for i = 1, 50 do
            local co = coroutine.wrap(function() return count(yields()) end)
            assert(co() == 1)
            assert(co(i, i, i) == 3)
        end

        -- Recursion through an open-result call overflows like any other.
        local depth = 0
        local function open() depth = depth + 1; return count(open()) end
        local ok, message = pcall(open)
        assert(not ok and message:find("stack overflow", 1, true), message)
        assert(depth == EXPECTED_DEPTH, "depth " .. depth)
        local function down(n) if n == 0 then return 0 end return 1 + (select(2, nil, down(n - 1))) end
        assert(down(1000) == 1000)
    "##);
}

#[test]
fn call_metamethods_enter_their_handlers() {
    run(r##"
        local calls = 0
        local mt = {__call = function(self, a, b) calls = calls + 1; return a + (b or 0), self end}
        local obj = setmetatable({}, mt)
        for i = 1, 300 do
            local r, s = obj(i, 1)
            assert(r == i + 1 and s == obj)
            assert(obj(i) == i)
            assert(select("#", obj(i)) == 2)
        end
        assert(calls == 900)

        -- A variadic handler receives the object first.
        mt.__call = function(self, ...) assert(self == obj); return select("#", ...), ... end
        for i = 1, 300 do
            local n, a, b = obj(i, 2)
            assert(n == 2 and a == i and b == 2)
            assert(obj() == 0)
        end

        -- A native handler, then a Lua one installed while the loop runs.
        mt.__call = rawequal
        assert(obj(obj) == true and obj(1) == false)
        for i = 1, 300 do
            if i == 150 then mt.__call = function(self, a) return -a end end
            local r = obj(i, 0)
            if i >= 150 then assert(r == -i) else assert(r == false) end
        end

        -- More arguments than the handler has parameters, and fewer.
        mt.__call = function(self, a, b, c) return c == nil and (a or 0) + (b or 0) end
        for i = 1, 300 do
            assert(obj(i, 1, nil, 4, 5, 6, 7, 8) == i + 1)
            assert(obj() == 0)
        end

        -- The handler recurses through the object.
        mt.__call = function(self, n) if n == 0 then return 0 end return 1 + self(n - 1) end
        assert(obj(1000) == 1000)
        local ok, message = pcall(obj, 1e7)
        assert(not ok and tostring(message):find("stack overflow", 1, true), message)
        assert(obj(1000) == 1000)

        -- An error in the handler, and a handler that goes away.
        mt.__call = function(self, a) error("handler " .. a, 0) end
        for i = 1, 100 do
            local ok, message = pcall(function() return obj(i) end)
            assert(not ok and message == "handler " .. i, message)
        end
        mt.__call = nil
        local ok, message = pcall(function() return obj(1) end)
        assert(not ok and message:find("attempt to call", 1, true), message)
        setmetatable(obj, nil)
        ok, message = pcall(function() return obj(1) end)
        assert(not ok and message:find("attempt to call", 1, true), message)
    "##);
}

#[test]
fn one_call_site_and_depth_alternates_between_callees_of_every_shape() {
    run(r##"
        local up = 10
        local function plain(a) return a + 1 end
        local function captures(a) return a + up end
        local function wide(a, b, c, d, e, f) local x, y, z = a, a, a return x + y + z end
        local function variadic(...) return (...) + select("#", ...) end
        local function none() end
        local function tailing(a) return plain(a) end
        local function make(k) return function(a) return a + k end end
        local family = {make(100), make(200), make(300)}
        local callees = {plain, captures, wide, variadic, none, tailing, family[1], family[2], family[3]}
        local expected = {
            function(i) return i + 1 end,
            function(i) return i + 10 end,
            function(i) return 3 * i end,
            function(i) return i + 1 end,
            function(i) return nil end,
            function(i) return i + 1 end,
            function(i) return i + 100 end,
            function(i) return i + 200 end,
            function(i) return i + 300 end,
        }
        for i = 1, 2000 do
            local which = i % #callees + 1
            local result = callees[which](i)
            assert(result == expected[which](i), which)
        end

        -- The record a callee leaves is reused by a callee that raises, and
        -- the error names the right line.
        local function raises(a) error("raised " .. a) end
        local line = debug.getinfo(raises, "S").linedefined
        for i = 1, 300 do
            assert(plain(i) == i + 1)
            local ok, message = pcall(raises, i)
            assert(not ok and message:find(":" .. line .. ": raised " .. i, 1, true), message)
            assert(captures(i) == i + 10)
        end

        -- Closures of one prototype with their own upvalue cells.
        local counters = {}
        for n = 1, 3 do
            local count = 0
            counters[n] = function() count = count + n return count end
        end
        for i = 1, 300 do
            for n = 1, 3 do assert(counters[n]() == i * n) end
        end

        -- `debug.getinfo` in a callee that reuses another callee's record.
        local function where() return debug.getinfo(1, "S").linedefined, debug.getinfo(2, "l").currentline end
        local function other() return 0 end
        for i = 1, 300 do
            other()
            local defined, current = where()
            assert(defined == debug.getinfo(where, "S").linedefined)
            assert(current == debug.getinfo(1, "l").currentline - 2)
        end
    "##);
}

/// The same callee called again at one depth is the call the fast loop
/// takes on the record's plain mark alone. Each loop below runs it often
/// enough to be marked, then does something to the running activation or to
/// its closure that the mark does not cover, and calls again.
#[test]
fn a_repeated_callee_whose_activation_changes_under_it() {
    run(r##"
        local unpack = table.unpack or unpack

        -- Interior state made while the callee runs: a hook set inside it,
        -- which then sees its return and the next call.
        local events = {}
        local function hooked(i)
            if i == 50 then
                debug.sethook(function(event) events[#events + 1] = event end, "cr")
            end
            return i * 2
        end
        local total = 0
        for i = 1, 60 do total = total + hooked(i) end
        debug.sethook()
        assert(total == 60 * 61, total)
        assert(events[1] == "return", events[1])
        local calls = 0
        for _, event in ipairs(events) do
            if event == "call" then calls = calls + 1 end
        end
        assert(calls >= 10, calls)

        -- An error raised in the marked activation, caught, and the callee
        -- called again at the same depth.
        local function fragile(i)
            if i % 7 == 0 then error("seven " .. i, 0) end
            return i + 1
        end
        for i = 1, 100 do
            local ok, value = pcall(fragile, i)
            if i % 7 == 0 then
                assert(not ok and value == "seven " .. i, value)
            else
                assert(ok and value == i + 1)
            end
            local direct = (i % 7 == 0) and i + 1 or fragile(i)
            assert(direct == i + 1)
        end

        -- A local of the callee captured by a closure it returns: the plain
        -- return leaves it to the general one, which closes it.
        local function capturing(i)
            local held = i * 3
            if i % 5 == 0 then return function() return held end end
            return held
        end
        for i = 1, 100 do
            local result = capturing(i)
            if i % 5 == 0 then result = result() end
            assert(result == i * 3)
        end

        -- The callee yields from a marked activation and is resumed.
        local function pausing(i)
            if i % 3 == 0 then coroutine.yield(i) end
            return i
        end
        local thread = coroutine.wrap(function()
            local sum = 0
            for i = 1, 90 do sum = sum + pausing(i) end
            return "done", sum
        end)
        local yields = 0
        while true do
            local a, b = thread()
            if a == "done" then
                assert(b == 90 * 91 / 2)
                break
            end
            yields = yields + 1
            assert(a == yields * 3)
        end
        assert(yields == 30)

        -- A tail call out of the marked activation, then the callee again.
        local function landing(i) return i - 1 end
        local function hopping(i)
            if i % 4 == 0 then return landing(i) end
            return i
        end
        for i = 1, 100 do
            local expected = (i % 4 == 0) and i - 1 or i
            assert(hopping(i) == expected)
        end

        -- A vararg callee and a plain one take turns under a marked caller.
        local function variadic(...) return select('#', ...) end
        local function fixed(a, b) return (a or 0) + (b or 0) end
        local function driver(i)
            return variadic(i, i, i) + fixed(i, 1) + variadic()
        end
        for i = 1, 100 do assert(driver(i) == 3 + i + 1) end

        -- The callee's upvalue is rejoined between calls.
        if debug.upvaluejoin then
            local a, b = 1, 100
            local function reads_a() return a end
            local function reads_b() return b end
            for i = 1, 40 do assert(reads_a() == 1) end
            debug.upvaluejoin(reads_a, 1, reads_b, 1)
            for i = 1, 40 do assert(reads_a() == 100) end
            b = 7
            assert(reads_a() == 7)
        end

        -- And its upvalue is set through the debug library.
        do
            local seen = 5
            local function reads() return seen end
            for i = 1, 40 do assert(reads() == 5) end
            debug.setupvalue(reads, 1, 9)
            for i = 1, 40 do assert(reads() == 9) end
        end

        -- The callee's environment changes between calls.
        if setfenv then
            local function global_reader() return marker end
            marker = "outer"
            for i = 1, 40 do assert(global_reader() == "outer") end
            setfenv(global_reader, { marker = "inner" })
            for i = 1, 40 do assert(global_reader() == "inner") end
            marker = nil
        end

        -- Levels and lines as the debug library reports them from inside a
        -- callee that runs on a marked record.
        local function inspect()
            local info = debug.getinfo(2, "l")
            local name = debug.getlocal(2, 1)
            return info.currentline, name
        end
        local function inspected(first)
            local line, name = inspect()
            return line, name, first
        end
        for i = 1, 60 do
            local line, name, first = inspected(i)
            assert(line == debug.getinfo(inspected, "S").linedefined + 1, line)
            assert(name == "first" and first == i)
        end

        -- A traceback through three marked activations.
        local function leaf() return debug.traceback("here", 1) end
        local function middle() local text = leaf() return text end
        local function root() local text = middle() return text end
        local text
        for i = 1, 30 do text = root() end
        local _, count = text:gsub("in ", "")
        assert(count >= 3, text)

        -- Strings as arguments and results, and strings left in the
        -- registers the next call's arguments go to.
        local function joins(a, b) local both = a .. b return both end
        for i = 1, 100 do
            assert(joins("k", i) == "k" .. i)
            assert(joins(i, "v") == i .. "v")
        end

        -- Calls whose results are dropped, and calls with several.
        local dropped = 0
        local function bump(i) dropped = dropped + i return i end
        local function pair(i) return i, -i end
        for i = 1, 100 do
            bump(i)
            local x, y, z = pair(i)
            assert(x == i and y == -i and z == nil)
            local packed = { unpack({ pair(i) }) }
            assert(#packed == 2)
        end
        assert(dropped == 5050)
    "##);
}

/// A to-be-closed local in a callee that is called again and again: its
/// return is the general one each time, and the handler runs once per call.
#[test]
fn a_repeated_callee_with_a_local_to_close() {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        for profile in [CompatibilityProfile::Lua54, CompatibilityProfile::Lua55] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            lua.vm_mut().use_synchronous_jit_compilation();
            lua.exec(
                r#"
                local closed = 0
                local handle = setmetatable({}, { __close = function() closed = closed + 1 end })
                local function scoped(i)
                    if i % 2 == 0 then
                        local guard <close> = handle
                        return i
                    end
                    return -i
                end
                local sum = 0
                for i = 1, 200 do sum = sum + scoped(i) end
                assert(closed == 100, closed)
                assert(sum == 100, sum)
                "#,
            )
            .unwrap_or_else(|error| panic!("{profile} {mode}: {error:?}"));
        }
    }
}

/// Callees that take turns at one depth with every argument count from none
/// to nine, each callee taking fewer, as many and more than it declares, from
/// callers whose windows differ in size, and with strings, tables and nil
/// among the arguments and among what the last callee left in its registers.
#[test]
fn callees_take_turns_at_one_depth_with_every_argument_count() {
    run(r#"
        local function f0() return 1 end
        local function f1(a) return a end
        local function f2(a, b) local t = b; return a, t end
        local function f3(a, b, c) local x, y, z; return a, b, c, x, y, z end
        local function f5(a, b, c, d, e) return e, d, c, b, a end
        local function f9(a, b, c, d, e, f, g, h, i)
            local last = i
            return a, b, c, d, e, f, g, h, last
        end
        local function leaves(a, b, c)
            -- Strings and a table in the registers the next callee takes.
            local s, t, u = "left" .. tostring(a), { b }, c
            return s, t, u
        end
        local callees = { f0, f1, f2, f3, f5, f9 }
        local function narrow(f, n, i)
            if n == 0 then return f() end
            if n == 1 then return f(i) end
            if n == 2 then return f(i, "s" .. i) end
            if n == 3 then return f(i, nil, i + 2) end
            if n == 4 then return f(i, i + 1, i + 2, i + 3) end
            if n == 5 then return f(i, i + 1, "five", i + 3, i + 4) end
            if n == 6 then return f(i, i + 1, i + 2, i + 3, i + 4, i + 5) end
            if n == 7 then return f(i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6) end
            if n == 8 then return f(i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7) end
            return f(i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7, i + 8)
        end
        -- The same calls without a tail call, so that the callee's record
        -- sits under this caller's window and returns into it.
        local function plain(f, n, i)
            local a, b, c, d, e, g, h, j, k
            if n == 0 then a, b, c, d, e, g, h, j, k = f()
            elseif n == 1 then a, b, c, d, e, g, h, j, k = f(i)
            elseif n == 2 then a, b, c, d, e, g, h, j, k = f(i, "s" .. i)
            elseif n == 3 then a, b, c, d, e, g, h, j, k = f(i, nil, i + 2)
            elseif n == 4 then a, b, c, d, e, g, h, j, k = f(i, i + 1, i + 2, i + 3)
            elseif n == 5 then a, b, c, d, e, g, h, j, k = f(i, i + 1, "five", i + 3, i + 4)
            elseif n == 6 then a, b, c, d, e, g, h, j, k = f(i, i + 1, i + 2, i + 3, i + 4, i + 5)
            elseif n == 7 then
                a, b, c, d, e, g, h, j, k = f(i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6)
            elseif n == 8 then
                a, b, c, d, e, g, h, j, k = f(i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7)
            else
                a, b, c, d, e, g, h, j, k =
                    f(i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7, i + 8)
            end
            return a, b, c, d, e, g, h, j, k
        end
        local function wide(f, n, i)
            local p, q, r, s, t, u, v, w = 1, 2, 3, 4, 5, 6, 7, 8
            local a, b = plain(f, n, i)
            return a, b, p + q + r + s + t + u + v + w
        end
        local function verify(got, count, which, n, i, where)
            if which == 1 then
                assert(count == 1 and got[1] == 1, where)
            elseif which == 2 then
                assert(count == 1 and got[1] == (n >= 1 and i or nil), where)
            elseif which == 3 then
                local second = nil
                if n == 2 then second = "s" .. i elseif n >= 4 then second = i + 1 end
                assert(count == 2 and got[1] == (n >= 1 and i or nil), where)
                assert(got[2] == second, where)
            elseif which == 4 then
                assert(count == 6, where)
                assert(got[4] == nil and got[5] == nil and got[6] == nil, where)
                local third = nil
                if n == 5 then third = "five" elseif n >= 3 then third = i + 2 end
                assert(got[3] == third, where)
            elseif which == 5 then
                assert(count == 5 and got[5] == (n >= 1 and i or nil), where)
                assert(got[1] == (n >= 5 and i + 4 or nil), where)
                if n == 5 then assert(got[3] == "five", where) end
            else
                assert(count == 9 and got[9] == (n >= 9 and i + 8 or nil), where)
                assert(got[1] == (n >= 1 and i or nil), where)
            end
        end
        local counts = { 1, 1, 2, 6, 5, 9 }
        for round = 1, 40 do
            for n = 0, 9 do
                for which = 1, #callees do
                    local f = callees[which]
                    local i = round * 100 + n
                    local where = "round " .. round .. " callee " .. which .. " with " .. n
                    local got = { narrow(f, n, i) }
                    verify(got, select('#', narrow(f, n, i)), which, n, i, where .. " tail")
                    local direct = { plain(f, n, i) }
                    verify(direct, counts[which], which, n, i, where .. " plain")
                    local a, b, c = wide(f, n, i)
                    assert(c == 36 and a == got[1] and b == got[2], where .. " wide")
                    if round % 7 == 0 then
                        local s, t, u = leaves(i, n, which)
                        assert(s == "left" .. i and t[1] == n and u == which, where)
                    end
                end
            end
        end
    "#);
}

#[test]
fn a_callee_warm_at_its_depth_leaves_the_short_call_and_return_when_it_must() {
    run(r#"
        local function id(x) return x end
        local function add(a, b) return a + b end
        -- Warm the record at this depth: the calls after the second take
        -- the short push and the short return.
        for i = 1, 8 do assert(id(i) == i and add(i, 1) == i + 1) end

        -- Other result counts from the same callee and depth.
        local a, b = id(1)
        assert(a == 1 and b == nil)
        id(2)
        local packed = { id(3) }
        assert(#packed == 1 and packed[1] == 3)
        local function two() return 1, 2 end
        for _ = 1, 4 do
            local x = two()
            local y, z = two()
            assert(x == 1 and y == 1 and z == 2)
        end

        -- Strings as arguments, results, and stale register contents.
        for i = 1, 4 do
            assert(id("s" .. i) == "s" .. i)
            assert(id(i) == i)
            assert(add(i, 2) == i + 2)
        end

        -- The caller's window moves: the same callee under callers with
        -- other frame sizes, at the same depth.
        local function wide(n)
            local p, q, r, s, t, u = 1, 2, 3, 4, 5, 6
            return id(n) + p + q + r + s + t + u
        end
        local function narrow(n) return id(n) end
        for i = 1, 6 do
            assert(wide(i) == i + 21)
            assert(narrow(i) == i)
        end

        -- A callee whose return has a captured local to close.
        local kept
        local function capture(x)
            local cell = x
            kept = function() cell = cell + 1; return cell end
            return cell
        end
        for i = 1, 4 do
            assert(capture(i) == i)
            assert(kept() == i + 1 and kept() == i + 2)
            assert(id(i) == i)
        end

        -- A hook set inside a warm callee sees that callee return, and
        -- every call and return after it; taken off, the calls go on.
        local events = {}
        local function hooked(x)
            if x == 3 then
                debug.sethook(function(event) events[#events + 1] = event end, "cr")
            end
            return x
        end
        local total = 0
        for i = 1, 5 do total = total + hooked(i) end
        debug.sethook()
        assert(total == 15)
        local calls, returns = 0, 0
        for _, event in ipairs(events) do
            if event == "call" then calls = calls + 1 end
            if event == "return" or event == "tail return" then returns = returns + 1 end
        end
        -- The return of `hooked(3)`, two more calls with their returns, and
        -- the call of `debug.sethook` that takes the hook off.
        assert(returns >= 3 and calls >= 3, calls .. " " .. returns)
        for i = 1, 4 do assert(hooked(i + 5) == i + 5) end

        -- A count hook runs between warm calls.
        local ticks = 0
        debug.sethook(function() ticks = ticks + 1 end, "", 1)
        for i = 1, 20 do assert(add(i, i) == 2 * i) end
        debug.sethook()
        assert(ticks > 20, ticks)

        -- A collection asked for inside a warm callee.
        local function churn(n)
            local t = { n, {}, tostring(n) }
            if n % 3 == 0 then collectgarbage() end
            if n % 5 == 0 then collectgarbage("step") end
            return #t
        end
        for i = 1, 30 do assert(churn(i) == 3) end

        -- An error out of a warm callee, and the depth is usable after it.
        local function fails(x) if x == 4 then error("boom") end return x end
        for i = 1, 6 do
            local ok, value = pcall(fails, i)
            assert(ok == (i ~= 4))
            assert(i == 4 or value == i)
            assert(i ~= 4 or tostring(value):find("boom", 1, true))
            assert(id(i) == i)
        end
        local trace
        local function deep(x) if x == 3 then trace = debug.traceback("here") end return x end
        for i = 1, 4 do deep(i) end
        assert(trace:find("here", 1, true) and trace:find("deep", 1, true), trace)

        -- A coroutine runs warm calls of its own and yields between them.
        local co = coroutine.wrap(function()
            for i = 1, 6 do coroutine.yield(id(i) + add(i, i)) end
        end)
        for i = 1, 6 do assert(co() == 3 * i and id(i) == i) end

        -- The closure's environment changes under a warm record.
        if setfenv then
            local function reads() return marker end
            marker = 1
            for _ = 1, 4 do assert(reads() == 1) end
            setfenv(reads, { marker = 2 })
            for _ = 1, 4 do assert(reads() == 2) end
            marker = nil
        end

        -- Variadic and tail-calling callees take turns with a plain one.
        local function variadic(...) return select('#', ...) end
        local function tail(x) return id(x) end
        for i = 1, 6 do
            assert(id(i) == i)
            assert(variadic(i, i) == 2)
            assert(tail(i) == i)
            assert(add(i, i) == 2 * i)
        end
        local info
        local function inspect(x) info = debug.getinfo(1, "l"); return x end
        for i = 1, 3 do assert(inspect(i) == i) end
        assert(info.currentline > 0)
    "#);
}
