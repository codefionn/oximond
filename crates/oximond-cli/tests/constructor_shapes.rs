//! Tables built by `{a = x, b = y}` constructors share one key list per site
//! instead of owning their keys. None of that may be visible: these programs
//! pin iteration order, nil handling, deletion during traversal, metatables,
//! the collector and the debug library on such tables, with the constructors
//! run often enough to go through the fast loop and the tracing compiler.

use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const ALL_PROFILES: [CompatibilityProfile; 4] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn run(source: &str) {
    for profile in ALL_PROFILES {
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
fn iteration_order_matches_a_dynamically_built_table() {
    run(r#"
        local function keys(t)
            local out = {}
            for k in pairs(t) do out[#out + 1] = k end
            return table.concat(out, ",")
        end
        local function build(i)
            return {zed = i, alpha = i, mid = i, beta = i, omega = i}
        end
        local dynamic = {}
        for _, k in ipairs({"omega", "beta", "mid", "alpha", "zed"}) do dynamic[k] = 1 end
        local want = keys(dynamic)
        for i = 1, 400 do
            local t = build(i)
            assert(keys(t) == want, keys(t))
            assert(next(t) == next(dynamic))
        end
    "#);
}

#[test]
fn keys_added_and_removed_after_construction() {
    run(r#"
        for i = 1, 400 do
            local t = {a = i, b = i + 1, c = i + 2}
            t.d = 4
            t[1] = "one"
            t.b = nil
            t.e = nil
            local n = 0
            for k, v in pairs(t) do n = n + 1 end
            assert(n == 4, n)
            assert(t.a == i and t.b == nil and t.c == i + 2 and t.d == 4 and t[1] == "one")
            t.b = 9
            assert(t.b == 9)
            local s = {x = 1, y = 2}
            s.x = nil
            s.y = nil
            assert(next(s) == nil)
            s.x = 3
            assert(next(s) == "x")
        end
    "#);
}

#[test]
fn clearing_fields_during_traversal_keeps_next_working() {
    run(r"
        for i = 1, 300 do
            local t = {a = 1, b = 2, c = 3, d = 4}
            local seen = 0
            for k, v in pairs(t) do
                seen = seen + 1
                t[k] = nil
            end
            assert(seen == 4 and next(t) == nil)
            local u = {a = 1, b = 2, c = 3}
            local k = next(u)
            u[k] = nil
            local k2 = next(u, k)
            assert(k2 ~= nil and u[k2] ~= nil)
            u[k2] = u[k2] + 1
        end
    ");
}

#[test]
fn nil_values_in_a_constructor_leave_the_key_absent() {
    run(r#"
        local function nothing() return nil end
        for i = 1, 300 do
            local t = {a = nil, b = 1}
            assert(t.a == nil and t.b == 1)
            local n = 0
            for _ in pairs(t) do n = n + 1 end
            assert(n == 1 and next(t) == "b")
            assert(rawget(t, "a") == nil)
            local u = {a = nothing(), b = nothing(), c = i}
            n = 0
            for _ in pairs(u) do n = n + 1 end
            assert(n == 1 and u.c == i)
            u.a = 1
            n = 0
            for _ in pairs(u) do n = n + 1 end
            assert(n == 2)
            local v = {p = nil, q = nil}
            assert(next(v) == nil)
        end
    "#);
}

#[test]
fn length_with_mixed_array_and_hash_constructors() {
    run(r#"
        local rawlen = rawlen or function(t) return #t end
        for i = 1, 300 do
            local t = {10, 20, 30, name = "x", other = i}
            assert(#t == 3 and rawlen(t) == 3, #t)
            t[4] = 40
            assert(#t == 4)
            t.name = nil
            assert(#t == 4 and rawlen(t) == 4)
            local h = {a = 1, b = 2}
            assert(#h == 0 and rawlen(h) == 0)
        end
    "#);
}

#[test]
fn weak_tables_built_by_constructors() {
    run(r#"
        local function weak(mode)
            return setmetatable({a = {}, b = {}, c = 1}, {__mode = mode})
        end
        for _, mode in ipairs({"v", "k", "kv"}) do
            local t = weak(mode)
            collectgarbage()
            collectgarbage()
            if mode ~= "k" then
                assert(t.a == nil and t.b == nil, mode)
            end
            assert(t.c == 1)
        end
        local keyed = setmetatable({}, {__mode = "k"})
        local holder = {x = {}, y = {}}
        keyed[holder.x] = 1
        keyed[holder.y] = 2
        holder.x = nil
        collectgarbage()
        collectgarbage()
        local n = 0
        for _ in pairs(keyed) do n = n + 1 end
        assert(n == 1, n)
    "#);
}

#[test]
fn metatables_built_by_constructors_keep_their_hooks() {
    run(r#"
        local order = {}
        local function make(i)
            return setmetatable({}, {
                __index = function(t, k) return k .. i end,
                __newindex = function(t, k, v) order[#order + 1] = k; rawset(t, k, v) end,
                __tostring = function() return "obj" .. i end,
            })
        end
        for i = 1, 200 do
            local o = make(i)
            assert(o.foo == "foo" .. i)
            o.bar = 1
            o.bar = 2
            assert(tostring(o) == "obj" .. i)
        end
        assert(#order == 200)

        local closed = 0
        local collected = 0
        for i = 1, 50 do
            setmetatable({}, {__gc = function() collected = collected + 1 end})
            local meta = {__gc = function() collected = collected + 1 end, __index = {}}
            setmetatable({payload = i}, meta)
        end
        collectgarbage()
        collectgarbage()
        collectgarbage()
        if _VERSION ~= "Lua 5.1" then
            assert(collected == 100, collected)
        end
    "#);
}

#[test]
fn sort_and_concat_on_records() {
    run(r#"
        local function record(i)
            return {id = i, name = "n" .. i, score = (i * 7) % 11}
        end
        local list = {}
        for i = 1, 200 do list[i] = record(i) end
        table.sort(list, function(x, y)
            if x.score ~= y.score then return x.score < y.score end
            return x.id < y.id
        end)
        for i = 2, #list do
            local a, b = list[i - 1], list[i]
            assert(a.score < b.score or (a.score == b.score and a.id < b.id))
        end
        local parts = {}
        for i = 1, 5 do parts[i] = list[i].name end
        assert(#table.concat(parts, ",") > 0)
        local rec = record(3)
        rec.id, rec.name, rec.score = nil, nil, nil
        assert(next(rec) == nil)
    "#);
}

#[test]
fn memory_accounting_matches_a_table_built_dynamically() {
    run(r#"
        local function measure(build)
            collectgarbage()
            collectgarbage("stop")
            local before = collectgarbage("count")
            local keep = {}
            for i = 1, 500 do keep[i] = build(i) end
            local after = collectgarbage("count")
            collectgarbage("restart")
            return after - before
        end
        -- rawset calls are not SetField, so the shape analysis leaves these unshaped.
        local function same(label, shaped, plain)
            local x, y = measure(shaped), measure(plain)
            assert(x == y, label .. ": " .. x .. " vs " .. y)
        end
        same("three keys",
            function(i) return {a = i, b = i, c = i} end,
            function(i)
                local t = {}
                rawset(t, "a", i) rawset(t, "b", i) rawset(t, "c", i)
                return t
            end)
        local KEYS = {"a", "b", "c", "d", "e", "f"}
        same("six keys",
            function(i) return {a = i, b = i, c = i, d = i, e = i, f = i} end,
            function(i)
                local t = {}
                for _, k in ipairs(KEYS) do t[k] = i end
                return t
            end)
        same("partly filled",
            function(i) return {a = i, b = nil, c = nil, d = nil, e = nil} end,
            function(i)
                local t = {}
                rawset(t, "a", i)
                return t
            end)
    "#);
}

#[test]
fn debug_library_sees_a_half_built_constructor() {
    run(r#"
        local function build(i)
            local t = {a = i, b = i + 1, c = (function()
                local name, value = debug.getlocal(2, 1)
                assert(name == "i" and value == i)
                return 3
            end)()}
            return t
        end
        for i = 1, 100 do
            local t = build(i)
            assert(t.a == i and t.b == i + 1 and t.c == 3)
        end
        -- The table under construction sits in a temporary of the caller. A
        -- profile that exposes it through getlocal must show only the keys
        -- stored so far, and writes made through it must survive the
        -- constructor's remaining stores.
        local exposed = 0
        local function half_built(i)
            local half
            local t = {a = i, b = (function()
                for k = 1, 64 do
                    local name, v = debug.getlocal(2, k)
                    if name == nil then break end
                    if type(v) == "table" then half = v break end
                end
                if half then
                    local keys = {}
                    for key in pairs(half) do keys[#keys + 1] = key end
                    assert(#keys == 1 and keys[1] == "a", #keys)
                    assert(next(half, "a") == nil)
                    assert(rawget(half, "c") == nil)
                    half.zz = true
                    half.a = nil
                end
                return i + 1
            end)(), c = i + 2}
            return t, half
        end
        for i = 1, 100 do
            local t, half = half_built(i)
            assert(t.b == i + 1 and t.c == i + 2)
            if half then
                exposed = exposed + 1
                assert(half == t)
                local keys = {}
                for key in pairs(t) do keys[#keys + 1] = key end
                table.sort(keys)
                assert(table.concat(keys, ",") == "b,c,zz", table.concat(keys, ","))
                assert(t.a == nil and t.zz == true)
            else
                assert(t.a == i)
            end
        end
        assert(exposed == 0 or exposed == 100, exposed)
        local function probe()
            local seen
            local t = {x = 1, y = (function()
                seen = debug.getinfo(1, "l").currentline
                return 2
            end)(), z = 3}
            return t, seen
        end
        local t, line = probe()
        assert(t.x == 1 and t.y == 2 and t.z == 3 and line ~= nil)
    "#);
}

#[test]
fn coroutines_can_yield_inside_constructor_values() {
    run(r#"
        local function gen()
            for i = 1, 5 do
                local t = {first = i, second = coroutine.yield(i), third = i * 3}
                coroutine.yield(t)
            end
        end
        for round = 1, 60 do
            local co = coroutine.wrap(gen)
            for i = 1, 5 do
                assert(co() == i)
                local t = co("v" .. i)
                assert(t.first == i and t.second == "v" .. i and t.third == i * 3)
            end
        end
    "#);
}
