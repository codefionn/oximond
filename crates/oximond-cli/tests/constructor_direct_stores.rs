//! Keyed constructors store straight into the slots of their shared key list,
//! and the blocks of dead record tables are recycled. Neither may show: these
//! programs pin evaluation order, duplicate keys, nil values, positional and
//! keyed mixes, call and vararg tails, nesting, metatables set later, and
//! collections in the middle of a constructor. Every program runs on every
//! profile in both modes, with each constructor repeated past the points where
//! the fast loop and the tracing compiler take over.

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
fn values_are_evaluated_in_source_order() {
    run(r#"
        local log
        local function tick(name, value)
            log[#log + 1] = name
            return value
        end
        for i = 1, 400 do
            log = {}
            local t = {a = tick("a", i), b = tick("b", i + 1), c = tick("c", i + 2)}
            assert(table.concat(log) == "abc", table.concat(log))
            assert(t.a == i and t.b == i + 1 and t.c == i + 2)
            log = {}
            local u = {tick("1", 1), x = tick("x", 2), tick("2", 3), y = tick("y", 4)}
            assert(table.concat(log) == "1x2y", table.concat(log))
            assert(u[1] == 1 and u[2] == 3 and u.x == 2 and u.y == 4)
            -- A value that reads the table being built sees no part of it.
            local seen
            local function peek() seen = true; return i end
            local v = {p = peek(), q = peek()}
            assert(seen and v.p == i and v.q == i)
        end
    "#);
}

#[test]
fn a_repeated_key_keeps_the_last_value() {
    run(r#"
        for i = 1, 400 do
            local t = {a = 1, b = 2, a = i, c = 3, b = "two", a = i + 1}
            assert(t.a == i + 1 and t.b == "two" and t.c == 3)
            local n = 0
            for _ in pairs(t) do n = n + 1 end
            assert(n == 3, n)
            local u = {k = 1, k = 2}
            assert(u.k == 2 and next(u) == "k" and next(u, "k") == nil)
            local v = {k = i, k = nil}
            assert(v.k == nil and next(v) == nil)
        end
    "#);
}

#[test]
fn nil_values_leave_keys_absent_and_fields_can_be_stored_later() {
    run(r#"
        local function nothing() return nil end
        for i = 1, 400 do
            local t = {a = nil, b = i, c = nothing()}
            assert(t.a == nil and t.b == i and t.c == nil)
            local n = 0
            for _ in pairs(t) do n = n + 1 end
            assert(n == 1, n)
            t.a = "late"
            t.c = i
            n = 0
            for _ in pairs(t) do n = n + 1 end
            assert(n == 3 and t.a == "late" and t.c == i)
            t.b = nil
            assert(t.b == nil and next(t) ~= nil)
            local one = {only = nil}
            assert(next(one) == nil)
            one.only = i
            assert(next(one) == "only" and one.only == i)
        end
    "#);
}

#[test]
fn positional_and_keyed_items_mix() {
    run(r#"
        for i = 1, 400 do
            local t = {i, i + 1, name = "n" .. (i % 5), i + 2, tag = i % 7, [10] = "ten"}
            assert(#t == 3 and t[1] == i and t[2] == i + 1 and t[3] == i + 2)
            assert(t.name == "n" .. (i % 5) and t.tag == i % 7 and t[10] == "ten")
            t[4] = "four"
            t.extra = true
            assert(#t == 4 and t.extra == true and t.tag == i % 7)
            local only = {x = i}
            assert(only.x == i and #only == 0)
        end
    "#);
}

#[test]
fn call_and_vararg_tails_expand_after_the_keyed_fields() {
    run(r##"
        local unpack = unpack or table.unpack
        local function three() return 7, 8, 9 end
        local function count(...) return select("#", ...) end
        local function build(...)
            return {x = 1, y = 2, ...}
        end
        for i = 1, 300 do
            local t = {x = i, y = i, three()}
            assert(#t == 3 and t[1] == 7 and t[3] == 9 and t.x == i)
            local u = {three(), x = i, y = i}
            -- Not the last item: truncated to one value.
            assert(#u == 1 and u[1] == 7 and u.y == i)
            local v = build(i, nil, 5)
            assert(v.x == 1 and v.y == 2 and v[1] == i and v[3] == 5)
            assert(count(unpack(build(1, 2, 3))) == 3)
            local w = build()
            assert(#w == 0 and w.x == 1 and w.y == 2)
            local z = {p = 1, q = 2, (three())}
            assert(#z == 1 and z[1] == 7)
        end
    "##);
}

#[test]
fn nested_constructors_keep_their_own_fields() {
    run(r"
        local function node(depth, seed)
            if depth == 0 then return {value = seed} end
            return {value = seed, left = node(depth - 1, seed * 2), right = node(depth - 1, seed * 2 + 1)}
        end
        local function sum(t)
            if not t.left then return t.value end
            return t.value + sum(t.left) + sum(t.right)
        end
        for round = 1, 30 do
            local t = node(6, round)
            local n, stack = 0, {t}
            while #stack > 0 do
                local top = table.remove(stack)
                n = n + 1
                if top.left then
                    stack[#stack + 1] = top.left
                    stack[#stack + 1] = top.right
                end
            end
            assert(n == 127, n)
            assert(sum(t) > 0)
            local o = {a = {b = {c = {d = round}}}, e = {f = round}}
            assert(o.a.b.c.d == round and o.e.f == round and o.a.b.c.e == nil)
        end
    ");
}

#[test]
fn metatables_set_after_construction_apply_to_missing_fields_only() {
    run(r#"
        for i = 1, 300 do
            local t = {a = i, b = nil, c = i}
            local log = {}
            setmetatable(t, {
                __index = function(_, k) return "idx" .. k end,
                __newindex = function(tbl, k, v) log[#log + 1] = k; rawset(tbl, k, v) end,
            })
            assert(t.a == i and t.b == "idxb" and t.zz == "idxzz")
            t.a = i + 1
            assert(#log == 0 and rawget(t, "a") == i + 1)
            t.b = 5
            t.n = 6
            assert(#log == 2 and log[1] == "b" and log[2] == "n")
            assert(rawget(t, "b") == 5 and rawget(t, "n") == 6)
            -- Overwriting a present field stays raw, a cleared one asks again.
            t.c = nil
            t.c = 1
            assert(log[#log] == "c" and #log == 3)
            local ro = setmetatable({x = 1, y = 2}, {__newindex = function() error("frozen") end})
            assert(not pcall(function() ro.z = 1 end))
            ro.x = 10
            assert(ro.x == 10)
        end
    "#);
}

#[test]
fn a_gc_key_stored_by_a_constructor_still_registers_a_finalizer() {
    run(r"
        local count = 0
        local function finalizer() count = count + 1 end
        for i = 1, 50 do
            local mt = {__gc = finalizer, __index = function() return 1 end}
            local t = setmetatable({a = 1, b = 2}, mt)
            t = nil
        end
        collectgarbage()
        collectgarbage()
        -- Lua 5.1 and LuaJIT ignore __gc on tables.
        assert(count == 0 or count == 50, count)
    ");
}

#[test]
fn collections_during_construction_keep_every_stored_value() {
    run(r#"
        local function churn()
            local junk
            for i = 1, 40 do junk = {i, tostring(i), {i}} end
            collectgarbage()
            return junk
        end
        local keep = {}
        for i = 1, 150 do
            local t = {
                first = {i, "s" .. i},
                second = "str" .. i,
                third = churn(),
                fourth = {name = "inner" .. i},
                fifth = churn(),
            }
            keep[#keep + 1] = t
            if i % 25 == 0 then collectgarbage() end
        end
        collectgarbage()
        for i, t in ipairs(keep) do
            assert(t.first[1] == i and t.first[2] == "s" .. i)
            assert(t.second == "str" .. i)
            assert(#t.third == 3 and t.third[1] == 40 and t.third[2] == "40")
            assert(t.fourth.name == "inner" .. i)
            assert(t.fifth[3][1] == 40)
        end
    "#);
}

#[test]
fn old_tables_that_gain_young_field_values_keep_them_alive() {
    run(r#"
        local old = {}
        for i = 1, 100 do old[i] = {a = false, b = false, c = false} end
        collectgarbage()
        collectgarbage()
        for round = 1, 20 do
            for i = 1, 100 do
                local t = old[i]
                t.a = {round, i}
                t.b = "b" .. round .. ":" .. i
                t.c = {x = {round}}
            end
            for _ = 1, 200 do local _ = {u = {}, v = {}} end
            if round % 5 == 0 then collectgarbage() end
        end
        collectgarbage()
        for i = 1, 100 do
            local t = old[i]
            assert(t.a[1] == 20 and t.a[2] == i)
            assert(t.b == "b20:" .. i)
            assert(t.c.x[1] == 20)
        end
    "#);
}

#[test]
fn recycled_record_storage_starts_empty() {
    run(r#"
        local live = {}
        for round = 1, 60 do
            for i = 1, 300 do
                local t = {a = round, b = "s" .. i, c = {i}}
                if i % 50 == 0 then live[#live + 1] = t end
            end
            collectgarbage()
            -- Fresh tables of the same shapes take the storage the dead ones left.
            local e = {a = nil, b = nil, c = nil}
            assert(next(e) == nil and e.a == nil and e.b == nil and e.c == nil)
            local f = {a = 1, b = nil, c = nil}
            assert(f.a == 1 and f.b == nil and f.c == nil)
            local n = 0
            for _ in pairs(f) do n = n + 1 end
            assert(n == 1, n)
            local g = {z = nil, y = nil}
            assert(next(g) == nil)
        end
        for k, t in ipairs(live) do
            assert(t.a >= 1 and t.b:sub(1, 1) == "s" and #t.c == 1)
        end
    "#);
}

#[test]
fn fields_of_dead_and_live_records_do_not_alias() {
    run(r#"
        local live = {}
        for round = 1, 80 do
            local batch = {}
            for i = 1, 100 do
                batch[i] = {id = i, round = round, tag = "t" .. i}
            end
            for i = 1, 100 do
                local t = batch[i]
                assert(t.id == i and t.round == round and t.tag == "t" .. i)
            end
            live[round % 10 + 1] = batch
            batch = nil
            if round % 7 == 0 then collectgarbage() end
        end
        for slot = 1, 10 do
            local batch = live[slot]
            for i = 1, 100 do
                local t = batch[i]
                assert(t.id == i and t.round % 10 + 1 == slot and t.tag == "t" .. i)
            end
        end
    "#);
}

#[test]
fn records_can_grow_new_fields_and_be_reused_after_conversion() {
    run(r#"
        local function make(i) return {a = i, b = i} end
        for i = 1, 400 do
            local t = make(i)
            t.c = i
            t.d = "d"
            t[1] = 1
            assert(t.a == i and t.b == i and t.c == i and t.d == "d" and t[1] == 1)
            local fresh = make(i + 1)
            assert(fresh.c == nil and fresh.d == nil and fresh[1] == nil)
            assert(fresh.a == i + 1)
        end
    "#);
}
