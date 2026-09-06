//! The first string-key stores into fresh tables, in both execution modes:
//! a new key in an empty or a small table, growth past the ordered
//! representation, iteration order and length afterwards, weak tables,
//! `__newindex` and `__index`, and tables that hold each other.

use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua54,
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
            if let Err(error) = lua.exec(source) {
                panic!("{profile} {mode}: {error:?}");
            }
        }
    }
}

#[test]
fn a_new_key_lands_in_an_empty_table() {
    run(r#"
        for i = 1, 3000 do
            local t = {}
            t.name = i
            assert(t.name == i and #t == 0)
            local n = 0
            for k, v in pairs(t) do n = n + 1; assert(k == "name" and v == i) end
            assert(n == 1)
        end
    "#);
}

#[test]
fn tables_grow_key_by_key_and_keep_their_order() {
    run(r#"
        local names = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
            "m", "n", "o", "p", "q", "r", "s", "t"}
        for round = 1, 300 do
            for count = 0, #names do
                local t = {}
                for i = 1, count do t[names[i]] = i end
                local seen = {}
                for k, v in pairs(t) do seen[#seen + 1] = k; assert(t[k] == v) end
                assert(#seen == count)
                table.sort(seen)
                for i = 1, count do assert(seen[i] == names[i]) end
                assert(#t == 0)
                local k, v = next(t)
                if count == 0 then assert(k == nil) else assert(k ~= nil and v ~= nil) end
            end
        end
    "#);
}

#[test]
fn array_part_and_string_keys_share_a_table() {
    run(r#"
        for round = 1, 2000 do
            local t = {}
            t.x = round
            t[1] = "one"
            t.y = round + 1
            t[2] = "two"
            assert(#t == 2 and t.x == round and t.y == round + 1)
            t.x = nil
            assert(t.x == nil and t.y == round + 1 and #t == 2)
            t.z = 3
            local n = 0
            for _ in pairs(t) do n = n + 1 end
            assert(n == 4)
        end
    "#);
}

#[test]
fn deleted_keys_can_come_back_in_a_small_table() {
    run(r"
        for round = 1, 2000 do
            local t = {}
            t.a = 1
            t.a = nil
            assert(next(t) == nil)
            t.a = 2
            t.b = 3
            t.a = nil
            t.c = 4
            assert(t.a == nil and t.b == 3 and t.c == 4)
        end
    ");
}

#[test]
fn newindex_and_index_metatables_see_new_keys_of_fresh_tables() {
    run(r#"
        local log = {}
        local proxy = {__newindex = function(t, k, v) log[#log + 1] = k; rawset(t, k, v) end}
        local inherit = {__index = {base = "from index"}}
        for round = 1, 500 do
            local watched = setmetatable({}, proxy)
            watched.first = round
            watched.first = round + 1
            watched.second = 2
            assert(watched.first == round + 1 and watched.second == 2)
            local plain = setmetatable({}, inherit)
            plain.own = round
            assert(plain.own == round and plain.base == "from index")
            assert(rawget(plain, "base") == nil)
        end
        assert(#log == 1000, #log)
        assert(log[1] == "first" and log[2] == "second")
    "#);
}

#[test]
fn weak_tables_drop_dead_values_stored_under_new_keys() {
    run(r#"
        local weak = setmetatable({}, {__mode = "v"})
        local strong = {}
        for i = 1, 300 do
            local value = {}
            weak["k" .. i] = value
            if i % 3 == 0 then strong[#strong + 1] = value end
        end
        collectgarbage()
        collectgarbage()
        local live = 0
        for _ in pairs(weak) do live = live + 1 end
        assert(live == 100, live)

        local weak_keys = setmetatable({}, {__mode = "k"})
        local kept = {}
        for i = 1, 300 do
            local key = {}
            key.tag = i
            weak_keys[key] = i
            if i % 5 == 0 then kept[#kept + 1] = key end
        end
        collectgarbage()
        collectgarbage()
        live = 0
        for _ in pairs(weak_keys) do live = live + 1 end
        assert(live == 60, live)
    "#);
}

#[test]
fn tables_that_hold_each_other_survive_and_die_with_the_collector() {
    run(r#"
        local probe = setmetatable({}, {__mode = "k"})
        local roots = {}
        for round = 1, 40 do
            for i = 1, 500 do
                local left, right = {}, {}
                left.other = right
                right.other = left
                assert(left.other.other == left)
                assert(right.other.other == right)
                roots[i] = left
                if i == 1 then probe[left] = true end
            end
            for i = 1, 500 do
                local left = roots[i]
                assert(left.other.other == left)
                assert(left.other ~= left)
            end
            if round % 2 == 0 then roots = {} end
            collectgarbage()
        end
        roots = nil
        collectgarbage()
        collectgarbage()
        assert(next(probe) == nil)
    "#);
}

#[test]
fn cyclic_tables_keep_their_fields_after_a_loop_exit() {
    run(r"
        local last
        for i = 1, 5000 do
            local a, b = {}, {}
            a.peer = b
            a.n = i
            b.peer = a
            b.n = -i
            last = a
        end
        assert(last.n == 5000 and last.peer.n == -5000)
        assert(last.peer.peer == last)
        local count = 0
        for k in pairs(last) do count = count + 1 end
        assert(count == 2)
    ");
}

#[test]
fn a_table_can_hold_itself() {
    run(r"
        local keep
        for i = 1, 5000 do
            local t = {}
            t.me = t
            t.i = i
            assert(t.me == t and t.me.me.i == i)
            keep = t
        end
        assert(keep.me == keep and keep.i == 5000)
    ");
}

#[test]
fn a_cycle_through_three_tables_and_a_later_overwrite() {
    run(r"
        local sum, expected = 0, 0
        for i = 1, 4000 do expected = expected + i % 7 end
        for i = 1, 4000 do
            local a, b, c = {}, {}, {}
            a.next = b
            b.next = c
            c.next = a
            c.next = nil
            c.next = b
            assert(a.next.next.next == b)
            assert(c.next.next == c)
            sum = sum + i % 7
        end
        assert(sum == expected, sum)
    ");
}

#[test]
fn exits_after_a_cycle_closes_leave_whole_tables_behind() {
    run(r"
        local kept = {}
        local odd = 0
        for i = 1, 6000 do
            local a, b = {}, {}
            a.peer = b
            b.peer = a
            if i % 97 == 0 then
                -- a branch the recording did not take
                a.extra = i
                b.extra = -i
                kept[#kept + 1] = a
            end
            if i % 2 == 1 then odd = odd + 1 end
            assert(a.peer.peer == a and b.peer.peer == b)
        end
        assert(odd == 3000)
        assert(#kept == 61)
        for _, a in ipairs(kept) do
            assert(a.extra == -a.peer.extra and a.peer.peer == a)
        end
    ");
}

#[test]
fn a_cycle_closed_by_a_number_keyed_store_still_works() {
    run(r"
        local keep
        for i = 1, 4000 do
            local a, b = {}, {}
            a[1] = b
            b[1] = a
            b.label = i
            keep = a
            assert(a[1][1] == a and #a == 1)
        end
        assert(keep[1].label == 4000)
    ");
}

#[test]
fn nil_stores_into_a_shaped_table_in_a_cycle_keep_later_fills() {
    run(r#"
        local total, kept = 0, nil
        for i = 1, 20000 do
            local left, right = {a = 1, b = 0, c = 0}, {}
            left.b = nil
            left.c = nil
            left.other = right
            right.other = left
            left.a = nil
            left.b = 2.5
            left.a = i
            left.c = i + 0.5
            if i % 999 == 0 then kept = left end
            total = total + (left.b or 0) + (left.c or 0) + (left.a or 0)
        end
        assert(total == 400080000, total)
        local keys = {}
        for k in pairs(kept) do keys[#keys + 1] = k end
        table.sort(keys)
        assert(table.concat(keys, ",") == "a,b,c,other", table.concat(keys, ","))
        assert(kept.a == 19980 and kept.b == 2.5 and kept.c == 19980.5)
        assert(kept.other.other == kept)
    "#);
}
