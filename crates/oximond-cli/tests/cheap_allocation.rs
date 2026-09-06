//! Short-lived tables and closures come from recycled storage: constructors
//! take array buffers from a free list sized by the compiler's item counts,
//! tables are written into their arena slots, and closures keep none or one
//! capture inline. None of that may be visible. These programs run the
//! constructors often enough to go through the fast loop and the tracing
//! compiler, in every profile, and check weak tables, finalizers, the old
//! generation and closures made in loops against plain Lua semantics.

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
fn array_constructors_hold_exactly_their_items_and_grow_past_them() {
    run(r#"
        local unpack = table.unpack or unpack
        for i = 1, 600 do
            local one = {i}
            local two = {i, i + 1}
            local five = {i, i + 1, i + 2, i + 3, i + 4}
            assert(#one == 1 and #two == 2 and #five == 5)
            assert(one[1] == i and two[2] == i + 1 and five[5] == i + 4)
            assert(one[2] == nil and two[3] == nil and five[6] == nil)
            -- Growing past the sized buffer keeps every value.
            two[3] = "x"
            two[4] = "y"
            five[6] = 6
            for k = 7, 30 do five[k] = k end
            assert(#two == 4 and two[3] == "x" and two[4] == "y")
            assert(#five == 30 and five[1] == i and five[30] == 30)
            -- A hole in a constructor is a hole.
            local holes = {i, nil, i + 2}
            assert(holes[1] == i and holes[2] == nil and holes[3] == i + 2)
            -- Trailing multiple results are not counted in the size.
            local function three() return 1, 2, 3 end
            local spread = {three()}
            local mixed = {i, three()}
            assert(#spread == 3 and #mixed == 4 and mixed[4] == 3)
            local packed = {unpack(five, 1, 3)}
            assert(#packed == 3 and packed[3] == i + 2)
            -- Nested constructors and mixed array and record items.
            local nested = {{i}, {i, i}, {x = i, {i}}}
            assert(nested[1][1] == i and #nested[2] == 2 and nested[3].x == i)
            assert(nested[3][1][1] == i)
            local rec = {i, i + 1, name = "r", i + 2, [10] = 10}
            assert(#rec == 3 and rec.name == "r" and rec[10] == 10 and rec[3] == i + 2)
        end
    "#);
}

#[test]
fn large_constructors_past_the_counted_size_and_empty_ones_work() {
    run(r#"
        -- More items than the item count can carry (it stops at 255).
        local items = {}
        for i = 1, 300 do items[#items + 1] = tostring(i) end
        local source = "return {" .. table.concat(items, ",") .. "}"
        local chunk = (loadstring or load)(source)
        for round = 1, 20 do
            local t = chunk()
            assert(#t == 300 and t[1] == 1 and t[255] == 255 and t[300] == 300)
        end
        for i = 1, 500 do
            local empty = {}
            assert(next(empty) == nil and #empty == 0)
            empty[1] = i
            assert(#empty == 1 and empty[1] == i)
        end
    "#);
}

#[test]
fn recycled_array_buffers_never_show_old_contents() {
    run(r#"
        for round = 1, 40 do
            local batch = {}
            for i = 1, 300 do batch[i] = {"s" .. i, i, true} end
            for i = 1, 300 do
                assert(batch[i][1] == "s" .. i and batch[i][2] == i and batch[i][3] == true)
            end
            batch = nil
            collectgarbage()
            -- Fresh tables of the same sizes start empty, not with the
            -- previous tenants' values.
            for i = 1, 300 do
                local fresh = {}
                assert(fresh[1] == nil and #fresh == 0)
                fresh[2] = "late"
                assert(fresh[1] == nil)
                local sized = {nil, nil, nil}
                assert(sized[1] == nil and sized[2] == nil and sized[3] == nil)
            end
        end
    "#);
}

#[test]
fn weak_tables_drop_pooled_array_and_record_tables() {
    run(r#"
        local weak_values = setmetatable({}, {__mode = "v"})
        local weak_keys = setmetatable({}, {__mode = "k"})
        local strong = {}
        for i = 1, 400 do
            local array = {i, i + 1, i + 2}
            local record = {a = i, b = i + 1, c = i + 2}
            weak_values[i] = array
            weak_values[-i] = record
            weak_keys[array] = i
            weak_keys[record] = i
            if i % 50 == 0 then strong[#strong + 1] = {array, record} end
        end
        collectgarbage()
        collectgarbage()
        local kept_values, kept_keys = 0, 0
        for _, v in pairs(weak_values) do kept_values = kept_values + 1 end
        for _, v in pairs(weak_keys) do kept_keys = kept_keys + 1 end
        -- Only the tables the strong list still holds survive, with their
        -- contents intact.
        assert(kept_values == 16, kept_values)
        assert(kept_keys == 16, kept_keys)
        for n, pair in ipairs(strong) do
            local i = n * 50
            assert(pair[1][1] == i and pair[1][3] == i + 2)
            assert(pair[2].a == i and pair[2].c == i + 2)
            assert(weak_values[i] == pair[1] and weak_values[-i] == pair[2])
            assert(weak_keys[pair[1]] == i and weak_keys[pair[2]] == i)
        end
        -- The freed slots and buffers are reused without disturbing them.
        for i = 1, 2000 do local t = {i, i}; local r = {a = i, b = i} end
        for n, pair in ipairs(strong) do
            assert(pair[1][2] == n * 50 + 1 and pair[2].b == n * 50 + 1)
        end
    "#);
}

#[test]
fn finalizers_see_intact_pooled_tables_once_each() {
    run(r#"
        local probe = false
        setmetatable({}, {__gc = function() probe = true end})
        collectgarbage()
        collectgarbage()
        if not probe then return end -- no table finalizers in this profile

        local calls, sum, order = 0, 0, {}
        local mt = {__gc = function(t)
            calls = calls + 1
            sum = sum + t[1] + t[2] + t.id
            order[#order + 1] = t.id
        end}
        for i = 1, 200 do
            local t = {i, i * 2, id = i}
            setmetatable(t, mt)
        end
        for i = 1, 200 do
            local t = {i, i * 2}
            t.id = 1000 + i
            setmetatable(t, mt)
        end
        collectgarbage()
        collectgarbage()
        assert(calls == 400, calls)
        local want = 0
        for i = 1, 200 do want = want + i + i * 2 + i + i + i * 2 + 1000 + i end
        assert(sum == want, sum)
        -- Finalizers run in the reverse of the order they were marked in.
        assert(order[1] == 1200 and order[200] == 1001 and order[201] == 200 and order[400] == 1)
        collectgarbage()
        assert(calls == 400, "a finalizer ran twice")
        -- Storage freed after the finalizers is reusable.
        for i = 1, 500 do local t = {i, i} assert(t[1] == i and t[2] == i) end
    "#);
}

#[test]
fn a_finalized_table_that_is_resurrected_keeps_its_array() {
    run(r"
        local probe = false
        setmetatable({}, {__gc = function() probe = true end})
        collectgarbage()
        collectgarbage()
        if not probe then return end

        local revived = {}
        local mt = {__gc = function(t) revived[#revived + 1] = t end}
        for i = 1, 100 do setmetatable({i, i + 1, i + 2}, mt) end
        collectgarbage()
        assert(#revived == 100, #revived)
        for _, t in ipairs(revived) do
            assert(#t == 3 and t[3] == t[1] + 2)
        end
        -- Churn the allocator while the revived tables are alive.
        for i = 1, 3000 do local x = {i, i, i} end
        for _, t in ipairs(revived) do assert(t[2] == t[1] + 1) end
        revived = nil
        collectgarbage()
    ");
}

#[test]
fn tables_surviving_into_the_old_generation_keep_their_contents() {
    run(r#"
        pcall(collectgarbage, "generational")
        local old = {}
        for i = 1, 300 do
            old[i] = {i, i + 1, i + 2, tag = "t" .. i}
            if i % 20 == 0 then collectgarbage("step") end
        end
        -- Two full cycles age every table into the old generation.
        collectgarbage()
        collectgarbage()
        -- Young tables stored into old ones and old ones rewritten: the
        -- write barrier has to keep the young ones alive.
        for round = 1, 20 do
            for i = 1, 300, 7 do
                old[i].child = {round, i}
                old[i][4] = {round * i}
            end
            for i = 1, 400 do local garbage = {i, i, i} end
            collectgarbage("step")
            for i = 1, 300, 7 do
                assert(old[i].child[1] == round and old[i].child[2] == i)
                assert(old[i][4][1] == round * i)
            end
        end
        collectgarbage()
        for i = 1, 300 do
            assert(old[i][1] == i and old[i][3] == i + 2 and old[i].tag == "t" .. i)
        end
        -- An old table emptied and refilled reuses its buffer correctly.
        for i = 1, 300 do
            local t = old[i]
            for k = #t, 1, -1 do t[k] = nil end
            for k = 1, 6 do t[k] = k * i end
        end
        collectgarbage()
        for i = 1, 300 do assert(#old[i] == 6 and old[i][6] == 6 * i) end
        pcall(collectgarbage, "incremental")
    "#);
}

#[test]
fn closures_made_in_a_loop_capture_their_own_iteration() {
    run(r"
        local none, one, two, three, nine = {}, {}, {}, {}, {}
        for i = 1, 300 do
            local a, b, c = i, i * 2, i * 3
            local d, e, f, g, h, j = i + 1, i + 2, i + 3, i + 4, i + 5, i + 6
            none[i] = function() return 7 end
            one[i] = function() return a end
            two[i] = function() return a + b end
            three[i] = function() return a + b + c end
            nine[i] = function() return a + b + c + d + e + f + g + h + j end
        end
        collectgarbage()
        for i = 1, 300 do
            assert(none[i]() == 7)
            assert(one[i]() == i)
            assert(two[i]() == i * 3)
            assert(three[i]() == i * 6)
            assert(nine[i]() == i * 6 + (i + 1) + (i + 2) + (i + 3) + (i + 4) + (i + 5) + (i + 6))
        end
    ");
}

#[test]
fn closures_that_share_a_variable_see_each_others_writes() {
    run(r"
        local getters, setters = {}, {}
        for i = 1, 300 do
            local shared = i
            local other = -i
            getters[i] = function() return shared, other end
            setters[i] = function(v) shared = v; other = other - 1 end
        end
        for i = 1, 300 do
            local s, o = getters[i]()
            assert(s == i and o == -i)
            setters[i](i * 10)
            s, o = getters[i]()
            assert(s == i * 10 and o == -i - 1, s)
        end
        collectgarbage()
        for i = 1, 300 do
            setters[i](0)
            local s, o = getters[i]()
            assert(s == 0 and o == -i - 2)
        end
        -- A loop variable captured by closures made in a nested function.
        local makers = {}
        for i = 1, 200 do
            makers[i] = (function(base)
                local count = 0
                return function() count = count + base; return count end
            end)(i)
        end
        for i = 1, 200 do
            assert(makers[i]() == i and makers[i]() == 2 * i)
        end
    ");
}

#[test]
fn upvalue_inspection_and_replacement_work_for_small_and_large_capture_lists() {
    run(r#"
        if not (debug and debug.getupvalue and debug.setupvalue) then return end
        for i = 1, 100 do
            local a, b, c = i, i + 1, i + 2
            local one = function() return a end
            local three = function() return a + b + c end
            local name, value = debug.getupvalue(one, 1)
            assert(name == "a" and value == i)
            assert(debug.getupvalue(one, 2) == nil)
            debug.setupvalue(one, 1, 500)
            assert(one() == 500 and a == 500)
            local names = {}
            for k = 1, 3 do names[k] = debug.getupvalue(three, k) end
            assert(names[1] == "a" and names[2] == "b" and names[3] == "c")
            debug.setupvalue(three, 2, 1000)
            assert(b == 1000 and three() == 500 + 1000 + i + 2)
            if debug.upvaluejoin then
                local x, y = i, -i
                local reads_x = function() return x end
                local reads_y = function() return y end
                debug.upvaluejoin(reads_y, 1, reads_x, 1)
                assert(reads_y() == i and reads_x() == i)
                x = 77
                assert(reads_y() == 77)
                local p, q, r = 1, 2, 3
                local wide = function() return p + q + r end
                local narrow = function() return x end
                debug.upvaluejoin(wide, 3, narrow, 1)
                assert(wide() == 1 + 2 + 77)
                x = 3
                assert(wide() == 6)
            end
        end
        collectgarbage()
    "#);
}

#[test]
fn closures_survive_collection_with_their_captures_in_every_generation() {
    run(r#"
        pcall(collectgarbage, "generational")
        local keep = {}
        for i = 1, 400 do
            local a, b = i, {i, i + 1}
            keep[i] = function() return a, b[1], b[2] end
            if i % 40 == 0 then collectgarbage("step") end
        end
        collectgarbage()
        for round = 1, 10 do
            for i = 1, 300 do local f = function() return i end end
            collectgarbage("step")
        end
        collectgarbage()
        for i = 1, 400 do
            local a, b1, b2 = keep[i]()
            assert(a == i and b1 == i and b2 == i + 1)
        end
        pcall(collectgarbage, "incremental")
    "#);
}

#[test]
fn coroutines_and_errors_inside_closure_and_table_churn() {
    run(r"
        local co = coroutine.wrap(function()
            for i = 1, 200 do
                local t = {i, i + 1}
                local f = function() return t[1] + t[2] end
                coroutine.yield(f())
            end
        end)
        for i = 1, 200 do assert(co() == 2 * i + 1) end
        for i = 1, 200 do
            local ok, err = pcall(function()
                local t = {i, i}
                local f = function() return t[1] end
                error({f()})
            end)
            assert(not ok and err[1] == i)
        end
    ");
}
