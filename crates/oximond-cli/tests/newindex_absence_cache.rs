//! A metatable that gains `__newindex` after stores cached its absence must
//! see the next absent-key store go through it. Each case runs a loop so the
//! constructor shape and the write site are filled before the metamethod
//! appears.

use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 2] =
    [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54];

/// Runs `source`, which must `return true` when every check holds.
fn check(source: &str) {
    for profile in PROFILES {
        let mut lua = Lua::builder()
            .profile(profile)
            .execution_mode(ExecutionMode::Interpreter)
            .build()
            .unwrap();
        let result = lua
            .exec(source)
            .unwrap_or_else(|error| panic!("{profile:?}: {error}"));
        assert_eq!(
            format!("{result:?}"),
            "[Boolean(true)]",
            "{profile:?}: {source}"
        );
    }
}

#[test]
fn newindex_added_after_stores_straight_line() {
    check(
        "local mt = {}
         local o = setmetatable({a = 1}, mt)
         o.b = 2
         mt.__newindex = function(t, k, v) rawset(t, k, v * 10) end
         o.c = 3
         return o.b == 2 and o.c == 30",
    );
}

#[test]
fn newindex_added_after_stores_in_a_loop() {
    check(
        "local mt = {}
         local ok = true
         for i = 1, 6 do
             local o = setmetatable({a = 1}, mt)
             o.b = i
             if i <= 4 and o.b ~= i then ok = false end
             if i > 4 and o.b ~= i * 10 then ok = false end
             if i == 4 then
                 mt.__newindex = function(t, k, v) rawset(t, k, v * 10) end
             end
         end
         local o = setmetatable({a = 1}, mt)
         o.b = 7
         return ok and o.b == 70",
    );
}

#[test]
fn newindex_added_between_iterations_hits_the_filled_site() {
    // The same store site runs on every pass: raw before, handler after.
    check(
        "local mt = {}
         local seen = {}
         for i = 1, 8 do
             local o = setmetatable({a = 1}, mt)
             o.b = i
             seen[i] = o.b
             if i == 5 then
                 mt.__newindex = function(t, k, v) rawset(t, k, -v) end
             end
         end
         return seen[4] == 4 and seen[5] == 5 and seen[6] == -6 and seen[8] == -8",
    );
}

#[test]
fn local_setmetatable_alias_is_tracked() {
    check(
        "local setmetatable = setmetatable
         local mt = {}
         local last
         for i = 1, 6 do
             local o = setmetatable({a = 1}, mt)
             o.b = i
             last = o.b
             if i == 3 then
                 mt.__newindex = function(t, k, v) rawset(t, k, v * 10) end
             end
         end
         return last == 60",
    );
}

#[test]
fn newindex_table_receives_absent_keys() {
    check(
        "local sink = {}
         local mt = {}
         local o = setmetatable({a = 1}, mt)
         o.b = 2
         mt.__newindex = sink
         o.c = 3
         o.a = 9
         return o.b == 2 and rawget(o, 'c') == nil and sink.c == 3 and o.a == 9",
    );
}

#[test]
fn newindex_table_added_after_the_site_fills() {
    check(
        "local sink = {}
         local mt = {}
         local ok = true
         for i = 1, 6 do
             local o = setmetatable({a = 1}, mt)
             o.b = i
             if i <= 4 and rawget(o, 'b') ~= i then ok = false end
             if i > 4 and (rawget(o, 'b') ~= nil or sink.b ~= i) then ok = false end
             if i == 4 then mt.__newindex = sink end
         end
         return ok",
    );
}

#[test]
fn stored_nil_counts_as_absent_for_newindex() {
    check(
        "local calls = 0
         local mt = {__newindex = function(t, k, v) calls = calls + 1; rawset(t, k, v) end}
         local o = setmetatable({}, mt)
         o.x = 1
         o.x = nil
         o.x = 2
         return calls == 2 and o.x == 2",
    );
}

#[test]
fn newindex_set_to_nil_again_stores_raw() {
    check(
        "local mt = {__newindex = function(t, k, v) rawset(t, k, v + 1) end}
         local o = setmetatable({}, mt)
         o.a = 1
         mt.__newindex = nil
         o.b = 1
         return o.a == 2 and o.b == 1",
    );
}

#[test]
fn table_that_is_its_own_metatable() {
    check(
        "local t = {a = 1}
         setmetatable(t, t)
         t.b = 2
         t.__newindex = function(t, k, v) rawset(t, k, v * 10) end
         t.c = 3
         return t.b == 2 and rawget(t, 'c') == 30",
    );
}

#[test]
fn shaped_class_slot_filled_by_a_site_store_is_visible_through_index() {
    // A key that a constructor's straight-line code stores later is a nil
    // slot of the shape until that store runs. `mk` builds a class, reads
    // `greet` through an object (the inherited-read cache then records that
    // no table has it), and only then runs `t.greet = val`. The first pass
    // adds the slot through the slow store, which fills the write site. Every
    // later pass adds it through the site, which must drop the cached miss
    // so the object's next read sees the value.
    check(
        "local val = 'v'
         local before, after, later = {}, {}, {}
         local function mk(i)
             local t = {name = 'x', __index = false}
             t.__index = t
             local o = setmetatable({}, t)
             before[i] = o.greet
             t.greet = val
             after[i] = o.greet
             return o
         end
         local prev
         for i = 1, 6 do
             if prev then later[i] = prev.greet end
             prev = mk(i)
             val = 'v' .. i
         end
         for i = 1, 6 do
             if before[i] ~= nil or after[i] ~= (i == 1 and 'v' or 'v' .. (i - 1)) then
                 return false
             end
         end
         return later[2] == 'v' and later[6] == 'v4'",
    );
}

#[test]
fn constructor_shaped_class_filled_in_a_loop() {
    check(
        "local out = {}
         for i = 1, 6 do
             local class = {a = 1, b = nil}
             class.__index = class
             local o = setmetatable({}, class)
             local before = o.b
             class.b = i
             out[i] = (before == nil) and o.b == i
         end
         for i = 1, 6 do if not out[i] then return false end end
         return true",
    );
}

#[test]
fn protected_metatable_added_late() {
    check(
        "local mt = {}
         local o = setmetatable({}, mt)
         local _ = getmetatable(o)
         mt.__metatable = 'locked'
         local ok = pcall(setmetatable, o, {})
         return getmetatable(o) == 'locked' and not ok",
    );
}
