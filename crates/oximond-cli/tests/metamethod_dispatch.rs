//! Operator and index metamethods reached from the interpreter's fast loop.

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
fn arithmetic_metamethods_store_their_first_result() {
    run(r#"
        local mt = {}
        mt.__add = function(a, b) return a.v + (type(b) == "table" and b.v or b), "extra" end
        mt.__sub = function(a, b) return nil end
        mt.__unm = function(a, b) assert(rawequal(a, b)); return -a.v end
        mt.__mul = function(a, b) end
        local x = setmetatable({v = 2}, mt)
        local y = setmetatable({v = 5}, mt)
        for i = 1, 300 do
            assert(x + y == 7)
            assert(x + i == 2 + i)
            assert(x - y == nil)
            assert(-x == -2)
            assert(x * y == nil)
        end
        -- The right operand's handler runs when the left has none.
        local left = setmetatable({}, {})
        local right = setmetatable({}, {__add = function(a, b) return "right" end})
        assert(left + right == "right")
        assert(1 + right == "right")
    "#);
}

#[test]
fn comparison_metamethods_store_their_truth() {
    run(r#"
        local mt = {}
        mt.__eq = function(a, b) return a.v == b.v and "yes" or nil end
        mt.__lt = function(a, b) return a.v < b.v and 1 or false end
        mt.__le = function(a, b) return a.v <= b.v and 1 or false end
        local a = setmetatable({v = 1}, mt)
        local b = setmetatable({v = 1}, mt)
        local c = setmetatable({v = 2}, mt)
        for i = 1, 300 do
            assert((a == b) == true)
            assert((a == c) == false)
            assert((a ~= c) == true)
            assert((a < c) == true)
            assert((c < a) == false)
            assert((a <= c) == true)
            assert((c <= a) == false)
        end
    "#);
}

#[test]
fn metamethod_results_reach_captured_locals() {
    run(r"
        local mt = {__add = function(a, b) return 40 + b end}
        local t = setmetatable({}, mt)
        local total = 0
        local function read() return total end
        for i = 1, 200 do
            total = t + 2
            assert(read() == 42)
        end
    ");
}

#[test]
fn index_chains_through_tables_and_functions() {
    run(r#"
        local base = {greet = function(self) return "hi " .. self.name end, level = 1}
        base.__index = base
        local derived = setmetatable({level = 2}, base)
        derived.__index = derived
        local object = setmetatable({name = "o"}, derived)
        local calls = 0
        local dynamic = setmetatable({}, {__index = function(t, k)
            calls = calls + 1
            return k .. "!"
        end})
        for i = 1, 300 do
            assert(object:greet() == "hi o")
            assert(object.level == 2)
            assert(object.missing == nil)
            assert(dynamic.x == "x!")
            assert(dynamic[i] == i .. "!")
        end
        assert(calls == 600)
        -- A stored nil still consults `__index`.
        local shadow = setmetatable({}, {__index = {k = "fallback"}})
        rawset(shadow, "k", nil)
        assert(shadow.k == "fallback")
        -- A table `__index` whose own metatable has a function handler.
        local deep = setmetatable({}, {__index = setmetatable({}, {__index = function(t, k)
            return "deep " .. k
        end})})
        assert(deep.z == "deep z")
    "#);
}

#[test]
fn metamethod_errors_and_missing_handlers_keep_their_messages() {
    run(r#"
        local ok, message = pcall(function() return {} + 1 end)
        assert(not ok and string.find(message, "arithmetic"), message)
        local ok, message = pcall(function() return -{} end)
        assert(not ok and string.find(message, "arithmetic"), message)
        local ok, message = pcall(function() return {} < {} end)
        assert(not ok and string.find(message, "compare"), message)
        local ok, message = pcall(function()
            return setmetatable({}, {__add = function() error("inside") end}) + 1
        end)
        assert(not ok and string.find(message, "inside"), message)
        assert(({} == {}) == false)
    "#);
}

#[test]
fn a_hook_set_inside_a_metamethod_takes_effect() {
    run(r#"
        local lines = 0
        local mt = {__add = function(a, b)
            debug.sethook(function() lines = lines + 1 end, "l")
            return 1
        end}
        local t = setmetatable({}, mt)
        local x = t + t
        local y = 1
        local z = 2
        debug.sethook()
        assert(x == 1 and lines > 0, lines)
    "#);
}

#[test]
fn concatenation_handlers_see_each_pairwise_step() {
    run(r#"
        local mt = {__concat = function(a, b)
            local left = type(a) == "table" and "[" .. a.name .. "]" or a
            local right = type(b) == "table" and "[" .. b.name .. "]" or b
            return left .. right
        end}
        local x = setmetatable({name = "x"}, mt)
        local y = setmetatable({name = "y"}, mt)
        for i = 1, 300 do
            assert(x .. y == "[x][y]")
            assert("a" .. x == "a[x]")
            assert(x .. 1 == "[x]1")
            -- Right-associated: the handler's result feeds the next step.
            assert("a" .. x .. "b" .. y == "a[x]b[y]")
            assert(1 .. 2 .. "3" == "123")
        end
        local ok, message = pcall(function() return "a" .. {} end)
        assert(not ok and string.find(message, "concatenate"), message)
    "#);
}

#[test]
fn native_handlers_receive_the_operands() {
    run(r#"
        local t = setmetatable({}, {__add = rawequal, __index = rawget})
        local u = setmetatable({}, getmetatable(t))
        for i = 1, 300 do
            assert(t + t == true)
            assert(t + u == false)
            assert(t.missing == nil)
        end
        rawset(t, "present", 1)
        assert(t.present == 1)
        local failing = setmetatable({}, {__index = string.rep})
        local ok, message = pcall(function() return failing.x end)
        assert(not ok, message)
    "#);
}
