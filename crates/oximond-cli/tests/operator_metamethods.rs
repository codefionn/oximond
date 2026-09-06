//! Operator metamethods with plain Lua handlers, which the fast loop enters
//! directly: handler choice, per-profile comparison rules, error messages,
//! and handlers that change while a loop runs. Every script runs in each
//! profile and execution mode with `PROFILE`, `LUA51` (the 5.1 comparison
//! and length rules) and `LUA53` (5.3 and later) defined.

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
    run_in(&PROFILES, source);
}

fn run_in(profiles: &[CompatibilityProfile], source: &str) {
    for &profile in profiles {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let selector = profile.selector();
            let prelude = format!(
                "local PROFILE = '{selector}'\n\
                 local LUA51 = PROFILE == 'lua51' or PROFILE == 'luajit21'\n\
                 local LUA53 = PROFILE == 'lua53' or PROFILE == 'lua54' or PROFILE == 'lua55'\n"
            );
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            if let Err(error) = lua.exec(format!("{prelude}{source}")) {
                panic!("{profile} {mode}: {error:?}");
            }
        }
    }
}

#[test]
fn the_left_operand_chooses_the_handler_before_the_right() {
    run(r#"
        local L = setmetatable({id = "L"}, {
            __add = function(a, b) return "L" end,
            __sub = function(a, b) return "Lsub" end,
        })
        local R = setmetatable({id = "R"}, {
            __add = function(a, b) return "R" end,
            __mul = function(a, b) return "Rmul" end,
        })
        local D = setmetatable({id = "x"}, {__div = function(a, b)
            return (type(a) == "table" and a.id or a) .. "/" .. (type(b) == "table" and b.id or b)
        end})
        local E = setmetatable({id = "y"}, getmetatable(D))
        for i = 1, 300 do
            assert(L + R == "L")
            assert(R + L == "R")
            assert(L + 1 == "L")
            assert(1 + L == "L")
            assert(R + 1.5 == "R")
            assert(L - R == "Lsub")
            assert(R - L == "Lsub")
            assert(L * R == "Rmul")
            assert(D / 2 == "x/2")
            assert(2 / D == "2/x")
            assert(D / E == "x/y")
            assert(E / D == "y/x")
        end
        local ok, message = pcall(function() return L / R end)
        assert(not ok and message:find("attempt to perform arithmetic on") and message:find("table value"), message)
        local ok, message = pcall(function() return 1 % L end)
        assert(not ok and message:find("attempt to perform arithmetic on") and message:find("table value"), message)
    "#);
}

#[test]
fn constants_and_every_binary_operator_reach_their_handlers() {
    run(r#"
        local seen = {}
        local mt = {}
        for _, name in ipairs {"add", "sub", "mul", "div", "mod", "pow"} do
            mt["__" .. name] = function(a, b)
                return name .. ":" .. (type(a) == "table" and "t" or tostring(a))
                    .. ":" .. (type(b) == "table" and "t" or tostring(b))
            end
        end
        local t = setmetatable({}, mt)
        for i = 1, 300 do
            assert(t + 2 == "add:t:2")
            assert(2 + t == "add:2:t")
            assert(t - 2 == "sub:t:2")
            assert(t * 3 == "mul:t:3")
            assert(t / 4 == "div:t:4")
            assert(t % 5 == "mod:t:5")
            assert(t ^ 6 == "pow:t:6")
            assert(t + t == "add:t:t")
        end
    "#);
}

#[test]
fn bitwise_and_floor_division_handlers_in_the_profiles_that_have_them() {
    run_in(
        &[
            CompatibilityProfile::Lua53,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::Lua55,
        ],
        r#"
        local mt = {}
        for _, name in ipairs {"idiv", "band", "bor", "bxor", "shl", "shr"} do
            mt["__" .. name] = function(a, b) return name end
        end
        mt.__bnot = function(a, b) return "bnot" end
        local t = setmetatable({}, mt)
        for i = 1, 300 do
            assert(load("local t = ...; return t // 2")(t) == "idiv")
            assert(load("local t = ...; return t & 2")(t) == "band")
            assert(load("local t = ...; return 1 | t")(t) == "bor")
            assert(load("local t = ...; return t ~ t")(t) == "bxor")
            assert(load("local t = ...; return t << 1")(t) == "shl")
            assert(load("local t = ...; return t >> 1")(t) == "shr")
            assert(load("local t = ...; return ~t")(t) == "bnot")
        end
        local ok, message = pcall(load("local t = ...; return t | 1"), setmetatable({}, {}))
        assert(not ok and message:find("attempt to perform bitwise operation on") and message:find("table value"), message)
    "#,
    );
}

#[test]
fn unary_minus_passes_its_operand_twice() {
    run(r##"
        local count
        local t = setmetatable({}, {__unm = function(a, b)
            assert(rawequal(a, b))
            return "neg"
        end})
        local nargs = setmetatable({}, {__unm = function(...) return select("#", ...) end})
        for i = 1, 300 do
            assert(-t == "neg")
            assert(-nargs == 2)
        end
        local ok, message = pcall(function() return -setmetatable({}, {}) end)
        assert(not ok and message:find("attempt to perform arithmetic on") and message:find("table value"), message)
    "##);
}

#[test]
fn equality_runs_only_for_two_tables_and_applies_the_profile_rule() {
    run(r#"
        local calls = 0
        local same = function(a, b) calls = calls + 1; return true end
        local mt1 = {__eq = same}
        local a = setmetatable({}, mt1)
        local b = setmetatable({}, mt1)
        local c = setmetatable({}, {__eq = function(x, y) calls = calls + 1; return true end})
        local plain = setmetatable({}, {})
        for i = 1, 300 do
            assert(a == b)
            assert(not (a ~= b))
            if LUA53 then
                -- The left handler, else the right one.
                assert(a == c)
                assert(c == a)
                assert(plain == a)
                assert(a == plain)
            else
                -- Both handlers must be the same function.
                assert(not (a == c))
                assert(not (c == a))
                assert(not (plain == a))
                assert(a ~= plain)
            end
        end
        -- Raw-equal values and values of other types never call it.
        calls = 0
        for i = 1, 300 do
            assert(a == a)
            assert(not (a == 1))
            assert(not (a == "a"))
            assert(a ~= nil)
            assert(not (a == print))
        end
        assert(calls == 0, calls)
        -- The result is converted to a boolean.
        local truthy = setmetatable({}, {__eq = function() return "yes" end})
        local falsy = setmetatable({}, getmetatable(truthy))
        getmetatable(truthy).__eq = function() return nil end
        assert((truthy == falsy) == false)
        getmetatable(truthy).__eq = function() return 0 end
        assert((truthy == falsy) == true)
        if true then
            assert(truthy == falsy)
        end
    "#);
}

#[test]
fn ordering_handlers_convert_their_result_and_follow_the_profile_rule() {
    run(r#"
        local mt = {__lt = function(a, b)
            return (type(a) == "table" and a.v or a) < (type(b) == "table" and b.v or b) and 1 or nil
        end}
        local a = setmetatable({v = 1}, mt)
        local b = setmetatable({v = 2}, mt)
        local other = setmetatable({v = 2}, {__lt = function() return true end})
        for i = 1, 300 do
            assert((a < b) == true)
            assert((b < a) == false)
            assert((b > a) == true)
            assert((a > b) == false)
            if a < b then else error("branch") end
            if not (b < a) then else error("branch") end
            if LUA51 then
                -- Operands of one type, with the same handler.
                assert(not pcall(function() return a < 3 end))
                assert(not pcall(function() return 3 < a end))
                assert(not pcall(function() return a < other end))
            else
                assert((a < 3) == true)
                assert((0 < a) == true)
                assert((a < other) == true)
            end
        end
        local ok, message = pcall(function() return {} < {} end)
        assert(not ok and message:find("attempt to compare two table values"), message)
        local ok, message = pcall(function() return {} < 1 end)
        assert(not ok and message:find("attempt to compare"), message)
    "#);
}

#[test]
fn less_equal_falls_back_to_the_negated_less_than_where_the_profile_has_it() {
    run(r#"
        local mt = {__lt = function(a, b) return a.v < b.v end}
        local a = setmetatable({v = 1}, mt)
        local b = setmetatable({v = 2}, mt)
        local c = setmetatable({v = 1}, mt)
        if PROFILE == "lua55" then
            for i = 1, 300 do
                local ok, message = pcall(function() return a <= b end)
                assert(not ok and message:find("attempt to compare two table values"), message)
                local ok = pcall(function() if a <= b then return 1 end end)
                assert(not ok)
            end
        else
            for i = 1, 300 do
                assert((a <= b) == true)
                assert((b <= a) == false)
                assert((a <= c) == true)
                assert((b >= a) == true)
                assert((a >= b) == false)
                if a <= c then else error("branch") end
                if b <= a then error("branch") end
            end
        end
        -- With its own `__le`, that handler decides and `__lt` is not asked.
        local asked = 0
        local own = {
            __lt = function() asked = asked + 1; return true end,
            __le = function(x, y) return "le" end,
        }
        local x, y = setmetatable({}, own), setmetatable({}, own)
        for i = 1, 300 do
            assert((x <= y) == true)
            if not (y <= x) then error("branch") end
        end
        assert(asked == 0)
        -- A handler for `__le` that is false stays false.
        local no = setmetatable({}, {__le = function() return false end, __lt = function() return true end})
        for i = 1, 300 do
            assert((no <= no) == false)
        end
    "#);
}

#[test]
fn concatenation_handlers_see_numbers_and_strings_on_either_side() {
    run(r#"
        local C = setmetatable({}, {__concat = function(a, b)
            return (type(a) == "table" and "C" or tostring(a)) .. "+"
                .. (type(b) == "table" and "C" or tostring(b))
        end})
        local D = setmetatable({}, getmetatable(C))
        for i = 1, 300 do
            assert(C .. 1 == "C+1")
            assert(1 .. C == "1+C")
            assert("s" .. C == "s+C")
            assert(C .. "s" == "C+s")
            assert(C .. 2.5 == "C+2.5")
            assert(C .. D == "C+C")
            assert(nil .. C == "nil+C")
            assert(C .. true == "C+true")
        end
        -- A plain table has none.
        local ok, message = pcall(function() return {} .. "x" end)
        assert(not ok and message:find("attempt to concatenate") and message:find("table value"), message)
        local ok, message = pcall(function() return "x" .. {} end)
        assert(not ok and message:find("attempt to concatenate") and message:find("table value"), message)
        -- The left handler wins over the right.
        local left = setmetatable({}, {__concat = function() return "left" end})
        local right = setmetatable({}, {__concat = function() return "right" end})
        assert(left .. right == "left")
        assert(right .. left == "right")
        assert(1 .. right == "right")
    "#);
}

#[test]
fn length_handlers_apply_to_tables_and_never_to_strings() {
    run(r##"
        local T = setmetatable({1, 2, 3}, {__len = function(t, u)
            return rawequal(t, u) and 42 or -1
        end})
        for i = 1, 300 do
            if LUA51 then
                assert(#T == 3)
            else
                assert(#T == 42)
            end
            assert(#"abc" == 3)
        end
        -- Even a string metatable's `__len` is not asked for strings.
        local string_meta = getmetatable("")
        string_meta.__len = function() return 99 end
        assert(#"abc" == 3)
        string_meta.__len = nil
        local nargs = setmetatable({}, {__len = function(...) return select("#", ...) end})
        assert(#nargs == (LUA51 and 0 or 2))
    "##);
}

#[test]
fn a_handler_replaced_between_iterations_takes_effect_at_once() {
    run(r#"
        local mt = {__add = function() return 1 end}
        local t = setmetatable({}, mt)
        local results = {}
        for i = 1, 400 do
            if i == 100 then mt.__add = function() return 2 end end
            if i == 200 then mt.__add = nil end
            if i == 250 then rawset(mt, "__add", function() return 3 end) end
            if i == 300 then
                local key = "__" .. "add"
                mt[key] = rawequal
            end
            if i == 350 then mt.__add = setmetatable({}, {__call = function(_, a, b) return "call" end}) end
            if i >= 200 and i < 250 then
                local ok, message = pcall(function() return t + 1 end)
                assert(not ok and message:find("attempt to perform arithmetic on") and message:find("table value"), message)
            else
                local r = t + 1
                local expected = i < 100 and 1 or i < 200 and 2 or i < 300 and 3
                    or i < 350 and false or "call"
                if i >= 300 and i < 350 then
                    assert(r == false)
                else
                    assert(r == expected, i)
                end
            end
        end
    "#);
}

#[test]
fn replaced_handlers_may_be_collected_while_the_loop_runs() {
    run(r#"
        local mt = {}
        local t = setmetatable({}, mt)
        for i = 1, 300 do
            mt.__add = function() return i end
            mt.__lt = function() return i % 2 == 0 end
            mt.__concat = function() return "c" .. i end
            if i % 50 == 0 then collectgarbage() end
            assert(t + t == i)
            assert((t < t) == (i % 2 == 0))
            assert(t .. t == "c" .. i)
        end
    "#);
}

#[test]
fn a_different_metatable_changes_the_handler() {
    run(r#"
        local first = {__add = function() return "first" end, __lt = function() return true end}
        local second = {__add = function() return "second" end, __lt = function() return false end}
        local bare = {}
        local t = setmetatable({}, first)
        local u = setmetatable({}, first)
        for i = 1, 300 do
            local metatable = (i % 3 == 0) and first or (i % 3 == 1) and second or bare
            setmetatable(t, metatable)
            if metatable == bare then
                assert(not pcall(function() return t + 1 end))
                assert(t + u == "first")
                assert(not pcall(function() return t < t end))
            elseif metatable == first then
                assert(t + 1 == "first")
                assert((t < t) == true)
            else
                assert(t + 1 == "second")
                assert((t < t) == false)
                assert(t + u == "second")
            end
        end
        -- Instances sharing one metatable share its handlers.
        for i = 1, 300 do
            assert(u + 1 == "first")
        end
        first.__add = function() return "changed" end
        assert(u + 1 == "changed" and setmetatable({}, first) + 1 == "changed")
    "#);
}

#[test]
fn handlers_that_are_not_plain_closures_still_work() {
    run(r##"
        local varargs = setmetatable({}, {
            __add = function(...) return select("#", ...) end,
            __lt = function(...) return select("#", ...) == 2 end,
        })
        local native = setmetatable({}, {__add = rawequal, __eq = rawequal})
        local callable = setmetatable({}, {__add = setmetatable({}, {__call = function(self, a, b)
            return "called"
        end})})
        for i = 1, 300 do
            assert(varargs + 1 == 2)
            assert((varargs < varargs) == true)
            assert(native + native == true)
            assert(native + 1 == false)
            assert(callable + 1 == "called")
        end
    "##);
}

#[test]
fn handlers_returning_nothing_and_many_values() {
    run(r#"
        local nothing = setmetatable({}, {
            __add = function() end,
            __lt = function() end,
            __concat = function() end,
            __unm = function() end,
        })
        local many = setmetatable({}, {
            __add = function() return 1, 2, 3 end,
            __eq = function() return false, true end,
        })
        local other = setmetatable({}, getmetatable(many))
        for i = 1, 300 do
            assert(nothing + 1 == nil)
            assert((nothing < nothing) == false)
            assert(nothing .. "x" == nil)
            assert(-nothing == nil)
            local a, b = many + 1
            assert(a == 1 and b == nil)
            assert((many == other) == false)
            assert((many ~= other) == true)
        end
    "#);
}

#[test]
fn operator_handlers_may_yield() {
    run_in(
        &[
            CompatibilityProfile::Lua52,
            CompatibilityProfile::Lua53,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::Lua55,
        ],
        r#"
        local mt = {
            __add = function(a, b) return coroutine.yield("add") end,
            __lt = function(a, b) return coroutine.yield("lt") end,
            __eq = function(a, b) return coroutine.yield("eq") end,
            __concat = function(a, b) return coroutine.yield("concat") end,
            __len = function(a) return coroutine.yield("len") end,
        }
        local t, u = setmetatable({}, mt), setmetatable({}, mt)
        local co = coroutine.wrap(function()
            local results = {}
            for i = 1, 50 do
                results[#results + 1] = t + 1
                results[#results + 1] = t < u
                results[#results + 1] = t == u
                results[#results + 1] = t .. "s"
                results[#results + 1] = #t
                if t < u then results[#results + 1] = "then" else results[#results + 1] = "else" end
                if t == u then results[#results + 1] = "then" else results[#results + 1] = "else" end
            end
            return results
        end)
        local order = {"add", "lt", "eq", "concat", "len", "lt", "eq"}
        local answers = {10, 0, nil, "cat", 7, false, 1}
        local event = co()
        local step = 0
        while type(event) == "string" do
            step = step + 1
            assert(event == order[(step - 1) % 7 + 1], event)
            event = co(answers[(step - 1) % 7 + 1])
        end
        assert(type(event) == "table" and #event == 350, type(event))
        assert(event[1] == 10 and event[2] == true and event[3] == false)
        assert(event[4] == "cat" and event[5] == 7 and event[6] == "else" and event[7] == "then")
    "#,
    );
}

#[test]
fn yielding_across_an_operator_metamethod_is_an_error_in_lua_51() {
    run_in(
        &[CompatibilityProfile::Lua51],
        r#"
        local t = setmetatable({}, {__add = function() return coroutine.yield(1) end})
        local co = coroutine.create(function() return t + 1 end)
        local ok, message = coroutine.resume(co)
        assert(not ok and message:find("attempt to yield across"), message)
    "#,
    );
}

#[test]
fn errors_inside_handlers_unwind_through_the_operator() {
    run(r#"
        local t = setmetatable({}, {
            __add = function() error("add failed") end,
            __lt = function() error({code = 7}) end,
            __concat = function() error("concat failed", 0) end,
        })
        for i = 1, 100 do
            local ok, message = pcall(function() return t + 1 end)
            assert(not ok and message:find("add failed"), message)
            local ok, message = pcall(function() if t < t then return 1 end end)
            assert(not ok and type(message) == "table" and message.code == 7)
            local ok, message = pcall(function() return t .. "x" end)
            assert(not ok and message == "concat failed", message)
        end
        -- A deep chain of handlers overflows the stack without crashing.
        local depth = 0
        local deep
        deep = setmetatable({}, {__add = function(a, b) depth = depth + 1; return a + b end})
        local ok, message = pcall(function() return deep + 1 end)
        assert(not ok and message:find("stack overflow"), message)
        assert(depth > 50)
    "#);
}

#[test]
fn call_and_return_hooks_see_each_handler_call() {
    run(r#"
        local handler
        local mt = {__add = function(a, b) return 1 end, __lt = function() return true end}
        handler = mt.__add
        local lt = mt.__lt
        local t = setmetatable({}, mt)
        local calls, returns = 0, 0
        debug.sethook(function(event)
            local info = debug.getinfo(2, "f")
            if info.func == handler or info.func == lt then
                if event == "call" then calls = calls + 1 else returns = returns + 1 end
            end
        end, "cr")
        for i = 1, 100 do
            local x = t + t
            local y = t < t
        end
        debug.sethook()
        assert(calls == 200 and returns == 200, calls .. " " .. returns)
    "#);
}

#[test]
fn a_hook_set_by_a_handler_applies_to_the_rest_of_the_loop() {
    run(r#"
        local lines = 0
        local mt = {__lt = function()
            debug.sethook(function() lines = lines + 1 end, "l")
            return true
        end}
        local t = setmetatable({}, mt)
        local x
        if t < t then x = 1 end
        local y = 2
        local z = 3
        debug.sethook()
        assert(x == 1 and lines >= 2, lines)
    "#);
}

#[test]
fn tracebacks_name_the_event_of_a_handler_frame() {
    run_in(
        &[CompatibilityProfile::Lua54, CompatibilityProfile::Lua55],
        r#"
        local names = {}
        local mt = {}
        for _, event in ipairs {"add", "unm", "lt", "le", "eq", "concat", "len"} do
            mt["__" .. event] = function()
                names[#names + 1] = debug.traceback("x"):match("in metamethod '(%w+)'") or "none"
                return true
            end
        end
        local t, u = setmetatable({}, mt), setmetatable({}, mt)
        for i = 1, 100 do
            names = {}
            local _ = t + 1, -t, t < u, t <= u, t == u, t .. "x", #t
            if t < u then end
            if t == u then end
            assert(table.concat(names, " ") == "add unm lt le eq concat len lt eq",
                table.concat(names, " "))
        end
    "#,
    );
}

#[test]
fn string_arithmetic_through_the_string_metatable_still_delegates() {
    run_in(
        &[CompatibilityProfile::Lua54, CompatibilityProfile::Lua55],
        r#"
        local V = setmetatable({}, {__add = function(a, b)
            return (type(a) == "string" and "s" or "v") .. (type(b) == "string" and "s" or "v")
        end})
        for i = 1, 300 do
            assert("10" + 1 == 11)
            assert(V + "1" == "vs")
            assert("1" + V == "sv")
            assert(1 + V == "nv" or true)
        end
    "#,
    );
}

#[test]
fn cached_arithmetic_handlers_and_the_operands_that_miss_the_cache() {
    run(r##"
        local mt = {__add = function(a, b) return "add" end}
        local other = {__add = function(a, b) return "other" end}
        local A = setmetatable({}, mt)
        local B = setmetatable({}, mt)
        local C = setmetatable({}, other)
        local plain, bare = {}, setmetatable({}, {})
        local function check(expected)
            -- Both tables, a constant, a number register, a string, another
            -- metatable on either side, and a table that has no handler.
            local one = 1
            assert(A + B == expected)
            assert(A + 1 == expected)
            assert(A + one == expected)
            assert(1 + A == expected)
            assert(one + A == expected)
            assert(A + "x" == expected)
            assert("x" + A == expected)
            assert(A + C == expected)
            assert(C + A == "other")
            assert(plain + A == expected)
            assert(A + plain == expected)
            assert(bare + A == expected)
            assert(A + bare == expected)
        end
        for i = 1, 300 do check("add") end
        mt.__add = function(a, b) return "replaced" end
        for i = 1, 300 do check("replaced") end
        -- A handler that is no plain closure leaves the cached entry.
        mt.__add = setmetatable({}, {__call = function(self, a, b) return "callable" end})
        for i = 1, 300 do check("callable") end
        mt.__add = function(a, b) return "again" end
        for i = 1, 300 do check("again") end

        -- The metatable changes and then goes away.
        for i = 1, 300 do
            if i == 100 then setmetatable(A, other) end
            if i == 200 then setmetatable(A, nil) end
            local ok, result = pcall(function() return A + 1 end)
            if i < 100 then
                assert(ok and result == "again")
            elseif i < 200 then
                assert(ok and result == "other")
            else
                assert(not ok and result:find("attempt to perform arithmetic on", 1, true)
                    and result:find("table value", 1, true) and result:find("'A'", 1, true), result)
            end
        end
    "##);
}

#[test]
fn cached_comparison_handlers_and_the_pairs_that_miss_the_cache() {
    run(r##"
        local calls = 0
        local mt = {
            __lt = function(a, b) calls = calls + 1 return rawget(a, "v") < rawget(b, "v") end,
            __le = function(a, b) calls = calls + 1 return rawget(a, "v") <= rawget(b, "v") end,
            __eq = function(a, b) calls = calls + 1 return rawget(a, "v") == rawget(b, "v") end,
        }
        local other = {
            __lt = function() return true end,
            __le = function() return true end,
            __eq = function() return true end,
        }
        local A = setmetatable({v = 1}, mt)
        local B = setmetatable({v = 2}, mt)
        local A2 = setmetatable({v = 1}, mt)
        local C = setmetatable({v = 1}, other)
        local plain = {}
        for i = 1, 300 do
            assert(A < B and not (B < A) and not (A < A2))
            assert(A <= B and A <= A2 and not (B <= A))
            assert(A == A2 and A ~= B and not (A == B))
            -- One table compared with itself calls no handler.
            local before = calls
            assert(A == A and not (A ~= A))
            assert(calls == before)
            -- The same comparisons as branches.
            local taken = 0
            if A < B then taken = taken + 1 end
            if B < A then taken = taken + 2 end
            if A <= B then taken = taken + 4 end
            if A == A2 then taken = taken + 8 end
            if A ~= A2 then taken = taken + 16 end
            if not (A < B) then taken = taken + 32 end
            if A == B then taken = taken + 64 end
            if B <= A then taken = taken + 128 end
            assert(taken == 13, taken)
            -- A table without a metatable and a number are never equal to A.
            assert(A ~= plain and A ~= 1 and plain ~= A)

            -- Two metatables: the profile decides.
            local ok, result = pcall(function() return A < C end)
            if LUA51 then
                assert(not ok and result:find("attempt to compare two table values", 1, true), result)
            else
                assert(ok and result == false)
            end
            ok, result = pcall(function() return C <= A end)
            if LUA51 then
                assert(not ok and result:find("attempt to compare two table values", 1, true), result)
            else
                assert(ok and result == true)
            end
            assert((A == C) == LUA53 and (C == A) == LUA53)
        end

        -- A handler that is no plain closure, then a plain one again.
        mt.__lt = rawequal
        for i = 1, 300 do
            assert(not (A < B) and A < A)
        end
        mt.__lt = function(a, b) return rawget(a, "v") > rawget(b, "v") end
        for i = 1, 300 do
            assert(B < A and not (A < B))
        end
    "##);
}

#[test]
fn tables_without_handlers_raise_the_arithmetic_and_comparison_errors() {
    run(r##"
        local plain, bare = {}, setmetatable({}, {})
        local function fails(f, ...)
            local ok, message = pcall(f)
            assert(not ok, "no error")
            for i = 1, select("#", ...) do
                assert(message:find((select(i, ...)), 1, true), message)
            end
        end
        for i = 1, 300 do
            fails(function() return plain + 1 end, "attempt to perform arithmetic on", "table value", "'plain'")
            fails(function() return 1 + plain end, "attempt to perform arithmetic on", "table value", "'plain'")
            fails(function() return plain * bare end, "attempt to perform arithmetic on", "table value")
            fails(function() return bare % 2 end, "attempt to perform arithmetic on", "table value", "'bare'")
            fails(function() return 2 / bare end, "attempt to perform arithmetic on", "table value", "'bare'")
            fails(function() return plain < bare end, "attempt to compare two table values")
            fails(function() return bare <= bare end, "attempt to compare two table values")
            fails(function() return plain <= 1 end, "attempt to compare table with number")
            fails(function() return 1 < plain end, "attempt to compare number with table")
            fails(function() if plain < 1 then return 1 end end, "attempt to compare table with number")
            fails(function() if 1 <= bare then return 1 end end, "attempt to compare number with table")
            assert(plain ~= bare and plain == plain and not (plain == 1) and not (bare == "x"))
            assert("10" + 1 == 11 and "3" * "4" == 12)
            fails(function() return "a" + 1 end, "attempt to")
        end
    "##);
}

#[test]
fn runaway_handlers_overflow_the_stack_and_the_stack_recovers() {
    run(r##"
        local depth = 0
        local deep = setmetatable({}, {__add = function(a, b) depth = depth + 1 return a + b end})
        local ok, message = pcall(function() return deep + 1 end)
        assert(not ok and message:find("stack overflow", 1, true), message)
        assert(depth >= 4093 and depth <= 4094, depth)

        depth = 0
        local ordered = setmetatable({}, {})
        getmetatable(ordered).__lt = function(a, b) depth = depth + 1 return a < b end
        ok, message = pcall(function() return ordered < ordered end)
        assert(not ok and message:find("stack overflow", 1, true), message)
        assert(depth >= 4093 and depth <= 4094, depth)

        -- Plain calls and handlers work at every depth afterwards.
        local value = setmetatable({}, {
            __add = function(a, b) return 7 end,
            __eq = function(a, b) return true end,
        })
        local twin = setmetatable({}, getmetatable(value))
        local function down(n)
            if n == 0 then return value + 1 end
            local result = down(n - 1)
            return result
        end
        for n = 0, 60 do
            assert(down(n) == 7)
            assert(value == twin)
        end
        assert(down(3000) == 7)
    "##);
}

#[test]
fn handlers_follow_calls_of_every_other_kind_at_the_same_depth() {
    run(r##"
        local mt = {
            __add = function(a, b) return "add" end,
            __lt = function(a, b) return true end,
            __tostring = function(a) return "T" end,
            __concat = function(a, b) return "cat" end,
            __index = function(t, k) return k end,
            __call = function(self, a) return a end,
        }
        local A = setmetatable({}, mt)
        local B = setmetatable({}, mt)
        local function variadic(...) return select("#", ...) end
        local function two(a, b) return a, b end
        local function tail(a) return two(a, a) end
        local wrapped = coroutine.wrap(function()
            while true do coroutine.yield(A + B) end
        end)
        for i = 1, 300 do
            -- Each call below leaves a different activation record behind at
            -- the depth the next handler runs at.
            assert(variadic(1, 2, 3) == 3)
            assert(A + B == "add")
            assert(tostring(A) == "T")
            assert(A + 1 == "add")
            assert(A .. B == "cat")
            assert(A < B)
            assert(select("#", pcall(two, 1, 2)) == 3)
            assert(1 + A == "add")
            assert(A.field == "field")
            if A < B then else error("branch") end
            assert(A(i) == i)
            assert(A + B == "add")
            assert(select("#", tail(1)) == 2)
            assert(A + B == "add")
            assert(wrapped() == "add")
            assert(A < B)
            assert(("x"):rep(2) == "xx")
            assert(A + B == "add")
        end
    "##);
}

#[test]
fn a_handler_sees_its_caller_in_tracebacks_and_getinfo() {
    run(r##"
        local trace, line
        local mt = {
            __add = function(a, b)
                trace = debug.traceback("tb", 1)
                line = debug.getinfo(2, "l").currentline
                return 1
            end,
            __lt = function(a, b) error("ordered") end,
        }
        local A = setmetatable({}, mt)
        local B = setmetatable({}, mt)
        local function caller()
            local sum = A + B
            return sum
        end
        local first = debug.getinfo(caller, "S").linedefined + 1
        for i = 1, 300 do
            trace, line = nil, nil
            assert(caller() == 1)
            assert(line == first, line)
            local lines = {}
            for found in trace:gmatch(":(%d+):") do lines[#lines + 1] = tonumber(found) end
            assert(lines[2] == first, trace)
            local ok, message = pcall(function() return A < B end)
            assert(not ok and message:find("ordered", 1, true)
                and message:find(":" .. (first - 5) .. ":", 1, true), message)
        end
    "##);
}
