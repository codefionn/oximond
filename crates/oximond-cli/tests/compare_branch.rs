//! `TestEqual`, `TestLessThan`, `TestLessEqual` and their constant forms run a
//! comparison in a branch condition and take the `Jump` that follows.
//!
//! Each battery compiles a condition twice: written as a comparison inside
//! `if`, `while` or `repeat` (the compiler emits a test), and as
//! `local c = (comparison)` followed by the same statement on `c` (the
//! comparison opcode and `JumpIfFalse`). Both must give the same branch
//! decisions, errors, and metamethod calls with their arguments in order.

use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn run_in(profiles: &[CompatibilityProfile], source: &str) {
    for &profile in profiles {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            let script = format!(
                "PROFILE = '{profile}'\nMODE = '{mode}'\n\
                 if PROFILE == 'lua51' then load = loadstring end\n{source}"
            );
            if let Err(error) = lua.exec(&script) {
                panic!("{profile} {mode}: {error:?}");
            }
        }
    }
}

const HELPERS: &str = r##"
LOG = {}
local function tag(v)
    if type(v) == "table" then return "t" .. tostring(rawget(v, "id")) end
    return tostring(v)
end
local function logged(name, result)
    return function(a, b)
        LOG[#LOG + 1] = name .. ":" .. tag(a) .. "," .. tag(b)
        return result
    end
end
local function metatable(result)
    return { __eq = logged("eq", result), __lt = logged("lt", result), __le = logged("le", result) }
end
local MT_TRUE, MT_NIL, MT_ZERO = metatable(true), metatable(nil), metatable(0)
local function object(id, mt) return setmetatable({ id = id }, mt) end

-- Name and value of every operand the batteries try. Values are immutable,
-- so one set serves all runs.
local OPERANDS = {
    { "0", 0 }, { "1", 1 }, { "-1", -1 }, { "2.5", 2.5 }, { "1.0", 1.0 }, { "1e300", 1e300 },
    { "nan", 0 / 0 }, { "a", "a" }, { "1s", "1" }, { "true", true }, { "false", false },
    { "nil", nil }, { "table", { id = 0 } }, { "print", print },
    { "objT1", object(1, MT_TRUE) }, { "objT2", object(2, MT_TRUE) },
    { "objN3", object(3, MT_NIL) }, { "objN4", object(4, MT_NIL) },
    { "objZ5", object(5, MT_ZERO) }, { "objZ6", object(6, MT_ZERO) },
}
if PROFILE ~= "lua51" and PROFILE ~= "lua52" and PROFILE ~= "luajit21" then
    OPERANDS[#OPERANDS + 1] = { "maxint", math.maxinteger }
    OPERANDS[#OPERANDS + 1] = { "2^53", 9007199254740992 }
end
local N = #OPERANDS

local function describe(ok, ...)
    local n = select('#', ...)
    local parts = { tostring(ok) }
    for i = 1, n do
        local v = select(i, ...)
        if not ok then
            v = tostring(v):gsub("^[^:]*:%d+: ", "")
        end
        parts[#parts + 1] = tostring(v)
    end
    return table.concat(parts, "|")
end

-- Runs `source` with the operands and describes the outcome and the
-- metamethod calls it made.
local function run(source, x, y)
    LOG = {}
    local f = assert(load(source))
    local outcome = describe(pcall(f, x, y))
    return outcome .. "#" .. table.concat(LOG, ";")
end

local ITERATIONS = 120
-- `shapes` write the condition as `EXPR`; `value` replaces it by `c`.
local STATEMENTS = {
    ["if"] = function(expr, prelude)
        return prelude .. "if " .. expr .. " then s = s .. 'T' else s = s .. 'F' end"
    end,
    ["not"] = function(expr, prelude)
        return prelude .. "if not (" .. expr .. ") then s = s .. 'T' else s = s .. 'F' end"
    end,
    ["elseif"] = function(expr, prelude)
        return prelude .. "if g then s = s .. 'G' elseif " .. expr
            .. " then s = s .. 'T' else s = s .. 'F' end"
    end,
    ["while"] = function(expr, prelude)
        return prelude .. "while " .. expr .. " do s = s .. 'W' break end s = s .. 'e'"
    end,
    ["repeat"] = function(expr, prelude)
        return "local n = 0 repeat n = n + 1 if n > 1 then break end " .. prelude
            .. "until " .. expr .. " s = s .. 'r' .. n"
    end,
}

local function program(kind, expr, reference)
    local statement
    if reference then
        local prelude = "local c = (" .. expr .. ") "
        -- The reference tests the stored boolean instead of the comparison.
        if kind == "elseif" then
            statement = "if g then s = s .. 'G' elseif c then s = s .. 'T' else s = s .. 'F' end"
            -- the comparison is evaluated only when `g` is false
            statement = "if g then s = s .. 'G' else local c = (" .. expr .. ") "
                .. "if c then s = s .. 'T' else s = s .. 'F' end end"
        elseif kind == "while" then
            statement = "do local c = (" .. expr .. ") while c do s = s .. 'W' break end s = s .. 'e' end"
        elseif kind == "repeat" then
            statement = "local n = 0 repeat n = n + 1 if n > 1 then break end "
                .. "local c = (" .. expr .. ") until c s = s .. 'r' .. n"
        elseif kind == "not" then
            statement = prelude .. "if not c then s = s .. 'T' else s = s .. 'F' end"
        else
            statement = prelude .. "if c then s = s .. 'T' else s = s .. 'F' end"
        end
    else
        statement = STATEMENTS[kind](expr, "")
    end
    return "local x, y = ... local g = false local s = '' "
        .. "for i = 1, " .. ITERATIONS .. " do " .. statement .. " end return s"
end

local function check(kind, expr, x, y)
    local test = run(program(kind, expr, false), x, y)
    local reference = run(program(kind, expr, true), x, y)
    if test ~= reference then
        error(("%s `%s`: test form %s, reference %s"):format(kind, expr, test, reference))
    end
end
"##;

#[test]
fn constant_comparisons_give_the_value_form_decisions() {
    run_in(
        &PROFILES,
        &format!(
            r#"{HELPERS}
            local operators = {{ "==", "~=", "<", "<=", ">", ">=" }}
            local constants = {{ "0", "1", "-1", "2.5", "1.0", "9007199254740993", "'a'", "'1'",
                "nil", "true", "false" }}
            local kinds = {{ "if", "not", "elseif", "while", "repeat" }}
            local checked = 0
            local stride = (MODE == "tracing") and 7 or 2
            local count = 0
            for _, kind in ipairs(kinds) do
                for _, op in ipairs(operators) do
                    for _, k in ipairs(constants) do
                        for _, shape in ipairs({{ "x " .. op .. " " .. k, k .. " " .. op .. " x" }}) do
                            for i = 1, N do
                                count = count + 1
                                if count % stride == 0 then
                                    check(kind, shape, OPERANDS[i][2])
                                    checked = checked + 1
                                end
                            end
                        end
                    end
                end
            end
            assert(checked > 1000, checked)
            "#
        ),
    );
}

#[test]
fn register_comparisons_give_the_value_form_decisions() {
    run_in(
        &PROFILES,
        &format!(
            r#"{HELPERS}
            local operators = {{ "==", "~=", "<", "<=", ">", ">=" }}
            local kinds = {{ "if", "not", "while", "repeat" }}
            local stride = (MODE == "tracing") and 9 or 2
            local count, checked = 0, 0
            for _, kind in ipairs(kinds) do
                for _, op in ipairs(operators) do
                    for i = 1, N do
                        for j = 1, N do
                            count = count + 1
                            if count % stride == 0 then
                                check(kind, "x " .. op .. " y", OPERANDS[i][2], OPERANDS[j][2])
                                checked = checked + 1
                            end
                        end
                    end
                end
            end
            assert(checked > 1000, checked)
            "#
        ),
    );
}

#[test]
fn negations_and_parentheses_flip_the_branch() {
    run_in(
        &PROFILES,
        &format!(
            r#"{HELPERS}
            local shapes = {{
                "not (x < y)", "not not (x < y)", "((x <= y))", "not (x ~= y)", "not x == y",
                "not (x > 1)", "not (1 <= x)", "(x ~= 1)", "not (not (x ~= nil))",
            }}
            for _, shape in ipairs(shapes) do
                for _, kind in ipairs({{ "if", "while", "repeat", "elseif" }}) do
                    for i = 1, N, (MODE == "tracing") and 3 or 1 do
                        for j = 1, N, 5 do
                            check(kind, shape, OPERANDS[i][2], OPERANDS[j][2])
                        end
                    end
                end
            end
            "#
        ),
    );
}

#[test]
fn metamethods_run_in_the_register_form_order() {
    run_in(
        &PROFILES,
        &format!(
            r#"{HELPERS}
            local a, b = object(1, MT_TRUE), object(2, MT_TRUE)
            local function log_of(source)
                LOG = {{}}
                local f = assert(load(source))
                local r = f(a, b)
                return table.concat(LOG, ";") .. "=" .. tostring(r)
            end
            local function branch(expr)
                return "local a, b = ... if " .. expr .. " then return 'T' else return 'F' end"
            end
            assert(log_of(branch("a < b")) == "lt:t1,t2=T")
            assert(log_of(branch("a > b")) == "lt:t2,t1=T")
            assert(log_of(branch("a == b")) == "eq:t1,t2=T")
            assert(log_of(branch("a ~= b")) == "eq:t1,t2=F")
            assert(log_of(branch("not (a < b)")) == "lt:t1,t2=F")
            assert(log_of(branch("a <= b")) == "le:t1,t2=T")
            assert(log_of(branch("a >= b")) == "le:t2,t1=T")
            assert(log_of("local a, b = ... while a < b do return 'T' end return 'F'")
                == "lt:t1,t2=T")
            assert(log_of("local a, b = ... repeat until a > b return 'T'") == "lt:t2,t1=T")
            -- A result of nil or false takes the other branch; 0 is true.
            local n, m = object(3, MT_NIL), object(4, MT_NIL)
            local z, w = object(5, MT_ZERO), object(6, MT_ZERO)
            LOG = {{}}
            assert(assert(load("local a, b = ... if a < b then return 'T' end return 'F'"))(n, m) == "F")
            assert(assert(load("local a, b = ... if a < b then return 'T' end return 'F'"))(z, w) == "T")
            assert(assert(load("local a, b = ... if a == b then return 'T' end return 'F'"))(n, m) == "F")
            assert(assert(load("local a, b = ... if a ~= b then return 'T' end return 'F'"))(z, w) == "F")
            -- A constant operand is the metamethod's argument, on the side it
            -- was written. `__eq` runs only for two tables, so a constant
            -- never reaches it.
            if PROFILE ~= "lua51" and PROFILE ~= "luajit21" then
                assert(log_of(branch("a < 1")) == "lt:t1,1=T")
                assert(log_of(branch("a > 1")) == "lt:1,t1=T")
                assert(log_of(branch("1 < a")) == "lt:1,t1=T")
                assert(log_of(branch("1 > a")) == "lt:t1,1=T")
                assert(log_of(branch("a <= 2.5")) == "le:t1,2.5=T")
                assert(log_of(branch("a >= 2.5")) == "le:2.5,t1=T")
                assert(log_of(branch("2.5 <= a")) == "le:2.5,t1=T")
                assert(log_of(branch("2.5 >= a")) == "le:t1,2.5=T")
                assert(log_of(branch("not (a < -1)")) == "lt:t1,-1=F")
            else
                -- Lua 5.1 compares only operands of one type through `__lt`.
                for _, shape in ipairs({{ "x < 1", "1 < x", "x <= 1", "1 >= x" }}) do
                    check("if", shape, a)
                end
            end
            assert(log_of(branch("a == 1")) == "=F")
            assert(log_of(branch("a ~= 'x'")) == "=T")
            assert(log_of(branch("a == nil")) == "=F")
            "#
        ),
    );
}

#[test]
fn lua51_less_equal_falls_back_to_a_reversed_less_than() {
    run_in(
        &[
            CompatibilityProfile::Lua51,
            CompatibilityProfile::Lua52,
            CompatibilityProfile::Lua53,
            CompatibilityProfile::LuaJit21,
        ],
        &format!(
            r#"{HELPERS}
            local mt = {{ __lt = logged("lt", false) }}
            local a, b = object(1, mt), object(2, mt)
            local function decide(source)
                LOG = {{}}
                local r = assert(load(source))(a, b)
                return r .. "|" .. table.concat(LOG, ";")
            end
            -- `a <= b` is `not (b < a)`: __lt runs with swapped operands.
            assert(decide("local a, b = ... if a <= b then return 'T' end return 'F'") == "T|lt:t2,t1")
            assert(decide("local a, b = ... if a >= b then return 'T' end return 'F'") == "T|lt:t1,t2")
            assert(decide("local a, b = ... if not (a <= b) then return 'T' end return 'F'")
                == "F|lt:t2,t1")
            local mt_true = {{ __lt = logged("lt", true) }}
            local c, d = object(3, mt_true), object(4, mt_true)
            LOG = {{}}
            assert(assert(load("local a, b = ... if a <= b then return 'T' end return 'F'"))(c, d) == "F")
            assert(table.concat(LOG, ";") == "lt:t4,t3")
            -- The value form decides the same way.
            for _, kind in ipairs({{ "if", "not", "while", "repeat" }}) do
                check(kind, "x <= y", a, b)
                check(kind, "x >= y", a, b)
                check(kind, "x <= y", c, d)
            end
            "#
        ),
    );
}

#[test]
fn lua55_has_no_less_equal_fallback() {
    run_in(
        &[CompatibilityProfile::Lua55],
        &format!(
            r#"{HELPERS}
            local mt = {{ __lt = logged("lt", false) }}
            local a, b = object(1, mt), object(2, mt)
            local ok, message = pcall(load("local a, b = ... if a <= b then return 'T' end return 'F'"), a, b)
            assert(not ok and message:find("attempt to compare"), message)
            assert(#LOG == 0)
            "#
        ),
    );
}

#[test]
fn a_yield_inside_a_comparison_metamethod_decides_the_branch_when_resumed() {
    run_in(
        &[
            CompatibilityProfile::Lua52,
            CompatibilityProfile::Lua53,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::Lua55,
            CompatibilityProfile::LuaJit21,
        ],
        &format!(
            r#"{HELPERS}
            local function transcript(result, source)
                local mt = {{
                    __lt = function(a, b) coroutine.yield("lt " .. tag(a) .. tag(b)) return result end,
                    __le = function(a, b) coroutine.yield("le " .. tag(a) .. tag(b)) return result end,
                    __eq = function(a, b) coroutine.yield("eq " .. tag(a) .. tag(b)) return result end,
                }}
                local a, b = object(1, mt), object(2, mt)
                local co = coroutine.create(assert(load(source)))
                local out = {{}}
                while true do
                    local ok, value = coroutine.resume(co, a, b)
                    out[#out + 1] = tostring(ok) .. ":" .. tostring(value)
                    if coroutine.status(co) == "dead" then break end
                end
                return table.concat(out, ";")
            end
            local function loop(expr)
                return "local a, b = ... local s = '' for i = 1, 40 do if " .. expr
                    .. " then s = s .. 'T' else s = s .. 'F' end end return s"
            end
            local function loop_value(expr)
                return "local a, b = ... local s = '' for i = 1, 40 do local c = (" .. expr
                    .. ") if c then s = s .. 'T' else s = s .. 'F' end end return s"
            end
            local function while_loop(expr)
                return "local a, b = ... local s = '' for i = 1, 40 do while " .. expr
                    .. " do s = s .. 'T' break end s = s .. 'e' end return s"
            end
            local function while_value(expr)
                return "local a, b = ... local s = '' for i = 1, 40 do local c = (" .. expr
                    .. ") while c do s = s .. 'T' break end s = s .. 'e' end return s"
            end
            local function until_loop(expr)
                return "local a, b = ... local s = '' for i = 1, 40 do local n = 0 repeat n = n + 1 "
                    .. "if n > 1 then break end until " .. expr .. " s = s .. n end return s"
            end
            local function until_value(expr)
                return "local a, b = ... local s = '' for i = 1, 40 do local n = 0 repeat n = n + 1 "
                    .. "if n > 1 then break end local c = (" .. expr .. ") until c s = s .. n end return s"
            end
            local pairs_of_forms = {{
                {{ loop, loop_value }}, {{ while_loop, while_value }}, {{ until_loop, until_value }},
            }}
            for _, expr in ipairs({{ "a < b", "a > b", "a <= b", "a >= b", "a == b", "a ~= b",
                    "not (a < b)" }}) do
                for _, result in ipairs({{ true, false, 0, "x" }}) do
                    for _, forms in ipairs(pairs_of_forms) do
                        local test = transcript(result, forms[1](expr))
                        local reference = transcript(result, forms[2](expr))
                        assert(test == reference, expr .. " " .. tostring(result)
                            .. ": " .. test .. " vs " .. reference)
                    end
                end
            end
            -- The first resume yields from the first comparison.
            local mt = {{ __lt = function() coroutine.yield("first") return false end }}
            local a, b = object(1, mt), object(2, mt)
            local co = coroutine.wrap(function()
                if a < b then return "T" else return "F" end
            end)
            assert(co() == "first")
            assert(co() == "F")
            "#
        ),
    );
}

#[test]
fn a_yield_across_a_comparison_in_lua51_raises_the_value_form_error() {
    run_in(
        &[CompatibilityProfile::Lua51],
        &format!(
            r#"{HELPERS}
            local mt = {{ __lt = function() coroutine.yield("y") return true end }}
            local a, b = object(1, mt), object(2, mt)
            local function attempt(source)
                local co = coroutine.create(assert(load(source)))
                return describe(coroutine.resume(co, a, b))
            end
            local test = attempt("local a, b = ... if a < b then return 'T' end return 'F'")
            local reference = attempt("local a, b = ... local c = (a < b) if c then return 'T' end return 'F'")
            assert(test == reference, test .. " vs " .. reference)
            assert(test:find("yield across"), test)
            "#
        ),
    );
}

#[test]
fn integer_and_float_comparisons_are_exact() {
    run_in(
        &[
            CompatibilityProfile::Lua53,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::Lua55,
        ],
        &format!(
            r#"{HELPERS}
            local big = 9007199254740993
            local function branch(expr, x)
                return assert(load("local x = ... if " .. expr .. " then return 'T' end return 'F'"))(x)
            end
            -- 2^53 + 1 is not representable as a float.
            for _, shape in ipairs({{ "x == 9007199254740992.0", "x < 9007199254740992.0",
                    "x <= 9007199254740992.0", "x > 9007199254740992.0", "x >= 9007199254740992.0",
                    "x == 9007199254740993", "x < 9007199254740993", "x == 1", "x == 1.0",
                    "x < math.maxinteger + 0.0", "9007199254740992.0 < x", "9007199254740993 <= x" }}) do
                for _, v in ipairs({{ big, big - 1, 9007199254740992.0, 1, 1.0, math.maxinteger,
                        math.mininteger, -0.0, 0.0 }}) do
                    check("if", shape, v)
                    check("while", shape, v)
                end
            end
            assert(branch("x == 1", 1.0) == "T" and branch("x == 1.0", 1) == "T")
            assert(branch("x < 9007199254740993", 9007199254740992.0) == "T")
            assert(branch("x == 9007199254740993", 9007199254740992.0) == "F")
            assert(branch("x >= 9007199254740993", 9007199254740992.0) == "F")
            "#
        ),
    );
}

#[test]
fn luajit_64_bit_cdata_compares_like_the_value_form() {
    run_in(
        &[CompatibilityProfile::LuaJit21],
        &format!(
            r#"{HELPERS}
            local values = {{ 1LL, 2LL, -1LL, 1ULL, 2ULL, 18446744073709551615ULL, 1, 2, 1.0, 2.5,
                -1, "1", nil, true, {{}} }}
            local shapes = {{ "x == 1", "x ~= 1", "x < 2", "x <= 2", "x > 1", "x >= 1", "x == 1LL",
                "x < 2LL", "x <= 1ULL", "x == 1ULL", "x == y", "x < y", "x <= y", "x > y", "x >= y",
                "1LL == x", "2 > x", "x == 18446744073709551615ULL" }}
            local checked = 0
            for _, shape in ipairs(shapes) do
                for _, kind in ipairs({{ "if", "not", "while", "repeat" }}) do
                    for i = 1, 15 do
                        for j = 1, 15 do
                            check(kind, shape, values[i], values[j])
                            checked = checked + 1
                        end
                    end
                end
            end
            assert(checked > 1000)
            local function branch(expr, x)
                return assert(load("local x = ... if " .. expr .. " then return 'T' end return 'F'"))(x)
            end
            assert(branch("x == 1", 1LL) == "T")
            assert(branch("x == 1", 1ULL) == "T")
            assert(branch("x == 1", 2LL) == "F")
            assert(branch("x < 2", 1LL) == "T")
            assert(branch("x == 1LL", 1) == "T")
            assert(branch("x == nil", 1LL) == "F")
            assert(branch("x == 'a'", 1LL) == "F")
            "#
        ),
    );
}

#[test]
fn nil_boolean_and_string_equality_use_the_constant_test() {
    run_in(
        &PROFILES,
        &format!(
            r#"{HELPERS}
            local function branch(expr, x)
                return assert(load("local x = ... if " .. expr .. " then return 'T' end return 'F'"))(x)
            end
            assert(branch("x == nil", nil) == "T" and branch("x == nil", false) == "F")
            assert(branch("x ~= nil", false) == "T" and branch("x ~= nil", nil) == "F")
            assert(branch("x == true", true) == "T" and branch("x == true", 1) == "F")
            assert(branch("x == false", false) == "T" and branch("x == false", nil) == "F")
            assert(branch("x == 'abc'", 'ab' .. 'c') == "T" and branch("x == 'abc'", 'abd') == "F")
            assert(branch("x == 'abc'", 5) == "F" and branch("x ~= 'abc'", {{}}) == "T")
            assert(branch("x == 1", '1') == "F" and branch("x ~= 1", '1') == "T")
            assert(branch("'abc' == x", 'abc') == "T" and branch("nil == x", nil) == "T")
            assert(branch("x == 0", -0.0) == "T" and branch("x == 0", 0.0) == "T")
            local nan = 0 / 0
            assert(branch("x == x", nan) == "F" and branch("x ~= x", nan) == "T")
            assert(branch("x < 1", nan) == "F" and branch("x >= 1", nan) == "F")
            assert(branch("not (x < 1)", nan) == "T" and branch("not (x >= 1)", nan) == "T")
            assert(branch("1 > x", nan) == "F" and branch("not (1 > x)", nan) == "T")
            "#
        ),
    );
}

#[test]
fn ordering_errors_match_the_value_form() {
    run_in(
        &PROFILES,
        &format!(
            r#"{HELPERS}
            local bad = {{ nil, true, "a", {{}}, print, 1 }}
            for _, shape in ipairs({{ "x < 1", "x <= 1", "x > 1", "x >= 1", "1 < x", "1 <= x",
                    "x < y", "x <= y", "x > y", "x >= y", "x < 'a'" }}) do
                for _, kind in ipairs({{ "if", "while", "repeat" }}) do
                    for i = 1, 6 do
                        for j = 1, 6 do
                            check(kind, shape, bad[i], bad[j])
                        end
                    end
                end
            end
            local ok, message = pcall(load("local x = ... if x < 1 then return 1 end"), {{}})
            assert(not ok and message:find("attempt to compare"), message)
            "#
        ),
    );
}

#[test]
fn loops_exit_and_continue_by_the_test_branch() {
    run_in(
        &PROFILES,
        r"
        local function count_while(limit)
            local i = 0
            while i < limit do i = i + 1 end
            return i
        end
        local function count_until(limit)
            local i = 0
            repeat i = i + 1 until i >= limit
            return i
        end
        local function count_down(start)
            local i = start
            while i > 0 do i = i - 1 end
            return i
        end
        local function nested(n)
            local total = 0
            local i = 0
            while i < n do
                local j = 0
                repeat
                    j = j + 1
                    if j == 3 then total = total + 100 elseif j ~= 5 then total = total + 1 end
                until j >= 6
                i = i + 1
            end
            return total
        end
        for round = 1, 200 do
            assert(count_while(50) == 50 and count_while(0) == 0 and count_while(-3) == 0)
            assert(count_until(50) == 50 and count_until(0) == 1)
            assert(count_down(40) == 0 and count_down(-1) == -1)
            assert(nested(3) == 3 * (100 + 4))
        end
        -- Closures capture a fresh local per repeat iteration and see it in
        -- the condition.
        local fns = {}
        local k = 0
        repeat
            k = k + 1
            local captured = k
            fns[k] = function() captured = captured + 1 return captured end
        until captured >= 3
        assert(k == 3 and fns[1]() == 2 and fns[3]() == 4)
        local m = 0
        repeat
            m = m + 1
            local v = m
            local f = function() return v end
        until f() >= 4
        assert(m == 4)
        -- break and goto-free exits from inside tested loops.
        local seen = 0
        for i = 1, 100 do
            local j = i
            while j % 7 ~= 0 do j = j + 1 if j > 50 then break end end
            seen = seen + 1
        end
        assert(seen == 100)
        ",
    );
}

#[test]
fn large_loop_bodies_keep_their_jumps() {
    // The jump after a test is a 24-bit jump, so a body longer than a
    // JumpIfFalse could reach still compiles and runs.
    let mut body = String::new();
    for _ in 0..40_000 {
        body.push_str("x = x + 1\n");
    }
    let source = format!("local x = 0\nlocal n = ...\nif n < 1 then\n{body}\nend\nreturn x");
    run_in(
        &[CompatibilityProfile::Lua54, CompatibilityProfile::Lua51],
        &format!(
            "local f = assert(load([==[{source}]==]))\nassert(f(5) == 0)\nassert(f(0) == 40000)"
        ),
    );
}

#[test]
fn line_hooks_see_the_same_lines_as_before_the_tests() {
    // The sequences are what the interpreter reported before the compiler
    // emitted tests, for these programs: a comparison and its branch share a
    // line, and a condition that spans lines keeps the old code.
    run_in(
        &[
            CompatibilityProfile::Lua51,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::LuaJit21,
        ],
        r#"
        local function lines_of(source, ...)
            local f = assert(load(source))
            local seen = {}
            debug.sethook(function(_, line)
                if debug.getinfo(2, "f").func == f then seen[#seen + 1] = line end
            end, "l")
            local r = f(...)
            debug.sethook()
            return table.concat(seen, ",") .. "=" .. tostring(r)
        end
        local progs = {
[[
local x = ...
local n = 0
if x < 2 then
  n = 1
else
  n = 2
end
while n < 4 do
  n = n + 1
end
repeat
  n = n - 1
until n <= 2
return n]],
[[
local x = ...
local t = 0
for i = 1, 3 do
  if x < i then t = t + 1 end
  if x == i then t = t + 10 elseif x ~= 2 then t = t + 100 end
end
return t]],
[[
local x = ...
local t = 0
while x < 3 and t < 5 do x = x + 1 t = t + 1 end
repeat t = t - 1 until t < 2 or x > 100
return t]],
[[
local x = ...
local t = 0
if x < 2 then t = 1 end
while t < 3 do t = t + 1 end
repeat t = t - 1 until t <= 0
if
 x
 < 2 then t = 7 end
return t]],
        }
        local jit = PROFILE == "luajit21"
        local expected = {
            { "1,2,3,4,8,9,8,9,8,9,8,12,13,12,13,14=2", "1,2,3,6,8,9,8,9,8,12,13,12,13,14=2" },
            { "1,2,3,4,5,3,4,5,3,4,5,3,7=212", "1,2,3,4,5,3,4,5,3,4,5,3,7=210" },
            { "1,2,3,3,3,4,5=1", "1,2,3,4,5=-1" },
            { "1,2,3,4,4,4,5,5,5,7,8,7,8,9=7", "1,2,3,4,4,4,4,5,5,5,7,8,7,9=0" },
        }
        if jit then
            expected[1] = { "1,2,3,4,8,9,8,9,8,9,8,11,12,13,11,12,13,14=2",
                "1,2,3,6,8,9,8,9,8,11,12,13,11,12,13,14=2" }
        end
        for index, program in ipairs(progs) do
            for which, x in ipairs({ 1, 3 }) do
                local got = lines_of(program, x)
                assert(got == expected[index][which], index .. " x=" .. x .. ": " .. got)
            end
        end
        "#,
    );
}

#[test]
fn a_metamethod_sees_the_comparison_line_in_its_caller() {
    run_in(
        &PROFILES,
        r#"
        local mt = { __lt = function(a, b)
            local info = debug.getinfo(2, "l")
            LINE = info.currentline
            return true
        end }
        local a, b = setmetatable({}, mt), setmetatable({}, mt)
        local f = assert(load("local a, b = ...\nlocal r = 0\nif\na < b then\nr = 1\nend\nreturn r"))
        LINE = nil
        local r = f(a, b)
        local g = assert(load("local a, b = ...\nlocal r = 0\nlocal c = a < b\nif c then\nr = 1\nend\nreturn r"))
        local test_line = LINE
        LINE = nil
        g(a, b)
        assert(r == 1 and test_line ~= nil and LINE == 3)
        assert(test_line == 4)
        "#,
    );
}

#[test]
fn tracebacks_and_error_positions_name_the_metamethod_as_before() {
    run_in(
        &[
            CompatibilityProfile::Lua51,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::LuaJit21,
        ],
        r#"
        local function trace_of(source, a, b)
            local f = assert(load(source, "=chunk"))
            TB = nil
            f(a, b)
            return (tostring(TB):gsub("%(load%):%d+", "(load):N"))
        end
        local mt = {
            __lt = function() TB = debug.traceback("tb", 1) return true end,
            __le = function() TB = debug.traceback("tb", 1) return true end,
            __eq = function() TB = debug.traceback("tb", 1) return true end,
        }
        local a, b = setmetatable({}, mt), setmetatable({}, mt)
        -- The test and the value form sit on the same lines, so their
        -- tracebacks from inside the metamethod must read the same.
        for _, expr in ipairs({ "a < b", "a > b", "a <= b", "a >= b", "a == b", "a ~= b" }) do
            local test = trace_of("local a, b = ...\nif " .. expr
                .. " then return 1 end\nreturn 'no'", a, b)
            local value = trace_of("local a, b = ...\nlocal c = " .. expr
                .. " if c then return 1 end\nreturn 'no'", a, b)
            assert(test == value, expr .. ":\n" .. test .. "\nvs\n" .. value)
            assert(test:find("chunk:2:", 1, true), test)
        end
        -- An error raised by a comparison carries the line of the comparison.
        local function error_of(source, a, b)
            local f = assert(load(source, "=chunk"))
            return select(2, pcall(f, a, b))
        end
        for _, expr in ipairs({ "a < 1", "1 < a", "a <= b", "a > nil" }) do
            local plain = { }
            local test = error_of("local a, b = ...\nif " .. expr .. " then return 1 end", plain, plain)
            local value = error_of("local a, b = ...\nlocal c = " .. expr .. " if c then return 1 end", plain, plain)
            assert(test == value and tostring(test):find("chunk:2:", 1, true), tostring(test))
        end
        "#,
    );
}

#[test]
fn hot_loops_in_both_modes_keep_the_value_form_counts() {
    run_in(
        &PROFILES,
        r"
        local function tally(n)
            local lt, le, eq, ne, gt, ge = 0, 0, 0, 0, 0, 0
            for i = 1, n do
                local m = i % 7
                if m < 3 then lt = lt + 1 end
                if m <= 3 then le = le + 1 end
                if m == 3 then eq = eq + 1 end
                if m ~= 3 then ne = ne + 1 end
                if m > 3 then gt = gt + 1 end
                if m >= 3 then ge = ge + 1 end
                if 2 < m then lt = lt + 1000 end
                if not (m < 5) then le = le + 1000 end
            end
            return lt, le, eq, ne, gt, ge
        end
        local lt, le, eq, ne, gt, ge = tally(7000)
        assert(lt == 3000 + 1000 * 4000, lt)
        assert(le == 4000 + 1000 * 2000, le)
        assert(eq == 1000 and ne == 6000 and gt == 3000 and ge == 4000)
        ",
    );
}

#[test]
fn a_repeat_body_with_a_to_be_closed_local_compares_before_closing() {
    run_in(
        &[CompatibilityProfile::Lua54, CompatibilityProfile::Lua55],
        r#"
        local log = {}
        local function closer(name)
            return setmetatable({}, { __close = function(_, err)
                log[#log + 1] = name .. ":" .. tostring(err ~= nil and type(err) or "nil")
            end })
        end
        local mt = {
            __lt = function() log[#log + 1] = "lt"; return true end,
            __le = function() log[#log + 1] = "le"; return true end,
            __eq = function() log[#log + 1] = "eq"; return true end,
        }
        local t, u = setmetatable({}, mt), setmetatable({}, mt)
        local f = load([[
            local t, u, closer = ...
            repeat local c <close> = closer("close") until t < u
            repeat local c <close> = closer("close") until t <= u
            repeat local c <close> = closer("close") until t == u
        ]])
        f(t, u, closer)
        assert(table.concat(log, " ") == "lt close:nil le close:nil eq close:nil", table.concat(log, " "))
        log = {}
        local ok, message = pcall(load([[
            local closer, x = ...
            repeat local c <close> = closer("close") until x < 1
        ]]), closer, {})
        assert(not ok and tostring(message):find("attempt to compare"), tostring(message))
        assert(log[1] == "close:string", log[1])
        "#,
    );
}
