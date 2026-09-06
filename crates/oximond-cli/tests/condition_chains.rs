//! `and`, `or` and `not` trees in `if`, `elseif`, `while` and `repeat`
//! conditions compile to jump chains. Each battery compiles a condition
//! twice: written in the statement (the compiler emits a jump chain), and as
//! `local c = (expr)` followed by the same statement on `c` (a boolean and
//! one `JumpIfFalse`). Both must make the same decisions, call the same
//! functions and metamethods in the same order, and raise the same errors.

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

/// Builds and compares the two forms of a condition.
///
/// The condition's operands are calls that log their name and return the
/// value the run assigns, so the log shows evaluation order and
/// short-circuiting:
///
/// - `A` is `a() < b()`, `B` is `c() == 1`, `C` is `d()`
/// - `D` is `e().x <= 2`, `E` is `2 < f()`
const HELPERS: &str = r##"
LOG = {}
local function tag(v)
    if type(v) == "table" then return "t" .. tostring(rawget(v, "id")) end
    return tostring(v)
end
local function log(entry) LOG[#LOG + 1] = entry end
local function logged(name, result)
    return function(a, b)
        log(name .. ":" .. tag(a) .. "," .. tag(b))
        return result
    end
end
local function object(id, lt, le, eq)
    return setmetatable({ id = id }, {
        __lt = logged("lt", lt), __le = logged("le", le), __eq = logged("eq", eq),
    })
end
local function proxy(id, value)
    return setmetatable({ id = id }, { __index = function(_, key)
        log("index:" .. id .. "." .. key)
        return value
    end })
end
local NIL = {}
local nan = 0 / 0

local POOLS = {
    plain = {
        a = { 1, 2.5, nan, 0, -1, 2 }, b = { 1, 2.5, nan, 3, -1, 0 },
        c = { 1, 2, 1.0, NIL, false, "1", nan },
        d = { NIL, false, 0, "s", true, {} },
        e = { { x = 1 }, { x = 5 }, { x = nan }, { x = 2.0 } },
        f = { 1, 3, nan, 2.5 },
    },
    meta = {
        a = { object(1, true, true, true), object(2, nil, nil, nil), object(3, 0, 0, 0), 1 },
        b = { object(4, true, false, true), object(5, false, true, false), 2, object(6, 0, nil, 0) },
        c = { object(7, true, true, true), 1, 2, proxy(8, 1) },
        d = { object(9, true, true, true), true, NIL, false },
        e = { proxy(10, 1), proxy(11, 5), proxy(12, NIL), { x = 1 } },
        f = { object(13, true, false, true), 3, object(14, nil, nil, nil), 1 },
    },
    errors = {
        a = { 1, "x", NIL, false, {}, print }, b = { 2, "y", NIL, true, {}, 1 },
        c = { 1, NIL, "1", false },
        d = { NIL, false, 1, {} },
        e = { NIL, 7, "str", { x = 1 }, { x = "z" }, { x = false } },
        f = { NIL, "x", {}, 3, 1.5 },
    },
}

local LEAVES = { "a() < b()", "c() == 1", "d()", "e().x <= 2", "2 < f()" }

local function trees(l1_limit, l2_limit)
    local l0 = {}
    for _, leaf in ipairs(LEAVES) do
        l0[#l0 + 1] = leaf
        l0[#l0 + 1] = "not (" .. leaf .. ")"
    end
    local l1 = {}
    for _, left in ipairs(l0) do
        for _, right in ipairs(l0) do
            l1[#l1 + 1] = left .. " and " .. right
            l1[#l1 + 1] = left .. " or " .. right
        end
    end
    local seed = 12345
    local function pick(list)
        seed = (seed * 1103515245 + 12345) % 2147483648
        return list[seed % #list + 1]
    end
    local out = {}
    for _, leaf in ipairs(l0) do out[#out + 1] = leaf end
    local used = 0
    local stride = math.max(1, math.floor(#l1 / l1_limit))
    for index = 1, #l1, stride do out[#out + 1] = l1[index] end
    for _ = 1, l2_limit do
        local shape = seed % 5
        local inner = "(" .. pick(l1) .. ")"
        local leaf = pick(l0)
        if shape == 0 then out[#out + 1] = inner .. " and " .. leaf
        elseif shape == 1 then out[#out + 1] = leaf .. " or " .. inner
        elseif shape == 2 then out[#out + 1] = "not " .. inner
        elseif shape == 3 then out[#out + 1] = inner .. " or (" .. pick(l1) .. ")"
        else out[#out + 1] = leaf .. " and not " .. inner end
        seed = seed + 7
    end
    return out
end

local STATEMENTS = {
    ["if"] = function(expr)
        return "if " .. expr .. " then s = s .. 'T' else s = s .. 'F' end"
    end,
    ["not"] = function(expr)
        return "if not (" .. expr .. ") then s = s .. 'T' else s = s .. 'F' end"
    end,
    ["elseif"] = function(expr)
        return "if g then s = s .. 'G' elseif " .. expr .. " then s = s .. 'T' else s = s .. 'F' end"
    end,
    ["while"] = function(expr)
        return "while " .. expr .. " do s = s .. 'W' break end s = s .. 'e'"
    end,
    ["repeat"] = function(expr)
        return "local n = 0 repeat n = n + 1 if n > 1 then break end until " .. expr
            .. " s = s .. 'r' .. n"
    end,
}
local REFERENCES = {
    ["if"] = function(expr)
        return "local c = (" .. expr .. ") if c then s = s .. 'T' else s = s .. 'F' end"
    end,
    ["not"] = function(expr)
        return "local c = (" .. expr .. ") if not c then s = s .. 'T' else s = s .. 'F' end"
    end,
    ["elseif"] = function(expr)
        return "if g then s = s .. 'G' else local c = (" .. expr
            .. ") if c then s = s .. 'T' else s = s .. 'F' end end"
    end,
    ["while"] = function(expr)
        return "do local c = (" .. expr .. ") while c do s = s .. 'W' break end s = s .. 'e' end"
    end,
    ["repeat"] = function(expr)
        return "local n = 0 repeat n = n + 1 if n > 1 then break end local c = (" .. expr
            .. ") until c s = s .. 'r' .. n"
    end,
}

local function describe(ok, ...)
    local parts = { tostring(ok) }
    for i = 1, select('#', ...) do parts[#parts + 1] = tostring((select(i, ...))) end
    return table.concat(parts, "|")
end

local function program(forms, kind, expr, iterations)
    return "local V, log = ... local g = false local s = '' "
        .. "local function a() log('a') return V.a end "
        .. "local function b() log('b') return V.b end "
        .. "local function c() log('c') return V.c end "
        .. "local function d() log('d') return V.d end "
        .. "local function e() log('e') return V.e end "
        .. "local function f() log('f') return V.f end "
        .. "for i = 1, " .. iterations .. " do " .. forms[kind](expr) .. " end return s"
end

local function run(source, values)
    LOG = {}
    local chunk = assert(load(source, "=cond"))
    local outcome = describe(pcall(chunk, values, log))
    return outcome .. "#" .. table.concat(LOG, ";")
end

local function check(kind, expr, values, iterations)
    local test = run(program(STATEMENTS, kind, expr, iterations), values)
    local reference = run(program(REFERENCES, kind, expr, iterations), values)
    if test ~= reference then
        error(("%s `%s`: chain %s, reference %s"):format(kind, expr, test, reference))
    end
    return test
end

-- Runs every tree with several assignments drawn from the pool.
local function battery(pool_name, l1_limit, l2_limit, assignments, iterations)
    local pool = POOLS[pool_name]
    local seed = 99
    local function value(list)
        seed = (seed * 1103515245 + 12345) % 2147483648
        local v = list[math.floor(seed / 7) % #list + 1]
        if v == NIL then return nil end
        return v
    end
    local checked, errors = 0, 0
    for _, expr in ipairs(trees(l1_limit, l2_limit)) do
        for _, kind in ipairs({ "if", "not", "elseif", "while", "repeat" }) do
            for _ = 1, assignments do
                local values = {}
                for _, key in ipairs({ "a", "b", "c", "d", "e", "f" }) do
                    values[key] = value(pool[key])
                end
                local outcome = check(kind, expr, values, iterations)
                if outcome:find("^false") then errors = errors + 1 end
                checked = checked + 1
            end
        end
    end
    return checked, errors
end
"##;

fn script(body: &str) -> String {
    format!("{HELPERS}\n{body}")
}

#[test]
fn operand_order_and_short_circuiting_match_the_value_form() {
    run_in(
        &PROFILES,
        &script(
            r#"
            local checked = battery("plain", 120, 80, 3, 2)
            assert(checked > 3000, checked)
            "#,
        ),
    );
}

#[test]
fn comparison_and_index_metamethods_run_in_the_value_form_order() {
    run_in(
        &PROFILES,
        &script(
            r#"
            local checked = battery("meta", 120, 80, 3, 2)
            assert(checked > 3000, checked)
            "#,
        ),
    );
}

#[test]
fn errors_and_their_messages_match_the_value_form() {
    run_in(
        &PROFILES,
        &script(
            r#"
            local checked, errors = battery("errors", 120, 80, 3, 2)
            assert(checked > 3000, checked)
            -- A good share of the runs raise: the batteries must compare messages.
            assert(errors > 300, errors)
            "#,
        ),
    );
}

#[test]
fn hot_loops_decide_like_the_value_form_in_both_modes() {
    run_in(
        &PROFILES,
        &script(
            r#"
            local checked = battery("plain", 30, 20, 2, 150)
            assert(checked > 500, checked)
            checked = battery("meta", 20, 10, 1, 150)
            assert(checked > 150, checked)
            "#,
        ),
    );
}

#[test]
fn nil_false_nan_and_cdata_operands_match_the_value_form() {
    run_in(
        &PROFILES,
        &script(
            r#"
            local values = { 1, 0, nil, false, true, nan, "a", 2.5, {} }
            if PROFILE == "luajit21" then
                for _, literal in ipairs({ "1LL", "2ULL", "18446744073709551615ULL" }) do
                    values[#values + 1] = assert(load("return " .. literal))()
                end
            end
            local shapes = {
                "x == 1 and y", "x < y or y", "not (x == y and y)", "x and y < 2", "x or y == nil",
                "(x < 1 or y > 1) and x ~= y", "not (x ~= nil or y <= 1)", "x == nil or y == false",
                "not (x < y) and x", "x and not (y >= 1)",
            }
            local function run_pair(expr, x, y)
                local function go(source)
                    local ok, result = pcall(assert(load(source, "=cond")), x, y)
                    return tostring(ok) .. "|" .. tostring(result)
                end
                local test = go("local x, y = ... local r = 'F' if " .. expr .. " then r = 'T' end return r")
                local reference = go("local x, y = ... local r = 'F' local c = (" .. expr
                    .. ") if c then r = 'T' end return r")
                assert(test == reference, expr .. ": " .. test .. " vs " .. reference)
                local loop = go("local x, y = ... local r = 0 local n = 0 while " .. expr
                    .. " do n = n + 1 if n > 2 then break end end return n")
                local loop_ref = go("local x, y = ... local r = 0 local n = 0 while true do local c = (" .. expr
                    .. ") if not c then break end n = n + 1 if n > 2 then break end end return n")
                assert(loop == loop_ref, expr .. ": " .. loop .. " vs " .. loop_ref)
            end
            for _, expr in ipairs(shapes) do
                for i = 1, #values do
                    for j = 1, #values do
                        run_pair(expr, values[i], values[j])
                    end
                end
            end
            "#,
        ),
    );
}

#[test]
fn a_yield_inside_a_comparison_metamethod_resumes_into_the_chain() {
    run_in(
        &PROFILES,
        &script(
            r#"
            local function transcript(result, source)
                local mt = {
                    __lt = function(a, b) coroutine.yield("lt " .. tag(a) .. tag(b)) return result end,
                    __le = function(a, b) coroutine.yield("le " .. tag(a) .. tag(b)) return result end,
                    __eq = function(a, b) coroutine.yield("eq " .. tag(a) .. tag(b)) return result end,
                }
                local a, b = setmetatable({ id = 1 }, mt), setmetatable({ id = 2 }, mt)
                local co = coroutine.create(assert(load(source, "=cond")))
                local out = {}
                for _ = 1, 40 do
                    local ok, value = coroutine.resume(co, a, b)
                    out[#out + 1] = tostring(ok) .. ":" .. tostring(value)
                    if coroutine.status(co) == "dead" then break end
                end
                return table.concat(out, ";")
            end
            local function chain(expr)
                return "local a, b = ... local s = '' for i = 1, 6 do if " .. expr
                    .. " then s = s .. 'T' else s = s .. 'F' end end return s"
            end
            local function value(expr)
                return "local a, b = ... local s = '' for i = 1, 6 do local c = (" .. expr
                    .. ") if c then s = s .. 'T' else s = s .. 'F' end end return s"
            end
            local function until_chain(expr)
                return "local a, b = ... local s = '' for i = 1, 6 do local n = 0 repeat n = n + 1 "
                    .. "if n > 1 then break end until " .. expr .. " s = s .. n end return s"
            end
            local function until_value(expr)
                return "local a, b = ... local s = '' for i = 1, 6 do local n = 0 repeat n = n + 1 "
                    .. "if n > 1 then break end local c = (" .. expr .. ") until c s = s .. n end return s"
            end
            local function while_chain(expr)
                return "local a, b = ... local s = '' for i = 1, 6 do while " .. expr
                    .. " do s = s .. 'T' break end s = s .. 'e' end return s"
            end
            local function while_value(expr)
                return "local a, b = ... local s = '' for i = 1, 6 do local c = (" .. expr
                    .. ") while c do s = s .. 'T' break end s = s .. 'e' end return s"
            end
            local exprs = {
                "a < b and b < a", "a < b or b <= a", "not (a < b and a <= b)", "a == b or a < b",
                "a < b and a", "a <= b or b", "not (a < b or b == a)", "(a < b or a > b) and a ~= b",
                "b and a < b",
            }
            for _, expr in ipairs(exprs) do
                for _, result in ipairs({ true, false, 0, "x" }) do
                    for _, forms in ipairs({
                        { chain, value }, { until_chain, until_value }, { while_chain, while_value },
                    }) do
                        local test = transcript(result, forms[1](expr))
                        local reference = transcript(result, forms[2](expr))
                        assert(test == reference, expr .. " " .. tostring(result)
                            .. ": " .. test .. " vs " .. reference)
                    end
                end
            end
            -- The first resume yields from the first comparison of the chain, and
            -- the second decides the branch.
            if PROFILE ~= "lua51" then
                local mt = { __lt = function() coroutine.yield("first") return false end }
                local a, b = setmetatable({}, mt), setmetatable({}, mt)
                local co = coroutine.wrap(function()
                    if a < b or b < a then return "T" else return "F" end
                end)
                assert(co() == "first")
                assert(co() == "first")
                assert(co() == "F")
            end
            "#,
        ),
    );
}

#[test]
fn line_hooks_report_the_lines_they_reported_before_chains() {
    // The sequences are what the interpreter reported before the compiler
    // emitted jump chains, for these programs: every operand of a
    // single-line condition shares the condition's line, and a condition
    // that spans lines keeps the old code.
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
        local x, y = ...
        local t = 0
        if x < 2 and y < 5 then
          t = 1
        end
        if x < 2 or y < 5 then
          t = t + 2
        end
        if not (x < 2 and y < 5) then
          t = t + 4
        end
        return t]],
        [[
        local x, y = ...
        local t = 0
        while x < 4 and y > 0 do
          x = x + 1
          y = y - 1
          t = t + 1
        end
        repeat
          t = t + 1
        until t > 5 or x == 0 or y ~= 0
        return t]],
        [[
        local x, y = ...
        local t = 0
        if x and y < 3 then t = 1
        elseif y or x > 7 then t = 2
        elseif not x or y == nil then t = 3
        else t = 4 end
        for i = 1, 3 do
          if i < x and (y == 1 or i == 2) then t = t + 10 end
        end
        return t]],
        [[
        local x, y = ...
        local t = 0
        if x < 2 and
           y < 5 then
          t = 1
        end
        if x < 2 or
           y < 5 then
          t = t + 2
        end
        return t]],
        }
        local expected = {
            { "1,2,3,4,6,7,9,12=3", "1,2,3,6,7,9,10,12=6", "1,2,3,6,7,9,10,12=6" },
            { "1,2,3,4,5,6,3,9,10,9,10,9,10,9,10,9,10,11=6", "1,2,3,4,5,6,3,9,10,11=2", "1,2,3,9,10,9,10,9,10,9,10,9,10,9,10,11=6" },
            { "1,2,3,7,8,7,8,7,8,7,10=1", "1,2,3,4,7,8,7,8,7,8,7,10=12", "1,2,3,7,8,7,8,7,8,7,10=11" },
            { "1,2,3,4,3,5,7,9,11=3", "1,2,3,7,8,7,9,11=2", "1,2,3,7,8,7,9,11=2" },
        }
        if PROFILE == 'luajit21' then
            expected[2] = { "1,2,3,4,5,6,3,8,9,10,8,9,10,8,9,10,8,9,10,8,9,10,11=6", "1,2,3,4,5,6,3,8,9,10,11=2", "1,2,3,8,9,10,8,9,10,8,9,10,8,9,10,8,9,10,8,9,10,11=6" }
        end
        for index, program in ipairs(progs) do
            for which, args in ipairs({ { 1, 1 }, { 3, 4 }, { 8, 0 } }) do
                local got = lines_of(program, args[1], args[2])
                assert(got == expected[index][which], index .. " " .. which .. ": " .. got)
            end
        end
        "#,
    );
}
