//! `AddK`, `SubtractK`, `MultiplyK`, `DivideK` and `ModuloK` behave as a
//! `LoadConstant` into a register followed by the register opcode.
//!
//! Each battery compiles `x op K` with a literal (the constant opcode) and
//! `local k = K; x op k` (the register opcode) and compares what both give
//! for the same operands: values, integer or float subtype, error messages
//! with the operand names, metamethod arguments and their order.

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
                "PROFILE = '{profile}'\nif PROFILE == 'lua51' then load = loadstring end\n{source}"
            );
            if let Err(error) = lua.exec(&script) {
                panic!("{profile} {mode}: {error:?}");
            }
        }
    }
}

const HELPERS: &str = r#"
local function describe(ok, ...)
    local n = select('#', ...)
    local parts = { tostring(ok) }
    for i = 1, n do
        local v = select(i, ...)
        if not ok then
            v = tostring(v):gsub("^[^:]*:%d+: ", "")
        end
        local kind = (math and math.type and math.type(v)) or type(v)
        parts[#parts + 1] = kind .. ":" .. tostring(v)
    end
    return table.concat(parts, "|")
end

-- Compiles `body` with the operand `x` and the constant text `k`, runs it in
-- a loop hot enough for the fast loop and the tracer, and describes the last
-- result or the error.
local function eval(template, k, x)
    local source = template:gsub("KONST", k)
    local f = assert(load(source))
    return describe(pcall(f, x))
end
"#;

#[test]
fn constants_give_the_register_form_results() {
    run_in(
        &PROFILES,
        &format!(
            r#"{HELPERS}
            local operators = {{ "+", "-", "*", "/", "%" }}
            local constants = {{ "3", "2.5", "0", "0.0", "1e300", "255", "9007199254740993",
                "0.5", "7", "100000000000" }}
            local operands = {{ 7, -7, 7.5, 0, 0.0, -0.0, 1e308, 2^53, "10", "0x10", " 5 ", "abc",
                false, true, {{}}, print }}
            if PROFILE ~= "lua51" and PROFILE ~= "lua52" and PROFILE ~= "luajit21" then
                operands[#operands + 1] = math.maxinteger
                operands[#operands + 1] = math.mininteger
            end
            local checked = 0
            for _, op in ipairs(operators) do
                for _, k in ipairs(constants) do
                  -- Lua 5.4 compiles `x - 0` as an addition of zero (the
                  -- unchanged subtract_zero sequence), so it differs on -0.0.
                  if not (op == "-" and k == "0" and (PROFILE == "lua54" or PROFILE == "lua55")) then
                    local with_constant = "local x = ... local r for i = 1, 300 do r = x "
                        .. op .. " KONST end return r"
                    local with_register = "local x = ... local k = KONST local r for i = 1, 300 do r = x "
                        .. op .. " k end return r"
                    for _, x in ipairs(operands) do
                        local a = eval(with_constant, k, x)
                        local b = eval(with_register, k, x)
                        if a ~= b then
                            error(("x %s %s with x=%s: %s vs %s"):format(op, k, tostring(x), a, b))
                        end
                        checked = checked + 1
                    end
                    local a = eval(with_constant, k, nil)
                    local b = eval(with_register, k, nil)
                    assert(a == b, a .. " vs " .. b)
                  end
                end
            end
            assert(checked > 500)
            "#
        ),
    );
}

#[test]
fn integer_division_and_modulo_by_zero_raise_the_register_form_errors() {
    run_in(
        &[
            CompatibilityProfile::Lua53,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::Lua55,
        ],
        r#"
        local ok, message = pcall(function(x) return x % 0 end, 5)
        assert(not ok and message:find("attempt to perform 'n%%0'"), message)
        local ok2, message2 = pcall(function(x) local k = 0 return x % k end, 5)
        assert(not ok2 and message:gsub("^[^:]*:%d+: ", "") == message2:gsub("^[^:]*:%d+: ", ""))
        assert(5 / 0 == math.huge)
        local function divide(x) return x / 0 end
        assert(divide(5) == math.huge and divide(-5) == -math.huge)
        assert(math.type(5 % 3) == "integer" and (5.0 % 3) == 2.0)
        -- Float modulo by a float zero is nan, not an error.
        local function nan(x) return x % 0.0 end
        local r = nan(5)
        assert(r ~= r)
        "#,
    );
}

#[test]
fn integer_and_float_results_follow_the_profile() {
    run_in(
        &[CompatibilityProfile::Lua54, CompatibilityProfile::Lua55],
        r#"
        local function add(x) return x + 1 end
        local function sub(x) return x - 1 end
        local function mul(x) return x * 2 end
        local function div(x) return x / 2 end
        local function mod(x) return x % 4 end
        for i = 1, 400 do
            assert(math.type(add(i)) == "integer" and add(i) == i + 1)
            assert(math.type(sub(i)) == "integer")
            assert(math.type(mul(i)) == "integer")
            assert(math.type(div(i)) == "float")
            assert(math.type(mod(i)) == "integer" and mod(i) == i % 4)
            assert(math.type(add(i + 0.5)) == "float")
        end
        assert(add(math.maxinteger) == math.mininteger)
        assert(mod(-1) == 3 and mod(-4) == 0)
        assert(add("10") == 11 and math.type(add("10")) == "integer")
        assert(add("0x10") == 17)
        "#,
    );
    run_in(
        &[CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21],
        r#"
        local function add(x) return x + 1 end
        local function mod(x) return x % 4 end
        for i = 1, 400 do
            assert(add(i) == i + 1 and mod(i) == i % 4)
            assert(add("10") == 11)
        end
        assert(mod(-1) == 3 and mod(-4) == 0)
        assert(add(2^53) == 2^53 + 1)
        "#,
    );
}

#[test]
fn metamethods_receive_the_constant_as_the_second_argument() {
    run_in(
        &PROFILES,
        &format!(
            r#"{HELPERS}
            local seen
            local function record(name)
                return function(a, b)
                    seen = {{ name, a, b }}
                    return name
                end
            end
            local mt = {{
                __add = record("add"), __sub = record("sub"), __mul = record("mul"),
                __div = record("div"), __mod = record("mod"),
            }}
            local object = setmetatable({{}}, mt)
            local function check(name, result, constant)
                assert(result == name, name)
                assert(seen[1] == name and seen[2] == object, name)
                assert(seen[3] == constant, name .. " got " .. tostring(seen[3]))
                assert(math.type == nil or math.type(seen[3]) == math.type(constant))
            end
            local constant = 5
            if PROFILE == "lua51" or PROFILE == "lua52" or PROFILE == "luajit21" then
                constant = 5.0
            end
            for i = 1, 300 do
                check("add", object + 5, constant)
                check("sub", object - 5, constant)
                check("mul", object * 5, constant)
                check("div", object / 5, constant)
                check("mod", object % 5, constant)
            end
            -- The constant on the right never reaches its own handler.
            local right = setmetatable({{}}, {{ __add = function() return "right" end }})
            assert((right + 2) == "right")
            -- A left operand without a handler stays an error naming it.
            local ok, message = pcall(function() local t = {{}} return t + 1 end)
            assert(not ok and message:find("attempt to perform arithmetic on", 1, true) and message:find("'t'", 1, true), message)
            "#
        ),
    );
}

#[test]
fn errors_name_the_left_operand() {
    run_in(
        &PROFILES,
        &format!(
            r#"{HELPERS}
            local templates = {{
                "local x for i = 1, 300 do local r = x OP 2 end",
                "for i = 1, 300 do local r = missing_global OP 2 end",
                "local t = {{}} for i = 1, 300 do local r = t.field OP 2 end",
                "local u local function f() return u OP 2 end for i = 1, 300 do f() end",
                "local s = 'text' for i = 1, 300 do local r = s OP 2 end",
                "for i = 1, 300 do local r = {{}} OP 2 end",
            }}
            for _, op in ipairs{{ "+", "-", "*", "/", "%" }} do
                for _, template in ipairs(templates) do
                    local constant = template:gsub("OP", function() return op end)
                    local register = template:gsub("OP 2", function() return op .. " two" end)
                    register = "local two = 2 " .. register
                    local ok1, m1 = pcall(assert(load(constant)))
                    local ok2, m2 = pcall(assert(load(register)))
                    assert(ok1 == ok2, template)
                    if not ok1 then
                        m1 = m1:gsub("^[^:]*:%d+: ", "")
                        m2 = m2:gsub("^[^:]*:%d+: ", "")
                        assert(m1 == m2, m1 .. " vs " .. m2)
                        assert(m1:find("attempt to", 1, true), m1)
                    end
                end
            end
            "#
        ),
    );
}

#[test]
fn string_coercion_in_the_right_profiles() {
    run_in(
        &[
            CompatibilityProfile::Lua51,
            CompatibilityProfile::Lua52,
            CompatibilityProfile::Lua53,
            CompatibilityProfile::LuaJit21,
        ],
        r#"
        local function add(x) return x + 1 end
        local function mul(x) return x * 2 end
        for i = 1, 300 do
            assert(add("10") == 11 and add("0x10") == 17 and add(" 7 ") == 8)
            assert(mul("1e2") == 200)
        end
        assert(not pcall(add, "x"))
        "#,
    );
    run_in(
        &[CompatibilityProfile::Lua54, CompatibilityProfile::Lua55],
        r#"
        local function add(x) return x + 1 end
        for i = 1, 300 do
            assert(add("10") == 11 and math.type(add("10")) == "integer")
            assert(add("1.5") == 2.5)
        end
        local ok, message = pcall(add, "x")
        assert(not ok, message)
        local ok2, message2 = pcall(function(x) local k = 1 return x + k end, "x")
        assert(not ok2 and message:gsub("^[^:]*:%d+: ", "") == message2:gsub("^[^:]*:%d+: ", ""), message)
        "#,
    );
}

#[test]
fn a_metamethod_may_yield_inside_the_constant_operation() {
    run_in(
        &[
            CompatibilityProfile::Lua52,
            CompatibilityProfile::Lua53,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::Lua55,
        ],
        r#"
        local mt = {
            __add = function(a, b) local got = coroutine.yield("add", b) return got + 1 end,
            __mod = function(a, b) return coroutine.yield("mod", b) end,
        }
        local object = setmetatable({}, mt)
        local thread = coroutine.wrap(function()
            local total = 0
            for i = 1, 3 do
                total = total + (object + 10)
                total = total + (object % 7)
            end
            return total
        end)
        local first, constant = thread()
        assert(first == "add" and constant == 10)
        local steps = 0
        local value = 5
        local result
        while true do
            result = { thread(value) }
            steps = steps + 1
            if result[1] ~= "add" and result[1] ~= "mod" then break end
            value = result[1] == "add" and 100 or 1
        end
        assert(steps == 6, steps)
        assert(result[1] == 211, result[1])
        "#,
    );
}

#[test]
fn line_hooks_see_the_same_lines_for_both_forms() {
    run_in(
        &PROFILES,
        r#"
        local function lines(source)
            local seen = {}
            local f = assert(load(source))
            debug.sethook(function(_, line) seen[#seen + 1] = line end, "l")
            f(3)
            debug.sethook()
            return table.concat(seen, ",")
        end
        local constant = lines("local x = ...\nlocal y = x\n  + 1\nlocal z = y\n  * 2\nreturn z")
        local register = lines("local x, one, two = ..., 1, 2\nlocal y = x\n  + one\nlocal z = y\n  * two\nreturn z")
        assert(constant ~= "", constant)
        assert(constant == register, constant .. " vs " .. register)
        "#,
    );
}

#[test]
fn line_event_after_sethook_is_not_swallowed() {
    // Stock Lua 5.1, 5.4 and LuaJIT report the line right after sethook.
    run_in(
        &PROFILES,
        "assert(load([==[local x = 7\nlocal lines = {}\ndebug.sethook(function(_, l) lines[#lines + 1] = l end, 'l')\nlocal y = x + 1\ndebug.sethook()\nassert(table.concat(lines, ',') == '4,5', table.concat(lines, ',')) ]==]))()",
    );
    run_in(
        &PROFILES,
        "assert(load([==[local lines = {}\ndebug.sethook(function(_, l) lines[#lines + 1] = l end, 'l')\nlocal q = 1\ndebug.sethook()\nassert(table.concat(lines, ',') == '3,4', table.concat(lines, ',')) ]==]))()",
    );
}

#[test]
fn luajit_int64_cdata_operands_match_the_register_form() {
    run_in(
        &[CompatibilityProfile::LuaJit21],
        &format!(
            r#"{HELPERS}
            local operands = {{ "5LL", "-5LL", "5ULL", "9223372036854775807LL", "0LL" }}
            local constants = {{ "3", "2.5", "0", "7", "1e3" }}
            for _, op in ipairs{{ "+", "-", "*", "/", "%" }} do
                for _, k in ipairs(constants) do
                    for _, x in ipairs(operands) do
                        local a = "local x = " .. x .. " local r for i = 1, 300 do r = x " .. op .. " " .. k .. " end return r"
                        local b = "local x = " .. x .. " local k = " .. k .. " local r for i = 1, 300 do r = x " .. op .. " k end return r"
                        local ra = describe(pcall(assert(load(a))))
                        local rb = describe(pcall(assert(load(b))))
                        assert(ra == rb, op .. " " .. k .. " " .. x .. ": " .. ra .. " vs " .. rb)
                    end
                end
            end
            "#
        ),
    );
}

#[test]
fn hot_loops_with_constant_operands_agree_across_execution_modes() {
    let source = r#"
        local function work(n)
            local sum, mixed, wrap = 0, 0.5, 1
            for i = 1, n do
                sum = sum + i * 3 - 1
                mixed = (mixed + 1.5) * 1.0001 - 0.25
                wrap = (wrap * 7 + 3) % 1000003
                if i % 5 == 0 then sum = sum % 99991 end
            end
            return sum, mixed, wrap
        end
        local a, b, c = work(5000)
        return string.format("%s %.6f %s", tostring(a), b, tostring(c))
    "#;
    for profile in PROFILES {
        let mut results = Vec::new();
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            let values = lua.exec(source).unwrap();
            results.push(format!("{values:?}"));
        }
        assert_eq!(results[0], results[1], "{profile}");
    }
}

/// A chunk dumped under Lua 5.4 holds `Constant::Integer` operands. Profiles
/// without integers must load them as floats, as `LoadConstant` does.
#[test]
fn dumped_integer_constants_behave_as_the_register_form_without_integers() {
    let path = std::env::temp_dir().join(format!("oximond-k-dump-{}.bin", std::process::id()));
    let path_text = path.to_str().unwrap().replace('\\', "/");
    let dump = |body: &str| {
        let mut lua = Lua::builder()
            .profile(CompatibilityProfile::Lua54)
            .execution_mode(ExecutionMode::Interpreter)
            .build()
            .unwrap();
        lua.exec(format!(
            "local f = io.open('{path_text}', 'wb') \
             f:write(string.dump(function(x) {body} end)) f:close()"
        ))
        .unwrap();
        std::fs::read(&path).unwrap()
    };
    for op in ["+", "-", "*", "/", "%"] {
        let with_constant = dump(&format!(
            "local r for i = 1, 200 do r = x {op} 3 end return r"
        ));
        let with_register = dump(&format!(
            "local k = 3 local r for i = 1, 200 do r = x {op} k end return r"
        ));
        for profile in [
            CompatibilityProfile::Lua51,
            CompatibilityProfile::Lua52,
            CompatibilityProfile::LuaJit21,
        ] {
            for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
                let mut lua = Lua::builder()
                    .profile(profile)
                    .execution_mode(mode)
                    .build()
                    .unwrap();
                let mut run = |bytes: &[u8], x: &str| -> String {
                    std::fs::write(&path, bytes).unwrap();
                    let values = lua
                        .exec(format!(
                            "local ls = loadstring or load \
                             local f = io.open('{path_text}', 'rb') local s = f:read('*a') f:close() \
                             return tostring((assert(ls(s)))({x}))"
                        ))
                        .unwrap();
                    format!("{values:?}")
                };
                for x in ["10.5", "-3.25", "7", "0.5"] {
                    let a = run(&with_constant, x);
                    let b = run(&with_register, x);
                    assert_eq!(a, b, "{profile} {mode} x {op} 3, x={x}");
                }
            }
        }
    }
    let _ = std::fs::remove_file(&path);
}
