//! Runtime type-error messages, spelled as each reference interpreter
//! spells them.

use oximond::{CompatibilityProfile, ExecutionMode, Lua};

/// Runs `cases`, a Lua table of `{function, expected message}` pairs, and
/// checks each message after its `source:line: ` prefix.
fn check(profiles: &[CompatibilityProfile], cases: &str) {
    let source = format!(
        r#"
        local up = nil
        local cases = {cases}
        for i, case in ipairs(cases) do
            local ok, message = pcall(case[1])
            assert(not ok, "case " .. i .. " did not fail")
            message = string.gsub(message, "^[^:]*:%d+: ", "")
            assert(message == case[2], "case " .. i .. ": " .. message)
        end
    "#
    );
    for &profile in profiles {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            if let Err(error) = lua.exec(&source) {
                panic!("{profile} {mode}: {error:?}");
            }
        }
    }
}

const ALL: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

#[test]
fn unnamed_operands_have_no_expected_type_detail() {
    check(
        &ALL,
        r#"{
            {function() return (nil).x end, "attempt to index a nil value"},
            {function() return (1).x end, "attempt to index a number value"},
            {function() (nil)[1] = 2 end, "attempt to index a nil value"},
            {function() return (function() end).x end, "attempt to index a function value"},
            {function() return 1 + {} end, "attempt to perform arithmetic on a table value"},
            {function() return 1 .. {} end, "attempt to concatenate a table value"},
            {function() return #nil end, "attempt to get length of a nil value"},
            {function() return 1 < "x" end, "attempt to compare number with string"},
            {function() return {} < {} end, "attempt to compare two table values"},
        }"#,
    );
}

const NAMED_CASES: &str = r#"{
    {function() local t = nil; return t.x end, "index", "local 't'", "nil"},
    {function() local t = nil; t.x = 1 end, "index", "local 't'", "nil"},
    {function() undefinedglobal.y = 1 end, "index", "global 'undefinedglobal'", "nil"},
    {function() local t = {} t.a.b = 1 end, "index", "field 'a'", "nil"},
    {function() up.x = 1 end, "index", "upvalue 'up'", "nil"},
    {function() local t = {} return 1 + t end, "perform arithmetic on", "local 't'", "table"},
    {function() local t = {} return t .. "x" end, "concatenate", "local 't'", "table"},
    {function() local u = {} return "x" .. u .. "y" end, "concatenate", "local 'u'", "table"},
    {function() return "x" .. undefinedglobal end, "concatenate", "global 'undefinedglobal'", "nil"},
    {function() local t = {} return #t.x end, "get length of", "field 'x'", "nil"},
    {function() local f; return f() end, "call", "local 'f'", "nil"},
    {function() local t = {} return t:m() end, "call", "method 'm'", "nil"},
}"#;

/// Builds the expected messages from `NAMED_CASES` in the old or new layout.
fn named_cases(new_layout: bool) -> String {
    let format = if new_layout {
        r#"function(op, name, type) return "attempt to " .. op .. " a " .. type .. " value (" .. name .. ")" end"#
    } else {
        r#"function(op, name, type) return "attempt to " .. op .. " " .. name .. " (a " .. type .. " value)" end"#
    };
    format!(
        r"(function()
            local format = {format}
            local cases = {NAMED_CASES}
            for _, case in ipairs(cases) do
                case[2] = format(case[2], case[3], case[4])
            end
            return cases
        end)()"
    )
}

#[test]
fn lua51_lua52_and_luajit_name_the_operand_before_its_type() {
    check(
        &[
            CompatibilityProfile::Lua51,
            CompatibilityProfile::Lua52,
            CompatibilityProfile::LuaJit21,
        ],
        &named_cases(false),
    );
}

#[test]
fn lua53_and_later_name_the_operand_after_its_type() {
    check(
        &[
            CompatibilityProfile::Lua53,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::Lua55,
        ],
        &named_cases(true),
    );
}

#[test]
fn fields_read_through_other_keys_are_named_per_profile() {
    let cases = |unknown: &str, integer: &str| {
        format!(
            r#"{{
                {{function() local t = {{}} return t[1].b end, "attempt to index {integer}"}},
                {{function() local t = {{}} local k = "a" return t[k].b end, "attempt to index {unknown}"}},
                {{function() local t = {{}} return t[256] + 1 end, "attempt to perform arithmetic on {unknown}"}},
            }}"#
        )
    };
    check(
        &[CompatibilityProfile::Lua51, CompatibilityProfile::Lua52],
        &cases("field '?' (a nil value)", "field '?' (a nil value)"),
    );
    check(
        &[CompatibilityProfile::LuaJit21],
        &cases("a nil value", "a nil value"),
    );
    check(
        &[CompatibilityProfile::Lua53],
        &cases("a nil value (field '?')", "a nil value (field '?')"),
    );
    check(
        &[CompatibilityProfile::Lua54, CompatibilityProfile::Lua55],
        &cases(
            "a nil value (field '?')",
            "a nil value (field 'integer index')",
        ),
    );
}

#[test]
fn lua54_string_arithmetic_reports_the_string_metamethod() {
    check(
        &[CompatibilityProfile::Lua54, CompatibilityProfile::Lua55],
        r#"{
            {function() return "a" + 1 end, "attempt to add a 'string' with a 'number'"},
            {function() return 1 + "a" end, "attempt to add a 'number' with a 'string'"},
            {function() return "10" + {} end, "attempt to add a 'string' with a 'table'"},
            {function() return {} + "10" end, "attempt to add a 'table' with a 'string'"},
            {function() local s, n = "a", 1 return n - s end, "attempt to sub a 'number' with a 'string'"},
            {function() return -"a" end, "attempt to unm a 'string' with a 'string'"},
            {function() return "a" // 1 end, "attempt to idiv a 'string' with a 'number'"},
        }"#,
    );
    check(
        &[CompatibilityProfile::Lua53],
        r#"{
            {function() local s = "a" return s + 1 end, "attempt to perform arithmetic on a string value (local 's')"},
        }"#,
    );
}

#[test]
fn library_argument_errors_name_the_function_as_lua_does() {
    let cases = |named: &str, bare: &str| {
        format!(
            r#"{{
                {{function() return string.rep() end, "bad argument #1 to '{named}' (string expected, got no value)"}},
                {{function() return string.rep("x", "y") end, "bad argument #2 to '{named}' (number expected, got string)"}},
                {{function() return math.floor("a") end, "bad argument #1 to '{floor}' (number expected, got string)"}},
                {{function() return string.format("%d", "a") end, "bad argument #2 to '{format}' (number expected, got string)"}},
                {{function() return select("a") end, "bad argument #1 to '{select}' (number expected, got string)"}},
                {{function() return pcall(string.rep) end, "bad argument #1 to '{bare}' (string expected, got no value)"}},
            }}"#,
            floor = if named == "?" { "?" } else { "floor" },
            format = if named == "?" { "?" } else { "format" },
            select = if named == "?" { "?" } else { "select" },
        )
    };
    // `pcall` returns the error; a `pcall` case checks its message.
    let unwrap = |cases: String| {
        cases.replace(
            "function() return pcall(string.rep) end",
            "function() local _, message = pcall(string.rep) error(message) end",
        )
    };
    check(&[CompatibilityProfile::Lua51], &unwrap(cases("rep", "?")));
    check(&[CompatibilityProfile::LuaJit21], &unwrap(cases("?", "?")));
    check(
        &[
            CompatibilityProfile::Lua52,
            CompatibilityProfile::Lua53,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::Lua55,
        ],
        &unwrap(cases("rep", "string.rep")),
    );
}

#[test]
fn method_calls_do_not_count_their_receiver() {
    check(
        &[CompatibilityProfile::Lua51, CompatibilityProfile::Lua54],
        r#"{
            {function() local s = "x" return s:rep({}) end, "bad argument #1 to 'rep' (number expected, got table)"},
        }"#,
    );
    check(
        &[CompatibilityProfile::LuaJit21],
        r#"{
            {function() local s = "x" return s:rep({}) end, "bad argument #2 to '?' (number expected, got table)"},
        }"#,
    );
}

#[test]
fn numeric_for_errors_name_the_bad_control_value() {
    check(
        &[
            CompatibilityProfile::Lua51,
            CompatibilityProfile::Lua52,
            CompatibilityProfile::Lua53,
            CompatibilityProfile::LuaJit21,
        ],
        r#"{
            {function() for i = "a", 2 do end end, "'for' initial value must be a number"},
            {function() for i = 1, {} do end end, "'for' limit must be a number"},
            {function() for i = 1, 2, nil do end end, "'for' step must be a number"},
        }"#,
    );
    check(
        &[CompatibilityProfile::Lua54, CompatibilityProfile::Lua55],
        r#"{
            {function() for i = "a", 2 do end end, "bad 'for' initial value (number expected, got string)"},
            {function() for i = 1, {} do end end, "bad 'for' limit (number expected, got table)"},
            {function() for i = 1, 2, nil do end end, "bad 'for' step (number expected, got nil)"},
        }"#,
    );
}

#[test]
fn lua54_names_string_constant_operands() {
    check(
        &[CompatibilityProfile::Lua54, CompatibilityProfile::Lua55],
        r#"{
            {function() return "a" & 1 end, "attempt to perform bitwise operation on a string value (constant 'a')"},
            {function() return ("x")() end, "attempt to call a string value (constant 'x')"},
        }"#,
    );
    check(
        &[CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21],
        r#"{
            {function() return ("x")() end, "attempt to call a string value"},
        }"#,
    );
}

/// Checks `pcall(f, ...)` messages, which carry no position.
fn check_pcall(profiles: &[CompatibilityProfile], cases: &str) {
    let source = format!(
        r#"
        local cases = {cases}
        for i, case in ipairs(cases) do
            local ok, message = pcall((table.unpack or unpack)(case, 1, case.n))
            assert(not ok, "case " .. i .. " did not fail")
            assert(message == case.expected, "case " .. i .. ": " .. tostring(message))
        end
    "#
    );
    for &profile in profiles {
        let mut lua = Lua::builder().profile(profile).build().unwrap();
        if let Err(error) = lua.exec(&source) {
            panic!("{profile}: {error:?}");
        }
    }
}

#[test]
fn library_checks_word_their_errors_as_lua51_does() {
    check_pcall(
        &[CompatibilityProfile::Lua51],
        r#"{
            {os.date, {}, n = 2, expected = "bad argument #1 to '?' (string expected, got table)"},
            {io.open, "x", {}, n = 3, expected = "bad argument #2 to '?' (string expected, got table)"},
            {debug.getinfo, {}, n = 2, expected = "bad argument #1 to '?' (function or level expected)"},
            {string.gsub, "a", "b", true, n = 4, expected = "bad argument #3 to '?' (string/function/table expected)"},
            {table.insert, {}, {}, 1, n = 4, expected = "bad argument #2 to '?' (number expected, got table)"},
            {coroutine.resume, 1, n = 2, expected = "bad argument #1 to '?' (coroutine expected)"},
            {setmetatable, {}, 1, n = 3, expected = "bad argument #2 to '?' (nil or table expected)"},
            {type, n = 1, expected = "bad argument #1 to '?' (value expected)"},
            {xpcall, print, n = 2, expected = "bad argument #2 to '?' (value expected)"},
            {collectgarbage, "bogus", n = 2, expected = "bad argument #1 to '?' (invalid option 'bogus')"},
            {string.format, "%d", n = 2, expected = "bad argument #2 to '?' (no value)"},
        }"#,
    );
}

#[test]
fn library_checks_word_their_errors_as_luajit_does() {
    check_pcall(
        &[CompatibilityProfile::LuaJit21],
        r#"{
            {bit.band, {}, n = 2, expected = "bad argument #1 to '?' (number expected, got table)"},
            {io.input, {}, n = 2, expected = "bad argument #1 to '?' (string expected, got table)"},
            {xpcall, print, n = 2, expected = "bad argument #2 to '?' (function expected, got no value)"},
            {string.format, "%d", n = 2, expected = "bad argument #2 to '?' (value expected)"},
            {coroutine.status, 1, n = 2, expected = "bad argument #1 to '?' (coroutine expected)"},
        }"#,
    );
}

#[test]
fn library_checks_word_their_errors_as_lua54_does() {
    check_pcall(
        &[CompatibilityProfile::Lua54, CompatibilityProfile::Lua55],
        r#"{
            {os.date, {}, n = 2, expected = "bad argument #1 to 'os.date' (string expected, got table)"},
            {io.close, {}, n = 2, expected = "bad argument #1 to 'io.close' (FILE* expected, got table)"},
            {debug.getinfo, {}, n = 2, expected = "bad argument #1 to 'debug.getinfo' (number expected, got table)"},
            {string.gsub, "a", "b", true, n = 4, expected = "bad argument #3 to 'string.gsub' (string/function/table expected, got boolean)"},
            {coroutine.resume, 1, n = 2, expected = "bad argument #1 to 'coroutine.resume' (thread expected, got number)"},
            {setmetatable, {}, 1, n = 3, expected = "bad argument #2 to 'setmetatable' (nil or table expected, got number)"},
            {rawlen, 1, n = 2, expected = "bad argument #1 to 'rawlen' (table or string expected, got number)"},
            {io.read, {}, n = 2, expected = "bad argument #1 to 'io.read' (string expected, got table)"},
            {string.byte, "abc", math.huge, n = 3, expected = "bad argument #2 to 'string.byte' (number has no integer representation)"},
            {string.pack, "i", {}, n = 3, expected = "bad argument #2 to 'string.pack' (number expected, got table)"},
        }"#,
    );
}

#[test]
fn argument_free_calls_expect_a_value_where_lua_does() {
    check_pcall(
        &[CompatibilityProfile::Lua51],
        r#"{
            {pairs, n = 1, expected = "bad argument #1 to '?' (table expected, got no value)"},
            {ipairs, n = 1, expected = "bad argument #1 to '?' (table expected, got no value)"},
            {math.max, n = 1, expected = "bad argument #1 to '?' (number expected, got no value)"},
        }"#,
    );
    check_pcall(
        &[CompatibilityProfile::Lua52],
        r#"{
            {pairs, n = 1, expected = "bad argument #1 to 'pairs' (table expected, got no value)"},
            {math.max, n = 1, expected = "bad argument #1 to 'math.max' (number expected, got no value)"},
        }"#,
    );
    check_pcall(
        &[CompatibilityProfile::LuaJit21],
        r#"{
            {pairs, n = 1, expected = "bad argument #1 to '?' (value expected)"},
            {ipairs, n = 1, expected = "bad argument #1 to '?' (value expected)"},
            {math.min, n = 1, expected = "bad argument #1 to '?' (number expected, got no value)"},
            {type, n = 1, expected = "bad argument #1 to '?' (value expected)"},
        }"#,
    );
    check_pcall(
        &[CompatibilityProfile::Lua54, CompatibilityProfile::Lua55],
        r#"{
            {pairs, n = 1, expected = "bad argument #1 to 'pairs' (value expected)"},
            {ipairs, n = 1, expected = "bad argument #1 to 'ipairs' (value expected)"},
            {math.max, n = 1, expected = "bad argument #1 to 'math.max' (value expected)"},
            {math.min, n = 1, expected = "bad argument #1 to 'math.min' (value expected)"},
            {getmetatable, n = 1, expected = "bad argument #1 to 'getmetatable' (value expected)"},
            {rawlen, n = 1, expected = "bad argument #1 to 'rawlen' (table or string expected, got no value)"},
        }"#,
    );
}

#[test]
fn multiline_expression_errors_use_lua_instruction_lines() {
    let cases = [
        ("return\n {}\n <\n 1", 4),
        ("return\n {}\n <=\n (1\n )", 5),
        ("return\n {}\n >\n 1", 4),
        ("return\n {}\n >=\n 1", 4),
        ("return\n {}\n ..\n 1", 3),
        ("return\n {}\n ..\n 1\n ..\n 2", 5),
        ("return\n (nil)\n :foo()", 3),
        ("return\n ({})\n :foo()", 3),
        ("local value =\n (nil)\n :foo()", 3),
    ];
    for profile in ALL {
        for (source, line) in cases {
            let line = if profile.is_lua51_family() && source.contains("..") {
                source.lines().count()
            } else {
                line
            };
            let mut lua = Lua::builder().profile(profile).build().unwrap();
            let probe = format!(
                "local f = assert(loadstring and loadstring({source:?}, '@probe') or load({source:?}, '@probe')); local ok, e = pcall(f); assert(not ok); assert(string.match(e, '^probe:(%d+):') == '{line}', e)"
            );
            lua.exec(&probe)
                .unwrap_or_else(|error| panic!("{profile}: {error}"));
        }
    }
}
