//! `string.format`, `string.byte`, `string.sub`, `string.rep` and
//! `string.gsub` edge cases. Every expected value is what the stock
//! interpreter for the profile prints (lua5.1, lua5.4, lua5.5, luajit).

use oximond::{CompatibilityProfile, ExecutionMode, Lua};

fn run(profile: CompatibilityProfile, source: &str) {
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

const MODERN: [CompatibilityProfile; 2] =
    [CompatibilityProfile::Lua54, CompatibilityProfile::Lua55];

#[test]
fn format_q_escapes_control_bytes_like_each_profile() {
    let source = r#"
        local s = string.format("%q", "a\tb\1c\0012\r\n\0\127\200\0019")
        assert(s == EXPECTED, s)
    "#;
    // Lua 5.1 keeps control bytes other than CR, NUL and newline raw.
    run(
        CompatibilityProfile::Lua51,
        &source.replace("EXPECTED", r#""\"a\tb\1c\0012\\r\\\n\\000\127\200\0019\"""#),
    );
    // Later versions and LuaJIT write \ddd, padded only before a digit.
    let modern = r#""\"a\\9b\\1c\\0012\\13\\\n\\0\\127\200\\0019\"""#;
    for profile in MODERN.into_iter().chain([CompatibilityProfile::LuaJit21]) {
        run(profile, &source.replace("EXPECTED", modern));
    }
}

#[test]
fn format_d_refuses_floats_without_an_integer_value_from_lua53() {
    for profile in MODERN {
        run(
            profile,
            r#"
            local ok, message = pcall(string.format, "%d", 1.5)
            assert(not ok and message:find("number has no integer representation", 1, true), message)
            assert(not pcall(string.format, "%d", "1.5"))
            assert(not pcall(string.format, "%x", 1.5))
            assert(not pcall(string.format, "%d", 2^63))
            assert(string.format("%d", 2.0) == "2")
            assert(string.format("%d", "2.0") == "2")
            assert(string.format("%5.2d|", 3) == "   03|")
        "#,
        );
    }
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        run(
            profile,
            r#"
            assert(string.format("%d", 1.5) == "1")
            assert(string.format("%d", "1.5") == "1")
            assert(string.format("%x", 1.5) == "1")
            assert(string.format("%d", 2^63) == "-9223372036854775808")
        "#,
        );
    }
}

#[test]
fn format_c_of_zero_ends_the_output_only_in_lua51() {
    run(
        CompatibilityProfile::Lua51,
        r#"
        assert(#string.format("%c%c", 66, 0) == 1)
        assert(string.format("a%cb", 0) == "ab")
        assert(string.format("%5c|", 0) == "    |")
        assert(string.format("%-5c|", 0) == "|")
    "#,
    );
    for profile in [
        CompatibilityProfile::LuaJit21,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        run(
            profile,
            r#"
            assert(string.format("a%cb", 0) == "a\0b")
            assert(string.format("%c%c", 66, 0) == "B\0")
            assert(string.format("%-5c|", 0) == "\0    |")
        "#,
        );
    }
}

#[test]
fn byte_and_sub_clamp_huge_integers_and_reject_unrepresentable_floats() {
    for profile in MODERN {
        run(
            profile,
            r##"
            local max, min = math.maxinteger, math.mininteger
            assert(select("#", string.byte("abc", max)) == 0)
            assert(select("#", string.byte("abc", min)) == 0)
            assert(select("#", string.byte("abc", max - 1, max)) == 0)
            assert(select("#", string.byte("abc", min, max)) == 3)
            assert(string.sub("abc", min, max) == "abc")
            assert(string.sub("abc", max) == "")
            for _, call in ipairs {
                function() return string.byte("abc", 2^63) end,
                function() return string.byte("abc", 1e300) end,
                function() return string.byte("abc", 1.5) end,
                function() return string.sub("abc", 2, 2^63) end,
                function() return string.sub("abc", 0/0) end,
            } do
                local ok, message = pcall(call)
                assert(not ok and message:find("number has no integer representation", 1, true), message)
            end
            assert(string.byte("abc", -2^63) == nil)
        "##,
        );
    }
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        run(
            profile,
            r##"
            assert(select("#", string.byte("abc", 2^63)) == 0)
            assert(select("#", string.byte("abc", 1e300)) == 0)
            assert(select("#", string.byte("abc", -1e300)) == 0)
            assert(select("#", string.byte("abc", 1.5)) == 1)
            -- Out of range converts to the minimum integer, so an end index
            -- of 2^63 selects nothing, as in stock.
            assert(string.sub("abc", 2, 2^63) == "")
            assert(string.sub("abc", -1e300, 2) == "ab")
        "##,
        );
    }
}

#[test]
fn rep_takes_a_separator_except_in_stock_lua51() {
    run(
        CompatibilityProfile::Lua51,
        r#"
        assert(string.rep("ab", 3, ",") == "ababab")
        assert(string.rep("ab", -1) == "")
        assert(string.rep("ab", 1e300) == "")
    "#,
    );
    for profile in [
        CompatibilityProfile::LuaJit21,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        run(
            profile,
            r#"
            assert(string.rep("ab", 3, ",") == "ab,ab,ab")
            assert(string.rep("x", 1, ",") == "x")
            assert(string.rep("ab", 0, ",") == "")
            assert(string.rep("ab", -1, ",") == "")
            assert(string.rep("ab", 3, nil) == "ababab")
        "#,
        );
    }
}

#[test]
fn gsub_skips_an_empty_match_right_after_a_match_from_lua53() {
    // Literal, function and table replacements take different paths.
    let check = r#"
        local function check(expected, count, s, pattern, replacement, limit)
            local result, n = string.gsub(s, pattern, replacement, limit)
            assert(result == expected and n == count, result .. " " .. tostring(n))
        end
        local function wrap(x) return "<" .. x .. ">" end
        check("ANCHORED", 1, "aaa", "^a", "-")
        check("ANCHORED", 1, "aaa", "^a", function() return "-" end)
    "#;
    for profile in MODERN {
        run(
            profile,
            &format!(
                r#"{}
                check("-,-y,,z", 2, "x,y,,z", "x*", "-", 2)
                check("-,-y-,-,-z-", 6, "x,y,,z", "x*", "-")
                check("-a-c-", 3, "abc", "b*", "-")
                check("-abc", 1, "abc", "b*", "-", 1)
                check("-a-c", 2, "abc", "b*", "-", 2)
                check("-a-c-", 3, "abc", "b*", "-", 3)
                check("abc", 0, "abc", "b*", "-", 0)
                check("<>a<b>c<>", 3, "abc", "b*", wrap)
                check("aBc", 3, "abc", "b*", {{ b = "B" }})
                check("abbc", 3, "abc", "b*", "%0%0")
                check("-a-bc", 2, "abc", "", "-", 2)
                check("-", 1, "", "x*", "-")
                check("abc-", 1, "abc", "$", "-")
            "#,
                check.replace("ANCHORED", "-aa")
            ),
        );
    }
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        run(
            profile,
            &format!(
                r#"{}
                check("--,y,,z", 2, "x,y,,z", "x*", "-", 2)
                check("-a--c-", 4, "abc", "b*", "-")
                check("-abc", 1, "abc", "b*", "-", 1)
                check("-a-c", 2, "abc", "b*", "-", 2)
                check("-a--c", 3, "abc", "b*", "-", 3)
                check("<>a<b><>c<>", 4, "abc", "b*", wrap)
                check("-a-bc", 2, "abc", "", "-", 2)
            "#,
                check.replace("ANCHORED", "-aa")
            ),
        );
    }
}
