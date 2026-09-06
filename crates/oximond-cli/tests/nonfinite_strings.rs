//! String-to-number conversion of C's infinity and NaN spellings, which
//! Lua 5.1 (`strtod`) and `LuaJIT` accept and Lua 5.2 and later reject.

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

const ACCEPTED: &str = r#"
    local function isnan(x) return x ~= x end
    for _, s in ipairs({"inf", "+inf", "INF", "Infinity", " iNfInItY\t"}) do
        assert(tonumber(s) == math.huge, s)
        assert(tonumber(s, 10) == math.huge, s)
        assert(s + 0 == math.huge, s)
        assert(s * "2" == math.huge, s)
    end
    for _, s in ipairs({"-inf", "-Infinity", "  -INF\t"}) do
        assert(tonumber(s) == -math.huge, s)
        assert(-s == math.huge, s)
    end
    for _, s in ipairs({"nan", "NaN", " nan ", "-nan"}) do
        assert(isnan(tonumber(s)), s)
        assert(isnan(s + 0), s)
    end
    for _, s in ipairs({"inff", "in", "infinit", "nanx", "0xinf", "1inf"}) do
        assert(tonumber(s) == nil, s)
        assert(not pcall(function() return s + 0 end), s)
    end
    local count = 0
    for _ = 1, "inf" do
        count = count + 1
        if count == 3 then break end
    end
    assert(count == 3)
"#;

#[test]
fn lua51_converts_strtod_infinity_and_nan_spellings() {
    run(CompatibilityProfile::Lua51, ACCEPTED);
    run(
        CompatibilityProfile::Lua51,
        r#"
        local x = tonumber("nan(123)")
        assert(x ~= x)
        assert(tostring(tonumber("-nan")) == "-nan")
        assert(tostring(tonumber("nan")) == "nan")
    "#,
    );
}

#[test]
fn luajit_converts_scanner_infinity_and_nan_spellings() {
    run(CompatibilityProfile::LuaJit21, ACCEPTED);
    run(
        CompatibilityProfile::LuaJit21,
        r#"
        assert(tonumber("nan(123)") == nil)
        assert(tostring(tonumber("-nan")) == "nan")
    "#,
    );
}

#[test]
fn lua52_and_later_reject_infinity_and_nan_strings() {
    for profile in [
        CompatibilityProfile::Lua52,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        run(
            profile,
            r#"
            for _, s in ipairs({"inf", "-inf", "nan", "Infinity", " NaN "}) do
                assert(tonumber(s) == nil, s)
                assert(not pcall(function() return s + 0 end), s)
            end
            assert(tonumber("1e999") == math.huge)
        "#,
        );
    }
}

#[test]
fn library_numbers_convert_infinity_and_nan_strings_per_profile() {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        run(
            profile,
            r#"
            assert(math.floor("inf") == math.huge and math.max("-inf", 1) == 1)
            local x = math.abs("nan")
            assert(x ~= x)
            assert(string.format("%5.1f", "inf") == "  inf")
            assert(string.format("%g", "-Infinity") == "-inf")
            -- `%s` keeps the string itself.
            assert(string.format("%s", "Infinity") == "Infinity")
        "#,
        );
    }
    for profile in [
        CompatibilityProfile::Lua52,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        run(
            profile,
            r#"
            assert(not pcall(math.floor, "inf"))
            assert(not pcall(string.format, "%f", "nan"))
        "#,
        );
    }
}

#[test]
fn integer_library_arguments_convert_numeric_strings() {
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::LuaJit21,
    ] {
        run(
            profile,
            r#"
            assert(string.rep("x", "3") == "xxx")
            assert(string.sub("hello", "2", "3") == "el")
            assert(string.char("65") == "A")
        "#,
        );
    }
}
