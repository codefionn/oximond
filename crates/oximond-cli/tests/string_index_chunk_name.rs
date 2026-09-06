//! Float string indices and numeric `load` chunk names.

use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const ALL: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn run(source: &str) {
    for profile in ALL {
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

fn run_in(profiles: &[CompatibilityProfile], source: &str) {
    for profile in profiles {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(*profile)
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
fn float_indices_in_sub_and_byte() {
    run(r#"
        local s = "hello"
        assert(string.sub(s, 1.0) == "hello")
        assert(string.sub(s, -1.0) == "o")
        assert(string.sub(s, 2.0, 3.0) == "el")
        assert(string.sub(s, -0.0) == "hello")
        assert(string.sub(s, -2^31, 2) == "he")
        assert(string.byte(s, 1.0) == 104)
        assert(string.byte(s, -1.0) == 111)
        assert(string.byte(s, 1.0, 2.0) == 104)
        assert(string.rep("a", 3.0) == "aaa")
    "#);
    // Stock LuaJIT narrows these to 32 bits, which Oximond does not copy.
    run_in(
        &[
            CompatibilityProfile::Lua51,
            CompatibilityProfile::Lua52,
            CompatibilityProfile::Lua53,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::Lua55,
        ],
        r#"
        local s = "hello"
        assert(string.sub(s, 2^31) == "")
        assert(string.sub(s, 1, 2^31) == "hello")
        assert(string.byte(s, 2^31) == nil)
    "#,
    );
    // Stock 5.1, 5.2 and LuaJIT truncate a fraction and turn a value outside
    // the integer range, infinity and NaN into the minimum integer.
    run_in(
        &[
            CompatibilityProfile::Lua51,
            CompatibilityProfile::Lua52,
            CompatibilityProfile::LuaJit21,
        ],
        r#"
        local s = "hello"
        assert(string.sub(s, 1, 1e300) == "")
        assert(string.sub(s, -1e300, 2) == "he")
        assert(string.sub(s, 1e300) == "hello")
        assert(string.sub(s, 1.5) == "hello")
        assert(string.byte(s, 1.5) == 104)
        for _, bad in ipairs({0/0, 1/0, -1/0}) do
            assert(string.sub(s, bad) == "hello")
            assert(string.byte(s, bad) == nil)
        end
    "#,
    );
    // Stock 5.3 and later refuse a number with no integer representation.
    run_in(
        &[
            CompatibilityProfile::Lua53,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::Lua55,
        ],
        r#"
        local s = "hello"
        for _, bad in ipairs({1.5, 0/0, 1/0, -1/0, 1e300}) do
            for _, call in ipairs({string.sub, string.byte}) do
                local ok, message = pcall(call, s, bad)
                assert(not ok)
                assert(string.find(message, "number has no integer representation", 1, true), message)
            end
        end
        assert(not pcall(string.sub, s, 1, 1e300))
    "#,
    );
}

#[test]
fn numeric_chunk_names() {
    run(r#"
        local ls = loadstring or load
        local f = ls("return 1", 42)
        assert(f and f() == 1)
        local g, message = ls("x x", 42)
        assert(g == nil)
        assert(string.find(message, '[string "42"]:1:', 1, true), message)
        local h, message = ls("x x", 7.5)
        assert(h == nil)
        assert(string.find(message, '[string "7.5"]:1:', 1, true), message)
        -- reader form (5.1's loadstring takes only strings, so use load)
        local parts = {"return ", "2"}
        local i = 0
        local r = load(function() i = i + 1; return parts[i] end, 5)
        assert(r and r() == 2)
        -- other types keep the error
        for _, bad in ipairs({true, {}, print}) do
            local f2, m2 = ls("return 1", bad)
            assert(f2 == nil)
            assert(string.find(m2, "load chunk name must be a string, got", 1, true), m2)
        end
    "#);
}
