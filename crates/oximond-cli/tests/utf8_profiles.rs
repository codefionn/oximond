//! UTF-8 contracts exercised through Lua calls and generic-for execution.

use oximond::{CompatibilityProfile, ExecutionMode, Lua};

#[test]
fn utf8_library_availability_and_iterators_follow_the_profile() {
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua52,
        CompatibilityProfile::LuaJit21,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            let modern = matches!(
                profile,
                CompatibilityProfile::Lua53
                    | CompatibilityProfile::Lua54
                    | CompatibilityProfile::Lua55
            );
            let source = if modern {
                r##"
                local s = utf8.char(65, 0x20ac, 0x1f642)
                local expected_positions = {1, 2, 5}
                local expected_codes = {65, 0x20ac, 0x1f642}
                for repetition = 1, 60 do
                    local count = 0
                    for position, code in utf8.codes(s) do
                        count = count + 1
                        assert(position == expected_positions[count])
                        assert(code == expected_codes[count])
                        assert(math.type(position) == "integer")
                        assert(math.type(code) == "integer")
                    end
                    assert(count == 3)
                    local f, state, control = utf8.codes(s)
                    assert(type(f) == "function" and state == s and control == 0)
                    assert(f == utf8.codes("other"))
                    local position, code = f(state, control)
                    assert(position == 1 and code == 65)
                    position, code = f(state, position)
                    assert(position == 2 and code == 0x20ac)
                    position, code = f(state, position)
                    assert(position == 5 and code == 0x1f642)
                    assert(select("#", f(state, position)) == 0)
                    assert(f("z", 0) == 1)
                    assert(f("z", 0) == 1)
                end
                local bad = "a" .. string.char(255)
                local f, state, control = utf8.codes(bad)
                local position, code = f(state, control)
                assert(position == 1 and code == 97)
                assert(not pcall(f, state, position))
                assert(utf8.len(s, 2, 2) == 1)
                assert(utf8.codepoint(bad, 1) == 97)
                assert(utf8.offset(bad, 2) == 2)
                local function check_integer_error(f, ...)
                    local ok, message = pcall(f, ...)
                    assert(not ok)
                    assert(string.find(message, "number has no integer representation", 1, true))
                end
                for _, value in ipairs({1.5, math.huge, -math.huge, 0/0, 1e30}) do
                    check_integer_error(utf8.char, value)
                    check_integer_error(utf8.offset, "abc", value)
                    check_integer_error(utf8.offset, "abc", 1, value)
                    check_integer_error(utf8.len, "abc", value)
                    check_integer_error(utf8.len, "abc", 1, value)
                    check_integer_error(utf8.codepoint, "abc", value)
                    check_integer_error(utf8.codepoint, "abc", 1, value)
                end
                local ok, message = pcall(utf8.offset, "abc", 1, 0)
                assert(not ok)
                local bounds = _VERSION == "Lua 5.3" and "position out of range" or "position out of bounds"
                assert(string.find(message, bounds, 1, true))
                local surrogate = utf8.char(0xd800)
                if _VERSION == "Lua 5.3" then
                    assert(utf8.len(surrogate) == 1)
                    assert(not pcall(utf8.char, 0x110000))
                else
                    assert(utf8.len(surrogate) == nil)
                    assert(utf8.len(surrogate, 1, -1, true) == 1)
                    local extended = utf8.char(0x7fffffff)
                    assert(#extended == 6)
                    assert(utf8.codepoint(extended, 1, 1, true) == 0x7fffffff)
                    for position, code in utf8.codes(extended, true) do
                        assert(position == 1 and code == 0x7fffffff)
                    end
                end
                "##
            } else {
                "assert(utf8 == nil)"
            };
            lua.exec(source)
                .unwrap_or_else(|error| panic!("{profile} {mode}: {error:?}"));
        }
    }
}
