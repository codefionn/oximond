//! The metatable shared by all strings, and Lua 5.4's string arithmetic
//! metamethods in it.

use oximond::{CompatibilityProfile, ExecutionMode, Lua};

fn run(profiles: &[CompatibilityProfile], source: &str) {
    for &profile in profiles {
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

const ALL: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

#[test]
fn strings_share_a_metatable_indexing_the_string_library() {
    run(
        &ALL,
        r#"
        local mt = getmetatable("")
        assert(type(mt) == "table" and mt.__index == string)
        assert(getmetatable("x") == mt)
        assert(("abc"):upper() == "ABC" and ("x").len == string.len)
        -- The metatable, not the global, finds the library.
        local library = string
        string = nil
        assert(("keep"):upper() == "KEEP")
        string = library
        debug.setmetatable("", nil)
        local ok, message = pcall(function() return ("x"):upper() end)
        assert(not ok and string.find(message, "attempt to index a string value", 1, true), message)
        debug.setmetatable("", mt)
        assert(("back"):upper() == "BACK")
    "#,
    );
}

#[test]
fn lua51_to_lua53_strings_have_only_index_and_convert_in_the_vm() {
    run(
        &[
            CompatibilityProfile::Lua51,
            CompatibilityProfile::Lua52,
            CompatibilityProfile::Lua53,
            CompatibilityProfile::LuaJit21,
        ],
        r#"
        local mt = getmetatable("")
        assert(next(mt) == "__index" and next(mt, "__index") == nil)
        assert("10" + 1 == 11)
        debug.setmetatable("", nil)
        assert("10" + 1 == 11)
        debug.setmetatable("", mt)
    "#,
    );
}

#[test]
fn lua54_string_arithmetic_runs_through_the_metatable() {
    run(
        &[CompatibilityProfile::Lua54, CompatibilityProfile::Lua55],
        r#"
        local mt = getmetatable("")
        for _, event in ipairs({"__add", "__sub", "__mul", "__div", "__mod", "__pow", "__unm", "__idiv"}) do
            assert(type(mt[event]) == "function", event)
        end
        assert("10" + 1 == 11 and math.type("10" + 1) == "integer")
        assert("3" * "4" == 12 and "7" % "-3" == -2 and "7.5" % 2 == 1.5)
        assert("-7" // 2 == -4 and "2" ^ "3" == 8.0 and -"0x10" == -16)
        assert(" 5 " - 1 == 4 and "1e2" * 1 == 100.0 and "1" / "2" == 0.5)
        assert(mt.__add("10", 2) == 12 and mt.__unm("3") == -3)
        local function message(f, ...)
            local ok, err = pcall(f, ...)
            assert(not ok)
            return (string.gsub(err, "^[^:]*:%d+: ", ""))
        end
        assert(message(function() return "a" + 1 end) == "attempt to add a 'string' with a 'number'")
        assert(message(function() return {} + "10" end) == "attempt to add a 'table' with a 'string'")
        assert(message(function() return -"a" end) == "attempt to unm a 'string' with a 'string'")
        assert(message(mt.__unm, "a") == "attempt to unm a 'string' with a 'nil'")
        -- Errors from the arithmetic itself carry no position.
        local ok, err = pcall(function() return "1" // "0" end)
        assert(not ok and err == "attempt to divide by zero", err)
        ok, err = pcall(function() return "1" % "0" end)
        assert(not ok and err == "attempt to perform 'n%0'", err)
        -- A second operand with its own metamethod takes the operation.
        local object = setmetatable({}, {__add = function(a, b) return "object" end})
        assert("10" + object == "object" and "x" + object == "object")
        assert(object + "10" == "object")
        -- So it does when the metamethod is called directly, as trymt does.
        assert(mt.__add("1", object) == "object" and mt.__add("x", object) == "object")
        assert(select(2, pcall(mt.__add, "1", object)) == "object")
        assert(select(2, xpcall(mt.__add, print, "1", object)) == "object")
        assert(mt.__add("1", 2) == 3)
        assert(message(mt.__add, object, "1") == "attempt to add a 'table' with a 'string'")
        -- Without the metatable, strings do not convert.
        debug.setmetatable("", nil)
        assert(message(function() local s = "10" return s + 1 end)
            == "attempt to perform arithmetic on a string value (local 's')")
        debug.setmetatable("", mt)
        -- Bitwise operators have no string metamethods.
        assert(not pcall(function() return "3" | 4 end))
    "#,
    );
}
