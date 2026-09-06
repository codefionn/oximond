//! A `t[k]` read that misses a table without a metatable yields nil in the
//! fast loop. These programs pin what the register-keyed read must keep:
//! odd keys, array holes, `__index` in every form, and string receivers.

use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const ALL_PROFILES: [CompatibilityProfile; 4] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn run(source: &str) {
    for profile in ALL_PROFILES {
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

#[test]
fn plain_table_misses_read_nil() {
    run(r#"
        local t = {10, 20, nil, 40, x = 1, [2.5] = "f", [true] = "b"}
        local nan, nilkey = 0 / 0, nil
        local keys = {1, 1.0, 3, 5, 0, -1, "x", "y", 2.5, 2.0, true, false, nan, nilkey, 1e300}
        local want = {10, 10, nil, nil, nil, nil, 1, nil, "f", 20, "b", nil, nil, nil, nil}
        for round = 1, 300 do
            for i = 1, 15 do
                local k = keys[i]
                local got = t[k]
                if got ~= want[i] then error("key " .. i .. " round " .. round) end
            end
        end
        local hole = {1, 2, 3}
        hole[2] = nil
        for i = 1, 200 do assert(hole[2] == nil and hole[4] == nil and hole[3] == 3) end
        local empty = {}
        for i = 1, 200 do assert(empty[i] == nil and empty[nan] == nil) end
    "#);
}

#[test]
fn metatables_still_apply_on_misses() {
    run(r#"
        local log = {}
        local fn = setmetatable({own = 1, [1] = "one"}, {__index = function(t, k)
            log[#log + 1] = tostring(k)
            return "dyn"
        end})
        local chain = setmetatable({}, {__index = setmetatable({[7] = "mid"}, {__index = {[8] = "deep"}})})
        local k7, k8, k9, own = 7, 8, 9, "own"
        for i = 1, 200 do
            assert(fn[1] == "one" and fn[own] == 1)
            assert(chain[k7] == "mid" and chain[k8] == "deep" and chain[k9] == nil)
        end
        assert(fn[5] == "dyn" and fn[6] == "dyn")
        assert(table.concat(log, ",") == "5,6")
        local nonmeta = setmetatable({}, {})
        for i = 1, 200 do assert(nonmeta[i] == nil) end
        local late = {}
        for i = 1, 200 do assert(late[i] == nil) end
        setmetatable(late, {__index = function(_, k) return k * 2 end})
        assert(late[4] == 8)
    "#);
}

#[test]
fn string_receivers_and_bad_receivers() {
    run(r#"
        local s = "abc"
        local k = "len"
        for i = 1, 200 do assert(s[k] == string.len and s[1] == nil) end
        local n = 5
        local ok, err = pcall(function() return n[1] end)
        assert(not ok and tostring(err):find("attempt to index")
            and tostring(err):find("number value"), err)
        local ok2, err2 = pcall(function() local t = nil; return t[1] end)
        assert(not ok2 and tostring(err2):find("attempt to index")
            and tostring(err2):find("nil value"), err2)
    "#);
}
