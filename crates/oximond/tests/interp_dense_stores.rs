//! Register-keyed stores into the array part on the interpreter's fast
//! loop. A number or boolean over a number or boolean is written in place,
//! with no poll of the collector; every other store takes the paths it
//! took before. The answers are the language's in every profile.

use oximond::runtime::Value;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn run(profile: CompatibilityProfile, mode: ExecutionMode, source: &str) -> String {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    match lua.exec(source).unwrap().as_slice() {
        [Value::String(text)] => String::from_utf8_lossy(text.as_bytes()).into_owned(),
        other => panic!("{other:?}"),
    }
}

fn run_everywhere(source: &str, expected: &str) {
    for profile in PROFILES {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            assert_eq!(run(profile, mode, source), expected, "{profile:?} {mode:?}");
        }
    }
}

#[test]
fn scalars_overwrite_scalars_in_place_and_other_stores_keep_their_path() {
    run_everywhere(
        r#"
        local t = { 1, 2.5, true, "s", {}, 6 }
        local values = { 10, false, 7.5, 4, true, "text" }
        -- 1..3: a scalar over a scalar. 4, 5: a scalar over a string and a
        -- table. 6: a string over a scalar.
        for i = 1, 6 do
            t[i] = values[i]
        end
        -- Past the end: the array grows.
        local n = 7
        t[n] = 70
        -- A float key that is an array index.
        local k = 2.0
        t[k] = true
        -- A key that is none: the hash part.
        local h = 2.5
        t[h] = 25
        local out = {}
        for i = 1, 7 do
            out[#out + 1] = tostring(t[i])
        end
        out[#out + 1] = tostring(t[2.5])
        out[#out + 1] = tostring(#t)
        return table.concat(out, "|")
        "#,
        "10|true|7.5|4|true|text|70|25|7",
    );
}

#[test]
fn newindex_runs_for_an_absent_key_and_not_for_a_present_scalar() {
    run_everywhere(
        r#"
        local log = {}
        local proxy = setmetatable({}, {
            __newindex = function(t, k, v)
                log[#log + 1] = k
                rawset(t, k, v)
            end,
        })
        for round = 1, 3 do
            for i = 1, 2 do
                proxy[i] = round * 10 + i
            end
        end
        return table.concat(log, ",") .. "|" .. proxy[1] .. "|" .. proxy[2]
        "#,
        "1,2|31|32",
    );
}

#[test]
fn a_long_run_of_in_place_stores_still_collects_garbage_made_beside_it() {
    // The in-place store skips the collector poll; the allocation beside it
    // must still reach one, or the heap grows without bound.
    run_everywhere(
        r#"
        local t = { 0, 0, 0, 0 }
        local peak = 0
        for i = 1, 200000 do
            local garbage = { i }
            t[i % 4 + 1] = garbage[1]
            if i % 20000 == 0 then
                local used = collectgarbage("count")
                if used > peak then peak = used end
            end
        end
        return tostring(peak < 20000) .. "|" .. t[1] .. "|" .. t[4]
        "#,
        "true|200000|199999",
    );
}
