//! An outer side trace that runs past the loops around its root: through a
//! short loop beside them, unrolled, to the function's return, where it
//! leaves for the interpreter. The results are the interpreter's in every
//! profile, including when the short loop changes length or the function
//! returns values.

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

fn outcome(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
) -> (Result<Vec<Value>, String>, Lua) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let results = lua.exec(source).map_err(|error| format!("{error:?}"));
    (results, lua)
}

/// Runs `source` traced and interpreted in every profile and checks that the
/// outcomes agree. In Lua 5.1, whose loops are float loops a tree carries
/// (integer loops in later profiles may not repay their outer sides), the
/// tree must also have taken a side past its nest, a side of a side.
fn assert_beyond(source: &str) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?} {source}");
        assert!(expected.is_ok(), "{profile:?} {expected:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        if profile == CompatibilityProfile::Lua51 {
            assert!(stats.side_installs >= 2, "{profile:?} {stats:?}");
        }
    }
}

#[test]
fn a_nest_and_the_short_loop_after_it_run_to_the_return() {
    assert_beyond(
        "local items = {}
         for i = 1, 6 do items[i] = {x = i, v = i * 0.5} end
         local function step(dt)
             for i = 1, #items do
                 local a = items[i]
                 for j = i + 1, #items do
                     local b = items[j]
                     a.v = a.v + (b.x - a.x) * dt
                 end
             end
             for i = 1, #items do
                 local a = items[i]
                 a.x = a.x + a.v * dt
             end
         end
         for _ = 1, 12000 do step(0.001) end
         local total = 0
         for i = 1, #items do total = total + items[i].x + items[i].v end
         return math.floor(total * 1000)",
    );
}

#[test]
fn the_short_loop_changing_length_leaves_at_its_step() {
    assert_beyond(
        "local items = {}
         for i = 1, 6 do items[i] = {x = i, v = i * 0.5} end
         local function step(dt, n)
             for i = 1, #items do
                 local a = items[i]
                 for j = i + 1, #items do
                     local b = items[j]
                     a.v = a.v + (b.x - a.x) * dt
                 end
             end
             local moved = 0
             for i = 1, n do
                 local a = items[i]
                 a.x = a.x + a.v * dt
                 moved = moved + 1
             end
             return moved
         end
         local count = 0
         for round = 1, 12000 do
             count = count + step(0.001, round % 997 == 0 and 4 or 6)
         end
         local total = 0
         for i = 1, #items do total = total + items[i].x + items[i].v end
         return math.floor(total * 1000), count",
    );
}
