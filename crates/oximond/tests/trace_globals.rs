//! Globals read and written by a trace tree whose outer sides run the loops
//! around its root.
//!
//! A trace holds a global it never stores in a window slot, read once at the
//! tree's entry. A global the tree stores is read and written through the
//! environment table instead. An outer side records code of the loops around
//! the root, so the stores there decide too: a slot read before such a store
//! kept the entry value for the whole run, and the Lua 5.1 and `LuaJIT`
//! profiles (which read globals with `GETGLOBAL`) lost every update after the
//! side was installed.

use oximond::runtime::Value;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const ALL: [CompatibilityProfile; 6] = [
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

/// Runs `source` in both modes of every profile: the results agree, and the
/// tree ran natively with an outer side installed.
fn assert_tree(source: &str) {
    for profile in ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        assert_eq!(
            format!("{actual:?}"),
            format!("{expected:?}"),
            "{profile:?} {source}"
        );
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.traces_compiled > 0, "{profile:?} {source} {stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert!(stats.side_installs > 0, "{profile:?} {source} {stats:?}");
        assert!(stats.iterations > 5000, "{profile:?} {source} {stats:?}");
    }
}

#[test]
fn an_outer_side_updates_a_global_it_reads() {
    assert_tree(
        "local g = -1 G = 0.5 local t = {x = 1}
         for i = 1, 3000 do
           if g < 0 then
             G = G + 1
             for j = 1, 4 do t.x = j end
           end
         end
         return G",
    );
}

#[test]
fn an_outer_side_stores_a_global_read_into_a_local() {
    assert_tree(
        "local g = -1 G = 0.5 local s = 0
         for i = 1, 3000 do
           if g < 0 then
             local v = G
             G = v + 1
             for j = 1, 4 do s = s + j end
           end
         end
         return G, s",
    );
}

#[test]
fn the_root_reads_a_global_an_outer_side_stores() {
    assert_tree(
        "local g = -1 G = 0.5 local s = 0 local t = {x = 1}
         for i = 1, 3000 do
           if g < 0 then
             G = G + 1
             for j = 1, 4 do s = s + G t.x = j end
           end
         end
         return G, s",
    );
}

#[test]
fn an_outer_side_reads_a_global_the_root_stores() {
    assert_tree(
        "local g = -1 G = 0 local s = 0
         for i = 1, 3000 do
           if g < 0 then
             s = s + G
             for j = 1, 4 do G = G + j end
           end
         end
         return G, s",
    );
}

#[test]
fn globals_stored_two_loops_out() {
    assert_tree(
        "local g = -1 G = 0 H = 0.5 local s = 0 local t = {x = 1}
         for k = 1, 20 do
           H = H * 1.5
           for i = 1, 300 do
             if g < 0 then
               G = G + 1
               for j = 1, 4 do s = s + H t.x = j end
             end
           end
         end
         return G, H, s",
    );
}
