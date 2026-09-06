//! Lane array stores: `region.solids[idx] = region.solids[idx] + bit`, where
//! the trace meets `region` and `region.solids` through lanes (Mineclonia's
//! `mark_occupied` in `mcl_util/shape.lua`), overwrites the slot in place
//! through a `LaneArrayStore` instead of the raw-set helper, whose possible
//! rehash made the recorder refuse every recording that also read a field
//! by lane. A slot that changes type, a nil hole, a metatable with
//! `__newindex` that appears, a table swapped under the field and a table
//! made read-only all give the interpreter's results in every profile.

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

/// Runs `source` traced and interpreted in every profile, checks that the
/// outcomes agree, that native code ran and that it stored into a lane's
/// array in place, and hands the tracing statistics to `check`.
fn assert_agrees(
    source: &str,
    check: impl Fn(CompatibilityProfile, &oximond::vm::TraceStatistics),
) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?} {source}");
        assert!(expected.is_ok(), "{profile:?} {expected:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert!(
            stats.lane_array_stores_compiled > 0,
            "{profile:?} {stats:?}"
        );
        check(profile, &stats);
    }
}

/// Regions of 16 counters each, and `mark`, which bumps counter `idx` of a
/// region through the region's `solids` field. `mark (region, idx, bit)`
/// is the store under test; `size` is a field the loop reads by lane.
const REGIONS: &str = "
    local regions = {}
    for i = 1, 4 do
        local solids = {}
        for j = 1, 16 do solids[j] = 0 end
        regions[i] = { solids = solids, size = 16 }
    end
    local function mark (region, idx, bit)
        region.solids[idx] = region.solids[idx] + bit
    end
    local function sum ()
        local total, count = 0, 0
        for i = 1, 4 do
            for j = 1, 16 do
                local v = regions[i].solids[j]
                if v ~= nil then total = total + v; count = count + 1 end
            end
        end
        return total, count
    end
";

#[test]
fn counters_bumped_through_a_field_are_stored_in_place() {
    let source = format!(
        "{REGIONS}
         for k = 1, 40000 do
             local region = regions[k % 4 + 1]
             mark (region, k % region.size + 1, 1)
         end
         return sum ()"
    );
    assert_agrees(&source, |_, _| {});
}

#[test]
fn a_store_reads_back_what_it_stored() {
    // The second read of the slot follows the store.
    let source = format!(
        "{REGIONS}
         local seen = 0
         for k = 1, 40000 do
             local region = regions[k % 4 + 1]
             local idx = k % region.size + 1
             mark (region, idx, 2)
             seen = seen + region.solids[idx]
         end
         return seen, sum ()"
    );
    assert_agrees(&source, |_, _| {});
}

#[test]
fn a_slot_that_changes_type_leaves_the_store() {
    // From k = 20000 on, slot 5 of region 2 holds a float that the store
    // keeps adding to; in the integer profiles the trace stored integers.
    let source = format!(
        "{REGIONS}
         for k = 1, 40000 do
             if k == 20000 then regions[2].solids[5] = 0.25 end
             local region = regions[k % 4 + 1]
             mark (region, k % region.size + 1, 1)
         end
         return sum (), regions[2].solids[5]"
    );
    assert_agrees(&source, |_, _| {});
}

#[test]
fn a_nil_hole_takes_the_raw_set() {
    // A store into a nil slot adds the key: the in-place store must leave.
    let source = format!(
        "{REGIONS}
         local function fill (region, idx, bit)
             region.solids[idx] = (region.solids[idx] or 100) + bit
         end
         for k = 1, 40000 do
             if k == 15000 then regions[3].solids[7] = nil end
             if k == 15001 then regions[1].solids[16] = nil end
             local region = regions[k % 4 + 1]
             fill (region, k % region.size + 1, 1)
         end
         return sum ()"
    );
    assert_agrees(&source, |_, _| {});
}

#[test]
fn a_newindex_that_appears_runs_on_the_hole_it_finds() {
    // The metatable arrives mid-run. `__newindex` runs only for the slot
    // made nil; the stores into slots that hold a number stay raw.
    let source = format!(
        "{REGIONS}
         local calls = 0
         local function fill (region, idx, bit)
             region.solids[idx] = (region.solids[idx] or 1000) + bit
         end
         for k = 1, 40000 do
             if k == 10000 then
                 setmetatable (regions[1].solids, {{
                     __newindex = function (t, key, v)
                         calls = calls + 1
                         rawset (t, key, v + 1)
                     end,
                 }})
             end
             if k == 10001 then regions[1].solids[3] = nil end
             if k == 30000 then regions[1].solids[9] = nil end
             local region = regions[k % 4 + 1]
             fill (region, k % region.size + 1, 1)
         end
         return calls, sum ()"
    );
    assert_agrees(&source, |_, _| {});
}

#[test]
fn a_table_swapped_under_the_field_takes_the_stores() {
    // `region.solids` names a fresh array from k = 25000 on: the stores go
    // to it, and the old one keeps what it had.
    let source = format!(
        "{REGIONS}
         local old = regions[4].solids
         for k = 1, 40000 do
             if k == 25000 then
                 local fresh = {{}}
                 for j = 1, 16 do fresh[j] = 7 end
                 regions[4].solids = fresh
             end
             local region = regions[k % 4 + 1]
             mark (region, k % region.size + 1, 1)
         end
         local kept = 0
         for j = 1, 16 do kept = kept + old[j] end
         return kept, sum ()"
    );
    assert_agrees(&source, |_, _| {});
}

#[test]
fn booleans_toggle_in_place() {
    let source = "
        local grids = {}
        for i = 1, 4 do
            local flags = {}
            for j = 1, 12 do flags[j] = false end
            grids[i] = { flags = flags, size = 12 }
        end
        for k = 1, 40000 do
            local grid = grids[k % 4 + 1]
            local idx = k % grid.size + 1
            grid.flags[idx] = not grid.flags[idx]
        end
        local on = 0
        for i = 1, 4 do
            for j = 1, 12 do
                if grids[i].flags[j] then on = on + 1 end
            end
        end
        return on";
    assert_agrees(source, |_, _| {});
}

#[test]
fn a_read_only_array_refuses_the_native_store() {
    // The loop compiles while the array is writable; then the array is made
    // read-only, and every store must raise the interpreter's error.
    let setup = format!(
        "{REGIONS}
         function run ()
             for k = 1, 40000 do
                 local region = regions[k % 4 + 1]
                 mark (region, k % region.size + 1, 1)
             end
             return sum ()
         end
         frozen = regions[2].solids"
    );
    let check = "
        local ok = pcall (run)
        return ok, frozen[1], frozen[16]";
    for profile in PROFILES {
        let mut results = Vec::new();
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            lua.vm_mut().use_synchronous_jit_compilation();
            let _: Vec<Value> = lua.exec(&setup).unwrap();
            let writable: Vec<Value> = lua.exec(check).unwrap();
            assert_eq!(writable[0], Value::Boolean(true), "{profile:?} {mode:?}");
            let Value::Table(table) = lua.vm().runtime().get_global("frozen") else {
                panic!("a table")
            };
            lua.vm_mut()
                .runtime_mut()
                .table_set_readonly(table, true)
                .unwrap();
            let refused: Vec<Value> = lua.exec(check).unwrap();
            assert_eq!(refused[0], Value::Boolean(false), "{profile:?} {mode:?}");
            if mode == ExecutionMode::Tracing {
                let stats = lua.vm().jit_statistics().trace;
                assert!(
                    stats.lane_array_stores_compiled > 0,
                    "{profile:?} {stats:?}"
                );
            }
            results.push((writable, refused));
        }
        assert_eq!(results[0], results[1], "{profile:?}");
    }
}
