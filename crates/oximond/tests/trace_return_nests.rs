//! A nest inside a function called once per use, whose innermost body
//! returns early on some calls, called by a wrapper that returns right after
//! the call (`return not region_evaluate(...)` in `mcl_util/shape.lua`'s
//! `equal_p`). The locals around the inner loop hold nil on some calls and
//! numbers on others. A side of the inner loop's tree that leaves through
//! the return ends at the root frame's return instead of going on into the
//! wrapper, which returns before it could call the function again (the
//! unit tests of `returns_after_call` in `vm/jit/trace/native.rs` check that
//! rule on bytecode). Such a side leaves the tree through its terminal exit
//! in front of the `Return`, a `LoopComplete` exit with no guard: before the
//! rule there were none in these programs, since every side that reached the
//! return went on into the wrapper and aborted there. The
//! results are the interpreter's in every profile, also where the loops
//! leave through their guards: a metatable that appears, a nil hole, a type
//! that changes, a table swapped mid-loop. Each of those tests first runs
//! the program up to the call where its edge fires and checks that the nest
//! ran natively by then, sides that end at the return included: with
//! synchronous compiles the run up to that call is the same in both
//! programs.

use oximond::runtime::Value;
use oximond::vm::{TraceExitKind, TraceStatistics};
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
/// outcomes agree and are not errors, and returns the traced run's
/// statistics by profile.
fn run_all(source: &str) -> Vec<(CompatibilityProfile, TraceStatistics)> {
    PROFILES
        .iter()
        .map(|&profile| {
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
            let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
            assert_eq!(actual, expected, "{profile:?} {source}");
            assert!(expected.is_ok(), "{profile:?} {expected:?}");
            (profile, lua.vm().jit_statistics().trace)
        })
        .collect()
}

/// The calls of `region` in the programs below.
const CALLS: u64 = 30_000;

/// `region_evaluate` in small. `EDGE` runs before each call and may change
/// the cells, the bounds or the target.
const REGION: &str = "
    local function region(cells, nx, ny, nz, lo, hi, target)
        for x = 1, nx do
            local lx = lo[x]
            local rx = hi[x]
            for y = 1, ny do
                local ly = lo[y + 4]
                local ry = hi[y + 4]
                for z = 1, nz do
                    local v = cells[(x * 7 + y * 3 + z) % 64 + 1]
                    if v == target then
                        return true
                    end
                end
                if ly and ry and ly > ry then
                    return true
                end
            end
            if lx and rx and lx > rx then
                return true
            end
        end
        return false
    end

    local function differs(cells, nx, lo, hi, target)
        return not region(cells, nx, 4, 6, lo, hi, target)
    end

    local cells = {}
    for i = 1, 64 do cells[i] = i % 13 end
    local bounds = {{}, {1, 2, 3, 4, 1, 2, 3, 4}, {0.5, 1, nil, 2, 0.25}}
    local same, other = 0, 0
    for call = 1, 30000 do
        local target = (call % 5 == 0) and (call % 13) or 99
        local lo = bounds[call % 3 + 1]
        local hi = bounds[(call + 1) % 3 + 1]
        EDGE
        if differs(cells, 3 + call % 3, lo, hi, target) then
            other = other + 1
        else
            same = same + 1
        end
    end
    return same, other
";

fn region(edge: &str) -> String {
    REGION.replace("EDGE", edge)
}

/// The runs that left the tree through a side's terminal exit in front of
/// the root frame's `Return`.
fn return_exits(stats: &TraceStatistics) -> u64 {
    stats.interpreter_exits_of(TraceExitKind::LoopComplete, None)
}

/// Checks that the nest ran natively in `stats`, with `returns` runs at
/// least that ended at the root frame's return.
fn assert_native(profile: CompatibilityProfile, stats: &TraceStatistics, returns: u64) {
    assert!(stats.traces_compiled >= 1, "{profile:?} {stats:?}");
    assert!(stats.native_entries > CALLS / 10, "{profile:?} {stats:?}");
    assert!(stats.iterations > CALLS / 10, "{profile:?} {stats:?}");
    assert!(return_exits(stats) >= returns, "{profile:?} {stats:?}");
}

/// Runs the program up to call `call`, where an edge case fires, and checks
/// that the nest ran natively by then.
fn assert_native_before(call: u64) {
    let prefix = region(&format!("if call == {call} then return same, other end"));
    for (profile, stats) in run_all(&prefix) {
        assert_native(profile, &stats, 1);
    }
}

/// The nest runs natively: the inner loop has a tree that many calls enter.
/// Which calls return early depends on the target and the bounds, so a side
/// may leave through either return, and sides end there.
#[test]
fn a_nest_that_returns_early_runs_natively() {
    for (profile, stats) in run_all(&region("")) {
        assert_native(profile, &stats, 10);
    }
}

#[test]
fn a_metatable_that_appears_on_the_cells_leaves_through_a_guard() {
    assert_native_before(20_000);
    run_all(&region(
        "if call == 20000 then
            cells[9] = nil
            setmetatable(cells, {__index = function(_, k) return k % 11 end})
        end",
    ));
}

#[test]
fn a_nil_hole_in_the_cells_leaves_through_a_guard() {
    assert_native_before(15_000);
    run_all(&region(
        "if call == 18000 then cells[17] = nil end
        if call == 24000 then cells[17] = 4 end",
    ));
}

/// The hole is there from the first call. Target 4 then has no cell left
/// under the first `x` (cells[17] was its only one there), so on the calls
/// with target 4, `lo` empty and `hi` the full bounds, the first match is
/// cells[30] under the second `x`. The tree carries the outer loop through
/// its sides: the side that ends the `y` loop was recorded as the `x` loop
/// ended, and the side that goes on with `x` from its guard links back to
/// the `y` loop's header. Only that guard's snapshot holds the next `x`,
/// and the root used to go on with the `x` it was entered with, so it read
/// the first `x`'s cells on every outer iteration and missed the match
/// (13999 and 16001 in Lua 5.4, from call 810 on).
#[test]
fn a_nil_hole_from_the_first_call_moves_the_match_to_the_next_outer_iteration() {
    for (profile, stats) in run_all(&region("if call == 1 then cells[17] = nil end")) {
        assert_native(profile, &stats, 1);
    }
}

#[test]
fn cells_that_change_type_leave_through_a_guard() {
    assert_native_before(15_000);
    run_all(&region(
        "if call == 15000 then for i = 1, 64, 3 do cells[i] = (i % 13) + 0.5 end end
        if call == 21000 then for i = 2, 64, 5 do cells[i] = tostring(i) end end
        if call == 26000 then target = 'x' end",
    ));
}

#[test]
fn bounds_that_change_type_leave_through_a_guard() {
    assert_native_before(20_000);
    run_all(&region(
        "if call > 20000 and call % 7 == 0 then lo = {false, 'a', 1, nil, 3.5, true} end",
    ));
}

#[test]
fn cells_swapped_mid_loop_are_read_from_the_new_table() {
    assert_native_before(12_000);
    // `__index` swaps the table the loop reads in the middle of a call: the
    // loop reads `cells` from its register, so the swap shows on the next
    // call; the metamethod runs while the trace is inside the nest.
    run_all(&region(
        "if call == 12000 then
            local fresh = {}
            for i = 1, 64 do fresh[i] = (i * 5) % 13 end
            fresh[30] = nil
            setmetatable(fresh, {__index = function(_, k)
                cells = {}
                for i = 1, 64 do cells[i] = (i * 3) % 13 end
                return k % 13
            end})
            cells = fresh
        end",
    ));
}
