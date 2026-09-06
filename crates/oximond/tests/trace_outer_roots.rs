//! Outer roots that end at an inner header, and loop links. A nest whose
//! inner loops no tree of their own can carry (two loops side by side inside
//! the outer one, or an inner loop whose root was given up) is recorded from
//! the outer loop's header: the root ends in front of the first inner loop
//! with a terminal exit, the inner loop runs as a loop side of the outer
//! tree, the side of its closing guard runs the rest of the outer iteration
//! (ending in front of the next inner loop, which is a loop side too) and
//! links back to the outer header. A side of a loop side that comes round to
//! that loop's header rejoins it (`TraceTerminator::LinkLoop`). The results
//! are the interpreter's in every profile, also where the loops leave
//! through their guards: a type that changes, a nil hole, a metatable that
//! appears, a table swapped mid-loop, an error.

use oximond::runtime::Value;
use oximond::vm::TraceStatistics;
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

/// The outer trips of the nests below.
const OUTER: u64 = 20_000;

/// Checks that the outer loop ran natively as a tree with `loop_sides` loop
/// sides at least: far fewer entries than outer trips, and more native
/// iterations than outer trips.
fn assert_outer_tree(profile: CompatibilityProfile, stats: &TraceStatistics, loop_sides: u64) {
    assert!(stats.loop_sides >= loop_sides, "{profile:?} {stats:?}");
    assert!(stats.native_entries >= 1, "{profile:?} {stats:?}");
    assert!(stats.native_entries < OUTER / 4, "{profile:?} {stats:?}");
    assert!(stats.iterations > OUTER, "{profile:?} {stats:?}");
}

/// Two `while` loops side by side in a `for` loop. Each has a root of its
/// own, but no outer side of either can run the other (`nest::can_cover`
/// says no for siblings), so the outer loop's root ends in front of the
/// first and both run as loop sides of its tree.
const SIBLINGS: &str = "
    local function run(n)
        local acc = 0
        for i = 1, n do
            local j = 0
            while j < 40 do
                acc = acc + j
                j = j + 1
            end
            local k = 0
            while k < 30 do
                acc = acc + k * 2
                k = k + 1
            end
        end
        return acc
    end
    return run(20000)
";

#[test]
fn two_sibling_while_loops_in_a_for_run_as_loop_sides_of_the_outer_root() {
    for (profile, stats) in run_all(SIBLINGS) {
        assert_outer_tree(profile, &stats, 2);
    }
}

#[test]
fn sibling_loops_of_changing_lengths_agree_with_the_interpreter() {
    // The first loop runs no trips on some outer iterations and a growing
    // number on others; the second leaves through a `break`.
    let source = "
        local function run(n)
            local acc = 0
            for i = 1, n do
                local j = 0
                while j < i % 9 do
                    acc = acc + j
                    j = j + 1
                end
                local k = 0
                while k < 30 do
                    if k == i % 31 then break end
                    acc = acc + k
                    k = k + 1
                end
            end
            return acc
        end
        return run(20000)
    ";
    for (profile, stats) in run_all(source) {
        assert!(stats.native_entries >= 1, "{profile:?} {stats:?}");
    }
}

/// A `for ... in pairs(t)` loop whose root is given up: on its first outer
/// iterations, when the inner header gets hot, the body takes a path that
/// calls `os.clock`, which no recording takes. Later outer iterations no
/// longer do, and the outer loop's root ends in front of the inner loop.
const REJECTED_PAIRS: &str = "
    local t = {}
    for i = 1, 50 do t[i * 7] = i end
    local function run(n)
        local acc = 0
        for i = 1, n do
            for k, v in pairs(t) do
                if i <= 30 then local _ = os.clock() end
                acc = acc + v + k
            end
        end
        return acc
    end
    return run(20000)
";

#[test]
fn a_nest_whose_pairs_root_was_rejected_runs_the_pairs_loop_as_a_loop_side() {
    for (profile, stats) in run_all(REJECTED_PAIRS) {
        assert_outer_tree(profile, &stats, 1);
    }
}

/// A nest like [`REJECTED_PAIRS`] whose inner loop is `inner`.
fn rejected_inner(inner: &str) -> String {
    format!(
        "local t = {{}}
         for i = 1, 50 do t[i] = i end
         local function run(n)
             local acc = 0
             for i = 1, n do
                 {inner}
                     if i <= 30 then local _ = os.clock() end
                     acc = acc + v
                 end
             end
             return acc
         end
         return run(20000)"
    )
}

#[test]
fn a_rejected_numeric_loop_runs_as_a_loop_side() {
    for (profile, stats) in run_all(&rejected_inner("for j = 1, #t do local v = t[j]")) {
        assert_outer_tree(profile, &stats, 1);
    }
}

#[test]
fn a_rejected_ipairs_loop_runs_as_a_loop_side_and_leaves_at_its_end() {
    // The loop side runs the trips. Its end reads past the array of a table
    // the tree reads by lane, which no side records (a nil read on a lane
    // needs a proof of no metatable), so every outer iteration leaves there.
    for (profile, stats) in run_all(&rejected_inner("for _, v in ipairs(t) do")) {
        assert!(stats.loop_sides >= 1, "{profile:?} {stats:?}");
        assert!(stats.iterations > OUTER * 40, "{profile:?} {stats:?}");
    }
}

/// A loop side whose body branches on data: the side of that branch guard
/// runs the rest of the iteration and comes round to the loop side's header,
/// where it rejoins the loop instead of ending in front of a copy of it.
const LINKED: &str = "
    local out = {0}
    local function run(n)
        local acc = 0
        for i = 1, n do
            local j = 0
            while j < 40 do
                if j % 3 == 0 then out[1] = out[1] + j else acc = acc + 1 end
                j = j + 1
            end
            local k = 0
            while k < 30 do
                acc = acc + k * 2
                k = k + 1
            end
        end
        return acc, out[1]
    end
    return run(20000)
";

#[test]
fn a_side_of_a_loop_side_rejoins_its_loop() {
    for (profile, stats) in run_all(LINKED) {
        assert_outer_tree(profile, &stats, 2);
        assert!(stats.loop_links >= 1, "{profile:?} {stats:?}");
    }
}

#[test]
fn a_side_that_writes_what_the_loop_side_holds_still_ends_in_front_of_a_copy() {
    // The side of `j == 7` lowers `lim`, which the loop side reads from its
    // entry state: rejoining it would keep the old bound, so the side ends in
    // front of a copy of the loop instead.
    let source = "
        local function run(n)
            local acc = 0
            for i = 1, n do
                local j, lim = 0, 40 + i % 3
                while j < lim do
                    if j == 7 then lim = lim - 5 end
                    acc = acc + j
                    j = j + 1
                end
                local k = 0
                while k < 30 do
                    acc = acc + k
                    k = k + 1
                end
            end
            return acc
        end
        return run(20000)
    ";
    for (profile, stats) in run_all(source) {
        assert!(stats.native_entries < OUTER / 4, "{profile:?} {stats:?}");
    }
}

#[test]
fn loop_sides_of_an_outer_root_leave_where_the_interpreter_would() {
    // A float appears in the accumulator, a nil hole and then a metatable
    // appear in the table the first loop reads, and the table is swapped
    // for another one late in the run. Every change leaves through a guard
    // of the loop side or of the sides around it.
    let source = "
        local rows = {}
        for r = 1, 40 do rows[r] = r end
        local other = {}
        for r = 1, 40 do other[r] = r * 2 end
        local function run(n)
            local acc = 0
            for i = 1, n do
                if i == 9000 then acc = acc + 0.5 end
                if i == 12000 then rows[17] = nil end
                if i == 13000 then
                    rows[17] = 17
                    setmetatable(rows, {__index = function() return 1 end})
                end
                if i == 16000 then rows = other end
                local j = 1
                while j <= 40 do
                    local v = rows[j]
                    if v then acc = acc + v end
                    j = j + 1
                end
                local k = 0
                while k < 30 do
                    acc = acc + k
                    k = k + 1
                end
            end
            return acc
        end
        return run(20000)
    ";
    for (profile, stats) in run_all(source) {
        assert!(stats.native_entries >= 1, "{profile:?} {stats:?}");
    }
}

#[test]
fn an_error_inside_a_loop_side_reports_what_the_interpreter_reports() {
    let source = "
        local function run(n)
            local acc = 0
            for i = 1, n do
                local j = 0
                while j < 40 do
                    local d = (i == n and j == 20) and {} or 1
                    acc = acc + j * d
                    j = j + 1
                end
                local k = 0
                while k < 30 do
                    acc = acc + k
                    k = k + 1
                end
            end
            return acc
        end
        local ok, message = pcall(run, 20000)
        return ok, message, debug.traceback ~= nil
    ";
    for (profile, stats) in run_all(source) {
        assert!(stats.native_entries >= 1, "{profile:?} {stats:?}");
    }
}

#[test]
fn a_count_hook_keeps_loop_sides_out_and_fires_where_the_interpreter_fires() {
    // With a count hook installed the outer loop's root never ends in front
    // of an inner loop: the hook's budget runs through the root's loop only.
    let source = "
        local fired = 0
        debug.sethook(function() fired = fired + 1 end, '', 997)
        local function run(n)
            local acc = 0
            for i = 1, n do
                local j = 0
                while j < 40 do
                    acc = acc + j
                    j = j + 1
                end
                local k = 0
                while k < 30 do
                    acc = acc + k * 2
                    k = k + 1
                end
            end
            return acc
        end
        local acc = run(4000)
        debug.sethook()
        return acc, fired
    ";
    for (profile, stats) in run_all(source) {
        assert_eq!(stats.loop_sides, 0, "{profile:?} {stats:?}");
        assert_eq!(stats.loop_links, 0, "{profile:?} {stats:?}");
    }
}

/// Checks that the tree ran no more native iterations than the nest has
/// trips (`trips`, its inner trips and outer trips together) and a margin:
/// a side that resumed the interpreter with a stale loop variable runs outer
/// iterations again, and their trips with them.
fn assert_trips(profile: CompatibilityProfile, stats: &TraceStatistics, trips: u64) {
    assert!(stats.native_entries >= 1, "{profile:?} {stats:?}");
    assert!(
        stats.iterations <= trips + trips / 10,
        "{profile:?} {stats:?}"
    );
}

/// A local the inner loop carries turns from an integer into a float on
/// every hundredth outer iteration, inside a `while` loop with a sibling.
/// The side of that branch guard ends in front of a copy of the loop, which
/// runs with a float. Neither reads the outer loop's variable, which the
/// tree's links have moved on since the run's entry, so both must still
/// hand it on as a value: the exit they end in restores it.
#[test]
fn a_carried_local_that_turns_float_in_sibling_while_loops_agrees_with_the_interpreter() {
    let source = "
        local function run(n)
            local acc = 0
            for i = 1, n do
                local x = 0
                local j = 0
                while j < 40 do
                    if i % 100 == 0 then x = x + 0.5 else x = x + 1 end
                    j = j + 1
                end
                acc = acc + x
                local k = 0
                while k < 30 do
                    acc = acc + k
                    k = k + 1
                end
            end
            return acc
        end
        return run(20000)
    ";
    for (profile, stats) in run_all(source) {
        assert_trips(profile, &stats, OUTER * 71);
    }
}

#[test]
fn a_carried_local_that_turns_float_late_in_the_loop_agrees_with_the_interpreter() {
    // The float appears on the 26th trip, and only the second sibling runs
    // after it: the loop side of the copy runs 14 trips with a float.
    let source = "
        local function run(n)
            local acc = 0
            for i = 1, n do
                local x = 0
                local j = 0
                while j < 40 do
                    if j == 25 and i % 100 == 0 then x = x + 0.5 end
                    j = j + 1
                end
                acc = acc + x
                local k = 0
                while k < 30 do k = k + 1 end
            end
            return acc
        end
        return run(20000)
    ";
    for (profile, stats) in run_all(source) {
        assert_trips(profile, &stats, OUTER * 71);
    }
}

#[test]
fn a_carried_local_that_turns_float_in_a_rejected_while_loop_agrees_with_the_interpreter() {
    let source = "
        local function run(n)
            local acc = 0
            for i = 1, n do
                local x = 0
                local j = 0
                while j < 40 do
                    if i <= 30 then local _ = os.clock() end
                    if i % 100 == 0 then x = x + 0.5 else x = x + 1 end
                    j = j + 1
                end
                acc = acc + x
            end
            return acc
        end
        return run(20000)
    ";
    for (profile, stats) in run_all(source) {
        assert_trips(profile, &stats, OUTER * 41);
    }
}

#[test]
fn a_carried_local_that_turns_float_beside_a_loop_link_agrees_with_the_interpreter() {
    // The shape of [`LINKED`]: the side of `j % 3 == 0` rejoins the loop
    // side, and on every hundredth outer iteration a float appears on the
    // 26th trip.
    let source = "
        local out = {0}
        local function run(n)
            local acc = 0
            for i = 1, n do
                local x = 0
                local j = 0
                while j < 40 do
                    if j % 3 == 0 then out[1] = out[1] + j else x = x + 1 end
                    if j == 25 and i % 100 == 0 then x = x + 0.5 end
                    j = j + 1
                end
                acc = acc + x
                local k = 0
                while k < 30 do
                    acc = acc + k * 2
                    k = k + 1
                end
            end
            return acc, out[1]
        end
        return run(20000)
    ";
    for (profile, stats) in run_all(source) {
        assert_trips(profile, &stats, OUTER * 71);
        assert!(stats.loop_links >= 1, "{profile:?} {stats:?}");
    }
}

#[test]
fn a_loop_side_whose_accumulator_passes_through_unchanged_compiles_and_agrees() {
    // `acc = acc + 0` folds away in the integer profiles, so the second loop
    // side hands its accumulator on unchanged: a parameter its preheader may
    // read, which is only defined at its header. Each outer iteration also
    // stores what the first loop made, so a wrong value shows per iteration.
    let source = "
        local xs = {}
        local function run(n)
            local acc = 0
            for i = 1, n do
                local x = 0
                local j = 0
                while j < 40 do
                    if i % 100 == 0 then x = x + 0.5 end
                    j = j + 1
                end
                acc = acc + x
                xs[i] = x
                for _ = 1, 30 do acc = acc + 0 end
            end
            return acc
        end
        local acc = run(3000)
        local bad = 0
        for i = 1, 3000 do
            local want = (i % 100 == 0) and 20.0 or 0
            if xs[i] ~= want or math.type and math.type(xs[i]) ~= math.type(want) then
                bad = bad + 1
            end
        end
        return acc, bad
    ";
    for (profile, stats) in run_all(source) {
        assert!(stats.native_entries >= 1, "{profile:?} {stats:?}");
    }
}
