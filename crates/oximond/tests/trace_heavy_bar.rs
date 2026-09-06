//! The productivity bar of a heavy loop body (a few inlined calls and many
//! table reads per iteration, `TRACE_MIN_ITERATIONS_HEAVY` in the trace
//! policy): one loop-back per entry. Every run is credited one loop-back,
//! also one that left before its first, so against that bar a trace whose
//! runs all leave inside an inlined call before the first loop-back used to
//! stay installed for good, and the interpreter rebuilt the calls' frames on
//! every iteration (`shape-469.lua`'s driver loop: 38000 such entries). Such
//! a run now counts no loop-back against the heavy bar, and the trace
//! retires. A run that leaves in the loop's own frame, or after a loop-back,
//! keeps its credit, and its trace stays. The results are the
//! interpreter's in every profile.

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

/// Loop trips of the programs below.
const TRIPS: u64 = 20_000;
/// The trips before the loop's path turns: the trace is recorded and runs
/// productively on these.
const SETTLE: u64 = 400;

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

/// A loop over `TRIPS` whose body calls `cell` three times; each call sums
/// fifty array slots, so the body weighs as a heavy one. `inner` is Lua
/// code inside `cell` (after its sum, with `i` in scope), and `outer` is
/// code in the loop's own frame after the calls. Either may call
/// `os.clock`, which no recording takes: a guard in front of it fails on the
/// trips that call it, and no side trace can continue that guard.
fn heavy_loop(inner: &str, outer: &str) -> String {
    let sum = (1..=50)
        .map(|slot| format!("t[{slot}]"))
        .collect::<Vec<_>>()
        .join(" + ");
    format!(
        "local a, b, c = {{}}, {{}}, {{}}
         for k = 1, 50 do a[k] = k; b[k] = 2 * k; c[k] = 3 * k end
         local clock = 0
         local function cell(t, i)
             local s = {sum}
             {inner}
             return s
         end
         local function run(n)
             local acc = 0
             for i = 1, n do
                 acc = acc + cell(a, i) + cell(b, i) + cell(c, i)
                 {outer}
             end
             return acc
         end
         return run({TRIPS}), clock"
    )
}

#[test]
fn a_heavy_body_that_leaves_inside_a_call_before_its_first_loop_back_retires() {
    // Past `SETTLE`, every trip calls `os.clock` inside the third call: each
    // run leaves at the guard in front of it, three frames deep, before its
    // first loop-back. The trace retires within a few windows instead of
    // being entered on every one of the remaining trips.
    let source = heavy_loop(
        &format!(
            "if i > {SETTLE} and t == c then clock = clock + (os.clock() >= 0 and 1 or 0) end"
        ),
        "",
    );
    for (profile, stats) in run_all(&source) {
        assert!(stats.traces_compiled >= 1, "{profile:?} {stats:?}");
        assert!(stats.iterations >= SETTLE / 2, "{profile:?} {stats:?}");
        assert!(
            stats.native_entries < (TRIPS - SETTLE) / 4,
            "{profile:?} {stats:?}"
        );
    }
}

#[test]
fn a_heavy_body_that_leaves_in_its_own_frame_stays_native() {
    // The same loop leaves past `SETTLE` in its own frame, after the three
    // calls ran natively. Each such run did the iteration's work and is
    // credited its loop-back, so the trace stays and is entered on nearly
    // every trip.
    let source = heavy_loop(
        "",
        &format!("if i > {SETTLE} then clock = clock + (os.clock() >= 0 and 1 or 0) end"),
    );
    for (profile, stats) in run_all(&source) {
        assert!(stats.traces_compiled >= 1, "{profile:?} {stats:?}");
        assert!(
            stats.native_entries > (TRIPS - SETTLE) / 2,
            "{profile:?} {stats:?}"
        );
    }
}

#[test]
fn a_heavy_body_that_leaves_inside_a_call_after_a_loop_back_stays_native() {
    // Every fifth trip leaves inside a call, at the `i % 5` guard or, once
    // a side continues that guard, at the `os.clock` guard of the side. A
    // run entered after such a trip takes four loop-backs first, is
    // credited them, and keeps its trace.
    let source = heavy_loop(
        "if i % 5 == 3 and t == c then clock = clock + (os.clock() >= 0 and 1 or 0) end",
        "",
    );
    for (profile, stats) in run_all(&source) {
        assert!(stats.traces_compiled >= 1, "{profile:?} {stats:?}");
        assert!(stats.native_entries > TRIPS / 8, "{profile:?} {stats:?}");
        assert!(stats.iterations > TRIPS / 2, "{profile:?} {stats:?}");
    }
}

#[test]
fn a_heavy_body_whose_tables_change_mid_loop_agrees() {
    // The heavy loop that leaves inside its third call, while its tables
    // change under it: a slot of `b` turns into a nil hole that a metatable
    // fills, and `a` is swapped for another table. The runs leave at the
    // guards in front of those reads, and the program finishes as the
    // interpreter does.
    let source = heavy_loop(
        &format!(
            "if i > {SETTLE} and t == c then clock = clock + (os.clock() >= 0 and 1 or 0) end"
        ),
        "if i == 300 then
             b[25] = nil
             setmetatable(b, {__index = function(_, k) return k end})
         elseif i == 350 then
             local swapped = {}
             for k = 1, 50 do swapped[k] = k + 0.5 end
             a = swapped
         end",
    );
    for (profile, stats) in run_all(&source) {
        assert!(stats.traces_compiled >= 1, "{profile:?} {stats:?}");
        assert!(stats.native_entries >= 1, "{profile:?} {stats:?}");
    }
}
