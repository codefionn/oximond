//! `while` loops with a few trips inside a recording. A loop around a call
//! that runs a short `while` loop used to be turned away at the first
//! backward jump of the `while` (an inner loop), and the `while` loop became a
//! root of its own that ran a few trips per entry and was retired as
//! unproductive. The recording that unrolls short loops now follows the jump,
//! one copy of the body per trip, and the loop's test is a guard on every
//! trip: a run of another length leaves there and takes a side trace.
//! Every program runs traced and interpreted in every profile and must give
//! the same results, whatever the loop does: constant trips, trips that
//! change, a `break`, a `return` from inside the loop, a loop in a loop, a
//! loop that outgrows the unroll limit, a metatable that appears, a value
//! that changes type, an error inside the loop.

use oximond::runtime::Value;
use oximond::vm::TraceAbortReason;
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

struct Seen {
    profile: CompatibilityProfile,
    compiled: u64,
    native_entries: u64,
    iterations: u64,
    inner_loop_aborts: u64,
    ok: bool,
    stats: String,
}

/// Runs `source` traced and interpreted in every profile and checks that the
/// outcomes agree. Returns what each traced run did.
fn run_all(source: &str) -> Vec<Seen> {
    PROFILES
        .into_iter()
        .map(|profile| {
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
            let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
            assert_eq!(actual, expected, "{profile:?} {source}");
            let stats = lua.vm().jit_statistics().trace;
            Seen {
                profile,
                compiled: stats.traces_compiled,
                native_entries: stats.native_entries,
                iterations: stats.iterations,
                inner_loop_aborts: stats.recording_aborts[TraceAbortReason::InnerLoop.index()],
                ok: actual.is_ok(),
                stats: format!("{stats:?}"),
            }
        })
        .collect()
}

/// [`run_all`], and the loop around the `while` loop must have run native
/// code for most of its `outer_iterations`: the short loop is part of its
/// trace, so the iterations the trace counts are the outer loop's.
fn assert_traced(source: &str, outer_iterations: u64) {
    for seen in run_all(source) {
        let Seen {
            profile,
            compiled,
            native_entries,
            iterations,
            stats,
            ..
        } = seen;
        assert!(compiled >= 1, "{profile:?} {stats}");
        assert!(native_entries >= 1, "{profile:?} {stats}");
        assert!(
            iterations * 10 >= outer_iterations * 9,
            "{profile:?}: {iterations} of {outer_iterations} iterations ran native: {stats}"
        );
    }
}

/// [`run_all`], for a program the traced run must have run natively for part
/// of its time (a loop that outgrows the unroll limit keeps a trace of its
/// own, whose iterations are counted by the root, not by the loop around it).
fn assert_native(source: &str) {
    for seen in run_all(source) {
        let profile = seen.profile;
        assert!(seen.compiled >= 1, "{profile:?} {}", seen.stats);
        assert!(seen.native_entries >= 1, "{profile:?} {}", seen.stats);
        assert!(seen.iterations >= 1000, "{profile:?} {}", seen.stats);
    }
}

/// A `while` loop with a compound condition in a function the loop around it
/// calls, walking two counters to their limits in `steps` trips.
fn compound(steps: u32, vary: bool) -> String {
    let start = if vary { "1 + i % 3" } else { "1" };
    format!(
        "local function run(a, b, am, bm)
             local n = 0
             while a ~= am or b ~= bm do
                 n = n + a * 2 + b
                 if a ~= am then a = a + 1 end
                 if b ~= bm then b = b + 1 end
             end
             return n
         end
         local t = 0
         for i = 1, 20000 do t = t + run(1, {start}, {steps} + 1, {steps} + 1) end
         return t"
    )
}

#[test]
fn a_short_while_loop_in_a_call_is_unrolled_into_the_loop_around_it() {
    for steps in [1, 2, 3, 5] {
        assert_traced(&compound(steps, false), 20000);
    }
    // Eight trips per call pay for a trace of the loop itself, which takes
    // the loop: the results must agree.
    run_all(&compound(8, false));
}

#[test]
fn a_while_loop_whose_trips_change_takes_side_traces() {
    // Three lengths in turn: the guards of the loop's test take a side each.
    assert_traced(&compound(3, true), 20000);
}

#[test]
fn a_while_loop_past_the_unroll_limit_keeps_its_own_trace() {
    // Sixty trips per call are no short loop: the results must still agree,
    // whichever trace ends up carrying it.
    assert_native(&compound(60, false));
    assert_native(&compound(25, false));
    // The recording that met the loop gave up at the sixth trip, as an inner
    // loop, instead of unrolling twenty-five of them.
    for seen in run_all(&compound(25, false)) {
        assert!(
            seen.inner_loop_aborts >= 1,
            "{:?} {}",
            seen.profile,
            seen.stats
        );
    }
}

#[test]
fn a_while_loop_left_by_break_is_unrolled() {
    let source = "local function run(n)
             local k, t = 0, 0
             while k ~= n + 2 or k < 3 do
                 t = t + k
                 k = k + 1
                 if k == 5 then break end
             end
             return t
         end
         local total = 0
         for i = 1, 20000 do total = total + run(i % 3 + 1) end
         return total";
    assert_traced(source, 20000);
}

#[test]
fn a_while_loop_left_by_return_in_a_called_function_agrees() {
    let source = "local data = {3, 1, 4, 1}
         local function find(x)
             local i = 1
             while i <= #data and data[i] ~= nil do
                 if data[i] == x then return i end
                 i = i + 1
             end
             return 0
         end
         local t = 0
         for i = 1, 20000 do t = t + find(i % 5) end
         return t";
    // Five different answers, each a different trip count and exit: every
    // one is a guard of the unrolled loop and must give the interpreter's
    // result. The loop around the call runs natively.
    assert_traced(source, 20000);
}

#[test]
fn nested_while_loops_in_a_numeric_for_run_natively() {
    let source = "local function run(n, m)
             local t, i = 0, 0
             while i ~= n or t < 0 do
                 local j = 0
                 while j ~= m do
                     t = t + i * j
                     j = j + 1
                 end
                 i = i + 1
             end
             return t
         end
         local total = 0
         for k = 1, 20000 do total = total + run(2, 3) end
         return total";
    assert_traced(source, 20000);
}

#[test]
fn a_generic_for_inside_a_while_inside_a_numeric_for_agrees() {
    let source = "local items = {1, 2, 3}
         local t = 0
         for i = 1, 20000 do
             local j = 0
             while j < i % 4 do
                 for _, v in ipairs(items) do t = t + v * j end
                 j = j + 1
             end
         end
         return t";
    // Not native yet (the generic for inside the `while` is no loop a
    // recording can unroll and the `while` has too few trips to be a root):
    // the results must agree.
    run_all(source);
}

#[test]
fn a_short_ipairs_loop_inside_a_numeric_for_agrees() {
    // Not native yet: the step that ends an `ipairs` loop is not a guard a
    // recording can follow, so a three-element loop is no inner loop that
    // can be unrolled and its own root runs too few trips per entry. The
    // results must agree meanwhile.
    let source = "local items = {1, 2, 3}
         local t = 0
         for i = 1, 20000 do
             for _, v in ipairs(items) do t = t + v * (i % 3) end
         end
         return t";
    run_all(source);
}

/// The loop around the call ran natively until the program changed under it
/// at iteration `at` of 20000: the results (or the error) are the
/// interpreter's, and `before` iterations at least ran native code first.
fn assert_leaves_native(source: &str, before: u64, ok: bool) {
    for seen in run_all(source) {
        let profile = seen.profile;
        assert_eq!(seen.ok, ok, "{profile:?} {}", seen.stats);
        assert!(seen.compiled >= 1, "{profile:?} {}", seen.stats);
        assert!(seen.native_entries >= 1, "{profile:?} {}", seen.stats);
        assert!(
            seen.iterations >= before,
            "{profile:?}: {} iterations before the change: {}",
            seen.iterations,
            seen.stats
        );
    }
}

#[test]
fn a_metatable_that_appears_inside_the_loop_leaves_the_trace() {
    // The array the trace reads from is swapped for an empty table whose
    // `__index` answers: the array guard fails and the interpreter runs the
    // metamethod. The sum is a number, not an error.
    let source = "local cells = {1, 2, 3, 4}
         local function run(i)
             local k, t = 1, 0
             while k ~= 4 or i < 0 do
                 t = t + cells[k]
                 k = k + 1
             end
             return t
         end
         local total = 0
         for i = 1, 20000 do
             if i == 15000 then
                 cells = setmetatable({}, { __index = function(_, k) return k * 100 end })
             end
             total = total + run(i)
         end
         return total";
    assert_leaves_native(source, 10000, true);
    for profile in PROFILES {
        let (result, _) = outcome(profile, ExecutionMode::Tracing, source);
        // 15000 calls sum 1 + 2 + 3, the rest 100 + 200 + 300.
        let expected = 14999.0 * 6.0 + 5001.0 * 600.0;
        match result.unwrap().first() {
            Some(Value::Number(n)) => assert_eq!(*n, expected, "{profile:?}"),
            Some(Value::Integer(n)) => assert_eq!(*n as f64, expected, "{profile:?}"),
            other => panic!("{profile:?}: {other:?}"),
        }
    }
}

#[test]
fn a_value_that_changes_type_inside_the_loop_leaves_the_trace() {
    let source = "local function run(x)
             local k, t = 0, 0
             while k ~= 3 or x == nil do
                 t = t + x
                 k = k + 1
             end
             return t
         end
         local total = 0
         for i = 1, 20000 do
             total = total + run(i == 12000 and \"7\" or i)
         end
         return total";
    assert_leaves_native(source, 10000, true);
}

#[test]
fn an_error_inside_the_loop_is_the_interpreters() {
    let source = "local function run(x)
             local k, t = 0, 0
             while k ~= 3 or x == nil do
                 t = t + x
                 k = k + 1
             end
             return t
         end
         local total = 0
         for i = 1, 20000 do
             total = total + run(i == 15000 and {} or i)
         end
         return total";
    assert_leaves_native(source, 10000, false);
}

/// A nest in the root frame: a `while` loop with a compound condition around
/// a `while` loop that runs one to three trips. The inner loop is a root of
/// its own first, runs too few trips per entry and is retired; the loop
/// around it, turned away for meeting it, records again and unrolls it.
fn root_nest(outer: u32) -> String {
    format!(
        "local t, i = 0, 0
         local limit = {outer}
         while i ~= limit or t < 0 do
             local j, m = 0, 1 + i % 3
             while j ~= m or t < 0 do
                 t = t + i * j + 1
                 j = j + 1
             end
             i = i + 1
         end
         return t"
    )
}

#[test]
fn a_short_while_in_the_root_frame_is_unrolled_once_its_root_is_retired() {
    assert_traced(&root_nest(30000), 30000);
}

#[test]
fn a_numeric_loop_in_a_while_loop_in_the_root_frame_runs_natively() {
    let source = "local t, i = 0, 0
         while i ~= 20000 or t < 0 do
             for j = 1, 1 + i % 3 do t = t + j * i end
             i = i + 1
         end
         return t";
    assert_traced(source, 20000);
}

#[test]
fn a_short_loop_entered_many_times_in_the_root_frame_agrees_and_runs_natively() {
    // Two trips per entry, thirty thousand entries.
    let source = "local t = 0
         for i = 1, 30000 do
             local j = 0
             while j < 2 do t = t + i + j; j = j + 1 end
         end
         return t";
    assert_traced(source, 30000);
}

#[test]
fn a_root_frame_loop_left_by_break_keeps_running_natively() {
    let source = "local t = 0
         for i = 1, 30000 do
             local j = 0
             while j < 10 or t < 0 do
                 t = t + i + j
                 j = j + 1
                 if j > i % 3 then break end
             end
         end
         return t";
    // The loop around it is native but for the iterations that record and
    // compile the variants of the inner loop's length.
    assert_traced(source, 25000);
}

#[test]
fn a_root_frame_loop_left_by_return_in_a_function_agrees() {
    let source = "local function first(i)
             local j = 0
             while j < 6 or i < 0 do
                 if j * j > i % 7 then return j end
                 j = j + 1
             end
             return -1
         end
         local t = 0
         for i = 1, 30000 do t = t + first(i) end
         return t";
    assert_traced(source, 30000);
}
