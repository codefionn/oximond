//! Short loops inside a recording. A numeric for loop with a few trips is not
//! a trace of its own: the loop around it records the nest with the short
//! loops unrolled, one copy of the body per trip, and the latch of each copy
//! is a guard. A run of another length leaves there and takes a side trace.
//! Every program below runs traced and interpreted in every profile and must
//! give the same results, whatever the trips do: none, one, as many as the
//! unroll limit allows, one more, or a number that changes between runs, with
//! a table that grows and shrinks under the bound, a `__len` metamethod, a
//! `break`, a loop inside a loop, a loop beside another.

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

/// What a traced run did.
struct Seen {
    profile: CompatibilityProfile,
    compiled: u64,
    native_entries: u64,
    iterations: u64,
    outer_sides: u64,
    recording_starts: u64,
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
                outer_sides: stats.outer_sides,
                recording_starts: stats.recording_starts,
                stats: format!("{stats:?}"),
            }
        })
        .collect()
}

/// [`run_all`], and the loop around the short loops must have run native
/// code for most of its iterations: one trace of the nest, entered a few
/// times, with the short loops inside it.
fn assert_nest_traced(source: &str, outer_iterations: u64) {
    for seen in run_all(source) {
        let Seen {
            profile,
            compiled,
            native_entries,
            iterations,
            outer_sides,
            stats,
            ..
        } = seen;
        assert!(compiled >= 1, "{profile:?} {stats}");
        assert!(native_entries >= 1, "{profile:?} {stats}");
        assert!(
            iterations * 10 >= outer_iterations * 9,
            "{profile:?}: {iterations} of {outer_iterations} iterations ran native: {stats}"
        );
        assert_eq!(outer_sides, 0, "{profile:?} {stats}");
    }
}

/// `for outer = 1, N do for i = 1, <bound> do ... end end` with the inner
/// loop `trips` long, summing a table.
fn constant_trips(trips: u32) -> String {
    format!(
        "local cells = {{}}
         for i = 1, 40 do cells[i] = i * 1.5 end
         local total = 0
         for outer = 1, 30000 do
             local sum = 0
             for i = 1, {trips} do sum = sum + cells[i] * outer end
             total = (total + sum) % 1000003
         end
         return total"
    )
}

#[test]
fn a_loop_of_a_constant_number_of_trips_is_unrolled_into_the_loop_around_it() {
    // Zero trips run no body at all, one is a single copy, eight is the
    // trips a short loop may have to give way (see
    // `policy::PREFER_UNROLL_TRIPS`), nine and fifteen are past it and
    // sixteen past the unroll limit: those keep a tree of the inner loop.
    for trips in [0, 1, 2, 3, 5, 8] {
        let source = constant_trips(trips);
        if trips == 0 {
            // No trace has a loop to run, but the result must agree.
            run_all(&source);
        } else {
            assert_nest_traced(&source, 30000);
        }
    }
    for trips in [9, 15, 16, 17, 40] {
        run_all(&constant_trips(trips));
    }
}

#[test]
fn a_loop_of_a_few_trips_that_changes_them_keeps_its_own_tree() {
    // Three to six trips in turn: a side trace for each value would compile
    // one after the other, so the inner loop is the root and the outer loop
    // runs in its tree (an outer side). The results agree all the same.
    for seen in run_all(
        "local cells = {}
         for i = 1, 6 do cells[i] = i * 3 + 1 end
         local total = 0
         for outer = 1, 60000 do
             local n = 3 + outer % 4
             local sum = 0
             for i = 1, n do sum = sum + cells[i] * outer end
             total = (total + sum) % 1000003
         end
         return total",
    ) {
        let Seen {
            profile,
            compiled,
            outer_sides,
            stats,
            ..
        } = seen;
        assert!(compiled >= 1, "{profile:?} {stats}");
        assert!(outer_sides >= 1, "{profile:?} {stats}");
    }
}

#[test]
fn an_outer_loop_that_cannot_record_gives_the_short_loop_back() {
    // The loop around makes a closure each iteration, which no recording
    // follows, so the nest cannot be unrolled. The short loop gave way to it
    // once; when that recording fails it records for itself again and the
    // run goes on as before, with the outer loop turned away for good.
    for seen in run_all(
        "local cells = {}
         for i = 1, 6 do cells[i] = i * 1.5 end
         local total = 0
         for outer = 1, 60000 do
             local unused = function() return outer end
             local sum = 0
             for i = 1, 4 do sum = sum + cells[i] * outer end
             total = (total + sum) % 1000003
         end
         return total",
    ) {
        let Seen {
            profile,
            recording_starts,
            stats,
            ..
        } = seen;
        // A few recordings in all: the short loop's two closings, the outer
        // loop's attempts, the short loop's own.
        assert!(recording_starts <= 16, "{profile:?} {stats}");
    }
}

/// Two loops side by side inside the loop around: neither can be the root of
/// a tree that reaches the other, so the loop around records, with `trips`
/// and three trips unrolled.
fn siblings(trips: u32) -> String {
    format!(
        "local cells = {{}}
         for i = 1, 40 do cells[i] = i * 1.5 end
         local total = 0
         for outer = 1, 30000 do
             local sum = 0
             for i = 1, {trips} do sum = sum + cells[i] * outer end
             for i = 1, 3 do sum = sum - cells[i] end
             total = (total + sum) % 1000003
         end
         return total"
    )
}

#[test]
fn the_most_trips_a_recording_unrolls_and_one_more() {
    // A loop is unrolled while it has at most 15 trips left at its first
    // latch (`recorder::MAX_UNROLLED_LOOP_TRIPS`): sixteen trips are, seventeen
    // are not. The loop of seventeen is a trace of its own then, and the
    // outer loop runs on through the side at its exit.
    for trips in [4, 14, 15, 16] {
        assert_nest_traced(&siblings(trips), 30000);
    }
    for seen in run_all(&siblings(17)) {
        let Seen {
            profile,
            native_entries,
            iterations,
            stats,
            ..
        } = seen;
        // The loop's own trips are the trace's iterations: about sixteen
        // for each outer iteration, where a nest in one trace has one.
        assert!(iterations >= 400_000, "{profile:?} {stats}");
        // The side at its exit meets the loop of three, is recorded again
        // with that loop unrolled and runs on into the next outer iteration,
        // so native code is not left once per outer iteration (30000
        // entries when that side was given up).
        assert!(native_entries <= 1_000, "{profile:?} {stats}");
    }
}

#[test]
fn a_branching_callee_in_front_of_a_short_loop_is_unrolled_with_it() {
    // The callee alternates between its arms and the loop after it runs
    // eight trips. The loop around records the nest with the call inlined
    // and the loop unrolled, and the arm it did not record takes a side.
    for seen in run_all(
        "local function bump(x)
             if x % 2 == 0 then return x + 1 else return x + 2 end
         end
         local total = 0
         for outer = 1, 60000 do
             local value = bump(outer)
             for inner = 1, 8 do total = total + (value + inner) % 31 end
         end
         return total",
    ) {
        let Seen {
            profile,
            native_entries,
            iterations,
            outer_sides,
            stats,
            ..
        } = seen;
        assert!(iterations > 55000, "{profile:?} {stats}");
        assert!(native_entries < 100, "{profile:?} {stats}");
        assert_eq!(outer_sides, 0, "{profile:?} {stats}");
    }
}

#[test]
fn a_loop_bounded_by_the_length_of_a_table() {
    assert_nest_traced(
        "local t = {}
         for i = 1, 5 do t[i] = i * 1.5 end
         local total = 0
         for outer = 1, 30000 do
             local s = 0
             for i = 1, #t do s = s + t[i] * outer end
             total = (total + s) % 1000003
         end
         return total",
        30000,
    );
}

#[test]
fn the_length_of_the_table_changes_under_the_bound() {
    // The table grows and shrinks while the loop runs: a longer table leaves
    // at the last unrolled latch, a shorter at an earlier one. Past the
    // limit of unrolled trips the loop does not unroll at all.
    run_all(
        "local t = {}
         for i = 1, 4 do t[i] = i * 1.5 end
         local total = 0
         for outer = 1, 30000 do
             if outer == 9000 then t[5] = 7.5 end
             if outer == 12000 then t[6] = 2.25; t[7] = 3.5 end
             if outer == 15000 then t[7] = nil; t[6] = nil end
             if outer == 20000 then t[5] = nil; t[4] = nil end
             if outer == 25000 then for i = 4, 30 do t[i] = i end end
             local s = 0
             for i = 1, #t do s = s + t[i] * outer end
             total = (total + s) % 1000003
         end
         return total, #t",
    );
}

#[test]
fn table_length_through_insert_and_remove() {
    run_all(
        "local t = {1.5, 2.5, 3.5}
         local total = 0
         for outer = 1, 30000 do
             if outer % 5000 == 0 then table.insert(t, outer * 0.5) end
             if outer % 7500 == 0 then table.remove(t) end
             local s = 0
             for i = 1, #t do s = s + t[i] end
             total = (total + s * outer) % 1000003
         end
         return total, #t",
    );
}

#[test]
fn a_len_metamethod_decides_the_bound_in_the_profiles_that_honour_it() {
    // Lua 5.2 and later call `__len` for tables; 5.1 and LuaJIT ignore it
    // for tables. The loop's bound is whatever `#t` is for the profile.
    run_all(
        "local t = setmetatable({}, {__len = function() return 4 end})
         for i = 1, 6 do rawset(t, i, i * 2.5) end
         local total = 0
         for outer = 1, 20000 do
             local s = 0
             for i = 1, #t do s = s + t[i] * outer end
             total = (total + s) % 1000003
         end
         return total",
    );
}

#[test]
fn a_len_metamethod_whose_answer_changes() {
    run_all(
        "local n = 3
         local t = setmetatable({}, {__len = function() return n end})
         for i = 1, 8 do rawset(t, i, i * 2.5) end
         local total = 0
         for outer = 1, 20000 do
             if outer == 6000 then n = 5 end
             if outer == 12000 then n = 2 end
             local s = 0
             for i = 1, #t do s = s + t[i] * outer end
             total = (total + s) % 1000003
         end
         return total",
    );
}

#[test]
fn trips_that_change_between_runs() {
    // The bound is a variable that takes several values in turn, some of
    // them zero, one past the limit, and many.
    run_all(
        "local cells = {}
         for i = 1, 40 do cells[i] = i * 1.5 end
         local counts = {3, 3, 3, 0, 1, 3, 16, 3, 2, 3, 3, 30, 3}
         local total = 0
         for outer = 1, 30000 do
             local n = counts[outer // 700 % #counts + 1]
             local s = 0
             for i = 1, n do s = s + cells[i] * outer end
             total = (total + s) % 1000003
         end
         return total",
    );
}

#[test]
fn trips_that_cycle_with_the_outer_counter() {
    // The bound depends on the counter of the loop around: every run of a
    // different length, a few lengths in turn, never the one recorded.
    run_all(
        "local cells = {}
         for i = 1, 20 do cells[i] = i * 1.5 end
         local total = 0
         for outer = 1, 40000 do
             local s = 0
             for i = 1, outer % 4 do s = s + cells[i] * outer end
             total = (total + s) % 1000003
         end
         return total",
    );
    run_all(
        "local cells = {}
         for i = 1, 20 do cells[i] = i * 1.5 end
         local total = 0
         for outer = 1, 40000 do
             local s = 0
             for i = 1, 3 + outer % 4 do s = s + cells[i] * outer end
             total = (total + s) % 1000003
         end
         return total",
    );
}

#[test]
fn a_run_of_zero_trips_in_a_triangular_nest() {
    // `for j = i + 1, n` runs 0 trips for the last `i`: the prep of the
    // loop is a guard, and the iteration skips the latch.
    run_all(
        "local items = {}
         for i = 1, 5 do items[i] = {x = i, v = i * 0.5} end
         local function advance(dt)
             for i = 1, #items do
                 local a = items[i]
                 for j = i + 1, #items do
                     local b = items[j]
                     local d = b.x - a.x
                     a.v = a.v + d * dt
                     b.v = b.v - d * dt
                 end
             end
         end
         for _ = 1, 20000 do advance(0.0001) end
         local total = 0
         for i = 1, #items do total = total + items[i].v end
         return math.floor(total * 1000)",
    );
}

#[test]
fn a_break_inside_the_unrolled_loop() {
    assert_nest_traced(
        "local cells = {}
         for i = 1, 10 do cells[i] = i end
         local total = 0
         for outer = 1, 30000 do
             local s = 0
             for i = 1, 6 do
                 if cells[i] > 4 then break end
                 s = s + cells[i] * outer
             end
             total = (total + s) % 1000003
         end
         return total",
        30000,
    );
}

#[test]
fn a_break_that_fires_on_other_trips_later() {
    run_all(
        "local cells = {}
         for i = 1, 10 do cells[i] = i end
         local total = 0
         for outer = 1, 30000 do
             if outer == 10000 then cells[3] = 99 end
             if outer == 20000 then cells[3] = 3; cells[5] = 99 end
             local s = 0
             for i = 1, 6 do
                 if cells[i] > 4 then break end
                 s = s + cells[i] * outer
             end
             total = (total + s) % 1000003
         end
         return total",
    );
}

#[test]
fn short_loops_inside_short_loops() {
    // 3 x 5 trips unrolled in the loop around, and 2 x 2 x 2.
    assert_nest_traced(
        "local t = {}
         for i = 1, 5 do t[i] = i * 1.5 end
         local total = 0
         for outer = 1, 30000 do
             local s = 0
             for i = 1, 3 do
                 for j = 1, #t do s = s + t[j] * i end
             end
             total = (total + s + outer) % 1000003
         end
         return total",
        30000,
    );
    assert_nest_traced(
        "local t = {}
         for i = 1, 6 do t[i] = i * 0.5 end
         local total = 0
         for outer = 1, 30000 do
             local s = 0
             for a = 1, 2 do
                 for b = 1, 2 do
                     for c = 1, 2 do s = s + t[a + b + c - 2] end
                 end
             end
             total = (total + s * outer) % 1000003
         end
         return total",
        30000,
    );
}

#[test]
fn short_loops_side_by_side() {
    // Two loops beside each other inside the loop around: neither can be
    // the root of a tree that reaches the other.
    assert_nest_traced(
        "local t = {}
         for i = 1, 6 do t[i] = i * 1.5 end
         local total = 0
         for outer = 1, 30000 do
             local s = 0
             for i = 1, outer % 2 do s = s + t[i] * outer end
             for i = 1, #t do s = s + t[i] end
             total = (total + s) % 1000003
         end
         return total",
        30000,
    );
}

#[test]
fn short_loops_side_by_side_with_trips_that_change() {
    run_all(
        "local t = {}
         for i = 1, 12 do t[i] = i * 1.5 end
         local total = 0
         for outer = 1, 40000 do
             local s = 0
             for i = 1, outer % 3 do s = s + t[i] * outer end
             for i = 1, 2 + outer // 10000 do s = s + t[i] end
             total = (total + s) % 1000003
         end
         return total",
    );
}

#[test]
fn an_error_inside_an_unrolled_loop_is_caught() {
    run_all(
        "local cells = {}
         for i = 1, 6 do cells[i] = i * 1.5 end
         local total, caught = 0, 0
         for outer = 1, 30000 do
             if outer == 21000 then cells[4] = 'x' end
             if outer == 22000 then cells[4] = 4 end
             local ok, s = pcall(function()
                 local sum = 0
                 for i = 1, 5 do sum = sum + cells[i] * outer end
                 return sum
             end)
             if ok then total = (total + s) % 1000003 else caught = caught + 1 end
         end
         return total, caught",
    );
}

#[test]
fn a_hook_fires_on_every_line_of_an_unrolled_loop() {
    run_all(
        "local cells = {}
         for i = 1, 6 do cells[i] = i * 1.5 end
         local total, lines = 0, 0
         for outer = 1, 20000 do
             if outer == 12000 then
                 debug.sethook(function() lines = lines + 1 end, 'l')
             end
             if outer == 12100 then debug.sethook() end
             local s = 0
             for i = 1, 4 do s = s + cells[i] * outer end
             total = (total + s) % 1000003
         end
         return total, lines",
    );
}

#[test]
fn a_loop_inside_a_function_called_from_a_loop() {
    run_all(
        "local cells = {}
         for i = 1, 6 do cells[i] = i * 1.5 end
         local function f(n, k)
             local s = 0
             for i = 1, n do s = s + cells[i] * k end
             return s
         end
         local total = 0
         for outer = 1, 30000 do
             total = (total + f(4, outer) + f(outer % 4, outer)) % 1000003
         end
         return total",
    );
}

#[test]
fn flushing_the_traces_while_the_nest_runs() {
    run_all(
        "local t = {}
         for i = 1, 5 do t[i] = i * 1.5 end
         local total = 0
         for outer = 1, 30000 do
             if outer % 8000 == 0 and jit and jit.flush then jit.flush() end
             local s = 0
             for i = 1, #t do s = s + t[i] * outer end
             total = (total + s) % 1000003
         end
         return total",
    );
}

#[test]
fn a_coroutine_yields_from_inside_the_nest() {
    run_all(
        "local t = {}
         for i = 1, 4 do t[i] = i * 1.5 end
         local co = coroutine.wrap(function()
             local total = 0
             for outer = 1, 20000 do
                 local s = 0
                 for i = 1, #t do s = s + t[i] * outer end
                 total = (total + s) % 1000003
                 if outer % 4000 == 0 then coroutine.yield(total) end
             end
             return total
         end)
         local last
         for _ = 1, 6 do last = co() end
         return last",
    );
}

#[test]
fn a_float_loop_with_a_fractional_step() {
    run_all(
        "local total = 0
         for outer = 1, 30000 do
             local s = 0
             for x = 0, 1, 0.25 do s = s + x * outer end
             total = (total + s) % 1000003
         end
         return total",
    );
}

#[test]
fn a_counting_down_loop() {
    run_all(
        "local t = {}
         for i = 1, 6 do t[i] = i * 1.5 end
         local total = 0
         for outer = 1, 30000 do
             local s = 0
             for i = 5, 1, -1 do s = s + t[i] * outer end
             total = (total + s) % 1000003
         end
         return total",
    );
}

/// `body` (a program that ends in `return`) run as the body of a function
/// the program calls: the frame has a closure, so it is not a top-level
/// chunk, and its loops' counters are only known from its registers.
fn in_function(body: &str) -> String {
    format!("local function run(n)\n{body}\nend\nreturn run(1)")
}

#[test]
fn short_loops_in_a_function_are_unrolled_into_the_loop_around_them() {
    // Constant trips, a table length, a break and nested short loops.
    for trips in [1, 3, 5, 8] {
        assert_nest_traced(&in_function(&constant_trips(trips)), 30000);
    }
    assert_nest_traced(
        &in_function(
            "local t = {}
             for i = 1, 5 do t[i] = i * 1.5 end
             local total = 0
             for outer = 1, 30000 do
                 local s = 0
                 for i = 1, #t do s = s + t[i] * outer end
                 total = (total + s) % 1000003
             end
             return total",
        ),
        30000,
    );
    assert_nest_traced(
        &in_function(
            "local cells = {}
             for i = 1, 10 do cells[i] = i end
             local total = 0
             for outer = 1, 30000 do
                 local s = 0
                 for i = 1, 6 do
                     if cells[i] > 4 then break end
                     s = s + cells[i] * outer
                 end
                 total = (total + s) % 1000003
             end
             return total",
        ),
        30000,
    );
    assert_nest_traced(
        &in_function(
            "local t = {}
             for i = 1, 5 do t[i] = i * 1.5 end
             local total = 0
             for outer = 1, 30000 do
                 local s = 0
                 for i = 1, 3 do
                     for j = 1, #t do s = s + t[j] * i end
                 end
                 total = (total + s + outer) % 1000003
             end
             return total",
        ),
        30000,
    );
}

#[test]
fn a_function_with_a_short_loop_called_again_with_other_bounds() {
    run_all(
        "local cells = {}
         for i = 1, 20 do cells[i] = i * 0.5 end
         local function run(outer, trips)
             local total = 0
             for o = 1, outer do
                 local s = 0
                 for i = 1, trips do s = s + cells[i] * o end
                 total = (total + s) % 1000003
             end
             return total
         end
         return run(20000, 4) + run(20000, 0) + run(20000, 1) + run(20000, 16)
             + run(20000, 4) + run(3, 4)",
    );
}

/// nbody's shape: a function with a nest and then a 5-trip loop over a table
/// of records, called from a hot loop. The 5-trip loop has no loop around it
/// in its function, so on its own it is a standalone root that loses to the
/// interpreter (it is rejected as a short loop). Once the tree has run long
/// enough, the side that returns from the function runs on through the
/// 5-trip loop and into the caller, so no iteration of it is interpreted.
/// In the integer profiles the side that starts at the zero-trip arm of the
/// last pair loop used to be recorded at once, meet the 5-trip loop as an
/// inner loop (a side unrolls only when its root met inner loops) or run to
/// the function's return with nowhere to go, and burn all its attempts.
#[test]
fn a_short_loop_after_a_nest_runs_native_once_the_tree_links_through_the_return() {
    let source = "
        local bodies = {}
        for k = 1, 5 do bodies[k] = {x = k, v = k * 0.5, m = 1 / k} end
        local function advance(dt)
            for i = 1, #bodies do
                local a = bodies[i]
                for j = i + 1, #bodies do
                    local b = bodies[j]
                    local d = a.x - b.x
                    local f = dt * b.m / (d * d + 1)
                    a.v = a.v - d * f
                    b.v = b.v + d * f
                end
            end
            for i = 1, #bodies do
                local body = bodies[i]
                body.x = body.x + dt * body.v
            end
        end
        for _ = 1, 40000 do advance(0.01) end
        local total = 0
        for i = 1, #bodies do total = total + bodies[i].x + bodies[i].v end
        return string.format('%.9f', total)
    ";
    for seen in run_all(source) {
        // The tree is entered once per call until it links, and a handful of
        // times after: far fewer entries than the 40000 calls.
        assert!(
            seen.native_entries < 40_000 / 2,
            "{:?}: {} entries\n{}",
            seen.profile,
            seen.native_entries,
            seen.stats
        );
        assert!(seen.compiled >= 1, "{:?}\n{}", seen.profile, seen.stats);
    }
}
