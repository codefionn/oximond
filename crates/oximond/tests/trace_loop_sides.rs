//! Loop sides: a numeric for loop beside the nest a tree carries runs as a
//! loop of its own inside the tree, instead of being unrolled into the outer
//! side that reaches it. The tree's outer side peels the loop's first trip and
//! ends in front of the second, and the loop side runs the rest in place. The
//! results are the interpreter's in every profile, whatever the loop does:
//! it runs no trips, one, two or a number that changes with every call, it
//! leaves through a guard or a `break`, it raises an error, a hook or a
//! coroutine interrupts it, or the tree goes away while it runs.

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

/// The profiles whose numbers are all floats: their traces are float traces,
/// which the tree carries through the loops around its root (integer loops
/// in the later profiles may not repay the outer side).
fn has_float_traces(profile: CompatibilityProfile) -> bool {
    matches!(
        profile,
        CompatibilityProfile::Lua51 | CompatibilityProfile::Lua52 | CompatibilityProfile::LuaJit21
    )
}

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

/// The nest of `n` bodies: a triangular pair loop whose inner bounds come from
/// the outer variable, and the update loop after it.
const NEST: &str = "
local items = {}
for i = 1, 6 do items[i] = {x = i, v = i * 0.5} end
";

/// What a traced run did, for the checks of the caller.
struct Seen {
    profile: CompatibilityProfile,
    loop_sides: u64,
    native_entries: u64,
    stats: String,
}

/// Runs `source` traced and interpreted in every profile and checks that the
/// outcomes agree. Returns what each traced run did.
fn run_all(source: &str) -> Vec<Seen> {
    let mut all = Vec::new();
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        all.push(Seen {
            profile,
            loop_sides: stats.loop_sides,
            native_entries: stats.native_entries,
            stats: format!("{stats:?}"),
        });
    }
    all
}

/// [`run_all`], and in the float profiles the tree must have installed a loop
/// side and entered native code.
fn assert_loop_side(source: &str) {
    for seen in run_all(source) {
        let Seen {
            profile,
            loop_sides,
            native_entries,
            stats,
        } = seen;
        assert!(native_entries > 0, "{profile:?} {stats}");
        if has_float_traces(profile) {
            assert!(loop_sides >= 1, "{profile:?} {stats}");
        }
    }
}

#[test]
fn the_update_loop_after_a_triangular_nest_runs_as_a_loop_side() {
    assert_loop_side(&format!(
        "{NEST}
         local function step(dt)
             for i = 1, #items do
                 local a = items[i]
                 for j = i + 1, #items do
                     local b = items[j]
                     local d = b.x - a.x
                     a.v = a.v + d * dt
                     b.v = b.v - d * dt
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
         return math.floor(total * 1000)"
    ));
}

#[test]
fn a_loop_side_runs_loops_of_no_trips_one_trip_two_trips_and_many() {
    // The update loop runs 6, 0, 1, 2, 6, 3 and 1000 trips in turn: the
    // trips the loop side was recorded for, fewer, and more than any
    // unrolled copy could hold.
    assert_loop_side(&format!(
        "{NEST}
         for i = 7, 1000 do items[i] = {{x = i, v = 0.25}} end
         local counts = {{6, 0, 1, 2, 6, 3, 1000}}
         local function step(dt, n)
             for i = 1, 6 do
                 local a = items[i]
                 for j = i + 1, 6 do
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
         for round = 1, 9000 do
             count = count + step(0.0001, counts[round % 7 + 1])
         end
         local total = 0
         for i = 1, 6 do total = total + items[i].x + items[i].v end
         return math.floor(total * 1000), count"
    ));
}

#[test]
fn a_loop_side_whose_trip_count_stays_one_value_then_changes() {
    assert_loop_side(&format!(
        "{NEST}
         local function step(dt, n)
             for i = 1, #items do
                 local a = items[i]
                 for j = i + 1, #items do
                     local b = items[j]
                     a.v = a.v + (b.x - a.x) * dt
                 end
             end
             for i = 1, n do
                 local a = items[i]
                 a.x = a.x + a.v * dt
             end
         end
         for round = 1, 12000 do step(0.001, round <= 8000 and 6 or 3) end
         local total = 0
         for i = 1, #items do total = total + items[i].x + items[i].v end
         return math.floor(total * 1000)"
    ));
}

#[test]
fn a_loop_side_leaves_through_a_break() {
    assert_loop_side(&format!(
        "{NEST}
         local function step(dt, limit)
             for i = 1, #items do
                 local a = items[i]
                 for j = i + 1, #items do
                     local b = items[j]
                     a.v = a.v + (b.x - a.x) * dt
                 end
             end
             local last = 0
             for i = 1, #items do
                 local a = items[i]
                 if a.x > limit then break end
                 a.x = a.x + a.v * dt
                 last = i
             end
             return last
         end
         local sum = 0
         for round = 1, 12000 do sum = sum + step(0.001, round % 5 == 0 and 3.2 or 1e9) end
         local total = 0
         for i = 1, #items do total = total + items[i].x + items[i].v end
         return math.floor(total * 1000), sum"
    ));
}

#[test]
fn a_loop_side_meets_a_field_that_changes_type() {
    // A string that reads as a number, then a boolean in the loop's own
    // item, then a table: each leaves the loop through its guard, and the
    // interpreter's arithmetic or error follows.
    assert_loop_side(&format!(
        "{NEST}
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
         local errors = 0
         for round = 1, 12000 do
             if round == 9000 then items[4].x = '7' end
             if round == 9500 then items[4].x = 7 end
             if round == 10000 then items[5].v = true end
             if round == 10100 then items[5].v = 0.5 end
             if round == 10500 then items[2].x = {{}} end
             if round == 10600 then items[2].x = 2 end
             if not pcall(step, 0.001) then errors = errors + 1 end
         end
         local total = 0
         for i = 1, #items do total = total + items[i].x + items[i].v end
         return math.floor(total * 1000), errors"
    ));
}

#[test]
fn an_error_in_the_loop_side_is_caught_by_the_caller() {
    assert_loop_side(&format!(
        "{NEST}
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
         local messages = {{}}
         for round = 1, 12000 do
             if round % 3000 == 0 then items[6].v = nil end
             if round % 3000 == 1 then items[6].v = 1 end
             local ok, message = pcall(step, 0.001)
             if not ok then messages[#messages + 1] = message end
         end
         local total = 0
         for i = 1, #items do total = total + items[i].x end
         return math.floor(total * 1000), table.concat(messages, '|')"
    ));
}

#[test]
fn a_hook_set_while_the_loop_side_runs_sees_what_the_interpreter_shows() {
    for kind in ["l", "c", "r"] {
        assert_loop_side(&format!(
            "{NEST}
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
             local hits = 0
             for round = 1, 12000 do
                 if round == 9000 then
                     debug.sethook(function() hits = hits + 1 end, '{kind}')
                 end
                 if round == 9010 then debug.sethook() end
                 step(0.001)
             end
             local total = 0
             for i = 1, #items do total = total + items[i].x + items[i].v end
             return math.floor(total * 1000), hits"
        ));
    }
}

#[test]
fn a_count_hook_set_mid_run_stops_after_the_budget_it_gave() {
    for count in [1, 7, 100] {
        assert_loop_side(&format!(
            "{NEST}
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
             local hits = 0
             for round = 1, 12000 do
                 if round == 9000 then
                     debug.sethook(function() hits = hits + 1 end, '', {count})
                 end
                 if round == 9020 then debug.sethook() end
                 step(0.001)
             end
             local total = 0
             for i = 1, #items do total = total + items[i].x + items[i].v end
             return math.floor(total * 1000), hits"
        ));
    }
}

#[test]
fn a_coroutine_yielding_between_calls_resumes_into_the_same_tree() {
    assert_loop_side(&format!(
        "{NEST}
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
         local worker = coroutine.wrap(function()
             for round = 1, 12000 do
                 step(0.001)
                 if round % 250 == 0 then coroutine.yield(round) end
             end
             return -1
         end)
         local last, yields = 0, 0
         while last ~= -1 do last = worker(); yields = yields + 1 end
         local total = 0
         for i = 1, #items do total = total + items[i].x + items[i].v end
         return math.floor(total * 1000), yields"
    ));
}

#[test]
fn a_coroutine_that_yields_inside_the_loop_after_the_nest_is_not_a_loop_side() {
    // The yield ends every recording of the loop: the results stay right.
    run_all(&format!(
        "{NEST}
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
                 if i == 3 and a.x > 1e9 then coroutine.yield(i) end
             end
         end
         local worker = coroutine.wrap(function()
             for round = 1, 12000 do step(0.001) end
             return -1
         end)
         local result = worker()
         local total = 0
         for i = 1, #items do total = total + items[i].x + items[i].v end
         return math.floor(total * 1000), result"
    ));
}

#[test]
fn flushing_the_jit_while_a_loop_side_is_hot_leaves_the_program_right() {
    assert_loop_side(&format!(
        "{NEST}
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
         for round = 1, 12000 do
             if round % 2500 == 0 and jit and jit.flush then jit.flush() end
             step(0.001)
         end
         local total = 0
         for i = 1, #items do total = total + items[i].x + items[i].v end
         return math.floor(total * 1000)"
    ));
}

#[test]
fn a_nest_that_calls_itself_recursively_gives_the_interpreters_answer() {
    // The recursion enters the nest again from inside the loop after it, so a
    // loop side can never be the whole story: the answers stay the same.
    run_all(&format!(
        "{NEST}
         local depth = 0
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
                 if i == 6 and depth < 2 and a.x > 5.5 then
                     depth = depth + 1
                     step(dt)
                     depth = depth - 1
                 end
             end
         end
         for _ = 1, 12000 do step(0.0005) end
         local total = 0
         for i = 1, #items do total = total + items[i].x + items[i].v end
         return math.floor(total * 1000)"
    ));
}

#[test]
fn a_loop_beside_a_nest_that_is_a_while_loop_or_a_generic_for_agrees_with_the_interpreter() {
    run_all(&format!(
        "{NEST}
         local function step(dt)
             for i = 1, #items do
                 local a = items[i]
                 for j = i + 1, #items do
                     local b = items[j]
                     a.v = a.v + (b.x - a.x) * dt
                 end
             end
             local i = 1
             while i <= #items do
                 local a = items[i]
                 a.x = a.x + a.v * dt
                 i = i + 1
             end
             for _, a in ipairs(items) do a.v = a.v * 0.9999 end
         end
         for _ = 1, 12000 do step(0.001) end
         local total = 0
         for i = 1, #items do total = total + items[i].x + items[i].v end
         return math.floor(total * 1000)"
    ));
}

#[test]
fn the_loop_side_polls_so_a_long_loop_gives_the_vm_its_turns() {
    // Hundreds of thousands of trips in one run: the loop side leaves for the
    // interpreter at its poll, which runs the collector and the hooks, and
    // goes back in.
    assert_loop_side(&format!(
        "{NEST}
         local big = {{}}
         for i = 1, 1000 do big[i] = {{w = i * 0.5}} end
         local function step(dt, n)
             for i = 1, 6 do
                 local a = items[i]
                 for j = i + 1, 6 do
                     local b = items[j]
                     a.v = a.v + (b.x - a.x) * dt
                 end
             end
             local sum = 0
             for i = 1, n do
                 sum = sum + big[i % 1000 + 1].w
             end
             return sum
         end
         local acc = 0
         for round = 1, 3000 do acc = acc + step(0.001, round % 500 == 0 and 600000 or 6) end
         local total = 0
         for i = 1, 6 do total = total + items[i].x + items[i].v end
         return math.floor(total * 1000), acc"
    ));
}

#[test]
fn a_while_loop_side_beyond_the_nest_whose_carried_local_turns_float_agrees() {
    // The `while` loop after the nest sums into a local that turns float on
    // every fiftieth call, on the third trip: the side of that branch guard
    // ends in front of a copy of the loop, which runs on with a float. The
    // nest's outer loop counts in `t`, which neither reads: the exit they end
    // in must restore it as the tree's links left it.
    for seen in run_all(&format!(
        "{NEST}
         local function step(dt, round)
             local t = 0
             for i = 1, #items do
                 local a = items[i]
                 t = t + i
                 for j = i + 1, #items do
                     local b = items[j]
                     a.v = a.v + (b.x - a.x) * dt
                 end
             end
             local s = 0
             local i = 1
             while i <= #items do
                 if i == 3 and round % 50 == 0 then s = s + 0.5 else s = s + 1 end
                 items[i].x = items[i].x + items[i].v * dt
                 i = i + 1
             end
             return s + t
         end
         local acc = 0
         for round = 1, 12000 do acc = acc + step(0.001, round) end
         local total = 0
         for i = 1, #items do total = total + items[i].x + items[i].v end
         return math.floor(total * 1000), acc"
    )) {
        let Seen {
            profile,
            native_entries,
            stats,
            ..
        } = seen;
        assert!(native_entries > 0, "{profile:?} {stats}");
    }
}

#[test]
fn a_loop_side_whose_invariant_branch_flips_leaves_with_the_state_it_started_from() {
    // `flag` is invariant in the update loop, so its guard moves into the
    // loop side's preheader (`scale` is a value select on it in front of
    // the loop). Every 97th call flips it there. The loop side
    // started from the outer side's native state (the loop's first trip
    // peeled, `i` at 2, `a` and the bounds in registers the interpreter has
    // not seen), and the exit has to restore that state: leaving the
    // registers unchanged resumed the loop with the previous call's `a` and
    // `i` (an index of nil). The same exit lost `ly_dom` in Mineclonia's
    // `region_op` (`mcl_util/shape.lua:359`).
    assert_loop_side(&format!(
        "{NEST}
         local function step(dt, flag)
             for i = 1, #items do
                 local a = items[i]
                 for j = i + 1, #items do
                     local b = items[j]
                     local d = b.x - a.x
                     a.v = a.v + d * dt
                     b.v = b.v - d * dt
                 end
             end
             local scale = flag and 2 or 3
             for i = 1, #items do
                 local a = items[i]
                 if flag then
                     a.x = a.x + a.v * dt
                 else
                     a.x = a.x - a.v * dt / scale
                 end
             end
         end
         for n = 1, 12000 do step(0.001, n % 97 ~= 0) end
         local total = 0
         for i = 1, #items do total = total + items[i].x + items[i].v end
         return math.floor(total * 1000)"
    ));
}
