//! `pairs`, `next` and `ipairs` in a trace: a loop over a table compiles, the
//! trace steps the traversal with a `table.next` helper (or the
//! `table.next_box` one, which keeps the key in a lane), and the results,
//! the order of the keys and the errors are the interpreter's in every
//! profile.
//!
//! The loops run natively for nearly every iteration (a floor on the trace
//! statistics' `iterations`). A step the trace did not see (a handler in
//! `__pairs`, a replaced `pairs` or `next`, an invalid continuation key, a
//! value that is no table) leaves the trace before the call and the
//! interpreter makes it, raising the interpreter's error.

use oximond::runtime::Value;
use oximond::vm::{TraceAbortReason, TraceStatistics};
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

/// The outcome as text, which tells an integer from a float.
fn show(result: &Result<Vec<Value>, String>) -> String {
    format!("{result:?}")
}

/// Whether a trace calls a step of `next` (the keyed form or the one that
/// keeps its control value in a lane).
fn next_helper_called(lua: &Lua) -> bool {
    lua.vm()
        .trace_helper_calls()
        .iter()
        .any(|name| name.starts_with("table.next"))
}

fn no_native_aborts(profile: CompatibilityProfile, source: &str, stats: &TraceStatistics) {
    for name in ["next", "pairs", "ipairs"] {
        assert_eq!(
            stats.native_aborts.of(name),
            0,
            "{profile:?} {source}: {:?}",
            stats.native_aborts
        );
    }
}

/// Runs `source` in both modes of each profile and checks that the results
/// agree, that at least `floor` iterations ran inside native code, that the
/// `next` helper is installed in a trace, and that no recording gave up on
/// `next`, `pairs` or `ipairs`.
fn assert_loop(source: &str, floor: u64) {
    assert_running(source, floor, Some(true));
}

/// [`assert_loop`] for a traversal of an array part that no loop encloses,
/// which the trace reads in place: the `next` helper is not called.
fn assert_inline(source: &str, floor: u64) {
    assert_running(source, floor, Some(false));
}

/// [`assert_loop`] for a traversal that reads the array part in place where
/// it can and calls the helper past a hole: either may run.
fn assert_either(source: &str, floor: u64) {
    assert_running(source, floor, None);
}

fn assert_running(source: &str, floor: u64, helper: impl Into<Option<bool>>) {
    let helper = helper.into();
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.traces_compiled > 0, "{profile:?} {source} {stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert!(
            stats.iterations >= floor,
            "{profile:?} {source}: {} iterations, wanted {floor}: {stats:?}",
            stats.iterations
        );
        if let Some(helper) = helper {
            assert_eq!(
                next_helper_called(&lua),
                helper,
                "{profile:?} {source}: {:?}",
                lua.vm().trace_helper_calls()
            );
        }
        no_native_aborts(profile, source, &stats);
    }
}

/// Checks only that the two modes agree, success or error alike.
fn assert_agrees(source: &str) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(
            show(&actual),
            show(&expected),
            "{profile:?} {source} {:?}",
            lua.vm().jit_statistics().trace
        );
    }
}

/// [`assert_agrees`] for a loop that must also have run natively.
fn assert_agrees_native(source: &str) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
    }
}

/// [`assert_agrees_native`] with a floor on the iterations that ran inside
/// native code, so a loop that quietly stays interpreted or leaves the trace
/// on nearly every step fails.
fn assert_agrees_floor(source: &str, floor: u64) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert!(
            stats.iterations >= floor,
            "{profile:?} {source}: {} iterations, wanted {floor}: {stats:?}",
            stats.iterations
        );
    }
}

const ROUNDS: u64 = 400;

/// `rounds` traversals of the table `t` that `setup` builds, with `body`
/// run per entry; `n` sums, `h` hashes the order of the values and `c`
/// counts.
fn traversal(setup: &str, iterator: &str, body: &str, rounds: u64) -> String {
    format!(
        "local n, h, c = 0, 0, 0 {setup} \
         for r = 1, {rounds} do \
           for k, v in {iterator} do {body} h = (h * 31 + v) % 1000003 c = c + 1 end \
         end \
         return n, h, c"
    )
}

fn pairs_of(setup: &str, body: &str, entries: u64) -> (String, u64) {
    (
        traversal(setup, "pairs(t)", body, ROUNDS),
        ROUNDS * entries * 9 / 10,
    )
}

fn next_of(setup: &str, body: &str, entries: u64) -> (String, u64) {
    (
        traversal(setup, "next, t", body, ROUNDS),
        ROUNDS * entries * 9 / 10,
    )
}

const ARRAY: &str = "local t = {} for i = 1, 100 do t[i] = i * 2 end";
const STRINGS: &str = "local t = {} for i = 1, 100 do t['key' .. i] = i end";
const SPARSE: &str = "local t = {} for i = 1, 100 do t[i * 1000] = i end";
const FLOATS: &str = "local t = {} for i = 1, 100 do t[i + 0.5] = i end";
const BOOLEANS: &str = "local t = {[true] = 3, [false] = 5}";
const TABLES: &str = "local t = {} for i = 1, 100 do t[{}] = i end";
const HOLES: &str =
    "local t = {} for i = 1, 100 do t[i] = i end for i = 1, 100, 3 do t[i] = nil end";
const MIXED: &str =
    "local t = {} for i = 1, 60 do t[i] = i end for i = 1, 60 do t['k' .. i] = i + 100 end";

#[test]
fn a_loop_no_other_loop_encloses_reads_an_array_part_in_place() {
    let source = "local n, t = 0, {} for i = 1, 20000 do t[i] = i end \
                  for k, v in pairs(t) do n = n + k + v end return n";
    assert_inline(source, 15000);
    let source = "local n, t = 0, {} for i = 1, 20000 do t[i] = i end \
                  for k, v in next, t do n = n + k + v end return n";
    assert_inline(source, 15000);
}

#[test]
fn pairs_over_an_array_part() {
    let (source, floor) = pairs_of(ARRAY, "n = n + k + v", 100);
    assert_inline(&source, floor);
    let (source, floor) = pairs_of(ARRAY, "n = n + v", 100);
    assert_inline(&source, floor);
}

#[test]
fn pairs_over_string_keys() {
    let (source, floor) = pairs_of(STRINGS, "n = n + v", 100);
    assert_loop(&source, floor);
}

#[test]
fn pairs_over_sparse_integer_keys() {
    let (source, floor) = pairs_of(SPARSE, "n = n + k + v", 100);
    assert_loop(&source, floor);
}

#[test]
fn pairs_over_float_keys() {
    let (source, floor) = pairs_of(FLOATS, "n = n + k * 2 + v", 100);
    assert_loop(&source, floor);
}

#[test]
fn pairs_over_boolean_keys() {
    // Two entries a round, and the loop ends every other step.
    let (source, floor) = pairs_of(BOOLEANS, "if k then n = n + v else n = n - v end", 2);
    assert_loop(&source, floor / 2);
}

#[test]
fn pairs_over_table_keys() {
    let (source, floor) = pairs_of(TABLES, "n = n + v", 100);
    assert_loop(&source, floor);
}

#[test]
fn pairs_over_an_array_with_holes() {
    let (source, floor) = pairs_of(HOLES, "n = n + k + v", 66);
    assert_either(&source, floor / 2);
}

#[test]
fn pairs_over_the_array_and_the_hash_part() {
    // The keys change type half way, and the loop takes both.
    let (source, floor) = pairs_of(MIXED, "n = n + v", 120);
    assert_loop(&source, floor);
}

#[test]
fn next_loops_run_like_pairs() {
    for setup in [STRINGS, SPARSE, FLOATS, TABLES] {
        let (source, floor) = next_of(setup, "n = n + v", 100);
        assert_loop(&source, floor);
    }
    let (source, floor) = next_of(ARRAY, "n = n + k", 100);
    assert_inline(&source, floor);
    let (source, floor) = next_of(HOLES, "n = n + v", 66);
    assert_either(&source, floor / 2);
    // The third value of the generic `for` is the start: `next, t, nil`.
    let source = traversal(STRINGS, "next, t, nil", "n = n + v", ROUNDS);
    assert_loop(&source, ROUNDS * 90);
}

#[test]
fn a_loop_over_the_keys_only() {
    let source = format!(
        "local n = 0 {STRINGS} for r = 1, {ROUNDS} do for k in pairs(t) do n = n + #k end end return n"
    );
    assert_loop(&source, ROUNDS * 90);
    let source = format!(
        "local n = 0 {ARRAY} for r = 1, {ROUNDS} do for k in pairs(t) do n = n + k end end return n"
    );
    assert_inline(&source, ROUNDS * 90);
}

#[test]
fn values_that_are_not_numbers() {
    let source = format!(
        "local n = 0 local t = {{}} for i = 1, 100 do t[i] = 'v' .. i end \
         for r = 1, {ROUNDS} do for k, v in pairs(t) do n = n + #v end end return n"
    );
    assert_loop(&source, ROUNDS * 90);
    // Tables as values move from one array to another.
    let source = format!(
        "local n = 0 local t, out = {{}}, {{}} for i = 1, 100 do t[i] = {{ i }} out[i] = false end \
         for r = 1, {ROUNDS} do for k, v in pairs(t) do out[k] = v end end return out[7][1]"
    );
    assert_loop(&source, ROUNDS * 90);
    let source = format!(
        "local n = 0 local t = {{}} for i = 1, 100 do t[i] = (i % 2 == 0) end \
         for r = 1, {ROUNDS} do for k, v in pairs(t) do if v then n = n + k end end end return n"
    );
    assert_inline(&source, ROUNDS * 90);
}

#[test]
fn nested_pairs_loops() {
    // The outer pairs loop leaves the trace once per inner traversal (the
    // inner loop's end is a side exit), so the exits are bounded by that.
    for (row, helper) in [("row['c' .. j]", true), ("row[j]", false)] {
        let source = format!(
            "local n = 0 local t = {{}} \
            for i = 1, 20 do local row = {{}} for j = 1, 30 do {row} = i * j end t[i] = row end \
            for r = 1, 100 do \
              for i, row in pairs(t) do for j, x in pairs(row) do n = n + x end end \
            end return n"
        );
        for profile in PROFILES {
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, &source);
            let (actual, lua) = outcome(profile, ExecutionMode::Tracing, &source);
            assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
            let stats = lua.vm().jit_statistics().trace;
            assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
            assert!(
                stats.iterations >= 100 * 20 * 30 * 9 / 10,
                "{profile:?} {source}: {stats:?}"
            );
            if helper {
                assert!(next_helper_called(&lua), "{profile:?} {source}");
            }
            let exits: u64 = stats.exits.iter().sum();
            assert!(
                exits <= 100 * 20 * 2,
                "{profile:?} {source}: {exits} exits: {stats:?}"
            );
            no_native_aborts(profile, &source, &stats);
        }
    }
}

#[test]
fn a_pairs_loop_inside_a_long_outer_loop_stays_native() {
    // The end of the traversal is a branch a side trace takes, so the outer
    // loop runs in native code too: the exits stay far below the entries,
    // whatever the shape of the table. An array part alone is the exception,
    // as it is for `ipairs`: its traversal leaves the trace once per run,
    // where a side would make the hot loop share registers with the loop
    // around it.
    for table in [STRINGS, SPARSE, ARRAY, HOLES, MIXED] {
        let source = format!(
            "local n, c = 0, 0 {table} \
             for r = 1, 3000 do for k, v in pairs(t) do n = n + v c = c + 1 end end return n, c"
        );
        let array_only = table == ARRAY || table == HOLES;
        for profile in PROFILES {
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, &source);
            let (actual, lua) = outcome(profile, ExecutionMode::Tracing, &source);
            assert_eq!(show(&actual), show(&expected), "{profile:?} {table}");
            let stats = lua.vm().jit_statistics().trace;
            assert!(
                stats.iterations >= 3000 * 60 * 9 / 10,
                "{profile:?} {table} {stats:?}"
            );
            let exits: u64 = stats.exits.iter().sum();
            if array_only {
                assert!(
                    exits <= 3000 + 10 && exits * 2 > 3000,
                    "{profile:?} {table}: {exits} exits: {stats:?}"
                );
            } else {
                assert!(
                    exits * 10 < 3000,
                    "{profile:?} {table}: {exits} exits: {stats:?}"
                );
            }
        }
    }
}

#[test]
fn a_body_exit_on_the_first_entry_after_a_side_trace_links_back() {
    // The side trace that starts the next traversal links back into the
    // loop. A guard of the body that fails on that first entry leaves with
    // the key the loop holds, so the interpreter goes on with the entry
    // after it and drops none.
    for table in [STRINGS, SPARSE, MIXED] {
        for condition in [
            "r == 1999",
            "r == 1999 and v == 100",
            "r == 1999 and #tostring(k) > 0",
        ] {
            let source = format!(
                "local n, c = 0, 0 {table} \
                 for r = 1, 2000 do for k, v in pairs(t) do n = n + v \
                   if {condition} then c = c + 1 end end end return n, c"
            );
            for profile in PROFILES {
                let (expected, _) = outcome(profile, ExecutionMode::Interpreter, &source);
                let (actual, _) = outcome(profile, ExecutionMode::Tracing, &source);
                assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
            }
        }
    }
    // The key is read in the body.
    let source = format!(
        "local n, c = 0, 0 {STRINGS} \
         for r = 1, 2000 do for k, v in pairs(t) do n = n + v + #k \
           if r == 1999 then c = c + 1 end end end return n, c"
    );
    assert_agrees(&source);
}

#[test]
fn a_side_trace_that_deletes_an_entry_ahead_turns_off_the_native_hash_step() {
    // The delete sits in a side branch, so only the side's IR moves the hash
    // storage. The tree must still read the entries through the runtime and
    // skip the deleted one, as the interpreter does.
    let sparse = "local total, c = 0, 0 \
         local t = {} for i = 1, 100 do t[1000 + i * 7] = i end \
         for r = 1, 2000 do \
           for _, v in pairs(t) do total = total + v c = c + 1 \
             if v == 10 then t[1000 + 90 * 7] = nil end end \
           t[1000 + 90 * 7] = 90 \
         end return total, c";
    assert_either(sparse, 100_000);
    // A small table whose entries keep insertion order, and a body that reads
    // the integer keys.
    let ordered = "local s = 0 \
         local u = {[100] = 1, [200] = 2, [300] = 3, [400] = 4, x = 5} \
         for r = 1, 2000 do \
           for k, v in pairs(u) do \
             if type(k) == 'number' then s = s + k end s = s * 3 % 1000003 + v \
             if v == 2 then u[300] = nil end end \
           u[300] = 3 \
         end return s";
    assert_agrees_native(ordered);
}

#[test]
fn empty_tables_and_short_traversals() {
    // Loops too short to record: the results are the interpreter's.
    assert_agrees(
        "local n = 0 for r = 1, 3000 do for k, v in pairs({}) do n = n + 1 end end return n",
    );
    assert_agrees(
        "local n, t = 0, {} for r = 1, 3000 do for k, v in next, t do n = n + 1 end n = n + 1 end return n",
    );
    assert_agrees(
        "local n, t = 0, {5} for r = 1, 3000 do for k, v in pairs(t) do n = n + v end end return n",
    );
    assert_agrees(
        "local n, t = 0, {x = 1} for r = 1, 3000 do for k, v in pairs(t) do n = n + v end end return n",
    );
    // A table that empties out while the loop is hot.
    assert_agrees_native(
        "local n, t = 0, {} for i = 1, 100 do t[i] = i end \
         for r = 1, 30000 do \
           if r == 2000 then for i = 1, 100 do t[i] = nil end end \
           for k, v in pairs(t) do n = n + v end \
         end return n",
    );
}

#[test]
fn a_call_of_next_outside_a_loop() {
    // The emptiness test and the first entry.
    assert_agrees_native(
        "local n, t = 0, {a = 1} for r = 1, 30000 do if next(t) ~= nil then n = n + 1 end end return n",
    );
    assert_agrees_native(
        "local n, t = 0, {7, 8} for r = 1, 30000 do local k, v = next(t) n = n + k + v + r % 7 end return n",
    );
    assert_agrees_native(
        "local n, t = 0, {} for r = 1, 30000 do if next(t) == nil then n = n + 1 end end return n",
    );
}

#[test]
fn ipairs_called_in_an_outer_loop() {
    let source = "local values = {} for i = 1, 3000 do values[i] = i % 113 end \
        local total = 0 \
        for _ = 1, 40 do for _, value in ipairs(values) do total = total + value end end \
        return total";
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(
            stats.iterations >= 40 * 3000 * 9 / 10,
            "{profile:?} {stats:?}"
        );
        assert_eq!(stats.native_aborts.of("ipairs"), 0, "{profile:?} {stats:?}");
    }
}

#[test]
fn assigning_to_existing_fields_while_traversing() {
    // The clearing idiom on a hash part (a store under a computed string key
    // is no trace yet), and updates in place.
    let source = format!(
        "local n = 0 local t = {{}} \
         for r = 1, {ROUNDS} do \
           for i = 1, 50 do t['k' .. i] = i end \
           for k, v in pairs(t) do n = n + v t[k] = nil end \
         end return n, next(t)"
    );
    assert_agrees(&source);
    let source = format!(
        "local n = 0 {STRINGS} \
         for r = 1, {ROUNDS} do for k, v in pairs(t) do t[k] = v + 1 n = n + v end end return n"
    );
    assert_agrees(&source);
    // On an array part they do run natively.
    let source = format!(
        "local n = 0 {ARRAY} \
         for r = 1, {ROUNDS} do for k, v in pairs(t) do t[k] = v + 1 n = n + v end end return n"
    );
    assert_agrees_native(&source);
    let source = format!(
        "local n = 0 {ARRAY} \
         for r = 1, {ROUNDS} do for k, v in pairs(t) do t[k] = nil n = n + v end \
           for i = 1, 100 do t[i] = i end end return n, next(t)"
    );
    assert_agrees_native(&source);
    // Sparse integer keys.
    let source = format!(
        "local n = 0 {SPARSE} \
         for r = 1, {ROUNDS} do for k, v in pairs(t) do t[k] = v + 1 n = n + v end end return n"
    );
    assert_agrees(&source);
}

#[test]
fn a_pairs_handler_decides_where_the_profile_honours_it() {
    let handler = "setmetatable(t, {__pairs = function(self) \
        return function(_, k) if not k then return 1, 1000 end end, self, nil end})";
    // From the start.
    let source = format!(
        "local n = 0 {STRINGS} {handler} \
         for r = 1, {ROUNDS} do for k, v in pairs(t) do n = n + v end end return n"
    );
    assert_agrees(&source);
    // Added after the loop is hot.
    let source = format!(
        "local n = 0 {STRINGS} \
         for r = 1, {ROUNDS} do \
           if r == {} then {handler} end \
           for k, v in pairs(t) do n = n + v end \
         end return n",
        ROUNDS * 3 / 4
    );
    assert_agrees_native(&source);
    // A metatable with other metamethods is no handler.
    let source = format!(
        "local n = 0 {STRINGS} setmetatable(t, {{__index = function() return 1 end}}) \
         for r = 1, {ROUNDS} do for k, v in pairs(t) do n = n + v end end return n"
    );
    assert_loop(&source, ROUNDS * 90);
}

#[test]
fn a_replaced_pairs_or_next() {
    let source = format!(
        "local n = 0 {STRINGS} \
         for r = 1, {ROUNDS} do \
           if r == {} then pairs = function(t) return next, {{ z = 1000 }}, nil end end \
           for k, v in pairs(t) do n = n + v end \
         end return n",
        ROUNDS * 3 / 4
    );
    assert_agrees_native(&source);
    let source = format!(
        "local n = 0 {STRINGS} local real = next \
         for r = 1, {ROUNDS} do \
           if r == {} then next = function(t, k) if k == nil then return 'x', 1000 end end end \
           for k, v in next, t do n = n + v end \
         end return n",
        ROUNDS * 3 / 4
    );
    assert_agrees(&source);
    let source = format!(
        "local n = 0 {STRINGS} \
         for r = 1, {ROUNDS} do \
           local it = next \
           if r == {} then it = function(t, k) if k == nil then return 'x', 1000 end end end \
           for k, v in it, t do n = n + v end \
         end return n",
        ROUNDS * 3 / 4
    );
    assert_agrees_native(&source);
}

#[test]
fn an_invalid_continuation_key_raises_the_interpreters_error() {
    let keys = "local t, keys = {}, {} for i = 1, 10 do t['k' .. i] = i end \
        for k in pairs(t) do keys[#keys + 1] = k end local bad = {} bad[1] = 'zz' \
        for i = 1, 10 do t['k' .. i] = i end";
    let source = format!(
        "{keys} local n, last = 0, nil \
        for r = 1, 30000 do \
          local key = keys[3] \
          if r > 29900 then key = bad[1] end \
          local ok, k, v = pcall(next, t, key) \
          if ok then n = n + v else last = k end \
        end return n, last"
    );
    assert_agrees(&source);
    let source = format!(
        "{keys} local n = 0 \
        for r = 1, 30000 do \
          local key = keys[3] \
          if r > 29900 then key = bad[1] end \
          local k, v = next(t, key) \
          n = n + v \
        end return n"
    );
    assert_agrees_native(&source);
    let source = format!(
        "{keys} local n = 0 \\
        local function step(key) local k, v = next(t, key) n = n + v end \\
        for r = 1, 30000 do \\
          local key = keys[3] \\
          if r > 29900 then key = bad[1] end \\
          local ok, err = pcall(step, key) \\
          if not ok then n = n .. err end \\
        end return n"
    );
    assert_agrees(&source);
    let source = "local t = {} for i = 1, 10 do t[i] = i end \
        local n = 0 \
        for r = 1, 30000 do \
          local key = 3 \
          if r > 29900 then key = 99 end \
          local k, v = next(t, key) \
          n = n + (v or 0) \
        end return n";
    assert_agrees_native(source);
}

#[test]
fn next_on_a_value_that_is_not_a_table() {
    let source = "local t = {1, 2, 3} local n = 0 \
        for r = 1, 30000 do \
          local x = t \
          if r > 29900 then x = 5 end \
          local ok, k, v = pcall(next, x) \
          if ok then n = n + v else n = n - 1 end \
        end return n";
    assert_agrees(source);
    let source = "local t = {1, 2, 3} local n = 0 \
        for r = 1, 30000 do \
          local x = t \
          if r > 29900 then x = 5 end \
          local k, v = next(x) \
          n = n + v \
        end return n";
    assert_agrees_native(source);
    let source = "local t = {1, 2, 3} local n = 0 \
        for r = 1, 30000 do \
          local x = t \
          if r > 29900 then x = nil end \
          for k, v in pairs(x) do n = n + v end \
        end return n";
    assert_agrees_native(source);
}

#[test]
fn an_array_part_with_keys_that_continue_in_the_hash_part() {
    // The keys 0 and below are hash keys: a traversal that reaches one
    // continues in the hash part, not at the first slot of the array.
    let source = format!(
        "local n, c = 0, 0 {ARRAY} t[0] = 1000 t[-1] = 2000 t[101] = 3000 t[500] = 4000 \
         for r = 1, {ROUNDS} do for k, v in pairs(t) do n = n + k * v c = c + 1 end end \
         return n, c"
    );
    assert_agrees_native(&source);
    let source = format!(
        "local n, c = 0, 0 {ARRAY} t[0] = 1000 \
         for r = 1, {ROUNDS} do for k, v in next, t, 0 do n = n + k * v c = c + 1 end end \
         return n, c"
    );
    assert_agrees(&source);
    // A start the table has no entry for raises.
    let source = format!(
        "local n = 0 {ARRAY} \
         for r = 1, 3000 do for k, v in next, t, 0 do n = n + k * v end end return n"
    );
    assert_agrees(&source);
    let source = format!(
        "local n = 0 {HOLES} \
         for r = 1, 3000 do for k, v in next, t, 1 do n = n + k * v end end return n"
    );
    assert_agrees(&source);
}

#[test]
fn elements_removed_or_changed_during_an_array_traversal() {
    // The next slot disappears before the traversal gets to it.
    for body in [
        "if k % 7 == 0 then t[k + 1] = nil end n = n + v",
        "if k % 7 == 0 then t[k + 1] = false end n = n + (v or 1)",
        "t[k] = nil n = n + v",
        "if k == 50 then t[101] = nil t[100] = nil end n = n + v",
        "t[k] = v + 1 n = n + v",
    ] {
        let source = format!(
            "local n = 0 local t = {{}} \
             for r = 1, {ROUNDS} do \
               for i = 1, 100 do t[i] = i end \
               for k, v in pairs(t) do {body} end \
             end return n, next(t)"
        );
        assert_agrees_native(&source);
    }
}

#[test]
fn the_iterators_through_locals_and_upvalues() {
    let source = format!(
        "local pairs, next, ipairs = pairs, next, ipairs local n = 0 {STRINGS} {ARRAY} \
         local u = {{}} for i = 1, 100 do u[i] = i end \
         for r = 1, {ROUNDS} do \
           for k, v in pairs(t) do n = n + v end \
           for k, v in next, t do n = n + v end \
           for k, v in ipairs(u) do n = n + v end \
         end return n"
    );
    assert_agrees_native(&source);
    let source = format!(
        "local n = 0 {STRINGS} \
         local function sum() local s = 0 for k, v in pairs(t) do s = s + v end return s end \
         for r = 1, {ROUNDS} do n = n + sum() end return n"
    );
    assert_agrees_native(&source);
}

/// The shape of `benchmarks/lua/table/hash_iteration.lua`: string keys and
/// sparse integer keys in one table, traversed in an outer loop. The first
/// variable is never read, so the trace takes keys of both types.
const HASH_ITERATION: &str = "local values = {} \
    for i = 1, 128 do values['item_' .. i] = i values[10000 + i] = i + 128 end \
    local total, count = 0, 0 \
    for _ = 1, 1000 do \
      for _, value in pairs(values) do total = total + value count = count + 1 end \
    end \
    return total, count";

/// Runs `source` and checks that it agrees with the interpreter, that nearly
/// every one of `iterations` ran natively, and that the exits are fewer than
/// one in `per_exit` iterations.
fn assert_stays_native(source: &str, iterations: u64, rounds: u64) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(
            stats.iterations >= iterations * 9 / 10,
            "{profile:?}: {} iterations, wanted {iterations}: {stats:?}",
            stats.iterations
        );
        let exits: u64 = stats.exits.iter().sum();
        assert!(
            exits * 10 < rounds,
            "{profile:?}: {exits} exits in {rounds} rounds: {stats:?}"
        );
        assert!(next_helper_called(&lua), "{profile:?}");
        no_native_aborts(profile, source, &stats);
    }
}

#[test]
fn keys_of_two_types_stay_in_one_native_loop() {
    assert_stays_native(HASH_ITERATION, 1000 * 256, 1000);
}

#[test]
fn the_array_part_and_the_hash_part_stay_in_one_native_loop() {
    let source = format!(
        "local n = 0 {MIXED} \
         for r = 1, 1000 do for k, v in pairs(t) do n = n + v end end return n"
    );
    assert_stays_native(&source, 1000 * 120, 1000);
    // With the key read as well: it has the type of the first entries, and
    // the entries of another type leave the loop for a side trace.
    let source = format!(
        "local n = 0 {MIXED} \
         for r = 1, 1000 do for k, v in pairs(t) do if k == 1 then n = n + v end end end return n"
    );
    assert_agrees_native(&source);
}

#[test]
fn a_key_the_body_reads_is_a_value_of_the_loop() {
    // Keys held by lanes: the key of the last entry outlives the step that
    // found the next.
    let source = format!(
        "local n, last = 0, '' {STRINGS} \
         for r = 1, {ROUNDS} do \
           local prev = '' \
           for k, v in pairs(t) do n = n + #prev + #k prev = k end \
           last = prev \
         end return n, last"
    );
    assert_loop(&source, ROUNDS * 90);
    // Keys that index a second table.
    let source = format!(
        "local n = 0 {STRINGS} local u = {{}} for k, v in pairs(t) do u[k] = 2 * v end \
         for r = 1, {ROUNDS} do for k, v in pairs(t) do n = n + u[k] end end return n"
    );
    assert_loop(&source, ROUNDS * 90);
    // Table keys.
    let source = format!(
        "local n = 0 {TABLES} local seen = 0 \
         for r = 1, {ROUNDS} do for k, v in pairs(t) do if k then seen = seen + 1 end n = n + v end end \
         return n, seen"
    );
    assert_loop(&source, ROUNDS * 90);
}

#[test]
fn a_third_variable_and_a_variable_the_body_overwrites() {
    let source = format!(
        "local n = 0 {STRINGS} \
         for r = 1, {ROUNDS} do for k, v, extra in pairs(t) do n = n + v if extra then n = n + 1000 end end end \
         return n"
    );
    assert_loop(&source, ROUNDS * 90);
    let source = format!(
        "local n = 0 {STRINGS} \
         for r = 1, {ROUNDS} do for k, v in pairs(t) do local kk = 5 local vv = v + kk n = n + vv end end return n"
    );
    assert_loop(&source, ROUNDS * 90);
}

#[test]
fn leaving_a_loop_in_the_middle_continues_the_interpreters_traversal() {
    // A guard fails part of the way (a float among integers, a value that
    // ends the loop): the interpreter continues from the key it finds in the
    // control register, and `next` goes on from the last key.
    let source = format!(
        "local n, resumed = 0, 0 {STRINGS} t.k50 = 0.5 t.k75 = 2.25 \
         for r = 1, {ROUNDS} do \
           local last for k, v in pairs(t) do n = n + v last = k if v == 60 + r % 7 then break end end \
           local k2, v2 = next(t, last) resumed = resumed + (v2 or 0) \
         end return n, resumed"
    );
    assert_agrees_native(&source);
    let source = format!(
        "local n = 0 {STRINGS} t.k50 = 0.5 t.k75 = 2.25 \
         for r = 1, {ROUNDS} do for k, v in pairs(t) do n = n + v end end return n"
    );
    assert_agrees_native(&source);
    // An error part of the way, caught by the caller.
    let source = format!(
        "local n, caught = 0, 0 {STRINGS} \
         local function run(stop) local s = 0 for k, v in pairs(t) do if v == stop then error('stop') end s = s + v end return s end \
         for r = 1, {ROUNDS} do \
           local ok, result = pcall(run, 1000 + 20 * (r % 3 - 1) + 20) \
           if ok then n = n + result else caught = caught + 1 end \
         end return n, caught"
    );
    assert_agrees_native(&source);
}

#[test]
fn two_traversals_of_one_table() {
    let source = "local n = 0 local t = {} for i = 1, 30 do t['k' .. i] = i end \
         for r = 1, 100 do for k, v in pairs(t) do for k2, v2 in pairs(t) do n = n + v * v2 end end end \
         return n";
    assert_agrees_native(source);
    // The inner traversal clears what the outer one has not reached.
    let source = format!(
        "local n = 0 \
         for r = 1, {ROUNDS} do \
           local t = {{}} for i = 1, 30 do t['k' .. i] = i end \
           for k, v in pairs(t) do n = n + v t[k] = nil end \
         end return n"
    );
    assert_agrees(&source);
}

#[test]
fn a_traversal_in_a_coroutine() {
    let source = format!(
        "local n = 0 {STRINGS} \
         local co = coroutine.wrap(function() \
           for r = 1, {ROUNDS} do for k, v in pairs(t) do n = n + v end coroutine.yield(r) end \
         end) \
         for r = 1, {ROUNDS} do co() end return n"
    );
    assert_agrees_native(&source);
}

#[test]
fn a_slot_the_array_part_grew_past_is_no_continuation_key() {
    // `t[13]` grows the array part past two slots nothing was stored in. The
    // interpreter raises for `next (t, 12)`: the slot is neither filled nor a
    // deleted key. The trace reads the slot the key names before it steps to
    // the next, so a key like that leaves for the interpreter.
    let source = "local t = {} for i = 1, 10 do t[i] = true end t[13] = true \
        local ks, rounds = {}, 300000 \
        for r = 1, rounds do ks[r] = 3 end \
        for r = rounds - 99, rounds do ks[r] = 12 end \
        local n = 0 \
        for r = 1, rounds do local k, v = next(t, ks[r]) if v then n = n + k end end \
        return n";
    assert_agrees_native(source);
    // The slot that follows a deleted key is a valid step, and so is the one
    // after a filled slot next to a hole.
    let source = "local t = {} for i = 1, 10 do t[i] = true end t[13] = true \
        local n = 0 \
        for r = 1, 300000 do local k, v = next(t, 10) n = n + k end \
        return n";
    assert_agrees_native(source);
    let source = "local t = {} for i = 1, 10 do t[i] = true end t[13] = true \
        local ks, rounds = {}, 300000 \
        for r = 1, rounds do ks[r] = 3 end \
        for r = rounds - 99, rounds do ks[r] = 12 end \
        local n, bad = 0, 0 \
        for r = 1, rounds do \
          local ok, k = pcall(next, t, ks[r]) \
          if ok then n = n + k else bad = bad + 1 end \
        end return n, bad";
    assert_agrees_native(source);
}

#[test]
fn loops_that_read_neither_variable() {
    for variables in ["_", "_, _", "k, v", "_, _, _"] {
        let source = format!(
            "local c = 0 {MIXED} \
             for r = 1, {ROUNDS} do for {variables} in pairs(t) do c = c + 1 end end return c"
        );
        assert_loop(&source, ROUNDS * 100);
    }
    let source = format!(
        "local c = 0 {STRINGS} \
         for r = 1, {ROUNDS} do for _ in next, t do c = c + 1 end end return c"
    );
    assert_loop(&source, ROUNDS * 90);
}

/// A function that reports the loop variables `k`, `v` and `_` of the nearest
/// frame that has them, the way `debug.getlocal` shows them to a callee or an
/// error handler.
const LOCALS: &str = "local function names() \
        local level = 2 \
        while debug.getinfo(level, 'l') do \
          local found, i = {}, 1 \
          while true do \
            local name, value = debug.getlocal(level, i) \
            if not name then break end \
            if name == 'k' or name == 'v' or name == '_' then \
              found[#found + 1] = name .. '=' .. tostring(value) \
            end \
            i = i + 1 \
          end \
          if #found > 0 then return table.concat(found, ',') end \
          level = level + 1 \
        end \
        return 'none' \
      end";

/// Runs `source` and checks that the native run agrees with the interpreter
/// and that the interpreter's report names `wanted`, so a test cannot pass
/// by having nothing to compare.
fn assert_exits_report(source: &str, wanted: &str) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(
            show(&expected).contains(wanted),
            "{profile:?} {source}: {expected:?} lacks {wanted}"
        );
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert!(stats.iterations > 10_000, "{profile:?} {source} {stats:?}");
    }
}

#[test]
fn an_exit_restores_the_loop_variables_the_body_never_reads() {
    let table = "local t = {} for i = 1, 50 do t['k' .. i] = i end";
    // The stage-6 report: the body reads the value only, and a function the
    // exit calls reads the key.
    let source = format!(
        "{LOCALS} {table} local n, log = 0, {{}} \
         for r = 1, 3000 do \
           for k, v in pairs(t) do \
             n = n + v \
             if v == 25 then n = n + 1 if r % 1000 == 0 then log[#log + 1] = names() end end \
           end \
         end return n, table.concat(log, ';')"
    );
    assert_exits_report(&source, "k=k25,v=25");
    // The body reads the key only: the value is not a result of the step.
    let source = format!(
        "{LOCALS} {table} local n, log, u = 0, {{}}, {{}} for k, v in pairs(t) do u[k] = v end \
         for r = 1, 3000 do \
           for k, v in pairs(t) do \
             n = n + 1 \
             if u[k] == 25 then n = n + 1 if r % 1000 == 0 then log[#log + 1] = names() end end \
           end \
         end return n, table.concat(log, ';')"
    );
    assert_exits_report(&source, "k=k25,v=25");
    // Neither variable is read: the exit comes from a counter.
    let source = format!(
        "{LOCALS} {table} local n, c, log = 0, 0, {{}} \
         for r = 1, 3000 do \
           for k, v in pairs(t) do \
             c = c + 1 \
             if c % 50000 == 25 then n = n + 1 log[#log + 1] = names() end \
           end \
         end return c, n, table.concat(log, ';')"
    );
    assert_exits_report(&source, "k=k");
    // The names the programmer gave the variables do not matter, and the
    // loop may run over `next, t`.
    let source = format!(
        "{LOCALS} {table} local n, c, log = 0, 0, {{}} \
         for r = 1, 3000 do \
           for _, _ in next, t do \
             c = c + 1 \
             if c % 50000 == 25 then n = n + 1 log[#log + 1] = names() end \
           end \
         end return c, n, table.concat(log, ';')"
    );
    assert_exits_report(&source, "_=");
    // A table of several kinds of values and keys.
    let source = format!(
        "{LOCALS} local t = {{}} for i = 1, 20 do t['k' .. i] = i t[i * 1000] = 'v' .. i \
           t[i + 0.5] = {{ i }} end \
         local c, log = 0, {{}} \
         for r = 1, 2000 do \
           for k, v in pairs(t) do \
             c = c + 1 \
             if c % 30000 == 17 then log[#log + 1] = names() end \
           end \
         end return c, table.concat(log, ';')"
    );
    assert_exits_report(&source, "k=");
}

#[test]
fn an_exit_in_the_array_part_of_a_mixed_table_restores_the_loop_variables() {
    // The native step stores the key in the control lane (and the value in
    // the lane after it), and an exit restores the variables from them.
    let table = "local t = {} for i = 1, 40 do t[i] = i * 3 end \
                 for i = 1, 20 do t['k' .. i] = i end";
    // The value is read, the key is not: the exit is at an array entry.
    let source = format!(
        "{LOCALS} {table} local n, log = 0, {{}} \
         for r = 1, 3000 do \
           for k, v in pairs(t) do \
             n = n + v \
             if v == 60 then n = n + 1 if r % 1000 == 0 then log[#log + 1] = names() end end \
           end \
         end return n, table.concat(log, ';')"
    );
    assert_exits_report(&source, "k=20,v=60");
    // Neither is read: the exit comes from a counter, in the array part and
    // in the hash part.
    for modulus in [50_000, 61_000, 79_000] {
        let source = format!(
            "{LOCALS} {table} local c, log = 0, {{}} \
             for r = 1, 3000 do \
               for k, v in pairs(t) do \
                 c = c + 1 \
                 if c % {modulus} == 33 then log[#log + 1] = names() end \
               end \
             end return c, table.concat(log, ';')"
        );
        assert_exits_report(&source, "k=");
    }
    // The key is read only.
    let source = format!(
        "{LOCALS} {table} local n, log = 0, {{}} \
         for r = 1, 3000 do \
           for k, v in pairs(t) do \
             if k == 30 then n = n + 1 if r % 1000 == 0 then log[#log + 1] = names() end end \
           end \
         end return n, table.concat(log, ';')"
    );
    assert_exits_report(&source, "k=30,v=90");
}

#[test]
fn an_exit_in_the_hash_part_restores_the_loop_variables() {
    // Native steps along the hash part leave the key to the end of the run:
    // an exit restores the control value and the variables from what the
    // last step moved to, whatever the kind of the key.
    let tables = [
        "local t = {} for i = 1, 40 do t['k' .. i] = i t[1000 + i] = i + 100 end",
        "local t = {} for i = 1, 60 do t['k' .. i] = i end",
        "local t = {} for i = 1, 60 do t[i * 7] = i end",
        "local t = {} for i = 1, 30 do t['k' .. i] = i t[i + 0.5] = i t[{}] = i end \
         t[true] = 1 t[false] = 2",
        "local t = {} for i = 1, 40 do t[i] = i end for i = 1, 40 do t['k' .. i] = i end",
    ];
    for table in tables {
        for modulus in [20_011, 31_007, 45_001] {
            // Neither variable is read.
            let source = format!(
                "{LOCALS} {table} local c, log = 0, {{}} \
                 for r = 1, 3000 do \
                   for k, v in pairs(t) do \
                     c = c + 1 \
                     if c % {modulus} == 7 then log[#log + 1] = names() end \
                   end \
                 end return c, table.concat(log, ';')"
            );
            assert_exits_report(&source, "k=");
            // The value only.
            let source = format!(
                "{LOCALS} {table} local c, n, log = 0, 0, {{}} \
                 for r = 1, 3000 do \
                   for k, v in pairs(t) do \
                     c = c + 1 n = n + v \
                     if c % {modulus} == 7 then log[#log + 1] = names() end \
                   end \
                 end return c, n, table.concat(log, ';')"
            );
            assert_exits_report(&source, "k=");
        }
    }
    // The error comes from the middle of the hash part, and the handler
    // reads the variables.
    for table in tables {
        let source = format!(
            "{LOCALS} {table} local c = 0 \
             local function run() \
               for r = 1, 3000 do \
                 for k, v in pairs(t) do \
                   c = c + 1 \
                   if c == 100003 then error('boom') end \
                 end \
               end \
             end \
             local ok, report = xpcall(run, function(m) return tostring(m) .. '|' .. names() end) \
             return ok, report, c"
        );
        assert_exits_report(&source, "boom");
    }
}

#[test]
fn changes_during_a_traversal_of_a_table_with_both_parts() {
    // Clearing every entry as the traversal reaches it, array part and hash
    // part, and refilling for the next round.
    let source = "local t, n, c = {}, 0, 0 \
        for r = 1, 400 do \
          for i = 1, 30 do t[i] = i end \
          for i = 1, 20 do t['k' .. i] = i end \
          for k, v in pairs(t) do n = n + v c = c + 1 t[k] = nil end \
        end return n, c, next(t)";
    assert_agrees_native(source);
    // Overwriting the values, and clearing the odd ones.
    let source = "local t, n, c = {}, 0, 0 \
        for i = 1, 30 do t[i] = i end \
        for i = 1, 20 do t['k' .. i] = i end \
        for r = 1, 400 do \
          for k, v in pairs(t) do \
            n = n + v c = c + 1 \
            if v % 7 == 0 then t[k] = nil elseif v % 2 == 0 then t[k] = v + 1 end \
          end \
          for i = 1, 30 do t[i] = i end \
          for i = 1, 20 do t['k' .. i] = i end \
        end return n, c";
    assert_agrees_native(source);
    // The elements change kind half way, and a hole appears.
    let source = "local t, n, c = {}, 0, 0 \
        for i = 1, 30 do t[i] = i end \
        t[10] = 1.5 t[11] = false t[12] = nil t[13] = 'text' \
        for i = 1, 20 do t['k' .. i] = i end \
        for r = 1, 600 do \
          for k, v in pairs(t) do c = c + 1 if type(v) == 'number' then n = n + v end end \
        end return n, c";
    assert_agrees_native(source);
    // A key the body removes ahead of the traversal, and one it adds behind.
    let source = "local t, n, c = {}, 0, 0 \
        for r = 1, 600 do \
          for i = 1, 30 do t[i] = i end \
          t.x, t.y = 1, 2 \
          for k, v in pairs(t) do \
            c = c + 1 n = n + v \
            if k == 5 then t[20] = nil t[21] = nil end \
            if k == 'x' then t.y = nil end \
          end \
          for i = 1, 30 do t[i] = nil end t.x, t.y = nil, nil \
        end return n, c";
    assert_agrees(source);
}

#[test]
fn an_error_handler_sees_the_loop_variables_the_body_never_reads() {
    let table = "local t = {} for i = 1, 50 do t['k' .. i] = i end";
    // The interpreter's handler finds no frame with the variables (the
    // erroring frame lists no locals), so the runs must agree on that, on the
    // count of steps, and on the error.
    for head in ["k, v", "_, v", "k, _", "_, _"] {
        let source = format!(
            "{LOCALS} {table} local c = 0 \
             local function run() \
               for r = 1, 3000 do \
                 for {head} in pairs(t) do \
                   c = c + 1 \
                   if c == 120007 then error('boom') end \
                 end \
               end \
             end \
             local ok, report = xpcall(run, function(m) return tostring(m) .. '|' .. names() end) \
             local again, caught = pcall(run) \
             return ok, report, again, c"
        );
        assert_exits_report(&source, "boom");
    }
}

/// Runs `source` in the tracing mode of `profile` and returns the outcome,
/// the VM, and how many steps of `next` the trace helpers made meanwhile.
fn traced_steps(
    profile: CompatibilityProfile,
    source: &str,
) -> (Result<Vec<Value>, String>, Lua, u64) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(ExecutionMode::Tracing)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let before = lua.vm().trace_next_helper_steps();
    let results = lua.exec(source).map_err(|error| format!("{error:?}"));
    let steps = lua.vm().trace_next_helper_steps() - before;
    (results, lua, steps)
}

/// Checks that `source` agrees with the interpreter in every profile, runs at
/// least `floor` iterations natively and has the trace helpers make at most
/// `most` steps of `next`.
fn assert_steps(source: &str, floor: u64, most: u64) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua, steps) = traced_steps(profile, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(
            stats.iterations >= floor,
            "{profile:?} {source}: {} iterations, wanted {floor}: {stats:?}",
            stats.iterations
        );
        assert!(
            steps <= most,
            "{profile:?} {source}: {steps} helper steps, at most {most}: {stats:?} {:?}",
            lua.vm().trace_helper_calls()
        );
        no_native_aborts(profile, source, &stats);
    }
}

#[test]
fn an_array_part_is_read_in_place_inside_an_outer_loop() {
    // 400 traversals of 100 entries, steps of the helper only at the ends.
    for iterator in ["pairs(t)", "next, t"] {
        let source = traversal(ARRAY, iterator, "n = n + k", 400);
        assert_steps(&source, 400 * 100 * 9 / 10, 400 / 10);
    }
}

#[test]
fn an_array_part_with_holes_is_read_in_place() {
    // A hole every third slot, and every sixth.
    for step in [3, 6] {
        let setup = format!(
            "local t = {{}} for i = 1, 120 do t[i] = i end for i = 1, 120, {step} do t[i] = nil end"
        );
        let source = traversal(&setup, "pairs(t)", "n = n + k", 400);
        let entries = 400 * (120 - 120 / step);
        assert_steps(&source, entries * 8 / 10, entries / 20);
    }
}

#[test]
fn long_runs_of_holes_are_skipped_natively() {
    // Runs of 5 and of 40 nil slots between the entries, far past any
    // distance a trace could unroll, and a run that ends the array part.
    for (gap, slots) in [(6, 1200), (41, 1230)] {
        let setup = format!(
            "local t = {{}} for i = 1, {slots} do t[i] = i end \
             for i = 1, {slots} do if i % {gap} ~= 1 then t[i] = nil end end"
        );
        let source = traversal(&setup, "pairs(t)", "n = n + k", 600);
        let entries = 600 * (slots / gap);
        assert_steps(&source, entries * 8 / 10, entries / 20);
    }
    // The same with a hash part after the array, and in the `next` form.
    let setup = "local t = {} for i = 1, 600 do t[i] = i end \
        for i = 1, 600 do if i % 7 ~= 1 then t[i] = nil end end t.a, t.b = 1, 2";
    let source = traversal(setup, "next, t", "n = n + 1", 300);
    assert_agrees_native(&source);
}

#[test]
fn pairs_over_an_array_makes_no_helper_steps_in_an_outer_loop() {
    // The shape of the benchmark: `pairs` over a long array again and again.
    let setup = "local t = {} for i = 1, 5000 do t[i] = i end";
    let source = traversal(setup, "pairs(t)", "n = n + v", 24);
    assert_steps(&source, 24 * 5000 * 9 / 10, 24);
    let source = traversal(setup, "ipairs(t)", "n = n + v", 24);
    assert_steps(&source, 24 * 5000 * 9 / 10, 0);
}

#[test]
fn the_array_part_of_a_table_with_a_hash_part_is_stepped_natively() {
    let setup = "local t = {} for i = 1, 200 do t[i] = i end \
        t.a, t.b, t.c, t.d = 1, 2, 3, 4";
    let source = traversal(setup, "pairs(t)", "n = n + 1", 400);
    // 204 entries per traversal, of which four come from the hash part.
    assert_steps(&source, 400 * 204 * 8 / 10, 400 * 4 * 3 / 2);
    // The key is read as well: integers in the array part, strings after.
    // The strings of the hash part leave the loop for a side trace, which
    // takes the helper, and the array part stays native.
    let source = traversal(
        setup,
        "pairs(t)",
        "if type(k) == 'number' then n = n + k end",
        400,
    );
    assert_agrees_native(&source);
    // A short array part in front of a long hash part: the native step
    // costs the hash part nothing.
    let setup = "local t = {10, 20, 30} for i = 1, 100 do t['s' .. i] = i end";
    let source = traversal(setup, "pairs(t)", "n = n + 1", 400);
    assert_steps(&source, 400 * 103 * 8 / 10, 400 * 100 * 3 / 2);
}

#[test]
fn a_table_with_both_parts_is_one_version_of_the_loop() {
    // One version of the loop traverses the array part and the hash part:
    // the traversal does not leave natively and enter again at the border
    // between them. A big array with a few strings, a few integers with many
    // strings, and an array with holes in front of strings.
    for (array, hash) in [(200, 4), (3, 120), (60, 60)] {
        let setup = format!(
            "local t = {{}} for i = 1, {array} do t[i] = i end \
             for i = 1, {hash} do t['s' .. i] = i end"
        );
        let source = traversal(&setup, "pairs(t)", "n = n + 1", 800);
        let entries = 800 * (array + hash);
        for profile in PROFILES {
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, &source);
            let (actual, lua) = outcome(profile, ExecutionMode::Tracing, &source);
            assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
            let stats = lua.vm().jit_statistics().trace;
            assert!(
                stats.iterations >= entries * 9 / 10,
                "{profile:?} {source}: {stats:?}"
            );
            let exits: u64 = stats.exits.iter().sum();
            assert!(
                exits * 10 < 800,
                "{profile:?} {source}: {exits} exits: {stats:?}"
            );
            let entries: u64 = stats.native_entries;
            assert!(entries * 10 < 800, "{profile:?} {source}: {stats:?}");
            no_native_aborts(profile, &source, &stats);
        }
    }
}

#[test]
fn the_hash_part_is_stepped_by_the_helper_when_the_trace_may_move_it() {
    // The counter the tests above rely on does count: a body that adds keys
    // to another table may move hash storage, so the snapshot of the hash
    // part is not read in native code and every step is the helper's.
    let source = traversal(
        &format!("{STRINGS} local u = {{}}"),
        "pairs(t)",
        "u[v * 1000] = v",
        400,
    );
    for profile in PROFILES {
        let (_, _, steps) = traced_steps(profile, &source);
        assert!(steps >= 400 * 100 * 8 / 10, "{profile:?}: {steps}");
    }
}

#[test]
fn the_hash_part_is_stepped_natively() {
    // A traversal of the hash part reads the table's snapshot of it in
    // native code: the helper makes the first step of a traversal (and the
    // one that ends it), not each of the hundreds between.
    for setup in [
        STRINGS,
        SPARSE,
        "local t = {} for i = 1, 60 do t['k' .. i] = i t[1000 + i] = i * 2 end",
        "local t = {} for i = 1, 80 do t[i + 0.5] = i t['s' .. i] = i end",
    ] {
        for body in ["n = n + 1", "n = n + v"] {
            let source = traversal(setup, "pairs(t)", body, 400);
            assert_steps(&source, 400 * 100 * 8 / 10 * 6 / 10, 400 * 6);
        }
    }
    // The key is read: an integer key is the call's result, and a string key
    // would need a lane, so only the integer keys are native.
    let source = traversal(SPARSE, "pairs(t)", "n = n + k", 400);
    assert_steps(&source, 400 * 100 * 8 / 10, 400 * 6);
    // Tombstones: entries the body deletes in one round are skipped in the
    // next, and keys added between traversals take a new snapshot.
    let source = "local t, n, c = {}, 0, 0 \
        for i = 1, 100 do t['k' .. i] = i end \
        for r = 1, 600 do \
          for k, v in pairs(t) do n = n + v c = c + 1 end \
          if r % 50 == 0 then t['k' .. math.floor(r / 50)] = nil end \
          if r % 75 == 0 then t['x' .. r] = r end \
        end return n, c";
    assert_agrees_native(source);
    // A table that is too small to keep a snapshot, and a hash part with
    // table and boolean keys, take the helper's steps.
    let source = traversal(
        "local t = {a = 1, b = 2, c = 3, [true] = 4, [false] = 5}",
        "pairs(t)",
        "n = n + v",
        2000,
    );
    assert_agrees_native(&source);
    let source = traversal(TABLES, "pairs(t)", "n = n + v", 400);
    assert_agrees_native(&source);
}

#[test]
fn float_keys_give_the_integer_keys_the_interpreter_gives() {
    // On Lua 5.3 and later `next (t, 3.0)` finds the key 4 as an integer.
    let source = "local t = {} for i = 1, 10 do t[i] = i * 3 end \
        local xs, n, last = {}, 0, 0 \
        for r = 1, 5 do xs[r] = 2.0 + r end \
        for r = 1, 30000 do local k, v = next(t, xs[r % 5 + 1]) n = n + k + v last = k end \
        return n, last, math.type and math.type(last)";
    assert_agrees_native(source);
    // Where floats and integers are told apart, the loop runs natively and
    // no recording diverges on the key the step gives.
    for profile in [
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let (_, lua) = outcome(profile, ExecutionMode::Tracing, source);
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.iterations >= 25_000, "{profile:?} {stats:?}");
        assert_eq!(
            stats.aborts(TraceAbortReason::Divergence),
            0,
            "{profile:?} {stats:?}"
        );
        no_native_aborts(profile, source, &stats);
    }
}

#[test]
fn a_function_the_body_calls_sees_the_loop_variables() {
    // The call leaves the trace through the guard before it; the function
    // reads the variables of the loop that called it and raises them as the
    // error.
    let table = "local t = {} for i = 1, 50 do t['k' .. i] = i end";
    for head in ["k, v", "_, v", "k, _", "_, _"] {
        let source = format!(
            "{LOCALS} {table} local c = 0 \
             local function boom() error(names()) end \
             local function run() \
               for r = 1, 3000 do \
                 for {head} in pairs(t) do \
                   c = c + 1 \
                   if c == 120007 then boom() end \
                 end \
               end \
             end \
             local ok, report = pcall(run) \
             return ok, report, c"
        );
        assert_exits_report(&source, "=");
    }
}

#[test]
fn an_array_part_that_ends_in_nil_slots_goes_on_into_the_hash_part() {
    let setup = "local t = {} for i = 1, 40 do t[i] = i end t.a = 1 t.b = 2 \
        for i = 38, 40 do t[i] = nil end";
    let source = traversal(setup, "pairs(t)", "n = n + 1", 600);
    assert_agrees_native(&source);
}

#[test]
fn the_key_zero_continues_in_the_hash_part() {
    let source = "local t = {} for i = 1, 10 do t[i] = i end t[0] = 100 t.x = 7 \
        local xs, n = {}, 0 \
        for r = 1, 30000 do xs[r] = r % 4 end \
        for r = 1, 30000 do local k, v = next(t, xs[r]) n = n + v + r % 3 end \
        return n";
    assert_agrees_native(source);
}

#[test]
fn a_next_call_on_the_traversed_table_does_not_move_the_loop() {
    // A keyed `next` or a `next(t)` from the start in the body, or in a
    // function it calls, steps the same table; the loop's own position is
    // not the one they move.
    let bodies = [
        "local a = next(t, k) n = n + (a and 1 or 0)",
        "local a, b = next(t) n = n + b",
        "n = n + first(t)",
        "local a = next(t, k) local b = next(t) n = n + (a and 1 or 0)",
    ];
    for setup in [STRINGS, SPARSE, MIXED] {
        for body in bodies {
            let setup =
                format!("{setup} local function first(x) local a, b = next(x) return b end");
            let source = traversal(&setup, "pairs(t)", body, 400);
            assert_agrees_native(&source);
        }
    }
}

#[test]
fn a_loop_that_starts_at_a_slot_without_a_value_is_the_interpreters() {
    // The step after a key is native only when the key names a slot that
    // holds a value; a loop entered with a key that names a hole, a slot
    // past the array or a key of the hash part is left to the interpreter
    // (which raises for a key that is no continuation).
    let source = "local t = {} for i = 1, 100 do t[i] = i end t[10] = nil t.x = 5 \
        local function walk(from) \
          local n = 0 \
          for k, v in next, t, from do n = n + v end \
          return n \
        end \
        local total, bad = 0, 0 \
        for r = 1, 3000 do \
          local from = 1 \
          if r % 7 == 0 then from = 10 elseif r % 11 == 0 then from = 500 \
          elseif r % 13 == 0 then from = 'x' elseif r % 17 == 0 then from = 0 end \
          local ok, n = pcall(walk, from) \
          if ok then total = total + n else bad = bad + 1 end \
        end \
        return total, bad";
    assert_agrees_native(source);
}

#[test]
fn a_pairs_loop_in_an_outer_loop_resumes_after_each_exit() {
    // The key an inner loop carries is the one its last step found, also
    // after the loop was left to a side trace and came back.
    for setup in [ARRAY, HOLES, MIXED] {
        let source = format!(
            "local n, c = 0, 0 {setup} \
             for r = 1, 600 do \
               for k, v in pairs(t) do n = n + v c = c + 1 end \
               for k, v in pairs(t) do if type(k) == 'number' and k % 2 == 0 then n = n + 1 end end \
             end return n, c"
        );
        assert_agrees_native(&source);
    }
}

// Writes during a traversal, emptiness tests and manual stepping.

/// A table of `n` computed string keys `k1` .. `kn` holding `i`.
const KEYED: &str = "local t = {} for i = 1, 50 do t['k' .. i] = i end";

#[test]
fn clearing_the_integer_keys_during_a_traversal() {
    let source = "local n, t = 0, {} \
        for r = 1, 400 do \
          for i = 1, 100 do t[i] = i end \
          for k, v in pairs(t) do n = n + v t[k] = nil end \
        end return n, next(t)";
    assert_agrees_floor(source, 400 * 100 * 6 / 10);
    let source = "local n, t = 0, {} \
        for r = 1, 400 do \
          for i = 1, 100 do t[i * 3] = i end \
          for k, v in pairs(t) do n = n + v t[k] = nil end \
        end return n, next(t)";
    assert_agrees_floor(source, 400 * 100 * 6 / 10);
}

#[test]
fn clearing_computed_string_keys_during_a_traversal() {
    let source = "local n, t = 0, {} \
        for r = 1, 400 do \
          for i = 1, 50 do t['k' .. i] = i end \
          for k, v in pairs(t) do n = n + v t[k] = nil end \
        end return n, next(t)";
    assert_either(source, 400 * 50 * 6 / 10);
    // The table is built once and emptied once: the loop clears what it
    // reaches while the trace is running.
    let source = format!(
        "local n = 0 {KEYED} \
         for r = 1, 400 do \
           for k, v in pairs(t) do n = n + v t[k] = nil end \
           for i = 1, 50 do t['k' .. i] = i end \
         end return n, next(t)"
    );
    assert_either(&source, 400 * 50 * 6 / 10);
}

#[test]
fn updating_values_during_a_traversal() {
    for setup in [ARRAY, STRINGS, KEYED, MIXED, SPARSE] {
        let source = format!(
            "local n = 0 {setup} \
             for r = 1, 400 do for k, v in pairs(t) do t[k] = v * 2 % 1009 n = n + v end end \
             return n"
        );
        assert_either(&source, 400 * 50 * 6 / 10);
        let source = format!(
            "local n = 0 {setup} \
             for r = 1, 400 do for k, v in next, t do t[k] = v + 1 n = n + v + 1 end end \
             return n"
        );
        assert_either(&source, 400 * 50 * 6 / 10);
    }
}

#[test]
fn updating_another_table_during_a_traversal() {
    for setup in [ARRAY, STRINGS, MIXED] {
        let source = format!(
            "local n, u = 0, {{}} {setup} \
             for r = 1, 400 do \
               for k, v in pairs(t) do u[k] = v n = n + v end \
               for k, v in pairs(u) do u[k] = nil end \
             end return n, next(u)"
        );
        assert_either(&source, 400 * 50 * 6 / 10);
    }
}

#[test]
fn nil_assignment_of_keys_not_yet_visited() {
    // Integer keys: the key after the one reached goes, array part and hash
    // part alike.
    // HOLES leaves the array part on a side exit each round, so its floor
    // is lower.
    for (setup, floor) in [
        (ARRAY, 400 * 100 * 4 / 10),
        (SPARSE, 400 * 100 * 4 / 10),
        (HOLES, 400 * 60 * 3 / 10),
    ] {
        let refill = setup.replace("local t = {} ", "");
        let source = format!(
            "local n, c = 0, 0 {setup} \
             for r = 1, 400 do \
               for k, v in pairs(t) do \
                 n = n + v c = c + 1 \
                 if v % 10 == 0 then t[k + 1] = nil t[k + 1000] = nil end \
               end \
               {refill} \
             end return n, c"
        );
        assert_agrees_floor(&source, floor);
    }
    // String keys named by computed strings, ahead of the traversal.
    for setup in [STRINGS, MIXED] {
        let refill = setup.replace("local t = {} ", "");
        let source = format!(
            "local n, c = 0, 0 {setup} \
             for r = 1, 400 do \
               for k, v in pairs(t) do \
                 n = n + v c = c + 1 \
                 if v % 10 == 0 then t['key' .. (v + 1)] = nil t['k' .. (v + 1)] = nil end \
               end \
               {refill} \
             end return n, c"
        );
        assert_agrees_floor(&source, 400 * 100 * 6 / 10);
    }
    let source = "local n, c, t = 0, 0, {} \
        for r = 1, 400 do \
          for i = 1, 50 do t['k' .. i] = i end \
          for k, v in pairs(t) do n = n + v c = c + 1 t['k' .. (v + 1)] = nil end \
        end return n, c";
    assert_agrees_floor(source, 400 * 50 * 6 / 10);
}

#[test]
fn adding_keys_during_a_traversal() {
    // A growing array part: the traversal ends when the interpreter's does.
    let source = "local n, c = 0, 0 \
        for r = 1, 200 do \
          local t = {1, 2, 3} \
          for k, v in pairs(t) do n = n + v c = c + 1 if k < 40 then t[k + 3] = v end end \
        end return n, c";
    assert_agrees_floor(source, 200 * 39 * 8 / 10);
    // A rehash in a hash part: new keys while the cursor walks. The hash h
    // depends on the traversal order, so the modes must agree on it exactly.
    let source = "local n, c, h = 0, 0, 0 \
        for r = 1, 300 do \
          local t = {} \
          for i = 1, 30 do t['k' .. i] = i end \
          local added = 0 \
          for k, v in pairs(t) do \
            n = n + v c = c + 1 \
            h = (h * 31 + #k + v) % 1000003 \
            if added < 40 and v % 3 == 0 then \
              added = added + 1 t['x' .. r .. '_' .. added] = 100 + added \
            end \
          end \
        end return n, c, h";
    assert_agrees_floor(source, 300 * 40 * 6 / 10);
    // A deleted key brought back: the tombstone is reused.
    let source = format!(
        "local n, c = 0, 0 {KEYED} \
         for r = 1, 300 do \
           for k, v in pairs(t) do \
             n = n + v c = c + 1 \
             if v == 25 then t.k1 = nil t.k1 = 7 end \
             if v == 26 then t.k3 = nil end \
           end \
           t.k3 = 3 \
         end return n, c"
    );
    assert_agrees_floor(&source, 300 * 50 * 8 / 10);
    // The same, with the delete and the store on the traversal's key: a
    // deleted key stored again in place, a fused pair.
    let source = format!(
        "local n, c = 0, 0 {KEYED} \
         for r = 1, 300 do \
           for k, v in pairs(t) do \
             n = n + v c = c + 1 \
             if v == 25 then t[k] = nil t[k] = 7 end \
           end \
           t.k3 = 3 \
         end return n, c"
    );
    assert_agrees_floor(&source, 300 * 50 * 8 / 10);
    let source = format!(
        "local n, c = 0, 0 {KEYED} \
         for r = 1, 300 do \
           for k, v in pairs(t) do \
             n = n + v c = c + 1 \
             t[k] = nil \
             t['k' .. v] = v \
           end \
         end return n, c"
    );
    assert_agrees_floor(&source, 300 * 50 * 8 / 10);
}

#[test]
fn an_emptiness_test_in_a_hot_loop() {
    for table in ["{}", "{a = 1}", "{1}", "{n = 1, 2, 3}"] {
        let source = format!(
            "local n, t = 0, {table} \
             for r = 1, 200000 do if next(t) == nil then n = n + 1 else n = n + 2 end end \
             return n"
        );
        assert_running(&source, 150_000, None);
    }
    // The table empties and fills while the loop runs.
    let source = "local n, t = 0, {} \
        for r = 1, 200000 do \
          if next(t) == nil then n = n + 1 t.x = r else n = n + 2 t.x = nil end \
        end return n";
    assert_agrees_floor(source, 200_000 * 8 / 10);
    let source = "local n, t = 0, {} \
        for r = 1, 200000 do \
          if next(t) == nil then n = n + 1 t[1] = r else n = n + 2 t[1] = nil end \
        end return n";
    assert_agrees_floor(source, 200_000 * 8 / 10);
    // A walk over tables of which the first one recorded is empty.
    let source = "local n, ts = 0, {} \
        for i = 1, 100 do ts[i] = (i % 3 == 1) and {} or {i} end \
        for r = 1, 2000 do for i = 1, 100 do \
          if next(ts[i]) == nil then n = n + 1 else n = n + 2 end \
        end end return n";
    assert_agrees_floor(source, 200_000 * 8 / 10);
}

#[test]
fn stepping_by_hand_with_next() {
    let source = format!(
        "local n, c = 0, 0 {KEYED} \
         for r = 1, 400 do \
           local k, v = next(t) \
           while k do n = n + v c = c + 1 k, v = next(t, k) end \
         end return n, c"
    );
    assert_running(&source, 400 * 50 * 8 / 10, None);
    let source = format!(
        "local n, c = 0, 0 {MIXED} \
         for r = 1, 400 do \
           local k, v = next(t) \
           while k do n = n + v c = c + 1 k, v = next(t, k) end \
         end return n, c"
    );
    assert_running(&source, 400 * 120 * 8 / 10, None);
    let source = format!(
        "local n, c = 0, 0 {ARRAY} \
         for r = 1, 400 do \
           local k = next(t) \
           while k do n = n + k c = c + 1 k = next(t, k) end \
         end return n, c"
    );
    assert_running(&source, 400 * 100 * 8 / 10, None);
    // Stepping with a clearing write.
    let source = format!(
        "local n, c = 0, 0 {KEYED} \
         for r = 1, 400 do \
           local k, v = next(t) \
           while k do n = n + v c = c + 1 t[k] = nil k, v = next(t, k) end \
           for i = 1, 50 do t['k' .. i] = i end \
         end return n, c"
    );
    assert_agrees_floor(&source, 400 * 50 * 6 / 10);
}

#[test]
fn a_store_ahead_of_or_behind_the_traversal_reaches_the_next_step() {
    // The value a step reads is the one the body stored a moment before, for
    // the key just ahead of the cursor (array part, hash part with integer
    // keys, string keys) and for the one just behind it. The store sits in
    // the loop's root path, or in a branch most iterations skip (a side
    // trace).
    // Each store names a key the table already holds (a new key could keep
    // the traversal going for ever), so its condition bounds the values.
    let writes = [
        // Integer keys.
        (
            "k < 100 and k % 5 == 1",
            "t[k + 1] = 1000 + v",
            "k > 1 and k % 5 == 2",
            "t[k - 1] = 2000 + v",
            ARRAY,
        ),
        (
            "k < 100000 and k % 5000 == 0",
            "t[k + 1000] = 1000 + v",
            "k > 1000 and k % 3000 == 0",
            "t[k - 1000] = 2000 + v",
            SPARSE,
        ),
        // String keys named by computing them.
        (
            "v < 100 and v % 5 == 1",
            "t['key' .. (v + 1)] = 1000 + v",
            "v < 100 and v % 5 == 2 and v > 1",
            "t['key' .. (v - 1)] = 2000 + v",
            STRINGS,
        ),
        (
            "v < 100 and v % 5 == 1",
            "t['k' .. (v + 1)] = 1000 + v",
            "v < 100 and v % 5 == 2 and v > 1",
            "t['k' .. (v - 1)] = 2000 + v",
            MIXED,
        ),
    ];
    for (ahead_when, ahead, behind_when, behind, setup) in writes {
        for always in [true, false] {
            for (when, write) in [(ahead_when, ahead), (behind_when, behind)] {
                // In the root path, or in a branch some iterations skip.
                let guard = if always { "true" } else { "c % 7 == 0" };
                // The refill at the end of a round restores the table.
                let refill = setup.replace("local t = {} ", "");
                let source = format!(
                    "local n, c = 0, 0 {setup} \
                     for r = 1, 400 do \
                       for k, v in pairs(t) do \
                         n = n + v c = c + 1 \
                         if {guard} and {when} then {write} end \
                       end \
                       {refill} \
                     end return n, c"
                );
                assert_agrees_floor(&source, 400 * 100 * 6 / 10);
            }
        }
    }
}

#[test]
fn a_store_to_another_table_leaves_the_steps_of_this_one_native() {
    let source = format!(
        "local n, u = 0, {{}} {MIXED} \
         for r = 1, 600 do \
           for k, v in pairs(t) do u['x' .. v] = v u[k] = v n = n + v end \
         end return n, next(u)"
    );
    assert_either(&source, 600 * 120 * 6 / 10);
}

#[test]
fn a_store_in_a_side_branch_turns_off_the_native_hash_step_of_the_tree() {
    // The root path only reads, so the tree starts out stepping the hash part
    // natively. A branch taken now and then deletes the key just ahead of the
    // cursor and one behind it, by computed names; its side trace joins the
    // tree, and from then on the tree's steps must see those deletions.
    let source = "local n, c, t = 0, 0, {} \
        for r = 1, 600 do \
          for i = 1, 50 do t['k' .. i] = i end \
          for k, v in pairs(t) do \
            n = n + v c = c + 1 \
            if c % 11 == 0 then t['k' .. (v + 1)] = nil t['k' .. (v - 1)] = nil end \
          end \
        end return n, c";
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.side_installs > 0, "{profile:?} {stats:?}");
        assert!(stats.iterations >= 12000, "{profile:?} {stats:?}");
        no_native_aborts(profile, source, &stats);
    }
    // The same stores in the root path.
    let source = source.replace("if c % 11 == 0 then", "if true then");
    assert_agrees_floor(&source, 12000);
}

/// Whether a trace called the helper `name` (see `trace_helper_calls`).
fn helper_called(lua: &Lua, name: &'static str) -> bool {
    lua.vm().trace_helper_calls().contains(&name)
}

#[test]
fn updating_and_clearing_in_place_keeps_the_hash_part_stepped_natively() {
    // An overwrite of a number with a number, and the deletion of a key of a
    // hashed table, leave the entries where they are, so a trace that does
    // not read the key keeps stepping the hash part itself: the helper makes
    // the steps at the ends of the parts only.
    let source = format!(
        "local n = 0 {STRINGS} \
         for r = 1, 400 do for _, v in pairs(t) do t['key' .. v] = v + 1 n = n + v end \
           for i = 1, 100 do t['key' .. i] = i end end return n"
    );
    assert_steps(&source, 400 * 100 * 9 / 10, 400 * 3);
    let source = format!(
        "local n = 0 {STRINGS} \
         for r = 1, 400 do \
           for _, v in pairs(t) do n = n + v t['key' .. v] = nil end \
           for i = 1, 100 do t['key' .. i] = i end \
         end return n, next(t) ~= nil"
    );
    assert_steps(&source, 400 * 100 * 9 / 10, 400 * 3);
    // Booleans and floats over values of their own type.
    let source = "local n, t = 0, {} \
        for i = 1, 100 do t['key' .. i] = i end \
        for r = 1, 400 do \
          for _, v in pairs(t) do \
            if v % 2 == 0 then t['key' .. v] = 2.5 else t['key' .. v] = 3.5 end \
            n = n + 1 \
          end \
          for i = 1, 100 do t['key' .. i] = i end \
        end return n";
    assert_agrees_floor(source, 400 * 100 * 8 / 10);
}

#[test]
fn the_helpers_that_keep_the_layout_make_the_stores_that_can() {
    // A key read by the body is stepped by the helper, and the store of the
    // key it just reached is the overwrite or the delete in place.
    let cases = [
        (
            format!(
                "local n = 0 {STRINGS} \
                 for r = 1, 400 do for k, v in pairs(t) do t[k] = v + 1 n = n + v end end return n"
            ),
            "table.set_present",
        ),
        (
            format!(
                "local n = 0 {STRINGS} \
                 for r = 1, 400 do \
                   for k, v in pairs(t) do n = n + v t[k] = nil end \
                   for i = 1, 100 do t['key' .. i] = i end \
                 end return n"
            ),
            "table.delete_string",
        ),
        // A value of another type, and a key the table lacks, take the
        // general store.
        (
            format!(
                "local n = 0 {STRINGS} \
                 for r = 1, 400 do for k, v in pairs(t) do t[k] = v + 0.5 n = n + v end \
                   for k, v in pairs(t) do t[k] = 1 end end return n"
            ),
            "table.set_string",
        ),
    ];
    for (source, helper) in cases {
        for profile in PROFILES {
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, &source);
            let (actual, lua) = outcome(profile, ExecutionMode::Tracing, &source);
            assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
            let stats = lua.vm().jit_statistics().trace;
            assert!(
                stats.iterations >= 400 * 100 * 4 / 10,
                "{profile:?} {stats:?}"
            );
            // Where numbers are floats only, adding 0.5 keeps the type.
            let integers = !matches!(
                profile,
                CompatibilityProfile::Lua51
                    | CompatibilityProfile::Lua52
                    | CompatibilityProfile::LuaJit21
            );
            if helper != "table.set_string" || integers {
                assert!(
                    helper_called(&lua, helper),
                    "{profile:?} {source}: {:?}",
                    lua.vm().trace_helper_calls()
                );
            }
        }
    }
}

#[test]
fn stores_that_change_what_a_slot_holds_still_agree() {
    // A number replaced by a string or a table, and by nil, then back; small
    // tables (which shift entries on a delete) and large ones. The slot
    // changes kind between rounds, so only the agreement is checked there.
    for count in [6, 100] {
        let source = format!(
            "local n, t = 0, {{}} \
             for i = 1, {count} do t['key' .. i] = i end \
             for r = 1, 400 do \
               for k, v in pairs(t) do \
                 n = n + 1 \
                 if r % 4 == 0 then t[k] = tostring(v) \
                 elseif r % 4 == 1 then t[k] = tonumber(v) or 1 \
                 elseif r % 4 == 2 then t[k] = v ~= nil and 2.5 or 1 \
                 else t[k] = 7 end \
               end \
             end return n"
        );
        assert_agrees_native(&source);
        // Clearing keys that may be absent already.
        let source = format!(
            "local n, t = 0, {{}} \
             for r = 1, 400 do \
               for i = 1, {count} do t['key' .. i] = i end \
               for k, v in pairs(t) do n = n + v t[k] = nil t['key' .. (v + 1)] = nil end \
             end return n, next(t)"
        );
        assert_agrees_native(&source);
    }
    // A metatable with __newindex takes the store of an absent key.
    let source = "local n, log, t = 0, 0, {} \
        for i = 1, 60 do t['key' .. i] = i end \
        setmetatable(t, {__newindex = function(self, k, v) log = log + 1 end}) \
        for r = 1, 400 do \
          for k, v in pairs(t) do n = n + v t['key' .. (v + 1)] = nil t[k] = nil end \
          for i = 1, 60 do rawset(t, 'key' .. i, i) end \
        end return n, log, next(t) ~= nil";
    assert_agrees_native(source);
}

/// A trace reads a function or string in a field as a constant that entry
/// checked. A store under a run-time string key that names that field must
/// leave the trace, so the interpreter sees the field change and raises.
#[test]
fn a_store_under_a_run_time_key_does_not_replace_a_field_the_trace_reads_as_a_constant() {
    let setup = "local M = {f = function(x) return x + 1 end, s = 'abc', k = 's'} \
        for i = 1, 60 do M['k' .. i] = i end \
        local names = {} \
        for i = 1, 20000 do names[i] = 'z' .. i end";
    for (field, body) in [("f", "n = n + M.f(i)"), ("s", "n = n + #M.s")] {
        for store in ["M[names[i]] = nil", "M[names[i]] = i"] {
            let source = format!(
                "{setup} names[15000] = '{field}' \
                 local n = 0 \
                 for i = 1, 20000 do {body} {store} end \
                 return n"
            );
            assert_agrees_native(&source);
            let (expected, _) = outcome(
                CompatibilityProfile::Lua54,
                ExecutionMode::Interpreter,
                &source,
            );
            assert!(expected.is_err(), "{source}: {expected:?}");
        }
    }
    // Stores under other keys keep the trace: the loop runs natively.
    let source = format!(
        "{setup} local n = 0 \
         for i = 1, 20000 do n = n + M.f(i) + #M.s M[names[i]] = nil end \
         return n"
    );
    assert_agrees_floor(&source, 15_000);
}
