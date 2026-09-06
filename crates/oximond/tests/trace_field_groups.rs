//! Fields of one table read or written by the same stretch of a trace share
//! one guard on the row's presence mask (see `slot_group` in the trace
//! emitter). A table whose field changed its type, or lost it, leaves the
//! trace at the first access of the group and the
//! interpreter redoes the accesses, so every loop must give the interpreter's
//! result, errors included.
//!
//! The groups of 64 fields and more, which fall back to a compare of every slot,
//! are tested on hand-built traces in `jit/trace/table_rows_tests.rs`.

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

/// Runs `source` in both modes of every profile: the outcomes agree, and the
/// tracing run entered native code and left it at least once.
fn assert_agrees(source: &str) {
    for profile in ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(
            format!("{actual:?}"),
            format!("{expected:?}"),
            "{profile:?} {source}"
        );
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        // The loop completes once; more exits mean a guard failed.
        let exits: u64 = stats.exits.iter().sum::<u64>() + stats.side_exits.iter().sum::<u64>();
        assert!(exits > 1, "{profile:?} {source} {stats:?}");
    }
}

/// Four tables with the fields `a`, `b` and `c`, in the array `ts`.
const TABLES: &str = "
    local ts = {}
    for k = 1, 4 do ts[k] = {a = k, b = 2, c = 3} end
";

fn with_tables(body: &str) -> String {
    format!("{TABLES}\n{body}")
}

#[test]
fn a_field_that_changes_its_type_midway_leaves_the_trace_not_the_process() {
    assert_agrees(&with_tables(
        r"
        local s = 0
        for i = 1, 100000 do
            if i == 60000 then ts[2].c = '7' end
            local t = ts[i % 4 + 1]
            s = s + t.a + t.b + t.c
        end
        return s
        ",
    ));
}

#[test]
fn a_table_whose_metatable_answers_for_a_field_takes_over_midway() {
    // `c` is zero in the row of a table that has a metatable and lacks it.
    assert_agrees(&with_tables(
        r"
        local s = 0
        for i = 1, 100000 do
            if i == 60000 then
                ts[3] = setmetatable({a = 10, b = 20}, {__index = function() return 5 end})
            end
            if i == 80000 then ts[3].c = 6 end
            local t = ts[i % 4 + 1]
            s = s + t.a * 2 + t.b + t.c
        end
        return s
        ",
    ));
}

#[test]
fn a_field_removed_midway_reads_nil_like_the_interpreter() {
    assert_agrees(&with_tables(
        r"
        local s = 0
        for i = 1, 100000 do
            if i == 60000 then ts[2].b = nil end
            local t = ts[i % 4 + 1]
            s = s + t.a + (t.b or 7) + t.c
        end
        return s
        ",
    ));
    // Arithmetic on the missing field raises the same error in both modes.
    assert_agrees(&with_tables(
        r"
        local s = 0
        for i = 1, 100000 do
            if i == 60000 then ts[2].c = nil end
            local t = ts[i % 4 + 1]
            s = s + t.a + t.b + t.c
        end
        return s
        ",
    ));
}

#[test]
fn a_field_absent_when_the_row_was_filled_and_set_later() {
    assert_agrees(&with_tables(
        r"
        ts[2].c = nil
        local s = 0
        for i = 1, 100000 do
            if i == 30000 then ts[2].c = 5 end
            if i == 60000 then ts[2].c = '9' end
            local t = ts[i % 4 + 1]
            s = s + t.a + t.b + (t.c or 0)
        end
        return s
        ",
    ));
}

#[test]
fn a_store_to_the_table_ends_the_group() {
    // The loads of `a` and `c` and the store to `b` are in one stretch.
    assert_agrees(&with_tables(
        r"
        local s = 0
        for i = 1, 100000 do
            local t = ts[i % 4 + 1]
            local x = t.a
            t.b = x + i
            s = s + t.b + t.c
            if i == 60000 then ts[2].c = true; ts[3].c = '8' end
        end
        return s
        ",
    ));
    assert_agrees(&with_tables(
        r"
        local s = 0
        for i = 1, 100000 do
            local t = ts[i % 4 + 1]
            t.c = t.a + t.b + i
            s = s + t.c
            if i == 60000 then ts[2].a = '4' end
        end
        return s
        ",
    ));
}

#[test]
fn a_group_of_fields_read_through_a_closure_cell() {
    // `ts` is an upvalue of the pinned closure, so its table words sit in
    // the closure's row, which is as long as the rows of the fields.
    assert_agrees(&with_tables(
        r"
        local function f(i)
            local t = ts[i % 4 + 1]
            return i + t.a + t.b + t.c
        end
        local s = 0
        for i = 1, 100000 do
            if i == 60000 then ts[2].b = '4' end
            s = s + f(i)
        end
        return s
        ",
    ));
    assert_agrees(&with_tables(
        r"
        local function make()
            return function(i)
                local t = ts[i % 4 + 1]
                t.c = t.a + t.b + i
                return t.c
            end
        end
        local f = make()
        local s = 0
        for i = 1, 100000 do
            s = s + f(i)
            if i == 60000 then ts[2].a = '6' end
        end
        return s
        ",
    ));
}

#[test]
fn a_function_trace_reads_its_fields_one_guard_at_a_time() {
    // A function trace gives every table a lane of its own and keeps no slot
    // cache, so it checks no group. A node that changes a field's type
    // leaves the trace all the same.
    assert_agrees(
        r"
        local function make(depth, seed)
            if depth == 0 then return {value = seed} end
            return {value = seed, left = make(depth - 1, seed * 2 - 1), right = make(depth - 1, seed * 2)}
        end
        local function check(tree)
            if not tree.left then return tree.value end
            return tree.value + check(tree.left) - check(tree.right)
        end
        local total = 0
        for round = 1, 60 do
            local tree = make(9, round)
            if round == 40 then tree.left.right.value = '3' end
            total = total + check(tree)
        end
        return total
        ",
    );
}

#[test]
fn more_tables_than_the_lane_index_has_places_go_through_the_admit_stub() {
    // The run's lane index holds a few hundred places, so a loop over more
    // distinct tables misses it again and again and asks the VM for each
    // lane through the stub that keeps the loop's registers.
    assert_agrees(
        r"
        local ts = {}
        for k = 1, 1500 do ts[k] = {a = k * 0.5, b = k, c = 2.5} end
        local s = 0.0
        local t0 = 0.25
        for pass = 1, 40 do
            if pass == 30 then ts[777].c = '4' end
            for i = 1, #ts do
                local t = ts[i]
                s = s + t.a * t0 + t.b + t.c
                t0 = t0 + 0.125
            end
        end
        return s
        ",
    );
}
