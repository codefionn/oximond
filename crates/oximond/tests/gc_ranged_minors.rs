//! Minor collections of an old table that received young stores, as a
//! program builds a list of fresh tables. A minor collection traces only the
//! array slots stored since the previous one, and an insert or remove that
//! shifts the list moves every slot after its place, so these programs must
//! keep every element alive across many collections in both execution modes.

use oximond::runtime::Value;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

fn run(mode: ExecutionMode, source: &str) -> Vec<Value> {
    run_in(CompatibilityProfile::Lua51, mode, source)
}

fn run_in(profile: CompatibilityProfile, mode: ExecutionMode, source: &str) -> Vec<Value> {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    lua.exec(source).unwrap_or_else(|error| panic!("{error:?}"))
}

/// Every other iteration stores a fresh table into an array that a minor
/// collection has already seen, with the constructor's table garbage in
/// between: each stored table must read back intact.
const APPENDS: &str = r"
local list, n = {}, 0
for i = 1, 100000 do
    local t = {i, i * 2, tag = i % 7}
    if i % 2 == 0 then
        n = n + 1
        list[n] = t
    end
end
local bad = 0
for j = 1, n do
    local t = list[j]
    local i = 2 * j
    if t[1] ~= i or t[2] ~= 2 * i or t.tag ~= i % 7 then
        bad = bad + 1
    end
end
return n, bad
";

/// Front inserts shift every table already in the list. Each shift must keep
/// the tables that moved alive through the collections the garbage triggers.
const FRONT_INSERTS: &str = r"
local list = {}
for i = 1, 300 do
    table.insert(list, {i})
end
for round = 1, 300 do
    table.insert(list, 1, {round})
    local garbage = {round, round}
    table.insert(list, {round + 300})
    garbage = nil
end
local sum, count = 0, 0
for _, t in ipairs(list) do
    count = count + 1
    sum = sum + t[1]
end
return count, sum
";

#[test]
fn appended_tables_survive_collections_in_both_modes() {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let results = run(mode, APPENDS);
        assert_eq!(
            results,
            vec![Value::from(50000.0), Value::from(0.0)],
            "{mode:?}"
        );
    }
}

#[test]
fn front_inserts_keep_shifted_tables_alive_in_both_modes() {
    // The initial tables hold 1..=300, the front inserts 1..=300 and the
    // appended tables 301..=600: 45150 + 45150 + 135150.
    let expected_sum = 225_450.0;
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let results = run(mode, FRONT_INSERTS);
        assert_eq!(
            results,
            vec![Value::from(900.0), Value::from(expected_sum)],
            "{mode:?}"
        );
    }
}

/// Sorting an old array of fresh tables permutes young children into slots no
/// store indexed. Each element must read back after the steps that follow.
const SORT_OLD_ARRAY: &str = r"
local n = 3000
local list = {}
for i = 1, n do list[i] = {i} end
collectgarbage()
for i = 1, n, 3 do list[i] = {n - i + 1000} end
table.sort(list, function(a, b) return a[1] < b[1] end)
for _ = 1, 6 do
    for k = 1, 200 do local garbage = {k, k} end
    collectgarbage('step')
end
local bad, last = 0, -1
for i = 1, n do
    local t = list[i]
    if type(t) ~= 'table' or type(t[1]) ~= 'number' or t[1] < last then
        bad = bad + 1
    else
        last = t[1]
    end
end
return n, bad
";

/// `table.move` inside one table shifts fresh children across slots the same
/// way: the shifted tables must survive the collections after it. A parallel
/// list of numbers takes the same moves and is the model.
const MOVE_WITHIN_TABLE: &str = r"
local n = 2000
local list, model = {}, {}
for i = 1, n do list[i] = {i}; model[i] = i end
collectgarbage()
for round = 1, 4 do
    table.move(list, 1, n - 1, 2)
    table.move(model, 1, n - 1, 2)
    list[1] = {-round}
    model[1] = -round
    for _ = 1, 3 do
        for k = 1, 300 do local garbage = {k} end
        collectgarbage('step')
    end
end
local bad = 0
for i = 1, n do
    local t = list[i]
    if type(t) ~= 'table' or t[1] ~= model[i] then bad = bad + 1 end
end
return n, bad
";

#[test]
fn sorting_an_old_array_of_fresh_tables_keeps_every_element() {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let results = run(mode, SORT_OLD_ARRAY);
        assert_eq!(
            results,
            vec![Value::from(3000.0), Value::from(0.0)],
            "{mode:?}"
        );
    }
}

#[test]
fn moving_within_an_old_array_keeps_shifted_tables_alive() {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let results = run_in(CompatibilityProfile::Lua54, mode, MOVE_WITHIN_TABLE);
        assert_eq!(
            results,
            vec![Value::Integer(2000), Value::Integer(0)],
            "{mode:?}"
        );
    }
}
