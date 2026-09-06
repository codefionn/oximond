//! A loop that makes a table each round, keeps it in a ring and calls one
//! of many closures of one prototype (`benchmarks/lua/opt/retained_slots`).
//! The trace has short ways for each step: a list constructor is born with
//! its array part, a table stored into a list takes a write barrier that
//! reads the list once, a closure the run has met keeps its lane in a memo,
//! and a second read of one cell in a call reuses the first. Each program
//! here gives the interpreter's result under every profile, and several
//! push an input off the short way: a cell written between two reads, more
//! closures than the memo or the bank holds, stores that walk an old list
//! in each direction across collections, a constructor with keys.

use oximond::runtime::Value;
use oximond::vm::TraceStatistics;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua51,
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

/// Runs `source` in both modes of every profile; the results agree.
fn run_each(source: &str) -> Vec<TraceStatistics> {
    let mut all = Vec::new();
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?}: {expected:?}");
        assert_eq!(
            format!("{actual:?}"),
            format!("{expected:?}"),
            "{profile:?}"
        );
        all.push(lua.vm().jit_statistics().trace);
    }
    all
}

/// Asserts that every profile ran native iterations.
fn assert_native(all: &[TraceStatistics]) {
    for stats in all {
        assert!(stats.iterations > 0, "no native iteration: {stats:?}");
    }
}

/// The benchmark's loop over `closures` closures and a ring of `ring`
/// tables, with `{BODY}` as the body of each closure.
fn retained(closures: usize, ring: usize, body: &str) -> String {
    format!(
        r"
local function make(seed)
  local object = {{ v = seed }}
  local n = seed
  return function()
    {body}
  end
end
local closures = {{}}
for k = 1, {closures} do closures[k] = make(k) end
local ring = {{}}
for k = 1, {ring} do ring[k] = {{ 0, 0 }} end
collectgarbage()
local sum = 0
for i = 1, 60000 do
  local made = {{ i % 17, i % 23 }}
  ring[i % {ring} + 1] = made
  sum = (sum + closures[i % {closures} + 1]() + made[1] + made[2]) % 1000003
end
local held = 0
for k = 1, {ring} do held = held + ring[k][1] * 3 + ring[k][2] end
return sum, held
"
    )
}

#[test]
fn two_reads_of_one_cell_in_a_call_agree() {
    assert_native(&run_each(&retained(
        64,
        256,
        "object.v = (object.v + 1) % 1009; return object.v",
    )));
}

#[test]
fn a_cell_written_between_two_reads_is_read_again() {
    // The second read of `n` comes after the closure's own store to it.
    assert_native(&run_each(&retained(
        64,
        256,
        "local before = n; n = (before + 3) % 1013; return before + n",
    )));
    // A table cell swapped between two reads of it.
    assert_native(&run_each(&retained(
        8,
        16,
        "local first = object; object = { v = (first.v + 1) % 1009 }; return first.v + object.v",
    )));
}

#[test]
fn a_cell_written_through_another_closure_is_read_again() {
    // `bump` shares `n` with the closure the loop calls and writes it
    // between that closure's two reads.
    let source = r"
local function make(seed)
  local n = seed
  local function bump() n = (n + 5) % 1013 end
  return function()
    local before = n
    bump()
    return before * 2 + n
  end
end
local closures = {}
for k = 1, 16 do closures[k] = make(k) end
local sum = 0
for i = 1, 40000 do
  sum = (sum + closures[i % 16 + 1]()) % 1000003
end
return sum
";
    // Today the run leaves at each call: a closure that shares a cell
    // another one writes has no row. The result holds either way.
    run_each(source);
}

#[test]
fn a_cell_written_by_a_call_out_of_the_trace_is_read_again() {
    // `load` makes a chunk the trace cannot inline: the interpreter runs
    // the store between the two reads.
    let source = r"
local function make(seed)
  local object = { v = seed }
  local set = (loadstring or load)('local f, t = ...; f(t)')
  local function swap(t) object = t end
  return function(i)
    local first = object
    if i % 211 == 0 then set(swap, { v = i % 97 }) end
    return first.v + object.v
  end
end
local closures = {}
for k = 1, 16 do closures[k] = make(k) end
local sum = 0
for i = 1, 40000 do
  sum = (sum + closures[i % 16 + 1](i)) % 1000003
end
return sum
";
    run_each(source);
}

#[test]
fn more_closures_than_the_memo_or_the_bank_holds_still_agree() {
    // 64 places in the memo, 256 lanes in the bank.
    for closures in [65, 130, 300, 1000] {
        run_each(&retained(
            closures,
            32,
            "object.v = (object.v + 1) % 1009; return object.v + n",
        ));
    }
}

#[test]
fn closures_made_and_dropped_during_the_loop_still_agree() {
    // A closure's handle is taken again by a new closure after a
    // collection: the memo's answer for the old one must not serve it.
    let source = r"
local function make(seed)
  local object = { v = seed }
  return function()
    object.v = (object.v + 1) % 1009
    return object.v
  end
end
local closures = {}
for k = 1, 48 do closures[k] = make(k) end
local sum = 0
for i = 1, 60000 do
  if i % 1500 == 0 then
    for k = 1, 48, 3 do closures[k] = make(i + k) end
    collectgarbage()
  end
  sum = (sum + closures[i % 48 + 1]()) % 1000003
end
return sum
";
    assert_native(&run_each(source));
}

/// Stores a new table into an old list each round, at an index `{INDEX}`
/// gives, with a collection of kind `{COLLECT}` every 997 rounds, and reads
/// every slot back at the end.
fn ring(size: usize, index: &str, collect: &str) -> String {
    format!(
        r"
local size = {size}
local ring = {{}}
for k = 1, size do ring[k] = {{ k, -k }}
end
collectgarbage()
collectgarbage()
local sum = 0
for i = 1, 50000 do
  local made = {{ i, i % 29 }}
  ring[{index}] = made
  if i % 997 == 0 then collectgarbage({collect}) end
  sum = (sum + made[2] + ring[(i * 7) % size + 1][2]) % 1000003
end
local held = 0
for k = 1, size do
  local entry = ring[k]
  held = (held + entry[1] * k + entry[2]) % 1000003
end
return sum, held
"
    )
}

#[test]
fn tables_stored_into_an_old_list_survive_the_collections_between() {
    for collect in ["'step', 0", "'collect'", "'count'"] {
        // Up, down, in strides, and into one slot.
        for index in [
            "i % size + 1",
            "size - i % size",
            "(i * 37) % size + 1",
            "5",
            "i % 3 + 1",
        ] {
            assert_native(&run_each(&ring(256, index, collect)));
        }
    }
}

#[test]
fn tables_stored_into_a_growing_list_survive() {
    // The list's array part moves as it grows, so a slot's address does.
    let source = r"
local list = {}
local sum = 0
for i = 1, 30000 do
  local made = { i % 13, i }
  list[#list + 1] = made
  if i % 4999 == 0 then collectgarbage() end
  sum = (sum + list[(i * 5) % #list + 1][1]) % 1000003
end
local held = 0
for k = 1, #list do held = (held + list[k][2] * 3 + list[k][1]) % 1000003 end
return sum, held, #list
";
    assert_native(&run_each(source));
}

#[test]
fn a_table_and_other_values_stored_into_one_slot_survive() {
    // The slot takes a table, a number, a string and nil in turn: only the
    // table goes the short way.
    let source = r#"
local ring = {}
for k = 1, 64 do ring[k] = { k } end
collectgarbage()
local names = { "a", "bb", "ccc" }
local sum = 0
for i = 1, 60000 do
  local slot = i % 64 + 1
  local kind = i % 5
  if kind == 0 then ring[slot] = { i % 31 }
  elseif kind == 1 then ring[slot] = i
  elseif kind == 2 then ring[slot] = names[i % 3 + 1]
  elseif kind == 3 then ring[slot] = { i % 7, i % 11 }
  else ring[slot] = false end
  if i % 2003 == 0 then collectgarbage() end
  local seen = ring[(i * 3) % 64 + 1]
  if type(seen) == "table" then sum = sum + seen[1]
  elseif type(seen) == "number" then sum = sum + seen % 13
  elseif type(seen) == "string" then sum = sum + #seen end
  sum = sum % 1000003
end
return sum
"#;
    assert_native(&run_each(source));
}

#[test]
fn list_constructors_of_every_short_length_agree() {
    // Lengths 0 to 9 with and without keys, nil among the values, and a
    // length read off each: a table born with its array part has the border
    // the interpreter's has.
    let source = r"
local sum = 0
local kept = {}
for i = 1, 30000 do
  local a = {}
  local b = { i }
  local c = { i, i + 1, i + 2 }
  local d = { i, nil, i + 2 }
  local e = { 1, 2, 3, 4, 5, 6, 7, 8, i }
  local f = { i, i + 1, x = i, y = 2 }
  local g = { x = i }
  local h = { i % 3 == 0 and i or nil, i }
  a[1] = i
  b[2] = i
  c[4] = i
  g[1] = i
  kept[i % 50 + 1] = h
  sum = (sum + #a + #b * 2 + #c * 3 + (d[1] + d[3]) + #e + e[9] + f[1] + f.x + f.y
    + #f + g.x + #g + (h[1] or 0) + h[2] + c[4] + b[2]) % 1000003
end
local held = 0
for k = 1, 50 do held = held + (kept[k][1] or 1) + kept[k][2] end
return sum, held
";
    assert_native(&run_each(source));
}

#[test]
fn a_loop_that_makes_tables_until_the_collector_runs_agrees() {
    // No table is kept: the made tables fill the bank's lanes and the young
    // generation over and over, and the run leaves for each collection.
    let source = r"
local sum = 0
for i = 1, 200000 do
  local made = { i % 17, i % 23, i % 5 }
  sum = (sum + made[1] + made[2] * made[3]) % 1000003
end
return sum
";
    assert_native(&run_each(source));
}
