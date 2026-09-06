//! Tables a trace makes and stores into lists, with collections in between.
//!
//! A trace makes a table that started empty and took string fields with the
//! shape of those fields, where the helper finds the shape and the slots in
//! one call. A table stored into a list takes a write barrier that answers
//! in line when no major cycle runs and the list is young, or is old and
//! already remembers the slot. Each program here gives the interpreter's
//! result under every profile. Several push an input off the short way: a
//! table with an array part, an integer key, more fields than a helper takes,
//! a field stored or removed after the table is made, a store over a string,
//! a list that grows old, and stores while a major cycle is under way.

use oximond::runtime::Value;
use oximond::vm::TraceStatistics;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 4] = [
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua51,
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

/// A loop that makes a table with `{MAKE}` (the table is `t`, the counter
/// `i`), keeps every third one in a ring of 512, and at the end walks the
/// ring with `pairs` and writes down every key and value in the order met.
fn made(make: &str) -> String {
    format!(
        r"
local ring = {{}}
for k = 1, 512 do ring[k] = false end
local sum = 0
for i = 1, 40000 do
  local t = {{}}
  {make}
  if i % 3 == 0 then ring[i % 512 + 1] = t end
  sum = (sum + (t.a or 0) + (t.b or 0)) % 1000003
end
local out = {{}}
for k = 1, 512 do
  local t = ring[k]
  if t then
    for key, value in pairs(t) do
      out[#out + 1] = tostring(key) .. '=' .. (value == t and 'self' or tostring(value))
    end
    out[#out + 1] = ';'
  end
end
return sum, #out, table.concat(out, ',')
"
    )
}

#[test]
fn a_table_of_one_field_keeps_it() {
    assert_native(&run_each(&made("t.a = i")));
}

#[test]
fn a_table_of_several_fields_iterates_as_the_interpreter_does() {
    // Stored out of sorted order, so the order `pairs` gives is the order
    // of the stores and not of the keys.
    assert_native(&run_each(&made("t.b = i % 7; t.zeta = i; t.a = i % 5")));
    assert_native(&run_each(&made(
        "t.b = i % 7; t.a = i % 5; t.c = 'x'; t.d = true; t.e = i * 0.5",
    )));
}

#[test]
fn a_field_that_closes_a_cycle_is_part_of_the_table() {
    assert_native(&run_each(&made("t.a = i; t.me = t")));
    assert_native(&run_each(&made("t.me = t")));
    // Two tables that hold each other.
    assert_native(&run_each(&made(
        "local u = {}; u.a = i % 11; u.up = t; t.a = i; t.down = u",
    )));
}

#[test]
fn a_table_with_an_array_part_or_an_integer_key_stays_plain() {
    assert_native(&run_each(&made("t[1] = i; t.a = i % 9")));
    assert_native(&run_each(&made("t.a = i % 9; t[1] = i; t[2] = i + 1")));
    // Keys a trace may not take at all: the results still agree.
    run_each(&made("t.a = i % 9; t[1000] = i"));
    run_each(&made("t.a = i % 9; t[i % 4 == 0] = i"));
    run_each(&made("t.a = i % 9; t[0.5] = i"));
}

#[test]
fn more_fields_than_a_helper_takes_stay_plain() {
    assert_native(&run_each(&made(
        "t.a = i; t.b = 2; t.c = 3; t.d = 4; t.e = 5; t.f = 6; t.g = 7; t.h = 8",
    )));
}

#[test]
fn a_field_stored_or_removed_after_the_table_is_made_is_seen() {
    // The table escapes into the ring first, then takes a key its shape
    // does not have, loses one it has, and takes one of them back.
    let source = r"
local ring = {}
for k = 1, 64 do ring[k] = false end
local sum = 0
for i = 1, 40000 do
  local t = {}
  t.a = i
  t.b = i % 13
  local slot = i % 64 + 1
  ring[slot] = t
  local held = ring[slot]
  held.late = i % 5
  if i % 2 == 0 then held.a = nil end
  if i % 4 == 0 then held.a = -i end
  if i % 8 == 0 then held[1] = i end
  sum = (sum + (held.a or 7) + held.b + held.late) % 1000003
end
local out = {}
for k = 1, 64 do
  for key, value in pairs(ring[k]) do
    out[#out + 1] = tostring(key) .. '=' .. tostring(value)
  end
  out[#out + 1] = ';'
end
return sum, table.concat(out, ',')
";
    assert_native(&run_each(source));
}

/// A loop that stores a fresh table into `list` each round with `{STORE}`
/// (the table is `t`, the counter `i`, the list holds `n` slots), with
/// `{TUNE}` run first, and then checks every slot.
fn stored(tune: &str, fill: &str, store: &str, rounds: usize) -> String {
    format!(
        r"
{tune}
local n = 700
local list = {{}}
for k = 1, n do list[k] = {fill} end
collectgarbage()
collectgarbage()
local sum = 0
for i = 1, {rounds} do
  local t = {{}}
  t.v = i
  t.w = i % 31
  {store}
  sum = (sum + t.w) % 1000003
end
local bad, tables, total = 0, 0, 0
for k = 1, #list do
  local t = list[k]
  if type(t) == 'table' then
    tables = tables + 1
    if type(t.v) ~= 'number' or t.w ~= t.v % 31 then bad = bad + 1 end
    total = (total + t.v) % 1000003
  end
end
return sum, bad, tables, total
"
    )
}

/// Reads the count of broken tables out of a result of [`stored`].
fn assert_none_bad(source: &str) {
    assert_native(&run_each(source));
    let (results, _) = outcome(CompatibilityProfile::Lua51, ExecutionMode::Tracing, source);
    assert_eq!(
        results.unwrap()[1],
        Value::from(0.0),
        "a stored table broke"
    );
}

#[test]
fn tables_stored_into_an_old_list_survive_minor_collections() {
    // The list is old after the two full collections; every store is of a
    // young table. Up, down and scattered, so the remembered range grows
    // at each end and in the middle.
    assert_none_bad(&stored(
        "",
        "{v = 0, w = 0}",
        "list[i % n + 1] = t",
        120_000,
    ));
    assert_none_bad(&stored(
        "",
        "{v = 0, w = 0}",
        "list[n - i % n] = t",
        120_000,
    ));
    assert_none_bad(&stored(
        "",
        "{v = 0, w = 0}",
        "list[(i * 37) % n + 1] = t",
        120_000,
    ));
}

#[test]
fn tables_stored_over_strings_and_booleans_release_them() {
    // A slot that holds a string releases it on the store; one that holds
    // false owns nothing. Both lists end up full of tables.
    assert_none_bad(&stored("", "'s' .. k", "list[i % n + 1] = t", 60_000));
    assert_none_bad(&stored("", "false", "list[i % n + 1] = t", 60_000));
    // Strings come back between the tables, so the store over a string
    // keeps happening after the trace is compiled.
    assert_none_bad(&stored(
        "",
        "'s' .. k",
        "if i % 3 == 0 then list[i % n + 1] = 'again' .. (i % 50) else list[(i * 7) % n + 1] = t end",
        90_000,
    ));
}

#[test]
fn tables_stored_while_a_major_cycle_runs_survive_it() {
    // A short pause starts major cycles often, and the small step leaves
    // each one under way across many stores: the in-line exit must not
    // answer then. A profile without an option refuses it, hence `pcall`.
    let tune = "pcall(collectgarbage, 'incremental'); pcall(collectgarbage, 'setpause', 100); \
                pcall(collectgarbage, 'setstepmul', 100)";
    assert_none_bad(&stored(
        tune,
        "{v = 0, w = 0}",
        "list[i % n + 1] = t",
        150_000,
    ));
    // The same with explicit steps from the loop, which a trace leaves
    // native code for.
    assert_none_bad(&stored(
        tune,
        "{v = 0, w = 0}",
        "list[(i * 37) % n + 1] = t; if i % 257 == 0 then collectgarbage('step', 1) end",
        150_000,
    ));
    // Full collections between the stores.
    assert_none_bad(&stored(
        "",
        "{v = 0, w = 0}",
        "list[n - i % n] = t; if i % 4099 == 0 then collectgarbage() end",
        100_000,
    ));
}

#[test]
fn a_list_that_grows_old_keeps_the_tables_appended_to_it() {
    // The list starts young and is promoted while the loop appends, so the
    // barrier's answer for it changes under the compiled trace.
    let source = r"
local list, count = {}, 0
local sum = 0
for i = 1, 200000 do
  local t = {}
  t.v = i
  t.w = i % 31
  if i % 5 == 0 then
    count = count + 1
    list[count] = t
  end
  sum = (sum + t.w) % 1000003
end
local bad = 0
for k = 1, count do
  local t = list[k]
  if type(t) ~= 'table' or t.v ~= k * 5 or t.w ~= t.v % 31 then bad = bad + 1 end
end
return sum, count, bad
";
    assert_native(&run_each(source));
    let (results, _) = outcome(CompatibilityProfile::Lua51, ExecutionMode::Tracing, source);
    assert_eq!(results.unwrap()[2], Value::from(0.0));
}

#[test]
fn a_tree_of_made_tables_is_whole_after_many_collections() {
    // Each node is made empty and takes its children as fields: the made
    // tables hold other made tables, and most die young.
    let source = r"
local function build(depth)
  local node = {}
  if depth > 0 then
    node.left = build(depth - 1)
    node.right = build(depth - 1)
  end
  node.depth = depth
  return node
end
local function count(node)
  if node.left then return 1 + count(node.left) + count(node.right) end
  return 1
end
local keep = build(10)
local total = 0
for round = 1, 400 do
  total = total + count(build(6))
  if round % 100 == 0 then keep = build(10) end
end
return total, count(keep), keep.depth, keep.left.left.depth
";
    run_each(source);
}
