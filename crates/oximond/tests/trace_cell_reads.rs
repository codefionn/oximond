//! Upvalue reads in a frame that runs whichever closure of a prototype its
//! call had (Mineclonia's ersatz `index_biomes`, one closure per preset).
//! A run remembers what each cell read answered and answers again from the
//! memo while the cell holds the same value. These loops change the cells
//! between calls: a bump through the closure's own function, a store the
//! interpreter makes, a swap to another table, string or function, and an
//! open cell that the enclosing frame writes. Every program gives the
//! interpreter's result under each profile. Today each of these changes
//! leaves the run, so the memo starts empty after it; the tests hold the
//! results while the recorder learns to keep such changes native.

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

/// Asserts that some profile read cells through the helpers natively.
fn assert_cell_helpers(all: &[TraceStatistics]) {
    assert!(
        all.iter()
            .any(|stats| stats.iterations > 0 && stats.closure_helper_calls > 0),
        "no profile read a cell through the helpers: {:?}",
        all.iter()
            .map(|stats| (stats.iterations, stats.closure_helper_calls))
            .collect::<Vec<_>>()
    );
}

/// Eight presets made by one constructor, each a method that reads a table,
/// a string, a function and a number out of its cells. `{SETUP}` comes
/// before the presets are made and `{STEP}` runs before each call.
fn presets(setup: &str, step: &str) -> String {
    format!(
        r#"
local translations = {{}}
for k, name in ipairs({{ "Plains", "Desert", "Forest", "Taiga" }}) do
  translations[k + 10] = name
end
local others = {{ [11] = "Ocean", [12] = "Desert", [13] = "Swamp", [14] = "Mesa" }}
local v = {{ x = 0, y = 0, z = 0 }}
local function plus(a, b) return a + b end
local function minus(a, b) return a - b end
{setup}
local function make(offset, fallback, op)
  local table_cell, name_cell, op_cell, offset_cell = translations, fallback, op, offset
  return {{
    index = function(self, x, y)
      v.x = x
      v.y = op_cell(y, offset_cell)
      local name = table_cell[11 + x % 4] or name_cell
      return name, v.y
    end,
    set_table = function(t) table_cell = t end,
    set_name = function(s) name_cell = s end,
    set_op = function(f) op_cell = f end,
    bump = function() offset_cell = offset_cell + 1 end,
  }}
end
local presets = {{}}
for k = 1, 8 do
  presets[k] = make(k, "Plains" .. k, k % 2 == 0 and plus or minus)
end
local function run(n)
  local count, sum = 0, 0
  for i = 1, n do
    local preset = presets[i % 8 + 1]
    {step}
    local name, y = preset:index(i, i % 7)
    if name == "Desert" then count = count + 1 end
    sum = (sum + y + #name) % 1000003
  end
  return count, sum
end
return run(20000)
"#
    )
}

#[test]
fn unchanged_cells_read_through_the_helpers_give_the_interpreters_result() {
    let all = run_each(&presets("", ""));
    assert_cell_helpers(&all);
}

#[test]
fn a_cell_the_loop_bumps_natively_is_read_again() {
    // Every 37th trip bumps a number cell through the closure's own
    // function, a store the trace makes itself.
    let all = run_each(&presets("", "if i % 37 == 0 then preset.bump() end"));
    assert_cell_helpers(&all);
}

#[test]
fn a_table_cell_swapped_mid_loop_is_read_again() {
    run_each(&presets(
        "",
        "if i % 1000 == 0 then preset.set_table(i % 2000 == 0 and translations or others) end",
    ));
}

#[test]
fn a_string_cell_swapped_mid_loop_is_read_again() {
    run_each(&presets(
        "",
        r#"if i % 501 == 0 then preset.set_name("Name" .. i % 3) end"#,
    ));
}

#[test]
fn a_function_cell_swapped_mid_loop_is_read_again() {
    run_each(&presets(
        "",
        "if i % 777 == 0 then preset.set_op(i % 2 == 0 and plus or minus) end",
    ));
}

#[test]
fn a_cell_written_by_the_interpreter_is_read_again() {
    // `load` runs the store in a chunk the trace cannot inline, so the
    // interpreter makes it.
    run_each(&presets(
        "local setter = (loadstring or load)('local p, t = ...; p.set_table(t)')",
        "if i % 1013 == 0 then setter(preset, i % 2 == 0 and others or translations) end",
    ));
}

#[test]
fn an_open_cell_the_enclosing_frame_writes_is_read_again() {
    // The closures share `current`, a local of the frame that runs the loop:
    // its cell stays open, and the loop's own register writes reach it.
    let source = r#"
local names = { "a", "bb", "ccc" }
local function run(n)
  local current = names[1]
  local tables = { { 1 }, { 2, 2 } }
  local held = tables[1]
  local fs = {}
  for k = 1, 8 do
    fs[k] = { f = function(self, x) return #current + #held + x * k end }
  end
  local s = 0
  for i = 1, n do
    if i % 97 == 0 then current = names[i % 3 + 1] end
    if i % 131 == 0 then held = tables[i % 2 + 1] end
    local p = fs[i % 8 + 1]
    s = (s + p:f(i)) % 1000003
  end
  return s
end
return run(20000)
"#;
    run_each(source);
}

// ---- one cell reached several times in an iteration ---------------------------
//
// An inlined closure that reads, stores and reads one cell reaches it once:
// the later accesses take the address the first found and a read takes the
// number the trace last read or stored. These programs put between the
// accesses what may make that wrong.

#[test]
fn a_cell_read_stored_and_read_again_in_one_iteration_holds_what_was_stored() {
    run_each(
        r"
local function make(k)
  local count = 0
  return function(x)
    count = count + 1
    local seen = count
    count = count + x % 3
    return (x * k + count + seen) % 1009
  end
end
local g1, g2, g3, g4 = make(1), make(2), make(3), make(4)
local total = 0
for i = 1, 6000 do
  total = (total + g1(i)) % 1000003
  g1, g2, g3, g4 = g2, g3, g4, g1
end
return total
",
    );
}

#[test]
fn closures_that_share_a_cell_see_each_others_stores() {
    // `bump` and `peek` of one pair share `count`; a pair's own `scale` is
    // its alone. The loop calls both of one pair in an iteration, so a read
    // through one closure follows a store through the other.
    run_each(
        r"
local function pair(scale)
  local count = 0
  local function bump(x)
    count = count + x
    return count * scale
  end
  local function peek(x)
    local before = count
    count = count + 1
    return before + count + x
  end
  return bump, peek
end
local b1, p1 = pair(1)
local b2, p2 = pair(2)
local b3, p3 = pair(3)
local total = 0
for i = 1, 6000 do
  local a = b1(i % 5)
  local b = p1(a % 7)
  local c = b1(1)
  total = (total + a + b + c) % 1000003
  b1, p1, b2, p2, b3, p3 = b2, p2, b3, p3, b1, p1
end
return total
",
    );
}

#[test]
fn a_cell_that_changes_type_or_is_written_by_a_call_between_two_reads_is_read_again() {
    // Between the two reads of `count`: a call the trace does not inline
    // that stores a number, one that stores a string, a table or nil, a
    // `pcall`, and a store the closure itself makes with another type.
    run_each(
        r"
local log = 0
local function make(k)
  local count = 0
  local function poke(v) count = v end
  local function step(x, mode)
    local before = count
    if mode == 1 then poke(before + 2)
    elseif mode == 2 then poke('7')
    elseif mode == 3 then pcall(poke, before + 0.5)
    elseif mode == 4 then count = before + 0.25
    elseif mode == 5 then poke(nil) end
    local after = count
    if after == nil then count = 1; after = 1 end
    count = after + 1
    log = log + 1
    return (x * k + before + count) % 1009
  end
  return step
end
local g1, g2, g3 = make(1), make(2), make(3)
local total = 0
for i = 1, 9000 do
  local mode = 0
  if i % 1000 > 900 then mode = i % 6 end
  total = (total + g1(i, mode)) % 1000003
  g1, g2, g3 = g2, g3, g1
end
return total, log
",
    );
}

#[test]
fn a_cell_a_library_callback_writes_between_two_reads_is_read_again() {
    // `string.gsub` and `table.sort` call back into a closure that shares
    // `count`, from library code the trace does not see into.
    run_each(
        r"
local function make(k)
  local count = 0
  local function bump() count = count + k end
  local function less(a, b) count = count + 1; return a < b end
  local list = { 3, 1, 2 }
  local function step(x)
    local before = count
    local _, n = string.gsub('ab', '%a', bump)
    local middle = count
    if x % 16 == 0 then
      list[1], list[2], list[3] = 3, 1, 2
      table.sort(list, less)
    end
    local after = count
    count = after + 1
    return (x * k + before * 3 + middle * 5 + after * 7 + n) % 1009
  end
  return step
end
local g1, g2, g3 = make(1), make(2), make(3)
local total = 0
for i = 1, 6000 do
  total = (total + g1(i)) % 1000003
  g1, g2, g3 = g2, g3, g1
end
return total
",
    );
}
