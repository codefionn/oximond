//! Registers that hold nil on some entries of a loop and a number on others.
//! Mineclonia's shape code tests `lx and ly and ...` in its z loop, where
//! `lx` is nil on the last x of a region: a root typed for one of the two
//! left at entry every time the other came along, and retired. The root now
//! takes such a register untyped, guards its type where the body reads it,
//! and a side runs the iterations of the other type. The results are the
//! interpreter's in every profile, and the loop runs natively.

use oximond::runtime::Value;
use oximond::vm::TraceExitKind;
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
    lua.vm_mut().take_untyped_trace_inputs();
    let results = lua.exec(source).map_err(|error| format!("{error:?}"));
    (results, lua)
}

/// What a traced run did.
struct Seen {
    profile: CompatibilityProfile,
    side_installs: u64,
    outer_sides: u64,
    native_entries: u64,
    entry_mismatches: u64,
    iterations: u64,
    version_installs: u64,
    stats: String,
}

/// Runs `source` traced and interpreted in every profile, checks that the
/// outcomes agree, and returns what each traced run did.
fn run_all(source: &str) -> Vec<Seen> {
    let mut all = Vec::new();
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        all.push(Seen {
            profile,
            side_installs: stats.side_installs,
            outer_sides: stats.outer_sides,
            native_entries: stats.native_entries,
            entry_mismatches: stats.exits_of(TraceExitKind::EntryMismatch),
            iterations: stats.iterations,
            version_installs: stats.version_installs,
            stats: format!("{stats:?}"),
        });
    }
    all
}

/// The nest of Mineclonia's `region_evaluate`: `dom` gives `lx`, `rx`, `ly`
/// and `ry` for each x and y, and the z loop reads them. Each is nil on its
/// own x or y, so the four make more type combinations than a loop has root
/// versions. `total` starts as a float, so that an integer sum turning
/// float in the later profiles does not ask for versions of its own. 300
/// runs of 4 x 3 x (20..24) z trips.
fn nest(dom: &str) -> String {
    format!(
        "
local function dom(i, n, k, call)
{dom}
end
local function run(nx, ny, nz, call)
  local total = 0.0
  local x = 1
  while x <= nx do
    local lx, rx = dom(x, nx, 0, call), dom(x, nx, call, call)
    local y = 1
    while y <= ny do
      local ly, ry = dom(y, ny, 1, call), dom(y, ny, call + 1, call)
      local z = 1
      while z <= nz do
        local l_on = lx and ly and (lx * 3 + ly - z)
        local r_on = rx and ry and (rx * 2 - ry + z)
        if l_on then total = total + l_on else total = total + 1 end
        if r_on then total = total + r_on else total = total - 1 end
        z = z + 1
      end
      y = y + 1
    end
    x = x + 1
  end
  return total
end
local sum = 0
for call = 1, 300 do sum = sum + run(4, 3, 20 + call % 5, call) end
return sum
"
    )
}

/// The z trips of a [`nest`] run.
const TRIPS: u64 = 300 * 4 * 3 * 22;

/// Checks that the z loop ran natively: sides took the other type, and
/// almost no entry left for a type the root did not expect.
fn assert_native(seen: &[Seen]) {
    for run in seen {
        assert!(run.side_installs > 0, "{:?} {}", run.profile, run.stats);
        assert!(
            run.entry_mismatches < 100,
            "{:?} {}",
            run.profile,
            run.stats
        );
        assert!(
            run.iterations * 10 >= TRIPS * 8,
            "{:?} {}",
            run.profile,
            run.stats
        );
        // The four registers make 16 type combinations against
        // `MAX_TRACE_VERSIONS` of 4, and typed versions churned through
        // them. Now a new version comes only when the untyped list grows,
        // at most once per register.
        assert!(run.version_installs <= 4, "{:?} {}", run.profile, run.stats);
    }
}

#[test]
fn registers_nil_on_their_own_trips_run_natively_and_agree_with_the_interpreter() {
    let seen = run_all(&nest(
        "  if (i + k) % n == 0 then return nil end
  return i",
    ));
    assert_native(&seen);
}

#[test]
fn a_register_that_turns_from_float_to_nil_and_back_between_entries_agrees() {
    // The x that gives nil moves from call to call, and the numbers are
    // floats in every profile.
    let seen = run_all(&nest(
        "  if (i + k + call) % 3 == 0 then return nil end
  return i * 0.5",
    ));
    assert_native(&seen);
}

#[test]
fn a_root_recorded_with_nil_takes_a_float_through_its_side() {
    // The first entries see nil, so the root is recorded for nil.
    let seen = run_all(&nest(
        "  if call < 40 or (i + k) % n == 0 then return nil end
  return i + 0.25",
    ));
    for run in &seen {
        assert!(run.side_installs > 0, "{:?} {}", run.profile, run.stats);
    }
}

#[test]
fn a_register_that_flips_between_integer_float_and_nil_agrees() {
    // Integers and floats are one type in Lua 5.1 and 5.2 and two in the
    // later profiles.
    run_all(&nest(
        "  if (i + k) % n == 0 then return nil end
  if (i + call) % 2 == 0 then return i + 0.5 end
  return i",
    ));
}

#[test]
fn a_register_that_is_sometimes_a_string_or_a_boolean_agrees() {
    // A string coerces in arithmetic; `false` takes the else branch. Neither
    // is a nil-or-number register, so these entries leave the tree.
    run_all(&nest(
        "  if (i + k) % n == 0 then return nil end
  if call % 7 == 0 then return tostring(i) end
  if call % 11 == 0 then return false end
  return i",
    ));
}

#[test]
fn a_register_written_in_the_loop_after_a_nil_entry_agrees() {
    // `lx` changes inside the z loop, so it is a carried register, not one
    // the root takes untyped.
    run_all(
        "
local total = 0
for call = 1, 200 do
  local lx = (call % 3 == 0) and nil or call * 0.5
  local z = 1
  while z <= 30 do
    if lx then total = total + lx; lx = (z % 4 == 0) and nil or lx + 1
    else total = total - 1; lx = (z % 5 == 0) and z or nil end
    z = z + 1
  end
end
return total
",
    );
}

#[test]
fn an_error_through_a_nil_register_matches_the_interpreter() {
    // The arithmetic on nil raises once the tree runs natively.
    run_all(
        "
local function run(lx)
  local t = 0
  for z = 1, 50 do t = t + (lx or 0) * z end
  return t
end
local sum = 0
for call = 1, 120 do sum = sum + run((call % 4 ~= 0) and call or nil) end
local ok, err = pcall(function()
  local lx = nil
  local t = 0
  for z = 1, 50 do if z > 40 then t = t + lx * z else t = t + z end end
  return t
end)
return sum, ok
",
    );
}

/// The `y` loop of [`nest`] computes `ly` and `ry` again for every `y`,
/// and the `x` loop `lx` and `rx`, all of which the root of the `z` loop
/// takes untyped. The outer sides that run those loops and link back to
/// the `z` loop write them into the root's input slots, so the tree carries
/// the whole nest: far fewer entries than `y` trips, every `z` trip native.
#[test]
fn outer_sides_that_write_untyped_registers_link_and_carry_the_nest() {
    let seen = run_all(&nest(
        "  if (i + k) % n == 0 then return nil end
  return i",
    ));
    assert_native(&seen);
    for run in &seen {
        assert!(run.outer_sides > 0, "{:?} {}", run.profile, run.stats);
        // 300 calls of 4 x 3 `y` trips each enter the root once without
        // outer sides.
        assert!(
            run.native_entries * 2 < 300 * 4 * 3,
            "{:?} {}",
            run.profile,
            run.stats
        );
        assert!(run.iterations >= TRIPS, "{:?} {}", run.profile, run.stats);
    }
}

#[test]
fn an_outer_side_that_writes_false_into_an_untyped_register_agrees() {
    // `false` has a tag of its own: the link stores it, the root's type
    // guard leaves for it, and the exit writes it back into the frame.
    let seen = run_all(&nest(
        "  if (i + k) % n == 0 then return nil end
  if call > 150 and (i + call) % 5 == 0 then return false end
  return i",
    ));
    for run in &seen {
        assert!(run.side_installs > 0, "{:?} {}", run.profile, run.stats);
    }
}

#[test]
fn an_error_on_a_register_an_outer_side_made_nil_matches_the_interpreter() {
    // The outer side links with `ly` nil and the `z` loop does arithmetic
    // on it without a test: the error and its message are the interpreter's.
    run_all(
        "
local function dom(i, n) if i % n == 0 then return nil end return i end
local function run(ny, nz, bad)
  local total = 0
  local y = 1
  while y <= ny do
    local ly = dom(y, ny)
    local z = 1
    while z <= nz do
      if ly or not bad then total = total + (ly or 0) * z else total = total + ly end
      z = z + 1
    end
    y = y + 1
  end
  return total
end
local sum = 0
for call = 1, 200 do sum = sum + run(5, 6 + call % 3, false) end
local ok, err = pcall(run, 5, 7, true)
return sum, ok, err
",
    );
}

/// `lz` is nil on some entries of the `z` loop and a number on others, and
/// both the root (which counts it down) and the outer side (which sets it
/// again for each `y`) write it, so the root carries it as a loop
/// parameter of one type. The outer side still links where the types
/// agree, and the results are the interpreter's.
#[test]
fn a_register_both_the_root_and_an_outer_side_write_keeps_linking() {
    let seen = run_all(
        "
local function dom(i, n) if i % n == 0 then return nil end return i * 1.5 end
local function run(ny, nz)
  local total = 0.0
  local y = 1
  while y <= ny do
    local lz = dom(y, ny)
    local z = 1
    while z <= nz do
      if lz then total = total + lz; lz = lz - 0.5 else total = total - 1 end
      z = z + 1
    end
    y = y + 1
  end
  return total
end
local sum = 0.0
for call = 1, 300 do sum = sum + run(6, 20 + call % 4) end
return sum
",
    );
    for run in &seen {
        assert!(run.side_installs > 0, "{:?} {}", run.profile, run.stats);
    }
}

/// A loop whose untyped register turns into a table with a `__mul`
/// metamethod on later calls: the type guard leaves for the table, and the
/// interpreter (or a side) calls the metamethod. Before that the loop runs
/// natively with nil and numbers.
#[test]
fn a_register_that_becomes_a_table_with_a_metamethod_leaves_and_agrees() {
    let seen = run_all(
        "
local weird = setmetatable({}, {__mul = function(a, b) return 2 * b end})
local function run(lx, nz)
  local t = 0.0
  local z = 1
  while z <= nz do
    if lx then t = t + lx * z else t = t - z end
    z = z + 1
  end
  return t
end
local sum = 0.0
for call = 1, 400 do
  local lx
  if call > 300 and call % 3 == 0 then lx = weird
  elseif call % 2 == 0 then lx = nil
  else lx = call * 0.5 end
  sum = sum + run(lx, 20 + call % 4)
end
return sum
",
    );
    for run in &seen {
        assert!(run.native_entries > 0, "{:?} {}", run.profile, run.stats);
        assert!(run.iterations > 0, "{:?} {}", run.profile, run.stats);
    }
}

/// Nil gains a `__mul` metamethod halfway through: the side that runs the
/// nil entries was recorded without one, and arithmetic on nil now calls it.
/// The results are the interpreter's.
#[test]
fn a_metatable_on_nil_that_appears_mid_run_agrees() {
    let seen = run_all(
        "
local function run(lx, nz, any)
  local t = 0.0
  local z = 1
  while z <= nz do
    if lx or any then t = t + lx * z else t = t - z end
    z = z + 1
  end
  return t
end
local sum = 0.0
for call = 1, 400 do
  local any = call > 250
  if call == 251 then
    debug.setmetatable(nil, {__mul = function(a, b) return 3 * b end})
  end
  local lx = (call % 3 == 0) and nil or call * 0.25
  sum = sum + run(lx, 20 + call % 5, any)
end
debug.setmetatable(nil, nil)
return sum
",
    );
    for run in &seen {
        assert!(run.native_entries > 0, "{:?} {}", run.profile, run.stats);
    }
}
