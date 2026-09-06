//! Calls of closures read out of a table field whose prototype made several
//! closures (Mineclonia's `placement.chunk_test`, one closure of
//! `build_random_spread_chunk_test` per placement). The call is guarded on
//! the closure, and each closure gets a side of its own: a prototype guard
//! for such calls ran this file's first loop at 1.87 G instructions against
//! 0.97 G with the sides, so the identity guard stays. The loops run
//! natively, and every program gives the interpreter's result under each
//! profile, also when a field takes a closure of another prototype, a
//! native, or a closure whose cell the loop writes.

use oximond::runtime::Value;
use oximond::vm::{TraceGuardKind, TraceStatistics};
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

/// Placements of five closures of one constructor, each with its own
/// spacing and a shared helper read through a cell, called through a field
/// in turn.
const PLACEMENTS: &str = r"
local floor = math.floor
local function mix(a, b) return (a * 31 + b) % 65521 end
local function build(spacing, salt)
  return function(x)
    return mix(floor(x / spacing), salt)
  end
end
local placements = {}
for i = 1, 5 do placements[i] = { test = build(i + 2, i * 7) } end
";

#[test]
fn a_field_call_that_takes_turns_between_closures_of_one_prototype_runs_natively() {
    let source = format!(
        "{PLACEMENTS}
local function run()
  local s = 0
  for i = 1, 20000 do
    local p = placements[i % 5 + 1]
    s = (s + p.test(i)) % 1000003
  end
  return s
end
return run()
"
    );
    for stats in run_each(&source) {
        assert!(stats.iterations > 15_000, "{stats:?}");
        // The sides take the other closures: few iterations leave at the
        // callee guard.
        assert!(
            stats.side_exits_of(TraceGuardKind::Callee) < 100,
            "{stats:?}"
        );
    }
}

#[test]
fn a_tail_call_through_a_field_runs_each_closure_natively() {
    let source = format!(
        "{PLACEMENTS}
local function starts(p, x) return p.test(x) end
local function run()
  local s = 0
  for i = 1, 20000 do
    s = (s + starts(placements[i % 5 + 1], i)) % 1000003
  end
  return s
end
return run()
"
    );
    for stats in run_each(&source) {
        assert!(stats.iterations > 15_000, "{stats:?}");
    }
}

/// A field that takes a closure made in the loop and later a native: the
/// side for the new closure is guarded on its prototype, which native code
/// reads out of the lane's row. The lookup of `p.test` gives a lane the run
/// added, and its row once held what an earlier run left there (the new
/// closure's prototype), so closures of `build` ran `x % 13` (tracing 619102,
/// interpreter 645337, from iteration 15750 on).
#[test]
fn a_field_call_guarded_on_the_prototype_reads_the_row_of_the_closure_it_looked_up() {
    let source = r"
local function build(k)
  return function(x) return x * k end
end
local placements = {}
for i = 1, 5 do placements[i] = { test = build(i) } end
local function run(n)
  local s = 0
  for i = 1, n do
    if i == 10000 then placements[3].test = function(x) return x % 13 end end
    if i == 14000 then placements[2].test = math.abs end
    local p = placements[i % 5 + 1]
    s = (s + p.test(i)) % 1000003
  end
  return s
end
return run(20000)
";
    for stats in run_each(source) {
        assert!(stats.iterations > 15_000, "{stats:?}");
    }
}

#[test]
fn a_field_that_takes_another_callee_mid_loop_gives_the_interpreters_result() {
    // From iteration 10000 on, one placement holds a closure of another
    // prototype; then a native function; then a closure that writes its
    // own cell.
    let source = format!(
        "{PLACEMENTS}
local count = 0
local function run(n, from)
  local s = 0
  for i = from, n do
    if i == 10000 then placements[3].test = function(x) return x % 13 end end
    if i == 14000 then placements[2].test = math.abs end
    if i == 18000 then
      placements[4].test = function(x) count = count + 1; return count end
    end
    local p = placements[i % 5 + 1]
    s = (s + p.test(i)) % 1000003
  end
  return s
end
return run(20000, 1), count
"
    );
    run_each(&source);
}

#[test]
fn a_closure_whose_cell_the_loop_writes_gives_the_interpreters_result() {
    // The loop writes `bias`, a cell every closure of the prototype shares
    // and the window holds.
    let source = r"
local bias = 0
local function build(k)
  return function(x) return x * k + bias end
end
local fs = {}
for i = 1, 4 do fs[i] = { f = build(i) } end
local function run()
  local s = 0
  for i = 1, 20000 do
    bias = i % 7
    s = (s + fs[i % 4 + 1].f(i)) % 1000003
  end
  return s
end
return run(), bias
";
    run_each(source);
}

#[test]
fn a_closure_that_calls_the_function_its_cell_holds_follows_a_swap() {
    // Each closure calls the function its own cell holds: two closures hold
    // `double`, two `triple`, and one swaps its cell's function mid-loop.
    let source = r"
local function double(x) return x * 2 end
local function triple(x) return x * 3 end
local function build(g, k)
  local cell = g
  return {
    f = function(x) return cell(x) + k end,
    set = function(h) cell = h end,
  }
end
local objects = {}
for i = 1, 4 do objects[i] = build(i % 2 == 0 and double or triple, i) end
local function run()
  local s = 0
  for i = 1, 20000 do
    if i == 12000 then objects[2].set(triple) end
    s = (s + objects[i % 4 + 1].f(i)) % 1000003
  end
  return s
end
return run()
";
    run_each(source);
}
