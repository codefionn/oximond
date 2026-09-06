//! Calls inside inlined frames of a callee the recording already guarded:
//! a helper kept in an upvalue (Mineclonia's `load`-made ULL helpers) that
//! one iteration calls many times takes one callee guard, and one snapshot,
//! not one per call. The loops record under a `maxsnap` they would not fit
//! with a guard per call, run natively, and give the interpreter's results,
//! also when the upvalue is rewritten in the middle of an iteration, when the
//! callee comes out of a table field instead, and when the helper raises an
//! error after the guard it did not take.

use oximond::runtime::Value;
use oximond::vm::{TraceAbortReason, TraceStatistics};
use oximond::{CompatibilityProfile, ExecutionMode, Lua};
use std::fmt::Write as _;

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

/// Runs `source` in both modes of `profile`; the results (or errors) agree.
fn run_with(profile: CompatibilityProfile, source: &str) -> TraceStatistics {
    let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
    let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
    assert_eq!(
        format!("{actual:?}"),
        format!("{expected:?}"),
        "{profile:?}"
    );
    lua.vm().jit_statistics().trace
}

/// A loop that calls `step`, which calls the upvalue helper `mix` `calls`
/// times. `extra` goes at the end of the loop body.
fn helper_loop(calls: usize, extra: &str) -> String {
    let mut source = String::from(
        "local function mix(a, b) return (a * 31 + b) % 65521 end\n\
         local function step(x, i)\n",
    );
    for j in 0..calls {
        writeln!(source, "  x = mix(x, i + {j})").unwrap();
    }
    write!(
        source,
        "  return x\nend\n\
         local function run(n)\n  local x = 1\n  for i = 1, n do\n    x = step(x, i)\n{extra}\n  end\n  return x\nend\n\
         return run(4000)\n"
    )
    .unwrap();
    source
}

fn assert_native(profile: CompatibilityProfile, stats: &TraceStatistics) {
    assert!(stats.traces_compiled >= 1, "{profile:?}: {stats:?}");
    assert!(stats.native_entries >= 1, "{profile:?}: {stats:?}");
    assert_eq!(
        stats.aborts(TraceAbortReason::LimitExceeded),
        0,
        "{profile:?}: {stats:?}"
    );
}

#[test]
fn many_calls_of_one_upvalue_helper_run_natively_in_every_profile() {
    let source = helper_loop(40, "");
    for profile in PROFILES {
        let stats = run_with(profile, &source);
        assert_native(profile, &stats);
    }
}

#[test]
fn many_calls_of_one_upvalue_helper_fit_a_small_snapshot_limit() {
    // Forty calls, each with a guard and a snapshot of its own, would not fit
    // twenty snapshots. One guard for the helper leaves room.
    let source = format!("jit.opt.start('maxsnap=20')\n{}", helper_loop(40, ""));
    let stats = run_with(CompatibilityProfile::LuaJit21, &source);
    assert_native(CompatibilityProfile::LuaJit21, &stats);
}

#[test]
fn an_upvalue_rewritten_between_calls_is_guarded_again() {
    // `step` swaps the helper halfway through the iteration on some
    // iterations: the calls after the write go to the other closure.
    let source = "
local function mix(a, b) return (a * 31 + b) % 65521 end
local function other(a, b) return (a * 7 + b * 3) % 65521 end
local current = mix
local function step(x, i)
  x = current(x, i)
  x = current(x, i + 1)
  if i % 97 == 0 then current = other else current = mix end
  x = current(x, i + 2)
  x = current(x, i + 3)
  return x
end
local function run(n)
  local x = 1
  for i = 1, n do x = step(x, i) end
  return x
end
return run(4000)
";
    for profile in PROFILES {
        run_with(profile, source);
    }
}

#[test]
fn a_helper_swapped_by_the_loop_leaves_and_the_results_agree() {
    // The loop itself replaces the helper every 500 iterations, after the
    // trace pinned the first one.
    let source = helper_loop(
        8,
        "    if i % 500 == 0 then mix = function(a, b) return (a + b * 5) % 65521 end end",
    );
    for profile in PROFILES {
        run_with(profile, &source);
    }
}

#[test]
fn a_callee_read_out_of_a_table_field_keeps_a_guard_per_call() {
    // Each call reads `t.f` again: another value for the recording, so each
    // call is guarded, and a field swapped mid-iteration is seen.
    let source = "
local t = {}
function t.f(a, b) return (a * 31 + b) % 65521 end
local function g(a, b) return (a * 3 + b) % 65521 end
local function step(x, i)
  x = t.f(x, i)
  if i % 89 == 0 then t.f, g = g, t.f end
  x = t.f(x, i + 1)
  x = t.f(x, i + 2)
  return x
end
local function run(n)
  local x = 1
  for i = 1, n do x = step(x, i) end
  return x
end
return run(4000)
";
    for profile in PROFILES {
        let stats = run_with(profile, source);
        assert!(stats.traces_compiled >= 1, "{profile:?}: {stats:?}");
    }
}

#[test]
fn an_error_in_a_helper_after_its_first_call_reports_as_the_interpreter() {
    // The fifth call of the helper raises on iteration 3001: the error and
    // its message come out the same with the guard of that call elided.
    let source = "
local function mix(a, b)
  if b == 3005 then error('bad mix ' .. a) end
  return (a * 31 + b) % 65521
end
local function step(x, i)
  x = mix(x, i) x = mix(x, i + 1) x = mix(x, i + 2)
  x = mix(x, i + 3) x = mix(x, i + 4) x = mix(x, i + 5)
  return x
end
local function run(n)
  local x = 1
  for i = 1, n do x = step(x, i) end
  return x
end
local ok, message = pcall(run, 4000)
return ok, message
";
    for profile in PROFILES {
        run_with(profile, source);
    }
}
