//! Recording limits at `LuaJIT` scale and upvalue writes of closures in
//! sides: loops that needed more than the old fixed caps (long bodies, many
//! upvalues, globals, fields and sides) compile and run natively with the
//! interpreter's results, a closure iterator that keeps its position in
//! captured locals runs in root and side traces, and the limits of
//! `jit.opt.start` apply per VM.

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

/// Runs `source` in both modes of every profile; the results agree. Returns
/// the statistics of the tracing run under the first profile.
fn run_both(source: &str) -> TraceStatistics {
    run_each(source).remove(0)
}

/// [`run_both`], returning the statistics of the tracing run of every
/// profile.
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

fn limit_aborts(stats: &TraceStatistics) -> u64 {
    stats.aborts(TraceAbortReason::LimitExceeded)
}

#[test]
fn a_loop_with_many_upvalues_and_globals_runs_natively() {
    let mut source = String::new();
    for i in 0..40 {
        writeln!(source, "local u{i} = {i}").unwrap();
    }
    for i in 0..30 {
        writeln!(source, "g{i} = {}", i * 2).unwrap();
    }
    source.push_str("local function run()\n  local s = 0\n  for i = 1, 400 do\n    s = s + i");
    for i in 0..40 {
        write!(source, " + u{i}").unwrap();
    }
    for i in 0..30 {
        write!(source, " + g{i}").unwrap();
    }
    source.push_str("\n  end\n  return s\nend\nreturn run()\n");
    let stats = run_both(&source);
    assert_eq!(limit_aborts(&stats), 0, "{stats:?}");
    assert!(
        stats.traces_compiled > 0 && stats.iterations > 300,
        "{stats:?}"
    );
}

#[test]
fn a_loop_with_many_table_fields_runs_natively() {
    let mut source = String::from("local t = {}\n");
    for i in 0..60 {
        writeln!(source, "t.f{i} = {i}").unwrap();
    }
    source.push_str("local s = 0\nfor i = 1, 20000 do\n  s = s + i");
    for i in 0..60 {
        write!(source, " + t.f{i}").unwrap();
    }
    source.push_str("\n  t.f3 = t.f3 + 1\nend\nreturn s, t.f3\n");
    let stats = run_both(&source);
    assert_eq!(limit_aborts(&stats), 0, "{stats:?}");
    assert!(
        stats.traces_compiled > 0 && stats.iterations > 300,
        "{stats:?}"
    );
}

#[test]
fn a_long_loop_body_runs_natively() {
    let mut source = String::from("local s, k = 0, 3\nfor i = 1, 3000 do\n");
    for j in 0..800 {
        writeln!(source, "  s = s + i * k + {j}").unwrap();
    }
    source.push_str("end\nreturn s\n");
    let stats = run_both(&source);
    assert_eq!(limit_aborts(&stats), 0, "{stats:?}");
    assert!(
        stats.traces_compiled > 0 && stats.iterations > 300,
        "{stats:?}"
    );
}

/// A loop whose body takes one of three arms in turn.
const THREE_ARMS: &str = "
    local s = 0
    for i = 1, 30000 do
      local m = i % 3
      -- The `%` keeps each arm impure, so each branch keeps its guard.
      if m == 0 then s = (s + i) % 1000003 end
      if m == 1 then s = (s - 2 * i) % 1000003 end
      if m == 2 then s = (s + 3) % 1000003 end
    end
    return s
";

#[test]
fn jit_opt_start_maxside_limits_the_sides_of_a_tree() {
    let free = run_both(THREE_ARMS);
    assert!(free.side_traces_recorded >= 2, "{free:?}");
    let source = format!("jit.opt.start('maxside=0')\n{THREE_ARMS}");
    let capped = run_both_with_jit(&source);
    assert_eq!(capped.side_traces_recorded, 0, "{capped:?}");
    assert!(capped.traces_compiled > 0, "{capped:?}");
}

/// [`run_both`] for a source that uses the `jit` table (`LuaJIT` profile only).
fn run_both_with_jit(source: &str) -> TraceStatistics {
    let profile = CompatibilityProfile::LuaJit21;
    let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
    let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
    assert!(expected.is_ok(), "{expected:?}");
    assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
    lua.vm().jit_statistics().trace
}

/// A loop of `body` additions, in a function so that the loop is worth a trace.
fn addition_loop(body: usize) -> String {
    let mut source =
        String::from("local function run()\n  local s, k = 0, 3\n  for i = 1, 3000 do\n");
    for j in 0..body {
        writeln!(source, "    s = s + i * k + {j}").unwrap();
    }
    source.push_str("  end\n  return s\nend\nreturn run()\n");
    source
}

#[test]
fn jit_opt_start_maxrecord_sets_the_recording_limit_of_the_vm() {
    let long = addition_loop(100);
    // The default limit takes a body of 100 statements.
    let free = run_both_with_jit(&long);
    assert_eq!(limit_aborts(&free), 0, "{free:?}");
    assert!(free.traces_compiled > 0, "{free:?}");
    // A lower limit makes the recording give up.
    let low = run_both_with_jit(&format!("jit.opt.start('maxrecord=50')\n{long}"));
    assert!(limit_aborts(&low) > 0, "{low:?}");
    assert_eq!(low.traces_compiled, 0, "{low:?}");
    // Raising it again restores the trace.
    let raised = run_both_with_jit(&format!(
        "jit.opt.start('maxrecord=50')\njit.opt.start('maxrecord=4000')\n{long}"
    ));
    assert_eq!(limit_aborts(&raised), 0, "{raised:?}");
    assert!(raised.traces_compiled > 0, "{raised:?}");
}

#[test]
fn jit_opt_start_maxsnap_hotloop_and_maxtrace_apply() {
    // One guard per statement: a lower snapshot limit aborts.
    let mut guards = String::from("local function run()\n  local s = 0\n  for i = 1, 3000 do\n");
    for j in 0..30 {
        // The `%` keeps the arm impure: a value select has no snapshot.
        writeln!(guards, "    if i > {j} then s = (s + 1) % 1000003 end").unwrap();
    }
    guards.push_str("  end\n  return s\nend\nreturn run()\n");
    let free = run_both_with_jit(&guards);
    assert_eq!(limit_aborts(&free), 0, "{free:?}");
    let low = run_both_with_jit(&format!("jit.opt.start('maxsnap=10')\n{guards}"));
    assert!(limit_aborts(&low) > 0, "{low:?}");
    // A higher hot count starts the recording later: with a loop shorter
    // than the count nothing is recorded.
    let cold = run_both_with_jit(&format!(
        "jit.opt.start('hotloop=100000')\n{}",
        addition_loop(2)
    ));
    assert_eq!(cold.recording_starts, 0, "{cold:?}");
    // Room for one trace: the second loop evicts the first or is turned away.
    let two = format!(
        "{}\nlocal a = run()\nlocal function run2()\n  local s = 0\n  for i = 1, 3000 do s = s + i * 2 end\n  return s\nend\nreturn a + run2()\n",
        addition_loop(2).replace("return run()\n", "")
    );
    let roomy = run_both_with_jit(&two);
    assert_eq!(
        roomy.cache_evictions + roomy.cache_rejections,
        0,
        "{roomy:?}"
    );
    let tight = run_both_with_jit(&format!("jit.opt.start('maxtrace=1')\n{two}"));
    assert!(
        tight.cache_evictions + tight.cache_rejections > 0,
        "{tight:?}"
    );
}

/// A closure iterator in the style of `make_ipos_iterator` of Mineclonia's
/// levelgen: the position lives in captured locals that the closure writes on
/// each call, in another branch each time a row ends.
const IPOS: &str = "
    local function make()
      local x1, x2, y1, y2, z1, z2, iy, iz
      local function iterate()
        local x, y, z = nil, nil, nil
        if iz < z2 then
          iz = iz + 1
          z = iz
          y = iy
          x = x1
        elseif iy < y2 then
          iy = iy + 1
          iz = z1
          z = iz
          y = iy
          x = x1
        elseif x1 < x2 then
          x1 = x1 + 1
          iz = z1
          iy = y1
          z = iz
          y = iy
          x = x1
        end
        return x, y, z
      end
      local function start(ix1, iy1, iz1, ix2, iy2, iz2)
        x1, x2, y1, y2, z1, z2 = ix1, ix2, iy1, iy2, iz1, iz2
        iy, iz = y1, z1 - 1
        return iterate
      end
      return start, iterate
    end
    local ipos, iterate = make()
";

#[test]
fn a_closure_iterator_that_writes_its_upvalues_runs_natively() {
    let source = format!(
        "{IPOS}
        local sum = 0
        for rep = 1, 200 do
          for x, y, z in ipos(0, 0, 0, 5, 5, 60) do
            sum = sum + x * 7 + y * 3 + z
          end
        end
        return sum
        "
    );
    let stats = run_both(&source);
    assert!(
        stats.traces_compiled > 0 && stats.iterations > 1000,
        "{stats:?}"
    );
    // The root and the sides for the other two branches of the iterator
    // write the captured locals natively: no side recording gives up at a
    // `SetUpvalue`.
    assert!(stats.side_traces_recorded >= 2, "{stats:?}");
    assert!(stats.side_installs >= 1, "{stats:?}");
    assert_eq!(stats.side_aborts, 0, "{stats:?}");
    assert_eq!(
        stats.aborts(TraceAbortReason::UnsupportedOperand),
        0,
        "{stats:?}"
    );
}

#[test]
fn an_exit_inside_a_closure_iterator_leaves_its_upvalues_as_the_interpreter_does() {
    // The loop leaves in the middle of a row, in different rows, and the
    // upvalues of the iterator are read back after each break.
    let source = format!(
        "{IPOS}
        local sum = 0
        local seen = 0
        for rep = 1, 300 do
          local stop = rep % 7
          for x, y, z in ipos(0, 0, 0, 4, 4, 40) do
            sum = sum + x * 7 + y * 3 + z
            if x == stop and y == stop % 3 and z == 20 + stop then
              break
            end
          end
          local i = 1
          while true do
            local name, value = debug.getupvalue(iterate, i)
            if not name then break end
            seen = seen * 31 + (tonumber(value) or 0) + i
            seen = seen % 1000003
            i = i + 1
          end
        end
        return sum, seen
        "
    );
    let stats = run_both(&source);
    assert!(
        stats.traces_compiled > 0 && stats.iterations > 1000,
        "{stats:?}"
    );
    assert!(stats.side_installs >= 2, "{stats:?}");
}

/// A loop whose body takes one of sixteen arms in turn: more sides than the
/// old limit of eight per tree.
#[test]
fn a_loop_with_sixteen_arms_holds_more_than_eight_sides() {
    let mut source = String::from("local s = 0\nfor i = 1, 60000 do\n  local m = i % 16\n");
    for k in 0..16 {
        // The `%` keeps the arm impure, so each branch keeps its guard.
        writeln!(
            source,
            "  if m == {k} then s = (s + {} * i) % 1000003 end",
            k + 1
        )
        .unwrap();
    }
    source.push_str("end\nreturn s\n");
    let stats = run_both(&source);
    assert_eq!(limit_aborts(&stats), 0, "{stats:?}");
    assert!(stats.side_installs > 8, "{stats:?}");
    // `maxside` bounds the sides of a tree.
    let capped = run_both_with_jit(&format!("jit.opt.start('maxside=8')\n{source}"));
    assert!(capped.side_installs <= 8, "{capped:?}");
}

/// More loop headers than the old table of 1024 sites: the later ones still
/// compile.
#[test]
fn headers_past_the_old_site_table_still_compile() {
    let mut source = String::from("jit.opt.start('maxtrace=100000')\nlocal t = 0\n");
    for k in 0..1200 {
        writeln!(
            source,
            "t = t + (function() local s = 0 for i = 1, 300 do s = s + i + {k} end return s end)()"
        )
        .unwrap();
    }
    source.push_str("return t\n");
    let stats = run_both_with_jit(&source);
    assert!(stats.traces_compiled > 1_100, "{stats:?}");
}

/// An outer loop around an inner loop of eight trips. Once a recording has
/// met the inner loop, the next one unrolls it, up to 160 guest instructions.
/// The inner loop takes a long arm once in 40 outer iterations, so a recording
/// that happens to see the arm overflows when unrolled, while the nest
/// without unrolling is small.
const ARM_IN_UNROLLED_LOOP: &str = "
    local cells = {}
    for i = 1, 40 do cells[i] = i * 3 end
    local total = 0
    for outer = 1, 20000 do
      local sum = 0
      for i = 1, 8 do
        local c = cells[i]
        sum = (sum + c * 2 + outer) % 100003
        sum = (sum + c * 3 + outer) % 100003
        sum = (sum + c * 4 + outer) % 100003
        if i == 2 and outer % 40 == 0 then
          ARM
        end
      end
      total = (total + sum) % 1000003
    end
    return total
";

/// An outer loop that calls a branching function and runs an inner loop of
/// eight trips. The first recording takes the short arm of the function and
/// fits when unrolled. The side trace for the other arm, which has 60
/// statements, records the arm and then the unrolled loop, and overflows the
/// 240 guest instructions of a side.
const ARM_IN_CALLEE: &str = "
    local cells = {}
    for i = 1, 40 do cells[i] = i * 3 end
    local total = 0
    local function bump(x)
      if x % 2 == 0 then return x + 1 end
      ARM
      return x + 2
    end
    for outer = 1, 20000 do
      local sum = bump(outer)
      for i = 1, 8 do
        local c = cells[i]
        sum = (sum + c * 2 + outer) % 100003
      end
      total = (total + sum) % 1000003
    end
    return total
";

fn with_arm(template: &str, statements: usize) -> String {
    let mut arm = String::new();
    for j in 0..statements {
        write!(arm, "x = (x * {} + 7) % 1009; ", j + 3).unwrap();
    }
    // The loop variant names its accumulator `sum`.
    let arm = if template.contains("local function bump") {
        arm
    } else {
        arm.replace('x', "sum")
    };
    template.replace("ARM", &arm)
}

#[test]
fn an_unrolled_nest_past_the_unroll_limit_restarts_without_unrolling() {
    let all = run_each(&with_arm(ARM_IN_UNROLLED_LOOP, 40));
    // Which recording meets the arm depends on the profile's counters, so
    // the restart is seen in at least one of them. A tree whose outer sides
    // carry the whole nest (the inner root's exits take the work around the
    // loop, see "Nested loops" in the trace module notes) records no outer
    // loop at all, and there is nothing to restart: 155k of the 160k trips
    // run natively.
    assert!(
        all.iter()
            .any(|stats| stats.unroll_restarts > 0 || stats.iterations >= 150_000),
        "{all:?}"
    );
    for stats in all {
        // The overflow is a restart of the recording, not an abort.
        assert_eq!(limit_aborts(&stats), 0, "{stats:?}");
        assert!(
            stats.traces_compiled > 0 && stats.native_entries > 0 && stats.iterations > 0,
            "{stats:?}"
        );
        // A restart is not a start: every start ends in a trace or an abort.
        assert!(
            stats.recording_starts <= stats.traces_recorded + stats.total_aborts(),
            "{stats:?}"
        );
    }
}

#[test]
fn a_side_trace_past_the_unroll_limit_restarts_without_unrolling() {
    let stats = run_both(&with_arm(ARM_IN_CALLEE, 60));
    assert!(stats.unroll_restarts > 0, "{stats:?}");
    assert_eq!(limit_aborts(&stats), 0, "{stats:?}");
    // The retry loop ends: the side restarts a bounded number of times.
    assert!(stats.unroll_restarts < 20, "{stats:?}");
    assert!(
        stats.traces_compiled > 0 && stats.native_entries > 0 && stats.iterations > 100,
        "{stats:?}"
    );
}

#[test]
fn nests_under_the_unroll_limit_do_not_restart() {
    let stats = run_both(&with_arm(ARM_IN_UNROLLED_LOOP, 0));
    assert_eq!(stats.unroll_restarts, 0, "{stats:?}");
    assert_eq!(limit_aborts(&stats), 0, "{stats:?}");
}

/// A loop whose rare arm calls a function with a short loop of `TRIPS` trips
/// and about 17 guest instructions a trip (`digits` in
/// `mineclonia/open_results`). The side for the arm inlines the call and
/// unrolls its loop.
const LOOP_IN_RARE_CALLEE: &str = "
    local cells = {}
    for i = 1, TRIPS do cells[i] = (i * 7) % 10 end
    local function digits(list)
      local total = 0
      for i = 1, #list do
        local c = list[i]
        total = (total * 10 + (c * 3 + i) % 10 + (c + i * 2) % 7) % 1000003
      end
      return total
    end
    local sum = 0
    for i = 1, 24000 do
      sum = (sum + i * 7919) % 1000000007
      if i % 32 == 0 then
        sum = (sum + digits(cells)) % 1000000007
      end
    end
    return sum
";

#[test]
fn a_side_that_unrolls_a_ten_trip_loop_of_a_callee_fits_the_unroll_limit() {
    // Ten trips are about 200 guest instructions with the arm around them.
    // The side is kept, so the arm costs no exit: the loop is entered a few
    // dozen times in 24 000 trips. Without the side each of the 750 arms
    // left the trace and entered the callee's own.
    // Lua 5.4 does not unroll here (its side stops at the inner loop), so
    // the limit is seen under the other two profiles.
    let source = LOOP_IN_RARE_CALLEE.replace("TRIPS", "10");
    for stats in run_each(&source).into_iter().skip(1) {
        assert_eq!(stats.unroll_restarts, 0, "{stats:?}");
        assert_eq!(stats.side_aborts, 0, "{stats:?}");
        assert!(stats.side_traces_recorded > 0, "{stats:?}");
        assert!(stats.native_entries < 100, "{stats:?}");
        assert!(stats.iterations > 23_000, "{stats:?}");
    }
}

#[test]
fn a_side_whose_callee_loop_is_past_the_unroll_limit_keeps_the_two_loops_apart() {
    // 24 trips are past the limit. The side is given up after a bounded
    // number of tries, and the two loops run as traces of their own with the
    // interpreter's results.
    let source = LOOP_IN_RARE_CALLEE.replace("TRIPS", "24");
    for stats in run_each(&source) {
        assert_eq!(stats.side_traces_recorded, 0, "{stats:?}");
        assert!(stats.side_recordings <= 4, "{stats:?}");
        assert_eq!(limit_aborts(&stats), 0, "{stats:?}");
        assert!(stats.traces_compiled >= 2, "{stats:?}");
        assert!(stats.native_entries > 700, "{stats:?}");
        assert!(stats.iterations > 23_000, "{stats:?}");
    }
}
