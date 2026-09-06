//! The attribution sampler: samples land on the source line of the hot loop,
//! interpreted or traced; a VM without a collector records nothing; a loop
//! the tracer gave up on reports why, and a compiled one reports covered.

use oximond::vm::attribution::Collector;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};
use std::time::Duration;

fn run(mode: ExecutionMode, attach: bool, source: &str) -> std::sync::Arc<Collector> {
    let collector = Collector::new(Duration::from_micros(500), None);
    let mut lua = Lua::builder()
        .profile(CompatibilityProfile::LuaJit21)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    if attach {
        lua.vm_mut().set_attribution(Some(collector.clone()));
    }
    let chunk = lua.compile("=attr_test", source).unwrap();
    lua.execute(&chunk).unwrap();
    collector
}

// Line 3 is the hot loop; the rest runs once.
const INTERPRETED: &str = "local i = 0
local s = 0
while i < 6000000 do i = i + 1 end
local cold = 0
for i = 1, 10 do cold = cold + i end
return s
";

#[test]
fn interpreted_samples_land_on_the_hot_line() {
    let collector = run(ExecutionMode::Interpreter, true, INTERPRETED);
    let report = collector.snapshot();
    let thread = report.values().next().expect("a thread was sampled");
    assert!(thread.interpreter >= 20, "{}", collector.render());
    assert_eq!(thread.native, 0);
    let hot = collector.line_samples("attr_test", 3);
    assert!(
        hot * 10 >= thread.interpreter * 9,
        "line 3 has {hot} of {}\n{}",
        thread.interpreter,
        collector.render()
    );
    assert_eq!(collector.line_samples("attr_test", 5), 0);
    // The loop is the innermost loop of the samples, and nothing traced it.
    let (key, stat) = thread
        .loops
        .iter()
        .max_by_key(|(_, stat)| stat.innermost)
        .unwrap();
    assert!(key.0.ends_with("attr_test"));
    assert_eq!(stat.header_line, 3);
    assert_eq!(stat.snapshot.state, "never-hot");
}

// The inner loop is hot enough to compile; the outer loop repeats it.
const TRACED: &str = "local t = 0
local s = 0
for r = 1, 6000 do for i = 1, 100000 do s = s + i end end
return s
";

#[test]
fn traced_samples_land_on_the_trace_header() {
    let collector = run(ExecutionMode::Tracing, true, TRACED);
    let report = collector.snapshot();
    let thread = report.values().next().expect("a thread was sampled");
    assert!(thread.native >= 20, "{}", collector.render());
    let (key, stat) = thread
        .loops
        .iter()
        .max_by_key(|(_, stat)| stat.native)
        .unwrap();
    assert!(key.0.ends_with("attr_test"));
    assert_eq!(stat.header_line, 3, "{}", collector.render());
    assert_eq!(stat.snapshot.state, "compiled", "{}", collector.render());
    assert!(stat.snapshot.entries > 0);
    // Most of the time ran in native code.
    assert!(thread.native > thread.interpreter, "{}", collector.render());
}

// The loop on line 8 spends its time in `work`, which has no loop of its
// own. The sampler takes its samples at backedges and calls, so the calls in
// `work` put samples in a frame with no loop.
const NEAREST: &str = "local floor = math.floor
local function work(n)
  local s = floor(n / 3)
  s = s + floor(n / 7)
  return s + floor(n / 11)
end
local a = 0
for i = 1, 2000000 do a = a + work(i) end
return a
";

#[test]
fn a_loop_counts_the_samples_of_the_loopless_calls_it_makes() {
    let collector = run(ExecutionMode::Interpreter, true, NEAREST);
    let report = collector.snapshot();
    let thread = report.values().next().expect("a thread was sampled");
    let (_, stat) = thread
        .loops
        .iter()
        .find(|(key, stat)| key.0.ends_with("attr_test") && stat.header_line == 8)
        .unwrap_or_else(|| panic!("{}", collector.render()));
    // Samples inside `work` count toward the loop as the nearest loop on
    // the stack, not as samples of its own code.
    assert!(stat.nearest >= 20, "{}", collector.render());
    assert!(stat.nearest > stat.innermost * 2, "{}", collector.render());
    assert_eq!(stat.nearest, stat.inclusive, "{}", collector.render());
}

#[test]
fn a_vm_without_a_collector_records_nothing() {
    let collector = run(ExecutionMode::Interpreter, false, INTERPRETED);
    assert!(collector.snapshot().is_empty());
    assert_eq!(collector.line_samples("attr_test", 3), 0);
}

// `debug.getinfo` is a call the tracer cannot follow, so the loop that makes it
// is given up on.
const ABORTING: &str = "local t = 0
local s = 0
for r = 1, 20 do for i = 1, 20000 do local x = debug.getinfo(1) end end
return s
";

#[test]
fn an_aborted_loop_reports_its_reason() {
    let collector = run(ExecutionMode::Tracing, true, ABORTING);
    let report = collector.snapshot();
    let thread = report.values().next().unwrap();
    let (_, stat) = thread
        .loops
        .iter()
        .filter(|(_, stat)| stat.snapshot.state != "never-hot")
        .max_by_key(|(_, stat)| stat.innermost)
        .unwrap_or_else(|| panic!("{}", collector.render()));
    assert_eq!(stat.header_line, 3);
    assert_eq!(stat.snapshot.state, "rejected", "{}", collector.render());
    assert!(
        stat.snapshot.reason.starts_with("UnsupportedOperand at ")
            && stat.snapshot.reason.ends_with("attr_test:3"),
        "{}",
        collector.render()
    );
    assert!(stat.snapshot.attempts > 0);
    assert_eq!(stat.native, 0);
    // The report file text carries the same state.
    let text = collector.render();
    assert!(
        text.lines().any(|line| line.starts_with("loop\t")
            && line.contains("\trejected\tUnsupportedOperand at ")),
        "{text}"
    );
}

// The loop at line 5 aborts in `f`, a call its recording inlined: the reason
// names the callee's line, read from the callee's chunk, not the line the
// abort's pc would have in the loop's chunk.
const ABORTING_IN_CALLEE: &str = "local function f(x)
  return debug.getinfo(1) and x
end
local s = 0
for r = 1, 20 do for i = 1, 20000 do s = s + f(i) end end
return s
";

#[test]
fn an_abort_in_an_inlined_call_names_the_callee_line() {
    let collector = run(ExecutionMode::Tracing, true, ABORTING_IN_CALLEE);
    let report = collector.snapshot();
    let thread = report.values().next().unwrap();
    let (_, stat) = thread
        .loops
        .iter()
        .filter(|(_, stat)| stat.header_line == 5 && stat.snapshot.state != "never-hot")
        .max_by_key(|(_, stat)| stat.innermost)
        .unwrap_or_else(|| panic!("{}", collector.render()));
    assert!(
        stat.snapshot.reason.ends_with("attr_test:2"),
        "{}",
        collector.render()
    );
}

// A tick makes the fast loop leave at a backedge. A loop instruction that has
// already stepped its registers must not be run again from the general loop:
// every iteration runs exactly once, with the sampler ticking as fast as the
// timer thread can.
const LOOPS: &str = "local n, m, p, q = 0, 0, 0, 0
for i = 1, 3000000 do n = n + i end
local t = {}
for i = 1, 300000 do t[i] = i end
for _, v in ipairs(t) do m = m + v end
for k, v in pairs(t) do p = p + k + v end
for i = 1, 3000 do for j = 1, 300 do q = q + 1 end end
return n, m, p, q
";

#[test]
fn sampling_runs_every_loop_iteration_once() {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let collector = Collector::new(Duration::from_micros(5), None);
        let mut lua = Lua::builder()
            .profile(CompatibilityProfile::LuaJit21)
            .execution_mode(mode)
            .build()
            .unwrap();
        lua.vm_mut().use_synchronous_jit_compilation();
        lua.vm_mut().set_attribution(Some(collector.clone()));
        let chunk = lua.compile("=loops", LOOPS).unwrap();
        let results = lua.execute(&chunk).unwrap();
        let numbers: Vec<f64> = results
            .iter()
            .map(|value| match value {
                oximond::Value::Number(number) => *number,
                oximond::Value::Integer(number) => *number as f64,
                other => panic!("{other:?}"),
            })
            .collect();
        let sum = |n: f64| n * (n + 1.0) / 2.0;
        assert_eq!(
            numbers,
            [
                sum(3_000_000.0),
                sum(300_000.0),
                2.0 * sum(300_000.0),
                900_000.0
            ],
            "{mode:?}"
        );
        let samples: u64 = collector
            .snapshot()
            .values()
            .map(|t| t.interpreter + t.native)
            .sum();
        assert!(samples > 10, "{mode:?}: only {samples} samples");
    }
}

// `f` has no loop; its calls come from the loop on line 3.
const CALLER: &str = "local abs = math.abs
local function f(x) return abs(x) + 1 end
local s = 0
for i = 1, 3000000 do s = f(s) end
return s
";

#[test]
fn a_function_without_a_loop_names_the_loop_that_calls_it() {
    let collector = run(ExecutionMode::Interpreter, true, CALLER);
    let report = collector.snapshot();
    let thread = report.values().next().unwrap();
    let (_, f) = thread
        .functions
        .iter()
        .find(|((_, defined), _)| *defined == 2)
        .unwrap_or_else(|| panic!("{}", collector.render()));
    assert!(f.own > 0, "{}", collector.render());
    let via = f
        .via
        .as_ref()
        .unwrap_or_else(|| panic!("{}", collector.render()));
    let caller = &thread.loops[via];
    assert_eq!(caller.header_line, 4);
    // The loop counts the time of the function it calls.
    assert!(caller.inclusive >= f.own);
}

// One arm of the branch is rare enough that the trace leaves at its guard
// before a side trace takes the arm over, and every pass ends the loop.
// In `EXITING`, the `%` keeps the arms impure, so the branch keeps its guard instead of becoming a value select.
const EXITING: &str = "local s = 0
for r = 1, 4000 do
  for i = 1, 200 do
    if (i * r) % 3 == 0 then s = (s + 1) % 1000003 else s = (s - 1) % 1000003 end
  end
end
return s
";

#[test]
fn a_compiled_loop_reports_its_interpreter_exits_by_kind_and_guard() {
    let collector = run(ExecutionMode::Tracing, true, EXITING);
    let report = collector.snapshot();
    let thread = report.values().next().unwrap();
    let text = collector.render();
    let (_, stat) = thread
        .loops
        .iter()
        .filter(|(_, stat)| stat.snapshot.entries > 0)
        .max_by_key(|(_, stat)| stat.snapshot.entries)
        .unwrap_or_else(|| panic!("{text}"));
    let kinds = &stat.snapshot.exit_kinds;
    assert!(kinds.contains("LoopComplete/Branch="), "{text}");
    assert!(kinds.contains("GuardFail/Branch="), "{text}");
    // The report line carries them too.
    assert!(
        text.lines()
            .any(|line| line.starts_with("loop\t") && line.contains("GuardFail/Branch=")),
        "{text}"
    );
}

/// A host that spends its time in its own code between calls of a small Lua
/// function (Luanti converting positions through `read_vector`): the ticks
/// that pass in host code are seen when the host next calls in, and they
/// are host samples, not samples of the function it calls.
#[test]
fn ticks_that_pass_in_host_code_are_host_samples() {
    let collector = Collector::new(Duration::from_micros(500), None);
    let mut lua = Lua::builder()
        .profile(CompatibilityProfile::LuaJit21)
        .execution_mode(ExecutionMode::Tracing)
        .build()
        .unwrap();
    lua.vm_mut().set_attribution(Some(collector.clone()));
    lua.exec("function read_vector(v) return v.x, v.y, v.z end")
        .unwrap();
    let read_vector = lua.vm_mut().runtime().get_global("read_vector");
    let table = lua.vm_mut().runtime_mut().create_table().unwrap();
    let vector = oximond::Value::Table(table);
    for _ in 0..200 {
        // Host work: a little longer than a period, well under the stale
        // bound of four.
        let start = std::time::Instant::now();
        while start.elapsed() < Duration::from_micros(700) {
            std::hint::spin_loop();
        }
        lua.vm_mut()
            .call_value(read_vector.clone(), vec![vector.clone()])
            .unwrap();
    }
    let report = collector.snapshot();
    let thread = report.values().next().expect("a thread was sampled");
    assert!(thread.host >= 50, "{}", collector.render());
    // The function itself runs four instructions per call.
    assert!(
        thread.interpreter * 10 <= thread.host,
        "{}",
        collector.render()
    );
}
