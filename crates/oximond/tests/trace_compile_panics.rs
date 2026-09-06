//! A trace compile that panics (a codegen bug, such as Cranelift's register
//! allocator panicking on code release builds do not verify) rejects that
//! trace only. The broker catches the panic around the one compile, so the
//! background worker keeps compiling and a synchronous compile returns to the
//! VM. The program's results stay the interpreter's, later loops still run
//! natively, and `compile_panics` counts what happened.

use oximond::runtime::Value;
use oximond::vm::{TraceAbortReason, TraceStatistics};
use oximond::{CompatibilityProfile, ExecutionMode, Lua};
use std::time::{Duration, Instant};

const PROFILES: [CompatibilityProfile; 2] =
    [CompatibilityProfile::Lua54, CompatibilityProfile::LuaJit21];

/// The first loop to get hot is `sum`'s, so its compile is request 0.
const SOURCE: &str = "
local function sum(n)
  local s = 0
  for i = 1, n do s = s + i end
  return s
end
local function squares(n)
  local t = {}
  for i = 1, n do t[i] = i * i end
  local s = 0
  for i = 1, n do s = s + t[i] end
  return s
end
local a = sum(3000)
local b = squares(3000)
local c = sum(4000)
local d = squares(4000)
return a, b, c, d
";

/// A loop that runs natively once its compile lands.
const LATER: &str = "
local s = 0
for i = 1, 20000 do s = s + i % 7 end
return s
";

enum Broker {
    Synchronous,
    Background,
}

fn lua(profile: CompatibilityProfile, mode: ExecutionMode, broker: &Broker) -> Lua {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    match broker {
        Broker::Synchronous => lua.vm_mut().use_synchronous_jit_compilation(),
        Broker::Background => lua.vm_mut().use_background_jit_compilation(),
    }
    lua.vm_mut().ignore_remaining_work();
    lua
}

fn interpreted(profile: CompatibilityProfile, source: &str) -> Vec<Value> {
    lua(profile, ExecutionMode::Interpreter, &Broker::Synchronous)
        .exec(source)
        .unwrap()
}

fn stats(lua: &Lua) -> TraceStatistics {
    lua.vm().jit_statistics().trace
}

#[test]
fn a_synchronous_compile_panic_rejects_one_trace_and_later_ones_compile() {
    for profile in PROFILES {
        let mut traced = lua(profile, ExecutionMode::Tracing, &Broker::Synchronous);
        traced.vm_mut().panic_on_trace_compiles(&[0]);
        let results = traced.exec(SOURCE).unwrap();
        assert_eq!(results, interpreted(profile, SOURCE), "{profile:?}");
        let stats = stats(&traced);
        assert_eq!(stats.compile_panics, 1, "{profile:?}: {stats:?}");
        assert!(
            stats.aborts(TraceAbortReason::CompileRejected) >= 1,
            "{profile:?}: {stats:?}"
        );
        assert!(stats.traces_compiled >= 1, "{profile:?}: {stats:?}");
        assert!(stats.native_entries >= 1, "{profile:?}: {stats:?}");
        assert!(stats.iterations > 1000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_background_compile_panic_leaves_the_worker_compiling() {
    for profile in PROFILES {
        let mut traced = lua(profile, ExecutionMode::Tracing, &Broker::Background);
        traced.vm_mut().panic_on_trace_compiles(&[0]);
        let results = traced.exec(SOURCE).unwrap();
        assert_eq!(results, interpreted(profile, SOURCE), "{profile:?}");
        // The worker answers when the thread scheduler lets it: run a loop
        // until its result is in. A dead worker never answers.
        let expected = interpreted(profile, LATER);
        let started = Instant::now();
        loop {
            assert_eq!(traced.exec(LATER).unwrap(), expected, "{profile:?}");
            let stats = stats(&traced);
            if stats.compile_panics == 1 && stats.traces_compiled >= 1 && stats.native_entries >= 1
            {
                break;
            }
            assert!(
                started.elapsed() < Duration::from_secs(30),
                "{profile:?}: no trace compiled after the panic: {stats:?}"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let stats = stats(&traced);
        assert_eq!(stats.compile_panics, 1, "{profile:?}: {stats:?}");
    }
}

#[test]
fn every_compile_panicking_leaves_the_interpreter_running_the_program() {
    for profile in PROFILES {
        let mut traced = lua(profile, ExecutionMode::Tracing, &Broker::Synchronous);
        traced
            .vm_mut()
            .panic_on_trace_compiles(&(0..1000).collect::<Vec<_>>());
        let results = traced.exec(SOURCE).unwrap();
        assert_eq!(results, interpreted(profile, SOURCE), "{profile:?}");
        let stats = stats(&traced);
        assert_eq!(stats.traces_compiled, 0, "{profile:?}: {stats:?}");
        assert_eq!(stats.native_entries, 0, "{profile:?}: {stats:?}");
        assert!(stats.compile_panics >= 1, "{profile:?}: {stats:?}");
        // A rejected site backs off like any compile rejection: no request
        // loop.
        assert_eq!(
            stats.compile_panics, stats.compile_submissions,
            "{profile:?}: {stats:?}"
        );
        assert!(stats.compile_submissions < 20, "{profile:?}: {stats:?}");
    }
}
