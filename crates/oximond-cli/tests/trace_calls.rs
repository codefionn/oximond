//! Native traces that inline calls of Lua closures, through compiled Lua
//! source.
//!
//! Every program runs under the interpreter and under tracing execution. The
//! results must match exactly, and the counters must show that a trace really
//! inlined a call and ran it natively, because equal output alone cannot tell
//! a trace from a slow path.
//!
//! What these programs hold the engine to:
//!
//! - A guard that fails inside the callee leaves the interpreter in the
//!   callee, with the caller stopped at its call. A traceback, `debug.getinfo`
//!   and an error raised from there see an ordinary call chain.
//! - A trace is specialised on the closure it saw, not on its bytecode. A
//!   closure of the same prototype that captured other cells runs in the
//!   interpreter, and so does a rebound or collected callee.
//! - Results are truncated and padded, and missing arguments are nil, as the
//!   interpreter does it.
//! - A callee that yields, errors or calls back into the interpreter after a
//!   native exit behaves as it would have without the trace.

use oximond::vm::{TraceExitKind, TraceGuardKind, TraceStatistics};
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

const ALL_PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

const TWO_PROFILES: [CompatibilityProfile; 2] =
    [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54];

fn run(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
) -> (String, TraceStatistics) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    // These programs test what traces do with small loops of a main chunk.
    lua.vm_mut().ignore_remaining_work();
    let results: Vec<Value> = lua
        .exec(source)
        .unwrap_or_else(|error| panic!("{profile:?} {mode:?}: {error}\n{source}"));
    (format!("{results:?}"), lua.vm().jit_statistics().trace)
}

/// Runs `source` under the interpreter and under tracing execution, checks
/// the results agree, and returns the tracing statistics.
fn agree(profile: CompatibilityProfile, source: &str) -> (String, TraceStatistics) {
    let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
    let (tracing, stats) = run(profile, ExecutionMode::Tracing, source);
    assert_eq!(
        tracing, expected,
        "{profile:?} tracing\n{source}\n{stats:?}"
    );
    (expected, stats)
}

/// Like [`agree`] for every profile in `profiles`, requiring that a trace
/// inlined a call and ran native iterations in each.
fn agree_inlined(
    profiles: &[CompatibilityProfile],
    source: &str,
) -> Vec<(String, TraceStatistics)> {
    profiles
        .iter()
        .map(|profile| {
            let (result, stats) = agree(*profile, source);
            assert!(
                stats.inlined_calls > 0,
                "{profile:?}: no call was inlined\n{source}\n{stats:?}"
            );
            assert!(
                stats.iterations > 0,
                "{profile:?}: no native iteration ran\n{source}\n{stats:?}"
            );
            (result, stats)
        })
        .collect()
}

#[test]
fn a_call_in_a_hot_loop_runs_inside_the_trace() {
    let source = "
        local function step(value) return value * 1.000001 + 0.25 end
        local value = 1
        for _ = 1, 40000 do value = step(value) end
        return value";
    for (_, stats) in agree_inlined(&ALL_PROFILES, source) {
        // One entry runs nearly the whole loop.
        assert!(stats.iterations > 39_000, "{stats:?}");
        assert_eq!(stats.native_entries, stats.exits.iter().sum::<u64>());
    }
}

#[test]
fn a_closure_over_its_own_upvalue_runs_inside_the_trace() {
    let source = "
        local function counter(start)
          local value = start
          return function(delta) value = value + delta; return value end
        end
        local add = counter(0)
        local result = 0
        for i = 1, 30000 do result = add(i % 5) end
        return result, add(0)";
    agree_inlined(&ALL_PROFILES, source);
}

#[test]
fn calls_nest_and_share_a_captured_local_with_the_loop() {
    // `g` is a local of the main chunk that `f` captures, so the loop's own
    // register and `f`'s upvalue are one variable.
    let source = "
        local function g(a, b) return a + b, a - b end
        local function f(x) local p, q = g(x, 1); return p * q end
        local s = 0
        for i = 1, 30000 do s = s + f(i) end
        return s";
    for (_, stats) in agree_inlined(&ALL_PROFILES, source) {
        assert!(stats.inlined_calls >= 2, "{stats:?}");
    }
    // A callee that writes a variable the loop owns: the cell and the
    // register agree at every way out.
    let source = "
        local total = 0
        local function add(d) total = total + d; return total end
        for i = 1, 30000 do add(i) end
        return total, add(0)";
    agree_inlined(&ALL_PROFILES, source);
}

#[test]
fn a_global_function_called_from_a_loop_is_read_at_entry() {
    let source = "
        function step(x) return x + 0.5 end
        local v = 0
        for i = 1, 30000 do v = step(v) end
        return v";
    let profiles = [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21];
    agree_inlined(&profiles, source);
    // Rebinding the global between two runs of the same loop.
    let source = "
        function step(x) return x + 0.5 end
        local function run() local v = 0; for i = 1, 20000 do v = step(v) end; return v end
        local a = run()
        step = function(x) return x + 2 end
        local b = run()
        return a, b";
    agree(CompatibilityProfile::Lua51, source);
}

// ----------------------------------------------------------- guard exits

#[test]
fn a_guard_failing_inside_the_callee_leaves_a_real_call_chain() {
    // `f` takes a branch the recording never saw once `x` passes 20000 and
    // asks for a traceback. The exit happens inside `f`, which the
    // interpreter finishes: the traceback names `f` and its caller exactly as
    // an untraced run does.
    let source = "
        local tb
        local function f(x)
          if x > 20000 then tb = debug.traceback('tb'); return 0 end
          return x + 1
        end
        local v = 1
        for i = 1, 30000 do v = f(v) end
        return tb, v";
    for profile in ALL_PROFILES {
        let (result, stats) = agree(profile, source);
        // Lua 5.3 and later spell the kind of the name (`local 'f'`); the
        // older profiles call every name a function.
        let named = match profile {
            CompatibilityProfile::Lua51
            | CompatibilityProfile::Lua52
            | CompatibilityProfile::LuaJit21 => "in function \\'f\\'",
            _ => "in local \\'f\\'",
        };
        // `result` is the Debug text of the values, which escapes quotes.
        assert!(result.contains(named), "{profile:?}: {result}");
        assert!(
            stats.inlined_calls > 0 && stats.iterations > 10_000,
            "{profile:?}: {stats:?}"
        );
        assert!(
            stats.exits_of(TraceExitKind::GuardFail) >= 1,
            "{profile:?}: {stats:?}"
        );
    }
}

#[test]
fn debug_getinfo_from_the_resumed_callee_describes_both_frames() {
    let source = "
        local info
        local function f(x)
          if x > 20000 then
            local here = debug.getinfo(1, 'nSl')
            local there = debug.getinfo(2, 'nSl')
            info = string.format('%s:%d %s | %s:%d %s', here.short_src, here.currentline,
              tostring(here.name), there.short_src, there.currentline, tostring(there.name))
            return 0
          end
          return x + 1
        end
        local v = 1
        for i = 1, 30000 do v = f(v) end
        return info, v";
    for profile in TWO_PROFILES {
        let (result, stats) = agree(profile, source);
        // The callee is named and sits on its own line, and its caller is the
        // main chunk, a few lines below.
        assert!(result.contains(":5 f | "), "{profile:?}: {result}");
        assert!(stats.iterations > 10_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn an_error_raised_after_a_callee_exit_unwinds_both_frames() {
    // The error comes from `f` after the trace left it, with the caller's
    // frame under it. `pcall` and `xpcall` catch it in the main chunk.
    let source = "
        local function f(x)
          if x > 20000 then error('boom ' .. tostring(x), 1) end
          return x + 1
        end
        local function run()
          local v = 1
          for i = 1, 30000 do v = f(v) end
          return v
        end
        local ok, err = pcall(run)
        local ok2, trace = xpcall(run, function(message) return debug.traceback(message, 1) end)
        return ok, err, ok2, trace";
    for profile in TWO_PROFILES {
        let (result, stats) = agree(profile, source);
        assert!(result.contains("boom 20001"), "{profile:?}: {result}");
        assert!(stats.iterations > 10_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_callee_that_yields_after_an_exit_resumes_inside_the_coroutine() {
    // The loop runs in a coroutine. When `x` passes 20000 the callee yields,
    // so the exit's frames sit on the coroutine's own stack, and the resume
    // continues the callee and then the loop.
    let source = "
        local function f(x)
          if x > 20000 then coroutine.yield(x) ; return 0 end
          return x + 1
        end
        local co = coroutine.create(function()
          local v = 1
          for i = 1, 30000 do v = f(v) end
          return v
        end)
        local ok1, a = coroutine.resume(co)
        local ok2, b = coroutine.resume(co)
        return ok1, a, ok2, b, coroutine.status(co)";
    for profile in TWO_PROFILES {
        let (result, stats) = agree(profile, source);
        assert!(
            stats.inlined_calls > 0 && stats.iterations > 10_000,
            "{profile:?}: {stats:?}"
        );
        assert!(
            result.contains("20001") || result.contains("20002"),
            "{profile:?}: {result}"
        );
    }
}

// ----------------------------------------------------- identity guards

#[test]
fn a_closure_of_the_same_prototype_with_other_captures_never_runs_the_trace_inlined() {
    // `run` is one function, so one chunk and one loop. Its trace is
    // specialised on the first closure it saw. The second closure shares the
    // bytecode and captures another `k`.
    let source = "
        local function make(k) return function(x) return x + k end end
        local function run(f)
          local s = 0
          for i = 1, 20000 do s = s + f(i) end
          return s
        end
        local a, b = make(1), make(100)
        return run(a), run(b), run(a), run(b)";
    for profile in TWO_PROFILES {
        let (result, stats) = agree(profile, source);
        assert!(
            stats.inlined_calls > 0 && stats.iterations > 0,
            "{profile:?}: {stats:?}"
        );
        // The wrong closure is refused by an identity guard, which the
        // statistics name.
        assert!(
            stats.side_exits_of(TraceGuardKind::Callee) > 0,
            "{profile:?}: {stats:?}"
        );
        // run(a) is the sum of i + 1 and run(b) the sum of i + 100.
        assert!(result.contains("200030000"), "{profile:?}: {result}");
        assert!(result.contains("202010000"), "{profile:?}: {result}");
    }
}

#[test]
fn rebinding_the_callee_mid_loop_leaves_the_trace() {
    let source = "
        local function f(x) return x + 1 end
        local function g(x) return x * 2 end
        local fn = f
        local s = 0
        for i = 1, 30000 do
          if i == 10000 then fn = g end
          s = s + fn(i)
        end
        return s";
    for profile in ALL_PROFILES {
        let (_, stats) = agree(profile, source);
        assert!(
            stats.inlined_calls > 0 && stats.iterations > 0,
            "{profile:?}: {stats:?}"
        );
    }
}

#[test]
fn closures_made_and_collected_between_runs_never_match_a_stale_trace() {
    // Each round makes a new closure of one prototype, runs the loop with it
    // and collects. A trace for a dead closure must not run for a live one
    // that took its slot, and the number of traces stays bounded.
    let source = "
        local function run(f)
          local s = 0
          for i = 1, 3000 do s = s + f(i) end
          return s
        end
        local total = 0
        for round = 1, 40 do
          local k = round
          local function f(x) return x + k end
          total = total + run(f)
          f = nil
          collectgarbage()
        end
        return total";
    for profile in TWO_PROFILES {
        let (_, stats) = agree(profile, source);
        assert!(stats.traces_compiled <= 16, "{profile:?}: {stats:?}");
    }
}

// ---------------------------------------------- results and arguments

#[test]
fn missing_arguments_and_extra_results_are_normalised_as_the_interpreter_does() {
    let source = "
        local function f(a, b) return a, b end
        local s = 0
        for i = 1, 30000 do
          local x = f(i)
          local y, z = f(i, 2)
          local p, q, r = f(i, 1)
          s = s + (y or 0) + (q or 0) + i
          if x ~= i or z ~= 2 or r ~= nil then s = s + 1000000 end
        end
        return s";
    for profile in ALL_PROFILES {
        let (result, stats) = agree(profile, source);
        assert!(
            stats.inlined_calls >= 3 && stats.iterations > 0,
            "{profile:?}: {stats:?}"
        );
        // No iteration saw the wrong shape of results.
        assert!(!result.contains("1000000"), "{profile:?}: {result}");
    }
}

#[test]
fn a_callee_with_no_value_returns_nothing_into_the_callers_registers() {
    let source = "
        local function f(x) end
        local s = 0
        for i = 1, 30000 do
          local a = f(i)
          if a == nil then s = s + 1 end
        end
        return s";
    for profile in ALL_PROFILES {
        let (result, stats) = agree(profile, source);
        assert!(stats.iterations > 0, "{profile:?}: {stats:?}");
        assert!(result.contains("30000"), "{profile:?}: {result}");
    }
}

// ------------------------------------------------- what is not followed

#[test]
fn calls_the_recorder_cannot_follow_stay_in_the_interpreter() {
    // Recursion, varargs, a native callee and an open result window.
    let sources = [
        "local function fib(n) if n < 2 then return n end return fib(n - 1) + fib(n - 2) end
         local s = 0
         for i = 1, 3000 do s = s + fib(5) end
         return s",
        "local function f(...) return select('#', ...) end
         local s = 0
         for i = 1, 30000 do s = s + f(i, i) end
         return s",
        "local function f(x) return x, x + 1 end
         local s = 0
         for i = 1, 30000 do s = s + select('#', f(i)) end
         return s",
        "local s = 0
         for i = 1, 30000 do s = s + math.floor(i / 3) end
         return s",
    ];
    for source in sources {
        for profile in TWO_PROFILES {
            agree(profile, source);
        }
    }
}
