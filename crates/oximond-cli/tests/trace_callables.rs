//! Traces that call native functions and read functions out of tables,
//! through compiled Lua source.
//!
//! A trace calls a pure numeric native (`math.sin`, `math.max`, `tonumber`
//! of a float) itself, and reads a function out of a table field (a method,
//! `math.sin`) as a constant the entry checked. Every program runs under the
//! interpreter and under tracing execution. The results must match exactly,
//! and the counters must show that a trace ran natively, because equal
//! output alone cannot tell a trace from a slow path.
//!
//! What these programs hold the engine to:
//!
//! - A function that is rebound (the table field, the global, a local) is
//!   never called through a stale trace, in the middle of a run or between
//!   two runs of the same loop.
//! - An argument that is not a float any more (an integer, a string, nil)
//!   leaves the trace before the call, and the interpreter raises its own
//!   error or converts as it always did.
//! - Results, nan and infinity included, match to the bit.

use oximond::vm::{TraceExitKind, TraceStatistics};
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

const ALL_PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

/// The profiles that read a global with one instruction. Lua 5.2 and later
/// read it as a field of the `_ENV` table.
const FLAT_GLOBALS: [CompatibilityProfile; 2] =
    [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21];

type Runner = fn(CompatibilityProfile, ExecutionMode, &str) -> (String, TraceStatistics);

fn run(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
) -> (String, TraceStatistics) {
    run_configured(profile, mode, source, false)
}

/// [`run`] for a loop whose remaining work falls under the bar a trace
/// compile must repay. The trace still compiles, so the test can check what
/// it does.
fn run_ignoring_remaining_work(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
) -> (String, TraceStatistics) {
    run_configured(profile, mode, source, true)
}

fn run_configured(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
    ignore_remaining_work: bool,
) -> (String, TraceStatistics) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    if ignore_remaining_work {
        lua.vm_mut().ignore_remaining_work();
    }
    let outcome: Result<Vec<Value>, _> = lua.exec(source);
    let text = match outcome {
        Ok(results) => format!("{results:?}"),
        Err(error) => format!("error: {error}"),
    };
    (text, lua.vm().jit_statistics().trace)
}

/// Runs `source` under the interpreter and under tracing execution, checks
/// the results (or the error) agree, and returns the tracing statistics.
fn agree(profile: CompatibilityProfile, source: &str) -> (String, TraceStatistics) {
    agree_with(run, profile, source)
}

/// [`agree`] with [`run_ignoring_remaining_work`].
fn agree_ignoring_remaining_work(
    profile: CompatibilityProfile,
    source: &str,
) -> (String, TraceStatistics) {
    agree_with(run_ignoring_remaining_work, profile, source)
}

fn agree_with(
    runner: Runner,
    profile: CompatibilityProfile,
    source: &str,
) -> (String, TraceStatistics) {
    let (expected, _) = runner(profile, ExecutionMode::Interpreter, source);
    let (tracing, stats) = runner(profile, ExecutionMode::Tracing, source);
    assert_eq!(
        tracing, expected,
        "{profile:?} tracing\n{source}\n{stats:?}"
    );
    (expected, stats)
}

/// Like [`agree`], requiring that a trace ran at least `iterations` native
/// iterations.
fn agree_native(profile: CompatibilityProfile, source: &str, iterations: u64) -> String {
    native_iterations(agree(profile, source), profile, source, iterations)
}

/// [`agree_native`] with [`run_ignoring_remaining_work`].
fn agree_native_ignoring_remaining_work(
    profile: CompatibilityProfile,
    source: &str,
    iterations: u64,
) -> String {
    native_iterations(
        agree_ignoring_remaining_work(profile, source),
        profile,
        source,
        iterations,
    )
}

fn native_iterations(
    (result, stats): (String, TraceStatistics),
    profile: CompatibilityProfile,
    source: &str,
    iterations: u64,
) -> String {
    assert!(
        stats.iterations >= iterations,
        "{profile:?}: {} native iterations\n{source}\n{stats:?}",
        stats.iterations
    );
    result
}

#[test]
fn a_math_call_in_a_hot_loop_runs_inside_the_trace() {
    // `math.sin` is read out of the global table `math`, a float goes in and
    // a float comes out.
    let source = "
        local total = 0.5
        for i = 1, 40000 do
            local x = (i % 1000) / 1000
            total = total + math.sin(x) * math.cos(x) + math.sqrt(x + 1)
        end
        return string.format('%.9f', total)";
    for profile in FLAT_GLOBALS {
        let (_, stats) = agree(profile, source);
        assert!(stats.iterations > 39_000, "{profile:?}: {stats:?}");
        assert_eq!(stats.exits_of(TraceExitKind::LoopComplete), 1, "{stats:?}");
    }
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
    // With the functions in locals every profile runs it natively.
    let source = "
        local sin, cos, sqrt = math.sin, math.cos, math.sqrt
        local total = 0.5
        for i = 1, 40000 do
            local x = (i % 1000) / 1000
            total = total + sin(x) * cos(x) + sqrt(x + 1)
        end
        return string.format('%.9f', total)";
    for profile in ALL_PROFILES {
        let (_, stats) = agree(profile, source);
        assert!(stats.iterations > 39_000, "{profile:?}: {stats:?}");
        assert_eq!(stats.exits_of(TraceExitKind::LoopComplete), 1, "{stats:?}");
    }
}

#[test]
fn two_argument_natives_and_tonumber_run_inside_the_trace() {
    let source = "
        local hi, lo = math.max, math.min
        local convert = tonumber
        local total = 0
        for i = 1, 30000 do
            local x = i / 7
            total = total + hi(x, 3.5) - lo(x, 100.25) + convert(x)
        end
        return total";
    for profile in ALL_PROFILES {
        agree_native(profile, source, 29_000);
    }
}

#[test]
fn natives_compute_nan_and_infinity_like_the_interpreter() {
    let source = "
        local out = {}
        local values = {0.0, -0.0, 1.0, -1.0, 0.5, 1e308, -1e308, 1/0, -1/0, 0/0}
        for round = 1, 3000 do
            local v = values[round % #values + 1]
            local a, b, c = math.sin(v), math.sqrt(v), math.max(v, 1.0)
            out[1] = tostring(a) .. tostring(b) .. tostring(c)
        end
        return out[1]";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
    // The same through a loop whose arguments stay floats, so one trace
    // handles every value.
    let source = "
        local sin, sqrt, floor = math.sin, math.sqrt, math.floor
        local x = 1e-3
        local acc = 0
        for i = 1, 30000 do
            x = x * 1.0007
            acc = acc + sin(x) + sqrt(x) * 1e-9
        end
        return string.format('%.12g', acc), x";
    for profile in ALL_PROFILES {
        agree_native(profile, source, 29_000);
    }
}

#[test]
fn rebinding_a_math_function_mid_loop_leaves_the_trace() {
    // After 20000 iterations `math.sqrt` is another function, which the
    // trace must not call as the old one.
    let source = "
        local total = 0
        for i = 1, 30000 do
            if i == 20000 then math.sqrt = function(x) return -x end end
            total = total + math.sqrt(i + 0.5)
        end
        return total";
    for profile in FLAT_GLOBALS {
        let (_, stats) = agree_ignoring_remaining_work(profile, source);
        assert!(stats.iterations > 10_000, "{profile:?}: {stats:?}");
    }
    for profile in ALL_PROFILES {
        agree_ignoring_remaining_work(profile, source);
    }
    // Rebinding a local that holds the native.
    let source = "
        local f = math.sin
        local total = 0
        for i = 1, 30000 do
            if i == 20000 then f = math.cos end
            total = total + f(i + 0.5)
        end
        return total";
    for profile in ALL_PROFILES {
        agree_native_ignoring_remaining_work(profile, source, 10_000);
    }
    // Between two runs of one loop: the second run finds another function
    // in the field, so its entry refuses the trace.
    let source = "
        local function run(count)
          local total = 0
          for i = 1, count do total = total + math.sin(i + 0.25) end
          return total
        end
        local a = run(20000)
        math.sin = function(x) return x * 2 end
        local b = run(20000)
        math.sin = nil
        local ok, message = pcall(run, 3)
        return a, b, ok, message";
    for profile in ALL_PROFILES {
        agree_ignoring_remaining_work(profile, source);
    }
}

#[test]
fn an_argument_that_stops_being_a_float_leaves_before_the_call() {
    // The argument becomes an integer (in the profiles that have them), a
    // numeric string, nil and a table. The interpreter converts or raises.
    let source = "
        local out = {}
        local function probe(v) local ok, r = pcall(math.sqrt, v); return tostring(ok) .. tostring(r) end
        local total = 0
        for i = 1, 30000 do
            local v = i + 0.5
            if i == 25000 then v = 16 end
            if i == 26000 then v = '81' end
            total = total + math.sqrt(v)
        end
        local bad = {}
        for i = 1, 30000 do
            local v = i + 0.5
            if i == 29000 then v = nil end
            local ok, r = pcall(function() return math.sin(v) end)
            if not ok then bad[#bad + 1] = r end
        end
        return total, #bad, bad[1]";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn a_native_in_a_rarely_taken_branch_records_as_a_side_or_stays_exact() {
    let source = "
        local sin = math.sin
        local total = 0
        for i = 1, 40000 do
            local x = i / 3
            if i % 5 == 0 then
                total = total + sin(x)
            else
                total = total + x
            end
        end
        return string.format('%.6f', total)";
    for profile in ALL_PROFILES {
        agree_native(profile, source, 10_000);
    }
}

#[test]
fn a_method_call_runs_inside_the_trace() {
    let source = "
        local object = {value = 0}
        function object:add(delta)
          self.value = self.value + delta
          return self.value
        end
        local total = 0
        for i = 1, 40000 do total = total + object:add(i % 7) end
        return total, object.value";
    for profile in ALL_PROFILES {
        let (_, stats) = agree(profile, source);
        assert!(stats.inlined_calls > 0, "{profile:?}: {stats:?}");
        assert!(stats.iterations > 39_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn replacing_the_method_between_runs_never_calls_the_old_one() {
    // Two objects with the same shape and different methods go through one
    // loop. The trace pinned the first method, so the second object's entry
    // is refused, and so is a method replaced in place.
    let source = "
        local function run(object)
          local total = 0
          for i = 1, 20000 do total = total + object:step(i) end
          return total
        end
        local a = {k = 1}
        function a:step(i) return i + self.k end
        local b = {k = 1}
        function b:step(i) return i * 2 + self.k end
        local r1 = run(a)
        local r2 = run(b)
        a.step = function(self, i) return -i end
        local r3 = run(a)
        a.step = math.abs
        local r4 = pcall(run, a)
        return r1, r2, r3, r4";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn replacing_the_method_mid_loop_leaves_the_trace() {
    let source = "
        local object = {n = 0}
        function object:bump(d) self.n = self.n + d; return self.n end
        local total = 0
        for i = 1, 30000 do
            if i == 20000 then
              function object:bump(d) self.n = self.n - d; return self.n * 2 end
            end
            total = total + object:bump(1)
        end
        return total, object.n";
    for profile in ALL_PROFILES {
        let (_, stats) = agree(profile, source);
        assert!(stats.iterations > 10_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_call_through_call_runs_inside_the_trace() {
    let source = "
        local callable = setmetatable({bias = 11}, {
          __call = function(self, a, b) return self.bias + a * 3 + b end,
        })
        local total = 0
        for i = 1, 40000 do total = total + callable(i % 97, i % 13) end
        return total";
    for profile in ALL_PROFILES {
        let (_, stats) = agree(profile, source);
        assert!(stats.inlined_calls > 0, "{profile:?}: {stats:?}");
        assert!(stats.iterations > 39_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn objects_of_one_class_share_a_trace_through_call() {
    // The trace checks the handler, not which table it is, so all four
    // objects stay inside it.
    let source = "
        local class = {__call = function(self, a) return self.v + a end}
        local objects = {}
        for i = 1, 4 do objects[i] = setmetatable({v = i * 100}, class) end
        local total = 0
        for i = 1, 40000 do total = total + objects[i % 4 + 1](i % 7) end
        return total";
    for profile in ALL_PROFILES {
        let (_, stats) = agree(profile, source);
        assert!(stats.iterations > 39_000, "{profile:?}: {stats:?}");
    }
    // A table made each iteration may reuse a freed one's address. Each call
    // must still see its own fields.
    let fresh = "
        local class = {__call = function(self, a) return self.v + a end}
        local total = 0
        for i = 1, 30000 do
          local object = {v = i}
          setmetatable(object, class)
          total = total + object(1)
        end
        return total";
    for profile in ALL_PROFILES {
        agree(profile, fresh);
    }
}

#[test]
fn a_changed_call_handler_or_metatable_leaves_the_trace() {
    let source = "
        local class = {__call = function(self, a) return self.v + a end}
        local object = setmetatable({v = 1}, class)
        local other = {__call = function(self, a) return a * 2 end}
        local total = 0
        for i = 1, 40000 do
          if i == 10000 then class.__call = function(self, a) return self.v - a end end
          if i == 20000 then setmetatable(object, other) end
          if i == 30000 then other.__call = math.abs end
          total = total + object(i % 9)
        end
        other.__call = nil
        return total, pcall(function() for i = 1, 10 do total = total + object(i) end end)";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn a_native_stored_as_a_method_is_called_like_any_other_native() {
    let source = "
        local ops = {square_root = math.sqrt, wave = math.sin}
        local total = 0
        for i = 1, 30000 do
            local x = i + 0.5
            total = total + ops.square_root(x) + ops.wave(x)
        end
        return string.format('%.6f', total)";
    for profile in ALL_PROFILES {
        agree_native(profile, source, 29_000);
    }
}

#[test]
fn a_function_read_from_a_table_restores_into_a_register_at_an_exit() {
    // `f` is live across a guard that fails: the exit must write the
    // function (a Lua closure and a native one) back into its register.
    let source = "
        local t = {inc = function(x) return x + 1 end, root = math.sqrt}
        local keep
        local total = 0
        for i = 1, 30000 do
            local f = t.inc
            local g = t.root
            total = total + f(i) + g(i + 0.5)
            if i == 29000 then keep = {f, g} end
            if i == 29500 then total = total + keep[1](1) + keep[2](4.0) end
        end
        return total";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

// ------------------------------------------------------- `__index` chains

#[test]
fn fields_behind_a_table_valued_index_are_read_natively() {
    let source = "
        local fallback = {x = 3, y = 5}
        local object = setmetatable({}, {__index = fallback})
        local total = 0
        for i = 1, 40000 do total = total + object.x + object.y + i % 11 end
        return total";
    for profile in ALL_PROFILES {
        agree_native(profile, source, 39_000);
    }
}

#[test]
fn methods_behind_a_class_table_are_called_inside_the_trace() {
    let source = "
        local Account = {}
        Account.__index = Account
        function Account:deposit(n) self.balance = self.balance + n; return self.balance end
        local account = setmetatable({balance = 0}, Account)
        local total = 0
        for i = 1, 40000 do total = total + account:deposit(i % 5) end
        return total, account.balance";
    for profile in ALL_PROFILES {
        let (_, stats) = agree(profile, source);
        assert!(stats.inlined_calls > 0, "{profile:?}: {stats:?}");
        assert!(stats.iterations > 39_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn an_inheritance_chain_resolves_each_field_at_its_own_depth() {
    let source = "
        local Base = {scale = 2, offset = 7}
        Base.__index = Base
        local Mid = setmetatable({scale = 3}, Base)
        Mid.__index = Mid
        local object = setmetatable({offset = 1}, Mid)
        local total = 0
        for i = 1, 40000 do total = total + object.scale * i + object.offset end
        return total";
    for profile in ALL_PROFILES {
        agree_native_ignoring_remaining_work(profile, source, 39_000);
    }
}

#[test]
fn a_field_that_starts_shadowing_the_fallback_is_never_read_from_the_old_slot() {
    // The object gets its own `x` between two runs of one loop, and in the
    // middle of a run. The trace recorded `x` one hop away.
    let source = "
        local fallback = {x = 3}
        local object = setmetatable({}, {__index = fallback})
        local function run(count)
          local total = 0
          for i = 1, count do total = total + object.x end
          return total
        end
        local a = run(20000)
        object.x = 100
        local b = run(20000)
        object.x = nil
        local c = run(20000)
        fallback.x = 4
        local d = run(20000)
        return a, b, c, d";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
    let source = "
        local fallback = {x = 3}
        local object = setmetatable({}, {__index = fallback})
        local total = 0
        for i = 1, 30000 do
          if i == 20000 then object.x = 100 end
          total = total + object.x
        end
        return total";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn a_changed_chain_between_runs_is_not_read_through_a_stale_slot() {
    let source = "
        local one = {x = 1}
        local two = {x = 2}
        local meta = {__index = one}
        local object = setmetatable({}, meta)
        local function run(count)
          local total = 0
          for i = 1, count do total = total + object.x end
          return total
        end
        local a = run(20000)
        meta.__index = two
        local b = run(20000)
        meta.__index = function(_, key) return 5 end
        local c = run(20000)
        setmetatable(object, nil)
        local ok, message = pcall(run, 3)
        return a, b, c, ok, message";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn a_store_to_an_inherited_field_goes_to_the_object() {
    // `object.count = object.count + 1` reads the fallback's value and
    // writes the object's own slot, which has to stay exact.
    let source = "
        local fallback = {count = 10}
        local object = setmetatable({}, {__index = fallback})
        for i = 1, 30000 do object.count = object.count + 1 end
        return object.count, fallback.count, rawget(object, 'count')";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

// ----------------------------------------------------------------- varargs

#[test]
fn a_call_of_a_vararg_function_runs_inside_the_trace() {
    let source = "
        local function sum(...)
          local a, b, c, d, e = ...
          return a + b + c + d + e
        end
        local total = 0
        for i = 1, 40000 do total = total + sum(i, 2, 3, 4, 5) end
        return total";
    for profile in ALL_PROFILES {
        let (_, stats) = agree(profile, source);
        assert!(stats.inlined_calls > 0, "{profile:?}: {stats:?}");
        assert!(stats.iterations > 39_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn varargs_forward_through_a_tail_call() {
    let source = "
        local function sum(a, b, c, d, e) return a + b + c + d + e end
        local function forward(...) return sum(...) end
        local total = 0
        for i = 1, 40000 do total = total + forward(i, 2, 3, 4, 5) end
        return total";
    for profile in ALL_PROFILES {
        let (_, stats) = agree(profile, source);
        assert!(stats.inlined_calls > 1, "{profile:?}: {stats:?}");
        assert!(stats.iterations > 39_000, "{profile:?}: {stats:?}");
    }
    // Forwarded with a fixed argument in front, and through a plain call.
    let source = "
        local function add3(a, b, c) return a * 100 + b * 10 + c end
        local function pass(first, ...) return add3(first, ...) end
        local function twice(...) local x = pass(...); return x + pass(...) end
        local total = 0
        for i = 1, 30000 do total = total + twice(i % 7, 2, 3) end
        return total";
    for profile in ALL_PROFILES {
        agree_native(profile, source, 29_000);
    }
}

#[test]
fn missing_and_extra_varargs_are_padded_and_dropped_like_the_interpreter_does() {
    let source = "
        local function pick(...)
          local a, b, c = ...
          return (a or 0) + (b or 0) + (c or 0)
        end
        local total = 0
        for i = 1, 30000 do
          total = total + pick(i) + pick(i, 1) + pick(i, 1, 2) + pick(i, 1, 2, 3, 4)
        end
        return total";
    for profile in ALL_PROFILES {
        agree_native(profile, source, 29_000);
    }
}

#[test]
fn a_guard_inside_a_vararg_callee_starts_the_call_over_in_the_interpreter() {
    // The callee branches on its argument, the branch the recording did not
    // see runs after 20000 iterations, and asks for a traceback and the
    // varargs themselves. The interpreter must run that call from its start
    // with the frame it builds, so `select` and `debug.traceback` see it.
    let source = "
        local seen
        local function f(x, ...)
          if x > 20000 then
            seen = select('#', ...) .. ':' .. (select(2, ...) or 'nil') .. debug.traceback('t')
            return 0
          end
          local a, b = ...
          return x + a + b
        end
        local total = 0
        for i = 1, 30000 do total = total + f(i, 1, 2) end
        return total, seen";
    for profile in ALL_PROFILES {
        let (result, stats) = agree(profile, source);
        assert!(result.contains("2:2"), "{profile:?}: {result}");
        assert!(stats.iterations > 10_000, "{profile:?}: {stats:?}");
    }
    // An error raised from the callee after an exit unwinds the right frames.
    let source = "
        local function f(x, ...)
          local a = ...
          if x == 25000 then error('boom ' .. a) end
          return x + a
        end
        local total = 0
        local ok, message = pcall(function()
          for i = 1, 30000 do total = total + f(i, 1) end
        end)
        return total, ok, message";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn a_store_inside_a_vararg_callee_is_never_replayed() {
    // The callee writes a field before a guard can fail: a trace that left
    // through the call-site snapshot would run the store twice. The
    // recording refuses it, so the counts stay exact.
    let source = "
        local counter = {n = 0}
        local function bump(...)
          counter.n = counter.n + select('#', ...)
          local d = ...
          if d > 20000 then return -1 end
          return counter.n
        end
        local total = 0
        for i = 1, 30000 do total = total + bump(i, 1) end
        return total, counter.n";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
    let source = "
        local counter = {n = 0}
        local function bump(d, ...)
          local extra = ...
          counter.n = counter.n + 1
          if d > 20000 then return -1 end
          return counter.n + extra
        end
        local total = 0
        for i = 1, 30000 do total = total + bump(i, 1) end
        return total, counter.n";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn a_vararg_callee_may_write_an_upvalue_and_call_another_function() {
    let source = "
        local total = 0
        local function inc(n) return n + 1 end
        local function accumulate(...)
          local a, b = ...
          total = total + inc(a) + b
          return total
        end
        local last = 0
        for i = 1, 30000 do last = accumulate(i % 10, 1) end
        return last, total";
    for profile in ALL_PROFILES {
        agree_native(profile, source, 29_000);
    }
}

// ------------------------------------------------------------------ ipairs

#[test]
fn an_ipairs_loop_reads_its_elements_inside_the_trace() {
    let source = "
        local values = {}
        for i = 1, 5000 do values[i] = i % 113 end
        local total = 0
        for round = 1, 12 do
          for _, value in ipairs(values) do total = total + value end
        end
        return total";
    for profile in ALL_PROFILES {
        agree_native(profile, source, 50_000);
    }
}

#[test]
fn an_ipairs_loop_stops_where_the_interpreter_stops() {
    // A nil hole ends the loop early, an element of another type is read by
    // the interpreter, and a `break` leaves from inside.
    let source = "
        local out = {}
        for round = 1, 400 do
          local values = {}
          for i = 1, 60 do values[i] = i end
          if round % 3 == 0 then values[40] = nil end
          if round % 5 == 0 then values[30] = 0.5 end
          if round % 7 == 0 then values[20] = true end
          local total, count = 0, 0
          for index, value in ipairs(values) do
            if type(value) ~= 'number' then break end
            if round % 11 == 0 and index > 50 then break end
            total = total + value
            count = index
          end
          out[#out + 1] = total .. ':' .. count
        end
        return table.concat(out, ',')";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn an_ipairs_loop_that_changes_its_table_sees_what_the_interpreter_sees() {
    // The loop appends while it walks, then overwrites and shortens: ipairs
    // reads the live table at every step.
    let source = "
        local values = {1, 2, 3}
        local total = 0
        for round = 1, 3000 do
          for index, value in ipairs(values) do
            total = total + value
            if index == #values and #values < 40 then values[#values + 1] = index + value end
          end
          if round % 500 == 0 then values[#values] = nil end
        end
        return total, #values";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
    let source = "
        local values = {}
        for i = 1, 3000 do values[i] = i end
        local total = 0
        for round = 1, 12 do
          for index, value in ipairs(values) do
            if round == 6 and index == 1000 then values[index + 1] = nil end
            if round == 9 and index == 2000 then values[index] = 1.5 end
            total = total + (values[index] or 0)
          end
        end
        return total";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn nested_ipairs_loops_and_a_replaced_iterator_stay_exact() {
    let source = "
        local rows = {}
        for r = 1, 40 do
          local row = {}
          for c = 1, 40 do row[c] = r * c end
          rows[r] = row
        end
        local total = 0
        for round = 1, 30 do
          for _, row in ipairs(rows) do
            for _, value in ipairs(row) do total = total + value end
          end
        end
        return total";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
    // A loop run twice with another iterator the second time.
    let source = "
        local values = {}
        for i = 1, 2000 do values[i] = i end
        local function run(iterate)
          local total = 0
          for _, value in iterate(values) do total = total + value end
          return total
        end
        local a = run(ipairs)
        local b = run(function(t) return function(_, i) i = i + 2; if t[i] then return i, t[i] end end, t, -1 end)
        local c = run(ipairs)
        return a, b, c";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn ipairs_follows_index_metamethods_where_the_profile_does() {
    // Lua 5.3 and later read elements through `__index`: a proxy with no
    // array part of its own must keep working.
    let source = "
        local backing = {}
        for i = 1, 200 do backing[i] = i end
        local proxy = setmetatable({}, {__index = backing})
        local total = 0
        for round = 1, 200 do
          for _, value in ipairs(proxy) do total = total + value end
        end
        return total";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

// ------------------------------------------- calls in the outer part of a nest

#[test]
fn a_branching_callee_in_front_of_an_inner_loop_runs_inside_the_tree() {
    // The callee alternates between its arms, and the inner loop is short,
    // but with more trips than a loop gives way with (`PREFER_UNROLL_TRIPS`:
    // with eight or fewer the outer loop records the nest and unrolls it, see
    // `tests/trace_short_loops.rs`). The trace of the inner loop takes side
    // traces that call the callee, one per arm, and run the outer iteration
    // around the inner loop.
    let source = "
        local function bump(x)
          if x % 2 == 0 then return x + 1 else return x + 2 end
        end
        local total = 0
        for outer = 1, 60000 do
          local value = bump(outer)
          for inner = 1, 12 do total = total + (value + inner) % 31 end
        end
        return total";
    for profile in ALL_PROFILES {
        let (_, stats) = agree(profile, source);
        assert!(stats.iterations > 400_000, "{profile:?}: {stats:?}");
        assert!(stats.side_installs >= 2, "{profile:?}: {stats:?}");
        // The two arms stay in native code, so almost no outer iteration
        // goes through the interpreter.
        assert!(stats.native_entries < 6_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_callee_side_that_meets_a_new_arm_or_an_error_leaves_with_the_real_call() {
    // A third arm that the sides never saw, an error from inside the callee
    // and a traceback taken in it: the interpreter makes the call itself.
    let source = "
        local tb
        local function pick(x)
          if x % 4 == 0 then return x + 1
          elseif x % 4 == 1 then return x + 2
          elseif x == 50002 then tb = debug.traceback('inside'); return 0
          else return x + 3 end
        end
        local total = 0
        for outer = 1, 60000 do
          local value = pick(outer)
          for inner = 1, 6 do total = total + (value * inner) % 17 end
        end
        return total, tb";
    for profile in ALL_PROFILES {
        let (result, _) = agree(profile, source);
        assert!(result.contains("inside"), "{profile:?}: {result}");
    }
    let source = "
        local function check(x)
          if x % 2 == 0 then return x end
          if x > 40000 then error('odd and large: ' .. x) end
          return x + 1
        end
        local total = 0
        local ok, message = pcall(function()
          for outer = 1, 60000 do
            local value = check(outer)
            for inner = 1, 5 do total = total + value % (inner + 1) end
          end
        end)
        return total, ok, message";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn a_callee_that_stores_before_its_branch_is_never_replayed() {
    // The store comes first, so a guard behind it cannot leave before the
    // call: the count of stores must match the interpreter's exactly.
    let source = "
        local log = {n = 0}
        local function step(x)
          log.n = log.n + 1
          if x % 2 == 0 then return x + 1 else return x - 1 end
        end
        local total = 0
        for outer = 1, 50000 do
          local value = step(outer)
          for inner = 1, 4 do total = total + (value + inner) % 13 end
        end
        return total, log.n";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn nested_calls_and_methods_in_the_outer_part_stay_exact() {
    let source = "
        local function inner(x) if x > 100 then return x - 100 else return x end end
        local function outer(x) local y = inner(x) if y % 3 == 0 then return y + 1 end return y end
        local object = {k = 7}
        function object:scale(x) if x % 5 == 0 then return x * self.k end return x end
        local total = 0
        for i = 1, 40000 do
          local a = outer(i % 211)
          local b = object:scale(i)
          for j = 1, 3 do total = total + (a + b + j) % 19 end
        end
        return total";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn a_call_the_side_cannot_follow_in_the_outer_part_leaves_the_loop_correct() {
    // `string.rep` and a callee with an upvalue are calls a side does not
    // inline. The nest still computes what the interpreter computes.
    let source = "
        local calls = 0
        local function counted(x) calls = calls + 1; return x + calls % 3 end
        local total = 0
        for i = 1, 30000 do
          local n = #string.rep('x', i % 5)
          local value = counted(i) + n
          for j = 1, 4 do total = total + (value + j) % 11 end
        end
        return total, calls";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn a_callee_rebound_between_runs_of_the_nest_is_called_through_its_own_guard() {
    let source = "
        local function a(x) if x % 2 == 0 then return x + 1 else return x + 2 end end
        local function b(x) if x % 2 == 0 then return x - 1 else return x - 2 end end
        local f = a
        local function run()
          local total = 0
          for outer = 1, 20000 do
            local value = f(outer)
            for inner = 1, 6 do total = total + (value + inner) % 29 end
          end
          return total
        end
        local r1 = run()
        f = b
        local r2 = run()
        f = a
        local r3 = run()
        return r1, r2, r3";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

// ------------------------------------------------- retries on a filling table

#[test]
fn a_loop_that_fills_an_array_before_it_overwrites_it_is_traced_once_it_settles() {
    // The first 5000 iterations append, which a trace cannot do. Under tracing
    // execution the retries keep waiting longer, so one lands after
    // the fill and records the overwrite.
    let source = "
        local values = {}
        for i = 1, 60000 do values[(i - 1) % 5000 + 1] = i % 103 end
        local total = 0
        for i = 1, 5000 do total = total + values[i] end
        return total";
    for profile in ALL_PROFILES {
        let (_, stats) = agree(profile, source);
        assert!(stats.iterations > 40_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_loop_that_always_appends_keeps_its_retries_cheap() {
    // Every run of the loop fills a new table, so no attempt ever finds a
    // steady state. The answer must not depend on it.
    let source = "
        local total = 0
        for round = 1, 12 do
          local values = {}
          for i = 1, 4000 do values[i] = (i * round) % 97 end
          table.sort(values)
          total = total + values[1] + values[#values]
        end
        return total";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn a_loop_that_calls_many_different_functions_pins_them_all() {
    // Twelve natives and a closure in one body: more callables than a handful.
    let source = "
        local sin, cos, tan, sqrt, exp = math.sin, math.cos, math.tan, math.sqrt, math.exp
        local rad, deg, sinh, cosh, tanh = math.rad, math.deg, math.sinh, math.cosh, math.tanh
        local max, min = math.max, math.min
        local function bend(x) if x > 3 then return x - 3 end return x end
        local total = 0
        for i = 1, 30000 do
          local x = (i % 29) / 7 + 0.1
          total = total + sin(x) + cos(x) + tan(x) + sqrt(x) + exp(x / 9) + rad(x)
          total = total + deg(x) + sinh(x) + cosh(x) + tanh(x) + max(x, 1.5) + min(x, 0.5) + bend(x)
        end
        return string.format('%.6f', total)";
    for profile in ALL_PROFILES {
        agree_native(profile, source, 29_000);
    }
}

const ENV_PROFILES: [CompatibilityProfile; 4] = [
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
];

#[test]
fn math_loops_through_env_run_natively_in_lua_52_and_later() {
    // Natives the trace holds, a constant, and a loop with no call at all
    // besides `math.pi`: none needs a call through the VM to trace.
    let sources = [
        "local total = 0.5
         for i = 1, 40000 do
           local x = (i % 1000) / 1000
           total = total + math.sin(x) * math.cos(x) + math.sqrt(x + 1)
         end
         return string.format('%.9f', total)",
        "local total = 0
         for i = 1, 40000 do total = total + math.pi * (i % 7) end
         return string.format('%.9f', total)",
        "local total = 0
         for i = 1, 40000 do total = total + math.max(i, 3) - math.min(i, 9) end
         return total",
    ];
    for source in sources {
        for profile in ENV_PROFILES {
            let (_, stats) = agree(profile, source);
            assert!(
                stats.iterations > 39_000,
                "{profile:?}: {stats:?}\n{source}"
            );
            assert_eq!(stats.exits_of(TraceExitKind::LoopComplete), 1, "{stats:?}");
        }
    }
}

#[test]
fn replacing_env_or_the_math_table_mid_loop_leaves_the_trace() {
    // The math table is replaced in _ENV, the whole _ENV is rebound, and a
    // field of the table is replaced. Each result must match the interpreter.
    let sources = [
        "local total = 0
         for i = 1, 30000 do
           if i == 20000 then math = {sqrt = function(x) return -x end, sin = math.sin} end
           total = total + math.sqrt(i + 0.5)
         end
         return total",
        "local e1 = _ENV
         local e2 = setmetatable({math = {sqrt = function(x) return 1 end}}, {__index = _ENV})
         local function f()
           local total = 0
           for i = 1, 30000 do
             if i == 20000 then _ENV = e2 end
             total = total + math.sqrt(i + 0.5)
           end
           return total
         end
         return f()",
        "local total = 0
         for i = 1, 30000 do
           if i == 20000 then math.pi = 3 end
           total = total + math.pi
         end
         return total",
    ];
    for source in sources {
        for profile in ENV_PROFILES {
            let (_, stats) = agree_ignoring_remaining_work(profile, source);
            assert!(
                stats.iterations > 10_000,
                "{profile:?}: {stats:?}\n{source}"
            );
        }
    }
}
