//! Loops through tail calls, through compiled Lua source.
//!
//! A tail call replaces its frame in place, so self tail recursion and a
//! cycle of functions that tail call each other are loops: the frame comes
//! back to pc 0 of the same function at the same depth. Tracing mode records
//! such a loop from pc 0 and closes it at the tail call back into the closure
//! it started with. The functions in between run in the root frame (later
//! activations of it), and an exit while one runs restores the frame running
//! that function, with the tail call count and the name the interpreter would
//! have given it.
//!
//! Every program runs under the interpreter and under tracing execution. The
//! results must match exactly (error messages and tracebacks included), and
//! the counters must show that a trace ran natively.

use oximond::vm::{TraceAbortReason, TraceStatistics};
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

const ALL_PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

/// Lua 5.1 counts each tail call as a traceback level, Lua 5.4 flags the
/// frame. One float profile and one integer profile.
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
/// compiled and ran native iterations in each.
fn agree_native(profiles: &[CompatibilityProfile], source: &str) -> Vec<(String, TraceStatistics)> {
    profiles
        .iter()
        .map(|profile| {
            let (result, stats) = agree(*profile, source);
            assert!(
                stats.traces_compiled > 0 && stats.native_entries > 0 && stats.iterations > 0,
                "{profile:?}: no trace ran natively\n{source}\n{stats:?}"
            );
            (result, stats)
        })
        .collect()
}

/// The shape of `benchmarks/lua/calls/tail_call.lua`: `even` is a local that
/// tail calls the global `odd`, which tail calls `even` through an upvalue.
const MUTUAL: &str = "
    local function even(n, total)
        if n == 0 then
            return total
        end
        return odd(n - 1, total + n % 13)
    end
    function odd(n, total)
        if n == 0 then
            return total
        end
        return even(n - 1, total + n % 17)
    end
";

#[test]
fn mutual_tail_recursion_runs_as_a_trace() {
    // An even count ends in `even`, an odd one in `odd`: the loop's exit
    // is in the root chunk's activation or in the one that replaced it.
    let source = format!("{MUTUAL} return even(20000, 0), even(20001, 0), odd(20000, 0)");
    for (_, stats) in agree_native(&ALL_PROFILES, &source) {
        // The trace carries nearly all of the 60000 tail calls. A recording
        // that starts in the function with the smaller register window
        // refuses the tail call into the wider one; the wider one's site
        // carries the cycle.
        assert!(stats.iterations > 25_000, "{stats:?}");
        assert!(
            stats.aborts(TraceAbortReason::RootTailCall) <= 1,
            "{stats:?}"
        );
    }
}

#[test]
fn self_tail_recursion_runs_as_a_trace() {
    let source = "
        local function loop(n, acc)
            if n == 0 then return acc end
            return loop(n - 1, acc + n)
        end
        return loop(30000, 0), loop(100, 0.5)";
    for (_, stats) in agree_native(&ALL_PROFILES, source) {
        assert!(stats.iterations > 29_000, "{stats:?}");
    }
}

#[test]
fn a_long_loop_polls_and_resumes_with_the_count_it_reached() {
    // More iterations than one native run's budget: the poll exit restores
    // the root closure's activation at its header, named after the closing
    // tail call, and the next tail call enters again. At the end the
    // traceback's levels count every tail call.
    let source = "
        local function levels()
            local low, high = 1, 1
            while debug.getinfo(high, 'l') do low, high = high, high * 2 end
            while high - low > 1 do
                local mid = (low + high) // 2
                if debug.getinfo(mid, 'l') then low = mid else high = mid end
            end
            return low
        end
        local function loop(n, acc)
            if n == 0 then
                local info = debug.getinfo(1, 'nSlt')
                return acc, levels(), info.name, info.currentline, info.istailcall
            end
            return loop(n - 1, acc + n % 7)
        end
        return loop(300000, 0)";
    for (_, stats) in agree_native(
        &[CompatibilityProfile::Lua53, CompatibilityProfile::Lua54],
        source,
    ) {
        assert!(stats.iterations > 290_000, "{stats:?}");
    }
    let source = source.replace("// 2", "/ 2 - (low + high) / 2 % 1");
    for (_, stats) in agree_native(&[CompatibilityProfile::Lua51], &source) {
        assert!(stats.iterations > 290_000, "{stats:?}");
    }
}

/// `odd` fails at `n == 5`, many tail calls into a traced run, and the
/// program reports what the error and the stack look like there.
fn failing_mutual(failure: &str) -> String {
    format!(
        "
        local function report(message)
            local lines = {{ tostring(message), debug.traceback('tb', 2) }}
            local level = 2
            while true do
                local info = debug.getinfo(level, 'nSlt')
                if not info then break end
                lines[#lines + 1] = table.concat({{
                    tostring(info.name), tostring(info.namewhat), info.what,
                    tostring(info.currentline), tostring(info.istailcall),
                }}, ' ')
                level = level + 1
            end
            return table.concat(lines, '\\n')
        end
        local function even(n, total)
            if n == 0 then return total end
            return odd(n - 1, total + n % 13)
        end
        function odd(n, total)
            if n == 5 then {failure} end
            if n == 0 then return total end
            return even(n - 1, total + n % 17)
        end
        local ok, message = pcall(even, 3000, 0)
        local traced = select(2, xpcall(even, report, 3001, 0))
        return ok, message, traced"
    )
}

#[test]
fn an_error_after_many_tail_calls_reports_as_the_interpreter_does() {
    for failure in [
        "error('x', 1)",
        "error('x', 2)",
        "local missing = nil; return missing + n",
    ] {
        agree_native(&TWO_PROFILES, &failing_mutual(failure));
    }
}

#[test]
fn a_traceback_inside_the_loop_counts_the_tail_calls() {
    // A non-tail call inside the replaced activation: the exit restores the
    // root frame running `odd` and pushes the callee above it.
    let source = format!(
        "{MUTUAL}
        local seen
        local function check(n)
            if n == 7 then
                local info = debug.getinfo(2, 'nSlt')
                seen = table.concat({{ debug.traceback('at 7', 1),
                    tostring(info.name), tostring(info.currentline),
                    tostring(info.istailcall) }}, '\\n')
            end
            return n
        end
        function odd(n, total)
            if n == 0 then return total end
            return even(check(n) - 1, total + n % 17)
        end
        local result = even(4000, 0)
        return result, seen"
    );
    agree_native(&TWO_PROFILES, &source);
}

#[test]
fn rebinding_the_global_in_the_cycle_mid_run_leaves_the_trace() {
    let source = format!(
        "{MUTUAL}
        local first = even(4000, 0)
        local original = odd
        local calls = 0
        odd = function(n, total)
            calls = calls + 1
            if n % 1000 == 0 then odd = original end
            return original(n, total + 1)
        end
        local second = even(4000, 0)
        odd = function(n, total) return -total end
        local third = even(4000, 0)
        return first, second, third, calls"
    );
    agree_native(&TWO_PROFILES, &source);
}

#[test]
fn closures_of_one_prototype_share_the_trace_with_their_own_cells() {
    let source = "
        local function make(step)
            local function loop(n, acc)
                if n == 0 then return acc end
                return loop(n - 1, acc + step)
            end
            return loop
        end
        local one, three = make(1), make(3)
        local total = 0
        for round = 1, 6 do
            total = total + one(2000 + round, 0) + three(2000, round)
        end
        return total";
    agree_native(&TWO_PROFILES, source);
}

#[test]
fn tail_calls_the_recorder_refuses_stay_correct() {
    // A vararg callee, a callee that captures its locals, a tail call to a
    // native function at the root, and a cycle through another closure of
    // the root chunk.
    for source in [
        "
        local function rest(...) local n, acc = ...; return n, acc end
        local function loop(n, acc)
            if n == 0 then return acc end
            local m, a = rest(n - 1, acc + 1)
            return loop(m, a)
        end
        local function spread(n, acc)
            if n == 0 then return acc end
            return spread2(n - 1, acc + 2)
        end
        function spread2(...)
            local n, acc = ...
            return spread(n, acc)
        end
        return loop(3000, 0), spread(3000, 0)",
        "
        local keep
        local function loop(n, acc)
            if n == 0 then return acc end
            keep = function() return n end
            return loop(n - 1, acc + keep())
        end
        return loop(3000, 0)",
        "
        local floor = math.floor
        local function loop(n, acc)
            if n == 0 then return floor(acc) end
            return loop(n - 1, acc + 0.5)
        end
        local function last(n) return floor(n * 1.5) end
        local function chain(n, acc)
            if n == 0 then return last(acc) end
            return chain(n - 1, acc + 1)
        end
        return loop(3000, 0), chain(3000, 0)",
        "
        local function make()
            local other
            local function loop(n, acc)
                if n == 0 then return acc end
                return other(n - 1, acc + 1)
            end
            return loop, function(f) other = f end
        end
        local a, set_a = make()
        local b, set_b = make()
        set_a(b)
        set_b(a)
        return a(3000, 0)",
    ] {
        for profile in TWO_PROFILES {
            agree(profile, source);
        }
    }
}

#[test]
fn a_coroutine_yield_inside_the_cycle_resumes_where_it_left() {
    let source = format!(
        "{MUTUAL}
        local base = odd
        function odd(n, total)
            if n % 997 == 0 then coroutine.yield(n) end
            return base(n, total)
        end
        local co = coroutine.wrap(function() return 'done', even(5000, 0) end)
        local seen = {{}}
        while true do
            local a, b = co()
            seen[#seen + 1] = tostring(a) .. ':' .. tostring(b)
            if a == 'done' then break end
        end
        return table.concat(seen, ',')"
    );
    for profile in TWO_PROFILES {
        agree(profile, &source);
    }
}

#[test]
fn types_that_change_between_runs_record_another_version() {
    let source = format!(
        "{MUTUAL}
        return even(3000, 0), even(3000, 0.5), odd(3001, 2)"
    );
    agree_native(&TWO_PROFILES, &source);
}

#[test]
fn a_rejected_tail_cycle_leaves_later_cycles_to_the_tracer() {
    // The branch alternates every call, and side traces of tail loops are
    // refused, so the tracer gives `alternate` up: its landings then skip
    // the lookup. The cycle that runs afterwards still gets its trace.
    let source = format!(
        "{MUTUAL}
        local function alternate(n, acc)
            if n == 0 then return acc end
            if n % 2 == 0 then
                return alternate(n - 1, acc + 1)
            else
                return alternate(n - 1, acc + 2)
            end
        end
        local first = alternate(40000, 0)
        return first, even(40000, 0), alternate(1000, 0)"
    );
    for (_, stats) in agree_native(&TWO_PROFILES, &source) {
        // One iteration of `even`'s trace runs two of its 40000 tail calls;
        // `alternate`'s traces leave after one iteration each.
        assert!(stats.iterations > 19_000, "{stats:?}");
    }
}

#[test]
fn a_count_hook_sees_every_instruction_of_a_tail_cycle() {
    // Under a hook the fast loop runs the cycle's tail calls itself; the
    // counts must match the interpreter's.
    let source = format!(
        "{MUTUAL}
        local warm = even(5000, 0)
        local count = 0
        debug.sethook(function() count = count + 1 end, '', 7)
        local hooked = even(5000, 0)
        debug.sethook()
        return warm, hooked, count"
    );
    for profile in TWO_PROFILES {
        agree(profile, &source);
    }
}
