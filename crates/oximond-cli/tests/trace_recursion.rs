//! Function-entry traces: self recursion, through compiled Lua source.
//!
//! A function that calls itself directly is traced from its entry: the
//! recorder follows one activation to its return, and each direct self call
//! in it runs the trace again as a nested native activation. An exit in a
//! nested activation rebuilds the interpreter frames of every activation
//! below it, stopped at their self calls, so errors, tracebacks and
//! `debug.getinfo` see the stack the interpreter would have built.
//!
//! Every program runs under the interpreter and under tracing execution. The
//! results must match exactly (error messages and tracebacks included), and
//! the counters show whether the recursion ran natively.

use oximond::vm::{TraceAbortReason, TraceExitKind, TraceGuardKind, TraceStatistics};
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

/// The float profiles (one of them `LuaJIT`'s) and one integer profile.
const PROFILES: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::LuaJit21,
];

/// Every compatibility profile.
const ALL_PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

/// One float profile and one integer profile.
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
/// the results agree, and returns them with the tracing statistics.
fn agree(profile: CompatibilityProfile, source: &str) -> (String, TraceStatistics) {
    let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
    let (tracing, stats) = run(profile, ExecutionMode::Tracing, source);
    assert_eq!(
        tracing, expected,
        "{profile:?} tracing\n{source}\n{stats:?}"
    );
    (expected, stats)
}

/// Like [`agree`] for every profile in `profiles`, requiring that a function
/// trace compiled and made at least `self_calls` native self calls.
fn agree_recursive(
    profiles: &[CompatibilityProfile],
    source: &str,
    self_calls: u64,
) -> Vec<(String, TraceStatistics)> {
    profiles
        .iter()
        .map(|profile| {
            let (result, stats) = agree(*profile, source);
            assert!(
                stats.compiled_function_traces > 0
                    && stats.native_self_calls >= self_calls
                    && stats.native_returns > 0,
                "{profile:?}: the recursion did not run natively\n{source}\n{stats:?}"
            );
            (result, stats)
        })
        .collect()
}

#[test]
fn fib_runs_natively() {
    // The shape of `benchmarks/lua/calls/recursive_fib.lua`.
    let source = "
        local function fib(n)
            if n < 2 then
                return n
            end
            return fib(n - 1) + fib(n - 2)
        end
        local total = 0
        for _ = 1, 4 do
            total = total + fib(20)
        end
        return total, fib(0), fib(1), fib(2)";
    for (_, stats) in agree_recursive(&PROFILES, source, 50_000) {
        // The base case becomes the side of the guard the root records.
        assert!(stats.side_installs > 0, "{stats:?}");
    }
}

#[test]
fn a_global_recursive_function_runs_natively() {
    let source = "
        function fib(n)
            if n < 2 then return n end
            return fib(n - 1) + fib(n - 2)
        end
        return fib(21)";
    // Lua 5.2 and later read a global out of the `_ENV` table, which no
    // function trace reads its callee from.
    agree_recursive(
        &[CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21],
        source,
        10_000,
    );
    agree(CompatibilityProfile::Lua54, source);
}

#[test]
fn factorial_runs_natively() {
    // Lua 5.4's integers wrap past 20!, as the interpreter's do.
    let source = "
        local function fact(n)
            if n <= 1 then return 1 end
            return n * fact(n - 1)
        end
        local total = 0
        for i = 1, 400 do
            total = total + fact(i % 30)
        end
        return total, fact(25)";
    agree_recursive(&PROFILES, source, 2_000);
}

/// A recursion whose self calls are leaves (a clamp that recurses once, a
/// tree recursion called at depth one) gains nothing from a function trace:
/// entering and leaving costs more than the one activation each entry runs.
/// The trace retires, and the entries stop going native.
#[test]
fn a_recursion_whose_self_calls_are_leaves_stops_entering_its_trace() {
    let source = "
        local function clamp(x)
            if x < 0 then return -clamp(-x) end
            if x > 10 then return 10 end
            return x
        end
        local function f(n)
            if n == 0 then return 1 end
            return f(n - 1) + f(n - 1)
        end
        local s = 0
        for i = 1, 20000 do s = s + clamp((i % 21) - 10) + f(1) end
        return s";
    for profile in PROFILES {
        let (_, stats) = agree(profile, source);
        // About 50,000 self calls, each a one-activation entry while a trace
        // is installed.
        assert!(
            stats.native_returns < 5_000,
            "{profile:?}: the leaf self calls kept entering\n{stats:?}"
        );
    }
}

#[test]
fn two_self_calls_with_two_parameters_run_natively() {
    // An ackermann-like recursion without tail calls: the inner call's
    // result is the outer call's argument.
    let source = "
        local function ack(m, n)
            if m == 0 then return n + 1 end
            if n == 0 then
                local r = ack(m - 1, 1)
                return r
            end
            local inner = ack(m, n - 1)
            local r = ack(m - 1, inner)
            return r
        end
        return ack(2, 300), ack(3, 4)";
    agree_recursive(&PROFILES, source, 10_000);
}

#[test]
fn a_recursive_sum_over_an_array_range_runs_natively() {
    let source = "
        local floor = math.floor
        local function sum(t, lo, hi)
            if lo == hi then return t[lo] end
            local mid = MIDDLE
            return sum(t, lo, mid) + sum(t, mid + 1, hi)
        end
        local values = {}
        for i = 1, 2000 do values[i] = i * 3 % 17 end
        local total = 0
        for _ = 1, 10 do total = total + sum(values, 1, #values) end
        return total";
    agree_recursive(
        &[CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21],
        &source.replace("MIDDLE", "floor((lo + hi) / 2)"),
        20_000,
    );
    agree_recursive(
        &[CompatibilityProfile::Lua54],
        &source.replace("MIDDLE", "(lo + hi) // 2"),
        20_000,
    );
}

#[test]
fn recursion_beyond_the_native_depth_cap_stays_correct() {
    // Deeper than one native run nests: the run leaves at its cap and the
    // interpreter's next self call enters again from there.
    let source = "
        local function sum(n)
            if n == 0 then return 0 end
            return n + sum(n - 1)
        end
        local total = 0
        for i = 1, 30 do total = total + sum(i) end
        return total, sum(3000), sum(3500)";
    for (_, stats) in agree_recursive(&TWO_PROFILES, source, 3_000) {
        assert!(stats.unwound_activations > 0, "{stats:?}");
    }
}

#[test]
fn an_exit_deep_in_the_recursion_rebuilds_the_frames() {
    // The base case calls `debug.traceback` and `debug.getinfo`, which no
    // trace can hold: every run leaves from its deepest activation, and the
    // interpreter sees every frame below it.
    let source = "
        local seen
        local function levels()
            local level = 1
            while debug.getinfo(level, 'l') do level = level + 1 end
            return level
        end
        local function walk(n)
            if n == 0 then
                local info = debug.getinfo(2, 'nSl')
                seen = table.concat({ debug.traceback('deep', 1), tostring(levels()),
                    tostring(info.name), tostring(info.currentline) }, '\\n')
                return 0
            end
            return walk(n - 1) + 1
        end
        local total = 0
        for i = 1, 60 do total = total + walk(i % 40) end
        return total, seen";
    for profile in TWO_PROFILES {
        let (_, stats) = agree(profile, source);
        // Every run leaves from its deepest activation: none returns.
        assert!(
            stats.compiled_function_traces > 0
                && stats.native_self_calls > 500
                && stats.unwound_activations > 500,
            "{profile:?}: {stats:?}"
        );
    }
}

#[test]
fn a_type_change_deep_in_the_recursion_leaves_and_stays_correct() {
    // At one depth a value turns into a string: the guard on its type (or
    // the arithmetic) leaves there, and the interpreter finishes the call.
    let source = "
        local function f(n, x)
            if n == 0 then return x end
            local v = f(n - 1, x)
            return v + 1
        end
        local total = 0
        for i = 1, 200 do total = total + f(30, i) end
        local ok, message = pcall(f, 30, 'x')
        local coerced = f(30, '10')
        return total, ok, message, coerced";
    agree_recursive(&TWO_PROFILES, source, 3_000);
}

#[test]
fn an_error_deep_in_the_recursion_reports_as_the_interpreter_does() {
    let source = "
        local function report(message)
            local lines = { tostring(message), debug.traceback('tb', 2) }
            local level = 2
            while true do
                local info = debug.getinfo(level, 'nSl')
                if not info then break end
                lines[#lines + 1] = table.concat({ tostring(info.name),
                    tostring(info.namewhat), info.what, tostring(info.currentline) }, ' ')
                level = level + 1
            end
            return table.concat(lines, '\\n')
        end
        local function f(n, fail)
            if n == 0 then
                if fail then error('boom at the bottom') end
                return 0
            end
            return f(n - 1, fail) + n
        end
        local total = 0
        for i = 1, 100 do total = total + f(i % 25, false) end
        local traced = select(2, xpcall(f, report, 20, true))
        local ok, message = pcall(f, 12, true)
        return total, traced, ok, message";
    agree_recursive(&TWO_PROFILES, source, 1_000);
}

#[test]
fn unbounded_recursion_overflows_at_the_interpreters_depth() {
    // The limit is learned natively first, then a recursion with no end
    // overflows the stack: the error and the depth it reached match.
    let source = "
        local deepest = 0
        local function note(n) if n > deepest then deepest = n end end
        local function down(n, limit)
            if n >= limit then return 0 end
            if n % 64 == 0 then note(n) end
            return down(n + 1, limit) + 1
        end
        local total = 0
        for i = 1, 40 do total = total + down(0, i * 5) end
        local ok, message = pcall(down, 0, math.huge)
        return total, ok, message, deepest";
    for profile in TWO_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn rebinding_the_recursive_global_stays_correct() {
    let source = "
        function count(n)
            if n == 0 then return 0 end
            return count(n - 1) + 1
        end
        local original = count
        local first = 0
        for i = 1, 50 do first = first + count(i) end
        -- Rebound between calls: the entry no longer finds its closure.
        count = function(n) return -n end
        local second = original(40)
        count = original
        -- Rebound from the base case, through a native function call.
        local function rebinding(n)
            if n == 0 then
                rawset(_G, 'count', function(m) return 1000 end)
                return 0
            end
            return count(n - 1) + 1
        end
        count = rebinding
        local third = rebinding(30)
        count = original
        local fourth = 0
        for i = 1, 50 do fourth = fourth + count(i) end
        return first, second, third, fourth";
    // Lua 5.2 and later read a global out of the `_ENV` table, which no
    // function trace reads its callee from.
    agree_recursive(
        &[CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21],
        source,
        1_000,
    );
    agree(CompatibilityProfile::Lua54, source);
}

#[test]
fn recursion_inside_a_coroutine_and_a_yield_from_the_base_case() {
    let source = "
        local function fib(n)
            if n < 2 then return n end
            return fib(n - 1) + fib(n - 2)
        end
        local inside = coroutine.wrap(function() return fib(20) end)()
        local function walk(n)
            if n == 0 then coroutine.yield('bottom') return 1 end
            return walk(n - 1) + 1
        end
        local co = coroutine.wrap(function()
            local total = 0
            for i = 1, 40 do total = total + walk(i % 30) end
            return 'done', total
        end)
        local seen = {}
        while true do
            local a, b = co()
            if a == 'done' then seen[#seen + 1] = tostring(b) break end
            seen[#seen + 1] = a
        end
        return inside, table.concat(seen, ',')";
    agree_recursive(&TWO_PROFILES, source, 5_000);
}

#[test]
fn a_hook_set_from_the_base_case_sees_every_later_event() {
    // The hook is set from an exit path (the base case calls a native the
    // trace cannot hold), so every frame is the interpreter's by then, and
    // the trace is not entered again while the hook is set.
    let source = "
        local lines = 0
        local function walk(n)
            if n == 0 then
                if not debug.gethook() then
                    debug.sethook(function() lines = lines + 1 end, 'l')
                end
                return 0
            end
            return walk(n - 1) + 1
        end
        local function plain(n)
            if n == 0 then return 0 end
            return plain(n - 1) + 1
        end
        local before = 0
        for i = 1, 200 do before = before + plain(i % 40) end
        local total = walk(30) + walk(30)
        debug.sethook()
        return before, total, lines";
    for profile in TWO_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn shapes_the_recorder_refuses_stay_correct() {
    // A store, a table constructor, an upvalue store, a tail self call, a
    // vararg function and a self call through a local the function writes.
    for source in [
        "
        local t = {}
        local function f(n)
            if n == 0 then return 0 end
            t[n] = n
            return f(n - 1) + t[n]
        end
        local s = 0
        for i = 1, 100 do s = s + f(i % 30) end
        return s",
        "
        local function f(n)
            if n == 0 then return 0 end
            local t = { n }
            return f(n - 1) + t[1]
        end
        local s = 0
        for i = 1, 100 do s = s + f(i % 30) end
        return s",
        "
        local calls = 0
        local function f(n)
            if n == 0 then return 0 end
            calls = calls + 1
            return f(n - 1) + 1
        end
        local s = 0
        for i = 1, 100 do s = s + f(i % 30) end
        return s, calls",
        "
        local function f(n, acc)
            if n == 0 then return acc end
            if n % 2 == 0 then return f(n - 1, acc + 1) end
            return f(n - 1, acc) + 1
        end
        local s = 0
        for i = 1, 100 do s = s + f(i % 30, 0) end
        return s",
        "
        local function f(n, ...)
            if n == 0 then return select('#', ...) end
            return f(n - 1, n, ...) + 1
        end
        local s = 0
        for i = 1, 100 do s = s + f(i % 20) end
        return s",
        "
        local function f(n)
            if n == 0 then return 0 end
            local g = f
            if n % 7 == 0 then g = function(m) return m end end
            return g(n - 1) + 1
        end
        local s = 0
        for i = 1, 100 do s = s + f(i % 30) end
        return s",
    ] {
        for profile in TWO_PROFILES {
            agree(profile, source);
        }
    }
}

#[test]
fn closures_of_one_prototype_run_with_their_own_cells() {
    // Each closure's self calls go to itself: the entry checks the cell.
    let source = "
        local function make(step)
            local f
            f = function(n)
                if n == 0 then return 0 end
                return f(n - 1) + step
            end
            return f
        end
        local one, three = make(1), make(3)
        local total = 0
        for i = 1, 100 do total = total + one(i % 40) + three(i % 30) end
        return total";
    agree_recursive(&TWO_PROFILES, source, 1_000);
}

#[test]
fn ackermann_with_tail_calls_runs_natively() {
    // `return ack(m - 1, ack(m, n - 1))`: the inner call's open result feeds
    // a tail call of the function itself, which starts the activation over.
    let source = "
        local function ack(m, n)
            if m == 0 then return n + 1 end
            if n == 0 then return ack(m - 1, 1) end
            return ack(m - 1, ack(m, n - 1))
        end
        local total = 0
        for i = 1, 30 do total = total + ack(2, i) end
        return total, ack(2, 300), ack(3, 5)";
    agree_recursive(&PROFILES, source, 100_000);
}

/// A function that tail calls itself on some paths and calls itself on
/// others, and reports the stack at the bottom: the frames an exit rebuilds
/// carry the tail call counts and names the interpreter's in-place tail
/// calls give them (Lua 5.1 shows each as a traceback level, 5.4 flags it).
fn tail_mixing(bottom: &str) -> String {
    format!(
        "
        local function levels()
            local level = 1
            while debug.getinfo(level, 'l') do level = level + 1 end
            return level
        end
        local seen
        local failing = false
        local function f(n, acc)
            if n == 0 then
                {bottom}
                return acc
            end
            if n % 3 == 0 then return f(n - 1, acc + 1) end
            return f(n - 1, acc) + 1
        end
        local total = 0
        for i = 1, 200 do total = total + f(i % 40, 0) end
        local deep = f(37, 0)
        return total, deep, seen"
    )
}

#[test]
fn tail_calls_of_the_function_restore_their_counts_and_names() {
    let quiet = tail_mixing("");
    agree_recursive(&TWO_PROFILES, &quiet, 3_000);
    // The bottom reports the stack once the trace runs: every frame on it
    // came from an exit's rebuild.
    let report = tail_mixing(
        "if acc > 3 then
            local info = debug.getinfo(2, 'nSlt')
            seen = table.concat({ debug.traceback('bottom', 1), tostring(levels()),
                tostring(info.name), tostring(info.currentline),
                tostring(info.istailcall) }, '\\n')
        end",
    );
    for profile in TWO_PROFILES {
        agree(profile, &report);
    }
    // An error at the bottom, caught with the stack in the handler.
    let failing = tail_mixing("if failing and acc > 5 then error('bottom') end");
    let failing = failing.replace(
        "local deep = f(37, 0)",
        "failing = true
        local deep = select(2, xpcall(f, function(m) return m .. debug.traceback('', 2) end, 37, 0))",
    );
    for profile in TWO_PROFILES {
        agree(profile, &failing);
    }
}

/// The shape of `benchmarks/lua/algorithm/binary_trees.lua`: a recursive
/// function that builds a table around its recursive calls, and another that
/// walks the tables with a field read that is absent at the leaves.
const TREES: &str = "
    local function make(depth, seed)
        if depth == 0 then
            return {value = seed}
        end
        return {
            value = seed,
            left = make(depth - 1, seed * 2 - 1),
            right = make(depth - 1, seed * 2),
        }
    end
    local function check(tree)
        if not tree.left then
            return tree.value
        end
        return tree.value + check(tree.left) - check(tree.right)
    end";

#[test]
fn a_recursive_builder_and_a_recursive_walker_run_natively() {
    // Deeper than a run has lanes for, so the runs end and start again.
    let source = format!(
        "{TREES}
        local total = 0
        for round = 1, 6 do total = total + check(make(11, round)) end
        return total, check(make(5, 1)), check(make(0, 7)), check(make(1, 3))"
    );
    for profile in TWO_PROFILES {
        let (_, stats) = agree(profile, &source);
        assert!(
            stats.compiled_function_traces >= 2 && stats.native_self_calls > 30_000,
            "{profile:?}: both functions run as function traces\n{stats:?}"
        );
        // The builder makes its tables natively, one per call.
        assert!(
            stats.table_new_helper_calls > 20_000,
            "{profile:?}: {stats:?}"
        );
        assert_eq!(stats.aborts(TraceAbortReason::SelfCall), 0, "{profile:?}");
        assert_eq!(
            stats.aborts(TraceAbortReason::UnsupportedOperand),
            0,
            "{profile:?}"
        );
        // A run that is out of lanes ends, which is no branch to record a
        // side for: one side for each function takes the path the first
        // recording did not.
        assert!(stats.side_recordings <= 4, "{profile:?}: {stats:?}");
    }
}

/// Trees whose even levels share one child between both sides when `share`
/// is set, and walkers that tell the two apart with `==` and with `rawequal`.
/// A function trace gives a table a new lane at every read, so only the
/// handles tell a shared child from two equal ones.
const SHARED_CHILDREN: &str = "
    local function make(depth, share)
        if depth == 0 then return {v = 1} end
        if share and depth % 2 == 0 then
            local child = make(depth - 1, share)
            return {left = child, right = child, v = depth}
        end
        return {left = make(depth - 1, share), right = make(depth - 1, share), v = depth}
    end
    local function eq(tree)
        local l = tree.left
        if not l then return tree.v end
        local r = tree.right
        if l == r then return 1000 + eq(l) end
        return eq(l) + eq(r)
    end
    local function raw(tree)
        local l = tree.left
        if not l then return tree.v end
        if rawequal(l, tree.right) then return 1000 + raw(l) end
        return raw(l) + raw(tree.right)
    end";

#[test]
fn a_walker_warmed_on_distinct_children_finds_the_shared_ones() {
    // The recording takes the unequal branch of `l == r`; a node whose two
    // children are one table must still take the other.
    let source = format!(
        "{SHARED_CHILDREN}
        local distinct, shared = make(8, false), make(8, true)
        local total = 0
        for round = 1, 200 do total = total + eq(distinct) + raw(distinct) end
        for round = 1, 200 do total = total + eq(shared) + raw(shared) end
        return total, eq(shared), raw(shared), eq(distinct), raw(distinct)"
    );
    // `make` alone recurses natively a few hundred times; the walkers make
    // over 100000 native self calls between them, so this shows they ran.
    for (_, stats) in agree_recursive(&ALL_PROFILES, &source, 100_000) {
        assert!(stats.compiled_function_traces >= 3, "{stats:?}");
    }
}

#[test]
fn a_walker_warmed_on_shared_children_does_not_see_distinct_ones_as_shared() {
    // The reverse order: the recording takes the equal branch, and the guard
    // must leave for two children that are different tables.
    let source = format!(
        "{SHARED_CHILDREN}
        local distinct, shared = make(8, false), make(8, true)
        local total = 0
        for round = 1, 200 do total = total + eq(shared) + raw(shared) end
        for round = 1, 200 do total = total + eq(distinct) + raw(distinct) end
        return total, eq(shared), raw(shared), eq(distinct), raw(distinct)"
    );
    // `make` alone recurses natively a few hundred times; the walkers make
    // over 100000 native self calls between them, so this shows they ran.
    for (_, stats) in agree_recursive(&ALL_PROFILES, &source, 100_000) {
        assert!(stats.compiled_function_traces >= 3, "{stats:?}");
    }
}

#[test]
fn a_shared_table_used_as_a_key_in_a_function_trace_is_one_key() {
    // Every read of a table in a function trace has a lane of its own, so a
    // table key must be found by identity, never by lane number. A table key
    // is not a native key today: the recorder refuses `marked[l]` as an
    // unsupported operand, so `walk` runs in the interpreter, finds the entry
    // the filler made for the shared child, and never uses a lane number.
    // Only `make` recurses natively (about 2000 calls); a walker that
    // recorded would add tens of thousands. If table keys ever record, this
    // must become a native_self_calls check on the walker.
    let source = format!(
        "{SHARED_CHILDREN}
        local function fill(tree, marked, depth)
            local l = tree.left
            if not l then return end
            if depth % 2 == 0 then marked[l] = depth end
            fill(l, marked, depth - 1)
            fill(tree.right, marked, depth - 1)
        end
        local function walk(tree, marked)
            local l = tree.left
            if not l then return 0 end
            local mark = marked[l] or 0
            return mark + walk(l, marked) + walk(tree.right, marked)
        end
        local shared, distinct = make(10, true), make(10, false)
        local marked_shared, marked_distinct = {{}}, {{}}
        fill(shared, marked_shared, 10)
        fill(distinct, marked_distinct, 10)
        local total = 0
        for round = 1, 300 do
            total = total + walk(distinct, marked_distinct) + walk(shared, marked_shared)
        end
        return total, walk(shared, marked_shared), walk(distinct, marked_distinct)"
    );
    for profile in PROFILES {
        let (_, stats) = agree(profile, &source);
        assert!(
            stats.recording_aborts[TraceAbortReason::UnsupportedOperand.index()] > 0
                && stats.hash_helper_calls == 0
                && stats.native_self_calls < 2_500,
            "{profile:?}: {stats:?}"
        );
    }
}

#[test]
fn an_absent_field_read_on_a_table_argument_neither_exits_nor_calls_the_hash_helper() {
    // The leaves lack `left`: the read is nil from the table's row, with no
    // lookup in the VM, and the leaf and the inner node are two paths of one
    // tree of traces. After a few exits that make the second path, the walk
    // goes on without leaving: twice the walks, the same exits.
    let walks = |rounds: u32| {
        format!(
            "{TREES}
            local tree = make(9, 1)
            local total = 0
            for _ = 1, {rounds} do total = total + check(tree) end
            return total"
        )
    };
    for profile in TWO_PROFILES {
        let (_, warm) = agree(profile, &walks(30));
        let (_, longer) = agree(profile, &walks(60));
        for stats in [&warm, &longer] {
            assert_eq!(stats.hash_helper_calls, 0, "{profile:?}: {stats:?}");
            assert!(
                stats.side_installs >= 1 && stats.native_self_calls > 25_000,
                "{profile:?}: {stats:?}"
            );
        }
        assert_eq!(
            longer.exits_of(TraceExitKind::GuardFail),
            warm.exits_of(TraceExitKind::GuardFail),
            "{profile:?}: a node that lacks the field is a path of the trace\n{warm:?}\n{longer:?}"
        );
        assert_eq!(
            longer.side_exits_of(TraceGuardKind::ArrayBounds),
            warm.side_exits_of(TraceGuardKind::ArrayBounds),
            "{profile:?}"
        );
        assert!(
            longer.native_self_calls > 2 * warm.native_self_calls - 2_000,
            "{profile:?}: the second half ran natively as well\n{warm:?}\n{longer:?}"
        );
    }
}

#[test]
fn a_node_whose_fields_change_mid_run_leaves_the_trace_and_stays_correct() {
    // The trace has seen leaves that lack `left` and inner nodes that have
    // it. Then a leaf gets a metatable whose `__index` supplies the field, a
    // leaf gets `false` (present, and false), and an inner node loses the
    // field: each is a table the trace's rows cannot answer for.
    let program = |mutate: bool| {
        format!(
            "{TREES}
            local mutate = {mutate}
            local tree = make(8, 1)
            local function leaf(node, depth)
                if depth == 0 then return node end
                return leaf(node.right, depth - 1)
            end
            local total = 0
            for _ = 1, 40 do total = total + check(tree) end
            local through = leaf(tree, 8)
            local other = make(3, 100)
            if mutate then setmetatable(through, {{__index = {{left = other, right = other}}}}) end
            for _ = 1, 20 do total = total + check(tree) end
            local plain = leaf(tree.left, 7)
            if mutate then plain.left = false end
            for _ = 1, 20 do total = total + check(tree) end
            local inner = leaf(tree.left.left, 6)
            if mutate then inner.left = nil end
            for _ = 1, 20 do total = total + check(tree) end
            return total, check(tree), through.left == other, rawget(through, 'left')"
        )
    };
    for profile in TWO_PROFILES {
        let (changed, stats) = agree(profile, &program(true));
        let (unchanged, quiet) = agree(profile, &program(false));
        assert_ne!(changed, unchanged, "{profile:?}: the changes show");
        assert!(stats.native_self_calls > 5_000, "{profile:?}: {stats:?}");
        assert!(
            stats.exits_of(TraceExitKind::GuardFail) > quiet.exits_of(TraceExitKind::GuardFail),
            "{profile:?}: a table that answers another way leaves the trace\n{stats:?}\n{quiet:?}"
        );
    }
}

#[test]
fn a_field_of_another_type_leaves_the_trace_and_reports_as_the_interpreter_does() {
    let source = format!(
        "{TREES}
        local tree = make(8, 3)
        local total = 0
        for _ = 1, 40 do total = total + check(tree) end
        local function report(message)
            return message .. debug.traceback('', 2)
        end
        tree.right.right.value = 'x'
        local ok, message = pcall(check, tree)
        local traced = select(2, xpcall(check, report, tree))
        tree.right.right.value = '12'
        local coerced = check(tree)
        return total, ok, message, traced, coerced"
    );
    for profile in TWO_PROFILES {
        let (result, stats) = agree(profile, &source);
        assert!(result.contains("attempt to"), "{profile:?}: {result}");
        assert!(stats.native_self_calls > 3_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn an_error_at_the_bottom_of_a_builder_reports_as_the_interpreter_does() {
    let source = "
        local failing = false
        local function make(depth, seed)
            if depth == 0 then
                if failing then error('boom at the bottom') end
                return {value = seed}
            end
            return {
                value = seed,
                left = make(depth - 1, seed * 2),
                right = make(depth - 1, seed * 2 + 1),
            }
        end
        local function report(message)
            local lines = { tostring(message), debug.traceback('tb', 2) }
            local level = 2
            while true do
                local info = debug.getinfo(level, 'nSl')
                if not info then break end
                lines[#lines + 1] = table.concat({ tostring(info.name),
                    tostring(info.namewhat), info.what, tostring(info.currentline) }, ' ')
                level = level + 1
            end
            return table.concat(lines, '\\n')
        end
        local made = 0
        for i = 1, 60 do
            local tree = make(6, i)
            made = made + tree.value
        end
        failing = true
        local traced = select(2, xpcall(make, report, 7, 1))
        local ok, message = pcall(make, 4, 1)
        failing = false
        local after = make(3, 5)
        return made, traced, ok, message, after.value, after.left.right.value";
    for profile in TWO_PROFILES {
        let (_, stats) = agree(profile, source);
        assert!(
            stats.compiled_function_traces > 0 && stats.native_self_calls > 2_000,
            "{profile:?}: {stats:?}"
        );
    }
}

#[test]
fn a_builder_without_an_end_overflows_the_stack_at_the_interpreters_depth() {
    // The builder records (it writes no upvalue), warms with bounded limits,
    // then runs past the stack's end natively.
    let source = "
        local function build(d, limit)
            if d == limit then return {depth = d} end
            return {depth = d, inner = build(d + 1, limit)}
        end
        local total = 0
        for i = 1, 60 do total = total + build(1, 5 + i % 7).depth end
        local ok, message = pcall(build, 1, -1)
        local traced = select(2, xpcall(build, debug.traceback, 1, -1))
        local lines = 0
        for _ in tostring(traced):gmatch('[^\\n]+') do lines = lines + 1 end
        return total, ok, message, lines";
    for profile in TWO_PROFILES {
        let (_, stats) = agree(profile, source);
        assert!(
            stats.compiled_function_traces > 0 && stats.native_self_calls > 0,
            "{profile:?}: {stats:?}"
        );
        assert!(stats.unwound_activations > 0, "{profile:?}: {stats:?}");
    }
}

#[test]
fn walkers_whose_leaves_leave_the_trace_stop_entering_it() {
    // Every walk ends at a leaf whose path the trace has no side for (the
    // string length cannot be recorded), so each entry unwinds its self
    // calls. Those calls did no work, and the trace must retire.
    for leaf in ["{value = 'leaf'}", "{value = 'leaf', left = false}"] {
        let source = format!(
            "local function mk(d)
                if d == 0 then return {leaf} end
                return {{value = 'n', left = mk(d - 1), right = mk(d - 1)}}
            end
            local function cnt(t)
                if not t.left then return #t.value end
                return cnt(t.left) + cnt(t.right)
            end
            local tree = mk(9)
            local n = 0
            for i = 1, 100 do n = n + cnt(tree) end
            return n"
        );
        for profile in TWO_PROFILES {
            let (_, stats) = agree(profile, &source);
            assert!(
                stats.unwound_activations < stats.native_self_calls / 2
                    || stats.native_entries < 20_000,
                "{profile:?}: {stats:?}"
            );
        }
    }
}

#[test]
fn collecting_garbage_while_trees_are_built_keeps_the_heap_bounded() {
    // Every round builds a tree and drops it, collections come in the middle
    // of the recursion, and a table made natively must survive them while
    // the recursion above it still holds it.
    let source = format!(
        "{TREES}
        local peak = 0
        local total = 0
        for round = 1, 60 do
            total = total + check(make(9, round))
            local used = collectgarbage('count')
            if used > peak then peak = used end
        end
        local kept = make(9, 3)
        collectgarbage()
        collectgarbage()
        local settled = collectgarbage('count')
        return total, check(kept), peak < 60000, settled < 20000"
    );
    for profile in TWO_PROFILES {
        let (result, stats) = agree(profile, &source);
        assert!(result.contains("Boolean(true)"), "{profile:?}: {result}");
        assert!(
            stats.table_new_helper_calls > 10_000,
            "{profile:?}: {stats:?}"
        );
    }
}

#[test]
fn a_table_made_in_a_function_trace_does_not_cross_a_self_call_natively() {
    // g makes `n` and passes it to itself; the callee reads fields of it, so
    // the recording must refuse rather than read a row of an earlier run.
    let source = "
        local function g(t, d)
            if d == 0 then return t.x end
            local c, n
            if t.k then
                c = t.c
                n = {x = d}
            else
                n = {x = d}
                c = t.c
            end
            return g(n, 0) + g(c, d - 1)
        end
        local function mk(d)
            if d == 0 then return {x = 1000, k = false} end
            return {x = 1000 * d, k = (d % 3 == 0), c = mk(d - 1)}
        end
        local tree = mk(30)
        local s = 0
        for i = 1, 400 do s = s + g(tree, 1 + i % 25) end
        return s";
    for profile in TWO_PROFILES {
        let (_, stats) = agree(profile, source);
        assert_eq!(stats.table_new_helper_calls, 0, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_table_a_self_call_returned_does_not_cross_another_self_call() {
    // `r` is made by the callee and handed to a second self call, whose
    // field reads must see the table itself, not a row an earlier run left.
    let source = "
        local function h(t, d)
            if d == 0 then return {x = t.x + 1, c = t.c} end
            local r = h(t.c, d - 1)
            return h(r, 0)
        end
        local function mk(d, base)
            if d == 0 then return {x = base} end
            return {x = base + d, c = mk(d - 1, base)}
        end
        local s = 0
        for i = 1, 3000 do s = s + h(mk(12, i), 1 + i % 10).x end
        return s";
    for profile in TWO_PROFILES {
        let (_, stats) = agree(profile, source);
        assert!(
            stats.compiled_function_traces >= 1 && stats.native_self_calls > 0,
            "{profile:?}: the builder still runs natively\n{stats:?}"
        );
    }
}
