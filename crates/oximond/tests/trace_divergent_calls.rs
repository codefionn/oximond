//! Loops that call a different function each iteration: a call site that
//! takes turns between N closures (distinct prototypes, one prototype with
//! other upvalues, chunks made by `load`), a callee swapped in mid-loop, and
//! an error raised by one target. Every program gives the interpreter's
//! result, and the loops run natively. Also the producer and consumer pairs
//! of open results that nest (`f (g (x))` where `g` holds a pair of its own).

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
    // A side compiled into its tree is entered by a jump, which is counted
    // only on request (a side compiled apart is called, and always counted).
    lua.vm_mut().count_trace_links();
    let results = lua.exec(source).map_err(|error| format!("{error:?}"));
    (results, lua)
}

/// Runs `source` in both modes of every profile; the results agree. Returns
/// the statistics of the tracing run of every profile.
fn run_each(source: &str) -> Vec<TraceStatistics> {
    let mut all = Vec::new();
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(
            format!("{actual:?}"),
            format!("{expected:?}"),
            "{profile:?}"
        );
        all.push(lua.vm().jit_statistics().trace);
    }
    all
}

/// A loop that picks one of `n` closures from a table by iteration.
fn table_of_closures(n: usize, same_prototype: bool) -> String {
    let mut source = String::new();
    source.push_str("local fns = {}\n");
    if same_prototype {
        writeln!(
            source,
            "for k = 1, {n} do\n  local step = k * 3\n  local count = 0\n  \
             fns[k] = function(x) count = count + 1; return x + step + count % 3 end\nend"
        )
        .unwrap();
    } else {
        for k in 1..=n {
            writeln!(
                source,
                "fns[{k}] = function(x) return x * {} + {k} end",
                k % 5 + 1
            )
            .unwrap();
        }
    }
    writeln!(
        source,
        "local sum = 0\nfor i = 1, 30000 do\n  sum = (sum + fns[i % {n} + 1](i)) % 1000003\nend\nreturn sum"
    )
    .unwrap();
    source
}

fn assert_native(stats: &TraceStatistics, what: &str) {
    assert!(stats.traces_compiled > 0, "{what}: {stats:?}");
    assert!(stats.native_entries > 0, "{what}: {stats:?}");
    assert!(stats.iterations > 20000, "{what}: {stats:?}");
}

#[test]
fn a_call_site_over_n_closures_of_distinct_prototypes_runs_natively() {
    for n in [2, 4, 16] {
        for stats in run_each(&table_of_closures(n, false)) {
            assert_native(&stats, &format!("n = {n}"));
            // One side for each target past the root's, each hanging from the
            // guard of the one before.
            assert_eq!(stats.side_installs, (n - 1) as u64, "n = {n}: {stats:?}");
            assert!(stats.native_entries < 400, "n = {n}: {stats:?}");
            assert_eq!(stats.aborts(TraceAbortReason::Divergence), 0, "{stats:?}");
        }
    }
}

#[test]
fn a_call_site_over_n_closures_of_one_prototype_runs_natively() {
    for n in [2, 4, 16] {
        for stats in run_each(&table_of_closures(n, true)) {
            assert_native(&stats, &format!("n = {n}"));
            assert_eq!(stats.aborts(TraceAbortReason::Divergence), 0, "{stats:?}");
        }
    }
}

#[test]
fn functions_made_by_load_are_trace_targets() {
    for n in [2, 4, 16] {
        let source = format!(
            "local load = loadstring or load
             local fns = {{}}
             for k = 1, {n} do
               fns[k] = load('local k = ' .. k .. '\\nreturn function(x) return x * k + 1 end')()
             end
             -- one chunk, called for several closures
             local maker = load('local k = ...\\nreturn function(x) return x + k end')
             local extra = {{}}
             for k = 1, {n} do extra[k] = maker(k) end
             local sum = 0
             for i = 1, 30000 do
               sum = (sum + fns[i % {n} + 1](i) + extra[(i * 7) % {n} + 1](i)) % 1000003
             end
             return sum"
        );
        for stats in run_each(&source) {
            assert_native(&stats, &format!("n = {n}"));
            // The `n` chunks of `load` have a prototype each: a side for each
            // past the root's. The closures of the one chunk made `n` times
            // share a prototype guard and add none.
            assert_eq!(stats.side_installs, (n - 1) as u64, "n = {n}: {stats:?}");
            assert!(stats.native_entries < 400, "n = {n}: {stats:?}");
            assert_eq!(stats.aborts(TraceAbortReason::Divergence), 0, "{stats:?}");
        }
    }
}

#[test]
fn a_callee_swapped_mid_loop_leaves_the_trace() {
    let source = "
        local fns = {}
        for k = 1, 4 do fns[k] = function(x) return x + k end end
        local sum = 0
        for i = 1, 30000 do
          if i == 20000 then
            fns[2] = function(x) return x * 2 end
            fns[3] = 7
          end
          local f = fns[i % 4 + 1]
          if type(f) == 'function' then sum = (sum + f(i)) % 1000003 else sum = sum + f end
        end
        return sum";
    // The same loop with no swap: one trace, a prototype guard, no side.
    for stats in run_each(&source.replace("i == 20000", "i == -1")) {
        assert_eq!(stats.side_installs, 0, "{stats:?}");
    }
    for stats in run_each(source) {
        assert!(stats.traces_compiled > 0, "{stats:?}");
        // The new function of another prototype and the number that is no
        // function each leave through their guard and get a side after the
        // swap, and the iterations after it stay native.
        assert_eq!(stats.side_installs, 2, "{stats:?}");
        assert!(stats.side_entries >= 1, "{stats:?}");
        assert!(stats.iterations > 29000, "{stats:?}");
        assert!(stats.native_entries < 100, "{stats:?}");
    }
}

#[test]
fn an_error_raised_by_one_target_is_the_interpreters() {
    let source = "
        local fns = {}
        for k = 1, 4 do fns[k] = function(x) return x + k end end
        fns[3] = function(x) if x > 25000 then error('boom ' .. x) end return x end
        local sum = 0
        local ok, message = pcall(function()
          for i = 1, 30000 do
            sum = (sum + fns[i % 4 + 1](i)) % 1000003
          end
        end)
        return ok, message, sum";
    // The error is the interpreter's: the first call of `fns[3]` past 25000.
    let (expected, _) = outcome(
        CompatibilityProfile::Lua54,
        ExecutionMode::Interpreter,
        source,
    );
    assert!(
        format!("{expected:?}").contains("boom 25002"),
        "{expected:?}"
    );
    for stats in run_each(source) {
        // The loop ran natively, the error-raising target had a side, and the
        // trace was entered long before the error.
        assert!(stats.native_entries > 0, "{stats:?}");
        assert!(stats.side_installs >= 1, "{stats:?}");
        assert!(stats.iterations > 20000, "{stats:?}");
    }
}

#[test]
fn producer_and_consumer_pairs_inside_an_inlined_producer_nest() {
    // `inner` holds a pair of its own (`twice (twice (a))`) between the call
    // of `inner` that makes open results and the call of `add` that takes
    // them.
    let source = "
                local function add(a, b) return a + (b or 0) end
        local function twice(a) return a * 2 end
        local function inner(a) local r = twice (twice (a)); return r end
        local sum = 0
        for i = 1, 30000 do
          sum = sum + add(inner(i))
        end
        return sum";
    for stats in run_each(source) {
        assert_native(&stats, "nested open results");
        assert_eq!(stats.aborts(TraceAbortReason::Divergence), 0, "{stats:?}");
    }
}
