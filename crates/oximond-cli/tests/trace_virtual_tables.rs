//! Table constructors that die inside a loop iteration stay virtual in a
//! trace: no table is allocated, the array and fields are SSA values, and an
//! exit that needs the table builds it from a recipe.
//!
//! Every program runs under the interpreter and under tracing execution in
//! every profile, and the results must match. Counters then show what the
//! trace did: how many real tables the run allocated, how many constructors
//! it charged as phantom debt, which exits it took and which recordings it
//! rejected. The invariant behind the accounting tests is that each
//! constructor runs exactly once, so real tables plus net phantom charges
//! equal the tables the interpreter allocated.

use oximond::runtime::Runtime;
use oximond::vm::{TraceAbortReason, TraceExitKind, TraceStatistics};
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

const ALL_PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

/// Iterations of the loops below. A trace needs a few hundred to compile.
const N: usize = 6_000;

struct Outcome {
    result: String,
    stats: TraceStatistics,
    /// Tables the runtime allocated.
    created: u64,
    /// Constructors charged as phantom debt and not taken back, and their
    /// bytes.
    elided: (u64, u64),
}

fn run(profile: CompatibilityProfile, mode: ExecutionMode, source: &str) -> Outcome {
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
    let runtime = lua.vm().runtime();
    Outcome {
        result: format!("{results:?}"),
        stats: lua.vm().jit_statistics().trace,
        created: runtime.tables_created(),
        elided: runtime.elided_tables_total(),
    }
}

/// Runs `source` in every profile under both engines and checks the results
/// agree. Returns each profile with the interpreter's and the tracing run.
fn agree(source: &str) -> Vec<(CompatibilityProfile, Outcome, Outcome)> {
    ALL_PROFILES
        .iter()
        .map(|&profile| {
            let interpreter = run(profile, ExecutionMode::Interpreter, source);
            let tracing = run(profile, ExecutionMode::Tracing, source);
            assert_eq!(
                tracing.result, interpreter.result,
                "{profile:?}\n{source}\n{:?}",
                tracing.stats
            );
            (profile, interpreter, tracing)
        })
        .collect()
}

/// Lua source that prints a table's contents in a fixed order.
const DUMP: &str = "
local function dump(t)
  local keys = {}
  for k in pairs(t) do keys[#keys + 1] = k end
  table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)
  local out = {}
  for _, k in ipairs(keys) do out[#out + 1] = tostring(k) .. '=' .. tostring(t[k]) end
  return table.concat(out, ',')
end
";

/// Each constructor runs once: real tables plus net phantom charges are the
/// interpreter's allocations.
fn assert_each_constructor_ran_once(
    profile: CompatibilityProfile,
    interpreter: &Outcome,
    tracing: &Outcome,
) {
    // Lua 5.1's scalar-table fragment elides in the interpreter too.
    assert_eq!(
        tracing.created + tracing.elided.0,
        interpreter.created + interpreter.elided.0,
        "{profile:?}: real {} + phantom {} against the interpreter's {} + {}\n{:?}",
        tracing.created,
        tracing.elided.0,
        interpreter.created,
        interpreter.elided.0,
        tracing.stats
    );
}

fn assert_traced(profile: CompatibilityProfile, tracing: &Outcome, least: u64) {
    assert!(
        tracing.stats.iterations >= least,
        "{profile:?}: {} native iterations, wanted {least}\n{:?}",
        tracing.stats.iterations,
        tracing.stats
    );
    assert!(tracing.stats.traces_compiled >= 1, "{profile:?}");
}

/// The tables a loop of `n` iterations allocates for real, and the
/// constructors it elides, with the loop body `body` and `n` read from the
/// `arg` slot of the source.
fn counts(profile: CompatibilityProfile, program: &str, n: usize) -> (u64, u64) {
    let source = program.replace("COUNT", &n.to_string());
    let tracing = run(profile, ExecutionMode::Tracing, &source);
    (tracing.created, tracing.elided.0)
}

/// The warmed steady state allocates no table: tripling the iterations adds
/// no real table, and adds the constructors as phantom charges. The checksum
/// matches the interpreter's.
fn assert_steady_state_allocates_nothing(program: &str) {
    let small = 20_000;
    let large = 60_000;
    for &profile in &ALL_PROFILES {
        let source = program.replace("COUNT", &large.to_string());
        let interpreter = run(profile, ExecutionMode::Interpreter, &source);
        let tracing = run(profile, ExecutionMode::Tracing, &source);
        assert_eq!(tracing.result, interpreter.result, "{profile:?}\n{program}");
        let (created_small, elided_small) = counts(profile, program, small);
        assert!(
            tracing.created <= created_small + 4,
            "{profile:?}: {created_small} real tables at {small} iterations, {} at {large}",
            tracing.created
        );
        assert!(
            tracing.elided.0 >= elided_small + (large - small) as u64 - 16,
            "{profile:?}: phantom charges {elided_small} at {small}, {} at {large}",
            tracing.elided.0
        );
        assert_traced(profile, &tracing, (large - 1_000) as u64);
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
    }
}

#[test]
fn table_churn_allocates_no_table_in_steady_state() {
    assert_steady_state_allocates_nothing(
        "local total = 0
         for i = 1, COUNT do
           local value = {i, i + 1, i + 2, label = 'node'}
           total = total + value[2]
         end
         return total",
    );
}

#[test]
fn array_constructors_allocate_no_table_in_steady_state() {
    assert_steady_state_allocates_nothing(
        "local total = 0
         for i = 1, COUNT do
           local v = {i, i + 1, i + 2}
           total = total + v[2] + #v
         end
         return total",
    );
}

#[test]
fn field_constructors_allocate_no_table_in_steady_state() {
    assert_steady_state_allocates_nothing(
        "local total = 0
         for i = 1, COUNT do
           local v = {x = i, y = i + 1, z = i % 7}
           total = total + v.x + v.z
         end
         return total",
    );
}

#[test]
fn mixed_constructors_allocate_no_table_in_steady_state() {
    assert_steady_state_allocates_nothing(
        "local total = 0
         for i = 1, COUNT do
           local v = {i, i + 1, tag = i % 7, n = 2}
           total = total + v[1] + v.tag + v.n
         end
         return total",
    );
}

#[test]
fn a_rare_branch_that_keeps_the_table_builds_it_for_its_side() {
    let source = format!(
        "{DUMP}
         local kept, count = {{}}, 0
         local total = 0
         for i = 1, {N} do
           local v = {{i, i + 1, tag = i % 5}}
           if i % 50 == 0 then
             count = count + 1
             kept[count] = v
           end
           total = total + v[1] + v.tag
         end
         local kept_sum = 0
         for j = 1, count do kept_sum = kept_sum + kept[j][2] + kept[j].tag end
         return total, count, kept_sum, dump(kept[1]), dump(kept[count])"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_traced(profile, &tracing, 5_000);
        // The rare branch became a side of the tree: the run builds the
        // table on the way into it, and the loop stays in native code.
        assert!(
            tracing.stats.side_installs >= 1,
            "{profile:?}: the rare branch got no side\n{:?}",
            tracing.stats
        );
        assert!(
            tracing.stats.exits_of(TraceExitKind::GuardFail) < (N / 50 / 2) as u64,
            "{profile:?}: the kept tables still leave the trace\n{:?}",
            tracing.stats
        );
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
        // The kept tables are real; the others stayed virtual.
        assert!(
            tracing.created + (N as u64) / 2 < interpreter.created,
            "{profile:?}: {} real tables of the interpreter's {}",
            tracing.created,
            interpreter.created
        );
    }
}

#[test]
fn two_locals_naming_one_virtual_table_restore_as_one_table() {
    let source = format!(
        "{DUMP}
         local kept, check, total = {{}}, 0, 0
         for i = 1, {N} do
           local v = {{i, i + 1, tag = i}}
           local w = v
           w.tag = w.tag + 100
           if i % 40 == 0 then
             -- Runs in the interpreter, on the table the exit built.
             w.extra = 7
             v[3] = 9
             kept[#kept + 1] = w
             check = check + (v.extra or -1000000) + w[3] + v.tag
           end
           total = total + w[1] + v.tag
         end
         return total, check, #kept, dump(kept[1]), dump(kept[#kept])"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_traced(profile, &tracing, 5_000);
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
        assert!(
            tracing.stats.exits_of(TraceExitKind::GuardFail) > 0,
            "{profile:?}\n{:?}",
            tracing.stats
        );
    }
}

#[test]
fn numeric_keys_normalize_and_duplicate_keys_overwrite() {
    // Integer and float spellings of one key, a fractional key, a key past
    // the array, a duplicate field, and a positional item that a later
    // explicit key replaces. The table is kept every 25th iteration.
    let source = format!(
        "{DUMP}
         local kept, total = {{}}, 0
         for i = 1, {N} do
           local v = {{[1] = i, [2.0] = i + 1, [3] = i + 2, [0.5] = i, [10] = i + 3,
                      x = i, x = i + 4, i * 2, [2] = i + 5}}
           total = total + v[1] + v[2.0] + v[2] + v[3] + v[0.5] + v[10] + v.x
           if i % 25 == 0 then kept[#kept + 1] = v end
         end
         return total, #kept, dump(kept[1]), dump(kept[#kept])"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_traced(profile, &tracing, 5_000);
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
    }
}

#[test]
fn nil_values_leave_no_entry_and_nil_stores_delete() {
    let source = format!(
        "{DUMP}
         local kept, total = {{}}, 0
         for i = 1, {N} do
           local v = {{x = i, y = i + 1, z = nil}}
           local w = {{i, nil, i + 2}}
           v.x = nil
           total = total + (v.x or 100) + v.y + (v.z or 1000) + w[1] + (w[2] or 7) + w[3]
           if i % 30 == 0 then kept[#kept + 1] = v; kept[#kept + 1] = w end
         end
         return total, #kept, dump(kept[1]), dump(kept[2]), dump(kept[#kept])"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_traced(profile, &tracing, 5_000);
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
    }
}

#[test]
fn the_length_of_a_virtual_array_is_traced_when_its_border_is_unique() {
    let source = format!(
        "{DUMP}
         local kept, total = {{}}, 0
         for i = 1, {N} do
           local v = {{i, i + 1, i + 2}}
           local n = #v
           v[4] = i
           v[2] = nil -- would leave a hole: the interpreter's border rules apply
           total = total + n + #v
           if i % 100 == 0 then kept[#kept + 1] = v end
         end
         return total, #kept, dump(kept[1])"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
        // The hole rejects the recording; the interpreter's answer stands.
        assert_eq!(tracing.elided.0, 0, "{profile:?}");
    }
    // Without the hole the whole thing traces.
    let source = format!(
        "{DUMP}
         local kept, total = {{}}, 0
         for i = 1, {N} do
           local v = {{i, i + 1, i + 2}}
           local n = #v
           v[4] = i
           v[4] = nil
           v[3] = nil
           total = total + n + #v
           if i % 100 == 0 then kept[#kept + 1] = v end
         end
         return total, #kept, dump(kept[1]), #kept[1]"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_traced(profile, &tracing, 5_000);
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
    }
}

#[test]
fn a_dead_register_holding_last_iterations_table_is_not_a_carry() {
    // The register of `v` still holds the last iteration's table when the
    // next one begins, and that iteration writes it before reading it.
    let source = format!(
        "local total = 0
         for i = 1, {N} do
           local v = {{i, i + 1}}
           local w = v
           total = total + w[1] + v[2]
         end
         return total"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_traced(profile, &tracing, 5_000);
        assert!(
            tracing.elided.0 >= 5_000,
            "{profile:?}\n{:?}",
            tracing.stats
        );
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
    }
}

#[test]
fn a_local_declared_outside_the_loop_is_a_carry() {
    // `v` stays in scope after the loop, so what it holds at the loop-back is
    // observable and the table would have to exist.
    let source = format!(
        "local v
         local total = 0
         for i = 1, {N} do
           v = {{i, i + 1}}
           total = total + v[1] + v[2]
         end
         return total, v[1]"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_eq!(tracing.created, interpreter.created, "{profile:?}");
        assert_eq!(tracing.stats.traces_compiled, 0, "{profile:?}");
    }
}

/// Escapes the recorder rejects before it looks at the table: the operation
/// itself is one it does not record (a native call with a table argument, a
/// traversal), so the abort is an unsupported operand or opcode.
fn strict_reason_unknown(name: &str) -> bool {
    matches!(
        name,
        "call of a native function"
            | "pairs"
            | "ipairs"
            | "next"
            | "setmetatable"
            | "getmetatable"
            | "rawequal"
    )
}

/// A loop whose table escapes in the way `body` says. The recording must be
/// rejected (nothing elided) and the result must be the interpreter's.
fn assert_escape_rejected(name: &str, setup: &str, body: &str, reason: TraceAbortReason) {
    let source = format!(
        "{DUMP}
         {setup}
         local total = 0
         for i = 1, {N} do
           local v = {{i, i + 1, tag = i}}
           {body}
         end
         return total"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_eq!(
            tracing.elided.0, 0,
            "{name} {profile:?}: no constructor may be elided\n{:?}",
            tracing.stats
        );
        assert_eq!(tracing.created, interpreter.created, "{name} {profile:?}");
        assert!(
            tracing.stats.total_aborts() > 0,
            "{name} {profile:?}: the recording must be rejected\n{:?}",
            tracing.stats
        );
        if reason == TraceAbortReason::VirtualEscape && !strict_reason_unknown(name) {
            assert!(
                tracing.stats.aborts(reason) > 0,
                "{name} {profile:?}: wanted an abort for {reason:?}\n{:?}",
                tracing.stats
            );
        }
    }
}

/// A loop whose table escapes in the way `body` says, and that the trace
/// follows: the table is made by the trace where it escapes, every profile
/// agrees with the interpreter, and no recording is rejected for the escape.
fn assert_escape_made(name: &str, setup: &str, body: &str) {
    let source = format!(
        "{DUMP}
         {setup}
         local total = 0
         for i = 1, {N} do
           local v = {{i, i + 1, tag = i}}
           {body}
         end
         return total"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_eq!(
            tracing.stats.total_aborts() + tracing.stats.side_aborts,
            0,
            "{name} {profile:?}: no recording is abandoned\n{:?}",
            tracing.stats
        );
        assert_traced(profile, &tracing, (N - 1_000) as u64);
        assert!(
            tracing.stats.table_new_helper_calls >= (N - 1_000) as u64,
            "{name} {profile:?}: the trace makes the table every iteration\n{:?}",
            tracing.stats
        );
        // No phantom charge stands for a table that is made for real.
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
    }
}

#[test]
fn a_store_into_an_object_makes_the_table_in_the_trace() {
    assert_escape_made(
        "store into a table",
        "local box = {}",
        "box[1] = v; total = total + v[1]",
    );
    assert_escape_made(
        "store into a field",
        "local box = {}",
        "box.slot = v; total = total + v[1]",
    );
}

#[test]
fn a_table_stored_into_another_virtual_table_stays_virtual() {
    // Neither table is made: the inner one is an item of the outer one, and
    // reading it back names it again.
    for body in [
        "local w = {}; w[1] = v; total = total + w[1][1]",
        "local w = {inner = v}; total = total + w.inner[1]",
        "local w = {v, {v[2]}}; total = total + w[1][1] + w[2][1]",
    ] {
        let source = format!(
            "{DUMP}
             local total = 0
             for i = 1, {N} do
               local v = {{i, i + 1, tag = i}}
               {body}
             end
             return total"
        );
        for (profile, interpreter, tracing) in agree(&source) {
            assert_traced(profile, &tracing, (N - 1_000) as u64);
            assert_eq!(
                tracing.stats.table_new_helper_calls, 0,
                "{body} {profile:?}: no table is made\n{:?}",
                tracing.stats
            );
            assert!(
                tracing.elided.0 >= 2 * (N as u64 - 1_000),
                "{body} {profile:?}: both constructors are charged\n{:?}",
                tracing.stats
            );
            assert_each_constructor_ran_once(profile, &interpreter, &tracing);
        }
    }
}

#[test]
fn a_nested_constructor_allocates_nothing_in_steady_state() {
    assert_steady_state_allocates_nothing(
        "local total = 0
         for i = 1, COUNT do
           local b1, b2 = i % 3, i % 5
           local t = {b1, b2, {b1 + b2}, tag = {b2}}
           total = total + t[3][1] + t[1] * t[2] + t.tag[1]
         end
         return total",
    );
}

#[test]
fn a_call_argument_rejects_the_recording() {
    assert_escape_rejected(
        "call of a Lua function",
        "local function first(t) return t[1] end",
        "total = total + first(v)",
        // The table is made for the call, but the callee reads a number key
        // of a table met by lane, which no trace does natively.
        TraceAbortReason::UnsupportedOperand,
    );
    assert_escape_made(
        "call of a native function",
        "",
        "total = total + select('#', v) + (type(v) == 'table' and 1 or 0)",
    );
}

#[test]
fn a_return_rejects_the_recording() {
    // The constructor sits in a callee, and the callee returns it.
    let source = format!(
        "local function make(i) return {{i, i + 1}} end
         local total = 0
         for i = 1, {N} do
           local v = make(i)
           total = total + v[1] + v[2]
         end
         return total"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_eq!(tracing.elided.0, 0, "{profile:?}");
        assert_eq!(tracing.created, interpreter.created, "{profile:?}");
    }
}

#[test]
fn a_table_passed_to_an_inlined_function_is_made_in_the_trace() {
    // The callee keeps the table in a ring, so the table must be real.
    assert_escape_made(
        "call that stores its argument",
        "local ring = {}; local function put(t, i) ring[i % 64 + 1] = t end",
        "put(v, i); total = total + 1",
    );
    assert_escape_made(
        "call that ignores its argument",
        "local function skip(t) return 1 end",
        "total = total + skip(v)",
    );
}

#[test]
fn a_table_returned_from_an_inlined_constructor_is_made_in_the_trace() {
    let source = format!(
        "local ring = {{}}
         local function make(i) return {{v = i, w = i + 1}} end
         local total = 0
         for i = 1, {N} do
           local keep = make(i)
           ring[i % 64 + 1] = keep
           total = total + i % 3
         end
         return total, ring[5].w"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_eq!(
            tracing.stats.aborts(TraceAbortReason::VirtualEscape),
            0,
            "{profile:?}\n{:?}",
            tracing.stats
        );
        assert!(tracing.stats.traces_recorded >= 1, "{profile:?}");
        assert_traced(profile, &tracing, (N - 1_000) as u64);
        assert!(
            tracing.stats.table_new_helper_calls >= (N - 1_000) as u64,
            "{profile:?}: the trace makes the table\n{:?}",
            tracing.stats
        );
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
    }
}

#[test]
fn a_returned_table_that_a_local_outside_the_loop_holds_rejects_the_recording() {
    // `keep` is a named local in scope at the header, so the lane it holds at
    // the loop-back is observable: the next iteration's snapshots restore it
    // (debug.getlocal sees it too), and the loop-back drops made lanes.
    let source = format!(
        "local function make(i) return {{v = i, w = i + 1}} end
         local total, keep = 0
         for i = 1, {N} do
           keep = make(i)
           total = total + i % 3
         end
         return total, keep.v, keep.w"
    );
    for (profile, _, tracing) in agree(&source) {
        assert!(
            tracing.stats.aborts(TraceAbortReason::VirtualEscape) > 0,
            "{profile:?}\n{:?}",
            tracing.stats
        );
        assert_eq!(tracing.stats.traces_compiled, 0, "{profile:?}");
    }
}

/// The loop of the report: a table `make` returns is held by a local that
/// lives outside the loop, and `probe` reads that local while the loop runs.
fn carried_source(probe: &str, extra: &str) -> String {
    format!(
        "local ring, seen = {{}}, {{}}
         local function make(i) local t = {{i, i}}; return t end
         {extra}
         local keep
         for i = 1, 300000 do
           if i % 50000 == 0 then
             {probe}
           end
           keep = make(i)
           ring[i % 8 + 1] = keep
         end
         return table.concat(seen, ' ')"
    )
}

fn assert_carry_refused(source: &str) {
    for (profile, interpreter, tracing) in agree(source) {
        assert!(
            interpreter
                .result
                .contains("49999 99999 149999 199999 249999 299999"),
            "{profile:?}: {}",
            interpreter.result
        );
        assert!(
            tracing.stats.aborts(TraceAbortReason::VirtualEscape) > 0,
            "{profile:?}: the carried table must refuse the recording\n{:?}",
            tracing.stats
        );
        assert_eq!(
            tracing.stats.traces_compiled, 0,
            "{profile:?}: the refused recording must compile nothing\n{:?}",
            tracing.stats
        );
    }
}

#[test]
fn debug_getlocal_sees_the_table_a_carried_local_holds() {
    let source = carried_source(
        "local _, v = debug.getlocal(1, 4) seen[#seen+1] = tostring(v and v[1])",
        "",
    );
    assert_carry_refused(&source);
}

#[test]
fn debug_getlocal_in_a_called_function_sees_the_carried_table() {
    let source = carried_source(
        "seen[#seen+1] = look()",
        "local function look() local _, v = debug.getlocal(2, 5) return tostring(v and v[1]) end",
    );
    assert_carry_refused(&source);
}

#[test]
fn a_collection_keeps_the_table_a_carried_local_holds() {
    // This pins the refusal now. If carried made-table lanes are ever kept
    // rooted, it becomes a liveness check.
    // `make` stays free of stores into a table with a metatable, so the
    // recording reaches the loop-back. The probe adds the carried table to the
    // weak-keyed set just before the collection, so it must survive it.
    let source = "local ring, weak = {}, setmetatable({}, {__mode = 'k'})
         local function make(i) local t = {i, i}; return t end
         local keep
         local bad = 0
         for i = 1, 200000 do
           if i % 25000 == 0 then
             weak[keep] = true
             collectgarbage()
             local n = 0
             for _ in pairs(weak) do n = n + 1 end
             if n ~= 1 or keep[1] ~= i - 1 then bad = bad + 1 end
             weak[keep] = nil
           end
           keep = make(i)
           ring[i % 8 + 1] = keep
         end
         collectgarbage()
         return bad, keep[1]";
    for (profile, interpreter, tracing) in agree(source) {
        assert!(
            tracing.stats.aborts(TraceAbortReason::VirtualEscape) > 0,
            "{profile:?}"
        );
        assert_eq!(tracing.stats.traces_compiled, 0, "{profile:?}");
        assert_eq!(tracing.created, interpreter.created, "{profile:?}");
    }
}

#[test]
fn an_upvalue_or_global_store_rejects_the_recording() {
    // The table is made, and the cell then takes a value the trace does not
    // write: the recording is rejected all the same.
    assert_escape_rejected(
        "upvalue store",
        "local held; local function look() return held end",
        "held = v; total = total + look()[1]",
        TraceAbortReason::UnsupportedOperand,
    );
    let source = format!(
        "local total = 0
         for i = 1, {N} do
           local v = {{i, i + 1}}
           GLOBAL_HOLD = v
           total = total + GLOBAL_HOLD[2]
         end
         return total"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_eq!(tracing.elided.0, 0, "{profile:?}");
        assert_eq!(tracing.created, interpreter.created, "{profile:?}");
    }
}

#[test]
fn identity_and_raw_operations_reject_the_recording() {
    assert_escape_rejected(
        "equality",
        "local other = {}",
        "if v == other then total = total + 1000 end; total = total + v[1]",
        TraceAbortReason::VirtualEscape,
    );
    assert_escape_rejected(
        "inequality",
        "local other = {}",
        "if v ~= other then total = total + 1 end",
        TraceAbortReason::VirtualEscape,
    );
    // A native called with the table gets the table the trace made.
    assert_escape_made(
        "rawequal",
        "local other = {}",
        "if rawequal(v, other) then total = total + 1000 end; total = total + 1",
    );
    // The table is made, and a table key is a store the trace does not make.
    assert_escape_rejected(
        "use as a key",
        "local seen = {}",
        "seen[v] = i; total = total + seen[v]",
        TraceAbortReason::UnsupportedOperand,
    );
}

#[test]
fn metatables_and_traversal_reject_the_recording() {
    assert_escape_rejected(
        "setmetatable",
        "local mt = {__index = function() return 5 end}",
        "setmetatable(v, mt); total = total + v.missing + v[1]",
        TraceAbortReason::VirtualEscape,
    );
    assert_escape_made(
        "getmetatable",
        "",
        "if getmetatable(v) == nil then total = total + 1 end",
    );
    assert_escape_rejected(
        "pairs",
        "",
        "for k, x in pairs(v) do total = total + x end",
        TraceAbortReason::VirtualEscape,
    );
    assert_escape_rejected(
        "ipairs",
        "",
        "for _, x in ipairs(v) do total = total + x end",
        TraceAbortReason::VirtualEscape,
    );
    assert_escape_made("next", "", "local k, x = next(v); total = total + x");
}

#[test]
fn a_truth_test_makes_the_table_and_a_computed_key_rejects_the_recording() {
    assert_escape_made("truth test", "", "if v then total = total + 1 end");
    // The key could name another slot in the next iteration.
    assert_escape_rejected(
        "computed key",
        "",
        "total = total + v[i % 2 + 1]",
        TraceAbortReason::VirtualEscape,
    );
    assert_escape_rejected(
        "computed string key",
        "local names = {'tag', 'tag'}",
        "total = total + v[names[1 + i % 2]]",
        TraceAbortReason::VirtualEscape,
    );
}

#[test]
fn a_table_live_across_the_loop_back_rejects_the_recording() {
    let source = format!(
        "local prev
         local total = 0
         for i = 1, {N} do
           local v = {{i, i + 1}}
           if prev then total = total + prev[1] end
           prev = v
         end
         return total"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_eq!(tracing.elided.0, 0, "{profile:?}\n{:?}", tracing.stats);
        assert_eq!(tracing.created, interpreter.created, "{profile:?}");
        assert!(
            tracing.stats.total_aborts() > 0,
            "{profile:?}\n{:?}",
            tracing.stats
        );
    }
}

#[test]
fn a_captured_local_holding_a_table_rejects_the_recording() {
    // A closure reads `v`'s cell, so the table is reachable from outside.
    let source = format!(
        "local v
         local function peek() return v end
         local total = 0
         for i = 1, {N} do
           v = {{i, i + 1}}
           total = total + peek()[2]
         end
         return total"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_eq!(tracing.elided.0, 0, "{profile:?}\n{:?}", tracing.stats);
        assert_eq!(tracing.created, interpreter.created, "{profile:?}");
    }
}

#[test]
fn a_virtual_table_stays_through_an_inlined_call_that_leaves() {
    // The table is live in the caller while a callee runs, and the callee's
    // guard fails now and then, so the exit rebuilds the callee's frame and
    // builds the caller's table.
    let source = format!(
        "{DUMP}
         local function step(x) if x % 61 == 0 then return x + 0.5 end return x + 1 end
         local kept, total = {{}}, 0
         for i = 1, {N} do
           local v = {{i, tag = i}}
           local y = step(i)
           total = total + v[1] + v.tag + y
           if i % 500 == 0 then kept[#kept + 1] = v end
         end
         return total, #kept, dump(kept[1])"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_traced(profile, &tracing, 3_000);
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
    }
}

#[test]
fn a_trace_that_polls_while_a_dead_table_sits_in_a_register_resumes_cleanly() {
    // A long loop polls many times. The register of the last iteration's
    // table is dead at every poll, so the poll restores nil for it.
    let n = 400_000;
    let source = format!(
        "local total = 0
         for i = 1, {n} do
           local v = {{i, i + 1, k = i}}
           total = total + v[1] + v.k
         end
         return total"
    );
    for &profile in &ALL_PROFILES {
        let interpreter = run(profile, ExecutionMode::Interpreter, &source);
        let tracing = run(profile, ExecutionMode::Tracing, &source);
        assert_eq!(tracing.result, interpreter.result, "{profile:?}");
        assert_traced(profile, &tracing, 300_000);
        let left = tracing.stats.exits_of(TraceExitKind::Poll)
            + tracing.stats.exits_of(TraceExitKind::LoopComplete);
        assert!(
            left >= 1 || tracing.stats.iterations == 0,
            "{profile:?}\n{:?}",
            tracing.stats
        );
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
    }
}

#[test]
fn luajit_64_bit_integers_ride_in_a_virtual_table() {
    let source = format!(
        "{DUMP}
         local kept = {{}}
         local total = 0LL
         local mask = 0ULL
         for i = 1, {N} do
           local v = {{i * 1LL, 7ULL + i, plain = i}}
           total = total + v[1]
           mask = mask + v[2]
           if i % 100 == 0 then kept[#kept + 1] = v end
         end
         return tostring(total), tostring(mask), #kept, dump(kept[1]), dump(kept[#kept])"
    );
    let profile = CompatibilityProfile::LuaJit21;
    let interpreter = run(profile, ExecutionMode::Interpreter, &source);
    let tracing = run(profile, ExecutionMode::Tracing, &source);
    assert_eq!(tracing.result, interpreter.result, "{:?}", tracing.stats);
    assert_traced(profile, &tracing, 5_000);
    assert!(tracing.elided.0 > 0, "{:?}", tracing.stats);
    assert_each_constructor_ran_once(profile, &interpreter, &tracing);
}

#[test]
fn phantom_debt_grows_with_the_elided_constructors() {
    // With the collector stopped, the heap count rises by the constructors'
    // footprints whether the tables exist or not.
    let source = format!(
        "collectgarbage('stop')
         local before = collectgarbage('count')
         local total = 0
         for i = 1, {N} do
           local v = {{i, i + 1, i + 2, tag = i}}
           total = total + v[1] + v.tag
         end
         local after = collectgarbage('count')
         return total, after - before"
    );
    for &profile in &ALL_PROFILES {
        let source = source.clone();
        let mut interpreter = Lua::builder()
            .profile(profile)
            .execution_mode(ExecutionMode::Interpreter)
            .build()
            .unwrap();
        let grown_interpreter = grown(&mut interpreter, &source);
        let mut tracing = Lua::builder()
            .profile(profile)
            .execution_mode(ExecutionMode::Tracing)
            .build()
            .unwrap();
        tracing.vm_mut().use_synchronous_jit_compilation();
        tracing.vm_mut().ignore_remaining_work();
        let grown_tracing = grown(&mut tracing, &source);
        let stats = tracing.vm().jit_statistics().trace;
        assert!(stats.iterations > 5_000, "{profile:?}\n{stats:?}");
        // A phantom charge is the footprint of the table the interpreter
        // would have made, a little less since the interpreter's table also
        // has a handle and an arena slot.
        let charged = (N as u64 * Runtime::elided_footprint(4) as u64) as f64 / 1024.0;
        assert!(
            grown_tracing >= charged * 0.9,
            "{profile:?}: {grown_tracing} KB against {charged}"
        );
        let ratio = grown_tracing / grown_interpreter;
        assert!(
            (0.5..=1.3).contains(&ratio),
            "{profile:?}: heap grew by {grown_tracing} KB traced, {grown_interpreter} KB interpreted"
        );
    }
}

fn grown(lua: &mut Lua, source: &str) -> f64 {
    let results: Vec<Value> = lua.exec(source).unwrap();
    match results.last() {
        Some(Value::Number(kilobytes)) => *kilobytes,
        other => panic!("heap growth is a number, got {other:?}"),
    }
}

#[test]
fn a_materialized_table_is_charged_once() {
    // Every kept table is charged as phantom debt when its constructor runs
    // and taken back when the exit builds it. What stays is the footprint of
    // the constructors that never existed.
    let source = format!(
        "local kept, count = {{}}, 0
         local total = 0
         for i = 1, {N} do
           local v = {{i, i + 1, tag = i % 5}}
           if i % 50 == 0 then count = count + 1; kept[count] = v end
           total = total + v[1] + v.tag
         end
         return total, count"
    );
    for &profile in &ALL_PROFILES {
        let tracing = run(profile, ExecutionMode::Tracing, &source);
        let interpreter = run(profile, ExecutionMode::Interpreter, &source);
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
        // Three entries each: two array slots and a field.
        let per_table = Runtime::elided_footprint(3) as u64;
        assert_eq!(
            tracing.elided.1,
            tracing.elided.0 * per_table,
            "{profile:?}: the bytes charged are the elided tables' footprints"
        );
    }
}

/// A function whose loop reaches `op` on a virtual table only in its last
/// iteration, so the op raises an error the caller catches. Called many
/// times, the loop traces without the op and the exit that reaches it records
/// the op on the virtual table. The error text and the checksum must match
/// the interpreter's.
fn assert_rare_op_rejected(name: &str, op: &str, reason: TraceAbortReason) {
    let source = format!(
        "local function run(n)
           local total = 0
           for i = 1, n do
             local v = {{i, i + 1, tag = i}}
             if i == n then total = total + {op} end
             total = total + v[1]
           end
           return total
         end
         local sum, errors = 0, ''
         for r = 1, 200 do
           local ok, e = pcall(run, 300)
           if ok then sum = sum + e else errors = e end
         end
         return sum, errors"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
        // The op sits on a rare branch, so the exit to it either records a
        // side trace that aborts on the op, or the side trace is refused
        // because the exit needs a virtual table.
        let stats = &tracing.stats;
        assert!(
            stats.aborts(reason) + stats.side_aborts + stats.side_rejections > 0,
            "{name} {profile:?}: wanted a rejection ({reason:?})\n{stats:?}"
        );
        assert!(
            stats.exits_of(TraceExitKind::GuardFail) > 0,
            "{name} {profile:?}"
        );
    }
}

#[test]
fn metamethod_capable_ops_on_the_table_reject_the_recording() {
    assert_rare_op_rejected("concat", "#(v .. 'x')", TraceAbortReason::VirtualEscape);
    assert_rare_op_rejected("arithmetic", "(v + 1)", TraceAbortReason::VirtualEscape);
    assert_rare_op_rejected("negation", "(-v)", TraceAbortReason::VirtualEscape);
    assert_rare_op_rejected(
        "less than",
        "(v < v and 1 or 0)",
        TraceAbortReason::VirtualEscape,
    );
    assert_rare_op_rejected(
        "less or equal",
        "(v <= 3 and 1 or 0)",
        TraceAbortReason::VirtualEscape,
    );
}

#[test]
fn a_constructor_the_recorder_does_not_understand_is_not_virtual() {
    // Open call results and varargs fill the table through opcodes the
    // recorder does not know.
    let sources = [
        "local function two(i) return i, i + 1 end
         local total = 0
         for i = 1, COUNT do
           local v = {i, two(i)}
           total = total + v[2] + #v
         end
         return total",
        "local function f(...)
           local total = 0
           for i = 1, COUNT do
             local v = {...}
             total = total + #v + i
           end
           return total
         end
         return f(1, 2, 3)",
    ];
    for program in sources {
        let source = program.replace("COUNT", &N.to_string());
        for (profile, interpreter, tracing) in agree(&source) {
            assert_each_constructor_ran_once(profile, &interpreter, &tracing);
            assert_eq!(
                tracing.elided.0, 0,
                "{profile:?}: nothing may be elided\n{program}\n{:?}",
                tracing.stats
            );
            assert_eq!(tracing.created, interpreter.created, "{profile:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// A table that escapes is made by the trace.
//
// The programs below keep the table in a ring (or another holder) so that it
// outlives the iteration, and read it the ways a program can: through the
// register that built it, through the holder, and by identity. Every result is
// the interpreter's, and the counters say the trace made the tables.

/// Each constructor ran at least once, and a table the trace made that an
/// exit then abandoned (the exit builds the iteration's tables again, from
/// their recipes, for the interpreter) is the only way one ran twice: at most
/// `per_exit` tables for each exit that left the trace.
fn assert_constructors_ran_once_or_were_abandoned(
    profile: CompatibilityProfile,
    interpreter: &Outcome,
    tracing: &Outcome,
    per_exit: u64,
) {
    let baseline = interpreter.created + interpreter.elided.0;
    let traced = tracing.created + tracing.elided.0;
    let exits = tracing.stats.exits.iter().sum::<u64>();
    assert!(
        traced >= baseline && traced <= baseline + per_exit * exits,
        "{profile:?}: {traced} constructors ran in the trace against the interpreter's {baseline}, with {exits} exits\n{:?}",
        tracing.stats
    );
}

/// The checks every program of this section shares: the results agree (done by
/// `agree`), no escape rejected the recording, the loop ran natively and made
/// a table for about every iteration, and each constructor ran once.
fn assert_made_in_the_trace(source: &str, least_tables: u64) {
    for (profile, interpreter, tracing) in agree(source) {
        assert_eq!(
            tracing.stats.aborts(TraceAbortReason::VirtualEscape),
            0,
            "{profile:?}\n{source}\n{:?}",
            tracing.stats
        );
        assert_traced(profile, &tracing, (N - 1_000) as u64);
        assert!(
            tracing.stats.table_new_helper_calls >= least_tables,
            "{profile:?}: {} tables made, wanted {least_tables}\n{source}",
            tracing.stats.table_new_helper_calls
        );
        assert_constructors_ran_once_or_were_abandoned(profile, &interpreter, &tracing, 4);
    }
}

#[test]
fn a_fresh_table_stored_into_a_ring_runs_natively_and_leaves_no_escape_abort() {
    let source = format!(
        "local ring = {{}}
         for i = 1, 8 do ring[i] = {{0}} end
         local total = 0
         for i = 1, {N} do
           local pair = {{i % 17, i % 23}}
           ring[i % 8 + 1] = pair
           total = (total + pair[1] + pair[2]) % 1000003
         end
         return total, #ring, ring[2][1], ring[2][2], ring[8][1]"
    );
    assert_made_in_the_trace(&source, (N - 1_000) as u64);
}

#[test]
fn a_table_stored_in_two_places_is_one_table() {
    // The same object lands in two holders, in two different slots each
    // time, and the holders agree with the register on what it holds.
    let source = format!(
        "local a, b = {{}}, {{}}
         for i = 1, 8 do a[i] = {{0}}; b[i] = {{0}} end
         local total = 0
         for i = 1, {N} do
           local v = {{i, x = i * 2}}
           a[i % 5 + 1] = v
           b[i % 3 + 1] = v
           total = total + v.x + v[1]
         end
         local shared = 0
         for i = 1, 5 do
           for j = 1, 3 do
             if rawequal(a[i], b[j]) then shared = shared + 1 end
           end
         end
         return total, shared, a[1].x, b[1].x, a[4][1]"
    );
    assert_made_in_the_trace(&source, (N - 1_000) as u64);
}

#[test]
fn identity_of_a_table_that_escaped_is_the_interpreters() {
    // Comparisons, `rawequal` and aliasing of a table the trace made, against
    // a table that exists before the loop and one stored by it.
    let source = format!(
        "local ring, other1, other2 = {{}}, {{1}}, {{2}}
         local total, same = 0, 0
         for i = 1, {N} do
           local v = {{i, i + 1, tag = i}}
           local alias = v
           ring[i % 7 + 1] = v
           if rawequal(alias, v) then same = same + 1 end
           if alias == v then same = same + 1 end
           if v ~= other1 then same = same + 1 end
           if v == other2 then same = same + 1000000 end
           total = total + v[1] + alias.tag
         end
         return total, same"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_traced(profile, &tracing, (N - 1_000) as u64);
        assert_constructors_ran_once_or_were_abandoned(profile, &interpreter, &tracing, 1);
    }
    // A table as a key is a store the trace does not make: the recording is
    // rejected and the interpreter's answer stands.
    let source = format!(
        "local seen, total = {{}}, 0
         for i = 1, {N} do
           local v = {{i, tag = i}}
           seen[v] = i
           total = total + seen[v] + v.tag
         end
         local count = 0
         for _ in pairs(seen) do count = count + 1 end
         return total, count"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_eq!(tracing.created, interpreter.created, "{profile:?}");
    }
}

#[test]
fn a_table_that_escapes_only_on_a_rare_branch_is_made_there() {
    // The table is made on the iterations that store it and built from its
    // recipe at the exit of the others, whichever side the trace was
    // recorded on.
    for rare in ["i % 97 == 0", "i % 97 ~= 0"] {
        let source = format!(
            "{DUMP}
             local ring, total = {{}}, 0
             for i = 1, {N} do
               local v = {{i, tag = i}}
               if {rare} then ring[i % 5 + 1] = v end
               total = total + v[1] + v.tag
             end
             return total, dump(ring[1] or {{}}), dump(ring[2] or {{}}), dump(ring[3] or {{}})"
        );
        for (profile, interpreter, tracing) in agree(&source) {
            assert_traced(profile, &tracing, (N / 2) as u64);
            assert_constructors_ran_once_or_were_abandoned(profile, &interpreter, &tracing, 1);
        }
    }
}

#[test]
fn a_write_after_the_escape_reaches_the_table_both_ways() {
    // A write through the register, and one through the holder, after the
    // table is stored. The trace models neither and the answer is the
    // interpreter's.
    for write in [
        "v[1] = i + 1; v.x = v.x + 1",
        "ring[i % 4 + 1].x = 5",
        "ring[i % 4 + 1].x = ring[i % 4 + 1].x + 1",
    ] {
        let source = format!(
            "local ring = {{}}
             for i = 1, 8 do ring[i] = {{x = 0}} end
             local total = 0
             for i = 1, {N} do
               local v = {{i, x = i}}
               ring[i % 4 + 1] = v
               {write}
               total = total + v[1] + v.x
             end
             return total, ring[1].x, ring[2][1], ring[3].x"
        );
        for (profile, _, tracing) in agree(&source) {
            // A rejection or a trace: either way the heap holds the same.
            let _ = (profile, &tracing);
        }
    }
    // A write to a table that existed before the table escaped cannot have
    // touched it: the trace reads the escaped table from its model and
    // writes the other through its slot.
    let source = format!(
        "local ring, seed = {{}}, {{1}}
         for i = 1, 8 do ring[i] = {{0}} end
         local total = 0
         for i = 1, {N} do
           local v = {{i, x = i}}
           ring[i % 4 + 1] = v
           seed[1] = seed[1] + 1
           total = total + v[1] + v.x + seed[1]
         end
         return total, seed[1], ring[1][1], ring[4].x"
    );
    assert_made_in_the_trace(&source, (N - 1_000) as u64);
}

#[test]
fn a_metatable_set_before_the_escape_is_on_the_table_that_escaped() {
    let source = format!(
        "local ring = {{}}
         for i = 1, 8 do ring[i] = {{0}} end
         local plain, handled = {{}}, {{__index = function() return -1 end}}
         local total = 0
         for i = 1, {N} do
           local v = {{i, x = i}}
           setmetatable(v, i % 2 == 0 and handled or plain)
           ring[i % 4 + 1] = v
           total = total + v[1] + (v.missing or 0)
         end
         return total, getmetatable(ring[1]) == handled, getmetatable(ring[2]) == handled"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_constructors_ran_once_or_were_abandoned(profile, &interpreter, &tracing, 1);
    }
}

#[test]
fn a_metatable_set_after_the_escape_reaches_the_real_table() {
    // The ring entries must carry the metatable once the loop is over, and
    // read `__index` through it: a recording that only models the table
    // would leave them bare.
    let source = format!(
        "local ring, mt = {{}}, {{__index = function() return 'z' end}}
         for i = 1, 16 do ring[i] = {{0}} end
         for i = 1, {N} do
           local t = {{i}}
           ring[i % 16 + 1] = t
           setmetatable(t, mt)
         end
         local bound = 0
         for i = 1, 16 do
           if getmetatable(ring[i]) == mt then bound = bound + 1 end
         end
         return bound, ring[1].zz, ring[7][1]"
    );
    for (profile, interpreter, _) in agree(&source) {
        assert!(
            interpreter.result.contains("16"),
            "{profile:?}: {}",
            interpreter.result
        );
    }
}

/// The peak of `collectgarbage('count')` over the loop of `source`, in
/// kilobytes (the script returns it first).
fn peak_of(profile: CompatibilityProfile, mode: ExecutionMode, source: &str) -> f64 {
    let outcome = run(profile, mode, source);
    let digits: String = outcome
        .result
        .chars()
        .skip_while(|c| *c != '(')
        .skip(1)
        .take_while(|c| *c != ')')
        .collect();
    digits
        .parse()
        .unwrap_or_else(|_| panic!("{profile:?}: peak in {}", outcome.result))
}

#[test]
fn tables_made_in_the_trace_keep_the_heap_near_the_interpreters() {
    // Debt from tables the trace makes must reach the collector within a
    // few steps, whatever the table's size: the peak stays within a small
    // factor of the interpreter's, with a wide table and a nest too.
    for (name, constructor) in [
        ("small", "{i, i + 1, x = i}"),
        (
            "wide",
            "{i, i + 1, a = 1, b = 2, c = 3, d = 4, e = 5, f = 6, g = 7, h = 8, k = 9}",
        ),
        ("nest", "{i, {i + 1, {i + 2, {i + 3}}}}"),
    ] {
        let source = format!(
            "local ring, total, peak = {{}}, 0, 0
             for i = 1, 256 do ring[i] = {{0}} end
             collectgarbage()
             local base = collectgarbage('count')
             for i = 1, 400000 do
               local v = {constructor}
               ring[i % 256 + 1] = v
               total = total + v[1]
               if i % 97 == 0 then
                 local grown = collectgarbage('count') - base
                 if grown > peak then peak = grown end
               end
             end
             return math.floor(peak), total"
        );
        for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
            let interpreter = peak_of(profile, ExecutionMode::Interpreter, &source);
            let tracing = peak_of(profile, ExecutionMode::Tracing, &source);
            assert!(
                tracing <= interpreter * 3.0 + 256.0,
                "{profile:?} {name}: peak {tracing} kB against the interpreter's {interpreter}"
            );
        }
    }
}

#[test]
fn a_ring_store_beside_a_field_store_to_another_table_runs_natively() {
    // The shape of opt/retained_slots without its closure call: an array
    // store into a ring that starts empty (the first keys land in its hash
    // part) and a field store into a second table. The set may move the
    // ring's hash part, which entry allows as long as the field's table is
    // another one.
    let source = format!(
        "local ring = {{}}
         local first, second = {{1}}, {{v = 2}}
         local total = 0
         for i = 1, {N} do
           local pair = {{i % 17, i % 23}}
           ring[i % 256 + 1] = pair
           total = (total + first[1] + second.v + pair[1] + pair[2]) % 1000003
           first[1] = (first[1] + 1) % 7
           second.v = (second.v + pair[2]) % 11
         end
         return total, #ring"
    );
    assert_made_in_the_trace(&source, (N - 1_000) as u64);
    for (_, _, tracing) in agree(&source) {
        assert!(tracing.stats.traces_recorded >= 1, "{:?}", tracing.stats);
    }
}

#[test]
fn a_ring_store_into_the_table_a_field_store_names_is_still_right() {
    // The ring and the field's table are one table in the second loop:
    // entry must refuse the trace recorded over two, not let the set move
    // storage the field address points into.
    let source = format!(
        "local ring, other = {{}}, {{v = 2}}
         local total = 0
         local function run(ring, other, n)
           for i = 1, n do
             ring[i % 64 + 1] = i
             total = total + other.v
             other.v = (other.v + 1) % 5
           end
         end
         run(ring, other, {N})
         run(other, other, {N})
         local count = 0
         for _ in pairs(other) do count = count + 1 end
         return total, count, other.v, #ring"
    );
    agree(&source);
}

#[test]
fn collector_steps_between_escaped_tables_leak_nothing() {
    // The peak allows for the nursery (1 MiB at least), which fills with
    // dead tables between minor collections; a leak of the 120000 tables
    // would pass it many times over.
    let n = 120_000;
    let source = format!(
        "local ring, total, peak = {{}}, 0, 0
         for i = 1, 64 do ring[i] = {{0}} end
         collectgarbage()
         local base = collectgarbage('count')
         for i = 1, {n} do
           local v = {{i, i + 1, tag = i}}
           ring[i % 64 + 1] = v
           total = total + v[1] + v.tag
           if i % 1000 == 0 then
             collectgarbage('step')
             local grown = collectgarbage('count') - base
             if grown > peak then peak = grown end
           end
         end
         collectgarbage()
         return total, peak < 4096, collectgarbage('count') - base < 2048, #ring"
    );
    for &profile in &ALL_PROFILES {
        let interpreter = run(profile, ExecutionMode::Interpreter, &source);
        let tracing = run(profile, ExecutionMode::Tracing, &source);
        assert_eq!(tracing.result, interpreter.result, "{profile:?}");
        assert!(
            tracing.result.contains("Boolean(true), Boolean(true)"),
            "{profile:?}: the heap stays bounded: {}",
            tracing.result
        );
        assert_traced(profile, &tracing, (n - 5_000) as u64);
        assert!(tracing.stats.table_new_helper_calls >= (n - 5_000) as u64);
    }
}

#[test]
fn a_weak_valued_ring_lets_the_made_tables_go() {
    let source = format!(
        "local ring = setmetatable({{}}, {{__mode = 'v'}})
         for i = 1, 16 do ring[i] = {{0}} end
         local strong, total = {{}}, 0
         for i = 1, {N} do
           local v = {{i, i + 1}}
           ring[i % 16 + 1] = v
           if i % 100 == 0 then strong[#strong + 1] = v end
           total = total + v[1]
         end
         collectgarbage()
         collectgarbage()
         local alive = 0
         for _, kept in pairs(ring) do alive = alive + 1 end
         return total, #strong, alive <= 16, strong[3][2]"
    );
    assert_made_in_the_trace(&source, (N - 1_000) as u64);
}

#[test]
fn a_nested_table_escapes_through_its_parent() {
    let source = format!(
        "local ring = {{}}
         for i = 1, 8 do ring[i] = {{0}} end
         local total = 0
         for i = 1, {N} do
           local v = {{i, {{i + 1, {{i + 2}}}}, inner = {{z = i}}}}
           ring[i % 8 + 1] = v
           total = total + v[2][1] + v[2][2][1] + v.inner.z
         end
         return total, ring[3][2][2][1], ring[4].inner.z, ring[5][2][1]"
    );
    // Four tables are made for each iteration's one store.
    assert_made_in_the_trace(&source, 4 * (N - 1_000) as u64);
}

#[test]
fn a_run_out_of_lanes_for_made_tables_leaves_before_the_store_and_loses_nothing() {
    // Six strings ride the loop-back, more than a release can keep, so the
    // run never drops the lanes of its tables: it leaves when the bank is
    // full, the interpreter makes the iteration's table itself, and the
    // trace is entered again.
    let source = format!(
        "local s1, s2, s3, s4, s5, s6 = 'a', 'bb', 'ccc', 'dddd', 'eeeee', 'ffffff'
         local ring = {{}}
         for i = 1, 8 do ring[i] = {{0}} end
         local total = 0
         for i = 1, {N} do
           local v = {{i, tag = i}}
           ring[i % 8 + 1] = v
           s1, s2, s3, s4, s5, s6 = s2, s3, s4, s5, s6, s1
           total = total + v[1] + #s1
         end
         return total, s1, s2, s3, s4, s5, s6, ring[1].tag, ring[2][1]"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_traced(profile, &tracing, (N / 2) as u64);
        assert!(
            tracing.stats.exits_of(TraceExitKind::GuardFail) >= 5,
            "{profile:?}: the full bank sends the run back\n{:?}",
            tracing.stats
        );
        assert_constructors_ran_once_or_were_abandoned(profile, &interpreter, &tracing, 1);
    }
}

#[test]
fn an_exit_after_the_table_is_made_restores_the_table_that_was_made() {
    // The key of the store is a string now and then, which the trace did not
    // record: it leaves at the guard on the key's type, after it made the
    // iteration's table. The register holds that table when the interpreter
    // goes on, and it is the one the program stores and compares.
    let source = format!(
        "local ring = {{}}
         for i = 1, 8 do ring[i] = {{0}} end
         local kept, total = {{}}, 0
         for i = 1, {N} do
           local v = {{i, tag = i}}
           local k = i % 211 == 0 and 'name' or i % 8 + 1
           ring[k] = v
           if i % 211 == 0 then kept[#kept + 1] = v end
           total = total + v[1] + v.tag
         end
         return total, #kept, kept[1][1], kept[#kept].tag, ring.name[1],
                rawequal(ring.name, kept[#kept])"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert!(
            tracing.result.contains("Boolean(true)"),
            "{profile:?}: {}",
            tracing.result
        );
        assert_traced(profile, &tracing, (N / 2) as u64);
        assert!(
            tracing.stats.exits_of(TraceExitKind::GuardFail) >= 1,
            "{profile:?}\n{:?}",
            tracing.stats
        );
        assert_constructors_ran_once_or_were_abandoned(profile, &interpreter, &tracing, 1);
    }
}

#[test]
fn the_dead_slots_benchmark_records_and_runs_natively_without_making_a_table() {
    // The benchmark's loop, shortened: a call whose body builds six tables,
    // one of them inside another, and none of them lives past its block.
    let source = include_str!("../../../benchmarks/lua/opt/dead_slots.lua")
        .replace("400000 * scale", "20000")
        .replace("print(total)", "return total");
    for (profile, interpreter, tracing) in agree(&source) {
        assert_eq!(
            tracing.stats.aborts(TraceAbortReason::VirtualEscape),
            0,
            "{profile:?}\n{:?}",
            tracing.stats
        );
        assert!(tracing.stats.traces_recorded >= 1, "{profile:?}");
        assert_traced(profile, &tracing, 19_000);
        assert_eq!(tracing.stats.table_new_helper_calls, 0, "{profile:?}");
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
        assert!(
            tracing.created + 1_000 < interpreter.created || interpreter.created < 1_000,
            "{profile:?}: {} real tables of the interpreter's {}",
            tracing.created,
            interpreter.created
        );
    }
}

#[test]
fn a_table_the_trace_cannot_make_is_made_by_the_interpreter_which_fails_as_it_would() {
    // The handle space runs out a few thousand tables into the loop. The
    // helper that makes a table in the trace sees the failure first and
    // misses: the interpreter then runs the constructor itself and raises
    // the error, with the holder as full as it had been left.
    let setup = "holder = {} collectgarbage() collectgarbage()";
    // One line: the interpreter names the line of the instruction before a
    // failing constructor, which the replay after a miss does not.
    let source = "local function fill(holder)
           for i = 1, 20000 do local t = {i, tag = i} holder[#holder + 1] = t end
         end
         local ok, err = pcall(fill, holder)
         local last = holder[#holder]
         return ok, tostring(err), #holder, last[1], last.tag, rawequal(last, holder[#holder])";
    for profile in ALL_PROFILES {
        let mut seen = Vec::new();
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            lua.vm_mut().use_synchronous_jit_compilation();
            lua.vm_mut().ignore_remaining_work();
            lua.exec(setup).unwrap();
            lua.vm_mut().runtime_mut().limit_table_growth(3_000);
            let results: Vec<Value> = lua.exec(source).unwrap();
            let stats = lua.vm().jit_statistics().trace;
            seen.push((format!("{results:?}"), stats));
        }
        let (interpreter, tracing) = (&seen[0], &seen[1]);
        assert!(
            interpreter.0.contains("Boolean(false)"),
            "{profile:?} {}",
            interpreter.0
        );
        assert_eq!(tracing.0, interpreter.0, "{profile:?}");
        // The trace was running when the failure came, and it made tables.
        assert!(tracing.1.native_entries > 0, "{profile:?} {:?}", tracing.1);
        assert!(
            tracing.1.table_new_helper_calls > 0,
            "{profile:?} {:?}",
            tracing.1
        );
    }
}

/// A loop whose body closes a table cycle and then stores again. The
/// recording must go on (no escape rejects it) and agree with the
/// interpreter. `exit` is a condition that leaves the trace after the
/// stores, and the loop checks the tables it left behind.
fn assert_cycle_followed(name: &str, body: &str) {
    let source = format!(
        "{DUMP}
         local total, keep = 0, nil
         for i = 1, {N} do
           local left, right = {{}}, {{}}
           left.other = right
           right.other = left
           {body}
           if i % 997 == 0 then keep = {{left, right}} end
           total = total + (left.other.other.n or 0) + (right.other.n or 0) + (left[1] or 0)
         end
         return total, tostring(keep[1].n), tostring(keep[2].n), tostring(keep[1][1]), keep[1].other == keep[2], keep[2].other == keep[1], keep[1].other.other == keep[1]"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_eq!(
            tracing.stats.total_aborts() + tracing.stats.side_aborts,
            0,
            "{name} {profile:?}: no recording is abandoned\n{:?}",
            tracing.stats
        );
        assert_traced(profile, &tracing, (N - 1_000) as u64);
        // An exit before a table is stored abandons the tables the trace
        // made for the iteration, and the interpreter makes them again.
        assert!(
            tracing.created >= interpreter.created && tracing.created - interpreter.created <= 64,
            "{name} {profile:?}: {} tables against the interpreter's {}\n{:?}",
            tracing.created,
            interpreter.created,
            tracing.stats
        );
        assert!(tracing.stats.table_new_helper_calls >= (N - 1_000) as u64);
    }
}

#[test]
fn a_string_store_after_the_cycle_closes_keeps_recording() {
    assert_cycle_followed("string", "left.n = i; right.n = -i");
}

#[test]
fn an_array_store_after_the_cycle_closes_keeps_recording() {
    assert_cycle_followed(
        "array",
        "left[1] = i; left[2] = i; right[1] = 5; left.n = i; right.n = 1",
    );
}

#[test]
fn an_overwrite_after_the_cycle_closes_keeps_recording() {
    assert_cycle_followed(
        "overwrite",
        "left.other = nil; left.other = right; right.other = left; left.n = 1; left.n = left.n + i; right.n = 2",
    );
}

#[test]
fn a_third_table_holding_both_after_the_cycle_closes_keeps_recording() {
    assert_cycle_followed(
        "third",
        "local both = {left, right}; both.tag = i; both[3] = both; left.n = i; right.n = 3",
    );
}

#[test]
fn a_store_that_closes_a_second_cycle_keeps_recording() {
    assert_cycle_followed(
        "second",
        "local third = {}; third.l = left; left.third = third; third.r = right; third.me = third; left.n = i; right.n = 3",
    );
}

#[test]
fn a_side_exit_after_the_cycle_closes_and_after_the_next_store() {
    let source = format!(
        "{DUMP}
         local total, seen = 0, {{}}
         for i = 1, {N} do
           local left, right = {{}}, {{}}
           left.other = right
           right.other = left
           if i % 1000 == 0 then seen[#seen + 1] = tostring(left.n) .. '|' .. tostring(right.n) .. tostring(left.other.other == left) .. tostring(right.other == left.other.other.other) end
           left.n = i
           if i % 1500 == 0 then seen[#seen + 1] = tostring(left.n) .. '|' .. tostring(right.n) .. tostring(left.other.other == left) end
           right.n = left.n + 1
           if i > {N} - 3 then seen[#seen + 1] = tostring(right.n) .. tostring(right.other.other == right) end
           total = total + right.n
         end
         return total, table.concat(seen, ';')"
    );
    for (profile, _, tracing) in agree(&source) {
        assert_eq!(
            tracing.stats.total_aborts() + tracing.stats.side_aborts,
            0,
            "{profile:?}: no recording is abandoned\n{:?}",
            tracing.stats
        );
        assert_traced(profile, &tracing, (N - 1_000) as u64);
    }
}

/// A table that escaped and was then stored into is read back through the
/// table that holds it: the second load must see the store, not merge with
/// the first.
fn assert_store_seen_through_holder(name: &str, setup: &str) {
    let source = format!(
        "{DUMP}
         local G, total = {{}}, 0
         for i = 1, {N} do
           {setup}
           l.n = 1
           G[1] = l
           local a = G[1].n
           l.n = i
           local b = G[1].n
           total = total + a + b
         end
         return total"
    );
    for (_, interpreter, tracing) in agree(&source) {
        assert_eq!(
            interpreter.result, tracing.result,
            "{name}: a store into an escaped table is seen by a later load"
        );
    }
}

#[test]
fn a_store_into_a_table_in_a_cycle_is_seen_by_a_load_through_its_holder() {
    assert_store_seen_through_holder("cycle", "local l, r = {}, {}; l.other = r; r.other = l");
}

#[test]
fn a_store_into_a_table_that_escaped_without_a_cycle_is_seen_by_a_load() {
    assert_store_seen_through_holder("plain", "local l = {}");
}
