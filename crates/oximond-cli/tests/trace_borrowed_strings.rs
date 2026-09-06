//! Strings read out of a table slot are borrowed by a trace: the trace holds
//! the slot's address, not a counted reference or a lane of the run's root
//! bank. An exit clones the string out of the slot into the register it
//! restores.
//!
//! Every program runs under the interpreter and under tracing execution in
//! every profile, and the results must match. Counters show that the loops
//! ran natively, that they needed no lane per string, and that a table set
//! after a live borrow rejects the recording instead of risking a stale slot.

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

/// More distinct strings than a run's root bank has table lanes
/// (`MAX_TRACE_TABLE_LANES` is 256).
const KEYS: usize = 3000;

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

/// Runs `source` in every profile under both engines, checks the results
/// agree, and returns each profile's result and tracing statistics.
fn agree(source: &str) -> Vec<(CompatibilityProfile, String, TraceStatistics)> {
    ALL_PROFILES
        .iter()
        .map(|&profile| {
            let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
            let (traced, stats) = run(profile, ExecutionMode::Tracing, source);
            assert_eq!(traced, expected, "{profile:?}\n{source}\n{stats:?}");
            (profile, expected, stats)
        })
        .collect()
}

fn keys_prelude() -> String {
    format!(
        "local keys = {{}}
         for i = 1, {KEYS} do keys[i] = 'key' .. i end"
    )
}

#[test]
fn strings_read_into_locals_restore_exactly_at_a_guard_exit() {
    // The loop leaves through a branch guard at iteration 2501 with the
    // string in `k` (and a copy of it in `c`), which the exit restores from
    // the slot. The interpreter then keeps both. The restored values are the
    // very strings of the table.
    let source = format!(
        "{}
         local kept, kept_copy, count = nil, nil, 0
         for i = 1, #keys do
           local k = keys[i]
           local c = k
           count = count + 1
           if count > 2500 then kept = k; kept_copy = c; break end
         end
         return kept, kept_copy, count, kept == keys[2501], rawequal(kept, keys[2501]),
                rawequal(kept_copy, keys[2501]), #kept",
        keys_prelude()
    );
    for (profile, result, stats) in agree(&source) {
        assert!(result.contains("key2501"), "{profile:?}: {result}");
        assert!(
            stats.iterations >= 1_000,
            "{profile:?}: {} native iterations\n{stats:?}",
            stats.iterations
        );
        assert_eq!(
            stats.aborts(TraceAbortReason::UnsupportedOperand),
            0,
            "{profile:?}: a string read must not abort the recording\n{stats:?}"
        );
        assert_eq!(
            stats.aborts(TraceAbortReason::BorrowOutlivesEffect),
            0,
            "{profile:?}\n{stats:?}"
        );
        assert!(
            stats.exits_of(TraceExitKind::GuardFail) >= 1,
            "{profile:?}: the loop must leave through the guard\n{stats:?}"
        );
    }
}

#[test]
fn a_run_over_more_strings_than_lanes_never_runs_out_of_lanes() {
    // 9000 reads of 3000 distinct strings pass through the trace. Had each
    // taken a lane of the root bank (256 of them), the read would have left
    // the trace at the 257th string, over and over. The only exits are the
    // few the loop itself takes.
    let source = format!(
        "{}
         local total, kept = 0, nil
         for round = 1, 3 do
           for i = 1, #keys do
             local k = keys[i]
             total = total + 1
             if total == 8765 then kept = k end
           end
         end
         return total, kept",
        keys_prelude()
    );
    for (profile, result, stats) in agree(&source) {
        assert!(result.contains("key2765"), "{profile:?}: {result}");
        assert!(
            stats.iterations >= 6_000,
            "{profile:?}: {} native iterations\n{stats:?}",
            stats.iterations
        );
        let leaves = stats.exits_of(TraceExitKind::GuardFail);
        assert!(
            leaves <= 20,
            "{profile:?}: {leaves} guard exits for 9000 reads, a lane per string would leave every 256\n{stats:?}"
        );
    }
}

#[test]
fn the_guard_that_fails_on_the_last_slot_restores_the_previous_string() {
    // The loop reads past the end of the array: that read leaves, and the
    // interpreter finishes the iteration. A branch guard leaves once with the
    // string live in `got`, which the exit rebuilds from its slot.
    let source = format!(
        "{}
         local kept, n = nil, 0
         for i = 1, #keys + 3 do
           local got = keys[i]
           n = n + 1
           if n == 2999 then kept = got end
         end
         return kept, n",
        keys_prelude()
    );
    for (profile, result, _) in agree(&source) {
        assert!(result.contains("key2999"), "{profile:?}: {result}");
    }
}

#[test]
fn a_table_set_after_a_live_borrow_rejects_the_recording() {
    // `k` still holds the borrow when the append stores into `out`. The VM
    // call of a set ends every borrow, and the recorder has no operation that
    // owns the string first, so the recording is rejected, the interpreter
    // runs the loop, and the results agree.
    let source = format!(
        "{}
         local out = {{}}
         local kept
         for i = 1, #keys do
           local k = keys[i]
           out[i] = i
           if i == #keys then kept = k end
         end
         return kept, #out",
        keys_prelude()
    );
    for (profile, result, stats) in agree(&source) {
        assert!(result.contains("key3000"), "{profile:?}: {result}");
        assert!(
            stats.aborts(TraceAbortReason::BorrowOutlivesEffect) > 0,
            "{profile:?}: the borrow must reject the recording\n{stats:?}"
        );
    }
}

#[test]
fn a_store_that_drops_the_source_slot_never_leaves_a_dangling_string() {
    // Each iteration reads a string and then sets its slot to nil: the slot
    // that a restore would read is gone. The trace must not be built around
    // the stale slot. `kept` is the string after the loop, and after
    // collections that free whatever nothing holds.
    let source = format!(
        "{}
         local kept, n = nil, 0
         for i = 1, #keys do
           local k = keys[i]
           keys[i] = nil
           n = n + 1
           if n == 2999 then kept = k; break end
         end
         collectgarbage()
         collectgarbage()
         return kept, n, keys[2999], keys[3000]",
        keys_prelude()
    );
    for (profile, result, stats) in agree(&source) {
        assert!(result.contains("key2999"), "{profile:?}: {result}");
        assert!(
            stats.aborts(TraceAbortReason::BorrowOutlivesEffect) > 0,
            "{profile:?}: the borrow must reject the recording\n{stats:?}"
        );
    }
}

#[test]
fn a_set_before_the_read_is_not_a_live_borrow() {
    // The set precedes the read in each iteration, so no register holds a
    // borrow when it runs: the loop traces, and the exit restores the string.
    let source = format!(
        "{}
         local out = {{}}
         local kept
         local n = 0
         for i = 1, #keys do
           out[i] = i
           local k = keys[i]
           n = n + 1
           if n > 2800 then kept = k; break end
         end
         return kept, #out, n",
        keys_prelude()
    );
    for (profile, result, stats) in agree(&source) {
        assert!(result.contains("key2801"), "{profile:?}: {result}");
        assert_eq!(
            stats.aborts(TraceAbortReason::BorrowOutlivesEffect),
            0,
            "{profile:?}\n{stats:?}"
        );
        assert!(stats.iterations >= 1_000, "{profile:?}\n{stats:?}");
    }
}

#[test]
fn borrowed_strings_do_not_compare_by_address() {
    // Equal bytes in two allocations are equal strings, and a string never
    // equals a number. Both compare in the interpreter.
    let source = format!(
        "{}
         local twins = {{}}
         for i = 1, {KEYS} do twins[i] = 'key' .. i end
         local same, diff, other = 0, 0, 0
         for i = 1, #keys do
           local a, b = keys[i], twins[i]
           if a == b then same = same + 1 end
           if a == 1 then other = other + 1 end
           if a ~= twins[(i % #twins) + 1] then diff = diff + 1 end
         end
         return same, diff, other",
        keys_prelude()
    );
    for (profile, result, _) in agree(&source) {
        assert!(result.contains("3000"), "{profile:?}: {result}");
    }
}

#[test]
fn the_strings_of_an_ipairs_loop_survive_a_break() {
    let source = format!(
        "{}
         local last, n = nil, 0
         for i, s in ipairs(keys) do
           last = s
           n = n + 1
           if n == 2700 then break end
         end
         return last, n",
        keys_prelude()
    );
    for (profile, result, _) in agree(&source) {
        assert!(result.contains("key2700"), "{profile:?}: {result}");
    }
}

#[test]
fn a_side_trace_takes_the_borrow_its_parent_holds() {
    // The branch alternates, so its guard fails half the time and a side
    // trace is recorded for the other arm. That arm reads `k`, which the
    // parent holds as a borrow: the side gets the slot's address as a
    // borrowed input of its own (not a lane id), and its exits restore the
    // string from the slot. Results agree whatever the tree does.
    let source = format!(
        "{}
         local odd, even, kept, n = 0, 0, nil, 0
         for round = 1, 4 do
           for i = 1, #keys do
             local k = keys[i]
             n = n + 1
             if i % 2 == 0 then
               even = even + 1
             else
               odd = odd + 1
               if n % 7 == 0 then kept = k end
             end
           end
         end
         return odd, even, kept, n",
        keys_prelude()
    );
    for (profile, result, stats) in agree(&source) {
        assert!(result.contains("6000"), "{profile:?}: {result}");
        assert!(stats.iterations >= 1_000, "{profile:?}\n{stats:?}");
    }
}
