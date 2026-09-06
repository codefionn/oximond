//! A string key known only at run time (`values[keys[i]]`) traces natively:
//! the key is a borrow of an array slot (or a lane string). A borrowed key in
//! a trace that moves no hash storage is probed in native code, in a string
//! index of the table (any representation); every other lookup is a read-only
//! call into the VM.
//!
//! Every program runs under the interpreter and under tracing execution in
//! every profile, and the results must match. Counters show that the loops
//! ran natively, that a run over more strings than lanes needs no lane per
//! string, and that every way the lookup can fail leaves the trace with the
//! interpreter's state.

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

/// `keys[i]` names `values`: both tables hold the same `KEYS` strings, and
/// the value of `key<i>` is `i % 107`.
fn prelude() -> String {
    format!(
        "local keys, values = {{}}, {{}}
         for i = 1, {KEYS} do
           local k = 'key' .. i
           keys[i] = k
           values[k] = i % 107
         end"
    )
}

fn assert_ran_natively(profile: CompatibilityProfile, stats: &TraceStatistics, least: u64) {
    assert!(
        stats.iterations >= least,
        "{profile:?}: {} native iterations, wanted {least}\n{stats:?}",
        stats.iterations
    );
    assert!(stats.native_entries >= 1, "{profile:?}\n{stats:?}");
    assert!(
        stats.traces_compiled >= 1,
        "{profile:?}: the dynamic key must not abort the recording\n{stats:?}"
    );
}

#[test]
fn a_dynamic_key_loop_runs_natively_over_more_strings_than_lanes() {
    // 9000 lookups of 3000 distinct strings. A lane per string would leave
    // the trace at the 257th, over and over.
    let sum: usize = (1..=KEYS).map(|i| i % 107).sum::<usize>() * 3;
    let source = format!(
        "{}
         local total = 0
         for round = 1, 3 do
           for i = 1, #keys do
             total = total + values[keys[i]]
           end
         end
         return total",
        prelude()
    );
    for (profile, result, stats) in agree(&source) {
        assert!(result.contains(&sum.to_string()), "{profile:?}: {result}");
        assert_ran_natively(profile, &stats, 6_000);
        let leaves = stats.exits_of(TraceExitKind::GuardFail);
        assert!(
            leaves <= 20,
            "{profile:?}: {leaves} guard exits for 9000 lookups\n{stats:?}"
        );
    }
}

#[test]
fn equal_bytes_in_different_allocations_hit() {
    // The probes are built apart from the keys the table holds, with short
    // and long names, so the lookup compares bytes and not addresses.
    let source = format!(
        "{}
         local long, long_values, long_probes = 'a_rather_long_prefix_that_leaves_the_inline_form_', {{}}, {{}}
         for i = 1, {KEYS} do
           long_values[long .. i] = i
           long_probes[i] = long .. i
         end
         local probes = {{}}
         for i = 1, {KEYS} do probes[i] = 'key' .. i end
         local short, longs = 0, 0
         for round = 1, 3 do
           for i = 1, #probes do
             short = short + values[probes[i]]
             longs = longs + long_values[long_probes[i]]
           end
         end
         return short, longs",
        prelude()
    );
    let short: usize = (1..=KEYS).map(|i| i % 107).sum::<usize>() * 3;
    let longs: usize = (1..=KEYS).sum::<usize>() * 3;
    for (profile, result, stats) in agree(&source) {
        assert!(result.contains(&short.to_string()), "{profile:?}: {result}");
        assert!(result.contains(&longs.to_string()), "{profile:?}: {result}");
        assert_ran_natively(profile, &stats, 3_000);
    }
}

#[test]
fn missing_keys_read_as_nil_through_a_side_trace() {
    // One probe in four names a key the table lacks. The lookup that found a
    // number leaves at the first miss; a side trace takes the nil arm. The
    // side takes the key as the borrow its parent holds.
    let source = format!(
        "{}
         local probes = {{}}
         for i = 1, {KEYS} do
           probes[i] = (i % 4 == 0) and ('absent' .. i) or keys[i]
         end
         local total, missing = 0, 0
         for round = 1, 3 do
           for i = 1, #probes do
             local value = values[probes[i]]
             if value then total = total + value else missing = missing + 1 end
           end
         end
         return total, missing",
        prelude()
    );
    let missing = KEYS / 4 * 3;
    for (profile, result, stats) in agree(&source) {
        assert!(
            result.contains(&missing.to_string()),
            "{profile:?}: {result}"
        );
        assert_ran_natively(profile, &stats, 3_000);
        // (Whether the side is entered or runs as a member of the tree
        // depends on when each loop got hot, which the fill loops above
        // change, so the side trace's existence is what is checked.)
        assert!(
            stats.side_installs > 0,
            "{profile:?}: the nil arm should run in a side trace\n{stats:?}"
        );
    }
}

#[test]
fn a_loop_of_only_missing_keys_reads_nil_natively() {
    let source = format!(
        "{}
         local ghosts = {{}}
         for i = 1, {KEYS} do ghosts[i] = 'ghost' .. i end
         local seen = 0
         for round = 1, 2 do
           for i = 1, #ghosts do
             if not values[ghosts[i]] then seen = seen + 1 end
           end
         end
         return seen",
        prelude()
    );
    for (profile, result, stats) in agree(&source) {
        assert!(result.contains("6000"), "{profile:?}: {result}");
        assert_ran_natively(profile, &stats, 3_000);
    }
}

#[test]
fn a_table_with_a_metatable_is_looked_up_by_the_interpreter() {
    // `__index` answers missing keys, with a function and with a table. The
    // lookup never runs it. A function that answers (here or at the end of a
    // chain) is a call the trace makes behind a check of which handler runs,
    // so the loop may run natively and must give the interpreter's results.
    // A table that answers is not recorded: the probes of present keys read
    // raw natively and the `ghost` probes leave the trace.
    for handler in [
        "function(_, key) return #key end",
        "setmetatable({}, {__index = function(_, key) return 3 end})",
        "{ghost = 4}",
    ] {
        let source = format!(
            "{}
             setmetatable(values, {{__index = {handler}}})
             local probes = {{}}
             for i = 1, {KEYS} do probes[i] = (i % 3 == 0) and 'ghost' or keys[i] end
             local total = 0
             for round = 1, 2 do
               for i = 1, #probes do
                 total = total + (values[probes[i]] or 0)
               end
             end
             return total",
            prelude()
        );
        // The loop that fills the tables runs natively (a store under a
        // computed key is traced), and only that loop may.
        let fills: Vec<_> = agree(&prelude())
            .into_iter()
            .map(|(profile, _, stats)| (profile, stats.native_entries))
            .collect();
        for ((profile, result, stats), (_, fill_entries)) in agree(&source).into_iter().zip(fills) {
            assert!(!result.contains("Nil"), "{profile:?}: {result}");
            assert!(
                stats.native_entries > fill_entries,
                "{profile:?}: the probe loop never ran natively\n{stats:?}"
            );
        }
    }
}

#[test]
fn a_metatable_set_between_entries_is_seen_by_the_next_entry() {
    // Rounds 1 and 2 run the trace over a plain table. Before round 3 the
    // table gets an `__index`, and every probe in round 3 and 4 is a miss it
    // answers. The entry refuses the table, so the interpreter answers them.
    let source = format!(
        "{}
         local probes = {{}}
         for i = 1, {KEYS} do probes[i] = (i % 2 == 0) and 'ghost' or keys[i] end
         local total = 0
         for round = 1, 4 do
           if round == 3 then
             setmetatable(values, {{__index = function() return 1000 end}})
           end
           for i = 1, #probes do
             total = total + (values[probes[i]] or 0)
           end
         end
         return total",
        prelude()
    );
    let odd: usize = (1..=KEYS).filter(|i| i % 2 == 1).map(|i| i % 107).sum();
    let expected = odd * 4 + (KEYS / 2) * 1000 * 2;
    for (profile, result, stats) in agree(&source) {
        assert!(
            result.contains(&expected.to_string()),
            "{profile:?}: {result}"
        );
        assert_ran_natively(profile, &stats, 1_000);
    }
}

#[test]
fn a_result_that_changes_type_leaves_the_trace_exactly() {
    // The lookup specialises on the number it found. Between rounds a few
    // values become floats, a boolean and nil; each exit hands the
    // interpreter the registers it needs (`k`, `total`) intact.
    let source = format!(
        "{}
         local total, last = 0, nil
         for round = 1, 5 do
           if round == 2 then values[keys[700]] = 0.5 end
           if round == 3 then values[keys[1400]] = true end
           if round == 4 then values[keys[2100]] = nil end
           if round == 5 then values[keys[700]] = 7; values[keys[1400]] = 9 end
           for i = 1, #keys do
             local k = keys[i]
             local v = values[k]
             if v == true then total = total + 1000
             elseif v then total = total + v
             else total = total + 5000 end
             if i == #keys then last = k end
           end
         end
         return total, last",
        prelude()
    );
    for (profile, result, stats) in agree(&source) {
        assert!(result.contains("key3000"), "{profile:?}: {result}");
        assert_ran_natively(profile, &stats, 1_000);
        assert!(
            stats.exits_of(TraceExitKind::GuardFail) >= 1,
            "{profile:?}: the changed values should leave the trace\n{stats:?}"
        );
    }
}

#[test]
fn guard_and_loop_exits_restore_the_key_and_the_value() {
    // The loop leaves through a branch guard at the first value of 100 with
    // the key in `kept` and the value in `found`: the exit clones the string
    // out of its slot. The interpreter then keeps both.
    let source = format!(
        "{}
         local kept, found, n = nil, nil, 0
         for i = 1, #keys do
           local k = keys[i]
           local v = values[k]
           n = n + 1
           if v == 100 and i > 2500 then kept = k; found = v; break end
         end
         return kept, found, n, rawequal(kept, keys[n])",
        prelude()
    );
    for (profile, result, stats) in agree(&source) {
        assert!(result.contains("key2561"), "{profile:?}: {result}");
        assert_ran_natively(profile, &stats, 1_000);
        assert!(
            stats.exits_of(TraceExitKind::GuardFail) >= 1,
            "{profile:?}\n{stats:?}"
        );
    }
}

#[test]
fn a_long_run_polls_and_resumes_with_the_right_state() {
    // Far more iterations than one native run is allowed: the trace stops at
    // its poll and the interpreter re-enters it.
    let source = format!(
        "{}
         local total = 0
         for i = 1, 700000 do
           local k = keys[i % {KEYS} + 1]
           total = total + values[k]
         end
         return total",
        prelude()
    );
    let sum: usize = (1..=700_000usize).map(|i| (i % KEYS + 1) % 107).sum();
    for (profile, result, stats) in agree(&source) {
        assert!(result.contains(&sum.to_string()), "{profile:?}: {result}");
        assert_ran_natively(profile, &stats, 100_000);
        assert!(
            stats.exits_of(TraceExitKind::Poll) >= 1,
            "{profile:?}: a long run must stop at its poll\n{stats:?}"
        );
    }
}

#[test]
fn a_store_between_the_key_read_and_the_lookup_rejects_the_recording() {
    // `keys[i] = nil` may drop the slot the borrowed key points at, and the
    // lookup still needs it: the recording is rejected instead of reading a
    // slot that is gone, and the results agree.
    let source = format!(
        "{}
         local total = 0
         for i = 1, #keys do
           local k = keys[i]
           keys[i] = nil
           total = total + values[k]
         end
         collectgarbage()
         return total, keys[1], keys[3000]",
        prelude()
    );
    let sum: usize = (1..=KEYS).map(|i| i % 107).sum();
    for (profile, result, stats) in agree(&source) {
        assert!(result.contains(&sum.to_string()), "{profile:?}: {result}");
        assert!(
            stats.aborts(TraceAbortReason::BorrowOutlivesEffect) > 0,
            "{profile:?}: the borrow must reject the recording\n{stats:?}"
        );
    }
}

#[test]
fn a_store_after_the_lookup_is_fine_once_the_key_is_dead() {
    let source = format!(
        "{}
         local total, out = 0, {{}}
         for i = 1, #keys do
           total = total + values[keys[i]]
           keys[i] = nil
         end
         return total",
        prelude()
    );
    let sum: usize = (1..=KEYS).map(|i| i % 107).sum();
    for (profile, result, _) in agree(&source) {
        assert!(result.contains(&sum.to_string()), "{profile:?}: {result}");
    }
}

#[test]
fn a_string_held_in_a_local_is_a_lane_key() {
    // The key is not read out of a table in the loop: the entry gives the
    // string a lane, and the lookup resolves it through the root bank.
    let source = format!(
        "{}
         local key = 'key' .. 77
         local other = 'key' .. 78
         local total = 0
         for i = 1, 20000 do
           total = total + values[key] + values[other]
         end
         return total",
        prelude()
    );
    for (profile, result, stats) in agree(&source) {
        assert!(
            result.contains(&(20000 * (77 + 78)).to_string()),
            "{profile:?}: {result}"
        );
        assert_ran_natively(profile, &stats, 10_000);
    }
}

#[test]
fn every_table_representation_and_value_kind_traces() {
    // A small table (ordered scan), a record built by a constructor (shaped),
    // a large one (hashed index), with integers, floats and booleans.
    let source = "
        local names = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'}
        local shaped = {a = 1, b = 2, c = 3, d = 4, e = 5, f = 6, g = 7, h = 8}
        local small = {}
        for i, name in ipairs(names) do small[name] = i * 2 end
        local floats = {a = 0.5, b = 1.5, c = 2.5, d = 3.5, e = 4.5, f = 5.5, g = 6.5, h = 7.5}
        local flags = {a = true, b = false, c = true, d = false, e = true, f = false, g = true, h = false}
        local wide, wide_names = {}, {}
        for i = 1, 400 do
          local name = 'wide_' .. i
          wide_names[i] = name
          wide[name] = i
        end
        local sums, float_sum, set, wide_sum = 0, 0, 0, 0
        for round = 1, 3000 do
          for i = 1, 8 do
            local name = names[i]
            sums = sums + shaped[name] + small[name]
            float_sum = float_sum + floats[name]
            if flags[name] then set = set + 1 end
          end
        end
        for round = 1, 20 do
          for i = 1, 400 do wide_sum = wide_sum + wide[wide_names[i]] end
        end
        return sums, float_sum, set, wide_sum";
    for (profile, result, stats) in agree(source) {
        assert!(result.contains("324000"), "{profile:?}: {result}");
        assert!(result.contains("96000"), "{profile:?}: {result}");
        assert!(result.contains("12000"), "{profile:?}: {result}");
        assert!(result.contains("1604000"), "{profile:?}: {result}");
        // The eight-trip loop is unrolled into the loop of the rounds, whose
        // 3000 iterations count where the 24000 inner trips used to.
        assert_ran_natively(profile, &stats, 2_900);
    }
}

#[test]
fn two_tables_share_one_borrowed_key() {
    let source = format!(
        "{}
         local second = {{}}
         for i = 1, {KEYS} do second[keys[i]] = i end
         local total = 0
         for round = 1, 3 do
           for i = 1, #keys do
             local k = keys[i]
             total = total + values[k] + second[k]
           end
         end
         return total",
        prelude()
    );
    let sum: usize = (1..=KEYS).map(|i| i % 107 + i).sum::<usize>() * 3;
    for (profile, result, stats) in agree(&source) {
        assert!(result.contains(&sum.to_string()), "{profile:?}: {result}");
        assert_ran_natively(profile, &stats, 3_000);
    }
}

#[test]
fn a_collection_between_runs_never_sees_a_stale_key() {
    // The outer loop shuffles the key array and collects garbage between
    // runs of the trace. Each entry borrows afresh.
    let source = format!(
        "{}
         local total = 0
         for round = 1, 6 do
           for i = 1, #keys - 1 do
             total = total + values[keys[i]]
           end
           keys[1], keys[#keys] = keys[#keys], keys[1]
           local garbage = {{}}
           for j = 1, 500 do garbage[j] = {{j, tostring(j)}} end
           collectgarbage()
         end
         return total",
        prelude()
    );
    for (profile, result, stats) in agree(&source) {
        assert!(!result.contains("Nil"), "{profile:?}: {result}");
        assert_ran_natively(profile, &stats, 3_000);
    }
}

#[test]
fn a_side_trace_uses_and_restores_the_key_its_parent_borrowed() {
    // The branch on the value alternates, so the other arm becomes a side
    // trace. It starts with the key as a borrowed input, keeps it in `kept`
    // (an exit restores the string from the slot) and breaks out of the loop
    // from inside the side.
    let source = format!(
        "{}
         local even, odd, kept = 0, 0, nil
         for round = 1, 4 do
           for i = 1, #keys do
             local k = keys[i]
             local v = values[k]
             if v % 2 == 0 then
               even = even + v
             else
               odd = odd + v
               if round == 4 and i > 2900 and v == 45 then kept = k; break end
             end
           end
         end
         return even, odd, kept",
        prelude()
    );
    for (profile, result, stats) in agree(&source) {
        assert!(result.contains("key2934"), "{profile:?}: {result}");
        assert_ran_natively(profile, &stats, 5_000);
        assert!(
            stats.side_traces_recorded >= 1,
            "{profile:?}: the other arm should become a side trace\n{stats:?}"
        );
    }
}

/// Reads through a probed index keep every semantic of the lookup: the tests
/// below change the table between and inside the runs of a loop that reads
/// it, in each way a table can change, and compare with the interpreter.
mod probed_reads {
    use super::*;

    /// The loop that reads `values[keys[i]]` into `total`, three rounds,
    /// with `between` run before each round (in the outer loop, so the
    /// inner loop's trace is entered after it) and `inside` run in each
    /// iteration, after the read.
    fn rounds(between: &str, inside: &str) -> String {
        format!(
            "{}
             local total, other = 0, 0
             for round = 1, 6 do
               {between}
               for i = 1, #keys do
                 local value = values[keys[i]]
                 if type(value) == 'number' then total = total + value else other = other + 1 end
                 {inside}
               end
             end
             return total, other",
            prelude()
        )
    }

    fn sum_of_values() -> usize {
        (1..=KEYS).map(|i| i % 107).sum()
    }

    #[test]
    fn a_rehash_between_runs_rebuilds_the_index() {
        // Round 3 adds 20000 keys, which rehashes the table many times over:
        // every address the index of rounds 1 and 2 held is gone.
        let source = rounds(
            "if round == 3 then
               for i = 1, 20000 do values['extra' .. i] = i end
             end
             if round == 5 then
               for i = 1, 20000 do values['extra' .. i] = nil end
               for i = 1, 20000 do values['more' .. i] = i end
             end",
            "",
        );
        let expected = (sum_of_values() * 6).to_string();
        for (profile, result, stats) in agree(&source) {
            assert!(result.contains(&expected), "{profile:?}: {result}");
            assert_ran_natively(profile, &stats, 12_000);
        }
    }

    #[test]
    fn a_rehash_inside_the_loop_falls_back_to_the_runtime() {
        // A store of a new key moves hash storage, so the trace cannot trust
        // an index: the lookups are the runtime's, and are right.
        let source = rounds(
            "",
            "if i % 7 == 0 then values['grow' .. round .. '_' .. i] = i end",
        );
        let expected = (sum_of_values() * 6).to_string();
        for (profile, result, stats) in agree(&source) {
            assert!(result.contains(&expected), "{profile:?}: {result}");
            assert_ran_natively(profile, &stats, 6_000);
        }
    }

    #[test]
    fn a_key_deleted_between_and_inside_runs_reads_nil() {
        // Round 2 deletes a third of the keys; the loop then counts them as
        // `other`. Round 4 puts them back with another value, and from round
        // 5 the loop itself deletes every key after it reads it.
        let source = rounds(
            "if round == 2 then
               for i = 3, #keys, 3 do values[keys[i]] = nil end
             end
             if round == 4 then
               for i = 3, #keys, 3 do values[keys[i]] = 1 end
             end",
            "if round >= 5 then values[keys[i]] = nil end",
        );
        for (profile, result, stats) in agree(&source) {
            // Rounds 1, 3 and 4... the exact figures are the interpreter's.
            assert!(!result.contains("Nil"), "{profile:?}: {result}");
            assert_ran_natively(profile, &stats, 6_000);
        }
    }

    #[test]
    fn a_value_that_changes_type_between_runs_is_read_with_its_new_type() {
        // Integer, then float, then boolean, then string, then integer again.
        let source = rounds(
            "local kinds = {7, 1.5, true, 'text', 9}
             local replacement = kinds[round]
             if replacement ~= nil then
               for i = 2, #keys, 2 do values[keys[i]] = replacement end
             end",
            "",
        );
        for (profile, result, stats) in agree(&source) {
            assert!(!result.contains("Nil"), "{profile:?}: {result}");
            assert_ran_natively(profile, &stats, 3_000);
        }
    }

    #[test]
    fn an_index_metamethod_set_mid_loop_answers_the_misses() {
        // The loop sets `__index` on the table at iteration 1500 of every
        // round. Reads of present keys stay raw; the ghosts it reads after
        // that run the handler.
        let source = format!(
            "{}
             local probes = {{}}
             for i = 1, #keys do probes[i] = (i % 5 == 0) and ('ghost' .. i) or keys[i] end
             local mt = {{__index = function(_, key) return #key end}}
             local total = 0
             for round = 1, 4 do
               for i = 1, #probes do
                 if round == 2 and i == 1500 then setmetatable(values, mt) end
                 total = total + (values[probes[i]] or -1)
               end
               setmetatable(values, nil)
             end
             return total",
            prelude()
        );
        for (profile, result, stats) in agree(&source) {
            assert!(!result.contains("Nil"), "{profile:?}: {result}");
            assert_ran_natively(profile, &stats, 3_000);
        }
    }

    #[test]
    fn weak_tables_and_a_collection_between_runs() {
        // Weak values: every 97th entry holds a table nobody else holds, and
        // a collection between rounds removes it (the layout changes). Weak
        // keys: the strings stay (strings are never collected from a weak
        // table), and the table keys come and go beside them.
        for mode in ["v", "k", "kv"] {
            let source = format!(
                "local keys = {{}}
                 local values = setmetatable({{}}, {{__mode = '{mode}'}})
                 local held = {{}}
                 for i = 1, {KEYS} do
                   local k = 'key' .. i
                   keys[i] = k
                   if i % 97 == 0 then values[k] = {{}} else values[k] = i % 107 end
                   values[{{}}] = i
                   held[#held + 1] = {{}}
                 end
                 local total, tables = 0, 0
                 for round = 1, 6 do
                   for i = 1, #keys do
                     local value = values[keys[i]]
                     if type(value) == 'number' then total = total + value
                     elseif value ~= nil then tables = tables + 1 end
                     if round == 3 and i == 1000 then collectgarbage() end
                   end
                   if round % 2 == 0 then collectgarbage() end
                 end
                 return total, tables"
            );
            for (profile, result, stats) in agree(&source) {
                assert!(!result.contains("Nil"), "{profile:?} {mode}: {result}");
                assert_ran_natively(profile, &stats, 1_500);
            }
        }
    }

    #[test]
    fn long_numeric_and_near_equal_string_keys_find_their_own_entries() {
        // Keys that are numbers in disguise (`'1'` is not 1), strings that
        // differ in the last byte or by a zero byte, inline strings of every
        // length up to the limit and long strings that need a byte compare.
        // Each probe is built apart from the key the table stores.
        let source = "
            local names, values = {}, {}
            local function add(name, value) names[#names + 1] = name; values[name] = value end
            for i = 1, 40 do add(tostring(i), i) end
            add('01', 1000)
            add('1.0', 2000)
            add('1e1', 3000)
            add('0x10', 4000)
            add('a', 5000)
            add('a\\0', 6000)
            add('a\\0\\0', 7000)
            for length = 1, 40 do add(string.rep('k', length), length * 10) end
            for length = 20, 26 do add(string.rep('é', length), length * 100) end
            values[1] = 'array one'
            values[10] = 'ten'
            values[1.5] = 'float key'
            local probes = {}
            for i, name in ipairs(names) do probes[i] = ('' .. name):sub(1) end
            for i = 1, 60 do
                for j = 1, #names do local _ = probes[j] end
            end
            local total, count = 0, 0
            for round = 1, 8 do
                for i = 1, #probes do
                    local value = values[probes[i]]
                    total = total + value
                    count = count + 1
                end
            end
            return total, count";
        for (profile, result, stats) in agree(source) {
            assert!(!result.contains("Nil"), "{profile:?}: {result}");
            assert_ran_natively(profile, &stats, 100);
        }
    }

    /// The counters across two runs of one program that differ only in
    /// how many rounds the loop makes.
    fn calls_for(profile: CompatibilityProfile, keys: usize, rounds: usize) -> (u64, u64) {
        let source = format!(
            "local keys, values = {{}}, {{}}
             for i = 1, {keys} do
               local k = 'key' .. i
               keys[i] = k
               values[k] = i % 107
             end
             local total = 0
             for round = 1, {rounds} do
               for i = 1, #keys do total = total + values[keys[i]] end
             end
             return total"
        );
        let (_, stats) = run(profile, ExecutionMode::Tracing, &source);
        (stats.hash_helper_calls, stats.iterations)
    }

    #[test]
    fn a_hot_read_loop_asks_the_runtime_for_nothing_after_warmup() {
        // Tables of every representation: ordered (12 keys), hashed with few
        // keys, hashed with many. After the first entries, more rounds add
        // iterations and no helper call.
        for profile in ALL_PROFILES {
            for keys in [12, 40, 3000] {
                let (few_calls, few) = calls_for(profile, keys, 20);
                let (many_calls, many) = calls_for(profile, keys, 120);
                assert!(
                    many > few + 50 * keys as u64,
                    "{profile:?} {keys}: {few} then {many} iterations"
                );
                assert_eq!(
                    many_calls, few_calls,
                    "{profile:?} {keys} keys: the hot loop still calls the runtime"
                );
                // And what is called before the table has an index is at
                // most a couple of runs' worth.
                assert!(
                    few_calls <= 2 * keys as u64,
                    "{profile:?} {keys} keys: {few_calls} calls"
                );
            }
        }
    }
}
