//! Traces that read nil past the end of an array and fill an array's holes.
//!
//! A read of a key outside the array part of a table with no metatable and an
//! empty hash part is nil, and a trace makes it itself (`TraceOp::ArrayLoad`
//! with `past_end`), so a side trace that starts on the iteration that ran
//! off the end installs instead of missing the table. A store into a nil hole
//! of such a table is a plain write of the slot, unless it fills the slot after
//! the border. Every program runs under the interpreter and under tracing
//! execution in every profile, and the results must match exactly.

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

/// Runs `source` in every profile under the interpreter and under tracing
/// execution, checks the results agree, and returns each profile's result
/// and tracing statistics.
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

/// [`agree`], requiring that loops ran natively in every profile.
fn agree_native(source: &str) {
    for (profile, _, stats) in agree(source) {
        assert!(
            stats.iterations >= 300,
            "{profile:?}: {} native iterations\n{source}\n{stats:?}",
            stats.iterations
        );
    }
}

/// The program of `opt/range_fail`: reads that run off the end of the array
/// by an offset that changes per round.
const RANGE_FAIL: &str = "
    local n = 3000
    local values = {}
    for i = 1, n do values[i] = i % 97 end
    local total, missing = 0, 0
    for round = 1, 24 do
      local offset = (round % 4) * 500
      for i = 1, n do
        local v = values[i + offset]
        if v then total = total + v else missing = missing + 1 end
      end
    end
    return total, missing";

#[test]
fn a_read_past_the_end_of_an_array_installs_its_side_trace() {
    for (profile, result, stats) in agree(RANGE_FAIL) {
        assert!(
            result.contains("2602800") && result.contains("18000"),
            "{profile:?}: {result}"
        );
        assert_eq!(
            stats.aborts(TraceAbortReason::TableMiss),
            0,
            "{profile:?}\n{stats:?}"
        );
        // The outer side that carries the `round` loop is recorded as soon
        // as its exit fails (the latch it resumes at loops back), and the
        // scout for the code beyond the nest behind it may run out of budget
        // once: that is the only abort.
        assert!(
            stats.side_installs >= 1 && stats.side_aborts <= 1,
            "{profile:?}: the side trace of the first iteration past the end installs\n{stats:?}"
        );
        // Most of the 72000 iterations run natively, past the end or not.
        assert!(stats.iterations >= 50_000, "{profile:?}\n{stats:?}");
    }
}

#[test]
fn a_key_past_the_array_end_that_lives_in_the_hash_part_is_found() {
    // The hash part holds keys past the array's end: the table is not solid,
    // so the trace never reads a nil for them.
    let source = "
        local t = {}
        for i = 1, 300 do t[i] = i end
        t[400], t[401], t[1000] = 4, 5, 6
        local s = 0
        for round = 1, 30 do
          for i = 1, 300 do
            local v = t[i + (round % 2) * 100]
            if v then s = s + v end
            s = s + (t[1000] or 0)
          end
        end
        return s, t[400], t[1000], t[999]";
    // Recording an integer key that lives in the hash part still aborts with
    // TableMiss, so this loop mostly runs in the interpreter. The test pins
    // that tracing never answers such a key with a past-end nil.
    for (profile, result, _stats) in agree(source) {
        // The sum is stock Lua's (t[1000] alone is read 9000 times).
        assert!(result.contains("1332810"), "{profile:?}: {result}");
    }
}

#[test]
fn a_key_stored_past_the_end_mid_loop_stops_reading_nil() {
    // Key 150 is nil past the end of a 100 slot array until iteration 250,
    // then holds a number in the hash part. The read of it is nil in the
    // trace only while the header says the hash part is empty.
    let source = "
        local t = {}
        for i = 1, 100 do t[i] = i end
        local found, last = 0, 0
        for i = 1, 1000 do
          local v = t[150]
          if v then found = found + 1; last = v end
          if i == 250 or i == 600 then t[150] = i end
        end
        return found, last, t[150], #t";
    for (profile, result, stats) in agree(source) {
        assert!(result.contains("750"), "{profile:?}: {result}\n{stats:?}");
    }
}

#[test]
fn a_metatable_with_index_that_appears_mid_loop_runs_for_the_misses() {
    let source = "
        local t = {}
        for i = 1, 200 do t[i] = i end
        local s = 0
        for i = 1, 3000 do
          local v = t[i]
          if v then s = s + v else s = s - 1 end
          if i == 1500 then
            setmetatable(t, {__index = function(_, k) return 1000 end})
          end
        end
        local meta = 0
        for i = 190, 260 do meta = meta + t[i] end
        return s, meta";
    agree_native(source);
}

#[test]
fn a_border_that_moves_with_nil_stores_changes_what_is_past_the_end() {
    let source = "
        local t = {}
        for i = 1, 400 do t[i] = i end
        local seen, nils = 0, 0
        for i = 1, 1200 do
          local v = t[#t + 1]
          if v then seen = seen + 1 else nils = nils + 1 end
          local w = t[400 + (i % 3)]
          if w then seen = seen + w else nils = nils + 1 end
          if i % 2 == 0 then t[#t] = nil end
        end
        return seen, nils, #t";
    agree_native(source);
}

#[test]
fn growth_across_the_capacity_mid_loop_keeps_reads_exact() {
    let source = "
        local t = {}
        local s, nils = 0, 0
        for i = 1, 5000 do
          local ahead = t[i]
          if ahead then s = s + ahead else nils = nils + 1 end
          local further = t[i + 3]
          if further then s = s + 1 end
          t[i] = i
          if i % 7 == 0 then t[i + 1] = i end
        end
        return s, nils, #t";
    agree_native(source);
}

#[test]
fn zero_negative_fractional_and_float_keys_read_as_they_do_in_the_interpreter() {
    let source = "
        local t = {}
        for i = 1, 100 do t[i] = i end
        local s, nils = 0, 0
        for i = 1, 6000 do
          local a, b, c = t[0], t[-i], t[1.5]
          local d = t[(i % 150) + 0.0]
          local e = t[i % 3 + 100.0]
          local f = t[(i % 150) + 0.5]
          local nan = t[0/0]
          if nan then s = s + 1000 end
          if a then s = s + 1 else nils = nils + 1 end
          if b then s = s + 1 else nils = nils + 1 end
          if c then s = s + 1 else nils = nils + 1 end
          if d then s = s + d else nils = nils + 1 end
          if e then s = s + 1 else nils = nils + 1 end
          if f then s = s + 1 else nils = nils + 1 end
        end
        return s, nils";
    agree_native(source);
}

#[test]
fn an_append_into_a_table_being_traversed_agrees() {
    let source = "
        local t = {}
        for i = 1, 300 do t[i] = i end
        local s = 0
        for i, v in ipairs(t) do
          s = s + v
          if i <= 200 then t[#t + 1] = v end
        end
        local n = 0
        for _ = 1, 300 do
          for k, v in pairs(t) do
            if v > 0 then n = n + 1 end
            t[k] = v
          end
        end
        return s, #t, n";
    agree_native(source);
}

#[test]
fn a_hot_append_loop_does_not_call_the_set_helper() {
    let source = "
        local t = {}
        for i = 1, 60000 do t[i] = i * 2 end
        local s = 0
        for i = 1, #t do s = s + t[i] end
        return #t, s";
    for (profile, result, stats) in agree(source) {
        assert!(result.contains("60000"), "{profile:?}: {result}");
        assert!(stats.iterations >= 100_000, "{profile:?}\n{stats:?}");
        // The array grows by reserve inside the trace; the VM's set entry is
        // not called after the first iterations.
        assert!(
            stats.array_set_helper_calls <= 4,
            "{profile:?}: {} set helper calls\n{stats:?}",
            stats.array_set_helper_calls
        );
    }
}

#[test]
fn a_sieve_fills_holes_without_the_set_helper() {
    let source = "
        local found = 0
        for _ = 1, 3 do
          local limit = 20000
          local composite = {}
          for i = 2, limit do
            if not composite[i] then
              found = found + 1
              local multiple = i * i
              while multiple <= limit do
                composite[multiple] = true
                multiple = multiple + i
              end
            end
          end
        end
        return found";
    for (profile, result, stats) in agree(source) {
        // 2262 primes below 20000, three times.
        assert!(result.contains("6786"), "{profile:?}: {result}");
        // 64000 stores per round, all made in the trace: the ones inside the
        // array part and the ones that extend it by a hole and a value.
        assert!(
            stats.array_set_helper_calls < 2_000,
            "{profile:?}: {} set helper calls\n{stats:?}",
            stats.array_set_helper_calls
        );
    }
}

#[test]
fn a_hole_below_the_border_successor_fills_in_the_trace_and_the_border_moves_in_the_vm() {
    // Filling the slot after the border moves the border over the slots that
    // follow, which the VM does; the trace fills the others itself. The
    // length shows the border.
    let source = "
        local out = 0
        for round = 1, 40 do
          local t = {}
          for i = 1, 200 do t[i * 2] = true end
          t[1] = true
          for i = 3, 399, 2 do t[i] = true end
          out = out + #t
        end
        local u = {}
        for i = 1, 100 do u[i] = i end
        for i = 1, 100 do u[i] = nil end
        for i = 100, 1, -1 do u[i] = i end
        return out, #u";
    agree_native(source);
}

#[test]
fn a_collector_cycle_and_a_weak_table_do_not_disturb_filled_holes() {
    let source = "
        local weak = setmetatable({}, {__mode = 'v'})
        local plain = {}
        local s = 0
        for i = 1, 6000 do
          plain[i * 2] = true
          weak[i * 2] = true
          if i % 1500 == 0 then collectgarbage() end
        end
        for i = 1, 12000 do
          if plain[i] then s = s + 1 end
          if weak[i] then s = s + 2 end
        end
        for i = 1, 6000 do plain[i * 2 - 1] = i % 2 == 0 end
        return s, #plain > 0, #weak > 0";
    agree_native(source);
}

#[test]
fn a_table_with_newindex_keeps_every_store_in_the_vm() {
    let source = "
        local log = 0
        local t = setmetatable({}, {__newindex = function(t, k, v) log = log + 1; rawset(t, k, v) end})
        for i = 1, 400 do t[i * 2] = true end
        for i = 1, 400 do t[i * 2 - 1] = true end
        local n = 0
        for i = 1, 800 do if t[i] then n = n + 1 end end
        return log, n, #t";
    agree(source);
}
