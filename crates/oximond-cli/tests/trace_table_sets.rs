//! Traces that store keys a table does not hold yet, and read its nil holes.
//!
//! A store into a nil hole, past the end of the array, into the hash part, or
//! of nil is a raw set the trace asks the VM for (`TraceOp::ArraySet`). The
//! VM refuses it, and the trace leaves before the instruction, wherever the
//! interpreter would not store raw: a key that is absent from a table with a
//! metatable, a NaN key. A read of a nil hole of a table without a metatable
//! is nil. Every program runs under the interpreter and under tracing
//! execution in every profile, and the results must match exactly, error
//! messages included. Counters show that the loops really ran natively.

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
    judge_short_loops: bool,
) -> (String, TraceStatistics) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    if !judge_short_loops {
        lua.vm_mut().ignore_remaining_work();
    }
    let results: Vec<Value> = lua
        .exec(source)
        .unwrap_or_else(|error| panic!("{profile:?} {mode:?}: {error}\n{source}"));
    (format!("{results:?}"), lua.vm().jit_statistics().trace)
}

/// Runs `source` in every profile under the interpreter and under tracing
/// execution, checks the results agree, and returns each profile's result
/// and tracing statistics.
///
/// These programs test what traces do with small loops of a main chunk, so
/// the tracer compiles them however little work they have left.
fn agree(source: &str) -> Vec<(CompatibilityProfile, String, TraceStatistics)> {
    agree_judging(source, false)
}

/// [`agree`], where the tracer leaves a loop of the main chunk that cannot
/// repay a compile to the interpreter, as it does by default.
fn agree_top_level(source: &str) -> Vec<(CompatibilityProfile, String, TraceStatistics)> {
    agree_judging(source, true)
}

fn agree_judging(
    source: &str,
    judge_short_loops: bool,
) -> Vec<(CompatibilityProfile, String, TraceStatistics)> {
    ALL_PROFILES
        .iter()
        .map(|&profile| {
            let (expected, _) = run(
                profile,
                ExecutionMode::Interpreter,
                source,
                judge_short_loops,
            );
            let (traced, stats) = run(profile, ExecutionMode::Tracing, source, judge_short_loops);
            assert_eq!(traced, expected, "{profile:?}\n{source}\n{stats:?}");
            (profile, expected, stats)
        })
        .collect()
}

/// [`agree`], requiring at least `iterations` native loop iterations and no
/// recording that missed a table, in every profile.
fn agree_natively(source: &str, iterations: u64) -> Vec<(CompatibilityProfile, String)> {
    agree(source)
        .into_iter()
        .map(|(profile, result, stats)| {
            assert!(
                stats.iterations >= iterations,
                "{profile:?}: {} native iterations, wanted {iterations}\n{source}\n{stats:?}",
                stats.iterations
            );
            assert_eq!(
                stats.aborts(TraceAbortReason::TableMiss),
                0,
                "{profile:?}\n{source}\n{stats:?}"
            );
            (profile, result)
        })
        .collect()
}

#[test]
fn appends_past_the_end_grow_the_array_inside_the_trace() {
    // One append per iteration, then two slots per iteration (a hole and the
    // value, as a table of booleans grows), then reads of what was stored.
    let source = "
        local t = {}
        for i = 1, 3000 do t[i] = i * 2 end
        local flags = {}
        local m = 4
        while m <= 6000 do flags[m] = true; m = m + 2 end
        local s, n = 0, 0
        for i = 1, #t do s = s + t[i] end
        for i = 1, 6000 do if flags[i] then n = n + 1 end end
        return #t, s, t[3000], n, flags[6000], flags[5999]";
    for (_, result) in agree_natively(source, 10_000) {
        assert!(result.contains("3000"), "{result}");
    }
}

#[test]
fn a_sieve_fills_the_holes_of_its_table_and_reads_them_back() {
    let source = "
        local found = 0
        for _ = 1, 3 do
          local limit = 6000
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
    for (_, result) in agree_natively(source, 20_000) {
        // 783 primes below 6000, three times.
        assert!(result.contains("2349"), "{result}");
    }
}

#[test]
fn storing_nil_deletes_present_keys_and_leaves_absent_ones_absent() {
    // A nil value stores through the VM: it deletes a present key (the
    // array's border and `next` see the deletion) and leaves an absent one
    // absent, inside the array and past it.
    let source = "
        local t = {}
        local none = nil
        for i = 1, 2000 do
          t[i] = i
          t[i + 5000] = none
        end
        for i = 3, 2000, 3 do t[i - 1] = none end
        for i = 1, 2000, 7 do t[i] = none; t[i + 9000] = none end
        local n, s = 0, 0
        for k, v in pairs(t) do n = n + 1; s = s + k end
        return n, s, t[2000], t[1999], t[1998], t[7000]";
    agree_natively(source, 1_000);
}

#[test]
fn hash_inserts_grow_and_rehash_the_table_inside_the_trace() {
    // Fractional and negative keys go to the hash part, which grows and
    // rehashes many times while the trace runs. The order `pairs` visits
    // keys in is the table's own, so it must come out the same.
    let source = "
        local t = {}
        for i = 1, 3000 do
          t[i * 7 + 0.5] = i
          t[-i] = i
          t[i] = i
        end
        local n, s = 0, 0
        local order = {}
        for k, v in pairs(t) do
          n = n + 1
          s = s + v
          if n <= 40 then order[#order + 1] = tostring(k) end
        end
        return n, s, t[7.5], t[-3000], t[3000], table.concat(order, ',')";
    agree_natively(source, 2_000);
}

#[test]
fn a_metatable_with_newindex_that_appears_mid_loop_gets_every_absent_key() {
    // From the iteration that sets the metatable on, every key the loop
    // stores is absent, so `__newindex` runs for each, in the interpreter.
    let source = "
        local log = {}
        local t = {}
        local mt = {
          __newindex = function(tbl, k, v) log[#log + 1] = k; rawset(tbl, k, v * 10) end,
        }
        for i = 1, 2000 do
          if i == 1000 then setmetatable(t, mt) end
          t[i] = i
        end
        return #log, log[1], log[#log], t[999], t[1000], t[2000]";
    for (profile, result, stats) in agree(source) {
        assert!(stats.iterations >= 500, "{profile:?}\n{stats:?}");
        assert!(result.contains("1001"), "{profile:?}: {result}");
    }
}

#[test]
fn a_metatable_set_between_runs_sends_absent_keys_to_newindex() {
    // A fill runs natively on a plain table, then on tables with a metatable:
    // `__newindex` must see every absent key, and present keys are stored raw
    // (Lua runs `__newindex` only for absent keys).
    let source = "
        local log = 0
        local mt = {__newindex = function(tbl, k, v) log = log + 1; rawset(tbl, k, -v) end}
        local function fill(t, n) for i = 1, n do t[i] = i end return t end
        local plain = fill({}, 1500)
        local hooked = fill(setmetatable({}, mt), 1500)
        local half = {}
        for i = 1, 750 do half[i] = 0 end
        setmetatable(half, mt)
        fill(half, 1500)
        local indexed = fill(setmetatable({}, {__index = function() return 0 end}), 1500)
        return log, plain[1500], hooked[1], hooked[1500], half[750], half[751], indexed[1500]";
    for (profile, result, stats) in agree(source) {
        assert!(stats.iterations >= 1_000, "{profile:?}\n{stats:?}");
        assert!(result.contains("2250"), "{profile:?}: {result}");
    }
}

#[test]
fn keys_of_other_types_leave_the_trace_and_errors_are_the_interpreters() {
    let cases = [
        // A string key after many number keys.
        "local t = {}
         for i = 1, 2000 do
           local k = i
           if i > 1000 then k = 'k' .. i end
           t[k] = i
         end
         return t[1000], t.k1001, t.k2000, #t",
        // Fractional keys, and integral floats, which are integer keys.
        "local t = {}
         for i = 1, 2000 do t[i + 0.5] = i; t[i * 1.0] = -i end
         local s = 0
         for i = 1, 2000 do s = s + t[i] + t[i + 0.5] end
         return s, t[1.5], t[2000], #t",
        // A NaN key is an error, raised by the interpreter at the store.
        "local function fill(t, n)
           for i = 1, n do
             local k = i
             if i == n then k = 0 / 0 end
             t[k] = i
           end
         end
         local t = {}
         local ok, err = pcall(fill, t, 2000)
         return ok, err, #t",
        // So is a nil key.
        "local function fill(t, n)
           for i = 1, n do
             local k = i
             if i == n then k = nil end
             t[k] = i
           end
         end
         local t = {}
         local ok, err = pcall(fill, t, 2000)
         return ok, err, #t",
    ];
    for source in cases {
        for (profile, _, stats) in agree(source) {
            assert!(stats.iterations >= 500, "{profile:?}\n{source}\n{stats:?}");
        }
    }
}

#[test]
fn two_registers_that_hold_one_growing_table_see_its_new_storage() {
    // A store through one register grows the array; a read through the
    // other register right after it must find the moved storage.
    let source = "
        local a = {}
        local b = a
        local s = 0
        for i = 1, 3000 do
          a[i] = i
          b[i] = b[i] + 1
          s = s + b[i]
        end
        return s, #a, a[3000], b[1]";
    agree_natively(source, 2_000);
}

#[test]
fn a_nil_read_leaves_when_the_table_gains_an_index_metamethod() {
    // The first count reads the holes of a plain table natively. The second
    // runs on the same table once it has an `__index`, which must answer
    // for every hole.
    let source = "
        local t = {}
        for i = 2, 3000, 2 do t[i] = true end
        local function count(t)
          local n = 0
          for i = 1, 3000 do if not t[i] then n = n + 1 end end
          return n
        end
        local a = count(t)
        local b = count(t)
        setmetatable(t, {__index = function(_, k) return k % 3 == 0 end})
        local c = count(t)
        return a, b, c";
    for (profile, result, stats) in agree(source) {
        assert!(stats.iterations >= 3_000, "{profile:?}\n{stats:?}");
        assert!(result.contains("1500"), "{profile:?}: {result}");
        assert!(result.contains("1000"), "{profile:?}: {result}");
    }
}

/// `table/iteration`: a loop of the main chunk fills an array once, then
/// `ipairs` runs over it again and again. The fill does less work than a
/// compile takes, so it is not compiled. The read of an `ipairs` step leaves at
/// the end of the array once per run of the loop; a side recorded there would
/// only abort, so that guard takes none.
#[test]
fn a_one_shot_fill_is_not_compiled_and_the_end_of_ipairs_takes_no_side() {
    let source = "local values = {} \
         for i = 1, 10000 do values[i] = i % 113 end \
         local total = 0 \
         for _ = 1, 40 do for _, value in ipairs(values) do total = total + value end end \
         return total";
    for (profile, _, stats) in agree_top_level(source) {
        assert!(
            stats.recording_aborts[TraceAbortReason::ShortLoop.index()] > 0,
            "{profile:?}: the fill was not judged too short\n{stats:?}"
        );
        assert_eq!(
            stats.traces_compiled, 1,
            "{profile:?}: only the ipairs loop compiles\n{stats:?}"
        );
        assert_eq!(
            stats.side_recordings, 0,
            "{profile:?}: the end of ipairs recorded a side\n{stats:?}"
        );
        assert!(
            stats.iterations >= 40 * 9_000,
            "{profile:?}: the ipairs loop ran natively\n{stats:?}"
        );
    }
}
