//! Traces that read and write numeric keys held by a table's hash part.
//!
//! `t[0]`, a negative key, a sparse integer key and a float that equals an
//! integer live in the hash part (or the array part, after a rehash). A trace
//! reads them with a raw lookup over both parts (`TraceOp::HashLoad` with a
//! number key) instead of aborting with `TableMiss`. A nil slot of a table
//! with a metatable still runs `__index` in the interpreter. Every program
//! runs under the interpreter and under tracing execution in every profile,
//! and the results must match exactly.

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

/// Runs `source`, checks the results agree, and that no recording missed the
/// table and loops ran natively.
fn agree_without_miss(source: &str) -> Vec<(CompatibilityProfile, String, TraceStatistics)> {
    let all = agree(source);
    for (profile, _, stats) in &all {
        assert_eq!(
            stats.aborts(TraceAbortReason::TableMiss),
            0,
            "{profile:?}\n{source}\n{stats:?}"
        );
        assert!(
            stats.iterations >= 300,
            "{profile:?}: {} native iterations\n{source}\n{stats:?}",
            stats.iterations
        );
    }
    all
}

#[test]
fn zero_negative_and_sparse_integer_keys_are_read_natively() {
    let source = "
        local t = {}
        for i = 1, 50 do t[i] = i end
        t[0], t[-1], t[1000000] = 7, 11, 13
        local s = 0
        for round = 1, 40 do
          for i = 0, 49 do
            s = s + t[i] + t[-1] + t[1000000]
          end
        end
        return s";
    // 40 * (7 + (1 + .. + 49) + 50 * 24)
    let expected = (40 * (7 + 1225 + 50 * 24)).to_string();
    for (profile, result, stats) in agree_without_miss(source) {
        assert!(result.contains(&expected), "{profile:?}: {result}");
        assert!(stats.hash_helper_calls > 0, "{profile:?}\n{stats:?}");
    }
}

#[test]
fn an_existing_hash_part_numeric_key_is_written() {
    let source = "
        local t = {}
        for i = 1, 20 do t[i] = i end
        t[0], t[-5] = 1, 2
        for i = 1, 3000 do
          t[0] = t[0] + 1
          t[-5] = t[-5] + t[0] % 3
        end
        return t[0], t[-5], t[1], t[20]";
    let all = agree(source);
    for (profile, result, stats) in all {
        assert!(result.contains("3001"), "{profile:?}: {result}");
        assert_eq!(
            stats.aborts(TraceAbortReason::TableMiss),
            0,
            "{profile:?}\n{stats:?}"
        );
        assert!(stats.iterations >= 300, "{profile:?}\n{stats:?}");
    }
}

#[test]
fn a_float_key_equal_to_an_integer_is_the_same_key() {
    let source = "
        local t = {}
        for i = 1, 8 do t[i] = i end
        t[10] = 100
        t[0] = 5
        local a, b, c = 0, 0, 0
        for i = 1, 2000 do
          local f = 10.0
          local z = 0.0
          a = a + t[f]
          b = b + t[10] + t[z]
          c = c + (t[10.5] or 1) + (t[2.0] == t[2] and 1 or 0)
        end
        return a, b, c";
    for (profile, result, _) in agree_without_miss(source) {
        assert!(
            result.contains("200000") && result.contains("210000") && result.contains("4000"),
            "{profile:?}: {result}"
        );
    }
}

#[test]
fn a_key_that_moves_from_the_hash_to_the_array_part_is_still_found() {
    // Key 5 starts in the hash part (the array part has 4 slots), then the
    // table is filled around it, so a rehash moves it into the array part.
    let source = "
        local t = {}
        t[1], t[2], t[3], t[4] = 1, 2, 3, 4
        t[6] = 6
        t[5] = 5
        local s = 0
        for i = 1, 4000 do
          s = s + t[5] + t[6]
          if i == 2000 then
            for k = 7, 64 do t[k] = k end
          end
        end
        return s, t[5], t[64]";
    for (profile, result, _) in agree_without_miss(source) {
        assert!(result.contains("44000"), "{profile:?}: {result}");
    }
}

#[test]
fn a_present_hash_key_of_a_table_with_a_metatable_is_read_natively() {
    // `__index` only runs for an absent key, so a present `t[0]` is read raw.
    let source = "
        local t = setmetatable({}, { __index = function(_, k) return k * 2 end })
        for i = 1, 20 do rawset(t, i, i) end
        rawset(t, 0, 3)
        local s = 0
        for i = 1, 3000 do s = s + t[0] end
        return s";
    for (profile, result, stats) in agree(source) {
        assert!(result.contains("9000"), "{profile:?}: {result}");
        assert_eq!(
            stats.aborts(TraceAbortReason::TableMiss),
            0,
            "{profile:?}\n{stats:?}"
        );
        assert!(stats.traces_compiled > 0, "{profile:?}\n{stats:?}");
        assert!(stats.iterations >= 300, "{profile:?}\n{stats:?}");
    }
}

#[test]
fn a_nil_slot_with_an_index_metamethod_still_runs_it() {
    let source = "
        local t = setmetatable({}, { __index = function(_, k) return k * 2 end })
        for i = 1, 20 do rawset(t, i, i) end
        rawset(t, 0, 3)
        local s = 0
        for i = 1, 3000 do
          s = s + t[0]
          if i > 1500 then s = s + t[-1] end
        end
        return s";
    // 3000 * 3 + 1500 * -2
    for (profile, result, stats) in agree(source) {
        assert!(result.contains("6000"), "{profile:?}: {result}");
        // The recording that meets the absent key leaves it to the
        // interpreter, which runs the metamethod.
        assert!(stats.traces_compiled > 0, "{profile:?}\n{stats:?}");
    }
}

#[test]
fn a_key_that_turns_nil_mid_loop_falls_back_to_the_interpreter() {
    let source = "
        local t = {}
        for i = 1, 30 do t[i] = i end
        t[0] = 4
        local s, nils = 0, 0
        for i = 1, 4000 do
          local v = t[0]
          if v then s = s + v else nils = nils + 1 end
          if i == 2500 then t[0] = nil end
        end
        return s, nils";
    for (profile, result, _) in agree(source) {
        assert!(
            result.contains("10000") && result.contains("1500"),
            "{profile:?}: {result}"
        );
    }
}

#[test]
fn spectral_norm_style_loops_link_their_outer_loop() {
    let source = "
        local n = 100
        local function mul(input, output)
          for i = 0, n - 1 do
            local total = 0
            for j = 0, n - 1 do total = total + input[j] * (i + j + 1) end
            output[i] = total
          end
        end
        local u, v = {}, {}
        for i = 0, n - 1 do u[i] = 1 end
        for _ = 1, 30 do mul(u, v) mul(v, u) for i = 0, n - 1 do u[i] = u[i] / 1000 end end
        return u[0], u[n - 1]";
    for (profile, _, stats) in agree(source) {
        assert_eq!(
            stats.aborts(TraceAbortReason::TableMiss),
            0,
            "{profile:?}\n{stats:?}"
        );
        // Integer loops (Lua 5.3 and later) follow the integer policy for
        // outer sides, which this nest's body does not meet.
        if !matches!(
            profile,
            CompatibilityProfile::Lua51
                | CompatibilityProfile::Lua52
                | CompatibilityProfile::LuaJit21
        ) {
            continue;
        }
        // 60 calls of 100 outer iterations: entering the inner loop from the
        // interpreter on each would take 6000 entries.
        assert!(
            stats.native_entries < 400,
            "{profile:?}: {} native entries\n{stats:?}",
            stats.native_entries
        );
        assert!(stats.outer_sides >= 1, "{profile:?}\n{stats:?}");
        assert!(stats.covered_loops >= 1, "{profile:?}\n{stats:?}");
    }
}
