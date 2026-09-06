//! Native traces over tables, through compiled Lua source.
//!
//! Every program runs under the interpreter and under tracing execution, in
//! the profiles it makes sense for. The results must match exactly, error messages included, and counters must
//! show that native trace code really ran, because equal output alone cannot
//! tell a trace from a slow path.
//!
//! What these programs hold the engine to:
//!
//! - A store a trace commits is committed once. A guard that fails after it
//!   resumes after it, and the interpreter never repeats it (the plan's
//!   worked recovery case).
//! - A slot that is nil, past the array, a string, a boolean or the other
//!   number type is the interpreter's business: the trace leaves before the
//!   access, so errors and coercions are the interpreter's own.
//! - Tables that alias, equal strings from different allocations, a
//!   metatable that appears between runs, tables that grow or rehash between
//!   runs, slots that change type, weak tables, and a table length all give
//!   the interpreter's answer.

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

const HAS_GLOBAL_OPCODE: [CompatibilityProfile; 2] =
    [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21];

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
    // These programs test what traces do with small loops of a main chunk.
    lua.vm_mut().ignore_remaining_work();
    let results: Vec<Value> = lua
        .exec(source)
        .unwrap_or_else(|error| panic!("{profile:?} {mode:?}: {error}\n{source}"));
    (format!("{results:?}"), lua.vm().jit_statistics().trace)
}

/// Runs `source` under the interpreter and under tracing execution, checks
/// the results agree, and returns the result and the tracing statistics.
fn agree(profile: CompatibilityProfile, source: &str) -> (String, TraceStatistics) {
    let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
    let (tracing, stats) = run(profile, ExecutionMode::Tracing, source);
    assert_eq!(
        tracing, expected,
        "{profile:?} tracing\n{source}\n{stats:?}"
    );
    (expected, stats)
}

/// Like [`agree`] for every profile in `profiles`, requiring native
/// iterations in each. Returns the per-profile results and statistics.
fn agree_natively(
    profiles: &[CompatibilityProfile],
    source: &str,
) -> Vec<(String, TraceStatistics)> {
    profiles
        .iter()
        .map(|profile| {
            let (result, stats) = agree(*profile, source);
            assert!(
                stats.iterations > 0,
                "{profile:?}: no native iteration ran\n{source}\n{stats:?}"
            );
            (result, stats)
        })
        .collect()
}

/// A numeric value of the program, as one literal for every profile.
const FILL: &str = "
    local function fill(t, n)
      for i = 1, n do t[i] = (i % 7) + 1 end
      return t
    end";

#[test]
fn dense_array_reads_and_in_place_overwrites_run_natively() {
    let source = format!(
        "{FILL}
        local function bump(t, n)
          local sum = 0
          for i = 1, n do
            local x = t[i]
            t[i] = x * 2 + 1
            sum = sum + x
          end
          return sum
        end
        local t = fill({{}}, 400)
        local total = 0
        for round = 1, 12 do
          -- keep the numbers small so doubling stays exact
          total = total + bump(t, 400) % 1000
          for i = 1, 400 do t[i] = (i + round) % 5 + 1 end
        end
        return total, t[1], t[400]"
    );
    agree_natively(&ALL_PROFILES, &source);
}

#[test]
fn the_plans_worked_recovery_case_never_replays_a_committed_store() {
    // The trace records the `x > 0` arm over positive numbers. Later the
    // table holds numbers that are not positive: the branch guard fails
    // after the store, and the interpreter resumes in the else arm. A restart
    // at the loop header would load the already incremented slot and store
    // again.
    let source = "
        local function run(t, n)
          local sum, i = 0, 1
          while i <= n do
            local x = t[i]
            t[i] = x + 1
            -- The call keeps the arm impure: a value select would leave
            -- the branch no guard to fail.
            if x > 0 then sum = sum + x else sum = sum + math.abs(x) end
            i = i + 1
          end
          return sum
        end
        local t = {}
        for i = 1, 300 do t[i] = (i % 7) + 1 end
        local total = 0
        for k = 1, 20 do total = total + run(t, 300) end
        t[150] = -9
        t[151] = 0
        t[152] = -1
        for k = 1, 10 do total = total + run(t, 300) end
        return total, t[1], t[149], t[150], t[151], t[152], t[153], t[300]";
    for (result, stats) in agree_natively(&ALL_PROFILES, source) {
        assert!(result.contains("Number") || result.contains("Integer"));
        // The failing branch ended runs that had committed a store.
        assert!(
            stats.exits_of(TraceExitKind::CommittedEffect) > 0,
            "{stats:?}"
        );
    }
}

#[test]
fn receivers_that_alias_are_one_table() {
    let source = "
        local function copy_bump(t, u, n)
          local sum = 0
          for i = 1, n do
            local x = u[i]
            t[i] = x + 1
            sum = sum + u[i]
          end
          return sum
        end
        local a, b = {}, {}
        for i = 1, 200 do a[i] = i % 5 + 1; b[i] = i % 3 + 1 end
        local total = 0
        for r = 1, 12 do
          total = total + copy_bump(a, a, 200)
          total = total + copy_bump(a, b, 200)
          total = total + copy_bump(b, a, 200)
        end
        return total, a[1], a[200], b[1], b[200]";
    agree_natively(&ALL_PROFILES, source);
}

#[test]
fn aliased_receivers_share_one_field_of_a_hashed_table() {
    // More than 16 entries puts the table in its hashed form, where two
    // bindings of one field used to resolve through two `&mut` borrows.
    let source = "
        local function bump(p, q, n)
          local s = 0
          for i = 1, n do
            p.x = p.x + 1
            s = s + q.x
          end
          return s
        end
        local t = {}
        for k = 1, 30 do t['k' .. k] = k end
        t.x = 0.5
        local total = 0
        for r = 1, 12 do total = total + bump(t, t, 100) end
        return total, t.x";
    agree_natively(&ALL_PROFILES, source);
}

#[test]
fn equal_strings_from_different_allocations_name_the_same_field() {
    let source = "
        local function bump(p, n)
          local s = 0
          for i = 1, n do
            p.x = p.x + 1
            s = s + p.x
          end
          return s
        end
        local function made(extra)
          local p = {}
          -- the key is built at run time, not the chunk's constant
          p[string.char(120)] = 0.5
          p[string.rep('y', 1 + extra)] = 1
          return p
        end
        local total = 0
        for r = 1, 40 do
          local p = made(r % 3)
          total = total + bump(p, 100)
          local q = {}
          q['x' .. ''] = 2.5
          total = total + bump(q, 100)
        end
        return total";
    agree_natively(&ALL_PROFILES, source);
}

#[test]
fn a_metatable_that_appears_between_runs_changes_nothing_for_present_slots() {
    let source = "
        local calls, newindex = 0, 0
        local mt = {
          __index = function(t, k) calls = calls + 1; return 100 end,
          __newindex = function(t, k, v) newindex = newindex + 1; rawset(t, k, v) end,
        }
        local function sum(t, n)
          local s = 0
          for i = 1, n do s = s + t[i] end
          return s
        end
        local function overwrite(t, n)
          for i = 1, n do t[i] = t[i] + 1 end
        end
        local t = {}
        for i = 1, 50 do t[i] = i end
        local total = 0
        for r = 1, 40 do total = total + sum(t, 50) end
        setmetatable(t, mt)
        -- present slots: neither metamethod runs
        total = total + sum(t, 50)
        overwrite(t, 50)
        -- missing slots: __index runs once for each
        total = total + sum(t, 60)
        return total, calls, newindex, t[1], t[50], rawget(t, 51)";
    for (result, stats) in agree_natively(&ALL_PROFILES, source) {
        assert!(result.contains("10"), "{result}");
        assert!(stats.iterations > 0);
    }
}

#[test]
fn tables_that_grow_and_rehash_between_runs_are_rebound_at_each_entry() {
    let source = "
        local function churn(t, p, n)
          local s = 0
          for i = 1, n do
            s = s + t[(i % 40) + 1] + p.x
            if i % 97 == 0 then t[#t + 1] = i end
            if i % 53 == 0 then p['k' .. i] = i end
          end
          return s
        end
        local t, p = {}, {x = 1}
        for i = 1, 60 do t[i] = i end
        local total = 0
        for r = 1, 30 do total = total + churn(t, p, 300) end
        local count = 0
        for _ in pairs(p) do count = count + 1 end
        return total, #t, count";
    agree_natively(&ALL_PROFILES, source);
}

#[test]
fn a_slot_that_changes_type_runs_in_the_interpreter_with_its_own_errors() {
    let source = "
        local function sum(t, n)
          local s = 0
          for i = 1, n do s = s + t[i] end
          return s
        end
        local t = {}
        for i = 1, 100 do t[i] = i end
        local out = {}
        for r = 1, 30 do out[1] = sum(t, 100) end
        t[40] = '5'
        out[2] = sum(t, 100)
        t[41] = true
        local ok, message = pcall(sum, t, 100)
        out[3] = tostring(ok) .. ':' .. tostring(message)
        t[41] = 41
        t[42] = nil
        local ok2, message2 = pcall(sum, t, 100)
        out[4] = tostring(ok2) .. ':' .. tostring(message2)
        t[42] = 42
        out[5] = sum(t, 100)
        return out[1], out[2], out[3], out[4], out[5]";
    let outcomes = agree_natively(&ALL_PROFILES, source);
    // The error messages are the interpreter's, with their names.
    assert!(outcomes[0].0.contains("attempt to"), "{}", outcomes[0].0);
}

#[test]
fn a_field_that_changes_type_is_refused_at_entry() {
    let source = "
        local function bump(p, n)
          local s = 0
          for i = 1, n do
            p.x = p.x + 1
            s = s + p.x
          end
          return s
        end
        local p = {x = 1.5}
        local total = 0
        for r = 1, 30 do total = total + bump(p, 100) end
        p.x = '10'
        total = total + bump(p, 100)
        p.x = 2.5
        total = total + bump(p, 100)
        local ok = pcall(bump, {x = true}, 100)
        local ok2 = pcall(bump, {}, 100)
        return total, p.x, ok, ok2";
    agree_natively(&ALL_PROFILES, source);
}

#[test]
fn weak_tables_keep_their_numbers_and_lose_collected_values_between_runs() {
    let source = "
        local weak = setmetatable({}, {__mode = 'v'})
        for i = 1, 100 do weak[i] = i end
        weak[101] = {}
        weak[102] = {}
        local keep = {}
        weak[103] = keep
        local function sum(t, n)
          local s = 0
          for i = 1, n do
            local x = t[i]
            if type(x) == 'number' then s = s + x end
          end
          return s
        end
        local function plain(t, n)
          local s = 0
          for i = 1, 100 do s = s + t[i] end
          return s
        end
        local total = 0
        for r = 1, 30 do total = total + plain(weak, 100) end
        collectgarbage()
        collectgarbage()
        for r = 1, 30 do total = total + plain(weak, 100) end
        total = total + sum(weak, 103)
        return total, weak[101], weak[103] == keep";
    agree_natively(&ALL_PROFILES, source);
}

#[test]
fn the_length_operator_runs_natively_and_respects_len_metamethods() {
    let source = "
        local function loop(t)
          local s, i = 0, 1
          while i <= #t do
            s = s + t[i]
            i = i + 1
          end
          return s
        end
        local t = {}
        for i = 1, 300 do t[i] = i % 9 end
        local total = 0
        for r = 1, 20 do total = total + loop(t) end
        local proxy = setmetatable({}, {__len = function() return 3 end})
        proxy[1], proxy[2], proxy[3], proxy[4] = 1, 2, 3, 4
        total = total + loop(proxy)
        return total, #t, #proxy";
    for profile in ALL_PROFILES {
        let (_, stats) = agree(profile, source);
        assert!(stats.iterations > 0, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_string_a_callee_loads_is_restored_as_that_string() {
    // The callee's key constant belongs to the callee's chunk. A guard that
    // fails after it was loaded must restore the same string, not whatever
    // sits at that index of the loop's own constants.
    let source = "
        local t = {x = 0.5}
        for i = 1, 50 do t[i] = i end
        local function get(t, i)
          local k = 'x'
          local v = t[i]
          if v == nil then v = 0 end
          return v + t[k]
        end
        local acc = 0
        for r = 1, 40 do
          for i = 1, 50 do acc = acc + get(t, i) end
        end
        for i = 45, 60 do acc = acc + get(t, i) end
        return acc";
    for (result, stats) in agree_natively(&ALL_PROFILES, source) {
        assert!(!result.contains("Nil"), "{result}");
        let _ = stats;
    }
}

#[test]
fn boolean_arrays_flip_in_place() {
    let source = "
        local function flip(flags, n)
          local on = 0
          for i = 1, n do
            if flags[i] then on = on + 1 end
            flags[i] = not flags[i]
          end
          return on
        end
        local flags = {}
        for i = 1, 200 do flags[i] = i % 3 == 0 end
        local total = 0
        for r = 1, 30 do total = total + flip(flags, 200) end
        return total, flags[1], flags[3], flags[200]";
    agree_natively(&ALL_PROFILES, source);
}

#[test]
fn a_table_held_in_an_upvalue_or_a_global_is_bound_like_a_local() {
    let upvalue = "
        local data = {}
        for i = 1, 200 do data[i] = i % 11 end
        local function go(n)
          local s = 0
          for i = 1, n do
            s = s + data[i]
            data[i] = data[i] + 1
          end
          return s
        end
        local total = 0
        for r = 1, 30 do total = total + go(200) end
        return total, data[1], data[200]";
    agree_natively(&ALL_PROFILES, upvalue);
    let global = "
        data = {}
        for i = 1, 200 do data[i] = i % 11 end
        function go(n)
          local s = 0
          for i = 1, n do
            s = s + data[i]
            data[i] = data[i] + 1
          end
          return s
        end
        local total = 0
        for r = 1, 30 do total = total + go(200) end
        return total, data[1], data[200]";
    // Only the profiles that read a global with one instruction trace it.
    agree_natively(&HAS_GLOBAL_OPCODE, global);
    for profile in ALL_PROFILES {
        agree(profile, global);
    }
}

#[test]
fn loops_the_tracer_cannot_follow_still_give_the_interpreters_answer() {
    // Appending, sparse stores, string keys from a table, table values from
    // an array, and a table that changes between iterations.
    let source = "
        local function appendix(n)
          local t = {}
          for i = 1, n do t[#t + 1] = i * 2 end
          return t
        end
        local function sparse(n)
          local t = {}
          for i = 1, n do t[i * 7] = i end
          local s = 0
          for k, v in pairs(t) do s = s + v end
          return s
        end
        local function nested(rows, n)
          local s = 0
          for i = 1, n do
            local row = rows[i]
            s = s + row.a + row.b
          end
          return s
        end
        local function hopping(a, b, n)
          local t, s = a, 0
          for i = 1, n do
            s = s + t[1]
            t = (t == a) and b or a
          end
          return s
        end
        local rows = {}
        for i = 1, 80 do rows[i] = {a = i, b = i * 2} end
        local total = 0
        for r = 1, 12 do
          total = total + #appendix(200) + sparse(100) + nested(rows, 80)
          total = total + hopping({1.5}, {2.5}, 100)
        end
        return total";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn integer_and_float_arrays_keep_their_slot_types() {
    let source = "
        local function mix(t, n)
          local s = 0
          for i = 1, n do
            s = s + t[i]
            t[i] = t[i] + 1
          end
          return s
        end
        local ints, floats = {}, {}
        for i = 1, 100 do ints[i] = i; floats[i] = i + 0.5 end
        local total = 0
        for r = 1, 30 do total = total + mix(ints, 100) + mix(floats, 100) end
        return total, ints[1], floats[1], math.type and math.type(ints[1])";
    for profile in [
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let (_, stats) = agree(profile, source);
        assert!(stats.iterations > 0, "{profile:?}: {stats:?}");
    }
}

#[test]
fn values_of_different_types_compare_unequal_and_never_run_eq() {
    // `__eq` only runs for two tables, so a table against a number, nil or a
    // boolean is plain false, and a loop that tests a loaded value against
    // nil traces. Two different tables with `__eq` stay in the interpreter.
    let source = "
        local eqs = 0
        local mt = {__eq = function() eqs = eqs + 1; return true end}
        local a, b = setmetatable({}, mt), setmetatable({}, mt)
        local function scan(t, n, probe)
          local hits, nils = 0, 0
          for i = 1, n do
            local v = t[i]
            if v == nil then nils = nils + 1 end
            if v == true then hits = hits + 1000 end
            if v == probe then hits = hits + 1 end
          end
          return hits, nils
        end
        local t = {}
        for i = 1, 100 do t[i] = i % 4 + 0.5 end
        local h, n, total = 0, 0, 0
        for r = 1, 40 do
          local x, y = scan(t, 100, 2.5)
          total = total + x + y
          local z = scan(t, 100, a)
          total = total + z
        end
        total = total + (a == b and 1 or 0)
        return total, eqs";
    for (result, stats) in agree_natively(&ALL_PROFILES, source) {
        assert!(result.contains("Integer") || result.contains("Number"));
        let _ = stats;
    }
}

#[test]
fn a_string_read_natively_compares_with_a_constant_natively() {
    // The string is read natively (see below), and its comparison with a
    // constant compares the bytes in a helper call, so these record and run
    // natively instead of being refused.
    let cases = [
        "local t = {} for i = 1, 50 do t[i] = 's' end \
         local n = 0 for i = 1, 400 do if t[1] == 's' then n = n + 1 end end return n",
        "local words = {} for i = 1, 50 do words[i] = 'w' end \
         local n = 0 for i = 1, 400 do if words[(i % 50) + 1] == 'w' then n = n + 1 end end return n",
    ];
    for source in cases {
        for (_, stats) in agree_natively(&ALL_PROFILES, source) {
            assert_eq!(
                stats.aborts(TraceAbortReason::UnsupportedOperand),
                0,
                "{source}\n{stats:?}"
            );
        }
    }
}

#[test]
fn stores_of_new_keys_and_reads_of_holes_record_and_run_natively() {
    // What the recorder once refused and now records as a raw set or a nil
    // read (see trace_table_sets.rs for the details of each).
    let cases = [
        // Appending grows the array.
        "local t = {} for i = 1, 400 do t[i] = i end return #t",
        // A key in the hash part.
        "local t = {} for i = 1, 400 do t[i * 3] = i end return t[3]",
        // A nil slot of a table with no metatable reads as nil.
        "local t = {} for i = 1, 50 do t[i] = i end t[25] = nil \
         local s = 0 for i = 1, 400 do s = s + (t[25] or 1) end return s",
        // A store that changes the slot's type.
        "local t = {} for i = 1, 500 do t[i] = true end \
         for i = 1, 400 do t[i] = 5 end return t[1], t[500]",
        // A table read out of a table indexed by number: one lookup call.
        "local rows = {} for i = 1, 50 do rows[i] = {i} end \
         local s = 0 for i = 1, 400 do s = s + rows[(i % 50) + 1][1] end return s",
        // A string read out of a table by number, used by its length.
        "local words = {} for i = 1, 50 do words[i] = 'w' .. i end \
         local n = 0 for i = 1, 400 do n = n + #words[(i % 50) + 1] end return n",
        // A table that changes between iterations is read through its lane.
        "local a, b = {1.5}, {2.5} local t, s = a, 0 \
         for i = 1, 400 do s = s + t[1] if t == a then t = b else t = a end end return s",
        // A field with a string value is a constant of the trace.
        "local p = {name = 'x'} local n = 0 \
         for i = 1, 400 do if p.name == 'x' then n = n + 1 end end return n",
        // The field changes between two runs of the loop: entry refuses
        // the stale constant.
        "local p = {name = 'x'} local n = 0 \
         local function run() for i = 1, 400 do if p.name == 'x' then n = n + 1 end end end \
         run() p.name = 'y' run() p.name = 'x' run() return n",
    ];
    for source in cases {
        for (_, stats) in agree_natively(&ALL_PROFILES, source) {
            assert!(stats.traces_recorded > 0, "{source}\n{stats:?}");
        }
    }
}

#[test]
fn a_metatable_and_a_type_change_that_happen_inside_the_loop() {
    // One loop runs hot on a plain table of numbers. At fixed iterations its
    // rare arms add a metatable, replace a slot with a string, clear a slot
    // and append. Each arm leaves the trace through a guard and runs in the
    // interpreter, and the trace is entered again afterwards with the
    // table's storage looked up afresh.
    let source = "
        local log = {}
        local mt = {
          __index = function(t, k) log[#log + 1] = 'index' .. k; return 1000 end,
          __newindex = function(t, k, v) log[#log + 1] = 'new' .. k; rawset(t, k, v) end,
        }
        local t = {}
        for i = 1, 400 do t[i] = i % 10 end
        local s = 0
        for i = 1, 400 do
          if i == 100 then setmetatable(t, mt) end
          if i == 150 then t[300] = '7' end
          if i == 200 then t[350] = nil end
          if i == 250 then t[401] = 5 end
          local x = t[i]
          s = s + x
          t[i] = (i % 10) + 1
          if i == 300 then t[i] = 'back' end
        end
        return s, #log, log[1], log[2], t[300], t[401], t[1], t[400]";
    for (result, stats) in agree_natively(&ALL_PROFILES, source) {
        assert!(result.contains("index"), "{result}");
        // Every arm ended a native run through a guard.
        assert!(
            stats.exits_of(TraceExitKind::GuardFail)
                + stats.exits_of(TraceExitKind::CommittedEffect)
                >= 3
        );
    }
}

#[test]
fn stores_and_loads_inside_an_inlined_callee_leave_through_the_callees_frames() {
    // The callee stores into a table its caller passed, so the guards of
    // the store and of the load sit inside the callee's frame. A failing
    // guard must rebuild the caller and the callee, with the key string the
    // callee loaded.
    let source = "
        local function put(t, i, v)
          local name = 'count'
          t[i] = v
          t[name] = t[name] + 1
        end
        local function fetch(t, i)
          return t[i]
        end
        local t = {count = 0.5}
        for i = 1, 100 do t[i] = i end
        local s = 0
        for r = 1, 20 do
          for i = 1, 100 do
            put(t, i, i + r)
            s = s + fetch(t, i)
          end
        end
        -- a slot outside the array: the guard fails inside `put`
        put(t, 101, 7)
        local ok, message = pcall(put, {}, 1, 1)
        return s, t.count, t[101], t[100], ok, message";
    agree_natively(&ALL_PROFILES, source);
}

// ------------------------------------------------------ generated programs

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(bound).unwrap()).unwrap()
    }

    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.below(items.len())]
    }
}

/// Builds programs that loop over arrays and fields of tables, mixing the
/// accesses a trace repeats natively with the ones it must leave for
/// (fractional and out-of-range keys, nil, string and boolean slots, type
/// changes, aliasing receivers, metatables). The tables are rebuilt each
/// round with a few slots corrupted at seeded places.
struct TableGenerator {
    rng: Rng,
}

impl TableGenerator {
    /// An index expression into an array of about 40 elements.
    fn index(&mut self) -> String {
        match self.rng.below(9) {
            0..=2 => "i".to_owned(),
            3 => "(i % 37) + 1".to_owned(),
            4 => format!("((i * {}) % 41) + 1", 2 + self.rng.below(5)),
            5 => format!("i + {}", self.rng.below(6)),
            6 => "(i % 45) + 1".to_owned(),
            7 => "i / 2".to_owned(),
            _ => "40 - (i % 40)".to_owned(),
        }
    }

    fn array(&mut self) -> &'static str {
        self.rng.pick(&["a", "a", "b", "c"])
    }

    /// A numeric expression of bounded size.
    fn number(&mut self, depth: usize) -> String {
        let leaf = depth == 0 || self.rng.below(3) == 0;
        if leaf {
            return match self.rng.below(9) {
                0 => "s".to_owned(),
                1 => "k".to_owned(),
                2 => "i".to_owned(),
                3 => format!("{}", self.rng.below(9)),
                4 => format!("{}.5", self.rng.below(5)),
                5 | 6 => format!("{}[{}]", self.array(), self.index()),
                7 => self.rng.pick(&["p.x", "p.y", "q.x"]).to_owned(),
                _ => self.rng.pick(&["#a", "#b", "#c"]).to_owned(),
            };
        }
        let left = self.number(depth - 1);
        let right = self.number(depth - 1);
        let operator = self.rng.pick(&["+", "-", "*", "+", "-"]);
        format!("({left} {operator} {right})")
    }

    fn condition(&mut self) -> String {
        match self.rng.below(6) {
            0 => format!("{} > {}", self.number(1), self.number(1)),
            1 => format!("{} <= {}", self.number(1), self.number(1)),
            2 => format!("i % {} == {}", 2 + self.rng.below(5), self.rng.below(2)),
            3 => "p.flag".to_owned(),
            4 => "not q.flag".to_owned(),
            _ => format!("{}[{}] == nil", self.array(), self.index()),
        }
    }

    fn statement(&mut self, depth: usize) -> String {
        match self.rng.below(if depth == 0 { 8 } else { 6 }) {
            0 | 1 => format!("s = (s + {}) % 100003", self.number(2)),
            2 => format!("k = (k + {}) % 1009", self.number(1)),
            3 => format!(
                "{}[{}] = ({}) % 997",
                self.array(),
                self.index(),
                self.number(2)
            ),
            4 => format!("p.x = (p.x + {}) % 997", self.number(1)),
            5 => self
                .rng
                .pick(&["p.flag = not p.flag", "q.flag = not q.flag"])
                .to_owned(),
            6 => {
                let otherwise = if self.rng.below(2) == 0 {
                    format!(" else {}", self.statement(depth + 1))
                } else {
                    String::new()
                };
                format!(
                    "if {} then {}{otherwise} end",
                    self.condition(),
                    self.statement(depth + 1)
                )
            }
            _ => format!("f[{}] = not f[{}]", self.index(), self.index()),
        }
    }

    fn program(&mut self) -> String {
        let count = 3 + self.rng.below(5);
        let body: Vec<String> = (0..count).map(|_| self.statement(0)).collect();
        let hole = 1 + self.rng.below(40);
        let alias = self.rng.below(3);
        let corruption = self.rng.below(6);
        let metatable = self.rng.below(4) == 0;
        format!(
            "
            local function build(round)
              local a, b, c, f = {{}}, {{}}, {{}}, {{}}
              for i = 1, 40 do
                a[i] = (i * 7 + round) % 13
                b[i] = ((i * 5) % 11) + 0.5
                c[i] = i % 3
                f[i] = i % 2 == 0
              end
              local p = {{x = 1.5, y = 2, flag = true}}
              local q = {{x = 3, flag = false}}
              if round % 3 == {corruption} % 3 then
                local hole = (round * {hole}) % 40 + 1
                if round % 2 == 0 then a[hole] = nil else a[hole] = 'z' end
              end
              if round % 4 == 3 then b[7] = true end
              if {alias} == 1 then c = a elseif {alias} == 2 then b = a end
              if {metatable} then
                setmetatable(a, {{__index = function(t, k) return 4 end}})
              end
              return a, b, c, f, p, q
            end
            local function run(a, b, c, f, p, q, n)
              local s, k = 0, 0
              for i = 1, n do
                {body}
              end
              return s, k, p.x
            end
            local out = {{}}
            for round = 1, 9 do
              local a, b, c, f, p, q = build(round)
              local ok, s, k, x = pcall(run, a, b, c, f, p, q, 120)
              out[#out + 1] = tostring(ok) .. ':' .. tostring(s) .. ':' .. tostring(k) .. ':' .. tostring(x)
              local sum = 0
              for i = 1, 40 do
                if type(a[i]) == 'number' then sum = sum + a[i] end
                if type(b[i]) == 'number' then sum = sum + b[i] end
                if f[i] then sum = sum + 1 end
              end
              out[#out + 1] = sum
            end
            return table.concat(out, ',')",
            body = body.join("\n                ")
        )
    }
}

#[test]
fn generated_table_loops_match_the_interpreter_and_run_natively() {
    let mut native_runs = 0;
    let mut runs = 0;
    let mut iterations = 0;
    let mut table_exits = 0;
    for seed in 1..=120_u64 {
        let mut generator = TableGenerator {
            rng: Rng(0x9e37_79b9_7f4a_7c15 ^ seed.wrapping_mul(0x1000_0000_01b3) | 1),
        };
        let source = generator.program();
        for profile in [
            CompatibilityProfile::Lua51,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::LuaJit21,
        ] {
            let (expected, _) = run(profile, ExecutionMode::Interpreter, &source);
            let (tracing, stats) = run(profile, ExecutionMode::Tracing, &source);
            assert_eq!(
                tracing, expected,
                "seed {seed} {profile:?} tracing\n{source}\n{stats:?}"
            );
            runs += 1;
            native_runs += usize::from(stats.iterations > 0);
            iterations += stats.iterations;
            table_exits += stats.side_exits_of(oximond::vm::TraceGuardKind::ArrayBounds);
        }
    }
    assert!(
        native_runs * 3 > runs,
        "{native_runs} of {runs} runs ran natively"
    );
    assert!(iterations > 50_000, "{iterations} native iterations");
    assert!(table_exits > 100, "{table_exits} array guard exits");
}

#[test]
fn a_loop_over_integer_arrays_and_then_float_arrays_gets_a_version_for_each() {
    // The first call reads integers, the later ones floats, as the unit
    // tables of a numeric program do. A trace that guarded the wrong type at
    // every access would leave at every iteration: entry tells the versions
    // apart by the type of slot 1.
    let source = "
        local function sum(t, n)
          local s = 0
          for i = 1, n do s = s + t[i] end
          return s
        end
        local ints, floats = {}, {}
        for i = 1, 120 do ints[i] = i; floats[i] = i + 0.5 end
        local total = 0
        for r = 1, 60 do
          total = total + sum(ints, 120)
          total = total + sum(floats, 120)
        end
        return total";
    for profile in [
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let (_, stats) = agree(profile, source);
        // Both element types ran natively: more than half of the 14400
        // iterations of each, and few exits.
        assert!(stats.iterations > 10_000, "{profile:?}: {stats:?}");
        assert!(stats.version_installs >= 1, "{profile:?}: {stats:?}");
        assert!(
            stats.side_exits_of(oximond::vm::TraceGuardKind::ArrayBounds) < 400,
            "{profile:?}: {stats:?}"
        );
    }
}

#[test]
fn a_loop_from_zero_over_an_array_records_despite_the_hash_resident_first_key() {
    // `t[0]` lives in the hash part, so a recording that starts on the
    // iteration with key 0 cannot repeat its access. Another iteration can,
    // so the recording is retried instead of giving the
    // loop up.
    let source = "
        local function sum(t, n)
          local s = 0.5
          for j = 0, n - 1 do s = s + t[j] end
          return s
        end
        local t = {}
        for i = 0, 119 do t[i] = i + 0.5 end
        local total = 0.0
        for r = 1, 300 do total = total + sum(t, 120) end
        return total";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
    let (_, stats) = run(CompatibilityProfile::Lua54, ExecutionMode::Tracing, source);
    assert!(
        stats.iterations > 20_000,
        "the loop never settled into a trace: {stats:?}"
    );
}
