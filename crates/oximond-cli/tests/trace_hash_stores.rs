//! A store under a string key known only at run time (`values[keys[i]] = n`)
//! traces natively. An overwrite of a present key with a number or boolean of
//! the type it holds is made in the emitted code, through the string index of
//! the table that a lookup probes, when the trace moves no hash storage. A
//! deletion and the reinsertion of a deleted key are native too; other
//! insertions call into the VM, which makes one probe each.
//!
//! Every program runs under the interpreter and under tracing execution in
//! every profile, and the results must match. Counters show which stores
//! native code made and which it left to the VM's helpers.

use oximond::vm::TraceStatistics;
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

/// The statistics of a tracing run of `source` in `profile`.
fn stats_of(profile: CompatibilityProfile, source: &str) -> TraceStatistics {
    run(profile, ExecutionMode::Tracing, source).1
}

/// Every profile's results must hold no failed check: the programs below
/// return the comparisons they make.
fn assert_all_true(results: &[(CompatibilityProfile, String, TraceStatistics)]) {
    for (profile, result, stats) in results {
        assert!(
            !result.contains("false"),
            "{profile:?}: {result}\n{stats:?}"
        );
        assert!(result.contains("true"), "{profile:?}: {result}");
    }
}

/// `keys` string keys holding the number `0`, in `values`, with the names in
/// `names` (both are arrays of the same strings).
fn prelude(keys: usize) -> String {
    format!(
        "local names, values = {{}}, {{}}
         for i = 1, {keys} do
           local k = 'key' .. i
           names[i] = k
           values[k] = 0
         end"
    )
}

/// `values` holds `keys` strings, each with the number `0`; the loop
/// overwrites them `rounds * keys` times.
fn overwrite_source(keys: usize, rounds: usize) -> String {
    format!(
        "{}
         for round = 1, {rounds} do
           for i = 1, #names do values[names[i]] = i % 101 end
         end
         local total = 0
         for i = 1, #names do total = total + values[names[i]] end
         return total",
        prelude(keys)
    )
}

/// The same table, its keys deleted and stored again in every round.
fn churn_source(keys: usize, rounds: usize) -> String {
    format!(
        "{}
         for round = 1, {rounds} do
           for i = 1, #names do
             local k = names[i]
             values[k] = nil
             values[k] = i % 101
           end
         end
         local total, count = 0, 0
         for k, v in pairs(values) do total = total + v; count = count + 1 end
         return total, count, next(values) ~= nil",
        prelude(keys)
    )
}

#[test]
fn a_hot_overwrite_loop_makes_no_helper_call_after_warmup() {
    // Tables of every representation: ordered (12 keys), hashed with few
    // keys, hashed with many. After the first entries, more rounds add
    // iterations and no call of the helper that overwrites.
    for profile in ALL_PROFILES {
        for keys in [12, 40, 3000] {
            let few = stats_of(profile, &overwrite_source(keys, 20));
            let many = stats_of(profile, &overwrite_source(keys, 120));
            assert!(
                many.iterations > few.iterations + 50 * keys as u64,
                "{profile:?} {keys}: {} then {} iterations",
                few.iterations,
                many.iterations
            );
            assert_eq!(
                many.set_present_helper_calls, few.set_present_helper_calls,
                "{profile:?} {keys} keys: the hot loop still calls the helper"
            );
            // Even the first run needs none: a trace's first entry indexes
            // the table at once.
            assert_eq!(
                many.set_present_helper_calls, 0,
                "{profile:?} {keys} keys: {many:?}"
            );
            assert_eq!(many.set_string_helper_calls, few.set_string_helper_calls);
        }
    }
}

#[test]
fn a_loop_recorded_while_its_table_fills_gets_a_version_that_overwrites_natively() {
    // The keys are absent while the trace records (in the profiles that
    // record early), so its store adds them through the general helper. Once whole runs only overwrite, the header
    // records another version, whose stores are native.
    let source = "local names, values = {}, {}
         for i = 1, 64 do names[i] = 'key' .. i end
         for i = 1, 1200000 do values[names[(i - 1) % 64 + 1]] = i % 101 end
         local total = 0
         for i = 1, 64 do total = total + values[names[i]] end
         return total";
    for (profile, _, stats) in agree(source) {
        // A profile whose trace records after the table filled needs none.
        assert!(
            stats.version_installs >= 1 || stats.set_string_helper_calls == 0,
            "{profile:?}: {stats:?}"
        );
        assert!(
            stats.set_string_helper_calls < stats.iterations / 2,
            "{profile:?}: {stats:?}"
        );
    }
}

#[test]
fn deleting_and_storing_again_runs_natively_in_a_hashed_table() {
    for profile in ALL_PROFILES {
        for keys in [100, 3000] {
            let few = stats_of(profile, &churn_source(keys, 20));
            let many = stats_of(profile, &churn_source(keys, 120));
            assert!(many.iterations > few.iterations + 50 * keys as u64);
            for (name, few, many) in [
                (
                    "delete",
                    few.delete_string_helper_calls,
                    many.delete_string_helper_calls,
                ),
                (
                    "revive",
                    few.set_revive_helper_calls,
                    many.set_revive_helper_calls,
                ),
                (
                    "set",
                    few.set_string_helper_calls,
                    many.set_string_helper_calls,
                ),
            ] {
                assert_eq!(many, few, "{profile:?} {keys} keys: {name} still calls");
            }
            assert_eq!(many.delete_string_helper_calls, 0, "{profile:?} {keys}");
            assert_eq!(many.set_revive_helper_calls, 0, "{profile:?} {keys}");
        }
    }
}

#[test]
fn native_stores_leave_the_table_as_the_interpreter_does() {
    // The entry count, the traversal and the values after a loop of
    // overwrites, deletes and stores of deleted keys are the interpreter's.
    for keys in [12, 100, 3000] {
        let source = format!(
            "{}
             for round = 1, 8 do
               for i = 1, #names do
                 local k = names[i]
                 if i % 3 == 0 then values[k] = nil
                 elseif i % 3 == 1 then values[k] = i + round
                 else values[k] = nil; values[k] = i * round end
               end
             end
             local order = {{}}
             for k, v in pairs(values) do order[#order + 1] = k .. '=' .. tostring(v) end
             return table.concat(order, ','), #order",
            prelude(keys)
        );
        for (profile, result, stats) in agree(&source) {
            assert!(result.contains("key"), "{profile:?}: {result}\n{stats:?}");
        }
    }
}

#[test]
fn every_key_deleted_natively_leaves_an_empty_table_that_fills_again() {
    for keys in [100, 3000] {
        let source = format!(
            "{}
             for round = 1, 4 do
               for i = 1, #names do values[names[i]] = nil end
               local empty = next(values) == nil
               for i = 1, #names do values[names[i]] = i end
               local count = 0
               for k, v in pairs(values) do count = count + 1 end
               if not (empty and count == #names) then return false, round end
             end
             return true",
            prelude(keys)
        );
        let results = agree(&source);
        assert_all_true(&results);
        for (profile, _, stats) in &results {
            assert!(stats.iterations > 3 * keys as u64, "{profile:?}: {stats:?}");
            assert_eq!(stats.set_revive_helper_calls, 0, "{profile:?}: {stats:?}");
        }
    }
}

#[test]
fn a_key_that_may_name_a_metamethod_goes_to_the_vm() {
    // The table is a metatable whose `__index` the loop deletes and stores
    // again among a hundred other keys. Native code leaves the keys that start
    // with two underscores to the helpers, which leave the trace, so the
    // object sees `__index` missing for exactly the reads after the deletes.
    let source = "
        local base = { inherited = 'yes' }
        local store = {}
        local names = {}
        for i = 1, 100 do names[i] = 'key' .. i; store[names[i]] = i end
        names[101] = '__index'
        store.__index = base
        local obj = setmetatable({}, store)
        local misses, hits = 0, 0
        for round = 1, 40 do
          for i = 1, #names do
            local k = names[i]
            store[k] = nil
            if obj.inherited == nil then misses = misses + 1 else hits = hits + 1 end
            if k == '__index' then store[k] = base else store[k] = i end
          end
        end
        return misses == 40, hits == 40 * 100, store.__index == base";
    assert_all_true(&agree(source));
    // A number stored under `__newindex` is the same case for the revive.
    let source = "
        local store, names = {}, {}
        for i = 1, 100 do names[i] = 'key' .. i; store[names[i]] = i end
        names[101] = '__newindex'
        store.__newindex = 1
        local obj = setmetatable({}, store)
        local errors = 0
        for round = 1, 30 do
          for i = 1, #names do
            local k = names[i]
            store[k] = nil
            if not pcall(function() obj.field = 1 end) then errors = errors + 1 end
            rawset(obj, 'field', nil)
            store[k] = i
          end
        end
        return errors == 30 * 101 - 30, store.__newindex == 101";
    assert_all_true(&agree(source));
}

/// A table of `keys` plain keys and the keys in `special`, all holding
/// numbers, whose keys the loop deletes and stores again `rounds` times.
fn special_keys_source(special: &[&str], rounds: usize) -> String {
    let list = special
        .iter()
        .map(|name| format!("'{name}'"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{}
         for _, name in ipairs({{ {list} }}) do
           names[#names + 1] = name
           values[name] = 1
         end
         for round = 1, {rounds} do
           for i = 1, #names do
             local k = names[i]
             values[k] = nil
             values[k] = round
           end
         end
         local total, count = 0, 0
         for k, v in pairs(values) do total = total + v; count = count + 1 end
         return total, count",
        prelude(100)
    )
}

#[test]
fn native_deletes_and_revives_leave_metamethod_names_to_the_helpers() {
    // Every round deletes and stores `__index` and `__foo`: a key that starts
    // with two underscores may be a handler a trace assumed absent, so native
    // code never takes it. The helpers decline it too, the trace leaves, and
    // the interpreter makes the store: the calls grow with the rounds, and
    // the table ends as the interpreter leaves it.
    for profile in ALL_PROFILES {
        let source = |rounds| special_keys_source(&["__index", "__foo"], rounds);
        let few = stats_of(profile, &source(20));
        let many = stats_of(profile, &source(120));
        assert!(
            many.delete_string_helper_calls >= few.delete_string_helper_calls + 100,
            "{profile:?}: {few:?} {many:?}"
        );
        assert!(many.iterations > few.iterations, "{profile:?}");
        agree(&source(120));
        // The plain names beside them are stored natively: the calls are the
        // two special keys' and not the hundred others'.
        assert!(
            many.delete_string_helper_calls < 120 * 4,
            "{profile:?}: {many:?}"
        );
    }
}

#[test]
fn a_store_under_a_key_a_binding_fixes_misses_to_the_vm() {
    // `holder.f` is read as a function the trace calls, so the run pins the
    // key `f`: the inner loop deletes it among others and the read after that
    // sees nil. A run that pins a key leaves every delete to the helper, which
    // compares the key. Each round puts `f` back before the loop runs again.
    let source = "
        local function inner(holder, names, back)
          local calls, absent = 0, 0
          for i = 1, #names do
            local k = names[i]
            holder[k] = nil
            local g = holder.f
            if g then calls = calls + g() else absent = absent + 1 end
            holder[k] = back[i]
          end
          return calls, absent
        end
        local f0 = function() return 1 end
        local holder, names, back = { f = f0 }, {}, {}
        for i = 1, 1000 do names[i] = 'key' .. i; holder[names[i]] = i; back[i] = i end
        names[1001] = 'f'
        local calls, absent = 0, 0
        for round = 1, 30 do
          holder.f = f0
          local c, a = inner(holder, names, back)
          calls, absent = calls + c, absent + a
        end
        return calls == 30 * 1000, absent == 30, holder.f == nil";
    let results = agree(source);
    assert_all_true(&results);
    for (profile, _, stats) in &results {
        assert!(
            stats.delete_string_helper_calls > 1000,
            "{profile:?}: the deletes did not go through the helper\n{stats:?}"
        );
    }
}

#[test]
fn a_loop_that_deletes_a_global_through_the_table_that_holds_it_sees_it_gone() {
    // The loop reads the global `limit` and deletes and stores again the keys
    // of `_G`, `limit` among them: a delete native code made behind the read
    // of an earlier iteration would leave that read standing.
    let source = "
        limit = 7
        local names = {}
        for i = 1, 100 do names[i] = 'extra' .. i; _G[names[i]] = i end
        names[101] = 'limit'
        local function body(env, names)
          local total = 0
          for round = 1, 30 do
            for i = 1, #names do
              local k = names[i]
              env[k] = nil
              total = total + limit
              env[k] = 7
            end
          end
          return total
        end
        local ok, total = pcall(body, _G, names)
        return ok == false, type(total) == 'string', limit == nil";
    assert_all_true(&agree(source));
}

#[test]
fn a_read_only_table_still_refuses_every_native_store() {
    let setup = "
        names = {}
        function fill(t) for i = 1, 3000 do t[names[i]] = i end end
        function churn(t)
          for i = 1, #names do t[names[i]] = nil; t[names[i]] = i + 1 end
        end
        function bump(t) for i = 1, #names do t[names[i]] = i + 2 end end
        for i = 1, 3000 do names[i] = 'key' .. i end
        guarded = {}
        fill(guarded)
        for round = 1, 20 do churn(guarded); bump(guarded) end";
    let check = "
        local deleted = pcall(churn, guarded)
        local bumped = pcall(bump, guarded)
        return deleted, bumped, guarded.key1, guarded.key3000";
    for profile in ALL_PROFILES {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            lua.vm_mut().use_synchronous_jit_compilation();
            lua.vm_mut().ignore_remaining_work();
            let _: Vec<Value> = lua.exec(setup).unwrap();
            let Value::Table(table) = lua.vm().runtime().get_global("guarded") else {
                panic!("a table")
            };
            let writable: Vec<Value> = lua.exec(check).unwrap();
            lua.vm_mut()
                .runtime_mut()
                .table_set_readonly(table, true)
                .unwrap();
            let refused: Vec<Value> = lua.exec(check).unwrap();
            // Both pcalls fail; the values are those the last writable run
            // left.
            assert_eq!(refused[0], Value::Boolean(false), "{profile:?} {mode:?}");
            assert_eq!(refused[1], Value::Boolean(false), "{profile:?} {mode:?}");
            assert_eq!(refused[2], writable[2], "{profile:?} {mode:?}");
            assert_eq!(refused[3], writable[3], "{profile:?} {mode:?}");
            assert_eq!(writable[0], Value::Boolean(true), "{profile:?} {mode:?}");
        }
    }
}

#[test]
fn stores_into_a_table_other_tables_cached_reads_depend_on_are_the_vms() {
    // `hop` is a link of the `__index` chain behind `leaf`: the read of
    // `leaf.inherited` before the loop caches the chain and watches `hop`. The
    // loop deletes and stores again the keys of `hop`, and then the key that
    // `leaf` reads is stored in it: the cached read must follow. A watched
    // table takes no native store: its deletes are the helper's.
    let source = "
        local far = { inherited = 'far' }
        local hop = setmetatable({}, { __index = far })
        local leaf = setmetatable({}, { __index = hop })
        local names = {}
        for i = 1, 1000 do names[i] = 'key' .. i; hop[names[i]] = i end
        local first = leaf.inherited
        for round = 1, 30 do
          for i = 1, #names do
            local k = names[i]
            hop[k] = nil
            hop[k] = i
          end
        end
        local second = leaf.inherited
        hop.inherited = 'near'
        return first == 'far', second == 'far', leaf.inherited == 'near'";
    let results = agree(source);
    assert_all_true(&results);
    for (profile, _, stats) in &results {
        assert!(
            stats.delete_string_helper_calls >= 29 * 1000,
            "{profile:?}: a watched table was deleted from natively\n{stats:?}"
        );
    }
}

#[test]
fn a_traversal_survives_the_deletes_a_native_loop_makes_beside_it() {
    let source = format!(
        "{}
         local visited, deleted = 0, 0
         for k, v in pairs(values) do
           visited = visited + 1
           local victim = names[(visited * 7) % #names + 1]
           if values[victim] ~= nil then deleted = deleted + 1 end
           values[victim] = nil
         end
         local left = 0
         for k in pairs(values) do left = left + 1 end
         return visited, deleted, left, visited + left >= #names",
        prelude(3000)
    );
    for (profile, result, stats) in agree(&source) {
        assert!(result.contains("true"), "{profile:?}: {result}\n{stats:?}");
    }
}

#[test]
fn a_key_stored_again_keeps_its_place_in_the_traversal() {
    let source = format!(
        "{}
         local function order()
           local out = {{}}
           for k in pairs(values) do out[#out + 1] = k end
           return table.concat(out, ',')
         end
         local before = order()
         for i = 1, #names, 2 do
           local k = names[i]
           values[k] = nil
           values[k] = 5
         end
         local same = order() == before
         for i = 1, #names, 3 do values[names[i]] = nil end
         local without = order()
         for i = 1, #names, 3 do values[names[i]] = 6 end
         return same, without ~= before, order() == before",
        prelude(300)
    );
    let results = agree(&source);
    assert_all_true(&results);
}

#[test]
fn a_fresh_table_stored_into_an_old_table_survives_an_incremental_cycle() {
    // The table is old before the loop, and every iteration stores a new table
    // under an existing key (the store the barrier covers), with collector
    // steps between the rounds. Every table must be there afterwards.
    let source = format!(
        "{}
         collectgarbage(); collectgarbage()
         for round = 1, 30 do
           for i = 1, #names do values[names[i]] = {{ i, round }} end
           collectgarbage('step', 4)
         end
         collectgarbage()
         local ok = true
         for i = 1, #names do
           local held = values[names[i]]
           if type(held) ~= 'table' or held[1] ~= i or held[2] ~= 30 then ok = false end
         end
         return ok",
        prelude(300)
    );
    let results = agree(&source);
    assert_all_true(&results);
}

#[test]
fn a_store_that_changes_the_type_or_adds_a_key_is_the_vms() {
    // Overwrites with another type, stores of new keys and of strings under
    // present keys are not native: they still happen, in order.
    let source = format!(
        "{}
         for round = 1, 20 do
           for i = 1, #names do
             local k = names[i]
             if round % 3 == 0 then values[k] = tostring(i)
             elseif round % 3 == 1 then values[k] = i + 0.5
             else values[k] = i % 2 == 0 end
           end
           values['extra' .. round] = round
         end
         local total, strings, booleans = 0, 0, 0
         for k, v in pairs(values) do
           if type(v) == 'number' then total = total + v
           elseif type(v) == 'string' then strings = strings + 1
           else booleans = booleans + 1 end
         end
         return total, strings, booleans",
        prelude(100)
    );
    agree(&source);
}

/// Delete and store again every key, `rounds` times. `before` runs before
/// the loop in the round `trigger` (0 for never) and `inner` runs inside it.
fn churn_with(setup: &str, trigger: usize, branch: &str, inner: &str, rounds: usize) -> String {
    format!(
        "{}
         {setup}
         for round = 1, {rounds} do
           if round == {trigger} then {branch} end
           for i = 1, #names do
             local k = names[i]
             values[k] = nil
             values[k] = i % 101
             {inner}
           end
         end
         local total, count = 0, 0
         for k, v in pairs(values) do total = total + v; count = count + 1 end
         return total, count, next(values) ~= nil, hits",
        prelude(300)
    )
}

#[test]
fn a_metatable_added_inside_a_natively_storing_loop_sees_the_revives() {
    // From round 40 on, the revive of a deleted key is a store of an absent
    // key and must reach __newindex. Native revives check no metatable
    // inside the run, so the trace must leave them to the VM once it exists.
    let setup = "local hits = 0
                 local function count(t, k, v) hits = hits + 1; rawset(t, k, v) end";
    let branch = "setmetatable(values, {__newindex = count})";
    let source = churn_with(setup, 40, branch, "", 80);
    for (profile, result, stats) in agree(&source) {
        // 300 keys revived in each of the 41 rounds from the 40th on.
        assert!(result.contains("12300"), "{profile:?}: {result}\n{stats:?}");
    }
    // Without the branch a hot run stays native.
    let hot = churn_with(setup, 0, branch, "", 80);
    for profile in ALL_PROFILES {
        let stats = stats_of(profile, &hot);
        assert_eq!(stats.set_revive_helper_calls, 0, "{profile:?}: {stats:?}");
        assert_eq!(
            stats.delete_string_helper_calls, 0,
            "{profile:?}: {stats:?}"
        );
    }
}

#[test]
fn weak_tables_delete_and_revive_through_the_vm_and_agree() {
    for mode in ["k", "v"] {
        let setup = format!("values = setmetatable(values, {{__mode = '{mode}'}}); local hits = 0");
        let source = churn_with(&setup, 0, "", "", 80);
        for (profile, result, stats) in agree(&source) {
            assert!(result.contains("300"), "{profile:?} {mode}: {result}");
            // A weak table has a metatable, so the trace's binding of a
            // table without one fails and the stores are the VM's.
            assert!(
                stats.delete_string_helper_calls > 0,
                "{profile:?} {mode}: {stats:?}"
            );
            assert!(
                stats.set_string_helper_calls > 0,
                "{profile:?} {mode}: {stats:?}"
            );
        }
    }
}

#[test]
fn a_collector_step_inside_a_natively_churning_loop_loses_nothing() {
    // Steps and allocation run while native deletes and revives happen.
    let setup = "local hits = 0; collectgarbage(); collectgarbage()";
    let inner =
        "if i % 50 == 0 then collectgarbage('step', 2); local t = {i, round}; hits = hits + #t end";
    let source = churn_with(setup, 0, "", inner, 60);
    for (profile, result, stats) in agree(&source) {
        assert!(result.contains("300"), "{profile:?}: {result}\n{stats:?}");
    }
    let after = format!(
        "{}
         {setup}
         for round = 1, 60 do
           for i = 1, #names do
             local k = names[i]
             values[k] = nil
             values[k] = i % 101
             if i % 50 == 0 then collectgarbage('step', 2); local t = {{i}} end
           end
         end
         collectgarbage()
         local n, total = 0, 0
         for k, v in pairs(values) do n = n + 1; total = total + v end
         local firstk = next(values)
         return n == 300, firstk ~= nil, total > 0",
        prelude(300)
    );
    assert_all_true(&agree(&after));
}

#[test]
fn a_loop_that_churns_and_adds_a_key_leaves_the_native_index() {
    // New keys grow the table (rehashes) between native deletes and revives.
    let setup = "local hits = 0";
    let inner = "if i % 10 == 0 then values['new' .. round .. '_' .. i] = i; hits = hits + 1 end";
    let source = churn_with(setup, 0, "", inner, 40);
    for (profile, result, stats) in agree(&source) {
        assert!(result.contains("1200"), "{profile:?}: {result}\n{stats:?}");
    }
}

/// A proxy whose `__newindex` handler stores into `backing`, a table the
/// handler holds as an upvalue, under `keys[i]`. The handler is inlined into
/// the loop, so the store is a string-key store into an entry input.
fn proxy_source(setup: &str, inner: &str, rounds: usize) -> String {
    proxy_value_source(setup, inner, "i % 101", rounds)
}

/// [`proxy_source`] with the stored value `value` (an expression in `i`). The
/// result adds how many keys the proxy itself holds, and `handled`, which
/// the setup and the loop body may count.
fn proxy_value_source(setup: &str, inner: &str, value: &str, rounds: usize) -> String {
    format!(
        "local keys, backing = {{}}, {{}}
         for i = 1, 64 do keys[i] = 'item_' .. i end
         local proxy = setmetatable({{}}, {{__newindex = function(_, key, value)
           backing[key] = value
         end}})
         local handled = 0
         {setup}
         for i = 1, {rounds} do
           {inner}
           proxy[keys[(i - 1) % 64 + 1]] = {value}
         end
         local total, count, raw = 0, 0, 0
         for k, v in pairs(backing) do total = total + (tonumber(v) or (v == true and 1 or 0)); count = count + 1 end
         for k, v in pairs(proxy) do raw = raw + 1 end
         return total, count, rawget(proxy, 'item_1'), raw, handled"
    )
}

#[test]
fn a_store_into_a_table_filling_at_the_start_matches_the_interpreter() {
    // The trace is recorded while `backing` still lacks the keys, so its
    // stores are the set-string helper's; the results are the interpreter's.
    let source = proxy_source("", "", 200_000);
    for (profile, result, stats) in agree(&source) {
        assert!(result.contains("64"), "{profile:?}: {result}\n{stats:?}");
    }
}

#[test]
fn a_store_into_a_loaded_receiver_that_gains_a_newindex_mid_loop_reaches_it() {
    // From round 100000 on, `backing` has a __newindex that doubles what it
    // is given for the keys it lacks. Every key is present by then, so the
    // overwrites stay raw, and the new key the loop adds afterwards reaches
    // the handler.
    let setup = "local hits = 0";
    let inner = "if i == 100000 then
                   setmetatable(backing, {__newindex = function(t, k, v) hits = hits + 1; rawset(t, k, v * 2) end})
                   keys[#keys + 1] = 'late'
                 end
                 if i > 100000 and i % 50000 == 0 then proxy[keys[65]] = 7 end";
    let source = proxy_source(setup, inner, 200_000);
    for (profile, result, stats) in agree(&source) {
        assert!(result.contains("65"), "{profile:?}: {result}\n{stats:?}");
    }
}

#[test]
fn a_store_into_a_receiver_the_upvalue_swaps_for_another_table_matches() {
    // The backing table is swapped for another one mid-loop; the trace's
    // binding of the old one must not take the store.
    let setup = "local other = {}";
    let inner = "if i == 100000 then
                   for k, v in pairs(backing) do other[k] = v end
                   backing = other
                 end";
    let source = proxy_source(setup, inner, 200_000);
    for (profile, result, stats) in agree(&source) {
        assert!(result.contains("64"), "{profile:?}: {result}\n{stats:?}");
    }
}

#[test]
fn a_store_into_a_table_filling_at_the_start_leaves_the_helper_after_the_fill() {
    // The trace is recorded while `backing` lacks every key. Once the keys are
    // there, the policy records the version that stores in place, so the
    // helper serves only the entries that watch for it, not the whole loop.
    let source = proxy_source("", "", 400_000);
    for (profile, _, stats) in agree(&source) {
        assert!(
            stats.set_string_helper_calls <= 20_000,
            "{profile:?}: {stats:?}"
        );
        assert!(stats.iterations > 300_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_rawset_on_the_proxy_mid_loop_stops_its_newindex() {
    // From round 100000 on, the proxy holds `item_1` itself, so stores to it
    // stay on the proxy and the handler never sees that key again.
    let inner = "if i == 100000 then rawset(proxy, keys[1], -1) end";
    let source = proxy_source("", inner, 200_000);
    for (profile, result, stats) in agree(&source) {
        assert!(!result.contains("Nil"), "{profile:?}: {result}\n{stats:?}");
    }
}

#[test]
fn a_rawset_then_a_deletion_on_the_proxy_hands_the_key_back_to_newindex() {
    let inner = "if i == 60000 then rawset(proxy, keys[3], 5) end
                 if i == 120000 then rawset(proxy, keys[3], nil) end";
    let source = proxy_source("", inner, 200_000);
    agree(&source);
}

#[test]
fn a_replaced_newindex_mid_loop_takes_over() {
    let setup = "local meta = getmetatable(proxy)";
    let inner = "if i == 100000 then
                   meta.__newindex = function(_, key, value)
                     handled = handled + 1
                     backing[key] = value + 1000
                   end
                 end";
    let source = proxy_source(setup, inner, 200_000);
    for (profile, result, stats) in agree(&source) {
        assert!(
            !result.ends_with(", 0]"),
            "{profile:?}: {result}\n{stats:?}"
        );
    }
}

#[test]
fn a_newindex_removed_or_made_a_table_mid_loop_takes_effect() {
    let setup = "local meta, other = getmetatable(proxy), {}";
    let inner = "if i == 70000 then meta.__newindex = other end
                 if i == 140000 then meta.__newindex = nil end";
    let source = proxy_source(setup, inner, 200_000);
    agree(&source);
    let swapped = "local meta = getmetatable(proxy)";
    let inner = "if i == 100000 then setmetatable(proxy, {__newindex = function(_, k, v) backing[k] = -v end}) end";
    agree(&proxy_source(swapped, inner, 200_000));
}

#[test]
fn a_backing_table_that_grows_and_rehashes_mid_loop_keeps_the_stores() {
    // The backing table gains hundreds of keys in a burst now and then, which
    // moves its string index under the trace.
    let inner = "if i % 40000 == 0 then
                   for j = 1, 700 do backing['grow' .. i .. '_' .. j] = j end
                 end";
    let source = proxy_source("", inner, 200_000);
    for (profile, result, stats) in agree(&source) {
        assert!(result.contains("64"), "{profile:?}: {result}\n{stats:?}");
    }
}

#[test]
fn storing_nil_through_the_handler_deletes_the_key() {
    let source = proxy_value_source("", "", "(i % 7 ~= 0) and (i % 101) or nil", 200_000);
    agree(&source);
    // Every key is deleted at the end of the loop.
    let inner = "local v = i % 101; if i > 150000 then v = nil end";
    let source = proxy_value_source("", inner, "v", 200_000);
    for (profile, result, stats) in agree(&source) {
        assert!(
            result.contains("Number(0.0), Number(0.0)")
                || result.contains("Integer(0), Integer(0)"),
            "{profile:?}: {result}\n{stats:?}"
        );
    }
}

#[test]
fn a_value_that_changes_type_mid_loop_is_stored_as_it_is() {
    let value = "(i < 100000) and (i % 101) or ((i < 150000) and tostring(i % 11) or (i % 2 == 0))";
    agree(&proxy_value_source("", "", value, 200_000));
}

#[test]
fn non_string_keys_at_the_site_reach_the_handler_too() {
    // Every eighth key is a number, a float or a boolean. The handler's
    // store is a string-key store only some of the time.
    let setup = "keys[8] = 8; keys[16] = 2.5; keys[24] = true; keys[32] = 8";
    agree(&proxy_source(setup, "", 200_000));
}

#[test]
fn a_store_that_first_appears_on_a_side_trace_still_gets_the_in_place_version() {
    // The root has no string store. After the table fills, the branch that
    // stores becomes a side trace, whose helper stores must switch to the
    // in-place version rather than stay on the helper.
    let source = "local keys = {} for n = 1, 64 do keys[n] = 'k' .. n end \
                  local t, s = {}, 0 \
                  for i = 1, 2000000 do \
                    local k = keys[(i - 1) % 64 + 1] \
                    if i > 5000 then t[k] = i else s = s + 1 end \
                  end \
                  return s, t.k1, t.k64";
    agree(source);
    let stats = stats_of(CompatibilityProfile::Lua54, source);
    assert!(
        stats.set_string_helper_calls < 600_000,
        "helper stores stayed unbounded: {stats:?}"
    );
}

#[test]
fn a_loop_that_only_inserts_returns_to_the_full_budget_after_a_few_polls() {
    // Every store adds a new key, so the in-place version never applies. The
    // trace polls early while it waits for overwrites (1024 iterations per
    // entry), gives up after a few polls and then runs with the whole budget:
    // the entries must not grow with the iterations.
    let source = "local names = {} \
                  for i = 1, 600000 do names[i] = 'k' .. i end \
                  local t = {} \
                  for i = 1, 600000 do t[names[i]] = i end \
                  return t.k1, t.k600000";
    for (profile, _, stats) in agree(source) {
        if profile == CompatibilityProfile::Lua54 {
            // Two loops of 600k iterations: at 1024 per entry the insert loop
            // alone would enter some 580 times.
            assert!(
                stats.native_entries < 100,
                "{profile:?}: the trace kept polling every 1024 iterations: {stats:?}"
            );
        }
    }
}

#[test]
fn a_nest_that_fills_a_big_table_before_it_overwrites_it_still_gets_the_in_place_version() {
    // The first round inserts 20000 keys through the helper, which is many
    // polls that do not qualify. The rounds after it only overwrite, and the
    // trace must still ask for the in-place version: giving up on the fill
    // would leave 1.2M helper stores.
    let source = "local names = {} \
                  for i = 1, 20000 do names[i] = 'k' .. i end \
                  local t = {} \
                  for round = 1, 60 do \
                    for i = 1, 20000 do t[names[i]] = i + round end \
                  end \
                  return t.k1, t.k20000";
    agree(source);
    let stats = stats_of(CompatibilityProfile::Lua54, source);
    assert!(
        stats.set_string_helper_calls < 300_000,
        "helper stores stayed on the helper after the fill: {stats:?}"
    );
}

#[test]
fn a_small_fill_followed_by_overwrites_in_one_long_loop_gets_the_in_place_version() {
    // The shape of table/metamethod_newindex: a __newindex proxy stores 256
    // string keys round-robin. The first poll sees only the fill, so the
    // back-off must not park the trace on the helper for the rest of the
    // loop before a poll sees the overwrites.
    let source = "local keys, backing = {}, {} \
                  for n = 1, 256 do keys[n] = 'item_' .. n end \
                  local proxy = setmetatable({}, { __newindex = function(_, k, v) backing[k] = v end }) \
                  for i = 1, 600000 do proxy[keys[(i - 1) % 256 + 1]] = i % 101 end \
                  return backing.item_1, backing.item_256";
    agree(source);
    let stats = stats_of(CompatibilityProfile::Lua51, source);
    assert!(
        stats.set_string_helper_calls < 10_000,
        "helper stores stayed on the helper: {stats:?}"
    );
}
