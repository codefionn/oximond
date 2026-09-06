//! Programs that wait for the collector must see it make progress. The
//! upstream `gc.lua` suites loop until a finalizer runs or the heap
//! shrinks, and used to spin until the conformance deadline.

use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

fn run(profile: CompatibilityProfile, source: &str) -> Vec<Value> {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(ExecutionMode::Interpreter)
        .build()
        .unwrap();
    lua.exec(source)
        .unwrap_or_else(|error| panic!("{profile}: {error}"))
}

#[test]
fn automatic_collection_runs_table_finalizers() {
    let source = r"
        local finish = false
        local u = setmetatable({}, {__gc = function () finish = true end})
        local n = 0
        repeat u = {}; n = n + 1 until finish or n > 1000000
        return finish
    ";
    for profile in [
        CompatibilityProfile::Lua52,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        assert_eq!(
            run(profile, source),
            vec![Value::Boolean(true)],
            "{profile}"
        );
    }
}

#[test]
fn lua51_tables_ignore_gc_metamethods() {
    let source = r"
        local ran = false
        setmetatable({}, {__gc = function () ran = true end})
        collectgarbage()
        return ran
    ";
    assert_eq!(
        run(CompatibilityProfile::Lua51, source),
        vec![Value::Boolean(false)]
    );
}

#[test]
fn weak_values_reachable_only_from_a_finalized_table_are_cleared_first() {
    // Lua 5.1's weak-table bug, as tested by Lua 5.2+ `gc.lua`.
    let source = r"
        local a, t = {}, {x = 10}
        local C = setmetatable({key = t}, {__mode = 'v'})
        local C1 = setmetatable({[t] = 1}, {__mode = 'k'})
        a.x = t
        local seen
        setmetatable(a, {__gc = function ()
            seen = C.key == nil and type(next(C1)) == 'table'
        end})
        a, t = nil
        collectgarbage()
        collectgarbage()
        return seen, next(C) == nil and next(C1) == nil
    ";
    assert_eq!(
        run(CompatibilityProfile::Lua54, source),
        vec![Value::Boolean(true), Value::Boolean(true)]
    );
}

#[test]
fn incremental_mode_shrinks_the_heap_after_a_stop() {
    // The pace check from Lua 5.4's `gc.lua`: after a stopped collector
    // lets the heap triple, the restarted one must bring it back down.
    let source = r#"
        local function gcinfo () return collectgarbage"count" * 1024 end
        collectgarbage("incremental")
        collectgarbage(); collectgarbage()
        local x = gcinfo()
        collectgarbage"stop"
        repeat local a = {} until gcinfo() > 3 * x
        collectgarbage"restart"
        local n = 0
        repeat
            local a = {}
            n = n + 1
        until gcinfo() <= x * 2 or n > 200000
        return n <= 200000
    "#;
    assert_eq!(
        run(CompatibilityProfile::Lua54, source),
        vec![Value::Boolean(true)]
    );
}

const EPHEMERON_PROFILES: [CompatibilityProfile; 4] = [
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
];

/// Prepends a `count(t)` helper that counts a table's entries.
fn with_count(body: &str) -> String {
    format!(
        "local function count(t)
            local n = 0
            for _ in pairs(t) do n = n + 1 end
            return n
        end
        {body}"
    )
}

#[test]
fn weak_keys_are_ephemerons_from_lua52() {
    // A key reachable only through its own value, and a 100-link chain in
    // which each key is reachable only through the previous entry's value,
    // as in `gc.lua`'s ephemeron checks.
    let source = with_count(
        r"
        local a = setmetatable({}, {__mode = 'k'})
        do local k = {}; a[k] = {k} end
        local x
        for i = 1, 100 do local n = {}; a[n] = {k = {x}}; x = n end
        collectgarbage()
        local n, links = x, 0
        while n do n = a[n].k[1]; links = links + 1 end
        x = nil
        collectgarbage()
        return links == 100, count(a) == 0
        ",
    );
    for profile in EPHEMERON_PROFILES {
        assert_eq!(
            run(profile, &source),
            vec![Value::Boolean(true); 2],
            "{profile}"
        );
    }
}

#[test]
fn ephemeron_chains_across_two_tables_live_and_die_together() {
    // `t1[a] = {b}` and `t2[b] = {a}`: `a` keeps `b`'s entry alive through
    // `t1`, and dropping `a` frees both entries in one collection.
    let source = with_count(
        r"
        local t1 = setmetatable({}, {__mode = 'k'})
        local t2 = setmetatable({}, {__mode = 'k'})
        local a = {}
        do local b = {}; t1[a] = {b}; t2[b] = {a} end
        collectgarbage()
        local kept = t2[t1[a][1]][1] == a
        a = nil
        collectgarbage()
        return kept, count(t1) == 0, count(t2) == 0
        ",
    );
    for profile in EPHEMERON_PROFILES {
        assert_eq!(
            run(profile, &source),
            vec![Value::Boolean(true); 3],
            "{profile}"
        );
    }
}

#[test]
fn incremental_steps_and_generational_mode_collect_ephemeron_cycles() {
    let body = r"
        local a = setmetatable({}, {__mode = 'k'})
        local live = {}
        a[live] = {live}
        for i = 1, 50 do local k = {}; a[k] = {k, {}} end
        repeat until collectgarbage('step', 0)
        repeat until collectgarbage('step', 0)
        return count(a) == 1, a[live][1] == live
    ";
    let incremental = with_count(body);
    for profile in EPHEMERON_PROFILES {
        assert_eq!(
            run(profile, &incremental),
            vec![Value::Boolean(true); 2],
            "{profile}"
        );
    }
    let generational = with_count(&format!("collectgarbage('generational')\n{body}"));
    for profile in [CompatibilityProfile::Lua54, CompatibilityProfile::Lua55] {
        assert_eq!(
            run(profile, &generational),
            vec![Value::Boolean(true); 2],
            "{profile}"
        );
    }
}

#[test]
fn lua51_weak_keys_hold_their_values_strongly() {
    // Lua 5.1 and LuaJIT are not ephemeron collectors: a value that refers
    // to its own weak key keeps the entry alive.
    let source = with_count(
        r"
        local a = setmetatable({}, {__mode = 'k'})
        do local k = {}; a[k] = {k} end
        do local k = {}; a[k] = true end
        collectgarbage()
        collectgarbage()
        return count(a)
        ",
    );
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        assert_eq!(run(profile, &source), vec![Value::Number(1.0)], "{profile}");
    }
}

#[test]
fn a_finalizer_in_incremental_mode_runs_after_older_garbage_is_collected() {
    // `gc.lua`'s `GC()` waits for a finalizer and then expects an old
    // ephemeron chain to be gone. A large live heap keeps the collector's
    // old generation from filling, so only the finalizer asks for the
    // major cycle that frees the chain.
    let source = with_count(
        r"
        local live = {}
        for i = 1, 100000 do live[i] = {} end
        local a = setmetatable({}, {__mode = 'k'})
        local x
        for i = 1, 100 do local n = {}; a[n] = {x}; x = n end
        collectgarbage()
        x = nil
        local finished = false
        local u = setmetatable({}, {__gc = function () finished = true end})
        repeat u = {} until finished
        return count(a) == 0, #live == 100000
        ",
    );
    for profile in EPHEMERON_PROFILES {
        assert_eq!(
            run(profile, &source),
            vec![Value::Boolean(true); 2],
            "{profile}"
        );
    }
}
