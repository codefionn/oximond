//! Native traces over loops that assign globals (`SetGlobal`), in the
//! profiles that have a global opcode.
//!
//! Every program runs under the interpreter and under tracing execution. The
//! results must match, and the loops that can run natively must show native
//! iterations, because equal output alone cannot tell a trace from a slow
//! path. A trace writes a global in place in the environment table, so what
//! these programs hold it to is every way the interpreter would not have
//! stored in place: a `__newindex` on the environment, an environment that
//! `setfenv` replaces, a value the table cannot hold in place, and a hook.

use oximond::vm::TraceStatistics;
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

const PROFILES: [CompatibilityProfile; 2] =
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
    let results: Vec<Value> = lua
        .exec(source)
        .unwrap_or_else(|error| panic!("{profile:?} {mode:?}: {error}\n{source}"));
    (format!("{results:?}"), lua.vm().jit_statistics().trace)
}

/// Runs `source` in both modes of every profile, checks the results agree and
/// returns the tracing statistics of each profile.
fn agree(source: &str) -> Vec<TraceStatistics> {
    PROFILES
        .iter()
        .map(|profile| {
            let (expected, _) = run(*profile, ExecutionMode::Interpreter, source);
            let (tracing, stats) = run(*profile, ExecutionMode::Tracing, source);
            assert_eq!(
                tracing, expected,
                "{profile:?} tracing\n{source}\n{stats:?}"
            );
            stats
        })
        .collect()
}

/// Like [`agree`], requiring that most of `min` iterations ran natively.
fn agree_natively(source: &str, min: u64) {
    for stats in agree(source) {
        assert!(stats.iterations >= min, "{stats:?}\n{source}");
    }
}

#[test]
fn a_loop_that_assigns_a_global_runs_natively() {
    agree_natively(
        "
        g = 0
        for i = 1, 200000 do g = g + i end
        return g",
        190_000,
    );
}

#[test]
fn a_global_read_after_its_store_sees_the_stored_value() {
    agree_natively(
        "
        a = 0
        b = 1
        k = 3
        for i = 1, 100000 do
          a = a + k
          b = a * 2 + b - a
          a = b - k
        end
        return a, b",
        90_000,
    );
}

#[test]
fn a_boolean_global_and_a_global_read_by_a_callee_stay_exact() {
    agree(
        "
        flag = true
        n = 0
        local function peek() return n end
        local seen = 0
        for i = 1, 5000 do
          n = n + 1
          flag = not flag
          seen = seen + peek()
        end
        return n, flag, seen",
    );
}

#[test]
fn a_table_field_alias_of_the_environment_stays_exact() {
    agree(
        "
        g = 0
        for i = 1, 50000 do
          g = g + 1
          _G.g = _G.g + 1
          g = g + _G.g
        end
        return g",
    );
}

#[test]
fn a_global_that_is_deleted_mid_loop_runs_newindex_after_it() {
    let source = "
        local calls = 0
        setmetatable(_G, { __newindex = function(t, k, v)
          calls = calls + 1
          rawset(t, k, v)
        end })
        x = 0
        for i = 1, 60000 do
          x = i
          if i % 7000 == 0 then x = nil end
        end
        return calls, x";
    for stats in agree(source) {
        assert!(stats.iterations > 0, "{stats:?}");
    }
}

#[test]
fn an_environment_with_newindex_for_a_new_global_is_never_stored_in_place() {
    agree(
        "
        local log = {}
        setmetatable(_G, { __newindex = function(t, k, v)
          log[#log + 1] = k
          rawset(t, k, v)
        end })
        for i = 1, 3000 do
          fresh = i
          again = fresh + 1
        end
        return #log, fresh, again",
    );
}

#[test]
fn setfenv_in_the_loop_moves_the_stores_to_the_new_environment() {
    for stats in &agree(
        "
        g = 0
        local other = setmetatable({ g = 100 }, { __index = _G })
        local function run()
          for i = 1, 40000 do
            g = g + 1
            if i == 20000 then setfenv(1, other) end
          end
        end
        run()
        return g, other.g, rawget(_G, 'g')",
    ) {
        assert!(stats.iterations > 0, "{stats:?}");
    }
}

#[test]
fn one_function_run_in_other_environments_stores_into_each() {
    let source = "
        local function count(n)
          for i = 1, n do total = total + i end
        end
        local one = setmetatable({ total = 0 }, { __index = _G })
        local two = setmetatable({ total = 1000 }, { __index = _G })
        local three = setmetatable({ total = 0.5 }, { __index = _G })
        for round = 1, 20 do
          setfenv(count, one)
          count(2000)
          setfenv(count, two)
          count(2000)
          setfenv(count, three)
          count(2000)
        end
        return one.total, two.total, three.total";
    for stats in agree(source) {
        assert!(stats.iterations > 0, "{stats:?}");
    }
}

#[test]
fn a_global_that_changes_type_between_entries_falls_back() {
    agree(
        "
        local function bump() for i = 1, 3000 do v = v + 1 end end
        v = 0
        bump()
        v = 'str'
        local ok = pcall(bump)
        v = 2.5
        bump()
        v = true
        local ok2 = pcall(bump)
        return v, ok, ok2",
    );
}

#[test]
fn stores_of_strings_and_tables_are_exact() {
    agree(
        "
        s = ''
        t = nil
        n = 0
        for i = 1, 3000 do
          n = n + 1
          if i % 100 == 0 then s = 'v' .. i end
          t = { i }
          if i % 7 == 0 then collectgarbage() end
        end
        return n, s, t[1]",
    );
    agree(
        "
        keep = {}
        for i = 1, 4000 do
          last = tostring(i)
          keep[i % 10] = last
          total = (total or 0) + i
        end
        return last, total, keep[3]",
    );
}

#[test]
fn a_store_that_would_error_errors_the_same_way() {
    agree(
        "
        g = 0
        local ok, err = pcall(function()
          for i = 1, 3000 do
            g = g + 1
            if i == 2500 then error('stop at ' .. g) end
          end
        end)
        return g, ok, err",
    );
}

#[test]
fn count_and_line_hooks_see_the_same_globals_in_both_modes() {
    agree(
        "
        g = 0
        local fired, seen = 0, 0
        debug.sethook(function() fired = fired + 1; seen = seen + g end, '', 100)
        for i = 1, 5000 do g = g + 1 end
        debug.sethook()
        return g, fired > 0, seen > 0",
    );
    agree(
        "
        g = 0
        local lines = 0
        debug.sethook(function() lines = lines + 1 end, 'l')
        for i = 1, 300 do g = g + 1 end
        debug.sethook()
        return g, lines",
    );
    agree(
        "
        g = 0
        for i = 1, 5000 do g = g + 1 end
        local fired = 0
        debug.sethook(function() fired = fired + 1 end, '', 50)
        for i = 1, 5000 do g = g + 1 end
        debug.sethook()
        return g, fired",
    );
}
