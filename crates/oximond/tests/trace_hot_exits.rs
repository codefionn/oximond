//! Hot exits become native code. A guard that fails often after the loop body
//! committed stores, a guard inside an inlined call, and a guard inside the
//! iterator closure of a generic for that writes its upvalues all start side
//! traces. Nested loops run natively too, whether the inner loop ends, breaks
//! or returns. Every source runs in each profile and must agree with the
//! interpreter; the traced run must have entered native code and run a side.

use oximond::runtime::Value;
use oximond::vm::TraceAbortReason;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn outcome(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
) -> (Result<Vec<Value>, String>, Lua) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let results = lua.exec(source).map_err(|error| format!("{error:?}"));
    (results, lua)
}

/// Runs `source` in every profile traced and interpreted, checks they agree,
/// and checks that the traced run entered native code, installed a side trace
/// and ran it natively: a run linked back from it (`side_entries`) or left
/// through one of its own guards (`side_member_exits`).
fn assert_side(source: &str) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        let text = format!("{stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {text}");
        assert!(stats.side_installs >= 1, "{profile:?} {text}");
        assert!(
            stats.side_entries + stats.side_member_exits >= 1,
            "{profile:?} {text}"
        );
    }
}

/// Runs `source` in every profile traced and interpreted and checks that they
/// agree and that the inner loop ran natively: native entries, a compiled
/// trace, and most of the loop's trips inside native code. Whether the loops
/// around it join the tree depends on the profile and on what an outer side
/// would save (see "Nested loops" in the trace module notes), so the rest is
/// not asserted. `trips` is the number of inner trips the source runs.
fn assert_nest(source: &str, trips: u64, sides: bool) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        let text = format!("{stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {text}");
        assert!(stats.traces_compiled >= 1, "{profile:?} {text}");
        assert!(stats.iterations * 2 >= trips, "{profile:?} {text}");
        // The inner loop's side traces ran natively, and the loops around it
        // did not keep recording and aborting: a header turns away at the
        // second abort at its inner loop, and a nest has two headers around
        // the innermost.
        // Whether the profile's loops end in a side or in the loop's own
        // exit depends on its unrolling and tiering; the LuaJIT profile is
        // the one the game runs.
        if sides && profile == CompatibilityProfile::LuaJit21 {
            assert!(
                stats.side_entries + stats.side_member_exits >= 1,
                "{profile:?} {text}"
            );
        }
        for reason in [TraceAbortReason::InnerLoop, TraceAbortReason::LoopNotClosed] {
            assert!(
                stats.recording_aborts[reason.index()] <= 4,
                "{profile:?} {reason:?} {text}"
            );
        }
    }
}

#[test]
fn a_branch_in_an_inlined_call_after_a_store_gets_a_side() {
    assert_side(
        "local t, out = {}, {}
         for i = 1, 4096 do t[i] = i end
         local function pick(v, n) if n % 8 ~= 0 then return v + 1 else return v - 1 end end
         local sum = 0
         for r = 1, 40 do
             for i = 1, 4096 do
                 out[i] = t[i] * 2
                 sum = sum + pick(t[i], i)
             end
         end
         return sum, #out",
    );
}

#[test]
fn a_branch_in_an_iterator_closure_that_writes_upvalues_gets_a_side() {
    assert_side(
        "local x, y, z, nx, ny, nz
         local function step()
             if z < nz then z = z + 1; return x, y, z
             elseif y < ny then y = y + 1; z = 1; return x, y, z
             elseif x < nx then x = x + 1; y = 1; z = 1; return x, y, z end
         end
         local function iter(a, b, c) x, y, z, nx, ny, nz = 1, 1, 0, a, b, c; return step end
         local g, sum = {}, 0
         for rep = 1, 30 do
             for px, py, pz in iter(8, 8, 40) do
                 g[(px * 8 + py) * 40 + pz] = px + py + pz
                 sum = sum + g[(px * 8 + py) * 40 + pz]
             end
         end
         return sum",
    );
}

#[test]
fn nested_numeric_loops_run_natively() {
    assert_nest(
        "local m, sum = {}, 0
         for i = 1, 40 do m[i] = {} for j = 1, 40 do m[i][j] = i * j end end
         for r = 1, 600 do
             for i = 1, 40 do
                 local row = m[i]
                 for j = 1, 40 do
                     row[j] = row[j] + 1
                     if j % 16 == 0 then sum = sum + row[j] end
                 end
             end
         end
         return sum",
        600 * 40 * 40,
        true,
    );
}

#[test]
fn nested_loops_with_a_break_out_of_the_inner_loop_run_natively() {
    assert_nest(
        "local sum = 0
         local t = {}
         for i = 1, 64 do t[i] = i end
         for r = 1, 3000 do
             for i = 1, 64 do
                 for j = 1, 64 do
                     if t[j] > i then break end
                     sum = sum + t[j]
                 end
                 t[i] = t[i] + 0
             end
         end
         return sum",
        3000 * 64 * 32,
        true,
    );
}

#[test]
fn an_inner_loop_with_an_early_return_runs_natively() {
    assert_nest(
        "local t = {}
         for i = 1, 100 do t[i] = i % 17 end
         local function find(limit)
             for i = 1, 100 do
                 for j = 1, 8 do
                     if t[i] + j == limit then return i * 10 + j end
                 end
             end
             return 0
         end
         local sum = 0
         for r = 1, 30000 do sum = sum + find(r % 24) end
         return sum",
        100_000,
        // The side of the exit that returns runs up to the `Return` and
        // leaves to the interpreter there (see "Nested loops").
        true,
    );
}

#[test]
fn early_returns_of_other_shapes_agree_with_the_interpreter() {
    // Several values, a call's open results, a table, a string and a
    // captured local: the exit in front of the `Return` must publish every
    // register the return reads.
    for returned in [
        "i, j, limit",
        "i * 10 + j, t[i]",
        "pack(i, j)",
        "{i, j}",
        "'k' .. i",
        "(function() return i + j end)()",
        "captured + i, j",
    ] {
        let source = format!(
            "local t = {{}}
             for i = 1, 100 do t[i] = i % 17 end
             local captured = 5
             local function pack(...) return ... end
             local function find(limit)
                 for i = 1, 100 do
                     for j = 1, 8 do
                         if t[i] + j == limit then return {returned} end
                     end
                 end
                 captured = captured + 1
                 return 0
             end
             local sum = 0
             for r = 1, 20000 do
                 local a, b, c = find(r % 24)
                 if type(a) == 'table' then a = a[1] + a[2] end
                 if type(a) == 'string' then a = #a end
                 sum = sum + a + (b or 0) + (c or 0)
             end
             return sum, captured"
        );
        for profile in PROFILES {
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, &source);
            let (actual, _) = outcome(profile, ExecutionMode::Tracing, &source);
            assert_eq!(actual, expected, "{profile:?} {returned}");
        }
    }
}

#[test]
fn an_ipairs_loop_inside_a_numeric_for_runs_natively() {
    assert_side(
        "local rows = {}
         for i = 1, 30 do local r = {} for j = 1, 12 do r[j] = i + j end rows[i] = r end
         local sum = 0
         for rep = 1, 300 do
             for i = 1, #rows do
                 for _, v in ipairs(rows[i]) do
                     sum = sum + v
                     -- The `%` keeps the arm impure: no value select.
                     if v % 5 == 0 then sum = (sum + 1) % 1000003 end
                 end
             end
         end
         return sum",
    );
}

#[test]
fn hot_exits_of_the_iterator_loop_stay_out_of_the_interpreter() {
    // One exit per row of 40 trips: without a side trace each of the 8 * 8 *
    // 30 rows went back to the interpreter.
    let source = "local x, y, z, nx, ny, nz
         local function step()
             if z < nz then z = z + 1; return x, y, z
             elseif y < ny then y = y + 1; z = 1; return x, y, z
             elseif x < nx then x = x + 1; y = 1; z = 1; return x, y, z end
         end
         local function iter(a, b, c) x, y, z, nx, ny, nz = 1, 1, 0, a, b, c; return step end
         local g, sum = {}, 0
         for rep = 1, 200 do
             for px, py, pz in iter(8, 8, 40) do
                 g[(px * 8 + py) * 40 + pz] = px + py + pz
                 sum = sum + g[(px * 8 + py) * 40 + pz]
             end
         end
         return sum";
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        let branch = Some(oximond::vm::TraceGuardKind::Branch);
        let left = stats.interpreter_exits_of(oximond::vm::TraceExitKind::GuardFail, branch)
            + stats.interpreter_exits_of(oximond::vm::TraceExitKind::CommittedEffect, branch);
        assert!(
            left < 200,
            "{profile:?} {left} branch exits stayed: {stats:?}"
        );
    }
}

#[test]
fn a_branch_in_an_iterator_closure_after_an_upvalue_write_stays_out_of_the_interpreter() {
    // The step writes `z` and `calls` before every branch, so the guards
    // after them fail with those stores committed, inside the inlined call.
    // Exits that rebuild the callee's frame take no side trace of their own
    // (see "Side traces" in the trace module notes), but the ones that
    // resume in the root frame do, and they carry the loop: the interpreter
    // keeps well under one exit in twenty iterations.
    let source = "local x, y, z, nx, ny, nz, calls
         calls = 0
         local function step()
             calls = calls + 1
             z = z + 1
             if z <= nz then return x, y, z end
             z = 1
             if y < ny then y = y + 1; return x, y, z
             elseif x < nx then x = x + 1; y = 1; return x, y, z end
         end
         local function iter(a, b, c) x, y, z, nx, ny, nz = 1, 1, 0, a, b, c; return step end
         local g, sum = {}, 0
         for rep = 1, 200 do
             for px, py, pz in iter(8, 8, 40) do
                 g[(px * 8 + py) * 40 + pz] = px + py + pz
                 sum = sum + g[(px * 8 + py) * 40 + pz]
             end
         end
         return sum, calls";
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        let text = format!("{stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {text}");
        if profile == CompatibilityProfile::LuaJit21 {
            assert!(stats.side_installs >= 1, "{profile:?} {text}");
            assert!(
                stats.side_entries + stats.side_member_exits >= 1,
                "{profile:?} {text}"
            );
            let branch = Some(oximond::vm::TraceGuardKind::Branch);
            let left = stats.interpreter_exits_of(oximond::vm::TraceExitKind::GuardFail, branch)
                + stats.interpreter_exits_of(oximond::vm::TraceExitKind::CommittedEffect, branch);
            assert!(
                left < 512_000 / 20,
                "{profile:?} {left} exits stayed: {text}"
            );
        }
    }
}

#[test]
fn a_metatable_that_appears_while_a_side_runs_leaves_to_the_interpreter() {
    assert_side(
        "local x, y, z, nx, ny, nz
         local function step()
             if z < nz then z = z + 1; return x, y, z
             elseif y < ny then y = y + 1; z = 1; return x, y, z
             elseif x < nx then x = x + 1; y = 1; z = 1; return x, y, z end
         end
         local function iter(a, b, c) x, y, z, nx, ny, nz = 1, 1, 0, a, b, c; return step end
         local g, sum, log = {}, 0, 0
         for rep = 1, 40 do
             if rep == 30 then
                 setmetatable(g, {__newindex = function(t, k, v) log = log + 1 rawset(t, k, v) end})
                 for k in pairs(g) do g[k] = nil end
             end
             for px, py, pz in iter(8, 8, 40) do
                 g[(px * 8 + py) * 40 + pz] = px + py + pz
                 sum = sum + px
             end
         end
         return sum, log",
    );
}

/// Runs `source` in every profile traced and interpreted, checks they agree,
/// and checks that the traced run entered native code.
fn assert_agrees_native(source: &str) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
    }
}

#[test]
fn a_store_through_an_inlined_newindex_that_grows_the_table_keeps_the_border() {
    // The trace inlines the `__newindex` function and stores through
    // `rawset`. The store reaches the table by lane while the loop binds the
    // same table for `#t`, so the border word must be rewritten by the store
    // helper: it used to stay one behind once the array part grew, and the
    // next key named a slot that already held a value, losing one iteration
    // per growth.
    assert_agrees_native(
        "local log, t = 0, {}
         local function run(n) for i = 1, n do t[#t + 1] = i end end
         for round = 1, 4 do
             run(40000)
             if round == 2 then
                 setmetatable(t, {__newindex = function(t, k, v) log = log + 1 rawset(t, k, v) end})
             end
         end
         return #t, log",
    );
}

#[test]
fn a_branch_after_a_store_through_an_inlined_newindex_keeps_every_iteration() {
    // The same loop with a pop in every fourth iteration: the branch guard
    // that follows a committed store takes a side, which must start from the
    // state the interpreter would resume from.
    assert_agrees_native(
        "local log, t = 0, {}
         local function run(n)
             for i = 1, n do
                 t[#t + 1] = i
                 if i % 4 == 0 then t[#t] = nil end
             end
         end
         for round = 1, 4 do
             run(40000)
             if round == 2 then
                 setmetatable(t, {__newindex = function(t, k, v) log = log + 1 rawset(t, k, v) end})
             end
         end
         return #t, log",
    );
}

#[test]
fn a_table_read_that_misses_after_a_store_gets_a_side() {
    // The read of `t[i]` leaves at every third key, a hole, after
    // `out[i] = i` committed: the exit is a `CommittedEffect` of an array
    // guard and starts a side like a branch does.
    assert_side(
        "local t, out = {}, {}
         for i = 1, 2048 do if i % 3 ~= 0 then t[i] = i end out[i] = 0 end
         local s = 0
         for r = 1, 60 do
             for i = 1, 2048 do
                 out[i] = i
                 local v = t[i]
                 if v then s = s + v else s = s + 1 end
             end
         end
         return s, #out",
    );
}

#[test]
fn a_loop_that_meets_an_inner_while_is_turned_away_at_the_second_abort() {
    // The outer `while` meets the inner `while`, which no unrolling can take
    // over: the recording of the outer loop used to abort three times with
    // `InnerLoop` before its header was rejected. The second abort rejects
    // it, the first when the inner loop already has a trace.
    let source = "local t, sum = {}, 0
         for i = 1, 64 do t[i] = i end
         local r = 0
         while r < 400 do
             r = r + 1
             local i = 1
             while i <= 64 do
                 sum = sum + t[i]
                 i = i + 1
             end
         end
         return sum";
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        let text = format!("{stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {text}");
        assert!(
            stats.recording_aborts[TraceAbortReason::InnerLoop.index()] <= 2,
            "{profile:?} {text}"
        );
    }
}

#[test]
fn a_loop_that_calls_a_function_with_a_while_is_turned_away_at_the_first_abort() {
    // The `while` of `halve` cannot be unrolled and the call is taken on
    // every iteration: the recording of the outer loop is not restarted for
    // later iterations, and the header is rejected at its first abort.
    let source = "local function halve(n) local k = 0 while n > 1 do n = (n - n % 2) / 2 k = k + 1 end return k end
         local sum = 0
         for i = 1, 3000 do sum = sum + halve(i + 1000) end
         return sum";
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        let text = format!("{stats:?}");
        assert!(
            stats.recording_aborts[TraceAbortReason::InnerLoop.index()] <= 1,
            "{profile:?} {text}"
        );
    }
}
