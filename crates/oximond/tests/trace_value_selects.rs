//! Value selects for pure two-arm branches. A branch on a boolean whose two
//! arms only load constants, move registers or do number arithmetic, and
//! join a few instructions later, records both arms and picks the joined
//! value with a select instead of guarding its direction. A condition that
//! flips from one iteration to the next then never leaves the trace. The
//! results are the interpreter's in every profile.

use oximond::runtime::Value;
use oximond::vm::{TraceGuardKind, TraceStatistics};
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

/// Runs `source` traced and interpreted in each of `profiles`, checks that
/// the outcomes agree and are not errors, and returns the traced run's
/// statistics by profile.
fn run_in(
    profiles: &[CompatibilityProfile],
    source: &str,
) -> Vec<(CompatibilityProfile, TraceStatistics)> {
    profiles
        .iter()
        .map(|&profile| {
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
            let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
            assert_eq!(actual, expected, "{profile:?} {source}");
            assert!(expected.is_ok(), "{profile:?} {expected:?}");
            (profile, lua.vm().jit_statistics().trace)
        })
        .collect()
}

fn run_all(source: &str) -> Vec<(CompatibilityProfile, TraceStatistics)> {
    run_in(&PROFILES, source)
}

fn branch_exits(stats: &TraceStatistics) -> u64 {
    stats.side_exits[TraceGuardKind::Branch.index()]
}

/// Checks that the loop ran natively with a select and that no branch guard
/// ever failed.
fn assert_selected(profile: CompatibilityProfile, stats: &TraceStatistics) {
    assert!(stats.traces_compiled >= 1, "{profile:?} {stats:?}");
    assert!(stats.native_entries >= 1, "{profile:?} {stats:?}");
    assert!(stats.iterations >= 100_000, "{profile:?} {stats:?}");
    assert!(stats.value_selects_compiled >= 1, "{profile:?} {stats:?}");
    assert_eq!(branch_exits(stats), 0, "{profile:?} {stats:?}");
}

#[test]
fn an_and_or_of_a_called_predicate_is_a_select() {
    // `p` reads a flag table whose pattern changes from one iteration to the
    // next, so a guard on its answer would fail on about every third trip.
    let source = "local flags = {}
         for i = 1, 64 do flags[i] = (i % 3 ~= 0) end
         local function p(x) return flags[x % 64 + 1] end
         local t, s = 0.0, 0.25
         for i = 1, 200000 do
           t = t + (p(i) and 1.0 or 0.0) * s
         end
         return t";
    for (profile, stats) in run_all(source) {
        assert_selected(profile, &stats);
    }
}

#[test]
fn a_select_inside_an_inlined_call_that_wrote_an_upvalue() {
    // The density loop's shape: the callee writes an upvalue and then adds
    // the selected weight, so a guard there would leave through the call's
    // frames.
    let source = "local flags = {}
         for i = 1, 64 do flags[i] = (i % 3 ~= 0) end
         local function walkable(x) return flags[x % 64 + 1] end
         local ssum, bsum = 0.0, 0.0
         local function sample(x, s)
           bsum = bsum + s
           ssum = ssum + (walkable(x) and 1.0 or 0.0) * s
         end
         for i = 1, 200000 do
           sample(i, 0.5)
           sample(i + 7, 0.25)
         end
         return ssum, bsum";
    for (profile, stats) in run_all(source) {
        assert_selected(profile, &stats);
        assert_eq!(stats.inner_exits, 0, "{profile:?} {stats:?}");
    }
}

#[test]
fn an_if_else_statement_on_a_comparison_is_a_select() {
    // A compare-and-branch test whose arms write one local from arithmetic.
    let source = "local u = 0
         for i = 1, 200000 do
           local v
           if i % 3 == 0 then v = i * 2 else v = i - 1 end
           u = u + v
         end
         return u";
    for (profile, stats) in run_all(source) {
        assert_selected(profile, &stats);
    }
}

#[test]
fn an_if_without_else_selects_against_the_value_before_the_branch() {
    // One arm writes the register, the other leaves it as it was.
    let source = "local flags = {}
         for i = 1, 64 do flags[i] = (i % 5 < 2) end
         local u = 0.0
         for i = 1, 200000 do
           local v = i * 0.5
           if flags[i % 64 + 1] then v = -v end
           u = u + v
         end
         return u";
    for (profile, stats) in run_all(source) {
        assert_selected(profile, &stats);
    }
}

#[test]
fn x_and_nil_or_y_always_gives_y() {
    // The true arm writes nil and then tests it, so both arms end with `y`:
    // the two agree and need no select at all, and nothing guards `x`.
    let source = "local flags = {}
         for i = 1, 64 do flags[i] = (i % 3 == 0) end
         local u = 0
         for i = 1, 200000 do
           local x = flags[i % 64 + 1]
           local v = x and nil or i
           u = u + v
         end
         return u";
    for (profile, stats) in run_all(source) {
        assert!(stats.iterations >= 100_000, "{profile:?} {stats:?}");
        assert_eq!(branch_exits(&stats), 0, "{profile:?} {stats:?}");
    }
}

#[test]
fn an_arm_that_writes_nil_against_a_boolean_keeps_its_guard() {
    // `x and nil` is nil in one arm and `false` in the other: no select, and
    // the results stay the interpreter's.
    let source = "local flags = {}
         for i = 1, 64 do flags[i] = (i % 3 == 0) end
         local nils = 0
         for i = 1, 200000 do
           local x = flags[i % 64 + 1]
           local v = x and nil
           if v == nil then nils = nils + 1 end
         end
         return nils";
    run_all(source);
}

#[test]
fn not_x_and_1_or_2_selects_integers() {
    let profiles = [
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ];
    let source = "local flags = {}
         for i = 1, 64 do flags[i] = (i % 3 == 0) end
         local u = 0
         for i = 1, 200000 do
           u = u + (not flags[i % 64 + 1] and 1 or 2)
         end
         return u, math.type(u)";
    for (profile, stats) in run_in(&profiles, source) {
        assert_selected(profile, &stats);
    }
}

#[test]
fn a_nan_comparison_selects_like_the_interpreter() {
    // Every ordering with a NaN operand is false, and `not (a >= b)` is not
    // `a < b` there: the selects take the comparison's own outcome.
    let source = "local nan = 0 / 0
         local u, w, z = 0, 0, 0
         for i = 1, 200000 do
           local a = (i % 3 == 0) and nan or i * 1.0
           u = u + ((a < 100000.0) and 1.5 or 2.5)
           local v
           if not (a >= 100000.0) then v = 3.0 else v = 0.5 end
           w = w + v
           if a <= 100000.0 then z = z + 1.0 else z = z - 1.0 end
         end
         return u, w, z";
    for (profile, stats) in run_all(source) {
        assert_selected(profile, &stats);
        assert!(stats.value_selects_compiled >= 4, "{profile:?} {stats:?}");
    }
}

#[test]
fn an_arm_with_a_table_read_or_a_call_keeps_its_guard() {
    // Neither arm is pure: the read may run `__index`, the call anything.
    let source = "local flags, t = {}, {}
         for i = 1, 64 do flags[i] = (i % 3 == 0); t[i] = i end
         local function f(x) return x + 1 end
         local u = 0
         for i = 1, 200000 do
           local k = i % 64 + 1
           local x = flags[k]
           u = u + (x and t[k] or 0) + (x and f(i) or 1)
         end
         return u";
    for (profile, stats) in run_all(source) {
        assert_eq!(stats.value_selects_compiled, 0, "{profile:?} {stats:?}");
    }
}

#[test]
fn an_arm_with_an_integer_modulo_keeps_its_guard() {
    // An integer `%` raises an error on a zero divisor, so it is not total.
    let profiles = [CompatibilityProfile::Lua53, CompatibilityProfile::Lua54];
    let source = "local flags = {}
         for i = 1, 64 do flags[i] = (i % 3 == 0) end
         local u = 0
         for i = 1, 200000 do
           local d = i % 7 + 1
           u = u + (flags[i % 64 + 1] and i % d or 0)
         end
         return u";
    for (profile, stats) in run_in(&profiles, source) {
        assert_eq!(stats.value_selects_compiled, 0, "{profile:?} {stats:?}");
    }
}

#[test]
fn a_condition_that_turns_nil_leaves_through_its_type_guard() {
    // The flag table gets holes halfway: the read that gave a boolean gives
    // nil, which the select's condition cannot be.
    let source = "local flags = {}
         for i = 1, 64 do flags[i] = (i % 3 == 0) end
         local u = 0.0
         for i = 1, 200000 do
           if i == 100000 then
             for k = 1, 64, 5 do flags[k] = nil end
           end
           u = u + (flags[i % 64 + 1] and 1.0 or 0.5)
         end
         return u";
    for (profile, stats) in run_all(source) {
        assert!(stats.value_selects_compiled >= 1, "{profile:?} {stats:?}");
    }
}

#[test]
fn an_arm_operand_that_changes_type_leaves_through_its_guard() {
    // `y` is an integer, then a float: the arm's arithmetic was recorded for
    // one type, and the entry or a guard sends the other to the interpreter.
    let source = "local flags = {}
         for i = 1, 64 do flags[i] = (i % 3 == 0) end
         local y = 3
         local u = 0
         for i = 1, 200000 do
           if i == 100000 then y = 3.5 end
           u = u + (flags[i % 64 + 1] and y * 2 or y + 1)
         end
         return u";
    run_all(source);
}

#[test]
fn a_flag_table_swapped_mid_loop_or_given_a_metatable_agrees() {
    // The condition comes from a table that is replaced by one with holes
    // and an `__index` that answers for them.
    let source = "local flags = {}
         for i = 1, 64 do flags[i] = (i % 3 == 0) end
         local u = 0.0
         for i = 1, 200000 do
           if i == 60000 then
             local other = {}
             for k = 1, 64 do flags[k] = nil; other[k] = (k % 4 == 0) end
             flags = other
           elseif i == 120000 then
             for k = 1, 64, 2 do flags[k] = nil end
             setmetatable(flags, { __index = function (_, k) return k % 2 == 1 end })
           end
           u = u + (flags[i % 64 + 1] and 1.0 or 0.25)
         end
         return u";
    run_all(source);
}

#[test]
fn a_count_hook_sees_the_interpreters_counts() {
    // The two arms run different instruction counts, so a recording that
    // counts for a hook keeps its guard: the hook fires as often as it does
    // in the interpreter.
    let source = "local flags = {}
         for i = 1, 64 do flags[i] = (i % 3 == 0) end
         local hits = 0
         debug.sethook(function () hits = hits + 1 end, '', 997)
         local u = 0.0
         for i = 1, 200000 do
           u = u + (flags[i % 64 + 1] and 1.0 or 0.25)
         end
         debug.sethook()
         return u, hits";
    for (profile, stats) in run_all(source) {
        assert!(stats.iterations >= 100_000, "{profile:?} {stats:?}");
        assert_eq!(stats.value_selects_compiled, 0, "{profile:?} {stats:?}");
    }
}

#[test]
fn an_arm_branch_on_a_carried_local_guards_its_type_at_entry() {
    // The `if i % 2 == 0` arms are a select whose true arm branches on `x`.
    // Only `x`'s type decides that branch, and no op reads `x`, so the
    // entry contract must guard it: the outer loop makes `x` nil between
    // runs of the inner loop, and the trace recorded for a float must not
    // keep its arm.
    let source = "local s = 0
         for outer = 1, 6 do
           local x = 1.5
           if outer > 3 then x = nil end
           for i = 1, 40000 do
             local v
             if i % 2 == 0 then if x then v = 1 else v = 2 end else v = 3 end
             s = s + v
           end
         end
         return s";
    for (profile, stats) in run_all(source) {
        assert!(stats.value_selects_compiled >= 1, "{profile:?} {stats:?}");
    }
    // The other way round: nil first, then a number.
    let source = "local s = 0
         for outer = 1, 6 do
           local x
           if outer > 3 then x = 7 end
           for i = 1, 40000 do
             local v
             if i % 2 == 0 then if x then v = 1 else v = 2 end else v = 3 end
             s = s + v
           end
         end
         return s";
    run_all(source);
}

#[test]
fn an_arm_join_on_a_carried_nil_guards_its_type_at_entry() {
    // The arms move `a` or `b`, both nil when recorded: the join takes `a`
    // for both without a select, and no op reads `b`. When `b` later holds
    // a number, the trace recorded for nil must not run.
    let source = "local s = 0
         for outer = 1, 6 do
           local a, b
           if outer > 3 then b = 5 end
           for i = 1, 40000 do
             local v
             if i % 2 == 0 then v = a else v = b end
             if v then s = s + v else s = s + 1 end
           end
         end
         return s";
    run_all(source);
}

/// The profiles whose numbers are all floats. Their traces carry a tree's
/// loop side; the integer profiles do not build one for these nests.
fn has_float_traces(profile: CompatibilityProfile) -> bool {
    matches!(
        profile,
        CompatibilityProfile::Lua51 | CompatibilityProfile::Lua52 | CompatibilityProfile::LuaJit21
    )
}

#[test]
fn a_select_in_front_of_a_loop_side_stored_through_a_raw_set_agrees() {
    // The reduced shape-359 miscompile (`mcl_util/shape.lua:359`,
    // `region_op` with `OP_SUB`'s `l and not r`). `scale` is a select on
    // `flag` in front of the update loop, which the tree runs as a loop
    // side, and the select's value goes into a nil hole of `marks` through
    // the raw set (`TABLE_SET_INDEX`). The select leaves the loop's `if
    // flag` as its only branch on an invariant, so that guard moves into
    // the loop side's preheader. Every 97th call flips it there. Before the
    // loop side's entry snapshot named its entry state, that exit left the
    // interpreter's registers as the previous call left them, and the loop
    // resumed with a stale `i` (an index of nil, "attempt to index local
    // 'a'"). The select is not at fault: the CLI repro fails the same way
    // with `OXIMOND_JIT_SELECTS=0`, where the guard on `flag` in front of
    // the loop leaves the loop's own guard invariant all the same.
    let source = "local items = {}
         for i = 1, 6 do items[i] = {x = i, v = i * 0.5} end
         local marks = {}
         local function step(dt, flag, n)
           for i = 1, #items do
             local a = items[i]
             for j = i + 1, #items do
               local b = items[j]
               local d = b.x - a.x
               a.v = a.v + d * dt
               b.v = b.v - d * dt
             end
           end
           local scale = flag and 2 or 3
           for i = 1, #items do
             local a = items[i]
             if flag then
               a.x = a.x + a.v * dt
             else
               a.x = a.x - a.v * dt / scale
             end
             local k = (n * 7 + i) % 50 + 1
             marks[k] = (marks[k] or 0) + scale
           end
         end
         for n = 1, 12000 do step(0.001, n % 97 ~= 0, n) end
         local total = 0
         for i = 1, #items do total = total + items[i].x + items[i].v end
         for k, v in pairs(marks) do total = total + k * v end
         return math.floor(total * 1000)";
    for (profile, stats) in run_all(source) {
        assert!(stats.native_entries >= 1, "{profile:?} {stats:?}");
        if has_float_traces(profile) {
            assert!(stats.loop_sides >= 1, "{profile:?} {stats:?}");
            assert!(stats.value_selects_compiled >= 1, "{profile:?} {stats:?}");
        }
    }
}

#[test]
fn region_ops_select_of_two_booleans_into_a_nil_hole_agrees() {
    // `region_op`'s nest: a `y` while loop around a `z` while loop, the
    // dominating values nil or a number, and `op (l_on or false, r_on or
    // false)` with `OP_SUB`'s `l and not r` a select on booleans, whose
    // answer decides a raw-set store into a nil hole.
    let source = "local function OP_SUB (l, r) return l and not r end
         local function dom (il, ir, l_max, r_max)
           if il == l_max then return nil, ir ~= r_max and ir or nil
           elseif ir == r_max then return il ~= l_max and il or nil, nil end
           return il, ir
         end
         local function region_op (lsize, rsize, op, solids)
           local count = 0
           local ly_max, ry_max = lsize + 1, rsize + 1
           local ly, ry = 1, 1
           local y = 0
           while ly ~= ly_max or ry ~= ry_max do
             local lz_max, rz_max = lsize + 1, rsize + 1
             local lz, rz = 1, 1
             local ly_dom, ry_dom = dom (ly, ry, ly_max, ry_max)
             local z = 0
             while lz ~= lz_max or rz ~= rz_max do
               local lz_dom, rz_dom = dom (lz, rz, lz_max, rz_max)
               local l_on = ly_dom and lz_dom and (ly_dom + lz_dom) % 3 ~= 0
               local r_on = ry_dom and rz_dom and (ry_dom * rz_dom) % 4 == 1
               if op (l_on or false, r_on or false) then
                 local idx = y * 64 + z + 1
                 solids[idx] = (solids[idx] or 0) + 1
                 count = count + 1
               end
               if lz ~= lz_max and (rz == rz_max or lz <= rz) then
                 lz = lz + 1
               else
                 rz = rz + 1
               end
               z = z + 1
             end
             if ly ~= ly_max and (ry == ry_max or ly <= ry) then
               ly = ly + 1
             else
               ry = ry + 1
             end
             y = y + 1
           end
           return count
         end
         local total, seed = 0, 7
         for it = 1, 3000 do
           seed = (seed * 1103515245 + 12345) % 2147483648
           local a = seed % 5 + 1
           local b = math.floor (seed / 256) % 5 + 1
           local solids = {}
           total = total + region_op (a, b, OP_SUB, solids)
           for k, v in pairs (solids) do total = total + k * v end
         end
         return total";
    for (profile, stats) in run_all(source) {
        assert!(stats.native_entries >= 1, "{profile:?} {stats:?}");
        assert!(stats.value_selects_compiled >= 1, "{profile:?} {stats:?}");
    }
}

#[test]
fn a_loop_sides_hoisted_field_guard_that_fails_agrees() {
    // A field of the loop side that turns a string every 97th call, read
    // beside a select stored through the raw set: the trace leaves at the
    // field's type guard and the program finishes as the interpreter does.
    // The guard stays in the loop: the side's stores to lane tables may
    // write `cfg`, so its read is not hoisted. This test therefore does not
    // reach the side's entry snapshot of commit 7c89c5bf, and passes with
    // that snapshot switched off (checked in a scratch copy on 2026-11-02,
    // also with the read moved to a local, an argument, or a branch on the
    // field). `a_select_in_front_of_a_loop_side_stored_through_a_raw_set_agrees`
    // above and `a_loop_side_whose_invariant_branch_flips_leaves_with_the_state_it_started_from`
    // in `trace_loop_sides.rs` fail without it.
    let source = "local items = {}
         for i = 1, 6 do items[i] = {x = i, v = i * 0.5} end
         local cfg = {k = 2.0}
         local marks = {}
         local function step(dt, n)
           for i = 1, #items do
             local a = items[i]
             for j = i + 1, #items do
               local b = items[j]
               local d = b.x - a.x
               a.v = a.v + d * dt
               b.v = b.v - d * dt
             end
           end
           for i = 1, #items do
             local a = items[i]
             local on = a.v > 0 and 1 or 2
             a.x = a.x + a.v * dt * cfg.k
             local key = (n + i) % 40 + 1
             marks[key] = (marks[key] or 0) + on
           end
         end
         for n = 1, 12000 do
           cfg.k = (n % 97 == 0) and '3' or 2.0
           step(0.001, n)
         end
         local total = 0
         for i = 1, #items do total = total + items[i].x + items[i].v end
         for k, v in pairs(marks) do total = total + k * v end
         return math.floor(total * 1000)";
    for (profile, stats) in run_all(source) {
        assert!(stats.native_entries >= 1, "{profile:?} {stats:?}");
        if has_float_traces(profile) {
            assert!(stats.loop_sides >= 1, "{profile:?} {stats:?}");
        }
    }
}
