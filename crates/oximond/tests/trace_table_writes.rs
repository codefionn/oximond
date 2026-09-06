//! `table.insert` and `table.remove` as library helpers of a trace, and the
//! array stores around them.
//!
//! The loops compile without an abort, enter native code, and give the
//! interpreter's results in every profile. An insert or remove moves the
//! elements of a dense list in a helper that rewrites the words native code
//! holds for the table (its array part and its border), so a read of `#t` or
//! `t[k]` next to one sees the table as it stands. A call the helper cannot
//! answer the way the interpreter does (a metamethod, a position out of
//! range, another kind of value, a replaced function) leaves the trace before
//! the call, and the interpreter makes it.

use oximond::runtime::Value;
use oximond::vm::{TraceAbortReason, TraceExitKind, TraceStatistics};
use oximond::{CompatibilityProfile, ExecutionMode, Lua, LuaJitExtension};

const ALL: [CompatibilityProfile; 6] = [
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
        .luajit_extension(LuaJitExtension::Bit)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let results = lua.exec(source).map_err(|error| format!("{error:?}"));
    (results, lua)
}

/// The outcome as text, which tells an integer from a float.
fn show(result: &Result<Vec<Value>, String>) -> String {
    format!("{result:?}")
}

/// Checks that no recording was abandoned for a reason a call of the table
/// library could give. The loops that fill a table beforehand are too short
/// or too shallow to record, or make tables, which is no fault of the call.
fn assert_no_hard_aborts(profile: CompatibilityProfile, source: &str, stats: &TraceStatistics) {
    for name in ["table.insert", "table.remove"] {
        assert_eq!(
            stats.native_aborts.of(name),
            0,
            "{profile:?} {source}: {:?}",
            stats.native_aborts
        );
    }
    for reason in TraceAbortReason::ALL {
        if matches!(
            reason,
            TraceAbortReason::ShortLoop
                | TraceAbortReason::LoopNotClosed
                | TraceAbortReason::InnerLoop
                | TraceAbortReason::VirtualEscape
        ) {
            continue;
        }
        assert_eq!(
            stats.recording_aborts[reason.index()],
            0,
            "{profile:?} {source}: {reason:?} {stats:?}"
        );
    }
}

/// Runs `source` in both modes of each profile and checks that the loops
/// traced and ran natively with no abort, that the results agree, and that
/// the traces call the helpers named in `helpers` and no others.
fn assert_native(source: &str, helpers: &[&str]) {
    for profile in ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.traces_compiled > 0, "{profile:?} {source} {stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert_no_hard_aborts(profile, source, &stats);
        let mut called = lua.vm().trace_helper_calls();
        called.sort_unstable();
        called.dedup();
        // A number is an integer in the profiles that have them.
        let number = match profile {
            CompatibilityProfile::Lua53
            | CompatibilityProfile::Lua54
            | CompatibilityProfile::Lua55 => "table.remove_integer",
            _ => "table.remove_float",
        };
        let helpers: Vec<&str> = helpers
            .iter()
            .map(|name| {
                if *name == "table.remove_number" {
                    number
                } else {
                    name
                }
            })
            .collect();
        for name in &helpers {
            assert!(
                called.contains(name),
                "{profile:?} {source}: {called:?} lacks {name}"
            );
        }
        for name in &called {
            assert!(
                helpers.contains(name),
                "{profile:?} {source}: unexpected helper {name}"
            );
        }
    }
}

/// Like [`assert_native`] for a loop that meets something the recording did
/// not see: the trace runs natively, leaves through a guard, and the
/// interpreter finishes the loop, raising its error if it has one.
fn assert_exits(source: &str) {
    for profile in ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert!(
            stats.exits_of(TraceExitKind::GuardFail)
                + stats.exits_of(TraceExitKind::CommittedEffect)
                > 0,
            "{profile:?} {source} {stats:?}"
        );
    }
}

/// Checks only that both modes agree and that native code ran.
fn assert_traced(source: &str) {
    for profile in ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
    }
}

/// Like [`assert_traced`], and the traces must call the `DELETE_STRING` and
/// `SET_REVIVE` helpers (`min_delete`, `min_revive` times at least), so the
/// cold path of the fused store ran inside native code.
fn assert_slow_path(source: &str, min_delete: u64, min_revive: u64) {
    for profile in ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert!(
            stats.delete_string_helper_calls >= min_delete,
            "{profile:?} {source} {stats:?}"
        );
        assert!(
            stats.set_revive_helper_calls >= min_revive,
            "{profile:?} {source} {stats:?}"
        );
    }
}

/// Like [`assert_traced`], and at least `iterations` loop iterations must
/// complete inside native code, so the loops of the program (not just one of
/// them) ran natively.
fn assert_clean(source: &str, iterations: u64) {
    for profile in ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(
            stats.iterations >= iterations,
            "{profile:?} {source} {stats:?}"
        );
    }
}

// append and pop

#[test]
fn append_and_pop_in_hot_loops() {
    assert_native(
        "local t = {} \
         for i = 1, 100000 do table.insert(t, i) end \
         local s = 0 \
         for i = 1, 100000 do s = s + table.remove(t) end \
         return s, #t",
        &[],
    );
}

#[test]
fn a_list_grows_past_its_capacity() {
    assert_native(
        "local t = {} \
         for i = 1, 300000 do table.insert(t, i * 0.5) end \
         return #t, t[1], t[150000], t[300000]",
        &[],
    );
}

#[test]
fn popping_to_empty_gives_nil() {
    // The loop pops past the end: once the list is empty, `remove` gives nil
    // and changes nothing, and the sum of nothing is the same.
    assert_traced(
        "local t = {} for i = 1, 1000 do t[i] = i end \
         local s, nils = 0, 0 \
         for i = 1, 100000 do \
           local v = table.remove(t) \
           if v == nil then nils = nils + 1 else s = s + v end \
         end \
         return s, nils, #t",
    );
}

#[test]
fn a_remove_of_an_empty_list_is_a_helper_that_changes_nothing() {
    assert_native(
        "local t = {} local n = 0 \
         for i = 1, 100000 do if table.remove(t) == nil then n = n + 1 end end \
         return n, #t",
        &["table.remove_nothing"],
    );
}

#[test]
fn booleans_and_tables_come_back_from_a_pop() {
    assert_native(
        "local flags = {} for i = 1, 100000 do flags[i] = i > 0 end \
         local n = 0 \
         for i = 1, 100000 do if table.remove(flags) then n = n + 1 end end \
         local boxes = {} for i = 1, 100 do boxes[i] = {v = i} end \
         local s = 0 \
         for round = 1, 100000 do \
           local b = table.remove(boxes) s = s + b.v table.insert(boxes, 1, b) \
         end \
         return n, s, #boxes, boxes[1].v, boxes[100].v",
        &["table.remove_table", "table.insert"],
    );
}

#[test]
fn strings_come_back_from_a_pop() {
    assert_native(
        "local t = {} for i = 1, 100000 do t[i] = 'k' .. (i % 10) end \
         local n = 0 \
         for i = 1, 100000 do local s = table.remove(t) n = n + #s end \
         local u = {'a', 'b', 'c'} local text = '' \
         for i = 1, 100000 do \
           local s = table.remove(u, 1) table.insert(u, s) \
           if i > 99990 then text = text .. s end \
         end \
         return n, #t, text, u[1]",
        &["concat", "table.remove_string"],
    );
}

#[test]
fn strings_go_in() {
    assert_traced(
        "local t = {} \
         for i = 1, 100000 do table.insert(t, 'k' .. i) end \
         for i = 1, 100000 do table.insert(t, 'constant') end \
         return #t, t[1], t[100000], t[200000]",
    );
}

// positions

#[test]
fn positional_insert_and_remove_at_front_middle_and_end() {
    assert_native(
        "local t = {} for i = 1, 64 do t[i] = i end \
         local s = 0 \
         for i = 1, 100000 do \
           local n = #t \
           table.insert(t, 1, i) \
           table.insert(t, n // 2 + 1, i) \
           table.insert(t, #t + 1, i) \
           s = s + table.remove(t, 1) + table.remove(t, #t // 2) + table.remove(t, #t) \
         end \
         local sum = 0 for i = 1, #t do sum = sum + t[i] end \
         return s, sum, #t"
            .replace("n // 2", "(n - n % 2) / 2")
            .replace("#t // 2", "(#t - #t % 2) / 2")
            .as_str(),
        &["table.insert", "table.remove_number"],
    );
}

#[test]
fn a_fractional_position_is_truncated() {
    assert_traced(
        "local t = {} for i = 1, 10 do t[i] = i end \
         local s = 0 \
         for i = 1, 100000 do \
           table.insert(t, 2.5, i) \
           s = s + table.remove(t, 2.9) \
         end \
         return s, #t, t[1], t[2], t[3]",
    );
}

// reads next to writes

#[test]
fn reads_of_the_length_and_the_elements_see_each_write() {
    assert_native(
        "local t = {1, 2, 3} local s = 0 \
         for i = 1, 100000 do \
           table.insert(t, i) \
           s = s + #t + t[#t] \
           table.insert(t, 1, i) \
           s = s + #t + t[1] + t[#t] \
           s = s + table.remove(t) + #t \
           s = s + table.remove(t, 1) + #t + t[#t] \
         end \
         return s, #t",
        &["table.insert", "table.remove_number"],
    );
}

#[test]
fn the_border_after_a_store_is_read_again() {
    // `t[#t + 1] = v` changes the border, which the trace now reads again.
    assert_native(
        "local t = {} \
         for i = 1, 100000 do t[#t + 1] = i end \
         local s = 0 for i = 1, #t do s = s + t[i] end \
         return #t, s",
        &[],
    );
}

#[test]
fn two_names_for_one_table_see_the_same_border() {
    assert_native(
        "local t = {} local u = t local s = 0 \
         for i = 1, 100000 do \
           table.insert(u, i) \
           table.insert(t, 1, i) \
           s = s + #t + #u \
           s = s + table.remove(t) + #u \
           s = s + table.remove(t, 1) + #u \
         end \
         return s, #t",
        &["table.insert", "table.remove_number"],
    );
}

#[test]
fn a_loop_over_the_elements_while_the_list_changes_size() {
    assert_native(
        "local t = {} for i = 1, 50 do t[i] = i end \
         local s = 0 \
         for i = 1, 100000 do \
           table.insert(t, i) \
           table.remove(t, 1) \
           s = s + t[1] + t[#t] \
         end \
         return s, #t",
        &["table.remove_number"],
    );
}

// holes and nil

#[test]
fn a_list_with_a_hole_is_left_to_the_interpreter() {
    // The border of `t` is 3 and the table has a hash part after the hole:
    // the helper declines it, and the answer is the interpreter's.
    assert_traced(
        "local t = {1, 2, 3, nil, 5, 6} local s = 0 \
         for i = 1, 100000 do \
           table.insert(t, i) \
           s = s + #t + table.remove(t) \
         end \
         return s, #t, t[5], t[6]",
    );
}

#[test]
fn nil_values_and_nil_fields() {
    assert_traced(
        "local t = {n = 0} local s = 0 \
         for i = 1, 100000 do \
           table.insert(t, i) \
           t.n = t.n + 1 \
           if i % 7 == 0 then table.remove(t) end \
           s = s + #t \
         end \
         return s, #t, t.n",
    );
}

#[test]
fn a_nil_inserted_changes_nothing_at_the_end() {
    // The recording sees a number; the value turns nil late in the run.
    assert_exits(
        "local t = {} local v = 1 \
         for i = 1, 100000 do \
           if i == 90000 then v = nil end \
           table.insert(t, v) \
         end \
         return #t",
    );
}

// stores into the array part

#[test]
fn stores_inside_at_the_end_and_outside_the_array_part() {
    assert_traced(
        "local t = {} for i = 1, 8 do t[i] = i end \
         local s = 0 \
         for i = 1, 100000 do \
           t[i % 8 + 1] = i \
           t[#t + 1] = i \
           s = s + #t \
           if #t > 100 then t[#t] = nil end \
         end \
         return s, #t, t[1], t[8], t[9]",
    );
    assert_traced(
        "local t = {} \
         for i = 1, 100000 do t[i] = i t[i + 5] = i end \
         local c = 0 for k, v in pairs(t) do c = c + 1 end \
         return #t, c",
    );
}

// exits

#[test]
fn a_metatable_gained_after_the_loop_is_hot() {
    // From Lua 5.3 on `table.insert` follows `__newindex`, `__index` and
    // `__len`; before, it is raw. Either way the answer is the interpreter's.
    for metamethod in [
        "__newindex = function(t, k, v) log = log + 1 rawset(t, k, v) end",
        "__index = function(t, k) log = log + 1 return nil end",
        "__len = function(t) log = log + 1 return rawlen and rawlen(t) or #t end",
    ] {
        let source = format!(
            "local log = 0 local t = {{}} \
             for i = 1, 100000 do \
               if i == 90000 then setmetatable(t, {{{metamethod}}}) end \
               table.insert(t, i) \
               if i % 2 == 0 then table.remove(t) end \
             end \
             return #t, log"
        );
        assert_traced(&source);
    }
}

#[test]
fn a_table_that_gains_a_metatable_leaves_the_helper() {
    // The metatable arrives from a function the loop calls, between two runs
    // of the hot loop. The next run leaves the trace at the first call, and the
    // interpreter makes it: through `__newindex` from Lua 5.3 on, raw before.
    let source = "local log = 0 local t = {} \
         local function arm() \
           setmetatable(t, {__newindex = function(t, k, v) log = log + 1 rawset(t, k, v) end}) \
         end \
         for round = 1, 3 do \
           for i = 1, 50000 do table.insert(t, i) end \
           arm() \
         end \
         return #t, log";
    for profile in ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        // The table stops being appendable in every profile: the trace
        // leaves, the interpreter makes the call (raw, or through
        // `__newindex` from Lua 5.3 on).
        assert!(
            stats.exits_of(TraceExitKind::GuardFail)
                + stats.exits_of(TraceExitKind::CommittedEffect)
                > 0,
            "{profile:?} {stats:?}"
        );
    }
}

#[test]
fn a_position_out_of_range_raises_the_interpreters_error() {
    assert_exits(
        "local t = {} \
         local ok, err = pcall(function() \
           for i = 1, 100000 do \
             local place = #t + 1 \
             if i == 90000 then place = place + 5 end \
             table.insert(t, place, i) \
           end \
         end) \
         return ok, err, #t",
    );
}

#[test]
fn a_position_of_zero_raises_the_interpreters_error() {
    assert_exits(
        "local t = {} \
         local ok, err = pcall(function() \
           for i = 1, 100000 do \
             local place = #t + 1 \
             if i == 90000 then place = 0 end \
             table.insert(t, place, i) \
           end \
         end) \
         return ok, err, #t",
    );
}

#[test]
fn a_removal_out_of_range_gives_nil_in_the_interpreter() {
    assert_exits(
        "local t = {} for i = 1, 100 do t[i] = i end local place = 1 local s = 0 \
         for i = 1, 100000 do \
           if i == 90000 then place = 1000000 end \
           table.insert(t, i) \
           local v = table.remove(t, place) \
           if v == nil then s = s + 1 else s = s + v end \
         end \
         return s, #t",
    );
}

#[test]
fn a_value_that_is_no_table_raises_the_interpreters_error() {
    assert_exits(
        "local t = {} local target = t \
         local ok, err = pcall(function() \
           for i = 1, 100000 do \
             if i == 90000 then target = 5 end \
             table.insert(target, i) \
           end \
         end) \
         return ok, err, #t",
    );
}

#[test]
fn a_replaced_insert_runs_in_the_interpreter() {
    assert_exits(
        "local t = {} local calls = 0 \
         for i = 1, 100000 do \
           if i == 90000 then table.insert = function(list, v) calls = calls + 1 end end \
           table.insert(t, i) \
         end \
         return #t, calls",
    );
}

#[test]
fn a_replaced_remove_runs_in_the_interpreter() {
    assert_exits(
        "local t = {} for i = 1, 100000 do t[i] = i end local calls = 0 local s = 0 \
         for i = 1, 100000 do \
           if i == 90000 then table.remove = function(list) calls = calls + 1 return -1 end end \
           s = s + table.remove(t) \
         end \
         return #t, calls, s",
    );
}

#[test]
fn another_kind_of_element_leaves_the_trace() {
    assert_exits(
        "local t = {} for i = 1, 100000 do t[i] = i end t[89990] = 'text' \
         local s = 0 local n = 0 \
         for i = 1, 100000 do \
           local v = table.remove(t) \
           if type(v) == 'number' then s = s + v else n = n + 1 end \
         end \
         return s, n, #t",
    );
}

// the collector

#[test]
fn tables_and_strings_survive_a_collector_that_runs_between_the_runs() {
    // Tables move between lists by insert and remove while the collector
    // steps between the traced loops. Every element must stay alive and
    // intact.
    let source = "local a, b = {}, {} \
         for i = 1, 300 do a[i] = {i, 'v' .. i} end \
         local sum = 0 \
         for round = 1, 200 do \
           for i = 1, 300 do local box = table.remove(a) table.insert(b, box) end \
           collectgarbage('step', 10) \
           for i = 1, 300 do local box = table.remove(b) table.insert(a, 1, box) end \
           collectgarbage('step', 10) \
           for i = 1, 300 do local box = a[i] sum = sum + box[1] + #box[2] end \
         end \
         collectgarbage() \
         local ok = 0 \
         for i = 1, 300 do if a[i][2] == 'v' .. a[i][1] then ok = ok + 1 end end \
         return sum, ok, #a, #b";
    assert_clean(source, 100_000);
}

#[test]
fn a_collector_with_a_low_pause_keeps_what_a_trace_appends() {
    // `keep` lives through every cycle; each round the trace puts tables made
    // a moment before into it, which the collector (stepping all the time)
    // must see, and takes some out again.
    let source = "if pcall(collectgarbage, 'setpause', 20) then collectgarbage('setstepmul', 1000) \
         else collectgarbage('incremental', 20, 1000, 0) end \
         local keep = {} local seen = 0 \
         for round = 1, 60 do \
           local fresh = {} \
           for i = 1, 2000 do fresh[i] = {i, round} end \
           for i = 1, 2000 do table.insert(keep, fresh[i]) end \
           local junk = {} for i = 1, 2000 do junk[i] = {i, i} end \
           for i = 1, 1000 do local gone = table.remove(keep) seen = seen + gone[2] end \
         end \
         collectgarbage() \
         local s, r = 0, 0 for i = 1, #keep do s = s + keep[i][1] r = r + keep[i][2] end \
         return #keep, s, r, seen";
    assert_clean(source, 100_000);
}

#[test]
fn a_collector_with_a_low_pause_keeps_the_strings_a_trace_appends() {
    // Strings made a moment before go into `keep` in place, and some come out
    // again by a pop, while the collector steps all the time. What stays must
    // be intact after a full collection.
    let source = "if pcall(collectgarbage, 'setpause', 20) then collectgarbage('setstepmul', 1000) \
         else collectgarbage('incremental', 20, 1000, 0) end \
         local keep = {} \
         for round = 1, 60 do \
           for i = 1, 3000 do keep[#keep + 1] = tostring(i * round) end \
           local junk = {} for i = 1, 2000 do junk[i] = {i, i} end \
           for i = 1, 1500 do keep[#keep] = nil end \
         end \
         collectgarbage() \
         local n, bytes = 0, 0 \
         for i = 1, #keep do bytes = bytes + #keep[i] if tonumber(keep[i]) then n = n + 1 end end \
         return #keep, n, bytes, keep[1], keep[#keep]";
    assert_clean(source, 150_000);
}

// appends and pops in place

/// Like [`assert_native`] with no helper, for a loop that also never leaves
/// its trace through a guard: the appends, pops and overwrites are the
/// trace's own, with no call into the VM and no exit.
fn assert_in_place(source: &str) {
    assert_native(source, &[]);
    for profile in ALL {
        let (_, lua) = outcome(profile, ExecutionMode::Tracing, source);
        let stats = lua.vm().jit_statistics().trace;
        assert_eq!(
            stats.exits_of(TraceExitKind::GuardFail)
                + stats.exits_of(TraceExitKind::CommittedEffect),
            0,
            "{profile:?} {source} {stats:?}"
        );
    }
}

#[test]
fn a_stack_pushes_and_pops_in_place() {
    assert_in_place(
        "local t = {} local s = 0 \
         for i = 1, 200000 do \
           table.insert(t, i) \
           table.insert(t, i * 2) \
           table.insert(t, i * 3) \
           s = s + table.remove(t) + #t + table.remove(t) \
         end \
         return s, #t, t[1], t[#t]",
    );
}

#[test]
fn indexed_appends_and_nil_pops_run_in_place() {
    assert_in_place(
        "local t = {} \
         for i = 1, 100000 do t[#t + 1] = i * 0.5 end \
         local s = 0 \
         for i = 1, 60000 do s = s + t[#t] t[#t] = nil end \
         return s, #t, t[#t], t[#t + 1]",
    );
}

#[test]
fn a_nil_append_is_a_no_op_that_stays_in_the_trace() {
    // Storing nil one past the end of a dense list changes nothing, and it
    // is not a reason to leave: not alone, and not mixed with real appends.
    assert_in_place(
        "local t = {} local n = 0 \
         for i = 1, 100000 do t[#t + 1] = nil n = n + #t end \
         return #t, n, t[1]",
    );
    assert_in_place(
        "local t = {1.5} local n = 0 \
         for i = 1, 100000 do \
           t[#t + 1] = nil \
           t[#t + 1] = i * 0.5 \
           t[#t + 1] = nil \
           n = n + #t \
         end \
         return #t, n, t[1], t[#t]",
    );
    // Strings and tables hold references, and a nil append releases none.
    assert_in_place(
        "local boxes = {} for i = 1, 10 do boxes[i] = {i} end \
         local t, b = {'x'}, {boxes[1]} local n = 0 \
         for i = 1, 100000 do \
           t[#t + 1] = nil b[#b + 1] = nil \
           t[#t + 1] = 's' b[#b + 1] = boxes[i % 10 + 1] \
           t[#t + 1] = nil b[#b + 1] = nil \
           n = n + #t + #b \
         end \
         return #t, #b, n, t[#t], b[#b][1]",
    );
}

#[test]
fn every_kind_of_scalar_goes_in_place() {
    assert_in_place(
        "local t = {} \
         for i = 1, 30000 do \
           t[#t + 1] = i \
           t[#t + 1] = i + 0.5 \
           t[#t + 1] = i % 2 == 0 \
         end \
         local ints, floats = 0, 0 \
         for i = 1, #t, 3 do ints = ints + t[i] floats = floats + t[i + 1] end \
         return #t, ints, floats, t[3], t[6], t[89999], t[90000], \
           math.type and math.type(t[1]), math.type and math.type(t[2])",
    );
}

#[test]
fn a_table_with_an_index_metamethod_only_takes_appends_in_place() {
    // `__index` and `__newindex`-free metatables are what objects have. An
    // append is a raw store whatever the profile's `table.insert` follows.
    assert_in_place(
        "local base = {extra = 1} \
         local obj = setmetatable({}, {__index = base}) \
         for i = 1, 100000 do table.insert(obj, i) end \
         local s = 0 \
         for i = 1, 90000 do s = s + table.remove(obj) end \
         return #obj, s, obj.extra, obj[1]",
    );
}

#[test]
fn strings_and_tables_go_in_place_and_come_back() {
    assert_native(
        "local boxes = {} for i = 1, 1000 do boxes[i] = {v = i} end \
         local t, b = {}, {} local n = 0 \
         for i = 1, 300000 do t[#t + 1] = tostring(i % 1000) b[#b + 1] = boxes[i % 1000 + 1] end \
         for i = 1, 100000 do n = n + b[#b].v b[#b] = nil end \
         return n, #t, #b, t[1], t[300000], b[1].v, b[#b].v",
        &["tostring"],
    );
}

#[test]
fn strings_are_overwritten_in_place_and_the_old_ones_released() {
    assert_native(
        "local t = {} for i = 1, 1000 do t[i] = 'old' .. i end \
         for round = 1, 200 do \
           for i = 1, 1000 do t[i] = 'new' .. round .. ':' .. i end \
         end \
         collectgarbage() \
         return #t, t[1], t[1000], t[500]",
        &["concat"],
    );
}

#[test]
fn strings_and_tables_pop_by_a_nil_store_without_leaving() {
    assert_in_place(
        "local t = {} local b = {} \
         local mt = {} \
         for i = 1, 200000 do t[i] = 's' .. (i % 7) b[i] = setmetatable({i}, mt) end \
         for i = 1, 150000 do t[#t] = nil b[#b] = nil end \
         return #t, #b, t[#t], b[#b][1]",
    );
}

#[test]
fn a_constant_string_is_appended_in_place() {
    assert_in_place(
        "local t = {} \
         for i = 1, 100000 do t[#t + 1] = 'constant' end \
         return #t, t[1], t[100000]",
    );
}

#[test]
fn pops_leave_the_keys_next_resumes_from() {
    // A pop deletes the last key, and `next` may be asked to go on from a
    // key a loop deleted. The keys a trace popped must be as deleted as the
    // interpreter's pops leave them.
    for popped in ["t[#t] = nil", "table.remove(t)"] {
        let source = format!(
            "local t = {{}} \
             for i = 1, 300000 do t[#t + 1] = i end \
             for i = 1, 200000 do {popped} end \
             local a = {{pcall(next, t, 250000)}} \
             local b = {{pcall(next, t, 100000)}} \
             local c = {{pcall(next, t, 300001)}} \
             local d = {{pcall(next, t, 100001)}} \
             for i = 1, 150000 do t[#t + 1] = i end \
             local e = {{pcall(next, t, 250000)}} \
             local count = 0 for k, v in pairs(t) do count = count + 1 end \
             for k in pairs(t) do t[k] = nil end \
             return #t, count, a[1], a[2], b[1], b[2], c[1], d[1], e[1], next(t)"
        );
        assert_traced(&source);
    }
}

#[test]
fn a_pop_loop_that_deletes_while_traversing() {
    assert_traced(
        "local t = {} \
         for i = 1, 300000 do t[#t + 1] = i end \
         for i = 1, 250000 do t[#t] = nil end \
         local seen = 0 \
         for k, v in pairs(t) do seen = seen + v t[k] = nil end \
         return seen, #t, next(t)",
    );
}

#[test]
fn the_interpreter_and_the_trace_share_a_table() {
    assert_traced(
        "local t = {} \
         local function push(n) for i = 1, n do t[#t + 1] = i end end \
         local function pop(n) local s = 0 for i = 1, n do s = s + table.remove(t) end return s end \
         local total = 0 \
         for round = 1, 50 do \
           push(5000) \
           total = total + pop(3000) \
           t[#t + 1] = 'x' t[#t] = nil \
           local u = {unpack and unpack(t, 1, 3) or table.unpack(t, 1, 3)} \
           total = total + #u + #t \
         end \
         return total, #t",
    );
}

#[test]
fn the_collector_is_charged_for_the_slots_a_trace_appends() {
    // A million appends are 16 MB of array slots: the count must show
    // them, or a collector paced by it would never run.
    let source = "collectgarbage() local before = collectgarbage('count') \
         local t = {} \
         for i = 1, 100000 do t[#t + 1] = i end \
         local grown = collectgarbage('count') - before \
         return grown > 1000, #t";
    for profile in ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?}");
        assert!(
            lua.vm().jit_statistics().trace.native_entries > 0,
            "{profile:?}"
        );
    }
}

#[test]
fn a_table_that_gains_a_hash_part_leaves_the_trace() {
    assert_exits(
        "local t = {} local named = 0 \
         for i = 1, 100000 do \
           if i == 60000 then t.name = 'x' end \
           t[#t + 1] = i \
           if i % 2 == 0 then t[#t] = nil end \
         end \
         return #t, t.name, t[1], t[#t]",
    );
}

#[test]
fn a_table_that_gains_a_hole_leaves_the_trace() {
    assert_exits(
        "local t = {} \
         for i = 1, 100000 do \
           t[#t + 1] = i \
           if i == 60000 then t[10] = nil end \
         end \
         return #t, t[9], t[10], t[11]",
    );
}

#[test]
fn a_table_that_goes_read_only_or_gets_a_metamethod_between_runs() {
    let source = "local log = 0 local t = {} \
         local function run(n) for i = 1, n do t[#t + 1] = i if i % 4 == 0 then t[#t] = nil end end end \
         for round = 1, 4 do \
           run(40000) \
           if round == 2 then setmetatable(t, {__newindex = function(t, k, v) log = log + 1 rawset(t, k, v) end}) end \
         end \
         return #t, log";
    assert_traced(source);
}

#[test]
fn a_float_key_names_the_slot_after_the_last() {
    // Lua 5.1 and LuaJIT index with floats: `#t + 1` and a loop counter are
    // floats that name integers.
    assert_traced(
        "local t = {} \
         for i = 1, 100000 do t[i * 1.0] = i end \
         for i = 1, 50000 do t[#t * 1.0] = nil end \
         local k = 0.5 \
         for i = 1, 1000 do t[#t + 1] = i end \
         return #t, t[1], t[#t], t[50001]",
    );
}

// a delete and a store of one key, fused

#[test]
fn a_key_deleted_and_stored_again_keeps_the_traversal_order() {
    // The hash_churn shape: each iteration deletes a key and stores a number
    // into it. The traversal after the loop must list the keys in the
    // interpreter's order, and the live count must be the interpreter's.
    assert_traced(
        "local keys = {} local t = {} \
         for i = 1, 101 do keys[i] = 'k' .. i t[keys[i]] = i end \
         for i = 1, 200000 do \
           local k = keys[(i - 1) % 101 + 1] \
           t[k] = nil \
           t[k] = i % 101 \
         end \
         local n, order = 0, {} \
         for k in pairs(t) do n = n + 1 order[#order + 1] = k end \
         return n, t.k1, t.k101, table.concat(order, ',')",
    );
}

#[test]
fn a_live_boolean_is_replaced_by_a_float_through_the_fused_store() {
    // The entries start as booleans, so the first stores change the type of
    // a live entry and the later ones take floats in place.
    assert_traced(
        "local t = {} local keys = {} \
         for i = 1, 50 do keys[i] = 'b' .. i t[keys[i]] = true end \
         for i = 1, 100000 do \
           local k = keys[i % 50 + 1] \
           t[k] = nil \
           t[k] = i * 0.5 \
         end \
         local s = 0 \
         for i = 1, 50 do s = s + t[keys[i]] end \
         return s, type(t.b1), type(t.b50)",
    );
}

#[test]
fn a_key_absent_on_some_iterations_goes_through_both_helpers() {
    // The table is emptied now and then, so the deleted keys are absent for a
    // while: those stores go through the delete and the store helper in turn.
    assert_traced(
        "local t = {} local keys = {} for i = 1, 64 do keys[i] = 'm' .. i end \
         local s = 0 \
         for i = 1, 150000 do \
           if i % 1000 == 0 then t = {} end \
           local k = keys[i % 64 + 1] \
           t[k] = nil \
           t[k] = i % 2 == 0 \
           if t[k] then s = s + 1 end \
         end \
         local n = 0 for _ in pairs(t) do n = n + 1 end \
         return s, n, t.m1, t.m64",
    );
}

#[test]
fn an_underscore_key_in_a_plain_table_goes_through_both_helpers() {
    // A key that may name a metamethod is never stored in place, even in a
    // table with no metatable.
    assert_traced(
        "local t = {} local keys = {'__x', '__gc', 'plain', '__index'} \
         for i = 1, 100000 do \
           local k = keys[i % 4 + 1] \
           t[k] = nil \
           t[k] = i * 1.0 \
         end \
         return t.__x, t.__gc, t.plain, t.__index",
    );
}

#[test]
fn a_table_with_a_newindex_sees_the_deletes_and_the_stores() {
    // A metatable stays on the slow path: the store of an absent key calls
    // __newindex, and the count of those calls must match the interpreter's.
    assert_traced(
        "local log = 0 local t = setmetatable({}, {__newindex = function(t, k, v) \
           log = log + 1 rawset(t, k, v) end}) \
         local keys = {} for i = 1, 16 do keys[i] = 'n' .. i end \
         for i = 1, 100000 do \
           local k = keys[i % 16 + 1] \
           t[k] = nil \
           t[k] = i % 7 == 0 and 1.25 or 2.5 \
         end \
         return log, t.n1, t.n16",
    );
}

#[test]
fn a_guard_between_the_delete_and_the_store_keeps_both() {
    // The branch between the two stores can leave the trace. Each side must
    // see the key deleted, so the pair is not fused.
    assert_traced(
        "local t = {} local keys = {} \
         for i = 1, 32 do keys[i] = 'g' .. i t[keys[i]] = 1.5 end \
         local s = 0 \
         for i = 1, 100000 do \
           local k = keys[i % 32 + 1] \
           t[k] = nil \
           if i % 3 == 0 then s = s + 1 end \
           t[k] = i * 0.25 \
         end \
         return s, t.g1, t.g32",
    );
}

#[test]
fn a_live_string_deleted_and_given_a_number_takes_the_slow_path() {
    // Each fused pair finds a live string, so DELETE_STRING and SET_REVIVE
    // run through the stub inside the trace, with the floats live. The table
    // after the loop must match the interpreter's.
    assert_slow_path(
        "local t = {} local keys = {} \
         for i = 1, 40 do keys[i] = 's' .. i end \
         for i = 1, 40 do t[keys[i]] = 1 end \
         local a, b, c = 0.5, 1.25, 3.5 \
         for i = 1, 100000 do \
           local k = keys[i % 40 + 1] \
           t[k] = nil \
           t[k] = i * 0.5 + a \
           t[keys[(i + 17) % 40 + 1]] = 'w' \
           a = a * 0.75 + b b = b + a * 1e-3 c = c * 1.0000001 + a \
         end \
         local n, s = 0, 0 \
         for k, v in pairs(t) do n = n + 1 if type(v) == 'number' then s = s + v end end \
         return a, b, c, n, s, t.s1, t.s40",
        1000,
        1000,
    );
}

#[test]
fn churn_then_fresh_inserts_keep_the_live_count_exact() {
    // The fused revive raises the live-entry counter; a wrong count would
    // trigger a rehash too early or too late, or lose entries.
    assert_traced(
        "local t = {} local keys = {} \
         for i = 1, 101 do keys[i] = 'c' .. i t[keys[i]] = i end \
         for i = 1, 100000 do \
           local k = keys[i % 101 + 1] \
           t[k] = nil \
           t[k] = i % 13 + 0.5 \
         end \
         for i = 1, 5000 do t['f' .. i] = i end \
         for i = 1, 101 do t[keys[i]] = nil end \
         for i = 1, 5000 do t['g' .. i] = i end \
         local n, s = 0, 0 \
         for k, v in pairs(t) do n = n + 1 s = s + v end \
         return n, s",
    );
}

#[test]
fn an_integer_stored_through_the_fused_store_keeps_its_type() {
    // Lua 5.3 and later keep integers apart from floats; the stored tag must
    // be the integer one.
    assert_traced(
        "local t = {} local keys = {} \
         for i = 1, 30 do keys[i] = 'i' .. i t[keys[i]] = 0.5 end \
         for i = 1, 100000 do \
           local k = keys[i % 30 + 1] \
           t[k] = nil \
           t[k] = i % 7 \
         end \
         local s = 0 \
         for i = 1, 30 do s = s + t[keys[i]] end \
         return s, t.i1, t.i1 / 2, t.i30",
    );
}

// cold helper calls through the register-preserving stub

#[test]
fn live_floats_survive_a_slow_store_taken_on_some_iterations() {
    // Fresh keys per iteration and deletes ahead make rehashes drop
    // tombstones, so some fused pairs find the key and others miss. The
    // floats the loop keeps in registers must come back whole from every
    // helper call.
    assert_slow_path(
        "local t = {} local keys = {} \
         for i = 1, 40 do keys[i] = 'z' .. i end \
         for i = 1, 40 do t[keys[i]] = 1 end \
         local a, b, c = 0.5, 1.25, 3.5 \
         for i = 1, 100000 do \
           local k = keys[i % 40 + 1] \
           t[keys[(i + 9) % 40 + 1]] = nil \
           t[k] = nil \
           t[k] = i * 0.5 + a \
           t['f' .. i] = 1.0 \
           a = a * 0.75 + b b = b + a * 1e-3 c = c * 1.0000001 + a \
         end \
         local n, s = 0, 0 \
         for k, v in pairs(t) do n = n + 1 s = s + v end \
         return a, b, c, n, s",
        1000,
        1000,
    );
}

#[test]
fn a_helper_that_leaves_through_its_guard_after_the_stub_resumes_the_interpreter() {
    // The same shape with deletes ahead and a fresh key per iteration: a
    // rehash drops tombstones, a later fused pair misses, and the SET_REVIVE
    // helper leaves the trace through its guard. The interpreter resumes
    // and the results agree.
    let source = "local t = {} local keys = {} \
         for i = 1, 40 do keys[i] = 'x' .. i end \
         for i = 1, 40 do t[keys[i]] = 1 end \
         local a, b, c = 0.5, 1.25, 3.5 \
         for i = 1, 100000 do \
           local k = keys[i % 40 + 1] \
           t[keys[(i + 9) % 40 + 1]] = nil \
           t[k] = nil \
           t[k] = i * 0.5 + a \
           t['f' .. i] = 1.0 \
           t[keys[(i + 17) % 40 + 1]] = 'w' \
           a = a * 0.75 + b b = b + a * 1e-3 c = c * 1.0000001 + a \
         end \
         local n, s = 0, 0 \
         for k, v in pairs(t) do n = n + 1 if type(v) == 'number' then s = s + v end end \
         return a, b, c, n, s";
    assert_slow_path(source, 1000, 1000);
    for profile in ALL {
        let (_, lua) = outcome(profile, ExecutionMode::Tracing, source);
        let stats = lua.vm().jit_statistics().trace;
        assert!(
            stats.exits[TraceExitKind::GuardFail.index()] > 0,
            "{profile:?} {stats:?}"
        );
    }
}

/// A traced loop appends fresh tables to a list that a collection already
/// made old, with the constructor's garbage between the stores. The native
/// store reports the slot it wrote (`table_native_stored_slot`), so each minor
/// collection traces only the slots stored since the last one; every element
/// must still read back, in all profiles.
#[test]
fn traced_appends_of_fresh_tables_to_an_old_list_survive_many_minors() {
    let source = r"
        local list = {}
        for i = 1, 8 do list[i] = {i} end
        collectgarbage()
        local n = #list
        for i = 1, 60000 do
            local t = {i, i + 1}
            local garbage = {i, i, i}
            n = n + 1
            list[n] = t
        end
        local bad = 0
        for j = 1, 8 do
            if list[j][1] ~= j then bad = bad + 1 end
        end
        for j = 9, n do
            local t = list[j]
            local i = j - 8
            if t[1] ~= i or t[2] ~= i + 1 then bad = bad + 1 end
        end
        return n, bad
    ";
    for profile in ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        let Ok(values) = &actual else { unreachable!() };
        assert_eq!(values.len(), 2, "{profile:?}");
        assert!(
            matches!(values[1], Value::Integer(0)) || values[1] == Value::from(0.0),
            "{profile:?}: {values:?}"
        );
    }
}
