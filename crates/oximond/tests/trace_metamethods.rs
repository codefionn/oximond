//! Function metamethods of table accesses in traces: `__index` for an absent
//! key, `__newindex` for a new one, and `__len` for `#t`.
//!
//! A handler that is a Lua closure runs inside the trace as an inlined call.
//! The loops compile, enter native code, stay in it for nearly every
//! iteration, and give the interpreter's results in every profile. The check
//! of which handler runs is about the metatable (a class guard compares the
//! receiver's metatable with the one the trace pinned, or a helper walks the
//! metatables as the interpreter does), so one trace serves every table that
//! shares a class. A table that has the key itself, a metatable swapped or a
//! handler replaced after the loop is hot leaves the trace (before the access,
//! or at the next entry for a class whose handler changed), and the
//! interpreter makes the access.

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

/// The profiles where `#t` asks `__len`.
const LEN_ASKED: [CompatibilityProfile; 4] = [
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
];

/// The profiles where `#t` takes the border whatever `__len` says.
const LEN_IGNORED: [CompatibilityProfile; 2] =
    [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21];

/// The profiles with `rawlen`, `table.unpack` and `table.move`.
const TABLE_LIBRARY: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
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

/// Checks that no recording was abandoned because the recorder could not
/// follow an access. The loops that fill a table beforehand are too short or
/// make tables, which is no fault of the access.
fn assert_no_hard_aborts(profile: CompatibilityProfile, source: &str, stats: &TraceStatistics) {
    for reason in [
        TraceAbortReason::UnsupportedOperand,
        TraceAbortReason::UnsupportedOpcode,
        TraceAbortReason::TableMiss,
        TraceAbortReason::Divergence,
        TraceAbortReason::CompileRejected,
        TraceAbortReason::LimitExceeded,
        TraceAbortReason::BorrowOutlivesEffect,
        TraceAbortReason::SelfCall,
    ] {
        assert_eq!(
            stats.recording_aborts[reason.index()],
            0,
            "{profile:?} {source}: {reason:?} {stats:?}"
        );
    }
    assert_eq!(
        stats.native_aborts.iter().count(),
        0,
        "{profile:?} {source}: {:?}",
        stats.native_aborts
    );
}

/// Runs `source` in both modes of each of `profiles` and checks that the
/// loops traced with no abort, that at least `iterations` loop iterations
/// completed in native code (so the metamethod ran inside the trace instead
/// of leaving it), that the results agree, and that the trace calls the
/// helpers named in `helpers` or checks a class guard in place of a handler
/// helper among them (see [`picked_by_class`]).
fn assert_inlined(
    profiles: &[CompatibilityProfile],
    source: &str,
    iterations: u64,
    helpers: &[&str],
) {
    for &profile in profiles {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.traces_compiled > 0, "{profile:?} {source} {stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert!(
            stats.iterations >= iterations,
            "{profile:?} {source}: {} iterations {stats:?}",
            stats.iterations
        );
        assert_no_hard_aborts(profile, source, &stats);
        let called = lua.vm().trace_helper_calls();
        let classes = lua.vm().trace_class_guard_events();
        for name in helpers {
            assert!(
                called.contains(name) || picked_by_class(name, &classes),
                "{profile:?} {source}: {called:?} lacks {name}, class guards {classes:?}"
            );
        }
    }
}

/// Whether the class guards of the installed traces, whose metatable events
/// are `classes`, stand for the handler helper `name`. A handler of the
/// receiver's own metatable is picked by a class guard (a compare of the
/// table's metatable with the one the trace pinned) instead of a helper that
/// walks the metatables, and only for the events of that helper.
fn picked_by_class(name: &str, classes: &[String]) -> bool {
    classes.iter().any(|event| match name {
        "index.handler" => event == "__index",
        "newindex.handler" => event == "__newindex",
        "operator.handler" => !matches!(event.as_str(), "__index" | "__newindex" | "__len"),
        _ => false,
    })
}

/// Like [`assert_inlined`] for a loop that meets something the recording did
/// not see: the trace runs natively, leaves through a guard, and the
/// interpreter makes the access, so the results still agree.
fn assert_exits(profiles: &[CompatibilityProfile], source: &str, iterations: u64) {
    assert_leaves(profiles, source, iterations, &[TraceExitKind::GuardFail]);
}

/// Like [`assert_exits`], where the trace may leave through any of `kinds`.
fn assert_leaves(
    profiles: &[CompatibilityProfile],
    source: &str,
    iterations: u64,
    kinds: &[TraceExitKind],
) {
    for &profile in profiles {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert!(
            kinds.iter().any(|kind| stats.exits_of(*kind) > 0),
            "{profile:?} {source} {stats:?}"
        );
        assert!(
            stats.iterations >= iterations,
            "{profile:?} {source}: {} iterations {stats:?}",
            stats.iterations
        );
    }
}

/// Like [`assert_agree`], and the trace ran natively and left through a guard
/// inside the handler after the handler had stored: the exit rebuilds the
/// handler's frame, and the error that follows has the interpreter's message
/// and traceback.
fn assert_rebuilt(profiles: &[CompatibilityProfile], source: &str) {
    for &profile in profiles {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert!(
            stats.exits_of(TraceExitKind::CommittedEffect) > 0,
            "{profile:?} {source} {stats:?}"
        );
    }
}

/// Like [`assert_agree`], and the recording was refused with the native
/// function `name` among the native aborts: no compiled trace leaves at every
/// entry for lack of a native handler.
fn assert_aborts_native(profiles: &[CompatibilityProfile], source: &str, name: &str) {
    for &profile in profiles {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(
            stats
                .native_aborts
                .iter()
                .any(|(known, count)| known == name && count > 0),
            "{profile:?} {source}: {:?}",
            stats.native_aborts
        );
        assert!(
            stats.native_entries == 0 || stats.iterations > 0,
            "{profile:?} {source}: a trace that exits at every entry {stats:?}"
        );
    }
}

/// Checks only that both modes agree (an error included), for a program the
/// recorder may or may not take.
fn assert_agree(profiles: &[CompatibilityProfile], source: &str) {
    for &profile in profiles {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, _) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
    }
}

// __index

#[test]
fn a_function_index_answers_absent_keys_and_a_present_key_reads_raw() {
    assert_inlined(
        &ALL,
        "local base = {a = 1, b = 2} \
         local mt = {__index = function(t, k) return base[k] end} \
         local o = setmetatable({own = 10}, mt) \
         local s = 0 \
         for i = 1, 100000 do s = s + o.own + o.a + o.b end \
         return s",
        90000,
        &["index.handler"],
    );
}

#[test]
fn a_function_index_answers_run_time_keys_that_stay_absent() {
    assert_inlined(
        &ALL,
        "local fallback = {alpha = 3, beta = 5, gamma = 7, delta = 11} \
         local keys = {'alpha', 'beta', 'missing_a', 'gamma', 'delta', 'missing_b'} \
         local object = setmetatable({}, {__index = function(_, key) return fallback[key] end}) \
         local total, missing = 0, 0 \
         for i = 1, 100000 do \
           local value = object[keys[(i - 1) % 6 + 1]] \
           if value then total = total + value else missing = missing + 1 end \
         end \
         return total, missing",
        90000,
        &["index.handler"],
    );
}

#[test]
fn a_function_index_reads_present_run_time_keys_raw() {
    assert_inlined(
        &ALL,
        "local object = setmetatable({a = 1, b = 100, c = 2}, \
           {__index = function(_, key) return 7 end}) \
         local keys = {'a', 'b', 'x', 'c', 'y', 'b'} \
         local total = 0 \
         for i = 1, 100000 do \
           total = total + object[keys[(i - 1) % 6 + 1]] \
         end \
         return total",
        90000,
        &["index.handler"],
    );
}

#[test]
fn a_function_index_gives_its_first_value_or_nil() {
    let source = "local mt = {__index = function(t, k) \
                    if k == 'none' then return end \
                    if k == 'nil' then return nil, 5 end \
                    return 2, 9, 10 end} \
         local o = setmetatable({}, mt) \
         local s, nils = 0, 0 \
         for i = 1, 100000 do \
           local a = o.some \
           local b = o.none \
           local c = o['nil'] \
           if b == nil and c == nil then nils = nils + 1 end \
           s = s + a \
         end \
         return s, nils";
    assert_inlined(&ALL, source, 90000, &["index.handler"]);
}

#[test]
fn a_function_index_gives_a_non_number_that_the_loop_tests() {
    assert_inlined(
        &ALL,
        "local o = setmetatable({}, {__index = function(t, k) return k == 'yes' end}) \
         local s = 0 \
         for i = 1, 100000 do \
           if o.yes then s = s + 1 end \
           if not o.no then s = s + 1 end \
         end \
         return s",
        90000,
        &["index.handler"],
    );
}

#[test]
fn a_function_index_takes_number_keys() {
    assert_inlined(
        &ALL,
        "local o = setmetatable({}, {__index = function(t, i) return i * 2 end}) \
         local s = 0 \
         for i = 1, 100000 do s = s + o[i] + o[i + 0.5] end \
         return s",
        90000,
        &["index.handler"],
    );
}

#[test]
fn a_function_index_runs_in_a_chain_of_tables() {
    // The chain ends in a function: the handler is called with the table
    // that holds it, which the helper finds.
    let source = "local top = setmetatable({}, {__index = function(t, k) return k == 'q' and 5 or 6 end}) \
         local mid = setmetatable({}, {__index = top}) \
         local o = setmetatable({}, {__index = mid}) \
         local s = 0 \
         for i = 1, 100000 do s = s + o.q + o.r end \
         return s";
    assert_inlined(&ALL, source, 90000, &["index.handler"]);
}

#[test]
fn a_function_index_reads_a_table_that_has_its_own_index_chain() {
    let source = "local base = setmetatable({}, {__index = {m = 7}}) \
         local f = setmetatable({}, {__index = function(t, k) return base[k] end}) \
         local s = 0 \
         for i = 1, 100000 do s = s + f.m + f.m end \
         return s";
    assert_inlined(&ALL, source, 90000, &["index.handler"]);
}

#[test]
fn a_function_index_that_uses_the_table_it_was_called_with() {
    // The handler of a table further down the chain gets that table: whether
    // the trace takes it or the interpreter does, the result is the same.
    assert_agree(
        &ALL,
        "local top = setmetatable({tag = 4}, {__index = function(t, k) return rawget(t, 'tag') end}) \
         local o = setmetatable({}, {__index = top}) \
         local s = 0 \
         for i = 1, 100000 do s = s + o.anything end \
         return s",
    );
}

#[test]
fn a_function_index_calls_another_function() {
    let source = "local inner = setmetatable({}, {__index = function(t, k) return 3 end}) \
         local outer = setmetatable({}, {__index = function(t, k) return inner[k] + 1 end}) \
         local s = 0 \
         for i = 1, 100000 do s = s + outer.a end \
         return s";
    assert_inlined(&ALL, source, 90000, &["index.handler"]);
}

#[test]
fn a_function_index_with_varargs_or_a_tail_call() {
    assert_inlined(
        &ALL,
        "local va = setmetatable({}, {__index = function(...) return select('#', ...) end}) \
         local tc = setmetatable({}, {__index = function(t, k) return math.floor(2.5) end}) \
         local s = 0 \
         for i = 1, 100000 do s = s + va.x + tc.y end \
         return s",
        90000,
        &["index.handler"],
    );
}

#[test]
fn the_objects_of_one_class_share_a_trace() {
    // Many tables, one metatable: the helper checks the class.
    let source = "local fallback = {x = 1, y = 2} \
         local class = {__index = function(t, k) return fallback[k] end} \
         local objects = {} \
         for i = 1, 32 do objects[i] = setmetatable({}, class) end \
         local s = 0 \
         for i = 1, 100000 do \
           local o = objects[i % 32 + 1] \
           s = s + o.x + o.y \
         end \
         return s";
    assert_inlined(&ALL, source, 90000, &["index.handler"]);
}

#[test]
fn another_class_leaves_the_trace_and_the_results_agree() {
    // The first 32 tables are of one class, the next of another.
    let source = "local first = {__index = function(t, k) return 1 end} \
         local second = {__index = function(t, k) return 100 end} \
         local objects = {} \
         for i = 1, 64 do objects[i] = setmetatable({}, i <= 32 and first or second) end \
         local s = 0 \
         for i = 1, 100000 do \
           local o = objects[(i - 1) % 64 + 1] \
           s = s + o.x \
         end \
         return s";
    assert_exits(&ALL, source, 1000);
}

#[test]
fn a_handler_replaced_after_the_loop_is_hot_leaves_the_trace() {
    assert_exits(
        &ALL,
        "local mt = {__index = function(t, k) return 1 end} \
         local o = setmetatable({}, mt) \
         local s = 0 \
         for i = 1, 100000 do \
           s = s + o.x \
           if i == 50000 then mt.__index = function(t, k) return 2 end end \
         end \
         return s",
        45000,
    );
}

#[test]
fn a_metatable_swapped_after_the_loop_is_hot_leaves_the_trace() {
    assert_exits(
        &ALL,
        "local mt1 = {__index = function(t, k) return 10 end} \
         local mt2 = {__index = function(t, k) return 20 end} \
         local p = setmetatable({}, mt1) \
         local s = 0 \
         for i = 1, 100000 do \
           s = s + p.y \
           if i == 50000 then setmetatable(p, mt2) end \
         end \
         return s",
        45000,
    );
}

#[test]
fn a_key_that_becomes_present_is_read_raw() {
    assert_exits(
        &ALL,
        "local mt = {__index = function(t, k) return 10 end} \
         local q = setmetatable({}, mt) \
         local s = 0 \
         for i = 1, 100000 do \
           s = s + q.z \
           if i == 50000 then q.z = 1 end \
         end \
         return s",
        45000,
    );
}

#[test]
fn a_handler_that_becomes_a_table_or_nothing_leaves_the_trace() {
    assert_exits(
        &ALL,
        "local mt = {__index = function(t, k) return 3 end} \
         local r = setmetatable({}, mt) \
         local s = 0 \
         for i = 1, 100000 do \
           s = s + r.w \
           if i == 50000 then mt.__index = {w = 4} end \
         end \
         return s",
        45000,
    );
    assert_exits(
        &ALL,
        "local mt = {__index = function(t, k) return 3 end} \
         local u = setmetatable({}, mt) \
         local s = 0 \
         for i = 1, 100000 do \
           s = s + (u.w or 5) \
           if i == 50000 then mt.__index = nil end \
         end \
         return s",
        45000,
    );
}

#[test]
fn an_error_in_a_function_index_is_caught_by_a_pcall_in_the_loop() {
    // The `pcall` inlines as a protected frame, the handler inside it, and
    // the error in the handler is caught on the way out of both.
    let source = "local mt = {__index = function(t, k) \
                    if k == 4 then error('boom ' .. k) end return 1 end} \
         local o = setmetatable({}, mt) \
         local function get(t, k) return t[k] end \
         local s, e = 0, 0 \
         for i = 1, 100000 do \
           local ok, v = pcall(get, o, i % 4 + 1) \
           if ok then s = s + v else e = e + 1 end \
         end \
         return s, e";
    for profile in ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert!(stats.iterations >= 90000, "{profile:?} {stats:?}");
        assert_no_hard_aborts(profile, source, &stats);
    }
}

#[test]
fn an_error_in_a_function_index_that_nothing_catches_in_the_loop() {
    // The late error comes out of the handler with the interpreter's message
    // and traceback, whatever state the trace left the frames in.
    assert_rebuilt(
        &ALL,
        "local cnt = {n = 0} \
         local q = setmetatable({}, {__index = function(t, k) \
           cnt.n = cnt.n + 1 \
           if cnt.n == 90000 then error('late') end \
           return 1 end}) \
         local function run() local s = 0 for i = 1, 100000 do s = s + q.y end return s end \
         local ok, e = xpcall(run, debug.traceback) \
         return ok, e, cnt.n",
    );
}

#[test]
fn a_strict_environment_answers_a_missing_global() {
    assert_inlined(
        &ALL,
        "setmetatable(_G, {__index = function(t, k) return 42 end}) \
         local s = 0 \
         for i = 1, 100000 do s = s + undefined_name end \
         return s",
        90000,
        &["index.handler"],
    );
}

#[test]
fn reads_around_a_handler_see_what_the_handler_wrote() {
    // The handler stores into tables the loop reads before and after it: no
    // read moves across it, and a border read after it sees its append.
    assert_inlined(
        &ALL,
        "local t = {x = 0} \
         local o = setmetatable({}, {__index = function(_, k) t.x = t.x + 1 return t.x end}) \
         local s = 0 \
         for i = 1, 100000 do \
           local a = t.x \
           local b = o.foo \
           local c = t.x \
           local d = rawget(t, 'x') \
           s = s + a + b + c + d \
         end \
         return s, t.x",
        90000,
        &["index.handler"],
    );
    assert_inlined(
        &ALL,
        "local list = {} \
         local grow = setmetatable({}, {__index = function(_, k) list[#list + 1] = k return #list end}) \
         local s = 0 \
         for i = 1, 50000 do \
           local before = #list \
           local n = grow[i] \
           local after = #list \
           s = s + before + n + after \
         end \
         return s, #list",
        45000,
        &["index.handler"],
    );
    assert_inlined(
        &ALL,
        "local cnt = {n = 0} \
         local p = setmetatable({}, {__newindex = function(t, k, v) cnt.n = cnt.n + v end}) \
         local s = 0 \
         for i = 1, 100000 do \
           local a = cnt.n \
           p.x = 1 \
           local b = cnt.n \
           s = s + a + b \
         end \
         return s, cnt.n",
        90000,
        &["newindex.handler"],
    );
}

// __newindex

#[test]
fn a_function_newindex_fills_a_backing_table() {
    assert_inlined(
        &ALL,
        "local keys, backing = {}, {} \
         for i = 1, 64 do keys[i] = 'item_' .. i end \
         local proxy = setmetatable({}, {__newindex = function(_, key, value) backing[key] = value end}) \
         for i = 1, 100000 do proxy[keys[(i - 1) % 64 + 1]] = i % 101 end \
         local total = 0 \
         for i = 1, 64 do total = total + (backing[keys[i]] or 0) end \
         return total, rawget(proxy, 'item_1')",
        90000,
        &["newindex.handler"],
    );
}

#[test]
fn a_function_newindex_takes_a_constant_key() {
    assert_inlined(
        &ALL,
        "local store = {} \
         local q = setmetatable({}, {__newindex = function(t, k, v) store[k] = v + 1 end}) \
         for i = 1, 100000 do q.x = i end \
         return store.x, rawget(q, 'x')",
        90000,
        &["newindex.handler"],
    );
}

#[test]
fn a_function_newindex_that_rawsets_elsewhere() {
    assert_inlined(
        &ALL,
        "local store = {} local set = rawset \
         local keys = {'a', 'b', 'c'} \
         local p = setmetatable({}, {__newindex = function(t, k, v) set(store, k, v) end}) \
         for i = 1, 100000 do p[keys[i % 3 + 1]] = i end \
         return store.a, store.b, store.c, rawget(p, 'a')",
        90000,
        &["newindex.handler", "table.set_string_walk"],
    );
    // Through the global `rawset`, a store under a run-time key could
    // overwrite the global the trace read once at entry (a global slot on
    // lua51 and luajit21, a constant field elsewhere): the trace must not
    // keep the old value.
    assert_agree(
        &ALL,
        "local store = {} \
         local keys = {'a', 'b', 'c'} \
         local p = setmetatable({}, {__newindex = function(t, k, v) rawset(store, k, v) end}) \
         for i = 1, 100000 do p[keys[i % 3 + 1]] = i end \
         return store.a, store.b, store.c, rawget(p, 'a')",
    );
}

#[test]
fn a_function_newindex_that_rawsets_into_the_table_itself() {
    // The new key lands in the table, so a later write of it is raw: the
    // handler runs once per key. Trace or not, the results agree.
    assert_agree(
        &ALL,
        "local r = setmetatable({}, {__newindex = function(t, k, v) rawset(t, k, v * 2) end}) \
         for i = 1, 100000 do r[i] = i end \
         return #r, r[1], r[100000]",
    );
}

#[test]
fn a_rawset_of_a_number_key_into_a_table_with_a_newindex_is_refused() {
    // The array store of a trace leaves a new key of a table with a
    // metatable to the interpreter, so the recording names `rawset`.
    assert_aborts_native(
        &ALL,
        "local p = setmetatable({}, {__newindex = function() end}) \
         for i = 1, 300000 do rawset(p, i, i) end \
         return #p, rawget(p, 5)",
        "rawset",
    );
}

#[test]
fn a_native_newindex_under_a_run_time_key_is_refused() {
    assert_aborts_native(
        &ALL,
        "local q = setmetatable({}, {__newindex = rawequal}) \
         local keys = {'a', 'b', 'c'} \
         for i = 1, 300000 do q[keys[i % 3 + 1]] = i end \
         return rawget(q, 'a')",
        "rawequal",
    );
}

#[test]
fn a_native_newindex_under_a_constant_key_is_refused_even_for_nil() {
    assert_aborts_native(
        &ALL,
        "local q = setmetatable({}, {__newindex = rawset}) \
         for i = 1, 300000 do q.x = nil end \
         return rawget(q, 'x')",
        "rawset",
    );
}

#[test]
fn a_handler_shared_by_a_class_and_a_table_chain_gets_the_table_it_was_found_at() {
    // `a` keeps the handler directly; `b` reaches the same closure through
    // a table `__index` or `__newindex`, so the handler gets `proto`, not `b`.
    // The first object the loop meets is a direct one, and the chain's
    // objects pass the same check with the right table.
    assert_inlined(
        &ALL,
        "local f = function(t, k) return t.tag end \
         local mt = {__index = f} \
         local proto = setmetatable({tag = 100}, mt) \
         local a = setmetatable({tag = 1}, mt) \
         local b = setmetatable({tag = 2}, {__index = proto}) \
         local objects = {} \
         for i = 1, 64 do objects[i] = (i <= 32) and a or b end \
         local s = 0 \
         for i = 1, 100000 do \
           local o = objects[(i - 1) % 64 + 1] \
           s = s + o.x \
         end \
         return s",
        90000,
        &["index.handler"],
    );
    assert_inlined(
        &ALL,
        "local count = 0 \
         local f = function(t, k, v) count = count + t.tag end \
         local mt = {__newindex = f} \
         local proto = setmetatable({tag = 100}, mt) \
         local a = setmetatable({tag = 1}, mt) \
         local b = setmetatable({tag = 2}, {__newindex = proto}) \
         local objects = {} \
         for i = 1, 64 do objects[i] = (i <= 32) and a or b end \
         for i = 1, 100000 do \
           local o = objects[(i - 1) % 64 + 1] \
           o.x = i \
         end \
         return count",
        90000,
        &["newindex.handler"],
    );
}

#[test]
fn a_function_newindex_stores_values_of_every_type() {
    // Strings, nil and tables go through the runtime's store; numbers and
    // booleans overwrite their slot in place. A trace holds one kind of
    // store to a table, so the two kinds are two loops.
    assert_inlined(
        &ALL,
        "local st = {} \
         local w = setmetatable({}, {__newindex = function(t, k, v) st[k] = v end}) \
         for i = 1, 100000 do w.s = 'str' w.n = nil w.t = st end \
         return st.s, st.n, st.t == st",
        90000,
        &[
            "newindex.handler",
            "table.set_string",
            "table.delete_string",
        ],
    );
    assert_inlined(
        &ALL,
        "local st = {} \
         local w = setmetatable({}, {__newindex = function(t, k, v) st[k] = v end}) \
         for i = 1, 100000 do w.b = true w.f = 1.5 w.i = i end \
         return st.b, st.f, st.i",
        90000,
        &["newindex.handler"],
    );
}

#[test]
fn a_table_newindex_redirects_the_store() {
    assert_inlined(
        &ALL,
        "local store = {} \
         local keys = {'a', 'b', 'c', 'd'} \
         local p = setmetatable({}, {__newindex = store}) \
         for i = 1, 100000 do p[keys[i % 4 + 1]] = i p.zz = i end \
         return store.a, store.b, store.c, store.d, store.zz, rawget(p, 'a')",
        90000,
        &["table.set_string_walk"],
    );
}

#[test]
fn a_table_newindex_chain_ends_in_a_function() {
    assert_inlined(
        &ALL,
        "local store = {} \
         local inner = setmetatable({}, {__newindex = function(t, k, v) store[k] = v end}) \
         local p = setmetatable({}, {__newindex = inner}) \
         for i = 1, 100000 do p.x = i end \
         return store.x, rawget(p, 'x'), rawget(inner, 'x')",
        90000,
        &["newindex.handler"],
    );
}

#[test]
fn a_newindex_handler_replaced_or_a_key_made_present_leaves_the_trace() {
    // The trace pins `mt` as a class and its entry checks that `mt.__newindex`
    // is still the recorded closure. The replacement runs in the interpreter
    // (a trace's store under a `__` key misses), so the next entry is refused
    // (`EntryMismatch`) instead of a per-access guard failing inside the
    // trace, and the interpreter runs the new handler. A trace that checks
    // the handler with a helper leaves through that guard instead.
    assert_leaves(
        &ALL,
        "local store = {} \
         local mt = {__newindex = function(t, k, v) store[k] = v end} \
         local p = setmetatable({}, mt) \
         for i = 1, 100000 do \
           p.x = i \
           if i == 50000 then mt.__newindex = function(t, k, v) store[k] = -v end end \
         end \
         return store.x",
        45000,
        &[TraceExitKind::GuardFail, TraceExitKind::EntryMismatch],
    );
    assert_exits(
        &ALL,
        "local store = {} \
         local p = setmetatable({}, {__newindex = function(t, k, v) store[k] = v end}) \
         for i = 1, 100000 do \
           p.x = i \
           if i == 50000 then rawset(p, 'x', 0) end \
         end \
         return store.x, rawget(p, 'x')",
        45000,
    );
}

#[test]
fn an_error_in_a_function_newindex_is_the_interpreters() {
    assert_rebuilt(
        &ALL,
        "local store = {} \
         local p = setmetatable({}, {__newindex = function(t, k, v) \
           store.x = v \
           if v == 90000 then error('late') end end}) \
         local function run() for i = 1, 100000 do p.x = i end end \
         local ok, e = xpcall(run, debug.traceback) \
         return ok, e, store.x",
    );
}

#[test]
fn a_strict_environment_takes_a_new_global() {
    assert_inlined(
        &ALL,
        "local store = {} \
         setmetatable(_G, {__newindex = function(t, k, v) store[k] = v end}) \
         for i = 1, 100000 do newglobal = i end \
         return store.newglobal, rawget(_G, 'newglobal')",
        90000,
        &["newindex.handler"],
    );
}

// __len

#[test]
fn a_function_len_answers_the_length_in_the_profiles_that_ask() {
    let source = "local o = setmetatable({1, 2, 3}, {__len = function(t, u) return 42 end}) \
         local s = 0 \
         for i = 1, 100000 do s = s + #o end \
         return s";
    assert_inlined(&LEN_ASKED, source, 90000, &["len.handler"]);
    // Lua 5.1 and LuaJIT take the border, and no handler is called.
    assert_inlined(&LEN_IGNORED, source, 90000, &[]);
}

#[test]
fn a_function_len_is_called_with_the_table_twice() {
    assert_inlined(
        &LEN_ASKED,
        "local o = setmetatable({}, {__len = function(a, b) return a == b and 7 or 8 end}) \
         local s = 0 \
         for i = 1, 100000 do s = s + #o end \
         return s",
        90000,
        &["len.handler"],
    );
}

#[test]
fn a_function_len_may_give_a_float_a_string_or_anything() {
    // Not an integer: the profile's number, a string, a boolean, a table.
    let float = "local o = setmetatable({}, {__len = function() return 2.5 end}) \
         local s = 0 \
         for i = 1, 100000 do s = s + #o end \
         return s";
    assert_inlined(&LEN_ASKED, float, 90000, &["len.handler"]);
    assert_inlined(&LEN_IGNORED, float, 90000, &[]);
    let truthiness = |result: &str| {
        format!(
            "local o = setmetatable({{}}, {{__len = function() return {result} end}}) \
             local c = 0 \
             for i = 1, 100000 do if #o then c = c + 1 end end \
             return c"
        )
    };
    // A boolean or nil result is not a number and still runs natively.
    for result in ["true", "nil"] {
        let source = truthiness(result);
        assert_inlined(&LEN_ASKED, &source, 90000, &["len.handler"]);
        assert_agree(&LEN_IGNORED, &source);
    }
    // These stay interpreted: the truthiness test of a string aborts the
    // recording, and a returned table escapes it.
    for result in ["'x'", "{}"] {
        assert_agree(&ALL, &truthiness(result));
    }
    // A string that reaches arithmetic converts, or raises the error the
    // interpreter raises.
    assert_agree(
        &ALL,
        "local o = setmetatable({}, {__len = function() return '7' end}) \
         local s = 0 \
         for i = 1, 100000 do s = s + #o end \
         return s",
    );
    assert_agree(
        &ALL,
        "local o = setmetatable({}, {__len = function() return {} end}) \
         local s = 0 \
         local ok, e = pcall(function() for i = 1, 100000 do s = s + #o end end) \
         return ok, e",
    );
}

#[test]
fn a_function_len_that_is_replaced_leaves_the_trace() {
    let source = "local mt = {__len = function() return 3 end} \
         local o = setmetatable({1, 2, 3, 4, 5}, mt) \
         local s = 0 \
         for i = 1, 100000 do \
           s = s + #o \
           if i == 50000 then mt.__len = function() return 100 end end \
         end \
         return s";
    assert_exits(&LEN_ASKED, source, 45000);
    assert_agree(&LEN_IGNORED, source);
}

#[test]
fn an_error_in_a_function_len_is_the_interpreters() {
    let source = "local cnt = {n = 0} \
         local r = setmetatable({}, {__len = function() \
           cnt.n = cnt.n + 1 \
           if cnt.n == 90000 then error('late') end return 1 end}) \
         local function run() local s = 0 for i = 1, 100000 do s = s + #r end return s end \
         local ok, e = xpcall(run, debug.traceback) \
         return ok, e, cnt.n";
    assert_rebuilt(&LEN_ASKED, source);
    // Lua 5.1 and LuaJIT never call the handler.
    assert_agree(&LEN_IGNORED, source);
}

// The table library

#[test]
fn the_table_library_honours_the_metamethods_it_would_in_the_interpreter() {
    // A proxy whose `__index`, `__newindex` and `__len` are functions: the
    // library functions of Lua 5.3 and later go through them. Native code
    // runs them or leaves; it never reads the table raw.
    let source = "local store = {} \
         local proxy = setmetatable({}, { \
           __index = function(t, k) return store[k] end, \
           __newindex = function(t, k, v) store[k] = v end, \
           __len = function() return #store end}) \
         local total = 0 \
         for i = 1, 20000 do table.insert(proxy, i) total = total + #proxy end \
         local s = 0 \
         for i = 1, 20000 do s = s + table.remove(proxy) end \
         for i = 1, 100 do table.insert(proxy, i) end \
         local t2 = 0 \
         for i = 1, 5000 do \
           t2 = t2 + #table.concat(proxy, ',', 1, 3) \
           local a, b = table.unpack(proxy, 1, 2) \
           t2 = t2 + a + b \
           table.move(proxy, 1, 3, 4) \
         end \
         return total, #store, s, t2, store[4]";
    assert_agree(&TABLE_LIBRARY, source);
    // The library calls on the proxy are named among what aborted a
    // recording, or left a trace: they are not read raw.
    for profile in TABLE_LIBRARY {
        let (_, lua) = outcome(profile, ExecutionMode::Tracing, source);
        let stats = lua.vm().jit_statistics().trace;
        assert!(
            stats
                .native_aborts
                .iter()
                .any(|(name, _)| name.starts_with("table."))
                || stats.exits_of(TraceExitKind::GuardFail) > 0,
            "{profile:?}: {:?} {stats:?}",
            stats.native_aborts
        );
    }
}

#[test]
fn the_table_library_reads_a_length_and_elements_that_metamethods_change() {
    let source = "local function mk(mt) return setmetatable({'a', 'b', 'c', 'd', 'e'}, mt) end \
         local variants = { \
           len = mk({__len = function() return 3 end}), \
           index = setmetatable({'a', 'b'}, {__index = function(t, i) return 'z' .. i end, \
                                             __len = function() return 4 end}), \
           newindex = mk({__newindex = function(t, k, v) rawset(t, k, v .. '!') end}), \
           idxtable = mk({__index = {}}), \
           plain = mk(nil)} \
         local out = {} \
         for name, t in pairs(variants) do \
           local acc = 0 \
           for i = 1, 3000 do \
             acc = acc + #table.concat(t, ',') \
             local a, b, c = table.unpack(t) \
             acc = acc + (a and #a or 0) + (b and #b or 0) + (c and #c or 0) \
           end \
           out[#out + 1] = name .. '=' .. acc \
         end \
         table.sort(out) \
         local out2 = {} \
         for name, t in pairs(variants) do \
           local acc = 0 \
           for i = 1, 3000 do \
             table.insert(t, 'q') acc = acc + #t \
             table.remove(t) table.remove(t) \
             table.insert(t, 1, 'w') table.remove(t, 1) \
             table.insert(t, 'r') \
           end \
           out2[#out2 + 1] = name .. '=' .. acc .. '/' .. #t \
         end \
         table.sort(out2) \
         return table.concat(out, ' '), table.concat(out2, ' ')";
    assert_agree(&TABLE_LIBRARY, source);
}

// What is still not recorded names its metamethod

#[test]
fn a_native_index_handler_is_named_in_the_native_aborts() {
    let source = "local p = setmetatable({}, {__index = rawget}) \
         local s = 0 \
         for i = 1, 100000 do s = s + (p.x or 1) end \
         return s";
    for profile in ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(
            stats.native_aborts.of("rawget") > 0,
            "{profile:?}: {:?} {stats:?}",
            stats.native_aborts
        );
    }
}

#[test]
fn an_access_the_recorder_cannot_follow_names_its_metamethod() {
    // A table-valued `__newindex` takes a string key in the trace; a number
    // key stays with the interpreter, and the abort names the metamethod.
    let source = "local store = {} \
         local p = setmetatable({}, {__newindex = store}) \
         for i = 1, 100000 do p[i] = i end \
         return #store, rawget(p, 1)";
    for profile in ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(
            stats.native_aborts.of("__newindex") > 0,
            "{profile:?}: {:?} {stats:?}",
            stats.native_aborts
        );
    }
    // A handler that is a native function names the function instead.
    let source = "local q = setmetatable({}, {__newindex = rawset}) \
         local keys = {'a', 'b'} \
         local s = 0 \
         for i = 1, 100000 do q.x = nil s = s + 1 end \
         return s";
    assert_agree(&ALL, source);
}

// Stores under run-time string keys beside facts the entry checked once

const NAMES: &str = "local names = {} for i = 1, 200000 do names[i] = 'x' end ";

#[test]
fn a_string_store_does_not_hide_a_callee_it_overwrites() {
    let source = format!(
        "local M = {{}} function M.f() return 1 end {NAMES} names[150000] = 'f' \
         local n = 0 \
         local ok = pcall(function() \
           for i = 1, 200000 do n = n + M.f() M[names[i]] = 5 end end) \
         return n, ok"
    );
    assert_agree(&ALL, &source);
}

#[test]
fn a_string_store_does_not_hide_a_string_field_it_overwrites() {
    let direct = format!(
        "local T = {{s = 'abc'}} {NAMES} names[150000] = 's' local n = 0 \
         for i = 1, 200000 do n = n + #T.s T[names[i]] = 'zzzzz' end return n"
    );
    assert_agree(&ALL, &direct);
    let shadowed = format!(
        "local Base = {{s = 'abc'}} local obj = setmetatable({{}}, {{__index = Base}}) \
         {NAMES} names[150000] = 's' local n = 0 \
         for i = 1, 200000 do n = n + #obj.s rawset(obj, names[i], 'zz') end return n"
    );
    assert_agree(&ALL, &shadowed);
    let fallback = format!(
        "local Base = {{s = 'abc'}} local obj = setmetatable({{}}, {{__index = Base}}) \
         {NAMES} names[150000] = 's' local n = 0 \
         for i = 1, 200000 do n = n + #obj.s Base[names[i]] = 'zzzzz' end return n"
    );
    assert_agree(&ALL, &fallback);
}

#[test]
fn a_string_store_of_a_metamethod_name_is_the_interpreters() {
    let len = format!(
        "local mt = {{}} local t = setmetatable({{1, 2, 3}}, mt) \
         local lenf = setmetatable({{}}, {{__call = function() return 10 end}}) \
         {NAMES} names[150000] = '__len' local s = 0 \
         for i = 1, 200000 do s = s + #t mt[names[i]] = lenf end return s"
    );
    assert_agree(&LEN_ASKED, &len);
    let rawset = format!(
        "local mt = {{}} local t = setmetatable({{1, 2, 3}}, mt) \
         {NAMES} names[150000] = '__len' local s = 0 \
         for i = 1, 200000 do s = s + #t rawset(mt, names[i], function() return 10 end) end \
         return s"
    );
    assert_agree(&LEN_ASKED, &rawset);
    let index = format!(
        "local Base = {{s = 'abc'}} local mt = {{__index = Base}} \
         local obj = setmetatable({{}}, mt) {NAMES} names[150000] = '__index' \
         local n = 0 \
         for i = 1, 200000 do n = n + #obj.s mt[names[i]] = {{s = 'zzzzz'}} end return n"
    );
    assert_agree(&ALL, &index);
}

#[test]
fn a_rawset_through_a_handler_rebinds_a_global_the_trace_read_once() {
    let source = "local keys = {} for i = 1, 200000 do keys[i] = 'x' end keys[150000] = 'rawset' \
         local p = setmetatable({}, {__newindex = function(t, k, v) rawset(_G, k, v) end}) \
         local n = 0 \
         local ok, e = pcall(function() for i = 1, 200000 do p[keys[i]] = i n = n + 1 end end) \
         return ok, e, n";
    assert_agree(&ALL, source);
}

#[test]
fn a_store_through_the_environment_can_unbind_a_global_function() {
    let source = "function h() return 1 end \
         local names = {} for i = 1, 200000 do names[i] = 'x' end names[150000] = 'h' \
         local env = _G local s = 0 \
         local ok, e = pcall(function() \
             for i = 1, 200000 do s = s + h() env[names[i]] = nil end end) \
         return ok, e, s";
    assert_agree(&ALL, source);
}

// Operators

/// The profiles with integer division and the bitwise operators.
const OPERATORS_53: [CompatibilityProfile; 3] = TABLE_LIBRARY;

/// The profiles where `pairs` calls a `__pairs` handler.
const PAIRS_HANDLER: [CompatibilityProfile; 4] = LEN_ASKED;

/// The profiles where `a <= b` falls back on `not (b < a)` without `__le`.
const LE_FALLBACK: [CompatibilityProfile; 5] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::LuaJit21,
];

/// A loop of `20000` iterations of the operator `op` with the handler `event`
/// on the left, on the right, between two objects and, with `strings`, with a
/// string on the other side (the recorder leaves a bitwise operator on a
/// string to the interpreter).
fn binary_operator(event: &str, op: &str, strings: bool) -> String {
    let mixed = if strings {
        format!("+ (q {op} '5') + ('9' {op} p)")
    } else {
        String::new()
    };
    format!(
        "local function n(v) if type(v) == 'table' then return v.x end return v end \
         local C = {{{event} = function(a, b) return n(a) {op} n(b) end}} \
         local p = setmetatable({{x = 7}}, C) \
         local q = setmetatable({{x = 3}}, C) \
         local acc = 0 \
         for i = 1, 20000 do \
           acc = acc + (p {op} q) + (p {op} 2) + (11 {op} q) {mixed} \
         end \
         return acc"
    )
}

#[test]
fn an_arithmetic_metamethod_runs_inside_the_trace_on_either_side() {
    for (event, op) in [
        ("__add", "+"),
        ("__sub", "-"),
        ("__mul", "*"),
        ("__div", "/"),
        ("__mod", "%"),
        ("__pow", "^"),
    ] {
        assert_inlined(
            &ALL,
            &binary_operator(event, op, true),
            19000,
            &["operator.handler"],
        );
    }
}

#[test]
fn integer_division_and_bitwise_metamethods_run_inside_the_trace() {
    assert_inlined(
        &OPERATORS_53,
        &binary_operator("__idiv", "//", true),
        19000,
        &["operator.handler"],
    );
    for (event, op) in [
        ("__band", "&"),
        ("__bor", "|"),
        ("__bxor", "~"),
        ("__shl", "<<"),
        ("__shr", ">>"),
    ] {
        assert_inlined(
            &OPERATORS_53,
            &binary_operator(event, op, false),
            19000,
            &["operator.handler"],
        );
    }
}

#[test]
fn unary_minus_gets_the_operand_twice_and_complement_once() {
    let negate = "local C = {__unm = function(a, b) return a == b and -a.x or 0 end} \
         local p = setmetatable({x = 7}, C) \
         local acc = 0 \
         for i = 1, 20000 do acc = acc + (-p) end \
         return acc";
    assert_inlined(&ALL, negate, 19000, &["operator.handler"]);
    // The interpreter calls `__bnot` with the operand alone.
    let complement = "local C = {__bnot = function(a, b) if b == nil then return a.x end return 0 end} \
         local p = setmetatable({x = 7}, C) \
         local acc = 0 \
         for i = 1, 20000 do acc = acc + (~p) end \
         return acc";
    assert_inlined(&OPERATORS_53, complement, 19000, &["operator.handler"]);
}

#[test]
fn the_left_operand_is_asked_first_and_the_right_one_when_it_has_no_handler() {
    // Two classes: the left one's handler answers, else the right one's.
    let source = "local A = {__add = function(a, b) return 1 end} \
         local B = {__add = function(a, b) return 20 end} \
         local a, b = setmetatable({}, A), setmetatable({}, B) \
         local plain = setmetatable({}, {}) \
         local acc = 0 \
         for i = 1, 20000 do acc = acc + (a + b) + (b + a) + (plain + b) + (a + plain) end \
         return acc";
    assert_inlined(&ALL, source, 19000, &["operator.handler"]);
}

#[test]
fn one_trace_serves_every_object_of_a_class() {
    let source = "local C = {} C.__add = function(a, b) return a.x + b.x end \
         local objects = {} \
         for i = 1, 50 do objects[i] = setmetatable({x = i}, C) end \
         local acc = 0 \
         for i = 1, 20000 do \
           acc = acc + (objects[i % 50 + 1] + objects[(i * 7) % 50 + 1]) \
         end \
         return acc";
    assert_inlined(&ALL, source, 19000, &["operator.handler"]);
}

#[test]
fn another_class_or_handler_leaves_the_trace_and_the_interpreter_answers() {
    // After the loop is hot the objects change class, and then handler.
    let source = "local A = {__add = function(a, b) return 1 end} \
         local B = {__add = function(a, b) return 2 end} \
         local p, q = setmetatable({}, A), setmetatable({}, A) \
         local acc = 0 \
         for i = 1, 20000 do \
           if i == 12000 then setmetatable(p, B) end \
           if i == 16000 then B.__add = function() return 9 end end \
           acc = acc + (p + q) \
         end \
         return acc";
    assert_exits(&ALL, source, 11000);
}

#[test]
fn a_handler_that_tail_calls_a_native_gives_the_first_value_it_returns() {
    // `return f (...)` of a native returns the open results of the call.
    let source = "local C = {\
           __add = function(a, b) return math.max(a.x, b.x) end, \
           __lt = function(a, b) return rawequal(a.x, b.x) end, \
           __le = function(a, b) return math.abs(a.x - b.x) < 2 end, \
           __eq = function(a, b) return rawequal(a.x, b.x) end, \
           __concat = function(a, b) return tostring(a.x) end} \
         local p, q = setmetatable({x = 3}, C), setmetatable({x = 5}, C) \
         local acc = 0 \
         for i = 1, 20000 do \
           acc = acc + (p + q) \
           if p < q then acc = acc + 1 end \
           local r = q <= p \
           if r then acc = acc + 10 end \
           if p == q then acc = acc + 100 end \
           acc = acc + #(p .. 'z') \
         end \
         return acc";
    assert_inlined(&ALL, source, 19000, &["operator.handler"]);
}

#[test]
fn a_handler_that_returns_nothing_or_many_values_gives_one() {
    for body in ["return", "return 5, 6, 7", "return nil", "return false"] {
        let source = format!(
            "local C = {{__add = function(a, b) {body} end, __lt = function(a, b) {body} end}} \
             local p = setmetatable({{}}, C) \
             local acc, hits = 0, 0 \
             for i = 1, 20000 do \
               local r = p + p \
               if r then acc = acc + r end \
               if p < p then hits = hits + 1 end \
             end \
             return acc, hits"
        );
        assert_inlined(&ALL, &source, 19000, &["operator.handler"]);
    }
}

#[test]
fn a_handler_that_stores_and_then_leaves_the_trace_is_rebuilt_for_the_interpreter() {
    // The store commits before the guard on the counter fails, so the exit
    // rebuilds the handler's frame and its return goes where the operator's
    // does.
    let operator = "local C = {__add = function(a, b) \
             a.hits = a.hits + 1 \
             if a.hits < 15000 then return a.x + b.x end \
             return 100 end} \
         local p = setmetatable({x = 1, hits = 0}, C) \
         local q = setmetatable({x = 2, hits = 0}, C) \
         local acc = 0 \
         for i = 1, 20000 do acc = acc + (p + q) end \
         return acc, p.hits";
    assert_rebuilt(&ALL, operator);
    let ordering = |tail: &str, handler: &str| {
        format!(
            "local C = {{{handler} = function(a, b) \
                 a.hits = a.hits + 1 \
                 if a.hits < 15000 then return a.x < b.x end \
                 return true end}} \
             local p = setmetatable({{x = 1, hits = 0}}, C) \
             local q = setmetatable({{x = 2, hits = 0}}, C) \
             local acc = 0 \
             for i = 1, 20000 do {tail} end \
             return acc, p.hits"
        )
    };
    // The answer goes to a register.
    let value = ordering("local r = p < q if r then acc = acc + 1 end", "__lt");
    assert_rebuilt(&ALL, &value);
    // The answer picks the branch of a test.
    let branch = ordering("if p < q then acc = acc + 1 end", "__lt");
    assert_rebuilt(&ALL, &branch);
    let equal = ordering("if p == q then acc = acc + 1 end", "__eq");
    assert_rebuilt(&ALL, &equal);
    // `q <= p` answered by the swapped `__lt`, negated.
    let fallback = ordering("if q <= p then acc = acc + 1 end", "__lt");
    assert_rebuilt(&LE_FALLBACK, &fallback);
}

#[test]
fn an_error_in_a_handler_has_the_interpreters_message() {
    let source = "local C = {__add = function(a, b) \
             if b.x == 0 then error('zero') end \
             return a.x + b.x end} \
         local p = setmetatable({x = 1}, C) \
         local q = setmetatable({x = 2}, C) \
         local acc = 0 \
         local ok, e = pcall(function() \
           for i = 1, 20000 do \
             if i == 16000 then q.x = 0 end \
             acc = acc + (p + q) \
           end \
         end) \
         return ok, e, acc";
    assert_agree(&ALL, source);
    // The error travels through the operator's frame.
    let traceback = "local C = {__add = function(a, b) \
             if b.x == 0 then error('zero') end \
             return a.x + b.x end} \
         local p = setmetatable({x = 1}, C) \
         local q = setmetatable({x = 2}, C) \
         local acc = 0 \
         local ok, e = xpcall(function() \
           for i = 1, 20000 do \
             if i == 16000 then q.x = 0 end \
             acc = acc + (p + q) \
           end \
         end, debug.traceback) \
         return ok, e, acc";
    assert_agree(&ALL, traceback);
}

#[test]
fn an_operator_with_no_handler_raises_the_interpreters_error() {
    let source = "local C = {__add = function(a, b) return 1 end} \
         local p = setmetatable({}, C) \
         local acc = 0 \
         local ok, e = pcall(function() \
           for i = 1, 20000 do \
             acc = acc + (p + 1) \
             if i == 15000 then setmetatable(p, nil) end \
           end \
         end) \
         return ok, e, acc";
    assert_agree(&ALL, source);
}

// Comparison

#[test]
fn eq_runs_between_two_objects_of_one_class() {
    let source = "local calls = 0 \
         local C = {__eq = function(a, b) calls = calls + 1 return a.x == b.x end} \
         local a, b = setmetatable({x = 1}, C), setmetatable({x = 1}, C) \
         local c = setmetatable({x = 2}, C) \
         local n = 0 \
         for i = 1, 20000 do \
           if a == b then n = n + 1 end \
           if a == c then n = n + 100 end \
           if a ~= c then n = n + 1 end \
           local r = b == c \
           if not r then n = n + 1 end \
         end \
         return n, calls";
    assert_inlined(&ALL, source, 19000, &["operator.handler"]);
}

#[test]
fn eq_does_not_run_for_the_same_table_or_a_number() {
    let source = "local calls = 0 \
         local C = {__eq = function(a, b) calls = calls + 1 return false end} \
         local a = setmetatable({v = 1}, C) \
         local n = 0 \
         for i = 1, 20000 do \
           if a == a then n = n + a.v + i % 7 end \
           if a == 1 then n = n + 100 end \
           if a ~= i then n = n + 1 end \
         end \
         return n, calls";
    assert_inlined(&ALL, source, 19000, &[]);
    // A second table is another matter.
    let identical = "local calls = 0 \
         local C = {__eq = function(a, b) calls = calls + 1 return false end} \
         local a = setmetatable({}, C) \
         local b = setmetatable({}, C) \
         local n = 0 \
         for i = 1, 20000 do \
           local x = a \
           if i > 15000 then x = b end \
           if a == x then n = n + 1 end \
         end \
         return n, calls";
    assert_agree(&ALL, identical);
}

#[test]
fn eq_between_tables_without_a_handler_is_false() {
    let source = "local a, b = setmetatable({}, {}), {} \
         local n = 0 \
         for i = 1, 20000 do \
           if a == b then n = n + 100 end \
           if a ~= b then n = n + 1 end \
         end \
         return n";
    assert_inlined(&ALL, source, 19000, &["operator.handler"]);
}

#[test]
fn eq_asks_the_left_handler_first_in_5_3_and_wants_a_shared_one_before_that() {
    // Two classes with their own handlers: Lua 5.3 and later call the left
    // one, earlier profiles need both operands to name the same function.
    let source = "local calls = 0 \
         local A = {__eq = function(a, b) calls = calls + 1 return true end} \
         local B = {__eq = function(a, b) calls = calls + 10 return false end} \
         local a, b = setmetatable({}, A), setmetatable({}, B) \
         local n = 0 \
         for i = 1, 20000 do \
           if a == b then n = n + 1 end \
           if b == a then n = n + 1000 end \
         end \
         return n, calls";
    assert_inlined(&TABLE_LIBRARY, source, 19000, &["operator.handler"]);
    assert_agree(&ALL, source);
    // A handler both classes share is called in every profile.
    let shared = "local calls = 0 \
         local function same(a, b) calls = calls + 1 return true end \
         local A, B = {__eq = same}, {__eq = same} \
         local a, b = setmetatable({}, A), setmetatable({}, B) \
         local n = 0 \
         for i = 1, 20000 do if a == b then n = n + 1 end end \
         return n, calls";
    assert_inlined(&ALL, shared, 19000, &["operator.handler"]);
}

#[test]
fn ordering_metamethods_answer_with_the_truth_of_their_first_value() {
    let source = "local C = {__lt = function(a, b) if a.x < b.x then return a.x - b.x end end, \
                 __le = function(a, b) if a.x <= b.x then return 5 end end} \
         local a, b = setmetatable({x = 1}, C), setmetatable({x = 2}, C) \
         local n = 0 \
         for i = 1, 20000 do \
           if a < b then n = n + 1 end \
           if b < a then n = n + 10 end \
           if a <= a then n = n + 100 end \
           if b <= a then n = n + 1000 end \
           local r = a < b \
           if r == true then n = n + 10000 end \
         end \
         return n";
    assert_inlined(&ALL, source, 19000, &["operator.handler"]);
}

#[test]
fn le_falls_back_on_the_swapped_lt_where_the_profile_has_it() {
    let source = "local calls = 0 \
         local C = {__lt = function(a, b) calls = calls + 1 return a.x < b.x end} \
         local a, b = setmetatable({x = 1}, C), setmetatable({x = 2}, C) \
         local n = 0 \
         for i = 1, 20000 do \
           if a <= b then n = n + 1 end \
           if b <= a then n = n + 10 end \
           if a <= a then n = n + 100 end \
           local r = b <= a \
           if r then n = n + 1000 end \
         end \
         return n, calls";
    assert_inlined(&LE_FALLBACK, source, 19000, &["operator.handler"]);
    // Lua 5.5 raises the error of a comparison.
    assert_agree(&[CompatibilityProfile::Lua55], source);
}

#[test]
fn an_le_handler_wins_over_the_lt_fallback() {
    let source = "local C = {__lt = function(a, b) return false end, \
                 __le = function(a, b) return true end} \
         local a, b = setmetatable({x = 1}, C), setmetatable({x = 2}, C) \
         local n = 0 \
         for i = 1, 20000 do if b <= a then n = n + 1 end end \
         return n";
    assert_inlined(&ALL, source, 19000, &["operator.handler"]);
}

#[test]
fn ordering_mixes_objects_with_numbers_and_strings_where_the_profile_allows() {
    let source = "local C = {__lt = function(a, b) \
               local x = type(a) == 'table' and a.x or a \
               local y = type(b) == 'table' and b.x or b \
               return tonumber(x) < tonumber(y) end} \
         local a = setmetatable({x = 5}, C) \
         local n = 0 \
         for i = 1, 20000 do \
           if a < 6 then n = n + 1 end \
           if 4 < a then n = n + 10 end \
           if a < '3' then n = n + 100 end \
           if 3 < a then n = n + 1000 end \
         end \
         return n";
    // Lua 5.1 compares only values of one type.
    assert_inlined(
        &[
            CompatibilityProfile::Lua52,
            CompatibilityProfile::Lua53,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::Lua55,
        ],
        source,
        19000,
        &["operator.handler"],
    );
    assert_agree(&ALL, source);
}

#[test]
fn ordering_handlers_of_two_classes_follow_the_profile() {
    let source = "local calls = 0 \
         local A = {__lt = function(a, b) calls = calls + 1 return true end} \
         local B = {__lt = function(a, b) calls = calls + 10 return false end} \
         local a, b = setmetatable({}, A), setmetatable({}, B) \
         local plain = setmetatable({}, {}) \
         local n = 0 \
         local ok, e = pcall(function() \
           for i = 1, 20000 do \
             if a < b then n = n + 1 end \
             if b < a then n = n + 1000 end \
             if plain < a then n = n + 100000 end \
           end \
         end) \
         return n, calls, ok, e";
    assert_agree(&ALL, source);
}

// Concatenation

#[test]
fn concat_runs_the_handler_for_each_object_in_a_chain() {
    let source = "local function s(v) \
           if type(v) == 'table' then return '<' .. v.x .. '>' end return tostring(v) end \
         local C = {__concat = function(a, b) return s(a) .. s(b) end} \
         local p = setmetatable({x = 'p'}, C) \
         local q = setmetatable({x = 'q'}, C) \
         local len = 0 \
         for i = 1, 20000 do \
           local e1 = p .. 'a' \
           local e2 = 'b' .. p \
           local e3 = 1 .. p .. 2.5 \
           local e4 = p .. q \
           local e5 = 'x' .. p .. 'y' .. q .. 'z' \
           len = len + #e1 + #e2 + #e3 + #e4 + #e5 \
         end \
         return len";
    assert_inlined(&ALL, source, 19000, &["operator.handler"]);
}

#[test]
fn concat_continues_with_the_value_a_handler_returns() {
    // The first handler returns an object, so the next pair needs a handler
    // again; the second returns a number that the next step concatenates.
    let source = "local C = {} \
         local chain = setmetatable({x = 'c'}, C) \
         C.__concat = function(a, b) \
           if type(a) == 'string' and a == 'x' then return 42 end \
           if type(b) == 'string' and b == 'y' then return chain end \
           return 'z' end \
         local p = setmetatable({}, C) \
         local len = 0 \
         for i = 1, 20000 do \
           local e = 'w' .. 'x' .. p .. 'y' \
           len = len + #e \
         end \
         return len";
    assert_inlined(&ALL, source, 19000, &["operator.handler"]);
}

#[test]
fn concat_of_numbers_and_strings_around_an_object_matches_the_interpreter() {
    let source = "local function s(v) if type(v) == 'table' then return 'T' end return v end \
         local C = {__concat = function(a, b) return s(a) .. s(b) end} \
         local p = setmetatable({}, C) \
         local out = {} \
         for i = 1, 20000 do \
           out[i % 5 + 1] = i .. p .. 1.5 .. 'x' .. -0.0 .. p \
         end \
         return table.concat(out, ',')";
    assert_inlined(&ALL, source, 19000, &["operator.handler"]);
}

#[test]
fn a_concat_handler_that_stores_runs_and_is_rebuilt_at_an_exit() {
    // The last step of a chain may store: the handler runs natively, and an
    // exit inside it rebuilds the frame of the chain's continuation.
    let source = "local C = {__concat = function(a, b) \
             a.hits = a.hits + 1 \
             if a.hits < 15000 then return 'x' end \
             return 'yy' end} \
         local p = setmetatable({hits = 0}, C) \
         local len = 0 \
         for i = 1, 20000 do len = len + #(p .. 'a') end \
         return len, p.hits";
    assert_rebuilt(&ALL, source);
    // The strings to its right are folded before the handler runs.
    let folded = "local C = {__concat = function(a, b) \
             a.hits = a.hits + 1 \
             if a.hits < 15000 then return b end \
             return 'yy' end} \
         local p = setmetatable({hits = 0}, C) \
         local len = 0 \
         for i = 1, 20000 do len = len + #(p .. 'k' .. i) end \
         return len, p.hits";
    assert_rebuilt(&ALL, folded);
    // The traceback shows the handler under the concatenation.
    let error = "local C = {__concat = function(a, b) \
             a.hits = a.hits + 1 \
             if a.hits == 15000 then error('late') end \
             return 'x' end} \
         local p = setmetatable({hits = 0}, C) \
         local len = 0 \
         local function run() for i = 1, 20000 do len = len + #(p .. 'a') end end \
         local ok, e = xpcall(run, debug.traceback) \
         return ok, e, len, p.hits";
    assert_rebuilt(&ALL, error);
}

#[test]
fn a_concat_chain_goes_on_after_a_handler_only_while_it_has_stored_nothing() {
    // A step after a handler that stored would restart the chain at an exit
    // and store again, so the loop is left to the interpreter.
    let source = "local C = {__concat = function(a, b) a.hits = a.hits + 1 return 'x' end} \
         local p = setmetatable({hits = 0}, C) \
         local len = 0 \
         for i = 1, 20000 do len = len + #('w' .. p .. 'a') end \
         return len, p.hits";
    assert_agree(&ALL, source);
}

// tostring

#[test]
fn tostring_runs_a_tostring_handler_inside_the_trace() {
    let source = "local C = {__tostring = function(o) return 'obj' .. o.x end} \
         local objects = {} \
         for i = 1, 20 do objects[i] = setmetatable({x = i}, C) end \
         local len = 0 \
         for i = 1, 20000 do len = len + #tostring(objects[i % 20 + 1]) end \
         return len";
    assert_inlined(&ALL, source, 19000, &["operator.handler"]);
}

#[test]
fn a_tostring_handler_can_feed_the_call_that_takes_its_result() {
    // `f (tostring (o))` takes the one value `tostring` gives as an open
    // result, whatever the handler returns beyond the first.
    let source = "local C = {__tostring = function(o) return 'obj' .. o.x, 'more', 'values' end} \
         local objects = {} \
         for i = 1, 20 do objects[i] = setmetatable({x = i}, C) end \
         local function size(s) return #s end \
         local total = 0 \
         for i = 1, 20000 do total = total + size(tostring(objects[i % 20 + 1])) end \
         return total";
    assert_inlined(&ALL, source, 19000, &["operator.handler"]);
    // As one of several results, the others nil.
    let several = "local C = {__tostring = function(o) return 'o' .. o.x end} \
         local o = setmetatable({x = 1}, C) \
         local function pair(v) local a, b = tostring(v) return a, b end \
         local total = 0 \
         for i = 1, 20000 do \
           local a, b = pair(o) \
           total = total + #a + (b == nil and 1 or 0) \
         end \
         return total";
    assert_inlined(&ALL, several, 19000, &["operator.handler"]);
    // A tail call of `tostring` is left to the interpreter.
    let tail = "local C = {__tostring = function(o) return 'o' .. o.x end} \
         local o = setmetatable({x = 1}, C) \
         local function name(v) return tostring(v) end \
         local total = 0 \
         for i = 1, 20000 do total = total + #name(o) end \
         return total";
    assert_agree(&ALL, tail);
}

#[test]
fn a_tostring_handler_may_tail_call() {
    // `return tostring (o.v)` returns the open results of a native call.
    let source = "local C = {__tostring = function(o) return tostring(o.v) end} \
         local o = setmetatable({v = 12345}, C) \
         local len = 0 \
         for i = 1, 20000 do len = len + #tostring(o) end \
         return len";
    assert_inlined(&ALL, source, 19000, &["operator.handler"]);
}

#[test]
fn a_name_does_not_change_what_a_tostring_handler_answers() {
    let source = "local C = {__name = 'Thing', __tostring = function(o) return 'thing' end} \
         local o = setmetatable({}, C) \
         local len = 0 \
         for i = 1, 20000 do len = len + #tostring(o) end \
         return len";
    assert_inlined(&ALL, source, 19000, &["operator.handler"]);
}

#[test]
fn a_name_without_a_tostring_handler_is_the_tables_plain_text() {
    // No profile spells a table by its `__name`: the text is `table: N`.
    for name in ["'Thing'", "5", "{}", "nil"] {
        let source = format!(
            "local o = setmetatable({{}}, {{__name = {name}}}) \
             local len, head = 0, '' \
             for i = 1, 20000 do \
               local s = tostring(o) \
               len = len + #s \
               head = s:sub(1, 7) \
             end \
             return len, head"
        );
        assert_inlined(&ALL, &source, 19000, &["operator.handler", "tostring"]);
    }
}

#[test]
fn a_tostring_handler_that_returns_a_non_string_has_the_interpreters_result() {
    for result in ["42", "{}", "nil", "true", "2.5"] {
        let source = format!(
            "local C = {{__tostring = function(o) return {result} end}} \
             local o = setmetatable({{}}, C) \
             local len = 0 \
             local ok, e = pcall(function() \
               for i = 1, 20000 do len = len + #tostring(o) end \
             end) \
             return ok, e, len"
        );
        assert_agree(&ALL, &source);
    }
}

#[test]
fn a_tostring_handler_that_errors_has_the_interpreters_message() {
    let source = "local C = {__tostring = function(o) \
             if o.x == 0 then error('no string') end return 'ok' end} \
         local o = setmetatable({x = 1}, C) \
         local len = 0 \
         local ok, e = pcall(function() \
           for i = 1, 20000 do \
             if i == 16000 then o.x = 0 end \
             len = len + #tostring(o) \
           end \
         end) \
         return ok, e, len";
    assert_agree(&ALL, source);
}

#[test]
fn a_tostring_handler_that_stores_and_then_leaves_the_trace_is_rebuilt() {
    let source = "local C = {__tostring = function(o) \
             o.hits = o.hits + 1 \
             if o.hits < 15000 then return 'ok' end \
             return 'late' end} \
         local o = setmetatable({hits = 0}, C) \
         local len = 0 \
         for i = 1, 20000 do len = len + #tostring(o) end \
         return len, o.hits";
    assert_rebuilt(&ALL, source);
    // The error comes out of the handler under `tostring`, with the
    // traceback the interpreter shows.
    let error = "local C = {__tostring = function(o) \
             o.hits = o.hits + 1 \
             if o.hits == 15000 then error('late') end \
             return 'ok' end} \
         local o = setmetatable({hits = 0}, C) \
         local len = 0 \
         local function run() for i = 1, 20000 do len = len + #tostring(o) end end \
         local ok, e = xpcall(run, debug.traceback) \
         return ok, e, len, o.hits";
    assert_rebuilt(&ALL, error);
}

// pairs

#[test]
fn pairs_runs_a_pairs_handler_and_loops_over_the_iterator_it_returns() {
    let source = "local limit = 3 \
         local function step(state, k) \
           if k < limit then return k + 1, k * 2 end end \
         local C = {__pairs = function(t) return step, t, 0 end} \
         local o = setmetatable({}, C) \
         local n = 0 \
         for i = 1, 5000 do \
           for k, v in pairs(o) do n = n + k + v end \
         end \
         return n";
    assert_inlined(&PAIRS_HANDLER, source, 10000, &["operator.handler"]);
    // The Lua 5.1 family never asks, and traverses the empty table itself.
    assert_agree(&LEN_IGNORED, source);
}

#[test]
fn a_pairs_handler_that_returns_next_traverses_another_table() {
    let source = "local other = {10, 20, 30} \
         local C = {__pairs = function(t) return next, other, nil end} \
         local o = setmetatable({}, C) \
         local n = 0 \
         for i = 1, 5000 do \
           for k, v in pairs(o) do n = n + k + v end \
         end \
         return n";
    assert_agree(&ALL, source);
}

// Handlers the recorder does not follow

#[test]
fn a_native_operator_handler_is_named_among_the_native_aborts() {
    for (event, expression) in [
        ("__add", "p + i"),
        ("__sub", "p - i"),
        ("__unm", "-p"),
        ("__lt", "p < p"),
        ("__le", "p <= p"),
        ("__concat", "p .. i"),
        ("__eq", "p == q"),
    ] {
        let source = format!(
            "local C = {{{event} = rawequal}} \
             local p = setmetatable({{}}, C) \
             local q = setmetatable({{}}, C) \
             local acc = 0 \
             for i = 1, 20000 do local r = {expression} if r then acc = acc + 1 end end \
             return acc"
        );
        assert_aborts_native(&ALL, &source, "rawequal");
    }
    let tostring = "local C = {__tostring = type} \
         local p = setmetatable({}, C) \
         local len = 0 \
         for i = 1, 20000 do len = len + #tostring(p) end \
         return len";
    assert_aborts_native(&ALL, tostring, "__tostring");
}

#[test]
fn a_handler_that_is_a_callable_table_names_its_metamethod() {
    for (event, expression) in [
        ("__add", "p + i"),
        ("__mul", "i * p"),
        ("__unm", "-p"),
        ("__lt", "p < p"),
        ("__concat", "p .. i"),
        ("__eq", "p == q"),
    ] {
        let source = format!(
            "local callable = setmetatable({{}}, {{__call = function() return 3 end}}) \
             local C = {{{event} = callable}} \
             local p = setmetatable({{}}, C) \
             local q = setmetatable({{}}, C) \
             local acc = 0 \
             for i = 1, 20000 do local r = {expression} if r then acc = acc + 1 end end \
             return acc"
        );
        assert_aborts_native(&ALL, &source, event);
    }
}
