//! Table and bit library functions a trace records as operations of its own:
//! `rawget`, `rawset`, `rawlen`, `unpack`, `getmetatable`, `setmetatable`,
//! and the `bit` and `bit32` libraries on plain numbers.
//!
//! The loops compile without an abort, enter native code, and give the
//! interpreter's results in every profile that has the function. `rawget`,
//! `rawset`, `rawlen`, `unpack` and the bit functions make no call out of the
//! generated code; `getmetatable` and `setmetatable` call a helper (native code
//! never holds a table's header). A value the recording did not see leaves
//! the trace and the interpreter makes the call, raising its error if it has
//! one.

use oximond::runtime::Value;
use oximond::vm::TraceExitKind;
use oximond::{CompatibilityProfile, ExecutionMode, Lua, LuaJitExtension};

const ALL: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

/// The profiles that have `rawlen`.
const RAWLEN: [CompatibilityProfile; 4] = [
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
];

/// The profiles with `bit32`.
const BIT32: [CompatibilityProfile; 2] = [CompatibilityProfile::Lua52, CompatibilityProfile::Lua53];

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

/// The outcome as text, which tells `-0.0` from `0.0`, an integer from a
/// float, and compares NaN with itself.
fn show(result: &Result<Vec<Value>, String>) -> String {
    format!("{result:?}")
}

/// Runs `source` in both modes of each profile and checks that the loop
/// traced and ran natively with no abort, that the results agree, and that
/// the trace calls only the helpers named in `helpers` (none for an inline
/// function).
fn assert_native(profiles: &[CompatibilityProfile], source: &str, helpers: &[&str]) {
    for &profile in profiles {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.traces_compiled > 0, "{profile:?} {source} {stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert_eq!(stats.total_aborts(), 0, "{profile:?} {source} {stats:?}");
        let mut called = lua.vm().trace_helper_calls();
        called.sort_unstable();
        called.dedup();
        let mut wanted = helpers.to_vec();
        wanted.sort_unstable();
        assert_eq!(called, wanted, "{profile:?} {source}");
        if helpers.is_empty() {
            assert_eq!(
                lua.vm().trace_external_calls(),
                0,
                "{profile:?} {source}: {:?}",
                lua.vm().trace_helper_calls()
            );
        }
    }
}

fn assert_inline(profiles: &[CompatibilityProfile], source: &str) {
    assert_native(profiles, source, &[]);
}

/// Like [`assert_native`] for a loop that meets a value the recording did not
/// see: the trace runs natively, leaves through a guard, and the interpreter
/// finishes the loop, raising its error if it has one.
fn assert_exits(profiles: &[CompatibilityProfile], source: &str) {
    for &profile in profiles {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert!(
            stats.exits_of(TraceExitKind::GuardFail) > 0,
            "{profile:?} {source} {stats:?}"
        );
    }
}

/// Like [`assert_agree`], and the loop also compiled and entered native code
/// (it may leave through guards: edge values do).
fn assert_traced(profiles: &[CompatibilityProfile], source: &str) {
    for &profile in profiles {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.traces_compiled > 0, "{profile:?} {source} {stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
    }
}

/// Checks only that both modes agree (edge values: whatever the trace does
/// with them, the answer is the interpreter's).
fn assert_agree(profiles: &[CompatibilityProfile], source: &str) {
    for &profile in profiles {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, _) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
    }
}

// rawget and rawset

#[test]
fn rawget_reads_the_array_and_the_fields() {
    assert_inline(
        &ALL,
        "local t = {10, 20, 30, 40, x = 5, y = 6.5} local s = 0 \
         for i = 1, 100000 do \
           local k = i % 4 + 1 \
           s = s + rawget(t, k) + rawget(t, 'x') + rawget(t, 'y') \
         end return s",
    );
}

#[test]
fn rawget_ignores_index_and_rawset_ignores_newindex() {
    // Every key is present, so the metamethods would not run for a plain
    // access either; they raise if they do run.
    let guard = "local mt = {__index = function() error('index') end, \
                              __newindex = function() error('newindex') end}";
    assert_inline(
        &ALL,
        &format!(
            "{guard} local t = setmetatable({{1, 2, 3, 4, x = 1}}, mt) local s = 0 \
             for i = 1, 100000 do \
               local k = i % 4 + 1 \
               rawset(t, k, rawget(t, k) + 1) \
               rawset(t, 'x', rawget(t, 'x') + 1) \
               s = s + rawget(t, k) \
             end return s, t[1], t[2], t.x"
        ),
    );
}

#[test]
fn rawset_returns_the_table() {
    assert_inline(
        &ALL,
        "local t = {1, 2, 3} local n = 0 \
         for i = 1, 100000 do \
           local r = rawset(t, 2, i) \
           if r == t then n = n + 1 end \
           if rawequal(rawset(t, 3, i), t) then n = n + 1 end \
         end return n, t[2], t[3]",
    );
}

#[test]
fn rawset_appends_to_a_table_without_a_metatable() {
    assert_inline(
        &ALL,
        "local t = {} \
         for i = 1, 200000 do rawset(t, i, i * 2) end \
         local s = 0 for i = 1, #t do s = s + t[i] end return s, #t",
    );
}

#[test]
fn rawget_of_a_field_found_through_index_leaves_the_trace() {
    // `t.x` finds the field in the class; `rawget` is nil. The recording sees
    // a present field, then the table changes.
    assert_exits(
        &ALL,
        "local class = {x = 9} \
         local t = setmetatable({x = 1}, {__index = class}) local s = 0 \
         for i = 1, 100000 do \
           if i == 90000 then t.x = nil end \
           s = s + (rawget(t, 'x') or 100) end return s",
    );
}

#[test]
fn rawget_of_a_slot_that_changes_type_leaves_the_trace() {
    assert_exits(
        &ALL,
        "local t = {1, 2, 3} \
         return pcall(function() local s = 0 for i = 1, 100000 do \
           s = s + rawget(t, 2) \
           if i == 99000 then t[2] = {} end end return s end)",
    );
}

#[test]
fn rawset_of_a_slot_that_changes_type_stores_raw_in_the_interpreter() {
    assert_exits(
        &ALL,
        "local t = setmetatable({1, 2, 3}, {__newindex = function() error('newindex') end}) \
         local s = 0 for i = 1, 100000 do \
           rawset(t, 2, i < 99000 and 5 or 'text') end \
         return t[2], rawget(t, 2)",
    );
}

#[test]
fn rawset_of_an_absent_key_on_a_table_with_newindex_stores_raw() {
    assert_agree(
        &ALL,
        "local seen = 0 \
         local t = setmetatable({}, {__newindex = function() seen = seen + 1 end}) \
         for i = 1, 20000 do rawset(t, i, i) end \
         local s = 0 for i = 1, 20000 do s = s + rawget(t, i) end return s, seen, #t",
    );
}

// rawlen

#[test]
fn rawlen_of_tables_and_strings() {
    assert_inline(
        &RAWLEN,
        "local t, s, n = {1, 2, 3, 4, 5}, 'hello', 0 \
         for i = 1, 100000 do n = n + rawlen(t) + rawlen(s) + rawlen('abc') end return n",
    );
}

#[test]
fn rawlen_ignores_len() {
    // The entry takes the border of a table that has `__len`, which `#`
    // would call.
    assert_inline(
        &RAWLEN,
        "local t = setmetatable({1, 2, 3}, {__len = function() return 100 end}) local n = 0 \
         for i = 1, 100000 do n = n + rawlen(t) end return n, #t",
    );
    assert_agree(
        &RAWLEN,
        "local t = setmetatable({1, 2, 3}, {__len = function() return 100 end}) local n = 0 \
         for i = 1, 20000 do n = n + rawlen(t) + #t end return n",
    );
}

// unpack

#[test]
fn unpack_of_a_fixed_length() {
    assert_inline(
        &ALL,
        "local unpack = unpack or table.unpack local t = {1, 2, 3, 4} local s = 0 \
         for i = 1, 100000 do local a, b, c, d = unpack(t) s = s + a + b + c + d end return s",
    );
}

#[test]
fn unpack_with_bounds() {
    assert_inline(
        &ALL,
        "local unpack = unpack or table.unpack local t = {1, 2, 3, 4, 5} local s = 0 \
         for i = 1, 100000 do \
           local a, b = unpack(t, 2, 3) \
           local c = unpack(t, 4) \
           local d, e, f = unpack(t, 1, 3) \
           s = s + a + b + c + d + e + f end return s",
    );
}

#[test]
fn unpack_in_open_result_position() {
    assert_inline(
        &ALL,
        "local unpack = unpack or table.unpack local t = {1, 2, 3} \
         local function add(a, b, c) return a + b + c end \
         local s = 0 for i = 1, 100000 do s = s + add(unpack(t)) end return s",
    );
    assert_inline(
        &ALL,
        "local unpack = unpack or table.unpack local t = {4, 5, 6} \
         local s = 0 for i = 1, 100000 do s = s + select('#', unpack(t)) end return s",
    );
}

#[test]
fn unpack_of_a_table_with_len_takes_the_border_where_the_library_does() {
    let source = "local unpack = unpack or table.unpack \
         local t = setmetatable({1, 2, 3}, {__len = function() return 2 end}) local s = 0 \
         for i = 1, 100000 do local a, b, c = unpack(t) s = s + a + b + (c or 0) end return s";
    assert_inline(
        &[CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21],
        source,
    );
    assert_agree(&ALL, source);
}

#[test]
fn unpack_of_an_empty_range_gives_nothing() {
    assert_inline(
        &ALL,
        "local unpack = unpack or table.unpack local t = {1, 2, 3} local s = 0 \
         for i = 1, 100000 do local a = unpack(t, 3, 2) if a == nil then s = s + 1 end end \
         return s",
    );
}

#[test]
fn unpack_of_a_table_whose_length_changes_leaves_the_trace() {
    assert_exits(
        &ALL,
        "local unpack = unpack or table.unpack local t = {1, 2, 3} local s = 0 \
         for i = 1, 100000 do \
           if i == 90000 then t[4] = 4 end \
           local a, b, c, d = unpack(t) s = s + a + b + c + (d or 0) end return s",
    );
}

#[test]
fn unpack_with_bounds_that_change_leaves_the_trace() {
    assert_exits(
        &ALL,
        "local unpack = unpack or table.unpack local t = {1, 2, 3, 4, 5} local s = 0 \
         for i = 1, 100000 do \
           local a, b = unpack(t, 1, i < 90000 and 2 or 3) s = s + a + b end return s",
    );
}

// getmetatable and setmetatable

#[test]
fn setmetatable_in_a_hot_loop_stays_native() {
    assert_native(
        &ALL,
        "local t, mt = {}, {} local n = 0 \
         for i = 1, 100000 do \
           local r = setmetatable(t, mt) \
           if r == t then n = n + 1 end \
           setmetatable(t, nil) \
         end return n, getmetatable(t)",
        &["setmetatable"],
    );
}

#[test]
fn getmetatable_gives_the_metatable_or_nil() {
    assert_native(
        &ALL,
        "local mt = {} local a, b = setmetatable({}, mt), {} local n = 0 \
         for i = 1, 100000 do \
           if getmetatable(a) == mt then n = n + 1 end \
           if getmetatable(b) == nil then n = n + 1 end \
           if rawequal(getmetatable(a), mt) then n = n + 1 end \
         end return n",
        &["getmetatable", "getmetatable_absent"],
    );
}

#[test]
fn getmetatable_of_a_string_is_the_string_metatable() {
    assert_native(
        &ALL,
        "local s, n = 'abc', 0 local smt = getmetatable('') \
         for i = 1, 100000 do \
           if getmetatable(s) == smt then n = n + 1 end \
           if getmetatable(i) == nil then n = n + 1 end \
         end return n",
        &["getmetatable", "getmetatable_absent"],
    );
}

#[test]
fn getmetatable_of_a_protected_metatable_is_the_field() {
    assert_agree(
        &ALL,
        "local t = setmetatable({}, {__metatable = 'locked'}) local n = 0 \
         for i = 1, 20000 do if getmetatable(t) == 'locked' then n = n + 1 end end return n",
    );
}

#[test]
fn a_protected_metatable_leaves_the_trace_and_the_interpreter_raises() {
    assert_exits(
        &ALL,
        "local t, plain, locked = {}, {}, {__metatable = 1} \
         return pcall(function() \
           for i = 1, 100000 do \
             if i == 99000 then setmetatable(t, locked) end \
             setmetatable(t, plain) \
           end end)",
    );
}

#[test]
fn a_metatable_with_a_finalizer_or_weak_mode_is_the_interpreters() {
    assert_agree(
        &ALL,
        "local plain, weak = {}, {__mode = 'k'} local t = {} local seen = 0 \
         local fin = {__gc = function() seen = seen + 1 end} \
         for i = 1, 20000 do \
           setmetatable(t, i < 19000 and plain or weak) \
         end \
         local keep = setmetatable({}, weak) \
         local made = {} \
         for i = 1, 20000 do setmetatable(made, i < 19000 and plain or fin) end \
         collectgarbage() \
         return getmetatable(t) == weak, getmetatable(made) == fin",
    );
}

#[test]
fn setmetatable_beside_a_field_it_does_not_touch_stays_native() {
    // The field is the table's own, so changing the metatable cannot change
    // what the read finds, and the helper moves no storage.
    assert_native(
        &ALL,
        "local t, mt = {n = 0, a = {}}, {} \
         for i = 1, 100000 do t.n = t.n + 1 setmetatable(t, mt) end return t.n",
        &["setmetatable"],
    );
}

#[test]
fn a_metatable_change_beside_an_index_read_gives_the_interpreters_result() {
    // `t.x` is found through `__index`, which the entry resolves once. The
    // call in the same loop changes it, so the trace must not read a stale
    // class: the recorder refuses the loop.
    let source = "local a, b = {__index = {x = 1}}, {__index = {x = 100}} \
         local t = setmetatable({}, a) local s = 0 \
         for i = 1, 50000 do \
           setmetatable(t, b) s = s + t.x \
           setmetatable(t, a) s = s + t.x \
         end return s";
    assert_agree(&ALL, source);
    for profile in ALL {
        let (_, lua) = outcome(profile, ExecutionMode::Tracing, source);
        let stats = lua.vm().jit_statistics().trace;
        assert_eq!(stats.native_entries, 0, "{profile:?} {stats:?}");
    }
}

#[test]
fn a_metatable_change_beside_a_length_gives_the_interpreters_result() {
    // The entry checks that the table has no `__len`; the swap gives it one
    // in the next iteration of the same run.
    let source = "local a, b = {}, {__len = function() return 100 end} \
         local t = setmetatable({1, 2, 3}, a) local m1, m2 = b, a local s = 0 \
         for i = 1, 50000 do \
           s = s + #t \
           setmetatable(t, m1) \
           m1, m2 = m2, m1 \
         end return s";
    assert_agree(&ALL, source);
    for profile in ALL {
        let (_, lua) = outcome(profile, ExecutionMode::Tracing, source);
        let stats = lua.vm().jit_statistics().trace;
        // Lua 5.1 and LuaJIT take the border whatever `__len` says, so the
        // swap changes nothing the trace leaned on.
        let ignores_len = matches!(
            profile,
            CompatibilityProfile::Lua51 | CompatibilityProfile::LuaJit21
        );
        if ignores_len {
            assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
            assert_eq!(stats.exits_of(TraceExitKind::GuardFail), 0, "{profile:?}");
        } else {
            // A recording either refuses the loop or checks which `__len`
            // runs on every iteration (a function handler is inlined), and
            // the swap sends each run to the interpreter: it never leans on
            // a stale class.
            assert!(
                stats.native_entries == 0 || stats.exits_of(TraceExitKind::GuardFail) > 0,
                "{profile:?} {stats:?}"
            );
        }
    }
}

#[test]
fn a_metatable_swap_beside_no_metatable_assumption_gives_the_interpreters_result() {
    // A hole read as nil assumes no `__index` can run.
    let source = "local a, b = {}, {__index = function() return 5 end} \
         local t = setmetatable({1, nil, 3}, a) local m1, m2 = b, a local s = 0 \
         for i = 1, 50000 do \
           s = s + (t[2] or 0) \
           setmetatable(t, m1) \
           m1, m2 = m2, m1 \
         end return s";
    assert_agree(&ALL, source);
}

/// Like [`assert_native`], and the loop made its tables without allocating
/// them: the constructors stayed virtual with their metatable.
fn assert_virtual(profiles: &[CompatibilityProfile], source: &str) {
    assert_native(profiles, source, &["plain_metatable"]);
    for &profile in profiles {
        let (_, lua) = outcome(profile, ExecutionMode::Tracing, source);
        let (tables, _) = lua.vm().runtime().elided_tables_total();
        assert!(
            tables > 1000,
            "{profile:?} {source}: {tables} tables elided"
        );
    }
}

#[test]
fn setmetatable_of_a_new_table_keeps_the_table_virtual() {
    assert_virtual(
        &ALL,
        "local mt = {__index = {}} local s = 0 \
         for i = 1, 100000 do \
           local c = setmetatable({name = i}, mt) \
           c.x = i c.y = i + 1 \
           s = s + c.name + c.x + c.y \
         end return s",
    );
}

#[test]
fn a_virtual_table_with_a_metatable_reads_absent_keys_without_index() {
    // The metatable has no `__index`, so an absent key is nil and the length
    // is the border.
    assert_virtual(
        &ALL,
        "local mt = {} local s = 0 \
         for i = 1, 100000 do \
           local c = setmetatable({i, i}, mt) \
           if c.missing == nil then s = s + 1 end \
           s = s + #c + c[1] \
         end return s",
    );
}

#[test]
fn an_exit_sets_the_metatable_of_a_virtual_table() {
    // The branch on `data[2]` fails after the table exists, and the
    // interpreter's other path asks for its metatable.
    let source = "local mt = {__index = {kind = 'K'}} local data = {5, 5, 5} \
         local s, seen = 0, 0 \
         for i = 1, 100000 do \
           local t = setmetatable({v = i}, mt) \
           if data[2] > 0 then s = s + t.v else \
             if rawequal(getmetatable(t), mt) and t.kind == 'K' then seen = seen + 1 end end \
           if i == 90000 then data[2] = -1 end \
         end return s, seen";
    assert_exits(&ALL, source);
    for profile in ALL {
        let (results, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(
            lua.vm().runtime().elided_tables_total().0 > 1000,
            "{profile:?}"
        );
        let Ok(values) = results else {
            panic!("{profile:?}")
        };
        assert_eq!(values.len(), 2, "{profile:?}");
        assert_ne!(values[1], Value::Integer(0), "{profile:?} {values:?}");
        assert_ne!(values[1], Value::Number(0.0), "{profile:?} {values:?}");
    }
}

#[test]
fn a_metatable_that_gains_newindex_sends_the_call_to_the_interpreter() {
    let source = "local mt = {} local s, set = 0, 0 \
         for i = 1, 100000 do \
           local c = setmetatable({}, mt) \
           c.x = i \
           s = s + c.x \
           if i == 90000 then \
             mt.__newindex = function(t, k, v) set = set + 1 rawset(t, k, v * 2) end end \
         end return s, set";
    assert_exits(&ALL, source);
}

/// Like [`assert_exits`] for a loop whose metatable gains a metamethod the
/// loop never leans on: the trace leaves once for the branch that adds it
/// and keeps running natively, instead of leaving through the check of the
/// `setmetatable` call in every later iteration.
fn assert_stays_native(profiles: &[CompatibilityProfile], source: &str) {
    for &profile in profiles {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert_eq!(stats.total_aborts(), 0, "{profile:?} {source} {stats:?}");
        let exits = stats.exits_of(TraceExitKind::GuardFail);
        assert!(exits <= 3, "{profile:?} {source}: {exits} exits {stats:?}");
        let (tables, _) = lua.vm().runtime().elided_tables_total();
        assert!(
            tables > 1000,
            "{profile:?} {source}: {tables} tables elided"
        );
    }
}

#[test]
fn a_metatable_that_gains_an_unrelated_metamethod_stays_native() {
    // The loop reads an absent field, which leans on `__index` being
    // absent, and never takes a length or stores a new key.
    let gain = |field: &str, function: &str| {
        format!(
            "local mt = {{}} local s = 0 \
             for i = 1, 100000 do \
               local c = setmetatable({{}}, mt) \
               if c.x == nil then s = s + 1 end \
               if i == 50000 then mt.{field} = {function} end \
             end return s"
        )
    };
    assert_stays_native(&ALL, &gain("__len", "function() return 9 end"));
    assert_stays_native(&ALL, &gain("__call", "function() return 9 end"));
    assert_stays_native(&ALL, &gain("__tostring", "function() return 'x' end"));
    // A store of a new key leans on `__newindex` instead.
    assert_stays_native(
        &ALL,
        "local mt = {} local s = 0 \
         for i = 1, 100000 do \
           local c = setmetatable({}, mt) \
           c.x = i s = s + c.x \
           if i == 50000 then mt.__index = function() return 1 end end \
         end return s",
    );
    assert_stays_native(
        &ALL,
        "local mt = {} local s = 0 \
         for i = 1, 100000 do \
           local c = setmetatable({i, i}, mt) \
           s = s + #c \
           if i == 50000 then mt.__newindex = function() end end \
         end return s",
    );
}

#[test]
fn a_metatable_that_gains_the_metamethod_the_loop_leans_on_leaves_the_trace() {
    assert_exits(
        &ALL,
        "local mt = {} local s = 0 \
         for i = 1, 100000 do \
           local c = setmetatable({}, mt) \
           if c.x == nil then s = s + 1 end \
           if i == 50000 then mt.__index = function() return 5 end end \
         end return s",
    );
    // `#` never asks `__len` in Lua 5.1 and LuaJIT, so gaining it is unrelated.
    let length = "local mt = {} local s = 0 \
         for i = 1, 100000 do \
           local c = setmetatable({i, i}, mt) \
           s = s + #c \
           if i == 50000 then mt.__len = function() return 100 end end \
         end return s";
    assert_stays_native(
        &[CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21],
        length,
    );
    assert_exits(
        &[
            CompatibilityProfile::Lua52,
            CompatibilityProfile::Lua53,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::Lua55,
        ],
        length,
    );
}

#[test]
fn a_metatable_that_gains_gc_or_mode_sends_the_call_to_the_interpreter() {
    for field in ["__gc", "__mode"] {
        let source = format!(
            "local mt = {{}} local s = 0 \
             for i = 1, 100000 do \
               local c = setmetatable({{}}, mt) \
               c.x = i \
               s = s + c.x \
               if i == 90000 then mt.{field} = {} end \
             end return s",
            if field == "__gc" {
                "function() end"
            } else {
                "'k'"
            }
        );
        assert_exits(&ALL, &source);
    }
}

#[test]
fn a_virtual_table_whose_metatable_has_the_metamethod_escapes() {
    // A store of an absent key, a read of one and a length each reach a
    // metamethod the recorder does not follow, so the loop stays correct.
    assert_agree(
        &ALL,
        "local seen = 0 \
         local mt = {__newindex = function(t, k, v) seen = seen + 1 rawset(t, k, v) end} \
         local s = 0 \
         for i = 1, 50000 do \
           local c = setmetatable({}, mt) c.x = i s = s + c.x end \
         return s, seen",
    );
    assert_agree(
        &ALL,
        "local mt = {__index = function(t, k) return 7 end} local s = 0 \
         for i = 1, 50000 do local c = setmetatable({a = i}, mt) s = s + c.a + c.b end return s",
    );
    assert_agree(
        &ALL,
        "local mt = {__len = function() return 9 end} local s = 0 \
         for i = 1, 50000 do local c = setmetatable({i}, mt) s = s + #c end return s",
    );
}

#[test]
fn a_virtual_table_that_escapes_with_a_metatable_gives_the_interpreters_result() {
    assert_agree(
        &ALL,
        "local mt = {__index = {n = 1}} local keep = {} local s = 0 \
         for i = 1, 50000 do \
           local c = setmetatable({v = i}, mt) \
           keep[i % 8 + 1] = c \
           s = s + c.v end \
         return s, getmetatable(keep[1]) == mt, keep[2].n",
    );
}

#[test]
fn setmetatable_with_a_non_table_raises_the_interpreters_error() {
    assert_exits(
        &ALL,
        "local t, mt = {}, {} local box = {mt} \
         return pcall(function() for i = 1, 100000 do \
           if i == 99000 then box[1] = 5 end \
           setmetatable(t, box[1]) end end)",
    );
}

// bit and bit32

/// Operands at the edges of the conversions: powers of two, ties, signs of
/// zero, NaN and the infinities.
const EDGES: &str = "{0, 1, -1, 2, 255, 2^31, -2^31, 2^31 - 1, 2^32, 2^32 + 1, -2^32, 2^53, \
                     -2^53, 2^51, 2^51 + 1, -2^51, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, -0.0, \
                     0/0, 1/0, -1/0, 3.7, 1e10, 123456789, 0xdeadbeef, 2^63, -2^63, 1e300}";

/// Operands every profile's `bit32` accepts: whole numbers.
const WHOLE: &str = "{0, 1, -1, 2, 255, 2^31, -2^31, 2^31 - 1, 2^32, 2^32 + 1, -2^32, 2^53, \
                     -2^53, 2^51, 2^51 + 1, -2^51, 123456789, 0xdeadbeef, 0x80000000, 7}";

/// A loop that applies `library.name` to every pair of `values` many times
/// and returns the results as text, so a wrong bit shows.
fn bit_edges(library: &str, name: &str, arity: usize, values: &str) -> String {
    let call = if arity == 1 {
        format!("{library}.{name}(a)")
    } else {
        format!("{library}.{name}(a, b)")
    };
    format!(
        "local v = {values} local out, n = {{}}, 0 \
         for round = 1, 60 do for i = 1, #v do for j = 1, #v do \
           local a, b = v[i], v[j] \
           n = n + 1 out[n] = {call} \
         end end end \
         for k = 1, n do out[k] = tostring(out[k]) end \
         return table.concat(out, ',')"
    )
}

#[test]
fn bit_functions_on_numbers() {
    assert_inline(
        &[CompatibilityProfile::LuaJit21],
        "local s = 0 for i = 1, 100000 do \
           s = s + bit.band(i, 0xff) + bit.bor(i, 0x100) + bit.bxor(i, 0x55) \
             + bit.lshift(i, 3) + bit.rshift(i, 2) + bit.arshift(-i, 4) \
             + bit.rol(i, 5) + bit.ror(i, 7) + bit.bnot(i) + bit.tobit(i * 65536.5) end \
         return s",
    );
    assert_inline(
        &[CompatibilityProfile::LuaJit21],
        "local s = 0 for i = 1, 100000 do \
           s = s + bit.band(i, 0xff, 0x3c, 0x1f) + bit.bor(i, 1, 2) + bit.bxor(i, 3, 5, 9) end \
         return s",
    );
}

#[test]
fn bit_on_a_profile_with_integers_gives_floats() {
    assert_inline(
        &[
            CompatibilityProfile::Lua53,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::Lua55,
            CompatibilityProfile::Lua51,
            CompatibilityProfile::Lua52,
        ],
        "local s, last = 0 for i = 1, 100000 do \
           last = bit.band(i, 0xff) + bit.lshift(i, 31) + bit.rol(i, 33) + bit.arshift(-i, 1) \
           s = s + last end \
         return s, last, math.type and math.type(last)",
    );
}

#[test]
fn bit_edge_values_match_the_interpreter() {
    for name in ["tobit", "bnot"] {
        assert_traced(&ALL, &bit_edges("bit", name, 1, EDGES));
    }
    for name in [
        "band", "bor", "bxor", "lshift", "rshift", "arshift", "rol", "ror",
    ] {
        assert_traced(&ALL, &bit_edges("bit", name, 2, EDGES));
    }
}

#[test]
fn bit_shift_counts() {
    assert_inline(
        &[CompatibilityProfile::LuaJit21],
        "local v, s = {0, 1, 31, 32, 33, -1, 63, 64, 1e10, 2^31}, 0 \
         for i = 1, 100000 do local c = v[i % #v + 1] \
           s = s + bit.lshift(0x80000001, c) + bit.rshift(0x80000001, c) \
             + bit.arshift(0x80000001, c) + bit.rol(0x80000001, c) + bit.ror(0x80000001, c) end \
         return s",
    );
}

#[test]
fn bit_with_a_string_or_missing_argument_raises_the_interpreters_error() {
    assert_exits(
        &[CompatibilityProfile::LuaJit21],
        "local box = {7} return pcall(function() local s = 0 \
           for i = 1, 100000 do \
             if i == 99000 then box[1] = {} end \
             s = s + bit.band(i, box[1]) end return s end)",
    );
}

#[test]
fn bit32_functions_on_numbers() {
    assert_inline(
        &BIT32,
        "local s = 0 for i = 1, 100000 do \
           s = s + bit32.band(i, 0xff) + bit32.bor(i, 0x100) + bit32.bxor(i, 0x55) \
             + bit32.lshift(i, 3) + bit32.rshift(i, 2) + bit32.arshift(i, 4) \
             + bit32.lrotate(i, 5) + bit32.rrotate(i, 7) + bit32.bnot(i) \
             + bit32.extract(i, 3, 5) + bit32.replace(i, 21, 4, 6) \
             + (bit32.btest(i, 8) and 1 or 0) + bit32.band(i, 0xff, 0x3c) end \
         return s",
    );
}

#[test]
fn bit32_arshift_of_a_negative_word() {
    assert_inline(
        &BIT32,
        "local s = 0 for i = 1, 100000 do \
           s = s + bit32.arshift(0x80000000 + i, 4) + bit32.arshift(0xfffffff0, 28) end \
         return s",
    );
}

#[test]
fn bit32_edge_values_match_the_interpreter() {
    // The float profile reads every number; the integer one raises for a
    // fraction, so its operands are whole.
    assert_traced(
        &[CompatibilityProfile::Lua52],
        &bit_edges("bit32", "bnot", 1, EDGES),
    );
    assert_traced(&BIT32, &bit_edges("bit32", "bnot", 1, WHOLE));
    for name in [
        "band", "bor", "bxor", "lshift", "rshift", "arshift", "lrotate", "rrotate", "btest",
    ] {
        assert_traced(
            &[CompatibilityProfile::Lua52],
            &bit_edges("bit32", name, 2, EDGES),
        );
        assert_traced(&BIT32, &bit_edges("bit32", name, 2, WHOLE));
    }
}

#[test]
fn bit32_shift_counts() {
    assert_inline(
        &BIT32,
        "local v, s = {0, 1, 31, 32, 33, -1, -31, -32, -33, 63, 64, -64}, 0 \
         for i = 1, 100000 do local c = v[i % #v + 1] \
           s = s + bit32.lshift(0x80000001, c) + bit32.rshift(0x80000001, c) \
             + bit32.lrotate(0x80000001, c) + bit32.rrotate(0x80000001, c) end \
         return s",
    );
}

#[test]
fn bit32_arshift_counts_outside_the_word_leave_the_trace() {
    assert_exits(
        &BIT32,
        "local s = 0 for i = 1, 100000 do \
           s = s + bit32.arshift(0x80000001, i < 90000 and 3 or 33) end return s",
    );
}

#[test]
fn bit32_field_errors_leave_the_trace_and_the_interpreter_raises() {
    assert_exits(
        &BIT32,
        "return pcall(function() local s = 0 for i = 1, 100000 do \
           s = s + bit32.extract(i, 3, i < 99000 and 5 or 40) end return s end)",
    );
    assert_exits(
        &BIT32,
        "return pcall(function() local s = 0 for i = 1, 100000 do \
           s = s + bit32.replace(i, 1, i < 99000 and 3 or 33, 1) end return s end)",
    );
    assert_exits(
        &BIT32,
        "return pcall(function() local s = 0 for i = 1, 100000 do \
           s = s + bit32.extract(i, i < 99000 and 3 or -1, 2) end return s end)",
    );
}

#[test]
fn bit32_with_a_fractional_number_raises_on_integer_profiles() {
    assert_exits(
        &[CompatibilityProfile::Lua53],
        "return pcall(function() local s = 0 for i = 1, 100000 do \
           s = s + bit32.band(i < 99000 and i or i + 0.5, 255) end return s end)",
    );
}
