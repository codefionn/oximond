//! `==` and `~=` of strings in a trace: the comparison of a string with a
//! string constant (`TestEqualK`, the shape of `if name ~= "void"` in
//! Mineclonia's template loader) and of two strings in registers (`Equal`,
//! `TestEqual`). Two strings are equal when their bytes are, which no lane
//! comparison tells, so the trace calls the `string.equal` helper; two root
//! string constants fold, and a value of another type is never equal to a
//! string, which the entry or read guards decide. `__eq` never runs for a
//! string. Every profile gives the interpreter's results, at both
//! optimization levels.

use oximond::compat::TracingOptLevel;
use oximond::runtime::Value;
use oximond::vm::{TraceAbortReason, TraceExitKind, TraceStatistics};
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

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
    level: TracingOptLevel,
    source: &str,
) -> (Result<Vec<Value>, String>, Lua) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .tracing_opt_flags(level.flags())
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let results = lua.exec(source).map_err(|error| format!("{error:?}"));
    (results, lua)
}

fn show(result: &Result<Vec<Value>, String>) -> String {
    format!("{result:?}")
}

/// Runs `source` in the interpreter and in the tracing mode at both levels of
/// every profile, checks that the results agree, and hands the tracing run's
/// statistics and helper names to `check`.
fn each(source: &str, check: impl Fn(CompatibilityProfile, &TraceStatistics, &[&str])) {
    for profile in ALL {
        let (expected, _) = outcome(
            profile,
            ExecutionMode::Interpreter,
            TracingOptLevel::O1,
            source,
        );
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        for level in [TracingOptLevel::O0, TracingOptLevel::O1] {
            let (actual, lua) = outcome(profile, ExecutionMode::Tracing, level, source);
            assert_eq!(
                show(&actual),
                show(&expected),
                "{profile:?} {level:?} {source}"
            );
            let stats = lua.vm().jit_statistics().trace;
            check(profile, &stats, &lua.vm().trace_helper_calls());
        }
    }
}

/// The loop compiled, ran natively with no abort, and compares bytes through
/// `string.equal`.
fn assert_native_equal(source: &str) {
    each(source, |profile, stats, helpers| {
        assert!(stats.traces_compiled > 0, "{profile:?} {source} {stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert!(stats.iterations > 1000, "{profile:?} {source} {stats:?}");
        assert_eq!(stats.total_aborts(), 0, "{profile:?} {source} {stats:?}");
        assert!(
            helpers.contains(&"string.equal"),
            "{profile:?} {source} {helpers:?}"
        );
    });
}

/// The loop ran natively and the string comparison was never the reason a
/// recording gave up, though a guard may leave the trace.
fn assert_native_exits(source: &str) {
    each(source, |profile, stats, _| {
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert_eq!(
            stats.aborts(TraceAbortReason::UnsupportedOperand),
            0,
            "{profile:?} {source} {stats:?}"
        );
    });
}

#[test]
fn a_string_read_from_a_table_against_a_string_constant_compares_bytes() {
    // The names are made at run time, so none is the constant's allocation.
    assert_native_equal(
        "local names = {} \
         for k = 1, 10 do names[k] = ('x' .. (k == 3 and 'void' or 'node' .. k)):sub(2) end \
         local kept, voids = 0, 0 \
         for i = 1, 20000 do \
           local name = names[i % 10 + 1] \
           if name ~= 'void' then kept = kept + #name else voids = voids + 1 end \
           if name == 'node7' then kept = kept + 1000 end \
         end return kept, voids",
    );
}

#[test]
fn two_strings_in_registers_compare_bytes() {
    // `Equal` writes the outcome; `TestEqual` branches on it.
    assert_native_equal(
        "local a, b = {}, {} \
         for k = 1, 6 do a[k] = ('k' .. k % 3):sub(1) b[k] = ('k' .. k % 2):sub(1) end \
         local n, m = 0, 0 \
         for i = 1, 20000 do \
           local x, y = a[i % 6 + 1], b[i % 5 + 1] \
           local same = x == y \
           if same then n = n + 1 end \
           if x ~= y then m = m + 1 end \
           if x == x then m = m + 2 end \
         end return n, m",
    );
}

#[test]
fn a_string_against_a_string_the_loop_makes() {
    assert_native_equal(
        "local key, n = 'k5', 0 \
         for i = 1, 20000 do \
           local s = 'k' .. (i % 10) \
           if s == key then n = n + 1 end \
           if s ~= 'k7' then n = n + 10 end \
         end return n",
    );
}

#[test]
fn two_string_constants_fold() {
    // Two root strings compare by their bytes at recording time.
    each(
        "local n = 0 \
         for i = 1, 100000 do \
           local a, b = 'same', 'same' \
           if a == b then n = n + 1 end \
           if a == 'other' then n = n + 1000 end \
         end return n",
        |profile, stats, helpers| {
            assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
            assert_eq!(stats.total_aborts(), 0, "{profile:?} {stats:?}");
            assert!(
                !helpers.contains(&"string.equal"),
                "{profile:?} {helpers:?}"
            );
        },
    );
}

#[test]
fn string_equality_never_asks_eq() {
    // A string metatable's `__eq` never runs for two strings, nor for a
    // string and a table whose metatable has one.
    assert_native_exits(
        "getmetatable('').__eq = function() error('__eq ran') end \
         local mt = {__eq = function() error('__eq ran') end} \
         local values = {} \
         for k = 1, 8 do values[k] = ('v' .. k % 4):sub(1) end \
         local n = 0 \
         for i = 1, 20000 do \
           local v = values[i % 8 + 1] \
           if v == 'v1' then n = n + 1 end \
           if v == values[1] then n = n + 10 end \
           if i == 15000 then values[2] = setmetatable({}, mt) end \
         end return n",
    );
}

#[test]
fn a_value_of_another_type_leaves_the_trace() {
    // A number, a boolean, nil (a hole) and a table where the recording saw
    // a string: never equal to the constant, and the interpreter decides.
    for late in ["42", "false", "nil", "{}"] {
        assert_native_exits(&format!(
            "local names = {{}} \
             for k = 1, 10 do names[k] = ('n' .. k):sub(1) end \
             local n = 0 \
             for i = 1, 20000 do \
               if i == 12000 then names[4] = {late} end \
               local name = names[i % 10 + 1] \
               if name ~= 'n4' then n = n + 1 else n = n + 100 end \
             end return n"
        ));
    }
}

#[test]
fn a_table_swapped_in_the_loop_gives_the_new_strings() {
    assert_native_exits(
        "local first, second = {}, {} \
         for k = 1, 10 do first[k] = ('a' .. k):sub(1) second[k] = ('b' .. k):sub(1) end \
         second[5] = ('x' .. 'a5'):sub(2) \
         local names, n = first, 0 \
         for i = 1, 20000 do \
           if i == 10000 then names = second end \
           local name = names[i % 10 + 1] \
           if name == 'a5' then n = n + 1 end \
         end return n",
    );
}

#[test]
fn ordered_string_comparisons_still_agree() {
    // `<` and `<=` of strings stay with the interpreter.
    each(
        "local words = {'pear', 'apple', 'fig', 'apple'} \
         local n = 0 \
         for i = 1, 20000 do \
           local w = words[i % 4 + 1] \
           if w < 'banana' then n = n + 1 end \
           if w <= 'fig' then n = n + 10 end \
         end return n",
        |_, _, _| {},
    );
}

#[test]
fn a_rare_arm_that_skips_a_call_result_does_not_change_the_entry_types() {
    // The template loop's shape: a `pairs` loop in a rare arm leaves its
    // third call result in a temporary the common arm uses for a number.
    // That temporary is dead at the header (the call writes it), so the
    // void arm's interpreter iterations do not make the trace's entry
    // check fail.
    let source = "local names = {} \
         for k = 1, 10 do names[k] = ('x' .. (k == 3 and 'void' or 'node' .. k)):sub(2) end \
         local ids = {} for k = 1, 10 do ids[names[k]] = 100 + k end \
         local function lookup(s) return ids[s] end \
         local function convert(data) \
           for i = 1, #data.nodes do \
             local id = data.nodes[i] \
             if i % 97 == 0 then \
               for _, list in pairs(data.meta.inventory) do \
                 if type(list) ~= 'table' then return nil end \
               end \
             end \
             local name = data.names[id] \
             if name ~= 'void' then \
               local ok, cid = pcall(lookup, name) \
               data.nodes[i] = cid * 256 + id \
             else \
               data.nodes[i] = 0 \
             end \
           end \
           return data \
         end \
         local s = 0 \
         for r = 1, 20 do \
           local data = {nodes = {}, names = names, meta = {inventory = {main = {}}}} \
           for i = 1, 3000 do data.nodes[i] = (i * 7 + r) % 10 + 1 end \
           convert(data) \
           for i = 1, 3000 do s = s + data.nodes[i] end \
         end return s";
    each(source, |profile, stats, _| {
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert!(stats.iterations > 30000, "{profile:?} {stats:?}");
        assert_eq!(
            stats.exits_of(TraceExitKind::EntryMismatch),
            0,
            "{profile:?} {stats:?}"
        );
    });
}
