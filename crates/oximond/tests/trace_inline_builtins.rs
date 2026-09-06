//! Library functions a trace records as operations of its own, with no call:
//! `type`, `rawequal`, `select`, `assert`, `math.floor`, `math.ceil`,
//! `math.abs`, `math.tointeger`, `math.type` and `string.byte` with several
//! results. The loop compiles without an abort, enters native code, makes no
//! call into the library, and gives the interpreter's results in every
//! profile that has the function. A value the recording did not see leaves
//! the trace and the interpreter makes the call.

use oximond::runtime::Value;
use oximond::vm::TraceExitKind;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const ALL: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

/// The profiles with tagged integers (`math.tointeger`, `math.type`).
const INTEGERS: [CompatibilityProfile; 3] = [
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
/// traced and ran natively, made no call into the library, and that the
/// results agree.
fn assert_inline(profiles: &[CompatibilityProfile], source: &str) {
    for &profile in profiles {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.traces_compiled > 0, "{profile:?} {source} {stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert_eq!(stats.total_aborts(), 0, "{profile:?} {source} {stats:?}");
        assert_eq!(
            lua.vm().trace_external_calls(),
            0,
            "{profile:?} {source}: {:?}",
            lua.vm().trace_helper_calls()
        );
    }
}

/// [`assert_inline`] for a loop that makes no call into the library in the
/// code that runs, though the recording made some (a `string.sub` whose
/// string the optimizer dropped).
fn assert_inline_optimized(profiles: &[CompatibilityProfile], source: &str) {
    for &profile in profiles {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.traces_compiled > 0, "{profile:?} {source} {stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert_eq!(stats.total_aborts(), 0, "{profile:?} {source} {stats:?}");
        assert_eq!(
            lua.vm().trace_compiled_external_calls(),
            0,
            "{profile:?} {source}: {:?}",
            lua.vm().trace_helper_calls()
        );
    }
}

/// Like [`assert_inline`] for a loop that meets a value the recording did
/// not see: the trace runs natively, leaves through a guard, and the
/// interpreter finishes the loop, raising its error if it has one.
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

/// Checks only that both modes agree (edge values: whatever the trace does
/// with them, the answer is the interpreter's).
fn assert_agree(profiles: &[CompatibilityProfile], source: &str) {
    for &profile in profiles {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, _) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
    }
}

#[test]
fn type_of_each_kind_of_value_is_a_constant() {
    assert_inline(
        &ALL,
        "local t, f, s, n = {}, print, 'abc', 0 \
         for i = 1, 100000 do \
           if type(i) == 'number' then n = n + 1 end \
           if type(i / 3) == 'number' then n = n + 2 end \
           if type(t) == 'table' then n = n + 4 end \
           if type(f) == 'function' then n = n + 8 end \
           if type(s) == 'string' then n = n + 16 end \
           if type(nil) == 'nil' then n = n + 32 end \
           if type(i > 5) == 'boolean' then n = n + 64 end \
           if type(i) ~= 'string' then n = n + 128 end \
         end return n",
    );
}

#[test]
fn type_result_survives_the_loop() {
    assert_inline(
        &ALL,
        "local n, t = 0, {} \
         for i = 1, 100000 do local name = type(t) n = n + #name end return n",
    );
}

#[test]
fn rawequal_compares_without_a_call() {
    assert_inline(
        &ALL,
        "local t, n = {}, 0 \
         for i = 1, 100000 do \
           if rawequal(i, i) then n = n + 1 end \
           if rawequal(i, 1.5) then n = n + 1000 end \
           if rawequal(t, t) then n = n + 2 end \
           if rawequal(t, 1) then n = n + 1000 end \
           if rawequal(i > 3, true) then n = n + 4 end \
           if rawequal(nil, nil) then n = n + 8 end \
         end return n",
    );
}

#[test]
fn rawequal_of_distinct_tables_ignores_eq() {
    assert_inline(
        &ALL,
        "local a, b = {}, {} \
         local c = setmetatable({}, {__eq = function() return true end}) \
         local d = setmetatable({}, getmetatable(c)) \
         local n = 0 \
         for i = 1, 100000 do \
           if rawequal(a, b) then n = n + 1000 end \
           if rawequal(a, a) then n = n + 1 end \
           if rawequal(c, d) then n = n + 1000 end \
           if rawequal(c, c) then n = n + 2 end \
         end return n",
    );
}

#[test]
fn rawequal_of_an_integer_and_a_float() {
    assert_inline(
        &ALL,
        "local n = 0 for i = 1, 100000 do \
           if rawequal(i, i + 0.0) then n = n + 1 end end return n",
    );
}

#[test]
fn select_counts_and_picks_varargs() {
    assert_inline(
        &ALL,
        "local function f(...) \
           local a, b = select(2, ...) \
           local c = select(-1, ...) \
           return select('#', ...) + a + b + c + (select(1, ...)) end \
         local s = 0 for i = 1, 100000 do s = s + f(i, 2, 3, 4) end return s",
    );
}

#[test]
fn select_past_the_end_gives_nothing() {
    assert_inline(
        &ALL,
        "local function f(...) \
           local a, b = select(4, ...) \
           return select('#', select(4, ...)) + (a or 7) + (b or 11) end \
         local s = 0 for i = 1, 100000 do s = s + f(i, 2, 3) end return s",
    );
}

#[test]
fn select_in_open_result_position() {
    assert_inline(
        &ALL,
        "local function g(...) return select(2, ...) end \
         local function pick(a, b, c) return a + b + c end \
         local s = 0 for i = 1, 100000 do s = s + pick(g(i, 2, 3, 4)) end return s",
    );
}

#[test]
fn select_with_a_count_of_zero() {
    assert_inline(
        &ALL,
        "local function f(...) return select('#', ...) end \
         local s = 0 for i = 1, 100000 do s = s + f() + f(i) + f(i, i) end return s",
    );
}

#[test]
fn select_of_fixed_arguments() {
    assert_inline(
        &ALL,
        "local s = 0 for i = 1, 100000 do \
           s = s + select('#', i, 2, 3) + select(2, i, 20, 30) + select(-1, i, 5, 6) end return s",
    );
}

#[test]
fn select_with_a_zero_index_raises_the_interpreters_error() {
    assert_exits(
        &ALL,
        "local function f(k, ...) return (select(k, ...)) end \
         return pcall(function() local s = 0 for i = 1, 100000 do \
           s = s + f(i < 99000 and 2 or 0, i, 5, 6) end return s end)",
    );
}

#[test]
fn select_with_an_index_that_changes_leaves_the_trace() {
    assert_exits(
        &ALL,
        "local function f(k, ...) return (select(k, ...)) end \
         local s = 0 for i = 1, 100000 do s = s + f(i < 99000 and 2 or 3, i, 5, 6) end return s",
    );
}

#[test]
fn assert_is_a_truthiness_guard() {
    assert_inline(
        &ALL,
        "local t, s = {1, 2}, 0 \
         for i = 1, 100000 do \
           assert(i > 0) \
           assert(t, 'a table') \
           local a, b = assert(i, 'msg') \
           s = s + a + #b + select('#', assert(i, 1, 2)) \
         end return s",
    );
}

#[test]
fn assert_that_fails_raises_the_interpreters_error() {
    assert_exits(
        &ALL,
        "return pcall(function() local n = 0 for i = 1, 100000 do \
           assert(i < 99000, 'boom at ' .. 'the end') n = n + 1 end return n end)",
    );
    assert_exits(
        &ALL,
        "return pcall(function() local n = 0 for i = 1, 100000 do \
           assert(i < 99000) n = n + 1 end return n end)",
    );
    assert_exits(
        &ALL,
        "return pcall(function() local n = 0 for i = 1, 100000 do \
           assert(i < 99000, 42) n = n + 1 end return n end)",
    );
}

#[test]
fn floor_and_ceil_of_floats_and_integers() {
    assert_inline(
        &ALL,
        "local n = 0 for i = 1, 100000 do \
           n = n + math.floor(i / 3) + math.ceil(i / 7) + math.floor(-i / 4) \
             + math.ceil(-i / 4) + math.floor(i) + math.ceil(i) end return n",
    );
}

#[test]
fn floor_result_keeps_the_interpreters_number_type() {
    // The subtype is an integer on 5.3 and later and a float elsewhere.
    assert_inline(
        &ALL,
        "local last, c for i = 1, 100000 do last = math.floor(i / 2) c = math.ceil(i / 2) end \
         return last, c, math.type and math.type(last), 1 / last",
    );
}

#[test]
fn floor_of_negative_zero_and_halves() {
    assert_inline(
        &ALL,
        "local a, b, c, d for i = 1, 100000 do \
           a = math.floor(-0.0) b = math.ceil(-0.5) c = math.floor(0.5) d = math.ceil(-0.0) end \
         return a, b, c, d, 1 / a, 1 / b, 1 / c, 1 / d",
    );
}

#[test]
fn floor_and_ceil_edge_values_match_the_interpreter() {
    assert_agree(
        &ALL,
        "local v = {-0.0, 0.5, -0.5, 0/0, 1/0, -1/0, 2^63, -2^63, 1e300, -1e300, 2^53, \
                    9007199254740993, 1e15 + 0.5, -1e15 - 0.5, 0.1, -0.9, 4.0, -4.0} \
         local out = {} \
         for round = 1, 3000 do for k = 1, #v do \
           local x = v[k] out[k] = tostring(math.floor(x)) .. '/' .. tostring(math.ceil(x)) end end \
         return table.concat(out, ' ')",
    );
    assert_agree(
        &ALL,
        "local f = {2^63, -2^63, 2^62, -2^62 - 1, 1e300, 0/0} local out = {} \
         for round = 1, 20000 do local k = round % #f + 1 \
           out[k] = tostring(math.floor(f[k])) .. tostring(math.ceil(f[k])) end \
         return table.concat(out, ' ')",
    );
}

#[test]
fn floor_and_ceil_of_integer_edges() {
    assert_agree(
        &INTEGERS,
        "local v = {math.mininteger, math.maxinteger, 0, -1, 1, 2^53 // 1 | 0} local out = {} \
         for round = 1, 3000 do for k = 1, #v do \
           out[k] = tostring(math.floor(v[k])) .. '/' .. tostring(math.ceil(v[k])) end end \
         return table.concat(out, ' ')",
    );
}

#[test]
fn a_float_that_leaves_the_integers_leaves_the_trace() {
    // The loop records with floats that become integers, then meets one
    // that does not fit and stays a float.
    assert_exits(
        &INTEGERS,
        "local n = 0 for i = 1, 100000 do \
           local x = i < 99000 and i / 2 or 2^70 n = n + math.floor(x) end return n",
    );
    assert_exits(
        &INTEGERS,
        "local last for i = 1, 100000 do \
           local x = i < 99000 and i / 2 or 0/0 last = math.ceil(x) end return last",
    );
}

#[test]
fn abs_of_integers_and_floats() {
    assert_inline(
        &ALL,
        "local n = 0 for i = 1, 100000 do \
           n = n + math.abs(i - 50000) + math.abs((i - 50000) / 3) + math.abs(-i) end return n",
    );
}

#[test]
fn abs_edge_values() {
    assert_agree(
        &ALL,
        "local v = {-0.0, 0.0, -1.5, 1/0, -1/0, 0/0, -1e300} local out = {} \
         for round = 1, 3000 do for k = 1, #v do out[k] = tostring(math.abs(v[k])) end end \
         return table.concat(out, ' ')",
    );
    assert_agree(
        &INTEGERS,
        "local v = {math.mininteger, math.maxinteger, -1, 0, 5} local out = {} \
         for round = 1, 3000 do for k = 1, #v do out[k] = tostring(math.abs(v[k])) end end \
         return table.concat(out, ' ')",
    );
}

#[test]
fn abs_of_the_most_negative_integer_wraps() {
    assert_inline(
        &INTEGERS,
        "local m, x = math.mininteger, 0 for i = 1, 100000 do x = math.abs(m) end \
         return x, math.abs(m) == m",
    );
}

#[test]
fn tointeger_and_math_type() {
    assert_inline(
        &INTEGERS,
        "local n, t = 0, {} for i = 1, 100000 do \
           n = n + math.tointeger(i) + math.tointeger(i * 1.0) \
             + (math.tointeger(i + 0.5) or 7) + (math.tointeger(0/0) or 9) \
           if math.type(i) == 'integer' then n = n + 1 end \
           if math.type(i / 2) == 'float' then n = n + 2 end \
           if math.type('x') == nil then n = n + 4 end \
           if math.type(t) == nil then n = n + 8 end \
         end return n",
    );
}

#[test]
fn tointeger_edge_values() {
    assert_agree(
        &INTEGERS,
        "local v = {3.0, 3.5, 2^63, -2^63, -0.0, 1/0, -1/0, 0/0, 2^53, 1e300, '3', '3.0', 'x', \
                    true, {}, math.maxinteger, math.mininteger} local out = {} \
         for round = 1, 3000 do for k = 1, #v do \
           out[k] = tostring(math.tointeger(v[k])) .. ':' .. tostring(math.type(v[k])) end end \
         return table.concat(out, ' ')",
    );
}

#[test]
fn tointeger_of_a_float_that_stops_being_integral_leaves() {
    assert_exits(
        &INTEGERS,
        "local n = 0 for i = 1, 100000 do \
           local x = i < 99000 and i * 1.0 or i + 0.5 n = n + (math.tointeger(x) or -1) end \
         return n",
    );
}

#[test]
fn string_byte_gives_several_results() {
    assert_inline(
        &ALL,
        "local s, n = 'hello world', 0 \
         for i = 1, 100000 do \
           local k = i % 5 + 1 \
           local a, b, c = string.byte(s, k, k + 2) \
           n = n + a + b + c + string.byte(s, 1, 3) \
         end return n",
    );
}

#[test]
fn string_byte_of_a_run_as_the_last_value() {
    assert_inline(
        &ALL,
        "local s = 'abcdefgh' local function f(k) return string.byte(s, k, k + 3) end \
         local n = 0 for i = 1, 100000 do local a, b, c, d = f(i % 4 + 1) n = n + a + b + c + d end \
         return n",
    );
}

#[test]
fn string_byte_edge_indices_match_the_interpreter() {
    assert_agree(
        &ALL,
        "local s, out = 'hello', {} \
         local cases = {{1, 1}, {2, 1}, {0, 2}, {-1, -1}, {-3, -1}, {1, 9}, {6, 7}, {10, 12}, \
                        {-9, 2}, {2, -2}, {3, 3}, {5, 5}, {1, 5}} \
         for round = 1, 2000 do for k = 1, #cases do \
           local c = cases[k] \
           out[k] = select('#', string.byte(s, c[1], c[2])) .. ':' \
             .. table.concat({string.byte(s, c[1], c[2])}, ',') end end \
         return table.concat(out, ' ')",
    );
}

#[test]
fn string_byte_run_that_leaves_the_string_leaves_the_trace() {
    assert_exits(
        &ALL,
        "local s, n = 'hello world', 0 \
         for i = 1, 100000 do \
           local k = i < 99000 and 2 or 10 + i % 1 \
           local a, b = string.byte(s, k, k + 1) n = n + a + (b or 0) end return n",
    );
}

#[test]
fn an_argument_of_another_type_leaves_to_the_interpreter() {
    assert_exits(
        &ALL,
        "local t, n = {1.5} , 0 for i = 1, 100000 do \
           if i == 99000 then t[1] = '2.5' end n = n + math.floor(t[1]) end return n",
    );
    assert_exits(
        &ALL,
        "return pcall(function() local t, n = {1.5}, 0 for i = 1, 100000 do \
           if i == 99000 then t[1] = {} end n = n + math.abs(t[1]) end return n end)",
    );
}

#[test]
fn a_replaced_function_leaves_to_the_interpreter() {
    for name in ["math.floor", "math.ceil", "math.abs"] {
        assert_exits(
            &ALL,
            &format!(
                "local n = 0 for i = 1, 100000 do \
                   if i == 99000 then {name} = function() return 1000 end end \
                   n = n + {name}(i / 3) end return n"
            ),
        );
    }
    assert_exits(
        &ALL,
        "local n = 0 local ty = type for i = 1, 100000 do \
           if i == 99000 then ty = function() return 'number' end end \
           if ty(i) == 'number' then n = n + 1 end end return n",
    );
}

#[test]
fn a_loop_that_aborted_on_math_floor_no_longer_does() {
    let (_, lua) = outcome(
        CompatibilityProfile::Lua54,
        ExecutionMode::Tracing,
        "local n = 0 for i = 1, 100000 do n = n + math.floor(i / 3) end return n",
    );
    let stats = lua.vm().jit_statistics().trace;
    assert_eq!(stats.native_aborts.of("math.floor"), 0, "{stats:?}");
    assert!(stats.native_entries > 0, "{stats:?}");
}

#[test]
fn the_callee_is_pinned_so_other_functions_still_trace() {
    // The same call sites with another function in the variable.
    assert_inline(
        &ALL,
        "local f = math.floor local n = 0 for i = 1, 100000 do n = n + f(i / 3) end return n",
    );
}

/// Each edge value alone in a hot loop, so the trace records with it and
/// runs natively when the recorder takes it.
#[test]
fn edge_values_one_at_a_time_match_the_interpreter() {
    let floats = [
        "-0.0",
        "0.0",
        "0.5",
        "-0.5",
        "(0/0)",
        "(1/0)",
        "(-1/0)",
        "2^63",
        "-2^63",
        "1e300",
        "-1e300",
        "2^53",
        "-(2^53)",
        "(1e15 + 0.5)",
        "4.0",
        "-4.0",
        "0.1",
        "-0.9",
        "(2^63 - 1024)",
        "(-2^63 + 1024)",
    ];
    let integers = [
        "math.mininteger",
        "math.maxinteger",
        "0",
        "-1",
        "1",
        "(math.mininteger + 1)",
    ];
    let calls = ["math.floor", "math.ceil", "math.abs"];
    for call in calls {
        for value in floats {
            let source = format!(
                "local x, r = {value} for i = 1, 20000 do r = {call}(x) end \
                 return r, 1 / r, x"
            );
            assert_agree(&ALL, &source);
        }
        for value in integers {
            let source =
                format!("local x, r = {value} for i = 1, 20000 do r = {call}(x) end return r, x");
            assert_agree(&INTEGERS, &source);
        }
    }
    let converted = [
        "3.0",
        "3.5",
        "'3'",
        "'3.0'",
        "'x'",
        "2^63",
        "-2^63",
        "-0.0",
        "(1/0)",
        "(0/0)",
        "2^53",
        "true",
        "nil",
        "math.maxinteger",
        "math.mininteger",
        "-(2^63)",
    ];
    for value in converted {
        let source = format!(
            "local x, r, t = {value} for i = 1, 20000 do r = math.tointeger(x) t = math.type(x) end \
             return r, t, x"
        );
        assert_agree(&INTEGERS, &source);
    }
}

#[test]
fn a_call_that_takes_open_results_needs_no_guard_of_its_own() {
    // `abs (floor (x))` passes the open results of `floor` to `abs`.
    assert_inline(
        &ALL,
        "local n = 0 for i = 1, 100000 do \
           n = n + math.abs(math.floor(i / 3 - 5000)) + math.floor(math.ceil(i / 7)) \
             + math.abs(math.abs(i - 50000)) end return n",
    );
    // `type (select (...))`: the open results of `select` feed `type`.
    assert_inline(
        &ALL,
        "local function f(...) if type(select(2, ...)) == 'number' then return 1 end return 0 end \
         local n = 0 for i = 1, 100000 do n = n + f(i, 2.5) end return n",
    );
}

#[test]
fn floor_and_ceil_of_floats_outside_the_integers_stay_floats() {
    for value in ["2^70", "-(2^70)", "1/0", "-1/0", "0/0", "2^63", "-(2^63)"] {
        for function in ["floor", "ceil"] {
            assert_inline(
                &ALL,
                &format!(
                    "local x, r = {value} \
                     for i = 1, 300000 do r = math.{function}(x) end \
                     return r, math.type and math.type(r)"
                ),
            );
        }
    }
}

#[test]
fn floor_of_a_float_moving_back_into_range_leaves_the_trace() {
    assert_exits(
        &INTEGERS,
        "local x, r, n = 2^70, 0, 0 \
         for i = 1, 300000 do \
           if i == 250000 then x = 7.5 end \
           r = math.floor(x) n = n + 1 end \
         return r, math.type(r), n",
    );
    assert_exits(
        &INTEGERS,
        "local x, r = 7.5, 0 \
         for i = 1, 300000 do \
           if i == 250000 then x = 0/0 end \
           r = math.ceil(x) end \
         return r ~= r, math.type(r)",
    );
}

#[test]
fn select_in_the_root_frame_of_a_vararg_function() {
    let source = "local function f(...) local s = 0 \
         for i = 1, 100000 do s = s + select('#', ...) + (select(2, ...)) end \
         return s end \
         return f(1, 2, 3)";
    assert_inline(&ALL, source);
    // Called twice with other counts: the second entry meets a different
    // vararg count, which the entry refuses or a guard leaves.
    assert_agree(
        &ALL,
        "local function f(...) local s = 0 \
         for i = 1, 100000 do s = s + select('#', ...) + (select(2, ...) or 0) end \
         return s end \
         return f(1, 2, 3), f(1), f(4, 5), f()",
    );
}

#[test]
fn select_of_forwarded_root_varargs() {
    assert_inline(
        &ALL,
        "local function g(...) return select('#', ...) end \
         local function f(...) local s = 0 \
         for i = 1, 100000 do s = s + g(...) end return s end \
         return f(1, 2, 3)",
    );
}

#[test]
fn a_root_vararg_count_guard_leaves_on_another_count() {
    for &profile in &ALL {
        let source = "local function f(...) local s = 0 \
             for i = 1, 100000 do s = s + select('#', ...) end return s end \
             local a = f(1, 2, 3) local b = f(1) local c = f(7, 8, 9) return a, b, c";
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert_eq!(stats.total_aborts(), 0, "{profile:?} {stats:?}");
    }
}

#[test]
fn a_named_vararg_table_is_read_live_not_from_the_frame() {
    // Lua 5.5 expands `...` from the table `t`, which the loop rewrites.
    let lua55 = [CompatibilityProfile::Lua55];
    assert_agree(
        &lua55,
        "local function f(...t) local s = 0 \
         for i = 1, 100000 do if i == 50000 then t[2] = 100 end \
         local a, b = ... s = s + b end return s end \
         return f(1, 2, 3)",
    );
    assert_agree(
        &lua55,
        "local function f(...t) local s = 0 \
         for i = 1, 100000 do if i == 50000 then t[2] = 100 end \
         s = s + (select(2, ...)) end return s end \
         return f(1, 2, 3)",
    );
    assert_agree(
        &lua55,
        "local function f(...t) local s = 0 \
         for i = 1, 100000 do if i == 50000 then t.n = 1 end \
         s = s + select('#', ...) end return s end \
         return f(1, 2, 3)",
    );
}

#[test]
fn a_wrapper_around_type_traces() {
    assert_inline(
        &ALL,
        "local function kind(v) return type(v) end \
         local n, t = 0, {} \
         for i = 1, 100000 do \
           if kind(i) == 'number' then n = n + 1 end \
           if kind(t) ~= 'string' then n = n + 2 end \
           n = n + #kind(i > 5) \
         end return n",
    );
}

#[test]
fn type_as_an_open_argument_traces() {
    assert_inline(
        &ALL,
        "local function size(s) return #s end \
         local function is_number(s) return s == 'number' end \
         local n = 0 \
         for i = 1, 100000 do \
           n = n + size(type(i)) \
           if is_number(type(i)) then n = n + 1000 end \
           if is_number(type(nil)) then n = n + 100000 end \
         end return n",
    );
}

#[test]
fn type_as_an_argument_of_a_native_traces() {
    assert_native_helpers(
        &ALL,
        "local n = 0 \
         for i = 1, 100000 do \
           local k = type(i) \
           n = n + #tostring(k) + select('#', type(i), type(n)) end \
         return n",
        &["tostring"],
    );
}

#[test]
fn math_type_returned_from_an_inlined_function_traces() {
    assert_inline(
        &INTEGERS,
        "local function kind(v) return math.type(v) end \
         local n = 0 \
         for i = 1, 100000 do \
           if kind(i) == 'integer' then n = n + 1 end \
           if kind(i / 2) == 'float' then n = n + 2 end \
           if kind('x') == nil then n = n + 4 end \
           local k = kind(i) n = n + #k \
         end return n",
    );
}

#[test]
fn type_carried_across_iterations_traces() {
    assert_inline(
        &ALL,
        "local last, n = 'none', 0 \
         for i = 1, 100000 do last = type(i) n = n + #last end \
         return n, last",
    );
    assert_inline(
        &ALL,
        "local last, n, t = 'none', 0, {} \
         for i = 1, 100000 do \
           if i % 3 == 0 then last = type(t) else last = type(i) end \
           n = n + #last \
         end return n, last",
    );
}

#[test]
fn a_carried_type_that_is_read_before_it_is_written_stays_correct() {
    // The previous iteration's string is read: a path the trace does not
    // hold, but the results are the interpreter's.
    assert_agree(
        &ALL,
        "local last, n = 'none', 0 \
         for i = 1, 100000 do n = n + #last last = type(i) end \
         return n, last",
    );
}

#[test]
fn a_carried_type_leaves_through_an_exit_with_its_string() {
    // The exit lands where the string lives in a register: the guard fails
    // late in the run and the interpreter reads `last`.
    assert_exits(
        &ALL,
        "local last, n = 'none', 0 \
         for i = 1, 100000 do \
           if i < 99000 then last = type(i) else last = type(n) end \
           n = n + #last \
         end return n, last",
    );
}

#[test]
fn an_exit_before_the_write_restores_the_carried_type_string() {
    // The last iteration leaves before `last` is written again, so the
    // register must come back as the string the previous iteration made.
    assert_exits(
        &ALL,
        "local last, n = 'none', 0 \
         for i = 1, 100000 do \
           if i == 100000 then break end \
           last = type(i) n = n + 1 \
         end return n, last, #last",
    );
}

#[test]
fn a_type_string_stored_in_a_table_is_the_interpreters() {
    assert_agree(
        &ALL,
        "local t, n = {}, 0 \
         for i = 1, 100000 do t.kind = type(i) t[type(t) .. 'x'] = i n = n + #t.kind end \
         return n, t.kind, t.tablex",
    );
}

#[test]
fn select_past_the_end_of_the_arguments_gives_nothing() {
    // `k` is `#args + 2`: stock Lua clips the index and gives no values.
    assert_inline(
        &ALL,
        "local function f(...) \
           local a = select(4, ...) \
           return select('#', select(4, ...)) + (a == nil and 1 or 2) end \
         local s = 0 for i = 1, 100000 do s = s + f(i, 2) end return s",
    );
    assert_inline(
        &ALL,
        "local s = 0 for i = 1, 100000 do s = s + select('#', select(9, i, 2, 3)) end return s",
    );
}

#[test]
fn select_with_a_negative_index_before_the_first_still_raises() {
    assert_exits(
        &ALL,
        "local function f(k, ...) return (select(k, ...)) end \
         return pcall(function() local s = 0 for i = 1, 100000 do \
           s = s + f(i < 99000 and -1 or -4, i, 5, 6) end return s end)",
    );
}

#[test]
fn rawequal_of_a_type_and_a_string_constant_folds() {
    assert_inline(
        &ALL,
        "local n = 0 for i = 1, 100000 do \
           if rawequal(type(i), 'number') then n = n + 1 end \
           if rawequal(type(i), 'string') then n = n + 1000 end \
           if rawequal('a', 'a') then n = n + 2 end \
         end return n",
    );
}

#[test]
fn rawequal_of_two_strings_compares_their_bytes() {
    assert_native_helpers(
        &ALL,
        "local a, b, c = 'abc', 'ab' .. 'c', 'abd' \
         local n = 0 \
         for i = 1, 100000 do \
           if rawequal(a, b) then n = n + 1 end \
           if rawequal(a, c) then n = n + 1000 end \
           if rawequal(a, a) then n = n + 2 end \
           if rawequal(b, 'abc') then n = n + 4 end \
           if rawequal(type(i), c) then n = n + 100000 end \
         end return n",
        &["string.equal"],
    );
}

#[test]
fn rawequal_of_strings_made_in_the_loop() {
    assert_native_helpers(
        &ALL,
        "local n, key = 0, 'k5' \
         for i = 1, 100000 do \
           local s = 'k' .. (i % 10) \
           if rawequal(s, key) then n = n + 1 end \
           if rawequal(s, 'k7') then n = n + 1000 end \
         end return n",
        &["concat", "string.equal"],
    );
}

#[test]
fn rawequal_of_strings_ignores_eq() {
    // A string metatable with `__eq` never runs for two strings.
    assert_native_helpers(
        &ALL,
        "getmetatable('').__eq = function() return true end \
         local a, b = 'x' .. 'y', 'xz' \
         local n = 0 for i = 1, 100000 do if rawequal(a, b) then n = n + 1 end end return n",
        &["string.equal"],
    );
}

/// Like [`assert_inline`], except that the trace calls exactly the helpers
/// named in `helpers`.
fn assert_native_helpers(profiles: &[CompatibilityProfile], source: &str, helpers: &[&str]) {
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
    }
}

#[test]
fn a_root_string_carried_through_an_inlined_call_is_not_reentered_each_run() {
    // `id('hello')` is a root string constant behind an inlined call. Carried
    // into the next iteration, a table read on it would miss on every run.
    let source = "local function id(s) return s end \
         local last, n = 'x', 0 \
         local t = {x = 1, hello = 5} \
         for i = 1, 50000 do n = n + t[last]; last = id('hello') end \
         return n, last";
    for &profile in &ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries < 100, "{profile:?} {stats:?}");
    }
}

// ---- string.byte of a string.sub result --------------------------------------
//
// `string.byte (string.sub (s, i, j), a, b)` with constant positions reads the
// bytes of `s` itself, and `string.sub` makes no string when only a recovery
// path would hold it. `#` of the substring and `==` against a string constant
// read the source the same way. Every profile records this: those without
// integers truncate a float index, the others take integer indices as they are.

#[test]
fn string_byte_of_a_substring_makes_no_call() {
    let source = "local byte, sub = string.byte, string.sub \
         local s, n = string.rep('0123456789abcdef', 16), 0 \
         for i = 1, 100000 do \
           local k = (i - 1) % 200 + 1 \
           local slice = sub(s, k, k + 3) \
           local a, b, c, d = byte(slice, 1, 4) \
           n = n + a + b + c + d + byte(s, k) \
         end return n";
    assert_inline_optimized(&ALL, source);
}

#[test]
fn string_byte_of_a_substring_with_other_positions_and_forms() {
    for source in [
        // One position twice, and a position range that starts past the
        // first byte.
        "local s, n = string.rep('abcdefgh', 40), 0 \
         for i = 1, 100000 do \
           local k = i % 300 + 1 \
           local slice = string.sub(s, k, k + 5) \
           n = n + string.byte(slice, 1) + string.byte(slice, 6) \
         end return n",
        "local s, n = string.rep('abcdefgh', 40), 0 \
         for i = 1, 100000 do \
           local k = i % 300 + 1 \
           local a, b, c = string.byte(string.sub(s, k, k + 5), 3, 5) \
           n = n + a + b + c \
         end return n",
        "local s, n = string.rep('abcdefgh', 40), 0 \
         for i = 1, 100000 do \
           local k = i % 300 + 1 \
           local slice = string.sub(s, k, k + 5) \
           local a, b = string.byte(slice, 2, 3) \
           local c = string.byte(slice) \
           n = n + a + b + c \
         end return n",
    ] {
        assert_inline_optimized(&ALL, source);
    }
}

#[test]
fn the_method_form_of_sub_keeps_only_the_method_check() {
    // `s:sub (...)` checks which function the string metatable gives (one
    // helper call), and then the substring goes as for `string.sub`.
    let source = "local s, n = string.rep('abcdefgh', 40), 0 \
         for i = 1, 100000 do \
           local k = i % 300 + 1 \
           local slice = s:sub(k, k + 5) \
           local a, b = string.byte(slice, 1, 2) \
           n = n + a + b + string.byte(slice, 6) \
         end return n";
    for profile in ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert_eq!(stats.total_aborts(), 0, "{profile:?} {stats:?}");
        assert_eq!(
            lua.vm().trace_compiled_external_calls(),
            1,
            "{profile:?}: {:?}",
            lua.vm().trace_helper_calls()
        );
    }
}

#[test]
fn a_substring_with_an_edge_index_is_the_interpreters() {
    // Negative, zero, reversed, past the end, fractional and huge indices,
    // empty strings, and a byte count the substring does not have. Every case
    // runs the same hot loop, so the one the recording saw goes native and the
    // others leave or fall back.
    assert_agree(
        &ALL,
        "local out = {} \
         local strings = {'hello world', '', 'x', '\\0b\\0c\\0', '\\200\\255x\\128'} \
         local cases = {{1, 4}, {2, 5}, {0, 3}, {-3, -1}, {-100, 3}, {3, 2}, {5, 100}, {1, 0}, \
                        {1.5, 4}, {2, 4.5}, {-0.5, 3}, {1e30, 2}, {1, 1e30}, {-1e30, 3}, \
                        {2 ^ 31, 3}, {1, 2 ^ 31 - 1}, {1, 2 ^ 31}, {0 / 0, 3}, {2, 0 / 0}, \
                        {1 / 0, 1 / 0}, {-1 / 0, 4}, {4, 4}, {11, 11}, {12, 12}} \
         for si = 1, #strings do for ci = 1, #cases do \
           local s, c, acc = strings[si], cases[ci], {} \
           for round = 1, 400 do \
             local slice = string.sub(s, c[1], c[2]) \
             local a, b, d = string.byte(slice, 1, 3) \
             acc[#acc + 1] = (a or -1) + (b or -2) + (d or -3) + #slice \
           end \
           out[#out + 1] = table.concat(acc, ',', 1, 3) .. '/' .. #acc \
         end end \
         return table.concat(out, ' ')",
    );
}

#[test]
fn a_fractional_end_index_leaves_the_substring_length() {
    // The recording sees whole indices, so `#slice` is `j - i + 1`. A later
    // end of 4.5 is 4 to a profile without integers, which truncates it, and
    // an error to one with them. Either way the trace has to leave rather
    // than count 3.5 bytes. The index comes from a table, so no branch of the
    // loop tells the two rounds apart.
    let source = "local s, n, ends = 'hello world', 0, {} \
                  for i = 1, 100000 do ends[i] = i <= 50000 and 4 or 4.5 end \
                  for i = 1, 100000 do \
                    local slice = string.sub(s, 2, ends[i]) \
                    n = n + #slice \
                  end return n";
    assert_exits(&ALL, source);
    assert_agree(&ALL, source);
}

#[test]
fn a_fractional_end_index_leaves_a_substring_comparison() {
    // The recording compares a two-byte substring with a three-byte text and
    // guards that the lengths differ. An end of 4.5 truncates to 4 in a
    // profile without integers, so the substring is "ell" and equal; the
    // trace must not take `4.5 - 2 != 2` as the answer.
    let source = "local s, n, ends = 'hello world', 0, {} \
                  for i = 1, 100000 do ends[i] = i <= 50000 and 3 or 4.5 end \
                  for i = 1, 100000 do \
                    local slice = string.sub(s, 2, ends[i]) \
                    if slice == 'ell' then n = n + 1 end \
                  end return n";
    assert_exits(&ALL, source);
    assert_agree(&ALL, source);
}

#[test]
fn a_substring_shorter_than_the_bytes_asked_for_leaves_the_trace() {
    // Four bytes asked for; the substring shrinks near the end of the string.
    assert_exits(
        &ALL,
        "local s, n = string.rep('0123456789', 10), 0 \
         for i = 1, 100000 do \
           local k = i < 99000 and 5 or 99 \
           local slice = string.sub(s, k, k + 3) \
           local a, b, c, d = string.byte(slice, 1, 4) \
           n = n + a + b + c + (d or 1000) \
         end return n",
    );
    assert_agree(
        &ALL,
        "local s, n = string.rep('0123456789', 10), 0 \
         for i = 1, 100000 do \
           local k = i < 99000 and 5 or 99 \
           local slice = string.sub(s, k, k + 3) \
           local a, b, c, d = string.byte(slice, 1, 4) \
           n = n + a + b + c + (d or 1000) \
         end return n",
    );
}

#[test]
fn a_substring_that_starts_before_the_string_leaves_the_trace() {
    // Index 0 clips to the first byte: the substring is not the source from
    // `i` on, so the start has to be guarded.
    assert_exits(
        &ALL,
        "local s, n = 'abcdefghij', 0 \
         for i = 1, 100000 do \
           local k = i < 99000 and 3 or 0 \
           local slice = string.sub(s, k, k + 3) \
           local a, b = string.byte(slice, 1, 2) \
           n = n + a + b \
         end return n",
    );
    assert_agree(
        &ALL,
        "local s, n = 'abcdefghij', 0 \
         for i = 1, 100000 do \
           local k = i < 99000 and 3 or 0 \
           local slice = string.sub(s, k, k + 3) \
           local a, b = string.byte(slice, 1, 2) \
           n = n + a + b \
         end return n",
    );
}

#[test]
fn a_substring_that_escapes_after_the_loop_warmed_up_is_made() {
    // The recording elides the string. After 90000 rounds the loop stores it,
    // returns it, compares it and joins it: each is a use the trace did not
    // see, and the string must be the right one.
    for body in [
        "if i > 90000 then kept[#kept + 1] = slice end",
        "if i > 90000 and slice == 'bcde' then kept[#kept + 1] = 'eq' end",
        "if i > 90000 then kept[#kept + 1] = slice .. '|' end",
        "if i > 90000 then kept[#kept + 1] = #slice end",
        "if i > 90000 then kept[#kept + 1] = string.upper(slice) end",
    ] {
        let source = format!(
            "local s, n, kept = 'abcdefghij', 0, {{}} \
             for i = 1, 100000 do \
               local k = i % 6 + 1 \
               local slice = string.sub(s, k, k + 3) \
               local a, b = string.byte(slice, 1, 2) \
               n = n + a + b \
               {body} \
             end return n, #kept, table.concat(kept, ',', 1, 12)"
        );
        assert_agree(&ALL, &source);
    }
}

#[test]
fn the_exit_makes_the_substring_the_optimizer_dropped() {
    // The trace holds no string.sub, and the guard that leaves after 90000
    // rounds restores `slice` in a register: the exit makes it.
    let source = "local s, n, kept = 'abcdefghij', 0, {} \
         for i = 1, 100000 do \
           local k = i % 6 + 1 \
           local slice = string.sub(s, k, k + 3) \
           local a, b = string.byte(slice, 1, 2) \
           n = n + a + b \
           if i > 90000 then kept[#kept + 1] = slice end \
         end return n, #kept, kept[1], kept[2], kept[6], kept[10000]";
    for profile in ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert!(
            stats.exits_of(TraceExitKind::GuardFail) > 0,
            "{profile:?} {stats:?}"
        );
    }
    // The first trace, the one that dropped the call, compiled.
    let (_, lua) = outcome(
        CompatibilityProfile::Lua51,
        ExecutionMode::Tracing,
        "local s, n = 'abcdefghij', 0 \
         for i = 1, 100000 do \
           local k = i % 6 + 1 \
           local slice = string.sub(s, k, k + 3) \
           local a, b = string.byte(slice, 1, 2) \
           n = n + a + b \
           if i > 99990 then n = n + #slice end \
         end return n",
    );
    assert_eq!(lua.vm().trace_compiled_external_calls(), 0);
}

#[test]
fn a_substring_that_is_returned_or_kept_by_a_later_iteration() {
    assert_agree(
        &ALL,
        "local s = 'abcdefghij' \
         local function f(k) \
           local slice = string.sub(s, k, k + 3) \
           local a, b = string.byte(slice, 1, 2) \
           if k == 9 then return slice, a + b end \
           return a, b \
         end \
         local n, last = 0, nil \
         for i = 1, 100000 do \
           local x, y = f(i % 9 + 1) \
           if type(x) == 'string' then last = x end \
           n = n + (type(x) == 'number' and x or 0) + y \
         end return n, last",
    );
    assert_agree(
        &ALL,
        "local s = 'abcdefghij' local prev, n = '', 0 \
         for i = 1, 100000 do \
           local k = i % 6 + 1 \
           local slice = string.sub(s, k, k + 3) \
           local a, b = string.byte(slice, 1, 2) \
           n = n + a + b + #prev \
           prev = slice \
         end return n, prev",
    );
}

#[test]
fn string_sub_or_string_byte_replaced_in_the_loop_leaves_the_trace() {
    let sub_replaced = "local byte, sub = string.byte, string.sub \
         local s, n = 'abcdefghij', 0 \
         for i = 1, 100000 do \
           local k = i % 6 + 1 \
           local slice = sub(s, k, k + 3) \
           local a, b = byte(slice, 1, 2) \
           n = n + a + b \
           if i == 99000 then sub = function(str, x, y) return 'ZZZZ' end end \
         end return n";
    assert_exits(&ALL, sub_replaced);
    assert_agree(&ALL, sub_replaced);
    let byte_replaced = "local byte, sub = string.byte, string.sub \
         local s, n = 'abcdefghij', 0 \
         for i = 1, 100000 do \
           local k = i % 6 + 1 \
           local slice = sub(s, k, k + 3) \
           local a, b = byte(slice, 1, 2) \
           n = n + a + b \
           if i == 99000 then byte = function(str, x, y) return 1, 2 end end \
         end return n";
    assert_exits(&ALL, byte_replaced);
    assert_agree(&ALL, byte_replaced);
    // The global functions replaced.
    let global_replaced = "local s, n = 'abcdefghij', 0 \
         for i = 1, 100000 do \
           local k = i % 6 + 1 \
           local slice = string.sub(s, k, k + 3) \
           local a, b = string.byte(slice, 1, 2) \
           n = n + a + b \
           if i == 99000 then string.sub = function() return 'qq' end end \
         end return n";
    assert_agree(&ALL, global_replaced);
}

#[test]
fn the_string_metatable_replaced_in_the_loop_changes_the_method() {
    assert_agree(
        &ALL,
        "local s, n = 'abcdefghij', 0 \
         for i = 1, 100000 do \
           local k = i % 6 + 1 \
           local slice = s:sub(k, k + 3) \
           local a, b = string.byte(slice, 1, 2) \
           n = n + a + b \
           if i == 99000 then getmetatable('').__index = {sub = function() return 'zz' end} end \
         end return n",
    );
}

#[test]
fn indices_that_change_type_mid_loop_are_the_interpreters() {
    assert_agree(
        &ALL,
        "local s, n = 'abcdefghij', 0 \
         for i = 1, 100000 do \
           local k = i % 6 + 1 \
           if i == 99000 then k = '2' end \
           local slice = string.sub(s, k, k + 3) \
           local a, b = string.byte(slice, 1, 2) \
           n = n + a + b \
         end return n",
    );
    assert_agree(
        &ALL,
        "return pcall(function() local s, n = 'abcdefghij', 0 \
         for i = 1, 100000 do \
           local k = i % 6 + 1 \
           if i == 99000 then k = {} end \
           local slice = string.sub(s, k, 4) \
           local a, b = string.byte(slice, 1, 2) \
           n = n + a + b \
         end return n end)",
    );
}

// ---- # and == of a string.sub result ----------------------------------------

#[test]
fn the_length_and_equality_of_a_substring_make_no_call() {
    // The substring has `#` and `==` readers only. The comparison goes the
    // same way every round, so the loop is one trace with no side trace to
    // hold the string.
    for text in ["abcd", "abce"] {
        let source = format!(
            "local sub = string.sub \
             local s, n = string.rep('abcdefgh', 40), 0 \
             for i = 1, 100000 do \
               local k = (i - 1) % 8 * 8 + 1 \
               local slice = sub(s, k, k + 3) \
               if slice == '{text}' then n = n + 1 end \
               n = n + #slice \
             end return n"
        );
        assert_inline_optimized(&ALL, &source);
    }
    assert_inline_optimized(
        &ALL,
        "local sub = string.sub \
         local s, n = string.rep('abcdefgh', 40), 0 \
         for i = 1, 100000 do \
           local k = i % 300 + 1 \
           n = n + #sub(s, k, k + 3) \
         end return n",
    );
    // A text of another length is told by the length alone, and the text on
    // the left reads the same.
    assert_inline_optimized(
        &ALL,
        "local sub = string.sub \
         local s, n = string.rep('abcdefgh', 40), 0 \
         for i = 1, 100000 do \
           local k = i % 300 + 1 \
           if 'abcdef' == sub(s, k, k + 3) then n = n + 1 end \
           local same = sub(s, k, k + 1) == 'zz' \
           if same then n = n + 2 end \
         end return n",
    );
}

#[test]
fn the_length_and_equality_of_a_substring_follow_the_interpreter() {
    // Reversed, negative, clipped, empty and out of range ranges over a few
    // strings, each in its own hot loop, so the recording took one range and
    // the others fall back or leave.
    assert_agree(
        &ALL,
        "local out = {} \
         local strings = {'hello world', '', 'x', 'abcdabcd', '\\0b\\0c\\0'} \
         local cases = {{1, 4}, {2, 5}, {5, 8}, {0, 3}, {-3, -1}, {-100, 3}, {3, 2}, {5, 100}, \
                        {1, 0}, {4, 4}, {11, 11}, {12, 12}, {2, 1}, {3, 3}} \
         for si = 1, #strings do for ci = 1, #cases do \
           local s, c, acc = strings[si], cases[ci], 0 \
           for round = 1, 400 do \
             local slice = string.sub(s, c[1], c[2]) \
             if slice == 'abcd' then acc = acc + 1000 end \
             if slice == 'hell' then acc = acc + 100 end \
             if slice == '' then acc = acc + 10 end \
             acc = acc + #slice \
           end \
           out[#out + 1] = acc \
         end end \
         return table.concat(out, ' ')",
    );
}

#[test]
fn a_substring_read_by_length_and_equality_escapes_after_warmup() {
    // After 90000 rounds the range becomes empty, then runs past the end,
    // and the substring is kept: every guard leaves and the exit makes it.
    let source = "local sub, kept = string.sub, {} \
         local s, n = string.rep('abcdefgh', 40), 0 \
         for i = 1, 100000 do \
           local k = (i - 1) % 8 * 8 + 1 \
           local last = k + 3 \
           if i > 99000 then last = k + 2 end \
           if i > 99500 then k = 400 end \
           local slice = sub(s, k, last) \
           if slice == 'abcd' then n = n + 1 end \
           n = n + #slice \
           if i > 99900 then kept[#kept + 1] = slice end \
         end return n, #kept, kept[1], kept[100]";
    for profile in ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?}: {expected:?}");
        assert_eq!(show(&actual), show(&expected), "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert!(
            stats.exits_of(TraceExitKind::GuardFail) > 0,
            "{profile:?} {stats:?}"
        );
    }
}
