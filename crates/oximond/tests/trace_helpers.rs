//! Library calls a trace makes through the helper table
//! ([`oximond::jit_ir::trace::TraceOp::CallHelper`]): the loop compiles
//! without an abort, the trace calls the helper, and the result is the
//! interpreter's, in every profile.

use oximond::runtime::Value;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn run(profile: CompatibilityProfile, mode: ExecutionMode, source: &str) -> (Vec<Value>, Lua) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let results = lua.exec(source).unwrap();
    (results, lua)
}

/// Runs `source` traced and interpreted in every profile, checks that the
/// results agree, that no recording aborted and that the trace calls
/// `helper`.
fn assert_traced_through(source: &str, helper: &str) {
    assert_traced_through_by_profile(source, |_| helper);
}

/// Like [`assert_traced_through`] for a call whose helper depends on the
/// profile: the trace must call exactly the one `helper` names.
fn assert_traced_through_by_profile<'a>(
    source: &str,
    helper: impl Fn(CompatibilityProfile) -> &'a str,
) {
    for profile in PROFILES {
        let helper = helper(profile);
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.traces_compiled > 0, "{profile:?} {stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert_eq!(stats.total_aborts(), 0, "{profile:?} {stats:?}");
        assert!(
            lua.vm().trace_helper_calls().contains(&helper),
            "{profile:?}: wanted {helper}, got {:?}",
            lua.vm().trace_helper_calls()
        );
    }
}

#[test]
fn string_length_of_a_root_string_calls_the_helper() {
    assert_traced_through(
        "local s = 0 for i = 1, 100000 do s = s + string.len('hello, world') + i % 3 end return s",
        "string.len",
    );
}

#[test]
fn string_length_of_a_borrowed_string_calls_the_helper() {
    assert_traced_through(
        "local t = {'a', 'bb', 'ccc', 'dddd'} local s = 0 \
         for i = 1, 100000 do s = s + string.len(t[i % 4 + 1]) end return s",
        "string.len_ref",
    );
}

#[test]
fn length_operator_of_a_borrowed_string_calls_the_helper() {
    assert_traced_through(
        "local t = {'a', 'bb', 'ccc', 'dddd'} local s = 0 \
         for i = 1, 100000 do s = s + #t[i % 4 + 1] end return s",
        "string.len_ref",
    );
}

#[test]
fn a_string_the_trace_holds_by_lane_needs_no_helper() {
    for profile in PROFILES {
        let source = "local s = 0 for i = 1, 100000 do s = s + #tostring(i) end return s";
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        // `tostring` is a helper, but the length of its result is an inline
        // read of the run's length table.
        let calls = lua.vm().trace_helper_calls();
        assert!(calls.contains(&"tostring"), "{profile:?}: {calls:?}");
        assert!(
            !calls.iter().any(|name| name.starts_with("string.len")),
            "{profile:?}: {calls:?}"
        );
    }
}

/// Runs `source` and gives its results, or the error it raised as text.
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

#[test]
fn string_char_calls_the_helper() {
    assert_traced_through(
        "local n, last = 0, '' for i = 1, 100000 do \
           last = string.char(65 + i % 26, 66, 48 + i % 10) n = n + #last end \
         return n, last",
        "string.char",
    );
}

#[test]
fn tostring_of_numbers_and_strings_calls_the_helper() {
    assert_traced_through(
        "local n, last = 0, '' for i = 1, 100000 do \
           last = tostring(i * 3) n = n + #last + #tostring(i / 4) + #tostring('abc') end \
         return n, last",
        "tostring",
    );
}

#[test]
fn tostring_of_small_numbers_calls_the_helper() {
    assert_traced_through(
        "local n = 0 for i = 1, 100000 do n = n + #tostring(i % 200) end return n",
        "tostring",
    );
}

#[test]
fn string_format_calls_a_helper() {
    // Lua 5.1 registers the formatter as a plain native, LuaJIT has its own,
    // and the later profiles route through the VM builtin.
    assert_traced_through_by_profile(
        "local n, last = 0, '' for i = 1, 100000 do \
           last = string.format('%d:%s:%5.2f', i, 'x', i / 3) n = n + #last end \
         return n, last",
        |profile| match profile {
            CompatibilityProfile::Lua51 => "string.format_native",
            CompatibilityProfile::LuaJit21 => "string.format_luajit",
            _ => "string.format",
        },
    );
}

#[test]
fn string_find_calls_the_helper() {
    // A plain find that finds the text, and one that does not.
    assert_traced_through(
        "local s, n = 'hello wonderful world', 0 for i = 1, 100000 do \
           n = n + string.find(s, 'wor', 1, true) + (string.find(s, 'xyz', 1, true) or i % 3) end \
         return n",
        "string.find",
    );
}

#[test]
fn string_find_that_misses_calls_the_helper() {
    assert_traced_through(
        "local s, n = 'hello wonderful world', 0 for i = 1, 100000 do \
           if string.find(s, 'xyz', 1, true) == nil then n = n + i % 7 end end \
         return n",
        "string.find_absent",
    );
}

#[test]
fn table_concat_calls_the_helper() {
    assert_traced_through(
        "local t, n, last = {'a', 'bc', 'def'}, 0, '' for i = 1, 100000 do \
           last = table.concat(t, ',') n = n + #last + #table.concat(t) end \
         return n, last",
        "table.concat",
    );
}

#[test]
fn concatenation_calls_the_helper() {
    assert_traced_through(
        "local n, last = 0, '' for i = 1, 100000 do \
           last = 'a' .. i .. 'b' .. i * 0.5 n = n + #last + #(last .. 'x') end \
         return n, last",
        "concat",
    );
}

#[test]
fn a_helper_that_cannot_answer_leaves_to_the_interpreter() {
    // The trace records with bytes in range. Past 255 the helper misses and
    // the interpreter raises the error the call raises.
    for profile in PROFILES {
        let source = "local ok, err = pcall(function() \
               local n = 0 for i = 0, 100000 do n = n + #string.char(i % 300) end return n end) \
             return ok, err";
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        assert!(matches!(&actual, Ok(values) if values[0] == Value::Boolean(false)));
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert!(
            stats.exits_of(oximond::vm::TraceExitKind::GuardFail) > 0,
            "{profile:?} {stats:?}"
        );
    }
}

#[test]
fn an_argument_of_another_type_leaves_to_the_interpreter() {
    // Each loop records with one argument type and meets another near its
    // end: a string the call coerces, or a value it rejects.
    let sources = [
        "local n = 0 for i = 1, 100000 do \
           n = n + #string.char(i < 99000 and 65 or '66') end return n",
        "return pcall(function() local n = 0 for i = 1, 100000 do \
           n = n + #string.char(i < 99000 and 65 or {}) end return n end)",
        "local n = 0 for i = 1, 100000 do \
           n = n + #tostring(i < 99000 and i or true) end return n",
        "local n, p = 0, 'wor' for i = 1, 100000 do \
           if i == 99000 then p = 5 end \
           n = n + (string.find('hello wonderful world', p, 1, true) or 0) end return n",
        "return pcall(function() local n, p = 0, 'wor' for i = 1, 100000 do \
           if i == 99000 then p = {} end \
           n = n + (string.find('hello wonderful world', p, 1, true) or 0) \
         end return n end)",
        "return pcall(function() local n, t, v = 0, {'a', 'b'}, {} for i = 1, 100000 do \
           if i == 99000 then t[2] = v end \
           n = n + #table.concat(t, ',') end return n end)",
    ];
    for profile in PROFILES {
        for source in sources {
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
            let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
            assert_eq!(actual, expected, "{profile:?} {source}");
            let stats = lua.vm().jit_statistics().trace;
            assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
            assert!(
                stats.exits_of(oximond::vm::TraceExitKind::GuardFail) > 0,
                "{profile:?} {source} {stats:?}"
            );
        }
    }
}

#[test]
fn a_tostring_metamethod_on_strings_leaves_to_the_interpreter() {
    // The loop is hot and traced before the metamethod appears; the run that
    // follows meets it at the call, and the interpreter calls it.
    let steps = [
        "function count(s) local n = 0 \
           for i = 1, 3000 do n = n + #tostring(s) end return n end",
        "return count('abc')",
        "return count('abc')",
        "getmetatable('').__tostring = function() return 'TOSTRING' end \
         return count('abc')",
        "getmetatable('').__tostring = nil return count('abc')",
    ];
    for profile in PROFILES {
        let mut interpreted = Lua::builder()
            .profile(profile)
            .execution_mode(ExecutionMode::Interpreter)
            .build()
            .unwrap();
        let mut traced = Lua::builder()
            .profile(profile)
            .execution_mode(ExecutionMode::Tracing)
            .build()
            .unwrap();
        traced.vm_mut().use_synchronous_jit_compilation();
        for (position, step) in steps.iter().enumerate() {
            let before = traced.vm().jit_statistics().trace;
            let expected = interpreted.exec(step).map_err(|error| format!("{error:?}"));
            let actual = traced.exec(step).map_err(|error| format!("{error:?}"));
            assert_eq!(actual, expected, "{profile:?} step {position}");
            if position == 3 {
                // The helper missed in native code: a guard left the trace.
                let after = traced.vm().jit_statistics().trace;
                assert!(
                    after.exits_of(oximond::vm::TraceExitKind::GuardFail)
                        > before.exits_of(oximond::vm::TraceExitKind::GuardFail),
                    "{profile:?} {after:?}"
                );
            }
            if position == 2 {
                assert!(
                    traced.vm().trace_helper_calls().contains(&"tostring"),
                    "{profile:?}: {:?}",
                    traced.vm().trace_helper_calls()
                );
            }
        }
        let stats = traced.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
    }
}

#[test]
fn a_replaced_library_function_leaves_to_the_interpreter() {
    for profile in PROFILES {
        let source = "local function run() local n = 0 \
               for i = 1, 3000 do n = n + #string.char(65 + i % 26) end return n end \
             local before = run() \
             string.char = function() return 'zzz' end \
             return before, run()";
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert!(
            lua.vm().trace_helper_calls().contains(&"string.char"),
            "{profile:?}: {:?}",
            lua.vm().trace_helper_calls()
        );
    }
}

#[test]
fn a_loop_past_the_string_budget_leaves_the_trace_and_stays_correct() {
    // Each iteration builds 30 strings of 40 KB, which the run keeps until
    // the loop-back; a run has room for about one megabyte of new strings, so
    // the helper misses part way through an iteration and the interpreter
    // makes the call.
    let terms: Vec<String> = (0..30)
        .map(|k| format!("string.byte(big .. (i + {k}), -1)"))
        .collect();
    let source = format!(
        "local big = string.rep('a', 40000) local n = 0 \
         for i = 1, 3000 do n = n + {} end return n",
        terms.join(" + ")
    );
    for profile in PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, &source);
        let (actual, lua) = run(profile, ExecutionMode::Tracing, &source);
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert!(
            stats.exits_of(oximond::vm::TraceExitKind::GuardFail) > 0,
            "{profile:?} {stats:?}"
        );
    }
}

#[test]
fn a_call_on_loop_invariant_arguments_still_runs() {
    // The calls take nothing the loop changes: the optimizer may run them
    // once, ahead of the loop.
    assert_traced_through(
        "local c, s, n = 12345, 'abc', 0 for i = 1, 100000 do \
           local t = tostring(c) .. s n = n + #t + i % 2 end return n",
        "tostring",
    );
}

/// Runs `source` traced and interpreted in every profile and checks that the
/// results agree and that the trace ran.
fn assert_traced_like_interpreter(source: &str) {
    for profile in PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.traces_compiled > 0, "{profile:?} {stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
    }
}

#[test]
fn format_and_tostring_reuse_lanes_across_releases_and_a_late_exit() {
    // Every iteration publishes strings into lanes an earlier release
    // emptied. At iteration 70000 the first `%d` operand is a numeric string
    // and at 90000 a float with a fraction, which the formatter takes
    // differently (and, from Lua 5.3 on, refuses), so the trace leaves
    // part way through an iteration.
    assert_traced_like_interpreter(
        "local n, last = 0, '' \
         for i = 1, 100000 do \
           local v = i \
           if i == 70000 then v = '12' end \
           local ok, text = pcall(string.format, '%06d %x', v, i % 4096) \
           if i == 90000 then ok, text = pcall(string.format, '%06d', 1.5) end \
           last = string.format('%.3f|%s', i / 7, tostring(i % 101) .. tostring(ok)) \
           n = n + #last + #tostring(text) \
         end \
         return n, last",
    );
}

#[test]
fn table_concat_of_freshly_made_pieces_matches_the_interpreter() {
    // The pieces are tostring results, which the trace makes in lanes of its
    // own; a number among them at one round and a late long piece change what
    // the concat sees.
    assert_traced_like_interpreter(
        "local total, last = 0, '' \
         for round = 1, 120 do \
           local parts = {} \
           for i = 1, 600 do \
             parts[i] = tostring(i % 101) \
             if round == 100 and i == 300 then parts[i] = i end \
             if round == 110 and i == 10 then parts[i] = string.rep('z', 70) end \
           end \
           last = table.concat(parts, ',') \
           total = total + #last \
         end \
         return total, last",
    );
}

#[test]
fn growing_a_string_one_byte_at_a_time_matches_the_interpreter() {
    // The naive_concat shape, with a number joining the string late.
    assert_traced_like_interpreter(
        "local total, value = 0, '' \
         for round = 1, 60 do \
           value = '' \
           for i = 1, 400 do \
             if round == 50 and i == 200 then value = value .. i else \
               value = value .. string.char(65 + (i * 7 + round) % 26) end \
           end \
           total = total + #value \
         end \
         return total, value",
    );
}
