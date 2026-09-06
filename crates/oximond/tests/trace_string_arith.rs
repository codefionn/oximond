//! Arithmetic on strings in a trace, in every profile.
//!
//! The interpreter converts a numeric string operand: to a float before Lua
//! 5.3 and in Lua 5.3 itself, to an integer or a float as the numeral reads
//! through the string metatable's arithmetic metamethods in Lua 5.4 and 5.5.
//! A trace holds the conversion as a helper call. The loops compile, run
//! natively and give the interpreter's results, integer or float. A string
//! that stops converting, converts to the other kind of number or loses its
//! metamethod sends the run back to the interpreter, which raises the error
//! or runs the metamethod.

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

fn run(
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
    lua.vm_mut().ignore_remaining_work();
    let results = lua.exec(source).map_err(|error| format!("{error:?}"));
    (results, lua)
}

fn has_floor_division(profile: CompatibilityProfile) -> bool {
    matches!(
        profile,
        CompatibilityProfile::Lua53 | CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
    )
}

/// Runs `source` interpreted and traced in `profile` and checks the results
/// agree and succeed, and that the traced run ran natively most of its
/// iterations, aborted nowhere and called a conversion helper.
fn check_profile(profile: CompatibilityProfile, source: &str) {
    let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
    let (actual, lua) = run(profile, ExecutionMode::Tracing, source);
    assert_eq!(actual, expected, "{profile:?} {source}");
    assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
    let stats = lua.vm().jit_statistics().trace;
    assert!(stats.traces_compiled > 0, "{profile:?} {source} {stats:?}");
    assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
    assert!(stats.iterations > 10000, "{profile:?} {source} {stats:?}");
    assert_eq!(
        stats.total_aborts(),
        0,
        "{profile:?} {source} {stats:?} {:?}",
        stats.native_aborts
    );
    let calls = lua.vm().trace_helper_calls();
    assert!(
        calls.contains(&"string.number_integer") || calls.contains(&"string.number_float"),
        "{profile:?} {source}: got {calls:?}"
    );
}

/// Like [`check_profile`] for a loop that meets a bad case late: the results
/// (or the errors) agree and the loop ran natively before.
fn check_exit(profile: CompatibilityProfile, source: &str) {
    let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
    let (actual, lua) = run(profile, ExecutionMode::Tracing, source);
    assert_eq!(actual, expected, "{profile:?} {source}");
    let stats = lua.vm().jit_statistics().trace;
    assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
    assert!(stats.iterations > 10000, "{profile:?} {source} {stats:?}");
}

/// Three strings that convert to numbers of one kind, in a table so a trace
/// reads them as borrowed strings.
const SETS: [(&str, &str); 5] = [
    ("integers", r#"{"10", "42", "7"}"#),
    ("floats", r#"{"1.5", "2.25", "10.0"}"#),
    ("hexadecimal", r#"{"0x10", "0xff", "0x7"}"#),
    ("exponents", r#"{"1e2", "2.5e1", "3E0"}"#),
    ("padded", r#"{" 10 ", "\t42\n", " 7"}"#),
];

/// The loops over one set of strings: each operator with the string on
/// either side of a number, with a float, and with a string on both sides.
/// The result's type is part of the answer.
fn loops(profile: CompatibilityProfile, set: &str) -> Vec<String> {
    let mut operators = vec!["+", "-", "*", "/", "%", "^"];
    if has_floor_division(profile) {
        operators.push("//");
    }
    let mut sources = Vec::new();
    for operator in operators {
        for expression in [
            format!("s {operator} 3"),
            format!("3 {operator} s"),
            format!("s {operator} 1.5"),
            format!("2.5 {operator} s"),
            format!("s {operator} t"),
        ] {
            sources.push(format!(
                "local strs = {set} local n, last = 0, nil \
                 for i = 1, 20000 do \
                   local s, t = strs[i % 3 + 1], strs[(i + 1) % 3 + 1] \
                   last = {expression} n = n + last \
                 end \
                 return n, last, math.type and math.type(last)"
            ));
        }
    }
    sources.push(format!(
        "local strs = {set} local n, last = 0, nil \
         for i = 1, 20000 do local s = strs[i % 3 + 1] last = -s n = n + last end \
         return n, last, math.type and math.type(last)"
    ));
    sources
}

#[test]
fn numeric_strings_convert_in_every_operator() {
    for profile in PROFILES {
        for (_, set) in SETS {
            for source in loops(profile, set) {
                check_profile(profile, &source);
            }
        }
    }
}

#[test]
fn constants_and_arithmetic_chains_convert() {
    for profile in PROFILES {
        for body in [
            // The constant forms of the instructions.
            "last = s * 2",
            "last = s + 1",
            "last = s - 1",
            "last = s / 2",
            "last = s % 7",
            "last = s ^ 2",
            // Several conversions of one iteration and a use of the result.
            "last = (s + 1) * (t - 2) + (s / t)",
            "last = -s + -t",
            "last = (s + t) % 5 + #s",
        ] {
            let source = format!(
                "local strs = {{\"10\", \"42\", \"7\"}} local n, last = 0, nil \
                 for i = 1, 20000 do \
                   local s, t = strs[i % 3 + 1], strs[(i + 1) % 3 + 1] \
                   {body} n = n + last \
                 end return n, last, math.type and math.type(last)"
            );
            check_profile(profile, &source);
        }
    }
}

#[test]
fn a_string_built_in_the_loop_converts() {
    // The strings are lanes of the run, not borrowed from a table.
    for profile in PROFILES {
        check_profile(
            profile,
            "local n = 0 for i = 1, 20000 do \
               local s = string.format('%d', i) n = n + s * 2 - s end \
             return n, math.type and math.type(n)",
        );
        check_profile(
            profile,
            "local n = 0 for i = 1, 20000 do \
               local year = string.match('2024-01-17', '^(%d+)') n = n + year + 1 end \
             return n, math.type and math.type(n)",
        );
    }
}

#[test]
fn nonfinite_spellings_follow_the_profile() {
    // `inf` and `nan` convert in Lua 5.1 and LuaJIT only.
    for profile in PROFILES {
        for spelling in ["inf", "nan", "0x", "1e", "1 2", ""] {
            let source = format!(
                "local n = 0 local s = '{spelling}' \
                 for i = 1, 20000 do \
                   if i == 19900 then s = '5' end \
                   local ok, r = pcall(function() return s + 1 end) \
                   if ok and r == r then n = n + r end \
                 end return n"
            );
            let (expected, _) = run(profile, ExecutionMode::Interpreter, &source);
            let (actual, _) = run(profile, ExecutionMode::Tracing, &source);
            assert_eq!(actual, expected, "{profile:?} {source}");
        }
    }
}

#[test]
fn a_string_that_does_not_convert_leaves_to_the_interpreter() {
    for profile in PROFILES {
        for (late, expression) in [
            ("'abc'", "s + 1"),
            ("'abc'", "1 + s"),
            ("'abc'", "-s"),
            ("'abc'", "s * s"),
            ("''", "s ^ 2"),
            ("'1x'", "s / 2"),
        ] {
            check_exit(
                profile,
                &format!(
                    "local s = '12' local n = 0 \
                     for i = 1, 20000 do \
                       if i == 19900 then s = {late} end \
                       n = n + ({expression}) \
                     end return n"
                ),
            );
        }
        // The error is caught, so the loop goes on after it.
        check_exit(
            profile,
            "local s = '12' local n = 0 local f = function(s) return s + 1 end \
             for i = 1, 20000 do \
               if i % 5000 == 0 then s = 'abc' else s = '12' end \
               local ok, r = pcall(f, s) \
               if ok then n = n + r else n = n + #r end \
             end return n",
        );
    }
}

#[test]
fn a_string_that_converts_to_another_kind_leaves() {
    // An integer numeral, then a float one: the kind of the result changes
    // in Lua 5.4 and later, and the trace leaves before the instruction.
    for profile in PROFILES {
        for expression in ["s + 1", "1 - s", "s * 2", "s // 2", "s % 3", "-s", "s / 2"] {
            if expression.contains("//") && !has_floor_division(profile) {
                continue;
            }
            check_exit(
                profile,
                &format!(
                    "local s = '12' local n = 0 local last \
                     for i = 1, 20000 do \
                       if i == 19900 then s = '12.5' end \
                       last = {expression} n = n + last \
                     end return n, last, math.type and math.type(last)"
                ),
            );
            check_exit(
                profile,
                &format!(
                    "local s = '12.5' local n = 0 local last \
                     for i = 1, 20000 do \
                       if i == 19900 then s = '0x10' end \
                       last = {expression} n = n + last \
                     end return n, last, math.type and math.type(last)"
                ),
            );
        }
    }
}

#[test]
fn a_replaced_string_metamethod_runs() {
    // Lua 5.4 and 5.5 look the arithmetic up in the string metatable, which
    // the program replaces late. Before Lua 5.4 the field changes nothing.
    for profile in PROFILES {
        for field in [
            "__add", "__sub", "__mul", "__div", "__mod", "__pow", "__unm",
        ] {
            let expression = match field {
                "__add" => "s + 1",
                "__sub" => "s - 1",
                "__mul" => "s * 2",
                "__div" => "s / 2",
                "__mod" => "s % 3",
                "__pow" => "s ^ 2",
                _ => "-s",
            };
            check_exit(
                profile,
                &format!(
                    "local s = '12' local n = 0 local mt = getmetatable('') \
                     for i = 1, 20000 do \
                       if i == 19900 then mt.{field} = function(a, b) return 1000 end end \
                       n = n + ({expression}) \
                     end return n"
                ),
            );
            // A metamethod that is nil from the start of the recording on.
            check_exit(
                profile,
                &format!(
                    "local s = '12' local n = 0 local mt = getmetatable('') \
                     local f = function(s) return {expression} end \
                     for i = 1, 20000 do \
                       if i == 19900 then mt.{field} = nil end \
                       local ok, r = pcall(f, s) \
                       if ok then n = n + r else n = n + #r end \
                     end return n"
                ),
            );
        }
    }
}

#[test]
fn a_removed_string_metatable_is_the_interpreters_to_answer() {
    for profile in PROFILES {
        check_exit(
            profile,
            "local s = '12' local n = 0 local f = function(s) return s + 1 end \
             for i = 1, 20000 do \
               if i == 19900 then debug.setmetatable('', nil) end \
               local ok, r = pcall(f, s) \
               if ok then n = n + r else n = n + #r end \
             end return n",
        );
        check_exit(
            profile,
            "local s = '12' local n = 0 local f = function(s) return -s end \
             for i = 1, 20000 do \
               if i == 19900 then debug.setmetatable('', nil) end \
               local ok, r = pcall(f, s) \
               if ok then n = n + r else n = n + #r end \
             end return n",
        );
    }
}

#[test]
fn a_number_metatable_takes_the_operation_first() {
    // With a string on the left, Lua 5.4 gives a number's own metamethod the
    // operation (as lstrlib's `trymt` does).
    for profile in PROFILES {
        check_exit(
            profile,
            "local s = '12' local n = 0 \
             for i = 1, 20000 do \
               if i == 19900 then debug.setmetatable(0, { __add = function(a, b) return 7 end }) end \
               n = n + (s + 1) \
             end return n",
        );
    }
}

#[test]
fn a_table_operand_keeps_its_metamethod() {
    // A string and a table with `__add`: the recorder leaves it alone.
    for profile in PROFILES {
        let source = "local mt = { __add = function(a, b) return 3 end } \
                      local t = setmetatable({}, mt) local s = '10' local n = 0 \
                      for i = 1, 20000 do n = n + (s + t) + (t + s) end return n";
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, _) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
    }
}

#[test]
fn bitwise_operations_on_strings_stay_with_the_interpreter() {
    // Only Lua 5.3 converts a string in a bitwise operation, and a trace
    // leaves that to the interpreter: the results are its own.
    for profile in PROFILES {
        for expression in ["s | 1", "s & 3", "s ~ 5", "s << 1", "s >> 1", "~s"] {
            let source = format!(
                "local s = '12' local n = 0 \
                 for i = 1, 20000 do \
                   local ok, r = pcall(function() return {expression} end) \
                   if ok then n = n + r else n = n + #r end \
                 end return n"
            );
            // Lua 5.1, 5.2 and LuaJIT have no such operators.
            let (expected, _) = run(profile, ExecutionMode::Interpreter, &source);
            let (actual, _) = run(profile, ExecutionMode::Tracing, &source);
            assert_eq!(actual, expected, "{profile:?} {source}");
        }
    }
}

#[test]
fn integer_and_float_results_keep_their_kind() {
    // The sum of integer-looking strings is an integer in Lua 5.4 and 5.5
    // and a float in the other profiles.
    for profile in PROFILES {
        let source = "local strs = {'10', '42', '7'} local n = 0 \
                      for i = 1, 20000 do n = n + strs[i % 3 + 1] end \
                      return n, math.type and math.type(n)";
        check_profile(profile, source);
        let (results, _) = run(profile, ExecutionMode::Tracing, source);
        let results = results.unwrap();
        let integer = matches!(
            profile,
            CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
        );
        assert_eq!(
            matches!(results[0], Value::Integer(_)),
            integer,
            "{profile:?} {results:?}"
        );
    }
}
