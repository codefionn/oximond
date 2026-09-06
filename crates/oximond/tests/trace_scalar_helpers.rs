//! `string.sub`, `string.rep`, `string.upper`, `string.lower`,
//! `string.reverse`, `string.byte`, `math.random`, `math.modf`, `math.log`,
//! `math.pow`, `tonumber` and `tostring` in a trace, in every profile.
//!
//! Each test runs its source interpreted and traced and compares the results,
//! and checks that the loops really ran natively: they compile, the trace is
//! entered, no recording aborts, and the helper the call is meant to use is
//! installed in the trace. A loop that silently stays interpreted fails.

use std::fmt::Write as _;

use oximond::runtime::Value;
use oximond::vm::TraceExitKind;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn has_integers(profile: CompatibilityProfile) -> bool {
    !matches!(
        profile,
        CompatibilityProfile::Lua51 | CompatibilityProfile::Lua52 | CompatibilityProfile::LuaJit21
    )
}

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
    let results = lua.exec(source).map_err(|error| format!("{error:?}"));
    (results, lua)
}

/// Runs `source` interpreted and traced in `profile`, checks that the
/// results agree and that the traced run recorded without an abort, entered
/// a trace and installed every helper in `helpers`. Returns the results.
fn check_profile(
    profile: CompatibilityProfile,
    source: &str,
    helpers: &[&str],
) -> Result<Vec<Value>, String> {
    let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
    let (actual, lua) = run(profile, ExecutionMode::Tracing, source);
    assert_eq!(actual, expected, "{profile:?}");
    let stats = lua.vm().jit_statistics().trace;
    assert!(stats.traces_compiled > 0, "{profile:?} {stats:?}");
    assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
    assert_eq!(stats.total_aborts(), 0, "{profile:?} {stats:?}");
    let calls = lua.vm().trace_helper_calls();
    for helper in helpers {
        assert!(
            calls.contains(helper),
            "{profile:?}: wanted {helper}, got {calls:?}"
        );
    }
    actual
}

/// [`check_profile`] in every profile, with the same helpers.
fn check(source: &str, helpers: &[&str]) {
    for profile in PROFILES {
        check_profile(profile, source, helpers).unwrap();
    }
}

/// The text of a string result.
fn text(values: &[Value]) -> String {
    match values.first() {
        Some(Value::String(text)) => String::from_utf8_lossy(text.as_bytes()).into_owned(),
        other => panic!("not a string: {other:?}"),
    }
}

/// Source that runs `call` (an expression of `k` and the locals `r1`, `r2`)
/// in a hot loop for each of the argument lists `cases`, as a function
/// each, and returns what every case gave as one string.
fn each_case(prelude: &str, cases: &[String], body: &str, iterations: u32) -> String {
    let mut source = format!("{prelude}\nlocal out = {{}}\n");
    for (position, case) in cases.iter().enumerate() {
        let function = format!(
            "local function case{position}()\n  local r1, r2\n  \
               for k = 1, {iterations} do {body} end\n  \
               return tostring(r1) .. ',' .. tostring(r2)\nend\n"
        );
        source.push_str(&function.replace("ARGS", case));
        let _ = writeln!(source, "out[#out + 1] = case{position}()");
    }
    source + "return table.concat(out, ';')"
}

/// Strings with embedded zeros, bytes of the upper half and mixed case, held
/// in a table so a trace reads them as borrowed strings.
const STRINGS: &str =
    r#"local T = {"hello, world", "", "\0b\0c\0", "\200\255x\128", "MiXeD 123 \233 \0Z"}"#;

/// The arguments `T[n]` for each string of [`STRINGS`], followed by `rest`.
fn with_strings(rest: &[&str]) -> Vec<String> {
    let mut cases = Vec::new();
    for string in 1..=5 {
        for rest in rest {
            cases.push(if rest.is_empty() {
                format!("T[{string}]")
            } else {
                format!("T[{string}], {rest}")
            });
        }
    }
    cases
}

#[test]
fn string_sub_in_general() {
    // Negative, zero, past the end, reversed and huge indices, on strings
    // with embedded zeros and high bytes.
    let cases = with_strings(&[
        "1, 2",
        "2, -1",
        "-3, -1",
        "0, 5",
        "5, 100",
        "-100, 3",
        "3, 2",
        "4611686018427387904, 1",
        "1, 4611686018427387904",
        "-4611686018427387904, -1",
        "-1, -1",
        "0, 0",
        "2",
        "-2",
        "0",
    ]);
    let source = each_case(STRINGS, &cases, "r1 = string.sub(ARGS)", 20000);
    check(&source, &["string.sub"]);
}

#[test]
fn string_sub_followed_by_string_byte_in_general() {
    // Every index pair on every string, with the substring read by `byte`
    // (one result and several) and by `#`: whatever the trace does with the
    // pair, the answer is the interpreter's.
    let cases = with_strings(&[
        "1, 2",
        "2, -1",
        "-3, -1",
        "0, 5",
        "5, 100",
        "-100, 3",
        "3, 2",
        "1, 4611686018427387904",
        "-4611686018427387904, -1",
        "0, 0",
        "2",
        "-2",
    ]);
    let source = each_case(
        STRINGS,
        &cases,
        "local piece = string.sub(ARGS) \
         local a, b = string.byte(piece, 1, 2) \
         r1 = (a or -1) * 1000 + (b or -1) + #piece \
         r2 = string.byte(piece, 2) or -2",
        20000,
    );
    check(&source, &["string.sub"]);
}

#[test]
fn string_sub_with_two_arguments_and_constants() {
    check(
        "local s = 'abcdefghij' local n = 0
         for k = 1, 60000 do
           local r = string.sub(s, k % 11 - 3) .. string.sub('constant', 2, k % 9)
           n = n + #r
         end
         return n",
        &["string.sub"],
    );
}

#[test]
fn string_sub_and_byte_with_float_indices_and_constants() {
    // Whole floats are indices in every profile; the trace reads them with
    // no call plan.
    let floats = [
        "2.0",
        "-1.0",
        "0.0",
        "-0.0",
        "1e0",
        "3.0, 7.0",
        "-4.0, -2.0",
        "1, 2.0",
        "2.0, 1",
    ];
    let cases = with_strings(&floats);
    let source = each_case(STRINGS, &cases, "r1 = string.sub(ARGS)", 20000);
    check(&source, &["string.sub"]);
    // The recorder leaves some byte ranges to the interpreter; the results
    // agree all the same.
    let source = each_case(STRINGS, &cases, "r1, r2 = string.byte(ARGS)", 20000);
    for profile in PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, &source);
        let (actual, _) = run(profile, ExecutionMode::Tracing, &source);
        assert_eq!(actual, expected, "{profile:?}");
    }
}

#[test]
fn string_sub_and_byte_with_indices_the_native_converts() {
    // Fractions, numeric strings, huge floats and a number as the source go
    // to the native's own conversion (or its error): both modes agree, in
    // every profile.
    let odd = [
        "2.5",
        "1, 2.5",
        "'2'",
        "1, '3'",
        "-'1'",
        "1e300",
        "-1e300, 1e300",
        "2^31",
        "-2^31 - 1",
    ];
    let cases = with_strings(&odd);
    for call in ["r1 = string.sub(ARGS)", "r1, r2 = string.byte(ARGS)"] {
        let source = each_case(
            STRINGS,
            &cases,
            &format!("local ok, v = pcall(function() {call} end) r2 = r2 or v"),
            2000,
        );
        for profile in PROFILES {
            let (expected, _) = run(profile, ExecutionMode::Interpreter, &source);
            let (actual, _) = run(profile, ExecutionMode::Tracing, &source);
            assert_eq!(actual, expected, "{profile:?} {call}");
        }
    }
    let source = "local t = 0 for k = 1, 60000 do
           t = t + #string.sub(12345, 2, 3 + k % 3) + string.byte(12345, -2) end return t";
    for profile in PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, _) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        assert!(expected.is_ok(), "{expected:?}");
    }
}

#[test]
fn string_byte_counts_around_the_bytes_a_string_has() {
    // Each count of results against ranges that give fewer, as many and more
    // bytes, empty ranges, and strings that are empty or short.
    for wanted in 0..=4 {
        for range in [
            "1, 4", "2, 3", "3, 2", "-2, -1", "0, 9", "5, 5", "1, 1", "-9, 9",
        ] {
            let names: Vec<String> = (0..wanted).map(|n| format!("v{n}")).collect();
            let body = if wanted == 0 {
                format!("t = t + select('#', string.byte(s, {range}))")
            } else {
                let sum: Vec<String> = names.iter().map(|name| format!("({name} or 7)")).collect();
                format!(
                    "local {} = string.byte(s, {range}) t = t + {} + select('#', string.byte(s, {range}))",
                    names.join(", "),
                    sum.join(" + ")
                )
            };
            let source = format!(
                "local t = 0 local words = {{'abcd', '', 'x'}}
                 for k = 1, 20000 do local s = words[k % 3 + 1] {body} end return t"
            );
            for profile in PROFILES {
                let (expected, _) = run(profile, ExecutionMode::Interpreter, &source);
                let (actual, _) = run(profile, ExecutionMode::Tracing, &source);
                assert_eq!(actual, expected, "{profile:?} {wanted} {range}");
                assert!(expected.is_ok(), "{expected:?}");
            }
        }
    }
}

#[test]
fn string_byte_with_more_results_than_the_loop_writes_in_place() {
    let source = "local s, t = 'abcdefghijkl', 0
         for k = 1, 5000 do
           local a, b, c, d, e, f, g, h, i, j = string.byte(s, 1, -1)
           t = t + a + j + select('#', string.byte(s, 1, -1)) + #{string.byte(s, 2, 5)}
         end return t";
    for profile in PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, _) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let total = 5000 * (97 + 106 + 12 + 4);
        let want = if has_integers(profile) {
            Value::Integer(total)
        } else {
            Value::Number(total as f64)
        };
        assert_eq!(expected, Ok(vec![want]), "{profile:?}");
    }
}

#[test]
fn string_sub_and_byte_arguments_that_error_say_the_same() {
    // A bad argument turns up after the loop has been hot: both modes raise
    // the same message from the native.
    for call in [
        "string.sub(s)",
        "string.sub(s, 1, nil)",
        "string.sub(s, nil)",
        "string.sub(s, {})",
        "string.sub(s, 1, 'x')",
        "string.sub(s, 1.5)",
        "string.sub({}, 1)",
        "string.byte(s, {})",
        "string.byte(s, 1, 'x')",
        "string.byte(s, 1.5)",
        "string.byte({}, 1)",
        "string.byte()",
        "string.sub(12345, 2, 3)",
        "string.byte(12345, -2, -1)",
    ] {
        let source = format!(
            "local s, t = 'hello', 0
             local function f(k) if k == 59000 then s = nil end return {call} end
             local function g(k) local v = f(k) return v end
             local ok, err = pcall(function()
               for k = 1, 60000 do t = t + #tostring(g(k)) s = s or 'hello' end end)
             return tostring(ok) .. '|' .. tostring(err) .. '|' .. t"
        );
        for profile in PROFILES {
            let (expected, _) = run(profile, ExecutionMode::Interpreter, &source);
            let (actual, _) = run(profile, ExecutionMode::Tracing, &source);
            assert_eq!(actual, expected, "{profile:?} {call}");
        }
    }
}

#[test]
fn string_rep_with_counts_and_separators() {
    let mut rest = Vec::new();
    for count in ["0", "1", "3", "-1", "17"] {
        rest.push(count.to_string());
        for separator in ["''", "', '"] {
            rest.push(format!("{count}, {separator}"));
        }
    }
    let rest: Vec<&str> = rest.iter().map(String::as_str).collect();
    let cases = with_strings(&rest);
    let source = each_case(STRINGS, &cases, "r1 = string.rep(ARGS)", 20000);
    check(&source, &["string.rep"]);
}

#[test]
fn string_upper_lower_and_reverse_on_all_bytes() {
    let cases = with_strings(&[""]);
    for (name, call) in [
        ("string.upper", "string.upper(ARGS)"),
        ("string.lower", "string.lower(ARGS)"),
        ("string.reverse", "string.reverse(ARGS)"),
    ] {
        let source = each_case(STRINGS, &cases, &format!("r1 = {call}"), 20000);
        check(&source, &[name]);
    }
}

#[test]
fn string_byte_of_a_string_the_trace_made() {
    // The bytes of a fresh string come from a helper, as many as the call
    // gave, and a different count leaves the trace.
    check(
        "local s = 'hello world' local t = 0
         for k = 1, 60000 do
           local r = string.sub(s, k % 5 + 1, k % 5 + 4)
           local a, b, c, d = string.byte(r, 1, 4)
           t = t + a + b + c + d + string.byte(r, -1)
         end
         return t",
        &["string.sub", "string.byte"],
    );
    check(
        "local t = 0
         for k = 1, 60000 do
           local r = string.rep('xy', k < 59000 and 2 or 1)
           local a, b = string.byte(r, 1, 2)
           t = t + a + (b or 9)
         end
         return t",
        &["string.rep", "string.byte"],
    );
}

#[test]
fn math_modf_of_integers_floats_and_specials() {
    let values = [
        "3.5",
        "-3.5",
        "-3.0",
        "-2147483648.0",
        "0.0",
        "-0.0",
        "1/0",
        "-1/0",
        "0/0",
        "1e300",
        "2^53",
        "7",
        "-7",
        "0.25",
        "-0.75",
        "1e15 + 0.5",
    ];
    let cases: Vec<String> = values.iter().map(|value| (*value).to_string()).collect();
    let source = each_case(
        "local modf = math.modf",
        &cases,
        "r1, r2 = modf(ARGS)",
        20000,
    );
    for profile in PROFILES {
        let helpers: &[&str] = if has_integers(profile) {
            &["math.modf_integer", "math.modf_float"]
        } else {
            &["math.modf_float"]
        };
        check_profile(profile, &source, helpers).unwrap();
    }
}

#[test]
fn math_log_with_bases_and_special_arguments() {
    let mut cases = Vec::new();
    for base in ["2", "10", "E", "1", "0", "-2", "1/0"] {
        for x in ["8", "0", "-1", "2.5", "1e300"] {
            cases.push(format!("{x}, {base}"));
        }
    }
    let source = each_case(
        "local log, E = math.log, math.exp(1)",
        &cases,
        "r1 = log(ARGS)",
        60000,
    );
    for profile in PROFILES {
        let (_, lua) = run(profile, ExecutionMode::Tracing, &source);
        check_profile(profile, &source, &[]).unwrap();
        // The logarithm is a call out of the generated code that no helper
        // makes.
        assert!(lua.vm().trace_external_calls() > 0, "{profile:?}");
    }
    let one = each_case(
        "local log = math.log",
        &["8.0".to_string(), "k".to_string()],
        "r1 = log(ARGS)",
        60000,
    );
    check(&one, &[]);
}

/// The logarithm in a base the loop changes: the profile's special cases for
/// bases 2 and 10 (and none for others, or `LuaJIT`'s reciprocal form) must
/// hold in the trace as in the interpreter, bit for bit.
#[test]
fn math_log_with_a_changing_base_follows_the_profile() {
    let source = "
local log, B, X = math.log, {2, 10, 3, 0.5, 2.0, 10.0}, {1000, 8, 1e15, 27, 0.1}
local h = 0
for k = 1, 60000 do
  local r = log(X[k % 5 + 1], B[k % 6 + 1])
  h = h + (r ~= r and 1 or r * 1e-3)
end
local out = {}
for i = 1, 5 do for j = 1, 6 do out[#out + 1] = string.format('%.17g', log(X[i], B[j])) end end
return h, table.concat(out, ',')";
    for profile in PROFILES {
        let results = check_profile(profile, source, &[]).unwrap();
        if profile == CompatibilityProfile::Lua55 {
            // Stock 5.5 gives exactly 3 for log(1000, 10).
            assert!(
                text(&results[1..]).starts_with("9.965784284662087,3,"),
                "{results:?}"
            );
        }
    }
}

/// `math.pow` is gone from Lua 5.5 and kept in the other profiles, and
/// Lua 5.2's `assert ()` with no argument fails with the default message.
#[test]
fn profile_library_differences_hold_in_a_traced_loop() {
    let source = "
local n, failures = 0, 0
for k = 1, 60000 do
  if math.pow then n = n + 1 end
  local ok, message = pcall(assert)
  if not ok then failures = failures + (message == 'assertion failed!' and 1 or 100000) end
end
return n, failures";
    for profile in PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, _) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let pow = if profile == CompatibilityProfile::Lua55 {
            0
        } else {
            60000
        };
        // Only Lua 5.2 raises the default message; the others reject the call
        // for its missing argument.
        let failures = if profile == CompatibilityProfile::Lua52 {
            60000
        } else {
            6_000_000_000
        };
        let results = actual.unwrap();
        assert_eq!(results.len(), 2, "{profile:?}");
        assert_eq!(results[0], number(profile, pow), "{profile:?}");
        assert_eq!(results[1], number(profile, failures), "{profile:?}");
    }
}

fn number(profile: CompatibilityProfile, value: i64) -> Value {
    if has_integers(profile) {
        Value::Integer(value)
    } else {
        #[allow(clippy::cast_precision_loss, reason = "the counts are small")]
        Value::Number(value as f64)
    }
}

#[test]
fn math_pow_where_installed() {
    let source = "
local pow = math.pow
if not pow then return 'none' end
local t = 0
for k = 1, 60000 do t = t + pow(k % 7, 2) + pow(2, k % 5) + pow(k, 0.5) end
return t";
    for profile in PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let installed = expected != Ok(vec![Value::from("none")]);
        if installed {
            check_profile(profile, source, &[]).unwrap();
        }
    }
}

#[test]
fn tonumber_of_strings_with_edge_inputs() {
    let inputs = [
        "' 0x1p4 '",
        "'1e'",
        "'10'",
        "'9223372036854775808'",
        "'  12  '",
        "'0x'",
        "'abc'",
        "'1e5'",
        "'  -0x10  '",
        "'inf'",
        "'nan'",
        "''",
        "'0x7fffffffffffffff'",
        "'-9223372036854775808'",
        "'1_000'",
        "'1.5e+3'",
        "'.5'",
        "'5.'",
        "'0b1'",
        "'1 2'",
    ];
    let cases: Vec<String> = inputs.iter().map(|input| (*input).to_string()).collect();
    let source = each_case(
        "local tonumber = tonumber",
        &cases,
        "r1 = tonumber(ARGS)",
        20000,
    );
    for profile in PROFILES {
        let helpers: &[&str] = if has_integers(profile) {
            &["tonumber_integer", "tonumber_float", "tonumber_nil"]
        } else {
            &["tonumber_float", "tonumber_nil"]
        };
        check_profile(profile, &source, helpers).unwrap();
    }
}

#[test]
fn tonumber_with_bases() {
    let inputs = [
        "'1010', 2",
        "'ff', 16",
        "'FF', 16",
        "'zz', 36",
        "'Zz', 36",
        "'12', 2",
        "'  ff  ', 16",
        "'-ff', 16",
        "'7fffffffffffffff', 16",
        "'ffffffffffffffff', 16",
        "'g', 16",
        "'', 10",
        "'10', 10",
        "' 10 ', 10",
        "'1e1', 10",
        "'777', 8",
        "'0x10', 16",
    ];
    let cases: Vec<String> = inputs.iter().map(|input| (*input).to_string()).collect();
    let source = each_case(
        "local tonumber = tonumber",
        &cases,
        "r1 = tonumber(ARGS)",
        20000,
    );
    for profile in PROFILES {
        let helpers: &[&str] = if has_integers(profile) {
            &["tonumber_integer", "tonumber_nil"]
        } else {
            &["tonumber_float", "tonumber_nil"]
        };
        check_profile(profile, &source, helpers).unwrap();
    }
}

#[test]
fn tonumber_of_strings_from_a_table_with_mixed_results() {
    // The classes of the answers differ from one iteration to the next, so
    // the helper leaves the trace, the trace is retired, and the results
    // stay the interpreter's.
    let source = "
local words = {'10', '2.5', 'x', '0x10', '7', '1e2', ' 3 ', '-1'}
local t, bad = 0, 0
for k = 1, 60000 do
  local v = tonumber(words[k % 8 + 1])
  if v then t = t + v else bad = bad + 1 end
end
return t, bad";
    for profile in PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert_eq!(stats.total_aborts(), 0, "{profile:?} {stats:?}");
    }
}

#[test]
fn tostring_of_booleans_and_nil() {
    check(
        "local n = 0
         for k = 1, 60000 do
           local flag = k % 3 == 0
           n = n + #tostring(flag) + #tostring(nil) + #tostring(not flag)
         end
         return n",
        &["tostring"],
    );
}

/// What `math.random` gives, drawn in a hot loop.
const DRAW: &str = "
local function draw(n)
  local s = 0
  local last = 0
  for k = 1, n do
    s = s + math.random() * 1000 + math.random(10) + math.random(5, 20)
    last = math.random(1000000)
  end
  return s, last
end";

#[test]
fn math_random_sequence_matches_the_interpreter_in_every_profile() {
    let source = format!(
        "{DRAW}
math.randomseed(42)
local a, la = draw(30000)
return a, la, math.random(), math.random(100)"
    );
    for profile in PROFILES {
        let helpers: &[&str] = if has_integers(profile) {
            &["math.random_float", "math.random_integer"]
        } else {
            &["math.random_float"]
        };
        check_profile(profile, &source, helpers).unwrap();
    }
}

#[test]
fn math_randomseed_between_traced_loops_is_observed() {
    let source = format!(
        "{DRAW}
math.randomseed(42)
local a, la = draw(30000)
math.randomseed(42)
local b, lb = draw(30000)
math.randomseed(7)
local c, lc = draw(30000)
return a == b, la == lb, a ~= c, la ~= lc, a, b, c"
    );
    for profile in PROFILES {
        let values = check_profile(profile, &source, &["math.random_float"]).unwrap();
        // The same seed gives the same draw, another seed another one.
        assert_eq!(
            &values[..4],
            &[
                Value::Boolean(true),
                Value::Boolean(true),
                Value::Boolean(true),
                Value::Boolean(true)
            ],
            "{profile:?}"
        );
    }
}

/// A draw advances the generator and nothing a table holds, so the reads
/// around it are still merged and the invariant one is still hoisted.
#[test]
fn math_random_does_not_block_table_load_forwarding_or_hoisting() {
    let source = "
math.randomseed(3)
local t = {1.5, 2.5, 3.5, 4.5}
local scale = {2.0}
local s = 0
for i = 1, 60000 do
  local j = i % 4 + 1
  local before = t[j]
  s = s + before * math.random() * scale[1]
  s = s + t[j]
end
return s";
    for profile in PROFILES {
        let helper = "math.random_float";
        check_profile(profile, source, &[helper]).unwrap();
        let (_, lua) = run(profile, ExecutionMode::Tracing, source);
        let trace = lua.vm().jit_statistics().trace;
        let optimizer = trace.optimizer;
        // The recording answers the read from the slot it read (slot
        // facts), or the optimizer merges the two loads.
        assert!(
            trace.slot_reads_forwarded > 0
                || optimizer.loads_merged + optimizer.loads_forwarded > 0,
            "{profile:?}: the read of t[j] after the draw is a new load: {trace:?}"
        );
        assert!(
            optimizer.loads_hoisted > 0,
            "{profile:?}: scale[1] is invariant across the draw: {optimizer:?}"
        );
    }
}

#[test]
fn a_loop_that_updates_a_table_with_draws_keeps_its_forwarded_reads() {
    let source = "
math.randomseed(5)
local t = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5}
for k = 1, 96000 do
  local i = k % 8 + 1
  t[i] = t[i] % 1000 + math.random() * t[i]
end
return t[1], t[4], t[8]";
    for profile in PROFILES {
        check_profile(profile, source, &["math.random_float"]).unwrap();
        let (_, lua) = run(profile, ExecutionMode::Tracing, source);
        let trace = lua.vm().jit_statistics().trace;
        let optimizer = trace.optimizer;
        // The recording answers the read from the slot it read (slot
        // facts), or the optimizer merges the two loads.
        assert!(
            trace.slot_reads_forwarded > 0
                || optimizer.loads_merged + optimizer.loads_forwarded > 0,
            "{profile:?}: the second read of t[i] is the first: {trace:?}"
        );
    }
}

#[test]
fn math_random_is_not_hoisted_out_of_the_loop() {
    let source = "
math.randomseed(1)
local changes, prev = 0, -1
for k = 1, 60000 do
  local r = math.random(1000000)
  if r ~= prev then changes = changes + 1 end
  prev = r
end
return changes";
    for profile in PROFILES {
        let helper = if has_integers(profile) {
            "math.random_integer"
        } else {
            "math.random_float"
        };
        let values = check_profile(profile, source, &[helper]).unwrap();
        let changes = match values[0] {
            Value::Integer(changes) => {
                assert!(changes > 59000, "{profile:?}: {changes}");
                continue;
            }
            Value::Number(changes) => changes,
            ref other => panic!("{other:?}"),
        };
        assert!(changes > 59000.0, "{profile:?}: {changes}");
    }
}

#[test]
fn math_random_with_arguments_the_interpreter_rejects_leaves_the_state_alone() {
    // The draw that raises leaves the trace before the generator moves, so
    // the draws after the error are the interpreter's.
    let source = "
math.randomseed(5)
local ok, err = pcall(function()
  local t = 0
  for k = 1, 60000 do
    t = t + math.random(k < 59990 and 10 or -5)
  end
  return t
end)
return ok, err, math.random(1000), math.random(1000), math.random()";
    for profile in PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
    }
}

/// Runs `r = {call}` in a loop under pcall interpreted and traced in every
/// profile. `call` may name `A`, which is `good` until the last iterations
/// and `bad` after. The trace must be entered before the bad argument
/// arrives, and the error must be the interpreter's.
fn check_late_error(good: &str, bad: &str, call: &str, helpers: &[&str]) {
    let source = format!(
        "local ok, err = pcall(function()
           local r
           local bad, ts = {{}}, setmetatable({{}}, {{__tostring = error}})
           for k = 1, 60000 do
             local A = {good}
             if k >= 59990 then A = {bad} end
             r = {call}
           end
           return r
         end)
         return ok, err"
    );
    let case = format!("{call} with A = {good} then {bad}");
    for profile in PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, &source);
        let (actual, lua) = run(profile, ExecutionMode::Tracing, &source);
        assert_eq!(actual, expected, "{case} {profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.traces_compiled > 0, "{case} {profile:?} {stats:?}");
        assert!(stats.native_entries > 0, "{case} {profile:?} {stats:?}");
        let calls = lua.vm().trace_helper_calls();
        for helper in helpers {
            assert!(
                calls.contains(helper),
                "{case} {profile:?}: wanted {helper}, got {calls:?}"
            );
        }
    }
}

#[test]
fn wrong_arguments_leave_the_trace_and_raise_the_interpreters_error() {
    // Each loop passes a good argument until its last iterations and then a
    // bad one of the same type, so the trace is entered, the helper misses,
    // and the interpreter raises.
    let cases: [(&str, &str, &str, &[&str]); 7] = [
        ("1.0", "1.5", "string.sub('abc', A)", &["string.sub"]),
        ("3", "4294967296", "string.rep('x', A)", &["string.rep"]),
        (
            "3",
            "4294967296",
            "string.rep('x', A, 'y')",
            &["string.rep"],
        ),
        ("3", "1e300", "string.rep('abc', A)", &["string.rep"]),
        ("10", "99", "tonumber('10', A)", &[]),
        ("10", "1", "tonumber('10', A)", &[]),
        ("10", "-5", "math.random(A)", &[]),
    ];
    for (good, bad, call, helpers) in cases {
        check_late_error(good, bad, call, helpers);
    }
}

#[test]
fn arguments_of_the_wrong_type_leave_the_trace_and_raise_the_interpreters_error() {
    // The last iterations switch to a table, nil, a boolean or a string that
    // is no number, which the helpers do not take.
    let cases: [(&str, &str, &str, &[&str]); 14] = [
        ("'x'", "bad", "string.upper(A)", &["string.upper"]),
        ("'x'", "bad", "string.lower(A)", &["string.lower"]),
        ("'x'", "true", "string.reverse(A)", &["string.reverse"]),
        ("'abc'", "nil", "string.sub(A, 1, 2)", &["string.sub"]),
        ("1", "bad", "string.sub('abc', A, 2)", &["string.sub"]),
        ("2", "'x'", "string.sub('abcdef', 1, A)", &["string.sub"]),
        ("3", "bad", "string.rep('x', A)", &["string.rep"]),
        ("'y'", "bad", "string.rep('x', 3, A)", &["string.rep"]),
        ("'10'", "bad", "tonumber(A, 10)", &[]),
        ("10", "bad", "tonumber('10', A)", &[]),
        ("3", "bad", "math.random(A)", &[]),
        ("1.5", "'x'", "math.modf(A)", &[]),
        ("2", "bad", "math.log(A, 2)", &[]),
        ("true", "ts", "tostring(A)", &[]),
    ];
    for (good, bad, call, helpers) in cases {
        check_late_error(good, bad, call, helpers);
    }
}

#[test]
fn string_rep_over_the_size_limit_matches_the_interpreters_error() {
    let source = "
local ok, err = pcall(function()
  local n = 0
  for k = 1, 60000 do
    local r = string.rep('abcdefgh', k < 59990 and 5 or 600000000)
    n = n + #r
  end
  return n
end)
return ok, err";
    for profile in PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let message = text(&actual.unwrap()[1..]);
        assert!(message.contains("too large"), "{profile:?}: {message}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert!(
            lua.vm().trace_helper_calls().contains(&"string.rep"),
            "{profile:?}"
        );
    }
}

fn benchmark(path: &str) -> String {
    let path = format!("{}/../../benchmarks/lua/{path}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{path}: {error}"))
}

#[test]
fn the_byte_and_sub_benchmark_compiles() {
    let source = benchmark("string/byte_sub.lua");
    // In every profile the loop reads the source string, and the optimized
    // trace makes no call at all: the substring only a recovery path would
    // hold is never made.
    for profile in PROFILES {
        check_profile(profile, &source, &[]).unwrap();
        let (_, lua) = run(profile, ExecutionMode::Tracing, &source);
        assert_eq!(lua.vm().trace_compiled_external_calls(), 0, "{profile:?}");
        assert!(
            !lua.vm().trace_helper_calls().contains(&"string.byte"),
            "{profile:?}"
        );
    }
}

#[test]
fn the_tonumber_benchmark_compiles() {
    let source = benchmark("stdlib/tonumber_parse.lua");
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::LuaJit21,
    ] {
        check_profile(profile, &source, &["tonumber_nil"]).unwrap();
    }
}

// ---- calls whose arguments are another call's results ------------------------
//
// `string.upper (tostring (k))` takes the open results of the inner call. The
// interpreter keeps them in a slot of the frame that no snapshot rebuilds, so
// when the outer call misses the trace leaves through the snapshot from
// before the inner call, and the interpreter makes both again.

/// Runs `source` in every profile interpreted and traced, checks that the
/// results (or errors) agree and that every loop compiled and ran natively
/// without an abort. Returns the results.
fn check_nested(source: &str, helpers: &[&str]) {
    check_nested_exits(source, helpers, false);
}

/// [`check_nested`], and with `leaves` the traced run left a trace through a
/// guard at least once.
fn check_nested_exits(source: &str, helpers: &[&str], leaves: bool) {
    for profile in PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.traces_compiled > 0, "{profile:?} {source} {stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        // The loops run 60000 iterations: most of them are native ones.
        assert!(stats.iterations > 10_000, "{profile:?} {source} {stats:?}");
        assert_eq!(
            stats.total_aborts(),
            0,
            "{profile:?} {source} {stats:?} {:?}",
            stats.native_aborts
        );
        if leaves {
            assert!(
                stats.exits_of(TraceExitKind::GuardFail) > 0,
                "{profile:?} {source} {stats:?}"
            );
        }
        let calls = lua.vm().trace_helper_calls();
        for helper in helpers {
            assert!(
                calls.contains(helper),
                "{profile:?} {source}: wanted {helper}, got {calls:?}"
            );
        }
    }
}

/// `body` in a hot loop of `k = 1, 60000`, wrapped in a `pcall` so an error
/// the late iterations raise is a result both runs compare.
fn nested_loop(prelude: &str, body: &str) -> String {
    format!(
        "{prelude}
local h = 0
local ok, message = pcall(function()
  for k = 1, 60000 do
    {body}
  end
end)
return h, ok, message"
    )
}

#[test]
fn a_helper_takes_the_results_of_a_helper() {
    check_nested(
        &nested_loop("", "h = (h + #string.upper(tostring(k))) % 1000003"),
        &["tostring", "string.upper"],
    );
    check_nested(
        &nested_loop(
            "local T = {'hello, world', 'abc', 'xYz!!'}",
            "h = (h + #string.reverse(string.sub(T[k % 3 + 1], 2))) % 1000003",
        ),
        &["string.sub", "string.reverse"],
    );
    check_nested(
        &nested_loop(
            "local T = {'Hello', 'ABC', 'xYz'}",
            "h = (h + #string.rep(string.lower(T[k % 3 + 1]), 2)) % 1000003",
        ),
        &["string.lower", "string.rep"],
    );
    check_nested(
        &nested_loop("", "h = (h + #string.rep(tostring(k), 3, '-')) % 1000003"),
        &["tostring", "string.rep"],
    );
}

#[test]
fn a_helper_that_misses_late_leaves_before_the_inner_call() {
    // The count of `string.rep` grows past what a string may hold, and the
    // text `tonumber` gets stops being a number: each outer call misses in
    // the trace and the interpreter makes both calls, raising or giving what
    // it does.
    check_nested_exits(
        &nested_loop(
            "local T = {'ab', 'cd'}",
            "h = (h + #string.rep(string.lower(T[k % 2 + 1]), k < 59000 and 2 or 1e10)) % 1000003",
        ),
        &[],
        true,
    );
    check_nested_exits(
        &nested_loop(
            "local T = {'11', '12', '13'}",
            // From 59000 on the substring is empty. (A start of 2 gave a
            // digit: only the branch guard on `k < 59000` left, and a value
            // select has none.)
            "local v = tonumber(string.sub(T[k % 3 + 1], k < 59000 and 1 or 3, 2)) \
             h = (h + (v or 7)) % 1000003",
        ),
        &[],
        true,
    );
    check_nested_exits(
        &nested_loop(
            "local T = {'10', '20', '30'} local bad = {'x', 'y', 'z'}",
            "local v = tonumber(string.upper((k < 59000 and T or bad)[k % 3 + 1])) \
             h = (h + (v or 7)) % 1000003",
        ),
        &[],
        true,
    );
}

#[test]
fn table_insert_takes_the_result_of_a_function() {
    // An inlined Lua function, a helper and a library function in a field.
    // The table keeps a first element: an array that is empty at every entry
    // is not one the trace takes (see `bind_tables`).
    check_nested(
        &nested_loop(
            "local t = {0} local function f(x) return x * 2 end",
            "table.insert(t, f(k)) h = (h + table.remove(t)) % 1000003",
        ),
        &[],
    );
    check_nested(
        &nested_loop(
            "local t = {0} local s = 'abc'",
            "table.insert(t, string.upper(s)) h = (h + #table.remove(t) + k) % 1000003",
        ),
        &["string.upper"],
    );
    check_nested(
        &nested_loop(
            "local t = {} local function f(x) return x end",
            "table.insert(t, f(k)) if #t > 50 then t[#t] = nil end h = (h + #t) % 1000003",
        ),
        &[],
    );
    // The inner function gives nil late: the trace leaves at the function's
    // branch guard, and the error comes from the `h + table.remove(t)` that
    // follows (`table.insert(t, nil)` itself raises nothing).
    check_nested(
        &nested_loop(
            "local t = {0} local function f(x) if x < 59000 then return x end end",
            "table.insert(t, f(k)) h = (h + table.remove(t)) % 1000003",
        ),
        &[],
    );
}

#[test]
fn a_late_exit_after_an_inlined_producer_does_not_replay_it() {
    // The trace leaves before the insert; folding `calls` into the result
    // shows a second run of `f` in the interpreter's replay.
    check_nested_exits(
        &nested_loop(
            "local t, calls = {}, 0 local function f(x) calls = calls + 1 return x * 2 end",
            "if k == 59000 then t.tag = 1 end table.insert(t, f(k)) \
             h = (h + t[#t] + #t + calls) % 1000003",
        ),
        &[],
        true,
    );
    check_nested_exits(
        &nested_loop(
            "local T = {'12', 'zz'} local function g(x) return T[1 + math.floor(x / 59000)] end",
            "local v = tonumber(g(k)) h = (h + (v or 7)) % 1000003",
        ),
        &[],
        true,
    );
}

#[test]
fn tonumber_takes_the_result_of_a_function() {
    check_nested(
        &nested_loop(
            "local function g(x) return tostring(x) end",
            "h = (h + tonumber(g(k))) % 1000003",
        ),
        &["tostring"],
    );
    check_nested(
        &nested_loop(
            "local T = {'1', '22', '333'}",
            "h = (h + tonumber(string.sub(T[k % 3 + 1], 1, 2))) % 1000003",
        ),
        &["string.sub"],
    );
}

#[test]
fn nested_calls_blame_the_function_that_aborts() {
    // Whatever the recording cannot take, the counters name the call that
    // failed, not an earlier function met on the way.
    let source = nested_loop(
        "local T = {'a,b', 'c,d'}",
        "local a, b = string.find(string.upper(T[k % 2 + 1]), '(%a),') h = (h + a) % 1000003",
    );
    for profile in PROFILES {
        let (_, lua) = run(profile, ExecutionMode::Tracing, &source);
        let stats = lua.vm().jit_statistics().trace;
        for (name, count) in stats.native_aborts.iter() {
            assert!(
                name != "string.upper" || count == 0,
                "{profile:?}: string.upper did not abort: {:?}",
                stats.native_aborts
            );
        }
    }
}
