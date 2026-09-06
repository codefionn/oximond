//! Integer and float subtypes of library results under each profile.
//!
//! Lua 5.3 and later specify integer results for lengths, positions, counts,
//! and rounded values. The battery in `fixtures/integer_results.lua` prints
//! the `tostring` and `math.type` of each result, and is diffed against the
//! stock `lua5.4` and `lua5.5` interpreters when they are installed.

use std::path::{Path, PathBuf};
use std::process::Command;

use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

fn battery() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/integer_results.lua")
}

fn stdout_of(command: &mut Command) -> String {
    // `os.time` and `os.date` read the local zone, so pin it for both sides.
    let output = command
        .env("TZ", "UTC")
        .env_remove("LUA_INIT")
        .env_remove("LUA_INIT_5_4")
        .env_remove("LUA_INIT_5_5")
        .output()
        .expect("run Lua");
    assert!(
        output.status.success(),
        "{command:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("battery prints UTF-8")
}

fn assert_matches_reference(profile: &str, reference: &str) {
    if !Path::new(reference).exists() {
        eprintln!("skipping {profile}: {reference} is not installed");
        return;
    }
    let expected = stdout_of(Command::new(reference).arg(battery()));
    for mode in ["interpreter", "tracing"] {
        let actual = stdout_of(
            Command::new(env!("CARGO_BIN_EXE_oximond"))
                .arg(format!("--compat={profile}"))
                .arg(format!("--execution={mode}"))
                .arg(battery()),
        );
        let mismatches: Vec<String> = expected
            .lines()
            .zip(actual.lines())
            .filter(|(expected, actual)| expected != actual)
            .map(|(expected, actual)| format!("  {reference}: {expected}\n  oximond:  {actual}"))
            .collect();
        assert!(
            mismatches.is_empty() && expected.lines().count() == actual.lines().count(),
            "{profile} {mode} differs from {reference}:\n{}",
            mismatches.join("\n")
        );
    }
}

#[test]
fn lua54_results_match_stock_lua54() {
    assert_matches_reference("lua54", "/usr/bin/lua5.4");
}

#[test]
fn lua55_results_match_stock_lua55() {
    assert_matches_reference("lua55", "/usr/bin/lua5.5");
}

/// One call per fixed builtin. Every result is integral, so profiles without
/// tagged integers must return floats and the others integers.
const INTEGRAL_RESULTS: &str = r##"
    local t = {}
    for index in ipairs({ "a" }) do t[#t + 1] = index end
    for position in string.gmatch("ab", "()b") do t[#t + 1] = position end
    return math.floor(3.7), math.ceil(3.2), math.max(1.5, 2), math.min(2, 2.5),
        math.random(4, 4), (math.modf(3.7)), select("#", 1, 2), string.len("abc"),
        string.byte("a"), (string.find("abc", "b")), select(2, string.gsub("abc", "b", "x")),
        string.match("abc", "()c"),
        os.time({ year = 2020, month = 1, day = 1, hour = 0 }), os.date("!*t", 0).year,
        tonumber("ff", 16), t[1], t[2]
"##;

#[test]
fn integral_builtin_results_follow_the_profile_subtype() {
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua52,
        CompatibilityProfile::LuaJit21,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let tagged = matches!(
            profile,
            CompatibilityProfile::Lua53 | CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
        );
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            let results = lua
                .exec(INTEGRAL_RESULTS)
                .unwrap_or_else(|error| panic!("{profile} {mode}: {error:?}"));
            assert_eq!(results.len(), 17, "{profile} {mode}");
            for (index, result) in results.iter().enumerate() {
                assert!(
                    if tagged {
                        matches!(result, Value::Integer(_))
                    } else {
                        matches!(result, Value::Number(_))
                    },
                    "{profile} {mode}: result {} is {result:?}",
                    index + 1
                );
            }
        }
    }
}

/// Prelude giving every profile a `kind` that reports the subtype.
const KIND: &str = "local kind = math.type or function() return 'number' end\n";

/// Programs whose callees the interpreter's inline numeric plans could take.
/// Each must print what stock Lua prints, including the integer subtype.
const INLINE_CALL_PROGRAMS: &[(&str, &str)] = &[
    (
        "zero-argument integer constant",
        "local function f() return 9 end local b = f() print(b, kind(b))",
    ),
    (
        "ignored float argument",
        "local function f(x) return 1 end local b = f(1.5) print(b, kind(b))",
    ),
    (
        "several arguments",
        "local function f(a, b) return 3 end local b = f(1.5, 2.5) print(b, kind(b))",
    ),
    (
        "float arithmetic control",
        "local function f(x) return x + 1 end local b = f(1.5) print(b, kind(b))",
    ),
    (
        "float constant",
        "local function f() return 0.5 end local b = f() print(b, kind(b))",
    ),
    (
        "vararg forwarder",
        "local function f() return 9 end local function w(...) return f(...) end \
         local b = w() print(b, kind(b))",
    ),
    (
        "hot loop",
        "local function f(x) return 7 end local s = 0 \
         for i = 1, 3000 do s = f(i + 0.5) end print(s, kind(s))",
    ),
    (
        "hot loop float arithmetic",
        "local function f(x) return x * 2 + 1 end local s = 0 \
         for i = 1, 3000 do s = f(i + 0.5) end print(s, kind(s))",
    ),
    (
        "coroutine",
        "local function f() return 9 end \
         local co = coroutine.wrap(function() coroutine.yield(f()) end) \
         local b = co() print(b, kind(b))",
    ),
    (
        "table result",
        "local function g() return {9} end local t = g() print(t[1], kind(t[1]))",
    ),
];

/// Programs that need integer syntax, so only the integer profiles run them.
const INTEGER_SYNTAX_PROGRAMS: &[(&str, &str)] = &[
    (
        "integer multiplication",
        "local function f() return 2 * 3 end local b = f() print(b, kind(b))",
    ),
    (
        "floor division",
        "local function f() return 7 // 2 end local b = f() print(b, kind(b))",
    ),
    (
        "integer addition to a float argument",
        "local function f(x) return 1 + 2 end local b = f(1.5) print(b, kind(b))",
    ),
];

fn run_source(command: &mut Command, source: &str, tag: &str) -> String {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("inline_int_{tag}.lua"));
    std::fs::write(&path, source).expect("write program");
    command.arg(&path);
    let output = command.output().expect("run Lua");
    assert!(
        output.status.success(),
        "{command:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("UTF-8 output")
}

fn check_inline_programs(profile: &str, reference: &str, programs: &[(&str, &str)]) {
    if !Path::new(reference).exists() {
        eprintln!("skipping {profile}: {reference} is not installed");
        return;
    }
    for (index, (name, body)) in programs.iter().enumerate() {
        let source = format!("{KIND}{body}\n");
        let tag = format!("{profile}_{index}");
        let expected = run_source(&mut Command::new(reference), &source, &format!("ref_{tag}"));
        for mode in ["interpreter", "tracing"] {
            let mut command = Command::new(env!("CARGO_BIN_EXE_oximond"));
            command
                .arg(format!("--compat={profile}"))
                .arg(format!("--execution={mode}"));
            let actual = run_source(&mut command, &source, &format!("{mode}_{tag}"));
            assert_eq!(
                actual, expected,
                "{name}: {profile} {mode} differs from {reference}"
            );
        }
    }
}

#[test]
fn inline_call_results_keep_their_subtype_lua54() {
    check_inline_programs("lua54", "/usr/bin/lua5.4", INLINE_CALL_PROGRAMS);
    check_inline_programs("lua54", "/usr/bin/lua5.4", INTEGER_SYNTAX_PROGRAMS);
}

#[test]
fn inline_call_results_keep_their_subtype_lua55() {
    check_inline_programs("lua55", "/usr/bin/lua5.5", INLINE_CALL_PROGRAMS);
    check_inline_programs("lua55", "/usr/bin/lua5.5", INTEGER_SYNTAX_PROGRAMS);
}

#[test]
fn inline_call_results_are_unchanged_under_lua51() {
    check_inline_programs("lua51", "/usr/bin/lua5.1", INLINE_CALL_PROGRAMS);
}

#[test]
fn inline_call_results_are_unchanged_under_luajit() {
    check_inline_programs("luajit21", "/usr/bin/luajit", INLINE_CALL_PROGRAMS);
}
