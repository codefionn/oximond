//! Loop bookkeeping follows the source lines emitted by each stock compiler.

use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn trace(source: &str, profile: CompatibilityProfile, mode: ExecutionMode) -> String {
    trace_target(source, false, profile, mode)
}

fn trace_target(
    source: &str,
    nested: bool,
    profile: CompatibilityProfile,
    mode: ExecutionMode,
) -> String {
    let source = format!(
        r#"local target = assert((loadstring or load)([==[{source}]==], "@line-target"))
if {nested} then target = target() end
local lines = {{}}
debug.sethook(function(_, line)
  local info = debug.getinfo(2, "S")
  if info.source == "@line-target" then lines[#lines + 1] = line end
end, "l")
target()
debug.sethook()
return table.concat(lines, ",")"#
    );
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    let results = lua
        .exec(&source)
        .unwrap_or_else(|error| panic!("{profile:?}/{mode:?}: {error}"));
    let [Value::String(lines)] = results.as_slice() else {
        panic!("{profile:?}/{mode:?}: unexpected results {results:?}");
    };
    String::from_utf8(lines.as_bytes().to_vec()).unwrap()
}

#[test]
fn generic_for_exhaustion_reports_end_only_in_lua54_and_lua55() {
    let source = "local total = 0\nfor _, value in ipairs({1}) do\n  total = total + value\nend\nreturn total";
    for profile in PROFILES {
        let expected = if matches!(
            profile,
            CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
        ) {
            "1,2,3,2,4,5"
        } else {
            "1,2,3,2,5"
        };
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            assert_eq!(
                trace(source, profile, mode),
                expected,
                "{profile:?}/{mode:?}"
            );
        }
    }
}

#[test]
fn repeat_reports_the_header_on_every_luajit_iteration() {
    let source = "local index = 0\nrepeat\n  index = index + 1\nuntil index >= 2\nreturn index";
    for profile in PROFILES {
        let expected = if profile == CompatibilityProfile::LuaJit21 {
            "1,2,3,4,2,3,4,5"
        } else {
            "1,3,4,3,4,5"
        };
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            assert_eq!(
                trace(source, profile, mode),
                expected,
                "{profile:?}/{mode:?}"
            );
        }
    }
}

#[test]
fn empty_generic_for_does_not_report_a_body_step() {
    let source = "local total = 0\nfor _, value in ipairs({}) do\n  total = total + value\nend\nreturn total";
    for profile in PROFILES {
        let expected = if matches!(
            profile,
            CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
        ) {
            "1,2,4,5"
        } else {
            "1,2,5"
        };
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            assert_eq!(
                trace(source, profile, mode),
                expected,
                "{profile:?}/{mode:?}"
            );
        }
    }
}

#[test]
fn nested_functions_keep_the_profile_loop_lines() {
    let cases = [
        (
            "return function()\nlocal total = 0\nfor _, value in ipairs({1}) do\n  total = total + value\nend\nreturn total\nend",
            "2,3,4,3,6",
            "2,3,4,3,5,6",
            "2,3,4,3,6",
        ),
        (
            "return function()\nlocal index = 0\nrepeat\n  index = index + 1\nuntil index >= 2\nreturn index\nend",
            "2,4,5,4,5,6",
            "2,4,5,4,5,6",
            "2,3,4,5,3,4,5,6",
        ),
    ];
    for (source, legacy, modern, luajit) in cases {
        for profile in PROFILES {
            let expected = match profile {
                CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55 => modern,
                CompatibilityProfile::LuaJit21 => luajit,
                _ => legacy,
            };
            for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
                assert_eq!(
                    trace_target(source, true, profile, mode),
                    expected,
                    "{profile:?}/{mode:?}"
                );
            }
        }
    }
}

#[test]
fn a_value_list_over_several_lines_reports_each_values_line() {
    // Stock Lua 5.1, 5.4 and LuaJIT report these sequences. Each value runs
    // on its own line, and what follows the list (nil fills, moves) on the
    // statement's last line.
    let cases = [
        (
            "local function g(x) return x end\nlocal a,\n b = 1,\n g(2)\nreturn a + b",
            "1,3,4,1,5",
        ),
        ("local a =\n 1\nreturn a", "2,3"),
        ("local a, b =\n 1, 2\nreturn a", "2,3"),
        ("local a, b = 1,\n 2\nreturn a", "1,2,3"),
        ("local a = 1,\n 2\nreturn a", "1,2,3"),
        ("local a, b, c = 1,\n 2\nreturn a", "1,2,3"),
    ];
    for (source, expected) in cases {
        for profile in PROFILES {
            for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
                assert_eq!(
                    trace(source, profile, mode),
                    expected,
                    "{source:?} {profile:?}/{mode:?}"
                );
            }
        }
    }
}
