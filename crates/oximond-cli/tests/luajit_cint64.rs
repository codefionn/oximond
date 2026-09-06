//! `LuaJIT` 2.1's 64-bit integer cdata literals and arithmetic.
//!
//! `fixtures/luajit_cint64.lua` prints the type and text of each result, and
//! `fixtures/luajit_cint64.expected` is its output under `LuaJIT` 2.1. The
//! battery is also diffed against a live `luajit` when one is installed.

use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn stdout_of(command: &mut Command) -> String {
    let output = command.env_remove("LUA_INIT").output().expect("run Lua");
    assert!(
        output.status.success(),
        "{command:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("battery prints UTF-8")
}

fn assert_same(reference: &str, expected: &str, actual: &str, context: &str) {
    let mismatches: Vec<String> = expected
        .lines()
        .zip(actual.lines())
        .filter(|(expected, actual)| expected != actual)
        .map(|(expected, actual)| format!("  {reference}: {expected}\n  oximond: {actual}"))
        .collect();
    assert!(
        mismatches.is_empty() && expected.lines().count() == actual.lines().count(),
        "{context} differs from {reference}:\n{}",
        mismatches.join("\n")
    );
}

#[test]
fn cint64_battery_matches_luajit() {
    let expected = std::fs::read_to_string(fixture("luajit_cint64.expected")).unwrap();
    let live = Path::new("/usr/bin/luajit")
        .exists()
        .then(|| stdout_of(Command::new("/usr/bin/luajit").arg(fixture("luajit_cint64.lua"))));
    for mode in ["interpreter", "tracing"] {
        let actual = stdout_of(
            Command::new(env!("CARGO_BIN_EXE_oximond"))
                .arg("--compat=luajit21")
                .arg(format!("--execution={mode}"))
                .arg(fixture("luajit_cint64.lua")),
        );
        assert_same("recorded LuaJIT", &expected, &actual, mode);
        if let Some(live) = &live {
            assert_same("/usr/bin/luajit", live, &actual, mode);
        }
    }
}

#[test]
fn cint64_literals_need_the_luajit_profile() {
    for profile in ["lua51", "lua54"] {
        let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .arg(format!("--compat={profile}"))
            .args(["-e", "return 1ll"])
            .output()
            .expect("run oximond");
        assert!(!output.status.success(), "{profile} accepted `1ll`");
    }
}

/// Register-operand ULL/LL arithmetic, comparisons and bit operations over a
/// grid of operands. The expected output was recorded from `LuaJIT` 2.1 on x64,
/// whose bit library converts doubles beyond 2^51 by adding 2^52 + 2^51 and
/// keeping the low word, so any drift in values or types fails.
#[test]
fn cint64_operand_grid_is_unchanged() {
    let expected = std::fs::read_to_string(fixture("luajit_cint64_ops.expected")).unwrap();
    let live = Path::new("/usr/bin/luajit")
        .exists()
        .then(|| stdout_of(Command::new("/usr/bin/luajit").arg(fixture("luajit_cint64_ops.lua"))));
    for mode in ["interpreter", "tracing"] {
        let actual = stdout_of(
            Command::new(env!("CARGO_BIN_EXE_oximond"))
                .arg("--compat=luajit21")
                .arg(format!("--execution={mode}"))
                .arg(fixture("luajit_cint64_ops.lua")),
        );
        assert_same("recorded LuaJIT", &expected, &actual, mode);
        if let Some(live) = &live {
            assert_same("/usr/bin/luajit", live, &actual, mode);
        }
    }
}
