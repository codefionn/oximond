//! Scalar natives store their result straight into the destination register.
//!
//! `fixtures/native_results_bit.lua` runs the bit library over numbers, 64-bit
//! integer cdata and mixes of both. `fixtures/native_results_bit.expected` is
//! its output under `LuaJIT` 2.1, which is also diffed live when `luajit` is
//! installed. `fixtures/native_results_errors.lua` fails scalar natives in
//! every call shape the interpreter runs in place.

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

fn oximond(profile: &str, mode: &str, script: &str) -> String {
    stdout_of(
        Command::new(env!("CARGO_BIN_EXE_oximond"))
            .arg(format!("--compat={profile}"))
            .arg(format!("--execution={mode}"))
            .arg(fixture(script)),
    )
}

fn assert_lines_equal(reference: &str, expected: &str, actual: &str, context: &str) {
    let mismatches: Vec<String> = expected
        .lines()
        .zip(actual.lines())
        .filter(|(expected, actual)| expected != actual)
        .take(10)
        .map(|(expected, actual)| format!("  {reference}: {expected}\n  oximond: {actual}"))
        .collect();
    assert!(
        mismatches.is_empty() && expected.lines().count() == actual.lines().count(),
        "{context} differs from {reference}:\n{}",
        mismatches.join("\n")
    );
}

#[test]
fn bit_results_match_luajit() {
    let expected = std::fs::read_to_string(fixture("native_results_bit.expected")).unwrap();
    let live = Path::new("/usr/bin/luajit")
        .exists()
        .then(|| stdout_of(Command::new("/usr/bin/luajit").arg(fixture("native_results_bit.lua"))));
    for mode in ["interpreter", "tracing"] {
        let actual = oximond("luajit21", mode, "native_results_bit.lua");
        assert_lines_equal("recorded LuaJIT", &expected, &actual, mode);
        if let Some(live) = &live {
            assert_lines_equal("/usr/bin/luajit", live, &actual, mode);
        }
    }
}

#[test]
fn failing_scalar_natives_keep_their_errors() {
    for profile in ["lua51", "luajit21"] {
        let expected = std::fs::read_to_string(fixture(&format!(
            "native_results_errors.{profile}.expected"
        )))
        .unwrap();
        // Both files are the stock output: `lua5.1` and `luajit` print them
        // line for line.
        let stock = if profile == "lua51" {
            "/usr/bin/lua5.1"
        } else {
            "/usr/bin/luajit"
        };
        let live = Path::new(stock)
            .exists()
            .then(|| stdout_of(Command::new(stock).arg(fixture("native_results_errors.lua"))));
        for mode in ["interpreter", "tracing"] {
            let actual = oximond(profile, mode, "native_results_errors.lua");
            assert_lines_equal(
                "recorded stock output",
                &expected,
                &actual,
                &format!("{profile} {mode}"),
            );
            if let Some(live) = &live {
                assert_lines_equal(stock, live, &actual, mode);
            }
        }
    }
}
