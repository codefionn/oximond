//! Bad-argument messages, `LuaJIT` string indices and `x - 0` signs against
//! the output stock Lua printed.
//!
//! The `.expected` files in `fixtures/` are the output of `lua5.1`,
//! `lua5.4`, `lua5.5` and `luajit` for the matching `.lua` fixture, and both
//! execution modes must print them.

use std::path::Path;
use std::process::Command;

fn check(fixture: &str, profile: &str, expected_suffix: &str) {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let expected =
        std::fs::read_to_string(fixtures.join(format!("{fixture}.{expected_suffix}.expected")))
            .expect("expected output");
    for mode in ["interpreter", "tracing"] {
        let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .arg(format!("--compat={profile}"))
            .arg(format!("--execution={mode}"))
            .arg(format!("{fixture}.lua"))
            .current_dir(&fixtures)
            .env("OXIMOND_JIT_SYNC", "1")
            .output()
            .expect("run oximond");
        assert!(
            output.status.success(),
            "{profile} {mode}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let actual = String::from_utf8(output.stdout).expect("UTF-8 output");
        let differences: Vec<String> = expected
            .lines()
            .zip(actual.lines())
            .filter(|(expected, actual)| expected != actual)
            .map(|(expected, actual)| format!("  stock:   {expected}\n  oximond: {actual}"))
            .collect();
        assert!(
            differences.is_empty() && expected.lines().count() == actual.lines().count(),
            "{fixture} {profile} {mode} differs from stock:\n{}",
            differences.join("\n")
        );
    }
}

#[test]
fn luajit_names_the_function_and_locates_all_but_tail_calls() {
    check("argument_errors", "luajit21", "luajit21");
}

#[test]
fn lua51_argument_errors_are_unchanged() {
    check("argument_errors", "lua51", "lua51");
}

#[test]
fn luajit_string_indices_are_32_bit() {
    check("luajit_indices", "luajit21", "luajit21");
}

#[test]
fn lua54_subtracting_integer_zero_is_an_addition() {
    check("subtract_zero", "lua54", "lua54");
}

#[test]
fn lua55_subtracting_integer_zero_is_an_addition() {
    check("subtract_zero", "lua55", "lua55");
}
