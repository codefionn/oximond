//! Upvalue writes through `pcall` and `xpcall`, and `error` levels, checked
//! against output captured from stock Lua 5.1, 5.4, 5.5 and `LuaJIT`.
//!
//! The fixture writes captured locals from protected functions and from Lua
//! message handlers, raises string, number and table errors at levels 0 to 3
//! (called directly, through `pcall` and through `xpcall`), and ends in a hot
//! loop so tracing mode runs it compiled. 5.5 and 5.4 print the same lines, as
//! do 5.1 and `LuaJIT`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn check(profile: &str, expected: &str) {
    let expected = std::fs::read_to_string(fixture(expected)).unwrap();
    for mode in ["interpreter", "tracing"] {
        let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .arg(format!("--compat={profile}"))
            .arg(format!("--execution={mode}"))
            .arg(fixture("protected_errors.lua"))
            .output()
            .expect("run oximond");
        assert!(
            output.status.success(),
            "{profile} {mode}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            expected,
            "{profile} {mode}"
        );
    }
}

#[test]
fn lua51_matches_stock() {
    check("lua51", "protected_errors.lua51.stdout");
}

#[test]
fn luajit21_matches_stock() {
    check("luajit21", "protected_errors.luajit.stdout");
}

#[test]
fn lua54_matches_stock() {
    check("lua54", "protected_errors.lua54.stdout");
}

#[test]
fn lua55_matches_stock() {
    check("lua55", "protected_errors.lua54.stdout");
}
