//! `pcall` and `xpcall` call a native directly and the native calls back into
//! Lua or into another native that raises: `string.gsub`, `table.sort`,
//! `table.foreach`, `tostring`, `string.format` and `load`. The error must
//! come back as `false, message` instead of leaving the process, with the
//! message stock Lua 5.1, 5.4, 5.5 and `LuaJIT` print.
//!
//! The fixture runs from its own directory so chunk names match the stock
//! output. 5.4 and 5.5 print the same lines, as do 5.1 and `LuaJIT` apart from
//! `jit`-only differences the fixture skips.

use std::path::{Path, PathBuf};
use std::process::Command;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn check(profile: &str, expected: &str) {
    let expected = std::fs::read_to_string(fixtures().join(expected)).unwrap();
    for mode in ["interpreter", "tracing"] {
        let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .current_dir(fixtures())
            .arg(format!("--compat={profile}"))
            .arg(format!("--execution={mode}"))
            .arg("protected_natives.lua")
            .output()
            .expect("run oximond");
        assert!(
            output.status.success(),
            "{profile} {mode}: status {:?}, stderr {}",
            output.status.code(),
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
    check("lua51", "protected_natives.lua51.stdout");
}

#[test]
fn lua54_matches_stock() {
    check("lua54", "protected_natives.lua54.stdout");
}

#[test]
fn lua55_matches_stock() {
    check("lua55", "protected_natives.lua54.stdout");
}

#[test]
fn luajit21_matches_stock() {
    check("luajit21", "protected_natives.luajit.stdout");
}
