//! Objects built with `setmetatable({...}, Class)` and then stored into, with
//! `__newindex` absent, added later and removed again. The `.stdout` files in
//! `fixtures/` were captured from `lua5.1` and `lua5.4`. The fixture sorts
//! every key list, so the output does not depend on iteration order.

use std::path::PathBuf;
use std::process::Command;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn run(profile: &str, mode: &str) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args(["--compat", profile, "--execution", mode])
        .arg("setmetatable_objects.lua")
        .current_dir(fixtures())
        .output()
        .expect("run oximond");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{profile}/{mode}: {output:?}"
    );
    String::from_utf8(output.stdout).expect("utf-8 output")
}

#[test]
fn setmetatable_objects_match_stock() {
    for profile in ["lua51", "lua54"] {
        let expected = std::fs::read_to_string(
            fixtures().join(format!("setmetatable_objects.{profile}.stdout")),
        )
        .expect("read expected output");
        for mode in ["interpreter", "tracing"] {
            assert_eq!(run(profile, mode), expected, "{profile}/{mode}");
        }
    }
}
