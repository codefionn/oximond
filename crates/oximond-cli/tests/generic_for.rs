//! `ForInCall` and `ForInLoop` run a generic `for` in every profile and both
//! execution modes.
//!
//! `generic_for.lua` and `generic_for_hot.lua` cover ipairs, pairs, next, Lua
//! closure and stateless iterators, a callable table, natives, iterator
//! errors, closures capturing the loop variables, assignment to loop
//! variables, `break`, `goto`, a first result of `false`, coroutines and
//! yields inside iterators, and loops hot enough to be traced. Their
//! `.stdout` files were captured from the stock interpreters (`lua5.1`,
//! `lua5.4`, `lua5.5`, `luajit`). `generic_for_hooks.lua` checks hook events,
//! call names and traceback text; its expected output is what the loop gave
//! before the two opcodes existed.

use std::path::PathBuf;
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn run_file(profile: &str, mode: &str, script: &str) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args(["--compat", profile, "--execution", mode])
        .arg(script)
        .current_dir(fixture(""))
        // Synchronous compilation makes a hot loop run its trace in this run.
        .env("OXIMOND_JIT_SYNC", "1")
        .output()
        .expect("run oximond");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{profile}/{mode}/{script}: {output:?}"
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn check(stem: &str, profiles: &[&str]) {
    for profile in profiles {
        let expected = std::fs::read_to_string(fixture(&format!("{stem}.{profile}.stdout")))
            .expect("read expected output");
        for mode in ["interpreter", "tracing"] {
            let actual = run_file(profile, mode, &format!("{stem}.lua"));
            assert_eq!(actual, expected, "{profile}/{mode}/{stem}");
        }
    }
}

#[test]
fn generic_for_matches_stock_lua() {
    check("generic_for", &["lua51", "lua54", "lua55", "luajit21"]);
}

#[test]
fn hot_generic_for_loops_match_stock_lua_when_traced() {
    check("generic_for_hot", &["lua51", "lua54", "lua55", "luajit21"]);
}

#[test]
fn generic_for_hooks_and_names_are_unchanged() {
    check(
        "generic_for_hooks",
        &["lua51", "lua52", "lua53", "lua54", "lua55", "luajit21"],
    );
}

#[test]
fn lua52_and_lua53_agree_across_execution_modes() {
    for stem in ["generic_for", "generic_for_hot"] {
        for profile in ["lua52", "lua53"] {
            let interpreter = run_file(profile, "interpreter", &format!("{stem}.lua"));
            let tracing = run_file(profile, "tracing", &format!("{stem}.lua"));
            assert_eq!(interpreter, tracing, "{profile}/{stem}");
        }
    }
}

#[test]
fn multiline_generic_for_header_matches_stock_lua() {
    check(
        "generic_for_multiline",
        &["lua51", "lua54", "lua55", "luajit21"],
    );
}
