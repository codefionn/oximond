//! Yield across metamethods, `__close` handlers and iterators. The fixtures
//! in `fixtures/` run under the interpreter and tracing execution, and their
//! `.stdout` files were captured from the stock interpreters (`lua5.1`,
//! `lua5.4`, `lua5.5`, `luajit`).

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
        .output()
        .expect("run oximond");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{profile}/{mode}/{script}: {output:?}"
    );
    String::from_utf8(output.stdout).expect("utf-8 output")
}

fn check(script: &str, expected_stem: &str, profiles: &[&str]) {
    for profile in profiles {
        let expected =
            std::fs::read_to_string(fixture(&format!("{expected_stem}.{profile}.stdout")))
                .expect("read expected output");
        for mode in ["interpreter", "tracing"] {
            assert_eq!(
                run_file(profile, mode, script),
                expected,
                "{profile}/{mode}"
            );
        }
    }
}

#[test]
fn metamethods_iterators_and_hot_loops_match_stock() {
    check(
        "yield_meta.lua",
        "yield_meta",
        &["lua54", "lua55", "luajit21"],
    );
}

#[test]
fn close_handlers_bitwise_pairs_and_refusals_match_stock() {
    check(
        "yield_meta_close.lua",
        "yield_meta_close",
        &["lua54", "lua55"],
    );
}

#[test]
fn luajit_tostring_yields_and_other_c_calls_refuse() {
    check("yield_meta_luajit.lua", "yield_meta_luajit", &["luajit21"]);
}

#[test]
fn luajit_positions_only_non_tail_yield_refusals() {
    check(
        "yield_meta_luajit_tail.lua",
        "yield_meta_luajit_tail",
        &["luajit21"],
    );
}

#[test]
fn lua51_refuses_every_metamethod_with_the_stock_message() {
    check("yield_meta_lua51.lua", "yield_meta_lua51", &["lua51"]);
}

#[test]
fn refusal_messages_follow_the_profile() {
    let source = r"local co = coroutine.create(function()
        table.sort({3, 2, 1}, function(a, b) local r = coroutine.yield(1) return a < b end)
      end)
      print(coroutine.resume(co))";
    for (profile, expected) in [
        (
            "lua51",
            "false\tattempt to yield across metamethod/C-call boundary\n",
        ),
        (
            "lua54",
            "false\tattempt to yield across a C-call boundary\n",
        ),
        (
            "lua55",
            "false\tattempt to yield across a C-call boundary\n",
        ),
    ] {
        for mode in ["interpreter", "tracing"] {
            let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
                .args(["--compat", profile, "--execution", mode, "-e", source])
                .output()
                .expect("run oximond");
            assert_eq!(
                output.status.code(),
                Some(0),
                "{profile}/{mode}: {output:?}"
            );
            assert_eq!(
                String::from_utf8(output.stdout).expect("utf-8 output"),
                expected,
                "{profile}/{mode}"
            );
        }
    }
}
