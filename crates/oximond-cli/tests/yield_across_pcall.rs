//! Yield across pcall and xpcall. The fixtures in `fixtures/` run under the
//! interpreter and tracing execution, and their `.stdout` files were captured
//! from the stock interpreters (`lua5.4`, `lua5.5`, `luajit`).

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

fn run_source(profile: &str, mode: &str, source: &str) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args(["--compat", profile, "--execution", mode, "-e", source])
        .output()
        .expect("run oximond");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{profile}/{mode}: {output:?}"
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

const ALL: [&str; 3] = ["lua54", "lua55", "luajit21"];

#[test]
fn yield_resume_error_handler_and_nesting_match_stock() {
    check("yield_across_pcall.lua", "yield_across_pcall", &ALL);
}

#[test]
fn isyieldable_and_hot_loops_match_stock() {
    check(
        "yield_across_pcall_extra.lua",
        "yield_across_pcall_extra",
        &ALL,
    );
}

#[test]
fn yield_inside_xpcall_handler_is_refused_like_stock() {
    check(
        "yield_across_pcall_handler.lua",
        "yield_across_pcall_handler",
        &ALL,
    );
}

#[test]
fn pcall_of_coroutine_yield_matches_stock() {
    check(
        "yield_across_pcall_direct.lua",
        "yield_across_pcall_direct",
        &ALL,
    );
}

#[test]
fn close_while_suspended_inside_pcall_matches_stock() {
    check(
        "yield_across_pcall_close.lua",
        "yield_across_pcall_close",
        &["lua54", "lua55"],
    );
}

#[test]
fn lua51_still_refuses_to_yield_across_pcall() {
    for mode in ["interpreter", "tracing"] {
        let out = run_source(
            "lua51",
            mode,
            r"local co = coroutine.create(function()
                 return pcall(function() return coroutine.yield(1) end)
               end)
               print(coroutine.resume(co))
               print(pcall(coroutine.wrap(function()
                 return xpcall(function() coroutine.yield(1) end, function(m) return m end)
               end)))
               local c2 = coroutine.create(function() return pcall(coroutine.yield, 1) end)
               print(coroutine.resume(c2))",
        );
        let lines: Vec<&str> = out.lines().collect();
        assert!(
            lines[0].starts_with("true\tfalse\t")
                && lines[0].ends_with("attempt to yield across metamethod/C-call boundary"),
            "{lines:?}"
        );
        assert!(
            lines[1].ends_with("attempt to yield across metamethod/C-call boundary"),
            "{lines:?}"
        );
        assert_eq!(
            lines[2],
            "true\tfalse\tattempt to yield across metamethod/C-call boundary"
        );
    }
}

#[test]
fn pcall_of_yield_outside_a_coroutine_reports_the_stock_message() {
    for profile in ["lua54", "lua55"] {
        for mode in ["interpreter", "tracing"] {
            assert_eq!(
                run_source(profile, mode, "print(pcall(coroutine.yield, 1))"),
                "false\tattempt to yield from outside a coroutine\n"
            );
        }
    }
}
