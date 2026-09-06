//! Builtins as metamethods, yields across library C calls, the `ipairs`
//! iterator through `pcall`, generic-for iterators that return `false`, Lua
//! 5.1 `print`, `string.gmatch` carets and empty matches. The `.stdout`
//! files were captured from the stock interpreters (`lua5.1`, `lua5.4`,
//! `lua5.5`, `luajit`) and the fixtures run under the interpreter and tracing
//! execution.

use std::fmt::Write;
use std::path::PathBuf;
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn run_file(profile: &str, mode: &str, script: &str) -> Vec<u8> {
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
    output.stdout
}

fn check(stem: &str, profiles: &[&str]) {
    for profile in profiles {
        let expected = std::fs::read(fixture(&format!("{stem}.{profile}.stdout")))
            .expect("read expected output");
        for mode in ["interpreter", "tracing"] {
            let actual = run_file(profile, mode, &format!("{stem}.lua"));
            assert_eq!(
                String::from_utf8_lossy(&actual),
                String::from_utf8_lossy(&expected),
                "{profile}/{mode}/{stem}"
            );
        }
    }
}

#[test]
fn builtins_as_index_and_newindex_run_and_yield() {
    check(
        "close_gaps_builtin_meta",
        &["lua51", "lua54", "lua55", "luajit21"],
    );
}

#[test]
fn builtins_as_close_handlers_run_and_yield() {
    check("close_gaps_builtin_close", &["lua54", "lua55"]);
}

#[test]
fn library_c_calls_refuse_yields_from_their_metamethods() {
    check("close_gaps_c_boundary", &["lua54", "lua55"]);
}

#[test]
fn the_ipairs_iterator_follows_index_when_called_through_pcall() {
    check("close_gaps_ipairs_call", &["lua54", "lua55"]);
}

#[test]
fn generic_for_ends_on_nil_only_and_lua51_refuses_iterator_yields() {
    check(
        "close_gaps_for_iterator",
        &["lua51", "lua54", "lua55", "luajit21"],
    );
}

#[test]
fn lua51_print_stops_at_nul_and_gmatch_treats_a_caret_literally() {
    check(
        "close_gaps_print_gmatch",
        &["lua51", "lua54", "lua55", "luajit21"],
    );
}

#[test]
fn lua53_and_later_gmatch_skip_an_empty_match_at_the_previous_end() {
    check(
        "close_gaps_gmatch_empty",
        &["lua51", "lua54", "lua55", "luajit21"],
    );
}

#[test]
fn twenty_five_nested_generic_for_loops_compile() {
    let mut source = String::from("local count = 0\n");
    for depth in 0..25 {
        writeln!(source, "for k{depth} in pairs({{1}}) do")
            .expect("formatting into a String cannot fail");
    }
    source.push_str("count = count + 1\n");
    source.push_str(&"end\n".repeat(25));
    source.push_str("print(count)\n");
    let path = std::env::temp_dir().join(format!("nested_for_{}.lua", std::process::id()));
    std::fs::write(&path, source).expect("write script");
    for profile in ["lua51", "lua54", "lua55", "luajit21"] {
        for mode in ["interpreter", "tracing"] {
            let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
                .args(["--compat", profile, "--execution", mode])
                .arg(&path)
                .output()
                .expect("run oximond");
            assert_eq!(
                output.status.code(),
                Some(0),
                "{profile}/{mode}: {output:?}"
            );
            assert_eq!(output.stdout, b"1\n", "{profile}/{mode}");
        }
    }
    let _ = std::fs::remove_file(&path);
}
