//! Library behaviour that used to differ from stock Lua: `load` readers,
//! builtins as index metamethods, the `table` functions and `ipairs` with
//! `__index`/`__newindex`/`__len` (Lua 5.3 and later), replacement-string and
//! pattern error messages, `table.foreach`, `loadfile` errors, and
//! `__tostring` through `tostring`, `print` and `string.format`.
//!
//! The `.stdout` files in `fixtures/` were captured from `lua5.1`, `lua5.4`,
//! `lua5.5` and `luajit` running the same script from that directory. Each
//! runs under the interpreter and tracing execution.

use std::path::PathBuf;
use std::process::Command;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn run_file(profile: &str, mode: &str, script: &str) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args(["--compat", profile, "--execution", mode])
        .arg(script)
        .current_dir(fixtures())
        .output()
        .expect("run oximond");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{profile}/{mode}/{script}: {output:?}"
    );
    String::from_utf8(output.stdout).expect("utf-8 output")
}

fn check(name: &str, profiles: &[&str]) {
    let script = format!("stdlib_gaps_{name}.lua");
    for profile in profiles {
        let expected = std::fs::read_to_string(
            fixtures().join(format!("stdlib_gaps_{name}.{profile}.stdout")),
        )
        .expect("read expected output");
        for mode in ["interpreter", "tracing"] {
            assert_eq!(
                run_file(profile, mode, &script),
                expected,
                "{profile}/{mode}/{script}"
            );
        }
    }
}

const ALL: [&str; 4] = ["lua51", "lua54", "lua55", "luajit21"];

#[test]
fn table_functions_follow_metamethods_where_the_profile_does() {
    check("table_metamethods", &ALL);
}

#[test]
fn load_readers_run_through_xpcall_and_name_native_readers() {
    check("load_reader", &ALL);
}

#[test]
fn builtins_work_as_index_metamethods() {
    check("index_builtin", &ALL);
}

#[test]
fn gsub_replacement_errors_match_stock() {
    check("gsub_capture", &ALL);
}

#[test]
fn pattern_errors_read_as_stock_does() {
    check("pattern_errors", &ALL);
}

#[test]
fn loadfile_and_dofile_errors_have_no_os_error_suffix() {
    check("loadfile_errors", &ALL);
}

#[test]
fn table_foreach_returns_nothing_unless_the_callback_does() {
    check("table_foreach", &["lua51", "luajit21"]);
}

#[test]
fn tostring_and_print_call_and_check_tostring() {
    check("tostring_metamethod", &ALL);
}

#[test]
fn string_format_calls_tostring() {
    check("format_tostring", &["lua54", "lua55", "luajit21"]);
}

#[test]
fn declining_natives_work_as_metamethod_handlers() {
    check("native_handlers", &ALL);
}

#[test]
fn recursion_through_table_functions_and_print_is_a_catchable_error() {
    check("c_stack_overflow", &ALL);
}

#[test]
fn builtins_work_as_tostring_metamethods() {
    check("tostring_builtin", &ALL);
}

#[test]
fn lua51_load_takes_a_reader_and_loadstring_a_string() {
    check("load_arguments", &ALL);
}
