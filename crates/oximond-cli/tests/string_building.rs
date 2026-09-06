//! Single-pass string building: `..`, `table.concat`, `string.format` and
//! plain `string.find` write their results once, and format numbers straight
//! into the output. The `.stdout` fixtures were captured from the stock
//! interpreters (`lua5.1` for 5.1 and 5.2, `lua5.4` for 5.3 and 5.4, `lua5.5`,
//! `luajit`) and the scripts run under the interpreter and tracing modes.

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

/// Integers, integral floats, `-0.0`, NaN, infinities, large and small
/// magnitudes and the Lua 5.3+ float suffix, through `..`, `table.concat`
/// and `string.format("%s")`, in results that fit the stack buffer and in
/// results that do not.
#[test]
fn numbers_in_concatenation_table_concat_and_format_match_stock() {
    check(
        "string_building_numbers",
        &["lua51", "lua52", "lua53", "lua54", "lua55", "luajit21"],
    );
}

#[test]
fn errors_and_edge_cases_match_stock() {
    check(
        "string_building_errors",
        &["lua51", "lua53", "lua54", "lua55", "luajit21"],
    );
}
