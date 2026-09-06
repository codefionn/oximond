//! Float `%` and `^` against the output stock Lua printed.
//!
//! `fixtures/float_arith.lua` prints 17 digits of edge cases (infinities,
//! NaN, signed zero, a divisor whose rounding shows) and of a loop that gets
//! hot, so under tracing execution the operations run in native code. The
//! `.expected` files are the output of `lua5.4`, `lua5.5` and `luajit` for it.
//! `float_arith.lua51.expected` is the `luajit` output (identical to what an
//! unfused Lua 5.1 prints): the `lua5.1` of the reference machine is built
//! with FP contraction and fuses the multiply and subtract of
//! `a - floor(a / b) * b`, a build artefact oximond does not follow.
//!
//! The profiles differ on purpose. Lua 5.1, 5.2 and `LuaJIT` round the
//! product, Lua 5.3 and later use `fmod`, and 5.4 and 5.5 compute `a ^ 2` as
//! `a * a`.

use std::path::Path;
use std::process::Command;

fn run(profile: &str, mode: &str) -> String {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .arg(format!("--compat={profile}"))
        .arg(format!("--execution={mode}"))
        .arg(fixtures.join("float_arith.lua"))
        .env("OXIMOND_JIT_SYNC", "1")
        .output()
        .expect("run oximond");
    assert!(
        output.status.success(),
        "{profile} {mode}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("UTF-8 output")
}

fn check(profile: &str) {
    let expected = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("tests/fixtures/float_arith.{profile}.expected")),
    )
    .expect("expected output");
    for mode in ["interpreter", "tracing"] {
        let actual = run(profile, mode);
        let differences: Vec<String> = expected
            .lines()
            .zip(actual.lines())
            .filter(|(expected, actual)| expected != actual)
            .map(|(expected, actual)| format!("  stock:   {expected}\n  oximond: {actual}"))
            .collect();
        assert!(
            differences.is_empty() && expected.lines().count() == actual.lines().count(),
            "{profile} {mode} differs from stock:\n{}",
            differences.join("\n")
        );
    }
}

#[test]
fn lua51_float_modulo_is_the_unfused_floor_formula() {
    check("lua51");
}

#[test]
fn lua54_float_modulo_is_fmod_and_squares_are_products() {
    check("lua54");
}

#[test]
fn lua55_float_modulo_is_fmod_and_squares_are_products() {
    check("lua55");
}

#[test]
fn luajit_float_modulo_rounds_the_product_and_power_is_pow() {
    check("luajit21");
}

/// The hot loop really runs natively, or the tracing check above would only
/// repeat the interpreter.
#[test]
fn the_float_loop_runs_in_native_code() {
    for profile in ["lua51", "lua54", "luajit21"] {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .arg(format!("--compat={profile}"))
            .arg("--execution=tracing")
            .arg(fixtures.join("float_arith.lua"))
            .env("OXIMOND_JIT_SYNC", "1")
            .env("OXIMOND_JIT_STATS", "1")
            .output()
            .expect("run oximond");
        let stderr = String::from_utf8_lossy(&output.stderr);
        let line = stderr
            .lines()
            .find(|line| line.starts_with("trace:"))
            .unwrap_or_else(|| panic!("{profile}: no trace line: {stderr}"));
        assert!(!line.contains("traces_compiled: 0"), "{profile}: {line}");
        assert!(!line.contains("native_entries: 0"), "{profile}: {line}");
    }
}
