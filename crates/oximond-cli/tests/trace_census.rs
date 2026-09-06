//! `OXIMOND_JIT_CENSUS` prints one `jit-census:` JSON line per native compile
//! on stderr, and prints nothing without it. `benchmarks/trace_census.py`
//! parses the lines.

use std::process::Command;

fn run(census: bool) -> String {
    let source = "local s = 0 for i = 1, 3000000 do s = s + i * 0.25 end print(s)";
    let mut command = Command::new(env!("CARGO_BIN_EXE_oximond"));
    command
        .args(["--execution=tracing", "-e", source])
        .env("OXIMOND_JIT_SYNC", "1")
        // The census counts Cranelift compiles, and MIR would take this loop
        // and keep it.
        .env("OXIMOND_JIT_BACKEND", "cranelift")
        .env_remove("OXIMOND_JIT_CENSUS");
    if census {
        command.env("OXIMOND_JIT_CENSUS", "1");
    }
    let output = command.output().expect("run oximond");
    assert!(output.status.success(), "{output:?}");
    String::from_utf8(output.stderr).unwrap()
}

#[test]
fn census_lines_appear_only_when_asked_for() {
    assert!(!run(false).contains("jit-census:"));
    let text = run(true);
    let line = text
        .lines()
        .find(|line| line.starts_with("jit-census: {"))
        .unwrap_or_else(|| panic!("no census line in {text:?}"));
    for key in [
        "\"root\":",
        "\"tier\":",
        "\"clif_total\":",
        "\"clif_ops\":",
        "\"members\":",
        "\"guards\":",
    ] {
        assert!(line.contains(key), "{key} missing in {line}");
    }
    assert!(line.contains("\"Arithmetic."), "{line}");
}
