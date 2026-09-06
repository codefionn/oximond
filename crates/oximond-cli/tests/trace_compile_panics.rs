//! A trace compile that panics rejects that trace only, with or without
//! `OXIMOND_JIT_SYNC`. `OXIMOND_JIT_INJECT_COMPILE_PANICS` makes chosen
//! compiles panic; `OXIMOND_JIT_TRACE` reports the panic and
//! `OXIMOND_JIT_STATS` counts it, and the loop after it still runs natively.

use std::process::Command;

const SOURCE: &str = "
local function sum(n) local s = 0 for i = 1, n do s = s + i end return s end
print(sum(5000))
local t = 0
for i = 1, 3000000 do t = t + i * 0.25 end
print(t)
";

fn run(execution: &str, sync: bool) -> (String, String) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_oximond"));
    command
        .args([execution, "-e", SOURCE])
        .env("OXIMOND_JIT_INJECT_COMPILE_PANICS", "0")
        .env("OXIMOND_JIT_TRACE", "1")
        .env("OXIMOND_JIT_STATS", "1")
        .env_remove("OXIMOND_JIT_SYNC");
    if sync {
        command.env("OXIMOND_JIT_SYNC", "1");
    }
    let output = command.output().expect("run oximond");
    assert!(output.status.success(), "{output:?}");
    (
        String::from_utf8(output.stdout).unwrap(),
        String::from_utf8(output.stderr).unwrap(),
    )
}

fn check(sync: bool) {
    let (expected, _) = run("--execution=interpreter", sync);
    let (actual, stderr) = run("--execution=tracing", sync);
    assert_eq!(actual, expected, "{stderr}");
    assert!(
        stderr.contains("panicked, trace rejected: injected compile panic"),
        "{stderr}"
    );
    assert!(
        stderr.contains("jit-trace: compile panic 1: injected compile panic"),
        "{stderr}"
    );
    assert!(stderr.contains("compile_panics: 1,"), "{stderr}");
    // The second loop compiled after the panic and ran natively.
    assert!(stderr.contains("jit-trace: compiled trace"), "{stderr}");
    assert!(!stderr.contains("native_entries: 0,"), "{stderr}");
}

#[test]
fn a_synchronous_compile_panic_is_reported_and_later_loops_compile() {
    check(true);
}

#[test]
fn a_background_compile_panic_is_reported_and_later_loops_compile() {
    check(false);
}
