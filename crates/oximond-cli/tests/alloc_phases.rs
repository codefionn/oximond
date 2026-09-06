//! `OXIMOND_ALLOC_PHASES` writes the per-phase allocation report of an
//! `alloc-phases` build. `benchmarks/measure.py` validates it in full; these
//! tests check that the phases land where the work happens.
#![cfg(feature = "alloc-phases")]

use std::io::Write;
use std::process::Command;

static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// A loop the JIT compiles, with tables the trace builds.
const SOURCE: &str = "local t = {} for i = 1, 100 do t[i] = i end local s = 0 \
    for n = 1, 2000 do for i = 1, 100 do s = s + t[i] end end print(s)";

fn report(execution: &str, source: &str) -> String {
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let directory = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("alloc-phases-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let script = directory.join(format!("script_{id}.lua"));
    let output = directory.join(format!("report_{id}.json"));
    std::fs::File::create(&script)
        .unwrap()
        .write_all(source.as_bytes())
        .unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .arg(format!("--execution={execution}"))
        .arg(&script)
        .env("OXIMOND_JIT_SYNC", "1")
        .env("OXIMOND_ALLOC_PHASES", &output)
        .env_remove("OXIMOND_COUNTERS")
        .output()
        .unwrap();
    assert!(status.status.success(), "{status:?}");
    let text = std::fs::read_to_string(&output).unwrap();
    let _ = std::fs::remove_file(&script);
    let _ = std::fs::remove_file(&output);
    text
}

/// The number after `"key":` inside the object of `phase`.
fn phase_value(report: &str, phase: &str, key: &str) -> u64 {
    let start = report
        .find(&format!("\"{phase}\":{{\"alloc_count\""))
        .unwrap_or_else(|| panic!("{phase} missing from {report}"));
    let rest = &report[start..];
    let at = rest.find(&format!("\"{key}\":")).unwrap() + key.len() + 3;
    let digits: String = rest[at..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().unwrap()
}

#[test]
fn tracing_compiles_in_the_compiler_phases_and_reconciles() {
    let report = report("tracing", SOURCE);
    assert!(
        report.starts_with("{\"schema\":\"oximond-alloc-phases/1\""),
        "{report}"
    );
    assert!(report.contains("\"reconciled\":true"), "{report}");
    assert!(report.contains("\"jit_sync\":\"on\""), "{report}");
    for phase in [
        "unclassified",
        "recording",
        "optimization",
        "codegen",
        "installation",
        "execution",
    ] {
        assert!(
            phase_value(&report, phase, "alloc_count") > 0,
            "{phase}: {report}"
        );
    }
    // A synchronous broker compiles on the VM thread, and guest code runs
    // there.
    for phase in [
        "recording",
        "optimization",
        "codegen",
        "installation",
        "execution",
    ] {
        assert_eq!(
            phase_value(&report, phase, "alloc_count_other"),
            0,
            "{phase}: {report}"
        );
    }
}

#[test]
fn the_interpreter_never_enters_the_compiler_phases() {
    let report = report("interpreter", SOURCE);
    assert!(report.contains("\"reconciled\":true"), "{report}");
    for phase in [
        "recording",
        "optimization",
        "codegen",
        "installation",
        "exit_recovery",
    ] {
        assert_eq!(
            phase_value(&report, phase, "alloc_count"),
            0,
            "{phase}: {report}"
        );
    }
    assert!(
        phase_value(&report, "execution", "alloc_count") > 0,
        "{report}"
    );
}
