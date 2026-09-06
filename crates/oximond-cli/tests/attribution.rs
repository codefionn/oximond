//! `OXIMOND_ATTRIBUTION` writes the sampler report, and `OXIMOND_PERF_MAP`
//! names the compiled traces for `perf report`. `perf record` sets
//! `PERF_BUILDID_DIR`, which makes Cranelift write generic names for the same
//! ranges; the trace map replaces them.

use std::path::PathBuf;
use std::process::{Command, Stdio};

const SOURCE: &str = "local t = os.clock()
local s = 0
repeat for i = 1, 100000 do s = s + i end until os.clock() - t > 0.5
print(s)";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("oximond-attr-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn the_report_file_names_the_traced_loop() {
    let dir = scratch("report");
    let report = dir.join("report.tsv");
    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args(["--execution=tracing", "-e", SOURCE])
        .env("OXIMOND_ATTRIBUTION", &report)
        .output()
        .expect("run oximond");
    assert!(output.status.success(), "{output:?}");
    let text = std::fs::read_to_string(&report).expect("report written at exit");
    assert!(text.starts_with("oximond-attribution\t1\t"), "{text}");
    let compiled = text
        .lines()
        .find(|line| line.starts_with("loop\t") && line.contains("\tcompiled\t"))
        .unwrap_or_else(|| panic!("{text}"));
    let fields: Vec<&str> = compiled.split('\t').collect();
    // loop, thread, source, defined, pc, header line, innermost, inclusive,
    // native, state, ...
    assert_eq!(fields[5], "3", "{compiled}");
    assert!(fields[8].parse::<u64>().unwrap() > 0, "{compiled}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_perf_map_names_each_trace_by_its_header_line() {
    let child = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args(["--execution=tracing", "-e", SOURCE])
        .env("OXIMOND_PERF_MAP", "1")
        .stdout(Stdio::null())
        .spawn()
        .expect("run oximond");
    let map = PathBuf::from(format!("/tmp/perf-{}.map", child.id()));
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let text = std::fs::read_to_string(&map).expect("perf map");
    let _ = std::fs::remove_file(&map);
    let line = text
        .lines()
        .find(|line| line.contains(" oximond trace#"))
        .unwrap_or_else(|| panic!("{text}"));
    assert!(line.ends_with(":3"), "{line}");
    let mut fields = line.splitn(3, ' ');
    assert!(usize::from_str_radix(fields.next().unwrap(), 16).unwrap() > 0);
    assert!(usize::from_str_radix(fields.next().unwrap(), 16).unwrap() > 0);
}

#[test]
fn perf_record_attributes_samples_to_named_traces() {
    let dir = scratch("perf");
    let data = dir.join("perf.data");
    let recorded = Command::new("perf")
        .args(["record", "-q", "-F", "5000", "-o"])
        .arg(&data)
        .arg("--")
        .arg(env!("CARGO_BIN_EXE_oximond"))
        .args(["--execution=tracing", "-e", SOURCE])
        .env("OXIMOND_PERF_MAP", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    // `perf` may be missing or not permitted to sample here.
    let Ok(status) = recorded else { return };
    if !status.success() || !data.exists() {
        return;
    }
    let report = Command::new("perf")
        .args(["report", "--stdio", "--sort", "sym", "-i"])
        .arg(&data)
        .output()
        .expect("perf report");
    let text = String::from_utf8_lossy(&report.stdout);
    let _ = std::fs::remove_dir_all(dir);
    let line = text
        .lines()
        .find(|line| line.contains("oximond trace#"))
        .unwrap_or_else(|| panic!("no named trace in:\n{text}"));
    let percent: f64 = line
        .split('%')
        .next()
        .unwrap()
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("{line}"));
    assert!(percent > 10.0, "{line}");
}
