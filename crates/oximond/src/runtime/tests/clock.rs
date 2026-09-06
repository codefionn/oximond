use super::*;

#[cfg(target_os = "linux")]
#[test]
fn clock_reports_process_cpu_ticks() {
    fn ticks() -> u64 {
        let stat = std::fs::read_to_string("/proc/self/stat").unwrap();
        let fields: Vec<_> = stat
            .rsplit_once(") ")
            .unwrap()
            .1
            .split_whitespace()
            .collect();
        fields[11].parse::<u64>().unwrap() + fields[12].parse::<u64>().unwrap()
    }

    let output = std::process::Command::new("getconf")
        .arg("CLK_TCK")
        .output()
        .unwrap();
    assert!(output.status.success());
    let rate = std::str::from_utf8(&output.stdout)
        .unwrap()
        .trim()
        .parse::<f64>()
        .unwrap();
    let runtime = Runtime::new();
    let before = ticks() as f64 / rate;
    let observed = runtime.process_cpu_seconds();
    let after = ticks() as f64 / rate;
    assert!(
        before <= observed && observed <= after,
        "{before} <= {observed} <= {after}"
    );
}
