//! `OXIMOND_OP_STATS` writes the operation statistics of an `op-stats` build
//! as JSON to the file it names, at a clean exit. A build without the feature
//! writes nothing. `benchmarks/measure.py` reads the report.
//!
//! The counters are process-wide, so every case runs its own process.

use std::path::PathBuf;
use std::process::Command;

static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn scratch(name: &str) -> PathBuf {
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let directory =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("op-stats-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    directory.join(format!("{id}-{name}"))
}

/// Runs `source` and returns its stdout and the report, if one was written.
fn run(compat: &str, execution: &str, source: &str, report: bool) -> (String, Option<String>) {
    let script = scratch("case.lua");
    std::fs::write(&script, source).unwrap();
    let path = scratch("report.json");
    let mut command = Command::new(env!("CARGO_BIN_EXE_oximond"));
    command
        .arg(format!("--compat={compat}"))
        .arg(format!("--execution={execution}"))
        .arg(&script)
        .env_remove("OXIMOND_OP_STATS")
        .env("OXIMOND_JIT_SYNC", "1");
    if report {
        command.env("OXIMOND_OP_STATS", &path);
    }
    let output = command.output().unwrap();
    assert!(output.status.success(), "{output:?}");
    (
        String::from_utf8(output.stdout).unwrap(),
        std::fs::read_to_string(&path).ok(),
    )
}

const ARRAY_LOOP: &str = "
local a = {}
for i = 1, 1000 do a[i] = i end
local s = 0
for r = 1, 50 do
  for i = 1, 999 do s = s + a[i] + a[i + 1] end
end
print(s)
";

#[cfg(not(feature = "op-stats"))]
#[test]
fn a_build_without_the_feature_writes_no_report() {
    let (stdout, report) = run("lua54", "tracing", ARRAY_LOOP, true);
    assert_eq!(stdout, "49999950\n");
    assert!(report.is_none());
}

#[cfg(feature = "op-stats")]
mod enabled {
    use super::*;

    /// The integer value of `key` in the report's counters object. A missing
    /// key fails the test: the report must name every counter.
    fn counter(report: &str, key: &str) -> u64 {
        let counters = &report[report.find("\"counters\":{").expect("counters object")..];
        let needle = format!("\"{key}\":");
        let start = counters
            .find(&needle)
            .unwrap_or_else(|| panic!("{key} missing from {report}"))
            + needle.len();
        counters[start..]
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .unwrap()
            .parse()
            .unwrap()
    }

    #[test]
    fn the_report_names_its_schema_build_and_mode() {
        let (_, report) = run("lua54", "interpreter", "print(1)", true);
        let report = report.expect("an op-stats build writes the report");
        assert!(
            report.starts_with("{\"schema\":\"oximond-op-stats/2\""),
            "{report}"
        );
        assert!(report.contains("\"build\":[\"op-stats\""), "{report}");
        assert!(report.contains("\"execution\":\"interpreter\""), "{report}");
        assert!(report.contains("\"jit_sync\":\"on\""), "{report}");
        let (_, none) = run("lua54", "interpreter", "print(1)", false);
        assert!(none.is_none());
    }

    #[test]
    fn jit_sync_follows_the_brokers_rule() {
        for (value, expected) in [
            (None, "off"),
            (Some(""), "off"),
            (Some("0"), "off"),
            (Some("1"), "on"),
        ] {
            let script = scratch("sync.lua");
            std::fs::write(&script, "print(1)").unwrap();
            let path = scratch("sync.json");
            let mut command = Command::new(env!("CARGO_BIN_EXE_oximond"));
            command
                .arg(&script)
                .env_remove("OXIMOND_JIT_SYNC")
                .env("OXIMOND_OP_STATS", &path);
            if let Some(value) = value {
                command.env("OXIMOND_JIT_SYNC", value);
            }
            assert!(command.output().unwrap().status.success());
            let report = std::fs::read_to_string(&path).unwrap();
            assert!(
                report.contains(&format!("\"jit_sync\":\"{expected}\"")),
                "{value:?}: {report}"
            );
        }
    }

    #[test]
    fn counting_does_not_change_output() {
        for execution in ["interpreter", "tracing"] {
            let (with, _) = run("lua54", execution, ARRAY_LOOP, true);
            let (without, _) = run("lua54", execution, ARRAY_LOOP, false);
            assert_eq!(with, without);
            assert_eq!(with, "49999950\n");
        }
    }

    #[test]
    fn a_counted_array_loop_has_range_once_candidates_and_native_checks() {
        let (_, report) = run("lua54", "tracing", ARRAY_LOOP, true);
        let report = report.unwrap();
        assert!(counter(&report, "h3.ir.trees_compiled") >= 1, "{report}");
        let sites = counter(&report, "h1.ir.array_sites");
        assert!(sites >= 2, "{report}");
        assert_eq!(counter(&report, "h1.ir.array_sites.range_once"), sites);
        let checks = counter(&report, "h1.native.bound_checks");
        // Two checks per native iteration, and most of 50 * 999 iterations
        // run natively.
        assert!(checks > 2 * 40 * 999, "{report}");
        assert_eq!(checks % 2, 0);
        assert_eq!(
            counter(&report, "h1.native.bound_checks.range_once"),
            checks
        );
        assert_eq!(counter(&report, "h1.native.integrality_checks"), 0);
    }

    #[test]
    fn float_keys_need_integrality_checks() {
        // Lua 5.1 has only floats.
        let (stdout, report) = run("lua51", "tracing", ARRAY_LOOP, true);
        assert_eq!(stdout, "49999950\n");
        let report = report.unwrap();
        assert!(
            counter(&report, "h1.ir.array_sites.float_key") >= 1,
            "{report}"
        );
        assert!(
            counter(&report, "h1.native.integrality_checks") > 0,
            "{report}"
        );
    }

    const CONSTRUCTOR_LOOP: &str = "
local total = 0
for i = 1, 20000 do
  local v = {i, i + 1, tag = i % 5}
  if i % 500 == 0 then total = total + 1 end
  total = total + v[1] + v.tag
end
print(total)
";

    #[test]
    fn a_constructor_in_a_loop_leaves_virtual_snapshot_entries() {
        let (_, report) = run("lua54", "tracing", CONSTRUCTOR_LOOP, true);
        let report = report.unwrap();
        assert!(counter(&report, "h3.ir.trees_compiled") >= 1, "{report}");
        assert!(
            counter(&report, "h3.ir.snapshot_entries.virtual") > 0,
            "{report}"
        );
        assert_eq!(
            counter(&report, "h3.ir.snapshot_entries.other"),
            0,
            "{report}"
        );
        assert!(counter(&report, "h3.ir.snapshot_virtuals") > 0, "{report}");
        assert!(
            counter(&report, "h3.ir.snapshot_recipe_entries") > 0,
            "{report}"
        );
        assert_eq!(
            counter(&report, "h3.ir.snapshot_recipes_unknown"),
            0,
            "{report}"
        );
        let operands = counter(&report, "h3.ir.snapshot_recipe_entries");
        assert_eq!(
            operands,
            counter(&report, "h3.ir.snapshot_recipe_entries.value")
                + counter(&report, "h3.ir.snapshot_recipe_entries.const")
                + counter(&report, "h3.ir.snapshot_recipe_entries.root"),
            "{report}"
        );
        assert_eq!(
            counter(&report, "h3.ir.snapshot_recipe_entries.value"),
            counter(&report, "h3.ir.snapshot_recipe_values.feeding")
                + counter(&report, "h3.ir.snapshot_recipe_values.recovery"),
            "{report}"
        );
    }

    #[test]
    fn the_interpreter_compiles_and_runs_no_native_code() {
        let (_, report) = run("lua54", "interpreter", ARRAY_LOOP, true);
        let report = report.unwrap();
        assert_eq!(counter(&report, "h3.ir.trees_compiled"), 0);
        assert_eq!(counter(&report, "h1.native.bound_checks"), 0);
        assert!(counter(&report, "h1.interp.get_table.integer_key") >= 2 * 50 * 999);
    }

    #[test]
    fn integer_modulo_stays_on_the_fast_arm() {
        // Both signs and a negative divisor; every operand is an integer.
        let source = "
local s = 0
for i = 1, 1000 do
  s = s + i % 7 + (-i) % 3 + i % -5
end
print(s)
";
        let (stdout, report) = run("lua54", "interpreter", source, true);
        assert_eq!(stdout, "2004\n");
        assert_eq!(counter(&report.unwrap(), "h2.fast.arithmetic_slow"), 0);
        // Integer % float still takes the general path.
        let source = "
local s = 0
for i = 1, 1000 do s = s + i % 2.5 end
print(s)
";
        let (stdout, report) = run("lua54", "interpreter", source, true);
        assert_eq!(stdout, "1000.0\n");
        assert!(counter(&report.unwrap(), "h2.fast.arithmetic_slow") >= 1000);
    }

    const CALLS: &str = "
local function make(k) return function(x, y) return x + y + k end end
local fs = { make(1), make(2) }
local function other(x, y) return x - y end
local s = 0
for i = 1, 4000 do
  local f = fs[i % 2 + 1]
  s = s + f(i, 2)
  local g = (i % 100 == 0) and other or fs[1]
  s = s + g(i, 1)
end
print(s)
";

    #[test]
    fn call_sites_tell_closures_of_one_prototype_from_other_prototypes() {
        let (stdout, report) = run("lua54", "interpreter", CALLS, true);
        let (expected, _) = run("lua54", "interpreter", CALLS, false);
        assert_eq!(stdout, expected);
        let report = report.unwrap();
        let calls = counter(&report, "h4.call_sites.closure_calls");
        assert!(calls >= 8000, "{report}");
        // The first site alternates between two closures of `make`'s
        // prototype; the second mostly calls one closure.
        assert!(
            counter(&report, "h6.calls.same_prototype_other_closure") >= 3990,
            "{report}"
        );
        assert!(
            counter(&report, "h6.calls.other_prototype") >= 70,
            "{report}"
        );
        assert!(
            counter(&report, "h6.calls.same_closure") >= 3800,
            "{report}"
        );
        assert!(counter(&report, "h6.closures_created") >= 3);
        assert!(counter(&report, "h4.call_sites.arguments") >= 2 * 8000);
        assert!(counter(&report, "h2.fast.dispatches") > 0);
    }

    #[test]
    fn collections_scan_frame_roots() {
        let source = "
local keep = {}
for i = 1, 20000 do local t = {i} keep[i % 10 + 1] = t end
collectgarbage()
print(#keep)
";
        let (stdout, report) = run("lua54", "interpreter", source, true);
        assert_eq!(stdout, "10\n");
        let report = report.unwrap();
        assert!(counter(&report, "h5.gc.root_scans") >= 1, "{report}");
        assert!(counter(&report, "h5.gc.frame_slots") >= 1, "{report}");
        assert!(
            counter(&report, "h5.gc.frame_slots.reference")
                >= counter(&report, "h5.gc.frame_slots.reference_dead")
        );
    }
}
