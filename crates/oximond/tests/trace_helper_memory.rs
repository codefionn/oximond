//! A traced loop that makes a short-lived string per iteration through the
//! library helpers runs in bounded memory: the run drops the strings it made
//! at its safepoints and leaves the trace when its room for strings is used.
//! This file holds one test, so the process's peak is that test's.

use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

/// The peak resident size of this process, in kilobytes.
fn peak_kilobytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

#[test]
fn a_million_short_lived_strings_run_in_bounded_memory() {
    let source = "local n = 0 for i = 1, 1000000 do \
                    local s = tostring(i) .. 'x' n = n + #s end return n";
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::LuaJit21,
    ] {
        let mut lua = Lua::builder()
            .profile(profile)
            .execution_mode(ExecutionMode::Tracing)
            .build()
            .unwrap();
        lua.vm_mut().use_synchronous_jit_compilation();
        let results = lua.exec(source).unwrap();
        // The digits of 1 to 1000000, and an 'x' for each.
        let expected: i64 =
            9 + 90 * 2 + 900 * 3 + 9_000 * 4 + 90_000 * 5 + 900_000 * 6 + 7 + 1_000_000;
        let value = match results.first() {
            Some(Value::Integer(n)) => *n,
            #[allow(clippy::cast_possible_truncation)]
            Some(Value::Number(n)) => *n as i64,
            other => panic!("{profile:?}: {other:?}"),
        };
        assert_eq!(value, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert_eq!(stats.total_aborts(), 0, "{profile:?} {stats:?}");
        // The run reuses the lanes of the strings it released, so it leaves
        // the trace only a handful of times, not every few hundred strings.
        assert!(
            stats.exits_of(oximond::vm::TraceExitKind::GuardFail) < 10,
            "{profile:?} {stats:?}"
        );
        assert!(
            lua.vm().trace_helper_calls().contains(&"tostring"),
            "{profile:?}: {:?}",
            lua.vm().trace_helper_calls()
        );
    }
    if let Some(peak) = peak_kilobytes() {
        assert!(peak < 128 * 1024, "peak resident size {peak} kB");
    }
}
