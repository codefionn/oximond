//! Replays the checked-in regression corpus through embedded oximond in
//! both execution modes, with synchronous JIT compilation.
//!
//! Each entry is a generated (often reduced) program with the output its
//! profile's stock interpreter printed. An entry with a `.known-failure`
//! file lists the exact `mode:category` differences an open oximond bug
//! causes; the test fails if they change or go away, so a fix updates the
//! corpus in the same commit.

use std::collections::BTreeSet;
use std::path::Path;

use oximond::{CompatibilityProfile, ExecutionMode, Lua};
use oximond_difftest::campaign::read_corpus;
use oximond_difftest::case::{End, Observation, difference};
use oximond_difftest::profile::Profile;

fn compatibility(profile: Profile) -> CompatibilityProfile {
    match profile {
        Profile::Lua51 => CompatibilityProfile::Lua51,
        Profile::Lua52 => CompatibilityProfile::Lua52,
        Profile::Lua53 => CompatibilityProfile::Lua53,
        Profile::Lua54 => CompatibilityProfile::Lua54,
        Profile::Lua55 => CompatibilityProfile::Lua55,
        Profile::LuaJit21 => CompatibilityProfile::LuaJit21,
    }
}

fn run(profile: Profile, mode: ExecutionMode, source: &str) -> Observation {
    let mut lua = Lua::builder()
        .profile(compatibility(profile))
        .execution_mode(mode)
        .build()
        .expect("the standard library installs");
    lua.vm_mut().use_synchronous_jit_compilation();
    let end = match lua.compile("@case.lua", source) {
        Err(error) => End::Error(error.to_string()),
        Ok(chunk) => match lua.execute(&chunk) {
            Ok(_) => End::Success,
            Err(error) => End::Error(error.to_string()),
        },
    };
    Observation {
        stdout: lua.runtime_mut().take_output(),
        end,
    }
}

/// Runs the corpus on a thread with the CLI's stack size: entries that
/// recurse to stock's C-call limit need more than a test thread's default.
#[test]
fn corpus_entries_print_what_stock_lua_printed() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(replay_corpus)
        .expect("spawn the corpus thread")
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
}

fn replay_corpus() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let entries = read_corpus(&directory).expect("the corpus reads");
    assert!(!entries.is_empty(), "the corpus is empty");
    let mut problems = Vec::new();
    for entry in &entries {
        let expected = Observation {
            stdout: entry.expected_stdout.clone(),
            end: End::Success,
        };
        let mut signature = BTreeSet::new();
        for (name, mode) in [
            ("interpreter", ExecutionMode::Interpreter),
            ("tracing", ExecutionMode::Tracing),
        ] {
            let actual = run(entry.profile, mode, &entry.source);
            if let Some(category) = difference(&expected, &actual) {
                signature.insert(format!("{name}:{category}"));
            }
        }
        let known = entry.known_failure.clone().unwrap_or_default();
        if signature != known {
            problems.push(format!(
                "{} ({}): expected [{}], got [{}]",
                entry.name,
                entry.profile,
                known.into_iter().collect::<Vec<_>>().join(","),
                signature.into_iter().collect::<Vec<_>>().join(",")
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "corpus mismatches:\n{}",
        problems.join("\n")
    );
}
