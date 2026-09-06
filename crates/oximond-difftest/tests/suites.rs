//! Fixture and workload sources must be valid and deterministic on stock Lua.
//! New mismatches belong in campaign artifacts rather than generator failures.

use std::path::Path;

use oximond_difftest::case::{End, find_oracle, run_oracle};
use oximond_difftest::fixtures::fixtures;
use oximond_difftest::profile::Profile;
use oximond_difftest::run::Limits;
use oximond_difftest::workloads::workloads;

struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new() -> Self {
        let result = std::process::Command::new("mktemp")
            .args(["-d", "-t", "difftest-suites.XXXXXXXX"])
            .output()
            .unwrap();
        assert!(result.status.success());
        Self(String::from_utf8(result.stdout).unwrap().trim().into())
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn isolated_suites_finish_deterministically_and_clean_up_on_stock_lua() {
    let scratch = Scratch::new();
    for profile in Profile::ALL {
        let Ok(oracle) = find_oracle(profile, None) else {
            eprintln!("Skipping {profile}: stock interpreter unavailable");
            continue;
        };
        for (name, source) in fixtures(profile).into_iter().chain(workloads(profile)) {
            let directory = scratch.0.join(profile.name()).join(name);
            std::fs::create_dir_all(&directory).unwrap();
            std::fs::write(directory.join("case.lua"), source).unwrap();
            let first = run_oracle(&oracle, &directory, Limits::default());
            assert_eq!(first.observation.end, End::Success, "{profile}/{name}");
            let second = run_oracle(&oracle, &directory, Limits::default());
            assert_eq!(first.observation, second.observation, "{profile}/{name}");
            let files: Vec<_> = std::fs::read_dir(&directory)
                .unwrap()
                .map(|entry| entry.unwrap().file_name())
                .collect();
            assert_eq!(
                files,
                [Path::new("case.lua").file_name().unwrap()],
                "{profile}/{name} left fixture files"
            );
        }
    }
}
