//! The JIT property, tested with proptest: for generated hot-loop programs,
//! every tracing variant of [`Mode::JIT`] (the shipped JIT, MIR with Cranelift
//! for what MIR declines; the same loading its own recorded trace profile; MIR
//! alone; Cranelift alone) prints and fails exactly as oximond's interpreter
//! does.
//!
//! The strategy draws a seed and the shrinkable knobs of
//! [`HotKnobs`](oximond_difftest::generate::HotKnobs). A failing case is
//! shrunk by proptest, reduced by the difftest reducer, and reported as the
//! reduced program with a one-line replay command. The default run is a fixed
//! set of cases and is deterministic.
//!
//! Environment:
//! - `OXIMOND_DIFFTEST_OXIMOND`: the binary under test (the release build for
//!   long runs). Without it the test uses `target/debug/oximond`, building it
//!   if it is missing.
//! - `OXIMOND_PROPTEST_CASES`: number of generated programs (default 4).
//! - `OXIMOND_PROPTEST_SEED`: seed of proptest's own random stream (default 0,
//!   a neutral choice: seeds 0 to 10 all pass). A different value draws
//!   different programs.
//! - `OXIMOND_PROPTEST_ASYNC`: set to `1` to run `tracing-async` and
//!   `tracing-mir-async` (compilation on the background thread) as well.
//! - `OXIMOND_PROPTEST_MIN_NATIVE`: least share of cases in which a variant
//!   must enter native code (default 0.6).
//! - `OXIMOND_PROPTEST_LIGHT_CASES`: extra cases run under `tracing` and
//!   `tracing-mir` only (default 8). They widen the draw behind the 60% check
//!   for the variants whose native share is lowest.
//! - `OXIMOND_PROPTEST_MIN_TRIPS`, `OXIMOND_PROPTEST_MAX_TRIPS`: bounds of the
//!   trip-count knob (default 4000 to 8000). The anchor loop of every program
//!   (see `HotKnobs::anchor`) always compiles, so the check counts only native
//!   entries beyond what the anchor alone gives. The random sections need no
//!   more trips than the debug binary can afford: a loaded
//!   profile keeps the interpreter off its fast loop, so a program with many
//!   interpreted trips takes about 7 times as long in a profile variant.
//!
//! The fake-binary tests prove the harness itself: a variant that prints
//! something else is caught, shrunk and reduced, and a run in which no native
//! code ran fails instead of passing.

#![cfg(unix)]

use std::cell::RefCell;
use std::fmt::Write as _;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use proptest::strategy::Strategy;
use proptest::test_runner::{Config, RngAlgorithm, TestCaseError, TestError, TestRng, TestRunner};

use oximond_difftest::case::{CaseResult, Engines, Mode, Oracle, Verdict, run_case};
use oximond_difftest::compare::Comparison;
use oximond_difftest::generate::{HotKnobs, Size, generate_hot};
use oximond_difftest::jit::{Baseline, check_non_vacuity, summarize_results_above};
use oximond_difftest::profile::Profile;
use oximond_difftest::reduce::reduce;
use oximond_difftest::run::Limits;

const PROFILES: [Profile; 6] = [
    Profile::Lua54,
    Profile::LuaJit21,
    Profile::Lua51,
    Profile::Lua55,
    Profile::Lua52,
    Profile::Lua53,
];

#[derive(Clone, Debug)]
struct Input {
    seed: u64,
    profile: Profile,
    knobs: HotKnobs,
}

/// The seed draws the program's base shape the way a campaign does (see
/// [`HotKnobs::for_seed`]), which keeps most cases productive. The other
/// fields are caps on that shape and a feature mask, so proptest shrinks toward
/// two sections, three statements, depth one, the fewest trips, no extra value
/// types and the fewest features the case still fails with.
/// A cap that is mostly the largest value, so most programs keep the shape the
/// seed drew, and that shrinks toward `low`.
fn cap(low: u32, high: u32) -> impl Strategy<Value = u32> {
    proptest::prop_oneof![1 => low..=high, 4 => proptest::strategy::Just(high)]
}

fn strategy(min_trips: u32, max_trips: u32) -> impl Strategy<Value = Input> {
    (
        0..(1_u64 << 32),
        0..PROFILES.len(),
        cap(2, 3),
        cap(3, 6),
        cap(min_trips, max_trips),
        cap(1, 2),
        cap(0, 3),
        proptest::prop_oneof![
            1 => 0..=HotKnobs::ALL_FEATURES,
            4 => proptest::strategy::Just(HotKnobs::ALL_FEATURES),
        ],
    )
        .prop_map(
            move |(seed, profile, sections, statements, trips, depth, types, mask)| {
                let base = HotKnobs::for_seed(seed, Size::Medium);
                Input {
                    seed,
                    profile: PROFILES[profile],
                    knobs: HotKnobs {
                        sections: base.sections.min(sections),
                        statements: base.statements.min(statements),
                        trips: base.trips.clamp(min_trips, trips),
                        depth: base.depth.min(depth),
                        types: base.types.min(u8::try_from(types).expect("at most 3")),
                        features: base.features & mask & !HotKnobs::UNSUPPORTED,
                        anchor: true,
                    },
                }
            },
        )
}

struct Harness {
    binary: PathBuf,
    work: PathBuf,
    variants: Vec<Mode>,
    cases: u32,
    /// Extra cases run under the plain tracing variants only (see
    /// `light_engines`).
    light_cases: u32,
    seed: u8,
    min_trips: u32,
    max_trips: u32,
    min_native_share: f64,
    reduce_attempts: usize,
}

/// The synchronous variants of [`Mode::JIT`], and with `OXIMOND_PROPTEST_ASYNC` set to
/// anything but `0` the two background-compilation variants as well.
fn variants() -> Vec<Mode> {
    let mut variants = Mode::JIT.to_vec();
    if std::env::var("OXIMOND_PROPTEST_ASYNC").is_ok_and(|value| value != "0") {
        variants.extend([Mode::TracingAsync, Mode::TracingMirAsync]);
    }
    variants
}

fn env_number<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(default)
}

impl Harness {
    fn new(binary: PathBuf, name: &str) -> Self {
        let work = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join("jit-property")
            .join(name);
        let _ = fs::remove_dir_all(&work);
        Self {
            binary,
            work,
            variants: variants(),
            cases: env_number("OXIMOND_PROPTEST_CASES", 4),
            light_cases: env_number("OXIMOND_PROPTEST_LIGHT_CASES", 8),
            seed: env_number("OXIMOND_PROPTEST_SEED", 0),
            min_trips: env_number("OXIMOND_PROPTEST_MIN_TRIPS", 4_000),
            max_trips: env_number("OXIMOND_PROPTEST_MAX_TRIPS", 8_000),
            min_native_share: env_number("OXIMOND_PROPTEST_MIN_NATIVE", 0.6),
            reduce_attempts: 200,
        }
    }

    fn engines(&self) -> Engines {
        Engines {
            oximond: self.binary.clone(),
            limits: Limits {
                timeout: Duration::from_secs(60),
                memory_mib: 2048,
                ..Limits::default()
            },
            modes: self.variants.clone(),
            trace_profile_record_dir: None,
            trace_profile_load_dir: None,
            comparison: Comparison::Exact,
        }
    }

    /// The engines for the extra cases: only the variants that run without a
    /// recorded profile. A profile variant costs several times a plain one
    /// (it runs twice and loads the profile), and it is not the one whose
    /// native share is low, so these cases widen the draw the 60% check sees
    /// for the plain variants at a fraction of the price.
    fn light_engines(&self) -> Engines {
        let mut engines = self.engines();
        engines
            .modes
            .retain(|mode| matches!(mode, Mode::Tracing | Mode::TracingMir));
        engines
    }

    /// Runs the property. `Err` is the failure report, not a panic, so the
    /// fake-binary tests can inspect it.
    fn run_property(&self) -> Result<String, String> {
        let engines = self.engines();
        let light = self.light_engines();
        let drawn = std::cell::Cell::new(0_u32);
        let results = RefCell::new(Vec::new());
        let config = Config {
            cases: self.cases + self.light_cases,
            max_shrink_iters: 60,
            failure_persistence: None,
            ..Config::default()
        };
        let mut seed = [0_u8; 32];
        seed[0] = self.seed;
        seed[1] = 0x6a;
        let mut runner =
            TestRunner::new_with_rng(config, TestRng::from_seed(RngAlgorithm::ChaCha, &seed));
        let directory = self.work.join("case");
        let outcome = runner.run(&strategy(self.min_trips, self.max_trips), |input| {
            let program = generate_hot(input.seed, input.profile, &input.knobs);
            let oracle = Oracle::interpreter(&self.binary, input.profile);
            let started = Instant::now();
            // The first `cases` draws run every variant, the rest the light set.
            let engines = if drawn.get() < self.cases {
                &engines
            } else {
                &light
            };
            drawn.set(drawn.get() + 1);
            let result = run_case(
                &program.render(),
                &directory,
                input.profile,
                Some(&oracle),
                engines,
            )
            .map_err(TestCaseError::fail)?;
            let elapsed = started.elapsed();
            if std::env::var_os("OXIMOND_PROPTEST_VERBOSE").is_some() {
                let native = result
                    .modes
                    .iter()
                    .map(|run| {
                        format!(
                            "{}={}",
                            run.label,
                            run.jit.as_ref().map_or(0, |jit| jit.native_entries)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                eprintln!(
                    "case {} {} {} ({:.1} s): {native}",
                    input.profile,
                    input.seed,
                    input.knobs.describe(),
                    elapsed.as_secs_f64()
                );
            }
            match result.verdict {
                Verdict::Pass => {
                    results.borrow_mut().push((input.profile, result));
                    Ok(())
                }
                _ => Err(TestCaseError::fail(format!(
                    "{} {}",
                    result.verdict.name(),
                    result.signature.join(",")
                ))),
            }
        });
        match outcome {
            Ok(()) => {
                let results = results.into_inner();
                // The anchor kernel is in every program and always compiles, so
                // count only what the random sections added to it.
                let baseline = self.anchor_baseline(&results)?;
                let stats = summarize_results_above(
                    results.iter().map(|(profile, result)| (*profile, result)),
                    &baseline,
                );
                let mut summary = format!("{} cases passed\n", results.len());
                for ((profile, label), row) in &stats {
                    let _ = writeln!(
                        summary,
                        "{profile} {label}: native {}/{}, MIR trees {}/{}",
                        row.native_cases, row.cases, row.mir_accepted_trees, row.mir_fallback_trees
                    );
                }
                check_non_vacuity(&stats, self.min_native_share)
                    .map_err(|problem| format!("vacuous run: {problem}\n{summary}"))?;
                Ok(summary)
            }
            Err(TestError::Fail(reason, input)) => Err(self.report(&input, &reason.to_string())),
            Err(TestError::Abort(reason)) => Err(format!("proptest aborted: {reason}")),
        }
    }

    /// What the anchor kernel alone does under each variant, for each profile
    /// in `results`.
    fn anchor_baseline(&self, results: &[(Profile, CaseResult)]) -> Result<Baseline, String> {
        let engines = self.engines();
        let directory = self.work.join("baseline");
        let mut baseline = Baseline::new();
        let profiles: std::collections::BTreeSet<Profile> =
            results.iter().map(|(profile, _)| *profile).collect();
        for profile in profiles {
            let program = generate_hot(0, profile, &HotKnobs::anchor_only());
            let oracle = Oracle::interpreter(&self.binary, profile);
            let result = run_case(
                &program.render(),
                &directory,
                profile,
                Some(&oracle),
                &engines,
            )
            .map_err(|error| format!("anchor baseline for {profile}: {error}"))?;
            if result.verdict != Verdict::Pass {
                return Err(format!(
                    "anchor baseline for {profile}: {} {}",
                    result.verdict.name(),
                    result.signature.join(",")
                ));
            }
            for run in result.modes {
                if let Some(jit) = run.jit {
                    baseline.insert((profile, run.label), jit);
                }
            }
        }
        Ok(baseline)
    }

    /// Reduces the shrunk input and writes the report.
    fn report(&self, input: &Input, reason: &str) -> String {
        let engines = self.engines();
        let oracle = Oracle::interpreter(&self.binary, input.profile);
        let program = generate_hot(input.seed, input.profile, &input.knobs);
        let directory = self.work.join("reduce");
        let first = run_case(
            &program.render(),
            &directory,
            input.profile,
            Some(&oracle),
            &engines,
        );
        let target: Vec<String> = first.map(|result| result.signature).unwrap_or_default();
        let reduction = reduce(
            &program,
            |candidate| {
                run_case(
                    &candidate.render(),
                    &directory,
                    input.profile,
                    Some(&oracle),
                    &engines,
                )
                .is_ok_and(|result| {
                    result.verdict == Verdict::Mismatch
                        && result.signature.iter().any(|entry| target.contains(entry))
                })
            },
            self.reduce_attempts,
            Instant::now() + Duration::from_secs(300),
        );
        let reduced = reduction.program.render();
        let saved = self.work.join("failure");
        let _ = fs::create_dir_all(&saved);
        let original_path = saved.join("original.lua");
        let reduced_path = saved.join("reduced.lua");
        let _ = fs::write(&original_path, program.render());
        let _ = fs::write(&reduced_path, &reduced);
        let modes: Vec<&str> = target
            .iter()
            .filter_map(|entry| entry.split(':').next())
            .collect();
        let modes = if modes.is_empty() {
            self.variants
                .iter()
                .map(|mode| mode.name())
                .collect::<Vec<_>>()
        } else {
            modes
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect()
        };
        format!(
            "JIT variant differs from the interpreter: {reason}\n\
             profile {}, seed {}, shrunk knobs {}\n\
             signature: {}\n\
             reduced program ({} of {} nodes):\n{reduced}\n\
             replay: oximond-difftest replay-file {} --profile {} --interpreter-oracle --modes {} --oximond {}\n\
             original: {}\n",
            input.profile,
            input.seed,
            input.knobs.describe(),
            target.join(","),
            reduction.program.size(),
            program.size(),
            reduced_path.display(),
            input.profile,
            modes.join(","),
            self.binary.display(),
            original_path.display(),
        )
    }
}

fn locate_binary() -> Result<PathBuf, String> {
    if let Ok(path) = std::env::var("OXIMOND_DIFFTEST_OXIMOND") {
        return fs::canonicalize(&path).map_err(|error| format!("{path}: {error}"));
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let target =
        std::env::var_os("CARGO_TARGET_DIR").map_or_else(|| root.join("target"), PathBuf::from);
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    // A no-op when the binary is current, as it is after `cargo test -p
    // oximond-cli`.
    let status = std::process::Command::new(cargo)
        .args(["build", "-p", "oximond-cli", "--quiet"])
        .current_dir(&root)
        .status()
        .map_err(|error| error.to_string())?;
    if !status.success() {
        return Err("cargo build -p oximond-cli failed".to_owned());
    }
    fs::canonicalize(target.join("debug/oximond")).map_err(|error| error.to_string())
}

#[test]
fn tracing_variants_match_the_interpreter() {
    let binary = locate_binary().expect("the oximond binary");
    let harness = Harness::new(binary, "real");
    match harness.run_property() {
        Ok(summary) => {
            eprintln!("{summary}");
            let _ = fs::remove_dir_all(&harness.work);
        }
        Err(report) => panic!("{report}"),
    }
}

// ----- the harness against fake binaries ----------------------------------

/// A fake `oximond`: prints two fixed observations, writes the recording
/// backend into a recorded profile, refuses a profile that does not exist, and
/// reports the statistics a tracing run prints. `@BREAK@` is a shell
/// condition that makes the second observation wrong.
const FAKE: &str = r#"#!/bin/sh
execution=""
record=""
load=""
file=""
while [ $# -gt 0 ]; do
  case "$1" in
    --execution) execution="$2"; shift 2 ;;
    --compat) shift 2 ;;
    --trace-profile-record) record="$2"; shift 2 ;;
    --trace-profile-load) load="$2"; shift 2 ;;
    -*) shift ;;
    *) file="$1"; shift ;;
  esac
done
# The anchor-only baseline program (knobs sections=0) enters native code 5
# times, a full program 9 times.
entries=9
if grep -q 'sections=0,' "$file" 2>/dev/null; then entries=5; fi
if [ -n "$load" ] && [ ! -f "$load" ]; then echo "oximond: cannot read $load" >&2; exit 70; fi
if [ -n "$record" ]; then echo "${OXIMOND_JIT_BACKEND:-cranelift}" > "$record"; fi
echo "o1#1 same"
if @BREAK@; then echo "o2#1 WRONG"; else echo "o2#1 right"; fi
if [ "$execution" = tracing ] && [ -n "$OXIMOND_JIT_STATS" ]; then
  @STATS@
fi
"#;

const NATIVE: &str = r#"echo "trace: TraceStatistics { traces_compiled: 1, native_entries: $entries, iterations: 100 }" >&2
  if [ "$OXIMOND_JIT_BACKEND" = mir ] || [ "$OXIMOND_JIT_BACKEND" = mir-only ]; then
    echo "mir: accepted_trees=$((entries - 4)) accepted_ir=3 fallback_trees=0 fallback_ir=0 reasons={}" >&2
  else
    echo "mir: accepted_trees=0 accepted_ir=0 fallback_trees=0 fallback_ir=0 reasons={}" >&2
  fi"#;

/// Native code only ever runs the anchor: the same figures for every program.
const ANCHOR_ONLY: &str = r#"echo "trace: TraceStatistics { traces_compiled: 1, native_entries: 5, iterations: 100 }" >&2
  if [ "$OXIMOND_JIT_BACKEND" = mir ] || [ "$OXIMOND_JIT_BACKEND" = mir-only ]; then
    echo "mir: accepted_trees=1 accepted_ir=3 fallback_trees=0 fallback_ir=0 reasons={}" >&2
  else
    echo "mir: accepted_trees=0 accepted_ir=0 fallback_trees=0 fallback_ir=0 reasons={}" >&2
  fi"#;

const COLD: &str =
    r#"echo "trace: TraceStatistics { traces_compiled: 0, native_entries: 0, iterations: 0 }" >&2"#;

const MIR_FALLBACK_ONLY: &str = r#"echo "trace: TraceStatistics { traces_compiled: 1, native_entries: $entries, iterations: 100 }" >&2
  echo "mir: accepted_trees=0 accepted_ir=0 fallback_trees=1 fallback_ir=9 reasons={}" >&2"#;

fn fake(name: &str, break_when: &str, stats: &str) -> Harness {
    let mut harness = Harness::new(PathBuf::new(), name);
    fs::create_dir_all(&harness.work).unwrap();
    let path = harness.work.join("fake-oximond");
    fs::write(
        &path,
        FAKE.replace("@BREAK@", break_when)
            .replace("@STATS@", stats),
    )
    .unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    // Another test thread may fork between our write and the close of its
    // descriptor, and the child holds the file open for writing until its own
    // exec. Until then, running the script fails with ETXTBSY (errno 26), the
    // oracle run fails, and the case reads as a mismatch. Run it once here and
    // retry until it starts.
    for _ in 0..200 {
        match std::process::Command::new(&path)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
        {
            Err(error) if error.raw_os_error() == Some(26) => {
                std::thread::sleep(Duration::from_millis(10));
            }
            _ => break,
        }
    }
    harness.binary = path;
    harness.cases = 4;
    harness.light_cases = 0;
    harness.reduce_attempts = 40;
    harness
}

#[test]
fn a_good_fake_passes_and_counts_native_runs() {
    let harness = fake("good", "false", NATIVE);
    let summary = harness.run_property().unwrap();
    assert!(summary.starts_with("4 cases passed\n"), "{summary}");
    // One row per profile and variant: sum them per variant. Every case of
    // the fake enters native code once, and the MIR variants accept four trees
    // each beyond the anchor-only baseline and hand none back.
    let mut totals = std::collections::BTreeMap::<&str, [u32; 4]>::new();
    for line in summary.lines().skip(1) {
        let (label, rest) = line
            .split_once(' ')
            .and_then(|(_, rest)| rest.split_once(": "))
            .expect("a summary row");
        let numbers: Vec<u32> = rest
            .split(|c: char| !c.is_ascii_digit())
            .filter(|part| !part.is_empty())
            .map(|part| part.parse().unwrap())
            .collect();
        let total = totals.entry(label).or_default();
        for (slot, number) in total.iter_mut().zip(numbers) {
            *slot += number;
        }
    }
    let labels: Vec<&str> = totals.keys().copied().collect();
    assert_eq!(
        labels,
        [
            "tracing",
            "tracing-mir",
            "tracing-mir-only",
            "tracing-mir-profile",
        ],
        "{summary}"
    );
    for (label, [native, cases, accepted, fallback]) in &totals {
        assert_eq!((*native, *cases), (4, 4), "{label}: {summary}");
        let mir = label.contains("mir");
        assert_eq!(
            (*accepted, *fallback),
            (if mir { 16 } else { 0 }, 0),
            "{label}"
        );
    }
    let _ = fs::remove_dir_all(&harness.work);
}

#[test]
fn a_variant_that_prints_something_else_is_caught_shrunk_and_reduced() {
    let harness = fake(
        "broken-mir",
        r#"[ "$execution" = tracing ] && [ "$OXIMOND_JIT_BACKEND" = mir ]"#,
        NATIVE,
    );
    let report = harness.run_property().unwrap_err();
    assert!(report.contains("tracing-mir:stdout@o2"), "{report}");
    // The mismatch holds for every program, so proptest shrinks the caps all
    // the way down, including the value-type mix and the feature mask.
    assert!(
        report.contains("sections=2,statements=3,trips=4000,depth=1,types=0,features="),
        "{report}"
    );
    assert!(report.contains("reduced program"), "{report}");
    assert!(
        report.contains("replay: oximond-difftest replay-file"),
        "{report}"
    );
    assert!(
        report.contains("--interpreter-oracle --modes tracing-mir"),
        "{report}"
    );
    // The Cranelift variant and the interpreter agree, so only MIR variants
    // are named.
    assert!(!report.contains("tracing:stdout"), "{report}");
    let _ = fs::remove_dir_all(&harness.work);
}

#[test]
fn a_profile_that_changes_the_output_is_caught() {
    // Wrong only when a profile is loaded, as a stale or mis-keyed profile
    // would make it.
    let harness = fake(
        "broken-profile",
        r#"grep -q mir "$load_file" 2>/dev/null"#,
        NATIVE,
    );
    // `$load_file` is never set, so nothing breaks: the control.
    assert!(harness.run_property().is_ok());
    let _ = fs::remove_dir_all(&harness.work);
    let harness = fake("broken-load", r#"[ -n "$load" ]"#, NATIVE);
    let report = harness.run_property().unwrap_err();
    assert!(
        report.contains("profile:stdout@o2") || report.contains("-profile:stdout@o2"),
        "{report}"
    );
    assert!(report.contains("tracing-mir-profile"), "{report}");
    let _ = fs::remove_dir_all(&harness.work);
}

#[test]
fn a_run_that_never_enters_native_code_is_vacuous() {
    let harness = fake("cold", "false", COLD);
    let report = harness.run_property().unwrap_err();
    assert!(report.starts_with("vacuous run:"), "{report}");
    assert!(
        report.contains("0 of 4 cases entered native code"),
        "{report}"
    );
    let _ = fs::remove_dir_all(&harness.work);
}

#[test]
fn a_run_whose_only_native_code_is_the_anchor_is_vacuous() {
    let harness = fake("anchor-only", "false", ANCHOR_ONLY);
    let report = harness.run_property().unwrap_err();
    assert!(report.starts_with("vacuous run:"), "{report}");
    assert!(
        report.contains("0 of 4 cases entered native code"),
        "{report}"
    );
    assert!(
        report.contains("tracing-mir: no tree compiled through MIR"),
        "{report}"
    );
    let _ = fs::remove_dir_all(&harness.work);
}

#[test]
fn a_mir_variant_that_compiles_nothing_through_mir_is_vacuous() {
    let harness = fake("no-mir", "false", MIR_FALLBACK_ONLY);
    let report = harness.run_property().unwrap_err();
    assert!(
        report.contains("tracing-mir: no tree compiled through MIR"),
        "{report}"
    );
    let _ = fs::remove_dir_all(&harness.work);
}

#[test]
fn profile_variants_record_with_the_right_backend_and_load_what_they_recorded() {
    let mut harness = fake("profiles", "false", NATIVE);
    harness.variants = vec![
        Mode::Tracing,
        Mode::TracingProfile,
        Mode::TracingMir,
        Mode::TracingMirProfile,
        Mode::TracingMirCrossProfile,
        Mode::TracingMirOnly,
    ];
    let engines = harness.engines();
    let oracle = Oracle::interpreter(&harness.binary, Profile::Lua54);
    let directory = harness.work.join("direct");
    let result = run_case(
        "print 1\n",
        &directory,
        Profile::Lua54,
        Some(&oracle),
        &engines,
    )
    .unwrap();
    assert_eq!(result.verdict, Verdict::Pass, "{:?}", result.signature);
    let command = |label: &str| {
        result
            .modes
            .iter()
            .find(|run| run.label == label)
            .unwrap_or_else(|| panic!("{label} ran"))
            .command
            .iter()
            .map(|part| part.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join(" ")
    };
    // Cranelift's profile is loaded by Cranelift and by MIR, MIR's by MIR.
    assert!(command("tracing-profile").contains("lua54-tracing-case.lua.trace-profile"));
    assert!(command("tracing-mir-profile").contains("lua54-tracing-mir-case.lua.trace-profile"));
    assert!(command("tracing-mir-crossprofile").contains("lua54-tracing-case.lua.trace-profile"));
    for label in [
        "tracing-profile",
        "tracing-mir-profile",
        "tracing-mir-crossprofile",
    ] {
        assert!(command(label).contains("--trace-profile-load"), "{label}");
        assert!(
            !command(label).contains("--trace-profile-record"),
            "{label}"
        );
    }
    let environment = |label: &str| {
        result
            .modes
            .iter()
            .find(|run| run.label == label)
            .unwrap()
            .environment
            .clone()
    };
    let backend = |label: &str| {
        environment(label)
            .into_iter()
            .find(|(key, _)| key == "OXIMOND_JIT_BACKEND")
            .map(|(_, value)| value)
    };
    assert_eq!(backend("tracing").as_deref(), Some("cranelift"));
    assert_eq!(backend("tracing-profile").as_deref(), Some("cranelift"));
    assert_eq!(backend("tracing-mir").as_deref(), Some("mir"));
    assert_eq!(backend("tracing-mir-profile").as_deref(), Some("mir"));
    assert_eq!(backend("tracing-mir-crossprofile").as_deref(), Some("mir"));
    assert_eq!(backend("tracing-mir-only").as_deref(), Some("mir-only"));
    // The recorded profiles hold the recording backend.
    let profiles = directory.join("trace-profiles");
    let recorded: Vec<(String, String)> = fs::read_dir(&profiles)
        .unwrap()
        .flat_map(|hash| fs::read_dir(hash.unwrap().path()).unwrap())
        .map(|entry| {
            let path = entry.unwrap().path();
            (
                path.file_name().unwrap().to_string_lossy().into_owned(),
                fs::read_to_string(&path).unwrap().trim().to_owned(),
            )
        })
        .collect();
    assert!(
        recorded
            .iter()
            .any(|(name, backend)| name.contains("tracing-mir-") && backend == "mir"),
        "{recorded:?}"
    );
    assert!(
        recorded
            .iter()
            .any(|(name, backend)| !name.contains("-mir-") && backend == "cranelift"),
        "{recorded:?}"
    );
    let _ = fs::remove_dir_all(&harness.work);
}

/// Draws `OXIMOND_PROPTEST_PROBE` inputs from the strategy (seeded by
/// `OXIMOND_PROPTEST_SEED`) and prints, for each, how many native entries the
/// plain tracing variant made. It answers "how productive is the strategy"
/// without running the whole property.
#[test]
#[ignore = "prints the native entries of strategy draws: OXIMOND_PROPTEST_PROBE=N"]
fn native_share_probe() {
    let binary = locate_binary().expect("the oximond binary");
    let mut harness = Harness::new(binary, "probe");
    harness.variants = vec![Mode::Tracing];
    let engines = harness.engines();
    let mut seed = [0_u8; 32];
    seed[0] = harness.seed;
    seed[1] = 0x6a;
    let count: u32 = env_number("OXIMOND_PROPTEST_PROBE", 20);
    let config = Config {
        cases: count,
        failure_persistence: None,
        ..Config::default()
    };
    let mut runner =
        TestRunner::new_with_rng(config, TestRng::from_seed(RngAlgorithm::ChaCha, &seed));
    let (native, total) = (RefCell::new(0), RefCell::new(0));
    let directory = harness.work.join("case");
    runner
        .run(&strategy(harness.min_trips, harness.max_trips), |input| {
            // Without the anchor, so the probe sees what the random sections do.
            let knobs = HotKnobs {
                anchor: false,
                ..input.knobs
            };
            let program = generate_hot(input.seed, input.profile, &knobs);
            let oracle = Oracle::interpreter(&harness.binary, input.profile);
            let result = run_case(
                &program.render(),
                &directory,
                input.profile,
                Some(&oracle),
                &engines,
            )
            .unwrap();
            let entries = result
                .modes
                .iter()
                .map(|run| run.jit.as_ref().map_or(0, |jit| jit.native_entries))
                .max()
                .unwrap_or(0);
            *total.borrow_mut() += 1;
            *native.borrow_mut() += u32::from(entries > 0);
            eprintln!(
                "probe {} {} {} native={entries} {}",
                input.profile,
                input.seed,
                input.knobs.describe(),
                result.verdict.name()
            );
            Ok(())
        })
        .unwrap();
    eprintln!("probe native {}/{}", native.borrow(), total.borrow());
}

#[test]
#[ignore = "prints one program: OXIMOND_PROPTEST_PRINT=profile,seed,sections,statements,trips,depth,types,features[,anchor]"]
fn print_program() {
    let text = std::env::var("OXIMOND_PROPTEST_PRINT").expect("OXIMOND_PROPTEST_PRINT");
    let f: Vec<&str> = text.split(',').collect();
    let n = |i: usize| f[i].parse::<u64>().unwrap();
    let knobs = HotKnobs {
        sections: n(2) as u32,
        statements: n(3) as u32,
        trips: n(4) as u32,
        depth: n(5) as u32,
        types: n(6) as u8,
        features: n(7) as u8,
        anchor: f.get(8).is_some_and(|flag| *flag == "anchor"),
    };
    println!(
        "{}",
        generate_hot(n(1), Profile::parse(f[0]).unwrap(), &knobs).render()
    );
}
