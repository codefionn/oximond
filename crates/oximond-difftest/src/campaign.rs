//! Seed campaigns, failure artifacts, replay and the regression corpus.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::case::{
    CASE_FILE, CaseResult, EngineRun, Engines, Mode, Oracle, VS_INTERPRETER, Verdict, difference,
    difference_with, first_difference, first_difference_with, run_case, run_oximond,
};
use crate::compare::Comparison;
use crate::generate::{GENERATOR_VERSION, Size, generate_excluding, generate_hot_seed};
use crate::profile::Profile;
use crate::program::Program;
use crate::reduce::reduce;
use crate::rng::Rng;

/// The seeds a campaign runs, each on every profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Seeds {
    List(Vec<u64>),
    /// An endless stream of seeds: output `n` of `SplitMix64(master)` is
    /// seed `n`, so the same master seed replays the same order. Needs a
    /// duration.
    Random {
        master: u64,
    },
}

impl Seeds {
    /// Seed number `index`, or `None` past the end of a list.
    #[must_use]
    pub fn nth(&self, index: usize) -> Option<u64> {
        match self {
            Self::List(seeds) => seeds.get(index).copied(),
            Self::Random { master } => Some(Rng::nth(*master, index as u64)),
        }
    }
}

/// Settings shared by every case of a campaign.
#[derive(Clone, Debug)]
pub struct Settings {
    pub profiles: Vec<Profile>,
    pub seeds: Seeds,
    /// No case starts after this much time; cases already running finish.
    pub duration: Option<Duration>,
    pub size: Size,
    pub engines: Engines,
    /// The oracle per profile, or why there is none.
    pub oracles: BTreeMap<Profile, Result<Oracle, String>>,
    pub out: PathBuf,
    pub jobs: usize,
    pub reduce: bool,
    pub reduce_attempts: usize,
    pub reduce_time: Duration,
    /// Re-runs of a mismatching mode that decide whether it is stable.
    pub reruns: usize,
    /// Feature families the generator leaves out.
    pub exclude: BTreeSet<String>,
    /// Generate hot-loop programs ([`generate_hot_seed`]) instead of the
    /// default mix. `exclude` does not apply to them.
    pub hot: bool,
}

/// One attempted case, as `cases.tsv` lists it.
#[derive(Clone, Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "one flag per recorded outcome"
)]
pub struct Record {
    pub profile: Profile,
    pub seed: u64,
    pub verdict: Verdict,
    pub signature: Vec<String>,
    pub modes_agree: bool,
    /// A passing case whose oracle comparison required numeric tolerance.
    pub approximate_match: bool,
    pub jit_productive: bool,
    /// JIT activity of the synchronous tracing run, when there was one.
    pub jit: Option<crate::case::JitActivity>,
    /// JIT activity of every tracing variant that ran, by label.
    pub variant_jit: Vec<(String, crate::case::JitActivity)>,
    /// The oracle ended with an uncaught Lua error (the variants still had
    /// to match it). Tracing runs print no statistics then.
    pub oracle_error: bool,
    pub stable: Option<bool>,
    pub features: BTreeSet<&'static str>,
    pub elapsed: Duration,
    pub artifact: Option<PathBuf>,
}

impl Record {
    fn tsv(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            self.profile,
            self.seed,
            self.verdict.name(),
            if self.signature.is_empty() {
                "-".to_owned()
            } else {
                self.signature.join(",")
            },
            self.modes_agree,
            self.jit_productive,
            self.stable
                .map_or("-", |stable| if stable { "stable" } else { "flaky" }),
            self.elapsed.as_millis(),
            self.artifact
                .as_ref()
                .map_or_else(|| "-".to_owned(), |path| path.display().to_string()),
            self.features.iter().copied().collect::<Vec<_>>().join(","),
            self.approximate_match,
        )
    }
}

const TSV_HEADER: &str = "profile\tseed\tverdict\tsignature\tmodes_agree\tjit_productive\tstability\tmillis\tartifact\tfeatures\tapproximate_match\n";

/// Runs every (profile, seed) pair and writes `cases.tsv`, `report.md`
/// and failure artifacts under `settings.out`. Cases go seed by seed, each
/// seed on every profile, so a campaign cut short by `settings.duration`
/// covers the profiles evenly.
///
/// # Errors
///
/// Returns an error if the output directory cannot be written, if a case
/// went unrecorded, or if random seeds come without a duration.
///
/// # Panics
///
/// Panics if a worker thread panicked while holding the records lock.
#[allow(
    clippy::too_many_lines,
    reason = "the worker loop reads best in one place"
)]
pub fn run_campaign(settings: &Settings) -> Result<Vec<Record>, String> {
    if matches!(settings.seeds, Seeds::Random { .. }) && settings.duration.is_none() {
        return Err("random seeds need a duration".to_owned());
    }
    if settings.profiles.is_empty() {
        return Err("no profiles".to_owned());
    }
    fs::create_dir_all(&settings.out)
        .map_err(|error| format!("{}: {error}", settings.out.display()))?;
    let tsv_path = settings.out.join("cases.tsv");
    let mut tsv =
        File::create(&tsv_path).map_err(|error| format!("{}: {error}", tsv_path.display()))?;
    tsv.write_all(TSV_HEADER.as_bytes())
        .map_err(|error| error.to_string())?;
    let tsv = Arc::new(Mutex::new(tsv));
    let next = Arc::new(AtomicUsize::new(0));
    let claimed = Arc::new(AtomicUsize::new(0));
    let records = Arc::new(Mutex::new(Vec::new()));
    let failed_writes = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let deadline = settings.duration.map(|duration| started + duration);
    let total = match &settings.seeds {
        Seeds::List(seeds) => Some(seeds.len() * settings.profiles.len()),
        Seeds::Random { .. } => None,
    };
    let mut workers = Vec::new();
    for _ in 0..settings.jobs.max(1) {
        let next = Arc::clone(&next);
        let claimed = Arc::clone(&claimed);
        let records = Arc::clone(&records);
        let tsv = Arc::clone(&tsv);
        let failed_writes = Arc::clone(&failed_writes);
        let settings = settings.clone();
        workers.push(thread::spawn(move || {
            loop {
                if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                    break;
                }
                let index = next.fetch_add(1, Ordering::Relaxed);
                let profiles = &settings.profiles;
                let Some(seed) = settings.seeds.nth(index / profiles.len()) else {
                    break;
                };
                let profile = profiles[index % profiles.len()];
                claimed.fetch_add(1, Ordering::Relaxed);
                let record = run_seed(&settings, profile, seed).unwrap_or_else(|error| {
                    eprintln!("{profile} seed {seed}: harness error: {error}");
                    Record {
                        profile,
                        seed,
                        verdict: Verdict::HarnessError,
                        signature: vec![error],
                        modes_agree: false,
                        approximate_match: false,
                        jit_productive: false,
                        jit: None,
                        variant_jit: Vec::new(),
                        oracle_error: false,
                        stable: None,
                        features: BTreeSet::new(),
                        elapsed: Duration::ZERO,
                        artifact: None,
                    }
                });
                if record.verdict != Verdict::Pass && record.verdict != Verdict::Inconclusive {
                    eprintln!(
                        "{profile} seed {seed}: {} {}{}",
                        record.verdict.name(),
                        record.signature.join(","),
                        record
                            .artifact
                            .as_ref()
                            .map_or_else(String::new, |path| format!(" -> {}", path.display()))
                    );
                }
                let written = tsv
                    .lock()
                    .map_err(|_| "cases.tsv lock poisoned".to_owned())
                    .and_then(|mut file| {
                        file.write_all(record.tsv().as_bytes())
                            .map_err(|error| error.to_string())
                    });
                if let Err(error) = written {
                    eprintln!("cannot record {profile} seed {seed}: {error}");
                    failed_writes.fetch_add(1, Ordering::Relaxed);
                }
                let mut records = records
                    .lock()
                    .expect("no worker panics while holding the lock");
                records.push(record);
                let done = records.len();
                if done % 100 == 0 {
                    let total = total.map_or_else(String::new, |total| format!("/{total}"));
                    eprintln!(
                        "{done}{total} cases, {:.0}s",
                        started.elapsed().as_secs_f64()
                    );
                }
            }
        }));
    }
    for worker in workers {
        worker
            .join()
            .map_err(|_| "a campaign worker panicked".to_owned())?;
    }
    let mut records = Arc::try_unwrap(records)
        .map_err(|_| "workers still hold the records".to_owned())?
        .into_inner()
        .map_err(|_| "a worker poisoned the records".to_owned())?;
    records.sort_by_key(|record| (record.profile, record.seed));
    let claimed = claimed.load(Ordering::Relaxed);
    let expected = if deadline.is_none() {
        total.unwrap_or(claimed)
    } else {
        claimed
    };
    if records.len() != expected {
        return Err(format!(
            "{} of {expected} cases produced no record",
            expected.saturating_sub(records.len())
        ));
    }
    let failed_writes = failed_writes.load(Ordering::Relaxed);
    if failed_writes > 0 {
        return Err(format!(
            "{failed_writes} cases could not be written to cases.tsv"
        ));
    }
    let report = render_report(settings, &records, started.elapsed());
    let report_path = settings.out.join("report.md");
    fs::write(&report_path, &report)
        .map_err(|error| format!("{}: {error}", report_path.display()))?;
    print!("{report}");
    Ok(records)
}

fn run_seed(settings: &Settings, profile: Profile, seed: u64) -> Result<Record, String> {
    let started = Instant::now();
    let program = if settings.hot {
        generate_hot_seed(seed, profile, settings.size)
    } else {
        generate_excluding(seed, profile, settings.size, &settings.exclude)
    };
    let source = program.render();
    let work = settings.out.join("work").join(format!("{profile}-{seed}"));
    let oracle = settings
        .oracles
        .get(&profile)
        .and_then(|oracle| oracle.as_ref().ok());
    let result = run_case(&source, &work, profile, oracle, &settings.engines)?;
    let mut stable = None;
    let mut artifact = None;
    let keep = !matches!(result.verdict, Verdict::Pass | Verdict::Inconclusive)
        || (result.verdict == Verdict::Inconclusive && !result.modes_agree);
    if keep {
        let category = match result.verdict {
            Verdict::Mismatch => "failures",
            Verdict::Inconclusive => "disagreements",
            _ => "invalid",
        };
        let directory = settings
            .out
            .join(category)
            .join(format!("{profile}-{seed}"));
        if result.verdict == Verdict::Mismatch {
            stable = Some(check_stability(settings, profile, &work, &result));
        }
        let context = ArtifactContext {
            profile,
            seed: Some(seed),
            size: Some(settings.size),
            oracle,
            engines: &settings.engines,
            stable,
            hot: settings.hot,
        };
        save_artifact(&directory, &source, &program, &result, &context)?;
        if settings.reduce && result.verdict == Verdict::Mismatch {
            reduce_into(&directory, &program, profile, oracle, settings, &result)?;
        }
        artifact = Some(directory);
    }
    let _ = fs::remove_dir_all(&work);
    Ok(Record {
        profile,
        seed,
        verdict: result.verdict.clone(),
        signature: result.signature.clone(),
        modes_agree: result.modes_agree,
        approximate_match: settings.engines.comparison != Comparison::Exact
            && result.verdict == Verdict::Pass
            && result.oracle.as_ref().is_some_and(|oracle| {
                result.modes.iter().any(|run| {
                    difference(&oracle.observation, &run.observation).is_some()
                        && result.oracle_alternate.as_ref().is_none_or(|alternate| {
                            difference(&alternate.observation, &run.observation).is_some()
                        })
                })
            }),
        jit_productive: result.jit_productive(),
        jit: result
            .modes
            .iter()
            .find(|run| run.label == crate::case::Mode::Tracing.name())
            .and_then(|run| run.jit.clone())
            .map(|mut jit| {
                jit.raw.clear();
                jit
            }),
        variant_jit: result
            .modes
            .iter()
            .filter_map(|run| {
                run.jit.clone().map(|mut jit| {
                    jit.raw.clear();
                    (run.label.clone(), jit)
                })
            })
            .collect(),
        oracle_error: result
            .oracle
            .as_ref()
            .is_some_and(|run| matches!(run.observation.end, crate::case::End::Error(_))),
        stable,
        features: program.features(),
        elapsed: started.elapsed(),
        artifact,
    })
}

/// Re-runs each mismatching mode; the failure is stable if every rerun
/// prints exactly what the first run printed.
fn check_stability(
    settings: &Settings,
    profile: Profile,
    work: &Path,
    result: &CaseResult,
) -> bool {
    let failing: Vec<&EngineRun> = result
        .modes
        .iter()
        .filter(|run| {
            result
                .signature
                .iter()
                .any(|entry| entry.starts_with(&format!("{}:", run.label)))
        })
        .collect();
    for _ in 0..settings.reruns {
        for run in &failing {
            let Some(mode) = crate::case::Mode::parse(&run.label) else {
                continue;
            };
            let again = run_oximond(&settings.engines, profile, mode, work);
            if again.observation != run.observation {
                return false;
            }
        }
    }
    true
}

/// What an artifact records besides the runs themselves.
pub struct ArtifactContext<'a> {
    pub profile: Profile,
    pub seed: Option<u64>,
    pub size: Option<Size>,
    pub oracle: Option<&'a Oracle>,
    pub engines: &'a Engines,
    pub stable: Option<bool>,
    /// The seed names a hot-loop program (`--hot` replays it).
    pub hot: bool,
}

/// Writes the source, every engine's raw output and a `meta.txt` that
/// says how to replay the case.
///
/// # Errors
///
/// Returns an error if the directory cannot be written.
#[allow(clippy::too_many_lines, reason = "one line of meta.txt per field")]
pub fn save_artifact(
    directory: &Path,
    source: &str,
    program: &Program,
    result: &CaseResult,
    context: &ArtifactContext<'_>,
) -> Result<(), String> {
    let write = |name: &str, bytes: &[u8]| {
        let path = directory.join(name);
        fs::write(&path, bytes).map_err(|error| format!("{}: {error}", path.display()))
    };
    fs::create_dir_all(directory).map_err(|error| format!("{}: {error}", directory.display()))?;
    write(CASE_FILE, source.as_bytes())?;
    let mut meta = String::new();
    let _ = writeln!(meta, "generator_version: {GENERATOR_VERSION}");
    let _ = writeln!(
        meta,
        "seed_algorithm: splitmix64, stream = splitmix64(seed ^ profile_index << 56 ^ version << 48)"
    );
    if let Some(seed) = context.seed {
        let _ = writeln!(meta, "seed: {seed}");
    }
    let _ = writeln!(meta, "profile: {}", context.profile);
    let _ = writeln!(
        meta,
        "float_comparison: {}",
        comparison_description(context.engines.comparison)
    );
    if let Some(size) = context.size {
        let _ = writeln!(meta, "size: {}", size.name());
    }
    let _ = writeln!(meta, "verdict: {}", result.verdict.name());
    let _ = writeln!(meta, "signature: {}", result.signature.join(","));
    if let Some(stable) = context.stable {
        let _ = writeln!(meta, "stable_on_rerun: {stable}");
    }
    let _ = writeln!(meta, "modes_agree: {}", result.modes_agree);
    let _ = writeln!(
        meta,
        "features: {}",
        program.features().into_iter().collect::<Vec<_>>().join(",")
    );
    match context.oracle {
        Some(oracle) => {
            let _ = writeln!(
                meta,
                "oracle: {} ({})",
                oracle.command.display(),
                oracle.banner
            );
        }
        None => {
            let _ = writeln!(meta, "oracle: none");
        }
    }
    let _ = writeln!(meta, "oximond: {}", context.engines.oximond.display());
    let _ = writeln!(meta, "oximond_sha256: {}", sha256(&context.engines.oximond));
    let _ = writeln!(meta, "campaign_revision: {}", git_revision());
    let limits = context.engines.limits;
    let _ = writeln!(
        meta,
        "limits: timeout={}s memory={}MiB output_cap={}B",
        limits.timeout.as_secs_f64(),
        limits.memory_mib,
        limits.output_cap
    );
    let mut runs: Vec<&EngineRun> = result.oracle.iter().collect();
    runs.extend(result.oracle_alternate.iter());
    runs.extend(result.modes.iter());
    for run in &runs {
        let label = &run.label;
        write(&format!("{label}.stdout"), &run.outcome.stdout)?;
        write(&format!("{label}.stderr"), &run.outcome.stderr)?;
        let environment: Vec<String> = run
            .environment
            .iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect();
        let command: Vec<String> = run
            .command
            .iter()
            .map(|part| part.to_string_lossy().into_owned())
            .collect();
        let _ = writeln!(
            meta,
            "{label}.command: {} {}",
            environment.join(" "),
            command.join(" ")
        );
        let _ = writeln!(meta, "{label}.status: {}", run.outcome.status.describe());
        let _ = writeln!(meta, "{label}.end: {}", run.observation.end.describe());
        let _ = writeln!(meta, "{label}.millis: {}", run.outcome.elapsed.as_millis());
        if let Some(jit) = &run.jit {
            let _ = writeln!(
                meta,
                "{label}.jit: productive={} traces_compiled={} native_entries={} side_exits={}",
                jit.productive(),
                jit.traces_compiled,
                jit.native_entries,
                jit.side_exits
            );
        }
    }
    if let Some(oracle) = &result.oracle {
        let mut diff = String::new();
        for run in &result.modes {
            if let Some(line) = first_difference_with(
                &oracle.observation.stdout,
                &run.observation.stdout,
                context.engines.comparison,
            ) {
                let _ = writeln!(
                    diff,
                    "{} first differs at stdout line {}:",
                    run.label, line.line
                );
                let _ = writeln!(diff, "  oracle: {}", line.expected);
                let _ = writeln!(diff, "  {}: {}", run.label, line.actual);
            }
            if oracle.observation.end != run.observation.end {
                let _ = writeln!(
                    diff,
                    "{} ends with {}, oracle with {}",
                    run.label,
                    run.observation.end.describe(),
                    oracle.observation.end.describe()
                );
            }
        }
        write("diff.txt", diff.as_bytes())?;
    }
    if let Some(interpreter) = result
        .modes
        .iter()
        .find(|run| run.label == Mode::Interpreter.name())
    {
        // Each JIT mode against oximond's own interpreter, exactly.
        let mut diff = String::new();
        for run in result
            .modes
            .iter()
            .filter(|run| run.label != interpreter.label)
        {
            if let Some(line) =
                first_difference(&interpreter.observation.stdout, &run.observation.stdout)
            {
                let _ = writeln!(
                    diff,
                    "{} first differs from the interpreter at stdout line {}:",
                    run.label, line.line
                );
                let _ = writeln!(diff, "  interpreter: {}", line.expected);
                let _ = writeln!(diff, "  {}: {}", run.label, line.actual);
            }
            if interpreter.observation.end != run.observation.end {
                let _ = writeln!(
                    diff,
                    "{} ends with {}, interpreter with {}",
                    run.label,
                    run.observation.end.describe(),
                    interpreter.observation.end.describe()
                );
            }
        }
        if !diff.is_empty() {
            write("diff-vs-interpreter.txt", diff.as_bytes())?;
        }
    }
    let oximond = context.engines.oximond.display();
    let mut comparison_flags = comparison_arguments(context.engines.comparison);
    if context.oracle.is_some_and(|oracle| oracle.interpreter) {
        let modes: Vec<&str> = context
            .engines
            .modes
            .iter()
            .map(|mode| mode.name())
            .collect();
        let _ = write!(
            comparison_flags,
            " --interpreter-oracle --modes {}",
            modes.join(",")
        );
    }
    let hot_flag = if context.hot { " --hot" } else { "" };
    let _ = writeln!(
        meta,
        "replay_file: oximond-difftest replay-file {} --profile {} --oximond {oximond}{comparison_flags}",
        directory.join(CASE_FILE).display(),
        context.profile
    );
    if let (Some(seed), Some(size)) = (context.seed, context.size) {
        let _ = writeln!(
            meta,
            "replay_seed: oximond-difftest replay --seed {seed} --profile {} --size {}{hot_flag} --oximond {oximond}{comparison_flags} (generator {GENERATOR_VERSION} only)",
            context.profile,
            size.name()
        );
    }
    write("meta.txt", meta.as_bytes())
}

fn reduce_into(
    directory: &Path,
    program: &Program,
    profile: Profile,
    oracle: Option<&Oracle>,
    settings: &Settings,
    original: &CaseResult,
) -> Result<(), String> {
    let work = directory.join("reduce-work");
    let target: BTreeSet<String> = original.signature.iter().cloned().collect();
    // Only the failing modes need to run while reducing, and the interpreter
    // when a mode is judged against it.
    let against_interpreter = target
        .iter()
        .any(|entry| entry.contains(&format!(":{VS_INTERPRETER}:")));
    let mut engines = settings.engines.clone();
    engines.modes.retain(|mode| {
        (against_interpreter && *mode == Mode::Interpreter)
            || target
                .iter()
                .any(|entry| entry.starts_with(&format!("{}:", mode.name())))
    });
    let check = |candidate: &Program| {
        let Ok(result) = run_case(&candidate.render(), &work, profile, oracle, &engines) else {
            return false;
        };
        result.verdict == Verdict::Mismatch
            && result.signature.iter().any(|entry| target.contains(entry))
    };
    let reduction = reduce(
        program,
        check,
        settings.reduce_attempts,
        Instant::now() + settings.reduce_time,
    );
    let _ = fs::remove_dir_all(&work);
    let reduced = reduction.program.render();
    let minimized = directory.join("minimized");
    let result = run_case(&reduced, &minimized, profile, oracle, &engines)?;
    let context = ArtifactContext {
        profile,
        seed: None,
        size: None,
        oracle,
        engines: &engines,
        stable: None,
        hot: false,
    };
    save_artifact(&minimized, &reduced, &reduction.program, &result, &context)?;
    let mut log = format!(
        "attempts: {}\naccepted: {}\nexhausted: {}\noriginal_nodes: {}\nreduced_nodes: {}\ntarget_signature: {}\n",
        reduction.attempts,
        reduction.accepted,
        reduction.exhausted,
        program.size(),
        reduction.program.size(),
        target.into_iter().collect::<Vec<_>>().join(",")
    );
    for line in &reduction.log {
        let _ = writeln!(log, "{line}");
    }
    fs::write(directory.join("reduction.log"), log).map_err(|error| error.to_string())
}

/// Reduces a saved source file without its seed. The file is parsed back
/// into statements line by line at the top level only, so reduction is
/// coarser than for a seed.
#[must_use]
pub fn program_from_source(source: &str) -> Program {
    use crate::program::Node;
    let mut header = Vec::new();
    let mut nodes = Vec::new();
    for line in source.lines() {
        if nodes.is_empty()
            && let Some(comment) = line.strip_prefix("-- ")
        {
            header.push(comment.to_owned());
            continue;
        }
        nodes.push(Node::leaf(line));
    }
    Program { header, nodes }
}

fn sha256(path: &Path) -> String {
    std::process::Command::new("sha256sum")
        .arg(path)
        .output()
        .ok()
        .and_then(|output| {
            String::from_utf8_lossy(&output.stdout)
                .split_whitespace()
                .next()
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "unknown".to_owned())
}

fn git_revision() -> String {
    let revision = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_default();
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .is_ok_and(|output| !output.stdout.is_empty());
    if revision.is_empty() {
        "unknown".to_owned()
    } else if dirty {
        format!("{revision} (dirty)")
    } else {
        revision
    }
}

/// A signature with each statement tag cut to its letters
/// (`tracing:stdout@L12` becomes `tracing:stdout@L`).
fn coarse(signature: &[String]) -> String {
    let entries: BTreeSet<String> = signature
        .iter()
        .map(|entry| match entry.split_once('@') {
            Some((head, tag)) => {
                let letters: String = tag.chars().take_while(char::is_ascii_alphabetic).collect();
                format!("{head}@{letters}")
            }
            None => entry.clone(),
        })
        .collect();
    entries.into_iter().collect::<Vec<_>>().join(",")
}

#[allow(clippy::too_many_lines, reason = "one report section after another")]
fn render_report(settings: &Settings, records: &[Record], elapsed: Duration) -> String {
    let mut report = String::new();
    let _ = writeln!(report, "# oximond-difftest campaign\n");
    let _ = writeln!(report, "- generator version: {GENERATOR_VERSION}");
    let _ = writeln!(
        report,
        "- float comparison: {}",
        comparison_description(settings.engines.comparison)
    );
    let _ = writeln!(report, "- size: {}", settings.size.name());
    let excluded: Vec<&str> = settings.exclude.iter().map(String::as_str).collect();
    let _ = writeln!(
        report,
        "- excluded families: {}",
        if excluded.is_empty() {
            "none".to_owned()
        } else {
            excluded.join(", ")
        }
    );
    let ran = records
        .iter()
        .map(|record| record.seed)
        .collect::<BTreeSet<_>>()
        .len();
    match &settings.seeds {
        Seeds::List(seeds) => {
            let _ = writeln!(
                report,
                "- seeds: {} ({}..={}), {ran} ran",
                seeds.len(),
                seeds.first().copied().unwrap_or_default(),
                seeds.last().copied().unwrap_or_default()
            );
        }
        Seeds::Random { master } => {
            let _ = writeln!(
                report,
                "- seeds: {ran} random, master seed {master} (`--master-seed {master}` replays the order)"
            );
        }
    }
    if let Some(duration) = settings.duration {
        let _ = writeln!(
            report,
            "- duration: {:.0}s (no case started after it)",
            duration.as_secs_f64()
        );
    }
    let modes: Vec<&str> = settings
        .engines
        .modes
        .iter()
        .map(|mode| mode.name())
        .collect();
    let _ = writeln!(report, "- modes: {}", modes.join(", "));
    let _ = writeln!(
        report,
        "- oximond: {} (sha256 {})",
        settings.engines.oximond.display(),
        sha256(&settings.engines.oximond)
    );
    let _ = writeln!(report, "- revision: {}", git_revision());
    let _ = writeln!(
        report,
        "- elapsed: {:.1}s with {} jobs\n",
        elapsed.as_secs_f64(),
        settings.jobs
    );
    let _ = writeln!(report, "## Oracles\n");
    for (profile, oracle) in &settings.oracles {
        match oracle {
            Ok(oracle) => {
                let _ = writeln!(
                    report,
                    "- {profile}: {} ({})",
                    oracle.command.display(),
                    oracle.banner
                );
            }
            Err(error) => {
                let _ = writeln!(
                    report,
                    "- {profile}: none, cases are inconclusive ({error})"
                );
            }
        }
    }
    let _ = writeln!(report, "\n## Verdicts\n");
    let _ = writeln!(
        report,
        "| profile | attempted | pass | mismatch | oracle invalid | inconclusive (modes agree / disagree) | JIT productive | approximate matches |"
    );
    let _ = writeln!(report, "|---|---|---|---|---|---|---|---|");
    for profile in &settings.profiles {
        let rows: Vec<&Record> = records
            .iter()
            .filter(|record| record.profile == *profile)
            .collect();
        let count = |verdict: &Verdict| {
            rows.iter()
                .filter(|record| &record.verdict == verdict)
                .count()
        };
        let invalid = rows
            .iter()
            .filter(|record| {
                matches!(
                    record.verdict,
                    Verdict::OracleSyntaxError
                        | Verdict::OracleRuntimeError
                        | Verdict::OracleCrash
                        | Verdict::OracleTimeout
                )
            })
            .count();
        let agree = rows
            .iter()
            .filter(|record| record.verdict == Verdict::Inconclusive && record.modes_agree)
            .count();
        let disagree = count(&Verdict::Inconclusive) - agree;
        let productive = rows.iter().filter(|record| record.jit_productive).count();
        let approximate_matches = rows
            .iter()
            .filter(|record| record.approximate_match)
            .count();
        let _ = writeln!(
            report,
            "| {profile} | {} | {} | {} | {invalid} | {agree} / {disagree} | {productive} | {approximate_matches} |",
            rows.len(),
            count(&Verdict::Pass),
            count(&Verdict::Mismatch),
        );
    }
    let _ = writeln!(report, "\n## JIT activity (synchronous tracing runs)\n");
    let _ = writeln!(
        report,
        "| profile | cases with traces compiled | cases where native code ran | traces compiled | native entries | native iterations | exits to interpreter | guard failures | side traces installed |"
    );
    let _ = writeln!(report, "|---|---|---|---|---|---|---|---|---|");
    for profile in &settings.profiles {
        let runs: Vec<&crate::case::JitActivity> = records
            .iter()
            .filter(|record| record.profile == *profile)
            .filter_map(|record| record.jit.as_ref())
            .collect();
        let sum = |field: fn(&crate::case::JitActivity) -> u64| {
            runs.iter().map(|jit| field(jit)).sum::<u64>()
        };
        let _ = writeln!(
            report,
            "| {profile} | {} | {} | {} | {} | {} | {} | {} | {} |",
            runs.iter().filter(|jit| jit.traces_compiled > 0).count(),
            runs.iter().filter(|jit| jit.productive()).count(),
            sum(|jit| jit.traces_compiled),
            sum(|jit| jit.native_entries),
            sum(|jit| jit.iterations),
            sum(|jit| jit.exits),
            sum(|jit| jit.side_exits),
            sum(|jit| jit.side_installs),
        );
    }
    if records.iter().any(|record| !record.variant_jit.is_empty())
        && (settings
            .engines
            .modes
            .iter()
            .any(|mode| !Mode::ALL.contains(mode))
            || settings
                .oracles
                .values()
                .any(|oracle| oracle.as_ref().is_ok_and(|oracle| oracle.interpreter)))
    {
        report.push_str(&crate::jit::render_variant_tables(records));
    }
    let mut invalid_kinds: BTreeMap<&str, usize> = BTreeMap::new();
    for record in records {
        if !matches!(
            record.verdict,
            Verdict::Pass | Verdict::Mismatch | Verdict::Inconclusive
        ) {
            // Harness errors land here too, under their own name.
            *invalid_kinds.entry(record.verdict.name()).or_default() += 1;
        }
    }
    if !invalid_kinds.is_empty() {
        let _ = writeln!(
            report,
            "\nOracle-invalid cases by kind (generator or oracle problems, not oximond results):"
        );
        for (kind, count) in invalid_kinds {
            let _ = writeln!(report, "- {kind}: {count}");
        }
    }
    let _ = writeln!(report, "\n## Failure signatures\n");
    let exact: BTreeSet<(Profile, String)> = records
        .iter()
        .filter(|record| record.verdict == Verdict::Mismatch)
        .map(|record| (record.profile, record.signature.join(",")))
        .collect();
    let _ = writeln!(
        report,
        "{} exact signatures. Grouped below by mode, category and observation kind (the tag letters); \
         one group can hold several root causes, so reduce a few seeds of each.\n",
        exact.len()
    );
    let mut signatures: BTreeMap<(Profile, String), Vec<&Record>> = BTreeMap::new();
    for record in records
        .iter()
        .filter(|record| record.verdict == Verdict::Mismatch)
    {
        signatures
            .entry((record.profile, coarse(&record.signature)))
            .or_default()
            .push(record);
    }
    if signatures.is_empty() {
        let _ = writeln!(report, "None.");
    }
    for ((profile, signature), rows) in &signatures {
        let flaky = rows
            .iter()
            .filter(|record| record.stable == Some(false))
            .count();
        let examples: Vec<String> = rows
            .iter()
            .take(5)
            .map(|record| record.seed.to_string())
            .collect();
        let _ = writeln!(
            report,
            "- {profile} `{signature}`: {} cases ({flaky} flaky), seeds {}",
            rows.len(),
            examples.join(", ")
        );
    }
    let _ = writeln!(report, "\n## Feature ledger\n");
    let _ = writeln!(
        report,
        "Source presence among cases the oracle accepted (or all cases without an oracle). Native activity is measured for the whole case, not for individual features. A feature may be cold or never execute even when another loop compiles. Use run-workloads for isolated hot workloads.\n"
    );
    let mut ledger: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for record in records {
        let valid = matches!(
            record.verdict,
            Verdict::Pass | Verdict::Mismatch | Verdict::Inconclusive
        );
        if !valid {
            continue;
        }
        for feature in &record.features {
            let entry = ledger.entry(feature).or_default();
            entry.0 += 1;
            if record.jit_productive {
                entry.1 += 1;
            }
        }
    }
    let _ = writeln!(
        report,
        "| feature in source | cases | cases with native activity anywhere |"
    );
    let _ = writeln!(report, "|---|---|---|");
    for (feature, (cases, productive)) in ledger {
        let _ = writeln!(report, "| {feature} | {cases} | {productive} |");
    }
    report
}

/// A regression-corpus entry: a saved source with its expected output.
#[derive(Clone, Debug)]
pub struct CorpusEntry {
    pub name: String,
    pub profile: Profile,
    pub source: String,
    pub expected_stdout: Vec<u8>,
    /// A known oximond failure: the exact `mode:category` entries it is
    /// expected to fail with, from `NAME.known-failure`. Any other failure,
    /// a new failing mode, or passing reports a problem.
    pub known_failure: Option<BTreeSet<String>>,
}

/// Reads every `NAME.lua` with its `NAME.stdout` in `directory`. The
/// profile comes from the `profile=` field of the first comment line.
///
/// # Errors
///
/// Returns an error for an unreadable entry or a missing profile.
pub fn read_corpus(directory: &Path) -> Result<Vec<CorpusEntry>, String> {
    let mut entries = Vec::new();
    let mut paths: Vec<PathBuf> = fs::read_dir(directory)
        .map_err(|error| format!("{}: {error}", directory.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "lua"))
        .collect();
    paths.sort();
    for path in paths {
        let source =
            fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        let profile = source
            .lines()
            .next()
            .and_then(|line| {
                line.split_whitespace()
                    .find_map(|field| field.strip_prefix("profile="))
            })
            .and_then(Profile::parse)
            .ok_or_else(|| format!("{}: first line has no profile=", path.display()))?;
        let expected_path = path.with_extension("stdout");
        let expected_stdout = fs::read(&expected_path)
            .map_err(|error| format!("{}: {error}", expected_path.display()))?;
        let known_failure = fs::read_to_string(path.with_extension("known-failure"))
            .ok()
            .map(|text| {
                text.lines()
                    .map(str::trim)
                    .filter(|line| !line.is_empty() && !line.starts_with('#'))
                    .map(str::to_owned)
                    .collect()
            });
        entries.push(CorpusEntry {
            name: path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            profile,
            source,
            expected_stdout,
            known_failure,
        });
    }
    Ok(entries)
}

/// Generates `seed` for `profile`, records the oracle's output and saves
/// both as a corpus entry named `name`.
///
/// # Errors
///
/// Returns an error if the oracle does not run the program successfully.
pub fn add_to_corpus(
    directory: &Path,
    name: &str,
    source: &str,
    oracle: &Oracle,
    limits: crate::run::Limits,
) -> Result<(), String> {
    let work = directory.join(format!(".{name}-work"));
    fs::create_dir_all(&work).map_err(|error| error.to_string())?;
    fs::write(work.join(CASE_FILE), source).map_err(|error| error.to_string())?;
    let run = crate::case::run_oracle(oracle, &work, limits);
    let _ = fs::remove_dir_all(&work);
    if run.observation.end != crate::case::End::Success {
        return Err(format!(
            "the oracle ended with {}",
            run.observation.end.describe()
        ));
    }
    fs::write(directory.join(format!("{name}.lua")), source).map_err(|error| error.to_string())?;
    fs::write(
        directory.join(format!("{name}.stdout")),
        &run.outcome.stdout,
    )
    .map_err(|error| error.to_string())
}

/// Runs every corpus entry in every mode against its recorded output.
/// Returns one line per problem.
///
/// # Errors
///
/// Returns an error if the corpus cannot be read.
pub fn check_corpus(
    directory: &Path,
    engines: &Engines,
    scratch: &Path,
) -> Result<Vec<String>, String> {
    let mut problems = Vec::new();
    for entry in read_corpus(directory)? {
        let work = scratch.join(&entry.name);
        fs::create_dir_all(&work).map_err(|error| error.to_string())?;
        fs::write(work.join(CASE_FILE), &entry.source).map_err(|error| error.to_string())?;
        let expected = crate::case::Observation {
            stdout: entry.expected_stdout.clone(),
            end: crate::case::End::Success,
        };
        let mut failing = Vec::new();
        let mut signature = BTreeSet::new();
        for &mode in &engines.modes {
            let run = run_oximond(engines, entry.profile, mode, &work);
            if let Some(category) = difference_with(&expected, &run.observation, engines.comparison)
            {
                signature.insert(format!("{}:{category}", mode.name()));
                let detail = first_difference_with(
                    &entry.expected_stdout,
                    &run.observation.stdout,
                    engines.comparison,
                )
                .map_or_else(
                    || run.observation.end.describe(),
                    |line| {
                        format!(
                            "line {}: expected {} got {}",
                            line.line, line.expected, line.actual
                        )
                    },
                );
                failing.push(format!(
                    "{} {} {category}: {detail}",
                    entry.name,
                    mode.name()
                ));
            }
        }
        let _ = fs::remove_dir_all(&work);
        match &entry.known_failure {
            None => problems.extend(failing),
            Some(known) if signature.is_empty() => problems.push(format!(
                "{}: known failure {} now passes; remove its .known-failure",
                entry.name,
                known.iter().cloned().collect::<Vec<_>>().join(",")
            )),
            Some(known) if *known != signature => problems.push(format!(
                "{}: known failure changed from {} to {}",
                entry.name,
                known.iter().cloned().collect::<Vec<_>>().join(","),
                signature.iter().cloned().collect::<Vec<_>>().join(",")
            )),
            Some(_) => {}
        }
    }
    Ok(problems)
}

fn comparison_description(comparison: Comparison) -> String {
    match comparison {
        Comparison::Exact => "exact".to_owned(),
        Comparison::Approximate { absolute, relative } => {
            format!("approximate (absolute={absolute}, relative={relative})")
        }
    }
}

fn comparison_arguments(comparison: Comparison) -> String {
    match comparison {
        Comparison::Exact => String::new(),
        Comparison::Approximate { absolute, relative } => {
            format!(
                " --float-compare approximate --float-abs-tol {absolute} --float-rel-tol {relative}"
            )
        }
    }
}
