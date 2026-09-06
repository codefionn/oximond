//! External test-corpus acquisition and compatibility-ledger support.
//!
//! The corpora are intentionally kept outside the source tree. This crate only
//! records their pins and prepares local copies for a conformance run.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const LUA51: &str = "lua51";
const LUA52: &str = "lua52";
const LUA53: &str = "lua53";
const LUA54: &str = "lua54";
const LUA55: &str = "lua55";
const LUAJIT: &str = "luajit";
const DEFAULT_TEST_TIMEOUT: Duration = Duration::from_mins(1);
/// Address-space cap per test process. Upstream stress files such as
/// `heavy.lua` grow tables until allocation fails, which would otherwise
/// take the whole machine with them.
const DEFAULT_TEST_MEMORY_LIMIT_MIB: u64 = 4 * 1024;
const CHILD_POLL_INTERVAL: Duration = Duration::from_millis(10);
const REQUIRED_SOURCE_FIELDS: &[&str] = &["corpus", "kind", "url", "pin", "archive", "directory"];
const REQUIRED_LEDGER_FIELDS: &[&str] = &["corpus", "path", "state", "milestone", "reason"];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Source {
    pub corpus: String,
    pub kind: String,
    pub url: String,
    pub pin: String,
    pub archive: String,
    pub directory: String,
    /// A checked-in unified diff, relative to the source manifest.
    pub patches: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Sources {
    sources: Vec<Source>,
    manifest_directory: Option<PathBuf>,
}

impl Sources {
    /// Loads and validates a source manifest.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or is not a valid manifest.
    pub fn load(path: &Path) -> Result<Self, String> {
        let path = absolute_path(path)?;
        let contents = fs::read_to_string(&path)
            .map_err(|error| format!("cannot read source manifest {}: {error}", path.display()))?;
        let mut sources = Self::parse(&contents)?;
        sources.manifest_directory = path.parent().map(Path::to_path_buf);
        Ok(sources)
    }

    /// Parses and validates a source manifest.
    /// Use [`Self::load`] when preparing sources with patch paths, which are
    /// resolved relative to the manifest file.
    ///
    /// # Errors
    ///
    /// Returns an error when required fields are absent or corpus identifiers repeat.
    pub fn parse(contents: &str) -> Result<Self, String> {
        let records = parse_records(contents, "source")?;
        let mut sources = Vec::with_capacity(records.len());
        for record in records {
            require_fields(&record, REQUIRED_SOURCE_FIELDS, "source")?;
            sources.push(Source {
                corpus: record["corpus"].clone(),
                kind: record["kind"].clone(),
                url: record["url"].clone(),
                pin: record["pin"].clone(),
                archive: record["archive"].clone(),
                directory: record["directory"].clone(),
                patches: record.get("patches").cloned(),
            });
        }

        if sources.is_empty() {
            return Err("source manifest must define at least one source.".to_owned());
        }
        let mut corpora = BTreeSet::new();
        for source in &sources {
            if source.corpus.is_empty() {
                return Err("source manifest corpus must not be empty.".to_owned());
            }
            if source.patches.as_deref().is_some_and(str::is_empty) {
                return Err(format!(
                    "source manifest patch path for `{}` must not be empty.",
                    source.corpus
                ));
            }
            if !corpora.insert(&source.corpus) {
                return Err(format!(
                    "source manifest repeats corpus `{}`.",
                    source.corpus
                ));
            }
        }
        if sources
            .iter()
            .any(|source| source.kind != "archive" && source.kind != "git")
        {
            return Err("source manifest kind must be `archive` or `git`.".to_owned());
        }
        Ok(Self {
            sources,
            manifest_directory: None,
        })
    }

    #[must_use]
    pub fn get(&self, corpus: &str) -> Option<&Source> {
        self.sources.iter().find(|source| source.corpus == corpus)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Source> {
        self.sources.iter()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LedgerEntry {
    pub corpus: String,
    pub path: String,
    pub state: String,
    pub milestone: String,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ledger {
    entries: Vec<LedgerEntry>,
}

impl Ledger {
    /// Loads and parses a compatibility ledger.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or is not a valid ledger.
    pub fn load(path: &Path) -> Result<Self, String> {
        let contents = fs::read_to_string(path)
            .map_err(|error| format!("cannot read ledger {}: {error}", path.display()))?;
        Self::parse(&contents)
    }

    /// Parses a compatibility ledger.
    ///
    /// # Errors
    ///
    /// Returns an error when an entry is malformed.
    pub fn parse(contents: &str) -> Result<Self, String> {
        let records = parse_records(contents, "entry")?;
        let mut entries = Vec::with_capacity(records.len());
        for record in records {
            require_fields(&record, REQUIRED_LEDGER_FIELDS, "ledger entry")?;
            entries.push(LedgerEntry {
                corpus: record["corpus"].clone(),
                path: record["path"].clone(),
                state: record["state"].clone(),
                milestone: record["milestone"].clone(),
                reason: record["reason"].clone(),
            });
        }
        Ok(Self { entries })
    }

    /// Checks ledger policy against the declared sources.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown corpus, an invalid state, duplicate paths,
    /// an empty required field, or a missing `LuaJIT` bootstrap entry.
    pub fn validate(&self, sources: &Sources) -> Result<(), String> {
        if self.entries.is_empty() {
            return Err("ledger must have at least one entry.".to_owned());
        }
        let mut seen = BTreeSet::new();
        for entry in &self.entries {
            if sources.get(&entry.corpus).is_none() {
                return Err(format!("ledger names unknown corpus `{}`.", entry.corpus));
            }
            if !matches!(
                entry.state.as_str(),
                "required" | "deferred" | "expected-fail"
            ) {
                return Err(format!(
                    "ledger entry `{}` has invalid state `{}`.",
                    entry.path, entry.state
                ));
            }
            if entry.path.is_empty() || entry.reason.is_empty() || entry.milestone.is_empty() {
                return Err(format!(
                    "ledger entry `{}` has an empty required field.",
                    entry.path
                ));
            }
            if !seen.insert((&entry.corpus, &entry.path)) {
                return Err(format!(
                    "ledger has duplicate entry `{}` in {}.",
                    entry.path, entry.corpus
                ));
            }
        }
        let first_luajit = self.entries.iter().find(|entry| entry.corpus == LUAJIT);
        match first_luajit {
            Some(entry)
                if entry.path == "test/lang/assignment.lua" && entry.state == "required" =>
            {
                Ok(())
            }
            Some(_) => Err(
                "the first LuaJIT ledger entry must require test/lang/assignment.lua.".to_owned(),
            ),
            None => Err("ledger has no LuaJIT entries.".to_owned()),
        }
    }

    pub fn entries_for(&self, corpus: &str) -> impl Iterator<Item = &LedgerEntry> {
        self.entries
            .iter()
            .filter(move |entry| entry.corpus == corpus)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SourceStatus {
    Ready,
    Missing,
    Invalid(String),
}

/// Result of running the exact executable ledger entries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunResult {
    report: String,
    failures: Vec<String>,
}

impl RunResult {
    /// Returns the separate corpus result sections.
    #[must_use]
    pub fn report(&self) -> &str {
        &self.report
    }

    /// Returns whether no required, stale-ledger, or infrastructure failure occurred.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }

    /// Formats runner failures for a human diagnostic.
    #[must_use]
    pub fn failure_message(&self) -> String {
        self.failures.join("; ")
    }
}

/// Downloads and validates each external corpus beneath `root`.
///
/// # Errors
///
/// Returns an error when a directory cannot be created, a process fails, or a pin
/// does not verify, or a declared patch cannot be applied. Preparation attempts
/// every corpus before returning.
pub fn prepare(sources: &Sources, root: &Path) -> Result<(), String> {
    fs::create_dir_all(root)
        .map_err(|error| format!("cannot create corpus directory {}: {error}", root.display()))?;
    let mut failures = Vec::new();
    for source in sources.iter() {
        let result = match source.kind.as_str() {
            "archive" => prepare_archive(source, root),
            "git" => prepare_git(source, root),
            _ => unreachable!("source manifest validation permits only archive or git"),
        }
        .and_then(|()| rebuild_patched_tree(source, root, sources.manifest_directory.as_deref()));
        match result {
            Ok(()) => println!("{}: ready", source.corpus),
            Err(error) => {
                eprintln!("{}: {error}", source.corpus);
                failures.push(format!("{}: {error}", source.corpus));
            }
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "could not prepare all corpora: {}",
            failures.join("; ")
        ))
    }
}

/// Renders a report with one section for every declared corpus.
///
/// # Errors
///
/// Returns an error when the ledger violates its policy.
pub fn render_report(sources: &Sources, ledger: &Ledger, root: &Path) -> Result<String, String> {
    ledger.validate(sources)?;
    let mut report = String::from("Oximond conformance report\n");
    for source in sources.iter() {
        let corpus = &source.corpus;
        let status = source_status(source, root);
        write!(
            report,
            "\n[{corpus}]\nsource = {}\npin = {}\nstatus = {}\n",
            source.url,
            source.pin,
            status_label(&status)
        )
        .expect("writing to a string cannot fail");
        if let Some(patch) = &source.patches {
            writeln!(report, "patch = {patch}").expect("writing to a string cannot fail");
            if let Some(manifest_directory) = sources.manifest_directory.as_deref() {
                let resolved = resolve_patch_path(manifest_directory, patch)?;
                let hash = sha256_file(&resolved).map_err(|error| {
                    format!(
                        "cannot hash {} patch {}: {error}",
                        source.corpus,
                        resolved.display()
                    )
                })?;
                writeln!(report, "patch-sha256 = {hash}").expect("writing to a string cannot fail");
            }
        }
        let mut counts = BTreeMap::<&str, usize>::new();
        for entry in ledger.entries_for(corpus) {
            *counts.entry(&entry.state).or_default() += 1;
        }
        for state in ["required", "expected-fail", "deferred"] {
            writeln!(
                report,
                "{state} = {}",
                counts.get(state).copied().unwrap_or(0)
            )
            .expect("writing to a string cannot fail");
        }
        if let SourceStatus::Invalid(reason) = status {
            writeln!(report, "detail = {reason}").expect("writing to a string cannot fail");
        }
    }
    Ok(report)
}

/// Runs every exact-path executable ledger entry with the given Oximond executable.
///
/// All corpora must already verify as prepared. Declared patches are reapplied
/// to fresh copies of the pinned inputs before execution. Required entries must pass.
/// Expected-fail entries must return a nonzero status. An expected-fail success
/// is a stale ledger entry and makes [`RunResult::passed`] return `false`.
///
/// # Errors
///
/// Returns an error when the ledger is invalid, a corpus is not prepared, or a
/// executable entry contains a wildcard path.
pub fn run_required(
    sources: &Sources,
    ledger: &Ledger,
    root: &Path,
    oximond: &Path,
) -> Result<RunResult, String> {
    ledger.validate(sources)?;
    require_prepared(sources, root)?;
    let executable = absolute_path(oximond)?;
    if !executable.is_file() {
        return Err(format!(
            "Oximond executable {} does not exist. Build it or pass `--oximond PATH`.",
            executable.display()
        ));
    }
    validate_prepared_ledger_coverage(ledger, sources, root)?;
    // Rebuilding replaces each `patched` tree, which would pull the working
    // directory out from under a concurrent run's tests.
    let _run_lock = lock_run_directory(root)?;
    rebuild_patched_trees(sources, root)?;
    validate_patched_ledger_paths(ledger, sources, root)?;
    build_native_cli_fixtures(ledger, sources, root)?;
    let patch_provenance = render_patch_provenance(sources)?;
    let root = absolute_path(root)?;
    let counts = run_ledger_entries(
        ledger,
        sources,
        &root,
        |corpus, path, working_directory, corpus_root| {
            execute_test(&executable, corpus, path, working_directory, corpus_root)
        },
    )?;
    Ok(RunResult {
        report: format!(
            "{}\n[executable]\npath = {}\n{}",
            render_run_report(&counts),
            executable.display(),
            patch_provenance
        ),
        failures: counts.failures,
    })
}

fn build_native_cli_fixtures(
    ledger: &Ledger,
    sources: &Sources,
    root: &Path,
) -> Result<(), String> {
    for source in sources.iter() {
        if !matches!(source.corpus.as_str(), LUA54 | LUA55)
            || !ledger
                .entries_for(&source.corpus)
                .any(|entry| entry.path == "main.lua" && entry.state != "deferred")
        {
            continue;
        }
        let tree = if source.patches.is_some() {
            "patched"
        } else {
            "source"
        };
        build_native_cli_fixture(&root.join(&source.directory).join(tree))?;
    }
    Ok(())
}

fn build_native_cli_fixture(corpus: &Path) -> Result<(), String> {
    // These are public API headers only. The module resolves its API calls
    // against Oximond, and never links a stock Lua interpreter.
    let headers = corpus.join(".native-headers");
    fs::create_dir_all(&headers)
        .map_err(|error| format!("cannot create {}: {error}", headers.display()))?;
    for (name, content) in [
        (
            "lua.h",
            include_str!("../../oximond-capi/vendor/lua51/lua.h"),
        ),
        (
            "luaconf.h",
            include_str!("../../oximond-capi/vendor/lua51/luaconf.h"),
        ),
        (
            "lauxlib.h",
            include_str!("../../oximond-capi/vendor/lua51/lauxlib.h"),
        ),
        (
            "modern-module.h",
            include_str!("../../oximond-capi/vendor/lua51/modern-module.h"),
        ),
    ] {
        fs::write(headers.join(name), content)
            .map_err(|error| format!("cannot write native module header {name}: {error}"))?;
    }
    let compiler = std::env::var_os("CC").unwrap_or_else(|| OsString::from("cc"));
    run_process(
        Command::new(compiler)
            .args(["-shared", "-fPIC", "-O2"])
            .arg("-I")
            .arg(&headers)
            .arg("-include")
            .arg(headers.join("modern-module.h"))
            .arg(corpus.join("libs/lib22.c"))
            .arg("-o")
            .arg(corpus.join("libs/lib2-v2.so")),
        &format!("build native CLI fixture in {}", corpus.display()),
    )
}

fn lock_run_directory(root: &Path) -> Result<fs::File, String> {
    let path = root.join(".run.lock");
    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .map_err(|error| format!("cannot open run lock {}: {error}", path.display()))?;
    match file.try_lock() {
        Ok(()) => return Ok(file),
        Err(fs::TryLockError::WouldBlock) => {
            eprintln!("oximond-conformance: waiting for another run to finish");
        }
        Err(fs::TryLockError::Error(error)) => {
            return Err(format!("cannot lock {}: {error}", path.display()));
        }
    }
    file.lock()
        .map_err(|error| format!("cannot lock {}: {error}", path.display()))?;
    Ok(file)
}

fn render_patch_provenance(sources: &Sources) -> Result<String, String> {
    let Some(manifest_directory) = sources.manifest_directory.as_deref() else {
        return Ok(String::new());
    };
    let mut report = String::new();
    for source in sources.iter() {
        let Some(patch) = &source.patches else {
            continue;
        };
        let resolved = resolve_patch_path(manifest_directory, patch)?;
        let hash = sha256_file(&resolved).map_err(|error| {
            format!(
                "cannot hash {} patch {}: {error}",
                source.corpus,
                resolved.display()
            )
        })?;
        writeln!(
            report,
            "\n[{} patch]\npath = {patch}\nsha256 = {hash}",
            source.corpus
        )
        .expect("writing to a string cannot fail");
    }
    Ok(report)
}

fn rebuild_patched_trees(sources: &Sources, root: &Path) -> Result<(), String> {
    for source in sources.iter() {
        rebuild_patched_tree(source, root, sources.manifest_directory.as_deref())?;
    }
    Ok(())
}

fn validate_patched_ledger_paths(
    ledger: &Ledger,
    sources: &Sources,
    root: &Path,
) -> Result<(), String> {
    for entry in &ledger.entries {
        if has_wildcard(&entry.path) {
            continue;
        }
        let source = sources
            .get(&entry.corpus)
            .ok_or_else(|| format!("source `{}` is missing", entry.corpus))?;
        if source.patches.is_none() {
            continue;
        }
        let path = root
            .join(&source.directory)
            .join("patched")
            .join(&entry.path);
        if !path.is_file() {
            return Err(format!(
                "patch for {} removed ledger test `{}`; patched suites must retain every classified test",
                source.corpus, entry.path
            ));
        }
    }
    Ok(())
}

fn validate_prepared_ledger_coverage(
    ledger: &Ledger,
    sources: &Sources,
    root: &Path,
) -> Result<(), String> {
    for entry in &ledger.entries {
        if has_wildcard(&entry.path) {
            continue;
        }
        let source = sources
            .get(&entry.corpus)
            .ok_or_else(|| format!("source `{}` is missing", entry.corpus))?;
        let path = root
            .join(&source.directory)
            .join("source")
            .join(&entry.path);
        if !path.is_file() {
            return Err(format!(
                "ledger entry `{}` in {} does not name a prepared file",
                entry.path, entry.corpus
            ));
        }
    }

    for source in sources.iter().filter(|source| source.kind == "archive") {
        let directory = root.join(&source.directory).join("source");
        let classified: BTreeSet<&str> = ledger
            .entries_for(&source.corpus)
            .filter(|entry| !has_wildcard(&entry.path))
            .map(|entry| entry.path.as_str())
            .collect();
        for entry in fs::read_dir(&directory)
            .map_err(|error| format!("cannot enumerate {}: {error}", directory.display()))?
        {
            let entry = entry
                .map_err(|error| format!("cannot enumerate {}: {error}", directory.display()))?;
            let file_type = entry
                .file_type()
                .map_err(|error| format!("cannot inspect {}: {error}", entry.path().display()))?;
            if !file_type.is_file() || entry.path().extension().is_none_or(|value| value != "lua") {
                continue;
            }
            let name = entry.file_name().into_string().map_err(|_| {
                format!(
                    "{} suite contains a non-UTF-8 test filename: {}",
                    source.corpus,
                    entry.path().display()
                )
            })?;
            if !classified.contains(name.as_str()) {
                return Err(format!(
                    "{} test `{name}` has no exact compatibility-ledger entry",
                    source.corpus
                ));
            }
        }
    }
    Ok(())
}

#[derive(Debug, Default)]
struct RunCounts {
    corpora: Vec<String>,
    passed: BTreeMap<String, usize>,
    failed: BTreeMap<String, usize>,
    expected_failures: BTreeMap<String, usize>,
    unexpected_passes: BTreeMap<String, usize>,
    infrastructure_errors: BTreeMap<String, usize>,
    self_skipped: BTreeMap<String, usize>,
    deferred: BTreeMap<String, usize>,
    failures: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum TestExecution {
    Passed,
    Nonzero(String),
    Infrastructure(String),
    SelfSkipped(String),
}

fn require_prepared(sources: &Sources, root: &Path) -> Result<(), String> {
    let mut missing = Vec::new();
    for source in sources.iter() {
        match source_status(source, root) {
            SourceStatus::Ready => {}
            SourceStatus::Missing => missing.push(format!("{} is missing", source.corpus)),
            SourceStatus::Invalid(reason) => {
                missing.push(format!("{} is invalid: {reason}", source.corpus));
            }
        }
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "corpora are not prepared. Run `oximond-conformance prepare`: {}",
            missing.join("; ")
        ))
    }
}

fn run_ledger_entries<F>(
    ledger: &Ledger,
    sources: &Sources,
    root: &Path,
    mut execute: F,
) -> Result<RunCounts, String>
where
    F: FnMut(&str, &Path, &Path, &Path) -> TestExecution,
{
    let mut counts = RunCounts::default();
    for source in sources.iter() {
        let corpus = source.corpus.as_str();
        counts.corpora.push(corpus.to_owned());
        let tree = if source.patches.is_some() {
            "patched"
        } else {
            "source"
        };
        let corpus_root = root.join(&source.directory).join(tree);
        for entry in ledger.entries_for(corpus) {
            if entry.state == "deferred" {
                *counts.deferred.entry(corpus.to_owned()).or_default() += 1;
                continue;
            }
            if has_wildcard(&entry.path) {
                return Err(format!(
                    "{} entry `{}` in {corpus} has a wildcard path. Exact paths are required.",
                    entry.state, entry.path
                ));
            }
            let path = corpus_root.join(&entry.path);
            let working_directory = path.parent().ok_or_else(|| {
                format!(
                    "test path `{}` does not have a parent directory",
                    path.display()
                )
            })?;
            match (
                entry.state.as_str(),
                execute(corpus, &path, working_directory, &corpus_root),
            ) {
                ("required", TestExecution::Passed) => {
                    *counts.passed.entry(corpus.to_owned()).or_default() += 1;
                }
                ("required", TestExecution::Nonzero(error)) => {
                    *counts.failed.entry(corpus.to_owned()).or_default() += 1;
                    counts.failures.push(error);
                }
                ("expected-fail", TestExecution::Passed) => {
                    *counts
                        .unexpected_passes
                        .entry(corpus.to_owned())
                        .or_default() += 1;
                    counts.failures.push(format!(
                        "{} unexpectedly passed; remove or update its expected-fail ledger entry",
                        path.display()
                    ));
                }
                ("expected-fail", TestExecution::Nonzero(_)) => {
                    *counts
                        .expected_failures
                        .entry(corpus.to_owned())
                        .or_default() += 1;
                }
                (_, TestExecution::Infrastructure(error)) => {
                    *counts
                        .infrastructure_errors
                        .entry(corpus.to_owned())
                        .or_default() += 1;
                    counts.failures.push(error);
                }
                (_, TestExecution::SelfSkipped(error)) => {
                    *counts.self_skipped.entry(corpus.to_owned()).or_default() += 1;
                    counts.failures.push(error);
                }
                (state, _) => {
                    return Err(format!("unsupported executable ledger state `{state}`."));
                }
            }
        }
    }
    Ok(counts)
}

static EXECUTION_ARGUMENTS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
static TRACE_PROFILE_DIRECTORIES: std::sync::OnceLock<(Option<PathBuf>, Option<PathBuf>)> =
    std::sync::OnceLock::new();

/// Sets extra command-line arguments (for example `--execution tracing`) passed
/// to the Oximond executable before each test. Only the first call has effect.
pub fn set_execution_arguments(arguments: Vec<String>) {
    let _ = EXECUTION_ARGUMENTS.set(arguments);
}

/// Selects separate profile files for each executable ledger entry.
pub fn set_trace_profile_directories(record: Option<PathBuf>, load: Option<PathBuf>) {
    let _ = TRACE_PROFILE_DIRECTORIES.set((record, load));
}

fn trace_profile_path(directory: &Path, corpus: &str, path: &Path, corpus_root: &Path) -> PathBuf {
    let relative = path.strip_prefix(corpus_root).unwrap_or(path);
    let mut profile = directory.join(corpus).join(relative);
    let filename = profile.file_name().unwrap_or_default().to_string_lossy();
    profile.set_file_name(format!("{filename}.trace-profile"));
    profile
}

fn execute_test(
    executable: &Path,
    corpus: &str,
    path: &Path,
    working_directory: &Path,
    corpus_root: &Path,
) -> TestExecution {
    let launch = TestLaunch::new(working_directory, corpus_root);
    let mut command = memory_limited_command(executable, configured_test_memory_limit_mib());
    command.arg("--compat").arg(compatibility_selector(corpus));
    if let Some(arguments) = EXECUTION_ARGUMENTS.get() {
        command.args(arguments);
    }
    if let Some((record, load)) = TRACE_PROFILE_DIRECTORIES.get() {
        if let Some(directory) = load {
            command.arg("--trace-profile-load").arg(trace_profile_path(
                directory,
                corpus,
                path,
                corpus_root,
            ));
        }
        if let Some(directory) = record {
            let profile = trace_profile_path(directory, corpus, path, corpus_root);
            if let Some(parent) = profile.parent()
                && let Err(error) = std::fs::create_dir_all(parent)
            {
                return TestExecution::Infrastructure(format!(
                    "cannot create {}: {error}",
                    parent.display()
                ));
            }
            command.arg("--trace-profile-record").arg(profile);
        }
    }
    if matches!(corpus, LUA52 | LUA53 | LUA54 | LUA55)
        && path.file_name().is_some_and(|name| name == "all.lua")
    {
        // This is the upstream invocation for the portable test run. It makes
        // the aggregate skip the internal C fixture and non-portable checks.
        command.args(["-e", "_U=true"]);
    }
    if corpus == LUA55 && path.file_name().is_some_and(|name| name == "goto.lua") {
        // Lua 5.5's suite uses this upstream switch to test `global` as a
        // reserved word. Without it, the file tests the optional lexer mode.
        command.args(["-e", "T=true"]);
    }
    command.arg(path).current_dir(&launch.working_directory);
    if launch.clears_host_lua_path {
        command.env_remove("LUA_PATH");
    }
    command.env("LUA_PATH", &launch.lua_path);
    match capture_output(&mut command, configured_test_timeout()) {
        Ok(output) if output.status.success() => {
            classify_successful_test_output(path, &output.stdout, &output.stderr)
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let detail = stderr.trim();
            let detail = if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            };
            TestExecution::Nonzero(format!(
                "{} exited with {}{detail}",
                path.display(),
                output.status
            ))
        }
        Err(CaptureError::Launch(error)) => TestExecution::Infrastructure(format!(
            "cannot launch {} for {}: {error}",
            executable.display(),
            path.display()
        )),
        Err(CaptureError::TimedOut { timeout, stderr }) => {
            let detail = String::from_utf8_lossy(&stderr);
            let detail = detail.trim();
            let detail = if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            };
            TestExecution::Nonzero(format!(
                "{} timed out after {} seconds{detail}",
                path.display(),
                timeout.as_secs()
            ))
        }
    }
}

fn classify_successful_test_output(path: &Path, stdout: &[u8], stderr: &[u8]) -> TestExecution {
    let skipped_testc = contains_testc_skip(stdout) || contains_testc_skip(stderr);
    if skipped_testc {
        return TestExecution::SelfSkipped(format!(
            "{} exited successfully without running its testC checks",
            path.display()
        ));
    }
    TestExecution::Passed
}

fn compatibility_selector(corpus: &str) -> &str {
    match corpus {
        LUA51 => "lua51",
        LUA52 => "lua52",
        LUA53 => "lua53",
        LUA54 => "lua54",
        LUA55 => "lua55",
        LUAJIT => "luajit21",
        _ => corpus,
    }
}

#[derive(Debug)]
enum CaptureError {
    Launch(std::io::Error),
    TimedOut { timeout: Duration, stderr: Vec<u8> },
}

fn configured_test_timeout() -> Duration {
    std::env::var("OXIMOND_CONFORMANCE_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map_or(DEFAULT_TEST_TIMEOUT, Duration::from_secs)
}

/// Zero disables the cap.
fn configured_test_memory_limit_mib() -> u64 {
    std::env::var("OXIMOND_CONFORMANCE_MEMORY_LIMIT_MIB")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_TEST_MEMORY_LIMIT_MIB)
}

/// Starts `executable` under an address-space limit. The shell's `ulimit`
/// sets `RLIMIT_AS` without the unsafe `pre_exec` hook, and `exec` keeps the
/// test as the direct child so deadlines still kill it.
#[cfg(unix)]
fn memory_limited_command(executable: &Path, limit_mib: u64) -> Command {
    if limit_mib == 0 {
        return Command::new(executable);
    }
    let mut command = Command::new("sh");
    command
        .args(["-c", "ulimit -v \"$1\" && shift && exec \"$@\"", "sh"])
        .arg(limit_mib.saturating_mul(1024).to_string())
        .arg(executable);
    command
}

#[cfg(not(unix))]
fn memory_limited_command(executable: &Path, _limit_mib: u64) -> Command {
    Command::new(executable)
}

fn capture_output(command: &mut Command, timeout: Duration) -> Result<Output, CaptureError> {
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command.spawn().map_err(CaptureError::Launch)?;
    let stdout = child
        .stdout
        .take()
        .expect("stdout was configured as a pipe");
    let stderr = child
        .stderr
        .take()
        .expect("stderr was configured as a pipe");
    let stdout_reader = thread::spawn(move || read_pipe(stdout));
    let stderr_reader = thread::spawn(move || read_pipe(stderr));
    let started = Instant::now();

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let stdout = join_pipe(stdout_reader).map_err(CaptureError::Launch)?;
                let stderr = join_pipe(stderr_reader).map_err(CaptureError::Launch)?;
                return Ok(Output {
                    status,
                    stdout,
                    stderr,
                });
            }
            Ok(None) if started.elapsed() < timeout => thread::sleep(CHILD_POLL_INTERVAL),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = join_pipe(stdout_reader);
                let stderr = join_pipe(stderr_reader).unwrap_or_default();
                return Err(CaptureError::TimedOut { timeout, stderr });
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = join_pipe(stdout_reader);
                let _ = join_pipe(stderr_reader);
                return Err(CaptureError::Launch(error));
            }
        }
    }
}

fn read_pipe(mut pipe: impl Read) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    pipe.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn join_pipe(reader: thread::JoinHandle<std::io::Result<Vec<u8>>>) -> std::io::Result<Vec<u8>> {
    reader
        .join()
        .map_err(|_| std::io::Error::other("child output reader panicked"))?
}

fn contains_testc_skip(output: &[u8]) -> bool {
    const MARKER: &[u8] = b">>> testC not active: skipping";
    output.windows(MARKER.len()).any(|window| window == MARKER)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TestLaunch {
    working_directory: PathBuf,
    lua_path: OsString,
    clears_host_lua_path: bool,
}

impl TestLaunch {
    fn new(working_directory: &Path, corpus_root: &Path) -> Self {
        let local_modules = working_directory.join("?.lua");
        let local_packages = working_directory.join("?").join("init.lua");
        let corpus_modules = corpus_root.join("?.lua");
        let corpus_packages = corpus_root.join("?").join("init.lua");
        let lua_path = format!(
            "?;{};{};{};{}",
            local_modules.display(),
            local_packages.display(),
            corpus_modules.display(),
            corpus_packages.display()
        );
        Self {
            working_directory: working_directory.to_path_buf(),
            lua_path: OsString::from(lua_path),
            clears_host_lua_path: true,
        }
    }
}

fn absolute_path(path: &Path) -> Result<PathBuf, String> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        std::env::current_dir()
            .map(|directory| directory.join(path))
            .map_err(|error| format!("cannot resolve {}: {error}", path.display()))
    }
}

fn render_run_report(counts: &RunCounts) -> String {
    let mut report = String::from("Oximond conformance run\n");
    for corpus in &counts.corpora {
        writeln!(report, "\n[{corpus}]").expect("writing to a string cannot fail");
        writeln!(
            report,
            "passed = {}",
            counts.passed.get(corpus).copied().unwrap_or(0)
        )
        .expect("writing to a string cannot fail");
        writeln!(
            report,
            "failed = {}",
            counts.failed.get(corpus).copied().unwrap_or(0)
        )
        .expect("writing to a string cannot fail");
        writeln!(
            report,
            "expected-failures = {}",
            counts.expected_failures.get(corpus).copied().unwrap_or(0)
        )
        .expect("writing to a string cannot fail");
        writeln!(
            report,
            "unexpected-passes = {}",
            counts.unexpected_passes.get(corpus).copied().unwrap_or(0)
        )
        .expect("writing to a string cannot fail");
        writeln!(
            report,
            "infrastructure-errors = {}",
            counts
                .infrastructure_errors
                .get(corpus)
                .copied()
                .unwrap_or(0)
        )
        .expect("writing to a string cannot fail");
        writeln!(
            report,
            "self-skipped = {}",
            counts.self_skipped.get(corpus).copied().unwrap_or(0)
        )
        .expect("writing to a string cannot fail");
        writeln!(
            report,
            "deferred = {}",
            counts.deferred.get(corpus).copied().unwrap_or(0)
        )
        .expect("writing to a string cannot fail");
    }
    report
}

fn has_wildcard(path: &str) -> bool {
    path.contains(['*', '?', '['])
}

#[must_use]
pub fn source_status(source: &Source, root: &Path) -> SourceStatus {
    match source.kind.as_str() {
        "archive" => archive_status(source, root),
        "git" => git_status(source, root),
        _ => SourceStatus::Invalid("unrecognised source kind".to_owned()),
    }
}

fn prepare_archive(source: &Source, root: &Path) -> Result<(), String> {
    let directory = root.join(&source.directory);
    let archive = directory.join(&source.archive);
    let extracted = directory.join("source");
    if matches!(archive_status(source, root), SourceStatus::Ready) {
        return Ok(());
    }
    fs::create_dir_all(&directory)
        .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
    let temporary = directory.join(format!(
        ".{}.{}.download",
        source.archive,
        std::process::id()
    ));
    let _ = fs::remove_file(&temporary);
    // --retry alone only covers timeouts and a few HTTP codes; --retry-all-
    // errors makes it cover transport failures like a TLS handshake cut
    // short, which the pinned lua.org mirrors occasionally do. The archive
    // is SHA-256-pinned below, so retrying any error is safe.
    run_process(
        Command::new("curl")
            .arg("--fail")
            .arg("--location")
            .arg("--retry")
            .arg("5")
            .arg("--retry-all-errors")
            .arg("--silent")
            .arg("--show-error")
            .arg("--output")
            .arg(&temporary)
            .arg(&source.url),
        &format!("download {} archive", source.corpus),
    )?;
    let actual = sha256_file(&temporary)?;
    if actual != source.pin {
        let _ = fs::remove_file(&temporary);
        return Err(format!(
            "SHA-256 mismatch: expected {}, got {actual}",
            source.pin
        ));
    }
    fs::rename(&temporary, &archive).map_err(|error| {
        format!(
            "cannot save verified archive {}: {error}",
            archive.display()
        )
    })?;
    if extracted.is_dir() {
        return Ok(());
    }
    fs::create_dir_all(&extracted)
        .map_err(|error| format!("cannot create {}: {error}", extracted.display()))?;
    run_process(
        Command::new("tar")
            .arg("-xzf")
            .arg(&archive)
            .arg("--strip-components=1")
            .arg("-C")
            .arg(&extracted),
        &format!("extract {} archive", source.corpus),
    )
}

fn prepare_git(source: &Source, root: &Path) -> Result<(), String> {
    let directory = root.join(&source.directory).join("source");
    if directory.join(".git").is_dir() {
        run_process(
            Command::new("git")
                .arg("-C")
                .arg(&directory)
                .arg("fetch")
                .arg("--depth")
                .arg("1")
                .arg("origin")
                .arg(&source.pin),
            &format!("fetch pinned {} test commit", source.corpus),
        )?;
    } else {
        if directory.exists() {
            return Err(format!(
                "{} exists but is not a git checkout",
                directory.display()
            ));
        }
        let parent = directory
            .parent()
            .ok_or_else(|| "git checkout path has no parent".to_owned())?;
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        run_process(
            Command::new("git")
                .arg("clone")
                .arg("--no-checkout")
                .arg(&source.url)
                .arg(&directory),
            &format!("clone {} test corpus", source.corpus),
        )?;
        run_process(
            Command::new("git")
                .arg("-C")
                .arg(&directory)
                .arg("fetch")
                .arg("--depth")
                .arg("1")
                .arg("origin")
                .arg(&source.pin),
            &format!("fetch pinned {} test commit", source.corpus),
        )?;
    }
    run_process(
        Command::new("git")
            .arg("-C")
            .arg(&directory)
            .arg("checkout")
            .arg("--detach")
            .arg(&source.pin),
        &format!("check out pinned {} test commit", source.corpus),
    )?;
    match git_status(source, root) {
        SourceStatus::Ready => Ok(()),
        SourceStatus::Missing => Err("git checkout disappeared while preparing it".to_owned()),
        SourceStatus::Invalid(reason) => Err(reason),
    }
}

/// Recreates the execution tree from the verified immutable input, then applies
/// the suite patch without consulting any enclosing Git repository.
fn rebuild_patched_tree(
    source: &Source,
    root: &Path,
    manifest_directory: Option<&Path>,
) -> Result<(), String> {
    let root = absolute_path(root)?;
    let Some(patch_name) = &source.patches else {
        return Ok(());
    };
    let manifest_directory = manifest_directory.ok_or_else(|| {
        format!(
            "{} declares patches but its source manifest was not loaded from a file",
            source.corpus
        )
    })?;
    let patch = resolve_patch_path(manifest_directory, patch_name)?;
    if !patch.is_file() {
        return Err(format!(
            "{} patch file does not exist: {}",
            source.corpus,
            patch.display()
        ));
    }

    let directory = root.join(&source.directory);
    let patched = directory.join("patched");
    let staging = directory.join(format!(".patched-{}", std::process::id()));
    if staging.exists() {
        fs::remove_dir_all(&staging).map_err(|error| {
            format!(
                "cannot remove stale patch staging tree {}: {error}",
                staging.display()
            )
        })?;
    }
    fs::create_dir_all(&staging).map_err(|error| {
        format!(
            "cannot create patch staging tree {}: {error}",
            staging.display()
        )
    })?;

    let result = populate_pristine_tree(source, &root, &staging)
        .and_then(|()| apply_patch(source, &staging, &patch));
    if let Err(error) = result {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }
    if patched.exists() {
        if !patched.is_dir() {
            let _ = fs::remove_dir_all(&staging);
            return Err(format!(
                "patched tree path is not a directory: {}",
                patched.display()
            ));
        }
        fs::remove_dir_all(&patched).map_err(|error| {
            format!("cannot replace patched tree {}: {error}", patched.display())
        })?;
    }
    fs::rename(&staging, &patched)
        .map_err(|error| format!("cannot publish patched tree {}: {error}", patched.display()))
}

fn resolve_patch_path(manifest_directory: &Path, patch_name: &str) -> Result<PathBuf, String> {
    let patch = Path::new(patch_name);
    if patch.is_absolute() {
        return Err(format!(
            "patch path must be relative to the source manifest: {patch_name}"
        ));
    }
    Ok(normalize_path(&manifest_directory.join(patch)))
}

fn normalize_path(path: &Path) -> PathBuf {
    use std::path::Component;

    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::RootDir | Component::Prefix(_) | Component::Normal(_) => {
                normalized.push(component.as_os_str());
            }
        }
    }
    normalized
}

fn populate_pristine_tree(source: &Source, root: &Path, destination: &Path) -> Result<(), String> {
    match source.kind.as_str() {
        "archive" => {
            let archive = root.join(&source.directory).join(&source.archive);
            run_process(
                Command::new("tar")
                    .arg("-xzf")
                    .arg(&archive)
                    .arg("--strip-components=1")
                    .arg("-C")
                    .arg(destination),
                &format!("extract pristine {} archive for patching", source.corpus),
            )
        }
        "git" => {
            let checkout = root.join(&source.directory).join("source");
            let archive = destination.with_extension(format!("{}.tar", std::process::id()));
            let result = run_process(
                Command::new("git")
                    .arg("-C")
                    .arg(&checkout)
                    .arg("archive")
                    .arg("--format=tar")
                    .arg(format!("--output={}", archive.display()))
                    .arg(&source.pin),
                &format!("archive pinned {} test commit for patching", source.corpus),
            )
            .and_then(|()| {
                run_process(
                    Command::new("tar")
                        .arg("-xf")
                        .arg(&archive)
                        .arg("-C")
                        .arg(destination),
                    &format!("extract pristine {} git tree for patching", source.corpus),
                )
            });
            let _ = fs::remove_file(&archive);
            result
        }
        _ => Err(format!("unrecognised source kind `{}`", source.kind)),
    }
}

fn apply_patch(source: &Source, tree: &Path, patch: &Path) -> Result<(), String> {
    let parent = tree
        .parent()
        .ok_or_else(|| format!("patch tree has no parent: {}", tree.display()))?;
    run_process(
        Command::new("git")
            // Git-format diffs can silently skip paths when apply discovers an
            // enclosing repository. --no-index alone does not prevent that.
            .env("GIT_CEILING_DIRECTORIES", parent)
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .arg("-C")
            .arg(tree)
            .arg("apply")
            .arg("--no-index")
            .arg("--whitespace=nowarn")
            .arg(patch),
        &format!("apply {} patch {}", source.corpus, patch.display()),
    )
}

fn archive_status(source: &Source, root: &Path) -> SourceStatus {
    let directory = root.join(&source.directory);
    let archive = directory.join(&source.archive);
    if !archive.is_file() || !directory.join("source").is_dir() {
        return SourceStatus::Missing;
    }
    match sha256_file(&archive) {
        Ok(actual) if actual == source.pin => SourceStatus::Ready,
        Ok(actual) => {
            SourceStatus::Invalid(format!("SHA-256 is {actual}, expected {}", source.pin))
        }
        Err(error) => SourceStatus::Invalid(error),
    }
}

fn git_status(source: &Source, root: &Path) -> SourceStatus {
    let directory = root.join(&source.directory).join("source");
    if !directory.join(".git").is_dir() {
        return SourceStatus::Missing;
    }
    match Command::new("git")
        .arg("-C")
        .arg(&directory)
        .arg("rev-parse")
        .arg("HEAD")
        .output()
    {
        Ok(output) if output.status.success() => {
            let actual = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if actual == source.pin {
                SourceStatus::Ready
            } else {
                SourceStatus::Invalid(format!("commit is {actual}, expected {}", source.pin))
            }
        }
        Ok(output) => SourceStatus::Invalid(format!(
            "cannot inspect git commit: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )),
        Err(error) => SourceStatus::Invalid(format!("cannot run git: {error}")),
    }
}

fn run_process(command: &mut Command, action: &str) -> Result<(), String> {
    let output = command
        .output()
        .map_err(|error| format!("cannot {action}: {error}"))?;
    if output.status.success() {
        Ok(())
    } else {
        let detail = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        Err(format!("cannot {action}: {detail}"))
    }
}

fn status_label(status: &SourceStatus) -> &'static str {
    match status {
        SourceStatus::Ready => "ready",
        SourceStatus::Missing => "missing",
        SourceStatus::Invalid(_) => "invalid",
    }
}

fn parse_records(
    contents: &str,
    table_name: &str,
) -> Result<Vec<BTreeMap<String, String>>, String> {
    let mut records = Vec::new();
    let mut current = None;
    for (index, raw_line) in contents.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("format =") {
            continue;
        }
        if line == format!("[[{table_name}]]") {
            if let Some(record) = current.take() {
                records.push(record);
            }
            current = Some(BTreeMap::new());
            continue;
        }
        let Some(record) = current.as_mut() else {
            return Err(format!(
                "line {} is outside a [[{table_name}]] table.",
                index + 1
            ));
        };
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!("line {} is not a key/value pair.", index + 1));
        };
        let key = key.trim();
        let value = value.trim();
        if key.is_empty() || !value.starts_with('"') || !value.ends_with('"') || value.len() < 2 {
            return Err(format!(
                "line {} must use a quoted string value.",
                index + 1
            ));
        }
        if record
            .insert(
                key.to_owned(),
                value[1..value.len() - 1].replace("\\\"", "\""),
            )
            .is_some()
        {
            return Err(format!("line {} repeats key `{key}`.", index + 1));
        }
    }
    if let Some(record) = current {
        records.push(record);
    }
    Ok(records)
}

fn require_fields(
    record: &BTreeMap<String, String>,
    fields: &[&str],
    record_name: &str,
) -> Result<(), String> {
    for field in fields {
        if !record.contains_key(*field) {
            return Err(format!("{record_name} is missing `{field}`."));
        }
    }
    Ok(())
}

#[must_use]
pub fn default_sources_path() -> PathBuf {
    workspace_path("tests/conformance/sources.toml")
}
#[must_use]
pub fn default_ledger_path() -> PathBuf {
    workspace_path("tests/conformance/ledger.toml")
}

fn workspace_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file =
        fs::File::open(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 8192];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hasher.finish())
}

struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    length: u64,
    buffered: usize,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09_e667,
                0xbb67_ae85,
                0x3c6e_f372,
                0xa54f_f53a,
                0x510e_527f,
                0x9b05_688c,
                0x1f83_d9ab,
                0x5be0_cd19,
            ],
            buffer: [0; 64],
            length: 0,
            buffered: 0,
        }
    }
    fn update(&mut self, input: &[u8]) {
        self.length += u64::try_from(input.len()).expect("input length fits u64");
        for &byte in input {
            self.buffer[self.buffered] = byte;
            self.buffered += 1;
            if self.buffered == 64 {
                self.compress();
                self.buffered = 0;
            }
        }
    }
    fn finish(mut self) -> String {
        let bit_length = self
            .length
            .checked_mul(8)
            .expect("message length fits u64 bits");
        self.buffer[self.buffered] = 0x80;
        self.buffered += 1;
        if self.buffered > 56 {
            self.buffer[self.buffered..].fill(0);
            self.compress();
            self.buffered = 0;
        }
        self.buffer[self.buffered..56].fill(0);
        self.buffer[56..].copy_from_slice(&bit_length.to_be_bytes());
        self.compress();
        let mut digest = String::with_capacity(64);
        for word in self.state {
            write!(digest, "{word:08x}").expect("writing to a string cannot fail");
        }
        digest
    }
    #[allow(clippy::many_single_char_names, clippy::too_many_lines)]
    fn compress(&mut self) {
        const K: [u32; 64] = [
            0x428a_2f98,
            0x7137_4491,
            0xb5c0_fbcf,
            0xe9b5_dba5,
            0x3956_c25b,
            0x59f1_11f1,
            0x923f_82a4,
            0xab1c_5ed5,
            0xd807_aa98,
            0x1283_5b01,
            0x2431_85be,
            0x550c_7dc3,
            0x72be_5d74,
            0x80de_b1fe,
            0x9bdc_06a7,
            0xc19b_f174,
            0xe49b_69c1,
            0xefbe_4786,
            0x0fc1_9dc6,
            0x240c_a1cc,
            0x2de9_2c6f,
            0x4a74_84aa,
            0x5cb0_a9dc,
            0x76f9_88da,
            0x983e_5152,
            0xa831_c66d,
            0xb003_27c8,
            0xbf59_7fc7,
            0xc6e0_0bf3,
            0xd5a7_9147,
            0x06ca_6351,
            0x1429_2967,
            0x27b7_0a85,
            0x2e1b_2138,
            0x4d2c_6dfc,
            0x5338_0d13,
            0x650a_7354,
            0x766a_0abb,
            0x81c2_c92e,
            0x9272_2c85,
            0xa2bf_e8a1,
            0xa81a_664b,
            0xc24b_8b70,
            0xc76c_51a3,
            0xd192_e819,
            0xd699_0624,
            0xf40e_3585,
            0x106a_a070,
            0x19a4_c116,
            0x1e37_6c08,
            0x2748_774c,
            0x34b0_bcb5,
            0x391c_0cb3,
            0x4ed8_aa4a,
            0x5b9c_ca4f,
            0x682e_6ff3,
            0x748f_82ee,
            0x78a5_636f,
            0x84c8_7814,
            0x8cc7_0208,
            0x90be_fffa,
            0xa450_6ceb,
            0xbef9_a3f7,
            0xc671_78f2,
        ];
        let mut words = [0_u32; 64];
        for (index, chunk) in self.buffer.as_chunks::<4>().0.iter().enumerate() {
            words[index] = u32::from_be_bytes(*chunk);
        }
        for index in 16..64 {
            let a = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let b = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(a)
                .wrapping_add(words[index - 7])
                .wrapping_add(b);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(choice)
                .wrapping_add(K[index])
                .wrapping_add(words[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_profile_paths_follow_corpus_and_test_path() {
        let root = Path::new("/corpora/lua54");
        let directory = Path::new("/profiles");
        let first = trace_profile_path(directory, LUA54, &root.join("test/a.lua"), root);
        let second = trace_profile_path(directory, LUA54, &root.join("test/b.lua"), root);
        assert_eq!(
            first,
            PathBuf::from("/profiles/lua54/test/a.lua.trace-profile")
        );
        assert_ne!(first, second);
    }

    const SOURCES: &str = include_str!("../../../tests/conformance/sources.toml");
    const LEDGER: &str = include_str!("../../../tests/conformance/ledger.toml");

    #[test]
    fn source_manifest_has_the_pinned_corpora() {
        let sources = Sources::parse(SOURCES).expect("valid source manifest");
        assert_eq!(
            sources.get(LUA51).expect("lua51 source").pin,
            "49e4ca6561f82ea605908c5041ab5fad66ed9930fa0686675bd51b02767f18ad"
        );
        assert_eq!(
            sources.get(LUA52).expect("lua52 source").pin,
            "07c1071c66d9cb4dc80200ed430fc21ebbbccf1f55148b87eb0b9522932e4eb6"
        );
        assert_eq!(
            sources.get(LUA53).expect("lua53 source").pin,
            "b80771238271c72565e5a1183292ef31bd7166414cd0d43a8eb79845fa7f599f"
        );
        assert_eq!(
            sources.get(LUA54).expect("lua54 source").pin,
            "7d971845f545ffc09fbb3128a86b2c6524161c70d0fdf0154a16e8c00c343fca"
        );
        assert_eq!(
            sources.get(LUA55).expect("lua55 source").pin,
            "da07b543872dc0bb2ff12aabd0c248578d78df3eb6b67efdc537a46d455c7f31"
        );
        assert_eq!(
            sources.get(LUAJIT).expect("luajit source").pin,
            "014708bceb70550a3ab8d539cff14d9085ca9cb8"
        );
    }

    #[test]
    fn corpora_run_under_their_own_compatibility_profiles() {
        assert_eq!(compatibility_selector(LUA51), "lua51");
        assert_eq!(compatibility_selector(LUA52), "lua52");
        assert_eq!(compatibility_selector(LUA53), "lua53");
        assert_eq!(compatibility_selector(LUA54), "lua54");
        assert_eq!(compatibility_selector(LUA55), "lua55");
        assert_eq!(compatibility_selector(LUAJIT), "luajit21");
    }

    #[test]
    fn ledger_validates_against_every_declared_corpus() {
        let sources = Sources::parse(SOURCES).expect("valid sources");
        let ledger = Ledger::parse(LEDGER).expect("valid ledger");
        ledger.validate(&sources).expect("valid ledger policy");
    }

    #[test]
    fn report_keeps_corpora_separate_when_not_prepared() {
        let sources = Sources::parse(SOURCES).expect("valid sources");
        let ledger = Ledger::parse(LEDGER).expect("valid ledger");
        let root =
            std::env::temp_dir().join(format!("oximond-conformance-test-{}", std::process::id()));
        let report = render_report(&sources, &ledger, &root).expect("report");
        assert!(report.contains("[lua51]"));
        assert!(report.contains("[lua52]"));
        assert!(report.contains("[lua53]"));
        assert!(report.contains("[lua54]"));
        assert!(report.contains("[lua55]"));
        assert!(report.contains("[luajit]"));
        assert!(report.contains("status = missing"));
    }

    #[test]
    fn sha256_matches_a_known_vector() {
        let mut hasher = Sha256::new();
        hasher.update(b"abc");
        assert_eq!(
            hasher.finish(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn parser_rejects_an_unquoted_value() {
        assert!(Sources::parse("[[source]]\ncorpus = lua51\n").is_err());
    }

    #[test]
    fn parser_rejects_an_empty_patch_path() {
        assert!(Sources::parse(
            "[[source]]\ncorpus = \"fixture\"\nkind = \"archive\"\nurl = \"unused\"\npin = \"pin\"\narchive = \"fixture.tar.gz\"\ndirectory = \"fixture\"\npatches = \"\"\n"
        )
        .is_err());
    }

    #[test]
    fn source_manifest_resolves_patch_paths_from_its_own_directory() {
        let root = std::env::current_dir()
            .expect("workspace directory")
            .join("target")
            .join(format!("patch-manifest-{}", std::process::id()));
        let manifest = root.join("manifests").join("sources.toml");
        fs::create_dir_all(manifest.parent().expect("manifest parent")).expect("create manifest");
        fs::write(
            &manifest,
            "[[source]]\ncorpus = \"fixture\"\nkind = \"archive\"\nurl = \"unused\"\npin = \"pin\"\narchive = \"fixture.tar.gz\"\ndirectory = \"fixture\"\npatches = \"../patches/fixture.patch\"\n",
        )
        .expect("write manifest");
        let relative_manifest = manifest
            .strip_prefix(std::env::current_dir().expect("workspace directory"))
            .expect("manifest below workspace");
        let sources = Sources::load(relative_manifest).expect("load manifest");
        let patch = resolve_patch_path(
            sources
                .manifest_directory
                .as_deref()
                .expect("manifest directory"),
            sources
                .get("fixture")
                .expect("fixture source")
                .patches
                .as_deref()
                .expect("patch"),
        )
        .expect("relative patch path");
        assert_eq!(patch, root.join("patches/fixture.patch"));
        assert!(resolve_patch_path(Path::new("/manifest"), "/absolute.patch").is_err());
        fs::remove_dir_all(root).expect("remove test files");
    }

    #[test]
    fn native_cli_fixture_builds_upstream_lua54_and_lua55_modules() {
        for (profile, source) in [
            (
                LUA54,
                include_str!("../../oximond-cli/tests/fixtures/lib22-lua54.c"),
            ),
            (
                LUA55,
                include_str!("../../oximond-cli/tests/fixtures/lib22-lua55.c"),
            ),
        ] {
            let root = unique_test_directory(profile);
            fs::create_dir_all(root.join("libs")).expect("create native fixture directory");
            fs::write(root.join("libs/lib22.c"), source).expect("write upstream native fixture");
            build_native_cli_fixture(&root).expect("build actual upstream native fixture");
            let output = Command::new("nm")
                .args(["-D", "--defined-only"])
                .arg(root.join("libs/lib2-v2.so"))
                .output()
                .expect("inspect native fixture symbols");
            assert!(output.status.success());
            assert!(String::from_utf8_lossy(&output.stdout).contains("luaopen_lib2"));
            fs::remove_dir_all(root).expect("remove native fixture files");
        }
    }

    #[test]
    fn patched_tree_is_fresh_and_leaves_the_upstream_archive_tree_untouched() {
        let root = unique_test_directory("patch-tree");
        fs::create_dir_all(&root).expect("create test repository");
        run_process(
            Command::new("git").arg("init").arg(&root),
            "initialize test repository",
        )
        .expect("test repository");
        fs::write(root.join(".gitignore"), "fixture/\n").expect("ignore corpus cache");
        let corpus = root.join("fixture");
        let payload = root.join("payload").join("lua");
        fs::create_dir_all(&payload).expect("create archive payload");
        fs::write(payload.join("test.lua"), "return 'upstream'\n").expect("write upstream test");
        fs::create_dir_all(&corpus).expect("create corpus directory");
        let archive = corpus.join("fixture.tar.gz");
        run_process(
            Command::new("tar")
                .arg("-czf")
                .arg(&archive)
                .arg("-C")
                .arg(root.join("payload"))
                .arg("lua"),
            "build test archive",
        )
        .expect("archive");
        let patch = root.join("fixture.patch");
        fs::write(
            &patch,
            "diff --git a/test.lua b/test.lua\n--- a/test.lua\n+++ b/test.lua\n@@ -1 +1 @@\n-return 'upstream'\n+return 'patched'\n",
        )
        .expect("write patch");
        let source = Source {
            corpus: "fixture".to_owned(),
            kind: "archive".to_owned(),
            url: "unused".to_owned(),
            pin: sha256_file(&archive).expect("archive hash"),
            archive: "fixture.tar.gz".to_owned(),
            directory: "fixture".to_owned(),
            patches: Some("fixture.patch".to_owned()),
        };
        let upstream = corpus.join("source");
        fs::create_dir_all(&upstream).expect("create mutable upstream tree");
        fs::write(upstream.join("test.lua"), "return 'changed locally'\n")
            .expect("change upstream");

        rebuild_patched_tree(&source, &root, Some(&root)).expect("apply valid patch");
        assert_eq!(
            fs::read_to_string(corpus.join("patched/test.lua")).expect("patched test"),
            "return 'patched'\n"
        );
        assert_eq!(
            fs::read_to_string(upstream.join("test.lua")).expect("upstream test"),
            "return 'changed locally'\n"
        );
        fs::write(corpus.join("patched/test.lua"), "return 'stale'\n").expect("taint patched tree");
        rebuild_patched_tree(&source, &root, Some(&root)).expect("reapply valid patch");
        assert_eq!(
            fs::read_to_string(corpus.join("patched/test.lua")).expect("fresh patched test"),
            "return 'patched'\n"
        );
        fs::write(
            &patch,
            "--- a/test.lua\n+++ b/test.lua\n@@ -1 +1 @@\n-return 'another version'\n+return 'patched'\n",
        )
        .expect("write stale patch");
        let error = rebuild_patched_tree(&source, &root, Some(&root))
            .expect_err("reject a patch for different upstream content");
        assert!(error.contains("apply fixture patch"));
        assert_eq!(
            fs::read_to_string(corpus.join("patched/test.lua")).expect("last complete tree"),
            "return 'patched'\n"
        );
        fs::write(&patch, "this is not a patch\n").expect("write invalid patch");
        let error =
            rebuild_patched_tree(&source, &root, Some(&root)).expect_err("reject invalid patch");
        assert!(error.contains("apply fixture patch"));
        assert!(
            !corpus
                .join(format!(".patched-{}", std::process::id()))
                .exists()
        );
        fs::remove_file(&patch).expect("remove patch");
        let error = rebuild_patched_tree(&source, &root, Some(&root))
            .expect_err("reject a missing patch instead of using an old patched tree");
        assert!(error.contains("fixture patch file does not exist"));
        fs::remove_dir_all(root).expect("remove test files");
    }

    #[test]
    fn relative_manifest_and_corpus_root_patch_a_pinned_git_source() {
        let workspace = std::env::current_dir().expect("workspace directory");
        let root = workspace
            .join("target")
            .join(format!("patch-git-{}", std::process::id()));
        let upstream = root.join("upstream");
        fs::create_dir_all(&upstream).expect("create upstream repository");
        run_process(
            Command::new("git").arg("init").arg(&upstream),
            "initialize upstream",
        )
        .expect("initialize upstream");
        fs::write(upstream.join("test.lua"), "return 'upstream'\n").expect("write upstream test");
        run_process(
            Command::new("git")
                .arg("-C")
                .arg(&upstream)
                .arg("add")
                .arg("test.lua"),
            "stage upstream test",
        )
        .expect("stage upstream test");
        run_process(
            Command::new("git")
                .arg("-C")
                .arg(&upstream)
                .arg("-c")
                .arg("user.name=Test")
                .arg("-c")
                .arg("user.email=test@example.invalid")
                .arg("commit")
                .arg("-m")
                .arg("fixture\n\nCo-Authored-By: Codex <codex@openai.com>"),
            "commit upstream test",
        )
        .expect("commit upstream test");
        let pin = Command::new("git")
            .arg("-C")
            .arg(&upstream)
            .arg("rev-parse")
            .arg("HEAD")
            .output()
            .expect("read upstream pin");
        assert!(pin.status.success());
        let pin = String::from_utf8(pin.stdout)
            .expect("UTF-8 pin")
            .trim()
            .to_owned();

        let manifest = root.join("manifests/sources.toml");
        fs::create_dir_all(manifest.parent().expect("manifest parent")).expect("create manifests");
        fs::create_dir_all(root.join("patches")).expect("create patches");
        fs::write(
            root.join("patches/fixture.patch"),
            "--- a/test.lua\n+++ b/test.lua\n@@ -1 +1 @@\n-return 'upstream'\n+return 'patched'\n",
        )
        .expect("write patch");
        fs::write(
            &manifest,
            format!(
                "[[source]]\ncorpus = \"fixture\"\nkind = \"git\"\nurl = \"{}\"\npin = \"{pin}\"\narchive = \"unused\"\ndirectory = \"fixture\"\npatches = \"../patches/fixture.patch\"\n",
                upstream.display()
            ),
        )
        .expect("write manifest");
        let relative_manifest = manifest
            .strip_prefix(&workspace)
            .expect("relative manifest");
        let sources = Sources::load(relative_manifest).expect("load relative manifest");
        let corpus_root = root.join("corpora");
        let relative_corpus_root = corpus_root
            .strip_prefix(&workspace)
            .expect("relative corpus root");
        prepare(&sources, relative_corpus_root).expect("prepare patched git source");
        assert_eq!(
            fs::read_to_string(corpus_root.join("fixture/patched/test.lua")).expect("patched test"),
            "return 'patched'\n"
        );
        fs::remove_dir_all(root).expect("remove test files");
    }

    fn expected_paths(root: &Path, corpus_root: &str, paths: &[&str]) -> Vec<(PathBuf, PathBuf)> {
        let corpus_root = root.join(corpus_root);
        paths
            .iter()
            .map(|path| {
                let path = corpus_root.join(path);
                let working_directory = path.parent().expect("test file has parent").to_path_buf();
                (path, working_directory)
            })
            .collect()
    }

    #[test]
    fn runner_executes_exact_paths_with_parent_directories_and_counts_states() {
        let sources = Sources::parse(SOURCES).expect("valid sources");
        let ledger = Ledger::parse(LEDGER).expect("valid ledger");
        let root = std::env::temp_dir().join(format!(
            "oximond-conformance-run-test-{}",
            std::process::id()
        ));
        let mut executed = Vec::new();
        let counts = run_ledger_entries(
            &ledger,
            &sources,
            &root,
            |corpus, path, working_directory, _| {
                executed.push((path.to_path_buf(), working_directory.to_path_buf()));
                match ledger
                    .entries_for(corpus)
                    .find(|entry| path.ends_with(&entry.path))
                    .expect("executed ledger entry")
                    .state
                    .as_str()
                {
                    "required" => TestExecution::Passed,
                    "expected-fail" => TestExecution::Nonzero("known failure".to_owned()),
                    state => panic!("{state} entries must not execute"),
                }
            },
        )
        .expect("required entry paths are exact");

        let mut expected = Vec::new();
        for source in sources.iter() {
            let paths: Vec<&str> = ledger
                .entries_for(&source.corpus)
                .filter(|entry| entry.state != "deferred")
                .map(|entry| entry.path.as_str())
                .collect();
            expected.extend(expected_paths(
                &root,
                &format!(
                    "{}/{}",
                    source.directory,
                    if source.patches.is_some() {
                        "patched"
                    } else {
                        "source"
                    }
                ),
                &paths,
            ));
        }
        assert_eq!(executed, expected);
        for source in sources.iter() {
            let required = ledger
                .entries_for(&source.corpus)
                .filter(|entry| entry.state == "required")
                .count();
            let expected_failures = ledger
                .entries_for(&source.corpus)
                .filter(|entry| entry.state == "expected-fail")
                .count();
            let deferred = ledger
                .entries_for(&source.corpus)
                .filter(|entry| entry.state == "deferred")
                .count();
            assert_eq!(
                counts.passed.get(&source.corpus).copied().unwrap_or(0),
                required
            );
            assert_eq!(
                counts
                    .expected_failures
                    .get(&source.corpus)
                    .copied()
                    .unwrap_or(0),
                expected_failures
            );
            assert_eq!(
                counts.deferred.get(&source.corpus).copied().unwrap_or(0),
                deferred
            );
        }
        assert_eq!(counts.failed.get(LUAJIT), None);
        assert_eq!(counts.expected_failures.get(LUA51), None);
        assert_eq!(counts.expected_failures.get(LUAJIT), None);
        assert_eq!(counts.unexpected_passes.get(LUA51), None);
        assert_eq!(counts.infrastructure_errors.get(LUA51), None);
        let report = render_run_report(&counts);
        assert!(report.contains("[lua51]\npassed ="));
        assert!(report.contains("[lua55]\npassed ="));
        assert!(report.contains("[luajit]\npassed ="));
    }

    #[test]
    fn runner_records_every_executable_state_transition() {
        let sources = Sources::parse(SOURCES).expect("valid sources");
        let ledger = Ledger {
            entries: vec![
                test_entry(LUA51, "required-pass.lua", "required"),
                test_entry(LUA51, "required-fail.lua", "required"),
                test_entry(LUA51, "expected-fail.lua", "expected-fail"),
                test_entry(LUA51, "stale.lua", "expected-fail"),
                test_entry(LUA51, "infrastructure.lua", "required"),
                test_entry(LUA51, "self-skipped.lua", "required"),
                test_entry(LUA51, "deferred.lua", "deferred"),
            ],
        };
        let counts = run_ledger_entries(
            &ledger,
            &sources,
            Path::new("/test-root"),
            |_, path, _, _| match path.file_name().and_then(std::ffi::OsStr::to_str) {
                Some("required-pass.lua" | "stale.lua") => TestExecution::Passed,
                Some("required-fail.lua") => TestExecution::Nonzero("test failed".to_owned()),
                Some("expected-fail.lua") => TestExecution::Nonzero("known failure".to_owned()),
                Some("infrastructure.lua") => {
                    TestExecution::Infrastructure("cannot launch".to_owned())
                }
                Some("self-skipped.lua") => {
                    TestExecution::SelfSkipped("missing testC fixture".to_owned())
                }
                _ => panic!("deferred entries must not execute"),
            },
        )
        .expect("exact paths");

        assert_eq!(counts.passed.get(LUA51), Some(&1));
        assert_eq!(counts.failed.get(LUA51), Some(&1));
        assert_eq!(counts.expected_failures.get(LUA51), Some(&1));
        assert_eq!(counts.unexpected_passes.get(LUA51), Some(&1));
        assert_eq!(counts.infrastructure_errors.get(LUA51), Some(&1));
        assert_eq!(counts.self_skipped.get(LUA51), Some(&1));
        assert_eq!(counts.deferred.get(LUA51), Some(&1));
        assert_eq!(counts.failures.len(), 4);
    }

    #[test]
    fn testc_self_skip_marker_is_not_confused_with_optional_locale_skips() {
        assert!(contains_testc_skip(
            b"\x07 >>> testC not active: skipping API tests <<<\n"
        ));
        assert!(!contains_testc_skip(
            b">>> pt_BR locale not available: skipping decimal point tests <<<"
        ));
    }

    #[test]
    fn testc_skip_is_never_accepted_as_full_conformance() {
        let marker = b">>> testC not active: skipping API tests <<<\n";
        let aggregate = Path::new("/corpora/lua51/source/all.lua");
        let direct_test = Path::new("/corpora/lua51/source/api.lua");

        assert!(matches!(
            classify_successful_test_output(aggregate, marker, b""),
            TestExecution::SelfSkipped(_)
        ));
        assert!(matches!(
            classify_successful_test_output(direct_test, marker, b""),
            TestExecution::SelfSkipped(_)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn process_capture_kills_a_test_after_its_deadline() {
        let started = Instant::now();
        let error = capture_output(
            Command::new("sh").args(["-c", "sleep 2"]),
            Duration::from_millis(20),
        )
        .expect_err("sleeping child should exceed its deadline");
        assert!(matches!(error, CaptureError::TimedOut { .. }));
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[cfg(unix)]
    #[test]
    fn memory_limited_command_caps_the_test_address_space() {
        let output = capture_output(
            memory_limited_command(Path::new("sh"), 64).args(["-c", "ulimit -v"]),
            Duration::from_secs(10),
        )
        .expect("limited shell should run");
        assert!(output.status.success());
        assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "65536");
    }

    #[test]
    fn prepared_stock_lua_suites_require_one_exact_ledger_entry_per_lua_file() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("current time")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "oximond-ledger-coverage-{}-{nonce}",
            std::process::id()
        ));
        let lua51_directory = root.join("lua51/source");
        let lua55_directory = root.join("lua55/source");
        fs::create_dir_all(&lua51_directory).expect("create Lua 5.1 test corpus");
        fs::create_dir_all(&lua55_directory).expect("create Lua 5.5 test corpus");
        fs::write(lua51_directory.join("first.lua"), b"return 1").expect("write Lua 5.1 test");
        fs::write(lua51_directory.join("notes.txt"), b"not a test").expect("write support file");
        fs::write(lua55_directory.join("first.lua"), b"return 1").expect("write Lua 5.5 test");

        let manifest = Sources::parse(SOURCES).expect("valid sources");
        let sources = Sources {
            sources: vec![
                manifest.get(LUA51).expect("Lua 5.1 source").clone(),
                manifest.get(LUA55).expect("Lua 5.5 source").clone(),
            ],
            manifest_directory: None,
        };
        let mut ledger = Ledger {
            entries: vec![
                test_entry(LUA51, "first.lua", "required"),
                test_entry(LUA55, "first.lua", "required"),
            ],
        };
        validate_prepared_ledger_coverage(&ledger, &sources, &root)
            .expect("the Lua 5.1 and Lua 5.5 tests are classified");

        fs::write(lua55_directory.join("second.lua"), b"return 2").expect("write second test");
        let error = validate_prepared_ledger_coverage(&ledger, &sources, &root)
            .expect_err("unclassified Lua test must fail");
        assert!(error.contains("second.lua"));
        assert!(error.contains(LUA55));

        ledger
            .entries
            .push(test_entry(LUA55, "second.lua", "deferred"));
        validate_prepared_ledger_coverage(&ledger, &sources, &root)
            .expect("required and deferred tests are both classified");

        ledger
            .entries
            .push(test_entry(LUA55, "missing.lua", "deferred"));
        let error = validate_prepared_ledger_coverage(&ledger, &sources, &root)
            .expect_err("stale exact ledger path must fail");
        assert!(error.contains("does not name a prepared file"));
        fs::remove_dir_all(&root).expect("remove test corpus");
    }

    #[test]
    fn runner_rejects_wildcard_executable_paths() {
        let sources = Sources::parse(SOURCES).expect("valid sources");
        for state in ["required", "expected-fail"] {
            let ledger = Ledger {
                entries: vec![test_entry(LUAJIT, "test/lang/*.lua", state)],
            };
            let error =
                run_ledger_entries(&ledger, &sources, Path::new("/unused"), |_, _, _, _| {
                    TestExecution::Passed
                })
                .expect_err("wildcard executable entry must be rejected");
            assert!(error.contains("wildcard path"));
        }
    }

    #[test]
    fn launch_uses_corpus_local_lua_path_without_host_inheritance() {
        let corpus_root = Path::new("/corpora/lua51/source");
        let working_directory = corpus_root.join("test").join("lang");
        let launch = TestLaunch::new(&working_directory, corpus_root);

        assert_eq!(launch.working_directory, working_directory);
        assert_eq!(
            launch.lua_path,
            OsString::from(
                "?;/corpora/lua51/source/test/lang/?.lua;/corpora/lua51/source/test/lang/?/init.lua;/corpora/lua51/source/?.lua;/corpora/lua51/source/?/init.lua",
            )
        );
        assert!(launch.clears_host_lua_path);
        assert!(!launch.lua_path.to_string_lossy().contains("host-lua-path"));
    }

    #[test]
    fn relative_paths_resolve_before_child_working_directory_changes() {
        let path = absolute_path(Path::new("target/debug/oximond")).expect("working directory");
        assert!(path.is_absolute());
        assert!(path.ends_with("target/debug/oximond"));
    }

    fn test_entry(corpus: &str, path: &str, state: &str) -> LedgerEntry {
        LedgerEntry {
            corpus: corpus.to_owned(),
            path: path.to_owned(),
            state: state.to_owned(),
            milestone: "test".to_owned(),
            reason: "test entry".to_owned(),
        }
    }

    fn unique_test_directory(label: &str) -> PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("current time")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "oximond-conformance-{label}-{}-{nonce}",
            std::process::id()
        ))
    }
}
