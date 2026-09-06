//! One differential case: a Lua source run by the stock oracle and by
//! oximond in each execution mode, then compared.
//!
//! # What the comparison normalizes
//!
//! - Standard output is compared byte for byte, never through UTF-8.
//! - An uncaught error is compared by its first stderr line without the
//!   interpreter's name prefix (`lua5.4: `, `oximond: runtime error: `).
//!   Exit codes differ by design (stock Lua exits 1, oximond 70), so only
//!   "failed" versus "succeeded" is compared. Tracebacks are ignored.
//! - Oximond's `OXIMOND_JIT_STATS` lines are removed from stderr before
//!   the comparison and kept as JIT coverage.
//!
//! Every program runs as `case.lua` in its own directory, so error
//! positions carry the same chunk name in every engine.

use crate::compare::{Comparison, stdout_equal};
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use crate::profile::Profile;
use crate::run::{Limits, Outcome, Status, run};

/// The file name every engine runs, so error positions agree.
pub const CASE_FILE: &str = "case.lua";
const COMPILE_PROBE_FILE: &str = "compile-only.lua";

/// Whether a source file should run or be rejected by the compiler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expectation {
    Run,
    Reject,
}

/// An oximond execution mode under test.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Mode {
    Interpreter,
    /// Tracing execution with synchronous compilation, so the point where
    /// a trace installs is the same on every run.
    Tracing,
    /// Tracing execution with background compilation, as the CLI runs by
    /// default. Install timing varies between runs.
    TracingAsync,
    /// Tracing execution with a trace profile recorded by an earlier
    /// Cranelift run of the same program and loaded into a Cranelift run.
    TracingProfile,
    /// Tracing execution with the MIR backend.
    TracingMir,
    /// Like [`Mode::TracingProfile`] with MIR for both the recording run and
    /// the loading run.
    TracingMirProfile,
    /// A profile recorded under Cranelift and loaded under MIR: profiles
    /// must not depend on the backend.
    TracingMirCrossProfile,
    /// MIR with background compilation.
    TracingMirAsync,
    /// MIR with no Cranelift at all (`OXIMOND_JIT_BACKEND=mir-only`): a
    /// tree MIR declines stays in the interpreter, sides compiled apart are
    /// declined and no tree reaches the Peak tier.
    TracingMirOnly,
}

const CRANELIFT_SYNC: &[(&str, &str)] = &[
    ("OXIMOND_JIT_SYNC", "1"),
    ("OXIMOND_JIT_STATS", "1"),
    ("OXIMOND_JIT_BACKEND", "cranelift"),
];
const MIR_SYNC: &[(&str, &str)] = &[
    ("OXIMOND_JIT_SYNC", "1"),
    ("OXIMOND_JIT_STATS", "1"),
    ("OXIMOND_JIT_BACKEND", "mir"),
];
const MIR_ONLY_SYNC: &[(&str, &str)] = &[
    ("OXIMOND_JIT_SYNC", "1"),
    ("OXIMOND_JIT_STATS", "1"),
    ("OXIMOND_JIT_BACKEND", "mir-only"),
];

impl Mode {
    /// The modes of a default campaign against stock Lua: the interpreter
    /// and the [`Mode::JIT`] variants. Each is compared with the stock
    /// interpreter, and the variants with oximond's interpreter too.
    pub const ALL: [Self; 5] = [
        Self::Interpreter,
        Self::TracingMir,
        Self::TracingMirProfile,
        Self::TracingMirOnly,
        Self::Tracing,
    ];
    /// The variants `run-jit` compares against oximond's interpreter: the
    /// JIT as it ships (MIR with Cranelift for what MIR declines), the same
    /// with a loaded trace profile, MIR alone and Cranelift alone.
    pub const JIT: [Self; 4] = [
        Self::TracingMir,
        Self::TracingMirProfile,
        Self::TracingMirOnly,
        Self::Tracing,
    ];
    /// Every mode `--modes` accepts.
    pub const EVERY: [Self; 9] = [
        Self::Interpreter,
        Self::Tracing,
        Self::TracingAsync,
        Self::TracingProfile,
        Self::TracingMir,
        Self::TracingMirProfile,
        Self::TracingMirCrossProfile,
        Self::TracingMirAsync,
        Self::TracingMirOnly,
    ];

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Interpreter => "interpreter",
            Self::Tracing => "tracing",
            Self::TracingAsync => "tracing-async",
            Self::TracingProfile => "tracing-profile",
            Self::TracingMir => "tracing-mir",
            Self::TracingMirProfile => "tracing-mir-profile",
            Self::TracingMirCrossProfile => "tracing-mir-crossprofile",
            Self::TracingMirAsync => "tracing-mir-async",
            Self::TracingMirOnly => "tracing-mir-only",
        }
    }

    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        Self::EVERY.into_iter().find(|mode| mode.name() == text)
    }

    const fn execution(self) -> &'static str {
        match self {
            Self::Interpreter => "interpreter",
            _ => "tracing",
        }
    }

    /// Whether the mode compiles through the MIR backend.
    #[must_use]
    pub const fn uses_mir(self) -> bool {
        matches!(
            self,
            Self::TracingMir
                | Self::TracingMirProfile
                | Self::TracingMirCrossProfile
                | Self::TracingMirAsync
                | Self::TracingMirOnly
        )
    }

    /// For a mode that loads a trace profile, the mode that records it: a
    /// separate earlier run of the same program.
    #[must_use]
    pub const fn profile_recorder(self) -> Option<Self> {
        match self {
            Self::TracingProfile | Self::TracingMirCrossProfile => Some(Self::Tracing),
            Self::TracingMirProfile => Some(Self::TracingMir),
            _ => None,
        }
    }

    /// Environment the mode runs with. Both tracing modes report JIT
    /// statistics; the interpreter sets the synchronous flag too so that
    /// only `--execution` differs between it and `Tracing`.
    #[must_use]
    pub const fn environment(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::Interpreter => &[("OXIMOND_JIT_SYNC", "1")],
            Self::Tracing | Self::TracingProfile => CRANELIFT_SYNC,
            Self::TracingAsync => &[
                ("OXIMOND_JIT_SYNC", "0"),
                ("OXIMOND_JIT_STATS", "1"),
                ("OXIMOND_JIT_BACKEND", "cranelift"),
            ],
            Self::TracingMir | Self::TracingMirProfile | Self::TracingMirCrossProfile => MIR_SYNC,
            Self::TracingMirOnly => MIR_ONLY_SYNC,
            Self::TracingMirAsync => &[
                ("OXIMOND_JIT_SYNC", "0"),
                ("OXIMOND_JIT_STATS", "1"),
                ("OXIMOND_JIT_BACKEND", "mir"),
            ],
        }
    }
}

/// A stock interpreter verified to match a profile.
#[derive(Clone, Debug)]
pub struct Oracle {
    pub command: PathBuf,
    pub banner: String,
    /// Arguments for a second, equally valid run of the same oracle. Stock
    /// `LuaJIT`'s trace compiler folds `x ^ 2` into `x * x` while its
    /// interpreter calls `pow`, so its output can depend on what it
    /// compiled; `-joff` gives the interpreter's answer. A mode that matches
    /// either run passes.
    pub alternate: Vec<OsString>,
    /// Arguments before the file name on every run: for the interpreter
    /// oracle, `--compat P --execution interpreter`.
    pub arguments: Vec<OsString>,
    /// Oximond's own interpreter. Its uncaught runtime errors are results
    /// to compare, not generator bugs.
    pub interpreter: bool,
}

impl Oracle {
    /// Oximond's interpreter as the oracle for `profile`: the property under
    /// test is that the JIT variants match it, so no stock interpreter is
    /// needed.
    #[must_use]
    pub fn interpreter(oximond: &Path, profile: Profile) -> Self {
        Self {
            command: oximond.to_path_buf(),
            banner: format!("oximond interpreter ({profile})"),
            alternate: Vec::new(),
            arguments: vec![
                "--compat".into(),
                profile.name().into(),
                "--execution".into(),
                "interpreter".into(),
            ],
            interpreter: true,
        }
    }
}

/// Finds `command` and checks that its `-v` banner names `profile`'s
/// version. A wrong version is an error, never a silent substitute.
///
/// # Errors
///
/// Returns why no matching oracle is available.
pub fn find_oracle(profile: Profile, command: Option<&Path>) -> Result<Oracle, String> {
    let command = command.map_or_else(
        || PathBuf::from(profile.default_oracle()),
        Path::to_path_buf,
    );
    // Cases run in separate directories. Resolve explicit paths while still
    // in the caller's directory, but leave bare commands to PATH lookup.
    let command = if command.components().count() > 1 {
        fs::canonicalize(&command)
            .map_err(|error| format!("{profile} oracle {}: {error}", command.display()))?
    } else {
        command
    };
    let limits = Limits {
        memory_mib: 0,
        ..Limits::default()
    };
    let outcome = run(
        &[command.clone().into(), "-v".into()],
        Path::new("."),
        &[],
        limits,
    );
    if let Status::SpawnFailed(error) = &outcome.status {
        return Err(format!("{profile} oracle unavailable: {error}"));
    }
    let mut text = outcome.stdout;
    text.extend_from_slice(&outcome.stderr);
    let banner = String::from_utf8_lossy(&text)
        .lines()
        .next()
        .unwrap_or_default()
        .to_owned();
    if !banner.starts_with(profile.oracle_banner()) {
        return Err(format!(
            "{} reports `{banner}`, not {}",
            command.display(),
            profile.oracle_banner()
        ));
    }
    let alternate = if profile == Profile::LuaJit21 {
        vec![OsString::from("-joff")]
    } else {
        Vec::new()
    };
    Ok(Oracle {
        command,
        banner,
        alternate,
        arguments: Vec::new(),
        interpreter: false,
    })
}

/// The comparable part of one run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Observation {
    pub stdout: Vec<u8>,
    pub end: End,
}

/// How a run ended, normalized across engines.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum End {
    Success,
    /// An uncaught Lua error with its normalized first stderr line.
    Error(String),
    Crash(String),
    Timeout,
    OutputCap,
    SpawnFailed(String),
}

impl End {
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::Success => "success".to_owned(),
            Self::Error(message) => format!("error: {message}"),
            Self::Crash(how) => format!("crash: {how}"),
            Self::Timeout => "timeout".to_owned(),
            Self::OutputCap => "output cap".to_owned(),
            Self::SpawnFailed(error) => format!("spawn failed: {error}"),
        }
    }
}

/// Splits oximond's JIT statistics lines out of stderr.
#[must_use]
pub fn split_jit_statistics(stderr: &[u8]) -> (Vec<u8>, Vec<String>) {
    let mut rest = Vec::new();
    let mut statistics = Vec::new();
    for line in stderr.split_inclusive(|&byte| byte == b'\n') {
        let text = String::from_utf8_lossy(line);
        if ["jit: ", "interp: ", "trace: ", "mir: "]
            .iter()
            .any(|prefix| text.starts_with(prefix))
        {
            statistics.push(text.trim_end().to_owned());
        } else {
            rest.extend_from_slice(line);
        }
    }
    (rest, statistics)
}

/// Removes the interpreter-name prefix from an uncaught-error line. The
/// rest stays bytes and is escaped, so distinct invalid UTF-8 never
/// compares equal.
#[must_use]
pub fn normalize_error_line(line: &[u8]) -> String {
    let mut text = line.strip_suffix(b"\r").unwrap_or(line);
    for prefix in [
        &b"lua5.1: "[..],
        b"lua5.2: ",
        b"lua5.3: ",
        b"lua5.4: ",
        b"lua5.5: ",
        b"lua: ",
        b"luajit: ",
        b"oximond: ",
    ] {
        if let Some(rest) = text.strip_prefix(prefix) {
            text = rest;
            break;
        }
    }
    for prefix in [&b"runtime error: "[..], b"compile error: "] {
        if let Some(rest) = text.strip_prefix(prefix) {
            text = rest;
            break;
        }
    }
    escape(text)
}

/// Normalizes a finished run.
#[must_use]
pub fn observe(outcome: &Outcome) -> Observation {
    let (stderr, _) = split_jit_statistics(&outcome.stderr);
    let end = match &outcome.status {
        Status::Exited(0) => End::Success,
        // Rust panics exit 101 and are crashes, not Lua errors.
        Status::Exited(101) => End::Crash("panic (exit 101)".to_owned()),
        Status::Exited(_) => {
            let first = stderr
                .split(|&byte| byte == b'\n')
                .next()
                .unwrap_or_default();
            End::Error(normalize_error_line(first))
        }
        Status::Signaled(signal) => End::Crash(format!("signal {signal}")),
        Status::TimedOut => End::Timeout,
        Status::OutputCapExceeded => End::OutputCap,
        Status::SpawnFailed(error) => End::SpawnFailed(error.clone()),
    };
    Observation {
        stdout: outcome.stdout.clone(),
        end,
    }
}

/// JIT activity in one tracing run, read from `OXIMOND_JIT_STATS`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct JitActivity {
    pub traces_compiled: u64,
    /// Entries into native trace code.
    pub native_entries: u64,
    /// Loop iterations completed inside native trace code.
    pub iterations: u64,
    /// Native exits back to the interpreter (snapshot restores).
    pub exits: u64,
    /// Guards that failed, by any kind.
    pub side_exits: u64,
    /// Side traces installed into a tree.
    pub side_installs: u64,
    /// Trees the MIR backend compiled (zero unless run under MIR).
    pub mir_accepted_trees: u64,
    /// Trees MIR handed back to Cranelift.
    pub mir_fallback_trees: u64,
    pub raw: Vec<String>,
}

impl JitActivity {
    /// Compiled code ran at least once.
    #[must_use]
    pub const fn productive(&self) -> bool {
        self.native_entries > 0
    }

    #[must_use]
    pub fn parse(lines: Vec<String>) -> Self {
        let mut activity = Self::default();
        for line in &lines {
            if line.starts_with("mir: ") {
                for (key, value) in numbers(line) {
                    match key.as_str() {
                        "accepted_trees" => activity.mir_accepted_trees += value,
                        "fallback_trees" => activity.mir_fallback_trees += value,
                        _ => {}
                    }
                }
                continue;
            }
            for (key, value) in numbers(line) {
                match key.as_str() {
                    "traces_compiled" => activity.traces_compiled += value,
                    "native_entries" => activity.native_entries += value,
                    "guard_exits" => activity.side_exits += value,
                    "iterations" => activity.iterations += value,
                    "side_installs" => activity.side_installs += value,
                    _ => {}
                }
            }
            if let Some(exits) = list_after(line, "side_exits: [") {
                activity.side_exits += exits;
            }
            if let Some(exits) = list_after(line, ", exits: [") {
                activity.exits += exits;
            }
        }
        activity.raw = lines;
        activity
    }
}

/// `key=value` and `key: value` pairs with integer values.
fn numbers(line: &str) -> Vec<(String, u64)> {
    let mut pairs = Vec::new();
    for token in line.split([' ', ',', '{', '}']) {
        if let Some((key, value)) = token.split_once('=')
            && let Ok(value) = value.parse()
        {
            pairs.push((key.to_owned(), value));
        }
    }
    let mut rest = line;
    while let Some(colon) = rest.find(": ") {
        let key = rest[..colon]
            .rsplit([' ', '{', ','])
            .next()
            .unwrap_or_default();
        let after = &rest[colon + 2..];
        let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
        if let Ok(value) = digits.parse() {
            pairs.push((key.to_owned(), value));
        }
        rest = after;
    }
    pairs
}

fn list_after(line: &str, marker: &str) -> Option<u64> {
    let start = line.find(marker)? + marker.len();
    let end = line[start..].find(']')? + start;
    Some(
        line[start..end]
            .split(',')
            .filter_map(|item| item.trim().parse::<u64>().ok())
            .sum(),
    )
}

/// Everything needed to run cases.
#[derive(Clone, Debug)]
pub struct Engines {
    pub oximond: PathBuf,
    pub limits: Limits,
    pub modes: Vec<Mode>,
    pub trace_profile_record_dir: Option<PathBuf>,
    pub trace_profile_load_dir: Option<PathBuf>,
    /// Numeric tolerance used only when comparing against the oracle.
    pub comparison: Comparison,
}

/// One engine's run of a case.
#[derive(Clone, Debug)]
pub struct EngineRun {
    pub label: String,
    pub command: Vec<OsString>,
    pub environment: Vec<(String, String)>,
    pub outcome: Outcome,
    pub observation: Observation,
    pub jit: Option<JitActivity>,
}

/// Runs the oracle on `case.lua` in `directory`.
#[must_use]
pub fn run_oracle(oracle: &Oracle, directory: &Path, limits: Limits) -> EngineRun {
    run_oracle_with(oracle, directory, limits, &[], "oracle")
}

fn run_oracle_with(
    oracle: &Oracle,
    directory: &Path,
    limits: Limits,
    arguments: &[OsString],
    label: &str,
) -> EngineRun {
    run_oracle_file_with(oracle, directory, limits, arguments, label, CASE_FILE)
}

fn run_oracle_file_with(
    oracle: &Oracle,
    directory: &Path,
    limits: Limits,
    arguments: &[OsString],
    label: &str,
    file: &str,
) -> EngineRun {
    let mut command = vec![oracle.command.clone().into_os_string()];
    command.extend(oracle.arguments.iter().cloned());
    command.extend(arguments.iter().cloned());
    command.push(file.into());
    let outcome = run(&command, directory, &[], limits);
    EngineRun {
        label: label.to_owned(),
        observation: observe(&outcome),
        command: crate::run::command_line(&command, limits),
        environment: Vec::new(),
        outcome,
        jit: None,
    }
}

/// Runs oximond in `mode` on `case.lua` in `directory`.
#[must_use]
pub fn run_oximond(engines: &Engines, profile: Profile, mode: Mode, directory: &Path) -> EngineRun {
    run_oximond_file(engines, profile, mode, directory, CASE_FILE)
}

fn run_oximond_file(
    engines: &Engines,
    profile: Profile,
    mode: Mode,
    directory: &Path,
    file: &str,
) -> EngineRun {
    if let Some(recorder) = mode.profile_recorder() {
        return run_profile_variant(engines, profile, mode, recorder, directory, file);
    }
    let mut command: Vec<OsString> = vec![
        engines.oximond.clone().into_os_string(),
        "--compat".into(),
        profile.name().into(),
        "--execution".into(),
        mode.execution().into(),
    ];
    if mode != Mode::Interpreter {
        if let Some(root) = &engines.trace_profile_load_dir {
            command.push("--trace-profile-load".into());
            command.push(trace_profile_path(root, directory, profile, mode, file).into_os_string());
        }
        if let Some(root) = &engines.trace_profile_record_dir {
            let path = trace_profile_path(root, directory, profile, mode, file);
            if let Some(parent) = path.parent()
                && let Err(error) = fs::create_dir_all(parent)
            {
                return failed_profile_run(
                    mode,
                    format!("cannot create {}: {error}", parent.display()),
                );
            }
            command.push("--trace-profile-record".into());
            command.push(path.into_os_string());
        }
    }
    command.push(file.into());
    let outcome = run(&command, directory, mode.environment(), engines.limits);
    let (_, statistics) = split_jit_statistics(&outcome.stderr);
    let jit = (mode != Mode::Interpreter).then(|| JitActivity::parse(statistics));
    EngineRun {
        label: mode.name().to_owned(),
        observation: observe(&outcome),
        command: crate::run::command_line(&command, engines.limits),
        environment: mode
            .environment()
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect(),
        outcome,
        jit,
    }
}

/// A variant that loads a profile: records it with `recorder` in an earlier
/// run, then runs with the profile loaded. The recording run is a tracing run
/// of its own. If it leaves no profile, it stands in for the variant, so a
/// crash there still shows.
fn run_profile_variant(
    engines: &Engines,
    profile: Profile,
    mode: Mode,
    recorder: Mode,
    directory: &Path,
    file: &str,
) -> EngineRun {
    let root = directory.join("trace-profiles");
    let path = trace_profile_path(&root, directory, profile, recorder, file);
    if let Some(parent) = path.parent()
        && let Err(error) = fs::create_dir_all(parent)
    {
        return failed_profile_run(mode, format!("cannot create {}: {error}", parent.display()));
    }
    let _ = fs::remove_file(&path);
    let mut command = traced_command(engines, profile, recorder);
    command.push("--trace-profile-record".into());
    command.push(path.clone().into_os_string());
    command.push(file.into());
    let mut recording = traced_run(engines, recorder, directory, &command);
    if !path.exists() {
        mode.name().clone_into(&mut recording.label);
        return recording;
    }
    let mut command = traced_command(engines, profile, mode);
    command.push("--trace-profile-load".into());
    command.push(path.into_os_string());
    command.push(file.into());
    traced_run(engines, mode, directory, &command)
}

fn traced_command(engines: &Engines, profile: Profile, mode: Mode) -> Vec<OsString> {
    vec![
        engines.oximond.clone().into_os_string(),
        "--compat".into(),
        profile.name().into(),
        "--execution".into(),
        mode.execution().into(),
    ]
}

/// How much longer a run that loads a trace profile may take. A loaded
/// profile keeps the interpreter off its fast loop for the whole program (see
/// `has_loaded_profile`), so an interpreter-bound program ran about 7 times as
/// long as without the profile in the stage-3 campaign. That is a speed
/// cliff and not a wrong answer. Four times the timeout covers the programs
/// the oracle finishes in under a third of the base timeout; slower ones can
/// still time out, and a timeout there is a harness limit, not a JIT bug.
const PROFILE_LOAD_TIMEOUT_FACTOR: u32 = 4;

/// `limits`, with the longer timeout of a command that loads a trace profile.
fn limits_for(command: &[OsString], mut limits: Limits) -> Limits {
    if command
        .iter()
        .any(|argument| argument == "--trace-profile-load")
    {
        limits.timeout *= PROFILE_LOAD_TIMEOUT_FACTOR;
    }
    limits
}

fn traced_run(engines: &Engines, mode: Mode, directory: &Path, command: &[OsString]) -> EngineRun {
    let limits = limits_for(command, engines.limits);
    let outcome = run(command, directory, mode.environment(), limits);
    let (_, statistics) = split_jit_statistics(&outcome.stderr);
    EngineRun {
        label: mode.name().to_owned(),
        observation: observe(&outcome),
        command: crate::run::command_line(command, limits),
        environment: mode
            .environment()
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect(),
        outcome,
        jit: Some(JitActivity::parse(statistics)),
    }
}

fn trace_profile_path(
    root: &Path,
    directory: &Path,
    profile: Profile,
    mode: Mode,
    file: &str,
) -> PathBuf {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let source = fs::read(directory.join(file))
        .unwrap_or_else(|_| directory.join(file).as_os_str().as_encoded_bytes().to_vec());
    for byte in source {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
    }
    root.join(format!("{hash:016x}")).join(format!(
        "{}-{}-{file}.trace-profile",
        profile.name(),
        mode.name()
    ))
}

#[cfg(test)]
mod trace_profile_tests {
    use super::{Mode, Profile, trace_profile_path};
    use std::fs;
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn identical_sources_share_a_profile_across_work_directories() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let temporary = std::env::temp_dir().join(format!(
            "oximond-profile-path-{}-{nonce}",
            std::process::id()
        ));
        let first_dir = temporary.join("first");
        let second_dir = temporary.join("second");
        fs::create_dir_all(&first_dir).unwrap();
        fs::create_dir_all(&second_dir).unwrap();
        fs::write(first_dir.join("case.lua"), b"print(1)").unwrap();
        fs::write(second_dir.join("case.lua"), b"print(1)").unwrap();
        let root = Path::new("/profiles");
        let first = trace_profile_path(root, &first_dir, Profile::Lua54, Mode::Tracing, "case.lua");
        let same_source =
            trace_profile_path(root, &second_dir, Profile::Lua54, Mode::Tracing, "case.lua");
        let async_mode = trace_profile_path(
            root,
            &first_dir,
            Profile::Lua54,
            Mode::TracingAsync,
            "case.lua",
        );
        assert_eq!(first, same_source);
        assert_ne!(first, async_mode);
        fs::write(second_dir.join("case.lua"), b"print(2)").unwrap();
        let changed_source =
            trace_profile_path(root, &second_dir, Profile::Lua54, Mode::Tracing, "case.lua");
        assert_ne!(first, changed_source);
        assert!(first.starts_with(root));
        fs::remove_dir_all(temporary).unwrap();
    }
}

fn failed_profile_run(mode: Mode, message: String) -> EngineRun {
    let outcome = Outcome {
        stdout: Vec::new(),
        stderr: message.into_bytes(),
        status: Status::SpawnFailed("trace profile directory failed".to_owned()),
        elapsed: std::time::Duration::ZERO,
    };
    EngineRun {
        label: mode.name().to_owned(),
        observation: observe(&outcome),
        command: Vec::new(),
        environment: Vec::new(),
        outcome,
        jit: None,
    }
}

/// The verdict for a whole case.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Verdict {
    /// Every mode matched the oracle.
    Pass,
    /// At least one mode differs from the oracle, or a JIT mode from
    /// oximond's interpreter (see [`interpreter_differences`]).
    Mismatch,
    /// The oracle rejected the source at compile time: a generator bug.
    OracleSyntaxError,
    /// The oracle raised an uncaught error: a generator bug.
    OracleRuntimeError,
    /// An expected rejection compiled in the oracle, so the fixture is invalid.
    OracleUnexpectedAcceptance,
    OracleCrash,
    OracleTimeout,
    /// No oracle for the profile, and the JIT modes agree with oximond's
    /// interpreter (or it did not run).
    Inconclusive,
    /// The harness itself failed (an unwritable directory, ...).
    HarnessError,
}

impl Verdict {
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Mismatch => "mismatch",
            Self::OracleSyntaxError => "oracle_syntax_error",
            Self::OracleRuntimeError => "oracle_runtime_error",
            Self::OracleUnexpectedAcceptance => "oracle_unexpected_acceptance",
            Self::OracleCrash => "oracle_crash",
            Self::OracleTimeout => "oracle_timeout",
            Self::Inconclusive => "inconclusive",
            Self::HarnessError => "harness_error",
        }
    }
}

/// How one mode's run differs from the oracle.
#[must_use]
pub fn difference(expected: &Observation, actual: &Observation) -> Option<String> {
    difference_with(expected, actual, Comparison::Exact)
}

/// Compares termination exactly and output with the selected numeric policy.
#[must_use]
pub fn difference_with(
    expected: &Observation,
    actual: &Observation,
    comparison: Comparison,
) -> Option<String> {
    let category = match (&expected.end, &actual.end) {
        (_, End::Crash(_)) => "crash",
        (_, End::Timeout) => "timeout",
        (_, End::OutputCap) => "output_cap",
        (_, End::SpawnFailed(_)) => "spawn_failed",
        (End::Success, End::Error(_)) => "unexpected_error",
        (End::Error(_), End::Success) => "missing_error",
        (End::Error(a), End::Error(b)) if a != b => "error_message",
        _ if !stdout_equal(&expected.stdout, &actual.stdout, comparison) => "stdout",
        _ => return None,
    };
    if category == "stdout" || category == "unexpected_error" || category == "missing_error" {
        let tag = first_difference_with(&expected.stdout, &actual.stdout, comparison)
            .map(|line| statement_tag(&line.expected));
        if let Some(tag) = tag {
            return Some(format!("{category}@{tag}"));
        }
    }
    Some(category.to_owned())
}

/// The middle part of a signature entry that compares a mode with oximond's
/// own interpreter instead of the oracle: `tracing-mir:vs-interpreter:stdout@o12`.
pub const VS_INTERPRETER: &str = "vs-interpreter";

/// How each mode differs from oximond's interpreter run among `modes`, as
/// signature entries `mode:vs-interpreter:category`. Empty when the
/// interpreter did not run. The comparison is exact: both runs are oximond.
#[must_use]
pub fn interpreter_differences(modes: &[EngineRun]) -> Vec<String> {
    let Some(reference) = modes
        .iter()
        .find(|run| run.label == Mode::Interpreter.name())
    else {
        return Vec::new();
    };
    modes
        .iter()
        .filter(|run| run.label != reference.label && run.observation != reference.observation)
        .filter_map(|run| {
            difference(&reference.observation, &run.observation)
                .map(|category| format!("{}:{VS_INTERPRETER}:{category}", run.label))
        })
        .collect()
}

/// Adds the comparison with oximond's interpreter to a signature against a
/// stock oracle. When the interpreter matched the oracle, a mode that
/// differs from one differs from the other and the signature already says
/// so. When it did not, or there is no oracle, a JIT mode can still be
/// wrong against the interpreter, and that is a bug of its own.
fn with_interpreter_differences(mut signature: Vec<String>, modes: &[EngineRun]) -> Vec<String> {
    let interpreter = format!("{}:", Mode::Interpreter.name());
    let interpreter_ran = modes
        .iter()
        .any(|run| run.label == Mode::Interpreter.name());
    if interpreter_ran
        && signature
            .iter()
            .any(|entry| entry.starts_with(&interpreter))
    {
        signature.extend(interpreter_differences(modes));
    }
    signature.sort();
    signature
}

/// The statement tag of an observation line (`o12#2 ...` is `o12`). The
/// reducer keeps tags as they are, so a reduced program that still fails at
/// the same statement keeps its signature.
fn statement_tag(line: &str) -> String {
    let tag = line.split('#').next().unwrap_or_default();
    if tag.is_empty() || tag.len() > 24 || !tag.chars().all(|c| c.is_ascii_alphanumeric()) {
        "?".to_owned()
    } else {
        tag.to_owned()
    }
}

/// The first output line that differs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineDifference {
    pub line: usize,
    pub expected: String,
    pub actual: String,
}

#[must_use]
pub fn first_difference(expected: &[u8], actual: &[u8]) -> Option<LineDifference> {
    first_difference_with(expected, actual, Comparison::Exact)
}

/// The first line that differs after applying the numeric policy.
#[must_use]
pub fn first_difference_with(
    expected: &[u8],
    actual: &[u8],
    comparison: Comparison,
) -> Option<LineDifference> {
    let expected_lines: Vec<&[u8]> = expected.split(|&byte| byte == b'\n').collect();
    let actual_lines: Vec<&[u8]> = actual.split(|&byte| byte == b'\n').collect();
    let count = expected_lines.len().max(actual_lines.len());
    (0..count).find_map(|index| {
        let a = expected_lines.get(index);
        let b = actual_lines.get(index);
        let equal = match (a, b) {
            (Some(a), Some(b)) => stdout_equal(a, b, comparison),
            _ => a == b,
        };
        (!equal).then(|| LineDifference {
            line: index + 1,
            expected: a.map_or_else(|| "<missing>".to_owned(), |bytes| escape(bytes)),
            actual: b.map_or_else(|| "<missing>".to_owned(), |bytes| escape(bytes)),
        })
    })
}

/// Bytes as text with non-printable bytes as `\xNN`.
#[must_use]
pub fn escape(bytes: &[u8]) -> String {
    let mut text = String::new();
    for &byte in bytes {
        if byte == b'\\' {
            text.push_str("\\\\");
        } else if (0x20..0x7f).contains(&byte) {
            text.push(char::from(byte));
        } else {
            let _ = write!(text, "\\x{byte:02x}");
        }
    }
    text
}

/// A case after all its runs.
#[derive(Clone, Debug)]
pub struct CaseResult {
    pub verdict: Verdict,
    pub oracle: Option<EngineRun>,
    pub modes: Vec<EngineRun>,
    /// `mode:category` for each mode that differs, sorted.
    pub signature: Vec<String>,
    /// For an inconclusive case: whether every mode printed the same.
    pub modes_agree: bool,
    /// The oracle's alternate run, made when a mode differed from the
    /// first run.
    pub oracle_alternate: Option<EngineRun>,
}

impl CaseResult {
    /// Compiled code ran in some tracing mode.
    #[must_use]
    pub fn jit_productive(&self) -> bool {
        self.modes
            .iter()
            .any(|run| run.jit.as_ref().is_some_and(JitActivity::productive))
    }
}

/// Writes `source` as `case.lua` in `directory` and runs every engine.
///
/// # Errors
///
/// Returns an error if the directory cannot be written.
pub fn run_case(
    source: &str,
    directory: &Path,
    profile: Profile,
    oracle: Option<&Oracle>,
    engines: &Engines,
) -> Result<CaseResult, String> {
    fs::create_dir_all(directory).map_err(|error| format!("{}: {error}", directory.display()))?;
    fs::write(directory.join(CASE_FILE), source)
        .map_err(|error| format!("{}: {error}", directory.display()))?;
    let oracle_run = oracle.map(|oracle| run_oracle(oracle, directory, engines.limits));
    if let Some(run) = &oracle_run {
        let verdict = match &run.observation.end {
            End::Success => None,
            End::Error(message) if is_syntax_error(message) => Some(Verdict::OracleSyntaxError),
            // The interpreter's error is the result the variants must match.
            End::Error(_) if oracle.is_some_and(|oracle| oracle.interpreter) => None,
            End::Error(_) => Some(Verdict::OracleRuntimeError),
            End::Timeout | End::OutputCap => Some(Verdict::OracleTimeout),
            End::Crash(_) | End::SpawnFailed(_) => Some(Verdict::OracleCrash),
        };
        if let Some(verdict) = verdict {
            return Ok(CaseResult {
                verdict,
                oracle: oracle_run,
                modes: Vec::new(),
                signature: Vec::new(),
                modes_agree: false,
                oracle_alternate: None,
            });
        }
    }
    let modes: Vec<EngineRun> = engines
        .modes
        .iter()
        .map(|&mode| run_oximond(engines, profile, mode, directory))
        .collect();
    let modes_agree = modes
        .windows(2)
        .all(|pair| pair[0].observation == pair[1].observation);
    let Some(oracle_run) = oracle_run else {
        // No oracle: the JIT modes can still be judged by the interpreter.
        let mut signature = interpreter_differences(&modes);
        signature.sort();
        return Ok(CaseResult {
            verdict: if signature.is_empty() {
                Verdict::Inconclusive
            } else {
                Verdict::Mismatch
            },
            oracle: None,
            modes,
            signature,
            modes_agree,
            oracle_alternate: None,
        });
    };
    let mut signature: Vec<String> = modes
        .iter()
        .filter_map(|run| {
            difference_with(
                &oracle_run.observation,
                &run.observation,
                engines.comparison,
            )
            .map(|category| format!("{}:{category}", run.label))
        })
        .collect();
    signature.sort();
    let mut oracle_alternate = None;
    if let Some(oracle) =
        oracle.filter(|oracle| !signature.is_empty() && !oracle.alternate.is_empty())
    {
        let alternate = run_oracle_with(
            oracle,
            directory,
            engines.limits,
            &oracle.alternate,
            "oracle-alternate",
        );
        if alternate.observation.end == End::Success {
            signature.retain(|entry| {
                let label = entry.split(':').next().unwrap_or_default();
                modes
                    .iter()
                    .find(|run| run.label == label)
                    .is_none_or(|run| {
                        difference_with(
                            &alternate.observation,
                            &run.observation,
                            engines.comparison,
                        )
                        .is_some()
                    })
            });
        }
        oracle_alternate = Some(alternate);
    }
    if !oracle.is_some_and(|oracle| oracle.interpreter) {
        signature = with_interpreter_differences(signature, &modes);
    }
    let verdict = if signature.is_empty() {
        Verdict::Pass
    } else {
        Verdict::Mismatch
    };
    Ok(CaseResult {
        verdict,
        oracle: Some(oracle_run),
        modes,
        signature,
        modes_agree,
        oracle_alternate,
    })
}

/// Checks the selected source expectation. Rejection checks compile the source
/// through `loadstring`/`load` without executing it, and compare only whether
/// the compiler accepted it. The original source remains in `case.lua`.
///
/// # Errors
///
/// Returns an error if the work directory cannot be written.
pub fn run_case_with_expectation(
    source: &str,
    directory: &Path,
    profile: Profile,
    oracle: Option<&Oracle>,
    engines: &Engines,
    expectation: Expectation,
) -> Result<CaseResult, String> {
    if expectation == Expectation::Run {
        return run_case(source, directory, profile, oracle, engines);
    }
    fs::create_dir_all(directory).map_err(|error| format!("{}: {error}", directory.display()))?;
    fs::write(directory.join(CASE_FILE), source)
        .map_err(|error| format!("{}: {error}", directory.display()))?;
    let mut literal = String::with_capacity(source.len() * 4 + 2);
    literal.push('"');
    for byte in source.bytes() {
        let _ = write!(literal, "\\{byte:03}");
    }
    literal.push('"');
    let probe = format!(
        "local compile = loadstring or load\nlocal function_result = compile({literal}, '@case.lua')\nprint(function_result and 'accepted' or 'rejected')\n"
    );
    fs::write(directory.join(COMPILE_PROBE_FILE), probe)
        .map_err(|error| format!("{}: {error}", directory.display()))?;

    let oracle_run = oracle.map(|oracle| {
        run_oracle_file_with(
            oracle,
            directory,
            engines.limits,
            &[],
            "oracle",
            COMPILE_PROBE_FILE,
        )
    });
    if let Some(run) = &oracle_run {
        let verdict = match &run.observation.end {
            End::Success if run.observation.stdout == b"rejected\n" => None,
            End::Success if run.observation.stdout == b"accepted\n" => {
                Some(Verdict::OracleUnexpectedAcceptance)
            }
            End::Success | End::Error(_) => Some(Verdict::OracleRuntimeError),
            End::Timeout | End::OutputCap => Some(Verdict::OracleTimeout),
            End::Crash(_) | End::SpawnFailed(_) => Some(Verdict::OracleCrash),
        };
        if let Some(verdict) = verdict {
            return Ok(CaseResult {
                verdict,
                oracle: oracle_run,
                modes: Vec::new(),
                signature: Vec::new(),
                modes_agree: false,
                oracle_alternate: None,
            });
        }
    }
    let modes: Vec<EngineRun> = engines
        .modes
        .iter()
        .map(|&mode| run_oximond_file(engines, profile, mode, directory, COMPILE_PROBE_FILE))
        .collect();
    let modes_agree = modes
        .windows(2)
        .all(|pair| pair[0].observation == pair[1].observation);
    let Some(oracle_run) = oracle_run else {
        // No oracle: the JIT modes can still be judged by the interpreter.
        let mut signature = interpreter_differences(&modes);
        signature.sort();
        return Ok(CaseResult {
            verdict: if signature.is_empty() {
                Verdict::Inconclusive
            } else {
                Verdict::Mismatch
            },
            oracle: None,
            modes,
            signature,
            modes_agree,
            oracle_alternate: None,
        });
    };
    let mut signature: Vec<String> = modes
        .iter()
        .filter_map(|run| {
            difference(&oracle_run.observation, &run.observation)
                .map(|category| format!("{}:{category}", run.label))
        })
        .collect();
    if !oracle.is_some_and(|oracle| oracle.interpreter) {
        signature = with_interpreter_differences(signature, &modes);
    }
    signature.sort();
    let verdict = if signature.is_empty() {
        Verdict::Pass
    } else {
        Verdict::Mismatch
    };
    Ok(CaseResult {
        verdict,
        oracle: Some(oracle_run),
        modes,
        signature,
        modes_agree,
        oracle_alternate: None,
    })
}

fn is_syntax_error(message: &str) -> bool {
    [
        "syntax error",
        "unexpected symbol",
        "expected near",
        "' expected",
        "malformed number",
        "unfinished string",
    ]
    .iter()
    .any(|marker| message.contains(marker))
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_profile_load_gets_a_longer_timeout_than_other_runs() {
        let limits = Limits::default();
        let plain: Vec<OsString> = vec!["oximond".into(), "case.lua".into()];
        let load: Vec<OsString> = vec![
            "oximond".into(),
            "--trace-profile-load".into(),
            "p".into(),
            "case.lua".into(),
        ];
        assert_eq!(limits_for(&plain, limits).timeout, limits.timeout);
        assert_eq!(
            limits_for(&load, limits).timeout,
            limits.timeout * PROFILE_LOAD_TIMEOUT_FACTOR
        );
    }

    use std::ffi::OsString;

    use super::{
        End, JitActivity, Limits, Observation, PROFILE_LOAD_TIMEOUT_FACTOR, difference, escape,
        limits_for, normalize_error_line, split_jit_statistics,
    };

    #[test]
    fn every_mode_name_parses_back_and_the_defaults_stay() {
        use super::Mode;
        for mode in Mode::EVERY {
            assert_eq!(Mode::parse(mode.name()), Some(mode));
        }
        assert_eq!(Mode::ALL[0], Mode::Interpreter);
        assert_eq!(Mode::ALL[1..], Mode::JIT);
        let names: Vec<&str> = Mode::JIT.iter().map(|mode| mode.name()).collect();
        assert_eq!(
            names,
            [
                "tracing-mir",
                "tracing-mir-profile",
                "tracing-mir-only",
                "tracing",
            ]
        );
    }

    #[test]
    fn mir_variants_select_the_mir_backend_and_profile_variants_name_a_recorder() {
        use super::Mode;
        let backend = |mode: Mode| {
            mode.environment()
                .iter()
                .find(|(key, _)| *key == "OXIMOND_JIT_BACKEND")
                .map(|(_, value)| *value)
        };
        assert_eq!(backend(Mode::Tracing), Some("cranelift"));
        assert_eq!(backend(Mode::TracingProfile), Some("cranelift"));
        for mode in [
            Mode::TracingMir,
            Mode::TracingMirProfile,
            Mode::TracingMirCrossProfile,
            Mode::TracingMirAsync,
        ] {
            assert_eq!(backend(mode), Some("mir"), "{}", mode.name());
            assert!(mode.uses_mir());
        }
        assert_eq!(backend(Mode::TracingMirOnly), Some("mir-only"));
        assert!(Mode::TracingMirOnly.uses_mir());
        assert_eq!(Mode::TracingMirOnly.profile_recorder(), None);
        assert_eq!(Mode::TracingProfile.profile_recorder(), Some(Mode::Tracing));
        assert_eq!(
            Mode::TracingMirProfile.profile_recorder(),
            Some(Mode::TracingMir)
        );
        // Cross: recorded under Cranelift, loaded under MIR.
        assert_eq!(
            Mode::TracingMirCrossProfile.profile_recorder(),
            Some(Mode::Tracing)
        );
        assert_eq!(Mode::Tracing.profile_recorder(), None);
        let sync = |mode: Mode| {
            mode.environment()
                .iter()
                .find(|(key, _)| *key == "OXIMOND_JIT_SYNC")
                .map(|(_, value)| *value)
        };
        assert_eq!(sync(Mode::TracingMir), Some("1"));
        assert_eq!(sync(Mode::TracingMirAsync), Some("0"));
    }

    #[test]
    fn the_mir_statistics_line_is_split_out_and_counted() {
        let stderr = b"trace: TraceStatistics { traces_compiled: 2, native_entries: 9 }\nmir: accepted_trees=3 accepted_ir=40 fallback_trees=1 fallback_ir=7 reasons={\"x\": 4}\noximond: boom\n";
        let (rest, statistics) = split_jit_statistics(stderr);
        assert_eq!(rest, b"oximond: boom\n");
        let jit = JitActivity::parse(statistics);
        assert_eq!(jit.mir_accepted_trees, 3);
        assert_eq!(jit.mir_fallback_trees, 1);
        assert_eq!(jit.traces_compiled, 2);
        assert_eq!(jit.native_entries, 9);
        assert!(jit.productive());
    }

    fn ok(stdout: &str) -> Observation {
        Observation {
            stdout: stdout.as_bytes().to_vec(),
            end: End::Success,
        }
    }

    #[cfg(unix)]
    #[test]
    fn relative_oracle_path_runs_from_case_directory() {
        use std::os::unix::fs::PermissionsExt;

        let scratch = std::process::Command::new("mktemp")
            .args(["-d", "./oracle-test.XXXXXXXX"])
            .output()
            .unwrap();
        assert!(scratch.status.success());
        let directory = std::path::PathBuf::from(String::from_utf8(scratch.stdout).unwrap().trim());
        let executable = directory.join("lua");
        std::fs::write(
            &executable,
            "#!/bin/sh\nif [ \"$1\" = -v ]; then echo 'Lua 5.4 test'; else cat \"$1\"; fi\n",
        )
        .unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
        let case = directory.join("case");
        std::fs::create_dir(&case).unwrap();
        std::fs::write(case.join(super::CASE_FILE), "observation\n").unwrap();
        let oracle = super::find_oracle(crate::profile::Profile::Lua54, Some(&executable)).unwrap();
        let result = super::run_oracle(&oracle, &case, crate::run::Limits::default());
        std::fs::remove_dir_all(&directory).unwrap();
        assert_eq!(result.observation, ok("observation\n"));
    }

    #[test]
    fn error_lines_lose_only_the_interpreter_prefix() {
        assert_eq!(
            normalize_error_line(b"lua5.4: case.lua:3: top"),
            "case.lua:3: top"
        );
        assert_eq!(
            normalize_error_line(b"oximond: runtime error: case.lua:3: top"),
            "case.lua:3: top"
        );
        assert_eq!(
            normalize_error_line(b"luajit: case.lua:3: \xff"),
            "case.lua:3: \\xff"
        );
    }

    #[test]
    fn differences_name_the_category_and_tag() {
        assert_eq!(difference(&ok("a#1 x\n"), &ok("a#1 x\n")), None);
        assert_eq!(
            difference(&ok("o12#1 x\n"), &ok("o12#1 y\n")).as_deref(),
            Some("stdout@o12")
        );
        let crash = Observation {
            stdout: Vec::new(),
            end: End::Crash("signal 11".into()),
        };
        assert_eq!(difference(&ok(""), &crash).as_deref(), Some("crash"));
        let error = Observation {
            stdout: b"L1#1 i1\n".to_vec(),
            end: End::Error("case.lua:9: bad".into()),
        };
        assert_eq!(
            difference(&ok("L1#1 i1\nL1#1 i2\n"), &error).as_deref(),
            Some("unexpected_error@L1")
        );
    }

    #[test]
    fn jit_statistics_are_split_and_counted() {
        let stderr = b"warn\njit: guard_exits=1\ntrace: TraceStatistics { traces_compiled: 3, native_entries: 5, iterations: 40, exits: [2, 1], side_exits: [1, 2, 0], side_installs: 1 }\n";
        let (rest, lines) = split_jit_statistics(stderr);
        assert_eq!(rest, b"warn\n");
        let activity = JitActivity::parse(lines);
        assert_eq!(activity.traces_compiled, 3);
        assert_eq!(activity.native_entries, 5);
        assert_eq!(activity.side_exits, 4);
        assert_eq!(activity.iterations, 40);
        assert_eq!(activity.exits, 3);
        assert_eq!(activity.side_installs, 1);
        assert!(activity.productive());
    }

    #[test]
    fn escapes_binary_bytes() {
        assert_eq!(escape(b"a\0\xff\\"), "a\\x00\\xff\\\\");
    }
}
