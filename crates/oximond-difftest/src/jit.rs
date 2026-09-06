//! The JIT property: every tracing variant prints what oximond's own
//! interpreter prints.
//!
//! The campaign machinery is the stock-oracle one with
//! [`Oracle::interpreter`](crate::case::Oracle::interpreter) as the oracle and
//! the variants of [`Mode::JIT`](crate::case::Mode::JIT) as the modes. This
//! module adds what that property needs beyond it: counts of how many cases
//! actually ran native code (a comparison of two interpreters in disguise
//! proves nothing), and the check that fails a run that did not.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::campaign::Record;
use crate::case::{CaseResult, End, JitActivity, Mode, Verdict};
use crate::profile::Profile;

/// Default share of cases whose tracing runs must have entered native code.
pub const DEFAULT_MIN_NATIVE_SHARE: f64 = 0.6;

/// What one variant did over a set of cases.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VariantStats {
    /// Cases the oracle accepted (passes and mismatches).
    pub cases: usize,
    /// Cases where the variant compiled at least one trace.
    pub compiled_cases: usize,
    /// Cases where the variant entered native code.
    pub native_cases: usize,
    /// Cases where MIR compiled at least one tree.
    pub mir_cases: usize,
    /// Cases whose oracle ended in an uncaught error, so no variant printed
    /// statistics.
    pub error_cases: usize,
    pub traces_compiled: u64,
    pub native_entries: u64,
    pub side_exits: u64,
    pub mir_accepted_trees: u64,
    pub mir_fallback_trees: u64,
}

impl VariantStats {
    fn add(&mut self, oracle_error: bool, jit: &JitActivity) {
        self.cases += 1;
        self.error_cases += usize::from(oracle_error);
        self.compiled_cases += usize::from(jit.traces_compiled > 0);
        self.native_cases += usize::from(jit.productive());
        self.mir_cases += usize::from(jit.mir_accepted_trees > 0);
        self.traces_compiled += jit.traces_compiled;
        self.native_entries += jit.native_entries;
        self.side_exits += jit.side_exits;
        self.mir_accepted_trees += jit.mir_accepted_trees;
        self.mir_fallback_trees += jit.mir_fallback_trees;
    }

    /// Share of cases that entered native code.
    #[must_use]
    pub fn native_share(&self) -> f64 {
        if self.cases == 0 {
            0.0
        } else {
            self.native_cases as f64 / self.cases as f64
        }
    }

    /// Share of cases where MIR compiled a tree.
    #[must_use]
    pub fn mir_share(&self) -> f64 {
        if self.cases == 0 {
            0.0
        } else {
            self.mir_cases as f64 / self.cases as f64
        }
    }

    fn merge(&mut self, other: &Self) {
        self.cases += other.cases;
        self.compiled_cases += other.compiled_cases;
        self.native_cases += other.native_cases;
        self.mir_cases += other.mir_cases;
        self.error_cases += other.error_cases;
        self.traces_compiled += other.traces_compiled;
        self.native_entries += other.native_entries;
        self.side_exits += other.side_exits;
        self.mir_accepted_trees += other.mir_accepted_trees;
        self.mir_fallback_trees += other.mir_fallback_trees;
    }
}

/// What a fixed part of every program (the anchor kernel) does on its own, per
/// profile and variant label. [`summarize_results_above`] subtracts it.
pub type Baseline = BTreeMap<(Profile, String), JitActivity>;

/// Per profile and variant label, over the cases the oracle accepted.
#[must_use]
pub fn summarize(records: &[Record]) -> BTreeMap<(Profile, String), VariantStats> {
    tally(records.iter().map(|record| {
        Tally {
            profile: record.profile,
            verdict: &record.verdict,
            oracle_error: record.oracle_error,
            variants: record
                .variant_jit
                .iter()
                .map(|(label, jit)| (label.as_str(), jit))
                .collect(),
        }
    }))
}

/// [`summarize`] for finished cases that are not campaign records.
#[must_use]
pub fn summarize_results<'a>(
    cases: impl IntoIterator<Item = (Profile, &'a CaseResult)>,
) -> BTreeMap<(Profile, String), VariantStats> {
    tally(cases.into_iter().map(|(profile, result)| {
        Tally {
            profile,
            verdict: &result.verdict,
            oracle_error: result
                .oracle
                .as_ref()
                .is_some_and(|run| matches!(run.observation.end, End::Error(_))),
            variants: result
                .modes
                .iter()
                .filter_map(|run| run.jit.as_ref().map(|jit| (run.label.as_str(), jit)))
                .collect(),
        }
    }))
}

/// [`summarize_results`] counting only activity beyond `baseline`: a case is
/// native when it entered native code more often than the baseline program
/// did, and MIR trees are those past the baseline's. A program whose own
/// sections never compile then counts as vacuous even if its fixed anchor did.
#[must_use]
pub fn summarize_results_above<'a>(
    cases: impl IntoIterator<Item = (Profile, &'a CaseResult)>,
    baseline: &Baseline,
) -> BTreeMap<(Profile, String), VariantStats> {
    let mut stats: BTreeMap<(Profile, String), VariantStats> = BTreeMap::new();
    for (profile, result) in cases {
        if !matches!(result.verdict, Verdict::Pass | Verdict::Mismatch) {
            continue;
        }
        let oracle_error = result
            .oracle
            .as_ref()
            .is_some_and(|run| matches!(run.observation.end, End::Error(_)));
        for run in &result.modes {
            let Some(jit) = run.jit.as_ref() else {
                continue;
            };
            let mut excess = jit.clone();
            if let Some(base) = baseline.get(&(profile, run.label.clone())) {
                excess.traces_compiled = jit.traces_compiled.saturating_sub(base.traces_compiled);
                excess.native_entries = jit.native_entries.saturating_sub(base.native_entries);
                excess.side_exits = jit.side_exits.saturating_sub(base.side_exits);
                excess.mir_accepted_trees = jit
                    .mir_accepted_trees
                    .saturating_sub(base.mir_accepted_trees);
                excess.mir_fallback_trees = jit
                    .mir_fallback_trees
                    .saturating_sub(base.mir_fallback_trees);
            }
            stats
                .entry((profile, run.label.clone()))
                .or_default()
                .add(oracle_error, &excess);
        }
    }
    stats
}

struct Tally<'a> {
    profile: Profile,
    verdict: &'a Verdict,
    oracle_error: bool,
    variants: Vec<(&'a str, &'a JitActivity)>,
}

fn tally<'a>(cases: impl Iterator<Item = Tally<'a>>) -> BTreeMap<(Profile, String), VariantStats> {
    let mut stats: BTreeMap<(Profile, String), VariantStats> = BTreeMap::new();
    for case in cases.filter(|case| matches!(case.verdict, Verdict::Pass | Verdict::Mismatch)) {
        for (label, jit) in case.variants {
            stats
                .entry((case.profile, label.to_owned()))
                .or_default()
                .add(case.oracle_error, jit);
        }
    }
    stats
}

/// The same, over all profiles, by variant label.
#[must_use]
pub fn by_variant(
    stats: &BTreeMap<(Profile, String), VariantStats>,
) -> BTreeMap<String, VariantStats> {
    let mut merged: BTreeMap<String, VariantStats> = BTreeMap::new();
    for ((_, label), row) in stats {
        merged.entry(label.clone()).or_default().merge(row);
    }
    merged
}

/// Fails if a variant's native-entry share is below `min_share`, or a MIR
/// variant compiled nothing through MIR. A comparison in which no variant
/// ran native code passes any JIT bug, so this is part of the property.
///
/// `tracing-mir-only` has no share floor: the trees MIR declines stay in the
/// interpreter there, so its share measures MIR's coverage, not the
/// programs. The other variants run the same programs and keep the floor.
///
/// # Errors
///
/// Returns every reason a variant was vacuous.
pub fn check_non_vacuity(
    stats: &BTreeMap<(Profile, String), VariantStats>,
    min_share: f64,
) -> Result<(), String> {
    let mut problems = Vec::new();
    for (label, row) in by_variant(stats) {
        if row.native_share() < min_share && label != Mode::TracingMirOnly.name() {
            problems.push(format!(
                "{label}: {} of {} cases entered native code ({:.0}%), below {:.0}%",
                row.native_cases,
                row.cases,
                row.native_share() * 100.0,
                min_share * 100.0
            ));
        }
        if Mode::parse(&label).is_some_and(Mode::uses_mir) && row.mir_accepted_trees == 0 {
            problems.push(format!("{label}: no tree compiled through MIR"));
        }
    }
    if stats.is_empty() {
        problems.push("no tracing variant reported statistics".to_owned());
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("; "))
    }
}

/// Markdown tables for a campaign report: per variant and profile, how many
/// cases entered native code and how MIR fared.
#[must_use]
pub fn render_variant_tables(records: &[Record]) -> String {
    let stats = summarize(records);
    let mut report = String::new();
    let _ = writeln!(report, "\n## JIT variants against the interpreter\n");
    let _ = writeln!(
        report,
        "Over cases the interpreter accepted. `native` is the share of cases whose variant entered native code, `errored` counts cases the interpreter ended with an uncaught error (no statistics are printed then). `MIR trees` is compiled through MIR / handed back to Cranelift (under `tracing-mir-only`, declined and left to the interpreter).\n"
    );
    let _ = writeln!(
        report,
        "| profile | variant | cases | compiled traces | native | native share | errored | native entries | guard failures | MIR cases | MIR trees |"
    );
    let _ = writeln!(report, "|---|---|---|---|---|---|---|---|---|---|---|");
    let row = |report: &mut String, profile: &str, label: &str, row: &VariantStats| {
        let _ = writeln!(
            report,
            "| {profile} | {label} | {} | {} | {} | {:.0}% | {} | {} | {} | {} | {} / {} |",
            row.cases,
            row.compiled_cases,
            row.native_cases,
            row.native_share() * 100.0,
            row.error_cases,
            row.native_entries,
            row.side_exits,
            row.mir_cases,
            row.mir_accepted_trees,
            row.mir_fallback_trees,
        );
    };
    for ((profile, label), stats) in &stats {
        row(&mut report, profile.name(), label, stats);
    }
    for (label, stats) in by_variant(&stats) {
        row(&mut report, "all", &label, &stats);
    }
    report
}
