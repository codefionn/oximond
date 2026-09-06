//! What the trace engine knows about one loop header, for the attribution
//! report (see `vm::attribution`).
//!
//! Nothing here changes a decision: it reads the site table and the cache.

use super::TracePolicy;
use super::policy::{RejectCause, SiteState};
use super::stats::{ExitMatrix, TraceAbortReason, TraceExitKind};
use crate::bytecode::Chunk;
use std::sync::Arc;

/// Where a loop header stands with the tracer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoverageState {
    /// The tracer never saw the header (or not often enough to keep a site).
    NeverHot,
    /// The header is counting backedges and has not reached the threshold.
    Counting,
    /// A recording is in progress.
    Recording,
    /// A recording closed and waits for its compile.
    Compiling,
    /// A recording failed and the header waits to try again.
    Aborted,
    /// The tracer gave the header up.
    Rejected,
    /// The header's traces were retired or evicted and it is not compiled.
    Retired,
    /// A compiled trace owns the header.
    Compiled,
    /// A compiled trace owns the header but often leaves to the interpreter
    /// for a reason other than finishing the loop (see
    /// [`Coverage::exited_often`]).
    CompiledExited,
    /// A trace tree of an inner loop runs this header's iterations.
    Covered,
}

impl CoverageState {
    /// A short stable name for reports.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::NeverHot => "never-hot",
            Self::Counting => "counting",
            Self::Recording => "recording",
            Self::Compiling => "compiling",
            Self::Aborted => "aborted",
            Self::Rejected => "rejected",
            Self::Retired => "retired",
            Self::Compiled => "compiled",
            Self::CompiledExited => "compiled-exited",
            Self::Covered => "covered",
        }
    }
}

/// A header's last recording abort.
#[derive(Clone, Debug)]
pub struct CoverageAbort {
    pub reason: TraceAbortReason,
    /// The chunk and pc of the instruction the recording met it at: the
    /// chunk of the frame running then, a callee's when the recording had
    /// inlined one. `None` for an abort with no instruction (a refused
    /// compile, an error) or a chunk that is gone.
    pub at: Option<(Arc<Chunk>, usize)>,
}

/// The trace state of one header.
#[derive(Clone, Debug)]
pub struct Coverage {
    pub state: CoverageState,
    /// The site's last recording abort.
    pub abort: Option<CoverageAbort>,
    /// Why a rejected header was given up, when it is rejected.
    pub rejected_by: Option<RejectCause>,
    /// Recording attempts so far.
    pub attempts: u32,
    /// Exits taken by the compiled trace that continue in the interpreter or
    /// a side, summed over its counted exits.
    pub exits: u64,
    /// Native runs of the compiled trace.
    pub entries: u64,
    /// Loop-backs the compiled trace ran natively.
    pub iterations: u64,
    /// Exits that resumed in the interpreter over the trace's life (kept
    /// across recompiles), by exit kind and guard column.
    pub interpreter_exits: ExitMatrix,
}

impl Coverage {
    /// Interpreter exits that were neither the loop finishing nor a poll (the
    /// trace's periodic safe point).
    #[must_use]
    pub fn abnormal_exits(&self) -> u64 {
        self.interpreter_exits
            .iter()
            .enumerate()
            .filter(|&(kind, _)| {
                kind != TraceExitKind::LoopComplete.index() && kind != TraceExitKind::Poll.index()
            })
            .flat_map(|(_, row)| row.iter())
            .sum()
    }

    /// Whether the abnormal exits reach a
    /// quarter of the native entries: the trace runs, but not for long.
    #[must_use]
    pub fn exited_often(&self) -> bool {
        let abnormal = self.abnormal_exits();
        abnormal > 0 && abnormal.saturating_mul(4) >= self.entries
    }
}

impl TracePolicy {
    /// The trace state of the header at `pc` of `chunk`
    /// ([`FUNCTION_ENTRY_PC`] for a function entry).
    pub(crate) fn coverage(&self, chunk: &Arc<Chunk>, pc: usize) -> Coverage {
        let mut found = Coverage {
            state: CoverageState::NeverHot,
            abort: None,
            rejected_by: None,
            attempts: 0,
            exits: 0,
            entries: 0,
            iterations: 0,
            interpreter_exits: ExitMatrix::default(),
        };
        let Some(site) = self.sites.site(chunk, pc) else {
            return found;
        };
        found.abort = site.reported_abort().map(|abort| CoverageAbort {
            reason: abort.reason,
            at: abort
                .at
                .as_ref()
                .and_then(|(chunk, pc)| Some((chunk.upgrade()?, *pc))),
        });
        found.attempts = u32::from(site.attempts);
        let retired = !site.retired_unproductive.is_empty();
        found.state = match site.state {
            SiteState::Cold { backedges } => {
                if found.abort.is_some() {
                    CoverageState::Aborted
                } else if retired {
                    CoverageState::Retired
                } else if backedges == 0 && found.attempts == 0 {
                    CoverageState::NeverHot
                } else {
                    CoverageState::Counting
                }
            }
            SiteState::Recording => CoverageState::Recording,
            SiteState::Queued | SiteState::Compiling => CoverageState::Compiling,
            SiteState::Cooldown { .. } => CoverageState::Aborted,
            SiteState::Rejected => {
                found.rejected_by = site.reject_cause;
                if retired {
                    CoverageState::Retired
                } else {
                    CoverageState::Rejected
                }
            }
            SiteState::Covered => CoverageState::Covered,
            SiteState::Installed { index } => {
                if let Some(trace) = self.cache.get(index) {
                    let (entries, iterations, exits) = trace.run_counts();
                    found.entries = entries;
                    found.iterations = iterations;
                    found.exits = exits;
                    found.interpreter_exits = *trace.interpreter_exit_counts();
                    if found.exited_often() {
                        CoverageState::CompiledExited
                    } else {
                        CoverageState::Compiled
                    }
                } else {
                    CoverageState::Retired
                }
            }
        };
        found
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn many_exits_other_than_a_finished_loop_mark_a_compiled_header() {
        let mut coverage = Coverage {
            state: CoverageState::Compiled,
            abort: None,
            rejected_by: None,
            attempts: 0,
            exits: 0,
            entries: 100,
            iterations: 1000,
            interpreter_exits: ExitMatrix::default(),
        };
        // Finishing the loop on every entry is the normal way out.
        coverage.interpreter_exits[TraceExitKind::LoopComplete.index()][1] = 100;
        assert!(!coverage.exited_often());
        coverage.interpreter_exits[TraceExitKind::GuardFail.index()][1] = 30;
        assert_eq!(coverage.abnormal_exits(), 30);
        assert!(coverage.exited_often());
    }
}
