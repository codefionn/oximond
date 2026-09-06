//! Trace engine counters reported through [`oximond::vm::JitStatistics`].

use crate::jit_ir::trace::OptimizeStats;

/// Why a recording stopped without producing a trace.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceAbortReason {
    /// The recorder met an instruction it cannot record.
    UnsupportedOpcode,
    /// The call depth changed in a way the recorder does not follow.
    DepthChange,
    /// The guest raised an error while recording.
    Error,
    /// The guest yielded while recording.
    Yield,
    /// A debug hook was installed or fired.
    Hook,
    /// The trace grew past an instruction, snapshot, or depth limit.
    LimitExceeded,
    /// The trace did not return to its loop header.
    LoopNotClosed,
    /// The recorded trace failed verification or lowering.
    CompileRejected,
    /// A flush or `jit.off` interrupted the recording.
    Flushed,
    /// An operand has a type or value the recorder does not model, such as a
    /// string, a table, or mixed number and nil operands.
    UnsupportedOperand,
    /// The compatibility profile cannot be traced. No profile uses this now:
    /// every profile records, and the variant stays so the counter layout
    /// and the retry classification do not change.
    UnsupportedProfile,
    /// The iteration took a backward branch to a different loop header, so a
    /// hotter inner loop exists.
    InnerLoop,
    /// A register changes type across the iteration, so the loop cannot
    /// carry it in a typed parameter. Retried once the state settles.
    TypeUnstable,
    /// Execution did not follow the path the recorder predicted, or the
    /// interpreter state contradicted the recorder's model.
    Divergence,
    /// The iteration being recorded indexed a table where native code cannot
    /// repeat the access: a key outside the array part (a hash key such as
    /// `t[0]`, the slot past the end), a nil slot, or an absent field. Which
    /// iteration that is depends on where in the loop the recording started,
    /// so another attempt usually succeeds.
    TableMiss,
    /// The loop closed with fewer iterations left in its run than a trace
    /// needs per entry to repay entering and leaving it, and no loop around
    /// it could carry it in a tree. Retried after a long wait, since a long
    /// loop may have been recorded near its end. A loop that is short every
    /// time is rejected before anything is compiled.
    ShortLoop,
    /// An operation that ends the lifetime of borrowed values (a table set
    /// that may resize the array, an opaque effect) met a borrowed string a
    /// register still holds. The recorder has no way to own the string
    /// first, so it rejects the recording. Retried like an unsupported
    /// operand: another path may not hold the borrow.
    BorrowOutlivesEffect,
    /// A table constructor's table, kept virtual in SSA values, reached
    /// something the recorder cannot resolve in the trace: a store into an
    /// object, a call argument, a return, an upvalue or global store, an
    /// identity or raw operation, a metatable or traversal call, a
    /// metamethod-capable operation, a key it cannot fix at record time, or
    /// a value live across the loop-back. Native code never allocates, so the
    /// recording is rejected. Retried: another path may not let the table
    /// escape.
    VirtualEscape,
    /// A tail call in the root frame that the recorder cannot follow as a
    /// replacement of the root frame: one in a side recording or outside a
    /// loop whose header is the chunk's first instruction, of a vararg
    /// function or with forwarded or open arguments, from a frame with
    /// captured locals, into a function with more registers than the root's
    /// window, or back into the root chunk with another closure. Retried
    /// like an unsupported operand: another path may not make the call.
    RootTailCall,
    /// A function-entry recording met what it cannot follow: a call of its
    /// own function that is no plain self call (a tail call, open results or
    /// varargs, another closure of the function, a callee not read from a
    /// cell or global the function leaves alone, arguments of other types
    /// than the entry's), a return of other than one value, or anything the
    /// function trace refuses (a store, a table constructor, an upvalue
    /// store, a loop, captured locals). Retried like an unsupported operand:
    /// another activation may take another path.
    SelfCall,
    /// The loop calls `coroutine.yield` on every iteration (nothing before
    /// the call can skip it), so the thread leaves the loop in the middle of
    /// each one, which no trace follows. Rejected at once, so the
    /// interpreter runs the loop with no repeated recording.
    YieldingLoop,
}

impl TraceAbortReason {
    /// Every reason, in counter order.
    pub const ALL: [Self; 21] = [
        Self::UnsupportedOpcode,
        Self::DepthChange,
        Self::Error,
        Self::Yield,
        Self::Hook,
        Self::LimitExceeded,
        Self::LoopNotClosed,
        Self::CompileRejected,
        Self::Flushed,
        Self::UnsupportedOperand,
        Self::UnsupportedProfile,
        Self::InnerLoop,
        Self::TypeUnstable,
        Self::Divergence,
        Self::TableMiss,
        Self::ShortLoop,
        Self::BorrowOutlivesEffect,
        Self::VirtualEscape,
        Self::RootTailCall,
        Self::SelfCall,
        Self::YieldingLoop,
    ];

    /// The index of this reason in [`TraceStatistics::recording_aborts`].
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// How a native trace returned control to the VM.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceExitKind {
    /// The live frame did not match the trace entry types.
    EntryMismatch,
    /// A guard failed and a snapshot restored the interpreter state.
    GuardFail,
    /// Execution reached an instruction the trace cannot run.
    UnsupportedBefore,
    /// A guard failed after an effect committed, so execution resumes after it.
    CommittedEffect,
    /// The loop ran to completion.
    LoopComplete,
    /// The trace stopped at a VM safepoint (budget, GC, or interrupt).
    Poll,
    /// The trace was invalidated while running.
    Invalidated,
}

impl TraceExitKind {
    /// Every kind, in counter order.
    pub const ALL: [Self; 7] = [
        Self::EntryMismatch,
        Self::GuardFail,
        Self::UnsupportedBefore,
        Self::CommittedEffect,
        Self::LoopComplete,
        Self::Poll,
        Self::Invalidated,
    ];

    /// The index of this kind in [`TraceStatistics::exits`].
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// The class of guard that caused a side exit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceGuardKind {
    /// A value had a different type than recorded.
    Type,
    /// A branch went the other way.
    Branch,
    /// An array index fell outside the recorded bounds.
    ArrayBounds,
    /// A table had a different layout than recorded.
    TableShape,
    /// A metatable changed or appeared.
    Metatable,
    /// Arithmetic overflowed or left the recorded domain.
    Arithmetic,
    /// A call target differed from the recorded callee.
    Callee,
    /// Any other guard.
    Other,
}

impl TraceGuardKind {
    /// Every kind, in counter order.
    pub const ALL: [Self; 8] = [
        Self::Type,
        Self::Branch,
        Self::ArrayBounds,
        Self::TableShape,
        Self::Metatable,
        Self::Arithmetic,
        Self::Callee,
        Self::Other,
    ];

    /// The index of this kind in [`TraceStatistics::side_exits`].
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Interpreter exits by [`TraceExitKind::index`], then guard column: a
/// [`TraceGuardKind::index`], or the last column for an exit with no guard.
pub type ExitMatrix = [[u64; TraceGuardKind::ALL.len() + 1]; TraceExitKind::ALL.len()];

/// Most distinct builtins [`NativeAborts`] tells apart; the rest are dropped.
pub const MAX_ABORT_NATIVES: usize = 48;

/// How many recordings aborted at a call of each builtin, by the builtin's
/// library-qualified name (`math.ceil`, `select`, `string.gsub`). Counted
/// when a recording aborts, never on the compiled path.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct NativeAborts {
    names: [&'static str; MAX_ABORT_NATIVES],
    counts: [u64; MAX_ABORT_NATIVES],
    len: usize,
}

/// Lists the builtins that caused an abort, not the empty slots after them.
impl std::fmt::Debug for NativeAborts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl Default for NativeAborts {
    fn default() -> Self {
        Self {
            names: [""; MAX_ABORT_NATIVES],
            counts: [0; MAX_ABORT_NATIVES],
            len: 0,
        }
    }
}

impl NativeAborts {
    /// Counts one abort at a call of the builtin called `name`. Past
    /// [`MAX_ABORT_NATIVES`] distinct names a new one is dropped.
    pub fn add(&mut self, name: &str) {
        if let Some(at) = self.names[..self.len]
            .iter()
            .position(|known| *known == name)
        {
            self.counts[at] += 1;
            return;
        }
        if self.len < MAX_ABORT_NATIVES {
            self.names[self.len] = intern(name);
            self.counts[self.len] = 1;
            self.len += 1;
        }
    }

    /// The aborts at calls of the builtin called `name`.
    #[must_use]
    pub fn of(&self, name: &str) -> u64 {
        self.iter()
            .find_map(|(known, count)| (known == name).then_some(count))
            .unwrap_or(0)
    }

    /// Every builtin that caused an abort, with its count, in the order of
    /// the first abort.
    pub fn iter(&self) -> impl Iterator<Item = (&'static str, u64)> + '_ {
        self.names[..self.len]
            .iter()
            .copied()
            .zip(self.counts[..self.len].iter().copied())
    }
}

/// The one copy of `name` that lives for the program. A process has a few
/// hundred builtins, so the set stays small.
fn intern(name: &str) -> &'static str {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static NAMES: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let mut names = NAMES
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(known) = names.get(name) {
        return known;
    }
    let leaked: &'static str = Box::leak(name.to_owned().into_boxed_str());
    names.insert(leaked);
    leaked
}

/// Counters for the trace engine. Native execution counters stay zero until
/// traces are compiled.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TraceStatistics {
    /// Recordings started at a loop header.
    pub recording_starts: u64,
    /// Recordings abandoned, by [`TraceAbortReason::index`].
    pub recording_aborts: [u64; TraceAbortReason::ALL.len()],
    /// Recordings abandoned at a call of a builtin the recorder does not
    /// handle, by the builtin's name.
    pub native_aborts: NativeAborts,
    /// Recordings that closed a loop and produced a verified trace.
    pub traces_recorded: u64,
    /// Guest instructions in the traces that closed.
    pub recorded_instructions: u64,
    /// Calls of Lua closures that the traces which closed inline. A call that
    /// two traces inline counts twice.
    pub inlined_calls: u64,
    /// Array reads the recordings answered from a slot they had already
    /// read or stored, with a move and no guard (`recorder::slot_facts`),
    /// in the traces that closed.
    pub slot_reads_forwarded: u64,
    /// In-place array stores the recordings proved from a slot they had
    /// already read or stored with a value of the stored type, which took
    /// no guard and no snapshot ([`crate::jit_ir::trace::TraceOp::ArrayOverwrite`]),
    /// in the traces that closed.
    pub array_overwrites: u64,
    /// Guest instructions the recorder saw from the fast interpreter loop.
    pub fast_path_records: u64,
    /// Guest instructions the recorder saw from the general dispatch loop.
    pub general_path_records: u64,
    /// Traces accepted by the native lowering layer.
    pub traces_compiled: u64,
    /// Entries into native trace code.
    pub native_entries: u64,
    /// Loop iterations completed inside native trace code.
    pub iterations: u64,
    /// Native trace exits, by [`TraceExitKind::index`].
    pub exits: [u64; TraceExitKind::ALL.len()],
    /// Side exits taken, by [`TraceGuardKind::index`].
    pub side_exits: [u64; TraceGuardKind::ALL.len()],
    /// Exits that left native code and resumed the interpreter, by
    /// [`TraceExitKind::index`] and then [`TraceGuardKind::index`]. The last
    /// column (`TraceGuardKind::ALL.len()`) holds the exits with no guard
    /// (a poll, a loop that ran out of trips). Unlike [`Self::exits`] it says
    /// which guard the interpreter takes over from, so a hot exit that no side
    /// trace took shows up as a large cell. Exits of side members count too.
    pub interpreter_exits: [[u64; TraceGuardKind::ALL.len() + 1]; TraceExitKind::ALL.len()],
    /// Compile requests the broker accepted: new traces and tree recompiles.
    pub compile_submissions: u64,
    /// Roots of short loops in a nest compiled at the baseline tier because
    /// the tree that replaces them follows (see "Nested loops" in the
    /// [module documentation](super)).
    pub baseline_roots: u64,
    /// Compile requests the broker refused (queue or byte budget). The
    /// recorded trace is dropped and its site backs off.
    pub compile_refusals: u64,
    /// Finished compilations discarded because a flush happened after their
    /// request, or because their site or trace no longer exists.
    pub stale_completions: u64,
    /// Compiles that panicked (a codegen bug, such as a Cranelift allocator
    /// panic). The broker catches the panic, so only that trace or side is
    /// rejected, as [`TraceAbortReason::CompileRejected`], and later ones
    /// still compile.
    pub compile_panics: u64,
    /// Traces installed as an additional version at a header that already
    /// had one.
    pub version_installs: u64,
    /// Installed traces dropped to make room under the cache caps.
    pub cache_evictions: u64,
    /// Traces dropped before installing because the caps left no room.
    pub cache_rejections: u64,
    /// Rejected-site records dropped to stay under the negative cache cap.
    pub negative_evictions: u64,
    /// Side recordings started: a branch guard of a trace failed often enough.
    pub side_recordings: u64,
    /// Side recordings that closed into a side trace.
    pub side_traces_recorded: u64,
    /// Side traces among [`Self::side_traces_recorded`] that link back to
    /// the root at once, recording nothing: the side of the root latch's
    /// guard when it resumes at the root's header.
    pub empty_sides: u64,
    /// Side recordings abandoned for any reason (the counters of
    /// [`Self::recording_aborts`] only count root recordings).
    pub side_aborts: u64,
    /// Side traces that could not join their tree: they failed its checks or
    /// the tree did not compile or fit the cache.
    pub side_rejections: u64,
    /// Trees installed with a new side trace.
    pub side_installs: u64,
    /// Side traces compiled apart from their tree and attached to the exit
    /// they continue (a subset of [`Self::side_installs`]).
    pub side_compiles: u64,
    /// Sides compiled apart that did not join their tree: the broker or the
    /// backend refused them, or the tree's code could not take them. They
    /// compile with the tree.
    pub detached_refusals: u64,
    /// Instructions handed to Cranelift, summed over the compiles that
    /// installed a tree (`CLIF instructions`).
    pub clif_instructions_tree: u64,
    /// The same, summed over the sides compiled apart.
    pub clif_instructions_side: u64,
    /// Trees recompiled because sides that were compiled apart and leave at
    /// the loop header proved hot (see `DETACHED_LINK_EXITS`).
    pub link_recompiles: u64,
    /// Native exits taken from a side trace of a tree, not from the root.
    pub side_member_exits: u64,
    /// Times a failing guard jumped into its side trace without leaving
    /// native code. Counted only when statistics are requested.
    pub side_entries: u64,
    /// The calls of `side_entries` that native code made itself from the
    /// exits of a tree, with no Rust frame between the tree and the side
    /// (see `Lowerer::emit_native_dispatch`). Counted only when statistics
    /// are requested.
    pub native_side_calls: u64,
    /// Side traces installed that run on through the loops around their
    /// root's loop (an outer side), taking the backward branch of at least
    /// one of them.
    pub outer_sides: u64,
    /// Recordings that unrolled loops and ran out of room, which the tree
    /// recorded again without unrolling (not counted as aborts).
    pub unroll_restarts: u64,
    /// Loop sides installed: loops beside a tree's root, or inside a root
    /// that ends in front of them, that run as loops of their own inside the
    /// tree (see "Loop sides" and "Outer roots" in the module documentation
    /// of the trace policy).
    pub loop_sides: u64,
    /// Sides that rejoin the loop of a loop side they descend from
    /// ([`oximond::jit_ir::trace::TraceTerminator::LinkLoop`]), counted in
    /// the optimized code of every tree compile that installed.
    pub loop_links: u64,
    /// Loop headers handed to a trace tree that runs them (`Covered`).
    pub covered_loops: u64,
    /// Covered headers rejected because their tree went away.
    pub uncovered_loops: u64,
    /// Covered headers rejected because their tree kept leaving them to the
    /// interpreter (a subset of `uncovered_loops`).
    pub covered_revoked: u64,
    /// Optimizer counters summed over every trace installed, per pass.
    pub optimizer: OptimizeStats,
    /// Function-entry traces (self recursion) accepted by the native lowering
    /// layer, a subset of [`Self::traces_compiled`].
    pub compiled_function_traces: u64,
    /// Self calls native code made: activations of function traces past the
    /// one each entry started with.
    pub native_self_calls: u64,
    /// Activations of function traces an exit rebuilt as interpreter frames
    /// below the one that left (the unwind of a deep exit).
    pub unwound_activations: u64,
    /// Entries of function traces that ended in a native return.
    pub native_returns: u64,
    /// Hash lookups that native code asked the runtime to answer, instead of
    /// probing the table's string index itself (see
    /// [`oximond::jit::TraceControl::hash_calls`]).
    pub hash_helper_calls: u64,
    /// Calls of the helper that stores under a run-time string key whatever
    /// the table holds ([`oximond::jit_ir::trace::HelperId::SET_STRING`]), and
    /// of the two that keep the table's layout: an overwrite of a present key
    /// with a number or boolean of its type (`SET_PRESENT`, which native code
    /// makes itself where the table has a string index), and the deletion of a
    /// key (`DELETE_STRING`).
    pub set_string_helper_calls: u64,
    pub set_present_helper_calls: u64,
    /// Calls of `SET_REVIVE`, which fills a deleted key in place and which
    /// native code makes itself where the table has a string index.
    pub set_revive_helper_calls: u64,
    pub delete_string_helper_calls: u64,
    /// Calls of the VM's table set entry for a store under a number key that
    /// native code did not make itself: a store that grows the array part,
    /// fills the slot after the border, deletes, or meets a metatable, a hash
    /// part or a string.
    pub array_set_helper_calls: u64,
    /// Calls of the helper that makes a table a virtual one stood for when it
    /// escapes ([`oximond::jit_ir::trace::HelperId::TABLE_NEW`]): one per
    /// table the trace allocates.
    pub table_new_helper_calls: u64,
    /// Calls of the helpers that read a callee's cell through its closure
    /// ([`oximond::jit_ir::trace::HelperId::CELL_GET_INTEGER`] and its
    /// siblings), or write one
    /// ([`oximond::jit_ir::trace::HelperId::CELL_SET`]), or check a callee's
    /// prototype ([`oximond::jit_ir::trace::HelperId::CLOSURE_SAME_PROTO`]).
    /// Native code does these itself through the rows of the closures where
    /// the cell holds a number or a boolean, or holds a table that the trace
    /// only reads, so only a write of a table or a string into a cell, a read
    /// in a trace that writes tables into cells, or a ninth slot comes here.
    pub closure_helper_calls: u64,
    /// Calls of the lookup helpers that read a field, a number slot or the
    /// length of a table met by lane (`TABLE_GET_*`, `TABLE_LENGTH`).
    pub table_lookup_helper_calls: u64,
    /// Array reads of tables met by lane that native code makes inline
    /// ([`oximond::jit_ir::trace::TraceOp::LaneArrayLoad`]), counted in the
    /// optimized code of every tree compile that installed.
    pub lane_array_loads_compiled: u64,
    /// In-place array stores into tables met by lane that native code makes
    /// inline ([`oximond::jit_ir::trace::TraceOp::LaneArrayStore`]), counted
    /// like [`Self::lane_array_loads_compiled`].
    pub lane_array_stores_compiled: u64,
    /// Pure two-arm branches recorded as a value select instead of a branch
    /// guard ([`oximond::jit_ir::trace::TraceOp::Select`]), counted in the
    /// optimized code of every tree compile that installed and in the
    /// recorded code of every side compiled apart.
    pub value_selects_compiled: u64,
    /// Table bindings a tree compile that installed demoted to lane
    /// operations, because a member of the tree writes their register (see
    /// [`oximond::jit_ir::trace::plan_tree`]).
    pub demoted_bindings: u64,
    /// Exits that rebuilt inlined frames above the root frame for the
    /// interpreter: a guard that failed inside an inlined call (a subset of
    /// [`Self::exits`]).
    pub inner_exits: u64,
    /// Side traces installed that start inside the inlined calls of the
    /// guard they continue (see "Sides from exits inside inlined frames" in
    /// the trace module docs; a subset of [`Self::side_installs`]).
    pub inner_sides: u64,
}

impl TraceStatistics {
    /// Recordings abandoned for `reason`.
    #[must_use]
    pub const fn aborts(&self, reason: TraceAbortReason) -> u64 {
        self.recording_aborts[reason.index()]
    }

    /// Native trace exits of `kind`.
    #[must_use]
    pub const fn exits_of(&self, kind: TraceExitKind) -> u64 {
        self.exits[kind.index()]
    }

    /// Side exits caused by `guard`.
    #[must_use]
    pub const fn side_exits_of(&self, guard: TraceGuardKind) -> u64 {
        self.side_exits[guard.index()]
    }

    /// Exits of `kind` caused by `guard` that resumed the interpreter. `None`
    /// is the column of exits with no guard.
    #[must_use]
    pub const fn interpreter_exits_of(
        &self,
        kind: TraceExitKind,
        guard: Option<TraceGuardKind>,
    ) -> u64 {
        let column = match guard {
            Some(guard) => guard.index(),
            None => TraceGuardKind::ALL.len(),
        };
        self.interpreter_exits[kind.index()][column]
    }

    /// Total recordings abandoned.
    #[must_use]
    pub fn total_aborts(&self) -> u64 {
        self.recording_aborts.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_aborts_print_only_the_entries_they_hold() {
        let mut aborts = NativeAborts::default();
        assert_eq!(format!("{aborts:?}"), "{}");
        aborts.add("pcall");
        aborts.add("string.gsub");
        aborts.add("pcall");
        assert_eq!(format!("{aborts:?}"), r#"{"pcall": 2, "string.gsub": 1}"#);
        let statistics = TraceStatistics {
            native_aborts: aborts,
            ..TraceStatistics::default()
        };
        assert!(!format!("{statistics:?}").contains(r#""""#));
    }
}
