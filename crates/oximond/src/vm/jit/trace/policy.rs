//! Per-site trace state: hotness, retry and permanent rejection.
//!
//! Every loop header the interpreter reaches by a backward step is a site,
//! keyed by chunk address and header pc. A site moves through one state
//! machine:
//!
//! ```text
//! Cold --(hot)--> Recording --(closed)--> Compiling --(installed)--> Installed
//!   ^                  |                      |                         |
//!   |                  |                      +--(refused)--> Cooldown  +--(unproductive)--> Cooldown
//!   |                  +--(transient abort)--> Cooldown --(expired)--> Recording
//!   |                  +--(structural abort, or too many retries)--> Rejected
//! ```
//!
//! `Covered` belongs to outer loops (see [`super`], "Nested loops"): an outer
//! header becomes `Covered` when a trace tree runs it.
//!
//! `Queued` is the resting state of a closed trace while native compilation
//! is off (tests that inspect recorded IR). `Compiling` waits for the broker.
//! An `Installed` site runs native code at its header until the trace proves
//! unproductive.
//!
//! A site keeps up to [`super::cache::MAX_TRACE_VERSIONS`] compiled versions
//! of its trace, one per entry type signature. `Recording` and `Compiling`
//! can overlap existing versions: the site then keeps its version chain and
//! returns to `Installed` when the extra version settles.
//!
//! `Rejected` sites are the negative cache. Only
//! [`super::cache::MAX_NEGATIVE_SITES`] of them are kept: the oldest is
//! forgotten first and its header starts counting again.
//!
//! A profile or compile rejection is final. An unsupported opcode or inner
//! loop may sit on a rarely taken path, so it retries like a type, branch,
//! limit, hook, or divergence abort: a bounded
//! number of retries, each after a longer cooldown. Flushing a policy drops
//! every site and bumps its epoch, so work that started before the flush
//! cannot install afterwards.

use super::TraceAbortReason;
use super::cache::MAX_NEGATIVE_SITES;
use crate::bytecode::Chunk;
use crate::jit_ir::trace::TraceIr;
use rustc_hash::FxHashMap;
use std::sync::{Arc, Weak};

/// The site pc of a function's entry: a function-entry trace (self
/// recursion) is keyed by its chunk and this pc, which no instruction has, so
/// it never shares a site with the loop header at pc 0 (a loop through tail
/// calls, or a loop at the top of the function).
pub(crate) const FUNCTION_ENTRY_PC: usize = usize::MAX;
/// Backedges a header takes before recording starts. A prototype value.
pub(crate) const TRACE_HOT_THRESHOLD: u32 = 64;
/// Interpreted backedges of a `Covered` header that make up one check of its
/// tree. Each is an outer iteration the tree did not run natively.
pub(crate) const COVERED_WINDOW: u32 = 64;
/// Native iterations the covering tree must run per interpreted outer
/// iteration, over a window, for the header to stay `Covered`. Below it the
/// outer side is not carrying the loop (its arm is not taken, or its guard
/// fails every time and has no side), so every outer iteration enters and
/// leaves the tree and runs the outer part interpreted, and the header is
/// rejected.
///
/// The bar is a cost comparison. An entry and exit plus the interpreted outer
/// part cost about 3400 instructions. It was set against the float loops of
/// the former region JIT, which saved about 20 instructions per iteration
/// less than a trace, so a float tree needs about 160 iterations to break
/// even (measured on a float nest: the tree started to win at 200 inner
/// iterations). An integer trace saves much more per iteration, so a lower
/// bar will do.
pub(crate) const COVERED_MIN_ITERATIONS: u64 = 64;
/// [`COVERED_MIN_ITERATIONS`] for trees whose loops carry floats.
pub(crate) const COVERED_MIN_ITERATIONS_FLOAT: u64 = 192;
/// Most sites the policy tracks. Further headers interpret.
pub(crate) const MAX_TRACE_SITES: usize = 16_384;
/// Times a site's recording may meet an inner loop inside an inlined call and
/// start again at the very next iteration, before that counts as an abort. A
/// loop that calls a function with a loop of its own only now and then (a
/// branch on `i % 32`) records that call whenever the recording starts on such
/// an iteration, and the iteration after it takes the path almost all of them
/// do.
pub(crate) const MAX_RECORDING_RESTARTS: u8 = 2;
/// Recordings of one site that may be dropped because they repeat a trace
/// that was retired for never paying, to record again at the very next
/// iteration. A loop that alternates between paths records the path of the
/// iteration it starts on, and the next iteration may take the other one.
pub(crate) const MAX_REPEAT_RESTARTS: u8 = 3;
/// Guest instructions within which a root recording that leaves its loop
/// recorded no trip: it started on the pass whose loop test ends the run
/// (`while a ~= x or b ~= y do`, two tests and a jump).
pub(crate) const EMPTY_PASS_INSTRUCTIONS: u32 = 6;
/// Recordings in a row that may start on such a pass and start again at the
/// next backedge without spending an attempt. The backedge after the one
/// that leaves is the first of the loop's next run, which stays unless that
/// run has one trip, so a loop of two or three trips closes at the first
/// restart. A loop of one trip per run uses these up and aborts as before.
pub(crate) const MAX_EMPTY_PASS_RESTARTS: u8 = 3;
/// Transient aborts one site may suffer before it is rejected for good.
pub(crate) const MAX_RECORDING_ATTEMPTS: u8 = 4;
/// [`MAX_RECORDING_ATTEMPTS`] for aborts that say the tables are not settled
/// yet (see [`TraceAbortReason::TableMiss`]). A loop that fills an array before it reads or overwrites it misses
/// on every iteration of the fill, and its waits quadruple, so the later
/// attempts reach the steady state a trace can run.
pub(crate) const MAX_TABLE_MISS_ATTEMPTS: u8 = 7;
/// Aborts in a row with the same reason at the same guest instruction after
/// which a site is rejected (or, with versions, takes no more), whatever its
/// retry budget. A recording that fails the same way three times starts on
/// iterations that all meet the failure, and a later recording that happens
/// to close would compile a trace that leaves at that instruction.
pub(crate) const MAX_SAME_ABORTS: u8 = 3;
/// Guest instructions a trace must take over from the interpreter to repay its
/// compile. A compile of a small loop (`table/iteration`'s fill, 13 body ops)
/// costs about nine million host instructions and 0.6 ms on the compile
/// thread. A trace saves some 35 of the interpreter's 40 host instructions per
/// guest instruction, so it breaks even at about 250 000 guest instructions,
/// which also takes the interpreter longer than the compile. A loop whose
/// remaining work is known to be less is not compiled (see
/// [`super::TraceAbortReason::ShortLoop`]).
pub(crate) const COMPILE_REPAYING_GUEST_WORK: f64 = 262_144.0;
/// Unproductive traces a site remembers. A loop whose recordings differ by
/// the data they met (a `string.gmatch` over patterns of several shapes) cycles
/// through a few traces, and each must be refused its second compile.
pub(crate) const RETIRED_UNPRODUCTIVE_KEPT: usize = 3;
/// Consults a site waits after its first transient abort. Each further abort
/// quadruples the wait.
pub(crate) const RETRY_COOLDOWN_BASE: u32 = 32;
/// Backedges a loop whose trace is helper-bound (see
/// `native::is_helper_bound`) runs in the interpreter before it records
/// again and compiles. Such a trace makes the same calls the interpreter
/// makes and saves only the dispatch around them, a few hundred instructions
/// per iteration, while its second recording and its compile cost about half
/// a million with the MIR backend (0.17M and 0.86M for the two loops of
/// `string/pattern_hot`): a loop that has not run this long is not expected
/// to repay them.
///
/// The wait is no longer than [`LONG_COOLDOWN`], so the loop counts it in
/// its own backedges. A longer one does not end in a nest: the loop around
/// gets hot first, cannot record past the loop that waits and is rejected,
/// and the skip windows of a rejected loop swallow the backedges of the
/// loops inside it. The wait was 20 000 when Cranelift compiled these traces
/// for millions, and `string/pattern_hot` then ran in the interpreter from
/// start to end.
pub(crate) const HELPER_BOUND_WAIT: u32 = LONG_COOLDOWN;
/// A cooldown longer than this many consults lets the interpreter skip the
/// lookups of a loop with no loop inside it, like a rejected one.
pub(crate) const LONG_COOLDOWN: u32 = 256;
/// Backedges the interpreter may skip while a header waits for its first
/// compile. The compile takes milliseconds, so its end is polled often enough
/// that a loop body of thousands of instructions loses little, and rarely
/// enough that a body of a few instructions does not leave the fast loop on
/// every iteration. Only while no trace is installed: the skip is global, and
/// would eat the entries of a trace whose header the skipped backedges reach.
pub(crate) const COMPILE_WAIT_SKIP: u32 = 32;
/// Traces the stub compile queue keeps. The oldest falls out first.
pub(crate) const TRACE_QUEUE_CAPACITY: usize = 16;
/// Times a branch guard of a trace must fail before the policy records a side
/// trace from it. Low enough that an alternating branch gets its side within
/// a few dozen iterations, high enough that a guard failing once in a while
/// does not.
pub(crate) const SIDE_EXIT_THRESHOLD: u32 = 16;
/// Trips per run a short loop may have for the loop around it to take it
/// into one trace instead of the short loop getting a trace of its own (see
/// `TracePolicy::defers_to_outer`). Unrolled, the trips cost their body each;
/// as a tree, they cost the loop's entry checks once per run and a loop-back
/// per trip. Measured on `for outer = 1, N do for i = 1, k do sum = sum +
/// cells[i] * outer end end` (instructions:u): up to eight trips the unrolled
/// nest is cheaper than the tree even in what it costs to get started (5.5M
/// against 6.5M at k = 8), and per outer iteration (30 against 205). At
/// fifteen the unrolled nest has 70 against 331 per iteration but costs 10.2M
/// to get started, which takes 14,000 iterations to repay.
pub(crate) const PREFER_UNROLL_TRIPS: f64 = 8.0;
/// Guest instructions one iteration of the loop around a short loop may have
/// with the loops inside unrolled, and backward branches it may follow, for
/// the nest to be recorded from that loop. The recording itself stops at
/// [`MAX_UNROLLED_GUEST_INSTRUCTIONS`](super::recorder::MAX_UNROLLED_GUEST_INSTRUCTIONS).
pub(crate) const PREFER_UNROLL_GUEST_INSTRUCTIONS: f64 = 120.0;
/// Backward branches of the unrolled loops that nest may have.
pub(crate) const PREFER_UNROLL_BACKEDGES: f64 = 24.0;
/// Lookups a rejected loop spends on a loop around it that is still counting,
/// before its own skip windows open anyway. The loop around gets hot after
/// 64 of its backedges, so several thousand of the inner loop's are enough
/// for one that is going to (an inner loop of 128 trips leaves it 64); an
/// outer loop with fewer iterations than that never gets hot, and the inner
/// loop must not pay a lookup for each of its backedges for all the
/// program's life. The windows that open afterwards end on the inner loop's
/// backedges nearly every time, so the loop around would hardly count again:
/// a rejected inner loop needs it hot, since the root of the loop around is
/// the one that can end in front of it (see "Outer roots").
pub(crate) const UNSKIPPED_LOOKUPS: u32 = 8_192;
/// Times a short loop's site gives way to the loop around it.
pub(crate) const MAX_DEFERRALS: u8 = 2;
/// Backedges a short loop that gave way waits before it may record again, if
/// the loop around it has not taken over by then.
pub(crate) const DEFERRAL_COOLDOWN: u32 = 1_024;
/// Side traces one tree may hold.
pub(crate) const MAX_SIDE_TRACES: usize = 64;
/// Deepest chain of sides: a side of a side of a side of the root. A call site
/// that takes turns between sixteen prototypes gets a side for each of the
/// fifteen that follow the root's, each hanging from the guard of the one
/// before. Measured on `fns[i % n + 1](i)` over `n` closures of distinct
/// prototypes (30,000 iterations): with a cap of eight, ten targets left the
/// ninth and tenth in the interpreter and twelve retired the tree for
/// unproductive entries (376 native iterations, four recompilations of 64
/// sides); with sixteen, every target up to seventeen runs natively.
pub(crate) const MAX_SIDE_DEPTH: u32 = 16;
/// Inlined frames above the root frame that a side may start inside (see
/// "Sides from exits inside inlined frames" in the trace module docs). An
/// exit that rebuilds more calls keeps its interpreter exit. Mineclonia's
/// density loop fails its guards three calls deep.
pub(crate) const MAX_SEED_FRAMES: usize = 4;
/// Guest instructions the sides of one tree that start inside inlined frames
/// may hold in all. Each such side runs the rest of the iteration from deep
/// inside it, and a loop body of many inlined calls has a guard in each, so
/// without a budget a tree could grow dozens of long sides whose compiles
/// cost more than the iterations they take from the interpreter. Past the
/// budget an exit inside inlined frames keeps its interpreter exit.
pub(crate) const MAX_SEEDED_GUEST_INSTRUCTIONS: u32 = 12_000;
/// Recordings (and rejected candidates) one guard gets before it stays an
/// exit for good.
pub(crate) const MAX_SIDE_ATTEMPTS: u8 = 3;
/// Productivity windows a trace may fail while a side trace for one of its
/// hot guards is wanted, recording or compiling. Past that it retires as a
/// trace that never became productive would.
///
/// A window is 32 entries and an entry of a trace that fails its branch at
/// once takes tens of nanoseconds, so this is a few milliseconds: about as
/// long as the worker needs to compile the tree. It counts entries, not time,
/// so a loaded machine retires the trace instead of waiting longer.
pub(crate) const SIDE_GRACE_WINDOWS: u16 = 4_096;
/// Entries a stepping stone of a short `while` or generic-for loop (see
/// `InstalledTrace::keep_stone`) may run without a side of its tree
/// landing or being on its way before it is judged like any root again.
/// Every entry of such a root leaves at a guard that ends its loop, but the
/// guards of the levels further out fail once per iteration of their own
/// loop, and each needs [`SIDE_EXIT_THRESHOLD`] failures to record. Each
/// level that lands starts the count again.
///
/// At 64, `n1`'s three `while` loops (some seven trips each) gave up in
/// about half of the runs, 247 entries in, between two landings that the
/// background compile spread apart. An entry of a stone that leaves at once
/// costs a few thousand instructions, so the wait costs at most a few
/// million.
pub(crate) const STONE_PATIENCE_ENTRIES: u64 = 512;
/// Stepping stones of one loop that may give up on their outer sides before
/// the loop's roots are judged like any root (see `Site::stone_gave_up`).
pub(crate) const STONE_GIVE_UPS: u8 = 2;
/// Iterations a guard that could take a side may go without failing before
/// the trace stops waiting for its side (no forgiveness).
pub(crate) const SIDE_QUIET_ITERATIONS: u64 = 1_024;

/// A loop header: the chunk's address and the pc of the header instruction.
pub(crate) type SiteKey = (usize, usize);

/// The address a chunk is keyed by.
pub(crate) fn chunk_address(chunk: &Arc<Chunk>) -> usize {
    Arc::as_ptr(chunk) as usize
}

/// The latch of a loop in a call that a site's recordings end in front of
/// (see "Loops in calls" in the module docs). The chunk is held weakly: the
/// weak handle keeps its allocation, so no chunk loaded later can take its
/// address and pass for it, and the chunk itself can still be freed.
#[derive(Clone, Debug)]
pub(crate) struct CalleeLoop {
    chunk: Weak<Chunk>,
    latch: usize,
}

impl CalleeLoop {
    pub(crate) fn new(chunk: &Arc<Chunk>, latch: usize) -> Self {
        Self {
            chunk: Arc::downgrade(chunk),
            latch,
        }
    }

    /// Whether `latch` of `chunk` is this loop's latch.
    pub(crate) fn is(&self, chunk: &Arc<Chunk>, latch: usize) -> bool {
        self.latch == latch && std::ptr::eq(self.chunk.as_ptr(), Arc::as_ptr(chunk))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SiteState {
    /// Counting backedges.
    Cold { backedges: u32 },
    /// A recording is in progress for this site.
    Recording,
    /// A trace closed and waits in the queue because native compilation is
    /// off.
    Queued,
    /// A closed trace waits for the broker to compile it.
    Compiling,
    /// A compiled trace owns the header. `index` names the newest version in
    /// the policy's trace table.
    Installed { index: u32 },
    /// Waiting to retry after a transient abort.
    Cooldown { remaining: u32 },
    /// The tracer gave the header up.
    Rejected,
    /// A trace tree runs this loop: the root trace of a loop inside it
    /// has a side trace that takes the branch back to this header, so a loop
    /// iteration never leaves native code until the tree exits. If the tree
    /// goes away (retired, evicted), or keeps handing outer iterations to the
    /// interpreter ([`COVERED_WINDOW`]), the header is rejected.
    Covered,
}

/// Where a call stands along the path a recording takes from its loop: the
/// call pcs of the frames between the loop's frame and the call's (outermost
/// first), then the pc of the call. A root that retires because its entries
/// left at the identity guard of a callee (another closure every iteration,
/// or every run of the loop) names the call this way in its loop's
/// [`Site::poly_calls`].
pub(crate) type PolyCallPath = Box<[usize]>;

/// How many call sites one loop remembers in [`Site::poly_calls`].
pub(crate) const MAX_POLY_CALLS: usize = 8;

/// How often a root has to have left at a callee guard of one call site
/// before its retirement names the site in [`Site::poly_calls`]: the
/// structure sets of Mineclonia's `get_structure_starts`
/// (structures.lua:744) left 33 times in 33 entries, the forks of its
/// xoroshiro self-test (random.lua:1159) 17 times in 17.
pub(crate) const POLY_CALL_EXITS: u32 = 4;

/// How many sides of a tree may continue the callee identity guards of one
/// call site before the next side there names the site in
/// [`Site::poly_calls`] instead and the loop records again: one closure
/// beside the one the root pinned keeps its side (a call that changes its
/// callee once, or takes turns between two), and a third closure shows the
/// site takes turns between many. A tree of identity-guarded sides grows a
/// side per closure (structures.lua loop@744 on the Mineclonia emerge: five
/// to seven sides of one `chunk_test` call, 1000 to 3400 guest instructions
/// each, up to 16.3 MB of metadata before another install evicted it), and
/// such a root never retires on its callee exits, since the sides continue
/// them.
pub(crate) const POLY_CALL_SIDES: usize = 1;

/// How often a root has to have left at branch guards hoisted in front of
/// its loop before its retirement sets [`Site::keep_guards`].
pub(crate) const HOISTED_GUARD_EXITS: u32 = 8;

#[derive(Debug)]
// The site's bookkeeping flags are a deliberate flat shape on this planner
// record.
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct Site {
    chunk: Weak<Chunk>,
    pub(crate) state: SiteState,
    /// Aborts so far, for the retry budget.
    pub(crate) attempts: u8,
    /// Recordings that were dropped and started again at the next iteration
    /// (see [`MAX_RECORDING_RESTARTS`]).
    pub(crate) restarts: u8,
    /// Recordings in a row that left the loop before they recorded a trip
    /// and started again (see [`MAX_EMPTY_PASS_RESTARTS`]).
    pub(crate) empty_passes: u8,
    /// Recordings dropped because they repeated a trace that never paid (see
    /// [`MAX_REPEAT_RESTARTS`]).
    pub(crate) repeat_restarts: u8,
    /// The newest compiled version at this header, if any. Each version
    /// links to the next older one through the trace table.
    pub(crate) head: Option<u32>,
    /// Record another version at the next consult.
    pub(crate) rerecord: bool,
    /// No further versions: a recording for one failed for good, or a
    /// version retired beside others. It holds while versions remain; the
    /// retirement of the last one opens the header again (see
    /// `TracePolicy::retire`).
    pub(crate) versions_closed: bool,
    /// For a `Covered` site: the cache index of the tree that covers it.
    pub(crate) cover_index: u32,
    /// Interpreted backedges since the last check of the covering tree.
    pub(crate) cover_consults: u32,
    /// The covering tree's native iteration count at the last check.
    pub(crate) cover_mark: u64,
    /// The header was covered once and given up for not being carried by its
    /// tree. It is not covered again.
    pub(crate) cover_revoked: bool,
    /// The header was rejected when a tree covered it, so giving up the
    /// coverage rejects it again instead of letting it trace on its own.
    pub(crate) covered_rejected: bool,
    /// Whether no other loop's backward branch lies inside this one, once
    /// known.
    pub(crate) leaf: Option<bool>,
    /// A recording from this header met a loop it could not follow (a
    /// restart or an abort for `InnerLoop`). The next recording unrolls the
    /// short numeric for loops it meets instead: the first one does not, so
    /// that a loop that meets an inner loop only on a rare path records the
    /// common one.
    pub(crate) inner_loop_met: bool,
    /// The latch of a loop in a call that a recording which unrolls loops
    /// could not follow: the next recording ends in front of it, and the
    /// loop runs as a loop side of the tree in the call's frames (see "Loops
    /// in calls" in the module docs).
    pub(crate) callee_loop: Option<CalleeLoop>,
    /// A recording that unrolled loops ran out of room (a restart without
    /// unrolling, not an abort). Later ones do not
    /// unroll: the nest is too big for one trace.
    pub(crate) unroll_overflowed: bool,
    /// A recording of this loop was refused because a store that moves hash
    /// storage sat beside a field access that addresses storage (a field
    /// binding, or the row of a table met by lane). The next recording reads
    /// and writes fields through the VM's lookups instead, which look the
    /// field up on each access.
    pub(crate) rowless: bool,
    /// A recording of this loop could not hold a call of a host function
    /// itself (the trace would hold something the function could change, or
    /// the function gave results the trace takes no word for). The next
    /// recording aborts at the call.
    pub(crate) no_vmcall: bool,
    /// A recording of this loop held a host call beside field bindings, so
    /// the next one reads the fields of bound tables through their lanes.
    pub(crate) lane_fields: bool,
    /// Call sites where a retired root of this loop left at the identity
    /// guard of its callee: each the pcs of the calls that lead from the
    /// loop's frame to the site's frame, then the call's own pc (see
    /// [`PolyCallPath`]). A later recording guards a call at such a site on
    /// the callee's prototype, so that one trace runs every closure of it.
    pub(crate) poly_calls: Vec<PolyCallPath>,
    /// A recording of this loop aborted inside a call it guarded on the
    /// prototype: the loop keeps identity guards from then on, and
    /// [`Self::note_poly_calls`] names no call again.
    pub(crate) poly_refused: bool,
    /// A retired root of this loop kept leaving at branch guards the
    /// optimizer hoisted in front of the loop, where no side can start: the
    /// next recording keeps its branch guards in the loop body
    /// ([`TraceIr::keep_branch_guards`](crate::jit_ir::trace::TraceIr)), and
    /// the other way of each gets a side. `get_structure_starts`
    /// (structures.lua:744) tests the signs of `cx` and `cz`, arguments that
    /// are one value in each run of the loop and another in the next: 796 of
    /// its 1757 exits on the Mineclonia emerge left there.
    pub(crate) keep_guards: bool,
    /// The registers of the root frame that entries of the loop found nil
    /// on some entries and a number on others (see
    /// `InstalledTrace::entry_flips`). A recording takes them as inputs of
    /// either type.
    pub(crate) lazy: Vec<u16>,
    /// The registers of [`Self::lazy`] whose newest version typed them
    /// anyway and was replaced for it once (see `TracePolicy::note_lazy`).
    /// A recording that started before the list grew installs a version
    /// typed for a listed register; one replacement per register bounds
    /// the churn when a recording cannot take the register untyped.
    pub(crate) lazy_retaken: Vec<u16>,
    /// A stepping stone that gave up over entries of types it had typed
    /// and the site lists in [`Site::lazy`] was not counted as a give-up
    /// (once: see the retirement in `native.rs`).
    pub(crate) stone_forgiven: bool,
    /// For the root of a short loop that a longer loop around it may take
    /// into one trace (see `TracePolicy::defers_to_outer`): the trips the
    /// last closed root of this site had per run, which the next one must
    /// have again for the loop to count as stable.
    pub(crate) trips_seen: Option<f64>,
    /// Times this site gave way to the loop around it.
    pub(crate) deferrals: u8,
    /// Backedges of this rejected loop that went to a lookup because a loop
    /// around it was still counting (see `enclosing_loop_unsettled`).
    pub(crate) unskipped: u32,
    /// This site does not give way to the loop around it: it did before and
    /// the nest did not record, or its trip count changes.
    pub(crate) no_deferral: bool,
    /// For the loop around a short loop that gave way: the short loop's
    /// header, which records again if this loop's recording fails.
    pub(crate) deferred_from: Option<usize>,
    /// The newest traces of this site that were retired without one
    /// productive window, oldest first (see [`RETIRED_UNPRODUCTIVE_KEPT`]).
    /// Recording one of them again would compile the same code to the same
    /// result.
    pub(crate) retired_unproductive: Vec<Arc<TraceIr>>,
    /// The root of this loop was retired twice for running too few trips per
    /// entry: the loop is short. A recording of a loop around it unrolls it
    /// (see `Recorder::may_unroll_while`) instead of ending at it.
    pub(crate) retired_short: bool,
    /// Times the root of this loop was retired for averaging fewer trips per
    /// entry than it needs.
    pub(crate) short_retirements: u8,
    /// Times a root of this loop was kept as a stepping stone (see
    /// `InstalledTrace::keep_stone`) and gave up on its outer sides, counted
    /// as [`STONE_GIVE_UPS`] at once when most of its entries found types no
    /// version takes. See [`Site::stone_gave_up`].
    pub(crate) stone_give_ups: u8,
    /// The site was rejected for meeting an inner loop and given another
    /// chance once that loop had no root that pays (once only).
    pub(crate) reopened: bool,
    /// A helper-bound root closed here once and was left to the interpreter
    /// for [`HELPER_BOUND_WAIT`] backedges (see `TracePolicy::finish_closed`).
    pub(crate) helper_bound_waited: bool,
    /// When the site was rejected, for the negative cache's eviction order.
    rejected_at: u64,
    /// The reason and guest instruction of the last recording abort, and how
    /// many aborts in a row had both (see [`MAX_SAME_ABORTS`]).
    last_abort: Option<(TraceAbortReason, usize)>,
    pub(crate) same_aborts: u8,
    /// The last abort for the attribution report: its reason also when it
    /// had no instruction (a refused compile), and the chunk of the frame it
    /// met it in, which is a callee's when the recording had inlined one.
    reported_abort: Option<ReportedAbort>,
    /// Why the site was last rejected (see [`RejectCause`]).
    pub(crate) reject_cause: Option<RejectCause>,
}

/// Why a site was rejected. A header given up by a path that does not
/// abort a recording (a trace retired too often, a tree that stopped
/// carrying it) has no abort to show, so the report names the path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RejectCause {
    /// The abort policy: too many aborts, the same one in a row, or one
    /// that is final (see the site's last abort).
    Aborts,
    /// Its traces were retired [`MAX_RECORDING_ATTEMPTS`] times.
    Retired,
    /// A recording repeated a trace that was retired for never paying.
    RepeatedUnpaid,
    /// It was covered by a tree that was removed, or that did not carry
    /// it while it was rejected before.
    Uncovered,
    /// Blacklisted directly (tests).
    #[cfg_attr(not(test), allow(dead_code))]
    Blacklisted,
}

/// A recording abort as the attribution reports it.
#[derive(Clone, Debug)]
pub(crate) struct ReportedAbort {
    pub(crate) reason: TraceAbortReason,
    /// The chunk and pc of the instruction the recording met it at.
    pub(crate) at: Option<(Weak<Chunk>, usize)>,
}

impl Site {
    /// Names `paths` in [`Self::poly_calls`] for the loop at `header_pc`,
    /// up to [`MAX_POLY_CALLS`]: the next recording guards those calls on
    /// the callee's prototype.
    pub(crate) fn note_poly_calls(&mut self, paths: Vec<PolyCallPath>, header_pc: usize) {
        if self.poly_refused {
            return;
        }
        for path in paths {
            if !self.poly_calls.contains(&path) && self.poly_calls.len() < MAX_POLY_CALLS {
                if super::debug_env::trace() {
                    eprintln!(
                        "jit-trace: the call at {path:?} from the loop at pc {header_pc} takes turns between callees; the next recording guards it on the prototype"
                    );
                }
                self.poly_calls.push(path);
            }
        }
    }

    fn new(chunk: &Arc<Chunk>) -> Self {
        Self {
            chunk: Arc::downgrade(chunk),
            state: SiteState::Cold { backedges: 0 },
            attempts: 0,
            restarts: 0,
            empty_passes: 0,
            repeat_restarts: 0,
            head: None,
            rerecord: false,
            versions_closed: false,
            cover_index: 0,
            cover_consults: 0,
            cover_mark: 0,
            cover_revoked: false,
            covered_rejected: false,
            leaf: None,
            inner_loop_met: false,
            callee_loop: None,
            unroll_overflowed: false,
            rowless: false,
            no_vmcall: false,
            lane_fields: false,
            poly_calls: Vec::new(),
            poly_refused: false,
            keep_guards: false,
            lazy: Vec::new(),
            lazy_retaken: Vec::new(),
            stone_forgiven: false,
            trips_seen: None,
            deferrals: 0,
            unskipped: 0,
            no_deferral: false,
            deferred_from: None,
            retired_unproductive: Vec::new(),
            retired_short: false,
            short_retirements: 0,
            stone_give_ups: 0,
            reopened: false,
            helper_bound_waited: false,
            rejected_at: 0,
            last_abort: None,
            same_aborts: 0,
            reported_abort: None,
            reject_cause: None,
        }
    }

    /// Whether the last recording abort of this site was for meeting an
    /// inner loop.
    /// Whether the stepping stones of this loop gave up for good: a root
    /// recorded here again is judged like any root, so a nest whose outer
    /// part cannot join the tree is not made a stone, kept and recompiled at
    /// every retirement. One give-up is not enough: the first stone of a
    /// loop may give up while the loop around it still runs its first,
    /// shorter trips (the edge of a mandelbrot), and the next one grows its
    /// tree.
    pub(crate) const fn stone_gave_up(&self) -> bool {
        self.stone_give_ups >= STONE_GIVE_UPS
    }

    pub(crate) fn last_abort_was_inner_loop(&self) -> bool {
        matches!(self.last_abort, Some((TraceAbortReason::InnerLoop, _)))
    }

    /// Counts a recording abort for `reason` at guest instruction `at`.
    /// Returns whether it is the [`MAX_SAME_ABORTS`]th in a row with the same
    /// reason at the same instruction. An abort with no instruction (an
    /// error, a flush, a refused compile) breaks the row.
    /// The reason and guest instruction of the last recording abort.
    pub(crate) const fn last_abort(&self) -> Option<(TraceAbortReason, usize)> {
        self.last_abort
    }

    /// The last abort with the chunk it was met in, for the report.
    pub(crate) fn reported_abort(&self) -> Option<&ReportedAbort> {
        self.reported_abort.as_ref()
    }

    /// [`Self::note_abort`] for an abort met in the frame of `chunk`, which
    /// is not the site's own when the recording had inlined a callee.
    pub(crate) fn note_abort_in(
        &mut self,
        reason: TraceAbortReason,
        at: Option<usize>,
        chunk: Option<&Arc<Chunk>>,
    ) -> bool {
        let repeated = self.note_abort(reason, at);
        if let (Some(report), Some(chunk)) = (self.reported_abort.as_mut(), chunk)
            && let Some((met, _)) = report.at.as_mut()
        {
            *met = Arc::downgrade(chunk);
        }
        repeated
    }

    pub(crate) fn note_abort(&mut self, reason: TraceAbortReason, at: Option<usize>) -> bool {
        self.reported_abort = Some(ReportedAbort {
            reason,
            at: at.map(|pc| (self.chunk.clone(), pc)),
        });
        let this = at.map(|pc| (reason, pc));
        if this.is_some() && this == self.last_abort {
            self.same_aborts = self.same_aborts.saturating_add(1);
        } else {
            self.same_aborts = 1;
        }
        self.last_abort = this;
        this.is_some() && self.same_aborts >= MAX_SAME_ABORTS
    }
}

/// How a recording abort affects its site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AbortClass {
    /// The bytecode or profile can never be traced as is.
    Permanent,
    /// Try again after `cooldown` consults, within the retry budget.
    Retry { cooldown: u32 },
    /// An external event ended the recording. The site is not at fault.
    Neutral,
}

/// Classifies `reason` for a site that has already aborted `attempts` times
/// (not counting this one).
pub(crate) const fn classify_abort(reason: TraceAbortReason, attempts: u8) -> AbortClass {
    match reason {
        TraceAbortReason::UnsupportedProfile
        | TraceAbortReason::CompileRejected
        | TraceAbortReason::YieldingLoop => AbortClass::Permanent,
        TraceAbortReason::Flushed => AbortClass::Neutral,
        // The register file usually settles after one more iteration.
        TraceAbortReason::TypeUnstable => AbortClass::Retry { cooldown: 1 },
        // The recorded iteration left the loop. An odd wait moves the next
        // attempt off the loop's trip count, so a short loop does not hit its
        // exit iteration every time.
        // The iteration that recording started on indexed outside the array
        // (`t[0]` in a loop that runs from 0, the slot past the end, a nil
        // hole). The same wait would start the next recording on the same
        // iteration, so it is odd for the same reason.
        TraceAbortReason::LoopNotClosed | TraceAbortReason::ShortLoop => AbortClass::Retry {
            // The offset moves each retry to another phase of the loop's
            // runs. The cooldowns alone are multiples of 32, so a loop whose
            // runs repeat with a period that divides them (an inner loop of
            // four, three, two, one and no trips has six backedges) would
            // start every attempt on the same trip, and fail it the same way.
            cooldown: (RETRY_COOLDOWN_BASE << (2 * if attempts < 3 { attempts } else { 3 }))
                + 1
                + 2 * (attempts as u32),
        },
        // The same wait keeps growing over the longer budget.
        TraceAbortReason::TableMiss => AbortClass::Retry {
            cooldown: (RETRY_COOLDOWN_BASE << (2 * if attempts < 5 { attempts } else { 5 })) + 1,
        },
        // The recorder sees one iteration's path. A rare branch may hold the
        // opcode or inner loop, so retry within the budget before giving up.
        TraceAbortReason::DepthChange
        | TraceAbortReason::UnsupportedOpcode
        | TraceAbortReason::InnerLoop
        | TraceAbortReason::Error
        | TraceAbortReason::Yield
        | TraceAbortReason::Hook
        | TraceAbortReason::LimitExceeded
        | TraceAbortReason::UnsupportedOperand
        | TraceAbortReason::BorrowOutlivesEffect
        | TraceAbortReason::VirtualEscape
        | TraceAbortReason::RootTailCall
        | TraceAbortReason::SelfCall
        | TraceAbortReason::Divergence => AbortClass::Retry {
            cooldown: RETRY_COOLDOWN_BASE << (2 * if attempts < 3 { attempts } else { 3 }),
        },
    }
}

/// How many transient aborts a site may suffer for `reason` before it is
/// rejected for good.
pub(crate) const fn attempt_limit(reason: TraceAbortReason) -> u8 {
    if matches!(reason, TraceAbortReason::TableMiss) {
        MAX_TABLE_MISS_ATTEMPTS
    } else {
        MAX_RECORDING_ATTEMPTS
    }
}

/// All sites of one VM.
#[derive(Debug)]
pub(crate) struct SiteTable {
    sites: FxHashMap<SiteKey, Site>,
    /// Headers turned away since the last dead-chunk sweep.
    full_misses: usize,
    /// Sites in the `Rejected` state.
    rejected: usize,
    /// Orders rejections.
    clock: u64,
    /// The cap on `rejected`.
    negative_cap: usize,
    /// The headers most recently seen rejected, replaced in turn. A loop
    /// that calls a function with a rejected loop of its own alternates
    /// between two. The weak handle keeps the chunk address from being
    /// reused while the memo lives. Cleared whenever a site is removed or
    /// replaced, the only ways out of `Rejected`.
    last_rejected: [Option<(usize, Weak<Chunk>)>; REJECTED_MEMO_SLOTS],
    /// The slot of `last_rejected` the next header replaces.
    next_rejected: usize,
    /// Addresses of chunks whose pc 0 is a rejected header, direct-mapped
    /// by [`tail_slot`], which a tail call landing there checks before it
    /// looks for a cycle (see [`Self::is_rejected_tail`]). Zero is empty.
    /// Cleared with `last_rejected`.
    rejected_tails: [usize; REJECTED_TAIL_SLOTS],
    /// The chunks of `rejected_tails`, whose weak handles keep the addresses
    /// from being reused while the memo holds them.
    rejected_tail_chunks: [Weak<Chunk>; REJECTED_TAIL_SLOTS],
    /// The address of the chunk whose function entry
    /// ([`FUNCTION_ENTRY_PC`]) was last found rejected, zero for none. A
    /// self call of that chunk offers its entry to nobody (see
    /// [`Self::is_rejected_entry`]). Cleared with `last_rejected`.
    rejected_entry: usize,
    /// The chunk of `rejected_entry`, whose weak handle keeps the address
    /// from going to another chunk (a function `load` makes in a loop)
    /// while the memo holds it.
    rejected_entry_chunk: Weak<Chunk>,
}

/// Slots of [`SiteTable`]'s memo of rejected headers.
const REJECTED_MEMO_SLOTS: usize = 4;
/// Slots of [`SiteTable`]'s memo of rejected tail call headers. A cycle of
/// tail calls lands on at most eight functions (see `TailCycles`).
const REJECTED_TAIL_SLOTS: usize = 8;

/// The slot of `rejected_tails` for the chunk at address `chunk`.
#[inline]
#[allow(clippy::cast_possible_truncation)]
const fn tail_slot(chunk: usize) -> usize {
    // Fibonacci hashing: the top three bits of the product.
    ((chunk as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 61) as usize
}

impl Default for SiteTable {
    fn default() -> Self {
        Self {
            sites: FxHashMap::default(),
            full_misses: 0,
            rejected: 0,
            clock: 0,
            negative_cap: MAX_NEGATIVE_SITES,
            last_rejected: Default::default(),
            next_rejected: 0,
            rejected_tails: [0; REJECTED_TAIL_SLOTS],
            rejected_tail_chunks: Default::default(),
            rejected_entry: 0,
            rejected_entry_chunk: Weak::new(),
        }
    }
}

impl SiteTable {
    /// Forgets the memos of rejected headers, as a site leaves `Rejected`
    /// or goes away.
    fn forget_rejected_memos(&mut self) {
        self.last_rejected = Default::default();
        self.rejected_tails = [0; REJECTED_TAIL_SLOTS];
        self.rejected_tail_chunks = Default::default();
        self.rejected_entry = 0;
        self.rejected_entry_chunk = Weak::new();
    }

    /// Whether the chunk at address `chunk` is remembered with a rejected
    /// header at pc 0, where its tail calls land.
    #[inline]
    pub(crate) fn is_rejected_tail(&self, chunk: usize) -> bool {
        self.rejected_tails[tail_slot(chunk)] == chunk
    }

    /// Remembers `chunk` for [`Self::is_rejected_tail`] if its header at pc 0
    /// is rejected.
    pub(crate) fn remember_rejected_tail(&mut self, chunk: *const Chunk) {
        let Some(site) = self
            .sites
            .get(&(chunk as usize, 0))
            .filter(|site| std::ptr::eq(site.chunk.as_ptr(), chunk))
        else {
            return;
        };
        if site.state == SiteState::Rejected {
            let slot = tail_slot(chunk as usize);
            self.rejected_tail_chunks[slot] = site.chunk.clone();
            self.rejected_tails[slot] = chunk as usize;
        }
    }

    /// Whether the chunk at address `chunk` is remembered with a rejected
    /// function entry.
    #[inline]
    pub(crate) fn is_rejected_entry(&self, chunk: usize) -> bool {
        self.rejected_entry == chunk
    }

    /// Remembers `chunk` for [`Self::is_rejected_entry`] if its function
    /// entry is rejected.
    pub(crate) fn remember_rejected_entry(&mut self, chunk: *const Chunk) {
        let Some(site) = self
            .sites
            .get(&(chunk as usize, FUNCTION_ENTRY_PC))
            .filter(|site| std::ptr::eq(site.chunk.as_ptr(), chunk))
        else {
            return;
        };
        if site.state == SiteState::Rejected {
            self.rejected_entry_chunk = site.chunk.clone();
            self.rejected_entry = chunk as usize;
        }
    }

    /// Whether `chunk` and `pc` are a header recently remembered as rejected.
    pub(crate) fn is_last_rejected(&self, chunk: &Arc<Chunk>, pc: usize) -> bool {
        self.is_last_rejected_at(Arc::as_ptr(chunk), pc)
    }

    /// [`Self::is_last_rejected`] for a chunk known by address.
    #[inline]
    pub(crate) fn is_last_rejected_at(&self, chunk: *const Chunk, pc: usize) -> bool {
        self.last_rejected.iter().any(|slot| {
            slot.as_ref()
                .is_some_and(|(key_pc, weak)| *key_pc == pc && std::ptr::eq(weak.as_ptr(), chunk))
        })
    }

    /// Remembers a header found in the `Rejected` state.
    pub(crate) fn remember_rejected(&mut self, chunk: &Arc<Chunk>, pc: usize) {
        self.last_rejected[self.next_rejected] = Some((pc, Arc::downgrade(chunk)));
        self.next_rejected = (self.next_rejected + 1) % REJECTED_MEMO_SLOTS;
    }

    /// The site for `chunk` and `pc`, created cold if it is new and the table
    /// has room. A dead chunk's entry whose address got reused is replaced.
    pub(crate) fn site_mut(&mut self, chunk: &Arc<Chunk>, pc: usize) -> Option<&mut Site> {
        let key = (chunk_address(chunk), pc);
        if self.sites.len() >= MAX_TRACE_SITES && !self.sites.contains_key(&key) {
            // Sweeping dead chunks walks the whole table. With live chunks it
            // frees nothing, so sweep once per table's worth of turned-away
            // headers and keep the other consults O(1).
            self.full_misses += 1;
            if self.full_misses < MAX_TRACE_SITES {
                return None;
            }
            self.full_misses = 0;
            self.forget_rejected_memos();
            self.sites.retain(|_, site| site.chunk.strong_count() > 0);
            self.recount_rejected();
            if self.sites.len() >= MAX_TRACE_SITES {
                return None;
            }
        }
        let site = self.sites.entry(key).or_insert_with(|| Site::new(chunk));
        if site.chunk.strong_count() == 0 || !std::ptr::eq(site.chunk.as_ptr(), Arc::as_ptr(chunk))
        {
            if site.state == SiteState::Rejected {
                self.rejected -= 1;
            }
            // `forget_rejected_memos`, field by field, while `site` lives.
            self.last_rejected = Default::default();
            self.rejected_tails = [0; REJECTED_TAIL_SLOTS];
            self.rejected_tail_chunks = Default::default();
            self.rejected_entry = 0;
            self.rejected_entry_chunk = Weak::new();
            *site = Site::new(chunk);
        }
        Some(site)
    }

    fn recount_rejected(&mut self) {
        self.rejected = self
            .sites
            .values()
            .filter(|site| site.state == SiteState::Rejected)
            .count();
    }

    /// Marks the site at `chunk` and `pc` rejected. Past the negative cache
    /// cap the oldest rejected site is forgotten, so its header starts
    /// counting again. Returns how many sites that dropped.
    pub(crate) fn reject(&mut self, chunk: &Arc<Chunk>, pc: usize, cause: RejectCause) -> u64 {
        let key = (chunk_address(chunk), pc);
        self.clock += 1;
        let clock = self.clock;
        let Some(site) = self.existing_mut(chunk, pc) else {
            return 0;
        };
        site.reject_cause = Some(cause);
        if site.state != SiteState::Rejected {
            site.state = SiteState::Rejected;
            self.rejected += 1;
        }
        // Re-rejecting a site refreshes its age.
        if let Some(site) = self.sites.get_mut(&key) {
            site.rejected_at = clock;
        }
        let mut dropped = 0;
        while self.rejected > self.negative_cap {
            let oldest = self
                .sites
                .iter()
                .filter(|(_, site)| site.state == SiteState::Rejected)
                .min_by_key(|(_, site)| site.rejected_at)
                .map(|(key, _)| *key);
            let Some(oldest) = oldest else {
                break;
            };
            self.sites.remove(&oldest);
            self.forget_rejected_memos();
            self.rejected -= 1;
            dropped += 1;
        }
        dropped
    }

    /// Takes the rejected site at `pc` out of the negative cache so that it
    /// counts again from `backedges` and records anew with a fresh retry
    /// budget, and says it met an inner loop (so the recording unrolls short
    /// ones). Returns whether it did: a site that was not rejected for an
    /// inner loop, or was reopened before, stays as it is.
    pub(crate) fn reopen_for_inner_loop(
        &mut self,
        chunk: &Arc<Chunk>,
        pc: usize,
        backedges: u32,
    ) -> bool {
        let Some(site) = self.existing_mut(chunk, pc) else {
            return false;
        };
        if site.state != SiteState::Rejected || !site.last_abort_was_inner_loop() || site.reopened {
            return false;
        }
        site.reopened = true;
        site.state = SiteState::Cold { backedges };
        site.inner_loop_met = true;
        site.attempts = 0;
        site.restarts = 0;
        site.same_aborts = 0;
        self.rejected -= 1;
        self.forget_rejected_memos();
        true
    }

    /// Hands the loop header at `pc` to a trace tree that runs it, creating
    /// its site if the interpreter has not reached it yet. Only a header that
    /// no trace of its own claims changes: a recorded or installed header
    /// keeps its state. Returns whether the site is covered.
    pub(crate) fn cover(&mut self, chunk: &Arc<Chunk>, pc: usize) -> bool {
        // A loop the interpreter has not yet taken the backedge of has no
        // site. The tree may run it before the interpreter ever does.
        let Some(site) = self.site_mut(chunk, pc) else {
            return false;
        };
        if site.cover_revoked {
            return false;
        }
        match site.state {
            SiteState::Covered => return true,
            SiteState::Cold { .. }
            | SiteState::Cooldown { .. }
            | SiteState::Rejected
            | SiteState::Queued => {}
            SiteState::Recording | SiteState::Compiling | SiteState::Installed { .. } => {
                return false;
            }
        }
        let was_rejected = site.state == SiteState::Rejected;
        site.covered_rejected = was_rejected;
        site.state = SiteState::Covered;
        if was_rejected {
            self.rejected -= 1;
            self.forget_rejected_memos();
        }
        true
    }

    /// Records which tree covers the header at `pc`, and where its iteration
    /// count stands, for the checks of [`SiteState::Covered`].
    pub(crate) fn note_cover(&mut self, chunk: &Arc<Chunk>, pc: usize, index: u32, mark: u64) {
        if let Some(site) = self.existing_mut(chunk, pc)
            && site.state == SiteState::Covered
            && (site.cover_index != index || site.cover_consults == 0)
        {
            site.cover_index = index;
            site.cover_consults = 0;
            site.cover_mark = mark;
        }
    }

    /// Ends the coverage of the header at `pc`: its tree is gone, so the
    /// header is rejected. `None` if the header was not covered, else the
    /// number of rejected sites the negative cache dropped.
    pub(crate) fn uncover(&mut self, chunk: &Arc<Chunk>, pc: usize) -> Option<u64> {
        match self.existing_mut(chunk, pc) {
            Some(site) if site.state == SiteState::Covered => {
                Some(self.reject(chunk, pc, RejectCause::Uncovered))
            }
            _ => None,
        }
    }

    /// Sets the negative cache cap.
    #[cfg(test)]
    pub(crate) fn set_negative_cap(&mut self, cap: usize) {
        self.negative_cap = cap;
    }

    /// Sites currently in the `Rejected` state.
    #[cfg(test)]
    pub(crate) const fn rejected_count(&self) -> usize {
        self.rejected
    }

    /// Number of sites, for tests.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.sites.len()
    }

    /// The site for `chunk` and `pc` if it exists and belongs to this chunk.
    /// Never creates one.
    pub(crate) fn existing_mut(&mut self, chunk: &Arc<Chunk>, pc: usize) -> Option<&mut Site> {
        self.existing_mut_at(Arc::as_ptr(chunk), pc)
    }

    /// [`Self::existing_mut`] for a chunk known by address.
    pub(crate) fn existing_mut_at(&mut self, chunk: *const Chunk, pc: usize) -> Option<&mut Site> {
        self.sites
            .get_mut(&(chunk as usize, pc))
            .filter(|site| std::ptr::eq(site.chunk.as_ptr(), chunk))
    }

    /// The site if it exists. Never creates one.
    pub(crate) fn site(&self, chunk: &Arc<Chunk>, pc: usize) -> Option<&Site> {
        self.sites
            .get(&(chunk_address(chunk), pc))
            .filter(|site| std::ptr::eq(site.chunk.as_ptr(), Arc::as_ptr(chunk)))
    }

    pub(crate) fn clear(&mut self) {
        self.sites.clear();
        self.forget_rejected_memos();
        self.rejected = 0;
    }

    #[cfg(test)]
    pub(crate) fn state_at_pc(&self, pc: usize) -> Option<SiteState> {
        self.sites
            .iter()
            .find(|((_, site_pc), _)| *site_pc == pc)
            .map(|(_, site)| site.state)
    }

    #[cfg(test)]
    pub(crate) fn state(&self, chunk: &Arc<Chunk>, pc: usize) -> Option<SiteState> {
        self.site(chunk, pc).map(|site| site.state)
    }

    /// Whether the only site at `pc` takes no more versions, for tests.
    #[cfg(test)]
    pub(crate) fn versions_closed_at_pc(&self, pc: usize) -> Option<bool> {
        self.sites
            .iter()
            .find(|((_, site_pc), _)| *site_pc == pc)
            .map(|(_, site)| site.versions_closed)
    }
}
