//! VM-owned native-tier state: the compile broker, the trace policy, and the
//! plan caches shared with interpreter fast paths.

use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::{Arc, Weak};

use crate::bytecode::Chunk;
use crate::jit::JitBroker;
use crate::jit_ir::{ChunkDigest, InlineCallPlan};
use crate::runtime::ClosureHandle;

use crate::compat::ExecutionMode;
use crate::vm::JitStatistics;
use crate::vm::jit::inline_table::InlineTablePlan;
use crate::vm::jit::scalar_table::PlanCache;
use crate::vm::jit::trace::TracePolicy;

pub(crate) const JIT_QUEUE_CAPACITY: usize = 16;
// Backedges of a loop the tracer rejected that the fast loop passes without a
// lookup, for the first skip.
pub(crate) const TRACING_REJECTED_BACKEDGE_SKIP: u32 = 64;
// The longest skip a loop that stays rejected earns. Each skip that no native
// run interrupted doubles the next one, up to this.
pub(crate) const TRACING_REJECTED_BACKEDGE_SKIP_MAX: u32 = 4_096;
/// Rejected headers whose skips [`RejectedSkips`] keeps at once. A program with
/// more rejected loops in play than this re-arms them in turn.
const REJECTED_SKIP_SLOTS: usize = 4;
pub(crate) const NATIVE_GC_PENDING_BUDGET: i64 = 1_024;
// Use a wide steady state slice to avoid bouncing through the VM in long
// loops. A slice charges the collector for the table constructors it elided
// only when it ends (about 200 bytes each), which at this size is tens of
// megabytes at most. A table a trace really makes misses once the collector
// is owed a step, so those do not pile up debt. A pending collection selects
// the much smaller budget above.
pub(crate) const NATIVE_STEADY_STATE_BUDGET: i64 = 262_144;
pub(crate) const MAX_INLINE_CALL_PLAN_CACHE: usize = 1_024;
/// Recent tail call landings [`TailCycles`] remembers, and so the longest
/// cycle of functions it recognises.
const TAIL_CYCLE_LANDINGS: usize = 8;
/// [`TAIL_CYCLE_LANDINGS`] as a tail call count.
const TAIL_CYCLE_CALLS: u32 = 8;

/// The skip window of one rejected loop header.
#[derive(Clone, Copy, Debug, Default)]
struct RejectedSkip {
    /// The address of the header chunk's code, which names the chunk without
    /// a handle in the fast loop. Zero for a slot that holds nothing.
    code: usize,
    pc: usize,
    /// Backedges of the header still to pass without a lookup.
    remaining: u32,
    /// The length of the next window of this header, which doubles while the
    /// loop stays rejected.
    next: u32,
}

/// The backedges the fast loop passes without a lookup because the tracer
/// rejected their loop, kept for each such header. A window belongs to one
/// header, so a trace whose header is reached inside it, or a loop inside the
/// rejected one that still counts, is consulted as usual. That lets a
/// rejected loop that runs for a few trips between native runs (the second
/// loop of a function whose first loop is a trace) skip its lookups too.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RejectedSkips {
    slots: [RejectedSkip; REJECTED_SKIP_SLOTS],
    /// The slot a header without one takes over.
    victim: usize,
}

impl RejectedSkips {
    /// Passes a backedge to `pc` of the chunk with code at `code` if its
    /// window has any left. Called by the fast loop on every backedge that
    /// would leave it for a lookup.
    #[inline(always)]
    pub(crate) fn pass(&mut self, code: usize, pc: usize) -> bool {
        for slot in &mut self.slots {
            if slot.pc == pc && slot.code == code && slot.remaining > 0 {
                slot.remaining -= 1;
                return true;
            }
        }
        false
    }

    /// Opens the next window of the header, the first one
    /// [`TRACING_REJECTED_BACKEDGE_SKIP`] long and each one after it twice as
    /// long as the one before, up to `cap` (at most
    /// [`TRACING_REJECTED_BACKEDGE_SKIP_MAX`]). Returns its length.
    pub(crate) fn arm(&mut self, code: usize, pc: usize, cap: u32) -> u32 {
        let at = self
            .slots
            .iter()
            .position(|slot| slot.code == code && slot.pc == pc)
            .unwrap_or_else(|| {
                let at = self.victim;
                self.victim = (self.victim + 1) % REJECTED_SKIP_SLOTS;
                self.slots[at] = RejectedSkip {
                    code,
                    pc,
                    remaining: 0,
                    next: TRACING_REJECTED_BACKEDGE_SKIP,
                };
                at
            });
        let slot = &mut self.slots[at];
        let length = slot.next.min(cap);
        slot.remaining = length;
        slot.next = length
            .saturating_mul(2)
            .min(cap)
            .min(TRACING_REJECTED_BACKEDGE_SKIP_MAX);
        length
    }
}

/// The latest Lua tail calls in tracing mode: the frame depth, the chunk the
/// call landed on (by address) and the frame's tail call count after it. A frame
/// whose tail calls go round a cycle lands on the same chunk again within a
/// few calls, which is what makes that chunk's pc 0 a loop header. A frame
/// that only passes through a chain of tail calls never does.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TailCycles {
    landings: [(usize, usize, u32); TAIL_CYCLE_LANDINGS],
    next: usize,
}

impl TailCycles {
    /// Notes a landing and answers whether the same frame landed on `chunk`
    /// within the last few tail calls. A frame at the same depth that came
    /// and went in between can only make a cycle look likely, which costs a
    /// lookup.
    ///
    /// The count saturates at `u32::MAX`, so past that a frame's landings
    /// all carry the same count. Equal counts then still tell a cycle, or a
    /// loop that ran more than 2^32 tail calls would stop being one.
    #[inline]
    pub(crate) fn landed(&mut self, depth: usize, chunk: usize, tail_calls: u32) -> bool {
        let cycle = self.landings.iter().any(|&(at, seen, count)| {
            at == depth
                && seen == chunk
                && ((count < tail_calls && tail_calls - count <= TAIL_CYCLE_CALLS)
                    || (count == u32::MAX && tail_calls == u32::MAX))
        });
        self.landings[self.next] = (depth, chunk, tail_calls);
        self.next = (self.next + 1) % TAIL_CYCLE_LANDINGS;
        cycle
    }
}

/// Everything the VM keeps for native execution and plan-based fast paths.
pub(crate) struct JitState {
    /// Created at the first trace compile request. Interpreter instances keep
    /// this empty.
    pub(crate) broker: Option<JitBroker>,
    pub(crate) chunk_digests: FxHashMap<usize, ChunkJitMetadata>,
    /// Results are immutable chunk-derived data keyed by generation-safe
    /// closure identity. `None` caches a deliberate refusal.
    pub(crate) inline_call_plans: FxHashMap<ClosureHandle, Option<CachedInlineCall>>,
    /// The last direct plan or permanent refusal. This avoids hashing the
    /// same closure on consecutive scalar calls. Forwarded plans and their
    /// retryable refusals stay in the map because a live target cell can
    /// change their result.
    pub(crate) inline_call_hot_cache: Option<(ClosureHandle, Option<CachedInlineCall>)>,
    /// Plans of the closures called most recently, direct-mapped by handle,
    /// for calls that borrow a plan without a clone or a lookup.
    pub(crate) inline_call_hot_plans:
        [Option<(ClosureHandle, Arc<InlineCallPlan>)>; HOT_INLINE_PLAN_SLOTS],
    /// Only vararg-forwarding refusals can become inlineable after a live
    /// closure rebinding. All other refusals stay final for the closure's
    /// immutable bytecode and bypass plan discovery on future calls.
    pub(crate) inline_call_retryable_refusals: FxHashSet<ClosureHandle>,
    /// Bounded, generation-safe table plans. `None` records an unsupported
    /// closure so a hot method does not need to be decoded every call.
    pub(crate) inline_table_plans: FxHashMap<ClosureHandle, Option<Arc<InlineTablePlan>>>,
    pub(crate) scalar_table_plans: PlanCache,
    pub(crate) statistics: JitStatistics,
    /// Trace engine ownership and state.
    pub(crate) trace: TracePolicy,
    /// Interpreter fast-path backedges, of any loop, to run before asking
    /// `maybe_run_native` again. Set when the tracer rejected a header while
    /// no trace is installed, or a header only waits for its compile, so a hot
    /// interpreted loop does not pay the full lookup on every iteration.
    pub(crate) backedge_skip: u32,
    /// The length of the next skip `backedge_skip` is armed with for a
    /// rejected loop while no trace is installed. It grows while the loop
    /// runs undisturbed and starts over after a native run.
    pub(crate) backedge_skip_next: u32,
    /// The skip windows of the loops the tracer rejected while traces are
    /// installed.
    pub(crate) rejected_skips: RejectedSkips,
    /// The target pc of the fast-path backedge that last handed control to
    /// the general loop. Only a lookup for that backedge arms a skip.
    pub(crate) fast_backedge_pc: Option<usize>,
    /// Set when a Lua tail call in tracing mode replaced the running frame,
    /// which now stands at pc 0 of the callee. The dispatch loop then
    /// consults the policy for that pc as a loop header, like a backward
    /// branch's target: a chain of tail calls that comes back to a function
    /// is a loop. The frame's `last_interpreted_pc` stays what the
    /// interpreter left (none), so this flag is the signal.
    pub(crate) tail_landing: bool,
    /// The latest tail call landings, which tell a cycle of tail calls from
    /// a chain that passes through.
    pub(crate) tail_cycles: TailCycles,
    /// Set when a Lua call in tracing mode pushed a frame of the calling
    /// frame's own chunk (direct self recursion), which now stands at pc 0.
    /// The dispatch loop then consults the policy for the function's entry
    /// (a function-entry trace, see `trace::recorder`, "Function traces").
    /// Like `tail_landing`, the frame's `last_interpreted_pc` (none) cannot
    /// tell, so this flag is the signal.
    pub(crate) call_landing: bool,
}

impl Default for JitState {
    fn default() -> Self {
        Self::new(ExecutionMode::default())
    }
}

impl JitState {
    pub(crate) fn new(mode: ExecutionMode) -> Self {
        Self {
            trace: TracePolicy::new(mode),
            broker: None,
            chunk_digests: FxHashMap::default(),
            inline_call_plans: FxHashMap::default(),
            inline_call_hot_cache: None,
            inline_call_hot_plans: std::array::from_fn(|_| None),
            inline_call_retryable_refusals: FxHashSet::default(),
            inline_table_plans: FxHashMap::default(),
            scalar_table_plans: PlanCache::default(),
            statistics: JitStatistics::default(),
            backedge_skip: 0,
            backedge_skip_next: TRACING_REJECTED_BACKEDGE_SKIP,
            rejected_skips: RejectedSkips::default(),
            fast_backedge_pc: None,
            tail_landing: false,
            tail_cycles: TailCycles::default(),
            call_landing: false,
        }
    }
}

/// The digest of `chunk`, cached by chunk identity. Takes the JIT state alone,
/// so a caller can hold a borrow of the runtime (a closure's chunk) while it
/// looks the digest up.
pub(crate) fn chunk_digest(jit: &mut JitState, chunk: &Arc<Chunk>) -> ChunkDigest {
    let identity = Arc::as_ptr(chunk) as usize;
    if let Some(metadata) = jit.chunk_digests.get(&identity)
        && metadata.owner.as_ptr() == Arc::as_ptr(chunk)
    {
        return metadata.digest;
    }
    if jit.chunk_digests.len() >= 1_024 {
        jit.chunk_digests
            .retain(|_, metadata| metadata.owner.strong_count() != 0);
        while jit.chunk_digests.len() >= 1_024 {
            let eviction = *jit
                .chunk_digests
                .keys()
                .next()
                .expect("a full metadata map has an entry to evict");
            jit.chunk_digests.remove(&eviction);
        }
    }
    let digest = ChunkDigest::of(chunk);
    jit.chunk_digests.insert(
        identity,
        ChunkJitMetadata {
            owner: Arc::downgrade(chunk),
            digest,
        },
    );
    digest
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ForwardedInlineCallGuard {
    pub(crate) target_digest: ChunkDigest,
}

pub(crate) const HOT_INLINE_PLAN_SLOTS: usize = 8;

#[derive(Clone)]
pub(crate) struct CachedInlineCall {
    pub(crate) plan: Arc<InlineCallPlan>,
    pub(crate) callee_digest: ChunkDigest,
    pub(crate) forwarded: Option<ForwardedInlineCallGuard>,
}

pub(crate) struct ChunkJitMetadata {
    pub(crate) owner: Weak<Chunk>,
    pub(crate) digest: ChunkDigest,
}

#[cfg(test)]
mod tests {
    use super::*;

    const CODE: usize = 0x1000;

    #[test]
    fn a_window_passes_exactly_its_length_of_backedges_of_its_own_header() {
        let mut skips = RejectedSkips::default();
        // Nothing is armed.
        assert!(!skips.pass(CODE, 7));
        let length = skips.arm(CODE, 7, u32::MAX);
        assert_eq!(length, TRACING_REJECTED_BACKEDGE_SKIP);
        // Another header of the same chunk, and the same pc of another chunk,
        // are not skipped.
        assert!(!skips.pass(CODE, 8));
        assert!(!skips.pass(CODE + 16, 7));
        for _ in 0..length {
            assert!(skips.pass(CODE, 7));
        }
        assert!(!skips.pass(CODE, 7), "the window ran out");
    }

    #[test]
    fn each_window_of_a_header_doubles_up_to_the_cap() {
        let mut skips = RejectedSkips::default();
        let mut length = skips.arm(CODE, 3, u32::MAX);
        assert_eq!(length, TRACING_REJECTED_BACKEDGE_SKIP);
        while length < TRACING_REJECTED_BACKEDGE_SKIP_MAX {
            let next = skips.arm(CODE, 3, u32::MAX);
            assert_eq!(next, length * 2);
            length = next;
        }
        assert_eq!(
            skips.arm(CODE, 3, u32::MAX),
            TRACING_REJECTED_BACKEDGE_SKIP_MAX
        );
    }

    #[test]
    fn a_capped_header_keeps_windows_of_the_cap() {
        let mut skips = RejectedSkips::default();
        let cap = TRACING_REJECTED_BACKEDGE_SKIP;
        for _ in 0..4 {
            assert_eq!(skips.arm(CODE, 5, cap), cap);
        }
        // A header with a longer cap, armed in the same table, is unaffected.
        assert_eq!(skips.arm(CODE, 6, u32::MAX), TRACING_REJECTED_BACKEDGE_SKIP);
        assert_eq!(
            skips.arm(CODE, 6, u32::MAX),
            2 * TRACING_REJECTED_BACKEDGE_SKIP
        );
    }

    #[test]
    fn more_headers_than_slots_take_over_slots_in_turn() {
        let mut skips = RejectedSkips::default();
        for pc in 0..REJECTED_SKIP_SLOTS {
            skips.arm(CODE, pc, u32::MAX);
        }
        for pc in 0..REJECTED_SKIP_SLOTS {
            assert!(skips.pass(CODE, pc), "pc {pc} holds its window");
        }
        // One more header takes the oldest slot, and starts at the first
        // window length.
        let newcomer = skips.arm(CODE, 99, u32::MAX);
        assert_eq!(newcomer, TRACING_REJECTED_BACKEDGE_SKIP);
        assert!(skips.pass(CODE, 99));
        assert!(!skips.pass(CODE, 0), "the first header lost its slot");
        assert!(skips.pass(CODE, 1));
    }

    #[test]
    fn a_rearmed_window_replaces_the_remains_of_the_last() {
        let mut skips = RejectedSkips::default();
        let first = skips.arm(CODE, 2, u32::MAX);
        assert!(skips.pass(CODE, 2));
        let second = skips.arm(CODE, 2, u32::MAX);
        assert_eq!(second, first * 2);
        for _ in 0..second {
            assert!(skips.pass(CODE, 2));
        }
        assert!(!skips.pass(CODE, 2));
    }
}
