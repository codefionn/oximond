//! Installing compiled traces and entering them.
//!
//! A closed trace goes to the compile broker (`oximond::jit::JitBroker`) as an
//! owned request and its site waits in `Compiling`. When the
//! code comes back at a VM safepoint, [`TracePolicy::install_compiled`] puts
//! it in the cache and the site becomes `Installed`. When the interpreter
//! reaches the header the policy hands out the table index, and
//! [`TracePolicy::enter`] runs the code:
//! 1. Check the frame is the trace's: same chunk, same header pc, and a
//!    window the snapshots fit. The caller has synced a frame with captured
//!    locals, so its registers hold what the cells do.
//!    The bank starts with the trace's pinned closures (lane `k` is pinned
//!    closure `k`), and a trace whose pinned closure is gone, whose cell
//!    aliases (names the trace holds one variable for) differ now, or whose
//!    deepest callee would push past the call-depth limit, does not enter.
//! 2. Read each entry input into a typed input slot: a register, or, past the
//!    registers, an upvalue cell or a global the trace reads (see
//!    [`oximond::jit_ir::trace::ExternalSlot`]). A table, closure or string
//!    becomes a lane id, an index into the per-run root bank that holds the
//!    value.
//!    The control register of a `next` loop whose trace keeps the key in a
//!    lane ([`TraceIr::boxed_control`]) is the one input that takes any
//!    value: it gets a lane of its own that holds the register's value
//!    whatever its type, and no other input can share the lane.
//!    Native code checks the types, so a mismatch executes no trace
//!    operation. A header can hold several versions of its trace, one per
//!    entry type signature, and an entry mismatch tries the next older
//!    version before giving up.
//! 3. Resolve the trace's table bindings (see
//!    [`oximond::jit_ir::trace::TableBinding`]): for each table the trace
//!    reads or writes, find its storage now and hand native code the
//!    addresses as extra input slots after the entry inputs. An array
//!    binding gives the array part's address and length, a field binding
//!    the address of a present slot that holds a value of the recorded type,
//!    a length binding the table's border, a bytes binding the address and
//!    length of a string's bytes (the string is an entry input, and the bank
//!    keeps it alive). A table with no such storage
//!    (the field is gone or changed type, a `__len` appeared) is an entry
//!    mismatch, and so is an array whose first slot is not of the type the
//!    recording saw there, so that the versions of a header can be told
//!    apart by the element type of the arrays they were recorded over. The
//!    addresses stay valid for the run because nothing in it can run Lua
//!    code or collect garbage, and a run changes table storage only by
//!    overwriting a payload with a value of the same type, or through a raw
//!    set the VM makes for it ([`RunTables`]), after which native code reads
//!    the array words the set rewrote.
//! 4. Run. A trace calls no native function except through the VM's library
//!    call entry ([`super::host`]), which allocates strings only. A call
//!    of a Lua closure runs inside the trace without a frame (see the module
//!    docs of [`super`]), so the interpreter frame, the cells and the globals
//!    stay exactly as they were until an exit. Nothing else can observe a
//!    cell the trace keeps in a native value.
//! 5. On an exit, rebuild the frame from the exit's snapshot. Each exit has
//!    a restore plan built and checked at install time: one action per
//!    register (keep, constant, root constant, a typed slot of the exit
//!    area, or an opaque lane), so an exit is a single pass with nothing
//!    left to fail. The plan also writes the upvalue cells the snapshot
//!    changes, those of pinned closures included, and, for an exit inside
//!    an inlined call, pushes the callee frames above the root frame. The
//!    plan applies exactly what [`super::deopt::restore_snapshot`] would, and
//!    a test holds the two together. The caller then flushes captured locals back into their
//!    cells.
//! 6. Clear the root bank. Between runs nothing the trace touched stays
//!    rooted, so a trace never keeps an object (or a weak table's entry)
//!    alive.
//!
//! # Function traces
//!
//! A function trace ([`TraceIr::function`]) is entered at pc 0 of a frame a
//! self call just pushed, with a direct caller, no interior state, and every
//! cell or global its self calls read their callee from holding the frame's
//! closure. A run nests activations up to a cap: [`MAX_NATIVE_ACTIVATIONS`],
//! and the frames the call-depth limit has left. An activation that returns
//! from the entered one ends the run with [`TraceRun::Returned`], and the VM
//! returns from the frame. An exit from activation `k` leaves the state of
//! each activation below it in its own part of the exit area, at the unwind
//! exit of the self call it stands at
//! ([`oximond::jit::TraceExitClass::Unwind`]): the restore applies the entered
//! frame's plan, then pushes a frame of the same closure for each further
//! activation (returning to the call window of the one below, as the
//! interpreter's call would) and applies its plan, and finally pushes the
//! frame that left and applies the exit's own plan. Function traces store
//! nothing and allocate nothing, so every plan only writes registers.
//!
//! # Side traces
//!
//! Every exit of a trace is counted per guard. A branch guard that has failed
//! [`SIDE_EXIT_THRESHOLD`] times asks [`TracePolicy::consider_side`] to start a
//! side recording from the state the exit just restored (see [`super`]). The
//! compiled code of a trace is a tree: its exits name snapshots of the root or
//! of a side ([`oximond::jit::ExitDescriptor::member`]), and a restore plan is
//! built for each from that member's own snapshot. A trace keeps the tree as
//! recorded ([`InstalledTrace::source_ir`]) and a version number that advances
//! when a recompile replaces it, so an optimized recompile of an older tree
//! never lands over a newer one. A trace that fails its productivity window
//! while a side trace is recording, compiling, or still wanted for a guard
//! that failed often is forgiven for a bounded number of windows, and the
//! window starts over when the new tree installs.
//!
//! # Not re-entering
//!
//! After an exit the dispatcher executes the instruction at the resume pc
//! before it consults the policy again, so a trace never re-enters the state
//! it just left and no frame marker is needed. A poll exit returns to the
//! VM's safepoint and the next backedge may enter again.
//!
//! # Productivity
//!
//! Entering and leaving a trace costs a few hundred instructions, so a trace
//! only pays when it runs several loop-backs per entry. Every
//! [`TRACE_PRODUCTIVITY_WINDOW`] entries (an entry whose types mismatch
//! stays out of the window; a long run of them retires the trace) the policy
//! compares the iterations it ran with
//! [`TRACE_MIN_ITERATIONS_PER_ENTRY`] per entry (fewer for a body that costs
//! the interpreter more, see [`min_iterations_of`]). A trace below that, such as
//! one that fails the same branch every other iteration, is retired and its
//! site backs off like a transient recording abort. After enough retirements
//! the site is rejected. Traces compile once, at the optimized tier: a
//! cheaper first tier for code that lives on would be compiled again. The one
//! exception is the root of a short loop in a nest, which only waits for the
//! tree that replaces it (see "Nested loops" in the module documentation).

use super::cache::{MAX_TRACE_VERSIONS, code_charge};
use super::debug_env;
#[cfg(test)]
use super::deopt::ExitValues;
use super::deopt::{
    CellTarget, RootTail, VirtualFieldKey, allocate_virtual_sized, apply_root_tail,
    clone_borrowed_string, fill_virtual_keyed, pinned_closure, push_inlined_frame,
    push_protected_frame, restore_tail_state, scalar_to_value, virtual_shape,
};
use super::host::{HostState, host_call};
use super::nest;
use super::policy::{
    HOISTED_GUARD_EXITS, MAX_RECORDING_ATTEMPTS, MAX_SEED_FRAMES, MAX_SEEDED_GUEST_INSTRUCTIONS,
    MAX_SIDE_ATTEMPTS, MAX_SIDE_DEPTH, POLY_CALL_EXITS, POLY_CALL_SIDES, PolyCallPath,
    RETIRED_UNPRODUCTIVE_KEPT, RETRY_COOLDOWN_BASE, RejectCause, SIDE_EXIT_THRESHOLD,
    SIDE_GRACE_WINDOWS, SIDE_QUIET_ITERATIONS, STONE_GIVE_UPS, STONE_PATIENCE_ENTRIES, SiteState,
};
use super::recorder::{CallerSeed, Recorder, SeedFrame, SeedRegister, SideSeed, WakeCandidate};
use super::side_queue::SideQueue;
use super::stats::{TraceAbortReason, TraceExitKind, TraceGuardKind, TraceStatistics};
#[cfg(test)]
use super::{CompileRequest, CompileTarget};
use super::{SideFailure, TracePolicy};
use crate::bytecode::Chunk;
use crate::bytecode::Constant;
use crate::bytecode::OpCode;
use crate::compat::CompatibilityProfile;
#[cfg(test)]
use crate::jit::OptimizationTier;
use crate::jit::fill_sealed;
use crate::jit::{
    CompiledTrace, ExitDescriptor, MAX_FUNCTION_TABLE_LANES, MAX_TRACE_TABLE_LANES, TraceBinding,
    TraceControl, TraceExitClass, TraceGuardClass, TraceInputSlot, TraceOutcome, TraceTables,
};
use crate::jit_ir::trace::ROOT_STRING_LANE_BASE;
use crate::jit_ir::trace::SnapshotPhase;
use crate::jit_ir::trace::TreeError;
use crate::jit_ir::trace::{
    BindingKind, CellAlias, CellSource, ExternalSlot, HostFunction,
    PINNED_BUILTIN_COROUTINE_RESUME, PINNED_BUILTIN_COROUTINE_YIELD,
    PINNED_BUILTIN_DEBUG_TRACEBACK, PINNED_BUILTIN_IPAIRS, PINNED_BUILTIN_IPAIRS_ITERATOR,
    PINNED_BUILTIN_NEXT, PINNED_BUILTIN_PAIRS, PINNED_BUILTIN_PCALL, PINNED_BUILTIN_STRING_FORMAT,
    PINNED_BUILTIN_STRING_GSUB, PINNED_BUILTIN_TOSTRING, PINNED_BUILTIN_XPCALL, TraceSide,
    pinned_builtin_kind, pinned_native_index,
};
use crate::jit_ir::trace::{
    Entry as FrameEntry, FrameId, RegisterEntry, Snapshot, TraceIr, TraceOp, TraceScalar,
    TraceType, VirtualAlloc, VirtualKey,
};
use crate::jit_ir::trace::{
    HelperId, ValueId, find_uses_prepared_pattern, gmatch_constant_lane, root_string_slot,
};
use crate::runtime::Value;
use crate::runtime::alloc_phase::{self, Phase};
use crate::runtime::{
    BuiltinFunction, CellHandle, ClosureHandle, Metamethod, NativeFunctionHandle, NativeSlotKind,
    Runtime, RuntimeError, TableHandle, TableShape,
};
use crate::stdlib::substring_range;
use crate::vm::stack::{CallStack, FrameMut};
use crate::vm::{CallReturn, MAX_LUA_CALL_FRAMES};
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::fmt::Write as _;
use std::sync::{Arc, Weak};

/// Loop-backs per entry above which an outer side no longer pays for an
/// integer trace (see `InstalledTrace::outer_side_pays`). At this length the
/// root spends several times more in the loop than it does entering and
/// leaving it.
pub(crate) const OUTER_SIDE_MAX_ITERATIONS_PER_ENTRY: u64 = 128;

/// Entries between productivity checks.
pub(crate) const TRACE_PRODUCTIVITY_WINDOW: u32 = 32;
/// Windows' worth of iterations above the bar a trace may bank against a
/// window below it. A loop whose entries come in bursts (an outer loop left for
/// an inner one at a run of small trip counts, then thousands of iterations in
/// one entry) keeps its trace, and a trace that stops paying for good still
/// retires within a few windows.
pub(crate) const TRACE_PRODUCTIVITY_CREDIT_WINDOWS: u64 = 4;
/// Loop-backs per entry a trace must average to stay installed. An entry and
/// exit cost about as much as a handful of interpreted iterations.
pub(crate) const TRACE_MIN_ITERATIONS_PER_ENTRY: u64 = 8;
/// Activations per entry a function trace's runs that leave at a callee
/// guard must reach not to count toward [`TRACE_CALLEE_STREAK_LIMIT`].
pub(crate) const TRACE_MIN_ACTIVATIONS_PER_ENTRY: u64 = 2;
/// Activations per entry, in halves, a function trace must average over its
/// life to stay installed: one and a half. An entry that runs one activation
/// and returns (a self call at the leaves) costs about two native
/// activations' savings more than the interpreter's call would; a recursion
/// whose self calls are nearly all leaves (a clamp that recurses once, `f(1)`
/// of a tree recursion) stays near one and retires. One whose leaf entries
/// come between deep ones (ackermann's inner calls) stays above it and pays.
pub(crate) const TRACE_MIN_HALF_ACTIVATIONS_PER_ENTRY: u64 = 3;
/// Entries of a function trace before its productivity counts.
pub(crate) const TRACE_FUNCTION_WARMUP: u64 = 8 * TRACE_PRODUCTIVITY_WINDOW as u64;
/// What an entry and exit cost, in operations of a loop body: twelve
/// iterations of a body of about forty. A body that does more work per
/// iteration needs fewer iterations per entry to pay for it.
///
/// Measured on `algorithm/nbody`'s position update (a body of 73 weighted
/// operations that runs five iterations per call): an entry and exit of a
/// trace that binds a few tables cost some 4 000 instructions, about three
/// interpreted iterations of that body, and a native iteration saves 60% of
/// an interpreted one. The loop needed six iterations per entry to break even,
/// and at four and five it cost more than the interpreter ran it for.
const TRACE_ENTRY_WORK: u64 = 480;
/// Fewest iterations per entry any trace may be held to. An entry and exit
/// still cost something when the body is large.
const TRACE_MIN_ITERATIONS_FLOOR: u64 = 3;
/// The bar of a body that does twice [`TRACE_ENTRY_WORK`] per iteration: one
/// native iteration saves more than its entry and exit cost. Mineclonia's
/// loop over a chunk's structure starts (`structures.lua:744`, a dozen
/// inlined calls and a thousand guest instructions) averages 1.2 loop-backs
/// per entry over the emerge, and retired below the floor of three. Held to
/// this bar, a run counts the loop-backs it ran, and the one loop-back every
/// run is credited only when it did not leave inside an inlined call (see
/// `InstalledTrace::note_window`).
const TRACE_MIN_ITERATIONS_HEAVY: u64 = 1;
/// What a table access costs the interpreter, against one for an arithmetic
/// operation: a lookup by name or index, with its dispatch, runs several
/// times what an add does.
const TABLE_ACCESS_WEIGHT: u64 = 6;
/// How many operations of a body one call that searches or rewrites a string
/// weighs when the tier a trace compiles at is chosen (see
/// [`is_helper_bound`]).
const HELPER_BOUND_OPS_PER_CALL: usize = 64;
/// What a trace saves on a call that searches or rewrites a string, against
/// one for an arithmetic operation. The search itself costs the same native
/// and interpreted, so this is the call's dispatch and the copies of its
/// arguments and results, not the search.
const PATTERN_CALL_WEIGHT: u64 = 8;
/// What a call the trace inlined saves the interpreter, against one
/// arithmetic operation: the call and the return, with their frame set up
/// and torn down, and the closure's arguments moved.
const INLINED_CALL_WEIGHT: u64 = 12;

/// Loop-backs per entry a trace of `ir` must average to stay installed:
/// [`TRACE_MIN_ITERATIONS_PER_ENTRY`], and fewer for a trace that reads and
/// writes fields of tables it meets by lane, searches strings, or inlined
/// calls. Such a body holds enough work that the interpreter would spend
/// more than an entry and exit cost on fewer iterations (the body of a loop
/// over records reads and writes a dozen fields; the loop over a chunk's
/// structure starts in Mineclonia's `structures.lua` makes a dozen calls in
/// three trips). Any other trace keeps the bar it always had.
pub(crate) fn min_iterations_of(ir: &TraceIr) -> u64 {
    // A function trace counts activations, each of which saves the
    // interpreter a call and a return besides its body: a few per entry pay
    // for entering and leaving.
    if ir.function.is_some() {
        return TRACE_MIN_ACTIVATIONS_PER_ENTRY;
    }
    let by_lane = |op: &TraceOp| {
        matches!(
            op,
            TraceOp::TableFieldLoad { .. }
                | TraceOp::TableFieldStore { .. }
                | TraceOp::LaneArrayLoad { .. }
                | TraceOp::LaneArrayStore { .. }
        )
    };
    // A call that searches or rewrites a string costs the interpreter
    // hundreds of instructions of dispatch and copies on top of the search,
    // which a trace saves.
    let by_pattern = |op: &TraceOp| match op {
        TraceOp::CallHelper { helper, .. } => helper.is_pattern_call(),
        _ => false,
    };
    let calls = ir
        .frames
        .iter()
        .filter(|frame| frame.inlined.is_some())
        .count() as u64;
    if calls == 0
        && !ir
            .body
            .iter()
            .any(|instruction| by_lane(&instruction.op) || by_pattern(&instruction.op))
    {
        return TRACE_MIN_ITERATIONS_PER_ENTRY;
    }
    let body: u64 = ir
        .body
        .iter()
        .map(|instruction| match instruction.op {
            TraceOp::ArrayLoad { .. }
            | TraceOp::LaneArrayLoad { .. }
            | TraceOp::NumberLoad { .. }
            | TraceOp::ArrayStore { .. }
            | TraceOp::ArrayOverwrite { .. }
            | TraceOp::ArraySet { .. }
            | TraceOp::FieldLoad { .. }
            | TraceOp::FieldStore { .. }
            | TraceOp::TableFieldLoad { .. }
            | TraceOp::TableFieldStore { .. }
            | TraceOp::LaneArrayStore { .. } => TABLE_ACCESS_WEIGHT,
            _ if by_pattern(&instruction.op) => PATTERN_CALL_WEIGHT,
            _ => 1,
        })
        .sum();
    let work = calls * INLINED_CALL_WEIGHT + body;
    if work >= 2 * TRACE_ENTRY_WORK {
        return TRACE_MIN_ITERATIONS_HEAVY;
    }
    (TRACE_ENTRY_WORK / work.max(1))
        .clamp(TRACE_MIN_ITERATIONS_FLOOR, TRACE_MIN_ITERATIONS_PER_ENTRY)
}

/// Whether the calls that search or rewrite a string are most of the work of
/// the trace `ir` and its side traces: each costs the interpreter hundreds of
/// instructions and runs the same host code native, so the code around the
/// calls has little left to save and the optimizing tier's compile (a few
/// times the baseline's) does not pay for itself.
pub(crate) fn is_helper_bound(ir: &TraceIr) -> bool {
    let (mut calls, mut operations) = (0_usize, 0_usize);
    let bodies = std::iter::once(&ir.body).chain(ir.sides.iter().map(|side| &side.ir.body));
    for body in bodies {
        for instruction in body {
            operations += 1;
            if let TraceOp::CallHelper { helper, .. } = &instruction.op
                && helper.is_pattern_call()
            {
                calls += 1;
            }
        }
    }
    calls > 0 && calls * HELPER_BOUND_OPS_PER_CALL >= operations
}

/// Whether `ir` or one of its sides takes a register as an entry input of
/// type `Any` (see `Recorder::with_lazy`). Such a tree grows a side for each
/// type and each outer loop it links back from: a region nest of `shape.lua`
/// compiled 66 times in 20000 calls. The MIR backend compiles the tree
/// itself. A side of it compiled apart goes to Cranelift, at the baseline
/// tier, whose single pass allocator takes a fraction of the backtracking
/// one's time.
pub(crate) fn takes_untyped(ir: &TraceIr) -> bool {
    std::iter::once(ir)
        .chain(ir.sides.iter().map(|side| &*side.ir))
        .any(|member| {
            member
                .entry
                .inputs
                .iter()
                .any(|input| input.ty == TraceType::Any)
        })
}

/// Whether every call in `ir` and its side traces that searches or rewrites
/// with a pattern names a root-string constant as the pattern and has a
/// prepared twin (see `Lowerer::prepared_pattern`), so that at
/// `TracingOptFlags::STRING_CONSTANTS` each one runs the twin. The steps of a
/// `string.gmatch` iterator have a twin only when the trace makes its iterators
/// from one constant pattern.
pub(crate) fn pattern_calls_are_prepared(ir: &TraceIr) -> bool {
    let constant_lane = |ir: &TraceIr, value: ValueId| {
        ir.preheader
            .iter()
            .chain(&ir.body)
            .find_map(|instruction| match instruction.op {
                TraceOp::Constant(TraceScalar::String(lane))
                    if instruction.result == Some(value) && root_string_slot(lane).is_some() =>
                {
                    Some(lane)
                }
                _ => None,
            })
    };
    // Whether all the `gmatch` calls of the tree name one constant pattern,
    // which the lowering guesses for a step of an iterator that entered a
    // trace (see `Lowerer::guessed_gmatch`).
    let guessed =
        gmatch_constant_lane(std::iter::once(ir).chain(ir.sides.iter().map(|s| &*s.ir))).is_some();
    let calls = |ir: &TraceIr| {
        // The iterators this trace makes from a constant pattern.
        let made: Vec<ValueId> = ir
            .preheader
            .iter()
            .chain(&ir.body)
            .filter_map(|instruction| match &instruction.op {
                TraceOp::CallHelper { helper, args, .. }
                    if *helper == HelperId::GMATCH_NEW
                        && args
                            .get(1)
                            .is_some_and(|pattern| constant_lane(ir, *pattern).is_some()) =>
                {
                    instruction.result
                }
                _ => None,
            })
            .collect();
        ir.body.iter().all(|instruction| {
            let TraceOp::CallHelper { helper, args, .. } = &instruction.op else {
                return true;
            };
            if !helper.is_pattern_call() {
                return true;
            }
            if helper.is_gmatch_step() {
                // The lowering binds a twin to a step whose iterator this
                // trace made from a constant pattern, or else guesses one.
                return args.first().is_some_and(|it| made.contains(it)) || guessed;
            }
            if *helper != HelperId::GMATCH_NEW
                && crate::jit::prepared_helper_address(*helper).is_none()
            {
                return false;
            }
            let Some(lane) = args.get(1).and_then(|pattern| constant_lane(ir, *pattern)) else {
                return false;
            };
            // Mirrors `Lowerer::prepared_pattern`.
            root_string_slot(lane)
                .and_then(|slot| ir.root_constants.get(slot))
                .is_some_and(|bytes| find_uses_prepared_pattern(*helper, bytes, args.len()))
        })
    };
    calls(ir) && ir.sides.iter().all(|side| calls(&side.ir))
}

/// Whether `ir` or one of its sides calls `string.gsub`. Such a call spends its
/// time rewriting the subject in the helper, so a compiled loop of it saves
/// the interpreter's dispatch only, which the entries cost back.
pub(crate) fn calls_gsub(ir: &TraceIr) -> bool {
    let has = |ir: &TraceIr| {
        ir.body.iter().any(|instruction| {
            matches!(
                &instruction.op,
                TraceOp::CallHelper { helper, .. } if *helper == HelperId::STRING_GSUB
            )
        })
    };
    has(ir) || ir.sides.iter().any(|side| has(&side.ir))
}

/// Mismatching entries in a row, with no matching one between, after which a
/// trace counts as unproductive. Sized to let a few mismatching calls, each
/// mismatching at every backedge, pass between two matching ones.
/// Consecutive entries that fail the pinned callee's identity guard at once
/// retire a trace: it was specialised on a callee the site no longer calls.
pub(crate) const TRACE_CALLEE_STREAK_LIMIT: u8 = 4;

pub(crate) const TRACE_IDLE_MISMATCH_LIMIT: u32 = 4 * TRACE_PRODUCTIVITY_WINDOW;
/// How many mismatches that the frame's register tags refused before any
/// input was marshalled weigh as much toward [`TRACE_IDLE_MISMATCH_LIMIT`] as
/// one that cost a whole entry. On the Mineclonia emerge (`OXIMOND_JIT_SYNC`)
/// the region nests of `shape.lua` find their outer `*_dom` locals nil on
/// some calls and numbers on others for long stretches; a version that
/// retired after 128 of them threw away a tree that ran six iterations per
/// entry. Weights of 4, 8, 16, 64 and never retiring gave 147.9, 147.4,
/// 146.1, 146.2 and 146.3 G against 149.7 G at one.
pub(crate) const REFUSED_MISMATCH_WEIGHT: u32 = 16;
/// Entries in a row that match none of a header's versions before the policy
/// records another version for the types it sees now.
pub(crate) const TRACE_VERSION_MISMATCH_STREAK: u8 = 4;
/// Calls of [`HelperId::SET_STRING`] in one run, every one of them an
/// overwrite of a scalar of the stored type, after which the policy records
/// another version. The trace was recorded while the table was still
/// filling, when the key was absent; a version recorded now stores in place
/// (`HelperId::SET_PRESENT`), which native code does without a call.
pub(crate) const TRACE_OVERWRITE_VERSION_CALLS: u64 = 256;
/// The most iterations one entry of a trace that watches for that version
/// (see [`OverwriteVersion::Watching`]) runs before it polls. The count above
/// is judged when an entry ends, and a loop that runs for a million
/// iterations would otherwise spend its whole first entry on the slow
/// helper. The first entry holds the inserts that filled the table, so the
/// one after it is the first that can qualify.
pub(crate) const TRACE_OVERWRITE_WATCH_FUEL: i64 = 1_024;
/// Polls of a watching trace that follow each other with no rest, so a fill
/// of a few hundred keys is out of the way within a few thousand iterations.
pub(crate) const TRACE_OVERWRITE_FREE_POLLS: u8 = 8;
/// The most doublings of the rest between polls of a watching trace. After
/// a poll that stored through the helper without qualifying (some store was
/// a real insert, or too few were made) the trace runs `1 << polls` entries
/// with its full budget before it polls again, so a loop that only inserts
/// new keys polls rarely while a table that was still filling gets another
/// look once it has stopped.
pub(crate) const TRACE_OVERWRITE_MAX_REST_SHIFT: u8 = 16;
/// Consults a header waits after the cache or broker had no room for its
/// trace, before it records again.
pub(crate) const COMPILE_REFUSAL_COOLDOWN: u32 = 1_024;
/// Consults a header waits when only the broker's queue was full. Slots free
/// as soon as one compile finishes, so this is one hot period, not a long
/// back-off.
pub(crate) const QUEUE_FULL_COOLDOWN: u32 = 64;
/// Recording starts skipped after the broker refused a request for lack of
/// room, unless a compile finishes first (see
/// `TracePolicy::broker_full_skips`).
pub(crate) const BROKER_FULL_SKIPS: u32 = 256;

/// Names a guard a side trace can continue, stably across recompiles of the
/// tree, which renumber guards and members: the side trace whose guard it is
/// (`None` for the root) as an index into [`TraceIr::sides`], and the guard's
/// [`oximond::jit_ir::trace::GuardInfo::origin`] in that trace.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SideKey {
    pub(crate) parent: Option<u32>,
    pub(crate) origin: u32,
}

/// The [`SideKey::parent`] of a loop side recorded right after the side it
/// continues, before that side joined the tree: the tree resolves it to the
/// side queued just before it (see `InstalledTrace::take_queued_tree`).
pub(crate) const PREVIOUS_SIDE: u32 = u32::MAX;

/// A guard that failed another [`SIDE_EXIT_THRESHOLD`] times, with the state
/// its exit restored: where the interpreter resumes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SideCandidate {
    pub(crate) index: u32,
    pub(crate) serial: u64,
    pub(crate) key: SideKey,
    pub(crate) pc: usize,
    pub(crate) last: Option<usize>,
    /// For a function trace: the activation the exit left from, whose frame
    /// is that many frames above the entered one.
    pub(crate) activation: usize,
    /// For an exit inside inlined calls: how many frames it pushed above the
    /// root frame. `pc` and `last` are then the innermost frame's.
    pub(crate) inner: usize,
}

/// What a trace's tree is doing about side traces.
#[derive(Debug, Default)]
struct SideBook {
    /// Side traces the tree may hold (`maxside`), from the policy at install.
    max_sides: usize,
    /// Guest instructions of the sides recorded for the tree that start
    /// inside inlined frames (see [`MAX_SEEDED_GUEST_INSTRUCTIONS`]).
    seeded_guest: u32,
    /// The guard being recorded now, if a recording of this trace is running.
    recording: Option<SideKey>,
    /// The guards whose sides the tree compile in flight holds.
    compiling: Vec<SideKey>,
    /// Sides recorded while a compile was in flight. They join the next tree
    /// compile together, so a loop with many arms waits for two compiles
    /// instead of one per arm.
    /// Each carries the tree version it was recorded against.
    queue: SideQueue<(SideKey, TraceSide, u32)>,
    /// A batched tree compile failed: sides now compile one at a time, so one
    /// bad side cannot cost the others their guards.
    singly: bool,
    /// The registers live at the loop header, for building trees.
    header_live: Vec<u16>,
    /// Recordings and rejected candidates per guard.
    tried: Vec<(SideKey, u8)>,
    /// Guards whose side ran to the return of the root frame with nowhere to
    /// go: the tree had neither run `SIDE_BEYOND_ITERATIONS` nor a caller to
    /// link into. Each waits for one of the two instead of spending attempts
    /// on a recording that ends the same way (the zero-trip arm of nbody's
    /// last pair loop, which is followed by the position update and the
    /// return).
    held_for_return: Vec<SideKey>,
    /// Productivity windows left that a failing trace may be forgiven while
    /// it waits for a side trace.
    grace: u16,
    /// The trace's iteration count at the last failure of a guard that could
    /// take a side. A guard that stopped failing stops being waited for.
    /// Sides in the source tree that the compiled code left out, because the
    /// optimizer removed or hoisted their parent guard. They take no part in
    /// execution, so they do not count against the tree's room.
    dead: usize,
    /// The enclosing loop headers each recorded side runs through (see
    /// [`InstalledTrace::covered`]), by guard, until its tree installs.
    crossed: Vec<(SideKey, Vec<usize>)>,
    /// The recompile of a baseline root whose outer side never landed (see
    /// [`InstalledTrace::stone_given_up`]).
    stone: StoneRecompile,
    /// The entry count at which the compile of the queued sides was held
    /// back for a guard that was about to ask for its side too.
    held_at: Option<u64>,
    /// The entries refused while that compile was held, which count toward
    /// its wait as entries do (see [`InstalledTrace::held_too_long`]).
    held_refused: u64,
    /// The scouting for the code beyond the nest that follows the outer side
    /// just queued (see [`BeyondScouting`]).
    beyond: Option<BeyondScouting>,
    /// Guards whose side could not be compiled apart from the tree, which
    /// compiles with the tree instead.
    no_detach: Vec<SideKey>,
    /// The recompile in flight puts the sides compiled apart inline (see
    /// [`InstalledTrace::wants_relink`]).
    relinking: bool,
    /// The recompile state the relink found, which it gives back.
    relink_prior: StoneRecompile,
    /// The side just compiled apart that landed, as its index in the tree's
    /// sides: the parent of a queued side recorded right behind it (see
    /// [`PREVIOUS_SIDE`]), which hangs from its terminal exit.
    landed: Option<usize>,
}

/// The attempts left to record the code beyond the nest right behind a queued
/// outer side, and the frame they watch.
#[derive(Debug)]
struct BeyondScouting {
    tries: u8,
    frame: ScoutFrame,
}

/// The frame a scouting watches: its call depth, the locals of it that closures
/// captured, and the coroutine it runs in.
pub(super) type ScoutFrame = (
    usize,
    Vec<(u8, CellHandle)>,
    Option<crate::runtime::ThreadHandle>,
);

/// Times a placeholder (see [`CompiledTrace::staged`]) may be entered with
/// nothing pending for it before it is dropped.
const STAGED_PATIENCE: u32 = 64;

/// How many recordings of the code beyond the nest a queued outer side may
/// spend: each guard of the side that ends an inner loop wakes the scout once
/// per outer iteration, and only the last of them leaves the nest.
pub(super) const EAGER_BEYOND_TRIES: u8 = 6;

/// Entries a tree compile is held back for a side that is about to be
/// recorded (see [`InstalledTrace::holds_for_pending`]). Only roots that other
/// loops enclose hold, and their outer sides are not compiled apart while the
/// tree is young (`SIDE_BEYOND_ITERATIONS`), for the same reason: the sides of a
/// nest come in bursts that one compile takes together. A side that returns
/// into the caller is compiled apart and does not wait.
///
/// Whole-suite counts (scale 3, tracing) do not move between 32 and 256
/// entries. At 16 `algorithm/mandelbrot` goes from 87.7M to 93.9M: the tree
/// compiles for the sides that are there and again for those that come a few
/// entries later.
const HELD_COMPILE_ENTRIES: u64 = 64;

/// Native iterations after which a tree at the optimized tier is compiled
/// again with Cranelift's optimizer (see [`OptimizationTier::Peak`]). That
/// compile costs two to three times the optimized one and makes the loop up
/// to an eighth faster, which a tree repays within about as many iterations
/// again; a loop that has run this long usually runs that much longer.
/// Which trees take the tier is `InstalledTrace::wants_peak`'s choice.
///
/// [`OptimizationTier::Peak`]: crate::jit::OptimizationTier::Peak
pub(super) const PEAK_ITERATIONS: u64 = 2_000_000;

/// Native iterations a tree must have run before an outer side that its guards
/// took (not one recorded ahead of the compile, see `TracePolicy::stage_root`)
/// may run on past the loops around the root: through the loops beside them,
/// to the frame's return (see `Recorder::exit_at`), and before such a side is
/// compiled apart from its tree (`InstalledTrace::plan_detached`).
///
/// An outer side runs once per outer iteration. In the tree it is a jump; apart
/// it costs the call, the run's loop parameters and the entry of the tree
/// again, 260 instructions or so, and its guard looks rare in a young tree
/// only if the inner loop is long: `algorithm/mandelbrot`'s failed 34 times in
/// the first 18 724 iterations and once in 25 over the run, and the sides
/// compiled apart cost it 24M instructions (87.7M to 112.2M at scale 3). A
/// young tree also has bursts of sides, which one compile of the tree takes
/// together. Sides that return into the caller and those recorded behind
/// them are not held to this: they save an entry of the tree per call.
///
/// Whole-suite counts (scale 3, tracing) do not move between 50 000 and
/// 200 000 iterations. Below that `mandelbrot` loses: 93.9M with the held
/// compile shortened to 16 entries, 112.1M at 4000 iterations.
const SIDE_BEYOND_ITERATIONS: u64 = 100_000;

/// Entries a tree must have had before a side that returns from the root
/// frame may run on in its caller (see `Recorder::return_through`), however
/// small the tree is. The side saves the entry and the exit of every call
/// (about 5000 instructions for a nest as small as `loop_in_call`'s, 8000 to
/// 9000 for nbody's) and costs the compile of the side and the sides behind
/// it, which are functions of their own beside the tree's code
/// (`InstalledTrace::plan_detached`, compiled at the baseline tier): it does
/// not follow the size of the tree, which the compile of a grown tree did
/// (this was 128 entries and 48 for every operation of the tree, about 9300
/// entries for nbody, whose grown tree cost 25M to compile and repaid after
/// 2900 calls).
///
/// Measured (instructions:u, tracing, `nbody`'s nest in a function called
/// `n` times, and a nest as small as `loop_in_call`'s): the sides cost 9.2M
/// for nbody's and 2.7M for the small nest over a run that never links, and
/// repay after about 1000 and 700 calls. A run shorter than that loses what
/// the sides cost if it makes more calls than the gate, as every threshold of
/// the tracer does (a loop compiles after 16 backedges); a longer one gains:
/// `algorithm/nbody` at scale 3 (3000 calls) is 51.7M with the old formula,
/// 43.3M at 128 entries, 42.7M at 64, 41.9M at 16 (before the sides were
/// compiled at the baseline tier, which takes 2.6M more off), and at scale 9
/// 118.8M against 60M. The whole suite stays within 1 percent of the totals
/// for any value from 8 to 128. The gate is 64 entries so that the call that
/// pays for the link is not one in the first dozens, and a term per operation
/// of the tree would cost nbody 1M per entry per operation (45.3M at 2 on top
/// of 128), more than the sides' own size says.
pub(super) const RETURN_THROUGH_ENTRIES: u64 = 64;

/// Whether a baseline-tier root has been recompiled at the policy's tier, or
/// an optimized tree at the peak tier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum StoneRecompile {
    /// Not asked for.
    #[default]
    Idle,
    /// The recompile is in flight: sides recorded meanwhile wait for it.
    InFlight,
    /// Landed or failed. Asked for once. A recompile refused for lack of
    /// room goes back to `Idle`.
    Done,
}

impl SideBook {
    /// Starts a hold at `entries` unless one is running. A new hold starts
    /// with no refused entries.
    fn hold(&mut self, entries: u64) {
        if self.held_at.is_none() {
            self.held_at = Some(entries);
            self.held_refused = 0;
        }
    }

    /// Ends the hold and forgets its refused entries, so that the next hold
    /// waits its full length. Returns whether there was one.
    fn release(&mut self) -> bool {
        self.held_refused = 0;
        self.held_at.take().is_some()
    }

    fn attempts(&self, key: SideKey) -> u8 {
        self.tried
            .iter()
            .find(|(known, _)| *known == key)
            .map_or(0, |(_, attempts)| *attempts)
    }

    /// Holds the guard until the tree can run a side on past the return.
    fn hold_for_return(&mut self, key: SideKey) {
        if !self.held_for_return.contains(&key) {
            self.held_for_return.push(key);
        }
    }

    /// Spends `attempts` of the guard's budget.
    fn spend(&mut self, key: SideKey, attempts: u8) {
        match self.tried.iter_mut().find(|(known, _)| *known == key) {
            Some((_, spent)) => *spent = spent.saturating_add(attempts),
            None => self.tried.push((key, attempts)),
        }
    }

    /// Whether a side for `key` is being recorded, compiled or waiting.
    fn has(&self, key: SideKey) -> bool {
        self.recording == Some(key)
            || self.compiling.contains(&key)
            || self
                .queue
                .queued()
                .iter()
                .any(|(queued, _, _)| *queued == key)
    }

    /// Whether a recorded side waits for its tree's compile.
    fn compiling_or_queued(&self) -> bool {
        !self.compiling.is_empty() || !self.queue.queued().is_empty()
    }

    /// Whether any side is on its way.
    fn busy(&self) -> bool {
        self.recording.is_some() || !self.compiling.is_empty() || !self.queue.queued().is_empty()
    }

    /// Sides recording, compiling or waiting, which count against the tree's
    /// room.
    fn in_progress(&self) -> usize {
        usize::from(self.recording.is_some()) + self.compiling.len() + self.queue.queued().len()
    }
}

/// A queued side that can join an installed tree as a function of its own
/// (see [`InstalledTrace::plan_detached`]).
pub(super) struct DetachPlan {
    /// The tree with the side added.
    pub(super) tree: TraceIr,
    /// The side, as an index into the tree's sides.
    pub(super) source: usize,
    /// The exit of the installed code that continues the side's guard.
    pub(super) exit: usize,
    /// Whether the side leaves where it would link back to the loop.
    pub(super) exit_style: bool,
    /// The sides that are compiled apart and leave at the loop header, with
    /// this one if it is.
    pub(super) detached: Vec<usize>,
    /// The sides that are compiled apart, with this one.
    pub(super) apart: Vec<usize>,
    /// The tree planned with `detached` and `apart`, as the compile of the
    /// side would plan it.
    pub(super) planned: Option<Arc<Vec<crate::jit_ir::trace::TreeMember>>>,
}

/// What a call of a side compiled apart costs the tree over the same side
/// inline, in instructions: the cell's load, the call and its return, the loop
/// parameters through the exit area, and the tree's entry again.
const CHILD_CALL_COST: u64 = 150;
/// The same for a side that native code calls from the tree's own exits,
/// which links back to the loop without leaving the run (see
/// `Lowerer::emit_native_dispatch`). The call is the dispatch block: the
/// cell's lookup, the three buffers saved, two counters and the call, 30
/// instructions counted in the machine code of `algorithm/nbody`'s tree. The
/// link writes the loop parameters (3 to 4 instructions each) and enters the
/// tree again (3 per input for the tag checks and the loads), about 70 for
/// `algorithm/nbody` and `algorithm/mandelbrot`: 100 in all, where a call
/// through the run costs [`CHILD_CALL_COST`].
const NATIVE_CALL_COST: u64 = 100;
/// What an exit that leaves at the loop header costs where an inline side
/// would link back: the exit's restore, the interpreter's dispatch and the
/// tree's entry, about two thousand instructions.
const LINK_EXIT_COST: u64 = 2_000;
/// The iterations per run past which a root keeps its bound arrays rather
/// than take a side that would demote them to lane reads (see
/// `InstalledTrace::demotion_costs_more`): [`LINK_EXIT_COST`] over the few
/// instructions a lane read adds, for a loop with a handful of reads.
const DEMOTE_MAX_TRIPS: u64 = 64;
/// What Cranelift spends per instruction of CLIF, in instructions
/// (`algorithm/nbody`'s tree measured 13 000).
const COMPILE_COST_PER_CLIF: u64 = 13_000;
/// What a compile of a tree by the MIR backend costs per operation of the
/// tree's trace IR, in instructions: the optimizer, the exit plan and the
/// lowering together. `algorithm/nbody`'s tree of 98 operations in four
/// members took 1.32M.
const MIR_TREE_COST_PER_OP: u64 = 14_000;
/// What a side compiled apart costs per operation of its trace IR. Such a
/// side goes to Cranelift: `algorithm/nbody`'s three took 0.62M for 16
/// operations and 1.34M for 22 at the baseline tier, and 2.19M for 18 at the
/// optimized tier. This is the low end, so a tree is compiled again only
/// where that is clearly the cheaper compile.
const APART_COST_PER_OP: u64 = 40_000;

/// Whether the MIR backend compiles a tree of `tree` operations, the sides
/// waiting among them, for less than those `waiting` operations cost
/// compiled apart.
const fn joining_costs_less(tree: u64, waiting: u64) -> bool {
    tree * MIR_TREE_COST_PER_OP <= waiting * APART_COST_PER_OP
}

#[cfg(test)]
#[test]
fn a_tree_joins_its_sides_only_while_it_is_small_beside_them() {
    // nbody's tree as its last three sides arrive: 58 operations of root
    // and 43 of sides, 56 more waiting.
    assert!(joining_costs_less(58 + 43 + 56, 56));
    // A side as long as its root, the arm of a branch at the loop's top.
    assert!(joining_costs_less(120, 60));
    // A short side of a long tree is compiled alone, and so is the second
    // side of a tree that grew.
    assert!(!joining_costs_less(400, 10));
    assert!(!joining_costs_less(58 + 600, 56));
    // Nothing waiting: nothing to join.
    assert!(!joining_costs_less(58, 0));
}
/// A guard that fails less often than once in this many iterations of the
/// tree may take a side compiled apart that links back to the loop. A call of
/// such a side costs the tree [`CHILD_CALL_COST`] instructions where an inline
/// side costs none, and the compile of the tree with the side inline costs
/// millions: the rarer the guard, the longer the calls take to add up to that.
/// A guard that fails more often gets its side in a compile of the tree.
const LINKED_RARITY: u64 = 512;
/// Iterations of a tree that its guards' counts need before they say a guard
/// is rare (see [`LINKED_RARITY`]).
const RARITY_EVIDENCE_ITERATIONS: u64 = 2_048;
/// The same for a side that leaves at the loop header where it would link
/// back, which costs [`LINK_EXIT_COST`] each time.
const EXIT_STYLE_RARITY: u64 = 16_384;

/// Whether the side at `index` of `tree`, or one it continues, returns into the
/// caller of the root frame.
fn tree_returns_through(tree: &TraceIr, index: usize) -> bool {
    let mut at = Some(index);
    while let Some(side) = at {
        let Some(held) = tree.sides.get(side) else {
            return false;
        };
        if tree_returns(held) {
            return true;
        }
        at = held.parent;
    }
    false
}

/// Whether `side` returns from the root frame into its caller.
fn tree_returns(side: &TraceSide) -> bool {
    side.ir.frames.iter().any(|info| {
        info.inlined
            .is_some_and(|call| matches!(call.entry, FrameEntry::Caller { .. }))
    })
}

/// The parts of a tree that [`grow_tree`] changes besides its sides, saved so
/// [`shrink_tree`] can take a refused side back out without a clone of the tree.
struct TreeGrowth {
    root_constants: Vec<Arc<[u8]>>,
    root_slot_count: u32,
    pinned_closures: Vec<u64>,
    cell_aliases: usize,
}

/// `tree` with `side` added. The tree's banks grow to the side's where the
/// side extends them.
fn tree_with_side(mut tree: TraceIr, side: TraceSide) -> TraceIr {
    grow_tree(&mut tree, side);
    tree
}

/// Whether a queued side that its tree refused with `error` (and, when the
/// grown tree failed to verify, `verified`) is stale rather than bad: it was
/// recorded against a tree that has grown since (`grown`: the tree's version
/// moved, or a side queued ahead of it joined in this batch), and the refusal
/// is one such growth explains: another side's window slots, bindings,
/// pins, parameter types or link values. Its guard asks again and records
/// against the tree as it is now; each such retry follows a landed side, and
/// the tree holds few, so it ends. A side refused for any other reason
/// spends its guard's attempts.
///
/// The window check of the verifier counts too. Two sides of one parent
/// recorded while a compile was in flight each extend the root's window
/// (a cell, a global, a root string) from the same tree: the first joins and
/// the second no longer extends the grown window. Mineclonia's density loop
/// (`mg_ersatz.lua:408`) lost its side for a hot guard that way, and the
/// guard's exits retired the whole tree.
fn refused_as_stale(
    error: &TreeError,
    verified: Option<&crate::jit_ir::trace::TraceVerifyError>,
    grown: bool,
) -> bool {
    let explained = matches!(
        error,
        TreeError::WindowMismatch { .. }
            | TreeError::BindingMismatch { .. }
            | TreeError::PinMismatch { .. }
            | TreeError::TypeConflict { .. }
            | TreeError::LinkValue { .. }
    ) || verified
        .is_some_and(crate::jit_ir::trace::TraceVerifyError::is_side_window);
    explained && grown
}

/// Adds `side` to `tree` in place, as [`tree_with_side`] does, and returns
/// what [`shrink_tree`] needs to undo it.
fn grow_tree(candidate: &mut TraceIr, side: TraceSide) -> TreeGrowth {
    let growth = TreeGrowth {
        root_constants: candidate.root_constants.clone(),
        root_slot_count: candidate.root_slot_count,
        pinned_closures: candidate.pinned_closures.clone(),
        cell_aliases: candidate.cell_aliases.len(),
    };
    // A side may name strings the root does not (an error value it
    // builds): the tree's bank grows to the side's, which keeps the
    // root's slots and every earlier side's.
    if side.ir.root_constants.len() > candidate.root_constants.len()
        && side
            .ir
            .root_constants
            .starts_with(&candidate.root_constants)
    {
        candidate.root_constants.clone_from(&side.ir.root_constants);
        candidate.root_slot_count = side.ir.root_slot_count;
    }
    // A side may find a callee cell that is a captured local of the
    // root frame: the tree takes the promise on, for every entry. The
    // promise may name a closure only the side pins, so the tree's
    // pins grow to the side's first (they extend the tree's, or the
    // tree refuses the side as a pin mismatch).
    if !side.ir.cell_aliases.is_empty()
        && side.ir.pinned_closures.len() > candidate.pinned_closures.len()
        && side
            .ir
            .pinned_closures
            .starts_with(&candidate.pinned_closures)
    {
        candidate
            .pinned_closures
            .clone_from(&side.ir.pinned_closures);
    }
    for alias in &side.ir.cell_aliases {
        if !candidate.cell_aliases.contains(alias) {
            candidate.cell_aliases.push(*alias);
        }
    }
    candidate.sides.push(side);
    growth
}

/// Takes back the side the last [`grow_tree`] added to `tree`, and what it
/// changed besides its sides.
fn shrink_tree(tree: &mut TraceIr, growth: TreeGrowth) {
    tree.sides.pop();
    tree.root_constants = growth.root_constants;
    tree.root_slot_count = growth.root_slot_count;
    tree.pinned_closures = growth.pinned_closures;
    tree.cell_aliases.truncate(growth.cell_aliases);
}

/// The depth of a side trace whose parent is `parent` (the root is 0): the
/// number of sides on the chain from the root to it, itself included.
fn side_depth(ir: &TraceIr, parent: Option<u32>) -> u32 {
    let mut depth = 1;
    let mut at = parent;
    while let Some(side) = at {
        depth += 1;
        at = ir
            .sides
            .get(side as usize)
            .and_then(|side| side.parent)
            .map(|parent| u32::try_from(parent).unwrap_or(0));
    }
    depth
}

/// The guards of `compiled` a side trace could continue: a branch guard of the
/// loop body that resumes in the root frame and has no side trace yet, or
/// (with `encloses`, when other loops contain the root's) one that ends the
/// loop. By exit index.
///
/// `chunk` is the root's chunk, whose instruction at an array guard's site
/// tells a table read from an iterator step: only the guard of a `t[k]` read
/// takes a side. `None` (the chunk is gone) offers no array guard.
#[cfg(test)]
pub(crate) fn side_keys_of(
    compiled: &CompiledTrace,
    restores: &[ExitRestore],
    ir: &TraceIr,
    encloses: bool,
    chunk: Option<&Chunk>,
) -> Box<[Option<SideKey>]> {
    side_exits_of(compiled, restores, ir, encloses, false, chunk).0
}

/// [`side_keys_of`] and, per exit, whether a side trace was refused only
/// because the exit builds a virtual table. A side starts from the registers
/// the exit would restore, and a virtual table exists nowhere else, so such an
/// exit gets no key. The policy counts the refusal when the guard has failed
/// [`SIDE_EXIT_THRESHOLD`] times, the point where a side would have started.
/// `stone` says the trace is a stepping stone (see
/// [`InstalledTrace::keep_stone`]), whose `ipairs` end takes an outer side.
pub(crate) fn side_exits_of(
    compiled: &CompiledTrace,
    restores: &[ExitRestore],
    ir: &TraceIr,
    encloses: bool,
    stone: bool,
    chunk: Option<&Chunk>,
) -> (Box<[Option<SideKey>]>, Box<[bool]>) {
    // A loop through tail calls takes no side trace: a side would start
    // with the root frame's tail call state, which side recordings do not
    // model.
    if ir.loop_tail.is_some() {
        let none = compiled.exits().iter().map(|_| None).collect();
        let refused = compiled.exits().iter().map(|_| false).collect();
        return (none, refused);
    }
    let attached: Vec<SideKey> = ir
        .sides
        .iter()
        .enumerate()
        .filter(|(index, _)| {
            compiled
                .side_members()
                .get(*index)
                .copied()
                .flatten()
                .is_some()
        })
        .map(|(_, side)| SideKey {
            parent: side.parent.and_then(|parent| u32::try_from(parent).ok()),
            origin: side.guard,
        })
        .collect();
    let steps_next = ir
        .pinned_closures
        .contains(&crate::jit_ir::trace::pinned_builtin_identity(
            crate::jit_ir::trace::PINNED_BUILTIN_NEXT,
        ));
    let candidates: Vec<Option<SideKey>> = compiled
        .exits()
        .iter()
        .zip(restores)
        .map(|(descriptor, restore)| {
            // A guard that ends the loop can take a side too when loops
            // enclose it: the side runs on through them and links back to
            // this loop's header (an outer side).
            let continues = match descriptor.class {
                TraceExitClass::GuardFail => true,
                // A guard that fails after stores of the iteration committed
                // (the loop body writes a table, then the iterator's branch
                // goes the other way) restores the state at the guard with
                // those stores done, and the interpreter carries on from
                // there. That is where a side starts, as for a guard that
                // fails before any store. A step of an unrolled `gmatch`
                // loop is one: it leaves after the steps before it committed
                // their cursors.
                TraceExitClass::CommittedEffect => matches!(
                    descriptor.guard,
                    Some(
                        TraceGuardClass::Host
                            | TraceGuardClass::Branch
                            | TraceGuardClass::Array
                            | TraceGuardClass::Callee
                            | TraceGuardClass::Class,
                    )
                ),
                TraceExitClass::LoopComplete => encloses,
                _ => false,
            };
            // A table access that leaves (a nil hole where the recording
            // read a value, a slot past the end) is the loop taking another
            // path as much as a branch is: the side records the access as
            // the interpreter now finds it.
            let guard_continues = match descriptor.guard {
                Some(TraceGuardClass::Branch) => true,
                Some(TraceGuardClass::Array) => matches!(
                    descriptor.class,
                    TraceExitClass::GuardFail | TraceExitClass::CommittedEffect
                ),
                // A `find` or `match` that went the other way (a match where
                // the recording found none, or captures of another shape).
                Some(TraceGuardClass::Host) => matches!(
                    descriptor.class,
                    TraceExitClass::GuardFail | TraceExitClass::CommittedEffect
                ),
                // A call site that rotates through callees: the side starts
                // at the call and pins the callee the interpreter now finds.
                // An object of another class likewise: the side starts at the
                // access and pins the class the interpreter now finds.
                Some(TraceGuardClass::Callee | TraceGuardClass::Class) => matches!(
                    descriptor.class,
                    TraceExitClass::GuardFail | TraceExitClass::CommittedEffect
                ),
                // The type guard of an input of either type (see
                // `Recorder::with_lazy`): the side takes the other type.
                // The type guards of loaded values take none.
                Some(TraceGuardClass::Type) => {
                    descriptor.class == TraceExitClass::GuardFail
                        && descriptor.origin.is_some_and(|origin| {
                            let member = match descriptor.member {
                                0 => Some(ir),
                                member => ir.sides.get(member - 1).map(|side| &*side.ir),
                            };
                            member.is_some_and(|member| member.guards_untyped_input(origin))
                        })
                }
                _ => false,
            };
            // The terminal exit of a side that ended in front of a numeric for
            // loop takes the loop as a side of its own.
            if terminal_enters_loop(compiled, descriptor, chunk)
                && ((encloses && continues) || ends_in_front_of_loop(ir))
            {
                // A loop side starts in the root frame, or in the frames of
                // the calls whose loop the terminal exit stands in front of
                // (see "Loops in calls"), when a side can rebuild them.
                if !restore.inner.is_empty() && !restore.seedable {
                    return None;
                }
                let parent = if descriptor.member == 0 {
                    None
                } else {
                    let source = compiled
                        .side_members()
                        .iter()
                        .position(|member| *member == Some(descriptor.member))?;
                    Some(u32::try_from(source).ok()?)
                };
                let key = SideKey {
                    parent,
                    origin: crate::jit_ir::trace::TERMINAL_GUARD,
                };
                return (!attached.contains(&key)).then_some(key);
            }
            // An exit inside inlined calls takes a side that starts there,
            // when the calls are ones a side can rebuild.
            if !continues || !guard_continues || !(restore.inner.is_empty() || restore.seedable) {
                return None;
            }
            let origin = descriptor.origin?;
            // A guard hoisted into the preheader fails before the loop runs.
            let member = compiled.lowered_member(descriptor.member);
            let preheader = member
                .preheader
                .iter()
                .filter(|instruction| instruction.op.guard().is_some())
                .count();
            let position = member
                .guards
                .iter()
                .position(|info| info.origin == origin)?;
            if position < preheader {
                return None;
            }
            // Of the table accesses only an array or hash read continues natively,
            // and only one of a `t[k]` instruction. The read of an `ipairs`
            // step leaves at the nil that ends the array, once per run of
            // the loop: a side would record the end of the iteration, which
            // aborts, and wastes the attempts. The read of a `next` step
            // leaves at the end of the array part, where a side records the
            // step into the hash part, unless the tables it ran over had none
            // (`array_steps_plain`): the end of the array part is then the
            // end of the traversal, once per run of the loop as for `ipairs`.
            // A stepping stone's `ipairs` end is where its outer side starts:
            // the side records the step that finds the end, leaves the loop
            // and runs on through the loops around it, as at the exit test of
            // a `while` loop. A long `ipairs` loop in a nest keeps no side
            // there, as before. A loop side of a root that ends in front of
            // it (see "Outer roots") is that loop: its end is where the rest
            // of the root's iteration starts, so it takes a side too.
            if descriptor.guard == Some(TraceGuardClass::Array) {
                let reads = member.body.iter().any(|instruction| {
                    matches!(instruction.op, TraceOp::ArrayLoad { guard, .. } | TraceOp::LaneArrayLoad { guard, .. } | TraceOp::NumberLoad { guard, .. } | TraceOp::ArrayNext { guard, .. } | TraceOp::HashLoad { guard, .. } if guard.index() == position)
                        || matches!(instruction.op, TraceOp::TableFieldLoad { guard, .. } if guard.index() == position)
                });
                let site = member.guards[position].site;
                // A read in an inlined call leaves through the snapshot
                // before the call (an exit that rebuilds frames has taken
                // no side already), and the side starts at the call.
                let indexes = site.frame != FrameId(0)
                    || chunk
                        .and_then(|chunk| chunk.code.get(site.pc))
                        .is_some_and(|instruction| {
                            matches!(
                                instruction.opcode(),
                                Ok(
                                    crate::bytecode::OpCode::GetTable
                                        | crate::bytecode::OpCode::GetField
                                        | crate::bytecode::OpCode::GetTabUp
                                )
                            ) || (instruction.opcode()
                                == Ok(crate::bytecode::OpCode::ForInCall)
                                && ((stone && encloses && !steps_next)
                                    || (steps_next && !ir.array_steps_plain)
                                    || (descriptor.member > 0
                                        && ends_in_front_of_loop(ir)
                                        && !steps_next)))
                        });
                if !reads || !indexes {
                    return None;
                }
            }
            // The end of an unrolled `ipairs` loop over a table met by lane
            // (`TABLE_ABSENT_INDEX`) fails when the list grew: the side
            // records the longer list, as a side does at a `while` test.
            if descriptor.guard == Some(TraceGuardClass::Host) {
                let searches = member.body.iter().any(|instruction| {
                    matches!(instruction.op, TraceOp::CallHelper { helper, guard, .. }
                        if guard.index() == position
                            && helper.miss_takes_side())
                });
                if !searches {
                    return None;
                }
            }
            let parent = if descriptor.member == 0 {
                None
            } else {
                let source = compiled
                    .side_members()
                    .iter()
                    .position(|member| *member == Some(descriptor.member))?;
                Some(u32::try_from(source).ok()?)
            };
            let key = SideKey { parent, origin };
            (!attached.contains(&key)).then_some(key)
        })
        .collect();
    // A side starts from the state its exit restores. The virtual tables of
    // that state are built in the run when the guard enters the side (see
    // `ExitRestore::bridges_to_side`); an exit that cannot do that gets no
    // side.
    let carries = |restore: &ExitRestore| restore.virtuals.is_empty() || restore.bridges_to_side();
    let refused = candidates
        .iter()
        .zip(restores)
        .map(|(key, restore)| key.is_some() && !carries(restore))
        .collect();
    let keys = candidates
        .into_iter()
        .zip(restores)
        .map(|(key, restore)| key.filter(|_| carries(restore)))
        .collect();
    (keys, refused)
}

/// Iterations the outermost loop around a root may have left when the root's
/// first outer side closes for the code beyond that loop to be recorded in the
/// same compile: the interpreter finishes the loop first, recording nothing,
/// and a loop with many iterations would keep it waiting.
pub(super) const EAGER_BEYOND_OUTER_TRIPS: f64 = 8.0;
/// Instructions the interpreter may run while a recording waits for the pc it
/// starts at (see [`EAGER_BEYOND_OUTER_TRIPS`]).
pub(super) const EAGER_BEYOND_WAIT: u32 = 1024;

/// The guards of an outer side whose failure edges leave the outermost loop
/// around the root's (or reach its latch, which ends it): the places the code
/// beyond the nest continues from.
pub(super) struct BeyondAnchor {
    /// Every guard of the side the interpreter may fail, those that leave the
    /// nest marked.
    pub(super) candidates: Vec<WakeCandidate>,
    /// The outermost loop.
    pub(super) outer: nest::LoopSpan,
    /// The root's closing branch.
    pub(super) latch: usize,
}

/// The guards of `side`, an outer side of the tree `root` (loop at `header`),
/// that fail when the outermost loop around the root ends. `None` when it has
/// none.
pub(super) fn beyond_anchor(
    chunk: &Chunk,
    root: &TraceIr,
    header: usize,
    side: &TraceIr,
) -> Option<BeyondAnchor> {
    if root.counted || root.function.is_some() || root.loop_tail.is_some() {
        return None;
    }
    let latch = root
        .poll_snapshot
        .and_then(|poll| match root.snapshots[poll.index()].phase {
            SnapshotPhase::After { pc, .. } => Some(pc),
            SnapshotPhase::Before { .. } => None,
        })?;
    let outer = *nest::enclosing_loops(chunk, nest::LoopSpan { header, latch }).last()?;
    let candidates: Vec<WakeCandidate> = side
        .guards
        .iter()
        .filter_map(|info| {
            let snapshot = &side.snapshots[info.snapshot.index()];
            match snapshot.phase {
                SnapshotPhase::After { pc, successor } if snapshot.frames.len() == 1 => {
                    Some(WakeCandidate {
                        guard_pc: pc,
                        successor,
                        origin: info.origin,
                        beyond: successor == outer.latch || successor == outer.latch + 1,
                        root: false,
                    })
                }
                _ => None,
            }
        })
        .collect();
    candidates
        .iter()
        .any(|candidate| candidate.beyond)
        .then_some(BeyondAnchor {
            candidates,
            outer,
            latch,
        })
}

/// Where a side recording starts, and what it may do there.
pub(super) struct SeedPlace {
    /// The guard (or terminal exit) the side continues.
    pub(super) key: SideKey,
    /// Where the exit resumes the interpreter, and the instruction it ran
    /// last.
    pub(super) pc: usize,
    pub(super) last: Option<usize>,
    /// The root's loop: its header and its closing branch (a function
    /// trace's span is the whole chunk).
    pub(super) header: usize,
    pub(super) latch: usize,
    /// Whether loops enclose the root's.
    pub(super) encloses: bool,
    /// Whether an outer side may go beyond the nest (see
    /// [`SIDE_BEYOND_ITERATIONS`]).
    pub(super) beyond: bool,
    /// The captured locals of the frame the side enters.
    pub(super) captured: Vec<(u8, CellHandle)>,
    /// The closure the frame runs, which a function trace's side needs.
    pub(super) closure: Option<ClosureHandle>,
    /// The frame that called the frame the side enters, when a side that
    /// returns from it may run on there (see [`caller_seed_of`]).
    pub(super) caller: Option<CallerSeed>,
    /// Whether a side that reaches a `Return` of the frame ends there with
    /// an exit to the interpreter: no caller could take it over.
    pub(super) return_exit: bool,
    /// The call depth of the frame the side's root frame is: the depth of
    /// the frame the exit restored, below any frames it pushed.
    pub(super) depth: usize,
    /// For an exit inside inlined calls: the frames it pushed above the
    /// root frame, outermost first, as the interpreter holds them now. The
    /// side starts inside them (see "Sides from exits inside inlined frames"
    /// in the module docs). Empty otherwise.
    pub(super) inner: Vec<LiveFrame>,
    /// For such an exit: the root frame's registers as the exit left them.
    pub(super) root_registers: Vec<Value>,
    /// For such an exit: the root frame's cells and environment.
    pub(super) root_cells: Vec<CellHandle>,
    pub(super) root_environment: Option<TableHandle>,
    /// The registers of either type the root's site asked for (see
    /// [`InstalledTrace::lazy`]).
    pub(super) lazy: Vec<u16>,
}

/// A frame an exit pushed above the root frame, as the interpreter holds it
/// when a side recording starts there.
pub(super) struct LiveFrame {
    pub(super) closure: Option<ClosureHandle>,
    pub(super) chunk: Arc<Chunk>,
    pub(super) registers: Vec<Value>,
}

/// The inlined calls a side for `place` starts inside, from `snapshot` of
/// `parent_ir` (the guard's state) and the frames the exit pushed. The calls
/// must be plain calls of the frame below with a fixed result count, of the
/// closure the parent pinned, with registers that hold typed values or
/// string constants of the root bank. `pinned` is the side's pin list,
/// which takes on any closure the tree's planned root does not hold yet.
/// `None` when a frame does not qualify (v1: no `pcall`, varargs, tail calls,
/// metamethod frames, closures of a run, virtual tables, substrings or
/// borrowed strings above the root frame).
fn seed_frames_of(
    parent_ir: &TraceIr,
    snapshot: &crate::jit_ir::trace::Snapshot,
    place: &SeedPlace,
    pinned: &mut Vec<u64>,
    looping: bool,
) -> Option<Vec<SeedFrame>> {
    if snapshot.frames.len() != 1 + place.inner.len() || place.inner.len() > MAX_SEED_FRAMES {
        return None;
    }
    let mut frames = Vec::with_capacity(place.inner.len());
    for (position, (held, live)) in snapshot.frames[1..].iter().zip(&place.inner).enumerate() {
        let info = parent_ir.frames.get(held.frame.index())?;
        let call = info.inlined?;
        if call.entry != FrameEntry::Call
            || call.protected != crate::jit_ir::trace::Protected::No
            || call.reexec
            || call.results == 0
            || call.tail_calls != 0
            || call.tail_site.is_some()
            || live.chunk.is_vararg
        {
            return None;
        }
        // The live frame runs the closure the parent pinned, or, in a frame
        // that runs whichever closure of the prototype its call had, the
        // closure the exit pushed it with: the register after the chunk's
        // holds it, and the side reads it from the parent's state like any
        // other register of the frame. The pinned closure stays the frame's
        // lane, as in the parent: it names the prototype.
        let identity = *parent_ir.pinned_closures.get(call.closure as usize)?;
        let chunk_registers = usize::from(live.chunk.register_count);
        let closure = live.closure?;
        if (!call.dynamic && closure.native_identity() != identity)
            || held.registers.len() != usize::from(info.register_count)
            || held.registers.len() != chunk_registers + usize::from(call.dynamic)
            || live.registers.len() < chunk_registers
        {
            return None;
        }
        let lane = pinned
            .iter()
            .position(|known| *known == identity)
            .unwrap_or_else(|| {
                pinned.push(identity);
                pinned.len() - 1
            });
        let closure_value = Value::Closure(closure);
        let innermost = position + 1 == place.inner.len();
        // An exit whose innermost frame resumes at a `return` of fixed results
        // (`if v == val then return i end`): the frame reads no register but
        // those before it ends, and runs nothing that could look at one with
        // `debug.getlocal`. The chunk's liveness keeps every named local in
        // scope, the borrowed element of a scan too. A function with a
        // to-be-closed local runs its `__close` handlers in the return, while
        // the frame and its locals are still there to look at: none of its
        // registers is dead then.
        let returned = innermost
            .then(|| live.chunk.code.get(place.pc))
            .flatten()
            .filter(|instruction| {
                instruction.opcode() == Ok(crate::bytecode::OpCode::Return) && instruction.b() > 0
            })
            .filter(|_| {
                !live.chunk.code.iter().any(|instruction| {
                    instruction.opcode() == Ok(crate::bytecode::OpCode::MarkClose)
                })
            })
            .map(|instruction| {
                usize::from(instruction.a())
                    ..usize::from(instruction.a()) + usize::from(instruction.b())
            });
        let dead = |register: usize| {
            returned
                .as_ref()
                .is_some_and(|kept| !kept.contains(&register))
        };
        let registers = held
            .registers
            .iter()
            .zip(
                live.registers[..chunk_registers]
                    .iter()
                    .chain(call.dynamic.then_some(&closure_value)),
            )
            .enumerate()
            .map(|(register, (entry, value))| match entry {
                RegisterEntry::Value(held)
                    if parent_ir
                        .value_type(*held)
                        .is_some_and(|ty| !ty.is_borrowed()) =>
                {
                    Some(SeedRegister::Live(value.clone()))
                }
                RegisterEntry::Const(_) => Some(SeedRegister::Live(value.clone())),
                RegisterEntry::Root(slot) => Some(SeedRegister::Root(*slot)),
                // A loop side of an `ipairs` loop in a call reads the
                // element again at its header (`reload_ipairs_element`): the
                // borrow the parent holds there is no input.
                RegisterEntry::Value(_) if looping && innermost => Some(SeedRegister::Reloaded),
                RegisterEntry::Value(_) if register < chunk_registers && dead(register) => {
                    Some(SeedRegister::Dead)
                }
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        let mut info = call;
        info.closure = u32::try_from(lane).ok()?;
        info.parent = crate::jit_ir::trace::FrameId(u32::try_from(position).ok()?);
        frames.push(SeedFrame {
            info,
            chunk: Arc::clone(&live.chunk),
            last: if position + 1 == place.inner.len() {
                place.last
            } else {
                Some(snapshot.frames[position + 2].frame)
                    .and_then(|next| parent_ir.frames.get(next.index())?.inlined)
                    .map(|next| next.call_pc)
            },
            registers,
        });
    }
    Some(frames)
}

/// Productivity windows a tree that compiles again with its branch guards
/// in the loop is kept through (see `InstalledTrace::guard_move_grace`).
const GUARD_MOVE_GRACE_WINDOWS: u8 = 2;

/// Whether a root that left at hoisted guards `hoisted` times in `entries`
/// entries leaves there often enough, a quarter of its entries or more, to
/// compile again with its branch guards in the loop (see
/// [`InstalledTrace::keep_hoisted_guards`]).
const fn leaves_often_at_hoisted_guards(hoisted: u64, entries: u64) -> bool {
    hoisted.saturating_mul(4) >= entries
}

/// The tree a compile landed, `tree`, with the flag that keeps branch guards
/// in the loop carried over from the tree it replaces, `old`: the flag is
/// set on the installed tree (`InstalledTrace::keep_hoisted_guards`, a
/// stepping stone), and a compile that was in flight then started from the
/// tree before it. A landed tree whose code hoisted a guard all the same
/// (`hoisted`) asks for another compile with the guards kept.
fn keep_guards_of(
    old: &TraceIr,
    tree: Arc<TraceIr>,
    hoisted: bool,
    wanted: &mut bool,
) -> Arc<TraceIr> {
    if !old.keep_branch_guards || tree.keep_branch_guards {
        return tree;
    }
    *wanted |= hoisted;
    let mut ir = (*tree).clone();
    ir.keep_branch_guards = true;
    Arc::new(ir)
}

/// Notes on `trace` the registers of the root frame that an entry it turned
/// away (`inputs`, the slots the entry filled) held nil where the trace wants
/// a number or a number where it wants nil (see
/// [`InstalledTrace::entry_flips`]).
fn note_entry_flips(trace: &mut InstalledTrace, inputs: &[TraceInputSlot], failed: usize) {
    let nil_or_number = |tag: i64| {
        tag == crate::jit::TRACE_TAG_NIL
            || tag == crate::jit::TRACE_TAG_FLOAT
            || tag == crate::jit::TRACE_TAG_INTEGER
    };
    let wanted = |ty: TraceType| match ty {
        TraceType::Nil => Some(crate::jit::TRACE_TAG_NIL),
        TraceType::Float => Some(crate::jit::TRACE_TAG_FLOAT),
        TraceType::Integer => Some(crate::jit::TRACE_TAG_INTEGER),
        _ => None,
    };
    let guest = trace.register_count;
    // Which registers an entry the version turned away held another type
    // in, at the 1st, 2nd, 4th, ... mismatch of the version.
    if debug_env::trace() && (trace.mismatches + 1).is_power_of_two() {
        let differ: Vec<(u16, TraceType, i64)> = trace
            .compiled
            .entry_contract()
            .iter()
            .zip(inputs)
            .filter(|((_, ty), slot)| {
                *ty != TraceType::Any && wanted(*ty).is_some_and(|tag| tag != slot.tag())
            })
            .map(|((register, ty), slot)| (*register, *ty, slot.tag()))
            .collect();
        let untyped: Vec<u16> = trace
            .compiled
            .entry_contract()
            .iter()
            .filter(|(_, ty)| *ty == TraceType::Any)
            .map(|(register, _)| *register)
            .collect();
        let at = trace
            .chunk
            .upgrade()
            .map(|chunk| debug_env::at_line(&chunk, trace.header_pc))
            .unwrap_or_default();
        // The input the entry check turned down first, of any type: a table
        // or closure input that held nil is not among `differ`.
        let first = trace
            .compiled
            .entry_contract()
            .get(failed)
            .zip(inputs.get(failed))
            .map(|((register, ty), slot)| (*register, *ty, slot.tag()));
        eprintln!(
            "jit-trace: entry {} of the trace at pc {}{at} mismatched: registers (wanted, found tag) {differ:?}, untyped {untyped:?}, site list {:?}, first failed input {first:?}",
            trace.mismatches + 1,
            trace.header_pc,
            trace.lazy
        );
    }
    for ((register, ty), slot) in trace.compiled.entry_contract().iter().zip(inputs) {
        let Some(wanted) = wanted(*ty) else {
            continue;
        };
        let found = slot.tag();
        // Only a change between nil and a number: an integer that became a
        // float is the same variable in another subtype.
        let flipped = (wanted == crate::jit::TRACE_TAG_NIL) != (found == crate::jit::TRACE_TAG_NIL);
        if usize::from(*register) < guest
            && nil_or_number(found)
            && flipped
            && !trace.entry_flips.contains(register)
        {
            trace.entry_flips.push(*register);
        }
    }
}

/// The seed of a side recording for `place` in `tree`, the root with the
/// sides it has so far, and the header and latch the recorder gets. `None`
/// when the tree cannot be planned or the guard is not in it.
///
/// `planned` is `tree` as `plan_tree` plans it (see `InstalledTrace::seed_plan`).
pub(super) fn side_seed(
    tree: &TraceIr,
    planned: &[crate::jit_ir::trace::TreeMember],
    place: &SeedPlace,
) -> Option<(SideSeed, usize, usize)> {
    let ir = tree;
    let terminal = place.key.origin == crate::jit_ir::trace::TERMINAL_GUARD;
    // The loop parameters of the tree as it is now: the side hands the
    // loop one value of the right type for each.
    let params: Vec<(u16, TraceType)> = planned[0]
        .ir
        .loop_params
        .iter()
        .filter_map(|param| Some((param.register?, param.ty)))
        .collect();
    // The guard's parent and the string constants its state holds.
    let parent_ir = match place.key.parent {
        None => ir,
        Some(side) => &*ir.sides.get(side as usize)?.ir,
    };
    // The state the guard (or the terminal exit) the side continues
    // restores.
    let snapshot = exit_snapshot_of(parent_ir, place.key.origin);
    let roots: Vec<(usize, u32)> = snapshot
        .and_then(|snapshot| snapshot.frames.first())
        .map(|frame| {
            frame
                .registers
                .iter()
                .enumerate()
                .filter_map(|(register, entry)| match entry {
                    RegisterEntry::Root(slot) => Some((register, *slot)),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default();
    let borrowed: Vec<usize> = snapshot
        .and_then(|snapshot| snapshot.frames.first())
        .map(|frame| {
            frame
                .registers
                .iter()
                .enumerate()
                .filter_map(|(register, entry)| match entry {
                    RegisterEntry::Value(value)
                        if parent_ir
                            .value_type(*value)
                            .is_some_and(TraceType::is_borrowed) =>
                    {
                        Some(register)
                    }
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default();
    // The registers that hold a table the parent met at run time (read out
    // of a field or a slot of another table, or given by a lookup call): the
    // side reads them by lane too, as the parent does, instead of binding
    // the table at entry. The parent's register is rewritten each time it
    // reads such a table, so a binding would be demoted anyway. So are the
    // tables a side that starts at an inlined call takes as its arguments
    // (see `seed_lane`).
    let lanes: Vec<usize> = if ir.function.is_some() {
        Vec::new()
    } else {
        snapshot
            .map(|snapshot| {
                (0..snapshot
                    .frames
                    .first()
                    .map_or(0, |frame| frame.registers.len()))
                    .filter(|register| seed_lane(ir, parent_ir, snapshot, *register))
                    .collect()
            })
            .unwrap_or_default()
    };
    // A counted tree counts its sides: the side continues the count of
    // the path that reached its guard.
    let counted = ir
        .counted
        .then(|| snapshot.map_or(0, |snapshot| snapshot.counted));
    let captured = &place.captured;
    let members = || std::iter::once(ir).chain(ir.sides.iter().map(|side| &*side.ir));
    let dynamic_cells = members().any(|member| {
        member
            .frames
            .iter()
            .any(|info| info.inlined.is_some_and(|call| call.dynamic))
    });
    // What the guard each side starts at commits for it beyond the writes
    // of its parent's recorded path, this side's guard included: a
    // `ForLoop` recorded as its loop ended steps the variable in its
    // snapshot alone (see `crate::jit_ir::trace::guard_writes`).
    let mut started = crate::jit_ir::trace::start_writes(ir, &ir.sides);
    if let Some(frame) = snapshot.and_then(|snapshot| snapshot.frames.first()) {
        crate::jit_ir::trace::frame_writes(parent_ir, frame, &mut started);
    }
    let committed = |register: u8| {
        let slot = (crate::jit_ir::trace::FrameId(0), u16::from(register));
        started.contains(&slot) || members().any(|member| member.written_registers.contains(&slot))
    };
    // A member holds a captured local in a register when it writes it (or
    // a guard commits a write for it), when an operation reads its entry
    // value, or when an exit restores that value (a side of a tree that
    // writes cells through closures leaves the interpreter's register alone
    // instead: it is the variable).
    let captured_busy = captured.iter().any(|(register, _)| {
        committed(*register)
            || members().any(|member| {
                member
                    .entry
                    .inputs
                    .iter()
                    .filter(|input| input.register == u16::from(*register))
                    .any(|input| {
                        let read = member
                            .preheader
                            .iter()
                            .chain(&member.body)
                            .any(|instruction| {
                                let mut read = false;
                                instruction
                                    .op
                                    .operands(|value| read |= value == input.value);
                                read
                            });
                        let restored = member.snapshots.iter().any(|snapshot| {
                            snapshot.frames.iter().any(|frame| {
                                frame.frame == crate::jit_ir::trace::FrameId(0)
                                    && frame.registers.get(usize::from(*register))
                                        == Some(&RegisterEntry::Value(input.value))
                            })
                        });
                        read || restored
                    })
            })
    });
    // Writes, either way: a write through a closure, and a member's write
    // of a captured local's register.
    let dynamic_cell_writes = members().any(|member| {
        member.preheader.iter().chain(&member.body).any(|instruction| {
            matches!(&instruction.op, TraceOp::CallHelper { helper, .. } if *helper == HelperId::CELL_SET)
        })
    });
    let captured_written = captured.iter().any(|(register, _)| committed(*register));
    // A side that starts inside the inlined calls the guard failed in
    // rebuilds them first. It does not also run on into the root frame's
    // caller (v1).
    let mut pinned = planned[0].ir.pinned_closures.clone();
    let frames = if place.inner.is_empty() {
        Vec::new()
    } else {
        if ir.function.is_some() || ir.loop_tail.is_some() {
            return None;
        }
        seed_frames_of(parent_ir, snapshot?, place, &mut pinned, terminal)?
    };
    let seeded = !frames.is_empty();
    // The loop sides among the guard's parent and its ancestors, which a side
    // that comes round to one's header rejoins.
    let loop_headers = if terminal || seeded || ir.function.is_some() {
        Vec::new()
    } else {
        ancestor_loops(ir, planned, place.key)
    };
    let seed = SideSeed {
        inner_loops: ends_in_front_of_loop(ir) && !ir.counted,
        loop_headers,
        frames,
        root_registers: place.root_registers.clone(),
        root_cells: if seeded {
            place.root_cells.clone()
        } else {
            Vec::new()
        },
        root_environment: place.root_environment.filter(|_| seeded),
        dynamic_cells,
        captured_busy,
        dynamic_cell_writes,
        captured_written,
        counted,
        start_pc: place.pc,
        last: place.last,
        externals: planned[0].ir.external.clone(),
        external_types: {
            let window = &planned[0].ir;
            let guest =
                usize::from(window.frames[0].register_count).saturating_sub(window.external.len());
            (0..window.external.len())
                .map(|position| {
                    window
                        .entry
                        .inputs
                        .iter()
                        .find(|input| usize::from(input.register) == guest + position)
                        .map(|input| input.ty)
                })
                .collect()
        },
        root_constants: ir.root_constants.clone(),
        roots,
        params,
        boxed: ir.boxed_control.map(usize::from),
        // The bindings before the plan demotes any, so the side names a
        // binding by the position the next plan gives it.
        bindings: crate::jit_ir::trace::tree_bindings(ir)
            .unwrap_or_else(|_| planned[0].ir.bindings.clone()),
        fields: planned[0].ir.fields.clone(),
        pinned,
        aliases: ir.cell_aliases.clone(),
        borrowed,
        lanes,
        lazy: if seeded || ir.function.is_some() {
            Vec::new()
        } else {
            // A register the root reads typed stays typed; one it reads as
            // either type, or not at all, may be either type in the side
            // when the parent's state leaves it as the run's entry found it.
            place
                .lazy
                .iter()
                .copied()
                .filter(|register| {
                    planned[0]
                        .ir
                        .entry
                        .inputs
                        .iter()
                        .find(|input| input.register == *register)
                        .is_none_or(|input| input.ty == TraceType::Any)
                })
                .filter(|register| {
                    snapshot
                        .and_then(|snapshot| snapshot.frames.first())
                        .and_then(|frame| frame.registers.get(usize::from(*register)))
                        == Some(&RegisterEntry::Unchanged)
                })
                .collect()
        },
        function: ir.function.as_ref().and_then(|function| {
            Some(super::recorder::FunctionSeed {
                closure: place.closure?,
                parameters: function.parameters.clone(),
                result: function.result,
                // The side's recorder works it out from the chunk.
                lanes: false,
            })
        }),
        return_exit: place.return_exit,
        caller: place.caller.clone().filter(|caller| {
            !terminal
                && !seeded
                && !ir.counted
                && ir.function.is_none()
                && ir.loop_tail.is_none()
                && tree_caller(ir, Some(&Weak::new()), None).is_none_or(|held| {
                    held.call_pc == caller.call_pc
                        && held.result_base == caller.result_base
                        && held.results == caller.results
                })
        }),
        outer: place.encloses && !terminal,
        beyond: place.encloses && !terminal && place.beyond,
        looping: terminal,
        loop_sides: place.encloses && !terminal && !ir.counted && place.beyond,
        // A guard that fails by taking the backward branch of a loop
        // around the root's resumes at that loop's header, and its
        // last instruction is the branch.
        entered: place
            .last
            .filter(|last| {
                place.encloses
                    && !terminal
                    && !seeded
                    && place.pc <= *last
                    && place.pc < place.header
                    && *last > place.latch
            })
            .map(|_| place.pc),
    };
    // A loop side is a loop of its own: it runs from the header the
    // terminal exit resumes at to the latch that branched back to it.
    let (header, latch) = match (terminal, place.last) {
        (true, Some(last)) => (place.pc, last),
        _ => (place.header, place.latch),
    };
    Some((seed, header, latch))
}

/// Whether the root `ir` ends in front of a loop inside its own with a
/// terminal exit (see "Outer roots" in the module docs of `trace`): the loops
/// inside its span are loop sides of its tree.
pub(super) fn ends_in_front_of_loop(ir: &TraceIr) -> bool {
    ir.function.is_none()
        && ir.loop_tail.is_none()
        && matches!(
            ir.terminator,
            crate::jit_ir::trace::TraceTerminator::Exit { .. }
        )
}

/// The loop sides of `tree` among the side `parent` and its ancestors,
/// innermost first, as `planned` (the tree's plan) has them: each with its
/// header and latch, its parameters, the registers it reads from its entry
/// state, and the registers the sides below it on the way to `parent` wrote,
/// with what each guard on the way commits for the side it starts
/// ([`crate::jit_ir::trace::guard_writes`]): of the loop side's own guard, only
/// what the snapshot alone holds.
fn ancestor_loops(
    tree: &TraceIr,
    planned: &[crate::jit_ir::trace::TreeMember],
    key: SideKey,
) -> Vec<super::recorder::SeedLoop> {
    use crate::jit_ir::trace::{FrameId, SnapshotPhase, TraceTerminator, guard_writes};
    let mut found = Vec::new();
    let mut written: Vec<u16> = Vec::new();
    let (mut origin, mut seeded) = (key.origin, 0);
    let mut at = key.parent.map(|parent| parent as usize);
    while let Some(index) = at {
        let (Some(side), Some(member)) = (tree.sides.get(index), planned.get(index + 1)) else {
            break;
        };
        let mut committed = Vec::new();
        guard_writes(&member.ir, origin, seeded, &mut committed);
        committed.retain(|slot| !member.ir.written_registers.contains(slot));
        let committed: Vec<u16> = committed
            .into_iter()
            .filter(|(frame, _)| *frame == FrameId(0))
            .map(|(_, register)| register)
            .collect();
        let looping = side.guard == crate::jit_ir::trace::TERMINAL_GUARD
            && matches!(member.ir.terminator, TraceTerminator::LoopBack { .. });
        let span = member.ir.poll_snapshot.and_then(|poll| {
            match member.ir.snapshots[poll.index()].phase {
                SnapshotPhase::After { pc, successor } => Some((successor, pc)),
                SnapshotPhase::Before { .. } => None,
            }
        });
        if looping && let Some((header, latch)) = span {
            let params: Option<Vec<(u16, TraceType)>> = member
                .ir
                .loop_params
                .iter()
                .map(|param| Some((param.register?, param.ty)))
                .collect();
            if let Some(params) = params {
                let invariant = member
                    .ir
                    .entry
                    .inputs
                    .iter()
                    .map(|input| input.register)
                    .filter(|register| !params.iter().any(|(held, _)| held == register))
                    .collect();
                found.push(super::recorder::SeedLoop {
                    side: index,
                    header,
                    latch,
                    params,
                    invariant,
                    written: written.iter().chain(&committed).copied().collect(),
                });
            }
        }
        written.extend(
            member
                .ir
                .written_registers
                .iter()
                .filter(|(frame, _)| *frame == FrameId(0))
                .map(|(_, register)| *register)
                .chain(committed),
        );
        (origin, seeded) = (side.guard, side.ir.seed_frames);
        at = side.parent;
    }
    found
}

/// Whether `value` of `ir` is a table the trace met at run time: one an
/// operation of the trace read out of another table or got from a lookup
/// call, which native code holds as a lane of the run's bank.
fn lane_table_value(ir: &TraceIr, value: crate::jit_ir::trace::ValueId) -> bool {
    use crate::jit_ir::trace::{HelperId, ValueDef};
    let Some(info) = ir.values.get(value.index()) else {
        return false;
    };
    if info.ty != TraceType::Table {
        return false;
    }
    let instruction = match info.def {
        ValueDef::Instruction(index) => ir.body.get(index),
        ValueDef::Preheader(index) => ir.preheader.get(index),
        ValueDef::Input(_) | ValueDef::FrameInput(_) | ValueDef::LoopParam(_) => None,
    };
    instruction.is_some_and(|instruction| match &instruction.op {
        TraceOp::TableFieldLoad { .. }
        | TraceOp::ArrayLoad { .. }
        | TraceOp::LaneArrayLoad { .. } => true,
        TraceOp::CallHelper { helper, .. } => matches!(
            *helper,
            HelperId::TABLE_GET_TABLE
                | HelperId::TABLE_REMOVE_TABLE
                | HelperId::INDEX_HANDLER
                | HelperId::NEWINDEX_HANDLER
                | HelperId::CELL_GET_TABLE
        ),
        _ => false,
    })
}

/// Whether a side seeded from `snapshot`, an exit state of `member_ir` (a
/// member of a tree whose root is `root`), reads the table in `register` by
/// lane (see [`SideSeed::lanes`]): a table the member met at run time
/// ([`lane_table_value`]), or, in a register the root writes, an argument of
/// the call the snapshot starts over. The side then starts at an inlined
/// call, and the member wrote the argument registers before it: an `ipairs`
/// list read out of another table, or a copy of a table it holds elsewhere.
/// `plan_tree` refuses a side that binds a table in a register the root
/// writes when a read of it has no lane form (a string element).
fn seed_lane(
    root: &TraceIr,
    member_ir: &TraceIr,
    snapshot: &crate::jit_ir::trace::Snapshot,
    register: usize,
) -> bool {
    use crate::jit_ir::trace::{FrameId, SnapshotPhase, ValueDef};
    let value = match snapshot
        .frames
        .first()
        .and_then(|frame| frame.registers.get(register))
    {
        Some(RegisterEntry::Value(value)) => value,
        // A table the trace never made, which the guard has the run build on
        // its way to the side (`TraceTables::materialize_exit`): another
        // table each time, known by the lane it is given.
        Some(RegisterEntry::Virtual(_)) => return true,
        _ => return false,
    };
    if lane_table_value(member_ir, *value) {
        return true;
    }
    // Only a register the root writes on every pass needs it: an entry
    // binding of any other one holds for the whole run, and is cheaper.
    let written = u16::try_from(register)
        .is_ok_and(|register| root.written_registers.contains(&(FrameId(0), register)));
    let argument = match snapshot.phase {
        SnapshotPhase::Before { pc } => member_ir.frames.iter().any(|frame| {
            frame.inlined.is_some_and(|call| {
                call.parent == FrameId(0)
                    && call.call_pc == pc
                    && register > usize::from(call.result_base)
            })
        }),
        SnapshotPhase::After { .. } => false,
    };
    let Some(info) = member_ir.values.get(value.index()) else {
        return false;
    };
    if !written || !argument || info.ty != TraceType::Table {
        return false;
    }
    match info.def {
        // The entry table of another register, copied here.
        ValueDef::Input(input) => member_ir
            .entry
            .inputs
            .get(input)
            .is_some_and(|input| usize::from(input.register) != register),
        // A table the member's loop body put in the register.
        ValueDef::Instruction(_) => true,
        _ => false,
    }
}

/// The frame that called the frame at `frame_index`, as a side that returns
/// from the frame and runs on in the caller needs it. `None` when the
/// frame is not one a fresh call of its function would have made (a tail
/// call, varargs, captured locals, a closure it may create, a protected
/// call, open results) or the caller is not a plain Lua frame in a call.
pub(super) fn caller_seed_of(
    frames: &CallStack,
    frame_index: usize,
    captured_locals: bool,
) -> Option<CallerSeed> {
    let below = frame_index.checked_sub(1)?;
    let root = frames.at(frame_index);
    let caller = frames.at(below);
    let Some(crate::vm::ReturnTarget::Direct(target)) = root.frame.caller.as_ref() else {
        return None;
    };
    if target.results == 0
        || captured_locals
        || root.frame.tail_calls != 0
        || root.frame.has_vararg_state()
        || root.frame.extras_ref().is_some()
        || root.frame.chunk.is_vararg
        || root.frame.chunk.arg_register.is_some()
        || !root.frame.chunk.prototypes.is_empty()
    {
        return None;
    }
    let call_pc = caller.frame.last_interpreted_pc.get()?;
    let chunk = Arc::clone(&caller.frame.chunk.as_arc());
    // The call instruction is a plain one with the result window the frame
    // returns to.
    let call = chunk.code.get(call_pc)?;
    let window = match call.opcode().ok()? {
        OpCode::Call => (call.a(), call.c()),
        OpCode::CallOne => (call.a(), 1),
        _ => return None,
    };
    if window != (target.base, target.results) {
        return None;
    }
    // A caller that returns right after the call (`return not f (x)`, a
    // method that wraps the root's function) never calls the root again
    // before its own `Return`, which the recording refuses: a side that went
    // on in it would abort there every time. The side ends at the root
    // frame's return instead (see `SideSeed::return_exit`).
    if returns_after_call(&chunk, call_pc) {
        return None;
    }
    let captured = frames
        .open_in(caller.frame.base, caller.frame.top)
        .iter()
        .filter_map(|open| u8::try_from(open.slot - caller.frame.base).ok())
        .collect();
    Some(CallerSeed {
        chunk,
        closure: caller.frame.closure,
        root: root.frame.closure?,
        call_pc,
        result_base: target.base,
        results: target.results,
        registers: caller.registers.to_vec(),
        captured,
        environment: caller.frame.environment,
    })
}

/// Whether the code of `chunk` after the call at `call_pc` runs straight
/// into a `Return`: only instructions that compute and fall through come
/// first, no branch, loop or call (a tail call too may call the function
/// again). A metamethod of one of them could still call the function, but a
/// link through the return would have to run that metamethod's call first,
/// which is no link the recorder makes. This only decides whether a side
/// tries to go on in the caller, so it may err either way.
fn returns_after_call(chunk: &Chunk, call_pc: usize) -> bool {
    for instruction in chunk.code.iter().skip(call_pc + 1) {
        match instruction.opcode() {
            Ok(OpCode::Return | OpCode::ReturnOpen | OpCode::ReturnVararg | OpCode::Halt) => {
                return true;
            }
            // Instructions that only compute and fall through.
            Ok(
                OpCode::LoadNil
                | OpCode::LoadConstant
                | OpCode::LoadConstantWide
                | OpCode::Move
                | OpCode::Negate
                | OpCode::Not
                | OpCode::Add
                | OpCode::Subtract
                | OpCode::Multiply
                | OpCode::Divide
                | OpCode::Modulo
                | OpCode::Power
                | OpCode::FloorDivide
                | OpCode::AddK
                | OpCode::SubtractK
                | OpCode::MultiplyK
                | OpCode::DivideK
                | OpCode::ModuloK
                | OpCode::Equal
                | OpCode::LessThan
                | OpCode::LessEqual
                | OpCode::Length
                | OpCode::Concat
                | OpCode::BitAnd
                | OpCode::BitOr
                | OpCode::BitXor
                | OpCode::ShiftLeft
                | OpCode::ShiftRight
                | OpCode::BitNot
                | OpCode::GetField
                | OpCode::GetTable
                | OpCode::GetUpvalue
                | OpCode::GetTabUp
                | OpCode::GetGlobal
                | OpCode::GetGlobalWide,
            ) => {}
            _ => return false,
        }
    }
    false
}

/// The snapshot of the guard of `ir` with `origin`, or of its terminal exit
/// for [`crate::jit_ir::trace::TERMINAL_GUARD`].
fn exit_snapshot_of(ir: &TraceIr, origin: u32) -> Option<&crate::jit_ir::trace::Snapshot> {
    if origin == crate::jit_ir::trace::TERMINAL_GUARD {
        return match ir.terminator {
            crate::jit_ir::trace::TraceTerminator::Exit { snapshot } => {
                ir.snapshots.get(snapshot.index())
            }
            _ => None,
        };
    }
    ir.guards
        .iter()
        .find(|info| info.origin == origin)
        .map(|info| &ir.snapshots[info.snapshot.index()])
}

/// Whether `descriptor` is the terminal exit of a member that ends in front
/// of a loop (a side, or a root that ends in front of a loop inside its own):
/// the loop's closing branch has branched back, and the interpreter resumes
/// at the loop's header. A loop side continues it.
fn terminal_enters_loop(
    compiled: &CompiledTrace,
    descriptor: &crate::jit::ExitDescriptor,
    chunk: Option<&Chunk>,
) -> bool {
    if descriptor.class != TraceExitClass::LoopComplete
        || descriptor.origin.is_some()
        || descriptor.guard.is_some()
    {
        return false;
    }
    let member = compiled.lowered_member(descriptor.member);
    let crate::jit_ir::trace::TraceTerminator::Exit { snapshot } = member.terminator else {
        return false;
    };
    if snapshot != descriptor.snapshot {
        return false;
    }
    let Some(chunk) = chunk else {
        return false;
    };
    let held = &member.snapshots[snapshot.index()];
    let crate::jit_ir::trace::SnapshotPhase::After { pc, successor } = held.phase else {
        return false;
    };
    // In front of a loop in a call: the recorder ended there only at the
    // loop's backward branch.
    if held.frames.len() > 1 {
        return successor <= pc;
    }
    Recorder::loop_latch_target(chunk, pc) == Some(successor)
}

/// Whether the latch at `pc` of `chunk` loops back when the interpreter
/// runs it now, so that an exit resuming there stays in its loop: the
/// backward jump of a `while` loop always does (its test is at the
/// header), and a numeric for latch does when the registers say it has at
/// least two more trips. A generic for latch, or registers that do not
/// describe the loop, say no.
fn latch_loops_back(chunk: &Chunk, pc: usize, registers: &[Value]) -> bool {
    let Some(latch) = chunk.code.get(pc) else {
        return false;
    };
    if latch.opcode() == Ok(crate::bytecode::OpCode::Jump) {
        return true;
    }
    if latch.opcode() != Ok(crate::bytecode::OpCode::NumericForLoop) {
        return false;
    }
    let base = usize::from(latch.a());
    let number = |register: usize| match registers.get(register)? {
        #[allow(clippy::cast_precision_loss)]
        Value::Integer(value) => Some(*value as f64),
        Value::Number(value) => Some(*value),
        _ => None,
    };
    let (Some(control), Some(limit), Some(step)) =
        (number(base), number(base + 1), number(base + 2))
    else {
        return false;
    };
    step != 0.0 && (limit - control) / step >= 2.0
}

/// Which exits of `compiled` end the loop: guards that resume outside the
/// root's span.
fn loop_ends_of(compiled: &CompiledTrace) -> Box<[bool]> {
    compiled
        .exits()
        .iter()
        .map(|descriptor| descriptor.class == TraceExitClass::LoopComplete)
        .collect()
}

/// Where a trace stands on asking for a version that overwrites in place
/// (see [`TRACE_OVERWRITE_VERSION_CALLS`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OverwriteVersion {
    /// No string store goes through the helper yet; a side may add one.
    NotApplicable,
    Watching,
    Wanted,
    Asked,
}

impl OverwriteVersion {
    /// The state of a trace whose IR has `ir_calls` helper calls: it watches
    /// when one of them stores a string key through the helper.
    fn of_helper_calls(ir_calls: &[crate::jit_ir::trace::HelperId]) -> Self {
        if ir_calls.contains(&crate::jit_ir::trace::HelperId::SET_STRING) {
            Self::Watching
        } else {
            Self::NotApplicable
        }
    }
}

/// The pending tree and its plan, with the state they were made from (see
/// [`InstalledTrace::seed_plan`]).
#[derive(Debug)]
struct SeedPlan {
    ir: Arc<TraceIr>,
    queue_gen: u64,
    tree: Arc<TraceIr>,
    members: Arc<Vec<crate::jit_ir::trace::TreeMember>>,
}

/// The planned members of a tree without its candidate side, with the state
/// they were planned from.
#[derive(Debug)]
struct BasePlan {
    ir: Arc<TraceIr>,
    header_live: Vec<u16>,
    detached: Vec<usize>,
    apart: Vec<usize>,
    members: Arc<Vec<crate::jit_ir::trace::TreeMember>>,
}

/// A compiled trace and what the policy tracks about it.
#[derive(Debug)]
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct InstalledTrace {
    chunk: Weak<Chunk>,
    header_pc: usize,
    /// Kept for exit snapshots and for recompiling at another tier. Shared
    /// with the broker request that compiled it. It holds the side traces
    /// of the tree as recorded.
    ir: Arc<TraceIr>,
    compiled: CompiledTrace,
    /// The profile the trace was recorded under, which side recordings use.
    profile: Option<CompatibilityProfile>,
    /// Advances every time a recompile replaces the tree, so a result for an
    /// older tree is not installed over a newer one.
    version: u32,
    /// How many times each exit was taken, for the exits side traces could
    /// continue (see `side_keys`).
    exit_hits: Vec<u32>,
    /// How many times each exit no side trace continues was taken (a guard
    /// the optimizer hoisted in front of the loop, a callee guard past the
    /// tree's budget for sides), in the same window as `exit_hits`: for the
    /// retirement's report and [`Self::callee_exit_paths`].
    cold_hits: Vec<u32>,
    /// Exits that resumed in the interpreter, by kind and guard, over the
    /// trace's whole life: kept across recompiles, for the attribution report.
    interp_exits: super::stats::ExitMatrix,
    /// The trace's iteration count when the exits' counts last started from
    /// zero: the window `exit_hits` counts failures in.
    hits_since: u64,
    /// The trace's iteration count at each exit's last failure, parallel to
    /// `exit_hits`. A guard that stopped failing stops being waited for.
    exit_last_hit: Vec<u64>,
    /// The guard each exit belongs to, when a side trace could continue it.
    side_keys: Box<[Option<SideKey>]>,
    /// Exits a side trace would continue but for the virtual table they
    /// build. Counted once in `side_rejections` when the guard reaches the
    /// side threshold.
    virtual_refused: Box<[bool]>,
    /// For each exit, whether it ends the loop (a guard that resumes outside
    /// the root's span), which is where an outer side starts.
    loop_ends: Box<[bool]>,
    side: SideBook,
    /// The sides of the source tree that were compiled apart from it and
    /// leave where they would link back to the loop, as indexes into
    /// `ir.sides`. The tree's root was planned without widening for them.
    detached: Vec<usize>,
    /// The sides of the source tree that are compiled apart from it,
    /// whether they leave at the loop header or link back to it.
    apart: Vec<usize>,
    /// The tree without the side a detach plan adds, planned for the state
    /// it was planned in (see [`InstalledTrace::base_plan`]).
    base_plan: RefCell<Option<BasePlan>>,
    /// The plan of the pending tree (see `seed_plan`).
    seed_plan: RefCell<Option<SeedPlan>>,
    /// The tree whose compile is in flight and its members, as
    /// `take_queued_tree` planned them. When that tree is installed with
    /// nothing queued behind it, it is the pending tree and this is its
    /// plan, which `seed_plan` then need not make again.
    grown_plan: Option<(Arc<TraceIr>, Arc<Vec<crate::jit_ir::trace::TreeMember>>)>,
    /// Calls of sides compiled apart, and exits of those that leave at the
    /// loop header, since the tree was compiled (see
    /// [`InstalledTrace::wants_relink`]).
    child_calls: u64,
    /// The calls among them that native code made from the tree's own exits
    /// (see [`NATIVE_CALL_COST`]); `child_calls` holds the others.
    native_calls: u64,
    link_exits: u64,
    /// The last unit number given to a side compiled apart, which names its
    /// exits in the codes the run returns.
    units: u32,
    /// Whether other loops contain this trace's loop in its chunk, which
    /// lets the loop's exits take outer side traces.
    pub(super) encloses: bool,
    /// Whether this is the root of a `while` or generic-for loop that other
    /// loops enclose: its trip count was not known when it closed, so it may
    /// turn out to be a stepping stone at its first productivity window
    /// (see [`InstalledTrace::keep_stone`]).
    stone_candidate: bool,
    /// For such a root whose window closed below the bar: the entry count
    /// when it became a stepping stone, or when a side of its tree last
    /// landed since. `None` while it has not run short.
    stone: Option<u64>,
    /// Entries that no version took (types none of them records) since
    /// the trace was installed, and the counts of entries and of such
    /// mismatches when it became a stepping stone.
    mismatches: u64,
    stone_start: (u64, u64),
    /// The stone's code hoisted a guard into the preheader, so the root is
    /// compiled again with its guards kept in the loop (see
    /// [`InstalledTrace::begin_keep_guards`]).
    keep_guards_wanted: bool,
    /// Productivity windows still forgiven because the tree is compiling
    /// again with its branch guards in the loop (see
    /// [`InstalledTrace::keep_hoisted_guards`]): until that code lands, the
    /// guards it moves have no side keys and the window that closes below
    /// the bar would retire the tree before a side can take them.
    guard_move_grace: u8,
    /// The backedge count of the innermost loop around this one when the
    /// trace was installed, while that loop was still counting.
    outer_backedges_at_install: u32,
    /// The headers of the enclosing loops this tree's side traces run
    /// through. Their sites are `Covered` while this trace lives.
    covered: Vec<usize>,
    /// Registers in the interpreter frame. The window the snapshots describe
    /// continues with `externals`.
    register_count: usize,
    /// What the window beyond the frame's registers stands for.
    externals: Box<[ExternalSlot]>,
    /// Whether the trace names upvalue cells, which entry checks for
    /// aliasing.
    has_upvalues: bool,
    /// Whether it names two or more: with one, only a captured local of the
    /// frame can share its cell.
    many_upvalues: bool,
    /// The closures the trace specialised on (see
    /// [`TraceIr::pinned_closures`]). Each run's bank starts with them.
    pinned: Box<[u64]>,
    /// The window slots of cells that some member of the tree writes, or
    /// every cell slot when a member writes a callee's cell through the live
    /// closure ([`written_cell_slots`]). A closure called through the
    /// prototype guard may not share one of these cells (`RunClosures::held`).
    written_cells: Box<[ExternalSlot]>,
    /// The cells the trace holds one variable for under two names. Entry
    /// checks each promise.
    aliases: Box<[CellAlias]>,
    /// The cells the last entry that passed the alias checks found
    /// ([`CellsMemo`]): an entry that finds the same ones passes them too.
    /// Cleared whenever the tree's names change.
    cells_memo: Option<CellsMemo>,
    /// The most frames a snapshot of the trace adds to the root frame. Entry
    /// needs room for them under the call-depth limit.
    max_inline_depth: usize,
    /// For a tree with a side that returns from the root frame into its
    /// caller and calls it again: the caller the entry insists on.
    caller: Option<CallerEntry>,
    /// The chunk of the caller that a side recorded for this tree returns
    /// into, once one was recorded.
    caller_chunk: Option<Weak<Chunk>>,
    /// The registers of that caller a closure had captured at every
    /// recording of a side that returned into it, which the entry insists
    /// the caller has no others than (a register of the caller that the
    /// trace writes must not be a captured variable).
    caller_captured: Option<Box<[u8]>>,
    /// The string constants `Root` entries restore, by root slot.
    roots: Box<[Value]>,
    /// For each table binding, the entry input holding its table and the
    /// string key of a field binding.
    bindings: Box<[BindingPlan]>,
    /// Whether every binding is of a kind [`bind_simple`] resolves.
    simple_binds: bool,
    /// How often the interpreter entered this trace while it was a
    /// placeholder with nothing pending (see [`STAGED_PATIENCE`]).
    staged_visits: u32,
    /// Where each entry input of a plain trace comes from, in contract
    /// order; empty for a trace whose entry has more to it (a `next` loop
    /// keeps its control value in a lane of its own).
    entry_plan: Box<[EntryStep]>,
    /// The fields of run-time tables the trace reads and writes: the key and
    /// the type each row slot must hold (see [`oximond::jit::TraceTables`]).
    fields: Box<[(Value, NativeSlotKind)]>,
    /// The positions among `fields` of the metatable word of a row
    /// ([`crate::jit_ir::trace::META_FIELD_KEY`]) and of its table address
    /// word ([`crate::jit_ir::trace::TABLE_ADDRESS_KEY`]), when the trace
    /// lists them.
    meta_field: crate::runtime::NativeRowWords,
    /// The first byte of a value of each of `fields`' kinds.
    field_tags: Box<[u8]>,
    /// The rows of the tables the trace met last, kept from run to run (see
    /// `RunTables::fill_table_row`).
    row_memo: Vec<usize>,
    /// Where each of `fields` was last found in a table: the next table is
    /// likely built alike.
    field_hints: Vec<usize>,
    /// The shapes the tables the trace makes are given, by the words of the
    /// keys that name them (see `HostState::shapes`), kept from run to run
    /// so that a shape is built once.
    table_shapes: Vec<([i64; 10], std::sync::Arc<crate::runtime::TableShape>)>,
    /// How to restore the frame at each exit, by exit index.
    restores: Vec<ExitRestore>,
    /// Loop-backs completed natively over the trace's life.
    iterations: u64,
    /// Native runs that executed code over the trace's life.
    entries: u64,
    /// Whether the trace computes with integers.
    integers: bool,
    /// Loop-backs per entry the trace must average, see
    /// [`min_iterations_of`].
    min_iterations: u64,
    /// Entries and loop-backs since the last productivity check.
    window_entries: u32,
    /// Entries since the last matching run, all of which mismatched.
    idle_mismatches: u32,
    /// Entries in a row that failed a callee identity guard before running
    /// [`TRACE_MIN_ITERATIONS_PER_ENTRY`] loop-backs.
    callee_streak: u8,
    window_iterations: u64,
    /// The loop-backs the window's entries are credited with for a heavy
    /// body: a run that left inside an inlined call before its first
    /// loop-back counts none (see [`Self::note_window`]).
    window_loop_backs: u64,
    /// Iterations the trace ran above the productivity bar in earlier
    /// windows, up to [`TRACE_PRODUCTIVITY_CREDIT_WINDOWS`] windows' worth.
    /// A window below the bar spends it before the trace retires.
    productivity_credit: u64,
    /// Productivity windows the trace has passed.
    productive_windows: u32,
    /// Entries in a row, counted on the newest version of a header, that
    /// matched none of the header's versions.
    pub(super) mismatch_streak: u8,
    /// The registers of the root frame that an entry this version turned
    /// away held nil where the version wants a number, or a number where it
    /// wants nil. The next version takes them as inputs of either type (see
    /// `Recorder::with_lazy`).
    pub(super) entry_flips: Vec<u16>,
    /// The frame registers the entry contract types, with the tag it wants
    /// for each, that [`frame_mismatch`] compares before an entry marshals
    /// anything. `None` until the code's own check first turns an entry
    /// away: a version whose entries all match never pays for the scan.
    frame_checks: Option<Box<[(u16, i64)]>>,
    /// The registers of either type the site had asked for when this
    /// version was installed (see `Recorder::with_lazy`). A side takes them
    /// as inputs of type `Any` too, the ones the root never read included.
    pub(super) lazy: Vec<u16>,
    /// Whether a run's string-key stores were all overwrites in place (see
    /// [`TRACE_OVERWRITE_VERSION_CALLS`]), so the header should record a
    /// version that makes them natively. Asked once per trace.
    overwrite_version: OverwriteVersion,
    /// Polls so far that stored through the helper without qualifying; the
    /// rest after the next one doubles with each.
    overwrite_polls: u8,
    /// Nonzero when the current entry runs at the watch fuel, so its end is a poll.
    overwrite_polling: u8,
    /// Entries left that run with the full budget before the next poll.
    overwrite_rest: u32,
    /// The cache's use counter at the last entry, for eviction order.
    pub(super) last_used: u64,
    /// Identifies this trace across reuse of its table slot.
    pub(super) serial: u64,
    /// The next older version at this header.
    pub(super) next: Option<u32>,
    /// For a function trace: its result type, activation stride and callee
    /// registers.
    function: Option<InstalledFunction>,
}

/// The tree a queued side grew ([`InstalledTrace::take_queued_tree`]), with
/// its members as planned when it was grown.
type PlannedTree = (TraceIr, Arc<Vec<crate::jit_ir::trace::TreeMember>>);

/// What [`InstalledTrace::take_queued_tree`] returns: the tree (`None` if no
/// side fit), the guards it adds, and how many sides were refused for good.
type QueuedTree = (Option<PlannedTree>, Vec<SideKey>, u64);

impl InstalledTrace {
    /// Native runs, loop-backs run natively, and counted exits over the
    /// trace's life, for the attribution report.
    pub(super) fn run_counts(&self) -> (u64, u64, u64) {
        (
            self.entries,
            self.iterations,
            self.exit_hits.iter().map(|&hits| u64::from(hits)).sum(),
        )
    }

    /// Exits that resumed in the interpreter, by kind and guard.
    pub(super) const fn interpreter_exit_counts(&self) -> &super::stats::ExitMatrix {
        &self.interp_exits
    }

    pub(super) const fn tree_version(&self) -> u32 {
        self.version
    }

    /// A side recording ended without a side trace. `attempts` of the guard's
    /// budget are spent: one for a failure another try may avoid, all of them
    /// for one that will repeat (an arm the recorder cannot follow).
    /// See [`SideBook::hold_for_return`].
    pub(super) fn hold_side_for_return(&mut self, key: SideKey) {
        self.side.hold_for_return(key);
    }

    /// Whether the guard waits for the tree to run past the return.
    #[cfg(test)]
    pub(super) fn side_is_held_for_return(&self, key: SideKey) -> bool {
        self.side.held_for_return.contains(&key)
    }

    pub(super) fn end_side_recording(&mut self, key: SideKey, attempts: u8) {
        if self.side.recording == Some(key) {
            self.side.recording = None;
        }
        self.side.spend(key, attempts);
    }

    /// A loop side is recording right behind the side that was just queued.
    pub(super) fn begin_chained_recording(&mut self, key: SideKey) {
        self.side.recording = Some(key);
    }

    /// Starts the scouting for the code beyond the nest behind the queued
    /// outer side, in the frame at `depth` with these captured locals.
    pub(super) fn begin_beyond_scouting(&mut self, frame: ScoutFrame) {
        self.side.beyond = Some(BeyondScouting {
            tries: EAGER_BEYOND_TRIES,
            frame,
        });
    }

    /// Spends one attempt of the scouting: the frame it watches, or `None`
    /// when none is left (the queued sides then compile without the code
    /// beyond the nest).
    pub(super) fn take_beyond_try(&mut self) -> Option<ScoutFrame> {
        let scouting = self.side.beyond.as_mut()?;
        scouting.tries = scouting.tries.checked_sub(1)?;
        Some(scouting.frame.clone())
    }

    /// The recording behind the queued side is over.
    pub(super) fn end_chained_recording(&mut self, key: SideKey) {
        if self.side.recording == Some(key) {
            self.side.recording = None;
        }
    }

    /// The last side queued, and the scouting's frame, for another attempt.
    pub(super) fn last_queued(&self) -> Option<&TraceSide> {
        self.side.queue.queued().last().map(|(_, side, _)| side)
    }

    /// Whether the compile of the queued sides of a root that other loops
    /// enclose waits for one more side: a guard that has failed for half
    /// its threshold and is still failing will start a recording within a few
    /// entries. One compile for both sides costs less than two, and the
    /// second would otherwise start only once the first has landed, with the
    /// entries in between. The wait is bounded ([`HELD_COMPILE_ENTRIES`]).
    pub(super) fn holds_for_pending(&mut self) -> bool {
        if !self.encloses || self.side.queue.queued().is_empty() {
            return false;
        }
        let pending = (0..self.exit_hits.len()).any(|exit| {
            self.exit_hits[exit] % SIDE_EXIT_THRESHOLD >= SIDE_EXIT_THRESHOLD / 2
                && self.exit_last_hit[exit] + SIDE_QUIET_ITERATIONS >= self.iterations
                && self.side_keys[exit].is_some_and(|key| !self.side.has(key))
                && self.side_possible(exit)
        });
        if pending {
            self.side.hold(self.entries);
        }
        pending
    }

    /// Whether a held compile has waited [`HELD_COMPILE_ENTRIES`] entries.
    pub(super) fn held_too_long(&mut self) -> bool {
        let over = self
            .side
            .held_at
            .is_some_and(|at| self.entries + self.side.held_refused >= at + HELD_COMPILE_ENTRIES);
        if over {
            self.side.release();
        }
        over
    }

    /// Ends the hold: the recording it waited for is over.
    pub(super) fn release_held(&mut self) -> bool {
        self.side.release()
    }

    /// An outer side recording ran into the last iteration of the loop around
    /// the root, which has no inner trips to link back through. The guard
    /// asks again at its next failure instead of a threshold's worth later:
    /// when the root's exits cycle with a period that divides the threshold
    /// (an inner loop of four, three, two, one and no trips), every ask would
    /// land on the same outer iteration.
    pub(super) fn retry_side_soon(&mut self, key: SideKey) {
        for (hits, exit_key) in self.exit_hits.iter_mut().zip(&self.side_keys) {
            if *exit_key == Some(key) {
                *hits = *hits / SIDE_EXIT_THRESHOLD * SIDE_EXIT_THRESHOLD + SIDE_EXIT_THRESHOLD - 1;
            }
        }
    }

    /// A side recording closed: the side waits in the queue, and the guard's
    /// recording is over.
    /// A side that starts inside inlined frames closed with `guest`
    /// instructions: it counts against the tree's budget for such sides.
    pub(super) fn note_seeded_side(&mut self, guest: u32) {
        self.side.seeded_guest = self.side.seeded_guest.saturating_add(guest);
    }

    /// Whether the tree may still record sides that start inside inlined
    /// frames (see [`MAX_SEEDED_GUEST_INSTRUCTIONS`]).
    fn seeds_left(&self) -> bool {
        self.side.seeded_guest < MAX_SEEDED_GUEST_INSTRUCTIONS
    }

    pub(super) fn side_recorded(
        &mut self,
        key: SideKey,
        side: TraceSide,
        live: Vec<u16>,
        crossed: Vec<usize>,
        returned_into: Option<(&Arc<Chunk>, &[u8])>,
    ) {
        let _icount = crate::jit::icount::scope("side-recorded");
        if let Some((chunk, captured)) = returned_into {
            self.caller_chunk = Some(Arc::downgrade(chunk));
            self.caller_captured = Some(match self.caller_captured.take() {
                Some(held) => held
                    .iter()
                    .copied()
                    .filter(|register| captured.contains(register))
                    .collect(),
                None => captured.into(),
            });
        }
        if self.side.recording == Some(key) {
            self.side.recording = None;
        }
        // A side that does not follow another starts a chain of its own.
        if key.parent != Some(PREVIOUS_SIDE) {
            self.side.landed = None;
        }
        self.side.crossed.retain(|(known, _)| *known != key);
        self.side.crossed.push((key, crossed));
        self.side.header_live = live;
        let version = self.version;
        self.side.queue.queue_push((key, side, version));
    }

    /// The queued sides added to the installed tree, one at a time, keeping
    /// those that fit. Returns `None` while a compile is in flight or nothing
    /// is queued. Otherwise: the tree (`None` if no side fit), the guards it
    /// adds, and how many sides were refused for good.
    ///
    /// The tree comes with its members as planned when it was grown, which
    /// the compile takes instead of planning it again.
    pub(super) fn take_queued_tree(&mut self) -> Option<QueuedTree> {
        let _icount = crate::jit::icount::scope("queue-tree");
        if !self.side.compiling.is_empty()
            || self.side.queue.queued().is_empty()
            || self.side.stone == StoneRecompile::InFlight
        {
            return None;
        }
        self.side.release();
        self.side.landed = None;
        let mut queued = self.side.queue.queue_take();
        if self.side.singly && queued.len() > 1 {
            // The rest wait for this one to land.
            let rest = queued.split_off(1);
            self.side.queue.queue_splice_front(rest);
        }
        let mut tree = (*self.ir).clone();
        tree.header_live.clone_from(&self.side.header_live);
        let mut keys = Vec::new();
        let mut refused = 0;
        // The side the entry before this one became in the tree, if it fit.
        let mut previous: Option<usize> = None;
        // The plan of the tree as it stands: the last side that fit.
        let mut last_plan: Option<Vec<crate::jit_ir::trace::TreeMember>> = None;
        for (key, mut side, recorded) in queued {
            let before = previous.take();
            if key.parent == Some(PREVIOUS_SIDE) {
                // A side recorded behind the side it continues goes where
                // that side went, or nowhere. One queued again after a failed
                // batch has its parent already, which the tree holds.
                match (before, side.parent) {
                    (Some(parent), _) => side.parent = Some(parent),
                    (None, Some(parent)) if parent < tree.sides.len() => {}
                    _ => continue,
                }
            }
            // The tree grows in place and shrinks back when the side is
            // refused. `verify_grown_tree` checks the side and what its growth
            // touched, not the root again.
            let growth = grow_tree(&mut tree, side);
            let added = tree.sides.len() - 1;
            let verified = crate::jit_ir::trace::verify_grown_tree(&tree, added);
            if cfg!(debug_assertions)
                && verified.is_ok()
                && let Err(error) = tree.verify()
            {
                panic!("a side that fits its tree leaves the tree invalid: {error}");
            }
            let planned =
                if tree.sides.len() <= self.side.max_sides + self.side.dead && verified.is_ok() {
                    crate::jit_ir::trace::plan_tree(&tree)
                } else {
                    Err(TreeError::NotSource)
                };
            if debug_env::trace()
                && let Err(error) = &planned
            {
                eprintln!(
                    "jit-trace: side for {key:?} does not fit the tree: {error} ({verified:?})"
                );
            }
            let costly = planned
                .as_ref()
                .is_ok_and(|members| self.demotion_costs_more(members));
            if costly && debug_env::trace() {
                eprintln!(
                    "jit-trace: side for {key:?} would demote bindings of a root whose runs are long"
                );
            }
            match planned {
                Ok(_) if costly => {
                    shrink_tree(&mut tree, growth);
                    self.side.spend(key, MAX_SIDE_ATTEMPTS);
                    refused += 1;
                }
                Ok(members) => {
                    keys.push(key);
                    previous = tree.sides.len().checked_sub(1);
                    last_plan = Some(members);
                }
                Err(error) => {
                    shrink_tree(&mut tree, growth);
                    let stale = refused_as_stale(
                        &error,
                        verified.as_ref().err(),
                        recorded != self.version || !keys.is_empty(),
                    );
                    if !stale {
                        self.side.spend(key, MAX_SIDE_ATTEMPTS);
                        refused += 1;
                    }
                }
            }
        }
        let tree = match last_plan {
            Some(plan) if !keys.is_empty() => Some((tree, Arc::new(plan))),
            _ => None,
        };
        Some((tree, keys, refused))
    }

    /// Whether the plan `members` demotes more table bindings to lane reads
    /// than the installed tree does, for a root whose runs average more than
    /// [`DEMOTE_MAX_TRIPS`] iterations. A lane array read costs a row load,
    /// two table words and a branch where a bound read costs nothing extra,
    /// and it loses the range checks of bound keys: in a long run that costs
    /// more than the exit and entry per outer iteration the side saves
    /// (`opt/range_offset`, whose outer side swaps the two arrays it binds,
    /// ran 2.2 times the instructions).
    fn demotion_costs_more(&self, members: &[crate::jit_ir::trace::TreeMember]) -> bool {
        members
            .first()
            .is_some_and(|root| root.demoted > self.compiled.demoted_bindings())
            && self.iterations > DEMOTE_MAX_TRIPS * self.entries.max(1)
    }

    /// Whether lowered member `member` is a side compiled apart that leaves at
    /// the loop header where it would link back.
    fn leaves_at_header(&self, member: usize) -> bool {
        self.detached
            .iter()
            .any(|source| self.compiled.side_members().get(*source) == Some(&Some(member)))
    }

    /// Whether the sides compiled apart have cost more than compiling the tree
    /// again with them inline would: a call of one costs the tree
    /// [`CHILD_CALL_COST`] instructions, an exit that leaves at the loop header
    /// [`LINK_EXIT_COST`], and the compile [`COMPILE_COST_PER_CLIF`] per
    /// instruction handed to Cranelift. The tree must be settled: nothing
    /// recording, compiling or queued, and no recompile in flight.
    pub(super) fn wants_relink(&self) -> bool {
        self.compiled.attached_sides() > 0
            && self.side.stone != StoneRecompile::InFlight
            && !self.side.busy()
            && !self.is_helper_bound()
            && self.child_calls * CHILD_CALL_COST
                + self.native_calls * NATIVE_CALL_COST
                + self.link_exits * LINK_EXIT_COST
                >= self.compiled.clif_insts_total() as u64 * COMPILE_COST_PER_CLIF
    }

    /// The recompile that puts the sides compiled apart inline is in flight.
    pub(super) fn begin_relink(&mut self) -> Arc<TraceIr> {
        self.side.relinking = true;
        self.side.relink_prior = self.side.stone;
        self.begin_stone_recompile()
    }

    /// Whether a compile of the whole tree with the queued sides in it costs
    /// less than compiling those sides apart. The MIR backend compiles a
    /// tree for a fraction of what Cranelift takes for a side
    /// ([`MIR_TREE_COST_PER_OP`] against [`APART_COST_PER_OP`]), and a side
    /// in the tree is a jump where one apart is a call
    /// ([`CHILD_CALL_COST`]), so a small tree takes its sides in. A tree
    /// that is large beside the sides waiting keeps compiling them apart:
    /// each compile of it would cost more than they do. Only a tree MIR
    /// compiled is asked: one it refused goes to Cranelift whole.
    fn joins_for_less(&self) -> bool {
        if !self.compiled.made_by_mir() {
            return false;
        }
        let ops = |ir: &TraceIr| (ir.preheader.len() + ir.body.len()) as u64;
        let waiting: u64 = self
            .side
            .queue
            .queued()
            .iter()
            .map(|(_, side, _)| ops(&side.ir))
            .sum();
        let tree =
            ops(&self.ir) + self.ir.sides.iter().map(|side| ops(&side.ir)).sum::<u64>() + waiting;
        joining_costs_less(tree, waiting)
    }

    /// Whether the first queued side may be compiled apart from the tree
    /// now: nothing is compiling, the queue holds no side that waits for
    /// another (see [`PREVIOUS_SIDE`]), and the tree has code to attach to.
    pub(super) fn can_detach_now(&self) -> bool {
        self.side.compiling.is_empty()
            && self.side.stone != StoneRecompile::InFlight
            && !self.compiled.is_staged()
            && self
                .side
                .queue
                .queued()
                .first()
                .is_some_and(|(key, side, _)| {
                    // A side recorded behind another continues that side's
                    // terminal exit, so it can hang from it once that side is
                    // part of the tree's code.
                    key.parent != Some(PREVIOUS_SIDE) || self.chain_parent(side).is_some()
                })
    }

    /// The side a queued side recorded behind another (see [`PREVIOUS_SIDE`])
    /// continues, if that side is part of the tree's code: the one that just
    /// landed, or the parent a failed batch gave the side.
    fn chain_parent(&self, side: &TraceSide) -> Option<usize> {
        let parent = self.side.landed.or_else(|| {
            side.parent
                .filter(|parent| u32::try_from(*parent).ok() != Some(PREVIOUS_SIDE))
        })?;
        self.compiled
            .side_members()
            .get(parent)
            .is_some_and(Option::is_some)
            .then_some(parent)
    }

    /// The members of the installed tree planned without the side a detach
    /// plan adds. Every candidate side of one state compares against the same
    /// members, so they are planned once per state: the cache is keyed on the
    /// tree, its header liveness, and the detached and apart sides, and a
    /// state change plans them again.
    fn base_plan(&self) -> Result<Arc<Vec<crate::jit_ir::trace::TreeMember>>, &'static str> {
        let _icount = crate::jit::icount::scope("base-plan");
        if let Some(cached) = self.base_plan.borrow().as_ref()
            && Arc::ptr_eq(&cached.ir, &self.ir)
            && cached.header_live == self.side.header_live
            && cached.detached == self.detached
            && cached.apart == self.apart
        {
            return Ok(Arc::clone(&cached.members));
        }
        let mut base = (*self.ir).clone();
        base.header_live.clone_from(&self.side.header_live);
        let members = Arc::new(
            crate::jit_ir::trace::plan_tree_with(&base, &self.detached, &self.apart)
                .map_err(|_| "the tree without it does not plan")?,
        );
        *self.base_plan.borrow_mut() = Some(BasePlan {
            ir: Arc::clone(&self.ir),
            header_live: self.side.header_live.clone(),
            detached: self.detached.clone(),
            apart: self.apart.clone(),
            members: Arc::clone(&members),
        });
        Ok(members)
    }

    /// The pending tree (what [`Self::pending_tree`] returns) and its plan.
    /// Both are made once per state of the queue: the cache is keyed on the
    /// installed tree and the queue's generation, which every change to the
    /// queue advances (see `SideQueue`). `None` when the tree cannot be
    /// planned.
    pub(super) fn seed_plan(
        &self,
    ) -> Option<(Arc<TraceIr>, Arc<Vec<crate::jit_ir::trace::TreeMember>>)> {
        let _icount = crate::jit::icount::scope("seed-plan");
        if let Some(cached) = self.seed_plan.borrow().as_ref()
            && Arc::ptr_eq(&cached.ir, &self.ir)
            && cached.queue_gen == self.side.queue.generation()
        {
            // The oracle for the cache key: whatever the queue went through,
            // a hit is the tree and the plan a fresh pass would make.
            debug_assert!(
                {
                    let fresh = self.pending_tree();
                    *cached.tree == fresh
                        && crate::jit_ir::trace::plan_tree(&fresh)
                            .is_ok_and(|members| members == *cached.members)
                },
                "the cached plan of the pending tree is stale"
            );
            return Some((Arc::clone(&cached.tree), Arc::clone(&cached.members)));
        }
        let tree = self.pending_tree();
        let members = Arc::new(crate::jit_ir::trace::plan_tree(&tree).ok()?);
        let tree = Arc::new(tree);
        *self.seed_plan.borrow_mut() = Some(SeedPlan {
            ir: Arc::clone(&self.ir),
            queue_gen: self.side.queue.generation(),
            tree: Arc::clone(&tree),
            members: Arc::clone(&members),
        });
        Some((tree, members))
    }

    /// The plan of the installed tree without its queued sides. With nothing
    /// queued that is the pending tree's plan, which the caches share.
    pub(super) fn installed_plan(&self) -> Option<Arc<Vec<crate::jit_ir::trace::TreeMember>>> {
        if self.side.queue.queued().is_empty() {
            return self.seed_plan().map(|(_, members)| members);
        }
        crate::jit_ir::trace::plan_tree(&self.ir).ok().map(Arc::new)
    }

    /// How the first queued side can join the tree without compiling the tree
    /// again, if it can.
    ///
    /// The side's code is a function of its own that the exit it continues
    /// calls, so it can join only if the tree's code is the code the tree
    /// would have with it: planned with the side, the root must be the root
    /// the code was compiled from (no register to carry, no entry input to
    /// add), and if the side links back to the loop it must write nothing
    /// the root's optimizer assumed no member writes. Otherwise a side that
    /// links back leaves at the loop header instead, which the interpreter
    /// resumes, and the tree compiles with it later if it proves hot (see
    /// [`DETACHED_LINK_EXITS`]).
    pub(super) fn plan_detached(
        &self,
        stones: bool,
        link_writes: bool,
        join: bool,
    ) -> Option<DetachPlan> {
        let plan = self.plan_detached_inner(stones, link_writes, join);
        if debug_env::trace()
            && let Some((key, _, _)) = self.side.queue.queued().first()
        {
            match &plan {
                Ok(plan) => eprintln!(
                    "jit-trace: side {key:?} compiles apart (exit {}, its guard failed {} \
                     times in {} iterations, {})",
                    plan.exit,
                    self.exit_hits[plan.exit],
                    self.iterations.saturating_sub(self.hits_since),
                    if plan.exit_style {
                        "leaves at the loop header"
                    } else {
                        "links back"
                    }
                ),
                Err(why) => eprintln!("jit-trace: side {key:?} compiles with its tree: {why}"),
            }
        }
        plan.ok()
    }

    fn plan_detached_inner(
        &self,
        stones: bool,
        link_writes: bool,
        join: bool,
    ) -> Result<DetachPlan, &'static str> {
        let Some((key, side, _)) = self.side.queue.queued().first() else {
            return Err("no side is queued");
        };
        // A side recorded behind another hangs from the terminal exit of the
        // side that landed.
        let mut chained;
        let side = if key.parent == Some(PREVIOUS_SIDE) {
            chained = side.clone();
            chained.parent = Some(
                self.chain_parent(side)
                    .ok_or("the side it follows did not land")?,
            );
            &chained
        } else {
            side
        };
        if self.ir.function.is_some()
            || self.ir.loop_tail.is_some()
            || (stones && self.compiled.tier() == crate::jit::OptimizationTier::Baseline)
        {
            return Err("the tree is a function, a tail call loop or a stepping stone");
        }
        if self.side.no_detach.contains(key) {
            return Err("it did not compile apart before");
        }
        if join && self.joins_for_less() {
            return Err("the tree compiles again with it for less than it costs apart");
        }
        // A side that rejoins a loop side's loop jumps into that loop's code,
        // which only the tree's own compile holds.
        if matches!(
            side.ir.terminator,
            crate::jit_ir::trace::TraceTerminator::LinkLoop { .. }
        ) {
            return Err("it rejoins the loop of a loop side");
        }
        // A side that ends in front of a loop comes with the loop side that
        // continues its terminal exit queued right behind it. Unless the side
        // returns into the caller, or is itself one of a chain compiled apart
        // (which the rule on enclosing loops below lets compile apart), the
        // loop side waits for the tree's compile, and finds no exit of the
        // side compiled apart to hang from: it never lands, and the side
        // leaves at that exit to the interpreter.
        let in_apart_chain = key.parent == Some(PREVIOUS_SIDE)
            && side
                .parent
                .is_some_and(|parent| self.apart.contains(&parent));
        if !tree_returns(side)
            && !in_apart_chain
            && self
                .side
                .queue
                .queued()
                .get(1)
                .is_some_and(|(next, _, _)| next.parent == Some(PREVIOUS_SIDE))
        {
            return Err("a loop side continues it");
        }
        // The exit that continues a side recorded behind another is the
        // terminal exit of the side it landed as.
        let guard = if key.parent == Some(PREVIOUS_SIDE) {
            SideKey {
                parent: side.parent.and_then(|parent| u32::try_from(parent).ok()),
                origin: key.origin,
            }
        } else {
            *key
        };
        // The sides of a loop inside other loops come in bursts (the side that
        // runs through a loop, the one behind it, the ones of its guards) while
        // the tree is young, and an outer side runs once per outer iteration,
        // where a side compiled apart costs the loop a call and a re-entry
        // (about [`CHILD_CALL_COST`]) every time: the guard that looks rare
        // in a young tree may fail on every outer iteration of an older one
        // (`algorithm/mandelbrot`: 34 failures in 18 724 iterations became one
        // in 25, and the call and re-entry cost 24M instructions). A tree that
        // has run as long as an outer side waits for (`SIDE_BEYOND_ITERATIONS`)
        // has had its burst and shows the rate. The side that returns into
        // the caller, and the sides recorded behind it, are different: they
        // save an entry and an exit of the tree per call, which costs far more
        // than their call.
        // A side that starts inside inlined calls runs the rest of the
        // iteration from deep inside it: its size follows that path, not the
        // tree, so it compiles apart whatever its guard's rate.
        let seeded = side.ir.seed_frames > 0;
        if self.encloses
            && self.iterations < SIDE_BEYOND_ITERATIONS
            && !seeded
            && !tree_returns(side)
            && !(key.parent == Some(PREVIOUS_SIDE)
                && side.parent.is_some_and(|parent| {
                    self.apart.contains(&parent) && tree_returns_through(&self.ir, parent)
                }))
        {
            return Err("other loops enclose the root's, whose sides come together");
        }
        let exit = self
            .side_keys
            .iter()
            .position(|held| *held == Some(guard))
            .ok_or("no exit continues its guard")?;
        if !self.compiled.can_attach_at(exit) {
            return Err("its exit takes no side");
        }
        // The guard builds the exit's virtual tables on its way into the
        // side (`TraceTables::materialize_exit`), which only a guard of the
        // tree's own code does: a side compiled apart starts from what the
        // exit's stub wrote.
        if self
            .restores
            .get(exit)
            .is_some_and(|restore| !restore.virtuals.is_empty())
        {
            return Err("its guard builds virtual tables on the way in");
        }
        let mut base = (*self.ir).clone();
        base.header_live.clone_from(&self.side.header_live);
        let tree = tree_with_side(base, side.clone());
        let source = tree.sides.len() - 1;
        if tree.sides.len() > self.side.max_sides + self.side.dead
            || crate::jit_ir::trace::verify_grown_tree(&tree, source).is_err()
        {
            return Err("the tree has no room, or does not verify");
        }
        if cfg!(debug_assertions)
            && let Err(error) = tree.verify()
        {
            panic!("a side that fits its tree leaves the tree invalid: {error}");
        }
        // A side that returns from the root frame into its caller runs in the
        // caller's frame, so the tree must be entered for that caller only
        // (`InstalledTrace::caller`, which the entry checks), and every
        // exit of the side rebuilds the caller's frame in the interpreter.
        // The side reads the caller's registers from the run's extra
        // inputs, which the entry fills from the frame it checked.
        let returns = tree.sides[source].ir.frames.iter().any(|info| {
            info.inlined
                .is_some_and(|call| matches!(call.entry, FrameEntry::Caller { .. }))
        });
        if returns {
            if self.compiled.resumes() {
                return Err("the tree resumes coroutines");
            }
            let caller = tree_caller(
                &tree,
                self.caller_chunk.as_ref(),
                self.caller_captured.as_deref(),
            )
            .ok_or("the caller it returns into is not known")?;
            if self
                .caller
                .as_ref()
                .is_some_and(|held| !held.same_as(&caller))
            {
                return Err("it returns into another caller than the tree's sides do");
            }
        }
        let old = self.base_plan()?;
        let mut apart = self.apart.clone();
        apart.push(source);
        let links = matches!(
            tree.sides[source].ir.terminator,
            crate::jit_ir::trace::TraceTerminator::Link { .. }
        );
        let same_root = |planned: &[crate::jit_ir::trace::TreeMember]| {
            planned.first().is_some_and(|root| root.ir == old[0].ir)
        };
        if !links {
            let planned = crate::jit_ir::trace::plan_tree_with(&tree, &self.detached, &apart)
                .map_err(|_| "it does not fit the tree")?;
            return if same_root(&planned) {
                Ok(DetachPlan {
                    tree,
                    source,
                    exit,
                    exit_style: true,
                    detached: self.detached.clone(),
                    apart,
                    planned: Some(Arc::new(planned)),
                })
            } else {
                Err("it changes the root")
            };
        }
        // The failures of the guard since the counts started, against the
        // iterations of the same time.
        let hits = u64::from(self.exit_hits[exit]);
        let iterations = self.iterations.saturating_sub(self.hits_since);
        // A tree that ran too few iterations since the counts started says
        // nothing about how often a guard fails.
        let seen = iterations >= RARITY_EVIDENCE_ITERATIONS;
        // A side that returns into the caller saves the entry and the exit
        // of every call of the function, which costs thousands of
        // instructions where its call costs [`CHILD_CALL_COST`], so its guard
        // need not be rare.
        let rare = seen && hits * LINKED_RARITY <= iterations;
        let mut linked_planned: Option<Vec<crate::jit_ir::trace::TreeMember>> = None;
        let linked: Result<(), &'static str> = if !(returns || rare || seeded) {
            Err("its guard is not rare")
        } else if !self.compiled.allows_linked_sides() {
            Err("the tree reads a key only its links keep valid")
        } else if !link_writes
            && crate::jit_ir::trace::apart_link_writes(&tree, source)
                .map_or(true, |writes| !writes.is_empty())
        {
            Err("its link writes registers the root does not carry")
        } else {
            match crate::jit_ir::trace::plan_tree_with(&tree, &self.detached, &apart) {
                Err(error) => {
                    if debug_env::trace() {
                        eprintln!("jit-trace: the tree with the side does not plan: {error}");
                    }
                    Err("it does not fit the tree with its link")
                }
                Ok(planned) if !same_root(&planned) => Err("its link changes the root"),
                Ok(planned) => {
                    if crate::jit_ir::trace::TreeWrites::of_members(
                        old.iter().map(|member| &member.ir),
                    )
                    .covers(&crate::jit_ir::trace::TreeWrites::of_members([
                        &planned[source + 1].ir,
                    ])) {
                        linked_planned = Some(planned);
                        Ok(())
                    } else {
                        Err("it stores where the tree assumes nothing is stored")
                    }
                }
            }
        };
        if linked.is_ok() {
            return Ok(DetachPlan {
                tree,
                source,
                exit,
                exit_style: false,
                detached: self.detached.clone(),
                apart,
                planned: linked_planned.map(Arc::new),
            });
        }
        let linked_why = linked.err().unwrap_or("it cannot link");
        // A side that cannot link back may leave at the loop header instead if
        // its guard is rare: every run of it then costs an entry of the tree,
        // which a frequent side would not repay.
        //
        // A side that runs on through the loops around the root exists to save
        // the entry of every outer iteration, which leaving at the loop header
        // would spend again.
        let through_loops = self
            .side
            .crossed
            .iter()
            .any(|(known, headers)| known == key && !headers.is_empty());
        if through_loops || returns {
            return Err(linked_why);
        }
        if !seen || hits * EXIT_STYLE_RARITY > iterations {
            return Err(linked_why);
        }
        if debug_env::trace() {
            eprintln!("jit-trace: {key:?} cannot link back from the side: {linked_why}");
        }
        let mut detached = self.detached.clone();
        detached.push(source);
        let planned = crate::jit_ir::trace::plan_tree_with(&tree, &detached, &apart)
            .map_err(|_| "it does not fit the tree")?;
        if same_root(&planned) {
            Ok(DetachPlan {
                tree,
                source,
                exit,
                exit_style: true,
                detached,
                apart,
                planned: Some(Arc::new(planned)),
            })
        } else {
            Err("it changes the root")
        }
    }

    /// Takes the first queued side out of the queue: its compile apart from
    /// the tree is on its way. Returns what the compile needs to know about
    /// the exit it continues.
    pub(super) fn begin_detached_compile(
        &mut self,
        plan: &DetachPlan,
    ) -> Option<crate::jit::SideContext> {
        let unit = self.units.checked_add(1)?;
        let context = self.compiled.side_context(plan.exit, unit)?;
        self.units = unit;
        let (key, _, _) = self.side.queue.queue_remove_first();
        self.side.release();
        self.side.landed = None;
        self.side.compiling = vec![key];
        Some(context)
    }

    /// Marks `key` as the side being compiled apart, for tests.
    #[cfg(test)]
    pub(super) fn begin_refused_detach_for_test(&mut self, key: SideKey) {
        self.side.compiling = vec![key];
    }

    /// A side compiled apart could not join the tree: the backend refused it
    /// or the tree's code could not take it. It goes back to the queue, to
    /// compile with the tree.
    pub(super) fn end_detached_compile_failed(
        &mut self,
        tree: &TraceIr,
        source: usize,
        mark: bool,
    ) {
        let keys = std::mem::take(&mut self.side.compiling);
        let Some(side) = tree.sides.get(source) else {
            return;
        };
        for key in keys {
            if mark {
                self.side.no_detach.push(key);
            }
            self.side
                .queue
                .queue_insert_front((key, side.clone(), self.version));
        }
    }

    /// Whether the calls that search or rewrite a string are most of the
    /// trace's work (see [`is_helper_bound`]).
    pub(super) fn is_helper_bound(&self) -> bool {
        is_helper_bound(&self.ir)
    }

    /// The optimization tier of the trace's code.
    pub(super) fn compiled_tier(&self) -> crate::jit::OptimizationTier {
        self.compiled.tier()
    }

    /// Whether the trace's code is the baseline tier's.
    pub(super) fn is_baseline(&self) -> bool {
        self.compiled.tier() == crate::jit::OptimizationTier::Baseline
    }

    /// Whether the trace's code is the peak tier's.
    pub(super) fn is_peak(&self) -> bool {
        self.compiled.tier() == crate::jit::OptimizationTier::Peak
    }

    /// Whether the trace has run [`PEAK_ITERATIONS`] at the optimized tier,
    /// with no tree compile or side on its way, and is no helper-bound tree,
    /// which gains nothing from better code around its calls.
    ///
    /// A tree MIR compiled stays as it is: the loop Cranelift's optimizer
    /// makes of it runs more instructions an iteration than MIR's (measured
    /// over the benchmark suite, 2.8% more in total). That holds with sides
    /// compiled apart too: `wants_relink` takes them inline with MIR when
    /// their calls have cost enough, and the Peak compile of such a tree ran
    /// `table/array_read` at 72M instructions where MIR's code runs it at
    /// 47M, `opt/range_fail` at 468M against 355M.
    pub(super) fn wants_peak(&self) -> bool {
        self.compiled.tier() == crate::jit::OptimizationTier::Optimized
            && self.iterations >= PEAK_ITERATIONS
            && self.side.stone == StoneRecompile::Idle
            && !self.side.busy()
            && !self.is_helper_bound()
            // The Peak tier compiles with Cranelift only.
            && !crate::jit::debug_env::backend_mir_only()
            && match crate::jit::debug_env::peak() {
                Some(always) => always,
                None => !self.compiled.made_by_mir(),
            }
    }

    /// Whether this is a root compiled at the baseline tier only to lead to
    /// a tree, and the tree is not coming: every guard that ends the loop has
    /// spent its side attempts and no side is on its way. Its code is what
    /// will run from here on, so it deserves the policy's tier.
    pub(super) fn stone_given_up(&self) -> bool {
        self.is_baseline()
            && self.side.stone == StoneRecompile::Idle
            && !self.side.busy()
            && self.ir.sides.is_empty()
            && self
                .side_keys
                .iter()
                .zip(&self.loop_ends)
                .filter(|(_, ends)| **ends)
                .all(|(key, _)| key.is_none_or(|key| self.side.attempts(key) >= MAX_SIDE_ATTEMPTS))
    }

    /// The recompile of the root at the policy's tier is in flight. The root
    /// is the tree, so it takes the place of a tree compile.
    pub(super) fn begin_stone_recompile(&mut self) -> Arc<TraceIr> {
        self.side.stone = StoneRecompile::InFlight;
        Arc::clone(&self.ir)
    }

    fn end_stone_recompile(&mut self) {
        if self.side.stone == StoneRecompile::InFlight {
            // A recompile that put the sides inline leaves the state it
            // found: a tree that was still to earn the peak tier still is,
            // and one that had its recompile still has had it.
            self.side.stone = if self.side.relinking {
                self.side.relink_prior
            } else {
                StoneRecompile::Done
            };
        }
        self.side.relinking = false;
    }

    /// Keeps the plan of the tree that is about to compile (see
    /// `grown_plan`).
    pub(super) fn remember_grown_plan(
        &mut self,
        tree: &Arc<TraceIr>,
        planned: &Arc<Vec<crate::jit_ir::trace::TreeMember>>,
    ) {
        self.grown_plan = Some((Arc::clone(tree), Arc::clone(planned)));
    }

    /// The tree is compiling: it holds the sides of these guards.
    pub(super) fn begin_side_compile(&mut self, keys: Vec<SideKey>) {
        self.side.compiling = keys;
    }

    /// The tree compile was refused for lack of room. Each side spends one
    /// of its guard's attempts and the guard asks again, so a side that
    /// does not fit is recorded [`MAX_SIDE_ATTEMPTS`] times at most: a tree
    /// whose side alone outgrows the cache's metadata cap was refused for
    /// room at every install, spent nothing, and recorded the same side
    /// again without end (145 recordings of one side of `structures.lua`
    /// loop@744 in one run of the `structures-calls` repro). A refused stone
    /// recompile is not asked for yet: the next side abort or tree failure
    /// asks again. Returns whether it was one.
    pub(super) fn end_side_compile_refused(&mut self) -> bool {
        self.grown_plan = None;
        for key in std::mem::take(&mut self.side.compiling) {
            self.side.spend(key, 1);
        }
        let stone = self.side.stone == StoneRecompile::InFlight;
        if stone {
            self.side.stone = if self.side.relinking {
                self.side.relink_prior
            } else {
                StoneRecompile::Idle
            };
            self.side.relinking = false;
        }
        stone
    }

    /// The tree compile failed. A single side would fail again, so its guard
    /// is done. A batch may hold one bad side among good ones: its sides go
    /// back to the queue, to compile one at a time. `tree` is the tree that
    /// failed, whose last sides are the batch.
    pub(super) fn end_side_compile_failed(&mut self, tree: &TraceIr) {
        self.grown_plan = None;
        self.end_stone_recompile();
        let keys = std::mem::take(&mut self.side.compiling);
        let base = tree.sides.len().checked_sub(keys.len());
        let Some(base) = base.filter(|_| keys.len() > 1) else {
            for key in keys {
                self.side.spend(key, MAX_SIDE_ATTEMPTS);
            }
            return;
        };
        self.side.singly = true;
        let version = self.version;
        let again: Vec<_> = keys
            .into_iter()
            .zip(&tree.sides[base..])
            .map(|(key, side)| (key, side.clone(), version))
            .collect();
        self.side.queue.queue_splice_front(again);
    }

    /// Whether a side trace for one of the trace's hot guards is worth waiting
    /// for: one is recording, compiling or queued, or a guard that failed
    /// often still has attempts left. `loop_ends` says whether the guards that
    /// end the loop count (an outer side, see [`Self::outer_hope`]).
    fn wants_side(&self, loop_ends: bool) -> bool {
        if !self.side_capable() {
            return false;
        }
        if self.live_sides() + self.side.in_progress() >= self.side.max_sides {
            return self.side.busy();
        }
        self.side.busy()
            || self
                .side_keys
                .iter()
                .zip(self.exit_hits.iter().zip(&self.exit_last_hit))
                .zip(&self.loop_ends)
                .any(|((key, (hits, last_hit)), ends)| {
                    (loop_ends || !*ends)
                        && key.is_some_and(|key| {
                            self.iterations.saturating_sub(*last_hit) < SIDE_QUIET_ITERATIONS
                                && *hits >= SIDE_EXIT_THRESHOLD / 2
                                && self.side.attempts(key) < MAX_SIDE_ATTEMPTS
                                && side_depth(&self.ir, key.parent) <= MAX_SIDE_DEPTH
                        })
                })
    }

    /// Native iterations the tree has run.
    pub(super) const fn iterations(&self) -> u64 {
        self.iterations
    }

    /// Native runs that executed code over the trace's life.
    pub(super) const fn entries(&self) -> u64 {
        self.entries
    }

    /// Whether the tree's loops carry no floats.
    pub(super) const fn integers(&self) -> bool {
        self.integers
    }

    /// Whether an outer side is on its way: one is recording, compiling or
    /// queued, or a guard that ends the loop failed often and has attempts
    /// left. A covered header whose tree is still growing sides is not yet
    /// judged by how many outer iterations come back to the interpreter.
    pub(super) fn outer_side_pending(&self) -> bool {
        self.wants_side(true)
    }

    /// Whether an outer side is worth its compile. Entering a trace and
    /// leaving it costs a thousand instructions or more, which an outer side
    /// saves per outer iteration, but it also costs a second compile of the
    /// whole tree. A loop that runs long per entry makes the saving small
    /// (a few percent of the run), so it pays only for float loops or loops
    /// that run briefly per entry. The rule was tuned while the former region
    /// JIT ran float loops; it has not been retuned for an interpreter-only
    /// alternative.
    fn outer_side_pays(&self) -> bool {
        !self.integers || self.iterations < OUTER_SIDE_MAX_ITERATIONS_PER_ENTRY * self.entries
    }

    /// The exits taken most in the current window, for the
    /// `OXIMOND_JIT_TRACE` report of a retirement: each as its member, the
    /// guard's origin and kind, the frame depth and pc it resumes at, and
    /// how often it fired.
    fn hottest_exits(&self) -> String {
        let mut hot: Vec<(u32, usize)> = self
            .exit_hits
            .iter()
            .zip(&self.cold_hits)
            .enumerate()
            .map(|(exit, (hits, cold))| (hits.saturating_add(*cold), exit))
            .filter(|(hits, _)| *hits > 0)
            .collect();
        hot.sort_unstable_by(|a, b| b.cmp(a));
        let exits = self.compiled.exits();
        let mut out = String::new();
        for &(hits, exit) in hot.iter().take(4) {
            let Some(descriptor) = exits.get(exit) else {
                continue;
            };
            let member = self.compiled.lowered_member(descriptor.member);
            let (pc, depth) =
                member
                    .snapshots
                    .get(descriptor.snapshot.index())
                    .map_or((0, 0), |snapshot| {
                        let pc = match snapshot.phase {
                            SnapshotPhase::Before { pc } | SnapshotPhase::After { pc, .. } => pc,
                        };
                        (pc, snapshot.frames.len())
                    });
            let _ = write!(
                out,
                "; exit {exit} (member {}, guard {:?} {:?}, frame {depth} pc {pc}) {hits}x",
                descriptor.member, descriptor.origin, descriptor.guard
            );
        }
        out
    }

    /// Whether a side in the tree continues the guard `descriptor` exits at.
    fn side_continues(&self, descriptor: &ExitDescriptor) -> bool {
        let Some(origin) = descriptor.origin else {
            return false;
        };
        // The side the exit's member is, as an index into the source sides
        // (`None` for the root).
        let parent = if descriptor.member == 0 {
            None
        } else {
            match self
                .compiled
                .side_members()
                .iter()
                .position(|&member| member == Some(descriptor.member))
            {
                Some(side) => Some(side),
                None => return false,
            }
        };
        self.ir
            .sides
            .iter()
            .any(|side| side.guard == origin && side.parent == parent)
    }

    /// The call sites whose callee identity guards this trace left at
    /// [`POLY_CALL_EXITS`] times or more in the current window, as the
    /// paths [`Site::poly_calls`](super::policy::Site) names them by: the
    /// call pcs of the snapshot's inlined frames, outermost first, then the
    /// pc of the call it resumes at. The guards of one call in the root and
    /// in a chain of sides (a side per callee) count together, and a guard
    /// that a side continues counts at least once: it left often enough for
    /// that side, so a chain of [`POLY_CALL_EXITS`] such guards saw that many
    /// callees even when the window restarted with each side.
    fn callee_exit_paths(&self) -> Vec<PolyCallPath> {
        let exits = self.compiled.exits();
        let mut paths: Vec<(PolyCallPath, u32)> = Vec::new();
        for (exit, (&hits, &cold)) in self.exit_hits.iter().zip(&self.cold_hits).enumerate() {
            let Some(path) = self.callee_exit_path(exit) else {
                continue;
            };
            let continued = exits
                .get(exit)
                .is_some_and(|descriptor| self.side_continues(descriptor));
            paths.push((path, callee_guard_hits(hits, cold, continued)));
        }
        poly_call_paths(paths)
    }

    /// The call site of exit `exit` when it leaves at a callee identity
    /// guard, named as [`Site::poly_calls`](super::policy::Site) names it:
    /// the call pcs of the snapshot's inlined frames, outermost first, then
    /// the pc of the call the exit resumes at.
    fn callee_exit_path(&self, exit: usize) -> Option<PolyCallPath> {
        let descriptor = self.compiled.exits().get(exit)?;
        if descriptor.class != TraceExitClass::GuardFail
            || descriptor.guard != Some(TraceGuardClass::Callee)
        {
            return None;
        }
        let member = self.compiled.lowered_member(descriptor.member);
        // A guard on the prototype already runs every closure of it; its
        // sides are one per prototype.
        if descriptor
            .origin
            .is_some_and(|origin| guards_on_prototype(member, origin))
        {
            return None;
        }
        let snapshot = member.snapshots.get(descriptor.snapshot.index())?;
        let SnapshotPhase::Before { pc } = snapshot.phase else {
            return None;
        };
        let mut path: Vec<usize> = snapshot
            .frames
            .iter()
            .skip(1)
            .filter_map(|frame| {
                member
                    .frames
                    .get(frame.frame.0 as usize)
                    .and_then(|info| info.inlined.as_ref())
                    .map(|inlined| inlined.call_pc)
            })
            .collect();
        if path.len() + 1 != snapshot.frames.len() {
            return None;
        }
        path.push(pc);
        Some(path.into_boxed_slice())
    }

    /// The call site of the callee guard a side for `key` would continue,
    /// when the tree already has [`POLY_CALL_SIDES`] sides that continue a
    /// callee guard of that site: a side for one more closure there would
    /// make the tree grow with the number of closures, as round 3's did on
    /// structures.lua loop@744 (five to seven sides of one `chunk_test`
    /// call, up to 16.3 MB of metadata, then evicted for room). The policy
    /// names the site instead, and the loop records again with the call
    /// guarded on the prototype.
    pub(super) fn poly_side_path(&self, key: SideKey) -> Option<PolyCallPath> {
        let exits = self.compiled.exits();
        let exit = self
            .side_keys
            .iter()
            .enumerate()
            .find_map(|(exit, held)| (*held == Some(key)).then_some(exit))?;
        let path = self.callee_exit_path(exit)?;
        let continued = (0..exits.len())
            .filter(|&other| {
                other != exit
                    && exits
                        .get(other)
                        .is_some_and(|descriptor| self.side_continues(descriptor))
                    && self.callee_exit_path(other).as_ref() == Some(&path)
            })
            .count();
        (continued >= POLY_CALL_SIDES).then_some(path)
    }

    /// Whether the callee the interpreter now finds at the call of the
    /// callee identity guard `key` names (`frame` stands at the call) runs
    /// the prototype of the closure the guard pinned: another closure of
    /// one prototype, which a prototype guard would carry. Callees of
    /// distinct prototypes keep a side each.
    fn callee_shares_prototype(
        &self,
        key: SideKey,
        frame: crate::vm::stack::FrameRef<'_>,
        runtime: &Runtime,
    ) -> bool {
        let Some(exit) = self.side_keys.iter().position(|held| *held == Some(key)) else {
            return false;
        };
        let Some(descriptor) = self.compiled.exits().get(exit) else {
            return false;
        };
        let Some(origin) = descriptor.origin else {
            return false;
        };
        let member = self.compiled.lowered_member(descriptor.member);
        let Some(pinned) =
            guard_pinned_closure(member, origin).and_then(|lane| self.pinned.get(lane as usize))
        else {
            return false;
        };
        let Some(callee) = frame
            .frame
            .chunk
            .code
            .get(frame.frame.pc)
            .copied()
            .and_then(super::recorder::callee_register)
        else {
            return false;
        };
        let Some(&Value::Closure(live)) = frame.registers.get(usize::from(callee)) else {
            return false;
        };
        let chunk_of = |handle: ClosureHandle| {
            runtime
                .closure_call_parts(handle)
                .map(|parts| Arc::as_ptr(parts.chunk))
        };
        match (
            chunk_of(live),
            chunk_of(ClosureHandle::from_native_identity(*pinned)),
        ) {
            (Some(live), Some(pinned)) => std::ptr::eq(live, pinned),
            _ => false,
        }
    }

    /// Whether the trace left [`HOISTED_GUARD_EXITS`] times or more in the
    /// current window at branch guards that no side continues and that
    /// leave through the loop's entry state: guards the optimizer hoisted in
    /// front of the loop.
    fn leaves_at_hoisted_guards(&self) -> bool {
        let exits = self.compiled.exits();
        let hits: u32 = self
            .cold_hits
            .iter()
            .enumerate()
            .filter(|&(exit, _)| {
                exits.get(exit).is_some_and(|descriptor| {
                    descriptor.class == TraceExitClass::GuardFail
                        && descriptor.guard == Some(TraceGuardClass::Branch)
                        && self
                            .compiled
                            .lowered_member(descriptor.member)
                            .snapshots
                            .get(descriptor.snapshot.index())
                            .is_some_and(|snapshot| {
                                snapshot.frames.len() == 1
                                    && snapshot.phase
                                        == (SnapshotPhase::Before { pc: self.header_pc })
                            })
                })
            })
            .map(|(_, hits)| *hits)
            .sum();
        hits >= HOISTED_GUARD_EXITS && !self.ir.keep_branch_guards
    }

    /// For a root whose entries keep leaving at branch guards the optimizer
    /// hoisted in front of the loop (a quarter of its entries or more, and
    /// [`HOISTED_GUARD_EXITS`] at least): from here on the tree keeps its
    /// branch guards in the loop body, and its code is compiled again with
    /// them there (see [`Self::begin_keep_guards`]), where a side can
    /// continue them. An argument that is one value in one call and another
    /// in the next (`chance` of Mineclonia's `fill_box_rotated`, 1.0 and 0.8
    /// in turn, mineshaft.lua:299) otherwise sends every run of the other
    /// value back to the interpreter at its first iteration, and the tree
    /// lives on the runs of the first. Returns whether it asked.
    fn keep_hoisted_guards(&mut self) -> bool {
        // A root that loops of its own frame enclose leaves at its entry
        // state when its loop ends (the inner loop of a nest, mineshaft
        // repro's `prepare_mapchunk`): those exits take outer sides once
        // the root is a stepping stone (see `keep_stone`), and recompiling
        // for them only delays that.
        if self.ir.keep_branch_guards
            || self.encloses
            || self.ir.function.is_some()
            || !self.side_capable()
            || self.compiled.optimize_stats().guards_hoisted == 0
            || !self.leaves_at_hoisted_guards()
        {
            return false;
        }
        let hoisted: u64 = self.cold_hits.iter().map(|hits| u64::from(*hits)).sum();
        if !leaves_often_at_hoisted_guards(hoisted, self.entries) {
            return false;
        }
        let mut ir = (*self.ir).clone();
        ir.keep_branch_guards = true;
        self.ir = Arc::new(ir);
        self.keep_guards_wanted = true;
        self.guard_move_grace = GUARD_MOVE_GRACE_WINDOWS;
        if debug_env::trace() {
            eprintln!(
                "jit-trace: the trace at pc {} left at hoisted guards in {hoisted} of {} entries; it compiles again with them in the loop",
                self.header_pc, self.entries
            );
        }
        true
    }

    /// Sides of the tree that the compiled code runs.
    fn live_sides(&self) -> usize {
        self.ir.sides.len().saturating_sub(self.side.dead)
    }

    /// Whether the root can take a side trace at all. `consider_side` refuses
    /// any other root for good, so nothing may wait for one.
    fn side_capable(&self) -> bool {
        // A function trace's guards continue to the activation's return.
        if self.ir.function.is_some() {
            return true;
        }
        self.ir.poll_snapshot.is_some_and(|poll| {
            matches!(
                self.ir.snapshots[poll.index()].phase,
                crate::jit_ir::trace::SnapshotPhase::After { .. }
            )
        })
    }

    /// Whether a window that closed below the productivity bar should be
    /// forgiven because a side trace is on its way.
    fn forgive_for_side(&mut self) -> bool {
        if self.side.grace == 0 || !self.wants_side(true) {
            return false;
        }
        self.side.grace -= 1;
        true
    }

    /// Whether a window that closed below the productivity bar leaves this
    /// root installed as a stepping stone: a `while` or generic-for loop
    /// that other loops enclose and that runs too few trips per entry to
    /// pay alone (n1's three `while` loops, an `ipairs` loop over three
    /// items). Its tree is the trace that pays, once outer sides carry it
    /// through the loops around it, and those sides are recorded from its
    /// exits, which retiring it would throw away.
    ///
    /// The first such window makes the root a stone and gives it its grace
    /// again. It is kept while a side is on its way, or while a side landed
    /// (or the stone began) less than [`STONE_PATIENCE_ENTRIES`] entries ago
    /// and at least half of the runs since left by guards that still have
    /// side attempts. Past that, once every guard spent its attempts (the
    /// outer part calls what no side can record, or its side recordings are
    /// rejected), or once more entries since it became a stone found types
    /// no version takes than ran (a root whose entries mostly did so
    /// before is not made a stone), it retires as any root does, and a second
    /// retirement marks the loop `retired_short` so the loops around it
    /// unroll it. The site counts the stones that gave up
    /// (`Site::stone_gave_up`), so after the second (or the first that
    /// mostly mismatched) the roots recorded there are judged like any root
    /// and not kept and recompiled at every retirement.
    fn keep_stone(&mut self) -> bool {
        if !self.stone_candidate || !self.side_capable() {
            return false;
        }
        // A tree whose entry finds other types most of the time (a local
        // that is nil on some outer iterations and a number on others)
        // pays for its entry checks and runs in the interpreter anyway.
        if self.mostly_mismatched() {
            return false;
        }
        // A root that just became a stone gets a window to show its exits.
        let Some(since) = self.stone else {
            self.stone = Some(self.entries);
            self.stone_start = (self.entries, self.mismatches);
            self.side.grace = SIDE_GRACE_WINDOWS;
            // From here on the tree keeps its branch guards in the loop body
            // (`TraceIr::keep_branch_guards`), and code that hoisted one is
            // compiled again.
            let mut ir = (*self.ir).clone();
            ir.keep_branch_guards = true;
            self.ir = Arc::new(ir);
            self.keep_guards_wanted = self.compiled.optimize_stats().guards_hoisted > 0;
            // The end of an `ipairs` loop takes an outer side now.
            let chunk = self.chunk.upgrade();
            (self.side_keys, self.virtual_refused) = side_exits_of(
                &self.compiled,
                &self.restores,
                &self.ir,
                self.encloses,
                true,
                chunk.as_deref(),
            );
            if debug_env::trace() {
                eprintln!(
                    "jit-trace: the trace at pc {}{} runs {} iterations in {} entries inside other loops, kept as a stepping stone",
                    self.header_pc,
                    self.chunk
                        .upgrade()
                        .map(|chunk| debug_env::at_line(&chunk, self.header_pc))
                        .unwrap_or_default(),
                    self.iterations,
                    self.entries
                );
            }
            return true;
        };
        // A side on its way is waited for as `forgive_for_side` waits, for
        // as long as the grace lasts.
        if self.side.busy() && self.side.grace > 0 {
            self.side.grace -= 1;
            return true;
        }
        // The runs still leave by guards that can take a side: such a guard
        // records once it has failed `SIDE_EXIT_THRESHOLD` times, which one
        // on the nest's common path reaches within a few dozen entries. A
        // root whose runs mostly leave where no side can start (a guard
        // hoisted into the preheader, true only of the iteration it was
        // recorded on) waits for nothing.
        let room = self.live_sides() + self.side.in_progress() < self.side.max_sides;
        let waiting = self.entries.saturating_sub(since);
        let live_hits: u64 = self
            .side_keys
            .iter()
            .zip(self.exit_hits.iter().zip(&self.exit_last_hit))
            .filter(|(key, (_, last_hit))| {
                self.iterations.saturating_sub(**last_hit) < SIDE_QUIET_ITERATIONS
                    && key.is_some_and(|key| {
                        self.side.attempts(key) < MAX_SIDE_ATTEMPTS
                            && side_depth(&self.ir, key.parent) <= MAX_SIDE_DEPTH
                    })
            })
            .map(|(_, (hits, _))| u64::from(*hits))
            .sum();
        let failing = live_hits > 0 && 2 * live_hits >= waiting;
        let hope = room && failing && waiting <= STONE_PATIENCE_ENTRIES;
        if !hope && debug_env::trace() {
            eprintln!(
                "jit-trace: the stepping stone at pc {}{} gave up on its outer sides ({} iterations in {} entries, {live_hits} of {waiting} runs left by guards with side attempts, room: {room}; interpreter exits by kind and guard {:?})",
                self.header_pc,
                self.chunk
                    .upgrade()
                    .map(|chunk| debug_env::at_line(&chunk, self.header_pc))
                    .unwrap_or_default(),
                self.iterations,
                self.entries,
                self.interp_exits
            );
        }
        hope
    }

    /// For a stepping stone whose code hoisted a branch guard: the tree to
    /// compile again with its branch guards kept in the loop, when nothing
    /// else compiles or records for it. The recompile takes the place of a
    /// tree compile and leaves the tier and stone state as it found them.
    pub(super) fn begin_keep_guards(&mut self) -> Option<Arc<TraceIr>> {
        if !self.keep_guards_wanted
            || self.side.busy()
            || self.side.stone == StoneRecompile::InFlight
            || self.compiled.is_staged()
        {
            return None;
        }
        self.keep_guards_wanted = false;
        Some(self.begin_relink())
    }

    /// Whether more entries found types no version takes than ran, since
    /// the trace became a stepping stone (or since it was installed, for one
    /// that is not one yet).
    fn mostly_mismatched(&self) -> bool {
        let (entries, mismatches) = self.stone_start;
        self.mismatches.saturating_sub(mismatches) > self.entries.saturating_sub(entries)
    }

    /// A side of the tree landed: a stepping stone waits for the next one
    /// from here.
    fn stone_progress(&mut self) {
        if self.stone.is_some() {
            self.stone = Some(self.entries);
        }
    }

    /// Whether the guard at `exit` can still take a side trace: it has a
    /// key with attempts left, and the tree has room or a side is on its
    /// way.
    fn side_possible(&self, exit: usize) -> bool {
        self.side_capable()
            && (self.live_sides() + self.side.in_progress() < self.side.max_sides
                || self.side.busy())
            && self.side_keys[exit].is_some_and(|key| {
                self.side.attempts(key) < MAX_SIDE_ATTEMPTS
                    && side_depth(&self.ir, key.parent) <= MAX_SIDE_DEPTH
            })
    }

    /// Executable bytes this trace charges the cache.
    pub(super) fn code_charge(&self) -> usize {
        self.compiled.code_sizes().map(code_charge).sum()
    }

    /// Metadata bytes this trace charges the cache.
    pub(super) fn meta_charge(&self) -> usize {
        meta_bytes_of(&self.ir, &self.compiled, &self.restores, self.roots.len())
    }

    /// [`Self::meta_charge`]'s parts for the trace log: the IR of the tree,
    /// the lowered IR of its members and the exits' restore plans, and the
    /// number of members.
    fn meta_parts(&self) -> ((usize, usize, usize), usize) {
        let members = self.compiled.member_count();
        let lowered = (0..members)
            .map(|member| self.compiled.lowered_member(member).estimated_bytes())
            .sum::<usize>();
        let ir = self.ir.estimated_bytes();
        let total = meta_bytes_of(&self.ir, &self.compiled, &self.restores, self.roots.len());
        ((ir, lowered, total.saturating_sub(ir + lowered)), members)
    }

    pub(super) const fn header_pc(&self) -> usize {
        self.header_pc
    }

    /// The tree as it will be once the queued sides join it, for the sides
    /// recorded behind them. A side recorded behind another (see
    /// [`PREVIOUS_SIDE`]) continues the entry queued before it.
    pub(super) fn pending_tree(&self) -> TraceIr {
        let _icount = crate::jit::icount::scope("pending-tree");
        let mut tree = (*self.ir).clone();
        let mut previous: Option<usize> = None;
        for (key, side, _) in self.side.queue.queued() {
            let mut side = side.clone();
            let before = previous.take();
            if key.parent == Some(PREVIOUS_SIDE) {
                let Some(parent) = before else {
                    continue;
                };
                side.parent = Some(parent);
            }
            tree.sides.push(side);
            previous = tree.sides.len().checked_sub(1);
        }
        tree
    }

    /// The tree as recorded: the root with the sides that joined it.
    pub(super) fn tree_ir(&self) -> &TraceIr {
        &self.ir
    }

    /// The profile the trace was recorded under.
    pub(super) const fn profile(&self) -> Option<CompatibilityProfile> {
        self.profile
    }

    pub(super) fn chunk(&self) -> Option<Arc<Chunk>> {
        self.chunk.upgrade()
    }

    pub(super) fn chunk_alive(&self) -> bool {
        self.chunk.strong_count() > 0
    }

    pub(super) fn profile_site(&self) -> Option<(Arc<Chunk>, usize)> {
        Some((self.chunk.upgrade()?, self.header_pc))
    }

    pub(super) fn profile_ir(&self) -> &TraceIr {
        &self.ir
    }

    pub(super) fn profile_address(&self) -> (usize, usize) {
        (self.chunk.as_ptr() as usize, self.header_pc)
    }
}

/// How many frames the deepest inlined call of `ir` adds to the root frame:
/// an exit inside it pushes that many.
/// The deepest call chain any member of the tree (the root and its sides)
/// inlines.
fn tree_inline_depth(ir: &TraceIr) -> usize {
    ir.sides
        .iter()
        .map(|side| inline_depth(&side.ir))
        .fold(inline_depth(ir), usize::max)
}

/// The frame below the root frame that a tree's sides return into (see
/// [`Entry::Caller`]), as entry checks it.
#[derive(Clone, Debug)]
pub(super) struct CallerEntry {
    /// The chunk the caller frame runs. The weak reference keeps the chunk's
    /// address from being used by another, so the pointer says which it is.
    chunk: Weak<Chunk>,
    /// The pinned closure the caller runs, if it runs one.
    closure: Option<u32>,
    /// The call in the caller's chunk that made the root frame.
    call_pc: usize,
    /// The caller's result window for the call.
    result_base: u8,
    results: u8,
    /// The registers of the caller a closure may have captured: no other
    /// register may be, since the trace writes registers of the caller that
    /// are not these.
    captured: Box<[u8]>,
}

/// The caller frame a tree returns into, if any of its members does, with
/// the chunk `chunk` the sides recorded it in.
fn tree_caller(
    ir: &TraceIr,
    chunk: Option<&Weak<Chunk>>,
    captured: Option<&[u8]>,
) -> Option<CallerEntry> {
    std::iter::once(ir)
        .chain(ir.sides.iter().map(|side| &*side.ir))
        .flat_map(|member| &member.frames)
        .find_map(|info| match info.inlined {
            Some(call) => match call.entry {
                FrameEntry::Caller {
                    call_pc,
                    result_base,
                    results,
                } => Some(CallerEntry {
                    chunk: chunk?.clone(),
                    closure: (call.closure != super::recorder::NO_CLOSURE).then_some(call.closure),
                    call_pc: call_pc as usize,
                    result_base,
                    results,
                    captured: captured.unwrap_or_default().into(),
                }),
                _ => None,
            },
            None => None,
        })
}

/// What the entry of a tree that returns into its caller reads from the
/// caller's frame.
#[derive(Debug)]
pub(super) struct CallerView {
    registers: Vec<Value>,
    environment: TableHandle,
    chunk: Arc<Chunk>,
}

impl CallerEntry {
    /// Whether `other` names the same caller: the same chunk, closure, call and
    /// result window.
    fn same_as(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.chunk, &other.chunk)
            && self.closure == other.closure
            && self.call_pc == other.call_pc
            && self.result_base == other.result_base
            && self.results == other.results
    }
}

/// The caller frame of the root frame at `frame_index`, if
/// that frame is the one a tree that returns into its caller was recorded to
/// return into: the frame `entry` names called the root frame at its call
/// `call_pc`, for the result window the loop's own call uses, and the root
/// frame is as fresh a call makes it. `None` when entry must not run.
fn caller_view(
    frames: &CallStack,
    frame_index: usize,
    entry: &CallerEntry,
    pinned: &[u64],
) -> Option<CallerView> {
    let below = frame_index.checked_sub(1)?;
    let root = frames.at(frame_index);
    let caller = frames.at(below);
    let Some(crate::vm::ReturnTarget::Direct(target)) = root.frame.caller.as_ref() else {
        return None;
    };
    if target.base != entry.result_base
        || target.results != entry.results
        // A frame the loop's own call would not have made: the second round
        // starts one that has no tail calls, varargs or interior state.
        || root.frame.tail_calls != 0
        || root.frame.has_vararg_state()
        || root.frame.extras_ref().is_some()
        || !std::ptr::eq(caller.frame.chunk.address(), entry.chunk.as_ptr())
        || caller.frame.closure.map(ClosureHandle::native_identity)
            != entry.closure.and_then(|lane| pinned.get(lane as usize).copied())
        || caller.frame.last_interpreted_pc.get() != Some(entry.call_pc)
    {
        return None;
    }
    // A register of the caller that a closure captured is the variable, not
    // the value in the register: the trace keeps registers in native values
    // and writes them back at its exits. It may only leave alone the ones
    // that were captured when its sides were recorded.
    let base = caller.frame.base;
    if frames.open_in(base, caller.frame.top).iter().any(|open| {
        !u8::try_from(open.slot - base).is_ok_and(|register| entry.captured.contains(&register))
    }) {
        return None;
    }
    Some(CallerView {
        registers: caller.registers.to_vec(),
        environment: caller.frame.environment,
        chunk: Arc::clone(&caller.frame.chunk.as_arc()),
    })
}

fn inline_depth(ir: &TraceIr) -> usize {
    let mut depths: Vec<usize> = Vec::with_capacity(ir.frames.len());
    for info in &ir.frames {
        let depth = info.inlined.map_or(0, |call| {
            depths.get(call.parent.index()).map_or(1, |d| d + 1)
        });
        depths.push(depth);
    }
    depths.into_iter().max().unwrap_or(0)
}

/// Metadata a trace holds: its source IR, the optimized IR its exits name,
/// and the restore plans.
///
/// Both IRs are charged in full, because both stay resident: the source IR
/// serves recompiling at the Optimized tier, the lowered one serves exit restores.
/// Since P5 a trace therefore costs about twice what it did, and the effective
/// capacity of `MAX_TRACE_META_BYTES` is roughly halved. That is deliberate.
fn meta_bytes_of(
    ir: &TraceIr,
    compiled: &CompiledTrace,
    restores: &[ExitRestore],
    roots: usize,
) -> usize {
    ir.estimated_bytes()
        + (0..compiled.member_count())
            .map(|member| compiled.lowered_member(member).estimated_bytes())
            .sum::<usize>()
        + roots * std::mem::size_of::<Value>()
        + restores
            .iter()
            .map(|restore| {
                std::mem::size_of::<ExitRestore>()
                    + restore.upvalues.len() * std::mem::size_of::<RegisterAction>()
                    + restore.changes.len() * std::mem::size_of::<(u16, RegisterAction)>()
                    + restore.nils.len() * std::mem::size_of::<u16>()
                    + restore.floats.len() * std::mem::size_of::<(u16, u32)>()
                    + restore
                        .virtuals
                        .iter()
                        .map(|plan| {
                            std::mem::size_of::<VirtualPlan>()
                                + (plan.array.len() + plan.fields.len())
                                    * (std::mem::size_of::<RegisterAction>()
                                        + std::mem::size_of::<Value>())
                        })
                        .sum::<usize>()
                    + restore
                        .inner
                        .iter()
                        .map(|plan| {
                            std::mem::size_of::<InnerPlan>()
                                + plan.writes.len() * std::mem::size_of::<(u16, RegisterAction)>()
                        })
                        .sum::<usize>()
            })
            .sum::<usize>()
}

/// Whether the guard `origin` of `ir` is a prototype guard: a branch guard on
/// [`HelperId::CLOSURE_SAME_PROTO`].
fn guards_on_prototype(ir: &TraceIr, origin: u32) -> bool {
    use crate::jit_ir::trace::ValueDef;
    ir.preheader.iter().chain(&ir.body).any(|instruction| {
        let TraceOp::GuardBranch {
            condition, guard, ..
        } = &instruction.op
        else {
            return false;
        };
        if ir.guards.get(guard.index()).map(|info| info.origin) != Some(origin) {
            return false;
        }
        let def = match ir.values.get(condition.index()).map(|info| info.def) {
            Some(ValueDef::Instruction(index)) => ir.body.get(index),
            Some(ValueDef::Preheader(index)) => ir.preheader.get(index),
            _ => None,
        };
        matches!(
            def.map(|instruction| &instruction.op),
            Some(TraceOp::CallHelper { helper, .. }) if *helper == HelperId::CLOSURE_SAME_PROTO
        )
    })
}

/// The pinned lane the identity guard `origin` of `ir` compares its callee
/// with: a branch guard on a closure comparison with a closure constant.
fn guard_pinned_closure(ir: &TraceIr, origin: u32) -> Option<u32> {
    use crate::jit_ir::trace::ValueDef;
    let op_of = |value: crate::jit_ir::trace::ValueId| {
        match ir.values.get(value.index()).map(|info| info.def) {
            Some(ValueDef::Instruction(index)) => ir.body.get(index),
            Some(ValueDef::Preheader(index)) => ir.preheader.get(index),
            _ => None,
        }
        .map(|instruction| &instruction.op)
    };
    ir.preheader.iter().chain(&ir.body).find_map(|instruction| {
        let TraceOp::GuardBranch {
            condition, guard, ..
        } = &instruction.op
        else {
            return None;
        };
        if ir.guards.get(guard.index()).map(|info| info.origin) != Some(origin) {
            return None;
        }
        let Some(TraceOp::Compare { left, right, .. }) = op_of(*condition) else {
            return None;
        };
        [*left, *right]
            .into_iter()
            .find_map(|side| match op_of(side) {
                Some(TraceOp::Constant(TraceScalar::Closure(lane))) => u32::try_from(*lane).ok(),
                _ => None,
            })
    })
}

/// How often one callee identity guard counts toward its call site in
/// [`InstalledTrace::callee_exit_paths`]: the exits that ran a side's
/// continuation (`hits`) and the exits no side continues (`cold`), and at
/// least one for a guard a linked side continues, which jumps there instead
/// of leaving, so its own count stays at zero.
fn callee_guard_hits(hits: u32, cold: u32, continued: bool) -> u32 {
    hits.saturating_add(cold).max(u32::from(continued))
}

/// The call sites among `exits` (one path and count per callee guard, the
/// root's and every side's) whose guards together left [`POLY_CALL_EXITS`]
/// times or more, in the order they first appear.
fn poly_call_paths(exits: Vec<(PolyCallPath, u32)>) -> Vec<PolyCallPath> {
    let mut paths: Vec<(PolyCallPath, u32)> = Vec::new();
    for (path, hits) in exits {
        match paths.iter_mut().find(|(known, _)| *known == path) {
            Some((_, total)) => *total = total.saturating_add(hits),
            None => paths.push((path, hits)),
        }
    }
    paths
        .into_iter()
        .filter(|&(_, hits)| hits >= POLY_CALL_EXITS)
        .map(|(path, _)| path)
        .collect()
}

#[cfg(test)]
mod mineshaft_rule_tests {
    use super::leaves_often_at_hoisted_guards;

    #[test]
    fn a_root_compiles_with_its_guards_kept_when_a_quarter_of_its_entries_leave_at_them() {
        assert!(!leaves_often_at_hoisted_guards(0, 100));
        assert!(!leaves_often_at_hoisted_guards(24, 100));
        assert!(leaves_often_at_hoisted_guards(25, 100));
        assert!(leaves_often_at_hoisted_guards(100, 100));
        assert!(leaves_often_at_hoisted_guards(u64::MAX, u64::MAX));
    }
}

#[cfg(test)]
mod poly_path_tests {
    use super::{POLY_CALL_EXITS, PolyCallPath, callee_guard_hits, poly_call_paths};

    fn path(pcs: &[usize]) -> PolyCallPath {
        pcs.to_vec().into_boxed_slice()
    }

    #[test]
    fn a_continued_guard_counts_once_and_a_cold_one_by_its_exits() {
        assert_eq!(callee_guard_hits(0, 0, false), 0);
        assert_eq!(callee_guard_hits(0, 0, true), 1);
        assert_eq!(callee_guard_hits(2, 3, false), 5);
        assert_eq!(callee_guard_hits(2, 3, true), 5);
        assert_eq!(callee_guard_hits(u32::MAX, 1, false), u32::MAX);
    }

    #[test]
    fn the_guards_of_a_root_and_its_chain_of_sides_count_together() {
        // A root's guard at the call that a side continues (1), the side's
        // guard that a second side continues (1), and that side's guard,
        // which leaves (POLY_CALL_EXITS - 2 times): one site, named.
        let site = path(&[14, 6]);
        let chain = vec![
            (site.clone(), callee_guard_hits(0, 0, true)),
            (site.clone(), callee_guard_hits(0, 0, true)),
            (
                site.clone(),
                callee_guard_hits(0, POLY_CALL_EXITS - 2, false),
            ),
        ];
        assert_eq!(poly_call_paths(chain), vec![site.clone()]);
        // One guard short of the count: not named.
        let short = vec![
            (site.clone(), callee_guard_hits(0, 0, true)),
            (
                site.clone(),
                callee_guard_hits(0, POLY_CALL_EXITS - 2, false),
            ),
        ];
        assert_eq!(poly_call_paths(short), [] as [std::boxed::Box<[usize]>; 0]);
    }

    #[test]
    fn guards_of_other_call_paths_do_not_add_up() {
        // The same call pc reached through another inlined call is another
        // site, and so is another call in the same frame.
        let half = POLY_CALL_EXITS / 2;
        let exits = vec![
            (path(&[14, 6]), half),
            (path(&[20, 6]), POLY_CALL_EXITS - half),
            (path(&[14, 9]), POLY_CALL_EXITS - half),
            (path(&[3]), POLY_CALL_EXITS),
        ];
        assert_eq!(poly_call_paths(exits), vec![path(&[3])]);
    }
}

/// The restore plan of every exit of `compiled`, by exit index. `None` when
/// an exit's snapshot does not fit the trace's frames.
#[cfg(test)]
pub(crate) fn restores_of(
    compiled: &CompiledTrace,
    register_count: usize,
    externals: &[ExternalSlot],
) -> Option<Vec<ExitRestore>> {
    restores_planned(compiled, register_count, externals, false)
}

/// [`restores_of`] for the plans an installed trace runs, which restore onto
/// the frame the trace was entered with: a register that an exit restores to
/// the value it was entered with is left as it is.
fn restores_of_entered(
    compiled: &CompiledTrace,
    register_count: usize,
    externals: &[ExternalSlot],
) -> Option<Vec<ExitRestore>> {
    restores_planned(compiled, register_count, externals, true)
}

/// [`restores_of_entered`] for the exits from `first` on: the plans of the
/// exits that were there before depend on nothing the exits added after them
/// change, so a side that joins the code is planned alone.
fn restores_of_entered_from(
    compiled: &CompiledTrace,
    first: usize,
    register_count: usize,
    externals: &[ExternalSlot],
) -> Option<Vec<ExitRestore>> {
    restores_planned_from(compiled, first, register_count, externals, true)
}

fn restores_planned(
    compiled: &CompiledTrace,
    register_count: usize,
    externals: &[ExternalSlot],
    entered: bool,
) -> Option<Vec<ExitRestore>> {
    restores_planned_from(compiled, 0, register_count, externals, entered)
}

fn restores_planned_from(
    compiled: &CompiledTrace,
    first: usize,
    register_count: usize,
    externals: &[ExternalSlot],
    entered: bool,
) -> Option<Vec<ExitRestore>> {
    let _icount = crate::jit::icount::scope("restores");
    // The entry inputs of each member by register, built when its first
    // exit is planned.
    let mut own: Vec<Option<OwnInputs>> = Vec::new();
    own.resize_with(compiled.member_count(), || None);
    let mut scratch = RestoreScratch::default();
    let restores: Option<Vec<ExitRestore>> = compiled
        .exits()
        .iter()
        .skip(first)
        .map(|descriptor| {
            // Each exit names a snapshot of its own trace of the tree.
            let lowered = compiled.lowered_member(descriptor.member);
            let own = own
                .get_mut(descriptor.member)?
                .get_or_insert_with(|| OwnInputs::of(lowered));
            // The root's entry state is what the frame holds, unless the
            // tree runs tail calls into other closures, is a function trace,
            // or runs a coroutine that may write the frame's cells.
            // `held_registers` in jit/trace/compile.rs bails out on the same
            // conditions, and exits leave out the values it finds held, so
            // change the two together.
            let root = compiled.lowered_member(0);
            let skip_unchanged = entered
                && descriptor.member == 0
                && root.function.is_none()
                && root.loop_tail.is_none()
                && root.boxed_control.is_none()
                && !root.boxed_value
                && !compiled.resumes();
            // A side compiled apart keeps the window slots it names beyond
            // the tree's (the registers of the caller it returns into), so
            // its snapshots are as long as its own window.
            let externals = if descriptor.member > 0 && lowered.external.len() > externals.len() {
                &lowered.external[..]
            } else {
                externals
            };
            ExitRestore::new_skipping(
                lowered,
                &lowered.snapshots[descriptor.snapshot.index()],
                descriptor,
                register_count,
                externals,
                skip_unchanged,
                own,
                &mut scratch,
            )
        })
        .collect();
    if crate::jit::debug_env::trace_ir_opt() {
        eprintln!("jit-trace: restore plans from exit {first}: {restores:#?}");
    }
    restores
}

/// The lists [`ExitRestore::new_skipping`] fills for one exit, kept between
/// the exits of a tree: each plan then takes one allocation of the right
/// size per list it needs, where a list grown from nothing and cut to size
/// took several.
#[derive(Default)]
pub(crate) struct RestoreScratch {
    changes: Vec<(u16, RegisterAction)>,
    nils: Vec<u16>,
    floats: Vec<(u16, u32)>,
}

/// The entry inputs of a trace by register. [`ExitRestore::new_skipping`]
/// asks of every register of every exit whether the snapshot leaves it as the
/// trace's own entry input. Scanning the inputs for each register made the
/// plans of a tree cost exits x registers x inputs, on every install.
pub(crate) struct OwnInputs {
    /// The first input that names each register.
    first: Box<[Option<crate::jit_ir::trace::ValueId>]>,
    /// The inputs that name a register another input named before them.
    rest: Box<[(usize, crate::jit_ir::trace::ValueId)]>,
}

impl OwnInputs {
    pub(crate) fn of(ir: &TraceIr) -> Self {
        let inputs = &ir.entry.inputs;
        let len = inputs
            .iter()
            .map(|input| usize::from(input.register) + 1)
            .max()
            .unwrap_or(0);
        let mut first = vec![None; len];
        let mut rest = Vec::new();
        for input in inputs {
            let register = usize::from(input.register);
            match &mut first[register] {
                slot @ None => *slot = Some(input.value),
                Some(_) => rest.push((register, input.value)),
            }
        }
        Self {
            first: first.into_boxed_slice(),
            rest: rest.into_boxed_slice(),
        }
    }

    /// Whether an entry input carries `value` in `register`.
    #[inline]
    fn holds(&self, register: usize, value: crate::jit_ir::trace::ValueId) -> bool {
        match self.first.get(register) {
            Some(Some(first)) => {
                *first == value
                    || self
                        .rest
                        .iter()
                        .any(|(named, held)| *named == register && *held == value)
            }
            _ => false,
        }
    }
}

/// Turns the restore plans of the unwind exits of a function trace (see
/// [`TraceExitClass::Unwind`]) into what they are: the frame of an
/// activation stopped at its self call, whose callee's frame the restore
/// pushes next. The call's result window comes from the instruction in
/// `chunk`. `None` when an unwind exit names no call there, or the chunk is
/// gone.
fn plan_unwinds(
    compiled: &CompiledTrace,
    restores: &mut [ExitRestore],
    chunk: Option<&Chunk>,
) -> Option<()> {
    for (descriptor, restore) in compiled.exits().iter().zip(restores) {
        if descriptor.class != TraceExitClass::Unwind {
            continue;
        }
        let chunk = chunk?;
        let snapshot =
            &compiled.lowered_member(descriptor.member).snapshots[descriptor.snapshot.index()];
        let SnapshotPhase::Before { pc } = snapshot.phase else {
            return None;
        };
        let instruction = chunk.code.get(pc)?;
        let call = match instruction.opcode() {
            Ok(OpCode::Call | OpCode::CallOpen) => CallReturn {
                base: instruction.a(),
                results: instruction.c(),
            },
            Ok(OpCode::CallOne) => CallReturn {
                base: instruction.a(),
                results: 1,
            },
            // The call of the copies in the window after the control registers.
            Ok(OpCode::ForInCall) => CallReturn {
                base: instruction.a() + 3,
                results: instruction.b(),
            },
            _ => return None,
        };
        if !restore.inner.is_empty() {
            return None;
        }
        restore.as_unwind(pc, call);
    }
    Some(())
}

/// What an installed function trace needs at entry and exit (see
/// [`TraceIr::function`]).
#[derive(Debug)]
pub(crate) struct InstalledFunction {
    /// The type of the value a return gives.
    result: Option<TraceType>,
    /// Exit area slots per activation (see
    /// [`CompiledTrace::activation_stride`]).
    stride: usize,
    /// The window registers every member of the tree reads its self calls'
    /// callee from, which must hold the frame's closure at entry.
    callee_registers: Box<[u16]>,
    /// The name an activation's frame takes after a tail call of its own
    /// function, which every such call of the tree gives alike.
    tail_site: Option<crate::vm::ActiveCallSite>,
}

impl InstalledFunction {
    /// The function facts of the tree `ir` compiled as `compiled`, if it is a
    /// function trace.
    /// `None` for a loop trace, and for a tree whose tail calls of its own
    /// function name the frame differently (no single name restores).
    fn of(ir: &TraceIr, compiled: &CompiledTrace, chunk: &Chunk) -> Option<Self> {
        let function = ir.function.as_ref()?;
        let mut sites = std::iter::once(ir)
            .chain(ir.sides.iter().map(|side| &*side.ir))
            .filter_map(|member| match member.terminator {
                crate::jit_ir::trace::TraceTerminator::TailCall { pc, .. } => {
                    Some(crate::vm::debug_info::call_site_at(chunk, pc + 1))
                }
                _ => None,
            });
        let tail_site = sites.next().flatten();
        if sites.any(|site| site != tail_site) {
            return None;
        }
        let mut callee_registers: Vec<u16> = function.callee_registers.clone();
        for side in &ir.sides {
            for register in side
                .ir
                .function
                .iter()
                .flat_map(|function| &function.callee_registers)
            {
                if !callee_registers.contains(register) {
                    callee_registers.push(*register);
                }
            }
        }
        Some(Self {
            // The tree's, which a root that only tail calls takes from its
            // sides (see `plan_tree`).
            result: compiled
                .lowered_ir()
                .function
                .as_ref()
                .and_then(|lowered| lowered.result),
            stride: compiled.activation_stride()?,
            callee_registers: callee_registers.into_boxed_slice(),
            tail_site,
        })
    }
}

/// The most activations of a function trace one native run nests, the
/// entered one excluded. Each costs a frame of native stack (128 bytes for
/// `fib`, a few hundred for a larger body, so a few hundred kilobytes at
/// most: coroutines run on the VM's own heap stacks, never on a small native
/// one), and an exit at the deepest pushes one interpreter frame per
/// activation. A deeper recursion leaves at the cap, and the interpreter's
/// next self call enters again from there. A recursion that deep spends most
/// of its time near the leaves in either case, so the cap costs one exit and
/// one entry per thousand activations.
pub(crate) const MAX_NATIVE_ACTIVATIONS: usize = 1_000;

/// What one register becomes when a trace leaves.
#[derive(Clone, Copy, Debug, PartialEq)]
enum RegisterAction {
    /// The interpreter register already holds the right value.
    Keep,
    Nil,
    Const(TraceScalar),
    Float(u32),
    Integer(u32),
    /// A number whose subtype was known only at run time: the float in the
    /// first slot, and an integer of that value when the second slot holds a
    /// true boolean.
    Number(u32, u32),
    /// The integer whose value sits in this slot of the exit area as a float.
    IntegerOfFloat(u32),
    /// The float constant (as bits), or its integer when the slot holds a
    /// true boolean.
    NumberConst(u64, u32),
    /// The `int64_t` cdata whose bits sit in this slot of the exit area.
    Int64(u32),
    /// The `uint64_t` cdata whose bits sit in this slot of the exit area.
    UInt64(u32),
    Boolean(u32),
    /// The table whose lane id sits in this slot of the exit area.
    Table(u32),
    /// The closure whose lane id sits in this slot of the exit area.
    Closure(u32),
    /// The coroutine whose lane id sits in this slot of the exit area.
    Thread(u32),
    /// The userdata whose lane id sits in this slot of the exit area.
    Userdata(u32),
    /// The string whose lane id sits in this slot of the exit area.
    String(u32),
    /// The borrowed string whose slot address sits in this slot of the exit
    /// area. Restoring clones the string out of the table slot, which its
    /// owner still holds (see [`clone_borrowed_string`]).
    BorrowedString(u32),
    /// The string constant in this root slot.
    Root(u32),
    /// The virtual table this index of [`ExitRestore::virtuals`] builds.
    Virtual(u32),
    /// The substring `string.sub (string, first, last)` of the string whose
    /// lane id sits in slot `string` of the exit area.
    Slice {
        string: u32,
        first: SliceIndex,
        last: SliceIndex,
    },
}

/// Where an exit reads an index of a substring from. The trace checked that
/// it lies strictly between -2^31 and 2^31, so truncation toward zero is the
/// conversion every profile makes.
#[derive(Clone, Copy, Debug, PartialEq)]
enum SliceIndex {
    Float(u32),
    Integer(u32),
    Number(f64),
}

impl SliceIndex {
    #[inline]
    #[allow(clippy::cast_possible_truncation)] // checked by the trace to be small
    fn read(self, area: &[i64]) -> isize {
        match self {
            Self::Float(slot) => {
                f64::from_bits(area[slot as usize].cast_unsigned()).trunc() as isize
            }
            Self::Integer(slot) => {
                let index = area[slot as usize];
                isize::try_from(index).unwrap_or(if index < 0 { isize::MIN } else { isize::MAX })
            }
            Self::Number(number) => number.trunc() as isize,
        }
    }
}

/// The substring `string.sub (text, first, last)` makes, for indices already
/// converted to integers.
fn slice_value(text: &Value, first: isize, last: isize) -> Value {
    let Value::String(text) = text else {
        unreachable!("a substring recipe names a string");
    };
    let bytes = text.as_bytes();
    let length = isize::try_from(bytes.len()).unwrap_or(isize::MAX);
    // The interpreter's own clamp, so the two cannot disagree.
    match substring_range(length, first, last) {
        Some(range) => Value::from(&bytes[range]),
        None => Value::from(""),
    }
}

/// How an exit builds one virtual table (see
/// [`oximond::jit_ir::trace::VirtualTable`]).
#[derive(Debug)]
pub(crate) struct VirtualPlan {
    /// The shape of a shaped table, `None` for a plain one.
    pub(crate) shape: Option<Arc<TableShape>>,
    /// The values at keys 1, 2, ...
    array: Box<[RegisterAction]>,
    /// The other fields, by key.
    fields: Box<[(VirtualKey, RegisterAction)]>,
    /// The string of each field's key that is one, made once: an exit that
    /// builds the table stores under a copy of it.
    strings: Box<[Option<crate::string::LuaString>]>,
    /// The `charge_elided_table` entry count to take back when the trace
    /// already charged this table as phantom debt.
    refund: Option<usize>,
    /// The metatable to set once the contents are stored: a table lane.
    metatable: Option<RegisterAction>,
}

/// The restore plan of one inlined frame of an exit.
#[derive(Debug)]
struct InnerPlan {
    /// The pinned closure the frame runs, as a lane of the bank.
    closure: u32,
    /// For a frame that runs whichever closure of the pinned one's prototype
    /// the call had: the exit area slot holding the lane of the closure of
    /// this run, which the frame is rebuilt with instead.
    live_closure: Option<u32>,
    result_base: u8,
    results: u8,
    tail_calls: u32,
    tail_site: Option<(u32, usize)>,
    protected: crate::jit_ir::trace::Protected,
    entry: crate::jit_ir::trace::Entry,
    /// The registers the restore writes, by position. A frame pushed afresh
    /// starts with every register nil, so its nil writes are left out; the
    /// root frame's caller, which the interpreter already holds, keeps its
    /// nil writes, and a register it leaves as it is has no entry.
    writes: Box<[(u16, RegisterAction)]>,
    pc: usize,
    last_interpreted_pc: Option<usize>,
}

/// The checked restore plan of one exit.
#[derive(Debug)]
pub(crate) struct ExitRestore {
    /// Where the root frame resumes. Inside an inlined call it stops at the
    /// call.
    pc: usize,
    last_interpreted_pc: Option<usize>,
    /// The registers the exit writes, by position: most exits leave most of
    /// the window as it is, and the restore touches only these. The writes
    /// that most exits make, of nil and of a float from the exit area, are
    /// kept apart in `nils` and `floats`, which take a tight loop each.
    changes: Box<[(u16, RegisterAction)]>,
    /// The registers the exit sets to nil.
    nils: Box<[u16]>,
    /// The registers the exit sets to a float, and the slot of the exit area
    /// that holds it.
    floats: Box<[(u16, u32)]>,
    /// The upvalue cells the exit writes. A cell the snapshot leaves
    /// unchanged has no entry.
    upvalues: Box<[(CellTarget, RegisterAction)]>,
    /// The registers of the frame below the root frame (its caller) the exit
    /// writes, from the window slots that stand for them
    /// ([`ExternalSlot::CallerRegister`]).
    callers: Box<[(u16, RegisterAction)]>,
    /// The inlined frames to push above the root frame, outermost first.
    inner: Box<[InnerPlan]>,
    /// Whether a side trace may start where this exit leaves inside inlined
    /// frames (see "Sides from exits inside inlined frames" in the module
    /// docs): every frame of `inner` is a plain call of the frame below it
    /// that keeps a fixed number of results, runs the pinned closure itself,
    /// made no tail call, and there are at most [`MAX_SEED_FRAMES`] of them.
    /// False for an exit that pushes no frame.
    seedable: bool,
    /// The virtual tables the exit builds before it writes any register,
    /// numbered by the order their first mention in the snapshot.
    pub(crate) virtuals: Box<[VirtualPlan]>,
    /// Where the innermost frame resumes, which the VM reports.
    resume_pc: usize,
    resume_last: Option<usize>,
    /// For a trace that loops through tail calls of the root frame: what
    /// those calls left on the frame, and where the tail call count is.
    root_tail: Option<(RootTail, RegisterAction)>,
    /// For the unwind exit of a function trace's self call: the call's
    /// result window, which the frame of the callee's activation returns to.
    /// The frame itself stops at the call (see [`Self::as_unwind`]).
    call: Option<CallReturn>,
    /// For a function trace whose function tail calls itself: where the
    /// activation's tail call count is (see [`Self::apply_function_count`]).
    function_count: Option<RegisterAction>,
}

impl ExitRestore {
    /// Leaves on `frame` the tail call count of the function trace's
    /// activation, from `area`, as the interpreter's in-place tail calls
    /// would have: a count the run changed renames the frame after the
    /// function's tail call (`site`, the same at every tail call of the
    /// tree). An activation that made no tail call keeps the frame's name.
    #[inline]
    fn apply_function_count(
        &self,
        frame: &mut FrameMut<'_>,
        area: &[i64],
        site: Option<&crate::vm::ActiveCallSite>,
    ) {
        let count = match self.function_count {
            Some(RegisterAction::Integer(slot)) => area[slot as usize],
            Some(RegisterAction::Const(TraceScalar::Integer(count))) => count,
            _ => return,
        };
        let count = u32::try_from(count).unwrap_or(u32::MAX);
        if count != frame.tail_calls {
            frame.clear_plain_mark();
            frame.tail_calls = count;
            frame.call_site = site.cloned().map(crate::vm::FrameSite::boxed);
        }
    }

    /// Makes this plan, of the `Before` snapshot of the self call at `pc`
    /// that `call` returns to, the restore of an activation whose callee
    /// left: the frame stands after the call, which it made last, with the
    /// callee's frame above it.
    fn as_unwind(&mut self, pc: usize, call: CallReturn) {
        self.pc = pc + 1;
        self.last_interpreted_pc = Some(pc);
        self.resume_pc = pc + 1;
        self.resume_last = Some(pc);
        self.call = Some(call);
    }
    /// Plans the restore of `snapshot` of `ir` for an exit that spills values
    /// as `descriptor` says. `None` when the snapshot does not fit the root
    /// frame's `register_count` registers and `externals` or an inlined
    /// frame of `ir`, names a value the exit does not spill, or writes a
    /// global.
    #[cfg(test)]
    pub(crate) fn new(
        ir: &TraceIr,
        snapshot: &Snapshot,
        descriptor: &ExitDescriptor,
        register_count: usize,
        externals: &[ExternalSlot],
    ) -> Option<Self> {
        Self::new_skipping(
            ir,
            snapshot,
            descriptor,
            register_count,
            externals,
            false,
            &OwnInputs::of(ir),
            &mut RestoreScratch::default(),
        )
    }

    /// [`Self::new`], leaving alone a register or cell that the snapshot
    /// restores to the value it had at entry when `skip_unchanged` says the
    /// entry state is the frame's (the root of a tree that neither loops
    /// through tail calls, nor is a function trace, nor resumes a coroutine):
    /// native code keeps its values to itself, so the frame still holds the
    /// register as it was entered.
    pub(crate) fn new_skipping(
        ir: &TraceIr,
        snapshot: &Snapshot,
        descriptor: &ExitDescriptor,
        register_count: usize,
        externals: &[ExternalSlot],
        skip_unchanged: bool,
        own: &OwnInputs,
        scratch: &mut RestoreScratch,
    ) -> Option<Self> {
        let own_input = |register: usize, entry: &RegisterEntry| matches!(entry, RegisterEntry::Value(value) if own.holds(register, *value));
        let unchanged =
            |register: usize, entry: &RegisterEntry| skip_unchanged && own_input(register, entry);
        // The compile leaves out of the exit a value the snapshot names only
        // in registers that keep the run's entry state, as the member's own
        // entry input there (`kept_by_frame` in `jit/trace/compile.rs`): the
        // frame holds it already.
        let kept_by_frame = |register: usize, entry: &RegisterEntry| {
            own_input(register, entry)
                && matches!(entry, RegisterEntry::Value(value)
                    if descriptor
                        .slots
                        .binary_search_by_key(value, |slot| slot.value)
                        .is_err())
        };
        let (root, inner) = snapshot.frames.split_first()?;
        if !ir.frames.get(root.frame.index())?.is_root()
            || root.registers.len() != register_count + externals.len()
        {
            return None;
        }
        let value_action = |entry: &RegisterEntry| -> Option<RegisterAction> {
            Some(match entry {
                // Handled by `action`: a recipe never holds one.
                RegisterEntry::Virtual(_) | RegisterEntry::Slice(_) => return None,
                RegisterEntry::Unchanged => RegisterAction::Keep,
                RegisterEntry::Const(scalar) if scalar.ty().is_opaque() => return None,
                RegisterEntry::Const(scalar) => RegisterAction::Const(*scalar),
                RegisterEntry::Root(slot) => RegisterAction::Root(*slot),
                RegisterEntry::Value(value) => {
                    let index = descriptor
                        .slots
                        .binary_search_by_key(value, |slot| slot.value)
                        .ok()?;
                    let slot = descriptor.slots[index];
                    match slot.ty {
                        TraceType::Nil => RegisterAction::Nil,
                        TraceType::Float => RegisterAction::Float(slot.slot),
                        TraceType::Integer => RegisterAction::Integer(slot.slot),
                        TraceType::Int64 => RegisterAction::Int64(slot.slot),
                        TraceType::UInt64 => RegisterAction::UInt64(slot.slot),
                        TraceType::Boolean => RegisterAction::Boolean(slot.slot),
                        TraceType::Table => RegisterAction::Table(slot.slot),
                        TraceType::Closure => RegisterAction::Closure(slot.slot),
                        TraceType::Thread => RegisterAction::Thread(slot.slot),
                        TraceType::Userdata => RegisterAction::Userdata(slot.slot),
                        TraceType::String => RegisterAction::String(slot.slot),
                        TraceType::StringRef => RegisterAction::BorrowedString(slot.slot),
                        TraceType::Any => return None,
                    }
                }
            })
        };
        // The virtual tables the frames name, in order of first mention, and
        // then the ones those hold.
        let mut virtual_index: Vec<Option<u32>> = vec![None; snapshot.virtuals.len()];
        let mut order: Vec<usize> = Vec::new();
        let mut mention = |id: usize, order: &mut Vec<usize>| -> Option<()> {
            let slot = virtual_index.get_mut(id)?;
            if slot.is_none() {
                *slot = Some(u32::try_from(order.len()).ok()?);
                order.push(id);
            }
            Some(())
        };
        // A snapshot without recipes names no virtual table: an entry that
        // does all the same finds none below, and the plan fails there.
        if !snapshot.virtuals.is_empty() {
            for entry in snapshot.frames.iter().flat_map(|frame| &frame.registers) {
                if let RegisterEntry::Virtual(id) = entry {
                    mention(id.index(), &mut order)?;
                }
            }
        }
        let mut next = 0;
        while next < order.len() {
            for entry in snapshot.virtuals[order[next]].entries() {
                if let RegisterEntry::Virtual(id) = entry {
                    mention(id.index(), &mut order)?;
                }
            }
            next += 1;
        }
        let mut virtuals = Vec::with_capacity(order.len());
        for id in order {
            let recipe = &snapshot.virtuals[id];
            let shape = virtual_shape(&recipe.alloc);
            if matches!(recipe.alloc, VirtualAlloc::Shaped { .. }) && shape.is_none() {
                return None;
            }
            // A table inside a table is built before any is filled, so its
            // handle is there when the outer one stores it.
            let stored = |entry: &RegisterEntry| match entry {
                RegisterEntry::Virtual(inner) => Some(RegisterAction::Virtual(
                    virtual_index.get(inner.index()).copied().flatten()?,
                )),
                entry => match value_action(entry)? {
                    RegisterAction::Keep => None,
                    action => Some(action),
                },
            };
            virtuals.push(VirtualPlan {
                shape,
                array: recipe
                    .array
                    .iter()
                    .map(stored)
                    .collect::<Option<Box<[_]>>>()?,
                fields: recipe
                    .fields
                    .iter()
                    .map(|(key, entry)| Some((key.clone(), stored(entry)?)))
                    .collect::<Option<Box<[_]>>>()?,
                strings: recipe
                    .fields
                    .iter()
                    .map(|(key, _)| match key {
                        VirtualKey::String(bytes) => {
                            Some(crate::string::LuaString::from(&bytes[..]))
                        }
                        _ => None,
                    })
                    .collect(),
                refund: recipe.phantom_entries,
                metatable: match &recipe.metatable {
                    Some(entry) => Some(value_action(entry)?),
                    None => None,
                },
            });
        }
        let action = |entry: &RegisterEntry| -> Option<RegisterAction> {
            match entry {
                RegisterEntry::Virtual(id) => Some(RegisterAction::Virtual(
                    virtual_index.get(id.index()).copied().flatten()?,
                )),
                RegisterEntry::Slice(id) => {
                    let recipe = snapshot.slices.get(id.index())?;
                    let RegisterAction::String(string) = value_action(&recipe.string)? else {
                        return None;
                    };
                    let index = |entry: &RegisterEntry| match value_action(entry)? {
                        RegisterAction::Float(slot) => Some(SliceIndex::Float(slot)),
                        RegisterAction::Integer(slot) => Some(SliceIndex::Integer(slot)),
                        RegisterAction::Const(TraceScalar::Float(number)) => {
                            Some(SliceIndex::Number(number))
                        }
                        #[allow(clippy::cast_precision_loss)] // a small index
                        RegisterAction::Const(TraceScalar::Integer(number)) => {
                            Some(SliceIndex::Number(number as f64))
                        }
                        _ => None,
                    };
                    Some(RegisterAction::Slice {
                        string,
                        first: index(&recipe.first)?,
                        last: index(&recipe.last)?,
                    })
                }
                entry => value_action(entry),
            }
        };
        let slot_of = |value: &crate::jit_ir::trace::ValueId| {
            let index = descriptor
                .slots
                .binary_search_by_key(value, |slot| slot.value)
                .ok()?;
            Some(descriptor.slots[index].slot)
        };
        // A register that holds a number of run-time subtype restores as a
        // float or as the integer of that float.
        let number_action = |entry: &RegisterEntry,
                             tag: &RegisterEntry|
         -> Option<RegisterAction> {
            Some(match (entry, tag) {
                (RegisterEntry::Value(value), RegisterEntry::Value(tag)) => {
                    RegisterAction::Number(slot_of(value)?, slot_of(tag)?)
                }
                (RegisterEntry::Value(value), RegisterEntry::Const(TraceScalar::Boolean(true))) => {
                    RegisterAction::IntegerOfFloat(slot_of(value)?)
                }
                (
                    RegisterEntry::Value(value),
                    RegisterEntry::Const(TraceScalar::Boolean(false)),
                ) => RegisterAction::Float(slot_of(value)?),
                (RegisterEntry::Const(TraceScalar::Float(constant)), RegisterEntry::Value(tag)) => {
                    RegisterAction::NumberConst(constant.to_bits(), slot_of(tag)?)
                }
                (
                    RegisterEntry::Const(TraceScalar::Float(constant)),
                    RegisterEntry::Const(TraceScalar::Boolean(integer)),
                ) => RegisterAction::Const(if *integer {
                    #[allow(clippy::cast_possible_truncation)]
                    TraceScalar::Integer(*constant as i64)
                } else {
                    TraceScalar::Float(*constant)
                }),
                _ => return None,
            })
        };
        // Only the registers the exit writes get an entry. A snapshot names
        // every register of the window and leaves most of them unchanged, and
        // a tree plans every exit again each time it grows: a wide frame paid
        // for an action per register per exit, kept or not.
        let RestoreScratch {
            changes,
            nils,
            floats,
        } = scratch;
        changes.clear();
        nils.clear();
        floats.clear();
        let no_numbers = snapshot.numbers.is_empty();
        if register_count > usize::from(u16::MAX) + 1 {
            return None;
        }
        for (register, entry) in root.registers.range(0, register_count).enumerate() {
            if no_numbers && matches!(entry, RegisterEntry::Unchanged) {
                continue;
            }
            let tag = if no_numbers {
                None
            } else {
                snapshot.number_tag(register)
            };
            let planned = match (tag, entry) {
                (Some(tag), _) => number_action(entry, &tag)?,
                (None, _) if unchanged(register, entry) => continue,
                // A value the exit spills is found by the one search of
                // its slots; only one it does not spill is asked about again.
                (None, RegisterEntry::Value(_)) => match action(entry) {
                    Some(planned) => planned,
                    None if kept_by_frame(register, entry) => continue,
                    None => return None,
                },
                (None, _) => action(entry)?,
            };
            #[allow(clippy::cast_possible_truncation)] // checked above
            let register = register as u16;
            match planned {
                RegisterAction::Keep => {}
                RegisterAction::Nil | RegisterAction::Const(TraceScalar::Nil) => {
                    nils.push(register);
                }
                RegisterAction::Float(slot) => floats.push((register, slot)),
                planned => changes.push((register, planned)),
            }
        }
        let mut upvalues = Vec::new();
        let mut callers = Vec::new();
        let mut tail_count = None;
        for (position, (slot, entry)) in externals
            .iter()
            .zip(root.registers.range(register_count, root.registers.len()))
            .enumerate()
        {
            // What the trace left alone needs no write, which is most of a
            // window of many globals. The tail call count is read even so.
            if matches!(entry, RegisterEntry::Unchanged) && !matches!(slot, ExternalSlot::TailCalls)
            {
                continue;
            }
            // A cell or global never holds a virtual table or a substring.
            if matches!(entry, RegisterEntry::Virtual(_) | RegisterEntry::Slice(_)) {
                return None;
            }
            // A cell the run read and left as it was needs no write.
            if matches!(
                slot,
                ExternalSlot::Upvalue(_) | ExternalSlot::ClosureUpvalue { .. }
            ) && unchanged(register_count + position, entry)
            {
                continue;
            }
            let target = match slot {
                ExternalSlot::CallerRegister(register) => {
                    // A register the run read and left as it was needs no
                    // write, which `unchanged` says for a loop's invariant.
                    if unchanged(register_count + position, entry) {
                        continue;
                    }
                    match action(entry)? {
                        RegisterAction::Keep => {}
                        action => callers.push((*register, action)),
                    }
                    continue;
                }
                ExternalSlot::Upvalue(index) => CellTarget::Frame(*index),
                ExternalSlot::ClosureUpvalue { closure, index } => CellTarget::Closure {
                    lane: *closure,
                    index: *index,
                },
                ExternalSlot::TailCalls => {
                    tail_count = Some(action(entry)?);
                    continue;
                }
                ExternalSlot::Global(_)
                | ExternalSlot::ClosureGlobal { .. }
                | ExternalSlot::CallerGlobal(_)
                | ExternalSlot::UpvalueField { .. }
                | ExternalSlot::ClosureUpvalueField { .. }
                | ExternalSlot::RootClosure
                | ExternalSlot::VarargCount
                | ExternalSlot::Vararg(_)
                | ExternalSlot::Environment => {
                    // A global is never written: only `Keep` is legal.
                    if action(entry)? != RegisterAction::Keep {
                        return None;
                    }
                    continue;
                }
            };
            match action(entry)? {
                RegisterAction::Keep => {}
                action => upvalues.push((target, action)),
            }
        }
        let resume = snapshot.resume_point();
        // Each frame stops at the call that made the next, and the innermost
        // resumes at the snapshot's phase.
        let call_of = |frame: FrameId| ir.frames.get(frame.index())?.inlined;
        let stopped_at = |next: Option<&crate::jit_ir::trace::SnapshotFrame>| match next {
            Some(next) => call_of(next.frame).map(|call| (call.call_pc + 1, Some(call.call_pc))),
            None => Some((resume.pc, resume.last_interpreted_pc)),
        };
        let (pc, last_interpreted_pc) = stopped_at(inner.first())?;
        let mut function_count = None;
        let root_tail = match (RootTail::of(ir, root.frame), tail_count) {
            // An activation always follows a counted tail call.
            (Some(tail), Some(RegisterAction::Keep)) if tail.closure.is_some() => return None,
            (Some(tail), Some(count)) => Some((tail, count)),
            (None, None) => None,
            // A function trace's activation counts its own tail calls.
            (None, Some(count)) if ir.function.is_some() => {
                function_count = Some(count);
                None
            }
            _ => return None,
        };
        let mut plans = Vec::with_capacity(inner.len());
        for (position, frame) in inner.iter().enumerate() {
            let call = call_of(frame.frame)?;
            let info = ir.frames.get(frame.frame.index())?;
            if frame.registers.len() != usize::from(info.register_count) {
                return None;
            }
            let (pc, last_interpreted_pc) = stopped_at(inner.get(position + 1))?;
            let caller = matches!(call.entry, FrameEntry::Caller { .. });
            let mut registers = frame
                .registers
                .iter()
                .map(|entry| match entry {
                    // The root frame's caller is the interpreter's own frame:
                    // a register the trace never wrote is as it holds it. Any
                    // other inlined frame has no interpreter register to keep.
                    RegisterEntry::Unchanged if caller => Some(RegisterAction::Keep),
                    RegisterEntry::Unchanged => None,
                    entry => action(entry),
                })
                .collect::<Option<Vec<_>>>()?;
            // The last register of a frame that runs the closure of this
            // run is that closure, not a register of the chunk.
            let live_closure = if call.dynamic {
                match registers.pop()? {
                    RegisterAction::Closure(slot) => Some(slot),
                    _ => return None,
                }
            } else {
                None
            };
            let mut writes = Vec::new();
            for (register, action) in registers.into_iter().enumerate() {
                let fresh_nil = !caller
                    && matches!(
                        action,
                        RegisterAction::Nil | RegisterAction::Const(TraceScalar::Nil)
                    );
                if action != RegisterAction::Keep && !fresh_nil {
                    writes.push((u16::try_from(register).ok()?, action));
                }
            }
            plans.push(InnerPlan {
                closure: call.closure,
                live_closure,
                result_base: call.result_base,
                results: call.results,
                tail_calls: call.tail_calls,
                tail_site: call.tail_site,
                protected: call.protected,
                entry: call.entry,
                writes: writes.into_boxed_slice(),
                pc,
                last_interpreted_pc,
            });
        }
        let seedable = !plans.is_empty()
            && plans.len() <= MAX_SEED_FRAMES
            && ir.function.is_none()
            && root.frame == FrameId(0)
            && root_tail.is_none()
            && virtuals.is_empty()
            && plans.iter().all(|plan| {
                plan.entry == FrameEntry::Call
                    && plan.protected == crate::jit_ir::trace::Protected::No
                    && plan.results > 0
                    && plan.tail_calls == 0
                    && plan.tail_site.is_none()
            });
        Some(Self {
            pc,
            last_interpreted_pc,
            changes: changes.as_slice().into(),
            nils: nils.as_slice().into(),
            floats: floats.as_slice().into(),
            upvalues: upvalues.into_boxed_slice(),
            callers: callers.into_boxed_slice(),
            inner: plans.into_boxed_slice(),
            seedable,
            virtuals: virtuals.into_boxed_slice(),
            resume_pc: resume.pc,
            resume_last: resume.last_interpreted_pc,
            root_tail,
            call: None,
            function_count,
        })
    }

    /// Builds the virtual tables of the exit: every table is allocated first,
    /// then each is filled, and only then may [`Self::apply`] write a
    /// register. Nothing here reaches a safepoint, so the collector never sees
    /// a partial table, and a partial table needs no root. A table whose
    /// recipe says the trace already charged it as phantom debt gives that
    /// charge back, since the allocation counts instead. The result is
    /// indexed as [`RegisterAction::Virtual`] is.
    ///
    /// # Errors
    ///
    /// The runtime's error when an allocation or a store fails. The tables
    /// made so far stay in the arena, unreachable, for the next collection;
    /// no register has been written.
    #[inline]
    pub(crate) fn materialize(
        &self,
        area: &[i64],
        bank: &[Value],
        roots: &[Value],
        runtime: &mut Runtime,
    ) -> Result<Vec<TableHandle>, RuntimeError> {
        if self.virtuals.is_empty() {
            return Ok(Vec::new());
        }
        self.materialize_virtuals(area, bank, roots, runtime)
    }

    /// [`Self::materialize`] for an exit that builds tables.
    #[inline(never)]
    fn materialize_virtuals(
        &self,
        area: &[i64],
        bank: &[Value],
        roots: &[Value],
        runtime: &mut Runtime,
    ) -> Result<Vec<TableHandle>, RuntimeError> {
        let mut handles = Vec::with_capacity(self.virtuals.len());
        for plan in &*self.virtuals {
            handles.push(allocate_virtual_sized(
                runtime,
                plan.shape.as_ref(),
                plan.array.len(),
            )?);
            if let Some(entries) = plan.refund {
                runtime.refund_elided_table(entries);
            }
        }
        for (plan, handle) in self.virtuals.iter().zip(&handles) {
            // Every table is allocated, so a table inside a table has its
            // handle.
            let value = |action: RegisterAction| {
                action_value(action, area, bank, roots, &handles).unwrap_or(Value::Nil)
            };
            let keys = plan.fields.iter().zip(&*plan.strings);
            fill_virtual_keyed(
                runtime,
                *handle,
                plan.array.iter().map(|action| value(*action)),
                keys.map(|((key, action), string)| {
                    let key = match string {
                        Some(string) => VirtualFieldKey::Made(string),
                        None => VirtualFieldKey::Recipe(key),
                    };
                    (key, value(*action))
                }),
            )?;
            if let Some(action) = plan.metatable {
                match action_value(action, area, bank, roots, &[]) {
                    Some(Value::Table(metatable)) => {
                        runtime.set_table_metatable(*handle, Some(metatable))?;
                    }
                    _ => {
                        return Err(RuntimeError::message(
                            "a virtual table's metatable is not a table",
                        ));
                    }
                }
            }
        }
        Ok(handles)
    }

    /// Writes the plan onto `frame`, reading spilled values from `area`, the
    /// lanes of tables and closures from `bank` and root constants from
    /// `roots`. Upvalue cells go through `runtime`. The inlined frames
    /// follow through [`Self::push_inner`]. A plan with a root tail state
    /// goes through [`Self::apply_cells`], [`Self::apply_root_tail`] and
    /// [`Self::apply_registers`] instead, since it may resize the frame.
    #[inline]
    pub(crate) fn apply(
        &self,
        frame: &mut FrameMut<'_>,
        area: &[i64],
        bank: &[Value],
        roots: &[Value],
        virtuals: &[TableHandle],
        runtime: &mut Runtime,
    ) {
        debug_assert!(self.root_tail.is_none());
        self.apply_cells(frame, area, bank, roots, virtuals, runtime);
        self.apply_registers(frame, area, bank, roots, virtuals);
    }

    /// Whether the plan restores what tail calls of the root frame left.
    #[inline]
    pub(crate) const fn has_root_tail(&self) -> bool {
        self.root_tail.is_some()
    }

    /// Leaves on the top frame at `index` what the root frame's tail calls
    /// did in the run (see [`apply_root_tail`]): its closure, its tail call
    /// count and its name. Runs after [`Self::apply_cells`], whose upvalue
    /// writes name the cells of the closure the frame was entered with, and
    /// before [`Self::apply_registers`].
    pub(crate) fn apply_root_tail(
        &self,
        frames: &mut CallStack,
        index: usize,
        area: &[i64],
        bank: &[Value],
        runtime: &Runtime,
    ) {
        let Some((tail, count)) = self.root_tail else {
            return;
        };
        let count = match count {
            RegisterAction::Integer(slot) => area[slot as usize],
            RegisterAction::Const(TraceScalar::Integer(count)) => count,
            // The count the frame has.
            _ => return,
        };
        let applied = apply_root_tail(frames, index, tail, count, bank, runtime);
        debug_assert!(applied.is_ok(), "an entered trace's pinned closures exist");
    }

    /// The upvalue cell writes of the plan.
    #[inline]
    pub(crate) fn apply_cells(
        &self,
        frame: &FrameMut<'_>,
        area: &[i64],
        bank: &[Value],
        roots: &[Value],
        virtuals: &[TableHandle],
        runtime: &mut Runtime,
    ) {
        for (target, action) in &*self.upvalues {
            let cell = match *target {
                CellTarget::Frame(index) => frame.upvalues.get(usize::from(index)).copied(),
                CellTarget::Closure { lane, index } => pinned_closure(bank, lane)
                    .and_then(|handle| runtime.closure_cell(handle, usize::from(index)).ok())
                    .flatten(),
                CellTarget::Caller(_) => None,
            };
            // A table the cell holds already: the usual case of a table
            // the run read from it, and no write to make.
            if let (RegisterAction::Table(slot), Some(cell)) = (*action, cell)
                && let Some(Value::Table(held)) = area
                    .get(slot as usize)
                    .and_then(|lane| bank.get(lane.cast_unsigned() as usize))
                && matches!(runtime.cell_ref(cell), Ok(Value::Table(current)) if current == held)
            {
                continue;
            }
            let Some(value) = action_value(*action, area, bank, roots, virtuals) else {
                continue;
            };
            if let Some(cell) = cell {
                // A cell the run read and left as it was (the snapshot names
                // the value it loaded) needs no write, and no write barrier.
                if runtime
                    .cell_ref(cell)
                    .is_ok_and(|current| same_value(current, &value))
                {
                    continue;
                }
                let written = runtime.cell_set(cell, value);
                debug_assert!(written.is_ok(), "an entered trace's cells exist");
            }
        }
    }

    /// Whether the plan writes registers of the root frame's caller.
    #[inline]
    pub(crate) const fn writes_caller(&self) -> bool {
        !self.callers.is_empty()
    }

    /// Whether a guard that leaves through this exit can enter a side with
    /// the virtual tables the exit builds made in the run
    /// ([`TraceTables::materialize_exit`]): the exit resumes in the root
    /// frame as the trace entered it, and builds no more tables than a
    /// helper's result words name lanes.
    pub(crate) fn bridges_to_side(&self) -> bool {
        self.inner.is_empty()
            && self.root_tail.is_none()
            && self.callers.is_empty()
            && self.call.is_none()
            && self.virtuals.len() <= crate::jit_ir::trace::MAX_HELPER_RESULTS
    }

    /// Charges again what [`Self::materialize`] took back for its tables.
    fn charge_virtuals(&self, runtime: &mut Runtime) {
        for plan in &*self.virtuals {
            if let Some(entries) = plan.refund {
                runtime.charge_elided_table(entries);
            }
        }
    }

    /// Writes the registers of the frame below the root frame at `index`,
    /// the caller's, that the window slots of the plan stand for.
    #[inline]
    pub(crate) fn apply_callers(
        &self,
        frames: &mut CallStack,
        index: usize,
        area: &[i64],
        bank: &[Value],
        roots: &[Value],
        virtuals: &[TableHandle],
    ) {
        let Some(below) = index.checked_sub(1) else {
            return;
        };
        let caller = frames.at_mut(below);
        for &(register, action) in &*self.callers {
            if let Some(slot) = caller.registers.get_mut(usize::from(register)) {
                write_action(slot, action, area, bank, roots, virtuals);
            }
        }
    }

    /// The register writes of the plan, and where the root frame resumes.
    #[inline]
    pub(crate) fn apply_registers(
        &self,
        frame: &mut FrameMut<'_>,
        area: &[i64],
        bank: &[Value],
        roots: &[Value],
        virtuals: &[TableHandle],
    ) {
        for &register in &*self.nils {
            if let Some(register) = frame.registers.get_mut(usize::from(register)) {
                *register = Value::Nil;
            }
        }
        for &(register, slot) in &*self.floats {
            if let (Some(register), Some(&bits)) = (
                frame.registers.get_mut(usize::from(register)),
                area.get(slot as usize),
            ) {
                *register = Value::Number(f64::from_bits(bits.cast_unsigned()));
            }
        }
        for &(register, action) in &*self.changes {
            if let Some(register) = frame.registers.get_mut(usize::from(register)) {
                write_action(register, action, area, bank, roots, virtuals);
            }
        }
        frame.pc = self.pc;
        frame.last_interpreted_pc = self.last_interpreted_pc.into();
    }

    /// Pushes the inlined frames of the exit above the root frame: the
    /// activation records the interpreter's calls would have made.
    #[inline]
    pub(crate) fn push_inner(
        &self,
        frames: &mut CallStack,
        area: &[i64],
        bank: &[Value],
        roots: &[Value],
        virtuals: &[TableHandle],
        runtime: &Runtime,
    ) {
        if !self.inner.is_empty() {
            self.push_inner_frames(frames, area, bank, roots, virtuals, runtime);
        }
    }

    /// [`Self::push_inner`] for an exit inside inlined calls.
    #[inline(never)]
    fn push_inner_frames(
        &self,
        frames: &mut CallStack,
        area: &[i64],
        bank: &[Value],
        roots: &[Value],
        virtuals: &[TableHandle],
        runtime: &Runtime,
    ) {
        for plan in &*self.inner {
            // The root frame has returned: its caller, which the
            // interpreter already holds, resumes where the plan says, with
            // the results of the call in its registers, and the root frame
            // goes as a `Return` would take it.
            if matches!(plan.entry, FrameEntry::Caller { .. }) {
                debug_assert!(frames.len() >= 2, "a returned-into caller has a frame");
                let below = frames.len() - 2;
                {
                    let mut caller = frames.at_mut(below);
                    for &(register, action) in &*plan.writes {
                        if let Some(register) = caller.registers.get_mut(usize::from(register)) {
                            write_action(register, action, area, bank, roots, virtuals);
                        }
                    }
                    caller.pc = plan.pc;
                    caller.last_interpreted_pc = plan.last_interpreted_pc.into();
                    caller.clear_open_results();
                }
                frames.retire_top();
                continue;
            }
            let handle = match plan.live_closure {
                Some(slot) => match &bank[area[slot as usize].cast_unsigned() as usize] {
                    Value::Closure(handle) => *handle,
                    _ => unreachable!("a frame's closure lane holds a closure"),
                },
                None => pinned_closure(bank, plan.closure)
                    .expect("an entered trace's pinned closures are in the bank"),
            };
            let index = push_protected_frame(
                frames,
                runtime,
                bank,
                handle,
                CallReturn {
                    base: plan.result_base,
                    results: plan.results,
                },
                (plan.protected, plan.entry),
            )
            .expect("an entered trace's pinned closures exist");
            let mut frame = frames.at_mut(index);
            for &(register, action) in &*plan.writes {
                if let Some(register) = frame.registers.get_mut(usize::from(register)) {
                    write_action(register, action, area, bank, roots, virtuals);
                }
            }
            frame.pc = plan.pc;
            frame.last_interpreted_pc = plan.last_interpreted_pc.into();
            restore_tail_state(&mut frame, (plan.tail_calls, plan.tail_site), bank, runtime);
        }
    }
}

/// The string a lane names: a lane of the run's bank, or a constant of the
/// trace's roots (a lane from [`oximond::jit_ir::trace::ROOT_STRING_LANE_BASE`]
/// on, which a trace holds when it carries a string constant).
#[inline]
fn string_lane<'a>(lane: u64, bank: &'a [Value], roots: &'a [Value]) -> &'a Value {
    let index = |lane: u64| usize::try_from(lane).expect("a lane fits the address space");
    match lane.checked_sub(crate::jit_ir::trace::ROOT_STRING_LANE_BASE) {
        Some(slot) => &roots[index(slot)],
        None => &bank[index(lane)],
    }
}

/// The number whose float bits are `bits`: that float, or when `integer` the
/// integer of the same value (the trace keeps such integers within 2^53, so
/// the conversion is exact).
#[inline]
fn number_value(bits: i64, integer: bool) -> Value {
    let float = f64::from_bits(bits.cast_unsigned());
    if integer {
        #[allow(clippy::cast_possible_truncation)]
        Value::Integer(float as i64)
    } else {
        Value::Number(float)
    }
}

/// Writes what `action` stands for into `register`.
///
/// Inlined into its loops: a host call from a trace restores a handful of
/// registers on every call, and a call per register cost more than the
/// write (a save of every callee-saved register and two jump tables).
#[allow(clippy::inline_always)]
#[inline(always)]
fn write_action(
    register: &mut Value,
    action: RegisterAction,
    area: &[i64],
    bank: &[Value],
    roots: &[Value],
    virtuals: &[TableHandle],
) {
    match action {
        RegisterAction::Keep => {}
        RegisterAction::Nil => *register = Value::Nil,
        RegisterAction::Const(scalar) => *register = scalar_to_value(scalar),
        RegisterAction::Float(slot) => {
            *register = Value::Number(f64::from_bits(area[slot as usize].cast_unsigned()));
        }
        RegisterAction::Integer(slot) => *register = Value::Integer(area[slot as usize]),
        RegisterAction::Number(slot, tag) => {
            *register = number_value(area[slot as usize], area[tag as usize] != 0);
        }
        RegisterAction::IntegerOfFloat(slot) => *register = number_value(area[slot as usize], true),
        RegisterAction::NumberConst(bits, tag) => {
            *register = number_value(bits.cast_signed(), area[tag as usize] != 0);
        }
        RegisterAction::Int64(slot) => *register = Value::Int64(area[slot as usize]),
        RegisterAction::UInt64(slot) => {
            *register = Value::UInt64(area[slot as usize].cast_unsigned());
        }
        RegisterAction::Boolean(slot) => {
            *register = Value::Boolean(area[slot as usize] != 0);
        }
        RegisterAction::Table(slot)
        | RegisterAction::Closure(slot)
        | RegisterAction::Thread(slot)
        | RegisterAction::Userdata(slot) => {
            *register = bank[area[slot as usize].cast_unsigned() as usize].clone();
        }
        RegisterAction::String(slot) => {
            *register = string_lane(area[slot as usize].cast_unsigned(), bank, roots).clone();
        }
        RegisterAction::BorrowedString(slot) => {
            // SAFETY: the exit area holds the address of a slot this run's
            // trace borrowed, and nothing has run since the exit.
            #[allow(unsafe_code)]
            {
                *register = unsafe { clone_borrowed_string(area[slot as usize].cast_unsigned()) };
            }
        }
        RegisterAction::Root(slot) => *register = roots[slot as usize].clone(),
        RegisterAction::Virtual(index) => *register = Value::Table(virtuals[index as usize]),
        RegisterAction::Slice {
            string,
            first,
            last,
        } => {
            *register = slice_value(
                string_lane(area[string as usize].cast_unsigned(), bank, roots),
                first.read(area),
                last.read(area),
            );
        }
    }
}

/// The value a restore action writes to an upvalue cell, or `None` for
/// `Keep`.
fn action_value(
    action: RegisterAction,
    area: &[i64],
    bank: &[Value],
    roots: &[Value],
    virtuals: &[TableHandle],
) -> Option<Value> {
    Some(match action {
        RegisterAction::Keep => return None,
        RegisterAction::Nil => Value::Nil,
        RegisterAction::Const(scalar) => scalar_to_value(scalar),
        RegisterAction::Float(slot) => {
            Value::Number(f64::from_bits(area[slot as usize].cast_unsigned()))
        }
        RegisterAction::Integer(slot) => Value::Integer(area[slot as usize]),
        RegisterAction::Number(slot, tag) => {
            number_value(area[slot as usize], area[tag as usize] != 0)
        }
        RegisterAction::IntegerOfFloat(slot) => number_value(area[slot as usize], true),
        RegisterAction::NumberConst(bits, tag) => {
            number_value(bits.cast_signed(), area[tag as usize] != 0)
        }
        RegisterAction::Int64(slot) => Value::Int64(area[slot as usize]),
        RegisterAction::UInt64(slot) => Value::UInt64(area[slot as usize].cast_unsigned()),
        RegisterAction::Boolean(slot) => Value::Boolean(area[slot as usize] != 0),
        RegisterAction::Table(slot)
        | RegisterAction::Closure(slot)
        | RegisterAction::Thread(slot)
        | RegisterAction::Userdata(slot) => {
            bank[area[slot as usize].cast_unsigned() as usize].clone()
        }
        RegisterAction::String(slot) => {
            string_lane(area[slot as usize].cast_unsigned(), bank, roots).clone()
        }
        RegisterAction::BorrowedString(slot) => {
            // SAFETY: as in `write_action`.
            #[allow(unsafe_code)]
            {
                unsafe { clone_borrowed_string(area[slot as usize].cast_unsigned()) }
            }
        }
        RegisterAction::Root(slot) => roots[slot as usize].clone(),
        RegisterAction::Virtual(index) => Value::Table(virtuals[index as usize]),
        RegisterAction::Slice {
            string,
            first,
            last,
        } => slice_value(
            string_lane(area[string as usize].cast_unsigned(), bank, roots),
            first.read(area),
            last.read(area),
        ),
    })
}

/// What entering a trace did.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum TraceRun {
    /// Nothing ran and the frame is untouched.
    NotEntered,
    /// Native code ran, but its exit could not build a virtual table. The
    /// frame is not restored; [`TracePolicy::take_restore_error`] has the
    /// runtime's error, which ends the run as the interpreter's own failed
    /// allocation would.
    Failed,
    /// Native code ran and the frame holds the restored interpreter state.
    Ran {
        /// The exit was a poll: run the VM safepoint and expect re-entry.
        poll: bool,
        /// Where the interpreter resumes, and its `last_interpreted_pc`.
        resume_pc: usize,
        last_interpreted_pc: Option<usize>,
    },
    /// A function trace's entered activation returned `value`. The frame
    /// is as the entry found it: the caller returns from it.
    Returned { value: Value },
}

/// The snapshot of exit `descriptor` of member `ir` as the generic restore
/// reads it: a root-frame register whose entry is the member's own entry
/// input for it, which the exit left out (`kept_by_frame` in
/// `jit/trace/compile.rs`), is `Unchanged`, since the frame holds it.
#[cfg(test)]
fn kept_by_frame(ir: &TraceIr, descriptor: &ExitDescriptor) -> Snapshot {
    let mut snapshot = ir.snapshots[descriptor.snapshot.index()].clone();
    if let Some(frame) = snapshot.frames.first_mut() {
        frame.registers.update(|register, entry| {
            if let RegisterEntry::Value(value) = *entry
                && ir
                    .entry
                    .inputs
                    .iter()
                    .any(|input| usize::from(input.register) == register && input.value == value)
                && descriptor
                    .slots
                    .binary_search_by_key(&value, |slot| slot.value)
                    .is_err()
            {
                RegisterEntry::Unchanged
            } else {
                *entry
            }
        });
    }
    snapshot
}

/// Reads exit values for the generic restore, which tests compare the plan
/// against.
#[cfg(test)]
pub(crate) struct ExitView<'a> {
    pub(crate) descriptor: &'a ExitDescriptor,
    pub(crate) area: &'a [i64],
}

#[cfg(test)]
impl ExitValues for ExitView<'_> {
    fn scalar(&self, value: ValueId) -> Option<TraceScalar> {
        self.descriptor.scalar(self.area, value)
    }
}

/// Whether two table, closure, string or native function values are the same
/// object.
// Thread and coroutine wrapper are distinct value kinds with their own
// payload types; their identity arms only happen to coincide.
#[allow(clippy::match_same_arms)]
fn same_object(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::String(a), Value::String(b)) => a.ptr_eq(b),
        (Value::Table(a), Value::Table(b)) => a == b,
        (Value::Closure(a), Value::Closure(b)) => a == b,
        (Value::NativeFunction(a), Value::NativeFunction(b)) => a == b,
        (Value::BuiltinFunction(a), Value::BuiltinFunction(b)) => a == b,
        (Value::GmatchIterator(a), Value::GmatchIterator(b)) => a == b,
        (Value::Thread(a), Value::Thread(b)) => a == b,
        (Value::CoroutineWrapper(a), Value::CoroutineWrapper(b)) => a == b,
        _ => false,
    }
}

/// Whether two values are the same one: the same number (by bits, so `NaN`
/// and `-0.0` are told apart), boolean, nil, or object. A conservative
/// answer: `false` for the kinds it does not tell.
#[inline]
fn same_value(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Nil, Value::Nil) => true,
        (Value::Boolean(a), Value::Boolean(b)) => a == b,
        (Value::Integer(a), Value::Integer(b)) => a == b,
        (Value::Number(a), Value::Number(b)) => a.to_bits() == b.to_bits(),
        (Value::Table(_), Value::Table(_))
        | (Value::Closure(_), Value::Closure(_))
        | (Value::NativeFunction(_), Value::NativeFunction(_))
        | (Value::Thread(_), Value::Thread(_)) => same_object(left, right),
        (Value::String(a), Value::String(b)) => a.ptr_eq(b),
        _ => false,
    }
}

/// The lane id of an opaque `value` for this run: its index in the root bank,
/// which keeps it alive. One object has one lane, so native code compares
/// identity by comparing ids.
fn lane_of(bank: &mut Vec<Value>, value: &Value) -> u64 {
    // The kind is matched once, not at every lane the search passes: the
    // bank holds every table and string the run has met.
    let found = match value {
        Value::String(text) => bank
            .iter()
            .position(|held| matches!(held, Value::String(known) if known.ptr_eq(text))),
        Value::Closure(closure) => bank
            .iter()
            .position(|held| matches!(held, Value::Closure(known) if known == closure)),
        // The library functions a trace reads from globals and upvalues.
        Value::NativeFunction(native) => bank
            .iter()
            .position(|held| matches!(held, Value::NativeFunction(known) if known == native)),
        Value::BuiltinFunction(builtin) => bank
            .iter()
            .position(|held| matches!(held, Value::BuiltinFunction(known) if known == builtin)),
        _ => bank.iter().position(|held| same_object(held, value)),
    };
    let index = found.unwrap_or_else(|| {
        bank.push(value.clone());
        bank.len() - 1
    });
    index as u64
}

/// How a value enters a run: a scalar carries its words, an object a lane of
/// the run's bank, which `Lane` builds the slot of.
enum Entry {
    Words(TraceInputSlot),
    Lane(fn(u64) -> TraceInputSlot),
}

fn entry_of(value: &Value) -> Entry {
    match value {
        Value::Nil => Entry::Words(TraceInputSlot::nil()),
        Value::Boolean(flag) => Entry::Words(TraceInputSlot::boolean(*flag)),
        Value::Integer(integer) => Entry::Words(TraceInputSlot::integer(*integer)),
        Value::Number(number) => Entry::Words(TraceInputSlot::float(*number)),
        Value::Int64(value) => Entry::Words(TraceInputSlot::int64(*value)),
        Value::UInt64(value) => Entry::Words(TraceInputSlot::uint64(*value)),
        Value::Table(_) => Entry::Lane(TraceInputSlot::table),
        Value::String(_) => Entry::Lane(TraceInputSlot::string),
        Value::Thread(_) => Entry::Lane(TraceInputSlot::thread),
        // A native function is a callable like a closure: the trace only
        // compares it by identity with the one it pinned.
        Value::Closure(_)
        | Value::CoroutineWrapper(_)
        | Value::NativeFunction(_)
        | Value::GmatchIterator(_)
        | Value::BuiltinFunction(
            BuiltinFunction::IPairsIterator
            | BuiltinFunction::Next
            | BuiltinFunction::Pairs
            | BuiltinFunction::IPairs
            | BuiltinFunction::ToString
            | BuiltinFunction::StringFormat
            | BuiltinFunction::StringGsub
            | BuiltinFunction::ProtectedCall
            | BuiltinFunction::ExtendedProtectedCall
            | BuiltinFunction::DebugTraceback
            | BuiltinFunction::CoroutineResume
            | BuiltinFunction::CoroutineYield,
        ) => Entry::Lane(TraceInputSlot::closure),
        _ => Entry::Words(TraceInputSlot::other()),
    }
}

#[inline(always)]
pub(crate) fn input_slot(value: &Value, bank: &mut Vec<Value>) -> TraceInputSlot {
    // The scalars, which most entry registers hold, are answered here
    // without the call and the search of the bank that objects take.
    match value {
        Value::Number(number) => TraceInputSlot::float(*number),
        Value::Integer(integer) => TraceInputSlot::integer(*integer),
        Value::Nil => TraceInputSlot::nil(),
        Value::Boolean(flag) => TraceInputSlot::boolean(*flag),
        Value::Table(table) => TraceInputSlot::table(lane_of_table(bank, *table)),
        _ => input_slot_of_object(value, bank),
    }
}

/// [`lane_of`] for a table: its place in the bank, which holds it from here
/// on.
#[inline(always)]
fn lane_of_table(bank: &mut Vec<Value>, table: TableHandle) -> u64 {
    let lane = bank
        .iter()
        .position(|held| matches!(held, Value::Table(known) if *known == table))
        .unwrap_or_else(|| {
            push_in_place(bank, Value::Table(table));
            bank.len() - 1
        });
    lane as u64
}

/// `Vec::push` that stays in the caller: the bank grows by a few lanes at
/// every entry, and the call to the compiler's generic push (whether it
/// inlines it moves with the code around) costs more than the push.
#[allow(unsafe_code)]
#[inline(always)]
fn push_in_place(vector: &mut Vec<Value>, value: Value) {
    let length = vector.len();
    if length == vector.capacity() {
        vector.reserve(16);
    }
    // SAFETY: the capacity covers one more element, which is written before
    // the length names it.
    unsafe {
        vector.as_mut_ptr().add(length).write(value);
        vector.set_len(length + 1);
    }
}

/// Writes the input slots that links of sides compiled apart rewrote into the
/// registers they stand for, in the root frame at `root` or the frame below it
/// (see [`CompiledTrace::writebacks`]). `words` holds a tag and bits for each.
/// A slot that holds no value the run could have written (a register the
/// entry could not give a slot) is left alone.
fn write_back_links(
    frames: &mut CallStack,
    root: usize,
    bank: &[Value],
    writebacks: &[crate::jit::Writeback],
    words: &[i64],
    runtime: &mut Runtime,
) {
    for (index, writeback) in writebacks.iter().enumerate() {
        let (Some(&tag), Some(&bits)) = (words.get(2 * index), words.get(2 * index + 1)) else {
            break;
        };
        let value = match tag {
            crate::jit::TRACE_TAG_FLOAT => Value::Number(f64::from_bits(bits.cast_unsigned())),
            crate::jit::TRACE_TAG_INTEGER => Value::Integer(bits),
            crate::jit::TRACE_TAG_BOOLEAN => Value::Boolean(bits != 0),
            crate::jit::TRACE_TAG_NIL => Value::Nil,
            crate::jit::TRACE_TAG_TABLE | crate::jit::TRACE_TAG_CLOSURE => {
                match usize::try_from(bits).ok().and_then(|lane| bank.get(lane)) {
                    Some(value) => value.clone(),
                    None => continue,
                }
            }
            _ => continue,
        };
        let (index, register) = match writeback.source {
            crate::jit::ExtraSource::Register(register) => (Some(root), register),
            crate::jit::ExtraSource::Caller(register) => (root.checked_sub(1), register),
            // A cell a link carried as a loop parameter goes back to the
            // cell. A global is read at entry and written by no link.
            crate::jit::ExtraSource::External(slot) => {
                if let Some(cell) = upvalue_cell(&frames.at_mut(root), runtime, bank, &slot) {
                    let written = runtime.cell_set(cell, value);
                    debug_assert!(written.is_ok(), "an entered trace's cells exist");
                }
                continue;
            }
        };
        let Some(index) = index else {
            continue;
        };
        if let Some(slot) = frames
            .at_mut(index)
            .registers
            .get_mut(usize::from(register))
        {
            *slot = value;
        }
    }
}

fn extra_slot(value: &Value, bank: &mut Vec<Value>) -> TraceInputSlot {
    match entry_of(value) {
        Entry::Words(slot) => slot,
        // A callable the side compares with one it pinned has a lane. Other
        // objects have none: the side that reads one declines.
        Entry::Lane(slot) if matches!(value, Value::Closure(_)) => slot(lane_of(bank, value)),
        Entry::Lane(_) => TraceInputSlot::other(),
    }
}

fn input_slot_of_object(value: &Value, bank: &mut Vec<Value>) -> TraceInputSlot {
    match entry_of(value) {
        Entry::Words(slot) => slot,
        Entry::Lane(slot) => slot(lane_of(bank, value)),
    }
}

/// Whether [`input_slot`] would give `value` the slot `held`, without
/// searching the bank or adding to it: an object holds when its lane is the
/// one `held` names.
fn input_slot_holds(value: &Value, held: &TraceInputSlot, bank: &[Value]) -> bool {
    match entry_of(value) {
        Entry::Words(slot) => slot == *held,
        Entry::Lane(slot) => {
            let probe = slot(0);
            probe.tag() == held.tag()
                && usize::try_from(held.bits())
                    .ok()
                    .and_then(|lane| bank.get(lane))
                    .is_some_and(|pinned| same_object(pinned, value))
        }
    }
}

const fn exit_kind(class: TraceExitClass) -> TraceExitKind {
    match class {
        // An unwind is an activation below the one that left: it counts with
        // that exit.
        TraceExitClass::GuardFail | TraceExitClass::Unwind => TraceExitKind::GuardFail,
        TraceExitClass::CommittedEffect => TraceExitKind::CommittedEffect,
        TraceExitClass::LoopComplete => TraceExitKind::LoopComplete,
        TraceExitClass::Poll => TraceExitKind::Poll,
    }
}

const fn guard_kind(class: TraceGuardClass) -> TraceGuardKind {
    match class {
        TraceGuardClass::Array => TraceGuardKind::ArrayBounds,
        TraceGuardClass::Type => TraceGuardKind::Type,
        TraceGuardClass::Branch => TraceGuardKind::Branch,
        TraceGuardClass::Callee | TraceGuardClass::Class => TraceGuardKind::Callee,
        TraceGuardClass::Depth | TraceGuardClass::Host => TraceGuardKind::Other,
    }
}

impl InstalledTrace {
    /// Records an entry that matched none of the header's versions. It stays
    /// out of the productivity window: a sibling signature that mismatches at
    /// every backedge would otherwise flood the window and retire a version
    /// that runs fine on its own entries. Only a long run of mismatches with
    /// no matching entry in between counts as unproductive. Returns whether
    /// that happened.
    ///
    /// A `refused` mismatch, which the frame's register tags decided before
    /// any input was marshalled, weighs [`REFUSED_MISMATCH_WEIGHT`] times
    /// less than one that paid for a whole entry: the run of them a version
    /// sits through costs little, and retiring it throws away a tree that
    /// pays on the entries it takes.
    fn note_mismatch(&mut self, refused: bool) -> bool {
        self.idle_mismatches += if refused { 1 } else { REFUSED_MISMATCH_WEIGHT };
        if self.idle_mismatches < TRACE_IDLE_MISMATCH_LIMIT * REFUSED_MISMATCH_WEIGHT {
            return false;
        }
        self.idle_mismatches = 0;
        true
    }

    /// Whether the loop sits inside other loops and has run fewer iterations
    /// per entry than it needs to pay for one. Such a root leaves after a
    /// trip or two on every outer iteration; the tree that is about to
    /// carry it through those loops (see "Nested loops" in the module
    /// documentation) is the trace that pays.
    fn runs_short(&self) -> bool {
        self.encloses && self.iterations < self.entries * self.min_iterations
    }

    /// Records an entry credited with `iterations` loop-backs, and with
    /// `loop_backs` against a heavy body's bar. Returns whether the window
    /// just closed below the productivity bar.
    fn note_entry(&mut self, iterations: u64, loop_backs: u64) -> bool {
        self.idle_mismatches = 0;
        self.window_loop_backs += loop_backs;
        self.note_window(iterations)
    }

    /// Adds one entry to the productivity window.
    fn note_window(&mut self, iterations: u64) -> bool {
        self.window_entries += 1;
        self.window_iterations += iterations;
        if self.window_entries < TRACE_PRODUCTIVITY_WINDOW {
            return false;
        }
        let loop_backs = std::mem::take(&mut self.window_loop_backs);
        // A function trace is judged by its average over its life, from its
        // `TRACE_FUNCTION_WARMUP` entry on. A recursion that went deep in the
        // interpreter while the trace compiled keeps making self calls from
        // those frames, mostly near the leaves, for a while after the trace
        // took over the rest in long runs. The trace has paid once its
        // average stood above the bar at a window's close: when it retires
        // later, a recording of the same trace may compile again.
        if self.function.is_some() {
            self.window_entries = 0;
            self.window_iterations = 0;
            let below = 2 * self.iterations < self.entries * TRACE_MIN_HALF_ACTIVATIONS_PER_ENTRY;
            self.productive_windows = self.productive_windows.saturating_add(u32::from(!below));
            return self.entries >= TRACE_FUNCTION_WARMUP && below;
        }
        let bar = u64::from(self.window_entries) * self.min_iterations;
        // A run is credited one loop-back even when it left before the
        // first: against a bar of one per entry, that passes every window.
        // A heavy body is held to `window_loop_backs`, which does not credit
        // a run that left inside an inlined call before its first loop-back.
        // Shape-469's driver loop (`for it = 1, 40000`, `shape:equal_p` and
        // the sturdy-face checks inlined, the region nest's short loops
        // unrolled) left four frames deep in each of 38000 entries, and kept
        // its tree.
        if self.min_iterations == TRACE_MIN_ITERATIONS_HEAVY {
            self.window_iterations = loop_backs;
        }
        let productive = self.window_iterations >= bar;
        self.productive_windows = self
            .productive_windows
            .saturating_add(u32::from(productive));
        let retire = if productive {
            let cap = TRACE_PRODUCTIVITY_CREDIT_WINDOWS
                * u64::from(TRACE_PRODUCTIVITY_WINDOW)
                * self.min_iterations;
            self.productivity_credit =
                (self.productivity_credit + (self.window_iterations - bar)).min(cap);
            false
        } else {
            let deficit = bar - self.window_iterations;
            let covered = self.productivity_credit >= deficit;
            self.productivity_credit = self.productivity_credit.saturating_sub(deficit);
            !covered
        };
        self.window_entries = 0;
        self.window_iterations = 0;
        retire
    }
}

/// What one attempt to run a version did.
enum Attempt {
    /// The version's entry types did not match. Nothing ran. `next` is the
    /// next older version at the header.
    Mismatch { next: Option<u32> },
    /// A [`Self::Mismatch`] that the tags of the frame's registers decided
    /// before anything was marshalled (see [`frame_mismatch`]): it cost a
    /// compare per register, not an entry.
    Refused { next: Option<u32> },
    /// The attempt ended, with or without running. `retire` says the
    /// version's productivity window closed below the bar.
    Done { run: TraceRun, retire: bool },
}

impl TracePolicy {
    /// Installs hand-built IR for the loop at `header_pc` of `chunk`, as a
    /// finished compile would, and returns the trace's index.
    #[cfg(test)]
    pub(crate) fn install_hand_built(
        &mut self,
        stats: &mut TraceStatistics,
        chunk: &Arc<Chunk>,
        header_pc: usize,
        ir: TraceIr,
    ) -> Option<u32> {
        install_helpers();
        let compiled = crate::jit::compile_trace(&ir, self.tier).ok()?;
        self.sites.site_mut(chunk, header_pc)?.state = SiteState::Compiling;
        self.install_compiled(stats, chunk, header_pc, Arc::new(ir), compiled);
        self.installed_index_at_pc(header_pc)
    }

    /// Queues, for every guard that ends the loop of the trace at `index`, a
    /// side the tree cannot take (the root's own IR, which fails to verify),
    /// and builds the queue as a finished side recording would. The guards
    /// spend their attempts on the refusal and never record again.
    #[cfg(test)]
    pub(crate) fn queue_sides_the_tree_refuses(&mut self, stats: &mut TraceStatistics, index: u32) {
        let Some(trace) = self.cache.get_mut(index) else {
            return;
        };
        let keys: Vec<SideKey> = trace
            .side_keys
            .iter()
            .zip(&trace.loop_ends)
            .filter_map(|(key, ends)| key.filter(|_| *ends))
            .collect();
        for key in keys {
            let side = TraceSide {
                parent: key.parent.map(|parent| parent as usize),
                guard: key.origin,
                ir: Box::new((*trace.ir).clone()),
            };
            trace.side_recorded(key, side, Vec::new(), Vec::new(), None);
        }
        self.submit_queued(stats, index);
    }

    /// Whether the installed trace `index` is a stepping stone that wants
    /// its recompile at the policy's tier now.
    #[cfg(test)]
    pub(crate) fn wants_stone_recompile(&self, index: u32) -> bool {
        self.cache
            .get(index)
            .is_some_and(InstalledTrace::stone_given_up)
    }

    /// Turns native compilation of closed traces on or off. With it off a
    /// closed trace waits in [`Self::queued`] and its site is `Queued`.
    #[cfg(test)]
    pub(crate) fn set_native_compilation(&mut self, enabled: bool) {
        self.compile = enabled;
    }

    /// Number of compiled traces the cache holds.
    #[cfg(test)]
    pub(crate) fn installed_count(&self) -> usize {
        self.cache.live()
    }

    /// Removes every installed trace and blacklists its header, as a trace
    /// that was retired often enough is: the site is rejected and the
    /// header never records again. Returns how many traces went.
    #[cfg(test)]
    pub(crate) fn blacklist_installed(&mut self, stats: &mut TraceStatistics) -> usize {
        let indexes = self.cache.installed_indexes();
        for &index in &indexes {
            let site = self.cache.get(index).and_then(InstalledTrace::profile_site);
            self.retire(stats, index);
            if let Some((chunk, header)) = site {
                self.sites.reject(&chunk, header, RejectCause::Blacklisted);
            }
        }
        indexes.len()
    }

    /// The optimization tier of the trace installed at `index`.
    #[cfg(test)]
    pub(crate) fn installed_tier(&self, index: u32) -> Option<OptimizationTier> {
        self.cache.get(index).map(|trace| trace.compiled.tier())
    }

    /// Holds the trace at `index` to the bar of a heavy body
    /// ([`min_iterations_of`]) and counts one productivity window of runs,
    /// each credited one loop-back and `loop_backs` against that bar. Returns
    /// whether the window closed below it.
    #[cfg(test)]
    pub(crate) fn heavy_window_retires(&mut self, index: u32, loop_backs: u64) -> bool {
        let Some(trace) = self.cache.get_mut(index) else {
            return false;
        };
        trace.min_iterations = TRACE_MIN_ITERATIONS_HEAVY;
        trace.productivity_credit = 0;
        trace.window_entries = 0;
        trace.window_iterations = 0;
        trace.window_loop_backs = 0;
        (0..TRACE_PRODUCTIVITY_WINDOW).fold(false, |_, _| trace.note_entry(1, loop_backs))
    }

    /// Clears the productivity window of the trace at `index`, so a test can
    /// drive one trace through many exits without retiring it.
    #[cfg(test)]
    pub(crate) fn forgive(&mut self, index: u32) {
        if let Some(trace) = self.cache.get_mut(index) {
            trace.window_entries = 0;
            trace.window_iterations = 0;
            trace.window_loop_backs = 0;
        }
    }

    /// The generic restore of exit `exit` of the trace at `index`, for tests
    /// that compare it with the plan `enter` applies.
    #[cfg(test)]
    pub(crate) fn generic_restore(
        &self,
        index: u32,
        exit: usize,
        area: &[i64],
        frame: &mut FrameMut<'_>,
        runtime: &mut Runtime,
    ) -> Option<Vec<(CellTarget, Value)>> {
        let trace = self.cache.get(index)?;
        let descriptor = &trace.compiled.exits()[exit];
        let view = ExitView { descriptor, area };
        let lowered = trace.compiled.lowered_member(descriptor.member);
        let snapshot = &kept_by_frame(lowered, descriptor);
        let virtuals =
            super::deopt::materialize_virtuals(runtime, snapshot, &view, &self.bank, &trace.roots)
                .ok()?;
        super::deopt::restore_window(
            frame,
            snapshot,
            &view,
            &self.bank,
            &trace.roots,
            &trace.externals,
            &virtuals,
        )
        .ok()
        .map(|restore| restore.upvalues)
    }

    /// The cell writes the plan of exit `exit` of the trace at `index` makes
    /// for `area`, for tests that compare them with [`Self::generic_restore`].
    #[cfg(test)]
    pub(crate) fn plan_upvalues(
        &self,
        index: u32,
        exit: usize,
        area: &[i64],
    ) -> Vec<(CellTarget, Value)> {
        let trace = self.cache.get(index).expect("installed");
        trace.restores[exit]
            .upvalues
            .iter()
            .filter_map(|(cell, action)| {
                action_value(*action, area, &self.bank, &trace.roots, &[])
                    .map(|value| (*cell, value))
            })
            .collect()
    }

    /// Applies the restore plan of exit `exit` of the trace at `index`, for
    /// tests that compare it with [`Self::generic_restore`].
    #[cfg(test)]
    pub(crate) fn plan_restore(
        &self,
        index: u32,
        exit: usize,
        area: &[i64],
        frame: &mut FrameMut<'_>,
        runtime: &mut Runtime,
    ) {
        let trace = self.cache.get(index).expect("installed");
        let restore = &trace.restores[exit];
        let tables = restore
            .materialize(area, &self.bank, &trace.roots, runtime)
            .expect("the test runtime allocates");
        restore.apply(frame, area, &self.bank, &trace.roots, &tables, runtime);
    }

    /// The generic restore of exit `exit` of the trace at `index` onto the
    /// whole stack, inlined frames included, for tests that compare it with
    /// [`Self::plan_restore_stack`]. The bank must hold the pinned closures.
    #[cfg(test)]
    pub(crate) fn generic_restore_stack(
        &self,
        index: u32,
        exit: usize,
        area: &[i64],
        frames: &mut CallStack,
        runtime: &mut Runtime,
    ) -> Option<Vec<(CellTarget, Value)>> {
        let trace = self.cache.get(index)?;
        let descriptor = &trace.compiled.exits()[exit];
        let view = ExitView { descriptor, area };
        let lowered = trace.compiled.lowered_member(descriptor.member);
        let snapshot = &kept_by_frame(lowered, descriptor);
        let virtuals =
            super::deopt::materialize_virtuals(runtime, snapshot, &view, &self.bank, &trace.roots)
                .ok()?;
        super::deopt::restore_snapshot(
            frames,
            frames.len() - 1,
            runtime,
            lowered,
            snapshot,
            &view,
            &self.bank,
            &trace.roots,
            &trace.externals,
            &virtuals,
        )
        .map_err(|error| eprintln!("generic restore: {error:?}"))
        .ok()
        .map(|restore| restore.upvalues)
    }

    /// Applies the restore plan of exit `exit` of the trace at `index` to the
    /// whole stack, inlined frames included. The bank must hold the pinned
    /// closures.
    #[cfg(test)]
    pub(crate) fn plan_restore_stack(
        &self,
        index: u32,
        exit: usize,
        area: &[i64],
        frames: &mut CallStack,
        runtime: &mut Runtime,
    ) {
        let trace = self.cache.get(index).expect("installed");
        let root = frames.len() - 1;
        let restore = &trace.restores[exit];
        let tables = restore
            .materialize(area, &self.bank, &trace.roots, runtime)
            .expect("the test runtime allocates");
        if restore.writes_caller() {
            restore.apply_callers(frames, root, area, &self.bank, &trace.roots, &tables);
        }
        restore.apply(
            &mut frames.at_mut(root),
            area,
            &self.bank,
            &trace.roots,
            &tables,
            runtime,
        );
        restore.push_inner(frames, area, &self.bank, &trace.roots, &tables, runtime);
    }

    /// The descriptor of exit `exit` of the trace at `index`, for tests.
    #[cfg(test)]
    pub(crate) fn exit_descriptor(&self, index: u32, exit: usize) -> Option<&ExitDescriptor> {
        self.cache
            .get(index)
            .and_then(|trace| trace.compiled.exits().get(exit))
    }

    /// Number of exits of the trace at `index`, for tests.
    #[cfg(test)]
    pub(crate) fn exit_count(&self, index: u32) -> usize {
        self.cache
            .get(index)
            .map_or(0, |trace| trace.compiled.exits().len())
    }

    /// Recompiles the trace at `index` so its guard branch `guard` (in the
    /// lowered trace's numbering) always exits, keeping its tier. False when
    /// the guard is not a branch or the compile fails.
    #[cfg(test)]
    pub(crate) fn force_guard(&mut self, index: u32, guard: usize) -> bool {
        let Some(trace) = self.cache.get_mut(index) else {
            return false;
        };
        let tier = trace.compiled.tier();
        match crate::jit::compile_trace_forcing_guard(&trace.ir, tier, guard) {
            Ok(compiled) => {
                trace.compiled = compiled;
                true
            }
            Err(_) => false,
        }
    }

    /// The names of the helpers the installed traces call, one per call site
    /// (see [`oximond::jit_ir::trace::TraceOp::CallHelper`]).
    pub(crate) fn installed_helper_names(&self) -> Vec<&'static str> {
        self.cache
            .profile_entries()
            .flat_map(|trace| trace.ir.helper_calls())
            .filter_map(|helper| helper.info().map(|info| info.name))
            .collect()
    }

    /// The single-run patterns the installed traces scan in their own code
    /// (see [`crate::jit::CompiledTrace::inline_runs`]).
    pub(crate) fn installed_inline_runs(&self) -> usize {
        self.cache
            .profile_entries()
            .map(|trace| trace.compiled.inline_runs())
            .sum()
    }

    /// The events of the metatable classes the installed traces check, one
    /// per class guard (see
    /// [`oximond::jit_ir::trace::TraceIr::class_guard_events`]).
    pub(crate) fn installed_class_guard_events(&self) -> Vec<String> {
        self.cache
            .profile_entries()
            .flat_map(|trace| trace.ir.class_guard_events())
            .map(|event| String::from_utf8_lossy(&event).into_owned())
            .collect()
    }

    /// How many operations of the installed traces call out of the generated
    /// code for a library function (see
    /// [`oximond::jit_ir::trace::TraceIr::external_calls`]).
    pub(crate) fn installed_external_calls(&self) -> usize {
        self.cache
            .profile_entries()
            .map(|trace| trace.ir.external_calls())
            .sum()
    }

    /// [`Self::installed_external_calls`] for the code that runs: the
    /// optimized traces, which can have dropped a call the recording made.
    pub(crate) fn installed_compiled_external_calls(&self) -> usize {
        self.cache
            .profile_entries()
            .map(|trace| {
                (0..trace.compiled.member_count())
                    .map(|member| trace.compiled.lowered_member(member).external_calls())
                    .sum::<usize>()
            })
            .sum()
    }

    /// The IR of the trace installed at `index`.
    #[cfg(test)]
    pub(crate) fn installed_ir(&self, index: u32) -> Option<&TraceIr> {
        self.cache.get(index).map(|trace| &*trace.ir)
    }

    /// The IR of every installed trace, wherever its header is.
    #[cfg(test)]
    pub(crate) fn installed_irs(&self) -> impl Iterator<Item = &TraceIr> {
        self.cache.profile_entries().map(|trace| &*trace.ir)
    }

    /// Side traces the tree of the trace installed at `index` holds, and how
    /// many of them the compiled code contains.
    #[cfg(test)]
    pub(crate) fn installed_sides(&self, index: u32) -> Option<(usize, usize)> {
        let trace = self.cache.get(index)?;
        let linked = trace
            .compiled
            .side_members()
            .iter()
            .filter(|member| member.is_some())
            .count();
        Some((trace.ir.sides.len(), linked))
    }

    /// The sides of the tree at `index` that are compiled apart from it, as
    /// indexes into its sides, for tests.
    #[cfg(test)]
    pub(crate) fn installed_apart(&self, index: u32) -> Vec<usize> {
        self.cache
            .get(index)
            .map_or_else(Vec::new, |trace| trace.apart.clone())
    }

    /// The compiled code of the trace installed at `index`, for tests.
    #[cfg(test)]
    pub(crate) fn installed_compiled(&self, index: u32) -> Option<&CompiledTrace> {
        self.cache.get(index).map(|trace| &trace.compiled)
    }

    /// Queues a recompile of the trace at `index` for the tree version before
    /// the current one, as an older request that lost a race would be.
    /// Returns whether the trace exists.
    #[cfg(test)]
    pub(crate) fn queue_stale_retree(&mut self, index: u32) -> bool {
        let Some(trace) = self.cache.get_mut(index) else {
            return false;
        };
        trace.version += 1;
        let request = CompileRequest {
            ir: Arc::clone(&trace.ir),
            tier: trace.compiled.tier(),
            epoch: self.epoch,
            target: CompileTarget::Retree {
                index,
                serial: trace.serial,
                version: trace.version - 1,
            },
            reserved: 0,
            side: None,
            planned: None,
            victims: Vec::new(),
        };
        self.outbox.push(request);
        true
    }

    /// Whether a side recording or a tree compile is outstanding for the
    /// trace at `index`, for tests.
    #[cfg(test)]
    pub(crate) fn side_pending(&self, index: u32) -> bool {
        self.cache.get(index).is_some_and(|trace| trace.side.busy())
    }

    /// The optimized IR the code of the trace installed at `index` was
    /// lowered from, for tests.
    #[cfg(test)]
    pub(crate) fn installed_lowered_ir(&self, index: u32) -> Option<&TraceIr> {
        self.cache
            .get(index)
            .map(|trace| trace.compiled.lowered_ir())
    }

    /// The index of the installed trace whose tree returns into its caller
    /// (see `Recorder::return_through`), for tests.
    #[cfg(test)]
    pub(crate) fn linked_trace_index(&self) -> Option<u32> {
        self.cache.installed_indexes().into_iter().find(|&index| {
            self.cache
                .get(index)
                .is_some_and(|trace| trace.caller.is_some())
        })
    }

    /// The exits of the trace at `index` whose snapshot has the root frame's
    /// caller above the root frame (the exit pops the root frame), for tests.
    #[cfg(test)]
    pub(crate) fn caller_exits(&self, index: u32) -> Vec<usize> {
        let Some(trace) = self.cache.get(index) else {
            return Vec::new();
        };
        trace
            .compiled
            .exits()
            .iter()
            .enumerate()
            .filter(|(_, descriptor)| {
                let lowered = trace.compiled.lowered_member(descriptor.member);
                lowered.snapshots[descriptor.snapshot.index()]
                    .frames
                    .iter()
                    .skip(1)
                    .any(|frame| {
                        lowered.frames[frame.frame.index()]
                            .inlined
                            .is_some_and(|call| matches!(call.entry, FrameEntry::Caller { .. }))
                    })
            })
            .map(|(exit, _)| exit)
            .collect()
    }

    /// The pinned closures of the trace at `index`, as the bank holds them
    /// during a run, for tests.
    #[cfg(test)]
    pub(crate) fn pinned_values(&self, index: u32) -> Vec<Value> {
        self.cache.get(index).map_or_else(Vec::new, |trace| {
            trace
                .pinned
                .iter()
                .map(|identity| {
                    Value::Closure(crate::runtime::ClosureHandle::from_native_identity(
                        *identity,
                    ))
                })
                .collect()
        })
    }

    /// The trace index of the only installed site at `pc`, for tests.
    #[cfg(test)]
    pub(crate) fn installed_index_at_pc(&self, pc: usize) -> Option<u32> {
        match self.site_state_any(pc)? {
            SiteState::Installed { index } => Some(index),
            _ => None,
        }
    }

    /// The versions installed at the header at `pc`, newest first, for tests.
    #[cfg(test)]
    pub(crate) fn versions_at_pc(&self, pc: usize) -> Vec<u32> {
        let Some(SiteState::Installed { index }) = self.site_state_any(pc) else {
            return Vec::new();
        };
        self.chain(Some(index))
    }

    /// Whether the only header at `pc` takes no more versions, for tests.
    #[cfg(test)]
    pub(crate) fn versions_closed_at_pc(&self, pc: usize) -> Option<bool> {
        self.sites.versions_closed_at_pc(pc)
    }

    /// Retires version `index` as the productivity rules would, for tests.
    #[cfg(test)]
    pub(crate) fn retire_for_test(&mut self, stats: &mut TraceStatistics, index: u32) {
        self.retire(stats, index);
    }

    /// The version chain starting at `head`, newest first. Bounded by the
    /// version cap, so a damaged chain cannot loop.
    pub(super) fn chain(&self, head: Option<u32>) -> Vec<u32> {
        let mut chain = Vec::new();
        let mut at = head;
        while let Some(index) = at {
            if chain.len() > self.cache.limits().versions {
                break;
            }
            let Some(trace) = self.cache.get(index) else {
                break;
            };
            chain.push(index);
            at = trace.next;
        }
        chain
    }

    /// Installs compiled code for a closed trace at its header. The
    /// site must still be waiting in `Compiling`. Makes room under the cache
    /// caps by evicting the least recently entered traces, or drops the
    /// trace and backs the site off when no room can be made.
    pub(super) fn install_compiled(
        &mut self,
        stats: &mut TraceStatistics,
        chunk: &Arc<Chunk>,
        header_pc: usize,
        ir: Arc<TraceIr>,
        compiled: CompiledTrace,
    ) {
        let waiting = self
            .sites
            .existing_mut(chunk, header_pc)
            .is_some_and(|site| site.state == SiteState::Compiling);
        if !waiting {
            stats.stale_completions += 1;
            return;
        }
        let _icount = crate::jit::icount::scope("install");
        let externals: Box<[ExternalSlot]> = ir.external.clone().into_boxed_slice();
        let register_count =
            usize::from(ir.frames[0].register_count).saturating_sub(externals.len());
        let Some(roots) = resolve_roots(&ir) else {
            // A root slot with no string constant behind it cannot restore.
            self.settle_site(
                stats,
                chunk,
                header_pc,
                TraceAbortReason::CompileRejected,
                None,
            );
            return;
        };
        let Some(bindings) = plan_bindings(&compiled) else {
            self.settle_site(
                stats,
                chunk,
                header_pc,
                TraceAbortReason::CompileRejected,
                None,
            );
            return;
        };
        let fields = plan_fields(&compiled);
        let restores =
            restores_of_entered(&compiled, register_count, &externals).and_then(|mut restores| {
                plan_unwinds(&compiled, &mut restores, Some(chunk))?;
                Some(restores)
            });
        let function = InstalledFunction::of(&ir, &compiled, chunk);
        if ir.function.is_some() != function.is_some() {
            self.settle_site(
                stats,
                chunk,
                header_pc,
                TraceAbortReason::CompileRejected,
                None,
            );
            return;
        }
        let Some(restores) = restores else {
            // Only plain root-frame snapshots restore through the fast path.
            self.settle_site(
                stats,
                chunk,
                header_pc,
                TraceAbortReason::CompileRejected,
                None,
            );
            return;
        };
        let head = self
            .sites
            .existing_mut(chunk, header_pc)
            .and_then(|site| site.head);
        let siblings = self.chain(head);
        if siblings.len() >= self.cache.limits().versions {
            stats.cache_rejections += 1;
            self.back_off_refused(chunk, header_pc, COMPILE_REFUSAL_COOLDOWN);
            return;
        }
        self.sweep_dead_traces();
        // A tree installed whole (one a trace profile recorded) can have a
        // side that pins closures behind the root's lanes. Lowering shares
        // them with the root, and an exit inside a call the side inlined
        // finds its closure there, as after a retree.
        let lowered = &compiled.lowered_ir().pinned_closures;
        let pinned: Box<[u64]> = if lowered.len() > ir.pinned_closures.len()
            && lowered.starts_with(&ir.pinned_closures)
        {
            lowered.clone().into_boxed_slice()
        } else {
            ir.pinned_closures.clone().into_boxed_slice()
        };
        let aliases: Box<[CellAlias]> = ir.cell_aliases.clone().into_boxed_slice();
        let max_inline_depth = tree_inline_depth(&ir);
        let caller = tree_caller(&ir, None, None);
        let encloses =
            ir.poll_snapshot
                .is_some_and(|poll| match ir.snapshots[poll.index()].phase {
                    crate::jit_ir::trace::SnapshotPhase::After { pc, .. } => {
                        let span = nest::LoopSpan {
                            header: header_pc,
                            latch: pc,
                        };
                        // A root that ends in front of the loops inside it
                        // carries them as loop sides: they are no siblings.
                        if ends_in_front_of_loop(&ir) {
                            nest::can_cover_from_outer_root(chunk, span)
                        } else {
                            nest::can_cover(chunk, span)
                        }
                    }
                    crate::jit_ir::trace::SnapshotPhase::Before { .. } => false,
                });
        // Where the backedge count of the loop around this one stands, so a
        // later outer exit compares the outer iterations since then with the
        // inner runs since then (see `consider_side`).
        let outer_backedges_at_install = if encloses {
            ir.poll_snapshot
                .and_then(|poll| match ir.snapshots[poll.index()].phase {
                    crate::jit_ir::trace::SnapshotPhase::After { pc, .. } => Some(pc),
                    crate::jit_ir::trace::SnapshotPhase::Before { .. } => None,
                })
                .and_then(|latch| {
                    nest::enclosing_loops(
                        chunk,
                        nest::LoopSpan {
                            header: header_pc,
                            latch,
                        },
                    )
                    .first()
                    .copied()
                })
                .and_then(|outer| self.sites.existing_mut(chunk, outer.header))
                .map_or(0, |site| match site.state {
                    SiteState::Cold { backedges } => backedges,
                    _ => 0,
                })
        } else {
            0
        };
        // A numeric for loop's trips were estimated when it closed, and a
        // short one was made a stepping stone then (`is_stepping_stone`).
        // The trips of any other loop are first seen in its window.
        let stone_candidate = encloses
            && function.is_none()
            && !self
                .sites
                .site(chunk, header_pc)
                .is_some_and(super::policy::Site::stone_gave_up)
            && ir
                .poll_snapshot
                .is_some_and(|poll| match ir.snapshots[poll.index()].phase {
                    crate::jit_ir::trace::SnapshotPhase::After { pc, .. } => {
                        Recorder::numeric_latch_target(chunk, pc).is_none()
                    }
                    crate::jit_ir::trace::SnapshotPhase::Before { .. } => false,
                });
        let lazy = self
            .sites
            .site(chunk, header_pc)
            .map(|site| site.lazy.clone())
            .unwrap_or_default();
        let (side_keys, virtual_refused) =
            side_exits_of(&compiled, &restores, &ir, encloses, false, Some(&**chunk));
        if debug_env::trace() {
            let inner = restores.iter().filter(|restore| !restore.inner.is_empty());
            eprintln!(
                "jit-trace: the root at pc {header_pc} has {} exits, {} of them inside inlined calls ({} a side can start in), {} with side keys",
                restores.len(),
                inner.clone().count(),
                inner.filter(|restore| restore.seedable).count(),
                side_keys.iter().filter(|key| key.is_some()).count(),
            );
        }
        let loop_ends = loop_ends_of(&compiled);
        let ir_has_integers = ir.values.iter().any(|value| value.ty == TraceType::Integer);
        let min_iterations = min_iterations_of(&ir);
        let overwrite_version = OverwriteVersion::of_helper_calls(&ir.helper_calls());
        let exit_hits = vec![0; compiled.exits().len()];
        let cold_hits = vec![0; compiled.exits().len()];
        let exit_last_hit = vec![0; compiled.exits().len()];
        let optimizer = compiled.optimize_stats();
        let clif_insts = compiled.clif_insts();
        let simple_binds = simple_bindings(&compiled);
        let entry_plan = entry_plan_of(&compiled, &ir, register_count);
        let written_cells = written_cell_slots(&ir);
        let trace = Box::new(InstalledTrace {
            chunk: Arc::downgrade(chunk),
            header_pc,
            ir,
            compiled,
            profile: self.profile,
            version: 0,
            exit_hits,
            cold_hits,
            interp_exits: Default::default(),
            hits_since: 0,
            exit_last_hit,
            side_keys,
            virtual_refused,
            loop_ends,
            side: SideBook {
                max_sides: self.max_sides,
                grace: SIDE_GRACE_WINDOWS,
                ..SideBook::default()
            },
            detached: Vec::new(),
            apart: Vec::new(),
            base_plan: RefCell::new(None),
            seed_plan: RefCell::new(None),
            grown_plan: None,
            child_calls: 0,
            native_calls: 0,
            link_exits: 0,
            units: 0,
            encloses,
            stone_candidate,
            stone: None,
            mismatches: 0,
            stone_start: (0, 0),
            keep_guards_wanted: false,
            guard_move_grace: 0,
            outer_backedges_at_install,
            covered: Vec::new(),
            register_count,
            has_upvalues: upvalue_slots(&externals) > 0,
            many_upvalues: upvalue_slots(&externals) > 1,
            pinned,
            aliases,
            cells_memo: None,
            written_cells,
            max_inline_depth,
            caller,
            caller_chunk: None,
            caller_captured: None,
            externals,
            roots,
            simple_binds,
            staged_visits: 0,
            entry_plan,
            bindings,
            field_hints: vec![0; crate::runtime::shape_cache_len(fields.len())],
            table_shapes: Vec::new(),
            meta_field: row_words_of(&fields),
            field_tags: fields.iter().map(|(_, kind)| kind.tag()).collect(),
            row_memo: new_row_memo(fields.len()),
            fields,
            restores,
            iterations: 0,
            entries: 0,
            integers: ir_has_integers,
            min_iterations,
            window_entries: 0,
            idle_mismatches: 0,
            callee_streak: 0,
            window_iterations: 0,
            window_loop_backs: 0,
            productivity_credit: 0,
            productive_windows: 0,
            mismatch_streak: 0,
            entry_flips: Vec::new(),
            frame_checks: None,
            lazy,
            overwrite_version,
            overwrite_polls: 0,
            overwrite_polling: 0,
            overwrite_rest: 0,
            last_used: 0,
            serial: 0,
            next: head,
            function,
        });
        let (code, meta) = (trace.code_charge(), trace.meta_charge());
        let Some(victims) = self.cache.plan_eviction(code, meta, true, &siblings) else {
            stats.cache_rejections += 1;
            self.back_off_refused(chunk, header_pc, COMPILE_REFUSAL_COOLDOWN);
            return;
        };
        for victim in victims {
            self.evict(stats, victim);
        }
        // Evicting may have removed a sibling's chain links, but never a
        // sibling itself (they are kept), so `head` still names the chain.
        let Some(index) = self.cache.insert(trace) else {
            stats.cache_rejections += 1;
            self.back_off_refused(chunk, header_pc, COMPILE_REFUSAL_COOLDOWN);
            return;
        };
        stats.traces_compiled += 1;
        self.name_in_perf_map(index);
        stats.clif_instructions_tree += clif_insts as u64;
        if let Some(trace) = self.cache.get(index) {
            stats.lane_array_loads_compiled += trace.compiled.lane_array_loads() as u64;
            stats.lane_array_stores_compiled += trace.compiled.lane_array_stores() as u64;
            stats.value_selects_compiled += trace.compiled.value_selects() as u64;
            stats.loop_links += trace.compiled.loop_links() as u64;
            stats.demoted_bindings += u64::from(trace.compiled.demoted_bindings());
        }
        if self
            .cache
            .get(index)
            .is_some_and(|trace| trace.function.is_some())
        {
            stats.compiled_function_traces += 1;
        }
        stats.optimizer.accumulate(optimizer);
        if head.is_some() {
            stats.version_installs += 1;
        }
        if debug_env::trace() && head.is_some() {
            let entry: Vec<(u16, TraceType)> = self
                .cache
                .get(index)
                .map(|trace| trace.compiled.entry_contract().to_vec())
                .unwrap_or_default();
            eprintln!(
                "jit-trace: installed version {} of the header at pc {header_pc}{}: entry {entry:?}",
                self.chain(Some(index)).len(),
                debug_env::at_line(chunk, header_pc),
            );
        }
        if let Some(site) = self.sites.existing_mut(chunk, header_pc) {
            site.head = Some(index);
            site.state = SiteState::Installed { index };
            site.rerecord = false;
        }
    }

    /// Names the code of the trace at `index` in the perf map (see
    /// [`super::perf_map`]), when `OXIMOND_PERF_MAP` asks for one.
    pub(super) fn name_in_perf_map(&self, index: u32) {
        if !super::perf_map::enabled() {
            return;
        }
        let Some(trace) = self.cache.get(index) else {
            return;
        };
        let place = trace.chunk.upgrade().map_or_else(
            || "?".to_owned(),
            |chunk| {
                let source =
                    String::from_utf8_lossy(chunk.debug.source_name().as_bytes()).into_owned();
                let pc = if trace.header_pc == super::policy::FUNCTION_ENTRY_PC {
                    0
                } else {
                    trace.header_pc
                };
                format!("{source}:{}", chunk.debug.line_at(pc).unwrap_or(0))
            },
        );
        for (position, (address, size)) in trace.compiled.code_ranges().into_iter().enumerate() {
            let side = if position == 0 {
                String::new()
            } else {
                format!(" side{position}")
            };
            let name = format!("oximond trace#{index} {place}{side}");
            super::perf_map::note(&name, address, size);
        }
    }

    /// Replaces the code of the trace at `index` with `compiled`, the code for
    /// `tree`: the trace's tree with one more side trace. Does nothing if that
    /// trace no longer exists, was evicted or flushed meanwhile, or its tree
    /// changed since the request (`version`). The code must fit the cache.
    pub(super) fn install_tree(
        &mut self,
        stats: &mut TraceStatistics,
        index: u32,
        serial: u64,
        version: u32,
        compiled: CompiledTrace,
        tree: Arc<TraceIr>,
    ) {
        let _icount = crate::jit::icount::scope("install");
        let Some(trace) = self
            .cache
            .get(index)
            .filter(|trace| trace.serial == serial && trace.version == version)
        else {
            stats.stale_completions += 1;
            return;
        };
        // Optimization may renumber guards and snapshots, so the restore
        // plans come from the new code's own IR.
        // A side may have named cells and globals the root does not: the
        // tree's window is the one the new code was lowered with.
        let externals: Box<[ExternalSlot]> = compiled.lowered_ir().external.clone().into();
        // A side may bind tables the root does not, and widening may add entry
        // inputs, so the entry plans come from the new code too.
        let chunk = trace.chunk.upgrade();
        let restores = restores_of_entered(&compiled, trace.register_count, &externals).and_then(
            |mut restores| {
                plan_unwinds(&compiled, &mut restores, chunk.as_deref())?;
                Some(restores)
            },
        );
        let function = chunk
            .as_deref()
            .and_then(|chunk| InstalledFunction::of(&tree, &compiled, chunk));
        let (Some(restores), Some(bindings), true) = (
            restores,
            plan_bindings(&compiled),
            tree.function.is_some() == function.is_some(),
        ) else {
            stats.cache_rejections += 1;
            self.fail_side(stats, index, serial, version, SideFailure::Failed(&tree));
            return;
        };
        // A side may have grown the bank of root strings.
        let roots = if tree.root_constants.len() == trace.roots.len() {
            None
        } else {
            let Some(roots) = resolve_roots(&tree) else {
                stats.cache_rejections += 1;
                self.fail_side(stats, index, serial, version, SideFailure::Failed(&tree));
                return;
            };
            Some(roots)
        };
        let (old_code, old_meta) = (trace.code_charge(), trace.meta_charge());
        let new_code = code_charge(compiled.code_size());
        let new_meta = meta_bytes_of(
            &tree,
            &compiled,
            &restores,
            roots
                .as_ref()
                .map_or(trace.roots.len(), |roots| roots.len()),
        );
        let (extra_code, extra_meta) = (
            new_code.saturating_sub(old_code),
            new_meta.saturating_sub(old_meta),
        );
        if extra_code > 0 || extra_meta > 0 {
            let Some(victims) = self
                .cache
                .plan_eviction(extra_code, extra_meta, false, &[index])
            else {
                if debug_env::trace() {
                    eprintln!(
                        "jit-trace: the tree of trace {index} with its sides refused for room ({extra_meta} more metadata bytes)"
                    );
                }
                stats.cache_rejections += 1;
                self.fail_side(stats, index, serial, version, SideFailure::Refused);
                return;
            };
            for victim in victims {
                self.evict(stats, victim);
            }
        }
        let Some(trace) = self.cache.get_mut(index) else {
            return;
        };
        // The enclosing loops the sides that landed run through.
        let mut crossed: Vec<usize> = Vec::new();
        let first = trace.ir.sides.len();
        let added = compiled.side_members().get(first..).map_or(0, |new| {
            new.iter().filter(|member| member.is_some()).count()
        });
        stats.side_installs += added as u64;
        stats.inner_sides += compiled.side_members().get(first..).map_or(0, |new| {
            new.iter()
                .zip(&tree.sides[first..])
                .filter(|(member, side)| member.is_some() && side.ir.seed_frames > 0)
                .count() as u64
        });
        stats.loop_sides += compiled.side_members().get(first..).map_or(0, |new| {
            new.iter()
                .zip(&tree.sides[first..])
                .filter(|(member, side)| {
                    member.is_some() && side.guard == crate::jit_ir::trace::TERMINAL_GUARD
                })
                .count() as u64
        });
        for (offset, key) in trace.side.compiling.iter().enumerate() {
            let landed = compiled
                .side_members()
                .get(first + offset)
                .is_some_and(std::option::Option::is_some);
            let headers = trace
                .side
                .crossed
                .iter()
                .position(|(known, _)| known == key)
                .map(|at| trace.side.crossed.swap_remove(at).1);
            if landed && let Some(headers) = headers {
                if !headers.is_empty() {
                    stats.outer_sides += 1;
                }
                for header in headers {
                    if !crossed.contains(&header) {
                        crossed.push(header);
                    }
                }
            }
        }
        // A side may pin natives and closures the root does not, behind the
        // root's lanes, and call deeper than the root does.
        trace.pinned = compiled
            .lowered_ir()
            .pinned_closures
            .clone()
            .into_boxed_slice();
        trace.max_inline_depth = tree_inline_depth(&tree);
        trace.caller = tree_caller(
            &tree,
            trace.caller_chunk.as_ref(),
            trace.caller_captured.as_deref(),
        );
        trace.function = function;
        if let Some(roots) = roots {
            trace.roots = roots;
        }
        // A side may have found a callee cell the root's captured local
        // stands for.
        trace.aliases = trace_aliases(&tree);
        trace.cells_memo = None;
        trace.written_cells = written_cell_slots(&tree);
        // A side may add the first string store through the helper.
        if trace.overwrite_version == OverwriteVersion::NotApplicable {
            trace.overwrite_version = OverwriteVersion::of_helper_calls(&tree.helper_calls());
        }
        let hoisted = compiled.optimize_stats().guards_hoisted > 0;
        trace.ir = keep_guards_of(&trace.ir, tree, hoisted, &mut trace.keep_guards_wanted);
        // The cached plan holds the old tree.
        *trace.base_plan.get_mut() = None;
        // The tree was planned when it grew. With nothing queued behind it,
        // it is the pending tree, and the next side recorded is seeded from
        // that plan (`seed_plan` checks a hit against a fresh plan in debug
        // builds).
        if let Some((planned_tree, members)) = trace.grown_plan.take()
            && Arc::ptr_eq(&planned_tree, &trace.ir)
            && trace.side.queue.queued().is_empty()
        {
            *trace.seed_plan.get_mut() = Some(SeedPlan {
                ir: Arc::clone(&trace.ir),
                queue_gen: trace.side.queue.generation(),
                tree: planned_tree,
                members,
            });
        }
        trace.version += 1;
        // Every side is part of the new code, and none is compiled apart.
        trace.side.landed = None;
        trace.detached.clear();
        trace.apart.clear();
        trace.side.no_detach.clear();
        trace.child_calls = 0;
        trace.native_calls = 0;
        trace.link_exits = 0;
        stats.clif_instructions_tree += compiled.clif_insts() as u64;
        stats.lane_array_loads_compiled += compiled.lane_array_loads() as u64;
        stats.lane_array_stores_compiled += compiled.lane_array_stores() as u64;
        stats.value_selects_compiled += compiled.value_selects() as u64;
        stats.loop_links += compiled.loop_links() as u64;
        stats.demoted_bindings += u64::from(compiled.demoted_bindings());
        if trace.side.relinking {
            stats.link_recompiles += 1;
        }
        trace.side.compiling.clear();
        trace.end_stone_recompile();
        // The new side trace gets its chance to be productive: the
        // entries counted so far ran the tree without it.
        trace.side.grace = SIDE_GRACE_WINDOWS;
        trace.stone_progress();
        trace.window_entries = 0;
        trace.window_iterations = 0;
        trace.window_loop_backs = 0;
        trace.idle_mismatches = 0;
        trace.has_upvalues = upvalue_slots(&externals) > 0;
        trace.many_upvalues = upvalue_slots(&externals) > 1;
        trace.externals = externals;
        trace.cells_memo = None;
        (trace.side_keys, trace.virtual_refused) = side_exits_of(
            &compiled,
            &restores,
            &trace.ir,
            trace.encloses,
            trace.stone.is_some(),
            trace.chunk.upgrade().as_deref(),
        );
        trace.loop_ends = loop_ends_of(&compiled);
        trace.exit_hits = vec![0; compiled.exits().len()];
        trace.cold_hits = vec![0; compiled.exits().len()];
        trace.hits_since = trace.iterations;
        trace.exit_last_hit = vec![0; compiled.exits().len()];
        trace.side.dead = compiled
            .side_members()
            .iter()
            .filter(|member| member.is_none())
            .count();
        trace.fields = plan_fields(&compiled);
        trace.meta_field = row_words_of(&trace.fields);
        trace.field_tags = trace.fields.iter().map(|(_, kind)| kind.tag()).collect();
        trace.row_memo = new_row_memo(trace.fields.len());
        trace.field_hints = vec![0; crate::runtime::shape_cache_len(trace.fields.len())];
        trace.simple_binds = simple_bindings(&compiled);
        trace.entry_plan = entry_plan_of(&compiled, &trace.ir, trace.register_count);
        // The tree's code may type its entries otherwise.
        if trace.frame_checks.is_some() {
            trace.frame_checks = frame_checks_of(&trace.ir, &compiled, trace.register_count);
        }
        trace.compiled = compiled;
        trace.restores = restores;
        trace.bindings = bindings;
        self.cache.recharge_code(old_code, new_code);
        self.cache.recharge_meta(old_meta, new_meta);
        self.name_in_perf_map(index);
        if !crossed.is_empty() {
            self.cover_loops(stats, index, &crossed);
        }
        // Sides recorded while this tree compiled join the next one.
        self.submit_queued(stats, index);
    }

    /// Joins the side `side`, compiled apart from the tree of the trace at
    /// `index`, to that tree: its exit calls the side from now on. `tree` is
    /// the tree with the side added, which the trace keeps as its source.
    /// Does nothing if that trace no longer exists, was evicted or flushed
    /// meanwhile, or its tree changed since the request (`version`); a side
    /// the tree's code cannot take goes back to the queue to compile with the
    /// tree.
    ///
    /// The new code changes nothing the tree's code does: the exit's cell is
    /// set last, after the restore plans of the side's exits, the counters
    /// and the cache's charges are in place, and it is one word that the code
    /// in flight reads once (see [`CompiledTrace::publish_attached`]).
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(super) fn install_side(
        &mut self,
        stats: &mut TraceStatistics,
        index: u32,
        serial: u64,
        version: u32,
        source: usize,
        side: crate::jit::CompiledSide,
        tree: Arc<TraceIr>,
    ) {
        let clif = side.clif_insts() as u64;
        let exit_style = side.exit_style();
        let Some(trace) = self
            .cache
            .get(index)
            .filter(|trace| trace.serial == serial && trace.version == version)
        else {
            stats.stale_completions += 1;
            return;
        };
        // Native code that called back into the VM is still running this
        // trace: its tables are lent to that run, so nothing joins the tree
        // now. The side goes back to the queue, held for a few entries (see
        // [`HELD_COMPILE_ENTRIES`]) so that it is asked for again once the run
        // is over and not at every safepoint of the run.
        if self.in_flight || self.cache.is_running(index) {
            if debug_env::trace() {
                eprintln!(
                    "jit-trace: side {source} of trace {index} waits: native code of its tree is running"
                );
            }
            if let Some(trace) = self.cache.get_mut(index) {
                trace.end_detached_compile_failed(&tree, source, false);
                let entries = trace.entries;
                trace.side.hold(entries);
            }
            return;
        }
        let chunk = trace.chunk.upgrade();
        let (old_code, old_meta) = (trace.code_charge(), trace.meta_charge());
        let Some(trace) = self.cache.get_mut(index) else {
            return;
        };
        if let Err(error) = trace.compiled.prepare_attach(side, source) {
            if debug_env::trace() {
                eprintln!("jit-trace: side {source} does not attach to its tree: {error:?}");
            }
            stats.cache_rejections += 1;
            self.fail_detached(stats, index, serial, version, source, &tree, true);
            return;
        }
        // The restore plans of the grown code, exit by exit, and what the
        // grown tree costs the cache. The plans of the exits the tree had
        // stay as they are: only the side's exits are planned.
        let mut restores = std::mem::take(&mut trace.restores);
        let kept = restores.len();
        let planned = restores_of_entered_from(
            &trace.compiled,
            kept,
            trace.register_count,
            &trace.externals,
        )
        .and_then(|added| {
            restores.extend(added);
            plan_unwinds(&trace.compiled, &mut restores, chunk.as_deref())
        });
        if planned.is_none() {
            if debug_env::trace() {
                eprintln!("jit-trace: side {source} has no restore plan");
            }
            restores.truncate(kept);
            trace.restores = restores;
            trace.compiled.cancel_attached();
            stats.cache_rejections += 1;
            self.fail_detached(stats, index, serial, version, source, &tree, true);
            return;
        }
        let new_code = trace.code_charge();
        let new_meta = meta_bytes_of(&tree, &trace.compiled, &restores, trace.roots.len());
        let (extra_code, extra_meta) = (
            new_code.saturating_sub(old_code),
            new_meta.saturating_sub(old_meta),
        );
        if extra_code > 0 || extra_meta > 0 {
            let Some(victims) = self
                .cache
                .plan_eviction(extra_code, extra_meta, false, &[index])
            else {
                if debug_env::trace() {
                    eprintln!(
                        "jit-trace: side {source} of trace {index} refused for room ({extra_meta} more metadata bytes)"
                    );
                }
                if let Some(trace) = self.cache.get_mut(index) {
                    restores.truncate(kept);
                    trace.restores = restores;
                    trace.compiled.cancel_attached();
                }
                stats.cache_rejections += 1;
                self.fail_detached(stats, index, serial, version, source, &tree, false);
                return;
            };
            for victim in victims {
                self.evict(stats, victim);
            }
        }
        let Some(trace) = self.cache.get_mut(index) else {
            return;
        };
        stats.side_installs += 1;
        stats.side_compiles += 1;
        if tree.sides[source].ir.seed_frames > 0 {
            stats.inner_sides += 1;
        }
        stats.value_selects_compiled += tree.sides[source]
            .ir
            .body
            .iter()
            .filter(|instruction| matches!(instruction.op, TraceOp::Select { .. }))
            .count() as u64;
        stats.clif_instructions_side += clif;
        if tree.sides[source].guard == crate::jit_ir::trace::TERMINAL_GUARD {
            stats.loop_sides += 1;
        }
        // The enclosing loops the side runs through.
        let mut crossed: Vec<usize> = Vec::new();
        for key in std::mem::take(&mut trace.side.compiling) {
            if let Some(at) = trace
                .side
                .crossed
                .iter()
                .position(|(known, _)| *known == key)
            {
                let headers = trace.side.crossed.swap_remove(at).1;
                if !headers.is_empty() {
                    stats.outer_sides += 1;
                }
                for header in headers {
                    if !crossed.contains(&header) {
                        crossed.push(header);
                    }
                }
            }
        }
        // A side may call deeper than the root does, or return into the
        // root's caller.
        trace.max_inline_depth = tree_inline_depth(&tree);
        trace.caller = tree_caller(
            &tree,
            trace.caller_chunk.as_ref(),
            trace.caller_captured.as_deref(),
        );
        // The side may pin closures behind the tree's lanes (a closure of the
        // caller it returns into); the tree's code names none of them, and
        // every run pins them with the rest.
        let side_pins = &tree.sides[source].ir.pinned_closures;
        if side_pins.len() > trace.pinned.len() && side_pins.starts_with(&trace.pinned) {
            trace.pinned = side_pins.clone().into_boxed_slice();
        }
        trace.aliases = trace_aliases(&tree);
        trace.cells_memo = None;
        trace.written_cells = written_cell_slots(&tree);
        if trace.overwrite_version == OverwriteVersion::NotApplicable {
            trace.overwrite_version = OverwriteVersion::of_helper_calls(&tree.helper_calls());
        }
        trace.ir = keep_guards_of(&trace.ir, tree, false, &mut trace.keep_guards_wanted);
        *trace.base_plan.get_mut() = None;
        trace.version += 1;
        trace.apart.push(source);
        trace.side.landed = Some(source);
        if exit_style {
            trace.detached.push(source);
        }
        trace.end_stone_recompile();
        // The new side gets its chance to be productive.
        trace.side.grace = SIDE_GRACE_WINDOWS;
        trace.stone_progress();
        trace.window_entries = 0;
        trace.window_iterations = 0;
        trace.window_loop_backs = 0;
        trace.idle_mismatches = 0;
        trace.restores = restores;
        (trace.side_keys, trace.virtual_refused) = side_exits_of(
            &trace.compiled,
            &trace.restores,
            &trace.ir,
            trace.encloses,
            trace.stone.is_some(),
            chunk.as_deref(),
        );
        trace.loop_ends = loop_ends_of(&trace.compiled);
        let exits = trace.compiled.exits().len();
        trace.exit_hits.resize(exits, 0);
        trace.cold_hits.resize(exits, 0);
        trace.exit_last_hit.resize(exits, 0);
        trace.side.dead = trace
            .compiled
            .side_members()
            .iter()
            .filter(|member| member.is_none())
            .count();
        trace.compiled.publish_attached();
        self.cache.recharge_code(old_code, new_code);
        self.cache.recharge_meta(old_meta, new_meta);
        self.name_in_perf_map(index);
        if !crossed.is_empty() {
            self.cover_loops(stats, index, &crossed);
        }
        // Sides recorded meanwhile are next.
        self.submit_queued(stats, index);
    }

    /// The tree at `index` now runs the loops whose headers are `headers`:
    /// an interpreter that reaches one leaves it to the tracer.
    fn cover_loops(&mut self, stats: &mut TraceStatistics, index: u32, headers: &[usize]) {
        let Some(chunk) = self
            .cache
            .get(index)
            .and_then(|trace| trace.chunk.upgrade())
        else {
            return;
        };
        for &header in headers {
            if self.sites.cover(&chunk, header)
                && let Some(trace) = self.cache.get_mut(index)
            {
                let mark = trace.iterations;
                if !trace.covered.contains(&header) {
                    trace.covered.push(header);
                    stats.covered_loops += 1;
                }
                self.sites.note_cover(&chunk, header, index, mark);
            }
        }
    }

    /// The tree at `index` no longer covers the header at `header`.
    pub(super) fn forget_covered(&mut self, index: u32, header: usize) {
        if let Some(trace) = self.cache.get_mut(index) {
            trace.covered.retain(|covered| *covered != header);
        }
    }

    /// Gives the loops a removed trace covered back to the policy.
    fn uncover_loops(
        &mut self,
        stats: &mut TraceStatistics,
        chunk: &Arc<Chunk>,
        covered: &[usize],
    ) {
        for &header in covered {
            if let Some(dropped) = self.sites.uncover(chunk, header) {
                stats.uncovered_loops += 1;
                stats.negative_evictions += dropped;
            }
        }
    }

    /// Evicts the trace at `index` to make room. Its header goes back to
    /// counting, or keeps its other versions.
    pub(super) fn evict(&mut self, stats: &mut TraceStatistics, index: u32) {
        let meta = self.cache.get(index).map(InstalledTrace::meta_parts);
        let record = self
            .cache
            .get(index)
            .filter(|_| debug_env::trace())
            .map(|trace| {
                format!(
                    "{} iterations in {} entries, {} sides{}",
                    trace.iterations,
                    trace.entries,
                    trace.live_sides(),
                    trace.hottest_exits()
                )
            });
        let Some(removed) = self.remove_version(index) else {
            return;
        };
        stats.cache_evictions += 1;
        if debug_env::trace()
            && let Some(chunk) = &removed.chunk
        {
            eprintln!(
                "jit-trace: evicted the trace at pc {}{} for room ({:?} metadata bytes as (IR, lowered IR, restore plans) over {} members; cache holds {} of {} metadata bytes; {})",
                removed.header_pc,
                debug_env::at_line(chunk, removed.header_pc),
                meta.map(|(parts, _)| parts),
                meta.map_or(0, |(_, members)| members),
                self.cache.meta_bytes(),
                self.cache.limits().meta_bytes,
                record.unwrap_or_default(),
            );
        }
        if let Some(chunk) = &removed.chunk {
            self.uncover_loops(stats, chunk, &removed.covered);
        }
        if removed.versions_remain {
            return;
        }
        if let Some(chunk) = removed.chunk
            && let Some(site) = self.sites.existing_mut(&chunk, removed.header_pc)
            && site.state == (SiteState::Installed { index })
        {
            // Eviction picks cold traces, so the header must get hot again
            // before it records.
            site.state = SiteState::Cold { backedges: 0 };
        }
    }

    /// Drops the traces of chunks that no longer exist.
    pub(super) fn sweep_dead_traces(&mut self) {
        for index in self.cache.dead_indexes() {
            let _ = self.remove_version(index);
        }
    }

    /// Removes the trace at `index` from the cache and from its header's
    /// version chain. Other versions keep the header: its state returns to
    /// `Installed` on the newest of them.
    fn remove_version(&mut self, index: u32) -> Option<RemovedVersion> {
        let trace = self.cache.remove(index)?;
        let chunk = trace.chunk.upgrade();
        let mut versions_remain = false;
        if let Some(chunk) = &chunk
            && let Some(site) = self.sites.existing_mut(chunk, trace.header_pc)
        {
            if site.head == Some(index) {
                site.head = trace.next;
            } else {
                let mut at = site.head;
                let mut steps = 0;
                while let Some(older) = at {
                    steps += 1;
                    let Some(entry) = self.cache.get_mut(older) else {
                        break;
                    };
                    if entry.next == Some(index) {
                        entry.next = trace.next;
                        break;
                    }
                    at = entry.next;
                    if steps > MAX_TRACE_VERSIONS + 1 {
                        break;
                    }
                }
            }
            versions_remain = site.head.is_some();
            if let Some(head) = site.head
                && site.state == (SiteState::Installed { index })
            {
                site.state = SiteState::Installed { index: head };
            }
        }
        Some(RemovedVersion {
            chunk,
            header_pc: trace.header_pc,
            versions_remain,
            covered: trace.covered,
        })
    }

    /// A trace could not be installed because the cache or the broker had no
    /// room. The site goes back to its other versions, or waits out a
    /// cooldown. Nothing counts against its retry budget: the trace was
    /// fine.
    pub(super) fn back_off_refused(&mut self, chunk: &Arc<Chunk>, header_pc: usize, cooldown: u32) {
        let Some(site) = self.sites.existing_mut(chunk, header_pc) else {
            return;
        };
        match site.head {
            Some(head) => site.state = SiteState::Installed { index: head },
            None => {
                site.state = SiteState::Cooldown {
                    remaining: cooldown,
                };
            }
        }
    }

    /// The error that ended the last run with [`TraceRun::Failed`].
    pub(crate) fn take_restore_error(&mut self) -> RuntimeError {
        self.restore_error
            .take()
            .unwrap_or_else(|| RuntimeError::message("a trace exit failed to restore"))
    }

    /// Runs the newest version of a header's trace, or an older one when the
    /// frame's types do not match it. `index` is the header's newest
    /// version, and the frame at `frame_index` of `frames` is at the loop
    /// header. `fuel` bounds the loop-backs before a poll exit. An exit
    /// inside an inlined call leaves the callee frames pushed above it.
    ///
    /// The run takes raw pointers because a trace that resumes coroutines
    /// runs the interpreter in the middle of it (see [`ResumeHost`]), which
    /// borrows the VM and the call stack as a whole. No reference to
    /// `*this`, `*stats`, `*frames` or `*runtime` lives across a native
    /// call: `run_pinned` makes them where it needs them and again after the
    /// call.
    ///
    /// # Safety
    ///
    /// `this`, `stats`, `frames` and `runtime` are valid and unaliased for the
    /// call, except for the VM that owns them, whose address
    /// ([`Self::set_resume_host`]) the helper of a resuming trace may use
    /// meanwhile. They must all derive from the one pointer that address
    /// does, or from the references that pointer was made from.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn enter_lent(
        this: *mut Self,
        stats: *mut TraceStatistics,
        index: u32,
        frames: *mut CallStack,
        frame_index: usize,
        fuel: i64,
        runtime: *mut Runtime,
    ) -> TraceRun {
        let mut at = index;
        // Whether every version that turned the frame away did so on its
        // register tags alone.
        let mut refused = true;
        // SAFETY: `this` is valid; the borrow ends here.
        let versions = unsafe { (*this).cache.version_limit() };
        for _ in 0..=versions {
            // SAFETY: as the caller promised.
            let attempt =
                unsafe { Self::run_version(this, stats, at, frames, frame_index, fuel, runtime) };
            // SAFETY: the run is over: nothing else uses these.
            let (policy, stats, frames) = unsafe { (&mut *this, &mut *stats, &*frames) };
            match attempt {
                Attempt::Done { run, retire } => {
                    if let Some(newest) = policy.cache.get_mut(index) {
                        newest.mismatch_streak = 0;
                    }
                    if policy
                        .cache
                        .get_mut(at)
                        .is_some_and(InstalledTrace::held_too_long)
                    {
                        policy.submit_queued(stats, at);
                    }
                    if let Some(ran) = policy.cache.get_mut(at)
                        && ran.overwrite_version == OverwriteVersion::Wanted
                    {
                        ran.overwrite_version = OverwriteVersion::Asked;
                        policy.ask_for_version(index);
                    }
                    if retire {
                        policy.retire(stats, at);
                    } else {
                        policy.consider_kept_guards(at);
                        policy.consider_relink(at);
                        policy.consider_peak(at);
                    }
                    if policy.side_candidate.is_some() {
                        // SAFETY: as above.
                        let runtime = unsafe { &*runtime };
                        policy.consider_side(stats, frames, frame_index, runtime);
                    }
                    return run;
                }
                Attempt::Mismatch { next } | Attempt::Refused { next } => {
                    refused &= matches!(attempt, Attempt::Refused { .. });
                    match next {
                        Some(next) => at = next,
                        None => break,
                    }
                }
            }
        }
        // SAFETY: as above.
        unsafe { (*this).entry_mismatched(&mut *stats, index, refused) }
    }

    /// [`Self::enter_lent`] for callers that hold references and lend no
    /// resume host: tests, which run no VM around the policy.
    #[cfg(test)]
    pub(crate) fn enter(
        &mut self,
        stats: &mut TraceStatistics,
        index: u32,
        frames: &mut CallStack,
        frame_index: usize,
        fuel: i64,
        runtime: &mut Runtime,
    ) -> TraceRun {
        debug_assert_eq!(self.resume_vm, 0, "a policy with a resume host enters lent");
        // SAFETY: the references are unique and nothing resumes meanwhile.
        #[allow(unsafe_code)]
        unsafe {
            Self::enter_lent(self, stats, index, frames, frame_index, fuel, runtime)
        }
    }

    /// A branch guard of a trace has failed often. Starts a side recording
    /// for it, if the tree has room, the guard has attempts left and nothing
    /// else is recording. The exit just restored the frame at the guard's
    /// resume pc, so the recording begins with the interpreter's next
    /// instruction.
    fn consider_side(
        &mut self,
        stats: &mut TraceStatistics,
        frames: &CallStack,
        frame_index: usize,
        runtime: &Runtime,
    ) {
        let _icount = crate::jit::icount::scope("consider-side");
        let Some(candidate) = self.side_candidate.take() else {
            return;
        };
        // The exit of a function trace's nested activation left that many
        // frames above the entered one.
        let frame_index = frame_index + candidate.activation;
        // The frame the side starts in: the root frame, or the innermost of
        // the inlined calls the exit pushed.
        let top = frame_index + candidate.inner;
        if top >= frames.len() {
            return;
        }
        let _phase = alloc_phase::enter(Phase::Recording);
        if self.recorder.is_some() || !self.compile {
            return;
        }
        // The broker has no room for the tree's recompile.
        if self.broker_full_skips > 0 {
            self.broker_full_skips -= 1;
            return;
        }
        let Some(trace) = self.cache.get(candidate.index) else {
            return;
        };
        let (Some(chunk), Some(profile)) = (trace.chunk.upgrade(), trace.profile) else {
            return;
        };
        let ir = &trace.ir;
        let root_header = trace.header_pc;
        let allowed = trace.serial == candidate.serial
            && !trace.side.has(candidate.key)
            && trace.live_sides() + trace.side.in_progress() < trace.side.max_sides
            && trace.side.attempts(candidate.key) < MAX_SIDE_ATTEMPTS
            && side_depth(ir, candidate.key.parent) <= MAX_SIDE_DEPTH
            && frames.at(top).frame.pc == candidate.pc
            && std::ptr::eq(
                frames.at(frame_index).frame.chunk.address(),
                Arc::as_ptr(&chunk),
            );
        if !allowed {
            return;
        }
        // One more side for one more closure at a call site whose guards
        // sides already continue: the loop records again with that call
        // guarded on the prototype, unless its site already names the call
        // (the recording could not guard it so) or gave the prototype guard
        // up.
        if let Some(path) = trace.poly_side_path(candidate.key)
            && trace.callee_shares_prototype(candidate.key, frames.at(top), runtime)
            && !self.cache.is_running(candidate.index)
            && self
                .sites
                .existing_mut(&chunk, root_header)
                .is_some_and(|site| !site.poly_refused && !site.poly_calls.contains(&path))
        {
            self.rerecord_for_poly_call(stats, candidate.index, path);
            return;
        }
        // A side that starts inside inlined calls: only for a loop trace, from
        // a guard (no loop side), while the tree's budget for such sides
        // lasts.
        let seeded = candidate.inner > 0;
        // The loop side of a loop in a call (see "Loops in calls") starts in
        // the call's frames from the terminal exit in front of it.
        let terminal = candidate.key.origin == crate::jit_ir::trace::TERMINAL_GUARD;
        if seeded
            && (ir.function.is_some()
                || (terminal && !ends_in_front_of_loop(ir))
                || candidate.activation > 0
                || !trace.seeds_left())
        {
            return;
        }
        // A function trace's side runs to the activation's return: the whole
        // function is its span.
        let function = ir.function.is_some();
        let latch = if function {
            Some(chunk.code.len().saturating_sub(1))
        } else {
            ir.poll_snapshot
                .and_then(|poll| match ir.snapshots[poll.index()].phase {
                    crate::jit_ir::trace::SnapshotPhase::After { pc, .. } => Some(pc),
                    crate::jit_ir::trace::SnapshotPhase::Before { .. } => None,
                })
        };
        let Some(latch) = latch else {
            return;
        };
        // A side that runs on through the loops around the root saves the
        // entry and exit the root takes for every outer iteration, at the
        // price of compiling the tree again. Where the entry is a small share
        // of the work of an integer loop, that price is not paid back.
        let outside_loop =
            !function && !seeded && !(trace.header_pc..=latch).contains(&candidate.pc);
        // The terminal exit of a side that ended in front of a loop: the
        // loop is recorded as a side of its own, wherever it sits. None of
        // the questions about the loops around the root apply to it.
        let terminal = candidate.key.origin == crate::jit_ir::trace::TERMINAL_GUARD;
        if terminal
            && (function
                || !(trace.encloses || ends_in_front_of_loop(ir))
                || candidate.last.is_none())
        {
            return;
        }
        let outside_loop = outside_loop && !terminal;
        // The frame that called the root frame, when a side that leaves the
        // root frame by its return may run on in it: the entry and exit of
        // every call is what it saves, which an outer side through the loops
        // around the root does not.
        let caller_frame = caller_seed_of(
            frames,
            frame_index,
            !frames
                .open_in(
                    frames.at(frame_index).frame.base,
                    frames.at(frame_index).frame.top,
                )
                .is_empty(),
        )
        .filter(|seed| {
            !seeded
                && trace
                    .caller_chunk
                    .as_ref()
                    .is_none_or(|held| std::ptr::eq(held.as_ptr(), Arc::as_ptr(&seed.chunk)))
        });
        // Until the tree has run enough entries the side does not pay for
        // the compile of the grown tree: the guard waits, with its attempts.
        let waiting = caller_frame.is_some()
            && trace.entries < RETURN_THROUGH_ENTRIES
            && nest::enclosing_loops(
                &chunk,
                nest::LoopSpan {
                    header: trace.header_pc,
                    latch,
                },
            )
            .last()
            .is_some_and(|outermost| candidate.pc > outermost.latch);
        if waiting {
            return;
        }
        // Until the tree has run enough entries the side waits for a caller
        // to link; after that, a return no caller takes (the seed drops it)
        // ends the side with an exit.
        let caller_possible = caller_frame.is_some() && trace.entries < RETURN_THROUGH_ENTRIES;
        let caller = caller_frame.filter(|_| trace.entries >= RETURN_THROUGH_ENTRIES);
        if caller.is_none()
            && trace.iterations < SIDE_BEYOND_ITERATIONS
            && trace.side.held_for_return.contains(&candidate.key)
        {
            return;
        }
        // The guard of an outer side at the outermost loop's latch ends the
        // nest: its side runs beyond it, which waits until the tree has run
        // long enough (see `SIDE_BEYOND_ITERATIONS`). The guard keeps its
        // attempts. A latch that loops back now (a `while` loop's jump, a
        // numeric for latch with trips left, where the exit of an inner loop
        // that ends the outer body resumes on every outer iteration) keeps
        // its side in the nest, which does not wait.
        if outside_loop
            && trace.encloses
            && candidate.key.parent.is_some()
            && trace.iterations < SIDE_BEYOND_ITERATIONS
            && nest::enclosing_loops(
                &chunk,
                nest::LoopSpan {
                    header: trace.header_pc,
                    latch,
                },
            )
            .last()
            .is_some_and(|outermost| outermost.latch == candidate.pc)
            && !latch_loops_back(&chunk, candidate.pc, frames.at(frame_index).registers)
        {
            return;
        }
        if debug_env::trace() && outside_loop {
            eprintln!(
                "jit-trace: outer exit of the trace at pc {}{} (loops enclose it: {}, {} iterations in {} entries, integers: {})",
                trace.header_pc,
                debug_env::at_line(&chunk, trace.header_pc),
                trace.encloses,
                trace.iterations,
                trace.entries,
                trace.integers
            );
        }
        // The loop around the root may have a trace of its own, or be about
        // to get one: it is still counting and has taken at least twice as
        // many backedges as this exit fired, so most of its iterations skip
        // the root's loop. The exit then leaves for the interpreter a couple
        // of instructions before that trace takes over, and an outer side
        // would only compile a second copy of the outer loop into this tree.
        let fired = trace
            .side_keys
            .iter()
            .zip(&trace.exit_hits)
            .filter(|(key, _)| **key == Some(candidate.key))
            .map(|(_, hits)| *hits)
            .max()
            .unwrap_or(0);
        // While the outer loop is recording or compiling its first trace, the
        // answer is a moment away: wait for it without spending an attempt.
        let mut outer_settling = false;
        let outer_owned = trace.encloses
            && outside_loop
            && nest::enclosing_loops(
                &chunk,
                nest::LoopSpan {
                    header: trace.header_pc,
                    latch,
                },
            )
            .first()
            .is_some_and(|outer| {
                self.sites
                    .existing_mut(&chunk, outer.header)
                    .is_some_and(|site| {
                        site.head.is_some()
                            || match site.state {
                                SiteState::Recording | SiteState::Compiling => {
                                    outer_settling = true;
                                    true
                                }
                                SiteState::Installed { .. } => true,
                                SiteState::Cold { backedges } => {
                                    let since =
                                        backedges.saturating_sub(trace.outer_backedges_at_install);
                                    u64::from(since) >= 2 * u64::from(fired)
                                }
                                _ => false,
                            }
                    })
            });
        if outer_settling {
            return;
        }
        if trace.encloses
            && outside_loop
            && (outer_owned || !(trace.outer_side_pays() || caller.is_some()))
        {
            // One attempt, not all: the averages that decided this move on,
            // and a later guard failure asks again.
            if let Some(trace) = self.cache.get_mut(candidate.index) {
                trace.side.spend(candidate.key, 1);
            }
            return;
        }
        // The captured locals of the frame the side enters. A tree that reads
        // and writes callee cells through closures may not also hold one of
        // them in a register.
        let captured: Vec<(u8, CellHandle)> = {
            let entered = &frames.at(frame_index).frame;
            frames
                .open_in(entered.base, entered.top)
                .iter()
                .filter_map(|open| Some((u8::try_from(open.slot - entered.base).ok()?, open.cell)))
                .collect()
        };
        let place = SeedPlace {
            key: candidate.key,
            pc: candidate.pc,
            last: candidate.last,
            header: if function { 0 } else { trace.header_pc },
            latch,
            encloses: trace.encloses,
            beyond: trace.iterations >= SIDE_BEYOND_ITERATIONS,
            captured: captured.clone(),
            closure: frames.at(frame_index).frame.closure,
            return_exit: !caller_possible,
            caller,
            depth: frames.len() - candidate.inner,
            inner: (frame_index + 1..=top)
                .map(|at| {
                    let live = frames.at(at);
                    LiveFrame {
                        closure: live.frame.closure,
                        chunk: Arc::clone(&live.frame.chunk.as_arc()),
                        registers: live.registers.to_vec(),
                    }
                })
                .collect(),
            root_registers: if seeded {
                frames.at(frame_index).registers.to_vec()
            } else {
                Vec::new()
            },
            root_cells: if seeded {
                frames.at(frame_index).frame.upvalues.to_vec()
            } else {
                Vec::new()
            },
            root_environment: seeded.then(|| frames.at(frame_index).frame.environment),
            lazy: trace.lazy.clone(),
        };
        // The guard's pc when it is the test of the root loop's latch, in the
        // root frame: a side of it that resumes at the header links back at
        // once (see `Recorder::close_side_at_start`). A terminal exit is no
        // guard of the latch: its side is a loop side.
        let member = match candidate.key.parent {
            None => Some(&**ir),
            Some(parent) => ir.sides.get(parent as usize).map(|side| &*side.ir),
        };
        let latch_guard = (!function
            && candidate.key.origin != crate::jit_ir::trace::TERMINAL_GUARD)
            .then(|| {
                member?
                    .guards
                    .iter()
                    .find(|guard| guard.origin == candidate.key.origin)
            })
            .flatten()
            .filter(|guard| guard.site.frame == FrameId(0) && guard.site.pc == latch)
            .map(|guard| guard.site.pc);
        let Some(planned) = trace.installed_plan() else {
            return;
        };
        let Some((seed, header, latch)) = side_seed(ir, &planned, &place) else {
            if seeded && let Some(trace) = self.cache.get_mut(candidate.index) {
                // The calls are ones no side can rebuild: the guard keeps its
                // exit for good.
                trace.side.spend(candidate.key, MAX_SIDE_ATTEMPTS);
            }
            return;
        };
        let depth = place.depth;
        let liveness = self.liveness_of(&chunk);
        // A side of a root that unrolls short loops unrolls them too: a
        // variant of the iteration has loops of its own that are as short as
        // the ones the root's recording unrolled, and one it cannot follow
        // would keep the variant out of the tree for good.
        let unroll = !function
            && self
                .sites
                .existing_mut(&chunk, root_header)
                .is_some_and(|site| {
                    (site.inner_loop_met || outside_loop) && !site.unroll_overflowed
                });
        let (rowless, poly_calls, keep_guards, callee_loop) = self
            .sites
            .existing_mut(&chunk, root_header)
            .map_or((false, Vec::new(), false, None), |site| {
                (
                    site.rowless,
                    site.poly_calls.clone(),
                    site.keep_guards,
                    site.callee_loop.clone().filter(|_| !function),
                )
            });
        // The entered frame's captured locals, so that a callee cell the side
        // reads can be told from the root's own variable.
        let short_loops = if unroll {
            self.short_loops_of(&chunk)
        } else {
            Vec::new()
        };
        let recorder = Recorder::new_side(chunk, depth, header, latch, profile, seed)
            .with_short_loops(short_loops)
            .with_callee_loop(callee_loop)
            .with_rowless(rowless)
            .with_poly_calls(poly_calls)
            .with_kept_guards(keep_guards)
            .with_latch_guard(latch_guard)
            .with_limits(self.record_limits)
            .with_captured(captured)
            .with_liveness(liveness)
            .with_rebuilt_callee_frames(self.rebuild_callee_frames)
            .with_value_selects(self.value_selects)
            .with_loop_unrolling(unroll);
        self.recorder = Some(Box::new(recorder));
        self.side_target = Some(super::SideTarget {
            index: candidate.index,
            serial: candidate.serial,
            key: candidate.key,
        });
        if let Some(trace) = self.cache.get_mut(candidate.index) {
            trace.side.recording = Some(candidate.key);
        }
        stats.side_recordings += 1;
    }

    /// Hands the registers that entries of the header whose newest version
    /// is `index` found nil on some entries and a number on others (see
    /// [`InstalledTrace::entry_flips`]) to the site, whose next recording
    /// takes them as inputs of either type.
    fn note_lazy(&mut self, index: u32) -> bool {
        if !self.untyped_inputs {
            return false;
        }
        let Some((Some(chunk), header_pc)) = self
            .cache
            .get(index)
            .map(|trace| (trace.chunk.upgrade(), trace.header_pc))
        else {
            return false;
        };
        let mut flips: Vec<u16> = Vec::new();
        for version in self.chain(Some(index)) {
            if let Some(trace) = self.cache.get(version) {
                flips.extend(&trace.entry_flips);
            }
        }
        // The registers the newest version turned entries away for although
        // the site lists them: its recording started before the list named
        // them (the list grows while a recording runs), so it typed them.
        let typed_listed: Vec<u16> = self
            .cache
            .get(index)
            .map(|trace| {
                trace
                    .entry_flips
                    .iter()
                    .copied()
                    .filter(|register| {
                        trace
                            .compiled
                            .entry_contract()
                            .iter()
                            .any(|(input, ty)| input == register && *ty != TraceType::Any)
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut grew = false;
        if let Some(site) = self.sites.existing_mut(&chunk, header_pc) {
            // Such a version is stale like one the list outgrew: it makes
            // way once per register, which bounds the churn when no
            // recording can take the register untyped.
            for register in typed_listed {
                if site.lazy.contains(&register) && !site.lazy_retaken.contains(&register) {
                    site.lazy_retaken.push(register);
                    grew = true;
                }
            }
            for register in flips {
                if !site.lazy.contains(&register) {
                    site.lazy.push(register);
                    grew = true;
                }
            }
            // The stepping stones that gave up typed a register the loop's
            // entries hold as either type: their outer sides could not link
            // with the other type. A root that takes it untyped is judged
            // as a stone afresh.
            if grew {
                site.stone_give_ups = 0;
            }
        }
        grew
    }

    /// Has the header whose newest version is `index` record another
    /// version at its next consult, while the cap allows.
    fn ask_for_version(&mut self, index: u32) {
        let Some((Some(chunk), header_pc)) = self
            .cache
            .get(index)
            .map(|trace| (trace.chunk.upgrade(), trace.header_pc))
        else {
            return;
        };
        let versions = self.chain(Some(index)).len();
        let limit = self.cache.limits().versions;
        if let Some(site) = self.sites.existing_mut(&chunk, header_pc)
            && versions < limit
            && !site.versions_closed
            && site.state == (SiteState::Installed { index })
        {
            site.rerecord = true;
        } else if debug_env::trace()
            && self
                .cache
                .get(index)
                .is_some_and(|trace| trace.mismatch_streak == TRACE_VERSION_MISMATCH_STREAK)
            && let Some(site) = self.sites.existing_mut(&chunk, header_pc)
        {
            eprintln!(
                "jit-trace: no other version for the header at pc {header_pc}{}: {versions} of {limit} versions, closed {}, state {:?}",
                debug_env::at_line(&chunk, header_pc),
                site.versions_closed,
                site.state
            );
        }
    }

    /// No version matched the frame's types. This counts as an unproductive
    /// entry of the newest version. A streak of them asks for another
    /// version recorded under the types seen now, while the cap allows.
    /// `refused` says every version turned the frame away on its register
    /// tags, before marshalling anything (see [`InstalledTrace::note_mismatch`]).
    fn entry_mismatched(
        &mut self,
        stats: &mut TraceStatistics,
        index: u32,
        refused: bool,
    ) -> TraceRun {
        // One entry, one exit: the entry checks ran and every version said
        // no.
        stats.native_entries += 1;
        stats.exits[TraceExitKind::EntryMismatch.index()] += 1;
        let Some(trace) = self.cache.get_mut(index) else {
            return TraceRun::NotEntered;
        };
        trace.mismatch_streak = trace.mismatch_streak.saturating_add(1);
        let streak = trace.mismatch_streak;
        let mut retire = trace.note_mismatch(refused);
        // A stepping stone that most entries miss gives up (see
        // `InstalledTrace::keep_stone`): its windows close only on the
        // entries that match, and those may well look hopeful.
        trace.mismatches += 1;
        if trace.stone.is_some() {
            retire |= trace.mismatches - trace.stone_start.1 >= STONE_PATIENCE_ENTRIES
                && trace.mostly_mismatched();
        }
        if streak >= TRACE_VERSION_MISMATCH_STREAK {
            // A register that turned out to hold nil and numbers makes the
            // versions typed for it stale: with the cap reached, the oldest
            // gives way to one that takes the register untyped. The set of
            // such registers only grows, so this ends.
            if self.note_lazy(index)
                && let chain = self.chain(Some(index))
                && chain.len() >= self.cache.limits().versions
                && let Some(&oldest) = chain.last()
                && oldest != index
                && let Some(removed) = self.remove_version(oldest)
                && let Some(chunk) = &removed.chunk
            {
                self.uncover_loops(stats, chunk, &removed.covered);
            }
            self.ask_for_version(index);
        }
        if retire {
            if debug_env::trace() {
                eprintln!(
                    "jit-trace: the trace at pc {} retires: {} entries matched none of its header's versions",
                    self.cache.get(index).map_or(0, |trace| trace.header_pc),
                    self.cache.get(index).map_or(0, |trace| trace.mismatches),
                );
            }
            self.retire(stats, index);
        }
        TraceRun::NotEntered
    }

    /// Pins version `at`, tries it, and unpins it. The version cannot be
    /// evicted or flushed away while its code runs.
    ///
    /// # Safety
    ///
    /// As [`Self::enter_lent`].
    #[allow(unsafe_code)]
    unsafe fn run_version(
        this: *mut Self,
        stats: *mut TraceStatistics,
        at: u32,
        frames: *mut CallStack,
        frame_index: usize,
        fuel: i64,
        runtime: *mut Runtime,
    ) -> Attempt {
        // SAFETY: `this` is valid and no reference to it lives across the
        // run, which makes its own.
        unsafe {
            (*this).cache.begin_run(at);
            (*this).in_flight = true;
        }
        // SAFETY: as the caller promised.
        let attempt =
            unsafe { Self::run_pinned(this, stats, at, frames, frame_index, fuel, runtime) };
        // SAFETY: the run is over.
        unsafe {
            (*this).in_flight = false;
            // Code a deferred flush dropped is freed here, after the run.
            drop((*this).cache.end_run());
        }
        attempt
    }

    /// # Safety
    ///
    /// As [`Self::enter_lent`]. Across the native call the run keeps no
    /// `&mut` to the policy, the cache, the cache entry, the VM, the runtime,
    /// the bank or the stack. It keeps:
    ///
    /// - the policy's buffers for the entry slots, the binding views, the
    ///   exit area, the table space and the table lanes (the last two are
    ///   boxed, so that what the run borrows is off the policy). A helper
    ///   that resumes a coroutine runs with native execution suspended and
    ///   flushes deferred (see [`Vm::resume_for_trace`]), so nothing it runs
    ///   enters, records, installs, evicts or retires a trace, or otherwise
    ///   touches these.
    /// - shared references to single fields of the cache entry (fields,
    ///   bindings, roots, compiled code) that nothing writes during the run,
    ///   and the raw words of its field hints, which only the run writes.
    /// - `RunTables`' raw pointers to the runtime and the bank, which the
    ///   helper's code does use, but only through the same pointers.
    ///
    /// Every other field of the policy, the cache entry, the statistics, the
    /// runtime and the stack are taken again from the raw pointers once the
    /// call returns.
    #[allow(
        unsafe_code,
        clippy::too_many_lines,
        reason = "one run: entry checks, the native call, the exit's restore and the counters"
    )]
    unsafe fn run_pinned(
        this: *mut Self,
        stats_at: *mut TraceStatistics,
        index: u32,
        frames_at: *mut CallStack,
        frame_index: usize,
        fuel: i64,
        runtime_at: *mut Runtime,
    ) -> Attempt {
        // SAFETY: the caller's promise. These are field borrows made through
        // the raw pointer, none of them of a field the resumed code uses.
        let (
            cache,
            inputs,
            binding_views,
            probe_scratch,
            table_space,
            table_lanes,
            exit_area,
            bank,
            hooks_installed,
            hook_budget,
            resume_vm,
            entry_reads,
            lane_index,
        ) = unsafe {
            (
                &mut (*this).cache,
                &mut (*this).inputs,
                &mut (*this).binding_views,
                &mut (*this).probe_scratch,
                &mut (*this).table_space,
                &mut (*this).table_lanes,
                &mut (*this).exit_area,
                &mut (*this).bank,
                &(*this).hooks_installed,
                &(*this).hook_budget,
                &(*this).resume_vm,
                &mut (*this).entry_reads,
                &mut (*this).lane_index,
            )
        };
        // SAFETY: as above.
        let (frames, runtime) = unsafe { (&mut *frames_at, &mut *runtime_at) };
        let Some(trace) = cache.get_mut(index) else {
            return Attempt::Done {
                run: TraceRun::NotEntered,
                retire: false,
            };
        };
        // A placeholder holds no code: its tree is still being recorded or
        // compiled. One that is entered again and again with nothing pending
        // was left behind (its compile failed twice, or the broker never
        // took the request), and is dropped like a trace that never paid.
        if trace.compiled.is_staged() {
            let pending = trace.side.busy() || trace.side.stone == StoneRecompile::InFlight;
            if !pending {
                trace.staged_visits += 1;
            }
            if trace.staged_visits == STAGED_PATIENCE && debug_env::trace() {
                eprintln!(
                    "jit-trace: the placeholder at pc {} retires: entered {} times with nothing pending",
                    trace.header_pc, trace.staged_visits,
                );
            }
            return Attempt::Done {
                run: TraceRun::NotEntered,
                retire: trace.staged_visits >= STAGED_PATIENCE,
            };
        }
        // Under a count hook only a counted trace runs, and for no more
        // iterations than the hook's count covers: the code never tests the
        // budget, so the fuel is what keeps every iteration inside it. The
        // interpreter fires the hook at the instruction that exhausts the
        // count, which no iteration of this run reaches. A count too small
        // for one iteration leaves the loop to the interpreter, without the
        // cost of an entry that would leave at once.
        let counted = trace.compiled.counted();
        let mut fuel = fuel;
        let mut budget_limited = false;
        if *hooks_installed && !counted {
            return Attempt::Done {
                run: TraceRun::NotEntered,
                retire: false,
            };
        }
        if let Some(budget) = *hook_budget {
            let iterations = (budget - 1) / i64::from(trace.compiled.max_counted()).max(1);
            if iterations < 1 {
                return Attempt::Done {
                    run: TraceRun::NotEntered,
                    retire: false,
                };
            }
            if iterations < fuel {
                fuel = iterations;
                budget_limited = true;
            }
        }
        // A trace that stores string keys through the helper polls early
        // until the policy knows whether to ask for the in-place version.
        trace.overwrite_polling = 0;
        if trace.overwrite_version == OverwriteVersion::Watching && !budget_limited {
            if trace.overwrite_rest > 0 {
                trace.overwrite_rest -= 1;
            } else if fuel > TRACE_OVERWRITE_WATCH_FUEL {
                fuel = TRACE_OVERWRITE_WATCH_FUEL;
                trace.overwrite_polling = 1;
            }
        }
        // A function trace whose tree is about to grow a side waits for it:
        // until then each run leaves at that guard, from as deep as the
        // recursion went, and every exit rebuilds the frames of the
        // activations below it, whose own self calls then enter for little
        // work. The interpreter runs the recursion meanwhile. So does a loop
        // inside other loops that leaves after a trip or two on every outer
        // iteration, while the side that carries it through them compiles.
        if trace.side.compiling_or_queued() && (trace.function.is_some() || trace.runs_short()) {
            // A compile that is held for a guard about to ask for its side
            // waits for entries (`held_too_long`), and a refused visit is
            // not one: without this count the wait never ends while the
            // queued side is what the refusals wait for.
            if trace.side.held_at.is_some() {
                trace.side.held_refused += 1;
            } else {
                trace.side.held_refused = 0;
            }
            return Attempt::Done {
                run: TraceRun::NotEntered,
                retire: false,
            };
        }
        let depth = frames.len();
        // The frame's captured locals: open upvalues, which name their
        // registers by stack slot. Usually none, so usually no allocation.
        let captured: Vec<crate::vm::stack::OpenUpvalue> = {
            let record = &frames[frame_index];
            let open = frames.open_in(record.base, record.top);
            if open.is_empty() {
                Vec::new()
            } else {
                open.to_vec()
            }
        };
        // A tree that returns into the root frame's caller reads the caller's
        // registers, and runs only for the caller it was recorded for.
        let caller_view = trace
            .caller
            .as_ref()
            .and_then(|entry| caller_view(frames, frame_index, entry, &trace.pinned));
        let caller_fits =
            trace.caller.is_none() || (caller_view.is_some() && !trace.compiled.resumes());
        let frame = frames.at_mut(frame_index);
        // A function trace is entered at its frame's first instruction.
        let entry_pc = if trace.function.is_some() {
            0
        } else {
            trace.header_pc
        };
        let fits = std::ptr::eq(trace.chunk.as_ptr(), frame.chunk.address())
            && frame.pc == entry_pc
            && frame.registers.len() == trace.register_count
            && caller_fits
            && (trace.caller.is_none() || captured.is_empty())
            && entered_as_recorded(&frame, &captured, &trace.ir);
        if !fits {
            // The frame never reached native code. It still counts as an
            // entry that achieved nothing.
            let retire = trace.note_entry(0, 0);
            if retire && debug_env::trace() {
                eprintln!(
                    "jit-trace: the trace at pc {} retires: its entries do not fit the frame (chunk {}, pc {}, registers {} against {}, caller {}, captured {}, as recorded {})",
                    trace.header_pc,
                    std::ptr::eq(trace.chunk.as_ptr(), frame.chunk.address()),
                    frame.pc == entry_pc,
                    frame.registers.len(),
                    trace.register_count,
                    caller_fits,
                    captured.len(),
                    entered_as_recorded(&frame, &captured, &trace.ir),
                );
            }
            return Attempt::Done {
                run: TraceRun::NotEntered,
                retire,
            };
        }

        // An exit inside the deepest inlined call pushes that many frames.
        if depth + trace.max_inline_depth > MAX_LUA_CALL_FRAMES {
            return Attempt::Mismatch { next: trace.next };
        }
        // The frame's own registers turn most mismatched entries away (a
        // local that is nil on one outer iteration and a number on the
        // next). Their tags are checked here, a compare each, before the
        // bank, the externals, the bindings and the native call cost an
        // entry that the code's own check would refuse anyway.
        if let Some(failed) = frame_mismatch(trace, &frame) {
            // Only untyped inputs read the flips, and the trace log the
            // tags: a refusal otherwise costs the scan alone.
            // SAFETY: a plain read of a flag no borrow above covers.
            if unsafe { (*this).untyped_inputs } || debug_env::trace() {
                frame_tags(trace, &frame, inputs);
                note_entry_flips(trace, inputs, failed);
            }
            return Attempt::Refused { next: trace.next };
        }
        // A coroutine the trace resumes may write a local of this frame that
        // one of its closures captured: the register is the variable. A
        // register the trace reads is checked again after each resume; one it
        // writes lives in a native value until an exit stores it, which the
        // thread would not see.
        if trace.compiled.resumes()
            && !trace.ir.resumes_only_in_vm_calls()
            && resume_writes_captured(frame.base, &captured, &trace.ir.written_registers)
        {
            return Attempt::Mismatch { next: trace.next };
        }
        // A function trace nests activations, each of which an exit may
        // push as a frame, as deep as the interpreter could have called.
        let depth_cap = trace.function.as_ref().map_or(0, |_| {
            MAX_LUA_CALL_FRAMES
                .saturating_sub(depth + trace.max_inline_depth)
                .min(MAX_NATIVE_ACTIVATIONS)
        });
        // The bank holds exactly what this run's lanes name. The pinned
        // closures come first, so lane `k` is pinned closure `k`. One that no
        // longer exists matches nothing the trace was specialised on.
        bank.clear();
        for identity in &*trace.pinned {
            if let Some(index) = pinned_native_index(*identity) {
                bank.push(Value::NativeFunction(NativeFunctionHandle::from_index(
                    index,
                )));
                continue;
            }
            if let Some(builtin) = pinned_builtin_kind(*identity) {
                // The builtins a trace pins: the `ipairs` iterator, and
                // `tostring` and `string.format` for host calls.
                bank.push(Value::BuiltinFunction(match builtin {
                    PINNED_BUILTIN_PCALL => BuiltinFunction::ProtectedCall,
                    PINNED_BUILTIN_XPCALL => BuiltinFunction::ExtendedProtectedCall,
                    PINNED_BUILTIN_DEBUG_TRACEBACK => BuiltinFunction::DebugTraceback,
                    PINNED_BUILTIN_TOSTRING => BuiltinFunction::ToString,
                    PINNED_BUILTIN_STRING_FORMAT => BuiltinFunction::StringFormat,
                    PINNED_BUILTIN_STRING_GSUB => BuiltinFunction::StringGsub,
                    PINNED_BUILTIN_NEXT => BuiltinFunction::Next,
                    PINNED_BUILTIN_PAIRS => BuiltinFunction::Pairs,
                    PINNED_BUILTIN_IPAIRS => BuiltinFunction::IPairs,
                    PINNED_BUILTIN_COROUTINE_RESUME => BuiltinFunction::CoroutineResume,
                    PINNED_BUILTIN_COROUTINE_YIELD => BuiltinFunction::CoroutineYield,
                    _ => {
                        debug_assert_eq!(builtin, PINNED_BUILTIN_IPAIRS_ITERATOR);
                        BuiltinFunction::IPairsIterator
                    }
                }));
                continue;
            }
            let handle = ClosureHandle::from_native_identity(*identity);
            if runtime.closure_call_parts(handle).is_none() {
                bank.clear();
                return Attempt::Mismatch { next: trace.next };
            }
            bank.push(Value::Closure(handle));
        }
        // Names the trace shares one variable for must name one cell now.
        // The cells an entry finds are those the last one found while no
        // closure's cell was replaced and the frame and its captured locals
        // name the same ones, which spares resolving and sorting them all.
        let cells_known = trace
            .cells_memo
            .as_ref()
            .is_some_and(|memo| memo.holds(&frame, &captured, runtime));
        if !cells_known {
            if !aliases_hold(&frame, &captured, runtime, bank, &trace.aliases) {
                bank.clear();
                return Attempt::Mismatch { next: trace.next };
            }
            if trace.has_upvalues
                && (trace.many_upvalues || !captured.is_empty())
                && upvalues_alias(&frame, &captured, runtime, bank, &trace.externals)
            {
                // The trace keeps each cell in one native value. Two upvalues
                // that share a cell, or one that shares a captured local's
                // cell, would need two views of one variable.
                bank.clear();
                return Attempt::Mismatch { next: trace.next };
            }
            if !trace.aliases.is_empty() || trace.has_upvalues {
                trace.cells_memo = Some(CellsMemo::of(&frame, &captured, runtime));
            }
        }
        // A trace that makes tables must find the arena with room for them
        // before any address of a table is taken. `Vec::reserve` does nothing
        // while the spare capacity already covers the headroom, so an entry
        // allocates only when the arena is low, and the headroom is capacity,
        // not a count the collector charges.
        if trace.compiled.makes_tables() {
            runtime.reserve_table_slots(TABLE_ALLOCATION_HEADROOM);
        }
        inputs.clear();
        binding_views.clear();
        // The control register of a `next` loop that keeps its value in a
        // lane (see `TraceIr::boxed_control`): the lane is made after the
        // other inputs, and no other input can share it.
        let mut boxed = None;
        let mut boxed_value = None;
        let control = trace.ir.boxed_control.map(usize::from);
        // Only a `next` loop keeps its control value in a lane of its own.
        let ordinary = control.is_none();
        let mut holders = GlobalHolders::default();
        let register_count = trace.register_count;
        let by_steps = !trace.entry_plan.is_empty()
            && trace.entry_plan.len() == trace.compiled.entry_contract().len();
        if by_steps {
            inputs.reserve(trace.entry_plan.len());
            for step in &*trace.entry_plan {
                let slot = match *step {
                    EntryStep::Register(register) => {
                        input_slot(&frame.registers[usize::from(register)], bank)
                    }
                    EntryStep::External(position) => {
                        match trace.externals.get(usize::from(position)) {
                            Some(external) => external_slot(
                                &frame,
                                runtime,
                                external,
                                bank,
                                (&mut holders, Some(entry_reads)),
                                caller_view.as_ref(),
                            )
                            .unwrap_or_else(TraceInputSlot::other),
                            None => TraceInputSlot::other(),
                        }
                    }
                };
                inputs.push(slot);
            }
        }
        if !by_steps {
            for (register, _) in trace.compiled.entry_contract() {
                let register = usize::from(*register);
                let slot = if let Some(external) =
                    trace.externals.get(register.wrapping_sub(register_count))
                {
                    external_slot(
                        &frame,
                        runtime,
                        external,
                        bank,
                        (&mut holders, Some(entry_reads)),
                        caller_view.as_ref(),
                    )
                    .unwrap_or_else(TraceInputSlot::other)
                } else if ordinary {
                    input_slot(&frame.registers[register], bank)
                } else if control == Some(register) {
                    boxed = Some((inputs.len(), register));
                    TraceInputSlot::other()
                } else if trace.ir.boxed_value
                    && control.map(|control| control + 2) == Some(register)
                {
                    // The second variable of the loop, when the body never reads
                    // it: its lane follows the control lane.
                    boxed_value = Some((inputs.len(), register));
                    TraceInputSlot::other()
                } else {
                    input_slot(&frame.registers[register], bank)
                };
                inputs.push(slot);
            }
        }
        // The registers sides compiled apart read and the entry does not: the
        // side checks their types itself.
        for (source, _) in trace.compiled.extras() {
            let held = match *source {
                crate::jit::ExtraSource::Register(register) => {
                    frame.registers.get(usize::from(register))
                }
                // The caller the tree's sides return into: the entry fits
                // only that caller (`trace.caller`).
                crate::jit::ExtraSource::Caller(register) => caller_view
                    .as_ref()
                    .and_then(|view| view.registers.get(usize::from(register))),
                crate::jit::ExtraSource::External(_) => None,
            };
            let slot = match source {
                crate::jit::ExtraSource::External(external) => external_slot(
                    &frame,
                    runtime,
                    external,
                    bank,
                    (&mut holders, Some(entry_reads)),
                    caller_view.as_ref(),
                )
                .unwrap_or_else(TraceInputSlot::other),
                _ => held.map_or_else(TraceInputSlot::other, |value| extra_slot(value, bank)),
            };
            inputs.push(slot);
        }
        let mut box_lanes = boxed.map(|(position, register)| {
            bank.push(frame.registers[register].clone());
            inputs[position] = TraceInputSlot::string((bank.len() - 1) as u64);
            bank.len() - 1..bank.len()
        });
        if let Some((position, register)) = boxed_value
            && let Some(lanes) = box_lanes.as_mut()
        {
            // The step finds this lane after the control lane.
            bank.push(frame.registers[register].clone());
            inputs[position] = TraceInputSlot::string((bank.len() - 1) as u64);
            lanes.end = bank.len();
        }
        // The lane two after the control lane holds the address of the
        // description of the hash part's snapshot, for the steps native code
        // takes along it ([`BoxedSteps`]); the one between is the value lane,
        // or unused.
        let mut cursor_lane = None;
        if let Some(lanes) = box_lanes.as_mut()
            && trace.ir.boxed_steps.is_some_and(|steps| steps.cursor)
        {
            while bank.len() < lanes.start + 2 {
                bank.push(Value::Nil);
            }
            bank.push(Value::Integer(0));
            cursor_lane = Some(bank.len() - 1);
            lanes.end = bank.len();
        }
        // Every cell or global a self call reads its callee from must hold
        // the frame's closure: the trace calls itself without a guard.
        if let Some(function) = &trace.function {
            let closure = frame.closure;
            let holds = |register: &u16| {
                trace
                    .compiled
                    .entry_contract()
                    .iter()
                    .position(|(held, _)| held == register)
                    .and_then(|position| inputs.get(position))
                    .is_some_and(|slot| {
                        slot.tag() == crate::jit::TRACE_TAG_CLOSURE
                            && usize::try_from(slot.bits())
                                .ok()
                                .and_then(|lane| bank.get(lane))
                                .is_some_and(|value| {
                                    matches!(value, Value::Closure(handle) if Some(*handle) == closure)
                                })
                    })
            };
            if closure.is_none() || !function.callee_registers.iter().all(holds) {
                bank.clear();
                return Attempt::Mismatch { next: trace.next };
            }
        }
        if trace.simple_binds
            && !bind_simple(
                &trace.bindings,
                trace.compiled.bindings(),
                inputs,
                binding_views,
                bank,
                runtime,
                entry_reads,
            )
        {
            // A table lost the storage the trace needs. Nothing executed.
            bank.clear();
            return Attempt::Mismatch { next: trace.next };
        }
        if !trace.simple_binds
            && !trace.bindings.is_empty()
            && !bind_tables(
                &trace.bindings,
                trace.compiled.bindings(),
                inputs,
                binding_views,
                bank,
                runtime,
                trace.compiled.probes_hash().then_some(&mut *probe_scratch),
                entry_reads,
                IndexUse {
                    first_run: trace.entries == 0,
                    holders: &holders,
                },
            )
        {
            // A table lost the storage the trace needs. Nothing executed.
            bank.clear();
            return Attempt::Mismatch { next: trace.next };
        }
        let area_len = match &trace.function {
            Some(function) => function.stride * (depth_cap + 1),
            None => trace.compiled.exit_area_len(),
        };
        if exit_area.len() < area_len {
            exit_area.resize(area_len, 0);
        }
        let mut control = TraceControl::new(fuel);
        // A counted trace with no hook to pay counts against a budget it
        // cannot exhaust.
        control.budget = if counted {
            hook_budget.unwrap_or(UNLIMITED_BUDGET)
        } else {
            0
        };
        control.depth_cap = i64::try_from(depth_cap).unwrap_or(0);
        control.barrier_idle = i64::from(runtime.gc_barrier_idle());
        let mut store_calls = StoreCalls::default();
        let closures = trace.compiled.uses_closure_rows().then(|| {
            Box::new(RunClosures {
                held: if trace.has_upvalues {
                    trace
                        .externals
                        .iter()
                        .filter(|slot| trace.written_cells.contains(slot))
                        .filter_map(|slot| upvalue_cell(&frame, runtime, bank, slot))
                        .collect()
                } else {
                    Vec::new()
                },
                lanes: None,
                table_cells: trace.compiled.uses_closure_table_cells(),
                followers: Vec::new(),
            })
        });
        // A trace that resumes coroutines lends the resumed code the frames it
        // runs on: nothing here touches `frames` until the run is over, and
        // the run keeps no reference to it across the call.
        let frame_base = frame.base;
        let captured_written: Vec<usize> = trace
            .ir
            .written_registers
            .iter()
            .filter(|(written, _)| *written == FrameId(0))
            .map(|(_, register)| usize::from(*register))
            .filter(|register| {
                captured
                    .iter()
                    .any(|open| open.slot.checked_sub(frame_base) == u32::try_from(*register).ok())
            })
            .collect();
        let written = if trace.compiled.resumes() {
            written_externals(&trace.ir)
        } else {
            Vec::new()
        };
        let resume_host = (trace.compiled.resumes() && *resume_vm != 0).then(|| {
            // The upvalues held in cells that are closed now: only a write
            // through the runtime changes one (`Runtime::cell_writes`).
            let (closed_cells, checks): (Vec<_>, Vec<_>) = trace
                .compiled
                .entry_contract()
                .iter()
                .enumerate()
                .filter_map(|(input, (register, _))| {
                    let slot = trace
                        .externals
                        .get(usize::from(*register).wrapping_sub(trace.register_count))?;
                    Some((input, *slot))
                })
                .filter(|(_, slot)| !written.contains(slot))
                .partition(|(_, slot)| {
                    matches!(
                        slot,
                        ExternalSlot::Upvalue(_) | ExternalSlot::ClosureUpvalue { .. }
                    ) && upvalue_cell(&frame, runtime, bank, slot)
                        .is_some_and(|cell| runtime.cell_open_slot(cell).is_none())
                });
            ResumeHost {
                vm: *resume_vm,
                frames: frames_at as usize,
                frame_index,
                restores: (trace.restores.as_ptr(), trace.restores.len()),
                area_len: trace.compiled.exit_area_len(),
                checks,
                closed_cells,
                written,
                captured: trace
                    .compiled
                    .entry_contract()
                    .iter()
                    .enumerate()
                    .filter_map(|(input, (register, _))| {
                        let register = usize::from(*register);
                        captured
                            .iter()
                            .any(|open| {
                                open.slot.checked_sub(frame_base) == u32::try_from(register).ok()
                            })
                            .then_some((input, register))
                    })
                    .filter(|(_, register)| !captured_written.contains(register))
                    .collect(),
                captured_written,
                aliases: trace.aliases.clone(),
                cells_seen: CellsMemo::of(&frame, &captured, runtime),
                upvalue_slots: if trace.has_upvalues {
                    trace
                        .externals
                        .iter()
                        .filter(|slot| {
                            matches!(
                                slot,
                                ExternalSlot::Upvalue(_) | ExternalSlot::ClosureUpvalue { .. }
                            )
                        })
                        .copied()
                        .collect()
                } else {
                    Vec::new()
                },
                many_upvalues: trace.many_upvalues,
            }
        });
        let mut returned_shapes = None;
        let outcome = if trace.compiled.needs_tables() {
            // `RunTables` keeps the runtime and the bank as the pointers the
            // run was lent, not as references: the helper that resumes a
            // coroutine runs the VM, which uses both, and `RunTables` takes
            // a reference from them only where it uses one. The bank is
            // named from `this`, not from the `bank` reference above.
            let function = trace.function.is_some();
            let makes_tables = trace.compiled.makes_tables();
            let shapes = if makes_tables {
                std::mem::take(&mut trace.table_shapes)
            } else {
                Vec::new()
            };
            // From here until the call returns, the run reaches the entry
            // only through `entry_at`, and no `&mut` to it or to the cache
            // is live. It holds shared views of single fields (the
            // resumed code never writes them: native execution is suspended
            // and flushes wait) and the raw words of the field hints, which
            // no view covers. `trace` is made again after the call.
            let entry_at: *mut InstalledTrace = trace;
            // SAFETY: the entry is valid for the run; each place below is a
            // separate field, and only the hints are written through a pointer.
            let (fields, bindings, roots, compiled, hints, meta_field, tags, memo) = unsafe {
                (
                    &(*entry_at).fields,
                    &(*entry_at).bindings,
                    &(*entry_at).roots,
                    &(*entry_at).compiled,
                    {
                        let words = &mut (*entry_at).field_hints;
                        (words.as_mut_ptr(), words.len())
                    },
                    (*entry_at).meta_field,
                    &(*entry_at).field_tags,
                    {
                        let words = &mut (*entry_at).row_memo;
                        (words.as_mut_ptr(), words.len())
                    },
                )
            };
            let mut tables = RunTables::new(
                runtime_at,
                // SAFETY: `this` is valid for the run.
                unsafe { std::ptr::addr_of_mut!((*this).bank) },
                table_lanes,
                lane_index,
                fields,
                hints,
                meta_field,
                box_lanes.as_ref(),
            );
            tables.set_bindings(bindings, inputs);
            tables.set_global_holders(holders);
            tables.set_cursor_lane(cursor_lane);
            tables.set_arrays(compiled.array_flags());
            tables.set_kinds(compiled.bindings());
            tables.set_roots(roots);
            tables.set_row_memo(memo, tags);
            tables.populate_index(makes_tables);
            tables.closures = closures;
            tables.resume = resume_host.map(|host| Box::new(ResumeState::new(Some(host))));
            tables.fresh_lanes = function;
            tables.host.shapes = shapes;
            if !function {
                // SAFETY: as the fields above; nothing replaces the plans
                // while the run holds the entry.
                let plans = unsafe { &(*entry_at).restores };
                tables.set_exit_plans(plans, exit_area.len());
            }
            let outcome = compiled.run_with_tables(
                inputs,
                binding_views,
                &mut control,
                exit_area,
                table_space,
                &mut tables,
            );
            // The lane holds the key the last native step moved to before
            // anything restores a register from it.
            tables.flush_cursor();
            store_calls = tables.store_calls;
            if makes_tables {
                returned_shapes = Some(std::mem::take(&mut tables.host.shapes));
            }
            outcome
        } else {
            trace
                .compiled
                .run_bound(inputs, binding_views, &mut control, exit_area)
        };
        // The native call may have run the interpreter (a resume), which used
        // the VM, the stack and everything else the run does not keep: the
        // references the run made before the call are not used again. Each
        // one is made again from the pointers the run was lent.
        // SAFETY: the call is over, and nothing else uses these.
        let (stats, runtime, frames) =
            unsafe { (&mut *stats_at, &mut *runtime_at, &mut *frames_at) };
        // SAFETY: as above, and the fields are distinct.
        let (cache, exit_area, bank, side_candidate, entered_recently, restore_error, hook_budget) = unsafe {
            (
                &mut (*this).cache,
                &mut (*this).exit_area,
                &mut (*this).bank,
                &mut (*this).side_candidate,
                &mut (*this).entered_recently,
                &mut (*this).restore_error,
                &mut (*this).hook_budget,
            )
        };
        // The entry is borrowed again from the cache just re-borrowed: no
        // reference to it was kept across the call.
        let Some(trace) = cache.get_mut(index) else {
            unreachable!("a run keeps its trace's cache entry: flushes wait for it")
        };
        if let Some(shapes) = returned_shapes {
            trace.table_shapes = shapes;
        }
        if let TraceOutcome::EntryMismatch { .. } | TraceOutcome::StoreEntryMissing = outcome {
            if let TraceOutcome::EntryMismatch { input } = outcome {
                note_entry_flips(trace, inputs, input);
                if trace.frame_checks.is_none() {
                    trace.frame_checks =
                        frame_checks_of(&trace.ir, &trace.compiled, trace.register_count);
                }
            }
            // Nothing executed. Another version may take the frame.
            bank.clear();
            return Attempt::Mismatch { next: trace.next };
        }
        // From here to the return, the work rebuilds interpreter state.
        let _phase = alloc_phase::enter(Phase::ExitRecovery);
        // Links of sides compiled apart rewrote registers of the frames that
        // the exits of the tree know nothing about: they go into the frames
        // before anything is restored over them.
        if let Some(words) = trace.compiled.writeback(exit_area) {
            write_back_links(
                frames,
                frame_index,
                bank,
                trace.compiled.writebacks(),
                words,
                runtime,
            );
        }
        let frame = frames.at_mut(frame_index);
        stats.native_entries += 1;
        stats.hash_helper_calls += u64::try_from(control.hash_calls).unwrap_or(0);
        stats.set_string_helper_calls += store_calls.set_string;
        if trace.overwrite_version == OverwriteVersion::Watching
            && store_calls.set_string >= TRACE_OVERWRITE_VERSION_CALLS
            && store_calls.set_string_overwrites == store_calls.set_string
        {
            trace.overwrite_version = OverwriteVersion::Wanted;
        } else if trace.overwrite_version == OverwriteVersion::Watching
            && trace.overwrite_polling != 0
            && store_calls.set_string > 0
        {
            // Only a real poll (an entry cut at the watch fuel) advances the
            // back-off, and the first few keep polling back to back: a
            // table's fill is one poll, and the overwrites that follow it in
            // the same loop must be seen within a few thousand iterations.
            trace.overwrite_polls = trace.overwrite_polls.saturating_add(1);
            trace.overwrite_rest = match trace
                .overwrite_polls
                .checked_sub(TRACE_OVERWRITE_FREE_POLLS)
            {
                Some(shift) => (1u32 << shift.min(TRACE_OVERWRITE_MAX_REST_SHIFT)) - 1,
                None => 0,
            };
        }
        stats.set_present_helper_calls += store_calls.set_present;
        stats.set_revive_helper_calls += store_calls.set_revive;
        stats.delete_string_helper_calls += store_calls.delete_string;
        stats.array_set_helper_calls += store_calls.set_array;
        stats.table_new_helper_calls += store_calls.table_new;
        stats.closure_helper_calls += store_calls.closure;
        stats.table_lookup_helper_calls += store_calls.table_lookup;
        stats.side_entries += u64::try_from(control.links).unwrap_or(0);
        stats.native_side_calls += u64::try_from(control.native_calls).unwrap_or(0);
        // The constructors the run elided are charged first, so that a table
        // this exit builds for real can take its own charge back.
        if control.elided_tables > 0 {
            runtime.charge_elided_tables(
                u64::try_from(control.elided_tables).unwrap_or(0),
                usize::try_from(control.elided_bytes).unwrap_or(0),
            );
        }
        // So do the fields it stored into tables it made.
        if control.initialized > 0 {
            runtime.charge_native_inits(usize::try_from(control.initialized).unwrap_or(0));
        }
        // The slots the run appended in place cost the collector what the
        // interpreter's pushes would have.
        if control.appended > 0 {
            runtime.charge_native_appends(usize::try_from(control.appended).unwrap_or(0));
        }
        let iterations = u64::try_from(fuel - control.fuel).unwrap_or(0);
        stats.iterations += iterations;
        // A function trace's entry that ran one activation and returned
        // counts like any other: a self call at the leaves costs an entry
        // and a return for one activation's work, more than the interpreter
        // spends on it (see `TRACE_MIN_HALF_ACTIVATIONS_PER_ENTRY`).
        // An exit of a function trace unwinds the activations below the one
        // it left from: the interpreter redoes their calls, so the self
        // calls that led there did no work the productivity bar may credit.
        // A run that went as deep as the trace may go did the work, and so
        // did one whose exit a side trace can still take over.
        let unwound = match outcome {
            TraceOutcome::Exit { index }
                if trace.function.is_some()
                    && trace.compiled.exits()[index].guard != Some(TraceGuardClass::Depth)
                    && !trace.side_possible(index) =>
            {
                u64::try_from(control.depth).unwrap_or(0)
            }
            _ => 0,
        };
        let credited = iterations.saturating_sub(unwound).max(1);
        trace.iterations += credited;
        // A run the count hook's budget cut short says nothing about the
        // loop: it neither counts as an entry nor as an unproductive one.
        let cut_short = budget_limited
            && matches!(
                outcome,
                TraceOutcome::Exit { index }
                    if trace.compiled.exits()[index].class == TraceExitClass::Poll
            );
        // So does a side compiled apart that leaves at the loop header where
        // it would link back: the run goes on from there, and the cost of the
        // exit is counted against the side (see `InstalledTrace::wants_relink`).
        let cut_short = cut_short
            || matches!(
                outcome,
                TraceOutcome::Exit { index }
                    if {
                        let descriptor = &trace.compiled.exits()[index];
                        descriptor.class == TraceExitClass::Poll
                            && descriptor.member > 0
                            && trace.leaves_at_header(descriptor.member)
                    }
            );
        let mut retire = if cut_short {
            false
        } else {
            trace.entries += 1;
            // A run that left inside an inlined call before its first
            // loop-back did part of one iteration: the interpreter rebuilds
            // the calls' frames and finishes them. The heavy bar does not
            // credit it.
            let left_inside_calls = matches!(
                outcome,
                TraceOutcome::Exit { index }
                    if iterations == 0 && !trace.restores[index].inner.is_empty()
            );
            trace.note_entry(credited, if left_inside_calls { 0 } else { credited })
        };
        // A trace that exits at once because a branch alternates is not
        // unproductive for good: a side trace is on its way for that guard.
        if retire && trace.forgive_for_side() {
            retire = false;
        }
        // A tree compiling again with its guards in the loop is kept until
        // that code can take sides there.
        if retire && trace.guard_move_grace > 0 {
            trace.guard_move_grace -= 1;
            retire = false;
        }
        // A short loop in a nest is kept while the tree that carries it
        // through the loops around it grows.
        if retire && trace.keep_stone() {
            retire = false;
        }
        if retire && debug_env::trace() {
            eprintln!(
                "jit-trace: the trace at pc {} retires below the productivity bar: grace {}, a side wanted {}, busy {}, {} live sides of {} at most",
                trace.header_pc,
                trace.side.grace,
                trace.wants_side(true),
                trace.side.busy(),
                trace.live_sides(),
                trace.side.max_sides,
            );
        }
        if trace.function.is_some() {
            // The activations past the entered one were native self calls.
            stats.native_self_calls += iterations.saturating_sub(1);
        }
        // The entered activation returned: the frame is untouched, and the
        // caller makes its return.
        let TraceOutcome::Exit { index: exit } = outcome else {
            let bits = exit_area.first().copied().unwrap_or(0);
            let value = match trace.function.as_ref().and_then(|function| function.result) {
                Some(TraceType::Float) => Value::Number(f64::from_bits(bits.cast_unsigned())),
                Some(TraceType::Integer) => Value::Integer(bits),
                Some(TraceType::Boolean) => Value::Boolean(bits != 0),
                // A table is a lane of the run's bank, which holds it until
                // the frame does.
                Some(TraceType::Table) => {
                    let held = usize::try_from(bits).ok().and_then(|lane| bank.get(lane));
                    debug_assert!(matches!(held, Some(Value::Table(_))), "a table result");
                    held.cloned().unwrap_or(Value::Nil)
                }
                _ => Value::Nil,
            };
            stats.native_returns += 1;
            bank.clear();
            *side_candidate = None;
            *entered_recently = true;
            cache.touch(index);
            return Attempt::Done {
                run: TraceRun::Returned { value },
                retire,
            };
        };
        // A function trace left from activation `deepest`, whose state is
        // in its own part of the exit area. The activations below it
        // unwound: each left through the exit its part names.
        let deepest = usize::try_from(control.depth).unwrap_or(0);
        let stride = trace
            .function
            .as_ref()
            .map_or(0, |function| function.stride);
        let area_of = |activation: usize| -> &[i64] {
            if stride == 0 {
                exit_area
            } else {
                &exit_area[activation * stride..(activation + 1) * stride]
            }
        };

        let descriptor = &trace.compiled.exits()[exit];
        // The iteration that left began with `control.budget`, and ran the
        // exit's counted instructions.
        if counted && let Some(budget) = hook_budget.as_mut() {
            *budget = control.budget - i64::from(descriptor.counted);
        }
        let callee_miss = descriptor.class == TraceExitClass::GuardFail
            && descriptor.guard == Some(TraceGuardClass::Callee)
            && iterations < trace.min_iterations;
        if callee_miss {
            trace.callee_streak = trace.callee_streak.saturating_add(1);
            // A trace pinned to a callee the site no longer calls retires,
            // unless a side trace for the miss can still be recorded: the
            // misses count toward the side threshold below, and the side
            // pins the callee that took the site over.
            let streak =
                trace.callee_streak >= TRACE_CALLEE_STREAK_LIMIT && !trace.side_possible(exit);
            if streak && !retire && debug_env::trace() {
                eprintln!(
                    "jit-trace: the trace at pc {} retires: {} entries in a row left at callee guards no side can continue (exit {exit}, side key {:?}, attempts {:?}, {} live sides, {} in progress, {} at most)",
                    trace.header_pc,
                    trace.callee_streak,
                    trace.side_keys[exit],
                    trace.side_keys[exit].map(|key| trace.side.attempts(key)),
                    trace.live_sides(),
                    trace.side.in_progress(),
                    trace.side.max_sides,
                );
            }
            retire |= streak;
        } else {
            trace.callee_streak = 0;
        }
        let restore = &trace.restores[exit];
        // A run with no lane left for a table it reads leaves through the
        // read's guard for want of room, not because the table went another
        // way: the side would record the path the trace has.
        let lanes_full = bank.len()
            >= if trace.function.is_some() {
                MAX_FUNCTION_TABLE_LANES
            } else {
                MAX_TRACE_TABLE_LANES
            };
        // An exit inside inlined calls asks for a side only while the tree's
        // budget for such sides lasts; past it the interpreter runs the rest.
        let seeds_left = restore.inner.is_empty() || trace.seeds_left();
        // A host call the trace made that left with the call complete stands
        // after the call, not where the exit's side would start.
        // SAFETY: as above; the field is distinct from those borrowed.
        let landed_now = unsafe { (*this).vmcall_landed.is_some() };
        let key = trace.side_keys[exit].filter(|_| !lanes_full && seeds_left && !landed_now);
        if key.is_none()
            && let Some(hits) = trace.cold_hits.get_mut(exit)
        {
            *hits = hits.saturating_add(1);
        }
        let cold_exit = key.is_none();
        if let Some(key) = key {
            // Count how often this branch guard fails. At the threshold the
            // policy records the other path as a side trace, starting from the
            // state this exit is about to restore.
            let hits = &mut trace.exit_hits[exit];
            *hits = hits.saturating_add(1);
            trace.exit_last_hit[exit] = trace.iterations;
            // Every threshold's worth of failures asks again, until the
            // guard's attempts are spent. `OXIMOND_JIT_SIDE_PHASE` moves
            // which failure that is.
            if hits
                .wrapping_add(debug_env::side_phase())
                .is_multiple_of(SIDE_EXIT_THRESHOLD)
            {
                // A side for an exit inside inlined calls starts in the
                // innermost frame the exit pushes.
                let inner = restore.inner.len();
                *side_candidate = Some(SideCandidate {
                    index,
                    serial: trace.serial,
                    key,
                    pc: if inner > 0 {
                        restore.resume_pc
                    } else {
                        restore.pc
                    },
                    last: if inner > 0 {
                        restore.resume_last
                    } else {
                        restore.last_interpreted_pc
                    },
                    activation: deepest,
                    inner,
                });
            }
        }
        if trace.virtual_refused[exit] {
            let hits = &mut trace.exit_hits[exit];
            *hits = hits.saturating_add(1);
            if *hits == SIDE_EXIT_THRESHOLD {
                stats.side_rejections += 1;
            }
        }
        // The virtual tables come first, all of them built before any
        // register is written. A failure leaves the frame as the entry state
        // and ends the run with the runtime's error.
        let own_area = area_of(deepest);
        let tail_site = trace
            .function
            .as_ref()
            .and_then(|function| function.tail_site.as_ref());
        // A host call the trace made itself is complete and the frame
        // stands after it: nothing to restore.
        // SAFETY: as above; the field is distinct from those borrowed.
        let landed = unsafe { (*this).vmcall_landed.take() };
        // SAFETY: as above.
        if landed.is_some() && unsafe { (*this).restore_error.is_some() } {
            // The call raised an error: the run ends, and the entry raises
            // it from the frame as the call left it.
            bank.clear();
            *side_candidate = None;
            return Attempt::Done {
                run: TraceRun::Failed,
                retire: false,
            };
        }
        let materialized = if landed.is_some() {
            Ok(Vec::new())
        } else {
            restore.materialize(own_area, bank, &trace.roots, runtime)
        };
        let tables = match materialized {
            Ok(tables) => tables,
            Err(error) => {
                bank.clear();
                *side_candidate = None;
                *restore_error = Some(error);
                return Attempt::Done {
                    run: TraceRun::Failed,
                    retire: false,
                };
            }
        };
        if landed.is_some() {
            // A call from inside inlined frames left them on the stack.
            debug_assert_eq!(deepest, 0);
        } else if deepest > 0 {
            // A function trace left from a nested activation. Each activation
            // below it stands at the self call that made the next: the
            // entered one in the frame the interpreter pushed, the others in
            // frames of the same closure pushed one above the other, as the
            // interpreter's calls would have left them.
            let closure = frame
                .closure
                .expect("an entered function trace has a closure");
            let mut call = None;
            for activation in 0..deepest {
                let area = area_of(activation);
                let unwind = usize::try_from(area[stride - 1]).unwrap_or(usize::MAX);
                let plan = &trace.restores[unwind];
                debug_assert!(plan.call.is_some(), "an unwind exit names its call");
                let at = match call {
                    None => frame_index,
                    Some(call) => push_inlined_frame(frames, runtime, closure, call)
                        .expect("an entered function's closure exists"),
                };
                plan.apply(
                    &mut frames.at_mut(at),
                    area,
                    bank,
                    &trace.roots,
                    &[],
                    runtime,
                );
                plan.apply_function_count(&mut frames.at_mut(at), area, tail_site);
                call = plan.call;
            }
            let at = push_inlined_frame(
                frames,
                runtime,
                closure,
                call.expect("an unwind exit names its call"),
            )
            .expect("an entered function's closure exists");
            restore.apply(
                &mut frames.at_mut(at),
                own_area,
                bank,
                &trace.roots,
                &tables,
                runtime,
            );
            restore.apply_function_count(&mut frames.at_mut(at), own_area, tail_site);
            stats.unwound_activations += deepest as u64;
        } else if restore.has_root_tail() {
            // Tail calls of the root frame may have replaced its closure: the
            // cells the plan writes are the entry closure's, and the
            // registers are the activation's.
            restore.apply_cells(&frame, own_area, bank, &trace.roots, &tables, runtime);
            restore.apply_root_tail(frames, frame_index, own_area, bank, runtime);
            let mut frame = frames.at_mut(frame_index);
            restore.apply_registers(&mut frame, own_area, bank, &trace.roots, &tables);
        } else {
            if restore.writes_caller() {
                restore.apply_callers(frames, frame_index, own_area, bank, &trace.roots, &tables);
            }
            let mut frame = frames.at_mut(frame_index);
            restore.apply(&mut frame, own_area, bank, &trace.roots, &tables, runtime);
            restore.apply_function_count(&mut frame, own_area, tail_site);
        }
        // Inside an inlined call, the callee frames go on top of the root.
        if landed.is_none() {
            restore.push_inner(frames, own_area, bank, &trace.roots, &tables, runtime);
        }
        if !restore.inner.is_empty() {
            stats.inner_exits += 1;
        }
        // Everything the lanes named is back in registers and cells.
        bank.clear();
        stats.exits[exit_kind(descriptor.class).index()] += 1;
        let column = descriptor
            .guard
            .map_or(TraceGuardKind::ALL.len(), |guard| guard_kind(guard).index());
        stats.interpreter_exits[exit_kind(descriptor.class).index()][column] += 1;
        trace.interp_exits[exit_kind(descriptor.class).index()][column] += 1;
        if descriptor.member > 0 {
            stats.side_member_exits += 1;
        }
        // What sides compiled apart cost: calls the run made, and exits that
        // leave at the loop header where an inline side would link back.
        trace.child_calls += u64::try_from(control.child_calls).unwrap_or(0);
        trace.native_calls += u64::try_from(control.native_calls).unwrap_or(0);
        if descriptor.class == TraceExitClass::Poll
            && descriptor.member > 0
            && trace.leaves_at_header(descriptor.member)
        {
            trace.link_exits += 1;
        }
        if matches!(
            descriptor.class,
            TraceExitClass::GuardFail | TraceExitClass::CommittedEffect
        ) && let Some(guard) = descriptor.guard
        {
            stats.side_exits[guard_kind(guard).index()] += 1;
        }
        *entered_recently = true;
        let poll = descriptor.class == TraceExitClass::Poll;
        let run = match landed {
            Some((resume, call)) => TraceRun::Ran {
                poll: false,
                resume_pc: resume,
                last_interpreted_pc: Some(call),
            },
            None => TraceRun::Ran {
                poll,
                resume_pc: restore.resume_pc,
                last_interpreted_pc: restore.resume_last,
            },
        };
        if cold_exit && !retire {
            trace.keep_hoisted_guards();
        }
        // Order entries by use for eviction.
        cache.touch(index);
        Attempt::Done { run, retire }
    }

    /// Removes the trace at `index`, whose tree was about to take another
    /// side for another closure at the call site `path`, and names the site
    /// in its loop's [`Site::poly_calls`](super::policy::Site): the loop
    /// records again once hot, with the call guarded on the prototype. It
    /// spends no recording attempt, as an eviction does not: a site is named
    /// once, and a loop names at most [`super::policy::MAX_POLY_CALLS`].
    fn rerecord_for_poly_call(
        &mut self,
        stats: &mut TraceStatistics,
        index: u32,
        path: PolyCallPath,
    ) {
        // What a retirement learns from the tree's exits, the replaced tree
        // teaches too: the next recording keeps its hoisted guards.
        let keep_guards = self
            .cache
            .get(index)
            .is_some_and(InstalledTrace::leaves_at_hoisted_guards);
        let record = self
            .cache
            .get(index)
            .filter(|_| debug_env::trace())
            .map(|trace| {
                format!(
                    "{} iterations in {} entries, {} sides",
                    trace.iterations,
                    trace.entries,
                    trace.live_sides(),
                )
            });
        let Some(removed) = self.remove_version(index) else {
            return;
        };
        let Some(chunk) = removed.chunk else {
            return;
        };
        self.uncover_loops(stats, &chunk, &removed.covered);
        let Some(site) = self.sites.existing_mut(&chunk, removed.header_pc) else {
            return;
        };
        if debug_env::trace() {
            eprintln!(
                "jit-trace: the tree at pc {}{} would take another side for another callee at {path:?}; it records again ({})",
                removed.header_pc,
                debug_env::at_line(&chunk, removed.header_pc),
                record.unwrap_or_default(),
            );
        }
        site.note_poly_calls(vec![path], removed.header_pc);
        site.keep_guards |= keep_guards;
        if !removed.versions_remain && site.state == (SiteState::Installed { index }) {
            site.state = SiteState::Cold { backedges: 0 };
        }
    }

    /// Removes the trace at `index` and backs its site off: a cooldown that
    /// grows with each retirement, then rejection. A header with other
    /// versions keeps them and its state, and records no further versions,
    /// so a productive version never pays for an unproductive sibling.
    /// Only the last version's retirement counts toward the budget. Past it
    /// the header is rejected.
    fn retire(&mut self, stats: &mut TraceStatistics, index: u32) {
        // A trace that never passed a productivity window did not pay for its
        // compile. The site keeps its recording, so that a recording of the
        // same trace (the loop takes the same path again) does not compile it
        // again.
        let never_paid = self
            .cache
            .get(index)
            .filter(|trace| trace.productive_windows == 0)
            .map(|trace| {
                // A stepping stone set `keep_branch_guards` on its IR after
                // it was recorded; the recording to compare with has not.
                if trace.ir.keep_branch_guards {
                    let mut ir = (*trace.ir).clone();
                    ir.keep_branch_guards = false;
                    Arc::new(ir)
                } else {
                    Arc::clone(&trace.ir)
                }
            });
        // Whether the loop averaged fewer trips per entry than its trace
        // needs: a loop that is short, not one retired for another reason
        // (a mismatching entry, a callee that changed).
        let short = self.cache.get(index).is_some_and(|trace| {
            trace.function.is_none()
                && trace.entries > 0
                && trace.iterations < trace.entries * trace.min_iterations
        });
        // A stepping stone retires only once it gave up on its outer sides.
        // One whose entries mostly found types no version takes gives up
        // for good: the next root meets the same types.
        let stone_give_ups = self.cache.get(index).map_or(0, |trace| {
            match (trace.stone.is_some(), trace.mostly_mismatched()) {
                (false, _) => 0,
                (true, false) => 1,
                (true, true) => STONE_GIVE_UPS,
            }
        });
        // The nil-or-number registers the trace typed at its entry.
        let typed_entries: Vec<u16> = self.cache.get(index).map_or_else(Vec::new, |trace| {
            trace
                .compiled
                .entry_contract()
                .iter()
                .filter(|(_, ty)| {
                    matches!(ty, TraceType::Nil | TraceType::Float | TraceType::Integer)
                })
                .map(|(register, _)| *register)
                .collect()
        });
        // The call sites whose callee guards the trace kept leaving at: the
        // next recording guards them on the prototype.
        let poly_calls = self
            .cache
            .get(index)
            .map(InstalledTrace::callee_exit_paths)
            .unwrap_or_default();
        // Likewise the branch guards hoisted in front of the loop, and the
        // guard of a callee the loop takes as it was entered (`op` of
        // Mineclonia's `region_evaluate`, one closure per caller): in front
        // of the loop it leaves with no side to continue it, and in the
        // loop a side takes the other callee.
        let keep_guards = self.cache.get(index).is_some_and(|trace| {
            trace.leaves_at_hoisted_guards() || trace.callee_streak >= TRACE_CALLEE_STREAK_LIMIT
        });
        // What the trace ran, for the report below.
        // A trace that made host calls and did not pay (the calls changed
        // what it read, or gave other results): the next recording aborts at
        // the call.
        let made_vm_calls = self
            .cache
            .get(index)
            .is_some_and(|trace| trace.ir.makes_vm_calls());
        let record = self
            .cache
            .get(index)
            .filter(|_| debug_env::trace())
            .map(|trace| {
                format!(
                    "; {} iterations in {} entries, {} mismatched, bar {}, {} sides, stone: {}{}",
                    trace.iterations,
                    trace.entries,
                    trace.mismatches,
                    trace.min_iterations,
                    trace.live_sides(),
                    trace.stone.is_some(),
                    trace.hottest_exits()
                )
            });
        let Some(removed) = self.remove_version(index) else {
            return;
        };
        let Some(chunk) = removed.chunk else {
            return;
        };
        self.uncover_loops(stats, &chunk, &removed.covered);
        let Some(site) = self.sites.existing_mut(&chunk, removed.header_pc) else {
            return;
        };
        // A stone that typed a register the site lists as nil on some
        // entries and a number on others (it was recorded before the list
        // named it, and outlived the versions that took it untyped) met
        // types the next root takes: that root is judged as a stone afresh,
        // once per site.
        let stone_give_ups = if stone_give_ups >= STONE_GIVE_UPS
            && !site.stone_forgiven
            && typed_entries
                .iter()
                .any(|register| site.lazy.contains(register))
        {
            site.stone_forgiven = true;
            0
        } else {
            stone_give_ups
        };
        site.stone_give_ups = site.stone_give_ups.saturating_add(stone_give_ups);
        site.no_vmcall |= made_vm_calls;
        if keep_guards && !site.keep_guards {
            if debug_env::trace() {
                eprintln!(
                    "jit-trace: the loop at pc {} left at hoisted guards; the next recording keeps them in the loop",
                    removed.header_pc
                );
            }
            site.keep_guards = true;
        }
        site.note_poly_calls(poly_calls, removed.header_pc);
        let owned = removed.versions_remain || site.state == (SiteState::Installed { index });
        if !owned {
            return;
        }
        // A loop that ran too few trips per entry is short: the loops around
        // it that were turned away for meeting it as an inner loop record
        // again, and unroll it.
        let loops = nest::loops_of(&chunk);
        let is_loop = short && loops.iter().any(|span| span.header == removed.header_pc);
        if is_loop && !removed.versions_remain {
            // Once is a loop that started slowly (the first pixels of a
            // kernel); a root recorded again that retires again is short.
            let again = self
                .sites
                .existing_mut(&chunk, removed.header_pc)
                .is_some_and(|site| {
                    site.short_retirements = site.short_retirements.saturating_add(1);
                    site.retired_short = site.short_retirements >= 2;
                    site.retired_short
                });
            if again {
                self.reopen_outer_loops(&chunk, removed.header_pc);
            }
        }
        let Some(site) = self.sites.existing_mut(&chunk, removed.header_pc) else {
            return;
        };
        match never_paid {
            Some(ir) => {
                if site.retired_unproductive.len() >= RETIRED_UNPRODUCTIVE_KEPT {
                    site.retired_unproductive.remove(0);
                }
                site.retired_unproductive.push(ir);
            }
            None => site.retired_unproductive.clear(),
        }
        if debug_env::trace() {
            eprintln!(
                "jit-trace: retired the trace at pc {}{} (attempt {}{})",
                removed.header_pc,
                debug_env::at_line(&chunk, removed.header_pc),
                site.attempts,
                record.unwrap_or_default()
            );
        }
        if removed.versions_remain {
            // A sibling signature proved unproductive. Drop just that
            // version and stop recording more, so the budget cannot throw
            // out a productive version next to it.
            site.versions_closed = true;
            site.rerecord = false;
            return;
        }
        // The last version is gone, and with it the sibling that closing
        // the header protected. The next attempt starts from one version
        // again and may ask for others: a header whose entries alternate
        // between two signatures (Mineclonia's `fill_box_rotated`, called
        // with an rng table by corridors and nil by staircases) otherwise
        // keeps one version per attempt, which the other signature's
        // entries retire by idle mismatches, until the attempts run out.
        site.versions_closed = false;
        site.attempts = site.attempts.saturating_add(1);
        if site.attempts >= MAX_RECORDING_ATTEMPTS {
            stats.negative_evictions +=
                self.sites
                    .reject(&chunk, removed.header_pc, RejectCause::Retired);
        } else {
            let wait = RETRY_COOLDOWN_BASE << (2 * u32::from(site.attempts.min(3)));
            site.state = SiteState::Cooldown { remaining: wait };
        }
    }
}

/// The budget a counted trace starts with when no hook counts for it.
const UNLIMITED_BUDGET: i64 = 1 << 62;

/// The cell a name refers to in this run, if it exists. Pinned closures are
/// the first lanes of `bank`.
fn cell_of(
    frame: &FrameMut<'_>,
    captured: &[crate::vm::stack::OpenUpvalue],
    runtime: &Runtime,
    bank: &[Value],
    source: CellSource,
) -> Option<CellHandle> {
    match source {
        CellSource::Upvalue(index) => frame.upvalues.get(usize::from(index)).copied(),
        CellSource::ClosureUpvalue { closure, index } => {
            let handle = pinned_closure(bank, closure)?;
            runtime
                .closure_cell(handle, usize::from(index))
                .ok()
                .flatten()
        }
        CellSource::Captured(register) => {
            let slot = frame.base + u32::from(u8::try_from(register).ok()?);
            captured
                .iter()
                .find(|open| open.slot == slot)
                .map(|open| open.cell)
        }
    }
}

/// The cell slots of the window that a member of the tree `ir` writes: its
/// root and every side. A cell the window only reads holds the value it held
/// at entry for the whole run, so a closure called through the prototype
/// guard that shares it reads the same value through its own row
/// (Mineclonia's `chunk_test` closures share `floor` and `set_region_seed`
/// with the `chunk_test` the root pinned). A member that writes a callee's
/// cell through the live closure ([`HelperId::CELL_SET`]) may write any of
/// them, so then every cell slot counts.
fn written_cell_slots(ir: &TraceIr) -> Box<[ExternalSlot]> {
    let members = std::iter::once(ir).chain(ir.sides.iter().map(|side| &*side.ir));
    let mut slots = Vec::new();
    for member in members {
        let writes_cells = member.preheader.iter().chain(&member.body).any(|instruction| {
            matches!(&instruction.op, TraceOp::CallHelper { helper, .. } if *helper == HelperId::CELL_SET)
        });
        let real =
            usize::from(member.frames[0].register_count).saturating_sub(member.external.len());
        for (position, slot) in member.external.iter().enumerate() {
            let written = writes_cells
                || member.written_registers.iter().any(|&(frame, register)| {
                    frame == FrameId(0) && usize::from(register) == real + position
                });
            if written && !slots.contains(slot) {
                slots.push(*slot);
            }
        }
    }
    slots.into_boxed_slice()
}

/// The promises of `ir` that two names share a cell, which an entry checks
/// with [`aliases_hold`]. A tree takes them from its IR each time a side
/// joins, since a side may have found a callee cell the root's captured local
/// stands for.
fn trace_aliases(ir: &TraceIr) -> Box<[CellAlias]> {
    ir.cell_aliases.clone().into_boxed_slice()
}

/// Whether every promise of the trace that two names share a cell holds for
/// this frame.
fn aliases_hold(
    frame: &FrameMut<'_>,
    captured: &[crate::vm::stack::OpenUpvalue],
    runtime: &Runtime,
    bank: &[Value],
    aliases: &[CellAlias],
) -> bool {
    aliases.iter().all(|alias| {
        let cell = cell_of(frame, captured, runtime, bank, alias.cell);
        cell.is_some() && cell == cell_of(frame, captured, runtime, bank, alias.is)
    })
}

/// The cell an upvalue slot of the trace names for this run, or `None` for a
/// slot that is not an upvalue or whose cell is missing. Pinned closures
/// are the first lanes of `bank`.
fn upvalue_cell(
    frame: &FrameMut<'_>,
    runtime: &Runtime,
    bank: &[Value],
    slot: &ExternalSlot,
) -> Option<CellHandle> {
    match slot {
        ExternalSlot::Upvalue(index) => frame.upvalues.get(usize::from(*index)).copied(),
        ExternalSlot::ClosureUpvalue { closure, index } => {
            let handle = pinned_closure(bank, *closure)?;
            runtime
                .closure_cell(handle, usize::from(*index))
                .ok()
                .flatten()
        }
        ExternalSlot::Global(_)
        | ExternalSlot::ClosureGlobal { .. }
        | ExternalSlot::CallerGlobal(_)
        | ExternalSlot::UpvalueField { .. }
        | ExternalSlot::ClosureUpvalueField { .. }
        | ExternalSlot::CallerRegister(_)
        | ExternalSlot::TailCalls
        | ExternalSlot::RootClosure
        | ExternalSlot::VarargCount
        | ExternalSlot::Vararg(_)
        | ExternalSlot::Environment => None,
    }
}

/// The tag the entry check reads for `register` of `frame`, `None` past its
/// end.
#[inline]
fn frame_tag(frame: &FrameMut<'_>, register: usize) -> Option<i64> {
    // The kinds a frame register holds at nearly every entry are answered
    // here: `entry_of` builds a slot for each value and reaches an object's
    // tag through a function pointer, which cost about fifty instructions a
    // register on every entry of a trace with a wide contract.
    Some(match frame.registers.get(register)? {
        Value::Nil => crate::jit::TRACE_TAG_NIL,
        Value::Boolean(_) => crate::jit::TRACE_TAG_BOOLEAN,
        Value::Integer(_) => crate::jit::TRACE_TAG_INTEGER,
        Value::Number(_) => crate::jit::TRACE_TAG_FLOAT,
        Value::Table(_) => crate::jit::TRACE_TAG_TABLE,
        Value::String(_) => crate::jit::TRACE_TAG_STRING,
        Value::Closure(_) => crate::jit::TRACE_TAG_CLOSURE,
        other => match entry_of(other) {
            Entry::Words(slot) => slot.tag(),
            Entry::Lane(slot) => slot(0).tag(),
        },
    })
}

/// The checks [`frame_mismatch`] makes for `compiled`: each frame register
/// its entry contract types, with the tag wanted. Only the frame's registers
/// are listed; the window beyond them (cells, globals, the caller) and the
/// control lane of a `next` loop are left to the code's own check. `None`
/// when nothing is left to check.
fn frame_checks_of(
    ir: &TraceIr,
    compiled: &CompiledTrace,
    register_count: usize,
) -> Option<Box<[(u16, i64)]>> {
    if ir.boxed_control.is_some() {
        return None;
    }
    let checks: Box<[(u16, i64)]> = compiled
        .entry_contract()
        .iter()
        .filter(|(register, _)| usize::from(*register) < register_count)
        .filter_map(|(register, ty)| Some((*register, crate::jit::entry_tag(*ty)?)))
        .collect();
    (!checks.is_empty()).then_some(checks)
}

/// Whether a register of `frame` turns the entry down: one whose tag is not
/// the one [`InstalledTrace::frame_checks`] wants. The position of the first
/// such register in the entry contract, for [`note_entry_flips`].
fn frame_mismatch(trace: &InstalledTrace, frame: &FrameMut<'_>) -> Option<usize> {
    let checks = trace.frame_checks.as_deref()?;
    let (register, _) = checks
        .iter()
        .find(|(register, wanted)| frame_tag(frame, usize::from(*register)) != Some(*wanted))?;
    trace
        .compiled
        .entry_contract()
        .iter()
        .position(|(input, _)| input == register)
}

/// Fills `inputs` for [`note_entry_flips`] after [`frame_mismatch`] refused
/// an entry: a slot per contract input with the tag `frame` holds, or the
/// tag wanted where the check looked at nothing.
fn frame_tags(trace: &InstalledTrace, frame: &FrameMut<'_>, inputs: &mut Vec<TraceInputSlot>) {
    let register_count = trace.register_count;
    let contract = trace.compiled.entry_contract();
    inputs.clear();
    inputs.extend(contract.iter().map(|(register, ty)| {
        let wanted = crate::jit::entry_tag(*ty).unwrap_or(crate::jit::TRACE_TAG_OTHER);
        let register = usize::from(*register);
        let tag = if register < register_count {
            frame_tag(frame, register).unwrap_or(crate::jit::TRACE_TAG_OTHER)
        } else {
            wanted
        };
        TraceInputSlot::from_words(tag, 0)
    }));
}

/// Whether `frame` stands at the header the way the trace's loop reaches it.
/// A loop through tail calls (see [`TraceIr::loop_tail`]) is entered where a
/// tail call landed: no previous pc (which its first snapshots restore), and
/// none of the interior state an in-place tail call refuses or clears
/// (captured locals, to-be-closed slots, suppressed hooks), which its exits
/// would otherwise lose. Any frame at the header of a loop that closes by a
/// branch will do.
fn entered_as_recorded(
    frame: &FrameMut<'_>,
    captured: &[crate::vm::stack::OpenUpvalue],
    ir: &TraceIr,
) -> bool {
    // A function trace is entered where a call pushed the frame: nothing ran
    // in it, it has no interior state, and its caller takes its results
    // directly, as the return the VM makes for it expects.
    if ir.function.is_some() {
        return frame.last_interpreted_pc.is_none()
            && matches!(frame.caller, Some(crate::vm::ReturnTarget::Direct(_)))
            && captured.is_empty()
            && frame.extras_ref().is_none_or(|extras| {
                extras.close_slots.is_empty() && !extras.suppress_debug_hooks
            })
            && !frame.has_vararg_state();
    }
    ir.loop_tail.is_none()
        || (frame.last_interpreted_pc.is_none()
            && !frame.has_vararg_state()
            && captured.is_empty()
            && frame
                .extras_ref()
                .is_none_or(|extras| extras.close_slots.is_empty() && !extras.suppress_debug_hooks))
}

/// Whether a local of the frame based at `base` that a closure captured is a
/// register the trace writes. The coroutine the trace resumes may write such
/// a local through the closure, and the trace's own write would come after.
fn resume_writes_captured(
    base: u32,
    captured: &[crate::vm::stack::OpenUpvalue],
    written: &[(FrameId, u16)],
) -> bool {
    captured.iter().any(|open| {
        open.slot
            .checked_sub(base)
            .and_then(|register| u16::try_from(register).ok())
            .is_some_and(|register| {
                written
                    .iter()
                    .any(|(frame, held)| *frame == FrameId(0) && *held == register)
            })
    })
}

/// Whether the cells of the trace's upvalue slots are one variable seen twice:
/// two slots with the same cell, or a slot with the cell of one of the frame's
/// captured locals.
fn upvalues_alias(
    frame: &FrameMut<'_>,
    captured: &[crate::vm::stack::OpenUpvalue],
    runtime: &Runtime,
    bank: &[Value],
    externals: &[ExternalSlot],
) -> bool {
    // Each cell is resolved once: a trace can carry hundreds of externals,
    // and resolving the earlier ones again for every slot made this check
    // quadratic on every entry.
    let mut cells = externals
        .iter()
        .filter_map(|slot| upvalue_cell(frame, runtime, bank, slot));
    let Some(first) = cells.next() else {
        return false;
    };
    let mut inline = [first; ALIAS_INLINE_CELLS];
    let mut spilled = Vec::new();
    let mut len = 1;
    for cell in cells {
        if len < ALIAS_INLINE_CELLS {
            inline[len] = cell;
        } else {
            if spilled.is_empty() {
                spilled.extend_from_slice(&inline);
            }
            spilled.push(cell);
        }
        len += 1;
    }
    if spilled.is_empty() {
        cells_alias(&mut inline[..len], captured)
    } else {
        cells_alias(&mut spilled, captured)
    }
}

/// The cells a frame's names stood for when [`aliases_hold`] and
/// [`upvalues_alias`] last held for it: the frame's upvalue cells, its
/// captured locals' cells by register, and [`Runtime::cell_rebinds`], which
/// covers the cells of the pinned closures. Both checks read nothing else
/// (the pinned closures are the trace's own), so while these stand, both
/// give the answer they gave.
#[derive(Debug)]
pub(super) struct CellsMemo {
    rebinds: u64,
    upvalues: Box<[CellHandle]>,
    captured: Box<[(u32, CellHandle)]>,
}

impl CellsMemo {
    fn of(
        frame: &FrameMut<'_>,
        captured: &[crate::vm::stack::OpenUpvalue],
        runtime: &Runtime,
    ) -> Self {
        Self {
            rebinds: runtime.cell_rebinds(),
            upvalues: frame.upvalues.iter().copied().collect(),
            captured: captured
                .iter()
                .map(|open| (open.slot.wrapping_sub(frame.base), open.cell))
                .collect(),
        }
    }

    /// Whether `frame` and its captured locals name the cells this memo
    /// remembers, with no closure's cell replaced since.
    fn holds(
        &self,
        frame: &FrameMut<'_>,
        captured: &[crate::vm::stack::OpenUpvalue],
        runtime: &Runtime,
    ) -> bool {
        self.rebinds == runtime.cell_rebinds()
            && *self.upvalues == *frame.upvalues
            && self.captured.len() == captured.len()
            && self
                .captured
                .iter()
                .zip(captured)
                .all(|((register, cell), open)| {
                    open.slot.wrapping_sub(frame.base) == *register && open.cell == *cell
                })
    }
}

/// How many upvalue cells [`upvalues_alias`] keeps on the stack.
const ALIAS_INLINE_CELLS: usize = 16;

/// Whether `cells` name one cell twice, or name the cell of a captured local.
/// Reorders `cells`.
fn cells_alias(cells: &mut [CellHandle], captured: &[crate::vm::stack::OpenUpvalue]) -> bool {
    if cells.len() <= ALIAS_INLINE_CELLS {
        return cells
            .iter()
            .enumerate()
            .any(|(position, cell)| cells[..position].contains(cell))
            || captured.iter().any(|open| cells.contains(&open.cell));
    }
    cells.sort_unstable();
    cells.windows(2).any(|pair| pair[0] == pair[1])
        || captured
            .iter()
            .any(|open| cells.binary_search(&open.cell).is_ok())
}

#[cfg(test)]
mod cell_alias_tests {
    use super::{ALIAS_INLINE_CELLS, cells_alias};
    use crate::runtime::{CellHandle, Runtime, Value};
    use crate::vm::stack::OpenUpvalue;

    fn cells(runtime: &mut Runtime, count: usize) -> Vec<CellHandle> {
        (0..count)
            .map(|n| runtime.create_cell(Value::Integer(n as i64)).unwrap())
            .collect()
    }

    #[test]
    fn a_shared_or_captured_cell_is_found_among_few_and_many_upvalues() {
        let mut runtime = Runtime::new();
        for count in [2, ALIAS_INLINE_CELLS, ALIAS_INLINE_CELLS + 1, 200] {
            let distinct = cells(&mut runtime, count);
            let local = runtime.create_cell(Value::Nil).unwrap();
            let open = |cell| OpenUpvalue { slot: 3, cell };
            assert!(!cells_alias(&mut distinct.clone(), &[]), "{count} distinct");
            assert!(
                !cells_alias(&mut distinct.clone(), &[open(local)]),
                "{count} distinct, an unrelated captured local"
            );
            // Two upvalues that name one cell, first and last.
            let mut shared = distinct.clone();
            *shared.last_mut().unwrap() = distinct[0];
            assert!(cells_alias(&mut shared, &[]), "{count} with a shared cell");
            // An upvalue that is also the frame's captured local.
            let last = *distinct.last().unwrap();
            assert!(
                cells_alias(&mut distinct.clone(), &[open(local), open(last)]),
                "{count} with a captured cell"
            );
        }
    }
}

/// What an upvalue cell or a global holds now, for [`external_slot`] and
/// [`external_holds`]: a slot of its own (a count) or a Lua value.
enum ExternalRead<'r> {
    Slot(TraceInputSlot),
    Value(&'r Value),
    Owned(Value),
}

/// The entry slot of an upvalue cell or a global. `None` when the cell is
/// missing or the global has no present raw value, which no input type
/// accepts. The table a global is read from goes into `holders`.
fn external_slot(
    frame: &FrameMut<'_>,
    runtime: &Runtime,
    external: &ExternalSlot,
    bank: &mut Vec<Value>,
    holders: (
        &mut GlobalHolders,
        Option<&mut crate::runtime::StringReadCache>,
    ),
    caller: Option<&CallerView>,
) -> Option<TraceInputSlot> {
    Some(
        match external_read(frame, runtime, external, bank, holders, caller)? {
            ExternalRead::Slot(slot) => slot,
            ExternalRead::Value(value) => input_slot(value, bank),
            ExternalRead::Owned(value) => input_slot(&value, bank),
        },
    )
}

/// Whether [`external_slot`] would give `held`, without searching the bank
/// or adding to it: an object holds when the lane `held` names pins it. The
/// check after a call from a trace makes one per external the trace reads,
/// and the bank holds every table and string the run has met.
fn external_holds(
    frame: &FrameMut<'_>,
    runtime: &Runtime,
    external: &ExternalSlot,
    bank: &[Value],
    holders: &mut GlobalHolders,
    held: &TraceInputSlot,
) -> bool {
    match external_read(frame, runtime, external, bank, (holders, None), None) {
        None => *held == TraceInputSlot::other(),
        Some(ExternalRead::Slot(slot)) => slot == *held,
        Some(ExternalRead::Value(value)) => input_slot_holds(value, held, bank),
        Some(ExternalRead::Owned(value)) => input_slot_holds(&value, held, bank),
    }
}

/// What `external` holds now; see [`external_slot`].
fn external_read<'r>(
    frame: &'r FrameMut<'_>,
    runtime: &'r Runtime,
    external: &ExternalSlot,
    bank: &[Value],
    (holders, mut reads): (
        &mut GlobalHolders,
        Option<&mut crate::runtime::StringReadCache>,
    ),
    caller: Option<&'r CallerView>,
) -> Option<ExternalRead<'r>> {
    // A constant name read from a table, through the entry's cache of reads
    // when the caller has one.
    let mut read = |table: TableHandle, name: &crate::string::LuaString| match reads.as_deref_mut()
    {
        Some(cache) => runtime.table_get_string_cached(table, name, cache),
        None => runtime.table_get_string_ref(table, name),
    };
    match external {
        ExternalSlot::Upvalue(_) | ExternalSlot::ClosureUpvalue { .. } => {
            let cell = upvalue_cell(frame, runtime, bank, external)?;
            Some(ExternalRead::Value(runtime.cell_ref(cell).ok()?))
        }
        // A register of the frame below the root frame.
        ExternalSlot::CallerRegister(register) => Some(ExternalRead::Value(
            caller?.registers.get(usize::from(*register))?,
        )),
        // A global of the chunk that called the root frame, from the
        // environment that frame runs in.
        ExternalSlot::CallerGlobal(constant) => {
            let caller = caller?;
            let Some(Constant::String(name)) =
                caller.chunk.constants.get(usize::try_from(*constant).ok()?)
            else {
                return None;
            };
            holders.add(caller.environment);
            Some(ExternalRead::Value(read(caller.environment, name)?))
        }
        // The frame's own count and closure, not Lua values of a cell.
        ExternalSlot::TailCalls => Some(ExternalRead::Slot(TraceInputSlot::integer(i64::from(
            frame.tail_calls,
        )))),
        ExternalSlot::RootClosure => Some(ExternalRead::Owned(Value::Closure(frame.closure?))),
        ExternalSlot::Environment => Some(ExternalRead::Owned(Value::Table(frame.environment))),
        ExternalSlot::VarargCount => Some(ExternalRead::Slot(TraceInputSlot::integer(
            i64::try_from(frame.varargs.len()).ok()?,
        ))),
        ExternalSlot::Vararg(index) => Some(ExternalRead::Value(
            frame
                .varargs
                .get(usize::try_from(*index).ok()?)
                .unwrap_or(&Value::Nil),
        )),
        ExternalSlot::Global(constant) => {
            let Some(Constant::String(name)) =
                frame.chunk.constants.get(usize::try_from(*constant).ok()?)
            else {
                return None;
            };
            holders.add(frame.environment);
            Some(ExternalRead::Value(read(frame.environment, name)?))
        }
        ExternalSlot::UpvalueField { upvalue, constant } => {
            let cell = *frame.upvalues.get(usize::from(*upvalue))?;
            let Constant::String(name) = frame
                .chunk
                .constants
                .get(usize::try_from(*constant).ok()?)?
            else {
                return None;
            };
            let Value::Table(table) = runtime.cell_ref(cell).ok()? else {
                return None;
            };
            holders.add(*table);
            Some(ExternalRead::Value(read(*table, name)?))
        }
        ExternalSlot::ClosureUpvalueField {
            closure,
            upvalue,
            constant,
        } => {
            let handle = pinned_closure(bank, *closure)?;
            let parts = runtime.closure_call_parts(handle)?;
            let cell = *parts.cells.get(usize::from(*upvalue))?;
            let Constant::String(name) = parts
                .chunk
                .constants
                .get(usize::try_from(*constant).ok()?)?
            else {
                return None;
            };
            let Value::Table(table) = runtime.cell_ref(cell).ok()? else {
                return None;
            };
            holders.add(*table);
            Some(ExternalRead::Owned(read(*table, name)?.clone()))
        }
        ExternalSlot::ClosureGlobal { closure, constant } => {
            let handle = pinned_closure(bank, *closure)?;
            let parts = runtime.closure_call_parts(handle)?;
            let Some(Constant::String(name)) =
                parts.chunk.constants.get(usize::try_from(*constant).ok()?)
            else {
                return None;
            };
            holders.add(parts.environment);
            Some(ExternalRead::Owned(read(parts.environment, name)?.clone()))
        }
    }
}

/// What entry needs to resolve one table binding.
#[derive(Debug)]
struct BindingPlan {
    /// The entry input that holds the table.
    input: usize,
    /// The key of a field binding, as a string value.
    key: Option<Value>,
    /// Whether some member of the tree stores into the binding's array
    /// ([`TraceOp::ArrayStore`] or [`TraceOp::ArraySet`]). Native code
    /// overwrites a slot of the array in place without asking the runtime,
    /// so entry refuses a read-only table for such a binding and leaves the
    /// store to the interpreter, which raises the error.
    writes: bool,
    /// Whether entry fixes the value under [`Self::key`] for the whole run
    /// (a [`BindingKind::Callee`] or [`BindingKind::StringField`]), which a
    /// store under that key must not change.
    pins: bool,
    /// Whether some member stores into the binding's table with a set that
    /// may move its hash part ([`TraceOp::ArraySet`] without a strict
    /// header). Entry refuses a run in which a field binding addresses that
    /// table's storage.
    moves: bool,
    /// What entry checks of a [`BindingKind::MetaClass`] binding, built once
    /// per trace: the events are strings, which entry would otherwise make at
    /// every run.
    class: Option<ClassPlan>,
}

/// The metatable class of a [`BindingKind::MetaClass`] binding, ready for the
/// entry check.
#[derive(Debug)]
struct ClassPlan {
    metatable: TableHandle,
    /// Each pinned event with the identity of the closure it holds, or
    /// [`crate::jit_ir::trace::META_PIN_ABSENT`].
    pins: Box<[(Value, Option<Metamethod>, u64)]>,
    bare: u8,
}

impl ClassPlan {
    fn new(metatable: u64, pins: &[crate::jit_ir::trace::MetaPin], bare: u8) -> Self {
        Self {
            metatable: TableHandle::from_native_identity(metatable),
            pins: pins
                .iter()
                .map(|pin| {
                    (
                        Value::String(crate::string::LuaString::from(&pin.event[..])),
                        cached_event(&pin.event),
                        pin.handler,
                    )
                })
                .collect(),
            bare,
        }
    }

    /// Whether `metatable` holds what the class pinned for every event: the
    /// raw value is the closure of the pinned identity, or absent.
    fn pins_hold(&self, runtime: &Runtime, metatable: TableHandle) -> bool {
        self.pins.iter().all(|(key, cached, handler)| {
            let Value::String(key) = key else {
                return false;
            };
            if *handler == crate::jit_ir::trace::META_PIN_ABSENT {
                return match cached {
                    Some(event) => runtime.metatable_lacks(metatable, *event),
                    None => matches!(
                        runtime.table_get_string_ref(metatable, key),
                        None | Some(Value::Nil)
                    ),
                };
            }
            matches!(
                runtime.table_get_string_ref(metatable, key),
                Some(Value::Closure(held)) if held.native_identity() == *handler
            )
        })
    }

    /// Whether the entry holds: the pinned metatable is alive and has what
    /// the class pinned, and the types the trace reads as having no handler
    /// have none.
    fn holds(&self, runtime: &Runtime) -> bool {
        use crate::jit_ir::trace::{META_BARE_NUMBERS, META_BARE_SIMPLE, META_BARE_STRINGS};
        if runtime.table_is_readonly(self.metatable).is_err()
            || !self.pins_hold(runtime, self.metatable)
        {
            return false;
        }
        let bare = |value: &Value| matches!(runtime.raw_metatable(value), Ok(None));
        if self.bare & META_BARE_NUMBERS != 0
            && !(bare(&Value::Number(0.0)) && bare(&Value::Integer(0)))
        {
            return false;
        }
        if self.bare & META_BARE_SIMPLE != 0 && !(bare(&Value::Nil) && bare(&Value::Boolean(false)))
        {
            return false;
        }
        if self.bare & META_BARE_STRINGS != 0 {
            // Strings share one metatable, which has no handler for the
            // events (or none at all).
            match runtime.raw_metatable(&Value::String(crate::string::LuaString::from(&b""[..]))) {
                Ok(None) => {}
                Ok(Some(strings)) => {
                    if !self.pins.iter().all(|(key, _, _)| {
                        let Value::String(key) = key else {
                            return false;
                        };
                        matches!(
                            runtime.table_get_string_ref(strings, key),
                            None | Some(Value::Nil)
                        )
                    }) {
                        return false;
                    }
                }
                Err(_) => return false,
            }
        }
        true
    }
}

/// The event a metatable's absence cache covers, by the name of its field.
fn cached_event(name: &[u8]) -> Option<Metamethod> {
    Some(match name {
        b"__newindex" => Metamethod::NewIndex,
        b"__gc" => Metamethod::Gc,
        b"__metatable" => Metamethod::Metatable,
        b"__mode" => Metamethod::Mode,
        b"__index" => Metamethod::Index,
        b"__len" => Metamethod::Len,
        _ => return None,
    })
}

/// Plans the binding resolution of `compiled`: which entry input holds each
/// bound table of its tree. `None` when a binding has no table input.
fn plan_bindings(compiled: &CompiledTrace) -> Option<Box<[BindingPlan]>> {
    let mut written = vec![false; compiled.bindings().len()];
    let mut moving = vec![false; compiled.bindings().len()];
    for member in 0..compiled.member_count() {
        let ir = compiled.lowered_member(member);
        for instruction in &ir.body {
            if let TraceOp::ArrayStore { binding, .. }
            | TraceOp::ArrayOverwrite { binding, .. }
            | TraceOp::ArraySet { binding, .. } = instruction.op
                && let Some(flag) = usize::try_from(binding)
                    .ok()
                    .and_then(|binding| written.get_mut(binding))
            {
                *flag = true;
            }
            if let TraceOp::ArraySet { binding, .. } = instruction.op
                && ir.set_moves_storage(binding)
                && let Some(flag) = usize::try_from(binding)
                    .ok()
                    .and_then(|binding| moving.get_mut(binding))
            {
                *flag = true;
            }
        }
    }
    compiled
        .bindings()
        .iter()
        .zip(written.into_iter().zip(moving))
        .map(|(binding, (writes, moves))| {
            let input = compiled
                .entry_contract()
                .iter()
                .position(|(register, ty)| {
                    *register == binding.register
                        && *ty
                            == if matches!(binding.kind, BindingKind::Bytes { .. }) {
                                TraceType::String
                            } else {
                                TraceType::Table
                            }
                })?;
            let key = match &binding.kind {
                BindingKind::Field { key, .. }
                | BindingKind::Callee { key, .. }
                | BindingKind::StringField { key, .. } => {
                    Some(Value::String(crate::string::LuaString::from(&key[..])))
                }
                BindingKind::Array { .. }
                | BindingKind::Length { .. }
                | BindingKind::Bytes { .. }
                | BindingKind::Header { .. }
                | BindingKind::MetaClass { .. }
                | BindingKind::NoMetatable
                | BindingKind::Hash => None,
            };
            let pins = matches!(
                binding.kind,
                BindingKind::Callee { .. }
                    | BindingKind::StringField { .. }
                    | BindingKind::MetaClass { .. }
            );
            let class = match &binding.kind {
                BindingKind::MetaClass {
                    metatable,
                    pins,
                    bare,
                } => Some(ClassPlan::new(*metatable, pins, *bare)),
                _ => None,
            };
            Some(BindingPlan {
                input,
                key,
                writes,
                pins,
                moves,
                class,
            })
        })
        .collect()
}

/// The kind of value a table slot must hold for a trace type.
const fn slot_kind(ty: TraceType) -> NativeSlotKind {
    match ty {
        TraceType::Float => NativeSlotKind::Number,
        TraceType::Integer => NativeSlotKind::Integer,
        TraceType::Table => NativeSlotKind::Table,
        TraceType::Nil => NativeSlotKind::Absent,
        _ => NativeSlotKind::Boolean,
    }
}

/// The key and slot kind of each field of run-time tables of `compiled`.
fn plan_fields(compiled: &CompiledTrace) -> Box<[(Value, NativeSlotKind)]> {
    compiled
        .fields()
        .iter()
        .map(|field| {
            (
                Value::String(crate::string::LuaString::from(&field.key[..])),
                slot_kind(field.ty),
            )
        })
        .collect()
}

/// Entries of a run's cache of coroutine lanes ([`RunTables::thread_lane`]).
const THREAD_LANE_CACHE: usize = 16;

/// What a run of a trace that meets coroutines keeps: the lanes it gave them
/// and, for a trace that resumes them, what it lends the helper that does.
#[derive(Debug)]
pub(super) struct ResumeState {
    pub(super) host: Option<ResumeHost>,
    /// The lanes [`RunTables::thread_lane`] gave recently, by identity.
    cache: [(u64, u32); THREAD_LANE_CACHE],
}

impl ResumeState {
    pub(super) const fn new(host: Option<ResumeHost>) -> Self {
        Self {
            host,
            cache: [(0, u32::MAX); THREAD_LANE_CACHE],
        }
    }
}

/// What a run of a trace that resumes coroutines lends the helper that does
/// it: the addresses of the VM that entered the trace and of the call stack
/// the trace runs on.
///
/// The helper runs the interpreter (`Vm::resume_for_trace`), which uses the
/// whole VM (its policy, statistics, runtime and bank) and the stack. So
/// while a native call is in flight the run holds no reference to any of
/// them. It holds the pointers it was lent (`TracePolicy::enter_lent`),
/// `RunTables` holds its runtime and bank as pointers, and everything the run
/// uses after the call it borrows again from them. What the run does keep
/// references to is storage the resumed code never touches: shared views of
/// fields of the running trace's cache entry (its hint words as a pointer),
/// and the heap buffers of the entry slots, binding
/// views, exit area, table space and table lanes. Native execution is
/// suspended for the resumed code and flushes wait for the run, so it
/// neither enters, records, installs, retires nor evicts a trace. The helper
/// follows the same rule: its references to the VM and the stack end before
/// it next uses the run's runtime or bank, and are made again after.
#[derive(Debug)]
pub(super) struct ResumeHost {
    pub(super) vm: usize,
    pub(super) frames: usize,
    /// The frame the trace runs on, in the stack at `frames`.
    pub(super) frame_index: usize,
    /// The restore plans of the trace's exits and the length of its exit
    /// area, for a helper that makes a host call and writes an exit's state
    /// back to the frame first ([`RunTables::restore_exit`]). The plans
    /// belong to the cache entry of the running trace, which flushes leave
    /// alone for the run.
    pub(super) restores: (*const ExitRestore, usize),
    pub(super) area_len: usize,
    /// The cells and globals the trace reads, by the entry input that holds
    /// each: the coroutine may have written them, which the helper checks
    /// with [`RunTables::externals_hold`].
    pub(super) checks: Vec<(usize, ExternalSlot)>,
    /// The upvalues among those reads whose cells were closed when the run
    /// began. A call after which [`Runtime::cell_writes`] stands where it
    /// stood wrote none of them, and the helper leaves them unchecked.
    pub(super) closed_cells: Vec<(usize, ExternalSlot)>,
    /// The cells and globals the trace writes. They live in native values
    /// until an exit stores them, so a host call stores them first (the
    /// restore of its `Before` snapshot) and the helper compares them with
    /// memory again afterwards ([`RunTables::externals_hold`]).
    pub(super) written: Vec<ExternalSlot>,
    /// The registers of the frame that a closure captured and the trace
    /// reads, by the entry input that holds each: the thread may have
    /// assigned them.
    pub(super) captured: Vec<(usize, usize)>,
    /// The captured registers the trace writes: stored by the call's restore,
    /// and compared with memory again after the call, as `written`.
    pub(super) captured_written: Vec<usize>,
    /// The trace's promises that two names share one cell, which entry
    /// checks ([`aliases_hold`]). A call that replaces a closure's cell
    /// (`debug.upvaluejoin`, `lua_upvaluejoin`) can break one, so the helper
    /// checks them again after it.
    pub(super) aliases: Box<[CellAlias]>,
    /// The upvalue slots of the window, which entry checks name distinct
    /// cells ([`upvalues_alias`]), and whether that check applies with no
    /// captured local (two or more of them). Empty when the trace names no
    /// upvalue.
    pub(super) upvalue_slots: Vec<ExternalSlot>,
    pub(super) many_upvalues: bool,
    /// The cells entry checked the aliases and upvalues against: a call
    /// after which they still stand needs neither check again.
    pub(super) cells_seen: CellsMemo,
}

/// How many tables a run of a trace that makes tables may make before it
/// leaves: the room the entry reserves in the arena of tables.
const TABLE_ALLOCATION_HEADROOM: usize = 4096;

/// The most lanes a run's bank may hold for a table to be found by scanning
/// the bank instead of through the lane map.
const SCAN_LANES: usize = 16;

/// The cells and globals `ir` and its sides write: the external windows
/// among their written registers.
fn written_externals(ir: &TraceIr) -> Vec<ExternalSlot> {
    let mut slots = Vec::new();
    for member in std::iter::once(ir).chain(ir.sides.iter().map(|side| &*side.ir)) {
        let Some(root) = member.frames.first() else {
            continue;
        };
        let first = usize::from(root.register_count).saturating_sub(member.external.len());
        for &(frame, register) in &member.written_registers {
            if frame == FrameId(0)
                && let Some(slot) = usize::from(register)
                    .checked_sub(first)
                    .and_then(|index| member.external.get(index))
                && !slots.contains(slot)
            {
                slots.push(*slot);
            }
        }
    }
    slots
}

/// How many of `externals` name an upvalue cell.
fn upvalue_slots(externals: &[ExternalSlot]) -> usize {
    externals
        .iter()
        .filter(|slot| {
            matches!(
                slot,
                ExternalSlot::Upvalue(_) | ExternalSlot::ClosureUpvalue { .. }
            )
        })
        .count()
}

/// The positions among `fields` of the words of a row that name no slot (the
/// metatable word and the table's address), when the trace lists them.
fn row_words_of(fields: &[(Value, NativeSlotKind)]) -> crate::runtime::NativeRowWords {
    let position = |wanted: &[u8]| {
        fields
            .iter()
            .position(|(key, _)| matches!(key, Value::String(text) if text.as_bytes() == wanted))
    };
    crate::runtime::NativeRowWords {
        meta: position(crate::jit_ir::trace::META_FIELD_KEY),
        address: position(crate::jit_ir::trace::TABLE_ADDRESS_KEY),
    }
}

/// The places of a [`LaneIndex`], a power of two, as native code knows it
/// (see [`crate::jit::TraceTables::lane_index`]).
const LANE_INDEX_PLACES: usize = crate::jit::TRACE_LANE_INDEX_PLACES;

/// The most tables a [`LaneIndex`] holds: with more the index would fill up,
/// and the run finds a table's lane as it does when the bank is large.
const LANE_INDEX_MAX: usize = 160;

/// One place of a [`LaneIndex`], laid out as native code reads it.
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct LaneEntry {
    identity: u64,
    lane: u32,
    /// The run the entry belongs to, shifted left by one: an entry of another
    /// run is free. The low bit says the table has its row filled.
    stamp: u32,
}

/// The lane of each table of a run's bank, by identity, in a small
/// open-addressed table that the policy keeps from run to run and a run
/// empties by taking a new stamp: a run that asks for the lane of the same
/// few tables again and again (a loop over an array of records) finds them in
/// a few instructions, and a table not in the index is not in the bank.
pub(super) struct LaneIndex {
    stamp: u32,
    entries: [LaneEntry; LANE_INDEX_PLACES],
    /// The run's memo of cell reads and of the lanes its table reads
    /// found, stamped like the entries (see [`super::helpers::ReadMemo`]).
    reads: super::helpers::ReadMemo,
}

impl std::fmt::Debug for LaneIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LaneIndex")
            .field("stamp", &self.stamp)
            .finish_non_exhaustive()
    }
}

impl Default for LaneIndex {
    fn default() -> Self {
        Self {
            stamp: 0,
            entries: [LaneEntry::default(); LANE_INDEX_PLACES],
            reads: super::helpers::ReadMemo::default(),
        }
    }
}

impl LaneIndex {
    /// Empties the index for a new run and returns its stamp.
    fn begin(&mut self) -> u32 {
        self.stamp += 1;
        // The stamp word has the run in its upper 31 bits.
        if self.stamp >= 1 << 31 {
            self.entries = [LaneEntry::default(); LANE_INDEX_PLACES];
            self.reads = super::helpers::ReadMemo::default();
            self.stamp = 1;
        }
        self.stamp
    }
}

/// A word that tells tables apart, from their identity.
#[inline]
fn mix_identity(identity: u64) -> usize {
    // The slot is the low word, the generation the high: both move.
    ((identity ^ (identity >> 32)).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize
}

/// What a run tells native code about the tables it meets: it gives a table
/// a lane of the run's root bank, which is what an exit restores from, and
/// finds the slots of a table's fields (see [`oximond::jit::TraceTables`]).
// The run's mode flags are a deliberate flat shape on this hot record.
#[allow(clippy::struct_excessive_bools)]
pub(super) struct RunTables<'a> {
    /// The run's runtime and root bank, as the pointers the run was lent
    /// (see [`ResumeHost`]): a helper that resumes a coroutine runs the VM,
    /// which uses both, so the run keeps no reference to either across a
    /// native call. [`Self::runtime`] and [`Self::bank`] make one where it is
    /// used.
    runtime: *mut Runtime,
    bank: *mut Vec<Value>,
    /// The lane of each table of the bank, by identity. A loop over an array
    /// of records asks once per iteration, so the lookup must not scan a
    /// large bank.
    lanes: &'a mut FxHashMap<u64, u32>,
    /// Whether `lanes` holds the bank's tables. A run over a small bank finds
    /// a table by scanning the bank, which costs less than hashing every
    /// table at entry, and builds the map only if the bank outgrows
    /// [`SCAN_LANES`].
    lanes_ready: bool,
    /// The lane of each table of the bank, by identity, when the bank is
    /// small (see [`LaneIndex`]); `index_ok` says it holds all of them.
    index: &'a mut LaneIndex,
    index_ok: bool,
    index_len: usize,
    stamp: u32,
    /// The lanes that hold no table identity: the control lane of a `next`
    /// loop and the lane of the value it keeps.
    control: Option<std::ops::Range<usize>>,
    /// What a trace that reads and writes callee cells natively needs of the
    /// closures of its run. Boxed and absent for every other trace: a run
    /// builds one per entry, and the struct moves through its builders.
    closures: Option<Box<RunClosures>>,
    /// What a trace that meets coroutines needs of its run, boxed like
    /// `closures` for the same reason (see [`ResumeState`]).
    pub(super) resume: Option<Box<ResumeState>>,
    /// Whether every table a native read meets gets a lane of its own, even
    /// one the bank holds already. Only a function trace's run, which stores
    /// nothing and compares two tables by the handles their lanes hold
    /// ([`HelperId::TABLE_SAME`]), not by lane number, can do without telling
    /// them apart. The other uses of lane numbers stay sound under fresh
    /// lanes: closures keep one lane each, stores reach only the tables the
    /// run made (by their `TABLE_NEW` op), and table keys are refused at
    /// recording. A change that relaxes one of these must revisit identity.
    fresh_lanes: bool,
    fields: &'a [(Value, NativeSlotKind)],
    /// The positions among `fields` of the words of a row that name no slot
    /// (see [`crate::runtime::NativeRowWords`]).
    meta_field: crate::runtime::NativeRowWords,
    /// The first byte a value of each field's kind has (see
    /// [`NativeSlotKind::tag`]).
    tags: &'a [u8],
    /// The trace's memo of rows ([`Self::fill_table_row`]) as a pointer and a
    /// length in words, like `hints`; empty for a run that keeps none.
    memo: (*mut usize, usize),
    /// Where each field was last found in a table, for the next lookup.
    /// The words of the trace's field hints, as a pointer and a length: the
    /// run keeps no reference to the cache entry across a native call, so a
    /// slice is made from them where a lookup uses it.
    hints: (*mut usize, usize),
    /// Which entry input holds the table of each binding, and the run's
    /// entry inputs, for [`TraceTables::set`].
    plans: &'a [BindingPlan],
    inputs: &'a [TraceInputSlot],
    /// The strings of the trace's root slots, which a host call reads.
    roots: &'a [Value],
    /// What the run's host calls remember.
    host: HostState,
    /// The run's string lengths ([`TraceTables::string_lengths`]), which the
    /// host call entry writes through.
    lengths: (*mut u32, usize),
    /// The run's flags of which lanes have a row ([`TraceTables::row_flags`]).
    rows: (*mut bool, usize),
    /// The run's row storage ([`TraceTables::row_space`]), which a host call
    /// makes again.
    space: *mut crate::jit::TraceTableSpace,
    /// [`crate::runtime::table_key_layout`] when the run began or last
    /// refilled its rows: while it stands, the ordered and hashed storage
    /// the rows point into has not moved (see [`TraceTables::row_holds`]).
    layout_seen: u64,
    /// Whether `layout_seen` moved before the refill under way, whose rows
    /// are then checked against their tables' entries ([`Self::refill_rows`]).
    layout_moved: bool,
    /// Which bindings of the run resolve an array part, for the table set
    /// entry.
    arrays: &'a [bool],
    /// The lane of the bank that holds the address of [`Self::cursor`], when
    /// the trace steps along the hash part in native code.
    cursor_lane: Option<usize>,
    /// The description of the hash part's snapshot native code reads.
    cursor: crate::runtime::NativeCursor,
    /// The snapshot of a hash part too small to keep one.
    cursor_scratch: crate::runtime::NativeCursorScratch,
    /// The kind of every binding of the run, for the words a table write
    /// rewrites (an array part and a border).
    kinds: &'a [crate::jit_ir::trace::TableBinding],
    /// Whether some binding of the run is a table border or a table header,
    /// words a write to the table rewrites.
    has_lengths: bool,
    /// The tables that hold the globals the run read at entry.
    holders: GlobalHolders,
    /// Whether some binding of the run pins the value under its key
    /// ([`PinnedKeys::holds`]).
    pins_keys: bool,
    /// The calls the run made of the store helpers, which stand for the
    /// stores native code did not make itself ([`StoreCalls`]).
    pub(super) store_calls: StoreCalls,
    /// The restore plans of the running trace's exits and the length of its
    /// exit area, as pointers like the hints: a guard that enters a side
    /// with the virtual tables of its exit builds them from the plan
    /// ([`TraceTables::materialize_exit`]). Empty for a run that has none.
    exit_plans: (*const ExitRestore, usize),
    exit_area_len: usize,
    /// Where the fields of the run's callee and string bindings were last
    /// found ([`Self::field_holds`]).
    field_slots: FieldSlots,
}

/// The bindings a run remembers a slot for.
const FIELD_SLOTS: usize = 8;

/// Where [`RunTables::field_holds`] last found the field of a binding: the
/// slot of the key in the binding's own table, with the key layout count of
/// that look.
#[derive(Clone, Copy)]
struct FieldSlot {
    binding: usize,
    table: TableHandle,
    layout: u64,
    slot: *const Value,
}

/// The slots of the first [`FIELD_SLOTS`] bindings a run checked after a
/// host call. The array is written before it is read, so a run that makes
/// no host call pays nothing for it.
struct FieldSlots {
    used: usize,
    slots: [std::mem::MaybeUninit<FieldSlot>; FIELD_SLOTS],
}

impl FieldSlots {
    const fn new() -> Self {
        Self {
            used: 0,
            slots: [std::mem::MaybeUninit::uninit(); FIELD_SLOTS],
        }
    }

    /// The slot remembered for `binding`, if it was found in `table` and
    /// the key layout count is still `layout`.
    #[allow(unsafe_code)]
    #[inline]
    fn get(&self, binding: usize, table: TableHandle, layout: u64) -> Option<*const Value> {
        self.slots[..self.used].iter().find_map(|slot| {
            // SAFETY: the first `used` slots are written.
            let slot = unsafe { slot.assume_init_ref() };
            (slot.binding == binding && slot.table == table && slot.layout == layout)
                .then_some(slot.slot)
        })
    }

    /// Remembers `found` in place of what its binding had.
    #[allow(unsafe_code)]
    fn set(&mut self, found: FieldSlot) {
        let place = self.slots[..self.used].iter().position(|slot| {
            // SAFETY: the first `used` slots are written.
            unsafe { slot.assume_init_ref() }.binding == found.binding
        });
        match place {
            Some(place) => self.slots[place] = std::mem::MaybeUninit::new(found),
            None if self.used < FIELD_SLOTS => {
                self.slots[self.used] = std::mem::MaybeUninit::new(found);
                self.used += 1;
            }
            None => {}
        }
    }
}

/// The closures of one run of a trace that reads and writes callee cells
/// natively (see [`CompiledTrace::uses_closure_rows`]).
#[derive(Default)]
pub(super) struct RunClosures {
    /// The cells the trace keeps in window slots: no closure that shares one
    /// may be called through the prototype guard, since the cells such a
    /// closure reads and writes are not the ones the slot stands for. Its
    /// row stays zero, which no prototype matches.
    held: Vec<CellHandle>,
    /// The lane of each closure of the bank, by identity, filled the first
    /// time the run asks for the lane of a closure (a closure loaded out of
    /// an array). Most runs never do.
    lanes: Option<FxHashMap<u64, u32>>,
    /// Whether the rows also name the lane of the table each cell holds
    /// ([`crate::jit::CLOSURE_ROW_TABLE_WORDS`]).
    table_cells: bool,
    /// The lanes the last closure row named for its cells, whose rows the
    /// space fills next ([`TraceTables::row_followers`]).
    followers: Vec<usize>,
}

/// How often a run called each helper that stores under a run-time string key
/// ([`HelperId::SET_STRING`], [`HelperId::SET_PRESENT`],
/// [`HelperId::SET_REVIVE`] and [`HelperId::DELETE_STRING`]), for
/// [`TraceStatistics`].
#[derive(Clone, Copy, Default)]
pub(super) struct StoreCalls {
    pub(super) set_string: u64,
    /// The calls of [`HelperId::SET_STRING`] that overwrote a present key's
    /// scalar with one of its type.
    pub(super) set_string_overwrites: u64,
    pub(super) set_present: u64,
    pub(super) set_revive: u64,
    pub(super) delete_string: u64,
    /// Calls of the table set entry (`run_table_set`): a store under a number
    /// key that native code did not make itself.
    pub(super) set_array: u64,
    /// Calls of [`HelperId::TABLE_NEW`], which made a table a virtual one
    /// stood for.
    pub(super) table_new: u64,
    /// Calls of the helpers that read or write a callee's cell, or check its
    /// prototype, which native code did not do itself.
    pub(super) closure: u64,
    /// Calls of the lookup helpers for a table met by lane (the
    /// `TABLE_GET_*` family and `TABLE_LENGTH`).
    pub(super) table_lookup: u64,
}

/// The tables whose fields a run read once at entry as globals (the
/// environment, or the table an upvalue holds; see `ExternalSlot::Global`).
/// A store under a run-time string key into one of them could rebind such a
/// global unseen. Past [`Self::CAPACITY`] tables every table counts.
#[derive(Clone, Copy)]
pub(super) struct GlobalHolders {
    /// The tables' identities ([`TableHandle::native_identity`]), with
    /// [`Self::FREE`] for an unused slot: a store compares its table with all
    /// of them at once, with no branch.
    tables: [u64; Self::CAPACITY],
    overflow: bool,
}

impl Default for GlobalHolders {
    fn default() -> Self {
        Self {
            tables: [Self::FREE; Self::CAPACITY],
            overflow: false,
        }
    }
}

impl GlobalHolders {
    const CAPACITY: usize = 4;
    const FREE: u64 = u64::MAX;

    fn add(&mut self, table: TableHandle) {
        let identity = table.native_identity();
        if self.tables.contains(&identity) {
            return;
        }
        match self.tables.iter_mut().find(|held| **held == Self::FREE) {
            Some(free) => *free = identity,
            None => self.overflow = true,
        }
    }

    #[inline]
    fn holds(&self, table: TableHandle) -> bool {
        self.overflow || self.tables.contains(&table.native_identity())
    }
}

/// What a store under a run-time string key must leave to the interpreter
/// (see [`RunTables::pinned_keys`]).
#[derive(Clone, Copy)]
pub(super) struct PinnedKeys<'a> {
    plans: &'a [BindingPlan],
    holders: GlobalHolders,
    /// Whether any of `plans` pins its key, so that `holds` has any to compare.
    pins_keys: bool,
}

impl PinnedKeys<'_> {
    /// Whether a binding of the run fixes the value under `key`.
    pub(super) fn holds(self, key: &[u8]) -> bool {
        self.pins_keys
            && self.plans.iter().any(|plan| {
                plan.pins
                    && (matches!(&plan.key, Some(Value::String(pinned)) if pinned.as_bytes() == key)
                        // A metatable class fixes the events of its
                        // metatable, which all start this way.
                        || (plan.class.is_some() && key.starts_with(b"__")))
            })
    }

    /// Whether `table` holds a global the run read at entry.
    pub(super) fn holds_globals(self, table: TableHandle) -> bool {
        self.holders.holds(table)
    }
}

impl<'a> RunTables<'a> {
    /// The run's runtime, for the statement that uses it.
    ///
    /// The pointer is the one the run was lent, valid for the run, and this
    /// makes a reference from it each time, so that none is held while a
    /// helper runs the interpreter (`resume_call` does not call this
    /// between the resume and the refresh that follows it). The reference is
    /// not tied to `self`, as the `&'a mut` this replaced was not, so that
    /// one statement can use the runtime and the run's other fields. The
    /// caller makes no second reference to the runtime that lives as long.
    #[allow(unsafe_code)]
    fn runtime<'r>(&self) -> &'r mut Runtime {
        // SAFETY: as above.
        unsafe { &mut *self.runtime }
    }

    /// The trace's field hints, for the statement that uses them. As
    /// [`Self::runtime`]: the words belong to the cache entry, which the
    /// resumed code never touches, and only the run writes them.
    #[allow(unsafe_code)]
    fn hints<'r>(&self) -> &'r mut [usize] {
        // SAFETY: the words of the entry's hint vector, valid for the run.
        unsafe { std::slice::from_raw_parts_mut(self.hints.0, self.hints.1) }
    }

    /// The run's root bank, for the statement that uses it. As
    /// [`Self::runtime`].
    #[allow(unsafe_code)]
    fn bank<'r>(&self) -> &'r mut Vec<Value> {
        // SAFETY: as `Self::runtime`.
        unsafe { &mut *self.bank }
    }

    /// The lane of a closure, adding it to the run's bank when it has none
    /// and the run has room (the bank is capped as for tables). The bank
    /// roots it until the run ends, like the tables.
    pub(super) fn closure_lane(&mut self, closure: ClosureHandle) -> Option<usize> {
        let identity = closure.native_identity();
        let bank = self.bank();
        let state = self.closures.get_or_insert_with(Box::default);
        let map = state.lanes.get_or_insert_with(|| {
            bank.iter()
                .enumerate()
                .filter_map(|(lane, held)| match held {
                    Value::Closure(known) => {
                        Some((known.native_identity(), u32::try_from(lane).ok()?))
                    }
                    _ => None,
                })
                .collect()
        });
        if let Some(lane) = map.get(&identity) {
            return Some(*lane as usize);
        }
        if bank.len() >= MAX_TRACE_TABLE_LANES {
            return None;
        }
        bank.push(Value::Closure(closure));
        let lane = bank.len() - 1;
        map.insert(identity, u32::try_from(lane).ok()?);
        Some(lane)
    }

    /// The lane of a coroutine, like [`Self::closure_lane`]: one thread has
    /// one lane, which the bank roots until the run ends. A loop over an array
    /// of coroutines asks once per iteration, so the lanes already given sit
    /// in a small direct-mapped cache by identity, and a miss looks through
    /// the bank for the thread before it adds a lane.
    pub(super) fn thread_lane(&mut self, thread: crate::runtime::ThreadHandle) -> Option<usize> {
        let bank = self.bank();
        let identity = thread.native_identity();
        let slot = (identity as usize) % THREAD_LANE_CACHE;
        let state = self
            .resume
            .get_or_insert_with(|| Box::new(ResumeState::new(None)));
        let (known, lane) = state.cache[slot];
        if known == identity && lane != u32::MAX {
            return Some(lane as usize);
        }
        let held = bank
            .iter()
            .position(|held| matches!(held, Value::Thread(other) if *other == thread));
        let lane = if let Some(lane) = held {
            lane
        } else {
            if bank.len() >= MAX_TRACE_TABLE_LANES {
                return None;
            }
            bank.push(Value::Thread(thread));
            bank.len() - 1
        };
        state.cache[slot] = (identity, u32::try_from(lane).ok()?);
        Some(lane)
    }

    /// The lane of a userdata: one userdata has one lane, which the bank
    /// roots until the run ends. The bank is searched, since a userdata read
    /// out of a table is rare next to a table.
    pub(super) fn userdata_lane(
        &mut self,
        userdata: crate::runtime::UserdataHandle,
    ) -> Option<usize> {
        let bank = self.bank();
        if let Some(lane) = bank
            .iter()
            .position(|held| matches!(held, Value::Userdata(other) if *other == userdata))
        {
            return Some(lane);
        }
        if bank.len() >= MAX_TRACE_TABLE_LANES {
            return None;
        }
        bank.push(Value::Userdata(userdata));
        Some(bank.len() - 1)
    }

    /// The tables of `bank` as it stands, which entry filled.
    ///
    /// `runtime` and `bank` must be valid for the whole run and used by
    /// nothing but the run and the code a helper of it runs.
    fn new(
        runtime: *mut Runtime,
        bank: *mut Vec<Value>,
        lanes: &'a mut FxHashMap<u64, u32>,
        index: &'a mut LaneIndex,
        fields: &'a [(Value, NativeSlotKind)],
        hints: (*mut usize, usize),
        meta_field: crate::runtime::NativeRowWords,
        control: Option<&std::ops::Range<usize>>,
    ) -> Self {
        Self {
            runtime,
            bank,
            lanes,
            lanes_ready: false,
            index,
            index_ok: false,
            index_len: 0,
            stamp: 0,
            control: control.cloned(),
            closures: None,
            resume: None,
            fresh_lanes: false,
            meta_field,
            tags: &[],
            memo: (std::ptr::null_mut(), 0),
            fields,
            hints,
            plans: &[],
            inputs: &[],
            roots: &[],
            host: HostState::default(),
            lengths: (std::ptr::null_mut(), 0),
            rows: (std::ptr::null_mut(), 0),
            space: std::ptr::null_mut(),
            layout_seen: crate::runtime::table_key_layout(),
            layout_moved: false,
            arrays: &[],
            cursor_lane: None,
            cursor: crate::runtime::NativeCursor::default(),
            cursor_scratch: crate::runtime::NativeCursorScratch::default(),
            kinds: &[],
            has_lengths: false,
            holders: GlobalHolders::default(),
            pins_keys: false,
            store_calls: StoreCalls::default(),
            exit_plans: (std::ptr::null(), 0),
            exit_area_len: 0,
            field_slots: FieldSlots::new(),
        }
    }

    /// Names the restore plans of the trace's exits and the length of the
    /// run's exit area (see [`TraceTables::materialize_exit`]). The plans
    /// must stay where they are for the run.
    pub(super) fn set_exit_plans(&mut self, plans: &[ExitRestore], area_len: usize) {
        self.exit_plans = (plans.as_ptr(), plans.len());
        self.exit_area_len = area_len;
    }

    /// Starts the index for this run from the bank as it stands: the tables
    /// of the inputs, and what the pinned lanes hold. Without room for them
    /// the run does without (see [`Self::find_table_lane`]).
    fn populate_index(&mut self, makes_tables: bool) {
        self.stamp = self.index.begin();
        self.index_len = 0;
        // A run that makes tables finds the lanes of the ones it made through
        // `made`, which the index leaves out, so it searches the bank as it
        // always did.
        self.index_ok = !makes_tables && self.bank().len() <= LANE_INDEX_MAX;
        if !self.index_ok {
            return;
        }
        // SAFETY: the run's bank, valid for the run.
        #[allow(unsafe_code)]
        let bank = unsafe { &*self.bank };
        for (lane, held) in bank.iter().enumerate() {
            let Value::Table(table) = held else {
                continue;
            };
            if self
                .control
                .as_ref()
                .is_some_and(|lanes| lanes.contains(&lane))
            {
                continue;
            }
            let _ = self.index_insert(table.native_identity(), lane);
        }
    }

    /// Adds the table with `identity` at `lane` to the index (a table there
    /// already has the later lane, as in the lane map) and returns its place.
    /// The index gives up when it holds too many.
    fn index_insert(&mut self, identity: u64, lane: usize) -> Option<usize> {
        let Ok(lane) = u32::try_from(lane) else {
            self.index_ok = false;
            return None;
        };
        let stamp = self.stamp << 1;
        let mut place = mix_identity(identity) & (LANE_INDEX_PLACES - 1);
        loop {
            let entry = &mut self.index.entries[place];
            if entry.stamp >> 1 != self.stamp {
                if self.index_len >= LANE_INDEX_MAX {
                    self.index_ok = false;
                    return None;
                }
                self.index_len += 1;
                *entry = LaneEntry {
                    identity,
                    lane,
                    stamp,
                };
                return Some(place);
            }
            if entry.identity == identity {
                entry.lane = lane;
                entry.stamp = stamp;
                return Some(place);
            }
            place = (place + 1) & (LANE_INDEX_PLACES - 1);
        }
    }

    /// The place in the index of the table `table`, if the bank still holds
    /// that table at the lane the place names. `None` for a table not in the
    /// bank, or when the index does not hold the bank's tables.
    #[inline]
    fn index_place(&self, table: TableHandle) -> Option<usize> {
        let identity = table.native_identity();
        let mut place = mix_identity(identity) & (LANE_INDEX_PLACES - 1);
        loop {
            let entry = &self.index.entries[place];
            if entry.stamp >> 1 != self.stamp {
                return None;
            }
            if entry.identity == identity {
                return matches!(self.bank().get(entry.lane as usize), Some(Value::Table(held)) if *held == table)
                    .then_some(place);
            }
            place = (place + 1) & (LANE_INDEX_PLACES - 1);
        }
    }

    /// The lane of `table` among the lanes the bank holds, from the index, or
    /// from the search of the bank while the run has made tables (they are
    /// not in the index) or the index does not hold the bank.
    fn known_lane(&mut self, table: TableHandle) -> Option<u32> {
        if self.index_ok && self.host.made.is_empty() {
            return self.index_find(table);
        }
        self.find_table_lane(table.native_identity())
    }

    /// The lane the index names for the table `table`, if the bank still
    /// holds it there.
    #[inline]
    fn index_find(&self, table: TableHandle) -> Option<u32> {
        self.index_place(table)
            .map(|place| self.index.entries[place].lane)
    }

    /// Tells native code that the table at `place` has its lane and its row.
    #[inline]
    fn index_ready(&mut self, place: usize) {
        self.index.entries[place].stamp |= 1;
    }

    /// The lane of the table with `identity` among the lanes the bank holds,
    /// found by a scan while the bank is small and through the map after.
    fn find_table_lane(&mut self, identity: u64) -> Option<u32> {
        if !self.lanes_ready {
            if self.bank().len() <= SCAN_LANES {
                let control = self.control.as_ref();
                // The last lane wins, as it does in the map.
                return self
                    .bank()
                    .iter()
                    .enumerate()
                    .rev()
                    .find_map(|(lane, held)| match held {
                        Value::Table(table)
                            if table.native_identity() == identity
                                && !control.is_some_and(|lanes| lanes.contains(&lane)) =>
                        {
                            u32::try_from(lane).ok()
                        }
                        _ => None,
                    });
            }
            self.build_lanes();
        }
        self.lanes.get(&identity).copied()
    }

    /// Fills the lane map from the bank as it stands.
    fn build_lanes(&mut self) {
        self.lanes.clear();
        let control = self.control.as_ref();
        let made = &self.host.made;
        // SAFETY: the run's bank, valid for the run.
        #[allow(unsafe_code)]
        let held_by_lane = unsafe { &*self.bank };
        for (lane, held) in held_by_lane.iter().enumerate() {
            // The control lane of a `next` loop holds a key, which changes:
            // it names no table. Nor does the lane of a value it keeps. A
            // table the run made is found through `made`: its lane is
            // released and handed out again, so a map entry would go stale.
            if let (Value::Table(table), Ok(lane), false) = (
                held,
                u32::try_from(lane),
                control.is_some_and(|lanes| lanes.contains(&lane))
                    || u32::try_from(lane).is_ok_and(|lane| made.contains(&lane)),
            ) {
                self.lanes.insert(table.native_identity(), lane);
            }
        }
        self.lanes_ready = true;
    }

    /// The run's root bank and the trace's root strings, to read, for a
    /// helper that must not touch the runtime meanwhile. Both are slices of
    /// heap storage, not of the VM's own bytes.
    pub(super) fn lanes(&self) -> (&[Value], &[Value]) {
        (self.bank().as_slice(), self.roots)
    }

    /// The parts of the run a library helper works on: the runtime, the root
    /// bank, the trace's root strings, what the run's string making
    /// remembers and the run's string lengths.
    #[allow(unsafe_code)]
    pub(super) fn library(
        &mut self,
    ) -> (
        &mut Runtime,
        &mut Vec<Value>,
        &[Value],
        &mut HostState,
        &mut [u32],
    ) {
        let lengths: &mut [u32] = if self.lengths.0.is_null() {
            &mut []
        } else {
            // SAFETY: the run's length table, set by `string_lengths`, lives
            // for the run and only this context writes it during a call.
            unsafe { std::slice::from_raw_parts_mut(self.lengths.0, self.lengths.1) }
        };
        // SAFETY: the pointers the run was lent, as in `Self::runtime`;
        // the other parts are fields of `self`, which is borrowed for them.
        unsafe {
            (
                &mut *self.runtime,
                &mut *self.bank,
                self.roots,
                &mut self.host,
                lengths,
            )
        }
    }

    /// The parts of the run a remembered read works on: the runtime, the
    /// root bank, the memo of reads and the run's stamp, which tells the
    /// memo's entries of this run from those of earlier ones.
    pub(super) fn read_memo(&mut self) -> (&Runtime, &[Value], &mut super::helpers::ReadMemo, u32) {
        // SAFETY: the pointers the run was lent, as in `Self::runtime`.
        #[allow(unsafe_code)]
        let (runtime, bank) = unsafe { (&*self.runtime, &*self.bank) };
        (runtime, bank.as_slice(), &mut self.index.reads, self.stamp)
    }

    /// The most lanes the run's bank holds for tables.
    const fn lane_cap(&self) -> usize {
        if self.fresh_lanes {
            MAX_FUNCTION_TABLE_LANES
        } else {
            MAX_TRACE_TABLE_LANES
        }
    }

    /// Gives the table `handle`, which the run just made, a lane of the bank
    /// and weighs it towards the next release. The lane is a vacated one or a
    /// new one within the lanes a table may have (a lane past them has no
    /// row, so no field of the table could be read natively). `None` when
    /// the run has no lane to spare.
    pub(super) fn made_table_lane(&mut self, handle: TableHandle) -> Option<usize> {
        let lane = self.reserve_made_lane()?;
        self.fill_made_lane(lane, handle)
    }

    /// The first step of [`Self::made_table_lane`]: a lane of the bank that
    /// holds nil, for a table the run is about to make, so that a run with
    /// no lane to spare makes no table. [`Self::fill_made_lane`] gives it
    /// the table, or [`Self::return_made_lane`] gives it back.
    #[inline]
    pub(super) fn reserve_made_lane(&mut self) -> Option<usize> {
        let bank = self.bank();
        while let Some(vacated) = self.host.vacated.pop() {
            // A string may have taken the lane since.
            if matches!(bank.get(vacated as usize), Some(Value::Nil)) {
                return Some(vacated as usize);
            }
        }
        if bank.len() < self.lane_cap() {
            bank.push(Value::Nil);
            return Some(bank.len() - 1);
        }
        None
    }

    /// Gives back a lane [`Self::reserve_made_lane`] reserved, which still
    /// holds nil: the next table the run makes takes it.
    pub(super) fn return_made_lane(&mut self, lane: usize) {
        if let Ok(lane) = u32::try_from(lane) {
            self.host.vacated.push(lane);
        }
    }

    /// The second step of [`Self::made_table_lane`]: puts `handle` in
    /// `lane`, which [`Self::reserve_made_lane`] reserved, and weighs the
    /// table towards the next release.
    #[allow(unsafe_code)]
    #[inline]
    pub(super) fn fill_made_lane(&mut self, lane: usize, handle: TableHandle) -> Option<usize> {
        self.bank()[lane] = Value::Table(handle);
        if self.fresh_lanes {
            // Nothing releases the tables of a function trace's run: they
            // stay in the bank until it ends.
            return Some(lane);
        }
        self.host.made.push(u32::try_from(lane).ok()?);
        if !self.lengths.0.is_null() && self.lengths.1 > super::host::MAX_HOST_BANK_LANES {
            // SAFETY: the run's length table, set by `string_lengths`, lives
            // for the run and only this context writes it during a call.
            let pending = unsafe { &mut *self.lengths.0.add(super::host::MAX_HOST_BANK_LANES) };
            *pending = pending.saturating_add(super::host::MADE_TABLE_WEIGHT);
        }
        Some(lane)
    }

    /// The lane of a table a host call gave: the one it has already (one
    /// object has one lane), or else a lane of the tables the run made,
    /// which the next release drops with them. The calls of a loop give a
    /// table each, and their rows would otherwise all be made again after
    /// every call. `None` when the run has no lane to spare.
    pub(super) fn result_table_lane(&mut self, table: TableHandle) -> Option<usize> {
        if !self.fresh_lanes
            && let Some(lane) = self.known_lane(table)
        {
            return Some(lane as usize);
        }
        self.made_table_lane(table)
    }

    /// Takes `lane`, which a lookup found for the table with `identity`, out
    /// of the lanes the next release drops when the run made the table. The
    /// lookup came from a read of the table through a field, a cell or an
    /// argument, so something other than the register that got the table
    /// from its call may hold it, and a lane released and handed out again
    /// would name another table there. The lane stays with the table until
    /// the run ends, and the lookups find it as they find any other lane.
    fn keep_made_lane(&mut self, lane: u32, identity: u64) {
        let Some(at) = self.host.made.iter().position(|made| *made == lane) else {
            return;
        };
        self.host.made.swap_remove(at);
        if self.index_ok {
            let _ = self.index_insert(identity, lane as usize);
        }
        if self.lanes_ready {
            self.lanes.insert(identity, lane);
        }
    }

    /// Drops the lanes of the tables the run made since the last release:
    /// the bank no longer holds them, the lane search forgets them, and a
    /// lane gets a row of its own for the next table it holds. A table the
    /// trace stored somewhere is held by what holds it.
    #[allow(unsafe_code)]
    pub(super) fn release_made_tables(&mut self) {
        let mut made = std::mem::take(&mut self.host.made);
        for lane in made.drain(..) {
            let index = lane as usize;
            if let Some(Value::Table(table)) = self.bank().get(index) {
                let _ = table;
                self.bank()[index] = Value::Nil;
                if index < self.rows.1 {
                    // SAFETY: the run's row flags, set by `row_flags`, live
                    // for the run and only this context writes them during
                    // a call.
                    unsafe { *self.rows.0.add(index) = false };
                }
                self.host.vacated.push(lane);
            }
        }
        self.host.made = made;
    }

    /// Names the lane that holds the address of the hash part's snapshot
    /// description (see `TraceIr::boxed_steps`).
    fn set_cursor_lane(&mut self, lane: Option<usize>) {
        self.cursor_lane = lane;
    }

    /// After a step of the boxed loop committed a key to the control lane:
    /// describes the snapshot of the hash part `table` keeps, when `hash` says
    /// the step came from it and the table has one, so that native code steps
    /// along it from the key the lane holds. Otherwise native code leaves the
    /// hash part to the helper.
    pub(super) fn sync_cursor(&mut self, table: TableHandle, hash: bool) {
        let Some(lane) = self.cursor_lane else {
            return;
        };
        let view = if hash {
            self.runtime().table_native_cursor(
                table,
                &self.bank()[lane - 2],
                &mut self.cursor_scratch,
            )
        } else {
            None
        };
        match view {
            Some(mut view) => {
                view.in_sync = 1;
                self.cursor = view;
                // The position is the run's own cell: another step along the
                // table (`next` in the loop body) moves the table's.
                self.cursor.position = std::ptr::from_mut(&mut self.cursor.position_value) as usize;
            }
            None => self.cursor.in_sync = 0,
        }
    }

    /// The control lane is written another way than by a step: native code
    /// does not step along the hash part until a helper step says it can.
    pub(super) fn desync_cursor(&mut self) {
        self.cursor.in_sync = 0;
    }

    /// Puts the key of the entry the last native step moved to in the control
    /// lane (the lane is two before the lane that holds the description of
    /// the snapshot). Native steps along the hash part leave it to this, so a
    /// step costs no count on a string, and it runs before anything reads the
    /// lane: a helper, and the end of the run.
    #[allow(unsafe_code)]
    pub(super) fn flush_cursor(&mut self) {
        let pending = std::mem::take(&mut self.cursor.pending);
        let Some(lane) = self.cursor_lane.filter(|_| pending != 0) else {
            return;
        };
        // SAFETY: `pending` is the address of a slot of the snapshot that was
        // valid when native code stepped to it, and nothing in the trace
        // moves hash storage since (see `BoxStep`).
        let key = unsafe { self.runtime().native_cursor_key(pending) };
        if let Some(held) = self.bank().get_mut(lane - 2) {
            *held = key;
        }
    }

    /// The control lane is overwritten, so the key native code stepped to is
    /// no longer the control value.
    pub(super) fn drop_pending_cursor(&mut self) {
        self.cursor.pending = 0;
    }

    /// Names the array bindings of the run.
    fn set_arrays(&mut self, arrays: &'a [bool]) {
        self.arrays = arrays;
    }

    /// Whether a table can still get a lane in the run's bank.
    pub(super) fn has_lane_room(&self) -> bool {
        self.has_lane_room_for(1)
    }

    /// Whether `count` tables can still get lanes in the run's bank.
    pub(super) fn has_lane_room_for(&self, count: usize) -> bool {
        self.bank().len() + count <= MAX_TRACE_TABLE_LANES
    }

    /// Names the kind of each binding of the run.
    fn set_kinds(&mut self, kinds: &'a [crate::jit_ir::trace::TableBinding]) {
        self.kinds = kinds;
        self.has_lengths = kinds.iter().any(|binding| {
            matches!(
                binding.kind,
                crate::jit_ir::trace::BindingKind::Length { .. }
                    | crate::jit_ir::trace::BindingKind::Header { .. }
            )
        });
    }

    /// After a write that may have moved the array part of `table` (now
    /// `view`) or changed its border: rewrites the words of every array and
    /// length binding that resolved it. `words` is the run's binding slots,
    /// one per binding.
    #[allow(unsafe_code)]
    pub(super) fn rewrite_bindings(
        &mut self,
        words: *mut TraceInputSlot,
        table: TableHandle,
        view: crate::runtime::NativeValueArray,
    ) {
        use crate::jit_ir::trace::BindingKind;
        let count = self.kinds.len().min(self.plans.len());
        if words.is_null() || count == 0 {
            return;
        }
        // SAFETY: `words` is the run's binding slots, one per binding of the
        // trace (see `run_table_set`), which nothing else touches during the
        // call.
        let words = unsafe { std::slice::from_raw_parts_mut(words, count) };
        for (index, word) in words.iter_mut().enumerate() {
            if matches!(self.kinds[index].kind, BindingKind::Array { .. })
                && self.bound_table(index) == Some(table)
            {
                word.set_array(view);
            }
        }
        if self.has_lengths {
            self.rewrite_length_words(words, table);
        }
    }

    /// Whether the field binding `index` names (found `depth` `__index` hops
    /// from its table) still holds a value `holds` accepts.
    ///
    /// A field of the binding's own table (`depth` zero) is looked up once.
    /// The slot it was found in is the key's while
    /// [`crate::runtime::table_key_layout`] stands, for a table that is
    /// ordered or hashed, so the checks after that read the slot. The bank
    /// holds the table for the run, so its storage is not freed meanwhile.
    /// A slot that no longer holds an accepted value says nothing (a hashed
    /// table deletes in place): the lookup gives the answer then.
    ///
    /// A shaped table is looked up every time. Its slots stand while its
    /// shape and its block do, but asking the table for the two costs more
    /// than the lookup saved: the same host call of Mineclonia's startup
    /// took 105 instructions more with shaped tables remembered than
    /// without.
    #[allow(unsafe_code)]
    fn field_holds(
        &mut self,
        index: usize,
        depth: u8,
        holds: impl Fn(&Value, &[Value]) -> bool,
    ) -> bool {
        let Some(table) = self.bound_table(index) else {
            return false;
        };
        let layout = crate::runtime::table_key_layout();
        if depth == 0
            && let Some(slot) = self.field_slots.get(index, table, layout)
            // SAFETY: the slot of a key of `table`'s ordered or hashed
            // entries, which have not moved since it was found (the layout
            // count stands), in a table the bank keeps alive.
            && holds(unsafe { &*slot }, self.bank())
        {
            #[cfg(debug_assertions)]
            {
                let Some(Value::String(key)) =
                    self.plans.get(index).and_then(|plan| plan.key.as_ref())
                else {
                    unreachable!("a remembered binding has a string key")
                };
                debug_assert!(
                    matches!(
                        self.runtime().table_string_held(table, key, 0),
                        Some((_, 0, held)) if std::ptr::eq(held, slot)
                    ),
                    "a remembered slot is the one a lookup finds"
                );
            }
            return true;
        }
        let Some(Value::String(key)) = self.plans.get(index).and_then(|plan| plan.key.as_ref())
        else {
            return false;
        };
        match self.runtime().table_string_held(table, key, depth) {
            Some((_, hops, held)) if hops == depth => {
                let accepted = holds(held, self.bank());
                if accepted && depth == 0 && self.runtime().table_keeps_key_slots(table) {
                    self.field_slots.set(FieldSlot {
                        binding: index,
                        table,
                        layout,
                        slot: std::ptr::from_ref(held),
                    });
                }
                accepted
            }
            _ => false,
        }
    }

    /// Whether every cell, global and captured local the trace reads still
    /// holds what the run started with, after a call that ran arbitrary Lua
    /// code. `frames` is the stack the trace runs on. `cells_written` is
    /// false when no cell was written during the call
    /// ([`Runtime::cell_writes`] stood still): the upvalues in cells closed
    /// at entry then hold without a look.
    pub(super) fn externals_hold(
        &mut self,
        frames: &mut CallStack,
        written: &[Option<TraceInputSlot>],
        cells_written: bool,
    ) -> bool {
        let Some(host) = self.resume.as_ref().and_then(|state| state.host.as_ref()) else {
            return true;
        };
        let writes = !host.written.is_empty() || !host.captured_written.is_empty();
        let closed = if cells_written {
            host.closed_cells.as_slice()
        } else {
            &[]
        };
        let quiet = host.checks.is_empty() && host.captured.is_empty() && closed.is_empty();
        // What the trace writes holds what memory held when the call began,
        // the trace's own values.
        if writes && self.written_externals(frames) != written {
            return false;
        }
        // A call that replaced a closure's cell may have parted two names the
        // trace holds one variable for, or joined two it holds apart.
        if cells_written && !self.cells_stand(frames) {
            return false;
        }
        if quiet {
            return true;
        }
        let Some(host) = self.resume.as_ref().and_then(|state| state.host.as_ref()) else {
            return true;
        };
        // A global is read straight from its table and compared with what the
        // run entered with, which is most of what a trace reads from outside.
        let global = |constant: &u32| {
            let frame = &frames[host.frame_index];
            let Some(Constant::String(name)) =
                frame.chunk.constants.get(usize::try_from(*constant).ok()?)
            else {
                return None;
            };
            Some(self.runtime().table_get_string_ref(frame.environment, name))
        };
        let mut others = false;
        for (input, slot) in &host.checks {
            let ExternalSlot::Global(constant) = slot else {
                others = true;
                continue;
            };
            let held = self.inputs.get(*input);
            let holds = match global(constant) {
                Some(Some(value)) => {
                    held.is_some_and(|held| input_slot_holds(value, held, self.bank()))
                }
                // No constant or no present raw value: no input type accepts it.
                _ => held == Some(&TraceInputSlot::other()),
            };
            if !holds {
                return false;
            }
        }
        let closed = if cells_written {
            host.closed_cells.as_slice()
        } else {
            &[]
        };
        if !others && host.captured.is_empty() && closed.is_empty() {
            return true;
        }
        // Each value is compared with the lane its input names: no search of
        // the bank, which holds every object the run has met.
        let frame = frames.at_mut(host.frame_index);
        let mut holders = GlobalHolders::default();
        let bank: &[Value] = self.bank();
        host.checks
            .iter()
            .filter(|(_, slot)| !matches!(slot, ExternalSlot::Global(_)))
            .chain(closed)
            .all(|(input, slot)| {
                self.inputs.get(*input).is_some_and(|held| {
                    external_holds(&frame, self.runtime(), slot, bank, &mut holders, held)
                })
            })
            && host.captured.iter().all(|(input, register)| {
                self.inputs
                    .get(*input)
                    .is_some_and(|held| match frame.registers.get(*register) {
                        Some(value) => input_slot_holds(value, held, bank),
                        None => *held == TraceInputSlot::other(),
                    })
            })
    }

    /// Whether the cells of the trace's names still stand as entry found
    /// them: each alias names one cell, and the upvalues it holds apart name
    /// distinct cells, none a captured local's.
    fn cells_stand(&self, frames: &mut CallStack) -> bool {
        let Some(host) = self.resume.as_ref().and_then(|state| state.host.as_ref()) else {
            return true;
        };
        if host.aliases.is_empty() && host.upvalue_slots.is_empty() {
            return true;
        }
        let captured: Vec<crate::vm::stack::OpenUpvalue> = {
            let record = &frames[host.frame_index];
            frames.open_in(record.base, record.top).to_vec()
        };
        let frame = frames.at_mut(host.frame_index);
        let (runtime, bank) = (self.runtime(), self.bank());
        if host.cells_seen.holds(&frame, &captured, runtime) {
            return true;
        }
        aliases_hold(&frame, &captured, runtime, bank, &host.aliases)
            && !((host.many_upvalues || !captured.is_empty())
                && upvalues_alias(&frame, &captured, runtime, bank, &host.upvalue_slots))
    }

    /// The cells and globals the trace writes, as memory holds them now.
    pub(super) fn written_externals(
        &mut self,
        frames: &mut CallStack,
    ) -> Vec<Option<TraceInputSlot>> {
        let Some(host) = self.resume.as_ref().and_then(|state| state.host.as_ref()) else {
            return Vec::new();
        };
        if host.written.is_empty() && host.captured_written.is_empty() {
            return Vec::new();
        }
        let frame = frames.at_mut(host.frame_index);
        let mut holders = GlobalHolders::default();
        let mut slots: Vec<Option<TraceInputSlot>> = host
            .written
            .iter()
            .map(|slot| {
                external_slot(
                    &frame,
                    self.runtime(),
                    slot,
                    self.bank(),
                    (&mut holders, None),
                    None,
                )
            })
            .collect();
        slots.extend(host.captured_written.iter().map(|register| {
            frame
                .registers
                .get(*register)
                .map(|held| input_slot(held, self.bank()))
        }));
        slots
    }

    /// Writes the state of exit `exit` back to the frame the run is on, as
    /// the exit would, from the exit area at `area` that native code filled
    /// with the exit's values. The run goes on: the lanes stay in the bank.
    /// Returns whether the plan was applied; an exit that pushes frames,
    /// rewrites the caller or a tail-called root is not (the recorder makes
    /// no call there).
    #[allow(unsafe_code)]
    pub(super) fn restore_exit(
        &mut self,
        exit: usize,
        area: *const i64,
        frames: &mut CallStack,
    ) -> Option<(usize, usize)> {
        let host = self.resume.as_ref().and_then(|state| state.host.as_ref())?;
        let (plans, area_len, frame_index) = (host.restores, host.area_len, host.frame_index);
        // SAFETY: the plans of the running trace's cache entry, valid for
        // the run.
        let plans = unsafe { std::slice::from_raw_parts(plans.0, plans.1) };
        let restore = plans.get(exit)?;
        // The root frame's caller is the interpreter's own frame, which an
        // exit that returned into it rewrites: the call is never made there.
        if restore.has_root_tail()
            || restore.writes_caller()
            || restore
                .inner
                .iter()
                .any(|plan| matches!(plan.entry, FrameEntry::Caller { .. }))
        {
            return None;
        }
        // SAFETY: the run's exit area, `area_len` words, which native code
        // wrote the exit's values into just before this call.
        let area = unsafe { std::slice::from_raw_parts(area, area_len) };
        let (bank, runtime) = (self.bank(), self.runtime());
        let tables = restore.materialize(area, bank, self.roots, runtime).ok()?;
        let mut frame = frames.at_mut(frame_index);
        restore.apply(&mut frame, area, bank, self.roots, &tables, runtime);
        restore.apply_function_count(&mut frame, area, None);
        // The inlined frames go on top of the root, as an exit leaves them,
        // so the host function sees the interpreter's stack. The innermost
        // one stands at the call; the caller takes them off again.
        restore.push_inner(frames, area, bank, self.roots, &tables, runtime);
        let pc = if restore.inner.is_empty() {
            restore.pc
        } else {
            restore.resume_pc
        };
        Some((pc, restore.inner.len()))
    }

    /// Makes the row of every lane that has one again, after a call that ran
    /// arbitrary Lua code, which may have moved the storage of any table,
    /// retyped a field or given a table another metatable. Native code reads
    /// a field through the slot address its row holds, and leaves through the
    /// guard when the row names none.
    ///
    /// A row is checked rather than filled where it can be
    /// ([`TraceTables::row_holds`]), which spares the key lookups of a fill:
    /// a row that names many fields of a big table cost a search per field
    /// each call. Most calls leave the keys of every table where they were,
    /// and the check then reads the slots' types. A call that moved some
    /// table's keys (it made a table, say) leaves the rows of the other
    /// tables standing too, each slot checked against its table's entries.
    #[allow(unsafe_code)]
    pub(super) fn refill_rows(&mut self) {
        if self.space.is_null() {
            return;
        }
        // SAFETY: `space` is the run's row storage (`row_space`), which lives
        // for the run; native code is stopped in a helper call and the
        // dispatcher holds no reference into it across the call.
        let space = unsafe { &mut *self.space };
        let lanes = self.bank().len();
        self.layout_moved = crate::runtime::table_key_layout() != self.layout_seen;
        space.refill(lanes, self);
        // Every row is now one of the tables as they stand.
        self.layout_seen = crate::runtime::table_key_layout();
    }

    /// Fills the row of `lane`, a table the run met after it began (the
    /// result of a host call), unless it is filled already.
    #[allow(unsafe_code)]
    pub(super) fn fill_row(&mut self, lane: usize) {
        if self.space.is_null() {
            return;
        }
        // SAFETY: as in `refill_rows`.
        let space = unsafe { &mut *self.space };
        space.ensure_row(lane, self);
    }

    /// [`Self::refill_rows`] for a host call, which leaves the rows of the
    /// tables the run made and the exit area (`area`, the call's `Before`
    /// snapshot) does not name unfilled, to be filled when the trace next
    /// asks for them. The trace holds a table it made only in the registers
    /// that snapshot describes, and the tables the calls of a loop gave are
    /// mostly dead by the next call: filling all of their rows each time is
    /// the cost of the call over again.
    #[allow(unsafe_code)]
    pub(super) fn refill_rows_in_use(&mut self, area: *const i64) {
        if self.space.is_null() {
            return;
        }
        // A run that made no table has no row to leave.
        if self.host.made.is_empty() {
            self.refill_rows();
            return;
        }
        let area_len = self
            .resume
            .as_ref()
            .and_then(|state| state.host.as_ref())
            .map_or(0, |host| host.area_len);
        // SAFETY: the run's exit area, `area_len` words, which native code
        // wrote the call's values into just before the call.
        let area = unsafe { std::slice::from_raw_parts(area, area_len) };
        let mut named = [0_u64; MAX_TRACE_TABLE_LANES / 64];
        for word in area {
            if let Ok(lane) = usize::try_from(*word)
                && lane < MAX_TRACE_TABLE_LANES
            {
                named[lane / 64] |= 1 << (lane % 64);
            }
        }
        for lane in &self.host.made {
            let lane = *lane as usize;
            if lane < self.rows.1
                && named
                    .get(lane / 64)
                    .is_some_and(|bits| bits >> (lane % 64) & 1 == 0)
            {
                // SAFETY: the run's row flags, set by `row_flags`, live for
                // the run and only this context writes them during a call.
                unsafe { *self.rows.0.add(lane) = false };
            }
        }
        self.refill_rows();
    }

    /// Rewrites the words of every array, length and header binding of the
    /// run from the tables as they stand, after a call that ran arbitrary Lua
    /// code (`HelperEffects::RESUMES`), which may have grown, shrunk or
    /// replaced the storage of any table, and checks the facts a binding
    /// fixed at entry (the callable or the string a field holds). `words` is
    /// the run's binding slots, one per binding. Returns whether every binding
    /// still holds; the run then must not go on.
    #[allow(unsafe_code)]
    pub(super) fn refresh_bindings(&mut self, words: *mut TraceInputSlot) -> bool {
        use crate::jit_ir::trace::BindingKind;
        let count = self.kinds.len().min(self.plans.len());
        if words.is_null() || count == 0 {
            return true;
        }
        // SAFETY: as in `rewrite_bindings`.
        let words = unsafe { std::slice::from_raw_parts_mut(words, count) };
        let mut counted: Vec<TableHandle> = Vec::new();
        let kinds = self.kinds;
        for index in 0..count {
            match kinds[index].kind {
                BindingKind::Array { .. } => {
                    let Some(table) = self.bound_table(index) else {
                        return false;
                    };
                    let Ok(view) = self.runtime().table_trace_array(table) else {
                        return false;
                    };
                    words[index].set_array(view);
                }
                BindingKind::Length { raw, .. } => {
                    let Some(table) = self.bound_table(index) else {
                        return false;
                    };
                    // The entry refused a table with a `__len` handler; the
                    // thread may have given it one.
                    if !raw
                        && !matches!(
                            self.runtime().table_length_without_len_metamethod(table),
                            Ok(Some(_))
                        )
                    {
                        return false;
                    }
                    if !counted.contains(&table) {
                        counted.push(table);
                    }
                }
                BindingKind::Header { .. } => {
                    let Some(table) = self.bound_table(index) else {
                        return false;
                    };
                    if !counted.contains(&table) {
                        counted.push(table);
                    }
                }
                // The slot a field binding names moves when the table's
                // storage does, and may now hold a value of another type.
                BindingKind::Field { ty, depth, .. } => {
                    let Some(table) = self.bound_table(index) else {
                        return false;
                    };
                    let Some(key) = self.plans.get(index).and_then(|plan| plan.key.clone()) else {
                        return false;
                    };
                    let Value::String(name) = &key else {
                        return false;
                    };
                    let holder = if depth == 0 {
                        table
                    } else {
                        match self.runtime().table_string_holder(table, name, depth) {
                            Some((holder, hops)) if hops == depth => holder,
                            _ => return false,
                        }
                    };
                    match self
                        .runtime()
                        .table_native_value_slot(holder, &key, slot_kind(ty))
                    {
                        Ok(Some(slot)) => words[index] = TraceBinding::field(slot).word(),
                        _ => return false,
                    }
                }
                // The string index of a hash part is not valid after code ran
                // that may have moved it: the lookups go to the VM, which
                // finds the key where it is now. A table that had no
                // metatable has none still, or a nil read could run `__index`.
                BindingKind::Hash | BindingKind::NoMetatable => {
                    if matches!(self.kinds[index].kind, BindingKind::NoMetatable) {
                        let Some(table) = self.bound_table(index) else {
                            return false;
                        };
                        if !matches!(self.runtime().raw_metatable(&Value::Table(table)), Ok(None)) {
                            return false;
                        }
                    }
                    words[index] = TraceBinding::callee().word();
                }
                // What the metatable of the class holds is what the guards
                // of the trace rely on.
                BindingKind::MetaClass { .. } => {
                    let holds = self
                        .plans
                        .get(index)
                        .and_then(|plan| plan.class.as_ref())
                        .is_some_and(|class| class.holds(self.runtime()));
                    if !holds {
                        return false;
                    }
                }
                BindingKind::Callee { depth, lane, .. } => {
                    if !self.field_holds(index, depth, |held, bank| {
                        usize::try_from(lane)
                            .ok()
                            .and_then(|lane| bank.get(lane))
                            .is_some_and(|pinned| same_object(held, pinned))
                    }) {
                        return false;
                    }
                }
                BindingKind::StringField {
                    depth, ref text, ..
                } if !self.field_holds(
                    index,
                    depth,
                    |held, _| matches!(held, Value::String(held) if held.as_bytes() == &text[..]),
                ) =>
                {
                    return false;
                }
                _ => {}
            }
        }
        for table in counted {
            self.rewrite_length_words(words, table);
        }
        true
    }

    /// Rewrites the word of every length binding that resolved `table`, whose
    /// border a write may have changed, and of every header binding of it,
    /// whose answer to "may native code append in place" a write may have
    /// changed. Callers ask `has_lengths` first.
    #[inline(never)]
    fn rewrite_length_words(&mut self, words: &mut [TraceInputSlot], table: TableHandle) {
        use crate::jit_ir::trace::BindingKind;
        let mut border = None;
        let mut header = None;
        for (index, word) in words.iter_mut().enumerate().take(self.kinds.len()) {
            match &self.kinds[index].kind {
                BindingKind::Length { ty, .. } => {
                    if self.bound_table(index) != Some(table) {
                        continue;
                    }
                    let length = *border
                        .get_or_insert_with(|| self.runtime().table_length(table).unwrap_or(0.0));
                    *word = if *ty == TraceType::Float {
                        TraceInputSlot::float(length)
                    } else {
                        // A border is a whole number below 2^53.
                        #[allow(clippy::cast_possible_truncation)]
                        TraceInputSlot::integer(length as i64)
                    };
                }
                BindingKind::Header { .. } => {
                    if self.bound_table(index) != Some(table) {
                        continue;
                    }
                    let found = *header
                        .get_or_insert_with(|| self.runtime().table_native_header(table).ok());
                    if let Some(found) = found {
                        #[allow(clippy::cast_possible_wrap)]
                        let address = found.address as i64;
                        *word = TraceInputSlot::from_words(found.word(), address);
                    }
                }
                _ => {}
            }
        }
    }

    /// Lets the run's host calls read the trace's root strings.
    fn set_roots(&mut self, roots: &'a [Value]) {
        self.roots = roots;
    }

    /// Gives the run the trace's memo of rows and the types its fields hold.
    fn set_row_memo(&mut self, memo: (*mut usize, usize), tags: &'a [u8]) {
        self.memo = memo;
        self.tags = tags;
    }

    /// Lets the run set keys of its bound tables.
    fn set_bindings(&mut self, plans: &'a [BindingPlan], inputs: &'a [TraceInputSlot]) {
        self.plans = plans;
        self.inputs = inputs;
        // Decided once per run: most traces pin no key, and a store then has
        // no plan to compare its key with.
        self.pins_keys = plans.iter().any(|plan| plan.pins);
    }

    /// The keys of the fields whose values a binding of the run fixes for
    /// the whole run (a function or string the trace reads as a constant),
    /// and the tables that hold the globals the run read at entry. A store
    /// under such a key, or into such a table, may replace a value the run
    /// holds, in the bound table or in a table its `__index` chain reaches,
    /// so the store must miss.
    pub(super) const fn pinned_keys(&self) -> PinnedKeys<'a> {
        PinnedKeys {
            plans: self.plans,
            holders: self.holders,
            pins_keys: self.pins_keys,
        }
    }

    /// Tells the run's stores which tables hold the globals it read.
    const fn set_global_holders(&mut self, holders: GlobalHolders) {
        self.holders = holders;
    }

    /// The table binding `binding` resolved at entry: the one in its entry
    /// input, which the bank holds for the run.
    fn bound_table(&self, binding: usize) -> Option<TableHandle> {
        let slot = self.inputs.get(self.plans.get(binding)?.input)?;
        if slot.tag() != crate::jit::TRACE_TAG_TABLE {
            return None;
        }
        match self.bank().get(usize::try_from(slot.bits()).ok()?)? {
            Value::Table(table) => Some(*table),
            _ => None,
        }
    }
}

/// The shapes whose rows a trace keeps from run to run ([`RunTables::fill_table_row`]),
/// a power of two.
const ROW_MEMO_WAYS: usize = 4;
/// The words in front of the row in a place of the memo: the id of the shape,
/// the table's metatable (zero for none, else the identity plus one), whether
/// it is watched, and the row's metatable word.
const ROW_MEMO_HEADER: usize = 4;

/// Makes `row` from the slots of a table of a shape whose offsets are
/// `offsets`, unless a slot does not hold the type its field wants (see
/// [`RunTables::fill_table_row`]). The word of field `words.meta` (the
/// metatable word) is `meta_word`, and the word of field `words.address` is
/// `address`, the address of this table: neither is a slot, and the address
/// is written fresh, never rebased, since tables of one shape share offsets
/// but not addresses.
///
/// The offsets and the address of the slots must be those of a table of one
/// shape, which the caller has checked: then each slot is one of the table's
/// values.
#[allow(unsafe_code)]
fn rebase_row(
    row: &mut [usize],
    offsets: &[usize],
    (slots, tags): (usize, &[u8]),
    words: crate::runtime::NativeRowWords,
    (meta_word, address): (usize, usize),
) -> bool {
    let meta = words.meta.unwrap_or(usize::MAX);
    let address_field = words.address.unwrap_or(usize::MAX);
    let count = row.len();
    if offsets.len() < count || tags.len() < count {
        return false;
    }
    // Slices of the one length, so that the indexing below needs no checks.
    let (offsets, tags) = (&offsets[..count], &tags[..count]);
    for field in 0..count {
        if field == meta {
            row[field] = meta_word;
            continue;
        }
        if field == address_field {
            row[field] = address;
            continue;
        }
        let slot = slots.wrapping_add(offsets[field]);
        // SAFETY: as above, and the first byte of a value is its type.
        if unsafe { *(slot as *const u8) } != tags[field] {
            return false;
        }
        row[field] = slot;
    }
    true
}

/// What the memo keeps of a table's metatable: its identity plus one, so that
/// zero is none.
#[inline]
fn row_meta_word(metatable: Option<TableHandle>) -> usize {
    // A 64-bit word: the identity fits `usize` on the targets that compile
    // native code.
    #[allow(clippy::cast_possible_truncation)]
    metatable.map_or(0, |metatable| {
        metatable.native_identity().wrapping_add(1) as usize
    })
}

/// The memo of a trace with `fields` row words: [`ROW_MEMO_WAYS`] places.
fn new_row_memo(fields: usize) -> Vec<usize> {
    vec![0; ROW_MEMO_WAYS * (ROW_MEMO_HEADER + fields)]
}

impl RunTables<'_> {
    /// Fills `row`, the fields' words of the row of `table`, as
    /// [`Runtime::table_native_row`] does, and zeroes it for a table the
    /// runtime cannot serve.
    ///
    /// Tables of one shape keep each field at one offset from their slots, so
    /// the trace keeps the offsets of the shapes it met last (and what else
    /// the row depends on: the metatable and whether the table is watched).
    /// A table of such a shape has its row made from its slots by adding the
    /// offsets, once the slots are checked for the types the fields must hold.
    /// A loop over records meets many tables of one shape, and this skips the
    /// lookup of every key for each.
    fn fill_table_row(&mut self, table: TableHandle, row: &mut [usize]) {
        let stride = ROW_MEMO_HEADER + self.fields.len();
        // A function trace meets a table once per call, mostly of a few
        // shapes with different keys (the nodes of a tree), whose rows the
        // memo cannot keep: it would cost a token and a miss for each.
        let kept = !self.fresh_lanes
            && self.memo.1 >= ROW_MEMO_WAYS * stride
            && row.len() == self.fields.len();
        let token = if kept {
            self.runtime().table_native_row_token(table)
        } else {
            None
        };
        let Some(token) = token else {
            self.fill_row_missed(table, row, None);
            return;
        };
        let place = ((token.shape.wrapping_mul(0x9E37_79B1) >> 8) & (ROW_MEMO_WAYS - 1)) * stride;
        // SAFETY: `memo` is the trace's memo, `ROW_MEMO_WAYS * stride` words
        // that only runs of this trace touch, one at a time.
        #[allow(unsafe_code)]
        let way = unsafe { std::slice::from_raw_parts_mut(self.memo.0.add(place), stride) };
        let (header, offsets) = way.split_at_mut(ROW_MEMO_HEADER);
        if header[0] == token.shape
            && header[1] == row_meta_word(token.metatable)
            && header[2] == usize::from(token.watched)
            && rebase_row(
                row,
                offsets,
                (token.slots, self.tags),
                self.meta_field,
                (header[3], token.address),
            )
        {
            #[cfg(debug_assertions)]
            {
                let mut fresh = vec![0; row.len()];
                let ok = self.runtime().table_native_row(
                    table,
                    self.fields,
                    self.hints(),
                    &mut fresh,
                    self.meta_field,
                );
                debug_assert!(ok.is_ok() && fresh == *row, "a kept row is the row it was");
            }
            return;
        }
        self.fill_row_missed(table, row, Some((token, place)));
    }

    /// [`Self::fill_table_row`] for a row the memo does not hold: asks the
    /// runtime for it and, when `kept` names the table's token and the place
    /// of its shape in the memo, remembers the offsets.
    #[cold]
    #[inline(never)]
    fn fill_row_missed(
        &mut self,
        table: TableHandle,
        row: &mut [usize],
        kept: Option<(crate::runtime::NativeRowToken, usize)>,
    ) {
        if self
            .runtime()
            .table_native_row(table, self.fields, self.hints(), row, self.meta_field)
            .is_err()
        {
            row.fill(0);
            return;
        }
        let Some((token, place)) = kept else {
            return;
        };
        // SAFETY: as in `fill_table_row`.
        #[allow(unsafe_code)]
        let way = unsafe {
            std::slice::from_raw_parts_mut(
                self.memo.0.add(place),
                ROW_MEMO_HEADER + self.fields.len(),
            )
        };
        let (header, offsets) = way.split_at_mut(ROW_MEMO_HEADER);
        // Keep the offsets when the row names a slot of the table for every
        // field: a missing key, another type or the marker of an absent key
        // would need a look at the table again.
        let marker = std::ptr::addr_of!(crate::runtime::NATIVE_ABSENT_FIELD) as usize;
        let span = crate::runtime::MAX_SHAPE_KEYS * std::mem::size_of::<Value>();
        let keepable = row.iter().enumerate().all(|(field, &slot)| {
            self.meta_field.names(field)
                || (slot != marker
                    && slot
                        .checked_sub(token.slots)
                        .is_some_and(|offset| offset < span))
        });
        if keepable {
            header[0] = token.shape;
            header[1] = row_meta_word(token.metatable);
            header[2] = usize::from(token.watched);
            header[3] = self.meta_field.meta.map_or(0, |field| row[field]);
            for (offset, &slot) in offsets.iter_mut().zip(row.iter()) {
                *offset = slot.wrapping_sub(token.slots);
            }
        } else {
            header[0] = 0;
        }
    }

    /// The lane of the table a closure's cell holds, like [`TraceTables::lane`]
    /// including the scan of the tables the run made. A trace that reads table
    /// cells natively never stores a table into a cell itself (see
    /// `closure_row_words`), but a helper can, and a cell may then name a
    /// table the run made. Such a table keeps the lane it was made with, or
    /// the cell would get a second lane for the same table.
    fn cell_table_lane(&mut self, table: crate::runtime::TableHandle) -> Option<usize> {
        let identity = table.native_identity();
        if !self.fresh_lanes {
            let found = self.known_lane(table);
            if let Some(lane) = found {
                self.keep_made_lane(lane, identity);
                return Some(lane as usize);
            }
            if let Some(lane) = self.host.made.iter().copied().find(|lane| {
                matches!(self.bank().get(*lane as usize), Some(Value::Table(held)) if *held == table)
            }) {
                self.keep_made_lane(lane, identity);
                return Some(lane as usize);
            }
        }
        if self.bank().len() >= self.lane_cap() {
            return None;
        }
        self.bank().push(Value::Table(table));
        let lane = self.bank().len() - 1;
        if !self.fresh_lanes {
            if self.index_ok {
                let _ = self.index_insert(identity, lane);
            }
            if self.lanes_ready {
                self.lanes.insert(identity, u32::try_from(lane).ok()?);
            }
        }
        Some(lane)
    }
}

impl TraceTables for RunTables<'_> {
    #[allow(unsafe_code)]
    fn materialize_exit(
        &mut self,
        exit: usize,
        area: *const i64,
        elided: &mut (i64, i64),
        lanes: &mut [i64],
    ) -> Option<usize> {
        if self.exit_plans.0.is_null() || self.fresh_lanes {
            return None;
        }
        // SAFETY: the plans of the running trace's cache entry, which stay
        // where they are for the run (see `set_exit_plans`).
        let plans = unsafe { std::slice::from_raw_parts(self.exit_plans.0, self.exit_plans.1) };
        let restore = plans.get(exit)?;
        let count = restore.virtuals.len();
        if count == 0 || count > lanes.len() || !restore.bridges_to_side() {
            return None;
        }
        // The checks of `table_new_plain`, for every table: room in the
        // arena (a table made with it full would move the tables native code
        // holds the addresses of), no collection owed, a lane each.
        let runtime = self.runtime();
        if !runtime.table_slots_are_spare(count) || runtime.gc_needs_collection() {
            return None;
        }
        let mut reserved = [0_usize; crate::jit_ir::trace::MAX_HELPER_RESULTS];
        for at in 0..count {
            let Some(lane) = self.reserve_made_lane() else {
                for lane in &reserved[..at] {
                    self.return_made_lane(*lane);
                }
                return None;
            };
            reserved[at] = lane;
        }
        // The constructors the run elided so far are charged first, as an
        // exit charges them, so that each table takes its own charge back.
        let runtime = self.runtime();
        if elided.0 > 0 {
            runtime.charge_elided_tables(
                u64::try_from(elided.0).unwrap_or(0),
                usize::try_from(elided.1).unwrap_or(0),
            );
        }
        *elided = (0, 0);
        // SAFETY: the run's exit area, which native code wrote the exit's
        // values into just before this call.
        let area = unsafe { std::slice::from_raw_parts(area, self.exit_area_len) };
        let (bank, runtime) = (self.bank(), self.runtime());
        let Ok(handles) = restore.materialize(area, bank, self.roots, runtime) else {
            // The exit the side now leaves through builds the tables again
            // and takes their charges back again (or raises the same error).
            restore.charge_virtuals(runtime);
            for lane in &reserved[..count] {
                self.return_made_lane(*lane);
            }
            return None;
        };
        // Each table keeps its lane until the run ends, like a table a
        // lookup found (`keep_made_lane`): the side may leave it in a
        // register the loop carries, which a release at the end of the
        // root's body could not know of.
        for (at, handle) in handles.iter().enumerate() {
            let lane = reserved[at];
            self.bank()[lane] = Value::Table(*handle);
            let identity = handle.native_identity();
            if self.index_ok {
                let _ = self.index_insert(identity, lane);
            }
            if self.lanes_ready
                && let Ok(lane) = u32::try_from(lane)
            {
                self.lanes.insert(identity, lane);
            }
            lanes[at] = i64::try_from(lane).unwrap_or(-1);
        }
        Some(count)
    }

    fn lane(&mut self, value: &Value) -> Option<usize> {
        if let Value::Closure(closure) = value {
            return self.closure_lane(*closure);
        }
        if let Value::Thread(thread) = value {
            return self.thread_lane(*thread);
        }
        if let Value::Userdata(userdata) = value {
            return self.userdata_lane(*userdata);
        }
        let Value::Table(table) = value else {
            return None;
        };
        let identity = table.native_identity();
        if !self.fresh_lanes {
            // The index holds every table of the bank but the ones the run
            // made, which `made` finds: a table it lacks is not there.
            let found = self.known_lane(*table);
            if let Some(lane) = found {
                self.keep_made_lane(lane, identity);
                return Some(lane as usize);
            }
            // A table the run made is not in the map (that costs a hash per
            // table): it has the lane it was made with.
            if let Some(lane) = self.host.made.iter().copied().find(|lane| {
                matches!(self.bank().get(*lane as usize), Some(Value::Table(held)) if held == table)
            }) {
                self.keep_made_lane(lane, identity);
                return Some(lane as usize);
            }
        }
        if self.bank().len() >= self.lane_cap() {
            return None;
        }
        // The bank roots the table, so nothing a collection between the run
        // and the restore frees it.
        self.bank().push(Value::Table(*table));
        let lane = self.bank().len() - 1;
        if !self.fresh_lanes {
            if self.index_ok {
                let _ = self.index_insert(identity, lane);
            }
            if self.lanes_ready {
                self.lanes.insert(identity, u32::try_from(lane).ok()?);
            }
        }
        Some(lane)
    }

    fn admit_table(
        &mut self,
        value: &Value,
        space: &mut crate::jit::TraceTableSpace,
    ) -> Option<i64> {
        let Value::Table(table) = value else {
            return None;
        };
        if self.fresh_lanes || !self.index_ok {
            return None;
        }
        let place = if let Some(place) = self.index_place(*table) {
            place
        } else {
            // A table the run made has its lane already, and a full bank
            // has none to give.
            if !self.host.made.is_empty() || self.bank().len() >= self.lane_cap() {
                return None;
            }
            self.bank().push(Value::Table(*table));
            let lane = self.bank().len() - 1;
            let identity = table.native_identity();
            if self.lanes_ready {
                self.lanes.insert(identity, u32::try_from(lane).ok()?);
            }
            self.index_insert(identity, lane)?
        };
        let lane = self.index.entries[place].lane as usize;
        if lane >= space.lanes() {
            return Some(-1);
        }
        space.claim_row(lane, |row| {
            // A row may be longer than the fields (the closure rows of the
            // same trace need more); the fill writes every field's word.
            let fields = self.fields.len().min(row.len());
            if fields < row.len() {
                row[fields..].fill(0);
            }
            self.fill_table_row(*table, &mut row[..fields]);
        });
        self.index_ready(place);
        i64::try_from(lane).ok()
    }

    fn lane_index(&mut self) -> Option<(usize, u32)> {
        (!self.fresh_lanes && self.index_ok)
            .then_some((self.index.entries.as_ptr() as usize, (self.stamp << 1) | 1))
    }

    fn admits_fresh(&self) -> bool {
        self.fresh_lanes
    }

    fn admit_fresh(
        &mut self,
        value: &Value,
        rows: &mut [usize],
        stride: usize,
        masked: bool,
        filled: &mut [bool],
    ) -> Option<i64> {
        if !self.fresh_lanes {
            return None;
        }
        // A closure takes the general path, which keeps one lane for it.
        let Value::Table(table) = value else {
            return None;
        };
        let lane = self.bank().len();
        if lane >= MAX_FUNCTION_TABLE_LANES {
            return Some(-1);
        }
        self.bank().push(Value::Table(*table));
        filled[lane] = true;
        fill_sealed(
            &mut rows[lane * stride..(lane + 1) * stride],
            masked,
            |row| {
                let fields = self.fields.len().min(row.len());
                if fields < row.len() {
                    row[fields..].fill(0);
                }
                self.fill_table_row(*table, &mut row[..fields]);
            },
        );
        i64::try_from(lane).ok()
    }

    /// Keeps the row of a table that still says what it said: the runtime
    /// checks it ([`Runtime::table_native_row_holds`]), word by word against
    /// the table's entries once some table's keys moved since the rows were
    /// filled ([`Self::refill_rows`]).
    fn row_holds(&mut self, lane: usize, row: &[usize]) -> bool {
        // The lanes of a `next` loop change what they hold as it steps.
        if self
            .control
            .as_ref()
            .is_some_and(|lanes| lanes.contains(&lane))
        {
            return false;
        }
        let Some(Value::Table(table)) = self.bank().get(lane) else {
            return false;
        };
        let table = *table;
        let fields = self.fields.len().min(row.len());
        // The words past the fields are zero, as the fill left them.
        let holds = self.runtime().table_native_row_holds(
            table,
            self.fields,
            &row[..fields],
            self.meta_field,
            self.layout_moved,
        );
        #[cfg(debug_assertions)]
        if holds {
            let mut fresh = vec![0; fields];
            let ok = self.runtime().table_native_row(
                table,
                self.fields,
                self.hints(),
                &mut fresh,
                self.meta_field,
            );
            debug_assert!(
                ok.is_ok()
                    && fresh
                        .iter()
                        .zip(&row[..fields])
                        .all(|(fresh, kept)| fresh == kept || *kept == 0),
                "a kept row is one a fill gives, or zero where it is not"
            );
        }
        holds
    }

    fn row(&mut self, lane: usize, row: &mut [usize]) {
        let fields = self.fields.len().min(row.len());
        match self.bank().get(lane) {
            Some(Value::Table(table)) => {
                // A read-only table gets an empty row, and a stale handle has
                // none. A row may be longer than the fields: it is as long as
                // the closure rows of the same trace need. The fill writes
                // every field's word.
                if fields < row.len() {
                    row[fields..].fill(0);
                }
                let table = *table;
                self.fill_table_row(table, &mut row[..fields]);
            }
            // A closure's row: its prototype, and where each of its first
            // cells keeps its value, which stays put for the run.
            Some(Value::Closure(handle)) if row.len() >= crate::jit::CLOSURE_ROW_WORDS => {
                row.fill(0);
                let Some(parts) = self.runtime().closure_call_parts(*handle) else {
                    return;
                };
                // A closure with a cell the trace holds in a window slot is
                // one variable seen twice, which the interpreter runs.
                if self
                    .closures
                    .as_ref()
                    .is_some_and(|state| parts.cells.iter().any(|cell| state.held.contains(cell)))
                {
                    return;
                }
                row[0] = std::sync::Arc::as_ptr(parts.chunk) as usize;
                let mut tables = [None; crate::jit::CLOSURE_ROW_CELLS];
                for ((cell, slot), table) in parts
                    .cells
                    .iter()
                    .take(crate::jit::CLOSURE_ROW_CELLS)
                    .zip(&mut row[1..])
                    .zip(&mut tables)
                {
                    if let Ok(value) = self.runtime().cell_ref(*cell) {
                        *slot = std::ptr::from_ref(value) as usize;
                        if let Value::Table(held) = value {
                            *table = Some(*held);
                        }
                    }
                }
                // The tables of the cells get lanes, and rows after this one,
                // so native code can name them without asking the VM.
                if self
                    .closures
                    .as_ref()
                    .is_some_and(|state| state.table_cells)
                    && row.len() >= crate::jit::CLOSURE_ROW_TABLE_WORDS
                {
                    for (position, table) in tables.into_iter().enumerate() {
                        let Some(table) = table else {
                            continue;
                        };
                        let Some(lane) = self.cell_table_lane(table) else {
                            continue;
                        };
                        row[crate::jit::CLOSURE_ROW_WORDS + position] = lane + 1;
                        if let Some(state) = self.closures.as_mut() {
                            state.followers.push(lane);
                        }
                    }
                }
            }
            _ => row.fill(0),
        }
    }

    fn row_followers(&mut self, lanes: &mut Vec<usize>) {
        if let Some(state) = self.closures.as_mut() {
            lanes.append(&mut state.followers);
        }
    }

    fn closure_lanes(&mut self, lanes: &mut Vec<usize>) {
        // Every callable has a lane typed as a closure, and the code reads the
        // row of whichever one the callee is: a native function's row is
        // zero, which no prototype matches.
        lanes.extend(self.bank().iter().enumerate().filter_map(|(lane, held)| {
            matches!(
                held,
                Value::Closure(_)
                    | Value::NativeFunction(_)
                    | Value::GmatchIterator(_)
                    | Value::BuiltinFunction(_)
            )
            .then_some(lane)
        }));
    }

    fn set(
        &mut self,
        binding: usize,
        key: TraceInputSlot,
        value: TraceInputSlot,
        words: &mut TraceInputSlot,
    ) -> bool {
        let Some(table) = self.bound_table(binding) else {
            return false;
        };
        let Some(key) = key.scalar_value() else {
            return false;
        };
        // A string or table is a lane of the run's bank: the table takes a
        // counted reference to the string, or the handle of the table.
        let scalar;
        let value: &Value = if matches!(
            value.tag(),
            crate::jit::TRACE_TAG_STRING | crate::jit::TRACE_TAG_TABLE
        ) {
            match lane_reference(self.bank(), self.roots, value.tag(), value.bits()) {
                Some(held) => held,
                None => return false,
            }
        } else {
            let Some(converted) = value.scalar_value() else {
                return false;
            };
            scalar = converted;
            &scalar
        };
        // The interpreter's raw store: it refuses an absent key of a table
        // with a metatable (`__newindex` may apply), a NaN key and a
        // read-only table. Nothing here collects.
        match self.runtime().table_trace_set(table, &key, value) {
            Some(view) => {
                words.set_array(view);
                true
            }
            None => false,
        }
    }

    fn host_call(
        &mut self,
        function: HostFunction,
        args: &[TraceInputSlot],
        out: &mut TraceInputSlot,
        lengths: &mut [u32],
    ) -> bool {
        let answered = host_call(
            self.runtime(),
            self.bank(),
            self.roots,
            &mut self.host,
            function,
            args,
            out,
            lengths,
        );
        if function == HostFunction::ReleaseStrings {
            self.release_made_tables();
        }
        answered
    }

    fn host_entry(&mut self) -> Option<crate::jit::HostEntry> {
        Some(crate::jit::HostEntry {
            function: run_host_call as *const () as usize,
            context: std::ptr::from_mut(self) as usize,
            set: run_table_set as *const () as usize,
            store: run_array_store as *const () as usize,
        })
    }

    fn roots_address(&self) -> (usize, usize) {
        (self.roots.as_ptr() as usize, self.roots.len())
    }

    fn bank_address(&mut self) -> usize {
        // A run adds lanes to the bank (tables up to the cap on table lanes,
        // strings up to the cap on string lanes). Room for all of them keeps
        // the bank where it is, so native code can hold its address.
        self.bank()
            .reserve(self.lane_cap() + super::host::MAX_HOST_BANK_LANES);
        if let Some(lane) = self.cursor_lane {
            // The run is final now: the description stays where it is.
            let address = std::ptr::from_mut(&mut self.cursor) as usize;
            self.bank()[lane] = Value::Integer(i64::try_from(address).unwrap_or(0));
        }
        self.bank().as_mut_ptr() as usize
    }

    fn row_flags(&mut self, filled: &mut [bool]) {
        self.rows = (filled.as_mut_ptr(), filled.len());
    }

    fn row_space(&mut self, space: &mut crate::jit::TraceTableSpace) {
        self.space = std::ptr::from_mut(space);
    }

    fn string_lengths(&mut self, lengths: &mut [u32]) {
        self.lengths = (lengths.as_mut_ptr(), lengths.len());
        for (lane, held) in self
            .bank()
            .iter()
            .enumerate()
            .take(lengths.len().saturating_sub(1))
        {
            if let Value::String(text) = held {
                lengths[lane] = u32::try_from(text.len()).unwrap_or(u32::MAX);
            }
        }
    }

    #[allow(unsafe_code)]
    fn hash_get(&mut self, binding: usize, key_tag: i64, key_bits: i64) -> TraceInputSlot {
        // Read-only, nonallocating, noncollecting, non-reentrant: it only
        // reads the key and the bound table (see `TraceTables::hash_get`).
        let Some(table) = self.bound_table(binding) else {
            return TraceInputSlot::other();
        };
        let number;
        let key: &Value = match key_tag {
            crate::jit::TRACE_TAG_INTEGER => {
                number = Value::Integer(key_bits);
                &number
            }
            crate::jit::TRACE_TAG_FLOAT => {
                number = Value::Number(f64::from_bits(key_bits.cast_unsigned()));
                &number
            }
            crate::jit::TRACE_TAG_STRING => {
                // A lane of the bank, or a root string the trace names.
                let lane = key_bits.cast_unsigned();
                let held = match lane.checked_sub(crate::jit_ir::trace::ROOT_STRING_LANE_BASE) {
                    Some(slot) => usize::try_from(slot).ok().and_then(|at| self.roots.get(at)),
                    None => usize::try_from(lane)
                        .ok()
                        .and_then(|at| self.bank().get(at)),
                };
                let Some(held) = held else {
                    return TraceInputSlot::other();
                };
                held
            }
            crate::jit::TRACE_TAG_BORROWED_STRING => {
                // SAFETY: a borrowed string is the address of a slot of an
                // array part that a bound table holds and that this run has
                // not moved: the IR verifier ends every borrow at the first
                // operation that could mutate, allocate, collect or re-enter
                // (`TraceOp::ends_borrows`), and the owner is a table of the
                // run's root bank, so it is alive. The lookup below only
                // reads.
                unsafe { &*(key_bits as usize as *const Value) }
            }
            _ => return TraceInputSlot::other(),
        };
        let found = match key {
            Value::String(key) => self.runtime().table_get_string_ref(table, key),
            // A number: a raw lookup over both parts, so a key that moved
            // between them is still found.
            number => self
                .runtime()
                .table_get_raw_value(table, number)
                .ok()
                .flatten(),
        };
        match found {
            None | Some(Value::Nil) => TraceInputSlot::nil(),
            Some(Value::Integer(value)) => TraceInputSlot::integer(*value),
            Some(Value::Number(value)) => TraceInputSlot::float(*value),
            Some(Value::Boolean(value)) => TraceInputSlot::boolean(*value),
            Some(_) => TraceInputSlot::other(),
        }
    }

    fn same_table(&mut self, a: usize, b: usize) -> bool {
        a == b
            || self
                .bound_table(a)
                .is_some_and(|table| self.bound_table(b) == Some(table))
    }

    fn rewrite_lengths(&mut self, binding: usize, words: &mut [TraceInputSlot]) {
        if !self.has_lengths {
            return;
        }
        if let Some(table) = self.bound_table(binding) {
            self.rewrite_length_words(words, table);
        }
    }
}

/// The function native code calls for a host call
/// ([`TraceTables::host_entry`]): `context` is the run's [`RunTables`].
/// Writes the answer to `out`, or [`TraceInputSlot::other`] when the call
/// needs the interpreter.
#[allow(unsafe_code)]
// `argc`/`args` are the FFI host-call control block's own names.
#[allow(clippy::similar_names)]
extern "C" fn run_host_call(
    context: usize,
    args: *const TraceInputSlot,
    out: *mut TraceInputSlot,
    function: i64,
    argc: i64,
) {
    // SAFETY: `context` is the address of the `RunTables` that handed out
    // this entry, which outlives the run and is used by nothing else while
    // native code calls here. `args` and `out` are the control block's
    // `host_args` and `host_result`, and the code stored `argc` (at most
    // `MAX_HOST_ARGS`) arguments before the call. `lengths` is the run's
    // length table, live for the run.
    let (tables, out) = unsafe { (&mut *(context as *mut RunTables<'static>), &mut *out) };
    let count = usize::try_from(argc)
        .unwrap_or(0)
        .min(crate::jit_ir::trace::MAX_HOST_ARGS);
    let args = unsafe { std::slice::from_raw_parts(args, count) };
    let lengths = unsafe { std::slice::from_raw_parts_mut(tables.lengths.0, tables.lengths.1) };
    let function = HostFunction::from_code(function);
    let answered = match function {
        Some(function) => host_call(
            tables.runtime(),
            tables.bank(),
            tables.roots,
            &mut tables.host,
            function,
            args,
            out,
            lengths,
        ),
        None => false,
    };
    if function == Some(HostFunction::ReleaseStrings) {
        tables.release_made_tables();
    }
    if !answered {
        *out = TraceInputSlot::other();
    }
}

/// Installs the addresses of the helpers this crate implements with the
/// backend, once, before a trace that calls one is compiled.
pub(crate) fn install_helpers() {
    use crate::jit_ir::trace::HelperId;
    static INSTALLED: std::sync::Once = std::sync::Once::new();
    INSTALLED.call_once(|| {
        let string_len: extern "C" fn(usize, *mut i64, i64) -> i64 = helper_string_len;
        let string_len_ref: extern "C" fn(usize, *mut i64, i64) -> i64 = helper_string_len_ref;
        let string_own: extern "C" fn(usize, *mut i64, i64) -> i64 = helper_string_own;
        super::helpers::install();
        crate::jit::install_helper(HelperId::STRING_OWN, string_own as *const () as usize);
        crate::jit::install_helper(HelperId::STRING_LEN, string_len as *const () as usize);
        crate::jit::install_helper(
            HelperId::STRING_LEN_REF,
            string_len_ref as *const () as usize,
        );
    });
}

/// Puts the library's helpers for making and filling tables back after a test
/// installed stand-ins for them. The addresses are process-wide, and the
/// first call of [`install_helpers`] replaces any stand-in installed before
/// it, so a test that installs stand-ins calls this first, and again once its
/// trace is compiled (a trace keeps the addresses it was compiled with).
#[cfg(test)]
pub(crate) fn restore_table_helpers() {
    install_helpers();
    super::helpers::restore_table_helpers();
}

/// Writes the byte count of `held` as the helper's one result. Misses (1) for
/// a value that is no string.
#[allow(unsafe_code)]
fn string_length_result(held: Option<&Value>, out: *mut i64) -> i64 {
    let Some(Value::String(text)) = held else {
        return 1;
    };
    let Ok(length) = i64::try_from(text.len()) else {
        return 1;
    };
    // SAFETY: `out` is the control block's `helper_out`, which holds a word
    // per result and which nothing else touches during the call.
    unsafe { *out = length };
    0
}

/// The helper `string.len` of a string lane of the bank or of the roots
/// (`extern "C" fn(context, out, lane) -> status`, see
/// [`oximond::jit_ir::trace::HELPERS`]). `context` is the run's [`RunTables`].
#[allow(unsafe_code)]
extern "C" fn helper_string_len(context: usize, out: *mut i64, lane: i64) -> i64 {
    // SAFETY: `context` is the address of the `RunTables` that handed out
    // the host entry, which outlives the run and is used by nothing else
    // while native code calls here. The helper only reads.
    let tables = unsafe { &*(context as *const RunTables<'static>) };
    let lane = lane.cast_unsigned();
    let held = match lane.checked_sub(crate::jit_ir::trace::ROOT_STRING_LANE_BASE) {
        Some(slot) => usize::try_from(slot)
            .ok()
            .and_then(|slot| tables.roots.get(slot)),
        None => usize::try_from(lane)
            .ok()
            .and_then(|lane| tables.bank().get(lane)),
    };
    string_length_result(held, out)
}

/// The helper `string.len_ref` of a string borrowed from a table slot: `slot`
/// is the address of the slot (`extern "C" fn(context, out, slot) -> status`).
#[allow(
    unsafe_code,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the word is the address of a slot of this process"
)]
extern "C" fn helper_string_len_ref(_context: usize, out: *mut i64, slot: i64) -> i64 {
    // SAFETY: a borrowed string is the address of a slot of an array part
    // that a bound table holds and that this run has not moved: the IR
    // verifier ends every borrow at the first operation that could mutate,
    // allocate, collect or re-enter (`TraceOp::ends_borrows`), and the owner
    // is a table of the run's root bank, so it is alive. The read only looks
    // at the string.
    let held = unsafe { &*(slot as usize as *const Value) };
    string_length_result(Some(held), out)
}

/// The helper `string.own` of a string borrowed from a table slot: gives the
/// string a lane of the run's bank (`extern "C" fn(context, out, slot) ->
/// status`). Misses (1) when the run has no room for one.
#[allow(
    unsafe_code,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the word is the address of a slot of this process"
)]
extern "C" fn helper_string_own(context: usize, out: *mut i64, slot: i64) -> i64 {
    // SAFETY: `context` is the address of the `RunTables` that handed out
    // the host entry, which outlives the run and is used by nothing else
    // while native code calls here.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    // SAFETY: as in `helper_string_len_ref`: the slot is alive until the
    // first operation that ends borrows, which this one is in front of.
    let Value::String(text) = (unsafe { &*(slot as usize as *const Value) }) else {
        return 1;
    };
    let text = text.clone();
    let (_, bank, _, state, lengths) = tables.library();
    let mut lane = TraceInputSlot::nil();
    if !super::host::publish_string(bank, state, lengths, text, &mut lane) {
        return 1;
    }
    // SAFETY: `out` is the control block's `helper_out`, which holds a word
    // per result and which nothing else touches during the call.
    unsafe { *out = lane.bits() };
    0
}

/// The string or table the entry tag and bits name: a lane of the run's bank,
/// or a root string of the trace.
fn lane_reference<'a>(
    bank: &'a [Value],
    roots: &'a [Value],
    tag: i64,
    bits: i64,
) -> Option<&'a Value> {
    let lane = bits.cast_unsigned();
    let held = match lane.checked_sub(ROOT_STRING_LANE_BASE) {
        Some(slot) if tag == crate::jit::TRACE_TAG_STRING => roots.get(usize::try_from(slot).ok()?),
        _ => bank.get(usize::try_from(lane).ok()?),
    }?;
    match (tag, held) {
        (crate::jit::TRACE_TAG_STRING, Value::String(_))
        | (crate::jit::TRACE_TAG_TABLE, Value::Table(_)) => Some(held),
        _ => None,
    }
}

/// The store of a string or table into an array slot of an appendable table
/// (`oximond::jit::HostEntry::store`): the table takes a counted reference to
/// the string, or the handle of the table, which the collector's write
/// barrier sees. `binding` is a binding of the register of the table that
/// holds the slot. `slot` is spare capacity when `replace` is zero, and
/// otherwise holds a value this releases; a nil only releases it. Returns 0, or -1 when it stored
/// nothing.
#[allow(
    unsafe_code,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the word is the address of a slot of this process"
)]
extern "C" fn run_array_store(
    context: usize,
    binding: i64,
    tag: i64,
    bits: i64,
    slot: i64,
    replace: i64,
) -> i64 {
    // SAFETY: `context` is the address of the `RunTables` that handed out this
    // entry, which outlives the run.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    if tag == crate::jit::TRACE_TAG_NIL {
        // A pop of a string or table: the slot is released and left nil,
        // and the code shrinks the array past it.
        if replace == 0 {
            return -1;
        }
        let slot = slot as usize as *mut Value;
        // SAFETY: as below, the slot holds a value that nothing else holds
        // a reference to during the call.
        unsafe { *slot = Value::Nil };
        return 0;
    }
    if tag == crate::jit::TRACE_TAG_TABLE {
        // A table is a handle: nothing is counted, and the barrier reads
        // both ages itself.
        let Some(&Value::Table(child)) = usize::try_from(bits)
            .ok()
            .and_then(|lane| tables.bank().get(lane))
        else {
            return -1;
        };
        let slot = slot as usize as *mut Value;
        // SAFETY: as below.
        unsafe {
            if replace == 0 {
                std::ptr::write(slot, Value::Table(child));
            } else {
                *slot = Value::Table(child);
            }
        }
        if let Some(table) = usize::try_from(binding)
            .ok()
            .and_then(|binding| tables.bound_table(binding))
        {
            tables
                .runtime()
                .table_native_stored_table_slot(table, slot, child);
        }
        return 0;
    }
    let Some(held) = lane_reference(tables.bank(), tables.roots, tag, bits) else {
        return -1;
    };
    let child = match held {
        Value::Table(child) => Some(*child),
        _ => None,
    };
    let copy = held.clone();
    let slot = slot as usize as *mut Value;
    // SAFETY: the slot is inside the array part of a table the run bound
    // as appendable, and the code checked that before it called: with
    // `replace` zero it is spare capacity (so nothing is released), else
    // a value that nothing else holds a reference to during the call.
    unsafe {
        if replace == 0 {
            std::ptr::write(slot, copy);
        } else {
            *slot = copy;
        }
    }
    if let Some(child) = child
        && let Some(table) = usize::try_from(binding)
            .ok()
            .and_then(|binding| tables.bound_table(binding))
    {
        tables
            .runtime()
            .table_native_stored_slot(table, slot, &Value::Table(child));
    }
    0
}

/// The function the table set stub calls ([`TraceTables::set`] for a run, and
/// the rewrite of the array words of every binding that names the table the
/// set may have moved). `context` is the run's [`RunTables`]. Returns 0, or -1
/// when nothing was stored.
#[allow(unsafe_code)]
extern "C" fn run_table_set(
    context: usize,
    words: *mut TraceInputSlot,
    binding: i64,
    key_tag: i64,
    key_bits: i64,
    value_tag: i64,
    value_bits: i64,
) -> i64 {
    // SAFETY: `context` is the address of the `RunTables` that handed out
    // this entry, which outlives the run; `words` is the run's input buffer
    // from its first binding slot, which holds one slot per binding and which
    // nothing else touches during the call.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    tables.store_calls.set_array += 1;
    let count = tables.arrays.len();
    let Ok(binding) = usize::try_from(binding) else {
        return -1;
    };
    if !tables.arrays.get(binding).copied().unwrap_or(false) {
        return -1;
    }
    let words = unsafe { std::slice::from_raw_parts_mut(words, count) };
    let Some(table) = tables.bound_table(binding) else {
        return -1;
    };
    let key = match key_tag {
        crate::jit::TRACE_TAG_INTEGER => Value::Integer(key_bits),
        crate::jit::TRACE_TAG_FLOAT => Value::Number(f64::from_bits(key_bits.cast_unsigned())),
        _ => return -1,
    };
    // A string is a lane of the run's bank: the table takes a counted
    // reference to it.
    let scalar;
    let value: &Value = match value_tag {
        crate::jit::TRACE_TAG_STRING | crate::jit::TRACE_TAG_TABLE => {
            match lane_reference(tables.bank(), tables.roots, value_tag, value_bits) {
                Some(held) => held,
                None => return -1,
            }
        }
        _ => {
            let Some(converted) = TraceInputSlot::from_words(value_tag, value_bits).scalar_value()
            else {
                return -1;
            };
            scalar = converted;
            &scalar
        }
    };
    // The interpreter's raw store: it refuses an absent key of a table with a
    // metatable (`__newindex` may apply), a NaN key and a read-only table.
    // Nothing here collects.
    let Some(view) = tables.runtime().table_trace_set(table, &key, value) else {
        return -1;
    };
    words[binding].set_array(view);
    // The set may have grown the table every other array binding of it names,
    // and changed its border, which a length binding (another binding) holds.
    if count > 1 {
        let (tag, bits) = (words[binding].tag(), words[binding].bits());
        for index in 0..count {
            if index != binding && tables.arrays[index] && tables.same_table(binding, index) {
                words[index] = TraceInputSlot::from_words(tag, bits);
            }
        }
        if tables.has_lengths {
            tables.rewrite_length_words(words, table);
        }
    }
    0
}

/// Where an entry input comes from (see [`InstalledTrace::entry_plan`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EntryStep {
    /// A register of the frame.
    Register(u16),
    /// An upvalue, global or other window slot beyond the frame's registers,
    /// by position among the trace's externals.
    External(u16),
}

/// The entry plan of a trace, or none when its entry is not plain.
fn entry_plan_of(
    compiled: &CompiledTrace,
    ir: &TraceIr,
    register_count: usize,
) -> Box<[EntryStep]> {
    if ir.boxed_control.is_some() || ir.boxed_value {
        return Box::default();
    }
    compiled
        .entry_contract()
        .iter()
        .map(|(register, _)| {
            let register = usize::from(*register);
            match register.checked_sub(register_count) {
                Some(external) => u16::try_from(external).map(EntryStep::External),
                None => u16::try_from(register).map(EntryStep::Register),
            }
        })
        .collect::<Result<Box<[_]>, _>>()
        .unwrap_or_default()
}

/// Whether `bind_simple` resolves every binding of `compiled`: an array part
/// to index, a length, and a field the table holds that the trace pinned as
/// a constant (the library function a loop calls).
fn simple_bindings(compiled: &CompiledTrace) -> bool {
    compiled.bindings().iter().all(|binding| {
        matches!(
            binding.kind,
            BindingKind::Array { first: None }
                | BindingKind::Callee { depth: 0, .. }
                | BindingKind::Length { .. }
        )
    })
}

/// [`bind_tables`] for a trace whose bindings are all of the kinds
/// [`simple_bindings`] names, which need none of its bookkeeping: they bind
/// no field twice, read no header and follow no `__index`.
fn bind_simple(
    plans: &[BindingPlan],
    bindings: &[crate::jit_ir::trace::TableBinding],
    inputs: &[TraceInputSlot],
    views: &mut Vec<TraceBinding>,
    bank: &[Value],
    runtime: &mut Runtime,
    reads: &mut crate::runtime::StringReadCache,
) -> bool {
    for (plan, binding) in plans.iter().zip(bindings) {
        let slot = inputs[plan.input];
        let Some(Value::Table(table)) = usize::try_from(slot.bits())
            .ok()
            .filter(|_| slot.tag() == crate::jit::TRACE_TAG_TABLE)
            .and_then(|lane| bank.get(lane))
        else {
            return false;
        };
        match &binding.kind {
            BindingKind::Array { .. } => {
                if plan.writes && runtime.table_is_readonly(*table).unwrap_or(true) {
                    return false;
                }
                let Ok(view) = runtime.table_trace_array(*table) else {
                    return false;
                };
                views.push(TraceBinding::array(view));
            }
            BindingKind::Callee { lane, .. } => {
                // The field must hold the callable the trace pinned: the
                // recorder read it as that constant.
                let Some(Value::String(key)) = &plan.key else {
                    return false;
                };
                let held = runtime.table_get_string_cached(*table, key, reads);
                let pinned = usize::try_from(*lane).ok().and_then(|lane| bank.get(lane));
                if !matches!((held, pinned), (Some(held), Some(pinned)) if same_object(held, pinned))
                {
                    return false;
                }
                views.push(TraceBinding::callee());
            }
            BindingKind::Length { ty, raw } => {
                let length = if *raw {
                    runtime.table_length(*table).map(Some)
                } else {
                    runtime.table_length_without_len_metamethod(*table)
                };
                let Ok(Some(length)) = length else {
                    return false;
                };
                views.push(if *ty == TraceType::Float {
                    TraceBinding::float_length(length)
                } else {
                    // A border is a whole number below 2^53.
                    #[allow(clippy::cast_possible_truncation)]
                    TraceBinding::integer_length(length as i64)
                });
            }
            _ => return false,
        }
    }
    true
}

/// Resolves every binding of a trace against the tables in the entry inputs
/// and appends one view per binding to `views`. Returns whether all of them hold.
/// See [`oximond::jit_ir::trace::TableBinding`] for what native code does with
/// the words.
fn bind_tables(
    plans: &[BindingPlan],
    bindings: &[crate::jit_ir::trace::TableBinding],
    inputs: &[TraceInputSlot],
    views: &mut Vec<TraceBinding>,
    bank: &[Value],
    runtime: &mut Runtime,
    probes: Option<&mut crate::runtime::NativeStringProbeScratch>,
    reads: &mut crate::runtime::StringReadCache,
    index_use: IndexUse<'_>,
) -> bool {
    let pins_keys = plans.iter().any(|plan| plan.pins);
    let base = views.len();
    // The string index of the table of a `Hash` or `NoMetatable` binding,
    // for a trace that probes it in native code ([`CompiledTrace::probes_hash`]),
    // and the tables already indexed: bindings of one table share an index.
    let mut probes = probes;
    if let Some(scratch) = probes.as_deref_mut() {
        scratch.begin();
    }
    let mut indexed: Vec<(TableHandle, Option<crate::runtime::NativeStringProbe>)> = Vec::new();
    // The table whose storage each binding resolved: the bound table, or for
    // a field behind `__index` the table that holds it.
    // A run binds a few tables, so the holders sit on the stack.
    let mut small_holders = [None; 8];
    let mut many_holders;
    let holders: &mut [Option<TableHandle>] = if plans.len() <= small_holders.len() {
        &mut small_holders[..plans.len()]
    } else {
        many_holders = vec![None; plans.len()];
        &mut many_holders
    };
    for (index, (plan, binding)) in plans.iter().zip(bindings).enumerate() {
        let slot = inputs[plan.input];
        if matches!(binding.kind, BindingKind::Bytes { .. }) {
            let Some(Value::String(text)) = usize::try_from(slot.bits())
                .ok()
                .filter(|_| slot.tag() == crate::jit::TRACE_TAG_STRING)
                .and_then(|lane| bank.get(lane))
            else {
                return false;
            };
            views.push(TraceBinding::bytes(crate::runtime::NativeBytes::of(text)));
            continue;
        }
        let Some(Value::Table(table)) = usize::try_from(slot.bits())
            .ok()
            .filter(|_| slot.tag() == crate::jit::TRACE_TAG_TABLE)
            .and_then(|lane| bank.get(lane))
        else {
            return false;
        };
        let table = *table;
        // A field may sit in a fallback table behind `__index`. The entry
        // follows exactly the hops the recording saw, so a key that has since
        // appeared on the way (or a chain that changed) refuses the entry.
        let table = match (&binding.kind, &plan.key) {
            (
                BindingKind::Field { depth, .. }
                | BindingKind::Callee { depth, .. }
                | BindingKind::StringField { depth, .. },
                Some(Value::String(key)),
            ) => {
                // A field the table holds itself needs no walk: the access
                // below finds it there or refuses the entry.
                if *depth == 0 {
                    table
                } else {
                    match runtime.table_string_holder(table, key, *depth) {
                        Some((holder, hops)) if hops == *depth => holder,
                        _ => return false,
                    }
                }
            }
            _ => table,
        };
        holders[index] = Some(table);
        // Registers that alias one table bind the same field more than once,
        // and two tables may reach one field through `__index`. Resolving it
        // again would form a second `&mut` to the slot and invalidate the
        // first address under Rust's aliasing rules, so reuse the earlier
        // view.
        if let BindingKind::Field { ty, .. } = &binding.kind
            && let Some(earlier) = (0..index).find(|&earlier| {
                holders[earlier] == Some(table)
                    && plans[earlier].key == plan.key
                    && matches!(&bindings[earlier].kind, BindingKind::Field { ty: other, .. } if other == ty)
            })
        {
            let view = views[base + earlier];
            views.push(view);
            continue;
        }
        let word = match &binding.kind {
            BindingKind::Array { first } => {
                // A trace recorded over integers is no use on an array of
                // floats: every access would leave. Slot 1 tells the
                // versions of a header apart.
                if let Some(first) = first {
                    let kind = slot_kind(*first);
                    let held = matches!(
                        runtime.table_raw_array_range(table, 1, 1),
                        Ok(Some([value])) if kind.holds(value)
                    );
                    if !held {
                        return false;
                    }
                }
                if plan.writes && runtime.table_is_readonly(table).unwrap_or(true) {
                    return false;
                }
                let Ok(view) = runtime.table_trace_array(table) else {
                    return false;
                };
                TraceBinding::array(view)
            }
            BindingKind::Field { ty, .. } => {
                let kind = slot_kind(*ty);
                let key = plan.key.as_ref().expect("a field binding has a key");
                match runtime.table_native_value_slot(table, key, kind) {
                    Ok(Some(slot)) => TraceBinding::field(slot),
                    _ => return false,
                }
            }
            BindingKind::Callee { lane, .. } => {
                // The field must hold the callable the trace pinned: the
                // recorder read it as that constant.
                let key = plan.key.as_ref().expect("a callee binding has a key");
                let Value::String(key) = key else {
                    return false;
                };
                let held = runtime.table_get_string_cached(table, key, reads);
                let pinned = usize::try_from(*lane).ok().and_then(|lane| bank.get(lane));
                if !matches!((held, pinned), (Some(held), Some(pinned)) if same_object(held, pinned))
                {
                    return false;
                }
                TraceBinding::callee()
            }
            BindingKind::MetaClass { .. } => {
                // The pinned metatable must still be the class the
                // recording saw. The input only gives the binding a table
                // of the run.
                if !plan
                    .class
                    .as_ref()
                    .is_some_and(|class| class.holds(runtime))
                {
                    return false;
                }
                TraceBinding::callee()
            }
            BindingKind::StringField { text, .. } => {
                // The field must still hold the string the trace read as a
                // constant.
                let key = plan.key.as_ref().expect("a string binding has a key");
                let Value::String(key) = key else {
                    return false;
                };
                match runtime.table_get_string_ref(table, key) {
                    Some(Value::String(held)) if held.as_bytes() == &text[..] => {}
                    _ => return false,
                }
                TraceBinding::callee()
            }
            // Resolved above, from a string input.
            BindingKind::Bytes { .. } => return false,
            BindingKind::Header { .. } => {
                let Ok(header) = runtime.table_native_header(table) else {
                    return false;
                };
                TraceBinding::header(header.word(), header.address)
            }
            // A nil the trace reads out of the table must be the access's
            // value: no `__index` may run.
            BindingKind::Hash => string_index(
                runtime,
                probes.as_deref_mut(),
                &mut indexed,
                (table, index_use, pins_keys),
            ),
            BindingKind::NoMetatable => {
                if !matches!(runtime.raw_metatable(&Value::Table(table)), Ok(None)) {
                    return false;
                }
                string_index(
                    runtime,
                    probes.as_deref_mut(),
                    &mut indexed,
                    (table, index_use, pins_keys),
                )
            }
            BindingKind::Length { ty, raw } => {
                let length = if *raw {
                    runtime.table_length(table).map(Some)
                } else {
                    runtime.table_length_without_len_metamethod(table)
                };
                let Ok(Some(length)) = length else {
                    return false;
                };
                if *ty == TraceType::Float {
                    TraceBinding::float_length(length)
                } else {
                    // A border is a whole number below 2^53.
                    #[allow(clippy::cast_possible_truncation)]
                    TraceBinding::integer_length(length as i64)
                }
            }
        };
        views.push(word);
    }
    !aliases_a_header(bindings, holders) && !moves_a_field_table(plans, bindings, holders)
}

/// Whether a set that may move the hash part of its table names a table that
/// a field binding addresses. Fields of other tables stay where they are.
fn moves_a_field_table(
    plans: &[BindingPlan],
    bindings: &[crate::jit_ir::trace::TableBinding],
    holders: &[Option<TableHandle>],
) -> bool {
    plans
        .iter()
        .enumerate()
        .filter(|(_, plan)| plan.moves)
        .any(|(index, _)| {
            bindings.iter().enumerate().any(|(other, binding)| {
                matches!(binding.kind, BindingKind::Field { .. })
                    && holders[other].is_some()
                    && holders[other] == holders[index]
            })
        })
}

/// The binding of a table whose hash part a lookup reads: the table's string
/// index when the trace probes it in native code (`probes` is the scratch for
/// the indexes of small tables) and the runtime has one, else a binding with
/// no index, which sends the lookups to the VM. A table bound twice is
/// indexed once; the tables already indexed are in `indexed`. The trace's
/// first run indexes a big table at once (`first_run`, see
/// [`Runtime::table_native_probe_now`]).
fn string_index(
    runtime: &mut Runtime,
    probes: Option<&mut crate::runtime::NativeStringProbeScratch>,
    indexed: &mut Vec<(TableHandle, Option<crate::runtime::NativeStringProbe>)>,
    (table, index_use, pins_keys): (TableHandle, IndexUse<'_>, bool),
) -> TraceBinding {
    let Some(scratch) = probes else {
        return TraceBinding::callee();
    };
    let found = if let Some((_, found)) = indexed.iter().find(|(held, _)| *held == table) {
        *found
    } else {
        let found = if index_use.first_run {
            runtime.table_native_probe_now(table, scratch)
        } else {
            runtime.table_native_probe(table, scratch)
        };
        indexed.push((table, found));
        found
    };
    // Native code deletes and revives entries only where no store of the run
    // may name a key the run fixes: a run that pins a key leaves them to the
    // helpers, which compare the key, and so does a table that holds a global
    // the run read at entry.
    let tombstones = !pins_keys && !index_use.holders.holds(table);
    found.map_or_else(TraceBinding::callee, |probe| {
        TraceBinding::probe(probe, tombstones)
    })
}

/// What a run knows about the use of the string indexes it binds.
#[derive(Clone, Copy)]
struct IndexUse<'a> {
    /// Whether this is the trace's first run, which is expected to be long.
    first_run: bool,
    /// The tables that hold the globals the run read at entry.
    holders: &'a GlobalHolders,
}

/// Whether some table that a header binding names is also named by a binding
/// of another register. A set updates the words of the bindings of its own
/// register only, so the run refuses such a table.
fn aliases_a_header(
    bindings: &[crate::jit_ir::trace::TableBinding],
    holders: &[Option<TableHandle>],
) -> bool {
    bindings.iter().enumerate().any(|(index, binding)| {
        matches!(binding.kind, BindingKind::Header { .. })
            && bindings.iter().enumerate().any(|(other, rival)| {
                rival.register != binding.register
                    && holders[other].is_some()
                    && holders[other] == holders[index]
                    && matches!(
                        rival.kind,
                        BindingKind::Array { .. }
                            | BindingKind::Length { .. }
                            | BindingKind::Header { .. }
                    )
            })
    })
}

/// The strings behind a trace's root slots, or `None` when a slot has no
/// string.
fn resolve_roots(ir: &TraceIr) -> Option<Box<[Value]>> {
    if ir.root_constants.len() != ir.root_slot_count as usize {
        return None;
    }
    Some(
        ir.root_constants
            .iter()
            .map(|text| Value::String(crate::string::LuaString::from(&text[..])))
            .collect(),
    )
}

/// What removing a version left behind.
struct RemovedVersion {
    chunk: Option<Arc<Chunk>>,
    header_pc: usize,
    versions_remain: bool,
    /// The loop headers the removed trace covered.
    covered: Vec<usize>,
}

#[cfg(test)]
mod grow_tests {
    use super::{grow_tree, refused_as_stale, shrink_tree};
    use crate::jit_ir::trace::{
        CellAlias, CellSource, SnapshotId, TraceBuilder, TraceFrameInfo, TraceId, TraceIr,
        TraceSide, TraceTerminator, TreeError,
    };
    use std::sync::Arc;

    fn bare(id: u32) -> TraceIr {
        let frame = TraceFrameInfo {
            register_count: 1,
            code_len: 8,
            inlined: None,
            activation: None,
        };
        TraceBuilder::new(TraceId(id), frame).finish(TraceTerminator::Exit {
            snapshot: SnapshotId(0),
        })
    }

    #[test]
    fn a_refused_side_leaves_the_tree_as_it_was() {
        let mut tree = bare(0);
        tree.root_constants = vec![Arc::from(&b"a"[..])];
        tree.root_slot_count = 1;
        tree.pinned_closures = vec![7];
        let before = tree.clone();

        let mut side = bare(1);
        side.root_constants = vec![Arc::from(&b"a"[..]), Arc::from(&b"b"[..])];
        side.root_slot_count = 2;
        side.pinned_closures = vec![7, 9];
        side.cell_aliases = vec![CellAlias {
            cell: CellSource::ClosureUpvalue {
                closure: 1,
                index: 0,
            },
            is: CellSource::Upvalue(0),
        }];
        let side = TraceSide {
            parent: None,
            guard: 0,
            ir: Box::new(side),
        };

        let growth = grow_tree(&mut tree, side);
        assert_eq!(tree.sides.len(), 1);
        assert_eq!(tree.root_constants.len(), 2);
        assert_eq!(tree.root_slot_count, 2);
        assert_eq!(tree.pinned_closures, [7, 9]);
        assert_eq!(tree.cell_aliases.len(), 1);

        shrink_tree(&mut tree, growth);
        assert_eq!(tree, before);
        assert_eq!(tree.root_constants, before.root_constants);
        assert_eq!(tree.pinned_closures, before.pinned_closures);
        assert_eq!(tree.cell_aliases, before.cell_aliases);
    }

    #[test]
    fn a_side_whose_window_another_side_outgrew_is_stale_not_bad() {
        // Two sides of one tree, recorded against the same version while a
        // compile was in flight, each name a root string the root does not:
        // the first joins and takes slot 1, and the second's slot 1 is
        // another string. The tree refuses the second, which records again
        // against the grown tree instead of spending its guard's attempts.
        let mut tree = bare(0);
        tree.root_constants = vec![Arc::from(&b"a"[..])];
        tree.root_slot_count = 1;
        let side_naming = |name: &'static [u8]| {
            let mut ir = bare(1);
            ir.root_constants = vec![Arc::from(&b"a"[..]), Arc::from(name)];
            ir.root_slot_count = 2;
            TraceSide {
                parent: None,
                guard: 0,
                ir: Box::new(ir),
            }
        };
        grow_tree(&mut tree, side_naming(b"b"));
        // The bare traces have no guards to continue, but the first side's
        // window is the tree's.
        assert!(
            crate::jit_ir::trace::verify_side_fit(&tree, 0, false)
                .err()
                .is_none_or(|error| !error.is_side_window())
        );
        grow_tree(&mut tree, side_naming(b"c"));
        let error = crate::jit_ir::trace::verify_side_fit(&tree, 1, false)
            .expect_err("the second side names another slot 1");
        assert!(error.is_side_window(), "{error:?}");
        let refused = TreeError::NotSource;
        assert!(refused_as_stale(&refused, Some(&error), true));
        // A side recorded against the tree as it is has no excuse.
        assert!(!refused_as_stale(&refused, Some(&error), false));
        // Nor does a side that fails another check.
        let other = crate::jit_ir::trace::TraceVerifyError::InvalidEntry {
            index: 1,
            detail: "a side trace has sides of its own",
        };
        assert!(!refused_as_stale(&refused, Some(&other), true));
        assert!(!refused_as_stale(&refused, None, true));
        assert!(refused_as_stale(
            &TreeError::WindowMismatch { side: 1 },
            None,
            true
        ));
    }

    #[test]
    fn a_second_hold_waits_its_full_length_after_a_first_one_that_counted_refusals() {
        let mut side = super::super::native::SideBook::default();
        side.hold(100);
        side.held_refused = 50;
        // The compile starts and ends the hold with refusals counted.
        assert!(side.release());
        assert_eq!(side.held_refused, 0);
        side.hold(200);
        assert_eq!(side.held_at, Some(200));
        assert_eq!(side.held_refused, 0);
        // A hold that is already running keeps its start and its count.
        side.held_refused = 3;
        side.hold(250);
        assert_eq!(side.held_at, Some(200));
        assert_eq!(side.held_refused, 3);
        // Ending it by timeout (see `held_too_long`) clears both too.
        side.release();
        side.hold(300);
        assert_eq!(side.held_refused, 0);
    }
}

#[cfg(test)]
mod return_tests {
    use super::returns_after_call;
    use crate::bytecode::{Chunk, OpCode};

    /// Whether each call of the first function `source` defines returns
    /// right after it, in the order of the calls.
    fn calls_that_return(source: &str) -> Vec<bool> {
        let chunk = crate::compiler::compile(source).expect("the source compiles");
        let function: &Chunk = &chunk.prototypes[0];
        function
            .code
            .iter()
            .enumerate()
            .filter(|(_, instruction)| {
                matches!(
                    instruction.opcode(),
                    Ok(OpCode::Call | OpCode::CallOpen | OpCode::CallOne | OpCode::CallVararg)
                )
            })
            .map(|(pc, _)| returns_after_call(function, pc))
            .collect()
    }

    #[test]
    fn a_call_whose_result_the_caller_returns_returns_after_the_call() {
        // `equal_p` of `mcl_util/shape.lua`.
        assert_eq!(
            calls_that_return("local function f(g, a, b) return not g(a, b, 1) end"),
            [true]
        );
        assert_eq!(
            calls_that_return("local function f(g, t) local r = g(t.x); return r + t.y end"),
            [true]
        );
    }

    #[test]
    fn a_call_followed_by_a_branch_a_loop_or_a_call_does_not() {
        assert_eq!(
            calls_that_return("local function f(g, a) if g(a) then return 1 end return 2 end"),
            [false]
        );
        assert_eq!(
            calls_that_return(
                "local function f(g, a) local n = 0 for i = 1, 3 do n = n + g(a) end return n end"
            ),
            [false]
        );
        assert_eq!(
            calls_that_return("local function f(g, a) local r = g(a) return g(r) end"),
            [false]
        );
    }
}

#[cfg(test)]
mod plan_tests {
    use super::{ExitRestore, RegisterAction, restores_of};
    use crate::jit::compile_trace;
    use crate::jit_ir::trace::{
        FrameId, NumberRegister, RegisterEntry, Snapshot, SnapshotFrame, SnapshotPhase,
        TraceBuilder, TraceFrameInfo, TraceId, TraceScalar, TraceTerminator, TraceType,
    };
    use crate::vm::jit::trace::OptimizationTier;

    const REGISTERS: usize = 6;

    /// A trace over a window of six registers that leaves at once: register
    /// 3 takes the float the trace entered with in register 0, register 4
    /// becomes nil and the rest stay as the interpreter left them.
    fn wide() -> crate::jit_ir::trace::TraceIr {
        use RegisterEntry::{Const, Unchanged, Value as Val};
        let mut b = TraceBuilder::new(
            TraceId(7),
            TraceFrameInfo {
                register_count: 6,
                code_len: 4,
                inlined: None,
                activation: None,
            },
        );
        b.mark_written(FrameId(0), 3);
        b.mark_written(FrameId(0), 4);
        let x = b.input(0, TraceType::Float);
        let exit = b.snapshot(Snapshot::new(
            SnapshotPhase::After {
                pc: 1,
                successor: 2,
            },
            0,
            vec![SnapshotFrame {
                frame: FrameId(0),
                last_interpreted_pc: Some(1),
                registers: vec![
                    Unchanged,
                    Unchanged,
                    Unchanged,
                    Val(x),
                    Const(TraceScalar::Nil),
                    Unchanged,
                ]
                .into(),
            }],
        ));
        b.finish(TraceTerminator::Exit { snapshot: exit })
    }

    #[test]
    fn a_plan_names_only_the_registers_the_exit_writes() {
        for tier in [OptimizationTier::Baseline, OptimizationTier::Optimized] {
            let compiled = compile_trace(&wide(), tier).unwrap();
            let restores = restores_of(&compiled, REGISTERS, &[]).unwrap();
            assert!(!restores.is_empty());
            for restore in &restores {
                assert_eq!(restore.nils.as_ref(), &[4]);
                assert_eq!(restore.floats.len(), 1);
                assert_eq!(restore.floats[0].0, 3);
                assert!(restore.changes.is_empty());
            }
        }
    }

    /// A snapshot with number registers does not skip its unchanged
    /// entries: a register tagged as a number of run-time subtype restores
    /// through its tag, and an unchanged one under such a tag has no plan.
    #[test]
    fn number_registers_are_planned_one_by_one() {
        let compiled = compile_trace(&wide(), OptimizationTier::Baseline).unwrap();
        let descriptor = &compiled.exits()[0];
        let lowered = compiled.lowered_member(descriptor.member);
        let snapshot = &lowered.snapshots[descriptor.snapshot.index()];
        let integer = RegisterEntry::Const(TraceScalar::Boolean(true));

        let mut tagged = snapshot.clone();
        tagged.numbers = vec![NumberRegister {
            register: 3,
            is_integer: integer,
        }];
        let plan = ExitRestore::new(lowered, &tagged, descriptor, REGISTERS, &[]).unwrap();
        assert!(plan.floats.is_empty());
        assert_eq!(plan.nils.as_ref(), &[4]);
        assert!(matches!(
            plan.changes.as_ref(),
            [(3, RegisterAction::IntegerOfFloat(_))]
        ));

        let mut stale = snapshot.clone();
        stale.numbers = vec![NumberRegister {
            register: 1,
            is_integer: integer,
        }];
        assert!(ExitRestore::new(lowered, &stale, descriptor, REGISTERS, &[]).is_none());
    }
}
