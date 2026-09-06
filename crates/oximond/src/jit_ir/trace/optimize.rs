//! Safe trace optimization.
//!
//! Every pass takes a verified [`TraceIr`] and returns an equivalent one. Two
//! traces are equivalent when, for every entry state, they commit the same
//! effects in the same order and every guard that fails restores the same
//! interpreter state. A pass may only drop a guard that cannot fail, and may
//! only drop a value that nothing, including a recovery snapshot, reads.
//!
//! The passes, in the order the optimized tier runs them:
//!
//! 1. Copy elimination removes `Move`s and forwards their source.
//! 2. Constant folding evaluates operations on known scalars through the
//!    shared numeric semantics in [`super::semantics`], so NaN payloads,
//!    signed zero, floor modulo, power and integer wrap-around match the
//!    evaluator and the native code. Integer identities (`x + 0`, `x * 1`,
//!    `x - x`) fold too. Float identities do not: `x + 0.0` is not `x` for
//!    `-0.0`, and `x * 1.0` may quiet a signalling NaN.
//! 3. Redundant guard elimination drops a type guard on a value whose type is
//!    already known, a repeat of an earlier guard on the same value, and a
//!    branch guard whose condition is a constant or was already guarded.
//! 4. Local value numbering merges identical pure operations. It never
//!    merges reads: that is the next pass's job, with the proof it needs.
//! 5. Store-to-load forwarding (only with [`OptimizeOptions::memory`]) removes
//!    a table load that an earlier store or load in the same segment already
//!    answers. A segment is the preheader or the body, and each starts with an
//!    empty cache, so no fact crosses the loop-back edge or reaches a side
//!    trace. A fact is keyed by the load kind, the table value and the key
//!    (an array index value, or a field's key string). Stores, their guards
//!    and their effects stay where they are. Only the later load goes, and its
//!    result becomes the stored value or the earlier load's result, in
//!    snapshots and loop-back arguments too. A store forwards only to a load
//!    whose expected type is the stored value's exact type, and never to one
//!    expecting `Nil` or `Table`. Nothing assumes two table values, bindings or
//!    registers name different tables, since an `__index` holder lets two
//!    bindings reach one slot. A store invalidates by key. An array store to
//!    key K drops every array fact whose key may equal K, on every table. Two
//!    keys differ only when both are non-NaN numeric constants with different
//!    values (`1` and `1.0` are one key, as are `0.0` and `-0.0`). A field
//!    store drops the field facts with the same key string on every table,
//!    and all array facts. A `LaneArrayLoad` keeps an array fact of its lane
//!    value with no binding, which array stores kill by key like any other.
//!    `ArraySet`, effects, calls and every operation this
//!    pass does not list drop everything, so a new operation is safe until it
//!    is taught here.
//! 6. Dead code elimination keeps guards, reads and effects, then every value
//!    they use, every value a snapshot reads, and every loop-back argument. A
//!    value only a snapshot reads stays live.
//! 7. Loop-invariant code motion moves pure operations whose operands are
//!    constants, entry inputs or loop parameters the loop-back passes on
//!    unchanged into the preheader. Entry inputs are read once
//!    and no iteration rewrites them, so the IR itself proves the invariance:
//!    a loop-carried value is a loop parameter and is never invariant. A
//!    branch guard on an invariant condition moves there too. It then leaves
//!    through a snapshot of the entry state, before the iteration runs, and
//!    the interpreter executes the iteration itself. A root's entry state is
//!    the interpreter's, so that snapshot leaves its registers alone. A side
//!    with a loop of its own (a member past the root, see
//!    [`TreeWrites::for_member`]) starts from its parent's native state,
//!    which the interpreter's registers do not hold, so its entry snapshot
//!    names every entry input and every loop parameter's first value. A side
//!    that starts inside inlined calls hoists no guard: its loop header is not
//!    its entry.
//!    With [`OptimizeOptions::memory`] the table reads move too, when the
//!    tree's [`TreeWrites`] prove nothing in the tree writes what they read:
//!    `FieldLoad` and `TableLength` as they are, and `ArrayLoad`,
//!    `LaneArrayLoad` and `TableFieldLoad` (not expecting `Nil`) only with
//!    `hoist_guards`, leaving
//!    through the entry-state snapshot like a hoisted branch guard. The table
//!    and the key must be invariant. A field read is blocked by a field store
//!    of the same key string in any member. An array read is blocked by any
//!    array store whose key may equal its key (the numeric-constant rule of
//!    pass 5), and by any field store. An `ArraySet`, an effect, a call or an
//!    unrecognised operation in any member blocks every read. Nothing relies
//!    on the preheader running again after a link: no member writes the
//!    location, so the value read once stays right.
//!
//! The baseline tier runs the cheap subset: copy elimination, redundant guard
//! elimination and dead code elimination. In debug builds the verifier runs
//! before the first pass and after each one.

use super::helper::{HelperEffects, HelperId, HelperType};
use super::ir::{
    BindingKind, EffectId, EntryContract, EntryInput, ExternalSlot, FrameInput, GuardId, GuardInfo,
    HostFunction, LoopParam, LoopParamInit, MAX_HOST_ARGS, SnapshotId, TraceInstruction, TraceIr,
    TraceOp, TraceScalar, TraceSite, TraceTerminator, TraceType, ValueDef, ValueId, ValueInfo,
};
use super::ir::{NumericOp, WideConversion};
use super::semantics;
use super::snapshot::{
    RegisterEntry, SliceId, SliceRecipe, Snapshot, SnapshotFrame, SnapshotPhase,
};
use crate::compat::TracingOptFlags;
use crate::jit_ir::Comparison;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::borrow::Cow;
use std::sync::Arc;

/// How much work the optimizer may do.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimizeLevel {
    /// The cheap subset: copies, redundant guards, dead code.
    Baseline,
    /// Every pass.
    Optimized,
}

/// What the optimizer is allowed to do. [`OptimizeOptions::for_level`] gives
/// the defaults of a level.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OptimizeOptions {
    pub level: OptimizeLevel,
    /// Move branch guards on invariant conditions into the preheader. Such a
    /// guard exits earlier than in the source trace, at the entry state, so
    /// the restored state differs from the source trace's even though both
    /// are valid interpreter states.
    pub hoist_guards: bool,
    /// Run the memory passes: loads that an earlier store or load in the same
    /// segment already answers. Off, the trace keeps every load, which makes
    /// the passes easy to measure against each other.
    pub memory: bool,
}

impl OptimizeOptions {
    /// The defaults of `level`.
    #[must_use]
    pub const fn for_level(level: OptimizeLevel) -> Self {
        Self {
            level,
            hoist_guards: matches!(level, OptimizeLevel::Optimized),
            memory: matches!(level, OptimizeLevel::Optimized),
        }
    }

    /// The defaults of `level`, less the optimizations `flags` leaves out.
    #[must_use]
    pub const fn for_level_and_flags(level: OptimizeLevel, flags: TracingOptFlags) -> Self {
        let defaults = Self::for_level(level);
        Self {
            level,
            hoist_guards: defaults.hoist_guards && flags.contains(TracingOptFlags::HOIST_GUARDS),
            memory: defaults.memory && flags.contains(TracingOptFlags::MEMORY),
        }
    }
}

/// How often each pass changed the trace.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OptimizeStats {
    pub copies_removed: u32,
    pub folded: u32,
    pub guards_removed: u32,
    pub merged: u32,
    pub dead_removed: u32,
    pub hoisted: u32,
    pub guards_hoisted: u32,
    /// Loads replaced by the value an earlier store or load of the same
    /// location left behind.
    pub loads_forwarded: u32,
    /// Loads merged into an identical earlier load.
    pub loads_merged: u32,
    /// Table reads moved into the preheader.
    pub loads_hoisted: u32,
}

impl OptimizeStats {
    /// The counters in a fixed order, for summing.
    const fn fields_mut(&mut self) -> [&mut u32; 10] {
        [
            &mut self.copies_removed,
            &mut self.folded,
            &mut self.guards_removed,
            &mut self.merged,
            &mut self.dead_removed,
            &mut self.hoisted,
            &mut self.guards_hoisted,
            &mut self.loads_forwarded,
            &mut self.loads_merged,
            &mut self.loads_hoisted,
        ]
    }

    /// Total number of changes.
    #[must_use]
    pub fn total(&self) -> u32 {
        let mut copy = *self;
        copy.fields_mut().into_iter().map(|field| *field).sum()
    }

    /// Adds `other`'s counters to these, saturating. For callers that sum the
    /// counters of many traces.
    pub fn accumulate(&mut self, other: Self) {
        self.combine(other, u32::saturating_add);
    }

    fn combine(&mut self, mut other: Self, op: fn(u32, u32) -> u32) {
        for (mine, theirs) in self.fields_mut().into_iter().zip(other.fields_mut()) {
            *mine = op(*mine, *theirs);
        }
    }
}

/// What the optimizer of one member reads of the tree's [`TreeWrites`] (see
/// [`TreeWrites::read_by`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemberWrites {
    /// The origins of the member's guards that sides continue, sorted.
    continued: Vec<u32>,
    /// Whether the member calls the substring helper, whose sinking reads
    /// the continued guards ([`sink_slices`]).
    slices: bool,
    /// For a loop, what its hoisting reads; `None` for a member that does
    /// not loop.
    looped: Option<LoopWrites>,
}

impl MemberWrites {
    /// Whether the member that optimized to `optimized` at `level` while it
    /// read `self` optimizes to the same trace when it reads `now`. Both
    /// views must be of the same member.
    ///
    /// Only two passes read the summary. Substring sinking reads the
    /// continued guards, so a member with a substring call takes the same
    /// result only from the same view. Hoisting reads all of it, but runs
    /// only at the Optimized level and only in a loop, and it is the last
    /// pass. A guard that a side now continues and did not before stays
    /// where it is instead of moving to the preheader, and a guard hoisting
    /// left in the body it would not have moved anyway: every other decision
    /// of the pass is the same then. So a loop takes the result when the tree
    /// writes look alike to it, every guard continued before still is, and
    /// none of the guards continued since sits in the result's preheader.
    #[must_use]
    pub fn optimizes_alike(&self, level: OptimizeLevel, optimized: &TraceIr, now: &Self) -> bool {
        debug_assert_eq!(
            (self.slices, self.looped.is_some()),
            (now.slices, now.looped.is_some()),
            "views of different members"
        );
        if self.slices {
            return self == now;
        }
        if level == OptimizeLevel::Baseline || self.looped.is_none() {
            return true;
        }
        if self.looped != now.looped {
            return false;
        }
        let continued = |origin: &u32| self.continued.binary_search(origin).is_ok();
        self.continued
            .iter()
            .all(|origin| now.continued.binary_search(origin).is_ok())
            && now
                .continued
                .iter()
                .filter(|origin| !continued(origin))
                .all(|origin| {
                    !optimized.preheader.iter().any(|instruction| {
                        instruction.op.guard().is_some_and(|guard| {
                            optimized
                                .guards
                                .get(guard.index())
                                .is_none_or(|info| info.origin == *origin)
                        })
                    })
                })
    }
}

// Each switch is one fact of the tree's writes that hoisting reads.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug, Eq)]
struct LoopWrites {
    /// Whether the member is a side (a side that starts in inlined calls
    /// keeps its guards).
    side: bool,
    unknown: bool,
    any_array_key: bool,
    /// Whether the tree stores any field: that keeps array reads too.
    writes_fields: bool,
    /// Whether some member may change a table's metatable: that keeps a
    /// metatable class guard in the loop.
    metatables: bool,
    /// The keys the tree stores that the member reads, sorted.
    read_fields: Vec<Arc<[u8]>>,
    /// The array keys the tree stores, each once.
    array_keys: Vec<FactKey>,
}

impl PartialEq for LoopWrites {
    fn eq(&self, other: &Self) -> bool {
        self.side == other.side
            && self.unknown == other.unknown
            && self.any_array_key == other.any_array_key
            && self.writes_fields == other.writes_fields
            && self.metatables == other.metatables
            && self.read_fields == other.read_fields
            && self.array_keys.len() == other.array_keys.len()
            && self
                .array_keys
                .iter()
                .all(|key| other.array_keys.contains(key))
    }
}

/// What every member of a tree writes, taken from the members before any
/// optimization. It lets one member's optimizer know about the stores of the
/// others: a side trace's store reaches the root's loop through a link.
///
/// Anything the summary does not list as a harmless operation counts as an
/// unknown write, so an operation added later is safe until it is taught here.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TreeWrites {
    /// The key strings of every field store.
    fields: Vec<Arc<[u8]>>,
    /// The normalised numeric constant keys of the array stores.
    array_keys: Vec<FactKey>,
    /// Whether an array store has a key that may equal any key.
    any_array_key: bool,
    /// Whether some member holds an `ArraySet`, an effect, a call or an
    /// operation this pass does not recognise.
    unknown: bool,
    /// Whether some member may change which metatable a table has: a call
    /// of a helper that does ([`HelperEffects::WRITES_METATABLE`]) or that
    /// runs other code ([`HelperEffects::resumes`]), or an operation that
    /// stands for code the trace did not lower.
    metatables: bool,
    /// The guards that side traces continue, as the member of the tree that
    /// holds the guard and its [`GuardInfo::origin`].
    continued: Vec<(usize, u32)>,
    /// The member of the tree the summary is read for (see
    /// [`Self::for_member`]).
    member: usize,
}

impl TreeWrites {
    /// The summary of `members`: the root and every side trace of a tree.
    #[must_use]
    pub fn of_members<'a>(members: impl IntoIterator<Item = &'a TraceIr>) -> Self {
        let mut writes = Self::default();
        for member in members {
            writes.add_member(member);
        }
        writes
    }

    /// The summary with the guards that a side trace of the tree continues,
    /// as (member, [`GuardInfo::origin`]) pairs. A guard that is hoisted fails
    /// once, at entry, where no side can take it, so the optimizer leaves
    /// these in the body.
    #[must_use]
    pub fn with_continued_guards(mut self, guards: impl IntoIterator<Item = (usize, u32)>) -> Self {
        self.continued.extend(guards);
        self
    }

    /// The summary as the optimizer of `member` reads it.
    #[must_use]
    pub fn for_member(&self, member: usize) -> Self {
        Self {
            member,
            ..self.clone()
        }
    }

    /// What the optimizer of `ir`, the member this summary is read for (see
    /// [`Self::for_member`]), may read of it. A tree compiled again can keep
    /// the earlier result of a member whose view did not change in a way
    /// that matters (see [`MemberWrites::optimizes_alike`]).
    ///
    /// The optimizer asks the summary two things. Which guards of the member
    /// sides continue (substring sinking and hoisting leave those alone), and,
    /// when the member is a loop, what the tree writes, for the reads that may
    /// move to the preheader. A member without a loop does not hoist, so it
    /// does not care what the tree writes, and a loop cares only about the
    /// field keys it reads itself.
    #[must_use]
    pub fn read_by(&self, ir: &TraceIr) -> MemberWrites {
        let mut continued: Vec<u32> = self
            .continued
            .iter()
            .filter(|(member, _)| *member == self.member)
            .map(|(_, origin)| *origin)
            .collect();
        continued.sort_unstable();
        continued.dedup();
        let looped = matches!(ir.terminator, TraceTerminator::LoopBack { .. }).then(|| {
            // Every key the loop may read a field by. The optimizer keeps the
            // member's bindings and fields as they are, so the keys of its
            // reads stay among these through every pass.
            let reads = |key: &[u8]| {
                ir.fields.iter().any(|field| *field.key == *key)
                    || ir.bindings.iter().any(|binding| {
                        matches!(&binding.kind, BindingKind::Field { key: read, .. } if **read == *key)
                    })
            };
            let mut read_fields: Vec<Arc<[u8]>> = self
                .fields
                .iter()
                .filter(|key| reads(key))
                .cloned()
                .collect();
            read_fields.sort_unstable();
            read_fields.dedup();
            let mut array_keys: Vec<FactKey> = Vec::with_capacity(self.array_keys.len());
            for key in &self.array_keys {
                if !array_keys.contains(key) {
                    array_keys.push(key.clone());
                }
            }
            LoopWrites {
                side: self.member > 0,
                unknown: self.unknown,
                any_array_key: self.any_array_key,
                writes_fields: !self.fields.is_empty(),
                metatables: self.metatables,
                read_fields,
                array_keys,
            }
        });
        let slices = ir.body.iter().any(|instruction| {
            matches!(
                instruction.op,
                TraceOp::CallHelper {
                    helper: HelperId::STRING_SLICE,
                    ..
                }
            )
        });
        MemberWrites {
            continued,
            slices,
            looped,
        }
    }

    /// Whether a side trace continues the guard of this member with this
    /// origin.
    fn continues(&self, origin: u32) -> bool {
        self.continued.contains(&(self.member, origin))
    }

    fn add_member(&mut self, ir: &TraceIr) {
        // Constants and copies of them, to normalise array store keys.
        let mut constants: Vec<Option<TraceScalar>> = vec![None; ir.values.len()];
        for instruction in ir.preheader.iter().chain(&ir.body) {
            match &instruction.op {
                TraceOp::Constant(scalar) => {
                    if let Some(result) = instruction.result {
                        constants[result.index()] = Some(*scalar);
                    }
                }
                TraceOp::Move { source } => {
                    if let Some(result) = instruction.result {
                        constants[result.index()] = constants[source.index()];
                    }
                }
                TraceOp::ArrayStore { key, .. }
                | TraceOp::ArrayOverwrite { key, .. }
                | TraceOp::LaneArrayStore { key, .. } => {
                    match numeric_key(constants[key.index()]) {
                        Some(key) => self.array_keys.push(key),
                        None => self.any_array_key = true,
                    }
                }
                TraceOp::FieldStore { binding, .. } => match &ir.bindings[*binding as usize].kind {
                    BindingKind::Field { key, .. } => self.fields.push(key.clone()),
                    _ => self.unknown = true,
                },
                TraceOp::TableFieldStore { field, .. } => {
                    self.fields.push(ir.fields[*field as usize].key.clone());
                }
                // Pure operations, guards and reads change no stored
                // location.
                TraceOp::Negate { .. }
                | TraceOp::IntToFloat { .. }
                | TraceOp::Numeric { .. }
                | TraceOp::ToWide { .. }
                | TraceOp::NativeUnary { .. }
                | TraceOp::NativeBinary { .. }
                | TraceOp::Arithmetic { .. }
                | TraceOp::Compare { .. }
                | TraceOp::Select { .. }
                | TraceOp::Truthy { .. }
                | TraceOp::Not { .. }
                | TraceOp::GuardType { .. }
                | TraceOp::GuardBranch { .. }
                | TraceOp::StringLength { .. }
                | TraceOp::StringByte { .. }
                | TraceOp::HostCall { .. }
                | TraceOp::HelperResult { .. }
                | TraceOp::ReleaseStrings { .. }
                | TraceOp::TableLength { .. }
                | TraceOp::ArrayLoad { .. }
                | TraceOp::LaneArrayLoad { .. }
                | TraceOp::NumberLoad { .. }
                | TraceOp::NumberTag { .. }
                | TraceOp::NumberModulo { .. }
                | TraceOp::BoolAnd { .. }
                | TraceOp::ArrayNext { .. }
                | TraceOp::FieldLoad { .. }
                | TraceOp::MetaGuard { .. }
                | TraceOp::TableFieldLoad { .. } => {}
                // A helper that only reads or allocates strings writes no
                // table; one that writes may write any.
                TraceOp::CallHelper { helper, .. } => {
                    let effects = helper.effects();
                    if effects.contains(HelperEffects::WRITES_HEAP) {
                        self.unknown = true;
                    }
                    if effects.contains(HelperEffects::WRITES_METATABLE) || effects.resumes() {
                        self.metatables = true;
                    }
                }
                // A table the trace does without is a count the run owes:
                // it touches no heap (an exit that builds the table has left
                // the loop, and the preheader runs again at the next entry).
                TraceOp::PhantomTable { .. } => {}
                // A raw store and a lookup may write or move storage, but
                // they change the metatable of no table that exists.
                TraceOp::ArraySet { .. } | TraceOp::HashLoad { .. } => {
                    self.unknown = true;
                }
                // An effect may be a call. Anything else is unknown. That
                // includes `TraceOp::Read`, on purpose: it stands for a table
                // access the trace did not lower, so it may write anything.
                _ => {
                    self.unknown = true;
                    self.metatables = true;
                }
            }
        }
    }

    /// Whether some member may write anywhere, so no read can move.
    #[must_use]
    pub const fn has_unknown_write(&self) -> bool {
        self.unknown
    }

    /// Whether everything `other` writes is something this summary already
    /// lists, so a member that writes no more than `other` can join the tree
    /// without invalidating a read the members here moved or forwarded.
    #[must_use]
    pub fn covers(&self, other: &Self) -> bool {
        if other.metatables && !self.metatables {
            return false;
        }
        if self.unknown {
            return true;
        }
        !other.unknown
            && (self.any_array_key || !other.any_array_key)
            && other.fields.iter().all(|key| self.fields.contains(key))
            && (self.any_array_key
                || other
                    .array_keys
                    .iter()
                    .all(|key| self.array_keys.contains(key)))
    }

    /// Whether no member writes a table at all.
    #[must_use]
    pub fn writes_nothing(&self) -> bool {
        !self.unknown && self.fields.is_empty() && self.array_keys.is_empty() && !self.any_array_key
    }

    /// Whether some member may change which metatable a table has.
    const fn changes_metatables(&self) -> bool {
        self.metatables
    }

    /// Whether a field read of `key` may see a write of the tree.
    fn writes_field(&self, key: &[u8]) -> bool {
        self.unknown || self.fields.iter().any(|stored| **stored == *key)
    }

    /// Whether an array read with the (normalised, if constant) `key` may see
    /// a write of the tree. A field store blocks it too, as it kills array
    /// facts in forwarding.
    fn writes_array(&self, key: Option<&FactKey>) -> bool {
        if self.unknown || self.any_array_key || !self.fields.is_empty() {
            return true;
        }
        match key {
            None => !self.array_keys.is_empty(),
            Some(key) => self.array_keys.iter().any(|stored| !stored.differs(key)),
        }
    }
}

/// Rounds of the cleanup passes before hoisting. Each round only runs while
/// the previous one changed something.
const MAX_ROUNDS: usize = 4;

/// Optimizes `ir` at `level`. Returns the trace unchanged if it does not
/// verify, or if an optimized trace fails verification in a release build, so
/// a caller can always lower the result. Debug builds panic on that, since it
/// is an optimizer bug.
#[must_use]
pub fn optimize_trace(ir: &TraceIr, level: OptimizeLevel) -> (TraceIr, OptimizeStats) {
    optimize_with(ir, OptimizeOptions::for_level(level))
}

/// [`optimize_trace`] with explicit options.
#[must_use]
pub fn optimize_with(ir: &TraceIr, options: OptimizeOptions) -> (TraceIr, OptimizeStats) {
    optimize_with_tree(ir, options, &TreeWrites::of_members([ir]))
}

/// Optimizes one member of a tree at `level`, with the optimizations
/// `flags` allows. `writes` summarises every
/// member of the tree, root and sides, taken before any of them was
/// optimized (see [`TreeWrites::of_members`]).
///
/// The member must verify: a tree is verified as it grows (see
/// `compile_with_plan` in `jit/trace/compile.rs`), so only debug builds
/// verify it again here. That check was a fifth of the optimizer's time on
/// the Luanti emerge. The optimized trace is verified in every build.
#[must_use]
pub fn optimize_tree_member(
    ir: &TraceIr,
    level: OptimizeLevel,
    flags: TracingOptFlags,
    writes: &TreeWrites,
) -> (TraceIr, OptimizeStats) {
    optimize_pipeline(
        ir,
        OptimizeOptions::for_level_and_flags(level, flags),
        writes,
        false,
        cfg!(debug_assertions),
    )
}

/// [`optimize_tree_member`] with explicit options.
#[must_use]
pub fn optimize_with_tree(
    ir: &TraceIr,
    options: OptimizeOptions,
    writes: &TreeWrites,
) -> (TraceIr, OptimizeStats) {
    optimize_pipeline(ir, options, writes, false, true)
}

/// One pass over a draft: it edits the draft in place and counts what it
/// changed in the stats. A pass that counts nothing has changed nothing a
/// finished trace would show.
type Pass<'p> = &'p dyn Fn(&mut Draft<'_>, &mut OptimizeStats);

/// The pipeline of [`optimize_with_tree`]. Every pass edits one draft in
/// place, and the draft is finished once, after the last pass.
///
/// `always_rebuild` runs the pipeline as it once ran: each pass gets a draft
/// of its own, built from the finished trace of the pass before it. It is
/// the reference the single-draft pipeline is checked against, by the tests
/// and, for every trace, by debug builds. `check_input` verifies `ir` first
/// and returns it unchanged when it does not verify.
fn optimize_pipeline(
    ir: &TraceIr,
    options: OptimizeOptions,
    writes: &TreeWrites,
    always_rebuild: bool,
    check_input: bool,
) -> (TraceIr, OptimizeStats) {
    if check_input && ir.verify().is_err() {
        return (ir.clone(), OptimizeStats::default());
    }
    let mut stats = OptimizeStats::default();
    // Load forwarding runs only on a trace with table loads. No pass adds
    // one, so the source answers for every round.
    let has_loads = has_table_loads(ir);
    let current = if always_rebuild {
        // Draft::finish renumbers the values and drops unreferenced
        // snapshots, so every pass sees a normal form.
        let mut current = Draft::new(ir).finish();
        run_passes(&options, writes, has_loads, &mut |name, pass| {
            let mut local = OptimizeStats::default();
            let mut draft = Draft::new(&current);
            pass(&mut draft, &mut local);
            // A pass that counted nothing leaves the trace as it was.
            let next = if local.total() == 0 {
                Draft::new(&current).finish()
            } else {
                draft.finish()
            };
            if cfg!(debug_assertions)
                && let Err(error) = next.verify_optimized()
            {
                panic!("optimizer pass {name} produced an invalid trace: {error}\n{next:#?}");
            }
            current = next;
            let changes = local.total();
            stats.accumulate(local);
            changes
        });
        current
    } else {
        let mut draft = icount("draft", || Draft::new(ir));
        run_passes(&options, writes, has_loads, &mut |name, pass| {
            let mut local = OptimizeStats::default();
            icount(name, || pass(&mut draft, &mut local));
            let changes = local.total();
            stats.accumulate(local);
            changes
        });
        icount("finish", || draft.finish())
    };
    if icount("verify", || current.verify_optimized()).is_err() {
        return (ir.clone(), OptimizeStats::default());
    }
    #[cfg(debug_assertions)]
    if !always_rebuild {
        let (reference, mut reference_stats) = optimize_pipeline(ir, options, writes, true, false);
        // A loop whose body emptied gets one dead constant from every
        // `finish`, which the rebuilding pipeline's next round removes and
        // counts again. The single draft is padded once, at the end.
        if matches!(
            current.body.as_slice(),
            [TraceInstruction {
                op: TraceOp::Constant(TraceScalar::Boolean(false)),
                ..
            }]
        ) {
            reference_stats.dead_removed = stats.dead_removed;
        }
        assert!(
            current == reference && stats == reference_stats,
            "the single-draft pipeline differs from the one that rebuilds after every pass \
             ({stats:?} against {reference_stats:?})\n\
             single draft: {current:#?}\nrebuilt: {reference:#?}"
        );
    }
    (current, stats)
}

/// Runs `run`, and with `OXIMOND_JIT_ICOUNT` set prints the instructions it
/// retired as `jit-icount: pass backend={name}`.
fn icount<T>(name: &'static str, run: impl FnOnce() -> T) -> T {
    crate::jit::icount::around("pass", 0, run, |_| name)
}

/// The passes of `options.level` in order, each handed to `step`, which
/// returns how many changes the pass counted.
fn run_passes(
    options: &OptimizeOptions,
    writes: &TreeWrites,
    has_loads: bool,
    step: &mut dyn FnMut(&'static str, Pass<'_>) -> u32,
) {
    let sink = |draft: &mut Draft<'_>, stats: &mut OptimizeStats| {
        sink_slices_in(draft, writes, stats);
    };
    match options.level {
        OptimizeLevel::Baseline => {
            step("copy-elimination", &eliminate_copies_in);
            step("guard-elimination", &eliminate_guards_in);
            step("concat-length", &sink_concat_lengths_in);
            step("substrings", &sink);
            step("dead-code", &eliminate_dead_code_in);
        }
        OptimizeLevel::Optimized => {
            // Load forwarding runs only while it can still find something:
            // never on a trace without table loads, and not again after a
            // round in which it removed nothing.
            let mut forward = options.memory && has_loads;
            for _ in 0..MAX_ROUNDS {
                let mut changes = 0;
                changes += step("copy-elimination", &eliminate_copies_in);
                changes += step("constant-folding", &fold_constants_in);
                changes += step("guard-elimination", &eliminate_guards_in);
                if forward {
                    let removed = step("load-forwarding", &forward_loads_in);
                    forward = removed != 0;
                    changes += removed;
                }
                changes += step("value-numbering", &number_values_in);
                changes += step("concat-length", &sink_concat_lengths_in);
                changes += step("substrings", &sink);
                changes += step("dead-code", &eliminate_dead_code_in);
                if changes == 0 {
                    break;
                }
            }
            let hoist = |draft: &mut Draft<'_>, stats: &mut OptimizeStats| {
                hoist_invariants_in(draft, options, writes, stats);
            };
            step("invariant-code-motion", &hoist);
        }
    }
}

/// Runs `pass` over a draft of `ir` and finishes it, or returns `None` when
/// the pass counted no change. The tests run single passes through this.
#[cfg(test)]
fn standalone(
    ir: &TraceIr,
    stats: &mut OptimizeStats,
    pass: impl FnOnce(&mut Draft<'_>, &mut OptimizeStats),
) -> Option<TraceIr> {
    let mut draft = Draft::new(ir);
    pass(&mut draft, stats);
    (stats.total() != 0).then(|| draft.finish())
}

#[cfg(test)]
fn eliminate_copies(ir: &TraceIr, stats: &mut OptimizeStats) -> Option<TraceIr> {
    standalone(ir, stats, eliminate_copies_in)
}

#[cfg(test)]
fn fold_constants(ir: &TraceIr, stats: &mut OptimizeStats) -> Option<TraceIr> {
    standalone(ir, stats, fold_constants_in)
}

#[cfg(test)]
fn eliminate_guards(ir: &TraceIr, stats: &mut OptimizeStats) -> Option<TraceIr> {
    standalone(ir, stats, eliminate_guards_in)
}

#[cfg(test)]
fn number_values(ir: &TraceIr, stats: &mut OptimizeStats) -> Option<TraceIr> {
    standalone(ir, stats, number_values_in)
}

#[cfg(test)]
fn eliminate_dead_code(ir: &TraceIr, stats: &mut OptimizeStats) -> Option<TraceIr> {
    standalone(ir, stats, eliminate_dead_code_in)
}

#[cfg(test)]
fn hoist_invariants(
    ir: &TraceIr,
    options: &OptimizeOptions,
    writes: &TreeWrites,
    stats: &mut OptimizeStats,
) -> Option<TraceIr> {
    standalone(ir, stats, |draft, stats| {
        hoist_invariants_in(draft, options, writes, stats);
    })
}

/// One instruction and the guard record it carries, with value ids that are
/// still the source trace's. A pass edits items and aliases values; `finish`
/// renumbers everything densely.
#[derive(Clone)]
struct Item {
    site: TraceSite,
    result: Option<ValueId>,
    op: TraceOp,
    guard: Option<GuardInfo>,
}

struct Draft<'a> {
    src: &'a TraceIr,
    preheader: Vec<Item>,
    body: Vec<Item>,
    /// Each value's replacement. A value whose entry is itself stands for
    /// itself.
    repl: Vec<ValueId>,
    /// The constant a value is known to hold, if any. Snapshot entries that
    /// name such a value become `Const`, even when the defining instruction
    /// is gone.
    consts: Vec<Option<TraceScalar>>,
    /// The source's snapshots, borrowed until a pass writes one. `finish`
    /// clones only the snapshots the trace still references, so a pass that
    /// changes nothing never copies them.
    snapshots: Cow<'a, [Snapshot]>,
}

fn resolve(repl: &[ValueId], mut value: ValueId) -> ValueId {
    while repl[value.index()] != value {
        value = repl[value.index()];
    }
    value
}

fn visit_operands_mut(op: &mut TraceOp, visit: impl FnMut(&mut ValueId)) {
    op.operands_mut(visit);
}

impl<'a> Draft<'a> {
    fn new(src: &'a TraceIr) -> Self {
        let item = |instruction: &TraceInstruction| Item {
            site: instruction.site,
            result: instruction.result,
            op: instruction.op.clone(),
            guard: instruction
                .op
                .guard()
                .map(|guard| src.guards[guard.index()]),
        };
        let preheader: Vec<Item> = src.preheader.iter().map(item).collect();
        let body: Vec<Item> = src.body.iter().map(item).collect();
        let mut consts = vec![None; src.values.len()];
        for item in preheader.iter().chain(&body) {
            if let (Some(result), TraceOp::Constant(scalar)) = (item.result, &item.op) {
                consts[result.index()] = Some(*scalar);
            }
        }
        let repl = (0..src.values.len())
            .map(|index| ValueId(u32::try_from(index).expect("value count fits u32")))
            .collect();
        Self {
            src,
            preheader,
            body,
            repl,
            consts,
            snapshots: Cow::Borrowed(src.snapshots.as_slice()),
        }
    }

    fn ty(&self, value: ValueId) -> TraceType {
        self.src.values[value.index()].ty
    }

    fn rep(&self, value: ValueId) -> ValueId {
        resolve(&self.repl, value)
    }

    fn constant(&self, value: ValueId) -> Option<TraceScalar> {
        self.consts[self.rep(value).index()]
    }

    /// The constant a snapshot may name for `value`. A pinned closure is a
    /// constant of the trace, but no snapshot can hold one: it restores
    /// through the value that carries it.
    fn snapshot_constant(&self, value: ValueId) -> Option<TraceScalar> {
        self.constant(value)
            .filter(|scalar| !scalar.ty().is_opaque())
    }

    /// Makes every use of `from` a use of `to`.
    fn alias(&mut self, from: ValueId, to: ValueId) {
        let to = self.rep(to);
        debug_assert_ne!(from, to);
        self.repl[from.index()] = to;
        if self.consts[from.index()].is_none() {
            self.consts[from.index()] = self.consts[to.index()];
        }
    }

    /// Rewrites each operand to its representative.
    fn canonicalize(&self, item: &mut Item) {
        visit_operands_mut(&mut item.op, |value| *value = self.rep(*value));
    }

    /// Runs `visit` over every item in execution order, canonicalizing its
    /// operands first, and keeps the items it returns `true` for. Nothing is
    /// copied: the items stay where they are.
    fn rewrite(&mut self, mut visit: impl FnMut(&mut Self, &mut Item) -> bool) {
        let mut preheader = std::mem::take(&mut self.preheader);
        self.rewrite_items(&mut preheader, &mut visit);
        self.preheader = preheader;
        let mut body = std::mem::take(&mut self.body);
        self.rewrite_items(&mut body, &mut visit);
        self.body = body;
    }

    fn rewrite_items(
        &mut self,
        items: &mut Vec<Item>,
        visit: &mut impl FnMut(&mut Self, &mut Item) -> bool,
    ) {
        items.retain_mut(|item| {
            self.canonicalize(item);
            visit(self, item)
        });
    }

    /// How many operands, loop arguments and returned values name each
    /// value, by representative.
    fn operand_uses(&self) -> Vec<u32> {
        let mut uses = vec![0_u32; self.src.values.len()];
        for item in self.preheader.iter().chain(&self.body) {
            item.op.operands(|value| uses[self.rep(value).index()] += 1);
        }
        match &self.src.terminator {
            TraceTerminator::LoopBack { args } | TraceTerminator::TailCall { args, .. } => {
                for arg in args {
                    uses[self.rep(*arg).index()] += 1;
                }
            }
            TraceTerminator::Return { value: Some(value) } => {
                uses[self.rep(*value).index()] += 1;
            }
            _ => {}
        }
        uses
    }

    /// Which snapshots a guard, the poll or the terminator restores from.
    /// `finish` keeps exactly these.
    fn referenced_snapshots(&self) -> Vec<bool> {
        let mut referenced = vec![false; self.snapshots.len()];
        for item in self.preheader.iter().chain(&self.body) {
            if let Some(guard) = item.guard {
                referenced[guard.snapshot.index()] = true;
            }
        }
        if let Some(poll) = self.src.poll_snapshot {
            referenced[poll.index()] = true;
        }
        if let TraceTerminator::Exit { snapshot }
        | TraceTerminator::Link { snapshot }
        | TraceTerminator::LinkLoop { snapshot, .. } = &self.src.terminator
        {
            referenced[snapshot.index()] = true;
        }
        referenced
    }

    /// Builds the dense, verified-shape trace the draft describes.
    fn finish(mut self) -> TraceIr {
        let src = self.src;

        // Snapshots survive when a guard, the poll or the terminator uses
        // them. Ids stay in source order.
        let referenced = self.referenced_snapshots();
        let repl = std::mem::take(&mut self.repl);
        let repl = &repl;
        let mut snapshot_ids: Vec<Option<SnapshotId>> = vec![None; self.snapshots.len()];
        let mut next_snapshot = 0_u32;
        for (index, used) in referenced.iter().enumerate() {
            if *used {
                snapshot_ids[index] = Some(SnapshotId(next_snapshot));
                next_snapshot += 1;
            }
        }

        let mut values: Vec<ValueInfo> = Vec::with_capacity(src.values.len());
        let mut ids: Vec<Option<ValueId>> = vec![None; src.values.len()];
        let define = |values: &mut Vec<ValueInfo>,
                      ids: &mut [Option<ValueId>],
                      old: ValueId,
                      ty: TraceType,
                      def: ValueDef| {
            values.push(ValueInfo { ty, def });
            let id = ValueId(u32::try_from(values.len() - 1).expect("value count fits u32"));
            ids[old.index()] = Some(id);
            id
        };

        let mut entry = EntryContract::default();
        for (index, input) in src.entry.inputs.iter().enumerate() {
            let value = define(
                &mut values,
                &mut ids,
                input.value,
                input.ty,
                ValueDef::Input(index),
            );
            entry.inputs.push(EntryInput {
                register: input.register,
                ty: input.ty,
                value,
            });
        }
        let mut frame_inputs = Vec::with_capacity(src.frame_inputs.len());
        for (index, input) in src.frame_inputs.iter().enumerate() {
            let value = define(
                &mut values,
                &mut ids,
                input.value,
                input.ty,
                ValueDef::FrameInput(index),
            );
            frame_inputs.push(FrameInput { value, ..*input });
        }
        let mut loop_params = Vec::with_capacity(src.loop_params.len());
        for (index, param) in src.loop_params.iter().enumerate() {
            let value = define(
                &mut values,
                &mut ids,
                param.value,
                param.ty,
                ValueDef::LoopParam(index),
            );
            loop_params.push(LoopParam {
                value,
                ty: param.ty,
                init: param.init,
                register: param.register,
            });
        }
        for param in &mut loop_params {
            if let LoopParamInit::Input(old) = param.init {
                let new = ids[resolve(repl, old).index()].expect("an entry input is defined");
                param.init = LoopParamInit::Input(new);
            }
        }

        for (index, item) in self.preheader.iter().enumerate() {
            if let Some(old) = item.result {
                define(
                    &mut values,
                    &mut ids,
                    old,
                    src.values[old.index()].ty,
                    ValueDef::Preheader(index),
                );
            }
        }
        for (index, item) in self.body.iter().enumerate() {
            if let Some(old) = item.result {
                define(
                    &mut values,
                    &mut ids,
                    old,
                    src.values[old.index()].ty,
                    ValueDef::Instruction(index),
                );
            }
        }
        // Resolve operands only after every definition has an id: a loop
        // argument or snapshot may name any of them.
        let new_id = |old: ValueId| -> ValueId {
            ids[resolve(repl, old).index()].expect("a live operand is defined")
        };

        let mut guards: Vec<GuardInfo> = Vec::new();
        let mut effects = 0_u32;
        let lower = |items: Vec<Item>, guards: &mut Vec<GuardInfo>, effects: &mut u32| {
            items
                .into_iter()
                .map(|item| {
                    let mut op = item.op;
                    visit_operands_mut(&mut op, |value| *value = new_id(*value));
                    match &mut op {
                        TraceOp::GuardType { guard, .. }
                        | TraceOp::GuardBranch { guard, .. }
                        | TraceOp::ArrayLoad { guard, .. }
                        | TraceOp::LaneArrayLoad { guard, .. }
                        | TraceOp::NumberLoad { guard, .. }
                        | TraceOp::ArrayNext { guard, .. }
                        | TraceOp::HashLoad { guard, .. }
                        | TraceOp::StringByte { guard, .. }
                        | TraceOp::HostCall { guard, .. }
                        | TraceOp::TableFieldLoad { guard, .. }
                        | TraceOp::MetaGuard { guard, .. }
                        | TraceOp::SelfCall { guard, .. } => {
                            let info = item.guard.expect("a guard item carries its record");
                            *guard =
                                GuardId(u32::try_from(guards.len()).expect("guard count fits u32"));
                            guards.push(GuardInfo {
                                snapshot: snapshot_ids[info.snapshot.index()]
                                    .expect("a used snapshot has an id"),
                                site: info.site,
                                origin: info.origin,
                            });
                        }
                        TraceOp::CallHelper { guard, effect, .. } => {
                            let info = item.guard.expect("a guard item carries its record");
                            *guard =
                                GuardId(u32::try_from(guards.len()).expect("guard count fits u32"));
                            guards.push(GuardInfo {
                                snapshot: snapshot_ids[info.snapshot.index()]
                                    .expect("a used snapshot has an id"),
                                site: info.site,
                                origin: info.origin,
                            });
                            if let Some(effect) = effect {
                                *effect = EffectId(*effects);
                                *effects += 1;
                            }
                        }
                        TraceOp::ArrayStore { guard, effect, .. }
                        | TraceOp::ArraySet { guard, effect, .. }
                        | TraceOp::TableFieldStore { guard, effect, .. }
                        | TraceOp::LaneArrayStore { guard, effect, .. } => {
                            let info = item.guard.expect("a guard item carries its record");
                            *guard =
                                GuardId(u32::try_from(guards.len()).expect("guard count fits u32"));
                            guards.push(GuardInfo {
                                snapshot: snapshot_ids[info.snapshot.index()]
                                    .expect("a used snapshot has an id"),
                                site: info.site,
                                origin: info.origin,
                            });
                            *effect = EffectId(*effects);
                            *effects += 1;
                        }
                        TraceOp::Effect { effect, .. }
                        | TraceOp::FieldStore { effect, .. }
                        | TraceOp::ArrayOverwrite { effect, .. } => {
                            *effect = EffectId(*effects);
                            *effects += 1;
                        }
                        _ => {}
                    }
                    TraceInstruction {
                        site: item.site,
                        result: item.result.map(new_id),
                        op,
                    }
                })
                .collect::<Vec<_>>()
        };
        let preheader = lower(
            std::mem::take(&mut self.preheader),
            &mut guards,
            &mut effects,
        );
        let mut body = lower(std::mem::take(&mut self.body), &mut guards, &mut effects);
        if body.is_empty()
            && let Some(poll) = src.poll_snapshot
        {
            // Every instruction was dead or moved out, but a trace the VM
            // polls needs something to loop over. One dead constant keeps the
            // loop well formed, and the backend drops it.
            let pc = self.snapshots[poll.index()].phase_pc();
            values.push(ValueInfo {
                ty: TraceType::Boolean,
                def: ValueDef::Instruction(0),
            });
            body.push(TraceInstruction {
                site: TraceSite {
                    frame: super::ir::FrameId(0),
                    pc,
                },
                result: Some(ValueId(
                    u32::try_from(values.len() - 1).expect("value count fits u32"),
                )),
                op: TraceOp::Constant(TraceScalar::Boolean(false)),
            });
        }

        // What a snapshot entry naming each source value becomes, worked out
        // once per value: snapshots name the same values many times over.
        let entry_of: Vec<Option<RegisterEntry>> = (0..src.values.len())
            .map(|index| {
                let value = resolve(
                    repl,
                    ValueId(u32::try_from(index).expect("value count fits u32")),
                );
                match self.consts[value.index()].filter(|scalar| !scalar.ty().is_opaque()) {
                    Some(scalar) => Some(RegisterEntry::Const(scalar)),
                    None => ids[value.index()].map(RegisterEntry::Value),
                }
            })
            .collect();
        let mut snapshots: Vec<Snapshot> = self
            .snapshots
            .iter()
            .enumerate()
            .filter_map(|(index, snapshot)| {
                let id = snapshot_ids[index]?;
                let mut snapshot = snapshot.clone();
                snapshot.id = id;
                Some(snapshot)
            })
            .collect();
        // One answer per value, so a run of a register map that snapshots
        // share is renamed once and stays shared.
        super::snapshot::rewrite_entries(&mut snapshots, |entry| {
            if let RegisterEntry::Value(value) = entry {
                *entry = entry_of[value.index()].expect("a live operand is defined");
            }
        });

        let terminator = match &src.terminator {
            TraceTerminator::LoopBack { args } => TraceTerminator::LoopBack {
                args: args.iter().map(|arg| new_id(*arg)).collect(),
            },
            TraceTerminator::Exit { snapshot } => TraceTerminator::Exit {
                snapshot: snapshot_ids[snapshot.index()].expect("the terminator's snapshot"),
            },
            TraceTerminator::Link { snapshot } => TraceTerminator::Link {
                snapshot: snapshot_ids[snapshot.index()].expect("the terminator's snapshot"),
            },
            TraceTerminator::LinkLoop { side, snapshot } => TraceTerminator::LinkLoop {
                side: *side,
                snapshot: snapshot_ids[snapshot.index()].expect("the terminator's snapshot"),
            },
            TraceTerminator::Return { value } => TraceTerminator::Return {
                value: value.map(new_id),
            },
            TraceTerminator::TailCall { args, pc } => TraceTerminator::TailCall {
                args: args.iter().map(|arg| new_id(*arg)).collect(),
                pc: *pc,
            },
        };
        TraceIr {
            id: src.id,
            frames: src.frames.clone(),
            pinned_closures: src.pinned_closures.clone(),
            cell_aliases: src.cell_aliases.clone(),
            root_slot_count: src.root_slot_count,
            loop_tail: src.loop_tail,
            function: src.function.clone(),
            root_constants: src.root_constants.clone(),
            external: src.external.clone(),
            entry,
            frame_inputs,
            seed_frames: src.seed_frames,
            bindings: src.bindings.clone(),
            fields: src.fields.clone(),
            loop_params,
            values,
            preheader,
            body,
            guards,
            snapshots,
            written_registers: src.written_registers.clone(),
            poll_snapshot: src
                .poll_snapshot
                .map(|poll| snapshot_ids[poll.index()].expect("the poll snapshot")),
            terminator,
            sides: src.sides.clone(),
            header_live: src.header_live.clone(),
            counted: src.counted,
            boxed_control: src.boxed_control,
            boxed_key: src.boxed_key,
            boxed_value: src.boxed_value,
            boxed_steps: src.boxed_steps,
            array_steps_plain: src.array_steps_plain,
            keep_branch_guards: src.keep_branch_guards,
        }
    }
}

/// Removes `Move`s and forwards their sources.
fn eliminate_copies_in(draft: &mut Draft<'_>, stats: &mut OptimizeStats) {
    let has_move = |item: &Item| matches!(item.op, TraceOp::Move { .. });
    if !draft.preheader.iter().any(has_move) && !draft.body.iter().any(has_move) {
        return;
    }
    draft.rewrite(|draft, item| {
        if let (TraceOp::Move { source }, Some(result)) = (&item.op, item.result) {
            draft.alias(result, *source);
            stats.copies_removed += 1;
            return false;
        }
        true
    });
}

/// What folding does to one operation.
enum Fold {
    Keep,
    Constant(TraceScalar),
    Alias(ValueId),
}

/// Folds operations whose operands are known, and applies exact identities.
fn fold_constants_in(draft: &mut Draft<'_>, stats: &mut OptimizeStats) {
    // Source of each kept negation, for `-(-x)`.
    let mut negations: HashMap<ValueId, ValueId> = HashMap::default();
    draft.rewrite(|draft, item| {
        let Some(result) = item.result else {
            return true;
        };
        let known = |value: ValueId| draft.constant(value);
        let outcome = match &item.op {
            TraceOp::Negate { source } => match known(*source) {
                Some(scalar) => semantics::negate(scalar).map_or(Fold::Keep, Fold::Constant),
                None => negations
                    .get(source)
                    .map_or(Fold::Keep, |inner| Fold::Alias(*inner)),
            },
            TraceOp::IntToFloat { source } => known(*source)
                .and_then(semantics::to_float)
                .map_or(Fold::Keep, Fold::Constant),
            TraceOp::Numeric { op, source } => known(*source)
                .and_then(|scalar| semantics::numeric(*op, scalar))
                .map_or(Fold::Keep, Fold::Constant),
            TraceOp::ToWide {
                source,
                target,
                conversion,
            } => known(*source)
                .and_then(|scalar| semantics::to_wide(scalar, *target, *conversion))
                .map_or(Fold::Keep, Fold::Constant),
            TraceOp::NativeUnary { function, source } => match known(*source) {
                Some(TraceScalar::Float(argument)) => {
                    Fold::Constant(TraceScalar::Float((function.0)(argument)))
                }
                _ => Fold::Keep,
            },
            TraceOp::NativeBinary {
                function,
                left,
                right,
            } => match (known(*left), known(*right)) {
                (Some(TraceScalar::Float(left)), Some(TraceScalar::Float(right))) => {
                    Fold::Constant(TraceScalar::Float((function.0)(left, right)))
                }
                _ => Fold::Keep,
            },
            TraceOp::Arithmetic { op, left, right } => match (known(*left), known(*right)) {
                (Some(a), Some(b)) => {
                    semantics::arithmetic(*op, a, b).map_or(Fold::Keep, Fold::Constant)
                }
                (a, b) => integer_identity(draft, *op, *left, *right, a, b),
            },
            TraceOp::Compare {
                comparison,
                left,
                right,
            } => match (known(*left), known(*right)) {
                (Some(a), Some(b)) => semantics::compare(*comparison, a, b)
                    .map_or(Fold::Keep, |flag| {
                        Fold::Constant(TraceScalar::Boolean(flag))
                    }),
                _ if left == right => same_operand_compare(draft.ty(*left), *comparison)
                    .map_or(Fold::Keep, |flag| {
                        Fold::Constant(TraceScalar::Boolean(flag))
                    }),
                _ => Fold::Keep,
            },
            TraceOp::Truthy { source } => match (draft.ty(*source), known(*source)) {
                (_, Some(scalar)) => Fold::Constant(TraceScalar::Boolean(scalar.is_truthy())),
                (TraceType::Nil, _) => Fold::Constant(TraceScalar::Boolean(false)),
                (
                    TraceType::Integer
                    | TraceType::Float
                    | TraceType::Int64
                    | TraceType::UInt64
                    | TraceType::Table
                    | TraceType::Closure
                    | TraceType::Thread
                    | TraceType::Userdata
                    | TraceType::String
                    | TraceType::StringRef,
                    _,
                ) => Fold::Constant(TraceScalar::Boolean(true)),
                (TraceType::Boolean, _) => Fold::Alias(*source),
                (TraceType::Any, _) => Fold::Keep,
            },
            TraceOp::Not { source } => match (draft.ty(*source), known(*source)) {
                (_, Some(scalar)) => Fold::Constant(TraceScalar::Boolean(!scalar.is_truthy())),
                (TraceType::Nil, _) => Fold::Constant(TraceScalar::Boolean(true)),
                (
                    TraceType::Integer
                    | TraceType::Float
                    | TraceType::Int64
                    | TraceType::UInt64
                    | TraceType::Table
                    | TraceType::Closure
                    | TraceType::Thread
                    | TraceType::Userdata
                    | TraceType::String
                    | TraceType::StringRef,
                    _,
                ) => Fold::Constant(TraceScalar::Boolean(false)),
                _ => Fold::Keep,
            },
            // A decided condition, or two equal arms, is one arm.
            TraceOp::Select {
                condition,
                if_true,
                if_false,
            } => match known(*condition) {
                Some(TraceScalar::Boolean(flag)) => {
                    Fold::Alias(if flag { *if_true } else { *if_false })
                }
                _ if if_true == if_false => Fold::Alias(*if_true),
                _ => Fold::Keep,
            },
            _ => Fold::Keep,
        };
        match outcome {
            Fold::Keep => {
                if let TraceOp::Negate { source } = &item.op {
                    negations.insert(result, *source);
                }
                true
            }
            Fold::Constant(scalar) => {
                if !matches!(item.op, TraceOp::Constant(_)) {
                    stats.folded += 1;
                }
                item.op = TraceOp::Constant(scalar);
                draft.consts[result.index()] = Some(scalar);
                true
            }
            Fold::Alias(to) => {
                stats.folded += 1;
                draft.alias(result, to);
                false
            }
        }
    });
}

/// `x + 0`, `x * 1`, `x - x` and friends, for integers only. Integer
/// arithmetic wraps and has no NaN or signed zero, so the identities are
/// exact.
fn integer_identity(
    draft: &Draft<'_>,
    op: super::ir::ArithmeticOp,
    left: ValueId,
    right: ValueId,
    a: Option<TraceScalar>,
    b: Option<TraceScalar>,
) -> Fold {
    use super::ir::ArithmeticOp::{Add, Multiply, Subtract};
    if draft.ty(left) != TraceType::Integer {
        return Fold::Keep;
    }
    let is =
        |scalar: Option<TraceScalar>, wanted: i64| scalar == Some(TraceScalar::Integer(wanted));
    match op {
        Add if is(b, 0) => Fold::Alias(left),
        Add if is(a, 0) => Fold::Alias(right),
        Subtract if is(b, 0) => Fold::Alias(left),
        Subtract if left == right => Fold::Constant(TraceScalar::Integer(0)),
        Multiply if is(b, 1) => Fold::Alias(left),
        Multiply if is(a, 1) => Fold::Alias(right),
        Multiply if is(a, 0) || is(b, 0) => Fold::Constant(TraceScalar::Integer(0)),
        _ => Fold::Keep,
    }
}

/// The value of `x op x` for a comparison, when it does not depend on `x`.
/// Floats are excluded: NaN is not equal to itself.
fn same_operand_compare(ty: TraceType, comparison: Comparison) -> Option<bool> {
    match ty {
        TraceType::Integer | TraceType::Int64 | TraceType::UInt64 => {
            Some(!matches!(comparison, Comparison::LessThan))
        }
        TraceType::Boolean
        | TraceType::Nil
        | TraceType::Table
        | TraceType::Closure
        | TraceType::Thread
        | TraceType::Userdata
        | TraceType::String => (comparison == Comparison::Equal).then_some(true),
        // A borrow is never compared (the verifier refuses it).
        TraceType::Float | TraceType::Any | TraceType::StringRef => None,
    }
}

/// Drops guards that cannot fail.
fn eliminate_guards_in(draft: &mut Draft<'_>, stats: &mut OptimizeStats) {
    // The value a guard on (source, type) produced.
    let mut typed: HashMap<(ValueId, TraceType), ValueId> = HashMap::default();
    // The outcome each guarded condition has already been held to.
    let mut branches: HashMap<ValueId, bool> = HashMap::default();
    // The tables already held to a metatable class. Only a call that changes
    // a metatable changes a row's metatable word, so the first guard of a
    // (table, class) pair answers for every later one until such a call.
    let mut classes: HashSet<(ValueId, u32)> = HashSet::default();
    draft.rewrite(|draft, item| match (&item.op, item.result) {
        (TraceOp::MetaGuard { table, binding, .. }, _) => {
            if !classes.insert((draft.rep(*table), *binding)) {
                stats.guards_removed += 1;
                return false;
            }
            true
        }
        // A call that changes a metatable ends what the guards before it
        // knew, and so does one that runs code that may (a host function or a
        // coroutine).
        (TraceOp::CallHelper { helper, .. }, _)
            if helper.effects().contains(HelperEffects::WRITES_METATABLE)
                || helper.effects().resumes() =>
        {
            classes.clear();
            true
        }
        (
            TraceOp::GuardType {
                source, expected, ..
            },
            Some(result),
        ) => {
            let (source, expected) = (*source, *expected);
            if draft.ty(source) != TraceType::Any {
                // A verified trace only guards a typed value with its own
                // type, so the guard passes.
                draft.alias(result, source);
                stats.guards_removed += 1;
                return false;
            }
            if let Some(earlier) = typed.get(&(source, expected)) {
                draft.alias(result, *earlier);
                stats.guards_removed += 1;
                return false;
            }
            typed.insert((source, expected), result);
            true
        }
        (
            TraceOp::GuardBranch {
                condition,
                expected,
                ..
            },
            _,
        ) => {
            let (condition, expected) = (*condition, *expected);
            let known = draft
                .constant(condition)
                .map(|scalar| scalar == TraceScalar::Boolean(true))
                .or_else(|| branches.get(&condition).copied());
            if known == Some(expected) {
                stats.guards_removed += 1;
                return false;
            }
            // A condition known to be the opposite always fails. Keep the
            // guard: the trace is unlikely, but still correct.
            branches.entry(condition).or_insert(expected);
            true
        }
        _ => true,
    });
}

/// Which table access a remembered fact describes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FactKind {
    Array,
    Field,
    TableField,
}

/// The key of a remembered location.
#[derive(Clone, Debug, Eq, PartialEq)]
enum FactKey {
    /// An integer constant array index, or a float constant with an integer
    /// value in `i64` range (`1.0` is `Int(1)`, and `-0.0` is `Int(0)`), as
    /// Lua normalises keys.
    Int(i64),
    /// A float constant with no integer value, by bits. It is never NaN.
    Float(u64),
    /// Any other array index, by value, NaN constants included. It may equal
    /// any other key.
    Value(ValueId),
    /// A field's key string.
    Name(std::sync::Arc<[u8]>),
}

impl FactKey {
    /// Whether two array keys are known to name different slots: only two
    /// different non-NaN numeric constants are. Constants are normalised, so
    /// unequal keys here are unequal numbers. Everything else may be equal.
    fn differs(&self, other: &Self) -> bool {
        let constant = |key: &Self| matches!(key, Self::Int(_) | Self::Float(_));
        constant(self) && constant(other) && self != other
    }
}

/// The binding a [`TraceOp::LaneArrayLoad`]'s fact names: none, since the
/// table is met by lane. No binding has this index.
const LANE_FACT_BINDING: u32 = u32::MAX;

/// What an earlier store or load of one location left behind.
struct Fact {
    kind: FactKind,
    binding: u32,
    table: ValueId,
    key: FactKey,
    value: ValueId,
    /// The type of `value`, which a load must expect exactly.
    ty: TraceType,
    /// Whether a store left it. A store never answers a load that expects
    /// `Nil` or `Table`.
    from_store: bool,
}

/// The facts of one segment.
#[derive(Default)]
struct Facts(Vec<Fact>);

impl Facts {
    fn find(&self, kind: FactKind, binding: u32, table: ValueId, key: &FactKey) -> Option<&Fact> {
        self.0.iter().find(|fact| {
            fact.kind == kind && fact.binding == binding && fact.table == table && fact.key == *key
        })
    }

    /// Drops the array facts whose key may equal `key`, on every table.
    fn kill_array(&mut self, key: &FactKey) {
        self.0
            .retain(|fact| fact.kind != FactKind::Array || fact.key.differs(key));
    }

    /// A field store of `name`: drops the field facts of that key string on
    /// every table, and every array fact.
    fn kill_field(&mut self, name: &[u8]) {
        self.0.retain(|fact| match (&fact.kind, &fact.key) {
            (FactKind::Array, _) => false,
            (_, FactKey::Name(other)) => **other != *name,
            _ => true,
        });
    }

    fn insert(&mut self, fact: Fact) {
        self.0.retain(|old| {
            !(old.kind == fact.kind
                && old.binding == fact.binding
                && old.table == fact.table
                && old.key == fact.key)
        });
        self.0.push(fact);
    }
}

/// Whether the preheader or body holds a load that [`forward_loads`] could
/// remove.
fn has_table_loads(ir: &TraceIr) -> bool {
    ir.preheader.iter().chain(&ir.body).any(|instruction| {
        matches!(
            instruction.op,
            TraceOp::ArrayLoad { .. }
                | TraceOp::LaneArrayLoad { .. }
                | TraceOp::FieldLoad { .. }
                | TraceOp::TableFieldLoad { .. }
        )
    })
}

/// Removes table loads that an earlier store or load of the same location in
/// the same segment answers. See the module docs for the rules.
fn forward_loads_in(draft: &mut Draft<'_>, stats: &mut OptimizeStats) {
    // Each segment starts with an empty cache.
    let mut preheader = std::mem::take(&mut draft.preheader);
    let mut facts = Facts::default();
    draft.rewrite_items(&mut preheader, &mut |draft, item| {
        forward_item(draft, &mut facts, stats, item)
    });
    draft.preheader = preheader;
    let mut body = std::mem::take(&mut draft.body);
    let mut facts = Facts::default();
    draft.rewrite_items(&mut body, &mut |draft, item| {
        forward_item(draft, &mut facts, stats, item)
    });
    draft.body = body;
}

/// The key of an array access: a numeric constant by its normalised value,
/// else by value id.
fn array_key(draft: &Draft<'_>, key: ValueId) -> FactKey {
    numeric_key(draft.constant(key)).unwrap_or(FactKey::Value(key))
}

/// The normalised key a numeric constant names, or `None` for anything that
/// may equal any key: a non-constant, a NaN or a constant of another type.
fn numeric_key(constant: Option<TraceScalar>) -> Option<FactKey> {
    Some(match constant {
        Some(TraceScalar::Integer(index)) => FactKey::Int(index),
        Some(TraceScalar::Float(value)) if !value.is_nan() => {
            // 2^63 is the first float above `i64::MAX`. The range check keeps
            // the cast exact.
            if value.fract() == 0.0
                && (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&value)
            {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "range-checked integral value"
                )]
                FactKey::Int(value as i64)
            } else {
                FactKey::Float(value.to_bits())
            }
        }
        _ => return None,
    })
}

fn forward_item(
    draft: &mut Draft<'_>,
    facts: &mut Facts,
    stats: &mut OptimizeStats,
    item: &Item,
) -> bool {
    // The load kind, binding, table, key and expected type, for a load.
    let load = match &item.op {
        TraceOp::ArrayLoad {
            binding,
            table,
            key,
            expected,
            ..
        } => Some((
            FactKind::Array,
            *binding,
            *table,
            array_key(draft, *key),
            *expected,
        )),
        // A table met by lane has no binding. Its facts are array facts, so
        // every array store (which may name the same table through a
        // binding or another lane) kills them by key, and anything that may
        // move an array kills them all.
        TraceOp::LaneArrayLoad {
            table,
            key,
            expected,
            ..
        } => Some((
            FactKind::Array,
            LANE_FACT_BINDING,
            *table,
            array_key(draft, *key),
            *expected,
        )),
        TraceOp::FieldLoad { binding, table } => {
            let BindingKind::Field { key, ty, .. } = &draft.src.bindings[*binding as usize].kind
            else {
                facts.0.clear();
                return true;
            };
            Some((
                FactKind::Field,
                *binding,
                *table,
                FactKey::Name(key.clone()),
                *ty,
            ))
        }
        TraceOp::TableFieldLoad { field, table, .. } => {
            let field_info = &draft.src.fields[*field as usize];
            Some((
                FactKind::TableField,
                *field,
                *table,
                FactKey::Name(field_info.key.clone()),
                field_info.ty,
            ))
        }
        _ => None,
    };
    if let (Some((kind, binding, table, key, expected)), Some(result)) = (load, item.result) {
        let known = facts.find(kind, binding, table, &key);
        let hit = known.and_then(|fact| {
            let answers = fact.ty == expected
                && (!fact.from_store || !matches!(expected, TraceType::Nil | TraceType::Table));
            answers.then_some((fact.value, fact.from_store))
        });
        if let Some((value, from_store)) = hit {
            if from_store {
                stats.loads_forwarded += 1;
            } else {
                stats.loads_merged += 1;
            }
            draft.alias(result, value);
            return false;
        }
        if known.is_none() {
            facts.insert(Fact {
                kind,
                binding,
                table,
                key,
                value: result,
                ty: expected,
                from_store: false,
            });
        }
        return true;
    }
    match &item.op {
        TraceOp::ArrayStore {
            binding,
            table,
            key,
            value,
            ..
        }
        | TraceOp::ArrayOverwrite {
            binding,
            table,
            key,
            value,
            ..
        } => {
            let key = array_key(draft, *key);
            facts.kill_array(&key);
            facts.insert(Fact {
                kind: FactKind::Array,
                binding: *binding,
                table: *table,
                key,
                value: *value,
                ty: draft.ty(*value),
                from_store: true,
            });
        }
        // A store into the array of a table met by lane kills the array
        // facts of its key on every table, as any array store does, and
        // answers a later lane read of the same table and key.
        TraceOp::LaneArrayStore {
            table, key, value, ..
        } => {
            let key = array_key(draft, *key);
            facts.kill_array(&key);
            facts.insert(Fact {
                kind: FactKind::Array,
                binding: LANE_FACT_BINDING,
                table: *table,
                key,
                value: *value,
                ty: draft.ty(*value),
                from_store: true,
            });
        }
        TraceOp::FieldStore {
            binding,
            table,
            value,
            ..
        } => {
            if let BindingKind::Field { key, .. } = &draft.src.bindings[*binding as usize].kind {
                facts.kill_field(key);
                facts.insert(Fact {
                    kind: FactKind::Field,
                    binding: *binding,
                    table: *table,
                    key: FactKey::Name(key.clone()),
                    value: *value,
                    ty: draft.ty(*value),
                    from_store: true,
                });
            } else {
                facts.0.clear();
            }
        }
        TraceOp::TableFieldStore {
            field,
            table,
            value,
            ..
        } => {
            let key = draft.src.fields[*field as usize].key.clone();
            facts.kill_field(&key);
            facts.insert(Fact {
                kind: FactKind::TableField,
                binding: *field,
                table: *table,
                key: FactKey::Name(key),
                value: *value,
                ty: draft.ty(*value),
                from_store: true,
            });
        }
        // Pure operations, guards and reads of immutable or constant data
        // change no stored location.
        TraceOp::Constant(_)
        | TraceOp::Move { .. }
        | TraceOp::Negate { .. }
        | TraceOp::IntToFloat { .. }
        | TraceOp::Numeric { .. }
        | TraceOp::ToWide { .. }
        | TraceOp::NativeUnary { .. }
        | TraceOp::NativeBinary { .. }
        | TraceOp::Arithmetic { .. }
        | TraceOp::Compare { .. }
        | TraceOp::Select { .. }
        | TraceOp::Truthy { .. }
        | TraceOp::Not { .. }
        | TraceOp::GuardType { .. }
        | TraceOp::GuardBranch { .. }
        | TraceOp::StringLength { .. }
        | TraceOp::StringByte { .. }
        | TraceOp::ArrayNext { .. }
        | TraceOp::NumberLoad { .. }
        | TraceOp::NumberTag { .. }
        | TraceOp::NumberModulo { .. }
        | TraceOp::BoolAnd { .. }
        | TraceOp::HostCall { .. }
        | TraceOp::HelperResult { .. }
        | TraceOp::ReleaseStrings { .. }
        | TraceOp::MetaGuard { .. }
        | TraceOp::TableLength { .. } => {}
        // A helper that does not write the heap is no barrier for cached
        // table reads, whatever else it reads or makes. One that writes
        // invalidates all of them: a helper's flags do not say which table.
        TraceOp::CallHelper { helper, .. } => {
            if helper.effects().contains(HelperEffects::WRITES_HEAP) {
                facts.0.clear();
            }
        }
        // ArraySet may move storage; an effect may be a call. Anything else
        // is unknown.
        _ => facts.0.clear(),
    }
    true
}

/// What an operation computes, for merging.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Key {
    Constant(u8, u64),
    Negate(ValueId),
    IntToFloat(ValueId),
    Numeric(NumericOp, ValueId),
    ToWide(ValueId, TraceType, u8),
    NativeUnary(usize, ValueId),
    NativeBinary(usize, ValueId, ValueId),
    Arithmetic(u8, ValueId, ValueId),
    Compare(u8, ValueId, ValueId),
    Truthy(ValueId),
    Not(ValueId),
    Select(ValueId, ValueId, ValueId),
}

fn scalar_key(scalar: TraceScalar) -> Key {
    match scalar {
        TraceScalar::Nil => Key::Constant(0, 0),
        TraceScalar::Boolean(flag) => Key::Constant(1, u64::from(flag)),
        TraceScalar::Integer(value) => Key::Constant(2, value.cast_unsigned()),
        TraceScalar::Float(value) => Key::Constant(3, value.to_bits()),
        TraceScalar::Table(lane) => Key::Constant(4, lane),
        TraceScalar::Closure(lane) => Key::Constant(5, lane),
        TraceScalar::String(lane) => Key::Constant(6, lane),
        TraceScalar::Int64(value) => Key::Constant(7, value.cast_unsigned()),
        TraceScalar::UInt64(value) => Key::Constant(8, value),
        TraceScalar::StringRef(bits) => Key::Constant(9, bits),
        TraceScalar::Thread(lane) => Key::Constant(10, lane),
        TraceScalar::Userdata(lane) => Key::Constant(11, lane),
    }
}

fn key_of(src: &TraceIr, op: &TraceOp) -> Option<Key> {
    use super::ir::ArithmeticOp as A;
    Some(match op {
        TraceOp::Constant(scalar) => scalar_key(*scalar),
        TraceOp::Negate { source } => Key::Negate(*source),
        TraceOp::IntToFloat { source } => Key::IntToFloat(*source),
        TraceOp::Numeric { op, source } => Key::Numeric(*op, *source),
        TraceOp::ToWide {
            source,
            target,
            conversion,
        } => Key::ToWide(
            *source,
            *target,
            u8::from(*conversion == WideConversion::Bits32),
        ),
        TraceOp::NativeUnary { function, source } => Key::NativeUnary(function.address(), *source),
        TraceOp::NativeBinary {
            function,
            left,
            right,
        } => Key::NativeBinary(function.address(), *left, *right),
        TraceOp::Arithmetic { op, left, right } => {
            let code = match op {
                A::Add => 0,
                A::Subtract => 1,
                A::Multiply => 2,
                A::Divide => 3,
                A::Modulo => 4,
                A::Power => 5,
                A::FloorDivide => 6,
                A::BitAnd => 7,
                A::BitOr => 8,
                A::BitXor => 9,
                A::ShiftLeft => 10,
                A::ShiftRight => 11,
                A::ArithmeticShiftRight => 12,
                A::RotateLeft => 13,
                A::RotateRight => 14,
                A::PowerSquare => 15,
                A::ModuloWith(kind) => 16 + *kind as u8,
            };
            // Integer add, multiply and the bitwise operations commute
            // exactly. Float operands keep their order: with two NaNs the
            // result may carry either payload.
            let commutes = matches!(op, A::Add | A::Multiply | A::BitAnd | A::BitOr | A::BitXor)
                && (src.values[left.index()].ty == TraceType::Integer
                    || src.values[left.index()].ty.is_wide());
            let (left, right) = if commutes && right < left {
                (*right, *left)
            } else {
                (*left, *right)
            };
            Key::Arithmetic(code, left, right)
        }
        TraceOp::Compare {
            comparison,
            left,
            right,
        } => {
            let code = match comparison {
                Comparison::Equal => 0,
                Comparison::LessThan => 1,
                Comparison::LessEqual => 2,
            };
            // Equality is symmetric for every type, NaN included.
            let (left, right) = if *comparison == Comparison::Equal && right < left {
                (*right, *left)
            } else {
                (*left, *right)
            };
            Key::Compare(code, left, right)
        }
        TraceOp::Truthy { source } => Key::Truthy(*source),
        TraceOp::Not { source } => Key::Not(*source),
        TraceOp::Select {
            condition,
            if_true,
            if_false,
        } => Key::Select(*condition, *if_true, *if_false),
        // Reads and effects are never merged. Guards are handled by guard
        // elimination, which knows their snapshots differ.
        _ => return None,
    })
}

/// Merges identical pure operations. Constants compare by bits, so `0.0`,
/// `-0.0` and NaNs with different payloads stay apart.
fn number_values_in(draft: &mut Draft<'_>, stats: &mut OptimizeStats) {
    let src = draft.src;
    // Sized for every item, so the table never grows while the pass runs.
    let mut seen: HashMap<Key, ValueId> = HashMap::with_capacity_and_hasher(
        draft.preheader.len() + draft.body.len(),
        rustc_hash::FxBuildHasher,
    );
    draft.rewrite(|draft, item| {
        let (Some(result), Some(key)) = (item.result, key_of(src, &item.op)) else {
            return true;
        };
        if let Some(earlier) = seen.get(&key) {
            draft.alias(result, *earlier);
            stats.merged += 1;
            return false;
        }
        seen.insert(key, result);
        true
    });
}

/// Takes the length of a concatenation from its operands where nothing else
/// reads the string: `#(a .. b)` makes no string.
///
/// A [`TraceOp::HostCall`] of [`HostFunction::StringLength`] on the result of a
/// [`HelperId::CONCAT`] call becomes one [`HostFunction::ConcatLength`] call
/// at the concatenation's place, on its operands and with its guard. The
/// guard's snapshot is the one before the concatenation, which the
/// interpreter replays whole, so the string that was never made is not
/// missed. That needs the string to have no other reader: no other operand,
/// no snapshot but the length's own (which goes with the length), no
/// terminator, and no side trace continuing from the length's guard. A trace
/// with any of those keeps the concatenation.
fn sink_concat_lengths_in(draft: &mut Draft<'_>, stats: &mut OptimizeStats) {
    let ir = draft.src;
    let measures = |item: &Item| {
        matches!(
            item.op,
            TraceOp::HostCall {
                function: HostFunction::StringLength,
                ..
            }
        )
    };
    if !draft.body.iter().any(measures) {
        return;
    }
    let operand_uses = draft.operand_uses();
    let values_of = |draft: &Draft<'_>, snapshot: SnapshotId| {
        let mut values = Vec::new();
        draft.snapshots[snapshot.index()].for_each_value(|value| {
            if draft.snapshot_constant(value).is_none() {
                values.push(draft.rep(value));
            }
        });
        values
    };
    // Every snapshot a guard, the poll or the terminator restores from reads
    // its values.
    // A finished trace gives each guard a snapshot record of its own, so a
    // snapshot counts once however many guards name it.
    let mut snapshot_reads = vec![0_u32; ir.values.len()];
    for (index, used) in draft.referenced_snapshots().into_iter().enumerate() {
        if used {
            let id = SnapshotId(u32::try_from(index).expect("snapshot count fits u32"));
            for value in values_of(draft, id) {
                snapshot_reads[value.index()] += 1;
            }
        }
    }

    let mut body = std::mem::take(&mut draft.body);
    let defined: HashMap<ValueId, usize> = body
        .iter()
        .enumerate()
        .filter_map(|(index, item)| Some((draft.rep(item.result?), index)))
        .collect();
    let mut gone = vec![false; body.len()];
    for at in 0..body.len() {
        let TraceOp::HostCall {
            function: HostFunction::StringLength,
            args,
            result,
            ..
        } = &body[at].op
        else {
            continue;
        };
        let (&[string], result, Some(own), Some(length)) =
            (&args[..], *result, body[at].guard, body[at].result)
        else {
            continue;
        };
        let string = draft.rep(string);
        let Some(&made) = defined.get(&string) else {
            continue;
        };
        let TraceOp::CallHelper {
            helper: HelperId::CONCAT,
            args: parts,
            ..
        } = &body[made].op
        else {
            continue;
        };
        let own_reads = values_of(draft, own.snapshot)
            .iter()
            .filter(|value| **value == string)
            .count();
        let sized = parts.len() >= 2
            && parts.len() <= MAX_HOST_ARGS
            && parts.iter().all(|part| {
                matches!(
                    draft.ty(draft.rep(*part)),
                    TraceType::String
                        | TraceType::StringRef
                        | TraceType::Integer
                        | TraceType::Float
                )
            });
        let only_reader = operand_uses[string.index()] == 1
            && snapshot_reads[string.index()] as usize == own_reads
            && !ir.sides.iter().any(|side| side.guard == own.origin);
        if !sized || !only_reader || made >= at || gone[made] {
            continue;
        }
        body[made].op = TraceOp::HostCall {
            function: HostFunction::ConcatLength,
            args: parts.clone(),
            result,
            guard: GuardId(0),
        };
        body[made].result = Some(length);
        gone[at] = true;
        stats.folded += 1;
    }
    let mut gone = gone.into_iter();
    body.retain(|_| !gone.next().unwrap_or(false));
    draft.body = body;
}

/// Drops a `string.sub` that only snapshots read: a [`HelperId::STRING_SLICE`]
/// call whose result no operation, loop argument or terminator uses. Each
/// snapshot that held the string names a [`SliceRecipe`] instead, and the
/// exit that restores it makes the substring from the source and the two
/// indices. The recorder guarded both indices into `-2^31 <= x < 2^31`,
/// where every profile it records for converts them by truncation, so the
/// exit's string is the one the call gave and no call of the interpreter's
/// can fail.
///
/// A snapshot keeps the call when it cannot hold a recipe: a virtual table
/// stores the string, the string goes to a cell or global slot, a guard that
/// a side trace continues restores it, or the link of a side trace does.
///
/// This pass cannot also drop the substring of `slice == other` for a string
/// `other` that is not a constant. The call's result is an operand of the
/// equality, so it is not snapshot-only here, and by the time the pass runs
/// the comparison is one opaque string-equality operation. Reading the
/// substring from its source needs the recorder: it must guard the length and
/// load each byte of `other` (see `record_slice_equality`, which does this
/// for a constant text whose bytes are known). A runtime string has no bound
/// on its length and its bytes are trace inputs, so the byte-guard scheme
/// keeps the call for it. An operation that compares `source[start..end]`
/// with `other` (a length check, then a memcmp helper) would drop the
/// substring there too; neither pass has one yet.
fn sink_slices_in(draft: &mut Draft<'_>, writes: &TreeWrites, stats: &mut OptimizeStats) {
    let ir = draft.src;
    let slices = |item: &Item| {
        matches!(
            item.op,
            TraceOp::CallHelper {
                helper: HelperId::STRING_SLICE,
                ..
            }
        )
    };
    if !draft.body.iter().any(slices) {
        return;
    }
    let operand_uses = draft.operand_uses();
    // A snapshot nothing restores from is gone from the finished trace, so
    // what it names holds no call back.
    let referenced = draft.referenced_snapshots();
    // The snapshots that cannot hold a recipe, whatever they name.
    let mut fixed = vec![false; draft.snapshots.len()];
    if let TraceTerminator::Link { snapshot } | TraceTerminator::LinkLoop { snapshot, .. } =
        &ir.terminator
    {
        fixed[snapshot.index()] = true;
    }
    for item in draft.preheader.iter().chain(&draft.body) {
        if let Some(guard) = item.guard
            && writes.continues(guard.origin)
        {
            fixed[guard.snapshot.index()] = true;
        }
    }
    let external = ir.external.len();

    let mut body = std::mem::take(&mut draft.body);
    let mut gone = vec![false; body.len()];
    for at in 0..body.len() {
        let TraceOp::CallHelper {
            helper: HelperId::STRING_SLICE,
            args,
            ..
        } = &body[at].op
        else {
            continue;
        };
        let (&[source, first, last], Some(own), Some(result)) =
            (&args[..], body[at].guard, body[at].result)
        else {
            continue;
        };
        let result = draft.rep(result);
        if operand_uses[result.index()] != 0 || writes.continues(own.origin) {
            continue;
        }
        let (source, first, last) = (draft.rep(source), draft.rep(first), draft.rep(last));
        let names = |entry: &RegisterEntry| matches!(entry, RegisterEntry::Value(value) if draft.rep(*value) == result);
        // Every snapshot that holds the string must take a recipe.
        let mut mentioning = Vec::new();
        let mut possible = true;
        for (index, snapshot) in draft.snapshots.iter().enumerate() {
            if !referenced[index] {
                continue;
            }
            let in_recipe = snapshot
                .virtuals
                .iter()
                .any(|table| table.entries().any(&names))
                || snapshot
                    .slices
                    .iter()
                    .any(|slice| slice.entries().any(&names));
            let mut held = false;
            for (position, frame) in snapshot.frames.iter().enumerate() {
                let guest =
                    frame
                        .registers
                        .len()
                        .saturating_sub(if position == 0 { external } else { 0 });
                for (register, entry) in frame.registers.iter().enumerate() {
                    if names(entry) {
                        held = true;
                        possible &= register < guest;
                    }
                }
            }
            if in_recipe {
                possible = false;
            }
            if held || in_recipe {
                possible &= !fixed[index];
                mentioning.push(index);
            }
        }
        if !possible {
            continue;
        }
        for index in mentioning {
            let snapshot = &mut draft.snapshots.to_mut()[index];
            let recipe = SliceRecipe {
                string: RegisterEntry::Value(source),
                first: RegisterEntry::Value(first),
                last: RegisterEntry::Value(last),
            };
            let id = SliceId(u32::try_from(snapshot.slices.len()).expect("recipe count fits u32"));
            snapshot.slices.push(recipe);
            let repl = &draft.repl;
            snapshot.entries_mut(|entry| {
                if matches!(entry, RegisterEntry::Value(value) if resolve(repl, *value) == result) {
                    *entry = RegisterEntry::Slice(id);
                }
            });
        }
        gone[at] = true;
        stats.folded += 1;
    }
    let mut gone = gone.into_iter();
    body.retain(|_| !gone.next().unwrap_or(false));
    draft.body = body;
}

/// Whether an operation may be removed when nothing reads its result.
const fn is_removable(op: &TraceOp) -> bool {
    // A bound field read and a bound length read cannot fail and change
    // nothing, so a dead one goes. Every other heap operation stays: an
    // array read may leave through its guard, and a store is an effect. A
    // hash lookup stays too, even with its result unused: its type guard
    // holds the speculation that later folding rests on (a lookup that found
    // nil made `if value` a constant branch), so only the guard keeps the
    // trace honest.
    // A phantom table charge is a count the run owes, with no result to read.
    matches!(op, TraceOp::FieldLoad { .. } | TraceOp::TableLength { .. })
        || !(op.touches_heap()
            || matches!(op, TraceOp::PhantomTable { .. })
            || matches!(op, TraceOp::GuardType { .. } | TraceOp::GuardBranch { .. }))
}

/// Removes values nothing reads. A snapshot reads its `Value` entries, so a
/// value only a recovery path needs stays. A value a snapshot names that is
/// a known constant does not: the snapshot carries the constant itself.
fn eliminate_dead_code_in(draft: &mut Draft<'_>, stats: &mut OptimizeStats) {
    let ir = draft.src;
    let mut live = vec![false; ir.values.len()];
    let used_snapshots = draft.referenced_snapshots();
    // Marking a value twice changes nothing, so the runs a snapshot shares
    // with the used one before it are not walked again.
    let mut previous: Option<&Snapshot> = None;
    for (index, snapshot) in draft.snapshots.iter().enumerate() {
        if !used_snapshots[index] {
            continue;
        }
        snapshot.for_each_value_since(previous, |value| {
            if draft.snapshot_constant(value).is_none() {
                live[draft.rep(value).index()] = true;
            }
        });
        previous = Some(snapshot);
    }
    #[cfg(debug_assertions)]
    {
        let mut whole = vec![false; ir.values.len()];
        for (index, snapshot) in draft.snapshots.iter().enumerate() {
            if used_snapshots[index] {
                snapshot.for_each_value(|value| {
                    if draft.snapshot_constant(value).is_none() {
                        whole[draft.rep(value).index()] = true;
                    }
                });
            }
        }
        assert!(
            live == whole,
            "skipping shared runs lost a value a snapshot reads"
        );
    }
    match &ir.terminator {
        TraceTerminator::LoopBack { args } | TraceTerminator::TailCall { args, .. } => {
            for arg in args {
                live[draft.rep(*arg).index()] = true;
            }
        }
        TraceTerminator::Return { value: Some(value) } => {
            live[draft.rep(*value).index()] = true;
        }
        _ => {}
    }

    // Walk backwards, so a use is seen before the definition it keeps alive.
    let mut removed = 0_u32;
    let mut keep: Vec<bool> = Vec::new();
    let mut sweep = |repl: &[ValueId], items: &mut Vec<Item>, live: &mut [bool]| {
        keep.clear();
        keep.resize(items.len(), true);
        let before = removed;
        for (index, item) in items.iter().enumerate().rev() {
            let needed = !is_removable(&item.op)
                || item
                    .result
                    .is_some_and(|result| live[resolve(repl, result).index()]);
            if needed {
                item.op
                    .operands(|value| live[resolve(repl, value).index()] = true);
            } else {
                keep[index] = false;
                removed += 1;
            }
        }
        if removed != before {
            let mut keep = keep.iter();
            items.retain(|_| *keep.next().expect("one flag per item"));
        }
    };
    sweep(&draft.repl, &mut draft.body, &mut live);
    sweep(&draft.repl, &mut draft.preheader, &mut live);
    stats.dead_removed += removed;
}

/// Moves loop-invariant work into the preheader.
///
/// A value is invariant when it depends only on constants and entry inputs.
/// The recorder makes a register an entry input only when the iteration never
/// writes it, and carries every written register through a loop parameter, so
/// "depends only on entry inputs" is a property of the IR, not a guess from
/// one observed iteration. Reads and effects are never invariant, and nor is
/// anything they feed, since no memory proof exists.
///
/// Pure operations are total (floats follow IEEE, integers wrap), so running
/// one before the first iteration cannot fail or change anything observable.
/// A branch guard on an invariant condition gives the same answer every
/// iteration, so checking it once before the loop only moves its exit
/// earlier, to the entry state, where the interpreter re-runs the iteration.
fn hoist_invariants_in(
    draft: &mut Draft<'_>,
    options: &OptimizeOptions,
    writes: &TreeWrites,
    stats: &mut OptimizeStats,
) {
    let ir = draft.src;
    // A side that starts inside inlined calls reaches its loop header only
    // after returning from them: there is no entry state at the header for
    // a guard to leave through.
    let side = writes.member > 0;
    let hoist_guards =
        options.hoist_guards && !ir.keep_branch_guards && !(side && ir.seed_frames > 0);
    let hoist_loads = options.memory;
    let Some(header) = loop_header(ir) else {
        return;
    };
    let mut invariant = vec![false; ir.values.len()];
    for input in &ir.entry.inputs {
        invariant[input.value.index()] = true;
    }
    // A loop parameter the loop-back hands on unchanged is invariant too: the
    // iteration never changes it (only a side trace's link does, and the
    // preheader runs again at every link, see `pass_through_params`).
    // An argument is compared by its representative: an earlier pass may have
    // aliased it to the parameter.
    let pass_through: Vec<usize> = match &ir.terminator {
        TraceTerminator::LoopBack { args } => ir
            .loop_params
            .iter()
            .zip(args)
            .enumerate()
            .filter(|(_, (param, arg))| param.value == draft.rep(**arg))
            .map(|(index, _)| index)
            .collect(),
        _ => Vec::new(),
    };
    for &index in &pass_through {
        invariant[ir.loop_params[index].value.index()] = true;
    }
    for item in &draft.preheader {
        if let Some(result) = item.result {
            invariant[result.index()] = true;
        }
    }
    let mut body = std::mem::take(&mut draft.body);
    for item in &mut body {
        draft.canonicalize(item);
    }

    // The release that ends the body keeps at most five strings.
    let mut release_room = match body.last().map(|item| &item.op) {
        Some(TraceOp::ReleaseStrings { keep }) => MAX_HOST_ARGS.saturating_sub(keep.len()),
        _ => 0,
    };
    let mut kept_strings: Vec<ValueId> = Vec::new();
    // First decide which operations move, then which constants they drag.
    let mut moving = vec![false; body.len()];
    let mut constant_index: HashMap<ValueId, usize> = HashMap::default();
    let mut hoisted_guards = 0_u32;
    for (index, item) in body.iter().enumerate() {
        // A guard a side trace continues stays where it was recorded: in the
        // preheader it would fail before the loop, where the side cannot
        // start, and a link to the loop header runs the preheader again, so
        // an invariant of the root's loop is no invariant of the tree.
        if item
            .guard
            .as_ref()
            .is_some_and(|guard| writes.continues(guard.origin))
        {
            continue;
        }
        // Whether the read at `item` may move: its operands are invariant and
        // nothing in the tree writes what it reads.
        let load = (|| -> Option<bool> {
            let mut all = true;
            item.op.operands(|value| all &= invariant[value.index()]);
            if !hoist_loads || !all {
                return None;
            }
            match &item.op {
                TraceOp::FieldLoad { binding, .. } => {
                    let BindingKind::Field { key, .. } =
                        &draft.src.bindings[*binding as usize].kind
                    else {
                        return None;
                    };
                    (!writes.writes_field(key)).then_some(false)
                }
                TraceOp::TableLength { .. } => (!writes.has_unknown_write()).then_some(false),
                TraceOp::TableFieldLoad { field, .. } => {
                    let key = &draft.src.fields[*field as usize].key;
                    (hoist_guards && !writes.writes_field(key)).then_some(true)
                }
                TraceOp::ArrayLoad { key, expected, .. } => (hoist_guards
                    && *expected != TraceType::Nil
                    && !writes.writes_array(numeric_key(draft.constant(*key)).as_ref()))
                .then_some(true),
                // The same test: an array store anywhere in the tree, or
                // anything that may move an array, keeps a lane read in the
                // loop. Its row word and the table's words are then the
                // preheader's for the whole run of the loop.
                TraceOp::LaneArrayLoad { key, .. } => (hoist_guards
                    && !writes.writes_array(numeric_key(draft.constant(*key)).as_ref()))
                .then_some(true),
                // A call of a library function is pure. Those that read a
                // table need the tree to write none.
                TraceOp::HostCall { function, .. } => {
                    let reads_tables = matches!(function, HostFunction::Native(_));
                    (hoist_guards && (!reads_tables || writes.writes_nothing())).then_some(true)
                }
                // A helper that does not write is as pure as a host call:
                // one that reads tables needs the tree to write none. One
                // with several results stays, since a later instruction
                // reads those from the body.
                TraceOp::CallHelper { helper, .. } => {
                    let effects = helper.effects();
                    let single = helper.info().is_some_and(|info| info.results.len() <= 1);
                    (hoist_guards
                        && single
                        && !effects.commits_effect()
                        && !effects.contains(HelperEffects::FRESH_TABLE)
                        && (!effects.contains(HelperEffects::READS_HEAP)
                            || writes.writes_nothing()))
                    .then_some(true)
                }
                _ => None,
            }
        })();
        let all_invariant = |op: &TraceOp| {
            let mut all = true;
            op.operands(|value| all &= invariant[value.index()]);
            all
        };
        // A string a hoisted call makes lives on past the loop-back: the
        // release at the end of the body keeps it, if it has room to.
        let makes_string = match &item.op {
            TraceOp::HostCall { result, .. } => Some(*result == TraceType::String),
            TraceOp::CallHelper { helper, .. } => helper
                .info()
                .map(|info| info.results.first() == Some(&HelperType::String)),
            _ => None,
        };
        if let (Some(keeps), Some(_)) = (makes_string, load) {
            if keeps && release_room == 0 {
                continue;
            }
            release_room -= usize::from(keeps);
            if keeps && let Some(result) = item.result {
                kept_strings.push(result);
            }
            moving[index] = true;
            if let Some(result) = item.result {
                invariant[result.index()] = true;
            }
            hoisted_guards += 1;
            continue;
        }
        match &item.op {
            TraceOp::Constant(_) => {
                if let Some(result) = item.result {
                    invariant[result.index()] = true;
                    constant_index.insert(result, index);
                }
            }
            TraceOp::Negate { .. }
            | TraceOp::IntToFloat { .. }
            | TraceOp::Numeric { .. }
            | TraceOp::ToWide { .. }
            | TraceOp::NativeUnary { .. }
            | TraceOp::NativeBinary { .. }
            | TraceOp::Arithmetic { .. }
            | TraceOp::Compare { .. }
            | TraceOp::Select { .. }
            | TraceOp::Truthy { .. }
            | TraceOp::Not { .. }
                if all_invariant(&item.op) =>
            {
                moving[index] = true;
                if let Some(result) = item.result {
                    invariant[result.index()] = true;
                }
            }
            TraceOp::FieldLoad { .. }
            | TraceOp::TableLength { .. }
            | TraceOp::TableFieldLoad { .. }
            | TraceOp::ArrayLoad { .. }
            | TraceOp::LaneArrayLoad { .. }
                if load.is_some() =>
            {
                moving[index] = true;
                if let Some(result) = item.result {
                    invariant[result.index()] = true;
                }
                if load == Some(true) {
                    hoisted_guards += 1;
                }
            }
            TraceOp::GuardBranch { condition, .. }
                if hoist_guards
                    && invariant[condition.index()]
                    && draft.constant(*condition).is_none() =>
            {
                moving[index] = true;
                hoisted_guards += 1;
            }
            // The metatable word of a table's row changes only when a call
            // changes the table's metatable, so in a tree without one the
            // class of a table the loop never replaces is the same in every
            // iteration: one check before the loop answers for all of them.
            TraceOp::MetaGuard { table, .. }
                if hoist_guards && invariant[table.index()] && !writes.changes_metatables() =>
            {
                moving[index] = true;
                hoisted_guards += 1;
            }
            _ => {}
        }
    }
    if !moving.iter().any(|moves| *moves) {
        draft.body = body;
        return;
    }
    if !kept_strings.is_empty()
        && let Some(TraceOp::ReleaseStrings { keep }) = body.last_mut().map(|item| &mut item.op)
    {
        keep.extend(kept_strings);
    }
    // A constant a moving operation reads moves with it, so the preheader
    // defines everything it uses.
    for index in 0..body.len() {
        if !moving[index] {
            continue;
        }
        body[index].op.operands(|value| {
            if let Some(constant) = constant_index.get(&value) {
                moving[*constant] = true;
            }
        });
    }

    let entry_snapshot = (hoisted_guards > 0).then(|| {
        let poll = &draft.snapshots[ir.poll_snapshot.expect("a header implies a poll").index()];
        // The entry state's previous pc: the latch, or none for a loop a
        // tail call enters.
        let last = match poll.phase {
            SnapshotPhase::After { pc, .. } => Some(pc),
            SnapshotPhase::Before { .. } => poll.frames[0].last_interpreted_pc,
        };
        let register_count = usize::from(ir.frames[0].register_count);
        let mut registers = vec![RegisterEntry::Unchanged; register_count];
        // A side's entry state is its parent's native state: the interpreter
        // still holds what the registers held before the tree's run, so every
        // register the side starts with is named. (The recorder makes every
        // live register a side reads or leaves alone an entry input.)
        if side {
            // A global's slot stays `Unchanged`: the restore plan refuses a
            // global slot with a value, and no member writes one.
            let guest = register_count.saturating_sub(ir.external.len());
            let named = |register: u16| {
                usize::from(register)
                    .checked_sub(guest)
                    .and_then(|slot| ir.external.get(slot))
                    .is_none_or(|slot| {
                        !matches!(
                            slot,
                            ExternalSlot::Global(_)
                                | ExternalSlot::ClosureGlobal { .. }
                                | ExternalSlot::CallerGlobal(_)
                                | ExternalSlot::UpvalueField { .. }
                                | ExternalSlot::ClosureUpvalueField { .. }
                                | ExternalSlot::Environment
                        )
                    })
            };
            for input in &ir.entry.inputs {
                if named(input.register)
                    && let Some(entry) = registers.get_mut(usize::from(input.register))
                {
                    *entry = RegisterEntry::Value(input.value);
                }
            }
            for param in &ir.loop_params {
                if let Some(register) = param.register
                    && named(register)
                    && let Some(entry) = registers.get_mut(usize::from(register))
                {
                    *entry = match param.init {
                        LoopParamInit::Input(value) => RegisterEntry::Value(value),
                        LoopParamInit::Const(scalar) => RegisterEntry::Const(scalar),
                    };
                }
            }
        }
        if !pass_through.is_empty() {
            // The preheader runs again after a link, when the interpreter's
            // registers are older than the loop's: name each parameter.
            for param in &ir.loop_params {
                if let Some(register) = param.register
                    && let Some(entry) = registers.get_mut(usize::from(register))
                {
                    *entry = RegisterEntry::Value(param.value);
                }
            }
        }
        // The control register of a `next` loop that keeps its key in a lane
        // is no loop parameter: a link leaves the key in the lane, and the
        // interpreter register is older. Name the lane's input. The loop
        // variables the body never reads are restored from lanes too: the
        // first from the control lane, the second from a lane of its own.
        if let Some(register) = ir.boxed_control {
            let input_of = |register: u16| {
                ir.entry
                    .inputs
                    .iter()
                    .find(|input| input.register == register)
                    .map(|input| input.value)
            };
            let control = input_of(register);
            let variables = [
                (register, control),
                (register + 1, control.filter(|_| ir.boxed_key)),
                (
                    register + 2,
                    input_of(register + 2).filter(|_| ir.boxed_value),
                ),
            ];
            for (register, value) in variables {
                if let Some(value) = value
                    && let Some(entry) = registers.get_mut(usize::from(register))
                {
                    *entry = RegisterEntry::Value(value);
                }
            }
        }
        draft.snapshots.to_mut().push(Snapshot::new(
            SnapshotPhase::Before { pc: header },
            0,
            vec![SnapshotFrame {
                frame: super::ir::FrameId(0),
                last_interpreted_pc: last,
                registers: registers.into(),
            }],
        ));
        SnapshotId(u32::try_from(draft.snapshots.len() - 1).expect("snapshots fit u32"))
    });
    let header_site = TraceSite {
        frame: super::ir::FrameId(0),
        pc: header,
    };

    let mut kept = Vec::with_capacity(body.len());
    for (index, mut item) in body.into_iter().enumerate() {
        if !moving[index] {
            kept.push(item);
            continue;
        }
        if item.guard.is_some() {
            let snapshot = entry_snapshot.expect("hoisted guards need an entry snapshot");
            item.site = header_site;
            item.guard = Some(GuardInfo {
                snapshot,
                site: header_site,
                origin: item.guard.map_or(0, |guard| guard.origin),
            });
            if matches!(item.op, TraceOp::GuardBranch { .. }) {
                stats.guards_hoisted += 1;
            } else {
                stats.loads_hoisted += 1;
            }
        } else if matches!(
            item.op,
            TraceOp::FieldLoad { .. } | TraceOp::TableLength { .. }
        ) {
            stats.loads_hoisted += 1;
        } else if !matches!(item.op, TraceOp::Constant(_)) {
            stats.hoisted += 1;
        }
        draft.preheader.push(item);
    }
    draft.body = kept;
}

/// The loop parameters (by index) that the loop-back passes on unchanged.
///
/// The recorder makes a register a loop parameter only when the iteration
/// writes it, so such a parameter appears when a tree widens its root with a
/// register only a side trace writes (see [`super::plan_tree`]). The root's
/// own loop never changes it. A link does, by jumping back to the header with
/// the side's value, and the backend runs the preheader on every such jump,
/// so work that depends on these parameters alone is invariant for as long as
/// the root loops.
#[must_use]
pub fn pass_through_params(ir: &TraceIr) -> Vec<usize> {
    let TraceTerminator::LoopBack { args } = &ir.terminator else {
        return Vec::new();
    };
    ir.loop_params
        .iter()
        .zip(args)
        .enumerate()
        .filter(|(_, (param, arg))| param.value == **arg)
        .map(|(index, _)| index)
        .collect()
}

/// The loop header pc, when the trace is a loop that can host a preheader:
/// a loop-back with a root-frame poll snapshot that resumes at the header.
fn loop_header(ir: &TraceIr) -> Option<usize> {
    if !matches!(ir.terminator, TraceTerminator::LoopBack { .. }) {
        return None;
    }
    let poll = &ir.snapshots[ir.poll_snapshot?.index()];
    match poll.phase {
        SnapshotPhase::After { successor, .. }
            if poll.frames.len() == 1 && poll.frames[0].frame.index() == 0 =>
        {
            Some(successor)
        }
        // A loop through tail calls: the closing tail call enters the header.
        SnapshotPhase::Before { pc }
            if ir.loop_tail.is_some()
                && poll.frames.len() == 1
                && poll.frames[0].frame.index() == 0 =>
        {
            Some(pc)
        }
        _ => None,
    }
}

#[cfg(test)]
#[path = "optimize_tests.rs"]
pub(in crate::jit_ir::trace) mod tests;
