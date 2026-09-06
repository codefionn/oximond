//! Operation statistics for performance hypotheses, compiled only with the
//! `op-stats` feature.
//!
//! Every count lives in a process-wide static of fixed size, so counting
//! allocates nothing and native trace code can bump a counter at a constant
//! address. The VM, the trace backend and the collector bump them behind
//! `#[cfg(feature = "op-stats")]`; a build without the feature has no
//! counter code at all.
//!
//! Each counter has a [`Kind`] that says what it counts. A static count
//! describes compiled trace IR once per compiled tree, whatever the tree
//! later executes; a dynamic count is bumped every time the operation runs.
//! The interpreter counts come from the fast loop and the general loop
//! separately, and neither is the interpreter's normal cost: the diagnostic
//! build runs slower than the ordinary one.
//!
//! Hot counters use a plain load and store rather than a locked add. Guest
//! code and native traces run on one thread; trace compilation may run on a
//! broker thread, so the static IR counts use [`add_shared`].
//!
//! [`report_json`] renders everything as one JSON document, schema
//! [`SCHEMA`]. `benchmarks/measure.py` parses it and refuses a report that
//! lacks a counter it needs, so names are only added, never renamed.

#![allow(
    clippy::declare_interior_mutable_const,
    clippy::large_stack_arrays,
    reason = "the EMPTY constants only initialize the statics"
)]
#![allow(
    clippy::inline_always,
    reason = "counting must not add a call to the interpreter's dispatch"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::must_use_candidate,
    reason = "indices are masked to the table sizes; callers may ignore a site change"
)]

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// The `schema` field of [`report_json`].
pub const SCHEMA: &str = "oximond-op-stats/2";

/// What a counter counts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    /// A property of optimized trace IR, counted once per compiled tree.
    StaticIr,
    /// Guest operations the fast interpreter loop executed.
    DynamicFastInterp,
    /// Guest operations the general interpreter loop executed.
    DynamicGeneralInterp,
    /// Guest operations either interpreter loop executed.
    DynamicInterp,
    /// Operations native trace code executed.
    DynamicNative,
    /// Work done at garbage collections.
    GcEvent,
}

impl Kind {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::StaticIr => "static_ir",
            Self::DynamicFastInterp => "dynamic_fast_interp",
            Self::DynamicGeneralInterp => "dynamic_general_interp",
            Self::DynamicInterp => "dynamic_interp",
            Self::DynamicNative => "dynamic_native",
            Self::GcEvent => "gc_event",
        }
    }
}

macro_rules! counters {
    ($($variant:ident = $name:literal, $kind:ident, $help:literal;)*) => {
        /// One scalar counter. The discriminant indexes [`COUNTERS`].
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        #[repr(usize)]
        pub enum Counter { $($variant,)* }

        impl Counter {
            pub const ALL: &[Self] = &[$(Self::$variant,)*];

            /// The report key. Keys are prefixed by the hypothesis they
            /// serve (`h1.` to `h6.`).
            #[must_use]
            pub const fn name(self) -> &'static str {
                match self { $(Self::$variant => $name,)* }
            }

            #[must_use]
            pub const fn kind(self) -> Kind {
                match self { $(Self::$variant => Kind::$kind,)* }
            }

            #[must_use]
            pub const fn help(self) -> &'static str {
                match self { $(Self::$variant => $help,)* }
            }
        }
    };
}

counters! {
    // H1: loop range proofs.
    IrArraySites = "h1.ir.array_sites", StaticIr,
        "bounds-checked array or byte accesses (ArrayLoad, ArrayStore, ArraySet, StringByte)";
    IrArraySitesConstKey = "h1.ir.array_sites.const_key", StaticIr,
        "array sites whose key is a constant";
    IrArraySitesFloatKey = "h1.ir.array_sites.float_key", StaticIr,
        "array sites whose key is a float, which also need an integrality check";
    IrArraySitesInduction = "h1.ir.array_sites.induction", StaticIr,
        "array sites whose key is a loop parameter stepped by an integral term (constant, or loop-invariant integer), plus or minus such a term";
    IrArraySitesRangeOnce = "h1.ir.array_sites.range_once", StaticIr,
        "syntactic range-once candidates: induction sites whose loop exit guard compares the induction with a loop-invariant bound, in a tree with no ArraySet; not a proof (comparison direction, step sign, initial integrality, overflow are unchecked)";
    IrArraySitesFloatInvariantInduction = "h1.ir.array_sites.float_invariant_induction", StaticIr,
        "float-key sites that are inductions only if a loop-invariant, non-constant float step or offset is integral (checked at entry by a hoist), such as Lua 5.1 numeric for loops";
    IrArraySitesFloatInvariantRangeOnce = "h1.ir.array_sites.float_invariant_range_once", StaticIr,
        "of those, syntactic range-once candidates (same caveats as h1.ir.array_sites.range_once)";
    NativeBoundChecksFloatInvariantInduction = "h1.native.bound_checks.float_invariant_induction", DynamicNative,
        "executed bound checks at float-invariant induction sites";
    NativeBoundChecksFloatInvariantRangeOnce = "h1.native.bound_checks.float_invariant_range_once", DynamicNative,
        "executed bound checks at float-invariant range-once candidate sites";
    IrArraySitesBytes = "h1.ir.array_sites.bytes", StaticIr,
        "of the array sites, StringByte reads of immutable string bytes";
    IrTypeGuards = "h1.ir.type_guards", StaticIr,
        "GuardType instructions in loop bodies";
    NativeBoundChecks = "h1.native.bound_checks", DynamicNative,
        "executed bound checks of array and byte accesses";
    NativeIntegralityChecks = "h1.native.integrality_checks", DynamicNative,
        "executed float-key integrality checks";
    NativeBoundChecksInduction = "h1.native.bound_checks.induction", DynamicNative,
        "executed bound checks at induction sites";
    NativeBoundChecksRangeOnce = "h1.native.bound_checks.range_once", DynamicNative,
        "executed bound checks at range-once sites";
    InterpGetTableIntegerKey = "h1.interp.get_table.integer_key", DynamicFastInterp,
        "GET_TABLE with an integer key";
    InterpGetTableFloatKey = "h1.interp.get_table.float_key", DynamicFastInterp,
        "GET_TABLE with a float key";
    InterpSetTableIntegerKey = "h1.interp.set_table.integer_key", DynamicFastInterp,
        "SET_TABLE with an integer key";
    InterpSetTableFloatKey = "h1.interp.set_table.float_key", DynamicFastInterp,
        "SET_TABLE with a float key";

    // H2: adaptive bytecode.
    FastDispatches = "h2.fast.dispatches", DynamicFastInterp,
        "instructions the fast loop dispatched";
    FastScalarMisses = "h2.fast.scalar_misses", DynamicFastInterp,
        "fast-loop instructions handed to the out-of-line scalar miss path";
    FastArithmeticSlow = "h2.fast.arithmetic_slow", DynamicFastInterp,
        "arithmetic that missed both the float and the integer case";
    FastExitsFallback = "h2.fast.exits.fallback", DynamicFastInterp,
        "fast-loop exits to the general loop";
    GeneralSteps = "h2.general.steps", DynamicGeneralInterp,
        "instructions the general loop dispatched";

    // H3: recovery recipes.
    IrMembers = "h3.ir.members", StaticIr,
        "compiled tree members (root and sides)";
    IrSnapshots = "h3.ir.snapshots", StaticIr,
        "snapshots of compiled members";
    IrSnapshotFrames = "h3.ir.snapshot_frames", StaticIr,
        "frames over all snapshots";
    IrSnapshotEntries = "h3.ir.snapshot_entries", StaticIr,
        "register entries over all snapshots";
    IrSnapshotValueEntries = "h3.ir.snapshot_entries.value", StaticIr,
        "entries that restore an SSA value";
    IrSnapshotConstEntries = "h3.ir.snapshot_entries.const", StaticIr,
        "entries that restore a constant";
    IrSnapshotUnchangedEntries = "h3.ir.snapshot_entries.unchanged", StaticIr,
        "entries left unchanged";
    IrSnapshotRootEntries = "h3.ir.snapshot_entries.root", StaticIr,
        "entries restored from the root bank";
    IrSnapshotVirtualEntries = "h3.ir.snapshot_entries.virtual", StaticIr,
        "frame registers that name a virtual table";
    IrSnapshotOtherEntries = "h3.ir.snapshot_entries.other", StaticIr,
        "entries of a kind this analysis does not know, or that may not appear where they do";
    IrSnapshotVirtuals = "h3.ir.snapshot_virtuals", StaticIr,
        "distinct virtual-table recipes the frames of each snapshot reach, summed over snapshots";
    IrSnapshotRecipeEntries = "h3.ir.snapshot_recipe_entries", StaticIr,
        "operands of the reached recipes";
    IrSnapshotRecipeValueEntries = "h3.ir.snapshot_recipe_entries.value", StaticIr,
        "recipe operands that restore an SSA value";
    IrSnapshotRecipeConstEntries = "h3.ir.snapshot_recipe_entries.const", StaticIr,
        "recipe operands that restore a constant";
    IrSnapshotRecipeRootEntries = "h3.ir.snapshot_recipe_entries.root", StaticIr,
        "recipe operands restored from the root bank";
    IrSnapshotRecipeValuesFeeding = "h3.ir.snapshot_recipe_values.feeding", StaticIr,
        "recipe value operands in snapshots whose values feed a side trace or loop header";
    IrSnapshotRecipeValuesRecovery = "h3.ir.snapshot_recipe_values.recovery", StaticIr,
        "recipe value operands in recovery-only snapshots";
    IrSnapshotRecipesUnknown = "h3.ir.snapshot_recipes_unknown", StaticIr,
        "snapshots whose recipe graph could not be walked; their values count as no opportunity";
    IrSnapshotValues = "h3.ir.snapshot_values", StaticIr,
        "distinct values snapshots name, per member";
    IrSnapshotOnlyValues = "h3.ir.snapshot_only_values", StaticIr,
        "distinct values only snapshots use: no instruction, guard or terminator reads them";
    IrSnapshotOnlyCheap = "h3.ir.snapshot_only_values.cheap_pure", StaticIr,
        "snapshot-only values an exit could recompute from a small pure expression over values live at every guard that names them";
    IrSnapshotOnlyInstructions = "h3.ir.snapshot_only_values.instructions", StaticIr,
        "snapshot-only values defined by a body or preheader instruction";
    IrGuards = "h3.ir.guards", StaticIr,
        "guards of compiled members";
    IrExitDescriptors = "h3.ir.exit_descriptors", StaticIr,
        "exit descriptors of compiled trees";
    IrExitAreaSlots = "h3.ir.exit_area_slots", StaticIr,
        "exit area slots, summed over compiled trees";
    IrExitAreaSlotsMax = "h3.ir.exit_area_slots.max", StaticIr,
        "largest exit area of any compiled tree";
    IrTreesCompiled = "h3.ir.trees_compiled", StaticIr,
        "trees these static counts describe";
    NativeExits = "h3.native.exits", DynamicNative,
        "exits native code took through an exit stub";
    NativeExitSlotsStored = "h3.native.exit_slots_stored", DynamicNative,
        "exit area slots those stubs stored";

    // H4: direct call windows and dead-argument moves.
    Activations = "h4.activations", DynamicInterp,
        "plain Lua activations pushed (push_activation)";
    ActivationArgumentSlots = "h4.activations.argument_slots", DynamicInterp,
        "parameter slots filled from arguments";
    ActivationStackArgumentSlots = "h4.activations.argument_slots.stack", DynamicInterp,
        "parameter slots cloned from caller stack slots";
    ActivationStringArguments = "h4.activations.string_arguments", DynamicInterp,
        "argument slots holding a string, whose clone is a reference-count increment";
    ActivationClearedSlots = "h4.activations.cleared_slots", DynamicInterp,
        "callee window slots set to nil at activation";
    CallSiteClosureCalls = "h4.call_sites.closure_calls", DynamicFastInterp,
        "fixed-argument closure calls the fast loop saw (CALL, CALL_ONE)";
    CallSiteArguments = "h4.call_sites.arguments", DynamicFastInterp,
        "fixed arguments of those calls";
    CallSiteCallerDeadArguments = "h4.call_sites.arguments.caller_dead", DynamicFastInterp,
        "arguments in registers the caller cannot observe after the call (bytecode liveness, which keeps named locals and captures observable)";
    CallSiteCallerDeadStringArguments = "h4.call_sites.arguments.caller_dead_string", DynamicFastInterp,
        "caller-dead arguments holding a string";
    CallSiteVarargCallee = "h4.call_sites.vararg_callee", DynamicFastInterp,
        "closure calls whose callee is vararg (move needs fallback)";
    CallSiteCapturedCaller = "h4.call_sites.captured_caller", DynamicFastInterp,
        "closure calls from a frame with captured locals (move needs fallback)";
    InlinePlanCalls = "h4.inline_plan_calls", DynamicInterp,
        "closure calls the interpreter ran as an inline numeric plan, which push no activation (the VM's interpreter_inline_calls statistic)";
    InlineTablePlanCalls = "h4.inline_table_plan_calls", DynamicInterp,
        "closure calls the interpreter ran as an inline table plan, which push no activation (interpreter_inline_table_calls)";
    GeneralCallsHooked = "h4.general.calls_hooked", DynamicGeneralInterp,
        "calls the general loop ran with debug hooks set (move needs fallback)";

    // H5: live-root maps and lazy dead-slot clearing.
    ClearRanges = "h5.clear_range.instructions", DynamicFastInterp,
        "CLEAR_RANGE instructions the fast loop ran";
    ClearRangeSlots = "h5.clear_range.slots", DynamicFastInterp,
        "slots those instructions cleared";
    ClearRangeSlotsNil = "h5.clear_range.slots.nil", DynamicFastInterp,
        "cleared slots that already held nil";
    ClearRangeSlotsScalar = "h5.clear_range.slots.scalar", DynamicFastInterp,
        "cleared slots that held a number, boolean, 64-bit cdata or light function handle";
    ClearRangeSlotsString = "h5.clear_range.slots.string", DynamicFastInterp,
        "cleared slots that held a string (Arc release, not GC retention)";
    ClearRangeSlotsReference = "h5.clear_range.slots.reference", DynamicFastInterp,
        "cleared slots that held a collectable reference (GC retention)";
    GcRootScans = "h5.gc.root_scans", GcEvent,
        "root sets built for a collection with the running stack";
    GcStackSlots = "h5.gc.stack_slots", GcEvent,
        "value-stack slots rooted";
    GcFrameSlots = "h5.gc.frame_slots", GcEvent,
        "slots inside Lua frame windows";
    GcFrameReferenceSlots = "h5.gc.frame_slots.reference", GcEvent,
        "frame slots holding a collectable reference";
    GcFrameDeadReferenceSlots = "h5.gc.frame_slots.reference_dead", GcEvent,
        "reference slots the frame cannot observe from its pc (conservative: named locals and captures stay observable)";
    GcFrameStringSlots = "h5.gc.frame_slots.string", GcEvent,
        "frame slots holding a string";
    GcFrameDeadStringSlots = "h5.gc.frame_slots.string_dead", GcEvent,
        "string slots the frame cannot observe from its pc";
    GcFramesUnknownPc = "h5.gc.frames_unknown_pc", GcEvent,
        "frames whose pc could not be mapped to liveness (all their slots counted as live)";

    // H6: closure-family specialization.
    CallSiteSameClosure = "h6.calls.same_closure", DynamicFastInterp,
        "closure calls whose callee is the site's previous closure";
    CallSiteSamePrototypeOtherClosure = "h6.calls.same_prototype_other_closure", DynamicFastInterp,
        "closure calls to a different closure of the site's previous prototype";
    CallSiteOtherPrototype = "h6.calls.other_prototype", DynamicFastInterp,
        "closure calls to a different prototype than the site's previous one";
    IrPinnedClosures = "h6.ir.pinned_closures", StaticIr,
        "closures compiled members pin";
    ClosuresCreated = "h6.closures_created", DynamicInterp,
        "closures created by the CLOSURE instruction";
}

const COUNTER_COUNT: usize = Counter::ALL.len();

/// Number of opcode slots in histograms; every opcode byte is below it.
pub const OPCODES: usize = 128;

/// Operand classes for arithmetic and comparison histograms.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum OperandClass {
    FloatFloat,
    IntegerInteger,
    MixedNumber,
    Other,
}

impl OperandClass {
    pub const ALL: [Self; 4] = [
        Self::FloatFloat,
        Self::IntegerInteger,
        Self::MixedNumber,
        Self::Other,
    ];

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::FloatFloat => "float_float",
            Self::IntegerInteger => "integer_integer",
            Self::MixedNumber => "mixed_number",
            Self::Other => "other",
        }
    }
}

/// A fixed-size table of guest sites (a chunk and a pc), each with two
/// observed identities `a` and `b`.
///
/// A site keeps the last `a` and `b` it saw and counts how often each
/// changed. A site that never changed `a` is monomorphic in `a`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum SiteTable {
    /// Arithmetic and comparison sites: `a` is the [`OperandClass`].
    Arithmetic,
    /// GET_FIELD and SET_FIELD sites: `a` is the table's shape identity,
    /// `b` the table's identity.
    Field,
    /// Closure call sites: `a` is the callee's prototype (chunk) identity,
    /// `b` the closure's identity.
    Call,
}

impl SiteTable {
    pub const ALL: [Self; 3] = [Self::Arithmetic, Self::Field, Self::Call];

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Arithmetic => "arithmetic",
            Self::Field => "field",
            Self::Call => "call",
        }
    }

    const fn help(self) -> &'static str {
        match self {
            Self::Arithmetic => {
                "a = operand class (float_float, integer_integer, mixed_number, other)"
            }
            Self::Field => "a = table shape identity, b = table identity",
            Self::Call => "a = callee prototype (chunk) identity, b = closure identity",
        }
    }
}

/// Slots per site table. A power of two.
pub const SITE_SLOTS: usize = 1 << 14;
/// Slots probed before a site counts as an overflow.
const SITE_PROBES: usize = 16;
/// A site with at least this many executions is hot.
pub const HOT_SITE: u64 = 1000;

struct Site {
    /// The site key plus one; zero marks an empty slot.
    key: AtomicU64,
    count: AtomicU64,
    a: AtomicU64,
    b: AtomicU64,
    a_changes: AtomicU64,
    b_changes: AtomicU64,
    /// Bit 0..: per-table flags the caller ORs in.
    flags: AtomicU64,
}

impl Site {
    const EMPTY: Self = Self {
        key: AtomicU64::new(0),
        count: AtomicU64::new(0),
        a: AtomicU64::new(0),
        b: AtomicU64::new(0),
        a_changes: AtomicU64::new(0),
        b_changes: AtomicU64::new(0),
        flags: AtomicU64::new(0),
    };
}

struct Sites {
    slots: [Site; SITE_SLOTS],
    overflow: AtomicU64,
}

impl Sites {
    const EMPTY: Self = Self {
        slots: [Site::EMPTY; SITE_SLOTS],
        overflow: AtomicU64::new(0),
    };
}

static COUNTERS: [AtomicU64; COUNTER_COUNT] = [const { AtomicU64::new(0) }; COUNTER_COUNT];
static FAST_OPCODES: [AtomicU64; OPCODES] = [const { AtomicU64::new(0) }; OPCODES];
static GENERAL_OPCODES: [AtomicU64; OPCODES] = [const { AtomicU64::new(0) }; OPCODES];
static FAST_PAIRS: [AtomicU64; OPCODES * OPCODES] =
    [const { AtomicU64::new(0) }; OPCODES * OPCODES];
/// The opcode the fast loop dispatched last, plus one; zero after a loop
/// entry, so a pair never spans an exit.
static FAST_PREVIOUS: AtomicU64 = AtomicU64::new(0);
static OPERAND_CLASSES: [AtomicU64; OPCODES * 4] = [const { AtomicU64::new(0) }; OPCODES * 4];
static SITES: [Sites; 3] = [Sites::EMPTY, Sites::EMPTY, Sites::EMPTY];

#[inline(always)]
fn bump_cell(cell: &AtomicU64, amount: u64) {
    cell.store(cell.load(Relaxed).wrapping_add(amount), Relaxed);
}

/// Adds one to `counter` on the guest thread.
#[inline(always)]
pub fn bump(counter: Counter) {
    bump_cell(&COUNTERS[counter as usize], 1);
}

/// Adds `amount` to `counter` on the guest thread.
#[inline(always)]
pub fn add(counter: Counter, amount: u64) {
    bump_cell(&COUNTERS[counter as usize], amount);
}

/// Adds `amount` to `counter` from any thread.
pub fn add_shared(counter: Counter, amount: u64) {
    COUNTERS[counter as usize].fetch_add(amount, Relaxed);
}

/// Raises `counter` to `value` if it is lower, from any thread.
pub fn max_shared(counter: Counter, value: u64) {
    COUNTERS[counter as usize].fetch_max(value, Relaxed);
}

/// The current value of `counter`.
#[must_use]
pub fn get(counter: Counter) -> u64 {
    COUNTERS[counter as usize].load(Relaxed)
}

/// The address of `counter`'s 64-bit cell, for native code that adds to it
/// with a plain load and store. The cell lives for the whole process.
#[must_use]
pub fn address(counter: Counter) -> usize {
    std::ptr::from_ref(&COUNTERS[counter as usize]) as usize
}

/// Counts one fast-loop dispatch of `opcode` and the pair it forms with the
/// previous one.
#[inline(always)]
pub fn fast_dispatch(opcode: u8) {
    let opcode = usize::from(opcode) & (OPCODES - 1);
    bump(Counter::FastDispatches);
    bump_cell(&FAST_OPCODES[opcode], 1);
    let previous = FAST_PREVIOUS.load(Relaxed);
    if previous != 0 {
        bump_cell(&FAST_PAIRS[(previous as usize - 1) * OPCODES + opcode], 1);
    }
    FAST_PREVIOUS.store(opcode as u64 + 1, Relaxed);
}

/// Ends the current pair sequence: the next dispatch starts a new one.
#[inline(always)]
pub fn fast_break_pairs() {
    FAST_PREVIOUS.store(0, Relaxed);
}

/// Counts one general-loop dispatch of `opcode`.
#[inline(always)]
pub fn general_dispatch(opcode: u8) {
    bump(Counter::GeneralSteps);
    bump_cell(&GENERAL_OPCODES[usize::from(opcode) & (OPCODES - 1)], 1);
}

/// Counts an arithmetic or comparison `opcode` with operands of `class`.
#[inline(always)]
pub fn operand_class(opcode: u8, class: OperandClass) {
    bump_cell(
        &OPERAND_CLASSES[(usize::from(opcode) & (OPCODES - 1)) * 4 + class as usize],
        1,
    );
}

/// What [`site`] found at a site, compared with the site's previous
/// execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SiteChange {
    /// The site's first execution, or a site the table had no room for.
    First,
    Same,
    /// `b` changed, `a` did not.
    OnlyB,
    /// `a` changed (`b` may have too).
    A,
}

/// Records one execution of the site at `pc` of the chunk at address
/// `chunk` in `table`, which saw identities `a` and `b` and sets `flags`.
pub fn site(table: SiteTable, chunk: usize, pc: usize, a: u64, b: u64, flags: u64) -> SiteChange {
    let sites = &SITES[table as usize];
    let key = (chunk as u64 ^ (pc as u64).rotate_left(40))
        .wrapping_add(1)
        .max(1);
    let mut index = (key.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 50) as usize & (SITE_SLOTS - 1);
    for _ in 0..SITE_PROBES {
        let slot = &sites.slots[index];
        let stored = slot.key.load(Relaxed);
        if stored == 0 {
            slot.key.store(key, Relaxed);
            slot.count.store(1, Relaxed);
            slot.a.store(a, Relaxed);
            slot.b.store(b, Relaxed);
            slot.flags.store(flags, Relaxed);
            return SiteChange::First;
        }
        if stored == key {
            bump_cell(&slot.count, 1);
            let a_changed = slot.a.load(Relaxed) != a;
            let b_changed = slot.b.load(Relaxed) != b;
            if a_changed {
                bump_cell(&slot.a_changes, 1);
                slot.a.store(a, Relaxed);
            }
            if b_changed {
                bump_cell(&slot.b_changes, 1);
                slot.b.store(b, Relaxed);
            }
            slot.flags.store(slot.flags.load(Relaxed) | flags, Relaxed);
            return match (a_changed, b_changed) {
                (true, _) => SiteChange::A,
                (false, true) => SiteChange::OnlyB,
                (false, false) => SiteChange::Same,
            };
        }
        index = (index + 1) & (SITE_SLOTS - 1);
    }
    bump_cell(&sites.overflow, 1);
    SiteChange::First
}

/// Summary of one site table.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SiteSummary {
    pub sites: u64,
    pub executions: u64,
    pub overflow: u64,
    pub hot_sites: u64,
    pub hot_executions: u64,
    /// Sites, and their executions, by whether `a` and `b` ever changed:
    /// index 0 neither, 1 only `b`, 2 only `a`, 3 both.
    pub by_change: [(u64, u64); 4],
    /// Hot sites only, as `by_change`.
    pub hot_by_change: [(u64, u64); 4],
    /// Executions of sites whose `a` never changed, by that `a` when it is
    /// below 16 (an [`OperandClass`] for the arithmetic table).
    pub stable_a_executions: [u64; 16],
    /// `a` changes and `b` changes over all sites.
    pub a_changes: u64,
    pub b_changes: u64,
    /// Executions of sites by flag bit, for bits 0 to 7.
    pub flag_executions: [u64; 8],
}

/// Summarizes `table`.
#[must_use]
pub fn site_summary(table: SiteTable) -> SiteSummary {
    let sites = &SITES[table as usize];
    let mut summary = SiteSummary {
        overflow: sites.overflow.load(Relaxed),
        ..SiteSummary::default()
    };
    for slot in &sites.slots {
        if slot.key.load(Relaxed) == 0 {
            continue;
        }
        let count = slot.count.load(Relaxed);
        let a_changes = slot.a_changes.load(Relaxed);
        let b_changes = slot.b_changes.load(Relaxed);
        let class = usize::from(a_changes != 0) * 2 + usize::from(b_changes != 0);
        summary.sites += 1;
        summary.executions += count;
        summary.by_change[class].0 += 1;
        summary.by_change[class].1 += count;
        if count >= HOT_SITE {
            summary.hot_sites += 1;
            summary.hot_executions += count;
            summary.hot_by_change[class].0 += 1;
            summary.hot_by_change[class].1 += count;
        }
        let a = slot.a.load(Relaxed);
        if a_changes == 0 && a < 16 {
            summary.stable_a_executions[a as usize] += count;
        }
        summary.a_changes += a_changes;
        summary.b_changes += b_changes;
        let flags = slot.flags.load(Relaxed);
        for (bit, executions) in summary.flag_executions.iter_mut().enumerate() {
            if flags >> bit & 1 == 1 {
                *executions += count;
            }
        }
    }
    summary
}

fn json_string(out: &mut String, text: &str) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            character if (character as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", character as u32);
            }
            character => out.push(character),
        }
    }
    out.push('"');
}

fn histogram(out: &mut String, cells: &[AtomicU64], name: impl Fn(usize) -> String) {
    out.push('{');
    let mut first = true;
    for (index, cell) in cells.iter().enumerate() {
        let value = cell.load(Relaxed);
        if value == 0 {
            continue;
        }
        if !first {
            out.push(',');
        }
        first = false;
        json_string(out, &name(index));
        let _ = write!(out, ":{value}");
    }
    out.push('}');
}

fn change_classes(out: &mut String, classes: &[(u64, u64); 4]) {
    out.push('{');
    for (index, name) in ["stable", "b_changed", "a_changed", "both_changed"]
        .iter()
        .enumerate()
    {
        if index != 0 {
            out.push(',');
        }
        let (sites, executions) = classes[index];
        let _ = write!(
            out,
            "\"{name}\":{{\"sites\":{sites},\"executions\":{executions}}}"
        );
    }
    out.push('}');
}

/// Renders every counter, histogram and site summary as one JSON object.
///
/// `opcode_name` names an opcode byte. `build` lists the features of the
/// build; `extra` adds top-level string fields such as the execution mode.
#[must_use]
pub fn report_json(
    opcode_name: impl Fn(u8) -> String,
    build: &[&str],
    extra: &[(&str, &str)],
) -> String {
    let mut out = String::with_capacity(64 * 1024);
    out.push('{');
    out.push_str("\"schema\":");
    json_string(&mut out, SCHEMA);
    out.push_str(",\"build\":[");
    for (index, feature) in build.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        json_string(&mut out, feature);
    }
    out.push(']');
    for (key, value) in extra {
        out.push(',');
        json_string(&mut out, key);
        out.push(':');
        json_string(&mut out, value);
    }
    out.push_str(",\"counters\":{");
    for (index, counter) in Counter::ALL.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        json_string(&mut out, counter.name());
        let _ = write!(out, ":{}", get(*counter));
    }
    out.push_str("},\"counter_info\":{");
    for (index, counter) in Counter::ALL.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        json_string(&mut out, counter.name());
        out.push_str(":{\"kind\":");
        json_string(&mut out, counter.kind().name());
        out.push_str(",\"help\":");
        json_string(&mut out, counter.help());
        out.push('}');
    }
    out.push_str("},\"histograms\":{\"fast_opcodes\":");
    histogram(&mut out, &FAST_OPCODES, |index| opcode_name(index as u8));
    out.push_str(",\"general_opcodes\":");
    histogram(&mut out, &GENERAL_OPCODES, |index| opcode_name(index as u8));
    out.push_str(",\"fast_opcode_pairs\":");
    histogram(&mut out, &FAST_PAIRS, |index| {
        format!(
            "{}>{}",
            opcode_name((index / OPCODES) as u8),
            opcode_name((index % OPCODES) as u8)
        )
    });
    out.push_str(",\"operand_classes\":");
    histogram(&mut out, &OPERAND_CLASSES, |index| {
        format!(
            "{}.{}",
            opcode_name((index / 4) as u8),
            OperandClass::ALL[index % 4].name()
        )
    });
    out.push_str("},\"histogram_info\":{");
    out.push_str("\"fast_opcodes\":{\"kind\":\"dynamic_fast_interp\"},");
    out.push_str("\"general_opcodes\":{\"kind\":\"dynamic_general_interp\"},");
    out.push_str("\"fast_opcode_pairs\":{\"kind\":\"dynamic_fast_interp\",\"help\":\"adjacent dispatches inside one fast-loop entry, previous>next\"},");
    out.push_str("\"operand_classes\":{\"kind\":\"dynamic_fast_interp\",\"help\":\"arithmetic and comparison operand classes in the fast loop\"}}");
    out.push_str(",\"sites\":{");
    for (index, table) in SiteTable::ALL.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        let summary = site_summary(*table);
        json_string(&mut out, table.name());
        out.push_str(":{\"kind\":\"dynamic_fast_interp\",\"help\":");
        json_string(&mut out, table.help());
        let _ = write!(
            out,
            ",\"slots\":{SITE_SLOTS},\"hot_threshold\":{HOT_SITE},\"sites\":{},\"executions\":{},\"overflow\":{},\"hot_sites\":{},\"hot_executions\":{},\"a_changes\":{},\"b_changes\":{}",
            summary.sites,
            summary.executions,
            summary.overflow,
            summary.hot_sites,
            summary.hot_executions,
            summary.a_changes,
            summary.b_changes,
        );
        out.push_str(",\"by_change\":");
        change_classes(&mut out, &summary.by_change);
        out.push_str(",\"hot_by_change\":");
        change_classes(&mut out, &summary.hot_by_change);
        out.push_str(",\"stable_a_executions\":[");
        for (position, value) in summary.stable_a_executions.iter().enumerate() {
            if position != 0 {
                out.push(',');
            }
            let _ = write!(out, "{value}");
        }
        out.push_str("],\"flag_executions\":[");
        for (position, value) in summary.flag_executions.iter().enumerate() {
            if position != 0 {
                out.push(',');
            }
            let _ = write!(out, "{value}");
        }
        out.push_str("]}");
    }
    out.push_str("}}");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // The statics are process-wide, so each test uses counters and site
    // tables no other test in this module touches.

    #[test]
    fn counter_names_are_unique_and_prefixed() {
        let mut names: Vec<&str> = Counter::ALL.iter().map(|c| c.name()).collect();
        assert!(names.iter().all(|name| name.starts_with('h')
            && name.as_bytes()[1].is_ascii_digit()
            && name.as_bytes()[2] == b'.'));
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), Counter::ALL.len());
        for (index, counter) in Counter::ALL.iter().enumerate() {
            assert_eq!(*counter as usize, index);
        }
    }

    #[test]
    fn sites_track_changes_per_identity() {
        let table = SiteTable::Call;
        assert_eq!(site(table, 0x1000, 7, 1, 10, 0), SiteChange::First);
        assert_eq!(site(table, 0x1000, 7, 1, 10, 0), SiteChange::Same);
        site(table, 0x1000, 7, 1, 10, 0);
        assert_eq!(site(table, 0x1000, 7, 1, 11, 0b10), SiteChange::OnlyB);
        site(table, 0x2000, 7, 2, 20, 0);
        assert_eq!(site(table, 0x2000, 7, 3, 21, 0), SiteChange::A);
        let summary = site_summary(table);
        assert_eq!(summary.sites, 2);
        assert_eq!(summary.executions, 6);
        assert_eq!(summary.by_change[1], (1, 4));
        assert_eq!(summary.by_change[3], (1, 2));
        assert_eq!(summary.a_changes, 1);
        assert_eq!(summary.b_changes, 2);
        assert_eq!(summary.stable_a_executions[1], 4);
        assert_eq!(summary.flag_executions[1], 4);
    }

    #[test]
    fn report_is_json_with_every_counter() {
        add(Counter::IrExitAreaSlots, 5);
        let report = report_json(
            |op| format!("op{op}"),
            &["op-stats"],
            &[("execution", "tracing")],
        );
        assert!(report.starts_with("{\"schema\":\"oximond-op-stats/2\""));
        for counter in Counter::ALL {
            assert!(report.contains(&format!("\"{}\":", counter.name())));
        }
        assert!(report.contains("\"execution\":\"tracing\""));
        assert_eq!(
            report.matches('{').count(),
            report.matches('}').count(),
            "{report}"
        );
    }
}
