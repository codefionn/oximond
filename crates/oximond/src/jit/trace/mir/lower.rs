//! Trace IR to MIR, in one forward pass.
//!
//! The ABI is the one `emit.rs` documents: `entry(inputs, control,
//! exit_area)`, fuel in the control block, a tag mismatch returns `-1`, a
//! guard stores its exit's slots into the exit area and the fuel into the
//! control block and returns the exit's index, the loop-back spends one unit
//! of fuel and leaves through the poll exit when it is gone. A counted tree
//! (a loop under a count hook) carries the hook's budget beside the fuel the
//! same way: loaded at entry, paid at each loop-back and link, stored by
//! every exit (see `Lowerer::budget`).
//!
//! How it maps onto `oximond_mir`:
//!
//! - The exit plan is the plan Cranelift uses. Snapshot `k` of the MIR
//!   function is exit descriptor `k`, and the code its exit returns is `k`. The slots of a descriptor, plus the fuel, are the values of the
//!   snapshot. A tag mismatch is a snapshot with no values that returns `-1`.
//! - Booleans are 64-bit integers holding 0 or 1. A `Compare` is not
//!   materialized: it stays a pending condition (`Cmp`) that a guard or a
//!   select consumes directly, and becomes a `Select` of 1 and 0 only when
//!   something else reads it.
//! - The root is region 0: entry checks, loads, preheader, the loop, the
//!   back edge with the poll guard. Each side of the tree is a region of its
//!   own. A guard that has a side passes the side's entry inputs, then the
//!   fuel, the inputs pointer and the control pointer, as the snapshot of the
//!   guard, and the side reloads what it needs of the root (table words, rows)
//!   from those pointers.
//! - When the root's loop is versioned (`RangePlan`), region 1 is the guarded
//!   loop that a failed range proof enters, and the sides start at region 2
//!   (`side_of`, with a copy of a side that links straight to the fast loop
//!   or the guarded loop after it). The guarded loop continues from the
//!   values held at the range checks, passed by the failing check. Its exits use snapshots of their own that
//!   return the same exit code as the descriptor.
//! - A function trace (`TraceIr::function`) is an activation that returns,
//!   calls itself and tail calls itself, with an entry of its own placed
//!   after it: see `lower/function.rs`. Its exits return `-1 - index`.
//! - What the lowering does not handle is rejected with a reason, and the
//!   tree goes to Cranelift.

use super::super::compile::{ExitPlan, LoweredMember, TraceCompileOptions};
use super::super::emit::{
    TrustedKey, closure_row_words, header_binding_of, keeps_slots, rows_masked, trace_admit_table,
    trace_materialize_exit, trace_mod_fmod53, trace_mod_fmod54, trace_number_bits, trace_pow,
    trusted_array_key,
};
use super::super::exit::{ExitSlot, TraceControl, TraceInputSlot, entry_tag, row_stride_of};
use crate::compat::FloatModulo;
use crate::compat::TracingOptFlags;
use crate::jit::OptimizationTier;
use crate::jit_ir::Comparison;
use crate::jit_ir::trace::ValueDef;
use crate::jit_ir::trace::range::{
    MAX_RANGE_TERM, MAX_VERSIONED_INSTRUCTIONS, RangeOptions, carried_for_links, fast_link,
    resizes_beside_sets, side_origins, slow_link, versioning_pays,
};
use crate::jit_ir::trace::{
    ArithmeticOp, BindingKind, GuardId, LoopParamInit, META_FIELD_KEY, NumericOp,
    ROOT_STRING_LANE_BASE, RegisterEntry, TraceIr, TraceOp, TraceScalar, TraceTerminator,
    TraceType, ValueId, WideConversion, meta_word,
};
use crate::jit_ir::trace::{RangeInduction, RangePlan, RangeShift, RangeStep, pass_through_params};
use crate::runtime::{
    NATIVE_VALUE_PAYLOAD_OFFSET, NATIVE_VALUE_TAG_BOOLEAN, NATIVE_VALUE_TAG_CLOSURE,
    NATIVE_VALUE_TAG_INTEGER, NATIVE_VALUE_TAG_NIL, NATIVE_VALUE_TAG_NUMBER,
    NATIVE_VALUE_TAG_STRING, NATIVE_VALUE_TAG_TABLE, NATIVE_VALUE_TAG_THREAD, NativeTableLayout,
};
use oximond_mir::ir::{COLD, Cc, Func, Inst, NONE, Op, SnapVal, Ty, Val, is_coldable};
use std::mem;

mod bytes;
mod calls;
pub(super) mod function;
mod made;
pub(super) mod modulo;
mod probe;
mod sets;
mod steps;

#[cfg(test)]
pub(in super::super) use probe::adopted_keys;

// Links lowered straight to the fast loop on this thread, for the tests.
#[cfg(test)]
thread_local! {
    pub(in super::super) static FAST_LINKS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    pub(in super::super) static GUARDED_LINKS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    /// Fast loops lowered with one closing compare (`Fold`).
    pub(in super::super) static FOLDED_LOOPS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// `2^52`: adding it to a float in `[1, 2^52)` leaves the integer part in the
/// low bits, and subtracting it again gives the float back only if it was
/// integral.
const FLOAT_INTEGER_MAGIC: f64 = 4_503_599_627_370_496.0;

/// Why a tree is not compiled here.
#[derive(Clone, Copy, Debug)]
pub(super) struct Reject {
    pub(super) what: &'static str,
    /// The trace instruction (in its member's preheader and body, numbered
    /// together), or `usize::MAX` for the tree.
    pub(super) at: usize,
}

impl Reject {
    pub(super) const fn new(what: &'static str) -> Self {
        Self {
            what,
            at: usize::MAX,
        }
    }
}

type Res<T> = Result<T, Reject>;

/// A comparison that no value holds yet.
#[derive(Clone, Copy)]
struct Cmp {
    a: Val,
    b: Val,
    cc: Cc,
    float: bool,
    live: bool,
}

const NO_CMP: Cmp = Cmp {
    a: NONE,
    b: NONE,
    cc: Cc::Eq,
    float: false,
    live: false,
};

/// Whether `op` writes no table and runs no other code, so the strings in the
/// array slots that the keys probed before it borrow are the same after it.
const fn keeps_probed_keys(op: &TraceOp) -> bool {
    matches!(
        op,
        TraceOp::Constant(_)
            | TraceOp::Move { .. }
            | TraceOp::Negate { .. }
            | TraceOp::IntToFloat { .. }
            | TraceOp::ToWide { .. }
            | TraceOp::Numeric { .. }
            | TraceOp::NativeUnary { .. }
            | TraceOp::NativeBinary { .. }
            | TraceOp::Arithmetic { .. }
            | TraceOp::Compare { .. }
            | TraceOp::Select { .. }
            | TraceOp::Truthy { .. }
            | TraceOp::Not { .. }
            | TraceOp::GuardType { .. }
            | TraceOp::GuardBranch { .. }
            | TraceOp::ArrayLoad { .. }
            | TraceOp::NumberLoad { .. }
            | TraceOp::LaneArrayLoad { .. }
            | TraceOp::NumberTag { .. }
            | TraceOp::BoolAnd { .. }
            | TraceOp::NumberModulo { .. }
            | TraceOp::HashLoad { .. }
            | TraceOp::FieldLoad { .. }
            | TraceOp::TableFieldLoad { .. }
            | TraceOp::MetaGuard { .. }
            | TraceOp::StringLength { .. }
            | TraceOp::StringByte { .. }
            | TraceOp::TableLength { .. }
    )
}

/// An offset into an array: a constant, or a value.
#[derive(Clone, Copy)]
enum Off {
    K(i64),
    V(Val),
    /// A value plus a constant.
    VK(Val, i64),
}

/// An array slot a straight-line access already reached.
#[derive(Clone, Copy)]
struct SlotFact {
    binding: u32,
    key: u32,
    off: Off,
    ty: TraceType,
}

/// A number of run-time subtype in a side's values: the place in the list, the
/// type the side takes, the float and the flag that says it is an integer.
#[derive(Clone, Copy)]
struct Conv {
    at: usize,
    ty: TraceType,
    float: Val,
    integer: Val,
}

/// The side an exit leads to, when it takes numbers of run-time subtype: the
/// exit, the side's values and the conversions that fill the empty places.
#[derive(Clone)]
struct PendingSide {
    exit: usize,
    list: Vec<Val>,
    convs: Vec<Conv>,
}

/// The words the entry resolved for a binding, as MIR values: the address
/// (or the typed word), the length and the length as a float. `NONE` where
/// the binding has no such word. For a string index `len` is its mask (with
/// the flags) and `lenf` is the mask as the probe uses it: an integer,
/// shifted to mask the offset of a bucket.
#[derive(Clone, Copy)]
struct Words {
    base: Val,
    len: Val,
    lenf: Val,
}

/// The snapshot of a guard whose side takes numbers of a run-time subtype: the
/// guard makes its own (see `guard_by_subtype`).
const BY_SUBTYPE: u32 = u32::MAX - 1;

/// The word of the control block that holds the count hook's budget
/// (`TraceControl::budget`), for a snapshot's `CONTROL_SLOT`.
const BUDGET_WORD: u32 = 1;
const _: () = assert!(mem::offset_of!(TraceControl, budget) == 8 * BUDGET_WORD as usize);

const NO_WORDS: Words = Words {
    base: NONE,
    len: NONE,
    lenf: NONE,
};

/// What the lowerer keeps between trees, so a steady-state compile allocates
/// nothing.
#[derive(Default)]
pub(super) struct Scratch {
    vals: Vec<Val>,
    cmps: Vec<Cmp>,
    konst: Vec<Option<TraceScalar>>,
    words: Vec<Words>,
    used: Vec<u8>,
    slots: Vec<SlotFact>,
    rows: Vec<(Val, Val)>,
    /// The array words of the tables met by lane that the straight-line code
    /// read: the lane and its words (see `lane_array_words`).
    lane_words: Vec<(Val, Words)>,
    /// The borrowed string keys the straight-line code probed an index with:
    /// the key's value index, the string's storage and where its probe
    /// starts (see `key_hash` in `probe.rs`). Dropped wherever the lane words
    /// are, and at any operation that may write a table. The last word is
    /// the number of the probe that read them.
    keys: Vec<(u32, Val, Val, u32)>,
    /// The probes of string indexes lowered so far.
    probes: u32,
    /// The numbers the straight-line code has as integers for `StringByte`
    /// and the bytes it loaded (see `bytes.rs`). Dropped wherever `keys`
    /// is, except after a table write: a bound string does not change.
    wholes: Vec<bytes::WholeFact>,
    bytes: Vec<bytes::ByteFact>,
    /// The address of the value a probed lookup found, while its type guard
    /// is being made: a side that starts with the same lookup takes it
    /// instead of probing again (see `takes_probe` in `probe.rs`).
    hand: Option<Val>,
    /// In such a side: the address its parent's guard passed and the place
    /// in the body of the lookup that takes it, until it does.
    handed: Option<(Val, usize)>,
    /// The operation lowered last may have written a table, so `keys` is
    /// dropped before the next one (the writing operation itself may still
    /// probe with a key read before it).
    keys_stale: bool,
    /// The upvalue cells the straight-line code reached in line: the
    /// closure's lane, the cell's index, the cell's address and the type its
    /// tag was checked for (see `closure_cell` in `calls.rs`). An access of
    /// the same cell for the same type takes the address and makes no check
    /// again. The last value is the number the cell holds, when the trace
    /// read or stored it and no store through another closure, which may
    /// share the cell, came after (`NONE` otherwise): a read takes it.
    /// Dropped where `keys` is, and before the operation that follows one
    /// that may run other code (`cells_stale`): only that could close,
    /// retype or replace a cell.
    cells: Vec<(Val, usize, Val, TraceType, Val)>,
    cells_stale: bool,
    fields: Vec<(Val, u32, Val)>,
    filled: Vec<bool>,
    side_snaps: Vec<u32>,
    /// The numbers of run-time subtype the last [`Lowerer::side_values`]
    /// found: the place in the list, the type the side takes, the float and
    /// the flag that says it is an integer.
    convs: Vec<Conv>,
    /// The virtual tables the last [`Lowerer::side_values`] found in the
    /// registers the side reads: the place in the list, and the table's
    /// place among the tables the exit builds
    /// (`Snapshot::virtual_place`). The side has the run build them where it
    /// starts (`Lowerer::build_entry_tables`), from the exit's state, which
    /// the guard hands it after its other values.
    bridges: Vec<(usize, usize)>,
    /// Per exit with a side that takes such numbers: the side's values with
    /// the places `convs` fills left empty, and those. The guard makes the
    /// conversions on its way to the side, not in the loop.
    pending: Vec<PendingSide>,
    /// The snapshots the guarded version of a versioned loop made, per exit.
    slow_snaps: Vec<u32>,
    buf: Vec<SnapVal>,
    list: Vec<Val>,
    /// For a root with a `Join`: the new value of each value that was defined
    /// above it and is used below (`NONE` for one not seen yet), and the
    /// values the cells keep across it.
    remap: Vec<Val>,
    kept: Vec<(Val, u32)>,
    /// Scratch: the values `visit_held` found.
    held: Vec<Val>,
    /// The words a helper call loaded beyond its first result: the value of
    /// the call, the result's index and the loaded word.
    hres: Vec<(u32, u8, Val)>,
    /// For each `NumberLoad`: its result's value index and whether the slot
    /// held an integer (a 0 or 1), for `NumberTag`.
    numtags: Vec<(u32, Val)>,
    /// The tables a `TABLE_NEW` call of the member made and left to be
    /// filled in place.
    made: Vec<made::MadeTable>,
    /// Where the loop parameters of the member being lowered start.
    inits: Vec<Val>,
    /// For a loop side: the frame cells and the values they keep for the
    /// code after its `Join`.
    carried: Vec<(Val, u32)>,
    /// Per member: the loop side that a nested side rejoins is `true`.
    joined: Vec<bool>,
    /// Per member: the member whose guard or terminal exit enters it.
    parent: Vec<usize>,
    /// For the member being lowered, per value: only exits, sides and links
    /// name it (see [`Lowerer::mark_cold`]).
    cold: Vec<bool>,
    /// The same for the fast version of the root (see
    /// [`Lowerer::mark_cold_fast`]).
    cold_fast: Vec<bool>,
    /// What `mark_cold` found the loop keeps, per value.
    hot: Vec<bool>,
    /// Per member, the region (side number, the root being none) its code is
    /// in, and the region of its copy for links that go straight to the fast
    /// loop (`u32::MAX` for a member with none, see [`Lowerer::link_fast`]).
    region_of: Vec<u32>,
    fast_region: Vec<u32>,
    /// Per member, the region of its copy for the guards of the guarded loop
    /// when its link may go straight to the guarded loop's header
    /// (`u32::MAX` for none, see [`Lowerer::link_slow`]).
    slow_region: Vec<u32>,
    /// What the lowering held when the range checks began, and the values
    /// among it the guarded loop is entered with (the fuel first) and the
    /// constants it makes again (see [`Lowerer::guard_entry`]).
    saved: Saved,
    entry_vals: Vec<Val>,
    /// For a fast loop that closes on one compare (`Fold`): the entry values
    /// under the names they have where the lowering is, which the loop
    /// enters the guarded loop with when it stops at the end of its arrays.
    /// Empty otherwise.
    entry_now: Vec<Val>,
    /// The same below the fast loop's `Join`: the cell and the type of each
    /// (see [`Lowerer::join_fast`]). Empty when the loop has no such `Join`.
    entry_cells: Vec<(u32, Ty)>,
    entry_consts: Vec<(Val, Ty, i64)>,
    /// For a versioned root that sides rejoin: the loop parameters the fast
    /// loop's join carries (what the loop-back changes and what a link may
    /// change without the checks running again), and the parameter of each
    /// induction whose integer copy the join carries after them.
    changing: Vec<usize>,
    shadow_roots: Vec<usize>,
    /// For a versioned root: its guards that belong to an access whose key
    /// the range plan proves (see [`Lowerer::index_within`]).
    proven_guards: Vec<usize>,
    /// The keys the range plan proves, each with whether its induction
    /// moves up.
    proven_keys: Vec<(usize, bool)>,
    /// Scratch of the proof that a link may skip the checks.
    origins: Vec<Option<ValueId>>,
    /// For the root of a function tree, per value: the input slot plus one
    /// of a cell or global entry input read where it is used, 0 for others.
    lazy_in: Vec<u32>,
    /// Per side region (the number `mark_side` gives it), what every entry
    /// into it passes at each place of its list, and whether one has (see
    /// [`Lowerer::note_entry`]).
    entry_args: Vec<Vec<EntryArg>>,
    entered: Vec<bool>,
    /// Per MIR value, the constant instruction that makes it (`NONE` for
    /// none), for the instructions up to `consts_seen`.
    const_def: Vec<Val>,
    consts_seen: usize,
}

/// What the entries into a side pass at one place of its list.
#[derive(Clone, Copy, PartialEq, Eq)]
enum EntryArg {
    /// Values that differ, or that no rule below covers.
    Other,
    /// The same constant from every entry: its type and bits.
    Const(Ty, i64),
    /// The value of an earlier place, from every entry.
    Same(u32),
}

/// Which code of a side is being lowered: its own, or the copy a guard of the
/// fast loop or one of the guarded loop enters, whose link goes straight to
/// that loop's `Join`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SideCopy {
    Own,
    Fast,
    Guarded,
}

/// The lowering's tables at one point, for the guarded loop to continue from.
#[derive(Default)]
struct Saved {
    inputs: Val,
    control: Val,
    rows: Val,
    vals: Vec<Val>,
    cmps: Vec<Cmp>,
    konst: Vec<Option<TraceScalar>>,
    words: Vec<Words>,
    slots: Vec<SlotFact>,
    rows_of: Vec<(Val, Val)>,
    fields: Vec<(Val, u32, Val)>,
    hres: Vec<(u32, u8, Val)>,
    numtags: Vec<(u32, Val)>,
    inits: Vec<Val>,
}

/// Whether the preheader of `ir` calls into the VM or stores into a table.
fn preheader_calls(ir: &TraceIr) -> bool {
    ir.preheader.iter().any(|instruction| {
        matches!(
            instruction.op,
            TraceOp::CallHelper { .. }
                | TraceOp::HostCall { .. }
                | TraceOp::ArraySet { .. }
                | TraceOp::ReleaseStrings { .. }
        )
    })
}

/// Whether the loop-back of `ir` passes some loop parameter on unchanged:
/// only a side's link changes it. Cranelift runs the preheader of such a root
/// or loop side at its outer header, which every link jumps to
/// (`pass_through_params`).
fn passes_params(ir: &TraceIr) -> bool {
    !pass_through_params(ir).is_empty()
}

#[allow(
    clippy::struct_excessive_bools,
    reason = "independent facts of the tree, as in the Cranelift lowering"
)]
pub(super) struct Lowerer<'a, 'f> {
    members: &'a [LoweredMember<'a>],
    plan: &'a ExitPlan,
    count_links: bool,
    /// The target has a fused multiply-add (`FNma`), which the float
    /// remainder uses where Cranelift does.
    fused: bool,
    /// The target shifts by a register (`Shl`, `Shr` and `Sar` with a count
    /// that is not a constant).
    shifts: bool,
    f: &'f mut Func,
    s: &'f mut Scratch,
    /// Bytes of one row of the run's row storage.
    row_bytes: i64,
    /// Words of a closure's row, 0 when no helper call reads a cell in line.
    closure_words: usize,
    /// A row's last word is its presence mask.
    row_masked: bool,
    /// The member being lowered, its trace and the values it reads.
    member: usize,
    ir: &'a TraceIr,
    inputs: Val,
    control: Val,
    fuel: Val,
    /// For a counted tree (`TraceIr::counted`), the count hook's budget at
    /// the start of the iteration being lowered, which every exit of the
    /// iteration stores into the control block beside the fuel; `NONE`
    /// otherwise. It travels with the fuel: a loop parameter after it, a
    /// side's entry value after it, a `Join` parameter after it.
    budget: Val,
    /// The tree is counted.
    counted: bool,
    rows: Val,
    /// For messages: the instruction being lowered.
    at: usize,
    /// Some side rejoins the loop: the root has an outer header.
    rejoin: bool,
    /// The member has a call that moves what the binding words name, so it
    /// reads them where it uses them, not once at entry.
    lazy: bool,
    /// The options let helpers take a prepared pattern.
    string_constants: bool,
    /// Tables met in an array are looked up in the run's lane index before
    /// the dispatcher is asked (no function trace, no table made, no
    /// coroutine resumed).
    lane_probe: bool,
    /// The key of an array step that every step after the first can take as
    /// it is (`trusted_array_key` of `emit.rs`).
    trusted: Option<TrustedKey>,
    /// The range plan the root's loop is lowered in two versions for
    /// (`root` takes it).
    range: Option<RangePlan>,
    /// The fast version of the root's loop while it is lowered.
    fast: Option<FastLoop>,
    /// The root's loop was versioned: region 1 holds the guarded loop the
    /// failed proofs enter.
    versioned: bool,
    /// The fast loop closes on one compare (`Fold`): the guarded loop is
    /// entered from inside it, with newer loop parameters than the range
    /// checks pass (see [`Self::guard_entry`]).
    folded: bool,
    /// The guarded version after a failed proof is being lowered, or a copy
    /// of a side (see [`Self::link_fast`]): its exits take snapshots of their
    /// own, since the values differ from the first lowering's.
    slow: bool,
    /// The copy of a side whose link goes straight to the fast loop is being
    /// lowered.
    fast_side: bool,
    /// The copy of a side whose link goes straight to the guarded loop is
    /// being lowered.
    slow_side: bool,
    /// The guard being made is a key check of the guarded loop on a key the
    /// range plan proves: its side is entered through its own code, not its
    /// copy for the guarded loop (see [`Self::index_within`]).
    key_miss: bool,
    /// The `TABLE_NEW` call being lowered makes the array part its length
    /// long for the fills that follow (`made.rs`).
    prefill_call: bool,
    /// The tree is a function trace: what its activation holds.
    function: Option<function::Activation>,
}

/// The fast version of the root's loop while it is lowered: the range plan
/// it relies on, and the integer copies of its inductions.
struct FastLoop {
    plan: RangePlan,
    /// The integer copy of each induction of the plan, which every
    /// parameter of the induction equals, times 16 (the size of a slot, so
    /// it is the offset of the slot it names): the induction, and the copy's
    /// value in the iteration being lowered. In the order of the header's
    /// extra parameters.
    shadows: Vec<(usize, Val)>,
    /// Per induction, an invariant step as an integer (`NONE` for a constant
    /// one), times 16 for an induction.
    steps: Vec<Val>,
    /// Each shift of the plan as an integer, times 16 for an induction.
    shifts: Vec<(RangeShift, Val)>,
    /// The guards that hold the next value of an induction to its bound,
    /// as compares of the integer copy (see [`Lowerer::bound_limits`]).
    limits: Vec<Limit>,
    /// Per load keyed on an open induction (binding and key offset): the
    /// first value of the copy whose key is past the end of the array. The
    /// load leaves through its guard unless the copy is below it.
    open_ends: Vec<(u32, i64, Val)>,
    /// Per induction: the loop carries the copy alone, no float and no
    /// integer of it. Its parameters are made from the copy where an exit
    /// reads them (see [`Lowerer::mark_cold_fast`]).
    dropped: Vec<bool>,
    /// Per copy: its value at the loop-back, once something in the iteration
    /// being lowered has made it (`NONE` before), so the next value of a
    /// dropped integer induction and the loop-back share one sum.
    stepped: Vec<Val>,
    /// The loop closes on one compare, see [`Fold`].
    fold: Option<Fold>,
    /// The loop's header is lowered: accesses with proven keys skip their
    /// bounds checks.
    active: bool,
    /// The scaled indexes the proven accesses of the iteration being lowered
    /// computed: the copy, the shift (or `NONE`) and the index times 16.
    scaled: Vec<(Val, Val, Val)>,
}

/// A guard of the fast loop on the next value of an induction against
/// the induction's bound, as a compare of the integer copy the iteration
/// starts with: the guard passes when `copy cc limit`.
#[derive(Clone, Copy)]
struct Limit {
    /// The compare the guard tests, and the outcome it expects.
    condition: ValueId,
    expected: bool,
    induction: usize,
    cc: Cc,
    at_start: Val,
    /// The same limit for the copy the next iteration starts with: the
    /// guard passes when `next copy cc at_next`.
    at_next: Val,
}

/// The fast loop closes on one compare (see [`Lowerer::close_folded`]): the
/// guard that holds an induction to its bound is the last thing the
/// body does, so the end of the loop, the end of the arrays the induction
/// indexes and the end of the fuel are one limit on the integer copy, made
/// where the loop starts.
#[derive(Clone, Copy)]
struct Fold {
    induction: usize,
    upward: bool,
    /// The bound guard the loop's end stands for.
    guard: GuardId,
    condition: ValueId,
    expected: bool,
    /// The last copy (times 16) whose keys are all inside their arrays
    /// (the first one, for a downward loop). Made below the fast `Join`.
    range_end: Val,
    /// The limit of the run of the loop being lowered: the next copy past it
    /// ends the run.
    end: Val,
}

/// Most iterations one run of a folded loop makes on fuel that is short of
/// its array before it goes on in the guarded loop, for a step that is no
/// constant: the product with the step (at most 2^35 as a scaled integer)
/// stays far inside an `i64`.
const FOLD_FUEL_CAP: i64 = 1 << 26;

/// The largest magnitude of a float induction's bound that the fast loop
/// compares as an integer. A bound past it takes the guarded version.
const MAX_INTEGER_BOUND: f64 = 1_099_511_627_776.0;

/// Most members (the root and its sides) a tree may have here.
const MAX_MEMBERS: usize = 80;

/// A binding word's use: the address and length, and the length as a float.
const USE_WORDS: u8 = 1;
const USE_FLOAT: u8 = 2;

/// Whether loop parameter `index` holds the value of `induction`'s root at
/// every header of the fast loop: the root itself, or a parameter the range
/// checks prove to start as the root does and which the loop-back steps with
/// it (see `Lowerer::fast_twin`).
fn rides(ir: &TraceIr, induction: &RangeInduction, args: &[ValueId], index: usize) -> bool {
    let params = &ir.loop_params;
    index == induction.root
        || (index > induction.root
            && induction.params.contains(&index)
            && args[index] == args[induction.root]
            && params[index].ty == params[induction.root].ty
            && params[index].value != args[index])
}

/// The counter of the compare `op` when it is the overflow test of a counted
/// integer loop, `MAX - step < r` or `ceiling < r` with a ceiling past any
/// key, on a parameter `r` of an upward integer induction of `plan` (see
/// `Lowerer::overflow_proven`).
fn overflow_counter(ir: &TraceIr, plan: &RangePlan, op: &TraceOp) -> Option<ValueId> {
    const BEYOND_ANY_KEY: i64 = 1 << 60;
    let TraceOp::Compare {
        comparison: Comparison::LessThan,
        left: ceiling,
        right: counter,
    } = op
    else {
        return None;
    };
    let induction = plan.inductions.iter().find(|induction| {
        !induction.float
            && induction.upward
            && induction
                .params
                .iter()
                .any(|param| ir.loop_params[*param].value == *counter)
    })?;
    let proven = match defining_op(ir, *ceiling) {
        Some(TraceOp::Constant(TraceScalar::Integer(ceiling))) => *ceiling >= BEYOND_ANY_KEY,
        Some(TraceOp::Arithmetic {
            op: ArithmeticOp::Subtract,
            left,
            right,
        }) => {
            matches!(
                defining_op(ir, *left),
                Some(TraceOp::Constant(TraceScalar::Integer(i64::MAX)))
            ) && induction.step
                == RangeStep::Invariant {
                    value: *right,
                    negated: false,
                }
        }
        _ => false,
    };
    proven.then_some(*counter)
}

/// The parameter `op` reads when `op` is the sum that makes the next value
/// of an integer induction of `plan` that is dropped (see
/// `Lowerer::mark_cold_fast`): the fast loop steps the copy instead.
fn stepped_parameter(
    ir: &TraceIr,
    plan: &RangePlan,
    dropped: &[bool],
    args: &[ValueId],
    op: &TraceOp,
) -> Option<ValueId> {
    let TraceOp::Arithmetic { left, right, .. } = op else {
        return None;
    };
    plan.inductions
        .iter()
        .zip(dropped)
        .filter(|(induction, dropped)| **dropped && !induction.float)
        .find_map(|(induction, _)| {
            let made_here =
                defining_op(ir, args[induction.root]).is_some_and(|next| std::ptr::eq(next, op));
            if !made_here {
                return None;
            }
            [*left, *right].into_iter().find(|operand| {
                (0..ir.loop_params.len()).any(|index| {
                    ir.loop_params[index].value == *operand && rides(ir, induction, args, index)
                })
            })
        })
}

/// The operation that defines `value` in the body or the preheader of `ir`.
fn defining_op(ir: &TraceIr, value: ValueId) -> Option<&TraceOp> {
    match ir.values.get(value.index())?.def {
        ValueDef::Instruction(index) => ir.body.get(index).map(|instruction| &instruction.op),
        ValueDef::Preheader(index) => ir.preheader.get(index).map(|instruction| &instruction.op),
        _ => None,
    }
}

/// The integer twin of a guard on the compare of the next value `n` of a
/// induction with its bound `b` (see `Lowerer::bound_limits`): whether
/// passing the guard keeps `n` at or below the limit (at or above it
/// otherwise), whether the limit rounds `b` down (up otherwise), and what is
/// added to the rounded bound. `next_left` says `n` is the left operand,
/// `strict` that the compare is `<`.
#[expect(
    clippy::match_same_arms,
    reason = "one row for each shape of the compare"
)]
fn limit_shape(next_left: bool, strict: bool, expected: bool) -> (bool, bool, i64) {
    match (next_left, strict, expected) {
        // n <= b, n < b
        (true, false, true) => (true, true, 0),
        (true, true, true) => (true, false, -1),
        // not (n <= b): n >= floor(b) + 1; not (n < b): n >= ceil(b)
        (true, false, false) => (false, true, 1),
        (true, true, false) => (false, false, 0),
        // b <= n, b < n
        (false, false, true) => (false, false, 0),
        (false, true, true) => (false, true, 1),
        // not (b <= n): n <= ceil(b) - 1; not (b < n): n <= floor(b)
        (false, false, false) => (true, false, -1),
        (false, true, false) => (true, true, 0),
    }
}

/// Whether `op` only computes a value: it stores nothing, calls nothing and
/// cannot leave the trace, so a guard may move past it.
fn computes_only(op: &TraceOp) -> bool {
    matches!(
        op,
        TraceOp::Constant(_)
            | TraceOp::Move { .. }
            | TraceOp::Negate { .. }
            | TraceOp::Arithmetic { .. }
            | TraceOp::IntToFloat { .. }
            | TraceOp::Compare { .. }
            | TraceOp::Truthy { .. }
            | TraceOp::Not { .. }
            | TraceOp::Select { .. }
            | TraceOp::BoolAnd { .. }
    )
}

/// When `op` compares the value an induction of `plan` is held to its bound
/// by with that bound: the induction, the value, and whether it is the next
/// value (the value the iteration starts with otherwise, for an induction
/// that tests first, see `RangeInduction::tested`).
fn bound_compare(
    ir: &TraceIr,
    plan: &RangePlan,
    args: &[ValueId],
    op: &TraceOp,
) -> Option<(usize, ValueId, bool)> {
    let TraceOp::Compare {
        comparison: Comparison::LessThan | Comparison::LessEqual,
        left,
        right,
    } = op
    else {
        return None;
    };
    plan.inductions
        .iter()
        .enumerate()
        .find_map(|(index, induction)| {
            let next = args[induction.root];
            let held = if induction.tested.is_some() {
                ir.loop_params[induction.root].value
            } else {
                next
            };
            (matches!(
                ir.values.get(next.index()).map(|info| info.def),
                Some(ValueDef::Instruction(_))
            ) && ((*left == held && Some(*right) == induction.bound)
                || (*right == held && Some(*left) == induction.bound))
                && left != right)
                .then_some((index, held, induction.tested.is_none()))
        })
}

fn class(ty: TraceType) -> Option<Ty> {
    match ty {
        TraceType::Float => Some(Ty::F64),
        TraceType::Nil
        | TraceType::Boolean
        | TraceType::Integer
        | TraceType::Table
        | TraceType::Closure
        | TraceType::Thread
        | TraceType::String
        | TraceType::Int64
        | TraceType::UInt64
        // The address of the table slot that holds the string.
        | TraceType::StringRef => Some(Ty::I64),
        TraceType::Userdata | TraceType::Any => None,
    }
}

fn tag_of(ty: TraceType) -> Option<i64> {
    match ty {
        TraceType::Float => Some(i64::from(NATIVE_VALUE_TAG_NUMBER)),
        TraceType::Integer => Some(i64::from(NATIVE_VALUE_TAG_INTEGER)),
        TraceType::Boolean => Some(i64::from(NATIVE_VALUE_TAG_BOOLEAN)),
        _ => None,
    }
}

/// Whether `x cc y` holds for two 64-bit integers.
fn int_holds(cc: Cc, x: i64, y: i64) -> bool {
    let (ux, uy) = (x.cast_unsigned(), y.cast_unsigned());
    match cc {
        Cc::Eq => x == y,
        Cc::Ne => x != y,
        Cc::Lt => x < y,
        Cc::Le => x <= y,
        Cc::Gt => x > y,
        Cc::Ge => x >= y,
        Cc::Ult => ux < uy,
        Cc::Ule => ux <= uy,
        Cc::Ugt => ux > uy,
        Cc::Uge => ux >= uy,
        // Float conditions: not folded.
        Cc::Nlt | Cc::Nle | Cc::Ngt | Cc::Nge => false,
    }
}

fn offset(value: usize) -> i64 {
    i64::try_from(value).expect("trace ABI offset fits")
}

impl<'a> Lowerer<'a, '_> {
    /// Lowers the whole tree into `f`.
    pub(super) fn run(
        members: &'a [LoweredMember<'a>],
        plan: &'a ExitPlan,
        options: TraceCompileOptions,
        tier: OptimizationTier,
        (fused, shifts): (bool, bool),
        f: &mut Func,
        s: &mut Scratch,
    ) -> Res<bool> {
        let root = members[0].ir;
        let function_tree = root.function.is_some();
        if function_tree {
            // An activation's entry and sides have no place for the count
            // hook's budget yet.
            if root.counted {
                return Err(Reject::new("counted function trace"));
            }
            function::check_tree(members, plan)?;
        }
        let rejoin = members[1..]
            .iter()
            .any(|member| matches!(member.ir.terminator, TraceTerminator::Link { .. }));
        // A root with parameters its loop passes on unchanged is one the tree
        // widened for sides that rejoin it. Cranelift's code for it calls
        // the sides compiled apart natively through the exit cells, which
        // this backend does not do (it returns the exit index and the run
        // loop calls the side: correct, but slower, and the tests of the
        // native link assert it), so a root with no side of its own that
        // rejoins stays with Cranelift. A tree that has such a side is
        // compiled here: the parameters it passes on are the ones its sides
        // change.
        if !rejoin
            && !function_tree
            && let TraceTerminator::LoopBack { args } = &root.terminator
            && root
                .loop_params
                .iter()
                .zip(args)
                .any(|(param, arg)| param.value == *arg)
            && !root.counted
            && super::super::compile::link_slots_of(root).is_some()
        {
            return Err(Reject::new("root that links to sides compiled apart"));
        }
        // A side that rejoins the loop hands the root new values for its
        // loop parameters through a `Join` of the root, which sits where
        // Cranelift's outer header does (see `Lowerer::join_root`). A root
        // that ends in front of an inner loop instead of looping (an outer
        // root, see "Outer roots" in `vm::jit::trace`) has that header too:
        // its body runs again from the `Join` for each link, the next outer
        // iteration.
        // A loop side that passes some parameter on unchanged has its `Join`
        // before its preheader, as Cranelift has the preheader at its outer
        // header then, so it runs again at each rejoin. What a preheader only
        // computes or reads may be run again. A call into the VM (a hoisted
        // concatenation, a table store) has an effect, and runs once in
        // Cranelift's code for a loop side whose preheader sits before its
        // header: the `Join` of such a loop side is below its preheader.
        for (index, member) in members.iter().enumerate() {
            if index > 0
                && members.iter().any(|other| other.link_loop == Some(index))
                && passes_params(member.ir)
                && preheader_calls(member.ir)
            {
                return Err(Reject::new(
                    "loop side whose preheader calls into the VM under a side that rejoins its loop",
                ));
            }
        }
        for (index, member) in members.iter().enumerate() {
            let ir = member.ir;
            // A counted tree is counted as a whole (`compile.rs` refuses one
            // that mixes), and its budget is a second loop value beside the
            // fuel (see `Lowerer::budget`).
            if ir.counted != root.counted {
                return Err(Reject::new(
                    "counted trace with a member that is not counted",
                ));
            }
            if function_tree {
                // `check_tree` saw the members.
                continue;
            }
            if ir.function.is_some() {
                return Err(Reject::new("function trace in a loop tree"));
            }
            if index > 0 {
                let is_loop_side = members.iter().any(|other| other.terminal == Some(index));
                if is_loop_side {
                    // A loop side runs its own loop, entered by the exit of
                    // the member before it, and the nested sides of its
                    // guards may rejoin that loop.
                    if !matches!(ir.terminator, TraceTerminator::LoopBack { .. }) {
                        return Err(Reject::new("loop side with an unsupported end"));
                    }
                    // A parameter without a register (a register of the
                    // calls' frames a loop side in a call carries, see
                    // "Loops in calls" in `vm::jit::trace`, or one that
                    // starts from a constant) is carried like any other.
                    // Only a side that rejoins the loop names the
                    // parameters by register, and `link_loop` refuses that
                    // side, as Cranelift's `LinkLoop` does.
                } else if !ir.loop_params.is_empty() {
                    return Err(Reject::new("loop side"));
                }
                match ir.terminator {
                    TraceTerminator::Exit { .. } | TraceTerminator::Link { .. } => {}
                    TraceTerminator::LoopBack { .. } if is_loop_side => {}
                    TraceTerminator::LinkLoop { .. } => {
                        if member.link_loop.is_none() {
                            return Err(Reject::new("side that rejoins a loop side that is gone"));
                        }
                    }
                    _ => return Err(Reject::new("side with an unsupported end")),
                }
            } else {
                if member.link_loop.is_some() {
                    return Err(Reject::new("root that rejoins a loop side"));
                }
                if !matches!(
                    ir.terminator,
                    TraceTerminator::LoopBack { .. } | TraceTerminator::Exit { .. }
                ) {
                    return Err(Reject::new("trace with an unsupported end"));
                }
            }
            if let Some(side) = member.terminal {
                if side <= index {
                    return Err(Reject::new("side numbered before its parent"));
                }
                if !matches!(ir.terminator, TraceTerminator::Exit { .. }) {
                    return Err(Reject::new("loop side after a trace that does not exit"));
                }
            }
            for &(guard, side) in &member.attached {
                let _ = guard;
                if side <= index {
                    return Err(Reject::new("side numbered before its parent"));
                }
            }
        }
        // Which member enters each side, and which loop sides a nested side
        // rejoins (it must be one around the nested side).
        s.parent.clear();
        s.parent.resize(members.len(), usize::MAX);
        s.joined.clear();
        s.joined.resize(members.len(), false);
        for (index, member) in members.iter().enumerate() {
            for &(_, side) in &member.attached {
                s.parent[side] = index;
            }
            if let Some(side) = member.terminal {
                s.parent[side] = index;
            }
        }
        for (index, member) in members.iter().enumerate() {
            if let Some(target) = member.link_loop {
                let mut at = s.parent[index];
                while at != usize::MAX && at != target {
                    at = s.parent[at];
                }
                if at != target || !members.iter().any(|m| m.terminal == Some(target)) {
                    return Err(Reject::new(
                        "side that rejoins a loop that is not around it",
                    ));
                }
                s.joined[target] = true;
            }
        }
        f.clear();
        f.reserve_snaps(plan.descriptors.len());
        // Native code returns the index of the exit it left through. An
        // activation of a function trace returns `-1 - index`, the code
        // `reserve_snaps` gives, which the entry turns into the index.
        if !function_tree {
            for (index, snap) in f.snaps.iter_mut().enumerate() {
                snap.code = index as i64;
            }
        }
        s.filled.clear();
        s.filled.resize(plan.descriptors.len(), false);
        s.side_snaps.clear();
        s.side_snaps.resize(plan.descriptors.len(), u32::MAX);
        s.pending.clear();
        s.slow_snaps.clear();
        s.slow_snaps.resize(plan.descriptors.len(), u32::MAX);
        if members.len() > MAX_MEMBERS {
            return Err(Reject::new("tree with many sides"));
        }
        // The row layout depends on every member; a fixed array keeps the
        // compile from allocating.
        let mut all = [root; MAX_MEMBERS];
        for (slot, member) in all.iter_mut().zip(members) {
            *slot = member.ir;
        }
        let all = &all[..members.len()];
        let closure_words = closure_row_words(all);
        let uses_rows = members.iter().any(|member| {
            member
                .ir
                .preheader
                .iter()
                .chain(&member.ir.body)
                .any(|instruction| Self::reads_rows(&instruction.op, member.ir, closure_words))
        });
        let row_masked = rows_masked(all);
        let row_bytes = if uses_rows {
            let stride = row_stride_of(root.fields.len(), closure_words, row_masked);
            offset(stride * mem::size_of::<usize>())
        } else {
            0
        };
        // The root's loop runs in a version without the bounds checks its
        // range plan proves, when the checks it saves are worth lowering the
        // tree twice: the decision Cranelift makes (`versioning_pays`).
        let tree_size: usize = members
            .iter()
            .map(|member| member.ir.preheader.len() + member.ir.body.len())
            .sum();
        // A set of the root's body is proven with the other accesses or
        // costs the tree its plan (`RangeOptions::sets`); every other
        // operation that may resize an array does.
        // A side may resize an array: its link then goes to the root's outer
        // `Join`, below which the array words are read again and the plan is
        // checked against them, and never straight to the fast loop.
        let resizes = resizes_beside_sets(root);
        let range = (tier != OptimizationTier::Baseline
            && options.opt_flags.contains(TracingOptFlags::RANGE_PROOFS)
            && !function_tree
            // A link that stores an untyped register must come back above
            // the type guards, which the fast loop's `Join` is not.
            && !root
                .entry
                .inputs
                .iter()
                .any(|input| input.ty == TraceType::Any)
            && tree_size <= MAX_VERSIONED_INSTRUCTIONS)
            .then(|| {
                RangePlan::with(
                    root,
                    RangeOptions {
                        resizes,
                        sets: true,
                        tests_first: true,
                        open: true,
                    },
                )
            })
            .flatten()
            .filter(|plan| versioning_pays(plan, tree_size));
        if range.is_some() {
            // A range-proven loop whose preheader calls into the VM is one
            // this backend has not been shown to run like Cranelift's yet.
            if preheader_calls(root) {
                return Err(Reject::new(
                    "range-proven loop whose preheader calls into the VM",
                ));
            }
            // The sets of the body are proven: the fast loop stores them in
            // place, and the guarded loop reads the words again after each.
            if root.preheader.iter().chain(&root.body).any(|instruction| {
                Self::moves_words(&instruction.op)
                    && !matches!(instruction.op, TraceOp::ArraySet { .. })
            }) {
                return Err(Reject::new(
                    "range-proven loop in a tree with a call that moves array words",
                ));
            }
        }
        let range_planned = range.is_some();
        // Where each member's code goes. A link of a side of a versioned root
        // may go straight to the fast loop when it brings the next values of
        // a run the checks covered (`fast_link`): the side is lowered twice
        // then, once for the guards of the guarded version and the preheader,
        // which link to the root's `Join`, and once for the guards of the fast
        // loop, which link to the `Join` right below the checks.
        s.fast_region.clear();
        s.fast_region.resize(members.len(), u32::MAX);
        s.slow_region.clear();
        s.slow_region.resize(members.len(), u32::MAX);
        s.changing.clear();
        s.shadow_roots.clear();
        s.proven_guards.clear();
        s.proven_keys.clear();
        if let Some(plan) = range.as_ref() {
            let passed = pass_through_params(root);
            s.changing
                .extend((0..root.loop_params.len()).filter(|index| !passed.contains(index)));
            s.changing.extend(carried_for_links(
                root,
                members[1..].iter().map(|member| member.ir),
                plan,
                &passed,
            ));
            s.changing.sort_unstable();
            s.shadow_roots
                .extend(plan.inductions.iter().map(|induction| induction.root));
            s.proven_guards.extend(
                (0..root.guards.len()).filter(|guard| plan.guards_proven_key(root, *guard)),
            );
            s.proven_keys.extend(
                plan.keys
                    .iter()
                    .map(|proven| (proven.key.index(), plan.inductions[proven.induction].upward)),
            );
            for &(guard, side) in members[0].attached.iter().filter(|_| rejoin) {
                let side_ir = members[side].ir;
                let TraceTerminator::Link { snapshot } = side_ir.terminator else {
                    continue;
                };
                let entered = &root.snapshots[root.guards[guard].snapshot.index()];
                side_origins(root, side_ir, entered, &mut s.origins);
                if !side_ir.sets_tables()
                    && fast_link(
                        plan,
                        root,
                        side_ir,
                        &s.origins,
                        &s.changing,
                        &side_ir.snapshots[snapshot.index()].frames[0],
                    )
                {
                    s.fast_region[side] = 0;
                }
                if slow_link(
                    root,
                    side_ir,
                    &s.changing,
                    &side_ir.snapshots[snapshot.index()].frames[0],
                ) {
                    s.slow_region[side] = 0;
                }
            }
        }
        s.region_of.clear();
        s.region_of.resize(members.len(), 0);
        let mut next_region = u32::from(range_planned);
        for member in 1..members.len() {
            s.region_of[member] = next_region;
            next_region += 1;
            if s.fast_region[member] != u32::MAX {
                s.fast_region[member] = next_region;
                next_region += 1;
            }
            if s.slow_region[member] != u32::MAX {
                s.slow_region[member] = next_region;
                next_region += 1;
            }
        }
        if s.entry_args.len() < next_region as usize {
            s.entry_args.resize_with(next_region as usize, Vec::new);
        }
        for args in &mut s.entry_args {
            args.clear();
        }
        s.entered.clear();
        s.entered.resize(next_region as usize, false);
        s.const_def.clear();
        s.consts_seen = 0;
        let mut lowerer = Lowerer {
            members,
            plan,
            count_links: options.count_links,
            fused,
            shifts,
            f,
            s,
            row_bytes,
            closure_words,
            row_masked,
            member: 0,
            ir: root,
            inputs: NONE,
            control: NONE,
            fuel: NONE,
            budget: NONE,
            counted: root.counted,
            rows: NONE,
            at: usize::MAX,
            rejoin,
            lazy: false,
            string_constants: options
                .opt_flags
                .contains(crate::compat::TracingOptFlags::STRING_CONSTANTS),
            trusted: trusted_array_key(members),
            lane_probe: members.iter().all(|member| {
                member.ir.function.is_none() && !member.ir.makes_tables() && !member.ir.resumes()
            }) && !crate::jit::trace::compile::tree_bridges_virtuals(members),
            range,
            fast: None,
            versioned: false,
            folded: false,
            slow: false,
            fast_side: false,
            slow_side: false,
            key_miss: false,
            prefill_call: false,
            function: None,
        };
        if function_tree {
            lowerer.activation()?;
        } else {
            lowerer.root()?;
        }
        // The guarded loop is region 1, ahead of the sides: a region's code
        // is made after the regions it enters, and its guards enter sides.
        if lowerer.versioned {
            lowerer.guarded_loop()?;
        }
        for member in 1..members.len() {
            lowerer.side(member, SideCopy::Own)?;
            if lowerer.s.fast_region[member] != u32::MAX {
                lowerer.side(member, SideCopy::Fast)?;
            }
            if lowerer.s.slow_region[member] != u32::MAX {
                lowerer.side(member, SideCopy::Guarded)?;
            }
        }
        Ok(lowerer.versioned)
    }

    fn reject<T>(&self, what: &'static str) -> Res<T> {
        Err(Reject { what, at: self.at })
    }

    // ---- values ---------------------------------------------------------

    fn begin_member(&mut self, member: usize) -> Res<()> {
        self.member = member;
        self.ir = self.members[member].ir;
        let values = self.ir.values.len();
        self.s.vals.clear();
        self.s.vals.resize(values, NONE);
        self.s.cmps.clear();
        self.s.cmps.resize(values, NO_CMP);
        self.s.konst.clear();
        self.s.konst.resize(values, None);
        let bindings = self.members[0].ir.bindings.len();
        self.s.words.clear();
        self.s.words.resize(bindings, NO_WORDS);
        self.s.used.clear();
        self.s.used.resize(bindings, 0);
        self.s.slots.clear();
        self.s.rows.clear();
        self.s.lane_words.clear();
        self.s.keys.clear();
        self.forget_bytes();
        self.s.cells.clear();
        self.s.hand = None;
        self.s.handed = None;
        self.s.fields.clear();
        self.s.hres.clear();
        self.s.numtags.clear();
        self.s.made.clear();
        self.s.lazy_in.clear();
        self.s.lazy_in.resize(values, 0);
        if let Some(activation) = self.function.as_mut() {
            activation.deeper = NONE;
            activation.below = NONE;
        }
        self.rows = NONE;
        self.lazy = self
            .ir
            .preheader
            .iter()
            .chain(&self.ir.body)
            .any(|instruction| Self::moves_words(&instruction.op));
        self.scan()?;
        self.mark_cold();
        Ok(())
    }

    /// Marks the values of the member that only exits, sides and links name,
    /// directly or through other such values, and whose instruction is float
    /// arithmetic the allocator can compute in an exit stub. It is
    /// `defer_cold_values` of `emit.rs`: the loop skips them (their MIR
    /// instructions are cold, see `oximond_mir::ir::COLD`) and each exit that
    /// reads one computes it in its stub. The operands are made before them,
    /// so they are available there. The walk is `TraceIr::mark_cold_values`,
    /// with this backend's list of operators.
    fn mark_cold(&mut self) {
        let ir = self.ir;
        self.s.cold.clear();
        self.s.cold.resize(ir.values.len(), false);
        if ir.function.is_some() {
            return;
        }
        let mut hot = mem::take(&mut self.s.hot);
        ir.mark_cold_values(
            |op| {
                matches!(
                    op,
                    ArithmeticOp::Add
                        | ArithmeticOp::Subtract
                        | ArithmeticOp::Multiply
                        | ArithmeticOp::Divide
                        | ArithmeticOp::Modulo
                        | ArithmeticOp::ModuloWith(FloatModulo::Floor)
                        | ArithmeticOp::FloorDivide
                )
            },
            &mut self.s.cold,
            &mut hot,
        );
        self.s.hot = hot;
    }

    /// [`Self::mark_cold`] for the fast version of the root's loop, in
    /// `cold_fast`: there the slot of a proven key of an induction is
    /// addressed with the induction's integer copy (`proven_off`), and an
    /// overwrite goes to the slot its earlier access found, so neither reads
    /// the key. A key that is float arithmetic (`i - 1.0`) is then needed by
    /// the exits alone.
    ///
    /// The guard that holds the next value of an induction to its bound
    /// compares the integer copy too ([`Self::bound_limits`]), so it reads no
    /// float either. When nothing else in the loop reads the parameters of
    /// such an induction or its next value, the loop drops them: the header
    /// carries the integer copy alone and an exit makes the float it stores
    /// from the copy. Returns, per induction, whether the loop drops its
    /// floats. What a guard hands to a side lowered with the tree is read
    /// where the guard stands, so it counts as a use, apart from the floats
    /// of the induction a folded loop closes on at the guard it closes on.
    fn mark_cold_fast(&mut self, plan: &RangePlan, args: &[ValueId]) -> Vec<bool> {
        let ir = self.ir;
        // The guard a folded loop closes on stands in the block past the
        // loop's limit, where the floats of its induction are made from the
        // copy (see [`Self::close_folded`]), so what its side takes of them
        // is no use inside the loop.
        let folded = self
            .fold_candidate(plan, args)
            .map(|(index, _, _, guard)| (guard.index(), &plan.inductions[index]));
        let mut seed = Vec::new();
        for &(guard, _) in &self.members[0].attached {
            if let Some(info) = ir.guards.get(guard) {
                let made_at_exit = |value: ValueId| {
                    folded.is_some_and(|(closing, induction)| {
                        closing == guard
                            && (value == args[induction.root]
                                || ir.loop_params.iter().enumerate().any(|(index, param)| {
                                    param.value == value && rides(ir, induction, args, index)
                                }))
                    })
                };
                ir.snapshots[info.snapshot.index()].for_each_value(|value| {
                    if !made_at_exit(value) {
                        seed.push(value);
                    }
                });
            }
        }
        // An integer induction is dropped only where a range check holds
        // its start inside an array, so the copy is its value times 16.
        let mut dropped: Vec<bool> = plan
            .inductions
            .iter()
            .enumerate()
            .map(|(index, induction)| {
                induction.float || plan.checks.iter().any(|check| check.induction == index)
            })
            .collect();
        let mut hot = mem::take(&mut self.s.hot);
        loop {
            ir.mark_cold_values_seeded(
                |op| {
                    matches!(
                        op,
                        ArithmeticOp::Add
                            | ArithmeticOp::Subtract
                            | ArithmeticOp::Multiply
                            | ArithmeticOp::Divide
                            | ArithmeticOp::Modulo
                            | ArithmeticOp::ModuloWith(FloatModulo::Floor)
                            | ArithmeticOp::FloorDivide
                    )
                },
                |op| {
                    if let Some((_, held, _)) = bound_compare(ir, plan, args, op) {
                        return Some(held);
                    }
                    // The overflow test of a counted integer loop is not
                    // made (`overflow_proven`), so it reads no counter.
                    if let Some(counter) = overflow_counter(ir, plan, op) {
                        return Some(counter);
                    }
                    // The step of a dropped integer induction is taken on
                    // the copy (`next_of_dropped`), so the sum that makes
                    // its next value reads no parameter.
                    if let Some(parameter) = stepped_parameter(ir, plan, &dropped, args, op) {
                        return Some(parameter);
                    }
                    let (TraceOp::ArrayLoad {
                        binding,
                        key,
                        past_end: false,
                        ..
                    }
                    | TraceOp::NumberLoad { binding, key, .. }
                    | TraceOp::ArrayStore { binding, key, .. }
                    | TraceOp::ArrayOverwrite { binding, key, .. }
                    | TraceOp::ArraySet { binding, key, .. }) = *op
                    else {
                        return None;
                    };
                    plan.key(binding, key).map(|_| key)
                },
                |index| {
                    !plan
                        .inductions
                        .iter()
                        .zip(&dropped)
                        .any(|(induction, dropped)| *dropped && rides(ir, induction, args, index))
                },
                &seed,
                &mut self.s.cold_fast,
                &mut hot,
            );
            // An induction whose float something reads after all keeps it,
            // and the walk runs again: its next value is a use of others.
            let mut kept = false;
            for (induction, dropped) in plan.inductions.iter().zip(&mut dropped) {
                if !*dropped {
                    continue;
                }
                let next = args[induction.root];
                // A float's next value is cold when nothing reads it. An
                // integer's is never marked so, and is read when it is hot.
                let next_read = if induction.float {
                    !self.s.cold_fast[next.index()]
                } else {
                    hot[next.index()]
                };
                let read = next_read
                    || (0..ir.loop_params.len()).any(|index| {
                        rides(ir, induction, args, index)
                            && hot[ir.loop_params[index].value.index()]
                    });
                if read {
                    *dropped = false;
                    kept = true;
                }
            }
            if !kept {
                break;
            }
        }
        self.s.hot = hot;
        dropped
    }

    /// Checks that every operation of the member is one this backend lowers
    /// and notes which binding words and rows it reads.
    fn scan(&mut self) -> Res<()> {
        let ir = self.ir;
        for (at, instruction) in ir.preheader.iter().chain(&ir.body).enumerate() {
            self.at = at;
            match &instruction.op {
                TraceOp::ArrayLoad { binding, key, .. }
                | TraceOp::ArraySet { binding, key, .. }
                | TraceOp::NumberLoad { binding, key, .. }
                | TraceOp::ArrayStore { binding, key, .. }
                | TraceOp::ArrayOverwrite { binding, key, .. }
                | TraceOp::ArrayNext { binding, key, .. }
                | TraceOp::StringByte { binding, key, .. } => {
                    if matches!(instruction.op, TraceOp::ArraySet { .. })
                        && !self.members[0].ir.bindings[*binding as usize].is_array()
                    {
                        return self.reject("array store through another binding");
                    }
                    self.s.used[*binding as usize] |= USE_WORDS;
                    if ir.value_type(*key) == Some(TraceType::Float) {
                        self.s.used[*binding as usize] |= USE_FLOAT;
                    }
                    // A store in place and a read past the end look at the
                    // flags of the table's header.
                    let reads_header = matches!(
                        instruction.op,
                        TraceOp::ArrayLoad { past_end: true, .. } | TraceOp::ArraySet { .. }
                    );
                    if reads_header && let Some((header, _)) = self.header_binding(*binding) {
                        self.s.used[header] |= USE_WORDS;
                    }
                }
                TraceOp::FieldLoad { binding, .. }
                | TraceOp::FieldStore { binding, .. }
                | TraceOp::TableLength { binding, .. } => {
                    self.s.used[*binding as usize] |= USE_WORDS;
                }
                TraceOp::StringLength { binding, .. } => {
                    self.s.used[*binding as usize] |= USE_WORDS | USE_FLOAT;
                }
                TraceOp::HashLoad {
                    binding,
                    key,
                    expected,
                    ..
                } => {
                    // The probe reads the words of the binding, unless the
                    // parent's guard passes what its own probe found.
                    if self.probes_load(*key, *expected)
                        && !(self.member != 0
                            && self.takes_probe(self.member).is_some_and(|last| at <= last))
                    {
                        self.s.used[*binding as usize] |= USE_WORDS;
                    }
                }
                TraceOp::Constant(_)
                | TraceOp::Move { .. }
                | TraceOp::Negate { .. }
                | TraceOp::IntToFloat { .. }
                | TraceOp::ToWide { .. }
                | TraceOp::Numeric { .. }
                | TraceOp::NativeUnary { .. }
                | TraceOp::NativeBinary { .. }
                | TraceOp::Arithmetic { .. }
                | TraceOp::Compare { .. }
                | TraceOp::Select { .. }
                | TraceOp::Truthy { .. }
                | TraceOp::Not { .. }
                | TraceOp::GuardType { .. }
                | TraceOp::GuardBranch { .. }
                | TraceOp::BoolAnd { .. }
                | TraceOp::TableFieldLoad { .. }
                | TraceOp::TableFieldStore { .. }
                | TraceOp::LaneArrayLoad { .. }
                | TraceOp::LaneArrayStore { .. }
                | TraceOp::MetaGuard { .. }
                | TraceOp::PhantomTable { .. }
                | TraceOp::HostCall { .. }
                | TraceOp::ReleaseStrings { .. }
                | TraceOp::NumberTag { .. }
                | TraceOp::NumberModulo { .. }
                | TraceOp::HelperResult { .. } => {}
                op if function::function_op(op) && self.members[0].ir.function.is_some() => {}
                TraceOp::CallHelper { helper, args, .. } => {
                    if let Some(binding) = self.native_store_binding(*helper, args) {
                        self.s.used[binding as usize] |= USE_WORDS;
                    }
                    // A native next step reads the words of the array part.
                    if let Some(binding) = self.box_step_of(*helper, args).and_then(|s| s.array) {
                        self.s.used[binding as usize] |= USE_WORDS;
                    }
                    // A prototype check or cell access is done in line.
                    if Self::reads_rows(&instruction.op, ir, self.closure_words) {
                        continue;
                    }
                    let Some(info) = helper.info() else {
                        return self.reject("helper without a description");
                    };
                    if super::super::helpers::helper_address(*helper).is_none() {
                        return self.reject("helper nobody installed");
                    }
                    // A pattern helper has a twin that takes the prepared
                    // pattern, which this backend does not build.
                    if self.string_constants
                        && super::super::helpers::prepared_helper_address(*helper).is_some()
                    {
                        return self.reject("pattern helper with a prepared twin");
                    }
                    if info.params.len() + 4 > 30 {
                        return self.reject("helper with many parameters");
                    }
                }
                other => return self.reject(unsupported(other)),
            }
        }
        self.at = usize::MAX;
        Ok(())
    }

    fn ty(&self, value: ValueId) -> TraceType {
        self.ir
            .value_type(value)
            .expect("a verified trace types every operand")
    }

    fn mty(&self, ty: TraceType) -> Res<Ty> {
        match class(ty) {
            Some(ty) => Ok(ty),
            None => self.reject("borrowed string, userdata or untyped value"),
        }
    }

    fn iconst(&mut self, value: i64) -> Val {
        self.f.iconst(value)
    }

    fn scalar(&mut self, scalar: TraceScalar) -> Res<Val> {
        Ok(match scalar {
            TraceScalar::Nil => self.f.iconst(0),
            TraceScalar::Boolean(value) => self.f.iconst(i64::from(value)),
            TraceScalar::Integer(value) | TraceScalar::Int64(value) => self.f.iconst(value),
            TraceScalar::Float(value) => self.f.fconst(value),
            TraceScalar::Table(lane)
            | TraceScalar::Closure(lane)
            | TraceScalar::Thread(lane)
            | TraceScalar::String(lane)
            | TraceScalar::UInt64(lane) => self.f.iconst(lane.cast_signed()),
            _ => return self.reject("borrowed string or userdata constant"),
        })
    }

    /// The MIR value of `value`, making the one a pending comparison stands
    /// for if nothing has yet.
    fn get(&mut self, value: ValueId) -> Res<Val> {
        let index = value.index();
        let val = self.s.vals[index];
        if val != NONE {
            return Ok(val);
        }
        if let Some(val) = self.lazy_input(value)? {
            return Ok(val);
        }
        let cmp = self.s.cmps[index];
        if !cmp.live {
            return self.reject("undefined value");
        }
        let (one, zero) = (self.f.iconst(1), self.f.iconst(0));
        let made = self.f.select_any(cmp.cc, cmp.a, cmp.b, one, zero);
        self.s.vals[index] = made;
        Ok(made)
    }

    fn set(&mut self, result: Option<ValueId>, val: Val) {
        if let Some(result) = result {
            self.s.vals[result.index()] = val;
        }
    }

    /// The condition `value` holds: a pending comparison, or a 0/1 value.
    fn cond(&mut self, value: ValueId) -> Res<Cond> {
        let cmp = self.s.cmps[value.index()];
        if cmp.live && self.s.vals[value.index()] == NONE {
            return Ok(Cond::Cmp(cmp));
        }
        Ok(Cond::Val(self.get(value)?))
    }

    fn un(&mut self, op: Op, ty: Ty, a: Val) -> Val {
        let dst = self.f.val(ty);
        let mut inst = Inst::new(op, ty);
        inst.dst = dst;
        inst.a = a;
        if self.f.cold && is_coldable(op) {
            inst.scale = COLD;
        }
        self.f.push(inst);
        dst
    }

    /// Whether `result` is a value only exits name (see [`Self::mark_cold`]).
    fn is_cold(&self, result: Option<ValueId>) -> bool {
        let cold = if self.member == 0 && self.fast.as_ref().is_some_and(|fast| fast.active) {
            &self.s.cold_fast
        } else {
            &self.s.cold
        };
        result.is_some_and(|result| cold.get(result.index()).copied().unwrap_or(false))
    }

    fn load8(&mut self, base: Val, off: Off, disp: i64) -> Val {
        let dst = self.f.val(Ty::I64);
        let mut inst = Inst::new(Op::Load8, Ty::I64);
        inst.dst = dst;
        inst.a = base;
        self.address(&mut inst, off, disp);
        self.f.push(inst);
        dst
    }

    fn store8(&mut self, base: Val, off: Off, disp: i64, value: Val) {
        let mut inst = Inst::new(Op::Store8, Ty::I64);
        inst.a = base;
        inst.c = value;
        self.address(&mut inst, off, disp);
        self.f.push(inst);
    }

    /// Fills the address fields of `inst`, which has its base in `a`.
    fn address(&mut self, inst: &mut Inst, off: Off, disp: i64) {
        inst.scale = 1;
        match off {
            Off::K(k) if i32::try_from(k + disp).is_ok() => {
                inst.b = NONE;
                inst.imm = k + disp;
            }
            Off::K(k) => {
                inst.b = self.f.iconst(k);
                inst.imm = disp;
            }
            Off::V(v) => {
                inst.b = v;
                inst.imm = disp;
            }
            Off::VK(v, k) => {
                inst.b = v;
                inst.imm = k + disp;
            }
        }
    }

    fn load_at(&mut self, ty: Ty, base: Val, off: Off, disp: i64) -> Val {
        let dst = self.f.val(ty);
        let mut inst = Inst::new(Op::Load, ty);
        inst.dst = dst;
        inst.a = base;
        self.address(&mut inst, off, disp);
        self.f.push(inst);
        dst
    }

    /// `[base + off + disp] += value`.
    fn add_at(&mut self, base: Val, off: Off, disp: i64, value: Val) {
        let mut inst = Inst::new(Op::AddMem, Ty::I64);
        inst.a = base;
        inst.c = value;
        self.address(&mut inst, off, disp);
        self.f.push(inst);
    }

    fn store_at(&mut self, base: Val, off: Off, disp: i64, value: Val) {
        let ty = self.f.val_ty[value as usize];
        let mut inst = Inst::new(Op::Store, ty);
        inst.a = base;
        inst.c = value;
        self.address(&mut inst, off, disp);
        self.f.push(inst);
    }

    // ---- exits and guards ------------------------------------------------

    /// Fills the snapshot of exit `index`: the slots of its descriptor, and
    /// the fuel the run has left, which goes to the control block, with the
    /// budget the iteration began with in a counted tree.
    ///
    /// Returns the snapshot that is the exit: `index` itself, except in the
    /// guarded version of a versioned loop, whose values are not the fast
    /// version's, so it has snapshots of its own that return the same code.
    fn fill_exit(&mut self, index: usize) -> Res<u32> {
        if self.slow {
            if self.s.slow_snaps[index] != u32::MAX {
                return Ok(self.s.slow_snaps[index]);
            }
        } else if self.s.filled[index] {
            return Ok(index as u32);
        }
        self.s.buf.clear();
        self.exit_slots(index)?;
        // The fuel, a counted tree's budget, and an activation's depth left.
        self.exit_extras();
        if self.slow {
            let snap = self.f.snap_returning(&self.s.buf, index as i64);
            self.s.slow_snaps[index] = snap;
            return Ok(snap);
        }
        self.f.set_snap(index as u32, &self.s.buf);
        self.s.filled[index] = true;
        Ok(index as u32)
    }

    fn exit_index(&self, guard: GuardId) -> usize {
        self.plan.members[self.member].first + guard.index()
    }

    /// The exit a guard leaves through: the snapshot, and the side it enters
    /// instead when the guard has one. The first call for an exit fills the
    /// snapshot and keeps it, so code that guards inside a skip calls this
    /// before the skip opens: the values a snapshot reads must be the ones
    /// every later guard of the exit can see.
    #[allow(
        clippy::single_match_else,
        reason = "the two arms are the two ways a guard can leave"
    )]
    fn guard_target(&mut self, guard: GuardId) -> Res<(u32, u32)> {
        let attached = self.members[self.member]
            .attached
            .iter()
            .find(|(attached, _)| *attached == guard.index())
            .map(|(_, side)| *side);
        let index = self.exit_index(guard);
        match attached {
            Some(side) => {
                let mut snap = self.s.side_snaps[index];
                if snap == u32::MAX {
                    self.side_args(side, guard)?;
                    if self.takes_probe(side).is_some() {
                        // The last value: what the lookup this guard checks
                        // found (see `takes_probe`).
                        let Some(hand) = self.s.hand.take() else {
                            return self.reject("side of a probed lookup entered by another guard");
                        };
                        self.s.list.push(hand);
                    }
                    if !self.s.bridges.is_empty() {
                        self.bridge_args(index)?;
                    }
                    if self.s.convs.is_empty() {
                        self.s.buf.clear();
                        for &val in &self.s.list {
                            self.s.buf.push(SnapVal { val, slot: 0 });
                        }
                        snap = self.f.snap_returning(&self.s.buf, 0);
                    } else {
                        // The guard leaves through its own exit when a
                        // number has the other subtype, and converts the
                        // numbers only on its way to the side.
                        let pending = PendingSide {
                            exit: index,
                            list: self.s.list.clone(),
                            convs: self.s.convs.clone(),
                        };
                        self.s.pending.push(pending);
                        self.fill_exit(index)?;
                        snap = BY_SUBTYPE;
                    }
                    self.s.side_snaps[index] = snap;
                }
                let region = self.side_entered(side);
                if snap != BY_SUBTYPE {
                    self.note_entry(region, snap);
                }
                Ok((snap, region))
            }
            None => Ok((self.fill_exit(index)?, NONE)),
        }
    }

    /// A guard: continue when `a cc b`, otherwise leave through `guard`'s
    /// exit or enter the side that continues it.
    fn guard_cc(&mut self, cc: Cc, a: Val, b: Val, guard: GuardId) -> Res<()> {
        // Two integer constants that pass: a check on a value every entry
        // of a side passes as the same constant (see `note_entry`), which
        // Cranelift folds as well.
        if let (Some((Ty::I64, x)), Some((Ty::I64, y))) = (self.const_of(a), self.const_of(b))
            && int_holds(cc, x, y)
        {
            return Ok(());
        }
        let (snap, side) = self.guard_target(guard)?;
        let ty = self.f.val_ty[a as usize];
        if snap == BY_SUBTYPE {
            return self.guard_by_subtype(cc, (a, b, ty), side, guard);
        }
        let mut inst = Inst::new(Op::Guard, ty);
        inst.cc = cc;
        inst.a = a;
        inst.b = b;
        inst.c = snap;
        inst.d = side;
        self.f.push(inst);
        Ok(())
    }

    /// A guard whose side takes numbers of a subtype that is known at run
    /// time, as Cranelift's `leave` checks it: when `a cc b` fails, the side
    /// is entered if the numbers have the subtypes it was recorded for and the
    /// guard's own exit is taken if not. The conversions and the check are
    /// made in the skip the failure runs, not in the loop.
    fn guard_by_subtype(
        &mut self,
        cc: Cc,
        (a, b, ty): (Val, Val, Ty),
        side: u32,
        guard: GuardId,
    ) -> Res<()> {
        let index = self.exit_index(guard);
        let exit = self.fill_exit(index)?;
        let (zero, one) = (self.iconst(0), self.iconst(1));
        let at = self
            .s
            .pending
            .iter()
            .position(|held| held.exit == index)
            .expect("guard_target made the side's values");
        let open = self.f.begin_if(cc.negate(ty == Ty::F64), a, b);
        let mut list = self.s.pending[at].list.clone();
        let convs = self.s.pending[at].convs.clone();
        let holds = self.convert_numbers(&mut list, &convs);
        self.s.buf.clear();
        for &val in &list {
            self.s.buf.push(SnapVal { val, slot: 0 });
        }
        let snap = self.f.snap_returning(&self.s.buf, 0);
        self.note_entry(side, snap);
        // The first guard continues when the subtypes hold, the second
        // never does.
        let mut to_exit = Inst::new(Op::Guard, Ty::I64);
        to_exit.cc = Cc::Ne;
        to_exit.a = holds;
        to_exit.b = zero;
        to_exit.c = exit;
        self.f.push(to_exit);
        let mut to_side = Inst::new(Op::Guard, Ty::I64);
        to_side.cc = Cc::Eq;
        to_side.a = zero;
        to_side.b = one;
        to_side.c = snap;
        to_side.d = side;
        self.f.push(to_side);
        self.f.end_if(open, &[]);
        Ok(())
    }

    /// For a guard whose side takes tables the trace never made
    /// (`Scratch::bridges`): gives each such place of `self.s.list` a
    /// placeholder, and adds the state of exit `index` after the other
    /// values, one value per slot the exit's stub writes. The side stores
    /// them where the stub would and has the run build the tables
    /// (`build_entry_tables`), so the loop the guard stands in makes no call
    /// and keeps nothing for one.
    fn bridge_args(&mut self, index: usize) -> Res<()> {
        let zero = self.iconst(0);
        for at in 0..self.s.bridges.len() {
            let place = self.s.bridges[at].0;
            self.s.list[place] = zero;
        }
        let plan = self.plan;
        for ExitSlot { value, ty, .. } in &plan.descriptors[index].slots {
            if matches!(ty, TraceType::Nil | TraceType::Any) {
                continue;
            }
            let want = self.mty(*ty)?;
            let val = self.get(*value)?;
            if self.f.val_ty[val as usize] != want {
                return self.reject("exit slot of another type than its value");
            }
            self.s.list.push(val);
        }
        Ok(())
    }

    /// The exit a side that takes built tables starts from, and each such
    /// input with the place of its table among the tables the exit builds:
    /// `None` for any other side.
    fn entry_bridge(&self, member: usize) -> Option<(usize, Vec<(ValueId, usize)>)> {
        let snapshot = self.entered_snapshot(member)?;
        let parent = *self.s.parent.get(member)?;
        let (guard, _) = self.members[parent]
            .attached
            .iter()
            .find(|(_, side)| *side == member)?;
        let tables: Vec<(ValueId, usize)> = self.members[member]
            .ir
            .entry
            .inputs
            .iter()
            .filter_map(|input| {
                match snapshot.frames[0]
                    .registers
                    .get(usize::from(input.register))
                {
                    Some(RegisterEntry::Virtual(id)) => {
                        Some((input.value, snapshot.virtual_place(*id)?))
                    }
                    _ => None,
                }
            })
            .collect();
        (!tables.is_empty()).then(|| (self.plan.members[parent].first + guard, tables))
    }

    /// The start of a side that takes tables the trace never made: `state`
    /// is the state of exit `exit` as the guard handed it (`bridge_args`),
    /// one value per slot. The code writes it to the exit area as the exit's
    /// stub would and has the run build the exit's virtual tables from it
    /// (`TraceTables::materialize_exit`), which names their lanes in the
    /// result words. A run that cannot build them now has built nothing, and
    /// the side leaves through that exit, whose state is the one just
    /// written.
    fn build_entry_tables(
        &mut self,
        exit: usize,
        state: &[(Val, u32)],
        tables: &[(ValueId, usize)],
    ) {
        let area = self.f.exit_area();
        self.s.buf.clear();
        for &(val, slot) in state {
            self.store_at(
                area,
                Off::K(0),
                offset(slot as usize * mem::size_of::<i64>()),
                val,
            );
            self.s.buf.push(SnapVal { val, slot });
        }
        self.exit_extras();
        let leave = self
            .f
            .snap_returning(&self.s.buf, i64::try_from(exit).expect("exit index fits"));
        let dispatch = self.control_word(mem::offset_of!(TraceControl, tables));
        let number = self.iconst(i64::try_from(exit).expect("exit index fits"));
        let status = self.f.call(
            trace_materialize_exit as *const () as usize,
            &[dispatch, self.control, area, number],
            Some(Ty::I64),
        );
        let zero = self.iconst(0);
        let mut built = Inst::new(Op::Guard, Ty::I64);
        built.cc = Cc::Eq;
        built.a = status;
        built.b = zero;
        built.c = leave;
        built.d = NONE;
        self.f.push(built);
        let out_at = mem::offset_of!(TraceControl, helper_out);
        for &(value, place) in tables {
            self.s.vals[value.index()] = self.load_at(
                Ty::I64,
                self.control,
                Off::K(0),
                offset(out_at + place * mem::size_of::<i64>()),
            );
        }
    }

    /// Leaves through `guard` unless the condition holds.
    fn guard_cond(&mut self, cond: Cond, expected: bool, guard: GuardId) -> Res<()> {
        match cond {
            Cond::Cmp(cmp) => {
                let cc = if expected {
                    cmp.cc
                } else {
                    cmp.cc.negate(cmp.float)
                };
                self.guard_cc(cc, cmp.a, cmp.b, guard)
            }
            Cond::Val(val) => {
                let zero = self.iconst(0);
                let cc = if expected { Cc::Ne } else { Cc::Eq };
                self.guard_cc(cc, val, zero, guard)
            }
        }
    }

    /// The values a side's entry takes, in `self.s.list`: the side's entry
    /// inputs and frame inputs from the state of the guard's snapshot, then
    /// the fuel, the budget of a counted tree, the inputs pointer and the
    /// control pointer.
    fn side_args(&mut self, side: usize, guard: GuardId) -> Res<()> {
        let ir = self.ir;
        let snapshot = ir.guards[guard.index()].snapshot;
        self.side_args_at(side, snapshot)
    }

    /// As [`Self::side_args`], from the state of `snapshot`.
    fn side_args_at(&mut self, side: usize, snapshot: crate::jit_ir::trace::SnapshotId) -> Res<()> {
        let ir = self.ir;
        let side_ir = self.members[side].ir;
        let snapshot = &ir.snapshots[snapshot.index()];
        let mut list = mem::take(&mut self.s.list);
        list.clear();
        self.s.convs.clear();
        self.s.bridges.clear();
        let result = self.side_values(side_ir, snapshot, &mut list);
        if result.is_ok() {
            list.push(self.fuel);
            if self.counted {
                list.push(self.budget);
            }
            // An activation's side reads the inputs pointer from the
            // control block, and takes the depth left and the exit area.
            if let Some(activation) = self.function {
                list.push(self.control);
                list.push(activation.left);
                list.push(activation.area);
            } else {
                list.push(self.inputs);
                list.push(self.control);
            }
        }
        self.s.list = list;
        result
    }

    /// The float and the integer flag of a register that holds a float or an
    /// integer at run time (`side_inputs_at` in `emit.rs`): the register holds
    /// the number as a float, and `tag` says whether it is an integer (a 0 or
    /// 1).
    fn number_parts(
        &mut self,
        snapshot: &crate::jit_ir::trace::Snapshot,
        register: usize,
        tag: RegisterEntry,
    ) -> Res<(Val, Val)> {
        let float = match snapshot.frames[0].registers.get(register) {
            Some(RegisterEntry::Value(value)) => self.get(*value)?,
            Some(RegisterEntry::Const(scalar)) => self.scalar(*scalar)?,
            _ => return self.reject("snapshot entry a side cannot start from"),
        };
        let integer = match tag {
            RegisterEntry::Value(value) => self.get(value)?,
            RegisterEntry::Const(scalar) => self.scalar(scalar)?,
            _ => return self.reject("snapshot entry a side cannot start from"),
        };
        if self.f.val_ty[float as usize] != Ty::F64 || self.f.val_ty[integer as usize] != Ty::I64 {
            return self.reject("side input of another type");
        }
        Ok((float, integer))
    }

    /// The value each number of run-time subtype in `convs` takes in `list`
    /// (the float as it is, or its truncation for an integer side), and the
    /// value that is 1 when every number has the subtype its side takes.
    fn convert_numbers(&mut self, list: &mut [Val], convs: &[Conv]) -> Val {
        let mut all = NONE;
        for &Conv {
            at,
            ty,
            float,
            integer,
        } in convs
        {
            let (value, holds) = if ty == TraceType::Float {
                let one = self.iconst(1);
                (float, self.f.bin(Op::Xor, integer, one))
            } else {
                (self.un(Op::F2I, Ty::I64, float), integer)
            };
            list[at] = value;
            all = if all == NONE {
                holds
            } else {
                self.f.bin(Op::And, all, holds)
            };
        }
        all
    }

    fn side_values(
        &mut self,
        side_ir: &TraceIr,
        snapshot: &crate::jit_ir::trace::Snapshot,
        list: &mut Vec<Val>,
    ) -> Res<()> {
        let ir = self.ir;
        for input in &side_ir.entry.inputs {
            let register = usize::from(input.register);
            // An activation's side reads the root's cells and globals itself.
            if self
                .window_input(snapshot, input.register, input.ty)
                .is_some()
            {
                continue;
            }
            if let Some(tag) = snapshot.number_tag(register) {
                if !matches!(input.ty, TraceType::Float | TraceType::Integer) {
                    return self.reject("side input of another type");
                }
                let (float, integer) = self.number_parts(snapshot, register, tag)?;
                self.s.convs.push(Conv {
                    at: list.len(),
                    ty: input.ty,
                    float,
                    integer,
                });
                list.push(NONE);
                continue;
            }
            if input.ty == TraceType::Any {
                // A side reads a register of either type only as the root's
                // entry input left it: its guard reads the root's slot, and
                // the parameter is a placeholder.
                let passed_on = match snapshot.frames[0].registers.get(register) {
                    Some(RegisterEntry::Unchanged) => ir
                        .entry
                        .inputs
                        .iter()
                        .find(|held| held.register == input.register)
                        .is_none_or(|held| held.ty == TraceType::Any),
                    Some(RegisterEntry::Value(value)) => self.any_input_of(*value).is_some(),
                    _ => false,
                };
                if !passed_on || self.any_slot(input.register).is_none() {
                    return self.reject("untyped side input the root does not hold");
                }
                list.push(self.f.iconst(0));
                continue;
            }
            if let Some(RegisterEntry::Virtual(id)) = snapshot.frames[0].registers.get(register) {
                // A table the trace never made: the guard has the run build
                // it, and the side takes its lane.
                let place = snapshot
                    .virtual_place(*id)
                    .filter(|place| *place < crate::jit_ir::trace::MAX_HELPER_RESULTS);
                let (Some(place), true, true, None) = (
                    place,
                    input.ty == TraceType::Table,
                    snapshot.bridges_virtuals(),
                    self.function,
                ) else {
                    return self.reject("virtual table a side cannot start from");
                };
                self.s.bridges.push((list.len(), place));
                list.push(NONE);
                continue;
            }
            let want = self.mty(input.ty)?;
            let val = match snapshot.frames[0].registers.get(register) {
                Some(RegisterEntry::Value(value)) => {
                    // A borrow goes to a borrowed input and only there: an
                    // owned string is a lane id, and the two are not
                    // interchangeable.
                    if self.ty(*value).is_borrowed() != input.ty.is_borrowed() {
                        return self.reject("borrowed string passed to a side");
                    }
                    self.get(*value)?
                }
                Some(RegisterEntry::Const(scalar)) => {
                    if input.ty.is_borrowed() {
                        return self.reject("borrowed string passed to a side");
                    }
                    self.scalar(*scalar)?
                }
                // A string constant of the trace's roots is the lane id of
                // its slot (an owned string input, never a borrowed one).
                Some(RegisterEntry::Root(slot)) if !input.ty.is_borrowed() => {
                    self.iconst((ROOT_STRING_LANE_BASE + u64::from(*slot)).cast_signed())
                }
                Some(RegisterEntry::Unchanged) => {
                    match ir
                        .entry
                        .inputs
                        .iter()
                        .find(|held| usize::from(held.register) == register)
                    {
                        Some(held) if held.ty.is_borrowed() != input.ty.is_borrowed() => {
                            return self.reject("borrowed string passed to a side");
                        }
                        // The slot's type is not known here.
                        Some(held) if held.ty == TraceType::Any => {
                            return self.reject("untyped input passed to a typed side input");
                        }
                        Some(held) => self.get(held.value)?,
                        None if input.ty.is_borrowed() => {
                            return self.reject("borrowed string passed to a side");
                        }
                        None => self.root_input(input.register)?,
                    }
                }
                _ => return self.reject("snapshot entry a side cannot start from"),
            };
            if self.f.val_ty[val as usize] != want {
                return self.reject("side input of another type");
            }
            list.push(val);
        }
        for input in &side_ir.frame_inputs {
            let want = self.mty(input.ty)?;
            let val = match snapshot
                .frames
                .get(input.frame.index())
                .and_then(|frame| frame.registers.get(usize::from(input.register)))
            {
                Some(RegisterEntry::Value(value))
                    if ir.value_type(*value) == Some(input.ty) && !input.ty.is_borrowed() =>
                {
                    self.get(*value)?
                }
                Some(RegisterEntry::Const(scalar)) if scalar.ty() == input.ty => {
                    self.scalar(*scalar)?
                }
                _ => return self.reject("frame register a side cannot start from"),
            };
            if self.f.val_ty[val as usize] != want {
                return self.reject("side frame input of another type");
            }
            list.push(val);
        }
        Ok(())
    }

    /// The root's entry input for `register`: its value in the root, and a
    /// reload from the input slots in a side.
    fn root_input(&mut self, register: u16) -> Res<Val> {
        let root = self.members[0].ir;
        let Some((index, input)) = root
            .entry
            .inputs
            .iter()
            .enumerate()
            .find(|(_, input)| input.register == register)
        else {
            return self.reject("register the root does not read");
        };
        if self.member == 0 {
            return self.get(input.value);
        }
        self.load_input(index, input.ty)
    }

    /// The entry input in slot `index`, as a value of type `ty`.
    fn load_input(&mut self, index: usize, ty: TraceType) -> Res<Val> {
        let at = offset(
            index * mem::size_of::<TraceInputSlot>() + mem::offset_of!(TraceInputSlot, bits),
        );
        Ok(match ty {
            TraceType::Float => self.load_at(Ty::F64, self.inputs, Off::K(0), at),
            TraceType::Boolean => {
                let bits = self.load_at(Ty::I64, self.inputs, Off::K(0), at);
                let (one, zero) = (self.f.iconst(1), self.f.iconst(0));
                let zero2 = self.f.iconst(0);
                self.f.select(Cc::Ne, bits, zero2, one, zero)
            }
            TraceType::Nil => self.f.iconst(0),
            TraceType::Integer
            | TraceType::Int64
            | TraceType::UInt64
            | TraceType::Table
            | TraceType::Closure
            | TraceType::Thread
            | TraceType::String => self.load_at(Ty::I64, self.inputs, Off::K(0), at),
            _ => return self.reject("entry input of a type the backend does not carry"),
        })
    }

    // ---- regions ----------------------------------------------------------

    /// The words of binding `index`: the address and length, the length as a
    /// float where it is used as one.
    fn load_binding(&mut self, index: usize) -> Res<Words> {
        let root = self.members[0].ir;
        let first = root.entry.inputs.len();
        let slot_size = mem::size_of::<TraceInputSlot>();
        let tag_at = mem::offset_of!(TraceInputSlot, tag);
        let bits_at = mem::offset_of!(TraceInputSlot, bits);
        let binding = &root.bindings[index];
        let used = self.s.used[index];
        let base_at = offset((first + index) * slot_size + bits_at);
        let aux_at = offset((first + index) * slot_size + tag_at);
        let mut words = NO_WORDS;
        match &binding.kind {
            BindingKind::Array { .. } | BindingKind::Bytes { .. } => {
                words.base = self.load_at(Ty::I64, self.inputs, Off::K(0), base_at);
                words.len = self.load_at(Ty::I64, self.inputs, Off::K(0), aux_at);
                if used & USE_FLOAT != 0 {
                    words.lenf = self.un(Op::I2F, Ty::F64, words.len);
                }
            }
            BindingKind::Field { .. } => {
                words.base = self.load_at(Ty::I64, self.inputs, Off::K(0), base_at);
            }
            BindingKind::Header { .. } => {
                // The table's address and its flags (`NativeTableHeader::word`).
                words.base = self.load_at(Ty::I64, self.inputs, Off::K(0), base_at);
                words.len = self.load_at(Ty::I64, self.inputs, Off::K(0), aux_at);
            }
            BindingKind::Length { ty, .. } => {
                let ty = self.mty(*ty)?;
                words.base = self.load_at(ty, self.inputs, Off::K(0), base_at);
            }
            BindingKind::NoMetatable | BindingKind::Hash if self.mir_probes() => {
                // The string index of the table: the address of its slots
                // (zero for none) and its mask. A binding with no index gets a
                // table of empty buckets and a zero mask, so the probe reads
                // something in line and finds no answer there.
                let slots = self.load_at(Ty::I64, self.inputs, Off::K(0), base_at);
                let mask = self.load_at(Ty::I64, self.inputs, Off::K(0), aux_at);
                let zero = self.iconst(0);
                let empty = self.iconst(probe::no_index());
                words.base = self.f.select(Cc::Ne, slots, zero, slots, empty);
                words.len = self.f.select(Cc::Ne, slots, zero, mask, zero);
                // The flags above the mask go out at the top.
                let shift = self.iconst(i64::from(crate::runtime::NATIVE_PROBE_SLOT_SHIFT));
                words.lenf = self.f.bin(Op::Shl, words.len, shift);
            }
            BindingKind::Callee { .. }
            | BindingKind::StringField { .. }
            | BindingKind::MetaClass { .. }
            | BindingKind::NoMetatable
            | BindingKind::Hash => {
                // Entry checked the table and nothing reads it but a lookup,
                // which has no index to probe: Cranelift's words are zero and
                // its lookup is the VM's, which an index of no buckets here
                // gives.
                words.base = self.iconst(probe::no_index());
                words.len = self.iconst(0);
                words.lenf = self.iconst(0);
            }
        }
        Ok(words)
    }

    /// Loads the binding words and the rows pointer the member reads. A
    /// member that has a call moving them loads each where it first uses it.
    fn load_words(&mut self) -> Res<()> {
        let root = self.members[0].ir;
        for index in 0..root.bindings.len() {
            if self.s.used[index] == 0 {
                continue;
            }
            if self.lazy {
                // Loaded at the use (`words_of`).
                self.s.words[index] = NO_WORDS;
                continue;
            }
            let words = self.load_binding(index)?;
            self.s.words[index] = words;
        }
        let uses_rows = self
            .ir
            .preheader
            .iter()
            .chain(&self.ir.body)
            .any(|instruction| Self::reads_rows(&instruction.op, self.ir, self.closure_words));
        if uses_rows {
            self.rows = self.load_at(
                Ty::I64,
                self.control,
                Off::K(0),
                offset(mem::offset_of!(TraceControl, rows)),
            );
        }
        Ok(())
    }

    /// Loads the words of every binding the member reads and does not hold.
    fn load_missing_words(&mut self) -> Res<()> {
        for index in 0..self.members[0].ir.bindings.len() {
            if self.s.used[index] != 0 && self.s.words[index].base == NONE {
                let words = self.load_binding(index)?;
                self.s.words[index] = words;
            }
        }
        Ok(())
    }

    fn root(&mut self) -> Res<()> {
        self.begin_member(0)?;
        let ir = self.ir;
        let (inputs, control) = self.f.entry();
        self.inputs = inputs;
        self.control = control;
        // A mismatch leaves before anything runs, with the code `-1`; the
        // run finds which input it was.
        let mismatch = self.f.snap_returning(&[], -1);
        for (index, input) in ir.entry.inputs.iter().enumerate() {
            // An input of either type is checked by its type guard, in the
            // loop (see `guard_any_input`).
            if input.ty == TraceType::Any {
                continue;
            }
            let Some(required) = entry_tag(input.ty) else {
                return self.reject("entry input without a tag");
            };
            let tag = self.load_at(
                Ty::I64,
                inputs,
                Off::K(0),
                offset(
                    index * mem::size_of::<TraceInputSlot>() + mem::offset_of!(TraceInputSlot, tag),
                ),
            );
            let wanted = self.iconst(required);
            let mut inst = Inst::new(Op::Guard, Ty::I64);
            inst.cc = Cc::Eq;
            inst.a = tag;
            inst.b = wanted;
            inst.c = mismatch;
            self.f.push(inst);
        }
        let fuel0 = self.load_at(
            Ty::I64,
            self.control,
            Off::K(0),
            offset(mem::offset_of!(TraceControl, fuel)),
        );
        self.fuel = fuel0;
        if self.counted {
            // The budget at the loop header, which the VM left in the
            // control block (`load_entry_budget` of `emit.rs`).
            self.budget = self.load_at(
                Ty::I64,
                self.control,
                Off::K(0),
                offset(mem::offset_of!(TraceControl, budget)),
            );
        }
        self.enter_loop()
    }

    /// The code of the root after its entry checks, with `self.fuel`,
    /// `self.inputs` and `self.control` set: the entry loads, the preheader
    /// and the loop. The guarded version of a versioned loop continues from
    /// the state of the range checks in a region of its own.
    ///
    /// When sides rejoin the loop the root has a `Join` for them, where
    /// Cranelift has its outer header (see [`Self::join_root`]): before the
    /// preheader when the loop passes some parameter on unchanged, since
    /// Cranelift runs the preheader at its outer header then, and after it
    /// otherwise, since a link jumps to the loop header and the preheader
    /// runs once. A root that ends in front of an inner loop (in an exit a
    /// loop side continues) has its `Join` there too, above its body: a link
    /// of the side that ends the outer iteration runs the body again, as
    /// Cranelift's jump to that root's header does.
    fn enter_loop(&mut self) -> Res<()> {
        let ir = self.ir;
        let joins = self.rejoin;
        let at_preheader = passes_params(ir);
        for (index, input) in ir.entry.inputs.iter().enumerate() {
            if input.ty == TraceType::Any {
                // Only type guards read it (the verifier holds it to that),
                // and they read the slot.
                self.s.vals[input.value.index()] = self.f.iconst(0);
                continue;
            }
            let val = self.load_input(index, input.ty)?;
            self.s.vals[input.value.index()] = val;
        }
        self.load_words()?;
        if self.member == 0 && !self.slow {
            self.check_trusted_key()?;
        }
        // The preheader reads a parameter the loop changes, or one the loop
        // passes on unchanged, as its first value (Cranelift's outer header
        // holds the same value), which is the one its start gives, or the
        // one a link brought.
        self.s.inits.clear();
        match &ir.terminator {
            TraceTerminator::LoopBack { .. } if joins && at_preheader => self.join_root()?,
            TraceTerminator::LoopBack { .. } => {
                for param in &ir.loop_params {
                    let val = self.param_init(param.init)?;
                    self.s.vals[param.value.index()] = val;
                    self.s.inits.push(val);
                }
            }
            // A root that ends in front of an inner loop keeps the parameters
            // of its header, which Cranelift defines from their starts. With
            // sides that rejoin it, the `Join` below the preheader defines
            // them (`at_preheader` is false, as it passes nothing on); without,
            // a start this backend cannot make stays undefined, and only a
            // read of the parameter refuses the tree.
            TraceTerminator::Exit { .. } if !joins => {
                for param in &ir.loop_params {
                    if let Ok(val) = self.param_init(param.init) {
                        self.s.vals[param.value.index()] = val;
                    }
                }
            }
            _ => {}
        }
        for (at, instruction) in ir.preheader.iter().enumerate() {
            self.at = at;
            self.op(&instruction.op, instruction.result)?;
        }
        if joins && !at_preheader {
            self.at = usize::MAX;
            self.join_root()?;
        }
        let preheader = ir.preheader.len();
        let exits = &self.plan.members[0];
        let lazy = self.lazy;
        match &ir.terminator {
            TraceTerminator::LoopBack { args } => {
                if let Some(plan) = self.range.take() {
                    let dropped = self.mark_cold_fast(&plan, args);
                    let fast = self.range_checks(plan, dropped, args)?;
                    self.fast = Some(fast);
                    self.versioned = true;
                    // The fast loop moves no array (its sets are proven and
                    // stored in place), so it reads the binding words once,
                    // here, where the guarded loop reads them at each use.
                    if self.lazy {
                        self.lazy = false;
                        self.load_missing_words()?;
                    }
                    let joined = self.s.fast_region.iter().any(|region| *region != u32::MAX);
                    // The end of the arrays is made once, above the `Join`,
                    // unless a link may bring arrays of another length.
                    let below = joined && self.moves_words_somewhere();
                    if !below {
                        self.fold_range()?;
                    }
                    if joined {
                        self.join_fast()?;
                        if below {
                            self.fold_range()?;
                        }
                        self.fold_recheck(args)?;
                    }
                }
                let run = self.run_loop(args, preheader);
                self.fast = None;
                self.lazy = lazy;
                self.s.entry_now.clear();
                self.s.entry_cells.clear();
                run?;
            }
            TraceTerminator::Exit { snapshot } => {
                for (at, instruction) in ir.body.iter().enumerate() {
                    self.at = preheader + at;
                    self.op(&instruction.op, instruction.result)?;
                }
                self.at = usize::MAX;
                self.ends_in_exit(0, *snapshot, exits.terminal)?;
            }
            _ => return self.reject("trace with an unsupported end"),
        }
        Ok(())
    }

    /// The `Join` that the sides rejoining the root's loop jump to, at the
    /// point of the root where Cranelift's outer header is (see
    /// [`Self::enter_loop`]). It takes the start of each loop parameter and
    /// the fuel; a side's link hands it the new values, so the code below
    /// reads the parameters of the join, as Cranelift's reads the outer
    /// header's, and nothing above it reruns: neither the entry loads, nor
    /// the preheader when the join sits below it.
    ///
    /// A region cannot see what another defined, so what the code above the
    /// join made and the code below it reads goes through frame cells: stored
    /// where it is made, loaded again right after the join (a store whose
    /// cell nobody loads is dropped by the allocator, and so is a load nobody
    /// uses).
    fn join_root(&mut self) -> Res<()> {
        let ir = self.ir;
        let mut init = mem::take(&mut self.s.list);
        init.clear();
        let mut failed = None;
        for param in &ir.loop_params {
            match self.param_init(param.init) {
                Ok(val)
                    if self
                        .mty(param.ty)
                        .is_ok_and(|ty| ty == self.f.val_ty[val as usize]) =>
                {
                    init.push(val);
                }
                Ok(_) => {
                    failed = Some(self.reject::<()>("loop parameter of another type"));
                    break;
                }
                Err(error) => {
                    failed = Some(Err(error));
                    break;
                }
            }
        }
        if let Some(failed) = failed {
            self.s.list = init;
            return failed;
        }
        init.push(self.fuel);
        if self.counted {
            init.push(self.budget);
        }
        // A side may move or re-border an array before it links back
        // (`table.insert`, a store past the border), so the binding words the
        // root holds are not the ones the rejoin finds. Cranelift forgets the
        // array words at the header every link jumps to, and so does this:
        // they are not carried through the cells, and read again below the
        // `Join`, with what was read through them.
        let moved = self.forget_for_join();
        self.keep_in_cells();
        let first = self.f.join(&init);
        self.s.list = init;
        self.s.inits.clear();
        for (k, param) in ir.loop_params.iter().enumerate() {
            let val = first + k as Val;
            self.s.vals[param.value.index()] = val;
            self.s.inits.push(val);
        }
        self.fuel = first + ir.loop_params.len() as Val;
        if self.counted {
            self.budget = self.fuel + 1;
        }
        self.load_from_cells();
        if moved {
            // Eagerly for a root that moves nothing itself, at the use for
            // one that does (`words_of`).
            self.load_words()?;
        }
        Ok(())
    }

    /// The root's second `Join`, right below the range checks and above the
    /// fast loop: the links that stay in the fast loop jump here (see
    /// [`Self::link_fast`]), where Cranelift's jump to the fast header is. It
    /// takes the parameters the loop carries (`Scratch::changing`), the fuel
    /// and the integer copies of the inductions. What the checks and
    /// the code above them made and the loop reads goes through frame cells,
    /// as at the first `Join`, so a link runs neither the checks, nor the
    /// entry loads, nor the preheader.
    fn join_fast(&mut self) -> Res<()> {
        let ir = self.ir;
        let mut init = mem::take(&mut self.s.list);
        init.clear();
        for k in 0..self.s.changing.len() {
            let param = &ir.loop_params[self.s.changing[k]];
            let val = self.get(param.value)?;
            init.push(val);
        }
        init.push(self.fuel);
        if self.counted {
            init.push(self.budget);
        }
        if let Some(fast) = &self.fast {
            init.extend(fast.shadows.iter().map(|(_, copy)| *copy));
        }
        let moved = self.forget_for_join();
        self.keep_in_cells();
        // What a folded loop enters the guarded loop with stays in cells
        // below the `Join`, and is loaded where the loop leaves: loaded here
        // it would cost every link a load for each value. The cell is the
        // one the value was just kept in, or one of its own when nothing
        // below the `Join` reads the value (the first is the fuel, which the
        // loop passes itself).
        for k in 1..self.s.entry_now.len() {
            let val = self.s.entry_now[k];
            let known = self.s.kept.iter().find(|(kept, _)| *kept == val);
            let cell = if let Some((_, cell)) = known {
                *cell
            } else {
                let cell = self.f.new_cell();
                self.f.cell_store(cell, val);
                cell
            };
            self.s.entry_cells.push((cell, self.f.val_ty[val as usize]));
        }
        self.s.entry_now.clear();
        let first = self.f.join(&init);
        self.s.list = init;
        let carried = self.s.changing.len() as Val;
        for k in 0..self.s.changing.len() {
            let index = self.s.changing[k];
            let val = first + k as Val;
            self.s.vals[ir.loop_params[index].value.index()] = val;
            self.s.inits[index] = val;
        }
        self.fuel = first + carried;
        let after = self.after_fuel();
        if let Some(fast) = self.fast.as_mut() {
            for (at, (_, copy)) in fast.shadows.iter_mut().enumerate() {
                *copy = after + at as Val;
            }
        }
        self.load_from_cells();
        if moved {
            self.load_words()?;
        }
        Ok(())
    }

    /// Whether some member may move or re-border an array.
    fn moves_words_somewhere(&self) -> bool {
        self.members.iter().any(|member| {
            member.ir.sets_tables()
                || member
                    .ir
                    .preheader
                    .iter()
                    .chain(&member.ir.body)
                    .any(|instruction| Self::moves_words(&instruction.op))
        })
    }

    /// Whether some member may move or re-border an array before it links
    /// back (`table.insert`, a store past the border), so the binding words a
    /// `Join` finds are not the ones held above it. Cranelift forgets the
    /// array words at the header every link jumps to, and so does this: they
    /// are not carried through cells and are read again below the `Join`,
    /// with what was read through them. Forgets them, and the lane words, and
    /// says whether it did.
    fn forget_for_join(&mut self) -> bool {
        let moved = self.moves_words_somewhere();
        if moved {
            self.s.words.fill(NO_WORDS);
            self.s.slots.clear();
            self.s.fields.clear();
            self.s.rows.clear();
        }
        self.s.lane_words.clear();
        self.s.keys.clear();
        self.forget_bytes();
        self.s.cells.clear();
        moved
    }

    /// Calls `visit` on every MIR value the lowering holds: the pointers, the
    /// values of the trace's values, the pending comparisons, the binding
    /// words, the slot, row and field facts, the extra words of helper
    /// results, the number tags, the loop parameters' starts and, in the fast
    /// version, the steps and shifts of the range plan. `NONE` is visited as
    /// well; it is the caller's to skip. The one list of what the lowering
    /// holds: what a `Join` keeps in cells, and what the guarded loop starts
    /// from, are made of it.
    fn visit_held(&mut self, visit: &mut impl FnMut(&mut Val)) {
        visit(&mut self.inputs);
        visit(&mut self.control);
        visit(&mut self.rows);
        let s = &mut *self.s;
        for val in &mut s.vals {
            visit(val);
        }
        for cmp in s.cmps.iter_mut().filter(|cmp| cmp.live) {
            visit(&mut cmp.a);
            visit(&mut cmp.b);
        }
        for words in &mut s.words {
            visit(&mut words.base);
            visit(&mut words.len);
            visit(&mut words.lenf);
        }
        for fact in &mut s.slots {
            if let Off::V(val) | Off::VK(val, _) = &mut fact.off {
                visit(val);
            }
        }
        for (lane, row) in &mut s.rows {
            visit(lane);
            visit(row);
        }
        for (lane, _, slot) in &mut s.fields {
            visit(lane);
            visit(slot);
        }
        for (_, _, val) in &mut s.hres {
            visit(val);
        }
        for (_, val) in &mut s.numtags {
            visit(val);
        }
        for val in &mut s.inits {
            visit(val);
        }
        for val in &mut s.entry_now {
            visit(val);
        }
        if let Some(fast) = self.fast.as_mut() {
            for val in &mut fast.steps {
                visit(val);
            }
            for (_, val) in &mut fast.shifts {
                visit(val);
            }
            for limit in &mut fast.limits {
                visit(&mut limit.at_start);
                visit(&mut limit.at_next);
            }
            for (_, _, end) in &mut fast.open_ends {
                visit(end);
            }
            if let Some(fold) = fast.fold.as_mut() {
                visit(&mut fold.range_end);
            }
        }
    }

    /// Stores each value the lowering holds, apart from constants (which the
    /// allocator makes again where they are read), in a frame cell of its
    /// own, for the code after a `Join` to read.
    fn keep_in_cells(&mut self) {
        let count = self.f.val_ty.len();
        let mut remap = mem::take(&mut self.s.remap);
        remap.clear();
        remap.resize(count, NONE);
        for inst in &self.f.insts {
            if inst.op == Op::Const {
                remap[inst.dst as usize] = inst.dst;
            }
        }
        self.s.kept.clear();
        let mut held = mem::take(&mut self.s.held);
        held.clear();
        self.visit_held(&mut |val| held.push(*val));
        for &val in &held {
            if val != NONE && remap[val as usize] == NONE {
                let cell = self.f.new_cell();
                self.f.cell_store(cell, val);
                self.s.kept.push((val, cell));
                // Not loaded again yet: marks the value as kept.
                remap[val as usize] = NONE - 1;
            }
        }
        self.s.held = held;
        self.s.remap = remap;
    }

    /// Loads the values [`Self::keep_in_cells`] stored, right after the
    /// `Join`, and has everything the lowering holds name the loaded ones.
    fn load_from_cells(&mut self) {
        for k in 0..self.s.kept.len() {
            let (old, cell) = self.s.kept[k];
            let ty = self.f.val_ty[old as usize];
            let new = self.f.cell_load(ty, cell);
            self.s.remap[old as usize] = new;
        }
        let remap = mem::take(&mut self.s.remap);
        // Values made since the cells were stored (the join's parameters) are
        // not in the table, and stay.
        self.visit_held(&mut |val| {
            if *val != NONE && (*val as usize) < remap.len() && remap[*val as usize] != NONE {
                *val = remap[*val as usize];
            }
        });
        self.s.remap = remap;
    }

    /// The header, the body and the back edge of the loop of the current
    /// member. The loop parameters start from `self.s.inits`, `args` is what
    /// the loop-back passes, the preheader (of `preheader` instructions) is
    /// done.
    fn run_loop(&mut self, args: &[ValueId], preheader: usize) -> Res<()> {
        let ir = self.ir;
        let exits = &self.plan.members[self.member];
        // In the fast version every parameter of an induction holds its
        // root's value at every header (the range checks proved the starts
        // equal, and the loop-back gives them one value), so the header
        // carries the root alone and the others read it.
        let mut starts = mem::take(&mut self.s.list);
        starts.clear();
        for (k, (param, arg)) in ir.loop_params.iter().zip(args).enumerate() {
            if self.carried(k, param.value, *arg, args) {
                starts.push(self.s.inits[k]);
            }
        }
        let carried = starts.len() as Val;
        self.fold_end()?;
        starts.push(self.fuel);
        if self.counted {
            starts.push(self.budget);
        }
        // The fast version carries the integer copies of its float
        // inductions after the fuel (and the budget).
        if let Some(fast) = &self.fast {
            starts.extend(fast.shadows.iter().map(|(_, copy)| *copy));
        }
        let first = self.f.header(&starts);
        self.s.list = starts;
        self.fuel = first + carried;
        let after = self.after_fuel();
        if let Some(fast) = self.fast.as_mut() {
            for (at, (_, copy)) in fast.shadows.iter_mut().enumerate() {
                *copy = after + at as Val;
            }
            fast.active = true;
            fast.scaled.clear();
            fast.stepped.clear();
            fast.stepped.resize(fast.shadows.len(), NONE);
        }
        let mut k = 0;
        for (index, (param, arg)) in ir.loop_params.iter().zip(args).enumerate() {
            if let Some(root) = self.fast_twin(index, args) {
                // The root comes first in parameter order.
                self.s.vals[param.value.index()] = self.s.vals[ir.loop_params[root].value.index()];
            } else if let Some(copy) = self.dropped_copy(index) {
                // The parameter the loop does not carry, made from the copy
                // by an exit that stores it.
                let float = param.ty == TraceType::Float;
                self.s.vals[param.value.index()] = self.uncopied(copy, float);
            } else if param.value != *arg {
                self.s.vals[param.value.index()] = first + k;
                k += 1;
            }
        }
        // What the code before the loop reached may be moved or retyped by a
        // later iteration in a member that has a call that moves storage,
        // and the words it read are read again. Without one the slot facts
        // are still dropped, as Cranelift drops them.
        if self.lazy {
            self.forget_words();
        } else {
            self.s.slots.clear();
        }
        // A lane's array words are read again in each iteration, as
        // Cranelift does at its loop header.
        self.s.lane_words.clear();
        self.s.keys.clear();
        self.forget_bytes();
        self.s.cells.clear();
        for (at, instruction) in ir.body.iter().enumerate() {
            self.at = preheader + at;
            if self.next_of_dropped(instruction.result, args) {
                continue;
            }
            self.op(&instruction.op, instruction.result)?;
        }
        self.at = usize::MAX;
        let poll = exits
            .poll
            .ok_or(Reject::new("looping trace without a poll exit"))?;
        // Everything the back edge and the poll exit read is made first, so
        // no instruction comes between the subtraction of the fuel and the
        // guard that reads its flags.
        let mut next = mem::take(&mut self.s.list);
        next.clear();
        for (index, (param, arg)) in ir.loop_params.iter().zip(args).enumerate() {
            if self.carried(index, param.value, *arg, args) {
                next.push(self.get(*arg)?);
            }
        }
        for slot in &self.plan.descriptors[poll].slots {
            if !matches!(slot.ty, TraceType::Nil | TraceType::Any) {
                self.get(slot.value)?;
            }
        }
        let copies = self.stepped_copies();
        self.next_from_copies(args, &copies);
        if self.member == 0 && self.fast.as_ref().is_some_and(|fast| fast.fold.is_some()) {
            let closed = self.close_folded(args, poll, &mut next, &copies);
            self.s.list = next;
            return closed;
        }
        // One unit of fuel per iteration; the poll exit when it is gone.
        let one = self.iconst(1);
        let remaining = self.f.bin(Op::Sub, self.fuel, one);
        let zero = self.iconst(0);
        self.fuel = remaining;
        next.push(remaining);
        let snap = self.fill_exit(poll)?;
        let mut inst = Inst::new(Op::Guard, Ty::I64);
        inst.cc = Cc::Gt;
        inst.a = remaining;
        inst.b = zero;
        inst.c = snap;
        // The subtraction just before it set the flags.
        inst.scale = 1;
        self.f.push(inst);
        // A counted loop pays for the iteration it completed, what its poll
        // exit says ran (a loop side's costs nothing, as in `emit.rs`). The
        // poll exit above reports the budget the iteration began with. The
        // caller gave fuel for no more iterations than the budget covers, so
        // the next one needs no test.
        if self.counted {
            let cost = if self.member == 0 {
                ir.poll_snapshot
                    .map_or(0, |poll| ir.snapshots[poll.index()].counted)
            } else {
                0
            };
            let paid = self.pay(cost);
            next.push(paid);
        }
        next.extend(copies);
        self.f.back(&next);
        self.s.list = next;
        Ok(())
    }

    // ---- range-proven loop version ---------------------------------------

    /// At the back edge of the fast loop: the next value of each induction
    /// whose floats the loop does not carry, made from the stepped copy for
    /// the exits that read it (the poll exit), so the copy the iteration
    /// started with is dead once it is stepped and the step is one add in
    /// place. The stepped copy is the next value times 16, exactly, as the
    /// copy is the parameter's.
    fn next_from_copies(&mut self, args: &[ValueId], copies: &[Val]) {
        if self.member != 0 {
            return;
        }
        let Some(fast) = self.fast.as_ref() else {
            return;
        };
        let roots: Vec<(usize, usize)> = fast
            .shadows
            .iter()
            .enumerate()
            .filter(|(_, (induction, _))| fast.dropped[*induction])
            .map(|(at, (induction, _))| (at, fast.plan.inductions[*induction].root))
            .collect();
        for (at, root) in roots {
            let float = self.ir.loop_params[root].ty == TraceType::Float;
            let next = self.uncopied(copies[at], float);
            self.s.vals[args[root].index()] = next;
        }
    }

    /// The value of an induction whose copy (the value times 16) is `copy`:
    /// the copy converted and scaled back for a float, shifted back for an
    /// integer. Exact either way, as the copy is the value times 16 and far
    /// below 2^53. Cold: an exit that stores the value computes it.
    fn uncopied(&mut self, copy: Val, float: bool) -> Val {
        self.f.cold = true;
        let value = if float {
            let scaled = self.un(Op::I2F, Ty::F64, copy);
            let sixteenth = self.f.fconst(0.0625);
            self.f.bin(Op::FMul, scaled, sixteenth)
        } else {
            self.f.sar_cold(copy, 4)
        };
        self.f.cold = false;
        value
    }

    /// Whether `result` is the next value of an integer induction the fast
    /// loop being lowered does not carry. It is then not computed from the
    /// parameter, which the loop does not hold: it is the stepped copy
    /// shifted back, made by the exits that read it. The stepped copy is
    /// made here, once, and the loop-back takes the same value.
    fn next_of_dropped(&mut self, result: Option<ValueId>, args: &[ValueId]) -> bool {
        let Some(result) = result else {
            return false;
        };
        let Some(fast) = self
            .fast
            .as_ref()
            .filter(|fast| self.member == 0 && fast.active)
        else {
            return false;
        };
        let Some(at) = fast.shadows.iter().position(|(index, _)| {
            let induction = &fast.plan.inductions[*index];
            fast.dropped[*index] && !induction.float && args[induction.root] == result
        }) else {
            return false;
        };
        let stepped = self.stepped_copy(at);
        self.s.vals[result.index()] = self.uncopied(stepped, false);
        true
    }

    /// The root of the induction of loop parameter `index` when the fast
    /// version lowers the loop and the parameter is not the root: a
    /// parameter the range checks proved to start as the root does, and
    /// which the loop-back steps with it (`args` are the loop-back values).
    /// `None` outside the fast version.
    fn fast_twin(&self, index: usize, args: &[ValueId]) -> Option<usize> {
        let fast = self.fast.as_ref().filter(|_| self.member == 0)?;
        let params = &self.ir.loop_params;
        fast.plan
            .inductions
            .iter()
            .filter(|induction| index > induction.root && induction.params.contains(&index))
            .map(|induction| induction.root)
            .find(|&root| {
                args[index] == args[root]
                    && params[index].ty == params[root].ty
                    && params[index].value != args[index]
            })
    }

    /// Whether the header of the loop being lowered carries loop parameter
    /// `index` (`value`, which the loop-back replaces with `arg`): one the
    /// loop changes, unless the fast version reads it from its induction's
    /// root or makes it from the root's integer copy.
    fn carried(&self, index: usize, value: ValueId, arg: ValueId, args: &[ValueId]) -> bool {
        value != arg && self.fast_twin(index, args).is_none() && self.dropped_copy(index).is_none()
    }

    /// The integer copy (times 16) of the induction rooted at loop
    /// parameter `index`, when the fast version being lowered carries no
    /// float for it (see [`Self::mark_cold_fast`]).
    fn dropped_copy(&self, index: usize) -> Option<Val> {
        let fast = self.fast.as_ref().filter(|_| self.member == 0)?;
        let at = fast
            .plan
            .inductions
            .iter()
            .position(|induction| induction.root == index)?;
        if !fast.dropped[at] {
            return None;
        }
        fast.shadows
            .iter()
            .find(|(induction, _)| *induction == at)
            .map(|(_, copy)| *copy)
    }

    /// Whether the branch guard on `condition` is the overflow test of a
    /// counted integer loop, `not (MAX - step < r)`, on a parameter `r` of
    /// an upward induction the fast version being lowered proved. Inside
    /// the fast version `r` is at most the length of an array plus an offset
    /// and a shift of at most 2^31 each, below 2^60 (an array of 16-byte
    /// slots holds fewer than 2^59), and the step is at most 2^31, so the
    /// test cannot fail and the guard is not made.
    fn overflow_proven(&self, condition: ValueId, expected: bool) -> bool {
        if expected || self.member != 0 {
            return false;
        }
        let Some(fast) = self.fast.as_ref().filter(|fast| fast.active) else {
            return false;
        };
        defining_op(self.ir, condition)
            .is_some_and(|op| overflow_counter(self.ir, &fast.plan, op).is_some())
    }

    /// The checks that let the root's loop run in its fast version, and the
    /// state that version lowers with. They run once where control reaches
    /// the loop from outside (the entry, and every link of a side, which
    /// jumps to the root's `Join` and so through here again), in the order
    /// Cranelift checks them (`lower_versions`). Each is a guard that
    /// enters the guarded version, a region of its own (see
    /// [`Self::guarded_loop`]) that continues from everything held at the
    /// checks, so a failed check costs nothing but the checks. See [`crate::jit_ir::trace::range`]
    /// for what they prove.
    ///
    /// A float induction's keys are integral, so the fast version carries an
    /// integer copy of each checked float parameter and indexes with it, with
    /// no conversion: the copies start as the parameters' starts, converted
    /// here, and step by the step as an integer.
    #[allow(clippy::too_many_lines, reason = "one check after the other")]
    fn range_checks(
        &mut self,
        plan: RangePlan,
        dropped: Vec<bool>,
        args: &[ValueId],
    ) -> Res<FastLoop> {
        let ir = self.ir;
        let folding = self.fold_candidate(&plan, args);
        self.folded = folding.is_some();
        let entry = self.guard_entry();
        if self.folded {
            self.s.entry_now.clone_from(&self.s.entry_vals);
            // The fuel comes first, and the loop passes its own.
            self.s.entry_now[0] = NONE;
        }
        // A float as the integer it is, or the guarded version when it is
        // not integral: the truncating conversion and the conversion back
        // give the float again only when it is an integer inside an `i64`
        // (a NaN compares unordered, and a float outside converts to `MIN`,
        // which is `-2^63` back: equal only to that float itself, which the
        // callers' compares of the integer then refuse as they refuse every
        // large one).
        let integral = |this: &mut Self, number: Val| {
            let whole = this.un(Op::F2I, Ty::I64, number);
            let back = this.un(Op::I2F, Ty::F64, whole);
            this.prove(Cc::Eq, back, number, entry);
            whole
        };
        // `low <= value <= high` of an integer and two constants, as one
        // unsigned compare of `value - low`.
        let between = |this: &mut Self, value: Val, low: i64, high: i64| {
            let moved = if low == 0 {
                value
            } else {
                let offset = this.iconst(-low);
                this.f.bin(Op::Add, value, offset)
            };
            let span = this.iconst(high - low);
            this.prove(Cc::Ule, moved, span, entry);
        };
        // Whether `value` lies in `[low, high]` (ordered compares for a
        // float, so NaN does not), each end optional.
        let within = |this: &mut Self, value: Val, low: Option<Val>, high: Option<Val>| {
            if let Some(low) = low {
                this.prove(Cc::Ge, value, low, entry);
            }
            if let Some(high) = high {
                this.prove(Cc::Le, value, high, entry);
            }
        };
        let constant = |this: &mut Self, float: bool, number: i64| {
            if float {
                #[allow(clippy::cast_precision_loss, reason = "small integers")]
                this.f.fconst(number as f64)
            } else {
                this.iconst(number)
            }
        };
        let negate = |this: &mut Self, float: bool, value: Val| {
            if float {
                this.un(Op::FNeg, Ty::F64, value)
            } else {
                let zero = this.iconst(0);
                this.f.bin(Op::Sub, zero, value)
            }
        };
        // The shifts as the numbers they add, checked further down.
        let mut shifts: Vec<(RangeShift, Val, Val)> = Vec::new();
        for check in &plan.checks {
            let Some(shift) = check.shift else {
                continue;
            };
            if shifts.iter().any(|(known, _, _)| *known == shift) {
                continue;
            }
            let float = plan.inductions[check.induction].float;
            let mut value = self.get(shift.value)?;
            if shift.negated {
                value = negate(self, float, value);
            }
            shifts.push((shift, value, value));
        }
        // First the compares that a loop running off its array fails: the
        // root's start inside `[first, last]` of each array, and the bound on
        // the side the loop moves towards. A shift moves both ends the other
        // way. With a shift not yet checked the ends may be off, but then its
        // check below takes the guarded version anyway.
        for check in &plan.checks {
            let induction = &plan.inductions[check.induction];
            let float = induction.float;
            let words = self.words_of(check.binding)?;
            if words.len == NONE {
                return self.reject("range check on an array binding without a length");
            }
            let shift = check.shift.map(|shift| {
                shifts
                    .iter()
                    .find(|(known, _, _)| *known == shift)
                    .map(|(_, value, _)| *value)
                    .expect("every shift has its value")
            });
            // A folded loop stops where its arrays end and goes on in the
            // guarded loop, so its bound may lie past them.
            let folded = folding.is_some_and(|(folded, ..)| folded == check.induction);
            // The start of a float induction with no shift is compared as
            // the integer copy, where the copy is made below.
            let by_copy = float && shift.is_none();
            if folded && by_copy {
                continue;
            }
            let mut first = constant(self, float, 1 - check.low);
            let mut last = if float {
                let length = if words.lenf == NONE {
                    self.un(Op::I2F, Ty::F64, words.len)
                } else {
                    words.lenf
                };
                if check.high == 0 {
                    length
                } else {
                    let high = constant(self, true, -check.high);
                    self.f.bin(Op::FAdd, length, high)
                }
            } else {
                let high = self.iconst(-check.high);
                self.f.bin(Op::Add, words.len, high)
            };
            if let Some(shift) = shift {
                let op = if float { Op::FSub } else { Op::Sub };
                first = self.f.bin(op, first, shift);
                last = self.f.bin(op, last, shift);
            }
            let start = self.get(ir.loop_params[induction.root].value)?;
            // An open induction has no bound: its loads compare their keys
            // with the length (`Self::proven_off`).
            if let Some(bound) = induction.bound.filter(|_| !folded) {
                let bound = self.get(bound)?;
                if induction.upward {
                    within(self, bound, None, Some(last));
                } else {
                    within(self, bound, Some(first), None);
                }
            }
            if !by_copy {
                within(self, start, Some(first), Some(last));
            }
        }
        // Every parameter of an induction starts where its root does, so all
        // of them hold the root's value at every header of the run.
        for induction in &plan.inductions {
            let root = self.get(ir.loop_params[induction.root].value)?;
            for &param in &induction.params {
                if param == induction.root {
                    continue;
                }
                let start = self.get(ir.loop_params[param].value)?;
                self.prove(Cc::Eq, start, root, entry);
            }
        }
        // The steps: of the right sign, small and, for a float, integral.
        let mut steps = Vec::with_capacity(plan.inductions.len());
        for induction in &plan.inductions {
            let RangeStep::Invariant { value, negated } = induction.step else {
                steps.push(NONE);
                continue;
            };
            let float = induction.float;
            let positive = induction.upward != negated;
            let (low, high) = if positive {
                (1, MAX_RANGE_TERM)
            } else {
                (-MAX_RANGE_TERM, -1)
            };
            let step = self.get(value)?;
            let whole = if float {
                let whole = integral(self, step);
                between(self, whole, low, high);
                whole
            } else {
                let low = self.iconst(low);
                let high = self.iconst(high);
                within(self, step, Some(low), Some(high));
                step
            };
            let four = self.iconst(4);
            let integer = self.f.bin(Op::Shl, whole, four);
            steps.push(if negated {
                negate(self, false, integer)
            } else {
                integer
            });
        }
        // The shifts: small and, for a float, integral, like a step.
        for entry_shift in &mut shifts {
            let (shift, value, _) = *entry_shift;
            let float = plan
                .checks
                .iter()
                .find(|check| check.shift == Some(shift))
                .is_some_and(|check| plan.inductions[check.induction].float);
            let whole = if float {
                let whole = integral(self, value);
                between(self, whole, -MAX_RANGE_TERM, MAX_RANGE_TERM);
                whole
            } else {
                let low = self.iconst(-MAX_RANGE_TERM);
                let high = self.iconst(MAX_RANGE_TERM);
                within(self, value, Some(low), Some(high));
                value
            };
            let four = self.iconst(4);
            entry_shift.2 = self.f.bin(Op::Shl, whole, four);
        }
        // The integer copy of each induction, which inside the array is
        // exact: a start that is not integral takes the guarded version.
        let mut shadows = Vec::new();
        for (index, induction) in plan.inductions.iter().enumerate() {
            let start = self.get(ir.loop_params[induction.root].value)?;
            if !induction.float {
                // An integer start was held to its arrays above.
                let four = self.iconst(4);
                shadows.push((index, self.f.bin(Op::Shl, start, four)));
                continue;
            }
            let whole = integral(self, start);
            // The start inside `[first, last]` of each array it indexes with
            // no shift (the shifted ones were compared as floats above).
            for check in &plan.checks {
                if check.induction != index || check.shift.is_some() {
                    continue;
                }
                let words = self.words_of(check.binding)?;
                if words.len == NONE {
                    return self.reject("range check on an array binding without a length");
                }
                let first = self.iconst(1 - check.low);
                let last = if check.high == 0 {
                    words.len
                } else {
                    let high = self.iconst(-check.high);
                    self.f.bin(Op::Add, words.len, high)
                };
                within(self, whole, Some(first), Some(last));
            }
            let four = self.iconst(4);
            shadows.push((index, self.f.bin(Op::Shl, whole, four)));
        }
        // The end of each array a load keyed on an open induction reads, as
        // the first copy whose key is past it: `16 (length - offset + 1)`.
        let mut open_ends = Vec::new();
        for key in &plan.keys {
            if plan.inductions[key.induction].bound.is_some() {
                continue;
            }
            let words = self.words_of(key.binding)?;
            if words.len == NONE {
                return self.reject("range check on an array binding without a length");
            }
            let Some(less) = (key.offset - 1).checked_mul(16) else {
                return self.reject("a proven key whose offset is no displacement");
            };
            let four = self.iconst(4);
            let scaled = self.f.bin(Op::Shl, words.len, four);
            let end = if less == 0 {
                scaled
            } else {
                let less = self.iconst(less);
                self.f.bin(Op::Sub, scaled, less)
            };
            open_ends.push((key.binding, key.offset, end));
        }
        let limits = self.bound_limits(&plan, &steps, args, entry)?;
        let fold = folding.map(|(induction, condition, expected, guard)| Fold {
            induction,
            upward: plan.inductions[induction].upward,
            guard,
            condition,
            expected,
            range_end: NONE,
            end: NONE,
        });
        Ok(FastLoop {
            plan,
            shadows,
            limits,
            open_ends,
            dropped,
            fold,
            steps,
            shifts: shifts
                .into_iter()
                .map(|(shift, _, integer)| (shift, integer))
                .collect(),
            active: false,
            scaled: Vec::new(),
            stepped: Vec::new(),
        })
    }

    /// The guards of the root's body that hold the next value of a float
    /// induction to the induction's bound, as compares of the induction's
    /// integer copy. `steps` are the invariant steps as integers times 16.
    ///
    /// In the fast loop the next value `n` is an integer (the copy plus the
    /// step, exactly) and the bound `b` is a number the checks held to at
    /// most 2^40 in magnitude (a bound past that takes the guarded version,
    /// checked here), so each compare has an integer twin:
    /// `n <= b` is `n <= floor(b)`, `n < b` is `n <= ceil(b) - 1`,
    /// `b <= n` is `n >= ceil(b)`, `b < n` is `n >= floor(b) + 1`, and a
    /// guard that expects the compare to fail tests the other side. The
    /// guard tests the copy the iteration starts with, which the exits read:
    /// `16 n <= 16 L` is `copy <= 16 L - 16 step`.
    fn bound_limits(
        &mut self,
        plan: &RangePlan,
        steps: &[Val],
        args: &[ValueId],
        entry: (u32, u32),
    ) -> Res<Vec<Limit>> {
        let ir = self.ir;
        let mut limits: Vec<Limit> = Vec::new();
        let mut held = vec![false; plan.inductions.len()];
        for instruction in &ir.body {
            let TraceOp::GuardBranch {
                condition,
                expected,
                ..
            } = instruction.op
            else {
                continue;
            };
            let Some(compare) = defining_op(ir, condition) else {
                continue;
            };
            let Some((index, next, on_next)) = bound_compare(ir, plan, args, compare) else {
                continue;
            };
            if limits
                .iter()
                .any(|limit| limit.condition == condition && limit.expected == expected)
            {
                continue;
            }
            let TraceOp::Compare {
                comparison, left, ..
            } = compare
            else {
                continue;
            };
            let induction = &plan.inductions[index];
            let Some(bound) = induction.bound else {
                continue;
            };
            let bound = self.get(bound)?;
            if !held[index] {
                held[index] = true;
                if induction.float {
                    let low = self.f.fconst(-MAX_INTEGER_BOUND);
                    let high = self.f.fconst(MAX_INTEGER_BOUND);
                    self.prove(Cc::Ge, bound, low, entry);
                    self.prove(Cc::Le, bound, high, entry);
                } else {
                    #[allow(clippy::cast_possible_truncation, reason = "2^40")]
                    let most = MAX_INTEGER_BOUND as i64;
                    let low = self.iconst(-most);
                    let high = self.iconst(most);
                    self.prove(Cc::Ge, bound, low, entry);
                    self.prove(Cc::Le, bound, high, entry);
                }
            }
            // Whether passing the guard keeps the next value at or below
            // the limit, whether the limit rounds the bound down, and what
            // is added to the rounded bound. An integer bound is its own
            // rounding.
            let (below, floor, add) =
                limit_shape(*left == next, *comparison == Comparison::LessThan, expected);
            let whole = if induction.float {
                let rounded = self.un(if floor { Op::FFloor } else { Op::FCeil }, Ty::F64, bound);
                self.un(Op::F2I, Ty::I64, rounded)
            } else {
                bound
            };
            let step = match induction.step {
                RangeStep::Constant(step) => self.iconst(step * 16),
                RangeStep::Invariant { .. } => steps[index],
            };
            let adjusted = if add == 0 {
                whole
            } else {
                let adjust = self.iconst(add);
                self.f.bin(Op::Add, whole, adjust)
            };
            let four = self.iconst(4);
            let scaled = self.f.bin(Op::Shl, adjusted, four);
            // A guard on the value the iteration starts with compares the
            // copy as it is.
            let limit = if on_next {
                self.f.bin(Op::Sub, scaled, step)
            } else {
                scaled
            };
            limits.push(Limit {
                condition,
                expected,
                induction: index,
                cc: if below { Cc::Le } else { Cc::Ge },
                at_start: limit,
                at_next: scaled,
            });
        }
        Ok(limits)
    }

    /// The induction the fast loop can close on, with the guard that
    /// holds it to its bound: the first induction with exactly one
    /// such guard, which keeps the next value on the side of the bound the
    /// loop moves towards and after which the body only computes values.
    /// The guard then moves to the back edge unchanged: nothing between its
    /// place and the back edge can be observed.
    fn fold_candidate(
        &self,
        plan: &RangePlan,
        args: &[ValueId],
    ) -> Option<(usize, ValueId, bool, GuardId)> {
        let ir = self.ir;
        if self.counted || self.member != 0 {
            return None;
        }
        let mut found: Vec<(usize, usize, ValueId, bool, GuardId, bool)> = Vec::new();
        for (at, instruction) in ir.body.iter().enumerate() {
            let TraceOp::GuardBranch {
                condition,
                expected,
                guard,
            } = instruction.op
            else {
                continue;
            };
            let Some(compare) = defining_op(ir, condition) else {
                continue;
            };
            // A guard on the value the iteration starts with stays where it
            // is: its exit reads that iteration's values.
            let Some((index, next, true)) = bound_compare(ir, plan, args, compare) else {
                continue;
            };
            let TraceOp::Compare {
                comparison, left, ..
            } = compare
            else {
                continue;
            };
            let (below, ..) =
                limit_shape(*left == next, *comparison == Comparison::LessThan, expected);
            found.push((index, at, condition, expected, guard, below));
        }
        (0..plan.inductions.len()).find_map(|index| {
            let mut guards = found.iter().filter(|entry| entry.0 == index);
            let &(_, at, condition, expected, guard, below) = guards.next()?;
            if guards.next().is_some() || below != plan.inductions[index].upward {
                return None;
            }
            ir.body[at + 1..]
                .iter()
                .all(|instruction| computes_only(&instruction.op))
                .then_some((index, condition, expected, guard))
        })
    }

    /// Whether the guard on `condition` is the one the fast loop being
    /// lowered closes on (see [`Fold`]).
    fn folds(&self, condition: ValueId, expected: bool) -> bool {
        self.member == 0
            && self
                .fast
                .as_ref()
                .filter(|fast| fast.active)
                .and_then(|fast| fast.fold)
                .is_some_and(|fold| fold.condition == condition && fold.expected == expected)
    }

    /// The step of induction `index` of the fast loop as an integer times 16.
    fn step16(&mut self, index: usize) -> Val {
        let fast = self.fast.as_ref().expect("a fast loop is being lowered");
        match fast.plan.inductions[index].step {
            RangeStep::Constant(step) => self.iconst(step * 16),
            RangeStep::Invariant { .. } => fast.steps[index],
        }
    }

    /// The integer copy of induction `index` in the iteration of the
    /// fast loop being lowered, and its place among the copies.
    fn copy_of(&self, index: usize) -> (usize, Val) {
        let fast = self.fast.as_ref().expect("a fast loop is being lowered");
        fast.shadows
            .iter()
            .enumerate()
            .find(|(_, (induction, _))| *induction == index)
            .map(|(at, (_, copy))| (at, *copy))
            .expect("an induction has a copy")
    }

    /// For a folded fast loop: the last copy whose keys all lie inside their
    /// arrays, `16 (length - greatest offset) - shift` for each array of an
    /// upward induction and `16 (1 - least offset) - shift` for a downward
    /// one, the nearest of them. The range checks hold the start of a run to
    /// that range and the loop stops at its end, so every iteration starts
    /// inside it.
    fn fold_range(&mut self) -> Res<()> {
        let Some(fast) = self.fast.as_ref() else {
            return Ok(());
        };
        let Some(fold) = fast.fold else {
            return Ok(());
        };
        let upward = fold.upward;
        let checks: Vec<(u32, i64, Val)> = fast
            .plan
            .checks
            .iter()
            .filter(|check| check.induction == fold.induction)
            .map(|check| {
                let shift = check.shift.map_or(NONE, |shift| {
                    fast.shifts
                        .iter()
                        .find(|(known, _)| *known == shift)
                        .map(|(_, value)| *value)
                        .expect("every shift was checked")
                });
                let offset = if upward { check.high } else { check.low };
                (check.binding, offset, shift)
            })
            .collect();
        let mut end = NONE;
        for (binding, offset, shift) in checks {
            let words = self.words_of(binding)?;
            if words.len == NONE {
                return self.reject("range check on an array binding without a length");
            }
            let edge = if !upward {
                self.iconst(1 - offset)
            } else if offset == 0 {
                words.len
            } else {
                let high = self.iconst(-offset);
                self.f.bin(Op::Add, words.len, high)
            };
            let four = self.iconst(4);
            let mut scaled = self.f.bin(Op::Shl, edge, four);
            if shift != NONE {
                scaled = self.f.bin(Op::Sub, scaled, shift);
            }
            end = if end == NONE {
                scaled
            } else {
                self.nearer(upward, end, scaled)
            };
        }
        if end == NONE {
            return self.reject("folded loop without a range check");
        }
        if let Some(fold) = self.fast.as_mut().and_then(|fast| fast.fold.as_mut()) {
            fold.range_end = end;
        }
        Ok(())
    }

    /// For a folded fast loop, right below its `Join`: a link that comes
    /// here brings a copy the checks never saw (its side passed the bound
    /// guard, and the bound may lie past the arrays), so the copy is held to
    /// the range of the arrays, and a link that brings one outside goes on
    /// in the guarded loop. The block a link inside the range skips loads
    /// what that entry takes.
    fn fold_recheck(&mut self, args: &[ValueId]) -> Res<()> {
        let Some(fold) = self.fast.as_ref().and_then(|fast| fast.fold) else {
            return Ok(());
        };
        let (_, copy) = self.copy_of(fold.induction);
        let out = if fold.upward { Cc::Gt } else { Cc::Lt };
        let open = self.f.begin_if(out, copy, fold.range_end);
        let snap = self.transfer_snapshot(args, false, self.fuel)?;
        let (zero, one) = (self.iconst(0), self.iconst(1));
        let mut inst = Inst::new(Op::Guard, Ty::I64);
        inst.cc = Cc::Eq;
        inst.a = zero;
        inst.b = one;
        inst.c = snap;
        inst.d = 0;
        self.f.push(inst);
        self.f.end_if(open, &[]);
        Ok(())
    }

    /// The one of `a` and `b` an induction reaches first: the lesser for an
    /// upward one.
    fn nearer(&mut self, upward: bool, a: Val, b: Val) -> Val {
        let cc = if upward { Cc::Lt } else { Cc::Gt };
        self.f.select_any(cc, a, b, a, b)
    }

    /// For a folded fast loop, right above its header: the limit of the run
    /// that starts here. The next copy past it ends the run. It is the
    /// nearer of the bound guard's limit and the end of the arrays
    /// ([`Self::fold_range`]), and the fuel has to last that far.
    ///
    /// A run makes no more iterations than its shortest array has slots: it
    /// starts inside the proven range of every check (the range checks, or
    /// the recheck below the fast `Join`, hold it there) and moves at least
    /// one key an iteration towards `range_end`, so with a check of offsets
    /// `low..=high` on an array of `length` slots it makes at most
    /// `length - (high - low)` iterations. Fuel of at least `length`
    /// therefore outlasts the run, and the limit needs nothing of it: the
    /// last iteration finds the fuel at zero at worst, which the block past
    /// the limit tests after the bound guard, as the loop had them.
    ///
    /// With less fuel (a count hook's budget, a watch poll) the limit is
    /// also held to the copy the fuel reaches, in a block the usual entry
    /// skips: with fuel `f` the `f`th iteration is the last (its poll finds
    /// the fuel gone), and its next copy is `copy + f * step`, so the limit
    /// is one short of that. Fuel beyond a cap counts as the cap there, which
    /// keeps the product in range; a run that long goes on in the guarded
    /// loop.
    fn fold_end(&mut self) -> Res<()> {
        if self.member != 0 {
            return Ok(());
        }
        let Some(fold) = self.fast.as_ref().and_then(|fast| fast.fold) else {
            return Ok(());
        };
        let fast = self.fast.as_ref().expect("checked above");
        let Some(limit) = fast
            .limits
            .iter()
            .find(|limit| limit.condition == fold.condition && limit.expected == fold.expected)
            .map(|limit| limit.at_next)
        else {
            return self.reject("folded loop without its limit");
        };
        let Some(binding) = fast
            .plan
            .checks
            .iter()
            .find(|check| check.induction == fold.induction)
            .map(|check| check.binding)
        else {
            return self.reject("folded loop without a range check");
        };
        let cap = match fast.plan.inductions[fold.induction].step {
            RangeStep::Constant(step) => (1 << 60) / (16 * step.abs().max(1)),
            RangeStep::Invariant { .. } => FOLD_FUEL_CAP,
        };
        let length = self.words_of(binding)?.len;
        if length == NONE {
            return self.reject("range check on an array binding without a length");
        }
        let (_, copy) = self.copy_of(fold.induction);
        let step = self.step16(fold.induction);
        let end = self.nearer(fold.upward, limit, fold.range_end);
        let open = self.f.begin_if(Cc::Lt, self.fuel, length);
        let cap = self.iconst(cap);
        let fuel = self.f.select_any(Cc::Lt, self.fuel, cap, self.fuel, cap);
        let span = self.f.bin(Op::Mul, fuel, step);
        let reach = self.f.bin(Op::Add, copy, span);
        let short = self.iconst(if fold.upward { -1 } else { 1 });
        let reach = self.f.bin(Op::Add, reach, short);
        // The limit is the value the select keeps in place (its false arm),
        // so the entry that skips the block finds it where the join reads it.
        let nearer = if fold.upward { Cc::Lt } else { Cc::Gt };
        let held = self.f.select_any(nearer, reach, end, reach, end);
        let end = self.f.end_if(open, &[(held, end)]);
        if let Some(fold) = self.fast.as_mut().and_then(|fast| fast.fold.as_mut()) {
            fold.end = end;
        }
        Ok(())
    }

    /// The snapshot a folded fast loop enters the guarded loop with, where a
    /// failed range check enters it ([`Self::guard_entry`]): the fuel `fuel`,
    /// what the lowering held at the checks under the names it has here
    /// (`Scratch::entry_now`), and the parameters the loop changes
    /// (`Scratch::changing`). At the back edge (`at_back`) they are the
    /// loop-back's values, the start of the next iteration; right below the
    /// fast `Join` they are the join's.
    fn transfer_snapshot(&mut self, args: &[ValueId], at_back: bool, fuel: Val) -> Res<u32> {
        let ir = self.ir;
        let mut vals = Vec::with_capacity(self.s.entry_now.len() + self.s.entry_cells.len());
        vals.extend(self.s.entry_now.iter().skip(1));
        for k in 0..self.s.entry_cells.len() {
            let (cell, ty) = self.s.entry_cells[k];
            vals.push(self.f.cell_load(ty, cell));
        }
        self.s.buf.clear();
        self.s.buf.push(SnapVal { val: fuel, slot: 0 });
        for val in vals {
            self.s.buf.push(SnapVal { val, slot: 0 });
        }
        for k in 0..self.s.changing.len() {
            let index = self.s.changing[k];
            let val = if at_back {
                self.get(args[index])?
            } else {
                self.get(ir.loop_params[index].value)?
            };
            if self.mty(ir.loop_params[index].ty)? != self.f.val_ty[val as usize] {
                return self.reject("loop parameter of another type");
            }
            self.s.buf.push(SnapVal { val, slot: 0 });
        }
        Ok(self.f.snap_returning(&self.s.buf, 0))
    }

    /// The back edge of a folded fast loop (see [`Fold`]): the copies step,
    /// the fuel pays its unit, and one compare of the next copy with the
    /// run's limit ([`Self::fold_end`]) closes the loop. Past the limit, in a
    /// block the loop skips, the three things the limit stands for are told
    /// apart, in the order the loop had them:
    ///
    /// - the bound guard, on the next copy: the loop ended, and the guard
    ///   leaves through its exit or enters its side as it would have in its
    ///   place in the body;
    /// - the fuel: gone, so the poll exit;
    /// - neither: the arrays end before the bound (or the fuel is beyond
    ///   the cap), and the run goes on in the guarded loop with the next
    ///   values, as a link of a side does.
    ///
    /// The exits read the values the iteration started with, which the
    /// block makes again from the stepped ones (the copy less the step, the
    /// fuel plus one), so the loop steps both in place.
    fn close_folded(
        &mut self,
        args: &[ValueId],
        poll: usize,
        next: &mut Vec<Val>,
        copies: &[Val],
    ) -> Res<()> {
        let ir = self.ir;
        let fast = self.fast.as_ref().expect("a folded loop is a fast loop");
        let fold = fast.fold.expect("the caller checked");
        let Some(limit) = fast
            .limits
            .iter()
            .find(|limit| limit.condition == fold.condition && limit.expected == fold.expected)
            .copied()
        else {
            return self.reject("folded loop without its limit");
        };
        let dropped: Vec<usize> = (0..fast.plan.inductions.len())
            .filter(|index| fast.dropped[*index])
            .collect();
        let (at, _) = self.copy_of(fold.induction);
        let stepped = copies[at];
        let one = self.iconst(1);
        let zero = self.iconst(0);
        let remaining = self.f.bin(Op::Sub, self.fuel, one);
        // The exits of the block are filled inside it, from its values:
        // no guard outside the block leaves through them.
        let out = if fold.upward { Cc::Gt } else { Cc::Lt };
        let open = self.f.begin_if(out, stepped, fold.end);
        let mut saved: Vec<(usize, Val)> = Vec::new();
        for index in dropped {
            let (at, _) = self.copy_of(index);
            let step = self.step16(index);
            let started = self.f.bin(Op::Sub, copies[at], step);
            // Cold: an exit that stores one computes it, and a side that
            // takes one has it made here.
            let float = self.fast.as_ref().expect("checked above").plan.inductions[index].float;
            let old = self.uncopied(started, float);
            let new = self.uncopied(copies[at], float);
            let fast = self.fast.as_ref().expect("checked above");
            let induction = &fast.plan.inductions[index];
            for (param, info) in ir.loop_params.iter().enumerate() {
                if rides(ir, induction, args, param) {
                    saved.push((info.value.index(), self.s.vals[info.value.index()]));
                    self.s.vals[info.value.index()] = old;
                }
            }
            let stepped = args[induction.root].index();
            saved.push((stepped, self.s.vals[stepped]));
            self.s.vals[stepped] = new;
        }
        let before = self.f.bin(Op::Add, remaining, one);
        self.fuel = before;
        let condition = fold.condition.index();
        let failed = self.iconst(i64::from(!fold.expected));
        let passed = self.s.vals[condition];
        self.s.vals[condition] = failed;
        let guarded = self.guard_cc(limit.cc, stepped, limit.at_next, fold.guard);
        self.s.vals[condition] = passed;
        guarded?;
        self.fuel = remaining;
        let snap = self.fill_exit(poll)?;
        let mut inst = Inst::new(Op::Guard, Ty::I64);
        inst.cc = Cc::Gt;
        inst.a = remaining;
        inst.b = zero;
        inst.c = snap;
        self.f.push(inst);
        let snap = self.transfer_snapshot(args, true, remaining)?;
        let mut inst = Inst::new(Op::Guard, Ty::I64);
        inst.cc = Cc::Eq;
        inst.a = zero;
        inst.b = one;
        inst.c = snap;
        inst.d = 0;
        self.f.push(inst);
        self.f.end_if(open, &[]);
        for (index, val) in saved.into_iter().rev() {
            self.s.vals[index] = val;
        }
        next.push(remaining);
        next.extend_from_slice(copies);
        self.f.back(next);
        #[cfg(test)]
        FOLDED_LOOPS.with(|count| count.set(count.get() + 1));
        Ok(())
    }

    /// The guard on `condition` as a compare of the integer copy of a float
    /// induction, inside the fast loop (see [`Self::bound_limits`]).
    fn bound_limit(&self, condition: ValueId, expected: bool) -> Option<(Cc, Val, Val)> {
        if self.member != 0 {
            return None;
        }
        let fast = self.fast.as_ref().filter(|fast| fast.active)?;
        let limit = fast
            .limits
            .iter()
            .find(|limit| limit.condition == condition && limit.expected == expected)?;
        let copy = fast
            .shadows
            .iter()
            .find(|(induction, _)| *induction == limit.induction)
            .map(|(_, copy)| *copy)?;
        Some((limit.cc, copy, limit.at_start))
    }

    /// A proof of the range plan: continue when `a cc b`, otherwise enter the
    /// guarded version (`entry` is its snapshot and side).
    fn prove(&mut self, cc: Cc, a: Val, b: Val, entry: (u32, u32)) {
        let ty = self.f.val_ty[a as usize];
        let mut inst = Inst::new(Op::Guard, ty);
        inst.cc = cc;
        inst.a = a;
        inst.b = b;
        inst.c = entry.0;
        inst.d = entry.1;
        self.f.push(inst);
    }

    /// The values the integer copies of the fast version take at the
    /// loop-back: each induction's copy plus its step.
    fn stepped_copies(&mut self) -> Vec<Val> {
        let count = self.fast.as_ref().map_or(0, |fast| fast.shadows.len());
        (0..count).map(|at| self.stepped_copy(at)).collect()
    }

    /// The value copy `at` of the fast version takes at the loop-back, made
    /// once in an iteration.
    fn stepped_copy(&mut self, at: usize) -> Val {
        let fast = self.fast.as_ref().expect("a fast loop is being lowered");
        if let Some(&made) = fast.stepped.get(at)
            && made != NONE
        {
            return made;
        }
        let (induction, current) = fast.shadows[at];
        let (step, value) = (fast.plan.inductions[induction].step, fast.steps[induction]);
        let stepped = if value == NONE {
            let RangeStep::Constant(step) = step else {
                unreachable!("an invariant step has its integer");
            };
            let step = self.iconst(step * 16);
            self.f.bin(Op::Add, current, step)
        } else {
            self.f.bin(Op::Add, current, value)
        };
        if let Some(slot) = self.fast.as_mut().and_then(|fast| fast.stepped.get_mut(at)) {
            *slot = stepped;
        }
        stepped
    }

    /// The offset of the slot of `key` on `binding` when the fast version
    /// proved the key: the index times 16 (computed once per index), less
    /// 16 for the first slot, as a displacement of the access. A float key
    /// is the integer copy of its induction plus the shift and the key's
    /// offset. `None` for a key the plan does not prove, and outside the
    /// fast version's loop. A key of an open induction is proven to be an
    /// integral number of at least one: whether it is inside the array is
    /// compared here, and a key past the end leaves through `guard`.
    fn proven_off(&mut self, binding: u32, key: ValueId, guard: GuardId) -> Res<Option<Off>> {
        if self.member != 0 {
            return Ok(None);
        }
        let Some(fast) = self.fast.as_ref().filter(|fast| fast.active) else {
            return Ok(None);
        };
        let Some(proven) = fast.plan.key(binding, key).copied() else {
            return Ok(None);
        };
        let copy = fast
            .shadows
            .iter()
            .find(|(induction, _)| *induction == proven.induction)
            .map(|(_, value)| *value)
            .expect("an induction has a copy");
        let shift = proven.shift.map_or(NONE, |shift| {
            fast.shifts
                .iter()
                .find(|(known, _)| *known == shift)
                .map(|(_, value)| *value)
                .expect("every shift was checked")
        });
        let index_offset = proven.offset - 1;
        let Some(disp) = index_offset
            .checked_mul(16)
            .filter(|disp| i32::try_from(*disp).is_ok())
        else {
            // The plan counted this access as proven (a set as one that
            // moves no storage), so the checked form may not stand in.
            return self.reject("a proven key whose offset is no displacement");
        };
        let known = self
            .fast
            .as_ref()
            .and_then(|fast| {
                fast.scaled
                    .iter()
                    .find(|(held, by, _)| *held == copy && *by == shift)
            })
            .map(|(_, _, scaled)| *scaled);
        let scaled = if let Some(scaled) = known {
            scaled
        } else {
            // The copy of an induction and its shift are scaled already.
            let scaled = if shift == NONE {
                copy
            } else {
                self.f.bin(Op::Add, copy, shift)
            };
            if let Some(fast) = self.fast.as_mut() {
                fast.scaled.push((copy, shift, scaled));
            }
            scaled
        };
        let end = self.fast.as_ref().and_then(|fast| {
            fast.open_ends
                .iter()
                .find(|(held, offset, _)| *held == binding && *offset == proven.offset)
                .map(|(_, _, end)| *end)
        });
        if let Some(end) = end {
            self.guard_cc(Cc::Lt, scaled, end, guard)?;
        }
        Ok(Some(Off::VK(scaled, disp)))
    }

    /// The side (region minus one) of `member`, which a guard enters.
    fn side_of(&self, member: usize) -> u32 {
        self.s.region_of[member]
    }

    /// The side a guard of the code being lowered enters to run `member`: its
    /// copy for the fast loop when the guard is in the body of the fast loop
    /// and the member has one, its copy for the guarded loop when the guard
    /// is in the guarded loop and it has one, else the member's code.
    fn side_entered(&self, member: usize) -> u32 {
        let copy = if self.member != 0 || self.key_miss {
            u32::MAX
        } else if self.slow {
            self.s.slow_region[member]
        } else if self.fast.as_ref().is_some_and(|fast| fast.active) {
            self.s.fast_region[member]
        } else {
            u32::MAX
        };
        if copy == u32::MAX {
            self.side_of(member)
        } else {
            copy
        }
    }

    /// The snapshot a failed range check enters the guarded loop with: the
    /// fuel, the budget of a counted tree and every value the lowering holds (see [`Self::visit_held`]),
    /// which is all the guarded loop reads of what the entry and the
    /// preheader made, so it runs neither again. The tables are saved for
    /// [`Self::guarded_loop`], which starts from them in a region of its own.
    fn guard_entry(&mut self) -> (u32, u32) {
        let count = self.f.val_ty.len();
        let mut seen = mem::take(&mut self.s.remap);
        seen.clear();
        seen.resize(count, NONE);
        self.s.entry_consts.clear();
        for inst in &self.f.insts {
            if inst.op == Op::Const {
                seen[inst.dst as usize] = inst.dst;
                self.s.entry_consts.push((inst.dst, inst.ty, inst.imm));
            }
        }
        let mut held = mem::take(&mut self.s.held);
        held.clear();
        held.push(self.fuel);
        if self.counted {
            held.push(self.budget);
        }
        self.s.entry_now.clear();
        self.s.entry_cells.clear();
        self.visit_held(&mut |val| held.push(*val));
        self.s.entry_vals.clear();
        self.s.buf.clear();
        for &val in &held {
            if val != NONE && seen[val as usize] == NONE {
                seen[val as usize] = NONE - 1;
                self.s.entry_vals.push(val);
                self.s.buf.push(SnapVal { val, slot: 0 });
            }
        }
        // A folded fast loop enters the guarded loop with newer parameters
        // than these: each parameter the loop changes is an entry value of
        // its own then, after the held ones, apart from whatever else names
        // the value it starts from.
        if self.folded {
            for k in 0..self.s.changing.len() {
                let param = &self.ir.loop_params[self.s.changing[k]];
                let val = self.s.vals[param.value.index()];
                self.s.buf.push(SnapVal { val, slot: 0 });
            }
        }
        self.s.held = held;
        self.s.remap = seen;
        let saved = &mut self.s.saved;
        saved.inputs = self.inputs;
        saved.control = self.control;
        saved.rows = self.rows;
        saved.vals.clone_from(&self.s.vals);
        saved.cmps.clone_from(&self.s.cmps);
        saved.konst.clone_from(&self.s.konst);
        saved.words.clone_from(&self.s.words);
        saved.slots.clone_from(&self.s.slots);
        saved.rows_of.clone_from(&self.s.rows);
        saved.fields.clone_from(&self.s.fields);
        saved.hres.clone_from(&self.s.hres);
        saved.numtags.clone_from(&self.s.numtags);
        saved.inits.clone_from(&self.s.inits);
        (self.f.snap_returning(&self.s.buf, 0), 0)
    }

    /// The guarded version of the root's loop: region 1, which a failed range
    /// check enters with every value the lowering held at the checks
    /// ([`Self::guard_entry`]). It continues from them: no entry loads, no
    /// preheader, only the loop as it was before range proofs. When sides
    /// link straight back to it, a `Join` of its own sits before the loop
    /// (see [`Self::link_slow`]).
    fn guarded_loop(&mut self) -> Res<()> {
        self.member = 0;
        self.ir = self.members[0].ir;
        self.slow = true;
        // The snapshots the fast version made for its sides name its values.
        self.s.side_snaps.fill(u32::MAX);
        self.s.pending.clear();
        let slot = self.f.mark_side();
        debug_assert_eq!(slot, 0);
        // The entry values, then the constants made again here.
        let first = self.f.val_ty.len() as Val;
        let mut remap = mem::take(&mut self.s.remap);
        remap.clear();
        remap.resize(first as usize, NONE);
        for k in 0..self.s.entry_vals.len() {
            let old = self.s.entry_vals[k];
            let ty = self.f.val_ty[old as usize];
            remap[old as usize] = self.f.val(ty);
        }
        let params = if self.folded {
            self.s.changing.len()
        } else {
            0
        };
        let first_param = self.f.val_ty.len() as Val;
        for k in 0..params {
            let param = &self.ir.loop_params[self.s.changing[k]];
            let ty = self.f.val_ty[self.s.vals[param.value.index()] as usize];
            self.f.val(ty);
        }
        let mut entry = Inst::new(Op::SideEntry, Ty::I64);
        entry.a = first;
        entry.b = (self.s.entry_vals.len() + params) as u32;
        self.f.push(entry);
        for k in 0..self.s.entry_consts.len() {
            let (old, ty, imm) = self.s.entry_consts[k];
            remap[old as usize] = match ty {
                Ty::I64 => self.f.iconst(imm),
                Ty::F64 => self.f.fconst(f64::from_bits(imm.cast_unsigned())),
            };
        }
        // Back to the tables of the checks, naming this region's values.
        let saved = mem::take(&mut self.s.saved);
        self.inputs = saved.inputs;
        self.control = saved.control;
        self.rows = saved.rows;
        self.s.vals.clone_from(&saved.vals);
        self.s.cmps.clone_from(&saved.cmps);
        self.s.konst.clone_from(&saved.konst);
        self.s.words.clone_from(&saved.words);
        self.s.slots.clone_from(&saved.slots);
        self.s.rows.clone_from(&saved.rows_of);
        self.s.fields.clone_from(&saved.fields);
        self.s.hres.clone_from(&saved.hres);
        self.s.numtags.clone_from(&saved.numtags);
        self.s.inits.clone_from(&saved.inits);
        self.s.saved = saved;
        self.s.lane_words.clear();
        self.s.keys.clear();
        self.forget_bytes();
        self.s.cells.clear();
        self.visit_held(&mut |val| {
            if *val != NONE && remap[*val as usize] != NONE {
                *val = remap[*val as usize];
            }
        });
        // The fuel and the budget are the first entry values (`guard_entry`).
        self.fuel = first;
        if self.counted {
            self.budget = first + 1;
        }
        self.s.remap = remap;
        for k in 0..params {
            let index = self.s.changing[k];
            let val = first_param + k as Val;
            self.s.vals[self.ir.loop_params[index].value.index()] = val;
            self.s.inits[index] = val;
        }
        self.at = usize::MAX;
        // A folded fast loop that a side linked back to may come here after
        // the side moved an array: the words are read again below a `Join`.
        if self.s.slow_region.iter().any(|region| *region != u32::MAX)
            || (self.folded && self.moves_words_somewhere())
        {
            self.join_slow()?;
        }
        let ir = self.ir;
        let TraceTerminator::LoopBack { args } = &ir.terminator else {
            return self.reject("trace with an unsupported end");
        };
        let result = self.run_loop(args, ir.preheader.len());
        self.slow = false;
        result
    }

    /// The guarded loop's `Join`, right above its loop: the links that stay
    /// in the guarded version jump here (see [`Self::link_slow`]), where
    /// Cranelift's jump to the slow header is. It takes the parameters the
    /// loop carries (`Scratch::changing`) and the fuel; what is held above it
    /// and the loop reads goes through frame cells, as at the other joins.
    fn join_slow(&mut self) -> Res<()> {
        let ir = self.ir;
        let mut init = mem::take(&mut self.s.list);
        init.clear();
        for k in 0..self.s.changing.len() {
            let param = &ir.loop_params[self.s.changing[k]];
            let val = self.get(param.value)?;
            init.push(val);
        }
        init.push(self.fuel);
        if self.counted {
            init.push(self.budget);
        }
        let moved = self.forget_for_join();
        self.keep_in_cells();
        let first = self.f.join(&init);
        self.s.list = init;
        for k in 0..self.s.changing.len() {
            let index = self.s.changing[k];
            let val = first + k as Val;
            self.s.vals[ir.loop_params[index].value.index()] = val;
            self.s.inits[index] = val;
        }
        self.fuel = first + self.s.changing.len() as Val;
        if self.counted {
            self.budget = self.fuel + 1;
        }
        self.load_from_cells();
        if moved {
            self.load_words()?;
        }
        Ok(())
    }

    /// The end of a member that ends in an exit: the terminal exit, or, when
    /// a loop side continues it, an entry into that side.
    fn ends_in_exit(
        &mut self,
        member: usize,
        snapshot: crate::jit_ir::trace::SnapshotId,
        terminal: Option<usize>,
    ) -> Res<()> {
        if let Some(side) = self.members[member].terminal {
            // The loop the trace ends in front of runs in place, entered with
            // the state of the exit.
            self.side_args_at(side, snapshot)?;
            if !self.s.bridges.is_empty() {
                return self.reject("virtual table a loop after the trace starts from");
            }
            let convs = mem::take(&mut self.s.convs);
            let mut list = mem::take(&mut self.s.list);
            let holds = self.convert_numbers(&mut list, &convs);
            self.s.list = list;
            self.s.convs = convs;
            if holds != NONE {
                // A number of the other subtype than the loop was recorded
                // for leaves through the exit.
                let terminal = terminal.ok_or(Reject::new("trace without a terminal exit"))?;
                let exit = self.fill_exit(terminal)?;
                let zero = self.iconst(0);
                let mut inst = Inst::new(Op::Guard, Ty::I64);
                inst.cc = Cc::Ne;
                inst.a = holds;
                inst.b = zero;
                inst.c = exit;
                self.f.push(inst);
            }
            self.s.buf.clear();
            for &val in &self.s.list {
                self.s.buf.push(SnapVal { val, slot: 0 });
            }
            let snap = self.f.snap_returning(&self.s.buf, 0);
            self.note_entry(self.side_of(side), snap);
            let mut inst = Inst::new(Op::Goto, Ty::I64);
            inst.c = snap;
            inst.d = self.side_of(side);
            self.f.push(inst);
            return Ok(());
        }
        let terminal = terminal.ok_or(Reject::new("trace without a terminal exit"))?;
        self.exit(terminal)
    }

    /// Notes what an entry with the values of snapshot `snap` passes to side
    /// region `region`: at each place, an earlier place's value again, a
    /// constant, or another value. Each further entry keeps only what it
    /// agrees with. The side makes the constants itself and reads an earlier
    /// parameter for a repeated value (see [`Self::side_code`]), so the
    /// parameters it would have taken are dead and the jump moves nothing
    /// into them. Cranelift lowers a side inside its parent, where such a
    /// value costs nothing; here each would be a move per entry, and a store
    /// when the side keeps it only for its exits.
    fn note_entry(&mut self, region: u32, snap: u32) {
        self.see_consts();
        let s = &mut *self.s;
        let region = region as usize;
        let first = !s.entered[region];
        s.entered[region] = true;
        let mut args = mem::take(&mut s.entry_args[region]);
        if first {
            args.clear();
        }
        let list = self.f.snap_of(snap);
        for (k, held) in list.iter().enumerate() {
            let val = held.val;
            let now = if let Some(earlier) = list[..k].iter().position(|e| e.val == val) {
                EntryArg::Same(earlier as u32)
            } else if s.const_def[val as usize] != NONE {
                let inst = &self.f.insts[s.const_def[val as usize] as usize];
                EntryArg::Const(inst.ty, inst.imm)
            } else {
                EntryArg::Other
            };
            if first {
                args.push(now);
            } else if args.get(k) != Some(&now)
                && let Some(arg) = args.get_mut(k)
            {
                *arg = EntryArg::Other;
            }
        }
        s.entry_args[region] = args;
    }

    /// Brings `Scratch::const_def` up to the instructions made so far.
    fn see_consts(&mut self) {
        let s = &mut *self.s;
        s.const_def.resize(self.f.val_ty.len(), NONE);
        for (at, inst) in self.f.insts.iter().enumerate().skip(s.consts_seen) {
            if inst.op == Op::Const {
                s.const_def[inst.dst as usize] = at as Val;
            }
        }
        s.consts_seen = self.f.insts.len();
    }

    /// The type and bits of `val` when a constant instruction makes it.
    fn const_of(&mut self, val: Val) -> Option<(Ty, i64)> {
        if val == NONE {
            return None;
        }
        self.see_consts();
        let at = *self.s.const_def.get(val as usize)?;
        (at != NONE).then(|| {
            let inst = &self.f.insts[at as usize];
            (inst.ty, inst.imm)
        })
    }

    fn param_init(&mut self, init: LoopParamInit) -> Res<Val> {
        match init {
            LoopParamInit::Input(value) => self.get(value),
            LoopParamInit::Const(scalar) => self.scalar(scalar),
        }
    }

    /// Ends the region by leaving through exit `index`.
    fn exit(&mut self, index: usize) -> Res<()> {
        let snap = self.fill_exit(index)?;
        let mut inst = Inst::new(Op::Exit, Ty::I64);
        inst.c = snap;
        self.f.push(inst);
        Ok(())
    }

    /// A side that rejoins the loop (`TraceTerminator::Link`): the state it
    /// ends in names one value per loop parameter of the root. They go to the
    /// root's `Join`, with one unit of fuel spent (the poll exit when it is
    /// gone).
    fn link(&mut self, snapshot: crate::jit_ir::trace::SnapshotId) -> Res<()> {
        let ir = self.ir;
        let root = self.members[0].ir;
        let state = &ir.snapshots[snapshot.index()].frames[0];
        let mut next = mem::take(&mut self.s.list);
        next.clear();
        for param in &root.loop_params {
            let Some(register) = param.register else {
                self.s.list = next;
                return self.reject("loop parameter without a register");
            };
            let val = match state.registers.get(usize::from(register)) {
                Some(RegisterEntry::Value(value)) => {
                    if self.ty(*value).is_borrowed() {
                        self.s.list = next;
                        return self.reject("borrowed string passed to the loop");
                    }
                    self.get(*value)
                }
                Some(RegisterEntry::Const(scalar)) => self.scalar(*scalar),
                _ => {
                    self.s.list = next;
                    return self.reject("loop parameter a side cannot link");
                }
            };
            match val {
                Ok(val) => next.push(val),
                Err(error) => {
                    self.s.list = next;
                    return Err(error);
                }
            }
        }
        // A register the root takes untyped arrives through its slot, which
        // the root's type guard reads again below the `Join` (a tree with
        // such a register has no versioned loop, so no link goes past it).
        if let Err(error) = self.store_untyped_link(state) {
            self.s.list = next;
            return Err(error);
        }
        let result = if self.fast_side {
            self.link_fast(&next)
        } else if self.slow_side {
            self.link_slow(&next)
        } else {
            self.link_back(&mut next)
        };
        self.s.list = next;
        result
    }

    /// The register of the entry input of either type that `value` is, in
    /// the member being lowered.
    fn any_input_of(&self, value: ValueId) -> Option<u16> {
        if self.ir.value_type(value) != Some(TraceType::Any) {
            return None;
        }
        match self.ir.values.get(value.index()).map(|info| info.def) {
            Some(ValueDef::Input(index)) => {
                self.ir.entry.inputs.get(index).map(|input| input.register)
            }
            _ => None,
        }
    }

    /// The root's input slot of `register`, if the root takes it untyped.
    fn any_slot(&self, register: u16) -> Option<usize> {
        self.members[0]
            .ir
            .entry
            .inputs
            .iter()
            .position(|input| input.register == register && input.ty == TraceType::Any)
    }

    /// The guard that leaves through `guard` (or enters its side) unless the
    /// root's entry input of either type that `source` stands for has type
    /// `expected`, and the input as a value of that type
    /// (`guard_any_input` of `emit.rs`). The tag and the bits come from the
    /// input's slot: it holds the run's entry state, or what the last link
    /// stored (see [`Self::store_untyped_link`]).
    fn guard_any_input(
        &mut self,
        source: ValueId,
        expected: TraceType,
        guard: GuardId,
    ) -> Res<Val> {
        let Some(slot) = self
            .any_input_of(source)
            .and_then(|register| self.any_slot(register))
        else {
            return self.reject("guard on an untyped value");
        };
        if !matches!(
            expected,
            TraceType::Nil | TraceType::Float | TraceType::Integer | TraceType::Boolean
        ) {
            return self.reject("untyped input guarded for a type with no scalar slot");
        }
        let Some(required) = entry_tag(expected) else {
            return self.reject("entry input without a tag");
        };
        let tag = self.load_at(
            Ty::I64,
            self.inputs,
            Off::K(0),
            offset(slot * mem::size_of::<TraceInputSlot>() + mem::offset_of!(TraceInputSlot, tag)),
        );
        let wanted = self.iconst(required);
        self.guard_cc(Cc::Eq, tag, wanted, guard)?;
        self.load_input(slot, expected)
    }

    /// Stores into the root's input slot of each register the root takes
    /// untyped the value the link state `link` gives it
    /// (`store_untyped_link` of `emit.rs`): the root's type guard reads it
    /// when the loop comes round, and the run leaves the slot for the VM.
    fn store_untyped_link(&mut self, link: &crate::jit_ir::trace::SnapshotFrame) -> Res<()> {
        let root = self.members[0].ir;
        for (slot, input) in root.entry.inputs.iter().enumerate() {
            if input.ty != TraceType::Any {
                continue;
            }
            let (val, ty) = match link.registers.get(usize::from(input.register)) {
                Some(RegisterEntry::Value(value)) => {
                    let ty = self.ty(*value);
                    // The side passes on the slot as it found it.
                    if ty == TraceType::Any {
                        continue;
                    }
                    (self.get(*value)?, ty)
                }
                Some(RegisterEntry::Const(scalar)) => (self.scalar(*scalar)?, scalar.ty()),
                Some(RegisterEntry::Unchanged) | None => continue,
                Some(_) => return self.reject("untyped register a link cannot store"),
            };
            if !matches!(
                ty,
                TraceType::Nil | TraceType::Float | TraceType::Integer | TraceType::Boolean
            ) {
                return self.reject("untyped register linked with a value that has no tag");
            }
            let Some(tag) = entry_tag(ty) else {
                return self.reject("entry input without a tag");
            };
            let at = offset(slot * mem::size_of::<TraceInputSlot>());
            let tag = self.iconst(tag);
            self.store_at(
                self.inputs,
                Off::K(0),
                at + offset(mem::offset_of!(TraceInputSlot, tag)),
                tag,
            );
            let bits = if ty == TraceType::Nil {
                self.iconst(0)
            } else {
                val
            };
            self.store_at(
                self.inputs,
                Off::K(0),
                at + offset(mem::offset_of!(TraceInputSlot, bits)),
                bits,
            );
        }
        Ok(())
    }

    /// The end of a link in the copy of a side that goes straight back to the
    /// fast loop, past the range checks (`fast_link` proved it brings the
    /// next values of the run the checks covered): the fast loop's `Join`,
    /// the second of the root's, takes the parameters the loop carries, the
    /// fuel that is left and the integer copy of each induction, which
    /// is the new value of the induction (it is integral and inside the
    /// array, so the conversion is exact), as Cranelift's `link_target` jumps
    /// to the fast header with its stepped copies.
    fn link_fast(&mut self, next: &[Val]) -> Res<()> {
        let root = self.members[0].ir;
        for (param, val) in root.loop_params.iter().zip(next.iter()) {
            if self.mty(param.ty)? != self.f.val_ty[*val as usize] {
                return self.reject("loop parameter of another type");
            }
        }
        let (remaining, paid) = self.spend_fuel()?;
        let mut args = Vec::with_capacity(self.s.changing.len() + 2 + self.s.shadow_roots.len());
        for k in 0..self.s.changing.len() {
            args.push(next[self.s.changing[k]]);
        }
        args.push(remaining);
        args.extend(paid);
        for k in 0..self.s.shadow_roots.len() {
            let value = next[self.s.shadow_roots[k]];
            let whole = if self.f.val_ty[value as usize] == Ty::F64 {
                self.un(Op::F2I, Ty::I64, value)
            } else {
                value
            };
            let four = self.iconst(4);
            args.push(self.f.bin(Op::Shl, whole, four));
        }
        self.f.rejoin_nth(0, 1, &args);
        #[cfg(test)]
        FAST_LINKS.with(|count| count.set(count.get() + 1));
        Ok(())
    }

    /// The end of a link in the copy of a side that goes straight back to the
    /// guarded loop (`slow_link`: it changes only the parameters that loop
    /// carries): the `Join` of region 1 takes them and the fuel that is left,
    /// as Cranelift's `link_target` jumps to the slow header.
    fn link_slow(&mut self, next: &[Val]) -> Res<()> {
        let root = self.members[0].ir;
        for (param, val) in root.loop_params.iter().zip(next.iter()) {
            if self.mty(param.ty)? != self.f.val_ty[*val as usize] {
                return self.reject("loop parameter of another type");
            }
        }
        let (remaining, paid) = self.spend_fuel()?;
        let mut args = Vec::with_capacity(self.s.changing.len() + 2);
        for k in 0..self.s.changing.len() {
            args.push(next[self.s.changing[k]]);
        }
        args.push(remaining);
        args.extend(paid);
        self.f.rejoin_nth(1, 0, &args);
        #[cfg(test)]
        GUARDED_LINKS.with(|count| count.set(count.get() + 1));
        Ok(())
    }

    /// The end of a link: `next` is the new value of each loop parameter,
    /// and the fuel that is left goes after them.
    fn link_back(&mut self, next: &mut Vec<Val>) -> Res<()> {
        let root = self.members[0].ir;
        for (param, val) in root.loop_params.iter().zip(next.iter()) {
            if self.mty(param.ty)? != self.f.val_ty[*val as usize] {
                return self.reject("loop parameter of another type");
            }
        }
        let (remaining, paid) = self.spend_fuel()?;
        next.push(remaining);
        next.extend(paid);
        self.f.rejoin(next);
        Ok(())
    }

    /// One unit of fuel for the iteration of a loop a side goes back to; the
    /// poll exit when it is gone. Returns the fuel that is left and, in a
    /// counted tree, the budget after the iteration: a link pays what its
    /// snapshot says the path ran, a link to a loop side nothing (as
    /// `emit.rs` does).
    fn spend_fuel(&mut self) -> Res<(Val, Option<Val>)> {
        let poll = self.plan.members[self.member]
            .poll
            .ok_or(Reject::new("looping side without a poll exit"))?;
        for slot in &self.plan.descriptors[poll].slots {
            if !matches!(slot.ty, TraceType::Nil | TraceType::Any) {
                self.get(slot.value)?;
            }
        }
        let one = self.iconst(1);
        let remaining = self.f.bin(Op::Sub, self.fuel, one);
        let zero = self.iconst(0);
        self.fuel = remaining;
        let snap = self.fill_exit(poll)?;
        let mut inst = Inst::new(Op::Guard, Ty::I64);
        inst.cc = Cc::Gt;
        inst.a = remaining;
        inst.b = zero;
        inst.c = snap;
        inst.scale = 1;
        self.f.push(inst);
        let paid = if self.counted {
            let ir = self.ir;
            let cost = match ir.terminator {
                TraceTerminator::Link { snapshot } => ir.snapshots[snapshot.index()].counted,
                _ => 0,
            };
            Some(self.pay(cost))
        } else {
            None
        };
        Ok((remaining, paid))
    }

    /// The budget after an iteration that ran `cost` counted instructions.
    fn pay(&mut self, cost: u32) -> Val {
        if cost == 0 {
            return self.budget;
        }
        let cost = self.iconst(i64::from(cost));
        self.f.bin(Op::Sub, self.budget, cost)
    }

    /// The first value a header or `Join` defines after the fuel (and the
    /// budget of a counted tree), with `self.fuel` set to the fuel's; sets
    /// `self.budget` to the budget's.
    fn after_fuel(&mut self) -> Val {
        if self.counted {
            self.budget = self.fuel + 1;
            self.fuel + 2
        } else {
            self.fuel + 1
        }
    }

    /// A side that rejoins the loop of the loop side around it
    /// (`TraceTerminator::LinkLoop`): the state it ends in names one value per
    /// loop parameter of that loop, which go with the fuel, one unit spent,
    /// to the loop side's `Join`.
    fn link_loop(&mut self, snapshot: crate::jit_ir::trace::SnapshotId) -> Res<()> {
        let ir = self.ir;
        let target = self.members[self.member]
            .link_loop
            .ok_or(Reject::new("side that rejoins a loop side that is gone"))?;
        let loop_ir = self.members[target].ir;
        let state = &ir.snapshots[snapshot.index()].frames[0];
        let mut next = mem::take(&mut self.s.list);
        next.clear();
        let mut built = Ok(());
        for param in &loop_ir.loop_params {
            let Some(register) = param.register else {
                built = self.reject("loop parameter without a register");
                break;
            };
            let val = match state.registers.get(usize::from(register)) {
                Some(RegisterEntry::Value(value))
                    if ir.value_type(*value) == Some(param.ty) && !param.ty.is_borrowed() =>
                {
                    self.get(*value)
                }
                Some(RegisterEntry::Const(scalar)) if scalar.ty() == param.ty => {
                    self.scalar(*scalar)
                }
                _ => self.reject("loop parameter a side cannot rejoin"),
            };
            match val {
                Ok(val)
                    if self
                        .mty(param.ty)
                        .is_ok_and(|ty| ty == self.f.val_ty[val as usize]) =>
                {
                    next.push(val);
                }
                Ok(_) => {
                    built = self.reject("loop parameter of another type");
                    break;
                }
                Err(error) => {
                    built = Err(error);
                    break;
                }
            }
        }
        if built.is_err() {
            self.s.list = next;
            return built;
        }
        if self.count_links {
            let at = offset(mem::offset_of!(TraceControl, links));
            let one = self.iconst(1);
            self.add_at(self.control, Off::K(0), at, one);
        }
        let (remaining, paid) = match self.spend_fuel() {
            Ok(spent) => spent,
            Err(error) => {
                self.s.list = next;
                return Err(error);
            }
        };
        next.push(remaining);
        next.extend(paid);
        self.f.rejoin_to(self.side_of(target) + 1, &next);
        self.s.list = next;
        Ok(())
    }

    /// The part of a loop side before its loop when sides nested in it
    /// rejoin the loop: what the entry gave it goes to frame cells, the loop
    /// parameters and the fuel go to a `Join` those sides jump to, and what
    /// the cells kept is read again after it. The `Join` works like the
    /// root's (see [`Self::join_root`]): the cells carry the entry values, so
    /// a link does not run the entry again.
    fn join_loop_side(&mut self) -> Res<()> {
        let ir = self.ir;
        let mut carried = mem::take(&mut self.s.carried);
        carried.clear();
        for input in &ir.entry.inputs {
            carried.push((self.s.vals[input.value.index()], u32::MAX));
        }
        for input in &ir.frame_inputs {
            carried.push((self.s.vals[input.value.index()], u32::MAX));
        }
        carried.push((self.inputs, u32::MAX));
        carried.push((self.control, u32::MAX));
        for held in &mut carried {
            let cell = self.f.new_cell();
            self.f.cell_store(cell, held.0);
            held.1 = cell;
        }
        let mut init = mem::take(&mut self.s.list);
        init.clear();
        let mut failed = None;
        for param in &ir.loop_params {
            match self.param_init(param.init) {
                Ok(val)
                    if self
                        .mty(param.ty)
                        .is_ok_and(|ty| ty == self.f.val_ty[val as usize]) =>
                {
                    init.push(val);
                }
                Ok(_) => {
                    failed = Some(Reject {
                        what: "loop parameter of another type",
                        at: self.at,
                    });
                    break;
                }
                Err(error) => {
                    failed = Some(error);
                    break;
                }
            }
        }
        if let Some(error) = failed {
            self.s.list = init;
            self.s.carried = carried;
            return Err(error);
        }
        init.push(self.fuel);
        if self.counted {
            init.push(self.budget);
        }
        let first = self.f.join(&init);
        self.s.list = init;
        self.s.inits.clear();
        for (k, param) in ir.loop_params.iter().enumerate() {
            let val = first + k as Val;
            self.s.vals[param.value.index()] = val;
            self.s.inits.push(val);
        }
        self.fuel = first + ir.loop_params.len() as Val;
        if self.counted {
            self.budget = self.fuel + 1;
        }
        let mut at = 0;
        for input in ir
            .entry
            .inputs
            .iter()
            .map(|input| input.value)
            .chain(ir.frame_inputs.iter().map(|input| input.value))
        {
            let (old, cell) = carried[at];
            at += 1;
            let ty = self.f.val_ty[old as usize];
            let val = self.f.cell_load(ty, cell);
            self.s.vals[input.index()] = val;
        }
        let (_, cell) = carried[at];
        self.inputs = self.f.cell_load(Ty::I64, cell);
        let (_, cell) = carried[at + 1];
        self.control = self.f.cell_load(Ty::I64, cell);
        self.s.carried = carried;
        self.rows = NONE;
        self.load_words()
    }

    /// Lowers `member`. With `Fast` or `Guarded` it is the copy a guard of
    /// the fast or the guarded loop enters, whose link goes straight to that
    /// loop's `Join` (`link_fast`, `link_slow`); a copy has snapshots of its
    /// own, as the guarded loop has.
    fn side(&mut self, member: usize, copy: SideCopy) -> Res<()> {
        if copy != SideCopy::Own {
            self.s.side_snaps.fill(u32::MAX);
            self.s.slow_snaps.fill(u32::MAX);
            self.slow = true;
            self.fast_side = copy == SideCopy::Fast;
            self.slow_side = copy == SideCopy::Guarded;
        }
        let result = self.side_code(member, copy);
        self.slow = false;
        self.fast_side = false;
        self.slow_side = false;
        result
    }

    fn side_code(&mut self, member: usize, copy: SideCopy) -> Res<()> {
        self.begin_member(member)?;
        let ir = self.ir;
        let slot = self.f.mark_side();
        debug_assert_eq!(
            slot,
            match copy {
                SideCopy::Own => self.side_of(member),
                SideCopy::Fast => self.s.fast_region[member],
                SideCopy::Guarded => self.s.slow_region[member],
            }
        );
        // Entry inputs, frame inputs, then fuel, the budget of a counted
        // tree, inputs pointer and control pointer, as the guard passes them
        // (`side_args_at`); an activation's side takes the fuel, the control
        // pointer, the depth left and the exit area, and loads the inputs
        // pointer and the root's cells itself.
        let first = self.f.val_ty.len() as Val;
        let mut count = 0;
        let entered = self.entered_snapshot(member);
        let mut windows = Vec::new();
        for input in &ir.entry.inputs {
            if let Some(snapshot) = entered
                && let Some(index) = self.window_input(snapshot, input.register, input.ty)
            {
                windows.push((input.value, index, input.ty));
                continue;
            }
            // An input of either type is a placeholder (see `side_values`).
            let ty = if input.ty == TraceType::Any {
                Ty::I64
            } else {
                self.mty(input.ty)?
            };
            let val = self.f.val(ty);
            self.s.vals[input.value.index()] = val;
            count += 1;
        }
        for input in &ir.frame_inputs {
            let ty = self.mty(input.ty)?;
            let val = self.f.val(ty);
            self.s.vals[input.value.index()] = val;
            count += 1;
        }
        self.fuel = self.f.val(Ty::I64);
        if self.counted {
            self.budget = self.f.val(Ty::I64);
            count += 1;
        }
        if self.function.is_some() {
            // An activation's control pointer, depth left and exit area
            // (see `side_args_at`).
            self.control = self.f.val(Ty::I64);
            let left = self.f.val(Ty::I64);
            let area = self.f.val(Ty::I64);
            count += 4;
            if let Some(activation) = self.function.as_mut() {
                activation.left = left;
                activation.area = area;
            }
        } else {
            self.inputs = self.f.val(Ty::I64);
            self.control = self.f.val(Ty::I64);
            count += 3;
        }
        // What the lookup of the parent's guard found, when this side starts
        // with that lookup (`guard_target` passes it last).
        self.s.handed = None;
        if let Some(last) = self.takes_probe(member) {
            self.s.handed = Some((self.f.val(Ty::I64), last));
            count += 1;
        }
        // The state of the exit the side starts from, when the run builds
        // tables of it here (`bridge_args` passes it last).
        let bridge = self.entry_bridge(member);
        let mut state = Vec::new();
        if let Some((exit, _)) = &bridge {
            if self.function.is_some() {
                return self.reject("virtual table a side cannot start from");
            }
            let plan = self.plan;
            for ExitSlot { ty, slot, .. } in &plan.descriptors[*exit].slots {
                if matches!(ty, TraceType::Nil | TraceType::Any) {
                    continue;
                }
                let ty = self.mty(*ty)?;
                state.push((self.f.val(ty), *slot));
                count += 1;
            }
        }
        let mut entry = Inst::new(Op::SideEntry, Ty::I64);
        entry.a = first;
        entry.b = count;
        self.f.push(entry);
        // A constant or a repeated value that every entry passes (see
        // `note_entry`): the parameter is left unread.
        let region = slot as usize;
        if self.s.entered[region] {
            let args = mem::take(&mut self.s.entry_args[region]);
            // The places of the list: the inputs passed, without the window
            // inputs an activation's side loads itself.
            let inputs = ir
                .entry
                .inputs
                .iter()
                .map(|input| input.value)
                .filter(|value| !windows.iter().any(|(window, _, _)| window == value))
                .chain(ir.frame_inputs.iter().map(|input| input.value));
            let mut placed = mem::take(&mut self.s.held);
            placed.clear();
            for (k, value) in inputs.enumerate() {
                let val = match args.get(k) {
                    Some(&EntryArg::Const(Ty::F64, bits)) => {
                        self.f.fconst(f64::from_bits(bits.cast_unsigned()))
                    }
                    Some(&EntryArg::Const(_, bits)) => self.f.iconst(bits),
                    Some(&EntryArg::Same(earlier)) => placed[earlier as usize],
                    _ => self.s.vals[value.index()],
                };
                self.s.vals[value.index()] = val;
                placed.push(val);
            }
            self.s.held = placed;
            self.s.entry_args[region] = args;
        }
        if self.count_links {
            let at = offset(mem::offset_of!(TraceControl, links));
            let one = self.iconst(1);
            self.add_at(self.control, Off::K(0), at, one);
        }
        if let Some((exit, tables)) = bridge {
            self.build_entry_tables(exit, &state, &tables);
        }
        if self.function.is_some() {
            self.inputs = self.load_at(
                Ty::I64,
                self.control,
                Off::K(0),
                offset(mem::offset_of!(TraceControl, inputs)),
            );
            for (value, index, ty) in windows {
                let val = self.load_input(index, ty)?;
                self.s.vals[value.index()] = val;
            }
            self.load_words()?;
            return self.member_code();
        }
        let preheader = ir.preheader.len();
        if let TraceTerminator::LoopBack { args } = &ir.terminator {
            // A loop side: its loop parameters start from the entry values,
            // or from the `Join` the sides nested in it rejoin.
            self.s.inits.clear();
            // Cranelift runs the preheader of a loop side once, before its
            // header, unless the loop passes a parameter on unchanged: then
            // it runs at the header every rejoin jumps to. The `Join` of a
            // loop side that sides rejoin is where that header is.
            let joined_above = self.s.joined[member] && passes_params(ir);
            if joined_above {
                self.join_loop_side()?;
            } else {
                self.load_words()?;
                for param in &ir.loop_params {
                    let val = self.param_init(param.init)?;
                    self.s.vals[param.value.index()] = val;
                    self.s.inits.push(val);
                }
            }
            for (at, instruction) in ir.preheader.iter().enumerate() {
                self.at = at;
                self.op(&instruction.op, instruction.result)?;
            }
            if self.s.joined[member] && !joined_above {
                self.at = usize::MAX;
                self.join_root()?;
            }
            return self.run_loop(args, preheader);
        }
        self.load_words()?;
        for (at, instruction) in ir.preheader.iter().enumerate() {
            self.at = at;
            self.op(&instruction.op, instruction.result)?;
        }
        for (at, instruction) in ir.body.iter().enumerate() {
            self.at = preheader + at;
            self.op(&instruction.op, instruction.result)?;
        }
        self.at = usize::MAX;
        match ir.terminator {
            TraceTerminator::Link { snapshot } => self.link(snapshot),
            TraceTerminator::LinkLoop { snapshot, .. } => self.link_loop(snapshot),
            TraceTerminator::Exit { snapshot } => {
                let terminal = self.plan.members[member].terminal;
                self.ends_in_exit(member, snapshot, terminal)
            }
            _ => self.reject("side with an unsupported end"),
        }
    }

    // ---- operations -------------------------------------------------------

    #[allow(clippy::too_many_lines)]
    fn op(&mut self, op: &TraceOp, result: Option<ValueId>) -> Res<()> {
        if mem::take(&mut self.s.keys_stale) {
            self.s.keys.clear();
        }
        self.s.keys_stale = !keeps_probed_keys(op);
        if mem::take(&mut self.s.cells_stale) {
            self.s.cells.clear();
        }
        self.s.cells_stale = !self.keeps_cells(op);
        match op {
            TraceOp::Constant(scalar) => {
                let val = self.scalar(*scalar)?;
                if let Some(result) = result {
                    self.s.konst[result.index()] = Some(*scalar);
                }
                self.set(result, val);
            }
            TraceOp::GuardType {
                source,
                expected,
                guard,
            } if self.ty(*source) == TraceType::Any => {
                let val = self.guard_any_input(*source, *expected, *guard)?;
                self.set(result, val);
            }
            TraceOp::Move { source } | TraceOp::GuardType { source, .. } => {
                if let Some(result) = result {
                    let from = source.index();
                    self.s.vals[result.index()] = self.s.vals[from];
                    self.s.cmps[result.index()] = self.s.cmps[from];
                    self.s.konst[result.index()] = self.s.konst[from];
                }
            }
            TraceOp::Negate { source } => {
                let value = self.get(*source)?;
                self.f.cold = self.is_cold(result);
                let val = match self.ty(*source) {
                    TraceType::Integer | TraceType::Int64 | TraceType::UInt64 => {
                        let zero = self.iconst(0);
                        self.f.bin(Op::Sub, zero, value)
                    }
                    _ => self.un(Op::FNeg, Ty::F64, value),
                };
                self.f.cold = false;
                self.set(result, val);
            }
            TraceOp::Arithmetic { op, left, right } => {
                let (l, r) = (self.get(*left)?, self.get(*right)?);
                self.f.cold = self.is_cold(result);
                let val = match self.ty(*left) {
                    TraceType::Integer => self.integer_arithmetic(*op, l, r, *right),
                    ty @ (TraceType::Int64 | TraceType::UInt64) => {
                        self.wide_arithmetic(*op, ty, l, r, *right)
                    }
                    _ => self.float_arithmetic(*op, l, r, *right),
                };
                self.f.cold = false;
                self.set(result, val?);
            }
            TraceOp::IntToFloat { source } => {
                let value = self.get(*source)?;
                let val = if self.ty(*source) == TraceType::UInt64 {
                    // Below 2^63 the signed conversion is right. From 2^63 up
                    // the value is halved with its low bit kept (so the
                    // rounding stays correct), converted and doubled.
                    let one = self.iconst(1);
                    let half = self.f.bin(Op::Shr, value, one);
                    let low = self.f.bin(Op::And, value, one);
                    let sticky = self.f.bin(Op::Or, half, low);
                    let big = self.un(Op::I2F, Ty::F64, sticky);
                    let doubled = self.f.bin(Op::FAdd, big, big);
                    let small = self.un(Op::I2F, Ty::F64, value);
                    let zero = self.iconst(0);
                    self.f.select_any(Cc::Lt, value, zero, doubled, small)
                } else {
                    self.un(Op::I2F, Ty::F64, value)
                };
                self.set(result, val);
            }
            TraceOp::Numeric { op, source } => {
                let value = self.get(*source)?;
                let val = match (op, self.ty(*source)) {
                    (NumericOp::Floor, TraceType::Float) => self.un(Op::FFloor, Ty::F64, value),
                    (NumericOp::Ceil, TraceType::Float) => self.un(Op::FCeil, Ty::F64, value),
                    (NumericOp::Abs, TraceType::Float) => self.un(Op::FAbs, Ty::F64, value),
                    // `MIN` wraps to itself, as `iabs` does.
                    (NumericOp::Abs, TraceType::Integer) => {
                        let zero = self.iconst(0);
                        let negated = self.f.bin(Op::Sub, zero, value);
                        self.f.select_any(Cc::Lt, value, zero, negated, value)
                    }
                    // Truncating and saturating, as `fcvt_to_sint_sat` does:
                    // `F2I` gives the minimum for NaN and for what is out of
                    // range, so 2^63 and up go to the maximum and NaN to
                    // zero. The recorder guards that the float is integral
                    // and in range first, so the fixes are never observable.
                    (NumericOp::FloatToInteger, TraceType::Float) => {
                        let truncated = self.un(Op::F2I, Ty::I64, value);
                        let two_63 = self.f.fconst(9_223_372_036_854_775_808.0);
                        let max = self.iconst(i64::MAX);
                        let zero = self.iconst(0);
                        let capped = self.f.select_any(Cc::Ge, value, two_63, max, truncated);
                        self.f.select_any(Cc::Ne, value, value, zero, capped)
                    }
                    // Adding `1.5 * 2^52` rounds the float to an integer in
                    // the low mantissa bits; the low word of the sum is the
                    // result.
                    (NumericOp::TrickBits, TraceType::Float) => {
                        let bias = self.f.fconst(6_755_399_441_055_744.0);
                        let sum = self.f.bin(Op::FAdd, value, bias);
                        let word = self.f.bits(sum);
                        let mask = self.iconst(0xffff_ffff);
                        self.f.bin(Op::And, word, mask)
                    }
                    _ => return self.reject("numeric operation on a type it is not defined for"),
                };
                self.set(result, val);
            }
            TraceOp::ToWide {
                source,
                target,
                conversion,
            } => {
                let value = self.get(*source)?;
                let val = match (self.ty(*source), conversion) {
                    // C's conversion truncates, and gives `INT64_MIN` for NaN
                    // and out of range values: what `cvttsd2si` does. An
                    // unsigned target converts `x - 2^64` from 2^63 up, which
                    // wraps the result.
                    (TraceType::Float, WideConversion::Truncate) => {
                        if *target == TraceType::Int64 {
                            self.un(Op::F2I, Ty::I64, value)
                        } else {
                            let two_63 = self.f.fconst(9_223_372_036_854_775_808.0);
                            let two_64 = self.f.fconst(18_446_744_073_709_551_616.0);
                            let wrapped = self.f.bin(Op::FSub, value, two_64);
                            let source = self.f.select_any(Cc::Ge, value, two_63, wrapped, value);
                            self.un(Op::F2I, Ty::I64, source)
                        }
                    }
                    (TraceType::Float, WideConversion::Bits32) => self.f.call(
                        trace_number_bits as *const () as usize,
                        &[value],
                        Some(Ty::I64),
                    ),
                    // The other cdata keeps its bits.
                    _ => value,
                };
                self.set(result, val);
            }
            TraceOp::NativeUnary { function, source } => {
                let value = self.get(*source)?;
                let val = if function.address() == crate::stdlib::numeric_sqrt_address() {
                    self.un(Op::FSqrt, Ty::F64, value)
                } else {
                    self.f.call(function.address(), &[value], Some(Ty::F64))
                };
                self.set(result, val);
            }
            TraceOp::NativeBinary {
                function,
                left,
                right,
            } => {
                let (l, r) = (self.get(*left)?, self.get(*right)?);
                let val = self.f.call(function.address(), &[l, r], Some(Ty::F64));
                self.set(result, val);
            }
            TraceOp::Compare {
                comparison,
                left,
                right,
            } => {
                let (mut lt, rt) = (self.ty(*left), self.ty(*right));
                let (mut a, mut b) = (self.get(*left)?, self.get(*right)?);
                if lt != rt {
                    // A float against an integer constant within 2^53: the
                    // constant converts exactly, so the float compare is the
                    // exact one (`w == 0` on the float side of a load of
                    // either subtype). Cranelift folds the same way.
                    const EXACT: i64 = 1 << 53;
                    let exact = |this: &mut Self, val: Val| {
                        this.const_of(val)
                            .filter(|(ty, bits)| *ty == Ty::I64 && (-EXACT..=EXACT).contains(bits))
                            .map(|(_, bits)| bits)
                    };
                    #[allow(clippy::cast_precision_loss, reason = "within 2^53")]
                    match (lt, rt) {
                        (TraceType::Float, TraceType::Integer) => match exact(self, b) {
                            Some(bits) => b = self.f.fconst(bits as f64),
                            None => {
                                return self.reject("comparison of an integer with a float");
                            }
                        },
                        (TraceType::Integer, TraceType::Float) => match exact(self, a) {
                            Some(bits) => {
                                a = self.f.fconst(bits as f64);
                                lt = TraceType::Float;
                            }
                            None => {
                                return self.reject("comparison of an integer with a float");
                            }
                        },
                        _ => return self.reject("comparison of an integer with a float"),
                    }
                }
                self.mty(lt)?;
                let float = lt == TraceType::Float;
                let unsigned = lt == TraceType::UInt64;
                let cc = match (comparison, unsigned) {
                    (Comparison::Equal, _) => Cc::Eq,
                    (Comparison::LessThan, false) => Cc::Lt,
                    (Comparison::LessEqual, false) => Cc::Le,
                    (Comparison::LessThan, true) => Cc::Ult,
                    (Comparison::LessEqual, true) => Cc::Ule,
                };
                if let Some(result) = result {
                    self.s.vals[result.index()] = NONE;
                    self.s.cmps[result.index()] = Cmp {
                        a,
                        b,
                        cc,
                        float,
                        live: true,
                    };
                }
            }
            TraceOp::Truthy { source } => match self.ty(*source) {
                TraceType::Boolean => {
                    if let Some(result) = result {
                        let from = source.index();
                        self.s.vals[result.index()] = self.s.vals[from];
                        self.s.cmps[result.index()] = self.s.cmps[from];
                    }
                }
                TraceType::Nil => {
                    let val = self.iconst(0);
                    self.set(result, val);
                }
                _ => {
                    let val = self.iconst(1);
                    self.set(result, val);
                }
            },
            TraceOp::Not { source } => match self.ty(*source) {
                TraceType::Boolean => match self.cond(*source)? {
                    Cond::Cmp(cmp) => {
                        if let Some(result) = result {
                            self.s.vals[result.index()] = NONE;
                            self.s.cmps[result.index()] = Cmp {
                                cc: cmp.cc.negate(cmp.float),
                                ..cmp
                            };
                        }
                    }
                    Cond::Val(val) => {
                        let one = self.iconst(1);
                        let flipped = self.f.bin(Op::Xor, val, one);
                        self.set(result, flipped);
                    }
                },
                TraceType::Nil => {
                    let val = self.iconst(1);
                    self.set(result, val);
                }
                _ => {
                    let val = self.iconst(0);
                    self.set(result, val);
                }
            },
            TraceOp::Select {
                condition,
                if_true,
                if_false,
            } => {
                self.mty(self.ty(*if_true))?;
                let cond = self.cond(*condition)?;
                let (t, e) = (self.get(*if_true)?, self.get(*if_false)?);
                let val = match cond {
                    Cond::Cmp(cmp) => self.f.select_any(cmp.cc, cmp.a, cmp.b, t, e),
                    Cond::Val(val) => {
                        let zero = self.iconst(0);
                        self.f.select_any(Cc::Ne, val, zero, t, e)
                    }
                };
                self.set(result, val);
            }
            TraceOp::BoolAnd { left, right } => {
                let (l, r) = (self.get(*left)?, self.get(*right)?);
                let val = self.f.bin(Op::And, l, r);
                self.set(result, val);
            }
            TraceOp::GuardBranch {
                condition,
                expected,
                guard,
            } => {
                // The guard a folded loop closes on is made at the back edge.
                if self.overflow_proven(*condition, *expected) || self.folds(*condition, *expected)
                {
                    let passed = self.iconst(i64::from(*expected));
                    self.s.vals[condition.index()] = passed;
                    self.s.cmps[condition.index()] = NO_CMP;
                    return Ok(());
                }
                // The bound of an induction, on its integer copy.
                let (cond, outcome) = match self.bound_limit(*condition, *expected) {
                    Some((cc, a, b)) => (
                        Cond::Cmp(Cmp {
                            a,
                            b,
                            cc,
                            float: false,
                            live: true,
                        }),
                        true,
                    ),
                    None => (self.cond(*condition)?, *expected),
                };
                // Where the guard leaves, the condition is the other of its
                // two values, and where it holds it is `expected`: a snapshot
                // that keeps the condition stores a constant, so a compare
                // that only the guard and its exits read is never made.
                let index = condition.index();
                let held = (self.s.vals[index], self.s.cmps[index]);
                let failed = self.iconst(i64::from(!*expected));
                self.s.vals[index] = failed;
                self.s.cmps[index] = NO_CMP;
                let guarded = self.guard_cond(cond, outcome, *guard);
                if guarded.is_err() {
                    (self.s.vals[index], self.s.cmps[index]) = held;
                    return guarded;
                }
                let passed = self.iconst(i64::from(*expected));
                self.s.vals[index] = passed;
            }
            TraceOp::ArrayLoad {
                binding,
                key,
                expected,
                past_end,
                guard,
                ..
            } => {
                let val = if *past_end {
                    self.array_nil_read(*binding, *key, *guard)?
                } else if matches!(
                    expected,
                    TraceType::Table
                        | TraceType::Closure
                        | TraceType::Thread
                        | TraceType::Nil
                        | TraceType::StringRef
                ) {
                    self.array_element_ref(*binding, *key, *expected, *guard)?
                } else if tag_of(*expected).is_none() {
                    return self.reject("array load of an element of an unsupported type");
                } else {
                    self.array_access(*binding, *key, *expected, None, *guard)?
                };
                self.set(result, val);
            }
            TraceOp::ArrayNext {
                binding,
                key,
                start,
                ty,
                guard,
                ..
            } => {
                let number = self.array_next(*binding, *key, (*start, *ty), result, *guard)?;
                self.set(result, number);
            }
            TraceOp::LaneArrayLoad {
                table,
                field,
                key,
                expected,
                guard,
            } => {
                let val =
                    self.lane_array_access((*table, *field), *key, *expected, None, *guard)?;
                self.set(result, val);
            }
            TraceOp::LaneArrayStore {
                table,
                field,
                key,
                value,
                guard,
                ..
            } => {
                let ty = self.ty(*value);
                if tag_of(ty).is_none() {
                    return self.reject("lane array store of a nil, table or string element");
                }
                let stored = self.get(*value)?;
                self.lane_array_access((*table, *field), *key, ty, Some(stored), *guard)?;
            }
            TraceOp::NumberLoad {
                binding,
                key,
                guard,
                ..
            } => self.number_load(*binding, *key, *guard, result)?,
            TraceOp::NumberTag { load } => {
                let found = self
                    .s
                    .numtags
                    .iter()
                    .find(|(held, _)| *held == load.index() as u32)
                    .map(|(_, tag)| *tag);
                let Some(tag) = found else {
                    return self.reject("number tag of a load that is not here");
                };
                self.set(result, tag);
            }
            TraceOp::NumberModulo {
                float,
                left,
                right,
                integer,
            } => {
                let val = self.number_modulo(*float, (*left, *right), *integer)?;
                self.set(result, val);
            }
            TraceOp::CallHelper {
                helper,
                args,
                guard,
                ..
            } => self.call_helper(*helper, args, result, *guard)?,
            TraceOp::HelperResult { call, index } => self.helper_result(*call, *index, result)?,
            TraceOp::HostCall {
                function,
                args,
                result: ty,
                guard,
            } => {
                let val = self.host_call(*function, args, *ty, *guard)?;
                self.set(result, val);
            }
            TraceOp::HashLoad {
                binding,
                key,
                expected,
                guard,
                ..
            } => {
                let val = self.hash_load(*binding, *key, *expected, *guard)?;
                self.set(result, val);
            }
            TraceOp::ArraySet {
                binding,
                key,
                value,
                guard,
                ..
            } => self.array_set(*binding, *key, *value, *guard)?,
            TraceOp::ReleaseStrings { keep } => self.release_strings(keep)?,
            TraceOp::ArrayStore {
                binding,
                key,
                value,
                guard,
                ..
            } => {
                let ty = self.ty(*value);
                if tag_of(ty).is_none() {
                    return self.reject("array store of a nil, table or string element");
                }
                let stored = self.get(*value)?;
                self.array_access(*binding, *key, ty, Some(stored), *guard)?;
            }
            TraceOp::ArrayOverwrite {
                binding,
                key,
                value,
                ..
            } => {
                let ty = self.ty(*value);
                if tag_of(ty).is_none() {
                    return self.reject("array store of a nil, table or string element");
                }
                let stored = self.get(*value)?;
                self.array_overwrite(*binding, *key, ty, stored)?;
            }
            TraceOp::FieldLoad { binding, .. } => {
                let BindingKind::Field { ty, .. } =
                    &self.members[0].ir.bindings[*binding as usize].kind
                else {
                    return self.reject("field load of another binding");
                };
                let ty = *ty;
                let slot = self.words_of(*binding)?.base;
                let val = self.payload_load(ty, slot)?;
                self.set(result, val);
            }
            TraceOp::FieldStore { binding, value, .. } => {
                let slot = self.words_of(*binding)?.base;
                let ty = self.ty(*value);
                let stored = self.get(*value)?;
                self.payload_store(ty, slot, stored)?;
            }
            TraceOp::TableLength { binding, .. } => {
                let val = self.words_of(*binding)?.base;
                self.set(result, val);
            }
            TraceOp::StringLength { binding, .. } => {
                let BindingKind::Bytes { ty } =
                    &self.members[0].ir.bindings[*binding as usize].kind
                else {
                    return self.reject("string length of another binding");
                };
                let words = self.words_of(*binding)?;
                let val = if *ty == TraceType::Float {
                    words.lenf
                } else {
                    words.len
                };
                self.set(result, val);
            }
            TraceOp::StringByte {
                binding,
                key,
                guard,
                ..
            } => {
                let BindingKind::Bytes { ty } =
                    &self.members[0].ir.bindings[*binding as usize].kind
                else {
                    return self.reject("string byte of another binding");
                };
                let ty = *ty;
                let words = self.words_of(*binding)?;
                let val = self.string_byte(*binding, words, *key, *guard, ty)?;
                self.set(result, val);
            }
            TraceOp::TableFieldLoad {
                field,
                table,
                guard,
            } => {
                let ty = self.members[0].ir.fields[*field as usize].ty;
                let slot = self.field_slot(*table, *field, *guard)?;
                let val = match ty {
                    TraceType::Nil => self.iconst(0),
                    // The lane of the table the field holds, which the VM
                    // gives (`slot_access` of `emit.rs`).
                    TraceType::Table => {
                        let tag = self.load8(slot, Off::K(0), 0);
                        let wanted = self.iconst(i64::from(NATIVE_VALUE_TAG_TABLE));
                        self.guard_cc(Cc::Eq, tag, wanted, *guard)?;
                        let lane = self.admit_lane(slot, true);
                        let zero = self.iconst(0);
                        self.guard_cc(Cc::Ge, lane, zero, *guard)?;
                        lane
                    }
                    _ => self.payload_load(ty, slot)?,
                };
                self.set(result, val);
            }
            TraceOp::TableFieldStore {
                field,
                table,
                value,
                guard,
                ..
            } => {
                let slot = self.field_slot(*table, *field, *guard)?;
                let ty = self.ty(*value);
                let stored = self.get(*value)?;
                self.payload_store(ty, slot, stored)?;
            }
            TraceOp::MetaGuard {
                table,
                binding,
                guard,
            } => {
                let BindingKind::MetaClass { metatable, .. } =
                    &self.members[0].ir.bindings[*binding as usize].kind
                else {
                    return self.reject("meta guard of another binding");
                };
                let identity = meta_word(*metatable).cast_signed();
                let Some(field) = self.members[0]
                    .ir
                    .fields
                    .iter()
                    .position(|field| field.key.as_ref() == META_FIELD_KEY)
                else {
                    return self.reject("meta guard without a metatable field");
                };
                let lane = self.get(*table)?;
                let row = self.row_address(lane)?;
                let word = self.load_at(
                    Ty::I64,
                    row,
                    Off::K(0),
                    offset(field * mem::size_of::<usize>()),
                );
                let wanted = self.iconst(identity);
                self.guard_cc(Cc::Eq, word, wanted, *guard)?;
            }
            TraceOp::PhantomTable { entries } => {
                let bytes =
                    i64::try_from(crate::runtime::Runtime::elided_footprint(*entries as usize))
                        .expect("a table footprint fits i64");
                for (field, amount) in [
                    (mem::offset_of!(TraceControl, elided_tables), 1),
                    (mem::offset_of!(TraceControl, elided_bytes), bytes),
                ] {
                    let step = self.iconst(amount);
                    self.add_at(self.control, Off::K(0), offset(field), step);
                }
            }
            TraceOp::SelfCall { args, guard } if self.function.is_some() => {
                self.self_call(args, *guard, result)?;
            }
            other => return self.reject(unsupported(other)),
        }
        Ok(())
    }

    fn integer_arithmetic(&mut self, op: ArithmeticOp, l: Val, r: Val, right: ValueId) -> Res<Val> {
        let mir = match op {
            ArithmeticOp::Add => Op::Add,
            ArithmeticOp::Subtract => Op::Sub,
            ArithmeticOp::Multiply => Op::Mul,
            ArithmeticOp::BitAnd => Op::And,
            ArithmeticOp::BitOr => Op::Or,
            ArithmeticOp::BitXor => Op::Xor,
            ArithmeticOp::Modulo | ArithmeticOp::FloorDivide => {
                return Ok(self.integer_division(op, l, r, right));
            }
            ArithmeticOp::ShiftLeft | ArithmeticOp::ShiftRight => {
                let Some(TraceScalar::Integer(distance)) = self.s.konst[right.index()] else {
                    if !self.shifts {
                        return self.reject("shift by a value on a target without BMI2");
                    }
                    return Ok(self.integer_shift(op, l, r));
                };
                // `a >> b` is `a << -b`, and negating the most negative
                // distance saturates, as the interpreter does.
                let distance = if op == ArithmeticOp::ShiftLeft {
                    distance
                } else if distance == i64::MIN {
                    i64::MAX
                } else {
                    -distance
                };
                // Left for a positive distance, logical right for a negative
                // one, zero from 64 places.
                return Ok(match distance {
                    0 => l,
                    1..=63 => {
                        let by = self.iconst(distance);
                        self.f.bin(Op::Shl, l, by)
                    }
                    -63..=-1 => {
                        let by = self.iconst(-distance);
                        self.f.bin(Op::Shr, l, by)
                    }
                    _ => self.iconst(0),
                });
            }
            _ => return self.reject("integer operation with no MIR form"),
        };
        Ok(self.f.bin(mir, l, r))
    }

    /// Lua's shift by a distance known at run time (`integer_shift` in
    /// `emit.rs`): left for a positive distance, logical right for a negative
    /// one, zero when the distance is 64 places or more. `a >> b` is `a <<
    /// -b`, and negating the most negative distance saturates, as the
    /// interpreter does. Both machine shifts run and a select takes one, since
    /// the count is reduced modulo 64 and the range test picks the zero.
    fn integer_shift(&mut self, op: ArithmeticOp, value: Val, count: Val) -> Val {
        let zero = self.iconst(0);
        let distance = if op == ArithmeticOp::ShiftLeft {
            count
        } else {
            let negated = self.f.bin(Op::Sub, zero, count);
            let most_negative = self.iconst(i64::MIN);
            let largest = self.iconst(i64::MAX);
            self.f
                .select_any(Cc::Eq, count, most_negative, largest, negated)
        };
        let flipped = self.f.bin(Op::Sub, zero, distance);
        let amount = self.f.select_any(Cc::Lt, distance, zero, flipped, distance);
        let left = self.f.bin(Op::Shl, value, amount);
        let right = self.f.bin(Op::Shr, value, amount);
        let shifted = self.f.select_any(Cc::Lt, distance, zero, right, left);
        // In range when -63 <= distance <= 63.
        let sixty_three = self.iconst(63);
        let offset = self.f.bin(Op::Add, distance, sixty_three);
        let limit = self.iconst(126);
        self.f.select_any(Cc::Ule, offset, limit, shifted, zero)
    }

    /// Floor division or modulo with the sign rules of Lua, the sequences of
    /// `emit.rs` (`integer_division_by_constant`, `integer_division`). Never
    /// traps: a divisor of 0 or -1 is replaced by 1 before the machine
    /// divides, then the results the replacement got wrong are selected away.
    fn integer_division(&mut self, op: ArithmeticOp, l: Val, r: Val, right: ValueId) -> Val {
        let modulo = op == ArithmeticOp::Modulo;
        if let Some(TraceScalar::Integer(divisor)) = self.s.konst[right.index()]
            && divisor > 0
        {
            // A power of two is a mask or an arithmetic shift, which round
            // toward negative infinity on their own.
            if divisor & (divisor - 1) == 0 {
                return if modulo {
                    let mask = self.iconst(divisor - 1);
                    self.f.bin(Op::And, l, mask)
                } else {
                    let by = self.iconst(i64::from(divisor.trailing_zeros()));
                    self.f.bin(Op::Sar, l, by)
                };
            }
            if modulo {
                let remainder = self.f.bin(Op::SRem, l, r);
                // A negative remainder is lifted into [0, divisor). The sum
                // cannot overflow: the remainder is above `-divisor`.
                let zero = self.iconst(0);
                let lifted = self.f.bin(Op::Add, remainder, r);
                return self
                    .f
                    .select_any(Cc::Lt, remainder, zero, lifted, remainder);
            }
            let sixty_three = self.iconst(63);
            let quotient = self.f.bin(Op::SDiv, l, r);
            // The truncated quotient is one too high exactly when the
            // remainder is negative.
            let product = self.f.bin(Op::Mul, quotient, r);
            let remainder = self.f.bin(Op::Sub, l, product);
            let sign = self.f.bin(Op::Sar, remainder, sixty_three);
            return self.f.bin(Op::Add, quotient, sign);
        }
        // `r` is 0 or -1 exactly when `r + 1` is 0 or 1, unsigned.
        let (one, zero) = (self.iconst(1), self.iconst(0));
        let shifted = self.f.bin(Op::Add, r, one);
        let divisor = self.f.select_any(Cc::Ule, shifted, one, one, r);
        let remainder = self.f.bin(Op::SRem, l, divisor);
        // Round toward negative infinity: the truncated remainder has the
        // wrong sign when it is non-zero and differs in sign from the divisor.
        let signs = self.f.bin(Op::Xor, remainder, divisor);
        if modulo {
            // The divisor is added when the signs differ and the remainder is
            // not zero. For a replaced divisor the remainder is 0, so this
            // adds 0.
            let differ = self.f.select_any(Cc::Lt, signs, zero, divisor, zero);
            let correction = self.f.select_any(Cc::Ne, remainder, zero, differ, zero);
            return self.f.bin(Op::Add, remainder, correction);
        }
        let differ = self.f.select_any(Cc::Lt, signs, zero, one, zero);
        let adjust = self.f.select_any(Cc::Ne, remainder, zero, differ, zero);
        let quotient = self.f.bin(Op::SDiv, l, divisor);
        let floored = self.f.bin(Op::Sub, quotient, adjust);
        let by_zero = self.f.select_any(Cc::Eq, r, zero, zero, floored);
        let minus_one = self.iconst(-1);
        let negated = self.f.bin(Op::Sub, zero, l);
        self.f.select_any(Cc::Eq, r, minus_one, negated, by_zero)
    }

    /// Arithmetic on two 64-bit integer cdata (`Int64` or `UInt64`), as
    /// `emit.rs` `wide_arithmetic` defines it: wrapping add, subtract and
    /// multiply, the bit operations, shifts and rotations by a constant (the
    /// count is taken modulo 64, as the `bit` library does), and division and
    /// remainder that never trap (a zero divisor gives `i64::MIN`).
    fn wide_arithmetic(
        &mut self,
        op: ArithmeticOp,
        ty: TraceType,
        l: Val,
        r: Val,
        right: ValueId,
    ) -> Res<Val> {
        let mir = match op {
            ArithmeticOp::Add => Op::Add,
            ArithmeticOp::Subtract => Op::Sub,
            ArithmeticOp::Multiply => Op::Mul,
            ArithmeticOp::BitAnd => Op::And,
            ArithmeticOp::BitOr => Op::Or,
            ArithmeticOp::BitXor => Op::Xor,
            ArithmeticOp::ShiftLeft
            | ArithmeticOp::ShiftRight
            | ArithmeticOp::ArithmeticShiftRight
            | ArithmeticOp::RotateLeft
            | ArithmeticOp::RotateRight => {
                let distance = match self.s.konst[right.index()] {
                    Some(TraceScalar::UInt64(count)) => count.cast_signed(),
                    Some(TraceScalar::Int64(count) | TraceScalar::Integer(count)) => count,
                    _ => return self.reject("shift of a 64-bit integer by a value"),
                };
                let count = distance & 63;
                if count == 0 {
                    return Ok(l);
                }
                let by = self.iconst(count);
                return Ok(match op {
                    ArithmeticOp::ShiftLeft => self.f.bin(Op::Shl, l, by),
                    ArithmeticOp::ShiftRight => self.f.bin(Op::Shr, l, by),
                    ArithmeticOp::ArithmeticShiftRight => self.f.bin(Op::Sar, l, by),
                    _ => {
                        // A rotation is the two shifts and an or.
                        let back = self.iconst(64 - count);
                        let (first, second) = if op == ArithmeticOp::RotateLeft {
                            (self.f.bin(Op::Shl, l, by), self.f.bin(Op::Shr, l, back))
                        } else {
                            (self.f.bin(Op::Shr, l, by), self.f.bin(Op::Shl, l, back))
                        };
                        self.f.bin(Op::Or, first, second)
                    }
                });
            }
            ArithmeticOp::Divide | ArithmeticOp::Modulo => {
                return Ok(self.wide_division(op, ty == TraceType::Int64, l, r));
            }
            _ => return self.reject("operation of 64-bit integers with no MIR form"),
        };
        Ok(self.f.bin(mir, l, r))
    }

    /// Division and remainder of two 64-bit integer cdata. A zero divisor
    /// gives `i64::MIN`, and the divisor the machine sees is never 0 or -1.
    fn wide_division(&mut self, op: ArithmeticOp, signed: bool, l: Val, r: Val) -> Val {
        let (one, zero) = (self.iconst(1), self.iconst(0));
        let failed = self.iconst(i64::MIN);
        let modulo = op == ArithmeticOp::Modulo;
        if signed {
            // A divisor of 0 or -1 is 0 or 1 after adding one, unsigned.
            let shifted = self.f.bin(Op::Add, r, one);
            let divisor = self.f.select_any(Cc::Ule, shifted, one, one, r);
            let result = if modulo {
                // Both replaced divisors leave a zero remainder.
                self.f.bin(Op::SRem, l, divisor)
            } else {
                let quotient = self.f.bin(Op::SDiv, l, divisor);
                let minus_one = self.iconst(-1);
                let negated = self.f.bin(Op::Sub, zero, l);
                self.f.select_any(Cc::Eq, r, minus_one, negated, quotient)
            };
            return self.f.select_any(Cc::Eq, r, zero, failed, result);
        }
        let divisor = self.f.select_any(Cc::Eq, r, zero, one, r);
        let result = self
            .f
            .bin(if modulo { Op::URem } else { Op::UDiv }, l, divisor);
        self.f.select_any(Cc::Eq, r, zero, failed, result)
    }

    fn float_arithmetic(&mut self, op: ArithmeticOp, l: Val, r: Val, right: ValueId) -> Res<Val> {
        Ok(match op {
            ArithmeticOp::Add => self.f.bin(Op::FAdd, l, r),
            ArithmeticOp::Subtract => self.f.bin(Op::FSub, l, r),
            ArithmeticOp::Multiply => self.f.bin(Op::FMul, l, r),
            ArithmeticOp::Divide => self.f.bin(Op::FDiv, l, r),
            ArithmeticOp::Modulo | ArithmeticOp::ModuloWith(FloatModulo::Floor) => {
                let quotient = self.f.bin(Op::FDiv, l, r);
                let floored = self.un(Op::FFloor, Ty::F64, quotient);
                let product = self.f.bin(Op::FMul, floored, r);
                self.f.bin(Op::FSub, l, product)
            }
            ArithmeticOp::FloorDivide => {
                let quotient = self.f.bin(Op::FDiv, l, r);
                self.un(Op::FFloor, Ty::F64, quotient)
            }
            ArithmeticOp::PowerSquare => {
                if matches!(
                    self.s.konst[right.index()],
                    Some(TraceScalar::Float(two)) if two.to_bits() == 2.0_f64.to_bits()
                ) {
                    self.f.bin(Op::FMul, l, l)
                } else {
                    // `a * a` for an exponent of 2, the `pow` helper
                    // otherwise, as `float_power_square` has it. The call is
                    // made either way (this backend has no cold path) and
                    // the select drops its result.
                    let square = self.f.bin(Op::FMul, l, l);
                    let power =
                        self.f
                            .call(trace_pow as *const () as usize, &[l, r], Some(Ty::F64));
                    let two = self.f.fconst(2.0);
                    self.f.select_any(Cc::Eq, r, two, square, power)
                }
            }
            ArithmeticOp::Power => {
                self.f
                    .call(trace_pow as *const () as usize, &[l, r], Some(Ty::F64))
            }
            // The compat function is what the inline fast paths of `emit.rs`
            // are exact against.
            ArithmeticOp::ModuloWith(FloatModulo::Fmod53) => self.float_fmod(l, r, false),
            ArithmeticOp::ModuloWith(FloatModulo::Fmod54) => self.float_fmod(l, r, true),
            _ => return self.reject("bitwise operation on floats"),
        })
    }

    // ---- tables -----------------------------------------------------------

    /// The payload of the value at `slot`, of type `ty`.
    fn payload_load(&mut self, ty: TraceType, slot: Val) -> Res<Val> {
        Ok(match ty {
            TraceType::Float => self.load_at(
                Ty::F64,
                slot,
                Off::K(0),
                i64::from(NATIVE_VALUE_PAYLOAD_OFFSET),
            ),
            TraceType::Integer => self.load_at(
                Ty::I64,
                slot,
                Off::K(0),
                i64::from(NATIVE_VALUE_PAYLOAD_OFFSET),
            ),
            TraceType::Boolean => {
                self.load8(slot, Off::K(0), i64::from(NATIVE_VALUE_PAYLOAD_OFFSET))
            }
            _ => return self.reject("load of a table, closure or string field"),
        })
    }

    fn payload_store(&mut self, ty: TraceType, slot: Val, value: Val) -> Res<()> {
        match ty {
            TraceType::Float | TraceType::Integer => {
                self.store_at(
                    slot,
                    Off::K(0),
                    i64::from(NATIVE_VALUE_PAYLOAD_OFFSET),
                    value,
                );
            }
            TraceType::Boolean => {
                self.store8(
                    slot,
                    Off::K(0),
                    i64::from(NATIVE_VALUE_PAYLOAD_OFFSET),
                    value,
                );
            }
            _ => return self.reject("store of a table, closure or string field"),
        }
        Ok(())
    }

    /// `val` as `base + constant` when one of the last instructions made it
    /// as that integer sum (the search is short: a key is used right after
    /// it is made).
    fn sum_with_constant(&mut self, val: Val) -> Option<(Val, i64)> {
        const NEAR: usize = 48;
        let at = self
            .f
            .insts
            .iter()
            .rev()
            .take(NEAR)
            .find(|inst| inst.dst == val)
            .filter(|inst| inst.op == Op::Add && inst.ty == Ty::I64)
            .map(|inst| (inst.a, inst.b))?;
        match (self.const_of(at.0), self.const_of(at.1)) {
            (None, Some((Ty::I64, addend))) => Some((at.0, addend)),
            (Some((Ty::I64, addend)), None) => Some((at.1, addend)),
            _ => None,
        }
    }

    /// The zero-based slot `key` names in an array or byte string with the
    /// words `words`, as an offset of `1 << shift` bytes per slot, with a
    /// guard that leaves through `guard` unless `key` is an integral number
    /// inside it.
    ///
    /// In the guarded version of a versioned loop, a key the range plan
    /// proves that misses its array (a loop over `t[0]` to `t[n]` starts
    /// outside) enters the side's own code, whose link checks the plan again
    /// at the root's outer `Join`: the keys after it may all be inside. The
    /// side's copy for the guarded loop, which a slot of another type enters,
    /// would run the rest of the loop with the checks.
    fn index_within(&mut self, words: Words, key: ValueId, guard: GuardId, shift: u32) -> Res<Off> {
        let miss = self.member == 0 && self.slow && self.s.proven_guards.contains(&guard.index());
        // Only a key the induction has yet to bring into the array is worth
        // the checks again: below slot 1 for one that moves up, past the
        // length for one that moves down. A key that left the array on the
        // far side stays out for the rest of the run, and its side goes on
        // in the guarded loop.
        let upward = self
            .s
            .proven_keys
            .iter()
            .find(|(proven, _)| miss && *proven == key.index())
            .map(|(_, upward)| *upward);
        self.index_checked(words, key, guard, shift, upward)
    }

    /// [`Self::index_within`]; `behind` is the direction of the induction
    /// of a key whose miss on the side it comes from enters the side's own
    /// code.
    fn index_checked(
        &mut self,
        words: Words,
        key: ValueId,
        guard: GuardId,
        shift: u32,
        behind: Option<bool>,
    ) -> Res<Off> {
        let constant_slot = match self.s.konst[key.index()] {
            Some(TraceScalar::Float(number))
                if (1.0..=FLOAT_INTEGER_MAGIC).contains(&number) && number.fract() == 0.0 =>
            {
                Some(number as i64 - 1)
            }
            Some(TraceScalar::Integer(number)) if number >= 1 => Some(number - 1),
            _ => None,
        };
        if let Some(slot) = constant_slot {
            let index = self.iconst(slot);
            self.guard_cc(Cc::Ult, index, words.len, guard)?;
            return Ok(Off::K(slot << shift));
        }
        let key_val = self.get(key)?;
        let index = if self.ty(key) == TraceType::Float {
            if words.lenf == NONE {
                return self.reject("float key without the length as a float");
            }
            let one = self.f.fconst(1.0);
            self.key_miss = behind == Some(true);
            let low = self.guard_cc(Cc::Ge, key_val, one, guard);
            self.key_miss = behind == Some(false);
            let high = low.and_then(|()| self.guard_cc(Cc::Le, key_val, words.lenf, guard));
            self.key_miss = false;
            high?;
            let magic = self.f.fconst(FLOAT_INTEGER_MAGIC);
            let shifted = self.f.bin(Op::FAdd, key_val, magic);
            let restored = self.f.bin(Op::FSub, shifted, magic);
            self.guard_cc(Cc::Eq, restored, key_val, guard)?;
            let whole = self.un(Op::F2I, Ty::I64, key_val);
            let minus = self.iconst(-1);
            self.f.bin(Op::Add, whole, minus)
        } else {
            // A key that is a sum with a constant (`i % n + 1`) is one less
            // without a second add: the sum itself is then read by the
            // exits alone and needs no place of its own in the loop.
            let index = match self.sum_with_constant(key_val) {
                Some((base, 1)) => base,
                Some((base, addend)) => {
                    let less = self.iconst(addend.wrapping_sub(1));
                    self.f.bin(Op::Add, base, less)
                }
                None => {
                    let minus = self.iconst(-1);
                    self.f.bin(Op::Add, key_val, minus)
                }
            };
            if let Some(upward) = behind {
                self.key_miss = true;
                let came = if upward {
                    let zero = self.iconst(0);
                    self.guard_cc(Cc::Ge, index, zero, guard)
                } else {
                    self.guard_cc(Cc::Lt, index, words.len, guard)
                };
                self.key_miss = false;
                came?;
            }
            self.guard_cc(Cc::Ult, index, words.len, guard)?;
            index
        };
        if shift == 0 {
            return Ok(Off::V(index));
        }
        let by = self.iconst(i64::from(shift));
        Ok(Off::V(self.f.bin(Op::Shl, index, by)))
    }

    /// An access to the array part of a bound table: leaves through the
    /// guard unless `key` is an integral number naming a slot inside the
    /// array that holds a value of type `ty`. A load returns the payload, a
    /// store (`stored`) overwrites it.
    fn array_access(
        &mut self,
        binding: u32,
        key: ValueId,
        ty: TraceType,
        stored: Option<Val>,
        guard: GuardId,
    ) -> Res<Val> {
        if !self.members[0].ir.bindings[binding as usize].is_array() {
            return self.reject("array access through another binding");
        }
        let words = self.words_of(binding)?;
        let fact = self
            .s
            .slots
            .iter()
            .find(|fact| fact.binding == binding && fact.key == key.index() as u32)
            .copied();
        let attached = self.members[self.member]
            .attached
            .iter()
            .any(|(attached, _)| *attached == guard.index());
        let off = match fact {
            Some(fact) if !attached => {
                if fact.ty == ty {
                    return self.payload_access(ty, words.base, fact.off, stored);
                }
                fact.off
            }
            _ => match self.proven_off(binding, key, guard)? {
                Some(off) => off,
                None => self.index_within(words, key, guard, 4)?,
            },
        };
        let tag = self.load8(words.base, off, 0);
        let wanted = self.iconst(tag_of(ty).expect("checked by the caller"));
        self.guard_cc(Cc::Eq, tag, wanted, guard)?;
        self.s
            .slots
            .retain(|fact| !(fact.binding == binding && fact.key == key.index() as u32));
        self.s.slots.push(SlotFact {
            binding,
            key: key.index() as u32,
            off,
            ty,
        });
        self.payload_access(ty, words.base, off, stored)
    }

    /// `ArrayOverwrite`: stores into the slot of `key` on `binding`, which
    /// the verifier proved an earlier access in this segment reached with a
    /// value of type `ty`. The store goes to the offset that access left in
    /// the slot facts. A member whose facts lost the slot (a guard attached
    /// a side in between, say) stays with Cranelift, which re-derives the
    /// address and traps on a failed check.
    fn array_overwrite(
        &mut self,
        binding: u32,
        key: ValueId,
        ty: TraceType,
        stored: Val,
    ) -> Res<()> {
        if !self.members[0].ir.bindings[binding as usize].is_array() {
            return self.reject("array access through another binding");
        }
        let words = self.words_of(binding)?;
        let Some(fact) = self
            .s
            .slots
            .iter()
            .find(|fact| fact.binding == binding && fact.key == key.index() as u32 && fact.ty == ty)
            .copied()
        else {
            return self.reject("array overwrite of a slot this member has not reached");
        };
        self.payload_access(ty, words.base, fact.off, Some(stored))?;
        Ok(())
    }

    /// `NumberLoad`: element `key` as a float, whichever subtype the slot
    /// holds, and a guard that leaves unless the slot is inside the array and
    /// holds a float or an integer of magnitude at most 2^53.
    ///
    /// A float costs the tag's compare: the integer's checks and its
    /// conversion are a skip on "not a float", as Cranelift branches on the
    /// tag, and the value and the integer flag are the skip's join values.
    /// (Built as selects of 0 and 1, the checks cost about 14 instructions a
    /// load more on `opt/mod_mixed`.)
    ///
    /// The magnitude test is one biased unsigned compare. Its two constants
    /// are too wide for immediates and are read from the constant pool.
    fn number_load(
        &mut self,
        binding: u32,
        key: ValueId,
        guard: GuardId,
        result: Option<ValueId>,
    ) -> Res<()> {
        if !self.members[0].ir.bindings[binding as usize].is_array() {
            return self.reject("array access through another binding");
        }
        let words = self.words_of(binding)?;
        let off = match self.proven_off(binding, key, guard)? {
            Some(off) => off,
            None => self.index_within(words, key, guard, 4)?,
        };
        let tag = self.load8(words.base, off, 0);
        let payload = i64::from(NATIVE_VALUE_PAYLOAD_OFFSET);
        let float = self.load_at(Ty::F64, words.base, off, payload);
        let (number, whole) = (
            self.iconst(i64::from(NATIVE_VALUE_TAG_NUMBER)),
            self.iconst(i64::from(NATIVE_VALUE_TAG_INTEGER)),
        );
        let (one, zero) = (self.iconst(1), self.iconst(0));
        let bias = self.iconst(1 << 53);
        let limit = self.iconst(1 << 54);
        // The exit's snapshot reads only values made before the skip opens.
        self.guard_target(guard)?;
        let open = self.f.begin_if(Cc::Ne, tag, number);
        self.guard_cc(Cc::Eq, tag, whole, guard)?;
        let integer = self.load_at(Ty::I64, words.base, off, payload);
        let biased = self.f.bin(Op::Add, integer, bias);
        self.guard_cc(Cc::Ule, biased, limit, guard)?;
        let converted = self.un(Op::I2F, Ty::F64, integer);
        let value = self.f.end_if(open, &[(converted, float), (one, zero)]);
        let is_integer = value + 1;
        self.set(result, value);
        if let Some(result) = result {
            self.s.numtags.push((result.index() as u32, is_integer));
        }
        Ok(())
    }

    /// The load of a table, closure, thread or hole from an array: the tag
    /// check, then for a handle the VM's admission of it to the run's bank,
    /// which gives its lane (and fills its row), or the guard leaves.
    fn array_element_ref(
        &mut self,
        binding: u32,
        key: ValueId,
        ty: TraceType,
        guard: GuardId,
    ) -> Res<Val> {
        if !self.members[0].ir.bindings[binding as usize].is_array() {
            return self.reject("array access through another binding");
        }
        let words = self.words_of(binding)?;
        // A key the range plan proved needs no bounds check here either: the
        // guard keeps the slot's type, as it does for a scalar load.
        match self.proven_off(binding, key, guard)? {
            Some(off) => self.element_ref_at(words, off, ty, guard),
            None => self.element_ref(words, key, ty, guard),
        }
    }

    /// [`Self::array_element_ref`] on the array words `words`, which a
    /// binding or a table met by lane gives.
    fn element_ref(
        &mut self,
        words: Words,
        key: ValueId,
        ty: TraceType,
        guard: GuardId,
    ) -> Res<Val> {
        let off = self.index_within(words, key, guard, 4)?;
        self.element_ref_at(words, off, ty, guard)
    }

    /// [`Self::element_ref`] of the slot at `off`, which is inside the array.
    fn element_ref_at(
        &mut self,
        words: Words,
        off: Off,
        ty: TraceType,
        guard: GuardId,
    ) -> Res<Val> {
        let wanted = match ty {
            TraceType::Table => NATIVE_VALUE_TAG_TABLE,
            TraceType::Closure => NATIVE_VALUE_TAG_CLOSURE,
            TraceType::Thread => NATIVE_VALUE_TAG_THREAD,
            TraceType::StringRef => NATIVE_VALUE_TAG_STRING,
            _ => NATIVE_VALUE_TAG_NIL,
        };
        let tag = self.load8(words.base, off, 0);
        let wanted = self.iconst(i64::from(wanted));
        self.guard_cc(Cc::Eq, tag, wanted, guard)?;
        if ty == TraceType::Nil {
            return Ok(self.iconst(0));
        }
        let slot = match off {
            Off::K(0) => words.base,
            Off::K(k) => {
                let at = self.iconst(k);
                self.f.bin(Op::Add, words.base, at)
            }
            Off::V(v) => self.f.bin(Op::Add, words.base, v),
            Off::VK(v, k) => {
                let at = self.iconst(k);
                let sum = self.f.bin(Op::Add, words.base, v);
                self.f.bin(Op::Add, sum, at)
            }
        };
        if ty == TraceType::StringRef {
            // A borrow: the address of the slot, with no lane.
            return Ok(slot);
        }
        let lane = self.admit_lane(slot, ty == TraceType::Table);
        let zero = self.iconst(0);
        self.guard_cc(Cc::Ge, lane, zero, guard)?;
        Ok(lane)
    }

    /// The first header binding on the register of the array binding `array`
    /// and whether it is strict, as `header_binding` of `emit.rs` finds it.
    pub(super) fn header_binding(&self, array: u32) -> Option<(usize, bool)> {
        header_binding_of(&self.members[0].ir.bindings, array)
    }

    /// The array words of the table in lane `table`, read from the table the
    /// word `field` of the lane's row names (`lane_array_words` of
    /// `emit.rs`): the guard leaves when the row has no address. The words are
    /// kept for the straight-line code after, and dropped where the words of
    /// the bindings are (see [`Self::forget_words`]), since a set may have
    /// moved the array. The length as a float is made for a float key.
    fn lane_array_words(
        &mut self,
        table: ValueId,
        field: u32,
        float_key: bool,
        guard: GuardId,
    ) -> Res<Words> {
        let lane = self.get(table)?;
        if let Some(at) = self.s.lane_words.iter().position(|(held, _)| *held == lane) {
            let mut words = self.s.lane_words[at].1;
            if float_key && words.lenf == NONE {
                words.lenf = self.un(Op::I2F, Ty::F64, words.len);
                self.s.lane_words[at].1 = words;
            }
            return Ok(words);
        }
        let Some(layout) = NativeTableLayout::get() else {
            return self.reject("table layout of this build is not known");
        };
        let row = self.row_address(lane)?;
        let address = self.load_at(
            Ty::I64,
            row,
            Off::K(0),
            offset(field as usize * mem::size_of::<usize>()),
        );
        let zero = self.iconst(0);
        self.guard_cc(Cc::Ne, address, zero, guard)?;
        let base = self.load_at(Ty::I64, address, Off::K(0), i64::from(layout.array_pointer));
        let len = self.load_at(Ty::I64, address, Off::K(0), i64::from(layout.array_length));
        let lenf = if float_key {
            self.un(Op::I2F, Ty::F64, len)
        } else {
            NONE
        };
        let words = Words { base, len, lenf };
        self.s.lane_words.push((lane, words));
        Ok(words)
    }

    /// A [`TraceOp::LaneArrayLoad`], or with `stored` a
    /// [`TraceOp::LaneArrayStore`]: the access of [`Self::array_access`] on
    /// the words of a table met by lane (`lane_array_load` of `emit.rs`). The
    /// store overwrites the payload of a slot whose tag checked out, so
    /// nothing moves and the words stay right.
    fn lane_array_access(
        &mut self,
        (table, field): (ValueId, u32),
        key: ValueId,
        ty: TraceType,
        stored: Option<Val>,
        guard: GuardId,
    ) -> Res<Val> {
        let float_key = self.ty(key) == TraceType::Float;
        let words = self.lane_array_words(table, field, float_key, guard)?;
        if matches!(
            ty,
            TraceType::Table | TraceType::Closure | TraceType::Thread | TraceType::StringRef
        ) {
            return self.element_ref(words, key, ty, guard);
        }
        if tag_of(ty).is_none() {
            return self.reject("lane array load of an element of an unsupported type");
        }
        let off = self.index_within(words, key, guard, 4)?;
        let tag = self.load8(words.base, off, 0);
        let wanted = self.iconst(tag_of(ty).expect("checked above"));
        self.guard_cc(Cc::Eq, tag, wanted, guard)?;
        self.payload_access(ty, words.base, off, stored)
    }

    fn payload_access(
        &mut self,
        ty: TraceType,
        base: Val,
        off: Off,
        stored: Option<Val>,
    ) -> Res<Val> {
        let disp = i64::from(NATIVE_VALUE_PAYLOAD_OFFSET);
        match (ty, stored) {
            (TraceType::Float, None) => Ok(self.load_at(Ty::F64, base, off, disp)),
            (TraceType::Integer, None) => Ok(self.load_at(Ty::I64, base, off, disp)),
            (TraceType::Boolean, None) => Ok(self.load8(base, off, disp)),
            (TraceType::Float | TraceType::Integer, Some(value)) => {
                self.store_at(base, off, disp, value);
                Ok(value)
            }
            (TraceType::Boolean, Some(value)) => {
                self.store8(base, off, disp, value);
                Ok(value)
            }
            _ => self.reject("array access of an unsupported element type"),
        }
    }

    /// The address of the row of the table in lane `lane`.
    fn row_address(&mut self, lane: Val) -> Res<Val> {
        if let Some((_, row)) = self.s.rows.iter().find(|(held, _)| *held == lane) {
            return Ok(*row);
        }
        if self.rows == NONE {
            return self.reject("row access without the rows pointer");
        }
        let stride = self.iconst(self.row_bytes);
        let bytes = self.f.bin(Op::Mul, lane, stride);
        let row = self.f.bin(Op::Add, self.rows, bytes);
        self.s.rows.push((lane, row));
        Ok(row)
    }

    /// The address of field `field` in the row of `table`, with a guard that
    /// leaves through `guard` when the row has none.
    fn field_slot(&mut self, table: ValueId, field: u32, guard: GuardId) -> Res<Val> {
        let lane = self.get(table)?;
        let attached = self.members[self.member]
            .attached
            .iter()
            .any(|(attached, _)| *attached == guard.index());
        if !attached
            && let Some((_, _, slot)) = self
                .s
                .fields
                .iter()
                .find(|(held, held_field, _)| *held == lane && *held_field == field)
        {
            return Ok(*slot);
        }
        let row = self.row_address(lane)?;
        // The other fields of the table that the straight-line code after
        // this access reads or writes get the same guard in one check, as
        // Cranelift does: the guard leaves at this access's snapshot.
        let group = if !attached && self.row_masked {
            self.slot_group(table, lane, field)
        } else {
            Vec::new()
        };
        let word = mem::size_of::<usize>();
        let first = self.load_at(Ty::I64, row, Off::K(0), offset(field as usize * word));
        let zero = self.iconst(0);
        if group.is_empty() {
            self.guard_cc(Cc::Ne, first, zero, guard)?;
            self.s.fields.push((lane, field, first));
            return Ok(first);
        }
        let mut slots = mem::take(&mut self.s.list);
        slots.clear();
        slots.push(first);
        for held in &group {
            slots.push(self.load_at(Ty::I64, row, Off::K(0), offset(*held as usize * word)));
        }
        if group.iter().all(|held| *held < 64) {
            // The row's presence mask (its last word) holds one bit per
            // word, so one compare checks every field.
            let needed = group
                .iter()
                .fold(1u64 << field, |bits, held| bits | 1 << held);
            let wanted = self.iconst(needed.cast_signed());
            let stride = self.row_bytes / 8;
            let mask = self.load_at(
                Ty::I64,
                row,
                Off::K(0),
                offset((stride as usize - 1) * word),
            );
            let have = self.f.bin(Op::And, mask, wanted);
            self.guard_cc(Cc::Eq, have, wanted, guard)?;
        } else {
            let one = self.iconst(1);
            let mut present = self.f.select(Cc::Ne, slots[0], zero, one, zero);
            for slot in &slots[1..] {
                let nonzero = self.f.select(Cc::Ne, *slot, zero, one, zero);
                present = self.f.bin(Op::And, present, nonzero);
            }
            self.guard_cc(Cc::Ne, present, zero, guard)?;
        }
        self.s.fields.push((lane, field, slots[0]));
        for (held, slot) in group.iter().zip(&slots[1..]) {
            self.s.fields.push((lane, *held, *slot));
        }
        let first = slots[0];
        self.s.list = slots;
        Ok(first)
    }

    /// The fields of `table`, other than `field`, that the straight-line ops
    /// after the one being lowered read or write before anything that could
    /// move or retype a slot, and whose slots are not held yet (the same
    /// group Cranelift checks at the first access, `slot_group`). An attached
    /// guard ends the group, and a store to the table does too.
    fn slot_group(&self, table: ValueId, lane: Val, field: u32) -> Vec<u32> {
        let ir = self.ir;
        let ahead = if self.at < ir.preheader.len() {
            &ir.preheader[self.at + 1..]
        } else if self.at == usize::MAX {
            &[]
        } else {
            let from = self.at - ir.preheader.len() + 1;
            ir.body.get(from..).unwrap_or(&[])
        };
        let member = &self.members[self.member];
        let blocked = |guard: GuardId| {
            member
                .attached
                .iter()
                .any(|(attached, _)| *attached == guard.index())
        };
        let mut group: Vec<u32> = Vec::new();
        for instruction in ahead {
            match &instruction.op {
                TraceOp::TableFieldLoad {
                    field: held,
                    table: on,
                    guard,
                }
                | TraceOp::TableFieldStore {
                    field: held,
                    table: on,
                    guard,
                    ..
                } => {
                    if blocked(*guard) {
                        break;
                    }
                    let cached = self
                        .s
                        .fields
                        .iter()
                        .any(|(l, f, _)| *l == lane && f == held);
                    if *on == table && *held != field && !cached && !group.contains(held) {
                        group.push(*held);
                    }
                    if *on == table && matches!(instruction.op, TraceOp::TableFieldStore { .. }) {
                        break;
                    }
                }
                op => {
                    if !keeps_slots(op) && !matches!(op, TraceOp::MetaGuard { .. }) {
                        break;
                    }
                    if op.guard().is_some_and(blocked) {
                        break;
                    }
                }
            }
        }
        group
    }
}

#[derive(Clone, Copy)]
enum Cond {
    Cmp(Cmp),
    Val(Val),
}

/// The reason for an operation the backend does not lower.
fn unsupported(op: &TraceOp) -> &'static str {
    match op {
        TraceOp::ToWide { .. } => "ToWide",
        TraceOp::NativeBinary { .. } => "NativeBinary call",
        TraceOp::ArrayLoad { .. } => "ArrayLoad",
        TraceOp::NumberLoad { .. } => "NumberLoad",
        TraceOp::NumberTag { .. } => "NumberTag",
        TraceOp::ArraySet { .. } => "ArraySet",
        TraceOp::HashLoad { .. } => "HashLoad",
        TraceOp::HostCall { .. } => "HostCall",
        TraceOp::CallHelper { .. } => "CallHelper",
        TraceOp::HelperResult { .. } => "HelperResult",
        TraceOp::ReleaseStrings { .. } => "ReleaseStrings",
        TraceOp::SelfCall { .. } => "SelfCall",
        TraceOp::Read { .. } | TraceOp::Effect { .. } => "Read or Effect",
        _ => "operation",
    }
}
