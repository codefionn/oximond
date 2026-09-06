//! Trace IR: a recorded path through guest code.
//!
//! A trace is straight-line SSA with typed values, guards that leave through
//! a [`Snapshot`], and a terminator that closes a loop or exits. The IR is
//! owned plain data (`Send + 'static`), so a worker thread can verify and
//! compile it without touching a live VM.
//!
//! - [`ir`] holds ids, types, operations, [`TraceIr`] and [`TraceBuilder`].
//!   A trace can inline calls of pinned closures: every call is a logical
//!   frame ([`InlinedFrame`]), and a snapshot inside one lists the frames
//!   from the root down to it.
//!   Table access is part of it: an op reads or overwrites a slot of storage
//!   the VM resolved at entry ([`TableBinding`]), an array op carries a guard
//!   that leaves before it, and a store is a committed effect.
//! - [`snapshot`] holds the exact-recovery description each guard carries.
//! - [`register_map`] holds a snapshot frame's registers in runs that
//!   consecutive snapshots share.
//! - [`loop_build`] decides how registers cross a loop-back.
//! - [`semantics`] holds the numeric operations the evaluator, the optimizer's
//!   constant folder and the backend share.
//! - [`tree`] turns a root trace and its side traces into the members the
//!   backend lowers as one native function.
//! - [`optimize`] rewrites a verified trace into an equivalent, cheaper one.
//! - [`range`] finds the array accesses of a counted loop whose keys the
//!   backend can check once, where the loop starts.
//! - [`verify`] checks dominance, operand types, snapshot completeness,
//!   effect ordering and loop arity.

#![allow(
    clippy::too_many_lines,
    clippy::match_same_arms,
    clippy::trivially_copy_pass_by_ref,
    clippy::missing_panics_doc,
    reason = "id conversions panic only past u32 table sizes"
)]

mod helper;
mod ir;
mod loop_build;
pub mod opportunity;
mod optimize;
pub mod range;
mod register_map;
mod semantics;
mod snapshot;
mod tree;
mod verify;

#[cfg(test)]
mod borrow_tests;
#[cfg(test)]
mod hash_tests;
#[cfg(test)]
pub mod reference;
#[cfg(test)]
mod string_tests;
#[cfg(test)]
mod table_rows_tests;
#[cfg(test)]
mod table_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tree_tests;
#[cfg(test)]
mod virtual_tests;

pub use helper::{
    BoxedValue, HELPER_COUNT, HELPERS, HelperEffects, HelperId, HelperInfo, HelperType,
    MAX_FIND_CAPTURES, MAX_HELPER_ARGS, MAX_HELPER_RESULTS, MAX_NEW_TABLE_KEYS, MAX_PACK_VALUES,
    MAX_PREFILLED_ARRAY, MAX_RESUME_ARGS, MAX_RESUME_VALUES, MAX_UNPACK_VALUES,
    MAX_UTF8_CODEPOINTS, NEXT_BOX_ROW, NEXT_KINDS, NEXT_ROW, NextKind, RESUME_SHAPES,
    TABLE_NEW_PREFILL,
};
pub use ir::{
    ArithmeticOp, BindingKind, BoxedSteps, CellAlias, CellSource, EffectId, Entry, EntryContract,
    EntryInput, ExternalSlot, FrameId, FrameInput, FunctionTrace, GuardId, GuardInfo, HostFunction,
    InlinedFrame, LoopParam, LoopParamInit, MAX_HOST_ARGS, MAX_INDEX_HOPS, MAX_TRACE_FRAME_DEPTH,
    META_BARE_NUMBERS, META_BARE_SIMPLE, META_BARE_STRINGS, META_FIELD_KEY, META_PIN_ABSENT,
    MetaPin, NativeBinary, NativeUnary, NumericOp, PINNED_BUILTIN_COROUTINE_RESUME,
    PINNED_BUILTIN_COROUTINE_YIELD, PINNED_BUILTIN_DEBUG_TRACEBACK, PINNED_BUILTIN_IPAIRS,
    PINNED_BUILTIN_IPAIRS_ITERATOR, PINNED_BUILTIN_NEXT, PINNED_BUILTIN_PAIRS,
    PINNED_BUILTIN_PCALL, PINNED_BUILTIN_STRING_FORMAT, PINNED_BUILTIN_STRING_GSUB,
    PINNED_BUILTIN_TOSTRING, PINNED_BUILTIN_XPCALL, Protected, ROOT_STRING_LANE_BASE,
    RootActivation, SnapshotId, TABLE_ADDRESS_KEY, TERMINAL_GUARD, TableBinding, TableField,
    TailSite, TraceBuilder, TraceFrameInfo, TraceId, TraceInstruction, TraceIr, TraceOp,
    TraceScalar, TraceSide, TraceSite, TraceTerminator, TraceType, ValueDef, ValueId, ValueInfo,
    WideConversion, find_uses_prepared_pattern, gmatch_constant_lane, meta_word,
    pinned_builtin_identity, pinned_builtin_kind, pinned_native_identity, pinned_native_index,
    root_string_slot,
};
pub use loop_build::{LoopRole, RegisterUse, loop_types_are_stable};
pub use optimize::{
    MemberWrites, OptimizeLevel, OptimizeOptions, OptimizeStats, TreeWrites, optimize_trace,
    optimize_tree_member, optimize_with, optimize_with_tree, pass_through_params,
};
pub use range::{ProvenKey, RangeCheck, RangeInduction, RangePlan, RangeShift, RangeStep};
pub use register_map::{RUN as REGISTER_RUN, RegisterMap, RunMemo};
pub use semantics::{
    arithmetic, compare, float_floor_divide, float_modulo, float_power, float_power_square,
    integer_float_equal, integer_float_order, integer_floor_divide, integer_floor_modulo,
    integer_shift, integer_to_float, negate, number_bits, to_float, to_wide, trick_bits,
    wide_arithmetic, wide_compare, wide_to_float,
};
pub use snapshot::{
    MAX_VIRTUAL_SHAPE_KEYS, NumberRegister, RegisterEntry, ResumePoint, SliceId, SliceRecipe,
    Snapshot, SnapshotFrame, SnapshotPhase, VirtualAlloc, VirtualId, VirtualKey, VirtualTable,
};
pub use tree::{
    LinkWrite, TreeError, TreeMember, apart_link_writes, plan_tree, plan_tree_with, tree_bindings,
    untyped_link_writes,
};
pub(crate) use tree::{frame_writes, guard_writes, start_writes};
pub use verify::{TraceVerifyError, verify_grown_tree, verify_side_fit, verify_trace};
