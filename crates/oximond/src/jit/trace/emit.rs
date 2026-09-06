//! Cranelift lowering of trace IR.
//!
//! The generated function has this shape:
//!
//! ```text
//! entry(inputs, control, exit_area):
//!     fuel = control.fuel
//!     for each input: if tag != required -> mismatch stub (returns -(i+1))
//!     load typed bits
//!     jump header(initial loop values..., fuel)
//! header(loop values..., fuel):
//!     body: straight-line SSA, each guard branches to its own cold stub
//!     loop-back: fuel -= 1; if fuel > 0 jump header(next values..., fuel)
//!                else -> poll stub
//! stub: store the snapshot's values into the exit area, store fuel,
//!       return the exit descriptor's index
//! ```
//!
//! A trace with side traces is one function. A guard that has a side does not
//! get a stub: its failing edge jumps into the side's blocks, in the same
//! function, with the guard snapshot's values as the side's entry values. A
//! side ends in the same loop-back as the root (it jumps to the root's header
//! with one value per loop parameter and spends one unit of fuel), in its own
//! poll stub, or in an exit stub of its own. Sides nest: a guard of a side can
//! have a side. Nothing in the compiled code is patched and nothing leaves the
//! function except through an exit stub.
//!
//! A side can also be compiled apart, as a function of its own with the same
//! signature (`lower` with a `DetachedSpec`). The exit of the guard it
//! continues returns to the caller of the tree as always, with the snapshot's
//! values in the exit area, and the run calls the side with the same buffers
//! if a cell says there is one (see `CompiledTrace::attach`). The side reads
//! its entry values from the exit area where that exit stored them, the root's
//! entry inputs from the input slots, and returns the code of an exit of its
//! own, `LINK_CODE` after leaving the loop parameters in the first slots of
//! the exit area, or `DECLINE_CODE` if a register it reads from an extra input
//! slot holds another type. The extra input slots follow the entry inputs and
//! the bindings. They hold the registers of the root frame the entry inputs
//! leave out, the registers of the caller frame a side that returns into it
//! reads, and the globals and upvalues such a side reads and never writes
//! (`ExtraSource`). The registers a linking side writes that are not loop
//! parameters it stores itself into the input slot of the entry input or
//! extra input they belong to before it returns `LINK_CODE`
//! (`Lowerer::store_link_writes`), and the run copies those slots to the end
//! of the exit area when it ends (`CompiledTrace::writebacks`). The tree's own
//! code is the same with or without such sides.
//!
//! Loop-carried values are block parameters, so the loop-back is a parallel
//! assignment by construction: a swap needs no temporary. Floats are `f64`,
//! integers `i64`, booleans and nil `i8`. Loop-invariant work the optimizer
//! moved into the preheader runs in the entry block, before the first jump.
//! `ArithmeticOp::Modulo` is the unfused `a - floor(a / b) * b` of Lua 5.1,
//! 5.2 and `LuaJIT`, lowered inline, and power calls the shared `float_power`.
//! `PowerSquare` (Lua 5.4 and 5.5) is `a * a` inline for an exponent of 2.
//! `ModuloWith` of Lua 5.3 and later is, on a target with a one-instruction
//! `fma`, a `trunc`/`fma` remainder checked for exactness inline, and on other
//! targets an exact inline integer remainder when both operands are integral
//! and below 2^53. Operands those paths decline call the shared
//! `oximond::compat` function. All of them are exactly what the
//! interpreter and the optimizer's constant folder compute.
//! Integer floor division, modulo and shifts are branch-free sequences that
//! never trap, whatever the operands: `sdiv` and `srem` trap on a zero divisor
//! and on `MIN / -1`, so those divisors are replaced before the division. The
//! recorder guards a zero divisor before the operation, so a zero here only
//! happens for an operation the optimizer moved ahead of its guard, and the
//! result is then unused. Comparing an integer with a float calls a shared
//! helper, because exact comparison needs more than a conversion.
//!
//! This module owns the executable-memory lifetime and the one unsafe call
//! that enters generated code.

use super::compile::{
    ExitPlan, Extra, ExtraSource, LoweredMember, TraceCompileError, TraceCompileOptions, Writeback,
    link_slots_of, trace_call_sides,
};
use super::exit::{
    CLOSURE_ROW_CELLS, ExitDescriptor, ExitSlot, PROBE_NO_TOMBSTONES, TRACE_PENDING_STRINGS,
    TRACE_TAG_BOOLEAN, TRACE_TAG_BORROWED_STRING, TRACE_TAG_CLOSURE, TRACE_TAG_FLOAT,
    TRACE_TAG_INTEGER, TRACE_TAG_NIL, TRACE_TAG_STRING, TRACE_TAG_TABLE, TableDispatch,
    TraceControl, TraceInputSlot, entry_tag, row_stride_of,
};
use super::prepared::Prepared;
use crate::compat::{FloatModulo, TracingOptFlags};
use crate::jit::OptimizationTier;
use crate::jit_ir::Comparison;
use crate::jit_ir::trace::range::{
    self, FLOAT_EXACT_LIMIT, FLOAT_ROUND_MAGIC, MAX_VERSIONED_INSTRUCTIONS, versioning_pays,
};
use crate::jit_ir::trace::{
    ArithmeticOp, BindingKind, BoxedValue, ExternalSlot, GuardId, HelperEffects, HelperId,
    HelperType, HostFunction, LinkWrite, LoopParamInit, MAX_HOST_ARGS, META_FIELD_KEY, NextKind,
    NumericOp, ROOT_STRING_LANE_BASE, RangePlan, RangeShift, RangeStep, RegisterEntry, Snapshot,
    TABLE_NEW_PREFILL, TableBinding, TraceInstruction, TraceIr, TraceOp, TraceScalar,
    TraceTerminator, TraceType, ValueDef, ValueId, WideConversion, find_uses_prepared_pattern,
    gmatch_constant_lane, meta_word, pass_through_params, root_string_slot,
};
use crate::runtime::{
    NATIVE_BOOLEAN_GROWTH_REACH, NATIVE_PROBE_SHIFT, NATIVE_PROBE_SLOT_HASH, NATIVE_PROBE_SLOT_KEY,
    NATIVE_PROBE_SLOT_SIZE, NATIVE_PROBE_SLOT_VALUE, NATIVE_STRING_HASH_OFFSET,
    NATIVE_STRING_INLINE_TAG, NATIVE_STRING_INLINE_WORDS, NATIVE_STRING_STORAGE_OFFSET,
    NATIVE_VALUE_PAYLOAD_OFFSET, NATIVE_VALUE_SIZE, NATIVE_VALUE_TAG_BOOLEAN,
    NATIVE_VALUE_TAG_CLOSURE, NATIVE_VALUE_TAG_INTEGER, NATIVE_VALUE_TAG_NIL,
    NATIVE_VALUE_TAG_NUMBER, NATIVE_VALUE_TAG_STRING, NATIVE_VALUE_TAG_TABLE,
    NATIVE_VALUE_TAG_THREAD, NativeTableHeader, NativeTableLayout,
};
use cranelift_codegen::ir::{
    AbiParam, Block, BlockArg, FuncRef, InstBuilder, MemFlags, SigRef, Signature, StackSlotData,
    StackSlotKind, UserFuncName, Value,
    condcodes::{FloatCC, IntCC},
    types,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, default_libcall_names};
use std::mem;
#[cfg(feature = "op-stats")]
use {crate::jit_ir::trace::opportunity::RangeFacts, crate::runtime::op_stats};

/// `2^52`. Adding it to a float in `[1, 2^52)` leaves the float's integer
/// part in the low bits of the sum's representation.
const FLOAT_INTEGER_MAGIC: f64 = 4_503_599_627_370_496.0;

/// What native code does for the call of a boxed `next` helper before it
/// calls it (see `Lowerer::call_helper_stepping`).
#[derive(Clone, Copy)]
pub(super) struct BoxStep {
    /// The binding of the table's array part, when native code steps along
    /// it.
    pub(super) array: Option<u32>,
    /// Whether native code steps along the hash part's snapshot.
    pub(super) hash: bool,
    /// Whether the run keeps the description of that snapshot in a lane.
    pub(super) cursor_lane: bool,
    /// Whether the profile's keys of the array part are integers.
    pub(super) integer: bool,
    /// The kind of the key the call gives, if it wants it.
    pub(super) key: Option<NextKind>,
    /// What the call does with the entry's value.
    pub(super) value: BoxedValue,
}

const POW_SYMBOL: &str = "oximond_trace_pow";
/// Out-of-line `%` of Lua 5.3 and later for the operands the inline integral
/// path does not take. Declared only in a trace that holds such a modulo, so
/// other traces compile exactly as before.
const FMOD53_SYMBOL: &str = "oximond_trace_mod_fmod53";
const FMOD54_SYMBOL: &str = "oximond_trace_mod_fmod54";
const ADMIT_TABLE_SYMBOL: &str = "oximond_trace_admit_table";
const ENSURE_ROW_SYMBOL: &str = "oximond_trace_ensure_row";
const RESERVE_ARRAY_SYMBOL: &str = "oximond_trace_array_reserve";
const TABLE_SET_STUB: &str = "oximond_trace_table_set_stub";
const ADMIT_TABLE_STUB: &str = "oximond_trace_admit_table_stub";
const HASH_GET_SYMBOL: &str = "oximond_trace_hash_get";
const HASH_GET_STUB: &str = "oximond_trace_hash_get_stub";
const RESERVE_ARRAY_STUB: &str = "oximond_trace_array_reserve_stub";
/// Prefix of the register-preserving stubs of cold helper calls, one per
/// argument count (see [`helper_stub_arity`]).
const HELPER_STUB_PREFIX: &str = "oximond_trace_helper_stub_";
/// A loop that makes strings releases the ones it no longer names once they
/// weigh this much (see `TRACE_PENDING_STRINGS`).
pub(super) const RELEASE_WEIGHT: i64 = 256;

type TraceEntry = unsafe extern "C" fn(*mut TraceInputSlot, *mut TraceControl, *mut i64) -> i64;

/// What the entry function of a function trace returns when the activation
/// the VM entered returned (see [`super::TraceOutcome::Returned`]). Exit
/// indices are not negative and entry mismatches are small negatives.
pub(super) const RETURNED_CODE: i64 = i64::MIN;

/// What a separately compiled side returns to the run that called it when it
/// links back to the root's loop: the values of the root's loop parameters are
/// in the first slots of the exit area, in order, and the fuel left in the
/// control block. The run enters the tree again with them
/// ([`super::compile::CompiledTrace::run`]).
pub(super) const LINK_CODE: i64 = i64::MIN + 1;

/// What a separately compiled side returns when it declines to run, because a
/// register it was recorded for holds another type: nothing ran, and the exit
/// that called it is the one the run left through.
pub(super) const DECLINE_CODE: i64 = i64::MIN + 2;

/// Exits of a separately compiled side return `unit << SIDE_CODE_SHIFT |
/// local index`, which no exit of the root's own code can (their indexes
/// are small). See [`super::compile::CompiledTrace::attach`].
pub(super) const SIDE_CODE_SHIFT: u32 = 24;

/// What a side compiled apart from its tree needs to know about the code that
/// calls it.
pub(super) struct DetachedSpec<'a> {
    /// Added to the index of each exit of the side to make the code it
    /// returns: `unit << SIDE_CODE_SHIFT`.
    pub(super) code_base: i64,
    /// The exit of the host that calls the side, and the snapshot it
    /// restores: the side's entry values.
    pub(super) host: &'a ExitDescriptor,
    pub(super) host_snapshot: &'a Snapshot,
    /// The root's entry inputs, by position in the input slots.
    pub(super) root_entry: &'a [(u16, TraceType)],
    /// Whether the side leaves through its poll exit where it would link
    /// back to the root's loop.
    pub(super) exit_style: bool,
    /// Whether the root hashes with the string index its tables publish
    /// (see `Lowerer::native_probe`).
    pub(super) native_probe: bool,
    /// Whether the tree's rows end in a presence mask (see
    /// [`rows_masked`]).
    pub(super) row_masked: bool,
    /// The registers the tree's run already gives input slots after the
    /// entry inputs (see [`super::compile::CompiledTrace::extras`]).
    pub(super) extras: &'a [Extra],
    /// The registers the side writes that are live where it links back and
    /// that the root does not carry as loop parameters: the side stores
    /// them into the run's input slots, and the VM writes the slots into the
    /// frames when the run leaves.
    pub(super) link_writes: &'a [LinkWrite],
    /// Registers of the root's frame: the ones before its window of cells
    /// and globals.
    pub(super) guest_registers: usize,
}

/// The status an activation of a function trace gives for exit `index`:
/// `-1 - index`. An activation that returned gives the fuel left instead,
/// which is positive (an activation runs only with fuel), so one register
/// carries both and an integer result still fits the return registers.
const fn exit_status(index: i64) -> i64 {
    -1 - index
}

/// What the lowering of a function trace needs beyond a loop trace's.
///
/// An activation is a function `(control, left, fuel, parameters...) ->
/// (status, result)`. `left` counts the activations the run may still
/// nest below this one, and `fuel` the activations it may still start; both
/// travel in registers. The input slots and the exit area are the control
/// block's (`inputs`, `area`): an activation reads a cell or global where it
/// uses one, and finds its part of the exit area (at depth
/// `control.depth_cap - left`) only when it leaves. A return gives the fuel
/// left as its status, which is positive. An exit stores the fuel and the
/// activation's depth in the control block and gives the status of its index
/// ([`exit_status`]), which every activation around it passes on.
struct FunctionLowering {
    /// The activation function itself, which a self call calls.
    body: FuncRef,
    /// Bytes of exit area each activation owns.
    stride_bytes: i64,
    /// Offset of the last slot of an activation's area, which holds the
    /// index of the exit it left through.
    header_offset: i32,
    /// The type of the function's result, as a Cranelift type.
    result: Option<types::Type>,
    /// The window register of the activation's tail call count, for a
    /// function that tail calls itself.
    count: Option<u16>,
    /// The block an activation starts at, which a tail call jumps back to:
    /// `(fuel, count?, parameters...)`.
    start: Option<Block>,
}

/// A finalized function and the module that owns its executable memory.
pub(super) struct TraceCode {
    /// Taken only by `Drop`, which hands the memory back to the system.
    module: Option<JITModule>,
    /// For code the MIR backend made: its piece of the process's executable
    /// region, which goes back to the region when the code is dropped.
    #[allow(
        dead_code,
        reason = "held so its drop gives the code back to the region"
    )]
    mir: Option<super::mir::Block>,
    entry: TraceEntry,
    code_size: usize,
    /// Instructions of the function as handed to Cranelift.
    clif_insts: usize,
    /// For a side compiled apart: the tree's extra inputs it needs.
    side_extras: Vec<Extra>,
    /// For a side compiled apart: the input slots it rewrites where it links
    /// back to the loop.
    side_writebacks: Vec<Writeback>,
    /// The objects the code's specialized helper calls name by address (see
    /// [`Lowerer::prepared_pattern`]). They live as long as the code can run.
    prepared: Vec<Prepared>,
    /// Whether the root's loop runs in the fast version its range plan makes
    /// (see [`Lowerer::lower_versions`]).
    #[cfg(test)]
    pub(super) versioned: bool,
    /// Whether the root's exits call the sides compiled apart from native
    /// code (see [`Lowerer::native_link_of_root`]).
    pub(super) native_link: bool,
    /// Call instructions in the lowered function, for tests that check an
    /// operation is inline.
    #[cfg(test)]
    pub(super) calls: usize,
    /// `fma` instructions in the lowered function.
    #[cfg(test)]
    pub(super) fmas: usize,
}

impl Drop for TraceCode {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        if let Some(module) = self.module.take() {
            // SAFETY: trace code never calls back into Lua and the VM runs a
            // trace to its exit before it retires, promotes or flushes one,
            // so no frame is executing this module's code. `entry` dies with
            // `self` and is never called again. Without this, the system
            // memory provider keeps finalized pages forever.
            unsafe { module.free_memory() };
        }
    }
}

/// The entry of the placeholder code: it mismatches at once, running nothing.
#[allow(unsafe_code)]
unsafe extern "C" fn staged_entry(
    _inputs: *mut TraceInputSlot,
    _control: *mut TraceControl,
    _exit_area: *mut i64,
) -> i64 {
    -1
}

impl TraceCode {
    /// Code that holds no function: the placeholder of a trace whose tree is
    /// still being recorded (see [`super::CompiledTrace::staged`]).
    pub(super) fn staged() -> Self {
        Self {
            module: None,
            mir: None,
            entry: staged_entry,
            code_size: 0,
            clif_insts: 0,
            side_extras: Vec::new(),
            side_writebacks: Vec::new(),
            prepared: Vec::new(),
            #[cfg(test)]
            versioned: false,
            native_link: false,
            #[cfg(test)]
            calls: 0,
            #[cfg(test)]
            fmas: 0,
        }
    }

    /// Code the MIR backend made (see [`super::mir`]).
    #[allow(unsafe_code)]
    fn from_mir(compiled: super::mir::Compiled) -> Self {
        // SAFETY: the address is the start of a block whose code is a
        // function of exactly the `TraceEntry` signature, which
        // `oximond_mir` emitted for the entry contract above. The block is
        // owned by the `TraceCode`, so the code outlives every use of the
        // pointer, and is read-only once written.
        // A function trace's entry follows its activation in the block.
        let entry = unsafe {
            mem::transmute::<usize, TraceEntry>(compiled.block.address() + compiled.entry)
        };
        Self {
            module: None,
            mir: Some(compiled.block),
            entry,
            code_size: compiled.code_size,
            clif_insts: 0,
            side_extras: Vec::new(),
            side_writebacks: Vec::new(),
            prepared: Vec::new(),
            #[cfg(test)]
            versioned: compiled.versioned,
            native_link: false,
            #[cfg(test)]
            calls: 0,
            #[cfg(test)]
            fmas: 0,
        }
    }

    /// The backend that made the code, for the measurement lines.
    pub(super) fn backend(&self) -> &'static str {
        if self.mir.is_some() {
            "mir"
        } else {
            "cranelift"
        }
    }

    /// Where the code is, if the MIR backend made it.
    #[cfg(test)]
    pub(super) fn mir_address(&self) -> Option<usize> {
        self.mir.as_ref().map(super::mir::Block::address)
    }

    pub(super) const fn code_size(&self) -> usize {
        self.code_size
    }

    /// The single-run patterns the code scans itself (see
    /// [`Lowerer::call_inline_run`]).
    pub(super) fn inline_runs(&self) -> usize {
        self.prepared
            .iter()
            .filter(|prepared| matches!(prepared, Prepared::RunClass(_)))
            .count()
    }

    /// Instructions of the function handed to Cranelift.
    pub(super) const fn clif_insts(&self) -> usize {
        self.clif_insts
    }

    /// For a side compiled apart: the tree's extra inputs after it, which
    /// start with the ones the tree had.
    pub(super) fn side_extras(&self) -> &[Extra] {
        &self.side_extras
    }

    /// For a side compiled apart: the input slots it rewrites where it links
    /// back to the loop, with the registers they stand for.
    pub(super) fn side_writebacks(&self) -> &[Writeback] {
        &self.side_writebacks
    }

    /// The address of the entry function, which an exit's cell holds to call
    /// this code as a side.
    pub(super) fn entry_address(&self) -> usize {
        self.entry as usize
    }

    /// Runs the code over `inputs`, the entry input slots followed by one
    /// slot per binding. The code writes binding slots only through a table
    /// set ([`TraceOp::ArraySet`]), which only a trace with bindings holds,
    /// and for those the caller passes a buffer it owns for the call.
    #[allow(unsafe_code)]
    pub(super) fn run(
        &self,
        inputs: *mut TraceInputSlot,
        control: &mut TraceControl,
        exit_area: &mut [i64],
    ) -> i64 {
        // SAFETY: `entry` is created only by `lower`, from a finalized
        // function with exactly the `TraceEntry` signature, and the retained
        // module owns the executable memory. `CompiledTrace::run` checked that
        // `inputs` and `exit_area` cover every slot the generated code
        // reads or writes. The code retains none of the pointers.
        unsafe { (self.entry)(inputs, control, exit_area.as_mut_ptr()) }
    }
}

/// Calls the side compiled apart whose entry is at `address`, a value
/// [`TraceCode::entry_address`] gave, for code its tree still owns, with the
/// buffers of the run that left the tree through the exit whose cell held it.
/// Only [`super::compile::CompiledTrace`] calls it, from a cell it published.
#[allow(unsafe_code)]
pub(super) fn call_side(
    address: usize,
    inputs: *mut TraceInputSlot,
    control: &mut TraceControl,
    exit_area: *mut i64,
) -> i64 {
    // SAFETY: the address is that of a finalized function with exactly the
    // `TraceEntry` signature, which the tree owning the cell keeps alive for
    // as long as the run (which borrows the tree) lasts. The exit area is
    // the run's, which covers the words the side reads and writes.
    unsafe {
        let entry = mem::transmute::<usize, TraceEntry>(address);
        entry(inputs, control, exit_area)
    }
}

/// The first `count` input slots at `inputs`, copied.
#[allow(unsafe_code)]
pub(super) fn copy_inputs(inputs: *const TraceInputSlot, count: usize) -> Vec<TraceInputSlot> {
    // SAFETY: the run's slots cover the entry inputs and the extra inputs
    // (`CompiledTrace::run_checked` asserted it), and are not written while
    // this reads them.
    unsafe { std::slice::from_raw_parts(inputs, count) }.to_vec()
}

/// Reads input slot `position` at `inputs`.
#[allow(unsafe_code)]
pub(super) fn read_input(inputs: *const TraceInputSlot, position: usize) -> TraceInputSlot {
    // SAFETY: `position` is a slot of the run's buffer, which covers the
    // entry inputs, the bindings and the extras (`run_checked` asserted it),
    // and nothing writes it while this reads.
    unsafe { inputs.add(position).read() }
}

/// Writes `slot` to input slot `position` at `inputs`, a buffer the run owns.
#[allow(unsafe_code)]
pub(super) fn write_input(inputs: *mut TraceInputSlot, position: usize, slot: TraceInputSlot) {
    // SAFETY: `position` is an entry input of the tree, which the run's
    // buffer covers, and the caller owns the buffer (it is a copy, or the
    // buffer `run_checked` made).
    unsafe { inputs.add(position).write(slot) };
}

pub(super) extern "C" fn trace_pow(left: f64, right: f64) -> f64 {
    crate::jit_ir::trace::float_power(left, right)
}

pub(super) extern "C" fn trace_mod_fmod53(left: f64, right: f64) -> f64 {
    crate::compat::fmod53_helper(left, right)
}

pub(super) extern "C" fn trace_mod_fmod54(left: f64, right: f64) -> f64 {
    crate::compat::fmod54_helper(left, right)
}

/// `bit` library operand conversion of a float, for [`TraceOp::ToWide`].
pub(super) extern "C" fn trace_number_bits(number: f64) -> i64 {
    i64::from(crate::jit_ir::trace::number_bits(number))
}

/// Gives the table in the array slot at `slot` its lane, or returns -1. Native
/// code calls it with the control block's dispatcher and the address of a slot
/// it has just checked to be inside an array and tagged as a table.
#[allow(unsafe_code)]
pub(super) extern "C" fn trace_admit_table(
    dispatch: *mut TableDispatch<'static>,
    slot: *const crate::runtime::Value,
) -> i64 {
    // SAFETY: the dispatcher is `TraceControl::tables`, which
    // `CompiledTrace::run_with_tables` sets to a live dispatcher for exactly
    // the duration of the run and never shares with another thread. Code that
    // calls this reads it from the control block, and code that needs tables
    // refuses to run without. `slot` is an element of an array part the run's
    // bindings hold, in bounds (the guard before the call checked it) and
    // holding an initialized `Value`, which nothing in the run replaces.
    let (met, slot) = unsafe { (&*dispatch, &*slot) };
    if let Some(lane) = met.admit_met(slot) {
        return lane;
    }
    trace_admit_unmet(dispatch, slot)
}

/// [`trace_admit_table`] for a value the closure memo does not answer. It
/// takes the caller's arguments in the caller's order, so the caller leaves
/// for it with a jump, and the memo's answer saves no register.
#[allow(unsafe_code)]
#[inline(never)]
extern "C" fn trace_admit_unmet(
    dispatch: *mut TableDispatch<'static>,
    slot: *const crate::runtime::Value,
) -> i64 {
    // SAFETY: as `trace_admit_table`.
    let (dispatch, slot) = unsafe { (&mut *dispatch, &*slot) };
    dispatch.admit(slot)
}

/// Fills the row of the table at lane `lane` unless it is filled, for a table
/// a helper has just given a lane (`table.remove` of a table): a field of
/// such a table is read through its row.
#[allow(unsafe_code)]
pub(super) extern "C" fn trace_ensure_row(dispatch: *mut TableDispatch<'static>, lane: i64) {
    // SAFETY: as `trace_admit_table`.
    let dispatch = unsafe { &mut *dispatch };
    if let Ok(lane) = usize::try_from(lane) {
        dispatch.ensure_row(lane);
    } else if let Ok(lane) = usize::try_from(!lane) {
        // A negative lane (its complement) asks for the row to be made again:
        // `setmetatable` changed the table's metatable word.
        dispatch.refresh_row(lane);
    }
}

/// Builds the virtual tables of exit `exit` for a guard that enters a side
/// (see [`TableDispatch::materialize_exit`]): 0 with the lanes in the control
/// block's `helper_out`, or 1 when the guard must leave through its exit.
#[allow(unsafe_code)]
pub(super) extern "C" fn trace_materialize_exit(
    dispatch: *mut TableDispatch<'static>,
    control: *mut TraceControl,
    area: *const i64,
    exit: i64,
) -> i64 {
    // SAFETY: as `trace_admit_table` for the dispatcher. `control` is the
    // run's control block, which native code holds and nothing else reads
    // during the call; the dispatcher is a separate object.
    let (dispatch, control) = unsafe { (&mut *dispatch, &mut *control) };
    let made =
        usize::try_from(exit).is_ok_and(|exit| dispatch.materialize_exit(control, area, exit));
    i64::from(!made)
}

/// Sets a key of the table bound as `binding` raw, for a
/// [`TraceOp::ArraySet`] native code could not do in place, and rewrites the
/// array words of every array binding in `words`, the run's binding slots.
/// Returns 0, or -1 when it stored nothing and the guard must leave. `key`
/// and `value` arrive as entry tag and bits (see [`TraceInputSlot`]).
#[allow(unsafe_code)]
pub(super) extern "C" fn trace_table_set(
    dispatch: *mut TableDispatch<'static>,
    words: *mut TraceInputSlot,
    binding: i64,
    key_tag: i64,
    key_bits: i64,
    value_tag: i64,
    value_bits: i64,
) -> i64 {
    // SAFETY: the dispatcher is `TraceControl::tables`, live for exactly the
    // run (see `trace_admit_table`). `words` is the run's own input buffer,
    // which `CompiledTrace::run_checked` made writable for the run and which
    // holds one slot per binding after the entry inputs; the code passes
    // the address of the first binding slot. Nothing else reads or writes
    // it during this call.
    let dispatch = unsafe { &mut *dispatch };
    let words = unsafe { std::slice::from_raw_parts_mut(words, dispatch.bindings) };
    let Ok(binding) = usize::try_from(binding) else {
        return -1;
    };
    // The key and value stay two words each until the runtime call builds
    // them: a `Value` built with narrow stores and copied whole stalls on
    // the store forwarding of every copy.
    let key = TraceInputSlot {
        tag: key_tag,
        bits: key_bits,
    };
    let value = TraceInputSlot {
        tag: value_tag,
        bits: value_bits,
    };
    if dispatch.set(binding, key, value, words) {
        0
    } else {
        -1
    }
}

/// Looks up a string in the table bound as `binding` for a
/// [`TraceOp::HashLoad`] and writes what it found, as an entry tag and bits, to
/// `out`. Read-only, nonallocating, noncollecting and non-reentrant (see
/// [`TraceTables::hash_get`](super::exit::TraceTables::hash_get)).
#[allow(unsafe_code)]
pub(super) extern "C" fn trace_hash_get(
    dispatch: *mut TableDispatch<'static>,
    out: *mut TraceInputSlot,
    binding: i64,
    key_tag: i64,
    key_bits: i64,
) {
    // SAFETY: the dispatcher is `TraceControl::tables`, live for exactly the
    // run (see `trace_admit_table`). `out` is `TraceControl::hash_result` of
    // the run's control block, which nothing else touches during the call.
    let (dispatch, out) = unsafe { (&mut *dispatch, &mut *out) };
    *out = match usize::try_from(binding) {
        Ok(binding) => dispatch.hash_get(binding, key_tag, key_bits),
        Err(_) => TraceInputSlot::other(),
    };
}

/// Calls a library function for a [`TraceOp::HostCall`]: `args` are the
/// control block's argument slots, `argc` of them, and the answer goes to
/// `out` as an entry tag and bits. When the VM cannot answer, `out` gets
/// [`TRACE_TAG_OTHER`] and the guard, which compares the tag with the one it
/// expects, leaves.
// `args`/`argc` are the native call ABI's terms for the argument slots and
// their count, the same names the VM's call shim uses.
#[allow(clippy::similar_names)]
#[allow(unsafe_code)]
pub(super) extern "C" fn trace_host_call(
    dispatch: *mut TableDispatch<'static>,
    args: *const TraceInputSlot,
    out: *mut TraceInputSlot,
    function: i64,
    argc: i64,
) {
    // SAFETY: the dispatcher is `TraceControl::tables`, live for exactly the
    // run (see `trace_admit_table`). `args` and `out` are the control
    // block's own `host_args` and `host_result`, which nothing else touches
    // during the call, and the code stored `argc` (at most `MAX_HOST_ARGS`)
    // arguments before it.
    let dispatch = unsafe { &mut *dispatch };
    let out = unsafe { &mut *out };
    let count = usize::try_from(argc).unwrap_or(0).min(MAX_HOST_ARGS);
    let args = unsafe { std::slice::from_raw_parts(args, count) };
    if !dispatch.host_call(function, args, out) {
        *out = TraceInputSlot::other();
    }
}

/// Defines a stub that calls `target` with its own arguments under a
/// convention that preserves every register: for the lookup of a number key
/// (see `hash_get_stub_id` in [`compile_trace`]) and the growth of an array.
fn define_forwarding_stub(
    module: &mut JITModule,
    id: cranelift_module::FuncId,
    signature: &Signature,
    target: cranelift_module::FuncId,
) -> Result<(), Box<cranelift_module::ModuleError>> {
    let mut context = module.make_context();
    let mut function_context = FunctionBuilderContext::new();
    context.func.signature = signature.clone();
    context.func.name = UserFuncName::user(0, id.as_u32());
    {
        let mut builder = FunctionBuilder::new(&mut context.func, &mut function_context);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        let params = builder.block_params(block).to_vec();
        let callee = module.declare_func_in_func(target, builder.func);
        builder.ins().call(callee, &params);
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }
    module.define_function(id, &mut context).map_err(Box::new)?;
    module.clear_context(&mut context);
    Ok(())
}

/// Defines the stub a trace calls for a table set: `(control, words, binding,
/// key tag, key bits, value tag, value bits)` under a convention that
/// preserves every register. It calls [`trace_table_set`] with the control
/// block's dispatcher and stores the result in [`TraceControl::set_status`].
fn define_table_set_stub(
    module: &mut JITModule,
    id: cranelift_module::FuncId,
    signature: &Signature,
) -> Result<(), Box<cranelift_module::ModuleError>> {
    let mut context = module.make_context();
    let mut function_context = FunctionBuilderContext::new();
    context.func.signature = signature.clone();
    context.func.name = UserFuncName::user(0, id.as_u32());
    {
        let mut builder = FunctionBuilder::new(&mut context.func, &mut function_context);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        let params = builder.block_params(block).to_vec();
        let control = params[0];
        // The VM's entry for a table set, or the jit crate's own through the
        // dispatcher: the control block names the function and its context.
        let entry = builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(mem::offset_of!(TraceControl, set_entry)),
        );
        let context_word = builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(mem::offset_of!(TraceControl, host_context)),
        );
        let mut arguments = vec![context_word];
        arguments.extend_from_slice(&params[1..]);
        let mut set_signature = module.make_signature();
        for parameter in &signature.params {
            set_signature.params.push(*parameter);
        }
        set_signature.returns.push(AbiParam::new(types::I64));
        let set_signature = builder.import_signature(set_signature);
        let call = builder
            .ins()
            .call_indirect(set_signature, entry, &arguments);
        let status = builder.inst_results(call)[0];
        builder.ins().store(
            MemFlags::new(),
            status,
            control,
            offset(mem::offset_of!(TraceControl, set_status)),
        );
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }
    module.define_function(id, &mut context).map_err(Box::new)?;
    module.clear_context(&mut context);
    Ok(())
}

/// The stubs that only forward to a helper whose address is fixed for the
/// process. Each is compiled the first time a trace asks for it, into a module
/// that is never freed, and every trace after that imports the same copy
/// instead of compiling its own into its module. A trace that stores into
/// tables pays for these once per process, not once per compile. The cells
/// are separate so a trace pays only for the stubs it calls.
pub(super) struct SharedStubs {
    admit_table: std::sync::OnceLock<usize>,
    reserve_array: std::sync::OnceLock<usize>,
    table_set: std::sync::OnceLock<usize>,
    hash_get: std::sync::OnceLock<usize>,
    helpers: std::sync::Mutex<std::collections::BTreeMap<usize, usize>>,
}

pub(super) fn shared_stubs() -> &'static SharedStubs {
    static STUBS: std::sync::OnceLock<SharedStubs> = std::sync::OnceLock::new();
    STUBS.get_or_init(|| SharedStubs {
        admit_table: std::sync::OnceLock::new(),
        reserve_array: std::sync::OnceLock::new(),
        table_set: std::sync::OnceLock::new(),
        hash_get: std::sync::OnceLock::new(),
        helpers: std::sync::Mutex::new(std::collections::BTreeMap::new()),
    })
}

/// Compiles one stub into a module of its own that the process keeps, and
/// returns its address. `symbols` are the fixed helper addresses the stub
/// imports; `build` declares and defines the stub and returns its id. The
/// stub is one block of straight-line code, so the cheapest settings give the
/// code every tier would. Cranelift's verifier runs when it does for the
/// trace modules: debug builds, or `OXIMOND_JIT_VERIFY`.
fn compile_shared_stub(
    symbols: &[(&str, usize)],
    build: impl FnOnce(&mut JITModule) -> Result<cranelift_module::FuncId, String>,
) -> Result<usize, String> {
    let mut builder = JITBuilder::with_flags(
        &[
            ("opt_level", "none"),
            (
                "enable_verifier",
                crate::jit::debug_env::cranelift_verifier(),
            ),
            ("regalloc_algorithm", "single_pass"),
        ],
        default_libcall_names(),
    )
    .map_err(|error| error.to_string())?;
    for (name, address) in symbols {
        builder.symbol(*name, *address as *const u8);
    }
    let mut module = JITModule::new(builder);
    let id = build(&mut module)?;
    module
        .finalize_definitions()
        .map_err(|error| error.to_string())?;
    let address = module.get_finalized_function(id) as usize;
    // The code must outlive every trace that calls it, so the module is kept.
    std::mem::forget(module);
    Ok(address)
}

fn once_address(
    cell: &std::sync::OnceLock<usize>,
    make: impl FnOnce() -> Result<usize, String>,
) -> Result<usize, String> {
    if let Some(address) = cell.get() {
        return Ok(*address);
    }
    // A race compiles a second copy that is dropped on the floor, which only
    // wastes its code.
    let address = make()?;
    Ok(*cell.get_or_init(|| address))
}

impl SharedStubs {
    /// The admission stub (see [`define_admit_table_stub`]).
    pub(super) fn admit_table(&self) -> Result<usize, String> {
        once_address(&self.admit_table, || {
            compile_shared_stub(
                &[(ADMIT_TABLE_SYMBOL, trace_admit_table as *const () as usize)],
                |module| {
                    let pointer = module.target_config().pointer_type();
                    let mut admit_signature = module.make_signature();
                    admit_signature.params.push(AbiParam::new(pointer));
                    admit_signature.params.push(AbiParam::new(pointer));
                    admit_signature.returns.push(AbiParam::new(types::I64));
                    let admit_table_id = module
                        .declare_function(ADMIT_TABLE_SYMBOL, Linkage::Import, &admit_signature)
                        .map_err(|error| error.to_string())?;
                    let stub_signature = admit_table_stub_signature(module);
                    let id = module
                        .declare_function(ADMIT_TABLE_STUB, Linkage::Local, &stub_signature)
                        .map_err(|error| error.to_string())?;
                    define_admit_table_stub(module, id, &stub_signature, admit_table_id)
                        .map_err(|error| error.to_string())?;
                    Ok(id)
                },
            )
        })
    }

    /// The array growth stub (see [`define_forwarding_stub`]).
    pub(super) fn reserve_array(&self) -> Result<usize, String> {
        once_address(&self.reserve_array, || {
            compile_shared_stub(
                &[(
                    RESERVE_ARRAY_SYMBOL,
                    crate::runtime::native_table_reserve as *const () as usize,
                )],
                |module| {
                    let reserve_signature = reserve_array_signature(module);
                    let target = module
                        .declare_function(RESERVE_ARRAY_SYMBOL, Linkage::Import, &reserve_signature)
                        .map_err(|error| error.to_string())?;
                    let mut stub_signature = reserve_signature;
                    stub_signature.call_conv = cranelift_codegen::isa::CallConv::PreserveAll;
                    let id = module
                        .declare_function(RESERVE_ARRAY_STUB, Linkage::Local, &stub_signature)
                        .map_err(|error| error.to_string())?;
                    define_forwarding_stub(module, id, &stub_signature, target)
                        .map_err(|error| error.to_string())?;
                    Ok(id)
                },
            )
        })
    }

    /// The table set stub (see [`define_table_set_stub`]).
    pub(super) fn table_set(&self) -> Result<usize, String> {
        once_address(&self.table_set, || {
            compile_shared_stub(&[], |module| {
                let stub_signature = table_set_stub_signature(module);
                let id = module
                    .declare_function(TABLE_SET_STUB, Linkage::Local, &stub_signature)
                    .map_err(|error| error.to_string())?;
                define_table_set_stub(module, id, &stub_signature)
                    .map_err(|error| error.to_string())?;
                Ok(id)
            })
        })
    }

    /// The number-key lookup stub (see [`define_forwarding_stub`]).
    pub(super) fn hash_get(&self) -> Result<usize, String> {
        once_address(&self.hash_get, || {
            compile_shared_stub(
                &[(HASH_GET_SYMBOL, trace_hash_get as *const () as usize)],
                |module| {
                    let get_signature = hash_get_signature(module);
                    let target = module
                        .declare_function(HASH_GET_SYMBOL, Linkage::Import, &get_signature)
                        .map_err(|error| error.to_string())?;
                    let mut stub_signature = get_signature;
                    stub_signature.call_conv = cranelift_codegen::isa::CallConv::PreserveAll;
                    let id = module
                        .declare_function(HASH_GET_STUB, Linkage::Local, &stub_signature)
                        .map_err(|error| error.to_string())?;
                    define_forwarding_stub(module, id, &stub_signature, target)
                        .map_err(|error| error.to_string())?;
                    Ok(id)
                },
            )
        })
    }

    /// The cold helper stub for `arity` integer arguments (see
    /// [`define_helper_stub`]).
    pub(super) fn helper(&self, arity: usize) -> Result<usize, String> {
        let mut helpers = self
            .helpers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(address) = helpers.get(&arity) {
            return Ok(*address);
        }
        let address = compile_shared_stub(&[], |module| {
            let signature = helper_stub_signature(module, arity);
            let id = module
                .declare_function(
                    &format!("{HELPER_STUB_PREFIX}{arity}"),
                    Linkage::Local,
                    &signature,
                )
                .map_err(|error| error.to_string())?;
            define_helper_stub(module, id, arity, &signature).map_err(|error| error.to_string())?;
            Ok(id)
        })?;
        helpers.insert(arity, address);
        Ok(address)
    }
}

fn admit_table_stub_signature(module: &JITModule) -> Signature {
    let pointer = module.target_config().pointer_type();
    let mut signature = module.make_signature();
    signature.call_conv = cranelift_codegen::isa::CallConv::PreserveAll;
    signature.params.push(AbiParam::new(pointer));
    signature.params.push(AbiParam::new(pointer));
    signature
}

fn reserve_array_signature(module: &JITModule) -> Signature {
    let mut signature = module.make_signature();
    signature
        .params
        .push(AbiParam::new(module.target_config().pointer_type()));
    signature
}

/// `(control, words, binding, key tag, key bits, value tag, value bits)`.
fn table_set_stub_signature(module: &JITModule) -> Signature {
    let pointer = module.target_config().pointer_type();
    let mut signature = module.make_signature();
    signature.call_conv = cranelift_codegen::isa::CallConv::PreserveAll;
    signature.params.push(AbiParam::new(pointer));
    signature.params.push(AbiParam::new(pointer));
    for _ in 0..5 {
        signature.params.push(AbiParam::new(types::I64));
    }
    signature
}

/// `(dispatch, out, binding, key tag, key bits)`.
fn hash_get_signature(module: &JITModule) -> Signature {
    let pointer = module.target_config().pointer_type();
    let mut signature = module.make_signature();
    signature.params.push(AbiParam::new(pointer));
    signature.params.push(AbiParam::new(pointer));
    for _ in 0..3 {
        signature.params.push(AbiParam::new(types::I64));
    }
    signature
}

/// Defines the stub a trace calls for the admission of a table its lane index
/// does not name: `(control, slot)` under a convention that preserves every
/// register. It calls [`trace_admit_table`] with the control block's
/// dispatcher and stores the lane in [`TraceControl::admit_lane`].
fn define_admit_table_stub(
    module: &mut JITModule,
    id: cranelift_module::FuncId,
    signature: &Signature,
    admit_table_id: cranelift_module::FuncId,
) -> Result<(), Box<cranelift_module::ModuleError>> {
    let mut context = module.make_context();
    let mut function_context = FunctionBuilderContext::new();
    context.func.signature = signature.clone();
    context.func.name = UserFuncName::user(0, id.as_u32());
    {
        let mut builder = FunctionBuilder::new(&mut context.func, &mut function_context);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        let params = builder.block_params(block).to_vec();
        let control = params[0];
        let dispatch = builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(mem::offset_of!(TraceControl, tables)),
        );
        let callee = module.declare_func_in_func(admit_table_id, builder.func);
        let call = builder.ins().call(callee, &[dispatch, params[1]]);
        let lane = builder.inst_results(call)[0];
        builder.ins().store(
            MemFlags::new(),
            lane,
            control,
            offset(mem::offset_of!(TraceControl, admit_lane)),
        );
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }
    module.define_function(id, &mut context).map_err(Box::new)?;
    module.clear_context(&mut context);
    Ok(())
}

/// The number of integer arguments a call of `helper` with `args` operands
/// passes, counted as [`call_helper_call`] builds its signature, or `None`
/// when the call passes a float, which no stub takes.
fn helper_stub_arity(helper: HelperId, args: usize) -> Option<usize> {
    let info = helper.info()?;
    // The out pointer, then the context and the optional words and count.
    let mut arity = 1;
    if info.context {
        arity += 1;
    }
    if info.effects.takes_words() {
        arity += 1;
    }
    if helper.is_vm_call() {
        arity += 2;
    }
    if info.required < info.params.len() {
        arity += 1;
    }
    for (position, param) in info.params.iter().enumerate() {
        if position >= args {
            // A parameter the call leaves out is nil: two words.
            arity += 2;
            continue;
        }
        arity += match param {
            HelperType::Value => 2,
            HelperType::Float => return None,
            _ => 1,
        };
    }
    Some(arity)
}

/// The signature of the stub a cold helper call goes through: `(callee,
/// control, args...)` with `arity` integers, under a convention that preserves
/// every register.
fn helper_stub_signature(module: &JITModule, arity: usize) -> Signature {
    let mut signature = module.make_signature();
    signature.call_conv = cranelift_codegen::isa::CallConv::PreserveAll;
    for _ in 0..arity + 2 {
        signature.params.push(AbiParam::new(types::I64));
    }
    signature
}

/// Defines the stub a cold helper call goes through (see
/// [`helper_stub_signature`]). It calls the helper at `callee` with the
/// `arity` integers after it, and stores the status the helper returns in
/// [`TraceControl::helper_status`]. The helper's own convention is the
/// system one, so the stub saves every register the helper clobbers.
fn define_helper_stub(
    module: &mut JITModule,
    id: cranelift_module::FuncId,
    arity: usize,
    signature: &Signature,
) -> Result<(), Box<cranelift_module::ModuleError>> {
    let mut context = module.make_context();
    let mut function_context = FunctionBuilderContext::new();
    context.func.signature = signature.clone();
    context.func.name = UserFuncName::user(0, id.as_u32());
    {
        let mut builder = FunctionBuilder::new(&mut context.func, &mut function_context);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        let params = builder.block_params(block).to_vec();
        let (callee, control) = (params[0], params[1]);
        let mut helper_signature = module.make_signature();
        for _ in 0..arity {
            helper_signature.params.push(AbiParam::new(types::I64));
        }
        helper_signature.returns.push(AbiParam::new(types::I64));
        let helper_signature = builder.import_signature(helper_signature);
        let call = builder
            .ins()
            .call_indirect(helper_signature, callee, &params[2..]);
        let status = builder.inst_results(call)[0];
        builder.ins().store(
            MemFlags::new(),
            status,
            control,
            offset(mem::offset_of!(TraceControl, helper_status)),
        );
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }
    module.define_function(id, &mut context).map_err(Box::new)?;
    module.clear_context(&mut context);
    Ok(())
}

#[allow(unsafe_code)]
fn trace_entry(code: *const u8) -> TraceEntry {
    // SAFETY: `code` comes from `get_finalized_function` for a function
    // declared as `(pointer, pointer, pointer) -> i64`. The module stays
    // owned by `TraceCode`, so the allocation outlives this pointer.
    unsafe { mem::transmute::<*const u8, TraceEntry>(code) }
}

const fn clif_type(ty: TraceType) -> types::Type {
    match ty {
        TraceType::Float => types::F64,
        TraceType::Integer
        | TraceType::Int64
        | TraceType::UInt64
        | TraceType::Table
        | TraceType::Closure
        | TraceType::Thread
        | TraceType::Userdata
        | TraceType::String
        | TraceType::StringRef => types::I64,
        TraceType::Boolean | TraceType::Nil | TraceType::Any => types::I8,
    }
}

fn offset(value: usize) -> i32 {
    i32::try_from(value).expect("trace ABI offset fits i32")
}

/// A loop parameter that holds the key of an array step ([`TraceOp::ArrayNext`])
/// and needs no check in the steps of the loop.
#[derive(Clone, Copy)]
pub(super) struct TrustedKey {
    /// The parameter, by value.
    pub(super) key: ValueId,
    /// The array binding the steps run over.
    pub(super) binding: u32,
    /// The entry input that holds the parameter's initial value, by value
    /// and by position.
    pub(super) init: ValueId,
    pub(super) input: usize,
}

/// How native code does a helper that reads or writes the cell of a callee
/// itself, through the row of the closure's lane ([`CLOSURE_ROW_WORDS`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ClosureCellUse {
    /// [`HelperId::CLOSURE_SAME_PROTO`]: compare the rows' prototype words.
    Proto,
    /// A read of upvalue `index` that holds a value of type `ty`.
    Get { index: usize, ty: TraceType },
    /// A read of upvalue `index` that holds a table: the lane the row names
    /// for the cell, kept only while the cell still holds that table.
    GetTable { index: usize },
    /// A write of a value of type `ty` over upvalue `index`, which keeps the
    /// cell's type.
    Set { index: usize, ty: TraceType },
}

/// A `==` of a borrowed string and a short root-string constant
/// ([`HelperId::STRING_EQUAL`], `x == "name"`), which native code answers
/// without calling the helper: the borrowed operand, and the storage words of
/// the constant ([`crate::runtime::native_inline_words`]). The borrowed
/// string is equal to the constant exactly when its storage starts with those
/// words. A borrow's slot holds a string for as long as the borrow lives (the
/// verifier ends borrows at every operation that could change it), so the
/// words are there to read, and the helper's miss for a slot that holds no
/// string never happens. `None` for any other call, which the helper makes.
pub(super) fn string_equal_constant(
    ir: &TraceIr,
    helper: HelperId,
    args: &[ValueId],
) -> Option<(ValueId, [u64; NATIVE_STRING_INLINE_WORDS])> {
    let [left, right] = *args else {
        return None;
    };
    if helper != HelperId::STRING_EQUAL {
        return None;
    }
    let words = |value: ValueId| {
        let instruction = match ir.values.get(value.index())?.def {
            ValueDef::Instruction(index) => ir.body.get(index)?,
            ValueDef::Preheader(index) => ir.preheader.get(index)?,
            _ => return None,
        };
        let TraceOp::Constant(TraceScalar::String(lane)) = instruction.op else {
            return None;
        };
        crate::runtime::native_inline_words(ir.root_constants.get(root_string_slot(lane)?)?)
    };
    let borrowed = |value: ValueId| ir.value_type(value) == Some(TraceType::StringRef);
    if borrowed(left) {
        Some((left, words(right)?))
    } else if borrowed(right) {
        Some((right, words(left)?))
    } else {
        None
    }
}

/// Whether native code compares the strings of a `==` of two strings
/// ([`HelperId::STRING_EQUAL`]) itself, by the words of their storage, and
/// calls the helper only when those cannot tell (see
/// `Lowerer::string_equal_by_words`): both operands are strings, held by lane
/// or borrowed.
pub(super) fn string_equal_by_words(ir: &TraceIr, helper: HelperId, args: &[ValueId]) -> bool {
    helper == HelperId::STRING_EQUAL
        && args.len() == 2
        && args.iter().all(|arg| {
            matches!(
                ir.value_type(*arg),
                Some(TraceType::String | TraceType::StringRef)
            )
        })
}

/// The integer a value is, when a `Constant` of `ir` defined it.
pub(super) fn constant_integer(ir: &TraceIr, value: ValueId) -> Option<i64> {
    let instruction = match ir.values.get(value.index())?.def {
        ValueDef::Instruction(index) => ir.body.get(index)?,
        ValueDef::Preheader(index) => ir.preheader.get(index)?,
        _ => return None,
    };
    match instruction.op {
        TraceOp::Constant(TraceScalar::Integer(index)) => Some(index),
        _ => None,
    }
}

/// How native code does a call of `helper` on `args` without calling it, if
/// it does: a prototype check, or a cell read or write of one of the first
/// [`CLOSURE_ROW_CELLS`] upvalues.
pub(super) fn closure_cell_use(
    ir: &TraceIr,
    helper: HelperId,
    args: &[ValueId],
    table_cells: bool,
) -> Option<ClosureCellUse> {
    let index = |position: usize| {
        let index = usize::try_from(constant_integer(ir, *args.get(position)?)?).ok()?;
        (index < CLOSURE_ROW_CELLS).then_some(index)
    };
    match helper {
        HelperId::CLOSURE_SAME_PROTO => Some(ClosureCellUse::Proto),
        HelperId::CELL_GET_INTEGER => Some(ClosureCellUse::Get {
            index: index(1)?,
            ty: TraceType::Integer,
        }),
        HelperId::CELL_GET_FLOAT => Some(ClosureCellUse::Get {
            index: index(1)?,
            ty: TraceType::Float,
        }),
        HelperId::CELL_GET_BOOLEAN => Some(ClosureCellUse::Get {
            index: index(1)?,
            ty: TraceType::Boolean,
        }),
        HelperId::CELL_GET_TABLE if table_cells => {
            Some(ClosureCellUse::GetTable { index: index(1)? })
        }
        HelperId::CELL_SET => {
            let ty = ir.value_type(*args.get(2)?)?;
            if !matches!(
                ty,
                TraceType::Integer | TraceType::Float | TraceType::Boolean
            ) {
                return None;
            }
            Some(ClosureCellUse::Set {
                index: index(1)?,
                ty,
            })
        }
        _ => None,
    }
}

/// Whether the rows of a tree's run end in a presence mask: some member that
/// is not a function trace reads or writes two fields of one table, which
/// the lowering checks together with one test of the mask (see
/// `Lowerer::slot_group`). The mask costs a word per row and a pass over the
/// row when the VM fills it, so a tree that never groups fields has none.
pub(super) fn rows_masked(members: &[&TraceIr]) -> bool {
    members.iter().any(|member| {
        if member.function.is_some() {
            return false;
        }
        let mut seen: Vec<(ValueId, u32)> = Vec::new();
        member
            .preheader
            .iter()
            .chain(&member.body)
            .any(|instruction| match &instruction.op {
                TraceOp::TableFieldLoad { table, field, .. }
                | TraceOp::TableFieldStore { table, field, .. } => {
                    let other = seen.iter().any(|(on, held)| on == table && held != field);
                    seen.push((*table, *field));
                    other
                }
                _ => false,
            })
    })
}

/// The words of the closure rows a tree of `members` needs: zero when no call
/// is done through the rows, [`CLOSURE_ROW_WORDS`](super::exit::CLOSURE_ROW_WORDS)
/// for cells of numbers and booleans, and
/// [`CLOSURE_ROW_TABLE_WORDS`](super::exit::CLOSURE_ROW_TABLE_WORDS) when the
/// reads of table-valued cells are native too.
///
/// A row names the table a cell held when it was filled, so table cells are
/// native only for a tree that never writes a table into a cell through a
/// helper: nothing in the run then changes what a cell holds, and the code
/// still checks the cell's payload against the lane's table.
pub(super) fn closure_row_words<'a>(members: &[&'a TraceIr]) -> usize {
    let helper_calls = |ir: &'a TraceIr| {
        ir.preheader
            .iter()
            .chain(&ir.body)
            .filter_map(move |instruction| match &instruction.op {
                TraceOp::CallHelper { helper, args, .. } => Some((ir, *helper, args)),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let calls: Vec<_> = members.iter().copied().flat_map(helper_calls).collect();
    let writes_tables = calls.iter().any(|(ir, helper, args)| {
        *helper == HelperId::CELL_SET
            && args
                .get(2)
                .is_none_or(|value| matches!(ir.value_type(*value), None | Some(TraceType::Table)))
    });
    // Code the run resumes may write any cell between two reads.
    let table_cells = !writes_tables
        && !members.iter().any(|member| member.resumes())
        && calls.iter().any(|(ir, helper, args)| {
            closure_cell_use(ir, *helper, args, true)
                .is_some_and(|cell| matches!(cell, ClosureCellUse::GetTable { .. }))
        });
    if table_cells {
        return super::exit::CLOSURE_ROW_TABLE_WORDS;
    }
    if calls
        .iter()
        .any(|(ir, helper, args)| closure_cell_use(ir, *helper, args, false).is_some())
    {
        super::exit::CLOSURE_ROW_WORDS
    } else {
        0
    }
}

/// The key of the root's array step that every step after the first can
/// take as it is: the slot it names holds a value because the step before
/// found it.
///
/// That holds when nothing in the tree clears a slot, the root's loop-back
/// passes the result of an `ArrayNext` over the binding for the parameter,
/// and every link of a side that comes back to the root's header hands over
/// the result of an `ArrayNext` over the same table. Entry then checks the
/// key it starts with ([`Lowerer::check_trusted_key`]).
pub(super) fn trusted_array_key(members: &[LoweredMember<'_>]) -> Option<TrustedKey> {
    let root = members[0].ir;
    if root.function.is_some()
        || members
            .iter()
            .any(|member| member.ir.may_clear_array_slots())
    {
        return None;
    }
    let TraceTerminator::LoopBack { args } = &root.terminator else {
        return None;
    };
    let array_register = |ir: &TraceIr, binding: u32| {
        let binding = ir.bindings.get(binding as usize)?;
        matches!(binding.kind, BindingKind::Array { .. }).then_some(binding.register)
    };
    root.loop_params.iter().zip(args).find_map(|(param, arg)| {
        let (LoopParamInit::Input(init), Some(register)) = (param.init, param.register) else {
            return None;
        };
        let input = root
            .entry
            .inputs
            .iter()
            .position(|input| input.value == init)?;
        if !matches!(param.ty, TraceType::Integer | TraceType::Float) {
            return None;
        }
        let binding = root
            .body
            .iter()
            .find_map(|instruction| match instruction.op {
                TraceOp::ArrayNext {
                    binding,
                    key,
                    start: false,
                    ..
                } if key == param.value && instruction.result == Some(*arg) => Some(binding),
                _ => None,
            })?;
        let table = array_register(root, binding)?;
        let steps_over_table = |ir: &TraceIr, value: ValueId| {
            ir.body.iter().any(|instruction| {
                instruction.result == Some(value)
                    && matches!(instruction.op, TraceOp::ArrayNext { binding, .. }
                        if array_register(ir, binding) == Some(table))
            })
        };
        // Every way back to the header hands over a key a step found.
        let links_found = members[1..].iter().all(|member| {
            let TraceTerminator::Link { snapshot } = &member.ir.terminator else {
                return true;
            };
            matches!(
                member.ir.snapshots[snapshot.index()].frames[0]
                    .registers
                    .get(usize::from(register)),
                Some(RegisterEntry::Value(value)) if steps_over_table(member.ir, *value)
            )
        });
        links_found.then_some(TrustedKey {
            key: param.value,
            binding,
            init,
            input,
        })
    })
}

/// Whether the tree's root reads an array key that only its members' links
/// keep valid (see [`trusted_array_key`]).
pub(super) fn has_trusted_key(members: &[LoweredMember<'_>]) -> bool {
    trusted_array_key(members).is_some()
}

/// The first header binding of `bindings` on the register of the array
/// binding `array`, and whether it is strict. Both backends find the header
/// of a table they set through this way.
pub(super) fn header_binding_of(bindings: &[TableBinding], array: u32) -> Option<(usize, bool)> {
    let register = bindings.get(array as usize)?.register;
    bindings
        .iter()
        .enumerate()
        .find_map(|(index, other)| match other.kind {
            BindingKind::Header { strict } if other.register == register => Some((index, strict)),
            _ => None,
        })
}

fn block_args(values: &[Value]) -> Vec<BlockArg> {
    values.iter().copied().map(BlockArg::Value).collect()
}

#[allow(clippy::too_many_lines)]
pub(super) fn lower(
    members: &[LoweredMember<'_>],
    plan: &ExitPlan,
    tier: OptimizationTier,
    forced_guard: Option<(usize, usize)>,
    options: TraceCompileOptions,
    detached: Option<&DetachedSpec<'_>>,
) -> Result<TraceCode, TraceCompileError> {
    // The MIR backend, when asked for: it takes the whole tree or none of
    // it, and what it refuses goes on to Cranelift below.
    if detached.is_none()
        && tier != OptimizationTier::Peak
        && crate::jit::debug_env::backend_mir()
        && let Some(compiled) = super::mir::lower(members, plan, tier, forced_guard, options)
    {
        return Ok(TraceCode::from_mir(compiled));
    }
    if crate::jit::debug_env::backend_mir_only() {
        if crate::jit::debug_env::trace() {
            eprintln!("jit-trace: native trace rejected: OXIMOND_JIT_BACKEND=mir-only");
        }
        return Err(TraceCompileError::NativeBackendRejected);
    }
    // A side compiled apart from its tree lowers its own trace, the last
    // member, where the tree's code lowers the root.
    let ir = members[usize::from(detached.is_some())].ir;
    if super::census::enabled() {
        super::census::begin();
    }
    // The trace optimizer has already folded, numbered and hoisted the
    // code, so Cranelift's own optimizer mostly finds little to do and costs
    // a third of the compile. Only a tree that ran long gets it.
    let optimization = match tier {
        OptimizationTier::Baseline | OptimizationTier::Optimized => "none",
        OptimizationTier::Peak => "speed",
    };
    let rejected = |error: &dyn std::fmt::Display| {
        if crate::jit::debug_env::trace() {
            eprintln!("jit-trace: native trace rejected: {error}");
        }
        TraceCompileError::NativeBackendRejected
    };
    // The baseline tier is for code that runs briefly: the single pass
    // allocator takes a fraction of the backtracking one's compile time.
    let allocator = match tier {
        OptimizationTier::Baseline => "single_pass",
        OptimizationTier::Optimized | OptimizationTier::Peak => "backtracking",
    };
    let mut jit_builder = JITBuilder::with_flags(
        &[
            ("opt_level", optimization),
            // The IR verifier already ran; Cranelift's own is for debug
            // builds, where it catches a value used where its definition
            // does not dominate, and for release builds run with
            // `OXIMOND_JIT_VERIFY`, where it turns such a codegen bug into a
            // rejection instead of an allocator panic.
            (
                "enable_verifier",
                crate::jit::debug_env::cranelift_verifier(),
            ),
            ("regalloc_algorithm", allocator),
        ],
        default_libcall_names(),
    )
    .map_err(|error| rejected(&error))?;
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    jit_builder.symbol_lookup_fn(Box::new(super::libm_lookup::libm_symbol));
    jit_builder.symbol(POW_SYMBOL, trace_pow as *const u8);
    let uses = |form: FloatModulo| {
        members.iter().any(|member| {
            member
                .ir
                .has_arithmetic(&|op| op == ArithmeticOp::ModuloWith(form))
        })
    };
    let uses_fmod53 = uses(FloatModulo::Fmod53);
    let uses_fmod54 = uses(FloatModulo::Fmod54);
    if uses_fmod53 {
        jit_builder.symbol(FMOD53_SYMBOL, trace_mod_fmod53 as *const u8);
    }
    if uses_fmod54 {
        jit_builder.symbol(FMOD54_SYMBOL, trace_mod_fmod54 as *const u8);
    }
    jit_builder.symbol(ADMIT_TABLE_SYMBOL, trace_admit_table as *const u8);
    let wants_admit_stub = members.iter().all(|member| {
        member.ir.function.is_none() && !member.ir.makes_tables() && !member.ir.resumes()
    });
    // Every member of the tree, the sides included: a set in any of them
    // may move the arrays all of them read.
    let sets_tables = members.iter().any(|member| member.ir.sets_tables());
    // The helpers that store into a hash part (see `hash_store`) take a
    // register-preserving stub on their cold path, one per argument count.
    let helper_arities: std::collections::BTreeSet<usize> = members
        .iter()
        .flat_map(|member| member.ir.preheader.iter().chain(&member.ir.body))
        .filter_map(|instruction| match &instruction.op {
            TraceOp::CallHelper { helper, args, .. }
                if matches!(
                    *helper,
                    HelperId::SET_PRESENT | HelperId::SET_REVIVE | HelperId::DELETE_STRING
                ) =>
            {
                helper_stub_arity(*helper, args.len())
            }
            _ => None,
        })
        .collect();
    let wants_hash_get = members.iter().any(|member| member.ir.reads_hash());
    let wants_hash_get_stub = members.iter().any(|member| member.ir.reads_number_hash());
    // The stubs that only forward to a helper come from one module per
    // process (see `SharedStubs`) and are imported by address.
    let stubs = shared_stubs();
    if wants_admit_stub {
        let address = stubs.admit_table().map_err(|error| rejected(&error))?;
        jit_builder.symbol(ADMIT_TABLE_STUB, address as *const u8);
    }
    if sets_tables {
        let address = stubs.reserve_array().map_err(|error| rejected(&error))?;
        jit_builder.symbol(RESERVE_ARRAY_STUB, address as *const u8);
        let address = stubs.table_set().map_err(|error| rejected(&error))?;
        jit_builder.symbol(TABLE_SET_STUB, address as *const u8);
    }
    for &arity in &helper_arities {
        let address = stubs.helper(arity).map_err(|error| rejected(&error))?;
        jit_builder.symbol(format!("{HELPER_STUB_PREFIX}{arity}"), address as *const u8);
    }
    if wants_hash_get {
        jit_builder.symbol(HASH_GET_SYMBOL, trace_hash_get as *const u8);
        if wants_hash_get_stub {
            let address = stubs.hash_get().map_err(|error| rejected(&error))?;
            jit_builder.symbol(HASH_GET_STUB, address as *const u8);
        }
    }
    jit_builder.symbol(ENSURE_ROW_SYMBOL, trace_ensure_row as *const u8);
    jit_builder.symbol(
        RESERVE_ARRAY_SYMBOL,
        crate::runtime::native_table_reserve as *const u8,
    );
    let mut module = JITModule::new(jit_builder);
    let pointer = module.target_config().pointer_type();

    let mut signature = module.make_signature();
    for _ in 0..3 {
        signature.params.push(AbiParam::new(pointer));
    }
    signature.returns.push(AbiParam::new(types::I64));
    let function_id = module
        .declare_function("oximond_trace", Linkage::Local, &signature)
        .map_err(|error| rejected(&error))?;
    // A function trace runs each activation in a function of its own, which
    // a self call calls again. `oximond_trace` checks the entry and calls it
    // for the activation the VM entered.
    let activation = match &ir.function {
        Some(function) => {
            let mut body_signature = module.make_signature();
            body_signature.params.push(AbiParam::new(pointer));
            // The activations the run may still nest, and the fuel left.
            body_signature.params.push(AbiParam::new(types::I64));
            body_signature.params.push(AbiParam::new(types::I64));
            // The activation's tail call count, for a function that tail
            // calls itself.
            if ir.external.contains(&ExternalSlot::TailCalls) {
                body_signature.params.push(AbiParam::new(types::I64));
            }
            for ty in &function.parameters {
                body_signature.params.push(AbiParam::new(clif_type(*ty)));
            }
            // The status (or the fuel left), and the result.
            body_signature.returns.push(AbiParam::new(types::I64));
            if let Some(ty) = function.result {
                body_signature.returns.push(AbiParam::new(clif_type(ty)));
            }
            let id = module
                .declare_function("oximond_trace_activation", Linkage::Local, &body_signature)
                .map_err(|error| rejected(&error))?;
            Some((id, body_signature))
        }
        None => None,
    };
    let mut pow_signature = module.make_signature();
    pow_signature.params.push(AbiParam::new(types::F64));
    pow_signature.params.push(AbiParam::new(types::F64));
    pow_signature.returns.push(AbiParam::new(types::F64));
    let pow_id = module
        .declare_function(POW_SYMBOL, Linkage::Import, &pow_signature)
        .map_err(|error| rejected(&error))?;

    // The large `Err` is `cranelift_module::ModuleError`, the backend's own
    // error type; the closure immediately folds it into `TraceCompileError`.
    #[allow(clippy::result_large_err)]
    let mut declare_fmod = |used: bool, symbol: &str| {
        used.then(|| module.declare_function(symbol, Linkage::Import, &pow_signature))
            .transpose()
            .map_err(|error| rejected(&error))
    };
    let fmod53_id = declare_fmod(uses_fmod53, FMOD53_SYMBOL)?;
    let fmod54_id = declare_fmod(uses_fmod54, FMOD54_SYMBOL)?;

    let mut admit_signature = module.make_signature();
    admit_signature.params.push(AbiParam::new(pointer));
    admit_signature.params.push(AbiParam::new(pointer));
    admit_signature.returns.push(AbiParam::new(types::I64));
    let admit_table_id = module
        .declare_function(ADMIT_TABLE_SYMBOL, Linkage::Import, &admit_signature)
        .map_err(|error| rejected(&error))?;

    let mut ensure_signature = module.make_signature();
    ensure_signature.params.push(AbiParam::new(pointer));
    ensure_signature.params.push(AbiParam::new(types::I64));
    let ensure_row_id = module
        .declare_function(ENSURE_ROW_SYMBOL, Linkage::Import, &ensure_signature)
        .map_err(|error| rejected(&error))?;

    let mut reserve_signature = module.make_signature();
    reserve_signature.params.push(AbiParam::new(pointer));
    let reserve_array_id = module
        .declare_function(RESERVE_ARRAY_SYMBOL, Linkage::Import, &reserve_signature)
        .map_err(|error| rejected(&error))?;

    // The root's loop runs in a version without the bounds checks its range
    // plan proves, when the checks it saves are worth lowering the tree twice.
    // The peak recompile keeps it: a counted array loop past the peak
    // threshold would otherwise go back to per-access checks.
    let tree_size: usize = members
        .iter()
        .map(|member| member.ir.preheader.len() + member.ir.body.len())
        .sum();
    let range_plan = (tier != OptimizationTier::Baseline
        && options.opt_flags.contains(TracingOptFlags::RANGE_PROOFS)
        && ir.function.is_none()
        && detached.is_none()
        && tree_size <= MAX_VERSIONED_INSTRUCTIONS)
        .then(|| RangePlan::of(ir, sets_tables))
        .flatten()
        .filter(|plan| versioning_pays(plan, tree_size));
    // The trace calls the set through a stub that preserves every register:
    // the call sits on a cold path, and a call that clobbers the caller-saved
    // registers (every float register) would make the allocator keep the
    // loop's values on the stack in the hot path too. The stub takes the
    // control block instead of the dispatcher and leaves the result in it.
    // The growth of an array sits on the cold branch of an append, like the
    // set itself: the same stub keeps it from spilling the loop's values.
    // The stubs come from `SharedStubs`, so the module only imports them.
    let reserve_array_stub_id = if sets_tables {
        let mut stub_signature = reserve_signature.clone();
        stub_signature.call_conv = cranelift_codegen::isa::CallConv::PreserveAll;
        Some(
            module
                .declare_function(RESERVE_ARRAY_STUB, Linkage::Import, &stub_signature)
                .map_err(|error| rejected(&error))?,
        )
    } else {
        None
    };
    let table_set_stub_id = if sets_tables {
        let stub_signature = table_set_stub_signature(&module);
        Some(
            module
                .declare_function(TABLE_SET_STUB, Linkage::Import, &stub_signature)
                .map_err(|error| rejected(&error))?,
        )
    } else {
        None
    };

    // A loop that meets the tables of an array again and again finds them in
    // the run's lane index, and asks the VM only for one it does not hold. The
    // call sits on a cold path of that loop, so it takes a stub that
    // preserves every register, as the table set does: a call that clobbers
    // the caller-saved registers would keep the loop's floats on the stack.
    let admit_table_stub_id = if wants_admit_stub {
        let stub_signature = admit_table_stub_signature(&module);
        Some(
            module
                .declare_function(ADMIT_TABLE_STUB, Linkage::Import, &stub_signature)
                .map_err(|error| rejected(&error))?,
        )
    } else {
        None
    };

    // Declared only for a tree that calls one of the hash store helpers.
    let mut helper_stub_ids = std::collections::BTreeMap::new();
    for &arity in &helper_arities {
        let signature = helper_stub_signature(&module, arity);
        let id = module
            .declare_function(
                &format!("{HELPER_STUB_PREFIX}{arity}"),
                Linkage::Import,
                &signature,
            )
            .map_err(|error| rejected(&error))?;
        helper_stub_ids.insert(arity, id);
    }

    // A plain call: the lookup sits on the hot path, where a
    // register-preserving stub would save and restore every register around
    // a call that clobbers few live values. Declared only for a tree that
    // looks up hashes, so other traces compile exactly as before.
    let hash_get_id = if wants_hash_get {
        let get_signature = hash_get_signature(&module);
        let id = module
            .declare_function(HASH_GET_SYMBOL, Linkage::Import, &get_signature)
            .map_err(|error| rejected(&error))?;
        Some((id, get_signature))
    } else {
        None
    };
    // A number key outside the array part is looked up where the loop does
    // not expect it (`t[0]` the first trip, a sparse key now and then), so
    // the call sits on a cold path of a loop that otherwise reads the array
    // part. A call that clobbers the caller-saved registers would make the
    // allocator keep the loop's values on the stack in the hot path too, as
    // for a table set, so the lookup goes through a stub that preserves
    // every register.
    let hash_get_stub_id = match &hash_get_id {
        Some((_, get_signature)) if wants_hash_get_stub => {
            let mut stub_signature = get_signature.clone();
            stub_signature.call_conv = cranelift_codegen::isa::CallConv::PreserveAll;
            Some(
                module
                    .declare_function(HASH_GET_STUB, Linkage::Import, &stub_signature)
                    .map_err(|error| rejected(&error))?,
            )
        }
        _ => None,
    };
    let hash_get_id = hash_get_id.map(|(id, _)| id);

    let mut context = module.make_context();
    let mut function_context = FunctionBuilderContext::new();
    let lowered_id = if let Some((id, body_signature)) = &activation {
        context.func.signature = body_signature.clone();
        *id
    } else {
        context.func.signature = signature.clone();
        function_id
    };
    context.func.name = UserFuncName::user(0, lowered_id.as_u32());
    let (side_extras, side_writebacks, prepared);
    #[cfg(test)]
    let versioned;
    let native_link;
    {
        let mut builder = FunctionBuilder::new(&mut context.func, &mut function_context);
        let function = activation.as_ref().map(|(id, _)| {
            let stride = plan.stride.expect("a function trace's plan has a stride");
            FunctionLowering {
                body: module.declare_func_in_func(*id, builder.func),
                stride_bytes: i64::try_from(stride * mem::size_of::<i64>())
                    .expect("an exit area fits i64"),
                header_offset: offset((stride - 1) * mem::size_of::<i64>()),
                result: ir.function.as_ref().and_then(|f| f.result).map(clif_type),
                count: ir
                    .external
                    .iter()
                    .position(|slot| *slot == ExternalSlot::TailCalls)
                    .map(|position| {
                        u16::try_from(
                            usize::from(ir.frames[0].register_count) - ir.external.len() + position,
                        )
                        .expect("window fits u16")
                    }),
                start: None,
            }
        });
        // Only fmod lowering asks; reading the flags allocates once.
        let has_fma = (fmod53_id.is_some() || fmod54_id.is_some()) && isa_has_fma(module.isa());
        let pow = module.declare_func_in_func(pow_id, builder.func);
        let fmod53 = fmod53_id.map(|id| module.declare_func_in_func(id, builder.func));
        let fmod54 = fmod54_id.map(|id| module.declare_func_in_func(id, builder.func));
        let admit_table = module.declare_func_in_func(admit_table_id, builder.func);
        let ensure_row = module.declare_func_in_func(ensure_row_id, builder.func);
        let reserve_array = module.declare_func_in_func(
            reserve_array_stub_id.unwrap_or(reserve_array_id),
            builder.func,
        );
        let table_set = table_set_stub_id.map(|id| module.declare_func_in_func(id, builder.func));
        let admit_table_stub =
            admit_table_stub_id.map(|id| module.declare_func_in_func(id, builder.func));
        let hash_get = hash_get_id.map(|id| module.declare_func_in_func(id, builder.func));
        let hash_get_number =
            hash_get_stub_id.map(|id| module.declare_func_in_func(id, builder.func));
        let helper_stubs: std::collections::BTreeMap<usize, FuncRef> = helper_stub_ids
            .iter()
            .map(|(arity, id)| (*arity, module.declare_func_in_func(*id, builder.func)))
            .collect();
        let irs: Vec<&TraceIr> = members.iter().map(|member| member.ir).collect();
        let closure_row_words = closure_row_words(&irs);
        let row_masked = detached.map_or_else(|| rows_masked(&irs), |spec| spec.row_masked);
        let mut lowerer = Lowerer {
            builder: &mut builder,
            members,
            ir,
            member: usize::from(detached.is_some()),
            plan,
            pow,
            fmod53,
            fmod54,
            has_fma,
            admit_table,
            admit_table_stub,
            ensure_row,
            reserve_array,
            table_set,
            direct_set: members.iter().any(|member| member.ir.calls_host()),
            hash_get,
            hash_get_number,
            helper_stubs,
            values: vec![None; ir.values.len()],
            helper_results: std::collections::HashMap::new(),
            made_tables: std::collections::HashMap::new(),
            next_slot: None,
            // The sides are members of their own once the tree is planned, so
            // the root's IR alone does not say what the tree does.
            moves_hash: members.iter().any(|member| member.ir.moves_hash_storage()),
            native_probe: match detached {
                // The root decided, and the side must not move what the root
                // probes.
                Some(spec) => spec.native_probe && !members[1].ir.moves_hash_storage(),
                None => {
                    members.iter().any(|member| member.ir.uses_string_index())
                        && !members.iter().any(|member| member.ir.moves_hash_storage())
                }
            },
            constants: vec![None; ir.values.len()],
            exit_area: None,
            control: None,
            rows: None,
            row_stride: row_stride_of(ir.fields.len(), closure_row_words, row_masked),
            row_masked,
            row_words: closure_row_words,
            // Only a loop that meets the tables of an array again and again,
            // and makes none, finds them in the run's lane index.
            lane_probe: members.iter().all(|member| {
                member.ir.function.is_none() && !member.ir.makes_tables() && !member.ir.resumes()
            }),
            header: None,
            native_dispatch: None,
            native_link: None,
            native_entry: None,
            inner: None,
            root_inputs: std::collections::HashMap::new(),
            any_slots: std::collections::HashMap::new(),
            bound: Vec::new(),
            bound_length_float: Vec::new(),
            words: None,
            reload_arrays: sets_tables,
            trusted_key: if detached.is_some() {
                None
            } else {
                trusted_array_key(members)
            },
            fresh_words: false,
            array_cache: vec![None; ir.bindings.len()],
            lane_array_cache: Vec::new(),
            budget: None,
            forced_guard,
            count_links: options.count_links,
            function,
            depth_left: None,
            activation_fuel: None,
            deeper: None,
            lazy: std::collections::HashMap::new(),
            entry_slots: std::collections::HashMap::new(),
            entry_inputs: None,
            tails: (0..plan.shared_stores().tails.len())
                .map(|_| Vec::new())
                .collect(),
            deferred: Vec::new(),
            member_fuel: None,
            rematerialized: std::collections::HashMap::new(),
            number_tags: std::collections::HashMap::new(),
            fused: std::collections::HashMap::new(),
            pending_delete: None,
            reload_inputs: false,
            versioned_loop: false,
            #[cfg(feature = "op-stats")]
            range_facts: members
                .iter()
                .map(|member| RangeFacts::of(member.ir, sets_tables))
                .collect(),
            range_plan,
            fast: None,
            slow_loop: None,
            root_side: false,
            loop_header: None,
            loop_stack: Vec::new(),
            side_origins: Vec::new(),
            slots: Vec::new(),
            row_addresses: Vec::new(),
            field_slots: Vec::new(),
            peak: tier == OptimizationTier::Peak,
            detached,
            side_extras: detached.map_or_else(Vec::new, |spec| spec.extras.to_vec()),
            side_writebacks: Vec::new(),
            string_constants: options
                .opt_flags
                .contains(TracingOptFlags::STRING_CONSTANTS),
            prepared: Vec::new(),
            prepared_patterns: std::collections::HashMap::new(),
            gmatch_patterns: std::collections::HashMap::new(),
        };
        if detached.is_some() {
            lowerer.lower_detached()?;
        } else {
            lowerer.lower()?;
        }
        side_extras = std::mem::take(&mut lowerer.side_extras);
        side_writebacks = std::mem::take(&mut lowerer.side_writebacks);
        native_link = lowerer.native_link.is_some();
        prepared = std::mem::take(&mut lowerer.prepared);
        #[cfg(test)]
        {
            versioned = lowerer.versioned_loop;
        }
        builder.seal_all_blocks();
        builder.finalize();
    }
    // The size of what Cranelift is asked to compile, which its time follows.
    let clif_insts: usize = context
        .func
        .layout
        .blocks()
        .map(|block| context.func.layout.block_insts(block).count())
        .sum();
    if super::census::enabled() {
        let irs: Vec<&TraceIr> = members.iter().map(|member| member.ir).collect();
        eprintln!(
            "jit-census: {}",
            super::census::report(&irs, tier, clif_insts, detached.is_some())
        );
    }
    if crate::jit::debug_env::disasm() {
        eprintln!("jit-disasm: trace clif\n{}", context.func.display());
    }
    if crate::jit::debug_env::disasm() {
        context.set_disasm(true);
    }
    module
        .define_function(lowered_id, &mut context)
        .map_err(|error| rejected(&error))?;
    if let Some(vcode) = context
        .compiled_code()
        .and_then(|code| code.vcode.as_deref())
    {
        eprintln!("jit-disasm: trace machine code\n{vcode}");
    }
    #[cfg(test)]
    let fmas = {
        let func = &context.func;
        func.layout
            .blocks()
            .flat_map(|block| func.layout.block_insts(block))
            .filter(|inst| func.dfg.insts[*inst].opcode() == cranelift_codegen::ir::Opcode::Fma)
            .count()
    };
    #[cfg(test)]
    let calls = {
        let func = &context.func;
        func.layout
            .blocks()
            .flat_map(|block| func.layout.block_insts(block))
            .filter(|inst| func.dfg.insts[*inst].opcode().is_call())
            .count()
    };
    let code_size = context
        .compiled_code()
        .map_or(0, |code| code.code_buffer().len());
    module.clear_context(&mut context);
    if let Some((body_id, body_signature)) = &activation {
        define_function_entry(
            &mut module,
            function_id,
            &signature,
            *body_id,
            body_signature,
            ir,
        )
        .map_err(|error| rejected(&error))?;
    }
    module
        .finalize_definitions()
        .map_err(|error| rejected(&error))?;
    let entry = trace_entry(module.get_finalized_function(function_id));
    Ok(TraceCode {
        module: Some(module),
        mir: None,
        entry,
        code_size,
        clif_insts,
        side_extras,
        side_writebacks,
        prepared,
        #[cfg(test)]
        versioned,
        native_link,
        #[cfg(test)]
        calls,
        #[cfg(test)]
        fmas,
    })
}

/// Defines `oximond_trace` for a function trace: it checks the entry input
/// tags, loads the parameters and runs the entered activation (`body`). A
/// return comes back as [`RETURNED_CODE`], with the result in slot 0 of the
/// exit area as an exit slot of its type holds it. Any other status is the
/// exit the run left through.
#[allow(
    clippy::too_many_lines,
    reason = "one entry function: checks, loads, call, result"
)]
fn define_function_entry(
    module: &mut JITModule,
    id: cranelift_module::FuncId,
    signature: &Signature,
    body_id: cranelift_module::FuncId,
    body_signature: &Signature,
    ir: &TraceIr,
) -> Result<(), Box<dyn std::fmt::Display>> {
    let function = ir.function.as_ref().expect("a function trace");
    let mut context = module.make_context();
    let mut function_context = FunctionBuilderContext::new();
    context.func.signature = signature.clone();
    context.func.name = UserFuncName::user(0, id.as_u32());
    {
        let mut builder = FunctionBuilder::new(&mut context.func, &mut function_context);
        let body = module.declare_func_in_func(body_id, builder.func);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        let params = builder.block_params(entry).to_vec();
        let (inputs, control, area) = (params[0], params[1], params[2]);
        let slot_size = mem::size_of::<TraceInputSlot>();
        for (index, input) in ir.entry.inputs.iter().enumerate() {
            let Some(required) = entry_tag(input.ty) else {
                return Err(Box::new("an entry input has no tag"));
            };
            let tag = builder.ins().load(
                types::I64,
                MemFlags::new(),
                inputs,
                offset(index * slot_size + mem::offset_of!(TraceInputSlot, tag)),
            );
            let matches = builder.ins().icmp_imm(IntCC::Equal, tag, required);
            let accepted = builder.create_block();
            let mismatch = builder.create_block();
            builder.set_cold_block(mismatch);
            builder.ins().brif(matches, accepted, &[], mismatch, &[]);
            builder.switch_to_block(mismatch);
            let code = builder.ins().iconst(
                types::I64,
                -i64::try_from(index).expect("input index fits") - 1,
            );
            builder.ins().return_(&[code]);
            builder.switch_to_block(accepted);
        }
        let cap = builder.ins().load(
            types::I64,
            MemFlags::trusted(),
            control,
            offset(mem::offset_of!(TraceControl, depth_cap)),
        );
        let fuel = builder.ins().load(
            types::I64,
            MemFlags::trusted(),
            control,
            offset(mem::offset_of!(TraceControl, fuel)),
        );
        for (field, value) in [
            (mem::offset_of!(TraceControl, inputs), inputs),
            (mem::offset_of!(TraceControl, area), area),
        ] {
            builder
                .ins()
                .store(MemFlags::trusted(), value, control, offset(field));
        }
        let mut arguments = vec![control, cap, fuel];
        // The frame's tail call count.
        if let Some(position) = ir
            .external
            .iter()
            .position(|slot| *slot == ExternalSlot::TailCalls)
        {
            let register = usize::from(ir.frames[0].register_count) - ir.external.len() + position;
            let Some(index) = ir
                .entry
                .inputs
                .iter()
                .position(|input| usize::from(input.register) == register)
            else {
                return Err(Box::new("the tail call count is no entry input"));
            };
            arguments.push(builder.ins().load(
                types::I64,
                MemFlags::new(),
                inputs,
                offset(index * slot_size + mem::offset_of!(TraceInputSlot, bits)),
            ));
        }
        for (register, ty) in function.parameters.iter().enumerate() {
            let Some(index) = ir
                .entry
                .inputs
                .iter()
                .position(|input| usize::from(input.register) == register)
            else {
                return Err(Box::new("a parameter is no entry input"));
            };
            let at = offset(index * slot_size + mem::offset_of!(TraceInputSlot, bits));
            arguments.push(match ty {
                TraceType::Float => builder.ins().load(types::F64, MemFlags::new(), inputs, at),
                TraceType::Boolean => {
                    let bits = builder.ins().load(types::I64, MemFlags::new(), inputs, at);
                    builder.ins().icmp_imm(IntCC::NotEqual, bits, 0)
                }
                TraceType::Nil => builder.ins().iconst(types::I8, 0),
                _ => builder.ins().load(types::I64, MemFlags::new(), inputs, at),
            });
        }
        debug_assert_eq!(arguments.len(), body_signature.params.len());
        let call = builder.ins().call(body, &arguments);
        let results = builder.inst_results(call).to_vec();
        let status = results[0];
        let returned = builder.ins().icmp_imm(IntCC::SignedGreaterThan, status, 0);
        let done = builder.create_block();
        let left = builder.create_block();
        builder.ins().brif(returned, done, &[], left, &[]);
        builder.switch_to_block(done);
        builder.ins().store(
            MemFlags::trusted(),
            status,
            control,
            offset(mem::offset_of!(TraceControl, fuel)),
        );
        if let Some(ty) = function.result {
            let value = results[1];
            let bits = match ty {
                TraceType::Float => Some(builder.ins().bitcast(types::I64, MemFlags::new(), value)),
                TraceType::Boolean => Some(builder.ins().uextend(types::I64, value)),
                TraceType::Nil | TraceType::Any => None,
                _ => Some(value),
            };
            if let Some(bits) = bits {
                builder.ins().store(MemFlags::new(), bits, area, 0);
            }
        }
        let code = builder.ins().iconst(types::I64, RETURNED_CODE);
        builder.ins().return_(&[code]);
        builder.switch_to_block(left);
        // The exit's index.
        let negated = builder.ins().ineg(status);
        let index = builder.ins().iadd_imm(negated, -1);
        builder.ins().return_(&[index]);
        builder.seal_all_blocks();
        builder.finalize();
    }
    module
        .define_function(id, &mut context)
        .map_err(|error| Box::new(error) as Box<dyn std::fmt::Display>)?;
    module.clear_context(&mut context);
    Ok(())
}

/// Whether `fma` is one instruction on `isa`: always on `AArch64`, the
/// `has_fma` flag on x86-64 (set from the host's CPU), never elsewhere.
fn isa_has_fma(isa: &dyn cranelift_codegen::isa::TargetIsa) -> bool {
    // `isa_flags` builds a vector of every flag; the JIT's ISA is the host's
    // and never changes, so look once instead of on every compile.
    static HAS_FMA: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if cfg!(target_arch = "aarch64") {
        return true;
    }
    *HAS_FMA.get_or_init(|| {
        isa.isa_flags()
            .iter()
            .any(|flag| flag.name == "has_fma" && flag.as_bool() == Some(true))
            // `trunc` is a library call without SSE4.1 on x86-64.
            && isa
                .isa_flags()
                .iter()
                .any(|flag| flag.name == "has_sse41" && flag.as_bool() == Some(true))
    })
}

// Both words of a nil are zero, which a call that leaves out a parameter relies on.
const _: () = assert!(TRACE_TAG_NIL == 0);

/// A table a [`HelperId::TABLE_NEW`] call made, as the code that follows it
/// can fill it.
struct MadeTable {
    /// The address of the table's first value slot, which the helper left in
    /// the control block: zero when the table has no shape.
    slots: Value,
    /// What the call named: the shape, whether it made the array part
    /// ([`Lowerer::prefills_array`]) and how long.
    made: super::made_table::MadeShape,
    /// The slots the code stored into in place so far.
    filled: Vec<usize>,
    /// Whether a store by helper may have moved the slots: a key the shape
    /// lacks (an array element, say) can grow the table's storage, so the
    /// address in `slots` is stale and later stores go through the helper.
    moved: bool,
}

/// The fast version of the root's loop while it is lowered: the range plan
/// it relies on, and the integer copies of its float inductions.
struct FastLoop {
    plan: RangePlan,
    /// The integer copy of each float induction of the plan, which every
    /// parameter of the induction equals: the induction, and the copy's
    /// value in the iteration being lowered. In the order of the fast
    /// header's extra parameters.
    shadows: Vec<(usize, Value)>,
    /// Per induction, an invariant step as an integer (`None` for a
    /// constant one).
    steps: Vec<Option<Value>>,
    /// Each shift of the plan as an integer.
    shifts: Vec<(RangeShift, Value)>,
    /// Per induction, the root parameter's value in the iteration being
    /// lowered.
    roots: Vec<Value>,
    /// The row addresses the proven accesses of the iteration being lowered
    /// already computed, so each index is scaled and added to its array's
    /// base once.
    rows: Vec<FastRow>,
}

/// A row address of the fast version: `row` is `base + index * 16` for the
/// index that `copy` (plus `shift`) gives in the iteration being lowered.
#[derive(Clone, Copy)]
struct FastRow {
    base: Value,
    copy: Value,
    shift: Option<Value>,
    row: Value,
}

/// An array slot an access of the code being lowered already reached: the
/// key is an integral number inside the array, the slot is at `slot`, and it
/// holds a value of type `ty`. It holds until an operation that may move
/// array storage or change the type of a slot (see [`keeps_slots`]).
#[derive(Clone, Copy)]
struct SlotFact {
    binding: u32,
    key: ValueId,
    slot: Value,
    ty: TraceType,
}

/// Whether `op` leaves every array slot where it is, holding a value of the
/// type it held. An in-place store keeps the slot's type by construction. A
/// read keeps everything. Anything else may call the VM.
pub(super) const fn keeps_slots(op: &TraceOp) -> bool {
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
            | TraceOp::Truthy { .. }
            | TraceOp::Not { .. }
            | TraceOp::BoolAnd { .. }
            | TraceOp::Select { .. }
            | TraceOp::NumberModulo { .. }
            | TraceOp::NumberTag { .. }
            | TraceOp::GuardType { .. }
            | TraceOp::GuardBranch { .. }
            | TraceOp::ArrayLoad { .. }
            | TraceOp::LaneArrayLoad { .. }
            | TraceOp::NumberLoad { .. }
            | TraceOp::ArrayStore { .. }
            | TraceOp::ArrayOverwrite { .. }
            | TraceOp::LaneArrayStore { .. }
            | TraceOp::ArrayNext { .. }
            | TraceOp::FieldLoad { .. }
            | TraceOp::FieldStore { .. }
            | TraceOp::TableLength { .. }
            | TraceOp::StringLength { .. }
            | TraceOp::StringByte { .. }
            | TraceOp::PhantomTable { .. }
    )
}

/// The block of an exit tail (see `share_exit_stores` in the compile
/// module), made by the first exit that jumps to it, with the value each of
/// its slots stored there: `None` for an entry input, which the tail loads
/// from its slot. An exit whose values differ (a loop parameter of the outer
/// header against one of the inner header) makes another variant of the tail.
struct ExitTail {
    block: Block,
    values: Vec<Option<Value>>,
    /// The variant of the next tail on the chain this block jumps to.
    next: Option<(usize, usize)>,
}

/// A `t[k] = nil` the emitter holds back for the `t[k] = v` after it (see
/// [`fused_delete_stores`] and [`Lowerer::hash_overwrite`]).
struct PendingDelete {
    args: Vec<ValueId>,
    guard: GuardId,
    /// The guard of the store the delete fuses with.
    store: GuardId,
    /// The table binding the delete's native code would use.
    binding: u32,
}

/// The delete and store pairs of a root member that [`Lowerer::hash_overwrite`]
/// may fuse, by the delete's guard and holding the store's guard. A pair is a
/// `DELETE_STRING` of a table and a key, then a `SET_REVIVE` of the same table
/// and key with a number, integer or boolean value. Only pure, guard-free
/// operations may lie between them: a guard or any other effect between would
/// expose a state without the delete.
///
/// `SET_PRESENT` is not a partner: after the delete the key is nil, and
/// `SET_PRESENT` must not store into a nil entry.
fn fused_delete_stores(ir: &TraceIr) -> std::collections::HashMap<GuardId, GuardId> {
    let mut fused = std::collections::HashMap::new();
    for (index, instruction) in ir.body.iter().enumerate() {
        let TraceOp::CallHelper {
            helper: HelperId::DELETE_STRING,
            args: delete_args,
            guard,
            ..
        } = &instruction.op
        else {
            continue;
        };
        for later in &ir.body[index + 1..] {
            match &later.op {
                TraceOp::Constant(_)
                | TraceOp::Move { .. }
                | TraceOp::Arithmetic { .. }
                | TraceOp::Compare { .. }
                | TraceOp::Select { .. } => continue,
                TraceOp::CallHelper {
                    helper: HelperId::SET_REVIVE,
                    args,
                    guard: store,
                    ..
                } if args.len() == 3
                    && args[..2] == delete_args[..]
                    && matches!(
                        ir.value_type(args[2]),
                        Some(TraceType::Float | TraceType::Integer | TraceType::Boolean)
                    ) =>
                {
                    fused.insert(*guard, *store);
                }
                _ => {}
            }
            break;
        }
    }
    fused
}

/// The lane of a prepared pattern, the lane of its template and the kind of
/// object (see [`Lowerer::prepared_index`]).
type PreparedKey = (u64, u64, u8);

/// A `string.find` or `string.match` of a single-run pattern that the trace
/// scans itself (see [`Lowerer::call_inline_run`]).
struct RunPlan {
    /// The address of the run's class, a table by byte value (1 for a
    /// member) that the prepared pattern owns.
    table: usize,
    anchored: bool,
    results: RunResults,
    /// The initial position, when the call gives an integer one.
    init: Option<ValueId>,
}

/// What the helper of a [`RunPlan`] gives.
#[derive(Clone, Copy, Eq, PartialEq)]
enum RunResults {
    /// The start of the match (`STRING_FIND`).
    Start,
    /// The start and the end (a `string.find` that wants both).
    Span,
    /// The start, the end and the text (a `string.find` of a captured run).
    SpanText,
    /// The matched text (a `string.match`).
    Whole,
    /// Nothing, and a miss when there is a match.
    Absent,
}

impl RunResults {
    const fn count(self) -> usize {
        match self {
            Self::Start | Self::Whole => 1,
            Self::Span => 2,
            Self::SpanText => 3,
            Self::Absent => 0,
        }
    }
}

// Deliberate shape: the bools are independent lowering facts about the
// trace being emitted, not one state machine.
#[allow(clippy::struct_excessive_bools)]
/// One entry of [`Lowerer::loop_stack`]: member index, header block, and the
/// register and type of each loop parameter.
type LoopFrame = (usize, Block, Vec<(u16, TraceType)>);

#[allow(
    clippy::struct_excessive_bools,
    reason = "independent capability and effect flags of the lowering, read one at a time"
)]
struct Lowerer<'a, 'f> {
    builder: &'a mut FunctionBuilder<'f>,
    /// Every trace of the tree: the root, then its sides.
    members: &'a [LoweredMember<'a>],
    /// The trace being lowered: the root, or a side while one is nested in
    /// it.
    ir: &'a TraceIr,
    /// Index of `ir` in `members`.
    member: usize,
    plan: &'a ExitPlan,
    pow: FuncRef,
    /// Imports of the out-of-line `%` helpers, when the trace needs them.
    fmod53: Option<FuncRef>,
    fmod54: Option<FuncRef>,
    /// Whether the target ISA lowers `fma` to one instruction. Without it
    /// Cranelift emits a libm call, so the inline exact `%` path is skipped.
    has_fma: bool,
    admit_table: FuncRef,
    /// [`Self::admit_table`] behind a stub that preserves every register, for
    /// the cold path of the lane index probe.
    admit_table_stub: Option<FuncRef>,
    /// Fills the row of a lane a helper made (see `trace_ensure_row`).
    ensure_row: FuncRef,
    /// Makes room for a slot in the array part of a table
    /// (`oximond::runtime::native_table_reserve`).
    reserve_array: FuncRef,
    /// The register-preserving stub that makes a table set, in a tree that
    /// holds one.
    table_set: Option<FuncRef>,
    /// Whether table sets call the VM directly (see `array_set`).
    direct_set: bool,
    /// The helper that looks up a string in a table, in a tree that holds a
    /// [`TraceOp::HashLoad`].
    hash_get: Option<FuncRef>,
    /// The lookup of a number key, through the register-preserving stub.
    hash_get_number: Option<FuncRef>,
    /// The register-preserving stubs of cold helper calls, by argument count
    /// (see [`helper_stub_arity`]).
    helper_stubs: std::collections::BTreeMap<usize, FuncRef>,
    /// The Cranelift value of each trace value.
    values: Vec<Option<Value>>,
    /// The results after the first of each helper call lowered so far, loaded
    /// from the control block right after the call, by the call's value.
    helper_results: std::collections::HashMap<ValueId, Vec<Value>>,
    /// The tables made by a [`HelperId::TABLE_NEW`] call lowered so far in the
    /// current member, by the call's value, and the fields of each that a
    /// [`HelperId::TABLE_INIT`] stored in place.
    made_tables: std::collections::HashMap<ValueId, MadeTable>,
    /// The array step just lowered and the address of the slot it found: the
    /// element read that follows it reads that slot, with no second bounds
    /// check. Any other instruction forgets it.
    next_slot: Option<(ValueId, u32, (Value, Value))>,
    /// Whether a member of the tree has an operation that may move hash
    /// storage.
    moves_hash: bool,
    /// Whether the tree looks up hashes and moves none, so a lookup probes
    /// the string index the VM published for its binding
    /// ([`Self::hash_probe`]).
    native_probe: bool,
    /// The scalar of each value a `Constant` defined so far.
    constants: Vec<Option<TraceScalar>>,
    /// Whether the options allow preparing constant strings
    /// ([`TracingOptFlags::STRING_CONSTANTS`]).
    string_constants: bool,
    /// The objects prepared for the code's helper calls.
    prepared: Vec<Prepared>,
    /// The index in `prepared` of the object of each pattern lane and kind
    /// (none for one that fails to validate).
    prepared_patterns: std::collections::HashMap<PreparedKey, Option<usize>>,
    /// The index in `prepared` of the `string.gmatch` pattern each iterator
    /// value was made from, for the steps that read it.
    gmatch_patterns: std::collections::HashMap<ValueId, usize>,
    exit_area: Option<Value>,
    control: Option<Value>,
    /// The address of the run's field rows, loaded once at entry.
    rows: Option<Value>,
    /// The words of a row: the fields of the trace's tables, or the words of a
    /// closure's row when the code reads and writes callee cells itself
    /// (see [`closure_cell_use`]).
    row_stride: usize,
    /// Whether each row ends in a presence mask (see [`rows_masked`]).
    row_masked: bool,
    /// Whether a table loaded out of an array is looked up in the run's lane
    /// index before the dispatcher is asked (see `admit_table_probe`).
    lane_probe: bool,
    /// [`CLOSURE_ROW_WORDS`] when the code does, and zero otherwise.
    row_words: usize,
    /// The root's loop header, which every link jumps back to.
    header: Option<Block>,
    /// The block that native exits of a root share: it calls the side the
    /// exit's cell holds (see [`Lowerer::emit_native_dispatch`]). Made by the
    /// first native exit.
    native_dispatch: Option<Block>,
    /// The link slots of the root when its exits call sides natively (see
    /// [`Lowerer::native_link_slots`]), decided once before anything is
    /// lowered so that every exit agrees.
    native_link: Option<Box<[(usize, TraceType)]>>,
    /// The block a native link enters the root again through. It holds the
    /// code of the entry, with the run's three buffers as block parameters.
    native_entry: Option<Block>,
    /// The root's own loop-back target and which loop parameters it passes,
    /// when a tree gives the root parameters only its sides change: see
    /// [`Lowerer::lower`].
    inner: Option<(Block, Vec<usize>)>,
    /// The Cranelift value of each entry input of the root, by register.
    /// They dominate every block, so a side can use one its parent lacks.
    root_inputs: std::collections::HashMap<u16, Value>,
    /// The input slot of each entry input of the root that holds nil or a
    /// number (type `Any`), by register. Its tag is not checked at entry: a
    /// type guard of the member that reads it loads the tag and the bits
    /// from the slot (see [`Self::guard_any_input`]).
    any_slots: std::collections::HashMap<u16, usize>,
    /// The two words the entry loaded for each table binding of the root: an
    /// array's address and length, a field slot's address, or a length.
    bound: Vec<(Value, Value)>,
    /// An array binding's length as a float, converted once at entry for
    /// float keys. `None` for the other bindings.
    bound_length_float: Vec<Option<Value>>,
    /// The address of the first binding slot of the run's input buffer.
    words: Option<Value>,
    /// Whether array words are loaded from `words` instead of once at entry:
    /// the tree holds a [`TraceOp::ArraySet`], whose call may grow a table
    /// and rewrites the words.
    reload_arrays: bool,
    /// The loop parameter whose array step needs no check of its key (see
    /// [`trusted_array_key`]).
    trusted_key: Option<TrustedKey>,
    /// Whether the root's loop reads array words from loads made in the
    /// header, just before the loop, instead of the values the entry loaded
    /// (see [`Self::load_fresh_words`]).
    fresh_words: bool,
    /// The array words loaded since the last point they may have changed (a
    /// loop header, the end of a set), by binding. The values dominate the
    /// code that follows them until the cache is cleared.
    array_cache: Vec<Option<(Value, Value, Value)>>,
    /// The array words of tables met by lane ([`Self::lane_array_words`]),
    /// by lane, since the last point they may have changed. Cleared with
    /// `array_cache`, and wherever `row_addresses` is, since they are read
    /// through the row.
    lane_array_cache: Vec<(Value, (Value, Value, Value))>,
    /// For a counted trace: the budget at the start of the iteration being
    /// lowered, which every exit of the iteration reports. The entry's value
    /// until the header, then the header's parameter.
    budget: Option<Value>,
    /// Test hook: the guard branch (member, guard index) always takes its
    /// exit, or its side trace.
    forced_guard: Option<(usize, usize)>,
    /// Whether a jump into a side trace counts itself in the control block.
    count_links: bool,
    /// For a function trace: what an activation needs (see
    /// [`FunctionLowering`]). The fuel then lives in the control block, which
    /// every activation spends, and exits leave it there.
    function: Option<FunctionLowering>,
    /// In an activation of a function trace: how many activations the run
    /// may still nest below it.
    depth_left: Option<Value>,
    /// In an activation of a function trace: the fuel left, which each
    /// self call's callee hands back.
    activation_fuel: Option<Value>,
    /// In an activation of a function trace, once a self call that
    /// dominates the code being lowered found room for its callee: the
    /// callee's `left`.
    deeper: Option<Value>,
    /// In an activation of a function trace: the root's entry inputs of cells
    /// and globals, by value, with their input slot. Each use loads the slot
    /// where it is, so an exit path's use costs the hot path nothing.
    lazy: std::collections::HashMap<ValueId, (usize, TraceType)>,
    /// The slot and type of each entry input of the root loop trace that the
    /// code of the exits and of the sides may load again (see
    /// [`Self::reload_inputs`]).
    entry_slots: std::collections::HashMap<ValueId, (usize, TraceType)>,
    /// The address of the entry input slots.
    entry_inputs: Option<Value>,
    /// Per exit tail of the plan, its block once an exit made it.
    tails: Vec<Vec<ExitTail>>,
    /// Set when the code lowered is a side compiled apart from its tree.
    detached: Option<&'a DetachedSpec<'a>>,
    /// For a side compiled apart: the registers it reads from the extra
    /// input slots, the tree's own and the side's.
    side_extras: Vec<Extra>,
    /// For a side compiled apart: the input slots its link rewrites.
    side_writebacks: Vec<Writeback>,
    /// Whether the root's loop was lowered in two versions (see
    /// [`Self::lower_versions`]).
    versioned_loop: bool,
    /// Per value of the member being lowered: whether it is computed where
    /// it is used rather than where it is defined (see
    /// [`Self::defer_cold_values`]).
    deferred: Vec<bool>,
    /// A fuel value for code lowered out of order, which never reads it.
    member_fuel: Option<Value>,
    /// Deferred values computed in a block, which later uses in the same
    /// block take again.
    rematerialized: std::collections::HashMap<ValueId, (Block, Value)>,
    /// Whether the root's entry inputs are loaded from their slots at each
    /// use: set while the code of an exit or of a side is lowered, which runs
    /// once the loop is left. An input the cold code names would otherwise
    /// stay in a register or a stack slot all through the loop.
    reload_inputs: bool,
    /// Per member, how its bounds-checked keys are formed, for the
    /// diagnostic counters.
    #[cfg(feature = "op-stats")]
    range_facts: Vec<RangeFacts>,
    /// The range proofs of the root's loop, when it is lowered in two
    /// versions (see [`Self::lower_versions`]).
    range_plan: Option<RangePlan>,
    /// Set while the fast version of the root's loop is lowered.
    fast: Option<FastLoop>,
    /// While the slow version of a root lowered in two versions is: its
    /// header and the parameters it carries (see [`Self::link_target`]).
    slow_loop: Option<(Block, Vec<usize>)>,
    /// Whether the side being lowered continues a guard of the root.
    root_side: bool,
    /// The header of the loop side being lowered, which its loop-back jumps
    /// to. `None` in the root and in every other side.
    loop_header: Option<Block>,
    /// The loop sides whose loops enclose the code being lowered, innermost
    /// last: member index, header block, and the register and type of each
    /// loop parameter. A side that ends in `TraceTerminator::LinkLoop` jumps
    /// to one of these headers.
    loop_stack: Vec<LoopFrame>,
    /// For a side of the root: per entry input of the side, the root value
    /// it starts as, if the root's snapshot names one (see
    /// [`Self::fast_link`]).
    side_origins: Vec<Option<ValueId>>,
    /// The slots the accesses lowered since the last operation that may
    /// move or retype them reached, in the current member.
    slots: Vec<SlotFact>,
    /// Whether the tree compiles at the peak tier.
    peak: bool,
    /// The row address each lane got in the straight line of code being
    /// lowered (see [`Self::row_address`]).
    row_addresses: Vec<(Value, Value)>,
    /// The slot address each (lane, field) got, past its guard, since the
    /// last operation that may move or retype a slot (see
    /// [`Self::field_slot`]).
    field_slots: Vec<(Value, u32, Value)>,
    /// Per number load ([`TraceOp::NumberLoad`]) of the current member: the
    /// boolean that says whether the slot held an integer.
    number_tags: std::collections::HashMap<ValueId, Value>,
    /// The delete and store pairs of the root's body being lowered (see
    /// [`fused_delete_stores`]). Empty outside that body.
    fused: std::collections::HashMap<GuardId, GuardId>,
    /// A delete held back for the store it fuses with, if the body has
    /// reached it. Only pure operations come between the two.
    pending_delete: Option<PendingDelete>,
}

impl Lowerer<'_, '_> {
    fn value(&mut self, value: ValueId) -> Value {
        if self.reload_inputs
            && self.member == 0
            && let Some(&(index, ty)) = self.entry_slots.get(&value)
            && let Some(inputs) = self.entry_inputs
        {
            return self.load_entry_input(inputs, index, ty);
        }
        if let Some(lowered) = self.values[value.index()] {
            return lowered;
        }
        if self.deferred.get(value.index()).copied().unwrap_or(false) {
            return self.rematerialize(value);
        }
        let (index, ty) = *self
            .lazy
            .get(&value)
            .expect("a verified trace defines every operand before use");
        self.load_input(index, ty)
    }

    /// The input slot `index` of a function trace's run, as a value of type
    /// `ty`, loaded here.
    fn load_input(&mut self, index: usize, ty: TraceType) -> Value {
        let control = self.control.expect("entry set the control block");
        let flags = MemFlags::trusted().with_readonly();
        let inputs = self.builder.ins().load(
            types::I64,
            flags,
            control,
            offset(mem::offset_of!(TraceControl, inputs)),
        );
        let at = offset(index * mem::size_of::<TraceInputSlot>())
            + offset(mem::offset_of!(TraceInputSlot, bits));
        match ty {
            TraceType::Float => self.builder.ins().load(types::F64, flags, inputs, at),
            TraceType::Boolean => {
                let bits = self.builder.ins().load(types::I64, flags, inputs, at);
                self.builder.ins().icmp_imm(IntCC::NotEqual, bits, 0)
            }
            TraceType::Nil => self.builder.ins().iconst(types::I8, 0),
            _ => self.builder.ins().load(types::I64, flags, inputs, at),
        }
    }

    /// The address of the exit area the current code spills into: a loop
    /// trace's, or the part of a function trace's run area that belongs to
    /// this activation.
    fn spill_area(&mut self) -> Value {
        if let Some(area) = self.exit_area {
            return area;
        }
        let control = self.control.expect("entry set the control block");
        let left = self.depth_left.expect("entry set the depth left");
        let stride = self
            .function
            .as_ref()
            .expect("an activation of a function trace")
            .stride_bytes;
        let base = self.builder.ins().load(
            types::I64,
            MemFlags::trusted(),
            control,
            offset(mem::offset_of!(TraceControl, area)),
        );
        let cap = self.builder.ins().load(
            types::I64,
            MemFlags::trusted(),
            control,
            offset(mem::offset_of!(TraceControl, depth_cap)),
        );
        let depth = self.builder.ins().isub(cap, left);
        let bytes = self.builder.ins().imul_imm(depth, stride);
        self.builder.ins().iadd(base, bytes)
    }

    fn define(&mut self, value: ValueId, lowered: Value) {
        self.values[value.index()] = Some(lowered);
    }

    /// Marks the values of the current member that only exits, sides and
    /// links name, directly or through other such values, and whose
    /// instruction is float arithmetic that cannot fail. The loop's body
    /// then skips them, and each cold use computes the value again where it
    /// is (see [`Self::rematerialize`]): a value a guard's snapshot keeps
    /// alive costs the hot path nothing. Its operands are defined before it,
    /// so they are available wherever it is used.
    fn defer_cold_values(&mut self) {
        let ir = self.ir;
        self.deferred = vec![false; ir.values.len()];
        self.rematerialized.clear();
        if self.function.is_some() {
            return;
        }
        let mut hot = Vec::new();
        ir.mark_cold_values(|_| true, &mut self.deferred, &mut hot);
    }

    /// Computes deferred `value` at the current point (see
    /// [`Self::defer_cold_values`]). The result is not kept: another use
    /// may be on a path this point does not dominate.
    fn rematerialize(&mut self, value: ValueId) -> Value {
        let block = self.builder.current_block();
        if let Some(&(made_in, made)) = self.rematerialized.get(&value)
            && Some(made_in) == block
        {
            return made;
        }
        let ir = self.ir;
        let ValueDef::Instruction(index) = ir.values[value.index()].def else {
            unreachable!("only body instructions are deferred");
        };
        let fuel = self.member_fuel.expect("a member is being lowered");
        let pending = self.next_slot.take();
        self.lower_instruction(&ir.body[index].op, Some(value), fuel, &[])
            .expect("deferred arithmetic lowers");
        self.next_slot = pending;
        let made = self.values[value.index()]
            .take()
            .expect("deferred arithmetic has a result");
        // A lowering that split the block ends in another one.
        if let Some(block) = self.builder.current_block() {
            self.rematerialized.insert(value, (block, made));
        }
        made
    }

    /// For a call of a `string.find` or `string.match` helper whose pattern
    /// is a root-string constant: the address of the helper's specialized
    /// twin and of the pattern prepared for it. `None` leaves the call to the
    /// generic helper: the flag is off, the pattern is not constant, it fails
    /// to validate, or the helper has no twin.
    fn prepared_pattern(&mut self, helper: HelperId, args: &[ValueId]) -> Option<(usize, usize)> {
        if !self.string_constants {
            return None;
        }
        let twin = crate::jit::trace::helpers::prepared_helper_address(helper)?;
        if matches!(
            helper,
            HelperId::STRING_FORMAT
                | HelperId::STRING_FORMAT_NATIVE
                | HelperId::STRING_FORMAT_LUAJIT
        ) {
            let (lane, root) = self.root_string(*args.first()?)?;
            let index = self.prepared_index((lane, u64::MAX, 3), (root, None))?;
            return Some((twin, self.prepared[index].address()));
        }
        if helper.is_gmatch_step() {
            // The step reads the pattern its iterator was made from. An
            // iterator that entered the trace is guessed to come from the
            // trace's one constant `gmatch` pattern; the step reads the
            // iterator's own pattern when the guess is wrong.
            let index = match self.gmatch_patterns.get(args.first()?) {
                Some(index) => *index,
                None => self.guessed_gmatch()?,
            };
            return Some((twin, self.prepared[index].address()));
        }
        let (pattern, root) = self.root_string(*args.get(1)?)?;
        // The plain-text helpers search text with no magic byte themselves,
        // and a plain-flag find is the native's.
        if !find_uses_prepared_pattern(helper, self.ir.root_constants.get(root)?, args.len()) {
            return None;
        }
        let index = if helper == HelperId::STRING_GSUB {
            let template = args.get(2).and_then(|arg| self.root_string(*arg));
            let key = (pattern, template.map_or(u64::MAX, |(lane, _)| lane), 2);
            self.prepared_index(key, (root, template.map(|(_, root)| root)))?
        } else {
            self.prepared_index((pattern, u64::MAX, 0), (root, None))?
        };
        Some((twin, self.prepared[index].address()))
    }

    /// For a `string.find` or `string.match` helper call whose pattern is a
    /// root-string constant that is a single run of one byte class (`%d+`,
    /// `[%w_]+`, `^%a+`), or one captured whole (`(%d+)`), and whose subject and initial position the emitted
    /// code can read: the plan to scan the subject in the trace itself. `None`
    /// leaves the call to its helper (or the helper's twin).
    fn inline_run_plan(
        &mut self,
        helper: HelperId,
        args: &[ValueId],
        guard: GuardId,
    ) -> Option<RunPlan> {
        if !self.string_constants
            || !(2..=3).contains(&args.len())
            || self.forced_guard == Some((self.member, guard.index()))
        {
            return None;
        }
        let candidates = [
            HelperId::STRING_FIND,
            HelperId::string_find_pattern(0, 0),
            HelperId::string_find_prefix(0, 0),
            HelperId::string_find_pattern(1, 1),
            HelperId::string_find_prefix(1, 1),
            HelperId::string_match(1, 1),
            HelperId::string_match_prefix(1, 1),
            HelperId::STRING_FIND_ABSENT,
            HelperId::STRING_MATCH_ABSENT,
        ];
        if !candidates.contains(&helper) {
            return None;
        }
        let subject = args[0];
        if !matches!(
            self.ir.value_type(subject)?,
            TraceType::String | TraceType::StringRef
        ) || matches!(
            self.constants[subject.index()],
            Some(TraceScalar::String(_))
        ) {
            return None;
        }
        let init = match args.get(2) {
            None => None,
            Some(&init) => match self.ir.value_type(init)? {
                TraceType::Integer | TraceType::Float => Some(init),
                TraceType::Nil => None,
                _ => return None,
            },
        };
        let (pattern, root) = self.root_string(args[1])?;
        if !find_uses_prepared_pattern(helper, self.ir.root_constants.get(root)?, args.len()) {
            return None;
        }
        let index = self.prepared_index((pattern, u64::MAX, 4), (root, None))?;
        let Prepared::RunClass(class) = &self.prepared[index] else {
            return None;
        };
        let (table, shape) = (class.table.as_ptr() as usize, class.shape);
        // A find shaped for another capture count misses, as its helper does.
        let results = if helper == HelperId::STRING_FIND {
            RunResults::Start
        } else if helper == HelperId::string_find_prefix(0, 0)
            || (helper == HelperId::string_find_pattern(0, 0) && !shape.captured)
        {
            RunResults::Span
        } else if helper == HelperId::string_find_pattern(1, 1)
            || helper == HelperId::string_find_prefix(1, 1)
        {
            if !shape.captured {
                return None;
            }
            RunResults::SpanText
        } else if helper == HelperId::string_match(1, 1)
            || helper == HelperId::string_match_prefix(1, 1)
        {
            RunResults::Whole
        } else if helper == HelperId::STRING_FIND_ABSENT || helper == HelperId::STRING_MATCH_ABSENT
        {
            RunResults::Absent
        } else {
            return None;
        };
        if helper.info()?.results.len() != results.count() {
            return None;
        }
        Some(RunPlan {
            table,
            anchored: shape.anchored,
            results,
            init,
        })
    }

    /// A `string.find` or `string.match` of a single-run pattern, scanned in
    /// the emitted code. A subject that is not a short string of the bank (a
    /// long string, a root constant held in a loop-carried value), an initial
    /// position the code does not judge, and a host that gives no bank go to
    /// the helper's twin. A result other than the recorded one leaves through
    /// the guard, as the helper's miss does. A match's text is made by the
    /// `string.sub` helper, which the interpreter's own substring path backs.
    #[allow(clippy::too_many_lines)]
    fn call_inline_run(
        &mut self,
        plan: &RunPlan,
        helper: HelperId,
        args: &[ValueId],
        result: Option<ValueId>,
        guard: GuardId,
        fuel: Value,
    ) -> Result<Option<Value>, TraceCompileError> {
        let control = self.control.expect("entry set the control block");
        let count = plan.results.count();
        let slow = self.builder.create_block();
        self.builder.set_cold_block(slow);
        let merge = self.builder.create_block();
        for _ in 0..count {
            self.builder.append_block_param(merge, types::I64);
        }
        let subject_ty = self
            .ir
            .value_type(args[0])
            .expect("a verified trace types every operand");
        let subject = self.value(args[0]);
        // The subject's value: a borrowed string is the address of one, a
        // lane is an index into the bank.
        let value_at = if subject_ty == TraceType::StringRef {
            subject
        } else {
            let local = self.builder.ins().icmp_imm(
                IntCC::UnsignedLessThan,
                subject,
                ROOT_STRING_LANE_BASE.cast_signed(),
            );
            let bank = self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                control,
                offset(mem::offset_of!(TraceControl, bank)),
            );
            let banked = self.builder.ins().icmp_imm(IntCC::NotEqual, bank, 0);
            let ready = self.builder.ins().band(local, banked);
            let have = self.builder.create_block();
            self.builder.ins().brif(ready, have, &[], slow, &[]);
            self.builder.switch_to_block(have);
            let at = self.builder.ins().ishl_imm(subject, 4);
            self.builder.ins().iadd(bank, at)
        };
        let payload = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            value_at,
            NATIVE_VALUE_PAYLOAD_OFFSET,
        );
        let data = self.builder.ins().iadd_imm(
            payload,
            i64::try_from(NATIVE_STRING_STORAGE_OFFSET).expect("small"),
        );
        let form = self.builder.ins().load(types::I8, MemFlags::new(), data, 0);
        let inline =
            self.builder
                .ins()
                .icmp_imm(IntCC::Equal, form, i64::from(NATIVE_STRING_INLINE_TAG));
        let short = self.builder.create_block();
        self.builder.ins().brif(inline, short, &[], slow, &[]);
        self.builder.switch_to_block(short);
        let length = self
            .builder
            .ins()
            .uload8(types::I64, MemFlags::new(), data, 1);
        let bytes = self.builder.ins().iadd_imm(data, 2);

        // Where the search starts: the interpreter's rule for a position in
        // `-2^31 <= init <= length + 1`. One past the end is a search that
        // finds nothing. Beyond it the profiles differ, so the helper judges.
        let first = match plan.init {
            None => self.builder.ins().iconst(types::I64, 0),
            Some(init) => {
                let is_float = self.ir.value_type(init) == Some(TraceType::Float);
                let init = self.value(init);
                // A float position counts when it is a whole number (the
                // interpreter's rule); the range test below bounds it.
                let (init, whole) = if is_float {
                    let whole = self.builder.ins().fcvt_to_sint_sat(types::I64, init);
                    let back = self.builder.ins().fcvt_from_sint(types::F64, whole);
                    (
                        whole,
                        Some(self.builder.ins().fcmp(FloatCC::Equal, back, init)),
                    )
                } else {
                    (init, None)
                };
                let biased = self.builder.ins().iadd_imm(init, 1 << 31);
                let fits = self
                    .builder
                    .ins()
                    .icmp_imm(IntCC::UnsignedLessThan, biased, 1 << 32);
                let negative = self.builder.ins().icmp_imm(IntCC::SignedLessThan, init, 0);
                let from_end = self.builder.ins().iadd(length, init);
                let from_end = self.builder.ins().iadd_imm(from_end, 1);
                let position = self.builder.ins().select(negative, from_end, init);
                let past = self.builder.ins().iadd_imm(length, 1);
                let within = self
                    .builder
                    .ins()
                    .icmp(IntCC::SignedLessThanOrEqual, position, past);
                let judged = self.builder.ins().band(fits, within);
                let judged = match whole {
                    Some(whole) => self.builder.ins().band(judged, whole),
                    None => judged,
                };
                let go = self.builder.create_block();
                self.builder.ins().brif(judged, go, &[], slow, &[]);
                self.builder.switch_to_block(go);
                let one = self.builder.ins().iconst(types::I64, 1);
                let position = self.builder.ins().smax(position, one);
                self.builder.ins().iadd_imm(position, -1)
            }
        };

        let table = self.builder.ins().iconst(
            types::I64,
            i64::try_from(plan.table).expect("address fits i64"),
        );
        let scan = self.builder.create_block();
        self.builder.append_block_param(scan, types::I64);
        let test = self.builder.create_block();
        let step = (!plan.anchored).then(|| self.builder.create_block());
        let run = self.builder.create_block();
        self.builder.append_block_param(run, types::I64);
        self.builder.append_block_param(run, types::I64);
        let run_test = self.builder.create_block();
        let run_step = self.builder.create_block();
        let missed = self.builder.create_block();
        let hit = self.builder.create_block();
        self.builder.append_block_param(hit, types::I64);
        self.builder.append_block_param(hit, types::I64);
        let absent = plan.results == RunResults::Absent;
        self.builder
            .set_cold_block(if absent { hit } else { missed });
        self.builder.ins().jump(scan, &[BlockArg::Value(first)]);

        // The first byte of the class at or after the start (only at the
        // start for an anchored pattern).
        self.builder.switch_to_block(scan);
        let at = self.builder.block_params(scan)[0];
        let more = self.builder.ins().icmp(IntCC::UnsignedLessThan, at, length);
        self.builder.ins().brif(more, test, &[], missed, &[]);
        self.builder.switch_to_block(test);
        let address = self.builder.ins().iadd(bytes, at);
        let byte = self
            .builder
            .ins()
            .uload8(types::I64, MemFlags::new(), address, 0);
        let member = self.builder.ins().iadd(table, byte);
        let member = self
            .builder
            .ins()
            .uload8(types::I64, MemFlags::new(), member, 0);
        let on_miss = step.unwrap_or(missed);
        let after = self.builder.ins().iadd_imm(at, 1);
        self.builder.ins().brif(
            member,
            run,
            &[BlockArg::Value(at), BlockArg::Value(after)],
            on_miss,
            &[],
        );
        if let Some(step) = step {
            self.builder.switch_to_block(step);
            self.builder.ins().jump(scan, &[BlockArg::Value(after)]);
        }

        // The end of the run.
        self.builder.switch_to_block(run);
        let (start, end) = (
            self.builder.block_params(run)[0],
            self.builder.block_params(run)[1],
        );
        let more = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, end, length);
        self.builder.ins().brif(
            more,
            run_test,
            &[],
            hit,
            &[BlockArg::Value(start), BlockArg::Value(end)],
        );
        self.builder.switch_to_block(run_test);
        let address = self.builder.ins().iadd(bytes, end);
        let byte = self
            .builder
            .ins()
            .uload8(types::I64, MemFlags::new(), address, 0);
        let member = self.builder.ins().iadd(table, byte);
        let member = self
            .builder
            .ins()
            .uload8(types::I64, MemFlags::new(), member, 0);
        self.builder.ins().brif(
            member,
            run_step,
            &[],
            hit,
            &[BlockArg::Value(start), BlockArg::Value(end)],
        );
        self.builder.switch_to_block(run_step);
        let longer = self.builder.ins().iadd_imm(end, 1);
        self.builder
            .ins()
            .jump(run, &[BlockArg::Value(start), BlockArg::Value(longer)]);

        // No run: the recorded call found one unless it is the absent kind.
        self.builder.switch_to_block(missed);
        if absent {
            self.builder.ins().jump(merge, &[]);
        } else {
            self.leave(guard, fuel)?;
        }

        // A run.
        self.builder.switch_to_block(hit);
        let (start, end) = (
            self.builder.block_params(hit)[0],
            self.builder.block_params(hit)[1],
        );
        let gave: Vec<BlockArg> = match plan.results {
            RunResults::Absent => {
                self.leave(guard, fuel)?;
                Vec::new()
            }
            RunResults::Start => {
                let first = self.builder.ins().iadd_imm(start, 1);
                vec![BlockArg::Value(first)]
            }
            RunResults::Span => {
                let first = self.builder.ins().iadd_imm(start, 1);
                vec![BlockArg::Value(first), BlockArg::Value(end)]
            }
            RunResults::Whole => {
                let first = self.builder.ins().iadd_imm(start, 1);
                let lane = self.run_text(args[0], subject_ty, (first, end), guard, fuel)?;
                vec![BlockArg::Value(lane)]
            }
            RunResults::SpanText => {
                let first = self.builder.ins().iadd_imm(start, 1);
                let lane = self.run_text(args[0], subject_ty, (first, end), guard, fuel)?;
                vec![
                    BlockArg::Value(first),
                    BlockArg::Value(end),
                    BlockArg::Value(lane),
                ]
            }
        };
        if !absent {
            self.builder.ins().jump(merge, &gave);
        }

        // The helper's twin.
        self.builder.switch_to_block(slow);
        let first = self.call_helper_call(helper, args, result, guard, fuel, false)?;
        let mut all: Vec<BlockArg> = Vec::with_capacity(count);
        if let Some(first) = first {
            all.push(BlockArg::Value(first));
            if let Some(rest) = result.and_then(|result| self.helper_results.get(&result)) {
                all.extend(rest.iter().map(|value| BlockArg::Value(*value)));
            }
        }
        while all.len() < count {
            all.push(BlockArg::Value(self.builder.ins().iconst(types::I64, 0)));
        }
        self.builder.ins().jump(merge, &all);
        self.builder.switch_to_block(merge);
        let merged = self.builder.block_params(merge).to_vec();
        if let (Some(result), true) = (result, merged.len() > 1) {
            self.helper_results.insert(result, merged[1..].to_vec());
        }
        Ok(merged.first().copied())
    }

    /// The text of bytes `from..=to` (one based) of the string in `subject`,
    /// made by the `string.sub` helper. Leaves through the guard when the
    /// helper misses.
    fn run_text(
        &mut self,
        subject: ValueId,
        ty: TraceType,
        (from, to): (Value, Value),
        guard: GuardId,
        fuel: Value,
    ) -> Result<Value, TraceCompileError> {
        let control = self.control.expect("entry set the control block");
        let address = crate::jit::trace::helpers::helper_address(HelperId::STRING_SUB)
            .ok_or(TraceCompileError::NativeBackendRejected)?;
        let value = self.value(subject);
        let (tag, bits) = if ty == TraceType::StringRef {
            (
                self.builder
                    .ins()
                    .iconst(types::I64, TRACE_TAG_BORROWED_STRING),
                value,
            )
        } else {
            self.scalar_word(ty, value)
        };
        let context = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(mem::offset_of!(TraceControl, host_context)),
        );
        let out_at = mem::offset_of!(TraceControl, helper_out);
        let out = self
            .builder
            .ins()
            .iadd_imm(control, i64::from(offset(out_at)));
        let three = self.builder.ins().iconst(types::I64, 3);
        let integer = self.builder.ins().iconst(types::I64, TRACE_TAG_INTEGER);
        let arguments = [context, out, three, tag, bits, integer, from, integer, to];
        let mut signature = Signature::new(self.builder.func.signature.call_conv);
        for _ in &arguments {
            signature.params.push(AbiParam::new(types::I64));
        }
        signature.returns.push(AbiParam::new(types::I64));
        let callee = self.native_address(address);
        let signature = self.builder.import_signature(signature);
        let call = self
            .builder
            .ins()
            .call_indirect(signature, callee, &arguments);
        let status = self.builder.inst_results(call)[0];
        let answered = self.builder.ins().icmp_imm(IntCC::Equal, status, 0);
        let pass = self.builder.create_block();
        let fail = self.builder.create_block();
        self.builder.set_cold_block(fail);
        self.builder.ins().brif(answered, pass, &[], fail, &[]);
        self.builder.switch_to_block(fail);
        self.leave(guard, fuel)?;
        self.builder.switch_to_block(pass);
        Ok(self
            .builder
            .ins()
            .load(types::I64, MemFlags::new(), control, offset(out_at)))
    }

    /// The index in `prepared` of the gmatch pattern the tree makes its
    /// iterators from, when every `string.gmatch` call in it (root and sides)
    /// names the same root-string constant.
    fn guessed_gmatch(&mut self) -> Option<usize> {
        let lane = gmatch_constant_lane(self.members.iter().map(|member| member.ir))?;
        let root = root_string_slot(lane)?;
        self.prepared_index((lane, u64::MAX, 1), (root, None))
    }

    /// The lane and root slot of `value` when it is a root-string constant.
    fn root_string(&self, value: ValueId) -> Option<(u64, usize)> {
        let TraceScalar::String(lane) = self.constants[value.index()]? else {
            return None;
        };
        let root = root_string_slot(lane)?;
        Some((lane, root))
    }

    /// The index in `prepared` of the object `key` names (pattern lane,
    /// template lane, kind: 0 for `find` and `match`, 1 for `gmatch`, 2 for
    /// `gsub`, 3 for `format`), built from the root constants `roots` on
    /// first use. `None` for a pattern or format that fails to validate.
    fn prepared_index(&mut self, key: PreparedKey, roots: (usize, Option<usize>)) -> Option<usize> {
        if let Some(index) = self.prepared_patterns.get(&key) {
            return *index;
        }
        let made = {
            let constants = &self.ir.root_constants;
            let bytes = constants.get(roots.0)?;
            match key.2 {
                0 => Prepared::pattern(bytes),
                1 => Prepared::gmatch(bytes),
                3 => Prepared::format(bytes),
                4 => Prepared::run_class(bytes),
                _ => Prepared::gsub(
                    bytes,
                    roots
                        .1
                        .and_then(|root| constants.get(root))
                        .map(|bytes| &bytes[..]),
                ),
            }
        };
        let index = made.map(|made| {
            self.prepared.push(made);
            self.prepared.len() - 1
        });
        self.prepared_patterns.insert(key, index);
        index
    }

    /// For a `string.gmatch` call with a constant pattern: notes the object
    /// its steps read, under the iterator `result`.
    fn note_gmatch(&mut self, args: &[ValueId], result: Option<ValueId>) {
        if !self.string_constants {
            return;
        }
        let (Some(result), Some(pattern)) = (result, args.get(1)) else {
            return;
        };
        let Some((lane, root)) = self.root_string(*pattern) else {
            return;
        };
        if let Some(index) = self.prepared_index((lane, u64::MAX, 1), (root, None)) {
            self.gmatch_patterns.insert(result, index);
        }
    }

    /// The address of a pure native function, as a call target.
    fn native_address(&mut self, address: usize) -> Value {
        let pointer = self.builder.func.dfg.value_type(
            // The entry block's first parameter is the inputs pointer.
            self.builder
                .block_params(self.builder.func.layout.entry_block().expect("entry block"))[0],
        );
        self.builder
            .ins()
            .iconst(pointer, i64::try_from(address).expect("address fits i64"))
    }

    /// Native code that adds one to the diagnostic `counter`.
    #[cfg(feature = "op-stats")]
    fn bump_counter(&mut self, counter: op_stats::Counter) {
        self.add_counter(counter, 1);
    }

    /// Native code that adds `amount` to the diagnostic `counter`.
    #[cfg(feature = "op-stats")]
    fn add_counter(&mut self, counter: op_stats::Counter, amount: i64) {
        let address = self.native_address(op_stats::address(counter));
        let old = self
            .builder
            .ins()
            .load(types::I64, MemFlags::trusted(), address, 0);
        let new = self.builder.ins().iadd_imm(old, amount);
        self.builder
            .ins()
            .store(MemFlags::trusted(), new, address, 0);
    }

    /// Counts one executed bound check of an access keyed by `key`.
    #[cfg(feature = "op-stats")]
    fn count_bound_check(&mut self, key: ValueId) {
        use crate::jit_ir::trace::opportunity::ArrayKeyClass;
        self.bump_counter(op_stats::Counter::NativeBoundChecks);
        match self.range_facts[self.member].class(self.ir, key) {
            ArrayKeyClass::Induction { range_once } => {
                self.bump_counter(op_stats::Counter::NativeBoundChecksInduction);
                if range_once {
                    self.bump_counter(op_stats::Counter::NativeBoundChecksRangeOnce);
                }
            }
            ArrayKeyClass::FloatInvariantInduction { range_once } => {
                self.bump_counter(op_stats::Counter::NativeBoundChecksFloatInvariantInduction);
                if range_once {
                    self.bump_counter(op_stats::Counter::NativeBoundChecksFloatInvariantRangeOnce);
                }
            }
            ArrayKeyClass::Constant | ArrayKeyClass::Other => {}
        }
    }

    /// The signature `extern "C" fn(f64 * arguments) -> f64`.
    fn native_signature(&mut self, arguments: usize) -> SigRef {
        let mut signature = Signature::new(self.builder.func.signature.call_conv);
        for _ in 0..arguments {
            signature.params.push(AbiParam::new(types::F64));
        }
        signature.returns.push(AbiParam::new(types::F64));
        self.builder.import_signature(signature)
    }

    // The arms spell out each IR scalar kind; `Integer` and `Int64` share a
    // lowering but stay separate rows so the kinds remain visible here.
    #[allow(clippy::match_same_arms)]
    fn constant(&mut self, scalar: TraceScalar) -> Value {
        match scalar {
            TraceScalar::Nil => self.builder.ins().iconst(types::I8, 0),
            TraceScalar::Boolean(flag) => self.builder.ins().iconst(types::I8, i64::from(flag)),
            TraceScalar::Integer(value) => self.builder.ins().iconst(types::I64, value),
            TraceScalar::Float(value) => self.builder.ins().f64const(value),
            TraceScalar::Int64(value) => self.builder.ins().iconst(types::I64, value),
            TraceScalar::UInt64(value) => {
                self.builder.ins().iconst(types::I64, value.cast_signed())
            }
            // A verified trace has no opaque constant. A lane id is a plain
            // integer, so this stays total.
            TraceScalar::Table(lane)
            | TraceScalar::Closure(lane)
            | TraceScalar::Thread(lane)
            | TraceScalar::Userdata(lane)
            | TraceScalar::String(lane)
            | TraceScalar::StringRef(lane) => {
                self.builder.ins().iconst(types::I64, lane.cast_signed())
            }
        }
    }

    fn lower(&mut self) -> Result<(), TraceCompileError> {
        self.lower_root()?;
        self.finish_native_dispatch();
        Ok(())
    }

    fn lower_root(&mut self) -> Result<(), TraceCompileError> {
        if self.function.is_some() {
            return self.lower_activation();
        }
        let ir = self.ir;
        let entry = self.builder.create_block();
        self.native_link = self.native_link_of_root();
        if self.native_link.is_some() {
            // A native link runs the entry again, as the run would, so that
            // nothing but the three buffers is live across the call of the
            // side: the loop keeps no value in a register that survives
            // calls. The function's own entry block cannot be jumped to.
            let first = self.builder.create_block();
            self.builder.append_block_params_for_function_params(first);
            for _ in 0..3 {
                self.builder.append_block_param(entry, types::I64);
            }
            self.builder.switch_to_block(first);
            let arguments = self.builder.block_params(first).to_vec();
            self.builder.ins().jump(entry, &block_args(&arguments));
            self.native_entry = Some(entry);
        } else {
            self.builder.append_block_params_for_function_params(entry);
        }
        let header = self.builder.create_block();
        for param in &ir.loop_params {
            self.builder.append_block_param(header, clif_type(param.ty));
        }
        self.builder.append_block_param(header, types::I64);
        if self.members[0].ir.counted {
            self.builder.append_block_param(header, types::I64);
        }

        self.header = Some(header);
        self.lower_entry(entry, header)?;

        self.builder.switch_to_block(header);
        self.forget_array_words();
        let parameters = self.builder.block_params(header).to_vec();
        for (index, param) in ir.loop_params.iter().enumerate() {
            self.define(param.value, parameters[index]);
        }
        let fuel = parameters[ir.loop_params.len()];
        if ir.counted {
            self.budget = Some(parameters[ir.loop_params.len() + 1]);
        }

        // A tree widens its root with the registers its sides write, and the
        // root passes each of those on unchanged at every loop-back. Only a
        // side's link changes one. Carrying them round the root's own loop
        // would cost a move or a spill reload per iteration, and would keep
        // the work that depends on them (a numeric for loop's step checks, say)
        // inside the loop. So the root loops on an inner header that carries
        // just the parameters its iteration changes. The outer header, which
        // entry and every link jump to, holds the others, and the preheader
        // runs there: it redoes the invariant work whenever a link brought new
        // values, and the inner loop sees them as plain values defined above.
        let passed = pass_through_params(ir);
        let plan = self.range_plan.take();
        let versioned = plan.is_some();
        self.versioned_loop = versioned;
        if passed.is_empty() && plan.is_none() {
            return self.lower_member_code(fuel);
        }
        if !passed.is_empty() {
            for (index, instruction) in ir.preheader.iter().enumerate() {
                self.lower_instruction(
                    &instruction.op,
                    instruction.result,
                    fuel,
                    &ir.preheader[index + 1..],
                )?;
            }
        }
        let mut changing: Vec<usize> = (0..ir.loop_params.len())
            .filter(|index| !passed.contains(index))
            .collect();
        if let Some(plan) = &plan {
            changing.extend(self.carried_for_links(plan, &passed));
            changing.sort_unstable();
        }
        let inner = self.builder.create_block();
        for &index in &changing {
            self.builder
                .append_block_param(inner, clif_type(ir.loop_params[index].ty));
        }
        self.builder.append_block_param(inner, types::I64);
        if ir.counted {
            self.builder.append_block_param(inner, types::I64);
        }
        let mut args: Vec<Value> = changing.iter().map(|index| parameters[*index]).collect();
        args.push(fuel);
        if let Some(budget) = self.budget {
            args.push(budget);
        }
        if !passed.is_empty() {
            self.load_fresh_words();
        }
        match plan {
            Some(plan) => self.lower_versions(plan, &changing, &args, inner)?,
            None => {
                self.builder.ins().jump(inner, &block_args(&args));
            }
        }
        self.builder.switch_to_block(inner);
        if versioned {
            self.slow_loop = Some((inner, changing.clone()));
        }
        self.lower_loop_version(inner, changing)
    }

    /// The parameters the root passes on unchanged that the versions of a
    /// root lowered in two carry anyway (see [`range::carried_for_links`]).
    fn carried_for_links(&self, plan: &RangePlan, passed: &[usize]) -> Vec<usize> {
        range::carried_for_links(
            self.ir,
            self.members[1..].iter().map(|member| member.ir),
            plan,
            passed,
        )
    }

    /// Lowers the root's loop on `block`, a header that carries the loop
    /// parameters `changing`, the fuel and the budget (for a counted trace).
    /// The values of the other parameters, and the array words of a loop
    /// that reads fresh ones, are those the code before it defined.
    fn lower_loop_version(
        &mut self,
        block: Block,
        changing: Vec<usize>,
    ) -> Result<(), TraceCompileError> {
        let ir = self.ir;
        if !self.fresh_words {
            self.forget_array_words();
        }
        self.slots.clear();
        // A version of the loop is a new region: the addresses the code
        // before it computed in another version are not on every path.
        self.row_addresses.clear();
        self.lane_array_cache.clear();
        self.field_slots.clear();
        self.next_slot = None;
        self.helper_results.clear();
        self.made_tables.clear();
        let parameters = self.builder.block_params(block).to_vec();
        for (position, &index) in changing.iter().enumerate() {
            self.define(ir.loop_params[index].value, parameters[position]);
        }
        let fuel = parameters[changing.len()];
        if ir.counted {
            self.budget = Some(parameters[changing.len() + 1]);
        }
        if let Some(fast) = self.fast.as_mut() {
            fast.rows.clear();
            fast.roots = fast
                .plan
                .inductions
                .iter()
                .map(|induction| {
                    let position = changing
                        .iter()
                        .position(|index| *index == induction.root)
                        .expect("an induction's root changes");
                    parameters[position]
                })
                .collect();
        }
        self.inner = Some((block, changing));
        self.lower_member_code(fuel)
    }

    /// Lowers the fast version of the root's loop: the range checks of
    /// `plan` in the current block, each of which continues at `slow` with
    /// `args` (the values of the parameters `changing`, the fuel and the
    /// budget) when it fails, and then the loop on a header of its own whose
    /// accesses with proven keys check no bounds.
    ///
    /// Every way into the loop passes the checks: the entry and every link
    /// of a side come through here, since they jump to the root's header,
    /// and the checks hold for the whole run of the loop from there (see
    /// [`crate::jit_ir::trace::range`]). The loop-back of the fast version
    /// stays in it, and that of the slow one in the slow one, which is the
    /// loop as it was before range proofs: a failed check costs nothing but
    /// the checks.
    ///
    /// A float induction's keys are integral, so the fast version carries an
    /// integer copy of each checked float parameter and indexes with it, with
    /// no conversion: the copies start as the parameters' starts, converted
    /// here, and step by the step as an integer.
    #[allow(clippy::too_many_lines, reason = "one check after the other")]
    fn lower_versions(
        &mut self,
        plan: RangePlan,
        changing: &[usize],
        args: &[Value],
        slow: Block,
    ) -> Result<(), TraceCompileError> {
        let ir = self.ir;
        let unless = |this: &mut Self, condition: Value| {
            let next = this.builder.create_block();
            this.builder
                .ins()
                .brif(condition, next, &[], slow, &block_args(args));
            this.builder.switch_to_block(next);
        };
        let max = crate::jit_ir::trace::range::MAX_RANGE_TERM;
        #[allow(clippy::cast_precision_loss, reason = "small integers")]
        let as_float = |number: i64| number as f64;
        // A float of magnitude at most 2^51 as the integer it is, or the slow
        // version when it is not integral: adding 1.5 * 2^52 rounds it to the
        // nearest integer, and the sum less the constant equals the float
        // only when the float is integral. The callers' ordered compares have
        // refused NaN and large numbers already.
        let integral = |this: &mut Self, number: Value| {
            let magic = this.builder.ins().f64const(FLOAT_ROUND_MAGIC);
            let shifted = this.builder.ins().fadd(number, magic);
            let rounded = this.builder.ins().fsub(shifted, magic);
            let ok = this.builder.ins().fcmp(FloatCC::Equal, rounded, number);
            unless(this, ok);
            // The integer by conversion, not from the sum's bits: the backend
            // would fold that arithmetic into each use in the loop. The
            // number is small and integral here, so the conversion is exact
            // and never traps.
            this.builder.ins().fcvt_to_sint(types::I64, number)
        };
        // Whether `value` lies in `[low, high]` (ordered compares for a
        // float, so NaN does not), each end optional.
        let within = |this: &mut Self,
                      float: bool,
                      value: Value,
                      low: Option<Value>,
                      high: Option<Value>| {
            for (limit, float_cc, int_cc) in [
                (
                    low,
                    FloatCC::GreaterThanOrEqual,
                    IntCC::SignedGreaterThanOrEqual,
                ),
                (high, FloatCC::LessThanOrEqual, IntCC::SignedLessThanOrEqual),
            ] {
                let Some(limit) = limit else {
                    continue;
                };
                let ok = if float {
                    this.builder.ins().fcmp(float_cc, value, limit)
                } else {
                    this.builder.ins().icmp(int_cc, value, limit)
                };
                unless(this, ok);
            }
        };
        let constant = |this: &mut Self, float: bool, number: i64| {
            if float {
                this.builder.ins().f64const(as_float(number))
            } else {
                this.builder.ins().iconst(types::I64, number)
            }
        };
        // The shifts as the numbers they add, checked further down.
        let mut shifts: Vec<(RangeShift, Value, Value)> = Vec::new();
        for check in &plan.checks {
            let Some(shift) = check.shift else {
                continue;
            };
            if shifts.iter().any(|(known, _, _)| *known == shift) {
                continue;
            }
            let float = plan.inductions[check.induction].float;
            let value = self.value(shift.value);
            let value = match (shift.negated, float) {
                (false, _) => value,
                (true, true) => self.builder.ins().fneg(value),
                (true, false) => self.builder.ins().ineg(value),
            };
            shifts.push((shift, value, value));
        }
        // First the compares that a loop running off its array fails: the
        // root's start inside `[first, last]` of each array, and the bound on
        // the side the loop moves towards. A shift moves both ends the other
        // way. With a shift not yet checked the ends may be off, but then
        // its check below takes the slow version anyway.
        for check in &plan.checks {
            let induction = &plan.inductions[check.induction];
            let float = induction.float;
            let (_, length, length_float) = self.array_words(check.binding);
            let shift = check.shift.map(|shift| {
                shifts
                    .iter()
                    .find(|(known, _, _)| *known == shift)
                    .map(|(_, value, _)| *value)
                    .expect("every shift has its value")
            });
            let mut first = constant(self, float, 1 - check.low);
            let mut last = if float {
                let length_float =
                    length_float.expect("an array binding has its length as a float");
                let high = constant(self, true, -check.high);
                self.builder.ins().fadd(length_float, high)
            } else {
                self.builder.ins().iadd_imm(length, -check.high)
            };
            if let Some(shift) = shift {
                if float {
                    first = self.builder.ins().fsub(first, shift);
                    last = self.builder.ins().fsub(last, shift);
                } else {
                    first = self.builder.ins().isub(first, shift);
                    last = self.builder.ins().isub(last, shift);
                }
            }
            let start = self.value(ir.loop_params[induction.root].value);
            // Every induction of a plan this backend asks for has a bound.
            if let Some(bound) = induction.bound {
                let bound = self.value(bound);
                if induction.upward {
                    within(self, float, bound, None, Some(last));
                } else {
                    within(self, float, bound, Some(first), None);
                }
            }
            within(self, float, start, Some(first), Some(last));
        }
        // Every parameter of an induction starts where its root does, so all
        // of them hold the root's value at every header of the run.
        for induction in &plan.inductions {
            let root = self.value(ir.loop_params[induction.root].value);
            for &param in &induction.params {
                if param == induction.root {
                    continue;
                }
                let start = self.value(ir.loop_params[param].value);
                let ok = if induction.float {
                    self.builder.ins().fcmp(FloatCC::Equal, start, root)
                } else {
                    self.builder.ins().icmp(IntCC::Equal, start, root)
                };
                unless(self, ok);
            }
        }
        // The steps: of the right sign, small and, for a float, integral.
        let mut steps = Vec::with_capacity(plan.inductions.len());
        for induction in &plan.inductions {
            let RangeStep::Invariant { value, negated } = induction.step else {
                steps.push(None);
                continue;
            };
            let float = induction.float;
            let positive = induction.upward != negated;
            let (low, high) = if positive { (1, max) } else { (-max, -1) };
            let step = self.value(value);
            let low = constant(self, float, low);
            let high = constant(self, float, high);
            within(self, float, step, Some(low), Some(high));
            let integer = if float { integral(self, step) } else { step };
            steps.push(Some(if negated {
                self.builder.ins().ineg(integer)
            } else {
                integer
            }));
        }
        // The shifts: small and, for a float, integral, like a step.
        for entry in &mut shifts {
            let (shift, value, _) = *entry;
            let float = plan
                .checks
                .iter()
                .find(|check| check.shift == Some(shift))
                .is_some_and(|check| plan.inductions[check.induction].float);
            let low = constant(self, float, -max);
            let high = constant(self, float, max);
            within(self, float, value, Some(low), Some(high));
            if float {
                entry.2 = integral(self, value);
            }
        }
        // The integer copy of each float induction, which inside the array is
        // exact: a start that is not integral takes the slow version.
        let mut shadows = Vec::new();
        for (index, induction) in plan.inductions.iter().enumerate() {
            if !induction.float {
                continue;
            }
            let start = self.value(ir.loop_params[induction.root].value);
            shadows.push((index, integral(self, start)));
        }
        let fast = self.builder.create_block();
        for &index in changing {
            self.builder
                .append_block_param(fast, clif_type(ir.loop_params[index].ty));
        }
        self.builder.append_block_param(fast, types::I64);
        if ir.counted {
            self.builder.append_block_param(fast, types::I64);
        }
        for _ in &shadows {
            self.builder.append_block_param(fast, types::I64);
        }
        let mut fast_args = args.to_vec();
        fast_args.extend(shadows.iter().map(|(_, value)| *value));
        self.builder.ins().jump(fast, &block_args(&fast_args));
        let cache = self.array_cache.clone();
        let lane_cache = self.lane_array_cache.clone();
        self.builder.switch_to_block(fast);
        let extra = self.builder.block_params(fast)[args.len()..].to_vec();
        for (shadow, value) in shadows.iter_mut().zip(extra) {
            shadow.1 = value;
        }
        self.fast = Some(FastLoop {
            plan,
            shadows,
            steps,
            shifts: shifts
                .into_iter()
                .map(|(shift, _, integer)| (shift, integer))
                .collect(),
            roots: Vec::new(),
            rows: Vec::new(),
        });
        let result = self.lower_loop_version(fast, changing.to_vec());
        self.fast = None;
        // The slow version starts from the words loaded before the checks.
        self.array_cache = cache;
        self.lane_array_cache = lane_cache;
        result
    }

    /// The values the integer copies of the fast version take at the
    /// loop-back: each induction's copy plus its step.
    fn shadow_nexts(&mut self) -> Vec<Value> {
        if self.member != 0 {
            return Vec::new();
        }
        self.stepped_copies()
    }

    /// The fast version's integer copies of the iteration being lowered,
    /// each plus its induction's step: what they are in the next iteration.
    fn stepped_copies(&mut self) -> Vec<Value> {
        let Some(fast) = self.fast.as_ref() else {
            return Vec::new();
        };
        let stepped: Vec<(Value, RangeStep, Option<Value>)> = fast
            .shadows
            .iter()
            .map(|&(induction, current)| {
                (
                    current,
                    fast.plan.inductions[induction].step,
                    fast.steps[induction],
                )
            })
            .collect();
        stepped
            .into_iter()
            .map(|(current, step, step_value)| match (step, step_value) {
                (_, Some(step)) => self.builder.ins().iadd(current, step),
                (RangeStep::Constant(step), None) => self.builder.ins().iadd_imm(current, step),
                (RangeStep::Invariant { .. }, None) => {
                    unreachable!("an invariant step has its integer")
                }
            })
            .collect()
    }

    /// Where a link with the state `link` and `next`, a value for every
    /// loop parameter of the root, goes: the root's header, which checks the
    /// range plan again when the root has one, or straight to the slow
    /// version's header when the link leaves a side of the root lowered in
    /// the slow version and changes none of the parameters that version
    /// does not carry. Such a link stays in the slow version as its own
    /// loop-back would: the checks only choose a version, and a side taken
    /// every few iterations would otherwise pay for them each time.
    fn link_target(
        &mut self,
        link: &crate::jit_ir::trace::SnapshotFrame,
        next: Vec<Value>,
    ) -> (Block, (Vec<Value>, Vec<Value>)) {
        let header = self.header.expect("the root made its header");
        if self.root_side
            && self.fast.is_some()
            && self.fast_link(link).is_some()
            && let Some((fast, changing)) = self.inner.clone()
        {
            let args: Vec<Value> = changing.iter().map(|index| next[*index]).collect();
            // After the fuel and the budget, as at the fast loop-back.
            let copies = self.stepped_copies();
            return (fast, (args, copies));
        }
        let Some((slow, changing)) = self.slow_loop.as_ref() else {
            return (header, (next, Vec::new()));
        };
        if !self.root_side || self.fast.is_some() {
            return (header, (next, Vec::new()));
        }
        let unchanged = range::slow_link(self.members[0].ir, self.ir, changing, link);
        if !unchanged {
            return (header, (next, Vec::new()));
        }
        (
            *slow,
            (
                changing.iter().map(|index| next[*index]).collect(),
                Vec::new(),
            ),
        )
    }

    /// The root value each entry input of side `side` starts as when the
    /// root's guard `guard` fails: the value the guard's snapshot names for
    /// the input's register, or the root's entry input of the register when
    /// the snapshot leaves it unchanged.
    fn side_origins_of(&self, side: usize, guard: usize) -> Vec<Option<ValueId>> {
        let root = self.members[0].ir;
        let snapshot = &root.snapshots[root.guards[guard].snapshot.index()];
        let mut origins = Vec::new();
        range::side_origins(root, self.members[side].ir, snapshot, &mut origins);
        origins
    }

    /// Whether a link of a side of the root, lowered in the fast version, may
    /// go straight back to the fast loop with no checks (see
    /// [`range::fast_link`]). `None` when it may not.
    fn fast_link(&self, link: &crate::jit_ir::trace::SnapshotFrame) -> Option<()> {
        let fast = self.fast.as_ref()?;
        let (_, changing) = self.inner.as_ref()?;
        range::fast_link(
            &fast.plan,
            self.members[0].ir,
            self.ir,
            &self.side_origins,
            changing,
            link,
        )
        .then_some(())
    }

    /// The proven slot of `key` on `binding` in the fast version, as a row
    /// address and a byte displacement from it: the slot is `row + disp`.
    /// Every access of one index shares the row (`base + index * 16`), so
    /// the scale and the add run once per index, and each access pays only
    /// its displacement, which the load and store encode.
    fn proven_slot(&mut self, binding: u32, key: ValueId) -> Option<(Value, i32)> {
        if self.member != 0 {
            return None;
        }
        let fast = self.fast.as_ref()?;
        let proven = *fast.plan.key(binding, key)?;
        let (copy, shift, index_offset) = if fast.plan.inductions[proven.induction].float {
            let copy = fast
                .shadows
                .iter()
                .find(|(induction, _)| *induction == proven.induction)
                .map(|(_, value)| *value)
                .expect("a float induction has a copy");
            let shift = proven.shift.map(|shift| {
                fast.shifts
                    .iter()
                    .find(|(known, _)| *known == shift)
                    .map(|(_, value)| *value)
                    .expect("every shift was checked")
            });
            (copy, shift, proven.offset - 1)
        } else {
            (self.value(key), None, -1)
        };
        let disp = i32::try_from(index_offset.checked_mul(16)?).ok()?;
        let base = self.array_words(binding).0;
        let known = self
            .fast
            .as_ref()?
            .rows
            .iter()
            .find(|fact| fact.base == base && fact.copy == copy && fact.shift == shift)
            .map(|fact| fact.row);
        let row = if let Some(row) = known {
            row
        } else {
            let index = match shift {
                Some(shift) => self.builder.ins().iadd(copy, shift),
                None => copy,
            };
            let scaled = self.builder.ins().ishl_imm(index, 4);
            let row = self.builder.ins().iadd(base, scaled);
            self.fast.as_mut()?.rows.push(FastRow {
                base,
                copy,
                shift,
                row,
            });
            row
        };
        Some((row, disp))
    }

    /// Lowers the body and the terminator of the current trace into the
    /// current block.
    fn lower_member_code(&mut self, fuel: Value) -> Result<(), TraceCompileError> {
        let ir = self.ir;
        self.defer_cold_values();
        self.member_fuel = Some(fuel);
        self.fused = if self.member == 0 {
            fused_delete_stores(ir)
        } else {
            std::collections::HashMap::new()
        };
        for (index, instruction) in ir.body.iter().enumerate() {
            if instruction
                .result
                .is_some_and(|result| self.deferred[result.index()])
            {
                continue;
            }
            // An activation's fuel is what its last self call left.
            let fuel = self.activation_fuel.unwrap_or(fuel);
            self.lower_instruction(
                &instruction.op,
                instruction.result,
                fuel,
                &ir.body[index + 1..],
            )?;
        }
        // Only pure operations can follow a held delete, so one left here
        // met no store before the end of the body.
        self.flush_pending_delete(fuel)?;
        self.fused.clear();
        let fuel = self.activation_fuel.unwrap_or(fuel);
        let exits = &self.plan.members[self.member];
        match &ir.terminator {
            TraceTerminator::LoopBack { args } if self.loop_header.is_some() => {
                // A loop side loops on its own header with every parameter.
                let target = self.loop_header.expect("checked by the guard");
                let next = args.iter().map(|arg| self.value(*arg)).collect();
                self.loop_back(target, (next, Vec::new()), fuel, exits.poll, 0)?;
            }
            TraceTerminator::LoopBack { args } => {
                // The root loops on its inner header when it has one, with
                // the parameters its iteration changes.
                let (target, wanted): (Block, Vec<ValueId>) = match &self.inner {
                    Some((inner, changing)) => {
                        (*inner, changing.iter().map(|index| args[*index]).collect())
                    }
                    None => (self.header.expect("the root made its header"), args.clone()),
                };
                let next = wanted.iter().map(|arg| self.value(*arg)).collect();
                let copies = self.shadow_nexts();
                // The iteration's cost is what its poll exit says ran.
                let cost = ir
                    .poll_snapshot
                    .map_or(0, |poll| ir.snapshots[poll.index()].counted);
                self.loop_back(target, (next, copies), fuel, exits.poll, cost)?;
            }
            TraceTerminator::Link { snapshot } if self.detached.is_some() => {
                self.detached_link(*snapshot, fuel, exits.poll)?;
            }
            TraceTerminator::Link { snapshot } => {
                // The final state names one value per loop parameter of the
                // root, by register.
                let link = &ir.snapshots[snapshot.index()].frames[0];
                let root = self.members[0].ir;
                let mut next = Vec::with_capacity(root.loop_params.len());
                for param in &root.loop_params {
                    let register = param
                        .register
                        .ok_or(TraceCompileError::NativeBackendRejected)?;
                    next.push(match link.registers.get(usize::from(register)) {
                        Some(RegisterEntry::Value(value)) => {
                            // A borrow is not the owned value a parameter
                            // holds.
                            if ir.value_type(*value).is_some_and(TraceType::is_borrowed) {
                                return Err(TraceCompileError::NativeBackendRejected);
                            }
                            self.value(*value)
                        }
                        Some(RegisterEntry::Const(scalar)) => self.constant(*scalar),
                        _ => return Err(TraceCompileError::NativeBackendRejected),
                    });
                }
                // A register the root takes untyped arrives through its slot,
                // which the root's type guard reads again: the link enters
                // the loop where the guard runs, not past it.
                let (target, next) = if self.store_untyped_link(link)? {
                    let header = self.header.expect("the root made its header");
                    (header, (next, Vec::new()))
                } else {
                    self.link_target(link, next)
                };
                let cost = ir.snapshots[snapshot.index()].counted;
                self.loop_back(target, next, fuel, exits.poll, cost)?;
            }
            TraceTerminator::LinkLoop { snapshot, .. } => {
                // The loop side `side` is an ancestor whose loop is being
                // lowered around this code: its header dominates the jump.
                let target = self.members[self.member]
                    .link_loop
                    .ok_or(TraceCompileError::NativeBackendRejected)?;
                let (header, params) = self
                    .loop_stack
                    .iter()
                    .rev()
                    .find(|(member, _, _)| *member == target)
                    .map(|(_, header, params)| (*header, params.clone()))
                    .ok_or(TraceCompileError::NativeBackendRejected)?;
                let link = &ir.snapshots[snapshot.index()].frames[0];
                let mut next = Vec::with_capacity(params.len());
                for (register, ty) in params {
                    next.push(match link.registers.get(usize::from(register)) {
                        Some(RegisterEntry::Value(value))
                            if ir.value_type(*value) == Some(ty) && !ty.is_borrowed() =>
                        {
                            self.value(*value)
                        }
                        Some(RegisterEntry::Const(scalar)) if scalar.ty() == ty => {
                            self.constant(*scalar)
                        }
                        _ => return Err(TraceCompileError::NativeBackendRejected),
                    });
                }
                if self.count_links {
                    self.count_link();
                }
                self.loop_back(header, (next, Vec::new()), fuel, exits.poll, 0)?;
            }
            TraceTerminator::Exit { snapshot } => {
                let exit = exits
                    .terminal
                    .ok_or(TraceCompileError::NativeBackendRejected)?;
                match self.members[self.member].terminal {
                    // The loop the trace ends in front of runs in place.
                    Some(side) => self.enter_loop_side(side, *snapshot, exit, fuel)?,
                    None => self.emit_exit(exit, fuel),
                }
            }
            TraceTerminator::TailCall { args, .. } => {
                // The frame starts over with the arguments, one tail call
                // more.
                let (start, register) = self
                    .function
                    .as_ref()
                    .and_then(|lowering| Some((lowering.start?, lowering.count?)))
                    .ok_or(TraceCompileError::NativeBackendRejected)?;
                let counted = ir
                    .entry
                    .inputs
                    .iter()
                    .find(|input| input.register == register)
                    .map(|input| input.value)
                    .ok_or(TraceCompileError::NativeBackendRejected)?;
                let count = self.value(counted);
                let next = self.builder.ins().iadd_imm(count, 1);
                let mut passed = vec![fuel, next];
                for arg in args {
                    passed.push(self.value(*arg));
                }
                self.builder.ins().jump(start, &block_args(&passed));
            }
            TraceTerminator::Return { value } => {
                let mut returned = vec![fuel];
                let wants = self
                    .function
                    .as_ref()
                    .ok_or(TraceCompileError::NativeBackendRejected)?
                    .result;
                match (value, wants) {
                    (Some(value), Some(_)) => returned.push(self.value(*value)),
                    (None, None) => {}
                    _ => return Err(TraceCompileError::NativeBackendRejected),
                }
                self.builder.ins().return_(&returned);
            }
        }
        Ok(())
    }

    /// The body of an activation of a function trace: the root's entry
    /// values (parameters from the call, nil, and the run's cells and globals
    /// from the input slots), the table words, one unit of fuel, and the
    /// root's code, which ends in returns.
    fn lower_activation(&mut self) -> Result<(), TraceCompileError> {
        let ir = self.ir;
        let function = ir
            .function
            .as_ref()
            .ok_or(TraceCompileError::NativeBackendRejected)?;
        let entry = self.builder.create_block();
        self.builder.append_block_params_for_function_params(entry);
        self.builder.switch_to_block(entry);
        let parameters = self.builder.block_params(entry).to_vec();
        let (control, left) = (parameters[0], parameters[1]);
        self.control = Some(control);
        self.depth_left = Some(left);
        let count = self.function.as_ref().and_then(|lowering| lowering.count);
        if !ir.bindings.is_empty() {
            let inputs = self.builder.ins().load(
                types::I64,
                MemFlags::trusted().with_readonly(),
                control,
                offset(mem::offset_of!(TraceControl, inputs)),
            );
            self.load_bindings(inputs);
        }
        // The activation starts here, and starts over here after a tail call
        // of its own function: the fuel, the tail call count and the
        // parameters are the block's.
        let start = self.builder.create_block();
        for value in &parameters[2..] {
            let ty = self.builder.func.dfg.value_type(*value);
            self.builder.append_block_param(start, ty);
        }
        self.builder
            .ins()
            .jump(start, &block_args(&parameters[2..]));
        self.builder.switch_to_block(start);
        if let Some(lowering) = self.function.as_mut() {
            lowering.start = Some(start);
        }
        let started = self.builder.block_params(start).to_vec();
        let fuel = started[0];
        let first_parameter = 1 + usize::from(count.is_some());
        let guest = usize::from(ir.frames[0].register_count).saturating_sub(ir.external.len());
        for (index, input) in ir.entry.inputs.iter().enumerate() {
            let register = usize::from(input.register);
            if register < function.parameters.len() {
                self.define(input.value, started[first_parameter + register]);
                self.root_inputs
                    .insert(input.register, started[first_parameter + register]);
            } else if Some(input.register) == count {
                self.define(input.value, started[1]);
                self.root_inputs.insert(input.register, started[1]);
            } else if register < guest {
                // A fresh activation holds nil past its parameters.
                if input.ty != TraceType::Nil {
                    return Err(TraceCompileError::NativeBackendRejected);
                }
                let nil = self.builder.ins().iconst(types::I8, 0);
                self.define(input.value, nil);
                self.root_inputs.insert(input.register, nil);
            } else if matches!(input.ty, TraceType::Any | TraceType::StringRef) {
                return Err(TraceCompileError::NativeBackendRejected);
            } else {
                // A cell or global the entry checked, read where it is used.
                self.lazy.insert(input.value, (index, input.ty));
            }
        }
        // Each activation spends one unit of fuel, so a long recursion still
        // comes back to the VM.
        let fuel = self.builder.ins().iadd_imm(fuel, -1);
        self.activation_fuel = Some(fuel);
        let go = self
            .builder
            .ins()
            .icmp_imm(IntCC::SignedGreaterThan, fuel, 0);
        let run = self.builder.create_block();
        let poll = self.builder.create_block();
        self.builder.set_cold_block(poll);
        self.builder.ins().brif(go, run, &[], poll, &[]);
        self.builder.switch_to_block(poll);
        let exit = self.plan.members[0]
            .poll
            .ok_or(TraceCompileError::MissingPollSnapshot)?;
        self.emit_exit(exit, fuel);
        self.builder.switch_to_block(run);
        self.lower_member_code(fuel)
    }

    /// A self call ([`TraceOp::SelfCall`]): runs the callee as a nested
    /// activation one area further into the exit area, unless the run is as
    /// deep as it may go, when the guard leaves before the call. A callee
    /// that left the trace makes this activation leave too: it spills the
    /// state it stands in at the call (the unwind exit) and passes the
    /// callee's status on.
    fn self_call(
        &mut self,
        args: &[ValueId],
        guard: GuardId,
        fuel: Value,
    ) -> Result<Option<Value>, TraceCompileError> {
        let (body, header_offset, result) = {
            let function = self
                .function
                .as_ref()
                .ok_or(TraceCompileError::NativeBackendRejected)?;
            (function.body, function.header_offset, function.result)
        };
        let unwind = self.plan.members[self.member]
            .unwinds
            .iter()
            .find(|(held, _)| *held == guard.index())
            .map(|(_, exit)| *exit)
            .ok_or(TraceCompileError::NativeBackendRejected)?;
        let control = self.control.expect("entry set the control block");
        let left = self.depth_left.expect("entry set the depth left");
        // An earlier self call of the activation that dominates this one
        // found room for the callee already: so is there now.
        let deeper = if let Some(checked) = self.deeper {
            checked
        } else {
            let room = self
                .builder
                .ins()
                .icmp_imm(IntCC::SignedGreaterThan, left, 0);
            let call = self.builder.create_block();
            let at_cap = self.builder.create_block();
            self.builder.set_cold_block(at_cap);
            self.builder.ins().brif(room, call, &[], at_cap, &[]);
            self.builder.switch_to_block(at_cap);
            self.leave(guard, fuel)?;
            self.builder.switch_to_block(call);
            let below = self.builder.ins().iadd_imm(left, -1);
            self.deeper = Some(below);
            below
        };
        let mut arguments = vec![control, deeper, fuel];
        // A nested activation starts with no tail call of its own.
        if self
            .function
            .as_ref()
            .is_some_and(|lowering| lowering.count.is_some())
        {
            arguments.push(self.builder.ins().iconst(types::I64, 0));
        }
        for arg in args {
            arguments.push(self.value(*arg));
        }
        let invoked = self.builder.ins().call(body, &arguments);
        let results = self.builder.inst_results(invoked).to_vec();
        let status = results[0];
        let returned = self
            .builder
            .ins()
            .icmp_imm(IntCC::SignedGreaterThan, status, 0);
        let back = self.builder.create_block();
        let unwinding = self.builder.create_block();
        self.builder.set_cold_block(unwinding);
        self.builder.ins().brif(returned, back, &[], unwinding, &[]);
        self.builder.switch_to_block(unwinding);
        let exit_area = self.spill_exit(unwind);
        let index = self
            .builder
            .ins()
            .iconst(types::I64, i64::try_from(unwind).expect("exit index fits"));
        self.builder
            .ins()
            .store(MemFlags::new(), index, exit_area, header_offset);
        let mut passed = vec![status];
        if let Some(ty) = result {
            passed.push(self.zero(ty));
        }
        self.builder.ins().return_(&passed);
        self.builder.switch_to_block(back);
        self.activation_fuel = Some(status);
        Ok(result.map(|_| results[1]))
    }

    /// A zero of a Cranelift type: the value an activation returns beside a
    /// status that is no return.
    fn zero(&mut self, ty: types::Type) -> Value {
        if ty == types::F64 {
            self.builder.ins().f64const(0.0)
        } else {
            self.builder.ins().iconst(ty, 0)
        }
    }

    /// Spends one unit of fuel and jumps to the root's loop header with
    /// `next`, or leaves through the poll exit when the fuel is spent.
    fn loop_back(
        &mut self,
        header: Block,
        (next, copies): (Vec<Value>, Vec<Value>),
        fuel: Value,
        poll_exit: Option<usize>,
        cost: u32,
    ) -> Result<(), TraceCompileError> {
        let remaining = self.builder.ins().iadd_imm(fuel, -1);
        let go = self
            .builder
            .ins()
            .icmp_imm(IntCC::SignedGreaterThan, remaining, 0);
        let poll = self.builder.create_block();
        self.builder.set_cold_block(poll);
        let mut loop_args = next;
        loop_args.push(remaining);
        // A counted trace pays for the iteration it completed. The caller
        // gave fuel for no more iterations than the budget covers, so the
        // next one needs no test. The poll exit reports the budget the
        // iteration began with, and its descriptor charges the whole path.
        if let Some(budget) = self.budget {
            let paid = self.builder.ins().iadd_imm(budget, -i64::from(cost));
            loop_args.push(paid);
        }
        // The fast version's integer copies of its float inductions.
        loop_args.extend(copies);
        self.builder
            .ins()
            .brif(go, header, &block_args(&loop_args), poll, &[]);
        self.builder.switch_to_block(poll);
        let exit = poll_exit.ok_or(TraceCompileError::MissingPollSnapshot)?;
        self.emit_exit(exit, remaining);
        Ok(())
    }

    /// The body of a side compiled apart from its tree: a function of the
    /// same signature as the tree's, which the exit of its parent calls with
    /// the run's own buffers. Its entry values are read from the exit area,
    /// where that exit stored the state it restores. It ends in an exit of
    /// its own, whose code carries the unit's base, or it links back to the
    /// loop by handing the loop parameters to the run loop
    /// (`CompiledTrace::run_sides`), which enters the root again.
    fn lower_detached(&mut self) -> Result<(), TraceCompileError> {
        let spec = self.detached.expect("a side compiled apart");
        let ir = self.ir;
        let entry = self.builder.create_block();
        self.builder.append_block_params_for_function_params(entry);
        self.builder.switch_to_block(entry);
        let parameters = self.builder.block_params(entry).to_vec();
        let (inputs, control, exit_area) = (parameters[0], parameters[1], parameters[2]);
        self.control = Some(control);
        self.exit_area = Some(exit_area);
        self.entry_inputs = Some(inputs);
        self.rows = Some(self.builder.ins().load(
            types::I64,
            MemFlags::new().with_readonly(),
            control,
            offset(mem::offset_of!(TraceControl, rows)),
        ));
        let fuel = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(mem::offset_of!(TraceControl, fuel)),
        );
        self.load_bindings(inputs);
        let (values, holds) = self.detached_inputs(spec)?;
        if let Some(holds) = holds {
            // A number of the other subtype than the side was recorded for
            // leaves through the exit that called it, whose state is still
            // in the exit area.
            let enter = self.builder.create_block();
            let other = self.builder.create_block();
            self.builder.set_cold_block(other);
            self.builder.ins().brif(holds, enter, &[], other, &[]);
            self.builder.switch_to_block(other);
            let code = self.builder.ins().iconst(types::I64, DECLINE_CODE);
            self.builder.ins().return_(&[code]);
            self.builder.switch_to_block(enter);
        }
        for (input, value) in ir.entry.inputs.iter().zip(&values) {
            self.define(input.value, *value);
        }
        for (input, value) in ir
            .frame_inputs
            .iter()
            .zip(values.iter().skip(ir.entry.inputs.len()))
        {
            self.define(input.value, *value);
        }
        if ir.loop_params.is_empty() {
            self.lower_member_code(fuel)?;
        } else {
            self.lower_loop_side(fuel)?;
        }
        Ok(())
    }

    /// [`Self::side_inputs_at`] for a side compiled apart: the values of its
    /// entry inputs read from the exit area, as the host's exit stored them,
    /// or from the root's entry inputs for a register the host left alone.
    fn detached_inputs(
        &mut self,
        spec: &DetachedSpec<'_>,
    ) -> Result<(Vec<Value>, Option<Value>), TraceCompileError> {
        let ir = self.ir;
        let snapshot = spec.host_snapshot;
        let area = self.exit_area.expect("entry set the exit area");
        let inputs = self.entry_inputs.expect("entry set the inputs");
        let mut values = Vec::with_capacity(ir.entry.inputs.len());
        let mut subtypes: Option<Value> = None;
        let mut extras_differ: Option<Value> = None;
        for input in &ir.entry.inputs {
            let register = usize::from(input.register);
            // A register past the host's window is a slot the side names
            // itself: the host left it alone.
            let entry = snapshot.frames[0]
                .registers
                .get(register)
                .or((register >= snapshot.frames[0].registers.len()
                    && register >= spec.guest_registers)
                    .then_some(&RegisterEntry::Unchanged));
            // A value the host's exit left out is one the frame keeps from the
            // run's entry (`kept_by_frame` in `compile.rs`): the register is
            // as the root was entered with it.
            let entry = match entry {
                Some(RegisterEntry::Value(value))
                    if snapshot.number_tag(register).is_none()
                        && spec
                            .host
                            .slots
                            .binary_search_by_key(value, |slot| slot.value)
                            .is_err() =>
                {
                    Some(&RegisterEntry::Unchanged)
                }
                entry => entry,
            };
            if let Some(tag) = snapshot.number_tag(register) {
                let float = match entry {
                    Some(RegisterEntry::Value(value)) => self.host_value(spec, area, *value)?,
                    Some(RegisterEntry::Const(scalar)) => self.constant(*scalar),
                    _ => return Err(TraceCompileError::NativeBackendRejected),
                };
                let integer = match tag {
                    RegisterEntry::Value(value) => self.host_value(spec, area, value)?,
                    RegisterEntry::Const(scalar) => self.constant(scalar),
                    _ => return Err(TraceCompileError::NativeBackendRejected),
                };
                let (value, holds) = match input.ty {
                    TraceType::Float => (float, self.builder.ins().bxor_imm(integer, 1)),
                    TraceType::Integer => (
                        self.builder.ins().fcvt_to_sint_sat(types::I64, float),
                        integer,
                    ),
                    _ => return Err(TraceCompileError::NativeBackendRejected),
                };
                subtypes = Some(match subtypes {
                    Some(held) => self.builder.ins().band(held, holds),
                    None => holds,
                });
                values.push(value);
                continue;
            }
            let value = match entry {
                Some(RegisterEntry::Value(value)) => {
                    let slot = spec
                        .host
                        .slots
                        .binary_search_by_key(value, |slot| slot.value)
                        .map(|at| spec.host.slots[at])
                        .map_err(|_| TraceCompileError::NativeBackendRejected)?;
                    if slot.ty.is_borrowed() != input.ty.is_borrowed() {
                        return Err(TraceCompileError::NativeBackendRejected);
                    }
                    self.host_value(spec, area, *value)?
                }
                Some(RegisterEntry::Const(_) | RegisterEntry::Root(_))
                    if input.ty.is_borrowed() =>
                {
                    return Err(TraceCompileError::NativeBackendRejected);
                }
                Some(RegisterEntry::Const(scalar)) => self.constant(*scalar),
                Some(RegisterEntry::Root(slot)) => self.constant(TraceScalar::String(
                    ROOT_STRING_LANE_BASE + u64::from(*slot),
                )),
                // The host did not change the register: it is the root's
                // entry input, read from its slot, or an extra input of the
                // run, which the side checks itself. A slot of the side's own
                // window, beyond the host's, is the same.
                Some(RegisterEntry::Unchanged) => {
                    if let Some(position) = spec
                        .root_entry
                        .iter()
                        .position(|(held, _)| usize::from(*held) == register)
                    {
                        let ty = spec.root_entry[position].1;
                        if ty.is_borrowed() || input.ty.is_borrowed() {
                            return Err(TraceCompileError::NativeBackendRejected);
                        }
                        if ty == TraceType::Any && input.ty == TraceType::Any {
                            // The side reads it as the root does: its type
                            // guards load the root's slot.
                            let placeholder = self.builder.ins().iconst(types::I8, 0);
                            self.any_slots.insert(input.register, position);
                            placeholder
                        } else if ty == TraceType::Any {
                            // The root's entry did not check this input: the
                            // side checks it is of its own type.
                            let required = entry_tag(input.ty)
                                .ok_or(TraceCompileError::NativeBackendRejected)?;
                            let tag = self.builder.ins().load(
                                types::I64,
                                MemFlags::new(),
                                inputs,
                                offset(position * mem::size_of::<TraceInputSlot>())
                                    + offset(mem::offset_of!(TraceInputSlot, tag)),
                            );
                            let difference = self.builder.ins().bxor_imm(tag, required);
                            extras_differ = Some(match extras_differ {
                                Some(earlier) => self.builder.ins().bor(earlier, difference),
                                None => difference,
                            });
                            self.load_entry_input(inputs, position, input.ty)
                        } else {
                            self.load_entry_input(inputs, position, ty)
                        }
                    } else {
                        let source = self
                            .extra_source(spec, register, input.ty)
                            .ok_or(TraceCompileError::NativeBackendRejected)?;
                        let slot = self.extra_position(spec, source, input.ty)?;
                        let required =
                            entry_tag(input.ty).ok_or(TraceCompileError::NativeBackendRejected)?;
                        let tag = self.builder.ins().load(
                            types::I64,
                            MemFlags::new(),
                            inputs,
                            offset(slot * mem::size_of::<TraceInputSlot>())
                                + offset(mem::offset_of!(TraceInputSlot, tag)),
                        );
                        let difference = self.builder.ins().bxor_imm(tag, required);
                        extras_differ = Some(match extras_differ {
                            Some(earlier) => self.builder.ins().bor(earlier, difference),
                            None => difference,
                        });
                        self.load_entry_input(inputs, slot, input.ty)
                    }
                }
                _ => return Err(TraceCompileError::NativeBackendRejected),
            };
            values.push(value);
        }
        // The registers of the inlined frames the side starts inside, as the
        // host's exit stored them (see `side_inputs_at`).
        for input in &ir.frame_inputs {
            let value = match snapshot
                .frames
                .get(input.frame.index())
                .and_then(|frame| frame.registers.get(usize::from(input.register)))
            {
                Some(RegisterEntry::Value(value)) if !input.ty.is_borrowed() => {
                    let slot = spec
                        .host
                        .slots
                        .binary_search_by_key(value, |slot| slot.value)
                        .map(|at| spec.host.slots[at])
                        .map_err(|_| TraceCompileError::NativeBackendRejected)?;
                    if slot.ty != input.ty {
                        return Err(TraceCompileError::NativeBackendRejected);
                    }
                    self.host_value(spec, area, *value)?
                }
                Some(RegisterEntry::Const(scalar)) if scalar.ty() == input.ty => {
                    self.constant(*scalar)
                }
                Some(RegisterEntry::Root(slot)) if input.ty == TraceType::String => self.constant(
                    TraceScalar::String(ROOT_STRING_LANE_BASE + u64::from(*slot)),
                ),
                _ => return Err(TraceCompileError::NativeBackendRejected),
            };
            values.push(value);
        }
        // The extra inputs hold the types the side was recorded with, or the
        // side declines to run.
        if let Some(differs) = extras_differ {
            let holds = self.builder.ins().icmp_imm(IntCC::Equal, differs, 0);
            subtypes = Some(match subtypes {
                Some(held) => self.builder.ins().band(held, holds),
                None => holds,
            });
        }
        Ok((values, subtypes))
    }

    /// Where the side finds register `register` of its frame when neither the
    /// host's exit nor the root's entry has it: a guest register of the root
    /// frame, or a register of the caller the side returns into (a slot of
    /// the side's own window). `None` for a register the side cannot read
    /// from an extra input slot.
    fn extra_source(
        &self,
        spec: &DetachedSpec<'_>,
        register: usize,
        ty: TraceType,
    ) -> Option<ExtraSource> {
        if register < spec.guest_registers {
            if !matches!(
                ty,
                TraceType::Float | TraceType::Integer | TraceType::Boolean
            ) {
                return None;
            }
            return Some(ExtraSource::Register(u16::try_from(register).ok()?));
        }
        if !matches!(
            ty,
            TraceType::Float | TraceType::Integer | TraceType::Boolean | TraceType::Closure
        ) {
            return None;
        }
        match self.ir.external.get(register - spec.guest_registers)? {
            ExternalSlot::CallerRegister(caller) => Some(ExtraSource::Caller(*caller)),
            // Read once at entry and never written (a write of one is a write
            // of a cell, which no link carries).
            slot @ (ExternalSlot::CallerGlobal(_)
            | ExternalSlot::ClosureGlobal { .. }
            | ExternalSlot::ClosureUpvalue { .. })
                if !self.ir.written_registers.iter().any(|(frame, written)| {
                    *frame == crate::jit_ir::trace::FrameId(0) && usize::from(*written) == register
                }) =>
            {
                Some(ExtraSource::External(*slot))
            }
            _ => None,
        }
    }

    /// The position in the run's input slots of the extra input `source`, which
    /// holds a value of type `ty`: the tree's own, or a new one after them.
    /// After the entry inputs and the bindings.
    fn extra_position(
        &mut self,
        spec: &DetachedSpec<'_>,
        source: ExtraSource,
        ty: TraceType,
    ) -> Result<usize, TraceCompileError> {
        let at = match self
            .side_extras
            .iter()
            .position(|(held, _)| *held == source)
        {
            Some(at) if self.side_extras[at].1 == ty => at,
            Some(_) => return Err(TraceCompileError::NativeBackendRejected),
            None => {
                self.side_extras.push((source, ty));
                self.side_extras.len() - 1
            }
        };
        Ok(spec.root_entry.len() + self.members[0].ir.bindings.len() + at)
    }

    /// The value `value` of the host trace, which the host's exit stored in
    /// the exit area, loaded as its type says.
    fn host_value(
        &mut self,
        spec: &DetachedSpec<'_>,
        area: Value,
        value: ValueId,
    ) -> Result<Value, TraceCompileError> {
        let slot = spec
            .host
            .slots
            .binary_search_by_key(&value, |slot| slot.value)
            .map(|at| spec.host.slots[at])
            .map_err(|_| TraceCompileError::NativeBackendRejected)?;
        let at = offset(slot.slot as usize * mem::size_of::<i64>());
        Ok(match slot.ty {
            TraceType::Float => self
                .builder
                .ins()
                .load(types::F64, MemFlags::new(), area, at),
            TraceType::Boolean => {
                let bits = self
                    .builder
                    .ins()
                    .load(types::I64, MemFlags::new(), area, at);
                self.builder.ins().icmp_imm(IntCC::NotEqual, bits, 0)
            }
            TraceType::Nil => self.builder.ins().iconst(types::I8, 0),
            TraceType::Any => return Err(TraceCompileError::NativeBackendRejected),
            _ => self
                .builder
                .ins()
                .load(types::I64, MemFlags::new(), area, at),
        })
    }

    /// The end of a side compiled apart that links back to the root's loop:
    /// it leaves through its poll exit, which restores the state and resumes
    /// at the loop header in the interpreter, or hands the root's code the
    /// loop parameters and returns [`LINK_CODE`].
    fn detached_link(
        &mut self,
        snapshot: crate::jit_ir::trace::SnapshotId,
        fuel: Value,
        poll_exit: Option<usize>,
    ) -> Result<(), TraceCompileError> {
        let spec = self.detached.expect("a side compiled apart");
        let poll_exit = poll_exit.ok_or(TraceCompileError::MissingPollSnapshot)?;
        let ir = self.ir;
        let remaining = self.builder.ins().iadd_imm(fuel, -1);
        if spec.exit_style {
            self.emit_exit(poll_exit, remaining);
            return Ok(());
        }
        let link = &ir.snapshots[snapshot.index()].frames[0];
        let root = self.members[0].ir;
        let mut next = Vec::with_capacity(root.loop_params.len());
        for param in &root.loop_params {
            let register = param
                .register
                .ok_or(TraceCompileError::NativeBackendRejected)?;
            next.push(match link.registers.get(usize::from(register)) {
                Some(RegisterEntry::Value(value)) => {
                    if ir.value_type(*value).is_some_and(TraceType::is_borrowed) {
                        return Err(TraceCompileError::NativeBackendRejected);
                    }
                    self.value(*value)
                }
                Some(RegisterEntry::Const(scalar)) => self.constant(*scalar),
                _ => return Err(TraceCompileError::NativeBackendRejected),
            });
        }
        let go = self
            .builder
            .ins()
            .icmp_imm(IntCC::SignedGreaterThan, remaining, 0);
        let linked = self.builder.create_block();
        let poll = self.builder.create_block();
        self.builder.set_cold_block(poll);
        self.builder.ins().brif(go, linked, &[], poll, &[]);
        self.builder.switch_to_block(poll);
        self.emit_exit(poll_exit, remaining);
        self.builder.switch_to_block(linked);
        let area = self.exit_area.expect("entry set the exit area");
        let control = self.control.expect("entry set the control block");
        self.store_link_writes(spec, link)?;
        for (index, (param, value)) in root.loop_params.iter().zip(next).enumerate() {
            let bits = match param.ty {
                TraceType::Nil | TraceType::Any => continue,
                TraceType::Boolean => self.builder.ins().uextend(types::I64, value),
                _ => value,
            };
            self.builder.ins().store(
                MemFlags::new(),
                bits,
                area,
                offset(index * mem::size_of::<i64>()),
            );
        }
        self.builder.ins().store(
            MemFlags::new(),
            remaining,
            control,
            offset(mem::offset_of!(TraceControl, fuel)),
        );
        let code = self.builder.ins().iconst(types::I64, LINK_CODE);
        self.builder.ins().return_(&[code]);
        Ok(())
    }

    /// Stores the registers the side writes beyond the root's loop parameters
    /// into the run's input slots, where the link leaves them: the root reads
    /// the entry inputs again when it is entered, the side reads its extras
    /// the next time it runs, and the VM writes the slots into the frames if
    /// the run leaves (see [`Writeback`]).
    fn store_link_writes(
        &mut self,
        spec: &DetachedSpec<'_>,
        link: &crate::jit_ir::trace::SnapshotFrame,
    ) -> Result<(), TraceCompileError> {
        let inputs = self.entry_inputs.expect("entry set the inputs");
        let slot_size = mem::size_of::<TraceInputSlot>();
        // A `next` loop keeps its control value in a lane of its own, which
        // the entry makes: its input slots are not the registers' values.
        if !spec.link_writes.is_empty()
            && (self.members[0].ir.boxed_control.is_some() || self.members[0].ir.boxed_value)
        {
            return Err(TraceCompileError::NativeBackendRejected);
        }
        for write in spec.link_writes {
            let allowed = matches!(
                write.ty,
                TraceType::Float
                    | TraceType::Integer
                    | TraceType::Boolean
                    | TraceType::Nil
                    | TraceType::Table
                    | TraceType::Closure
            ) && (write.caller.is_none()
                || matches!(
                    write.ty,
                    TraceType::Float | TraceType::Integer | TraceType::Boolean
                ));
            let bound = write.caller.is_none()
                && self.members[0]
                    .ir
                    .bindings
                    .iter()
                    .any(|binding| binding.register == write.register);
            if !allowed || bound {
                return Err(TraceCompileError::NativeBackendRejected);
            }
            let (source, entry) = match write.caller {
                Some(caller) => (ExtraSource::Caller(caller), None),
                None => (
                    ExtraSource::Register(write.register),
                    spec.root_entry
                        .iter()
                        .position(|(held, _)| *held == write.register),
                ),
            };
            let position = match entry {
                Some(position) if spec.root_entry[position].1 == write.ty => position,
                Some(_) => return Err(TraceCompileError::NativeBackendRejected),
                None => self.extra_position(spec, source, write.ty)?,
            };
            let writeback = Writeback { source, position };
            if !self.side_writebacks.contains(&writeback) {
                self.side_writebacks.push(writeback);
            }
            let bits = match link.registers.get(usize::from(write.register)) {
                Some(RegisterEntry::Value(value)) => {
                    if self.ir.value_type(*value) != Some(write.ty) {
                        return Err(TraceCompileError::NativeBackendRejected);
                    }
                    let value = self.value(*value);
                    match write.ty {
                        TraceType::Float => {
                            self.builder
                                .ins()
                                .bitcast(types::I64, MemFlags::new(), value)
                        }
                        TraceType::Boolean => self.builder.ins().uextend(types::I64, value),
                        TraceType::Nil => self.builder.ins().iconst(types::I64, 0),
                        _ => value,
                    }
                }
                Some(RegisterEntry::Const(scalar)) if scalar.ty() == write.ty => match write.ty {
                    TraceType::Float | TraceType::Integer | TraceType::Boolean => {
                        let value = self.constant(*scalar);
                        match write.ty {
                            TraceType::Float => {
                                self.builder
                                    .ins()
                                    .bitcast(types::I64, MemFlags::new(), value)
                            }
                            TraceType::Boolean => self.builder.ins().uextend(types::I64, value),
                            _ => value,
                        }
                    }
                    TraceType::Nil => self.builder.ins().iconst(types::I64, 0),
                    _ => return Err(TraceCompileError::NativeBackendRejected),
                },
                _ => return Err(TraceCompileError::NativeBackendRejected),
            };
            let tag = entry_tag(write.ty).ok_or(TraceCompileError::NativeBackendRejected)?;
            let tag = self.builder.ins().iconst(types::I64, tag);
            let at = offset(position * slot_size);
            self.builder.ins().store(
                MemFlags::new(),
                tag,
                inputs,
                at + offset(mem::offset_of!(TraceInputSlot, tag)),
            );
            self.builder.ins().store(
                MemFlags::new(),
                bits,
                inputs,
                at + offset(mem::offset_of!(TraceInputSlot, bits)),
            );
        }
        Ok(())
    }

    /// The values a side trace's entry inputs take when the guard `guard` of
    /// the current trace fails: its snapshot's entry for each input's
    /// register.
    ///
    /// A register that holds a number of run-time subtype starts a side
    /// recorded for one subtype. The second result is the condition that the
    /// number has that subtype, if any input needs one; the side runs only
    /// when it holds.
    fn side_inputs(
        &mut self,
        side: usize,
        guard: usize,
    ) -> Result<(Vec<Value>, Option<Value>), TraceCompileError> {
        let snapshot = self.ir.guards[guard].snapshot;
        self.side_inputs_at(side, snapshot)
    }

    /// [`Self::side_inputs`] for the state the snapshot `snapshot` of the
    /// current trace restores (a guard's, or its terminal exit's).
    fn side_inputs_at(
        &mut self,
        side: usize,
        snapshot: crate::jit_ir::trace::SnapshotId,
    ) -> Result<(Vec<Value>, Option<Value>), TraceCompileError> {
        let ir = self.ir;
        let snapshot = &ir.snapshots[snapshot.index()];
        let mut inputs = Vec::new();
        let mut subtypes: Option<Value> = None;
        for input in &self.members[side].ir.entry.inputs {
            let register = usize::from(input.register);
            if let Some(tag) = snapshot.number_tag(register) {
                let float = match snapshot.frames[0].registers.get(register) {
                    Some(RegisterEntry::Value(value)) => self.value(*value),
                    Some(RegisterEntry::Const(scalar)) => self.constant(*scalar),
                    _ => return Err(TraceCompileError::NativeBackendRejected),
                };
                let integer = match tag {
                    RegisterEntry::Value(value) => self.value(value),
                    RegisterEntry::Const(scalar) => self.constant(scalar),
                    _ => return Err(TraceCompileError::NativeBackendRejected),
                };
                let (value, holds) = match input.ty {
                    TraceType::Float => (float, self.builder.ins().bxor_imm(integer, 1)),
                    TraceType::Integer => (
                        self.builder.ins().fcvt_to_sint_sat(types::I64, float),
                        integer,
                    ),
                    _ => return Err(TraceCompileError::NativeBackendRejected),
                };
                subtypes = Some(match subtypes {
                    Some(held) => self.builder.ins().band(held, holds),
                    None => holds,
                });
                inputs.push(value);
                continue;
            }
            let value = match snapshot.frames[0].registers.get(register) {
                // A borrow goes to a borrowed input and only there: an owned
                // string is a lane id, and the two are not interchangeable.
                Some(RegisterEntry::Value(value))
                    if ir.value_type(*value).is_some_and(TraceType::is_borrowed)
                        != input.ty.is_borrowed() =>
                {
                    return Err(TraceCompileError::NativeBackendRejected);
                }
                Some(RegisterEntry::Value(value)) => self.value(*value),
                Some(RegisterEntry::Const(_) | RegisterEntry::Root(_))
                    if input.ty.is_borrowed() =>
                {
                    return Err(TraceCompileError::NativeBackendRejected);
                }
                Some(RegisterEntry::Const(scalar)) => self.constant(*scalar),
                // A string constant of the trace's roots is the lane id of
                // its slot (an owned string input, never a borrowed one).
                Some(RegisterEntry::Root(slot)) => self.constant(TraceScalar::String(
                    ROOT_STRING_LANE_BASE + u64::from(*slot),
                )),
                // The register is an invariant of this trace: an entry input.
                // A global a side reads and its parent does not is the root's.
                Some(RegisterEntry::Unchanged) => {
                    match ir
                        .entry
                        .inputs
                        .iter()
                        .find(|held| usize::from(held.register) == register)
                    {
                        Some(held) if held.ty.is_borrowed() != input.ty.is_borrowed() => {
                            return Err(TraceCompileError::NativeBackendRejected);
                        }
                        Some(held) => self.value(held.value),
                        None if input.ty.is_borrowed() => {
                            return Err(TraceCompileError::NativeBackendRejected);
                        }
                        None => *self
                            .root_inputs
                            .get(&input.register)
                            .ok_or(TraceCompileError::NativeBackendRejected)?,
                    }
                }
                _ => return Err(TraceCompileError::NativeBackendRejected),
            };
            inputs.push(value);
        }
        // The registers of the inlined frames the side starts inside, after
        // the window's: a value or a constant of the parent's state there.
        for input in &self.members[side].ir.frame_inputs {
            let value = match snapshot
                .frames
                .get(input.frame.index())
                .and_then(|frame| frame.registers.get(usize::from(input.register)))
            {
                Some(RegisterEntry::Value(value))
                    if ir.value_type(*value) == Some(input.ty) && !input.ty.is_borrowed() =>
                {
                    self.value(*value)
                }
                Some(RegisterEntry::Const(scalar)) if scalar.ty() == input.ty => {
                    self.constant(*scalar)
                }
                Some(RegisterEntry::Root(slot)) if input.ty == TraceType::String => self.constant(
                    TraceScalar::String(ROOT_STRING_LANE_BASE + u64::from(*slot)),
                ),
                _ => return Err(TraceCompileError::NativeBackendRejected),
            };
            inputs.push(value);
        }
        Ok((inputs, subtypes))
    }

    /// Adds a phantom table of `entries` entries to the control block's
    /// elided counters: one table, and the bytes of its footprint.
    fn count_phantom_table(&mut self, entries: u32) {
        let control = self.control.expect("entry set the control block");
        let bytes = i64::try_from(crate::runtime::Runtime::elided_footprint(entries as usize))
            .expect("a table footprint fits i64");
        for (field, amount) in [
            (mem::offset_of!(TraceControl, elided_tables), 1),
            (mem::offset_of!(TraceControl, elided_bytes), bytes),
        ] {
            let at = offset(field);
            let total = self
                .builder
                .ins()
                .load(types::I64, MemFlags::new(), control, at);
            let more = self.builder.ins().iadd_imm(total, amount);
            self.builder.ins().store(MemFlags::new(), more, control, at);
        }
    }

    /// Adds one to the control block's link counter.
    fn count_link(&mut self) {
        let control = self.control.expect("entry set the control block");
        let at = offset(mem::offset_of!(TraceControl, links));
        let links = self
            .builder
            .ins()
            .load(types::I64, MemFlags::new(), control, at);
        let more = self.builder.ins().iadd_imm(links, 1);
        self.builder.ins().store(MemFlags::new(), more, control, at);
    }

    /// The terminal exit of the current trace with the loop side `side`
    /// attached: the loop runs in place, entered with the state the exit
    /// `exit` restores (the snapshot `snapshot`).
    fn enter_loop_side(
        &mut self,
        side: usize,
        snapshot: crate::jit_ir::trace::SnapshotId,
        exit: usize,
        fuel: Value,
    ) -> Result<(), TraceCompileError> {
        let parent_origins = mem::take(&mut self.side_origins);
        let reloading = mem::replace(&mut self.reload_inputs, true);
        let inputs = self.side_inputs_at(side, snapshot);
        self.reload_inputs = reloading;
        let (inputs, subtypes) = inputs?;
        if let Some(holds) = subtypes {
            // A number of the other subtype than the loop was recorded for
            // leaves through the exit.
            let enter = self.builder.create_block();
            let other = self.builder.create_block();
            self.builder.set_cold_block(other);
            self.builder.ins().brif(holds, enter, &[], other, &[]);
            self.builder.switch_to_block(other);
            self.emit_exit(exit, fuel);
            self.builder.switch_to_block(enter);
        }
        if self.count_links {
            self.count_link();
        }
        let lowered = self.lower_side(side, &inputs, fuel);
        self.side_origins = parent_origins;
        lowered
    }

    /// Lowers a loop side: the loop invariant work once, then the loop on a
    /// header of its own that carries the loop parameters and the fuel. The
    /// loop leaves through the guards of its body, as the root's does.
    fn lower_loop_side(&mut self, fuel: Value) -> Result<(), TraceCompileError> {
        let ir = self.ir;
        // A parameter the loop-back passes on unchanged (one the tree widened
        // a member with, or an accumulator the optimizer found unchanged) is
        // an invariant the preheader may use, and the entry snapshot of its
        // hoisted guards names it. It is only defined at the header, and a
        // side that rejoins the loop there may bring it a new value, so the
        // preheader then runs at the header, as the root's does at its outer
        // header (see `lower`).
        let at_header = !pass_through_params(ir).is_empty();
        if !at_header {
            for (index, instruction) in ir.preheader.iter().enumerate() {
                self.lower_instruction(
                    &instruction.op,
                    instruction.result,
                    fuel,
                    &ir.preheader[index + 1..],
                )?;
            }
        }
        let mut initial: Vec<Value> = Vec::with_capacity(ir.loop_params.len() + 1);
        for param in &ir.loop_params {
            initial.push(match param.init {
                LoopParamInit::Input(value) => self.value(value),
                LoopParamInit::Const(scalar) => self.constant(scalar),
            });
        }
        initial.push(fuel);
        let header = self.builder.create_block();
        for param in &ir.loop_params {
            self.builder.append_block_param(header, clif_type(param.ty));
        }
        self.builder.append_block_param(header, types::I64);
        self.builder.ins().jump(header, &block_args(&initial));
        self.builder.switch_to_block(header);
        // What the code before the loop computed holds for the first
        // iteration only: a later one may have moved or retyped what it
        // reached.
        self.forget_array_words();
        self.slots.clear();
        self.row_addresses.clear();
        self.lane_array_cache.clear();
        self.field_slots.clear();
        self.next_slot = None;
        self.helper_results.clear();
        self.made_tables.clear();
        let parameters = self.builder.block_params(header).to_vec();
        for (index, param) in ir.loop_params.iter().enumerate() {
            self.define(param.value, parameters[index]);
        }
        let fuel = parameters[ir.loop_params.len()];
        if at_header {
            for (index, instruction) in ir.preheader.iter().enumerate() {
                self.lower_instruction(
                    &instruction.op,
                    instruction.result,
                    fuel,
                    &ir.preheader[index + 1..],
                )?;
            }
        }
        self.loop_header = Some(header);
        // The sides of this loop side's guards, lowered inside its body, may
        // rejoin its loop (`TraceTerminator::LinkLoop`) at this header.
        let params = ir
            .loop_params
            .iter()
            .map(|param| {
                param
                    .register
                    .map(|register| (register, param.ty))
                    .ok_or(TraceCompileError::NativeBackendRejected)
            })
            .collect::<Result<Vec<_>, _>>();
        let pushed = match params {
            Ok(params) => {
                self.loop_stack.push((self.member, header, params));
                true
            }
            // A parameter with no register cannot be linked to; no side of
            // this loop side may then link to it.
            Err(_) => false,
        };
        let lowered = self.lower_member_code(fuel);
        if pushed {
            self.loop_stack.pop();
        }
        lowered
    }

    /// Lowers side trace `side` in the current block, entered with `inputs`
    /// for its entry inputs. Every path of it ends in a jump or a return.
    fn lower_side(
        &mut self,
        side: usize,
        inputs: &[Value],
        fuel: Value,
    ) -> Result<(), TraceCompileError> {
        let ir = self.members[side].ir;
        let parent_ir = mem::replace(&mut self.ir, ir);
        let parent_member = mem::replace(&mut self.member, side);
        let parent_root_side = mem::replace(&mut self.root_side, parent_member == 0);
        let parent_values = mem::replace(&mut self.values, vec![None; ir.values.len()]);
        let parent_deferred = mem::take(&mut self.deferred);
        let parent_rematerialized = mem::take(&mut self.rematerialized);
        let parent_number_tags = mem::take(&mut self.number_tags);
        let parent_fuel_value = self.member_fuel;
        let parent_constants = mem::replace(&mut self.constants, vec![None; ir.values.len()]);
        // The side starts below the parent's loads, so it may use them. What
        // it loads itself does not reach back into the parent.
        let parent_cache = self.array_cache.clone();
        let parent_lane_cache = self.lane_array_cache.clone();
        // So may it use the row addresses the parent computed before the
        // guard, but the ones it computes stay in it.
        let parent_rows = self.row_addresses.clone();
        let parent_fields = self.field_slots.clone();
        // The slots the parent reached are named by the parent's keys.
        let parent_slots = mem::take(&mut self.slots);
        if self.fresh_words {
            // The side loads the words it reads itself, past the calls it
            // makes, so the loop's loads end where the loop does.
            self.forget_array_words();
        }
        // A self call in the side hands back fuel the parent's code after
        // the guard never sees.
        let parent_fuel = self.activation_fuel;
        let parent_deeper = self.deeper;
        let parent_lazy = mem::take(&mut self.lazy);
        // The extra results of a helper call are named by the side's own
        // value ids, which the parent's ids would collide with.
        let parent_results = mem::take(&mut self.helper_results);
        let parent_made = mem::take(&mut self.made_tables);
        let parent_loop = self.loop_header.take();
        for (input, value) in ir.entry.inputs.iter().zip(inputs) {
            // A side reads a register of either type only as the root's
            // entry input left it: its guard reads the root's slot.
            if input.ty == TraceType::Any
                && (!self.any_slots.contains_key(&input.register)
                    || self.root_inputs.get(&input.register) != Some(value))
            {
                return Err(TraceCompileError::NativeBackendRejected);
            }
            self.define(input.value, *value);
        }
        // The frame inputs follow the window's (see `side_inputs_at`).
        for (input, value) in ir
            .frame_inputs
            .iter()
            .zip(inputs.iter().skip(ir.entry.inputs.len()))
        {
            self.define(input.value, *value);
        }
        let result = if ir.loop_params.is_empty() {
            self.lower_member_code(fuel)
        } else {
            self.lower_loop_side(fuel)
        };
        self.loop_header = parent_loop;
        self.activation_fuel = parent_fuel;
        self.deeper = parent_deeper;
        self.lazy = parent_lazy;
        self.helper_results = parent_results;
        self.made_tables = parent_made;
        self.slots = parent_slots;
        self.root_side = parent_root_side;
        self.ir = parent_ir;
        self.member = parent_member;
        self.values = parent_values;
        self.deferred = parent_deferred;
        self.rematerialized = parent_rematerialized;
        self.number_tags = parent_number_tags;
        self.member_fuel = parent_fuel_value;
        self.constants = parent_constants;
        self.array_cache = parent_cache;
        self.lane_array_cache = parent_lane_cache;
        self.row_addresses = parent_rows;
        self.field_slots = parent_fields;
        result
    }

    /// For a counted trace, loads the count hook's budget the VM left in the
    /// control block: the budget at the loop header, which the iteration then
    /// pays out of.
    fn load_entry_budget(&mut self, control: Value) {
        if self.ir.counted {
            self.budget = Some(self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                control,
                offset(mem::offset_of!(TraceControl, budget)),
            ));
        }
    }

    /// Entry: check every input's tag, load its bits, and enter the loop.
    fn lower_entry(&mut self, entry: Block, header: Block) -> Result<(), TraceCompileError> {
        let ir = self.ir;
        self.builder.switch_to_block(entry);
        let parameters = self.builder.block_params(entry).to_vec();
        let (inputs, control, exit_area) = (parameters[0], parameters[1], parameters[2]);
        self.control = Some(control);
        self.exit_area = Some(exit_area);
        // Nothing in a run moves the rows, so one load serves every access.
        self.rows = Some(self.builder.ins().load(
            types::I64,
            MemFlags::new().with_readonly(),
            control,
            offset(mem::offset_of!(TraceControl, rows)),
        ));
        let fuel = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(mem::offset_of!(TraceControl, fuel)),
        );
        self.load_entry_budget(control);
        // One branch for every tag: the bits that differ from the required
        // tags, or-ed together. The VM finds which input it was (see
        // `CompiledTrace::run_checked`), so a mismatch returns the first
        // input's code, and the entry is one block rather than one per input.
        let slot_size = mem::size_of::<TraceInputSlot>();
        let mut differs: Option<Value> = None;
        for (index, input) in ir.entry.inputs.iter().enumerate() {
            if input.ty == TraceType::Any {
                continue;
            }
            let required = entry_tag(input.ty).ok_or(TraceCompileError::NativeBackendRejected)?;
            let base = offset(index * slot_size);
            let tag = self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                inputs,
                base + offset(mem::offset_of!(TraceInputSlot, tag)),
            );
            let difference = self.builder.ins().bxor_imm(tag, required);
            differs = Some(match differs {
                Some(earlier) => self.builder.ins().bor(earlier, difference),
                None => difference,
            });
        }
        if let Some(differs) = differs {
            let accepted = self.builder.create_block();
            let mismatch = self.builder.create_block();
            self.builder.set_cold_block(mismatch);
            self.builder
                .ins()
                .brif(differs, mismatch, &[], accepted, &[]);
            self.builder.switch_to_block(mismatch);
            let code = self.builder.ins().iconst(types::I64, -1);
            self.builder.ins().return_(&[code]);
            self.builder.switch_to_block(accepted);
        }
        for (index, input) in ir.entry.inputs.iter().enumerate() {
            if input.ty == TraceType::Any {
                // Only type guards read it (the verifier holds it to that),
                // and they read the slot.
                let placeholder = self.builder.ins().iconst(types::I8, 0);
                self.define(input.value, placeholder);
                self.root_inputs.insert(input.register, placeholder);
                self.any_slots.insert(input.register, index);
                continue;
            }
            let value = self.load_entry_input(inputs, index, input.ty);
            self.define(input.value, value);
            self.root_inputs.insert(input.register, value);
            if input.ty != TraceType::Nil {
                self.entry_slots.insert(input.value, (index, input.ty));
            }
        }
        self.entry_inputs = Some(inputs);
        self.load_bindings(inputs);
        self.check_trusted_key();
        // Loop-invariant work runs once, before the first iteration. A guard
        // in it leaves with the entry state and the full fuel. A loop with
        // parameters its iteration passes on unchanged runs it at the header
        // instead, where each link comes back to (see `lower`).
        if pass_through_params(ir).is_empty() {
            for (index, instruction) in ir.preheader.iter().enumerate() {
                self.lower_instruction(
                    &instruction.op,
                    instruction.result,
                    fuel,
                    &ir.preheader[index + 1..],
                )?;
            }
        }
        let mut initial: Vec<Value> = Vec::with_capacity(ir.loop_params.len() + 1);
        for param in &ir.loop_params {
            initial.push(match param.init {
                LoopParamInit::Input(value) => self.value(value),
                LoopParamInit::Const(scalar) => self.constant(scalar),
            });
        }
        initial.push(fuel);
        if let Some(budget) = self.budget {
            initial.push(budget);
        }
        self.builder.ins().jump(header, &block_args(&initial));
        Ok(())
    }

    /// The guard that leaves through `guard` unless the root's entry input
    /// of either type that `source` (an entry input of the member being
    /// lowered) stands for has type `expected`, and the input as a value of
    /// that type. The tag and the bits come from the input's slot, which
    /// holds the run's entry state: no member writes the register.
    fn guard_any_input(
        &mut self,
        source: ValueId,
        expected: TraceType,
        guard: GuardId,
        fuel: Value,
    ) -> Result<Value, TraceCompileError> {
        let register = match self.ir.values.get(source.index()).map(|info| info.def) {
            Some(ValueDef::Input(index)) => self.ir.entry.inputs[index].register,
            _ => return Err(TraceCompileError::NativeBackendRejected),
        };
        let (Some(&slot), Some(inputs), Some(required)) = (
            self.any_slots.get(&register),
            self.entry_inputs,
            entry_tag(expected),
        ) else {
            return Err(TraceCompileError::NativeBackendRejected);
        };
        let at = offset(slot * mem::size_of::<TraceInputSlot>())
            + offset(mem::offset_of!(TraceInputSlot, tag));
        let tag = self
            .builder
            .ins()
            .load(types::I64, MemFlags::new(), inputs, at);
        let holds = self.builder.ins().icmp_imm(IntCC::Equal, tag, required);
        let pass = self.builder.create_block();
        let fail = self.builder.create_block();
        self.builder.set_cold_block(fail);
        if self.forced_guard == Some((self.member, guard.index())) {
            let always = self.builder.ins().iconst(types::I8, 1);
            self.builder.ins().brif(always, fail, &[], pass, &[]);
        } else {
            self.builder.ins().brif(holds, pass, &[], fail, &[]);
        }
        self.builder.switch_to_block(fail);
        self.leave(guard, fuel)?;
        self.builder.switch_to_block(pass);
        Ok(self.load_entry_input(inputs, slot, expected))
    }

    /// Stores into the root's input slot of each register the root takes
    /// untyped (see [`Self::guard_any_input`]) the value the side's link
    /// state `link` gives it, so that the root's type guard reads it at the
    /// header and the run leaves it for the VM (see
    /// [`crate::jit_ir::trace::untyped_link_writes`]). Whether it stored any.
    fn store_untyped_link(
        &mut self,
        link: &crate::jit_ir::trace::SnapshotFrame,
    ) -> Result<bool, TraceCompileError> {
        let root = self.members[0].ir;
        let mut stored = false;
        for input in &root.entry.inputs {
            if input.ty != TraceType::Any {
                continue;
            }
            let (value, ty) = match link.registers.get(usize::from(input.register)) {
                Some(RegisterEntry::Value(value)) => {
                    let ty = self
                        .ir
                        .value_type(*value)
                        .ok_or(TraceCompileError::NativeBackendRejected)?;
                    // The side passes on the slot as it found it.
                    if ty == TraceType::Any {
                        continue;
                    }
                    (self.value(*value), ty)
                }
                Some(RegisterEntry::Const(scalar)) => (self.constant(*scalar), scalar.ty()),
                Some(RegisterEntry::Unchanged) | None => continue,
                Some(_) => return Err(TraceCompileError::NativeBackendRejected),
            };
            if !matches!(
                ty,
                TraceType::Nil | TraceType::Float | TraceType::Integer | TraceType::Boolean
            ) {
                return Err(TraceCompileError::NativeBackendRejected);
            }
            let (Some(&slot), Some(inputs)) =
                (self.any_slots.get(&input.register), self.entry_inputs)
            else {
                return Err(TraceCompileError::NativeBackendRejected);
            };
            let (tag, bits) = self.scalar_word(ty, value);
            let at = offset(slot * mem::size_of::<TraceInputSlot>());
            self.builder.ins().store(
                MemFlags::new(),
                tag,
                inputs,
                at + offset(mem::offset_of!(TraceInputSlot, tag)),
            );
            self.builder.ins().store(
                MemFlags::new(),
                bits,
                inputs,
                at + offset(mem::offset_of!(TraceInputSlot, bits)),
            );
            stored = true;
        }
        Ok(stored)
    }

    /// The entry input in slot `index` of `inputs`, of type `ty`.
    fn load_entry_input(&mut self, inputs: Value, index: usize, ty: TraceType) -> Value {
        let at = offset(index * mem::size_of::<TraceInputSlot>())
            + offset(mem::offset_of!(TraceInputSlot, bits));
        match ty {
            TraceType::Float => self
                .builder
                .ins()
                .load(types::F64, MemFlags::new(), inputs, at),
            TraceType::Boolean => {
                let bits = self
                    .builder
                    .ins()
                    .load(types::I64, MemFlags::new(), inputs, at);
                self.builder.ins().icmp_imm(IntCC::NotEqual, bits, 0)
            }
            TraceType::Nil | TraceType::Any => self.builder.ins().iconst(types::I8, 0),
            _ => self
                .builder
                .ins()
                .load(types::I64, MemFlags::new(), inputs, at),
        }
    }

    /// Loads the words the VM resolved for each table binding, which follow
    /// the entry inputs in the input slots.
    fn load_bindings(&mut self, inputs: Value) {
        // The bindings and the entry inputs are the root's, whichever trace
        // of the tree is being lowered.
        let ir = self.members[0].ir;
        let slot_size = mem::size_of::<TraceInputSlot>();
        // The VM resolved each bound table's storage before the call. Its
        // words follow the entry inputs.
        let first = ir.entry.inputs.len();
        self.words = Some(
            self.builder
                .ins()
                .iadd_imm(inputs, i64::from(offset(first * slot_size))),
        );
        for (index, binding) in ir.bindings.iter().enumerate() {
            let base = offset((first + index) * slot_size);
            let aux_at = base + offset(mem::offset_of!(TraceInputSlot, tag));
            let bits_at = base + offset(mem::offset_of!(TraceInputSlot, bits));
            let mut length_float = None;
            let words = match &binding.kind {
                BindingKind::Array { .. } | BindingKind::Bytes { .. } => {
                    let address =
                        self.builder
                            .ins()
                            .load(types::I64, MemFlags::new(), inputs, bits_at);
                    let length =
                        self.builder
                            .ins()
                            .load(types::I64, MemFlags::new(), inputs, aux_at);
                    length_float = Some(self.builder.ins().fcvt_from_sint(types::F64, length));
                    (address, length)
                }
                BindingKind::Field { .. } => {
                    let address =
                        self.builder
                            .ins()
                            .load(types::I64, MemFlags::new(), inputs, bits_at);
                    (address, address)
                }
                BindingKind::Header { .. } => {
                    // The table's address; whether it is appendable is read
                    // again where a set uses it, since the VM rewrites it.
                    let address =
                        self.builder
                            .ins()
                            .load(types::I64, MemFlags::new(), inputs, bits_at);
                    (address, address)
                }
                BindingKind::NoMetatable | BindingKind::Hash if self.native_probe => {
                    // The string index of the table: the address of its slots
                    // (zero for none) and its mask. A lookup reads them.
                    let slots =
                        self.builder
                            .ins()
                            .load(types::I64, MemFlags::new(), inputs, bits_at);
                    let mask = self
                        .builder
                        .ins()
                        .load(types::I64, MemFlags::new(), inputs, aux_at);
                    (slots, mask)
                }
                BindingKind::Callee { .. }
                | BindingKind::StringField { .. }
                | BindingKind::MetaClass { .. }
                | BindingKind::NoMetatable
                | BindingKind::Hash => {
                    // Entry checked the table. No operation reads it.
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    (zero, zero)
                }
                BindingKind::Length { ty, .. } => {
                    let clif = clif_type(*ty);
                    let length = self
                        .builder
                        .ins()
                        .load(clif, MemFlags::new(), inputs, bits_at);
                    (length, length)
                }
            };
            self.bound.push(words);
            self.bound_length_float.push(length_float);
        }
    }

    #[allow(clippy::too_many_lines)]
    fn lower_instruction(
        &mut self,
        op: &TraceOp,
        result: Option<ValueId>,
        fuel: Value,
        ahead: &[TraceInstruction],
    ) -> Result<(), TraceCompileError> {
        if !super::census::enabled() {
            return self.lower_instruction_inner(op, result, fuel, ahead);
        }
        let before = self.builder.func.dfg.num_insts();
        super::census::enter();
        let lowered = self.lower_instruction_inner(op, result, fuel, ahead);
        super::census::exit(op, self.builder.func.dfg.num_insts() - before);
        lowered
    }

    fn lower_instruction_inner(
        &mut self,
        op: &TraceOp,
        result: Option<ValueId>,
        fuel: Value,
        ahead: &[TraceInstruction],
    ) -> Result<(), TraceCompileError> {
        let ty_of = |this: &Self, value: ValueId| {
            this.ir
                .value_type(value)
                .expect("a verified trace types every operand")
        };
        // A held delete waits only for pure operations and its store.
        let held = match &self.pending_delete {
            None => true,
            Some(delete) => match op {
                TraceOp::Constant(_)
                | TraceOp::Move { .. }
                | TraceOp::Arithmetic { .. }
                | TraceOp::Compare { .. }
                | TraceOp::Select { .. } => true,
                TraceOp::CallHelper { guard, .. } => *guard == delete.store,
                _ => false,
            },
        };
        if !held {
            self.flush_pending_delete(fuel)?;
        }
        let fresh = self.next_slot.take();
        if !keeps_slots(op) {
            self.slots.clear();
            // Reading or overwriting a field in place leaves every slot where
            // and what it was.
            if !matches!(
                op,
                TraceOp::TableFieldLoad { .. }
                    | TraceOp::TableFieldStore { .. }
                    | TraceOp::LaneArrayStore { .. }
                    | TraceOp::MetaGuard { .. }
            ) {
                self.field_slots.clear();
            }
        }
        let lowered = match op {
            TraceOp::Constant(scalar) => {
                if let Some(result) = result {
                    self.constants[result.index()] = Some(*scalar);
                }
                Some(self.constant(*scalar))
            }
            TraceOp::GuardType {
                source,
                expected,
                guard,
            } if ty_of(self, *source) == TraceType::Any => {
                Some(self.guard_any_input(*source, *expected, *guard, fuel)?)
            }
            // A typed source can never fail the guard. It only narrows.
            TraceOp::Move { source } | TraceOp::GuardType { source, .. } => {
                Some(self.value(*source))
            }
            TraceOp::Negate { source } => {
                let source_value = self.value(*source);
                Some(match ty_of(self, *source) {
                    TraceType::Integer | TraceType::Int64 | TraceType::UInt64 => {
                        self.builder.ins().ineg(source_value)
                    }
                    _ => self.builder.ins().fneg(source_value),
                })
            }
            TraceOp::Arithmetic { op, left, right } => {
                let (left_value, right_value) = (self.value(*left), self.value(*right));
                Some(match ty_of(self, *left) {
                    TraceType::Integer => self.integer_arithmetic(*op, left_value, right_value)?,
                    ty @ (TraceType::Int64 | TraceType::UInt64) => {
                        self.wide_arithmetic(*op, ty == TraceType::Int64, left_value, right_value)
                    }
                    _ => self.float_arithmetic(*op, left_value, right_value),
                })
            }
            TraceOp::IntToFloat { source } => {
                let source_value = self.value(*source);
                Some(if ty_of(self, *source) == TraceType::UInt64 {
                    self.builder.ins().fcvt_from_uint(types::F64, source_value)
                } else {
                    self.builder.ins().fcvt_from_sint(types::F64, source_value)
                })
            }
            TraceOp::Numeric { op, source } => {
                let source_value = self.value(*source);
                Some(match op {
                    NumericOp::Floor => self.builder.ins().floor(source_value),
                    NumericOp::Ceil => self.builder.ins().ceil(source_value),
                    NumericOp::Abs if ty_of(self, *source) == TraceType::Integer => {
                        self.builder.ins().iabs(source_value)
                    }
                    NumericOp::Abs => self.builder.ins().fabs(source_value),
                    NumericOp::FloatToInteger => self
                        .builder
                        .ins()
                        .fcvt_to_sint_sat(types::I64, source_value),
                    NumericOp::TrickBits => {
                        let bias = self.builder.ins().f64const(6_755_399_441_055_744.0);
                        let sum = self.builder.ins().fadd(source_value, bias);
                        let packed = self.builder.ins().bitcast(types::I64, MemFlags::new(), sum);
                        self.builder.ins().band_imm(packed, 0xffff_ffff)
                    }
                })
            }
            TraceOp::ToWide {
                source,
                target,
                conversion,
            } => {
                let source_value = self.value(*source);
                Some(match (ty_of(self, *source), conversion) {
                    (TraceType::Float, WideConversion::Truncate) => {
                        self.float_to_wide(source_value, *target == TraceType::Int64)
                    }
                    (TraceType::Float, WideConversion::Bits32) => {
                        let address = self.native_address(trace_number_bits as *const () as usize);
                        let mut signature = Signature::new(self.builder.func.signature.call_conv);
                        signature.params.push(AbiParam::new(types::F64));
                        signature.returns.push(AbiParam::new(types::I64));
                        let signature = self.builder.import_signature(signature);
                        let call =
                            self.builder
                                .ins()
                                .call_indirect(signature, address, &[source_value]);
                        self.builder.inst_results(call)[0]
                    }
                    // The other cdata keeps its bits.
                    _ => source_value,
                })
            }
            TraceOp::NativeUnary { function, source } => {
                let argument = self.value(*source);
                // `math.sqrt` is the one library function the machine has an
                // instruction for. `sqrtsd` is correctly rounded like the
                // `f64::sqrt` the native calls, and a negative argument gives
                // the same NaN, so the call and its register shuffle go.
                if function.address() == crate::stdlib::numeric_sqrt_address() {
                    Some(self.builder.ins().sqrt(argument))
                } else {
                    let target = self.native_address(function.address());
                    let signature = self.native_signature(1);
                    let call = self
                        .builder
                        .ins()
                        .call_indirect(signature, target, &[argument]);
                    Some(self.builder.inst_results(call)[0])
                }
            }
            TraceOp::NativeBinary {
                function,
                left,
                right,
            } => {
                let (left_value, right_value) = (self.value(*left), self.value(*right));
                let target = self.native_address(function.address());
                let signature = self.native_signature(2);
                let call =
                    self.builder
                        .ins()
                        .call_indirect(signature, target, &[left_value, right_value]);
                Some(self.builder.inst_results(call)[0])
            }
            TraceOp::Compare {
                comparison,
                left,
                right,
            } => {
                let (left_value, right_value) = (self.value(*left), self.value(*right));
                let (left_ty, right_ty) = (ty_of(self, *left), ty_of(self, *right));
                Some(if left_ty != right_ty {
                    debug_assert!(!left_ty.is_wide() && !right_ty.is_wide());
                    // An integer against a float: exact, so not a conversion.
                    if left_ty == TraceType::Integer {
                        self.compare_integer_float(*comparison, left_value, right_value)
                    } else {
                        self.compare_float_integer(*comparison, left_value, right_value)
                    }
                } else if left_ty == TraceType::Float {
                    let condition = match comparison {
                        Comparison::Equal => FloatCC::Equal,
                        Comparison::LessThan => FloatCC::LessThan,
                        Comparison::LessEqual => FloatCC::LessThanOrEqual,
                    };
                    self.builder.ins().fcmp(condition, left_value, right_value)
                } else {
                    let unsigned = left_ty == TraceType::UInt64;
                    let condition = match (comparison, unsigned) {
                        (Comparison::Equal, _) => IntCC::Equal,
                        (Comparison::LessThan, false) => IntCC::SignedLessThan,
                        (Comparison::LessEqual, false) => IntCC::SignedLessThanOrEqual,
                        (Comparison::LessThan, true) => IntCC::UnsignedLessThan,
                        (Comparison::LessEqual, true) => IntCC::UnsignedLessThanOrEqual,
                    };
                    self.builder.ins().icmp(condition, left_value, right_value)
                })
            }
            TraceOp::Truthy { source } => Some(match ty_of(self, *source) {
                TraceType::Boolean => self.value(*source),
                TraceType::Nil => self.builder.ins().iconst(types::I8, 0),
                _ => self.builder.ins().iconst(types::I8, 1),
            }),
            TraceOp::Not { source } => Some(match ty_of(self, *source) {
                TraceType::Boolean => {
                    let source_value = self.value(*source);
                    self.builder.ins().icmp_imm(IntCC::Equal, source_value, 0)
                }
                TraceType::Nil => self.builder.ins().iconst(types::I8, 1),
                _ => self.builder.ins().iconst(types::I8, 0),
            }),
            // Both arms are already values: a branch-free pick.
            TraceOp::Select {
                condition,
                if_true,
                if_false,
            } => {
                let condition = self.value(*condition);
                let (if_true, if_false) = (self.value(*if_true), self.value(*if_false));
                Some(self.builder.ins().select(condition, if_true, if_false))
            }
            TraceOp::GuardBranch {
                condition,
                expected,
                guard,
            } => {
                let condition = self.value(*condition);
                let pass = self.builder.create_block();
                let fail = self.builder.create_block();
                self.builder.set_cold_block(fail);
                if self.forced_guard == Some((self.member, guard.index())) {
                    let always = self.builder.ins().iconst(types::I8, 1);
                    self.builder.ins().brif(always, fail, &[], pass, &[]);
                } else if *expected {
                    self.builder.ins().brif(condition, pass, &[], fail, &[]);
                } else {
                    self.builder.ins().brif(condition, fail, &[], pass, &[]);
                }
                self.builder.switch_to_block(fail);
                self.leave(*guard, fuel)?;
                self.builder.switch_to_block(pass);
                None
            }
            TraceOp::ArrayLoad {
                binding,
                key,
                expected,
                past_end,
                guard,
                ..
            } => match fresh {
                // The element of the slot the step before just found.
                Some((step, found, (slot, tag)))
                    if step == *key
                        && found == *binding
                        && !*past_end
                        && self.forced_guard != Some((self.member, guard.index())) =>
                {
                    Some(self.slot_read((slot, tag), *expected, *guard, fuel)?)
                }
                _ if *past_end => Some(self.array_nil_read(*binding, *key, *guard, fuel)?),
                _ => Some(self.array_access(*binding, *key, *expected, None, *guard, fuel)?),
            },
            TraceOp::LaneArrayLoad {
                table,
                field,
                key,
                expected,
                guard,
            } => {
                Some(self.lane_array_load((*table, *field), *key, *expected, None, *guard, fuel)?)
            }
            TraceOp::LaneArrayStore {
                table,
                field,
                key,
                value,
                guard,
                ..
            } => {
                let ty = ty_of(self, *value);
                let stored = self.value(*value);
                self.lane_array_load((*table, *field), *key, ty, Some(stored), *guard, fuel)?;
                None
            }
            TraceOp::NumberLoad {
                binding,
                key,
                guard,
                ..
            } => {
                let (number, is_integer) = self.number_load(*binding, *key, *guard, fuel)?;
                if let Some(result) = result {
                    self.number_tags.insert(result, is_integer);
                }
                Some(number)
            }
            TraceOp::NumberTag { load } => Some(
                *self
                    .number_tags
                    .get(load)
                    .ok_or(TraceCompileError::NativeBackendRejected)?,
            ),
            TraceOp::BoolAnd { left, right } => {
                let (left, right) = (self.value(*left), self.value(*right));
                Some(self.builder.ins().band(left, right))
            }
            TraceOp::NumberModulo {
                float,
                left,
                right,
                integer,
            } => {
                let (left, right, integer) =
                    (self.value(*left), self.value(*right), self.value(*integer));
                Some(self.number_modulo(*float, left, right, integer)?)
            }
            TraceOp::ArrayNext {
                binding,
                key,
                start,
                ty,
                guard,
                ..
            } => {
                let (number, slot) =
                    self.array_next(*binding, *key, (*start, *ty), *guard, fuel)?;
                self.next_slot = result.map(|result| (result, *binding, slot));
                Some(number)
            }
            TraceOp::HashLoad {
                binding,
                key,
                expected,
                guard,
                ..
            } => Some(self.hash_load(*binding, *key, *expected, *guard, fuel)?),
            TraceOp::TableFieldLoad {
                field,
                table,
                guard,
            } => {
                let ty = self.members[0].ir.fields[*field as usize].ty;
                let slot = self.field_slot(*table, *field, *guard, fuel, ahead)?;
                match ty {
                    // The row's marker says the key is absent and the table
                    // has no metatable: there is nothing to read.
                    TraceType::Nil => Some(self.builder.ins().iconst(types::I8, 0)),
                    TraceType::Table => {
                        let fail = self.builder.create_block();
                        self.builder.set_cold_block(fail);
                        Some(self.slot_access(slot, fail, ty, None, *guard, fuel)?)
                    }
                    _ => Some(self.builder.ins().load(
                        clif_type(ty),
                        MemFlags::new(),
                        slot,
                        NATIVE_VALUE_PAYLOAD_OFFSET,
                    )),
                }
            }
            TraceOp::TableFieldStore {
                field,
                table,
                value,
                guard,
                ..
            } => {
                let slot = self.field_slot(*table, *field, *guard, fuel, ahead)?;
                let stored = self.value(*value);
                self.builder.ins().store(
                    MemFlags::new(),
                    stored,
                    slot,
                    NATIVE_VALUE_PAYLOAD_OFFSET,
                );
                None
            }
            TraceOp::MetaGuard {
                table,
                binding,
                guard,
            } => {
                self.meta_guard(*table, *binding, *guard, fuel)?;
                None
            }
            TraceOp::ArrayStore {
                binding,
                key,
                value,
                guard,
                ..
            } => {
                let ty = ty_of(self, *value);
                let stored = self.value(*value);
                self.array_access(*binding, *key, ty, Some(stored), *guard, fuel)?;
                None
            }
            TraceOp::ArrayOverwrite {
                binding,
                key,
                value,
                ..
            } => {
                let ty = ty_of(self, *value);
                let stored = self.value(*value);
                self.array_overwrite(*binding, *key, ty, stored);
                None
            }
            TraceOp::ArraySet {
                binding,
                key,
                value,
                guard,
                ..
            } => {
                self.array_set(*binding, *key, *value, *guard, fuel)?;
                None
            }
            TraceOp::FieldLoad { binding, .. } => {
                let BindingKind::Field { ty, .. } =
                    &self.members[0].ir.bindings[*binding as usize].kind
                else {
                    return Err(TraceCompileError::NativeBackendRejected);
                };
                let ty = *ty;
                let slot = self.bound[*binding as usize].0;
                Some(self.builder.ins().load(
                    clif_type(ty),
                    MemFlags::new(),
                    slot,
                    NATIVE_VALUE_PAYLOAD_OFFSET,
                ))
            }
            TraceOp::FieldStore { binding, value, .. } => {
                let slot = self.bound[*binding as usize].0;
                let stored = self.value(*value);
                self.builder.ins().store(
                    MemFlags::new(),
                    stored,
                    slot,
                    NATIVE_VALUE_PAYLOAD_OFFSET,
                );
                None
            }
            TraceOp::PhantomTable { entries } => {
                self.count_phantom_table(*entries);
                None
            }
            // Read once at entry, unless a store may resize a table: the VM
            // then rewrites the word, which is read again at every use.
            TraceOp::TableLength { binding, .. } if self.reload_arrays => {
                let BindingKind::Length { ty, .. } =
                    &self.members[0].ir.bindings[*binding as usize].kind
                else {
                    return Err(TraceCompileError::NativeBackendRejected);
                };
                let words = self.words.expect("entry located the binding slots");
                let at = offset(
                    *binding as usize * mem::size_of::<TraceInputSlot>()
                        + mem::offset_of!(TraceInputSlot, bits),
                );
                Some(
                    self.builder
                        .ins()
                        .load(clif_type(*ty), MemFlags::new(), words, at),
                )
            }
            TraceOp::TableLength { binding, .. } => Some(self.bound[*binding as usize].0),
            // A string never changes, so its length is the entry's word.
            TraceOp::StringLength { binding, .. } => {
                let BindingKind::Bytes { ty } =
                    &self.members[0].ir.bindings[*binding as usize].kind
                else {
                    return Err(TraceCompileError::NativeBackendRejected);
                };
                Some(if *ty == TraceType::Float {
                    self.bound_length_float[*binding as usize]
                        .expect("a bytes binding has its length as a float")
                } else {
                    self.bound[*binding as usize].1
                })
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
                    return Err(TraceCompileError::NativeBackendRejected);
                };
                let ty = *ty;
                let (index, fail, base) = self.bound_index(*binding, *key, *guard);
                let pass = self.builder.create_block();
                self.builder.ins().jump(pass, &[]);
                self.builder.switch_to_block(fail);
                let exit = self.plan.members[self.member].first + guard.index();
                self.emit_exit(exit, fuel);
                self.builder.switch_to_block(pass);
                let address = self.builder.ins().iadd(base, index);
                let byte = self
                    .builder
                    .ins()
                    .uload8(types::I64, MemFlags::new(), address, 0);
                Some(if ty == TraceType::Float {
                    self.builder.ins().fcvt_from_uint(types::F64, byte)
                } else {
                    byte
                })
            }
            TraceOp::HostCall {
                function,
                args,
                result,
                guard,
            } => Some(self.host_call(*function, args, *result, *guard, fuel)?),
            TraceOp::CallHelper {
                helper,
                args,
                guard,
                ..
            } => self.call_helper(*helper, args, result, *guard, fuel)?,
            TraceOp::HelperResult { call, index } => {
                let extras = self
                    .helper_results
                    .get(call)
                    .ok_or(TraceCompileError::NativeBackendRejected)?;
                Some(extras[usize::from(*index) - 1])
            }
            TraceOp::ReleaseStrings { keep } => {
                self.release_strings(keep, fuel);
                None
            }
            TraceOp::SelfCall { args, guard } => self.self_call(args, *guard, fuel)?,
            TraceOp::Read { .. } | TraceOp::Effect { .. } => {
                return Err(TraceCompileError::NativeBackendRejected);
            }
        };
        if let (Some(result), Some(lowered)) = (result, lowered) {
            self.define(result, lowered);
        }
        Ok(())
    }

    /// The zero-based index `key` names in the array or bytes bound as
    /// `binding`, and the cold block that leaves through the guard's exit
    /// (the caller fills it) unless `key` is an integral number inside them.
    ///
    /// Also returns the address of slot 1.
    fn bound_index(&mut self, binding: u32, key: ValueId, guard: GuardId) -> (Value, Block, Value) {
        let (base, length, length_float) = self.array_words(binding);
        let fail = self.builder.create_block();
        self.builder.set_cold_block(fail);
        let index = self.index_within((length, length_float), key, Some(guard), fail);
        (index, fail, base)
    }

    /// The zero-based index `key` names in an array of `length` slots
    /// (`length_float` is the count as a float, needed for a float key), with
    /// a branch to `fail` unless `key` is an integral number inside it. The
    /// words come from an array binding ([`Self::bound_index`]) or from a
    /// table met by lane ([`Self::lane_array_words`]).
    fn index_within(
        &mut self,
        (length, length_float): (Value, Option<Value>),
        key: ValueId,
        guard: Option<GuardId>,
        fail: Block,
    ) -> Value {
        let key_value = self.value(key);
        let key_ty = self
            .ir
            .value_type(key)
            .expect("a verified trace types every operand");
        #[cfg(feature = "op-stats")]
        self.count_bound_check(key);
        // Lua's index `k` is slot `k - 1`. An integer key needs one unsigned
        // compare: a key below 1 wraps past every length. A float key must
        // also be exactly integral, which is checked without a conversion
        // instruction (a saturating float-to-integer conversion costs a
        // sequence on x86): inside `1 <= k <= len` it is below 2^52, and
        // adding 2^52 puts the integer in the low bits of the sum, which
        // is exact only for an integral `k`. NaN fails the ordered compares.
        let forced =
            guard.is_some_and(|guard| self.forced_guard == Some((self.member, guard.index())));
        // Each condition is its own branch to the shared exit, which the
        // backend fuses with its compare. One boolean built from the three
        // would cost a flag-to-register move and an `and` for each.
        let branch_unless = |this: &mut Self, condition: Value| {
            let next = this.builder.create_block();
            this.builder.ins().brif(condition, next, &[], fail, &[]);
            this.builder.switch_to_block(next);
        };
        if forced {
            let never = self.builder.ins().iconst(types::I8, 0);
            branch_unless(self, never);
        }
        let constant_slot = match self.constants[key.index()] {
            Some(TraceScalar::Float(number))
                if (1.0..=FLOAT_INTEGER_MAGIC).contains(&number) && number.fract() == 0.0 =>
            {
                Some(number as i64 - 1)
            }
            Some(TraceScalar::Integer(number)) if number >= 1 => Some(number - 1),
            _ => None,
        };
        if let Some(slot) = constant_slot {
            // A constant key is known to be integral: only the length is
            // left to check.
            let index = self.builder.ins().iconst(types::I64, slot);
            let in_range = self
                .builder
                .ins()
                .icmp(IntCC::UnsignedLessThan, index, length);
            branch_unless(self, in_range);
            index
        } else if key_ty == TraceType::Float {
            #[cfg(feature = "op-stats")]
            self.bump_counter(op_stats::Counter::NativeIntegralityChecks);
            let length_float = length_float.expect("an array binding has its length as a float");
            let one = self.builder.ins().f64const(1.0);
            let at_least_one = self
                .builder
                .ins()
                .fcmp(FloatCC::GreaterThanOrEqual, key_value, one);
            branch_unless(self, at_least_one);
            let at_most_length =
                self.builder
                    .ins()
                    .fcmp(FloatCC::LessThanOrEqual, key_value, length_float);
            branch_unless(self, at_most_length);
            let magic = self.builder.ins().f64const(FLOAT_INTEGER_MAGIC);
            let shifted = self.builder.ins().fadd(key_value, magic);
            let restored = self.builder.ins().fsub(shifted, magic);
            let exact = self.builder.ins().fcmp(FloatCC::Equal, restored, key_value);
            branch_unless(self, exact);
            let bits = self
                .builder
                .ins()
                .bitcast(types::I64, MemFlags::new(), shifted);
            let magic_bits = FLOAT_INTEGER_MAGIC.to_bits().cast_signed();
            self.builder.ins().iadd_imm(bits, -(magic_bits + 1))
        } else {
            let index = self.builder.ins().iadd_imm(key_value, -1);
            let in_range = self
                .builder
                .ins()
                .icmp(IntCC::UnsignedLessThan, index, length);
            branch_unless(self, in_range);
            index
        }
    }

    /// The array words of the table in lane `table`, a table met by lane
    /// ([`TraceOp::LaneArrayLoad`]): the address of slot 1, the slot count
    /// and the count as a float, read from the table itself, whose address
    /// is the word of field `field`
    /// ([`crate::jit_ir::trace::TABLE_ADDRESS_KEY`]) of the lane's
    /// row. A row the VM could not fill has the word zero, and then the code
    /// branches to `fail`, the access's exit.
    ///
    /// The words are kept per lane in the straight line of code, like the
    /// row addresses, and dropped wherever the words of array bindings are
    /// (see [`Self::forget_array_words`]): a set may have moved the array.
    /// The second result says whether they came from that cache; the caller
    /// caches new words only once the access's exit is emitted (see
    /// [`Self::lane_array_load`]).
    fn lane_array_words(
        &mut self,
        table: ValueId,
        field: u32,
        fail: Block,
    ) -> Result<((Value, Value, Value), bool), TraceCompileError> {
        let lane = self.value(table);
        if self.function.is_none()
            && let Some((_, words)) = self.lane_array_cache.iter().find(|(held, _)| *held == lane)
        {
            return Ok((*words, true));
        }
        let layout = NativeTableLayout::get().ok_or(TraceCompileError::NativeBackendRejected)?;
        let row = self.row_address(lane);
        let address = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            row,
            offset(field as usize * mem::size_of::<usize>()),
        );
        let pass = self.builder.create_block();
        self.builder.ins().brif(address, pass, &[], fail, &[]);
        self.builder.switch_to_block(pass);
        let base =
            self.builder
                .ins()
                .load(types::I64, MemFlags::new(), address, layout.array_pointer);
        let length =
            self.builder
                .ins()
                .load(types::I64, MemFlags::new(), address, layout.array_length);
        let length_float = self.builder.ins().fcvt_from_sint(types::F64, length);
        Ok(((base, length, length_float), false))
    }

    /// Whether the tree's rows hold table addresses, the word lane array
    /// views read ([`crate::jit_ir::trace::TABLE_ADDRESS_KEY`]).
    fn reads_table_addresses(&self) -> bool {
        self.function.is_none()
            && self.members[0]
                .ir
                .fields
                .iter()
                .any(|field| &*field.key == crate::jit_ir::trace::TABLE_ADDRESS_KEY)
    }

    /// A [`TraceOp::LaneArrayLoad`]: the array access of
    /// [`Self::array_access`] on the words [`Self::lane_array_words`] reads.
    /// With `stored` it is a [`TraceOp::LaneArrayStore`], which overwrites
    /// the payload of the slot once its type checked out, as an
    /// [`Self::array_access`] store does: the tag, the border and every GC
    /// edge stay, and so do the words, which no payload store moves.
    fn lane_array_load(
        &mut self,
        (table, field): (ValueId, u32),
        key: ValueId,
        ty: TraceType,
        stored: Option<Value>,
        guard: GuardId,
        fuel: Value,
    ) -> Result<Value, TraceCompileError> {
        let fail = self.builder.create_block();
        self.builder.set_cold_block(fail);
        if self.forced_guard == Some((self.member, guard.index())) {
            let pass = self.builder.create_block();
            let never = self.builder.ins().iconst(types::I8, 0);
            self.builder.ins().brif(never, pass, &[], fail, &[]);
            self.builder.switch_to_block(pass);
        }
        let (words, was_cached) = self.lane_array_words(table, field, fail)?;
        let (base, length, length_float) = words;
        let index = self.index_within((length, Some(length_float)), key, Some(guard), fail);
        debug_assert_eq!(NATIVE_VALUE_SIZE, 16);
        let scaled = self.builder.ins().ishl_imm(index, 4);
        let slot = self.builder.ins().iadd(base, scaled);
        let value = self.slot_access(slot, fail, ty, stored, guard, fuel)?;
        // The words are defined past the check of the row's address word,
        // which `fail` does not follow: a side trace that `fail` runs inline
        // must not find them in the cache. They are cached here, on the path
        // past the access, which they dominate.
        if !was_cached && self.function.is_none() {
            let lane = self.value(table);
            self.lane_array_cache.push((lane, words));
        }
        Ok(value)
    }

    /// The address of slot 1, the slot count, and the count as a float (for
    /// an array or bytes binding) of binding `binding`. A tree that sets
    /// tables reads an array's words from the run's binding slots at every
    /// access, since a set may have moved the array; otherwise they are the
    /// words the entry loaded.
    fn array_words(&mut self, binding: u32) -> (Value, Value, Option<Value>) {
        let index = binding as usize;
        if !((self.reload_arrays || self.fresh_words)
            && self.members[0].ir.bindings[index].is_array())
        {
            let (base, length) = self.bound[index];
            return (base, length, self.bound_length_float[index]);
        }
        if let Some((base, length, length_float)) = self.array_cache[index] {
            return (base, length, Some(length_float));
        }
        let words = self.words.expect("entry located the binding slots");
        let at = offset(index * mem::size_of::<TraceInputSlot>());
        let base = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            words,
            at + offset(mem::offset_of!(TraceInputSlot, bits)),
        );
        let length = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            words,
            at + offset(mem::offset_of!(TraceInputSlot, tag)),
        );
        let length_float = self.builder.ins().fcvt_from_sint(types::F64, length);
        self.array_cache[index] = Some((base, length, length_float));
        (base, length, Some(length_float))
    }

    /// Loads the words of the arrays the root's loop reads, here in the
    /// header and once for the run of the loop, and has the loop and the
    /// sides read those loads.
    ///
    /// The values the entry loaded are needed after the loop too (by the
    /// next run of it, past the calls the sides make), so they are live
    /// across calls, and the allocator, with few registers that survive a
    /// call, gives them a stack slot that the loop then reads in every
    /// iteration. The loads here end with the loop.
    fn load_fresh_words(&mut self) {
        if self.reload_arrays || self.function.is_some() {
            return;
        }
        self.fresh_words = true;
        self.forget_array_words();
        let mut used = vec![false; self.ir.bindings.len()];
        for instruction in &self.ir.body {
            if let TraceOp::ArrayLoad { binding, .. }
            | TraceOp::NumberLoad { binding, .. }
            | TraceOp::ArrayNext { binding, .. }
            | TraceOp::ArrayStore { binding, .. }
            | TraceOp::ArrayOverwrite { binding, .. } = &instruction.op
            {
                used[*binding as usize] = true;
            }
        }
        for (binding, used) in used.into_iter().enumerate() {
            if used && self.ir.bindings[binding].is_array() {
                self.array_words(u32::try_from(binding).expect("few bindings"));
            }
        }
    }

    /// Forgets the array words loaded so far: a set may have moved an array,
    /// or the code continues where earlier loads do not dominate it.
    fn forget_array_words(&mut self) {
        self.array_cache.fill(None);
        // A lane's array words are the same words, read through the table.
        self.lane_array_cache.clear();
        // A slot's address comes from the words.
        self.slots.clear();
        // So does a row: it adds an index to a base, which a block the
        // earlier load does not dominate must not reuse. Proven accesses are
        // lowered on the hot path only, never in an exit or fail block.
        if let Some(fast) = self.fast.as_mut() {
            fast.rows.clear();
        }
    }

    /// An access to the array part of a bound table. Leaves through the
    /// guard's exit unless `key` is an integral number naming a slot inside
    /// the array that holds a non-nil value whose tag matches `ty`. A load
    /// returns the payload. A store (`stored` is the value) overwrites the
    /// payload only: the tag, the border and every GC edge stay as they are,
    /// which is why the slot's type is checked first.
    fn array_access(
        &mut self,
        binding: u32,
        key: ValueId,
        ty: TraceType,
        stored: Option<Value>,
        guard: GuardId,
        fuel: Value,
    ) -> Result<Value, TraceCompileError> {
        let forced = self.forced_guard == Some((self.member, guard.index()));
        // An access to the slot an earlier one reached needs no bounds check,
        // and none of the slot's type when that is the type the earlier one
        // found (a store keeps it): a read then a write of `t[i]` check once.
        if !forced
            && let Some(fact) = self
                .slots
                .iter()
                .find(|fact| fact.binding == binding && fact.key == key)
                .copied()
        {
            if fact.ty == ty
                && matches!(
                    ty,
                    TraceType::Integer | TraceType::Float | TraceType::Boolean
                )
            {
                return Ok(match stored {
                    Some(value) => {
                        self.builder.ins().store(
                            MemFlags::new(),
                            value,
                            fact.slot,
                            NATIVE_VALUE_PAYLOAD_OFFSET,
                        );
                        value
                    }
                    None => self.builder.ins().load(
                        clif_type(ty),
                        MemFlags::new(),
                        fact.slot,
                        NATIVE_VALUE_PAYLOAD_OFFSET,
                    ),
                });
            }
            let fail = self.builder.create_block();
            self.builder.set_cold_block(fail);
            let result = self.slot_access(fact.slot, fail, ty, stored, guard, fuel)?;
            self.note_slot(binding, key, fact.slot, ty);
            return Ok(result);
        }
        debug_assert_eq!(NATIVE_VALUE_SIZE, 16);
        let proven = if forced {
            None
        } else {
            self.proven_slot(binding, key)
        };
        let (slot, fail) = if let Some((row, disp)) = proven {
            let fail = self.builder.create_block();
            self.builder.set_cold_block(fail);
            (self.builder.ins().iadd_imm(row, i64::from(disp)), fail)
        } else {
            let (index, fail, base) = self.bound_index(binding, key, guard);
            let scaled = self.builder.ins().ishl_imm(index, 4);
            (self.builder.ins().iadd(base, scaled), fail)
        };
        let result = self.slot_access(slot, fail, ty, stored, guard, fuel)?;
        self.note_slot(binding, key, slot, ty);
        Ok(result)
    }

    /// [`TraceOp::ArrayOverwrite`]: overwrites the payload of the slot of
    /// `key` on `binding`, which the verifier proved is inside the array and
    /// holds a value of type `ty` (an earlier access in this iteration
    /// reached it, and nothing since may move or retype a slot). A slot the
    /// code before reached is stored at its address. Otherwise the index and
    /// the tag are checked as an [`Self::array_access`] checks them, and a
    /// failed check, which the proof rules out, traps instead of storing
    /// anywhere.
    fn array_overwrite(&mut self, binding: u32, key: ValueId, ty: TraceType, stored: Value) {
        if let Some(fact) = self
            .slots
            .iter()
            .find(|fact| fact.binding == binding && fact.key == key && fact.ty == ty)
            .copied()
        {
            self.builder.ins().store(
                MemFlags::new(),
                stored,
                fact.slot,
                NATIVE_VALUE_PAYLOAD_OFFSET,
            );
            return;
        }
        let (base, length, length_float) = self.array_words(binding);
        let fail = self.builder.create_block();
        self.builder.set_cold_block(fail);
        let index = self.index_within((length, length_float), key, None, fail);
        let scaled = self.builder.ins().ishl_imm(index, 4);
        let slot = self.builder.ins().iadd(base, scaled);
        let tag = self.builder.ins().load(types::I8, MemFlags::new(), slot, 0);
        let wanted = match ty {
            TraceType::Float => NATIVE_VALUE_TAG_NUMBER,
            TraceType::Integer => NATIVE_VALUE_TAG_INTEGER,
            _ => NATIVE_VALUE_TAG_BOOLEAN,
        };
        let matches = self
            .builder
            .ins()
            .icmp_imm(IntCC::Equal, tag, i64::from(wanted));
        let pass = self.builder.create_block();
        self.builder.ins().brif(matches, pass, &[], fail, &[]);
        self.builder.switch_to_block(fail);
        self.builder
            .ins()
            .trap(cranelift_codegen::ir::TrapCode::unwrap_user(1));
        self.builder.switch_to_block(pass);
        self.builder
            .ins()
            .store(MemFlags::new(), stored, slot, NATIVE_VALUE_PAYLOAD_OFFSET);
        self.note_slot(binding, key, slot, ty);
    }

    /// [`TraceOp::NumberModulo`]: the integer floor modulo of two integers
    /// held as floats when `integer`, the float modulo `float` otherwise.
    fn number_modulo(
        &mut self,
        float: ArithmeticOp,
        left: Value,
        right: Value,
        integer: Value,
    ) -> Result<Value, TraceCompileError> {
        let whole = self.builder.create_block();
        let fractional = self.builder.create_block();
        let merge = self.builder.create_block();
        self.builder.append_block_param(merge, types::F64);
        self.builder
            .ins()
            .brif(integer, whole, &[], fractional, &[]);
        self.builder.switch_to_block(whole);
        // Both are integers within 2^53 in magnitude, so the conversions are
        // exact and the saturation never applies.
        let left_integer = self.builder.ins().fcvt_to_sint_sat(types::I64, left);
        let right_integer = self.builder.ins().fcvt_to_sint_sat(types::I64, right);
        // The divisor is not zero (the trace checked), and the dividend is
        // far from `MIN`, so `srem` needs neither of its special cases. The
        // floor form adds the divisor to a remainder of the other sign.
        let rem = self.builder.ins().srem(left_integer, right_integer);
        let signs = self.builder.ins().bxor(rem, right_integer);
        let differs = self.builder.ins().icmp_imm(IntCC::SignedLessThan, signs, 0);
        let nonzero = self.builder.ins().icmp_imm(IntCC::NotEqual, rem, 0);
        let adjust = self.builder.ins().band(differs, nonzero);
        let shifted = self.builder.ins().iadd(rem, right_integer);
        let remainder = self.builder.ins().select(adjust, shifted, rem);
        let as_float = self.builder.ins().fcvt_from_sint(types::F64, remainder);
        self.builder.ins().jump(merge, &[BlockArg::Value(as_float)]);
        self.builder.switch_to_block(fractional);
        let by_float = self.float_arithmetic(float, left, right);
        self.builder.ins().jump(merge, &[BlockArg::Value(by_float)]);
        self.builder.switch_to_block(merge);
        Ok(self.builder.block_params(merge)[0])
    }

    /// The element `key` of the array part of the bound table, when it holds
    /// an integer of magnitude at most 2^53 or a float ([`TraceOp::NumberLoad`]).
    /// Returns the number as a float and whether the slot held an integer.
    /// Leaves through the guard's exit for any other slot.
    fn number_load(
        &mut self,
        binding: u32,
        key: ValueId,
        guard: GuardId,
        fuel: Value,
    ) -> Result<(Value, Value), TraceCompileError> {
        let forced = self.forced_guard == Some((self.member, guard.index()));
        debug_assert_eq!(NATIVE_VALUE_SIZE, 16);
        let proven = if forced {
            None
        } else {
            self.proven_slot(binding, key)
        };
        let (slot, fail) = if let Some((row, disp)) = proven {
            let fail = self.builder.create_block();
            self.builder.set_cold_block(fail);
            (self.builder.ins().iadd_imm(row, i64::from(disp)), fail)
        } else {
            let (index, fail, base) = self.bound_index(binding, key, guard);
            let scaled = self.builder.ins().ishl_imm(index, 4);
            (self.builder.ins().iadd(base, scaled), fail)
        };
        let tag = self.builder.ins().load(types::I8, MemFlags::new(), slot, 0);
        let integer = self.builder.create_block();
        let other = self.builder.create_block();
        let float = self.builder.create_block();
        let merge = self.builder.create_block();
        self.builder.append_block_param(merge, types::F64);
        self.builder.append_block_param(merge, types::I8);
        let is_integer =
            self.builder
                .ins()
                .icmp_imm(IntCC::Equal, tag, i64::from(NATIVE_VALUE_TAG_INTEGER));
        self.builder
            .ins()
            .brif(is_integer, integer, &[], other, &[]);
        self.builder.switch_to_block(other);
        let is_float =
            self.builder
                .ins()
                .icmp_imm(IntCC::Equal, tag, i64::from(NATIVE_VALUE_TAG_NUMBER));
        self.builder.ins().brif(is_float, float, &[], fail, &[]);
        self.builder.switch_to_block(float);
        let held = self.builder.ins().load(
            types::F64,
            MemFlags::new(),
            slot,
            NATIVE_VALUE_PAYLOAD_OFFSET,
        );
        let no = self.builder.ins().iconst(types::I8, 0);
        self.builder
            .ins()
            .jump(merge, &[BlockArg::Value(held), BlockArg::Value(no)]);
        // An integer is exact as a float within 2^53 in magnitude.
        self.builder.switch_to_block(integer);
        let bits = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            slot,
            NATIVE_VALUE_PAYLOAD_OFFSET,
        );
        let shifted = self.builder.ins().iadd_imm(bits, 1_i64 << 53);
        let in_range =
            self.builder
                .ins()
                .icmp_imm(IntCC::UnsignedLessThanOrEqual, shifted, 1_i64 << 54);
        let exact = self.builder.create_block();
        self.builder.ins().brif(in_range, exact, &[], fail, &[]);
        self.builder.switch_to_block(exact);
        let converted = self.builder.ins().fcvt_from_sint(types::F64, bits);
        let yes = self.builder.ins().iconst(types::I8, 1);
        self.builder
            .ins()
            .jump(merge, &[BlockArg::Value(converted), BlockArg::Value(yes)]);
        // The slot is neither: leave through the guard's exit, which the
        // caller made (cold) and the branches above reach.
        self.builder.switch_to_block(fail);
        self.leave(guard, fuel)?;
        self.builder.switch_to_block(merge);
        let results = self.builder.block_params(merge);
        Ok((results[0], results[1]))
    }

    /// Notes that the access just lowered reached the slot of `key` on
    /// `binding` at `slot`, which holds a value of type `ty`.
    fn note_slot(&mut self, binding: u32, key: ValueId, slot: Value, ty: TraceType) {
        self.slots
            .retain(|fact| !(fact.binding == binding && fact.key == key));
        self.slots.push(SlotFact {
            binding,
            key,
            slot,
            ty,
        });
    }

    /// The nil read of [`TraceOp::ArrayLoad`] with `past_end`: nil when `key`
    /// names a slot inside the array that holds nil, and nil for any other key
    /// when the table's header says it is solid, since its hash part is empty
    /// and holds no such key. Leaves through the guard's exit otherwise. The
    /// verifier made sure the table has no metatable.
    fn array_nil_read(
        &mut self,
        binding: u32,
        key: ValueId,
        guard: GuardId,
        fuel: Value,
    ) -> Result<Value, TraceCompileError> {
        let (index, outside, base) = self.bound_index(binding, key, guard);
        let read = self.builder.create_block();
        let refused = self.builder.create_block();
        self.builder.set_cold_block(refused);
        debug_assert_eq!(NATIVE_VALUE_SIZE, 16);
        let scaled = self.builder.ins().ishl_imm(index, 4);
        let slot = self.builder.ins().iadd(base, scaled);
        let tag = self.builder.ins().load(types::I8, MemFlags::new(), slot, 0);
        let hole = self
            .builder
            .ins()
            .icmp_imm(IntCC::Equal, tag, i64::from(NATIVE_VALUE_TAG_NIL));
        self.builder.ins().brif(hole, read, &[], refused, &[]);
        self.builder.switch_to_block(outside);
        let header = self.header_binding(binding);
        match header {
            Some((header, _)) if self.forced_guard != Some((self.member, guard.index())) => {
                let words = self.words.expect("entry located the binding slots");
                let flags = self.builder.ins().load(
                    types::I64,
                    MemFlags::new(),
                    words,
                    offset(
                        header * mem::size_of::<TraceInputSlot>()
                            + mem::offset_of!(TraceInputSlot, tag),
                    ),
                );
                // Any nonzero word: the table is solid.
                self.builder.ins().brif(flags, read, &[], refused, &[]);
            }
            _ => {
                self.builder.ins().jump(refused, &[]);
            }
        }
        self.builder.switch_to_block(refused);
        self.leave(guard, fuel)?;
        self.builder.switch_to_block(read);
        Ok(self.builder.ins().iconst(types::I8, 0))
    }

    /// The element of the slot at `slot`, which holds a value: the read of
    /// an array step, which found the slot inside the array.
    fn slot_read(
        &mut self,
        (slot, tag): (Value, Value),
        ty: TraceType,
        guard: GuardId,
        fuel: Value,
    ) -> Result<Value, TraceCompileError> {
        let fail = self.builder.create_block();
        self.builder.set_cold_block(fail);
        self.slot_checked(slot, tag, fail, ty, None, guard, fuel)
    }

    /// The access of the slot at `slot` of type `ty`: the check of the slot's
    /// tag, which leaves through `fail` (the caller made it, cold) when it is
    /// another type, and then the load or store of the payload.
    fn slot_access(
        &mut self,
        slot: Value,
        fail: Block,
        ty: TraceType,
        stored: Option<Value>,
        guard: GuardId,
        fuel: Value,
    ) -> Result<Value, TraceCompileError> {
        let tag = self.builder.ins().load(types::I8, MemFlags::new(), slot, 0);
        self.slot_checked(slot, tag, fail, ty, stored, guard, fuel)
    }

    /// [`Self::slot_access`] for a slot whose `tag` the caller loaded.
    #[allow(clippy::too_many_arguments)]
    fn slot_checked(
        &mut self,
        slot: Value,
        tag: Value,
        fail: Block,
        ty: TraceType,
        stored: Option<Value>,
        guard: GuardId,
        fuel: Value,
    ) -> Result<Value, TraceCompileError> {
        let wanted = match ty {
            TraceType::Float => NATIVE_VALUE_TAG_NUMBER,
            TraceType::Integer => NATIVE_VALUE_TAG_INTEGER,
            TraceType::Table => NATIVE_VALUE_TAG_TABLE,
            TraceType::Closure => NATIVE_VALUE_TAG_CLOSURE,
            TraceType::Thread => NATIVE_VALUE_TAG_THREAD,
            TraceType::StringRef => NATIVE_VALUE_TAG_STRING,
            TraceType::Nil => NATIVE_VALUE_TAG_NIL,
            _ => NATIVE_VALUE_TAG_BOOLEAN,
        };
        let matches = self
            .builder
            .ins()
            .icmp_imm(IntCC::Equal, tag, i64::from(wanted));
        let pass = self.builder.create_block();
        self.builder.ins().brif(matches, pass, &[], fail, &[]);
        self.builder.switch_to_block(fail);
        self.leave(guard, fuel)?;
        self.builder.switch_to_block(pass);
        if matches!(
            ty,
            TraceType::Table | TraceType::Closure | TraceType::Thread
        ) {
            // The load's result is the table's (or closure's, or thread's) lane, which
            // only the VM can give: it adds the value to the run's bank.
            // Without a lane (the run has no room) the interpreter does the
            // access.
            return Ok(self.admit_table_lane(slot, guard, fuel, ty == TraceType::Table));
        }
        if ty == TraceType::StringRef {
            // A borrow: the address of the slot, which holds the string. No
            // refcount, no lane. The owner table is a bound entry input and
            // the verifier keeps every use before the next operation that
            // could move or replace the slot.
            return Ok(slot);
        }
        if ty == TraceType::Nil {
            // A hole: the verifier made sure no `__index` can run.
            return Ok(self.builder.ins().iconst(types::I8, 0));
        }
        Ok(match stored {
            Some(value) => {
                self.builder
                    .ins()
                    .store(MemFlags::new(), value, slot, NATIVE_VALUE_PAYLOAD_OFFSET);
                value
            }
            None => self.builder.ins().load(
                clif_type(ty),
                MemFlags::new(),
                slot,
                NATIVE_VALUE_PAYLOAD_OFFSET,
            ),
        })
    }

    /// The step along the array part of the table bound as `binding`
    /// ([`TraceOp::ArrayNext`]): the one-based index of the first slot with a
    /// value after the slot `key` names (the first slot of the array when
    /// `start`), found by a loop over the nil slots. The guard leaves when
    /// `key` is no slot that holds a value now, or when no slot with a value
    /// follows.
    fn array_next(
        &mut self,
        binding: u32,
        key: ValueId,
        (start, ty): (bool, TraceType),
        guard: GuardId,
        fuel: Value,
    ) -> Result<(Value, (Value, Value)), TraceCompileError> {
        debug_assert_eq!(NATIVE_VALUE_SIZE, 16);
        let (from, fail, base, before) = if start {
            let (base, _, _) = self.array_words(binding);
            let fail = self.builder.create_block();
            self.builder.set_cold_block(fail);
            if self.forced_guard == Some((self.member, guard.index())) {
                let never = self.builder.ins().iconst(types::I8, 0);
                let next = self.builder.create_block();
                self.builder.ins().brif(never, next, &[], fail, &[]);
                self.builder.switch_to_block(next);
            }
            (self.builder.ins().iconst(types::I64, 0), fail, base, None)
        } else if self.member == 0
            && self
                .trusted_key
                .is_some_and(|trusted| trusted.key == key && trusted.binding == binding)
        {
            // The step before found the slot the key names, and nothing
            // clears a slot (see `trusted_array_key`).
            let (base, _, _) = self.array_words(binding);
            let fail = self.builder.create_block();
            self.builder.set_cold_block(fail);
            if self.forced_guard == Some((self.member, guard.index())) {
                let never = self.builder.ins().iconst(types::I8, 0);
                let next = self.builder.create_block();
                self.builder.ins().brif(never, next, &[], fail, &[]);
                self.builder.switch_to_block(next);
            }
            let index = self.trusted_index(key);
            let scaled = self.builder.ins().ishl_imm(index, 4);
            let slot = self.builder.ins().iadd(base, scaled);
            (
                self.builder.ins().iadd_imm(index, 1),
                fail,
                base,
                Some(slot),
            )
        } else {
            // The key is a slot inside the array, and the slot holds a value.
            let (index, fail, base) = self.bound_index(binding, key, guard);
            let scaled = self.builder.ins().ishl_imm(index, 4);
            let slot = self.builder.ins().iadd(base, scaled);
            let tag = self.builder.ins().load(types::I8, MemFlags::new(), slot, 0);
            let empty =
                self.builder
                    .ins()
                    .icmp_imm(IntCC::Equal, tag, i64::from(NATIVE_VALUE_TAG_NIL));
            let held = self.builder.create_block();
            self.builder.ins().brif(empty, fail, &[], held, &[]);
            self.builder.switch_to_block(held);
            (
                self.builder.ins().iadd_imm(index, 1),
                fail,
                base,
                Some(slot),
            )
        };
        let (_, length, _) = self.array_words(binding);
        let found = self.scan_array((from, before), (base, length), fail);
        self.builder.switch_to_block(fail);
        self.leave(guard, fuel)?;
        self.builder.switch_to_block(found);
        let (index, slot, tag) = {
            let params = self.builder.block_params(found);
            (params[0], params[1], params[2])
        };
        let number = self.builder.ins().iadd_imm(index, 1);
        let number = if ty == TraceType::Float {
            self.builder.ins().fcvt_from_sint(types::F64, number)
        } else {
            number
        };
        Ok((number, (slot, tag)))
    }

    /// The failing edge of `guard`, in the current block: the side trace
    /// attached to the guard, in place (no exit, no frame, no return to the
    /// VM), or the guard's exit.
    fn leave(&mut self, guard: GuardId, fuel: Value) -> Result<(), TraceCompileError> {
        let side = self.members[self.member]
            .attached
            .iter()
            .find(|(attached, _)| *attached == guard.index())
            .map(|(_, side)| *side);
        if let Some(side) = side {
            let origins = if self.member == 0 {
                self.side_origins_of(side, guard.index())
            } else {
                Vec::new()
            };
            let parent_origins = mem::replace(&mut self.side_origins, origins);
            let reloading = mem::replace(&mut self.reload_inputs, true);
            let inputs = self.side_inputs(side, guard.index());
            self.reload_inputs = reloading;
            let (inputs, subtypes) = inputs?;
            if let Some(holds) = subtypes {
                // A number of the other subtype than the side was recorded
                // for leaves through the guard's own exit.
                let enter = self.builder.create_block();
                let other = self.builder.create_block();
                self.builder.set_cold_block(other);
                self.builder.ins().brif(holds, enter, &[], other, &[]);
                self.builder.switch_to_block(other);
                let exit = self.plan.members[self.member].first + guard.index();
                self.emit_exit(exit, fuel);
                self.builder.switch_to_block(enter);
            }
            if self.count_links {
                self.count_link();
            }
            let lowered = self.lower_side(side, &inputs, fuel);
            self.side_origins = parent_origins;
            lowered?;
        } else {
            let exit = self.plan.members[self.member].first + guard.index();
            self.emit_exit(exit, fuel);
        }
        Ok(())
    }

    /// A raw set of `key` in the table bound as `binding`
    /// ([`TraceOp::ArraySet`]): an in-place overwrite of a slot inside the
    /// array that holds a value of the stored type, and otherwise a call to
    /// the VM, which leaves through the guard's exit when it cannot store.
    fn array_set(
        &mut self,
        binding: u32,
        key: ValueId,
        value: ValueId,
        guard: GuardId,
        fuel: Value,
    ) -> Result<(), TraceCompileError> {
        let ty = self
            .ir
            .value_type(value)
            .expect("a verified trace types every operand");
        let stored = self.value(value);
        let done = self.builder.create_block();
        if self.set_through_header(binding, key, (ty, stored), done, (guard, fuel))? {
            return Ok(());
        }
        let through_vm = if matches!(ty, TraceType::Nil | TraceType::String | TraceType::Table) {
            // Storing nil changes the border or deletes a key, and a string
            // is a counted reference no payload overwrite can take over: the
            // VM's job.
            let through_vm = self.builder.create_block();
            self.builder.ins().jump(through_vm, &[]);
            through_vm
        } else {
            let (index, through_vm, base) = self.bound_index(binding, key, guard);
            let scaled = self.builder.ins().ishl_imm(index, 4);
            let slot = self.builder.ins().iadd(base, scaled);
            let tag = self.builder.ins().load(types::I8, MemFlags::new(), slot, 0);
            let wanted = match ty {
                TraceType::Float => NATIVE_VALUE_TAG_NUMBER,
                TraceType::Integer => NATIVE_VALUE_TAG_INTEGER,
                _ => NATIVE_VALUE_TAG_BOOLEAN,
            };
            let matches = self
                .builder
                .ins()
                .icmp_imm(IntCC::Equal, tag, i64::from(wanted));
            let overwrite = self.builder.create_block();
            self.builder
                .ins()
                .brif(matches, overwrite, &[], through_vm, &[]);
            self.builder.switch_to_block(overwrite);
            self.builder
                .ins()
                .store(MemFlags::new(), stored, slot, NATIVE_VALUE_PAYLOAD_OFFSET);
            self.builder.ins().jump(done, &[]);
            through_vm
        };
        self.builder.set_cold_block(through_vm);
        self.builder.switch_to_block(through_vm);
        let exit = self.plan.members[self.member].first + guard.index();
        if self.forced_guard == Some((self.member, guard.index())) {
            self.emit_exit(exit, fuel);
            self.builder.switch_to_block(done);
            self.forget_array_words();
            return Ok(());
        }
        let control = self.control.expect("entry set the control block");
        let words = self.words.expect("entry located the binding slots");
        let binding_value = self.builder.ins().iconst(types::I64, i64::from(binding));
        let key_ty = self
            .ir
            .value_type(key)
            .expect("a verified trace types every operand");
        let key_value = self.value(key);
        let (key_tag, key_bits) = self.scalar_word(key_ty, key_value);
        let (value_tag, value_bits) = self.scalar_word(ty, stored);
        let status = if self.direct_set {
            // A tree that calls the VM for strings has no loop whose
            // registers a call would spill: the set goes straight to the
            // VM's entry, without the stub that saves every register.
            let entry = self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                control,
                offset(mem::offset_of!(TraceControl, set_entry)),
            );
            let context = self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                control,
                offset(mem::offset_of!(TraceControl, host_context)),
            );
            let mut signature = Signature::new(self.builder.func.signature.call_conv);
            for _ in 0..7 {
                signature.params.push(AbiParam::new(types::I64));
            }
            signature.returns.push(AbiParam::new(types::I64));
            let signature = self.builder.import_signature(signature);
            let call = self.builder.ins().call_indirect(
                signature,
                entry,
                &[
                    context,
                    words,
                    binding_value,
                    key_tag,
                    key_bits,
                    value_tag,
                    value_bits,
                ],
            );
            self.builder.inst_results(call)[0]
        } else {
            self.builder.ins().call(
                self.table_set
                    .expect("a tree that sets tables has the stub"),
                &[
                    control,
                    words,
                    binding_value,
                    key_tag,
                    key_bits,
                    value_tag,
                    value_bits,
                ],
            );
            self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                control,
                offset(mem::offset_of!(TraceControl, set_status)),
            )
        };
        let refused = self.builder.create_block();
        self.builder.set_cold_block(refused);
        self.builder.ins().brif(status, refused, &[], done, &[]);
        self.builder.switch_to_block(refused);
        self.emit_exit(exit, fuel);
        self.builder.switch_to_block(done);
        // The call may have moved any array of the tree.
        self.forget_array_words();
        Ok(())
    }

    /// The part of a [`TraceOp::ArraySet`] a table with a header does in
    /// place, before the general path: see [`Self::set_in_place`]. Returns
    /// whether the set is complete, which it is for a strict header: a set
    /// that did not store in place leaves through its guard. The code goes
    /// on in the block the set continues in, or in `done` when it is
    /// complete.
    fn set_through_header(
        &mut self,
        binding: u32,
        key: ValueId,
        (ty, stored): (TraceType, Value),
        done: Block,
        (guard, fuel): (GuardId, Value),
    ) -> Result<bool, TraceCompileError> {
        let Some((header, strict)) = self.header_binding(binding) else {
            return Ok(false);
        };
        let Some(layout) = NativeTableLayout::get() else {
            return Ok(false);
        };
        // An appendable table takes an append, a pop and an overwrite in
        // place; anything else, or a table that is not appendable (the VM
        // rewrites the answer), continues. A strict header has no VM to
        // continue with: the guard leaves.
        let continues = self.builder.create_block();
        if self.forced_guard == Some((self.member, guard.index())) {
            self.builder.ins().jump(continues, &[]);
        } else {
            self.set_in_place((header, layout), key, (ty, stored), (continues, done));
        }
        self.builder.switch_to_block(continues);
        if !strict {
            return Ok(false);
        }
        self.leave(guard, fuel)?;
        self.builder.switch_to_block(done);
        self.forget_array_words();
        Ok(true)
    }

    /// The binding that holds the header of the table array binding `array`
    /// is bound to, if the trace has one.
    fn header_binding(&self, array: u32) -> Option<(usize, bool)> {
        header_binding_of(&self.members[0].ir.bindings, array)
    }

    /// The array and length words of the bindings of the table of binding
    /// `header` in the run's binding slots, set to a table of `length` slots
    /// at `pointer` (`None` leaves the address): an append, a pop or a growth
    /// did it in place, so the VM has not rewritten them. The length bindings
    /// follow only if the border moved with the length (`moves_border`): a
    /// growth with holes leaves it.
    fn store_array_words(
        &mut self,
        header: usize,
        (length, moves_border): (Value, bool),
        pointer: Option<Value>,
    ) {
        let words = self.words.expect("entry located the binding slots");
        let bindings = &self.members[0].ir.bindings;
        let register = bindings[header].register;
        let mine: Vec<(usize, BindingKind)> = bindings
            .iter()
            .enumerate()
            .filter(|(_, other)| other.register == register)
            .map(|(index, other)| (index, other.kind.clone()))
            .collect();
        let slot = mem::size_of::<TraceInputSlot>();
        for (index, kind) in mine {
            let at = offset(index * slot);
            let tag_at = at + offset(mem::offset_of!(TraceInputSlot, tag));
            let bits_at = at + offset(mem::offset_of!(TraceInputSlot, bits));
            match kind {
                BindingKind::Array { .. } => {
                    self.builder
                        .ins()
                        .store(MemFlags::new(), length, words, tag_at);
                    if let Some(pointer) = pointer {
                        self.builder
                            .ins()
                            .store(MemFlags::new(), pointer, words, bits_at);
                    }
                }
                BindingKind::Length { ty, .. } if moves_border => {
                    let border = if ty == TraceType::Float {
                        self.builder.ins().fcvt_from_sint(types::F64, length)
                    } else {
                        length
                    };
                    self.builder
                        .ins()
                        .store(MemFlags::new(), border, words, bits_at);
                }
                _ => {}
            }
        }
    }

    /// The part of a [`TraceOp::ArraySet`] native code does itself in a table
    /// whose header says it is appendable (a dense list, see
    /// `oximond::runtime::NativeTableHeader`): the key one past the last slot
    /// appends the value, the last slot with a nil pops it, and a slot inside
    /// the array takes a value in place. A solid table that is not a dense
    /// list (it has holes) takes the in-place store of a scalar too, except
    /// into the slot after the border, and a boolean a few slots past the
    /// end ([`Self::grow_with_holes`]). Jumps to `done` having stored, and
    /// to `continues` having stored nothing.
    ///
    /// A scalar goes straight into the slot. A string goes through the VM's
    /// store entry, which owns the reference count; the code does the rest.
    /// A slot that holds a value that needs releasing is never overwritten
    /// here, so a table that holds strings keeps the general path for those.
    /// A pop releases it through the same entry, with a nil.
    #[allow(clippy::too_many_lines)]
    fn set_in_place(
        &mut self,
        (header, layout): (usize, NativeTableLayout),
        key: ValueId,
        (ty, stored): (TraceType, Value),
        (continues, done): (Block, Block),
    ) {
        let reference = matches!(ty, TraceType::String | TraceType::Table);
        let words = self.words.expect("entry located the binding slots");
        let header_at = offset(header * mem::size_of::<TraceInputSlot>());
        let flags = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            words,
            header_at + offset(mem::offset_of!(TraceInputSlot, tag)),
        );
        let appendable = self.builder.create_block();
        let holey = self.builder.create_block();
        let is_dense = self
            .builder
            .ins()
            .icmp_imm(IntCC::Equal, flags, NativeTableHeader::DENSE);
        self.builder
            .ins()
            .brif(is_dense, appendable, &[], holey, &[]);
        self.builder.switch_to_block(appendable);
        let table = self.bound[header].0;
        let index = self.integral_key(key, continues);
        let field = |this: &mut Self, at: i32| {
            this.builder
                .ins()
                .load(types::I64, MemFlags::new(), table, at)
        };
        let length = field(self, layout.array_length);
        let scalar = matches!(
            ty,
            TraceType::Integer | TraceType::Float | TraceType::Boolean
        );
        let last = self.builder.ins().iadd_imm(length, 1);
        let appends = self.builder.ins().icmp(IntCC::Equal, index, last);
        let append = self.builder.create_block();
        let other = self.builder.create_block();
        self.builder.ins().brif(appends, append, &[], other, &[]);

        // Append: the slot after the last, with room for it.
        self.builder.switch_to_block(append);
        if ty == TraceType::Nil {
            // A nil one past the end of a dense list stores nothing: the key
            // is absent from the array and, as for an append, from the hash
            // part. The set is complete, so a strict header stays in the
            // trace.
            self.builder.ins().jump(done, &[]);
        } else {
            let capacity = field(self, layout.array_capacity);
            let full = self
                .builder
                .ins()
                .icmp(IntCC::UnsignedGreaterThanOrEqual, length, capacity);
            let grow = self.builder.create_block();
            let room = self.builder.create_block();
            self.builder.set_cold_block(grow);
            self.builder.ins().brif(full, grow, &[], room, &[]);
            self.builder.switch_to_block(grow);
            self.builder.ins().call(self.reserve_array, &[table]);
            self.builder.ins().jump(room, &[]);
            self.builder.switch_to_block(room);
            let pointer = field(self, layout.array_pointer);
            let scaled = self.builder.ins().ishl_imm(length, 4);
            let slot = self.builder.ins().iadd(pointer, scaled);
            if scalar {
                self.store_slot(slot, ty, stored);
            } else {
                self.store_reference(header, slot, (ty, stored), false, continues);
            }
            self.builder
                .ins()
                .store(MemFlags::new(), last, table, layout.array_length);
            self.builder
                .ins()
                .store(MemFlags::new(), last, table, layout.border);
            let control = self.control.expect("entry set the control block");
            let at = offset(mem::offset_of!(TraceControl, appended));
            let count = self
                .builder
                .ins()
                .load(types::I64, MemFlags::new(), control, at);
            let more = self.builder.ins().iadd_imm(count, 1);
            self.builder.ins().store(MemFlags::new(), more, control, at);
            self.store_array_words(header, (last, true), Some(pointer));
            self.builder.ins().jump(done, &[]);
        }

        // Inside the array: the last slot with nil pops, any slot takes a
        // scalar or a string over one that holds no reference.
        self.builder.switch_to_block(other);
        let slot_index = self.builder.ins().iadd_imm(index, -1);
        let inside = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, slot_index, length);
        let within = self.builder.create_block();
        self.builder.ins().brif(inside, within, &[], continues, &[]);
        self.builder.switch_to_block(within);
        let pointer = field(self, layout.array_pointer);
        let scaled = self.builder.ins().ishl_imm(slot_index, 4);
        let slot = self.builder.ins().iadd(pointer, scaled);
        let old_tag = self.builder.ins().load(types::I8, MemFlags::new(), slot, 0);
        // The tags below the string's are nil, boolean, integer and float:
        // values with nothing to release.
        let plain = self.builder.ins().icmp_imm(
            IntCC::UnsignedLessThan,
            old_tag,
            i64::from(NATIVE_VALUE_TAG_STRING),
        );
        let ready = self.builder.create_block();
        if reference || ty == TraceType::Nil {
            // The VM's entry releases whatever the slot held (a pop of a
            // string or table too, with a nil to store).
            self.builder.ins().jump(ready, &[]);
        } else {
            self.builder.ins().brif(plain, ready, &[], continues, &[]);
        }
        self.builder.switch_to_block(ready);
        if ty == TraceType::Nil {
            let is_last = self.builder.ins().icmp(IntCC::Equal, index, length);
            let pop = self.builder.create_block();
            self.builder.ins().brif(is_last, pop, &[], continues, &[]);
            self.builder.switch_to_block(pop);
            let released = self.builder.create_block();
            let shrink = self.builder.create_block();
            self.builder.ins().brif(plain, shrink, &[], released, &[]);
            self.builder.switch_to_block(released);
            self.store_reference(header, slot, (ty, stored), true, continues);
            self.builder.ins().jump(shrink, &[]);
            self.builder.switch_to_block(shrink);
            // The slot leaves the array, the border follows, and the key is
            // one of the deleted keys the table has not recorded.
            self.builder
                .ins()
                .store(MemFlags::new(), slot_index, table, layout.array_length);
            self.builder
                .ins()
                .store(MemFlags::new(), slot_index, table, layout.border);
            let pending = field(self, layout.pending_deletions);
            let lower = self
                .builder
                .ins()
                .icmp(IntCC::UnsignedLessThan, pending, length);
            let raised = self.builder.ins().select(lower, length, pending);
            self.builder
                .ins()
                .store(MemFlags::new(), raised, table, layout.pending_deletions);
            self.store_array_words(header, (slot_index, true), None);
        } else if scalar {
            self.store_slot(slot, ty, stored);
        } else {
            self.store_reference(header, slot, (ty, stored), true, continues);
        }
        self.builder.ins().jump(done, &[]);

        // A solid table whose border is below the end.
        self.builder.switch_to_block(holey);
        self.set_in_holes(
            (header, layout),
            key,
            (ty, stored),
            (flags, continues, done),
        );
    }

    /// The key `key` of a [`TraceOp::ArraySet`] as an integer; a float key
    /// that is not exactly integral jumps to `continues`.
    fn integral_key(&mut self, key: ValueId, continues: Block) -> Value {
        let key_ty = self
            .ir
            .value_type(key)
            .expect("a verified trace types every operand");
        let key_value = self.value(key);
        if key_ty == TraceType::Float {
            let converted = self.builder.ins().fcvt_to_sint_sat(types::I64, key_value);
            let back = self.builder.ins().fcvt_from_sint(types::F64, converted);
            let exact = self.builder.ins().fcmp(FloatCC::Equal, back, key_value);
            let next = self.builder.create_block();
            self.builder.ins().brif(exact, next, &[], continues, &[]);
            self.builder.switch_to_block(next);
            converted
        } else {
            key_value
        }
    }

    /// The part of a [`TraceOp::ArraySet`] native code does itself in a solid
    /// table whose border is below the end of its array part (`flags` is
    /// `NativeTableHeader::HOLEY`, or zero for a table that is not solid):
    /// the sieve's `composite[multiple] = true`. Only a scalar is stored, and
    /// only
    ///
    /// - over a slot inside the array part that holds no reference, except
    ///   the slot after the border, whose fill moves the border over the
    ///   slots that follow (the VM's job), and
    /// - for a boolean at most `NATIVE_BOOLEAN_GROWTH_REACH` slots past the
    ///   end of a non-empty array part, which the runtime stores by growing
    ///   the array part with nil holes for the slots between
    ///   (`append_dense_scalar`): done when the new slots fit the capacity and
    ///   no deletion is pending past the end. The border stays, since the
    ///   first new slot is a hole.
    ///
    /// Everything else, which includes the append, jumps to `continues`
    /// having stored nothing.
    fn set_in_holes(
        &mut self,
        (header, layout): (usize, NativeTableLayout),
        key: ValueId,
        (ty, stored): (TraceType, Value),
        (flags, continues, done): (Value, Block, Block),
    ) {
        if !matches!(
            ty,
            TraceType::Integer | TraceType::Float | TraceType::Boolean
        ) {
            self.builder.ins().jump(continues, &[]);
            return;
        }
        let solid = self.builder.create_block();
        self.builder.ins().brif(flags, solid, &[], continues, &[]);
        self.builder.switch_to_block(solid);
        let table = self.bound[header].0;
        let index = self.integral_key(key, continues);
        let field = |this: &mut Self, at: i32| {
            this.builder
                .ins()
                .load(types::I64, MemFlags::new(), table, at)
        };
        let length = field(self, layout.array_length);
        let slot_index = self.builder.ins().iadd_imm(index, -1);
        let inside = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, slot_index, length);
        let within = self.builder.create_block();
        let beyond = self.builder.create_block();
        self.builder.ins().brif(inside, within, &[], beyond, &[]);

        // Inside the array part.
        self.builder.switch_to_block(within);
        let border = field(self, layout.border);
        let successor = self.builder.ins().iadd_imm(border, 1);
        let moves = self.builder.ins().icmp(IntCC::Equal, index, successor);
        let fills = self.builder.create_block();
        self.builder.ins().brif(moves, continues, &[], fills, &[]);
        self.builder.switch_to_block(fills);
        let pointer = field(self, layout.array_pointer);
        let scaled = self.builder.ins().ishl_imm(slot_index, 4);
        let slot = self.builder.ins().iadd(pointer, scaled);
        let old_tag = self.builder.ins().load(types::I8, MemFlags::new(), slot, 0);
        // The tags below the string's hold nothing to release.
        let plain = self.builder.ins().icmp_imm(
            IntCC::UnsignedLessThan,
            old_tag,
            i64::from(NATIVE_VALUE_TAG_STRING),
        );
        let write = self.builder.create_block();
        self.builder.ins().brif(plain, write, &[], continues, &[]);
        self.builder.switch_to_block(write);
        self.store_slot(slot, ty, stored);
        self.builder.ins().jump(done, &[]);

        // Past the end.
        self.builder.switch_to_block(beyond);
        if ty != TraceType::Boolean {
            self.builder.ins().jump(continues, &[]);
            return;
        }
        let check = |this: &mut Self, condition: Value| {
            let next = this.builder.create_block();
            this.builder
                .ins()
                .brif(condition, next, &[], continues, &[]);
            this.builder.switch_to_block(next);
        };
        // Two or more past the end: the key after the last is an append.
        let last = self.builder.ins().iadd_imm(length, 1);
        let far = self
            .builder
            .ins()
            .icmp(IntCC::SignedGreaterThan, index, last);
        check(self, far);
        let reach = self
            .builder
            .ins()
            .iadd_imm(length, NATIVE_BOOLEAN_GROWTH_REACH as i64);
        let near = self
            .builder
            .ins()
            .icmp(IntCC::SignedLessThanOrEqual, index, reach);
        check(self, near);
        let started = self.builder.ins().icmp_imm(IntCC::NotEqual, length, 0);
        check(self, started);
        let capacity = field(self, layout.array_capacity);
        let fits = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThanOrEqual, index, capacity);
        check(self, fits);
        let pending = field(self, layout.pending_deletions);
        let settled = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThanOrEqual, pending, length);
        check(self, settled);
        // The holes: nil tags from the old end up to the slot before `index`.
        let pointer = field(self, layout.array_pointer);
        let fill = self.builder.create_block();
        self.builder.append_block_param(fill, types::I64);
        let filled = self.builder.create_block();
        let body = self.builder.create_block();
        self.builder.ins().jump(fill, &[length.into()]);
        self.builder.switch_to_block(fill);
        let current = self.builder.block_params(fill)[0];
        let more = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, current, slot_index);
        self.builder.ins().brif(more, body, &[], filled, &[]);
        self.builder.switch_to_block(body);
        let scaled = self.builder.ins().ishl_imm(current, 4);
        let hole = self.builder.ins().iadd(pointer, scaled);
        let nil = self
            .builder
            .ins()
            .iconst(types::I8, i64::from(NATIVE_VALUE_TAG_NIL));
        self.builder.ins().store(MemFlags::new(), nil, hole, 0);
        let next = self.builder.ins().iadd_imm(current, 1);
        self.builder.ins().jump(fill, &[next.into()]);
        self.builder.switch_to_block(filled);
        let scaled = self.builder.ins().ishl_imm(slot_index, 4);
        let slot = self.builder.ins().iadd(pointer, scaled);
        self.store_slot(slot, TraceType::Boolean, stored);
        self.builder
            .ins()
            .store(MemFlags::new(), index, table, layout.array_length);
        let control = self.control.expect("entry set the control block");
        let at = offset(mem::offset_of!(TraceControl, appended));
        let count = self
            .builder
            .ins()
            .load(types::I64, MemFlags::new(), control, at);
        let added = self.builder.ins().isub(index, length);
        let more = self.builder.ins().iadd(count, added);
        self.builder.ins().store(MemFlags::new(), more, control, at);
        self.store_array_words(header, (index, false), Some(pointer));
        self.builder.ins().jump(done, &[]);
    }

    /// Writes a scalar `value` of type `ty` into the array slot at address
    /// `slot`: its tag and its payload.
    fn store_slot(&mut self, slot: Value, ty: TraceType, value: Value) {
        let tag = match ty {
            TraceType::Float => NATIVE_VALUE_TAG_NUMBER,
            TraceType::Integer => NATIVE_VALUE_TAG_INTEGER,
            _ => NATIVE_VALUE_TAG_BOOLEAN,
        };
        let tag = self.builder.ins().iconst(types::I8, i64::from(tag));
        self.builder.ins().store(MemFlags::new(), tag, slot, 0);
        self.builder
            .ins()
            .store(MemFlags::new(), value, slot, NATIVE_VALUE_PAYLOAD_OFFSET);
    }

    /// Has the VM store the string `value` into the array slot at address
    /// `slot` (see [`oximond::jit::HostEntry::store`]); `replace` says the slot holds
    /// a value. Jumps to `refused` if the VM stored nothing.
    fn store_reference(
        &mut self,
        binding: usize,
        slot: Value,
        (ty, value): (TraceType, Value),
        replace: bool,
        refused: Block,
    ) {
        let control = self.control.expect("entry set the control block");
        let (tag, bits) = self.scalar_word(ty, value);
        let entry = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(mem::offset_of!(TraceControl, store_entry)),
        );
        let context = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(mem::offset_of!(TraceControl, host_context)),
        );
        let mut signature = Signature::new(self.builder.func.signature.call_conv);
        for _ in 0..6 {
            signature.params.push(AbiParam::new(types::I64));
        }
        signature.returns.push(AbiParam::new(types::I64));
        let signature = self.builder.import_signature(signature);
        let binding = i64::try_from(binding).expect("a binding index fits i64");
        let binding = self.builder.ins().iconst(types::I64, binding);
        let replace = self.builder.ins().iconst(types::I64, i64::from(replace));
        let call = self.builder.ins().call_indirect(
            signature,
            entry,
            &[context, binding, tag, bits, slot, replace],
        );
        let status = self.builder.inst_results(call)[0];
        let stored = self.builder.create_block();
        self.builder.ins().brif(status, refused, &[], stored, &[]);
        self.builder.switch_to_block(stored);
    }

    /// A string lookup in the table bound as `binding` ([`TraceOp::HashLoad`]).
    /// Leaves through the guard's exit unless the value found has the type
    /// `expected` needs. The lookup changes nothing, so the exit may replay
    /// the instruction.
    ///
    /// A key borrowed from an array slot, in a tree that moves no hash
    /// storage, is probed in the emitted code ([`Self::hash_probe`]); any
    /// other lookup is a call to the VM's helper, which leaves the value it
    /// found as an entry tag and bits in the control block.
    fn hash_load(
        &mut self,
        binding: u32,
        key: ValueId,
        expected: TraceType,
        guard: GuardId,
        fuel: Value,
    ) -> Result<Value, TraceCompileError> {
        let control = self.control.expect("entry set the control block");
        let key_ty = self
            .ir
            .value_type(key)
            .expect("a verified trace types every operand");
        if self.native_probe
            && key_ty == TraceType::StringRef
            && matches!(
                expected,
                TraceType::Float | TraceType::Integer | TraceType::Boolean | TraceType::Nil
            )
            && self.forced_guard != Some((self.member, guard.index()))
        {
            return self.hash_probe(binding, key, expected, guard, fuel);
        }
        self.hash_call(binding, key);
        let tag = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(
                mem::offset_of!(TraceControl, hash_result) + mem::offset_of!(TraceInputSlot, tag),
            ),
        );
        let wanted = entry_tag(expected).expect("a hash lookup expects a scalar");
        let matches = if self.forced_guard == Some((self.member, guard.index())) {
            self.builder.ins().iconst(types::I8, 0)
        } else {
            self.builder.ins().icmp_imm(IntCC::Equal, tag, wanted)
        };
        let pass = self.builder.create_block();
        let fail = self.builder.create_block();
        self.builder.set_cold_block(fail);
        self.builder.ins().brif(matches, pass, &[], fail, &[]);
        self.builder.switch_to_block(fail);
        self.leave(guard, fuel)?;
        self.builder.switch_to_block(pass);
        if expected == TraceType::Nil {
            return Ok(self.builder.ins().iconst(types::I8, 0));
        }
        let bits = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(
                mem::offset_of!(TraceControl, hash_result) + mem::offset_of!(TraceInputSlot, bits),
            ),
        );
        Ok(match expected {
            TraceType::Float => self
                .builder
                .ins()
                .bitcast(types::F64, MemFlags::new(), bits),
            TraceType::Boolean => self.builder.ins().ireduce(types::I8, bits),
            _ => bits,
        })
    }

    /// The call to the VM's lookup helper for the key in `key` of the table
    /// bound as `binding`, which leaves its answer in the control block's
    /// `hash_result`.
    fn hash_call(&mut self, binding: u32, key: ValueId) {
        let control = self.control.expect("entry set the control block");
        let key_ty = self
            .ir
            .value_type(key)
            .expect("a verified trace types every operand");
        let key_tag = match key_ty {
            TraceType::StringRef => TRACE_TAG_BORROWED_STRING,
            TraceType::Integer => TRACE_TAG_INTEGER,
            TraceType::Float => TRACE_TAG_FLOAT,
            _ => TRACE_TAG_STRING,
        };
        let key_tag = self.builder.ins().iconst(types::I64, key_tag);
        let key_bits = self.value(key);
        let key_bits = if key_ty == TraceType::Float {
            self.builder
                .ins()
                .bitcast(types::I64, MemFlags::new(), key_bits)
        } else {
            key_bits
        };
        let binding_value = self.builder.ins().iconst(types::I64, i64::from(binding));
        let dispatch = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(mem::offset_of!(TraceControl, tables)),
        );
        let out = self.builder.ins().iadd_imm(
            control,
            i64::from(offset(mem::offset_of!(TraceControl, hash_result))),
        );
        let callee = if matches!(key_ty, TraceType::Integer | TraceType::Float) {
            self.hash_get_number
                .expect("a tree that looks up number keys has the stub")
        } else {
            self.hash_get
                .expect("a tree that looks up hashes has the helper")
        };
        self.builder
            .ins()
            .call(callee, &[dispatch, out, binding_value, key_tag, key_bits]);
    }

    /// The probe of the string index bound as `binding` for the key in `key`
    /// (a string borrowed from an array slot), shared by lookups and stores.
    /// Continues in `hit`, whose one parameter is the address of the index's
    /// bucket for the key, in `miss` when the index has no such key, and in
    /// `slow` when native code cannot tell: the binding has no index, the key's
    /// hash is not computed yet, or a candidate or the key is a long string
    /// that needs a byte comparison. Returns the address of the key's storage,
    /// a value that stands in `hit` and the blocks after it.
    #[allow(clippy::too_many_lines)]
    fn probe_bucket(
        &mut self,
        binding: u32,
        key: ValueId,
        (hit, miss, slow): (Block, Block, Block),
    ) -> Value {
        let (slots, mask) = self.bound[binding as usize];
        let key_slot = self.value(key);
        let have_index = self.builder.create_block();
        let hashed = self.builder.create_block();
        let probing = self.builder.create_block();
        self.builder.append_block_param(probing, types::I64);
        let occupied = self.builder.create_block();
        let same_hash = self.builder.create_block();
        let compare = self.builder.create_block();
        let inline_key = self.builder.create_block();
        let next = self.builder.create_block();

        // The key's storage and hash. A binding with no index has zero slots.
        self.builder.ins().brif(slots, have_index, &[], slow, &[]);
        self.builder.switch_to_block(have_index);
        let payload = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            key_slot,
            NATIVE_VALUE_PAYLOAD_OFFSET,
        );
        let data = self.builder.ins().iadd_imm(
            payload,
            i64::try_from(NATIVE_STRING_STORAGE_OFFSET).expect("small"),
        );
        let hash = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            data,
            i32::try_from(NATIVE_STRING_HASH_OFFSET).expect("small"),
        );
        self.builder.ins().brif(hash, hashed, &[], slow, &[]);

        // The first bucket: where the runtime places the key
        // (`native_probe_start`), the high half of its hash.
        self.builder.switch_to_block(hashed);
        let start = self
            .builder
            .ins()
            .ushr_imm(hash, i64::from(NATIVE_PROBE_SHIFT));
        let start = self.builder.ins().band(start, mask);
        self.builder.ins().jump(probing, &[BlockArg::Value(start)]);

        self.builder.switch_to_block(probing);
        let at = self.builder.block_params(probing)[0];
        let offset_bytes = self
            .builder
            .ins()
            .imul_imm(at, i64::try_from(NATIVE_PROBE_SLOT_SIZE).expect("small"));
        let bucket = self.builder.ins().iadd(slots, offset_bytes);
        let stored = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            bucket,
            i32::try_from(NATIVE_PROBE_SLOT_KEY).expect("small"),
        );
        self.builder.ins().brif(stored, occupied, &[], miss, &[]);

        self.builder.switch_to_block(occupied);
        let stored_hash = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            bucket,
            i32::try_from(NATIVE_PROBE_SLOT_HASH).expect("small"),
        );
        let hash_matches = self.builder.ins().icmp(IntCC::Equal, stored_hash, hash);
        self.builder
            .ins()
            .brif(hash_matches, same_hash, &[], next, &[]);

        self.builder.switch_to_block(same_hash);
        let same_string = self.builder.ins().icmp(IntCC::Equal, stored, data);
        self.builder
            .ins()
            .brif(same_string, hit, &[BlockArg::Value(bucket)], compare, &[]);

        // Not the same allocation: two inline strings are equal when their
        // words are, and a long key needs a byte comparison, which the
        // helper does. A long stored string is no inline string's equal.
        self.builder.switch_to_block(compare);
        let key_form = self.builder.ins().load(types::I8, MemFlags::new(), data, 0);
        let key_inline = self.builder.ins().icmp_imm(
            IntCC::Equal,
            key_form,
            i64::from(NATIVE_STRING_INLINE_TAG),
        );
        self.builder
            .ins()
            .brif(key_inline, inline_key, &[], slow, &[]);
        self.builder.switch_to_block(inline_key);
        let mut all_equal = None;
        for word in 0..NATIVE_STRING_INLINE_WORDS {
            let at = i32::try_from(word * 8).expect("small");
            let ours = self
                .builder
                .ins()
                .load(types::I64, MemFlags::new(), data, at);
            let theirs = self
                .builder
                .ins()
                .load(types::I64, MemFlags::new(), stored, at);
            let same = self.builder.ins().icmp(IntCC::Equal, ours, theirs);
            all_equal = Some(match all_equal {
                None => same,
                Some(before) => self.builder.ins().band(before, same),
            });
        }
        let all_equal = all_equal.expect("an inline string has words");
        self.builder
            .ins()
            .brif(all_equal, hit, &[BlockArg::Value(bucket)], next, &[]);

        self.builder.switch_to_block(next);
        let after = self.builder.ins().iadd_imm(at, 1);
        let after = self.builder.ins().band(after, mask);
        self.builder.ins().jump(probing, &[BlockArg::Value(after)]);
        data
    }

    /// [`Self::hash_load`] with the probe in the emitted code, for a key
    /// borrowed from an array slot and a value of a native scalar type.
    ///
    /// The VM publishes, as the words of the binding, a string index of the
    /// bound table ([`oximond::runtime::NativeStringProbe`]: the address of its
    /// slots and its mask), or zero when it has none. No operation of the
    /// tree moves hash storage, so the index stays as it is for the whole
    /// run. The probe reads the key's cached hash, and takes the string's
    /// storage address as its identity: a candidate whose hash matches is the
    /// entry when it is the same allocation, or when both are inline strings
    /// with the same words. The helper answers instead (with exactly what it
    /// always answered) when the binding has no index, when the key's hash
    /// is not computed yet, and when a candidate or the key is a long string
    /// that needs a byte comparison. Finding no key is a nil, as it is for the
    /// helper; a stored nil is a deleted key and reads the same.
    #[allow(clippy::too_many_lines)]
    fn hash_probe(
        &mut self,
        binding: u32,
        key: ValueId,
        expected: TraceType,
        guard: GuardId,
        fuel: Value,
    ) -> Result<Value, TraceCompileError> {
        let control = self.control.expect("entry set the control block");
        let wanted_native = match expected {
            TraceType::Float => NATIVE_VALUE_TAG_NUMBER,
            TraceType::Integer => NATIVE_VALUE_TAG_INTEGER,
            TraceType::Boolean => NATIVE_VALUE_TAG_BOOLEAN,
            _ => NATIVE_VALUE_TAG_NIL,
        };
        let wanted_entry = entry_tag(expected).expect("a hash lookup expects a scalar");
        let result_ty = (expected != TraceType::Nil).then(|| clif_type(expected));

        let hit = self.builder.create_block();
        self.builder.append_block_param(hit, types::I64);
        let hit_typed = self.builder.create_block();
        let miss = self.builder.create_block();
        let slow = self.builder.create_block();
        let slow_typed = self.builder.create_block();
        let pass = self.builder.create_block();
        if let Some(ty) = result_ty {
            self.builder.append_block_param(pass, ty);
        }
        let fail = self.builder.create_block();
        self.builder.set_cold_block(slow);
        self.builder.set_cold_block(fail);

        self.probe_bucket(binding, key, (hit, miss, slow));

        // The entry: its value must have the type the trace expects.
        self.builder.switch_to_block(hit);
        let bucket = self.builder.block_params(hit)[0];
        let value = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            bucket,
            i32::try_from(NATIVE_PROBE_SLOT_VALUE).expect("small"),
        );
        let tag = self
            .builder
            .ins()
            .load(types::I8, MemFlags::new(), value, 0);
        let typed = self
            .builder
            .ins()
            .icmp_imm(IntCC::Equal, tag, i64::from(wanted_native));
        self.builder.ins().brif(typed, hit_typed, &[], fail, &[]);
        self.builder.switch_to_block(hit_typed);
        match result_ty {
            Some(ty) => {
                let loaded = self.builder.ins().load(
                    ty,
                    MemFlags::new(),
                    value,
                    NATIVE_VALUE_PAYLOAD_OFFSET,
                );
                self.builder.ins().jump(pass, &[BlockArg::Value(loaded)]);
            }
            None => {
                self.builder.ins().jump(pass, &[]);
            }
        }

        // No such key reads as nil, which only a trace that expects nil takes.
        self.builder.switch_to_block(miss);
        match result_ty {
            Some(_) => {
                self.builder.ins().jump(fail, &[]);
            }
            None => {
                self.builder.ins().jump(pass, &[]);
            }
        }

        // The helper's answer, as the call always gave it.
        self.builder.switch_to_block(slow);
        self.hash_call(binding, key);
        let answer = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(
                mem::offset_of!(TraceControl, hash_result) + mem::offset_of!(TraceInputSlot, tag),
            ),
        );
        let matches = self
            .builder
            .ins()
            .icmp_imm(IntCC::Equal, answer, wanted_entry);
        self.builder.ins().brif(matches, slow_typed, &[], fail, &[]);
        self.builder.switch_to_block(slow_typed);
        if let Some(ty) = result_ty {
            let bits = self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                control,
                offset(
                    mem::offset_of!(TraceControl, hash_result)
                        + mem::offset_of!(TraceInputSlot, bits),
                ),
            );
            let converted = match ty {
                types::F64 => self
                    .builder
                    .ins()
                    .bitcast(types::F64, MemFlags::new(), bits),
                types::I8 => self.builder.ins().ireduce(types::I8, bits),
                _ => bits,
            };
            self.builder.ins().jump(pass, &[BlockArg::Value(converted)]);
        } else {
            self.builder.ins().jump(pass, &[]);
        }

        self.builder.switch_to_block(fail);
        self.leave(guard, fuel)?;
        self.builder.switch_to_block(pass);
        Ok(match result_ty {
            Some(_) => self.builder.block_params(pass)[0],
            None => self.builder.ins().iconst(types::I8, 0),
        })
    }

    /// A call of a library function through the VM ([`TraceOp::HostCall`]):
    /// the arguments go to the control block as entry tags and bits, the
    /// helper leaves its answer there, and the guard leaves unless the answer
    /// has the tag `result` needs. The helper changes nothing a guest can
    /// see, so the exit may replay the instruction.
    /// Calls the VM's host call entry the control block names: `(context,
    /// args, out, function, argc)`.
    fn call_host(
        &mut self,
        control: Value,
        args: Value,
        out: Value,
        function: Value,
        count: Value,
    ) {
        let entry = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(mem::offset_of!(TraceControl, host_entry)),
        );
        let context = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(mem::offset_of!(TraceControl, host_context)),
        );
        let mut signature = Signature::new(self.builder.func.signature.call_conv);
        for _ in 0..3 {
            signature.params.push(AbiParam::new(types::I64));
        }
        signature.params.push(AbiParam::new(types::I64));
        signature.params.push(AbiParam::new(types::I64));
        let signature = self.builder.import_signature(signature);
        self.builder
            .ins()
            .call_indirect(signature, entry, &[context, args, out, function, count]);
    }

    /// A call of a helper ([`TraceOp::CallHelper`]). The step of a boxed
    /// `next` loop over a table with an array part first tries the step in
    /// native code (see [`Self::call_helper_stepping`]).
    fn call_helper(
        &mut self,
        helper: HelperId,
        args: &[ValueId],
        result: Option<ValueId>,
        guard: GuardId,
        fuel: Value,
    ) -> Result<Option<Value>, TraceCompileError> {
        // A held delete fuses with this store, or is emitted before it.
        if let Some(delete) = self.pending_delete.take() {
            if self.row_words == 0
                && helper == HelperId::SET_REVIVE
                && guard == delete.store
                && args[..2] == delete.args[..]
                && let Some(binding) = self.native_store_binding(helper, args, guard)
            {
                if let Some(made) = self.made_tables.get_mut(&args[0]) {
                    made.moved = true;
                }
                self.hash_overwrite(binding, &delete, (args, guard), fuel)?;
                return Ok(None);
            }
            self.flush_pending_delete_with(&delete, fuel)?;
        }
        // Any other store by helper may grow the table a made table's
        // slots belong to, which later in-place fills must not trust.
        if helper != HelperId::TABLE_INIT
            && helper.effects().contains(HelperEffects::WRITES_HEAP)
            && let Some(made) = args
                .first()
                .and_then(|table| self.made_tables.get_mut(table))
        {
            made.moved = true;
        }
        if self.row_words > 0
            && let Some(cell) = closure_cell_use(
                self.ir,
                helper,
                args,
                self.row_words == super::exit::CLOSURE_ROW_TABLE_WORDS,
            )
        {
            return self.closure_cell(cell, args, guard, fuel);
        }
        if helper == HelperId::TABLE_INIT {
            if self.table_init_in_place(args, guard, fuel)? {
                return Ok(None);
            }
            self.note_table_init_by_helper(args);
        }
        // A delete that fuses with the store after it waits for that store.
        if helper == HelperId::DELETE_STRING
            && self.row_words == 0
            && let Some(&store) = self.fused.get(&guard)
            && let Some(binding) = self.native_store_binding(helper, args, guard)
        {
            self.pending_delete = Some(PendingDelete {
                args: args.to_vec(),
                guard,
                store,
                binding,
            });
            return Ok(None);
        }
        if let Some(binding) = self.native_store_binding(helper, args, guard) {
            return self.hash_store(binding, (helper, args), result, guard, fuel);
        }
        if self.forced_guard != Some((self.member, guard.index()))
            && let Some((borrowed, words)) = string_equal_constant(self.ir, helper, args)
        {
            return Ok(Some(self.string_equal_inline(borrowed, words)));
        }
        if self.forced_guard != Some((self.member, guard.index()))
            && string_equal_by_words(self.ir, helper, args)
        {
            return self.string_equal_by_words(helper, args, result, guard, fuel);
        }
        if let Some(plan) = self.inline_run_plan(helper, args, guard) {
            return self.call_inline_run(&plan, helper, args, result, guard, fuel);
        }
        match self.box_step_of(helper, args, guard) {
            Some(step) => self.call_helper_stepping(step, (helper, args), result, guard, fuel),
            None => self.call_helper_call(helper, args, result, guard, fuel, false),
        }
    }

    /// `borrowed == constant` for a short root-string constant whose storage
    /// words are `words` (see [`string_equal_constant`]): the words of the
    /// borrowed string's storage against those, as a boolean.
    fn string_equal_inline(
        &mut self,
        borrowed: ValueId,
        words: [u64; NATIVE_STRING_INLINE_WORDS],
    ) -> Value {
        let value_at = self.value(borrowed);
        let payload = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            value_at,
            NATIVE_VALUE_PAYLOAD_OFFSET,
        );
        let mut different = None;
        for (word, constant) in words.into_iter().enumerate() {
            let at = i32::try_from(NATIVE_STRING_STORAGE_OFFSET + 8 * word).expect("small");
            let held = self
                .builder
                .ins()
                .load(types::I64, MemFlags::new(), payload, at);
            let apart = self.builder.ins().bxor_imm(held, constant.cast_signed());
            different = Some(match different {
                None => apart,
                Some(before) => self.builder.ins().bor(before, apart),
            });
        }
        let different = different.expect("an inline string has words");
        self.builder.ins().icmp_imm(IntCC::Equal, different, 0)
    }

    /// `==` of two strings by the words of their storage, the helper's answer
    /// when those cannot tell. A borrowed string is the address of its value;
    /// a lane names a value of the run's bank, or of the trace's roots from
    /// [`ROOT_STRING_LANE_BASE`] on. Two inline strings are equal exactly when
    /// their words are, an inline and a heap string never are, and a heap
    /// string is equal to itself. The helper compares two heap strings in
    /// two allocations, and a lane the code cannot read: a run with no bank
    /// or roots, a root slot past the roots, or a lane that holds no string.
    fn string_equal_by_words(
        &mut self,
        helper: HelperId,
        args: &[ValueId],
        result: Option<ValueId>,
        guard: GuardId,
        fuel: Value,
    ) -> Result<Option<Value>, TraceCompileError> {
        let control = self.control.expect("entry set the control block");
        let slow = self.builder.create_block();
        self.builder.set_cold_block(slow);
        let merge = self.builder.create_block();
        self.builder.append_block_param(merge, types::I8);
        let mut storages = [None; 2];
        for (position, &arg) in args.iter().enumerate() {
            let ty = self
                .ir
                .value_type(arg)
                .expect("a verified trace types every operand");
            let value = self.value(arg);
            let value_at = if ty == TraceType::StringRef {
                value
            } else {
                let load = |lowerer: &mut Self, field: usize| {
                    lowerer
                        .builder
                        .ins()
                        .load(types::I64, MemFlags::new(), control, offset(field))
                };
                let root = self.builder.ins().icmp_imm(
                    IntCC::UnsignedGreaterThanOrEqual,
                    value,
                    ROOT_STRING_LANE_BASE.cast_signed(),
                );
                let roots = load(self, mem::offset_of!(TraceControl, roots));
                let roots_len = load(self, mem::offset_of!(TraceControl, roots_len));
                let bank = load(self, mem::offset_of!(TraceControl, bank));
                let base = self.builder.ins().select(root, roots, bank);
                let lane_base = self
                    .builder
                    .ins()
                    .iconst(types::I64, ROOT_STRING_LANE_BASE.cast_signed());
                let zero = self.builder.ins().iconst(types::I64, 0);
                let first = self.builder.ins().select(root, lane_base, zero);
                let index = self.builder.ins().isub(value, first);
                // The bank holds every lane the run gave; a root slot is
                // checked against the roots.
                let unbounded = self.builder.ins().iconst(types::I64, -1);
                let limit = self.builder.ins().select(root, roots_len, unbounded);
                let inside = self
                    .builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThan, index, limit);
                let present = self.builder.ins().icmp_imm(IntCC::NotEqual, base, 0);
                let readable = self.builder.ins().band(inside, present);
                let read = self.builder.create_block();
                self.builder.ins().brif(readable, read, &[], slow, &[]);
                self.builder.switch_to_block(read);
                let scaled = self.builder.ins().ishl_imm(index, 4);
                let at = self.builder.ins().iadd(base, scaled);
                let tag = self
                    .builder
                    .ins()
                    .uload8(types::I64, MemFlags::new(), at, 0);
                let string = self.builder.ins().icmp_imm(
                    IntCC::Equal,
                    tag,
                    i64::from(NATIVE_VALUE_TAG_STRING),
                );
                let held = self.builder.create_block();
                self.builder.ins().brif(string, held, &[], slow, &[]);
                self.builder.switch_to_block(held);
                at
            };
            storages[position] = Some(self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                value_at,
                NATIVE_VALUE_PAYLOAD_OFFSET,
            ));
        }
        let [Some(left), Some(right)] = storages else {
            unreachable!("two operands");
        };
        let mut different = None;
        let mut firsts = [None; 2];
        for word in 0..NATIVE_STRING_INLINE_WORDS {
            let at = i32::try_from(NATIVE_STRING_STORAGE_OFFSET + 8 * word).expect("small");
            let ours = self
                .builder
                .ins()
                .load(types::I64, MemFlags::new(), left, at);
            let theirs = self
                .builder
                .ins()
                .load(types::I64, MemFlags::new(), right, at);
            if word == 0 {
                firsts = [Some(ours), Some(theirs)];
            }
            let apart = self.builder.ins().bxor(ours, theirs);
            different = Some(match different {
                None => apart,
                Some(before) => self.builder.ins().bor(before, apart),
            });
        }
        let different = different.expect("an inline string has words");
        let equal = self.builder.ins().icmp_imm(IntCC::Equal, different, 0);
        // The form byte comes first: two heap strings in two allocations need
        // their bytes compared.
        let [Some(ours), Some(theirs)] = firsts else {
            unreachable!("a first word each");
        };
        let forms = self.builder.ins().band(ours, theirs);
        let both_heap = self.builder.ins().band_imm(forms, 0xff);
        let both_heap = self.builder.ins().icmp_imm(IntCC::NotEqual, both_heap, 0);
        let apart = self.builder.ins().icmp(IntCC::NotEqual, left, right);
        let undecided = self.builder.ins().band(both_heap, apart);
        self.builder
            .ins()
            .brif(undecided, slow, &[], merge, &[BlockArg::Value(equal)]);
        self.builder.switch_to_block(slow);
        let answer = self
            .call_helper_call(helper, args, result, guard, fuel, false)?
            .expect("the comparison has a result");
        self.builder.ins().jump(merge, &[BlockArg::Value(answer)]);
        self.builder.switch_to_block(merge);
        Ok(Some(self.builder.block_params(merge)[0]))
    }

    /// Notes that the helper stores under the key of a [`HelperId::TABLE_INIT`]
    /// call that was not lowered. A key the made table's shape lacks may
    /// move the table's slots, which later in-place stores must not trust.
    fn note_table_init_by_helper(&mut self, args: &[ValueId]) {
        let Some(made) = self.made_tables.get(&args[0]) else {
            return;
        };
        if super::made_table::init_by_helper_moves(
            self.ir,
            &made.made,
            self.constants[args[1].index()],
            args[2],
        ) {
            self.made_tables
                .get_mut(&args[0])
                .expect("looked up above")
                .moved = true;
        }
    }

    /// A call of [`HelperId::TABLE_INIT`] done in the emitted code when it
    /// fills a field of a shaped table [`HelperId::TABLE_NEW`] made in this
    /// member, with a number, a boolean or a table: the field's slot is
    /// written where it is, a nil slot the shape named, and the run counts
    /// the field for the collector's charge. A table stored over needs the
    /// write barrier only while a major cycle is in progress, which the host
    /// says per run ([`TraceControl::barrier_idle`]); a run in progress
    /// takes the helper. Returns whether the call was lowered. Anything else
    /// (a second store under the key, a string, a key the shape lacks, a
    /// table with no shape) is the helper's.
    fn table_init_in_place(
        &mut self,
        args: &[ValueId],
        guard: GuardId,
        fuel: Value,
    ) -> Result<bool, TraceCompileError> {
        let Some(made) = self.made_tables.get(&args[0]) else {
            return Ok(false);
        };
        if made.moved {
            return Ok(false);
        }
        let slots = made.slots;
        let Some(field) =
            super::made_table::init_field(self.ir, &made.made, self.constants[args[1].index()])
        else {
            return Ok(false);
        };
        let ty = self
            .ir
            .value_type(args[2])
            .ok_or(TraceCompileError::NativeBackendRejected)?;
        if !matches!(
            ty,
            TraceType::Float | TraceType::Integer | TraceType::Boolean | TraceType::Table
        ) {
            return Ok(false);
        }
        let made = self.made_tables.get_mut(&args[0]).expect("looked up above");
        if made.filled.contains(&field) {
            return Ok(false);
        }
        made.filled.push(field);
        let control = self.control.expect("entry set the control block");
        let value = self.value(args[2]);
        let in_place = self.builder.create_block();
        let by_helper = self.builder.create_block();
        self.builder.set_cold_block(by_helper);
        let done = self.builder.create_block();
        // The helper makes the store when the table has no slots to name, or
        // (for a table stored) when the collector's barrier may be needed or
        // the host gave no bank to read the table's lane from.
        let mut ready = self.builder.ins().icmp_imm(IntCC::NotEqual, slots, 0);
        let bank = if ty == TraceType::Table {
            let idle = self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                control,
                offset(mem::offset_of!(TraceControl, barrier_idle)),
            );
            let bank = self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                control,
                offset(mem::offset_of!(TraceControl, bank)),
            );
            let idle = self.builder.ins().icmp_imm(IntCC::NotEqual, idle, 0);
            let banked = self.builder.ins().icmp_imm(IntCC::NotEqual, bank, 0);
            let both = self.builder.ins().band(idle, banked);
            ready = self.builder.ins().band(ready, both);
            Some(bank)
        } else {
            None
        };
        self.builder
            .ins()
            .brif(ready, in_place, &[], by_helper, &[]);
        self.builder.switch_to_block(in_place);
        let at = i32::try_from(field * mem::size_of::<crate::runtime::Value>())
            .map_err(|_| TraceCompileError::NativeBackendRejected)?;
        if let Some(bank) = bank {
            // The table is the value of its lane: both words of it move.
            let lane = value;
            let from = self.builder.ins().ishl_imm(lane, 4);
            let from = self.builder.ins().iadd(bank, from);
            for word in [0, 8] {
                let copied = self
                    .builder
                    .ins()
                    .load(types::I64, MemFlags::new(), from, word);
                self.builder
                    .ins()
                    .store(MemFlags::new(), copied, slots, at + word);
            }
        } else {
            let tag = match ty {
                TraceType::Float => NATIVE_VALUE_TAG_NUMBER,
                TraceType::Integer => NATIVE_VALUE_TAG_INTEGER,
                _ => NATIVE_VALUE_TAG_BOOLEAN,
            };
            let tag = self.builder.ins().iconst(types::I8, i64::from(tag));
            self.builder.ins().store(MemFlags::new(), tag, slots, at);
            self.builder.ins().store(
                MemFlags::new(),
                value,
                slots,
                at + NATIVE_VALUE_PAYLOAD_OFFSET,
            );
        }
        let counted = offset(mem::offset_of!(TraceControl, initialized));
        let total = self
            .builder
            .ins()
            .load(types::I64, MemFlags::new(), control, counted);
        let more = self.builder.ins().iadd_imm(total, 1);
        self.builder
            .ins()
            .store(MemFlags::new(), more, control, counted);
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(by_helper);
        self.call_helper_call(HelperId::TABLE_INIT, args, None, guard, fuel, false)?;
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
        Ok(true)
    }

    /// The binding whose string index a call of one of the three in-place
    /// store helpers ([`HelperId::SET_PRESENT`], [`HelperId::DELETE_STRING`]
    /// and [`HelperId::SET_REVIVE`]) stores through in native code
    /// ([`Self::hash_store`]), if the call has one: a root-trace call of a tree
    /// that moves no hash storage, with a key borrowed from an array slot (and
    /// a number or boolean to store), on a table that is an entry input the
    /// recorder bound for its hash part. A store into a deleted key needs the
    /// binding that checked the table has no metatable. Any other call is the
    /// VM's.
    fn native_store_binding(
        &self,
        helper: HelperId,
        args: &[ValueId],
        guard: GuardId,
    ) -> Option<u32> {
        let stores_value = match helper {
            HelperId::SET_PRESENT | HelperId::SET_REVIVE => true,
            HelperId::DELETE_STRING => false,
            _ => return None,
        };
        if !self.native_probe
            || self.member != 0
            || args.len() != 2 + usize::from(stores_value)
            || self.forced_guard == Some((self.member, guard.index()))
            || self.ir.value_type(args[1]) != Some(TraceType::StringRef)
            || (stores_value
                && !matches!(
                    self.ir.value_type(args[2]),
                    Some(TraceType::Float | TraceType::Integer | TraceType::Boolean)
                ))
        {
            return None;
        }
        let register = self
            .ir
            .entry
            .inputs
            .iter()
            .find(|input| input.value == args[0])?
            .register;
        let bindings = &self.members[0].ir.bindings;
        let found = bindings.iter().position(|binding| {
            binding.register == register
                && match binding.kind {
                    BindingKind::NoMetatable => true,
                    BindingKind::Hash => helper != HelperId::SET_REVIVE,
                    _ => false,
                }
        })?;
        u32::try_from(found).ok()
    }

    /// A call of one of the in-place store helpers with the store in the
    /// emitted code: the probe finds the key's bucket in the bound table's
    /// string index ([`Self::probe_bucket`]), and the entry is changed where
    /// it is, with no call. A call to the VM's helper answers instead, with
    /// exactly what it always answered, when the binding has no index or an
    /// index native code must not store through ([`PROBE_READ_ONLY`]: a
    /// read-only table, a watched one, the one a collection is tracing), when
    /// the probe cannot tell, when the key is absent, and in the cases each
    /// store lists below. The helper then stores or leaves through the guard.
    ///
    /// - [`HelperId::SET_PRESENT`]: the entry holds the type being stored. A
    ///   number or boolean over its own type keeps the table's layout, owns no
    ///   reference and needs no barrier.
    /// - [`HelperId::DELETE_STRING`]: the entry holds a number or boolean,
    ///   which becomes nil, a deleted key the traversals skip, and the table's
    ///   live-entry counter drops. The key must not be one that may name a
    ///   metamethod (its first two bytes are not `__`, which only an inline
    ///   string says), the table must be one the VM let native code delete in
    ///   ([`PROBE_NO_TOMBSTONES`]).
    /// - [`HelperId::SET_REVIVE`]: the entry holds nil, which takes the number
    ///   or boolean, and the counter rises; the same conditions on the key
    ///   and the table.
    #[allow(clippy::too_many_lines)]
    fn hash_store(
        &mut self,
        binding: u32,
        (helper, args): (HelperId, &[ValueId]),
        result: Option<ValueId>,
        guard: GuardId,
        fuel: Value,
    ) -> Result<Option<Value>, TraceCompileError> {
        let (slots, mask) = self.bound[binding as usize];
        let stored = (helper != HelperId::DELETE_STRING).then(|| {
            let ty = self
                .ir
                .value_type(args[2])
                .expect("a verified trace types every operand");
            (self.value(args[2]), ty)
        });
        let hit = self.builder.create_block();
        self.builder.append_block_param(hit, types::I64);
        let allowed = self.builder.create_block();
        let keyed = self.builder.create_block();
        let typed = self.builder.create_block();
        let miss = self.builder.create_block();
        let slow = self.builder.create_block();
        let done = self.builder.create_block();
        self.builder.set_cold_block(slow);
        let data = self.probe_bucket(binding, args[1], (hit, miss, slow));

        // The index says whether this table takes the store.
        self.builder.switch_to_block(hit);
        let bucket = self.builder.block_params(hit)[0];
        let open = if helper == HelperId::SET_PRESENT {
            self.builder
                .ins()
                .icmp_imm(IntCC::SignedGreaterThanOrEqual, mask, 0)
        } else {
            let flag = self.builder.ins().band_imm(mask, PROBE_NO_TOMBSTONES);
            self.builder.ins().icmp_imm(IntCC::Equal, flag, 0)
        };
        self.builder.ins().brif(open, allowed, &[], slow, &[]);
        self.builder.switch_to_block(allowed);
        if helper == HelperId::SET_PRESENT {
            self.builder.ins().jump(keyed, &[]);
        } else {
            // A key that may name a metamethod changes what a metatable's
            // lookups answer: the VM's. An inline string has its bytes
            // after the tag and the length, zero-padded; a long one is the
            // VM's too.
            let form = self.builder.ins().load(types::I8, MemFlags::new(), data, 0);
            let inline = self.builder.ins().icmp_imm(
                IntCC::Equal,
                form,
                i64::from(NATIVE_STRING_INLINE_TAG),
            );
            let inline_key = self.builder.create_block();
            self.builder.ins().brif(inline, inline_key, &[], slow, &[]);
            self.builder.switch_to_block(inline_key);
            let first = self
                .builder
                .ins()
                .load(types::I16, MemFlags::new(), data, 2);
            let plain = self.builder.ins().icmp_imm(IntCC::NotEqual, first, 0x5f5f);
            self.builder.ins().brif(plain, keyed, &[], slow, &[]);
        }
        self.builder.switch_to_block(keyed);
        let entry = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            bucket,
            i32::try_from(NATIVE_PROBE_SLOT_VALUE).expect("small"),
        );
        let tag = self
            .builder
            .ins()
            .load(types::I8, MemFlags::new(), entry, 0);
        let fits = match (helper, stored) {
            (HelperId::SET_PRESENT, Some((_, ty))) => {
                let wanted = match ty {
                    TraceType::Float => NATIVE_VALUE_TAG_NUMBER,
                    TraceType::Integer => NATIVE_VALUE_TAG_INTEGER,
                    _ => NATIVE_VALUE_TAG_BOOLEAN,
                };
                self.builder
                    .ins()
                    .icmp_imm(IntCC::Equal, tag, i64::from(wanted))
            }
            (HelperId::SET_REVIVE, _) => {
                self.builder
                    .ins()
                    .icmp_imm(IntCC::Equal, tag, i64::from(NATIVE_VALUE_TAG_NIL))
            }
            _ => {
                // A number, an integer or a boolean: the tags 1 to 3.
                let shifted = self.builder.ins().iadd_imm(tag, -1);
                self.builder
                    .ins()
                    .icmp_imm(IntCC::UnsignedLessThan, shifted, 3)
            }
        };
        self.builder.ins().brif(fits, typed, &[], slow, &[]);

        self.builder.switch_to_block(typed);
        match (helper, stored) {
            (HelperId::SET_PRESENT, Some((value, _))) => {
                self.builder.ins().store(
                    MemFlags::new(),
                    value,
                    entry,
                    NATIVE_VALUE_PAYLOAD_OFFSET,
                );
            }
            (_, stored) => {
                // The table's live-entry counter sits in the header slot
                // after the index, whose size the mask's low bits give.
                let slots_mask = self.builder.ins().band_imm(mask, PROBE_NO_TOMBSTONES - 1);
                let header = self.builder.ins().iadd_imm(slots_mask, 1);
                let header = self.builder.ins().imul_imm(
                    header,
                    i64::try_from(NATIVE_PROBE_SLOT_SIZE).expect("small"),
                );
                let header = self.builder.ins().iadd(slots, header);
                let counter = self.builder.ins().load(
                    types::I64,
                    MemFlags::new(),
                    header,
                    i32::try_from(NATIVE_PROBE_SLOT_HASH).expect("small"),
                );
                let live = self
                    .builder
                    .ins()
                    .load(types::I64, MemFlags::new(), counter, 0);
                if let Some((value, ty)) = stored {
                    let new_tag = match ty {
                        TraceType::Float => NATIVE_VALUE_TAG_NUMBER,
                        TraceType::Integer => NATIVE_VALUE_TAG_INTEGER,
                        _ => NATIVE_VALUE_TAG_BOOLEAN,
                    };
                    self.builder.ins().store(
                        MemFlags::new(),
                        value,
                        entry,
                        NATIVE_VALUE_PAYLOAD_OFFSET,
                    );
                    let new_tag = self.builder.ins().iconst(types::I8, i64::from(new_tag));
                    self.builder.ins().store(MemFlags::new(), new_tag, entry, 0);
                    let more = self.builder.ins().iadd_imm(live, 1);
                    self.builder.ins().store(MemFlags::new(), more, counter, 0);
                } else {
                    let nil = self
                        .builder
                        .ins()
                        .iconst(types::I8, i64::from(NATIVE_VALUE_TAG_NIL));
                    self.builder.ins().store(MemFlags::new(), nil, entry, 0);
                    let fewer = self.builder.ins().iadd_imm(live, -1);
                    self.builder.ins().store(MemFlags::new(), fewer, counter, 0);
                }
            }
        }
        self.builder.ins().jump(done, &[]);

        self.builder.switch_to_block(miss);
        self.builder.ins().jump(slow, &[]);
        self.builder.switch_to_block(slow);
        self.call_helper_call(helper, args, result, guard, fuel, true)?;
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
        Ok(None)
    }

    /// Lowers a held delete ([`PendingDelete`]) that met no store to fuse
    /// with: its own native store, as [`Self::call_helper`] would have made it.
    fn flush_pending_delete(&mut self, fuel: Value) -> Result<(), TraceCompileError> {
        if let Some(delete) = self.pending_delete.take() {
            self.flush_pending_delete_with(&delete, fuel)?;
        }
        Ok(())
    }

    fn flush_pending_delete_with(
        &mut self,
        delete: &PendingDelete,
        fuel: Value,
    ) -> Result<(), TraceCompileError> {
        self.hash_store(
            delete.binding,
            (HelperId::DELETE_STRING, &delete.args),
            None,
            delete.guard,
            fuel,
        )?;
        Ok(())
    }

    /// Fuses `t[k] = nil` with the `t[k] = v` after it, for the same table and
    /// key (see [`fused_delete_stores`]). The entry is probed once:
    ///
    /// - a live number, integer or boolean takes `v` where it stands. The
    ///   delete and the store would drop the live-entry counter and raise it
    ///   again, which nets to no change.
    /// - a deleted key (nil) takes `v` and the counter rises, as the
    ///   [`HelperId::SET_REVIVE`] arm of [`Self::hash_store`] does. The
    ///   delete of a nil key changes nothing.
    ///
    /// Any other entry, a miss, and a key or table native code may not store
    /// through go to both helpers in order, each with its own guard and
    /// snapshot.
    fn hash_overwrite(
        &mut self,
        binding: u32,
        delete: &PendingDelete,
        (store_args, store_guard): (&[ValueId], GuardId),
        fuel: Value,
    ) -> Result<(), TraceCompileError> {
        let (slots, mask) = self.bound[binding as usize];
        let ty = self
            .ir
            .value_type(store_args[2])
            .expect("a verified trace types every operand");
        let stored = self.value(store_args[2]);
        let new_tag = match ty {
            TraceType::Float => NATIVE_VALUE_TAG_NUMBER,
            TraceType::Integer => NATIVE_VALUE_TAG_INTEGER,
            _ => NATIVE_VALUE_TAG_BOOLEAN,
        };
        let hit = self.builder.create_block();
        self.builder.append_block_param(hit, types::I64);
        let allowed = self.builder.create_block();
        let inline_key = self.builder.create_block();
        let keyed = self.builder.create_block();
        let nil_check = self.builder.create_block();
        let overwrite = self.builder.create_block();
        let revive = self.builder.create_block();
        let miss = self.builder.create_block();
        let slow = self.builder.create_block();
        let done = self.builder.create_block();
        self.builder.set_cold_block(slow);
        let data = self.probe_bucket(binding, store_args[1], (hit, miss, slow));

        // The same tests the delete's native code makes.
        self.builder.switch_to_block(hit);
        let bucket = self.builder.block_params(hit)[0];
        let flag = self.builder.ins().band_imm(mask, PROBE_NO_TOMBSTONES);
        let open = self.builder.ins().icmp_imm(IntCC::Equal, flag, 0);
        self.builder.ins().brif(open, allowed, &[], slow, &[]);
        self.builder.switch_to_block(allowed);
        let form = self.builder.ins().load(types::I8, MemFlags::new(), data, 0);
        let inline =
            self.builder
                .ins()
                .icmp_imm(IntCC::Equal, form, i64::from(NATIVE_STRING_INLINE_TAG));
        self.builder.ins().brif(inline, inline_key, &[], slow, &[]);
        self.builder.switch_to_block(inline_key);
        let first = self
            .builder
            .ins()
            .load(types::I16, MemFlags::new(), data, 2);
        let plain = self.builder.ins().icmp_imm(IntCC::NotEqual, first, 0x5f5f);
        self.builder.ins().brif(plain, keyed, &[], slow, &[]);

        self.builder.switch_to_block(keyed);
        let entry = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            bucket,
            i32::try_from(NATIVE_PROBE_SLOT_VALUE).expect("small"),
        );
        let tag = self
            .builder
            .ins()
            .load(types::I8, MemFlags::new(), entry, 0);
        let shifted = self.builder.ins().iadd_imm(tag, -1);
        let live = self
            .builder
            .ins()
            .icmp_imm(IntCC::UnsignedLessThan, shifted, 3);
        self.builder
            .ins()
            .brif(live, overwrite, &[], nil_check, &[]);

        self.builder.switch_to_block(overwrite);
        self.builder
            .ins()
            .store(MemFlags::new(), stored, entry, NATIVE_VALUE_PAYLOAD_OFFSET);
        let new_tag_value = self.builder.ins().iconst(types::I8, i64::from(new_tag));
        self.builder
            .ins()
            .store(MemFlags::new(), new_tag_value, entry, 0);
        self.builder.ins().jump(done, &[]);

        self.builder.switch_to_block(nil_check);
        let nil = self
            .builder
            .ins()
            .icmp_imm(IntCC::Equal, tag, i64::from(NATIVE_VALUE_TAG_NIL));
        self.builder.ins().brif(nil, revive, &[], slow, &[]);

        self.builder.switch_to_block(revive);
        let slots_mask = self.builder.ins().band_imm(mask, PROBE_NO_TOMBSTONES - 1);
        let header = self.builder.ins().iadd_imm(slots_mask, 1);
        let header = self.builder.ins().imul_imm(
            header,
            i64::try_from(NATIVE_PROBE_SLOT_SIZE).expect("small"),
        );
        let header = self.builder.ins().iadd(slots, header);
        let counter = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            header,
            i32::try_from(NATIVE_PROBE_SLOT_HASH).expect("small"),
        );
        let live_count = self
            .builder
            .ins()
            .load(types::I64, MemFlags::new(), counter, 0);
        self.builder
            .ins()
            .store(MemFlags::new(), stored, entry, NATIVE_VALUE_PAYLOAD_OFFSET);
        let new_tag_value = self.builder.ins().iconst(types::I8, i64::from(new_tag));
        self.builder
            .ins()
            .store(MemFlags::new(), new_tag_value, entry, 0);
        let more = self.builder.ins().iadd_imm(live_count, 1);
        self.builder.ins().store(MemFlags::new(), more, counter, 0);
        self.builder.ins().jump(done, &[]);

        self.builder.switch_to_block(miss);
        self.builder.ins().jump(slow, &[]);

        self.builder.switch_to_block(slow);
        self.call_helper_call(
            HelperId::DELETE_STRING,
            &delete.args,
            None,
            delete.guard,
            fuel,
            true,
        )?;
        self.call_helper_call(
            HelperId::SET_REVIVE,
            store_args,
            None,
            store_guard,
            fuel,
            true,
        )?;
        self.builder.ins().jump(done, &[]);

        self.builder.switch_to_block(done);
        Ok(())
    }

    /// What a native step does for the call of `helper`, if the trace gives
    /// it one ([`oximond::jit_ir::trace::TraceIr::boxed_steps`]) and the call
    /// is one it can take: a key of the type the profile's keys of the array
    /// part have (an integer key of the hash part gives the same), and a value
    /// that is a number or a boolean (a string or a table needs a lane).
    fn box_step_of(
        &mut self,
        helper: HelperId,
        args: &[ValueId],
        guard: GuardId,
    ) -> Option<BoxStep> {
        let steps = self.members[0].ir.boxed_steps?;
        let (key, value) = helper.next_box_shape()?;
        let wanted = if steps.integer {
            NextKind::Integer
        } else {
            NextKind::Float
        };
        if args.len() != 2
            || self.forced_guard == Some((self.member, guard.index()))
            || key.is_some_and(|key| key != wanted)
            || matches!(value, BoxedValue::Kind(NextKind::String | NextKind::Table))
        {
            return None;
        }
        // The hash part's snapshot is the storage as it stands: no operation
        // of the tree may move it.
        let hash = steps.cursor && !self.tree_moves_hash_storage();
        if steps.array.is_none() && !hash {
            return None;
        }
        Some(BoxStep {
            array: steps.array,
            hash,
            cursor_lane: steps.cursor,
            integer: steps.integer,
            key,
            value,
        })
    }

    /// Whether any member of the tree, the root or a side, has an operation
    /// that may move the hash part of a table (see
    /// [`TraceIr::moves_hash_storage`]).
    const fn tree_moves_hash_storage(&self) -> bool {
        self.moves_hash
    }

    /// The step of a boxed `next` loop (`helper`, see
    /// [`HelperId::next_box`]) with a native attempt before the call.
    ///
    /// The steps along the array part: the control lane holds nil (the start
    /// of a traversal) or a whole number inside the array whose slot holds a
    /// value, and a slot with a value follows. The attempt finds that slot
    /// by a scan over the nil slots, checks the kind of its value, and only
    /// then overwrites the lane with the slot's index (the lane holds a
    /// number the code can replace with no count to keep).
    ///
    /// The steps along the hash part: the table keeps a snapshot of its
    /// entries in traversal order and the control lane holds the key at the
    /// snapshot's position (`NativeCursor::in_sync`). The attempt moves the
    /// position over the deleted entries to the next one, checks the kinds of
    /// its key and value, and stores the key in the lane, natively for an
    /// integer over a number and through the VM for the other keys, which
    /// own a count or a handle.
    ///
    /// An attempt that gives up changes nothing, and the helper makes the
    /// step: at the end of a part, at a key of another type than the part it
    /// is in, and at a value of another kind than the call expects.
    #[allow(clippy::too_many_lines)]
    fn call_helper_stepping(
        &mut self,
        step: BoxStep,
        (helper, args): (HelperId, &[ValueId]),
        result: Option<ValueId>,
        guard: GuardId,
        fuel: Value,
    ) -> Result<Option<Value>, TraceCompileError> {
        let info = helper
            .info()
            .ok_or(TraceCompileError::NativeBackendRejected)?;
        let control = self.control.expect("entry set the control block");
        let merge = self.builder.create_block();
        for kind in info.results {
            let ty = match kind {
                HelperType::Float => types::F64,
                HelperType::Boolean => types::I8,
                _ => types::I64,
            };
            self.builder.append_block_param(merge, ty);
        }
        let slow = self.builder.create_block();
        let attempt = self.builder.create_block();
        let bank = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(mem::offset_of!(TraceControl, bank)),
        );
        self.builder.ins().brif(bank, attempt, &[], slow, &[]);
        self.builder.switch_to_block(attempt);
        let lane = self.value(args[1]);
        let at = self.builder.ins().ishl_imm(lane, 4);
        let lane_at = self.builder.ins().iadd(bank, at);
        let tag = self
            .builder
            .ins()
            .load(types::I8, MemFlags::new(), lane_at, 0);
        // Where a step the array part does not take goes.
        let hash_attempt = self.builder.create_block();
        let number_tag = if step.integer {
            NATIVE_VALUE_TAG_INTEGER
        } else {
            NATIVE_VALUE_TAG_NUMBER
        };
        let view_at = step.cursor_lane.then(|| {
            // The description of the hash part's snapshot: the address the
            // lane two after the control lane holds.
            let lane_two = self.builder.ins().iadd_imm(lane_at, 32);
            self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                lane_two,
                NATIVE_VALUE_PAYLOAD_OFFSET,
            )
        });
        if let Some(binding) = step.array {
            // A control lane that does not hold the key yet is in the hash
            // part.
            if let Some(view) = view_at {
                let pending = self.builder.ins().load(
                    types::I64,
                    MemFlags::new(),
                    view,
                    i32::try_from(mem::offset_of!(crate::runtime::NativeCursor, pending))
                        .expect("small offset"),
                );
                let clear = self.builder.create_block();
                self.builder
                    .ins()
                    .brif(pending, hash_attempt, &[], clear, &[]);
                self.builder.switch_to_block(clear);
            }
            let (base, length, length_float) = self.array_words(binding);
            let start = self.builder.create_block();
            let held = self.builder.create_block();
            let keyed = self.builder.create_block();
            let is_nil =
                self.builder
                    .ins()
                    .icmp_imm(IntCC::Equal, tag, i64::from(NATIVE_VALUE_TAG_NIL));
            self.builder.ins().brif(is_nil, start, &[], keyed, &[]);
            // A key of the array part: a whole number inside it.
            self.builder.switch_to_block(keyed);
            let is_number = self
                .builder
                .ins()
                .icmp_imm(IntCC::Equal, tag, i64::from(number_tag));
            let numbered = self.builder.create_block();
            self.builder
                .ins()
                .brif(is_number, numbered, &[], hash_attempt, &[]);
            self.builder.switch_to_block(numbered);
            let key = self.builder.ins().load(
                if step.integer { types::I64 } else { types::F64 },
                MemFlags::new(),
                lane_at,
                NATIVE_VALUE_PAYLOAD_OFFSET,
            );
            let index =
                self.checked_key_index(key, !step.integer, (length, length_float), hash_attempt);
            // The slot of the key holds a value.
            let scaled = self.builder.ins().ishl_imm(index, 4);
            let key_slot = self.builder.ins().iadd(base, scaled);
            let slot_tag = self
                .builder
                .ins()
                .load(types::I8, MemFlags::new(), key_slot, 0);
            let hole = self.builder.ins().icmp_imm(
                IntCC::Equal,
                slot_tag,
                i64::from(NATIVE_VALUE_TAG_NIL),
            );
            self.builder.ins().brif(hole, hash_attempt, &[], held, &[]);
            self.builder.switch_to_block(held);
            let next = self.builder.ins().iadd_imm(index, 1);
            let scan = self.builder.create_block();
            self.builder.append_block_param(scan, types::I64);
            self.builder.ins().jump(scan, &[BlockArg::Value(next)]);
            self.builder.switch_to_block(start);
            let zero = self.builder.ins().iconst(types::I64, 0);
            self.builder.ins().jump(scan, &[BlockArg::Value(zero)]);
            self.builder.switch_to_block(scan);
            let from = self.builder.block_params(scan)[0];
            let found = self.scan_array((from, None), (base, length), hash_attempt);
            self.builder.switch_to_block(found);
            let (index, found_slot, element) = {
                let params = self.builder.block_params(found);
                (params[0], params[1], params[2])
            };
            // The value is of the kind the call wants, and its place is free.
            let value_lane = self.expect_step_value(&step, element, lane_at, hash_attempt);
            // Every check passed: the stores, and the results.
            let number = self.builder.ins().iadd_imm(index, 1);
            let float_number =
                (!step.integer).then(|| self.builder.ins().fcvt_from_sint(types::F64, number));
            let number = float_number.unwrap_or(number);
            let stored_tag = self.builder.ins().iconst(types::I8, i64::from(number_tag));
            self.builder
                .ins()
                .store(MemFlags::new(), stored_tag, lane_at, 0);
            self.builder.ins().store(
                MemFlags::new(),
                number,
                lane_at,
                NATIVE_VALUE_PAYLOAD_OFFSET,
            );
            if let Some(view) = view_at {
                // The lane no longer holds a key of the hash part.
                let off = self.builder.ins().iconst(types::I64, 0);
                self.builder.ins().store(MemFlags::new(), off, view, 0);
            }
            let gave = self.step_results(&step, found_slot, number, value_lane, element);
            self.builder.ins().jump(merge, &gave);
        } else {
            self.builder.ins().jump(hash_attempt, &[]);
        }
        // The hash part.
        self.builder.switch_to_block(hash_attempt);
        match view_at.filter(|_| step.hash) {
            None => {
                self.builder.ins().jump(slow, &[]);
            }
            Some(view) => self.hash_attempt(&step, (lane_at, view), merge, slow),
        }
        // The helper's step.
        self.builder.switch_to_block(slow);
        let first = self.call_helper_call(helper, args, result, guard, fuel, false)?;
        let mut all: Vec<BlockArg> = Vec::with_capacity(info.results.len());
        if let Some(first) = first {
            all.push(BlockArg::Value(first));
            if let Some(rest) = result.and_then(|result| self.helper_results.get(&result)) {
                all.extend(rest.iter().map(|value| BlockArg::Value(*value)));
            }
        }
        debug_assert_eq!(all.len(), info.results.len());
        self.builder.ins().jump(merge, &all);
        self.builder.switch_to_block(merge);
        let params = self.builder.block_params(merge).to_vec();
        if let (Some(result), true) = (result, params.len() > 1) {
            self.helper_results.insert(result, params[1..].to_vec());
        }
        Ok(params.first().copied())
    }

    /// Checks that the value of the entry a native step found (its tag is
    /// `element`) is of the kind the call wants, and that the lane that keeps
    /// it, for a loop variable the body never reads, can take a copy. Leaves
    /// through `fail` otherwise, before anything is stored. Returns the
    /// address of that lane.
    fn expect_step_value(
        &mut self,
        step: &BoxStep,
        element: Value,
        lane_at: Value,
        fail: Block,
    ) -> Option<Value> {
        let expect = |this: &mut Self, tag: u8| {
            let right = this
                .builder
                .ins()
                .icmp_imm(IntCC::Equal, element, i64::from(tag));
            let proceed = this.builder.create_block();
            this.builder.ins().brif(right, proceed, &[], fail, &[]);
            this.builder.switch_to_block(proceed);
        };
        match step.value {
            BoxedValue::Kind(NextKind::Integer) => expect(self, NATIVE_VALUE_TAG_INTEGER),
            BoxedValue::Kind(NextKind::Float) => expect(self, NATIVE_VALUE_TAG_NUMBER),
            BoxedValue::Kind(_) => expect(self, NATIVE_VALUE_TAG_BOOLEAN),
            BoxedValue::Skipped => {}
            BoxedValue::Lane => {
                // The lane after the control lane takes a copy of a number or
                // a boolean, over a value that holds no count either.
                let scalar = self.builder.ins().icmp_imm(
                    IntCC::UnsignedLessThanOrEqual,
                    element,
                    i64::from(NATIVE_VALUE_TAG_NUMBER),
                );
                let next_lane = self.builder.ins().iadd_imm(lane_at, 16);
                let old = self
                    .builder
                    .ins()
                    .load(types::I8, MemFlags::new(), next_lane, 0);
                let free = self.builder.ins().icmp_imm(
                    IntCC::UnsignedLessThanOrEqual,
                    old,
                    i64::from(NATIVE_VALUE_TAG_NUMBER),
                );
                let both = self.builder.ins().band(scalar, free);
                let proceed = self.builder.create_block();
                self.builder.ins().brif(both, proceed, &[], fail, &[]);
                self.builder.switch_to_block(proceed);
                return Some(next_lane);
            }
        }
        None
    }

    /// The results of a native step that passed its checks: copies the value
    /// into its lane when the loop keeps it there (`value_lane`), and gives
    /// the key (`number`) and the value read from the slot at `slot` as the
    /// call asks.
    fn step_results(
        &mut self,
        step: &BoxStep,
        slot: Value,
        number: Value,
        value_lane: Option<Value>,
        element: Value,
    ) -> Vec<BlockArg> {
        if let Some(next_lane) = value_lane {
            let payload = self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                slot,
                NATIVE_VALUE_PAYLOAD_OFFSET,
            );
            self.builder
                .ins()
                .store(MemFlags::new(), element, next_lane, 0);
            self.builder.ins().store(
                MemFlags::new(),
                payload,
                next_lane,
                NATIVE_VALUE_PAYLOAD_OFFSET,
            );
        }
        let mut gave: Vec<BlockArg> = Vec::with_capacity(2);
        if step.key.is_some() {
            gave.push(BlockArg::Value(number));
        }
        if let BoxedValue::Kind(kind) = step.value {
            let ty = match kind {
                NextKind::Integer => types::I64,
                NextKind::Float => types::F64,
                _ => types::I8,
            };
            let loaded =
                self.builder
                    .ins()
                    .load(ty, MemFlags::new(), slot, NATIVE_VALUE_PAYLOAD_OFFSET);
            gave.push(BlockArg::Value(loaded));
        }
        gave
    }

    /// The native attempt of a step along the hash part (see
    /// [`Self::call_helper_stepping`]); `view` is the address of the
    /// snapshot's description. Falls through to `slow` when it gives up.
    fn hash_attempt(
        &mut self,
        step: &BoxStep,
        (lane_at, view): (Value, Value),
        merge: Block,
        slow: Block,
    ) {
        // The lane holds the key at the snapshot's position.
        let field = |offset: usize| i32::try_from(offset).expect("small offset");
        let in_sync = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            view,
            field(mem::offset_of!(crate::runtime::NativeCursor, in_sync)),
        );
        let go = self.builder.create_block();
        self.builder.ins().brif(in_sync, go, &[], slow, &[]);
        self.builder.switch_to_block(go);
        let slots = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            view,
            field(mem::offset_of!(crate::runtime::NativeCursor, slots)),
        );
        let len = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            view,
            field(mem::offset_of!(crate::runtime::NativeCursor, len)),
        );
        let position_at = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            view,
            field(mem::offset_of!(crate::runtime::NativeCursor, position)),
        );
        let position = self
            .builder
            .ins()
            .load(types::I64, MemFlags::new(), position_at, 0);
        let first = self.builder.ins().iadd_imm(position, 1);
        let found = self.scan_cursor((slots, len), first, slow);
        let (index, entry, element, value_at) = {
            let params = self.builder.block_params(found);
            (params[0], params[1], params[2], params[3])
        };
        let value_lane = self.expect_step_value(step, element, lane_at, slow);
        // The key: an integer, as the call wants it, or any key when the
        // loop does not read it.
        let key_bits = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            entry,
            i32::try_from(crate::runtime::NATIVE_CURSOR_SLOT_KEY_BITS).expect("small offset"),
        );
        if step.key.is_some() {
            let key_tag = self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                entry,
                i32::try_from(crate::runtime::NATIVE_CURSOR_SLOT_KEY_TAG).expect("small offset"),
            );
            let is_integer = self.builder.ins().icmp_imm(
                IntCC::Equal,
                key_tag,
                i64::from(NATIVE_VALUE_TAG_INTEGER),
            );
            let proceed = self.builder.create_block();
            self.builder.ins().brif(is_integer, proceed, &[], slow, &[]);
            self.builder.switch_to_block(proceed);
        }
        // Every check passed. The snapshot's position moves to the entry, and
        // the control lane does not hold its key until the run needs the lane
        // (the VM commits `pending` then: see `RunTables::flush_cursor`).
        self.builder
            .ins()
            .store(MemFlags::new(), index, position_at, 0);
        self.builder.ins().store(
            MemFlags::new(),
            entry,
            view,
            field(mem::offset_of!(crate::runtime::NativeCursor, pending)),
        );
        let number = if step.integer {
            key_bits
        } else {
            self.builder.ins().fcvt_from_sint(types::F64, key_bits)
        };
        let gave = self.step_results(step, value_at, number, value_lane, element);
        self.builder.ins().jump(merge, &gave);
    }

    /// The scan of [`Self::hash_attempt`]: from the snapshot's index `first`,
    /// the first entry (of `len`, from address `slots`) whose value is not
    /// deleted, or `slow` when there is none. Returns the block that takes
    /// the index, the address of the entry, the kind of its value and the
    /// address of its value as parameters, and leaves the builder in it.
    fn scan_cursor(&mut self, (slots, len): (Value, Value), first: Value, slow: Block) -> Block {
        let scan = self.builder.create_block();
        let probe = self.builder.create_block();
        let found = self.builder.create_block();
        self.builder.append_block_param(scan, types::I64);
        self.builder.append_block_param(found, types::I64);
        self.builder.append_block_param(found, types::I64);
        self.builder.append_block_param(found, types::I8);
        self.builder.append_block_param(found, types::I64);
        self.builder.ins().jump(scan, &[BlockArg::Value(first)]);
        self.builder.switch_to_block(scan);
        let index = self.builder.block_params(scan)[0];
        let inside = self.builder.ins().icmp(IntCC::UnsignedLessThan, index, len);
        self.builder.ins().brif(inside, probe, &[], slow, &[]);
        self.builder.switch_to_block(probe);
        let scaled = self.builder.ins().ishl_imm(index, 5);
        let entry = self.builder.ins().iadd(slots, scaled);
        let value_at = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            entry,
            i32::try_from(crate::runtime::NATIVE_CURSOR_SLOT_VALUE).expect("small offset"),
        );
        let element = self
            .builder
            .ins()
            .load(types::I8, MemFlags::new(), value_at, 0);
        let deleted =
            self.builder
                .ins()
                .icmp_imm(IntCC::Equal, element, i64::from(NATIVE_VALUE_TAG_NIL));
        let after = self.builder.ins().iadd_imm(index, 1);
        self.builder.ins().brif(
            deleted,
            scan,
            &[BlockArg::Value(after)],
            found,
            &[
                BlockArg::Value(index),
                BlockArg::Value(entry),
                BlockArg::Value(element),
                BlockArg::Value(value_at),
            ],
        );
        self.builder.switch_to_block(found);
        found
    }

    /// The zero-based index of a key known to be an integral number inside
    /// the array.
    fn trusted_index(&mut self, key: ValueId) -> Value {
        let value = self.value(key);
        if self.ir.value_type(key) == Some(TraceType::Float) {
            let magic = self.builder.ins().f64const(FLOAT_INTEGER_MAGIC);
            let shifted = self.builder.ins().fadd(value, magic);
            let bits = self
                .builder
                .ins()
                .bitcast(types::I64, MemFlags::new(), shifted);
            let magic_bits = FLOAT_INTEGER_MAGIC.to_bits().cast_signed();
            self.builder.ins().iadd_imm(bits, -(magic_bits + 1))
        } else {
            self.builder.ins().iadd_imm(value, -1)
        }
    }

    /// At entry: declines the run (as an input of another type does) unless
    /// the trusted key names a slot of its array that holds a value.
    fn check_trusted_key(&mut self) {
        let Some(trusted) = self.trusted_key else {
            return;
        };
        let (base, length, length_float) = self.array_words(trusted.binding);
        let key = self.value(trusted.init);
        let float = self.ir.value_type(trusted.init) == Some(TraceType::Float);
        let decline = self.builder.create_block();
        self.builder.set_cold_block(decline);
        let index = self.checked_key_index(key, float, (length, length_float), decline);
        let scaled = self.builder.ins().ishl_imm(index, 4);
        let slot = self.builder.ins().iadd(base, scaled);
        let tag = self.builder.ins().load(types::I8, MemFlags::new(), slot, 0);
        let empty = self
            .builder
            .ins()
            .icmp_imm(IntCC::Equal, tag, i64::from(NATIVE_VALUE_TAG_NIL));
        let held = self.builder.create_block();
        self.builder.ins().brif(empty, decline, &[], held, &[]);
        self.builder.switch_to_block(decline);
        let code = self.builder.ins().iconst(
            types::I64,
            -i64::try_from(trusted.input).expect("input index fits") - 1,
        );
        self.builder.ins().return_(&[code]);
        self.builder.switch_to_block(held);
    }

    /// The zero-based index a number `key` names inside the array of `length`
    /// slots (`length_float` as a float, for a float key), checked as
    /// [`Self::bound_index`] checks a key: a block `fail` is where a key that
    /// is not an integral number from 1 to the length goes.
    fn checked_key_index(
        &mut self,
        key: Value,
        float: bool,
        (length, length_float): (Value, Option<Value>),
        fail: Block,
    ) -> Value {
        let branch_unless = |this: &mut Self, condition: Value| {
            let next = this.builder.create_block();
            this.builder.ins().brif(condition, next, &[], fail, &[]);
            this.builder.switch_to_block(next);
        };
        if float {
            let length_float = length_float.expect("an array binding has its length as a float");
            let one = self.builder.ins().f64const(1.0);
            let at_least_one = self
                .builder
                .ins()
                .fcmp(FloatCC::GreaterThanOrEqual, key, one);
            branch_unless(self, at_least_one);
            let at_most_length =
                self.builder
                    .ins()
                    .fcmp(FloatCC::LessThanOrEqual, key, length_float);
            branch_unless(self, at_most_length);
            let magic = self.builder.ins().f64const(FLOAT_INTEGER_MAGIC);
            let shifted = self.builder.ins().fadd(key, magic);
            let restored = self.builder.ins().fsub(shifted, magic);
            let exact = self.builder.ins().fcmp(FloatCC::Equal, restored, key);
            branch_unless(self, exact);
            let bits = self
                .builder
                .ins()
                .bitcast(types::I64, MemFlags::new(), shifted);
            let magic_bits = FLOAT_INTEGER_MAGIC.to_bits().cast_signed();
            self.builder.ins().iadd_imm(bits, -(magic_bits + 1))
        } else {
            let index = self.builder.ins().iadd_imm(key, -1);
            let in_range = self
                .builder
                .ins()
                .icmp(IntCC::UnsignedLessThan, index, length);
            branch_unless(self, in_range);
            index
        }
    }

    /// The scan of [`Self::array_next`] and of the native array step: from
    /// the zero-based slot `from`, the first slot of the array (`base`
    /// address of slot 1, `length` slots) that holds a value, or `fail` when
    /// there is none. Returns the block that takes the index and the address
    /// of that slot as its parameters. The builder is left in a block the
    /// scan ends in; the caller switches to the block it returns. `after_slot`
    /// is the address of the slot before `from`, when the caller has it.
    fn scan_array(
        &mut self,
        (from, after_slot): (Value, Option<Value>),
        (base, length): (Value, Value),
        fail: Block,
    ) -> Block {
        let scan = self.builder.create_block();
        let probe = self.builder.create_block();
        let found = self.builder.create_block();
        self.builder.append_block_param(scan, types::I64);
        self.builder.append_block_param(found, types::I64);
        self.builder.append_block_param(found, types::I64);
        self.builder.append_block_param(found, types::I8);
        // The slot right after the key holds a value in nearly every step:
        // look at it before the loop over the holes.
        let first = self.builder.create_block();
        let inside = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, from, length);
        self.builder.ins().brif(inside, first, &[], fail, &[]);
        self.builder.switch_to_block(first);
        let slot = if let Some(before) = after_slot {
            self.builder.ins().iadd_imm(before, 16)
        } else {
            let scaled = self.builder.ins().ishl_imm(from, 4);
            self.builder.ins().iadd(base, scaled)
        };
        let tag = self.builder.ins().load(types::I8, MemFlags::new(), slot, 0);
        let hole = self
            .builder
            .ins()
            .icmp_imm(IntCC::Equal, tag, i64::from(NATIVE_VALUE_TAG_NIL));
        let after = self.builder.ins().iadd_imm(from, 1);
        self.builder.ins().brif(
            hole,
            scan,
            &[BlockArg::Value(after)],
            found,
            &[
                BlockArg::Value(from),
                BlockArg::Value(slot),
                BlockArg::Value(tag),
            ],
        );
        self.builder.switch_to_block(scan);
        let index = self.builder.block_params(scan)[0];
        let inside = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, index, length);
        self.builder.ins().brif(inside, probe, &[], fail, &[]);
        self.builder.switch_to_block(probe);
        let scaled = self.builder.ins().ishl_imm(index, 4);
        let slot = self.builder.ins().iadd(base, scaled);
        let tag = self.builder.ins().load(types::I8, MemFlags::new(), slot, 0);
        let hole = self
            .builder
            .ins()
            .icmp_imm(IntCC::Equal, tag, i64::from(NATIVE_VALUE_TAG_NIL));
        let after = self.builder.ins().iadd_imm(index, 1);
        self.builder.ins().brif(
            hole,
            scan,
            &[BlockArg::Value(after)],
            found,
            &[
                BlockArg::Value(index),
                BlockArg::Value(slot),
                BlockArg::Value(tag),
            ],
        );
        found
    }

    /// A call of a helper ([`TraceOp::CallHelper`]): a direct call of its
    /// `extern "C"` function with the arguments in registers, `(context,) out,
    /// args...`, where `out` is the control block's result words. The status
    /// it returns is tested only for a helper that can miss, and then a
    /// nonzero one leaves through the guard, the helper having changed
    /// nothing. The results are loaded right away, so a later
    /// [`TraceOp::HelperResult`] reads a register, not the control block.
    ///
    /// A `cold` call sits on a cold path: when a stub exists for its argument
    /// count it goes through that stub, which preserves every register, so the
    /// call does not make the allocator keep the loop's values on the stack.
    #[allow(clippy::too_many_lines)]
    fn call_helper_call(
        &mut self,
        helper: HelperId,
        args: &[ValueId],
        result: Option<ValueId>,
        guard: GuardId,
        fuel: Value,
        cold: bool,
    ) -> Result<Option<Value>, TraceCompileError> {
        let info = helper
            .info()
            .ok_or(TraceCompileError::NativeBackendRejected)?;
        // A helper nobody installed cannot be called: the trace stays
        // interpreted.
        let mut address = crate::jit::trace::helpers::helper_address(helper)
            .ok_or(TraceCompileError::NativeBackendRejected)?;
        // A constant pattern calls the helper's twin, which takes the
        // validated pattern instead of reading it from the argument.
        if helper == HelperId::GMATCH_NEW {
            self.note_gmatch(args, result);
        }
        let prepared = self.prepared_pattern(helper, args);
        if let Some((twin, _)) = prepared {
            address = twin;
        }
        let control = self.control.expect("entry set the control block");
        let mut signature = Signature::new(self.builder.func.signature.call_conv);
        let mut arguments = Vec::with_capacity(args.len() + 2);
        if info.context {
            signature.params.push(AbiParam::new(types::I64));
            arguments.push(self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                control,
                offset(mem::offset_of!(TraceControl, host_context)),
            ));
        }
        signature.params.push(AbiParam::new(types::I64));
        let out_at = mem::offset_of!(TraceControl, helper_out);
        arguments.push(
            self.builder
                .ins()
                .iadd_imm(control, i64::from(offset(out_at))),
        );
        if info.effects.takes_words() {
            // The helper rewrites the words of the bindings it moves.
            signature.params.push(AbiParam::new(types::I64));
            arguments.push(match self.words {
                Some(words) => words,
                None => self.builder.ins().iconst(types::I64, 0),
            });
        }
        if helper.is_vm_call() {
            // The helper writes the call's `Before` state back to the frame
            // before it calls: native code spills it to the exit area first.
            let exit = self.plan.members[self.member].first + guard.index();
            let area = self.spill_exit(exit);
            signature.params.push(AbiParam::new(types::I64));
            signature.params.push(AbiParam::new(types::I64));
            arguments.push(area);
            arguments.push(
                self.builder
                    .ins()
                    .iconst(types::I64, i64::try_from(exit).expect("exit index fits")),
            );
        }
        if info.required < info.params.len() {
            // A variadic helper is told how many arguments the call gave.
            signature.params.push(AbiParam::new(types::I64));
            let mut count = i64::try_from(args.len()).expect("few arguments");
            if helper == HelperId::TABLE_NEW
                && result.is_some_and(|result| self.prefills_array(result, args))
            {
                count |= TABLE_NEW_PREFILL;
            }
            arguments.push(self.builder.ins().iconst(types::I64, count));
        }
        if let Some((_, object)) = prepared {
            signature.params.push(AbiParam::new(types::I64));
            arguments.push(
                self.builder
                    .ins()
                    .iconst(types::I64, i64::try_from(object).expect("address fits i64")),
            );
        }
        let mut nil_word = None;
        for (position, param) in info.params.iter().enumerate() {
            let Some(arg) = args.get(position) else {
                // A parameter the call leaves out is nil.
                debug_assert_eq!(*param, HelperType::Value);
                // Both words of nil are zero: one constant serves every
                // parameter left out, so it takes one register.
                let zero =
                    *nil_word.get_or_insert_with(|| self.builder.ins().iconst(types::I64, 0));
                for _ in 0..2 {
                    signature.params.push(AbiParam::new(types::I64));
                    arguments.push(zero);
                }
                continue;
            };
            let value = self.value(*arg);
            match param {
                HelperType::Value => {
                    let ty = self
                        .ir
                        .value_type(*arg)
                        .expect("a verified trace types every operand");
                    let (tag, bits) = if ty == TraceType::StringRef {
                        (
                            self.builder
                                .ins()
                                .iconst(types::I64, TRACE_TAG_BORROWED_STRING),
                            value,
                        )
                    } else {
                        self.scalar_word(ty, value)
                    };
                    signature.params.push(AbiParam::new(types::I64));
                    signature.params.push(AbiParam::new(types::I64));
                    arguments.push(tag);
                    arguments.push(bits);
                }
                HelperType::Float => {
                    signature.params.push(AbiParam::new(types::F64));
                    arguments.push(value);
                }
                HelperType::Boolean => {
                    signature.params.push(AbiParam::new(types::I64));
                    arguments.push(self.builder.ins().uextend(types::I64, value));
                }
                _ => {
                    signature.params.push(AbiParam::new(types::I64));
                    arguments.push(value);
                }
            }
        }
        signature.returns.push(AbiParam::new(types::I64));
        let callee = self.native_address(address);
        let fills_in_place = helper == HelperId::TABLE_NEW
            && result.is_some_and(|result| {
                self.fills_in_place(result) || self.prefills_array(result, args)
            });
        if fills_in_place {
            // The second word is the address of the new table's value slots,
            // which only a helper that has them writes.
            let none = self.builder.ins().iconst(types::I64, 0);
            self.builder.ins().store(
                MemFlags::new(),
                none,
                control,
                offset(out_at + mem::size_of::<i64>()),
            );
        }
        let stub = (cold && prepared.is_none())
            .then(|| self.helper_stubs.get(&signature.params.len()).copied())
            .flatten()
            .filter(|_| {
                signature
                    .params
                    .iter()
                    .all(|param| param.value_type == types::I64)
            });
        let status = if let Some(stub) = stub {
            debug_assert_eq!(
                helper_stub_arity(helper, args.len()),
                Some(signature.params.len()),
                "the stub of {helper:?} takes the arguments the call builds"
            );
            let mut stub_arguments = Vec::with_capacity(arguments.len() + 2);
            stub_arguments.push(callee);
            stub_arguments.push(control);
            stub_arguments.extend_from_slice(&arguments);
            self.builder.ins().call(stub, &stub_arguments);
            self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                control,
                offset(mem::offset_of!(TraceControl, helper_status)),
            )
        } else {
            let signature = self.builder.import_signature(signature);
            let call = self
                .builder
                .ins()
                .call_indirect(signature, callee, &arguments);
            self.builder.inst_results(call)[0]
        };
        if info.effects.contains(HelperEffects::MAY_EXIT) {
            let answered = if self.forced_guard == Some((self.member, guard.index())) {
                self.builder.ins().iconst(types::I8, 0)
            } else {
                self.builder.ins().icmp_imm(IntCC::Equal, status, 0)
            };
            let pass = self.builder.create_block();
            let fail = self.builder.create_block();
            self.builder.set_cold_block(fail);
            self.builder.ins().brif(answered, pass, &[], fail, &[]);
            self.builder.switch_to_block(fail);
            self.leave(guard, fuel)?;
            self.builder.switch_to_block(pass);
        }
        if info.effects.moves_storage() {
            // The call may have grown any table: array words are read again.
            self.array_cache.fill(None);
            self.lane_array_cache.clear();
        }
        let mut loaded = Vec::with_capacity(info.results.len());
        for (position, kind) in info.results.iter().enumerate() {
            let at = offset(out_at + position * mem::size_of::<i64>());
            let word = match kind {
                HelperType::Float => {
                    self.builder
                        .ins()
                        .load(types::F64, MemFlags::new(), control, at)
                }
                HelperType::Boolean => {
                    let bits = self
                        .builder
                        .ins()
                        .load(types::I64, MemFlags::new(), control, at);
                    self.builder.ins().ireduce(types::I8, bits)
                }
                _ => self
                    .builder
                    .ins()
                    .load(types::I64, MemFlags::new(), control, at),
            };
            loaded.push(word);
        }
        // The table a handler check gives is as much a table met at run time,
        // and so is one a lookup call gives (`TABLE_GET_TABLE`, every field
        // read of a rowless recording) in a tree with lane array views,
        // which read the table's address out of the lane's row. The rows are
        // not cleared between runs, and they are shared by every trace of the
        // thread, so an unfilled row holds what another run left there.
        // The same goes for a closure a lookup call gives (`TABLE_GET_CLOSURE`
        // of `p.test` before a call guarded on the prototype) in a tree that
        // reads closure rows: the prototype word of a stale row can be the
        // pinned closure's, and the call then ran another prototype's code.
        let fills_table = info.effects.contains(HelperEffects::REWRITES_ARRAYS)
            || matches!(
                helper,
                HelperId::INDEX_HANDLER | HelperId::NEWINDEX_HANDLER | HelperId::CELL_GET_TABLE
            )
            || self.reads_table_addresses()
            // A table a lookup call gives in a loop tree with row fields:
            // the tree may read its fields (`starts[hash].starts`), and its
            // row holds nothing or another run's words until it is filled,
            // as rows are not cleared between runs. Only the lookup: a table
            // a constructor or a cell read gives keeps the row it had. The
            // MIR backend makes the same fill (`fills_table` in
            // `mir/lower/calls.rs`).
            || (helper == HelperId::TABLE_GET_TABLE
                && self.function.is_none()
                && !self.members[0].ir.fields.is_empty());
        let fills_closure = self.row_words > 0;
        // Every result that is a table or a closure, not only the first: the
        // value of a `next` step is the second result, and a loop over a
        // table of tables reads its elements through the value's row
        // (`box[5]` in `for _, box in pairs (fixed)`).
        for (kind, &lane) in info.results.iter().zip(&loaded) {
            let fills = match kind {
                HelperType::Table => fills_table,
                HelperType::Closure => fills_closure,
                _ => false,
            };
            if !fills {
                continue;
            }
            // The table the helper took out of an array (or walked to) has a
            // lane, and the fields of the trace read it through the lane's
            // row.
            let dispatch = self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                control,
                offset(mem::offset_of!(TraceControl, tables)),
            );
            self.builder.ins().call(self.ensure_row, &[dispatch, lane]);
        }
        // The row of a table whose metatable the call changed names the old
        // one in the metatable word (see [`TraceOp::MetaGuard`]): make it
        // again, the complement of the lane asking for that.
        if info.effects.contains(HelperEffects::WRITES_METATABLE)
            && self.uses_meta_rows()
            && let Some(table) = args.first()
        {
            let lane = self.value(*table);
            let lane = self.builder.ins().bnot(lane);
            let dispatch = self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                control,
                offset(mem::offset_of!(TraceControl, tables)),
            );
            self.builder.ins().call(self.ensure_row, &[dispatch, lane]);
        }
        let first = loaded.first().copied();
        if let (Some(result), true) = (result, loaded.len() > 1) {
            self.helper_results.insert(result, loaded[1..].to_vec());
        }
        if let (true, Some(result)) = (fills_in_place, result) {
            let slots = self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                control,
                offset(out_at + mem::size_of::<i64>()),
            );
            self.made_tables.insert(
                result,
                MadeTable {
                    slots,
                    made: super::made_table::MadeShape {
                        shape: self.made_table_shape(args),
                        prefilled: self.prefills_array(result, args),
                        array: match self.constants[args[0].index()] {
                            Some(TraceScalar::Integer(length)) => length,
                            _ => 0,
                        },
                    },
                    filled: Vec::new(),
                    moved: false,
                },
            );
        }
        Ok(first)
    }

    /// Whether the trace lists the metatable word of a row (see
    /// [`TraceOp::MetaGuard`]).
    fn uses_meta_rows(&self) -> bool {
        self.members[0]
            .ir
            .fields
            .iter()
            .any(|field| field.key.as_ref() == META_FIELD_KEY)
    }

    /// Whether some [`HelperId::TABLE_INIT`] of the current member names the
    /// table `made` by a root string key, which the code can fill in place
    /// (see [`Self::table_init_in_place`]).
    fn fills_in_place(&self, made: ValueId) -> bool {
        super::made_table::fills_in_place(self.ir, made)
    }

    /// Whether the code fills the table `made`, which a [`HelperId::TABLE_NEW`]
    /// call with arguments `args` makes, as a short array in place: the call
    /// names a constant length of [`MAX_PREFILLED_ARRAY`] or fewer and no
    /// key, and the member stores a number or boolean under each integer key
    /// from one to the length, once, by calls of [`HelperId::TABLE_INIT`]. The
    /// helper then makes the slots (nil, counted as present), and each store
    /// overwrites one, so the table is the dense list the constructor makes.
    fn prefills_array(&self, made: ValueId, args: &[ValueId]) -> bool {
        super::made_table::prefills_array(self.ir, made, args, constant_integer)
    }

    /// The shape a [`HelperId::TABLE_NEW`] call with arguments `args` names,
    /// when every key is a root string of the trace.
    fn made_table_shape(
        &self,
        args: &[ValueId],
    ) -> Option<std::sync::Arc<crate::runtime::TableShape>> {
        super::made_table::made_table_shape(self.ir, args, &self.constants)
    }

    fn host_call(
        &mut self,
        function: HostFunction,
        args: &[ValueId],
        result: TraceType,
        guard: GuardId,
        fuel: Value,
    ) -> Result<Value, TraceCompileError> {
        // The maximum or minimum of two integers needs no call and cannot
        // fail.
        if matches!(function, HostFunction::MathMax | HostFunction::MathMin) {
            let (left, right) = (self.value(args[0]), self.value(args[1]));
            return Ok(if function == HostFunction::MathMax {
                self.builder.ins().smax(left, right)
            } else {
                self.builder.ins().smin(left, right)
            });
        }
        let control = self.control.expect("entry set the control block");
        // The length of a string held by lane is a word the VM keeps for
        // every lane. A root string (a lane from the root base on) is not in
        // that table: the constant goes to the VM, and so does an operand
        // that may hold one (a loop-carried constant), by a check at run
        // time.
        if function == HostFunction::StringLength
            && self.ir.value_type(args[0]) == Some(TraceType::String)
            && !matches!(
                self.constants[args[0].index()],
                Some(TraceScalar::String(_))
            )
        {
            let lane = self.value(args[0]);
            let inline = self.builder.create_block();
            let root = self.builder.create_block();
            let merge = self.builder.create_block();
            let merged = if result == TraceType::Float {
                types::F64
            } else {
                types::I64
            };
            self.builder.append_block_param(merge, merged);
            let is_root = self.builder.ins().icmp_imm(
                IntCC::UnsignedGreaterThanOrEqual,
                lane,
                ROOT_STRING_LANE_BASE.cast_signed(),
            );
            self.builder.set_cold_block(root);
            self.builder.ins().brif(is_root, root, &[], inline, &[]);
            self.builder.switch_to_block(inline);
            let table = self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                control,
                offset(mem::offset_of!(TraceControl, lengths)),
            );
            let at = self.builder.ins().ishl_imm(lane, 2);
            let address = self.builder.ins().iadd(table, at);
            let length = self.builder.ins().uload32(MemFlags::new(), address, 0);
            let length = if result == TraceType::Float {
                self.builder.ins().fcvt_from_uint(types::F64, length)
            } else {
                length
            };
            self.builder.ins().jump(merge, &[BlockArg::Value(length)]);
            self.builder.switch_to_block(root);
            let length = self.host_call_in_vm(function, args, result, guard, fuel)?;
            self.builder.ins().jump(merge, &[BlockArg::Value(length)]);
            self.builder.switch_to_block(merge);
            return Ok(self.builder.block_params(merge)[0]);
        }
        self.host_call_in_vm(function, args, result, guard, fuel)
    }

    /// A host call that the VM answers: the arguments as tagged words, the
    /// call, and a guard on the tag of the answer.
    fn host_call_in_vm(
        &mut self,
        function: HostFunction,
        args: &[ValueId],
        result: TraceType,
        guard: GuardId,
        fuel: Value,
    ) -> Result<Value, TraceCompileError> {
        let control = self.control.expect("entry set the control block");
        let slot_size = mem::size_of::<TraceInputSlot>();
        let args_at = mem::offset_of!(TraceControl, host_args);
        for (position, arg) in args.iter().enumerate() {
            let ty = self
                .ir
                .value_type(*arg)
                .expect("a verified trace types every operand");
            let value = self.value(*arg);
            let (tag, bits) = match ty {
                TraceType::StringRef => (
                    self.builder
                        .ins()
                        .iconst(types::I64, TRACE_TAG_BORROWED_STRING),
                    value,
                ),
                TraceType::String | TraceType::Table | TraceType::Integer => {
                    let tag = entry_tag(ty).expect("an opaque lane has an entry tag");
                    (self.builder.ins().iconst(types::I64, tag), value)
                }
                _ => self.scalar_word(ty, value),
            };
            let at = args_at + position * slot_size;
            self.builder.ins().store(
                MemFlags::new(),
                tag,
                control,
                offset(at + mem::offset_of!(TraceInputSlot, tag)),
            );
            self.builder.ins().store(
                MemFlags::new(),
                bits,
                control,
                offset(at + mem::offset_of!(TraceInputSlot, bits)),
            );
        }
        let args_address = self
            .builder
            .ins()
            .iadd_imm(control, i64::from(offset(args_at)));
        let out = self.builder.ins().iadd_imm(
            control,
            i64::from(offset(mem::offset_of!(TraceControl, host_result))),
        );
        let function_word = self.builder.ins().iconst(types::I64, function.code());
        let count = self.builder.ins().iconst(
            types::I64,
            i64::try_from(args.len()).expect("few arguments"),
        );
        self.call_host(control, args_address, out, function_word, count);
        let tag = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(
                mem::offset_of!(TraceControl, host_result) + mem::offset_of!(TraceInputSlot, tag),
            ),
        );
        let wanted = entry_tag(result).expect("a host call yields a scalar or a string");
        let matches = if self.forced_guard == Some((self.member, guard.index())) {
            self.builder.ins().iconst(types::I8, 0)
        } else {
            self.builder.ins().icmp_imm(IntCC::Equal, tag, wanted)
        };
        let pass = self.builder.create_block();
        let fail = self.builder.create_block();
        self.builder.set_cold_block(fail);
        self.builder.ins().brif(matches, pass, &[], fail, &[]);
        self.builder.switch_to_block(fail);
        self.leave(guard, fuel)?;
        self.builder.switch_to_block(pass);
        if result == TraceType::Nil {
            return Ok(self.builder.ins().iconst(types::I8, 0));
        }
        let bits = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(
                mem::offset_of!(TraceControl, host_result) + mem::offset_of!(TraceInputSlot, bits),
            ),
        );
        // The call may have moved any array of the tree: a host call reads
        // them, never writes, so the cached words stay valid.
        Ok(match result {
            TraceType::Float => self
                .builder
                .ins()
                .bitcast(types::F64, MemFlags::new(), bits),
            TraceType::Boolean => self.builder.ins().ireduce(types::I8, bits),
            _ => bits,
        })
    }

    /// [`TraceOp::ReleaseStrings`]: tells the VM which string lanes the next
    /// iteration can still name, through the host call helper.
    fn release_strings(&mut self, keep: &[ValueId], fuel: Value) {
        // When the strings made since the last release weigh enough: keeping
        // a few iterations' strings costs less than a call at every
        // loop-back, and the VM drops them all at the next release or when
        // the run ends.
        let _ = fuel;
        let call = self.builder.create_block();
        let after = self.builder.create_block();
        let control = self.control.expect("entry set the control block");
        let table = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(mem::offset_of!(TraceControl, lengths)),
        );
        let pending = self.builder.ins().uload32(
            MemFlags::new(),
            table,
            offset(TRACE_PENDING_STRINGS * mem::size_of::<u32>()),
        );
        let due = self
            .builder
            .ins()
            .icmp_imm(IntCC::UnsignedGreaterThan, pending, RELEASE_WEIGHT);
        self.builder.ins().brif(due, call, &[], after, &[]);
        self.builder.set_cold_block(call);
        self.builder.switch_to_block(call);
        let slot_size = mem::size_of::<TraceInputSlot>();
        let args_at = mem::offset_of!(TraceControl, host_args);
        for (position, lane) in keep.iter().enumerate() {
            let lane = self.value(*lane);
            let tag = self.builder.ins().iconst(types::I64, TRACE_TAG_STRING);
            let at = args_at + position * slot_size;
            self.builder.ins().store(
                MemFlags::new(),
                tag,
                control,
                offset(at + mem::offset_of!(TraceInputSlot, tag)),
            );
            self.builder.ins().store(
                MemFlags::new(),
                lane,
                control,
                offset(at + mem::offset_of!(TraceInputSlot, bits)),
            );
        }
        let args_address = self
            .builder
            .ins()
            .iadd_imm(control, i64::from(offset(args_at)));
        let out = self.builder.ins().iadd_imm(
            control,
            i64::from(offset(mem::offset_of!(TraceControl, host_result))),
        );
        let function = self
            .builder
            .ins()
            .iconst(types::I64, HostFunction::ReleaseStrings.code());
        let count = self
            .builder
            .ins()
            .iconst(types::I64, i64::try_from(keep.len()).expect("few lanes"));
        self.call_host(control, args_address, out, function, count);
        self.builder.ins().jump(after, &[]);
        self.builder.switch_to_block(after);
    }

    /// A scalar as the entry tag and bits a [`TraceInputSlot`] holds.
    fn scalar_word(&mut self, ty: TraceType, value: Value) -> (Value, Value) {
        let tag = entry_tag(ty).expect("a stored scalar has an entry tag");
        let tag = self.builder.ins().iconst(types::I64, tag);
        let bits = match ty {
            TraceType::Float => self
                .builder
                .ins()
                .bitcast(types::I64, MemFlags::new(), value),
            TraceType::Boolean => self.builder.ins().uextend(types::I64, value),
            TraceType::Nil => self.builder.ins().iconst(types::I64, 0),
            _ => value,
        };
        (tag, bits)
    }

    /// The lane of the table in the array slot at address `slot`, from the
    /// VM's dispatcher, which also fills the table's row. Leaves through the
    /// guard's exit when it has none to give.
    fn admit_table_lane(
        &mut self,
        slot: Value,
        guard: GuardId,
        fuel: Value,
        is_table: bool,
    ) -> Value {
        let control = self.control.expect("entry set the control block");
        let dispatch = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(mem::offset_of!(TraceControl, tables)),
        );
        // The table the slot holds may be one the run has a lane and a row
        // for already: the index of the run's lanes names it at the place of
        // its hash. Only the first place is looked at here, for a loop that
        // reads the same few tables again and again; the dispatcher answers
        // any other case. A function trace gives every table a lane of its
        // own and has no index.
        // Closures and threads count their slots and generations in arenas of
        // their own, so their handle words can equal a table's identity: only
        // a table load may probe the index.
        let lane = if self.lane_probe && is_table {
            self.admit_table_probe(control, dispatch, slot)
        } else {
            let call = self.builder.ins().call(self.admit_table, &[dispatch, slot]);
            self.builder.inst_results(call)[0]
        };
        let admitted = self
            .builder
            .ins()
            .icmp_imm(IntCC::SignedGreaterThanOrEqual, lane, 0);
        let pass = self.builder.create_block();
        let fail = self.builder.create_block();
        self.builder.set_cold_block(fail);
        self.builder.ins().brif(admitted, pass, &[], fail, &[]);
        self.builder.switch_to_block(fail);
        let exit = self.plan.members[self.member].first + guard.index();
        self.emit_exit(exit, fuel);
        self.builder.switch_to_block(pass);
        lane
    }

    /// The lane of the table in the array slot at `slot`: from the first
    /// place of the run's lane index when it names the table and says its row
    /// is filled, else from the dispatcher (see [`TraceTables::lane_index`]).
    fn admit_table_probe(&mut self, control: Value, dispatch: Value, slot: Value) -> Value {
        let fallback = self.builder.create_block();
        let fast = self.builder.create_block();
        let join = self.builder.create_block();
        self.builder.append_block_param(join, types::I64);
        let index = self.builder.ins().load(
            types::I64,
            MemFlags::trusted().with_readonly(),
            control,
            offset(mem::offset_of!(TraceControl, lane_index)),
        );
        let probe = self.builder.create_block();
        self.builder.ins().brif(index, probe, &[], fallback, &[]);
        self.builder.switch_to_block(probe);
        // The word of the table's handle: its slot, then its generation.
        let handle = self
            .builder
            .ins()
            .load(types::I64, MemFlags::new(), slot, offset(8));
        let high = self.builder.ins().ushr_imm(handle, 32);
        let mixed = self.builder.ins().bxor(handle, high);
        let hashed = self
            .builder
            .ins()
            .imul_imm(mixed, 0x9E37_79B9_7F4A_7C15_u64.cast_signed());
        let hashed = self.builder.ins().ushr_imm(hashed, 32);
        let place = self.builder.ins().band_imm(
            hashed,
            i64::try_from(super::exit::TRACE_LANE_INDEX_PLACES - 1).expect("places fit"),
        );
        let entry = self.builder.ins().ishl_imm(place, 4);
        let entry = self.builder.ins().iadd(index, entry);
        let identity = self
            .builder
            .ins()
            .load(types::I64, MemFlags::new(), entry, offset(0));
        let ready = self
            .builder
            .ins()
            .load(types::I32, MemFlags::new(), entry, offset(12));
        let ready = self.builder.ins().uextend(types::I64, ready);
        let wanted = self.builder.ins().load(
            types::I64,
            MemFlags::trusted().with_readonly(),
            control,
            offset(mem::offset_of!(TraceControl, lane_ready)),
        );
        let same = self.builder.ins().icmp(IntCC::Equal, identity, handle);
        let current = self.builder.ins().icmp(IntCC::Equal, ready, wanted);
        let hit = self.builder.ins().band(same, current);
        self.builder.ins().brif(hit, fast, &[], fallback, &[]);
        self.builder.switch_to_block(fast);
        let lane = self
            .builder
            .ins()
            .load(types::I32, MemFlags::new(), entry, offset(8));
        let lane = self.builder.ins().uextend(types::I64, lane);
        self.builder.ins().jump(join, &[lane.into()]);
        self.builder.set_cold_block(fallback);
        self.builder.switch_to_block(fallback);
        let lane = if let Some(stub) = self.admit_table_stub {
            self.builder.ins().call(stub, &[control, slot]);
            self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                control,
                offset(mem::offset_of!(TraceControl, admit_lane)),
            )
        } else {
            let call = self.builder.ins().call(self.admit_table, &[dispatch, slot]);
            self.builder.inst_results(call)[0]
        };
        self.builder.ins().jump(join, &[lane.into()]);
        self.builder.switch_to_block(join);
        self.builder.block_params(join)[0]
    }

    /// The address of the row of the table in lane `lane`: the run's rows
    /// plus the lane's offset. The code of a member reaches many fields of a
    /// few tables, and computes each row's address once, where it first
    /// needs it. The address is good wherever that code dominates, which the
    /// lowering keeps true by forgetting the addresses where it leaves the
    /// straight line of the member (a side, a second version of the loop).
    /// A function trace's activation reads the rows where it needs them, so
    /// it keeps none.
    fn row_address(&mut self, lane: Value) -> Value {
        let cached = self.function.is_none();
        if cached && let Some((_, row)) = self.row_addresses.iter().find(|(held, _)| *held == lane)
        {
            return *row;
        }
        let rows = if let Some(rows) = self.rows {
            rows
        } else {
            let control = self.control.expect("entry set the control block");
            self.builder.ins().load(
                types::I64,
                MemFlags::trusted().with_readonly(),
                control,
                offset(mem::offset_of!(TraceControl, rows)),
            )
        };
        let bytes = self.builder.ins().imul_imm(
            lane,
            i64::try_from(self.row_stride * mem::size_of::<usize>()).expect("row fits"),
        );
        let row = self.builder.ins().iadd(rows, bytes);
        if cached {
            self.row_addresses.push((lane, row));
        }
        row
    }

    /// The address of the run's rows.
    fn rows_base(&mut self) -> Value {
        if let Some(rows) = self.rows {
            return rows;
        }
        let control = self.control.expect("entry set the control block");
        self.builder.ins().load(
            types::I64,
            MemFlags::trusted().with_readonly(),
            control,
            offset(mem::offset_of!(TraceControl, rows)),
        )
    }

    /// Word `word` of the row of the closure in lane `lane`
    /// ([`super::exit::CLOSURE_ROW_WORDS`]).
    fn closure_row_word(&mut self, lane: Value, word: usize) -> Value {
        let rows = self.rows_base();
        let row = self.builder.ins().imul_imm(
            lane,
            i64::try_from(self.row_stride * mem::size_of::<usize>()).expect("row fits"),
        );
        let row = self.builder.ins().iadd(rows, row);
        self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            row,
            offset(word * mem::size_of::<usize>()),
        )
    }

    /// Leaves through `guard` unless `condition` holds.
    fn guard_condition(
        &mut self,
        condition: Value,
        guard: GuardId,
        fuel: Value,
    ) -> Result<(), TraceCompileError> {
        let pass = self.builder.create_block();
        let fail = self.builder.create_block();
        self.builder.set_cold_block(fail);
        if self.forced_guard == Some((self.member, guard.index())) {
            let never = self.builder.ins().iconst(types::I8, 0);
            self.builder.ins().brif(never, pass, &[], fail, &[]);
        } else {
            self.builder.ins().brif(condition, pass, &[], fail, &[]);
        }
        self.builder.switch_to_block(fail);
        self.leave(guard, fuel)?;
        self.builder.switch_to_block(pass);
        Ok(())
    }

    /// A call of a helper that reads or writes the cell of a callee, or asks
    /// whether two closures run one prototype, done through the closures'
    /// rows with no call (see [`closure_cell_use`]). The guard leaves where
    /// the helper would miss: the closure has no such cell, or the cell holds
    /// a value of another type.
    fn closure_cell(
        &mut self,
        cell: ClosureCellUse,
        args: &[ValueId],
        guard: GuardId,
        fuel: Value,
    ) -> Result<Option<Value>, TraceCompileError> {
        let lane = self.value(args[0]);
        let (index, ty) = match cell {
            ClosureCellUse::Proto => {
                let pinned = self.value(args[1]);
                let live = self.closure_row_word(lane, 0);
                let known = self.closure_row_word(pinned, 0);
                let equal = self.builder.ins().icmp(IntCC::Equal, live, known);
                let present = self.builder.ins().icmp_imm(IntCC::NotEqual, live, 0);
                return Ok(Some(self.builder.ins().band(equal, present)));
            }
            ClosureCellUse::GetTable { index } => {
                return self.closure_table_cell(lane, index, guard, fuel).map(Some);
            }
            ClosureCellUse::Get { index, ty } | ClosureCellUse::Set { index, ty } => (index, ty),
        };
        let slot = self.closure_row_word(lane, 1 + index);
        let present = self.builder.ins().icmp_imm(IntCC::NotEqual, slot, 0);
        self.guard_condition(present, guard, fuel)?;
        let wanted = match ty {
            TraceType::Float => NATIVE_VALUE_TAG_NUMBER,
            TraceType::Integer => NATIVE_VALUE_TAG_INTEGER,
            _ => NATIVE_VALUE_TAG_BOOLEAN,
        };
        let tag = self.builder.ins().load(types::I8, MemFlags::new(), slot, 0);
        let same = self
            .builder
            .ins()
            .icmp_imm(IntCC::Equal, tag, i64::from(wanted));
        self.guard_condition(same, guard, fuel)?;
        Ok(if let ClosureCellUse::Get { .. } = cell {
            Some(self.builder.ins().load(
                clif_type(ty),
                MemFlags::new(),
                slot,
                NATIVE_VALUE_PAYLOAD_OFFSET,
            ))
        } else {
            let stored = self.value(args[2]);
            self.builder
                .ins()
                .store(MemFlags::new(), stored, slot, NATIVE_VALUE_PAYLOAD_OFFSET);
            None
        })
    }

    /// A read of upvalue `index` that holds a table, with no call: the lane
    /// the closure's row names for the cell, after a check that the cell
    /// still holds the table the lane's bank entry does. The guard leaves for
    /// a closure with no such cell, a cell that holds no table, a cell the
    /// row has no lane for, or a cell that changed since the row was filled.
    fn closure_table_cell(
        &mut self,
        closure: Value,
        index: usize,
        guard: GuardId,
        fuel: Value,
    ) -> Result<Value, TraceCompileError> {
        let control = self.control.expect("entry set the control block");
        let slot = self.closure_row_word(closure, 1 + index);
        let named = self.closure_row_word(closure, 1 + CLOSURE_ROW_CELLS + index);
        let bank = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(mem::offset_of!(TraceControl, bank)),
        );
        let present = self.builder.ins().icmp_imm(IntCC::NotEqual, slot, 0);
        let has_lane = self.builder.ins().icmp_imm(IntCC::NotEqual, named, 0);
        let banked = self.builder.ins().icmp_imm(IntCC::NotEqual, bank, 0);
        let ready = self.builder.ins().band(present, has_lane);
        let ready = self.builder.ins().band(ready, banked);
        self.guard_condition(ready, guard, fuel)?;
        let tag = self.builder.ins().load(types::I8, MemFlags::new(), slot, 0);
        let is_table =
            self.builder
                .ins()
                .icmp_imm(IntCC::Equal, tag, i64::from(NATIVE_VALUE_TAG_TABLE));
        let lane = self.builder.ins().iadd_imm(named, -1);
        let at = self.builder.ins().ishl_imm(lane, 4);
        let entry = self.builder.ins().iadd(bank, at);
        let held = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            slot,
            NATIVE_VALUE_PAYLOAD_OFFSET,
        );
        let banked_table = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            entry,
            NATIVE_VALUE_PAYLOAD_OFFSET,
        );
        let same = self.builder.ins().icmp(IntCC::Equal, held, banked_table);
        let valid = self.builder.ins().band(is_table, same);
        self.guard_condition(valid, guard, fuel)?;
        Ok(lane)
    }

    /// A [`TraceOp::MetaGuard`]: one load of the metatable word of the
    /// table's row and one compare with the pinned metatable's identity, a
    /// constant of the code. A table whose row cannot name its metatable has
    /// the word zero, which no pinned identity equals, so it leaves too.
    fn meta_guard(
        &mut self,
        table: ValueId,
        binding: u32,
        guard: GuardId,
        fuel: Value,
    ) -> Result<(), TraceCompileError> {
        let BindingKind::MetaClass { metatable, .. } =
            &self.members[0].ir.bindings[binding as usize].kind
        else {
            return Err(TraceCompileError::NativeBackendRejected);
        };
        let identity = meta_word(*metatable).cast_signed();
        let Some(field) = self.members[0]
            .ir
            .fields
            .iter()
            .position(|field| field.key.as_ref() == META_FIELD_KEY)
        else {
            return Err(TraceCompileError::NativeBackendRejected);
        };
        let lane = self.value(table);
        let row = self.row_address(lane);
        let word = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            row,
            offset(field * mem::size_of::<usize>()),
        );
        let pass = self.builder.create_block();
        let fail = self.builder.create_block();
        self.builder.set_cold_block(fail);
        if self.forced_guard == Some((self.member, guard.index())) {
            let never = self.builder.ins().iconst(types::I8, 0);
            self.builder.ins().brif(never, pass, &[], fail, &[]);
        } else {
            let same = self.builder.ins().icmp_imm(IntCC::Equal, word, identity);
            self.builder.ins().brif(same, pass, &[], fail, &[]);
        }
        self.builder.switch_to_block(fail);
        self.leave(guard, fuel)?;
        self.builder.switch_to_block(pass);
        Ok(())
    }

    /// The address of field `field` in the row of the table in `table`, which
    /// the VM filled for its lane: the present value of the field's type, or
    /// zero, and then the guard leaves, so the interpreter does the access
    /// (it may run `__index` or `__newindex`, or raise an error).
    ///
    /// The other fields of the table that the straight-line code after this
    /// access reads or writes (see [`Self::slot_group`]) get the same guard
    /// in one check: the guard leaves at this access's snapshot, before any
    /// op of the group runs, so the interpreter redoes them itself.
    fn field_slot(
        &mut self,
        table: ValueId,
        field: u32,
        guard: GuardId,
        fuel: Value,
        ahead: &[TraceInstruction],
    ) -> Result<Value, TraceCompileError> {
        let lane = self.value(table);
        // An access the code before it already made to this field of this
        // table, with nothing since that could have moved or retyped the
        // slot, finds the slot present: the guard checked it. A guard that
        // a side trace continues, or that a test forces, is kept.
        let cached = self.function.is_none()
            && self.forced_guard.is_none()
            && !self.members[self.member]
                .attached
                .iter()
                .any(|(attached, _)| *attached == guard.index());
        if cached
            && let Some((_, _, slot)) = self
                .field_slots
                .iter()
                .find(|(held, held_field, _)| *held == lane && *held_field == field)
        {
            return Ok(*slot);
        }
        let row = self.row_address(lane);
        let group = if cached && self.row_masked {
            self.slot_group(ahead, table, lane, field)
        } else {
            Vec::new()
        };
        let mut slots = Vec::with_capacity(group.len() + 1);
        for held in std::iter::once(field).chain(group.iter().copied()) {
            let slot = self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                row,
                offset(held as usize * mem::size_of::<usize>()),
            );
            slots.push((held, slot));
        }
        let pass = self.builder.create_block();
        let fail = self.builder.create_block();
        self.builder.set_cold_block(fail);
        if self.forced_guard == Some((self.member, guard.index())) {
            let never = self.builder.ins().iconst(types::I8, 0);
            self.builder.ins().brif(never, pass, &[], fail, &[]);
        } else if let [(_, slot)] = slots[..] {
            self.builder.ins().brif(slot, pass, &[], fail, &[]);
        } else if self.row_masked && slots.iter().all(|(held, _)| *held < 64) {
            // The row's presence mask (its last word, see `row_stride_of`)
            // holds one bit per word, so one compare checks every field.
            let needed = slots.iter().fold(0u64, |bits, (held, _)| bits | 1 << held);
            let wanted = self.builder.ins().iconst(types::I64, needed as i64);
            let mask = self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                row,
                offset((self.row_stride - 1) * mem::size_of::<usize>()),
            );
            let have = self.builder.ins().band(mask, wanted);
            let present = self.builder.ins().icmp(IntCC::Equal, have, wanted);
            self.builder.ins().brif(present, pass, &[], fail, &[]);
        } else {
            let mut present: Option<Value> = None;
            for (_, slot) in &slots {
                let nonzero = self.builder.ins().icmp_imm(IntCC::NotEqual, *slot, 0);
                present = Some(match present {
                    None => nonzero,
                    Some(all) => self.builder.ins().band(all, nonzero),
                });
            }
            let present = present.expect("the access itself is in the group");
            self.builder.ins().brif(present, pass, &[], fail, &[]);
        }
        self.builder.switch_to_block(fail);
        self.leave(guard, fuel)?;
        self.builder.switch_to_block(pass);
        if self.function.is_none() {
            for (held, slot) in &slots {
                self.field_slots.push((lane, *held, *slot));
            }
        }
        Ok(slots[0].1)
    }

    /// The fields of `table`, other than `field`, that the straight-line ops
    /// in `ahead` read or write before anything that could move or retype a
    /// slot, and whose slots this code does not hold yet. Those ops are the
    /// ones that keep the slot cache alive (see `lower_instruction`), so the
    /// row of `table` reads the same words at the group's check as at each
    /// access. A forced or attached guard ends the group.
    fn slot_group(
        &self,
        ahead: &[TraceInstruction],
        table: ValueId,
        lane: Value,
        field: u32,
    ) -> Vec<u32> {
        let member = &self.members[self.member];
        let blocked = |guard: GuardId| {
            self.forced_guard == Some((self.member, guard.index()))
                || member
                    .attached
                    .iter()
                    .any(|(attached, _)| *attached == guard.index())
        };
        let mut group: Vec<u32> = Vec::new();
        for instruction in ahead {
            if instruction
                .result
                .is_some_and(|result| self.deferred.get(result.index()).copied().unwrap_or(false))
            {
                continue;
            }
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
                        .field_slots
                        .iter()
                        .any(|(l, f, _)| *l == lane && f == held);
                    if *on == table && *held != field && !cached && !group.contains(held) {
                        group.push(*held);
                    }
                    // A store to the table ends the group: an access after it
                    // keeps its own guard, so the exit that a missing field
                    // takes keeps the state it had before the store.
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

    fn float_arithmetic(&mut self, op: ArithmeticOp, left: Value, right: Value) -> Value {
        match op {
            ArithmeticOp::Add => self.builder.ins().fadd(left, right),
            ArithmeticOp::Subtract => self.builder.ins().fsub(left, right),
            ArithmeticOp::Multiply => self.builder.ins().fmul(left, right),
            ArithmeticOp::Divide => self.builder.ins().fdiv(left, right),
            ArithmeticOp::Modulo => {
                let quotient = self.builder.ins().fdiv(left, right);
                let floored = self.builder.ins().floor(quotient);
                let product = self.builder.ins().fmul(floored, right);
                self.builder.ins().fsub(left, product)
            }
            ArithmeticOp::Power => {
                let call = self.builder.ins().call(self.pow, &[left, right]);
                self.builder.inst_results(call)[0]
            }
            ArithmeticOp::FloorDivide => {
                let quotient = self.builder.ins().fdiv(left, right);
                self.builder.ins().floor(quotient)
            }
            ArithmeticOp::PowerSquare => self.float_power_square(left, right),
            ArithmeticOp::ModuloWith(FloatModulo::Fmod53) => {
                let helper = self.fmod53.expect("declared for a trace with this modulo");
                self.float_fmod(helper, left, right, false)
            }
            ArithmeticOp::ModuloWith(FloatModulo::Fmod54) => {
                let helper = self.fmod54.expect("declared for a trace with this modulo");
                self.float_fmod(helper, left, right, true)
            }
            ArithmeticOp::ModuloWith(FloatModulo::Floor) => {
                self.float_arithmetic(ArithmeticOp::Modulo, left, right)
            }
            _ => unreachable!("a verified trace has no bitwise float operation"),
        }
    }

    /// `a ^ b` of Lua 5.4 and 5.5: `a * a` inline for an exponent of 2, the
    /// `pow` helper otherwise (the same helper `Power` calls).
    fn float_power_square(&mut self, left: Value, right: Value) -> Value {
        let two = self.builder.ins().f64const(2.0);
        let is_square = self.builder.ins().fcmp(FloatCC::Equal, right, two);
        let square = self.builder.create_block();
        let general = self.builder.create_block();
        let merge = self.builder.create_block();
        self.builder.append_block_param(merge, types::F64);
        self.builder.set_cold_block(general);
        self.builder
            .ins()
            .brif(is_square, square, &[], general, &[]);
        self.builder.switch_to_block(square);
        let product = self.builder.ins().fmul(left, left);
        self.builder.ins().jump(merge, &[BlockArg::Value(product)]);
        self.builder.switch_to_block(general);
        let call = self.builder.ins().call(self.pow, &[left, right]);
        let power = self.builder.inst_results(call)[0];
        self.builder.ins().jump(merge, &[BlockArg::Value(power)]);
        self.builder.switch_to_block(merge);
        self.builder.block_params(merge)[0]
    }

    /// `a % b` of Lua 5.3 and later. With an inline `fma`, see
    /// [`Self::float_fmod_fma`]. Otherwise operands that are both integral with a
    /// magnitude below 2^53 and a non-zero divisor take an exact integer
    /// remainder with the floor sign fix, inline. A zero result has the sign
    /// of `a`, as `fmod` gives. Everything else calls `helper`.
    fn float_fmod(
        &mut self,
        helper: FuncRef,
        left: Value,
        right: Value,
        fifty_four: bool,
    ) -> Value {
        if self.has_fma {
            return self.float_fmod_fma(helper, left, right, fifty_four);
        }
        let limit = self.builder.ins().f64const(FLOAT_EXACT_LIMIT);
        let abs_left = self.builder.ins().fabs(left);
        let abs_right = self.builder.ins().fabs(right);
        // Ordered comparisons are false for NaN.
        let left_small = self.builder.ins().fcmp(FloatCC::LessThan, abs_left, limit);
        let right_small = self.builder.ins().fcmp(FloatCC::LessThan, abs_right, limit);
        let a = self.builder.ins().fcvt_to_sint_sat(types::I64, left);
        let b = self.builder.ins().fcvt_to_sint_sat(types::I64, right);
        let a_back = self.builder.ins().fcvt_from_sint(types::F64, a);
        let b_back = self.builder.ins().fcvt_from_sint(types::F64, b);
        let left_integral = self.builder.ins().fcmp(FloatCC::Equal, a_back, left);
        let right_integral = self.builder.ins().fcmp(FloatCC::Equal, b_back, right);
        let divisor_set = self.builder.ins().icmp_imm(IntCC::NotEqual, b, 0);
        let mut ok = self.builder.ins().band(left_small, right_small);
        ok = self.builder.ins().band(ok, left_integral);
        ok = self.builder.ins().band(ok, right_integral);
        ok = self.builder.ins().band(ok, divisor_set);
        let exact = self.builder.create_block();
        let general = self.builder.create_block();
        let merge = self.builder.create_block();
        self.builder.append_block_param(merge, types::F64);
        self.builder.set_cold_block(general);
        self.builder.ins().brif(ok, exact, &[], general, &[]);

        self.builder.switch_to_block(exact);
        // `|b| < 2^53`, so `srem` neither traps nor overflows.
        let remainder = self.builder.ins().srem(a, b);
        let differs = self.builder.ins().bxor(remainder, b);
        let signs_differ = self
            .builder
            .ins()
            .icmp_imm(IntCC::SignedLessThan, differs, 0);
        let non_zero = self.builder.ins().icmp_imm(IntCC::NotEqual, remainder, 0);
        let fix = self.builder.ins().band(signs_differ, non_zero);
        let shifted = self.builder.ins().iadd(remainder, b);
        let floored = self.builder.ins().select(fix, shifted, remainder);
        let magnitude = self.builder.ins().fcvt_from_sint(types::F64, floored);
        let signed_zero = self.builder.ins().fcopysign(magnitude, left);
        let is_zero = self.builder.ins().icmp_imm(IntCC::Equal, floored, 0);
        let result = self.builder.ins().select(is_zero, signed_zero, magnitude);
        self.builder.ins().jump(merge, &[BlockArg::Value(result)]);

        self.builder.switch_to_block(general);
        let call = self.builder.ins().call(helper, &[left, right]);
        let value = self.builder.inst_results(call)[0];
        self.builder.ins().jump(merge, &[BlockArg::Value(value)]);

        self.builder.switch_to_block(merge);
        self.builder.block_params(merge)[0]
    }

    /// `%` of Lua 5.3 and later on a target with a one-instruction `fma`,
    /// for every operand pair, integral or not: `q = trunc(a / b)` and
    /// `r = fma(-q, b, a)` are the exact remainder whenever the quotient is
    /// below `2^53` and `r` is zero or has the sign of `a`. The rounded
    /// quotient is never below the true one, so `q` is the true truncated
    /// quotient or one more; one more leaves a non-zero `r` of the wrong
    /// sign. A true `q` makes `a = q * b + r` exact with `|r| < |b|`, so no
    /// test of `|r|` is needed. A zero or infinite divisor, a NaN, an
    /// infinite dividend and a quotient of `2^53` or more fail the two tests
    /// (`r` is then NaN or not known exact) and take `helper`. A remainder
    /// of zero gets the sign of `a`, then the Lua sign fix. The straight-line
    /// path is about 20 operations with one branch. See
    /// `oximond::compat::fma_fmod_checked` for the checked form in Rust, and
    /// the test `range_and_sign_alone_prove_the_fma_remainder` there.
    fn float_fmod_fma(
        &mut self,
        helper: FuncRef,
        left: Value,
        right: Value,
        fifty_four: bool,
    ) -> Value {
        let limit = self.builder.ins().f64const(FLOAT_EXACT_LIMIT);
        let zero = self.builder.ins().f64const(0.0);
        let quotient = self.builder.ins().fdiv(left, right);
        let abs_quotient = self.builder.ins().fabs(quotient);
        let q = self.builder.ins().trunc(quotient);
        let neg_q = self.builder.ins().fneg(q);
        let r = self.builder.ins().fma(neg_q, right, left);
        // Ordered comparisons are false for NaN, so NaN operands fail them.
        let in_range = self
            .builder
            .ins()
            .fcmp(FloatCC::LessThan, abs_quotient, limit);
        let m = self.builder.ins().fcopysign(r, left);
        // `m == r` unless `r` is non-zero with a sign other than `a`'s.
        let same_sign = self.builder.ins().fcmp(FloatCC::Equal, m, r);
        let ok = self.builder.ins().band(in_range, same_sign);

        let fast = self.builder.create_block();
        let general = self.builder.create_block();
        let merge = self.builder.create_block();
        self.builder.append_block_param(merge, types::F64);
        self.builder.set_cold_block(general);
        self.builder.ins().brif(ok, fast, &[], general, &[]);

        self.builder.switch_to_block(fast);
        let fix = if fifty_four {
            // `m` is non-zero with a sign other than `b`'s. `b` is not NaN here.
            let m_bits = self.builder.ins().bitcast(types::I64, MemFlags::new(), m);
            let b_bits = self
                .builder
                .ins()
                .bitcast(types::I64, MemFlags::new(), right);
            let differ = self.builder.ins().bxor(m_bits, b_bits);
            let signs_differ = self
                .builder
                .ins()
                .icmp_imm(IntCC::SignedLessThan, differ, 0);
            let m_nonzero = self.builder.ins().fcmp(FloatCC::NotEqual, m, zero);
            self.builder.ins().band(signs_differ, m_nonzero)
        } else {
            let product = self.builder.ins().fmul(m, right);
            self.builder.ins().fcmp(FloatCC::LessThan, product, zero)
        };
        let shifted = self.builder.ins().fadd(m, right);
        let result = self.builder.ins().select(fix, shifted, m);
        self.builder.ins().jump(merge, &[BlockArg::Value(result)]);

        self.builder.switch_to_block(general);
        let call = self.builder.ins().call(helper, &[left, right]);
        let value = self.builder.inst_results(call)[0];
        self.builder.ins().jump(merge, &[BlockArg::Value(value)]);

        self.builder.switch_to_block(merge);
        self.builder.block_params(merge)[0]
    }

    fn integer_arithmetic(
        &mut self,
        op: ArithmeticOp,
        left: Value,
        right: Value,
    ) -> Result<Value, TraceCompileError> {
        Ok(match op {
            ArithmeticOp::Add => self.builder.ins().iadd(left, right),
            ArithmeticOp::Subtract => self.builder.ins().isub(left, right),
            ArithmeticOp::Multiply => self.builder.ins().imul(left, right),
            ArithmeticOp::BitAnd => self.builder.ins().band(left, right),
            ArithmeticOp::BitOr => self.builder.ins().bor(left, right),
            ArithmeticOp::BitXor => self.builder.ins().bxor(left, right),
            ArithmeticOp::Modulo | ArithmeticOp::FloorDivide => {
                self.integer_division(op, left, right)
            }
            ArithmeticOp::ShiftLeft => self.integer_shift(left, right),
            ArithmeticOp::ShiftRight => {
                // `a >> b` is `a << -b`, and negating the most negative
                // distance saturates, as the interpreter does.
                let negated = self.builder.ins().ineg(right);
                let most_negative = self.builder.ins().icmp_imm(IntCC::Equal, right, i64::MIN);
                let largest = self.builder.ins().iconst(types::I64, i64::MAX);
                let distance = self.builder.ins().select(most_negative, largest, negated);
                self.integer_shift(left, distance)
            }
            ArithmeticOp::Divide
            | ArithmeticOp::Power
            | ArithmeticOp::PowerSquare
            | ArithmeticOp::ModuloWith(_)
            | ArithmeticOp::ArithmeticShiftRight
            | ArithmeticOp::RotateLeft
            | ArithmeticOp::RotateRight => {
                return Err(TraceCompileError::NativeBackendRejected);
            }
        })
    }

    /// Arithmetic on two 64-bit integer cdata, as
    /// [`oximond::jit_ir::trace::wide_arithmetic`] defines it. Never traps:
    /// the divisor is replaced before `sdiv` and `udiv` can see a zero or the
    /// overflowing `MIN / -1`, and the results it got wrong are selected away.
    fn wide_arithmetic(
        &mut self,
        op: ArithmeticOp,
        signed: bool,
        left: Value,
        right: Value,
    ) -> Value {
        let builder = &mut *self.builder;
        match op {
            ArithmeticOp::Add => builder.ins().iadd(left, right),
            ArithmeticOp::Subtract => builder.ins().isub(left, right),
            ArithmeticOp::Multiply => builder.ins().imul(left, right),
            ArithmeticOp::BitAnd => builder.ins().band(left, right),
            ArithmeticOp::BitOr => builder.ins().bor(left, right),
            ArithmeticOp::BitXor => builder.ins().bxor(left, right),
            // Cranelift takes the count modulo 64, as the `bit` library does.
            ArithmeticOp::ShiftLeft => builder.ins().ishl(left, right),
            ArithmeticOp::ShiftRight => builder.ins().ushr(left, right),
            ArithmeticOp::ArithmeticShiftRight => builder.ins().sshr(left, right),
            ArithmeticOp::RotateLeft => builder.ins().rotl(left, right),
            ArithmeticOp::RotateRight => builder.ins().rotr(left, right),
            ArithmeticOp::Divide | ArithmeticOp::Modulo => {
                let one = builder.ins().iconst(types::I64, 1);
                let failed = builder.ins().iconst(types::I64, i64::MIN);
                let is_zero = builder.ins().icmp_imm(IntCC::Equal, right, 0);
                if signed {
                    // A divisor of 0 or -1 is 0 or 1 after adding one,
                    // unsigned.
                    let shifted = builder.ins().iadd_imm(right, 1);
                    let special =
                        builder
                            .ins()
                            .icmp_imm(IntCC::UnsignedLessThanOrEqual, shifted, 1);
                    let divisor = builder.ins().select(special, one, right);
                    if op == ArithmeticOp::Modulo {
                        // Both replaced divisors leave a zero remainder.
                        let remainder = builder.ins().srem(left, divisor);
                        builder.ins().select(is_zero, failed, remainder)
                    } else {
                        let quotient = builder.ins().sdiv(left, divisor);
                        let is_minus_one = builder.ins().icmp_imm(IntCC::Equal, right, -1);
                        let negated = builder.ins().ineg(left);
                        let quotient = builder.ins().select(is_minus_one, negated, quotient);
                        builder.ins().select(is_zero, failed, quotient)
                    }
                } else {
                    let divisor = builder.ins().select(is_zero, one, right);
                    let result = if op == ArithmeticOp::Modulo {
                        builder.ins().urem(left, divisor)
                    } else {
                        builder.ins().udiv(left, divisor)
                    };
                    builder.ins().select(is_zero, failed, result)
                }
            }
            ArithmeticOp::Power
            | ArithmeticOp::PowerSquare
            | ArithmeticOp::ModuloWith(_)
            | ArithmeticOp::FloorDivide => {
                unreachable!("a verified trace has no such cdata operation")
            }
        }
    }

    /// C's conversion of a float to `int64_t` (`signed`) or `uint64_t`:
    /// truncation, with `INT64_MIN` for NaN and out-of-range values.
    fn float_to_wide(&mut self, number: Value, signed: bool) -> Value {
        let builder = &mut *self.builder;
        let two_63 = builder.ins().f64const(9_223_372_036_854_775_808.0);
        let source = if signed {
            number
        } else {
            // From 2^63, convert `number - 2^64`, which wraps the result.
            let two_64 = builder.ins().f64const(18_446_744_073_709_551_616.0);
            let high = builder
                .ins()
                .fcmp(FloatCC::GreaterThanOrEqual, number, two_63);
            let wrapped = builder.ins().fsub(number, two_64);
            builder.ins().select(high, wrapped, number)
        };
        // The saturating conversion already gives `INT64_MIN` below the
        // range, so only NaN and values from 2^63 up need the select. The
        // ordered compare is false for NaN.
        let below = builder.ins().fcmp(FloatCC::LessThan, source, two_63);
        let converted = builder.ins().fcvt_to_sint_sat(types::I64, source);
        let failed = builder.ins().iconst(types::I64, i64::MIN);
        builder.ins().select(below, converted, failed)
    }

    /// The integer `value` when it is defined by an `iconst`.
    fn constant_integer(&self, value: Value) -> Option<i64> {
        let dfg = &self.builder.func.dfg;
        let cranelift_codegen::ir::ValueDef::Result(inst, _) = dfg.value_def(value) else {
            return None;
        };
        match dfg.insts[inst] {
            cranelift_codegen::ir::InstructionData::UnaryImm {
                opcode: cranelift_codegen::ir::Opcode::Iconst,
                imm,
            } => Some(imm.bits()),
            _ => None,
        }
    }

    /// `integer <comparison> float`, exact as [`oximond::jit_ir::trace::compare`]
    /// defines it (the integer is not converted). Inline and branch-free.
    fn compare_integer_float(
        &mut self,
        comparison: Comparison,
        integer: Value,
        float: Value,
    ) -> Value {
        self.mixed_compare(comparison, true, integer, float)
    }

    /// `float <comparison> integer`; see [`Self::compare_integer_float`].
    fn compare_float_integer(
        &mut self,
        comparison: Comparison,
        float: Value,
        integer: Value,
    ) -> Value {
        self.mixed_compare(comparison, false, integer, float)
    }

    /// Exact comparison of an integer and a float, `integer_left` telling
    /// which side the integer is on. NaN compares false.
    fn mixed_compare(
        &mut self,
        comparison: Comparison,
        integer_left: bool,
        integer: Value,
        float: Value,
    ) -> Value {
        // Within 2^53 the conversion of the integer is exact, so a float
        // compare answers (and folds when the integer is a constant).
        const EXACT: i64 = 1 << 53;
        let constant = self.constant_integer(integer);
        let builder = &mut *self.builder;
        let float_condition = match comparison {
            Comparison::Equal => FloatCC::Equal,
            Comparison::LessThan => FloatCC::LessThan,
            Comparison::LessEqual => FloatCC::LessThanOrEqual,
        };
        if let Some(constant) = constant.filter(|c| (-EXACT..=EXACT).contains(c)) {
            #[allow(clippy::cast_precision_loss)]
            let converted = builder.ins().f64const(constant as f64);
            return if integer_left {
                builder.ins().fcmp(float_condition, converted, float)
            } else {
                builder.ins().fcmp(float_condition, float, converted)
            };
        }
        let two_63 = builder.ins().f64const(9_223_372_036_854_775_808.0);
        if comparison == Comparison::Equal {
            // Equal when the float is integral, below 2^63 and converts
            // back to the integer. The round trip through the float rules
            // out a fraction; the bound rules out the saturated 2^63.
            let converted = builder.ins().fcvt_from_sint(types::F64, integer);
            let same = builder.ins().fcmp(FloatCC::Equal, converted, float);
            let below = builder.ins().fcmp(FloatCC::LessThan, float, two_63);
            let back = builder.ins().fcvt_to_sint_sat(types::I64, float);
            let round = builder.ins().icmp(IntCC::Equal, back, integer);
            let both = builder.ins().band(same, below);
            return builder.ins().band(both, round);
        }
        // An order against an integer is an order against the float rounded
        // toward the integer: `i < x` is `i < ceil(x)`, `i <= x` is
        // `i <= floor(x)`, `x < i` is `floor(x) < i`, `x <= i` is
        // `ceil(x) <= i`. A rounded float outside the integers' range, and
        // NaN, are settled by their own answer.
        let strict = comparison == Comparison::LessThan;
        let rounded = if strict == integer_left {
            builder.ins().ceil(float)
        } else {
            builder.ins().floor(float)
        };
        let converted = builder.ins().fcvt_to_sint_sat(types::I64, rounded);
        let condition = match (comparison, integer_left) {
            (Comparison::LessThan, _) => IntCC::SignedLessThan,
            _ => IntCC::SignedLessThanOrEqual,
        };
        let inside = if integer_left {
            builder.ins().icmp(condition, integer, converted)
        } else {
            builder.ins().icmp(condition, converted, integer)
        };
        let minus_two_63 = builder.ins().f64const(-9_223_372_036_854_775_808.0);
        let high = builder
            .ins()
            .fcmp(FloatCC::GreaterThanOrEqual, rounded, two_63);
        let low = builder.ins().fcmp(FloatCC::LessThan, rounded, minus_two_63);
        let ordered = builder.ins().fcmp(FloatCC::Equal, float, float);
        // Above the range the integer is the lesser; below it, the greater.
        let on_high = builder.ins().iconst(types::I8, i64::from(integer_left));
        let on_low = builder.ins().iconst(types::I8, i64::from(!integer_left));
        let below = builder.ins().select(low, on_low, inside);
        let settled = builder.ins().select(high, on_high, below);
        builder.ins().band(settled, ordered)
    }

    /// Floor division or modulo by a constant `divisor > 0`: no zero or -1 to
    /// replace, and the remainder only needs its sign fixed. A power of two
    /// is a mask or an arithmetic shift, which round toward negative
    /// infinity on their own.
    fn integer_division_by_constant(
        &mut self,
        op: ArithmeticOp,
        left: Value,
        divisor: i64,
    ) -> Value {
        let builder = &mut *self.builder;
        let power_of_two = divisor & (divisor - 1) == 0;
        if op == ArithmeticOp::Modulo {
            if power_of_two {
                return builder.ins().band_imm(left, divisor - 1);
            }
            let remainder = builder.ins().srem_imm(left, divisor);
            // A negative remainder is lifted into [0, divisor).
            let sign = builder.ins().sshr_imm(remainder, 63);
            let addend = builder.ins().band_imm(sign, divisor);
            return builder.ins().iadd(remainder, addend);
        }
        if power_of_two {
            return builder
                .ins()
                .sshr_imm(left, i64::from(divisor.trailing_zeros()));
        }
        let quotient = builder.ins().sdiv_imm(left, divisor);
        let remainder = builder.ins().srem_imm(left, divisor);
        // The truncated quotient is one too high exactly when the remainder
        // is negative.
        let sign = builder.ins().sshr_imm(remainder, 63);
        builder.ins().iadd(quotient, sign)
    }

    /// Floor division or modulo with the sign rules of Lua. Never traps: a
    /// divisor of 0 or -1 is replaced by 1 before `sdiv` and `srem`, then the
    /// results the replacement got wrong are selected away.
    fn integer_division(&mut self, op: ArithmeticOp, left: Value, right: Value) -> Value {
        if let Some(divisor) = self.constant_integer(right).filter(|divisor| *divisor > 0) {
            return self.integer_division_by_constant(op, left, divisor);
        }
        let builder = &mut *self.builder;
        // `right` is 0 or -1 exactly when `right + 1` is 0 or 1, unsigned.
        let shifted = builder.ins().iadd_imm(right, 1);
        let special = builder
            .ins()
            .icmp_imm(IntCC::UnsignedLessThanOrEqual, shifted, 1);
        let one = builder.ins().iconst(types::I64, 1);
        let zero = builder.ins().iconst(types::I64, 0);
        let divisor = builder.ins().select(special, one, right);
        let remainder = builder.ins().srem(left, divisor);
        // Round toward negative infinity: the truncated remainder has the
        // wrong sign when it is non-zero and differs in sign from the divisor.
        let nonzero = builder.ins().icmp_imm(IntCC::NotEqual, remainder, 0);
        let signs = builder.ins().bxor(remainder, divisor);
        let differ = builder.ins().icmp_imm(IntCC::SignedLessThan, signs, 0);
        let adjust = builder.ins().band(nonzero, differ);
        if op == ArithmeticOp::Modulo {
            // For a replaced divisor the remainder is 0, so this is 0.
            let correction = builder.ins().select(adjust, divisor, zero);
            builder.ins().iadd(remainder, correction)
        } else {
            let quotient = builder.ins().sdiv(left, divisor);
            let wide = builder.ins().uextend(types::I64, adjust);
            let floored = builder.ins().isub(quotient, wide);
            let is_zero = builder.ins().icmp_imm(IntCC::Equal, right, 0);
            let is_minus_one = builder.ins().icmp_imm(IntCC::Equal, right, -1);
            let negated = builder.ins().ineg(left);
            let by_zero = builder.ins().select(is_zero, zero, floored);
            builder.ins().select(is_minus_one, negated, by_zero)
        }
    }

    /// Lua's shift: left for a positive distance, logical right for a
    /// negative one, zero when the distance is 64 places or more.
    fn integer_shift(&mut self, value: Value, distance: Value) -> Value {
        let builder = &mut *self.builder;
        let negative = builder.ins().icmp_imm(IntCC::SignedLessThan, distance, 0);
        let negated = builder.ins().ineg(distance);
        let amount = builder.ins().select(negative, negated, distance);
        let left = builder.ins().ishl(value, amount);
        let right = builder.ins().ushr(value, amount);
        let shifted = builder.ins().select(negative, right, left);
        // In range when -63 <= distance <= 63.
        let offset = builder.ins().iadd_imm(distance, 63);
        let in_range = builder
            .ins()
            .icmp_imm(IntCC::UnsignedLessThanOrEqual, offset, 126);
        let zero = builder.ins().iconst(types::I64, 0);
        builder.ins().select(in_range, shifted, zero)
    }

    /// Fills the current block as the stub for exit `index`: spill the
    /// snapshot's values, record the fuel left, and return the index.
    fn emit_exit(&mut self, index: usize, fuel: Value) {
        if self.emit_shared_exit(index, fuel) {
            return;
        }
        self.spill_exit(index);
        let control = self.control.expect("entry set the control block");
        let code = self.exit_code(index);
        self.builder.ins().store(
            MemFlags::new(),
            fuel,
            control,
            offset(mem::offset_of!(TraceControl, fuel)),
        );
        if let Some(budget) = self.budget {
            self.builder.ins().store(
                MemFlags::new(),
                budget,
                control,
                offset(mem::offset_of!(TraceControl, budget)),
            );
        }
        let mut returned = vec![code];
        if let Some(function) = &self.function {
            // An activation also says how deep it is, which the activations
            // around it, unwinding, leave as it is; and it gives the status
            // of the exit and a result beside it.
            let result = function.result;
            returned[0] = self.builder.ins().iconst(
                types::I64,
                exit_status(i64::try_from(index).expect("exit index fits")),
            );
            let left = self.depth_left.expect("entry set the depth left");
            let cap = self.builder.ins().load(
                types::I64,
                MemFlags::new(),
                control,
                offset(mem::offset_of!(TraceControl, depth_cap)),
            );
            let depth = self.builder.ins().isub(cap, left);
            self.builder.ins().store(
                MemFlags::new(),
                depth,
                control,
                offset(mem::offset_of!(TraceControl, depth)),
            );
            if let Some(ty) = result {
                returned.push(self.zero(ty));
            }
        }
        if returned.len() == 1 {
            self.leave_exit(returned[0]);
        } else {
            self.builder.ins().return_(&returned);
        }
    }

    /// Leaves the function with the exit code `code`: a root that native
    /// code can link to goes through its dispatch block, which calls the
    /// side the code's cell holds, and every other root returns it.
    fn leave_exit(&mut self, code: Value) {
        if self.native_link_slots().is_some() {
            let dispatch = self.emit_native_dispatch();
            self.builder.ins().jump(dispatch, &block_args(&[code]));
        } else {
            self.builder.ins().return_(&[code]);
        }
    }

    /// The link slots of the root, when its exits call the side that the cell
    /// of the exit holds from native code: a root of a tree that can take a
    /// side linking back to its loop, which is not counted, not a function,
    /// not versioned by a range plan and not a side compiled apart. Such a
    /// root also has loop parameters its iteration passes on unchanged, as
    /// the links of the others do not come back through a header (see
    /// [`Self::lower_root`]).
    fn native_link_of_root(&self) -> Option<Box<[(usize, TraceType)]>> {
        if self.detached.is_some() || self.function.is_some() {
            return None;
        }
        let root = self.members[0].ir;
        if root.counted || self.range_plan.is_some() || pass_through_params(root).is_empty() {
            return None;
        }
        link_slots_of(root)
    }

    fn native_link_slots(&self) -> Option<&[(usize, TraceType)]> {
        self.native_link.as_deref()
    }

    /// The shared block of the native exits of a root (see [`Self::leave_exit`]),
    /// made by the first exit that needs it. The exit that jumps here has
    /// stored its values and the fuel left. The block calls the side the cell
    /// of the exit holds, if any, from native code, with no Rust frame between
    /// them:
    ///
    /// - a side that links back ([`LINK_CODE`]) writes the loop parameters it
    ///   left in the exit area into the link slots of the inputs, and enters
    ///   the root again, so the loop runs on without leaving;
    /// - a side that declined ([`DECLINE_CODE`]) gives this exit's code back,
    ///   as a run that calls it again would see it decline again;
    /// - any other code goes to [`trace_call_sides`], which follows the chain
    ///   of sides from it the way the run does, and returns the code the run
    ///   leaves through, or [`LINK_CODE`] to link.
    ///
    /// The cell is read at each exit, so a side compiled after the run started
    /// is taken from the next exit on. The block has no loop: a loop around
    /// the call makes the register allocator weigh the run's buffers as hot
    /// values and spread them over the root's own loop
    /// (`algorithm/mandelbrot` +6%).
    fn emit_native_dispatch(&mut self) -> Block {
        if let Some(block) = self.native_dispatch {
            return block;
        }
        let block = self.builder.create_block();
        self.builder.append_block_param(block, types::I64);
        self.native_dispatch = Some(block);
        block
    }

    /// Emits the body of the native dispatch block, if an exit made one. It
    /// is emitted when lowering is over, because the exits that jump to it
    /// are emitted in the middle of blocks that are not filled yet.
    fn finish_native_dispatch(&mut self) {
        let Some(block) = self.native_dispatch else {
            return;
        };
        let slots: Box<[(usize, TraceType)]> = self
            .native_link_slots()
            .expect("a root that links natively")
            .into();
        let entry = self.native_entry.expect("a root that links natively");
        let control = self.control.expect("entry set the control block");
        let exit_area = self.exit_area.expect("entry set the exit area");
        // The three buffers wait here while a side runs, so that no value
        // of the loop is live across the call (see `lower_root`).
        let saved = self.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            16,
            3,
        ));
        self.builder.switch_to_block(block);
        let code = self.builder.block_params(block)[0];
        // Not a value of the root's loop: the loop would keep it in a register
        // (or a stack slot) for the exits that jump here.
        let inputs = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(mem::offset_of!(TraceControl, inputs)),
        );
        let cells = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(mem::offset_of!(TraceControl, cells)),
        );
        let shifted = self.builder.ins().ishl_imm(code, 3);
        let cell_at = self.builder.ins().iadd(cells, shifted);
        let cell = self
            .builder
            .ins()
            .load(types::I64, MemFlags::new(), cell_at, 0);
        let plain = self.builder.create_block();
        let calling = self.builder.create_block();
        let link = self.builder.create_block();
        self.builder.set_cold_block(plain);
        self.builder.set_cold_block(link);
        self.builder.ins().brif(cell, calling, &[], plain, &[]);

        self.builder.switch_to_block(plain);
        self.builder.ins().return_(&[code]);

        self.builder.switch_to_block(calling);
        for (index, value) in [control, exit_area].into_iter().enumerate() {
            self.builder
                .ins()
                .stack_store(value, saved, i32::try_from(index * 8).unwrap_or(0));
        }
        for field in [
            mem::offset_of!(TraceControl, native_calls),
            mem::offset_of!(TraceControl, links),
        ] {
            let at = offset(field);
            let count = self
                .builder
                .ins()
                .load(types::I64, MemFlags::new(), control, at);
            let count = self.builder.ins().iadd_imm(count, 1);
            self.builder
                .ins()
                .store(MemFlags::new(), count, control, at);
        }
        let mut signature = Signature::new(self.builder.func.signature.call_conv);
        for _ in 0..3 {
            signature.params.push(AbiParam::new(types::I64));
        }
        signature.returns.push(AbiParam::new(types::I64));
        let signature = self.builder.import_signature(signature);
        let args = [inputs, control, exit_area];
        let call_inst = self.builder.ins().call_indirect(signature, cell, &args);
        let result = self.builder.inst_results(call_inst)[0];
        let not_link = self.builder.create_block();
        let is_link = self.builder.ins().icmp_imm(IntCC::Equal, result, LINK_CODE);
        self.builder.ins().brif(is_link, link, &[], not_link, &[]);

        self.builder.switch_to_block(not_link);
        let chained = self.builder.create_block();
        let is_declined = self
            .builder
            .ins()
            .icmp_imm(IntCC::Equal, result, DECLINE_CODE);
        let declined = self.builder.create_block();
        self.builder.set_cold_block(declined);
        self.builder
            .ins()
            .brif(is_declined, declined, &[], chained, &[]);

        // The side was called and did not run: the run must not call it again.
        self.builder.switch_to_block(declined);
        let one = self.builder.ins().iconst(types::I64, 1);
        self.builder.ins().store(
            MemFlags::new(),
            one,
            control,
            offset(mem::offset_of!(TraceControl, followed)),
        );
        self.builder.ins().return_(&[code]);

        self.builder.switch_to_block(chained);
        let mut chain = Signature::new(self.builder.func.signature.call_conv);
        for _ in 0..5 {
            chain.params.push(AbiParam::new(types::I64));
        }
        chain.returns.push(AbiParam::new(types::I64));
        let chain = self.builder.import_signature(chain);
        let function = self
            .builder
            .ins()
            .iconst(types::I64, trace_call_sides as *const () as usize as i64);
        let control_back = self.builder.ins().stack_load(types::I64, saved, 0);
        let area_back = self.builder.ins().stack_load(types::I64, saved, 8);
        let inputs_back = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            control_back,
            offset(mem::offset_of!(TraceControl, inputs)),
        );
        let tree = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            control_back,
            offset(mem::offset_of!(TraceControl, tree)),
        );
        let args = [tree, inputs_back, control_back, area_back, result];
        let chain_inst = self.builder.ins().call_indirect(chain, function, &args);
        let followed = self.builder.inst_results(chain_inst)[0];
        let done = self.builder.create_block();
        let linked_again = self
            .builder
            .ins()
            .icmp_imm(IntCC::Equal, followed, LINK_CODE);
        self.builder.ins().brif(linked_again, link, &[], done, &[]);
        self.builder.switch_to_block(done);
        let one = self.builder.ins().iconst(types::I64, 1);
        self.builder.ins().store(
            MemFlags::new(),
            one,
            control_back,
            offset(mem::offset_of!(TraceControl, followed)),
        );
        self.builder.ins().return_(&[followed]);

        // The side linked back: the root's loop parameters are in the exit
        // area. The link slots of the inputs take them, as the run would
        // write them, and the root is entered again from its start, which
        // checks the tags, reads the slots and runs the preheader as a run
        // that links does. The side left the fuel in the control block, where
        // the entry reads it.
        self.builder.switch_to_block(link);
        let control = self.builder.ins().stack_load(types::I64, saved, 0);
        let exit_area = self.builder.ins().stack_load(types::I64, saved, 8);
        let inputs = self.builder.ins().load(
            types::I64,
            MemFlags::new(),
            control,
            offset(mem::offset_of!(TraceControl, inputs)),
        );
        let slot_size = mem::size_of::<TraceInputSlot>();
        for (index, &(position, ty)) in slots.iter().enumerate() {
            let tag = match ty {
                TraceType::Float => TRACE_TAG_FLOAT,
                TraceType::Integer => TRACE_TAG_INTEGER,
                TraceType::Boolean => TRACE_TAG_BOOLEAN,
                TraceType::Table => TRACE_TAG_TABLE,
                TraceType::Closure => TRACE_TAG_CLOSURE,
                _ => TRACE_TAG_NIL,
            };
            let slot = offset(position * slot_size);
            let tag_value = self.builder.ins().iconst(types::I64, tag);
            self.builder.ins().store(
                MemFlags::new(),
                tag_value,
                inputs,
                slot + offset(mem::offset_of!(TraceInputSlot, tag)),
            );
            let at = offset(index * mem::size_of::<i64>());
            let bits = match ty {
                TraceType::Nil | TraceType::Any => self.builder.ins().iconst(types::I64, 0),
                _ => self
                    .builder
                    .ins()
                    .load(types::I64, MemFlags::new(), exit_area, at),
            };
            self.builder.ins().store(
                MemFlags::new(),
                bits,
                inputs,
                slot + offset(mem::offset_of!(TraceInputSlot, bits)),
            );
        }
        let one = self.builder.ins().iconst(types::I64, 1);
        self.builder.ins().store(
            MemFlags::new(),
            one,
            control,
            offset(mem::offset_of!(TraceControl, linked)),
        );
        self.builder
            .ins()
            .jump(entry, &block_args(&[inputs, control, exit_area]));
    }

    /// The value an exit stores for `value`: `None` for an entry input of
    /// the root, which exit code loads from its slot (see
    /// [`Self::reload_inputs`]).
    fn exit_value(&self, value: ValueId) -> Option<Value> {
        if self.member == 0 && self.entry_inputs.is_some() && self.entry_slots.contains_key(&value)
        {
            return None;
        }
        self.values[value.index()]
    }

    /// Fills the current block as the stub for exit `index` when the plan
    /// gives it tails: it stores its own values, and jumps to its first tail
    /// with the exit's index, the fuel and the budget. `false`, having
    /// emitted nothing, for any other exit. A tail whose values are other
    /// Cranelift values than an earlier exit stored there (the same trace
    /// value as a parameter of a later block) gets a block of its own: exits
    /// that store the same values share it.
    fn emit_shared_exit(&mut self, index: usize, fuel: Value) -> bool {
        if self.function.is_some() {
            return false;
        }
        let plan = self.plan;
        let shared = plan.shared_stores();
        let Some(first) = shared.stubs.get(index).and_then(|stub| stub.tail) else {
            return false;
        };
        let mut chain = Vec::new();
        let mut next = Some(first);
        while let Some(tail) = next {
            let values: Vec<Option<Value>> = shared.tails[tail]
                .slots
                .iter()
                .map(|slot| self.exit_value(slot.value))
                .collect();
            // The peak tier lets Cranelift's optimizer move the loop's values
            // around the exits, and it moves them worse when the exits share
            // several variants of a tail (hash_read_small runs 3% longer), so
            // there a tail takes one variant and the exits that differ spill
            // inline.
            if self.peak
                && let Some(existing) = self.tails[tail].first()
                && existing.values != values
            {
                return false;
            }
            chain.push((tail, values));
            next = shared.tails[tail].next;
        }
        let area = self.spill_area();
        self.store_slots(&shared.stubs[index].own, area);
        let code = self.exit_code(index);
        let mut args = vec![code, fuel];
        args.extend(self.budget);
        let width = args.len();
        // The variant of each tail on the chain, from the last up: one made
        // by an exit that stores the same values and continues to the same
        // variant of the next tail, or a new block.
        let mut made = Vec::new();
        let mut onward: Option<(usize, usize)> = None;
        for (tail, values) in chain.into_iter().rev() {
            let found = self.tails[tail]
                .iter()
                .position(|variant| variant.values == values && variant.next == onward);
            let variant = if let Some(found) = found {
                found
            } else {
                let block = self.builder.create_block();
                self.builder.set_cold_block(block);
                for _ in 0..width {
                    self.builder.append_block_param(block, types::I64);
                }
                self.tails[tail].push(ExitTail {
                    block,
                    values,
                    next: onward,
                });
                made.push((tail, self.tails[tail].len() - 1));
                self.tails[tail].len() - 1
            };
            onward = Some((tail, variant));
        }
        let (head, head_variant) = onward.expect("a chain has a first tail");
        let block = self.tails[head][head_variant].block;
        self.builder.ins().jump(block, &block_args(&args));
        // Every caller of this function switches to another block next, as
        // after any exit.
        for (tail, variant) in made {
            let block = self.tails[tail][variant].block;
            self.builder.switch_to_block(block);
            let params = self.builder.block_params(block).to_vec();
            self.store_slots(&shared.tails[tail].slots, area);
            if let Some((next_tail, next_variant)) = self.tails[tail][variant].next {
                let target = self.tails[next_tail][next_variant].block;
                self.builder.ins().jump(target, &block_args(&params));
                continue;
            }
            let control = self.control.expect("entry set the control block");
            self.builder.ins().store(
                MemFlags::new(),
                params[1],
                control,
                offset(mem::offset_of!(TraceControl, fuel)),
            );
            if self.budget.is_some() {
                self.builder.ins().store(
                    MemFlags::new(),
                    params[2],
                    control,
                    offset(mem::offset_of!(TraceControl, budget)),
                );
            }
            self.leave_exit(params[0]);
        }
        true
    }

    /// The code an exit of this code returns, as a value: its index, plus the
    /// unit's base for a side compiled apart from its tree.
    fn exit_code(&mut self, index: usize) -> Value {
        let base = self.detached.map_or(0, |spec| spec.code_base);
        let code = base + i64::try_from(index).expect("exit index fits");
        self.builder.ins().iconst(types::I64, code)
    }

    /// Stores the values the snapshot of exit `index` names into the exit
    /// area, and returns the area's address.
    fn spill_exit(&mut self, index: usize) -> Value {
        self.spill_exit_from(index, 0)
    }

    /// [`Self::spill_exit`] without the first `skip` slots.
    fn spill_exit_from(&mut self, index: usize, skip: usize) -> Value {
        let descriptor = &self.plan.descriptors[index];
        let exit_area = self.spill_area();
        #[cfg(feature = "op-stats")]
        {
            let stored = descriptor
                .slots
                .iter()
                .filter(|slot| !matches!(slot.ty, TraceType::Nil | TraceType::Any))
                .count();
            self.bump_counter(op_stats::Counter::NativeExits);
            self.add_counter(
                op_stats::Counter::NativeExitSlotsStored,
                i64::try_from(stored).expect("slot count fits i64"),
            );
        }
        let slots = descriptor.slots[skip..].to_vec();
        self.store_slots(&slots, exit_area);
        exit_area
    }

    /// Stores the values of `slots` into the exit area at `exit_area`.
    fn store_slots(&mut self, slots: &[ExitSlot], exit_area: Value) {
        for slot in slots {
            if matches!(slot.ty, TraceType::Nil | TraceType::Any) {
                continue;
            }
            let reloading = mem::replace(&mut self.reload_inputs, true);
            let value = self.value(slot.value);
            self.reload_inputs = reloading;
            self.store_slot_value(slot, value, exit_area);
        }
    }

    /// Stores `value`, which is what `slot` holds, into the exit area at
    /// `exit_area`.
    fn store_slot_value(&mut self, slot: &ExitSlot, value: Value, exit_area: Value) {
        let bits = match slot.ty {
            // A float goes to the exit area as it is: the store writes the
            // same eight bytes as the bitcast and the integer store it
            // replaces, with one machine instruction in place of two and one
            // instruction less for the compiler to lower.
            TraceType::Float
            | TraceType::Integer
            | TraceType::Int64
            | TraceType::UInt64
            | TraceType::Table
            | TraceType::Closure
            | TraceType::Thread
            | TraceType::Userdata
            | TraceType::String
            | TraceType::StringRef => value,
            TraceType::Boolean => self.builder.ins().uextend(types::I64, value),
            TraceType::Nil | TraceType::Any => return,
        };
        self.builder.ins().store(
            MemFlags::new(),
            bits,
            exit_area,
            offset(slot.slot as usize * mem::size_of::<i64>()),
        );
    }
}
