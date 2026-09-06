//! The browser target has no executable-memory allocator for Cranelift JIT
//! code, so traces never compile there.

use super::compile::{
    ExitPlan, Extra, LoweredMember, TraceCompileError, TraceCompileOptions, Writeback,
};
use super::exit::{ExitDescriptor, TraceControl, TraceInputSlot};
use crate::jit::OptimizationTier;
use crate::jit_ir::trace::{LinkWrite, Snapshot, TraceIr, TraceType};

pub(super) struct TraceCode {
    pub(super) native_link: bool,
}

impl TraceCode {
    /// The backend that made the code, for the measurement lines.
    pub(super) const fn backend(&self) -> &'static str {
        "wasm"
    }

    /// Code that holds no function (see the native backend's `staged`).
    pub(super) const fn staged() -> Self {
        Self { native_link: false }
    }

    pub(super) fn run(
        &self,
        _inputs: *mut TraceInputSlot,
        _control: &mut TraceControl,
        _exit_area: &mut [i64],
    ) -> i64 {
        unreachable!("wasm cannot compile native code")
    }

    pub(super) const fn code_size(&self) -> usize {
        0
    }

    pub(super) const fn inline_runs(&self) -> usize {
        0
    }

    pub(super) const fn clif_insts(&self) -> usize {
        0
    }

    pub(super) fn entry_address(&self) -> usize {
        0
    }

    pub(super) fn side_extras(&self) -> &[Extra] {
        &[]
    }

    pub(super) fn side_writebacks(&self) -> &[Writeback] {
        &[]
    }
}

/// See the native backend's `DetachedSpec`.
pub(super) struct DetachedSpec<'a> {
    pub(super) code_base: i64,
    pub(super) host: &'a ExitDescriptor,
    pub(super) host_snapshot: &'a Snapshot,
    pub(super) root_entry: &'a [(u16, TraceType)],
    pub(super) exit_style: bool,
    pub(super) row_masked: bool,
    pub(super) native_probe: bool,
    pub(super) extras: &'a [Extra],
    pub(super) link_writes: &'a [LinkWrite],
    pub(super) guest_registers: usize,
}

pub(super) const SIDE_CODE_SHIFT: u32 = 24;
pub(super) const LINK_CODE: i64 = i64::MIN + 1;
pub(super) const DECLINE_CODE: i64 = i64::MIN + 2;

/// Never called: no side compiles.
pub(super) fn call_side(
    _address: usize,
    _inputs: *mut TraceInputSlot,
    _control: &mut TraceControl,
    _exit_area: *mut i64,
) -> i64 {
    unreachable!("wasm cannot compile native code")
}

pub(super) fn copy_inputs(_inputs: *const TraceInputSlot, _count: usize) -> Vec<TraceInputSlot> {
    unreachable!("wasm cannot compile native code")
}

pub(super) fn read_input(_inputs: *const TraceInputSlot, _position: usize) -> TraceInputSlot {
    unreachable!("wasm cannot compile native code")
}

pub(super) fn write_input(_inputs: *mut TraceInputSlot, _position: usize, _slot: TraceInputSlot) {
    unreachable!("wasm cannot compile native code")
}

pub(super) fn rows_masked(_members: &[&TraceIr]) -> bool {
    false
}

pub(super) fn has_trusted_key(_members: &[LoweredMember<'_>]) -> bool {
    false
}

pub(super) fn lower(
    _members: &[LoweredMember<'_>],
    _plan: &ExitPlan,
    _tier: OptimizationTier,
    _forced_guard: Option<(usize, usize)>,
    _options: TraceCompileOptions,
    _detached: Option<&DetachedSpec<'_>>,
) -> Result<TraceCode, TraceCompileError> {
    Err(TraceCompileError::NativeBackendRejected)
}

/// What [`TraceCode::run`] returns when the trace returned.
pub(super) const RETURNED_CODE: i64 = i64::MIN;

/// No trace compiles, so none calls a closure through closure rows.
pub(super) fn closure_row_words(_members: &[&crate::jit_ir::trace::TraceIr]) -> usize {
    0
}

/// Never called: no native code exists to call it.
pub(super) extern "C" fn trace_host_call() {
    unreachable!("wasm cannot compile native code")
}

/// Never called: no native code exists to call it.
pub(super) extern "C" fn trace_table_set() {
    unreachable!("wasm cannot compile native code")
}
