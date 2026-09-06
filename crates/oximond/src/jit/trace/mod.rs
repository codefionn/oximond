//! Native execution of verified trace IR.
//!
//! [`compile_trace`] lowers an owned, verified [`oximond::jit_ir::trace::TraceIr`]
//! through Cranelift and returns a [`CompiledTrace`]. The compiler never reads
//! a live VM: its only input is the IR, so a worker thread can run it.
//!
//! - [`compile`] validates the IR, plans the tree of a root and its side
//!   traces, optimizes each member, plans the exits and owns the result.
//! - [`emit`] lowers the body: entry guards, typed loads, the steady-state
//!   loop with its fuel poll, and an out-of-line stub per guard.
//! - [`exit`] defines the typed ABI between native code and the VM.
//!
//! # Execution model
//!
//! A trace is entered at its loop header with the interpreter's register
//! window described by one [`TraceInputSlot`] per entry input. Native code
//! checks each tag, then loops. Loop-carried values live in registers (block
//! parameters), and nothing is written back to the window while the loop
//! runs. Every way out is a guard stub, the poll stub or the terminator, each
//! storing the live values its snapshot names into the exit area. The VM
//! rebuilds the interpreter frame from the snapshot, so a trace that has
//! committed no effect can always be abandoned without losing state.
//!
//! A tree of traces (a root and the side traces recorded from its guards) is
//! lowered as one function: see [`emit`]. Its exits come from every member,
//! each naming a snapshot of its own member.

#[cfg(not(target_arch = "wasm32"))]
mod census;
mod compile;
#[cfg(not(target_arch = "wasm32"))]
mod emit;
#[cfg(target_arch = "wasm32")]
#[path = "emit_wasm.rs"]
mod emit;
mod exit;
mod helpers;
#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod libm_lookup;
mod made_table;
mod member_cache;
mod mir;
mod prepared;

pub use compile::{
    AttachError, CompiledSide, CompiledTrace, Extra, ExtraSource, SideContext, TraceCompileError,
    TraceCompileOptions, Writeback, compile_side, compile_side_with_plan, compile_trace,
    compile_trace_forcing_guard, compile_trace_forcing_member_guard, compile_trace_with,
    compile_trace_with_plan,
};
pub use exit::{
    CLOSURE_ROW_CELLS, CLOSURE_ROW_TABLE_WORDS, CLOSURE_ROW_WORDS, ExitDescriptor, ExitSlot,
    HostEntry, MAX_FUNCTION_TABLE_LANES, MAX_TRACE_STRING_LANES, MAX_TRACE_TABLE_LANES,
    TRACE_LANE_INDEX_PLACES, TRACE_PENDING_STRINGS, TRACE_TAG_BOOLEAN, TRACE_TAG_BORROWED_STRING,
    TRACE_TAG_CLOSURE, TRACE_TAG_FLOAT, TRACE_TAG_INT64, TRACE_TAG_INTEGER, TRACE_TAG_NIL,
    TRACE_TAG_OTHER, TRACE_TAG_STRING, TRACE_TAG_TABLE, TRACE_TAG_THREAD, TRACE_TAG_UINT64,
    TraceBinding, TraceControl, TraceExitClass, TraceGuardClass, TraceInputSlot, TraceOutcome,
    TraceTableSpace, TraceTables, row_stride_of, seal_row,
};
pub(crate) use exit::{entry_tag, fill_sealed};
pub use helpers::{
    helper_address, install_helper, install_prepared_helper, prepared_helper_address,
};
pub use mir::{Counters as MirCounters, counters as mir_counters};
pub use prepared::Prepared;

#[cfg(test)]
mod closure_cell_tests;
#[cfg(test)]
mod function_tests;
#[cfg(test)]
mod hash_tests;
#[cfg(test)]
mod helper_tests;
#[cfg(test)]
mod meta_guard_tests;
#[cfg(test)]
mod new_table_tests;
#[cfg(test)]
mod resume_tests;
#[cfg(test)]
mod store_tests;
#[cfg(test)]
mod string_tests;
#[cfg(test)]
mod table_rows_tests;
#[cfg(test)]
mod table_tests;
#[cfg(test)]
mod table_tree_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tree_tests;
