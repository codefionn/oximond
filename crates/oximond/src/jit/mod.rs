#![deny(unsafe_code)]
// Native-only helpers remain compiled on wasm to keep the public interpreter
// plan API identical, but have no callers when code generation is disabled.
#![cfg_attr(target_arch = "wasm32", allow(dead_code, unused_imports))]

//! Native code for Oximond: the trace compiler and the inline table-call
//! plans the interpreter evaluates.
//!
//! Cranelift types and its native calling convention stay private to this
//! crate.

mod broker;
pub mod compile_dump;
pub(crate) mod debug_env;
pub(crate) mod icount;
mod inline_table;
mod trace;

pub use broker::{
    CompilationId, CompileOutput, CompletedTrace, DEFAULT_PENDING_TRACE_BYTES, JitBroker, SideJob,
    SubmitError,
};
pub use inline_table::{
    PreparedInlineTableCallPlan, evaluate_inline_table_call, evaluate_inline_table_call_number,
};
pub use trace::{
    AttachError, CLOSURE_ROW_CELLS, CLOSURE_ROW_TABLE_WORDS, CLOSURE_ROW_WORDS, CompiledSide,
    CompiledTrace, ExitDescriptor, ExitSlot, Extra, ExtraSource, HostEntry,
    MAX_FUNCTION_TABLE_LANES, MAX_TRACE_STRING_LANES, MAX_TRACE_TABLE_LANES, Prepared, SideContext,
    TRACE_LANE_INDEX_PLACES, TRACE_PENDING_STRINGS, TRACE_TAG_BOOLEAN, TRACE_TAG_BORROWED_STRING,
    TRACE_TAG_CLOSURE, TRACE_TAG_FLOAT, TRACE_TAG_INT64, TRACE_TAG_INTEGER, TRACE_TAG_NIL,
    TRACE_TAG_OTHER, TRACE_TAG_STRING, TRACE_TAG_TABLE, TRACE_TAG_THREAD, TRACE_TAG_UINT64,
    TraceBinding, TraceCompileError, TraceCompileOptions, TraceControl, TraceExitClass,
    TraceGuardClass, TraceInputSlot, TraceOutcome, TraceTableSpace, TraceTables, Writeback,
    compile_side, compile_side_with_plan, compile_trace, compile_trace_forcing_guard,
    compile_trace_forcing_member_guard, compile_trace_with, compile_trace_with_plan,
    helper_address, install_helper, install_prepared_helper, prepared_helper_address,
    row_stride_of, seal_row,
};
pub use trace::{MirCounters, mir_counters};
pub(crate) use trace::{entry_tag, fill_sealed};

/// The Cranelift optimization policy used for one compilation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OptimizationTier {
    /// Minimize compilation work for code that has only recently become hot.
    Baseline,
    /// Spend more compilation work on code expected to stay hot: the
    /// backtracking register allocator over the code the trace optimizer
    /// already cleaned up.
    Optimized,
    /// [`Self::Optimized`] and Cranelift's own optimizer on top, for code
    /// that has stayed hot for long enough to repay the longer compile.
    Peak,
}

#[cfg(test)]
mod tests {
    #[test]
    #[allow(unsafe_code)]
    fn native_value_layout_matches_the_published_tags() {
        use crate::runtime::{
            NATIVE_VALUE_PAYLOAD_OFFSET, NATIVE_VALUE_TAG_BOOLEAN, NATIVE_VALUE_TAG_INTEGER,
            NATIVE_VALUE_TAG_NIL, NATIVE_VALUE_TAG_NUMBER, NATIVE_VALUE_TAG_STRING,
            NATIVE_VALUE_TAG_TABLE, TableHandle, Value,
        };
        let tag = |value: &Value| {
            // SAFETY: `Value` is `repr(C, u8)`, so its first byte is the
            // initialized variant tag.
            unsafe { *std::ptr::from_ref(value).cast::<u8>() }
        };
        assert_eq!(tag(&Value::Nil), NATIVE_VALUE_TAG_NIL);
        assert_eq!(tag(&Value::Boolean(true)), NATIVE_VALUE_TAG_BOOLEAN);
        assert_eq!(tag(&Value::Integer(-9)), NATIVE_VALUE_TAG_INTEGER);
        assert_eq!(tag(&Value::Number(1.5)), NATIVE_VALUE_TAG_NUMBER);
        assert_eq!(tag(&Value::from("text")), NATIVE_VALUE_TAG_STRING);
        assert_eq!(
            tag(&Value::Table(TableHandle::from_native_identity(3))),
            NATIVE_VALUE_TAG_TABLE
        );
        let offset = usize::try_from(NATIVE_VALUE_PAYLOAD_OFFSET).unwrap();
        let number = Value::Number(1.5);
        // SAFETY: the payload of a `Number` is an initialized f64 at the offset.
        let payload = unsafe {
            std::ptr::from_ref(&number)
                .cast::<u8>()
                .add(offset)
                .cast::<f64>()
                .read()
        };
        assert_eq!(payload, 1.5);
        let integer = Value::Integer(-9);
        // SAFETY: the payload of an `Integer` is an initialized i64 at the offset.
        let payload = unsafe {
            std::ptr::from_ref(&integer)
                .cast::<u8>()
                .add(offset)
                .cast::<i64>()
                .read()
        };
        assert_eq!(payload, -9);
        let boolean = Value::Boolean(true);
        // SAFETY: the payload of a `Boolean` is an initialized bool at the offset.
        let payload = unsafe { std::ptr::from_ref(&boolean).cast::<u8>().add(offset).read() };
        assert_eq!(payload, 1);
    }
}
