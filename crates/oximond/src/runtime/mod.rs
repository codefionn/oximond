//! Runtime-owned values, tables, globals, and native functions.
#![allow(
    clippy::items_after_statements,
    clippy::collapsible_if,
    clippy::manual_unwrap_or,
    clippy::match_same_arms,
    clippy::obfuscated_if_else,
    clippy::option_if_let_else,
    clippy::option_option,
    clippy::doc_markdown,
    clippy::semicolon_if_nothing_returned,
    clippy::too_many_lines,
    clippy::unnecessary_lazy_evaluations
)]

pub mod alloc_phase;
mod arena;
mod capi_fast;
pub mod child_limits;
pub mod cint64;
mod clock;
mod closures;
mod error;
mod files;
mod gc;
mod helpers;
mod host;
mod inherit;
mod iterators;
mod metatable;
mod native_views;
mod number_format;
#[cfg(feature = "op-stats")]
pub mod op_stats;
mod operator_cache;
mod prefetch;
// Deliberate inception: `runtime/runtime.rs` holds the `Runtime` type, one
// file per concern like the rest of this tree.
#[allow(clippy::module_inception)]
mod runtime;
mod table_reclaim;
mod table_storage;
pub(crate) use table_storage::key_layout as table_key_layout;
mod tables;
mod userdata;
mod value;

pub use crate::string::{
    NATIVE_STRING_HASH_OFFSET, NATIVE_STRING_INLINE_TAG, NATIVE_STRING_INLINE_WORDS,
    native_inline_words,
};
pub use capi_fast::StringSet;
pub use closures::{
    CaptureCells, ClosureCallParts, ClosureCells, ClosureSnapshot, NO_SHAPE, StackAnchor,
};
pub use error::{ExitRequest, RuntimeError, RuntimeResult};
pub use files::{FileOpenMode, LinesFormats};
pub use gc::{GcMode, GcRootBuilder, GcRoots, GcStats, GcStepResult};
pub use helpers::{NonFiniteNumerals, io_error_text, parse_lua_integer_exact};
pub(crate) use helpers::{PlainNumeral, io_error_code, scan_plain_numeral};
pub use iterators::GmatchIteratorSnapshot;
pub use metatable::Metamethod;
pub use native_views::{
    NATIVE_ABSENT_FIELD, NATIVE_BOOLEAN_GROWTH_REACH, NATIVE_CURSOR_SLOT_KEY_BITS,
    NATIVE_CURSOR_SLOT_KEY_TAG, NATIVE_CURSOR_SLOT_SIZE, NATIVE_CURSOR_SLOT_VALUE,
    NATIVE_PROBE_SHIFT, NATIVE_PROBE_SLOT_HASH, NATIVE_PROBE_SLOT_KEY, NATIVE_PROBE_SLOT_OWNER,
    NATIVE_PROBE_SLOT_SHIFT, NATIVE_PROBE_SLOT_SIZE, NATIVE_PROBE_SLOT_VALUE,
    NATIVE_STRING_PROBE_START_OFFSET, NATIVE_STRING_STORAGE_OFFSET, NATIVE_VALUE_PAYLOAD_OFFSET,
    NATIVE_VALUE_SIZE, NATIVE_VALUE_TAG_BOOLEAN, NATIVE_VALUE_TAG_CLOSURE,
    NATIVE_VALUE_TAG_INTEGER, NATIVE_VALUE_TAG_NIL, NATIVE_VALUE_TAG_NUMBER,
    NATIVE_VALUE_TAG_STRING, NATIVE_VALUE_TAG_TABLE, NATIVE_VALUE_TAG_THREAD, NativeBytes,
    NativeCursor, NativeCursorScratch, NativeCursorSlot, NativeProbeSlot, NativeRowToken,
    NativeRowWords, NativeSlotKind, NativeStringProbe, NativeStringProbeScratch, NativeTableHeader,
    NativeTableLayout, NativeValueArray, NativeValueSlot, native_probe_start, native_table_reserve,
};
pub use number_format::{
    NumberFormat, NumberText, StackBuffer, format_integer, push_general, push_hex_float,
    small_integer_string,
};
pub use operator_cache::{OPERATOR_EVENTS, OperatorHandler};
pub use runtime::{HostRootsRefresher, JitOptions, NO_TABLE_IDENTITY, Runtime, WarningHandler};
pub use table_storage::entries::StringReadCache;
pub use table_storage::entries::{SHAPE_CACHE_LINES, shape_cache_len};
pub use table_storage::shape::{MAX_SHAPE_KEYS, TableShape};
pub use table_storage::{NextKey, NextRef};
pub use tables::{DenseScalarStore, SiteStore};
pub use tables::{FoundTable, SiteProbe};
pub use userdata::Finalizer;
pub(crate) use value::NativeWideBinary;
pub use value::{
    BuiltinFunction, CellHandle, ClosureHandle, DirectNativeKind, FileHandle, GmatchIteratorHandle,
    NATIVE_RESULTS_CAPACITY, NativeCallKind, NativeCallback, NativeFixedResultsProbe,
    NativeFunctionHandle, NativeNumericBinary, NativeNumericUnary, NativePair, NativePairCallback,
    NativePlan, NativeProbe, NativeResults, NativeResultsProbe, NativeScalarCallback, TableHandle,
    ThreadHandle, UserdataHandle, Value,
};

#[cfg(test)]
mod tests;
