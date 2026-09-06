//! Heap reference classification and per-object byte accounting for the collector.

use crate::runtime::closures::ClosureBody;
use crate::runtime::files::{FileStorage, RuntimeFile};
use crate::runtime::gc::{GcReference, RuntimeThread};
use crate::runtime::iterators::GmatchIterator;
use crate::runtime::table_storage::Table;
use crate::runtime::table_storage::key::TableKey;
use crate::runtime::value::{CellHandle, ClosureHandle, Value};

#[inline]
pub(crate) fn value_gc_reference(value: &Value) -> Option<GcReference> {
    match value {
        Value::Table(handle) => Some(GcReference::Table(*handle)),
        Value::File(handle) => Some(GcReference::File(*handle)),
        Value::Userdata(handle) => Some(GcReference::Userdata(*handle)),
        Value::GmatchIterator(handle) => Some(GcReference::GmatchIterator(*handle)),
        Value::Closure(handle) => Some(GcReference::Closure(*handle)),
        Value::Thread(handle) | Value::CoroutineWrapper(handle) => {
            Some(GcReference::Thread(*handle))
        }
        Value::Nil
        | Value::Boolean(_)
        | Value::Integer(_)
        | Value::Number(_)
        | Value::String(_)
        | Value::NativeFunction(_)
        | Value::BuiltinFunction(_)
        | Value::Int64(_)
        | Value::UInt64(_) => None,
    }
}

/// Narrows a table key to its collectable reference without cloning it.
#[inline]
pub(crate) fn key_gc_reference(key: &TableKey) -> Option<GcReference> {
    match key {
        TableKey::Table(handle) => Some(GcReference::Table(*handle)),
        TableKey::File(handle) => Some(GcReference::File(*handle)),
        TableKey::Userdata(handle) => Some(GcReference::Userdata(*handle)),
        TableKey::GmatchIterator(handle) => Some(GcReference::GmatchIterator(*handle)),
        TableKey::Closure(handle) => Some(GcReference::Closure(*handle)),
        TableKey::Thread(handle) | TableKey::CoroutineWrapper(handle) => {
            Some(GcReference::Thread(*handle))
        }
        TableKey::Boolean(_)
        | TableKey::Integer(_)
        | TableKey::Number(_)
        | TableKey::String(_)
        | TableKey::NativeFunction(_)
        | TableKey::BuiltinFunction(_)
        | TableKey::Int64(_)
        | TableKey::UInt64(_) => None,
    }
}

pub(crate) fn table_accounted_bytes(table: &Table) -> usize {
    crate::runtime::table_storage::TABLE_CHARGED_BYTES
        + table.array.capacity() * std::mem::size_of::<Value>()
        + table.entries.allocated_bytes()
        + table.deleted_array_keys.allocated_bytes()
        + table.deleted_entries.allocated_bytes()
}

pub(crate) fn closure_accounted_bytes(closure: &ClosureBody) -> usize {
    std::mem::size_of::<ClosureBody>() + closure.cells.len() * std::mem::size_of::<CellHandle>()
}

pub(crate) fn file_accounted_bytes(file: &RuntimeFile) -> usize {
    std::mem::size_of::<RuntimeFile>()
        + match &file.storage {
            FileStorage::Memory { bytes, .. } => bytes.len(),
            FileStorage::Named { pending, .. } => pending.len(),
            FileStorage::PipeRead { .. } | FileStorage::PipeWrite { .. } => 0,
            FileStorage::HostInput
            | FileStorage::HostOutput
            | FileStorage::HostError
            | FileStorage::Closed => 0,
        }
}

pub(crate) fn gmatch_iterator_accounted_bytes(iterator: &GmatchIterator) -> usize {
    std::mem::size_of::<GmatchIterator>() + iterator.source.len() + iterator.pattern.len()
}

pub(crate) fn thread_accounted_bytes(thread: &RuntimeThread) -> usize {
    std::mem::size_of::<RuntimeThread>()
        + (thread.roots.references.len() + thread.host_roots.references.len())
            * std::mem::size_of::<GcReference>()
        + (thread.roots.cells.len() + thread.host_roots.cells.len())
            * std::mem::size_of::<CellHandle>()
        + (thread.roots.closures.len() + thread.host_roots.closures.len())
            * std::mem::size_of::<ClosureHandle>()
}

pub(crate) fn scale_gc_work(bytes: usize, percent: f64) -> usize {
    if !percent.is_finite() || percent <= 0.0 {
        return 1;
    }
    let bytes = bytes.to_string().parse::<f64>().unwrap_or(f64::INFINITY);
    let scaled = (bytes * (percent / 100.0)).floor();
    scaled
        .to_string()
        .parse::<usize>()
        .unwrap_or(usize::MAX)
        .max(1)
}
