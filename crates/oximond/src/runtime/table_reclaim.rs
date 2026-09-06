//! Releases large, empty table allocations away from incremental GC slices.
//!
//! Values are drained on the runtime thread first. This worker never runs Lua
//! finalizers or accesses an arena. At most two tables can wait behind the
//! active release; a full queue leaves ownership with the collector for retry.

use std::sync::{
    OnceLock,
    mpsc::{self, SyncSender, TrySendError},
};

use crate::runtime::table_storage::Table;
use crate::runtime::value::Value;

const MIN_DEFERRED_BYTES: usize = 256 * 1024;
const QUEUE_CAPACITY: usize = 2;
static RECLAIMER: OnceLock<Option<SyncSender<Table>>> = OnceLock::new();

#[inline(always)]
pub(super) fn eligible(table: &Table) -> bool {
    let array = table
        .array
        .capacity()
        .saturating_mul(std::mem::size_of::<Value>());
    // A shaped table's entries are a few values, and counting the ones it
    // holds costs more than the rest of this check. The sweep asks for every
    // dead table, so a shaped table is weighed only when its array part
    // comes within those few values of the limit.
    if table.entries.is_shaped()
        && array
            < MIN_DEFERRED_BYTES
                - crate::runtime::table_storage::entries::SHAPED_MAX_ALLOCATED_BYTES
    {
        return false;
    }
    array.saturating_add(table.entries.allocated_bytes()) >= MIN_DEFERRED_BYTES
}

fn sender() -> Option<&'static SyncSender<Table>> {
    RECLAIMER
        .get_or_init(|| {
            let (sender, receiver) = mpsc::sync_channel::<Table>(QUEUE_CAPACITY);
            std::thread::Builder::new()
                .name("oximond-reclaim".into())
                .spawn(move || {
                    while let Ok(table) = receiver.recv() {
                        drop(table);
                    }
                })
                .ok()
                .map(|_| sender)
        })
        .as_ref()
}

/// Start before draining a large table, rather than on its final slice.
pub(super) fn prepare(table: &Table) {
    if eligible(table) {
        let _ = sender();
    }
}

/// Returns ownership on backpressure so no large allocation is freed inline.
pub(super) fn release(table: Table) -> Result<(), Table> {
    debug_assert!(table.array.is_empty());
    debug_assert!(table.entries.is_empty());
    debug_assert!(table.deleted_array_keys.is_empty(table.array.len()));
    debug_assert!(table.deleted_entries.is_empty());
    if let Some(sender) = sender() {
        send(sender, table)
    } else {
        // Resource exhaustion must not prevent collection from completing.
        drop(table);
        Ok(())
    }
}

fn send(sender: &SyncSender<Table>, table: Table) -> Result<(), Table> {
    match sender.try_send(table) {
        Ok(()) => Ok(()),
        Err(TrySendError::Full(table)) => Err(table),
        Err(TrySendError::Disconnected(table)) => {
            // There are no user callbacks in the worker, but retain a safe
            // synchronous fallback if it ever becomes unavailable.
            drop(table);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_queue_retains_storage_for_a_later_slice() {
        let (sender, receiver) = mpsc::sync_channel(1);
        let mut first = Table::default();
        first
            .array
            .reserve(MIN_DEFERRED_BYTES.div_ceil(std::mem::size_of::<Value>()));
        assert!(eligible(&first));
        assert!(send(&sender, first).is_ok());
        let mut second = Table::default();
        second
            .array
            .reserve(MIN_DEFERRED_BYTES.div_ceil(std::mem::size_of::<Value>()));
        let capacity = second.array.capacity();
        let Err(second) = send(&sender, second) else {
            panic!("full queue must leave storage with the collector");
        };
        assert_eq!(second.array.capacity(), capacity);
        drop(receiver.recv().unwrap());
        assert!(send(&sender, second).is_ok());
        assert_eq!(receiver.recv().unwrap().array.capacity(), capacity);
    }

    #[test]
    fn unavailable_worker_does_not_stall_collection() {
        let (sender, receiver) = mpsc::sync_channel(1);
        drop(receiver);
        assert!(send(&sender, Table::default()).is_ok());
    }
}
