//! Cache for the handler a metatable has under an operator metamethod event.
//!
//! An arithmetic, comparison, concatenation or length operator on a table
//! looks its handler up in the table's metatable by the event's key string.
//! The cache holds what that lookup found, keyed on the metatable and the
//! event number, so the next operation of any object of the class takes a
//! compare and a load. It shares the invalidation of the inherited-read
//! cache (see `oximond::runtime::inherit`): the metatable is marked `watched`, and every
//! mutable borrow of it, every store over one of its closures and the
//! collector clearing its weak entries advance [`Runtime::inherit_epoch`],
//! which drops every entry at once. A store that adds the event key to a
//! metatable that lacks it is such a borrow, or advances the epoch itself.

use crate::runtime::runtime::Runtime;
use crate::runtime::value::{ClosureHandle, TableHandle, Value};
use crate::string::LuaString;

/// Events the cache holds: the numbers of `oximond::vm`'s operator events.
pub const OPERATOR_EVENTS: usize = 32;

/// Metatables that share an event's entries before they evict each other.
const WAYS: usize = 4;

/// What a metatable holds under an event's key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OperatorHandler {
    /// No value, or a stored nil.
    Absent,
    /// A Lua closure.
    Closure(ClosureHandle),
    /// Any other value, which the caller reads itself.
    Other,
}

/// One cache entry. An `epoch` of zero is an empty entry.
#[derive(Clone)]
pub(crate) struct OperatorEntry {
    epoch: u64,
    metatable: Option<TableHandle>,
    handler: OperatorHandler,
}

impl Default for OperatorEntry {
    fn default() -> Self {
        Self {
            epoch: 0,
            metatable: None,
            handler: OperatorHandler::Other,
        }
    }
}

#[allow(clippy::inline_always)]
#[inline(always)]
fn entry_index(metatable: TableHandle, event: u8) -> usize {
    (usize::from(event) % OPERATOR_EVENTS) * WAYS + (metatable.0.slot as usize & (WAYS - 1))
}

impl Runtime {
    /// What `metatable` holds under the operator `event`, when a valid
    /// cache entry knows. `None` means the entry is empty, stale or for
    /// another metatable, and the caller resolves it.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn operator_handler_cached(
        &self,
        metatable: TableHandle,
        event: u8,
    ) -> Option<OperatorHandler> {
        let entry = self.operator_cache.get(entry_index(metatable, event))?;
        (entry.epoch == self.inherit_epoch && entry.metatable == Some(metatable))
            .then_some(entry.handler)
    }

    /// [`Self::operator_handler_cached`] that reads the metatable on a miss
    /// of the cache, under `key`, the string of the event's name, and records
    /// what it found for a closure or an absent value. A stale handle reads
    /// as `Other`, which the caller's own lookup reports.
    pub fn operator_handler(
        &mut self,
        metatable: TableHandle,
        event: u8,
        key: &LuaString,
    ) -> OperatorHandler {
        if let Some(handler) = self.operator_handler_cached(metatable, event) {
            return handler;
        }
        let handler = match self.table_get_string_ref(metatable, key) {
            None => OperatorHandler::Absent,
            Some(&Value::Closure(closure)) => OperatorHandler::Closure(closure),
            Some(_) => return OperatorHandler::Other,
        };
        if self.operator_cache.is_empty() {
            self.operator_cache = vec![OperatorEntry::default(); OPERATOR_EVENTS * WAYS];
        }
        if let Some(table) = self.tables.get_mut(metatable.0) {
            table.watched = true;
        }
        self.operator_cache[entry_index(metatable, event)] = OperatorEntry {
            epoch: self.inherit_epoch,
            metatable: Some(metatable),
            handler,
        };
        handler
    }
}
