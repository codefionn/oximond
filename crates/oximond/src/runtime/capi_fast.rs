//! Table reads and writes for the C API's string-keyed paths (`lua_getfield`,
//! `lua_setfield`, `lua_gettable` with a string key).
//!
//! Each method answers the common case from borrowed data and declines
//! everything else with `None` or the unconsumed value, so the C API's general
//! path (which owns the error messages and metamethod calls) stays the only
//! place that handles an unusual table.

use crate::runtime::error::RuntimeError;
use crate::runtime::metatable::Metamethod;
use crate::runtime::runtime::Runtime;
use crate::runtime::value::{TableHandle, Value};
use crate::string::LuaString;

/// How many `__index` tables a read follows before it declines. The general
/// path allows 100; a chain this deep is not a record or a class.
const INDEX_HOPS: u8 = 8;

/// What [`Runtime::capi_set_string`] did with its value.
pub enum StringSet {
    /// The value is stored.
    Done,
    /// Nothing changed: a metamethod or an error is possible, so the general
    /// path takes the value back.
    Declined(Value),
    /// The store ran and failed.
    Failed(RuntimeError),
}

impl Runtime {
    /// `table[key]` for a string `key` as `lua_gettable` reads it, borrowed:
    /// a present value, nil for an absent key of a table with no metatable,
    /// or the value an `__index` chain of tables holds. `None` when a handler
    /// function, a stale handle or a long chain needs the general path.
    #[inline]
    #[must_use]
    pub fn capi_get_string(&self, table: TableHandle, key: &LuaString) -> Option<&Value> {
        static NIL: Value = Value::Nil;
        let target = self.table(table).ok()?;
        match target.entries.get_string(key) {
            Some(value) if !matches!(value, Value::Nil) => return Some(value),
            _ => {}
        }
        if target.metatable.is_none() {
            return Some(&NIL);
        }
        self.capi_get_string_inherited(table, key)
    }

    /// The `__index` chain of [`Self::capi_get_string`], for a table whose own
    /// slot is empty and has a metatable.
    ///
    /// A chain that ends without the key answers nil here too: the last
    /// table has no metatable, or its metatable has no `__index`. A host
    /// that reads the optional fields of a record with a class behind it
    /// (Luanti's item and node definitions) asks for more absent keys than
    /// present ones, and the general path walked the same chain again after
    /// making a string of the key. Only an `__index` that is no table (a
    /// handler function, or a value with a metatable of its own) and a chain
    /// longer than [`INDEX_HOPS`] decline.
    #[cold]
    #[inline(never)]
    fn capi_get_string_inherited<'a>(
        &'a self,
        table: TableHandle,
        key: &LuaString,
    ) -> Option<&'a Value> {
        static NIL: Value = Value::Nil;
        let mut receiver = table;
        for _ in 0..=INDEX_HOPS {
            let current = self.table(receiver).ok()?;
            if let Some(value) = current
                .entries
                .get_string(key)
                .filter(|value| !matches!(value, Value::Nil))
            {
                return Some(value);
            }
            let Some(metatable) = current.metatable else {
                return Some(&NIL);
            };
            // Metatables are read raw: a stored nil is an absent `__index`.
            match self
                .table(metatable)
                .ok()?
                .entries
                .get_string(&self.index_metamethod_key)
            {
                Some(Value::Table(next)) => receiver = *next,
                None | Some(Value::Nil) => return Some(&NIL),
                Some(_) => return None,
            }
        }
        None
    }

    /// `table[n]` for a positive integer `n` read from the array part: a
    /// non-nil slot, borrowed. `None` for a hole, a hash-part key, or a stale
    /// handle.
    #[inline]
    #[must_use]
    pub fn capi_get_array(&self, table: TableHandle, n: i64) -> Option<&Value> {
        self.table_array_slot(table, n)
            .filter(|value| !matches!(value, Value::Nil))
    }

    /// `table[key] = value` for a string `key`, when no `__newindex` can
    /// apply: the key already holds a value, or the table has no metatable
    /// or one without `__newindex`. A readonly table is declined so the
    /// general path reports it.
    pub fn capi_set_string(
        &mut self,
        table: TableHandle,
        key: &LuaString,
        value: Value,
    ) -> StringSet {
        let Ok(target) = self.table(table) else {
            return StringSet::Declined(value);
        };
        if target.readonly {
            return StringSet::Declined(value);
        }
        let present = target
            .entries
            .get_string(key)
            .is_some_and(|held| !matches!(held, Value::Nil));
        if !present
            && let Some(metatable) = target.metatable
            && !self.metatable_lacks(metatable, Metamethod::NewIndex)
        {
            return StringSet::Declined(value);
        }
        match self.table_set(table, &Value::String(key.clone()), value) {
            Ok(()) => StringSet::Done,
            Err(error) => StringSet::Failed(error),
        }
    }

    /// [`Self::table_overwrite_scalar_string`] for a caller that treats a
    /// stale handle as a decline.
    #[inline]
    pub fn capi_overwrite_scalar(
        &mut self,
        table: TableHandle,
        key: &LuaString,
        value: &Value,
    ) -> bool {
        self.table_overwrite_scalar_string(table, key, value)
            .unwrap_or(false)
    }
}
