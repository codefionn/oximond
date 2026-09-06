//! Individual and per-type shared metatables.

use crate::runtime::error::{RuntimeError, RuntimeResult};
use crate::runtime::gc::GcReference;
use crate::runtime::runtime::Runtime;
use crate::runtime::table_storage::Table;
use crate::runtime::value::{ClosureHandle, NativeFunctionHandle, TableHandle, Value};
use crate::string::LuaString;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub(crate) enum SharedMetatable {
    Nil,
    Boolean,
    Number,
    String,
    Function,
    File,
    Thread,
    Cdata,
}

impl SharedMetatable {
    /// The number of kinds, which index [`Runtime`]'s table of them.
    pub(crate) const COUNT: usize = 8;
}

pub(crate) fn shared_metatable_kind(value: &Value) -> SharedMetatable {
    match value {
        Value::Nil => SharedMetatable::Nil,
        Value::Boolean(_) => SharedMetatable::Boolean,
        Value::Integer(_) | Value::Number(_) => SharedMetatable::Number,
        Value::String(_) => SharedMetatable::String,
        Value::File(_) => SharedMetatable::File,
        Value::Closure(_)
        | Value::NativeFunction(_)
        | Value::BuiltinFunction(_)
        | Value::GmatchIterator(_)
        | Value::CoroutineWrapper(_) => SharedMetatable::Function,
        Value::Thread(_) => SharedMetatable::Thread,
        Value::Int64(_) | Value::UInt64(_) => SharedMetatable::Cdata,
        Value::Table(_) | Value::Userdata(_) => unreachable!("individual metatable"),
    }
}

/// The metatable fields whose absence a table caches, so a store or a
/// `setmetatable` can tell they are missing with a bit test.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Metamethod {
    NewIndex,
    Gc,
    Metatable,
    /// `__mode`, which makes a table weak.
    Mode,
    /// `__index`, which a read of a missing key runs.
    Index,
    /// `__len`, which the length operator runs.
    Len,
}

impl Metamethod {
    /// Every event, in the order of their bits.
    pub(crate) const ALL: [Self; 6] = [
        Self::NewIndex,
        Self::Gc,
        Self::Metatable,
        Self::Mode,
        Self::Index,
        Self::Len,
    ];

    /// The bit of this event in a table's cache of absent metamethods and
    /// in the sets [`Runtime::metatable_lacks_all`] takes.
    #[must_use]
    pub(crate) const fn bit(self) -> u8 {
        1 << self as u8
    }
}

impl Runtime {
    /// The key of `event`'s metamethod, as the runtime holds it for its own
    /// lookups, so a caller that reads a metatable makes no string.
    #[must_use]
    pub fn metamethod_key(&self, event: Metamethod) -> &LuaString {
        match event {
            Metamethod::NewIndex => &self.newindex_metamethod_key,
            Metamethod::Gc => &self.gc_metamethod_key,
            Metamethod::Metatable => &self.metatable_metamethod_key,
            Metamethod::Mode => &self.mode_metamethod_key,
            Metamethod::Index => &self.index_metamethod_key,
            Metamethod::Len => &self.len_metamethod_key,
        }
    }

    /// Whether `metatable` holds no value under `event`'s key, read raw: a
    /// stored nil is absent. The answer is cached in the metatable until
    /// the next change to it, so a metatable without `__newindex` costs a
    /// bit test per store. `false` for a stale handle.
    #[inline]
    #[must_use]
    pub fn metatable_lacks(&self, metatable: TableHandle, event: Metamethod) -> bool {
        self.tables
            .get(metatable.0)
            .is_some_and(|table| self.table_lacks(table, event))
    }

    /// Whether `metatable` lacks every event of `events`, a set of
    /// [`Metamethod::bit`]s: [`Self::metatable_lacks`] of each, with one
    /// table lookup and, once every absence is cached, one test for the
    /// whole set. `false` for a stale handle.
    #[inline]
    #[must_use]
    pub(crate) fn metatable_lacks_all(&self, metatable: TableHandle, events: u8) -> bool {
        let Some(table) = self.tables.get(metatable.0) else {
            return false;
        };
        table.absent_metamethods.load(Ordering::Relaxed) & events == events
            || self.table_lacks_uncached(table, events)
    }

    /// [`Self::metatable_lacks_all`] while some absence is not cached: asks
    /// for each event, which caches the absences it finds.
    #[cold]
    fn table_lacks_uncached(&self, table: &Table, events: u8) -> bool {
        Metamethod::ALL
            .into_iter()
            .filter(|event| events & event.bit() != 0)
            .all(|event| self.table_lacks(table, event))
    }

    /// The `__newindex` of `metatable` as [`Self::metatable_lacks`] reads it,
    /// with the same absence cache. `None` when the metatable has no
    /// `__newindex`. `Some(None)` when it has one that is not a closure, or
    /// when the handle is stale.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn metatable_newindex(
        &self,
        metatable: TableHandle,
    ) -> Option<Option<ClosureHandle>> {
        let Some(table) = self.tables.get(metatable.0) else {
            return Some(None);
        };
        let bit = 1u8 << Metamethod::NewIndex as u8;
        let known = table.absent_metamethods.load(Ordering::Relaxed);
        if known & bit != 0 {
            return None;
        }
        // The scan is inlined: one lookup in a small metatable is the whole
        // cost of a store that `__newindex` takes.
        match table
            .entries
            .get_string_live_scan(self.metamethod_key(Metamethod::NewIndex))
        {
            None | Some(Value::Nil) => {
                // Only the thread that owns the runtime writes the byte.
                table
                    .absent_metamethods
                    .store(known | bit, Ordering::Relaxed);
                None
            }
            Some(&Value::Closure(closure)) => Some(Some(closure)),
            Some(_) => Some(None),
        }
    }

    /// [`Self::metatable_lacks`] for a table already looked up.
    #[inline]
    pub(crate) fn table_lacks(&self, table: &Table, event: Metamethod) -> bool {
        let bit = 1u8 << event as u8;
        let known = table.absent_metamethods.load(Ordering::Relaxed);
        if known & bit != 0 {
            return true;
        }
        let absent = table
            .entries
            .get_string(self.metamethod_key(event))
            .is_none_or(|value| matches!(value, Value::Nil));
        if absent {
            // Only the thread that owns the runtime writes the byte.
            table
                .absent_metamethods
                .store(known | bit, Ordering::Relaxed);
        }
        absent
    }

    /// Records the installed `setmetatable` native, which the fast loop
    /// recognises by handle and runs inline.
    pub fn set_setmetatable_native(&mut self, function: NativeFunctionHandle) {
        self.setmetatable_native = function.0;
    }

    /// Whether `function` is the installed `setmetatable` native. A program
    /// that rebinds the global holds another value and takes ordinary
    /// dispatch.
    #[inline(always)]
    #[must_use]
    pub fn is_setmetatable_native(&self, function: NativeFunctionHandle) -> bool {
        function.0 == self.setmetatable_native
    }

    /// Returns a value's unprotected metatable.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` has a stale table, userdata, or file handle.
    pub fn raw_metatable(&self, value: &Value) -> RuntimeResult<Option<TableHandle>> {
        match value {
            Value::Table(handle) => Ok(self.table(*handle)?.metatable),
            Value::Userdata(handle) => Ok(self.userdata(*handle)?.metatable),
            value => Ok(self.shared_metatable(value)),
        }
    }

    /// Returns a value's metatable while respecting a `__metatable` guard.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` has a stale table, userdata, or file handle.
    pub fn protected_metatable(&self, value: &Value) -> RuntimeResult<Value> {
        let Some(metatable) = self.raw_metatable(value)? else {
            return Ok(Value::Nil);
        };
        if self.metatable_lacks(metatable, Metamethod::Metatable) {
            return Ok(Value::Table(metatable));
        }
        let protected = self.table_get_string(metatable, &self.metatable_metamethod_key)?;
        Ok(if matches!(protected, Value::Nil) {
            Value::Table(metatable)
        } else {
            protected
        })
    }

    /// Changes a metatable without respecting a `__metatable` guard.
    ///
    /// # Errors
    ///
    /// Returns an error for stale value or metatable handles.
    pub fn raw_set_metatable(
        &mut self,
        value: &Value,
        metatable: Option<TableHandle>,
    ) -> RuntimeResult<()> {
        if let Some(metatable) = metatable {
            self.table(metatable)?;
        }
        match value {
            Value::Table(handle) => {
                let table = self.table_mut(*handle)?;
                table.ensure_writable()?;
                table.metatable = metatable;
                if let Some(metatable) = metatable {
                    self.register_table_finalizer(*handle, metatable)?;
                }
            }
            Value::Userdata(handle) => {
                self.userdata_mut(*handle)?.metatable = metatable;
                self.refresh_userdata_finalizer(*handle)?;
            }
            value => {
                let kind = shared_metatable_kind(value);
                self.shared_metatables[kind as usize] = metatable;
            }
        }
        if let Some(metatable) = metatable {
            let container = match value {
                Value::Table(handle) => Some(GcReference::Table(*handle)),
                Value::Userdata(handle) => Some(GcReference::Userdata(*handle)),
                _ => None,
            };
            self.gc_barrier(container, GcReference::Table(metatable));
        }
        Ok(())
    }

    /// Changes a table metatable while respecting a `__metatable` guard.
    ///
    /// # Errors
    ///
    /// Returns an error for stale handles, a protected table, or a readonly
    /// table.
    pub fn set_table_metatable(
        &mut self,
        table: TableHandle,
        metatable: Option<TableHandle>,
    ) -> RuntimeResult<()> {
        let entry = self.table(table)?;
        if let Some(current) = entry.metatable
            && !self.metatable_lacks(current, Metamethod::Metatable)
        {
            return Err(RuntimeError::message("cannot change a protected metatable"));
        }
        let mut finalizer = false;
        if let Some(metatable) = metatable {
            let Some(meta) = self.tables.get(metatable.0) else {
                return Err(RuntimeError::InvalidTable(metatable));
            };
            finalizer = self.table_finalizers_enabled && !self.table_lacks(meta, Metamethod::Gc);
        }
        entry.ensure_writable()?;
        self.table_mut(table)?.metatable = metatable;
        let Some(metatable) = metatable else {
            return Ok(());
        };
        if finalizer {
            self.register_table_finalizer(table, metatable)?;
        }
        self.gc_barrier(
            Some(GcReference::Table(table)),
            GcReference::Table(metatable),
        );
        Ok(())
    }

    /// Returns the stable table handle used as the Lua global environment.
    #[must_use]
    pub const fn global_environment(&self) -> TableHandle {
        self.global_environment
    }

    /// The private Lua registry shared by all threads in this runtime.
    #[must_use]
    pub const fn registry(&self) -> TableHandle {
        self.registry
    }

    /// Registers the package table retained by the `require` builtin.
    ///
    /// This table remains reachable even when Lua code replaces or clears the
    /// public global `package` binding.
    ///
    /// # Errors
    ///
    /// Returns an error when `package` is not a live table handle.
    pub fn set_package_require_table(&mut self, package: TableHandle) -> RuntimeResult<()> {
        self.table(package)?;
        self.package_require_table = Some(package);
        Ok(())
    }

    /// Returns the package table retained by the `require` builtin.
    #[must_use]
    pub const fn package_require_table(&self) -> Option<TableHandle> {
        self.package_require_table
    }
}
