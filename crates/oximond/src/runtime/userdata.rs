//! Userdata objects and `__gc` finalizer registration for userdata and
//! tables.

use crate::runtime::error::{RuntimeError, RuntimeResult};
use crate::runtime::gc::GcReference;
use crate::runtime::metatable::Metamethod;
use crate::runtime::runtime::Runtime;
use crate::runtime::value::{TableHandle, UserdataHandle, Value};
use std::collections::hash_map::Entry;

pub(crate) struct Userdata {
    pub(crate) metatable: Option<TableHandle>,
    pub(crate) environment: Option<TableHandle>,
    pub(crate) uservalues: Vec<Value>,
    pub(crate) light: bool,
    external_bytes: usize,
    allocation_order: u64,
    pub(crate) finalizer: UserdataFinalizerState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum UserdataFinalizerState {
    Unregistered,
    Registered(u64),
    Pending,
    Finalized,
}

impl Userdata {
    pub(crate) fn accounted_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(self.external_bytes)
            .saturating_add(
                self.uservalues
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Value>()),
            )
    }
}

/// An object whose `__gc` metamethod runs once it becomes unreachable.
/// Userdata can have finalizers in every profile; tables from Lua 5.2 on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FinalizerObject {
    Userdata(UserdataHandle),
    Table(TableHandle),
}

/// A table's finalizer bookkeeping. A table without an entry is not
/// registered, or its finalizer already ran.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TableFinalizerState {
    Registered(u64),
    Pending,
}

/// A finalizer removed from the runtime queue for execution by the VM:
/// `handler` is called with `object`.
pub struct Finalizer {
    pub object: Value,
    pub handler: Value,
}

impl Runtime {
    /// Allocates an opaque userdata with no metatable.
    ///
    /// # Errors
    ///
    /// Returns an error if the userdata arena has exhausted its handle space.
    pub fn create_userdata(&mut self) -> RuntimeResult<UserdataHandle> {
        let allocation_order = self.next_userdata_order;
        self.next_userdata_order = self
            .next_userdata_order
            .checked_add(1)
            .ok_or_else(|| RuntimeError::message("userdata allocation order exhausted"))?;
        let handle = self
            .userdatas
            .insert(Userdata {
                metatable: None,
                environment: None,
                uservalues: vec![Value::Nil],
                light: false,
                external_bytes: 0,
                allocation_order,
                finalizer: UserdataFinalizerState::Unregistered,
            })
            .map(UserdataHandle)
            .ok_or_else(|| RuntimeError::message("userdata handle space exhausted"))?;
        self.gc_charge(std::mem::size_of::<Userdata>() + std::mem::size_of::<Value>());
        Ok(handle)
    }

    /// Accounts for payload storage owned by a native userdata adapter.
    ///
    /// # Errors
    ///
    /// Returns an error when `handle` is stale.
    pub fn set_userdata_external_size(
        &mut self,
        handle: UserdataHandle,
        bytes: usize,
    ) -> RuntimeResult<()> {
        let userdata = self.userdata_mut(handle)?;
        let previous = userdata.external_bytes;
        userdata.external_bytes = bytes;
        self.gc_charge(bytes.saturating_sub(previous));
        Ok(())
    }

    pub(crate) fn userdata(&self, handle: UserdataHandle) -> RuntimeResult<&Userdata> {
        self.userdatas
            .get(handle.0)
            .ok_or_else(|| RuntimeError::InvalidUserdata(handle))
    }

    pub(crate) fn userdata_mut(&mut self, handle: UserdataHandle) -> RuntimeResult<&mut Userdata> {
        self.userdatas
            .get_mut(handle.0)
            .ok_or_else(|| RuntimeError::InvalidUserdata(handle))
    }

    /// Set the number of C API uservalue slots on a newly allocated userdata.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is stale or the new slots cannot be allocated.
    pub fn set_userdata_uservalue_count(
        &mut self,
        handle: UserdataHandle,
        count: usize,
    ) -> RuntimeResult<()> {
        let userdata = self.userdata_mut(handle)?;
        let old_capacity = userdata.uservalues.capacity();
        userdata
            .uservalues
            .try_reserve(count.saturating_sub(userdata.uservalues.len()))
            .map_err(|_| RuntimeError::Memory)?;
        userdata.uservalues.resize(count, Value::Nil);
        let growth = userdata.uservalues.capacity().saturating_sub(old_capacity);
        self.gc_charge(growth.saturating_mul(std::mem::size_of::<Value>()));
        Ok(())
    }

    /// Distinguishes C light userdata from full userdata in the Lua debug API.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is stale.
    pub fn set_userdata_light(&mut self, handle: UserdataHandle) -> RuntimeResult<()> {
        let userdata = self.userdata_mut(handle)?;
        userdata.light = true;
        userdata.uservalues.clear();
        Ok(())
    }

    /// Returns whether this userdata represents a C light userdata pointer.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is stale.
    pub fn userdata_is_light(&self, handle: UserdataHandle) -> RuntimeResult<bool> {
        Ok(self.userdata(handle)?.light)
    }

    /// Returns `None` when the requested slot does not exist.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is stale.
    pub fn userdata_uservalue(
        &self,
        handle: UserdataHandle,
        index: usize,
    ) -> RuntimeResult<Option<Value>> {
        let userdata = self.userdata(handle)?;
        Ok(index
            .checked_sub(1)
            .and_then(|index| userdata.uservalues.get(index).cloned()))
    }

    /// Returns whether the requested slot exists, even when its value is nil.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is stale.
    pub fn set_userdata_uservalue(
        &mut self,
        handle: UserdataHandle,
        index: usize,
        value: Value,
    ) -> RuntimeResult<bool> {
        let count = self.userdata(handle)?.uservalues.len();
        let Some(slot) = index.checked_sub(1) else {
            return Ok(false);
        };
        if slot >= count {
            return Ok(false);
        }
        self.gc_barrier_value(Some(GcReference::Userdata(handle)), &value);
        self.userdata_mut(handle)?.uservalues[slot] = value;
        Ok(true)
    }

    /// The Lua 5.1 environment associated with an opaque userdata.
    ///
    /// # Errors
    ///
    /// Returns an error when `handle` is stale.
    pub fn userdata_environment(
        &self,
        handle: UserdataHandle,
    ) -> RuntimeResult<Option<TableHandle>> {
        Ok(self.userdata(handle)?.environment)
    }

    /// Updates the environment and its incremental and generational GC edges.
    ///
    /// # Errors
    ///
    /// Returns an error when `handle` is stale or `environment` is not a
    /// live table.
    pub fn set_userdata_environment(
        &mut self,
        handle: UserdataHandle,
        environment: TableHandle,
    ) -> RuntimeResult<()> {
        self.userdata(handle)?;
        self.table_length(environment)?;
        self.gc_barrier(
            Some(GcReference::Userdata(handle)),
            GcReference::Table(environment),
        );
        self.userdata_mut(handle)?.environment = Some(environment);
        Ok(())
    }

    pub(crate) fn metatable_has_finalizer(&self, metatable: TableHandle) -> RuntimeResult<bool> {
        let table = self.table(metatable)?;
        Ok(!self.table_lacks(table, Metamethod::Gc))
    }

    pub(crate) fn refresh_userdata_finalizer(
        &mut self,
        handle: UserdataHandle,
    ) -> RuntimeResult<()> {
        let enabled = self
            .userdata(handle)?
            .metatable
            .map(|metatable| self.metatable_has_finalizer(metatable))
            .transpose()?
            .unwrap_or(false);
        self.set_userdata_finalizer_registration(handle, enabled)
    }

    pub(crate) fn refresh_metatable_finalizers(
        &mut self,
        metatable: TableHandle,
    ) -> RuntimeResult<()> {
        let enabled = self.metatable_has_finalizer(metatable)?;
        let mut handles = self
            .userdatas
            .live()
            .filter_map(|(key, userdata)| {
                (userdata.metatable == Some(metatable))
                    .then_some((userdata.allocation_order, UserdataHandle(key)))
            })
            .collect::<Vec<_>>();
        handles.sort_by_key(|(order, _)| *order);
        for (_, handle) in handles {
            self.set_userdata_finalizer_registration(handle, enabled)?;
        }
        Ok(())
    }

    pub(crate) fn set_userdata_finalizer_registration(
        &mut self,
        handle: UserdataHandle,
        enabled: bool,
    ) -> RuntimeResult<()> {
        let state = self.userdata(handle)?.finalizer;
        match (state, enabled) {
            (UserdataFinalizerState::Unregistered, true) => {
                let order = if self.table_finalizers_enabled {
                    let order = self.next_finalizer_order;
                    self.next_finalizer_order =
                        self.next_finalizer_order.checked_add(1).ok_or_else(|| {
                            RuntimeError::message("userdata finalizer order exhausted")
                        })?;
                    order
                } else {
                    // Lua 5.1 finalizes userdata in reverse creation order,
                    // whenever `__gc` was set on the metatable.
                    self.userdata(handle)?.allocation_order
                };
                self.userdata_mut(handle)?.finalizer = UserdataFinalizerState::Registered(order);
                self.finalizer_registry
                    .insert(order, FinalizerObject::Userdata(handle));
                // Registration during a cycle keeps the object for this one.
                self.gc_barrier(None, GcReference::Userdata(handle));
            }
            (UserdataFinalizerState::Registered(order), false) => {
                self.finalizer_registry.remove(&order);
                self.userdata_mut(handle)?.finalizer = UserdataFinalizerState::Unregistered;
            }
            (UserdataFinalizerState::Pending | UserdataFinalizerState::Finalized, _)
            | (UserdataFinalizerState::Unregistered, false)
            | (UserdataFinalizerState::Registered(_), true) => {}
        }
        Ok(())
    }

    /// Enables `__gc` finalizers for tables, which Lua 5.2 introduced.
    pub fn set_table_finalizers_enabled(&mut self, enabled: bool) {
        self.table_finalizers_enabled = enabled;
    }

    /// Registers `table` for finalization when `metatable` has a `__gc`
    /// field, as Lua 5.2 and later do when `setmetatable` runs. Adding `__gc`
    /// to the metatable afterwards does not register the table.
    pub(crate) fn register_table_finalizer(
        &mut self,
        table: TableHandle,
        metatable: TableHandle,
    ) -> RuntimeResult<()> {
        if !self.table_finalizers_enabled || !self.metatable_has_finalizer(metatable)? {
            return Ok(());
        }
        let order = self.next_finalizer_order;
        let Entry::Vacant(entry) = self.table_finalizers.entry(table) else {
            return Ok(());
        };
        self.next_finalizer_order = order
            .checked_add(1)
            .ok_or_else(|| RuntimeError::message("finalizer order exhausted"))?;
        entry.insert(TableFinalizerState::Registered(order));
        self.finalizer_registry
            .insert(order, FinalizerObject::Table(table));
        // Registration during a cycle keeps the object for this one.
        self.gc_barrier(None, GcReference::Table(table));
        Ok(())
    }
}
