//! The public `Runtime` table API (raw/metamethod-aware access, native views), plus concatenation and number coercion.

use crate::bytecode::{Chunk, FieldSite, SiteKind};
use crate::runtime::error::{RuntimeError, RuntimeResult};
use crate::runtime::gc::GcReference;
use crate::runtime::gc::heap::value_gc_reference;
use crate::runtime::helpers::{NonFiniteNumerals, parse_lua_integer_exact, parse_lua_number};
use crate::runtime::metatable::Metamethod;
use crate::runtime::native_views::{
    NativeCursor, NativeCursorScratch, NativeCursorSlot, NativeSlotKind, NativeStringProbe,
    NativeStringProbeScratch, NativeTableHeader, NativeValueArray, NativeValueSlot,
};
use crate::runtime::native_views::{NativeRowToken, NativeRowWords};
use crate::runtime::number_format::StackBuffer;
use crate::runtime::runtime::Runtime;
use crate::runtime::table_storage::ArraySlot;
use crate::runtime::table_storage::NextRef;
use crate::runtime::table_storage::Tombstoned;
use crate::runtime::table_storage::entries::StringReadCache;
use crate::runtime::table_storage::key::{
    GC_FIELD_HASH, TableKey, integer_key_from_number, integer_to_number,
};
use crate::runtime::table_storage::{
    DeletedArrayKeys, LazyDeletedKeys, concat_operand, concat_operand_length,
    unsigned_integer_to_number,
};
use crate::runtime::value::{ClosureHandle, TableHandle, Value};
use crate::string::LuaString;

/// What [`Runtime::table_site_probe`] found.
pub enum SiteProbe<'a> {
    /// The site's field, live and not nil.
    Hit(&'a Value),
    /// A table the site does not find a value in.
    Miss(FoundTable<'a>),
    /// No table: the holder is another value, or a stale handle.
    Absent,
}

/// A table that [`Runtime::table_site_probe`] looked up, for the lookup
/// that follows a missed site.
#[derive(Clone, Copy)]
pub struct FoundTable<'a>(&'a crate::runtime::table_storage::Table);

impl Runtime {
    /// Reserves empty table storage without inserting keys.
    ///
    /// # Errors
    /// Returns an error for stale handles, read-only tables or allocation failure.
    pub fn table_reserve(
        &mut self,
        handle: TableHandle,
        array: usize,
        hash: usize,
    ) -> RuntimeResult<()> {
        let table = self.table_mut(handle)?;
        table.ensure_writable()?;
        let previous =
            table.array.capacity() * std::mem::size_of::<Value>() + table.entries.allocated_bytes();
        let missing = array.saturating_sub(table.array.len());
        let result = table
            .array
            .try_reserve(missing)
            .map_err(|_| RuntimeError::message("not enough memory"))
            .and_then(|()| table.entries.reserve(hash));
        let allocated = (table.array.capacity() * std::mem::size_of::<Value>()
            + table.entries.allocated_bytes())
        .saturating_sub(previous);
        self.gc_charge(allocated);
        result
    }

    /// Clears keys while keeping the table identity, metatable and storage.
    ///
    /// # Errors
    /// Returns an error for stale handles or read-only tables.
    pub fn table_clear(&mut self, handle: TableHandle) -> RuntimeResult<()> {
        let table = self.table_mut(handle)?;
        table.ensure_writable()?;
        table.array.clear();
        table.entries.clear();
        table.deleted_array_keys = DeletedArrayKeys::default();
        table.deleted_entries = LazyDeletedKeys::default();
        table.border = 0;
        Ok(())
    }
    /// Changes Valkey's raw table mutation protection.
    ///
    /// # Errors
    /// Returns an error for a stale table handle.
    pub fn table_set_readonly(&mut self, table: TableHandle, enabled: bool) -> RuntimeResult<()> {
        self.table_mut(table)?.readonly = enabled;
        Ok(())
    }

    /// Returns whether a table rejects raw writes and metatable replacement.
    ///
    /// # Errors
    /// Returns an error for a stale table handle.
    pub fn table_is_readonly(&self, table: TableHandle) -> RuntimeResult<bool> {
        Ok(self.table(table)?.readonly)
    }
    /// Looks up a key in a table, returning `nil` when no entry exists.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid table handle. Lua reads with `nil` or
    /// NaN keys simply miss; writes still reject those keys.
    pub fn table_get(&self, table: TableHandle, key: &Value) -> RuntimeResult<Value> {
        self.table(table)?.get_value(key)
    }

    /// [`Self::table_get`] for the integer key `n`, without building a key
    /// value. An array-part hit (nil holes included) is one bounds-checked
    /// load; anything else takes the generic path with the profile's integer
    /// value, so the result is the same as `table_get`'s.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid table handle.
    #[inline]
    pub fn table_get_integer(&self, table: TableHandle, n: i64) -> RuntimeResult<Value> {
        let storage = self.table(table)?;
        if let Ok(index) = usize::try_from(n)
            && let Some(value) = storage.array_slot(index)
        {
            return Ok(value.clone());
        }
        storage.get_value(&self.integer_value(n))
    }

    /// The array-part slot of the integer key `n`, borrowed, a nil hole
    /// included. `None` when `n` is past the array part or the handle is
    /// invalid; the key may still live in the hash part then, so callers fall
    /// back to [`Self::table_get_integer`]. The `Option<&Value>` comes back in
    /// a register, which a C API read of one number per call wants.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn table_array_slot(&self, table: TableHandle, n: i64) -> Option<&Value> {
        // The arena lookup builds no error for a stale handle, so a hit is a
        // few loads with nothing to drop on the way out.
        let target = self.tables.get(table.0)?;
        // A key below one wraps to a huge offset, which fails the bound.
        target
            .array
            .get(usize::try_from((n as u64).wrapping_sub(1)).ok()?)
    }

    /// Looks up a string key in a table without cloning the key.
    ///
    /// Global reads resolve their names from chunk constants on every
    /// execution; borrowing the constant string avoids an `Arc` retain and
    /// release per lookup. Returns `nil` when no entry exists, exactly as
    /// [`Self::table_get`] does for a [`Value::String`] key.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid table handle.
    /// Borrows a present string-keyed value without `__index` dispatch.
    ///
    /// Missing and nil slots, and stale handles, return `None`; callers take
    /// their general indexing path for those.
    #[inline]
    #[must_use]
    pub fn table_get_string_ref(&self, table: TableHandle, key: &LuaString) -> Option<&Value> {
        self.table(table)
            .ok()?
            .entries
            .get_string(key)
            .filter(|value| !matches!(value, Value::Nil))
    }

    /// The table that holds the string `key` when `table[key]` is read, and
    /// how many table-valued `__index` hops away it is (0 for `table`
    /// itself). Every table before the holder has no value for `key` and a
    /// metatable whose raw `__index` is a table. `None` if the chain meets a
    /// function handler, a missing `__index`, a stale handle or more than
    /// `max_hops` hops before it finds a value, so the caller must use the
    /// full lookup.
    #[must_use]
    pub fn table_string_holder(
        &self,
        table: TableHandle,
        key: &LuaString,
        max_hops: u8,
    ) -> Option<(TableHandle, u8)> {
        self.table_string_held(table, key, max_hops)
            .map(|(holder, hops, _)| (holder, hops))
    }

    /// Whether the address of a value of `table`'s keyed entries stays the
    /// value of the same key while
    /// [`table_key_layout`](crate::runtime::table_key_layout) stands: an
    /// ordered or a hashed table. A shaped table changes its storage
    /// without moving the count.
    #[must_use]
    pub(crate) fn table_keeps_key_slots(&self, table: TableHandle) -> bool {
        self.tables
            .get(table.0)
            .is_some_and(|table| !table.entries.is_shaped())
    }

    /// [`Self::table_string_holder`] with the value the holder has for `key`,
    /// found by the same lookup.
    #[must_use]
    pub(crate) fn table_string_held(
        &self,
        table: TableHandle,
        key: &LuaString,
        max_hops: u8,
    ) -> Option<(TableHandle, u8, &Value)> {
        let mut receiver = table;
        for hops in 0..=max_hops {
            let current = self.tables.get(receiver.0)?;
            if let Some(value) = current
                .entries
                .get_string(key)
                .filter(|value| !matches!(value, Value::Nil))
            {
                return Some((receiver, hops, value));
            }
            let metatable = current.metatable?;
            // Metatables are read raw: a stored nil is an absent `__index`.
            match self
                .tables
                .get(metatable.0)?
                .entries
                .get_string(&self.index_metamethod_key)
            {
                Some(Value::Table(next)) => receiver = *next,
                _ => return None,
            }
        }
        None
    }

    /// # Errors
    /// Returns an error for a stale table handle.
    pub fn table_get_string(&self, table: TableHandle, key: &LuaString) -> RuntimeResult<Value> {
        Ok(self
            .table(table)?
            .get_borrowed_string(key)
            .unwrap_or(Value::Nil))
    }

    /// Reads a slot without dispatching `__index`.
    ///
    /// `None` means a missing slot on a metatabled table and asks the VM to
    /// take its normal indexing path. A missing slot without a metatable is
    /// `Some(Value::Nil)`.
    ///
    /// # Errors
    /// Returns an error for a stale table handle.
    #[inline]
    pub fn table_get_if_no_index(
        &self,
        table: TableHandle,
        key: &Value,
    ) -> RuntimeResult<Option<Value>> {
        let table = self.table(table)?;
        let value = table.get_value(key)?;
        if !matches!(value, Value::Nil) || table.metatable.is_none() {
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }

    /// Borrows a stored value without invoking `__index`.
    ///
    /// Nil, missing, and NaN-key reads return `None`. Existing values remain
    /// available when the table has a metatable. The returned reference stays
    /// valid only until the runtime is mutated.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale table handle.
    #[inline]
    pub fn table_get_raw_value<'a>(
        &'a self,
        table: TableHandle,
        key: &Value,
    ) -> RuntimeResult<Option<&'a Value>> {
        self.table(table)?.get_raw_value(key)
    }

    /// [`Self::table_get_raw_value`] for fast paths that handle every miss
    /// (absent value, invalid key, stale handle) the same way.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn table_get_raw_hit<'a>(&'a self, table: TableHandle, key: &Value) -> Option<&'a Value> {
        self.tables.get(table.0)?.get_raw_hit(key)
    }

    /// A present, non-nil value in the array part of `table` at the 1-based
    /// position `index`, or `None` when the array part does not hold it (the
    /// caller then takes the general lookup). Needs no `Value` key, so an
    /// integer loop reads elements without staging a key in memory.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn table_array_hit(&self, table: TableHandle, index: usize) -> Option<&Value> {
        let value = self.tables.get(table.0)?.array_slot(index)?;
        (!matches!(value, Value::Nil)).then_some(value)
    }

    /// [`Self::table_array_hit`] for a float key: `None` unless `key` is an
    /// exact positive integer.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn table_array_hit_number(&self, table: TableHandle, key: f64) -> Option<&Value> {
        self.table_array_hit(
            table,
            crate::runtime::table_storage::array_index_number(key)?,
        )
    }

    /// [`Self::table_get_raw_hit`] with the ordered string-key scan inlined,
    /// reading nil for a miss on a table without a metatable. Only the
    /// interpreter's register-keyed `GetTable` read uses it; other callers
    /// keep the out-of-line scan.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn table_get_field_hit<'a>(&'a self, table: TableHandle, key: &Value) -> Option<&'a Value> {
        static NIL: Value = Value::Nil;
        let table = self.tables.get(table.0)?;
        match table.get_raw_hit_scan(key) {
            Some(value) => Some(value),
            None if table.metatable.is_none() => Some(&NIL),
            None => None,
        }
    }

    /// A present string-keyed value for a constant name that the VM reads
    /// again and again, such as a global or a library function. Hashed
    /// tables answer from `cache` while they are unchanged. Misses, nil
    /// values and stale handles return `None`, as for
    /// [`Self::table_get_field_hit`].
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn table_get_string_cached<'a>(
        &'a self,
        table: TableHandle,
        key: &LuaString,
        cache: &mut StringReadCache,
    ) -> Option<&'a Value> {
        self.tables
            .get(table.0)?
            .entries
            .get_string_live_cached(key, cache)
    }

    /// [`Self::table_get_string_cached`] for an indexing read: an absent key
    /// of a table without a metatable reads as nil, as for
    /// [`Self::table_get_string_hit`]. `None` when `__index` may apply, or
    /// for a stale handle.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn table_get_string_cached_hit<'a>(
        &'a self,
        table: TableHandle,
        key: &LuaString,
        cache: &mut StringReadCache,
    ) -> Option<&'a Value> {
        static NIL: Value = Value::Nil;
        let table = self.tables.get(table.0)?;
        match table.entries.get_string_live_cached(key, cache) {
            Some(value) => Some(value),
            None if table.metatable.is_none() => Some(&NIL),
            None => None,
        }
    }

    /// The live value of the string-keyed field that an inline-cache `site`
    /// finds in `table`, or `None` for a site filled for another layout, an
    /// absent or nil field, or a stale handle. A hit is raw: no metatable
    /// matters for a field that is present.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn table_site_get<'a>(&'a self, table: TableHandle, site: &FieldSite) -> Option<&'a Value> {
        // An empty site, or one a fill was writing, has a guard of zero. A
        // read site that never fills (the table has an `__index`, say) is
        // empty on every execution: stop before the table lookup.
        let (guard, data) = site.load_pair()?;
        if guard == 0 {
            return None;
        }
        self.tables.get(table.0)?.entries.site_value(guard, data)
    }

    /// Reads the string-keyed field that an inline-cache `site` finds in the
    /// table that `holder` holds. The table is looked up before the site is
    /// read, so only the table stays live across the site's guard reads.
    /// With the handle live across them, the fast loop spilled its two
    /// halves to the stack. With the site read first, it ran out of
    /// registers and reloaded loop state after every hit. A miss hands back
    /// the table so the lookup that follows does not repeat it. A site that
    /// never fills (the table has an `__index`, say) then costs one table
    /// lookup per execution, as it did when an empty site stopped the read
    /// before the lookup.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn table_site_probe<'a>(&'a self, site: &FieldSite, holder: &Value) -> SiteProbe<'a> {
        let Some(table) = holder.table_handle_word() else {
            return SiteProbe::Absent;
        };
        let Some(table) = self.tables.get(table.0) else {
            return SiteProbe::Absent;
        };
        // An empty site, or one a fill was writing, has a guard of zero,
        // which no table layout carries.
        if let Some((guard, data)) = site.load_pair()
            && guard != 0
            && let Some(value) = table.entries.site_value(guard, data)
        {
            return SiteProbe::Hit(value);
        }
        SiteProbe::Miss(FoundTable(table))
    }

    /// [`Self::table_get_string_cached_hit`] for a table that
    /// [`Self::table_site_probe`] found.
    #[inline]
    #[must_use]
    pub fn found_table_get_string_cached_hit<'a>(
        &'a self,
        table: FoundTable<'a>,
        key: &LuaString,
        cache: &mut StringReadCache,
    ) -> Option<&'a Value> {
        static NIL: Value = Value::Nil;
        let table = table.0;
        match table.entries.get_string_live_cached(key, cache) {
            Some(value) => Some(value),
            None if table.metatable.is_none() => Some(&NIL),
            None => None,
        }
    }

    /// Fills the inline-cache site at `pc` of `chunk` for a read of `key`
    /// from `table`, if the table holds a live value under it. A site that
    /// has been replaced too often is closed, and a refusal to fill makes the
    /// chunk skip its next few read fills.
    #[inline]
    pub fn table_site_fill(&self, table: TableHandle, chunk: &Chunk, pc: usize, key: &LuaString) {
        if chunk.skip_fill(SiteKind::Read) {
            return;
        }
        if !chunk.field_site_open(SiteKind::Read, pc) {
            chunk.refuse_fills(SiteKind::Read);
            return;
        }
        if let Some(table) = self.tables.get(table.0)
            && let Some(fill) = table.entries.site_fill(key)
        {
            chunk.install_field_site(SiteKind::Read, pc, fill.guard, fill.data, fill.pin);
        }
    }

    /// [`Self::table_site_fill`] for the site of a store. A key that
    /// changes finalizers is never cached.
    pub fn table_site_fill_store(
        &mut self,
        table: TableHandle,
        chunk: &Chunk,
        pc: usize,
        key: &LuaString,
    ) {
        if (key.cached_hash() == GC_FIELD_HASH && key.as_bytes() == b"__gc")
            || chunk.skip_fill(SiteKind::Write)
        {
            return;
        }
        if !chunk.field_site_open(SiteKind::Write, pc) {
            chunk.refuse_fills(SiteKind::Write);
            return;
        }
        if let Some(table) = self.tables.get_mut(table.0)
            && let Some(fill) = table.entries.site_fill_mut(key)
        {
            chunk.install_field_site(SiteKind::Write, pc, fill.guard, fill.data, fill.pin);
        }
    }

    /// The common stores of [`Self::table_site_store`], with no call and
    /// nothing to save around one: a number or a boolean replaces a number
    /// or a boolean in a field the `site` finds in a table that is not
    /// readonly, or fills a field that a shaped table with no metatable
    /// lacks. Neither value owns anything and no value needs a barrier, so
    /// the store is a copy; an overwrite changes no layout, and an added
    /// field charges the collector as [`Self::table_site_store`] does.
    /// Returns [`SiteStore::Missed`], having stored nothing, for every other
    /// store, which takes [`Self::table_site_store`].
    #[allow(unsafe_code, clippy::inline_always)]
    #[inline(always)]
    pub fn table_site_store_scalar(
        &mut self,
        table: TableHandle,
        site: &FieldSite,
        source: &Value,
    ) -> SiteStore {
        if !matches!(
            source,
            Value::Boolean(_) | Value::Integer(_) | Value::Number(_)
        ) {
            return SiteStore::Missed;
        }
        let Some((guard, data)) = site.load_pair() else {
            return SiteStore::Missed;
        };
        if guard == 0 {
            return SiteStore::Missed;
        }
        if self.gc_partial_table == Some(table) {
            // A major cycle is tracing this table in chunks.
            self.gc_partial_dirty = true;
        }
        let Some(target) = self.tables.get_mut(table.0) else {
            return SiteStore::Missed;
        };
        if target.readonly {
            return SiteStore::Missed;
        }
        let Some((slot, shaped)) = target.entries.site_slot_any_mut(guard, data) else {
            return SiteStore::Missed;
        };
        match slot {
            Value::Boolean(_) | Value::Integer(_) | Value::Number(_) => {
                // SAFETY: neither value is a string, so both own no heap data.
                unsafe { copy_value_words(slot, source) };
                SiteStore::Overwrote
            }
            Value::Nil
                if shaped && target.metatable.is_none() && target.deleted_entries.is_empty() =>
            {
                // The table changes as `table_mut` records a change.
                *target.absent_metamethods.get_mut() = 0;
                if target.watched {
                    self.inherit_epoch += 1;
                }
                // SAFETY: the slot is nil and the source a scalar: neither
                // owns heap data.
                unsafe { copy_value_words(slot, source) };
                self.gc_charge(std::mem::size_of::<TableKey>() + std::mem::size_of::<Value>());
                SiteStore::Added
            }
            _ => SiteStore::Missed,
        }
    }

    /// Stores a copy of `source` in the field that an inline-cache `site`
    /// finds in `table`, when the table is not readonly.
    ///
    /// Overwriting a present field is a raw store whatever the metatable
    /// says, takes no accounting, and keeps the table's layout. A shaped
    /// table also takes a field of its shape that it lacks, as the string
    /// store would, when no `__newindex` applies (see
    /// [`Self::site_store_added`]). A nil `source` deletes the field and so
    /// declines.
    #[allow(unsafe_code, clippy::inline_always)]
    #[inline(always)]
    pub fn table_site_store(
        &mut self,
        table: TableHandle,
        site: &FieldSite,
        source: &Value,
    ) -> SiteStore {
        if matches!(source, Value::Nil) {
            return SiteStore::Missed;
        }
        let Some((guard, data)) = site.load_pair() else {
            return SiteStore::Missed;
        };
        if guard == 0 {
            return SiteStore::Missed;
        }
        if self.gc_partial_table == Some(table) {
            // A major cycle is tracing this table in chunks.
            self.gc_partial_dirty = true;
        }
        let Some(target) = self.tables.get_mut(table.0) else {
            return SiteStore::Missed;
        };
        if target.readonly {
            return SiteStore::Missed;
        }
        let Some((slot, shaped)) = target.entries.site_slot_any_mut(guard, data) else {
            return SiteStore::Missed;
        };
        let added = matches!(slot, Value::Nil);
        if added {
            // A field of the shape that this table lacks. Without a
            // metatable it is added here, as the string store's shaped
            // branch would; with one, `__newindex` decides out of line.
            // Only this path needs the facts, so an overwrite skips them.
            let raw_add = shaped && target.metatable.is_none() && target.deleted_entries.is_empty();
            if !raw_add {
                return self.site_store_added(table, guard, data, source);
            }
            // The table changes as `table_mut` records a change.
            *target.absent_metamethods.get_mut() = 0;
            if target.watched {
                self.inherit_epoch += 1;
            }
        } else if target.watched
            && (matches!(source, Value::Table(_) | Value::Closure(_))
                || matches!(slot, Value::Table(_) | Value::Closure(_)))
        {
            // A present field being overwritten makes no key present, so
            // `absent_metamethods` stays right.
            // A table stored over or in place of a table can redirect an
            // inherited read through a watched table, and a closure over or
            // in place of a closure replaces a cached `__index` handler (see
            // `crate::runtime::inherit`).
            self.inherit_epoch += 1;
        }
        // Most stores write a number, a boolean or a string, none of which
        // a barrier is for. Only the tag is tested here: a reference held
        // across the store went through the stack, and the test of it after
        // the store waited for that.
        let may_need_barrier = !matches!(
            source,
            Value::Boolean(_) | Value::Integer(_) | Value::Number(_) | Value::String(_)
        );
        if matches!(source, Value::String(_)) || matches!(slot, Value::String(_)) {
            drop(std::mem::replace(slot, source.clone()));
        } else {
            // SAFETY: neither value is a string, so both own no heap data.
            unsafe { copy_value_words(slot, source) };
        }
        if added {
            self.gc_charge(std::mem::size_of::<TableKey>() + std::mem::size_of::<Value>());
        }
        // A young container outside a major cycle has nothing to remember.
        if may_need_barrier
            && (self.incremental_gc.is_some() || self.tables.is_old(table.0))
            && let Some(reference) = value_gc_reference(source)
        {
            self.gc_barrier(Some(GcReference::Table(table)), reference);
        }
        if added {
            SiteStore::Added
        } else {
            SiteStore::Overwrote
        }
    }

    /// [`Self::table_site_store`] for a nil slot that it does not fill
    /// itself: adds the field to a shaped table with a metatable that lacks
    /// `__newindex`. Declines a table that is not shaped, has deleted keys
    /// (which the string store's shaped branch leaves to the general store),
    /// or whose metatable has `__newindex`.
    #[inline(never)]
    fn site_store_added(
        &mut self,
        table: TableHandle,
        guard: u64,
        data: usize,
        source: &Value,
    ) -> SiteStore {
        let Some(target) = self.tables.get_mut(table.0) else {
            return SiteStore::Missed;
        };
        if !target.deleted_entries.is_empty()
            || target.entries.shaped_site_slot_mut(guard, data).is_none()
        {
            return SiteStore::Missed;
        }
        let Some(metatable) = target.metatable else {
            return SiteStore::Missed;
        };
        if !self.metatable_lacks(metatable, Metamethod::NewIndex) {
            return SiteStore::Missed;
        }
        let Some(target) = self.tables.get_mut(table.0) else {
            return SiteStore::Missed;
        };
        // The field is new: what `table_mut` does for every other change.
        *target.absent_metamethods.get_mut() = 0;
        if target.watched {
            self.inherit_epoch += 1;
        }
        let Some(slot) = target.entries.shaped_site_slot_mut(guard, data) else {
            return SiteStore::Missed;
        };
        let value = source.clone();
        let barrier_value = value_gc_reference(&value);
        *slot = value;
        self.gc_charge(std::mem::size_of::<TableKey>() + std::mem::size_of::<Value>());
        if let Some(reference) = barrier_value
            && (self.incremental_gc.is_some() || self.tables.is_old(table.0))
        {
            self.gc_barrier(Some(GcReference::Table(table)), reference);
        }
        SiteStore::Added
    }

    /// Whether `table` keeps its fields in a shape's slots, so a store site
    /// filled for it also serves the stores that add those fields to the
    /// next table of the shape.
    #[must_use]
    pub fn table_is_shaped(&self, table: TableHandle) -> bool {
        self.tables
            .get(table.0)
            .is_some_and(|table| table.entries.is_shaped())
    }

    /// Borrows a range from a table's dense array, with one-based inclusive indexes.
    /// Returns `None` when any index lies outside the dense array.
    /// Nil slots remain in the slice so callers can report their usual error.
    ///
    /// # Errors
    /// Returns an error for a stale table handle.
    pub fn table_raw_array_range(
        &self,
        table: TableHandle,
        start: usize,
        end: usize,
    ) -> RuntimeResult<Option<&[Value]>> {
        let array = &self.table(table)?.array;
        Ok(start.checked_sub(1).and_then(|first| array.get(first..end)))
    }

    /// The elements `1..=#table` of a table's array part, where `#table` is
    /// its border, or `None` when the border reaches past the array part.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale table handle.
    #[inline]
    pub fn table_border_array(&self, table: TableHandle) -> RuntimeResult<Option<&[Value]>> {
        let table = self.table(table)?;
        Ok(table.array.get(..table.border))
    }

    /// The interpreter's field read: `table[key]` for a string `key` when
    /// it holds a value, or nil when it does not and `table` has no
    /// metatable. `None` when `__index` may apply, or for a stale handle.
    /// The ordered string-key scan is inlined, as for
    /// [`Self::table_get_field_hit`].
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn table_get_string_hit<'a>(
        &'a self,
        table: TableHandle,
        key: &LuaString,
    ) -> Option<&'a Value> {
        static NIL: Value = Value::Nil;
        let table = self.tables.get(table.0)?;
        match table.entries.get_string_live_scan(key) {
            Some(value) => Some(value),
            None if table.metatable.is_none() => Some(&NIL),
            None => None,
        }
    }

    /// Resolves a read of `table[key]` through table-valued `__index`
    /// chains, as Lua indexing does.
    ///
    /// Each hop reads its receiver raw, then follows a table-valued
    /// `__index` from its metatable; a key found nowhere reads as nil.
    /// Returns `Some(Ok(value))` for a resolved read, and
    /// `Some(Err((receiver, handler)))` when the chain ends in a non-table
    /// `__index` handler found on `receiver`, which the caller must invoke
    /// as `handler(receiver, key)`. A stale handle or a chain longer than
    /// the VM's hop budget returns `None`, so the caller falls back to the
    /// exact interpreter path and its errors.
    #[must_use]
    #[inline(never)]
    pub fn table_index_walk(
        &self,
        table: TableHandle,
        key: &Value,
    ) -> Option<Result<&Value, (TableHandle, &Value)>> {
        self.index_walk::<false>(table, key)
    }

    /// [`Self::table_index_walk`] for a caller that has just found `key`
    /// absent from `table` itself, as a string-keyed field read does before
    /// it leaves the fast loop: the walk goes straight to the metatable.
    #[must_use]
    #[inline(never)]
    pub fn table_index_walk_after_miss(
        &self,
        table: TableHandle,
        key: &Value,
    ) -> Option<Result<&Value, (TableHandle, &Value)>> {
        self.index_walk::<true>(table, key)
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn index_walk<const MISSED: bool>(
        &self,
        table: TableHandle,
        key: &Value,
    ) -> Option<Result<&Value, (TableHandle, &Value)>> {
        const MAX_WALK_HOPS: u8 = 100;
        static NIL: Value = Value::Nil;
        let mut receiver = table;
        let mut known_absent = MISSED;
        for _ in 0..MAX_WALK_HOPS {
            let current = self.tables.get(receiver.0)?;
            if !known_absent && let Some(value) = current.get_raw_hit(key) {
                return Some(Ok(value));
            }
            known_absent = false;
            let Some(metatable) = current.metatable else {
                return Some(Ok(&NIL));
            };
            // Metatables are read raw: a stored nil is an absent `__index`.
            match self
                .tables
                .get(metatable.0)?
                .entries
                .get_string(&self.index_metamethod_key)
            {
                None | Some(Value::Nil) => return Some(Ok(&NIL)),
                Some(Value::Table(next)) => receiver = *next,
                Some(handler) => return Some(Err((receiver, handler))),
            }
        }
        None
    }

    /// Borrows the array part of a table for a native trace, whether or not
    /// the table has a metatable.
    ///
    /// A trace reads a slot only after checking that it holds a non-nil
    /// value of the type it was recorded with, and overwrites only such a
    /// slot with a value of the same type. Neither consults `__index` or
    /// `__newindex`, which Lua only runs for an absent key, so a metatable
    /// does not matter. (A trace that reads nil out of a hole checks itself
    /// that the table has no metatable.) Every other store goes through
    /// [`Self::table_trace_set`], and everything else exits to the
    /// interpreter. The view follows [`NativeValueArray`]'s contract.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale table handle.
    pub fn table_trace_array(&mut self, table: TableHandle) -> RuntimeResult<NativeValueArray> {
        let table = self.table_mut(table)?;
        Ok(NativeValueArray {
            values: table.array.as_mut_ptr(),
            len: table.array.len(),
        })
    }

    /// The address of a table and whether native code may append to it and pop
    /// from it in place ([`NativeTableHeader`]), for a trace that binds the
    /// table's header. Native code that appends charges the collector for the
    /// slots it adds ([`Self::charge_native_appends`]), and a table it stores
    /// is given the write barrier ([`Self::table_native_stored`]).
    ///
    /// # Errors
    ///
    /// Returns an error for a stale table handle.
    pub fn table_native_header(&self, table: TableHandle) -> RuntimeResult<NativeTableHeader> {
        let target = self.table(table)?;
        let plain_metatable = target.metatable.is_none_or(|metatable| {
            self.metatable_lacks(metatable, Metamethod::NewIndex)
                && self.table(metatable).is_ok_and(|metatable| {
                    metatable
                        .entries
                        .get_string(&self.len_metamethod_key)
                        .is_none_or(|value| matches!(value, Value::Nil))
                })
        });
        let solid = plain_metatable
            && !target.readonly
            && !target.watched
            && target.entries.is_empty()
            && target.deleted_entries.is_empty()
            && target.deleted_array_keys.explicit_is_empty();
        Ok(NativeTableHeader {
            address: std::ptr::from_ref(target) as usize,
            appendable: solid && target.border == target.array.len(),
            solid,
        })
    }

    /// Charges the collector for `slots` array slots native code appended in
    /// place, as the interpreter charges each push.
    pub fn charge_native_appends(&mut self, slots: usize) {
        self.gc_charge(slots.saturating_mul(std::mem::size_of::<Value>()));
    }

    /// A raw store a native trace makes into a table
    /// ([`Self::table_set_if_no_newindex`] for a number key and a nil,
    /// boolean, number, string or table value), returning the table's array
    /// part as it
    /// stands afterwards, since the store may have grown or moved it.
    ///
    /// Returns `None`, having changed nothing, where the interpreter would not
    /// store raw: the key is absent and the table has a metatable with
    /// `__newindex`, the table is read-only, the key is NaN, or the handle is
    /// stale. A table stored is given the write barrier (a string is a counted
    /// reference and needs none). The view follows [`NativeValueArray`]'s
    /// contract.
    #[inline]
    pub fn table_trace_set(
        &mut self,
        table: TableHandle,
        key: &Value,
        value: &Value,
    ) -> Option<NativeValueArray> {
        debug_assert!(matches!(key, Value::Number(_) | Value::Integer(_)));
        debug_assert!(matches!(
            value,
            Value::Nil
                | Value::Boolean(_)
                | Value::Number(_)
                | Value::Integer(_)
                | Value::String(_)
                | Value::Table(_)
        ));
        // `__newindex` applies to an absent key only, and only if there is one.
        let plain = self
            .table(table)
            .ok()?
            .metatable
            .is_none_or(|metatable| self.metatable_lacks(metatable, Metamethod::NewIndex));
        let target = self.table_mut(table).ok()?;
        if target.readonly || (!plain && !matches!(target.get_raw_value(key), Ok(Some(_)))) {
            return None;
        }
        // The array slot the store wrote, when it wrote one: only that slot
        // needs the barrier's index.
        let (allocated, dense_index) = if let Some(allocated) = target.set_dense_scalar(key, value)
        {
            (allocated, None)
        } else if let Some(allocated) = target.set_dense_reference(key, value) {
            (
                allocated,
                Some(crate::runtime::table_storage::array_offset(key) as usize),
            )
        } else {
            (target.set_value(key, value.clone()).ok()?, None)
        };
        let view = NativeValueArray {
            values: target.array.as_mut_ptr(),
            len: target.array.len(),
        };
        self.gc_charge(allocated);
        if matches!(value, Value::Table(_)) {
            self.table_native_stored_at(table, dense_index, value);
        }
        Some(view)
    }

    /// The write barrier of a native store of `stored` into `table`, for a
    /// store native code made itself: an active major cycle marks the stored
    /// object, and a collector that traces `table` in chunks traces it again.
    /// A string needs none, being a counted reference.
    pub fn table_native_stored(&mut self, table: TableHandle, stored: &Value) {
        self.table_native_stored_at(table, None, stored);
    }

    /// [`Self::table_native_stored`] for a store into array slot `index`
    /// (zero-based) when the caller knows it. `None` names no slot.
    pub fn table_native_stored_at(
        &mut self,
        table: TableHandle,
        index: Option<usize>,
        stored: &Value,
    ) {
        self.table_native_stored_in(table, index.map(ArraySlot::Index), stored);
    }

    /// [`Self::table_native_stored_at`] for a store into the array part of
    /// `table` at the address `slot`, where native code wrote the value.
    /// An address outside the array part names no slot.
    #[inline]
    pub fn table_native_stored_slot(
        &mut self,
        table: TableHandle,
        slot: *const Value,
        stored: &Value,
    ) {
        self.table_native_stored_in(table, Some(ArraySlot::Address(slot as usize)), stored);
    }

    /// [`Self::table_native_stored_slot`] for a table.
    #[inline]
    pub fn table_native_stored_table_slot(
        &mut self,
        table: TableHandle,
        slot: *const Value,
        stored: TableHandle,
    ) {
        if self.gc_partial_table == Some(table) {
            self.gc_partial_dirty = true;
        }
        self.gc_barrier_array_table(table, slot as usize, stored);
    }

    #[inline]
    fn table_native_stored_in(
        &mut self,
        table: TableHandle,
        slot: Option<ArraySlot>,
        stored: &Value,
    ) {
        if self.gc_partial_table == Some(table) {
            self.gc_partial_dirty = true;
        }
        let Some(reference) = crate::runtime::gc::heap::value_gc_reference(stored) else {
            return;
        };
        match slot {
            Some(slot) => self.gc_barrier_array(table, slot, reference),
            None => self.gc_barrier(Some(GcReference::Table(table)), reference),
        }
    }

    /// `table.insert (t, value)` or `table.insert (t, position, value)` as a
    /// native trace makes it ([`Table::trace_insert`] with the collector's
    /// charge and write barrier), returning the table's array part as it
    /// stands afterwards, since the insert moved elements and may have moved
    /// the part.
    ///
    /// `position` is `None` for the end. `honours_metamethods` says whether
    /// the profile's `table.insert` follows `__index`, `__newindex` and
    /// `__len` (Lua 5.3 and later). Returns `None`, having changed nothing,
    /// where the interpreter's call might do something else: the table is
    /// not a plain list (see `Table::trace_insert`), `honours_metamethods`
    /// and the table has such a metamethod, the position is out of range, or
    /// the value is nil. The view follows [`NativeValueArray`]'s contract.
    pub fn table_trace_insert(
        &mut self,
        table: TableHandle,
        position: Option<usize>,
        value: &Value,
        honours_metamethods: bool,
    ) -> Option<NativeValueArray> {
        if honours_metamethods && self.table_needs_metamethods(table) {
            return None;
        }
        let target = self.table_mut(table).ok()?;
        let shifts = position.is_some_and(|place| place <= target.array.len());
        let allocated = target.trace_insert(position, value)?;
        if shifts {
            target.note_array_moved();
        }
        let view = NativeValueArray {
            values: target.array.as_mut_ptr(),
            len: target.array.len(),
        };
        self.gc_charge(allocated);
        // The elements that moved stay in the table, which a chunked trace of
        // it rescans (`table_mut` marked it); the new one is a store.
        if let Some(reference) = value_gc_reference(value) {
            self.gc_barrier(Some(GcReference::Table(table)), reference);
        }
        Some(view)
    }

    /// The raw `table.insert` of the library for a dense list: one move of the
    /// elements after `position` ([`Table::trace_insert`]) in place of a raw get
    /// and set per element, with one charge and at most one barrier, for the
    /// new value. `position` is the one-based place, `None` for the end.
    /// Metamethods are the caller's business, as for the raw loop. Returns
    /// false, having changed nothing, where the loop must run: not a dense
    /// list, a position outside `1..=#t + 1`, a nil value, a readonly table.
    pub fn table_array_shift_insert(
        &mut self,
        table: TableHandle,
        position: Option<usize>,
        value: &Value,
    ) -> bool {
        let Ok(target) = self.table_mut(table) else {
            return false;
        };
        let shifts = position.is_some_and(|place| place <= target.array.len());
        let Some(allocated) = target.trace_insert(position, value) else {
            return false;
        };
        if shifts {
            target.note_array_moved();
        }
        self.gc_charge(allocated);
        if let Some(reference) = value_gc_reference(value) {
            self.gc_barrier(Some(GcReference::Table(table)), reference);
        }
        true
    }

    /// The raw `table.remove` of the library for a dense list: the removed
    /// value, the elements after `position` moved down in one move
    /// ([`Table::trace_remove`]). `None`, having changed nothing, where the
    /// element loop must run (see [`Self::table_array_shift_insert`]) or the
    /// position is outside `1..=#t`.
    pub fn table_array_shift_remove(
        &mut self,
        table: TableHandle,
        position: Option<usize>,
    ) -> Option<Value> {
        let target = self.table_mut(table).ok()?;
        let length = target.array.len();
        let (removed, allocated) = target.trace_remove(position, |_| true)?;
        if position.unwrap_or(length) < length {
            target.note_array_moved();
        }
        self.gc_charge(allocated);
        Some(removed)
    }

    /// `table.remove (t)` or `table.remove (t, position)` as a native trace
    /// makes it, for a value `accept` takes: the removed value and the
    /// table's array part as it stands afterwards. See
    /// [`Self::table_trace_insert`] for `position`, `honours_metamethods` and
    /// when it returns `None`; a position outside `1..=#t` is one of them
    /// (see [`Self::table_trace_remove_nothing`]).
    pub fn table_trace_remove(
        &mut self,
        table: TableHandle,
        position: Option<usize>,
        honours_metamethods: bool,
        accept: impl FnOnce(&Value) -> bool,
    ) -> Option<(Value, NativeValueArray)> {
        if honours_metamethods && self.table_needs_metamethods(table) {
            return None;
        }
        let target = self.table_mut(table).ok()?;
        let length = target.array.len();
        let (removed, allocated) = target.trace_remove(position, accept)?;
        if position.unwrap_or(length) < length {
            target.note_array_moved();
        }
        let view = NativeValueArray {
            values: target.array.as_mut_ptr(),
            len: target.array.len(),
        };
        self.gc_charge(allocated);
        Some((removed, view))
    }

    /// Whether [`Self::table_trace_insert`] of a value that is not nil at
    /// `position` would insert. A recording asks it to know whether the call
    /// it sees would succeed.
    #[must_use]
    pub fn table_trace_insertable(
        &self,
        table: TableHandle,
        position: Option<usize>,
        honours_metamethods: bool,
    ) -> bool {
        if honours_metamethods && self.table_needs_metamethods(table) {
            return false;
        }
        self.table(table)
            .is_ok_and(|target| target.trace_insert_place(position).is_some())
    }

    /// The length of `table` if the calls above treat it as a dense list: no
    /// hole, no record of a deleted hash key, writable, and (with
    /// `honours_metamethods`) no metamethod `table.insert` follows. A
    /// recording asks it to know whether a call would succeed.
    #[must_use]
    pub fn table_trace_list_length(
        &self,
        table: TableHandle,
        honours_metamethods: bool,
    ) -> Option<usize> {
        if honours_metamethods && self.table_needs_metamethods(table) {
            return None;
        }
        self.table(table).ok()?.dense_length()
    }

    /// The element [`Self::table_trace_remove`] would take out, without
    /// taking it: a recording and a helper that must make room for the value
    /// before the table changes ask for it. `None` where the remove would
    /// decline.
    #[must_use]
    pub fn table_trace_removed(
        &self,
        table: TableHandle,
        position: Option<usize>,
        honours_metamethods: bool,
    ) -> Option<&Value> {
        if honours_metamethods && self.table_needs_metamethods(table) {
            return None;
        }
        self.table(table).ok()?.trace_removed(position)
    }

    /// Whether `table.remove (t, position)` gives nil and changes nothing:
    /// `position` (`None` for `t`'s last place) is outside `1..=#t`. False
    /// for a stale table, and for one whose profile's `table.remove` follows
    /// metamethods that it has.
    #[must_use]
    pub fn table_trace_remove_nothing(
        &self,
        table: TableHandle,
        position: Option<i64>,
        honours_metamethods: bool,
    ) -> bool {
        if honours_metamethods && self.table_needs_metamethods(table) {
            return false;
        }
        self.table(table)
            .is_ok_and(|target| target.trace_remove_nothing(position))
    }

    /// Number of slots in a table's array part. Index `k` of Lua (one-based)
    /// lives there exactly when `1 <= k <= len`.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale table handle.
    pub fn table_array_len(&self, table: TableHandle) -> RuntimeResult<usize> {
        Ok(self.table(table)?.array.len())
    }

    /// Borrows an existing present slot (any key, array or hash part) for a
    /// native trace, if it holds a value of `kind`. Missing slots, nil
    /// values, other types and invalid keys return `None`. The slot follows
    /// [`NativeValueSlot`]'s contract.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale table handle.
    pub fn table_native_value_slot(
        &mut self,
        table: TableHandle,
        key: &Value,
        kind: NativeSlotKind,
    ) -> RuntimeResult<Option<NativeValueSlot>> {
        let target = self.table_mut(table)?;
        // A native store of a table over a table could replace an `__index`
        // behind the inherited-read caches' back, so a trace gets no such
        // slot of a table they depend on.
        if target.watched && kind == NativeSlotKind::Table {
            return Ok(None);
        }
        Ok(target
            .raw_value_ptr(key, |value| kind.holds(value))?
            .map(|pointer| NativeValueSlot { pointer }))
    }

    /// [`Self::table_native_value_slot`] for many string keys at once, which
    /// costs one pass over a small table instead of a search per key. For
    /// each `(key, kind)` of `wanted` (a string and the kind of value it must
    /// hold) it writes the address of the present value to the same position
    /// of `out`, and zero for a key that is absent or holds another kind.
    /// A read-only table gets all zeros: a native store through a slot would
    /// bypass the error the interpreter raises. The addresses follow
    /// [`NativeValueSlot`]'s contract.
    ///
    /// `hints` holds one position per key, which the lookup tries first and
    /// updates when the key is elsewhere. Tables built alike keep a key at
    /// one position, so a caller that resolves many of them passes the same
    /// hints each time. Any value is a valid hint. One more hint than keys
    /// lets tables of one shape skip the search altogether.
    ///
    /// A key of kind [`NativeSlotKind::Absent`] gets the address of
    /// [`NATIVE_ABSENT_FIELD`](oximond::runtime::NATIVE_ABSENT_FIELD) when the table
    /// lacks it and has no metatable.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale table handle.
    ///
    /// # Panics
    ///
    /// Panics if `hints` or `out` is shorter than `wanted`.
    pub fn table_native_value_slots(
        &mut self,
        table: TableHandle,
        wanted: &[(Value, NativeSlotKind)],
        hints: &mut [usize],
        out: &mut [usize],
    ) -> RuntimeResult<()> {
        self.table_native_row(table, wanted, hints, out, NativeRowWords::default())
    }

    /// What a row of [`Self::table_native_row`] depends on besides the types of
    /// the values it names: where the table keeps its slots and the state
    /// that changes what the row says. A row built for a table with this
    /// token is the row of any table with the same token, once the types of
    /// the slots it names are checked again (the token cannot tell a store
    /// of another type). `None` for a stale handle, a read-only table (its
    /// row is empty) and a table that is not shaped.
    ///
    /// Does what [`Self::table_native_row`] does to the table's caches, so
    /// that the caller may use a row it kept instead of asking for one.
    pub fn table_native_row_token(&mut self, table: TableHandle) -> Option<NativeRowToken> {
        let target = self.table_mut(table).ok()?;
        if target.readonly {
            return None;
        }
        let (shape, slots) = target.entries.shaped_layout()?;
        Some(NativeRowToken {
            shape,
            slots,
            metatable: target.metatable,
            watched: target.watched,
            address: std::ptr::from_ref(target) as usize,
        })
    }

    /// [`Self::table_native_value_slots`] that also writes the metatable word
    /// of the row ([`crate::jit_ir::trace::TraceOp::MetaGuard`]) to
    /// `out[words.meta]`, from the same look at the table: the identity of
    /// its metatable (see [`crate::jit_ir::trace::meta_word`]), and zero for a
    /// table with none and for a read-only one, which the interpreter writes.
    /// Also writes the table's address to `out[words.address]`
    /// ([`crate::jit_ir::trace::TABLE_ADDRESS_KEY`]), zero for a read-only
    /// table, whose row is empty. A word past `out` writes nothing.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale table handle.
    ///
    /// # Panics
    ///
    /// Panics if `hints` or `out` is shorter than `wanted`.
    pub fn table_native_row(
        &mut self,
        table: TableHandle,
        wanted: &[(Value, NativeSlotKind)],
        hints: &mut [usize],
        out: &mut [usize],
        words: NativeRowWords,
    ) -> RuntimeResult<()> {
        assert!(
            out.len() >= wanted.len() && hints.len() >= wanted.len(),
            "one hint and one output slot per key"
        );
        let out = &mut out[..wanted.len()];
        let target = self.table_mut(table)?;
        if target.readonly {
            out.fill(0);
            return Ok(());
        }
        let metatable = target.metatable;
        // A key the table lacks is nil to a trace only when no `__index` can
        // answer for it.
        let absent = metatable.is_none().then_some(std::ptr::addr_of!(
            crate::runtime::native_views::NATIVE_ABSENT_FIELD
        ) as usize);
        target.string_value_ptrs(wanted, hints, out, absent);
        if target.watched {
            for ((_, kind), slot) in wanted.iter().zip(out.iter_mut()) {
                if *kind == NativeSlotKind::Table {
                    *slot = 0;
                }
            }
        }
        if let Some(field) = words.address.filter(|field| *field < out.len()) {
            out[field] = std::ptr::from_ref(&*target) as usize;
        }
        if let Some(field) = words.meta.filter(|field| *field < out.len()) {
            out[field] = native_meta_word(metatable);
        }
        Ok(())
    }

    /// Whether `row`, which [`Self::table_native_row`] filled for `table`
    /// with `wanted` and `words`, still says what it says, so that a caller
    /// may keep it instead of filling it again. `moved` says whether
    /// [`table_key_layout`](crate::runtime::table_key_layout) moved since the
    /// fill. While it stands, the ordered or hashed storage the row points
    /// into has not moved and an ordered table has gained no key. Once it
    /// moved, or for a shaped table (which may gain a key in a nil slot, and
    /// changes its storage without moving the count when it becomes
    /// ordered), each slot word is checked against the entries as they stand
    /// ([`TableEntries::holds_string_slot`]) and each marker of an absent key
    /// against a lookup; a hashed table cannot be checked so and needs a
    /// fill. This checks the rest: the table is still writable, every slot
    /// still holds a value of its field's kind, the metatable and address
    /// words match, a marker of an absent key stands where no metatable
    /// answers and no key can have come back (a hashed table revives a
    /// deleted key in place), and no table slot is left in a table that is
    /// watched now. A zero word stays zero, which is always a row's safe
    /// answer: native code leaves at it.
    ///
    /// Does what [`Self::table_native_row`] does to the table's caches.
    ///
    /// [`TableEntries::holds_string_slot`]: crate::runtime::table_storage::entries::TableEntries::holds_string_slot
    pub(crate) fn table_native_row_holds(
        &mut self,
        table: TableHandle,
        wanted: &[(Value, NativeSlotKind)],
        row: &[usize],
        words: NativeRowWords,
        moved: bool,
    ) -> bool {
        let Ok(target) = self.table_mut(table) else {
            return false;
        };
        let hashed = target.entries.is_indexed();
        let checked = moved || target.entries.is_shaped();
        if target.readonly || (checked && hashed) {
            return false;
        }
        let marker = std::ptr::addr_of!(crate::runtime::native_views::NATIVE_ABSENT_FIELD) as usize;
        let meta = words.meta.unwrap_or(usize::MAX);
        let address = words.address.unwrap_or(usize::MAX);
        for (field, ((key, kind), &word)) in wanted.iter().zip(row).enumerate() {
            if field == meta {
                if word != native_meta_word(target.metatable) {
                    return false;
                }
            } else if field == address {
                if word != std::ptr::from_ref(&*target) as usize {
                    return false;
                }
            } else if word == marker {
                if hashed || target.metatable.is_some() {
                    return false;
                }
                if checked
                    && !matches!(key, Value::String(key)
                        if target.entries.get_string_live(key).is_none())
                {
                    return false;
                }
            } else if word != 0 {
                if *kind == NativeSlotKind::Absent
                    || (*kind == NativeSlotKind::Table && target.watched)
                {
                    return false;
                }
                if checked
                    && !matches!(key, Value::String(key)
                        if target.entries.holds_string_slot(word, key))
                {
                    return false;
                }
                // SAFETY: the word is the address of a value of this table's
                // entries: unchecked, they have not moved or been freed since
                // the row was filled (the caller's promise); checked, the
                // entries hold the key there now. The first byte of a value
                // is its type.
                #[allow(unsafe_code)]
                if unsafe { *(word as *const u8) } != kind.tag() {
                    return false;
                }
            }
        }
        true
    }

    /// Sorts a dense array prefix of finite numbers in place.
    ///
    /// `false` leaves the table untouched. Callers must use their regular Lua
    /// path in that case, because it may need comparisons, metamethods, or
    /// error handling that this raw array operation deliberately skips.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale table handle.
    pub fn sort_dense_finite_number_prefix(
        &mut self,
        table: TableHandle,
        length: usize,
    ) -> RuntimeResult<bool> {
        let table = self.table_mut(table)?;
        table.ensure_writable()?;
        if table.array.len() != length {
            return Ok(false);
        }
        let Some(prefix) = table.array.get_mut(..length) else {
            return Ok(false);
        };
        if !prefix
            .iter()
            .all(|value| matches!(value, Value::Number(number) if number.is_finite()))
        {
            return Ok(false);
        }
        prefix.sort_unstable_by(|left, right| {
            let (Value::Number(left), Value::Number(right)) = (left, right) else {
                unreachable!("numeric array prefix was validated before sorting");
            };
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(true)
    }

    /// Returns the entry after `previous` in this table's stable traversal.
    ///
    /// A `nil` continuation starts a traversal. The returned pair is
    /// `(nil, nil)` at the end. A non-nil continuation must name an existing
    /// entry, matching Lua's invalid-continuation-key rule.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid table handle or continuation key.
    pub fn table_next(
        &self,
        table: TableHandle,
        previous: &Value,
    ) -> RuntimeResult<(Value, Value)> {
        let table = self.table(table)?;
        let (key, value) = match previous {
            Value::Nil => Ok(table.first_entry().unwrap_or((Value::Nil, Value::Nil))),
            value if table.array_contains(value) || table.deleted_array_contains(value) => {
                Ok(table
                    .array_entry_after(value)
                    .or_else(|| table.map_first_entry())
                    .unwrap_or((Value::Nil, Value::Nil)))
            }
            value => {
                let key = TableKey::try_from(value)
                    .map_err(|_| RuntimeError::InvalidTableContinuationKey)?;
                if !table.entries.contains_key(&key) && !table.deleted_entries.contains(&key) {
                    return Err(RuntimeError::InvalidTableContinuationKey);
                }
                Ok(table
                    .map_entry_after(&key)
                    .unwrap_or((Value::Nil, Value::Nil)))
            }
        }?;
        let key = match key {
            Value::Integer(value) if !self.tagged_integers => {
                Value::Number(integer_to_number(value))
            }
            Value::Number(value) if self.tagged_integers => {
                integer_key_from_number(value).map_or(Value::Number(value), Value::Integer)
            }
            key => key,
        };
        Ok((key, value))
    }

    /// Whether the table has an entry outside its array part, so a traversal
    /// of it goes on into the hash part.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid table handle.
    pub fn table_has_hash_part(&self, table: TableHandle) -> RuntimeResult<bool> {
        Ok(self.table(table)?.map_first_entry().is_some())
    }

    /// [`Self::table_next`] for the steps a table's traversal cursor and
    /// array part answer without lookups. `None` means the cursor declined
    /// (a deleted or invalid continuation key, a stale handle, a table still
    /// too young to have a cursor): the caller then takes
    /// [`Self::table_next`], which settles it and raises its errors.
    #[inline]
    pub fn table_next_fast(
        &mut self,
        table: TableHandle,
        previous: &Value,
    ) -> Option<(Value, Value)> {
        let tagged = self.tagged_integers;
        self.tables
            .get_mut(table.0)?
            .next_entry_fast(previous, tagged)
    }

    /// [`Self::table_next_fast`] that gives the entry as references into the
    /// table (see [`NextRef`]). `None` means the cursor declined.
    #[inline]
    pub fn table_next_ref(&mut self, table: TableHandle, previous: &Value) -> Option<NextRef<'_>> {
        let mut hint = usize::MAX;
        self.table_next_ref_hinted(table, previous, &mut hint)
    }

    /// [`Self::table_next_ref`] with the position of the entry the caller's
    /// last step returned, which a traversal finds its key at without a
    /// search (see `TableEntries::next_after_value`). Any value is safe.
    #[inline]
    pub(crate) fn table_next_ref_hinted(
        &mut self,
        table: TableHandle,
        previous: &Value,
        hint: &mut usize,
    ) -> Option<NextRef<'_>> {
        let tagged = self.tagged_integers;
        self.tables
            .get_mut(table.0)?
            .next_ref(previous, tagged, hint)
    }

    /// The traversal snapshot of `table` for native code ([`NativeCursor`]),
    /// whose position names the entry with key `previous`: the one the table
    /// keeps for its current layout when it is big enough to, and else one
    /// built in `scratch`. `None` when there is none (the table has no
    /// snapshot yet, or does not keep the entries in an order a snapshot can
    /// have). It has `in_sync` clear: the caller sets it when the control
    /// value of its loop is the key the position names.
    pub fn table_native_cursor(
        &mut self,
        table: TableHandle,
        previous: &Value,
        scratch: &mut NativeCursorScratch,
    ) -> Option<NativeCursor> {
        let (slots, len, position) = self
            .tables
            .get_mut(table.0)?
            .entries
            .native_cursor(previous, scratch)?;
        // SAFETY: `position` is the address of a cell the entries just named.
        #[allow(unsafe_code)]
        let position_value = unsafe { *(position as *const u64) };
        Some(NativeCursor {
            in_sync: 0,
            slots,
            len: len as u64,
            position,
            position_value,
            pending: 0,
        })
    }

    /// The string index of `table` for native code: every string key of the
    /// table with the address of its value ([`NativeStringProbe`]), which native
    /// trace code probes for a key it holds at run time. A hashed table keeps
    /// its index while its layout lasts, and builds it for a big table on the
    /// second request; a small one is indexed in `scratch` every time. `None`
    /// for a stale handle or a table that has not earned an index yet, which
    /// leaves the lookups to the runtime.
    ///
    /// The index and every address in it stay valid until the table gains or
    /// loses a key or moves its storage: a caller may use it for as long as
    /// it guarantees that none of that happens. Overwriting a value, or
    /// deleting a hashed table's key in place (it reads as nil), does not
    /// count.
    pub fn table_native_probe(
        &mut self,
        table: TableHandle,
        scratch: &mut NativeStringProbeScratch,
    ) -> Option<NativeStringProbe> {
        self.native_probe(table, scratch, false)
    }

    /// [`Self::table_native_probe`] that indexes a big hashed table on the
    /// first request. A trace's first run is one the caller expects to be long
    /// (a loop entered once runs to its end, or to the iteration budget, on
    /// the index it got), so the build is repaid before a second request
    /// would have come.
    pub fn table_native_probe_now(
        &mut self,
        table: TableHandle,
        scratch: &mut NativeStringProbeScratch,
    ) -> Option<NativeStringProbe> {
        self.native_probe(table, scratch, true)
    }

    fn native_probe(
        &mut self,
        table: TableHandle,
        scratch: &mut NativeStringProbeScratch,
        eager: bool,
    ) -> Option<NativeStringProbe> {
        let partial = self.gc_partial_table == Some(table);
        let target = self.tables.get_mut(table.0)?;
        let writable = !(target.readonly || target.watched || partial);
        let probe = target.entries.native_probe(scratch, eager)?;
        let tombstones = writable && probe.tombstones && target.deleted_entries.is_empty();
        Some(NativeStringProbe {
            writable,
            tombstones,
            ..probe
        })
    }

    /// The key of the slot at address `slot` of a [`NativeCursor`], in the
    /// profile's representation.
    ///
    /// # Safety
    ///
    /// `slot` must be the address of a slot of a snapshot that is valid now
    /// (see [`Self::table_native_cursor`]).
    #[allow(unsafe_code)]
    #[must_use]
    pub unsafe fn native_cursor_key(&self, slot: usize) -> Value {
        // SAFETY: the caller's promise; the key pointer is the table's own.
        let key = unsafe { &*((*(slot as *const NativeCursorSlot)).key as *const TableKey) };
        crate::runtime::table_storage::key_value(key, self.tagged_integers)
    }

    /// Returns the contiguous positive-integer border of a table.
    ///
    /// The map-backed first slice uses the first absent key in the sequence
    /// `1, 2, ...`; the reported length is that key minus one.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid table handle or if the representable
    /// positive-integer key space is exhausted.
    pub fn table_length(&self, table: TableHandle) -> RuntimeResult<f64> {
        let border = self.table(table)?.border;
        let border =
            u64::try_from(border).map_err(|_| RuntimeError::message("table length overflow"))?;
        Ok(unsigned_integer_to_number(border))
    }

    /// Returns the contiguous border of a table when Lua indexing would not
    /// dispatch a `__len` metamethod.
    ///
    /// Native border reads use this primitive: a table with any `__len`
    /// entry (including a stored nil, which Lua treats as absent) resolves
    /// through the exact interpreter instruction instead. Strings, non-table
    /// receivers, and `rawlen`-style operands never reach this helper.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale table handle.
    pub fn table_length_without_len_metamethod(
        &self,
        table: TableHandle,
    ) -> RuntimeResult<Option<f64>> {
        let handle = self.table(table)?;
        if let Some(metatable) = handle.metatable {
            let metatable = self.table(metatable)?;
            let has_len = metatable
                .entries
                .get_string(&self.len_metamethod_key)
                .is_some_and(|value| !matches!(value, Value::Nil));
            if has_len {
                return Ok(None);
            }
        }
        Ok(Some(self.table_length(table)?))
    }

    /// Whether Lua indexing, assignment or `#` of `table` could run a
    /// metamethod beyond what raw access does: its metatable has an
    /// `__index`, `__newindex` or `__len` entry. The library functions that
    /// follow metamethods from Lua 5.3 on take their raw path without one.
    #[inline]
    #[must_use]
    pub fn table_needs_metamethods(&self, table: TableHandle) -> bool {
        let Ok(handle) = self.table(table) else {
            return false;
        };
        let Some(metatable) = handle.metatable else {
            return false;
        };
        let Ok(metatable) = self.table(metatable) else {
            return false;
        };
        [
            &self.index_metamethod_key,
            newindex_key(),
            &self.len_metamethod_key,
        ]
        .into_iter()
        .any(|key| {
            metatable
                .entries
                .get_string(key)
                .is_some_and(|value| !matches!(value, Value::Nil))
        })
    }

    /// Returns the raw byte or array-border length of a string or table.
    ///
    /// This deliberately does not invoke a `__len` metamethod. It is the
    /// primitive used by Lua 5.2's `rawlen` base-library function.
    ///
    /// # Errors
    ///
    /// Returns a type error when `value` is neither a string nor a table.
    pub fn raw_length(&self, value: &Value) -> RuntimeResult<f64> {
        match value {
            Value::String(value) => Ok(unsigned_integer_to_number(
                u64::try_from(value.len())
                    .map_err(|_| RuntimeError::message("string length overflow"))?,
            )),
            Value::Table(table) => self.table_length(*table),
            value => Err(RuntimeError::TypeError {
                operation: "get raw length of",
                expected: "table or string",
                found: value.type_name(),
            }),
        }
    }

    /// Returns whether Lua warnings are currently emitted.
    #[must_use]
    pub const fn warnings_enabled(&self) -> bool {
        self.warnings_enabled
    }

    /// Enables or disables Lua warning output.
    pub fn set_warnings_enabled(&mut self, enabled: bool) {
        self.warnings_enabled = enabled;
    }

    #[must_use]
    pub const fn error_nil_as_string(&self) -> bool {
        self.error_nil_as_string
    }

    pub fn set_error_nil_as_string(&mut self, enabled: bool) {
        self.error_nil_as_string = enabled;
    }

    /// Whether `error` treats a number message as a string and prefixes the
    /// position (Lua 5.1/5.2 and LuaJIT test `lua_isstring`; 5.3+ test the type).
    #[must_use]
    pub const fn error_number_positions(&self) -> bool {
        self.error_number_positions
    }

    pub fn set_error_number_positions(&mut self, enabled: bool) {
        self.error_number_positions = enabled;
    }

    /// Concatenates two Lua string-or-number values.
    ///
    /// # Errors
    ///
    /// Returns a type error when either operand is not a string or number.
    pub fn concatenate(&mut self, left: &Value, right: &Value) -> RuntimeResult<Value> {
        if let Some(result) = self.concat_short([left, right]) {
            if !result.is_empty() {
                self.gc_charge(result.allocation_size());
            }
            return Ok(Value::String(result));
        }
        let left = concat_operand(left, self.number_format)?;
        let right = concat_operand(right, self.number_format)?;
        let length = left
            .len()
            .checked_add(right.len())
            .filter(|length| u32::try_from(*length).is_ok())
            .ok_or_else(|| RuntimeError::message("string length overflow"))?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(length)
            .map_err(|_| RuntimeError::message("string length overflow"))?;
        result.extend_from_slice(&left);
        result.extend_from_slice(&right);
        let result = LuaString::from(result);
        if !result.is_empty() {
            self.gc_charge(result.allocation_size());
        }
        Ok(Value::String(result))
    }

    /// Concatenates a sequence of Lua string-or-number values in one
    /// allocation. Values of other types return the normal primitive error so
    /// the VM can try `__concat` in language order.
    ///
    /// # Errors
    ///
    /// Returns a type error for a non-string, non-number operand or an
    /// overflow error when the resulting Lua string is too large.
    pub fn concatenate_many(&mut self, values: &[Value]) -> RuntimeResult<Value> {
        const MAX_LUA_STRING_BYTES: u64 = u32::MAX as u64;

        // Short results are assembled in one pass on the stack, numbers
        // formatted straight into the buffer, and copied once into their
        // string. The first operand that does not fit, or is not a string or
        // a number, leaves the pass for the sized one below.
        let result = if let Some(result) = self.concat_short(values) {
            result
        } else {
            let mut length = 0_u64;
            for value in values {
                length = length
                    .checked_add(
                        u64::try_from(concat_operand_length(value, self.number_format)?)
                            .map_err(|_| RuntimeError::message("string length overflow"))?,
                    )
                    .ok_or_else(|| RuntimeError::message("string length overflow"))?;
                if length > MAX_LUA_STRING_BYTES {
                    return Err(RuntimeError::message("string length overflow"));
                }
            }
            let length = usize::try_from(length)
                .map_err(|_| RuntimeError::message("string length overflow"))?;
            let mut result = Vec::new();
            result
                .try_reserve_exact(length)
                .map_err(|_| RuntimeError::message("string length overflow"))?;
            for value in values {
                result.extend_from_slice(&concat_operand(value, self.number_format)?);
            }
            LuaString::from(result)
        };
        if !result.is_empty() {
            self.gc_charge(result.allocation_size());
        }
        Ok(Value::String(result))
    }

    /// The concatenation of `values` when every one is a string or a number
    /// and the whole fits a stack buffer; `None` otherwise.
    #[inline]
    fn concat_short<'a>(&self, values: impl IntoIterator<Item = &'a Value>) -> Option<LuaString> {
        let format = self.number_format;
        let mut buffer = StackBuffer::<64>::new();
        for value in values {
            let fits = match value {
                Value::String(text) => buffer.push_bytes(text.as_bytes()),
                Value::Integer(number) => buffer.push_integer(*number),
                Value::Number(number) => buffer.push_number(format, *number),
                _ => return None,
            };
            if !fits {
                return None;
            }
        }
        Some(buffer.into_string())
    }

    /// Converts a Lua number or a numeric string to a tagged Lua 5.3+ number.
    ///
    /// Numbers pass through unchanged. A string converts the way the Lua
    /// lexer reads it: an integer numeral becomes `Value::Integer`, anything
    /// else numeric becomes `Value::Number`.
    ///
    /// # Errors
    ///
    /// Returns a Lua-style type error when `value` is neither a number nor a
    /// numeric string.
    pub fn coerce_numeric_value(value: &Value, operation: &'static str) -> RuntimeResult<Value> {
        match value {
            Value::Integer(_) | Value::Number(_) => Ok(value.clone()),
            Value::String(text) => match parse_lua_integer_exact(text.as_bytes()) {
                Some(integer) => Ok(Value::Integer(integer)),
                None => Self::coerce_number(value, operation).map(Value::Number),
            },
            value => Err(RuntimeError::TypeError {
                operation,
                expected: "number",
                found: value.type_name(),
            }),
        }
    }

    /// Converts a Lua number or a numeric string to an IEEE-754 number.
    ///
    /// This is intentionally limited to arithmetic and numeric-for control
    /// conversion. Ordered comparisons retain their type-strict behavior.
    ///
    /// # Errors
    ///
    /// Returns a Lua-style type error when `value` is not a numeric value or
    /// a string accepted by the Lua 5.1 numeric conversion subset.
    pub fn coerce_number(value: &Value, operation: &'static str) -> RuntimeResult<f64> {
        Self::coerce_number_in(value, operation, NonFiniteNumerals::Rejected)
    }

    /// A number or numeric string as a library function converts it, with
    /// the spellings of infinity and NaN this runtime's profile accepts.
    #[must_use]
    pub fn to_number(&self, value: &Value) -> Option<f64> {
        Self::coerce_number_in(value, "convert", self.nonfinite_numerals).ok()
    }

    /// Converts like [`Self::coerce_number`], also accepting the spellings
    /// of infinity and NaN that `nonfinite` allows.
    ///
    /// # Errors
    ///
    /// Returns a Lua-style type error when `value` is neither a number nor
    /// a string that converts to one.
    pub fn coerce_number_in(
        value: &Value,
        operation: &'static str,
        nonfinite: NonFiniteNumerals,
    ) -> RuntimeResult<f64> {
        match value {
            Value::Integer(value) => Ok(integer_to_number(*value)),
            Value::Number(value) => Ok(*value),
            Value::String(value) => {
                let bytes = value.as_bytes();
                parse_lua_number(bytes)
                    .or_else(|| nonfinite.parse(bytes))
                    .ok_or_else(|| RuntimeError::TypeError {
                        operation,
                        expected: "number",
                        found: "string",
                    })
            }
            value => Err(RuntimeError::TypeError {
                operation,
                expected: "number",
                found: value.type_name(),
            }),
        }
    }
    /// Setting `nil` removes an existing entry.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid table handle or an invalid key such as
    /// `nil` or NaN. The table is unchanged when key conversion fails.
    pub fn table_set(
        &mut self,
        table: TableHandle,
        key: &Value,
        value: Value,
    ) -> RuntimeResult<()> {
        let changes_gc = matches!(key, Value::String(name) if name.as_bytes() == b"__gc");
        let barrier_key = value_gc_reference(key);
        let barrier_value = value_gc_reference(&value);
        let allocated = {
            let table = self.table_mut(table)?;
            table.set_value(key, value)?
        };
        self.gc_charge(allocated);
        let container = Some(GcReference::Table(table));
        for reference in barrier_key.into_iter().chain(barrier_value) {
            self.gc_barrier(container, reference);
        }
        if changes_gc {
            self.refresh_metatable_finalizers(table)?;
        }
        Ok(())
    }

    /// [`Self::table_set_integer`] for a `Value::Number`, which a plain
    /// array write or append handles without building the value on the
    /// stack. Anything else takes the generic call.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid table handle or a readonly table.
    #[inline]
    pub fn table_set_integer_number(
        &mut self,
        table: TableHandle,
        n: i64,
        x: f64,
    ) -> RuntimeResult<()> {
        if (1..1_i64 << 53).contains(&n) {
            let allocated = self.table_mut(table)?.set_number_fast(n as usize, x);
            if let Some(allocated) = allocated {
                self.gc_charge(allocated);
                return Ok(());
            }
        }
        self.table_set_integer(table, n, Value::Number(x))
    }

    /// [`Self::table_set`] for the integer key `n`. Keys that are exact in an
    /// `f64` go straight to the array logic; larger ones take the generic
    /// path so a profile without integers rounds them as before. The GC
    /// charge and the write barrier on a collectable value are the same.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid table handle or a readonly table.
    #[inline]
    pub fn table_set_integer(
        &mut self,
        table: TableHandle,
        n: i64,
        value: Value,
    ) -> RuntimeResult<()> {
        if !(1..1_i64 << 53).contains(&n) {
            return self.table_set(table, &self.integer_value(n), value);
        }
        let barrier_value = value_gc_reference(&value);
        let allocated = self.table_mut(table)?.set_index(n as usize, value)?;
        self.gc_charge(allocated);
        if let Some(reference) = barrier_value {
            self.gc_barrier(Some(GcReference::Table(table)), reference);
        }
        Ok(())
    }

    /// Stores `value` under the string `key` of a table made with a shape that
    /// names the key, which holds nothing under it yet: the fill of a
    /// constructor. The store has the collector's write barrier. `false`,
    /// having stored nothing, for any other table, key or state, where
    /// [`Self::table_set`] does the store.
    pub fn table_init_shaped(
        &mut self,
        table: TableHandle,
        key: &LuaString,
        value: &Value,
    ) -> bool {
        let Some(target) = self.tables.get_mut(table.0) else {
            return false;
        };
        if target.readonly
            || target.metatable.is_some()
            || !target.deleted_entries.is_empty()
            || !target.entries.init_shaped_slot(key, value)
        {
            return false;
        }
        // The charge of a new key, as [`Self::table_set`] makes it.
        self.gc_charge(std::mem::size_of::<TableKey>() + std::mem::size_of::<Value>());
        self.gc_barrier_value(Some(GcReference::Table(table)), value);
        true
    }

    /// The address of the first value slot of `table`, if it is shaped (see
    /// [`Self::create_table_with_shape`]): the slot of the shape's key `i` is
    /// `i` values further. Native code fills a table it just made through
    /// them ([`Self::charge_native_inits`] accounts for the fields). The
    /// slots stay where they are until a store adds a key the shape lacks.
    pub fn table_shaped_slots(&mut self, table: TableHandle) -> Option<usize> {
        let target = self.tables.get_mut(table.0)?;
        target.entries.shaped_slots()
    }

    /// Gives the table `table`, which holds nothing, an array part of `count`
    /// nil slots counted as present, and returns the address of the first.
    /// The caller overwrites all of them before the table can be read.
    pub fn table_prefill_array(&mut self, table: TableHandle, count: usize) -> Option<usize> {
        let target = self.tables.get_mut(table.0)?;
        target.prefill_array(count)
    }

    /// Charges the collector for `fields` fields of shaped tables that native
    /// code stored in place, as [`Self::table_init_shaped`] charges each.
    pub fn charge_native_inits(&mut self, fields: usize) {
        self.gc_charge(
            fields.saturating_mul(std::mem::size_of::<TableKey>() + std::mem::size_of::<Value>()),
        );
    }

    /// Whether a store of a reference into a table made since the collector
    /// last ran needs no write barrier: no major cycle is in progress and
    /// nothing runs the collector until the caller returns.
    #[must_use]
    pub fn gc_barrier_idle(&self) -> bool {
        self.incremental_gc.is_none()
    }

    /// A raw store (`rawset`, `lua_rawseti`) of a number, integer or boolean
    /// under the integer key `n`, borrowing the value, in the two shapes a
    /// host rewriting or filling a whole array makes: an overwrite of an
    /// array slot that holds a scalar, and an append to a full array part
    /// with room to spare in a table without a hash part. A raw store
    /// ignores the metatable, and an overwrite changes no reference, hash
    /// entry, border or deleted-key record, so a writable table takes it in
    /// place. Integer keys never name a metamethod, so the absence cache
    /// stays valid without `table_mut`. Returns `false`, having changed
    /// nothing, for any other store. A leaf without calls.
    #[allow(unsafe_code, clippy::inline_always)]
    #[inline(always)]
    pub fn table_raw_store_scalar(&mut self, table: TableHandle, n: i64, value: &Value) -> bool {
        if self.tables.get(table.0).is_none() {
            return false;
        }
        // SAFETY: the handle names a live table, as just checked.
        unsafe { self.table_raw_store_scalar_live(table, n, value) }
    }

    /// [`Self::table_raw_store_scalar`] for a table the caller knows is
    /// live, which skips the arena's generation and occupancy checks. The
    /// C API stores into tables on its stacks, which the collector roots
    /// (see `Runtime::set_host_roots_refresher`), so a handle there always
    /// names a live table.
    ///
    /// # Safety
    ///
    /// `table` must name a live table of this runtime: one that a root
    /// holds, so no collection since it was made could have freed it.
    #[allow(unsafe_code, clippy::inline_always)]
    #[inline(always)]
    pub unsafe fn table_raw_store_scalar_live(
        &mut self,
        table: TableHandle,
        n: i64,
        value: &Value,
    ) -> bool {
        // SAFETY: the caller promises the table is live.
        let target = unsafe { self.tables.get_unchecked_mut(table.0) };
        // A key below one wraps to a huge offset, which fails the bound.
        let offset = (n as u64).wrapping_sub(1);
        if target.readonly {
            return false;
        }
        let length = target.array.len();
        if offset >= length as u64 {
            // SAFETY: as the caller promises.
            return offset == length as u64
                && unsafe { self.table_append_scalar_live(table, value) };
        }
        #[allow(clippy::cast_possible_truncation, reason = "below the array length")]
        // SAFETY: the offset is below the array length.
        let slot = unsafe { &mut *target.array.as_mut_ptr().add(offset as usize) };
        // The value is tested last, which leaves the lookup its register: a
        // test up front made the C API's `lua_rawseti` save three more.
        if !is_plain_scalar(slot) || !is_plain_scalar(value) {
            return false;
        }
        // SAFETY: both values are scalars, which own no heap data.
        unsafe { copy_value_words(slot, value) };
        true
    }

    /// The append half of [`Self::table_raw_store_scalar_live`], at the key
    /// one past the end of the array part.
    ///
    /// # Safety
    ///
    /// As for [`Self::table_raw_store_scalar_live`].
    #[allow(unsafe_code, clippy::inline_always)]
    #[inline(always)]
    unsafe fn table_append_scalar_live(&mut self, table: TableHandle, value: &Value) -> bool {
        // A table a major cycle is tracing in chunks takes the general path,
        // which marks it dirty. The slots alone tell the tables apart, since
        // both handles name live tables; leaving the generation out keeps
        // it out of a register.
        if self
            .gc_partial_table
            .is_some_and(|partial| partial.index() == table.index())
        {
            return false;
        }
        // SAFETY: the caller promises the table is live.
        let target = unsafe { self.tables.get_unchecked_mut(table.0) };
        if !target.append_scalar_in_place(value) {
            return false;
        }
        // `gc_charge` of one value. The counts are bytes allocated and
        // allocations made, which no program brings near the top of a
        // 64-bit word, so they add without the saturation and its branches.
        self.gc_debt = self.gc_debt.wrapping_add(std::mem::size_of::<Value>());
        self.allocation_serial = self.allocation_serial.wrapping_add(1);
        true
    }

    /// [`Self::table_array_slot`] for a table the caller knows is live (see
    /// [`Self::table_raw_store_scalar_live`]).
    ///
    /// # Safety
    ///
    /// As for [`Self::table_raw_store_scalar_live`].
    #[allow(unsafe_code, clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub unsafe fn table_array_slot_live(&self, table: TableHandle, n: i64) -> Option<&Value> {
        // SAFETY: the caller promises the table is live.
        let target = unsafe { self.tables.get_unchecked(table.0) };
        // A key below one wraps to a huge offset, which fails the bound.
        target
            .array
            .get(usize::try_from((n as u64).wrapping_sub(1)).ok()?)
    }

    /// [`Self::table_raw_store_scalar`] that also takes the next append
    /// or the fill of a hole in the array part of a table without a hash
    /// part. Returns `false`, having changed nothing, for any other store,
    /// which [`Self::table_set_integer`] handles.
    #[inline]
    pub fn table_raw_set_integer_scalar(
        &mut self,
        table: TableHandle,
        n: i64,
        value: &Value,
    ) -> bool {
        if self.table_raw_store_scalar(table, n, value) {
            return true;
        }
        // A table a major cycle is tracing in chunks takes the general path,
        // which marks it dirty.
        if self.gc_partial_table == Some(table) || !(1..i64::MAX).contains(&n) {
            return false;
        }
        let Some(target) = self.tables.get_mut(table.0) else {
            return false;
        };
        if target.readonly {
            return false;
        }
        let Ok(index) = usize::try_from(n) else {
            return false;
        };
        match target.set_dense_scalar_at(index, value) {
            Some(allocated) => {
                self.gc_charge(allocated);
                true
            }
            None => false,
        }
    }

    /// [`Self::table_set_dense_scalar`] for the interpreter's `SetTable`,
    /// which says whether the store was in place: a number or boolean over
    /// a present number, integer or boolean of the array part allocates
    /// nothing and changes no reference, so the caller owes the collector
    /// no poll after it.
    #[inline(always)]
    pub fn table_store_dense_scalar(
        &mut self,
        table: TableHandle,
        key: &Value,
        value: &Value,
    ) -> DenseScalarStore {
        let scalar = |value: &Value| {
            matches!(
                value,
                Value::Number(_) | Value::Integer(_) | Value::Boolean(_)
            )
        };
        if !scalar(value) {
            return DenseScalarStore::Declined;
        }
        let Some(target) = self.tables.get_mut(table.0) else {
            return DenseScalarStore::Declined;
        };
        let offset = crate::runtime::table_storage::array_offset(key);
        #[allow(clippy::cast_possible_truncation)]
        if target.readonly
            || offset >= target.array.len() as u64
            || !scalar(&target.array[offset as usize])
        {
            // As the rest of `table_set_dense_scalar`.
            if self.gc_partial_table == Some(table)
                || target.metatable.is_some()
                || target.readonly
                || offset >= i64::MAX as u64
            {
                return DenseScalarStore::Declined;
            }
            return match target.set_dense_scalar_at(offset as usize + 1, value) {
                Some(allocated) => {
                    self.gc_charge(allocated);
                    DenseScalarStore::Stored
                }
                None => DenseScalarStore::Declined,
            };
        }
        #[allow(clippy::cast_possible_truncation)]
        let slot = &mut target.array[offset as usize];
        // SAFETY: both are valid, aligned values that own nothing (a
        // number, an integer or a boolean), so the tag byte and the payload
        // word of the `repr(C, u8)` value copy it whole and nothing is
        // dropped. One copy for the three variants: building each at the
        // store kept the tag and the slot's address in memory across the
        // match.
        #[allow(unsafe_code)]
        unsafe {
            let (to, from) = (
                std::ptr::from_mut(slot).cast::<u8>(),
                std::ptr::from_ref(value).cast::<u8>(),
            );
            let tag = from.read();
            let payload = from.add(8).cast::<std::mem::MaybeUninit<u64>>().read();
            to.write(tag);
            to.add(8)
                .cast::<std::mem::MaybeUninit<u64>>()
                .write(payload);
        }
        DenseScalarStore::InPlace
    }

    /// Stores a number or boolean at an array key, borrowing the value. A
    /// present number, integer or boolean slot of any writable table is
    /// overwritten in place, since `__newindex` only fires for absent keys;
    /// every other array store needs a table without a metatable. Returns
    /// `false`, having changed nothing, for any store that needs the general
    /// path.
    #[inline(always)]
    pub fn table_set_dense_scalar(
        &mut self,
        table: TableHandle,
        key: &Value,
        value: &Value,
    ) -> bool {
        // Array keys only: the absence cache covers the string keys
        // `__newindex`, `__gc` and `__metatable`, so no `table_mut` is needed.
        let Some(target) = self.tables.get_mut(table.0) else {
            return false;
        };
        let offset = crate::runtime::table_storage::array_offset(key);
        if !target.readonly && offset < target.array.len() as u64 {
            let slot = &mut target.array[offset as usize];
            // Overwriting a live number or boolean with another changes no
            // reference, hash entry or deleted-key record, and `__newindex`
            // only fires for absent keys, so no other check applies.
            if matches!(
                slot,
                Value::Number(_) | Value::Integer(_) | Value::Boolean(_)
            ) {
                *slot = match value {
                    Value::Number(number) => Value::Number(*number),
                    Value::Integer(number) => Value::Integer(*number),
                    Value::Boolean(flag) => Value::Boolean(*flag),
                    _ => return false,
                };
                return true;
            }
        }
        if self.gc_partial_table == Some(table)
            || target.metatable.is_some()
            || target.readonly
            || offset >= i64::MAX as u64
        {
            return false;
        }
        #[allow(clippy::cast_possible_truncation)]
        match target.set_dense_scalar_at(offset as usize + 1, value) {
            Some(allocated) => {
                self.gc_charge(allocated);
                true
            }
            None => false,
        }
    }

    /// Stores a string, table, closure or function at an array key of a
    /// table without a metatable, borrowing the value, and runs the write
    /// barrier of a collectable one. Returns `false`, having changed nothing,
    /// for any store that needs the general path.
    #[inline(always)]
    pub fn table_set_dense_reference(
        &mut self,
        table: TableHandle,
        key: &Value,
        value: &Value,
    ) -> bool {
        // A table a major cycle is tracing in chunks takes the general path,
        // which marks it dirty.
        if self.gc_partial_table == Some(table) {
            return false;
        }
        // Array keys only, as in `table_set_dense_scalar`.
        let Some(target) = self.tables.get_mut(table.0) else {
            return false;
        };
        if target.metatable.is_some() {
            return false;
        }
        let Some(allocated) = target.set_dense_reference(key, value) else {
            return false;
        };
        let index = crate::runtime::table_storage::array_offset(key) as usize;
        self.gc_charge(allocated);
        // Only a table or closure is a collectable object.
        let stored = match value {
            Value::Table(handle) => GcReference::Table(*handle),
            Value::Closure(handle) => GcReference::Closure(*handle),
            _ => return true,
        };
        self.gc_barrier_array(table, ArraySlot::Index(index), stored);
        true
    }

    /// Writes a raw table slot when Lua cannot dispatch `__newindex`.
    ///
    /// Returns `false` without changing the table when the key is absent and
    /// the table has a metatable. The VM must resolve `__newindex` in that
    /// case. Tables without metatables and existing raw keys are updated with
    /// one arena lookup.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale table handle or an invalid key.
    pub fn table_set_if_no_newindex(
        &mut self,
        table: TableHandle,
        key: &Value,
        mut value: Value,
    ) -> RuntimeResult<bool> {
        // Array stores of numbers and booleans, the common table store,
        // copy the value into the slot field by field.
        let scalar = matches!(
            value,
            Value::Number(_) | Value::Integer(_) | Value::Boolean(_)
        );
        if scalar {
            let target = self.table_mut(table)?;
            if target.metatable.is_none()
                && let Some(allocated) = target.set_dense_scalar(key, &value)
            {
                self.gc_charge(allocated);
                return Ok(true);
            }
        }
        let barrier_value = value_gc_reference(&value);
        // Array stores of plain values need no barrier or finalizer check.
        // A scalar that reached this point has already been refused.
        if !scalar && barrier_value.is_none() && !matches!(value, Value::Nil) {
            let target = self.table_mut(table)?;
            if target.metatable.is_none() {
                match target.set_dense(key, value) {
                    Ok(allocated) => {
                        self.gc_charge(allocated);
                        return Ok(true);
                    }
                    Err(unwritten) => value = unwritten,
                }
            }
        }
        let changes_gc = matches!(key, Value::String(name) if name.cached_hash() == GC_FIELD_HASH && name.as_bytes() == b"__gc");
        let barrier_key = value_gc_reference(key);
        let allocated = {
            let mut target = self.table_mut(table)?;
            let mut raw = true;
            if let Some(metatable) = target.metatable
                && !target.has_raw_value(key)?
            {
                raw = self.metatable_lacks(metatable, Metamethod::NewIndex);
                // Borrowed again, as in `table_set_string_owned`.
                target = self.table_mut(table)?;
            }
            if raw {
                Some(target.set_value(key, value)?)
            } else {
                None
            }
        };
        let written = allocated.is_some();
        self.gc_charge(allocated.unwrap_or_default());
        if written {
            let container = Some(GcReference::Table(table));
            for reference in barrier_key.into_iter().chain(barrier_value) {
                self.gc_barrier(container, reference);
            }
        }
        if written && changes_gc {
            self.refresh_metatable_finalizers(table)?;
        }
        Ok(written)
    }

    /// Whether [`Self::table_overwrite_scalar_string`] would store `value`
    /// under `key` of `table` now.
    #[must_use]
    pub fn table_can_overwrite_scalar_string(
        &self,
        table: TableHandle,
        key: &LuaString,
        value: &Value,
    ) -> bool {
        self.table(table)
            .is_ok_and(|table| table.can_overwrite_scalar_string(key, value))
    }

    /// Overwrites the value of the present string `key` of `table` with the
    /// integer, float or boolean `value` when the table holds one of the same
    /// type, and returns whether it did. It changes no layout and allocates
    /// nothing, so it needs no barrier and a traversal that holds the entries'
    /// addresses stays valid. A store it declines changes nothing.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale table handle.
    pub fn table_overwrite_scalar_string(
        &mut self,
        table: TableHandle,
        key: &LuaString,
        value: &Value,
    ) -> RuntimeResult<bool> {
        // A scalar over a scalar of its type keeps every key present and holds
        // no reference: a table that is no metatable's has nothing to forget
        // and a collection that traces it in chunks nothing to rescan, so the
        // bookkeeping of `table_mut` is only the notice a watched table gets.
        let Some(target) = self.tables.get_mut(table.0) else {
            return Err(RuntimeError::InvalidTable(table));
        };
        let stored = target.overwrite_scalar_string(key, value);
        if stored && target.watched {
            self.inherit_epoch += 1;
        }
        Ok(stored)
    }

    /// Whether [`Self::table_delete_string_in_place`] would store nil under
    /// `key` of `table` now.
    #[must_use]
    pub fn table_can_delete_string_in_place(&self, table: TableHandle, key: &LuaString) -> bool {
        let Ok(target) = self.table(table) else {
            return false;
        };
        if target.readonly {
            return false;
        }
        if target.entries.get_string_live(key).is_none() {
            return self.absent_key_stores_raw(target);
        }
        target.can_tombstone_present_string(key)
    }

    /// Stores nil under the string `key` of `table` where that leaves every
    /// entry in its place, and returns whether it did. A key the table holds
    /// becomes a deleted key (a nil in its place) in a hashed table, and the
    /// layout number is kept, so a traversal that holds the entries'
    /// addresses stays valid and skips the deleted key. A key the table lacks
    /// stays absent, when no `__newindex` takes the store. A store it declines
    /// (another representation, a read-only table, the key `__gc`) changes
    /// nothing.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale table handle.
    pub fn table_delete_string_in_place(
        &mut self,
        table: TableHandle,
        key: &LuaString,
    ) -> RuntimeResult<bool> {
        let partial = self.gc_partial_table == Some(table);
        let Some(target) = self.tables.get_mut(table.0) else {
            return Err(RuntimeError::InvalidTable(table));
        };
        if target.readonly {
            return Ok(false);
        }
        // One probe finds the key and deletes it.
        match target.tombstone_live_string(key) {
            Tombstoned::Deleted => {
                // What `table_mut` does, on the table already found.
                if partial {
                    self.gc_partial_dirty = true;
                }
                target.forget_absent_metamethods();
                if target.watched {
                    self.inherit_epoch += 1;
                }
                Ok(true)
            }
            Tombstoned::Absent => {
                let metatable = target.metatable;
                Ok(metatable
                    .is_none_or(|metatable| self.metatable_lacks(metatable, Metamethod::NewIndex)))
            }
            Tombstoned::Unhashed => Ok(false),
        }
    }

    /// Whether [`Self::table_revive_string`] would store under `key` of
    /// `table` now.
    #[must_use]
    pub fn table_can_revive_string(&self, table: TableHandle, key: &LuaString) -> bool {
        self.table(table)
            .is_ok_and(|table| table.metatable.is_none() && table.can_revive_string(key))
    }

    /// Stores the integer, float or boolean `value` under `key` of `table`
    /// when the table, which has no metatable, holds the key as a deleted key
    /// (a nil in its place, see [`Self::table_delete_string_in_place`]), and
    /// returns whether it did. The entry takes the value where it stands, so
    /// the layout number is kept and a traversal that holds the entries'
    /// addresses stays valid. It allocates nothing and needs no barrier. A
    /// store it declines (an absent or live key, another representation, a
    /// read-only table, a metatable, the key `__gc`) changes nothing.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale table handle.
    pub fn table_revive_string(
        &mut self,
        table: TableHandle,
        key: &LuaString,
        value: &Value,
    ) -> RuntimeResult<bool> {
        let partial = self.gc_partial_table == Some(table);
        let Some(target) = self.tables.get_mut(table.0) else {
            return Err(RuntimeError::InvalidTable(table));
        };
        if target.metatable.is_some()
            || !matches!(
                value,
                Value::Boolean(_) | Value::Integer(_) | Value::Number(_)
            )
        {
            return Ok(false);
        }
        // Only a scalar gets here, and it owns no heap data.
        // SAFETY: a bitwise copy of a value that owns nothing.
        #[allow(unsafe_code)]
        let copy = unsafe { std::ptr::read(value) };
        if target.revive_string(key, copy).is_err() {
            return Ok(false);
        }
        // What `table_mut` does, on the table already found.
        if partial {
            self.gc_partial_dirty = true;
        }
        target.forget_absent_metamethods();
        if target.watched {
            self.inherit_epoch += 1;
        }
        Ok(true)
    }

    /// Whether a store of nil under a key `target` lacks is the raw no-op:
    /// the table has no metatable, or its metatable has no `__newindex`.
    fn absent_key_stores_raw(&self, target: &crate::runtime::table_storage::Table) -> bool {
        target
            .metatable
            .is_none_or(|metatable| self.metatable_lacks(metatable, Metamethod::NewIndex))
    }

    /// [`Self::table_set_if_no_newindex`] for a string key, which it
    /// borrows: field and global writes skip the array-slot attempt and the
    /// owned key that the general write builds.
    ///
    /// Returns `false`, having changed nothing, when the table has a
    /// metatable and the key holds no value, so `__newindex` may apply.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale table handle or a readonly table.
    // Out of line: inlined into the interpreter's fast loop, the store
    // costs field and global reads elsewhere in the loop several percent.
    #[inline(never)]
    pub fn table_set_string_if_no_newindex(
        &mut self,
        table: TableHandle,
        key: &LuaString,
        value: Value,
    ) -> RuntimeResult<bool> {
        self.table_set_string_owned(table, key, value)
    }

    /// [`Self::table_set_string_if_no_newindex`] for the fast loop's
    /// string-key store. When the store declines because the metatable has a
    /// `__newindex` closure, it keeps that closure for
    /// [`Self::take_newindex_hint`], so the caller enters it without a second
    /// lookup. The signature matches the plain store's, which keeps the call
    /// site in the interpreter loop unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale table handle or a readonly table.
    #[inline(never)]
    pub fn table_set_string_if_no_newindex_hinted(
        &mut self,
        table: TableHandle,
        key: &LuaString,
        value: Value,
    ) -> RuntimeResult<bool> {
        self.table_set_string_owned_with::<true>(table, key, value)
    }

    /// [`Self::table_set_string_if_no_newindex_hinted`] for a value the caller
    /// keeps in a register: it is copied only where it is stored. A store
    /// that a proxy takes (`table` stores nothing and its metatable has a
    /// `__newindex`) declines before any other work.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale table handle or a readonly table.
    #[allow(unsafe_code)]
    #[inline(never)]
    pub fn table_set_string_hinted_from(
        &mut self,
        table: TableHandle,
        key: &LuaString,
        source: &Value,
    ) -> RuntimeResult<bool> {
        if let Some(probe) = self.tables.get(table.0)
            && let Some(metatable) = probe.metatable
            && probe.entries.is_empty_ordered()
            && let Some(found) = self.metatable_newindex(metatable)
        {
            self.newindex_hint = found;
            return Ok(false);
        }
        let value = if let Value::String(text) = source {
            Value::String(text.clone())
        } else {
            // SAFETY: a value that is not a string owns no heap data, so a
            // bitwise copy is a valid second value.
            unsafe { std::ptr::read(source) }
        };
        self.table_set_string_owned_with::<true>(table, key, value)
    }

    /// Takes the closure that the last declining
    /// [`Self::table_set_string_if_no_newindex_hinted`] kept, leaving none.
    #[inline]
    pub fn take_newindex_hint(&mut self) -> Option<ClosureHandle> {
        self.newindex_hint.take()
    }

    /// [`Self::table_set_string_if_no_newindex`] for a value the caller
    /// keeps in a register: it is copied only where it is stored, and the
    /// table is resolved once.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale table handle or a readonly table.
    #[allow(unsafe_code)]
    #[inline]
    pub fn table_set_string_from(
        &mut self,
        table: TableHandle,
        key: &LuaString,
        source: &Value,
    ) -> RuntimeResult<bool> {
        // Only a string owns anything; the other variants are plain data,
        // so copying their bits skips the clone's dispatch.
        let value = if matches!(source, Value::String(_)) {
            source.clone()
        } else {
            // SAFETY: a non-string value owns no heap data, so a bitwise
            // copy is a valid second value.
            unsafe { std::ptr::read(source) }
        };
        self.table_set_string_owned(table, key, value)
    }

    /// The body of the string-key stores. A constructor's field stores take
    /// the shaped slot branch; every other store runs the general write on
    /// the table already resolved.
    #[inline(always)]
    fn table_set_string_owned(
        &mut self,
        table: TableHandle,
        key: &LuaString,
        value: Value,
    ) -> RuntimeResult<bool> {
        self.table_set_string_owned_with::<false>(table, key, value)
    }

    /// [`Self::table_set_string_owned`], which with `REPORT` also keeps a
    /// declining store's `__newindex` closure in `newindex_hint`.
    // Clippy's simplification for the absent-key probe rewrites the inner
    // `matches!` expansion into garbage; the boolean is already minimal.
    #[allow(clippy::nonminimal_bool)]
    #[inline(always)]
    fn table_set_string_owned_with<const REPORT: bool>(
        &mut self,
        table: TableHandle,
        key: &LuaString,
        value: Value,
    ) -> RuntimeResult<bool> {
        let changes_gc = key.cached_hash() == GC_FIELD_HASH && key.as_bytes() == b"__gc";
        // An absent key goes to `__newindex` if the metatable has one; a
        // stored nil counts as absent. Without one the store is raw. The
        // table is only read until the store is known to write: a store
        // that `__newindex` takes changes nothing, so it must not clear the
        // facts cached on the table or note a change, as `table_mut` does.
        let Some(probe) = self.tables.get_mut(table.0) else {
            return Err(RuntimeError::InvalidTable(table));
        };
        let mut absent = None;
        if let Some(metatable) = probe.metatable
            && (probe.entries.is_empty_ordered()
                || !probe
                    .entries
                    .get_string(key)
                    .is_some_and(|value| !matches!(value, Value::Nil)))
        {
            absent = Some(metatable);
        }
        let target = if let Some(metatable) = absent {
            if REPORT {
                // The same read as `metatable_lacks`, keeping the handler.
                if let Some(found) = self.metatable_newindex(metatable) {
                    self.newindex_hint = found;
                    return Ok(false);
                }
            } else if !self.metatable_lacks(metatable, Metamethod::NewIndex) {
                return Ok(false);
            }
            // Borrowed for the write: a table that is its own metatable has
            // just cached the absence of the key this store may add.
            self.table_mut(table)?
        } else {
            // What `table_mut` does, on the table already found.
            if self.gc_partial_table == Some(table) {
                self.gc_partial_dirty = true;
            }
            *probe.absent_metamethods.get_mut() = 0;
            if probe.watched {
                self.inherit_epoch += 1;
            }
            probe
        };
        if !changes_gc
            && !matches!(value, Value::Nil)
            && let Some(slot) = target.shaped_string_slot(key)
        {
            let barrier_value = value_gc_reference(&value);
            let previous = std::mem::replace(slot, value);
            let allocated = if matches!(previous, Value::Nil) {
                std::mem::size_of::<TableKey>() + std::mem::size_of::<Value>()
            } else {
                drop(previous);
                0
            };
            self.gc_charge(allocated);
            // A young container outside a major cycle has nothing to
            // remember: `gc_barrier` would test the same two things, out of
            // line.
            if let Some(reference) = barrier_value
                && (self.incremental_gc.is_some() || self.tables.is_old(table.0))
            {
                self.gc_barrier(Some(GcReference::Table(table)), reference);
            }
            return Ok(true);
        }
        // A scalar stored under a key the hashed table holds, as a value or as
        // a deleted key, goes in the one probe that finds it: it adds no
        // entry, so it charges the collector nothing, and it needs no barrier.
        // A present key never reaches `__newindex`. Shaped tables took their
        // slot above.
        if !changes_gc
            && matches!(
                value,
                Value::Boolean(_) | Value::Integer(_) | Value::Number(_)
            )
            && let Some((slot, live)) = target.string_slot_for_store(key)
        {
            if matches!(slot, Value::Nil) {
                *live += 1;
            }
            *slot = value;
            return Ok(true);
        }
        // A nil stored in a hashed table deletes the key in the one probe
        // that finds it, leaving a nil in its place, or finds it absent: what
        // the general store below comes to, without its owned key and its
        // three calls. Nothing is allocated and nil needs no barrier.
        if !changes_gc
            && matches!(value, Value::Nil)
            && !target.readonly
            && target.entries.tombstone_string(key).is_some()
        {
            return Ok(true);
        }
        let barrier_value = value_gc_reference(&value);
        let allocated = target.set_string(key, value)?;
        self.gc_charge(allocated);
        if let Some(reference) = barrier_value
            && (self.incremental_gc.is_some() || self.tables.is_old(table.0))
        {
            self.gc_barrier(Some(GcReference::Table(table)), reference);
        }
        if changes_gc {
            self.refresh_metatable_finalizers(table)?;
        }
        Ok(true)
    }

    /// The identity of `table`'s current shape for operation statistics:
    /// the shape's address, or 0 for a table without a shape or a stale
    /// handle. Reads nothing that can run guest code or allocate.
    #[cfg(feature = "op-stats")]
    #[must_use]
    pub fn table_shape_identity(&self, table: TableHandle) -> u64 {
        self.tables
            .get(table.0)
            .map_or(0, |table| table.entries.shape_identity())
    }
}

/// What [`Runtime::table_store_dense_scalar`] did.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseScalarStore {
    /// Nothing: the store takes the general path.
    Declined,
    /// A present number, integer or boolean took the value. Nothing was
    /// allocated.
    InPlace,
    /// The value is stored, and the collector may be charged for it.
    Stored,
}

/// What [`Runtime::table_site_store`] did.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SiteStore {
    /// Nothing: the store takes the general path.
    Missed,
    /// A present field took the value. Nothing was allocated.
    Overwrote,
    /// A shaped table gained the field, which the collector is charged for.
    Added,
}

/// The `__newindex` key, made on first use: only the 5.3+ table library asks,
/// so a runtime that never does allocates nothing for it.
fn newindex_key() -> &'static LuaString {
    static KEY: std::sync::OnceLock<LuaString> = std::sync::OnceLock::new();
    KEY.get_or_init(|| crate::string::shared_key(b"__newindex"))
}

/// Whether `value` is a boolean, an integer or a number: one subtraction and
/// one comparison on the tag, since those are the tags 1 to 3.
#[allow(clippy::inline_always)]
#[inline(always)]
fn is_plain_scalar(value: &Value) -> bool {
    matches!(
        value,
        Value::Boolean(_) | Value::Integer(_) | Value::Number(_)
    )
}

/// Copies a value that owns no heap data as a tag byte and a payload word,
/// each loaded on its own. A 16-byte load of a register that an arithmetic
/// result or a move wrote as two narrow stores cannot forward from them and
/// waits for both to leave the store buffer.
///
/// # Safety
///
/// Neither `to` nor `from` holds a string (or anything else that owns heap
/// data): `to` is overwritten without a drop.
#[allow(unsafe_code)]
#[inline(always)]
unsafe fn copy_value_words(to: &mut Value, from: &Value) {
    // SAFETY: `Value` is `repr(C, u8)`: the tag is byte 0 and the payload,
    // which may be partly uninitialised, the word at offset 8.
    unsafe {
        let (to, from) = (
            std::ptr::from_mut(to).cast::<u8>(),
            std::ptr::from_ref(from).cast::<u8>(),
        );
        let tag = from.read();
        let payload = from.add(8).cast::<std::mem::MaybeUninit<u64>>().read();
        to.write(tag);
        to.add(8)
            .cast::<std::mem::MaybeUninit<u64>>()
            .write(payload);
    }
}

/// The metatable word of a native row ([`Runtime::table_native_row`]): the
/// identity of the metatable, zero for none.
fn native_meta_word(metatable: Option<TableHandle>) -> usize {
    metatable.map_or(0, |metatable| {
        usize::try_from(crate::jit_ir::trace::meta_word(metatable.native_identity())).unwrap_or(0)
    })
}
