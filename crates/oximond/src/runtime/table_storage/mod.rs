//! The `Table` heap object: array part, entry map, and raw access.

pub(crate) mod array;
pub(crate) mod entries;
mod free_list;
pub(crate) mod key;
pub(crate) mod ordered_vec;
pub(crate) mod shape;

use crate::runtime::error::{RuntimeError, RuntimeResult};
use crate::runtime::native_views::NativeSlotKind;
use crate::runtime::number_format::{NumberFormat, NumberText, format_integer};
use crate::runtime::value::{TableHandle, Value};
use crate::string::LuaString;
pub(crate) use entries::Tombstoned;
use entries::{HASHED_TABLE_ENTRY_THRESHOLD, SPARSE_ARRAY_GROWTH_FLOOR, TableEntries};
use key::{TableKey, normalize_number_key};
use std::collections::BTreeSet;
use std::ops::Deref;

thread_local! {
    static KEY_LAYOUT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// A count of this thread's changes to where the unshaped tables keep
/// their entries: an ordered vector that gains, loses or moves an entry or
/// frees its block, a hashed map whose layout changes
/// (`IndexedTableEntries::layout_changed`), and the release of a hashed map
/// or a shaped block (a shaped table that becomes ordered). While it
/// stands still, the address of a value of an ordered or a hashed table is
/// still the value of the same key, and no ordered table gained a key.
/// A hashed table may still revive a deleted key in place, and any value
/// may change type.
///
/// The count is per OS thread, and a trace run that keeps its rows across a
/// host call assumes that only its own thread changed live tables during
/// the call. That holds while a runtime is used from one thread at a time:
/// the collector's marking threads only read, and the reclaim worker drops
/// dead tables. A C host that hands a `lua_State` to another OS thread in
/// the middle of a callback, and lets that thread change tables, would
/// change them behind the count.
#[inline]
pub(crate) fn key_layout() -> u64 {
    KEY_LAYOUT.with(std::cell::Cell::get)
}

/// Moves [`key_layout`] on.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(super) fn key_layout_changed() {
    KEY_LAYOUT.with(|count| count.set(count.get().wrapping_add(1)));
}

/// How far past the end of a non-empty array part a boolean store still grows
/// the array instead of going to the hash part.
pub(crate) const BOOLEAN_ARRAY_GROWTH_REACH: usize = 16;

#[derive(Default)]
pub(crate) struct Table {
    /// One-based integer keys which have reached the dense representation.
    /// Nil slots retain a hole until tail trimming can release them.
    pub(crate) array: array::ArrayPart,
    /// All non-array keys, including sparse positive integer keys.
    pub(crate) entries: TableEntries,
    /// Keys whose values were deleted after they had occupied table storage.
    /// Lua keeps dead hash nodes long enough for `next(t, current)` to resume
    /// when a loop deletes its current entry.
    pub(crate) deleted_array_keys: DeletedArrayKeys,
    pub(crate) deleted_entries: LazyDeletedKeys<TableKey>,
    /// The first-missing-key border for `1, 2, ...`.
    pub(crate) border: usize,
    pub(crate) metatable: Option<TableHandle>,
    pub(crate) readonly: bool,
    /// Set once the table is a metatable or an intermediate `__index` table
    /// of a cached inherited read (see `oximond::runtime::inherit`). A mutable borrow of
    /// a watched table invalidates those caches.
    pub(crate) watched: bool,
    /// One bit per [`Metamethod`](oximond::runtime::metatable::Metamethod) known to be
    /// absent from this table, as Lua's `flags` byte. A lookup that finds
    /// the key absent sets its bit; every mutable borrow through
    /// `Runtime::table_mut` clears them all. The paths that write without
    /// that borrow only overwrite present values or touch integer keys, so
    /// they never make an absent key present. Atomic only so that the
    /// collector's marking threads may share the arena.
    pub(crate) absent_metamethods: std::sync::atomic::AtomicU8,
    /// The array slots a remembered table received young stores into since
    /// the last minor collection. Valid only while the table is touched.
    pub(crate) gc_dirty: GcSlotRange,
    /// The array slots a minor collection must trace for this table when it
    /// is on the carry list. Empty for a table that is not carried.
    pub(crate) gc_carry: GcSlotRange,
}

/// The bytes a table is charged for. The collector's own range fields are
/// bookkeeping, so they stay out of the charge that paces collections.
pub(crate) const TABLE_CHARGED_BYTES: usize =
    std::mem::size_of::<Table>() - std::mem::size_of::<[GcSlotRange; 2]>();

/// The array slot a barrier names: by index, or by the address native code
/// wrote (resolved against the array part, and whole when it names no slot).
#[derive(Clone, Copy)]
pub(crate) enum ArraySlot {
    Index(usize),
    Address(usize),
}

/// A half-open range `[lo, hi)` of array slots that a minor collection traces
/// for a remembered table. [`Self::WHOLE`] means every slot, and it absorbs
/// any range it is unioned with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GcSlotRange {
    pub(crate) lo: u32,
    pub(crate) hi: u32,
}

impl GcSlotRange {
    /// No slot.
    pub(crate) const EMPTY: Self = Self { lo: 0, hi: 0 };
    /// Every slot, whatever the array holds.
    pub(crate) const WHOLE: Self = Self {
        lo: 0,
        hi: u32::MAX,
    };

    /// The range of the single slot at `index`.
    #[inline]
    pub(crate) fn one(index: usize) -> Self {
        match u32::try_from(index) {
            Ok(index) if index < u32::MAX - 1 => Self {
                lo: index,
                hi: index + 1,
            },
            _ => Self::WHOLE,
        }
    }

    #[inline]
    pub(crate) const fn is_empty(self) -> bool {
        self.lo >= self.hi
    }

    #[inline]
    pub(crate) const fn is_whole(self) -> bool {
        self.hi == u32::MAX
    }

    /// The smallest range covering both.
    #[inline]
    pub(crate) fn union(self, other: Self) -> Self {
        if self.is_empty() {
            other
        } else if other.is_empty() {
            self
        } else {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.max(other.hi),
            }
        }
    }
}

impl Table {
    /// The range of the array slot at `address`, whole when it is not a slot
    /// of the array part's buffer.
    #[inline]
    pub(crate) fn array_slot_range(&self, address: usize) -> GcSlotRange {
        let base = self.array.as_ptr() as usize;
        let size = std::mem::size_of::<Value>();
        let end = base + self.array.capacity() * size;
        if address >= base && address < end && (address - base).is_multiple_of(size) {
            GcSlotRange::one((address - base) / size)
        } else {
            GcSlotRange::WHOLE
        }
    }

    /// Whether the slots a minor collection traces for this table already
    /// include the array slot at `address`, so a store into it widens
    /// nothing. `false` for an address that names no slot, and for a slot
    /// whose index [`GcSlotRange::one`] does not take: the caller then asks
    /// [`Self::array_slot_range`], which answers both with the whole array.
    #[inline]
    pub(crate) fn gc_dirty_holds_address(&self, address: usize) -> bool {
        let size = std::mem::size_of::<Value>();
        let offset = address.wrapping_sub(self.array.as_ptr() as usize);
        let index = offset / size;
        offset.is_multiple_of(size)
            && index < self.array.capacity()
            && self.gc_dirty.lo as usize <= index
            && index < self.gc_dirty.hi as usize
    }

    /// Notes that an insert or remove moved values between array slots: a
    /// table already traced by range now traces the whole array. A table
    /// with both ranges empty is neither remembered nor carried, so it holds
    /// no young child and the move changes nothing.
    #[inline]
    pub(crate) fn note_array_moved(&mut self) {
        if !self.gc_dirty.is_empty() {
            self.gc_dirty = GcSlotRange::WHOLE;
        }
        if !self.gc_carry.is_empty() {
            self.gc_carry = GcSlotRange::WHOLE;
        }
    }

    /// Writes an empty table over `array` and `entries` into `slot`, field
    /// by field. A table built on the stack and copied is written with
    /// narrow stores and read back with wide loads, which cannot forward
    /// from them; written here, each field is stored once, where it stays.
    /// A panic in `array` or `entries` leaks the parts written so far.
    #[allow(unsafe_code, clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn init_parts(
        slot: &mut std::mem::MaybeUninit<Self>,
        array: impl FnOnce() -> array::ArrayPart,
        entries: impl FnOnce() -> TableEntries,
    ) {
        let table = slot.as_mut_ptr();
        // SAFETY: `table` points to writable memory for a `Table`, and every
        // field is written exactly once, none read.
        unsafe {
            // Each part is written the moment it exists, so none waits in a
            // stack slot for the other to be built.
            (&raw mut (*table).array).write(array());
            (&raw mut (*table).entries).write(entries());
            (&raw mut (*table).deleted_array_keys).write(DeletedArrayKeys::default());
            (&raw mut (*table).deleted_entries).write(LazyDeletedKeys::default());
            (&raw mut (*table).border).write(0);
            (&raw mut (*table).metatable).write(None);
            (&raw mut (*table).readonly).write(false);
            (&raw mut (*table).watched).write(false);
            (&raw mut (*table).absent_metamethods).write(std::sync::atomic::AtomicU8::new(0));
            (&raw mut (*table).gc_dirty).write(GcSlotRange::EMPTY);
            (&raw mut (*table).gc_carry).write(GcSlotRange::EMPTY);
        }
    }

    /// Writes into `slot` a table whose array part is `count` nil slots
    /// that count as present, and returns the address of the first: what
    /// [`Self::prefill_array`] makes of a new table with room for `count`
    /// values, in one step.
    #[allow(unsafe_code, clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn init_prefilled(slot: &mut std::mem::MaybeUninit<Self>, count: usize) -> usize {
        Self::init_parts(
            slot,
            || array::ArrayPart::nils(count),
            TableEntries::default,
        );
        let table = slot.as_mut_ptr();
        // SAFETY: `init_parts` wrote every field of the table.
        unsafe {
            (&raw mut (*table).border).write(count);
            (*table).array.as_mut_ptr() as usize
        }
    }

    /// Writes into `slot` an empty table whose string keys are those of
    /// `shape`, and returns the address of its first value slot: the slot of
    /// the shape's key `i` is `i` values further.
    #[allow(unsafe_code, clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn init_shaped(
        slot: &mut std::mem::MaybeUninit<Self>,
        shape: &std::sync::Arc<shape::TableShape>,
    ) -> usize {
        Self::init_parts(slot, array::ArrayPart::new, || {
            TableEntries::with_shape_ref(shape)
        });
        // SAFETY: `init_parts` wrote every field of the table.
        let table = unsafe { slot.assume_init_mut() };
        table
            .entries
            .shaped_slots()
            .expect("a table made with a shape is shaped")
    }
}

#[allow(
    clippy::box_collection,
    reason = "The box keeps the empty-table footprint small by deferring the BTreeSet header."
)]
pub(crate) struct LazyDeletedKeys<T>(pub(crate) Option<Box<BTreeSet<T>>>);

impl<T> Default for LazyDeletedKeys<T> {
    fn default() -> Self {
        Self(None)
    }
}

/// The deleted array keys of a table: [`LazyDeletedKeys`] plus keys that
/// native code popped off the end of a dense list without recording them.
///
/// A trace that pops the last element sets the array's length down and raises
/// `pending` to the key it took out; it cannot insert into a `BTreeSet`. Every
/// key in `len + 1..=pending` (`len` is the array's length) counts as
/// recorded, which is exactly the set of records the interpreter's pops would
/// have left: a pop records the key at the end and a push over a recorded key
/// forgets it, and the keys in between were all popped. Every method takes the
/// array's length, and anything that changes the set other than by shrinking
/// that range from below (a push at `len + 1`) first moves the range into the
/// set, so the records are always the ones the interpreter keeps.
#[derive(Default)]
pub(crate) struct DeletedArrayKeys {
    keys: LazyDeletedKeys<usize>,
    /// Offset of this field in [`Table`] is read by native code.
    pub(crate) pending: usize,
}

impl DeletedArrayKeys {
    /// Moves the pending range into the set, returning the bytes the set
    /// newly occupies.
    #[inline]
    fn settle(&mut self, len: usize) -> usize {
        let pending = std::mem::take(&mut self.pending);
        let mut allocated = 0;
        if pending > len {
            for key in len + 1..=pending {
                allocated += self.keys.insert_accounted(key);
            }
        }
        allocated
    }

    #[inline]
    pub(crate) fn is_empty(&self, len: usize) -> bool {
        self.pending <= len && self.keys.is_empty()
    }

    /// Whether the set holds no record (a pending range is not in it).
    pub(crate) fn explicit_is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    #[inline]
    pub(crate) fn contains(&self, key: usize, len: usize) -> bool {
        (len < key && key <= self.pending) || self.keys.contains(&key)
    }

    /// Records `key`, returning the bytes newly occupied (including those of
    /// a pending range this had to move into the set).
    #[inline]
    fn insert_accounted(&mut self, key: usize, len: usize) -> usize {
        let allocated = self.settle(len);
        allocated + self.keys.insert_accounted(key)
    }

    #[inline]
    fn remove(&mut self, key: usize, len: usize) -> bool {
        // Only a key inside the pending range makes a hole in it.
        if len < key && key <= self.pending {
            self.settle(len);
        }
        self.keys.remove(&key)
    }

    /// Forgets the record of `len + 1`, the key a push is about to give the
    /// array: the pending range shrinks from below by itself when the array
    /// grows.
    #[inline]
    fn forget_push(&mut self, key: usize) {
        self.keys.remove(&key);
    }

    /// Records the key of the last element of the array, which was just
    /// deleted and will be trimmed off the end, as a pending deletion.
    #[inline]
    fn record_tail(&mut self, key: usize) {
        self.pending = self.pending.max(key);
    }

    #[inline]
    fn remove_range(&mut self, low: usize, high: usize, len: usize) {
        // A range inside the array leaves the pending range alone.
        if high > len {
            self.settle(len);
        }
        self.keys.remove_range(low, high);
    }

    /// Takes one record out, for dismantling a dead table: the pending range
    /// gives its highest key.
    pub(crate) fn pop_first(&mut self, len: usize) -> Option<usize> {
        if let Some(key) = self.keys.pop_first() {
            return Some(key);
        }
        if self.pending > len {
            let key = self.pending;
            self.pending -= 1;
            return Some(key);
        }
        self.pending = 0;
        None
    }

    pub(crate) fn allocated_bytes(&self) -> usize {
        self.keys.allocated_bytes()
    }
}

impl<T: Ord> LazyDeletedKeys<T> {
    pub(crate) fn is_empty(&self) -> bool {
        self.0.as_ref().is_none_or(|keys| keys.is_empty())
    }

    pub(crate) fn contains(&self, key: &T) -> bool {
        self.0.as_ref().is_some_and(|keys| keys.contains(key))
    }

    fn insert(&mut self, key: T) -> bool {
        self.0
            .get_or_insert_with(|| Box::new(BTreeSet::new()))
            .insert(key)
    }

    fn insert_accounted(&mut self, key: T) -> usize {
        let allocated = self.allocated_bytes();
        self.insert(key)
            .then(|| self.allocated_bytes().saturating_sub(allocated))
            .unwrap_or(0)
    }

    fn remove(&mut self, key: &T) -> bool {
        self.0.as_mut().is_some_and(|keys| keys.remove(key))
    }

    /// Forgets every record in `low..=high`, as removing each one in turn
    /// does.
    fn remove_range(&mut self, low: T, high: T)
    where
        T: Copy,
    {
        let Some(keys) = self.0.as_mut() else {
            return;
        };
        if low > high || keys.is_empty() {
            return;
        }
        let found: Vec<T> = keys.range(low..=high).copied().collect();
        for key in found {
            keys.remove(&key);
        }
    }

    pub(crate) fn pop_first(&mut self) -> Option<T> {
        self.0.as_mut().and_then(|keys| keys.pop_first())
    }

    pub(crate) fn allocated_bytes(&self) -> usize {
        self.0.as_ref().map_or(0, |keys| {
            std::mem::size_of::<BTreeSet<T>>()
                + keys.len() * (std::mem::size_of::<T>() + 3 * std::mem::size_of::<usize>())
        })
    }
}

impl TableKey {
    pub(crate) fn value(&self) -> Value {
        match self {
            Self::Boolean(value) => Value::Boolean(*value),
            Self::Integer(value) => Value::Integer(*value),
            Self::Number(value) => Value::Number(f64::from_bits(*value)),
            Self::String(value) => Value::String(value.clone()),
            Self::Table(value) => Value::Table(*value),
            Self::File(value) => Value::File(*value),
            Self::Userdata(value) => Value::Userdata(*value),
            Self::GmatchIterator(value) => Value::GmatchIterator(*value),
            Self::Thread(value) => Value::Thread(*value),
            Self::CoroutineWrapper(value) => Value::CoroutineWrapper(*value),
            Self::NativeFunction(value) => Value::NativeFunction(*value),
            Self::Closure(value) => Value::Closure(*value),
            Self::BuiltinFunction(value) => Value::BuiltinFunction(*value),
            Self::Int64(value) => Value::Int64(*value),
            Self::UInt64(value) => Value::UInt64(*value),
        }
    }
}

impl Table {
    pub(crate) fn ensure_writable(&self) -> RuntimeResult<()> {
        if self.readonly {
            Err(RuntimeError::Message(
                "Attempt to modify a readonly table".into(),
            ))
        } else {
            Ok(())
        }
    }
    #[inline]
    pub(crate) fn get_value(&self, key: &Value) -> RuntimeResult<Value> {
        Ok(self.get_raw_value(key)?.cloned().unwrap_or(Value::Nil))
    }

    /// Stores the number `x` under the key `index` when the table is in the
    /// plain state [`Self::set_array`] handles with no bookkeeping beyond the
    /// write: writable, no hash entries or pending deletions, and the key
    /// either an overwrite at or below the border or the next append at the
    /// border. Returns the newly occupied bytes, or `None` for anything else,
    /// which then takes [`Self::set_index`]. The value is built in the array
    /// slot itself, so it is not copied through the stack with a wide load
    /// right after narrow stores.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn set_number_fast(&mut self, index: usize, x: f64) -> Option<usize> {
        if self.readonly
            || index == 0
            || !self.entries.is_empty()
            || !self.deleted_entries.is_empty()
            || !self.deleted_array_keys.is_empty(self.array.len())
        {
            return None;
        }
        if index <= self.border {
            self.array[index - 1] = Value::Number(x);
            Some(0)
        } else if index == self.border + 1 && self.border == self.array.len() {
            self.array.push(Value::Number(x));
            self.border = self.array.len();
            Some(std::mem::size_of::<Value>())
        } else {
            None
        }
    }

    /// [`Self::set_value`] for the key `index`, a positive integer that
    /// [`array_index`] would return for it. Skips the key conversion.
    #[inline]
    pub(crate) fn set_index(&mut self, index: usize, value: Value) -> RuntimeResult<usize> {
        self.ensure_writable()?;
        Ok(self.set_array(index, value))
    }

    /// String lookup that borrows its key, mirroring the string arm of
    /// [`Self::get_raw_value`] without requiring an owned [`Value::String`].
    pub(crate) fn get_borrowed_string(&self, key: &LuaString) -> Option<Value> {
        self.entries
            .get_string(key)
            .filter(|value| !matches!(value, Value::Nil))
            .cloned()
    }

    #[inline]
    pub(crate) fn get_raw_value(&self, key: &Value) -> RuntimeResult<Option<&Value>> {
        if matches!(key, Value::Nil) || matches!(key, Value::Number(value) if value.is_nan()) {
            return Ok(None);
        }
        if let Some(index) = array_index(key)
            && let Some(value) = self.array.get(index - 1)
        {
            return Ok((!matches!(value, Value::Nil)).then_some(value));
        }
        if let Value::String(string) = key {
            return Ok(self
                .entries
                .get_string(string)
                .filter(|value| !matches!(value, Value::Nil)));
        }
        let key = TableKey::try_from(key)?;
        Ok(self
            .entries
            .get(&key)
            .filter(|value| !matches!(value, Value::Nil)))
    }

    /// [`Self::get_raw_value`] for callers that treat an absent value and
    /// an invalid key alike. Array hits resolve inline; the result comes
    /// back in a register instead of through a `Result` in memory. String
    /// keys call an out-of-line leaf.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn get_raw_hit(&self, key: &Value) -> Option<&Value> {
        self.raw_hit::<false>(key)
    }

    /// [`Self::get_raw_hit`] with the ordered string scan inlined. For the
    /// VM's table-field read only.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn get_raw_hit_scan(&self, key: &Value) -> Option<&Value> {
        self.raw_hit::<true>(key)
    }

    /// The array part's slot for the 1-based `index`, if it has one.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn array_slot(&self, index: usize) -> Option<&Value> {
        self.array.get(index.checked_sub(1)?)
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn raw_hit<const SCAN: bool>(&self, key: &Value) -> Option<&Value> {
        let offset = match key {
            Value::String(string) => {
                return if SCAN {
                    self.entries.get_string_live_scan_ordered(string)
                } else {
                    self.entries.get_string_live(string)
                };
            }
            _ => array_offset(key),
        };
        // One unsigned compare covers both ends: an offset below zero wraps
        // to a huge value, so it never indexes the array.
        if offset < self.array.len() as u64 {
            let value = &self.array[offset as usize];
            return (!matches!(value, Value::Nil)).then_some(value);
        }
        self.get_raw_value(key).ok().flatten()
    }

    /// [`Self::get_raw_value_mut`] for native code: the slot's address, when
    /// `accept` takes its value. Addresses come from the storage's base
    /// pointer, so binding a second slot does not invalidate the first.
    pub(crate) fn raw_value_ptr(
        &mut self,
        key: &Value,
        accept: impl Fn(&Value) -> bool,
    ) -> RuntimeResult<Option<*mut Value>> {
        if matches!(key, Value::Nil) || matches!(key, Value::Number(value) if value.is_nan()) {
            return Ok(None);
        }
        if let Some(index) = array_index(key)
            && let Some(value) = self.array.get(index - 1)
        {
            return Ok(accept(value).then(|| self.array.as_mut_ptr().wrapping_add(index - 1)));
        }
        let key = TableKey::try_from(key)?;
        if !self
            .entries
            .get(&key)
            .is_some_and(|value| !matches!(value, Value::Nil) && accept(value))
        {
            return Ok(None);
        }
        Ok(self.entries.value_ptr(&key))
    }

    /// [`TableEntries::string_value_ptrs`] over this table's hash part.
    pub(crate) fn string_value_ptrs(
        &mut self,
        wanted: &[(Value, NativeSlotKind)],
        hints: &mut [usize],
        out: &mut [usize],
        absent: Option<usize>,
    ) {
        self.entries.string_value_ptrs(wanted, hints, out, absent);
    }

    pub(crate) fn insert_entry(&mut self, key: TableKey, value: Value) -> usize {
        let indexed = self.entries.is_indexed();
        let may_promote = self.entries.len().saturating_add(1) >= HASHED_TABLE_ENTRY_THRESHOLD;
        let allocated = (indexed || may_promote).then(|| self.entries.allocated_bytes());
        // The insertion already tells us whether the key was present. A
        // separate contains_key search doubles the work on hot overwrites.
        let is_new = self.entries.insert(key, value).is_none();
        if !indexed && self.entries.len() >= HASHED_TABLE_ENTRY_THRESHOLD {
            self.entries.promote();
        }
        allocated.map_or_else(
            || {
                usize::from(is_new)
                    * (std::mem::size_of::<TableKey>() + std::mem::size_of::<Value>())
            },
            |allocated| self.entries.allocated_bytes().saturating_sub(allocated),
        )
    }

    pub(crate) fn has_raw_value(&self, key: &Value) -> RuntimeResult<bool> {
        if let Some(index) = array_index(key) {
            if self
                .array
                .get(index - 1)
                .is_some_and(|value| !matches!(value, Value::Nil))
            {
                return Ok(true);
            }
            let key = array_table_key(index).expect("array indexes have numeric table keys");
            return Ok(self.entries.contains_key(&key));
        }
        // A tombstone is an absent key, as a nil field is.
        Ok(self
            .entries
            .get(&TableKey::try_from(key)?)
            .is_some_and(|value| !matches!(value, Value::Nil)))
    }

    /// Writes a non-nil value to an existing array slot, or appends it at
    /// the border, when the table has no hash entries or deletion records
    /// that [`Self::set_array`] would have to reconcile. Returns the bytes
    /// newly occupied, as [`Self::set_value`] does, or hands the value back
    /// having changed nothing.
    #[inline]
    pub(crate) fn set_dense(&mut self, key: &Value, value: Value) -> Result<usize, Value> {
        if self.readonly {
            return Err(value);
        }
        if !self.entries.is_empty()
            || !self.deleted_entries.is_empty()
            || !self.deleted_array_keys.is_empty(self.array.len())
        {
            return Err(value);
        }
        let Some(index) = array_index(key) else {
            return Err(value);
        };
        if index <= self.array.len() {
            self.array[index - 1] = value;
            return Ok(if index == self.border.saturating_add(1) {
                self.advance_border()
            } else {
                0
            });
        }
        if index == self.array.len() + 1 && self.border == self.array.len() {
            self.array.push(value);
            self.border = self.array.len();
            return Ok(std::mem::size_of::<Value>());
        }
        Err(value)
    }

    /// Appends the number, integer or boolean `value` right past a full
    /// array part that has room for it, in a table with no hash entries and
    /// no deleted-key records: the next store of a host filling an array.
    /// Each test is one comparison, the cheap ones first, and the value is
    /// copied as its tag and payload words. A leaf without calls; `false`,
    /// having changed nothing, for anything else, which
    /// [`Self::set_dense_scalar_at`] handles. An append occupies one value.
    #[allow(clippy::inline_always, unsafe_code)]
    #[inline(always)]
    pub(crate) fn append_scalar_in_place(&mut self, value: &Value) -> bool {
        let length = self.array.len();
        if length == self.array.capacity()
            || self.border != length
            || !matches!(
                value,
                Value::Boolean(_) | Value::Integer(_) | Value::Number(_)
            )
            || !self.entries.is_empty_ordered()
            || !self.deleted_entries.is_empty()
            || !self.deleted_array_keys.is_empty(length)
        {
            return false;
        }
        // SAFETY: the buffer has room past `length`; the value is a scalar,
        // whose tag and payload words are all of it, and the slot holds them
        // before the length takes it in.
        unsafe {
            let slot = self.array.as_mut_ptr().add(length).cast::<u8>();
            let from = std::ptr::from_ref(value).cast::<u8>();
            slot.write(from.read());
            slot.add(8)
                .cast::<std::mem::MaybeUninit<u64>>()
                .write(from.add(8).cast::<std::mem::MaybeUninit<u64>>().read());
            self.array.set_len(length + 1);
        }
        self.border = length + 1;
        true
    }

    /// [`Self::set_dense_scalar`] at a one-based `index` the caller already
    /// took from the key with [`array_offset`], for a table it found writable.
    /// Skips recomputing the index from the key.
    #[inline(always)]
    pub(crate) fn set_dense_scalar_at(&mut self, index: usize, value: &Value) -> Option<usize> {
        let copy = match value {
            Value::Number(number) => Value::Number(*number),
            Value::Integer(number) => Value::Integer(*number),
            Value::Boolean(flag) => Value::Boolean(*flag),
            _ => return None,
        };
        if !self.entries.is_empty()
            || !self.deleted_entries.is_empty()
            || !self.deleted_array_keys.is_empty(self.array.len())
        {
            return None;
        }
        let length = self.array.len();
        if index <= length {
            let slot = &mut self.array[index - 1];
            // Releasing a string is left to the general path.
            if matches!(slot, Value::String(_)) {
                return None;
            }
            // Only a string owns a resource.
            std::mem::forget(std::mem::replace(slot, copy));
            return Some(if index == self.border.saturating_add(1) {
                self.advance_border()
            } else {
                0
            });
        }
        self.append_dense_scalar(index, copy)
    }

    /// [`Self::set_dense`] for a number, integer, or boolean `value`, copied
    /// field by field instead of moved: a moved [`Value`] is built with
    /// narrow stores and copied with one wide load that cannot forward from
    /// them. Returns `None`, having changed nothing, where `set_dense` would
    /// hand the value back, and for any other kind of value.
    #[inline(always)]
    pub(crate) fn set_dense_scalar(&mut self, key: &Value, value: &Value) -> Option<usize> {
        if self.readonly {
            return None;
        }
        self.set_dense_scalar_at(array_index(key)?, value)
    }

    /// [`Self::set_dense`] for a string, table, closure or function `value`,
    /// cloned from a borrow into the slot. Returns `None`, having changed
    /// nothing, where `set_dense` would hand the value back, and for a value
    /// of any other kind. The caller owes the write barrier of a table or
    /// closure.
    #[inline(always)]
    pub(crate) fn set_dense_reference(&mut self, key: &Value, value: &Value) -> Option<usize> {
        if self.readonly
            || !self.entries.is_empty()
            || !self.deleted_entries.is_empty()
            || !self.deleted_array_keys.is_empty(self.array.len())
        {
            return None;
        }
        let index = array_index(key)?;
        // One dispatch on the value's kind both filters and copies it.
        let copy = clone_reference(value)?;
        let length = self.array.len();
        if index <= length {
            self.array[index - 1] = copy;
            return Some(if index == self.border.saturating_add(1) {
                self.advance_border()
            } else {
                0
            });
        }
        if index == length + 1 && self.border == length {
            self.append_unchecked(copy);
            self.border = length + 1;
            return Some(std::mem::size_of::<Value>());
        }
        // `copy` drops here, which releases the string it holds.
        None
    }

    /// Makes room for one more array value: a recycled buffer for the first
    /// ones, `Vec`'s doubling after that.
    #[inline(never)]
    pub(crate) fn grow_array(&mut self) {
        if self.array.capacity() == 0 {
            self.array.start(4);
        } else {
            self.array.reserve(1);
        }
    }

    /// `Vec::push` for [`Self::set_dense_reference`], which builds the value in
    /// the new slot: `push` is a call that takes the value from the stack.
    #[allow(unsafe_code, clippy::inline_always)]
    #[inline(always)]
    fn append_unchecked(&mut self, value: Value) {
        let length = self.array.len();
        if length == self.array.capacity() {
            self.grow_array();
        }
        self.array.spare_capacity_mut()[0].write(value);
        // SAFETY: the capacity exceeds `length`, and the slot at `length`
        // was just written.
        unsafe { self.array.set_len(length + 1) };
    }

    /// The push of [`Self::set_dense_scalar`], kept out of line so the
    /// overwrite beside it stays small. The value is taken apart and built
    /// again in its slot: a `Value` moved whole is loaded with one wide load
    /// from the narrow stores that built it, which cannot forward and stalls
    /// the pipeline.
    #[inline(never)]
    fn append_dense_scalar(&mut self, index: usize, copy: Value) -> Option<usize> {
        let length = self.array.len();
        if index == length + 1 && self.border == length {
            if length == self.array.capacity() {
                self.grow_array();
            }
            let slot = &mut self.array.spare_capacity_mut()[0];
            match copy {
                Value::Number(number) => slot.write(Value::Number(number)),
                Value::Integer(number) => slot.write(Value::Integer(number)),
                Value::Boolean(flag) => slot.write(Value::Boolean(flag)),
                _ => return None,
            };
            // SAFETY: the capacity exceeds `length`, and the slot at `length`
            // was just written.
            #[allow(unsafe_code)]
            unsafe {
                self.array.set_len(length + 1);
            }
            self.border = length + 1;
            return Some(std::mem::size_of::<Value>());
        }
        // A boolean a few slots past the end grows the array part, as
        // `set_array` decides for sieve-style tables. With no hash entries
        // there is nothing to absorb, so the growth is the holes and the value.
        if matches!(copy, Value::Boolean(_))
            && length != 0
            && index <= length.saturating_add(BOOLEAN_ARRAY_GROWTH_REACH)
        {
            self.array.resize_with(index, || Value::Nil);
            self.array[index - 1] = copy;
            return Some((index - length) * std::mem::size_of::<Value>() + self.advance_border());
        }
        None
    }

    /// Gives a table nothing was stored in an array part of `count` nil
    /// slots that count as present, and returns the address of the first slot.
    /// Native code overwrites every slot with a value before anything else
    /// reads the table, so the table is a dense list of `count` values again.
    /// `None`, having changed nothing, for a table with contents.
    pub(crate) fn prefill_array(&mut self, count: usize) -> Option<usize> {
        if count == 0
            || self.readonly
            || !self.array.is_empty()
            || self.border != 0
            || !self.entries.is_empty()
            || !self.deleted_entries.is_empty()
        {
            return None;
        }
        self.array.resize_with(count, || Value::Nil);
        self.border = count;
        Some(self.array.as_mut_ptr() as usize)
    }

    /// Whether the array part holds every integer key of the table up to its
    /// border and has no hole: a border at the end of the array, no record of
    /// a deleted hash key, and a table the interpreter may write. The hash
    /// part may hold other keys (it holds none inside the array, and none at
    /// the key after it, which the border would have absorbed).
    /// `table.insert` and `table.remove` then act on the array alone (see
    /// [`Self::trace_insert`]).
    #[inline]
    fn is_dense_list(&self) -> bool {
        !self.readonly && self.deleted_entries.is_empty() && self.border == self.array.len()
    }

    /// The length of the table when it is a dense list.
    #[inline]
    pub(crate) fn dense_length(&self) -> Option<usize> {
        self.is_dense_list().then_some(self.array.len())
    }

    /// `table.insert (t, value)` or `table.insert (t, position, value)` of
    /// the interpreter for a list that is dense, as one move of the elements
    /// after `position`. `position` is the one-based place the value takes,
    /// `None` for the end. Hands back the bytes newly occupied, as
    /// [`Self::set_value`] does, or `None`, having changed nothing, for a
    /// table or a call the interpreter's loop of raw gets and sets may treat
    /// differently: not a dense list, a position outside `1..=#t + 1`, nil.
    ///
    /// The loop it stands for pushes the last element one place up, shifts the
    /// rest and overwrites the place. Each of those sets forgets the record
    /// of a deleted array key it lands on, which is every key from `position`
    /// through `#t + 1`, and the push charges one slot.
    pub(crate) fn trace_insert(&mut self, position: Option<usize>, value: &Value) -> Option<usize> {
        if matches!(value, Value::Nil) {
            return None;
        }
        let position = self.trace_insert_place(position)?;
        let length = self.array.len();
        self.deleted_array_keys
            .remove_range(position, length, length);
        self.deleted_array_keys.forget_push(length + 1);
        self.array.insert(position - 1, value.clone());
        self.border = length + 1;
        Some(std::mem::size_of::<Value>())
    }

    /// The one-based place an insert at `position` (`None` for the end) takes
    /// in a table [`Self::trace_insert`] accepts, or `None` where it declines.
    // Inlined into `trace_insert`, whose helper call it is the larger part of:
    // left to the compiler it moves in and out as the code around it changes.
    #[inline(always)]
    pub(crate) fn trace_insert_place(&self, position: Option<usize>) -> Option<usize> {
        if !self.is_dense_list() {
            return None;
        }
        let length = self.array.len();
        let position = position.unwrap_or_else(|| length.saturating_add(1));
        if position < 1 || position > length.checked_add(1)? {
            return None;
        }
        // The push takes in the keys after it that the hash part holds.
        if !self.entries.is_empty()
            && (1..=2).any(|ahead| {
                length
                    .checked_add(ahead)
                    .and_then(array_table_key)
                    .is_none_or(|key| self.entries.contains_key(&key))
            })
        {
            return None;
        }
        Some(position)
    }

    /// The element `table.remove (t)` or `table.remove (t, position)` of the
    /// interpreter takes out of a dense list, if `accept` takes the value
    /// there, with the bytes newly occupied. `None`, having changed nothing,
    /// where [`Self::trace_insert`] would decline, for a position outside
    /// `1..=#t` (the call gives nil and changes nothing, see
    /// [`Self::trace_remove_nothing`]) and for a value `accept` refuses.
    ///
    /// The loop it stands for shifts the elements after `position` down and
    /// sets the last place to nil. The shifting sets forget the records of
    /// deleted array keys from `position` through `#t - 1`, and the last one
    /// records the key `#t` as deleted, which is a pending deletion
    /// (see [`DeletedArrayKeys`]) and charges nothing.
    pub(crate) fn trace_remove(
        &mut self,
        position: Option<usize>,
        accept: impl FnOnce(&Value) -> bool,
    ) -> Option<(Value, usize)> {
        if !self.is_dense_list() {
            return None;
        }
        let length = self.array.len();
        let position = position.unwrap_or(length);
        if position < 1 || position > length || !accept(&self.array[position - 1]) {
            return None;
        }
        self.deleted_array_keys
            .remove_range(position, length - 1, length);
        // The key `length` is the one deleted: the end of the pending range.
        self.deleted_array_keys.record_tail(length);
        let removed = self.array.remove(position - 1);
        self.border = length - 1;
        Some((removed, 0))
    }

    /// The element [`Self::trace_remove`] would take out, or `None` where it
    /// would decline.
    #[inline]
    pub(crate) fn trace_removed(&self, position: Option<usize>) -> Option<&Value> {
        if !self.is_dense_list() {
            return None;
        }
        let length = self.array.len();
        let position = position.unwrap_or(length);
        (1..=length)
            .contains(&position)
            .then(|| &self.array[position - 1])
    }

    /// Whether `table.remove (t, position)` gives nil and changes nothing:
    /// the position is outside `1..=#t`, which includes any position of an
    /// empty list.
    #[inline]
    pub(crate) fn trace_remove_nothing(&self, position: Option<i64>) -> bool {
        let length = i64::try_from(self.border).unwrap_or(i64::MAX);
        let position = position.unwrap_or(length);
        position < 1 || position > length
    }

    /// Returns bytes newly occupied by table storage. Replacements and
    /// removals do not add debt, so a hot overwrite loop cannot schedule GC.
    pub(crate) fn set_value(&mut self, key: &Value, value: Value) -> RuntimeResult<usize> {
        self.ensure_writable()?;
        if let Some(index) = array_index(key) {
            return Ok(self.set_array(index, value));
        }
        if let Value::String(string) = key {
            return self.set_string_entry(string, value);
        }
        self.set_entry(TableKey::try_from(key)?, value)
    }

    /// [`Self::set_value`] for a string key, which it borrows.
    pub(crate) fn set_string(&mut self, key: &LuaString, value: Value) -> RuntimeResult<usize> {
        self.ensure_writable()?;
        self.set_string_entry(key, value)
    }

    /// Whether [`Self::overwrite_scalar_string`] would store `value`.
    pub(crate) fn can_overwrite_scalar_string(&self, key: &LuaString, value: &Value) -> bool {
        !self.readonly
            && matches!(
                value,
                Value::Integer(_) | Value::Number(_) | Value::Boolean(_)
            )
            && self
                .entries
                .get_string_live(key)
                .is_some_and(|held| std::mem::discriminant(held) == std::mem::discriminant(value))
    }

    /// Overwrites the value of the present string `key` with the integer,
    /// float or boolean `value` when it holds one of the same type. The
    /// overwrite allocates nothing, moves no entry and changes no layout, so a
    /// traversal that holds the entries' addresses stays valid. `false`,
    /// having changed nothing, for any other store.
    pub(crate) fn overwrite_scalar_string(&mut self, key: &LuaString, value: &Value) -> bool {
        if self.readonly
            || !matches!(
                value,
                Value::Integer(_) | Value::Number(_) | Value::Boolean(_)
            )
        {
            return false;
        }
        match self.entries.get_string_mut(key) {
            Some(slot) if std::mem::discriminant(slot) == std::mem::discriminant(value) => {
                // Only a scalar gets here, and it owns no heap data.
                match (&mut *slot, value) {
                    (Value::Integer(held), Value::Integer(new)) => *held = *new,
                    (Value::Number(held), Value::Number(new)) => *held = *new,
                    (Value::Boolean(held), Value::Boolean(new)) => *held = *new,
                    _ => return false,
                }
                true
            }
            _ => false,
        }
    }

    /// The slot of `key` and the table's live-entry counter, where a hashed
    /// table that is writable and records no deleted keys of the other kind
    /// holds the key, as a value or as a deleted key (a nil), found in one
    /// probe. A store of a non-nil value there adds no entry and moves none;
    /// the caller counts a deleted key live again. `None` for any other
    /// table or key, which the general store takes.
    #[inline]
    pub(crate) fn string_slot_for_store(
        &mut self,
        key: &LuaString,
    ) -> Option<(&mut Value, &mut usize)> {
        if self.readonly || !self.deleted_entries.is_empty() {
            return None;
        }
        self.entries.indexed_slot_live(key)
    }

    /// Deletes the string `key` of a hashed table by leaving a nil in its
    /// place, in one probe (see [`TableEntries::tombstone_string`]). The
    /// entries stay where they are and the layout number does not change, so a
    /// traversal that holds their addresses stays valid and skips the nil.
    /// Says why it did not delete: for the key `__gc`, whose deletion changes
    /// the finalizer, [`Tombstoned::Unhashed`] when it is held. The caller
    /// has checked that the table is writable.
    pub(crate) fn tombstone_live_string(&mut self, key: &LuaString) -> Tombstoned {
        if key.cached_hash() == key::GC_FIELD_HASH && key.as_bytes() == b"__gc" {
            return if self.entries.get_string_live(key).is_some() {
                Tombstoned::Unhashed
            } else {
                Tombstoned::Absent
            };
        }
        match self.entries.tombstone_live_string(key) {
            Tombstoned::Unhashed if self.entries.get_string_live(key).is_none() => {
                Tombstoned::Absent
            }
            found => found,
        }
    }

    /// Forgets which metamethods this table was known to lack, once it is
    /// one a metatable lookup asked about. A table no lookup asked about
    /// (plain data) holds zero and is left alone, so a store to it writes
    /// nothing here.
    #[inline]
    pub(crate) fn forget_absent_metamethods(&mut self) {
        let known = self.absent_metamethods.get_mut();
        if *known != 0 {
            *known = 0;
        }
    }

    /// Whether [`Self::revive_string`] would store under `key`: a hashed
    /// table that holds the key as a deleted key and is writable.
    pub(crate) fn can_revive_string(&self, key: &LuaString) -> bool {
        !self.readonly
            && self.deleted_entries.is_empty()
            && !(key.cached_hash() == key::GC_FIELD_HASH && key.as_bytes() == b"__gc")
            && self.entries.holds_deleted_string(key)
    }

    /// Stores the non-nil `value` in the deleted key `key` of a hashed table,
    /// in one probe, and counts the entry live again (see
    /// [`TableEntries::revive_string`]). Hands `value` back for any other
    /// table or key.
    pub(crate) fn revive_string(&mut self, key: &LuaString, value: Value) -> Result<(), Value> {
        if self.readonly
            || !self.deleted_entries.is_empty()
            || (key.cached_hash() == key::GC_FIELD_HASH && key.as_bytes() == b"__gc")
        {
            return Err(value);
        }
        self.entries.revive_string(key, value)
    }

    /// Whether [`Self::tombstone_live_string`] would delete `key`.
    pub(crate) fn can_tombstone_present_string(&self, key: &LuaString) -> bool {
        !self.readonly
            && self.entries.is_indexed()
            && !(key.cached_hash() == key::GC_FIELD_HASH && key.as_bytes() == b"__gc")
            && self.entries.get_string_live(key).is_some()
    }

    /// The slot a plain write of a non-nil value to `key` stores into, when
    /// the table is shaped and its shape holds the key. Anything else takes
    /// the general path of [`Self::set_string`].
    #[inline]
    pub(crate) fn shaped_string_slot(&mut self, key: &LuaString) -> Option<&mut Value> {
        // The entries test first: it is the one that fails on most tables.
        let slot = self.entries.shaped_slot_mut(key)?;
        if self.readonly || !self.deleted_entries.is_empty() {
            return None;
        }
        Some(slot)
    }

    fn set_string_entry(&mut self, string: &LuaString, value: Value) -> RuntimeResult<usize> {
        // Overwriting an existing string-keyed field is the hottest table
        // write in field-heavy loops. Resolve it without building an owned
        // table key; a found key cannot also sit in the deleted-key set
        // because every insertion removes its key from that set first.
        //
        // Small record tables resolve a write, new key or not, in one
        // scan. A deleted-key record would need the key removed from
        // that set on insertion, so those take the general path.
        let value = if !matches!(value, Value::Nil) && self.deleted_entries.is_empty() {
            match self.entries.set_small_string(string, value) {
                Ok(true) => {
                    return Ok(std::mem::size_of::<TableKey>() + std::mem::size_of::<Value>());
                }
                Ok(false) => return Ok(0),
                Err(value) => value,
            }
        } else {
            value
        };
        let value = if !matches!(value, Value::Nil) && self.deleted_entries.is_empty() {
            match self.entries.revive_string(string, value) {
                Ok(()) => return Ok(0),
                Err(value) => value,
            }
        } else {
            value
        };
        if !matches!(value, Value::Nil)
            && let Some(slot) = self.entries.get_string_mut(string)
        {
            *slot = value;
            return Ok(0);
        }
        self.set_entry(TableKey::String(string.clone()), value)
    }

    /// Stores or deletes a non-array entry through the general path.
    fn set_entry(&mut self, key: TableKey, value: Value) -> RuntimeResult<usize> {
        if matches!(value, Value::Nil) {
            // A string key of a hashed table becomes a tombstone: no record
            // of the deletion, and no ordered-set work.
            if let TableKey::String(string) = &key
                && self.entries.tombstone_string(string).is_some()
            {
                return Ok(0);
            }
            let mut allocated = 0;
            if !self.entries.is_empty() && self.entries.remove(&key).is_some() {
                allocated = self.deleted_entries.insert_accounted(key);
            }
            Ok(allocated)
        } else {
            if !self.deleted_entries.is_empty() {
                self.deleted_entries.remove(&key);
            }
            Ok(self.insert_entry(key, value))
        }
    }

    pub(crate) fn set_array(&mut self, index: usize, value: Value) -> usize {
        if matches!(value, Value::Nil) {
            let mut allocated = 0;
            if index <= self.array.len() {
                if !matches!(self.array[index - 1], Value::Nil) {
                    // The last element, with a live one before it, is all
                    // the trim below takes off the end: a pending
                    // deletion, which costs no ordered-set insertion.
                    allocated = if index == self.array.len()
                        && (index == 1 || !matches!(self.array[index - 2], Value::Nil))
                    {
                        self.deleted_array_keys.record_tail(index);
                        0
                    } else {
                        self.deleted_array_keys
                            .insert_accounted(index, self.array.len())
                    };
                }
                self.array[index - 1] = Value::Nil;
                if index <= self.border {
                    self.border = index - 1;
                }
                self.trim_array_tail();
            } else if !self.entries.is_empty() {
                let Some(key) = array_table_key(index) else {
                    return allocated;
                };
                if self.entries.remove(&key).is_some() {
                    allocated = self.deleted_entries.insert_accounted(key);
                }
            }
            return allocated;
        }

        if index <= self.array.len() {
            if !self.deleted_array_keys.is_empty(self.array.len()) {
                self.deleted_array_keys.remove(index, self.array.len());
            }
            if !self.entries.is_empty() || !self.deleted_entries.is_empty() {
                let Some(key) = array_table_key(index) else {
                    return 0;
                };
                if !self.entries.is_empty() {
                    self.entries.remove(&key);
                }
                if !self.deleted_entries.is_empty() {
                    self.deleted_entries.remove(&key);
                }
            }
            self.array[index - 1] = value;
            if index == self.border.saturating_add(1) {
                return self.advance_border();
            }
            0
        } else if index == self.array.len().saturating_add(1) && self.border == self.array.len() {
            if !self.deleted_array_keys.is_empty(self.array.len()) {
                self.deleted_array_keys.forget_push(index);
            }
            if !self.entries.is_empty() || !self.deleted_entries.is_empty() {
                let Some(key) = array_table_key(index) else {
                    return 0;
                };
                if !self.entries.is_empty() {
                    self.entries.remove(&key);
                }
                if !self.deleted_entries.is_empty() {
                    self.deleted_entries.remove(&key);
                }
            }
            self.array.push(value);
            self.border = self.array.len();
            std::mem::size_of::<Value>() + self.absorb_sparse_tail()
        } else {
            let Some(key) = array_table_key(index) else {
                return 0;
            };
            // Dense numeric growth belongs in the array part. Sieve-style
            // tables receive most integral keys below a bound; without this,
            // every such write pays an ordered-map insertion. Small maps keep
            // the plain entry layout, mirroring Lua's sparse-key behavior.
            if (matches!(value, Value::Boolean(_))
                && (if self.array.is_empty() {
                    self.entries.len() >= 4 && index <= self.entries.len().saturating_mul(4)
                } else {
                    index <= self.array.len().saturating_add(BOOLEAN_ARRAY_GROWTH_REACH)
                }))
                || (self.entries.len() >= SPARSE_ARRAY_GROWTH_FLOOR
                    && index
                        <= self
                            .array
                            .len()
                            .saturating_add(self.entries.len())
                            .saturating_mul(4))
            {
                return self.grow_array_and_absorb(index, value);
            }
            if !self.deleted_array_keys.is_empty(self.array.len()) {
                self.deleted_array_keys.remove(index, self.array.len());
            }
            if !self.deleted_entries.is_empty() {
                self.deleted_entries.remove(&key);
            }
            self.insert_entry(key, value)
        }
    }

    /// Grows the array part to hold `index`, writes `value` there, and moves
    /// the integer-keyed entries the map held inside the grown range. The
    /// caller's density heuristic guarantees this stays close to full.
    pub(crate) fn grow_array_and_absorb(&mut self, index: usize, value: Value) -> usize {
        let previous_len = self.array.len();
        if !self.deleted_array_keys.is_empty(previous_len) {
            // The keys between the old end and the new one become holes.
            self.deleted_array_keys.settle(previous_len);
            self.deleted_array_keys.remove(index, previous_len);
        }
        if !self.deleted_entries.is_empty()
            && let Some(key) = array_table_key(index)
        {
            self.deleted_entries.remove(&key);
        }
        // Construct the holes in place: `resize` clones a `Value::Nil` from
        // the stack into every slot, and that copy stalled each growth of
        // sieve-style loops, which extend the array by one or two slots per
        // store.
        self.array.resize_with(index, || Value::Nil);
        if !self.entries.is_empty() {
            for (entry_index, stored) in self.entries.extract_integer_range(previous_len + 1, index)
            {
                self.array[entry_index - 1] = stored;
            }
        }
        self.array[index - 1] = value;
        let mut allocated = (index - previous_len) * std::mem::size_of::<Value>();
        allocated += self.advance_border();
        allocated
    }

    pub(crate) fn trim_array_tail(&mut self) {
        while matches!(self.array.last(), Some(Value::Nil)) {
            self.array.pop();
        }
        if self.border > self.array.len() {
            self.border = self.array.len();
        }
    }

    #[inline(never)]
    pub(crate) fn advance_border(&mut self) -> usize {
        while self.border < self.array.len() && !matches!(self.array[self.border], Value::Nil) {
            self.border += 1;
        }
        if self.border == self.array.len() {
            self.absorb_sparse_tail()
        } else {
            0
        }
    }

    pub(crate) fn absorb_sparse_tail(&mut self) -> usize {
        let original_len = self.array.len();
        while self.border == self.array.len() {
            if self.entries.is_empty() {
                break;
            }
            let Some(next) = self.array.len().checked_add(1) else {
                break;
            };
            let Some(key) = array_table_key(next) else {
                break;
            };
            let Some(value) = self.entries.remove(&key) else {
                break;
            };
            self.array.push(value);
            self.border = self.array.len();
        }
        (self.array.len() - original_len) * std::mem::size_of::<Value>()
    }

    pub(crate) fn array_contains(&self, value: &Value) -> bool {
        let Some(index) = array_index(value) else {
            return false;
        };
        self.array
            .get(index - 1)
            .is_some_and(|value| !matches!(value, Value::Nil))
    }

    pub(crate) fn deleted_array_contains(&self, value: &Value) -> bool {
        array_index(value)
            .is_some_and(|index| self.deleted_array_keys.contains(index, self.array.len()))
    }

    pub(crate) fn first_entry(&self) -> Option<(Value, Value)> {
        self.array_entry_after_index(0)
            .or_else(|| self.map_first_entry())
    }

    pub(crate) fn array_entry_after(&self, previous: &Value) -> Option<(Value, Value)> {
        let index = array_index(previous)?;
        self.array_entry_after_index(index)
    }

    pub(crate) fn array_entry_after_index(&self, index: usize) -> Option<(Value, Value)> {
        self.array
            .iter()
            .enumerate()
            .skip(index)
            .find(|(_, value)| !matches!(value, Value::Nil))
            .and_then(|(index, value)| {
                let index = index.checked_add(1)?;
                let index = u64::try_from(index).ok()?;
                Some((
                    Value::Number(unsigned_integer_to_number(index)),
                    value.clone(),
                ))
            })
    }

    /// [`Runtime::table_next`](oximond::runtime::Runtime::table_next)'s step, for the
    /// cases the traversal cursor and the array part answer, with the key in
    /// the representation of the profile (`tagged`: integers are integers).
    /// `None` leaves the step, and every error, to the general lookup.
    pub(crate) fn next_entry_fast(
        &mut self,
        previous: &Value,
        tagged: bool,
    ) -> Option<(Value, Value)> {
        let from = if matches!(previous, Value::Nil) {
            0
        } else if let Some(index) = array_index(previous) {
            if self
                .array
                .get(index - 1)
                .is_some_and(|value| !matches!(value, Value::Nil))
            {
                index
            } else if self.deleted_array_keys.is_empty(self.array.len()) {
                return self.hash_next_fast(Some(previous), tagged);
            } else {
                return None;
            }
        } else {
            return self.hash_next_fast(Some(previous), tagged);
        };
        // The array part from the slot `from` (zero-based) on.
        if let Some(tail) = self.array.get(from..)
            && let Some(offset) = tail.iter().position(|value| !matches!(value, Value::Nil))
        {
            let index = from + offset + 1;
            let key = if tagged {
                Value::Integer(i64::try_from(index).ok()?)
            } else {
                Value::Number(index as f64)
            };
            return Some((key, tail[offset].clone()));
        }
        self.hash_next_fast(None, tagged)
    }

    /// [`Self::next_entry_fast`] giving the entry as references
    /// into the table. The key of a hash entry is the table's own, so the
    /// caller makes the value (a string clones, a number copies) only if it
    /// needs it.
    ///
    /// The hash part's step has one call site for the three ways to reach
    /// it. `hint` is [`TableEntries::next_after_value`]'s.
    ///
    /// [`TableEntries::next_after_value`]: entries::TableEntries::next_after_value
    #[inline]
    pub(crate) fn next_ref(
        &mut self,
        previous: &Value,
        tagged: bool,
        hint: &mut usize,
    ) -> Option<NextRef<'_>> {
        // The key the hash part continues from: none after the array part.
        let hash_previous = 'array: {
            let from = if matches!(previous, Value::Nil) {
                0
            } else if let Some(index) = array_index(previous) {
                if self
                    .array
                    .get(index - 1)
                    .is_some_and(|value| !matches!(value, Value::Nil))
                {
                    index
                } else if self.deleted_array_keys.is_empty(self.array.len()) {
                    break 'array Some(previous);
                } else {
                    return None;
                }
            } else {
                break 'array Some(previous);
            };
            // The array part from the slot `from` (zero-based) on.
            let present = self
                .array
                .get(from..)
                .and_then(|tail| tail.iter().position(|value| !matches!(value, Value::Nil)))
                .map(|offset| from + offset);
            if let Some(slot) = present {
                return Some(NextRef::Array(slot + 1, &self.array[slot]));
            }
            None
        };
        self.hash_next_ref(hash_previous, tagged, hint)
    }

    #[inline]
    fn hash_next_ref(
        &mut self,
        previous: Option<&Value>,
        tagged: bool,
        hint: &mut usize,
    ) -> Option<NextRef<'_>> {
        match self.entries.next_after_value(previous, hint) {
            entries::HashNext::Entry(key, value) => Some(NextRef::Hash(
                match key {
                    TableKey::String(text) => NextKey::String(text),
                    key => NextKey::Plain(key_value(key, tagged)),
                },
                value,
            )),
            entries::HashNext::End => Some(NextRef::End),
            entries::HashNext::Slow => None,
        }
    }

    fn hash_next_fast(&mut self, previous: Option<&Value>, tagged: bool) -> Option<(Value, Value)> {
        let mut hint = usize::MAX;
        match self.entries.next_after_value(previous, &mut hint) {
            entries::HashNext::Entry(key, value) => {
                let key = match key {
                    TableKey::Integer(integer) if tagged => Value::Integer(*integer),
                    TableKey::Integer(integer) => Value::Number(key::integer_to_number(*integer)),
                    key => match key.value() {
                        Value::Number(number) if tagged => key::integer_key_from_number(number)
                            .map_or(Value::Number(number), Value::Integer),
                        key => key,
                    },
                };
                Some((key, value.clone()))
            }
            entries::HashNext::End => Some((Value::Nil, Value::Nil)),
            entries::HashNext::Slow => None,
        }
    }

    pub(crate) fn map_first_entry(&self) -> Option<(Value, Value)> {
        self.entries
            .first_after(None)
            .map(|(key, value)| (key.value(), value.clone()))
    }

    pub(crate) fn map_entry_after(&self, previous: &TableKey) -> Option<(Value, Value)> {
        self.entries
            .first_after(Some(previous))
            .map(|(key, value)| (key.value(), value.clone()))
    }
}

/// Clones a string, table, closure or function value, and `None` for any
/// other kind. A handle or function copies field by field.
#[inline(always)]
fn clone_reference(value: &Value) -> Option<Value> {
    Some(match value {
        Value::String(text) => Value::String(text.clone()),
        Value::Table(handle) => Value::Table(*handle),
        Value::Closure(handle) => Value::Closure(*handle),
        Value::NativeFunction(handle) => Value::NativeFunction(*handle),
        Value::BuiltinFunction(function) => Value::BuiltinFunction(*function),
        _ => return None,
    })
}

/// The zero-based array offset of a key, or a value of at least
/// `i64::MAX as u64` when the key is not an array key. A single unsigned
/// compare against the array length then checks both ends. The keys that
/// yield an offset below `i64::MAX as u64` are exactly those for which
/// [`array_index`] returns `Some`, at offset `index - 1`.
#[inline(always)]
pub(crate) fn array_offset(key: &Value) -> u64 {
    match key {
        Value::Integer(value) => (*value as u64).wrapping_sub(1),
        Value::Number(value) => array_offset_number(*value),
        _ => u64::MAX,
    }
}

/// [`array_offset`] for a number key: the raw truncation when it round-trips
/// exactly, else `u64::MAX`. The raw truncation yields `i64::MIN` for NaN and
/// for anything outside the `i64` range, which wraps to an offset past
/// `i64::MAX` and so is never an array key.
#[allow(clippy::float_cmp, clippy::cast_precision_loss, clippy::cast_sign_loss)]
#[inline(always)]
pub(crate) fn array_offset_number(value: f64) -> u64 {
    let integer = key::truncate_to_i64(value);
    if integer as f64 == value {
        (integer as u64).wrapping_sub(1)
    } else {
        u64::MAX
    }
}

/// A key of the hash part as the value `next` gives for it in the profile's
/// representation: integers and floats with an integer value are integers on
/// the profiles that have them, and floats on the others.
pub(crate) fn key_value(key: &TableKey, tagged: bool) -> Value {
    match key {
        TableKey::String(text) => Value::String(text.clone()),
        TableKey::Integer(integer) if tagged => Value::Integer(*integer),
        TableKey::Integer(integer) => Value::Number(key::integer_to_number(*integer)),
        key => match key.value() {
            Value::Number(number) if tagged => {
                key::integer_key_from_number(number).map_or(Value::Number(number), Value::Integer)
            }
            key => key,
        },
    }
}

/// The key of an entry of the hash part, as the table holds it.
pub enum NextKey<'a> {
    /// A string key: the table's own string.
    String(&'a LuaString),
    /// Any other key, in the profile's representation.
    Plain(Value),
}

/// The entry a traversal step found, as references into the table.
pub enum NextRef<'a> {
    /// A slot of the array part: its one-based index and value.
    Array(usize, &'a Value),
    /// An entry of the hash part: its key and value.
    Hash(NextKey<'a>, &'a Value),
    /// No entry follows.
    End,
}

#[inline(always)]
pub(crate) fn array_index(value: &Value) -> Option<usize> {
    if let Value::Integer(value) = value {
        return usize::try_from(*value).ok().filter(|index| *index > 0);
    }
    let Value::Number(value) = value else {
        return None;
    };
    array_index_number(*value)
}

#[cfg(test)]
mod array_offset_tests {
    use super::{array_index, array_offset};
    use crate::runtime::value::Value;

    #[test]
    fn offset_agrees_with_array_index() {
        let two_63 = 9_223_372_036_854_775_808.0_f64;
        let two_53 = 9_007_199_254_740_992.0_f64;
        let mut keys = vec![];
        for n in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            0.5,
            -0.5,
            1.0,
            1.5,
            2.0,
            -1.0,
            two_53,
            two_53.next_up(),
            two_53.next_down(),
            two_63,
            two_63.next_down(),
            two_63.next_up(),
            -two_63,
            (-two_63).next_up(),
            (-two_63).next_down(),
            f64::MAX,
            f64::MIN,
            f64::MIN_POSITIVE,
            0.5_f64.next_up(),
            1.0_f64.next_down(),
            1.0_f64.next_up(),
            two_63 * 2.0,
        ] {
            keys.push(Value::Number(n));
        }
        for i in [i64::MIN, i64::MIN + 1, -1, 0, 1, 2, i64::MAX - 1, i64::MAX] {
            keys.push(Value::Integer(i));
        }
        keys.push(Value::Nil);
        keys.push(Value::Boolean(true));
        for key in &keys {
            let offset = array_offset(key);
            let expected = array_index(key).map(|index| index as u64 - 1);
            let actual = (offset < i64::MAX as u64).then_some(offset);
            assert_eq!(actual, expected, "{key:?}");
        }
    }
}

#[cfg(test)]
mod array_index_tests {
    use super::array_index_number;
    use super::key::I64_EXCLUSIVE_UPPER_AS_F64;

    #[test]
    fn boundary_neighbours_of_the_index_range() {
        let two_53 = 9_007_199_254_740_992.0_f64;
        let two_63 = I64_EXCLUSIVE_UPPER_AS_F64;
        assert_eq!(array_index_number(two_53.next_down()), Some((1 << 53) - 1));
        assert_eq!(array_index_number(two_53.next_up()), Some((1 << 53) + 2));
        assert_eq!(array_index_number(0.5_f64.next_up()), None);
        assert_eq!(array_index_number(1.0_f64.next_down()), None);
        assert_eq!(array_index_number(1.0_f64.next_up()), None);
        assert_eq!(
            array_index_number(two_63.next_down()),
            Some(i64::MAX as usize - 1023)
        );
        assert_eq!(array_index_number(two_63.next_up()), None);
        assert_eq!(array_index_number(-two_63), None);
        assert_eq!(array_index_number((-two_63).next_down()), None);
    }

    #[test]
    fn only_exact_positive_integral_doubles_are_array_indexes() {
        assert_eq!(array_index_number(1.0), Some(1));
        assert_eq!(array_index_number(8000.0), Some(8000));
        assert_eq!(array_index_number(9_007_199_254_740_992.0), Some(1 << 53));
        for value in [
            0.0,
            -0.0,
            -1.0,
            0.5,
            1.5,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            I64_EXCLUSIVE_UPPER_AS_F64,
            I64_EXCLUSIVE_UPPER_AS_F64 * 2.0,
            f64::MAX,
            f64::MIN,
        ] {
            assert_eq!(array_index_number(value), None, "{value}");
        }
    }
}

#[cfg(test)]
mod dense_boolean_tests {
    use super::Table;
    use crate::runtime::value::Value;

    #[test]
    fn nearby_boolean_keys_move_into_array_and_reinsert_after_deletion() {
        let mut table = Table::default();
        for index in [4, 6, 8, 10] {
            table.set_array(index, Value::Boolean(true));
        }
        assert!(table.array.is_empty());
        table.set_array(12, Value::Boolean(true));
        assert_eq!(table.array.len(), 12);
        assert!(table.entries.is_empty());
        for index in [4, 6, 8, 10, 12] {
            assert_eq!(table.array[index - 1], Value::Boolean(true));
        }

        table.set_array(12, Value::Nil);
        assert!(table.deleted_array_keys.contains(12, table.array.len()));
        table.set_array(12, Value::Boolean(false));
        assert_eq!(table.array[11], Value::Boolean(false));
        assert!(!table.deleted_array_keys.contains(12, table.array.len()));
    }
}

/// Converts a Lua number to a one-based array index.
///
/// Array-backed keys also need an `i64` representation when a sparse entry
/// moves between the map and the array. Keeping the upper bound below 2^63
/// therefore prevents an integral double such as 2^63 from being accepted as
/// an array key and then silently dropped because it cannot form `TableKey`.
#[allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
#[inline(always)]
pub(crate) fn array_index_number(value: f64) -> Option<usize> {
    // The raw truncation yields `i64::MIN` for NaN, infinities, and anything
    // outside the `i64` range (2^63 included), so the range test on the
    // result excludes them. The round trip rejects fractions without a libm
    // `trunc` call.
    let integer = key::truncate_to_i64(value);
    if !(1..i64::MAX).contains(&integer) || integer as f64 != value {
        return None;
    }
    usize::try_from(integer).ok()
}

pub(crate) fn array_table_key(index: usize) -> Option<TableKey> {
    i64::try_from(index).ok().map(TableKey::Integer)
}

pub(crate) fn numbers_equal(left: f64, right: f64) -> bool {
    !left.is_nan() && !right.is_nan() && normalize_number_key(left) == normalize_number_key(right)
}

pub(crate) fn unsigned_integer_to_number(value: u64) -> f64 {
    value as f64
}

/// A concatenation operand's bytes: borrowed string contents or a number
/// formatted on the stack.
pub(crate) enum ConcatOperand<'a> {
    Bytes(&'a [u8]),
    Number(NumberText),
}

impl Deref for ConcatOperand<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::Bytes(bytes) => bytes,
            Self::Number(text) => text.as_bytes(),
        }
    }
}

pub(crate) fn concat_operand(
    value: &Value,
    format: NumberFormat,
) -> RuntimeResult<ConcatOperand<'_>> {
    match value {
        Value::String(value) => Ok(ConcatOperand::Bytes(value.as_bytes())),
        Value::Integer(value) => Ok(ConcatOperand::Number(format_integer(*value))),
        Value::Number(value) => Ok(ConcatOperand::Number(format.format(*value))),
        value => Err(RuntimeError::TypeError {
            operation: "concatenate",
            expected: "string or number",
            found: value.type_name(),
        }),
    }
}

pub(crate) fn concat_operand_length(value: &Value, format: NumberFormat) -> RuntimeResult<usize> {
    match value {
        Value::String(value) => Ok(value.len()),
        Value::Integer(_) | Value::Number(_) => Ok(concat_operand(value, format)?.len()),
        value => Err(RuntimeError::TypeError {
            operation: "concatenate",
            expected: "string or number",
            found: value.type_name(),
        }),
    }
}

#[cfg(test)]
mod pending_deletion_tests {
    use super::Table;
    use crate::runtime::native_views::NativeTableLayout;
    use crate::runtime::value::Value;
    use std::collections::BTreeSet;

    /// What the interpreter's records of deleted array keys hold for a list
    /// that changes only at its ends and by positional calls, as the sets
    /// before `DeletedArrayKeys` kept them (every pop inserted its key).
    #[derive(Default)]
    struct Model {
        list: Vec<i64>,
        deleted: BTreeSet<usize>,
    }

    impl Model {
        fn push(&mut self, value: i64) {
            self.list.push(value);
            self.deleted.remove(&self.list.len());
        }

        fn pop(&mut self) {
            let key = self.list.len();
            self.list.pop();
            self.deleted.insert(key);
        }

        fn overwrite(&mut self, index: usize, value: i64) {
            self.list[index - 1] = value;
            self.deleted.remove(&index);
        }

        fn insert(&mut self, position: usize, value: i64) {
            let length = self.list.len();
            for key in position..=length + 1 {
                self.deleted.remove(&key);
            }
            self.list.insert(position - 1, value);
        }

        fn remove(&mut self, position: usize) {
            let length = self.list.len();
            for key in position..length {
                self.deleted.remove(&key);
            }
            self.list.remove(position - 1);
            self.deleted.insert(length);
        }
    }

    struct Rng(u64);

    impl Rng {
        fn value(&mut self) -> i64 {
            i64::try_from(self.next(1000)).unwrap()
        }

        fn next(&mut self, bound: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            usize::try_from(self.0 >> 33).unwrap() % bound
        }
    }

    fn same(table: &Table, model: &Model, context: &str) {
        let length = table.array.len();
        assert_eq!(length, model.list.len(), "{context}");
        assert_eq!(table.border, length, "{context}");
        for key in 1..=model.list.len() + 40 {
            assert_eq!(
                table.deleted_array_keys.contains(key, length),
                model.deleted.contains(&key),
                "key {key} {context}"
            );
        }
        assert_eq!(
            table.deleted_array_keys.is_empty(length),
            model.deleted.is_empty(),
            "{context}"
        );
    }

    #[test]
    fn pops_and_pushes_keep_the_records_the_interpreter_would() {
        let mut rng = Rng(7);
        for round in 0..300 {
            let mut table = Table::default();
            let mut model = Model::default();
            for step in 0..120 {
                let context = format!("round {round} step {step}");
                let length = model.list.len();
                match rng.next(8) {
                    0..=2 => {
                        let value = rng.value();
                        table.set_array(length + 1, Value::Integer(value));
                        model.push(value);
                    }
                    3 | 4 if length > 0 => {
                        table.set_array(length, Value::Nil);
                        model.pop();
                    }
                    5 if length > 0 => {
                        let index = rng.next(length) + 1;
                        let value = rng.value();
                        table.set_array(index, Value::Integer(value));
                        model.overwrite(index, value);
                    }
                    6 if length < 30 => {
                        let position = rng.next(length + 1) + 1;
                        let value = rng.value();
                        table
                            .trace_insert(Some(position), &Value::Integer(value))
                            .unwrap();
                        model.insert(position, value);
                    }
                    7 if length > 0 => {
                        let position = rng.next(length) + 1;
                        table.trace_remove(Some(position), |_| true).unwrap();
                        model.remove(position);
                    }
                    _ => {}
                }
                same(&table, &model, &context);
            }
        }
    }

    #[test]
    #[allow(unsafe_code)]
    fn a_pop_done_in_place_is_a_record_all_the_same() {
        let layout = NativeTableLayout::get().expect("the vector layout is known");
        let mut table = Table::default();
        for index in 1..=10 {
            table.set_array(index, Value::Integer(i64::try_from(index).unwrap()));
        }
        let address = std::ptr::from_mut(&mut table) as usize;
        let word = |at: i32| (address + usize::try_from(at).unwrap()) as *mut usize;
        // SAFETY: the offsets are the layout's own, over a live table that
        // nothing else touches.
        unsafe {
            assert_eq!(*word(layout.array_length), 10);
            assert!(*word(layout.array_capacity) >= 10);
            assert_eq!(*word(layout.array_pointer), table.array.as_ptr() as usize);
            assert_eq!(*word(layout.border), 10);
            // Pop the last three the way native code does.
            for length in [10, 9, 8] {
                *word(layout.array_length) = length - 1;
                *word(layout.border) = length - 1;
                let pending = word(layout.pending_deletions);
                *pending = (*pending).max(length);
            }
        }
        assert_eq!(table.array.len(), 7);
        for key in 8..=10 {
            assert!(table.deleted_array_keys.contains(key, 7), "{key}");
        }
        assert!(!table.deleted_array_keys.contains(7, 7));
        assert!(!table.deleted_array_keys.contains(11, 7));
        // The interpreter goes on from there: a push over a record, a hole.
        table.set_array(8, Value::Integer(80));
        assert!(!table.deleted_array_keys.contains(8, 8));
        assert!(table.deleted_array_keys.contains(9, 8));
        assert!(table.deleted_array_keys.contains(10, 8));
        table.set_array(12, Value::Boolean(true));
        for key in [9, 10] {
            assert!(table.deleted_array_keys.contains(key, table.array.len()));
        }
    }
}
