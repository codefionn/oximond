//! Raw table views (array parts, slots) borrowed by native trace code.

use crate::runtime::value::Value;

/// `Value` is `repr(C, u8)`: the variant tag is the first byte, in declaration
/// order, and every payload starts at [`NATIVE_VALUE_PAYLOAD_OFFSET`]. Native
/// code decodes array slots with these constants.
pub const NATIVE_VALUE_TAG_NIL: u8 = 0;
pub const NATIVE_VALUE_TAG_BOOLEAN: u8 = 1;
pub const NATIVE_VALUE_TAG_INTEGER: u8 = 2;
pub const NATIVE_VALUE_TAG_NUMBER: u8 = 3;
pub const NATIVE_VALUE_TAG_STRING: u8 = 4;
pub const NATIVE_VALUE_TAG_TABLE: u8 = 5;
pub const NATIVE_VALUE_TAG_THREAD: u8 = 9;
pub const NATIVE_VALUE_TAG_CLOSURE: u8 = 11;
pub const NATIVE_VALUE_PAYLOAD_OFFSET: i32 = 8;
pub const NATIVE_VALUE_SIZE: usize = 16;
const _: () = assert!(std::mem::size_of::<Value>() == NATIVE_VALUE_SIZE);

/// A table's traversal snapshot, as native code reads it for the steps of a
/// `next` loop (see [`oximond::runtime::Runtime::table_native_cursor`]).
///
/// The slots are the hash part's entries in traversal order, each a
/// [`NATIVE_CURSOR_SLOT_SIZE`]-byte record: the key's kind and integer value,
/// the address of the entry's value (a nil value is a deleted key the
/// traversal skips) and the address of its key. `position` is the index of
/// the entry the last step returned. Everything stays valid until the
/// table's layout changes, and a trace that has an operation which might
/// change a layout does not read it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeCursor {
    /// One when the control value of the loop is the key at `*position`; the
    /// loop's steps keep it so, and anything that writes the control value
    /// another way clears it.
    pub in_sync: u64,
    /// The address of the first slot.
    pub slots: usize,
    /// The number of slots.
    pub len: u64,
    /// The address of the position, a `u64`. The run that takes the view
    /// points it at `position_value`, so that no other step along the table
    /// (a `next` call in the loop body) moves the position native code owns.
    pub position: usize,
    /// The position as it was when the view was taken, and the cell
    /// `position` names once the run owns the view.
    pub position_value: u64,
    /// The address of the slot of the entry the last native step moved to,
    /// whose key the control lane does not hold yet; zero for none. The run
    /// commits it to the lane before anything else reads the lane.
    pub pending: usize,
}

/// One entry of a [`NativeCursor`]'s snapshot, as native code reads it.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NativeCursorSlot {
    /// [`NATIVE_VALUE_TAG_INTEGER`] for an integer key, [`NATIVE_VALUE_TAG_STRING`]
    /// for a string key, and zero for any other.
    pub key_tag: u64,
    /// The integer key, else zero.
    pub key_bits: u64,
    /// The address of the entry's value.
    pub value: *const Value,
    /// The address of the entry's key, for the runtime to read.
    pub(crate) key: usize,
}

#[allow(unsafe_code)]
// SAFETY: the pointers are only read through the runtime's exclusive access
// while the layout they were taken from is current (see `NativeCursor`), so
// they behave like references into the storage that owns the entries.
unsafe impl Send for NativeCursorSlot {}
#[allow(unsafe_code)]
// SAFETY: as above; nothing reads them through a shared reference.
unsafe impl Sync for NativeCursorSlot {}

/// The snapshot a hash part too small to keep one gets for a native loop, and
/// the position cell its [`NativeCursor`] names.
#[derive(Debug, Default)]
pub struct NativeCursorScratch {
    pub(crate) slots: Vec<NativeCursorSlot>,
    pub(crate) position: u64,
}

/// The size of a slot of a [`NativeCursor`], and the offsets of its fields.
pub const NATIVE_CURSOR_SLOT_SIZE: usize = 32;
// Only JIT code reads the layout, and it compiles only for 64-bit targets.
#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(std::mem::size_of::<NativeCursorSlot>() == NATIVE_CURSOR_SLOT_SIZE);
    assert!(std::mem::offset_of!(NativeCursorSlot, key_tag) == NATIVE_CURSOR_SLOT_KEY_TAG);
    assert!(std::mem::offset_of!(NativeCursorSlot, key_bits) == NATIVE_CURSOR_SLOT_KEY_BITS);
    assert!(std::mem::offset_of!(NativeCursorSlot, value) == NATIVE_CURSOR_SLOT_VALUE);
};
/// The kind of the key: [`NATIVE_VALUE_TAG_INTEGER`], [`NATIVE_VALUE_TAG_STRING`]
/// or zero.
pub const NATIVE_CURSOR_SLOT_KEY_TAG: usize = 0;
/// The integer key.
pub const NATIVE_CURSOR_SLOT_KEY_BITS: usize = 8;
/// The address of the entry's value.
pub const NATIVE_CURSOR_SLOT_VALUE: usize = 16;

/// One slot of the string index native code probes for a dynamic string key
/// (see [`oximond::runtime::Runtime::table_native_probe`]).
///
/// An empty slot has `key == 0`. A full one names the entry's key string by
/// its storage address (what [`oximond::string::LuaString::identity`] returns),
/// with the string's cached hash beside it so a probe skips a candidate
/// without touching the key, and the address of the entry's value.
///
/// `owner` is the word a string value that holds the entry's key string has
/// in its payload (the storage address less
/// [`NATIVE_STRING_STORAGE_OFFSET`]), zero in an empty slot. A probe
/// compiled in line compares it with the key's payload as loaded, so a key
/// that is the table's own string costs no address arithmetic. It also
/// makes the slot 32 bytes, so a slot's offset is its index shifted and
/// native code masks the start with a mask shifted the same way.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NativeProbeSlot {
    pub hash: u64,
    pub key: usize,
    pub value: *const Value,
    pub owner: usize,
}

#[allow(unsafe_code)]
// SAFETY: the pointers are only read by native code that holds the table's
// layout fixed for its run, and by the runtime that built them; they behave
// like references into the storage that owns the entries.
unsafe impl Send for NativeProbeSlot {}
#[allow(unsafe_code)]
// SAFETY: as above.
unsafe impl Sync for NativeProbeSlot {}

impl NativeProbeSlot {
    pub(crate) const EMPTY: Self = Self {
        hash: 0,
        key: 0,
        value: std::ptr::null(),
        owner: 0,
    };

    /// The slot of the entry whose key string is `key`, with `hash` its
    /// cached hash and `value` the address of the entry's value.
    pub(crate) fn of(hash: u64, key: usize, value: *const Value) -> Self {
        Self {
            hash,
            key,
            value,
            owner: key - NATIVE_STRING_STORAGE_OFFSET,
        }
    }
}

/// The size of a [`NativeProbeSlot`] and the offsets of its fields.
pub const NATIVE_PROBE_SLOT_SIZE: usize = 1 << NATIVE_PROBE_SLOT_SHIFT;
/// The size as a shift: the offset of the slot at index `at` is
/// `at << NATIVE_PROBE_SLOT_SHIFT`.
pub const NATIVE_PROBE_SLOT_SHIFT: u32 = 5;
pub const NATIVE_PROBE_SLOT_HASH: usize = 0;
pub const NATIVE_PROBE_SLOT_KEY: usize = 8;
pub const NATIVE_PROBE_SLOT_VALUE: usize = 16;
pub const NATIVE_PROBE_SLOT_OWNER: usize = 24;
/// The most slots an index has: the offset of every slot fits the 32 bits
/// native code reads the start from (see [`NATIVE_PROBE_SHIFT`]).
pub const NATIVE_PROBE_MAX_SLOTS: usize = 1 << (32 - NATIVE_PROBE_SLOT_SHIFT);
#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(std::mem::size_of::<NativeProbeSlot>() == NATIVE_PROBE_SLOT_SIZE);
    assert!(std::mem::offset_of!(NativeProbeSlot, owner) == NATIVE_PROBE_SLOT_OWNER);
    assert!(std::mem::offset_of!(NativeProbeSlot, hash) == NATIVE_PROBE_SLOT_HASH);
    assert!(std::mem::offset_of!(NativeProbeSlot, key) == NATIVE_PROBE_SLOT_KEY);
    assert!(std::mem::offset_of!(NativeProbeSlot, value) == NATIVE_PROBE_SLOT_VALUE);
};

/// The probe starts at slot `native_probe_start(hash) & mask` and goes on to
/// the next slot (wrapping) until it meets the key or an empty slot.
///
/// The start is the top bits of the string's cached hash, whose last step
/// (`finish_hash` in `string/mod.rs`) spreads every bit of the hash into
/// them. Native code reads the high half of the hash with one 32-bit load at
/// [`NATIVE_STRING_PROBE_START_OFFSET`] of the string's storage and masks it
/// with the mask shifted by [`NATIVE_PROBE_SLOT_SHIFT`], which gives the
/// offset of the slot `native_probe_start(hash) & mask` with no multiply.
/// That is why the start leaves out the low bits of the half, and why an
/// index has at most [`NATIVE_PROBE_MAX_SLOTS`] slots. Without the last
/// step the high bits of an FNV fold are weak, and keys that differ in a
/// number at their end landed in runs: in the index of `player_name_1` to
/// `player_name_20000` the average key sat 15 buckets past its first. The test
/// `numbered_keys_do_not_pile_up_in_the_index` holds that under one.
pub const NATIVE_PROBE_SHIFT: u32 = 32 + NATIVE_PROBE_SLOT_SHIFT;

/// Where a probe for a string with `hash` starts, before the mask. Native
/// code computes the same.
#[must_use]
#[inline]
pub const fn native_probe_start(hash: u64) -> u64 {
    hash >> NATIVE_PROBE_SHIFT
}

/// Where native code reads the high half of a string's cached hash as a
/// zero-extended 32-bit word, from the string's storage: its bits from
/// [`NATIVE_PROBE_SLOT_SHIFT`] up are [`native_probe_start`] of the hash.
pub const NATIVE_STRING_PROBE_START_OFFSET: usize =
    crate::string::NATIVE_STRING_HASH_OFFSET + if cfg!(target_endian = "little") { 4 } else { 0 };

/// Where the storage of a string sits from the address a [`Value::String`]
/// holds in its payload: the payload is the address of the shared allocation's
/// header (two reference counts), and the storage follows. The runtime's
/// layout tests check it against [`oximond::string::LuaString::identity`].
pub const NATIVE_STRING_STORAGE_OFFSET: usize = 2 * std::mem::size_of::<usize>();

/// A string index over the string keys of one table, as native code probes
/// it: `slots` is the address of a power-of-two array of [`NativeProbeSlot`]
/// and `mask` is its length minus one. Valid while the table's key set and
/// layout stay as they were when the runtime took it.
///
/// `writable` says whether native code may overwrite a value the index
/// reaches, a number with a number or a boolean with a boolean, which needs no
/// barrier and no notice: the table is not read-only, not one that cached
/// inherited reads depend on, and not the one a major collection is tracing in
/// chunks. Every other write goes through the runtime.
///
/// `tombstones` says that native code may also delete a number or boolean in
/// place (leaving a nil the traversals skip) and revive such a nil with a
/// number or boolean, counting the table's live entries as the runtime does
/// ([`NATIVE_PROBE_SLOT_HASH`] of the slot after the index, which holds the
/// address of that counter, a `usize`). It holds for a hashed table that is
/// `writable` and records no deleted keys of the other kind. Neither store
/// adds an entry or moves one, so the index stays valid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeStringProbe {
    pub(crate) slots: usize,
    pub(crate) mask: u64,
    pub(crate) writable: bool,
    pub(crate) tombstones: bool,
}

impl NativeStringProbe {
    /// The address of the first slot.
    #[must_use]
    pub const fn slots(self) -> usize {
        self.slots
    }

    /// The number of slots minus one.
    #[must_use]
    pub const fn mask(self) -> u64 {
        self.mask
    }

    /// Whether native code may overwrite a scalar the index reaches with a
    /// scalar of its type.
    #[must_use]
    pub const fn writable(self) -> bool {
        self.writable
    }

    /// Whether native code may delete and revive scalars through the index
    /// (see the type's documentation).
    #[must_use]
    pub const fn tombstones(self) -> bool {
        self.tombstones
    }
}

/// Room for the string indexes of the small tables (an ordered or a shaped
/// representation) a run binds, reused between runs. A hashed table keeps its
/// own index (see [`oximond::runtime::Runtime::table_native_probe`]).
#[derive(Debug, Default)]
pub struct NativeStringProbeScratch {
    lists: Vec<Vec<NativeProbeSlot>>,
    used: usize,
}

impl NativeStringProbeScratch {
    /// Frees every list for the next run.
    pub fn begin(&mut self) {
        self.used = 0;
    }

    /// An empty list that stays where it is until the next [`Self::begin`].
    pub(crate) fn next(&mut self) -> &mut Vec<NativeProbeSlot> {
        if self.used == self.lists.len() {
            self.lists.push(Vec::new());
        }
        self.used += 1;
        &mut self.lists[self.used - 1]
    }
}

/// The array part of a table, borrowed by native code.
///
/// The address and length stay valid only while the caller has exclusive
/// access to the runtime and performs no other mutation of this table's
/// storage. Native code may read any slot and may overwrite the payload of a
/// `Number` slot with a number or of a `Boolean` slot with a boolean. Such a
/// store keeps the slot's type, the array border, and GC reachability intact.
/// Every other write must go through the runtime.
#[derive(Clone, Copy, Debug)]
pub struct NativeValueArray {
    pub(crate) values: *mut Value,
    pub(crate) len: usize,
}

impl NativeValueArray {
    /// Address of the first slot (Lua index 1).
    #[must_use]
    pub const fn as_ptr(self) -> *mut Value {
        self.values
    }

    /// Number of slots in the array part.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}

/// The bytes of a string, borrowed by native code.
///
/// The address stays valid while the string is alive, and its bytes never
/// change. The caller keeps the string alive for the whole run (the trace's
/// root bank holds it), so native code may read any byte below the length.
#[derive(Clone, Copy, Debug)]
pub struct NativeBytes {
    pointer: *const u8,
    len: usize,
}

impl NativeBytes {
    /// The bytes of `text`.
    #[must_use]
    pub fn of(text: &crate::string::LuaString) -> Self {
        let bytes = text.as_bytes();
        Self {
            pointer: bytes.as_ptr(),
            len: bytes.len(),
        }
    }

    /// Address of byte 0 (Lua index 1).
    #[must_use]
    pub const fn as_ptr(self) -> *const u8 {
        self.pointer
    }

    /// Number of bytes.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}

/// One table slot (a whole [`Value`]) borrowed by a native trace for the
/// length of one run.
///
/// The pointer stays valid only while the runtime performs no mutation of
/// the table's storage. A trace may overwrite the payload of a slot holding
/// a number, integer or boolean with a value of the same type, which keeps
/// the slot's tag, the table's border and GC reachability intact. The slot
/// holds a present, non-nil value when it is bound.
#[derive(Clone, Copy, Debug)]
pub struct NativeValueSlot {
    pub(crate) pointer: *mut Value,
}

impl NativeValueSlot {
    /// Address of the slot's [`Value`].
    #[must_use]
    pub const fn as_ptr(self) -> *mut Value {
        self.pointer
    }
}

/// What a row holds for a key of kind [`NativeSlotKind::Absent`] that the table
/// lacks: native code only tests the address against zero.
pub static NATIVE_ABSENT_FIELD: [u64; 2] = [0; 2];

/// Where a shaped table keeps its slots and what else its row depends on (see
/// [`Runtime::table_native_row_token`](crate::runtime::Runtime::table_native_row_token)).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeRowToken {
    /// The id of the table's shape.
    pub shape: usize,
    /// The address of the table's first value slot.
    pub slots: usize,
    /// The table's metatable.
    pub metatable: Option<crate::runtime::TableHandle>,
    /// Whether the table is watched by cached inherited reads.
    pub watched: bool,
    /// The table's address, which a row's address word holds (see
    /// [`NativeRowWords::address`]). Tables of one shape have different
    /// addresses, so a row kept for the shape never answers for it.
    pub address: usize,
}

/// The words of a row that name no slot: the positions among a trace's
/// fields of the words the runtime writes itself (see
/// [`Runtime::table_native_row`](crate::runtime::Runtime::table_native_row)).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NativeRowWords {
    /// The metatable word ([`crate::jit_ir::trace::META_FIELD_KEY`]).
    pub meta: Option<usize>,
    /// The table's address ([`crate::jit_ir::trace::TABLE_ADDRESS_KEY`]),
    /// which `NativeTableLayout` offsets read the array part's words from.
    pub address: Option<usize>,
}

impl NativeRowWords {
    /// Whether `field` is one of these words.
    #[must_use]
    pub fn names(self, field: usize) -> bool {
        self.meta == Some(field) || self.address == Some(field)
    }
}

/// The type of value a native trace requires in a slot it binds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeSlotKind {
    Boolean,
    Integer,
    Number,
    Table,
    /// No value at all: a key the table lacks. No slot holds it, so the
    /// runtime finds no address; the caller decides what a row says for it.
    Absent,
}

impl NativeSlotKind {
    /// The tag of the [`Value`] that has this kind, as native code reads it
    /// (nil for [`Self::Absent`]).
    #[must_use]
    pub const fn tag(self) -> u8 {
        match self {
            Self::Boolean => NATIVE_VALUE_TAG_BOOLEAN,
            Self::Integer => NATIVE_VALUE_TAG_INTEGER,
            Self::Number => NATIVE_VALUE_TAG_NUMBER,
            Self::Table => NATIVE_VALUE_TAG_TABLE,
            Self::Absent => NATIVE_VALUE_TAG_NIL,
        }
    }

    /// Whether `value` has this kind.
    #[must_use]
    pub const fn holds(self, value: &Value) -> bool {
        matches!(
            (self, value),
            (Self::Boolean, Value::Boolean(_))
                | (Self::Integer, Value::Integer(_))
                | (Self::Number, Value::Number(_))
                | (Self::Table, Value::Table(_))
        )
    }
}

/// Where native code finds the parts of a table it appends to or pops from
/// in place, as byte offsets from the table's address
/// ([`NativeTableHeader::address`]).
///
/// The array part is a `Vec<Value>` whose fields the standard library does
/// not lay out for us, so the offsets of its address, length and capacity are
/// found once by looking at a vector whose three words are known.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeTableLayout {
    /// The address of slot 1 of the array part.
    pub array_pointer: i32,
    /// The number of slots in the array part.
    pub array_length: i32,
    /// The slots the array part has room for.
    pub array_capacity: i32,
    /// The border (`#t` of a table with no hash part).
    pub border: i32,
    /// The highest key of the range of deleted array keys that native pops
    /// left unrecorded (see `DeletedArrayKeys`).
    pub pending_deletions: i32,
}

impl NativeTableLayout {
    /// The layout of this build, or `None` if the vector's layout is not the
    /// three distinct words this expects.
    #[must_use]
    #[allow(unsafe_code)]
    pub fn get() -> Option<Self> {
        static LAYOUT: std::sync::OnceLock<Option<NativeTableLayout>> = std::sync::OnceLock::new();
        *LAYOUT.get_or_init(|| {
            use crate::runtime::table_storage::{DeletedArrayKeys, Table};
            use std::mem::{offset_of, size_of};
            const _: () = assert!(size_of::<Vec<Value>>() == 3 * size_of::<usize>());
            let mut vector: Vec<Value> = Vec::with_capacity(5);
            vector.push(Value::Nil);
            vector.push(Value::Nil);
            if vector.capacity() != 5 {
                return None;
            }
            let pointer = vector.as_ptr() as usize;
            // SAFETY: a vector is three words (asserted above); this copies
            // them without using the copy as a vector.
            let words: [usize; 3] = unsafe { std::mem::transmute_copy(&vector) };
            let find = |wanted: usize| {
                let mut at = words
                    .iter()
                    .enumerate()
                    .filter(|(_, word)| **word == wanted);
                let (index, _) = at.next()?;
                at.next().is_none().then_some(index * size_of::<usize>())
            };
            let vector_at = offset_of!(Table, array);
            let offset = |inner: usize| i32::try_from(vector_at + inner).ok();
            Some(Self {
                array_pointer: offset(find(pointer)?)?,
                array_length: offset(find(2)?)?,
                array_capacity: offset(find(5)?)?,
                border: i32::try_from(offset_of!(Table, border)).ok()?,
                pending_deletions: i32::try_from(
                    offset_of!(Table, deleted_array_keys) + offset_of!(DeletedArrayKeys, pending),
                )
                .ok()?,
            })
        })
    }
}

/// A table as native code addresses it for the length of one run: its
/// address, and whether an append or a pop of the last element may be done in
/// place (see `Runtime::table_native_header`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeTableHeader {
    /// The address of the table, which stays put while the run creates no
    /// table (a run never does: tables it makes are made when it leaves).
    pub address: usize,
    /// Whether the table is a dense list that native code may append to and
    /// pop from: the array part holds every integer key up to the border,
    /// the hash part and the records of deleted keys are empty, nothing
    /// watches it, it is writable, and its metatable has neither
    /// `__newindex` nor `__len`.
    pub appendable: bool,
    /// Whether the table is dense enough for native code to read nil past
    /// its array part and to fill a hole inside it: the hash part and the
    /// records of deleted keys are empty, nothing watches it, it is writable,
    /// and its metatable has neither `__newindex` nor `__len`. Unlike
    /// [`Self::appendable`] the border may lie below the end of the array
    /// part. Every appendable table is solid.
    pub solid: bool,
}

/// How far past the end of a non-empty array part a boolean store still
/// grows the array part, with nil holes for the slots between, instead of
/// going to the hash part. Native code that grows the array part in place
/// follows the runtime's rule.
pub const NATIVE_BOOLEAN_GROWTH_REACH: usize =
    crate::runtime::table_storage::BOOLEAN_ARRAY_GROWTH_REACH;

impl NativeTableHeader {
    /// The [`Self::word`] of an appendable table.
    pub const DENSE: i64 = 1;
    /// The [`Self::word`] of a solid table that is not appendable.
    pub const HOLEY: i64 = 2;

    /// The flags native code reads from a header binding: [`Self::DENSE`] for
    /// an appendable table, [`Self::HOLEY`] for a solid one whose border is
    /// below the end of its array part, and zero for any other table. A
    /// nonzero word means the table is solid.
    #[must_use]
    pub const fn word(&self) -> i64 {
        if self.appendable {
            Self::DENSE
        } else if self.solid {
            Self::HOLEY
        } else {
            0
        }
    }
}

/// Makes room for one more slot in the array part of the table at `address`
/// (a [`NativeTableHeader::address`]), which native code calls when its
/// array part is full.
///
/// # Safety
///
/// `address` is the address of a live table, and nothing else holds a
/// reference into the table's array part for the duration of the call.
#[allow(unsafe_code)]
pub unsafe extern "C" fn native_table_reserve(address: usize) {
    // SAFETY: by the contract above.
    let table = unsafe { &mut *(address as *mut crate::runtime::table_storage::Table) };
    table.grow_array();
}
