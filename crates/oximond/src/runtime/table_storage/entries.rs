//! Non-array table entry storage: small sorted vectors promoted to hashed maps.

use crate::runtime::error::{RuntimeError, RuntimeResult};
use crate::runtime::native_views::{
    NATIVE_VALUE_TAG_NIL, NativeCursorScratch, NativeCursorSlot, NativeProbeSlot, NativeSlotKind,
    NativeStringProbe, NativeStringProbeScratch, native_probe_start,
};
use crate::runtime::table_storage::key::{
    StringProbe, StringSlotProbe, StringTableKey, TableKey, TableKeyBuildHasher,
};
use crate::runtime::table_storage::ordered_vec::OrderedVec;
use crate::runtime::table_storage::shape::{MAX_SHAPE_KEYS, TableShape};
use crate::runtime::value::Value;
use crate::string::LuaString;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Both representations iterate in key order, so this only trades insertion
/// cost against lookup cost. Ordered lookups binary-search with byte-wise
/// string comparisons (strings are not interned), which made field and
/// global reads on module-sized tables such as `math` or `_G` dominate
/// call-heavy code; hashed lookups compare one cached hash first.
pub(crate) const HASHED_TABLE_ENTRY_THRESHOLD: usize = 16;

/// Entry maps at least this large start absorbing dense integral writes into
/// the array part, so big numeric tables stop routing every write through
/// the ordered map.
pub(crate) const SPARSE_ARRAY_GROWTH_FLOOR: usize = 32;

pub(crate) enum TableEntries {
    Ordered(SmallOrderedEntries),
    /// Everything but a small ordered table, behind one thin pointer. That
    /// keeps the enum the size of the ordered vector and the test the inlined
    /// table reads make a single comparison; the hashed and the shaped
    /// representation are told apart through the pointer, out of line.
    Other(OtherEntries),
}

/// Header of a shaped table's allocation. The value slots, one per key of
/// the shape, follow it in the same allocation at [`SHAPED_SLOTS_OFFSET`], so
/// a record table costs its [`Table`](super::Table) and this one block.
#[repr(C)]
struct ShapedAlloc {
    shape: Arc<TableShape>,
}

const SHAPED_SLOTS_OFFSET: usize =
    std::mem::size_of::<ShapedAlloc>().next_multiple_of(std::mem::align_of::<Value>());

/// Recycles the blocks of dead shaped tables.
///
/// A record table is born and dies in the same few sizes (one per key count
/// up to [`MAX_SHAPE_KEYS`]), and a general allocator spends more on each
/// block than the constructor spends filling it. Each thread keeps one free
/// list per size. A list that runs dry first carves [`BATCH`] blocks from the
/// process-wide region of its size, which holds [`REGION_BLOCKS`] blocks and
/// is never freed; once the region is spent, blocks are allocated one at a
/// time. A list keeps at most [`POOL_CAP`] of those, and a block over the cap
/// goes straight back to the allocator. Region blocks are always listed, so
/// the pool never holds more than a few megabytes per size. A thread that
/// exits frees its listed blocks that are not in a region. A block freed on
/// another thread joins that thread's list. A free block's first word links
/// it to the next.
#[allow(unsafe_code)]
mod block_pool {
    use super::{MAX_SHAPE_KEYS, OtherEntries};
    use std::cell::Cell;
    use std::ptr::NonNull;
    use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

    /// Most allocator blocks one list keeps.
    pub(crate) const POOL_CAP: usize = 16384;
    /// Blocks in the region of one size.
    pub(crate) const REGION_BLOCKS: usize = 16384;
    /// Blocks a list carves from its region at once.
    const BATCH: usize = 32;

    /// The blocks of one size that threads carve in batches. The memory is
    /// allocated when the size is first used and untouched pages cost
    /// nothing until a block on them is.
    struct Region {
        base: AtomicPtr<u8>,
        /// Blocks handed out so far; may run past [`REGION_BLOCKS`].
        next: AtomicUsize,
    }

    static REGIONS: [Region; MAX_SHAPE_KEYS + 1] = [const {
        Region {
            base: AtomicPtr::new(std::ptr::null_mut()),
            next: AtomicUsize::new(0),
        }
    }; MAX_SHAPE_KEYS + 1];

    /// Distance between the blocks of `len` keys in a region.
    fn stride(len: usize) -> usize {
        let layout = OtherEntries::shaped_layout(len);
        layout.size().next_multiple_of(layout.align())
    }

    fn region_layout(len: usize) -> std::alloc::Layout {
        let align = OtherEntries::shaped_layout(len).align();
        std::alloc::Layout::from_size_align(stride(len) * REGION_BLOCKS, align)
            .expect("region layout is small")
    }

    /// Whether `block` lies in the region of `len` keys, whose blocks are
    /// never returned to the allocator.
    #[inline]
    fn in_region(len: usize, block: *mut u8) -> bool {
        let base = REGIONS[len].base.load(Ordering::Relaxed);
        !base.is_null() && (block.addr().wrapping_sub(base.addr())) < stride(len) * REGION_BLOCKS
    }

    struct Pool {
        heads: [Cell<*mut u8>; MAX_SHAPE_KEYS + 1],
        counts: [Cell<usize>; MAX_SHAPE_KEYS + 1],
    }

    /// Frees the listed blocks when its thread exits. The lists themselves
    /// have no destructor, so reaching them costs no liveness check; the
    /// guard is created when a list first gets a block.
    struct Guard;

    impl Drop for Guard {
        fn drop(&mut self) {
            let _ = POOL.try_with(|pool| {
                for len in 0..=MAX_SHAPE_KEYS {
                    let layout = OtherEntries::shaped_layout(len);
                    let mut block = pool.heads[len].replace(std::ptr::null_mut());
                    pool.counts[len].set(0);
                    while !block.is_null() {
                        // SAFETY: a listed block came from the allocator
                        // with this layout or from the region, and its
                        // first word holds the next link.
                        unsafe {
                            let next = block.cast::<*mut u8>().read();
                            if !in_region(len, block) {
                                std::alloc::dealloc(block, layout);
                            }
                            block = next;
                        }
                    }
                }
            });
        }
    }

    thread_local! {
        static POOL: Pool = const {
            Pool {
                heads: [const { Cell::new(std::ptr::null_mut()) }; MAX_SHAPE_KEYS + 1],
                counts: [const { Cell::new(0) }; MAX_SHAPE_KEYS + 1],
            }
        };
        static GUARD: Guard = const { Guard };
    }

    /// Makes the thread free its lists when it exits.
    #[cold]
    #[inline(never)]
    fn arm_guard() {
        // `Guard` has a destructor, so touching it registers that.
        let _ = GUARD.try_with(|_| ());
    }

    /// A listed block for a shape of `len` keys, if the list has one.
    #[inline]
    pub(super) fn take(len: usize) -> Option<NonNull<u8>> {
        POOL.try_with(|pool| {
            let block = match NonNull::new(pool.heads[len].get()) {
                Some(block) => block,
                None => refill(pool, len)?,
            };
            // SAFETY: a listed block is at least a header long and its
            // first word holds the next link written by `give`.
            let next = unsafe { block.as_ptr().cast::<*mut u8>().read() };
            pool.heads[len].set(next);
            // The next record table takes the block after this one.
            crate::runtime::prefetch::soon(next);
            pool.counts[len].set(pool.counts[len].get() - 1);
            Some(block)
        })
        .ok()
        .flatten()
    }

    /// Carves a batch of region blocks for `len` keys onto an empty list and
    /// returns the first, or `None` once the region is spent.
    #[cold]
    #[inline(never)]
    fn refill(pool: &Pool, len: usize) -> Option<NonNull<u8>> {
        let region = &REGIONS[len];
        let mut base = region.base.load(Ordering::Acquire);
        if base.is_null() {
            let layout = region_layout(len);
            // SAFETY: the layout holds at least one block, so it is not empty.
            let fresh = unsafe { std::alloc::alloc(layout) };
            if fresh.is_null() {
                return None;
            }
            match region.base.compare_exchange(
                std::ptr::null_mut(),
                fresh,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => base = fresh,
                Err(winner) => {
                    // SAFETY: `fresh` was allocated above with this layout
                    // and nothing else saw it.
                    unsafe { std::alloc::dealloc(fresh, layout) };
                    base = winner;
                }
            }
        }
        let first = region.next.fetch_add(BATCH, Ordering::Relaxed);
        if first >= REGION_BLOCKS {
            // Keep the counter from wrapping on a long run.
            region.next.store(REGION_BLOCKS, Ordering::Relaxed);
            return None;
        }
        let count = BATCH.min(REGION_BLOCKS - first);
        let stride = stride(len);
        for index in 0..count {
            // SAFETY: every block lies inside the region, and this thread
            // owns the blocks it just reserved.
            unsafe {
                let block = base.add((first + index) * stride);
                let next = if index + 1 < count {
                    base.add((first + index + 1) * stride)
                } else {
                    std::ptr::null_mut()
                };
                block.cast::<*mut u8>().write(next);
            }
        }
        // The list is empty, so the batch is the whole list.
        pool.heads[len].set(base.wrapping_add(first * stride));
        pool.counts[len].set(count);
        arm_guard();
        NonNull::new(pool.heads[len].get())
    }

    /// Takes back a block of `len` slots whose contents are dropped: lists
    /// it, or frees it when the list is full or the thread's pool is gone.
    ///
    /// # Safety
    ///
    /// `block` came from the allocator with the layout for `len` keys, and
    /// nothing refers to it any more.
    #[inline]
    pub(super) unsafe fn give(len: usize, block: *mut u8) {
        let listed = POOL
            .try_with(|pool| {
                let count = pool.counts[len].get();
                if count >= POOL_CAP && !in_region(len, block) {
                    return false;
                }
                if count == 0 {
                    arm_guard();
                }
                // SAFETY: the block holds at least a pointer.
                unsafe { block.cast::<*mut u8>().write(pool.heads[len].get()) };
                pool.heads[len].set(block);
                pool.counts[len].set(count + 1);
                true
            })
            .unwrap_or(false);
        if !listed {
            // SAFETY: as the caller promises.
            unsafe { std::alloc::dealloc(block, OtherEntries::shaped_layout(len)) };
        }
    }

    /// Blocks listed for `len` keys on this thread.
    #[cfg(test)]
    pub(super) fn listed(len: usize) -> usize {
        POOL.with(|pool| pool.counts[len].get())
    }
}

#[cfg(test)]
pub(crate) use block_pool::{POOL_CAP as BLOCK_POOL_CAP, REGION_BLOCKS as BLOCK_REGION_BLOCKS};

/// Blocks listed for `len` keys on this thread.
#[cfg(test)]
pub(crate) fn block_pool_listed(len: usize) -> usize {
    block_pool::listed(len)
}

/// Low bit of the pointer of a shaped table's block. Both blocks are
/// aligned to 8, so the bit is free, and the inlined field read tells the
/// two apart with a test on the pointer it already holds instead of a load.
const SHAPED_TAG: usize = 1;

const _: () = assert!(
    std::mem::align_of::<IndexedTableEntries>() > SHAPED_TAG
        && std::mem::align_of::<ShapedAlloc>() > SHAPED_TAG
        && std::mem::align_of::<Value>() > SHAPED_TAG
);

/// Owner of a hashed or a shaped table's entries: a pointer to a boxed
/// [`IndexedTableEntries`], or to a shaped block with [`SHAPED_TAG`] set.
#[repr(transparent)]
pub(crate) struct OtherEntries {
    base: std::ptr::NonNull<u8>,
}

#[allow(unsafe_code)]
// SAFETY: the pointer uniquely owns a heap block of `IndexedTableEntries` or
// of `Value`s and an `Arc<TableShape>`, as a `Box` would; see the assertion.
unsafe impl Send for OtherEntries {}
#[allow(unsafe_code)]
// SAFETY: as above, shared access only reads through `&self`.
unsafe impl Sync for OtherEntries {}

const _: () = {
    fn assert<T: Send + Sync>() {}
    let _ = assert::<Value>;
    let _ = assert::<Arc<TableShape>>;
    let _ = assert::<IndexedTableEntries>;
};

pub(crate) enum OtherRef<'a> {
    Indexed(&'a IndexedTableEntries),
    Shaped(&'a ShapedEntries),
}

pub(crate) enum OtherMut<'a> {
    Indexed(&'a mut IndexedTableEntries),
    Shaped(&'a mut ShapedEntries),
}

/// One value slot per key of the shape. A nil slot is an absent key, so a
/// table created with its constructor's keys looks empty until they are
/// stored. Operations that need owned keys or a changing key set convert the
/// table to [`SmallOrderedEntries`] first (see [`TableEntries::materialize`]).
///
/// This is a view of the allocation an [`OtherEntries`] owns, with the same
/// tagged pointer, so `&OtherEntries` casts to it. Slot access goes through the
/// pointer, never through a reference to the header, so it keeps the
/// provenance of the whole block.
#[repr(transparent)]
pub(crate) struct ShapedEntries {
    tagged: std::ptr::NonNull<u8>,
}

#[allow(
    clippy::cast_ptr_alignment,
    reason = "every block is allocated with the alignment of the type cast to, and the tag bit is cleared first"
)]
impl OtherEntries {
    fn indexed(entries: IndexedTableEntries) -> Self {
        Self {
            base: std::ptr::NonNull::from(Box::leak(Box::new(entries))).cast(),
        }
    }

    #[allow(unsafe_code)]
    #[inline]
    fn shaped(shape: Arc<TableShape>) -> Self {
        let len = shape.len();
        // SAFETY: a shape holds at most `MAX_SHAPE_KEYS` keys. Telling the
        // optimizer so keeps the nil fill below from becoming an unrolled
        // loop with a remainder, which cost more than the few slots.
        unsafe { std::hint::assert_unchecked(len <= MAX_SHAPE_KEYS) };
        let layout = Self::shaped_layout(len);
        let raw = match block_pool::take(len) {
            Some(block) => block.as_ptr(),
            // SAFETY: the layout holds at least the header, so its size is non-zero.
            None => unsafe { std::alloc::alloc(layout) },
        };
        let Some(block) = std::ptr::NonNull::new(raw) else {
            std::alloc::handle_alloc_error(layout);
        };
        // SAFETY: the block is large enough for the header and `len` values.
        unsafe {
            raw.cast::<ShapedAlloc>().write(ShapedAlloc { shape });
            let slots = raw.add(SHAPED_SLOTS_OFFSET).cast::<Value>();
            for index in 0..len {
                slots.add(index).write(Value::Nil);
            }
        }
        Self {
            base: block.map_addr(|addr| addr | SHAPED_TAG),
        }
    }

    pub(super) fn shaped_layout(len: usize) -> std::alloc::Layout {
        let align = std::mem::align_of::<ShapedAlloc>().max(std::mem::align_of::<Value>());
        std::alloc::Layout::from_size_align(
            SHAPED_SLOTS_OFFSET + len * std::mem::size_of::<Value>(),
            align,
        )
        .expect("shaped table layout is small")
    }

    #[inline]
    fn is_shaped(&self) -> bool {
        self.base.as_ptr().addr() & SHAPED_TAG != 0
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn view(&self) -> OtherRef<'_> {
        if self.is_shaped() {
            // SAFETY: the tag says which block this is; `ShapedEntries` and
            // `OtherEntries` are both a transparent pointer to it.
            OtherRef::Shaped(unsafe { &*std::ptr::from_ref(self).cast::<ShapedEntries>() })
        } else {
            // SAFETY: an untagged pointer is the box `Self::indexed` leaked.
            OtherRef::Indexed(unsafe { &*self.base.as_ptr().cast::<IndexedTableEntries>() })
        }
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn view_mut(&mut self) -> OtherMut<'_> {
        if self.is_shaped() {
            // SAFETY: as in `view`, and `&mut self` is the only access.
            OtherMut::Shaped(unsafe { &mut *std::ptr::from_mut(self).cast::<ShapedEntries>() })
        } else {
            // SAFETY: as above, and `&mut self` is the only access.
            OtherMut::Indexed(unsafe { &mut *self.base.as_ptr().cast::<IndexedTableEntries>() })
        }
    }

    /// [`TableEntries::get_string`] for a table that is not ordered: the one
    /// call its inlined copies make, which ends in the probe's or the
    /// shape's own leaf.
    #[inline(never)]
    fn get_string(&self, key: &LuaString) -> Option<&Value> {
        match self.view() {
            // A tombstone reads as absent here, as a nil field of a shape does.
            OtherRef::Indexed(entries) => entries
                .probe_string(key)
                .filter(|value| !matches!(value, Value::Nil)),
            OtherRef::Shaped(entries) => entries.get_live(key),
        }
    }

    /// The live value under `key` of a hashed or a shaped table, behind one
    /// call.
    #[inline(never)]
    fn get_live(&self, key: &LuaString) -> Option<&Value> {
        match self.view() {
            OtherRef::Indexed(entries) => entries.probe_string_live(key),
            OtherRef::Shaped(entries) => entries.get_live(key),
        }
    }

    #[inline(never)]
    fn len(&self) -> usize {
        match self.view() {
            OtherRef::Indexed(entries) => entries.live,
            OtherRef::Shaped(entries) => entries.present(),
        }
    }
}

#[allow(
    clippy::cast_ptr_alignment,
    reason = "every block is allocated with the alignment of the type cast to, and the tag bit is cleared first"
)]
impl Drop for OtherEntries {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        // A row may point into the block, of a live table that changes
        // representation (`TableEntries::materialize`) as much as of a dead
        // one.
        super::key_layout_changed();
        if !self.is_shaped() {
            // SAFETY: the block came from `Box::new` of an `IndexedTableEntries`.
            drop(unsafe { Box::from_raw(self.base.as_ptr().cast::<IndexedTableEntries>()) });
            return;
        }
        // SAFETY: the view is of this table's block, which holds the header
        // and `len` values from `Self::shaped`, dropped once here before it
        // is freed.
        unsafe {
            let shaped = &mut *std::ptr::from_mut(self).cast::<ShapedEntries>();
            let len = shaped.shape().len();
            let header = shaped.header();
            std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(shaped.slots(), len));
            let shape = std::ptr::read(&raw const (*header).shape);
            if shape.is_pinned() {
                // The table borrowed the count pinning leaked; see
                // `TableEntries::with_shape_ref`.
                std::mem::forget(shape);
            } else {
                drop(shape);
            }
            block_pool::give(len, header.cast());
        }
    }
}

#[allow(
    clippy::cast_ptr_alignment,
    reason = "every block is allocated with the alignment of the type cast to, and the tag bit is cleared first"
)]
impl ShapedEntries {
    /// The block's start, without the tag.
    #[inline]
    fn header(&self) -> *mut ShapedAlloc {
        self.tagged.as_ptr().wrapping_byte_sub(SHAPED_TAG).cast()
    }

    #[allow(unsafe_code)]
    #[inline]
    fn shape_arc(&self) -> &Arc<TableShape> {
        // SAFETY: the header is initialized and lives as long as `self`.
        unsafe { &(*self.header()).shape }
    }

    #[inline]
    fn shape(&self) -> &TableShape {
        self.shape_arc()
    }

    #[allow(unsafe_code)]
    #[inline]
    fn slots(&self) -> *mut Value {
        // SAFETY: the slots follow the header in the same block.
        unsafe {
            self.header()
                .cast::<u8>()
                .add(SHAPED_SLOTS_OFFSET)
                .cast::<Value>()
        }
    }

    #[allow(unsafe_code)]
    #[inline]
    fn values(&self) -> &[Value] {
        // SAFETY: `slots` points to `shape.len()` initialized values.
        unsafe { std::slice::from_raw_parts(self.slots(), self.shape().len()) }
    }

    #[allow(unsafe_code)]
    #[inline]
    fn values_mut(&mut self) -> &mut [Value] {
        // SAFETY: as in `values`, and `&mut self` is the only access.
        unsafe { std::slice::from_raw_parts_mut(self.slots(), self.shape().len()) }
    }

    /// [`TableEntries::holds_string_slot`] for a shaped table: `slot` is one
    /// of this block's slots, and the shape names `key` at it.
    fn holds_string_slot(&self, slot: usize, key: &LuaString) -> bool {
        let size = std::mem::size_of::<Value>();
        let Some(offset) = slot.checked_sub(self.slots().addr()) else {
            return false;
        };
        let index = offset / size;
        offset % size == 0
            && index < self.shape().len()
            && self.shape().hashes()[index] == key.cached_hash()
            && matches!(&self.shape().keys()[index], TableKey::String(stored)
                if stored.ptr_eq(key) || stored == key)
    }

    /// Slot `index`, which the caller took from this table's shape.
    #[allow(unsafe_code, clippy::inline_always)]
    #[inline(always)]
    fn slot_mut(&mut self, index: usize) -> &mut Value {
        debug_assert!(index < self.shape().len());
        // SAFETY: shape lookups return indexes below `shape.len()`, and
        // the block holds that many values.
        unsafe { &mut *self.slots().add(index) }
    }

    /// Present entries, which the ordered table would hold.
    #[allow(unsafe_code)]
    #[inline]
    fn present(&self) -> usize {
        let len = self.shape().len();
        // SAFETY: a shape holds at most `MAX_SHAPE_KEYS` keys. Telling the
        // optimizer so keeps it from vectorizing a loop of a few iterations,
        // whose setup cost several times the count itself.
        unsafe { std::hint::assert_unchecked(len <= MAX_SHAPE_KEYS) };
        self.values()[..len]
            .iter()
            .filter(|value| !matches!(value, Value::Nil))
            .count()
    }

    /// [`TableEntries::string_value_ptrs`] for a shaped table: the address
    /// of the slot of each wanted key that holds a value of its kind.
    ///
    /// Tables built by one constructor share a shape, so a caller that
    /// resolves many of them can keep what it found. After the `wanted.len()`
    /// hints the ordered representation uses come a counter and then
    /// [`SHAPE_CACHE_LINES`] lines of one shape id and, for each key, a word
    /// with the offset of its slot in the shape and the tag the slot must
    /// have (`usize::MAX` for a key the shape lacks); see
    /// [`shape_cache_len`]. A table of a shape on a line needs no key
    /// search. This writes every entry of `out`.
    #[allow(unsafe_code)]
    #[inline(never)]
    fn string_value_ptrs(
        &mut self,
        wanted: &[(Value, NativeSlotKind)],
        hints: &mut [usize],
        out: &mut [usize],
        absent: Option<usize>,
    ) {
        /// The slot's offset is below this bit, the tag it must have above.
        const TAG_SHIFT: u32 = usize::BITS - 8;
        const OFFSET_MASK: usize = (1 << TAG_SHIFT) - 1;
        /// A key the shape lacks, and one it lacks that is wanted `Absent`.
        const MISSING: usize = usize::MAX;
        const ABSENT: usize = usize::MAX - 1;
        let count = wanted.len();
        let out = &mut out[..count];
        let id = usize::try_from(self.shape().id).unwrap_or(0);
        let stride = count + 1;
        let first_line = count + 1;
        let mut local = Vec::new();
        let words: &[usize] = 'cached: {
            if hints.len() >= shape_cache_len(count) {
                for line in 0..SHAPE_CACHE_LINES {
                    let start = first_line + line * stride;
                    // SAFETY: the length check above covers every line.
                    if unsafe { *hints.get_unchecked(start) } == id {
                        break 'cached &hints[start + 1..start + stride];
                    }
                }
            }
            // Another shape: find the keys, and remember them on the line
            // after the one taken last.
            let shape = self.shape();
            let find = |(key, kind): &(Value, NativeSlotKind)| {
                let missing = if *kind == NativeSlotKind::Absent {
                    ABSENT
                } else {
                    MISSING
                };
                match key {
                    Value::String(key) => {
                        shape
                            .find_string(key, key.cached_hash())
                            .map_or(missing, |index| {
                                (index * std::mem::size_of::<Value>())
                                    | (usize::from(kind.tag()) << TAG_SHIFT)
                            })
                    }
                    _ => missing,
                }
            };
            if hints.len() >= shape_cache_len(count) {
                let victim = hints[count] % SHAPE_CACHE_LINES;
                hints[count] = hints[count].wrapping_add(1);
                let start = first_line + victim * stride;
                hints[start] = id;
                for (word, key) in hints[start + 1..start + stride].iter_mut().zip(wanted) {
                    *word = find(key);
                }
                &hints[start + 1..start + stride]
            } else {
                local.extend(wanted.iter().map(find));
                &local
            }
        };
        let slots = self.slots().cast::<u8>();
        let marker = absent.unwrap_or(0);
        for (found, &word) in out.iter_mut().zip(words) {
            if word >= ABSENT {
                // A key the shape lacks is nil, which only a wanted `Absent`
                // takes as its answer.
                *found = if word == ABSENT { marker } else { 0 };
                continue;
            }
            // SAFETY: the word came from a lookup in this shape, or in
            // another with this shape's id, which is this one.
            let slot = unsafe { slots.add(word & OFFSET_MASK) };
            // A slot is what the key wants when its tag is the one the word
            // names; nil is the tag of `Absent`.
            *found = if unsafe { *slot } == (word >> TAG_SHIFT) as u8 {
                if word >> TAG_SHIFT == usize::from(NATIVE_VALUE_TAG_NIL) {
                    marker
                } else {
                    slot as usize
                }
            } else {
                0
            };
        }
    }

    /// Stores `value` in the empty slot the shape names for `key`. `false`,
    /// having stored nothing, if the shape lacks the key or the slot holds a
    /// value.
    fn init_slot(&mut self, key: &LuaString, value: &Value) -> bool {
        let Some(index) = self.shape().find_string(key, key.cached_hash()) else {
            return false;
        };
        let slot = self.slot_mut(index);
        if !matches!(slot, Value::Nil) {
            return false;
        }
        *slot = value.clone();
        true
    }

    /// Out of line so the inlined ordered scan stays as small as before.
    #[inline(never)]
    fn get_live(&self, key: &LuaString) -> Option<&Value> {
        let index = self.shape().find_string(key, key.cached_hash())?;
        Some(&self.values()[index]).filter(|value| !matches!(value, Value::Nil))
    }

    /// [`Self::get_live`] with the shape scan inlined, for the VM's field
    /// read, which loses a call and a prologue per read on record tables.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[allow(unsafe_code)]
    fn get_live_scan(&self, key: &LuaString) -> Option<&Value> {
        let index = self.shape().find_string(key, key.cached_hash())?;
        // SAFETY: as in `slot_mut`.
        let value = unsafe { &*self.slots().add(index) };
        Some(value).filter(|value| !matches!(value, Value::Nil))
    }

    fn get(&self, key: &TableKey) -> Option<&Value> {
        let index = self.shape().position(key).ok()?;
        Some(&self.values()[index]).filter(|value| !matches!(value, Value::Nil))
    }

    #[inline(never)]
    fn get_string_mut(&mut self, key: &LuaString) -> Option<&mut Value> {
        let index = self.shape().find_string(key, key.cached_hash())?;
        Some(&mut self.values_mut()[index]).filter(|value| !matches!(value, Value::Nil))
    }

    /// Out of line so the ordered traversals stay as small as before.
    #[inline(never)]
    fn for_each(&self, mut visit: impl FnMut(&TableKey, &Value)) {
        for (key, value) in self.shape().keys().iter().zip(self.values()) {
            if !matches!(value, Value::Nil) {
                visit(key, value);
            }
        }
    }

    #[inline(never)]
    fn for_each_after(
        &self,
        after: Option<&TableKey>,
        limit: usize,
        mut visit: impl FnMut(&TableKey, &Value),
    ) -> Option<TableKey> {
        let mut last: Option<&TableKey> = None;
        let mut visited = 0;
        let mut start = self.start_after(after);
        while let Some(index) = self.next_present(start) {
            if visited == limit {
                return last.cloned();
            }
            let key = &self.shape().keys()[index];
            visit(key, &self.values()[index]);
            last = Some(key);
            visited += 1;
            start = index + 1;
        }
        None
    }

    /// A `next` step from the key `previous` names, found by its hash and
    /// identity. A key the shape lacks, a slot that holds nil (a key `next`
    /// may not continue from) and a key that is not a string take the general
    /// lookup, which raises the errors.
    ///
    /// `hint` works as in [`TableEntries::next_after_value`].
    fn next_after(&self, previous: Option<&Value>, hint: &mut usize) -> HashNext<'_> {
        let start = match previous {
            None => 0,
            Some(Value::String(key)) => {
                let index = match self.shape().keys().get(*hint) {
                    Some(TableKey::String(stored)) if stored.ptr_eq(key) => *hint,
                    _ => match self.shape().find_string(key, key.cached_hash()) {
                        Some(index) => index,
                        None => return HashNext::Slow,
                    },
                };
                if matches!(self.values()[index], Value::Nil) {
                    return HashNext::Slow;
                }
                index + 1
            }
            Some(_) => return HashNext::Slow,
        };
        match self.next_present(start) {
            Some(index) => {
                *hint = index;
                HashNext::Entry(&self.shape().keys()[index], &self.values()[index])
            }
            None => HashNext::End,
        }
    }

    #[inline(never)]
    fn first_after(&self, after: Option<&TableKey>) -> Option<(&TableKey, &Value)> {
        let index = self.next_present(self.start_after(after))?;
        Some((&self.shape().keys()[index], &self.values()[index]))
    }

    /// Index of the first present slot at or after `start`.
    fn next_present(&self, start: usize) -> Option<usize> {
        let values = self.values();
        (start..values.len()).find(|index| !matches!(values[*index], Value::Nil))
    }

    fn start_after(&self, after: Option<&TableKey>) -> usize {
        match after {
            Some(after) => self
                .shape()
                .position(after)
                .map_or_else(|index| index, |index| index + 1),
            None => 0,
        }
    }
}

/// The most [`TableEntries::allocated_bytes`] reports for a shaped table:
/// what an ordered table of [`MAX_SHAPE_KEYS`] entries holds.
pub(crate) const SHAPED_MAX_ALLOCATED_BYTES: usize = ordered_capacity(MAX_SHAPE_KEYS)
    * std::mem::size_of::<OrderedEntry>()
    + std::mem::size_of::<OrderedVec>();

/// Capacity the ordered representation holds after `count` insertions.
#[inline]
const fn ordered_capacity(count: usize) -> usize {
    // Mirrors `insert_ordered`'s growth: one entry, then four.
    match count {
        0 => 0,
        1 => 1,
        2..=4 => 4,
        count => count.next_power_of_two(),
    }
}

/// Inserts `entry` at `index`. The first entry gets a block of its own size:
/// many tables built by assignment never take a second key, and the vector's
/// next growth goes straight to four entries.
#[inline]
fn insert_ordered(entries: &mut OrderedVec, index: usize, entry: OrderedEntry) {
    entries.insert(index, entry);
}

/// Sorted key/value storage for tables below the hashed promotion threshold.
///
/// Record tables keep a handful of entries for their whole lifetime. A sorted
/// contiguous vector serves those sizes with binary-search lookups and
/// shift-on-insert writes, avoiding the B-tree node traversal and allocation
/// of the map representation while preserving its exact key ordering for
/// iteration, `next`, and range queries.
#[derive(Default)]
pub(crate) struct SmallOrderedEntries {
    entries: OrderedVec,
}

/// One ordered entry. `hash` is the cached hash of a string key, stored
/// inline so a string lookup rejects non-matching entries without touching
/// the key's `Arc`. It is zero for other key kinds; a colliding hash only
/// makes the lookup confirm the key, never accept a wrong one.
pub(crate) struct OrderedEntry {
    pub(super) key: TableKey,
    pub(super) value: Value,
    pub(super) hash: u64,
}

impl OrderedEntry {
    #[inline]
    pub(super) fn new(key: TableKey, value: Value) -> Self {
        let hash = match &key {
            // Lookups skip entries whose hash differs, so a stored zero would
            // hide the key. `cached_hash` is one load once the string is hashed.
            TableKey::String(string) => string.cached_hash(),
            _ => 0,
        };
        Self { key, value, hash }
    }
}

impl SmallOrderedEntries {
    /// The traversal snapshot as native code reads it, for a table too small
    /// to keep one: builds it in `scratch`, one slot per entry in key order,
    /// with the position of the entry `previous` names. `None` when no entry
    /// does. The addresses stay valid until the entries change.
    fn native_cursor(
        &mut self,
        previous: &Value,
        scratch: &mut NativeCursorScratch,
    ) -> Option<(usize, usize, usize)> {
        let key = TableKey::try_from(previous).ok()?;
        let position = self.position(&key).ok()?;
        scratch.slots.clear();
        scratch.slots.extend(self.entries.iter().map(|entry| {
            let (key_tag, key_bits) = native_key_words(&entry.key);
            NativeCursorSlot {
                key_tag,
                key_bits,
                value: std::ptr::from_ref(&entry.value),
                key: std::ptr::from_ref(&entry.key) as usize,
            }
        }));
        scratch.position = position as u64;
        Some((
            scratch.slots.as_ptr() as usize,
            scratch.slots.len(),
            std::ptr::from_mut(&mut scratch.position) as usize,
        ))
    }

    #[inline]
    fn position(&self, key: &TableKey) -> Result<usize, usize> {
        self.entries.binary_search_by(|entry| entry.key.cmp(key))
    }

    fn get(&self, key: &TableKey) -> Option<&Value> {
        self.position(key)
            .ok()
            .map(|index| &self.entries[index].value)
    }

    /// Index of the entry holding the string `key`, whose cached hash is
    /// `hash`. Compares the inline hashes first, then confirms by pointer
    /// identity and by bytes, both inline: strings are not interned, so a
    /// key built elsewhere is a different allocation on most hits. Only a
    /// hash match whose bytes differ continues out of line.
    ///
    /// Always inlined into its caller. The callers decide where the scan
    /// lands: [`Self::scan_string`] is the out-of-line leaf, and only [`TableEntries::get_string_live_scan`]
    /// is meant to be inlined (into the VM's table-field read).
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn find_string(&self, key: &LuaString, hash: u64) -> Option<usize> {
        for (index, entry) in self.entries.iter().enumerate() {
            if entry.hash == hash
                && let TableKey::String(stored) = &entry.key
            {
                if stored.ptr_eq(key) || stored == key {
                    return Some(index);
                }
                return self.find_string_after_collision(index + 1, key, hash);
            }
        }
        None
    }

    /// Continues [`Self::find_string`] from `start` after a hash match
    /// whose bytes differed (a true hash collision).
    #[cold]
    #[inline(never)]
    fn find_string_after_collision(
        &self,
        start: usize,
        key: &LuaString,
        hash: u64,
    ) -> Option<usize> {
        let entries = self.entries.get(start..)?;
        entries
            .iter()
            .position(|entry| {
                entry.hash == hash
                    && matches!(&entry.key, TableKey::String(stored) if stored == key)
            })
            .map(|offset| start + offset)
    }

    /// [`TableEntries::holds_string_slot`] for an ordered table: `slot` is
    /// the value of one of the entries, and that entry's key is `key`.
    fn holds_string_slot(&self, slot: usize, key: &LuaString) -> bool {
        let size = std::mem::size_of::<OrderedEntry>();
        let first = self.entries.as_ptr().addr() + std::mem::offset_of!(OrderedEntry, value);
        let Some(offset) = slot.checked_sub(first) else {
            return false;
        };
        offset % size == 0
            && self.entries.get(offset / size).is_some_and(|entry| {
                entry.hash == key.cached_hash()
                    && matches!(&entry.key, TableKey::String(stored)
                        if stored.ptr_eq(key) || stored == key)
            })
    }

    /// [`Self::find_string`] as an out-of-line leaf returning the value, for
    /// callers that do not earn the inlined scan.
    ///
    /// The leaf makes no calls but in tail position, so it saves no
    /// registers: a stored key that is the lookup key's own allocation (a
    /// shared constant, see [`crate::string::constants`]) answers in the
    /// loop, and anything else (an unhashed key, a hash match by another
    /// allocation) continues in [`Self::scan_string_confirm`].
    #[inline(never)]
    fn scan_string(&self, key: &LuaString) -> Option<&Value> {
        let hash = key.known_hash();
        if hash == 0 {
            return self.scan_string_confirm(key);
        }
        for entry in self.entries.iter() {
            if entry.hash == hash {
                if let TableKey::String(stored) = &entry.key
                    && stored.ptr_eq(key)
                {
                    return Some(&entry.value);
                }
                return self.scan_string_confirm(key);
            }
        }
        // Equal strings hash alike, so no entry holds the key.
        None
    }

    /// [`Self::scan_string`] past its identity scan: hashes the key if no
    /// one has, and compares bytes.
    #[inline(never)]
    fn scan_string_confirm(&self, key: &LuaString) -> Option<&Value> {
        let index = self.find_string(key, key.cached_hash())?;
        self.entries.get(index).map(|entry| &entry.value)
    }

    /// Address of the value stored under `key`, built from the vector's
    /// base pointer. No reference to any entry is formed, so addresses taken
    /// earlier for other keys stay valid under Rust's aliasing rules.
    fn value_ptr(&mut self, key: &TableKey) -> Option<*mut Value> {
        let index = self.position(key).ok()?;
        Some(
            self.entries
                .as_mut_ptr()
                .wrapping_add(index)
                .wrapping_byte_add(std::mem::offset_of!(OrderedEntry, value))
                .cast::<Value>(),
        )
    }

    /// [`TableEntries::string_value_ptrs`] for the ordered representation: a
    /// single scan of the entries, where one lookup per key would search the
    /// vector once to find the entry and again to take its address.
    fn string_value_ptrs(
        &mut self,
        wanted: &[(Value, NativeSlotKind)],
        hints: &mut [usize],
        out: &mut [usize],
    ) {
        // `out` holds the entry's index plus one until the base pointer is
        // known, so the lookups keep no reference to the entries.
        for (((key, kind), found), hint) in wanted.iter().zip(out.iter_mut()).zip(hints) {
            let Value::String(key) = key else {
                continue;
            };
            // Tables built alike hold a key at one position, so the position
            // of the last table settles most lookups with one comparison. A
            // key the table lacks is the common miss (a row names the fields
            // of every table of the trace): the scan compares the entries'
            // stored hashes, not their keys.
            let hash = key.cached_hash();
            let hinted = self.entries.get(*hint).is_some_and(|entry| {
                entry.hash == hash
                    && matches!(&entry.key, TableKey::String(stored)
                        if stored.ptr_eq(key) || stored == key)
            });
            let index = if hinted {
                Some(*hint)
            } else {
                self.find_string(key, hash).inspect(|index| *hint = *index)
            };
            let index = index.map(|index| (index, kind.holds(&self.entries[index].value)));
            if let Some((index, true)) = index {
                *found = index + 1;
            }
        }
        let base = self.entries.as_mut_ptr();
        for found in out {
            if *found != 0 {
                *found = base
                    .wrapping_add(*found - 1)
                    .wrapping_byte_add(std::mem::offset_of!(OrderedEntry, value))
                    as usize;
            }
        }
    }

    #[inline]
    fn insert(&mut self, key: TableKey, value: Value) -> Option<Value> {
        match self.position(&key) {
            Ok(index) => Some(std::mem::replace(&mut self.entries[index].value, value)),
            Err(index) => {
                insert_ordered(&mut self.entries, index, OrderedEntry::new(key, value));
                None
            }
        }
    }

    #[inline]
    fn remove(&mut self, key: &TableKey) -> Option<Value> {
        self.position(key)
            .ok()
            .map(|index| self.entries.remove(index).value)
    }
}

/// Hashed entries with the ordered key set that `next` walks.
pub(crate) struct IndexedTableEntries {
    pub(crate) values: hashbrown::HashMap<TableKey, Value, TableKeyBuildHasher>,
    /// Every key in `values`, tombstones included, in key order. Built on the
    /// first traversal, so a table that is filled and only looked up (the
    /// library tables a start installs) never pays for its B-tree. While it is
    /// unbuilt, insertions and removals leave it alone.
    order: OnceLock<BTreeSet<TableKey>>,
    /// Entries of `values` whose value is not nil. A deleted string key stays
    /// in `values` and `order` holding nil (a tombstone, as in PUC Lua), so
    /// deleting and reinserting it touches no ordered set. `compact` drops
    /// the tombstones once they outnumber the live entries.
    live: usize,
    /// Names this map's current bucket layout. Overwriting a value keeps it.
    /// Adding or removing a key, or any other change that can move or free
    /// a bucket, takes a fresh number from a process-wide counter, so no two
    /// layouts ever share one. [`StringReadCache`] relies on that.
    epoch: u64,
    /// The traversal cursor `next` keeps for this layout.
    cursor: NextCursor,
    /// The string index native code probes for this layout.
    probe: ProbeIndex,
}

/// The string index a hashed table keeps for native trace code: every string
/// key of the layout numbered `built`, with the address of its value. It is
/// valid exactly while `built` equals the map's layout epoch, which changes
/// whenever a key is added or removed or a bucket can move (see
/// [`IndexedTableEntries::layout_changed`]); overwriting a value, or deleting
/// a key where it stands, leaves every address in place.
///
/// An index costs one pass over the table, so a big table builds it only on
/// the second request for one layout: a loop that changes the key set between
/// runs never pays for one.
#[derive(Default)]
struct ProbeIndex {
    built: u64,
    seen: u64,
    slots: Vec<NativeProbeSlot>,
}

/// Tables with fewer entries than this build their index on the first request.
const PROBE_EAGER_ENTRIES: usize = 64;

/// The most entries a table with an index has: `fill_probe` gives it three
/// slots an entry, rounded up to a power of two.
const PROBE_MAX_ENTRIES: usize = crate::runtime::native_views::NATIVE_PROBE_MAX_SLOTS / 3;

/// The first slot a probe for `hash` looks at in an index with `mask`.
/// Native code computes the same (see `native_probe_start`).
#[inline]
fn probe_start(hash: u64, mask: usize) -> usize {
    native_probe_start(hash) as usize & mask
}

/// Fills `slots` with an index over `keys`, the string keys of a table and the
/// address of each one's value, at most a third full. A probe compiled in
/// line pays about twenty instructions for each bucket it walks past its
/// first, and an index near half full sends every second probe one bucket
/// on. With room for twice the keys, `table/hash_read` (2000 keys, scale 3)
/// counted 96.1 M instructions and `table/hash_write` 139.1 M; with room for
/// three times, 82.1 M and 117.7 M.
fn fill_probe<'a>(
    slots: &mut Vec<NativeProbeSlot>,
    count: usize,
    keys: impl Iterator<Item = (&'a LuaString, *const Value)>,
) -> NativeStringProbe {
    let capacity = (count * 3).next_power_of_two().max(4);
    slots.clear();
    // One slot beyond the index: the header of a hashed table's index, where
    // native code finds its live-entry counter ([`NativeStringProbe::live`]).
    slots.resize(capacity + 1, NativeProbeSlot::EMPTY);
    let mask = capacity - 1;
    for (key, value) in keys {
        let hash = key.cached_hash();
        let mut at = probe_start(hash, mask);
        while slots[at].key != 0 {
            at = (at + 1) & mask;
        }
        slots[at] = NativeProbeSlot::of(hash, key.identity(), value);
    }
    NativeStringProbe {
        slots: slots.as_ptr() as usize,
        mask: mask as u64,
        writable: false,
        tombstones: false,
    }
}

/// What `next` keeps between steps of a traversal over a hashed table: the
/// entries in key order, as pointers into the map's buckets, and the position
/// of the entry returned last.
///
/// The snapshot is valid exactly while `built` equals the map's layout epoch.
/// The epoch changes whenever a key is added or removed or a bucket can move
/// (see [`IndexedTableEntries::layout_changed`]), and overwriting a value
/// leaves buckets in place, so every pointer stays valid and current while
/// the epochs agree.
///
/// A snapshot costs one pass over the table, so it is built only after
/// `steps` traversal steps found the layout unchanged, at least a sixteenth
/// of the table: a loop that changes the key set between steps never pays for
/// one, and a traversal pays for it once.
#[derive(Default)]
struct NextCursor {
    built: u64,
    seen: u64,
    steps: usize,
    position: usize,
    slots: Vec<NativeCursorSlot>,
}

/// The outcome of a hashed `next` step that did not take the general path.
pub(crate) enum HashNext<'a> {
    Entry(&'a TableKey, &'a Value),
    /// No entry follows.
    End,
    /// The cursor cannot answer: use the general lookup.
    Slow,
}

/// The kind and the integer of `key` as a [`NativeCursorSlot`] holds them.
fn native_key_words(key: &TableKey) -> (u64, u64) {
    use crate::runtime::native_views::{NATIVE_VALUE_TAG_INTEGER, NATIVE_VALUE_TAG_STRING};
    match key {
        TableKey::Integer(integer) => {
            (u64::from(NATIVE_VALUE_TAG_INTEGER), integer.cast_unsigned())
        }
        TableKey::String(_) => (u64::from(NATIVE_VALUE_TAG_STRING), 0),
        _ => (0, 0),
    }
}

/// Whether `value` names `key` without building a key from it.
#[inline(always)]
fn key_names_value(key: &TableKey, value: &Value) -> bool {
    match (key, value) {
        (TableKey::String(key), Value::String(value)) => key.ptr_eq(value),
        (TableKey::Integer(key), Value::Integer(value)) => key == value,
        (TableKey::Integer(key), Value::Number(value)) => {
            super::key::integer_key_from_number(*value) == Some(*key)
        }
        (TableKey::Table(key), Value::Table(value)) => key == value,
        _ => false,
    }
}

/// Guards at and above this value belong to sites filled for an ordered
/// table: the guard is `ORDERED_SITE - index` of the entry. Layout numbers
/// and shape ids count up from one and never get this far, and an ordered
/// table holds fewer than 32 entries.
pub(crate) const ORDERED_SITE: u64 = u64::MAX;

/// What an inline-cache site stores for one table.
pub(crate) struct SiteFill {
    pub(crate) guard: u64,
    pub(crate) data: usize,
    /// For an ordered table, the key string whose address is `data`.
    pub(crate) pin: Option<LuaString>,
}

/// What [`TableEntries::tombstone_live_string`] found.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Tombstoned {
    /// The key held a value, which is now a deleted key.
    Deleted,
    /// A hashed table that holds no value under the key.
    Absent,
    /// A table that keeps no tombstones: the caller reads the key itself.
    Unhashed,
}

/// The last layout number handed out. Zero marks an empty cache slot.
static LAYOUT_EPOCH: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
thread_local! {
    /// How many hashed string probes this thread made, for tests that pin the
    /// probes a store costs.
    static STRING_PROBES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Counts one hashed string probe of a table (in tests only).
#[allow(clippy::inline_always)]
#[inline(always)]
fn note_string_probe() {
    #[cfg(test)]
    STRING_PROBES.with(|probes| probes.set(probes.get() + 1));
}

/// The hashed string probes this thread has made so far.
#[cfg(test)]
pub(crate) fn string_probes() -> u64 {
    STRING_PROBES.with(std::cell::Cell::get)
}

pub(crate) fn next_layout_epoch() -> u64 {
    LAYOUT_EPOCH.fetch_add(1, Ordering::Relaxed) + 1
}

impl IndexedTableEntries {
    fn new(values: hashbrown::HashMap<TableKey, Value, TableKeyBuildHasher>) -> Self {
        let live = values
            .values()
            .filter(|value| !matches!(value, Value::Nil))
            .count();
        Self {
            values,
            order: OnceLock::new(),
            live,
            epoch: next_layout_epoch(),
            cursor: NextCursor::default(),
            probe: ProbeIndex::default(),
        }
    }

    /// The ordered key set, built from `values` if no traversal has needed it.
    // Safe as map keys: `TableKey`'s only interior mutability is `LuaString`'s
    // write-once hash memo, a pure function of bytes no one mutates, and keys
    // hash and order by that immutable content.
    #[allow(clippy::mutable_key_type)]
    pub(crate) fn order(&self) -> &BTreeSet<TableKey> {
        self.order
            .get_or_init(|| self.values.keys().cloned().collect())
    }

    /// [`Self::order`] for a caller that changes the set.
    // Safe as map keys: `TableKey`'s only interior mutability is `LuaString`'s
    // write-once hash memo, a pure function of bytes no one mutates, and keys
    // hash and order by that immutable content.
    #[allow(clippy::mutable_key_type)]
    fn order_mut(&mut self) -> &mut BTreeSet<TableKey> {
        self.order
            .get_or_init(|| self.values.keys().cloned().collect());
        self.order.get_mut().expect("the set was just built")
    }

    /// Drops the tombstones, which takes a new layout number.
    fn compact(&mut self) {
        let dead: Vec<TableKey> = self
            .values
            .iter()
            .filter(|(_, value)| matches!(value, Value::Nil))
            .map(|(key, _)| key.clone())
            .collect();
        for key in &dead {
            if let Some(order) = self.order.get_mut() {
                order.remove(key);
            }
            self.values.remove(key);
        }
        self.layout_changed();
    }

    /// Tombstones that `values` holds beyond the live entries.
    fn dead(&self) -> usize {
        self.values.len() - self.live
    }

    /// The entry after `previous` (the first without one) in key order,
    /// through the traversal cursor.
    #[allow(unsafe_code)]
    fn next_after(&mut self, previous: Option<&Value>) -> HashNext<'_> {
        let epoch = self.epoch;
        if self.cursor.built != epoch {
            if self.cursor.seen != epoch {
                self.cursor.slots = Vec::new();
                self.cursor.built = 0;
                self.cursor.seen = epoch;
                self.cursor.steps = 0;
            }
            self.cursor.steps += 1;
            if self.cursor.steps < (self.values.len() / 16).max(4) {
                return HashNext::Slow;
            }
            let mut slots = Vec::with_capacity(self.values.len());
            for key in self.order() {
                if let Some((key, value)) = self.values.get_key_value(key) {
                    let (key_tag, key_bits) = native_key_words(key);
                    slots.push(NativeCursorSlot {
                        key_tag,
                        key_bits,
                        value,
                        key: std::ptr::from_ref(key) as usize,
                    });
                }
            }
            self.cursor.slots = slots;
            self.cursor.built = epoch;
        }
        let cursor = &mut self.cursor;
        let index = match previous {
            None => 0,
            Some(previous) => {
                let hinted = cursor.slots.get(cursor.position);
                // SAFETY: `built == epoch`, so the pointers are valid (see
                // `NextCursor`).
                if let Some(slot) = hinted
                    && key_names_value(unsafe { &*(slot.key as *const TableKey) }, previous)
                {
                    cursor.position + 1
                } else {
                    let Ok(key) = TableKey::try_from(previous) else {
                        return HashNext::Slow;
                    };
                    // SAFETY: as above.
                    match cursor.slots.binary_search_by(|slot| unsafe {
                        (*(slot.key as *const TableKey)).cmp(&key)
                    }) {
                        Ok(found) => found + 1,
                        Err(_) => return HashNext::Slow,
                    }
                }
            }
        };
        let mut index = index;
        while let Some(slot) = cursor.slots.get(index) {
            // SAFETY: as above. The references borrow `self`, which no
            // layout change can touch meanwhile.
            let (key, value) = unsafe { (&*(slot.key as *const TableKey), &*slot.value) };
            if !matches!(value, Value::Nil) {
                cursor.position = index;
                return HashNext::Entry(key, value);
            }
            // A tombstone: skip it.
            index += 1;
        }
        HashNext::End
    }

    /// The traversal snapshot as native code reads it: the address of the
    /// slots, how many there are and the address of the position of the entry
    /// returned last. `None` when no snapshot is built for the current
    /// layout (see [`NextCursor`]). The addresses stay valid until the layout
    /// changes.
    fn native_cursor(&mut self) -> Option<(usize, usize, usize)> {
        (self.cursor.built == self.epoch).then(|| {
            (
                self.cursor.slots.as_ptr() as usize,
                self.cursor.slots.len(),
                std::ptr::from_mut(&mut self.cursor.position) as usize,
            )
        })
    }

    /// The string index of the current layout for native code, built now if
    /// the table is small, has asked before or `eager` says the caller will
    /// use it for long (see [`ProbeIndex`]).
    fn native_probe(&mut self, eager: bool) -> Option<NativeStringProbe> {
        let epoch = self.epoch;
        // An index of more slots than native code can address has none.
        if self.values.len() > PROBE_MAX_ENTRIES {
            return None;
        }
        if self.probe.built != epoch {
            if !eager && self.values.len() >= PROBE_EAGER_ENTRIES && self.probe.seen != epoch {
                self.probe.seen = epoch;
                return None;
            }
            let strings = self.values.iter().filter_map(|(key, value)| match key {
                TableKey::String(string) => Some((string, std::ptr::from_ref(value))),
                _ => None,
            });
            fill_probe(&mut self.probe.slots, self.values.len(), strings);
            self.probe.built = epoch;
        }
        // The counter's address is written at every request: it names the
        // field of the entries as they sit now.
        let header = self.probe.slots.len() - 1;
        self.probe.slots[header].hash = std::ptr::from_mut(&mut self.live) as u64;
        Some(NativeStringProbe {
            slots: self.probe.slots.as_ptr() as usize,
            mask: (header - 1) as u64,
            writable: false,
            tombstones: true,
        })
    }

    /// Call before any change that adds, removes or moves a bucket.
    #[inline]
    fn layout_changed(&mut self) {
        super::key_layout_changed();
        self.epoch = next_layout_epoch();
    }

    /// The one hashed string probe. Every string getter calls this single
    /// out-of-line function, so hashbrown's lookup and the key comparison
    /// are compiled into it once, fully inlined, instead of being shared
    /// as separate calls between several callers. The callers keep the
    /// nil filter, so they can tail-call it.
    #[inline(never)]
    fn probe_string(&self, key: &LuaString) -> Option<&Value> {
        self.probe(key)
    }

    /// [`Self::probe_string`] without nil values.
    #[inline(never)]
    fn probe_string_live(&self, key: &LuaString) -> Option<&Value> {
        self.probe(key).filter(|value| !matches!(value, Value::Nil))
    }

    /// The one hashed probe of a store: the value under `key` for writing, a
    /// deleted key's nil included. Every store of a string key into a hashed
    /// table that finds its key takes one of these and no other.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn slot_mut(&mut self, key: &LuaString) -> Option<&mut Value> {
        note_string_probe();
        self.values.get_mut(&StringSlotProbe(key))
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn probe(&self, key: &LuaString) -> Option<&Value> {
        note_string_probe();
        let hash = key.known_hash();
        if hash == 0 {
            // An unhashed key, or one whose hash is zero: the general
            // lookup hashes it.
            return self.probe_string_by_bytes(key);
        }
        let probe = StringProbe {
            key,
            hash,
            heap_compare: std::cell::Cell::new(false),
        };
        let found = self.values.get(&probe);
        if found.is_none() && probe.heap_compare.get() {
            return self.probe_string_by_bytes(key);
        }
        found
    }

    /// [`Self::probe_string`] for a long key that is not the stored
    /// allocation, which needs a byte comparison. Out of line so that the
    /// probe loop makes no calls and keeps its registers.
    #[cold]
    #[inline(never)]
    fn probe_string_by_bytes(&self, key: &LuaString) -> Option<&Value> {
        self.values.get(&StringTableKey(key))
    }
}

/// Two-way sets of 56 bytes. They are not line-aligned: aligning them grew
/// every process's peak allocation by 112 bytes for no fewer instructions.
const READ_CACHE_SETS: usize = 32;

/// Consecutive misses after which the cache stands aside, and for how many
/// reads. A loop over many distinct keys never hits, and checking a slot
/// before every probe would only add to its cost.
const READ_CACHE_MISS_STREAK: u32 = 32;
const READ_CACHE_REST: u32 = 512;

#[derive(Clone, Copy, Default)]
struct ReadCacheSlot {
    /// Layout number of the hashed map the value lives in; zero when empty.
    epoch: u64,
    /// Identity of the key string, which the slot's entry in
    /// [`StringReadCache::keep`] holds alive so the address names no other
    /// string.
    key: usize,
    /// Address of the value in that map's bucket.
    value: usize,
}

#[derive(Default)]
struct ReadCacheSet {
    ways: [ReadCacheSlot; 2],
    /// Keys (truncated identities, so a false match only installs a key one
    /// read early) of recent reads that missed this set and have not
    /// installed. A key takes a way only when it misses twice with no other
    /// miss replacing its entry between, so reads of many distinct keys
    /// leave the set alone instead of refilling it each time. A hit leaves
    /// these alone.
    pending: [u32; 2],
}

/// The set a key identity maps to. Strings of this size come from the
/// allocator 64 bytes apart, so the low six bits carry nothing; the bits
/// above them pick the set, and a run of consecutive allocations covers all
/// sets. (The old `>> 4` index kept two constant bits and reached a quarter
/// of its slots.) Mixing the bits with a multiply cost about 1.5% of the
/// instructions of a global-read loop and is not needed for that stride.
#[inline(always)]
fn read_cache_set(identity: usize) -> usize {
    const _: () = assert!(READ_CACHE_SETS.is_power_of_two());
    (identity >> 6) % READ_CACHE_SETS
}

/// Remembers where recent constant-string reads found their value in hashed
/// tables, so a repeat read of the same key in an unchanged table skips the
/// hash probe.
///
/// A slot is valid while the map's layout number matches: overwriting a
/// value writes the bucket in place, so the cached address reads the new
/// value, and every change that could move or free the bucket takes a new
/// layout number (see [`IndexedTableEntries::layout_changed`]). Nothing here
/// looks at metatables. Only present, non-nil values are returned, which
/// raw reads and `__index` agree on.
///
/// Sets are chosen by key address. A key installs on its second miss, so
/// when three or more hot keys share one set (as the old one-way
/// index made likely, since allocations 64 bytes apart reached only a few
/// slots) each miss overwrote another key's pending entry before it could
/// install, and the miss streak then switched the cache off. The set has two
/// ways and the index uses the bits that vary, which makes that collision rare.
pub struct StringReadCache {
    sets: [ReadCacheSet; READ_CACHE_SETS],
    keep: [[Option<LuaString>; 2]; READ_CACHE_SETS],
    /// Reads left to answer without consulting the slots.
    rest: u32,
    streak: u32,
    hits: u64,
    installs: u64,
}

impl std::fmt::Debug for StringReadCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StringReadCache")
            .field("hits", &self.hits)
            .field("installs", &self.installs)
            .finish_non_exhaustive()
    }
}

impl Default for StringReadCache {
    fn default() -> Self {
        Self {
            sets: std::array::from_fn(|_| ReadCacheSet::default()),
            keep: std::array::from_fn(|_| [None, None]),
            rest: 0,
            streak: 0,
            hits: 0,
            installs: 0,
        }
    }
}

impl StringReadCache {
    /// Reads answered from a slot without probing.
    #[must_use]
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Reads that probed a hashed table and filled a slot.
    #[must_use]
    pub fn installs(&self) -> u64 {
        self.installs
    }
}

impl Default for TableEntries {
    fn default() -> Self {
        Self::Ordered(SmallOrderedEntries::default())
    }
}

impl TableEntries {
    pub(crate) fn reserve(&mut self, capacity: usize) -> RuntimeResult<()> {
        self.materialize();
        if capacity >= HASHED_TABLE_ENTRY_THRESHOLD {
            self.promote();
        }
        match self {
            Self::Ordered(entries) => entries
                .entries
                .try_reserve(capacity.saturating_sub(entries.entries.len()))
                .map_err(|()| RuntimeError::message("not enough memory")),
            Self::Other(other) => {
                let OtherMut::Indexed(entries) = other.view_mut() else {
                    unreachable!()
                };
                entries.layout_changed();
                entries
                    .values
                    .try_reserve(capacity.saturating_sub(entries.values.len()))
                    .map_err(|_| RuntimeError::message("not enough memory"))
            }
        }
    }

    pub(crate) fn clear(&mut self) {
        match self {
            Self::Ordered(entries) => entries.entries.clear(),
            Self::Other(other) => match other.view_mut() {
                OtherMut::Indexed(entries) => {
                    entries.layout_changed();
                    entries.values.clear();
                    entries.order.take();
                    entries.live = 0;
                }
                OtherMut::Shaped(entries) => {
                    for value in entries.values_mut() {
                        *value = Value::Nil;
                    }
                }
            },
        }
    }
    /// Storage for a table whose keys are those of `shape`, all absent.
    #[inline]
    #[cfg(test)]
    pub(crate) fn with_shape(shape: Arc<TableShape>) -> Self {
        Self::Other(OtherEntries::shaped(shape))
    }

    /// [`Self::with_shape`] for a borrowed shape. A pinned shape is not
    /// counted: the table borrows the count that pinning leaked.
    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn with_shape_ref(shape: &Arc<TableShape>) -> Self {
        let copy = if shape.is_pinned() {
            // SAFETY: pinning leaked one strong count that outlives every
            // table, and `OtherEntries::drop` forgets the copy of a pinned
            // shape instead of dropping it, so the count stays balanced.
            unsafe { Arc::from_raw(Arc::as_ptr(shape)) }
        } else {
            Arc::clone(shape)
        };
        Self::Other(OtherEntries::shaped(copy))
    }

    pub(crate) fn is_shaped(&self) -> bool {
        matches!(self, Self::Other(other) if other.is_shaped())
    }

    /// The id of the shape of a shaped table and the address of its first
    /// value slot, which together name where each key's slot is.
    pub(crate) fn shaped_layout(&self) -> Option<(usize, usize)> {
        let Self::Other(other) = self else {
            return None;
        };
        match other.view() {
            OtherRef::Shaped(entries) => Some((
                usize::try_from(entries.shape().id).ok()?,
                entries.slots() as usize,
            )),
            OtherRef::Indexed(_) => None,
        }
    }

    /// Whether a hashed table holds `key` as a deleted key: a nil in its
    /// place, which [`Self::get_string`] reads as absent.
    pub(crate) fn holds_deleted_string(&self, key: &LuaString) -> bool {
        match self {
            Self::Other(other) => match other.view() {
                OtherRef::Indexed(entries) => {
                    matches!(entries.probe_string(key), Some(Value::Nil))
                }
                OtherRef::Shaped(_) => false,
            },
            Self::Ordered(_) => false,
        }
    }

    pub(crate) fn is_indexed(&self) -> bool {
        matches!(self, Self::Other(other) if !other.is_shaped())
    }

    /// Converts a shaped table to the ordered representation with owned
    /// keys, dropping absent slots. Every operation the shaped storage does
    /// not serve itself calls this first, then runs unchanged. The capacity
    /// matches what the same insertions would have left, so the table's
    /// accounted size does not move.
    #[inline]
    pub(crate) fn materialize(&mut self) {
        if self.is_shaped() {
            self.materialize_shaped();
        }
    }

    #[cold]
    #[inline(never)]
    fn materialize_shaped(&mut self) {
        let Self::Other(mut other) = std::mem::take(self) else {
            unreachable!("checked above");
        };
        let OtherMut::Shaped(shaped) = other.view_mut() else {
            unreachable!("checked above");
        };
        let mut entries = OrderedVec::with_capacity(ordered_capacity(shaped.present()));
        let shape = shaped.shape_arc().clone();
        for ((key, hash), slot) in shape
            .keys()
            .iter()
            .zip(shape.hashes())
            .zip(shaped.values_mut())
        {
            let value = std::mem::replace(slot, Value::Nil);
            if !matches!(value, Value::Nil) {
                entries.push(OrderedEntry {
                    key: key.clone(),
                    value,
                    hash: *hash,
                });
            }
        }
        *self = Self::Ordered(SmallOrderedEntries { entries });
    }

    // Not marked inline, and `is_empty` through `len`: the callers that
    // inline them (array writes) came out with extra register spills in the
    // VM's dispatch loop when these were written any other way.
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Ordered(entries) => entries.entries.len(),
            Self::Other(other) => other.len(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The value slots of a shaped table, nil ones included, in key order.
    /// Shape keys are strings, so a caller that only needs the values (the
    /// collector marking a table whose keys hold no objects) reads them
    /// without the visitor call and the key list [`Self::for_each`] goes
    /// through. `None` for any other table.
    #[inline]
    pub(crate) fn shaped_values(&self) -> Option<&[Value]> {
        match self {
            Self::Other(other) => match other.view() {
                OtherRef::Shaped(entries) => Some(entries.values()),
                OtherRef::Indexed(_) => None,
            },
            Self::Ordered(_) => None,
        }
    }

    /// Whether the entries are ordered and none are stored: a lookup of any
    /// key finds nothing, which this decides without the call a scan costs.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn is_empty_ordered(&self) -> bool {
        matches!(self, Self::Ordered(entries) if entries.entries.is_empty())
    }

    pub(crate) fn get(&self, key: &TableKey) -> Option<&Value> {
        match self {
            Self::Ordered(entries) => entries.get(key),
            Self::Other(other) => match other.view() {
                OtherRef::Indexed(entries) => entries.values.get(key),
                OtherRef::Shaped(entries) => entries.get(key),
            },
        }
    }

    /// Looks up a string key without cloning its reference-counted backing
    /// storage.
    ///
    /// Ordered tables stay below [`HASHED_TABLE_ENTRY_THRESHOLD`] entries, so
    /// a scan comparing cached hashes beats a binary search whose every probe
    /// orders the key bytes (strings are not interned).
    ///
    /// Only the representation test is inlined. Each representation has its
    /// own out-of-line leaf ([`SmallOrderedEntries::scan_string`],
    /// [`IndexedTableEntries::probe_string`]), so a caller pays one call and
    /// one prologue, and carries neither the scan nor the probe.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn get_string(&self, key: &LuaString) -> Option<&Value> {
        match self {
            Self::Ordered(entries) => entries.scan_string(key),
            Self::Other(other) => other.get_string(key),
        }
    }

    /// [`Self::get_string`] that treats a stored nil as absent.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn get_string_live(&self, key: &LuaString) -> Option<&Value> {
        match self {
            Self::Ordered(entries) => entries
                .scan_string(key)
                .filter(|value| !matches!(value, Value::Nil)),
            Self::Other(other) => other.get_live(key),
        }
    }

    /// [`Self::get_string_live`] with the ordered scan inlined into the
    /// caller. Only the VM's field reads use it: inlining the scan into
    /// every caller made stdlib and string benchmarks worse. Hashed tables
    /// call the same out-of-line probe as [`Self::get_string`]. A shaped
    /// table's scan is inlined too, which is what makes a record read as
    /// cheap as an ordered one.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn get_string_live_scan(&self, key: &LuaString) -> Option<&Value> {
        self.live_scan::<true>(key)
    }

    /// [`Self::get_string_live_scan`] without the shaped scan, for the
    /// register-keyed `GetTable`: its integer-key path shares a handler
    /// with this read, and a second inlined scan there made the array reads
    /// in that handler spill more registers than they did before shaped
    /// tables existed.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn get_string_live_scan_ordered(&self, key: &LuaString) -> Option<&Value> {
        self.live_scan::<false>(key)
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn live_scan<const SHAPED: bool>(&self, key: &LuaString) -> Option<&Value> {
        let entries = match self {
            Self::Ordered(entries) => entries,
            Self::Other(other) if SHAPED => {
                return match other.view() {
                    OtherRef::Indexed(entries) => entries
                        .probe_string(key)
                        .filter(|value| !matches!(value, Value::Nil)),
                    OtherRef::Shaped(entries) => entries.get_live_scan(key),
                };
            }
            Self::Other(other) => return other.get_live(key),
        };
        let index = entries.find_string(key, key.cached_hash())?;
        Some(&entries.entries[index].value).filter(|value| !matches!(value, Value::Nil))
    }

    /// The live value an inline-cache site finds in this table, or `None`
    /// when the site was filled for another table layout. `guard` and
    /// `data` come from [`Self::site_fill`]:
    ///
    /// - an ordered table: [`ORDERED_SITE`] less the entry's index, and the
    ///   address of the entry's key string, which the site keeps alive. The
    ///   entry at that index must hold that very string;
    /// - a shaped table: the shape's id and the slot's index;
    /// - a hashed table: its layout number and the value's address.
    ///
    /// The numbers are unique per shape and per layout, so a table that was
    /// freed and replaced, converted, grown or cleared never matches a
    /// stale site.
    #[allow(clippy::inline_always, unsafe_code)]
    #[inline(always)]
    pub(crate) fn site_value(&self, guard: u64, data: usize) -> Option<&Value> {
        let value = match self {
            Self::Ordered(entries) => {
                let entry = entries
                    .entries
                    .get(usize::try_from(ORDERED_SITE.wrapping_sub(guard)).unwrap_or(usize::MAX))?;
                let TableKey::String(stored) = &entry.key else {
                    return None;
                };
                if stored.identity() != data {
                    return None;
                }
                &entry.value
            }
            Self::Other(other) => match other.view() {
                OtherRef::Shaped(entries) => {
                    if entries.shape().id != guard {
                        return None;
                    }
                    // SAFETY: the shape is the one the site was filled for.
                    // The site pairs the shape's id with a slot index of that
                    // shape (see `FieldSite::load` for why a reader never sees
                    // the data of another fill), so `data` is below its
                    // length. A shape's length never changes.
                    debug_assert!(data < entries.shape().len());
                    unsafe { &*entries.slots().add(data) }
                }
                OtherRef::Indexed(entries) => {
                    if entries.epoch != guard {
                        return None;
                    }
                    // SAFETY: the layout number is unchanged, so the map is
                    // the one that filled the site, with its buckets where
                    // they were. The address stays valid while `self` is
                    // borrowed.
                    unsafe { &*(data as *const Value) }
                }
            },
        };
        (!matches!(value, Value::Nil)).then_some(value)
    }

    /// The slot [`Self::site_value`] finds, for a store, nil or not. A nil slot of a shaped table is a
    /// field of its shape that it lacks; in the other representations it
    /// is a deleted key.
    #[allow(unsafe_code)]
    ///
    /// The flag is whether the table is shaped, so a store that finds a nil
    /// slot need not look again.
    #[inline(always)]
    pub(crate) fn site_slot_any_mut(
        &mut self,
        guard: u64,
        data: usize,
    ) -> Option<(&mut Value, bool)> {
        let mut shaped = false;
        let slot = match self {
            Self::Ordered(entries) => {
                let entry = entries.entries.get_mut(
                    usize::try_from(ORDERED_SITE.wrapping_sub(guard)).unwrap_or(usize::MAX),
                )?;
                let TableKey::String(stored) = &entry.key else {
                    return None;
                };
                if stored.identity() != data {
                    return None;
                }
                &mut entry.value
            }
            Self::Other(other) => match other.view_mut() {
                OtherMut::Shaped(entries) => {
                    if entries.shape().id != guard {
                        return None;
                    }
                    // A site pairs a shape's unique id with a slot index
                    // that `site_fill` took from that same shape, and a
                    // shape never changes, so the index is in range
                    // whenever the id matches (as in `site_value`).
                    debug_assert!(data < entries.shape().len());
                    shaped = true;
                    entries.slot_mut(data)
                }
                OtherMut::Indexed(entries) => {
                    if entries.epoch != guard {
                        return None;
                    }
                    // SAFETY: as in `site_value`; `&mut self` is the only
                    // access to the map.
                    unsafe { &mut *(data as *mut Value) }
                }
            },
        };
        Some((slot, shaped))
    }

    /// The slot a store site filled for a shaped table finds in this one,
    /// whatever it holds: a shape keeps a slot for each of its keys, present
    /// or not, so a store through the site may add the field. `None` for a
    /// table that is not of the site's shape.
    #[inline]
    pub(crate) fn shaped_site_slot_mut(&mut self, guard: u64, data: usize) -> Option<&mut Value> {
        let Self::Other(other) = self else {
            return None;
        };
        let OtherMut::Shaped(entries) = other.view_mut() else {
            return None;
        };
        if entries.shape().id != guard || data >= entries.shape().len() {
            return None;
        }
        Some(entries.slot_mut(data))
    }

    /// What makes [`Self::site_value`] find the live value under `key` in
    /// this table, if it holds one.
    pub(crate) fn site_fill(&self, key: &LuaString) -> Option<SiteFill> {
        match self {
            Self::Ordered(entries) => {
                let index = entries.find_string(key, key.cached_hash())?;
                let entry = &entries.entries[index];
                let TableKey::String(stored) = &entry.key else {
                    return None;
                };
                (!matches!(entry.value, Value::Nil)).then(|| SiteFill {
                    guard: ORDERED_SITE - u64::try_from(index).unwrap_or(u64::MAX),
                    data: stored.identity(),
                    pin: Some(stored.clone()),
                })
            }
            Self::Other(other) => match other.view() {
                OtherRef::Shaped(entries) => {
                    let index = entries.shape().find_string(key, key.cached_hash())?;
                    (!matches!(entries.values()[index], Value::Nil)).then(|| SiteFill {
                        guard: entries.shape().id,
                        data: index,
                        pin: None,
                    })
                }
                OtherRef::Indexed(entries) => {
                    let value = entries.probe_string_live(key)?;
                    Some(SiteFill {
                        guard: entries.epoch,
                        data: std::ptr::from_ref(value) as usize,
                        pin: None,
                    })
                }
            },
        }
    }

    /// [`Self::site_fill`] for a site that stores. The address of a hashed
    /// table's value comes from a mutable borrow.
    pub(crate) fn site_fill_mut(&mut self, key: &LuaString) -> Option<SiteFill> {
        if let Self::Other(other) = self
            && let OtherMut::Indexed(entries) = other.view_mut()
        {
            let epoch = entries.epoch;
            let value = entries.values.get_mut(&StringTableKey(key))?;
            return (!matches!(value, Value::Nil)).then(|| SiteFill {
                guard: epoch,
                data: std::ptr::from_mut(value) as usize,
                pin: None,
            });
        }
        self.site_fill(key)
    }

    /// [`Self::get_string_live_scan`] for a key that is read repeatedly: a
    /// hashed table answers from `cache` while its layout is unchanged.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn get_string_live_cached(
        &self,
        key: &LuaString,
        cache: &mut StringReadCache,
    ) -> Option<&Value> {
        let Self::Other(other) = self else {
            return self.get_string_live_scan(key);
        };
        let OtherRef::Indexed(entries) = other.view() else {
            return self.get_string_live_scan(key);
        };
        if cache.rest != 0 {
            cache.rest -= 1;
            return entries
                .probe_string(key)
                .filter(|value| !matches!(value, Value::Nil));
        }
        let identity = key.identity();
        let set = &mut cache.sets[read_cache_set(identity)];
        for way in &set.ways {
            if way.epoch == entries.epoch && way.key == identity {
                cache.hits += 1;
                cache.streak = 0;
                #[allow(unsafe_code)]
                // SAFETY: the layout number is unchanged, so the map is the one
                // that filled the slot, with its buckets where they were. The
                // address stays valid for as long as `self` is borrowed.
                let value = unsafe { &*(way.value as *const Value) };
                return Some(value).filter(|value| !matches!(value, Value::Nil));
            }
        }
        Self::miss_read_cache(entries, key, cache)
    }

    #[inline(never)]
    fn miss_read_cache<'a>(
        entries: &'a IndexedTableEntries,
        key: &LuaString,
        cache: &mut StringReadCache,
    ) -> Option<&'a Value> {
        let value = entries.probe_string(key);
        cache.streak += 1;
        if cache.streak >= READ_CACHE_MISS_STREAK {
            cache.streak = 0;
            cache.rest = READ_CACHE_REST;
        }
        let value = value?;
        let identity = key.identity();
        let index = read_cache_set(identity);
        let set = &mut cache.sets[index];
        #[allow(clippy::cast_possible_truncation)]
        let tag = identity as u32;
        let Some(pending) = set.pending.iter().position(|&pending| pending == tag) else {
            // First miss of this key: record it in a free pending entry, else
            // replace the second.
            let free = usize::from(set.pending[0] != 0);
            set.pending[free] = tag;
            return Some(value).filter(|value| !matches!(value, Value::Nil));
        };
        cache.installs += 1;
        set.pending[pending] = 0;
        // A key already in a way is refreshed there. A new key takes way 0 and
        // moves way 0's key to way 1, so a hit never writes to the set.
        let way = if set.ways[1].key == identity {
            1
        } else {
            if set.ways[0].key != identity {
                set.ways[1] = set.ways[0];
                cache.keep[index].swap(0, 1);
                set.ways[0].key = identity;
                cache.keep[index][0] = Some(key.clone());
            }
            0
        };
        let slot = &mut set.ways[way];
        slot.epoch = entries.epoch;
        slot.value = std::ptr::from_ref(value) as usize;
        Some(value).filter(|value| !matches!(value, Value::Nil))
    }

    /// Stores a non-nil `value` under a string key of a small ordered map.
    /// Overwrites need only the cached-hash scan; a new key then takes one
    /// ordered scan for its insertion point, without building a lookup key
    /// or binary-searching with full key compares. A shaped table takes the
    /// value in the slot of its key, which counts as new when the slot was
    /// absent. Returns whether the key is new, or hands `value` back when
    /// the map is hashed or not small.
    #[inline]
    pub(crate) fn set_small_string(
        &mut self,
        key: &LuaString,
        value: Value,
    ) -> Result<bool, Value> {
        let entries = match self {
            Self::Ordered(entries) => entries,
            Self::Other(other) => {
                let OtherMut::Shaped(entries) = other.view_mut() else {
                    return Err(value);
                };
                let Some(index) = entries.shape().find_string(key, key.cached_hash()) else {
                    return self.set_outside_shape(key, value);
                };
                let slot = entries.slot_mut(index);
                let was_absent = matches!(slot, Value::Nil);
                *slot = value;
                return Ok(was_absent);
            }
        };
        if entries.entries.len() >= 8 {
            return Err(value);
        }
        let hash = key.cached_hash();
        if let Some(index) = entries.find_string(key, hash) {
            entries.entries[index].value = value;
            return Ok(false);
        }
        // Key order ranks Boolean, Integer, and Number below String and
        // every other variant above it.
        let index = entries
            .entries
            .iter()
            .position(|entry| match &entry.key {
                TableKey::Boolean(_) | TableKey::Integer(_) | TableKey::Number(_) => false,
                TableKey::String(stored) => stored > key,
                _ => true,
            })
            .unwrap_or(entries.entries.len());
        insert_ordered(
            &mut entries.entries,
            index,
            OrderedEntry::new(TableKey::string(key), value),
        );
        Ok(true)
    }

    /// The slot a shaped table's shape holds for `key`. The constructor's
    /// field stores write through it in one step, where
    /// [`Self::set_small_string`] also covers the ordered and the
    /// outside-the-shape cases. Handing out the slot, not taking the value,
    /// keeps the miss on other tables free of moving it.
    #[inline]
    pub(crate) fn shaped_slot_mut(&mut self, key: &LuaString) -> Option<&mut Value> {
        let Self::Other(other) = self else {
            return None;
        };
        let OtherMut::Shaped(entries) = other.view_mut() else {
            return None;
        };
        let index = entries.shape().find_string(key, key.cached_hash())?;
        Some(entries.slot_mut(index))
    }

    /// A string store to a shaped table whose shape lacks the key converts
    /// the table, then stores as an ordered table does.
    #[cold]
    #[inline(never)]
    fn set_outside_shape(&mut self, key: &LuaString, value: Value) -> Result<bool, Value> {
        self.materialize();
        self.set_small_string(key, value)
    }

    /// Mutable string lookup that never clones the key, scanning ordered
    /// tables as [`Self::get_string`] does.
    pub(crate) fn get_string_mut(&mut self, key: &LuaString) -> Option<&mut Value> {
        match self {
            Self::Ordered(entries) => {
                let hash = key.cached_hash();
                let index = entries.find_string(key, hash)?;
                Some(&mut entries.entries[index].value)
            }
            Self::Other(other) => match other.view_mut() {
                // A tombstone is no slot: a write through it would bypass `live`.
                OtherMut::Indexed(entries) => entries
                    .slot_mut(key)
                    .filter(|value| !matches!(value, Value::Nil)),
                OtherMut::Shaped(entries) => entries.get_string_mut(key),
            },
        }
    }

    /// The address of the first value slot of a shaped table.
    pub(crate) fn shaped_slots(&mut self) -> Option<usize> {
        let Self::Other(other) = self else {
            return None;
        };
        if !other.is_shaped() {
            return None;
        }
        let OtherMut::Shaped(shaped) = other.view_mut() else {
            return None;
        };
        Some(shaped.slots() as usize)
    }

    /// Stores `value` under the string `key` of a shaped table whose shape
    /// names it, in the slot that holds nil. `false`, having stored nothing,
    /// for any other table or key.
    pub(crate) fn init_shaped_slot(&mut self, key: &LuaString, value: &Value) -> bool {
        let Self::Other(other) = self else {
            return false;
        };
        if !other.is_shaped() {
            return false;
        }
        let OtherMut::Shaped(shaped) = other.view_mut() else {
            return false;
        };
        shaped.init_slot(key, value)
    }

    /// For each `(key, kind)` of `wanted`, which holds a string, writes to the
    /// same position of `out` the address of the value stored under that
    /// key if it has `kind`, and zero otherwise. The addresses follow the
    /// contract of [`Self::value_ptr`].
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn string_value_ptrs(
        &mut self,
        wanted: &[(Value, NativeSlotKind)],
        hints: &mut [usize],
        out: &mut [usize],
        absent: Option<usize>,
    ) {
        // A shaped table keeps its values in slots the shape names, which
        // are as good as entries: reading them leaves the table shaped.
        if let Self::Other(other) = self
            && other.is_shaped()
            && let OtherMut::Shaped(shaped) = other.view_mut()
        {
            shaped.string_value_ptrs(wanted, hints, out, absent);
            return;
        }
        self.string_value_ptrs_unshaped(wanted, hints, out, absent);
    }

    /// Whether `slot`, an address [`Self::string_value_ptrs`] gave for the
    /// string `key` some time ago, is the address of the value stored under
    /// `key` now: the entries as they stand hold `key` at the place the
    /// address names. This reads the entries alone, never the memory at
    /// `slot`, which may have been freed since. It does not look at the
    /// value, which may have changed type or become nil.
    ///
    /// A hashed table answers `false`: it keeps its values in buckets that
    /// have no place to check.
    pub(crate) fn holds_string_slot(&self, slot: usize, key: &LuaString) -> bool {
        match self {
            Self::Ordered(entries) => entries.holds_string_slot(slot, key),
            Self::Other(other) => match other.view() {
                OtherRef::Shaped(entries) => entries.holds_string_slot(slot, key),
                OtherRef::Indexed(_) => false,
            },
        }
    }

    /// [`Self::string_value_ptrs`] for a table that is not shaped.
    #[inline(never)]
    fn string_value_ptrs_unshaped(
        &mut self,
        wanted: &[(Value, NativeSlotKind)],
        hints: &mut [usize],
        out: &mut [usize],
        absent: Option<usize>,
    ) {
        out.fill(0);
        // A table with no hash entries (an array, say) holds none of the
        // keys: only a wanted `Absent` has an answer, and it is the marker.
        if let Self::Ordered(entries) = &*self
            && entries.entries.is_empty()
        {
            if let Some(marker) = absent {
                for ((key, kind), found) in wanted.iter().zip(out.iter_mut()) {
                    if *kind == NativeSlotKind::Absent && matches!(key, Value::String(_)) {
                        *found = marker;
                    }
                }
            }
            return;
        }
        self.materialize();
        match self {
            Self::Ordered(entries) => {
                entries.string_value_ptrs(wanted, hints, out);
            }
            Self::Other(other) => {
                let OtherMut::Indexed(entries) = other.view_mut() else {
                    unreachable!("materialized above");
                };
                for ((key, kind), found) in wanted.iter().zip(out.iter_mut()) {
                    let Value::String(key) = key else {
                        continue;
                    };
                    let held = entries
                        .values
                        .get_mut(&StringTableKey(key))
                        .filter(|value| !matches!(value, Value::Nil));
                    match held {
                        Some(value) if kind.holds(value) => {
                            *found = std::ptr::from_mut(value) as usize;
                        }
                        Some(_) => {}
                        None => {
                            if let (NativeSlotKind::Absent, Some(marker)) = (kind, absent) {
                                *found = marker;
                            }
                        }
                    }
                }
            }
        }
        // A key an ordered table lacks.
        if let (Self::Ordered(entries), Some(marker)) = (&*self, absent) {
            for ((key, kind), found) in wanted.iter().zip(out.iter_mut()) {
                if let (NativeSlotKind::Absent, Value::String(key), 0) = (kind, key, *found)
                    && entries.find_string(key, key.cached_hash()).is_none()
                {
                    *found = marker;
                }
            }
        }
    }

    /// Address of the value under `key` for native code that holds several
    /// such addresses at once (see [`SmallOrderedEntries::value_ptr`]).
    pub(crate) fn value_ptr(&mut self, key: &TableKey) -> Option<*mut Value> {
        self.materialize();
        match self {
            Self::Ordered(entries) => entries.value_ptr(key),
            Self::Other(other) => {
                let OtherMut::Indexed(entries) = other.view_mut() else {
                    unreachable!("materialized above");
                };
                entries
                    .values
                    .get_mut(key)
                    .filter(|value| !matches!(value, Value::Nil))
                    .map(std::ptr::from_mut)
            }
        }
    }

    pub(crate) fn contains_key(&self, key: &TableKey) -> bool {
        self.get(key).is_some()
    }

    #[inline]
    pub(crate) fn insert(&mut self, key: TableKey, value: Value) -> Option<Value> {
        if let Self::Ordered(entries) = self {
            return entries.insert(key, value);
        }
        self.insert_other(key, value)
    }

    #[cold]
    #[inline(never)]
    fn insert_other(&mut self, key: TableKey, value: Value) -> Option<Value> {
        if let Self::Other(other) = self
            && let OtherMut::Shaped(entries) = other.view_mut()
        {
            // A non-nil store to a key of the shape fills or overwrites its
            // slot; an absent slot reports no previous value, as a new key.
            if !matches!(value, Value::Nil)
                && let Ok(index) = entries.shape().position(&key)
            {
                let old = std::mem::replace(&mut entries.values_mut()[index], value);
                return Some(old).filter(|old| !matches!(old, Value::Nil));
            }
            self.materialize();
        }
        match self {
            Self::Ordered(entries) => entries.insert(key, value),
            Self::Other(other) => {
                let OtherMut::Indexed(entries) = other.view_mut() else {
                    unreachable!("materialized above");
                };
                if entries.dead() > entries.live && !entries.values.contains_key(&key) {
                    entries.compact();
                }
                let is_nil = matches!(value, Value::Nil);
                match entries.values.entry(key) {
                    hashbrown::hash_map::Entry::Occupied(mut entry) => {
                        // A tombstone that gets a value is an insertion.
                        let was_nil = matches!(entry.get(), Value::Nil);
                        let old = entry.insert(value);
                        match (was_nil, is_nil) {
                            (true, false) => entries.live += 1,
                            (false, true) => entries.live -= 1,
                            _ => {}
                        }
                        (!was_nil).then_some(old)
                    }
                    hashbrown::hash_map::Entry::Vacant(entry) => {
                        if let Some(order) = entries.order.get_mut() {
                            order.insert(entry.key().clone());
                        }
                        entry.insert(value);
                        if !is_nil {
                            entries.live += 1;
                        }
                        entries.layout_changed();
                        None
                    }
                }
            }
        }
    }

    pub(crate) fn promote(&mut self) {
        self.materialize();
        let mut entries = match std::mem::take(self) {
            Self::Ordered(entries) => entries,
            other @ Self::Other(_) => {
                *self = other;
                return;
            }
        };
        *self = Self::Other(OtherEntries::indexed(IndexedTableEntries::new(
            entries
                .entries
                .take_all()
                .into_iter()
                .map(|entry| (entry.key, entry.value))
                .collect(),
        )));
    }

    #[inline]
    pub(crate) fn remove(&mut self, key: &TableKey) -> Option<Value> {
        if let Self::Ordered(entries) = self {
            return entries.remove(key);
        }
        self.remove_other(key)
    }

    #[cold]
    #[inline(never)]
    fn remove_other(&mut self, key: &TableKey) -> Option<Value> {
        self.materialize();
        match self {
            Self::Ordered(entries) => entries.remove(key),
            Self::Other(other) => {
                let OtherMut::Indexed(entries) = other.view_mut() else {
                    unreachable!("materialized above");
                };
                let value = entries.values.remove(key);
                if value.is_some() {
                    if let Some(order) = entries.order.get_mut() {
                        order.remove(key);
                    }
                    entries.layout_changed();
                }
                value
                    .filter(|value| !matches!(value, Value::Nil))
                    .inspect(|_| {
                        entries.live -= 1;
                    })
            }
        }
    }

    /// Deletes the string `key` of a hashed table by leaving a nil in its
    /// place. `None` when the table is not hashed, so the caller removes the
    /// key the general way. Otherwise `Some(())` with the key absent or
    /// tombstoned, and no ordered set touched.
    #[inline]
    pub(crate) fn tombstone_string(&mut self, key: &LuaString) -> Option<()> {
        let Self::Other(other) = self else {
            return None;
        };
        let OtherMut::Indexed(entries) = other.view_mut() else {
            return None;
        };
        note_string_probe();
        if let Some(slot) = entries.values.get_mut(&StringSlotProbe(key))
            && !matches!(slot, Value::Nil)
        {
            // The old value is dropped here, as `remove` would drop it.
            *slot = Value::Nil;
            entries.live -= 1;
        }
        Some(())
    }

    /// [`Self::tombstone_string`] for a key that holds a value, in one probe,
    /// which says why it did not delete: the key a hashed table lacks or holds as a deleted key is
    /// [`Tombstoned::Absent`], and a table that is not hashed is
    /// [`Tombstoned::Unhashed`], which the caller reads itself. One probe.
    #[inline]
    pub(crate) fn tombstone_live_string(&mut self, key: &LuaString) -> Tombstoned {
        let Self::Other(other) = self else {
            return Tombstoned::Unhashed;
        };
        let OtherMut::Indexed(entries) = other.view_mut() else {
            return Tombstoned::Unhashed;
        };
        note_string_probe();
        match entries.values.get_mut(&StringSlotProbe(key)) {
            Some(slot) if !matches!(slot, Value::Nil) => {
                *slot = Value::Nil;
                entries.live -= 1;
                Tombstoned::Deleted
            }
            _ => Tombstoned::Absent,
        }
    }

    /// The slot of the string `key` of a hashed table, a deleted key's nil
    /// included, with the counter of live entries beside it, in one probe.
    /// `None` when the table is not hashed or lacks the key.
    #[inline]
    pub(crate) fn indexed_slot_live(
        &mut self,
        key: &LuaString,
    ) -> Option<(&mut Value, &mut usize)> {
        let Self::Other(other) = self else {
            return None;
        };
        let OtherMut::Indexed(entries) = other.view_mut() else {
            return None;
        };
        note_string_probe();
        let slot = entries.values.get_mut(&StringSlotProbe(key))?;
        Some((slot, &mut entries.live))
    }

    /// Stores a non-nil `value` in the tombstone of the string `key` of a
    /// hashed table, which is an insertion that allocates nothing. Hands
    /// `value` back when the key has no tombstone there.
    #[inline]
    pub(crate) fn revive_string(&mut self, key: &LuaString, value: Value) -> Result<(), Value> {
        let Self::Other(other) = self else {
            return Err(value);
        };
        let OtherMut::Indexed(entries) = other.view_mut() else {
            return Err(value);
        };
        note_string_probe();
        match entries.values.get_mut(&StringSlotProbe(key)) {
            Some(slot) if matches!(slot, Value::Nil) => {
                *slot = value;
                entries.live += 1;
                Ok(())
            }
            _ => Err(value),
        }
    }

    /// Moves every integer-keyed entry in `start..=end` out of the map,
    /// returning `(array_index, value)` pairs for array absorption.
    pub(crate) fn extract_integer_range(
        &mut self,
        start: usize,
        end: usize,
    ) -> Vec<(usize, Value)> {
        match self {
            Self::Ordered(entries) => {
                // Range bounds mirror the key ordering of the ordered map
                // representation: integer keys sort contiguously, so the
                // insertion positions of the bounds are the scan window.
                let lower = entries
                    .position(&TableKey::Integer(start as i64))
                    .unwrap_or_else(|index| index);
                let upper = entries
                    .position(&TableKey::Integer(end as i64))
                    .map_or_else(|index| index, |index| index + 1);
                let mut extracted = Vec::new();
                for index in (lower..upper).rev() {
                    let OrderedEntry { key, value, .. } = entries.entries.remove(index);
                    if let TableKey::Integer(integer) = key {
                        if let Ok(index) = usize::try_from(integer) {
                            extracted.push((index, value));
                        }
                    }
                }
                extracted
            }
            Self::Other(other) => {
                let OtherMut::Indexed(entries) = other.view_mut() else {
                    // A shape holds string keys only.
                    return Vec::new();
                };
                // Drain matching keys in a single traversal. Collecting
                // them first would rehash every key on removal; the
                // order set still needs one removal per absorbed key.
                let mut extracted = Vec::new();
                entries.layout_changed();
                entries
                    .values
                    .extract_if(|key, _| {
                        matches!(key, TableKey::Integer(index)
                            if usize::try_from(*index).is_ok_and(|index| (start..=end).contains(&index)))
                    })
                    .for_each(|(key, value)| {
                        let TableKey::Integer(index) = key else {
                            unreachable!("extraction filters integer keys");
                        };
                        if let Some(order) = entries.order.get_mut() {
                            order.remove(&key);
                        }
                        if !matches!(value, Value::Nil) {
                            entries.live -= 1;
                        }
                        if let Ok(index) = usize::try_from(index) {
                            extracted.push((index, value));
                        }
                    });
                extracted
            }
        }
    }

    /// Visits every entry in unspecified order. The collector uses this to
    /// trace a whole entry part without per-entry lookups.
    #[inline]
    pub(crate) fn for_each(&self, mut visit: impl FnMut(&TableKey, &Value)) {
        match self {
            Self::Ordered(entries) => {
                for entry in entries.entries.iter() {
                    visit(&entry.key, &entry.value);
                }
            }
            Self::Other(other) => match other.view() {
                OtherRef::Indexed(entries) => {
                    for (key, value) in &entries.values {
                        if !matches!(value, Value::Nil) {
                            visit(key, value);
                        }
                    }
                }
                OtherRef::Shaped(entries) => entries.for_each(visit),
            },
        }
    }

    /// Visits at most `limit` entries in key order, starting after `after`.
    /// Returns the last visited key when entries may remain, or `None` once
    /// the traversal reached the end.
    pub(crate) fn for_each_after(
        &self,
        after: Option<&TableKey>,
        limit: usize,
        mut visit: impl FnMut(&TableKey, &Value),
    ) -> Option<TableKey> {
        use std::ops::Bound::{Excluded, Unbounded};
        let mut last: Option<&TableKey> = None;
        let mut visited = 0;
        match self {
            Self::Ordered(entries) => {
                let start = match after {
                    Some(after) => entries
                        .position(after)
                        .map_or_else(|index| index, |index| index + 1),
                    None => 0,
                };
                for entry in entries.entries.get(start..).unwrap_or_default() {
                    if visited == limit {
                        return last.cloned();
                    }
                    visit(&entry.key, &entry.value);
                    last = Some(&entry.key);
                    visited += 1;
                }
            }
            Self::Other(other) => {
                let entries = match other.view() {
                    OtherRef::Indexed(entries) => entries,
                    OtherRef::Shaped(entries) => {
                        return entries.for_each_after(after, limit, visit);
                    }
                };
                for key in entries
                    .order()
                    .range((after.map_or(Unbounded, Excluded), Unbounded))
                {
                    let Some(value) = entries.values.get(key) else {
                        continue;
                    };
                    if matches!(value, Value::Nil) {
                        continue;
                    }
                    if visited == limit {
                        return last.cloned();
                    }
                    visit(key, value);
                    last = Some(key);
                    visited += 1;
                }
            }
        }
        None
    }

    /// [`IndexedTableEntries::native_cursor`] of a hashed table, and for a
    /// small table the snapshot [`SmallOrderedEntries::native_cursor`]
    /// builds; `None` for the representation that shapes keys.
    pub(crate) fn native_cursor(
        &mut self,
        previous: &Value,
        scratch: &mut NativeCursorScratch,
    ) -> Option<(usize, usize, usize)> {
        match self {
            Self::Ordered(entries) => entries.native_cursor(previous, scratch),
            Self::Other(other) => match other.view_mut() {
                OtherMut::Indexed(entries) => entries.native_cursor(),
                OtherMut::Shaped(_) => None,
            },
        }
    }

    /// The string index of these entries for native code (see
    /// [`oximond::runtime::Runtime::table_native_probe`]): a hashed table's own, and for
    /// a small ordered or a shaped table one built in `scratch`. `None` when a
    /// big hashed table has not asked often enough to earn one.
    pub(crate) fn native_probe(
        &mut self,
        scratch: &mut NativeStringProbeScratch,
        eager: bool,
    ) -> Option<NativeStringProbe> {
        match self {
            Self::Ordered(entries) => Some(fill_probe(
                scratch.next(),
                entries.entries.len(),
                entries.entries.iter().filter_map(|entry| match &entry.key {
                    TableKey::String(string) => Some((string, std::ptr::from_ref(&entry.value))),
                    _ => None,
                }),
            )),
            Self::Other(other) => match other.view_mut() {
                OtherMut::Indexed(entries) => entries.native_probe(eager),
                OtherMut::Shaped(entries) => {
                    let shape = entries.shape();
                    let keys =
                        shape
                            .keys()
                            .iter()
                            .zip(entries.values())
                            .filter_map(|(key, value)| match key {
                                TableKey::String(string) => {
                                    Some((string, std::ptr::from_ref(value)))
                                }
                                _ => None,
                            });
                    Some(fill_probe(scratch.next(), shape.len(), keys))
                }
            },
        }
    }

    /// The entry after the key `previous` names (the first without one), in
    /// key order, without cloning keys or searching twice. Anything unusual
    /// (an invalid or deleted continuation key, shaped storage) is
    /// [`HashNext::Slow`], for the caller's general lookup to settle.
    ///
    /// `hint` is the position of the entry the last step returned, from
    /// any table: a string key found at that position by pointer identity
    /// is that entry (keys are unique), and anything else searches. A
    /// returned ordered or shaped entry leaves its position there.
    pub(crate) fn next_after_value(
        &mut self,
        previous: Option<&Value>,
        hint: &mut usize,
    ) -> HashNext<'_> {
        match self {
            Self::Ordered(entries) => {
                let start = match previous {
                    None => 0,
                    // A traversal hands back the table's own key, so the
                    // hint, or else the inline hashes and pointer identity,
                    // find it without building a key or comparing bytes in a
                    // binary search.
                    Some(Value::String(key)) => match entries.entries.get(*hint) {
                        Some(entry)
                            if matches!(&entry.key, TableKey::String(stored)
                                if stored.ptr_eq(key)) =>
                        {
                            *hint + 1
                        }
                        _ => match entries.find_string(key, key.cached_hash()) {
                            Some(found) => found + 1,
                            None => return HashNext::Slow,
                        },
                    },
                    Some(previous) => {
                        let Ok(key) = TableKey::try_from(previous) else {
                            return HashNext::Slow;
                        };
                        match entries.position(&key) {
                            Ok(found) => found + 1,
                            Err(_) => return HashNext::Slow,
                        }
                    }
                };
                match entries.entries.get(start) {
                    Some(entry) => {
                        *hint = start;
                        HashNext::Entry(&entry.key, &entry.value)
                    }
                    None => HashNext::End,
                }
            }
            Self::Other(other) => Self::other_next_after(other, previous, hint),
        }
    }

    /// [`Self::next_after_value`] for the hashed and shaped representations.
    /// A hashed table keeps its own cursor and leaves `hint` alone.
    fn other_next_after<'a>(
        other: &'a mut OtherEntries,
        previous: Option<&Value>,
        hint: &mut usize,
    ) -> HashNext<'a> {
        match other.view_mut() {
            OtherMut::Indexed(entries) => entries.next_after(previous),
            OtherMut::Shaped(entries) => entries.next_after(previous, hint),
        }
    }

    pub(crate) fn first_after(&self, after: Option<&TableKey>) -> Option<(&TableKey, &Value)> {
        use std::ops::Bound::{Excluded, Unbounded};
        match self {
            Self::Ordered(entries) => {
                let start = match after {
                    Some(after) => entries
                        .position(after)
                        .map_or_else(|index| index, |index| index + 1),
                    None => 0,
                };
                entries
                    .entries
                    .get(start..)?
                    .first()
                    .map(|entry| (&entry.key, &entry.value))
            }
            Self::Other(other) => match other.view() {
                OtherRef::Indexed(entries) => entries
                    .order()
                    .range((after.map_or(Unbounded, Excluded), Unbounded))
                    .find_map(|key| {
                        let value = entries.values.get(key)?;
                        (!matches!(value, Value::Nil)).then_some((key, value))
                    }),
                OtherRef::Shaped(entries) => entries.first_after(after),
            },
        }
    }

    #[inline]
    pub(crate) fn pop_first(&mut self) -> Option<(TableKey, Value)> {
        if let Self::Ordered(entries) = self {
            if entries.entries.is_empty() {
                return None;
            }
            let entry = entries.entries.remove(0);
            return Some((entry.key, entry.value));
        }
        self.pop_first_other()
    }

    #[cold]
    #[inline(never)]
    fn pop_first_other(&mut self) -> Option<(TableKey, Value)> {
        self.materialize();
        match self {
            Self::Ordered(entries) if entries.entries.is_empty() => None,
            Self::Ordered(entries) => {
                let entry = entries.entries.remove(0);
                Some((entry.key, entry.value))
            }
            Self::Other(other) => {
                let OtherMut::Indexed(entries) = other.view_mut() else {
                    unreachable!("materialized above");
                };
                loop {
                    let key = entries.order_mut().pop_first()?;
                    entries.layout_changed();
                    let value = entries
                        .values
                        .remove(&key)
                        .expect("ordered table key has a matching hashed value");
                    if !matches!(value, Value::Nil) {
                        entries.live -= 1;
                        return Some((key, value));
                    }
                }
            }
        }
    }

    pub(crate) fn allocated_bytes(&self) -> usize {
        match self {
            Self::Ordered(entries) => {
                entries.entries.capacity() * std::mem::size_of::<OrderedEntry>()
                    + std::mem::size_of::<OrderedVec>()
            }
            // What the ordered table holding the same entries reports, so a
            // table's accounted size does not depend on its representation.
            Self::Other(other) => match other.view() {
                OtherRef::Shaped(entries) => {
                    ordered_capacity(entries.present()) * std::mem::size_of::<OrderedEntry>()
                        + std::mem::size_of::<OrderedVec>()
                }
                OtherRef::Indexed(entries) => {
                    // Indexed tables own hash buckets for values and a separate
                    // ordered key set. There is no B-tree key/value payload after
                    // promotion, so account for each representation exactly once.
                    std::mem::size_of::<IndexedTableEntries>()
                        + entries.values.capacity()
                            * (std::mem::size_of::<TableKey>()
                                + std::mem::size_of::<Value>()
                                + std::mem::size_of::<usize>())
                        + entries.values.len()
                            * (std::mem::size_of::<TableKey>() + 3 * std::mem::size_of::<usize>())
                }
            },
        }
    }
}

/// The shapes a caller's hints can remember (see [`shape_cache_len`]).
pub const SHAPE_CACHE_LINES: usize = 4;

/// How many hints a caller of [`oximond::runtime::Runtime::table_native_value_slots`]
/// passes for `keys` keys to let tables of up to [`SHAPE_CACHE_LINES`]
/// shapes skip the key search: one per key for tables that are not shaped, a
/// counter, and per shape a shape id and a slot per key.
#[must_use]
pub const fn shape_cache_len(keys: usize) -> usize {
    keys + 1 + SHAPE_CACHE_LINES * (keys + 1)
}

/// Shape identity for operation statistics.
#[cfg(feature = "op-stats")]
impl TableEntries {
    /// The address of this table's shape, or 0 when it has none. Reads only
    /// the entries' tag and the shape pointer.
    pub(crate) fn shape_identity(&self) -> u64 {
        match self {
            Self::Other(other) => match other.view() {
                OtherRef::Shaped(entries) => Arc::as_ptr(entries.shape_arc()) as usize as u64,
                OtherRef::Indexed(_) => 0,
            },
            Self::Ordered(_) => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn string_key(text: &str) -> TableKey {
        TableKey::string(&LuaString::from(text))
    }

    fn order(entries: &TableEntries) -> Vec<String> {
        let mut out = Vec::new();
        let mut after: Option<TableKey> = None;
        while let Some((key, _)) = entries.first_after(after.as_ref()) {
            if let TableKey::String(text) = key {
                out.push(String::from_utf8_lossy(text.as_bytes()).into_owned());
            }
            after = Some(key.clone());
        }
        out
    }

    /// The rare representations hide behind one pointer, so the enum is no
    /// larger than the ordered vector and the inlined table reads test it
    /// as they did before shaped tables existed.
    #[test]
    fn rare_representations_add_no_size() {
        assert_eq!(
            std::mem::size_of::<TableEntries>(),
            std::mem::size_of::<SmallOrderedEntries>()
        );
        assert_eq!(
            std::mem::size_of::<OtherEntries>(),
            std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn next_order_survives_delete_and_reinsert() {
        let mut entries = TableEntries::default();
        for name in ["delta", "alpha", "echo", "charlie", "bravo", "foxtrot"] {
            entries.insert(string_key(name), Value::Integer(1));
        }
        assert_eq!(
            order(&entries),
            ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
        );
        entries.remove(&string_key("charlie"));
        entries.remove(&string_key("alpha"));
        assert_eq!(order(&entries), ["bravo", "delta", "echo", "foxtrot"]);
        entries.insert(string_key("charlie"), Value::Integer(2));
        entries.insert(string_key("aardvark"), Value::Integer(3));
        entries.insert(string_key("echo"), Value::Integer(4));
        assert_eq!(
            order(&entries),
            ["aardvark", "bravo", "charlie", "delta", "echo", "foxtrot"]
        );
        let echo = LuaString::from("echo");
        assert_eq!(entries.get_string(&echo), Some(&Value::Integer(4)));
        assert!(entries.get_string(&LuaString::from("alpha")).is_none());
    }

    #[test]
    fn key_stored_before_hashing_is_still_found() {
        let mut entries = SmallOrderedEntries::default();
        // Built directly, so nothing has hashed the string yet.
        let key = TableKey::String(LuaString::from("unhashed"));
        let at = entries.position(&key).unwrap_err();
        entries
            .entries
            .insert(at, OrderedEntry::new(key, Value::Integer(9)));
        let lookup = LuaString::from("unhashed");
        let index = entries.find_string(&lookup, lookup.cached_hash()).unwrap();
        assert_eq!(entries.entries[index].value, Value::Integer(9));
    }

    #[test]
    fn string_reads_find_keys_by_identity_bytes_and_unhashed_copies() {
        for count in [3, 40] {
            let mut entries = TableEntries::default();
            let stored: Vec<LuaString> = (0..count)
                .map(|index| LuaString::from(format!("field{index}")))
                .collect();
            for (index, key) in stored.iter().enumerate() {
                entries.insert(TableKey::string(key), Value::Integer(index as i64));
            }
            if count > HASHED_TABLE_ENTRY_THRESHOLD {
                entries.promote();
                assert!(entries.is_indexed());
            }
            for (index, key) in stored.iter().enumerate() {
                let want = Some(&Value::Integer(index as i64));
                // The stored allocation itself.
                assert_eq!(entries.get_string(key), want);
                // An equal string nothing has hashed yet.
                let copy = LuaString::from(format!("field{index}"));
                assert_eq!(copy.known_hash(), 0);
                assert_eq!(entries.get_string(&copy), want);
                // The same copy, now hashed.
                assert_eq!(entries.get_string(&copy), want);
                assert_eq!(entries.get_string_live(&copy), want);
            }
            let absent = LuaString::from("field_absent");
            assert_eq!(entries.get_string(&absent), None);
            assert_eq!(entries.get_string(&absent), None);
        }
    }

    #[test]
    fn colliding_cached_hashes_still_compare_bytes() {
        let mut entries = SmallOrderedEntries::default();
        for (name, number) in [("one", 1), ("two", 2), ("three", 3)] {
            let mut entry = OrderedEntry::new(string_key(name), Value::Integer(number));
            entry.hash = 0xfeed;
            let at = entries.position(&entry.key).unwrap_err();
            entries.entries.insert(at, entry);
        }
        for (name, number) in [("one", 1), ("two", 2), ("three", 3)] {
            let index = entries.find_string(&LuaString::from(name), 0xfeed).unwrap();
            assert_eq!(entries.entries[index].value, Value::Integer(number));
        }
        assert!(
            entries
                .find_string(&LuaString::from("four"), 0xfeed)
                .is_none()
        );
        // A non-string entry carrying the same hash is never accepted.
        let mut other = OrderedEntry::new(TableKey::Integer(7), Value::Nil);
        other.hash = 0xfeed;
        entries.entries.insert(0, other);
        assert!(
            entries
                .find_string(&LuaString::from("four"), 0xfeed)
                .is_none()
        );
        assert!(
            entries
                .find_string(&LuaString::from("two"), 0xfeed)
                .is_some()
        );
    }

    #[test]
    fn promotion_keeps_lookups_and_order() {
        let mut entries = TableEntries::default();
        for index in 0..20 {
            entries.insert(string_key(&format!("k{index:02}")), Value::Integer(index));
        }
        let before = order(&entries);
        entries.promote();
        assert!(entries.is_indexed());
        assert_eq!(order(&entries), before);
        assert_eq!(
            entries.get_string(&LuaString::from("k07")),
            Some(&Value::Integer(7))
        );
    }

    /// The key set a traversal builds stays in step with later insertions and
    /// removals, and one built after them holds the same keys.
    #[test]
    fn built_key_set_follows_later_changes() {
        let mut entries = TableEntries::default();
        for index in 0..20 {
            entries.insert(string_key(&format!("k{index:02}")), Value::Integer(index));
        }
        entries.promote();
        let mut late = TableEntries::default();
        for index in 0..20 {
            late.insert(string_key(&format!("k{index:02}")), Value::Integer(index));
        }
        late.promote();
        assert!(entries.first_after(None).is_some());
        for table in [&mut entries, &mut late] {
            table.insert(string_key("k99"), Value::Integer(99));
            table.insert(string_key("a00"), Value::Integer(0));
            table.remove(&string_key("k05"));
        }
        let built = order(&entries);
        assert_eq!(built, order(&late));
        assert_eq!(built.len(), 21);
        assert_eq!(built.first().map(String::as_str), Some("a00"));
        assert!(!built.iter().any(|name| name == "k05"));
    }

    /// The out-of-line leaf, the inlined scan, and the nil-filtering variants
    /// all agree, for both representations, on hits, misses and stored nils.
    #[test]
    fn string_lookup_entry_points_agree() {
        for count in [6, 20] {
            let mut entries = TableEntries::default();
            for index in 0..count {
                entries.insert(string_key(&format!("k{index:02}")), Value::Integer(index));
            }
            entries.insert(string_key("hole"), Value::Nil);
            if count > 8 {
                entries.promote();
            }
            assert_eq!(entries.is_indexed(), count > 8);
            for index in 0..count {
                // A fresh allocation, so the pointer-identity shortcut misses.
                let key = LuaString::from(format!("k{index:02}").as_str());
                let hit = Some(&Value::Integer(index));
                assert_eq!(entries.get_string(&key), hit);
                assert_eq!(entries.get_string_live(&key), hit);
                assert_eq!(entries.get_string_live_scan(&key), hit);
            }
            let hole = LuaString::from("hole");
            // A stored nil reads as absent in a hashed table; an ordered one
            // still returns the nil it holds.
            assert!(
                entries
                    .get_string(&hole)
                    .is_none_or(|value| matches!(value, Value::Nil))
            );
            assert!(entries.get_string_live(&hole).is_none());
            assert!(entries.get_string_live_scan(&hole).is_none());
            let absent = LuaString::from("absent");
            assert!(entries.get_string(&absent).is_none());
            assert!(entries.get_string_live(&absent).is_none());
            assert!(entries.get_string_live_scan(&absent).is_none());
        }
    }

    fn wide_entries() -> TableEntries {
        let mut entries = TableEntries::default();
        for index in 0..20 {
            entries.insert(string_key(&format!("key{index}")), Value::Integer(index));
        }
        entries.promote();
        entries
    }

    #[test]
    fn read_cache_follows_overwrites_and_drops_removed_keys() {
        let mut entries = wide_entries();
        let mut cache = StringReadCache::default();
        let key = LuaString::from("key7");
        // The second read of a key fills its slot; the third one hits.
        for _ in 0..3 {
            assert_eq!(
                entries.get_string_live_cached(&key, &mut cache),
                Some(&Value::Integer(7))
            );
        }
        assert_eq!((cache.hits(), cache.installs()), (1, 1));
        // An overwrite writes the bucket in place and keeps the cached slot.
        entries.insert(string_key("key7"), Value::Integer(70));
        assert_eq!(
            entries.get_string_live_cached(&key, &mut cache),
            Some(&Value::Integer(70))
        );
        assert_eq!((cache.hits(), cache.installs()), (2, 1));
        // A removal changes the layout, so the slot cannot answer.
        entries.remove(&string_key("key7"));
        assert_eq!(entries.get_string_live_cached(&key, &mut cache), None);
        // Growing past the capacity moves every bucket.
        entries.insert(string_key("key7"), Value::Integer(1));
        assert_eq!(
            entries.get_string_live_cached(&key, &mut cache),
            Some(&Value::Integer(1))
        );
        for index in 100..400 {
            entries.insert(string_key(&format!("key{index}")), Value::Integer(index));
        }
        assert_eq!(
            entries.get_string_live_cached(&key, &mut cache),
            Some(&Value::Integer(1))
        );
        entries.insert(string_key("key7"), Value::Nil);
        assert_eq!(entries.get_string_live_cached(&key, &mut cache), None);
    }

    #[test]
    fn read_cache_does_not_confuse_two_tables() {
        let mut first = wide_entries();
        let second = wide_entries();
        first.insert(string_key("key3"), Value::Integer(33));
        let mut cache = StringReadCache::default();
        let key = LuaString::from("key3");
        for _ in 0..3 {
            assert_eq!(
                first.get_string_live_cached(&key, &mut cache),
                Some(&Value::Integer(33))
            );
            assert_eq!(
                second.get_string_live_cached(&key, &mut cache),
                Some(&Value::Integer(3))
            );
        }
    }

    /// Allocates strings round-robin across `names`, so they share one
    /// allocator stream, and returns one string per name, all in the same
    /// set under [`read_cache_set`].
    fn keys_in_one_set(names: &[&str]) -> Vec<LuaString> {
        let pool: Vec<Vec<LuaString>> = {
            let mut pool = vec![Vec::new(); names.len()];
            for _ in 0..4096 {
                for (list, name) in pool.iter_mut().zip(names) {
                    list.push(LuaString::from(*name));
                }
            }
            pool
        };
        for set in 0..READ_CACHE_SETS {
            let picked: Vec<Option<&LuaString>> = pool
                .iter()
                .map(|list| {
                    list.iter()
                        .find(|key| read_cache_set(key.identity()) == set)
                })
                .collect();
            if picked.iter().all(Option::is_some) {
                return picked.into_iter().map(|key| key.unwrap().clone()).collect();
            }
        }
        panic!("no common set among {} allocations per name", 4096);
    }

    fn run_loop(keys: &[LuaString], rounds: usize) -> StringReadCache {
        let entries = wide_entries();
        let mut cache = StringReadCache::default();
        for _ in 0..rounds {
            for key in keys {
                let expected = key
                    .to_string()
                    .trim_start_matches("key")
                    .parse::<i64>()
                    .unwrap();
                assert_eq!(
                    entries.get_string_live_cached(key, &mut cache),
                    Some(&Value::Integer(expected))
                );
            }
        }
        cache
    }

    /// Strings are 64 bytes apart in the allocator. The old `(identity >> 4)
    /// % 64` index kept two constant low bits and reached a quarter of the
    /// slots (16 of 64).
    #[test]
    fn read_cache_set_spreads_strided_identities() {
        let base = 0x5555_0000_1000_usize;
        let mut seen = [false; READ_CACHE_SETS];
        for index in 0..256 {
            seen[read_cache_set(base + index * 64)] = true;
        }
        let reached = seen.iter().filter(|&&hit| hit).count();
        assert!(reached >= READ_CACHE_SETS - 2, "{reached} sets reached");
    }

    /// Two hot keys in one set both stay cached. One-way slots evicted each
    /// other and never installed.
    #[test]
    fn read_cache_keeps_two_hot_keys_that_share_a_set() {
        let keys = keys_in_one_set(&["key3", "key7"]);
        assert_eq!(
            read_cache_set(keys[0].identity()),
            read_cache_set(keys[1].identity())
        );
        let cache = run_loop(&keys, 1000);
        assert!(cache.hits() >= 1990, "{} hits", cache.hits());
        assert!(cache.installs() <= 4, "{} installs", cache.installs());
    }

    /// A key that alternates with an installed key in the same set installs
    /// in the free second way: a hit must not clear its pending entry.
    #[test]
    fn read_cache_installs_a_key_alternating_with_an_installed_one() {
        let keys = keys_in_one_set(&["key3", "key7"]);
        let entries = wide_entries();
        let mut cache = StringReadCache::default();
        for _ in 0..3 {
            entries.get_string_live_cached(&keys[0], &mut cache);
        }
        let warm = cache.hits();
        for _ in 0..1000 {
            entries.get_string_live_cached(&keys[0], &mut cache);
            entries.get_string_live_cached(&keys[1], &mut cache);
        }
        assert!(cache.hits() - warm >= 1990, "{} hits", cache.hits() - warm);
        assert_eq!(cache.installs(), 2);
    }

    /// Three hot keys in one set: the loop order of the performance test
    /// (two globals, two fields) with the same names. The cache keeps the two
    /// most recent installs: about 7500 of these 15000 reads hit, against
    /// about 4800 with 750f914's one-way index. Three keys in one set still
    /// cost an install every four reads; the index over the varying bits makes that rare.
    /// (The flaky CLI test saw 4998 of 20000 hits.)
    #[test]
    fn read_cache_survives_three_hot_keys_in_one_set() {
        let keys = keys_in_one_set(&["key3", "key7", "key11"]);
        let cache = run_loop(&keys, 5000);
        assert!(cache.hits() >= 7000, "{} hits", cache.hits());
    }

    /// The loop of the CLI test constant_global_and_field_reads_hit_the_string_cache,
    /// with its four keys in distinct sets: nearly every read hits.
    #[test]
    fn read_cache_serves_the_four_key_loop_order() {
        let pool: Vec<LuaString> = (0..64)
            .flat_map(|_| ["key3", "key7", "key11", "key15"].map(LuaString::from))
            .collect();
        // Four keys, each in a set of its own. Which sets a name's strings
        // reach depends on the allocator, so take the first unused one.
        let mut keys: Vec<LuaString> = Vec::new();
        for name in 0..4 {
            let key = pool
                .iter()
                .skip(name)
                .step_by(4)
                .find(|key| {
                    let set = read_cache_set(key.identity());
                    keys.iter()
                        .all(|taken| read_cache_set(taken.identity()) != set)
                })
                .expect("a set no earlier key took")
                .clone();
            keys.push(key);
        }
        let sets: Vec<usize> = keys
            .iter()
            .map(|key| read_cache_set(key.identity()))
            .collect();
        assert!((0..4).all(|a| (a + 1..4).all(|b| sets[a] != sets[b])));
        let cache = run_loop(&keys, 5000);
        assert!(cache.hits() >= 19990, "{} hits", cache.hits());
    }

    /// Keys longer than a string packs inline need a byte comparison, which
    /// the hashed probe and the shape scan hand to a cold path. A fresh
    /// allocation of the stored key must still be found, and a key that
    /// differs in its last byte must not be.
    #[test]
    fn long_keys_are_found_by_bytes() {
        let long = |tail: char| format!("a_key_longer_than_the_inline_limit_{tail}");
        let probe = |entries: &TableEntries, tail: char| {
            let key = LuaString::from(long(tail).as_str());
            let found = entries.get_string_live_scan(&key).cloned();
            assert_eq!(entries.get_string_live(&key).cloned(), found);
            assert_eq!(entries.get_string_live_scan_ordered(&key).cloned(), found);
            found
        };

        let mut hashed = TableEntries::default();
        for tail in 'a'..='t' {
            hashed.insert(
                string_key(&long(tail)),
                Value::Integer(i64::from(tail as u8)),
            );
        }
        hashed.promote();
        assert!(hashed.is_indexed());
        assert_eq!(probe(&hashed, 'a'), Some(Value::Integer(97)));
        assert_eq!(probe(&hashed, 't'), Some(Value::Integer(116)));
        assert_eq!(probe(&hashed, 'z'), None);

        let shape =
            TableShape::new(['b', 'a', 'c'].map(|tail| LuaString::from(long(tail).as_str())))
                .unwrap();
        let mut shaped = TableEntries::with_shape(shape);
        assert!(shaped.is_shaped());
        let key = LuaString::from(long('b').as_str());
        assert_eq!(shaped.set_small_string(&key, Value::Integer(2)), Ok(true));
        assert_eq!(probe(&shaped, 'b'), Some(Value::Integer(2)));
        assert_eq!(probe(&shaped, 'a'), None);
        assert_eq!(probe(&shaped, 'z'), None);
        assert!(shaped.is_shaped());
    }

    /// The shaped block frees its shape and the values in its slots exactly
    /// once, whether it is dropped as it is or after being converted.
    #[test]
    fn shaped_block_releases_its_shape_and_values() {
        let shape = TableShape::new(["x", "y", "z"].map(LuaString::from)).unwrap();
        for convert in [false, true] {
            let held = LuaString::from("a value held by a slot");
            let weak = held.downgrade();
            let mut entries = TableEntries::with_shape(shape.clone());
            assert_eq!(Arc::strong_count(&shape), 2);
            let key = LuaString::from("y");
            assert_eq!(
                entries.set_small_string(&key, Value::String(held)),
                Ok(true)
            );
            assert_eq!(entries.len(), 1);
            assert!(weak.is_alive());
            if convert {
                entries.materialize();
                assert!(!entries.is_shaped());
                assert_eq!(Arc::strong_count(&shape), 1);
                assert!(weak.is_alive());
            }
            drop(entries);
            assert_eq!(Arc::strong_count(&shape), 1);
            assert!(!weak.is_alive());
        }
    }
}
