//! The vector of a small ordered table, with recycled blocks for its first
//! entry.
//!
//! Most tables built by assignment never take a second key, and a general
//! allocator spends more on the one-entry block than the store spends
//! filling it. A thread keeps a list of the one-entry blocks of dead tables;
//! the first insert into an empty table takes one, and dropping a table
//! whose vector still has that capacity lists the block again. Growth past
//! one entry goes to the allocator, as `Vec` would. A full collection and a
//! dropped runtime trim the list to [`TRIM_FLOOR`] blocks.

#![allow(unsafe_code)]

use super::entries::OrderedEntry;
#[cfg(test)]
use super::free_list::CLOSED;
use super::free_list::FreeList;
use super::key_layout_changed;
use std::alloc::Layout;
use std::ptr::NonNull;

/// Capacity of the block the first insert takes.
const FIRST_CAPACITY: usize = 1;
/// Capacity the vector grows to from the first block, as `Vec` does for
/// small elements.
const MIN_GROWN_CAPACITY: usize = 4;
/// Most one-entry blocks one thread keeps.
const POOL_CAP: usize = 16384;
/// Blocks the list keeps through a trim, about 200 KiB: a program that
/// builds a few thousand tables of one key between full collections takes
/// them back without the allocator. With 256, allocation/gc_cycles (4000
/// such tables per `collectgarbage("collect")`) ran 21% more instructions.
pub(crate) const TRIM_FLOOR: usize = 4096;

/// A `Vec<OrderedEntry>` whose one-entry blocks are recycled. Only the
/// methods the ordered representation uses are here; the rest is the
/// slice's.
pub(crate) struct OrderedVec {
    /// Dangling while `cap` is zero.
    ptr: NonNull<OrderedEntry>,
    len: usize,
    cap: usize,
}

// SAFETY: the vector uniquely owns its block and its entries, as a
// `Vec<OrderedEntry>` does, and `OrderedEntry` is `Send` and `Sync`.
unsafe impl Send for OrderedVec {}
// SAFETY: shared access only reads through `&self`.
unsafe impl Sync for OrderedVec {}

const _: () = {
    fn assert<T: Send + Sync>() {}
    let _ = assert::<OrderedEntry>;
};

impl Default for OrderedVec {
    fn default() -> Self {
        Self::new()
    }
}

/// Frees the listed blocks when its thread exits. The list itself has no
/// destructor, so reaching it costs no liveness check; the guard is created
/// when the list first gets a block.
struct Guard;

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = POOL.try_with(|pool| pool.close(Layout::new::<OrderedEntry>()));
    }
}

/// Frees this thread's listed one-entry blocks down to [`TRIM_FLOOR`].
pub(crate) fn trim_pool() {
    let _ = POOL.try_with(|pool| pool.trim(TRIM_FLOOR, Layout::new::<OrderedEntry>()));
}

thread_local! {
    static POOL: FreeList<OrderedEntry> = const { FreeList::new() };
    static GUARD: Guard = const { Guard };
}

/// Makes the thread free its list when it exits.
#[cold]
#[inline(never)]
fn arm_guard() {
    // `Guard` has a destructor, so touching it registers that.
    let _ = GUARD.try_with(|_| ());
}

/// A block for one entry: a listed one, or one from the allocator.
#[inline]
fn take_block() -> Option<NonNull<OrderedEntry>> {
    let listed = POOL.try_with(FreeList::take).ok().flatten();
    listed.or_else(|| {
        // SAFETY: the layout of one entry is not empty.
        NonNull::new(unsafe { std::alloc::alloc(Layout::new::<OrderedEntry>()) }).map(NonNull::cast)
    })
}

/// Takes back a one-entry block whose entries are dropped: lists it, or
/// frees it when the list is full or the thread has freed it at exit.
///
/// # Safety
///
/// `block` came from the allocator with the layout of one entry, and nothing
/// refers to it any more.
#[inline]
unsafe fn give_block(block: NonNull<OrderedEntry>) {
    let listed = POOL
        // SAFETY: as the caller promises; every listed block has the
        // one-entry layout.
        .try_with(|pool| unsafe { pool.give(block.as_ptr(), POOL_CAP, arm_guard) })
        .unwrap_or(false);
    if !listed {
        // SAFETY: as the caller promises.
        unsafe { std::alloc::dealloc(block.as_ptr().cast(), Layout::new::<OrderedEntry>()) };
    }
}

/// Why a vector could not get a bigger block.
#[derive(Debug)]
enum GrowError {
    /// The capacity's size does not fit `isize`.
    CapacityOverflow,
    /// The allocator had no block of this layout.
    Alloc(Layout),
}

impl GrowError {
    /// Fails as `Vec` does: a panic for an impossible size, the allocation
    /// error handler for a refused allocation.
    #[cold]
    fn raise(self) -> ! {
        match self {
            Self::CapacityOverflow => panic!("capacity overflow"),
            Self::Alloc(layout) => std::alloc::handle_alloc_error(layout),
        }
    }
}

impl OrderedVec {
    pub(crate) const fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
        }
    }

    /// An empty vector with room for `capacity` entries.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let mut vector = Self::new();
        if capacity != 0 {
            vector
                .set_capacity(capacity)
                .unwrap_or_else(|error| error.raise());
        }
        vector
    }

    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        self.cap
    }

    #[inline]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut OrderedEntry {
        self.ptr.as_ptr()
    }

    /// Moves the entries to a block of exactly `capacity` entries, which is
    /// more than `len`. `Err` when the block cannot be had.
    #[cold]
    #[inline(never)]
    fn set_capacity(&mut self, capacity: usize) -> Result<(), GrowError> {
        debug_assert!(capacity >= self.len && capacity > self.cap);
        let fresh = if capacity == FIRST_CAPACITY {
            take_block().ok_or(GrowError::Alloc(Layout::new::<OrderedEntry>()))?
        } else {
            let layout =
                Layout::array::<OrderedEntry>(capacity).map_err(|_| GrowError::CapacityOverflow)?;
            // SAFETY: the layout is not empty.
            let block = unsafe { std::alloc::alloc(layout) };
            NonNull::new(block)
                .ok_or(GrowError::Alloc(layout))?
                .cast::<OrderedEntry>()
        };
        if self.cap != 0 {
            // SAFETY: the old block holds `len` entries, the new one has
            // room for them, and the blocks are distinct.
            unsafe { std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), fresh.as_ptr(), self.len) };
            // SAFETY: the old block's entries now live in the new one.
            unsafe { self.free_block() };
        }
        self.ptr = fresh;
        self.cap = capacity;
        Ok(())
    }

    /// Releases the block without dropping its entries.
    ///
    /// # Safety
    ///
    /// `cap` is not zero, and the entries were moved out or dropped. The
    /// caller replaces `ptr` and `cap` before the vector is used again.
    unsafe fn free_block(&mut self) {
        key_layout_changed();
        if self.cap == FIRST_CAPACITY {
            // SAFETY: a one-entry block came from the allocator with that
            // layout, and nothing refers to it.
            unsafe { give_block(self.ptr) };
        } else {
            let layout = Layout::array::<OrderedEntry>(self.cap).expect("layout was valid");
            // SAFETY: the block was allocated with this layout.
            unsafe { std::alloc::dealloc(self.ptr.as_ptr().cast(), layout) };
        }
    }

    /// The capacity after a full vector takes one more entry: one, then
    /// four, then twice as many.
    #[inline]
    fn grown(&self) -> usize {
        match self.cap {
            0 => FIRST_CAPACITY,
            cap => (cap * 2).max(MIN_GROWN_CAPACITY),
        }
    }

    /// Room for `additional` more entries, growing as `Vec::try_reserve`
    /// does.
    pub(crate) fn try_reserve(&mut self, additional: usize) -> Result<(), ()> {
        let required = self.len.checked_add(additional).ok_or(())?;
        if required <= self.cap {
            return Ok(());
        }
        let capacity = required.max(self.cap * 2).max(MIN_GROWN_CAPACITY);
        self.set_capacity(capacity).map_err(|_| ())
    }

    /// Inserts `entry` at `index`, which is at most the length.
    #[inline]
    pub(crate) fn insert(&mut self, index: usize, entry: OrderedEntry) {
        assert!(index <= self.len, "insertion index is past the end");
        key_layout_changed();
        if self.len == self.cap {
            let capacity = self.grown();
            self.set_capacity(capacity)
                .unwrap_or_else(|error| error.raise());
        }
        // SAFETY: there is room for one more entry, and the entries from
        // `index` on move up by one before the slot is written.
        unsafe {
            let at = self.ptr.as_ptr().add(index);
            std::ptr::copy(at, at.add(1), self.len - index);
            at.write(entry);
        }
        self.len += 1;
    }

    pub(crate) fn push(&mut self, entry: OrderedEntry) {
        self.insert(self.len, entry);
    }

    /// Takes out the entry at `index`.
    pub(crate) fn remove(&mut self, index: usize) -> OrderedEntry {
        assert!(index < self.len, "removal index is out of bounds");
        key_layout_changed();
        // SAFETY: `index` is in bounds; the entry is read out and the ones
        // after it close the gap.
        unsafe {
            let at = self.ptr.as_ptr().add(index);
            let entry = at.read();
            std::ptr::copy(at.add(1), at, self.len - index - 1);
            self.len -= 1;
            entry
        }
    }

    /// Moves the entries out into a `Vec`, leaving this one empty.
    pub(crate) fn take_all(&mut self) -> Vec<OrderedEntry> {
        key_layout_changed();
        let mut taken = Vec::with_capacity(self.len);
        // SAFETY: the first `len` entries are initialized, `taken` has room
        // for them, and the length drops to zero so they are not dropped
        // twice.
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), taken.as_mut_ptr(), self.len);
            taken.set_len(self.len);
        }
        self.len = 0;
        taken
    }

    /// Drops the entries and keeps the block.
    pub(crate) fn clear(&mut self) {
        key_layout_changed();
        let len = std::mem::replace(&mut self.len, 0);
        // SAFETY: the first `len` entries are initialized, and the length
        // is already zero, so a panicking drop cannot reach them again.
        unsafe {
            std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), len));
        }
    }
}

impl std::ops::Deref for OrderedVec {
    type Target = [OrderedEntry];

    #[inline]
    fn deref(&self) -> &[OrderedEntry] {
        // SAFETY: the first `len` entries are initialized, and the pointer
        // is aligned and non-null even for an empty vector.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl std::ops::DerefMut for OrderedVec {
    #[inline]
    fn deref_mut(&mut self) -> &mut [OrderedEntry] {
        // SAFETY: as in `deref`, and the borrow is unique.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for OrderedVec {
    #[inline]
    fn drop(&mut self) {
        // A dead table that never held a hash entry skips the call.
        if self.len != 0 {
            self.clear();
        }
        if self.cap != 0 {
            // SAFETY: the entries are dropped and the vector is going away.
            unsafe { self.free_block() };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::table_storage::key::TableKey;
    use crate::runtime::value::Value;

    fn entry(number: i64) -> OrderedEntry {
        OrderedEntry::new(TableKey::Integer(number), Value::Integer(number * 10))
    }

    fn numbers(vector: &OrderedVec) -> Vec<i64> {
        vector
            .iter()
            .map(|entry| match &entry.value {
                Value::Integer(number) => *number / 10,
                other => panic!("unexpected value {other:?}"),
            })
            .collect()
    }

    /// Blocks listed on this thread.
    fn listed() -> usize {
        POOL.with(FreeList::count)
    }

    #[test]
    fn capacity_goes_one_four_eight_sixteen() {
        let mut vector = OrderedVec::new();
        assert_eq!(vector.capacity(), 0);
        let mut seen = Vec::new();
        for index in 0..9 {
            vector.push(entry(index));
            if seen.last() != Some(&vector.capacity()) {
                seen.push(vector.capacity());
            }
        }
        assert_eq!(seen, [1, 4, 8, 16]);
        assert_eq!(numbers(&vector), (0..9).collect::<Vec<_>>());
    }

    #[test]
    fn insert_and_remove_keep_the_order() {
        let mut vector = OrderedVec::new();
        for number in [5, 1, 3] {
            let at = vector
                .iter()
                .position(|held| matches!(&held.value, Value::Integer(held) if *held > number * 10))
                .unwrap_or(vector.len());
            vector.insert(at, entry(number));
        }
        vector.insert(0, entry(0));
        assert_eq!(numbers(&vector), [0, 1, 3, 5]);
        assert_eq!(numbers(&OrderedVec::new()), Vec::<i64>::new());
        let removed = vector.remove(1);
        assert_eq!(&removed.value, &Value::Integer(10));
        assert_eq!(numbers(&vector), [0, 3, 5]);
        vector.clear();
        assert!(vector.is_empty());
        assert_eq!(vector.capacity(), 4);
        vector.push(entry(9));
        assert_eq!(numbers(&vector), [9]);
    }

    #[test]
    fn a_dead_one_entry_vector_hands_its_block_to_the_next() {
        let mut first = OrderedVec::new();
        first.push(entry(1));
        assert_eq!(first.capacity(), 1);
        let block = first.as_mut_ptr();
        let before = listed();
        drop(first);
        assert_eq!(listed(), before + 1);
        let mut second = OrderedVec::new();
        second.push(entry(2));
        assert_eq!(listed(), before);
        assert_eq!(second.as_mut_ptr(), block);
        assert_eq!(numbers(&second), [2]);
    }

    #[test]
    fn growing_past_the_first_block_lists_it_again() {
        let mut vector = OrderedVec::new();
        vector.push(entry(1));
        let before = listed();
        vector.push(entry(2));
        assert_eq!(vector.capacity(), 4);
        assert_eq!(listed(), before + 1);
        assert_eq!(numbers(&vector), [1, 2]);
    }

    #[test]
    fn reserve_and_take_all_move_the_entries() {
        let mut vector = OrderedVec::with_capacity(3);
        assert_eq!(vector.capacity(), 3);
        vector.push(entry(1));
        vector.try_reserve(10).unwrap();
        assert!(vector.capacity() >= 11);
        assert_eq!(numbers(&vector), [1]);
        vector.push(entry(2));
        let taken = vector.take_all();
        assert_eq!(taken.len(), 2);
        assert!(vector.is_empty());
        assert!(vector.try_reserve(usize::MAX).is_err());
    }

    #[test]
    #[should_panic(expected = "capacity overflow")]
    fn a_capacity_too_big_for_a_layout_panics_as_vec_does() {
        let _ = OrderedVec::with_capacity(usize::MAX / 2);
    }

    #[test]
    fn a_trim_keeps_the_floor() {
        let vectors: Vec<OrderedVec> = (0..TRIM_FLOOR + 8)
            .map(|index| {
                let mut vector = OrderedVec::new();
                vector.push(entry(i64::try_from(index).unwrap()));
                vector
            })
            .collect();
        drop(vectors);
        assert!(listed() >= TRIM_FLOOR + 8);
        trim_pool();
        assert_eq!(listed(), TRIM_FLOOR);
    }

    #[test]
    fn a_block_given_back_after_the_thread_freed_its_list_is_freed() {
        struct Late(Option<OrderedVec>);
        impl Drop for Late {
            fn drop(&mut self) {
                drop(self.0.take());
                LISTED_AFTER.store(listed(), std::sync::atomic::Ordering::SeqCst);
            }
        }
        thread_local! {
            static LATE: std::cell::RefCell<Late> = const { std::cell::RefCell::new(Late(None)) };
        }
        static LISTED_AFTER: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(usize::MAX - 1);
        std::thread::spawn(|| {
            // Registered before the guard, so it runs after it at exit.
            LATE.with(|late| {
                let mut vector = OrderedVec::new();
                vector.push(entry(1));
                late.borrow_mut().0 = Some(vector);
            });
            let mut first = OrderedVec::new();
            first.push(entry(2));
            drop(first);
            assert_eq!(listed(), 1);
        })
        .join()
        .unwrap();
        assert_eq!(
            LISTED_AFTER.load(std::sync::atomic::Ordering::SeqCst),
            CLOSED
        );
    }

    #[test]
    fn the_list_is_bounded() {
        let vectors: Vec<OrderedVec> = (0..POOL_CAP + 8)
            .map(|index| {
                let mut vector = OrderedVec::new();
                vector.push(entry(i64::try_from(index).unwrap()));
                vector
            })
            .collect();
        drop(vectors);
        assert!(listed() <= POOL_CAP);
    }
}
