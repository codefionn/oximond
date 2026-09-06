//! A per-thread free list of same-layout blocks, linked through their first
//! word. [`array`](super::array) keeps one per small capacity and
//! [`ordered_vec`](super::ordered_vec) one for its one-entry blocks.

#![allow(unsafe_code)]

use std::alloc::Layout;
use std::cell::Cell;
use std::ptr::NonNull;

/// The count of a list whose thread has freed it at exit. It is above any
/// cap, so blocks given back after that are freed, not listed, and the head
/// stays null, so nothing is taken from it.
pub(crate) const CLOSED: usize = usize::MAX;

/// Free blocks of one layout. The list has no destructor: its owner closes
/// it from a thread-local guard.
pub(crate) struct FreeList<T> {
    head: Cell<*mut T>,
    count: Cell<usize>,
}

impl<T> FreeList<T> {
    /// A listed block holds the link to the next one in its first word.
    const LINKABLE: () = assert!(
        std::mem::size_of::<T>() >= std::mem::size_of::<*mut u8>()
            && std::mem::align_of::<T>() >= std::mem::align_of::<*mut u8>()
    );

    pub(crate) const fn new() -> Self {
        let () = Self::LINKABLE;
        Self {
            head: Cell::new(std::ptr::null_mut()),
            count: Cell::new(0),
        }
    }

    /// Blocks listed, or [`CLOSED`].
    #[cfg(test)]
    pub(crate) fn count(&self) -> usize {
        self.count.get()
    }

    /// The block [`take`](Self::take) hands out next, or null.
    #[inline]
    pub(crate) fn head(&self) -> *mut T {
        self.head.get()
    }

    /// Unlinks the first listed block.
    #[inline]
    pub(crate) fn take(&self) -> Option<NonNull<T>> {
        let block = NonNull::new(self.head.get())?;
        // SAFETY: a listed block is at least a pointer long and its first
        // word holds the link `give` wrote.
        let next = unsafe { block.as_ptr().cast::<*mut T>().read() };
        self.head.set(next);
        self.count.set(self.count.get() - 1);
        Some(block)
    }

    /// Lists `block` unless the list holds `cap` blocks or is closed, and
    /// says whether it did. `first` runs before the list gets its first
    /// block.
    ///
    /// # Safety
    ///
    /// `block` is at least as large and aligned as `T`, nothing refers to
    /// it, and it came from the allocator with the layout every block of
    /// this list has.
    #[inline]
    pub(crate) unsafe fn give(&self, block: *mut T, cap: usize, first: impl FnOnce()) -> bool {
        let count = self.count.get();
        if count >= cap {
            return false;
        }
        if count == 0 {
            first();
        }
        // SAFETY: the block holds at least a pointer.
        unsafe { block.cast::<*mut T>().write(self.head.get()) };
        self.head.set(block);
        self.count.set(count + 1);
        true
    }

    /// Frees blocks of `layout` until `keep` are left.
    pub(crate) fn free_listed(&self, keep: usize, layout: Layout) {
        let mut count = self.count.get();
        let mut block = self.head.get();
        while count > keep && !block.is_null() {
            // SAFETY: a listed block came from the allocator with `layout`,
            // and its first word holds the next link.
            unsafe {
                let next = block.cast::<*mut T>().read();
                std::alloc::dealloc(block.cast(), layout);
                block = next;
            }
            count -= 1;
        }
        self.head.set(block);
        self.count.set(count);
    }

    /// Frees blocks of `layout` down to `floor`, unless the list is closed.
    pub(crate) fn trim(&self, floor: usize, layout: Layout) {
        let count = self.count.get();
        if count > floor && count != CLOSED {
            self.free_listed(floor, layout);
        }
    }

    /// Frees every block of `layout` at thread exit. Thread-local
    /// destructors that run after this may still give blocks back, and
    /// nothing would free what they listed, so the list stays closed.
    pub(crate) fn close(&self, layout: Layout) {
        self.free_listed(0, layout);
        self.count.set(CLOSED);
    }
}
