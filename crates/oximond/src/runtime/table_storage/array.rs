//! The array part of a table, backed by recycled small buffers.
//!
//! A constructor such as `{a, b, c}` makes a table whose array part is a few
//! values long and dies a moment later. A general allocator spends more on the
//! block than the constructor spends filling it, and a minor sweep that frees
//! thousands of them in a row pushes glibc off its thread cache onto the slow
//! free path. [`ArrayPart`] is a `Vec<Value>` that gives its buffer back to a
//! per-thread free list when it is dropped, and [`ArrayPart::with_capacity`]
//! takes from that list first.
//!
//! Every pooled buffer was allocated by the global allocator for exactly
//! `capacity` values, so a `Vec` rebuilt from it may grow or free it through
//! the allocator as usual, and a buffer the `Vec` grew is as good a pool
//! member as one the pool allocated. Buffers of 1 to [`MAX_POOLED`] values are
//! listed, one list per capacity. A list holds at most [`POOL_CAP`] buffers
//! ([`SHORT_POOL_CAP`] for the shortest capacities); a
//! buffer over the cap, or freed when the thread's pool is gone, goes back to
//! the allocator. A full collection and a dropped runtime trim each list to
//! [`TRIM_FLOOR`] buffers. A free buffer's first word links it to the next.

#[cfg(test)]
use super::free_list::CLOSED;
use super::free_list::FreeList;
use crate::runtime::value::Value;
use std::ops::{Deref, DerefMut};

/// Buffers each list keeps through a trim (a full collection or a dropped
/// runtime). The floor is per capacity, so a trim leaves at most
/// 256 * (1 + ... + 16) * 16 B = 0.56 MB (4096 would leave 8.9 MB of the 17.8
/// MB the lists can hold). Instruction counts of every table/* and
/// allocation/* row, tracing and interpreter, were identical at floors 0, 64,
/// 256, 1024, 4096 and 8192 (no trim), so this is the smallest floor that
/// keeps a warm list after a collection; see the commit message.
pub(crate) const TRIM_FLOOR: usize = 256;
/// Largest capacity, in values, whose buffers are listed.
pub(crate) const MAX_POOLED: usize = 16;
/// Most buffers one list keeps.
pub(crate) const POOL_CAP: usize = 8192;
/// Most buffers the list of a capacity up to [`SHORT_CAPACITY`] keeps. A
/// nursery holds about 10900 tables (see `GC_MIN_NURSERY_BYTES`), and a loop
/// that makes a short list per iteration frees that many buffers in one
/// minor collection and takes them back before the next. With [`POOL_CAP`]
/// a quarter of them went to the allocator and came back from it.
pub(crate) const SHORT_POOL_CAP: usize = 16384;
/// Largest capacity whose list keeps [`SHORT_POOL_CAP`] buffers. The four
/// lists hold 2.6 MB when full.
pub(crate) const SHORT_CAPACITY: usize = 4;

/// Most buffers the list of each capacity keeps, indexed by it. A table,
/// so that the drop of an array part reads its cap with one load: the drop
/// is in line in the drop of every table.
static POOL_CAPS: [usize; MAX_POOLED + 1] = {
    let mut caps = [POOL_CAP; MAX_POOLED + 1];
    let mut capacity = 0;
    while capacity <= SHORT_CAPACITY {
        caps[capacity] = SHORT_POOL_CAP;
        capacity += 1;
    }
    caps
};

/// Most buffers the list of `capacity` (at most [`MAX_POOLED`]) keeps.
#[inline]
fn pool_cap(capacity: usize) -> usize {
    POOL_CAPS[capacity]
}

fn layout(capacity: usize) -> std::alloc::Layout {
    std::alloc::Layout::array::<Value>(capacity).expect("pooled capacity is small")
}

/// One list per capacity, indexed by it; the list at 0 stays unused.
struct Pool {
    lists: [FreeList<Value>; MAX_POOLED + 1],
}

/// Frees the listed buffers when its thread exits. The lists have no
/// destructor, so reaching them costs no liveness check; the guard is created
/// when a list first gets a buffer.
struct Guard;

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = POOL.try_with(|pool| {
            for (capacity, list) in pool.lists.iter().enumerate().skip(1) {
                list.close(layout(capacity));
            }
        });
    }
}

/// Frees this thread's listed buffers down to [`TRIM_FLOOR`] per capacity.
pub(crate) fn trim_pool() {
    let _ = POOL.try_with(|pool| {
        for (capacity, list) in pool.lists.iter().enumerate().skip(1) {
            list.trim(TRIM_FLOOR, layout(capacity));
        }
    });
}

thread_local! {
    static POOL: Pool = const {
        Pool {
            lists: [const { FreeList::new() }; MAX_POOLED + 1],
        }
    };
    static GUARD: Guard = const { Guard };
}

#[cold]
#[inline(never)]
fn arm_guard() {
    // `Guard` has a destructor, so touching it registers that.
    let _ = GUARD.try_with(|_| ());
}

/// A listed buffer of `capacity` values (1 to [`MAX_POOLED`]), if there is one.
#[inline]
fn take(capacity: usize) -> Option<*mut Value> {
    POOL.try_with(|pool| {
        let list = &pool.lists[capacity];
        let block = list.take()?;
        crate::runtime::prefetch::soon(list.head());
        Some(block.as_ptr())
    })
    .ok()
    .flatten()
}

/// A new buffer of `capacity` values (1 to [`MAX_POOLED`]).
#[allow(unsafe_code)]
#[cold]
#[inline(never)]
fn fresh_block(capacity: usize) -> *mut Value {
    let layout = layout(capacity);
    // SAFETY: the layout is not empty, as `capacity` is at least 1.
    let block = unsafe { std::alloc::alloc(layout) };
    if block.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    block.cast()
}

/// Lists `block`, or frees it when its list is full or the pool is gone.
///
/// # Safety
///
/// `block` was allocated by the global allocator with the layout of
/// `capacity` values, holds no live value, and nothing refers to it.
#[allow(unsafe_code)]
#[inline]
unsafe fn give(capacity: usize, block: *mut Value) {
    let listed = POOL
        // SAFETY: as the caller promises; every buffer on this list has the
        // layout of `capacity` values.
        .try_with(|pool| unsafe { pool.lists[capacity].give(block, pool_cap(capacity), arm_guard) })
        .unwrap_or(false);
    if !listed {
        // SAFETY: as the caller promises.
        unsafe { std::alloc::dealloc(block.cast(), layout(capacity)) };
    }
}

/// The array part of a table: a `Vec<Value>` whose small buffers are recycled.
/// Transparent, so native code addresses its words as a vector's (see
/// `NativeTableLayout`).
#[derive(Default, Debug)]
#[repr(transparent)]
pub(crate) struct ArrayPart(Vec<Value>);

impl ArrayPart {
    #[inline]
    pub(crate) const fn new() -> Self {
        Self(Vec::new())
    }

    /// An empty array part that can hold `capacity` values without growing.
    #[allow(unsafe_code, clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        if (1..=MAX_POOLED).contains(&capacity) {
            let block = match take(capacity) {
                Some(block) => block,
                None => fresh_block(capacity),
            };
            // SAFETY: the buffer was allocated by the global allocator for
            // exactly `capacity` values, listed or fresh, and is unused.
            return Self(unsafe { Vec::from_raw_parts(block, 0, capacity) });
        }
        Self(Vec::with_capacity(capacity))
    }

    /// An array part of `count` nils with no room to spare, for a
    /// constructor whose code fills every slot itself.
    #[allow(unsafe_code, clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn nils(count: usize) -> Self {
        let mut part = Self::with_capacity(count);
        debug_assert!(part.0.capacity() >= count);
        // SAFETY: the buffer has room for `count` values, and each of the
        // first `count` slots is written before the length covers it.
        unsafe {
            let slots = part.0.as_mut_ptr();
            for index in 0..count {
                slots.add(index).write(Value::Nil);
            }
            part.0.set_len(count);
        }
        part
    }

    /// Takes a buffer for the first value of an empty array part.
    #[inline]
    pub(crate) fn start(&mut self, capacity: usize) {
        debug_assert_eq!(self.0.capacity(), 0);
        *self = Self::with_capacity(capacity);
    }
}

impl From<Vec<Value>> for ArrayPart {
    #[inline]
    fn from(values: Vec<Value>) -> Self {
        Self(values)
    }
}

impl<'a> IntoIterator for &'a ArrayPart {
    type Item = &'a Value;
    type IntoIter = std::slice::Iter<'a, Value>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a mut ArrayPart {
    type Item = &'a mut Value;
    type IntoIter = std::slice::IterMut<'a, Value>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

#[cfg(test)]
impl PartialEq<Vec<Value>> for ArrayPart {
    fn eq(&self, other: &Vec<Value>) -> bool {
        self.0 == *other
    }
}

impl Deref for ArrayPart {
    type Target = Vec<Value>;

    #[inline]
    fn deref(&self) -> &Vec<Value> {
        &self.0
    }
}

impl DerefMut for ArrayPart {
    #[inline]
    fn deref_mut(&mut self) -> &mut Vec<Value> {
        &mut self.0
    }
}

impl Drop for ArrayPart {
    #[allow(unsafe_code)]
    #[inline]
    fn drop(&mut self) {
        let capacity = self.0.capacity();
        if !(1..=MAX_POOLED).contains(&capacity) {
            return;
        }
        self.0.clear();
        let block = self.0.as_mut_ptr();
        std::mem::forget(std::mem::take(&mut self.0));
        // SAFETY: the buffer is the one the vector allocated for exactly
        // `capacity` values, now empty, and the vector no longer owns it.
        unsafe { give(capacity, block) };
    }
}

/// Buffers listed for `capacity` values on this thread.
#[cfg(test)]
pub(crate) fn listed(capacity: usize) -> usize {
    POOL.with(|pool| pool.lists[capacity].count())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_trim_keeps_the_floor() {
        let parts: Vec<ArrayPart> = (0..TRIM_FLOOR + 8)
            .map(|_| ArrayPart::with_capacity(5))
            .collect();
        drop(parts);
        assert!(listed(5) >= TRIM_FLOOR + 8);
        trim_pool();
        assert_eq!(listed(5), TRIM_FLOOR);
    }

    #[test]
    fn a_buffer_given_back_after_the_thread_freed_its_lists_is_freed() {
        struct Late(Option<ArrayPart>);
        impl Drop for Late {
            fn drop(&mut self) {
                drop(self.0.take());
                LISTED_AFTER.store(listed(3), std::sync::atomic::Ordering::SeqCst);
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
                late.borrow_mut().0 = Some(ArrayPart::with_capacity(3));
            });
            drop(ArrayPart::with_capacity(3));
            assert_eq!(listed(3), 1);
        })
        .join()
        .unwrap();
        assert_eq!(
            LISTED_AFTER.load(std::sync::atomic::Ordering::SeqCst),
            CLOSED
        );
    }

    #[test]
    fn a_dropped_buffer_is_the_next_one_taken() {
        let before = listed(3);
        let mut part = ArrayPart::with_capacity(3);
        part.push(Value::Integer(1));
        let address = part.as_ptr();
        drop(part);
        assert_eq!(listed(3), before + 1);
        let again = ArrayPart::with_capacity(3);
        assert_eq!(again.as_ptr(), address);
        assert!(again.is_empty());
        assert_eq!(again.capacity(), 3);
        assert_eq!(listed(3), before);
    }

    #[test]
    fn a_grown_buffer_joins_the_list_of_its_new_capacity() {
        let before = listed(4);
        let mut part = ArrayPart::with_capacity(1);
        for index in 0..4 {
            part.push(Value::Integer(index));
        }
        assert_eq!(part.capacity(), 4);
        drop(part);
        assert_eq!(listed(4), before + 1);
    }

    #[test]
    fn large_buffers_are_not_listed() {
        let part = ArrayPart::with_capacity(MAX_POOLED + 1);
        drop(part);
        let part = ArrayPart::with_capacity(0);
        assert_eq!(part.capacity(), 0);
    }

    #[test]
    fn the_list_is_capped() {
        for capacity in [2, SHORT_CAPACITY, SHORT_CAPACITY + 1] {
            let cap = pool_cap(capacity);
            let parts: Vec<ArrayPart> = (0..cap + 10)
                .map(|_| ArrayPart::with_capacity(capacity))
                .collect();
            let before = listed(capacity);
            drop(parts);
            assert_eq!(listed(capacity), cap.max(before), "capacity {capacity}");
        }
        assert_eq!(pool_cap(SHORT_CAPACITY), SHORT_POOL_CAP);
        assert_eq!(pool_cap(SHORT_CAPACITY + 1), POOL_CAP);
    }
}
