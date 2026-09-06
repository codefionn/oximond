//! A global allocator for the small blocks a Lua runtime makes by the
//! million: strings, table parts, call vectors.
//!
//! Starting a game such as Mineclonia makes about 25 million allocations,
//! almost all of them 16 to 64 bytes, and frees as many. Through glibc each
//! costs about 57 instructions on average: the thread cache holds only seven
//! blocks of a size, and a heap that keeps growing goes through `_int_malloc`
//! and the arena lock. [`SmallBlocks`] serves every request of at most
//! [`MAX_SMALL`] bytes and 16-byte alignment from per-thread free lists, one
//! per 16-byte size class, so the common allocation and free are a few
//! loads and stores. A list that runs dry takes a chain of blocks from the
//! depot, where threads leave blocks they gave up, and else carves a new
//! block from its [`SLAB_BYTES`] slab. Larger and more aligned requests go to
//! the system allocator.
//!
//! A block goes back to the list of the thread that frees it. A list that
//! reaches [`LIST_CAP`] blocks leaves half of them in the depot, so a thread
//! that frees what another allocates does not hoard them, and a thread that
//! exits leaves all of its blocks and the unused end of its slab. Slabs are
//! never returned to the system: the memory small blocks take peaks with the
//! program's live small data and is reused from then on.
//!
//! A thread's exit is noticed by a thread-local guard whose destructor runs
//! with the other thread-local destructors. Memory freed after it, by later
//! destructors or, on the main thread, by C++ static destructors and `atexit`
//! handlers, still goes to the thread's lists, but a list then holds at most
//! [`CLOSED_CAP`] blocks before it goes to the depot whole. So teardown takes
//! the depot lock once per batch, and a thread that ends strands fewer than
//! [`CLOSED_CAP`] blocks in each class it used after its guard ran.
//!
//! An embedder opts in by naming the allocator, as the C interface does:
//!
//! ```ignore
//! #[global_allocator]
//! static ALLOCATOR: oximond::alloc::SmallBlocks = oximond::alloc::SmallBlocks;
//! ```

#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

/// The largest request served from the size classes.
pub const MAX_SMALL: usize = 256;
/// The size classes: 16, 32, ..., [`MAX_SMALL`] bytes.
const CLASSES: usize = MAX_SMALL / 16;
/// The lists each thread keeps: one per class, and in tests some private
/// classes no request maps to, so a test can count a class's blocks exactly.
const SLOTS: usize = if cfg!(test) { CLASSES + 3 } else { CLASSES };
/// The bytes a thread takes from the system at once to carve blocks from.
pub const SLAB_BYTES: usize = 64 * 1024;
/// The most blocks one list holds before it leaves half in the depot.
const LIST_CAP: usize = 2048;
/// The most blocks one refill carves from the slab.
const CARVE_BATCH: usize = 16;
/// The most blocks one list holds after the thread's guard ran.
pub const CLOSED_CAP: usize = 32;

/// Serves small blocks from per-thread lists (see the module documentation).
pub struct SmallBlocks;

/// The class of a request of `size` bytes, from 1 to [`MAX_SMALL`].
#[inline(always)]
const fn class_of(size: usize) -> usize {
    size.div_ceil(16) - 1
}

/// The bytes of a block of class `class`.
#[inline(always)]
const fn class_size(class: usize) -> usize {
    (class + 1) * 16
}

#[inline(always)]
const fn is_small(layout: Layout) -> bool {
    layout.size() <= MAX_SMALL && layout.align() <= 16
}

/// Reads the word at `index` of a free block.
///
/// # Safety
///
/// `block` is a free block or slab tail at least `index + 1` words long.
#[inline(always)]
unsafe fn word(block: *mut u8, index: usize) -> *mut u8 {
    // SAFETY: as the caller promises.
    unsafe { block.cast::<*mut u8>().add(index).read() }
}

/// Writes the word at `index` of a free block.
///
/// # Safety
///
/// As [`word`].
#[inline(always)]
unsafe fn set_word(block: *mut u8, index: usize, value: *mut u8) {
    // SAFETY: as the caller promises.
    unsafe { block.cast::<*mut u8>().add(index).write(value) };
}

/// One thread's lists and the slab it carves from. It has no destructor, so
/// it stays reachable while other thread-local destructors free memory;
/// [`Guard`] empties it when the thread exits.
struct Local {
    heads: [Cell<*mut u8>; SLOTS],
    counts: [Cell<usize>; SLOTS],
    /// The next free byte of the current slab and the end of it.
    bump: Cell<*mut u8>,
    end: Cell<*mut u8>,
    state: Cell<State>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    /// The thread has not yet registered its [`Guard`].
    Unguarded,
    Guarded,
    /// The guard ran: lists hold at most [`CLOSED_CAP`] blocks, and the
    /// thread carves no more blocks.
    Closed,
}

thread_local! {
    static LOCAL: Local = const {
        Local {
            heads: [const { Cell::new(ptr::null_mut()) }; SLOTS],
            counts: [const { Cell::new(0) }; SLOTS],
            bump: Cell::new(ptr::null_mut()),
            end: Cell::new(ptr::null_mut()),
            state: Cell::new(State::Unguarded),
        }
    };
    static GUARD: Guard = const { Guard };
}

/// Leaves the thread's blocks and slab end for other threads at its exit.
struct Guard;

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = LOCAL.try_with(Local::close);
    }
}

/// A lock-protected list head. Locking allocates nothing.
struct Shelf(Mutex<ShelfHead>);

struct ShelfHead(*mut u8);

// SAFETY: the pointers name free memory that only the lock holder touches.
unsafe impl Send for ShelfHead {}

impl Shelf {
    const fn new() -> Self {
        Self(Mutex::new(ShelfHead(ptr::null_mut())))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ShelfHead> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Free blocks that threads left, per class: chains linked through each
/// block's first word, and the chains linked through the second word of
/// their first block.
static DEPOT: [Shelf; SLOTS] = [const { Shelf::new() }; SLOTS];

/// The chains in each class of the depot, changed under its lock. A thread
/// whose list runs dry reads it without the lock and carves a new block when
/// it reads zero, so the refills of a growing heap take no lock.
static DEPOT_CHAINS: [AtomicUsize; SLOTS] = [const { AtomicUsize::new(0) }; SLOTS];

/// The unused ends of slabs, which carving takes before a new slab. Each
/// holds its end and the next one in its first two words.
static TAILS: Shelf = Shelf::new();

/// Leaves the chain that starts at `head` in the depot.
///
/// # Safety
///
/// `head` starts a null-terminated chain of free blocks of `class`, linked
/// through their first words, that nothing else refers to.
unsafe fn depot_put(class: usize, head: *mut u8) {
    let mut depot = DEPOT[class].lock();
    // SAFETY: a block is at least two words long.
    unsafe { set_word(head, 1, depot.0) };
    depot.0 = head;
    DEPOT_CHAINS[class].fetch_add(1, Ordering::Relaxed);
}

/// Takes one chain from the depot, or null.
fn depot_take(class: usize) -> *mut u8 {
    if DEPOT_CHAINS[class].load(Ordering::Relaxed) == 0 {
        return ptr::null_mut();
    }
    let mut depot = DEPOT[class].lock();
    let head = depot.0;
    if !head.is_null() {
        // SAFETY: a chain's first block holds the next chain in its second
        // word.
        depot.0 = unsafe { word(head, 1) };
        DEPOT_CHAINS[class].fetch_sub(1, Ordering::Relaxed);
    }
    head
}

/// Keeps the unused slab bytes from `bump` to `end` for later carving.
///
/// # Safety
///
/// The bytes are the 16-aligned end of a slab that nothing uses.
unsafe fn tail_put(bump: *mut u8, end: *mut u8) {
    if bump.is_null() || end.addr() - bump.addr() < 16 {
        return;
    }
    let mut tails = TAILS.lock();
    // SAFETY: the tail is at least two words long.
    unsafe {
        set_word(bump, 0, end);
        set_word(bump, 1, tails.0);
    }
    tails.0 = bump;
}

/// Takes a kept slab end as its start and end, if there is one.
fn tail_take() -> Option<(*mut u8, *mut u8)> {
    let mut tails = TAILS.lock();
    let tail = tails.0;
    if tail.is_null() {
        return None;
    }
    // SAFETY: a kept tail holds its end and the next tail.
    unsafe {
        tails.0 = word(tail, 1);
        Some((tail, word(tail, 0)))
    }
}

impl Local {
    #[inline(always)]
    fn pop(&self, class: usize) -> *mut u8 {
        let head = self.heads[class].get();
        if !head.is_null() {
            // SAFETY: a listed block's first word links to the next one.
            self.heads[class].set(unsafe { word(head, 0) });
            self.counts[class].set(self.counts[class].get() - 1);
        }
        head
    }

    /// Lists `block`.
    ///
    /// # Safety
    ///
    /// `block` is a free block of `class` that nothing refers to.
    #[inline(always)]
    unsafe fn push(&self, class: usize, block: *mut u8) {
        if self.state.get() != State::Guarded {
            // SAFETY: as the caller promises.
            return unsafe { self.push_unguarded(class, block) };
        }
        // SAFETY: as the caller promises.
        if unsafe { self.link(class, block) } >= LIST_CAP {
            self.spill(class);
        }
    }

    /// Puts `block` at the head of its list and returns the list's length.
    ///
    /// # Safety
    ///
    /// As [`Local::push`].
    #[inline(always)]
    unsafe fn link(&self, class: usize, block: *mut u8) -> usize {
        // SAFETY: as the caller promises.
        unsafe { set_word(block, 0, self.heads[class].get()) };
        self.heads[class].set(block);
        let count = self.counts[class].get() + 1;
        self.counts[class].set(count);
        count
    }

    /// [`Local::push`] for a thread that has not registered its guard yet, or
    /// whose guard ran.
    ///
    /// # Safety
    ///
    /// As [`Local::push`].
    #[cold]
    #[inline(never)]
    unsafe fn push_unguarded(&self, class: usize, block: *mut u8) {
        if self.state.get() == State::Unguarded {
            // A thread that only frees what others allocated must leave
            // those blocks when it exits too.
            self.register();
        }
        // Linked after registering, which may have changed the list.
        // SAFETY: as the caller promises.
        let count = unsafe { self.link(class, block) };
        if self.state.get() == State::Closed {
            if count >= CLOSED_CAP {
                self.flush(class);
            }
        } else if count >= LIST_CAP {
            self.spill(class);
        }
    }

    /// Registers the thread's guard. Registering may allocate and free
    /// through this allocator, which then finds the lists open.
    #[cold]
    fn register(&self) {
        self.state.set(State::Guarded);
        if GUARD.try_with(|_| {}).is_err() {
            // The guard is gone: the thread is ending.
            self.close();
        }
    }

    /// Leaves every list and the slab end in the depot and keeps the lists
    /// short from now on.
    fn close(&self) {
        self.state.set(State::Closed);
        for class in 0..SLOTS {
            self.flush(class);
        }
        let bump = self.bump.replace(ptr::null_mut());
        let end = self.end.replace(ptr::null_mut());
        // SAFETY: a closed thread carves nothing, so nothing uses the rest
        // of the slab any more.
        unsafe { tail_put(bump, end) };
    }

    /// Leaves the whole list of `class` in the depot.
    fn flush(&self, class: usize) {
        let head = self.heads[class].replace(ptr::null_mut());
        self.counts[class].set(0);
        if !head.is_null() {
            // SAFETY: the list's blocks are free blocks of `class`, and no
            // list holds them any more.
            unsafe { depot_put(class, head) };
        }
    }

    /// Leaves the older half of a full list in the depot.
    #[cold]
    #[inline(never)]
    fn spill(&self, class: usize) {
        let keep = LIST_CAP / 2;
        let mut last = self.heads[class].get();
        // SAFETY: the list holds `LIST_CAP` blocks linked through their
        // first words.
        unsafe {
            for _ in 1..keep {
                last = word(last, 0);
            }
            let rest = word(last, 0);
            set_word(last, 0, ptr::null_mut());
            depot_put(class, rest);
        }
        self.counts[class].set(keep);
    }

    /// A block of `class` for an empty list: a chain from the depot, else a
    /// block carved from the slab.
    #[cold]
    #[inline(never)]
    fn refill(&self, class: usize) -> *mut u8 {
        if self.state.get() == State::Unguarded {
            self.register();
            // Registering may have listed blocks of this class.
            let block = self.pop(class);
            if !block.is_null() {
                return block;
            }
        }
        if self.state.get() == State::Closed {
            return self.refill_closed(class);
        }
        let chain = depot_take(class);
        if !chain.is_null() {
            let mut count = 0;
            let mut block = chain;
            while !block.is_null() {
                count += 1;
                // SAFETY: the chain's blocks link through their first words.
                block = unsafe { word(block, 0) };
            }
            self.heads[class].set(chain);
            self.counts[class].set(count);
            return self.pop(class);
        }
        let size = class_size(class);
        let mut bump = self.bump.get();
        if bump.is_null() || self.end.get().addr() - bump.addr() < size {
            // The rest of the slab is too small for this class but may
            // serve a smaller one.
            // SAFETY: nothing uses the rest of the slab.
            unsafe { tail_put(bump, self.end.get()) };
            let (start, end) = match tail_take() {
                Some((start, end)) if end.addr() - start.addr() >= size => (start, end),
                kept => {
                    if let Some((start, end)) = kept {
                        // SAFETY: as taken.
                        unsafe { tail_put(start, end) };
                    }
                    // SAFETY: the slab layout has a non-zero size.
                    let slab = unsafe { System.alloc(slab_layout()) };
                    if slab.is_null() {
                        return slab;
                    }
                    // SAFETY: the slab is `SLAB_BYTES` long.
                    (slab, unsafe { slab.add(SLAB_BYTES) })
                }
            };
            bump = start;
            self.end.set(end);
        }
        // Carve a batch: the first block is the caller's and the rest go on
        // the empty list, so a growing heap pays for one refill per batch.
        let count = ((self.end.get().addr() - bump.addr()) / size).min(CARVE_BATCH);
        let mut head = ptr::null_mut();
        for index in (1..count).rev() {
            // SAFETY: `count` blocks of `size` fit before the slab's end.
            unsafe {
                let block = bump.add(index * size);
                set_word(block, 0, head);
                head = block;
            }
        }
        self.heads[class].set(head);
        self.counts[class].set(count - 1);
        // SAFETY: as above.
        self.bump.set(unsafe { bump.add(count * size) });
        bump
    }
}

fn slab_layout() -> Layout {
    Layout::from_size_align(SLAB_BYTES, 16).expect("the slab layout is valid")
}

impl Local {
    /// [`Local::refill`] for a closed thread: at most [`CLOSED_CAP`] blocks
    /// of a depot chain, else one block from [`lone_block`].
    #[cold]
    #[inline(never)]
    fn refill_closed(&self, class: usize) -> *mut u8 {
        let chain = depot_take(class);
        if chain.is_null() {
            return lone_block(class);
        }
        let mut last = chain;
        let mut count = 1;
        // SAFETY: the chain's blocks link through their first words; the
        // rest goes back.
        unsafe {
            while count < CLOSED_CAP && !word(last, 0).is_null() {
                last = word(last, 0);
                count += 1;
            }
            let rest = word(last, 0);
            if !rest.is_null() {
                set_word(last, 0, ptr::null_mut());
                depot_put(class, rest);
            }
        }
        self.heads[class].set(chain);
        self.counts[class].set(count);
        self.pop(class)
    }
}

/// One block of `class` for a thread that cannot reach its lists, or that
/// finds no list blocks after its guard ran: from the depot, else from the
/// system. It is freed into a list like any other block, so a system block
/// has the class's full size.
#[cold]
fn lone_block(class: usize) -> *mut u8 {
    let chain = depot_take(class);
    if chain.is_null() {
        let layout =
            Layout::from_size_align(class_size(class), 16).expect("class layouts are valid");
        // SAFETY: the class layout has a non-zero size.
        return unsafe { System.alloc(layout) };
    }
    // SAFETY: the chain's blocks link through their first words; the rest
    // goes back.
    unsafe {
        let rest = word(chain, 0);
        if !rest.is_null() {
            depot_put(class, rest);
        }
    }
    chain
}

/// A block of `class`, or null when the system has no memory.
#[inline(always)]
fn alloc_class(class: usize) -> *mut u8 {
    LOCAL
        .try_with(|local| {
            let block = local.pop(class);
            if block.is_null() {
                local.refill(class)
            } else {
                block
            }
        })
        .unwrap_or_else(|_| lone_block(class))
}

/// Frees `block` of `class`.
///
/// # Safety
///
/// `block` came from [`alloc_class`] for `class` and nothing refers to it.
#[inline(always)]
unsafe fn dealloc_class(block: *mut u8, class: usize) {
    // SAFETY: as the caller promises.
    let listed = LOCAL.try_with(|local| unsafe { local.push(class, block) });
    if listed.is_err() {
        // SAFETY: as above; a lone block is a chain.
        unsafe {
            set_word(block, 0, ptr::null_mut());
            depot_put(class, block);
        }
    }
}

// SAFETY: a small block is `class_size` bytes of a slab, 16-aligned because
// slabs are and every class is a multiple of 16, or a system block of that
// size and alignment; it is handed out only while no list holds it.
// Everything else is the system allocator's.
unsafe impl GlobalAlloc for SmallBlocks {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !is_small(layout) {
            // SAFETY: as the caller promises.
            return unsafe { System.alloc(layout) };
        }
        alloc_class(class_of(layout.size()))
    }

    #[inline]
    unsafe fn dealloc(&self, block: *mut u8, layout: Layout) {
        if !is_small(layout) {
            // SAFETY: as the caller promises.
            return unsafe { System.dealloc(block, layout) };
        }
        // SAFETY: as the caller promises.
        unsafe { dealloc_class(block, class_of(layout.size())) };
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if !is_small(layout) {
            // SAFETY: as the caller promises.
            return unsafe { System.alloc_zeroed(layout) };
        }
        // SAFETY: as the caller promises.
        let block = unsafe { self.alloc(layout) };
        if !block.is_null() {
            // SAFETY: the block holds at least `layout.size()` bytes.
            unsafe { block.write_bytes(0, layout.size()) };
        }
        block
    }

    #[inline]
    unsafe fn realloc(&self, block: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let Ok(new_layout) = Layout::from_size_align(new_size, layout.align()) else {
            return ptr::null_mut();
        };
        let small = is_small(layout);
        if !small && !is_small(new_layout) {
            // SAFETY: as the caller promises.
            return unsafe { System.realloc(block, layout, new_size) };
        }
        if small && is_small(new_layout) && class_of(layout.size()) == class_of(new_size) {
            return block;
        }
        // SAFETY: as the caller promises.
        let moved = unsafe { self.alloc(new_layout) };
        if !moved.is_null() {
            // SAFETY: both blocks hold the smaller of the two sizes, and they
            // are distinct allocations.
            unsafe {
                ptr::copy_nonoverlapping(block, moved, layout.size().min(new_size));
                self.dealloc(block, layout);
            }
        }
        moved
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(size: usize) -> Layout {
        Layout::from_size_align(size, 8).unwrap()
    }

    #[test]
    fn requests_map_to_sixteen_byte_classes() {
        assert_eq!(class_of(1), 0);
        assert_eq!(class_of(16), 0);
        assert_eq!(class_of(17), 1);
        assert_eq!(class_of(MAX_SMALL), CLASSES - 1);
        assert_eq!(class_size(class_of(48)), 48);
        assert!(!is_small(Layout::from_size_align(16, 32).unwrap()));
        assert!(!is_small(layout(MAX_SMALL + 1)));
    }

    #[test]
    fn a_freed_block_is_the_next_one_of_its_class() {
        let allocator = SmallBlocks;
        // SAFETY: each block is freed once with the layout it was made with.
        unsafe {
            let first = allocator.alloc(layout(40));
            assert!(!first.is_null());
            assert_eq!(first.addr() % 16, 0);
            first.write_bytes(0xa5, 40);
            allocator.dealloc(first, layout(40));
            let second = allocator.alloc(layout(48));
            assert_eq!(first, second);
            let other = allocator.alloc(layout(40));
            assert_ne!(other, second);
            allocator.dealloc(second, layout(48));
            allocator.dealloc(other, layout(40));
        }
    }

    #[test]
    fn zeroed_and_moved_blocks_keep_their_bytes() {
        let allocator = SmallBlocks;
        // SAFETY: each block is freed once with its current layout.
        unsafe {
            let dirty = allocator.alloc(layout(64));
            dirty.write_bytes(0xff, 64);
            allocator.dealloc(dirty, layout(64));
            let zeroed = allocator.alloc_zeroed(layout(64));
            assert!((0..64).all(|offset| *zeroed.add(offset) == 0));
            for offset in 0..64u8 {
                *zeroed.add(usize::from(offset)) = offset;
            }
            let same = allocator.realloc(zeroed, layout(64), 60);
            assert_eq!(same, zeroed);
            // Into a larger class, out to the system, and back to a class.
            let larger = allocator.realloc(same, layout(60), 200);
            assert!((0..60u8).all(|offset| *larger.add(usize::from(offset)) == offset));
            let large = allocator.realloc(larger, layout(200), 4000);
            assert!((0..60u8).all(|offset| *large.add(usize::from(offset)) == offset));
            let back = allocator.realloc(large, layout(4000), 30);
            assert!((0..30u8).all(|offset| *back.add(usize::from(offset)) == offset));
            allocator.dealloc(back, layout(30));
        }
    }

    /// Private classes, one per test that counts a class's blocks.
    const SPILL_CLASS: usize = CLASSES;
    const HANDOFF_CLASS: usize = CLASSES + 1;
    const CLOSED_CLASS: usize = CLASSES + 2;

    /// The addresses of the blocks in the depot's `class`.
    fn depot_blocks(class: usize) -> std::collections::HashSet<usize> {
        // Allocated before locking, and too large for a class.
        let mut blocks = Vec::with_capacity(4 * LIST_CAP);
        let depot = DEPOT[class].lock();
        let mut chain = depot.0;
        while !chain.is_null() {
            let mut block = chain;
            while !block.is_null() {
                blocks.push(block.addr());
                // SAFETY: the depot's blocks link through their first words
                // and its chains through their first block's second word.
                block = unsafe { word(block, 0) };
            }
            // SAFETY: as above.
            chain = unsafe { word(chain, 1) };
        }
        drop(depot);
        blocks.into_iter().collect()
    }

    fn list_len(class: usize) -> usize {
        LOCAL.with(|local| local.counts[class].get())
    }

    fn chains(class: usize) -> usize {
        DEPOT_CHAINS[class].load(Ordering::Relaxed)
    }

    #[test]
    fn full_lists_and_exiting_threads_leave_blocks_for_other_threads() {
        // The blocks a thread frees past its cap, and all it holds when it
        // exits, serve other threads' allocations before new ones are carved.
        let wanted = LIST_CAP + 10;
        let freed: Vec<usize> = std::thread::spawn(move || {
            let blocks: Vec<*mut u8> = (0..wanted).map(|_| alloc_class(SPILL_CLASS)).collect();
            let addresses = blocks.iter().map(|block| block.addr()).collect();
            for block in blocks {
                // SAFETY: every block is freed once with its class.
                unsafe { dealloc_class(block, SPILL_CLASS) };
            }
            addresses
        })
        .join()
        .unwrap();
        let freed: std::collections::HashSet<usize> = freed.into_iter().collect();
        let reused = std::thread::spawn(move || {
            let blocks: Vec<*mut u8> = (0..wanted).map(|_| alloc_class(SPILL_CLASS)).collect();
            let reused = blocks
                .iter()
                .filter(|block| freed.contains(&block.addr()))
                .count();
            for block in blocks {
                // SAFETY: as above.
                unsafe { dealloc_class(block, SPILL_CLASS) };
            }
            reused
        })
        .join()
        .unwrap();
        // No other test uses the class, so the depot held every block the
        // first thread carved: the ones it freed and fewer than a batch more.
        assert!(reused > wanted - CARVE_BATCH, "{reused}");
    }

    #[test]
    fn a_thread_that_only_frees_leaves_the_blocks_when_it_exits() {
        let blocks: Vec<usize> = std::thread::spawn(|| {
            (0..100)
                .map(|_| alloc_class(HANDOFF_CLASS).expose_provenance())
                .collect()
        })
        .join()
        .unwrap();
        let freed = blocks.clone();
        std::thread::spawn(move || {
            for address in freed {
                // SAFETY: the thread that made the blocks has exited; each
                // is freed once with its class.
                unsafe { dealloc_class(ptr::with_exposed_provenance_mut(address), HANDOFF_CLASS) };
            }
        })
        .join()
        .unwrap();
        let depot = depot_blocks(HANDOFF_CLASS);
        let stranded = blocks.iter().filter(|block| !depot.contains(block)).count();
        assert_eq!(stranded, 0);
    }

    #[test]
    fn a_closed_thread_frees_and_allocates_in_batches() {
        std::thread::spawn(|| {
            let blocks: Vec<*mut u8> = (0..3 * CLOSED_CAP)
                .map(|_| alloc_class(CLOSED_CLASS))
                .collect();
            LOCAL.with(Local::close);
            assert_eq!(list_len(CLOSED_CLASS), 0);
            let before = chains(CLOSED_CLASS);
            for &block in &blocks[..2 * CLOSED_CAP] {
                // SAFETY: every block is freed once with its class.
                unsafe { dealloc_class(block, CLOSED_CLASS) };
            }
            // Two whole lists went to the depot, one lock each.
            assert_eq!(chains(CLOSED_CLASS), before + 2);
            assert_eq!(list_len(CLOSED_CLASS), 0);
            // An allocation takes one of those chains back.
            let again = alloc_class(CLOSED_CLASS);
            assert!(blocks.contains(&again));
            assert_eq!(list_len(CLOSED_CLASS), CLOSED_CAP - 1);
            assert_eq!(chains(CLOSED_CLASS), before + 1);
            for &block in &blocks[2 * CLOSED_CAP..] {
                // SAFETY: as above.
                unsafe { dealloc_class(block, CLOSED_CLASS) };
            }
            // SAFETY: as above.
            unsafe { dealloc_class(again, CLOSED_CLASS) };
        })
        .join()
        .unwrap();
    }
}
