//! Counting global allocator, compiled only with the `alloc-stats` feature.
//!
//! Every `alloc`, `alloc_zeroed` and `realloc` call counts as one allocation.
//! Bytes are cumulative. The peak is the highest live byte total seen. All
//! counters use relaxed atomics.

#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

#[cfg_attr(feature = "alloc-phases", allow(dead_code))]
pub struct CountingAllocator;

static COUNT: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);

/// Counts one allocation of `size` bytes and returns the live total after it.
pub(crate) fn grow(size: usize) -> u64 {
    let size = size as u64;
    COUNT.fetch_add(1, Relaxed);
    BYTES.fetch_add(size, Relaxed);
    let live = LIVE.fetch_add(size, Relaxed) + size;
    PEAK.fetch_max(live, Relaxed);
    live
}

pub(crate) fn shrink(size: usize) {
    LIVE.fetch_sub(size as u64, Relaxed);
}

// SAFETY: every method forwards to `System` with the caller's arguments
// unchanged and only updates counters around the call.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            grow(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            grow(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        shrink(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            shrink(layout.size());
            grow(new_size);
        }
        new
    }
}

#[cfg(not(feature = "alloc-phases"))]
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

// The phased allocator counts into the same statics through `grow` and
// `shrink`, so the global counters keep their meaning.
#[cfg(feature = "alloc-phases")]
#[global_allocator]
static ALLOCATOR: crate::alloc_phases::PhasedAllocator = crate::alloc_phases::PhasedAllocator;

/// Live bytes now.
#[cfg(feature = "alloc-phases")]
pub(crate) fn live() -> u64 {
    LIVE.load(Relaxed)
}

/// `(alloc_count, alloc_bytes, alloc_peak_bytes)` so far.
pub fn snapshot() -> (u64, u64, u64) {
    (COUNT.load(Relaxed), BYTES.load(Relaxed), PEAK.load(Relaxed))
}
