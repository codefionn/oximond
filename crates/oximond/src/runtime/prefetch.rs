//! Cache prefetch hints for the allocation paths.
//!
//! A free list hands out memory that was last touched a nursery ago, so the
//! first write to it waits on a cache miss. The allocator knows which block
//! it will hand out next and asks for it early.

/// Hints that the cache line at `ptr` will be written soon. Never faults,
/// whatever `ptr` points at, and does nothing off x86-64.
#[allow(unsafe_code, clippy::inline_always)]
#[inline(always)]
pub(crate) fn soon<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a prefetch is a hint; it does not dereference `ptr` in the
    // abstract machine and raises no fault for any address.
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr.cast());
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = ptr;
}
