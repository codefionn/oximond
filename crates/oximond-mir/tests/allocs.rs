//! A compile that reuses its `Compiler` allocates nothing.
#![cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]
#![allow(unsafe_code)]

use oximond_mir::ir::Func;
use oximond_mir::kernels;
use oximond_mir::ra::Compiler;
use oximond_mir::x64::X64;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

struct Counting;
static COUNT: AtomicU64 = AtomicU64::new(0);

// SAFETY: forwards to `System` unchanged and only counts around the calls.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        COUNT.fetch_add(1, Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        COUNT.fetch_add(1, Relaxed);
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static A: Counting = Counting;

#[test]
fn steady_state_compiles_do_not_allocate() {
    let mut c = Compiler::new(X64::detect());
    for k in kernels::ALL {
        let mut f = Func::new();
        (k.build)(&mut f);
        // The first compiles size the arenas.
        c.compile(&f);
        c.compile(&f);
        let before = COUNT.load(Relaxed);
        for _ in 0..5 {
            c.compile(&f);
        }
        let n = COUNT.load(Relaxed) - before;
        assert_eq!(n, 0, "{}: {n} allocations in 5 compiles", k.name);
    }
}
