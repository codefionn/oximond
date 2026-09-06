//! A call of a vararg function, a call that forwards `...`, and a tail call
//! allocate nothing: the extra arguments live on the value stack. A counting
//! allocator shows it. The counters are process-wide, so the tests
//! of this file take turns.

// A counting allocator has to implement the `unsafe` allocator trait.
#![allow(unsafe_code)]

use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static LARGEST: AtomicUsize = AtomicUsize::new(0);
/// The counters are process-wide, so the tests take turns.
static SERIAL: Mutex<()> = Mutex::new(());

fn note(size: usize) {
    ALLOCATIONS.fetch_add(1, Relaxed);
    LARGEST.fetch_max(size, Relaxed);
}

// SAFETY: every method forwards to the system allocator unchanged.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        // SAFETY: forwarded.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        // SAFETY: forwarded.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: forwarded.
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        note(size);
        // SAFETY: forwarded.
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The allocations `lua.exec(source)` makes, and the largest block it asked
/// for.
fn measure(lua: &mut Lua, source: &str) -> (usize, usize, Vec<Value>) {
    let before = ALLOCATIONS.load(Relaxed);
    LARGEST.store(0, Relaxed);
    let results = lua.exec(source).unwrap_or_else(|error| panic!("{error}"));
    (
        ALLOCATIONS.load(Relaxed) - before,
        LARGEST.load(Relaxed),
        results,
    )
}

#[test]
fn vararg_calls_forwarding_and_tail_calls_do_not_allocate_per_call() {
    let _turn = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut lua = Lua::builder()
        .profile(CompatibilityProfile::Lua54)
        .execution_mode(ExecutionMode::Interpreter)
        .build()
        .unwrap();
    lua.exec(
        r"
        function sum(...)
            local a, b, c, d, e = ...
            return a + b + c + d + e
        end
        function forward(...) return sum(...) end
        function relay(...) local x = sum(...) return x end
        function count(...) return select('#', ...) end
        function down(n, ...)
            if n == 0 then return ... end
            return down(n - 1, ...)
        end
        function ping(n, acc) if n == 0 then return acc end return pong(n - 1, acc + 1) end
        function pong(n, acc) if n == 0 then return acc end return ping(n - 1, acc + 1) end
        ",
    )
    .unwrap();
    let shapes = [
        ("sum", "total = total + sum(i, 2, 3, 4, 5)"),
        ("forward", "total = total + forward(i, 2, 3, 4, 5)"),
        ("relay", "total = total + relay(i, 2, 3, 4, 5)"),
        ("down", "total = total + down(3, i, 2)"),
        ("ping", "total = total + ping(4, i)"),
    ];
    for (name, statement) in shapes {
        let program = |iterations: usize| {
            format!("local total = 0 for i = 1, {iterations} do {statement} end return total")
        };
        // Warm the stack and the caches, then compare two lengths of loop:
        // whatever the loops allocate once (the chunk, the first growth of
        // the value stack) cancels out of the difference.
        measure(&mut lua, &program(100));
        let (small, _, _) = measure(&mut lua, &program(1_000));
        let (large, _, results) = measure(&mut lua, &program(101_000));
        assert_eq!(results.len(), 1);
        let per_hundred_thousand_calls = large.saturating_sub(small);
        println!("{name}: {small} allocations for 1000 calls, {large} for 101000");
        assert!(
            per_hundred_thousand_calls <= 8,
            "{name}: {per_hundred_thousand_calls} allocations for 100000 more calls"
        );
    }
}

#[test]
fn a_million_vararg_tail_calls_do_not_grow_the_value_stack() {
    let _turn = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut lua = Lua::builder()
        .profile(CompatibilityProfile::Lua54)
        .execution_mode(ExecutionMode::Interpreter)
        .build()
        .unwrap();
    lua.exec(
        r"
        function down(n, ...)
            if n == 0 then return select('#', ...) end
            return down(n - 1, ...)
        end
        function up(n, ...)
            if n == 0 then return select('#', ...) end
            return up(n - 1, n, ...)
        end
        ",
    )
    .unwrap();
    // A frame that grew by a vararg per hop would ask for a block of tens of
    // megabytes by a million hops.
    let (_, largest, results) = measure(&mut lua, "return down(1000000, 1, 2, 3)");
    assert_eq!(results, vec![Value::Integer(3)]);
    assert!(largest < 1 << 20, "largest block {largest}");
    // `up` keeps one more vararg per hop by design; 200 hops are 200 slots.
    let (_, largest, results) = measure(&mut lua, "return up(200, 1)");
    assert_eq!(results, vec![Value::Integer(201)]);
    assert!(largest < 1 << 20, "largest block {largest}");
}
