//! Process entry and stack setup for the Linux build.
//!
//! The Rust runtime's startup reads `/proc/self/maps` and parses it with
//! `sscanf` to find the main thread's stack guard, installs signal handlers,
//! and polls the standard descriptors. That is about 100000 instructions and
//! several system calls on every start. The CLI defines the C `main` itself
//! instead and does the one piece of that setup it needs: ignoring SIGPIPE,
//! so a write to a closed pipe is an I/O error, as it is in the Rust runtime.
//! The cost is that a real native stack overflow ends in a plain SIGSEGV
//! instead of the runtime's message. Lua recursion hits its depth limits first.
//! Closed descriptors 0-2 are no longer reopened on `/dev/null`, as in stock
//! Lua. A panic is caught here and exits with status 101, as the runtime's
//! `main` does.
#![allow(unsafe_code)]

/// Makes the main thread's stack at least `wanted` bytes deep by raising the
/// soft `RLIMIT_STACK`. Linux checks that limit each time the main stack
/// grows, and the kernel leaves at least 128 MiB between the stack and the
/// mmap area at exec, so the raise takes effect in this process. Returns
/// false when the hard limit forbids it or the platform has no such limit, and
/// the caller then runs on a thread with its own stack. Running on the main
/// thread skips a clone, a 64 MiB mmap, and a second malloc arena.
#[cfg(all(not(test), target_os = "linux", target_pointer_width = "64"))]
pub(crate) fn main_is_deep_enough(wanted: usize) -> bool {
    #[repr(C)]
    struct Rlimit {
        current: u64,
        maximum: u64,
    }
    const RLIMIT_STACK: i32 = 3;
    unsafe extern "C" {
        fn getrlimit(resource: i32, limit: *mut Rlimit) -> i32;
        fn setrlimit(resource: i32, limit: *const Rlimit) -> i32;
    }
    let wanted = wanted as u64;
    let mut limit = Rlimit {
        current: 0,
        maximum: 0,
    };
    // SAFETY: `limit` is a valid, writable `struct rlimit` for the 64-bit
    // Linux ABI, and getrlimit only writes it.
    if unsafe { getrlimit(RLIMIT_STACK, &raw mut limit) } != 0 {
        return false;
    }
    if limit.current >= wanted {
        return true;
    }
    if limit.maximum < wanted {
        return false;
    }
    // Shell children get the original limit back, as under stock Lua.
    oximond::runtime::child_limits::record_inherited_stack_limit(limit.current, limit.maximum);
    limit.current = wanted;
    // SAFETY: `limit` is a valid `struct rlimit`, and setrlimit only reads
    // it. Raising the soft limit within the hard limit is always allowed.
    unsafe { setrlimit(RLIMIT_STACK, &raw const limit) == 0 }
}

#[cfg(all(not(test), not(all(target_os = "linux", target_pointer_width = "64"))))]
pub(crate) fn main_is_deep_enough(_wanted: usize) -> bool {
    false
}

/// Makes glibc extend the heap by at least 4 MiB at a time instead of
/// 128 KiB. With transparent huge pages enabled, an extension that spans a
/// 2 MiB aligned range can be backed by one huge page on first touch, so
/// the first trace compiles fault far less. At scale 3, algorithm/nbody
/// tracing went from 499 to about 300 page faults and `string/utf8_pack`
/// from 598 to 410-480, with cycles:u 1% lower and the interpreter and
/// startup/empty unchanged. The count varies with how many huge pages the
/// kernel has free. Without huge pages this changes nothing: a 1 MiB pad
/// and the mmap and trim thresholds left the faults where they were. The
/// resident size stayed at 14 MiB.
#[cfg(all(not(test), target_os = "linux", target_env = "gnu"))]
fn grow_the_heap_in_large_steps() {
    unsafe extern "C" {
        fn mallopt(param: i32, value: i32) -> i32;
    }
    const M_TOP_PAD: i32 = -2;
    // SAFETY: mallopt takes plain integers and only sets an allocator
    // parameter under the main arena's lock. No other thread exists yet.
    unsafe { mallopt(M_TOP_PAD, 4 << 20) };
}

/// The C entry point. `std::env::args_os` still works under `no_main` on
/// glibc, which passes the arguments to the runtime's `.init_array` hook.
#[cfg(all(not(test), target_os = "linux"))]
#[unsafe(no_mangle)]
pub extern "C" fn main(_argc: i32, _argv: *const *const u8) -> i32 {
    unsafe extern "C" {
        fn signal(signum: i32, handler: usize) -> usize;
    }
    const SIGPIPE: i32 = 13;
    const SIG_IGN: usize = 1;
    // SAFETY: ignoring SIGPIPE is valid and touches no Rust state.
    unsafe { signal(SIGPIPE, SIG_IGN) };
    #[cfg(target_env = "gnu")]
    grow_the_heap_in_large_steps();
    // Unwinding out of an `extern "C"` function aborts. The panic hook has
    // already printed the message, and a panic on the interpreter thread
    // comes here through `resume_unwind`.
    std::panic::catch_unwind(crate::real_main_code).unwrap_or(101)
}
