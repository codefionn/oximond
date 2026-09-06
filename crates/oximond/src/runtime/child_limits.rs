//! Resource limits that shell children inherit.
//!
//! The CLI raises its own soft `RLIMIT_STACK` so the main thread can hold the
//! interpreter's deep recursion. Stock Lua leaves the limit alone, so a
//! command started by `os.execute` or `io.popen` must see the limit the
//! process was started with. The CLI records that limit here and every shell
//! child restores it before `exec`.
#![allow(unsafe_code)]

use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static RECORDED: AtomicBool = AtomicBool::new(false);
static SOFT: AtomicU64 = AtomicU64::new(0);
static HARD: AtomicU64 = AtomicU64::new(0);

/// Records the `RLIMIT_STACK` the process was started with. Children started
/// through [`restore_in_child`] get this limit back.
pub fn record_inherited_stack_limit(soft: u64, hard: u64) {
    SOFT.store(soft, Ordering::Relaxed);
    HARD.store(hard, Ordering::Relaxed);
    RECORDED.store(true, Ordering::Release);
}

/// Makes `command` restore the recorded stack limit in the child before it
/// runs. Does nothing when no limit was recorded.
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
pub fn restore_in_child(command: &mut Command) {
    use std::os::unix::process::CommandExt;

    #[repr(C)]
    struct Rlimit {
        current: u64,
        maximum: u64,
    }
    unsafe extern "C" {
        fn setrlimit(resource: i32, limit: *const Rlimit) -> i32;
    }
    const RLIMIT_STACK: i32 = 3;
    if !RECORDED.load(Ordering::Acquire) {
        return;
    }
    let limit = Rlimit {
        current: SOFT.load(Ordering::Relaxed),
        maximum: HARD.load(Ordering::Relaxed),
    };
    // SAFETY: the closure only calls setrlimit, which is async-signal-safe,
    // on a plain struct it owns. A failure is ignored: the child then keeps
    // the raised limit, which is harmless.
    unsafe {
        command.pre_exec(move || {
            setrlimit(RLIMIT_STACK, &raw const limit);
            Ok(())
        });
    }
}

#[cfg(not(all(target_os = "linux", target_pointer_width = "64")))]
pub fn restore_in_child(_command: &mut Command) {}
