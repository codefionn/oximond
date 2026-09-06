//! User-space instructions the calling thread has retired, for the compile
//! measurements of `docs/impl/native-backend/mir/backend.md`.
//!
//! With `OXIMOND_JIT_ICOUNT` set, each compile job, and each tree compile's
//! optimizer (`optimize`), exit plan (`plan`) and native compile (`tree`),
//! print a `jit-icount:` line with the instructions their thread retired,
//! which no whole-process counter can split from the VM thread's.
//! Without it nothing here runs. The count comes from `perf_event_open`
//! (`instructions:u`, this thread, no kernel), so it is exact and does not
//! move with load.

/// Instructions retired in user space by the calling thread so far, or 0 when
/// the system gives no counter.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[allow(unsafe_code)]
pub(crate) fn thread_instructions() -> u64 {
    use std::cell::Cell;
    use std::ffi::{c_int, c_long, c_void};

    unsafe extern "C" {
        fn syscall(number: c_long, ...) -> c_long;
        fn read(fd: c_int, buffer: *mut c_void, count: usize) -> isize;
        fn close(fd: c_int) -> c_int;
    }

    /// The thread's counter, opened on first use and closed with the thread.
    struct Counter(c_int);
    impl Drop for Counter {
        fn drop(&mut self) {
            if self.0 >= 0 {
                // SAFETY: closes a descriptor this thread opened, once.
                unsafe { close(self.0) };
            }
        }
    }

    thread_local! {
        static COUNTER: Cell<Option<c_int>> = const { Cell::new(None) };
        static GUARD: Cell<Option<Counter>> = const { Cell::new(None) };
    }

    let fd = COUNTER.with(|slot| {
        if let Some(fd) = slot.get() {
            return fd;
        }
        // `struct perf_event_attr`: type 0 (hardware), size 128, config 1
        // (instructions); the flag word has exclude_kernel and exclude_hv.
        let mut attr = [0_u64; 16];
        attr[0] = 128 << 32;
        attr[1] = 1;
        attr[5] = (1 << 5) | (1 << 6);
        // SAFETY: `perf_event_open` (298 on x86-64) reads `attr`, which
        // is a valid attribute block of the size it names; pid 0 and cpu -1
        // count the calling thread wherever it runs.
        let fd = unsafe {
            syscall(
                298,
                attr.as_ptr(),
                0 as c_int,
                -1 as c_int,
                -1 as c_int,
                0 as c_long,
            )
        };
        let fd = c_int::try_from(fd).unwrap_or(-1);
        slot.set(Some(fd));
        GUARD.with(|guard| guard.set(Some(Counter(fd))));
        fd
    });
    if fd < 0 {
        return 0;
    }
    let mut count = 0_u64;
    // SAFETY: reads eight bytes into `count`, which has room for them.
    let got = unsafe { read(fd, std::ptr::addr_of_mut!(count).cast(), 8) };
    if got == 8 { count } else { 0 }
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
pub(crate) fn thread_instructions() -> u64 {
    0
}

/// One phase of the trace pipeline being measured (see [`scope`]).
pub(crate) struct Scope {
    /// The phase and the thread's count when it began, with the instructions
    /// of the phases nested in it so far. `None` when nothing is measured.
    open: Option<(&'static str, u64)>,
}

thread_local! {
    /// The instructions of the phases nested in each open scope, innermost
    /// last.
    static NESTED: std::cell::RefCell<Vec<u64>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The instructions and the count of scopes each phase took, over all
/// threads, printed when the process exits.
static PHASES: std::sync::Mutex<Vec<(&'static str, u64, u64)>> = std::sync::Mutex::new(Vec::new());

/// With `OXIMOND_JIT_ICOUNT` set, counts the instructions this thread retires
/// until the returned scope is dropped toward the phase `name`, leaving out
/// those of the scopes opened inside it. The process prints one line per
/// phase when it exits: `jit-icount: phase backend={name} ir={scopes}
/// instructions={n}`. Without the variable nothing here runs.
#[allow(unsafe_code)]
pub(crate) fn scope(name: &'static str) -> Scope {
    if !super::debug_env::icount() {
        return Scope { open: None };
    }
    static REPORT: std::sync::Once = std::sync::Once::new();
    REPORT.call_once(|| {
        unsafe extern "C" {
            fn atexit(callback: extern "C" fn()) -> std::ffi::c_int;
        }
        extern "C" fn report() {
            if let Ok(phases) = PHASES.lock() {
                for (name, instructions, scopes) in phases.iter() {
                    eprintln!(
                        "jit-icount: phase backend={name} ir={scopes} instructions={instructions}"
                    );
                }
            }
        }
        // SAFETY: registers a function that takes no arguments and only
        // reads a static, to run once at process exit.
        unsafe {
            atexit(report);
        }
    });
    NESTED.with_borrow_mut(|nested| nested.push(0));
    Scope {
        open: Some((name, thread_instructions())),
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        let Some((name, began)) = self.open else {
            return;
        };
        let spent = thread_instructions().saturating_sub(began);
        let inner = NESTED.with_borrow_mut(|nested| {
            let inner = nested.pop().unwrap_or(0);
            if let Some(outer) = nested.last_mut() {
                *outer += spent;
            }
            inner
        });
        if let Ok(mut phases) = PHASES.lock() {
            let own = spent.saturating_sub(inner);
            match phases.iter_mut().find(|(known, _, _)| *known == name) {
                Some((_, instructions, scopes)) => {
                    *instructions += own;
                    *scopes += 1;
                }
                None => phases.push((name, own, 1)),
            }
        }
    }
}

/// Runs `run`, and with `OXIMOND_JIT_ICOUNT` set prints the instructions this
/// thread retired in it: `jit-icount: {what} backend={name} ir={ir}
/// instructions={n}`. `name` says which backend the result came from.
pub(crate) fn around<T>(
    what: &str,
    ir: usize,
    run: impl FnOnce() -> T,
    name: impl FnOnce(&T) -> &'static str,
) -> T {
    if !super::debug_env::icount() {
        return run();
    }
    let before = thread_instructions();
    let result = run();
    let spent = thread_instructions().saturating_sub(before);
    eprintln!(
        "jit-icount: {what} backend={} ir={ir} instructions={spent}",
        name(&result)
    );
    result
}
