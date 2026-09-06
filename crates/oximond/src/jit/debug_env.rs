//! The JIT debug environment variables the compiler reads, once per process.
//!
//! A compile runs for every trace and tree, so each variable is read on its
//! first use and the answer is kept. Setting one later in the same process
//! has no effect.

use std::sync::OnceLock;

/// Whether `OXIMOND_JIT_TRACE` is set: report compiles and rejections on
/// stderr.
pub(crate) fn trace() -> bool {
    static SET: OnceLock<bool> = OnceLock::new();
    *SET.get_or_init(|| std::env::var_os("OXIMOND_JIT_TRACE").is_some())
}

/// Whether `OXIMOND_JIT_NO_VMCALL` is set: a trace does not call a host C
/// function itself, so the recording of a loop that calls one aborts at the
/// call. For comparisons.
pub(crate) fn no_vmcall() -> bool {
    static SET: OnceLock<bool> = OnceLock::new();
    *SET.get_or_init(|| std::env::var_os("OXIMOND_JIT_NO_VMCALL").is_some())
}

/// `OXIMOND_JIT_VMCALL_MIN_GUEST`: the guest instructions a loop needs for
/// its trace to make host C calls itself, in place of the built-in bar. Meant
/// for measuring the bar.
pub(crate) fn vmcall_min_guest() -> Option<u32> {
    static BAR: OnceLock<Option<u32>> = OnceLock::new();
    *BAR.get_or_init(|| {
        std::env::var("OXIMOND_JIT_VMCALL_MIN_GUEST")
            .ok()
            .and_then(|value| value.parse().ok())
    })
}

/// Whether `OXIMOND_JIT_TRACE_IR_OPT` is set: print the optimized IR of every
/// tree.
pub(crate) fn trace_ir_opt() -> bool {
    static SET: OnceLock<bool> = OnceLock::new();
    *SET.get_or_init(|| std::env::var_os("OXIMOND_JIT_TRACE_IR_OPT").is_some())
}

/// Whether `OXIMOND_JIT_DISASM` is set: print the CLIF and the machine code of
/// every trace.
pub(crate) fn disasm() -> bool {
    static SET: OnceLock<bool> = OnceLock::new();
    *SET.get_or_init(|| std::env::var_os("OXIMOND_JIT_DISASM").is_some())
}

/// Whether `OXIMOND_JIT_VERIFY` is set to anything but `0` or the empty
/// string: run Cranelift's verifier in release builds too, which only debug
/// builds do by default. A function it refuses is rejected with the
/// verifier's message (under `OXIMOND_JIT_TRACE`) instead of reaching the
/// register allocator, where malformed code can panic or miscompile.
pub(crate) fn verify() -> bool {
    static SET: OnceLock<bool> = OnceLock::new();
    *SET.get_or_init(|| {
        std::env::var_os("OXIMOND_JIT_VERIFY")
            .is_some_and(|value| !value.is_empty() && value != "0")
    })
}

/// Whether Cranelift's verifier runs: always in debug builds, and in release
/// builds when [`verify`] says so.
pub(crate) fn cranelift_verifier() -> &'static str {
    if cfg!(debug_assertions) || verify() {
        "true"
    } else {
        "false"
    }
}

/// Whether `OXIMOND_JIT_CENSUS` is set: print one `jit-census:` JSON line per
/// native compile with the trace IR statistics and the CLIF each operation
/// lowered to. Off by default; the lowerer then pays one load per operation.
pub(crate) fn census() -> bool {
    static SET: OnceLock<bool> = OnceLock::new();
    *SET.get_or_init(|| std::env::var_os("OXIMOND_JIT_CENSUS").is_some())
}

/// Whether `OXIMOND_JIT_ICOUNT` is set: print the user-space instructions the
/// compile thread retired for each compile (see `icount`).
pub(crate) fn icount() -> bool {
    static SET: OnceLock<bool> = OnceLock::new();
    *SET.get_or_init(|| std::env::var_os("OXIMOND_JIT_ICOUNT").is_some())
}

/// Whether the Baseline and Optimized tiers try the MIR backend before
/// Cranelift: yes when `OXIMOND_JIT_BACKEND` is unset, empty, `mir` or
/// `mir-only`. `cranelift`, and any other value, keeps Cranelift for every
/// tree. Peak and sides compiled apart from their tree stay on Cranelift, as
/// does every tree MIR declines, unless [`backend_mir_only`] holds.
pub(crate) fn backend_mir() -> bool {
    static MIR: OnceLock<bool> = OnceLock::new();
    #[cfg(test)]
    if let Some(forced) = FORCE_BACKEND.with(std::cell::Cell::get) {
        return forced;
    }
    *MIR.get_or_init(|| {
        std::env::var_os("OXIMOND_JIT_BACKEND")
            .is_none_or(|value| value.is_empty() || value == "mir" || value == "mir-only")
    })
}

/// Whether `OXIMOND_JIT_BACKEND` is `mir-only`: MIR compiles every tree it
/// takes and nothing goes to Cranelift. A tree MIR declines and a side
/// compiled apart are rejected and stay in the interpreter, and no tree
/// moves up to the Peak tier. For differential testing of MIR on its own.
pub(crate) fn backend_mir_only() -> bool {
    static ONLY: OnceLock<bool> = OnceLock::new();
    #[cfg(test)]
    if FORCE_BACKEND.with(std::cell::Cell::get).is_some() {
        return false;
    }
    *ONLY.get_or_init(|| {
        std::env::var_os("OXIMOND_JIT_BACKEND").is_some_and(|value| value == "mir-only")
    })
}

/// `OXIMOND_JIT_PEAK`: `0` keeps every tree off the Peak tier and any other
/// value sends every tree there that has run long enough, the code MIR made
/// included. Unset or empty leaves the choice to `InstalledTrace::wants_peak`,
/// which keeps a tree MIR compiled as it is.
/// For measuring what the tier costs and gains.
pub(crate) fn peak() -> Option<bool> {
    #[cfg(test)]
    if let Some(forced) = FORCE_PEAK.with(std::cell::Cell::get) {
        return Some(forced);
    }
    static FORCED: OnceLock<Option<bool>> = OnceLock::new();
    *FORCED.get_or_init(|| {
        std::env::var_os("OXIMOND_JIT_PEAK")
            .filter(|value| !value.is_empty())
            .map(|value| value != "0")
    })
}

/// Whether the MIR backend probes the string index of the run in line (see
/// `mir/lower/probe.rs`): on unless `OXIMOND_JIT_MIR_PROBE` is `0` or the
/// empty string, which makes MIR decline the trees that probe and leaves them
/// to Cranelift (see "String-index probe" in
/// `docs/impl/native-backend/mir/backend.md`). Read once per process.
pub(crate) fn mir_probe() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    #[cfg(test)]
    if let Some(forced) = FORCE_MIR_PROBE.with(std::cell::Cell::get) {
        return forced;
    }
    *ON.get_or_init(|| {
        std::env::var_os("OXIMOND_JIT_MIR_PROBE")
            .is_none_or(|value| !value.is_empty() && value != "0")
    })
}

/// Lets the MIR backend probe the string index for the compiles of the
/// calling test's thread, as the default does, whatever
/// `OXIMOND_JIT_MIR_PROBE` says, for a test that compares MIR's probe with
/// Cranelift's. Each test runs on a thread of its own, so nothing resets it.
#[cfg(test)]
pub(crate) fn mir_probe_for_this_test() {
    FORCE_MIR_PROBE.with(|force| force.set(Some(true)));
}

/// Keeps the MIR backend from probing the string index for the compiles of
/// the calling test's thread, as `OXIMOND_JIT_MIR_PROBE=0` does.
#[cfg(test)]
pub(crate) fn mir_probe_off_for_this_test() {
    FORCE_MIR_PROBE.with(|force| force.set(Some(false)));
}

/// Keeps the compiles of the calling test's thread on Cranelift, for a test
/// that asserts what Cranelift emits (CLIF counts, the native link to sides
/// compiled apart). Each test runs on a thread of its own, so nothing resets
/// it.
#[cfg(test)]
pub(crate) fn cranelift_for_this_test() {
    FORCE_BACKEND.with(|force| force.set(Some(false)));
}

/// Keeps the sides of the trees the calling test's thread runs compiled
/// apart wherever the policy allowed that before small trees took their
/// sides in, for a test whose subject is a side compiled apart. Each test
/// runs on a thread of its own, so nothing resets it.
#[cfg(test)]
pub(crate) fn sides_apart_for_this_test() {
    SIDES_APART.with(|apart| apart.set(true));
}

#[cfg(test)]
thread_local! {
    /// Set by [`sides_apart_for_this_test`].
    pub(crate) static SIDES_APART: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Sends every tree the calling test's thread runs long enough to the Peak
/// tier, as `OXIMOND_JIT_PEAK=1` does, for a test of what the Peak compile
/// keeps of a tree MIR compiled.
#[cfg(test)]
pub(crate) fn peak_for_this_test() {
    FORCE_PEAK.with(|force| force.set(Some(true)));
}

#[cfg(test)]
thread_local! {
    /// The Peak policy of a test's own thread, see [`peak`].
    pub(crate) static FORCE_PEAK: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
thread_local! {
    /// Lets a test pick the backend for the compiles of its own thread,
    /// whatever the environment says.
    pub(crate) static FORCE_BACKEND: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
thread_local! {
    /// Turns the MIR probe on or off for the compiles of a test's own thread.
    static FORCE_MIR_PROBE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}
