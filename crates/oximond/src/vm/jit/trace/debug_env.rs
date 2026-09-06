//! The JIT debug environment variables, read once per process.
//!
//! `OXIMOND_JIT_TRACE` and `OXIMOND_JIT_TRACE_IR` gate diagnostics on the
//! recording, abort and compile paths, which run for every recording of every
//! loop. Reading the environment there costs a scan of `environ` each time,
//! so each variable is read on its first use and the answer is kept. Setting
//! one later in the same process has no effect.

use std::sync::OnceLock;

/// Whether `OXIMOND_JIT_TRACE` is set: report recordings, aborts, compiles and
/// side traces on stderr.
pub(crate) fn trace() -> bool {
    static SET: OnceLock<bool> = OnceLock::new();
    *SET.get_or_init(|| std::env::var_os("OXIMOND_JIT_TRACE").is_some())
}

/// `OXIMOND_JIT_SIDE_PHASE=N` (0 when unset): a guard records its side at
/// failures `SIDE_EXIT_THRESHOLD - N`, `2 * SIDE_EXIT_THRESHOLD - N` and so
/// on instead of at whole multiples. Which iteration a side records decides
/// its shape (a row's last cell or one in the middle), and under
/// `OXIMOND_JIT_SYNC` the phase is the only thing that moves it, so a sweep
/// over `N` reproduces the shapes a background compile lands on by chance.
pub(crate) fn side_phase() -> u32 {
    static PHASE: OnceLock<u32> = OnceLock::new();
    *PHASE.get_or_init(|| {
        std::env::var("OXIMOND_JIT_SIDE_PHASE")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0)
    })
}

/// Whether `OXIMOND_JIT_TRACE_IR` is set: print the recorded IR of every trace.
pub(crate) fn trace_ir() -> bool {
    static SET: OnceLock<bool> = OnceLock::new();
    *SET.get_or_init(|| std::env::var_os("OXIMOND_JIT_TRACE_IR").is_some())
}

/// Whether `OXIMOND_JIT_STATS` is set: compiled code counts its jumps into side
/// traces.
pub(crate) fn stats() -> bool {
    static SET: OnceLock<bool> = OnceLock::new();
    *SET.get_or_init(|| std::env::var_os("OXIMOND_JIT_STATS").is_some())
}

/// Whether `OXIMOND_JIT_SELECTS=0` is set: every branch on a boolean keeps
/// its guard, as before value selects (for measuring what they change).
pub(crate) fn selects_disabled() -> bool {
    static SET: OnceLock<bool> = OnceLock::new();
    *SET.get_or_init(|| std::env::var_os("OXIMOND_JIT_SELECTS").is_some_and(|value| value == "0"))
}

/// Whether `OXIMOND_JIT_DETACH=0` is set: every side joins its tree in a
/// compile of the whole tree, as before sides could be compiled apart.
pub(crate) fn detach_disabled() -> bool {
    static SET: OnceLock<bool> = OnceLock::new();
    *SET.get_or_init(|| std::env::var_os("OXIMOND_JIT_DETACH").is_some_and(|value| value == "0"))
}

/// Whether a tree the MIR backend compiled takes its sides in a compile of
/// the whole tree where that costs less than compiling them apart (see
/// `InstalledTrace::joins_for_less`). On unless `OXIMOND_JIT_JOIN=0` is set,
/// for measuring what the rule changes, or the calling test asked for its
/// sides apart.
pub(crate) fn joins_small_trees() -> bool {
    #[cfg(test)]
    if crate::jit::debug_env::SIDES_APART.with(std::cell::Cell::get) {
        return false;
    }
    static SET: OnceLock<bool> = OnceLock::new();
    *SET.get_or_init(|| std::env::var_os("OXIMOND_JIT_JOIN").is_none_or(|value| value != "0"))
}

/// Whether a site lists the registers its entries found nil on some entries
/// and a number on others, for its next root to take untyped. On unless
/// `OXIMOND_JIT_UNTYPED=0` is set: eight Mineclonia emerges each way took
/// 144.6 G with it against 146.3 G without (see the notes in `mod.rs`).
pub(crate) fn untyped_enabled() -> bool {
    static SET: OnceLock<bool> = OnceLock::new();
    *SET.get_or_init(|| std::env::var_os("OXIMOND_JIT_UNTYPED").is_none_or(|value| value != "0"))
}

/// ` at source:line` of `pc` in `chunk`, for the diagnostics that name a
/// loop by its header pc: two functions of a program often share one.
pub(crate) fn at_line(chunk: &crate::bytecode::Chunk, pc: usize) -> String {
    chunk.debug.line_at(pc).map_or_else(String::new, |line| {
        let source = String::from_utf8_lossy(chunk.debug.source_name().as_bytes()).into_owned();
        let source = source.strip_prefix(['@', '=']).unwrap_or(&source);
        format!(" at {source}:{line}")
    })
}
