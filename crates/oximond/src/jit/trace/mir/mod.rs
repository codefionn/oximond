//! The MIR backend: the default native backend of the Baseline and
//! Optimized tiers on x86-64 Linux and Windows, behind the same entry contract as the
//! Cranelift one (`emit.rs`).
//!
//! Those tiers try it first for every tree, unless
//! `OXIMOND_JIT_BACKEND=cranelift` keeps every tree on Cranelift. On other
//! targets it declines every tree. A tree it cannot compile (an operation it does not
//! lower, a shape it does not handle) is handed to Cranelift, with the
//! reason logged under `OXIMOND_JIT_TRACE` and counted. The Peak tier and
//! sides compiled apart always use Cranelift. `OXIMOND_JIT_BACKEND=mir-only`
//! turns all three off: a declined tree and a side compiled apart stay in
//! the interpreter, and no tree reaches the Peak tier.
//!
//! Trace IR is lowered straight into [`oximond_mir::ir::Func`] in one forward
//! pass ([`lower`]), and `oximond_mir` allocates registers and emits x86-64
//! in one backward pass. Every arena (the MIR function, the compiler's
//! buffers, the lowerer's tables) lives in a thread-local that is reused, so
//! a steady-state compile allocates nothing inside this module. The code goes
//! into one executable region for the whole process, carved per tree.
//!
//! See `docs/impl/native-backend/mir/backend.md` for the design and the
//! measurements.

#[cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]
mod lower;
#[cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]
mod support;
#[cfg(all(test, any(target_os = "linux", windows), target_arch = "x86_64"))]
pub(super) use lower::{FAST_LINKS, FOLDED_LOOPS, GUARDED_LINKS, adopted_keys};
#[cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]
pub(super) use support::{Block, Compiled, lower};
#[cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]
pub use support::{Counters, counters};

#[cfg(not(all(any(target_os = "linux", windows), target_arch = "x86_64")))]
mod unsupported;
#[cfg(not(all(any(target_os = "linux", windows), target_arch = "x86_64")))]
#[allow(unused_imports)]
pub(super) use unsupported::{Block, Compiled, lower};
#[cfg(not(all(any(target_os = "linux", windows), target_arch = "x86_64")))]
pub use unsupported::{Counters, counters};

#[cfg(all(test, any(target_os = "linux", windows), target_arch = "x86_64"))]
mod function_tests;
#[cfg(all(test, any(target_os = "linux", windows), target_arch = "x86_64"))]
mod link_tests;
#[cfg(all(test, any(target_os = "linux", windows), target_arch = "x86_64"))]
mod outer_tests;
#[cfg(all(test, any(target_os = "linux", windows), target_arch = "x86_64"))]
mod tests;
