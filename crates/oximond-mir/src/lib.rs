//! Prototype of a target-neutral machine IR (MIR) whose design goal is time
//! to execution: a dense IR, one backward pass that allocates registers and
//! emits x86-64 together, and a reference evaluator to check it against.
//!
//! This is a measurement tool. It covers the ops its kernels need and is not
//! wired into the oximond runtime.

pub mod buf;
pub mod eval;
#[cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]
pub mod exec;
pub mod ir;
pub mod kernels;
pub mod ra;
pub mod randgen;
pub mod rng;
pub mod target;
pub mod x64;

/// Result of the entry for an exit: `-1 - index`, as `exit_status` in
/// `crates/oximond/src/jit/trace/emit.rs`.
#[must_use]
pub const fn exit_status(index: u32) -> i64 {
    -1 - index as i64
}

/// Side-link codes of the current ABI, not used by the prototype.
pub const LINK_CODE: i64 = i64::MIN + 1;
pub const DECLINE_CODE: i64 = i64::MIN + 2;
