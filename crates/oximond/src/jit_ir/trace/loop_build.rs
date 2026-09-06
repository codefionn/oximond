//! Loop formation: turning one recorded iteration into a loop.
//!
//! A recorder sees one pass through the loop body. Every register it reads
//! before writing holds a value from the previous iteration, or from entry on
//! the first. This module decides how each such register travels:
//!
//! - A register the iteration writes and something reads from the previous
//!   iteration is *carried*: it becomes a loop parameter, initialised from an
//!   entry input, and the loop-back passes the iteration's final value for it.
//!   The loop-back arguments are all read before any parameter is assigned,
//!   so swaps (`a, b = b, a`) and aliased writes (`a, b = b, b`) are exact.
//!   Backends that lower parameters to block parameters get that for free.
//! - A register the iteration never writes, but operations read, is
//!   *invariant*: an entry input that every iteration shares. No loop-back
//!   move exists for it.
//! - Every other register needs nothing. Snapshots map an unwritten one to
//!   `Unchanged` and a written one to its value at that point.
//!
//! "Read by a snapshot" matters as much as "read by an operation": a guard
//! that fails before the iteration writes a register must restore the
//! previous iteration's value, so that value has to exist as an SSA value.

use super::ir::{TraceIr, TraceTerminator};

/// How the recorded iteration touches one register.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RegisterUse {
    /// The iteration writes the register.
    pub written: bool,
    /// An operation reads the register as the previous iteration left it.
    pub read_by_operations: bool,
    /// A snapshot maps the register to its value from before the iteration
    /// wrote it.
    pub read_by_snapshots: bool,
}

/// How a register crosses the loop-back.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoopRole {
    /// Nothing reads the previous iteration's value.
    Unused,
    /// An entry input shared by every iteration.
    Invariant,
    /// A loop parameter with a loop-back argument.
    Carried,
}

impl RegisterUse {
    /// The role this register plays in the loop.
    #[must_use]
    pub const fn role(self) -> LoopRole {
        let read = self.read_by_operations || self.read_by_snapshots;
        if self.written && read {
            LoopRole::Carried
        } else if !self.written && self.read_by_operations {
            LoopRole::Invariant
        } else {
            LoopRole::Unused
        }
    }

    /// Whether the register needs an entry input.
    #[must_use]
    pub const fn needs_entry_input(self) -> bool {
        !matches!(self.role(), LoopRole::Unused)
    }
}

/// Whether every loop-back argument has the type of its parameter. A register
/// that changes type across the iteration cannot ride in a typed parameter.
#[must_use]
pub fn loop_types_are_stable(ir: &TraceIr) -> bool {
    match &ir.terminator {
        TraceTerminator::LoopBack { args } => {
            args.len() == ir.loop_params.len()
                && args
                    .iter()
                    .zip(&ir.loop_params)
                    .all(|(arg, param)| ir.value_type(*arg) == Some(param.ty))
        }
        TraceTerminator::Exit { .. }
        | TraceTerminator::Link { .. }
        | TraceTerminator::LinkLoop { .. }
        | TraceTerminator::Return { .. }
        | TraceTerminator::TailCall { .. } => true,
    }
}
