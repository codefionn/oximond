#![allow(dead_code)]
//! Targets without the MIR backend: it never accepts a tree.

use super::super::compile::{ExitPlan, LoweredMember, TraceCompileOptions};
use crate::jit::OptimizationTier;

/// Code the backend made. There is none on this target.
pub(in super::super) struct Block;

pub(in super::super) struct Compiled {
    pub(in super::super) block: Block,
    pub(in super::super) code_size: usize,
    pub(in super::super) insts: usize,
    pub(in super::super) versioned: bool,
    pub(in super::super) entry: usize,
}

impl Block {
    pub(in super::super) fn address(&self) -> usize {
        0
    }
}

/// What the backend has accepted and refused so far.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Counters {
    pub accepted_trees: u64,
    pub accepted_ir: u64,
    pub fallback_trees: u64,
    pub fallback_ir: u64,
    pub reasons: std::collections::BTreeMap<&'static str, (u64, u64)>,
}

#[must_use]
pub fn counters() -> Counters {
    Counters::default()
}

pub(in super::super) fn lower(
    _members: &[LoweredMember<'_>],
    _plan: &ExitPlan,
    _tier: OptimizationTier,
    _forced_guard: Option<(usize, usize)>,
    _options: TraceCompileOptions,
) -> Option<Compiled> {
    None
}
