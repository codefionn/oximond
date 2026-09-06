//! The optimized members a tree compile takes from the tree's last compile.
//!
//! A tree is compiled again whenever a side joins it, and again at a higher
//! tier, and every compile optimizes every member. The optimizer is a pure
//! function of the member, the level, the flags and what the member reads of
//! the tree's writes ([`TreeWrites::read_by`]), so a member equal in all four
//! to one of the tree's last compile takes that result. Anything else a
//! member's optimization depends on is in the member itself: the tree plan
//! writes the shared bindings, pins, window and entry types into every
//! member, and the parent a side continues into its `parent`, which the
//! continued guards of [`TreeWrites`] carry. The tier enters through the
//! level; Optimized and Peak optimize alike. Lowering comes after and runs
//! for every compile, so the target and its calling convention do not
//! matter here.
//!
//! A member that kept its optimization keeps its exit plan too
//! ([`plans`]), when the plan was made in the same [`PlanContext`]: the plan
//! reads only the optimized member and that context.
//!
//! The cache is per compile thread and bounded twice: it remembers at most
//! [`MAX_TREES`] trees, and at most [`MAX_WEIGHT`] bytes of IR and plans (see
//! [`weight`]), dropping the trees compiled longest ago first.

use super::compile::{MemberPlan, PlanContext};
use crate::compat::TracingOptFlags;
use crate::jit_ir::trace::{
    MemberWrites, OptimizeLevel, OptimizeStats, TraceId, TraceIr, TreeMember, TreeWrites,
    optimize_tree_member,
};
use std::cell::RefCell;
use std::sync::Arc;

/// The most trees the cache remembers.
const MAX_TREES: usize = 256;

/// The most bytes of IR the cache holds (see [`weight`]), the sources and
/// the results together. The results are shared with the compiled trees, so
/// while a tree's code lives, only its sources cost the cache memory of its
/// own.
const MAX_WEIGHT: usize = 32 << 20;

thread_local! {
    static CACHE: RefCell<OptimizedMembers> = const { RefCell::new(OptimizedMembers::new()) };
}

/// Optimizes the members `planned` of tree `id` with the compile thread's
/// cache (see [`OptimizedMembers::optimize_tree`]).
pub(super) fn optimize_tree(
    id: TraceId,
    planned: &[TreeMember],
    level: OptimizeLevel,
    flags: TracingOptFlags,
    writes: &TreeWrites,
    total: &mut OptimizeStats,
) -> Vec<Arc<TraceIr>> {
    CACHE.with_borrow_mut(|cache| cache.optimize_tree(id, planned, level, flags, writes, total))
}

/// The exit plans in `context` that the members of tree `id`, which
/// [`optimize_tree`] just optimized from `count` planned members, kept from
/// an earlier compile, by planned member. A plan is kept for the same held
/// registers (`held`, by planned member; see `held_registers` in
/// `compile.rs`) only.
pub(super) fn plans(
    id: TraceId,
    count: usize,
    context: PlanContext,
    held: &[&[u16]],
) -> Vec<Option<Arc<MemberPlan>>> {
    CACHE.with_borrow(|cache| match cache.trees.last() {
        Some((tree, members)) if *tree == id && members.len() == count => members
            .iter()
            .zip(held)
            .map(|(member, held)| match &member.plan {
                Some((made_in, made_with, plan))
                    if *made_in == context && **made_with == **held =>
                {
                    Some(Arc::clone(plan))
                }
                _ => None,
            })
            .collect(),
        _ => vec![None; count],
    })
}

/// Keeps the exit plans `plans`, made in `context` with the held registers
/// each names, with the members of tree `id` that [`optimize_tree`] just
/// optimized, by planned member.
pub(super) fn keep_plans(
    id: TraceId,
    context: PlanContext,
    plans: impl IntoIterator<Item = (usize, Box<[u16]>, Arc<MemberPlan>)>,
) {
    CACHE.with_borrow_mut(|cache| {
        let OptimizedMembers { trees, weight, .. } = cache;
        let Some((tree, members)) = trees.last_mut() else {
            return;
        };
        if *tree != id {
            return;
        }
        for (index, held, plan) in plans {
            let Some(member) = members.get_mut(index) else {
                continue;
            };
            if let Some((_, _, old)) = &member.plan {
                *weight -= old.weight();
                member.weight -= old.weight();
            }
            *weight += plan.weight();
            member.weight += plan.weight();
            member.plan = Some((context, held, plan));
        }
    });
}

/// How many member optimizations this thread's cache answered and how many
/// it ran.
#[cfg(test)]
pub(super) fn counts() -> (u64, u64) {
    CACHE.with_borrow(|cache| (cache.hits, cache.misses))
}

/// Forgets every tree this thread's cache holds.
#[cfg(test)]
pub(super) fn clear() {
    CACHE.with_borrow_mut(|cache| *cache = OptimizedMembers::new());
}

/// About the bytes `ir` holds: its values, instructions, guards and
/// snapshots, which are nearly all of a trace of any size.
fn weight(ir: &TraceIr) -> usize {
    use crate::jit_ir::trace::{
        GuardInfo, RegisterEntry, Snapshot, SnapshotFrame, TraceInstruction, ValueInfo,
    };
    use std::mem::size_of;
    size_of::<TraceIr>()
        + ir.values.capacity() * size_of::<ValueInfo>()
        + (ir.preheader.capacity() + ir.body.capacity()) * size_of::<TraceInstruction>()
        + ir.guards.capacity() * size_of::<GuardInfo>()
        + ir.snapshots
            .iter()
            .map(|snapshot| {
                size_of::<Snapshot>()
                    + snapshot
                        .frames
                        .iter()
                        .map(|frame| {
                            size_of::<SnapshotFrame>()
                                + frame.registers.len() * size_of::<RegisterEntry>()
                        })
                        .sum::<usize>()
            })
            .sum::<usize>()
}

/// One member of a tree as the optimizer got it and what it made of it.
pub(super) struct OptimizedMember {
    level: OptimizeLevel,
    flags: TracingOptFlags,
    writes: MemberWrites,
    source: TraceIr,
    /// Shared with the compiled tree, which keeps it as long as it lives.
    pub(super) optimized: Arc<TraceIr>,
    stats: OptimizeStats,
    /// The member's exit plan, once a compile made one, and what it was
    /// made in: the context and the member's held registers.
    plan: Option<(PlanContext, Box<[u16]>, Arc<MemberPlan>)>,
    weight: usize,
}

impl OptimizedMember {
    fn matches(
        &self,
        source: &TraceIr,
        level: OptimizeLevel,
        flags: TracingOptFlags,
        writes: &MemberWrites,
    ) -> bool {
        // The cheap tests first: most misses differ in size.
        self.level == level
            && self.flags == flags
            && self.source.id == source.id
            && self.source.body.len() == source.body.len()
            && self.source.values.len() == source.values.len()
            && self.source.snapshots.len() == source.snapshots.len()
            && self.source == *source
            && self.writes.optimizes_alike(level, &self.optimized, writes)
    }
}

/// The optimized members of the trees compiled last, by root trace id,
/// least recently compiled first.
pub(super) struct OptimizedMembers {
    pub(super) trees: Vec<(TraceId, Vec<OptimizedMember>)>,
    /// The sum of the members' weights.
    weight: usize,
    hits: u64,
    misses: u64,
}

impl OptimizedMembers {
    pub(super) const fn new() -> Self {
        Self {
            trees: Vec::new(),
            weight: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// The members of `planned`, the plan of tree `id`, optimized at `level`
    /// with `flags`, each reading `writes` for itself. A member the tree's
    /// last compile optimized from the same input takes that result. Adds
    /// what the optimizer changed to `total`.
    pub(super) fn optimize_tree(
        &mut self,
        id: TraceId,
        planned: &[TreeMember],
        level: OptimizeLevel,
        flags: TracingOptFlags,
        writes: &TreeWrites,
        total: &mut OptimizeStats,
    ) -> Vec<Arc<TraceIr>> {
        let mut previous = match self.trees.iter().position(|(tree, _)| *tree == id) {
            Some(position) => self.trees.remove(position).1,
            None => Vec::new(),
        };
        self.weight -= previous.iter().map(|member| member.weight).sum::<usize>();
        let mut kept = Vec::with_capacity(planned.len());
        let mut optimized = Vec::with_capacity(planned.len());
        for (index, member) in planned.iter().enumerate() {
            let writes = writes.for_member(index);
            let read = writes.read_by(&member.ir);
            let known = previous
                .iter()
                .position(|known| known.matches(&member.ir, level, flags, &read));
            let entry = if let Some(position) = known {
                self.hits += 1;
                let entry = previous.swap_remove(position);
                // Debug builds, the tests among them, check every reuse
                // against what the optimizer makes now.
                debug_assert!(
                    optimize_tree_member(&member.ir, level, flags, &writes)
                        == ((*entry.optimized).clone(), entry.stats),
                    "the cached optimization of member {index} of tree {id:?} differs from a \
                     fresh one",
                );
                entry
            } else {
                self.misses += 1;
                let (ir, stats) = optimize_tree_member(&member.ir, level, flags, &writes);
                OptimizedMember {
                    level,
                    flags,
                    writes: read,
                    weight: weight(&member.ir) + weight(&ir),
                    source: member.ir.clone(),
                    optimized: Arc::new(ir),
                    stats,
                    plan: None,
                }
            };
            total.accumulate(entry.stats);
            optimized.push(Arc::clone(&entry.optimized));
            kept.push(entry);
        }
        let added: usize = kept.iter().map(|member| member.weight).sum();
        // A tree too big to keep is not kept, and does not push out the rest.
        if added <= MAX_WEIGHT / 4 {
            while !self.trees.is_empty()
                && (self.trees.len() >= MAX_TREES || self.weight + added > MAX_WEIGHT)
            {
                let (_, dropped) = self.trees.remove(0);
                self.weight -= dropped.iter().map(|member| member.weight).sum::<usize>();
            }
            self.weight += added;
            self.trees.push((id, kept));
        }
        optimized
    }

    /// How many member optimizations the cache answered and how many it
    /// ran.
    #[cfg(test)]
    pub(super) const fn counts(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }
}
