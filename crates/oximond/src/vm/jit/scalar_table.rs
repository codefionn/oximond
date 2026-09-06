//! Allocation-free execution of verified, non-escaping table fragments.

use crate::vm::stack::FrameMut;
use rustc_hash::FxHashMap;
use std::sync::Arc;

use crate::bytecode::Chunk;
use crate::compat::CompatibilityProfile;
use crate::jit_ir::{
    ScalarNumberOperation, ScalarTableOutput, ScalarTablePlan, validate_scalar_table_plan,
};
use crate::runtime::Value;

use crate::vm::Vm;
use crate::vm::arith::constant_value;

/// Scratch numbers a plan evaluates on the stack before spilling to the heap.
const SCALAR_INLINE_NUMBERS: usize = 16;
/// Longest number program a plan may carry.
const SCALAR_MAX_NUMBERS: usize = 256;

type CachedPlan = (Arc<Chunk>, Option<Arc<ScalarTablePlan>>);
type Site = (usize, usize);
type HotPlan = (Site, Option<Arc<ScalarTablePlan>>);

/// Hot entries of the plan cache, direct-mapped by site. A loop body
/// builds a few tables per iteration; each site keeps its own entry, so
/// they do not evict one another into the map lookup.
const HOT_ENTRIES: usize = 8;

#[derive(Default)]
pub(crate) struct PlanCache {
    plans: FxHashMap<Site, CachedPlan>,
    hot: [Option<HotPlan>; HOT_ENTRIES],
}

/// The hot entry `site` maps to.
#[inline(always)]
fn hot_slot((chunk, pc): Site) -> usize {
    (pc ^ (chunk >> 4)) % HOT_ENTRIES
}

impl PlanCache {
    /// Whether a recent visit found no plan for the `NewTable` at `pc` of
    /// `chunk`, so [`Vm::try_scalar_table_fragment`] would decline it too.
    /// Only the hot entries answer; anything else is unknown (`false`).
    #[inline(always)]
    pub(crate) fn known_absent(&self, chunk: &Arc<Chunk>, pc: usize) -> bool {
        self.known_absent_at(Arc::as_ptr(chunk) as usize, pc)
    }

    /// [`Self::known_absent`] keyed by the chunk's raw address, for callers
    /// that hold a `&Chunk` rather than an `Arc<Chunk>`.
    #[inline(always)]
    pub(crate) fn known_absent_at(&self, chunk: usize, pc: usize) -> bool {
        let site = (chunk, pc);
        matches!(&self.hot[hot_slot(site)], Some((hot, None)) if *hot == site)
    }
}

impl Vm {
    pub(crate) fn try_scalar_table_fragment(&mut self, frame: &mut FrameMut<'_>) -> bool {
        if self.compatibility != CompatibilityProfile::Lua51 || !self.debug_hooks.is_empty() {
            return false;
        }
        let start_pc = frame.pc - 1;
        let key = (frame.chunk.address() as usize, start_pc);
        let cache = &mut self.jit.scalar_table_plans;
        let hot_index = hot_slot(key);
        if cache.hot[hot_index]
            .as_ref()
            .is_none_or(|(site, _)| *site != key)
        {
            if cache.plans.len() >= 1024 && !cache.plans.contains_key(&key) {
                // The map owns each chunk. Invalidate address-based hot keys
                // before dropping those owners can allow an address reuse.
                cache.hot = Default::default();
                cache.plans.clear();
            }
            let (_, plan) = cache.plans.entry(key).or_insert_with(|| {
                let plan = validate_scalar_table_plan(
                    &frame.chunk,
                    start_pc,
                    frame.chunk.code.len(),
                    self.compatibility,
                )
                .map(Arc::new);
                (frame.chunk.as_arc().clone(), plan)
            });
            cache.hot[hot_index] = Some((key, plan.clone()));
        }
        let Some(plan) = cache.hot[hot_index]
            .as_ref()
            .expect("hot site was installed")
            .1
            .as_ref()
        else {
            return false;
        };
        // No output changes until every input passes. Coercion, metamethods,
        // and integer arithmetic remain on the original bytecode path.
        if plan
            .input_registers()
            .iter()
            .any(|&register| !matches!(frame.registers[usize::from(register)], Value::Number(_)))
        {
            return false;
        }
        // Plans are almost always short, so keep their scratch numbers in a
        // small array instead of zeroing a 2 KiB frame on every visit.
        let operation_count = plan.number_operations().len();
        let mut inline_numbers = [0.0; SCALAR_INLINE_NUMBERS];
        let mut spilled_numbers;
        let numbers: &mut [f64] = if operation_count <= SCALAR_INLINE_NUMBERS {
            &mut inline_numbers
        } else if operation_count <= SCALAR_MAX_NUMBERS {
            spilled_numbers = vec![0.0; operation_count];
            &mut spilled_numbers
        } else {
            return false;
        };
        for (index, operation) in plan.number_operations().iter().enumerate() {
            numbers[index] = match *operation {
                ScalarNumberOperation::Input { register } => {
                    let Value::Number(value) = frame.registers[usize::from(register)] else {
                        unreachable!("all scalar inputs were checked before evaluation")
                    };
                    value
                }
                ScalarNumberOperation::Constant { value } => value,
                ScalarNumberOperation::Add { left, right } => numbers[left] + numbers[right],
                ScalarNumberOperation::Subtract { left, right } => numbers[left] - numbers[right],
                ScalarNumberOperation::Multiply { left, right } => numbers[left] * numbers[right],
                ScalarNumberOperation::Negate { source } => -numbers[source],
            };
        }
        for (register, output) in plan.outputs() {
            frame.registers[usize::from(*register)] = match output {
                ScalarTableOutput::Number { result_index } => Value::Number(numbers[*result_index]),
                ScalarTableOutput::Constant(value) => constant_value(value, self.compatibility),
            };
        }
        frame.pc = plan.end_pc();
        // The fragment elided the table's allocation. Charge its phantom
        // bytes so `gcinfo` and collector pacing still see the heap Lua's
        // interpreter would have used; a completed cycle releases them.
        self.runtime.charge_elided_table(plan.field_count());
        true
    }
}
