//! Cached inline call plans and their currency checks.

use crate::bytecode::OpCode;
use crate::jit_ir::{
    ChunkDigest, InlineCallPlan, compose_vararg_forwarding_inline_call_plan_after_validation,
    validate_inline_call_plan, validate_inline_call_plan_after_validation,
};
use crate::runtime::{ClosureHandle, Value};
use crate::vm::Vm;
use crate::vm::jit::state::{
    CachedInlineCall, ForwardedInlineCallGuard, HOT_INLINE_PLAN_SLOTS, MAX_INLINE_CALL_PLAN_CACHE,
    chunk_digest,
};
use std::sync::Arc;

/// The hot plan slot a closure maps to: direct-mapped on its handle index.
fn hot_plan_slot(closure: ClosureHandle) -> usize {
    closure.index() as usize % HOT_INLINE_PLAN_SLOTS
}

impl Vm {
    /// Points at the plan for `closure` in the hot plan slots, or `None`
    /// unless `closure` is live and cached there.
    ///
    /// This is [`Self::inline_call_plan`]'s cache hit without a hash lookup
    /// or the reference-count traffic of cloning the plan, and calls that
    /// alternate between a few closures keep hitting. The pointer stays valid
    /// while the slot keeps its `Arc`: until a hot slot is overwritten,
    /// which only [`Self::remember_hot_inline_plan`] does.
    pub(crate) fn hot_inline_plan(&self, closure: ClosureHandle) -> Option<*const InlineCallPlan> {
        match &self.jit.inline_call_hot_plans[hot_plan_slot(closure)] {
            Some((hot_closure, plan)) if *hot_closure == closure => {
                if self.runtime.closure_environment(closure).is_err() {
                    return None;
                }
                Some(Arc::as_ptr(plan))
            }
            _ => None,
        }
    }

    /// Offers a plan for `closure` to the hot plan slots. A slot that
    /// already holds the closure is left alone.
    fn remember_hot_inline_plan(&mut self, closure: ClosureHandle, cached: &CachedInlineCall) {
        if cached.forwarded.is_some() {
            return;
        }
        let slot = &mut self.jit.inline_call_hot_plans[hot_plan_slot(closure)];
        if !matches!(slot, Some((hot_closure, _)) if *hot_closure == closure) {
            *slot = Some((closure, Arc::clone(&cached.plan)));
        }
    }

    #[allow(
        clippy::too_many_lines,
        reason = "keeps hot-cache eligibility beside plan discovery and retry rules"
    )]
    pub(crate) fn inline_call_plan(&mut self, closure: ClosureHandle) -> Option<CachedInlineCall> {
        if let Some((cached_closure, cached)) = &self.jit.inline_call_hot_cache
            && *cached_closure == closure
        {
            match cached {
                Some(cached) if self.runtime.closure_environment(closure).is_ok() => {
                    debug_assert!(cached.forwarded.is_none());
                    let cached = cached.clone();
                    self.remember_hot_inline_plan(closure, &cached);
                    return Some(cached);
                }
                // Refusing optimization executes nothing. As with a negative
                // map hit, the ordinary call path validates the live handle.
                None => return None,
                _ => {}
            }
        }
        match self.jit.inline_call_plans.get(&closure).cloned() {
            Some(None) if !self.jit.inline_call_retryable_refusals.contains(&closure) => {
                self.jit.inline_call_hot_cache = Some((closure, None));
                return None;
            }
            Some(Some(plan)) => {
                // A closure's chunk and upvalue count are immutable. The capture
                // cell itself may be rebound by debug.upvaluejoin, but the direct
                // evaluator reads that cell below on every call. Avoid cloning
                // and hashing immutable bytecode in the captured-closure hot
                // path while still rejecting a stale handle exactly as the
                // ordinary call path does.
                if plan.forwarded.is_none() && self.runtime.closure_environment(closure).is_ok() {
                    self.jit.inline_call_hot_cache = Some((closure, Some(plan.clone())));
                    self.remember_hot_inline_plan(closure, &plan);
                    return Some(plan);
                }
                if self.cached_inline_call_is_current(closure, &plan) {
                    return Some(plan);
                }
                self.jit.inline_call_plans.remove(&closure);
            }
            _ => {}
        }
        let snapshot = self.runtime.closure_snapshot(closure).ok()?;
        let chunk = snapshot.chunk();
        let nested_capture = chunk.upvalues.iter().any(|upvalue| {
            matches!(
                upvalue.source,
                crate::bytecode::UpvalueSource::ParentRegister(_)
                    | crate::bytecode::UpvalueSource::ParentUpvalue(_)
            )
        });
        let float_arith = self.compatibility.float_arith();
        let direct = if nested_capture {
            validate_inline_call_plan_after_validation(&chunk).ok()
        } else {
            validate_inline_call_plan(&chunk).ok()
        };
        let cached = direct
            .filter(|plan| usize::from(plan.upvalue_count()) == snapshot.cells().len())
            .map(|plan| plan.with_float_arith(float_arith))
            .map(|plan| CachedInlineCall {
                callee_digest: plan.callee_digest(),
                plan: Arc::new(plan),
                forwarded: None,
            })
            .or_else(|| {
                let [cell] = snapshot.cells() else {
                    return None;
                };
                let Value::Closure(target) = self.runtime.cell_get(*cell).ok()? else {
                    return None;
                };
                let target_snapshot = self.runtime.closure_snapshot(target).ok()?;
                if !target_snapshot.cells().is_empty() {
                    return None;
                }
                let target_plan =
                    validate_inline_call_plan_after_validation(&target_snapshot.chunk()).ok()?;
                let plan = compose_vararg_forwarding_inline_call_plan_after_validation(
                    &chunk,
                    &target_plan,
                )?
                .with_float_arith(float_arith);
                Some(CachedInlineCall {
                    callee_digest: ChunkDigest::of(&chunk),
                    forwarded: Some(ForwardedInlineCallGuard {
                        target_digest: target_plan.callee_digest(),
                    }),
                    plan: Arc::new(plan),
                })
            });
        // Plans compute in f64. Under a profile with an integer subtype, a
        // callee that loads an integer constant can return an integer, so it
        // gets no plan: the refusal is cached and the call hint stays plain.
        let integer_profile = self.compatibility.is_lua53_or_later();
        let cached =
            cached.filter(|cached| !(integer_profile && cached.plan.loads_integer_constant()));
        if self.jit.inline_call_plans.len() >= MAX_INLINE_CALL_PLAN_CACHE {
            let eviction = *self
                .jit
                .inline_call_plans
                .keys()
                .next()
                .expect("a full inline-plan cache has an entry to evict");
            self.jit.inline_call_plans.remove(&eviction);
            self.jit.inline_call_retryable_refusals.remove(&eviction);
        }
        if cached.is_none()
            && chunk
                .code
                .iter()
                .any(|instruction| instruction.opcode().ok() == Some(OpCode::TailCallVararg))
        {
            self.jit.inline_call_retryable_refusals.insert(closure);
        } else {
            self.jit.inline_call_retryable_refusals.remove(&closure);
        }
        self.jit.inline_call_plans.insert(closure, cached.clone());
        if cached.as_ref().is_none_or(|plan| plan.forwarded.is_none())
            && !self.jit.inline_call_retryable_refusals.contains(&closure)
        {
            self.jit.inline_call_hot_cache = Some((closure, cached.clone()));
            if let Some(cached) = &cached {
                self.remember_hot_inline_plan(closure, cached);
            }
        }
        cached
    }

    pub(crate) fn cached_inline_call_is_current(
        &mut self,
        closure: ClosureHandle,
        cached: &CachedInlineCall,
    ) -> bool {
        // Borrowing the closure's parts keeps the check free of the chunk and
        // cell reference-count traffic a snapshot costs.
        let Some(parts) = self.runtime.closure_call_parts(closure) else {
            return false;
        };
        if chunk_digest(&mut self.jit, parts.chunk) != cached.callee_digest {
            return false;
        }
        let Some(forwarded) = cached.forwarded else {
            return true;
        };
        let [cell] = &parts.cells[..] else {
            return false;
        };
        let Ok(Value::Closure(target)) = self.runtime.cell_get(*cell) else {
            return false;
        };
        self.runtime
            .closure_call_parts(target)
            .is_some_and(|target| {
                target.cells.is_empty()
                    && chunk_digest(&mut self.jit, target.chunk) == forwarded.target_digest
            })
    }
}
