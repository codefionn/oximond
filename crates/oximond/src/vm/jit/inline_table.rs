//! VM-facing wrapper around the shared inline table-call plan.

use crate::bytecode::Chunk;
use crate::bytecode::Instruction;
use crate::compat::CompatibilityProfile;
use crate::jit::{PreparedInlineTableCallPlan, evaluate_inline_table_call};
use crate::jit_ir::validate_inline_table_call_plan;
use crate::runtime::ClosureHandle;
use crate::runtime::{Runtime, Value};
use crate::vm::jit::inline_calls::{InlineResult, boxed};
use crate::vm::jit::state::MAX_INLINE_CALL_PLAN_CACHE;
use crate::vm::registers::write_scalar_call_result;
use crate::vm::stack::CallStack;
use crate::vm::{Vm, VmError, VmResult};
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct InlineTablePlan {
    prepared: PreparedInlineTableCallPlan,
}

impl InlineTablePlan {
    pub(crate) fn build(chunk: &Chunk, compatibility: CompatibilityProfile) -> Option<Self> {
        validate_inline_table_call_plan(chunk, compatibility)
            .map(PreparedInlineTableCallPlan::new)
            .map(|prepared| Self { prepared })
    }
    pub(crate) const fn parameter_count(&self) -> u8 {
        self.prepared.plan().parameter_count()
    }
    pub(crate) fn evaluate(&self, runtime: &mut Runtime, arguments: &[Value]) -> Option<Value> {
        evaluate_inline_table_call(&self.prepared, runtime, arguments)
    }
}

impl Vm {
    /// Attempts an effect-free table-method evaluation before building a Lua
    /// frame. The plan itself commits its sole raw write only after computing
    /// the returned value, so every guard miss can use the ordinary call path.
    #[inline]
    pub(crate) fn try_inline_table_call(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        closure: ClosureHandle,
        argument_start: usize,
        argument_count: usize,
    ) -> VmResult<bool> {
        self.inline_table_call(frames, instruction, closure, argument_start, argument_count)
            .map_err(|error| *error)
    }

    fn inline_table_call(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        closure: ClosureHandle,
        argument_start: usize,
        argument_count: usize,
    ) -> InlineResult {
        if !self.debug_hooks.is_empty() || self.jit.trace.recording() {
            return Ok(false);
        }
        // A cached plan is borrowed from the JIT state while the runtime is
        // written, so a hit costs no reference-count operation.
        let cached = if let Some(cached) = self.jit.inline_table_plans.get(&closure) {
            cached
        } else {
            self.inline_table_plan(closure);
            &self.jit.inline_table_plans[&closure]
        };
        let Some(plan) = cached else {
            return Ok(false);
        };
        if argument_count != usize::from(plan.parameter_count()) {
            return Ok(false);
        }
        // Evaluation finishes before the destination is written, including
        // when an argument aliases the destination register.
        let arguments = frames
            .last_registers()
            .expect("an inline call has an active caller frame")
            .get(argument_start..argument_start + argument_count)
            .ok_or_else(|| {
                boxed(VmError::RegisterRange {
                    start: instruction.a(),
                    count: instruction.b(),
                })
            })?;
        let Some(result) = plan.evaluate(&mut self.runtime, arguments) else {
            return Ok(false);
        };
        let mut frame = frames
            .last_mut()
            .expect("an inline call has an active caller frame");
        write_scalar_call_result(&mut frame, instruction, result).map_err(boxed)?;
        self.jit.statistics.interpreter_inline_table_calls = self
            .jit
            .statistics
            .interpreter_inline_table_calls
            .saturating_add(1);
        Ok(true)
    }

    pub(crate) fn inline_table_plan(
        &mut self,
        closure: ClosureHandle,
    ) -> Option<Arc<InlineTablePlan>> {
        if let Some(plan) = self.jit.inline_table_plans.get(&closure) {
            return plan.clone();
        }
        let cached = self
            .runtime
            .closure_snapshot(closure)
            .ok()
            .and_then(|snapshot| {
                snapshot
                    .cells()
                    .is_empty()
                    .then(|| InlineTablePlan::build(&snapshot.chunk(), self.compatibility))?
            })
            .map(Arc::new);
        if self.jit.inline_table_plans.len() >= MAX_INLINE_CALL_PLAN_CACHE {
            let eviction = *self
                .jit
                .inline_table_plans
                .keys()
                .next()
                .expect("a full inline-table-plan cache has an entry to evict");
            self.jit.inline_table_plans.remove(&eviction);
        }
        self.jit.inline_table_plans.insert(closure, cached.clone());
        cached
    }
}

#[cfg(test)]
mod tests {
    use super::InlineTablePlan;
    use crate::bytecode::{Chunk, Instruction, OpCode};
    use crate::compat::CompatibilityProfile;
    use crate::runtime::{Runtime, Value};

    #[test]
    fn dead_clear_elimination_preserves_returned_nil() {
        let chunk = Chunk::with_parts(
            4,
            3,
            Vec::new(),
            vec![
                Instruction::abc(OpCode::SetTable, 0, 1, 2),
                Instruction::clear_range(2, 1),
                Instruction::clear_range(3, 1),
                Instruction::return_(2),
            ],
            Vec::new(),
            Vec::new(),
        );
        let plan = InlineTablePlan::build(&chunk, CompatibilityProfile::Lua51).unwrap();
        let mut runtime = Runtime::new();
        let table = runtime.create_table().unwrap();
        let key = Value::from("value");
        runtime.table_set(table, &key, Value::Number(1.0)).unwrap();
        assert_eq!(
            plan.evaluate(
                &mut runtime,
                &[Value::Table(table), key.clone(), Value::Number(7.0)]
            ),
            Some(Value::Nil)
        );
        assert_eq!(runtime.table_get(table, &key).unwrap(), Value::Number(7.0));
    }
}
