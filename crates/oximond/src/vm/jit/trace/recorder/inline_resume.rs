//! `coroutine.resume` as a call of a helper that runs the coroutine from the
//! trace.
//!
//! A resume cannot be recorded the way a library call is: the recorder never
//! executes anything, and what the call gives is the thread's business. So
//! the recording does what it does for a self call. It guards the callee, then
//! sleeps through the call: the interpreter runs the thread, whose
//! instructions the recorder skips (they are another coroutine's), and wakes
//! at the instruction after the call, where the result registers say what the
//! thread handed back ([`Recorder::finish_resume`]). The call is then written
//! as the helper of that shape ([`HelperId::coroutine_resume`]): `true` and
//! values of the types the recording saw.
//!
//! Only a resume that succeeded records, with integers, floats and booleans
//! for values: a trace holds nothing else the thread hands back. A later
//! call that fails, finds the thread dead, or gives other values misses
//! *after* the thread ran, and the interpreter's replay of the call takes the
//! answer from the VM (see `Vm::resume_for_trace`). The root frame only: a
//! resume inside an inlined call would need the VM to rebuild frames the
//! thread could observe.
//!
//! `coroutine.yield` does not record. It leaves the loop through the thread
//! switch, in the middle of an iteration, which no trace can follow. A loop
//! that yields on every iteration (nothing before the call can skip it) is
//! rejected at the first recording that meets the call, so the interpreter
//! runs it without repeated attempts. A loop that yields on some iterations
//! only aborts the recording as an unsupported operand, and a later iteration
//! may record without it.

use super::{
    Flow, HookPoint, Operand, PrevPc, RecKind, Recorder, TraceAbortReason, TraceScalar, TraceType,
};
use crate::bytecode::OpCode;
use crate::jit_ir::trace::{
    HelperId, HelperType, MAX_RESUME_ARGS, MAX_RESUME_VALUES, PINNED_BUILTIN_COROUTINE_RESUME,
    pinned_builtin_identity,
};
use crate::runtime::Value;

/// The guest instructions a resume counts for, on top of the call of a
/// helper, when the policy weighs the work of a loop. The switch costs the
/// interpreter thousands of host instructions.
const RESUME_GUEST_WORK: u32 = 200;

/// A resume the recorder is sleeping through: the arguments of the helper
/// that stands for it and where the results go.
#[derive(Clone, Debug)]
pub(super) struct ResumeWait {
    /// The call instruction.
    pub(super) pc: usize,
    /// The call's caller depth.
    pub(super) depth: usize,
    /// The first result register, and how many results the call wants.
    base: usize,
    results: usize,
    /// The thread, then what the call passes it. For a function
    /// `coroutine.wrap` made, that function instead of the thread.
    arguments: Vec<(Operand, TraceType)>,
    /// Whether the call is of a function `coroutine.wrap` made.
    wrapped: bool,
}

impl Recorder {
    /// Whether the recorder is sleeping through a resume: the interpreter is
    /// running the thread, and the instructions it hooks are the thread's.
    pub(crate) const fn waits_for_resume(&self) -> bool {
        self.resume_wait.is_some()
    }

    /// A call of `coroutine.resume` at `at.pc`: guards the callee and sleeps
    /// until the call returns.
    pub(super) fn record_resume(
        &mut self,
        at: &HookPoint<'_>,
        callee: Operand,
        arguments: &[(Operand, TraceType)],
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        if results == 0
            || results > 1 + MAX_RESUME_VALUES
            || arguments.is_empty()
            || arguments.len() > 1 + MAX_RESUME_ARGS
            || arguments[0].1 != TraceType::Thread
            || !arguments[1..]
                .iter()
                .all(|(_, ty)| HelperType::Value.accepts(*ty))
            || !self.frames.is_empty()
            || self.function.is_some()
            || self.activation.is_some()
            || self.loop_tail.is_some()
            || self.reexec.is_some()
        {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let lane = self.pin(pinned_builtin_identity(PINNED_BUILTIN_COROUTINE_RESUME))?;
        self.guard_callee(pc, callee, lane);
        self.set_prev(PrevPc::At(pc));
        self.resume_wait = Some(ResumeWait {
            pc,
            depth: at.depth,
            base,
            results,
            arguments: arguments.to_vec(),
            wrapped: false,
        });
        Ok(Flow::Sleep)
    }

    /// A call of a function `coroutine.wrap` made at `at.pc`: sleeps until
    /// the call returns. There is no callee to guard: the function is a lane
    /// of the run's bank like any callable, and the helper checks that it is
    /// a wrapper each time, so one trace serves every wrapper.
    pub(super) fn record_wrapped_resume(
        &mut self,
        at: &HookPoint<'_>,
        callee: Operand,
        arguments: &[(Operand, TraceType)],
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        if results == 0
            || results > MAX_RESUME_VALUES
            || arguments.len() > MAX_RESUME_ARGS
            || !arguments
                .iter()
                .all(|(_, ty)| HelperType::Value.accepts(*ty))
            || !self.frames.is_empty()
            || self.function.is_some()
            || self.activation.is_some()
            || self.loop_tail.is_some()
            || self.reexec.is_some()
        {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        self.set_prev(PrevPc::At(pc));
        let mut all = vec![(callee, TraceType::Closure)];
        all.extend_from_slice(arguments);
        self.resume_wait = Some(ResumeWait {
            pc,
            depth: at.depth,
            base,
            results,
            arguments: all,
            wrapped: true,
        });
        Ok(Flow::Sleep)
    }

    /// The interpreter is back from the resume: the result registers hold
    /// what the thread gave. Writes the call as the helper of that shape.
    // The recording hooks hand the resume wait over by value; a reference
    // would spread through the hook API.
    #[allow(clippy::needless_pass_by_value)]
    pub(super) fn finish_resume(
        &mut self,
        at: &HookPoint<'_>,
        wait: ResumeWait,
    ) -> Result<(), TraceAbortReason> {
        if at.depth != wait.depth
            || !std::sync::Arc::ptr_eq(at.chunk, &self.chunk)
            || at.pc != wait.pc + 1
        {
            return Err(TraceAbortReason::Divergence);
        }
        let live = at
            .registers
            .get(wait.base..wait.base + wait.results)
            .ok_or(TraceAbortReason::Divergence)?;
        // A resume that failed (an error, a dead thread) has no recording.
        // The function a wrap made raises instead: a call that returned gave
        // its values, all of the results.
        let values = if wait.wrapped {
            live
        } else {
            if live[0] != Value::Boolean(true) {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            &live[1..]
        };
        let mut kinds = 0;
        let mut scale = 1;
        let mut types = Vec::with_capacity(values.len());
        for value in values {
            let (digit, ty) = match value {
                Value::Integer(_) if self.integers => (0, TraceType::Integer),
                Value::Number(_) => (1, TraceType::Float),
                Value::Boolean(_) => (2, TraceType::Boolean),
                _ => return Err(TraceAbortReason::UnsupportedOperand),
            };
            kinds += digit * scale;
            scale *= 3;
            types.push(ty);
        }
        self.host_work = self.host_work.saturating_add(RESUME_GUEST_WORK);
        let pc = wait.pc;
        let helper = if wait.wrapped {
            HelperId::coroutine_wrap(types.len(), kinds)
        } else {
            HelperId::coroutine_resume(types.len(), kinds)
        };
        let call = self.emit_helper_call(pc, helper, &wait.arguments);
        let mut values = Vec::with_capacity(wait.results);
        if !wait.wrapped {
            // The leading `true` is the helper's success, not one of its
            // results.
            values.push(self.emit(
                pc,
                RecKind::Constant(TraceScalar::Boolean(true)),
                Some(TraceType::Boolean),
            ));
        }
        for (index, ty) in types.into_iter().enumerate() {
            values.push(if index == 0 {
                call
            } else {
                self.emit(
                    pc,
                    RecKind::HelperResult {
                        call,
                        index: u8::try_from(index).map_err(|_| TraceAbortReason::Divergence)?,
                    },
                    Some(ty),
                )
            });
        }
        self.native_results(pc, &values, (wait.base, wait.results));
        Ok(())
    }

    /// A call of `coroutine.yield` at `at.pc` in the loop being recorded.
    pub(super) fn record_yield(&self, at: &HookPoint<'_>) -> TraceAbortReason {
        // Nothing before the call, from the header on, jumps over it inside
        // the loop (the loop's own exit test jumps out of it): every iteration
        // that gets here yields. A jump that lands between the call and the
        // latch makes the yield one iteration's affair.
        let skips = (self.header..at.pc).any(|pc| {
            let Some(instruction) = at.chunk.code.get(pc) else {
                return false;
            };
            let offset = match instruction.opcode() {
                Ok(OpCode::Jump) => instruction.jump_offset(),
                Ok(OpCode::JumpIfFalse) => {
                    i32::from(i16::from_ne_bytes(instruction.d().to_ne_bytes()))
                }
                _ => return false,
            };
            let target = i64::try_from(pc).unwrap_or(i64::MAX) + 1 + i64::from(offset);
            usize::try_from(target).is_ok_and(|target| target > at.pc && target <= self.latch)
        });
        if !skips && self.frames.is_empty() && self.side.is_none() && self.function.is_none() {
            TraceAbortReason::YieldingLoop
        } else {
            TraceAbortReason::Yield
        }
    }
}
