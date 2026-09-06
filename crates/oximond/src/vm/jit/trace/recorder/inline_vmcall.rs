//! A call of a host C function made from the trace
//! ([`HelperId::vm_call`]).
//!
//! A host function may do anything: call back into Lua, change any table,
//! global or upvalue, raise an error. The recording therefore does with the
//! call what it does with a resume: it sleeps through it. The interpreter
//! makes the call, whose instructions and the Lua code it calls back are no
//! part of the recording, and wakes at the instruction after it, where the
//! result registers say what the function gave ([`Recorder::finish_vm_call`]).
//! The call is then written as the helper of that shape: values of the kinds
//! the recording saw. The helper writes the state in front of the call back
//! to the frame, makes the call as the interpreter does and gives the
//! results, or misses (see `helpers/vmcall.rs`).
//!
//! The helper is one of those that run arbitrary Lua code
//! ([`HelperEffects::RESUMES`](crate::jit_ir::trace::HelperEffects)), so the
//! trace may hold nothing across it that the code could invalidate
//! ([`TraceIr::resume_conflict`](crate::jit_ir::trace::TraceIr)). A
//! recording that ends up holding such a thing is refused, and the next
//! recording of the site aborts at the call ([`Recorder::with_no_vmcall`]).
//!
//! The root frame or a frame the trace inlined outside a coroutine makes the
//! call, with up to three results of integers, floats, booleans, nil, strings
//! and tables, from `Call`, `CallOne` or `TailCall`. A table or string result
//! gets a lane of the run's bank when the helper gives it. Any other call
//! aborts the recording.

use super::{
    Flow, HookPoint, Kind, Operand, PrevPc, RecKind, Recorder, Sym, TraceAbortReason, TraceScalar,
    TraceType,
};
use crate::bytecode::OpCode;
use crate::jit_ir::trace::{HelperId, MAX_RESUME_VALUES};
use crate::runtime::{NativeFunctionHandle, Value};

/// A host call the recorder is sleeping through.
#[derive(Clone, Copy, Debug)]
pub(super) struct VmCallWait {
    /// The call instruction.
    pub(super) pc: usize,
    /// The call's caller depth.
    pub(super) depth: usize,
    /// The address of the chunk the call instruction is in: the root's, or
    /// that of the inlined function making the call.
    chunk: usize,
    /// The first result register, and how many results the call wants.
    base: usize,
    results: usize,
    /// Whether the call is a `TailCall`: the function gives all its results,
    /// as open results, to the `ReturnOpen` after the call.
    tail: bool,
    /// How many host boundaries deep the call is made, once the VM has said.
    host_depth: Option<usize>,
}

impl Recorder {
    /// Whether the instruction about to run, `depth` host boundaries deep,
    /// is code the host function the recorder sleeps through runs.
    pub(crate) fn skips_nested_host(&self, depth: usize) -> bool {
        self.vm_wait
            .as_ref()
            .and_then(|wait| wait.host_depth)
            .is_some_and(|host| depth > host)
    }

    /// Names the number of host boundaries the sleeping call is made at.
    pub(crate) fn latch_host_depth(&mut self, depth: usize) {
        if let Some(wait) = self.vm_wait.as_mut()
            && wait.host_depth.is_none()
        {
            wait.host_depth = Some(depth);
        }
    }

    /// A call of the host function `function` at `at.pc`: sleeps until the
    /// call returns, or `None` when the recording aborts at the call.
    pub(super) fn record_vm_call(
        &mut self,
        at: &HookPoint<'_>,
        function: NativeFunctionHandle,
        hoisted: bool,
        (base, results): (usize, usize),
    ) -> Result<Option<Flow>, TraceAbortReason> {
        let pc = at.pc;
        let runtime = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?
            .runtime;
        // The function is the one the instruction itself calls: a protected
        // call (`pcall (f, ...)`) names it as an argument.
        let tail = at
            .chunk
            .code
            .get(pc)
            .is_some_and(|instruction| instruction.opcode() == Ok(OpCode::TailCall));
        let plain_call = at.chunk.code.get(pc).is_some_and(|instruction| {
            let callee = match instruction.opcode() {
                Ok(OpCode::Call | OpCode::TailCall) => usize::from(instruction.a()),
                Ok(OpCode::CallOne) => usize::from(instruction.b()),
                _ => return false,
            };
            matches!(
                at.registers.get(callee),
                Some(Value::NativeFunction(held)) if *held == function
            )
        });
        if runtime.host_native_token(function).is_none()
            || hoisted
            || (results == 0 && !tail)
            || results > MAX_RESUME_VALUES
            || !plain_call
            || self.counting
            || self.side.is_some()
            || self.function.is_some()
            || self.loop_tail.is_some()
            || self.activation.is_some()
            || self.running_caller()
            || self.open.is_some()
            || self.reexec.is_some()
            // The frames the trace inlined are rebuilt for the call as an
            // exit rebuilds them; the root's caller is not.
            || self.in_caller()
            || self.coroutine.is_some()
            || self.no_vmcall
            || !self.limits.vmcall
            // A table the trace never made cannot be written back to the
            // frame and then again at an exit: the two would be two tables.
            || self.regs.iter().any(|register| matches!(register, Sym::Virtual(_)))
            || self.frames.iter().any(|frame| {
                frame.handler
                    || frame
                        .regs
                        .iter()
                        .chain(&frame.varargs)
                        .any(|register| matches!(register, Sym::Virtual(_)))
            })
        {
            return Ok(None);
        }
        self.own_borrows(pc);
        self.set_prev(PrevPc::At(pc));
        self.vm_wait = Some(VmCallWait {
            pc,
            depth: at.depth,
            chunk: std::sync::Arc::as_ptr(at.chunk) as usize,
            base,
            results,
            tail,
            host_depth: None,
        });
        Ok(Some(Flow::Sleep))
    }

    /// Whether operation `index` is a table a host call gave, or a copy of
    /// one.
    pub(super) fn is_call_table(&self, mut index: u32) -> bool {
        loop {
            let op = &self.ops[index as usize];
            if op.ty != Some(TraceType::Table) {
                return false;
            }
            match op.kind {
                RecKind::Move(Operand::Op(source)) => index = source,
                RecKind::CallHelper { helper, .. } => return helper.is_vm_call(),
                RecKind::HelperResult { call, .. } => {
                    return matches!(
                        self.ops[call as usize].kind,
                        RecKind::CallHelper { helper, .. } if helper.is_vm_call()
                    );
                }
                _ => return false,
            }
        }
    }

    /// Gives every string a register holds as a borrow of an array slot a
    /// lane of the bank before the call, which may move the slot. Each
    /// register that held the borrow holds the lane after it.
    fn own_borrows(&mut self, pc: usize) {
        let borrowed = |recorder: &Self, sym: &Sym| match sym {
            Sym::Op(index)
                if recorder.ops[*index as usize]
                    .ty
                    .is_some_and(TraceType::is_borrowed) =>
            {
                Some(*index)
            }
            _ => None,
        };
        let mut found: Vec<u32> = Vec::new();
        let mut note = |index: Option<u32>| {
            if let Some(index) = index
                && !found.contains(&index)
            {
                found.push(index);
            }
        };
        for sym in &self.regs {
            note(borrowed(self, sym));
        }
        for frame in &self.frames {
            for sym in frame.regs.iter().chain(&frame.varargs) {
                note(borrowed(self, sym));
            }
        }
        // The call is not one that wants the borrows ended before the fact:
        // the work of owning them is no guest work.
        let work = self.host_work;
        for old in found {
            let owned = self.emit_helper_call(
                pc,
                HelperId::STRING_OWN,
                &[(Operand::Op(old), TraceType::StringRef)],
            );
            let swap = |sym: &mut Sym| {
                if *sym == Sym::Op(old) {
                    *sym = Sym::Op(owned);
                }
            };
            self.regs.iter_mut().for_each(swap);
            for frame in &mut self.frames {
                frame
                    .regs
                    .iter_mut()
                    .chain(frame.varargs.iter_mut())
                    .for_each(swap);
            }
        }
        self.host_work = work;
    }

    /// The interpreter is back from the host function: the result registers
    /// hold what it gave. Writes the call as the helper of that shape.
    pub(super) fn finish_vm_call(
        &mut self,
        at: &HookPoint<'_>,
        wait: VmCallWait,
    ) -> Result<(), TraceAbortReason> {
        if at.depth != wait.depth
            || std::sync::Arc::as_ptr(at.chunk) as usize != wait.chunk
            || at.pc != wait.pc + 1
        {
            return Err(TraceAbortReason::Divergence);
        }
        // A tail call hands its results on as open results, to the
        // `ReturnOpen` that follows it.
        let live = if wait.tail {
            if !at
                .chunk
                .code
                .get(at.pc)
                .is_some_and(|next| next.opcode() == Ok(OpCode::ReturnOpen))
            {
                return Err(TraceAbortReason::Divergence);
            }
            at.open
        } else {
            at.registers
                .get(wait.base..wait.base + wait.results)
                .ok_or(TraceAbortReason::Divergence)?
        };
        if live.len() > MAX_RESUME_VALUES {
            self.refused_vmcall.set(true);
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let mut kinds = 0;
        let mut scale = 1;
        let mut types = Vec::with_capacity(live.len());
        for value in live {
            let (digit, ty) = match (value, Kind::of(value, self.integers)) {
                (_, Kind::Integer) => (0, TraceType::Integer),
                (_, Kind::Float) => (1, TraceType::Float),
                (_, Kind::Boolean) => (2, TraceType::Boolean),
                (_, Kind::Nil) => (3, TraceType::Nil),
                (_, Kind::Table) => (4, TraceType::Table),
                (_, Kind::String) => (5, TraceType::String),
                _ => {
                    // Results of another kind have no lane or word to travel
                    // in.
                    self.refused_vmcall.set(true);
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
            };
            kinds += digit * scale;
            scale *= 6;
            types.push(ty);
        }
        let pc = wait.pc;
        let call = self.emit_helper_call(pc, HelperId::vm_call(types.len(), kinds), &[]);
        let mut values = Vec::with_capacity(wait.results);
        for (index, ty) in types.into_iter().enumerate() {
            values.push(if ty == TraceType::Nil {
                // The function gave nil, which the helper checked: there is
                // no word to carry.
                self.scalar(pc, TraceScalar::Nil)
            } else if index == 0 {
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
        self.native_results(
            pc,
            &values,
            (wait.base, if wait.tail { 0 } else { wait.results }),
        );
        // The function may move any table's storage, which ends the borrows
        // of strings from array slots: the loop that holds one across the
        // call aborts its next recording at the call.
        if self.borrow_ended {
            self.refused_vmcall.set(true);
            return Err(TraceAbortReason::BorrowOutlivesEffect);
        }
        Ok(())
    }
}
