//! Linking through a return: a side that leaves the root frame by its
//! `Return`, runs on in the frame that called it, and calls the root frame's
//! function again (see "Linking through a return" in [`super::super`]).
//!
//! The caller's code is recorded as a real frame, like the code of an inlined
//! callee. The root frame has returned by then, so a guard in it leaves
//! through a snapshot whose first inner frame is the caller
//! ([`Entry::Caller`]): the exit writes the caller's registers, pops the root
//! frame as the interpreter's `Return` would, and resumes the caller at the
//! instruction the guard stands for. The caller's code may store, make
//! tables and call other functions, since every effect it commits is covered
//! by the snapshots of the guards that follow. The recorder refuses a
//! `Return` of the caller, closures, varargs, open results, tail calls and
//! the instructions that close upvalues there, and a write to a register a
//! closure captured.
//!
//! The caller's registers are never a frame of the trace. A register the
//! region reads is loaded from a window slot ([`ExternalSlot::CallerRegister`])
//! on first use, and a register it wrote that the code after the call can
//! still see goes back into its slot when the root function is called again.
//! The slots are loop parameters like the cells of a closure, so the caller's
//! state rides the loop-back and every exit writes it into the interpreter's
//! frame.

use super::{
    Flow, HookPoint, Inlined, Kind, MAX_TRACE_EXTERNALS, MAX_TRACE_INLINED_CALLS, Operand, PrevPc,
    RecKind, Recorder, Sym, TraceAbortReason, TraceScalar, TraceType,
};
use crate::bytecode::{Chunk, Instruction, OpCode, RegisterSet};
use crate::jit_ir::trace::{Entry, ExternalSlot, FrameId, InlinedFrame, Protected, TraceFrameInfo};
use crate::runtime::{ClosureHandle, Value};
use std::sync::Arc;

/// The lane of the closure of a caller that runs none.
pub(crate) const NO_CLOSURE: u32 = u32::MAX;

/// The frame that called the root frame, as a side recording starts: what the
/// recording needs to return into it.
#[derive(Clone, Debug)]
pub(crate) struct CallerSeed {
    /// The caller's chunk, and the closure it runs (a chunk the host
    /// executed runs none, and has no upvalues).
    pub(crate) chunk: Arc<Chunk>,
    pub(crate) closure: Option<ClosureHandle>,
    /// The closure the root frame runs, which the caller must call again.
    pub(crate) root: ClosureHandle,
    /// The call in the caller's chunk that made the root frame, and the
    /// window its results go to.
    pub(crate) call_pc: usize,
    pub(crate) result_base: u8,
    pub(crate) results: u8,
    /// The caller's registers when the recording starts.
    pub(crate) registers: Vec<Value>,
    /// The caller's registers that a closure captured: the variable, not the
    /// register, holds the value, so the trace leaves them alone.
    pub(crate) captured: Vec<u8>,
    /// The environment the caller reads its globals from.
    pub(crate) environment: crate::runtime::TableHandle,
}

impl Recorder {
    /// The caller the recording may return into, if the seed has one.
    pub(super) fn caller_seed(&self) -> Option<&CallerSeed> {
        self.side.as_ref().and_then(|seed| seed.caller.as_ref())
    }

    /// The chunk of the caller the recording returned into, if it did, and
    /// the registers of that caller a closure had captured then.
    pub(crate) fn returned_into(&self) -> Option<(&Arc<Chunk>, &[u8])> {
        let chunk = self.returned_into.as_ref()?;
        let captured = self
            .caller_seed()
            .map_or(&[][..], |seed| seed.captured.as_slice());
        Some((chunk, captured))
    }

    /// Whether the recording runs in the root frame's caller now.
    pub(super) fn in_caller(&self) -> bool {
        self.frames
            .first()
            .is_some_and(|frame| matches!(frame.entry, Entry::Caller { .. }))
    }

    /// Whether the running frame is the root frame's caller: the recording
    /// is in the caller's own code, not in a function it called.
    pub(super) fn running_caller(&self) -> bool {
        self.frames.len() == 1 && self.in_caller()
    }

    /// Whether a `Return` of the root frame continues into the caller. A
    /// side that went beyond the nest does, if it can. Anything it cannot
    /// follow ends the recording at the return as before. So does a second
    /// return: a call that comes back before the loop of the root does (a
    /// function that is given nothing to loop over) would make the recording
    /// span several activations, each with a caller frame of its own, which no
    /// exit rebuilds right. The side ends there, and the guard asks again.
    pub(super) fn may_return_through(&self) -> bool {
        self.outer_side
            && self.frames.is_empty()
            && self.activation.is_none()
            && self.function.is_none()
            && !self.counting
            && !self.looping
            && self.caller_seed().is_some()
            && self.returned_into.is_none()
    }

    /// The window register of caller register `register`, naming it on first
    /// use. Its entry type is what the register holds now: nothing in the
    /// region has written it before its first use.
    fn caller_slot(
        &mut self,
        at: &HookPoint<'_>,
        register: usize,
    ) -> Result<usize, TraceAbortReason> {
        let name = ExternalSlot::CallerRegister(
            u16::try_from(register).map_err(|_| TraceAbortReason::UnsupportedOperand)?,
        );
        if let Some(position) = self.externals.iter().position(|known| *known == name) {
            return Ok(self.register_count + position);
        }
        let seed = self.caller_seed().ok_or(TraceAbortReason::Divergence)?;
        if u8::try_from(register).is_ok_and(|narrow| seed.captured.contains(&narrow)) {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let live = at
            .registers
            .get(register)
            .ok_or(TraceAbortReason::Divergence)?;
        let kind = Kind::of(live, self.integers);
        if kind == Kind::Other || kind == Kind::Borrowed {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        if self.externals.len() >= MAX_TRACE_EXTERNALS {
            return Err(super::limit("externals"));
        }
        self.externals.push(name);
        self.external_cells.push(None);
        self.entry.push(kind);
        self.regs.push(Sym::Carried);
        self.written.push(false);
        Ok(self.regs.len() - 1)
    }

    /// Loads the caller registers the instruction at `at.pc` reads and the
    /// region has not touched yet from their window slots, so that `read`
    /// finds a value in every register of the caller. Nothing else in the
    /// region can read a register, so this is the only place a slot is read.
    pub(super) fn load_caller_reads(&mut self, at: &HookPoint<'_>) -> Result<(), TraceAbortReason> {
        let reads = at.chunk.instruction_reads(at.pc);
        let count = usize::from(at.chunk.register_count);
        for register in 0..count {
            let Ok(narrow) = u8::try_from(register) else {
                break;
            };
            if !reads.contains(narrow) {
                continue;
            }
            let untouched = self
                .frames
                .first()
                .is_some_and(|frame| frame.regs.get(register) == Some(&Sym::Carried));
            if !untouched {
                continue;
            }
            let slot = self.caller_slot(at, register)?;
            let kind = self.entry[slot];
            let ty = kind
                .trace_type()
                .ok_or(TraceAbortReason::UnsupportedOperand)?;
            let load = self.emit(at.pc, RecKind::Move(Operand::Carried(slot)), Some(ty));
            if let Some(frame) = self.frames.first_mut() {
                frame.regs[register] = Sym::Op(load);
            }
            self.caller_loads.push((register, load));
        }
        // An instruction whose register effects the analysis does not model
        // reads every register.
        if reads == RegisterSet::ALL && count > 0 {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        Ok(())
    }

    /// Records the `Return` at `at.pc` of the root frame as a return into
    /// its caller, which the recording follows from here.
    pub(super) fn return_through(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<Flow, TraceAbortReason> {
        let seed = self
            .caller_seed()
            .cloned()
            .ok_or(TraceAbortReason::Divergence)?;
        let opcode = instruction.opcode();
        let a = usize::from(instruction.a());
        let count = if opcode == Ok(OpCode::Halt) {
            0
        } else {
            match usize::from(instruction.b()) {
                0 => return Err(TraceAbortReason::UnsupportedOperand),
                count => count,
            }
        };
        if opcode == Ok(OpCode::ReturnOpen) || opcode == Ok(OpCode::ReturnVararg) {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        if count > 0 {
            self.check_register(a + count - 1)?;
        }
        let results = usize::from(seed.results);
        let base = usize::from(seed.result_base);
        if results == 0 || base + results > usize::from(seed.chunk.register_count) {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        if self.inlined.len() >= MAX_TRACE_INLINED_CALLS {
            return Err(super::limit("inlined calls"));
        }
        // The values the caller gets.
        let mut values = Vec::with_capacity(results);
        for index in 0..count.min(results) {
            if matches!(self.sym(a + index), Sym::Root(_)) {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            let sym = match self.read(at.registers, a + index)? {
                (Operand::Op(op), _) => Sym::Op(op),
                // A value the iteration did not write: its entry value.
                (carried @ Operand::Carried(_), ty) => {
                    Sym::Op(self.emit(at.pc, RecKind::Move(carried), Some(ty)))
                }
            };
            values.push(sym);
        }
        if values.len() < results {
            let nil = self.emit(
                at.pc,
                RecKind::Constant(TraceScalar::Nil),
                Some(TraceType::Nil),
            );
            values.resize(results, Sym::Op(nil));
        }
        // The root frame stands at its return for every snapshot from here:
        // the frame that follows is stopped at the call that made it.
        self.set_prev(PrevPc::At(at.pc));
        self.returned_into = Some(Arc::clone(&seed.chunk));
        // The closure's cells and environment are the caller's own, which a
        // read of an upvalue or a global goes through like a callee's.
        let lane = match seed.closure {
            Some(closure) => self.pin(closure.native_identity())?,
            None => NO_CLOSURE,
        };
        self.inlined.push(TraceFrameInfo {
            register_count: u16::from(seed.chunk.register_count),
            code_len: seed.chunk.code.len(),
            inlined: Some(InlinedFrame {
                closure: lane,
                parent: FrameId(self.root_frame_id()),
                call_pc: at.pc,
                result_base: 0,
                results: 0,
                reexec: false,
                tail_calls: 0,
                tail_site: None,
                protected: Protected::No,
                entry: Entry::Caller {
                    call_pc: u32::try_from(seed.call_pc)
                        .map_err(|_| TraceAbortReason::UnsupportedOperand)?,
                    result_base: seed.result_base,
                    results: seed.results,
                },
                dynamic: false,
            }),
            activation: None,
        });
        let mut regs = vec![Sym::Carried; usize::from(seed.chunk.register_count)];
        for (index, value) in values.into_iter().enumerate() {
            regs[base + index] = value;
        }
        let entry = Entry::Caller {
            call_pc: u32::try_from(seed.call_pc)
                .map_err(|_| TraceAbortReason::UnsupportedOperand)?,
            result_base: seed.result_base,
            results: seed.results,
        };
        self.frames.push(Inlined {
            chunk: Arc::clone(&seed.chunk),
            closure: lane,
            frame: u32::try_from(self.inlined.len()).expect("frame count fits u32"),
            regs,
            prev: PrevPc::At(seed.call_pc),
            call_pc: at.pc,
            result_base: base,
            results,
            varargs: Vec::new(),
            owns_reexec: false,
            tail_calls: 0,
            protected: Protected::No,
            handler: false,
            entry,
            after: super::Pending::Nothing,
            dynamic: None,
        });
        self.call_site = None;
        self.caller_loads.clear();
        Ok(Flow::LeaveRoot {
            resume: seed.call_pc + 1,
        })
    }

    /// Records the call at `at.pc` in the caller as the root function's next
    /// activation: the caller's state goes back into its window slots, the
    /// root frame starts over with the arguments, and the recording goes on
    /// at the function's first instruction.
    pub(super) fn call_root_again(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<Flow, TraceAbortReason> {
        let seed = self
            .caller_seed()
            .cloned()
            .ok_or(TraceAbortReason::Divergence)?;
        let pc = at.pc;
        let (callee, first_argument, fixed, base, results) = match instruction.opcode() {
            Ok(OpCode::CallOne) => (
                usize::from(instruction.b()),
                usize::from(instruction.c()),
                1,
                usize::from(instruction.a()),
                1,
            ),
            Ok(OpCode::Call) => {
                let base = usize::from(instruction.a());
                (
                    base,
                    base + 1,
                    usize::from(instruction.b()),
                    base,
                    usize::from(instruction.c()),
                )
            }
            _ => return Err(TraceAbortReason::UnsupportedOperand),
        };
        // The call the root frame was entered by: the same instruction and
        // the same result window, so the frames match the ones an exit finds.
        if pc != seed.call_pc
            || base != usize::from(seed.result_base)
            || results != usize::from(seed.results)
        {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let Value::Closure(handle) = at.registers[callee] else {
            return Err(TraceAbortReason::UnsupportedOperand);
        };
        if handle != seed.root
            || self.root_chunk().is_vararg
            || self.root_chunk().arg_register.is_some()
        {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        self.check_register(callee)?;
        if fixed > 0 {
            self.check_register(first_argument + fixed - 1)?;
        }
        let (callee_operand, _) = self.read(at.registers, callee)?;
        let mut arguments = Vec::with_capacity(fixed);
        for index in 0..fixed {
            arguments.push(self.read(at.registers, first_argument + index)?);
        }
        let lane = self.pin(handle.native_identity())?;
        // Another callee runs in the interpreter, which returns and calls it.
        self.guard_callee(pc, callee_operand, lane);

        // A fresh activation: the arguments, then nil, in every register.
        // The operations belong to the call, which is the caller's.
        let chunk = Arc::clone(&self.chunk);
        let parameters = usize::from(chunk.parameter_count).min(usize::from(chunk.register_count));
        let passed = parameters.min(arguments.len());
        let mut regs = Vec::with_capacity(self.register_count);
        for (source, ty) in &arguments[..passed] {
            let value = self.emit(pc, RecKind::Move(*source), Some(*ty));
            regs.push(Sym::Op(value));
        }
        if passed < self.register_count {
            let nil = self.emit(
                pc,
                RecKind::Constant(TraceScalar::Nil),
                Some(TraceType::Nil),
            );
            regs.resize(self.register_count, Sym::Op(nil));
        }

        // The caller's state at the call: what the code after the call can
        // still see goes back to the window.
        let live = self.chunk_of_caller().live_registers(pc + 1);
        let frame = self.frames.pop().ok_or(TraceAbortReason::Divergence)?;
        self.reexec = None;
        self.call_site = None;
        let wanted = |register: usize| {
            u8::try_from(register).is_ok_and(|narrow| live.contains(narrow))
                && !(base..base + results).contains(&register)
        };
        for (register, sym) in frame.regs.iter().enumerate() {
            let unchanged = match sym {
                Sym::Carried => true,
                Sym::Op(op) => self
                    .caller_loads
                    .iter()
                    .any(|(loaded, load)| *loaded == register && load == op),
                Sym::Root(_) | Sym::Virtual(_) => false,
            };
            if unchanged {
                continue;
            }
            // The variable of a captured register is its cell: a write the
            // register took would not reach the closures that share it.
            if u8::try_from(register).is_ok_and(|narrow| seed.captured.contains(&narrow)) {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            if !wanted(register) {
                continue;
            }
            // A table the region made that escaped is a real table by now,
            // and the slot holds it. One that has not escaped exists nowhere
            // else, and the root frame's exits could not build it.
            let held = match *sym {
                Sym::Op(_) => *sym,
                Sym::Virtual(id) => match self.virtuals[id as usize].real {
                    Some(escaped) => Sym::Op(escaped.table),
                    None => return Err(TraceAbortReason::UnsupportedOperand),
                },
                Sym::Carried | Sym::Root(_) => return Err(TraceAbortReason::UnsupportedOperand),
            };
            let slot = self.caller_slot(at, register)?;
            self.regs[slot] = held;
            self.written[slot] = true;
        }
        self.caller_loads.clear();

        for (register, sym) in regs.into_iter().enumerate() {
            self.regs[register] = sym;
            self.written[register] = true;
        }
        // The interpreter marks the call as the caller's last instruction and
        // keeps no previous pc in the new frame.
        self.prev = PrevPc::Unset;
        Ok(Flow::EnterRoot)
    }

    /// Whether the caller's code records `opcode`. The caller is a frame like
    /// an inlined callee's, so what a callee's code does, it does, except
    /// what needs the caller to be left or to be rebuilt by an exit: a
    /// return, a closure, varargs and open results, tail calls and the
    /// instructions that close upvalues. An instruction that reads or writes
    /// the closure's upvalues, or writes its globals, needs `closure`. A
    /// `GetTabUp` is a read of a global (of the caller's environment, which
    /// needs no closure) or it is refused.
    pub(super) const fn caller_may_run(opcode: OpCode, closure: bool) -> bool {
        match opcode {
            OpCode::GetUpvalue
            | OpCode::SetUpvalue
            | OpCode::SetGlobal
            | OpCode::SetGlobalWide
            | OpCode::SetTabUp => closure,
            OpCode::Return
            | OpCode::Halt
            | OpCode::ReturnOpen
            | OpCode::ReturnVararg
            | OpCode::Closure
            | OpCode::Vararg
            | OpCode::CallVararg
            | OpCode::CallOpen
            | OpCode::TableVararg
            | OpCode::TableOpen
            | OpCode::CloseUpvalue
            | OpCode::CloseLocal
            | OpCode::MarkClose
            | OpCode::ExtraArgument
            | OpCode::TailCall
            | OpCode::TailCallVararg
            | OpCode::TailCallOpen => false,
            _ => true,
        }
    }

    /// Whether `pc` is the call that made the root frame in the caller.
    pub(super) fn is_root_call(&self, pc: usize) -> bool {
        self.caller_seed().is_some_and(|seed| seed.call_pc == pc)
    }

    /// The chunk of the caller while the recording runs in it.
    fn chunk_of_caller(&self) -> Arc<Chunk> {
        self.frames
            .first()
            .map_or_else(|| Arc::clone(&self.chunk), |frame| Arc::clone(&frame.chunk))
    }

    /// Whether a taken branch from `pc` back to `target` in the caller's code
    /// goes around the call that made the root frame: the loop that calls it
    /// again. The recording follows it like the loop around the root's own.
    pub(super) fn caller_branch(&self, pc: usize, target: usize) -> bool {
        self.caller_seed()
            .is_some_and(|seed| target <= seed.call_pc && seed.call_pc < pc)
    }
}
