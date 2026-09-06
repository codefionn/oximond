//! The baseline interpreter dispatch loop, its GC safepoints, and opcode allocation classification.

use crate::bytecode::{Instruction, OpCode, ValidationError};
use crate::compat::CompatibilityProfile;
use crate::runtime::{GcStepResult, Value};
use crate::vm::arith::arithmetic_instruction;
use crate::vm::fast_path::FastExit;
use crate::vm::jit::trace::{HookPoint, HookSource, Surroundings, TraceAbortReason};
use crate::vm::registers::{
    branch_instruction, consume_wide_index, fixed_frame_values, for_in_loop_instruction,
    frame_values, load_instruction, load_wide_constant, numeric_for_instruction, table_instruction,
    table_open_instruction, table_vararg_instruction, vararg_values_instruction,
};
use crate::vm::stack::{CallStack, FrameMut};
use crate::vm::{
    AUTOMATIC_GC_STEP_KILOBYTES, DebugHookEvent, FastHookMode, FinalizerCompletion,
    TailCallArguments, Vm, VmError, VmResult, bump,
};

#[cfg(test)]
thread_local! {
    /// Opcodes the fast loop handed to the dispatch loop on this thread.
    pub(crate) static GENERAL_FALLBACKS: std::cell::RefCell<Vec<OpCode>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Records the instruction the dispatch loop is about to run after the fast
/// loop declined it.
#[cfg(test)]
fn note_general_fallback(frame: &crate::vm::Frame) {
    if let Some(opcode) = frame
        .chunk
        .code
        .get(frame.pc)
        .and_then(|instruction| instruction.opcode().ok())
    {
        GENERAL_FALLBACKS.with(|log| log.borrow_mut().push(opcode));
    }
}

impl Vm {
    /// Executes exactly one instruction.
    ///
    /// Tests use this per-instruction granularity; ordinary execution drives
    /// [`Self::interpret`], which keeps dispatching without a call boundary
    /// between instructions.
    #[cfg(test)]
    pub(crate) fn step(&mut self, frames: &mut CallStack) -> VmResult<Option<Vec<Value>>> {
        self.interpret(frames, true)
    }

    /// Dispatches instructions until an event that needs the outer loops.
    ///
    /// Completion, errors, coroutine yields, paused debug hooks, and native
    /// trace runs that request a safepoint return immediately. Everything else — calls, returns,
    /// table work, arithmetic — keeps looping here, so the per-instruction
    /// cost is the dispatch prologue instead of a full `step` call.
    /// `single_step` keeps the one-instruction contract for hook machinery
    /// and tests.
    #[allow(clippy::too_many_lines)]
    #[allow(
        clippy::unnecessary_lazy_evaluations,
        reason = "eager VmError construction is measured in every interpreter instruction"
    )]
    pub(crate) fn interpret(
        &mut self,
        frames: &mut CallStack,
        single_step: bool,
    ) -> VmResult<Option<Vec<Value>>> {
        let result = self.interpret_loop(frames, single_step);
        if result.is_err() {
            self.abort_recording_on_error();
        }
        result
    }

    /// An error ends any trace recording: unwinding resumes elsewhere. The
    /// exception is an error the recording predicted and models.
    #[cold]
    #[inline(never)]
    pub(crate) fn abort_recording_on_error(&mut self) {
        // A trace that models the unwinding of a protected call goes on.
        if self.jit.trace.recording_expects_error() {
            return;
        }
        self.jit
            .trace
            .abort_recording(&mut self.jit.statistics.trace, TraceAbortReason::Error);
    }

    /// Shows the recorder the instruction the general loop is about to run.
    #[inline(never)]
    fn record_general_hook(&mut self, frames: &CallStack, frame_index: usize, hooks_active: bool) {
        // `jit.off()` applies to traces: a recording stops when the JIT is off.
        if !self.runtime.jit_enabled() {
            self.jit
                .trace
                .abort_recording(&mut self.jit.statistics.trace, TraceAbortReason::Flushed);
            return;
        }
        // A host function the recording sleeps through runs this code.
        if self
            .jit
            .trace
            .skips_nested_host(self.host_yield_boundaries.len())
        {
            return;
        }
        let view = frames.at(frame_index);
        let chunk = view.frame.chunk.as_arc();
        let at = HookPoint {
            chunk: &chunk,
            depth: frames.len(),
            top_level: (0..=frame_index).all(|index| frames[index].closure.is_none()),
            pc: view.frame.pc,
            registers: view.registers,
            open: view.frame.open_values().unwrap_or_default(),
            coroutine: self.current_coroutine,
            hooks_active,
            source: HookSource::General,
            surroundings: Some(Surroundings {
                upvalues: &view.frame.upvalues,
                environment: view.frame.environment,
                runtime: &self.runtime,
                varargs: view.varargs,
            }),
        };
        let continues = self.jit.trace.record(&mut self.jit.statistics.trace, &at);
        if continues {
            self.jit
                .trace
                .latch_host_depth(self.host_yield_boundaries.len());
        }
        if !continues && self.jit.trace.has_compile_work() {
            self.service_trace_compilations();
        }
    }

    fn interpret_loop(
        &mut self,
        frames: &mut CallStack,
        single_step: bool,
    ) -> VmResult<Option<Vec<Value>>> {
        // Set when a counting fast loop handed back an instruction it had
        // not fetched: the prologue counts it, and it then runs here rather
        // than on a fast loop that would hand it back again.
        let mut general_once = false;
        loop {
            if self.sampler.is_some() {
                self.sample_interpreter(frames);
            }
            self.check_interrupt()?;
            let mut frame_index = frames.len() - 1;
            let native_entry_possible = self.native_entry_possible(&frames[frame_index]);
            if native_entry_possible {
                if self.maybe_run_native(frames)? {
                    self.gc_safepoint(frames)?;
                    self.gc_observed_allocation_serial = self.runtime.allocation_serial();
                    return Ok(None);
                }
                // A trace that left from inside an inlined call pushed the
                // callee frames above this one, and the interpreter resumes
                // the innermost.
                frame_index = frames.len() - 1;
            }
            let instruction_hooks_active = !self.debug_hooks.is_empty();
            if instruction_hooks_active {
                // Lua calls line hooks before executing the instruction. Keep the
                // program counter untouched while the callback runs, so its caller is
                // still the frame and source line being inspected.
                let skip_instruction_hooks = {
                    let frame = &mut frames[frame_index];
                    let skip = frame
                        .extras_ref()
                        .is_some_and(|extras| extras.skip_hook_pc_once == Some(frame.pc));
                    if skip && let Some(extras) = frame.extras_mut_ref() {
                        extras.skip_hook_pc_once = None;
                    }
                    skip
                };
                // Lines are tracked only under a line hook, as Lua 5.4 does,
                // so the line a newly set line hook starts from does not
                // depend on which loop ran the instructions before it.
                let line_event = if skip_instruction_hooks
                    || !self
                        .debug_hooks
                        .get(&self.current_coroutine)
                        .is_some_and(|hook| hook.mask.line)
                {
                    None
                } else {
                    let frame = &mut frames[frame_index];
                    let line = frame.chunk.debug.line_at(frame.pc);
                    let last_hook_line =
                        frame.extras_ref().and_then(|extras| extras.last_hook_line);
                    let last_hook_pc = frame.extras_ref().and_then(|extras| extras.last_hook_pc);
                    if line.is_some()
                        && (line != last_hook_line
                            || last_hook_pc.is_some_and(|previous| frame.pc <= previous))
                    {
                        frame.extras().last_hook_line = line;
                        line.map(DebugHookEvent::Line)
                    } else {
                        None
                    }
                };
                if let Some(event) = line_event
                    && { self.fire_debug_hook(frames, event)? }
                {
                    let frame = &mut frames[frame_index];
                    frame.extras().skip_hook_pc_once = Some(frame.pc);
                    return Ok(None);
                }
                if !skip_instruction_hooks && self.fire_count_hook(frames)? {
                    let frame = &mut frames[frame_index];
                    frame.extras().skip_hook_pc_once = Some(frame.pc);
                    return Ok(None);
                }
                // Count, call, and return hooks still let the fast loop
                // run, hooked. An instruction the prologue skipped runs here.
                if !skip_instruction_hooks
                    && !std::mem::take(&mut general_once)
                    && self.fast_path_eligible_hooked(&frames[frame_index], single_step)
                {
                    let mode = self.fast_hook_mode(frames);
                    if mode != FastHookMode::Off {
                        let (exit, counted) = if let FastHookMode::Hooked { count, calls } = mode {
                            let frame = &frames[frame_index];
                            let prologue_counted = count
                                && frame.chunk.code[frame.pc]
                                    .opcode()
                                    .map_or(true, OpCode::counts_for_hooks);
                            self.interpret_fast_hooked(frames, count, calls, prologue_counted)?
                        } else {
                            (self.interpret_fast(frames)?, true)
                        };
                        match exit {
                            FastExit::Fallback
                            | FastExit::IndexMiss
                            | FastExit::NewIndexMiss
                            | FastExit::Declined => {
                                frame_index = frames.len() - 1;
                                if !counted {
                                    general_once = true;
                                    continue;
                                }
                            }
                            FastExit::Safepoint => {
                                self.gc_safepoint(frames)?;
                                self.gc_observed_allocation_serial =
                                    self.runtime.allocation_serial();
                                continue;
                            }
                            FastExit::Yielded => return Ok(None),
                            FastExit::Returned => {
                                return Ok(Some(std::mem::take(&mut self.host_returned)));
                            }
                            FastExit::Backedge | FastExit::CountHook => continue,
                        }
                    }
                }
            }
            if !instruction_hooks_active
                && self.fast_path_eligible(&frames[frame_index], single_step)
            {
                match self.interpret_fast(frames)? {
                    // The instruction at `pc` runs below on the general path.
                    // Inline calls and returns may have switched the active
                    // frame; they only do so where the prologue above has
                    // nothing to do for the new one.
                    FastExit::Fallback
                    | FastExit::IndexMiss
                    | FastExit::NewIndexMiss
                    | FastExit::Declined => {
                        frame_index = frames.len() - 1;
                        #[cfg(test)]
                        note_general_fallback(&frames[frame_index]);
                    }
                    FastExit::Safepoint => {
                        self.gc_safepoint(frames)?;
                        self.gc_observed_allocation_serial = self.runtime.allocation_serial();
                        continue;
                    }
                    FastExit::Yielded => return Ok(None),
                    FastExit::Returned => return Ok(Some(std::mem::take(&mut self.host_returned))),
                    FastExit::Backedge => continue,
                    FastExit::CountHook => unreachable!("only a counting loop stops for the hook"),
                }
            }
            // The fast loop hooks the instructions it runs or declines, and
            // the recorder drops a repeat of the instruction it declined.
            if self.jit.trace.recording() {
                let hooks_active = instruction_hooks_active && !self.hooks_allow_traces(frames);
                self.record_general_hook(frames, frame_index, hooks_active);
            }
            let instruction = {
                let frame = &mut frames[frame_index];
                let instruction = *frame
                    .chunk
                    .code
                    .get(frame.pc)
                    .ok_or(VmError::ProgramCounterOutOfBounds(frame.pc))?;
                frame.last_interpreted_pc = crate::vm::frame::LastPc::some(frame.pc);
                if instruction_hooks_active {
                    frame.extras().last_hook_pc = Some(frame.pc);
                }
                frame.pc += 1;
                instruction
            };
            let opcode = instruction.opcode().map_err(|error| {
                VmError::InvalidChunk(ValidationError::InvalidOpcode {
                    pc: frames[frame_index].pc - 1,
                    error,
                })
            })?;
            #[cfg(feature = "op-stats")]
            {
                crate::vm::op_stats::general_dispatch(opcode as u8);
                if instruction_hooks_active
                    && matches!(
                        opcode,
                        OpCode::Call
                            | OpCode::ForInCall
                            | OpCode::CallOne
                            | OpCode::CallVararg
                            | OpCode::CallOpen
                            | OpCode::TailCall
                            | OpCode::TailCallVararg
                            | OpCode::TailCallOpen
                    )
                {
                    crate::vm::op_stats::bump(crate::vm::op_stats::Counter::GeneralCallsHooked);
                }
            }
            // Per-step profiling is off by default; one predictable branch
            // keeps the hot loop free of histogram accounting unless a probe
            // asked for it. Event counters in hot handlers are gated the same
            // way.
            if self.interpreter_profiling {
                bump(&self.interpreter_counters.steps);
                let opcode_index = opcode as usize;
                if opcode_index < self.interpreter_counters.steps_by_opcode.len() {
                    bump(&self.interpreter_counters.steps_by_opcode[opcode_index]);
                }
            }
            // Most bytecode in a hot numeric loop cannot allocate. Reading the
            // allocation serial and entering the GC safe-point machinery for every
            // one of those instructions made the interpreter spend a surprising
            // amount of its time asking an idle collector whether it had work.
            // Keep the precise post-allocation check for instructions that may
            // enter the runtime, invoke a metamethod, or allocate directly.
            let may_allocate = opcode_may_allocate(opcode);
            let mut fused_backedge = false;
            match opcode {
                OpCode::Call => {
                    self.call_frame(frames, instruction)?;
                    self.note_call_landing(frames, frame_index + 1);
                }
                OpCode::ForInCall => {
                    self.for_in_call_frame(frames, instruction)?;
                    self.note_call_landing(frames, frame_index + 1);
                }
                OpCode::ForInLoop => {
                    let frame = frames.at_mut(frame_index);
                    for_in_loop_instruction(frame.registers, &mut frame.frame.pc, instruction)?;
                }
                OpCode::CallOne => {
                    self.call_one_frame(frames, instruction)?;
                    self.note_call_landing(frames, frame_index + 1);
                }
                OpCode::CallVararg => self.call_vararg_frame(frames, instruction)?,
                OpCode::CallOpen => {
                    self.call_open_frame(frames, instruction)?;
                    self.note_call_landing(frames, frame_index + 1);
                }
                OpCode::TailCall => {
                    self.tail_call_frame(frames, instruction, TailCallArguments::Fixed)?;
                }
                OpCode::TailCallVararg => {
                    self.tail_call_frame(frames, instruction, TailCallArguments::Vararg)?;
                }
                OpCode::TailCallOpen => {
                    self.tail_call_frame(frames, instruction, TailCallArguments::Open)?;
                }
                OpCode::Closure
                | OpCode::SetUpvalue
                | OpCode::CloseUpvalue
                | OpCode::Vararg
                | OpCode::TableVararg
                | OpCode::TableOpen => {
                    self.execute_frame_local(frames, frame_index, instruction, opcode)?;
                }
                OpCode::GetUpvalue => {
                    self.get_upvalue(&mut frames.at_mut(frame_index), instruction)?;
                }
                OpCode::MarkClose => {
                    self.mark_close_slot(&mut frames.at_mut(frame_index), instruction.a())?;
                }
                OpCode::CloseLocal => {
                    self.start_close_local(frames, instruction.a(), Value::Nil)?;
                }
                OpCode::Return => {
                    if !self.debug_hooks.is_empty() && self.return_hook_pause(frames)? {
                        return Ok(None);
                    }
                    if instruction.b() == 1 {
                        let value = frames
                            .at_mut(frame_index)
                            .registers
                            .get(usize::from(instruction.a()))
                            .cloned()
                            .ok_or_else(|| VmError::RegisterRange {
                                start: instruction.a(),
                                count: 1,
                            })?;
                        if let Some(values) = self.finish_single_return_frame(frames, value)? {
                            return Ok(Some(values));
                        }
                    } else {
                        let values = frame_values(
                            &frames.at(frame_index),
                            instruction.a(),
                            instruction.b(),
                        )?;
                        if let Some(values) = self.finish_frame(frames, values)? {
                            return Ok(Some(values));
                        }
                    }
                }
                OpCode::ReturnVararg => {
                    if !self.debug_hooks.is_empty() && self.return_hook_pause(frames)? {
                        return Ok(None);
                    }
                    let mut values = fixed_frame_values(
                        &frames.at(frame_index),
                        instruction.a(),
                        instruction.b(),
                    )?;
                    values.extend(self.frame_varargs(frames, frame_index)?);
                    if let Some(values) = self.finish_frame(frames, values)? {
                        return Ok(Some(values));
                    }
                }
                OpCode::ReturnOpen => {
                    if !self.debug_hooks.is_empty() && self.return_hook_pause(frames)? {
                        return Ok(None);
                    }
                    let mut values = fixed_frame_values(
                        &frames.at(frame_index),
                        instruction.a(),
                        instruction.b(),
                    )?;
                    let open = frames[frame_index].take_open_results();
                    values.extend(open);
                    if let Some(values) = self.finish_frame(frames, values)? {
                        return Ok(Some(values));
                    }
                }
                OpCode::GetGlobal => self.get_global(frames, instruction)?,
                OpCode::SetGlobal => self.set_global(frames, instruction)?,
                OpCode::DeclareGlobal => self.declare_global(frames, instruction)?,
                OpCode::LoadConstantWide => {
                    let index = consume_wide_index(&mut frames[frame_index])?;
                    let frame = frames.at_mut(frame_index);
                    load_wide_constant(
                        &frame.frame.chunk,
                        frame.registers,
                        instruction,
                        index,
                        self.compatibility,
                    )?;
                }
                OpCode::GetGlobalWide => {
                    let index = consume_wide_index(&mut frames[frame_index])?;
                    self.get_global_index(frames, instruction, index)?;
                }
                OpCode::SetGlobalWide => {
                    let index = consume_wide_index(&mut frames[frame_index])?;
                    self.set_global_index(frames, instruction, index)?;
                }
                OpCode::DeclareGlobalWide => {
                    let index = consume_wide_index(&mut frames[frame_index])?;
                    self.declare_global_index(frames, instruction, index)?;
                }
                OpCode::GetTable => self.get_table(frames, instruction)?,
                OpCode::SetTable => self.set_table(frames, instruction)?,
                OpCode::GetField => self.get_field(frames, instruction)?,
                OpCode::SelfField => self.self_field(frames, instruction)?,
                OpCode::SetField => self.set_field(frames, instruction)?,
                OpCode::GetTabUp => self.get_tab_up(frames, instruction)?,
                OpCode::SetTabUp => self.set_tab_up(frames, instruction)?,
                OpCode::DeclareTable => self.declare_table(frames, instruction)?,
                OpCode::Concat => self.concat_instruction(frames, instruction)?,
                // Keep one instruction per dispatch here. Straight-line batches
                // regressed numeric and field-read workloads by 20 to 45%.
                OpCode::Add | OpCode::AddK => {
                    // The recorder sees each guest instruction on its own.
                    fused_backedge = !self.jit.trace.recording()
                        && self.try_fused_add_numeric_for_loop(frames, instruction, opcode)?;
                    if !fused_backedge {
                        self.scalar_instruction(frames, instruction, opcode)?;
                    }
                }
                OpCode::Negate
                | OpCode::BitNot
                | OpCode::SubtractK
                | OpCode::MultiplyK
                | OpCode::DivideK
                | OpCode::ModuloK
                | OpCode::Subtract
                | OpCode::Multiply
                | OpCode::Divide
                | OpCode::Modulo
                | OpCode::Power
                | OpCode::FloorDivide
                | OpCode::BitAnd
                | OpCode::BitOr
                | OpCode::BitXor
                | OpCode::ShiftLeft
                | OpCode::ShiftRight
                | OpCode::Equal
                | OpCode::LessThan
                | OpCode::LessEqual
                | OpCode::Length => self.scalar_instruction(frames, instruction, opcode)?,
                OpCode::TestEqual
                | OpCode::TestLessThan
                | OpCode::TestLessEqual
                | OpCode::TestEqualK
                | OpCode::TestLessThanK
                | OpCode::TestLessEqualK => self.test_instruction(frames, instruction, opcode)?,
                OpCode::Halt => {
                    if !self.debug_hooks.is_empty() && self.return_hook_pause(frames)? {
                        return Ok(None);
                    }
                    if let Some(values) = self.finish_frame(frames, Vec::new())? {
                        return Ok(Some(values));
                    }
                }
                OpCode::LoadNil | OpCode::ClearRange | OpCode::LoadConstant | OpCode::Move => {
                    // Register moves and constant loads are the most frequent
                    // interpreter operations. Dispatch them to `load_instruction`
                    // directly instead of the generic plain-instruction path to
                    // skip a second opcode match and call layer; the semantics
                    // are identical.
                    let frame = frames.at_mut(frame_index);
                    load_instruction(
                        &frame.frame.chunk,
                        frame.registers,
                        instruction,
                        opcode,
                        self.compatibility,
                    )?;
                }
                OpCode::Jump | OpCode::JumpIfFalse => {
                    // Branches are nearly as frequent. Same direct treatment as
                    // the register operations above.
                    let frame = frames.at_mut(frame_index);
                    branch_instruction(frame.registers, &mut frame.frame.pc, instruction, opcode)?;
                }
                _ => {
                    // Plain register operations use the same measured dispatch
                    // path. Reintroduce batching only with a benchmark win.
                    self.execute_plain(&mut frames.at_mut(frame_index), instruction, opcode)?;
                }
            }
            let backward_branch = fused_backedge
                || matches!(
                    opcode,
                    OpCode::Jump
                        | OpCode::JumpIfFalse
                        | OpCode::NumericForLoop
                        | OpCode::ForInLoop
                        | OpCode::TestEqual
                        | OpCode::TestLessThan
                        | OpCode::TestLessEqual
                        | OpCode::TestEqualK
                        | OpCode::TestLessThanK
                        | OpCode::TestLessEqualK
                ) && frames
                    .get(frame_index)
                    .is_some_and(|frame| frame.pc <= frame.last_interpreted_pc.unwrap_or(frame.pc));
            // A suspended coroutine pauses this invocation: the resuming
            // caller polls `pending_yield` between `step` calls.
            if self.yield_pending() {
                // A recording that sleeps through a resume is not the thread
                // that yields here: the switch back is what it waits for.
                if self.jit.trace.recording() && !self.jit.trace.recording_waits_for_resume() {
                    self.jit
                        .trace
                        .abort_recording(&mut self.jit.statistics.trace, TraceAbortReason::Yield);
                }
                return Ok(None);
            }
            // A backedge is a safe point, but it need not rebuild roots unless GC
            // debt actually requests a collection. This keeps allocation-free
            // loops in the interpreter's register dispatch path.
            //
            // Speculative `may_allocate` instructions additionally require an
            // observed allocation before they may step the collector. Stepping
            // right after a raw table read, for example, would snapshot the
            // freshly loaded value as a root — a `while t[k] do` loop would
            // re-mark its own condition value on every iteration and weak
            // entries would never clear. Lua likewise advances its collector
            // only from allocation points; backedges stay unconditional so an
            // active cycle still finishes inside allocation-free loops.
            if (backward_branch
                || (may_allocate
                    && self.runtime.allocation_serial() != self.gc_observed_allocation_serial))
                && self.runtime.gc_needs_collection()
            {
                self.gc_safepoint(frames)?;
                self.gc_observed_allocation_serial = self.runtime.allocation_serial();
            }
            if single_step {
                return Ok(None);
            }
        }
    }

    /// Runs the instructions that only touch the frame at `frame_index` (and
    /// the heap through it): closure creation, upvalue stores, and the vararg
    /// expansions. The dispatch loop and the fast loop's miss path share this,
    /// so the two cannot drift apart.
    #[inline(always)]
    pub(crate) fn execute_frame_local(
        &mut self,
        frames: &mut CallStack,
        frame_index: usize,
        instruction: Instruction,
        opcode: OpCode,
    ) -> VmResult<()> {
        match opcode {
            OpCode::Closure => {
                self.create_closure(frames, instruction)?;
            }
            OpCode::SetUpvalue => self.set_upvalue(&frames.at(frame_index), instruction)?,
            OpCode::CloseUpvalue => {
                self.close_captured_slot(frames, frame_index, instruction.a())?;
            }
            OpCode::Vararg => {
                if self.compatibility == CompatibilityProfile::Lua55 {
                    let values = self.frame_varargs(frames, frame_index)?;
                    vararg_values_instruction(
                        frames.at_mut(frame_index).registers,
                        instruction,
                        &values,
                    )?;
                } else {
                    let frame = frames.at_mut(frame_index);
                    vararg_values_instruction(frame.registers, instruction, frame.varargs)?;
                }
            }
            OpCode::TableVararg => {
                if self.compatibility == CompatibilityProfile::Lua55 {
                    let values = self.frame_varargs(frames, frame_index)?;
                    table_vararg_instruction(
                        &mut self.runtime,
                        &frames.at(frame_index),
                        instruction,
                        &values,
                    )?;
                } else {
                    let frame = frames.at(frame_index);
                    table_vararg_instruction(
                        &mut self.runtime,
                        &frame,
                        instruction,
                        frame.varargs,
                    )?;
                }
            }
            OpCode::TableOpen => {
                table_open_instruction(
                    &mut self.runtime,
                    &mut frames.at_mut(frame_index),
                    instruction,
                )?;
            }
            _ => return Err(VmError::InvalidDispatch(opcode)),
        }
        Ok(())
    }

    pub(crate) fn gc_safepoint(&mut self, frames: &mut CallStack) -> VmResult<()> {
        self.gc_safepoint_with_values(frames, &[])
    }

    pub(crate) fn gc_safepoint_with_values(
        &mut self,
        frames: &mut CallStack,
        values: &[Value],
    ) -> VmResult<()> {
        if self.runtime.gc_needs_collection() {
            // Only minor collections, cycle starts, and a major cycle's
            // atomic step read the VM roots. Other steps rely on barriers.
            let roots = self
                .runtime
                .gc_step_needs_roots()
                .then(|| self.gc_roots_with_coroutines(frames, values));
            if matches!(
                self.runtime
                    .gc_automatic_step(roots, AUTOMATIC_GC_STEP_KILOBYTES),
                GcStepResult::Complete
            ) {
                self.prune_collected_coroutines();
                self.advance_userdata_finalizers(frames, FinalizerCompletion::Automatic)?;
            }
        }
        Ok(())
    }

    pub(crate) fn execute_plain(
        &mut self,
        frame: &mut FrameMut<'_>,
        instruction: Instruction,
        opcode: OpCode,
    ) -> VmResult<()> {
        match opcode {
            OpCode::LoadNil | OpCode::ClearRange | OpCode::LoadConstant | OpCode::Move => {
                load_instruction(
                    &frame.frame.chunk,
                    frame.registers,
                    instruction,
                    opcode,
                    self.compatibility,
                )?;
            }
            OpCode::Not => arithmetic_instruction(
                frame.registers,
                instruction,
                opcode,
                self.compatibility.float_arith(),
            )?,
            OpCode::Jump | OpCode::JumpIfFalse => {
                branch_instruction(frame.registers, &mut frame.frame.pc, instruction, opcode)?;
            }
            OpCode::NumericForPrep | OpCode::NumericForLoop => {
                numeric_for_instruction(
                    frame.registers,
                    &mut frame.frame.pc,
                    instruction,
                    opcode,
                    self.compatibility,
                )?;
            }
            OpCode::NewTable => {
                // A recording sees the constructor and the reads after it one
                // instruction at a time, so it can keep the table virtual.
                if !self.jit.trace.recording() && self.try_scalar_table_fragment(frame) {
                    return Ok(());
                }
                table_instruction(
                    &mut self.runtime,
                    frame.registers,
                    &frame.frame.chunk,
                    frame.frame.pc - 1,
                    instruction,
                    opcode,
                )?;
            }
            OpCode::Call
            | OpCode::ForInCall
            | OpCode::ForInLoop
            | OpCode::CallOne
            | OpCode::CallVararg
            | OpCode::CallOpen
            | OpCode::TailCall
            | OpCode::TailCallVararg
            | OpCode::TailCallOpen
            | OpCode::Closure
            | OpCode::GetUpvalue
            | OpCode::SetUpvalue
            | OpCode::CloseUpvalue
            | OpCode::MarkClose
            | OpCode::CloseLocal
            | OpCode::Return
            | OpCode::ReturnVararg
            | OpCode::ReturnOpen
            | OpCode::Vararg
            | OpCode::TableVararg
            | OpCode::TableOpen
            | OpCode::GetGlobal
            | OpCode::SetGlobal
            | OpCode::DeclareGlobal
            | OpCode::LoadConstantWide
            | OpCode::GetGlobalWide
            | OpCode::SetGlobalWide
            | OpCode::DeclareGlobalWide
            | OpCode::ExtraArgument
            | OpCode::GetTable
            | OpCode::SetTable
            | OpCode::GetField
            | OpCode::SelfField
            | OpCode::SetField
            | OpCode::GetTabUp
            | OpCode::SetTabUp
            | OpCode::DeclareTable
            | OpCode::Negate
            | OpCode::BitNot
            | OpCode::AddK
            | OpCode::SubtractK
            | OpCode::MultiplyK
            | OpCode::DivideK
            | OpCode::ModuloK
            | OpCode::Add
            | OpCode::Subtract
            | OpCode::Multiply
            | OpCode::Divide
            | OpCode::Modulo
            | OpCode::Power
            | OpCode::FloorDivide
            | OpCode::BitAnd
            | OpCode::BitOr
            | OpCode::BitXor
            | OpCode::ShiftLeft
            | OpCode::ShiftRight
            | OpCode::Concat
            | OpCode::Equal
            | OpCode::LessThan
            | OpCode::LessEqual
            | OpCode::Length
            | OpCode::TestEqual
            | OpCode::TestLessThan
            | OpCode::TestLessEqual
            | OpCode::TestEqualK
            | OpCode::TestLessThanK
            | OpCode::TestLessEqualK
            | OpCode::Halt => return Err(VmError::InvalidDispatch(opcode)),
        }
        Ok(())
    }
}

/// Whether an instruction can charge the runtime's garbage collector while it
/// runs through the baseline interpreter. Plain register moves, branches, and
/// numeric loop latches are deliberately absent: they are the tight inner
/// loop for a large share of Lua programs.
pub(crate) const fn opcode_may_allocate(opcode: OpCode) -> bool {
    matches!(
        opcode,
        OpCode::Call
            | OpCode::ForInCall
            | OpCode::CallOne
            | OpCode::CallVararg
            | OpCode::CallOpen
            | OpCode::TailCall
            | OpCode::TailCallVararg
            | OpCode::TailCallOpen
            | OpCode::Closure
            | OpCode::GetGlobal
            | OpCode::SetGlobal
            | OpCode::DeclareGlobal
            | OpCode::GetGlobalWide
            | OpCode::SetGlobalWide
            | OpCode::DeclareGlobalWide
            | OpCode::NewTable
            | OpCode::GetTable
            | OpCode::SetTable
            | OpCode::GetField
            | OpCode::SelfField
            | OpCode::SetField
            | OpCode::GetTabUp
            | OpCode::SetTabUp
            | OpCode::DeclareTable
            | OpCode::Concat
            | OpCode::Vararg
            | OpCode::TableVararg
            | OpCode::TableOpen
            | OpCode::Negate
            | OpCode::BitNot
            | OpCode::AddK
            | OpCode::SubtractK
            | OpCode::MultiplyK
            | OpCode::DivideK
            | OpCode::ModuloK
            | OpCode::Add
            | OpCode::Subtract
            | OpCode::Multiply
            | OpCode::Divide
            | OpCode::Modulo
            | OpCode::Power
            | OpCode::FloorDivide
            | OpCode::BitAnd
            | OpCode::BitOr
            | OpCode::BitXor
            | OpCode::ShiftLeft
            | OpCode::ShiftRight
            | OpCode::Equal
            | OpCode::LessThan
            | OpCode::LessEqual
            | OpCode::TestEqual
            | OpCode::TestLessThan
            | OpCode::TestLessEqual
            | OpCode::TestEqualK
            | OpCode::TestLessThanK
            | OpCode::TestLessEqualK
            | OpCode::Length
    )
}
