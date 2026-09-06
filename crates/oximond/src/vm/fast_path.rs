//! Straight-line interpreter fast path for register, numeric, branch, and raw table instructions.
//!
//! The general dispatch loop in `interp.rs` re-checks hooks, native entry,
//! coroutine yields, and GC safepoints for
//! every instruction. Most instructions in a hot Lua loop can do none of
//! those things. This loop runs them back to back on the top frame and hands
//! control back at the first instruction that needs the general machinery,
//! with the program counter still pointing at it.

mod micro;

pub(crate) use micro::{Micro, Stop};

use crate::bytecode::{Chunk, Constant, Instruction, OpCode};
use crate::compat::{CompatibilityProfile, ExecutionMode, FloatArith, FloatModulo};
use crate::pattern::Pattern;
use crate::runtime::DenseScalarStore;
use crate::runtime::cint64::{self, CInt64, CIntArith};
use crate::runtime::{
    BuiltinFunction, ClosureHandle, DirectNativeKind, NativeCallKind, NativeCallback,
    NativeFunctionHandle, NativePairCallback, NativePlan, NextKey, NextRef, Runtime, RuntimeError,
    SiteProbe, SiteStore, TableHandle, Value,
};
use crate::vm::BACKEDGE_GC_POLL_INTERVAL;
use crate::vm::arith::{
    constant_value, index_metamethod_key, integer_float_order, integer_floor_mod_nonzero,
    integer_to_float, integer_to_number, newindex_metamethod_key, nonfinite_numerals,
    numeric_order, numeric_scalar_result, profile_integer_value,
};
use crate::vm::calls::IpairsIndex;
use crate::vm::continuations::gsub_literal;
use crate::vm::errors::call_argument_error;
use crate::vm::jit::trace::{HookPoint, HookSource, Surroundings, TraceAbortReason};
use crate::vm::registers::{
    clone_register_value, closure_handle, copy_register, copy_register_words,
    create_constructor_table_fast, push_result, set_register, store_clone, store_clone_scalar,
    store_result, write_call_results,
};
use crate::vm::stack::{
    ActivationArguments, CallStack, Entered, HandlerKey, PlainCall, REGISTER_WINDOW, TailStep,
    ensure_call_depth, register_window,
};
use crate::vm::{
    CALL_HINT_NUMERIC_PLAN, CALL_HINT_PLAIN, CALL_HINT_TABLE_PLAN, CallReturn, Frame, IndexAction,
    IndexCompletion, IndexContinuation, InlineCallArguments, MAX_LUA_CALL_FRAMES, OperatorMode,
    ReturnTarget, ScalarKind, Vm, VmError, VmResult,
};
use std::sync::Arc;

/// How [`Vm::fast_get_field_miss`] ended. An arm turns the two exits into
/// constant ones, so each keeps its own specialised exit tail.
#[derive(Clone, Copy)]
enum FieldMiss {
    /// The value was found, stored, and the site filled.
    Filled,
    /// The table does not hold the key: the general loop looks further.
    IndexMiss,
    /// The general loop repeats the instruction.
    Fallback,
}

/// Marks the branch that calls it as the unlikely one.
#[cold]
#[inline(never)]
fn rare_outcome() {}

/// What [`Vm::store_field`] did.
enum FieldStore {
    /// The value is stored.
    Stored,
    /// The value is stored over another, which a write site of the chunk can
    /// learn.
    Replaced,
    /// The value is stored, and the collector has work to do.
    Collect,
    /// Nothing was written: the key holds no value and the metatable has
    /// `__newindex`, which the general loop resolves.
    Missed,
    /// Nothing was written: the general loop repeats the store with
    /// `__newindex` or its error.
    Declined,
}

/// How a concatenation in the fast loop ended.
enum ConcatStep {
    /// The result is in the destination register.
    Done,
    /// The result is in the destination register, and the collector has work
    /// to do.
    Collect,
    /// Nothing was written: the miss path runs the instruction.
    Declined,
}

/// What [`Vm::new_table_in_place`] did.
enum TableCreate {
    /// The table is in the destination register.
    Created,
    /// The table is in the destination register, and the collector has work
    /// to do.
    Collect,
    /// Nothing was written: the miss path runs the instruction, which may
    /// take a scalar fragment or report an exhausted handle space.
    Declined,
}

/// How a host call a native trace makes went (`Vm::host_call_for_trace`).
pub(crate) enum TraceHostCall {
    /// The call was not made; the trace leaves in front of it.
    Declined,
    /// The call was made and the trace takes these result words.
    Answered([i64; 3]),
    /// The call was made and the trace leaves.
    Missed,
}

/// Why the fast path handed control back to the general loop.
pub(crate) enum FastExit {
    /// The instruction at `frame.pc` needs the general loop.
    Fallback,
    /// A completed instruction left the collector with work to do.
    Safepoint,
    /// A direct `coroutine.yield` parked its values in `pending_yield`; the
    /// dispatch loop returns to the resumer.
    Yielded,
    /// A backedge completed in tracing mode; a trace may own its header.
    Backedge,
    /// The `GetTable` or `GetField` at `frame.pc` missed its raw lookup. Resolved outside
    /// the loop, so its code stays out of the hot dispatch; never returned
    /// from [`Vm::interpret_fast`].
    IndexMiss,
    /// The `SetField` or `SetTable` at `frame.pc` found no raw value for its
    /// key and the metatable has `__newindex`, and it wrote nothing. The
    /// resolver may skip the raw probe. Resolved like `IndexMiss`; never
    /// returned from [`Vm::interpret_fast`].
    NewIndexMiss,
    /// The scalar miss path already ran or declined the instruction at
    /// `frame.pc`; hands control to the dispatch loop like `Fallback`, but
    /// without a second attempt. Never returned from [`Vm::interpret_fast`].
    Declined,
    /// The count hook fires before the instruction at `frame.pc`, which the
    /// counting loop left uncounted for the dispatch loop's prologue.
    CountHook,
    /// The bottom activation of a host entry returned: the run is over, and
    /// its values wait in `host_returned` (see [`Vm::return_to_host`]).
    Returned,
}

/// Stores the fast loop's resume position in the activation it runs: `pc`,
/// and the last executed instruction, which the loop derives from `jump`
/// (see `run_fast_activations`).
#[inline(always)]
fn store_loop_state(frame: &mut Frame, pc: usize, jump: u64) {
    frame.pc = pc;
    let after_jump = (jump >> 32) as usize;
    if pc != (jump as u32) as usize {
        frame.last_interpreted_pc = crate::vm::frame::LastPc::some(pc - 1);
    } else if after_jump != 0 {
        frame.last_interpreted_pc = crate::vm::frame::LastPc::some(after_jump - 1);
    }
}

/// What an out-of-line call routine of the fast loop did.
enum LoopStep {
    /// The activation switched: the loop goes on in the new top frame.
    Continue,
    /// The activation switched, and the collector has work to do.
    Safepoint,
    /// Nothing changed: the dispatch loop decides.
    Declined,
}

/// `Vararg`: copies the first `wanted` of the `available` varargs just below
/// the window that starts at `window` to the registers from `a`, and nil past
/// the last. Out of line, which keeps its loops out of the dispatch loop.
///
/// # Safety
///
/// `window` is the first register of the running window, in the value stack,
/// which holds the `available` varargs of the frame just below it, and
/// `a + wanted` registers.
#[allow(unsafe_code)]
#[inline(never)]
unsafe fn vararg_in_place(window: *mut Value, a: usize, wanted: u8, available: usize) {
    // SAFETY: the contract.
    unsafe { crate::vm::stack::copy_varargs(window, a, wanted, available) }
}

/// What a call or return routine reports to the fast loop. The error is
/// boxed so that the usual result fits two registers.
type FastStep = Result<Option<FastExit>, Box<VmError>>;

impl Vm {
    /// Whether the top frame may run on [`Self::interpret_fast`].
    ///
    /// Hooks, profiling, and single steps observe individual
    /// instructions.
    #[inline(always)]
    pub(crate) fn fast_path_eligible(&self, frame: &Frame, single_step: bool) -> bool {
        self.debug_hooks.is_empty() && self.fast_path_eligible_hooked(frame, single_step)
    }

    /// [`Self::fast_path_eligible`] before the hook check: whether the top
    /// frame could run on the fast loop if no hook observed it.
    #[inline(always)]
    pub(crate) fn fast_path_eligible_hooked(&self, frame: &Frame, single_step: bool) -> bool {
        !single_step
            && !self.interpreter_profiling
            && !self.jit.trace.has_loaded_profile()
            // Entering only to fall back at once costs a full prologue.
            && frame
                .chunk
                .code
                .get(frame.pc)
                .and_then(|instruction| instruction.opcode().ok())
                .is_some_and(|opcode| {
                    handles(opcode)
                        || matches!(
                            opcode,
                            OpCode::Call
                                | OpCode::ForInCall
                                | OpCode::CallOne
                                | OpCode::TailCall
                                | OpCode::Return
                                // A handler with an empty body starts on its
                                // return, which the fast loop takes too.
                                | OpCode::Halt
                        )
                        // The interpreter's own loop runs the calls, tail
                        // calls and returns that pass varargs on, and reads
                        // them; any other run would leave at once.
                        || (self.execution_mode == ExecutionMode::Interpreter
                            && self.debug_hooks.is_empty()
                            && matches!(
                                opcode,
                                OpCode::Vararg
                                    | OpCode::CallVararg
                                    | OpCode::TailCallVararg
                                    | OpCode::TailCallOpen
                                    | OpCode::ReturnVararg
                            ))
                })
    }

    /// Runs fast-path instructions until one needs the general loop.
    ///
    /// Every executed instruction leaves `pc` and `last_interpreted_pc`
    /// exactly as the general loop would. Operations whose operands miss the
    /// fast cases return [`FastExit::Fallback`] before changing any state, so
    /// the general loop repeats them with full metamethod and error handling.
    ///
    /// Calls to closures classified as plain Lua bodies, and returns from
    /// frames without interior state to a direct caller, switch activations
    /// here instead of leaving the loop. A return from the bottom frame goes
    /// back to the caller's dispatch loop.
    ///
    /// A captured local lives in its register, and its upvalue is an open
    /// cell that reads and writes that register (see `CallStack::open`), so
    /// nothing has to be copied around calls, returns and exits. Returns,
    /// tail calls and `CloseUpvalue` close the upvalues of the registers
    /// that go away.
    pub(crate) fn interpret_fast(&mut self, stack: &mut CallStack) -> VmResult<FastExit> {
        self.interpret_fast_inner::<false>(stack)
    }

    /// [`Self::interpret_fast`] under a hook, observing what
    /// `fast_hook_count` and `fast_hook_calls` ask for. See
    /// [`Self::interpret_fast_hooked`].
    pub(crate) fn interpret_fast_with_hooks(
        &mut self,
        stack: &mut CallStack,
    ) -> VmResult<FastExit> {
        // Only a count hook lets traces record and run. For any other the
        // recorder would abort on the next dispatch-loop instruction anyway.
        if self.jit.trace.recording() && !self.count_traces_allowed(stack) {
            self.jit
                .trace
                .abort_recording(&mut self.jit.statistics.trace, TraceAbortReason::Flushed);
        }
        self.interpret_fast_inner::<true>(stack)
    }

    /// Whether the hooked fast loop counts for a count hook that traces
    /// share: the loop observes no call or return, and every installed hook
    /// is one a counted trace can run under.
    fn count_traces_allowed(&self, stack: &CallStack) -> bool {
        self.fast_hook_count && !self.fast_hook_calls && self.hooks_allow_traces(stack)
    }

    #[inline(always)]
    fn interpret_fast_inner<const HOOKED: bool>(
        &mut self,
        stack: &mut CallStack,
    ) -> VmResult<FastExit> {
        // The resumes this loop makes in place (`Vm::resume_in_place`) are
        // the entries of `in_place` past the ones it found. Whatever way the
        // loop ends, a thread that still runs in place of its resumer goes
        // on as a nested resume, and the loop takes the resumer up again.
        let base = self.in_place.len();
        let outer = std::mem::replace(
            &mut self.in_place_base,
            if HOOKED { usize::MAX } else { base },
        );
        let exit = loop {
            let exit = self.interpret_fast_runs::<HOOKED>(stack);
            if self.in_place.len() <= base {
                break exit;
            }
            if let Some(exit) = self.leave_in_place_from(stack, base, exit) {
                break exit;
            }
        };
        self.in_place_base = outer;
        exit
    }

    /// The runs of the fast loop of one entry: [`Self::interpret_fast_inner`]
    /// without the bookkeeping of the resumes it makes in place.
    #[inline(always)]
    fn interpret_fast_runs<const HOOKED: bool>(
        &mut self,
        stack: &mut CallStack,
    ) -> VmResult<FastExit> {
        // A hooked loop offers backedges to the trace engine only under a
        // count hook that traces share, and it records only then.
        let hooked_traces = HOOKED && self.count_traces_allowed(stack);
        loop {
            // The recording instantiation hooks every instruction, and the
            // tracing one offers backedges and calls to native code. Each is
            // chosen once per entry here, so the others carry no check.
            let tracing =
                self.execution_mode == ExecutionMode::Tracing && self.runtime.jit_enabled();
            let exit = if HOOKED {
                if hooked_traces && tracing {
                    // Traces share only count hooks that observe no call.
                    if self.jit.trace.recording() {
                        self.run_fast_activations::<true, true, true, false>(stack, true)
                    } else {
                        self.run_fast_activations::<false, true, true, false>(stack, true)
                    }
                } else {
                    if self.jit.trace.recording() {
                        self.jit.trace.abort_recording(
                            &mut self.jit.statistics.trace,
                            TraceAbortReason::Flushed,
                        );
                    }
                    if self.fast_hook_count && !self.fast_hook_calls {
                        self.run_fast_activations::<false, true, true, false>(stack, false)
                    } else {
                        self.run_fast_activations::<false, true, false, false>(stack, false)
                    }
                }
            } else if self.jit.trace.recording() {
                if tracing {
                    self.run_fast_activations::<true, false, false, false>(stack, true)
                } else {
                    self.jit
                        .trace
                        .abort_recording(&mut self.jit.statistics.trace, TraceAbortReason::Flushed);
                    self.run_fast_activations::<false, false, false, false>(stack, false)
                }
            } else if tracing {
                self.run_fast_activations::<false, false, false, false>(stack, true)
            } else if self.execution_mode == ExecutionMode::Interpreter {
                self.run_fast_activations::<false, false, false, true>(stack, false)
            } else {
                self.run_fast_activations::<false, false, false, false>(stack, false)
            };
            match exit {
                // Misses may call metamethods, whose call and return events
                // the dispatch loop fires.
                Ok(exit @ (FastExit::IndexMiss | FastExit::NewIndexMiss | FastExit::Fallback))
                    if HOOKED && self.fast_hook_calls =>
                {
                    // The dispatch loop runs the store again; a closure the
                    // fast store kept for the resolver must not outlive it.
                    self.runtime.take_newindex_hint();
                    return Self::leave_fast(exit);
                }
                Ok(exit @ (FastExit::IndexMiss | FastExit::NewIndexMiss)) => {
                    #[cfg(test)]
                    {
                        self.index_miss_exits += 1;
                    }
                    match self.index_miss_activation(stack, matches!(exit, FastExit::NewIndexMiss))
                    {
                        Ok(None) => {}
                        Ok(Some(exit)) => return Self::leave_fast(exit),
                        Err(error) => return Err(error),
                    }
                }
                // Arithmetic, comparisons, and lengths that miss the fast
                // cases run here too, without a trip through the dispatch
                // loop. Keeping them on the ordinary fallback exit leaves
                // the fast loop's own code unchanged.
                Ok(FastExit::Fallback) => {
                    #[cfg(test)]
                    {
                        self.fast_fallback_exits += 1;
                    }
                    #[cfg(feature = "op-stats")]
                    crate::vm::op_stats::bump(crate::vm::op_stats::Counter::FastExitsFallback);
                    match self.run_scalar_miss(stack) {
                        Ok(None) => {}
                        Ok(Some(exit)) => return Self::leave_fast(exit),
                        Err(error) => return Err(error),
                    }
                }
                // A header with nothing to enter or record resumes here.
                Ok(FastExit::Backedge) if self.resume_fast_backedge(stack) => {}
                Ok(exit) => return Self::leave_fast(exit),
                Err(error) => return Err(error),
            }
        }
    }

    /// Hands control to the dispatch loop.
    #[inline(never)]
    fn leave_fast(exit: FastExit) -> VmResult<FastExit> {
        Ok(match exit {
            FastExit::IndexMiss | FastExit::NewIndexMiss | FastExit::Declined => FastExit::Fallback,
            exit => exit,
        })
    }

    /// Executes the operator instruction at the top frame's `pc` (the
    /// dispatch loop's `scalar_instruction` set: arithmetic, comparisons,
    /// bitwise operations, and lengths; concatenation; and the frame-local
    /// closure, upvalue, and vararg instructions) exactly as that
    /// loop would, with
    /// coercions, metamethod calls, and errors, without leaving fast mode.
    /// `None` continues the fast loop, possibly in a metamethod's new
    /// activation; an exit hands the dispatch loop whatever the fast loop
    /// cannot run next. Any other instruction exits unchanged.
    #[inline(never)]
    fn run_scalar_miss(&mut self, stack: &mut CallStack) -> VmResult<Option<FastExit>> {
        let top = stack.len() - 1;
        let frame = &stack[top];
        let Some(&instruction) = frame.chunk.code.get(frame.pc) else {
            return Ok(Some(FastExit::Fallback));
        };
        if instruction.opcode() == Ok(OpCode::NewTable) {
            return self.new_table_miss(stack, instruction);
        }
        if self.string_arithmetic_in_place(stack, top, instruction) {
            return Ok(None);
        }
        let Ok(
            opcode @ (OpCode::Add
            | OpCode::Subtract
            | OpCode::Multiply
            | OpCode::Divide
            | OpCode::Modulo
            | OpCode::AddK
            | OpCode::SubtractK
            | OpCode::MultiplyK
            | OpCode::DivideK
            | OpCode::ModuloK
            | OpCode::Power
            | OpCode::FloorDivide
            | OpCode::Negate
            | OpCode::Equal
            | OpCode::LessThan
            | OpCode::LessEqual
            | OpCode::TestEqual
            | OpCode::TestLessThan
            | OpCode::TestLessEqual
            | OpCode::TestEqualK
            | OpCode::TestLessThanK
            | OpCode::TestLessEqualK
            | OpCode::BitAnd
            | OpCode::BitOr
            | OpCode::BitXor
            | OpCode::ShiftLeft
            | OpCode::ShiftRight
            | OpCode::BitNot
            | OpCode::Length
            | OpCode::Concat
            | OpCode::Closure
            | OpCode::SetUpvalue
            | OpCode::CloseUpvalue
            | OpCode::Vararg
            | OpCode::TableVararg
            | OpCode::TableOpen),
        ) = instruction.opcode()
        else {
            return Ok(Some(FastExit::Fallback));
        };
        if opcode == OpCode::Length && self.length_in_place(stack, top, instruction) {
            return Ok(None);
        }
        // A plain Lua handler is entered without the scalar path's
        // continuation; the handler's own frame is what runs next.
        let entered = self.operator_miss_hot(stack, instruction, opcode);
        if !entered {
            let frame = &mut stack[top];
            frame.last_interpreted_pc = crate::vm::frame::LastPc::some(frame.pc);
            frame.pc += 1;
        }
        match opcode {
            _ if entered => {}
            OpCode::Concat => self.concat_instruction(stack, instruction)?,
            OpCode::Closure
            | OpCode::SetUpvalue
            | OpCode::CloseUpvalue
            | OpCode::Vararg
            | OpCode::TableVararg
            | OpCode::TableOpen => {
                self.execute_frame_local(stack, top, instruction, opcode)?;
            }
            OpCode::TestEqual
            | OpCode::TestLessThan
            | OpCode::TestLessEqual
            | OpCode::TestEqualK
            | OpCode::TestLessThanK
            | OpCode::TestLessEqualK => self.test_instruction(stack, instruction, opcode)?,
            _ => self.scalar_instruction(stack, instruction, opcode)?,
        }
        // Entering a handler allocates nothing.
        if !entered
            && self.runtime.allocation_serial() != self.gc_observed_allocation_serial
            && self.runtime.gc_needs_collection()
        {
            return Ok(Some(FastExit::Safepoint));
        }
        // A metamethod may have entered a frame the fast loop cannot run,
        // or set a hook.
        let top = stack.len() - 1;
        if !self.fast_path_eligible(&stack[top], false) {
            return Ok(Some(FastExit::Fallback));
        }
        Ok(None)
    }

    /// Stores `source` in upvalue `index`. Out of line, so the clone it needs
    /// stays out of the fast loop. `false` when nothing was written: the
    /// scalar miss path repeats the store and reports the error.
    #[inline(never)]
    fn store_upvalue(
        &mut self,
        upvalues: &[crate::runtime::CellHandle],
        index: usize,
        source: &Value,
    ) -> bool {
        let Some(&cell) = upvalues.get(index) else {
            return false;
        };
        // SAFETY: the running frame roots its captured cells, so a
        // collection keeps them alive.
        #[allow(unsafe_code)]
        unsafe {
            // A number over a number needs no clone, release or write
            // barrier.
            if let Value::Number(number) = *source
                && self.runtime.cell_set_number_live(cell, number)
            {
                return true;
            }
            self.runtime
                .cell_set_live(cell, clone_register_value(source));
        }
        true
    }

    /// Runs the `SetField` `instruction` on `registers` when the table
    /// takes the write raw. Out of line: inlined, the string store cost
    /// every other instruction of the fast loop a spilled register.
    #[inline(never)]
    fn store_field(
        &mut self,
        chunk: &Chunk,
        site: &crate::bytecode::FieldSite,
        registers: &[Value],
        instruction: Instruction,
    ) -> FieldStore {
        let Some(table) = &registers[usize::from(instruction.a())].table_handle_word() else {
            return FieldStore::Declined;
        };
        let source = &registers[usize::from(instruction.c())];
        // A scalar over a scalar, or a scalar into a field a shaped table
        // lacks, is a copy that needs no call.
        match self.runtime.table_site_store_scalar(*table, site, source) {
            SiteStore::Overwrote => return FieldStore::Stored,
            SiteStore::Added => return self.collect_after_store(),
            SiteStore::Missed => {}
        }
        self.store_field_general(chunk, *table, site, source, instruction)
    }

    /// [`Self::store_field`] for the stores that are not a scalar over a
    /// scalar. Out of line, so that the common one saves no registers.
    #[inline(never)]
    fn store_field_general(
        &mut self,
        chunk: &Chunk,
        table: TableHandle,
        site: &crate::bytecode::FieldSite,
        source: &Value,
        instruction: Instruction,
    ) -> FieldStore {
        // A site filled for this table's layout overwrites its field in
        // place, which allocates nothing, so no collection check follows.
        // A shaped table may also gain the field there.
        match self.runtime.table_site_store(table, site, source) {
            SiteStore::Overwrote => return FieldStore::Stored,
            SiteStore::Added => return self.collect_after_store(),
            SiteStore::Missed => {}
        }
        // The name matters only to the general store, so the hit path does
        // not fetch it.
        let Some(Constant::String(name)) = chunk.constants.get(usize::from(instruction.b())) else {
            return FieldStore::Declined;
        };
        let serial = self.runtime.allocation_serial();
        match self.runtime.table_set_string_from(table, name, source) {
            Ok(true)
                if self.runtime.allocation_serial() != self.gc_observed_allocation_serial
                    && self.runtime.gc_needs_collection() =>
            {
                FieldStore::Collect
            }
            // A store that allocated nothing is an overwrite, or a new key
            // in a hashed table with room; the next store to the table may
            // find the field in place. A store that added a field to a
            // shaped table fills the site too: the next table of the shape
            // takes the field through it.
            Ok(true) if self.runtime.allocation_serial() == serial => FieldStore::Replaced,
            Ok(true) if self.runtime.table_is_shaped(table) => FieldStore::Replaced,
            Ok(true) => FieldStore::Stored,
            // `Ok(false)` is the `__newindex` miss; an error declines.
            Ok(false) => FieldStore::Missed,
            Err(_) => FieldStore::Declined,
        }
    }

    /// `Stored`, or `Collect` when the store just made gave the collector
    /// work.
    #[inline(always)]
    fn collect_after_store(&self) -> FieldStore {
        if self.runtime.allocation_serial() != self.gc_observed_allocation_serial
            && self.runtime.gc_needs_collection()
        {
            FieldStore::Collect
        } else {
            FieldStore::Stored
        }
    }

    /// The `GetField` or `GetTabUp` read that missed its cache site:
    /// `table[constant c]` into `target`. `Some` is the exit to take (the
    /// general loop repeats the instruction). Out of line and cold, so the
    /// miss path adds nothing to the hit path's registers.
    #[cold]
    #[inline(never)]
    fn fast_get_field_miss(
        &mut self,
        chunk: &Chunk,
        at: *const Instruction,
        table: TableHandle,
        target: &mut Value,
        instruction: Instruction,
    ) -> FieldMiss {
        let Constant::String(name) = &chunk.constants[usize::from(instruction.c())] else {
            return FieldMiss::Fallback;
        };
        let Some(value) =
            self.runtime
                .table_get_string_cached_hit(table, name, &mut self.string_read_cache)
        else {
            return FieldMiss::IndexMiss;
        };
        store_clone(target, value);
        fill_read_site(&self.runtime, chunk, at, table, name, target);
        FieldMiss::Filled
    }

    /// Runs the `SetTabUp` `instruction`: the field named by constant `b` of
    /// the table in upvalue `a` takes register `c`, when the table takes the
    /// write raw. `Declined` leaves the instruction to the general loop. Out of
    /// line, as [`Self::store_field`] is.
    #[inline(never)]
    fn fast_set_tab_up(
        &mut self,
        chunk: &Chunk,
        upvalues: &[crate::runtime::CellHandle],
        at: *const Instruction,
        registers: &[Value],
        instruction: Instruction,
        #[cfg(feature = "op-stats")] pc: usize,
    ) -> FieldStore {
        let Some(&Value::Table(table)) = upvalues
            .get(usize::from(instruction.a()))
            .and_then(|cell| self.runtime.cell_ref(*cell).ok())
        else {
            return FieldStore::Declined;
        };
        #[cfg(feature = "op-stats")]
        crate::vm::op_stats::field(&self.runtime, chunk, pc, table);
        let Some(Constant::String(name)) = chunk.constants.get(usize::from(instruction.b())) else {
            return FieldStore::Declined;
        };
        let source = &registers[usize::from(instruction.c())];
        if let Some(site) = chunk.store_site_at(at) {
            match self.runtime.table_site_store(table, site, source) {
                SiteStore::Overwrote => return FieldStore::Stored,
                SiteStore::Added => return self.collect_after_store(),
                SiteStore::Missed => {}
            }
        }
        let serial = self.runtime.allocation_serial();
        match self.runtime.table_set_string_from(table, name, source) {
            Ok(true)
                if self.runtime.allocation_serial() != self.gc_observed_allocation_serial
                    && self.runtime.gc_needs_collection() =>
            {
                FieldStore::Collect
            }
            Ok(true) if self.runtime.allocation_serial() == serial => {
                fill_tab_up_store_site(&mut self.runtime, chunk, at, table, instruction);
                FieldStore::Replaced
            }
            Ok(true) => FieldStore::Stored,
            Ok(false) | Err(_) => FieldStore::Declined,
        }
    }

    /// `Closure`: the closure is made and stored in its register, capturing
    /// registers of the running frame as open upvalues. Nothing moves on the
    /// value stack, so the loop goes on in the same activation. An error
    /// declines: the miss path repeats the instruction and reports it
    /// (capturing a register twice finds the cell the first try made).
    #[inline(never)]
    fn closure_in_place(&mut self, stack: &mut CallStack, instruction: Instruction) -> ConcatStep {
        if self.create_closure(stack, instruction).is_err() {
            return ConcatStep::Declined;
        }
        if self.runtime.allocation_serial() != self.gc_observed_allocation_serial
            && self.runtime.gc_needs_collection()
        {
            return ConcatStep::Collect;
        }
        ConcatStep::Done
    }

    /// `CloseUpvalue`: the variable of register `a` goes out of scope. Only
    /// reached when a captured variable lives at or above its slot.
    #[inline(never)]
    fn close_upvalue_in_place(&mut self, stack: &mut CallStack, slot: u32) {
        if let Some(open) = stack.remove_open(slot) {
            // SAFETY: the slot is in the live window of `stack`.
            #[allow(unsafe_code)]
            unsafe {
                self.runtime.close_open_cell(open.cell);
            }
        }
    }

    /// `a = b .. ... .. c` when every operand is a string or a number: the
    /// result is built once in the register and the collector check follows.
    /// Anything else, including a result too long for the stack buffer,
    /// declines before changing anything and takes the general path.
    #[inline(never)]
    fn concat_in_place(&mut self, registers: &mut [Value], instruction: Instruction) -> ConcatStep {
        let first = usize::from(instruction.b());
        let last = usize::from(instruction.c());
        let Some(operands) = registers.get(first..=last) else {
            return ConcatStep::Declined;
        };
        let Ok(value) = self.runtime.concatenate_many(operands) else {
            return ConcatStep::Declined;
        };
        set_register(&mut registers[usize::from(instruction.a())], value);
        if self.runtime.allocation_serial() != self.gc_observed_allocation_serial
            && self.runtime.gc_needs_collection()
        {
            return ConcatStep::Collect;
        }
        ConcatStep::Done
    }

    /// Runs a `NewTable` in the fast loop that creates a plain table.
    /// Out of line: its arm inlined cost every other instruction a spilled
    /// register. Declines in Lua 5.1 unless a recent visit found no scalar
    /// fragment at this site, because the fragment moves the frame's `pc`,
    /// which the loop holds in a machine register; [`Self::new_table_miss`]
    /// then runs it.
    #[inline(never)]
    fn new_table_in_place(
        &mut self,
        chunk: &crate::bytecode::Chunk,
        registers: &mut [Value],
        instruction: Instruction,
        at: *const Instruction,
    ) -> TableCreate {
        // The loop passes the instruction's address instead of `pc`: a `pc`
        // argument cost the other opcodes an extra instruction each.
        let pc = (at as usize - chunk.code.as_ptr() as usize) / size_of::<Instruction>();
        if self.compatibility == CompatibilityProfile::Lua51
            && !self
                .jit
                .scalar_table_plans
                .known_absent_at(std::ptr::from_ref(chunk) as usize, pc)
        {
            return TableCreate::Declined;
        }
        let Some(table) = create_constructor_table_fast(&mut self.runtime, chunk, pc) else {
            return TableCreate::Declined;
        };
        set_register(
            &mut registers[usize::from(instruction.a())],
            Value::Table(table),
        );
        if self.runtime.allocation_serial() != self.gc_observed_allocation_serial
            && self.runtime.gc_needs_collection()
        {
            return TableCreate::Collect;
        }
        TableCreate::Created
    }

    /// `+ - * / %` with a numeric string operand, in profiles that convert
    /// strings to numbers themselves (Lua 5.1, 5.2, LuaJIT): the result is
    /// stored and the pc stepped, as the general path would, without cloning
    /// the operands or consulting metamethods. Declines (`false`, nothing
    /// changed) every other operand pair, so errors and metamethods keep
    /// their general path.
    #[inline]
    fn string_arithmetic_in_place(
        &mut self,
        stack: &mut CallStack,
        top: usize,
        instruction: Instruction,
    ) -> bool {
        if self.compatibility.is_lua53_or_later() || self.jit.trace.recording() {
            return false;
        }
        let Ok(opcode) = instruction.opcode() else {
            return false;
        };
        let (register_opcode, constant) = match opcode {
            OpCode::Add | OpCode::Subtract | OpCode::Multiply | OpCode::Divide | OpCode::Modulo => {
                (opcode, None)
            }
            OpCode::AddK
            | OpCode::SubtractK
            | OpCode::MultiplyK
            | OpCode::DivideK
            | OpCode::ModuloK => {
                let Some(register_opcode) = opcode.register_form() else {
                    return false;
                };
                let frame = &stack[top];
                let constant = constant_value(
                    &frame.chunk.constants[usize::from(instruction.c())],
                    self.compatibility,
                );
                (register_opcode, Some(constant))
            }
            _ => return false,
        };
        let frame = stack.at_mut(top);
        let left = &frame.registers[usize::from(instruction.b())];
        let right = match &constant {
            Some(constant) => constant,
            None => &frame.registers[usize::from(instruction.c())],
        };
        if !matches!(left, Value::String(_)) && !matches!(right, Value::String(_)) {
            return false;
        }
        let nonfinite = nonfinite_numerals(self.compatibility);
        let operand = |value: &Value| match value {
            Value::Number(_) | Value::String(_) => {
                Runtime::coerce_number_in(value, "perform arithmetic", nonfinite).ok()
            }
            _ => None,
        };
        let (Some(left), Some(right)) = (operand(left), operand(right)) else {
            return false;
        };
        let result = match register_opcode {
            OpCode::Add => left + right,
            OpCode::Subtract => left - right,
            OpCode::Multiply => left * right,
            OpCode::Divide => left / right,
            _ => self.compatibility.float_arith().modulo(left, right),
        };
        frame.registers[usize::from(instruction.a())] = Value::Number(result);
        frame.frame.last_interpreted_pc = crate::vm::frame::LastPc::some(frame.frame.pc);
        frame.frame.pc += 1;
        true
    }

    /// [`Self::run_scalar_miss`] for the fast loop's in-place misses: a
    /// decline becomes [`FastExit::Declined`], so `interpret_fast` does not
    /// offer the same instruction again.
    #[inline(never)]
    fn resolve_plain_miss(&mut self, stack: &mut CallStack) -> VmResult<Option<FastExit>> {
        Ok(match self.run_scalar_miss(stack)? {
            Some(FastExit::Fallback) => Some(FastExit::Declined),
            exit => exit,
        })
    }

    /// [`Self::resolve_plain_miss`] for the arithmetic instructions,
    /// comparisons and tests, whose miss with a table operand enters the
    /// operator's handler first.
    #[inline(never)]
    fn resolve_scalar_miss(&mut self, stack: &mut CallStack) -> VmResult<Option<FastExit>> {
        if let Some((instruction, opcode)) = Self::table_operator_miss(stack)
            && let Some(step) = self.enter_table_operator(stack, instruction, opcode)
        {
            return Ok(step);
        }
        self.resolve_plain_miss(stack)
    }

    /// The instruction at the top frame's `pc` when it is an arithmetic
    /// instruction, a comparison or a test with a table operand, which only
    /// a metamethod answers. A few loads and compares in line, so every other
    /// miss (a table's length, a constructor) goes on to
    /// [`Self::run_scalar_miss`] at once.
    #[allow(unsafe_code, clippy::inline_always)]
    #[inline(always)]
    fn table_operator_miss(stack: &mut CallStack) -> Option<(Instruction, OpCode)> {
        // SAFETY: a frame runs, so the top record is live, and its window
        // lies in the value stack.
        let (frame, window) = unsafe {
            let (frame, values) = stack.top_raw();
            (&*frame, values.add((*frame).base as usize).cast_const())
        };
        let instruction = *frame.chunk.code.get(frame.pc)?;
        // One table load rules out the misses that are no operators, before
        // the opcode is decoded.
        if !OPERATOR_MISS[usize::from(instruction.raw().to_le_bytes()[0])] {
            return None;
        }
        let opcode = instruction.opcode().ok()?;
        // The registers that hold the operands (the chunk is validated).
        let (first, second) = match opcode {
            OpCode::Add
            | OpCode::Subtract
            | OpCode::Multiply
            | OpCode::Divide
            | OpCode::Modulo
            | OpCode::Equal
            | OpCode::LessThan
            | OpCode::LessEqual => (instruction.b(), instruction.c()),
            OpCode::AddK
            | OpCode::SubtractK
            | OpCode::MultiplyK
            | OpCode::DivideK
            | OpCode::ModuloK => (instruction.b(), instruction.b()),
            OpCode::TestEqual | OpCode::TestLessThan | OpCode::TestLessEqual => {
                (instruction.a(), instruction.b())
            }
            OpCode::TestEqualK | OpCode::TestLessThanK | OpCode::TestLessEqualK => {
                (instruction.a(), instruction.a())
            }
            _ => return None,
        };
        // SAFETY: registers of the running window.
        let table = unsafe {
            matches!(*window.add(usize::from(first)), Value::Table(_))
                || matches!(*window.add(usize::from(second)), Value::Table(_))
        };
        table.then_some((instruction, opcode))
    }

    /// The miss [`Self::table_operator_miss`] found: enters the operator's
    /// plain Lua handler before anything else is tried.
    /// [`Self::run_scalar_miss`] gets there too, after the cases it rules out
    /// first (a new table, string arithmetic, a length), which cost every
    /// operator handler call their tests. `None`, with nothing changed, for a
    /// handler that is not a plain closure. Out of line, so the misses that
    /// are no operators do not pay for its frame.
    #[allow(clippy::option_option)]
    #[inline(never)]
    fn enter_table_operator(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
        opcode: OpCode,
    ) -> Option<Option<FastExit>> {
        // The operator cache answers the usual operators of a class; the
        // general routine works the others out.
        if !(self.operator_miss_cached(stack, instruction, opcode)
            || self.operator_miss_hot(stack, instruction, opcode))
        {
            return None;
        }
        // The handler's activation is the top frame now. Entering it
        // allocated nothing, so no collector check follows; whether the
        // fast loop can run its first instruction is all that is left to
        // ask, as `run_scalar_miss` does.
        let top = stack.len() - 1;
        if !self.fast_path_eligible(&stack[top], false) {
            return Some(Some(FastExit::Declined));
        }
        Some(None)
    }

    /// Runs the `NewTable` at the top frame's `pc` without leaving fast
    /// mode, as the dispatch loop's `execute_plain` would: a verified
    /// scalar fragment first, otherwise a fresh table. The fast loop builds
    /// tables in place through [`Self::new_table_in_place`]; this runs the
    /// constructors it declines (scalar fragments, an exhausted handle
    /// space).
    #[inline(never)]
    fn new_table_miss(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<Option<FastExit>> {
        let top = stack.len() - 1;
        let mut frame = stack.at_mut(top);
        let destination = usize::from(instruction.a());
        if destination >= frame.registers.len() {
            return Ok(Some(FastExit::Fallback));
        }
        let pc = frame.frame.pc;
        frame.frame.last_interpreted_pc = crate::vm::frame::LastPc::some(pc);
        frame.frame.pc = pc + 1;
        // A site recently found without a plan skips the plan lookup.
        let planless = self.compatibility != CompatibilityProfile::Lua51
            || self.jit.trace.recording()
            || self
                .jit
                .scalar_table_plans
                .known_absent(&frame.frame.chunk.as_arc(), pc);
        if planless || !self.try_scalar_table_fragment(&mut frame) {
            if let Some(table) =
                create_constructor_table_fast(&mut self.runtime, &frame.frame.chunk, pc)
            {
                set_register(&mut frame.registers[destination], Value::Table(table));
            } else {
                // An exhausted handle space errors on the general path.
                frame.frame.pc = pc;
                return Ok(Some(FastExit::Fallback));
            }
        }
        if self.runtime.allocation_serial() != self.gc_observed_allocation_serial
            && self.runtime.gc_needs_collection()
        {
            return Ok(Some(FastExit::Safepoint));
        }
        Ok(None)
    }

    /// Enters the plain Lua `__index` or `__newindex` handler that the
    /// `GetField`, `GetTable`, `SetField` or `SetTable` at `pc` of the
    /// running frame (the miss left the instruction unexecuted) found on the
    /// receiver's own metatable, and returns the new top record and its
    /// window. The handler's return resumes the caller after the instruction.
    /// Declines before anything changes for every other case: a table-valued,
    /// native or vararg handler, a chain, a store whose probe did not fail.
    /// The general resolver, [`Self::resolve_index_miss`], takes those.
    ///
    /// # Safety
    ///
    /// `frame` is the record of the live top frame, whose `pc` is the
    /// instruction (the loop stored its state), `window` its first register
    /// and `depth` the live count.
    #[allow(unsafe_code)]
    #[inline(always)]
    unsafe fn index_miss_hot<const NEWINDEX: bool>(
        &mut self,
        stack: &mut CallStack,
        frame: *mut Frame,
        window: *mut Value,
        depth: usize,
        instruction: Instruction,
    ) -> Option<(*mut Frame, *mut Value)> {
        // SAFETY: the contract.
        let (chunk, pc) = unsafe { (&*(*frame).chunk, (*frame).pc) };
        // A register of the frame's window: the operands are in it (the chunk
        // is validated). Single values, not the window: the handler's window
        // lies inside the same 256 slots.
        let register = |index: usize| -> &Value {
            debug_assert!(index < REGISTER_WINDOW);
            // SAFETY: the contract, and `index` addresses the frame's window.
            unsafe { &*window.add(index) }
        };
        let a = usize::from(instruction.a());
        let opcode = opcode_byte(instruction);
        let (field, table) = if NEWINDEX {
            (op::SET_FIELD, op::SET_TABLE)
        } else {
            (op::GET_FIELD, op::GET_TABLE)
        };
        if opcode != field && opcode != table {
            return None;
        }
        let Value::Table(receiver) = *register(if NEWINDEX {
            a
        } else {
            usize::from(instruction.b())
        }) else {
            return None;
        };
        // The string store that just declined kept its handler, nothing has
        // run since, and the lookup would find the same closure.
        let hint = if NEWINDEX {
            self.runtime.take_newindex_hint()
        } else {
            None
        };
        let closure = if let Some(closure) = hint {
            closure
        } else {
            let metatable = self.runtime.raw_metatable(&Value::Table(receiver)).ok()??;
            let name = if NEWINDEX {
                newindex_metamethod_key()
            } else {
                index_metamethod_key()
            };
            let Some(&Value::Closure(closure)) = self.runtime.table_get_string_ref(metatable, name)
            else {
                return None;
            };
            closure
        };
        // The operands stay in the caller's registers and chunk; the handler's
        // window takes copies.
        let (target, key, value) = if NEWINDEX {
            let key = if opcode == field {
                match chunk.constants.get(usize::from(instruction.b())) {
                    Some(Constant::String(name)) => HandlerKey::Constant(name),
                    _ => return None,
                }
            } else {
                let key = register(usize::from(instruction.b()));
                match key {
                    Value::Number(number) if number.is_nan() => return None,
                    Value::String(_) | Value::Number(_) | Value::Integer(_) | Value::Boolean(_) => {
                        HandlerKey::Register(std::ptr::from_ref(key))
                    }
                    _ => return None,
                }
            };
            (
                ReturnTarget::SetMetamethod,
                key,
                std::ptr::from_ref(register(usize::from(instruction.c()))),
            )
        } else {
            let key = if opcode == field {
                match chunk.constants.get(usize::from(instruction.c())) {
                    Some(Constant::String(name)) => HandlerKey::Constant(name),
                    _ => return None,
                }
            } else {
                HandlerKey::Register(std::ptr::from_ref(register(usize::from(instruction.c()))))
            };
            (
                ReturnTarget::IndexResult(instruction.a()),
                key,
                std::ptr::null(),
            )
        };
        // SAFETY: the contract, and the pushed record roots `closure`. The
        // operands lie in the caller's window, below the new one.
        unsafe {
            stack.push_handler(
                &self.runtime,
                closure,
                target,
                &raw const *register(if NEWINDEX {
                    a
                } else {
                    usize::from(instruction.b())
                }),
                key,
                value,
                frame,
                window,
                depth,
                pc,
            )
        }
    }

    /// [`Self::index_miss_hot`] for the read misses of a `SelfField` (a
    /// method call) and a `GetTabUp` (the table is the upvalue's cell). A
    /// `SelfField` has copied its receiver to `a + 1` before it missed, or
    /// does so here (the same copy) when it missed on a stale handle.
    ///
    /// Inlined into its one caller, which is cold: the call and its register
    /// saves cost more than the code it would keep out of that function.
    ///
    /// # Safety
    ///
    /// As for [`Self::index_miss_hot`].
    #[allow(unsafe_code)]
    #[inline(always)]
    unsafe fn index_miss_receiver_hot(
        &mut self,
        stack: &mut CallStack,
        frame: *mut Frame,
        window: *mut Value,
        depth: usize,
        instruction: Instruction,
    ) -> Option<(*mut Frame, *mut Value)> {
        // SAFETY: the contract.
        let (chunk, pc, upvalues) = unsafe { (&*(*frame).chunk, (*frame).pc, &(*frame).upvalues) };
        let a = usize::from(instruction.a());
        let b = usize::from(instruction.b());
        let method = opcode_byte(instruction) == op::SELF_FIELD;
        debug_assert!(method || opcode_byte(instruction) == op::GET_TAB_UP);
        debug_assert!(b < REGISTER_WINDOW && a + 1 < REGISTER_WINDOW);
        let receiver_slot: *const Value = if method {
            // SAFETY: the operands are in the frame's window (the chunk is
            // validated).
            unsafe { window.add(b) }
        } else {
            std::ptr::from_ref(self.runtime.cell_ref(*upvalues.get(b)?).ok()?)
        };
        // SAFETY: a register of the window or a live cell value.
        let Value::Table(receiver) = *(unsafe { &*receiver_slot }) else {
            return None;
        };
        let Some(Constant::String(name)) = chunk.constants.get(usize::from(instruction.c())) else {
            return None;
        };
        // The inherit cache knows a metatable whose own `__index` is a
        // closure, which saves the metatable lookup. A miss of the cache
        // looks the closure up and records it for the next one.
        let closure = if let Some(closure) = self.runtime.table_inherited_handler(receiver, name) {
            closure
        } else {
            let metatable = self.runtime.raw_metatable(&Value::Table(receiver)).ok()??;
            let Some(&Value::Closure(closure)) = self
                .runtime
                .table_get_string_ref(metatable, index_metamethod_key())
            else {
                return None;
            };
            let _ = self.runtime.table_inherited_resolve(receiver, name);
            closure
        };
        if method {
            // SAFETY: `a + 1` is in the frame's window, and a table handle
            // needs no drop.
            set_register(unsafe { &mut *window.add(a + 1) }, Value::Table(receiver));
        }
        // SAFETY: the contract, and the pushed record roots `closure`. The
        // receiver lies in the caller's window or a cell, below the new one.
        unsafe {
            stack.push_handler(
                &self.runtime,
                closure,
                ReturnTarget::IndexResult(instruction.a()),
                receiver_slot,
                HandlerKey::Constant(name),
                std::ptr::null(),
                frame,
                window,
                depth,
                pc,
            )
        }
    }

    /// The miss of the `SetTable` `instruction` at the running frame's `pc`,
    /// whose state the loop stored. A string store that declined because the
    /// receiver's metatable has a plain `__newindex` closure kept it, and the
    /// closure is entered without decoding the instruction again: the store
    /// established that register `a` holds the receiving table and `b` a
    /// string key. Returns `true` when the fast loop continues in the
    /// handler. Any other miss is [`Self::index_miss_activation`]'s, with
    /// nothing changed here.
    #[inline(never)]
    fn set_table_enter(&mut self, stack: &mut CallStack, instruction: Instruction) -> bool {
        if let Some(closure) = self.runtime.take_newindex_hint() {
            let depth = stack.len();
            // SAFETY: a frame runs, so the top record is the one at `depth - 1`,
            // and its `pc` is the instruction (the loop stored its state).
            #[allow(unsafe_code)]
            let pushed = unsafe {
                let (frame, values) = stack.top_raw();
                let window = values.add((*frame).base as usize);
                let register = |index: u8| -> *const Value { window.add(usize::from(index)) };
                // The operands lie in the caller's window, below the
                // handler's, as for `index_miss_hot`. Every decline of the
                // hinted store is taken here or by the exit that follows it,
                // so the hint is the one of this store.
                debug_assert!(matches!(*register(instruction.a()), Value::Table(_)));
                debug_assert!(matches!(*register(instruction.b()), Value::String(_)));
                stack.push_handler(
                    &self.runtime,
                    closure,
                    ReturnTarget::SetMetamethod,
                    register(instruction.a()),
                    HandlerKey::Register(register(instruction.b())),
                    register(instruction.c()),
                    frame,
                    window,
                    depth,
                    (*frame).pc,
                )
            };
            if pushed.is_some() {
                #[cfg(test)]
                {
                    self.index_handler_entries += 1;
                }
                return true;
            }
        }
        false
    }

    /// The raw miss of the `GetField`, `GetTable`, `SetField` or `SetTable`
    /// at the running frame's `pc`, whose state the loop stored. Enters a
    /// plain handler on the receiver's own metatable through
    /// [`Self::index_miss_hot`], else [`Self::resolve_index_miss`] decides.
    /// `None` continues the fast loop in the new top frame. An instruction it
    /// could not run is [`FastExit::Declined`], so that `interpret_fast` does
    /// not try it again.
    ///
    /// `newindex_missed` is `false` for a read and for a store whose probe
    /// did not establish that the key holds no raw value.
    #[cold]
    #[inline(never)]
    fn index_miss_activation(
        &mut self,
        stack: &mut CallStack,
        newindex_missed: bool,
    ) -> VmResult<Option<FastExit>> {
        let depth = stack.len();
        // SAFETY: a frame runs, so the top record is the one at `depth - 1`.
        #[allow(unsafe_code)]
        let (frame, window) = unsafe {
            let (frame, values) = stack.top_raw();
            (frame, values.add((*frame).base as usize))
        };
        // SAFETY: as above.
        #[allow(unsafe_code)]
        let instruction = unsafe {
            let record = &*frame;
            record.chunk.code.get(record.pc).copied()
        };
        let Some(instruction) = instruction else {
            return Ok(Some(FastExit::Declined));
        };
        // SAFETY: `frame` is the live top record, whose `pc` is the
        // instruction, `window` its first register and `depth` the live
        // count.
        #[allow(unsafe_code)]
        let pushed = unsafe {
            if newindex_missed {
                self.index_miss_hot::<true>(stack, frame, window, depth, instruction)
            } else {
                self.index_miss_hot::<false>(stack, frame, window, depth, instruction)
            }
        };
        if pushed.is_some() {
            #[cfg(test)]
            {
                self.index_handler_entries += 1;
            }
            return Ok(None);
        }
        Ok(match self.resolve_index_miss(stack, newindex_missed)? {
            Some(FastExit::Fallback) => Some(FastExit::Declined),
            exit => exit,
        })
    }

    /// [`Self::index_miss_activation`] for the raw read miss of a `SelfField`
    /// or a `GetTabUp`, which the outer loop left with its state stored: a
    /// plain `__index` closure on the receiver is entered through
    /// [`Self::index_miss_receiver_hot`], else [`Self::resolve_index_miss`]
    /// decides. `None` continues the fast loop in the new top frame.
    #[cold]
    #[inline(never)]
    fn index_receiver_activation(&mut self, stack: &mut CallStack) -> VmResult<Option<FastExit>> {
        let depth = stack.len();
        // SAFETY: a frame runs, so the top record is the one at `depth - 1`.
        #[allow(unsafe_code)]
        let (frame, window, instruction) = unsafe {
            let (frame, values) = stack.top_raw();
            let record = &*frame;
            (
                frame,
                values.add(record.base as usize),
                record.chunk.code.get(record.pc).copied(),
            )
        };
        let Some(instruction) = instruction else {
            return Ok(Some(FastExit::Declined));
        };
        // SAFETY: `frame` is the live top record, whose `pc` is the
        // instruction, `window` its first register and `depth` the live
        // count.
        #[allow(unsafe_code)]
        let pushed =
            unsafe { self.index_miss_receiver_hot(stack, frame, window, depth, instruction) };
        if pushed.is_some() {
            #[cfg(test)]
            {
                self.index_handler_entries += 1;
            }
            return Ok(None);
        }
        Ok(match self.resolve_index_miss(stack, false)? {
            Some(FastExit::Fallback) => Some(FastExit::Declined),
            exit => exit,
        })
    }

    /// Completes the `GetTable` or `GetField` at the top frame's `pc` after
    /// a raw miss by following table-valued `__index` chains, or enters the plain Lua
    /// `__index` function the chain ends in, as the dispatch loop's
    /// `run_index` would. `None` continues the fast loop; an exit hands the
    /// dispatch loop whatever the fast loop cannot run, having changed
    /// nothing unless the instruction completed.
    #[inline(never)]
    fn resolve_index_miss(
        &mut self,
        stack: &mut CallStack,
        newindex_missed: bool,
    ) -> VmResult<Option<FastExit>> {
        #[cfg(test)]
        {
            self.index_miss_resolves += 1;
        }
        let top = stack.len() - 1;
        let (frames, values) = stack.frames_and_values();
        let frame = &mut frames[top];
        let Some(&instruction) = frame.chunk.code.get(frame.pc) else {
            return Ok(Some(FastExit::Fallback));
        };
        let Some(registers) = values.get_mut(frame.base as usize..frame.top as usize) else {
            return Ok(Some(FastExit::Fallback));
        };
        if matches!(
            instruction.opcode(),
            Ok(OpCode::SetField | OpCode::SetTable)
        ) {
            return self.resolve_newindex_miss(stack, instruction, newindex_missed);
        }
        let destination = usize::from(instruction.a());
        let opcode = instruction.opcode();
        // A `SelfField` copies its receiver first, as the fast loop did
        // before it left (the same copy), then reads like a `GetField`.
        if matches!(opcode, Ok(OpCode::SelfField)) {
            let receiver = usize::from(instruction.b());
            if destination + 1 >= registers.len() || receiver >= registers.len() {
                return Ok(Some(FastExit::Fallback));
            }
            let value = registers[receiver].clone();
            set_register(&mut registers[destination + 1], value);
        }
        // A `GetField` names its key by a string constant. The key is owned
        // once: the walk reads it and the handler takes it as its argument.
        let is_field = matches!(
            opcode,
            Ok(OpCode::GetField | OpCode::SelfField | OpCode::GetTabUp)
        );
        let key = if is_field {
            match frame.chunk.constants.get(usize::from(instruction.c())) {
                Some(Constant::String(name)) => Some(Value::String(name.clone())),
                _ => return Ok(Some(FastExit::Fallback)),
            }
        } else {
            registers
                .get(usize::from(instruction.c()))
                .map(clone_register_value)
        };
        // A `GetTabUp` reads its table from an upvalue cell.
        let receiver = if matches!(opcode, Ok(OpCode::GetTabUp)) {
            frame
                .upvalues
                .get(usize::from(instruction.b()))
                .and_then(|cell| self.runtime.cell_ref(*cell).ok())
        } else {
            registers.get(usize::from(instruction.b()))
        };
        let (Some(&Value::Table(table)), Some(key), true) =
            (receiver, key, destination < registers.len())
        else {
            return Ok(Some(FastExit::Fallback));
        };
        // A `GetField` leaves the fast loop only after its raw read of the
        // table missed, so the walk need not repeat that read. A name found
        // in a table-valued `__index` chain, or nowhere, is also recorded for
        // the fast loop's next read of it.
        if is_field
            && let Value::String(name) = &key
            && let Some(value) = self.runtime.table_inherited_resolve(table, name)
        {
            store_clone(&mut registers[destination], value);
            frame.last_interpreted_pc = crate::vm::frame::LastPc::some(frame.pc);
            frame.pc += 1;
            return Ok(None);
        }
        let walk = if is_field {
            self.runtime.table_index_walk_after_miss(table, &key)
        } else {
            self.runtime.table_index_walk(table, &key)
        };
        let (receiver, closure) = match walk {
            Some(Ok(value)) => {
                store_clone(&mut registers[destination], value);
                frame.last_interpreted_pc = crate::vm::frame::LastPc::some(frame.pc);
                frame.pc += 1;
                return Ok(None);
            }
            Some(Err((receiver, &Value::Closure(closure)))) => (receiver, closure),
            // A runtime native handler runs here; embedders' host functions
            // need the VM's roots and take the dispatch loop.
            Some(Err((receiver, &Value::NativeFunction(function))))
                if self.native_runs_in_place(function) =>
            {
                frame.last_interpreted_pc = crate::vm::frame::LastPc::some(frame.pc);
                frame.pc += 1;
                let value =
                    self.call_runtime_native_first(function, &[Value::Table(receiver), key])?;
                let frame = stack.at_mut(top);
                set_register(&mut frame.registers[destination], value);
                if self.runtime.allocation_serial() != self.gc_observed_allocation_serial
                    && self.runtime.gc_needs_collection()
                {
                    return Ok(Some(FastExit::Safepoint));
                }
                return Ok(None);
            }
            // Anything unusual takes the dispatch loop.
            _ => return Ok(Some(FastExit::Fallback)),
        };
        // The handler's return writes its first result to the destination:
        // plain data in the frame, no continuation.
        if !self.enter_index_handler(
            stack,
            closure,
            ActivationArguments::Pair(Value::Table(receiver), key),
            ReturnTarget::IndexResult(instruction.a()),
        )? {
            return Ok(Some(FastExit::Fallback));
        }
        let top = stack.len() - 1;
        if !self.fast_path_eligible(&stack[top], false) {
            return Ok(Some(FastExit::Fallback));
        }
        Ok(None)
    }

    /// Enters the plain Lua `__index` or `__newindex` handler `closure` for
    /// the instruction at the top frame's `pc`, which resumes after it, as
    /// [`Self::push_metamethod_activation`] does with one lookup of the
    /// closure's parts. `false` leaves everything unchanged: a vararg handler
    /// takes the dispatch loop.
    #[inline(always)]
    fn enter_index_handler(
        &mut self,
        stack: &mut CallStack,
        closure: ClosureHandle,
        arguments: ActivationArguments,
        target: ReturnTarget,
    ) -> VmResult<bool> {
        let Some(parts) = self.runtime.closure_call_parts(closure) else {
            return Ok(false);
        };
        if parts.chunk.is_vararg {
            return Ok(false);
        }
        let mut caller = stack.at_mut(stack.len() - 1);
        caller.last_interpreted_pc = crate::vm::frame::LastPc::some(caller.pc);
        caller.pc += 1;
        ensure_call_depth(stack)?;
        // SAFETY: `parts` are `closure`'s own, and the frame roots `closure`.
        #[allow(unsafe_code)]
        unsafe {
            stack.push_activation_to(
                parts.chunk,
                parts.cells,
                parts.environment,
                closure,
                target,
                arguments,
            );
        }
        Ok(true)
    }

    /// Enters a plain Lua `__newindex` closure for the `SetField` or
    /// `SetTable` at the top frame's `pc`, after the store missed. The
    /// continuation discards the result. Every other case (a raw value
    /// present, a table-valued or native handler, a non-table receiver, a
    /// nil or NaN key) takes the dispatch loop, which repeats the
    /// instruction and does the work or raises the error.
    #[inline(never)]
    fn resolve_newindex_miss(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
        newindex_missed: bool,
    ) -> VmResult<Option<FastExit>> {
        // The handler the declined store found, if it was the string store.
        let hint = if newindex_missed {
            self.runtime.take_newindex_hint()
        } else {
            None
        };
        let top = stack.len() - 1;
        let (frames, values) = stack.frames_and_values();
        let frame = &mut frames[top];
        let Some(registers) = values.get_mut(frame.base as usize..frame.top as usize) else {
            return Ok(Some(FastExit::Fallback));
        };
        let Some(&Value::Table(table)) = registers.get(usize::from(instruction.a())) else {
            return Ok(Some(FastExit::Fallback));
        };
        let key = if instruction.opcode() == Ok(OpCode::SetField) {
            match frame.chunk.constants.get(usize::from(instruction.b())) {
                Some(Constant::String(name)) => Value::String(name.clone()),
                _ => return Ok(Some(FastExit::Fallback)),
            }
        } else {
            match registers.get(usize::from(instruction.b())) {
                Some(Value::Number(number)) if number.is_nan() => {
                    return Ok(Some(FastExit::Fallback));
                }
                Some(
                    key @ (Value::String(_)
                    | Value::Number(_)
                    | Value::Integer(_)
                    | Value::Boolean(_)),
                ) => clone_register_value(key),
                _ => return Ok(Some(FastExit::Fallback)),
            }
        };
        let Some(source) = registers.get(usize::from(instruction.c())) else {
            return Ok(Some(FastExit::Fallback));
        };
        let value = clone_register_value(source);
        // A raw value is written, not handed to `__newindex`. A
        // `NewIndexMiss` exit already established that the key holds none.
        if !newindex_missed {
            match self.runtime.table_get(table, &key) {
                Ok(Value::Nil) => {}
                _ => return Ok(Some(FastExit::Fallback)),
            }
        }
        // `unary_metamethod` for a table receiver, with the prebuilt key and
        // a borrowed read: no name match, no clone of the handler value. The
        // store that just declined may have read it already; nothing ran
        // since.
        let closure = match hint {
            Some(closure) if self.is_plain_closure(closure) => closure,
            _ => match self.runtime.raw_metatable(&Value::Table(table)) {
                Ok(Some(metatable)) => {
                    match self
                        .runtime
                        .table_get_string_ref(metatable, newindex_metamethod_key())
                    {
                        Some(&Value::Closure(closure)) if self.is_plain_closure(closure) => closure,
                        _ => return Ok(Some(FastExit::Fallback)),
                    }
                }
                _ => return Ok(Some(FastExit::Fallback)),
            },
        };
        // The handler's results are discarded, so its return needs no
        // continuation and nothing past the call reads the operands.
        if !self.enter_index_handler(
            stack,
            closure,
            ActivationArguments::Triple(Value::Table(table), key, value),
            ReturnTarget::SetMetamethod,
        )? {
            return Ok(Some(FastExit::Fallback));
        }
        let top = stack.len() - 1;
        if !self.fast_path_eligible(&stack[top], false) {
            return Ok(Some(FastExit::Fallback));
        }
        Ok(None)
    }

    // Kept out of line: inlined into `interpret_fast`, its register
    // allocation shifted with every change to the exit handling there.
    #[allow(clippy::too_many_lines)]
    // Lua number equality is exact, constants included.
    #[allow(clippy::float_cmp)]
    // The packed jump record holds 32-bit program counters; validation keeps
    // every chunk's code within them.
    #[allow(clippy::cast_possible_truncation)]
    #[inline(never)]
    fn run_fast_activations<
        const RECORD: bool,
        const HOOKED: bool,
        const COUNT_ONLY: bool,
        const DIRECT: bool,
    >(
        &mut self,
        stack: &mut CallStack,
        tracing: bool,
    ) -> VmResult<FastExit> {
        // `tracing` is tracing mode with native code enabled, which offers
        // backedges and calls to native code. Recording implies it. It is a
        // run-time flag, not a second instantiation: the interpreter and the
        // tracing mode then run the same machine code, so neither can lose to
        // the other on where the compiler happened to lay out the hot blocks.
        debug_assert!(!RECORD || tracing);
        // Plain closure calls push their activation without the general call
        // routine. Tracing mode has its own variant, which still offers calls
        // to native code and leaves the interpreter's untouched. Tracing mode
        // with native code disabled uses neither.
        //
        // `DIRECT` is that, fixed at compile time for the interpreter's own
        // instantiation, which then carries no test for it.
        //
        // Tracing mode without a recording takes the same plain calls and
        // returns at run time, with the cold landing offer for self calls
        // after each push (`DIRECT` is false there, so the interpreter's
        // instantiation carries none of it). A recording or a hooked run
        // keeps the general routines, which observe every call.
        let direct_calls = DIRECT || (tracing && !RECORD && !HOOKED);
        // The interpreter under a count hook that observes no call or return
        // takes the same plain calls and returns: the hook counts
        // instructions, which the loop does in whichever activation runs,
        // and a push or a pop changes nothing it reads. What the plain forms
        // decline goes on to the general routines below, as every hooked
        // call and return did before.
        let counted_calls = HOOKED
            && COUNT_ONLY
            && !RECORD
            && !tracing
            && self.execution_mode == ExecutionMode::Interpreter;
        // Tail calls, vararg calls and generic-for iterator calls keep
        // the interpreter-only gate.
        let direct_other = DIRECT;
        debug_assert!(
            !DIRECT || (!RECORD && !HOOKED && !tracing),
            "direct calls are the plain interpreter's"
        );
        debug_assert_eq!(
            DIRECT,
            !HOOKED && !RECORD && !tracing && self.execution_mode == ExecutionMode::Interpreter
        );
        let compatibility = self.compatibility;
        let float_arith = compatibility.float_arith();
        // Chosen once per run, not per operation: the hot `%` arm below
        // tests this bool and inlines the unfused floor formula for Lua 5.1,
        // 5.2 and LuaJIT, without matching on the profile's formula enum. The
        // quotient is floored in one instruction, so the bool is also false
        // on a CPU without it. That CPU and the `fmod` forms go out of line.
        let floor_modulo =
            float_arith.modulo == FloatModulo::Floor && FloatModulo::has_hardware_floor();
        let tagged_integers = compatibility.is_lua53_or_later();
        // The tag an integer operand of the arithmetic arms has: no value's
        // on a profile without integers, so one compare of the operand's
        // tag byte answers both "is it an integer" and "does the profile
        // have them".
        let integer_tag = if tagged_integers {
            crate::runtime::NATIVE_VALUE_TAG_INTEGER
        } else {
            NO_VALUE_TAG
        };
        let tagged_env = compatibility != CompatibilityProfile::Lua51
            && compatibility != CompatibilityProfile::LuaJit21;
        // What a hooked run observes is fixed for the run.
        // `COUNT_ONLY` is a hooked run that counts and observes no call or
        // return: both facts are then compile-time, which keeps the per-
        // instruction hook check to the budget alone.
        debug_assert!(!COUNT_ONLY || (HOOKED && self.fast_hook_count && !self.fast_hook_calls));
        let hook_count = HOOKED && (COUNT_ONLY || self.fast_hook_count);
        let hook_calls = HOOKED && !COUNT_ONLY && self.fast_hook_calls;
        // Fetching skips the bounds check, so the first activation's entry pc
        // is checked once here, and so is the chunk's validation. Unvalidated
        // chunks run on the general loop.
        {
            let depth = stack.len();
            let (frames, _) = stack.frames_and_values();
            let top = &frames[depth - 1];
            if top.pc >= top.chunk.code.len() {
                return Ok(FastExit::Fallback);
            }
        }
        // A recording that sleeps through a self call needs no hook below the
        // callee's frame: the recursion beneath runs without one.
        let mut record_depth = if RECORD {
            self.jit.trace.record_depth_limit()
        } else {
            usize::MAX
        };
        'activations: loop {
            // A dispatch pair never spans an exit or an activation switch.
            #[cfg(feature = "op-stats")]
            crate::vm::op_stats::fast_break_pairs();
            if HOOKED {
                self.count_fetched = crate::vm::NOT_FETCHED;
            }
            // The interpreter's plain calls look at the flag themselves (see
            // the `Call` arm), so a switch between its activations pays
            // nothing. Every other run checks at each switch.
            if (HOOKED || !direct_calls) && self.interrupt_pending() {
                return Ok(FastExit::Fallback);
            }
            let switch = {
                // SAFETY: the loop runs on a live top frame.
                #[allow(unsafe_code)]
                let (first_frame, mut window) = unsafe { stack.running_view() };
                // Only a recording asks (see `HookPoint::top_level`).
                let top_level = RECORD && {
                    let depth = stack.len();
                    let (frames, _) = stack.frames_and_values();
                    depth <= record_depth
                        && frames[..depth].iter().all(|frame| frame.closure.is_none())
                };
                // Register operands index a fixed-size window: no bounds checks.
                //
                // SAFETY (the next three bindings): they borrow the value
                // stack and the chunk of the running closure, whose frame
                // roots it. Nothing uses them after an operation that can
                // move the stack or pop the frame: those either switch
                // activations and bind them again, or leave the loop.
                // A tail call that keeps the window (see
                // `CallStack::tail_call_closure`) goes round this loop again,
                // which binds what follows from the same record instead of
                // going through the activation prologue.
                //
                // Its body keeps the indentation it had before the loop was
                // added (`rustfmt::skip`): moving the whole dispatch loop
                // one level in would bury its history and conflict with every
                // edit of its arms.
                // What the next pass of `'rebind` starts from: the entry pc and
                // the chunk. A call or a return that knows them from what it
                // just did hands them over; every other pass reads them from
                // the record (`rebind_from_record!`). Reading them back after
                // a call loaded what the call had just stored.
                //
                // SAFETY: `frame` is the live top record.
                #[allow(unsafe_code)]
                let (mut entry_pc, mut entry_chunk) =
                    unsafe { ((*first_frame).pc, (*first_frame).chunk.address()) };
                // The running record lives in memory, not in a register:
                // few arms read it, and the loop has more state than
                // registers. Left to the compiler it sat in a stack slot
                // all the same, and so did the addresses of its fields,
                // which were computed and stored again at every activation
                // switch. Volatile accesses keep it to the one slot; an arm
                // that reads the record more than once takes it into a
                // local first.
                let mut frame_home: *mut Frame = first_frame;
                macro_rules! frame {
                    () => {
                        slot_get(&frame_home)
                    };
                }
                macro_rules! set_frame {
                    ($record:expr) => {{
                        let record: *mut Frame = $record;
                        slot_set(&mut frame_home, record);
                    }};
                }
                // Binds the next pass to `frame` as the record holds it.
                macro_rules! rebind_from_record {
                    () => {{
                        // SAFETY: `frame` is the live top record.
                        #[allow(unsafe_code)]
                        unsafe {
                            let record = frame!();
                            entry_pc = (*record).pc;
                            entry_chunk = (*record).chunk.address();
                        }
                    }};
                }
                #[rustfmt::skip]
                let outcome = 'rebind: loop {
                // SAFETY: as above, for the window of the record just bound.
                #[allow(unsafe_code)]
                let registers: &mut [Value; REGISTER_WINDOW] = unsafe { &mut *window };
                // Dereference the tagged chunk word once per activation, not
                // at every constant fetch.
                #[allow(unsafe_code)]
                let chunk: &crate::bytecode::Chunk = unsafe { &*entry_chunk };
                // SAFETY: the home holds the live top record, which a tail
                // call may have rewritten since the last pass.
                #[allow(unsafe_code)]
                #[cfg(debug_assertions)]
                unsafe {
                    let current = &*frame!();
                    debug_assert!(std::ptr::eq(entry_chunk, current.chunk.address()));
                    debug_assert_eq!(entry_pc, current.pc);
                }
                // Only the first instruction's address is loop state: the
                // few arms that need the length read it from the chunk. Kept
                // as a slice, the length was loaded and spilled at every call
                // and return.
                let code: *const Instruction = chunk.code.as_ptr();
                // The chunk's read sites, empty placeholders until a fill
                // allocates them (a fill moves the chunk to its own): a hit
                // indexes them by pc. Read from the chunk at each use, like
                // `environment!()`: bound at every rebind, the pointer cost
                // each call and return a load, and kept live it pushed the
                // tracing loop's code pointer onto the stack.
                macro_rules! read_sites {
                    () => {
                        chunk.read_sites_first()
                    };
                }
                let mut pc = entry_pc;
                // The last executed pc is derived at exit instead of being
                // recorded by every instruction. Only a taken jump breaks
                // sequential execution, and it records where it jumped from
                // and to. Since that landing, execution ran straight from
                // `landing` to `pc`, so `pc == landing` exactly when the jump
                // was the last instruction executed; otherwise it was
                // `pc - 1`. Entry counts as a landing with no jump, meaning
                // nothing has executed yet.
                // Both are packed into one word to keep a register free: the
                // jump's pc plus one in the high half, zero for no jump, and
                // its landing in the low half.
                let mut jump = pc as u64;
                // Fetching below skips the bounds check: see `fetch` for why
                // `pc` is in range on every activation after the first.
                debug_assert!(pc < chunk.code.len());
                // Store the loop state back on every exit path.
                macro_rules! store_state {
                    () => {{
                        // SAFETY: the home holds the live top record.
                        #[allow(unsafe_code)]
                        let current = unsafe { &mut *frame!() };
                        store_loop_state(current, pc, jump);
                    }};
                }
                // The running frame's record.
                macro_rules! running {
                    () => {{
                        // SAFETY: `frame` is the live top record.
                        #[allow(unsafe_code)]
                        let running: &Frame = unsafe { &*frame!() };
                        running
                    }};
                }
                // The running closure's environment and captured cells, read
                // from its record where they are used. Hoisted at every
                // rebind, they cost a load, a spill and a reload on each call
                // and return, and few instructions use them.
                macro_rules! environment {
                    () => {{ running!().environment }};
                }
                macro_rules! upvalues {
                    () => {{ &running!().upvalues }};
                }
                // Gives back the budget the loop took for an instruction that
                // does not count. The three free opcodes call it first.
                macro_rules! refund_free {
                    () => {
                        if hook_count {
                            self.count_budget += 1;
                        }
                    };
                }
                // Under a count hook, records that the instruction at `pc` was
                // fetched and counted, for an exit that leaves `pc` on it
                // (see `Vm::interpret_fast_counted`). An exit taken after
                // `pc` moved on leaves the activation's `NOT_FETCHED`. Only
                // exits record this: the loop itself stores nothing per
                // instruction.
                macro_rules! mark_fetched {
                    () => {
                        if HOOKED {
                            self.count_fetched = crate::vm::fetched_mark(stack.len(), pc);
                        }
                    };
                }
                macro_rules! exit {
                    ($exit:expr) => {{
                        mark_fetched!();
                        store_state!();
                        return Ok($exit);
                    }};
                }
                // Moves to `target` after the taken jump at `pc`.
                macro_rules! jump {
                    ($target:expr) => {{
                        let target = $target;
                        if target <= pc && self.interrupt_pending() {
                            exit!(FastExit::Fallback);
                        }
                        jump = ((pc as u64 + 1) << 32) | target as u64;
                        pc = target;
                    }};
                }
                // The jump of a loop instruction that already changed the
                // loop's registers (`ForInLoop`, `NumericForLoop`). A pending
                // interrupt, or a tick of the attribution sampler (see
                // `vm::attribution`), is served after the jump, with the
                // instruction done: leaving before it would run it again,
                // and step the loop twice.
                macro_rules! jump_after {
                    ($target:expr) => {{
                        let target = $target;
                        let backward = target <= pc;
                        jump = ((pc as u64 + 1) << 32) | target as u64;
                        pc = target;
                        if backward && self.interrupt_pending() {
                            exit!(FastExit::Fallback);
                        }
                    }};
                }
                align_dispatch();
                let outcome = loop {
                    // `pc < code.len()` holds on entry (checked above) and after
                    // every step; see `fetch`.
                    let instruction = fetch_at(code, pc);
                    // Counts as `fire_count_hook` does: the fixed-frame
                    // compiler's register maintenance is free. The
                    // instruction that would fire the hook goes to the
                    // dispatch loop uncounted. Calls and returns go there
                    // counted, to fire their own events.
                    if HOOKED {
                        let opcode = opcode_byte(instruction);
                        if hook_count {
                            // Every instruction takes one, and the arms of the
                            // free opcodes give it back (`refund_free!`), so
                            // the common case looks at the budget alone. The
                            // instruction that finds the last one goes to the
                            // dispatch loop if it counts, and so does any
                            // instruction when a hook change zeroed the
                            // budget.
                            let budget = self.count_budget;
                            if budget <= 1 && (budget == 0 || COUNTED_OPCODES[opcode as usize] != 0)
                            {
                                exit!(FastExit::CountHook);
                            }
                            self.count_budget = budget - 1;
                        }
                        if hook_calls
                            && matches!(
                                opcode,
                                op::CALL
                                    | op::FOR_IN_CALL
                                    | op::CALL_ONE
                                    | op::TAIL_CALL
                                    | op::RETURN
                                    | op::HALT
                                    | op::CALL_OPEN
                                    | op::RETURN_OPEN
                            )
                        {
                            exit!(FastExit::Fallback);
                        }
                    }
                    // Runs the instruction at `pc`, a scalar operator the loop cannot
                    // finish, through `run_scalar_miss` and re-enters the loop,
                    // without leaving `run_fast_activations`.
                    // The arithmetic instructions, comparisons and tests ask
                    // first for an operator handler to enter
                    // (`resolve_scalar_miss`); the other instructions name
                    // `resolve_plain_miss`, which does not.
                    macro_rules! miss {
                        () => {
                            miss!(resolve_scalar_miss)
                        };
                        ($resolve:ident) => {{
                            #[cfg(feature = "op-stats")]
                            crate::vm::op_stats::bump(crate::vm::op_stats::Counter::FastScalarMisses);
                            if hook_calls {
                                exit!(FastExit::Fallback);
                            }
                            mark_fetched!();
                            store_state!();
                            match self.$resolve(stack)? {
                                None => continue 'activations,
                                Some(exit) => return Ok(exit),
                            }
                        }};
                    }

                    // The trace recorder sees the instruction and the live
                    // registers before it runs. When the recording ends the
                    // instruction goes to the general loop unexecuted.
                    if RECORD && stack.len() <= record_depth {
                        if !self.record_fast_hook(
                            &running!().chunk.as_arc(),
                            (stack.len(), top_level),
                            pc,
                            &registers[..],
                            upvalues!(),
                            environment!(),
                            // SAFETY: a frame's varargs lie just below its
                            // window, in the value stack.
                            #[allow(unsafe_code)]
                            (
                                {
                                    let held = running!().vararg_count();
                                    unsafe {
                                        std::slice::from_raw_parts(
                                            window.cast::<Value>().cast_const().sub(held),
                                            held,
                                        )
                                    }
                                },
                                running!().open_values().unwrap_or_default(),
                            ),
                        ) {
                            if hook_count && COUNTED_OPCODES[opcode_byte(instruction) as usize] == 0
                            {
                                refund_free!();
                            }
                            exit!(FastExit::Fallback);
                        }
                        record_depth = self.jit.trace.record_depth_limit();
                    }
                    // Read from the code, not shifted out of the instruction
                    // word. The shift made the compiler keep the word in the
                    // one callee-saved register that has a high byte, and
                    // each call and return arm that read `a` after it took
                    // that register away: every dispatch then cost a shift
                    // and a move more (+4% instructions on arithmetic loops).
                    //
                    // SAFETY: `pc < code.len()` (see `fetch`), and an
                    // instruction is four bytes with `a` in the second.
                    #[cfg(target_endian = "little")]
                    #[allow(unsafe_code)]
                    // A volatile read: the compiler otherwise takes the
                    // byte out of the word it just loaded, with a shift and
                    // a mask in every dispatch where this is one load.
                    let a = usize::from(unsafe {
                        std::ptr::read_volatile(code.cast::<u8>().add(pc * 4 + 1))
                    });
                    #[cfg(not(target_endian = "little"))]
                    let a = usize::from(instruction.a());
                    // One arm per arithmetic opcode, so the operator is a
                    // constant in the arm and no second dispatch follows the
                    // first. Floats are tested first; the rest is out of line.
                    macro_rules! arithmetic_arm {
                        ($opcode:expr, $float:expr, $integer:expr) => {{
                            let (left, right) = (
                                &registers[usize::from(instruction.b())],
                                &registers[usize::from(instruction.c())],
                            );
                            #[cfg(feature = "op-stats")]
                            crate::vm::op_stats::arithmetic($opcode as u8, chunk, pc, left, right);
                            // Build each result variant at its store; routing a
                            // generic `Value` through here stages it on the stack.
                            if let (Value::Number(left), Value::Number(right)) = (left, right) {
                                let result: f64 = ($float)(*left, *right);
                                set_register(&mut registers[a], Value::Number(result));
                            } else if value_tag(left) == integer_tag
                                && let Value::Integer(right) = right
                                && let Some(result) = ($integer)(integer_operand(left), *right)
                            {
                                set_register(&mut registers[a], Value::Integer(result));
                            } else if let Some((left, right)) = mixed_floats(left, right) {
                                // One integer and one float: the integer converts
                                // and the float operator applies, as the general
                                // path does, with no cold call.
                                let result: f64 = ($float)(left, right);
                                set_register(&mut registers[a], Value::Number(result));
                            } else {
                                #[cfg(feature = "op-stats")]
                                crate::vm::op_stats::bump(crate::vm::op_stats::Counter::FastArithmeticSlow);
                                let Some(value) =
                                    arithmetic_slow::<{ $opcode as u8 }>(left, right, float_arith)
                                else {
                                    miss!();
                                };
                                set_register(&mut registers[a], value);
                            }
                        }};
                    }
                    // The constant-operand arithmetic arms, one per opcode. Each arm
                    // reaches this code from the first dispatch with the opcode
                    // already known, so the `match` on it below folds away and no
                    // second jump table follows.
                    macro_rules! arithmetic_k_arm {
                        ($opcode:expr) => {{
                            let opcode: u8 = $opcode;
                            let left = &registers[usize::from(instruction.b())];
                            let constant = constant_at(chunk, instruction.c().into());
                            #[cfg(feature = "op-stats")]
                            crate::vm::op_stats::arithmetic(
                                opcode,
                                chunk,
                                pc,
                                left,
                                &constant_value(constant, compatibility),
                            );
                            if let (Value::Number(left), Constant::Number(right)) = (left, constant)
                            {
                                let result =
                                    float_k_op(opcode, floor_modulo, float_arith, *left, *right);
                                set_register(&mut registers[a], Value::Number(result));
                            } else if value_tag(left) == integer_tag
                                && let Constant::Integer(right) = constant
                                && let Some(result) = match opcode {
                                    op::ADD_K => Some(integer_operand(left).wrapping_add(*right)),
                                    op::SUBTRACT_K => {
                                        Some(integer_operand(left).wrapping_sub(*right))
                                    }
                                    op::MULTIPLY_K => {
                                        Some(integer_operand(left).wrapping_mul(*right))
                                    }
                                    op::DIVIDE_K => None,
                                    _ => integer_floor_mod_nonzero(integer_operand(left), *right),
                                }
                            {
                                set_register(&mut registers[a], Value::Integer(result));
                            } else if let Some((l, r)) = mixed_k_floats(left, constant) {
                                let result = float_k_op(
                                    opcode,
                                    floor_modulo,
                                    float_arith,
                                    l,
                                    r,
                                );
                                set_register(&mut registers[a], Value::Number(result));
                            } else {
                                #[cfg(feature = "op-stats")]
                                crate::vm::op_stats::bump(crate::vm::op_stats::Counter::FastArithmeticSlow);
                                let Some(register_opcode) = OpCode::try_from(opcode)
                                    .ok()
                                    .and_then(OpCode::register_form)
                                else {
                                    miss!();
                                };
                                let Some(value) = arithmetic_k_other(
                                    register_opcode,
                                    left,
                                    constant,
                                    compatibility,
                                    float_arith,
                                ) else {
                                    miss!();
                                };
                                set_register(&mut registers[a], value);
                            }
                        }};
                    }
                    // Comparisons, one arm per operator like `arithmetic_arm`.
                    macro_rules! comparison_arm {
                        ($compare:expr, $order:expr) => {{
                            let (left, right) = (
                                &registers[usize::from(instruction.b())],
                                &registers[usize::from(instruction.c())],
                            );
                            #[cfg(feature = "op-stats")]
                            crate::vm::op_stats::arithmetic(
                                opcode_byte(instruction),
                                chunk,
                                pc,
                                left,
                                right,
                            );
                            let result = match (left, right) {
                                (Value::Number(left), Value::Number(right)) => {
                                    ($compare)(*left, *right)
                                }
                                (Value::Integer(left), Value::Integer(right))
                                    if tagged_integers =>
                                {
                                    ($compare)(*left, *right)
                                }
                                // An integer and a float order exactly, never
                                // through a lossy cast.
                                (Value::Integer(_), Value::Number(_))
                                | (Value::Number(_), Value::Integer(_)) => {
                                    ($order)(numeric_order(left, right))
                                }
                                _ => {
                                    let Some(Value::Boolean(result)) =
                                        scalar_slow(instruction, left, right, float_arith)
                                    else {
                                        miss!();
                                    };
                                    result
                                }
                            };
                            set_register(&mut registers[a], Value::Boolean(result));
                        }};
                    }
                    let next = pc + 1;
                    #[cfg(feature = "op-stats")]
                    crate::vm::op_stats::fast_dispatch(opcode_byte(instruction));
                    // Runs the `Jump` word after a test when `result` equals the
                    // test's `expected` flag, and steps over it otherwise. The word
                    // is executed here, with no dispatch of its own, and the pc it
                    // records as last executed is the word's, as if the loop had run
                    // it. `pc` moves onto the word first, so a backedge or interrupt
                    // exit hands the dispatch loop a test already done.
                    macro_rules! test_branch {
                        ($result:expr) => {{
                            let result: bool = $result;
                            if result != (instruction.c() & 1 != 0) {
                                pc = next + 1;
                                continue;
                            }
                            let word = fetch_at(code, next);
                            pc = next;
                            let target = relative(pc + 1, word.jump_offset());
                            let backward = target <= pc;
                            if HOOKED && backward && self.interrupt_pending() {
                                // On the word, which is not counted yet.
                                store_state!();
                                return Ok(FastExit::Fallback);
                            }
                            jump!(target);
                            if backward && let Some(exit) = self.fast_backedge(tracing, pc, &chunk.code) {
                                exit!(exit);
                            }
                            continue;
                        }};
                    }
                    // Register tests, one arm per operator like `comparison_arm`.
                    macro_rules! test_arm {
                        ($comparison:expr, $compare:expr, $order:expr) => {{
                            let (left, right) =
                                (&registers[a], &registers[usize::from(instruction.b())]);
                            #[cfg(feature = "op-stats")]
                            crate::vm::op_stats::arithmetic(
                                opcode_byte(instruction),
                                chunk,
                                pc,
                                left,
                                right,
                            );
                            let result = match (left, right) {
                                (Value::Number(left), Value::Number(right)) => {
                                    ($compare)(*left, *right)
                                }
                                (Value::Integer(left), Value::Integer(right))
                                    if tagged_integers =>
                                {
                                    ($compare)(*left, *right)
                                }
                                (Value::Integer(_), Value::Number(_))
                                | (Value::Number(_), Value::Integer(_)) => {
                                    ($order)(numeric_order(left, right))
                                }
                                _ => {
                                    let Some(result) = test_slow_with(
                                        &self.runtime,
                                        $comparison,
                                        left,
                                        right,
                                        float_arith,
                                    ) else {
                                        miss!();
                                    };
                                    result
                                }
                            };
                            test_branch!(result)
                        }};
                    }
                    // Constant tests. `K[b]` reads as `LoadConstant` would load it;
                    // `constant_left` swaps the operands of an ordering.
                    macro_rules! test_k_arm {
                        ($comparison:expr, $compare:expr, $order:expr) => {{
                            let left = &registers[a];
                            let constant = constant_at(chunk, instruction.b().into());
                            let swapped = instruction.c() & 2 != 0;
                            #[cfg(feature = "op-stats")]
                            crate::vm::op_stats::arithmetic(
                                opcode_byte(instruction),
                                chunk,
                                pc,
                                left,
                                &constant_value(constant, compatibility),
                            );
                            let result = match (left, constant) {
                                (Value::Number(value), Constant::Number(other)) => {
                                    let (l, r) = if swapped {
                                        (*other, *value)
                                    } else {
                                        (*value, *other)
                                    };
                                    ($compare)(l, r)
                                }
                                (Value::Number(value), Constant::Integer(other))
                                    if !tagged_integers =>
                                {
                                    let other = integer_to_number(*other);
                                    let (l, r) = if swapped {
                                        (other, *value)
                                    } else {
                                        (*value, other)
                                    };
                                    ($compare)(l, r)
                                }
                                (Value::Integer(value), Constant::Integer(other))
                                    if tagged_integers =>
                                {
                                    let (l, r) = if swapped {
                                        (*other, *value)
                                    } else {
                                        (*value, *other)
                                    };
                                    ($compare)(l, r)
                                }
                                (Value::Integer(value), Constant::Number(other)) => {
                                    let order = integer_float_order(*value, *other);
                                    ($order)(if swapped {
                                        order.map(std::cmp::Ordering::reverse)
                                    } else {
                                        order
                                    })
                                }
                                (Value::Number(value), Constant::Integer(other)) => {
                                    let order =
                                        integer_float_order(*other, *value).map(std::cmp::Ordering::reverse);
                                    ($order)(if swapped {
                                        order.map(std::cmp::Ordering::reverse)
                                    } else {
                                        order
                                    })
                                }
                                _ => {
                                    let Some(result) = test_k_slow(
                                        $comparison,
                                        left,
                                        constant,
                                        swapped,
                                        compatibility,
                                        float_arith,
                                    ) else {
                                        miss!();
                                    };
                                    result
                                }
                            };
                            test_branch!(result)
                        }};
                    }
                    macro_rules! plain_return {
                        ($a:expr, $count:expr, $instruction:expr) => {{
                            let record = frame!();
                            // One value from a frame with the plain mark,
                            // to a caller that wants one, when no return
                            // has anything else to do: the mark and one
                            // flag stand for every other test. The rest
                            // goes on below.
                            if $count != 0 && !self.plain_return_watch {
                                debug_assert!(!self.return_watch_due());
                                // SAFETY: `record` is the live top record
                                // and the registers its window; `$a` is a
                                // register of a validated `Return`.
                                #[allow(unsafe_code)]
                                let caller_window = unsafe {
                                    stack.pop_marked_return(record, registers.as_mut_ptr(), $a)
                                };
                                if !caller_window.is_null() {
                                    // The caller is the top record again:
                                    // the one just below, with its window
                                    // where the return found it.
                                    #[allow(unsafe_code)]
                                    unsafe {
                                        let caller = record.sub(1);
                                        set_frame!(caller);
                                        entry_pc = (*caller).pc;
                                        entry_chunk = (*caller).chunk.address();
                                    }
                                    window = caller_window.cast::<[Value; REGISTER_WINDOW]>();
                                    continue 'rebind;
                                }
                            }
                            // SAFETY: `record` is the live top record.
                            #[allow(unsafe_code)]
                            let caller_window = if self.pending_gc_recheck_depth.is_none() {
                                // One test of the caller's tag picks the
                                // return: a match over both kinds became a
                                // jump table on every return.
                                unsafe {
                                    // A frame with the plain mark
                                    // returns direct, and the mark is
                                    // the first thing its return tests.
                                    if (*record).plain_marked() || (*record).returns_direct() {
                                        stack.pop_plain_return(
                                            record,
                                            registers.as_mut_ptr(),
                                            $a,
                                            $count,
                                        )
                                    } else {
                                        // A plain `__index` or `__newindex`
                                        // handler returns to the register
                                        // its miss named, here too.
                                        stack.pop_handler_return(
                                            record,
                                            registers.as_mut_ptr(),
                                            $a,
                                            $count,
                                        )
                                    }
                                }
                            } else {
                                None
                            };
                            if let Some(caller_window) = caller_window {
                                if HOOKED {
                                    // SAFETY: the caller's record is the
                                    // one just below this one, and the
                                    // return made it the live top.
                                    #[allow(unsafe_code)]
                                    self.restore_record_hook_line(unsafe { &mut *record.sub(1) });
                                } else if !self.debug_hooks.is_empty()
                                    || self.count_lease.is_some()
                                {
                                    self.restore_caller_hook_line(stack);
                                }
                                // The caller is the top record again,
                                // with its window as it left it. Its
                                // record is the one just below this
                                // one, and the return located its
                                // window.
                                #[allow(unsafe_code)]
                                unsafe {
                                    set_frame!(record.sub(1));
                                }
                                window = caller_window.cast::<[Value; REGISTER_WINDOW]>();
                                rebind_from_record!();
                                continue 'rebind;
                            }
                            // A counted run leaves the declined return to
                            // the general routine after the macro, which
                            // marks the instruction as counted.
                            if !HOOKED {
                                match self
                                    .return_declined_plain(stack, $instruction, pc, jump)
                                    .map_err(|error| *error)?
                                {
                                    None => continue 'activations,
                                    Some(exit) => return Ok(exit),
                                }
                            }
                        }};
                    }
                    // Dispatch on the raw opcode byte. The wildcard arm is unreachable
                    // (validated chunks hold no larger opcode), which lets the
                    // compiler drop the range check in front of the jump table.
                    match opcode_byte(instruction) {
                        op::MOVE => {
                            refund_free!();
                            copy_register(registers, a, usize::from(instruction.b()));
                        }
                        op::LOAD_CONSTANT => {
                            let slot = &mut registers[a];
                            // Mirrors `constant_value`, building each variant at its store.
                            match constant_at(chunk, instruction.d().into()) {
                                Constant::Number(number) => {
                                    set_register(slot, Value::Number(*number));
                                }
                                Constant::Integer(integer) if tagged_integers => {
                                    set_register(slot, Value::Integer(*integer));
                                }
                                Constant::Integer(integer) => {
                                    set_register(slot, Value::Number(integer_to_number(*integer)));
                                }
                                Constant::Boolean(flag) => {
                                    set_register(slot, Value::Boolean(*flag));
                                }
                                Constant::Nil => set_register(slot, Value::Nil),
                                Constant::Int64(value) => set_register(slot, Value::Int64(*value)),
                                Constant::UInt64(value) => {
                                    set_register(slot, Value::UInt64(*value));
                                }
                                // A loop reloading a constant usually finds it still in
                                // place; skipping the store skips two atomic updates.
                                Constant::String(text)
                                    if matches!(slot, Value::String(current)
                                        if current.identity() == text.identity()) => {}
                                constant @ Constant::String(_) => {
                                    *slot = constant_value(constant, compatibility);
                                }
                            }
                        }
                        op::LOAD_NIL => {
                            refund_free!();
                            set_register(&mut registers[a], Value::Nil);
                        }
                        op::CLEAR_RANGE => {
                            refund_free!();
                            // The window through `opaque`: the loop below
                            // otherwise steps a pointer from an address in
                            // the window that is computed and stored at
                            // every activation switch.
                            //
                            // SAFETY: the window is the running one.
                            #[allow(unsafe_code)]
                            let window: &mut [Value; REGISTER_WINDOW] =
                                unsafe { &mut *opaque(std::ptr::from_mut(registers)) };
                            let Some(range) =
                                window.get_mut(a..a + usize::from(instruction.b()))
                            else {
                                exit!(FastExit::Fallback);
                            };
                            #[cfg(feature = "op-stats")]
                            crate::vm::op_stats::clear_range(range);
                            // Only a string owns anything; every other slot
                            // takes the tag without a drop call.
                            for register in range {
                                set_register(register, Value::Nil);
                            }
                        }
                        op::NOT => {
                            let value = Value::Boolean(
                                !registers[usize::from(instruction.b())].is_truthy(),
                            );
                            set_register(&mut registers[a], value);
                        }
                        op::JUMP => {
                            let target = relative(next, instruction.jump_offset());
                            let backward = target <= pc;
                            jump!(target);
                            if backward && let Some(exit) = self.fast_backedge(tracing, pc, &chunk.code) {
                                exit!(exit);
                            }
                            continue;
                        }
                        op::JUMP_IF_FALSE => {
                            if registers[a].is_truthy() {
                                pc = next;
                                continue;
                            }
                            let target = relative(next, short_offset(instruction));
                            let backward = target <= pc;
                            jump!(target);
                            if backward && let Some(exit) = self.fast_backedge(tracing, pc, &chunk.code) {
                                exit!(exit);
                            }
                            continue;
                        }
                        op::NUMERIC_FOR_PREP => {
                            let target = relative(next, short_offset(instruction));
                            let Some(window) = registers.get_mut(a..a + 4) else {
                                exit!(FastExit::Fallback);
                            };
                            match (&window[0], &window[1], &window[2]) {
                                (
                                    Value::Integer(control),
                                    Value::Integer(limit),
                                    Value::Integer(step),
                                ) if *step != 0 => {
                                    let (control, limit, step) = (*control, *limit, *step);
                                    let continues = if step > 0 {
                                        control <= limit
                                    } else {
                                        control >= limit
                                    };
                                    if continues {
                                        set_register(&mut window[3], Value::Integer(control));
                                    } else {
                                        // Skip the latch too, so an empty range
                                        // cannot wrap its control value. Validation
                                        // keeps `target` in range but not the
                                        // instruction after it.
                                        if target + 1 >= chunk.code.len() {
                                            exit!(FastExit::Fallback);
                                        }
                                        jump!(target + 1);
                                        continue;
                                    }
                                }
                                (Value::Number(control), Value::Number(_), Value::Number(step)) => {
                                    let value = Value::Number(*control - *step);
                                    set_register(&mut window[0], value);
                                    jump!(target);
                                    continue;
                                }
                                _ => exit!(FastExit::Fallback),
                            }
                        }
                        op::NUMERIC_FOR_LOOP => {
                            let target = relative(next, short_offset(instruction));
                            // SAFETY: `Chunk::validate` checked that
                            // `a ..= a + 3` are registers of the chunk,
                            // which lie inside the window.
                            #[allow(unsafe_code)]
                            let window: &mut [Value; 4] = unsafe {
                                debug_assert!(a + 3 < REGISTER_WINDOW);
                                &mut *registers.as_mut_ptr().add(a).cast::<[Value; 4]>()
                            };
                            let [control, limit, step, variable] = window;
                            // The control and the visible variable are updated in
                            // place: only their payloads change.
                            let continues = match (control, &*limit, &*step) {
                                (
                                    Value::Number(control),
                                    Value::Number(limit),
                                    Value::Number(step),
                                ) => {
                                    let value = *control + *step;
                                    *control = value;
                                    let continues = if *step > 0.0 {
                                        value <= *limit
                                    } else {
                                        value >= *limit
                                    };
                                    if continues {
                                        match variable {
                                            Value::Number(variable) => *variable = value,
                                            variable => {
                                                set_register(variable, Value::Number(value));
                                            }
                                        }
                                    }
                                    continues
                                }
                                (
                                    Value::Integer(control),
                                    Value::Integer(limit),
                                    Value::Integer(step),
                                ) => {
                                    // Lua integer loops stop at the edge of the integer domain.
                                    match control.checked_add(*step) {
                                        None => false,
                                        Some(value) => {
                                            *control = value;
                                            let continues = if *step > 0 {
                                                value <= *limit
                                            } else {
                                                value >= *limit
                                            };
                                            if continues {
                                                match variable {
                                                    Value::Integer(variable) => *variable = value,
                                                    variable => {
                                                        set_register(
                                                            variable,
                                                            Value::Integer(value),
                                                        );
                                                    }
                                                }
                                            }
                                            continues
                                        }
                                    }
                                }
                                _ => exit!(FastExit::Fallback),
                            };
                            if continues {
                                // The latch jumps back to the body, so it
                                // always polls (the compiler emits no forward
                                // latch; a hand-made chunk with one only polls
                                // too). The poll follows the jump, because the
                                // control is already stepped (see `jump_after!`).
                                jump = ((pc as u64 + 1) << 32) | target as u64;
                                pc = target;
                                if self.interrupt_pending() {
                                    exit!(FastExit::Fallback);
                                }
                                if let Some(exit) = self.fast_backedge(tracing, pc, &chunk.code) {
                                    exit!(exit);
                                }
                            } else {
                                pc = next;
                            }
                            continue;
                        }
                        // Ends a generic `for` on a nil first variable, and
                        // otherwise jumps back like `NumericForLoop` does.
                        op::FOR_IN_LOOP => {
                            if !for_in_advance(registers, a) {
                                pc = next;
                                continue;
                            }
                            let target = relative(next, short_offset(instruction));
                            let backward = target <= pc;
                            jump_after!(target);
                            if backward && let Some(exit) = self.fast_backedge(tracing, pc, &chunk.code) {
                                exit!(exit);
                            }
                            continue;
                        }
                        op::ADD => {
                            arithmetic_arm!(OpCode::Add, |l, r| l + r, |l: i64, r: i64| Some(
                                l.wrapping_add(r)
                            ));
                        }
                        op::SUBTRACT => {
                            arithmetic_arm!(OpCode::Subtract, |l, r| l - r, |l: i64, r: i64| Some(
                                l.wrapping_sub(r)
                            ));
                        }
                        op::MULTIPLY => {
                            arithmetic_arm!(OpCode::Multiply, |l, r| l * r, |l: i64, r: i64| Some(
                                l.wrapping_mul(r)
                            ));
                        }
                        // Integer division may fail (zero divisor), so integers
                        // take the general path. Integer modulo handles every
                        // nonzero divisor here and leaves `n%0` to the general
                        // path, which raises the error.
                        op::DIVIDE => {
                            arithmetic_arm!(OpCode::Divide, |l, r| l / r, |_l: i64, _r: i64| None);
                        }
                        op::MODULO => arithmetic_arm!(
                            OpCode::Modulo,
                            |l, r| float_modulo(floor_modulo, float_arith, l, r),
                            integer_floor_mod_nonzero
                        ),
                        // The constant-operand forms of the arithmetic arms: the
                        // right operand is `K[c]`, a number, read as
                        // `LoadConstant` would load it. Each opcode gets its own
                        // arm, so the operator is known in the arm.
                        op::ADD_K => arithmetic_k_arm!(op::ADD_K),
                        op::SUBTRACT_K => arithmetic_k_arm!(op::SUBTRACT_K),
                        op::MULTIPLY_K => arithmetic_k_arm!(op::MULTIPLY_K),
                        op::DIVIDE_K => arithmetic_k_arm!(op::DIVIDE_K),
                        // Written with the opcode read from the instruction, which
                        // the compiler still folds to this arm's value. Spelling
                        // it `op::MODULO_K` instead gives the same arm, but moves
                        // the GetField arm's registers and costs it 8 instructions.
                        // Recheck GetField on table/field_read with
                        // benchmarks/optrace.py after a toolchain or loop change; if
                        // it no longer moves, spell it like the other four arms.
                        op::MODULO_K => arithmetic_k_arm!(opcode_byte(instruction)),
                        op::EQUAL => comparison_arm!(|l, r| l == r, order_eq),
                        op::LESS_THAN => comparison_arm!(|l, r| l < r, order_lt),
                        op::LESS_EQUAL => comparison_arm!(|l, r| l <= r, order_le),
                        op::TEST_EQUAL => test_arm!(OpCode::Equal, |l, r| l == r, order_eq),
                        op::TEST_LESS_THAN => test_arm!(OpCode::LessThan, |l, r| l < r, order_lt),
                        op::TEST_LESS_EQUAL => test_arm!(OpCode::LessEqual, |l, r| l <= r, order_le),
                        op::TEST_EQUAL_K => {
                            // Only numbers need the operand-pair ladder: another
                            // type never equals a number, and `__eq` runs for two
                            // tables or two userdata, which no constant is.
                            let left = &registers[a];
                            let constant = constant_at(chunk, instruction.b().into());
                            #[cfg(feature = "op-stats")]
                            crate::vm::op_stats::arithmetic(
                                opcode_byte(instruction),
                                chunk,
                                pc,
                                left,
                                &constant_value(constant, compatibility),
                            );
                            let result = match (left, constant) {
                                (Value::Number(value), Constant::Number(other)) => value == other,
                                (Value::Integer(value), Constant::Integer(other))
                                    if tagged_integers =>
                                {
                                    value == other
                                }
                                (Value::Number(value), Constant::Integer(other))
                                    if !tagged_integers =>
                                {
                                    *value == integer_to_number(*other)
                                }
                                // An integer and a float are equal when exactly so.
                                (Value::Number(value), Constant::Integer(other)) => {
                                    order_eq(integer_float_order(*other, *value))
                                }
                                (Value::Integer(value), Constant::Number(other)) => {
                                    order_eq(integer_float_order(*value, *other))
                                }
                                (Value::Nil, Constant::Nil) => true,
                                (Value::Boolean(value), Constant::Boolean(other)) => value == other,
                                (Value::String(value), Constant::String(other)) => value == other,
                                _ => {
                                    let Some(result) = test_k_slow(
                                        OpCode::Equal,
                                        left,
                                        constant,
                                        false,
                                        compatibility,
                                        float_arith,
                                    ) else {
                                        miss!();
                                    };
                                    result
                                }
                            };
                            test_branch!(result)
                        }
                        op::TEST_LESS_THAN_K => test_k_arm!(OpCode::LessThan, |l, r| l < r, order_lt),
                        op::TEST_LESS_EQUAL_K => test_k_arm!(OpCode::LessEqual, |l, r| l <= r, order_le),
                        op::POWER | op::FLOOR_DIVIDE | op::NEGATE => {
                            #[cfg(feature = "op-stats")]
                            crate::vm::op_stats::slow_operator(instruction, chunk, pc, &registers[..]);
                            let Some(value) = scalar_slow(
                                instruction,
                                &registers[usize::from(instruction.b())],
                                &registers[usize::from(instruction.c())],
                                float_arith,
                            ) else {
                                miss!(resolve_plain_miss);
                            };
                            set_register(&mut registers[a], value);
                        }
                        // A string's length is primitive in every profile: `#`
                        // consults `__len` only for other operands, which take
                        // the scalar miss path.
                        op::LENGTH => {
                            let Value::String(text) = &registers[usize::from(instruction.b())]
                            else {
                                miss!(resolve_plain_miss);
                            };
                            let length = i64::try_from(text.len()).unwrap_or(i64::MAX);
                            let value = if tagged_integers {
                                Value::Integer(length)
                            } else {
                                Value::Number(integer_to_number(length))
                            };
                            set_register(&mut registers[a], value);
                        }
                        op::GET_GLOBAL => {
                            let hit = self
                                .runtime
                                .table_site_get(environment!(), site_at(chunk, read_sites!(), pc));
                            if let Some(value) = hit {
                                store_clone_scalar(&mut registers[a], value);
                            } else {
                                let Some(Constant::String(name)) =
                                    chunk.constants.get(usize::from(instruction.d()))
                                else {
                                    exit!(FastExit::Fallback);
                                };
                                if tagged_env && name.as_bytes() == b"_ENV" {
                                    exit!(FastExit::Fallback);
                                }
                                let Some(value) = self.runtime.table_get_string_cached(
                                    environment!(),
                                    name,
                                    &mut self.string_read_cache,
                                ) else {
                                    exit!(FastExit::Fallback);
                                };
                                store_clone(&mut registers[a], value);
                                fill_read_site(
                                    &self.runtime,
                                    chunk,
                                    code.wrapping_add(pc),
                                    environment!(),
                                    name,
                                    &registers[a],
                                );
                            }
                        }
                        // A global the environment holds, or any global when
                        // it has no metatable, is written directly. An
                        // absent global behind a metatable may reach
                        // `__newindex`, and a readonly environment errors:
                        // both take the general loop.
                        op::SET_GLOBAL => {
                            let Some(Constant::String(name)) =
                                chunk.constants.get(usize::from(instruction.d()))
                            else {
                                exit!(FastExit::Fallback);
                            };
                            let value = clone_register_value(&registers[a]);
                            if let Ok(true) = self.runtime.table_set_string_if_no_newindex(
                                environment!(),
                                name,
                                value,
                            ) {
                                pc = next;
                                if self.runtime.allocation_serial()
                                    != self.gc_observed_allocation_serial
                                    && self.runtime.gc_needs_collection()
                                {
                                    exit!(FastExit::Safepoint);
                                }
                                continue;
                            }                            exit!(FastExit::Fallback)
                        }
                        op::GET_UPVALUE => {
                            let Some(cell) = upvalues!().cell(usize::from(instruction.d())) else {
                                exit!(FastExit::Fallback);
                            };
                            // SAFETY: the running frame roots its captured
                            // cells, so a collection keeps them alive.
                            #[allow(unsafe_code)]
                            let value = unsafe { self.runtime.cell_ref_live(cell) };
                            store_clone_scalar(&mut registers[a], value);
                        }
                        op::SET_UPVALUE => {
                            if !self.store_upvalue(
                                upvalues!(),
                                usize::from(instruction.d()),
                                &registers[a],
                            ) {
                                miss!(resolve_plain_miss);
                            }
                        }
                        op::GET_TABLE => {
                            let Value::Table(table) = registers[usize::from(instruction.b())]
                            else {
                                exit!(FastExit::Fallback);
                            };
                            let key = &registers[usize::from(instruction.c())];
                            #[cfg(feature = "op-stats")]
                            crate::vm::op_stats::table_key(
                                key,
                                crate::vm::op_stats::Counter::InterpGetTableIntegerKey,
                                crate::vm::op_stats::Counter::InterpGetTableFloatKey,
                            );
                            let hit = if let Value::String(name) = key {
                                self.runtime.table_get_string_cached_hit(
                                    table,
                                    name,
                                    &mut self.string_read_cache,
                                )
                            } else {
                                self.runtime.table_get_field_hit(table, key)
                            };
                            let Some(value) = hit else {
                                if HOOKED || RECORD {
                                    exit!(FastExit::IndexMiss);
                                }
                                // A plain `__index` function is entered at
                                // once and the loop goes on in it. Hooked and
                                // recording runs take the exit, whose
                                // resolution they observe.
                                //
                                // This site and the ones below are shaped by
                                // what the loop's code generation tolerates:
                                // a `continue 'activations` from a store arm,
                                // or from every read arm, cost each dispatch
                                // of the loop an instruction.
                                store_state!();
                                match self.index_miss_activation(stack, false)? {
                                    None => continue 'activations,
                                    Some(exit) => return Ok(exit),
                                }
                            };
                            store_clone_scalar(&mut registers[a], value);
                        }
                        op::GET_FIELD => {
                            let holder = &registers[usize::from(instruction.b())];
                            // Cache hits and misses alike, as `SetField` counts.
                            #[cfg(feature = "op-stats")]
                            if let Value::Table(table) = *holder {
                                crate::vm::op_stats::field(&self.runtime, chunk, pc, table);
                            }
                            match self
                                .runtime
                                .table_site_probe(site_at(chunk, read_sites!(), pc), holder)
                            {
                                SiteProbe::Hit(value) => {
                                    store_clone_scalar(&mut registers[a], value);
                                }
                                SiteProbe::Miss(found) => {
                                    let Constant::String(name) =
                                        &chunk.constants[usize::from(instruction.c())]
                                    else {
                                        exit!(FastExit::Fallback);
                                    };
                                    let Value::Table(table) =
                                        registers[usize::from(instruction.b())]
                                    else {
                                        unreachable!("a probe finds only a table");
                                    };
                                    if let Some(value) =
                                        self.runtime.found_table_get_string_cached_hit(
                                            found,
                                            name,
                                            &mut self.string_read_cache,
                                        )
                                    {
                                        store_and_fill_read_site(
                                            &self.runtime,
                                            chunk,
                                            code.wrapping_add(pc),
                                            table,
                                            name,
                                            &mut registers[a],
                                            value,
                                        );
                                    } else if let Some(value) =
                                        // The receiver lacks the field and has a
                                        // metatable: a cached `__index` chain may
                                        // know the answer.
                                        self.runtime.table_inherited_get(table, name)
                                    {
                                        store_clone(&mut registers[a], value);
                                    } else {
                                        if HOOKED || RECORD {
                                            exit!(FastExit::IndexMiss);
                                        }
                                        // The outer loop resolves it, as it does a call.
                                        store_state!();
                                        break (instruction, false);
                                    }
                                }
                                // A stale handle misses, as a lookup would.
                                SiteProbe::Absent => {
                                    if matches!(holder, Value::Table(_)) {
                                        if HOOKED || RECORD {
                                            exit!(FastExit::IndexMiss);
                                        }
                                        // The outer loop resolves it, as it does a call.
                                        store_state!();
                                        break (instruction, false);
                                    }
                                    exit!(FastExit::Fallback);
                                }
                            }
                        }
                        // A method lookup: the receiver goes to `a + 1`, then
                        // the method is read from it into `a`, as `GetField`
                        // reads, through the same site. A hit stores both
                        // after the probe; a miss copies first, so every exit
                        // that repeats the instruction finds the copy
                        // already made (it is the same copy).
                        op::SELF_FIELD => {
                            let receiver = usize::from(instruction.b());
                            let holder = &registers[receiver];
                            #[cfg(feature = "op-stats")]
                            if let Value::Table(table) = *holder {
                                crate::vm::op_stats::field(&self.runtime, chunk, pc, table);
                            }
                            match self
                                .runtime
                                .table_site_probe(site_at(chunk, read_sites!(), pc), holder)
                            {
                                SiteProbe::Hit(value) => {
                                    copy_register_words(registers, a + 1, receiver);
                                    store_clone_scalar(&mut registers[a], value);
                                }
                                SiteProbe::Miss(found) => {
                                    let Constant::String(name) =
                                        &chunk.constants[usize::from(instruction.c())]
                                    else {
                                        exit!(FastExit::Fallback);
                                    };
                                    let Value::Table(table) = registers[receiver] else {
                                        unreachable!("a probe finds only a table");
                                    };
                                    copy_register_words(registers, a + 1, receiver);
                                    if let Some(value) =
                                        self.runtime.found_table_get_string_cached_hit(
                                            found,
                                            name,
                                            &mut self.string_read_cache,
                                        )
                                    {
                                        store_and_fill_read_site(
                                            &self.runtime,
                                            chunk,
                                            code.wrapping_add(pc),
                                            table,
                                            name,
                                            &mut registers[a],
                                            value,
                                        );
                                    } else if let Some(value) =
                                        // The receiver lacks the method and has a
                                        // metatable: a cached `__index` chain may
                                        // know the answer.
                                        self.runtime.table_inherited_get(table, name)
                                    {
                                        store_clone(&mut registers[a], value);
                                    } else {
                                        if HOOKED || RECORD {
                                            exit!(FastExit::IndexMiss);
                                        }
                                        // The outer loop resolves it, as it does a call.
                                        store_state!();
                                        break (instruction, false);
                                    }
                                }
                                // A stale handle misses, as a lookup would.
                                SiteProbe::Absent => {
                                    if matches!(holder, Value::Table(_)) {
                                        if HOOKED || RECORD {
                                            exit!(FastExit::IndexMiss);
                                        }
                                        store_state!();
                                        break (instruction, false);
                                    }
                                    // `s:method (...)`: the string library.
                                    if !self.string_self_field(registers, instruction, chunk) {
                                        exit!(FastExit::Fallback);
                                    }
                                }
                            }
                        }
                        op::SET_FIELD => {
                            #[cfg(feature = "op-stats")]
                            if let Value::Table(table) = registers[a] {
                                crate::vm::op_stats::field(&self.runtime, chunk, pc, table);
                            }
                            let site = site_at(chunk, chunk.write_sites_first(), pc);
                            // A number over a number or a boolean over a
                            // boolean, the common update of a field, is a
                            // copy with no call.
                            let stored = if let Some(table) = registers[a].table_handle_word() {
                                match self.runtime.table_site_store_scalar(
                                    table,
                                    site,
                                    &registers[usize::from(instruction.c())],
                                ) {
                                    SiteStore::Overwrote => FieldStore::Stored,
                                    SiteStore::Added => self.collect_after_store(),
                                    SiteStore::Missed => {
                                        self.store_field(chunk, site, registers, instruction)
                                    }
                                }
                            } else {
                                self.store_field(chunk, site, registers, instruction)
                            };
                            // The common outcome is one test; the others
                            // are out of the way.
                            if !matches!(stored, FieldStore::Stored) {
                                rare_outcome();
                                match stored {
                                    FieldStore::Replaced => fill_store_site(
                                        &mut self.runtime,
                                        chunk,
                                        code.wrapping_add(pc),
                                        instruction,
                                        registers,
                                    ),
                                    FieldStore::Stored => {}
                                    FieldStore::Collect => {
                                        pc = next;
                                        exit!(FastExit::Safepoint);
                                    }
                                    FieldStore::Missed => {
                                        if HOOKED || RECORD {
                                            exit!(FastExit::NewIndexMiss);
                                        }
                                        store_state!();
                                        break (instruction, true);
                                    }
                                    FieldStore::Declined => exit!(FastExit::IndexMiss),
                                }
                            }
                        }
                        // A global of Lua 5.2+: the field of the table in an
                        // upvalue (`_ENV`), read through the same site as
                        // `GetField`. A miss, a metatable or an upvalue that
                        // is not a table takes the general loop, which
                        // repeats the instruction.
                        op::GET_TAB_UP => {
                            let Some(holder) = upvalues!()
                                .get(usize::from(instruction.b()))
                                .and_then(|cell| self.runtime.cell_ref(*cell).ok())
                            else {
                                exit!(FastExit::Fallback);
                            };
                            #[cfg(feature = "op-stats")]
                            if let Value::Table(table) = *holder {
                                crate::vm::op_stats::field(&self.runtime, chunk, pc, table);
                            }
                            if let SiteProbe::Hit(value) = self
                                .runtime
                                .table_site_probe(site_at(chunk, read_sites!(), pc), holder)
                            {
                                store_clone_scalar(&mut registers[a], value);
                            } else {
                                let Value::Table(table) = *holder else {
                                    exit!(FastExit::Fallback);
                                };
                                match self.fast_get_field_miss(
                                    chunk,
                                    code.wrapping_add(pc),
                                    table,
                                    &mut registers[a],
                                    instruction,
                                ) {
                                    FieldMiss::Filled => {}
                                    FieldMiss::IndexMiss => {
                                        if HOOKED || RECORD {
                                            exit!(FastExit::IndexMiss);
                                        }
                                        store_state!();
                                        break (instruction, false);
                                    }
                                    FieldMiss::Fallback => exit!(FastExit::Fallback),
                                }
                            }
                        }
                        op::SET_TAB_UP => {
                            match self.fast_set_tab_up(
                                chunk,
                                upvalues!(),
                                code.wrapping_add(pc),
                                registers,
                                instruction,
                                #[cfg(feature = "op-stats")]
                                pc,
                            ) {
                                FieldStore::Replaced | FieldStore::Stored => {}
                                FieldStore::Collect => {
                                    pc = next;
                                    exit!(FastExit::Safepoint);
                                }
                                FieldStore::Declined | FieldStore::Missed => exit!(FastExit::Fallback),
                            }
                        }
                        op::SET_TABLE => {
                            let Value::Table(table) = registers[a] else {
                                exit!(FastExit::Fallback);
                            };
                            let source = &registers[usize::from(instruction.c())];
                            #[cfg(feature = "op-stats")]
                            crate::vm::op_stats::table_key(
                                &registers[usize::from(instruction.b())],
                                crate::vm::op_stats::Counter::InterpSetTableIntegerKey,
                                crate::vm::op_stats::Counter::InterpSetTableFloatKey,
                            );
                            if let Value::String(name) = &registers[usize::from(instruction.b())] {
                                // The value is copied only where it is stored.
                                let stored = self
                                    .runtime
                                    .table_set_string_hinted_from(table, name, source);
                                match stored {
                                    Ok(true) => {
                                        pc = next;
                                        if self.runtime.allocation_serial()
                                            != self.gc_observed_allocation_serial
                                            && self.runtime.gc_needs_collection()
                                        {
                                            exit!(FastExit::Safepoint);
                                        }
                                        continue;
                                    }
                                    // Nothing was written: `__newindex` may apply.
                                    Ok(false) => {
                                        if HOOKED || RECORD {
                                            exit!(FastExit::NewIndexMiss);
                                        }
                                        // The outer loop resolves it, as it does a call.
                                        store_state!();
                                        break (instruction, true);
                                    }
                                    // The general loop repeats the instruction for its error.
                                    Err(_) => exit!(FastExit::Fallback),
                                }
                            }
                            let key = &registers[usize::from(instruction.b())];
                            // Numbers and booleans are stored by reference.
                            // One over another in the array part allocates
                            // nothing: no collector poll.
                            let stored =
                                match self.runtime.table_store_dense_scalar(table, key, source) {
                                    DenseScalarStore::InPlace => {
                                        pc = next;
                                        continue;
                                    }
                                    DenseScalarStore::Stored => true,
                                    DenseScalarStore::Declined => false,
                                };
                            // Strings, tables and functions go straight to their slot too.
                            if !stored
                                && !self.runtime.table_set_dense_reference(table, key, source)
                            {
                                let value = clone_register_value(source);
                                match self.runtime.table_set_if_no_newindex(table, key, value) {
                                    Ok(true) => {}
                                    // Nothing was written: `__newindex` may apply.
                                    Ok(false) => {
                                        if HOOKED || RECORD {
                                            exit!(FastExit::NewIndexMiss);
                                        }
                                        store_state!();
                                        break (instruction, true);
                                    }
                                    // The general loop repeats the instruction for its error.
                                    Err(_) => exit!(FastExit::Fallback),
                                }
                            }
                            pc = next;
                            if self.runtime.allocation_serial()
                                != self.gc_observed_allocation_serial
                                && self.runtime.gc_needs_collection()
                            {
                                exit!(FastExit::Safepoint);
                            }
                            continue;
                        }
                        // Activation switches run outside this loop, which keeps its
                        // dispatch state in machine registers. The state is stored as
                        // for an exit, with `pc` still at the switching instruction.
                        op::CALL | op::CALL_ONE => {
                            // A plain Lua closure is called without storing
                            // the loop's state. A hooked run keeps the general
                            // routine, which evaluates inline plans under a
                            // count hook.
                            //
                            // One copy of the code per form of the call keeps
                            // the decoding of its operands out of the run time.
                            macro_rules! plain_call {
                                ($callee:expr, $target:expr, $results:expr, $first:expr, $count:expr, $consecutive:expr) => {{
                                    let caller_chunk: *const crate::bytecode::Chunk = chunk;
                                    // What follows a push that bound the
                                    // loop to the new activation.
                                    macro_rules! landed {
                                        () => {{
                                            // Tracing mode offers the entry of
                                            // a direct self call to the tracer,
                                            // as `call_closure` does. The
                                            // interpreter's copy has no test.
                                            if !DIRECT
                                                && tracing
                                                && std::ptr::eq(entry_chunk, caller_chunk)
                                                && !self.jit.trace.entry_rejected(entry_chunk as usize)
                                                && let Some(exit) = self.note_plain_call_landing(stack)
                                            {
                                                return Ok(exit);
                                            }
                                            // Calls are safepoints. A check
                                            // that finds nothing to do is
                                            // remembered, so the calls that
                                            // follow without an allocation
                                            // skip it.
                                            let serial = self.runtime.allocation_serial();
                                            if serial != self.gc_observed_allocation_serial {
                                                if self.runtime.gc_needs_collection() {
                                                    return Ok(FastExit::Safepoint);
                                                }
                                                self.gc_observed_allocation_serial = serial;
                                            }
                                            continue 'rebind;
                                        }};
                                    }
                                    // A flag raised from another thread must
                                    // stop a run of calls that never jumps
                                    // backwards; the general path delivers it.
                                    let pushed = if self.interrupt_pending() {
                                        Entered::Other(PlainCall::Declined)
                                    } else {
                                        // The callee that ran last at the
                                        // new depth goes into the record it
                                        // left there.
                                        let caller = frame!();
                                        // SAFETY: `caller` is the live top
                                        // record and the registers its window.
                                        #[allow(unsafe_code)]
                                        let entered = unsafe {
                                            stack.push_marked_call_at(
                                                &self.runtime,
                                                caller,
                                                registers.as_mut_ptr(),
                                                $callee,
                                                $target,
                                                $results,
                                                $first,
                                                $count,
                                                pc,
                                            )
                                        };
                                        if !entered.0.is_null() {
                                            // The record of a marked push
                                            // is the one after the caller's,
                                            // and the push read its chunk. A
                                            // plain activation starts at
                                            // zero.
                                            // SAFETY: that record exists.
                                            #[allow(unsafe_code)]
                                            unsafe {
                                                set_frame!(caller.add(1));
                                            }
                                            window = entered.0.cast::<[Value; REGISTER_WINDOW]>();
                                            entry_pc = 0;
                                            entry_chunk = entered.1;
                                            landed!();
                                        } else {
                                            // Every other call. Its operands
                                            // and its callee are read again,
                                            // from a word and a window the
                                            // compiler cannot relate to the
                                            // ones above: kept from there,
                                            // they were spilled on the path
                                            // of the marked push.
                                            let word = Instruction::from_raw(opaque_word(
                                                instruction.raw() as usize,
                                            ) as u32);
                                            let window = opaque(registers.as_mut_ptr());
                                            let (function, first, count, results) = if $consecutive {
                                                (
                                                    usize::from(word.a()),
                                                    usize::from(word.a()) + 1,
                                                    usize::from(word.b()),
                                                    word.c(),
                                                )
                                            } else {
                                                (usize::from(word.b()), usize::from(word.c()), 1, 1)
                                            };
                                            // SAFETY: the register held a
                                            // closure a few instructions ago
                                            // and nothing wrote it since;
                                            // `frame!()` is the live top
                                            // record and `window` its window.
                                            #[allow(unsafe_code)]
                                            unsafe {
                                                let Some(callee) = closure_handle(&*window.add(function))
                                                else {
                                                    std::hint::unreachable_unchecked()
                                                };
                                                stack.push_closure_call(
                                                    &self.runtime,
                                                    frame!(),
                                                    window,
                                                    callee,
                                                    word.a(),
                                                    results,
                                                    first,
                                                    count,
                                                    pc,
                                                    $consecutive,
                                                )
                                            }
                                        }
                                    };
                                    match pushed {
                                        Entered::Fast { .. }
                                        | Entered::Other(PlainCall::Pushed) => {
                                            // The new activation is the top
                                            // record, and its window lies
                                            // inside the stack (the push
                                            // grew it): bind the loop to it
                                            // without the prologue. This
                                            // comes before the safepoint
                                            // check below, so that the
                                            // values stay in the loop's own
                                            // variables and are not spilled
                                            // and reloaded around its call.
                                            if let Entered::Fast {
                                                frame: pushed_frame,
                                                window: pushed_window,
                                            } = pushed
                                            {
                                                // The push held both in
                                                // registers; a plain
                                                // activation starts at zero.
                                                set_frame!(pushed_frame);
                                                window = pushed_window
                                                    .cast::<[Value; REGISTER_WINDOW]>();
                                                entry_pc = 0;
                                                // SAFETY: the pushed record
                                                // is the live top.
                                                #[allow(unsafe_code)]
                                                unsafe {
                                                    entry_chunk = (*pushed_frame).chunk.address();
                                                }
                                            } else {
                                                #[allow(unsafe_code)]
                                                unsafe {
                                                    let (top, values) = stack.top_raw();
                                                    set_frame!(top);
                                                    window = values
                                                        .add((*top).base as usize)
                                                        .cast::<[Value; REGISTER_WINDOW]>();
                                                }
                                                rebind_from_record!();
                                            }
                                            landed!();
                                        }
                                        // A counted run leaves every call
                                        // the push declined to the general
                                        // routine after the macro, which
                                        // marks the instruction as counted.
                                        Entered::Other(_) if HOOKED => {}
                                        Entered::Other(PlainCall::Leave) => {
                                            return Self::call_plain_leave(stack, pc, jump);
                                        }
                                        Entered::Other(PlainCall::Plan) => {
                                            self.remember_plan_callee(stack, instruction);
                                            match self
                                                .call_plain_declined(stack, instruction, pc, jump, tracing)
                                                .map_err(|error| *error)?
                                            {
                                                None => continue 'activations,
                                                Some(exit) => return Ok(exit),
                                            }
                                        }
                                        Entered::Other(PlainCall::Declined) => match self
                                            .call_plain_declined(stack, instruction, pc, jump, tracing)
                                            .map_err(|error| *error)?
                                        {
                                            None => continue 'activations,
                                            Some(exit) => return Ok(exit),
                                        },
                                    }
                                }};
                            }
                            if (direct_calls && !HOOKED && !RECORD) || counted_calls {
                                if opcode_byte(instruction) == op::CALL {
                                    if let Some(callee) = closure_handle(&registers[a])
                                        && self.plan_callee != callee.native_identity() as u32
                                    {
                                        plain_call!(
                                            callee,
                                            instruction.a(),
                                            instruction.c(),
                                            a + 1,
                                            usize::from(instruction.b()),
                                            true
                                        );
                                    }
                                } else if let Some(callee) =
                                    closure_handle(&registers[usize::from(instruction.b())])
                                    && self.plan_callee != callee.native_identity() as u32
                                {
                                    plain_call!(
                                        callee,
                                        instruction.a(),
                                        1,
                                        usize::from(instruction.c()),
                                        1,
                                        false
                                    );
                                }
                            }
                            let function = if opcode_byte(instruction) == op::CALL {
                                a
                            } else {
                                usize::from(instruction.b())
                            };
                            // `tostring` and a literal `string.gsub` run in
                            // place with the natives.
                            let native = matches!(
                                registers[function],
                                Value::NativeFunction(_)
                                    | Value::BuiltinFunction(
                                        BuiltinFunction::ToString
                                            | BuiltinFunction::StringGsub
                                            | BuiltinFunction::Pairs
                                            | BuiltinFunction::IPairs
                                            | BuiltinFunction::Next
                                    )
                            );
                            // A native runs in place: it needs no frame views,
                            // so the loop's state stays in machine registers.
                            if native {
                                // A host callback with a fixed result count
                                // runs here, with the loop's state stored for
                                // it and nothing bound again afterwards: the
                                // stack is parked where it stands while the
                                // callback runs (see `call_host_in_registers`),
                                // so the record, the window and the code this
                                // loop holds are the ones it comes back to.
                                // Leaving for `native_call_activation` cost a
                                // declined window call and a pass of the
                                // activation prologue per call.
                                if !HOOKED
                                    && !RECORD
                                    && let Value::NativeFunction(host) = registers[function]
                                    && (instruction.c() != 0
                                        || opcode_byte(instruction) != op::CALL)
                                    && self.host_call_in_fast_loop(
                                        host,
                                        if opcode_byte(instruction) == op::CALL {
                                            usize::from(instruction.b())
                                        } else {
                                            1
                                        },
                                    )
                                {
                                    store_state!();
                                    match self.host_call_activation(stack, instruction, host)? {
                                        None => {
                                            pc = next;
                                            continue;
                                        }
                                        Some(exit) => return Ok(exit),
                                    }
                                }
                                let window = (running!().top - running!().base) as usize;
                                // A call whose results are open, of a native
                                // that returns one value, leaves it as the
                                // open results without the general path
                                // (`call_native_in_window` declines it).
                                let step = if !HOOKED
                                    && !RECORD
                                    && instruction.c() == 0
                                    && opcode_byte(instruction) == op::CALL
                                {
                                    self.open_native_in_window(
                                        registers,
                                        frame!(),
                                        instruction,
                                        chunk.code.get(next).copied(),
                                        window,
                                    )
                                } else {
                                    self.call_native_in_window(
                                        registers,
                                        instruction,
                                        window,
                                        stack.len(),
                                        false,
                                    )
                                };
                                match step {
                                    Ok(NativeStep::Declined) => {}
                                    Ok(NativeStep::Done) => {
                                        pc = next;
                                        if self.interrupt_pending() {
                                            // Past the call: the next
                                            // instruction is not counted yet.
                                            store_state!();
                                            return Ok(FastExit::Fallback);
                                        }
                                        continue;
                                    }
                                    Ok(NativeStep::Safepoint) => {
                                        pc = next;
                                        exit!(FastExit::Safepoint);
                                    }
                                    Ok(NativeStep::Backedge) => {
                                        pc = next;
                                        exit!(FastExit::Backedge);
                                    }
                                    Ok(NativeStep::Hooked(native, argument_count)) => {
                                        pc = next;
                                        store_state!();
                                        if let Some(exit) =
                                            self.native_hooked(stack, native, argument_count)?
                                        {
                                            return Ok(exit);
                                        }
                                        continue 'activations;
                                    }
                                    Ok(NativeStep::Fused) => {
                                        pc = next + 1;
                                        if self.interrupt_pending() {
                                            store_state!();
                                            return Ok(FastExit::Fallback);
                                        }
                                        continue;
                                    }
                                    Ok(NativeStep::FusedExit(backedge)) => {
                                        pc = next + 1;
                                        exit!(if backedge {
                                            FastExit::Backedge
                                        } else {
                                            FastExit::Safepoint
                                        });
                                    }
                                    Err(error) => {
                                        // A fused pair's error belongs to its
                                        // second call.
                                        pc = next + usize::from(std::mem::take(
                                            &mut self.fused_open_error,
                                        ));
                                        store_state!();
                                        return Err(self.native_error(
                                            &stack[stack.len() - 1],
                                            stack.len().checked_sub(2).map(|i| &stack[i]),
                                            error,
                                        ));
                                    }
                                }
                            }
                            mark_fetched!();
                            store_state!();
                            #[cfg(feature = "op-stats")]
                            if let Value::Closure(closure) = registers[function] {
                                crate::vm::op_stats::closure_call(
                                    &self.runtime,
                                    &running!().chunk.as_arc(),
                                    pc,
                                    instruction,
                                    &registers[..],
                                    stack.open_top() > 0,
                                    closure,
                                );
                            }
                            // A table or userdata goes the same way, for the
                            // plain `__call` closure its metatable may hold.
                            if DIRECT
                                && matches!(
                                    registers[function],
                                    Value::Closure(_) | Value::Table(_) | Value::Userdata(_)
                                )
                            {
                                match self
                                    .call_closure_general(stack, instruction)
                                    .map_err(|error| *error)?
                                {
                                    None => continue 'activations,
                                    Some(exit) => return Ok(exit),
                                }
                            }
                            if tracing && matches!(registers[function], Value::Closure(_)) {
                                match self.call_closure::<true>(stack, instruction)? {
                                    None => continue 'activations,
                                    Some(exit) => return Ok(exit),
                                }
                            }
                            break (instruction, native);
                        }
                        // The iterator call of a generic `for` leaves for
                        // `for_in_call_activation`, which keeps its code out
                        // of this loop and the `Call` arm free of its test.
                        // A plain return pops its activation without storing
                        // the loop's state: nothing resumes in it. One copy of
                        // the code per form of the return keeps the decoding
                        // of its operand out of the run time.
                        // An arm each: one arm for both tested the
                        // opcode again on every return.
                        op::RETURN => {
                            if (direct_calls && !HOOKED && !RECORD) || counted_calls {
                                // A count of zero means an open range
                                // instead.
                                let count = usize::from(instruction.b());
                                if count != 0 {
                                    plain_return!(a, count, instruction);
                                }
                            }
                            mark_fetched!();
                            store_state!();
                            break (instruction, false);
                        }
                        op::HALT => {
                            if (direct_calls && !HOOKED && !RECORD) || counted_calls {
                                plain_return!(a, 0, instruction);
                            }
                            mark_fetched!();
                            store_state!();
                            break (instruction, false);
                        }
                        // A tail call of a plain or vararg Lua closure replaces
                        // the activation in place, without leaving the loop.
                        // `tail_call_in_loop` declines, changing nothing, for
                        // the cases that need the general routine.
                        op::TAIL_CALL | op::TAIL_CALL_VARARG | op::TAIL_CALL_OPEN => {
                            // Tracing mode takes them here too, and notes each
                            // landing for the tracer as the general path does
                            // (`tail_landing_leaves`).
                            if direct_calls && !HOOKED && !RECORD && !self.interrupt_pending() {
                                if opcode_byte(instruction) == op::TAIL_CALL {
                                    // SAFETY: `frame` is the live top record
                                    // and `window` the start of its window;
                                    // `instruction` is the `TailCall` at `pc`.
                                    #[allow(unsafe_code)]
                                    let replaced = unsafe {
                                        stack.tail_call_closure_hot(
                                            &self.runtime,
                                            frame!(),
                                            window.cast::<Value>(),
                                            instruction,
                                            pc,
                                        )
                                    };
                                    if !replaced.is_null() {
                                        // The activation restarts at pc 0 in
                                        // the chunk the record now runs.
                                        entry_pc = 0;
                                        entry_chunk = replaced;
                                        if !DIRECT && tracing && { let depth = stack.len(); self.tail_landing_leaves(stack, depth) } {
                                            return Ok(FastExit::Backedge);
                                        }
                                        continue 'rebind;
                                    }
                                }
                                let step = if opcode_byte(instruction) == op::TAIL_CALL {
                                    stack.tail_call_closure(&self.runtime, instruction, pc)
                                } else {
                                    stack.tail_call_closure_forward(&self.runtime, instruction, pc)
                                };
                                match step {
                                    TailStep::Replaced => {
                                        if !DIRECT && tracing && { let depth = stack.len(); self.tail_landing_leaves(stack, depth) } {
                                            return Ok(FastExit::Backedge);
                                        }
                                        rebind_from_record!();
                                        continue 'rebind;
                                    }
                                    TailStep::Moved => {
                                        // The record is the same, its window
                                        // moved (and the stack may have grown).
                                        #[allow(unsafe_code)]
                                        unsafe {
                                            let (top, values) = stack.top_raw();
                                            set_frame!(top);
                                            window = values
                                                .add((*top).base as usize)
                                                .cast::<[Value; REGISTER_WINDOW]>();
                                        }
                                        if !DIRECT && tracing && { let depth = stack.len(); self.tail_landing_leaves(stack, depth) } {
                                            return Ok(FastExit::Backedge);
                                        }
                                        rebind_from_record!();
                                        continue 'rebind;
                                    }
                                    TailStep::Unclassified => {
                                        if self.tail_call_unclassified(stack, instruction, pc) {
                                            if !DIRECT && tracing && { let depth = stack.len(); self.tail_landing_leaves(stack, depth) } {
                                                return Ok(FastExit::Backedge);
                                            }
                                            continue 'activations;
                                        }
                                    }
                                    TailStep::Declined => {
                                        // `return host(...)` runs the host
                                        // callback and returns its results
                                        // in place.
                                        if opcode_byte(instruction) == op::TAIL_CALL
                                            && let Value::NativeFunction(native) = registers[a]
                                            && self.host_call_in_fast_loop(
                                                native,
                                                usize::from(instruction.b()),
                                            )
                                        {
                                            store_state!();
                                            match self
                                                .host_tail_return(stack, instruction, native)
                                                .map_err(|error| *error)?
                                            {
                                                None => continue 'activations,
                                                Some(exit) => return Ok(exit),
                                            }
                                        }
                                        // `return native(...)`: a native that
                                        // returns one value runs in place, as
                                        // for a `Call`, and the frame returns
                                        // that value. The general route builds
                                        // the native's results as open results
                                        // and returns them through several
                                        // calls.
                                        // A tail call returns the native's results
                                        // whole: only a native that returns one
                                        // value runs here, and none that a hook
                                        // observes.
                                        if opcode_byte(instruction) == op::TAIL_CALL
                                            && matches!(registers[a], Value::NativeFunction(_))
                                            && !self.hooks_observe_calls
                                            && self.pending_gc_recheck_depth.is_none()
                                            && stack.can_pop_plain_return()
                                        {
                                            let window = (running!().top - running!().base) as usize;
                                            let call = Instruction::abc(
                                                OpCode::Call,
                                                instruction.a(),
                                                instruction.b(),
                                                1,
                                            );
                                            match self.call_native_in_window(
                                                registers, call, window, stack.len(), true,
                                            ) {
                                                Ok(NativeStep::Declined) => {}
                                                // A collection or a flush the
                                                // native asked for waits for
                                                // the next safepoint or
                                                // backedge; `Hooked` is not
                                                // produced for a single call.
                                                Ok(
                                                    NativeStep::Done
                                                    | NativeStep::Safepoint
                                                    | NativeStep::Backedge
                                                    | NativeStep::Hooked(..)
                                                    | NativeStep::Fused
                                                    | NativeStep::FusedExit(_),
                                                ) => {
                                                    pc = next;
                                                    plain_return!(
                                                        a,
                                                        1,
                                                        Instruction::abc(OpCode::Return, instruction.a(), 1, 0)
                                                    );
                                                }
                                                Err(error) => {
                                                    pc = next;
                                                    store_state!();
                                                    return Err(self.native_error(
                                                        &stack[stack.len() - 1],
                                                        stack.len().checked_sub(2).map(|i| &stack[i]),
                                                        error,
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            if opcode_byte(instruction) == op::TAIL_CALL {
                                mark_fetched!();
                                store_state!();
                                break (instruction, false);
                            }
                            exit!(FastExit::Fallback);
                        }
                        op::CALL_OPEN | op::RETURN_OPEN => {
                            // A native that takes the open results of the
                            // call before it as its last arguments leaves
                            // through `native_call_activation`, which runs
                            // it in place when it returns one value, not
                            // through the dispatch loop. Running it here
                            // made this arm a call site, and the spills it
                            // cost the loop were 3% on `c3` and `shape`.
                            mark_fetched!();
                            store_state!();
                            break (
                                instruction,
                                !HOOKED
                                    && !RECORD
                                    && opcode_byte(instruction) == op::CALL_OPEN
                                    && matches!(registers[a], Value::NativeFunction(_)),
                            );
                        }
                        // A call that passes the frame's varargs on, and a
                        // return of them, read them where they lie on the
                        // value stack. Both stay out of line, which keeps
                        // their code out of the dispatch loop.
                        op::CALL_VARARG => {
                            if direct_other && !HOOKED && !RECORD {
                                match self.forward_call_in_loop(stack, instruction, pc) {
                                    LoopStep::Continue => {
                                        // A pushed activation: bind the loop
                                        // to it, as for a plain call.
                                        #[allow(unsafe_code)]
                                        unsafe {
                                            let (top, values) = stack.top_raw();
                                            set_frame!(top);
                                            window = values
                                                .add((*top).base as usize)
                                                .cast::<[Value; REGISTER_WINDOW]>();
                                        }
                                        rebind_from_record!();
                                        continue 'rebind;
                                    }
                                    LoopStep::Safepoint => return Ok(FastExit::Safepoint),
                                    LoopStep::Declined => {}
                                }
                            }
                            exit!(FastExit::Fallback);
                        }
                        op::RETURN_VARARG => {
                            if direct_other
                                && !HOOKED
                                && !RECORD
                                && self.pending_gc_recheck_depth.is_none()
                                && stack.pop_vararg_return(a, usize::from(instruction.b()))
                            {
                                if !self.debug_hooks.is_empty() || self.count_lease.is_some() {
                                    self.restore_caller_hook_line(stack);
                                }
                                // The caller is the top record again.
                                #[allow(unsafe_code)]
                                unsafe {
                                    let (top, values) = stack.top_raw();
                                    set_frame!(top);
                                    window = values
                                        .add((*top).base as usize)
                                        .cast::<[Value; REGISTER_WINDOW]>();
                                }
                                rebind_from_record!();
                                continue 'rebind;
                            }
                            exit!(FastExit::Fallback);
                        }
                        // `Vararg` copies the first `b` varargs to the
                        // registers from `a`, nil past the last.
                        op::VARARG => {
                            if chunk.arg_register.is_some() {
                                miss!(resolve_plain_miss);
                            }
                            let available = running!().vararg_count();
                            // SAFETY: `window` is the running window, and a
                            // frame's varargs lie just below it.
                            #[allow(unsafe_code)]
                            unsafe {
                                vararg_in_place(
                                    window.cast::<Value>(),
                                    a,
                                    instruction.b(),
                                    available,
                                );
                            }
                        }
                        op::NEW_TABLE => {
                            match self.new_table_in_place(
                                chunk,
                                registers,
                                instruction,
                                code.wrapping_add(pc),
                            ) {
                                TableCreate::Created => {}
                                TableCreate::Collect => {
                                    pc = next;
                                    exit!(FastExit::Safepoint);
                                }
                                TableCreate::Declined => miss!(resolve_plain_miss),
                            }
                        }
                        // Bitwise operators and the frame-local instructions
                        // run out of line, in place; so does a concatenation
                        // of strings and numbers (see `concat_in_place`).
                        // Their code stays out of this loop: inlined, they
                        // cost every other instruction a spilled register.
                        op::CONCAT => match self.concat_in_place(registers, instruction) {
                            ConcatStep::Done => {}
                            ConcatStep::Collect => {
                                pc = next;
                                exit!(FastExit::Safepoint);
                            }
                            ConcatStep::Declined => miss!(resolve_plain_miss),
                        },
                        op::BIT_AND
                        | op::BIT_OR
                        | op::BIT_XOR
                        | op::SHIFT_LEFT
                        | op::SHIFT_RIGHT
                        | op::BIT_NOT
                        | op::TABLE_VARARG
                        | op::TABLE_OPEN => miss!(resolve_plain_miss),
                        op::CLOSURE => match self.closure_in_place(stack, instruction) {
                            ConcatStep::Done => {}
                            ConcatStep::Collect => {
                                pc = next;
                                exit!(FastExit::Safepoint);
                            }
                            ConcatStep::Declined => miss!(resolve_plain_miss),
                        },
                        op::CLOSE_UPVALUE => {
                            let slot = running!().base + a as u32;
                            if stack.open_top() > slot {
                                self.close_upvalue_in_place(stack, slot);
                            }
                        }
                        // Opcodes only the general loop runs.
                        op::EXTRA_ARGUMENT
                        | op::LOAD_CONSTANT_WIDE
                        | op::GET_GLOBAL_WIDE
                        | op::SET_GLOBAL_WIDE
                        | op::MARK_CLOSE
                        | op::CLOSE_LOCAL
                        | op::DECLARE_GLOBAL
                        | op::DECLARE_GLOBAL_WIDE
                        | op::DECLARE_TABLE => exit!(FastExit::Fallback),
                        op::FOR_IN_CALL => {
                            // An `ipairs` or `next` step runs out of line but
                            // without leaving the loop: it can neither fail
                            // nor call out, so there is no state to save.
                            // Anything else, and every hooked run, leaves as
                            // before.
                            if HOOKED {
                                mark_fetched!();
                                store_state!();
                                break (instruction, false);
                            }
                            match self.for_in_step(registers, &chunk.code, pc, !RECORD) {
                                ForInStep::Fallback => {
                                    store_state!();
                                    break (instruction, false);
                                }
                                ForInStep::Next => {}
                                // The `ForInLoop` ended the loop.
                                ForInStep::Exit => {
                                    pc = next + 1;
                                    continue;
                                }
                                // The `ForInLoop` jumped back.
                                ForInStep::Back(target) => {
                                    pc = next;
                                    let backward = target <= pc;
                                    jump_after!(target);
                                    if backward && let Some(exit) = self.fast_backedge(tracing, pc, &chunk.code)
                                    {
                                        exit!(exit);
                                    }
                                    continue;
                                }
                            }
                        }
                        _ => unreachable_opcode(),
                    }
                    pc = next;
                };
                break 'rebind outcome;
                };
                outcome
            };
            let (switch, native) = switch;
            let exit = match opcode_byte(switch) {
                op::RETURN | op::HALT | op::RETURN_OPEN => {
                    self.return_fast(stack, switch).map_err(|error| *error)?
                }
                // Native calls dominate library-heavy loops; they need far
                // less state than a frame switch, so they skip its saves.
                op::FOR_IN_CALL => {
                    self.for_in_call_activation(stack, switch, direct_other, tracing)?
                }
                // A raw miss that a metamethod may answer, left by the loop
                // with its state stored: a read, or a store that found no raw
                // value and a `__newindex`.
                op::GET_FIELD => self.index_miss_activation(stack, false)?,
                op::SELF_FIELD | op::GET_TAB_UP => self.index_receiver_activation(stack)?,
                op::SET_FIELD | op::SET_TABLE => {
                    if opcode_byte(switch) == op::SET_TABLE && self.set_table_enter(stack, switch) {
                        None
                    } else {
                        self.index_miss_activation(stack, true)?
                    }
                }
                _ if native => match self.native_call_activation(stack, switch)? {
                    Some(exit) => exit,
                    None => self.switch_activation(stack, switch)?,
                },
                _ => self.switch_activation(stack, switch)?,
            };
            if let Some(exit) = exit {
                return Ok(exit);
            }
        }
    }

    /// Performs the call `instruction` of the running activation
    /// by switching activations, or reports why the fast loop must stop.
    /// `None` continues the fast loop in the new activation.
    #[allow(clippy::too_many_lines)]
    #[inline(never)]
    // The explicit native arm documents the dispatch order; the trailing
    // wildcard subsumes it with the same fallback.
    #[allow(clippy::match_same_arms)]
    fn switch_activation(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<Option<FastExit>> {
        let depth = stack.len();
        let a = usize::from(instruction.a());
        match instruction.opcode() {
            Ok(OpCode::Call | OpCode::CallOne) => {
                let frame = stack.at(depth - 1);
                let (function, results, arguments, argument_count) =
                    if instruction.opcode() == Ok(OpCode::Call) {
                        let count = usize::from(instruction.b());
                        (a, instruction.c(), a + 1, count)
                    } else {
                        (
                            usize::from(instruction.b()),
                            1,
                            usize::from(instruction.c()),
                            1,
                        )
                    };
                if results == 0
                    || depth >= MAX_LUA_CALL_FRAMES
                    || arguments + argument_count > frame.registers.len()
                {
                    return self.call_declined(stack, instruction);
                }
                let closure = match frame.registers.get(function) {
                    Some(&Value::Closure(closure)) => closure,
                    // The general call path's first fast cases, in its order.
                    // Natives were offered to `native_call_activation` first;
                    // the ones it declined are host callbacks, which run here
                    // when nothing watches calls, and need the dispatch loop
                    // otherwise.
                    Some(&Value::NativeFunction(native))
                        if self.host_call_in_fast_loop(native, argument_count) =>
                    {
                        return self.host_call_activation(stack, instruction, native);
                    }
                    Some(&Value::NativeFunction(_)) => return Ok(Some(FastExit::Fallback)),
                    // `tostring` of a value without `__tostring`, as the
                    // general call path's borrowed formatting does.
                    Some(Value::BuiltinFunction(BuiltinFunction::ToString))
                        if argument_count > 0 =>
                    {
                        if let Some(exit) = self.fast_tostring_number(
                            stack,
                            usize::from(instruction.a()),
                            usize::from(results),
                            arguments,
                        )? {
                            return Ok(exit);
                        }
                        let call = Instruction::call(instruction.a(), 0, results);
                        return self.fast_native_step(stack, depth, |vm, stack| {
                            vm.try_tostring_register(stack, call, arguments)
                        });
                    }
                    Some(Value::BuiltinFunction(BuiltinFunction::IPairsIterator))
                        if instruction.opcode() == Ok(OpCode::Call) =>
                    {
                        return self.fast_native_step(stack, depth, |vm, stack| {
                            Ok(instruction.b() == 2
                                && instruction.c() == 2
                                && vm.try_call_ipairs_iterator_fixed(stack, a, a, 2)?)
                        });
                    }
                    // `coroutine.resume` and `coroutine.yield` switch stacks
                    // without the dispatch loop (see `Vm::fast_resume` and
                    // `Vm::fast_yield`). Anything they decline, an error
                    // included, takes the general path.
                    Some(Value::BuiltinFunction(BuiltinFunction::CoroutineResume))
                        if argument_count > 0 =>
                    {
                        let call = Instruction::call(instruction.a(), 0, results);
                        // A thread that waits in a plain yield runs in this
                        // loop, in place of its resumer.
                        if let Some(&Value::Thread(thread)) = frame.registers.get(arguments)
                            && self.runs_long(thread)
                            && self.resume_in_place(
                                stack,
                                call,
                                thread,
                                (arguments + 1, argument_count - 1),
                                false,
                            )
                        {
                            return Ok(None);
                        }
                        return self.fast_native_step(stack, depth, |vm, stack| {
                            vm.fast_resume(stack, call, arguments, argument_count)
                        });
                    }
                    Some(Value::BuiltinFunction(BuiltinFunction::CoroutineYield)) => {
                        let call = Instruction::call(instruction.a(), 0, results);
                        // The yield of a thread this loop resumed in place
                        // goes back to the resumer's frame.
                        if self.in_place.len() > self.in_place_base
                            && let Some(exit) =
                                self.yield_in_place(stack, call, arguments, argument_count)
                        {
                            return Ok(exit);
                        }
                        let frame = &mut stack[depth - 1];
                        let previous = frame.last_interpreted_pc;
                        frame.last_interpreted_pc = crate::vm::frame::LastPc::some(frame.pc);
                        frame.pc += 1;
                        if self.fast_yield(stack, call, arguments, argument_count) {
                            return Ok(Some(FastExit::Yielded));
                        }
                        let frame = &mut stack[depth - 1];
                        frame.pc -= 1;
                        frame.last_interpreted_pc = previous;
                        return Ok(Some(FastExit::Fallback));
                    }
                    // `pcall` of a Lua closure and a call through a plain
                    // `__call` closure push the callee's activation here;
                    // every other shape keeps the dispatch loop.
                    Some(Value::BuiltinFunction(BuiltinFunction::ProtectedCall))
                        if instruction.opcode() == Ok(OpCode::Call)
                            && argument_count > 0
                            && self.fast_protected_calls() =>
                    {
                        return Ok(self.protected_call_activation(stack, instruction));
                    }
                    // A function `coroutine.wrap` made resumes its thread
                    // the same way, with every argument the thread's (see
                    // `Vm::fast_wrapped_resume`). A generic `for` over one
                    // comes here as the `Call` its `ForInCall` stands for.
                    Some(&Value::CoroutineWrapper(thread)) => {
                        let call = Instruction::call(instruction.a(), 0, results);
                        if self.runs_long(thread)
                            && self.resume_in_place(
                                stack,
                                call,
                                thread,
                                (arguments, argument_count),
                                true,
                            )
                        {
                            return Ok(None);
                        }
                        return self.fast_native_step(stack, depth, |vm, stack| {
                            vm.fast_wrapped_resume(stack, call, thread, arguments, argument_count)
                        });
                    }
                    Some(Value::Table(_) | Value::Userdata(_)) if self.fast_protected_calls() => {
                        return Ok(self.call_metamethod_activation(
                            stack,
                            instruction,
                            function,
                            results,
                            argument_count,
                        ));
                    }
                    _ => return Ok(Some(FastExit::Fallback)),
                };
                let arguments = frame.base as usize + arguments;
                let Some(first) = self.runtime.closure_call_parts(closure) else {
                    return Ok(Some(FastExit::Fallback));
                };
                let hint = first.call_hint;
                if hint & (CALL_HINT_PLAIN | CALL_HINT_NUMERIC_PLAN | CALL_HINT_TABLE_PLAN) == 0 {
                    return Ok(Some(FastExit::Fallback));
                }
                self.closure_call_rest(
                    stack,
                    instruction,
                    closure,
                    results,
                    arguments,
                    argument_count,
                    hint,
                )
            }
            Ok(OpCode::CallOpen) => self.call_open_activation(stack, instruction),
            Ok(OpCode::TailCall) => {
                // The general path's in-place replacement, minus its dispatch
                // overhead. It may also finish this frame through an inline
                // plan; the dispatch loop then re-examines the new top frame.
                // A tail call neither deepens the stack nor jumps backward, so
                // nothing else polls the interrupt flag along a chain of them.
                if self.interrupt_pending() {
                    return Ok(Some(FastExit::Fallback));
                }
                let exit = self.fast_native_step(stack, depth, |vm, stack| {
                    vm.try_fixed_tail_call(stack, instruction)
                })?;
                if exit.is_none() && stack.len() != depth {
                    return Ok(Some(FastExit::Backedge));
                }
                // In tracing mode the callee's pc 0 is a loop header for the
                // chain of tail calls that may come back to it. A header with
                // nothing to enter or record settles here, like a backward
                // branch's; any other leaves for the dispatch loop's lookup.
                if exit.is_none()
                    && self.jit.tail_landing
                    && !self.settle_fast_tail_landing(&stack[depth - 1].chunk)
                {
                    return Ok(Some(FastExit::Backedge));
                }
                Ok(exit)
            }
            _ => Ok(Some(FastExit::Fallback)),
        }
    }

    /// The `ForInCall` at `pc` for the fast loop: one `ipairs` or `next` step
    /// in the frame's registers, with the `ForInLoop` after it when `fuse`
    /// allows (a recording has to see that instruction itself). A step the
    /// fast case does not cover is [`ForInStep::Fallback`]. See
    /// [`Self::for_in_builtin_step`].
    ///
    /// Out of line on purpose: inlined into the dispatch loop, it cost
    /// every other instruction a spilled register.
    #[inline(never)]
    fn for_in_step(
        &mut self,
        registers: &mut [Value; REGISTER_WINDOW],
        code: &[Instruction],
        pc: usize,
        fuse: bool,
    ) -> ForInStep {
        let instruction = fetch(code, pc);
        let a = usize::from(instruction.a());
        let variables = usize::from(instruction.b());
        // The common step: `ipairs` over an array slot. The element is not
        // nil, so the loop goes on and its control value is the index:
        // written wide, since a copy of the narrow stores would stall.
        let array_step = if variables == 2
            && let Value::BuiltinFunction(BuiltinFunction::IPairsIterator) = registers[a]
            && let Value::Table(table) = registers[a + 1]
        {
            match registers[a + 2] {
                Value::Integer(control) => {
                    let index = control.wrapping_add(1);
                    usize::try_from(index)
                        .ok()
                        .and_then(|slot| self.runtime.table_array_hit(table, slot))
                        .map(|value| (value, IpairsIndex::Integer(index)))
                }
                Value::Number(control) => {
                    let index = control + 1.0;
                    self.runtime
                        .table_array_hit_number(table, index)
                        .map(|value| (value, IpairsIndex::Number(index)))
                }
                _ => None,
            }
        } else {
            None
        };
        let live = if let Some((value, index)) = array_step {
            crate::vm::registers::store_clone_wide(&mut registers[a + 4], value);
            match index {
                IpairsIndex::Integer(index) => {
                    crate::vm::registers::store_integer_wide(&mut registers[a + 3], index);
                    crate::vm::registers::store_integer_wide(&mut registers[a + 2], index);
                }
                IpairsIndex::Number(index) => {
                    crate::vm::registers::store_number_wide(&mut registers[a + 3], index);
                    crate::vm::registers::store_number_wide(&mut registers[a + 2], index);
                }
            }
            self.jit.statistics.interpreter_ipairs_steps =
                self.jit.statistics.interpreter_ipairs_steps.wrapping_add(1);
            true
        } else {
            // One call for the other steps, so that the `ipairs` path above
            // saves no more registers than it uses.
            match self.for_in_other_step(registers, a, variables) {
                Some(more) => more,
                None => return ForInStep::Fallback,
            }
        };
        if fuse
            && let Some(&word) = code.get(pc + 1)
            && opcode_byte(word) == op::FOR_IN_LOOP
            && word.a() == instruction.a()
        {
            if !live && !for_in_advance(registers, a) {
                return ForInStep::Exit;
            }
            return ForInStep::Back(relative(pc + 2, short_offset(word)));
        }
        ForInStep::Next
    }

    /// The `SelfField` `instruction` on a string receiver, as the general path
    /// runs it through the string metatable's `__index` table: the method a
    /// raw read of that table finds goes to `a`, the string to `a + 1`.
    /// `false` changes nothing: no string receiver or metatable, an
    /// `__index` that is no table, or a method the raw read does not find.
    #[inline(never)]
    fn string_self_field(
        &self,
        registers: &mut [Value; REGISTER_WINDOW],
        instruction: Instruction,
        chunk: &Chunk,
    ) -> bool {
        let a = usize::from(instruction.a());
        let receiver = usize::from(instruction.b());
        let (Value::String(_), Some(Constant::String(name)), Some(metatable)) = (
            &registers[receiver],
            chunk.constants.get(usize::from(instruction.c())),
            self.runtime.string_metatable(),
        ) else {
            return false;
        };
        let Ok(Value::Table(index)) = self
            .runtime
            .table_get_string(metatable, self.runtime.index_metamethod_key())
        else {
            return false;
        };
        let method = match self.runtime.table_get_string(index, name) {
            Ok(Value::Nil) | Err(_) => return false,
            Ok(method) => method,
        };
        // The receiver first: `a` may be its register.
        copy_register_words(registers, a + 1, receiver);
        set_register(&mut registers[a], method);
        true
    }

    /// A `ForInCall` step [`Self::for_in_step`] does not take itself: `next`
    /// in place, then the other builtins. `Some(true)` wrote the loop's
    /// variables and its control value; `Some(false)` wrote the variables
    /// only (the end of a `next` traversal leaves nil in the first), for
    /// [`for_in_advance`] to settle. `None` changed nothing.
    #[inline(never)]
    fn for_in_other_step(
        &mut self,
        registers: &mut [Value; REGISTER_WINDOW],
        a: usize,
        variables: usize,
    ) -> Option<bool> {
        if let Value::BuiltinFunction(BuiltinFunction::Next) = registers[a]
            && let Some(more) = self.next_in_place(registers, a, variables)
        {
            return Some(more);
        }
        self.for_in_builtin_step(registers, a, variables)
            .then_some(false)
    }

    /// One `next` step of a generic `for` over the table in `registers[a + 1]`
    /// from the control value in `registers[a + 2]`, written in place: the
    /// key to the first variable and to the control register (what the
    /// `ForInLoop` after the call would copy), the value to the second, nil to
    /// the others. `Some(false)` is the end of the table, with nil in the
    /// first variable and the control left alone. `None` changes nothing: the
    /// state is not a table, or the cursor cannot settle the continuation
    /// (which includes the invalid key that raises an error).
    ///
    /// The entry comes by reference from the table, so the step builds no
    /// owned pair: a number key or value is copied, a string cloned once per
    /// register.
    #[inline(always)]
    fn next_in_place(
        &mut self,
        registers: &mut [Value; REGISTER_WINDOW],
        a: usize,
        variables: usize,
    ) -> Option<bool> {
        let Value::Table(table) = registers[a + 1] else {
            return None;
        };
        if variables == 0 {
            return None;
        }
        let tagged = self.runtime.tagged_integers();
        let (control, rest) = registers.split_at_mut(a + 3);
        let control = &mut control[a + 2];
        let step = self
            .runtime
            .table_next_ref_hinted(table, control, &mut self.next_hint)?;
        let value = match step {
            NextRef::Array(index, value) => {
                if tagged {
                    let index = i64::try_from(index).ok()?;
                    crate::vm::registers::store_integer_wide(&mut rest[0], index);
                    crate::vm::registers::store_integer_wide(control, index);
                } else {
                    #[allow(clippy::cast_precision_loss, reason = "an array index")]
                    let index = index as f64;
                    crate::vm::registers::store_number_wide(&mut rest[0], index);
                    crate::vm::registers::store_number_wide(control, index);
                }
                value
            }
            NextRef::Hash(NextKey::String(key), value) => {
                set_register(&mut rest[0], Value::String(key.clone()));
                set_register(control, Value::String(key.clone()));
                value
            }
            NextRef::Hash(NextKey::Plain(key), value) => {
                crate::vm::registers::store_clone_wide(&mut rest[0], &key);
                set_register(control, key);
                value
            }
            NextRef::End => {
                set_register(&mut rest[0], Value::Nil);
                for slot in &mut rest[1..variables] {
                    set_register(slot, Value::Nil);
                }
                return Some(false);
            }
        };
        if variables >= 2 {
            crate::vm::registers::store_clone_wide(&mut rest[1], value);
            for slot in &mut rest[2..variables] {
                set_register(slot, Value::Nil);
            }
        }
        Some(true)
    }

    /// Performs the iterator call `instruction` (a `ForInCall`) of a generic
    /// `for`, like [`Self::switch_activation`] does for a `Call`. `None`
    /// continues the fast loop.
    ///
    /// An `ipairs` or `next` step writes its results in place. Any other
    /// iterator is called as the three copies into the window at `a + 3` and
    /// the `Call` they stand for: a Lua closure takes the calls the fast loop
    /// makes for a `Call`, a native runs in the window.
    #[inline(never)]
    fn for_in_call_activation(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
        direct_calls: bool,
        tracing: bool,
    ) -> VmResult<Option<FastExit>> {
        let depth = stack.len();
        let a = usize::from(instruction.a());
        if !matches!(
            stack.at(depth - 1).registers.get(a),
            Some(Value::BuiltinFunction(_))
        ) {
            return self.for_in_call_other(stack, instruction, direct_calls, tracing);
        }
        let variables = usize::from(instruction.b());
        self.fast_native_step(stack, depth, |vm, stack| {
            Ok(
                vm.try_call_ipairs_iterator_fixed(stack, a, a + 3, variables)?
                    || vm.try_call_next_iterator_fixed(stack, a, a + 3, variables)?,
            )
        })
    }

    /// The `ForInCall` of an iterator that is not a builtin.
    #[inline(never)]
    fn for_in_call_other(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
        direct_calls: bool,
        tracing: bool,
    ) -> VmResult<Option<FastExit>> {
        let depth = stack.len();
        let a = usize::from(instruction.a());
        // Read before `frames_and_values` borrows `stack` mutably.
        #[cfg(feature = "op-stats")]
        let open_top = stack.open_top();
        let (frames, values) = stack.frames_and_values();
        let frame = &frames[depth - 1];
        let registers = register_window(values, frame.base as usize);
        copy_register(registers, a + 3, a);
        copy_register(registers, a + 4, a + 1);
        copy_register(registers, a + 5, a + 2);
        let call = Instruction::call(instruction.a() + 3, 2, instruction.b());
        if let Value::Closure(_closure) = registers[a + 3] {
            #[cfg(feature = "op-stats")]
            crate::vm::op_stats::closure_call(
                &self.runtime,
                &frame.chunk.as_arc(),
                frame.pc,
                call,
                &registers[..],
                open_top > 0,
                _closure,
            );
            if direct_calls {
                return self.call_closure::<false>(stack, call);
            }
            if tracing {
                return self.call_closure::<true>(stack, call);
            }
        } else if let Some(exit) = self.native_call_activation(stack, call)? {
            return Ok(exit);
        }
        self.switch_activation(stack, call)
    }

    /// Whether the fast loop may push `pcall` and `__call` activations
    /// itself: the plain interpreter with no hook, whose calls and returns
    /// nothing observes. Tracing mode and hooked runs keep the dispatch
    /// loop's frames for them.
    #[inline]
    fn fast_protected_calls(&self) -> bool {
        self.execution_mode == ExecutionMode::Interpreter
            && self.debug_hooks.is_empty()
            && self.count_lease.is_none()
    }

    /// `pcall(f, ...)` of a plain Lua closure `f` for the `Call`
    /// `instruction`: `f`'s activation is pushed over the arguments in the
    /// caller's window, returning to a [`ReturnTarget::Protected`] that
    /// [`Self::return_activation`] completes with `true` and the results.
    /// Errors unwind to that frame as they do for the dispatch loop's frame.
    /// Anything else leaves the call to the dispatch loop before anything
    /// changes, as does `pcall` with a stack too deep to push.
    #[inline(never)]
    fn protected_call_activation(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
    ) -> Option<FastExit> {
        let depth = stack.len();
        let a = usize::from(instruction.a());
        let frame = stack.at(depth - 1);
        let Some(&Value::Closure(closure)) = frame.registers.get(a + 1) else {
            return Some(FastExit::Fallback);
        };
        let start = frame.base as usize + a + 2;
        let count = usize::from(instruction.b()) - 1;
        if self.interrupt_pending() {
            return Some(FastExit::Fallback);
        }
        self.classify_closure_call(closure);
        let Some(parts) = self.runtime.closure_call_parts(closure) else {
            return Some(FastExit::Fallback);
        };
        if parts.call_hint & CALL_HINT_PLAIN == 0 {
            return Some(FastExit::Fallback);
        }
        let caller = &mut stack[depth - 1];
        caller.pc += 1;
        caller.last_interpreted_pc = crate::vm::frame::LastPc::some(caller.pc - 1);
        // SAFETY: `parts` are `closure`'s own, and the frame roots `closure`.
        #[allow(unsafe_code)]
        unsafe {
            stack.push_activation_to(
                parts.chunk,
                parts.cells,
                parts.environment,
                closure,
                ReturnTarget::Protected(CallReturn {
                    base: instruction.a(),
                    results: instruction.c(),
                }),
                ActivationArguments::Stack { start, count },
            );
        }
        #[cfg(test)]
        {
            self.protected_fast_pushes += 1;
        }
        if self.runtime.allocation_serial() != self.gc_observed_allocation_serial
            && self.runtime.gc_needs_collection()
        {
            return Some(FastExit::Safepoint);
        }
        None
    }

    /// A call of the table or userdata in register `function` whose
    /// metatable has a plain Lua closure as `__call`: the handler's
    /// activation is pushed with the callee as its first argument. A `Call`
    /// has the callee in the slot just below its arguments, so the handler
    /// reads both as one run of the stack. Handlers that are not plain
    /// closures (natives, nested `__call` objects) and callees without one
    /// are left to the dispatch loop, which builds the call or the error.
    #[inline(never)]
    fn call_metamethod_activation(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
        function: usize,
        results: u8,
        argument_count: usize,
    ) -> Option<FastExit> {
        let depth = stack.len();
        let frame = stack.at(depth - 1);
        let Ok(Some(metatable)) = self.runtime.raw_metatable(&frame.registers[function]) else {
            return Some(FastExit::Fallback);
        };
        let Some(&Value::Closure(handler)) = self
            .runtime
            .table_get_string_ref(metatable, crate::vm::arith::call_metamethod_key())
        else {
            return Some(FastExit::Fallback);
        };
        if self.interrupt_pending() {
            return Some(FastExit::Fallback);
        }
        self.classify_closure_call(handler);
        let Some(parts) = self.runtime.closure_call_parts(handler) else {
            return Some(FastExit::Fallback);
        };
        if parts.call_hint & CALL_HINT_PLAIN == 0 {
            return Some(FastExit::Fallback);
        }
        let target = CallReturn {
            base: instruction.a(),
            results,
        };
        if instruction.opcode() == Ok(OpCode::Call) {
            // The callee and its arguments are one run of the stack.
            let start = frame.base as usize + function;
            let caller = &mut stack[depth - 1];
            caller.pc += 1;
            caller.last_interpreted_pc = crate::vm::frame::LastPc::some(caller.pc - 1);
            // SAFETY: `parts` are `handler`'s own, and the frame roots
            // `handler`.
            #[allow(unsafe_code)]
            unsafe {
                stack.push_stack_activation(
                    parts.chunk,
                    parts.cells,
                    parts.environment,
                    parts.shape,
                    handler,
                    target,
                    start,
                    argument_count + 1,
                );
            }
        } else {
            let argument = usize::from(instruction.c());
            let arguments = ActivationArguments::Pair(
                clone_register_value(&frame.registers[function]),
                clone_register_value(&frame.registers[argument]),
            );
            let caller = &mut stack[depth - 1];
            caller.pc += 1;
            caller.last_interpreted_pc = crate::vm::frame::LastPc::some(caller.pc - 1);
            // SAFETY: `parts` are `handler`'s own, and the frame roots
            // `handler`.
            #[allow(unsafe_code)]
            unsafe {
                stack.push_activation(
                    parts.chunk,
                    parts.cells,
                    parts.environment,
                    handler,
                    target,
                    arguments,
                );
            }
        }
        #[cfg(test)]
        {
            self.call_handler_pushes += 1;
        }
        if self.runtime.allocation_serial() != self.gc_observed_allocation_serial
            && self.runtime.gc_needs_collection()
        {
            return Some(FastExit::Safepoint);
        }
        None
    }

    /// The rest of a closure call that [`Self::switch_activation`] has
    /// decoded: inline plans and the plain activation. `arguments` is the absolute stack slot of the first
    /// argument and `hint` the callee's call hint.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    #[inline(always)]
    fn closure_call_rest(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
        closure: ClosureHandle,
        results: u8,
        arguments: usize,
        argument_count: usize,
        hint: u8,
    ) -> VmResult<Option<FastExit>> {
        let depth = stack.len();
        // The activation every plain call ends in.
        macro_rules! push_plain_call {
            ($parts:expr) => {{
                let parts = $parts;
                let caller = &mut stack[depth - 1];
                caller.pc += 1;
                caller.last_interpreted_pc = crate::vm::frame::LastPc::some(caller.pc - 1);
                // SAFETY: `parts` are `closure`'s own, and the frame
                // roots `closure`.
                #[allow(unsafe_code)]
                unsafe {
                    stack.push_activation(
                        parts.chunk,
                        parts.cells,
                        parts.environment,
                        closure,
                        CallReturn {
                            base: instruction.a(),
                            results,
                        },
                        ActivationArguments::Stack {
                            start: arguments,
                            count: argument_count,
                        },
                    );
                }
                if self.runtime.allocation_serial() != self.gc_observed_allocation_serial
                    && self.runtime.gc_needs_collection()
                {
                    return Ok(Some(FastExit::Safepoint));
                }
                return Ok(None);
            }};
        }
        // A callee without an inline plan needs no second lookup of its
        // parts.
        if results != 1 || hint & (CALL_HINT_NUMERIC_PLAN | CALL_HINT_TABLE_PLAN) == 0 {
            if hint & CALL_HINT_PLAIN == 0 {
                return Ok(Some(FastExit::Fallback));
            }
            let Some(first) = self.runtime.closure_call_parts(closure) else {
                return Ok(Some(FastExit::Fallback));
            };
            push_plain_call!(first);
        }
        // Inline plans run in the order of the general call path;
        // a plan that declines leaves an ordinary activation.
        if results == 1 && hint & (CALL_HINT_NUMERIC_PLAN | CALL_HINT_TABLE_PLAN) != 0 {
            let start = arguments - stack[depth - 1].base as usize;
            let evaluated = (hint & CALL_HINT_NUMERIC_PLAN != 0
                && self.try_inline_numeric_call(
                    stack,
                    instruction.a(),
                    &Value::Closure(closure),
                    InlineCallArguments::Registers {
                        start,
                        count: argument_count,
                    },
                )?)
                || (hint & CALL_HINT_TABLE_PLAN != 0
                    && instruction.opcode() == Ok(OpCode::Call)
                    && self.try_inline_table_call(
                        stack,
                        instruction,
                        closure,
                        start,
                        argument_count,
                    )?);
            if evaluated {
                let caller = &mut stack[depth - 1];
                caller.pc += 1;
                caller.last_interpreted_pc = crate::vm::frame::LastPc::some(caller.pc - 1);
                if self.runtime.allocation_serial() != self.gc_observed_allocation_serial
                    && self.runtime.gc_needs_collection()
                {
                    return Ok(Some(FastExit::Safepoint));
                }
                return Ok(None);
            }
        }
        if hint & CALL_HINT_PLAIN == 0 {
            return Ok(Some(FastExit::Fallback));
        }
        let Some(parts) = self.runtime.closure_call_parts(closure) else {
            return Ok(Some(FastExit::Fallback));
        };
        push_plain_call!(parts)
    }

    /// In tracing mode, after a tail call in the loop replaced the activation
    /// at `depth` (counting records), now at pc 0 of its new chunk: notes the
    /// landing for the tracer as the general path does (see
    /// `Vm::try_fixed_tail_call`), and tells whether the loop leaves there
    /// for the dispatch loop's lookup, as the general path's `Backedge` does.
    /// A chain that only passes through costs the note.
    #[inline(never)]
    fn tail_landing_leaves(&mut self, stack: &CallStack, depth: usize) -> bool {
        let record = &stack[depth - 1];
        self.note_tail_landing(depth - 1, record);
        self.jit.tail_landing && !self.settle_fast_tail_landing(&record.chunk)
    }

    /// The tail call [`CallStack::tail_call_closure`] left for a callee nobody
    /// had classified yet: classifies it and tries again.
    #[cold]
    #[inline(never)]
    fn tail_call_unclassified(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
        pc: usize,
    ) -> bool {
        let a = usize::from(instruction.a());
        let Some(&Value::Closure(callee)) = stack.last_registers().and_then(|r| r.get(a)) else {
            return false;
        };
        self.classify_closure_call(callee);
        let step = if opcode_byte(instruction) == op::TAIL_CALL {
            stack.tail_call_closure(&self.runtime, instruction, pc)
        } else {
            stack.tail_call_closure_forward(&self.runtime, instruction, pc)
        };
        matches!(step, TailStep::Replaced | TailStep::Moved)
    }

    /// The fast loop's `CallVararg` of a Lua closure: the activation is
    /// pushed over the fixed arguments and the frame's varargs.
    #[inline(never)]
    fn forward_call_in_loop(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
        pc: usize,
    ) -> LoopStep {
        let a = usize::from(instruction.a());
        let Some(&Value::Closure(callee)) = stack.last_registers().and_then(|r| r.get(a)) else {
            return LoopStep::Declined;
        };
        if self.interrupt_pending() {
            return LoopStep::Declined;
        }
        let (target, results) = (instruction.a(), instruction.c());
        let count = usize::from(instruction.b());
        let mut step =
            stack.push_forward_call(&self.runtime, callee, target, results, a + 1, count, pc);
        if step == PlainCall::Leave {
            // A callee nobody has classified yet.
            self.classify_closure_call(callee);
            step =
                stack.push_forward_call(&self.runtime, callee, target, results, a + 1, count, pc);
        }
        if step != PlainCall::Pushed {
            return LoopStep::Declined;
        }
        // Calls are safepoints.
        let serial = self.runtime.allocation_serial();
        if serial != self.gc_observed_allocation_serial {
            if self.runtime.gc_needs_collection() {
                return LoopStep::Safepoint;
            }
            self.gc_observed_allocation_serial = serial;
        }
        LoopStep::Continue
    }

    /// Remembers the closure `instruction` calls as one the general routine
    /// calls better than a plain activation: its next calls skip the plain
    /// attempt.
    #[cold]
    #[inline(never)]
    fn remember_plan_callee(&mut self, stack: &CallStack, instruction: Instruction) {
        let function = if opcode_byte(instruction) == op::CALL {
            usize::from(instruction.a())
        } else {
            usize::from(instruction.b())
        };
        if let Some(&Value::Closure(closure)) = stack
            .last_registers()
            .and_then(|registers| registers.get(function))
        {
            self.plan_callee = closure.native_identity() as u32;
        }
    }

    /// The plain call that only the dispatch loop makes: the loop's state is
    /// stored and it exits, as the general routine would.
    #[cold]
    #[inline(never)]
    fn call_plain_leave(stack: &mut CallStack, pc: usize, jump: u64) -> VmResult<FastExit> {
        if let Some(frame) = stack.last_mut() {
            store_loop_state(frame.frame, pc, jump);
        }
        Ok(FastExit::Fallback)
    }

    /// The plain call that was declined: the general routine runs from the
    /// loop's stored state.
    #[cold]
    #[inline(never)]
    fn call_plain_declined(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
        pc: usize,
        jump: u64,
        tracing: bool,
    ) -> FastStep {
        if let Some(frame) = stack.last_mut() {
            store_loop_state(frame.frame, pc, jump);
        }
        if self.interrupt_pending() {
            return Ok(Some(FastExit::Fallback));
        }
        if tracing {
            return self
                .call_closure::<true>(stack, instruction)
                .map_err(Box::new);
        }
        self.call_closure_general(stack, instruction)
    }

    #[inline(always)]
    fn call_closure_general(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
    ) -> FastStep {
        self.call_closure::<false>(stack, instruction)
            .map_err(Box::new)
    }

    /// The fast loop's call of a Lua function: pushes a plain closure's
    /// activation straight from the decoded call, without the general
    /// routine's other cases. Everything else, meaning other callees, depth
    /// limits, and callees with inline plans, goes to the general routine's
    /// code before anything changes. `None` continues the fast loop in the
    /// new top frame.
    ///
    /// `TRACING` selects the variant that offers self calls to the tracer.
    /// The copies are separate, so the interpreter's carries nothing of
    /// tracing mode.
    #[inline(never)]
    fn call_closure<const TRACING: bool>(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<Option<FastExit>> {
        #[cfg(test)]
        if TRACING {
            self.tracing_out_of_line_calls += 1;
        }
        let depth = stack.len();
        let (function, results, arguments, argument_count) = if opcode_byte(instruction) == op::CALL
        {
            let a = usize::from(instruction.a());
            (a, instruction.c(), a + 1, usize::from(instruction.b()))
        } else {
            (
                usize::from(instruction.b()),
                1,
                usize::from(instruction.c()),
                1,
            )
        };
        let frame = stack.at(depth - 1);
        let Some(&Value::Closure(closure)) = frame.registers.get(function) else {
            // The interpreter's loop hands over the tables and userdata it
            // calls: a plain `__call` closure is pushed from here, without
            // the general routine's decoding of the call. Anything that
            // declines changed nothing, and the general routine decides.
            if !TRACING
                && results != 0
                && depth < MAX_LUA_CALL_FRAMES
                && arguments + argument_count <= frame.registers.len()
                && matches!(
                    frame.registers.get(function),
                    Some(Value::Table(_) | Value::Userdata(_))
                )
                && self.fast_protected_calls()
            {
                match self.call_metamethod_activation(
                    stack,
                    instruction,
                    function,
                    results,
                    argument_count,
                ) {
                    Some(FastExit::Fallback) => {}
                    exit => return Ok(exit),
                }
            }
            return self.switch_activation(stack, instruction);
        };
        if results == 0
            || depth >= MAX_LUA_CALL_FRAMES
            || arguments + argument_count > frame.registers.len()
        {
            return self.switch_activation(stack, instruction);
        }
        let arguments = frame.frame.base as usize + arguments;
        let Some(parts) = self.runtime.closure_call_parts(closure) else {
            return Ok(Some(FastExit::Fallback));
        };
        let hint = parts.call_hint;
        if hint & CALL_HINT_PLAIN == 0
            || (results == 1 && hint & (CALL_HINT_NUMERIC_PLAN | CALL_HINT_TABLE_PLAN) != 0)
        {
            if hint & (CALL_HINT_PLAIN | CALL_HINT_NUMERIC_PLAN | CALL_HINT_TABLE_PLAN) == 0 {
                return Ok(Some(FastExit::Fallback));
            }
            return self.closure_call_rest(
                stack,
                instruction,
                closure,
                results,
                arguments,
                argument_count,
                hint,
            );
        }
        let caller = &mut stack[depth - 1];
        // Direct self recursion offers the callee's entry to the tracer.
        let recursive = TRACING
            && std::ptr::eq(caller.chunk.address(), Arc::as_ptr(parts.chunk))
            && !self
                .jit
                .trace
                .entry_rejected(Arc::as_ptr(parts.chunk) as usize);
        caller.pc += 1;
        caller.last_interpreted_pc = crate::vm::frame::LastPc::some(caller.pc - 1);
        // SAFETY: `parts` are `closure`'s own, and the frame roots `closure`.
        #[allow(unsafe_code)]
        unsafe {
            stack.push_activation(
                parts.chunk,
                parts.cells,
                parts.environment,
                closure,
                CallReturn {
                    base: instruction.a(),
                    results,
                },
                ActivationArguments::Stack {
                    start: arguments,
                    count: argument_count,
                },
            );
        }
        if recursive && let Some(exit) = self.note_fast_call_landing(stack) {
            return Ok(Some(exit));
        }
        if self.runtime.allocation_serial() != self.gc_observed_allocation_serial
            && self.runtime.gc_needs_collection()
        {
            return Ok(Some(FastExit::Safepoint));
        }
        Ok(None)
    }

    /// A self call just pushed the top frame: its entry may be a
    /// function-entry site. One that only counts or is rejected settles
    /// here; anything else (a recording to start, a trace to enter) leaves
    /// for the dispatch loop, which finds `JitState::call_landing` set.
    #[inline(never)]
    fn note_fast_call_landing(&mut self, stack: &CallStack) -> Option<FastExit> {
        self.jit.call_landing = true;
        let top = stack.len() - 1;
        if self.settle_fast_call_landing(&stack[top].chunk) {
            None
        } else {
            Some(FastExit::Backedge)
        }
    }

    /// [`Self::note_fast_call_landing`] for a frame the fast loop's plain call
    /// pushed. That push leaves the registers past the callee's entry window
    /// as they were, and a function-entry recording or trace expects a fresh
    /// frame to hold nil there, so a landing that leaves the loop clears them
    /// first. One that only counts reads none.
    #[inline(never)]
    fn note_plain_call_landing(&mut self, stack: &mut CallStack) -> Option<FastExit> {
        self.jit.call_landing = true;
        let depth = stack.len();
        if self.settle_fast_call_landing(&stack[depth - 1].chunk) {
            return None;
        }
        let (frames, values) = stack.frames_and_values();
        let frame = &frames[depth - 1];
        let base = frame.base as usize;
        let parameters = usize::from(frame.chunk.parameter_count);
        let registers = usize::from(frame.chunk.register_count);
        if parameters < registers {
            values[base + parameters..base + registers].fill(Value::Nil);
        }
        Some(FastExit::Backedge)
    }

    /// `tostring` of a number whose type has no `__tostring`, written
    /// straight into the call's result registers. Anything else declines
    /// with `None` before changing anything, and takes the general
    /// formatting path. The caller has checked for captured locals.
    // The three cases are the fast-path protocol: decline, complete, or an exit.
    #[allow(clippy::option_option)]
    #[inline(never)]
    fn fast_tostring_number(
        &mut self,
        stack: &mut CallStack,
        a: usize,
        results: usize,
        argument: usize,
    ) -> VmResult<Option<Option<FastExit>>> {
        let depth = stack.len();
        let (frames, values) = stack.frames_and_values();
        let frame = &mut frames[depth - 1];
        let base = frame.base as usize;
        let window = (frame.top - frame.base) as usize;
        if results == 0 || a + results > window || argument >= window {
            return Ok(None);
        }
        let value = &values[base + argument];
        let text = match value {
            Value::Number(number) => self.runtime.number_format().format_string(*number),
            Value::Integer(_) => self.runtime.format_value(value)?,
            _ => return Ok(None),
        };
        if self.has_tostring_metamethod(value)? {
            return Ok(None);
        }
        let pc = frame.pc;
        frame.last_interpreted_pc = crate::vm::frame::LastPc::some(pc);
        frame.pc = pc + 1;
        if frame.extras.is_some() {
            frame.clear_open_results();
        }
        let (first, rest) = values[base + a..base + a + results]
            .split_first_mut()
            .expect("a fixed call result has at least one destination");
        set_register(first, Value::String(text));
        for slot in rest {
            set_register(slot, Value::Nil);
        }
        if self.runtime.allocation_serial() != self.gc_observed_allocation_serial
            && self.runtime.gc_needs_collection()
        {
            return Ok(Some(Some(FastExit::Safepoint)));
        }
        Ok(Some(None))
    }

    /// Performs the return or `Halt` `instruction` of the running activation
    /// by switching back to its caller, like [`Self::switch_activation`].
    /// Returns take their own function: they need far less state than calls,
    /// so they skip most of that function's register saves.
    #[inline(never)]
    fn return_activation(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<Option<FastExit>> {
        let depth = stack.len();
        let a = usize::from(instruction.a());
        // `Halt` ends a function body without a `return`: no values.
        // A `Return` count of zero means an open range instead.
        let count = match (opcode_byte(instruction), usize::from(instruction.b())) {
            (op::RETURN_OPEN, _) => {
                return self.return_open_activation(stack, instruction);
            }
            (op::HALT, _) => 0,
            (_, 0) => return Ok(Some(FastExit::Fallback)),
            (_, count) => count,
        };
        // The dispatch loop must see a return that ends its run, and
        // frames with interior state close slots, captures, or varargs.
        if depth <= 1 || self.pending_gc_recheck_depth.is_some() {
            if depth == 1 && stack.host_entry() {
                return self.return_to_host(stack, a, count);
            }
            return Ok(Some(FastExit::Fallback));
        }
        let frame = &stack[depth - 1];
        if let Some(ReturnTarget::SetMetamethod) = frame.caller {
            return Ok(self.return_set_metamethod(stack));
        }
        let mut protected = false;
        let (target, results) = match &frame.caller {
            Some(ReturnTarget::Direct(CallReturn { base, results })) => (*base, *results),
            // A `pcall` of a Lua closure that the fast loop pushed (see
            // `Self::protected_call_activation`): `true` and the values go
            // into the caller's window. Other runs keep the general return.
            Some(ReturnTarget::Protected(CallReturn { base, results }))
                if *results != 0 && self.fast_protected_calls() =>
            {
                protected = true;
                (*base, *results)
            }
            Some(
                ReturnTarget::Scalar(_)
                | ReturnTarget::Operator(_)
                | ReturnTarget::Index(_)
                | ReturnTarget::IndexResult(_)
                | ReturnTarget::SetMetamethod
                | ReturnTarget::ConcatRange(_),
            ) => {
                return Ok(self.return_to_continuation(stack, a, count));
            }
            _ => return Ok(Some(FastExit::Fallback)),
        };
        if frame.extras.is_some() || a + count > (frame.top - frame.base) as usize {
            return Ok(Some(FastExit::Fallback));
        }
        let source = frame.base as usize + a;
        let caller = &stack[depth - 2];
        let (target, results) = (usize::from(target), usize::from(results));
        if results == 0 || target + results > (caller.top - caller.base) as usize {
            return self.return_declined(stack, instruction);
        }
        let destination = caller.base as usize + target;
        {
            let (_, values) = stack.frames_and_values();
            // The callee's window lies above the caller's, so the
            // results copy straight across without staging values.
            let (below, returned) = values.split_at_mut(source);
            let mut targets = &mut below[destination..destination + results];
            if protected {
                set_register(&mut targets[0], Value::Boolean(true));
                targets = &mut targets[1..];
            }
            let copied = count.min(targets.len());
            for (slot, value) in targets.iter_mut().zip(&returned[..copied]) {
                store_clone(slot, value);
            }
            for slot in &mut targets[copied..] {
                set_register(slot, Value::Nil);
            }
        }
        self.retire_top_frame(stack);
        #[cfg(test)]
        if protected {
            self.protected_fast_returns += 1;
        }
        let caller = &mut stack[depth - 2];
        caller.clear_open_results();
        self.restore_caller_hook_line(stack);
        Ok(None)
    }

    /// The fixed return of `count` values from register `a` by the bottom
    /// activation of a host entry: the values go to `host_returned` and the
    /// run ends with [`FastExit::Returned`], as the dispatch loop's return
    /// with no caller ends it (`Vm::finish_frame`), without a trip through
    /// that loop. Anything a return observes or closes besides open
    /// captures (hooks, a recording, interior frame state, a deferred
    /// collector check) takes that loop instead.
    #[inline(never)]
    fn return_to_host(
        &mut self,
        stack: &mut CallStack,
        a: usize,
        count: usize,
    ) -> VmResult<Option<FastExit>> {
        let frame = &stack[0];
        let (base, top) = (frame.base as usize, frame.top as usize);
        if frame.caller.is_some()
            || frame.extras.is_some()
            || a + count > top - base
            || self.pending_gc_recheck_depth.is_some()
            || !self.debug_hooks.is_empty()
            || self.jit.trace.recording()
        {
            return Ok(Some(FastExit::Fallback));
        }
        let values = stack.values()[base + a..base + a + count].to_vec();
        self.close_top_frame_upvalues(stack);
        stack.discard_top();
        if self.runtime.allocation_serial() != self.gc_observed_allocation_serial
            || self.runtime.gc_needs_collection()
        {
            self.gc_safepoint_with_values(stack, &values)?;
            self.gc_observed_allocation_serial = self.runtime.allocation_serial();
        }
        self.host_returned = values;
        Ok(Some(FastExit::Returned))
    }

    /// The return of a plain `__index` or `__newindex` handler. An `__index`
    /// handler's [`ReturnTarget::IndexResult`] names the register that gets
    /// the first result, or nil for none, as [`Self::return_to_continuation`]
    /// writes it; a `__newindex` handler's result is discarded. Returns
    /// `false` before anything changes for every other return, which
    /// [`Self::return_activation`] decides.
    #[inline(always)]
    fn return_handler_fast(&mut self, stack: &mut CallStack, instruction: Instruction) -> bool {
        // The one test every other return pays.
        let depth = stack.len();
        if !matches!(
            stack[depth - 1].caller,
            Some(
                ReturnTarget::IndexResult(_)
                    | ReturnTarget::SetMetamethod
                    | ReturnTarget::Operator(_)
            )
        ) {
            return false;
        }
        let count = match opcode_byte(instruction) {
            op::RETURN => usize::from(instruction.b()),
            op::HALT => 0,
            _ => return false,
        };
        if (count == 0 && opcode_byte(instruction) == op::RETURN)
            || self.pending_gc_recheck_depth.is_some()
        {
            return false;
        }
        // SAFETY: a frame runs, so the top record is the live top,
        // and its window starts at its base.
        #[allow(unsafe_code)]
        let popped = unsafe {
            let (frame, values) = stack.top_raw();
            let window = values.add((*frame).base as usize);
            stack.pop_handler_return(frame, window, usize::from(instruction.a()), count)
        };
        if popped.is_none() {
            return false;
        }
        if !self.debug_hooks.is_empty() || self.count_lease.is_some() {
            self.restore_caller_hook_line(stack);
        }
        true
    }

    /// The return of a `__newindex` handler: its result is discarded, so the
    /// frame only retires and the caller's hook line is restored. Same effects
    /// as the `SetMetamethod` arm of [`Self::return_to_continuation`].
    #[inline(never)]
    fn return_set_metamethod(&mut self, stack: &mut CallStack) -> Option<FastExit> {
        let depth = stack.len();
        if stack[depth - 1].extras.is_some() || stack[depth - 2].extras.is_some() {
            return Some(FastExit::Fallback);
        }
        // Continuations are boxed; release this one now rather than
        // leaving it in the retired record.
        stack[depth - 1].clear_plain_mark();
        stack[depth - 1].caller = None;
        self.retire_top_frame(stack);
        if !self.debug_hooks.is_empty() || self.count_lease.is_some() {
            self.restore_caller_hook_line(stack);
        }
        None
    }

    /// Retires the top frame after closing the captured locals that live in
    /// its registers.
    #[inline(always)]
    fn retire_top_frame(&mut self, stack: &mut CallStack) {
        self.close_upvalues(stack, stack.top_base());
        stack.retire_top();
    }

    /// The return [`Self::pop_plain_return`] declined: the loop's state is
    /// stored and [`Self::return_activation`] decides.
    #[cold]
    #[inline(never)]
    fn return_declined_plain(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
        pc: usize,
        jump: u64,
    ) -> FastStep {
        // A handler's return into the instruction that called it needs none
        // of the loop's state: the caller resumes from its own record.
        if self.return_handler_fast(stack, instruction) {
            return Ok(None);
        }
        if let Some(frame) = stack.last_mut() {
            store_loop_state(frame.frame, pc, jump);
        }
        self.return_activation(stack, instruction).map_err(Box::new)
    }

    /// A return from the fast loop: the plain return of
    /// [`Self::pop_plain_return`], else [`Self::return_activation`]. The
    /// result is two words, so the common return passes it in registers.
    #[inline(never)]
    fn return_fast(&mut self, stack: &mut CallStack, instruction: Instruction) -> FastStep {
        if self.pop_plain_return(stack, instruction) {
            return Ok(None);
        }
        if self.return_handler_fast(stack, instruction) {
            return Ok(None);
        }
        self.return_activation(stack, instruction).map_err(Box::new)
    }

    /// The common return of [`Self::return_activation`]: fixed results from
    /// a frame without interior state to a direct caller that holds no
    /// captured locals. Returns `false` before anything changes for every
    /// other return, which the general routine decides.
    #[inline(always)]
    fn pop_plain_return(&mut self, stack: &mut CallStack, instruction: Instruction) -> bool {
        let count = match opcode_byte(instruction) {
            op::RETURN => usize::from(instruction.b()),
            op::HALT => 0,
            _ => return false,
        };
        // A `Return` count of zero means an open range instead.
        if (count == 0 && opcode_byte(instruction) == op::RETURN)
            || self.pending_gc_recheck_depth.is_some()
        {
            return false;
        }
        // SAFETY: a frame runs, and the top record is
        // the live top.
        #[allow(unsafe_code)]
        let popped = unsafe {
            let (frame, values) = stack.top_raw();
            let window = values.add((*frame).base as usize);
            stack.pop_plain_return(frame, window, usize::from(instruction.a()), count)
        };
        if popped.is_none() {
            return false;
        }
        if !self.debug_hooks.is_empty() || self.count_lease.is_some() {
            self.restore_caller_hook_line(stack);
        }
        true
    }

    /// [`Self::return_activation`] for a `ReturnOpen`: the fixed values and
    /// the open results of the frame go back to the caller, into its
    /// registers or, for an open result count, into its open results.
    #[inline(never)]
    fn return_open_activation(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<Option<FastExit>> {
        let depth = stack.len();
        if depth <= 1 || self.pending_gc_recheck_depth.is_some() {
            return Ok(Some(FastExit::Fallback));
        }
        let frame = &stack[depth - 1];
        let (target, results) = match &frame.caller {
            Some(ReturnTarget::Direct(CallReturn { base, results })) => {
                (usize::from(*base), usize::from(*results))
            }
            _ => return Ok(Some(FastExit::Fallback)),
        };
        let a = usize::from(instruction.a());
        let fixed = usize::from(instruction.b());
        let caller = &stack[depth - 2];
        let caller_window = (caller.top - caller.base) as usize;
        if frame.extras.is_some()
            || !frame.has_open_results()
            || a + fixed > (frame.top - frame.base) as usize
            || target + results > caller_window
            || target > caller_window
        {
            return Ok(Some(FastExit::Fallback));
        }
        let source = frame.base as usize + a;
        let destination = caller.base as usize + target;
        {
            let (frames, values) = stack.frames_and_values();
            let (lower, upper) = frames.split_at_mut(depth - 1);
            let (parent, callee_frame) = (&mut lower[depth - 2], &mut upper[0]);
            let open = callee_frame.open_values().unwrap_or_default();
            if results == 0 {
                let list = parent.open_buffer();
                list.extend_from_slice(&values[source..source + fixed]);
                list.extend_from_slice(open);
            } else {
                // The callee's window lies above the caller's.
                let (below, returned) = values.split_at_mut(source);
                let targets = &mut below[destination..destination + results];
                for (position, slot) in targets.iter_mut().enumerate() {
                    match position.checked_sub(fixed) {
                        None => store_clone(slot, &returned[position]),
                        Some(index) => match open.get(index) {
                            Some(value) => store_clone(slot, value),
                            None => set_register(slot, Value::Nil),
                        },
                    }
                }
                parent.clear_open_results();
            }
            callee_frame.clear_open_results();
        }
        self.retire_top_frame(stack);
        self.restore_caller_hook_line(stack);
        Ok(None)
    }

    /// Performs a `CallOpen`, whose last arguments are the open results of
    /// the frame, for a native or a plain Lua callee. Other callees take the
    /// general path.
    #[inline(never)]
    fn call_open_activation(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<Option<FastExit>> {
        let depth = stack.len();
        let frame = stack.at(depth - 1);
        let a = usize::from(instruction.a());
        let fixed = usize::from(instruction.b());
        if !frame.has_open_results()
            || depth >= MAX_LUA_CALL_FRAMES
            || a + 1 + fixed > frame.registers.len()
        {
            return Ok(Some(FastExit::Fallback));
        }
        let closure = match frame.registers.get(a) {
            Some(&Value::Closure(closure)) => closure,
            Some(Value::NativeFunction(_)) => {
                return Ok(self
                    .native_open_call(stack, instruction, false)?
                    .unwrap_or(Some(FastExit::Fallback)));
            }
            _ => return Ok(Some(FastExit::Fallback)),
        };
        self.push_open_closure_call(stack, instruction, closure, a + 1, fixed)
    }

    /// Pushes the activation of a plain Lua `closure` for the call
    /// `instruction` of the top frame, whose fixed arguments are `fixed`
    /// registers from `first`. A `CallOpen` appends the open results of the
    /// frame. A callee that is not plain falls back before anything changes.
    fn push_open_closure_call(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
        closure: ClosureHandle,
        first: usize,
        fixed: usize,
    ) -> VmResult<Option<FastExit>> {
        let depth = stack.len();
        let frame = stack.at(depth - 1);
        let arguments = frame.base as usize + first;
        let Some(parts) = self.runtime.closure_call_parts(closure) else {
            return Ok(Some(FastExit::Fallback));
        };
        if parts.call_hint & CALL_HINT_PLAIN == 0 {
            return Ok(Some(FastExit::Fallback));
        }
        let parameters = usize::from(parts.chunk.parameter_count);
        let caller = &mut stack[depth - 1];
        caller.pc += 1;
        caller.last_interpreted_pc = crate::vm::frame::LastPc::some(caller.pc - 1);
        // SAFETY: `parts` are `closure`'s own, and the frame roots `closure`.
        #[allow(unsafe_code)]
        unsafe {
            stack.push_stack_activation(
                parts.chunk,
                parts.cells,
                parts.environment,
                parts.shape,
                closure,
                CallReturn {
                    base: instruction.a(),
                    results: instruction.c(),
                },
                arguments,
                fixed,
            );
        }
        if opcode_byte(instruction) == op::CALL_OPEN {
            // The open results fill the parameters after the fixed ones.
            let (frames, values) = stack.frames_and_values();
            let new_base = frames[depth].base as usize;
            let caller = &mut frames[depth - 1];
            let open = caller.open_values().unwrap_or_default();
            for (position, value) in open
                .iter()
                .enumerate()
                .take(parameters.saturating_sub(fixed))
            {
                values[new_base + fixed + position] = value.clone();
            }
            caller.clear_open_results();
        }
        // Direct self recursion offers the callee's entry to the tracer.
        if self.jit.trace.enabled()
            && std::ptr::eq(
                stack[depth - 1].chunk.address(),
                stack[depth].chunk.address(),
            )
            && !self
                .jit
                .trace
                .entry_rejected(stack[depth].chunk.address() as usize)
            && let Some(exit) = self.note_fast_call_landing(stack)
        {
            return Ok(Some(exit));
        }
        if self.runtime.allocation_serial() != self.gc_observed_allocation_serial
            && self.runtime.gc_needs_collection()
        {
            return Ok(Some(FastExit::Safepoint));
        }
        Ok(None)
    }

    /// The cold side of [`Self::return_activation`]: a return the fixed-count
    /// path cannot take. A return into an open result count moves the values
    /// whole into the caller's open-results slot, as the general path's
    /// `write_call_results` does; anything else declines. It recomputes its
    /// operands so that the hot path keeps none of them alive for it.
    #[cold]
    #[inline(never)]
    fn return_declined(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<Option<FastExit>> {
        let depth = stack.len();
        let frame = &stack[depth - 1];
        let Some(ReturnTarget::Direct(CallReturn { base, results: 0 })) = &frame.caller else {
            return Ok(Some(FastExit::Fallback));
        };
        let target = usize::from(*base);
        let count = if opcode_byte(instruction) == op::HALT {
            0
        } else {
            usize::from(instruction.b())
        };
        let source = frame.base as usize + usize::from(instruction.a());
        let caller = &stack[depth - 2];
        if target > (caller.top - caller.base) as usize {
            return Ok(Some(FastExit::Fallback));
        }
        {
            let (frames, values) = stack.frames_and_values();
            frames[depth - 2]
                .open_buffer()
                .extend_from_slice(&values[source..source + count]);
        }
        self.retire_top_frame(stack);
        self.restore_caller_hook_line(stack);
        Ok(None)
    }

    /// Returns `count` values from register `a` of a metamethod's
    /// activation into the operator or index continuation that called it,
    /// as the general path's `finish_scalar` and `complete_index` do: the
    /// first value, or its truth for a comparison, goes to the
    /// continuation's destination register. Other continuations, and
    /// callers with interior state (whose captures that path syncs around
    /// the write), fall back.
    #[inline(never)]
    fn return_to_continuation(
        &mut self,
        stack: &mut CallStack,
        a: usize,
        count: usize,
    ) -> Option<FastExit> {
        let depth = stack.len();
        let frame = &stack[depth - 1];
        // `None` for a `__newindex` call, whose result is discarded.
        let plain_operator = matches!(frame.caller, Some(ReturnTarget::Operator(_)));
        let (destination, comparison) = match &frame.caller {
            // A test's result picks a branch; the general path takes it.
            Some(ReturnTarget::Scalar(continuation)) if continuation.test.is_some() => {
                return Some(FastExit::Fallback);
            }
            Some(ReturnTarget::Scalar(continuation)) => (
                Some(continuation.destination),
                (continuation.kind == ScalarKind::Comparison).then_some(continuation.invert),
            ),
            Some(ReturnTarget::Operator(operator)) => match operator.mode {
                OperatorMode::Value => (Some(operator.destination), None),
                OperatorMode::Compare { invert } => (Some(operator.destination), Some(invert)),
                OperatorMode::Branch { expected, invert } => {
                    return self.return_to_branch(stack, a, count, expected, invert);
                }
            },
            // The handler's result is the concatenation's value once no
            // operands remain to its left.
            Some(ReturnTarget::ConcatRange(continuation)) if continuation.remaining.is_empty() => {
                (Some(continuation.destination), None)
            }
            Some(ReturnTarget::Index(continuation)) => match continuation.as_ref() {
                IndexContinuation {
                    action: IndexAction::Get { destination },
                    completion: IndexCompletion::Frame,
                    ..
                } => (Some(*destination), None),
                IndexContinuation {
                    action: IndexAction::Set { .. },
                    completion: IndexCompletion::Frame,
                    ..
                } => (None, None),
                _ => return Some(FastExit::Fallback),
            },
            Some(ReturnTarget::IndexResult(destination)) => (Some(*destination), None),
            Some(ReturnTarget::SetMetamethod) => (None, None),
            _ => return Some(FastExit::Fallback),
        };
        let caller = &stack[depth - 2];
        if frame.extras.is_some() || caller.extras.is_some() {
            return Some(FastExit::Fallback);
        }
        if let Some(destination) = destination {
            let destination = usize::from(destination);
            if a + count > (frame.top - frame.base) as usize
                || destination >= (caller.top - caller.base) as usize
            {
                return Some(FastExit::Fallback);
            }
            let source = frame.base as usize + a;
            let destination = caller.base as usize + destination;
            let (_, values) = stack.frames_and_values();
            let (below, returned) = values.split_at_mut(source);
            let slot = &mut below[destination];
            let first = returned[..count].first();
            match comparison {
                Some(invert) => {
                    let truth = first.is_some_and(Value::is_truthy);
                    set_register(slot, Value::Boolean(truth != invert));
                }
                None => match first {
                    Some(value) => store_clone(slot, value),
                    None => set_register(slot, Value::Nil),
                },
            }
        }
        // Continuations are boxed; release this one now rather than
        // leaving it in the retired record. A plain operator target owns
        // nothing, and the record stays reusable as a direct caller does.
        stack[depth - 1].clear_plain_mark();
        stack[depth - 1].caller = if plain_operator {
            Some(ReturnTarget::Direct(CallReturn {
                base: 0,
                results: 0,
            }))
        } else {
            None
        };
        self.retire_top_frame(stack);
        self.restore_caller_hook_line(stack);
        None
    }

    /// [`Self::return_to_continuation`] for the plain handler of a test
    /// instruction: the truth of the first result, negated when `invert`,
    /// picks the caller's branch as `finish_scalar` does.
    #[inline(never)]
    fn return_to_branch(
        &mut self,
        stack: &mut CallStack,
        a: usize,
        count: usize,
        expected: bool,
        invert: bool,
    ) -> Option<FastExit> {
        let depth = stack.len();
        let frame = &stack[depth - 1];
        let caller = &stack[depth - 2];
        if frame.extras.is_some()
            || caller.extras.is_some()
            || a + count > (frame.top - frame.base) as usize
        {
            return Some(FastExit::Fallback);
        }
        let source = frame.base as usize + a;
        let first_truth = count != 0 && stack.frames_and_values().1[source].is_truthy();
        let caller = &mut stack[depth - 2];
        if crate::vm::instructions::branch_after_test(caller, first_truth != invert, expected)
            .is_err()
        {
            // The general path reports the broken branch word.
            return Some(FastExit::Fallback);
        }
        stack[depth - 1].caller = Some(ReturnTarget::Direct(CallReturn {
            base: 0,
            results: 0,
        }));
        self.retire_top_frame(stack);
        self.restore_caller_hook_line(stack);
        None
    }

    /// Calls the native that the running activation's `Call` or `CallOne`
    /// names, for the callers that leave the loop's registers first. The loop
    /// itself calls natives in place through
    /// [`Self::call_native_in_window`]. The outer `None` declines before
    /// anything changes. Otherwise the result is what
    /// [`Self::switch_activation`] returns.
    // The three cases are the fast-path protocol: decline, complete, or an exit.
    #[allow(clippy::option_option)]
    #[inline(never)]
    fn native_call_activation(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<Option<Option<FastExit>>> {
        // An open result count has its own path; the window one declines it.
        if instruction.c() == 0 && opcode_byte(instruction) == op::CALL {
            return self.native_open_call(stack, instruction, false);
        }
        let depth = stack.len();
        let index = depth - 1;
        // A host callback: the window path would only decline it.
        if let Some(native) = self.host_callee(stack, instruction) {
            return self
                .host_call_activation(stack, instruction, native)
                .map(Some);
        }
        let (frames, values) = stack.frames_and_values();
        let frame = &mut frames[index];
        let window = (frame.top - frame.base) as usize;
        let registers = register_window(values, frame.base as usize);
        let step = if opcode_byte(instruction) == op::CALL_OPEN {
            self.open_native_in_window(registers, frame, instruction, None, window)
        } else {
            self.call_native_in_window(registers, instruction, window, depth, false)
        };
        let step = match step {
            Ok(NativeStep::Declined) => return Ok(None),
            Ok(step) => step,
            Err(error) => {
                // The native ran with the program counter past its call, as
                // in the dispatch loop.
                let pc = frame.pc;
                frame.last_interpreted_pc = crate::vm::frame::LastPc::some(pc);
                frame.pc = pc + 1;
                let (frame, parent) = frame_and_parent(frames, index);
                return Err(self.native_error(frame, parent, error));
            }
        };
        let pc = frame.pc;
        frame.last_interpreted_pc = crate::vm::frame::LastPc::some(pc);
        frame.pc = pc + 1;
        self.after_native_step(stack, step)
    }

    /// The host callback a fixed `Call` or `CallOne` of the running frame
    /// calls, when the fast loop may run it (see
    /// [`Self::host_call_in_fast_loop`]). The window checks of the caller
    /// keep the operands inside the frame.
    #[inline]
    fn host_callee(
        &self,
        stack: &CallStack,
        instruction: Instruction,
    ) -> Option<NativeFunctionHandle> {
        let (function, argument_count) = match opcode_byte(instruction) {
            op::CALL if instruction.c() != 0 => {
                (usize::from(instruction.a()), usize::from(instruction.b()))
            }
            op::CALL_ONE => (usize::from(instruction.b()), 1),
            _ => return None,
        };
        let frame = stack.at(stack.len() - 1);
        match frame.registers.get(function) {
            Some(&Value::NativeFunction(native))
                if self.host_call_in_fast_loop(native, argument_count) =>
            {
                Some(native)
            }
            _ => None,
        }
    }

    /// Runs the host callback `native` for the fixed `Call` or `CallOne`
    /// `instruction` of the running frame without leaving the fast loop, as
    /// the dispatch loop runs it. The caller has checked
    /// [`Self::host_call_in_fast_loop`].
    #[inline(never)]
    fn host_call_activation(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
        native: NativeFunctionHandle,
    ) -> VmResult<Option<FastExit>> {
        let depth = stack.len();
        let one = opcode_byte(instruction) == op::CALL_ONE;
        let (call, arguments) = if one {
            (
                Instruction::call(instruction.a(), 1, 1),
                (usize::from(instruction.c()), 1),
            )
        } else {
            (
                instruction,
                (
                    usize::from(instruction.a()) + 1,
                    usize::from(instruction.b()),
                ),
            )
        };
        if depth >= MAX_LUA_CALL_FRAMES
            || arguments.0 + arguments.1 > stack.at(depth - 1).registers.len()
        {
            return Ok(Some(FastExit::Fallback));
        }
        let exit = self.fast_native_step(stack, depth, |vm, stack| {
            vm.call_host_fixed(stack, native, call, arguments, one)
        })?;
        // A flag raised while the callback ran must stop a run of calls
        // that never jumps backwards; the general path, past the call,
        // delivers it. So must a hook the callback installed (there was none
        // before it): the fast loop runs no hook, and the dispatch loop fires
        // it from the instruction after the call, as stock Lua does.
        if exit.is_none() && (self.interrupt_pending() || !self.debug_hooks.is_empty()) {
            return Ok(Some(FastExit::Fallback));
        }
        Ok(exit)
    }

    /// `return host(...)`: the fixed `TailCall` `instruction` at the running
    /// frame's pc of the host callback `native`, run in place and its
    /// results returned without leaving the fast loop. The dispatch loop
    /// takes the tail call as a call with open results and then the
    /// `ReturnOpen` after it, which left the loop twice. Luanti's content id
    /// cache (`return old_get_content_id(name)` in an `__index`) makes one
    /// for every lookup. The caller has stored the loop state and checked
    /// [`Self::host_call_in_fast_loop`].
    ///
    /// The frame then stands past the call with the results open, as the
    /// dispatch loop leaves it. A flag the callback raised or a hook it
    /// installed leaves the fast loop there.
    #[inline(never)]
    fn host_tail_return(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
        native: NativeFunctionHandle,
    ) -> FastStep {
        let depth = stack.len();
        let a = usize::from(instruction.a());
        let arguments = (a + 1, usize::from(instruction.b()));
        if depth >= MAX_LUA_CALL_FRAMES
            || arguments.0 + arguments.1 > stack.at(depth - 1).registers.len()
        {
            return Ok(Some(FastExit::Fallback));
        }
        let exit = self
            .fast_native_step(stack, depth, |vm, stack| {
                vm.call_host_open(stack, native, arguments)
            })
            .map_err(Box::new)?;
        if exit.is_some() || self.interrupt_pending() || !self.debug_hooks.is_empty() {
            return Ok(Some(exit.unwrap_or(FastExit::Fallback)));
        }
        // The frame stands at the `ReturnOpen` after the call. When it
        // returns just the results, a fixed return of them from `a` returns
        // the same values, and the plain and handler returns take that one in
        // place, where a `ReturnOpen` to an `__index` miss needs the dispatch
        // loop. Every other case resumes at the `ReturnOpen` itself.
        let mut frame = stack.at_mut(depth - 1);
        let next = frame.chunk.code.get(frame.pc).copied();
        let count = frame.open_values().map_or(0, <[Value]>::len);
        let Some(next) = next else {
            return Ok(None);
        };
        let Ok(encoded) = u8::try_from(count) else {
            return Ok(None);
        };
        if opcode_byte(next) != op::RETURN_OPEN
            || next.a() != instruction.a()
            || next.b() != 0
            || count == 0
            || a + count > frame.registers.len()
        {
            return Ok(None);
        }
        let mut results = frame.take_open_results();
        for (slot, value) in frame.registers[a..a + count]
            .iter_mut()
            .zip(results.drain(..))
        {
            *slot = value;
        }
        let fixed = Instruction::abc(OpCode::Return, instruction.a(), encoded, 0);
        if self.pop_plain_return(stack, fixed) || self.return_handler_fast(stack, fixed) {
            self.runtime.recycle_result_vec(results);
            return Ok(None);
        }
        // Neither took it, and neither changed anything: the results go
        // back to the frame's open results for the `ReturnOpen`.
        let mut frame = stack.at_mut(depth - 1);
        results.extend(
            frame.registers[a..a + count]
                .iter_mut()
                .map(|slot| std::mem::replace(slot, Value::Nil)),
        );
        frame.set_open_results(results);
        Ok(None)
    }

    /// Whether a native trace may make a host call itself right now: no
    /// coroutine (whose callback may yield), no hook of any kind, and no
    /// recording.
    pub(crate) fn vmcall_ready(&self) -> bool {
        self.current_coroutine.is_none()
            && !self.hooks_observe_calls
            && !self.jit.trace.recording()
            && self.debug_hooks.is_empty()
    }

    /// Calls the host callback that the `Call` or `CallOne` at `pc` of the
    /// top frame names, for a native trace in the middle of its run. The
    /// trace has written its state back to the frame (the restore of the
    /// call's `Before` snapshot), so the call is the interpreter's own: the
    /// sequence of `dispatch_call_value` for a native, with the arguments
    /// copied out of their registers. `shape` turns the results of the call
    /// into the words the trace takes, when they are of the kinds it
    /// recorded.
    ///
    /// An error is handed over directly: `vmcall_error` holds it as the
    /// dispatch loop would raise it, and the frame stands after the call, so
    /// nothing runs twice and no hook sees the call again.
    ///
    /// `Declined` changed nothing but the frame's pc, which the exit puts
    /// right: a hook, a coroutine, a callee that is no host callback, or an
    /// operand outside the window. `Answered` ran the call, wrote its
    /// results to the frame and gives their words. `Missed` ran the call
    /// and the trace must leave with the frame as the call left it (an error,
    /// other results than the recording saw, a hook installed, a local
    /// written, a flush requested): `vmcall_landed` says so, and the exit
    /// takes the frame in place of a restore.
    pub(crate) fn host_call_for_trace(
        &mut self,
        stack: &mut CallStack,
        index: usize,
        pc: usize,
        shape: impl FnOnce(&[Value], bool) -> Option<[i64; 3]>,
    ) -> TraceHostCall {
        let depth = stack.len();
        if index + 1 != depth || !self.vmcall_ready() || depth >= MAX_LUA_CALL_FRAMES {
            return TraceHostCall::Declined;
        }
        let Some(&instruction) = stack[index].chunk.code.get(pc) else {
            return TraceHostCall::Declined;
        };
        let call_one = opcode_byte(instruction) == op::CALL_ONE;
        // A tail call of a host function is the interpreter's call with open
        // results, which the `ReturnOpen` after it takes.
        let tail = opcode_byte(instruction) == op::TAIL_CALL;
        if (opcode_byte(instruction) == op::CALL && instruction.c() == 0)
            || !(call_one || tail || opcode_byte(instruction) == op::CALL)
        {
            return TraceHostCall::Declined;
        }
        let a = usize::from(instruction.a());
        // A `CallOne` names its callee in `b` and its one argument in `c`,
        // and stores its result at `a`, which a `Call` of the same shape
        // writes.
        let (callee, first, count, results, call) = if call_one {
            (
                usize::from(instruction.b()),
                usize::from(instruction.c()),
                1,
                1,
                Instruction::call(instruction.a(), 1, 1),
            )
        } else {
            (
                a,
                a + 1,
                usize::from(instruction.b()),
                usize::from(instruction.c()),
                instruction,
            )
        };
        let (native, mut arguments, direct) = {
            let frame = &stack[index];
            let window = (frame.top - frame.base) as usize;
            if a + results > window || first + count > window || callee >= window {
                return TraceHostCall::Declined;
            }
            let view = stack.at(index);
            let native = match view.registers.get(callee) {
                Some(Value::NativeFunction(native))
                    if self.runtime.host_native_token(*native).is_some() =>
                {
                    *native
                }
                _ => return TraceHostCall::Declined,
            };
            // A fixed call of a function with a direct entry reads its
            // arguments and stores its results in the registers, as the
            // fast loop's call does (`Vm::call_host_fixed`): `vmcall_ready`
            // has checked what `Vm::host_call_in_fast_loop` asks for.
            let direct = if tail {
                None
            } else {
                self.host_direct_entry(native, count)
            };
            let mut arguments = std::mem::take(&mut self.vmcall_arguments);
            if direct.is_none() {
                arguments.extend_from_slice(&view.registers[first..first + count]);
            }
            (native, arguments, direct)
        };
        let writes = self.local_writes;
        {
            let frame = &mut stack[index];
            frame.last_interpreted_pc = crate::vm::frame::LastPc::some(pc);
            frame.pc = pc + 1;
        }
        // The trace is in the middle of its run: native execution stays off
        // for whatever the callback runs, and a flush waits for the run.
        let held = self.runtime.suspend_jit();
        self.active_native_runs += 1;
        let result = match direct {
            Some(entry) => self
                .call_host_direct(stack, native, entry, (first, count), (a, results))
                .map(|()| None),
            None => self
                .call_native_yieldable(stack, native, &arguments, false)
                .map(Some),
        };
        self.active_native_runs -= 1;
        self.runtime.restore_jit(held);
        arguments.clear();
        self.vmcall_arguments = arguments;
        self.jit.statistics.host_trace_calls += 1;
        let values = match result {
            Ok(values) => values,
            Err(error) => {
                // The frame stands after the call, as the dispatch loop
                // leaves it for the error: the entry raises it from here.
                let error = self.native_error(
                    &stack[index],
                    index.checked_sub(1).map(|i| &stack[i]),
                    error,
                );
                self.vmcall_error = Some(error);
                self.jit.trace.fail_vmcall_landed(pc + 1, pc);
                return TraceHostCall::Missed;
            }
        };
        debug_assert!(self.pending_host_yield.is_none());
        let (words, written) = match values {
            // The results stand in the registers, nil past the last one,
            // which is how `shape` reads a result the call did not give.
            None => {
                stack[index].clear_open_results();
                let frame = stack.at(index);
                (shape(&frame.registers[a..a + results], false), Ok(()))
            }
            Some(values) => {
                let words = shape(&values, tail);
                let written = if tail {
                    stack.at_mut(index).set_open_results(values);
                    Ok(())
                } else {
                    write_call_results(&mut stack.at_mut(index), call, values)
                };
                (words, written)
            }
        };
        if written.is_err() {
            // Unreachable for a call whose window was checked above. The
            // call ran, so the exit must not restore and run it again.
            debug_assert!(false, "host call results outside a checked window");
            self.jit.trace.set_vmcall_landed(pc + 1, pc);
            return TraceHostCall::Missed;
        }
        // A call hook the function installed sees the call once it returned,
        // as the dispatch loop reports it (`Vm::observe_native_call`).
        if !tail
            && self.hooks_observe_calls
            && let Err(error) =
                self.observe_native_call(stack, Value::NativeFunction(native), count)
        {
            self.vmcall_error = Some(error);
            self.jit.trace.fail_vmcall_landed(pc + 1, pc);
            return TraceHostCall::Missed;
        }
        // The call is complete and the frame stands after it. A trace that
        // cannot go on (other results than it recorded, a hook installed by
        // the call, a flush requested) leaves with the frame as it is.
        // A write to a local may have changed a register the trace holds.
        let hooked = !self.debug_hooks.is_empty()
            || self.native_checks().is_some()
            || self.local_writes != writes;
        let Some(words) = words.filter(|_| !hooked) else {
            self.jit.trace.set_vmcall_landed(pc + 1, pc);
            return TraceHostCall::Missed;
        };
        TraceHostCall::Answered(words)
    }

    /// What a native call run inside a register window leaves for its
    /// caller to do.
    // The three cases are the fast-path protocol: decline, complete, or an exit.
    #[allow(clippy::option_option)]
    fn after_native_step(
        &mut self,
        stack: &mut CallStack,
        step: NativeStep,
    ) -> VmResult<Option<Option<FastExit>>> {
        Ok(Some(match step {
            NativeStep::Done => None,
            NativeStep::Safepoint => Some(FastExit::Safepoint),
            NativeStep::Backedge => Some(FastExit::Backedge),
            NativeStep::Hooked(native, argument_count) => {
                self.native_hooked(stack, native, argument_count)?
            }
            NativeStep::Declined => return Ok(None),
            // Only the loop's own open-results call passes the instruction
            // after it, which fusing needs.
            NativeStep::Fused | NativeStep::FusedExit(_) => {
                unreachable!("a fused native pair outside the fast loop")
            }
        }))
    }

    /// Reports a native call to the hook the native may have set, then ends
    /// it as every native call ends.
    #[cold]
    #[inline(never)]
    fn native_hooked(
        &mut self,
        stack: &mut CallStack,
        native: NativeFunctionHandle,
        argument_count: usize,
    ) -> VmResult<Option<FastExit>> {
        self.observe_native_call(stack, Value::NativeFunction(native), argument_count)?;
        Ok(self.native_checks())
    }

    /// Runs a native that returns one value as an open-results call
    /// without leaving the loop: the `Call` with open results (`c == 0`)
    /// whose value becomes the frame's open results, and the `CallOpen`
    /// whose last arguments they are. Mineclonia's 64-bit helpers write every
    /// half as `tonumber (band (value, 0xffffffffull))`, one of each; through
    /// the general dispatch the pair cost about twice its two calls made
    /// apart (900 against 480 instructions on a 1M-call loop).
    ///
    /// When the open-results `Call` is followed by such a `CallOpen`
    /// (`following`), the second call takes the first's value as its last
    /// argument directly and both complete: [`NativeStep::Fused`], with no
    /// open results stored. The second call's error sets
    /// `fused_open_error`, so the caller names its site.
    ///
    /// [`NativeStep::Declined`] means nothing changed: another callee, a
    /// multi-result native, an observing hook, a `CallOpen` with open results
    /// of its own or more staged arguments than [`OPEN_NATIVE_ARGUMENTS`].
    /// The caller has advanced nothing; on an error it names it after the
    /// call site, as for [`Self::call_native_in_window`].
    #[inline(never)]
    fn open_native_in_window(
        &mut self,
        registers: &mut [Value; REGISTER_WINDOW],
        frame: *mut Frame,
        instruction: Instruction,
        following: Option<Instruction>,
        window: usize,
    ) -> Result<NativeStep, RuntimeError> {
        if self.hooks_observe_calls {
            return Ok(NativeStep::Declined);
        }
        let a = usize::from(instruction.a());
        let fixed = usize::from(instruction.b());
        let opening = opcode_byte(instruction) == op::CALL;
        let results = usize::from(instruction.c());
        // SAFETY: `frame` is the live top record, whose window `registers`
        // is; the record and the window do not overlap.
        #[allow(unsafe_code)]
        let record = unsafe { &mut *frame };
        let Value::NativeFunction(native) = registers[a] else {
            return Ok(NativeStep::Declined);
        };
        let plan = self.runtime.native_plan(native);
        if !matches!(
            plan,
            NativePlan::Scalar(_) | NativePlan::Unary(..) | NativePlan::Binary(..)
        ) || a + 1 + fixed > window
        {
            return Ok(NativeStep::Declined);
        }
        let value = if opening {
            single_native_value(
                &mut self.runtime,
                native,
                plan,
                &registers[a + 1..a + 1 + fixed],
            )
        } else {
            // A `CallOpen` that leaves open results of its own goes to the
            // general path, which keeps the buffer for them.
            if results == 0 || a + results > window {
                return Ok(NativeStep::Declined);
            }
            let Some(open) = record.open_values() else {
                return Ok(NativeStep::Declined);
            };
            if fixed == 0 {
                single_native_value(&mut self.runtime, native, plan, open)
            } else {
                staged_native_value(
                    &mut self.runtime,
                    native,
                    plan,
                    &registers[a + 1..a + 1 + fixed],
                    open,
                )
            }
        };
        let (value, pure) = match value {
            Ok(Some(value)) => value,
            Ok(None) => return Ok(NativeStep::Declined),
            Err(error) => return Err(error),
        };
        #[cfg(test)]
        OPEN_NATIVES_IN_PLACE.with(|count| count.set(count.get() + 1));
        if opening {
            if let Some(following) = following
                && opcode_byte(following) == op::CALL_OPEN
                && (pure || (self.native_checks().is_none() && !self.interrupt_pending()))
            {
                match self.fused_open_native(registers, following, &value, window) {
                    Ok(Some((outer, outer_pure))) => {
                        let outer_a = usize::from(following.a());
                        let outer_results = usize::from(following.c());
                        record.clear_open_results();
                        set_register(&mut registers[outer_a], outer);
                        for slot in &mut registers[outer_a + 1..outer_a + outer_results] {
                            set_register(slot, Value::Nil);
                        }
                        if outer_pure {
                            return Ok(NativeStep::Fused);
                        }
                        return Ok(match self.native_checks() {
                            None => NativeStep::Fused,
                            Some(FastExit::Backedge) => NativeStep::FusedExit(true),
                            Some(_) => NativeStep::FusedExit(false),
                        });
                    }
                    Ok(None) => {}
                    Err(error) => {
                        self.fused_open_error = true;
                        return Err(error);
                    }
                }
            }
            record.open_buffer().push(value);
        } else {
            record.clear_open_results();
            set_register(&mut registers[a], value);
            for slot in &mut registers[a + 1..a + results] {
                set_register(slot, Value::Nil);
            }
        }
        if pure {
            return Ok(NativeStep::Done);
        }
        Ok(self.native_step(None))
    }

    /// The value of `outer`, a `CallOpen` of a native that returns one value,
    /// whose open results are the single `value`, as
    /// [`Self::open_native_in_window`] computes it for a `CallOpen`. `None`
    /// declines without effect, under the same conditions.
    #[inline(always)]
    fn fused_open_native(
        &mut self,
        registers: &[Value; REGISTER_WINDOW],
        outer: Instruction,
        value: &Value,
        window: usize,
    ) -> Result<Option<(Value, bool)>, RuntimeError> {
        let a = usize::from(outer.a());
        let fixed = usize::from(outer.b());
        let results = usize::from(outer.c());
        let Value::NativeFunction(native) = registers[a] else {
            return Ok(None);
        };
        let plan = self.runtime.native_plan(native);
        if !matches!(
            plan,
            NativePlan::Scalar(_) | NativePlan::Unary(..) | NativePlan::Binary(..)
        ) || results == 0
            || a + 1 + fixed > window
            || a + results > window
        {
            return Ok(None);
        }
        let value = if fixed == 0 {
            single_native_value(&mut self.runtime, native, plan, std::slice::from_ref(value))
        } else {
            staged_native_value(
                &mut self.runtime,
                native,
                plan,
                &registers[a + 1..a + 1 + fixed],
                std::slice::from_ref(value),
            )
        };
        #[cfg(test)]
        if matches!(value, Ok(Some(_))) {
            FUSED_OPEN_NATIVES.with(|count| count.set(count.get() + 1));
        }
        value
    }

    /// Calls the native that `instruction` names, a `Call` or `CallOne` in a
    /// frame whose registers are `registers` and whose window is `window`
    /// registers wide, and stores its results, as the general path's
    /// `try_call_native_scalar_range` (or, for a multi-result native,
    /// `try_call_native_multi_borrowed`) does. It needs no frame views: the
    /// caller owns the program counter. [`NativeStep::Declined`] means that
    /// nothing changed: the callee is not a native, or a host callback needs
    /// the VM's roots. Validation keeps the call's operands inside the
    /// window, which only a debug assertion checks. On an error the caller
    /// names it after the call site.
    ///
    /// A pure numeric native on plain numbers is called directly, with no
    /// result to move and no collector or flush check after it. Every other
    /// single-result native runs through its callback here. The natives that
    /// return vectors take [`Self::multiple_native`] and [`Self::other_native`].
    #[inline(never)]
    fn call_native_in_window(
        &mut self,
        registers: &mut [Value; REGISTER_WINDOW],
        instruction: Instruction,
        window: usize,
        depth: usize,
        single: bool,
    ) -> Result<NativeStep, RuntimeError> {
        let window = window.min(REGISTER_WINDOW);
        let a = usize::from(instruction.a());
        // Both call shapes write their results from register `a`.
        let (function, arguments, argument_count, results) = if opcode_byte(instruction) == op::CALL
        {
            (
                a,
                a + 1,
                usize::from(instruction.b()),
                usize::from(instruction.c()),
            )
        } else {
            (
                usize::from(instruction.b()),
                usize::from(instruction.c()),
                1,
                1,
            )
        };
        // Validation keeps every operand of a call inside the frame's window
        // (`a + b` and `a + c - 1` are registers), as for every other
        // instruction of the loop.
        debug_assert!(
            (arguments + argument_count)
                .max(a + results)
                .max(function + 1)
                <= window
        );
        if results == 0 || depth >= MAX_LUA_CALL_FRAMES {
            return Ok(NativeStep::Declined);
        }
        let Value::NativeFunction(native) = registers[function] else {
            // `pairs`, `ipairs` and `next` of a plain table.
            if let Value::BuiltinFunction(
                builtin
                @ (BuiltinFunction::Pairs | BuiltinFunction::IPairs | BuiltinFunction::Next),
            ) = registers[function]
            {
                return Ok(
                    if !single
                        && self.iteration_builtin_in_window(
                            registers,
                            builtin,
                            (a, results, arguments, argument_count),
                        )
                    {
                        NativeStep::Done
                    } else {
                        NativeStep::Declined
                    },
                );
            }
            // `tostring`, which has no table entry. It declines for anything
            // but a number.
            if argument_count != 0
                && !single
                && matches!(
                    registers[function],
                    Value::BuiltinFunction(BuiltinFunction::ToString)
                )
            {
                // A small integer's cached text allocates nothing, so no
                // collector check follows it.
                if self.tostring_cached(registers, instruction) {
                    return Ok(NativeStep::Done);
                }
                return self.tostring_in_window(registers, (a, results, arguments));
            }
            // `string.gsub` of a string with a literal replacement.
            if !single
                && (3..=4).contains(&argument_count)
                && matches!(
                    registers[function],
                    Value::BuiltinFunction(BuiltinFunction::StringGsub)
                )
                && !self.hooks_observe_calls
                && self.gsub_literal_in_window(registers, (a, results, arguments, argument_count))
            {
                return Ok(self.native_step(None));
            }
            return Ok(NativeStep::Declined);
        };
        // One table read tells the call's form.
        let callback = match self.runtime.native_plan(native) {
            // A pure numeric native on plain numbers cannot fail, allocate,
            // or run Lua: no collector or flush check follows it. A debug
            // hook wants the call reported, so it takes the callback.
            NativePlan::Unary(callback, unary) => {
                if results == 1
                    && argument_count == 1
                    && let Value::Number(x) = registers[arguments]
                    && !self.hooks_observe_calls
                {
                    set_register(&mut registers[a], Value::Number(unary(x)));
                    return Ok(NativeStep::Done);
                }
                callback
            }
            NativePlan::Binary(callback, binary) => {
                if results == 1
                    && argument_count == 2
                    && let (&Value::Number(x), &Value::Number(y)) =
                        (&registers[arguments], &registers[arguments + 1])
                    && !self.hooks_observe_calls
                {
                    set_register(&mut registers[a], Value::Number(binary(x, y)));
                    return Ok(NativeStep::Done);
                }
                // A `bit` operation on a 64-bit integer: as pure, with no
                // callback and no result to move.
                if results == 1
                    && argument_count == 2
                    && !self.hooks_observe_calls
                    && let Some(wide) = self.runtime.native_wide_binary(native)
                    && let Some(value) = wide(&registers[arguments], &registers[arguments + 1])
                {
                    set_register(&mut registers[a], value);
                    return Ok(NativeStep::Done);
                }
                callback
            }
            NativePlan::Scalar(callback) => {
                // `type(value)` stores the value's shared type name in place:
                // no callback, no result to move, nothing allocated.
                // Registration code calls it for every key it copies. A call
                // hook takes the callback, which reports the call; no
                // argument raises its error there.
                if argument_count != 0
                    && self
                        .runtime
                        .is_direct_native(native, DirectNativeKind::Type)
                    && !self.hooks_observe_calls
                {
                    let name = self.runtime.type_name_ref(&registers[arguments]).clone();
                    set_register(&mut registers[a], Value::String(name));
                    for slot in &mut registers[a + 1..a + results] {
                        set_register(slot, Value::Nil);
                    }
                    return Ok(NativeStep::Done);
                }
                // `setmetatable(table, table or nil)` runs here. A call hook,
                // another argument shape, or a bad argument takes the
                // callback, which raises the error.
                if argument_count == 2
                    && self.runtime.is_setmetatable_native(native)
                    && !self.hooks_observe_calls
                    && let Value::Table(table) = registers[arguments]
                    && let Some(metatable) = match registers[arguments + 1] {
                        Value::Nil => Some(None),
                        Value::Table(metatable) => Some(Some(metatable)),
                        _ => None,
                    }
                {
                    self.runtime.set_table_metatable(table, metatable)?;
                    set_register(&mut registers[a], Value::Table(table));
                    for slot in &mut registers[a + 1..a + results] {
                        set_register(slot, Value::Nil);
                    }
                    return Ok(self.native_step(None));
                }
                // `math.floor(number)` runs here: it allocates nothing and
                // cannot fail, so no collector or flush check follows it.
                if argument_count == 1
                    && results == 1
                    && self
                        .runtime
                        .is_direct_native(native, DirectNativeKind::MathFloor)
                    && !self.hooks_observe_calls
                {
                    match registers[arguments] {
                        Value::Integer(value) => {
                            set_register(&mut registers[a], Value::Integer(value));
                            return Ok(NativeStep::Done);
                        }
                        Value::Number(value) => {
                            let floor = self.runtime.integral_float_value(value.floor());
                            set_register(&mut registers[a], floor);
                            return Ok(NativeStep::Done);
                        }
                        _ => {}
                    }
                }
                callback
            }
            NativePlan::Paired(..) | NativePlan::Multiple(_) | NativePlan::Other if single => {
                return Ok(NativeStep::Declined);
            }
            NativePlan::Paired(callback, paired) => {
                if results <= PATTERN_RESULTS_IN_PLACE
                    && (2..=4).contains(&argument_count)
                    && self
                        .runtime
                        .is_direct_native(native, DirectNativeKind::Find)
                    && self.pattern_in_place(
                        registers,
                        (a, results, arguments, argument_count),
                        true,
                    )
                {
                    return Ok(self.native_step(None));
                }
                let shape = pack_call_shape(a, results, arguments, argument_count);
                return self.paired_native(registers, shape, callback, paired);
            }
            NativePlan::Multiple(callback) => {
                // `assert` of a true value gives back its arguments: they
                // move down to the result registers, with no results vector
                // built, nothing allocated and nothing that can fail. A
                // false value, a call with no argument and a call a hook
                // watches take the callback.
                if argument_count != 0
                    && registers[arguments].is_truthy()
                    && self
                        .runtime
                        .is_direct_native(native, DirectNativeKind::Assert)
                    && !self.hooks_observe_calls
                {
                    // A `Call`'s arguments lie above its first result
                    // register, so copying upwards overwrites none before it
                    // is read. A `CallOne` copies its one value.
                    debug_assert!(arguments > a || results == 1);
                    for index in 0..results {
                        if index < argument_count {
                            copy_register(registers, a + index, arguments + index);
                        } else {
                            set_register(&mut registers[a + index], Value::Nil);
                        }
                    }
                    return Ok(NativeStep::Done);
                }
                // `string.byte` of a string and plain indices writes its
                // bytes into the result registers: no results vector.
                if (1..=BYTE_RESULTS_IN_PLACE).contains(&results)
                    && (1..=3).contains(&argument_count)
                    && self
                        .runtime
                        .is_direct_native(native, DirectNativeKind::StringByte)
                    && self.string_byte_in_place(registers, (a, results, arguments, argument_count))
                {
                    return Ok(self.native_step(None));
                }
                if results <= PATTERN_RESULTS_IN_PLACE
                    && (2..=3).contains(&argument_count)
                    && self
                        .runtime
                        .is_direct_native(native, DirectNativeKind::Match)
                    && self.pattern_in_place(
                        registers,
                        (a, results, arguments, argument_count),
                        false,
                    )
                {
                    return Ok(self.native_step(None));
                }
                let shape = pack_call_shape(a, results, arguments, argument_count);
                return self.multiple_native(registers, shape, callback);
            }
            // A host callback runs outside the window (see
            // `host_call_activation`).
            NativePlan::Other if self.is_host_function(native) => {
                return Ok(NativeStep::Declined);
            }
            NativePlan::Other => {
                let shape = pack_call_shape(a, results, arguments, argument_count);
                return self.other_native(registers, shape, native);
            }
        };
        let mut returned = callback(
            &mut self.runtime,
            &registers[arguments..arguments + argument_count],
        );
        if !store_result(&mut registers[a], &mut returned) {
            let error = returned.expect_err("a result that was not stored is an error");
            // A native that declined a table with metamethods took no
            // effect; the general path runs it with the VM.
            if matches!(error, RuntimeError::Redispatch) {
                return Ok(NativeStep::Declined);
            }
            return Err(error);
        }
        // The value is stored or moved out: nothing is left to drop.
        std::mem::forget(returned);
        for slot in &mut registers[a + 1..a + results] {
            set_register(slot, Value::Nil);
        }
        Ok(self.native_step(if single {
            None
        } else {
            Some((native, argument_count))
        }))
    }

    /// `tostring(n)` for a small non-negative integer when the profile prints
    /// integral numbers bare: stores the shared cached string. Returns false,
    /// having changed nothing, for any other call, which takes the general
    /// native path.
    #[inline(always)]
    fn tostring_cached(
        &self,
        registers: &mut [Value; REGISTER_WINDOW],
        instruction: Instruction,
    ) -> bool {
        let a = usize::from(instruction.a());
        let (arguments, argument_count, results) = if opcode_byte(instruction) == op::CALL {
            (a + 1, instruction.b(), instruction.c())
        } else {
            (usize::from(instruction.c()), 1, 1)
        };
        // `results == 1` keeps the extra result registers out of this path.
        if results != 1 || argument_count == 0 {
            return false;
        }
        let Value::Number(number) = registers[arguments] else {
            return false;
        };
        // Non-negative floats order as their bit patterns do, and every
        // negative number, `-0`, and NaN has a larger pattern than 256.
        if number.to_bits() >= 256.0_f64.to_bits()
            || self.runtime.number_format().appends_float_suffix()
            || self.runtime.number_metatable().is_some()
            || self.hooks_observe_calls
        {
            return false;
        }
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "below 256"
        )]
        let digits = number as u8;
        #[allow(clippy::float_cmp, reason = "an exact round trip tells an integer")]
        if f64::from(digits) != number {
            return false;
        }
        let Some(text) = self.runtime.small_integer_text(u64::from(digits)) else {
            return false;
        };
        set_register(&mut registers[a], Value::String(text));
        true
    }

    /// `pairs(t)`, `ipairs(t)` and `next(t [, k])` of a table, written
    /// straight into the call's result registers as the general path's
    /// builtins return them. `false` changes nothing and leaves the call to
    /// that path: another argument, a call hook, `pairs` of a table with a
    /// metatable where `__pairs` exists (Lua 5.2 and later), or a `next`
    /// continuation the traversal cursor cannot settle (which includes the
    /// invalid key that raises an error). Nothing here allocates, so no
    /// collector check follows.
    #[inline(never)]
    fn iteration_builtin_in_window(
        &mut self,
        registers: &mut [Value; REGISTER_WINDOW],
        builtin: BuiltinFunction,
        (a, results, arguments, argument_count): (usize, usize, usize, usize),
    ) -> bool {
        if argument_count == 0 || self.hooks_observe_calls {
            return false;
        }
        let Value::Table(table) = registers[arguments] else {
            return false;
        };
        let (iterator, control) = match builtin {
            BuiltinFunction::Pairs => {
                if !self.compatibility.is_lua51_family()
                    && !matches!(self.runtime.raw_metatable(&Value::Table(table)), Ok(None))
                {
                    return false;
                }
                (Value::BuiltinFunction(BuiltinFunction::Next), Value::Nil)
            }
            BuiltinFunction::IPairs => (
                Value::BuiltinFunction(BuiltinFunction::IPairsIterator),
                crate::vm::arith::profile_integer_value(self.compatibility, 0),
            ),
            _ => {
                let previous = if argument_count >= 2 {
                    &registers[arguments + 1]
                } else {
                    &Value::Nil
                };
                let Some((key, value)) = self.runtime.table_next_fast(table, previous) else {
                    return false;
                };
                set_register(&mut registers[a], key);
                if results >= 2 {
                    set_register(&mut registers[a + 1], value);
                }
                for slot in registers.iter_mut().take(a + results).skip(a + 2) {
                    set_register(slot, Value::Nil);
                }
                return true;
            }
        };
        set_register(&mut registers[a], iterator);
        if results >= 2 {
            set_register(&mut registers[a + 1], Value::Table(table));
        }
        if results >= 3 {
            set_register(&mut registers[a + 2], control);
        }
        for slot in registers.iter_mut().take(a + results).skip(a + 3) {
            set_register(slot, Value::Nil);
        }
        true
    }

    /// `tostring` of a number whose type has no metatable, written straight
    /// into the call's result registers. Anything else, and any call a debug
    /// hook may observe, declines before changing anything and takes the
    /// general path.
    #[inline(always)]
    fn tostring_in_window(
        &mut self,
        registers: &mut [Value; REGISTER_WINDOW],
        (a, results, arguments): (usize, usize, usize),
    ) -> Result<NativeStep, RuntimeError> {
        let value = &registers[arguments];
        if !matches!(value, Value::Number(_) | Value::Integer(_))
            || self.runtime.number_metatable().is_some()
            || self.hooks_observe_calls
        {
            return Ok(NativeStep::Declined);
        }
        let text = match value {
            Value::Number(number) => self.runtime.number_format().format_string(*number),
            value => self.runtime.format_value(value)?,
        };
        set_register(&mut registers[a], Value::String(text));
        for slot in &mut registers[a + 1..a + results] {
            set_register(slot, Value::Nil);
        }
        Ok(self.native_step(None))
    }

    /// The error of a failed native call, named after its call site. Out of
    /// line: the call paths carry no formatting code.
    #[cold]
    #[inline(never)]
    fn native_error(&self, frame: &Frame, parent: Option<&Frame>, error: RuntimeError) -> VmError {
        call_argument_error(frame, parent, error, self.compatibility).into()
    }

    /// A multi-result native with a fixed-buffer form: its results go
    /// straight into the window, with no vector. A call the form declines
    /// takes the general callback.
    #[inline(never)]
    fn paired_native(
        &mut self,
        registers: &mut [Value; REGISTER_WINDOW],
        shape: u64,
        callback: NativeCallback,
        paired: NativePairCallback,
    ) -> Result<NativeStep, RuntimeError> {
        let [a, results, arguments, argument_count] = unpack_call_shape(shape);
        let mut pair = match paired(
            &mut self.runtime,
            &registers[arguments..arguments + argument_count],
        ) {
            Ok(Some(pair)) => pair,
            Ok(None) => return self.multiple_native(registers, shape, callback),
            Err(RuntimeError::Redispatch) => return Ok(NativeStep::Declined),
            Err(error) => return Err(error),
        };
        for (position, slot) in registers[a..a + results].iter_mut().enumerate() {
            set_register(slot, pair.take(position));
        }
        Ok(self.native_step(None))
    }

    /// `string.find(text, pattern, init)` (`find`) or `string.match(text,
    /// pattern, init)` for strings and a position that is an integer, a
    /// small exact float or left out, and a `find` that is not plain: the
    /// positions and captures go straight into the result registers, with no
    /// result vector. `false` before any write for any other call (a number
    /// argument, a position past the end, a malformed pattern, more captures
    /// than [`PATTERN_RESULTS_IN_PLACE`]), which the native judges and
    /// reports.
    // Exact comparison is required here (NaN semantics): a rounded comparison
    // would accept positions the native would reject.
    #[allow(clippy::float_cmp)]
    #[inline(never)]
    fn pattern_in_place(
        &mut self,
        registers: &mut [Value; REGISTER_WINDOW],
        (a, results, arguments, argument_count): (usize, usize, usize, usize),
        find: bool,
    ) -> bool {
        let given = &registers[arguments..arguments + argument_count];
        let (Value::String(source), Value::String(pattern)) = (&given[0], &given[1]) else {
            return false;
        };
        if find && given.get(3).is_some_and(Value::is_truthy) {
            return false;
        }
        let initial = match given.get(2) {
            None | Some(Value::Nil) => 1,
            Some(Value::Integer(value)) => match i32::try_from(*value) {
                Ok(small) => small as isize,
                Err(_) => return false,
            },
            Some(Value::Number(value)) => {
                #[allow(clippy::cast_possible_truncation)]
                let small = *value as i32;
                if f64::from(small) != *value {
                    return false;
                }
                small as isize
            }
            Some(_) => return false,
        };
        let Ok(length) = isize::try_from(source.len()) else {
            return false;
        };
        let mut initial = if initial < 0 {
            length.saturating_add(initial).saturating_add(1)
        } else {
            initial
        };
        // A start past the end is clamped by some profiles and fails in
        // others; the native knows which.
        if initial > length.saturating_add(1) {
            return false;
        }
        initial = initial.max(1);
        let Ok(initial) = usize::try_from(initial - 1) else {
            return false;
        };
        let Ok(pattern) = Pattern::shared(pattern.as_bytes(), false) else {
            return false;
        };
        let source = source.as_bytes();
        let found = match pattern.find(source, initial) {
            Ok(Some(found)) => found,
            // No match is one `nil`; an error is the native's to report.
            Ok(None) => {
                for slot in &mut registers[a..a + results] {
                    set_register(slot, Value::Nil);
                }
                return true;
            }
            Err(_) => return false,
        };
        let captures = found.captures();
        let leading = if find { 2 } else { 0 };
        // A `match` without captures answers the whole match.
        let count = if captures.is_empty() && !find {
            1
        } else {
            leading + captures.len()
        };
        if count > PATTERN_RESULTS_IN_PLACE {
            return false;
        }
        // The results may overlap the arguments the captures are cut from, so
        // they are built before the first write.
        let mut values = [const { Value::Nil }; PATTERN_RESULTS_IN_PLACE];
        if find {
            values[0] = self.runtime.integer_value(found.start() as i64 + 1);
            values[1] = self.runtime.integer_value(found.end() as i64);
        }
        if captures.is_empty() && !find {
            values[0] = Value::from(&source[found.start()..found.end()]);
        }
        for (slot, capture) in values[leading..].iter_mut().zip(captures) {
            *slot = if capture.is_position() {
                self.runtime.integer_value(capture.start() as i64 + 1)
            } else {
                Value::from(&source[capture.start()..capture.end()])
            };
        }
        for (position, slot) in registers[a..a + results].iter_mut().enumerate() {
            let value = if position < count {
                std::mem::replace(&mut values[position], Value::Nil)
            } else {
                Value::Nil
            };
            set_register(slot, value);
        }
        true
    }

    /// `string.gsub(text, pattern, replacement [, limit])` for three strings
    /// and a replacement without a `%`, with the limit left out, nil or an
    /// integer: the replacement text goes in as it is, so the call needs no
    /// continuation. `false` before any write for any other call, which
    /// takes the general path and its errors.
    #[inline(never)]
    fn gsub_literal_in_window(
        &mut self,
        registers: &mut [Value; REGISTER_WINDOW],
        (a, results, arguments, argument_count): (usize, usize, usize, usize),
    ) -> bool {
        let given = &registers[arguments..arguments + argument_count];
        let (Value::String(source), Value::String(pattern), Value::String(template)) =
            (&given[0], &given[1], &given[2])
        else {
            return false;
        };
        if template.as_bytes().contains(&b'%') {
            return false;
        }
        let maximum = match given.get(3) {
            None | Some(Value::Nil) => usize::MAX,
            Some(Value::Integer(limit)) => usize::try_from(*limit).unwrap_or(0),
            Some(_) => return false,
        };
        let Ok(pattern) = Pattern::shared(pattern.as_bytes(), false) else {
            return false;
        };
        let Some((output, count)) = gsub_literal(
            source.as_bytes(),
            &pattern,
            template.as_bytes(),
            (maximum, self.compatibility.is_lua53_or_later(), usize::MAX),
        ) else {
            return false;
        };
        let Ok(count) = i64::try_from(count) else {
            return false;
        };
        set_register(&mut registers[a], Value::bytes(output));
        if results > 1 {
            set_register(
                &mut registers[a + 1],
                profile_integer_value(self.compatibility, count),
            );
            for slot in &mut registers[a + 2..a + results] {
                set_register(slot, Value::Nil);
            }
        }
        true
    }

    /// `string.byte(text, i, j)` for a string and indices that are integers,
    /// small exact floats or left out, with at most [`BYTE_RESULTS_IN_PLACE`]
    /// results: the bytes go straight into the result registers, past the
    /// last byte nil. `false` before any write for any other call, which the
    /// native judges (a numeric string, an error, a wide index).
    #[inline(never)]
    fn string_byte_in_place(
        &mut self,
        registers: &mut [Value; REGISTER_WINDOW],
        (a, results, arguments, argument_count): (usize, usize, usize, usize),
    ) -> bool {
        // Exact comparison is required here (NaN semantics): the round trip
        // through `i32` must reproduce the value exactly or the index declines.
        #[allow(clippy::float_cmp)]
        fn index(argument: Option<&Value>, default: isize) -> Option<isize> {
            match argument {
                None | Some(Value::Nil) => Some(default),
                Some(Value::Integer(value)) => {
                    Some(isize::try_from(*value).unwrap_or(if *value < 0 {
                        isize::MIN
                    } else {
                        isize::MAX
                    }))
                }
                Some(Value::Number(value)) => {
                    #[allow(clippy::cast_possible_truncation)]
                    let small = *value as i32;
                    (f64::from(small) == *value).then_some(small as isize)
                }
                Some(_) => None,
            }
        }
        let given = &registers[arguments..arguments + argument_count];
        let Value::String(source) = &given[0] else {
            return false;
        };
        let Ok(length) = isize::try_from(source.len()) else {
            return false;
        };
        let Some(raw_start) = index(given.get(1), 1) else {
            return false;
        };
        let Some(raw_end) = index(given.get(2), raw_start) else {
            return false;
        };
        // The bytes are copied out first: the results may overlap the
        // arguments. The clamp is the one `string.byte` and `string.sub` use
        // everywhere.
        let mut bytes = [0_u8; BYTE_RESULTS_IN_PLACE];
        let mut count = 0;
        if let Some(range) = crate::stdlib::substring_range(length, raw_start, raw_end) {
            // More bytes than results: the extra are dropped.
            count = range.len().min(results);
            if count > BYTE_RESULTS_IN_PLACE {
                return false;
            }
            bytes[..count].copy_from_slice(&source.as_bytes()[range.start..range.start + count]);
        }
        for (position, slot) in registers[a..a + results].iter_mut().enumerate() {
            let value = match bytes[..count].get(position) {
                Some(byte) => self.runtime.integer_value(i64::from(*byte)),
                None => Value::Nil,
            };
            set_register(slot, value);
        }
        true
    }

    /// A native that returns any number of values.
    #[inline(never)]
    fn multiple_native(
        &mut self,
        registers: &mut [Value; REGISTER_WINDOW],
        shape: u64,
        callback: NativeCallback,
    ) -> Result<NativeStep, RuntimeError> {
        let [_, _, arguments, argument_count] = unpack_call_shape(shape);
        let returned = match callback(
            &mut self.runtime,
            &registers[arguments..arguments + argument_count],
        ) {
            Err(RuntimeError::Redispatch) => return Ok(NativeStep::Declined),
            result => result?,
        };
        Ok(self.store_returned(registers, shape, returned))
    }

    /// A native that the runtime alone cannot run. Closures registered by
    /// embedders run here too; only host callbacks, which need the VM's
    /// roots, decline.
    #[inline(never)]
    fn other_native(
        &mut self,
        registers: &mut [Value; REGISTER_WINDOW],
        shape: u64,
        native: NativeFunctionHandle,
    ) -> Result<NativeStep, RuntimeError> {
        let [_, _, arguments, argument_count] = unpack_call_shape(shape);
        let Some(returned) = self
            .runtime
            .call_native_unhosted(native, &registers[arguments..arguments + argument_count])?
        else {
            return Ok(NativeStep::Declined);
        };
        Ok(self.store_returned(registers, shape, returned))
    }

    /// Stores the vector of values a native returned into the call's result
    /// registers and ends the call.
    #[inline(always)]
    #[allow(unsafe_code)]
    fn store_returned(
        &mut self,
        registers: &mut [Value; REGISTER_WINDOW],
        shape: u64,
        mut returned: Vec<Value>,
    ) -> NativeStep {
        let [a, results, _, _] = unpack_call_shape(shape);
        // Strings move out; other values copy field by field. Moving whole
        // values staged them on the stack, and reloading those stalled.
        let place = |slot: &mut Value, value: Option<&mut Value>| match value {
            Some(value @ Value::String(_)) => {
                set_register(slot, std::mem::replace(value, Value::Nil));
            }
            Some(value) => store_clone(slot, value),
            None => set_register(slot, Value::Nil),
        };
        if results == 1 {
            place(&mut registers[a], returned.first_mut());
        } else {
            for (position, slot) in registers[a..a + results].iter_mut().enumerate() {
                place(slot, returned.get_mut(position));
            }
        }
        if returned.len() <= results {
            // SAFETY: every element was visited above. A string was swapped
            // for `Nil` and every other variant is a plain copy that owns no
            // resource, so dropping the elements would do nothing. Forgetting
            // them here spares the recycled vector a drop loop.
            unsafe { returned.set_len(0) };
        }
        self.runtime.recycle_result_vec(returned);
        self.native_step(None)
    }

    /// What follows a native call that stored its results. `observed` is the
    /// scalar native the dispatch loop would report to a call hook; its
    /// multi-result path reports none.
    #[inline(always)]
    fn native_step(&self, observed: Option<(NativeFunctionHandle, usize)>) -> NativeStep {
        if let Some((native, argument_count)) = observed
            && self.hooks_observe_calls
        {
            return NativeStep::Hooked(native, argument_count);
        }
        match self.native_checks() {
            None => NativeStep::Done,
            Some(FastExit::Backedge) => NativeStep::Backedge,
            Some(_) => NativeStep::Safepoint,
        }
    }

    /// The checks every completed native call ends with: the collector's
    /// safepoint, then a flush request that the dispatch loop's prologue
    /// services.
    #[inline(always)]
    fn native_checks(&self) -> Option<FastExit> {
        if self.runtime.allocation_serial() != self.gc_observed_allocation_serial
            && self.runtime.gc_needs_collection()
        {
            return Some(FastExit::Safepoint);
        }
        // A native such as `jit.flush` left a request that the dispatch
        // loop's prologue services; the call itself is complete.
        if self.runtime.jit_flush_requested() {
            return Some(FastExit::Backedge);
        }
        None
    }

    /// Calls the native in the function register of a `Call` with an open
    /// result count, of a `CallOpen`, or of a native `TailCall`, and stores
    /// its results: fixed counts as `Call` does, an open count whole in the
    /// frame's open-results slot. A `CallOpen` takes its last arguments from
    /// that slot. The program counter is already past the call when
    /// `advanced`. `None` means nothing changed and the general path takes
    /// the call: the callee is no plain native, a host callback needs the
    /// VM's roots, or an operand lies outside the window.
    // The three cases are the fast-path protocol: decline, complete, or an exit.
    #[allow(clippy::option_option)]
    #[inline(never)]
    fn native_open_call(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
        advanced: bool,
    ) -> VmResult<Option<Option<FastExit>>> {
        let depth = stack.len();
        let index = depth - 1;
        let (frames, values) = stack.frames_and_values();
        let frame = &mut frames[index];
        let base = frame.base as usize;
        let window = (frame.top - frame.base) as usize;
        let a = usize::from(instruction.a());
        let fixed = usize::from(instruction.b());
        let (results, consumes_open) = match opcode_byte(instruction) {
            op::CALL => (usize::from(instruction.c()), false),
            op::TAIL_CALL => (0, false),
            op::CALL_OPEN => (usize::from(instruction.c()), true),
            _ => return Ok(None),
        };
        let arguments = a + 1;
        if depth >= MAX_LUA_CALL_FRAMES
            || arguments + fixed > window
            || a >= window
            || a + results > window
            || (consumes_open && !frame.has_open_results())
        {
            return Ok(None);
        }
        let Value::NativeFunction(native) = values[base + a] else {
            return Ok(None);
        };
        let kind = self.runtime.native_call_kind(native);
        let pc = frame.pc;
        if !advanced {
            frame.last_interpreted_pc = crate::vm::frame::LastPc::some(pc);
            frame.pc = pc + 1;
        }
        // Without fixed arguments the open results are the argument list.
        let fixed_window = &values[base + arguments..base + arguments + fixed];
        let mut staged = None;
        if consumes_open && fixed != 0 {
            let mut list = self.runtime.result_vec();
            list.extend_from_slice(fixed_window);
            list.extend_from_slice(frame.open_values().unwrap_or_default());
            staged = Some(list);
        }
        let argument_window: &[Value] = match &staged {
            Some(list) => list,
            None if consumes_open => frame.open_values().unwrap_or_default(),
            None => fixed_window,
        };
        let argument_count = argument_window.len();
        if let NativeCallKind::Scalar(callback) = kind {
            let mut returned = callback(&mut self.runtime, argument_window);
            if let Some(list) = staged {
                self.runtime.recycle_result_vec(list);
            }
            if returned.is_err() {
                let error = returned.expect_err("checked above");
                if matches!(error, RuntimeError::Redispatch) {
                    // Declined without effect: the general path repeats the call.
                    if !advanced {
                        frame.pc = pc;
                    }
                    return Ok(None);
                }
                let (frame, parent) = frame_and_parent(frames, index);
                return Err(self.native_error(frame, parent, error));
            }
            if results == 0 {
                // The result replaces the arguments in the same buffer.
                push_result(frame.open_buffer(), &mut returned);
            } else {
                frame.clear_open_results();
                let (first, rest) = values[base + a..base + a + results]
                    .split_first_mut()
                    .expect("a fixed call result has at least one destination");
                store_result(first, &mut returned);
                for slot in rest {
                    set_register(slot, Value::Nil);
                }
            }
            // The value is stored or moved out: nothing is left to drop.
            std::mem::forget(returned);
            return self
                .finish_fast_native(stack, Some((native, argument_count)))
                .map(Some);
        }
        let (scalar, returned, observed) = match kind {
            NativeCallKind::Scalar(_) => unreachable!("single-result natives return above"),
            NativeCallKind::Multiple(callback) => {
                match callback(&mut self.runtime, argument_window) {
                    Ok(returned) => (None, returned, None),
                    Err(RuntimeError::Redispatch) => {
                        if !advanced {
                            frame.pc = pc;
                        }
                        if let Some(list) = staged {
                            self.runtime.recycle_result_vec(list);
                        }
                        return Ok(None);
                    }
                    Err(error) => {
                        let (frame, parent) = frame_and_parent(frames, index);
                        return Err(
                            call_argument_error(frame, parent, error, self.compatibility).into(),
                        );
                    }
                }
            }
            NativeCallKind::Other => {
                match self.runtime.call_native_unhosted(native, argument_window) {
                    Ok(Some(returned)) => (None, returned, None),
                    // The dispatch loop repeats the call.
                    Ok(None) => {
                        if !advanced {
                            frame.pc = pc;
                        }
                        if let Some(list) = staged {
                            self.runtime.recycle_result_vec(list);
                        }
                        return Ok(None);
                    }
                    Err(error) => {
                        let (frame, parent) = frame_and_parent(frames, index);
                        return Err(
                            call_argument_error(frame, parent, error, self.compatibility).into(),
                        );
                    }
                }
            }
        };
        if let Some(list) = staged {
            self.runtime.recycle_result_vec(list);
        }
        if results == 0 {
            // The result replaces the arguments in the same buffer.
            if let Some(value) = scalar {
                frame.open_buffer().push(value);
            } else {
                let old = frame.replace_open_results(returned);
                self.runtime.recycle_result_vec(old);
            }
        } else {
            frame.clear_open_results();
            let destination = &mut values[base + a..base + a + results];
            if let Some(value) = scalar {
                let (first, rest) = destination
                    .split_first_mut()
                    .expect("a fixed call result has at least one destination");
                set_register(first, value);
                for slot in rest {
                    set_register(slot, Value::Nil);
                }
            } else {
                let mut returned = returned;
                for (position, slot) in destination.iter_mut().enumerate() {
                    match returned.get_mut(position) {
                        Some(value) => {
                            set_register(slot, std::mem::replace(value, Value::Nil));
                        }
                        None => set_register(slot, Value::Nil),
                    }
                }
                self.runtime.recycle_result_vec(returned);
            }
        }
        self.finish_fast_native(stack, observed).map(Some)
    }

    /// A fixed `TailCall` of a native, with the program counter already past
    /// it: runs the native as a call with an open result count. `false`
    /// leaves the call to the general tail-call path, with nothing changed.
    /// The collector and flush checks that follow a native call are left to
    /// the caller's own.
    pub(crate) fn native_tail_call(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<bool> {
        Ok(self.native_open_call(stack, instruction, true)?.is_some())
    }

    /// The cold side of [`Self::switch_activation`]'s entry checks: a call
    /// with an open result count goes to [`Self::open_result_call`], any
    /// other declines.
    #[cold]
    #[inline(never)]
    fn call_declined(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<Option<FastExit>> {
        if opcode_byte(instruction) == op::CALL && instruction.c() == 0 {
            self.open_result_call(stack, instruction)
        } else {
            Ok(Some(FastExit::Fallback))
        }
    }

    /// [`Self::switch_activation`] for a `Call` with an open result count.
    /// Plain Lua callees switch here and natives run; the arms that serve
    /// fixed counts (`tostring`, inline plans) leave it to the dispatch loop.
    #[cold]
    #[inline(never)]
    fn open_result_call(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<Option<FastExit>> {
        let depth = stack.len();
        let frame = stack.at(depth - 1);
        let function = usize::from(instruction.a());
        let (arguments, argument_count) = (function + 1, usize::from(instruction.b()));
        if depth >= MAX_LUA_CALL_FRAMES || arguments + argument_count > frame.registers.len() {
            return Ok(Some(FastExit::Fallback));
        }
        let closure = match frame.registers.get(function) {
            Some(&Value::Closure(closure)) => closure,
            Some(Value::NativeFunction(_)) => {
                return Ok(self
                    .native_open_call(stack, instruction, false)?
                    .unwrap_or(Some(FastExit::Fallback)));
            }
            _ => return Ok(Some(FastExit::Fallback)),
        };
        self.push_open_closure_call(stack, instruction, closure, function + 1, argument_count)
    }

    /// The tail shared by [`Self::native_call_activation`]'s result shapes.
    /// `observed` is the scalar native the dispatch loop would report to a
    /// call hook; its multi-result path reports none.
    #[inline(always)]
    fn finish_fast_native(
        &mut self,
        stack: &mut CallStack,
        observed: Option<(NativeFunctionHandle, usize)>,
    ) -> VmResult<Option<FastExit>> {
        // Hooks keep the fast loop out, but the native may have set one.
        if let Some((native, argument_count)) = observed
            && self.hooks_observe_calls
        {
            self.observe_native_call(stack, Value::NativeFunction(native), argument_count)?;
        }
        Ok(self.native_checks())
    }

    /// Runs a non-Lua call `step` exactly as the dispatch loop would: with
    /// the program counter past the call while it runs, restored when the
    /// step declines so the general path can take the call instead.
    #[inline(always)]
    fn fast_native_step(
        &mut self,
        stack: &mut CallStack,
        depth: usize,
        step: impl FnOnce(&mut Self, &mut CallStack) -> VmResult<bool>,
    ) -> VmResult<Option<FastExit>> {
        let frame = &mut stack[depth - 1];
        // One word, so saving and restoring it never straddles two stores.
        let previous = frame.last_interpreted_pc;
        frame.last_interpreted_pc = crate::vm::frame::LastPc::some(frame.pc);
        frame.pc += 1;
        if !step(self, stack)? {
            let frame = &mut stack[depth - 1];
            frame.pc -= 1;
            frame.last_interpreted_pc = previous;
            return Ok(Some(FastExit::Fallback));
        }
        if self.runtime.allocation_serial() != self.gc_observed_allocation_serial
            && self.runtime.gc_needs_collection()
        {
            return Ok(Some(FastExit::Safepoint));
        }
        Ok(None)
    }

    /// Shows the recorder the instruction at `pc` before the fast loop runs
    /// it. Returns whether the recording goes on.
    #[inline(never)]
    fn record_fast_hook(
        &mut self,
        chunk: &Arc<crate::bytecode::Chunk>,
        (depth, top_level): (usize, bool),
        pc: usize,
        registers: &[Value],
        upvalues: &[crate::runtime::CellHandle],
        environment: crate::runtime::TableHandle,
        (varargs, open): (&[Value], &[Value]),
    ) -> bool {
        // A host function the recording sleeps through runs this code.
        if self
            .jit
            .trace
            .skips_nested_host(self.host_yield_boundaries.len())
        {
            return true;
        }
        let at = HookPoint {
            chunk,
            depth,
            top_level,
            pc,
            registers,
            open,
            coroutine: self.current_coroutine,
            hooks_active: !self.count_only_hooks(),
            source: HookSource::Fast,
            surroundings: Some(Surroundings {
                upvalues,
                environment,
                runtime: &self.runtime,
                varargs,
            }),
        };
        let continues = self.jit.trace.record(&mut self.jit.statistics.trace, &at);
        if continues {
            self.jit
                .trace
                .latch_host_depth(self.host_yield_boundaries.len());
        }
        // A recording that just closed has a compile request to hand over.
        if !continues && self.jit.trace.has_compile_work() {
            self.service_trace_compilations();
        }
        continues
    }

    /// Decides whether a completed backedge must leave the fast path.
    #[inline(always)]
    fn fast_backedge(
        &mut self,
        tracing: bool,
        target: usize,
        code: &[Instruction],
    ) -> Option<FastExit> {
        if tracing {
            // The last native lookup found nothing to enter; keep
            // interpreting without repeating it on every iteration. The
            // window also covers the collector poll, which runs at its end
            // (and at every allocation the loop makes), so a rejected loop
            // pays one counter check per backedge, like the interpreter's
            // collector countdown.
            if self.jit.backedge_skip > 0 {
                self.jit.backedge_skip -= 1;
                return None;
            }
            self.tracing_backedge_exit(target, code)
        } else {
            // Polling the collector costs more than the rest of a short
            // loop's backedge, so poll once per interval.
            self.backedge_gc_countdown -= 1;
            if self.backedge_gc_countdown == 0 {
                self.backedge_gc_countdown = BACKEDGE_GC_POLL_INTERVAL;
                if self.runtime.gc_needs_collection() {
                    return Some(FastExit::Safepoint);
                }
            }
            None
        }
    }

    /// The exit for a tracing backedge whose skip ran out. Out of line, so the
    /// tracing loop's body stays the size of the interpreter's and the
    /// countdown is all it adds to a rejected loop's backedge. A loop the
    /// tracer rejected has a window of its own to pass here (see
    /// `RejectedSkips`), which `None` says this backedge is in.
    #[cold]
    #[inline(never)]
    fn tracing_backedge_exit(&mut self, target: usize, code: &[Instruction]) -> Option<FastExit> {
        if self.jit.rejected_skips.pass(code.as_ptr() as usize, target) {
            return None;
        }
        if self.runtime.gc_needs_collection() {
            return Some(FastExit::Safepoint);
        }
        self.jit.fast_backedge_pc = Some(target);
        Some(FastExit::Backedge)
    }
}

/// The four operands of a native call, in one register: the first result
/// register, the result count, the first argument register and the argument
/// count. Each is at most 256.
const fn pack_call_shape(a: usize, results: usize, arguments: usize, argument_count: usize) -> u64 {
    (a as u64) | (results as u64) << 16 | (arguments as u64) << 32 | (argument_count as u64) << 48
}

/// The operands [`pack_call_shape`] packed.
const fn unpack_call_shape(shape: u64) -> [usize; 4] {
    [
        (shape & 0xffff) as usize,
        (shape >> 16 & 0xffff) as usize,
        (shape >> 32 & 0xffff) as usize,
        (shape >> 48) as usize,
    ]
}

/// What a native call that stored its results leaves for the loop.
// Two words at most, passed by value through the hot native-call path.
#[derive(Clone, Copy)]
enum NativeStep {
    /// Nothing: the activation goes on.
    Done,
    /// The call changed nothing; the general call path takes it.
    Declined,
    /// The collector has work to do.
    Safepoint,
    /// A `jit.flush` request waits for the dispatch loop.
    Backedge,
    /// The native may have set a hook, which sees the call.
    Hooked(NativeFunctionHandle, usize),
    /// The open-results call and the `CallOpen` after it both ran (see
    /// [`Vm::open_native_in_window`]); the loop moves past the second.
    Fused,
    /// As [`NativeStep::Fused`], then a flush request waits (`true`) or the
    /// collector has work to do (`false`).
    FusedExit(bool),
}

/// The value a native whose plan returns one value (`Scalar`, `Unary`,
/// `Binary`) gives for `arguments`, and whether it came from the pure
/// numeric form, which cannot fail, allocate or run Lua. `None` when the
/// native declined without effect ([`RuntimeError::Redispatch`]).
#[inline(always)]
fn single_native_value(
    runtime: &mut Runtime,
    native: NativeFunctionHandle,
    plan: NativePlan,
    arguments: &[Value],
) -> Result<Option<(Value, bool)>, RuntimeError> {
    let callback = match (plan, arguments) {
        (NativePlan::Unary(_, unary), [Value::Number(x)]) => {
            return Ok(Some((Value::Number(unary(*x)), true)));
        }
        (NativePlan::Binary(_, binary), [Value::Number(x), Value::Number(y)]) => {
            return Ok(Some((Value::Number(binary(*x, *y)), true)));
        }
        // A `bit` operation on a 64-bit integer.
        (NativePlan::Binary(callback, _), [x, y]) => {
            if let Some(wide) = runtime.native_wide_binary(native)
                && let Some(value) = wide(x, y)
            {
                return Ok(Some((value, true)));
            }
            callback
        }
        (
            NativePlan::Scalar(callback)
            | NativePlan::Unary(callback, _)
            | NativePlan::Binary(callback, _),
            _,
        ) => callback,
        _ => return Ok(None),
    };
    match callback(runtime, arguments) {
        Ok(value) => Ok(Some((value, false))),
        Err(RuntimeError::Redispatch) => Ok(None),
        Err(error) => Err(error),
    }
}

/// [`single_native_value`] for fixed arguments followed by open results,
/// staged in a fixed buffer. `None` beyond [`OPEN_NATIVE_ARGUMENTS`].
#[cold]
#[inline(never)]
fn staged_native_value(
    runtime: &mut Runtime,
    native: NativeFunctionHandle,
    plan: NativePlan,
    fixed: &[Value],
    open: &[Value],
) -> Result<Option<(Value, bool)>, RuntimeError> {
    let count = fixed.len() + open.len();
    if count > OPEN_NATIVE_ARGUMENTS {
        return Ok(None);
    }
    let mut staged: [Value; OPEN_NATIVE_ARGUMENTS] = std::array::from_fn(|_| Value::Nil);
    for (slot, value) in staged.iter_mut().zip(fixed.iter().chain(open)) {
        *slot = clone_register_value(value);
    }
    single_native_value(runtime, native, plan, &staged[..count])
}

#[cfg(test)]
thread_local! {
    /// Calls [`Vm::open_native_in_window`] ran on this thread.
    pub(crate) static OPEN_NATIVES_IN_PLACE: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
thread_local! {
    /// Open-results pairs [`Vm::open_native_in_window`] fused on this thread.
    pub(crate) static FUSED_OPEN_NATIVES: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

/// Most arguments [`Vm::open_native_in_window`] stages for a `CallOpen`
/// with fixed arguments before its open results.
const OPEN_NATIVE_ARGUMENTS: usize = 8;

/// Whether [`Vm::interpret_fast`] can execute `opcode` for common operands.
const fn handles(opcode: OpCode) -> bool {
    matches!(
        opcode,
        OpCode::Move
            | OpCode::LoadConstant
            | OpCode::LoadNil
            | OpCode::ClearRange
            | OpCode::Not
            | OpCode::Jump
            | OpCode::JumpIfFalse
            | OpCode::NumericForPrep
            | OpCode::NumericForLoop
            | OpCode::ForInLoop
            | OpCode::Add
            | OpCode::Subtract
            | OpCode::Multiply
            | OpCode::Divide
            | OpCode::Modulo
            | OpCode::AddK
            | OpCode::SubtractK
            | OpCode::MultiplyK
            | OpCode::DivideK
            | OpCode::ModuloK
            | OpCode::Power
            | OpCode::FloorDivide
            | OpCode::Negate
            | OpCode::Equal
            | OpCode::LessThan
            | OpCode::LessEqual
            | OpCode::TestEqual
            | OpCode::TestLessThan
            | OpCode::TestLessEqual
            | OpCode::TestEqualK
            | OpCode::TestLessThanK
            | OpCode::TestLessEqualK
            | OpCode::GetGlobal
            | OpCode::SetGlobal
            | OpCode::GetUpvalue
            | OpCode::SetUpvalue
            | OpCode::GetTable
            | OpCode::SetTable
            | OpCode::GetField
            | OpCode::SelfField
            | OpCode::SetField
            | OpCode::GetTabUp
            | OpCode::SetTabUp
    )
}

/// Reads a local of the fast loop that is kept in its stack slot: the access
/// is volatile, so the compiler neither keeps the value in a register across
/// the loop nor merges two reads.
#[inline(always)]
#[allow(unsafe_code)]
fn slot_get<T: Copy>(slot: &T) -> T {
    // SAFETY: a reference is valid for a read.
    unsafe { std::ptr::read_volatile(slot) }
}

/// Writes a local of the fast loop that is kept in its stack slot (see
/// [`slot_get`]).
#[inline(always)]
#[allow(unsafe_code)]
fn slot_set<T: Copy>(slot: &mut T, value: T) {
    // SAFETY: a reference is valid for a write, and `T` has no destructor.
    unsafe { std::ptr::write_volatile(slot, value) }
}

/// `pointer`, which the compiler cannot see through. Offsets from the result
/// are computed where they are used. From `frame` itself they were loop
/// invariant: the compiler computed each at every activation switch and
/// stored it in a stack slot, to save an `add` in the few arms that use it.
#[inline(always)]
#[allow(unsafe_code)]
pub(super) fn opaque<T>(pointer: *mut T) -> *mut T {
    #[cfg_attr(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        allow(unused_mut)
    )]
    let mut pointer = pointer;
    // SAFETY: the empty template changes nothing. Without `pure` the
    // compiler may not move, merge or drop the statement. The template
    // `/* {0} */` only prints the register name as a comment: the pointer
    // is carried as a register value and never dereferenced, so `nomem`
    // holds despite the pointer operand.
    #[allow(clippy::pointers_in_nomem_asm_block)]
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        core::arch::asm!(
            "/* {0} */",
            inout(reg) pointer,
            options(nomem, nostack, preserves_flags)
        );
    }
    pointer
}

/// [`opaque`] for a word: the same value, with nothing known about it.
#[inline(always)]
#[allow(unsafe_code)]
fn opaque_word(value: usize) -> usize {
    #[cfg_attr(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        allow(unused_mut)
    )]
    let mut value = value;
    // SAFETY: as for `opaque`: the empty template changes nothing.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        core::arch::asm!(
            "/* {0} */",
            inout(reg) value,
            options(nomem, nostack, preserves_flags)
        );
    }
    value
}

/// The most results of `string.byte` the dispatch loop writes in place.
const BYTE_RESULTS_IN_PLACE: usize = 8;

/// The most results of `string.find` and `string.match` the dispatch loop
/// writes in place.
const PATTERN_RESULTS_IN_PLACE: usize = 8;

/// Bytes of no-ops [`align_dispatch`] emits after the line boundary. They
/// are skipped, and only move where the loop's code lands. Tune by timing
/// the tight loops (`numeric/for_sum`, `control/branching`) after any change
/// to the loop: with the wrong value the dispatch sequence straddles a line
/// and they run a fifth slower, and `objdump` shows only an approximate
/// reason. Try the values from 1 to 64.
///
/// Last tuned with the numeric for latch trimmed: skews 0 and 24 to 60 in
/// steps of four kept `numeric/for_sum` within 1% of the old build, and 4 to
/// 20 cost it 10 to 25%. 32 also kept `opt/call_args`, `calls/closure` and
/// `calls/recursive_fib` within 1%.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const DISPATCH_SKEW: usize = 32;

/// Pads the code stream to a cache line. Placed just before the fast loop,
/// it fixes where the loop's dispatch sequence (fetch, decode, jump through
/// the table) falls in a line: the offset from the pad is set by the loop's
/// own code, not by what the linker put before it. A sequence that straddles
/// a line costs a tight loop a fifth of its cycles (frontend latency in
/// `perf stat -M frontend_bound_latency`). Check with `objdump` that the
/// sequence from the loop's `mov (...,%r15,4)` fetch to its `jmp *%rcx`
/// stays inside one 64-byte window after changing the loop's prologue.
#[inline(always)]
#[allow(unsafe_code)]
fn align_dispatch() {
    // SAFETY: the directive emits padding and touches nothing. The jump
    // skips the padding, which would otherwise run as no-ops on every
    // activation switch. The no-ops after the line boundary move the loop
    // head, which the loop's own spill stores place after them, to a line
    // offset at which the dispatch sequence fits in the line; the jump skips
    // them too.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::asm!(
            "jmp 2f",
            ".p2align 6",
            ".nops {skew}",
            "2:",
            skew = const DISPATCH_SKEW,
            options(nomem, nostack, preserves_flags)
        );
    }
}

/// The instruction at `pc`, without a bounds check.
///
/// The fast loop calls this once per instruction, and the check was a
/// measurable share of its dispatch. The caller guarantees `pc < code.len()`:
/// the first activation's entry pc is checked when the loop starts. Every
/// later activation starts in range by construction: a pushed callee or a
/// metamethod at pc 0 of validated, non-empty code; a return at the caller's
/// pc after a `Call`, which is never the last instruction; a miss, scalar miss, or
/// native step at `pc + 1` after a non-terminator. From there
/// `Chunk::validate` keeps `pc` in range. Every closure's chunk has passed it:
/// `Runtime::create_closure_in_environment` validates, and the VM's own
/// unchecked variant is only used for loaded chunks and their prototypes.
/// Validation rejects a chunk whose last instruction is not a `Return`,
/// `ReturnVararg`, `ReturnOpen` or `Halt`, and rejects jump targets outside
/// the code. The loop never steps past those four opcodes: it hands them to
/// the switch or exits with `pc` unchanged. Every other instruction has a
/// successor. A jump that lands past a target (the skipped latch of an empty
/// numeric loop) checks that itself.
#[inline(always)]
#[allow(unsafe_code)]
fn fetch(code: &[Instruction], pc: usize) -> Instruction {
    debug_assert!(pc < code.len(), "fast loop pc ran off the chunk");
    // SAFETY: the code was validated and the caller keeps `pc` in range, as
    // argued above.
    unsafe { *code.get_unchecked(pc) }
}

/// [`fetch`] from the address of a chunk's first instruction, for the fast
/// loop, which keeps no length: `code` is the first instruction of the
/// running chunk, validated, and the loop keeps `pc` in range of it, as for
/// [`fetch`].
#[allow(unsafe_code, clippy::inline_always)]
#[inline(always)]
fn fetch_at(code: *const Instruction, pc: usize) -> Instruction {
    // SAFETY: as for `fetch`: the code was validated and the loop keeps `pc`
    // in range.
    unsafe { *code.add(pc) }
}

/// What the fast loop does after [`Vm::for_in_step`].
enum ForInStep {
    /// The general path must take the `ForInCall`; nothing changed.
    Fallback,
    /// The step is done. The `ForInLoop` after it is left to the loop.
    Next,
    /// The step is done, and the `ForInLoop` after it ended the loop.
    Exit,
    /// The step is done, and the `ForInLoop` after it jumps to the pc.
    Back(usize),
}

/// The test and copy of `ForInLoop`: a non-nil first loop variable
/// (`R[a + 3]`) becomes the control value (`R[a + 2]`) and the loop goes on.
/// Returns whether it does.
///
/// Validation keeps `a ..= a + 3` inside the frame's registers, so the window
/// index needs no check of its own: with it, the arm spent more instructions
/// on bounds tests than on the loop.
#[inline(always)]
#[allow(unsafe_code)]
fn for_in_advance(registers: &mut [Value; REGISTER_WINDOW], a: usize) -> bool {
    debug_assert!(a + 3 < REGISTER_WINDOW);
    // SAFETY: `Chunk::validate` checked that `a + 3` is a register, so the
    // three pointers stay inside the window.
    unsafe {
        let control = registers.as_mut_ptr().add(a + 2);
        let first = control.add(1);
        if matches!(*first, Value::Nil) {
            return false;
        }
        if matches!(*control, Value::String(_)) | matches!(*first, Value::String(_)) {
            // A string needs its reference count kept.
            copy_register(registers, a + 2, a + 3);
        } else {
            std::ptr::copy(first, control, 1);
        }
    }
    true
}

/// The constant at `index`, without a bounds check. The K opcodes and
/// `LoadConstant` carry an index that `Chunk::validate` has checked against the
/// pool, and the fast loop only runs validated chunks (see `fetch`). Without
/// the check, an arm needs the pool's pointer but not its length, so a loop
/// that is short of registers spills one value fewer.
#[inline(always)]
#[allow(unsafe_code)]
fn constant_at(chunk: &crate::bytecode::Chunk, index: usize) -> &Constant {
    debug_assert!(index < chunk.constants.len(), "constant index out of range");
    // SAFETY: validation checked the index, as argued above.
    unsafe { chunk.constants.get_unchecked(index) }
}

/// A jump target. Chunks are validated at load time, so every target is in
/// range, which the fast loop's unchecked fetch relies on.
#[inline(always)]
fn relative(next: usize, offset: i32) -> usize {
    next.wrapping_add_signed(offset as isize)
}

#[inline(always)]
fn short_offset(instruction: Instruction) -> i32 {
    i32::from(i16::from_ne_bytes(instruction.d().to_ne_bytes()))
}

/// Whether an opcode byte is one [`Vm::table_operator_miss`] looks at: the
/// arithmetic instructions, comparisons and tests whose handlers the fast
/// loop enters.
static OPERATOR_MISS: [bool; 256] = {
    let mut table = [false; 256];
    let opcodes = [
        OpCode::Add,
        OpCode::Subtract,
        OpCode::Multiply,
        OpCode::Divide,
        OpCode::Modulo,
        OpCode::Equal,
        OpCode::LessThan,
        OpCode::LessEqual,
        OpCode::AddK,
        OpCode::SubtractK,
        OpCode::MultiplyK,
        OpCode::DivideK,
        OpCode::ModuloK,
        OpCode::TestEqual,
        OpCode::TestLessThan,
        OpCode::TestLessEqual,
        OpCode::TestEqualK,
        OpCode::TestLessThanK,
        OpCode::TestLessEqualK,
    ];
    let mut index = 0;
    while index < opcodes.len() {
        table[opcodes[index] as usize] = true;
        index += 1;
    }
    table
};

/// Raw opcode bytes for the fast loop's dispatch.
mod op {
    use crate::bytecode::OpCode;

    macro_rules! opcode_bytes {
        ($($name:ident = $opcode:ident),* $(,)?) => {
            $(pub(super) const $name: u8 = OpCode::$opcode as u8;)*
        };
    }

    macro_rules! opcode_list {
        ($($name:ident = $opcode:ident),* $(,)?) => {
            /// Every opcode byte the dispatch lists.
            #[cfg(test)]
            pub(super) const ALL: &[u8] = &[$(OpCode::$opcode as u8),*];
        };
    }

    macro_rules! both {
        ($($tokens:tt)*) => {
            opcode_bytes!($($tokens)*);
            opcode_list!($($tokens)*);
        };
    }

    both!(
        MOVE = Move,
        LOAD_CONSTANT = LoadConstant,
        LOAD_NIL = LoadNil,
        CLEAR_RANGE = ClearRange,
        NOT = Not,
        JUMP = Jump,
        JUMP_IF_FALSE = JumpIfFalse,
        NUMERIC_FOR_PREP = NumericForPrep,
        NUMERIC_FOR_LOOP = NumericForLoop,
        FOR_IN_CALL = ForInCall,
        FOR_IN_LOOP = ForInLoop,
        ADD = Add,
        SUBTRACT = Subtract,
        MULTIPLY = Multiply,
        DIVIDE = Divide,
        MODULO = Modulo,
        ADD_K = AddK,
        SUBTRACT_K = SubtractK,
        MULTIPLY_K = MultiplyK,
        DIVIDE_K = DivideK,
        MODULO_K = ModuloK,
        POWER = Power,
        FLOOR_DIVIDE = FloorDivide,
        NEGATE = Negate,
        EQUAL = Equal,
        LESS_THAN = LessThan,
        LESS_EQUAL = LessEqual,
        TEST_EQUAL = TestEqual,
        TEST_LESS_THAN = TestLessThan,
        TEST_LESS_EQUAL = TestLessEqual,
        TEST_EQUAL_K = TestEqualK,
        TEST_LESS_THAN_K = TestLessThanK,
        TEST_LESS_EQUAL_K = TestLessEqualK,
        GET_GLOBAL = GetGlobal,
        SET_GLOBAL = SetGlobal,
        GET_UPVALUE = GetUpvalue,
        SET_UPVALUE = SetUpvalue,
        GET_TABLE = GetTable,
        SET_TABLE = SetTable,
        GET_FIELD = GetField,
        SELF_FIELD = SelfField,
        SET_FIELD = SetField,
        GET_TAB_UP = GetTabUp,
        SET_TAB_UP = SetTabUp,
        CALL = Call,
        CALL_ONE = CallOne,
        TAIL_CALL = TailCall,
        RETURN = Return,
        HALT = Halt,
        LENGTH = Length,
        NEW_TABLE = NewTable,
        CONCAT = Concat,
        BIT_AND = BitAnd,
        BIT_OR = BitOr,
        BIT_XOR = BitXor,
        SHIFT_LEFT = ShiftLeft,
        SHIFT_RIGHT = ShiftRight,
        BIT_NOT = BitNot,
        CLOSURE = Closure,
        CLOSE_UPVALUE = CloseUpvalue,
        VARARG = Vararg,
        TABLE_VARARG = TableVararg,
        TABLE_OPEN = TableOpen,
        // Opcodes only the general loop runs, named so that the dispatch
        // lists every opcode and its jump table needs no range check.
        CALL_VARARG = CallVararg,
        RETURN_VARARG = ReturnVararg,
        CALL_OPEN = CallOpen,
        RETURN_OPEN = ReturnOpen,
        EXTRA_ARGUMENT = ExtraArgument,
        LOAD_CONSTANT_WIDE = LoadConstantWide,
        GET_GLOBAL_WIDE = GetGlobalWide,
        SET_GLOBAL_WIDE = SetGlobalWide,
        TAIL_CALL_VARARG = TailCallVararg,
        TAIL_CALL_OPEN = TailCallOpen,
        MARK_CLOSE = MarkClose,
        CLOSE_LOCAL = CloseLocal,
        DECLARE_GLOBAL = DeclareGlobal,
        DECLARE_GLOBAL_WIDE = DeclareGlobalWide,
        DECLARE_TABLE = DeclareTable,
    );
}

/// The opcode bytes the fast loop's dispatch lists, for tests.
#[cfg(test)]
pub(crate) fn dispatched_opcodes() -> &'static [u8] {
    op::ALL
}

/// The index of the instruction at `at` in `chunk`. The fast loop passes
/// instruction addresses to the calls below: a `pc` argument cost the other
/// opcodes an extra instruction each.
#[inline(always)]
fn pc_of(chunk: &Chunk, at: *const Instruction) -> usize {
    (at as usize - chunk.code.as_ptr() as usize) / size_of::<Instruction>()
}

/// The site at `pc` of `chunk`, given its `first` site from
/// [`Chunk::read_sites_first`] or [`Chunk::write_sites_first`]: an empty
/// placeholder until the chunk fills one.
#[allow(clippy::inline_always, unsafe_code)]
#[inline(always)]
fn site_at(
    chunk: &Chunk,
    first: *const crate::bytecode::FieldSite,
    pc: usize,
) -> &crate::bytecode::FieldSite {
    debug_assert!(pc < chunk.code.len());
    // SAFETY: `first` came from this chunk, so it points at one site per
    // instruction, valid for as long as the chunk lives. The fast loop runs
    // `chunk.code`, so `pc` is below its length.
    unsafe { &*first.add(pc) }
}

/// Stores `value`, which a read of `name` from `table` found, in `target` and
/// fills the read site of the instruction at `at`. Out of line and cold: the
/// `GetField` that missed its site keeps only the lookup in the fast loop.
#[cold]
#[inline(never)]
fn store_and_fill_read_site(
    runtime: &Runtime,
    chunk: &Chunk,
    at: *const Instruction,
    table: TableHandle,
    name: &crate::string::LuaString,
    target: &mut Value,
    value: &Value,
) {
    store_clone(target, value);
    // `fill_read_site`, without a second call: a loop whose sites keep
    // closing pays this on most reads.
    if !matches!(target, Value::Nil) {
        runtime.table_site_fill(table, chunk, pc_of(chunk, at), name);
    }
}

/// Fills the read site of the instruction at `at` after a read of `name`
/// from `table` that found `value`, unless it was nil. Out of line with the
/// other site code below: inlined into the fast loop, it costs every other
/// instruction a spilled register.
#[inline(never)]
fn fill_read_site(
    runtime: &Runtime,
    chunk: &Chunk,
    at: *const Instruction,
    table: TableHandle,
    name: &crate::string::LuaString,
    value: &Value,
) {
    if !matches!(value, Value::Nil) {
        runtime.table_site_fill(table, chunk, pc_of(chunk, at), name);
    }
}

/// Fills the write site of the `SetField` at `at` after a store over a value.
#[cold]
#[inline(never)]
fn fill_store_site(
    runtime: &mut Runtime,
    chunk: &Chunk,
    at: *const Instruction,
    instruction: Instruction,
    registers: &[Value],
) {
    if let Value::Table(table) = &registers[usize::from(instruction.a())]
        && let Some(Constant::String(name)) = chunk.constants.get(usize::from(instruction.b()))
    {
        runtime.table_site_fill_store(*table, chunk, pc_of(chunk, at), name);
    }
}

/// Whether each opcode byte advances a count hook: one lookup in the fast
/// loop instead of a comparison per free opcode.
static COUNTED_OPCODES: [u32; 256] = {
    let mut table = [0_u32; 256];
    let mut byte = 0;
    while byte < 256 {
        table[byte] = OpCode::byte_counts_for_hooks(byte as u8) as u32;
        byte += 1;
    }
    table
};

/// [`fill_store_site`] for the `SetTabUp` at `at` that wrote into `table`.
#[cold]
#[inline(never)]
fn fill_tab_up_store_site(
    runtime: &mut Runtime,
    chunk: &Chunk,
    at: *const Instruction,
    table: TableHandle,
    instruction: Instruction,
) {
    if let Some(Constant::String(name)) = chunk.constants.get(usize::from(instruction.b())) {
        runtime.table_site_fill_store(table, chunk, pc_of(chunk, at), name);
    }
}

#[inline(always)]
fn opcode_byte(instruction: Instruction) -> u8 {
    instruction.raw().to_le_bytes()[0]
}

/// The dispatch's unreachable arm.
///
/// Dispatch is on the raw opcode byte with no range check, so a byte that is
/// not an opcode must not reach it.
#[inline(always)]
#[allow(unsafe_code)]
fn unreachable_opcode() -> ! {
    debug_assert!(false, "opcode byte outside the validated opcode set");
    // SAFETY: the loop only runs chunks that passed `Chunk::validate`, which
    // rejects any instruction whose low byte is not an `OpCode`, and the
    // dispatch lists every `OpCode`.
    unsafe { std::hint::unreachable_unchecked() }
}

/// [`fast_scalar`] for the instruction's own opcode, out of line.
#[inline(never)]
fn scalar_slow(
    instruction: Instruction,
    left: &Value,
    right: &Value,
    float_arith: FloatArith,
) -> Option<Value> {
    let opcode = instruction.opcode().ok()?;
    // Equality that needs no numeric ladder: the raw comparison where `__eq`
    // cannot run.
    if opcode == OpCode::Equal
        && let Some(result) = plain_equal(left, right)
    {
        return result.map(Value::Boolean);
    }
    if let Some(value) = cint64_fast(opcode, left, right) {
        return Some(value);
    }
    if let Some(value) = fast_scalar(opcode, left, right, float_arith) {
        return Some(value);
    }
    // The byte order of two strings.
    match (opcode, left, right) {
        (OpCode::LessThan, Value::String(left), Value::String(right)) => {
            Some(Value::Boolean(left.cmp(right).is_lt()))
        }
        (OpCode::LessEqual, Value::String(left), Value::String(right)) => {
            Some(Value::Boolean(!left.cmp(right).is_gt()))
        }
        _ => None,
    }
}

/// What a comparison arm does with the exact order of an integer and a
/// float. NaN has no order, so every answer is false.
#[inline(always)]
fn order_eq(order: Option<std::cmp::Ordering>) -> bool {
    order == Some(std::cmp::Ordering::Equal)
}

#[inline(always)]
fn order_lt(order: Option<std::cmp::Ordering>) -> bool {
    order.is_some_and(std::cmp::Ordering::is_lt)
}

#[inline(always)]
fn order_le(order: Option<std::cmp::Ordering>) -> bool {
    order.is_some_and(std::cmp::Ordering::is_le)
}

/// The operands of an arithmetic arm as floats when exactly one is an
/// integer: the integer converts as the general path converts it.
#[inline(always)]
fn mixed_floats(left: &Value, right: &Value) -> Option<(f64, f64)> {
    match (left, right) {
        (Value::Integer(integer), Value::Number(float)) => {
            Some((integer_to_float(*integer), *float))
        }
        (Value::Number(float), Value::Integer(integer)) => {
            Some((*float, integer_to_float(*integer)))
        }
        _ => None,
    }
}

/// [`mixed_floats`] for a constant right operand. An integer constant of a
/// profile without integers loads as the same float, so it needs no profile
/// test.
#[inline(always)]
fn mixed_k_floats(left: &Value, constant: &Constant) -> Option<(f64, f64)> {
    match (left, constant) {
        (Value::Integer(integer), Constant::Number(float)) => {
            Some((integer_to_float(*integer), *float))
        }
        (Value::Number(float), Constant::Integer(integer)) => {
            Some((*float, integer_to_float(*integer)))
        }
        _ => None,
    }
}

/// A constant-operand arithmetic opcode on two floats.
#[inline(always)]
fn float_k_op(
    opcode: u8,
    floor_modulo: bool,
    float_arith: FloatArith,
    left: f64,
    right: f64,
) -> f64 {
    match opcode {
        op::ADD_K => left + right,
        op::SUBTRACT_K => left - right,
        op::MULTIPLY_K => left * right,
        op::DIVIDE_K => left / right,
        _ => float_modulo(floor_modulo, float_arith, left, right),
    }
}

/// `left % right` on floats. `floor_modulo` says the profile's `%` is the
/// floor formula and the CPU floors in one instruction, which is inlined;
/// the `fmod` forms and a CPU without it go out of line.
#[inline(always)]
#[allow(unsafe_code)]
fn float_modulo(floor_modulo: bool, float_arith: FloatArith, left: f64, right: f64) -> f64 {
    if floor_modulo {
        // SAFETY: the caller's `floor_modulo` includes
        // `FloatModulo::has_hardware_floor`.
        unsafe { FloatModulo::floor_apply_hardware(left, right) }
    } else {
        float_arith.modulo.apply_fmod(left, right)
    }
}

/// A tag byte no [`Value`] has.
const NO_VALUE_TAG: u8 = 0xff;

/// The tag byte of `value`.
#[inline(always)]
#[allow(unsafe_code)]
fn value_tag(value: &Value) -> u8 {
    // SAFETY: `Value` is `repr(C, u8)`: its tag is the byte at offset 0.
    unsafe { *std::ptr::from_ref(value).cast::<u8>() }
}

/// The integer in `value`, whose tag the caller found equal to the loop's
/// `integer_tag`.
#[inline(always)]
#[allow(unsafe_code)]
fn integer_operand(value: &Value) -> i64 {
    match value {
        Value::Integer(integer) => *integer,
        // SAFETY: `integer_tag` is `NATIVE_VALUE_TAG_INTEGER` or
        // `NO_VALUE_TAG`, which no value has, so a value whose tag equals
        // it is an integer.
        _ => unsafe { std::hint::unreachable_unchecked() },
    }
}

/// [`test_slow`], then [`table_test_slow`] for what it leaves: the
/// comparison arms' one out-of-line call, so that they pass the runtime
/// without a closure.
#[inline(never)]
fn test_slow_with(
    runtime: &Runtime,
    comparison: OpCode,
    left: &Value,
    right: &Value,
    float_arith: FloatArith,
) -> Option<bool> {
    test_slow(comparison, left, right, float_arith)
        .or_else(|| table_test_slow(runtime, comparison, left, right))
}

/// A test's comparison on operands that are not two floats or two
/// integers. `None` leaves metamethods, coercion errors and ordering errors to
/// the general path. The answers equal that path's: a primitive result, or
/// for equality the raw comparison when `__eq` cannot run.
#[inline(never)]
fn test_slow(
    comparison: OpCode,
    left: &Value,
    right: &Value,
    float_arith: FloatArith,
) -> Option<bool> {
    if comparison == OpCode::Equal
        && let Some(result) = plain_equal(left, right)
    {
        // Two distinct tables or userdata leave to `table_test_slow`.
        return result;
    }
    if let Some(Value::Boolean(result)) = cint64_fast(comparison, left, right) {
        return Some(result);
    }
    // A table never orders primitively: `__lt` or `__le` answers, or the
    // general path reports the error. (Equality with a table was decided
    // above, unless the other operand is numeric, which the ladder below
    // answers.) Asked after the 64-bit integers, which are tested on every
    // operation of theirs.
    if comparison != OpCode::Equal
        && (matches!(left, Value::Table(_)) || matches!(right, Value::Table(_)))
    {
        return None;
    }
    if let Some(Value::Boolean(result)) = fast_scalar(comparison, left, right, float_arith) {
        return Some(result);
    }
    match comparison {
        OpCode::Equal => {
            if crate::runtime::Runtime::raw_equal(left, right) {
                return Some(true);
            }
            // Only two tables or two userdata consult `__eq`.
            let eligible = matches!(
                (left, right),
                (Value::Table(_), Value::Table(_))
                    | (
                        Value::File(_) | Value::Userdata(_),
                        Value::File(_) | Value::Userdata(_)
                    )
            );
            (!eligible).then_some(false)
        }
        OpCode::LessThan => match (left, right) {
            (Value::String(left), Value::String(right)) => Some(left.cmp(right).is_lt()),
            _ => None,
        },
        OpCode::LessEqual => match (left, right) {
            (Value::String(left), Value::String(right)) => Some(!left.cmp(right).is_gt()),
            _ => None,
        },
        _ => None,
    }
}

/// Equality that no numeric ladder decides: `None` when either operand is a
/// number or a 64-bit cdata, which take the numeric ladder. Otherwise the
/// answer: the raw comparison, except that two distinct tables or two distinct
/// userdata may consult `__eq`, which is `Some(None)`.
// The three cases are the comparison protocol: not mine, decided, or `__eq`.
#[allow(clippy::option_option)]
#[inline]
fn plain_equal(left: &Value, right: &Value) -> Option<Option<bool>> {
    let numeric = |value: &Value| {
        matches!(
            value,
            Value::Number(_) | Value::Integer(_) | Value::Int64(_) | Value::UInt64(_)
        )
    };
    if numeric(left) || numeric(right) {
        return None;
    }
    Some(match (left, right) {
        (Value::String(left), Value::String(right)) => Some(left == right),
        (Value::Table(_), Value::Table(_))
        | (Value::File(_) | Value::Userdata(_), Value::File(_) | Value::Userdata(_)) => {
            crate::runtime::Runtime::raw_equal(left, right).then_some(true)
        }
        _ => Some(crate::runtime::Runtime::raw_equal(left, right)),
    })
}

/// The equality test of two distinct tables that [`test_slow`] left open:
/// unequal when neither has a metatable, since `__eq` then cannot run in any
/// profile. `None` leaves every other case to the general path.
#[cold]
#[inline(never)]
fn table_test_slow(
    runtime: &crate::runtime::Runtime,
    comparison: OpCode,
    left: &Value,
    right: &Value,
) -> Option<bool> {
    (comparison == OpCode::Equal
        && matches!((left, right), (Value::Table(_), Value::Table(_)))
        && matches!(runtime.raw_metatable(left), Ok(None))
        && matches!(runtime.raw_metatable(right), Ok(None)))
    .then_some(false)
}

/// [`test_slow`] for a constant operand, read as `LoadConstant` loads it. It
/// is the right operand, or the left when `swapped`.
#[cold]
#[inline(never)]
fn test_k_slow(
    comparison: OpCode,
    value: &Value,
    constant: &Constant,
    swapped: bool,
    compatibility: CompatibilityProfile,
    float_arith: FloatArith,
) -> Option<bool> {
    // No constant is a table or a userdata, so only a numeric constant or
    // operand needs the ladder; any other pair compares raw.
    if comparison == OpCode::Equal
        && !matches!(constant, Constant::Integer(_) | Constant::Number(_))
        && !matches!(value, Value::Number(_) | Value::Integer(_))
        && !matches!(constant, Constant::Int64(_) | Constant::UInt64(_))
        && !matches!(value, Value::Int64(_) | Value::UInt64(_))
    {
        return Some(match (value, constant) {
            (Value::Nil, Constant::Nil) => true,
            (Value::Boolean(value), Constant::Boolean(other)) => value == other,
            (Value::String(value), Constant::String(other)) => value == other,
            _ => false,
        });
    }
    let constant = constant_value(constant, compatibility);
    if swapped {
        test_slow(comparison, &constant, value, float_arith)
    } else {
        test_slow(comparison, value, &constant, float_arith)
    }
}

/// Arithmetic and comparisons where both operands are 64-bit integer cdata
/// or one is and the other is a number, with `cint64::arith` semantics and
/// none of `numeric_scalar_result`'s ladders. `None` for any other pair.
#[inline(always)]
fn cint64_fast(opcode: OpCode, left: &Value, right: &Value) -> Option<Value> {
    // The operands as bits and signedness, as `cint64::common_operands`
    // converts them: unsigned when either side is, numbers by truncation.
    let (a, b, unsigned) = match (left, right) {
        (Value::UInt64(a), Value::UInt64(b)) => (*a, *b, true),
        (Value::Int64(a), Value::Int64(b)) => (a.cast_unsigned(), b.cast_unsigned(), false),
        (Value::UInt64(a), Value::Int64(b)) => (*a, b.cast_unsigned(), true),
        (Value::Int64(a), Value::UInt64(b)) => (a.cast_unsigned(), *b, true),
        (Value::UInt64(a), Value::Number(b)) => (*a, cint64::number_to_u64(*b), true),
        (Value::Number(a), Value::UInt64(b)) => (cint64::number_to_u64(*a), *b, true),
        (Value::Int64(a), Value::Number(b)) => (
            a.cast_unsigned(),
            cint64::number_to_i64(*b).cast_unsigned(),
            false,
        ),
        (Value::Number(a), Value::Int64(b)) => (
            cint64::number_to_i64(*a).cast_unsigned(),
            b.cast_unsigned(),
            false,
        ),
        _ => return None,
    };
    let bits = match opcode {
        OpCode::Add => a.wrapping_add(b),
        OpCode::Subtract => a.wrapping_sub(b),
        OpCode::Multiply => a.wrapping_mul(b),
        // Division and modulo by zero give 2^63, and the one overflowing
        // signed quotient INT64_MIN, as in `cint64::arith`.
        OpCode::Divide if unsigned => a.checked_div(b).unwrap_or(1 << 63),
        OpCode::Modulo if unsigned => a.checked_rem(b).unwrap_or(1 << 63),
        OpCode::Divide => a
            .cast_signed()
            .checked_div(b.cast_signed())
            .unwrap_or(i64::MIN)
            .cast_unsigned(),
        OpCode::Modulo if b == 0 => 1 << 63,
        OpCode::Modulo => a
            .cast_signed()
            .wrapping_rem(b.cast_signed())
            .cast_unsigned(),
        OpCode::Equal => return Some(Value::Boolean(a == b)),
        OpCode::LessThan => {
            return Some(Value::Boolean(if unsigned {
                a < b
            } else {
                a.cast_signed() < b.cast_signed()
            }));
        }
        OpCode::LessEqual => {
            return Some(Value::Boolean(if unsigned {
                a <= b
            } else {
                a.cast_signed() <= b.cast_signed()
            }));
        }
        _ => {
            let (left, right) = if unsigned {
                (CInt64::Unsigned(a), CInt64::Unsigned(b))
            } else {
                (
                    CInt64::Signed(a.cast_signed()),
                    CInt64::Signed(b.cast_signed()),
                )
            };
            return cint64_other(opcode, left, right);
        }
    };
    Some(if unsigned {
        Value::UInt64(bits)
    } else {
        Value::Int64(bits.cast_signed())
    })
}

/// Division, modulo and power on converted operands; `None` for opcodes
/// 64-bit integers do not define, which the general path handles.
#[inline(never)]
fn cint64_other(opcode: OpCode, left: CInt64, right: CInt64) -> Option<Value> {
    let operation = match opcode {
        OpCode::Divide => CIntArith::Divide,
        OpCode::Modulo => CIntArith::Modulo,
        OpCode::Power => CIntArith::Power,
        _ => return None,
    };
    cint64::arith_converted(operation, left, right)
}

/// Arithmetic on operands that are not two floats or two integers, or an
/// integer operation that can fail: coercions, metamethods, errors. Out of
/// line so the loop keeps its locals in registers. The operator is a const
/// parameter, so each instance is specialized for its opcode and 64-bit
/// integer cdata (`LuaJIT` `ULL`/`LL`) take a short path.
#[cold]
#[inline(never)]
fn arithmetic_slow<const OP: u8>(
    left: &Value,
    right: &Value,
    float_arith: FloatArith,
) -> Option<Value> {
    let opcode = OpCode::try_from(OP).unwrap_or(OpCode::Add);
    if let Some(value) = cint64_fast(opcode, left, right) {
        return Some(value);
    }
    // A table operand has no primitive result: only a metamethod answers,
    // which the miss path enters. Trying the numeric ladders first cost a
    // hundred instructions on every operator handler call. Asked after the
    // 64-bit integers, which come here on every operation of theirs.
    if matches!(left, Value::Table(_)) || matches!(right, Value::Table(_)) {
        return None;
    }
    arithmetic_general(opcode, left, right, float_arith)
}

/// The `K` forms' operands that are not two floats: integers, mixed
/// numbers, coercions, metamethod candidates. The constant converts as
/// `LoadConstant` loads it.
#[cold]
#[inline(never)]
fn arithmetic_k_other(
    opcode: OpCode,
    left: &Value,
    constant: &Constant,
    compatibility: CompatibilityProfile,
    float_arith: FloatArith,
) -> Option<Value> {
    let right = constant_value(constant, compatibility);
    if let (Value::Integer(left), Value::Integer(right)) = (left, &right) {
        match opcode {
            OpCode::Add => return Some(Value::Integer(left.wrapping_add(*right))),
            OpCode::Subtract => return Some(Value::Integer(left.wrapping_sub(*right))),
            OpCode::Multiply => return Some(Value::Integer(left.wrapping_mul(*right))),
            OpCode::Modulo if *right != 0 => {
                return Some(Value::Integer(integer_floor_mod_nonzero(*left, *right)?));
            }
            _ => {}
        }
    }
    if let (Value::Number(left), Value::Number(right)) = (left, &right) {
        // Integer constants of profiles without integers.
        let (left, right) = (*left, *right);
        return Some(Value::Number(match opcode {
            OpCode::Add => left + right,
            OpCode::Subtract => left - right,
            OpCode::Multiply => left * right,
            OpCode::Divide => left / right,
            _ => float_arith.modulo(left, right),
        }));
    }
    if let Some(value) = cint64_fast(opcode, left, &right) {
        return Some(value);
    }
    // As in `arithmetic_slow`: a table has no primitive result.
    if matches!(left, Value::Table(_)) {
        return None;
    }
    arithmetic_general(opcode, left, &right, float_arith)
}

#[inline(never)]
fn arithmetic_general(
    opcode: OpCode,
    left: &Value,
    right: &Value,
    float_arith: FloatArith,
) -> Option<Value> {
    // A string operand converts on the general path.
    if matches!(left, Value::String(_)) || matches!(right, Value::String(_)) {
        return None;
    }
    fast_scalar(opcode, left, right, float_arith)
}

/// Primitive numeric arithmetic and comparison, or `None` for anything the
/// general scalar path must handle (coercions, metamethods, errors).
#[inline(always)]
#[allow(clippy::float_cmp)] // Lua number equality is exact.
fn fast_scalar(
    opcode: OpCode,
    left: &Value,
    right: &Value,
    float_arith: FloatArith,
) -> Option<Value> {
    if let (Value::Number(left), Value::Number(right)) = (left, right) {
        let (left, right) = (*left, *right);
        return Some(match opcode {
            OpCode::Add => Value::Number(left + right),
            OpCode::Subtract => Value::Number(left - right),
            OpCode::Multiply => Value::Number(left * right),
            OpCode::Divide => Value::Number(left / right),
            OpCode::Modulo => Value::Number(float_arith.modulo(left, right)),
            OpCode::Equal => Value::Boolean(left == right),
            OpCode::LessThan => Value::Boolean(left < right),
            OpCode::LessEqual => Value::Boolean(left <= right),
            OpCode::Negate => Value::Number(-left),
            _ => {
                return numeric_scalar_result(
                    opcode,
                    &Value::Number(left),
                    &Value::Number(right),
                    float_arith,
                )
                .ok()
                .flatten();
            }
        });
    }
    if opcode == OpCode::Negate {
        return match left {
            Value::Number(value) => Some(Value::Number(-*value)),
            Value::Integer(value) => Some(Value::Integer(value.wrapping_neg())),
            _ => None,
        };
    }
    if let (Value::Integer(left), Value::Integer(right)) = (left, right) {
        let (left, right) = (*left, *right);
        match opcode {
            OpCode::Add => return Some(Value::Integer(left.wrapping_add(right))),
            OpCode::Subtract => return Some(Value::Integer(left.wrapping_sub(right))),
            OpCode::Multiply => return Some(Value::Integer(left.wrapping_mul(right))),
            OpCode::Equal => return Some(Value::Boolean(left == right)),
            OpCode::LessThan => return Some(Value::Boolean(left < right)),
            OpCode::LessEqual => return Some(Value::Boolean(left <= right)),
            _ => {}
        }
    }
    numeric_scalar_result(opcode, left, right, float_arith)
        .ok()
        .flatten()
}

/// The frame at `index` and the frame below it, for naming the function in
/// an error. Error paths only.
fn frame_and_parent(frames: &[Frame], index: usize) -> (&Frame, Option<&Frame>) {
    (&frames[index], index.checked_sub(1).map(|i| &frames[i]))
}
