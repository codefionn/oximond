//! `error` and a failed `assert` inside a protected call of the same trace.
//!
//! The interpreter raises the error, unwinds to the nearest `pcall` or
//! `xpcall` and carries on after it, with `false` and the error value in the
//! call's result registers. The trace does not unwind. It builds the error
//! value the way the interpreter does, drops the protected call's frames and
//! writes `false` and the value where the call's results go. Then it leaves
//! through a guard that always fails, with a snapshot of that state: the
//! interpreter resumes right after the `pcall`, the frames the call made
//! gone. The guard counts like any other exit, so a hot error path grows a
//! side trace that starts after the call and runs on natively.
//!
//! The error value is the interpreter's, bit for bit. A string raised by
//! `error` at level 1 or 2, and the message of a failed `assert`, get the
//! `chunk:line: ` prefix of the function at that level. The recorder knows
//! the pc of every frame, so the prefix is a constant: a message that is a
//! constant becomes one constant string, any other message goes through the
//! `concat` helper behind the prefix. Level 0 and every value that is not a
//! string pass through unchanged. The recorder checks its prediction against
//! the register the interpreter fills after it has caught the error (see
//! [`Recorder::check_caught`]), and drops the recording if they differ.
//!
//! The recorder keeps going while the interpreter unwinds, so the recording
//! tells the VM that an error is expected (see `Recorder::expects_error`).
//! Anything the trace cannot reproduce exactly keeps the old behavior: the
//! recording ends at the error, and a trace that reaches one leaves before
//! it. That covers a debug hook (the recording refuses it), a to-be-closed
//! variable or a closure over a local of a dropped frame (the recording
//! refuses the instructions that make them, so no discarded frame holds
//! one), an error inside a re-executed region or a metamethod or helper (the
//! recording sees no `error` there), a handler that is not a Lua function
//! (`debug.traceback` needs the stack the error was raised on), a protected
//! call with open results, an `xpcall` that an inlined function made, and the
//! levels and values the recorder does not predict.
//!
//! A `pcall` made by an inlined function is caught the same way, except that
//! the trace does not leave after it. The exit would rebuild the inlined
//! function's frame, and a side trace continues from the root frame alone,
//! so it could never take that exit over. The recording goes on in the
//! function after the call instead, and the error path runs in the trace. An
//! `xpcall` made by an inlined function keeps the old behavior: its handler
//! region starts over at the snapshot of the outermost call.
//!
//! An `xpcall` is the same up to the handler, which the interpreter runs as
//! a call after the unwinding. The recording follows it as a call of the
//! Lua function pinned with the `xpcall`, in a region that starts over at
//! the `xpcall` when anything in it leaves the trace (the interpreter then
//! raises the error and runs the handler itself). That is only exact while
//! the protected call has stored nothing. When the handler returns, its
//! first value goes where the results go, and the exit follows.

use super::{
    Flow, HookPoint, Inlined, MAX_TRACE_INLINED_CALLS, Open, Operand, PrevPc, RecKind, Recorder,
    SnapshotPhase, Sym, TraceAbortReason, TraceScalar, TraceType,
};
use crate::bytecode::{Chunk, Instruction, OpCode};
use crate::compat::CompatibilityProfile;
use crate::jit_ir::trace::{
    Entry, FrameId, HelperId, InlinedFrame, MAX_TRACE_FRAME_DEPTH, Protected, TraceFrameInfo,
    pinned_builtin_kind, pinned_native_identity, pinned_native_index,
};
use crate::runtime::{ClosureHandle, NativeFunctionHandle, Runtime, Value};
use crate::vm::errors::{display_chunk_source, value_to_error_text};
use std::sync::Arc;

/// What the interpreter must show after it caught the error the recording
/// predicted.
#[derive(Debug)]
pub(super) enum CaughtCheck {
    /// A `pcall` caught it: `false` in `base` and, when the call keeps a
    /// second result, `expected` in `base + 1`.
    Result {
        base: usize,
        results: usize,
        expected: Value,
    },
    /// An `xpcall` caught it: its message handler starts with `argument` as
    /// its first register.
    Handler { argument: Value },
}

/// How the profile's `assert` turns its second argument into the error value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AssertStyle {
    /// Lua 5.1 and 5.2: `luaL_optstring`, so nil is the default message, a
    /// number is text and any other value an argument error.
    StringMessage,
    /// `LuaJIT`: a string or number is a message with a position, any other
    /// value (nil included) is raised unchanged.
    LuaJit,
    /// Lua 5.3 and later: `error` at level 1 with the value, whatever it is.
    Passthrough,
}

impl AssertStyle {
    pub(super) fn of(profile: CompatibilityProfile) -> Self {
        match profile {
            CompatibilityProfile::Lua51 | CompatibilityProfile::Lua52 => Self::StringMessage,
            CompatibilityProfile::LuaJit21 => Self::LuaJit,
            _ => Self::Passthrough,
        }
    }
}

/// The message the stock `assert` gives a falsy value that carries none.
const ASSERT_DEFAULT: &[u8] = b"assertion failed!";

impl Recorder {
    /// Whether the VM should let the recording survive the error the
    /// interpreter is about to raise.
    pub(crate) fn expects_error(&self) -> bool {
        self.caught.is_some()
    }

    /// Checks the interpreter's state after it caught the predicted error
    /// against the recording's. The hook is at the instruction after the
    /// protected call.
    pub(super) fn check_caught(&mut self, at: &HookPoint<'_>) -> Result<(), TraceAbortReason> {
        let status = |base: usize| {
            at.registers
                .get(base)
                .is_some_and(|status| matches!(status, Value::Boolean(false)))
        };
        let matches = match self.caught.take() {
            None => true,
            Some(CaughtCheck::Result {
                base,
                results,
                expected,
            }) => {
                status(base)
                    && (results < 2
                        || at
                            .registers
                            .get(base + 1)
                            .is_some_and(|value| Runtime::raw_equal(value, &expected)))
            }
            Some(CaughtCheck::Handler { argument }) => {
                at.chunk.parameter_count == 0
                    || at
                        .registers
                        .first()
                        .is_some_and(|value| Runtime::raw_equal(value, &argument))
            }
        };
        // The handler returned `false` and its value to the call's results.
        let handled = self.handled.take().is_none_or(|(base, _)| status(base));
        if matches && handled {
            Ok(())
        } else {
            Err(TraceAbortReason::Divergence)
        }
    }

    /// The index in `frames` of the protected frame an error raised now
    /// reaches, if the trace holds it and can stand in for the unwinding.
    fn catching_frame(&self) -> Option<usize> {
        if self.reexec.is_some() || self.function.is_some() {
            return None;
        }
        let index = self
            .frames
            .iter()
            .rposition(|frame| frame.protected != Protected::No)?;
        // A protected call made by an inlined function: the trace carries on
        // after it (see `catch`), because a side trace continues from the
        // root frame alone and could not start after a call that a rebuilt
        // frame made. An `xpcall` needs the root frame: its handler region
        // starts over at the snapshot of the outermost call.
        if index != 0 && self.frames[index].protected != Protected::Pcall {
            return None;
        }
        let frame = &self.frames[index];
        // Open results are not a register the check can read.
        let fixed = frame.results > 0;
        // Tail calls change the levels Lua 5.1 counts.
        let plain = self.frames[index..]
            .iter()
            .all(|frame| frame.tail_calls == 0);
        let handled = match frame.protected {
            Protected::No => false,
            Protected::Pcall => true,
            // The handler runs as a recorded call that starts over in the
            // interpreter when anything in it leaves the trace, which only
            // works while the protected call has stored nothing since it
            // began.
            Protected::Xpcall { handler } => {
                self.pinned.get(handler as usize).is_some_and(|identity| {
                    pinned_native_index(*identity).is_none()
                        && pinned_builtin_kind(*identity).is_none()
                }) && self
                    .call_site
                    .is_some_and(|(_, stores)| stores == self.stores)
            }
        };
        (fixed && plain && handled).then_some(index)
    }

    /// Whether the call at `at.pc` calls `function` itself, not as the
    /// function a protected native call makes.
    fn calls_directly(at: &HookPoint<'_>, function: NativeFunctionHandle) -> bool {
        let Some(instruction) = at.chunk.code.get(at.pc) else {
            return false;
        };
        let callee = match instruction.opcode() {
            Ok(OpCode::CallOne) => instruction.b(),
            _ => instruction.a(),
        };
        matches!(
            at.registers.get(usize::from(callee)),
            Some(Value::NativeFunction(live)) if *live == function
        )
    }

    /// The `chunk:line: ` the interpreter puts before a message raised at
    /// `level` (0, 1 or 2) by the instruction at `at.pc`, or `None` when it
    /// adds nothing.
    fn position_prefix(
        &self,
        at: &HookPoint<'_>,
        level: usize,
    ) -> Result<Option<String>, TraceAbortReason> {
        let position = |chunk: &Arc<Chunk>, pc: usize| {
            chunk.debug.line_at(pc).map(|line| {
                format!(
                    "{}:{line}: ",
                    display_chunk_source(chunk.debug.source_name())
                )
            })
        };
        match level {
            0 => Ok(None),
            1 => Ok(position(at.chunk, at.pc)),
            2 => {
                let running = self.frames.len() - 1;
                // The function a `pcall` called has no Lua caller.
                if self.frames[running].protected != Protected::No {
                    return Ok(None);
                }
                let (chunk, pc) = if running == 0 {
                    (self.root_chunk(), self.frames[0].call_pc)
                } else {
                    (
                        &self.frames[running - 1].chunk,
                        self.frames[running].call_pc,
                    )
                };
                Ok(position(chunk, pc))
            }
            _ => Err(TraceAbortReason::UnsupportedOperand),
        }
    }

    /// The level argument of `error`: nil or absent is 1, anything else has
    /// to be a constant the recording can see.
    fn error_level(
        &self,
        at: &HookPoint<'_>,
        (first_argument, fixed): (usize, usize),
    ) -> Result<usize, TraceAbortReason> {
        if fixed < 2 {
            return Ok(1);
        }
        let register = first_argument + 1;
        let live = match &at.registers[register] {
            Value::Nil => return Ok(1),
            Value::Integer(level) => *level,
            #[allow(clippy::cast_possible_truncation, reason = "a small whole number")]
            Value::Number(level) if level.fract() == 0.0 && level.abs() < 1.0e6 => *level as i64,
            _ => return Err(TraceAbortReason::UnsupportedOperand),
        };
        let constant = match self.sym(register) {
            Sym::Op(index) => self.constant_scalar(Operand::Op(index)),
            _ => None,
        };
        let known = match constant {
            Some(TraceScalar::Integer(level)) => level,
            #[allow(clippy::cast_possible_truncation, reason = "a small whole number")]
            Some(TraceScalar::Float(level)) if level.fract() == 0.0 && level.abs() < 1.0e6 => {
                level as i64
            }
            Some(TraceScalar::Nil) => return Ok(1),
            _ => return Err(TraceAbortReason::UnsupportedOperand),
        };
        if known != live {
            return Err(TraceAbortReason::Divergence);
        }
        usize::try_from(known).map_err(|_| TraceAbortReason::UnsupportedOperand)
    }

    /// The constant an operand is, through moves.
    fn constant_scalar(&self, operand: Operand) -> Option<TraceScalar> {
        let Operand::Op(index) = self.origin(operand) else {
            return None;
        };
        match self.ops[index as usize].kind {
            RecKind::Constant(scalar) => Some(scalar),
            _ => None,
        }
    }

    /// A string constant of the roots with `text`, as a symbol.
    fn root_text(&mut self, text: &[u8]) -> Result<Sym, TraceAbortReason> {
        let slot = self.root_slot(text)?;
        Ok(Sym::Root(slot))
    }

    /// The error value for the string message in `register`, raised with
    /// `prefix` before it (`None` for none): the symbol, and the value the
    /// interpreter will hold.
    fn prefixed_message(
        &mut self,
        at: &HookPoint<'_>,
        register: usize,
        prefix: Option<&str>,
    ) -> Result<(Sym, Value), TraceAbortReason> {
        let pc = at.pc;
        let Value::String(live) = &at.registers[register] else {
            return Err(TraceAbortReason::Divergence);
        };
        let Some(prefix) = prefix else {
            return Ok((
                self.string_symbol(at, register)?,
                at.registers[register].clone(),
            ));
        };
        let mut text = prefix.as_bytes().to_vec();
        text.extend_from_slice(value_to_error_text(&at.registers[register]).as_bytes());
        let expected = Value::from(String::from_utf8_lossy(&text).into_owned());
        if self.constant_text(at.registers, register).is_some() {
            return Ok((self.root_text(&text)?, expected));
        }
        // A message the trace computes goes through `concat`. The
        // interpreter turns bytes that are no UTF-8 into replacement
        // characters, so the helper misses on them, and so does a record
        // time message that is none.
        if std::str::from_utf8(live.as_bytes()).is_err() {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let slot = self.root_slot(prefix.as_bytes())?;
        let head = self.root_operand(pc, slot);
        let message = self.host_operand(pc, at.registers, register)?;
        if !matches!(message.1, TraceType::String | TraceType::StringRef) {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let joined = self.emit_library_helper(
            pc,
            HelperId::CONCAT_ERROR_TEXT,
            &[head, message],
            TraceType::String,
        );
        Ok((Sym::Op(joined), expected))
    }

    /// The symbol of the string in `register`: a constant of the roots or an
    /// operation.
    fn string_symbol(
        &mut self,
        at: &HookPoint<'_>,
        register: usize,
    ) -> Result<Sym, TraceAbortReason> {
        Ok(self.argument_symbol(at.pc, at.registers, register)?.0)
    }

    /// Whether the call at `at.pc` is a tail call.
    fn is_tail_call(at: &HookPoint<'_>) -> bool {
        at.chunk.code.get(at.pc).is_some_and(|instruction| {
            matches!(
                instruction.opcode(),
                Ok(OpCode::TailCall | OpCode::TailCallVararg | OpCode::TailCallOpen)
            )
        })
    }

    /// Whether the call of `assert` at `at.pc` fails with a falsy condition
    /// that a protected call of the trace catches.
    pub(super) fn catches_assert(
        &self,
        at: &HookPoint<'_>,
        function: NativeFunctionHandle,
        (first_argument, fixed): (usize, usize),
        with_varargs: bool,
    ) -> bool {
        !with_varargs
            && fixed >= 1
            && !at.registers[first_argument].is_truthy()
            && Self::calls_directly(at, function)
            && self.catching_frame().is_some()
    }

    /// A call of `error` inside a protected call of the trace.
    pub(super) fn record_error(
        &mut self,
        at: &HookPoint<'_>,
        (callee, function): (Operand, NativeFunctionHandle),
        (first_argument, fixed): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        if !Self::calls_directly(at, function) {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let catching = self
            .catching_frame()
            .ok_or(TraceAbortReason::UnsupportedOperand)?;
        let level = self.error_level(at, (first_argument, fixed))?;
        let runtime = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?
            .runtime;
        let lane = self.pin(pinned_native_identity(function.index()))?;
        self.guard_callee(pc, callee, lane);
        let value = if fixed == 0 {
            None
        } else {
            Some(&at.registers[first_argument])
        };
        let (symbol, expected) = match value {
            None | Some(Value::Nil) => {
                let handled = matches!(self.frames[catching].protected, Protected::Xpcall { .. });
                self.raised_nil(pc, handled)?
            }
            Some(Value::String(_)) => {
                let prefix = self.position_prefix(at, level)?;
                self.prefixed_message(at, first_argument, prefix.as_deref())?
            }
            // A number becomes text with a position, in the profiles that
            // do it.
            Some(Value::Integer(_) | Value::Number(_))
                if level > 0 && runtime.error_number_positions() =>
            {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            Some(live) => (self.passed_symbol(at, first_argument)?, live.clone()),
        };
        self.catch(at, catching, (symbol, expected))
    }

    /// A call of a failed `assert` inside a protected call of the trace: the
    /// condition in `first_argument` is nil or false.
    pub(super) fn record_assert_failure(
        &mut self,
        at: &HookPoint<'_>,
        (callee, function): (Operand, NativeFunctionHandle),
        (first_argument, fixed): (usize, usize),
        (condition, ty): (Option<Operand>, TraceType),
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let catching = self
            .catching_frame()
            .ok_or(TraceAbortReason::UnsupportedOperand)?;
        let lane = self.pin(pinned_native_identity(function.index()))?;
        self.guard_callee(pc, callee, lane);
        // Only the falsy condition the recording saw reaches this code.
        match (ty, condition) {
            (TraceType::Boolean, Some(operand)) => self.guard_flag(pc, operand, false),
            (TraceType::Nil, _) => {}
            _ => return Err(TraceAbortReason::UnsupportedOperand),
        }
        // LuaJIT's tail call replaces the frame, so the message the library
        // raises (not an `error` call) has no position to take.
        let prefix = if self.assert_style == AssertStyle::LuaJit && Self::is_tail_call(at) {
            None
        } else {
            self.position_prefix(at, 1)?
        };
        let message = at.registers.get(first_argument + 1).filter(|_| fixed > 1);
        let handled = matches!(self.frames[catching].protected, Protected::Xpcall { .. });
        let default = |recorder: &mut Self| {
            let mut text = prefix.clone().unwrap_or_default().into_bytes();
            text.extend_from_slice(ASSERT_DEFAULT);
            let expected = Value::from(String::from_utf8_lossy(&text).into_owned());
            Ok((recorder.root_text(&text)?, expected))
        };
        let (symbol, expected) = match (self.assert_style, message) {
            (_, Some(Value::String(_))) => {
                self.prefixed_message(at, first_argument + 1, prefix.as_deref())?
            }
            (_, None) | (AssertStyle::StringMessage, Some(Value::Nil)) => default(self)?,
            // A number becomes text with a position, and any other value an
            // argument error naming the position as well.
            (AssertStyle::StringMessage, Some(_))
            | (AssertStyle::LuaJit, Some(Value::Integer(_) | Value::Number(_))) => {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            (_, Some(Value::Nil)) => self.raised_nil(pc, handled)?,
            // Any other value is raised unchanged, a table the trace made
            // included.
            (_, Some(live)) => (self.passed_symbol(at, first_argument + 1)?, live.clone()),
        };
        self.catch(at, catching, (symbol, expected))
    }

    /// `error(nil)` and `error()`: nil, which Lua 5.5 turns into a message
    /// unless an `xpcall` handler gets it.
    fn raised_nil(&mut self, pc: usize, handled: bool) -> Result<(Sym, Value), TraceAbortReason> {
        if self.no_error_object && !handled {
            let text = b"<no error object>";
            return Ok((self.root_text(text)?, Value::from("<no error object>")));
        }
        let nil = self.emit(
            pc,
            RecKind::Constant(TraceScalar::Nil),
            Some(TraceType::Nil),
        );
        Ok((Sym::Op(nil), Value::Nil))
    }

    /// The symbol of a value `error` passes on unchanged.
    fn passed_symbol(
        &mut self,
        at: &HookPoint<'_>,
        register: usize,
    ) -> Result<Sym, TraceAbortReason> {
        self.check_register(register)?;
        match self.sym(register) {
            // A table the trace made reaches the caller's register.
            Sym::Virtual(id) => Ok(Sym::Virtual(id)),
            Sym::Root(slot) => Ok(Sym::Root(slot)),
            _ => Ok(self.argument_symbol(at.pc, at.registers, register)?.0),
        }
    }

    /// Ends the protected call `catching` with the error value `symbol`
    /// (which the interpreter will show as `expected`): the frames go, the
    /// call's results are `false` and the value, and the trace leaves. For
    /// an `xpcall` the message handler runs first.
    fn catch(
        &mut self,
        at: &HookPoint<'_>,
        catching: usize,
        (symbol, expected): (Sym, Value),
    ) -> Result<Flow, TraceAbortReason> {
        let levels = self.frames.len() - catching;
        let handler = match self.frames[catching].protected {
            Protected::Xpcall { handler } => Some(handler),
            _ => None,
        };
        let before = self.call_site.map(|(snapshot, _)| snapshot);
        let protected = self
            .frames
            .drain(catching..)
            .next()
            .expect("a protected frame");
        if self.frames.is_empty() {
            self.call_site = None;
        }
        let (base, results) = (protected.result_base, protected.results);
        self.check_register(base + results - 1)?;
        if let Some(lane) = handler {
            let before = before.ok_or(TraceAbortReason::UnsupportedOperand)?;
            self.enter_handler(at, &protected, (lane, before), (symbol, expected))?;
            return Ok(Flow::CaughtHandler { levels });
        }
        // The table lands in the caller's register: a cell of the root frame
        // must not hold one.
        if matches!(symbol, Sym::Virtual(_)) && results >= 2 && !self.may_hold_virtual(base + 1) {
            return Err(TraceAbortReason::VirtualEscape);
        }
        // A table the trace made has no value outside it for a side trace to
        // start from, so the recording goes on through the caught path
        // instead of leaving there: the error path then runs in this trace.
        // So does a protected call of an inlined function: the exit would
        // rebuild that function's frame, and a side trace could not take
        // the exit over, which would then leave at every iteration.
        let leaves = !matches!(symbol, Sym::Virtual(_)) && catching == 0;
        self.deliver_failure(&protected, Some(symbol), leaves);
        let resume = protected.call_pc + 1;
        self.caught = Some(CaughtCheck::Result {
            base,
            results,
            expected,
        });
        Ok(Flow::Caught { resume, levels })
    }

    /// Writes `false` and `value` (nil for none) where the results of the
    /// protected call `protected` go. With `leaves`, the trace ends there
    /// with the guard that always fails: the interpreter resumes after the
    /// call.
    fn deliver_failure(&mut self, protected: &Inlined, value: Option<Sym>, leaves: bool) {
        let (base, results) = (protected.result_base, protected.results);
        // What follows belongs to the frame that made the protected call.
        let pc = protected.call_pc;
        let status = self.emit(
            pc,
            RecKind::Constant(TraceScalar::Boolean(false)),
            Some(TraceType::Boolean),
        );
        self.write(base, Sym::Op(status));
        let nil = (results > 2 || (results == 2 && value.is_none())).then(|| {
            self.emit(
                pc,
                RecKind::Constant(TraceScalar::Nil),
                Some(TraceType::Nil),
            )
        });
        if results > 1 {
            let second =
                value.unwrap_or_else(|| Sym::Op(nil.expect("a nil for the missing value")));
            self.write(base + 1, second);
        }
        for register in base + 2..base + results {
            self.write(register, Sym::Op(nil.expect("a nil for the rest")));
        }
        // The call has run: the guard that always fails leaves after it.
        self.set_prev(PrevPc::At(pc));
        if !leaves {
            return;
        }
        let caught = self.snapshot(
            SnapshotPhase::After {
                pc,
                successor: pc + 1,
            },
            &[],
        );
        let never = self.emit(
            pc,
            RecKind::Constant(TraceScalar::Boolean(false)),
            Some(TraceType::Boolean),
        );
        self.emit(
            pc,
            RecKind::GuardBranch {
                condition: Operand::Op(never),
                expected: true,
                snapshot: caught,
            },
            None,
        );
    }

    /// Starts the message handler of an `xpcall` that caught an error: a
    /// call of the Lua function pinned as `lane` with the error value, in a
    /// region that starts over at `before` (the snapshot of the outermost
    /// call) when anything in it leaves the trace. The interpreter then
    /// raises the error and runs the handler itself.
    fn enter_handler(
        &mut self,
        at: &HookPoint<'_>,
        protected: &Inlined,
        (lane, before): (u32, u32),
        (symbol, expected): (Sym, Value),
    ) -> Result<(), TraceAbortReason> {
        if matches!(symbol, Sym::Virtual(_)) {
            return Err(TraceAbortReason::VirtualEscape);
        }
        let runtime = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?
            .runtime;
        let identity = *self
            .pinned
            .get(lane as usize)
            .ok_or(TraceAbortReason::Divergence)?;
        let handle = ClosureHandle::from_native_identity(identity);
        let chunk = Arc::clone(
            runtime
                .closure_call_parts(handle)
                .ok_or(TraceAbortReason::Divergence)?
                .chunk,
        );
        if chunk.is_vararg || chunk.arg_register.is_some() {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        // A handler already on the stack of inlined calls is recursion.
        if Arc::ptr_eq(&chunk, &self.chunk)
            || self
                .activation
                .as_ref()
                .is_some_and(|activation| Arc::ptr_eq(&activation.chunk, &chunk))
            || self
                .frames
                .iter()
                .any(|frame| Arc::ptr_eq(&frame.chunk, &chunk))
        {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        if self.frames.len() + 2 > MAX_TRACE_FRAME_DEPTH
            || self.inlined.len() >= MAX_TRACE_INLINED_CALLS
        {
            return Err(super::limit("inlined calls/depth"));
        }
        let register_count = usize::from(chunk.register_count);
        let nil = self.emit(
            protected.call_pc,
            RecKind::Constant(TraceScalar::Nil),
            Some(TraceType::Nil),
        );
        let mut regs = vec![Sym::Op(nil); register_count];
        if chunk.parameter_count > 0 && register_count > 0 {
            regs[0] = symbol;
        }
        let call_pc = protected.call_pc;
        self.set_prev(PrevPc::At(call_pc));
        let parent = FrameId(self.frame_id());
        self.inlined.push(TraceFrameInfo {
            register_count: u16::try_from(register_count).expect("register count fits u16"),
            code_len: chunk.code.len(),
            inlined: Some(InlinedFrame {
                closure: lane,
                parent,
                call_pc,
                result_base: u8::try_from(protected.result_base).expect("register fits u8"),
                results: u8::try_from(protected.results).expect("count fits u8"),
                reexec: true,
                tail_calls: 0,
                tail_site: None,
                protected: Protected::No,
                entry: Entry::Call,
                dynamic: false,
            }),
            activation: None,
        });
        self.frames.push(Inlined {
            chunk,
            closure: lane,
            frame: u32::try_from(self.inlined.len()).expect("frame count fits u32"),
            regs,
            prev: PrevPc::Unset,
            call_pc,
            result_base: protected.result_base,
            results: protected.results,
            varargs: Vec::new(),
            owns_reexec: true,
            tail_calls: 0,
            protected: Protected::No,
            handler: true,
            entry: Entry::Call,
            after: super::Pending::Nothing,
            dynamic: None,
        });
        self.reexec = Some(before);
        self.caught = Some(CaughtCheck::Handler { argument: expected });
        Ok(())
    }

    /// The `Return` of the message handler of a caught `xpcall`: the call
    /// results in `false` and the handler's first value.
    pub(super) fn return_from_handler(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        count: usize,
    ) -> Result<Flow, TraceAbortReason> {
        let a = usize::from(instruction.a());
        // `ReturnOpen` returns the open results of the call before it after
        // its fixed values. Only the first value of all of them counts.
        let mut open = Vec::new();
        if instruction.opcode() == Ok(OpCode::ReturnOpen) {
            for value in self.open.take().ok_or(TraceAbortReason::Divergence)? {
                open.push(match value {
                    Open::Op(op) => Sym::Op(op),
                    Open::Root(slot) => Sym::Root(slot),
                    Open::Char(_) => return Err(TraceAbortReason::UnsupportedOperand),
                });
            }
        }
        let value = if count == 0 {
            open.first().copied()
        } else {
            Some(match self.sym(a) {
                sym @ Sym::Root(_) => sym,
                _ => match self.read(at.registers, a)?.0 {
                    Operand::Op(op) => Sym::Op(op),
                    Operand::Carried(_) => return Err(TraceAbortReason::Divergence),
                },
            })
        };
        // Lua 5.5 replaces a nil result by a message after the handler.
        let nil_result = match value {
            None => true,
            Some(Sym::Op(op)) => self.ops[op as usize].ty == Some(TraceType::Nil),
            Some(_) => false,
        };
        if self.no_error_object && nil_result {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let handler = self.frames.pop().expect("a handler frame returns");
        self.reexec = None;
        if self.frames.is_empty() {
            self.call_site = None;
        }
        self.check_register(handler.result_base + handler.results.max(1) - 1)?;
        self.deliver_failure(&handler, value, true);
        self.handled = Some((handler.result_base, handler.results));
        Ok(Flow::Leave {
            resume: handler.call_pc + 1,
        })
    }
}
