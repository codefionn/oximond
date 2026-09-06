//! `pcall` and `xpcall` as protected frames of a trace.
//!
//! `pcall(f, ...)` records `f` as an inlined call that is marked protected
//! (see [`oximond::jit_ir::trace::Protected`]). The trace runs `f`'s operations
//! in place, and its return delivers `true` followed by `f`'s results, the way
//! the interpreter's protected return does. Every exit inside `f` rebuilds the
//! frames the interpreter has there, `f`'s frame returning to the protected
//! call of its caller, so an error the interpreter raises after the exit is
//! caught by that `pcall`. A native `f` (a builtin the trace inlines or calls
//! through a helper) is recorded as that call with `true` written ahead of its
//! results. Nothing of the protected call exists after it, so no frame is
//! rebuilt for it.
//!
//! The guards that pin `pcall` or `xpcall`, the `xpcall` handler and `f` leave
//! through the `Before` snapshot of the call. A helper that misses does the
//! same, and the interpreter makes the call itself and raises whatever error
//! it raises. An `error` or a failed `assert` inside the region ends in a
//! caught exit (see [`super::inline_caught`]) when the protected call is the
//! root frame's. Any other recording that meets one aborts, and a trace that
//! leaves because one is about to run exits before it.

use super::{
    Flow, HookPoint, Open, Operand, RecKind, Recorder, Sym, TraceAbortReason, TraceScalar,
    TraceType,
};
use crate::compat::CompatibilityProfile;
use crate::jit_ir::trace::{
    PINNED_BUILTIN_DEBUG_TRACEBACK, PINNED_BUILTIN_PCALL, PINNED_BUILTIN_XPCALL, Protected,
    pinned_builtin_identity, pinned_native_identity,
};
use crate::runtime::{BuiltinFunction, ClosureHandle, Value};

/// What recording a call of `pcall` or `xpcall` decided.
pub(super) enum ProtectedTarget {
    /// The protected function is a native one, and the whole call is
    /// recorded.
    Done(Flow),
    /// The protected function is the Lua closure `handle`, whose operand is
    /// `target`. The caller records the inlined call. `arguments` holds the
    /// callee's arguments.
    Closure {
        handle: ClosureHandle,
        target: Operand,
        protected: Protected,
    },
}

impl Recorder {
    /// Records the call of `pcall` or `xpcall` at `at.pc` whose arguments
    /// are `arguments`: the pins and their guards, then the call of the
    /// protected function. For a Lua closure the leading `f` (and `xpcall`'s
    /// handler) leave `arguments`. A native `f` is recorded in place.
    pub(super) fn record_protected_call(
        &mut self,
        at: &HookPoint<'_>,
        (builtin, callee): (BuiltinFunction, Operand),
        arguments: &mut Vec<(Operand, TraceType)>,
        (first_argument, fixed, with_varargs): (usize, usize, bool),
        (base, results): (usize, usize),
    ) -> Result<ProtectedTarget, TraceAbortReason> {
        let pc = at.pc;
        let extended = builtin == BuiltinFunction::ExtendedProtectedCall;
        let leading = 1 + usize::from(extended);
        if arguments.len() < leading
            || arguments[..leading]
                .iter()
                .any(|(_, ty)| *ty != TraceType::Closure)
        {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        // Lua 5.1's `xpcall(f, handler)` calls `f` with no arguments.
        let drops_arguments = extended && self.xpcall_drops_arguments;
        if drops_arguments {
            if with_varargs {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            arguments.truncate(leading);
        }
        let target = &at.registers[first_argument];
        // Nothing is guarded until the shape of the whole call is known.
        let handler = if extended {
            let (operand, _) = arguments[1];
            let identity = match &at.registers[first_argument + 1] {
                Value::Closure(handle) => handle.native_identity(),
                Value::NativeFunction(function) => pinned_native_identity(function.index()),
                Value::BuiltinFunction(BuiltinFunction::DebugTraceback) => {
                    pinned_builtin_identity(PINNED_BUILTIN_DEBUG_TRACEBACK)
                }
                _ => {
                    self.note_callable(&at.registers[first_argument + 1]);
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
            };
            Some((operand, identity))
        } else {
            None
        };
        if !matches!(
            target,
            Value::Closure(_)
                | Value::NativeFunction(_)
                | Value::BuiltinFunction(BuiltinFunction::ToString | BuiltinFunction::StringFormat)
        ) {
            self.note_callable(target);
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let identity = pinned_builtin_identity(if extended {
            PINNED_BUILTIN_XPCALL
        } else {
            PINNED_BUILTIN_PCALL
        });
        let lane = self.pin(identity)?;
        self.guard_callee(pc, callee, lane);
        let protected = match handler {
            Some((operand, identity)) => {
                let handler = self.pin(identity)?;
                self.guard_callee(pc, operand, handler);
                Protected::Xpcall { handler }
            }
            None => Protected::Pcall,
        };
        let (operand, _) = arguments[0];
        match target {
            Value::Closure(handle) => {
                let handle = *handle;
                arguments.drain(..leading);
                Ok(ProtectedTarget::Closure {
                    handle,
                    target: operand,
                    protected,
                })
            }
            native => {
                if with_varargs {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                // Lua 5.1's `xpcall` passes the native no arguments.
                let fixed = if drops_arguments { leading } else { fixed };
                let flow = self.record_protected_native_call(
                    at,
                    (native, operand),
                    (first_argument, fixed, leading),
                    (base, results),
                );
                if flow.is_err() {
                    self.note_callable(target);
                }
                flow.map(ProtectedTarget::Done)
            }
        }
    }

    /// Records the call of the native `native` that `pcall` or `xpcall` makes
    /// with the `fixed - leading` arguments after the `leading` ones.
    fn record_protected_native_call(
        &mut self,
        at: &HookPoint<'_>,
        (native, operand): (&Value, Operand),
        (first_argument, fixed, leading): (usize, usize, usize),
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let arguments_at = first_argument + leading;
        let count = fixed - leading;
        self.record_protected_native(pc, (base, results), |recorder, window| {
            if let Value::NativeFunction(function) = native
                && let Some(flow) = recorder.record_special_native(
                    at,
                    (operand, *function),
                    (arguments_at, count),
                    window,
                    (false, false, false),
                )
            {
                return flow;
            }
            // The rest of the arguments, as an unprotected call
            // reads them.
            let host = Self::is_host_callee(at, first_argument);
            let mut args = Vec::with_capacity(count);
            for index in 0..count {
                args.push(if host {
                    recorder.host_operand(pc, at.registers, arguments_at + index)?
                } else {
                    recorder.read(at.registers, arguments_at + index)?
                });
            }
            match native {
                Value::NativeFunction(function) => recorder.record_native_call(
                    at,
                    operand,
                    *function,
                    (&args, arguments_at),
                    window,
                    false,
                ),
                Value::BuiltinFunction(
                    builtin @ (BuiltinFunction::ToString | BuiltinFunction::StringFormat),
                ) => recorder.record_host_builtin(at, *builtin, operand, &args, window),
                _ => Err(TraceAbortReason::UnsupportedOperand),
            }
        })
    }

    /// Records the call of a native function that `pcall` makes, with
    /// `record` given the result window after the `true` that leads the
    /// protected call's results.
    fn record_protected_native(
        &mut self,
        pc: usize,
        (base, results): (usize, usize),
        record: impl FnOnce(&mut Self, (usize, usize)) -> Result<Flow, TraceAbortReason>,
    ) -> Result<Flow, TraceAbortReason> {
        // The call keeps `results` values: `true` and `results - 1` of the
        // callee's. A call that keeps only `true` leaves the register after
        // it alone, which the native call writes and this puts back.
        self.check_register(base + 1)?;
        let saved = (results == 1).then(|| self.save_register(base + 1));
        let kept = if results == 0 {
            0
        } else {
            (results - 1).max(1)
        };
        let flow = record(self, (base + 1, kept))?;
        if let Some(saved) = saved {
            self.restore_register(base + 1, saved);
        }
        let yes = self.emit(
            pc,
            RecKind::Constant(TraceScalar::Boolean(true)),
            Some(TraceType::Boolean),
        );
        if results == 0 {
            self.open
                .as_mut()
                .ok_or(TraceAbortReason::Divergence)?
                .insert(0, Open::Op(yes));
        } else {
            self.write(base, yes);
        }
        Ok(flow)
    }

    /// A register of the running frame as it stands, so that a write can be
    /// undone.
    fn save_register(&self, register: usize) -> (Sym, bool) {
        if let Some(frame) = self.frames.last() {
            (frame.regs[register], false)
        } else {
            (self.regs[register], self.written[register])
        }
    }

    fn restore_register(&mut self, register: usize, (sym, written): (Sym, bool)) {
        if let Some(frame) = self.frames.last_mut() {
            frame.regs[register] = sym;
        } else {
            self.regs[register] = sym;
            self.written[register] = written;
        }
    }
}

/// Whether `profile`'s `xpcall` takes no arguments for the protected function.
pub(super) const fn xpcall_drops_arguments(profile: CompatibilityProfile) -> bool {
    matches!(profile, CompatibilityProfile::Lua51)
}
