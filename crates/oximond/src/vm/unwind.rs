//! Frame return, close handling, and error unwinding through protected boundaries.

use crate::bytecode::Instruction;
use crate::runtime::{RuntimeError, Value};
use crate::vm::debug_info::capture_error_traceback;
use crate::vm::errors::vm_error_object;
use crate::vm::registers::write_call_results;
use crate::vm::stack::{CallStack, FrameMut, OwnedFrame};
use crate::vm::{
    CallReturn, DebugHookEvent, DebugHookReturn, DofileCompletion, FinalizerCompletion,
    ForeachCompletion, ForeachContinuation, Frame, GsubCompletion, IndexCompletion,
    ModuleCompletion, RequireCompletion, ReturnTarget, SortCompletion, SortContinuation,
    StringFormatCompletion, ToStringCompletion, Traceback, UnwindAction, UnwindClose,
    UnwindCompletion, Vm, VmError, VmResult,
};

impl Vm {
    /// A real return restores the caller's hook position even if only a
    /// count hook ran before the call. Hook callbacks keep their interrupted
    /// instruction's position instead.
    #[inline]
    pub(crate) fn restore_caller_hook_line(&self, frames: &mut CallStack) {
        if !self.restores_hook_line() {
            return;
        }
        let Some(mut frame) = frames.last_mut() else {
            return;
        };
        Self::restore_hook_line(&mut frame);
    }

    /// [`Self::restore_caller_hook_line`] for the record of the frame a
    /// return just made the top one. The counting fast loop holds that
    /// record, and the stack's view of its top frame costs as much again as
    /// the two stores.
    #[inline(never)]
    pub(crate) fn restore_record_hook_line(&self, frame: &mut Frame) {
        if self.restores_hook_line() {
            Self::restore_hook_line(frame);
        }
    }

    /// Whether a return restores the caller's hook position: the running
    /// thread has a hook that observes instructions, and no hook callback
    /// is running.
    #[inline(always)]
    fn restores_hook_line(&self) -> bool {
        // A live count lease means the running thread's count hook is
        // installed and idle: the common case under a count hook, answered
        // without the map lookup.
        let leased = self.debug_hook_depth == 0
            && self
                .count_lease
                .is_some_and(|lease| !lease.invalidated && lease.thread == self.current_coroutine);
        leased
            || !(self.debug_hooks.is_empty()
                || self.debug_hook_depth != 0
                || !self
                    .debug_hooks
                    .get(&self.current_coroutine)
                    .is_some_and(|hook| {
                        !hook.running
                            && (hook.mask.call
                                || hook.mask.return_
                                || hook.mask.line
                                || hook.count != 0)
                    }))
    }

    /// Sets the hook position of `frame` to the instruction it last ran.
    #[inline(always)]
    fn restore_hook_line(frame: &mut Frame) {
        let pc = frame.last_interpreted_pc.get();
        let line = pc.and_then(|pc| frame.chunk.debug.line_at(pc));
        let extras = frame.extras();
        extras.last_hook_pc = pc;
        extras.last_hook_line = line;
    }

    #[allow(clippy::too_many_lines)]
    // The two no-op targets sit at their `ReturnTarget` declaration positions;
    // merging the identical arms would reorder the match away from the enum.
    #[allow(clippy::match_same_arms)]
    pub(crate) fn finish_frame(
        &mut self,
        frames: &mut CallStack,
        values: Vec<Value>,
    ) -> VmResult<Option<Vec<Value>>> {
        // The returning frame's captured locals close while its registers
        // are still on the stack.
        self.close_top_frame_upvalues(frames);
        let caller = if frames
            .last()
            .expect("an executing frame exists")
            .extras
            .is_some()
        {
            let mut frame = frames.pop().expect("an executing frame exists");
            let caller = frame.caller.take();
            self.close_frame(frames, frame)?;
            caller
        } else {
            // Nothing to close: drop the record and registers in place.
            let top = frames.len() - 1;
            let frame = &mut frames[top];
            frame.clear_plain_mark();
            let returned = frame.caller.take();
            frames.discard_top();
            returned
        };
        let Some(caller) = caller else {
            if self.runtime.allocation_serial() != self.gc_observed_allocation_serial
                || self.runtime.gc_needs_collection()
            {
                self.gc_safepoint_with_values(frames, &values)?;
                self.gc_observed_allocation_serial = self.runtime.allocation_serial();
            }
            return Ok(Some(values));
        };
        if !matches!(caller, ReturnTarget::DebugHook(_)) {
            self.restore_caller_hook_line(frames);
        }
        match caller {
            ReturnTarget::Direct(caller) => {
                let mut caller_frame = frames.last_mut().expect("caller frame exists");
                write_call_results(
                    &mut caller_frame,
                    Instruction::call(caller.base, 0, caller.results),
                    values,
                )?;
            }
            ReturnTarget::Protected(caller) => {
                let mut caller_frame = frames.last_mut().expect("caller frame exists");
                self.write_protected_results(
                    &mut caller_frame,
                    caller.base,
                    caller.results,
                    true,
                    values,
                )?;
            }
            ReturnTarget::ExtendedProtectedTarget(continuation) => {
                self.finish_extended_protected_call(frames, continuation.target, true, values)?;
            }
            ReturnTarget::ExtendedProtectedHandler(target) => {
                self.finish_extended_protected_handler(
                    frames,
                    &target,
                    false,
                    vec![values.first().cloned().unwrap_or(Value::Nil)],
                )?;
            }
            ReturnTarget::TableForeach(continuation) => {
                let protected = matches!(continuation.completion, ForeachCompletion::Protected(_));
                let failure = continuation.clone();
                match self.resume_table_foreach(frames, *continuation, &values) {
                    Ok(()) => {}
                    Err(error) if protected => {
                        self.finish_protected_table_foreach_error(frames, &failure, error)?;
                    }
                    Err(error) => return Err(error),
                }
            }
            ReturnTarget::TableSort(continuation) => {
                let protected = matches!(continuation.completion, SortCompletion::Protected(_));
                let failure = continuation.clone();
                match self.resume_table_sort(frames, *continuation, &values) {
                    Ok(()) => {}
                    Err(error) if protected => {
                        self.finish_protected_table_sort_error(frames, &failure, error)?;
                    }
                    Err(error) => return Err(error),
                }
            }
            ReturnTarget::ProtectedSort(continuation) => {
                let mut values = values;
                values.insert(0, Value::Boolean(true));
                self.resume_table_sort(frames, *continuation, &values)?;
            }
            ReturnTarget::LoadReader(continuation) => {
                self.run_load_reader(frames, *continuation, Some(values))?;
            }
            ReturnTarget::Dofile(completion) => {
                self.finish_dofile(frames, completion, values)?;
            }
            ReturnTarget::Require(continuation) => {
                self.finish_require_result(frames, *continuation, values.first().cloned())?;
            }
            ReturnTarget::RequireSearch(search) => {
                let completion = search.completion;
                if let Err(error) = self.resume_require_search(frames, *search, values) {
                    let RequireCompletion::Protected(target) = completion else {
                        return Err(error);
                    };
                    let mut caller = frames.last_mut().expect("protected require caller exists");
                    self.write_protected_results(
                        &mut caller,
                        target.base,
                        target.results,
                        false,
                        vec![vm_error_object(&error)?],
                    )?;
                }
            }
            ReturnTarget::Module(continuation) => {
                let protected = matches!(continuation.completion, ModuleCompletion::Protected(_));
                let failure = continuation.completion;
                if let Err(error) = self.advance_module(frames, *continuation) {
                    if protected {
                        self.finish_protected_module_error(frames, failure, error)?;
                    } else {
                        return Err(error);
                    }
                }
            }
            ReturnTarget::StringGsub(continuation) => {
                self.resume_string_gsub(frames, *continuation, &values)?;
            }
            ReturnTarget::Index(continuation) => {
                self.resume_index(frames, *continuation, &values)?;
            }
            ReturnTarget::ProtectedIndex(continuation) => {
                self.complete_index(frames, *continuation, Value::Boolean(true))?;
            }
            ReturnTarget::SetMetamethod => {}
            ReturnTarget::IndexResult(destination) => {
                let frame = frames.last_mut().expect("metamethod caller exists");
                frame.registers[usize::from(destination)] =
                    values.first().cloned().unwrap_or(Value::Nil);
            }
            ReturnTarget::Operator(operator) => {
                self.finish_scalar(
                    frames,
                    operator.continuation(),
                    values.first().cloned().unwrap_or(Value::Nil),
                )?;
            }
            ReturnTarget::ToString(continuation) => {
                self.finish_tostring(
                    frames,
                    continuation.completion,
                    values.first().cloned().unwrap_or(Value::Nil),
                )?;
            }
            ReturnTarget::StringFormat(continuation) => {
                self.resume_string_format(
                    frames,
                    *continuation,
                    values.first().cloned().unwrap_or(Value::Nil),
                )?;
            }
            ReturnTarget::Scalar(continuation) => {
                self.finish_scalar(
                    frames,
                    *continuation,
                    values.first().cloned().unwrap_or(Value::Nil),
                )?;
            }
            ReturnTarget::ConcatRange(mut continuation) => {
                continuation.accumulator = values.first().cloned().unwrap_or(Value::Nil);
                self.resume_concat_range(frames, *continuation)?;
            }
            ReturnTarget::UserdataFinalizer(completion) => {
                self.advance_userdata_finalizers(frames, *completion)?;
            }
            ReturnTarget::CloseLocal => {}
            ReturnTarget::UnwindClose(continuation) => {
                if let UnwindAction::Return(error) =
                    self.advance_unwind_close(frames, *continuation)?
                {
                    return Err(error);
                }
            }
            ReturnTarget::DebugHook(hook_return) => {
                let DebugHookReturn {
                    thread,
                    remaining_tail_returns,
                } = *hook_return;
                self.debug_hook_depth = self.debug_hook_depth.saturating_sub(1);
                if let Some(hook) = self.debug_hooks.get_mut(&thread) {
                    hook.running = false;
                }
                if let Some(mut interrupted) = frames.last_mut() {
                    interrupted.extras().debug_hook_subject = None;
                    interrupted.extras().debug_hook_transfer = (0, 0);
                }
                if let Some(arguments) = self.pending_debug_sethook.take() {
                    let targets_current_thread = !matches!(arguments.first(), Some(Value::Thread(target)) if Some(*target) != self.current_coroutine);
                    self.debug_sethook(&arguments)?;
                    if targets_current_thread && let Some(mut interrupted) = frames.last_mut() {
                        let last_line = interrupted
                            .last_interpreted_pc
                            .get()
                            .and_then(|pc| interrupted.chunk.debug.line_at(pc));
                        let pc = interrupted.pc;
                        let extras = interrupted.extras();
                        extras.last_hook_line = last_line;
                        extras.skip_hook_pc_once = Some(pc);
                    }
                    self.fire_debug_hook(frames, DebugHookEvent::Return)?;
                }
                if remaining_tail_returns != 0 {
                    self.fire_debug_hook_with_tail_returns(
                        frames,
                        DebugHookEvent::TailReturn,
                        remaining_tail_returns - 1,
                    )?;
                }
            }
        }
        self.recheck_gc_after_frame_return(frames)?;
        Ok(None)
    }

    pub(crate) fn recheck_gc_after_frame_return(&mut self, frames: &mut CallStack) -> VmResult<()> {
        let Some(depth) = self.pending_gc_recheck_depth else {
            return Ok(());
        };
        if frames.len() >= depth {
            return Ok(());
        }
        self.pending_gc_recheck_depth = None;
        self.refresh_return_watch();
        if !self.runtime.gc_is_running() {
            return Ok(());
        }
        let roots = self.gc_roots_with_coroutines(frames, &[]);
        self.runtime.collect_full(roots);
        self.prune_collected_coroutines();
        self.advance_userdata_finalizers(frames, FinalizerCompletion::Automatic)
    }

    pub(crate) fn finish_single_return_frame(
        &mut self,
        frames: &mut CallStack,
        value: Value,
    ) -> VmResult<Option<Vec<Value>>> {
        let direct = frames.last().and_then(|frame| match frame.caller {
            Some(ReturnTarget::Direct(CallReturn { base, results: 1 })) => Some(base),
            _ => None,
        });
        let Some(base) = direct else {
            return self.finish_frame(frames, vec![value]);
        };

        self.close_top_frame_upvalues(frames);
        if frames
            .last()
            .expect("an executing frame exists")
            .extras
            .is_none()
        {
            // Nothing to close: retire the record for the next call at this
            // depth to reuse.
            frames.retire_top();
        } else {
            let frame = frames.pop().expect("an executing frame exists");
            self.close_frame(frames, frame)?;
        }
        self.restore_caller_hook_line(frames);
        let mut caller = frames.last_mut().expect("caller frame exists");
        caller.clear_open_results();
        *caller
            .registers
            .get_mut(usize::from(base))
            .ok_or(VmError::RegisterRange {
                start: base,
                count: 1,
            })? = value;
        self.recheck_gc_after_frame_return(frames)?;
        Ok(None)
    }

    pub(crate) fn finish_protected_table_foreach_error(
        &mut self,
        frames: &mut CallStack,
        continuation: &ForeachContinuation,
        error: VmError,
    ) -> VmResult<()> {
        let ForeachCompletion::Protected(target) = continuation.completion else {
            return Err(error);
        };
        let mut caller = frames
            .last_mut()
            .expect("table.foreach caller frame exists");
        self.write_protected_results(
            &mut caller,
            target.base,
            target.results,
            false,
            vec![vm_error_object(&error)?],
        )?;
        Ok(())
    }

    pub(crate) fn finish_protected_module_error(
        &mut self,
        frames: &mut CallStack,
        completion: ModuleCompletion,
        error: VmError,
    ) -> VmResult<()> {
        let ModuleCompletion::Protected(target) = completion else {
            return Err(error);
        };
        let mut caller = frames.last_mut().expect("module caller frame exists");
        self.write_protected_results(
            &mut caller,
            target.base,
            target.results,
            false,
            vec![vm_error_object(&error)?],
        )
    }

    pub(crate) fn finish_protected_table_sort_error(
        &mut self,
        frames: &mut CallStack,
        continuation: &SortContinuation,
        error: VmError,
    ) -> VmResult<()> {
        let SortCompletion::Protected(target) = continuation.completion else {
            return Err(error);
        };
        let mut caller = frames.last_mut().expect("table.sort caller frame exists");
        self.write_protected_results(
            &mut caller,
            target.base,
            target.results,
            false,
            vec![vm_error_object(&error)?],
        )?;
        Ok(())
    }

    /// Closes `frame`, which has already been popped from the stack whose
    /// remaining frames are `frames`.
    pub(crate) fn close_frame(
        &mut self,
        frames: &mut CallStack,
        frame: OwnedFrame,
    ) -> VmResult<()> {
        self.close_frame_with_error(frames, frame, &Value::Nil)
    }

    /// Closes a frame that a proper tail call replaces, retaining only its
    /// register allocation for the successor. This does every observable part
    /// of ordinary frame closure before the new activation starts.
    #[allow(
        clippy::manual_slice_fill,
        reason = "concrete nil resets avoid generic Value clone dispatch in measured tail-call loops"
    )]
    pub(crate) fn close_frame_for_tail_replacement(
        &mut self,
        frames: &mut CallStack,
        frame: &mut OwnedFrame,
    ) -> VmResult<Vec<Value>> {
        self.close_frame_contents(frames, &mut frame.view(), &Value::Nil)?;
        let mut registers = std::mem::take(&mut frame.registers);
        for register in &mut registers {
            *register = Value::Nil;
        }
        Ok(registers)
    }

    pub(crate) fn close_frame_with_error(
        &mut self,
        frames: &mut CallStack,
        mut frame: OwnedFrame,
        error: &Value,
    ) -> VmResult<()> {
        self.close_frame_contents(frames, &mut frame.view(), error)
    }

    pub(crate) fn close_frame_contents(
        &mut self,
        frames: &mut CallStack,
        frame: &mut FrameMut<'_>,
        error: &Value,
    ) -> VmResult<()> {
        // Ordinary calls close nothing. Skip the close-slot loop entirely
        // for them instead of constructing an iterator over an empty
        // collection on every return.
        let Some(extras) = frame.extras_ref() else {
            return Ok(());
        };
        if extras.close_slots.is_empty() {
            return Ok(());
        }
        while let Some(register) = frame
            .extras_ref()
            .and_then(|extras| extras.close_slots.last().copied())
        {
            self.close_local_slot(frames, frame, register, error.clone())?;
        }
        Ok(())
    }

    pub(crate) fn begin_error_unwind(
        &mut self,
        frames: &mut CallStack,
        error: VmError,
    ) -> VmResult<UnwindAction> {
        self.begin_error_unwind_mode(frames, error, false)
    }

    pub(crate) fn begin_coroutine_error_unwind(
        &mut self,
        frames: &mut CallStack,
        error: VmError,
    ) -> VmResult<UnwindAction> {
        let retain = matches!(
            self.compatibility,
            crate::compat::CompatibilityProfile::Lua54 | crate::compat::CompatibilityProfile::Lua55
        );
        self.begin_error_unwind_mode(frames, error, retain)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "Boundary selection and retained close/error state share one unwind pass"
    )]
    // Shared take-once unwind state between `ExtendedProtectedCall` clones on
    // the VM thread; `Vm` is `!Send` (it owns `Rc`s), so the `Arc` never crosses threads.
    #[allow(clippy::arc_with_non_send_sync)]
    fn begin_error_unwind_mode(
        &mut self,
        frames: &mut CallStack,
        error: VmError,
        retain_unprotected: bool,
    ) -> VmResult<UnwindAction> {
        if let Some(request) = error.exit_request() {
            return self.begin_exit_unwind(frames, error, request.close);
        }
        let finalizer_errors_warn = self.finalizer_errors_warn();
        let protected = frames.iter().rposition(|frame| {
            frame.caller.as_ref().is_some_and(|target| match target {
                ReturnTarget::Protected(_)
                | ReturnTarget::ExtendedProtectedTarget(_)
                | ReturnTarget::ExtendedProtectedHandler(_)
                | ReturnTarget::ProtectedIndex(_)
                | ReturnTarget::ProtectedSort(_)
                | ReturnTarget::LoadReader(_) => true,
                // Lua 5.4 runs every finalizer in a protected call.
                ReturnTarget::UserdataFinalizer(completion) => {
                    finalizer_errors_warn
                        || matches!(**completion, FinalizerCompletion::Protected { .. })
                }
                ReturnTarget::Dofile(completion) => {
                    matches!(completion, DofileCompletion::Protected(_))
                }
                ReturnTarget::TableSort(continuation) => {
                    matches!(continuation.completion, SortCompletion::Protected(_))
                }
                ReturnTarget::TableForeach(continuation) => {
                    matches!(continuation.completion, ForeachCompletion::Protected(_))
                }
                ReturnTarget::Require(continuation) => {
                    matches!(continuation.completion, RequireCompletion::Protected(_))
                }
                ReturnTarget::RequireSearch(search) => {
                    matches!(search.completion, RequireCompletion::Protected(_))
                }
                ReturnTarget::Module(continuation) => {
                    matches!(continuation.completion, ModuleCompletion::Protected(_))
                }
                ReturnTarget::ToString(continuation) => {
                    matches!(continuation.completion, ToStringCompletion::Protected(_))
                }
                ReturnTarget::StringFormat(continuation) => {
                    matches!(
                        continuation.completion,
                        StringFormatCompletion::Protected(_)
                    )
                }
                ReturnTarget::StringGsub(continuation) => {
                    matches!(continuation.completion, GsubCompletion::Protected(_))
                }
                ReturnTarget::Index(continuation) => matches!(
                    &continuation.completion,
                    IndexCompletion::StringGsub { continuation, .. }
                        if matches!(continuation.completion, GsubCompletion::Protected(_))
                ),
                _ => false,
            })
        });
        let error = if !error.is_hook_error()
            && frames
                .iter()
                .skip(protected.unwrap_or(0))
                .any(|frame| matches!(frame.caller, Some(ReturnTarget::DebugHook(_))))
        {
            VmError::HookError(Box::new(error))
        } else {
            error
        };
        // Lua 5.2 and 5.3 wrap an error in `GCTM` itself, so each finalizer
        // frame the unwind discards (not the boundary, which wraps on
        // delivery) wraps the error once, since a wrapped error is marked.
        let crossed_finalizer = self.wraps_finalizer_errors() && {
            let first = protected.map_or(0, |index| index + 1);
            frames
                .iter()
                .skip(first)
                .any(|frame| matches!(frame.caller, Some(ReturnTarget::UserdataFinalizer(_))))
        };
        let error = if crossed_finalizer {
            self.wrap_finalizer_error(error)
        } else {
            error
        };
        // `GCTM` raises the error again with `luaD_throw`, which calls no
        // message handler, so xpcall returns the error as it stands.
        let skip_handler = crossed_finalizer || error.is_wrapped_finalizer_error();
        // xpcall's handler receives the original nil. Other boundaries finish
        // error handling here, before close callbacks see the final error object.
        let error = if protected.is_some_and(|index| {
            matches!(
                frames[index].caller,
                Some(
                    ReturnTarget::ExtendedProtectedTarget(_)
                        | ReturnTarget::ExtendedProtectedHandler(_)
                )
            )
        }) {
            error
        } else {
            self.normalize_nil_error(error)
        };
        // A failed resume leaves the coroutine's stack intact. coroutine.close
        // later closes its resources with this error instead of nil.
        if protected.is_none() && retain_unprotected {
            return Ok(UnwindAction::Return(error));
        }
        let (closing, completion) = if let Some(index) = protected {
            // Only an xpcall handler reads the traceback (through
            // debug.traceback), so a plain pcall catching an error pays
            // nothing for it.
            let traceback = if matches!(
                frames[index].caller,
                Some(
                    ReturnTarget::ExtendedProtectedTarget(_)
                        | ReturnTarget::ExtendedProtectedHandler(_)
                )
            ) {
                self.with_base_level(capture_error_traceback(
                    frames,
                    self.native_metamethod_handler(frames),
                ))
            } else {
                Traceback {
                    debug_frames: Vec::new(),
                    frames: Vec::new(),
                }
            };
            // The frames leave for a stack of their own: close their
            // captured locals first, while the slots are where the cells
            // expect them.
            let level = frames[index].base;
            self.close_upvalues(frames, level);
            let mut closing = frames.split_off(index);
            let Some(target) = closing.first_mut().and_then(|frame| frame.caller.take()) else {
                return Ok(UnwindAction::Return(error));
            };
            (closing, UnwindCompletion::Protected { target, traceback })
        } else {
            self.close_upvalues(frames, 0);
            (std::mem::take(frames), UnwindCompletion::Unprotected)
        };
        if let UnwindCompletion::Protected {
            target:
                ReturnTarget::ExtendedProtectedTarget(mut target)
                | ReturnTarget::ExtendedProtectedHandler(mut target),
            traceback,
        } = completion
        {
            target.suppress_hooks |= error.is_hook_error();
            if target.handler_calls != 0
                && !closing.iter().any(|frame| {
                    frame.extras_ref().is_some_and(|extras| {
                        !extras.close_slots.is_empty() || extras.host_close_handler.is_some()
                    })
                })
            {
                return self.advance_unwind_close(
                    frames,
                    UnwindClose {
                        error,
                        frames: closing,
                        completion: UnwindCompletion::Protected {
                            target: ReturnTarget::ExtendedProtectedHandler(target),
                            traceback,
                        },
                    },
                );
            }
            let retry = !skip_handler
                && (target.handler_calls == 0
                    || self.can_retry_extended_protected_handler(&target));
            let object = if retry || skip_handler {
                vm_error_object(&error)?
            } else {
                Value::from("error in error handling")
            };
            target.pending_unwind = Some(std::sync::Arc::new(std::sync::Mutex::new(Some(
                UnwindClose {
                    error,
                    frames: closing,
                    completion: UnwindCompletion::ExtendedHandled {
                        continuation: target.clone(),
                        needs_handler: false,
                    },
                },
            ))));
            if retry {
                self.start_extended_protected_handler_with_traceback(
                    frames,
                    *target,
                    object,
                    Some(traceback),
                )?;
            } else {
                self.finish_extended_protected_handler(frames, &target, false, vec![object])?;
            }
            return Ok(UnwindAction::Continue);
        }
        self.advance_unwind_close(
            frames,
            UnwindClose {
                error,
                frames: closing,
                completion,
            },
        )
    }

    /// Unwinds all of `frames` for an `os.exit` request, which no protected
    /// call catches. A closing request on the main thread runs the pending
    /// `__close` handlers with a nil error, as `lua_close` does. Otherwise
    /// only upvalues close, so a host that keeps the state can still use it.
    fn begin_exit_unwind(
        &mut self,
        frames: &mut CallStack,
        error: VmError,
        close: bool,
    ) -> VmResult<UnwindAction> {
        let mut closing = CallStack::new();
        self.close_upvalues(frames, 0);
        append_exit_frames(&mut closing, std::mem::take(frames));
        if !close || self.current_coroutine.is_some() {
            closing.for_each_mut(|mut frame| {
                if let Some(extras) = frame.extras_mut_ref() {
                    extras.close_slots.clear();
                }
            });
        }
        self.advance_unwind_close(
            frames,
            UnwindClose {
                error,
                frames: closing,
                completion: UnwindCompletion::Unprotected,
            },
        )
    }

    pub(crate) fn resume_unwind_handler_error(
        &mut self,
        frames: &mut CallStack,
        error: VmError,
    ) -> VmResult<Option<UnwindAction>> {
        if let Some(request) = error.exit_request() {
            return self
                .begin_exit_unwind(frames, error, request.close)
                .map(Some);
        }
        let Some(index) = frames
            .iter()
            .rposition(|frame| matches!(frame.caller, Some(ReturnTarget::UnwindClose(_))))
        else {
            return Ok(None);
        };
        let level = frames[index].base;
        self.close_upvalues(frames, level);
        let mut closing = frames.split_off(index);
        let Some(ReturnTarget::UnwindClose(previous)) =
            closing.first_mut().and_then(|frame| frame.caller.take())
        else {
            unreachable!("unwind handler frame keeps its continuation");
        };
        self.advance_unwind_close(
            frames,
            UnwindClose {
                error,
                frames: closing,
                completion: UnwindCompletion::Resume(previous),
            },
        )
        .map(Some)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "Close dispatch keeps remaining frames and replacement errors together"
    )]
    // Shared take-once unwind state between `ExtendedProtectedCall` clones on
    // the VM thread; `Vm` is `!Send` (it owns `Rc`s), so the `Arc` never crosses threads.
    #[allow(clippy::arc_with_non_send_sync)]
    pub(crate) fn advance_unwind_close(
        &mut self,
        frames: &mut CallStack,
        mut continuation: UnwindClose,
    ) -> VmResult<UnwindAction> {
        if let UnwindCompletion::ExtendedHandled {
            continuation: target,
            needs_handler: true,
        } = &continuation.completion
        {
            let mut target = (**target).clone();
            let error = vm_error_object(&continuation.error)?;
            if let UnwindCompletion::ExtendedHandled { needs_handler, .. } =
                &mut continuation.completion
            {
                *needs_handler = false;
            }
            target.pending_unwind = Some(std::sync::Arc::new(std::sync::Mutex::new(Some(
                continuation,
            ))));
            self.start_extended_protected_handler(frames, target, error)?;
            return Ok(UnwindAction::Continue);
        }
        loop {
            // A stack kept from a failed coroutine still has its captured
            // locals open; close the top frame's before its registers go.
            self.close_top_frame_upvalues(&mut continuation.frames);
            let Some(mut frame) = continuation.frames.pop() else {
                break;
            };
            if let Some(ReturnTarget::DebugHook(hook_return)) = &frame.caller {
                self.debug_hook_depth = self.debug_hook_depth.saturating_sub(1);
                if let Some(hook) = self.debug_hooks.get_mut(&hook_return.thread) {
                    hook.running = false;
                }
                // The interrupted owner can be in either the pending unwind
                // or the retained protected-call stack.
                if let Some(mut interrupted) =
                    continuation.frames.last_mut().or_else(|| frames.last_mut())
                {
                    interrupted.extras().debug_hook_subject = None;
                    interrupted.extras().debug_hook_transfer = (0, 0);
                }
            }
            if let Some(handler) = frame
                .extras_mut_ref()
                .and_then(|extras| extras.host_close_handler.take())
            {
                let error = vm_error_object(&continuation.error)?;
                if let Err(error) =
                    self.call_native_above(frames, Some(&mut frame.view()), handler, &[error])
                {
                    replace_unwind_error(&mut continuation, error.into());
                }
            }
            while let Some(register) = frame
                .extras_ref()
                .and_then(|extras| extras.close_slots.last().copied())
            {
                if matches!(
                    continuation.completion,
                    UnwindCompletion::ExtendedHandled {
                        needs_handler: true,
                        ..
                    }
                ) {
                    continuation.frames.push_within_limit(frame)?;
                    return self.advance_unwind_close(frames, continuation);
                }
                let index = frame
                    .extras_ref()
                    .and_then(|extras| {
                        extras
                            .close_slots
                            .iter()
                            .rposition(|slot| *slot == register)
                    })
                    .expect("active close register exists");
                frame.extras().close_slots.remove(index);
                let value = frame.registers[usize::from(register)].clone();
                if matches!(value, Value::Nil) {
                    continue;
                }
                let handler = self
                    .runtime
                    .raw_metatable(&value)?
                    .ok_or_else(|| RuntimeError::message("variable got a non-closable value"))
                    .and_then(|metatable| {
                        self.runtime.table_get(metatable, &Value::from("__close"))
                    });
                // Closing for `os.exit` is not an error: handlers see nil.
                let error_value = if continuation.error.exit_request().is_some()
                    || matches!(continuation.error, VmError::CoroutineClosed)
                {
                    Value::Nil
                } else {
                    self.normalize_nil_error_value(vm_error_object(&continuation.error)?)
                };
                match handler {
                    Ok(Value::NativeFunction(function)) => {
                        if let Err(error) = self.call_native_above(
                            frames,
                            Some(&mut frame.view()),
                            function,
                            &[value, error_value],
                        ) {
                            replace_unwind_error(&mut continuation, error.into());
                        }
                    }
                    Ok(Value::Closure(closure)) => {
                        let body = match self.closure_body(closure) {
                            Ok(body) => body,
                            Err(error) => {
                                replace_unwind_error(&mut continuation, error);
                                continue;
                            }
                        };
                        continuation.frames.push_within_limit(frame)?;
                        let state = UnwindClose {
                            error: continuation.error.clone(),
                            frames: std::mem::take(&mut continuation.frames),
                            completion: continuation.completion,
                        };
                        let child = self.make_frame(
                            body.chunk(),
                            vec![value, error_value],
                            body.shared_cells(),
                            Some(ReturnTarget::UnwindClose(Box::new(state))),
                            None,
                            body.environment(),
                            Some(closure),
                        )?;
                        frames.push_within_limit(child)?;
                        return Ok(UnwindAction::Continue);
                    }
                    Ok(Value::BuiltinFunction(builtin)) => {
                        let arguments = [value, error_value];
                        match self.call_builtin(builtin, &arguments) {
                            Ok(_) => {}
                            Err(VmError::InvalidDispatch(_)) => {
                                // A builtin that needs the frame stack, such
                                // as coroutine.yield, runs in a frame of its
                                // own so that it can suspend.
                                let chunk = crate::vm::host::builtin_call_chunk(arguments.len())?;
                                continuation.frames.push_within_limit(frame)?;
                                let state = UnwindClose {
                                    error: continuation.error.clone(),
                                    frames: std::mem::take(&mut continuation.frames),
                                    completion: continuation.completion,
                                };
                                let child = self.make_frame(
                                    chunk,
                                    std::iter::once(Value::BuiltinFunction(builtin))
                                        .chain(arguments),
                                    crate::runtime::CaptureCells::default(),
                                    Some(ReturnTarget::UnwindClose(Box::new(state))),
                                    None,
                                    self.thread_environment,
                                    None,
                                )?;
                                frames.push_within_limit(child)?;
                                return Ok(UnwindAction::Continue);
                            }
                            Err(error) => replace_unwind_error(&mut continuation, error),
                        }
                    }
                    Ok(_) => replace_unwind_error(
                        &mut continuation,
                        RuntimeError::message("__close must be callable").into(),
                    ),
                    Err(error) => replace_unwind_error(&mut continuation, error.into()),
                }
            }
            if matches!(
                continuation.completion,
                UnwindCompletion::ExtendedHandled {
                    needs_handler: true,
                    ..
                }
            ) {
                continuation.frames.push_within_limit(frame)?;
                return self.advance_unwind_close(frames, continuation);
            }
        }
        match continuation.completion {
            UnwindCompletion::Unprotected => Ok(UnwindAction::Return(continuation.error)),
            UnwindCompletion::ExtendedHandled {
                continuation: target,
                ..
            } => {
                self.finish_extended_protected_handler(
                    frames,
                    &target,
                    false,
                    vec![vm_error_object(&continuation.error)?],
                )?;
                Ok(UnwindAction::Continue)
            }
            UnwindCompletion::Resume(previous) => {
                let mut previous = *previous;
                replace_unwind_error(&mut previous, continuation.error);
                self.advance_unwind_close(frames, previous)
            }
            UnwindCompletion::Protected { target, traceback } => {
                self.finish_unwind_delivery(frames, target, continuation.error, traceback)?;
                Ok(UnwindAction::Continue)
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    pub(crate) fn finish_unwind_delivery(
        &mut self,
        frames: &mut CallStack,
        target: ReturnTarget,
        error: VmError,
        traceback: Traceback,
    ) -> VmResult<()> {
        match target {
            ReturnTarget::Protected(target) => {
                let mut caller = frames.last_mut().expect("protected caller exists");
                self.write_protected_results(
                    &mut caller,
                    target.base,
                    target.results,
                    false,
                    vec![vm_error_object(&error)?],
                )?;
            }
            ReturnTarget::ExtendedProtectedTarget(mut continuation) => {
                continuation.suppress_hooks |= error.is_hook_error();
                self.start_extended_protected_handler_with_traceback(
                    frames,
                    *continuation,
                    vm_error_object(&error)?,
                    Some(traceback),
                )?;
            }
            ReturnTarget::ExtendedProtectedHandler(mut continuation) => {
                continuation.suppress_hooks |= error.is_hook_error();
                if self.can_retry_extended_protected_handler(&continuation) {
                    self.start_extended_protected_handler_with_traceback(
                        frames,
                        *continuation,
                        vm_error_object(&error)?,
                        Some(traceback),
                    )?;
                } else {
                    self.finish_extended_protected_handler(
                        frames,
                        &continuation,
                        false,
                        vec![Value::from("error in error handling")],
                    )?;
                }
            }
            ReturnTarget::ProtectedIndex(continuation) => {
                self.complete_index(frames, *continuation, Value::Boolean(false))?;
            }
            ReturnTarget::TableForeach(continuation) => {
                self.finish_protected_table_foreach_error(frames, &continuation, error)?;
            }
            ReturnTarget::TableSort(continuation) => {
                self.finish_protected_table_sort_error(frames, &continuation, error)?;
            }
            ReturnTarget::ProtectedSort(continuation) => self.resume_table_sort(
                frames,
                *continuation,
                &[Value::Boolean(false), vm_error_object(&error)?],
            )?,
            ReturnTarget::LoadReader(continuation) => self.finish_load(
                frames,
                continuation.completion,
                vec![Value::Nil, vm_error_object(&error)?],
            )?,
            ReturnTarget::Dofile(DofileCompletion::Protected(target)) => {
                let mut caller = frames.last_mut().expect("protected dofile caller exists");
                self.write_protected_results(
                    &mut caller,
                    target.base,
                    target.results,
                    false,
                    vec![vm_error_object(&error)?],
                )?;
            }
            ReturnTarget::Require(continuation) => {
                let RequireCompletion::Protected(target) = continuation.completion else {
                    return Err(error);
                };
                let mut caller = frames.last_mut().expect("protected require caller exists");
                self.write_protected_results(
                    &mut caller,
                    target.base,
                    target.results,
                    false,
                    vec![vm_error_object(&error)?],
                )?;
            }
            ReturnTarget::RequireSearch(search) => {
                let RequireCompletion::Protected(target) = search.completion else {
                    return Err(error);
                };
                let mut caller = frames.last_mut().expect("protected require caller exists");
                self.write_protected_results(
                    &mut caller,
                    target.base,
                    target.results,
                    false,
                    vec![vm_error_object(&error)?],
                )?;
            }
            ReturnTarget::Module(continuation) => {
                self.finish_protected_module_error(frames, continuation.completion, error)?;
            }
            ReturnTarget::ToString(continuation) => {
                let ToStringCompletion::Protected(target) = continuation.completion else {
                    return Err(error);
                };
                let mut caller = frames.last_mut().expect("protected tostring caller exists");
                self.write_protected_results(
                    &mut caller,
                    target.base,
                    target.results,
                    false,
                    vec![vm_error_object(&error)?],
                )?;
            }
            ReturnTarget::StringFormat(continuation) => {
                let StringFormatCompletion::Protected(target) = continuation.completion else {
                    return Err(error);
                };
                let mut caller = frames
                    .last_mut()
                    .expect("protected string.format caller exists");
                self.write_protected_results(
                    &mut caller,
                    target.base,
                    target.results,
                    false,
                    vec![vm_error_object(&error)?],
                )?;
            }
            ReturnTarget::Index(continuation) => {
                let IndexCompletion::StringGsub { continuation, .. } = continuation.completion
                else {
                    return Err(error);
                };
                let GsubCompletion::Protected(target) = continuation.completion else {
                    return Err(error);
                };
                let mut caller = frames
                    .last_mut()
                    .expect("protected string.gsub caller exists");
                self.write_protected_results(
                    &mut caller,
                    target.base,
                    target.results,
                    false,
                    vec![vm_error_object(&error)?],
                )?;
            }
            ReturnTarget::StringGsub(continuation) => {
                let GsubCompletion::Protected(target) = continuation.completion else {
                    return Err(error);
                };
                let mut caller = frames
                    .last_mut()
                    .expect("protected string.gsub caller exists");
                self.write_protected_results(
                    &mut caller,
                    target.base,
                    target.results,
                    false,
                    vec![vm_error_object(&error)?],
                )?;
            }
            // The finalizer's `__close` handlers have run, as `luaD_pcall`
            // closes them before `GCTM` warns. The remaining finalizers go on.
            ReturnTarget::UserdataFinalizer(completion) if self.finalizer_errors_warn() => {
                self.warn_finalizer_error(&error);
                self.advance_userdata_finalizers(frames, *completion)?;
            }
            ReturnTarget::UserdataFinalizer(completion)
                if matches!(*completion, FinalizerCompletion::Protected { .. }) =>
            {
                let FinalizerCompletion::Protected { target, .. } = *completion else {
                    unreachable!("guarded protected finalizer completion")
                };
                let mut caller = frames
                    .last_mut()
                    .expect("protected collectgarbage caller exists");
                self.write_protected_results(
                    &mut caller,
                    target.base,
                    target.results,
                    false,
                    vec![vm_error_object(&self.wrap_finalizer_error(error))?],
                )?;
            }
            ReturnTarget::UserdataFinalizer(_) => return Err(self.wrap_finalizer_error(error)),
            _ => return Err(error),
        }
        Ok(())
    }
}

/// Records `error`, raised while unwinding, as the error the unwind reports.
/// A pending `os.exit` request stays unless `error` is another request.
fn replace_unwind_error(continuation: &mut UnwindClose, error: VmError) {
    if let UnwindCompletion::ExtendedHandled { needs_handler, .. } = &mut continuation.completion {
        *needs_handler = true;
    }
    let current = &mut continuation.error;
    if current.exit_request().is_none() || error.exit_request().is_some() {
        *current = if current.is_hook_error() && !error.is_hook_error() {
            VmError::HookError(Box::new(error))
        } else {
            error
        };
    }
}

/// Rejoins parked xpcall targets below their handlers before process exit.
fn append_exit_frames(output: &mut CallStack, mut input: CallStack) {
    let mut ordered = Vec::new();
    while let Some(frame) = input.pop() {
        ordered.push(frame);
    }
    for mut frame in ordered.into_iter().rev() {
        match frame.caller.take() {
            Some(ReturnTarget::ExtendedProtectedHandler(target)) => {
                append_exit_handler(output, &target);
            }
            Some(ReturnTarget::UnwindClose(pending)) => append_exit_unwind(output, *pending),
            other => frame.caller = other,
        }
        output.push(frame);
    }
}

fn append_exit_handler(output: &mut CallStack, target: &crate::vm::ExtendedProtectedCall) {
    let pending = target
        .pending_unwind
        .as_ref()
        .and_then(|pending| pending.lock().expect("pending unwind lock").take());
    if let Some(pending) = pending {
        append_exit_unwind(output, pending);
    }
}

fn append_exit_unwind(output: &mut CallStack, pending: UnwindClose) {
    match pending.completion {
        UnwindCompletion::ExtendedHandled { continuation, .. } => {
            append_exit_handler(output, &continuation);
        }
        UnwindCompletion::Resume(previous) => append_exit_unwind(output, *previous),
        UnwindCompletion::Protected {
            target: ReturnTarget::ExtendedProtectedHandler(target),
            ..
        } => append_exit_handler(output, &target),
        _ => {}
    }
    append_exit_frames(output, pending.frames);
}
