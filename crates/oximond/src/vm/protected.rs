//! `pcall` and `xpcall` protected-call setup and result delivery.

use crate::bytecode::{CallKind, Instruction};
use crate::runtime::{BuiltinFunction, RuntimeError, Value};
use crate::string::LuaString;
use crate::vm::ActiveCallSite;
use crate::vm::arith::{call_metamethod_key, profile_usize_value};
use crate::vm::coroutines::coroutine_argument;
use crate::vm::debug_info::capture_traceback;
use crate::vm::debug_info::value_debug_level;
use crate::vm::errors::{
    call_type_error_anonymous, loaded_function_name, locate_error_below_protected_call,
    vm_error_object,
};
use crate::vm::registers::write_call_results;
use crate::vm::stack::{CallStack, FrameMut};
use crate::vm::{
    CallReturn, CollectGarbageAction, DebugHookEvent, DofileCompletion, ExtendedProtectedCall,
    FinalizerCompletion, ForeachCompletion, ForeachMode, GsubCompletion, LoadCompletion,
    MAX_INDEX_HOPS, MAX_LUA_CALL_FRAMES, ModuleCompletion, RequireCompletion, ReturnTarget,
    SortCompletion, Traceback, Vm, VmError, VmResult,
};

impl Vm {
    #[allow(clippy::too_many_lines)]
    pub(crate) fn start_extended_protected_call(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        mut arguments: Vec<Value>,
    ) -> VmResult<()> {
        // Lua 5.1 only needs a handler; LuaJIT and Lua 5.2 and later need a
        // function, and check it first.
        if self.compatibility == crate::compat::CompatibilityProfile::Lua51 {
            if arguments.len() < 2 {
                return Err(RuntimeError::argument_error(2, "value expected").into());
            }
        } else if !matches!(
            arguments.get(1),
            Some(
                Value::Closure(_)
                    | Value::NativeFunction(_)
                    | Value::BuiltinFunction(_)
                    | Value::GmatchIterator(_)
                    | Value::CoroutineWrapper(_)
            )
        ) {
            return Err(RuntimeError::bad_argument(2, "function", arguments.get(1)).into());
        }
        let target = arguments.remove(0);
        let handler = arguments.remove(0);
        // Lua 5.1's `xpcall(f, handler)` takes no arguments for `f`.
        if self.compatibility == crate::compat::CompatibilityProfile::Lua51 {
            arguments.clear();
        }
        self.start_extended_protected_target(
            frames,
            ExtendedProtectedCall {
                target: CallReturn {
                    base: instruction.a(),
                    results: instruction.c(),
                },
                handler,
                suppress_hooks: false,
                handler_calls: 0,
                pending_unwind: None,
            },
            target,
            arguments,
        )
    }

    /// Spells an argument error of a library `function` that `pcall` or
    /// `xpcall` called directly, as `luaL_argerror` does when the call has
    /// no name: Lua 5.2 and later name the function by its field in a
    /// loaded module, and the error carries no position.
    pub(crate) fn protected_error_object(
        &self,
        function: &Value,
        error: VmError,
    ) -> VmResult<Value> {
        vm_error_object(&self.protected_argument_error(function, error))
    }

    pub(crate) fn protected_argument_error(&self, function: &Value, error: VmError) -> VmError {
        self.protected_argument_error_extra(function, error, 0)
    }

    /// [`Self::protected_argument_error`] for a call through `extra` `__call`
    /// hops.
    pub(crate) fn protected_argument_error_extra(
        &self,
        function: &Value,
        error: VmError,
        extra: u8,
    ) -> VmError {
        let VmError::Runtime(runtime_error) = &error else {
            return error;
        };
        let Some((position, detail)) = runtime_error.argument_detail() else {
            return error;
        };
        let name = (!self.compatibility.is_lua51_family())
            .then(|| loaded_function_name(&self.runtime, function))
            .flatten()
            .unwrap_or_else(|| "?".to_owned());
        let site = ActiveCallSite {
            name: LuaString::from(name),
            kind: CallKind::Global,
        };
        RuntimeError::Raised(Value::from(crate::vm::errors::argument_message_extra(
            Some(&site),
            position,
            &detail,
            self.compatibility,
            usize::from(extra),
        )))
        .into()
    }

    /// Starts the protected function of Lua 5.1's `xpcall`.
    ///
    /// The caller lives below the function frame, so an eventual error can
    /// discard that frame before calling the saved Lua error handler.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn start_extended_protected_target(
        &mut self,
        frames: &mut CallStack,
        mut continuation: ExtendedProtectedCall,
        mut function: Value,
        mut arguments: Vec<Value>,
    ) -> VmResult<()> {
        let mut hops = 0_u8;
        loop {
            if let Value::NativeFunction(native) = function
                && let Some(delegate) = self.string_arithmetic_delegate(native, &arguments)?
            {
                function = delegate;
                continue;
            }
            match function {
                Value::NativeFunction(function) => {
                    match self.call_native(frames, function, &arguments) {
                        Ok(values) => self.finish_extended_protected_call(
                            frames,
                            continuation.target,
                            true,
                            values,
                        ),
                        Err(error) => {
                            let error = self.protected_argument_error_extra(
                                &Value::NativeFunction(function),
                                VmError::Runtime(locate_error_below_protected_call(
                                    frames,
                                    self.compatibility,
                                    error,
                                )),
                                hops,
                            );
                            continuation.suppress_hooks |= error.is_hook_error();
                            // Like `luaL_error` at level 1, whose caller here is
                            // `xpcall` itself, the error carries no position.
                            let traceback = self
                                .traceback_below_xpcall(frames, Value::NativeFunction(function));
                            self.start_extended_protected_handler_with_traceback(
                                frames,
                                continuation,
                                vm_error_object(&error)?,
                                Some(traceback),
                            )
                        }
                    }?;
                }
                Value::Closure(function) => {
                    if frames.len() >= MAX_LUA_CALL_FRAMES {
                        return self.start_extended_protected_handler(
                            frames,
                            continuation,
                            Value::from("stack overflow"),
                        );
                    }
                    let body = match self.closure_body(function) {
                        Ok(body) => body,
                        Err(error) => {
                            return self.start_extended_protected_handler(
                                frames,
                                continuation,
                                vm_error_object(&error)?,
                            );
                        }
                    };
                    let frame = self.make_frame(
                        body.chunk(),
                        arguments,
                        body.shared_cells(),
                        Some(ReturnTarget::ExtendedProtectedTarget(Box::new(
                            continuation,
                        ))),
                        None,
                        body.environment(),
                        Some(function),
                    )?;
                    frames.push(frame);
                    self.fire_debug_hook(frames, DebugHookEvent::Call)?;
                    return Ok(());
                }
                Value::GmatchIterator(iterator) => match self.call_gmatch_iterator(iterator) {
                    Ok(values) => self.finish_extended_protected_call(
                        frames,
                        continuation.target,
                        true,
                        values,
                    )?,
                    Err(error) => self.start_extended_protected_handler(
                        frames,
                        continuation,
                        vm_error_object(&error)?,
                    )?,
                },
                Value::CoroutineWrapper(thread) => {
                    match self.resume_wrapped_coroutine_mode(frames, thread, arguments, false) {
                        Ok(mut values) if matches!(values.first(), Some(Value::Boolean(true))) => {
                            values.remove(0);
                            self.finish_extended_protected_call(
                                frames,
                                continuation.target,
                                true,
                                values,
                            )?;
                        }
                        Ok(values) => self.start_extended_protected_handler(
                            frames,
                            continuation,
                            values.get(1).cloned().unwrap_or(Value::Nil),
                        )?,
                        Err(error) => self.start_extended_protected_handler(
                            frames,
                            continuation,
                            vm_error_object(&error)?,
                        )?,
                    }
                }
                Value::BuiltinFunction(builtin) => match self.call_builtin(builtin, &arguments) {
                    Ok(values) => self.finish_extended_protected_call(
                        frames,
                        continuation.target,
                        true,
                        values,
                    )?,
                    Err(VmError::InvalidDispatch(_)) => {
                        let handler = continuation.handler.clone();
                        let target = continuation.target;
                        let frame_target =
                            ReturnTarget::ExtendedProtectedTarget(Box::new(continuation));
                        if let Err(error) =
                            self.push_builtin_call_frame(frames, builtin, arguments, frame_target)
                        {
                            self.start_extended_protected_handler(
                                frames,
                                ExtendedProtectedCall {
                                    target,
                                    handler,
                                    suppress_hooks: error.is_hook_error(),
                                    handler_calls: 0,
                                    pending_unwind: None,
                                },
                                vm_error_object(&error)?,
                            )?;
                        }
                    }
                    Err(error) => {
                        let error = self.protected_argument_error_extra(
                            &Value::BuiltinFunction(builtin),
                            error,
                            hops,
                        );
                        continuation.suppress_hooks |= error.is_hook_error();
                        let traceback =
                            self.traceback_below_xpcall(frames, Value::BuiltinFunction(builtin));
                        self.start_extended_protected_handler_with_traceback(
                            frames,
                            continuation,
                            vm_error_object(&error)?,
                            Some(traceback),
                        )?;
                    }
                },
                callee => {
                    if hops == MAX_INDEX_HOPS {
                        self.start_extended_protected_handler(
                            frames,
                            continuation,
                            Value::from("metatable chain too long"),
                        )?;
                    } else if let Some(metatable) = self.runtime.raw_metatable(&callee)? {
                        let handler = self
                            .runtime
                            .table_get_string_ref(metatable, call_metamethod_key())
                            .cloned()
                            .unwrap_or(Value::Nil);
                        if matches!(handler, Value::Nil) {
                            self.start_extended_protected_handler(
                                frames,
                                continuation,
                                Value::from(call_type_error_anonymous(&callee).to_string()),
                            )?;
                        } else {
                            arguments.insert(0, callee);
                            function = handler;
                            hops += 1;
                            continue;
                        }
                    } else {
                        self.start_extended_protected_handler(
                            frames,
                            continuation,
                            Value::from(call_type_error_anonymous(&callee).to_string()),
                        )?;
                    }
                }
            }
            return Ok(());
        }
    }

    /// The stack an `xpcall` error handler sees when `function`, a native that
    /// `xpcall` called directly, raised the error: that function, `xpcall`,
    /// then the caller's stack.
    fn traceback_below_xpcall(&self, frames: &CallStack, function: Value) -> Traceback {
        let site = frames
            .last()
            .and_then(|frame| crate::vm::debug_info::active_call_site(&frame));
        let mut trace = capture_traceback(frames);
        trace.frames.insert(
            0,
            crate::vm::debug_info::native_trace_frame(None, Some("xpcall"), site),
        );
        trace.frames.insert(
            0,
            crate::vm::debug_info::native_trace_frame(Some(function), None, None),
        );
        self.with_base_level(trace)
    }

    /// Calls an `xpcall` message handler with the failed function's resources
    /// still open. PUC Lua reenters the handler if the handler itself raises.
    pub(crate) fn start_extended_protected_handler(
        &mut self,
        frames: &mut CallStack,
        continuation: ExtendedProtectedCall,
        error: Value,
    ) -> VmResult<()> {
        let traceback = self.with_base_level(capture_traceback(frames));
        self.start_extended_protected_handler_with_traceback(
            frames,
            continuation,
            error,
            Some(traceback),
        )
    }

    #[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
    pub(crate) fn start_extended_protected_handler_with_traceback(
        &mut self,
        frames: &mut CallStack,
        continuation: ExtendedProtectedCall,
        error: Value,
        traceback: Option<Traceback>,
    ) -> VmResult<()> {
        let mut roots = crate::runtime::GcRootBuilder::new();
        roots.value(&continuation.handler);
        roots.value(&error);
        if let Some(pending) = &continuation.pending_unwind
            && let Some(pending) = pending.lock().expect("pending unwind lock").as_ref()
        {
            crate::vm::continuations::root_unwind_close(&mut roots, pending);
        }
        self.suspended_roots.push(roots.finish());
        let result = self.run_extended_protected_handler(frames, continuation, error, traceback);
        self.suspended_roots.pop();
        result
    }

    #[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
    fn run_extended_protected_handler(
        &mut self,
        frames: &mut CallStack,
        mut continuation: ExtendedProtectedCall,
        error: Value,
        traceback: Option<Traceback>,
    ) -> VmResult<()> {
        continuation.handler_calls += 1;
        // The protected function wrote captured cells while the caller sat
        // suspended. Pull them into its registers before any handler call
        // flushes the registers back over the cells.
        let mut function = continuation.handler.clone();
        let mut arguments = vec![error];
        let mut hops = 0_u8;
        loop {
            match function {
                Value::NativeFunction(native) => {
                    let previous = self.debug_hook_depth;
                    if continuation.suppress_hooks {
                        self.debug_hook_depth += 1;
                    }
                    let result = self.call_native(frames, native, &arguments);
                    self.debug_hook_depth = previous;
                    match result {
                        Ok(values) => self.finish_extended_protected_handler(
                            frames,
                            &continuation,
                            false,
                            vec![values.first().cloned().unwrap_or(Value::Nil)],
                        )?,
                        Err(error @ RuntimeError::Exit(_)) => return Err(error.into()),
                        Err(error) if self.can_retry_extended_protected_handler(&continuation) => {
                            arguments = vec![vm_error_object(&error.into())?];
                            function = continuation.handler.clone();
                            continuation.handler_calls += 1;
                            hops = 0;
                            continue;
                        }
                        Err(_) => self.finish_extended_protected_handler(
                            frames,
                            &continuation,
                            false,
                            vec![Value::from("error in error handling")],
                        )?,
                    }
                }
                Value::Closure(function) => {
                    if frames.len() >= MAX_LUA_CALL_FRAMES {
                        return self.finish_extended_protected_handler(
                            frames,
                            &continuation,
                            false,
                            vec![Value::from("error in error handling")],
                        );
                    }
                    let Ok(body) = self.closure_body(function) else {
                        return self.finish_extended_protected_handler(
                            frames,
                            &continuation,
                            false,
                            vec![Value::from("error in error handling")],
                        );
                    };
                    let mut frame = self.make_frame(
                        body.chunk(),
                        arguments,
                        body.shared_cells(),
                        Some(ReturnTarget::ExtendedProtectedHandler(Box::new(
                            continuation.clone(),
                        ))),
                        None,
                        body.environment(),
                        Some(function),
                    )?;
                    frame.extras().traceback = traceback;
                    frame.extras().suppress_debug_hooks = continuation.suppress_hooks;
                    frames.push(frame);
                    self.fire_debug_hook(frames, DebugHookEvent::Call)?;
                    return Ok(());
                }
                Value::GmatchIterator(iterator) => match self.call_gmatch_iterator(iterator) {
                    Ok(values) => self.finish_extended_protected_handler(
                        frames,
                        &continuation,
                        false,
                        vec![values.first().cloned().unwrap_or(Value::Nil)],
                    )?,
                    Err(_) => self.finish_extended_protected_handler(
                        frames,
                        &continuation,
                        false,
                        vec![Value::from("error in error handling")],
                    )?,
                },
                Value::BuiltinFunction(BuiltinFunction::DebugTraceback) => {
                    let result = self.traceback_handler_result(&arguments, traceback.as_ref());
                    self.finish_extended_protected_handler(
                        frames,
                        &continuation,
                        false,
                        vec![result],
                    )?;
                }
                Value::BuiltinFunction(builtin) => match self.call_builtin(builtin, &arguments) {
                    Ok(values) => self.finish_extended_protected_handler(
                        frames,
                        &continuation,
                        false,
                        vec![values.first().cloned().unwrap_or(Value::Nil)],
                    )?,
                    Err(VmError::InvalidDispatch(_)) => {
                        let target =
                            ReturnTarget::ExtendedProtectedHandler(Box::new(continuation.clone()));
                        match self.push_builtin_call_frame(frames, builtin, arguments, target) {
                            Ok(()) => {
                                let mut frame =
                                    frames.last_mut().expect("handler frame was pushed");
                                frame.extras().traceback = traceback;
                                frame.extras().suppress_debug_hooks = continuation.suppress_hooks;
                            }
                            Err(error) if error.exit_request().is_some() => return Err(error),
                            Err(_) => self.finish_extended_protected_handler(
                                frames,
                                &continuation,
                                false,
                                vec![Value::from("error in error handling")],
                            )?,
                        }
                    }
                    Err(error) if error.exit_request().is_some() => return Err(error),
                    Err(error) if self.can_retry_extended_protected_handler(&continuation) => {
                        arguments = vec![vm_error_object(&error)?];
                        function = continuation.handler.clone();
                        continuation.handler_calls += 1;
                        hops = 0;
                        continue;
                    }
                    Err(_) => self.finish_extended_protected_handler(
                        frames,
                        &continuation,
                        false,
                        vec![Value::from("error in error handling")],
                    )?,
                },
                Value::CoroutineWrapper(thread) => {
                    match self.resume_wrapped_coroutine_mode(frames, thread, arguments, false) {
                        Ok(mut values) if matches!(values.first(), Some(Value::Boolean(true))) => {
                            values.remove(0);
                            self.finish_extended_protected_handler(
                                frames,
                                &continuation,
                                false,
                                vec![values.first().cloned().unwrap_or(Value::Nil)],
                            )?;
                        }
                        Err(error) if error.exit_request().is_some() => return Err(error),
                        Ok(_) | Err(_) => self.finish_extended_protected_handler(
                            frames,
                            &continuation,
                            false,
                            vec![Value::from("error in error handling")],
                        )?,
                    }
                }
                callee => {
                    if hops == MAX_INDEX_HOPS {
                        self.finish_extended_protected_handler(
                            frames,
                            &continuation,
                            false,
                            vec![Value::from("error in error handling")],
                        )?;
                    } else if let Some(metatable) = self.runtime.raw_metatable(&callee)? {
                        let handler = self
                            .runtime
                            .table_get_string_ref(metatable, call_metamethod_key())
                            .cloned()
                            .unwrap_or(Value::Nil);
                        if matches!(handler, Value::Nil) {
                            self.finish_extended_protected_handler(
                                frames,
                                &continuation,
                                false,
                                vec![Value::from("error in error handling")],
                            )?;
                        } else {
                            arguments.insert(0, callee);
                            function = handler;
                            hops += 1;
                            continue;
                        }
                    } else {
                        self.finish_extended_protected_handler(
                            frames,
                            &continuation,
                            false,
                            vec![Value::from("error in error handling")],
                        )?;
                    }
                }
            }
            return Ok(());
        }
    }

    pub(crate) fn can_retry_extended_protected_handler(
        &self,
        continuation: &ExtendedProtectedCall,
    ) -> bool {
        self.compatibility != crate::compat::CompatibilityProfile::LuaJit21
            && continuation.handler_calls < crate::vm::MAX_NESTED_RESUMES
    }

    pub(crate) fn finish_extended_protected_handler(
        &mut self,
        frames: &mut CallStack,
        continuation: &ExtendedProtectedCall,
        success: bool,
        values: Vec<Value>,
    ) -> VmResult<()> {
        let pending = continuation
            .pending_unwind
            .as_ref()
            .and_then(|pending| pending.lock().expect("pending unwind lock").take());
        if let Some(mut pending) = pending {
            pending.error = self.normalize_nil_error(crate::vm::errors::raised_value(
                values.first().cloned().unwrap_or(Value::Nil),
            ));

            if let crate::vm::UnwindAction::Return(error) =
                self.advance_unwind_close(frames, pending)?
            {
                return Err(error);
            }
            Ok(())
        } else {
            self.finish_extended_protected_call(frames, continuation.target, success, values)
        }
    }

    pub(crate) fn finish_extended_protected_call(
        &mut self,
        frames: &mut CallStack,
        target: CallReturn,
        success: bool,
        values: Vec<Value>,
    ) -> VmResult<()> {
        let mut caller = frames.last_mut().expect("extended protected caller exists");
        self.write_protected_results(&mut caller, target.base, target.results, success, values)
    }

    #[allow(clippy::too_many_lines)]
    pub(crate) fn start_protected_call(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        mut arguments: Vec<Value>,
    ) -> VmResult<()> {
        let frame_index = frames.len() - 1;
        if arguments.is_empty() {
            return Err(RuntimeError::argument_error(1, "value expected").into());
        }
        let mut target = arguments.remove(0);
        let mut hops = 0_u8;
        while !matches!(
            target,
            Value::NativeFunction(_)
                | Value::Closure(_)
                | Value::GmatchIterator(_)
                | Value::CoroutineWrapper(_)
                | Value::BuiltinFunction(_)
        ) {
            if hops == MAX_INDEX_HOPS {
                return self.write_protected_results(
                    &mut frames.at_mut(frame_index),
                    instruction.a(),
                    instruction.c(),
                    false,
                    vec![Value::from("metatable chain too long")],
                );
            }
            let Some(metatable) = self.runtime.raw_metatable(&target)? else {
                break;
            };
            let handler = self
                .runtime
                .table_get_string_ref(metatable, call_metamethod_key())
                .cloned()
                .unwrap_or(Value::Nil);
            if matches!(handler, Value::Nil) {
                break;
            }
            arguments.insert(0, target);
            target = handler;
            hops += 1;
        }
        if let Value::NativeFunction(native) = target
            && let Some(delegate) = self.string_arithmetic_delegate(native, &arguments)?
        {
            return self.start_protected_call(
                frames,
                instruction,
                std::iter::once(delegate).chain(arguments).collect(),
            );
        }
        match target {
            Value::CoroutineWrapper(thread) => {
                match self.resume_wrapped_coroutine_mode(frames, thread, arguments, false) {
                    Ok(mut results) if matches!(results.first(), Some(Value::Boolean(true))) => {
                        results.remove(0);
                        self.write_protected_results(
                            &mut frames.last_mut().expect("protected caller exists"),
                            instruction.a(),
                            instruction.c(),
                            true,
                            results,
                        )
                    }
                    Ok(results) => self.write_protected_results(
                        &mut frames.last_mut().expect("protected caller exists"),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![results.get(1).cloned().unwrap_or(Value::Nil)],
                    ),
                    Err(error) => self.write_protected_results(
                        &mut frames.last_mut().expect("protected caller exists"),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![vm_error_object(&error)?],
                    ),
                }
            }
            Value::NativeFunction(function) => match self.call_native(frames, function, &arguments)
            {
                Ok(results) => self.write_protected_results(
                    &mut frames.at_mut(frame_index),
                    instruction.a(),
                    instruction.c(),
                    true,
                    results,
                ),
                Err(error) => {
                    let error = self.protected_argument_error_extra(
                        &Value::NativeFunction(function),
                        VmError::Runtime(locate_error_below_protected_call(
                            frames,
                            self.compatibility,
                            error,
                        )),
                        hops,
                    );
                    // Like `luaL_error` at level 1, whose caller here is
                    // `pcall` itself, the error carries no position.
                    self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![vm_error_object(&error)?],
                    )
                }
            },
            Value::Closure(function) => {
                if frames.len() >= MAX_LUA_CALL_FRAMES {
                    return self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![Value::from("stack overflow")],
                    );
                }
                let body = match self.closure_body(function) {
                    Ok(body) => body,
                    Err(error) => {
                        return self.write_protected_results(
                            &mut frames.at_mut(frame_index),
                            instruction.a(),
                            instruction.c(),
                            false,
                            vec![vm_error_object(&error)?],
                        );
                    }
                };
                let frame = self.make_frame(
                    body.chunk(),
                    arguments,
                    body.shared_cells(),
                    Some(ReturnTarget::Protected(CallReturn {
                        base: instruction.a(),
                        results: instruction.c(),
                    })),
                    None,
                    body.environment(),
                    Some(function),
                )?;
                frames.push(frame);
                self.fire_debug_hook(frames, DebugHookEvent::Call)?;
                Ok(())
            }
            Value::GmatchIterator(iterator) => match self.call_gmatch_iterator(iterator) {
                Ok(results) => self.write_protected_results(
                    &mut frames.at_mut(frame_index),
                    instruction.a(),
                    instruction.c(),
                    true,
                    results,
                ),
                Err(error) => self.write_protected_results(
                    &mut frames.at_mut(frame_index),
                    instruction.a(),
                    instruction.c(),
                    false,
                    vec![vm_error_object(&error)?],
                ),
            },
            Value::BuiltinFunction(BuiltinFunction::TableSort) => {
                let continuation = SortCompletion::protected(instruction);
                match self.start_table_sort(frames, continuation, &arguments) {
                    Ok(()) => Ok(()),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::TableSort),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::TableForeach) => {
                let continuation = ForeachCompletion::protected(instruction);
                match self.start_table_foreach(frames, continuation, &arguments, ForeachMode::All) {
                    Ok(()) => Ok(()),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::TableForeach),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::TableForeachi) => {
                let continuation = ForeachCompletion::protected(instruction);
                match self.start_table_foreach(frames, continuation, &arguments, ForeachMode::Array)
                {
                    Ok(()) => Ok(()),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::TableForeachi),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(
                builtin @ (BuiltinFunction::Load
                | BuiltinFunction::LoadString
                | BuiltinFunction::LoadFile),
            ) => {
                let completion = LoadCompletion::protected(instruction);
                let started = if builtin == BuiltinFunction::LoadFile {
                    self.start_loadfile(frames, completion, &arguments)
                } else {
                    self.check_load_argument(builtin, &arguments)
                        .and_then(|()| self.start_load(frames, completion, &arguments))
                };
                match started {
                    Ok(()) => Ok(()),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(&Value::BuiltinFunction(builtin), error)?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::DoFile) => {
                match self.start_dofile(
                    frames,
                    DofileCompletion::protected(instruction),
                    &arguments,
                ) {
                    Ok(()) => Ok(()),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::DoFile),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::Require) => {
                match self.start_require(
                    frames,
                    RequireCompletion::protected(instruction),
                    &arguments,
                ) {
                    Ok(()) => Ok(()),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::Require),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::Module) => {
                match self.start_module(
                    frames,
                    ModuleCompletion::protected(instruction),
                    &arguments,
                ) {
                    Ok(()) => Ok(()),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::Module),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::DebugDebug) => {
                match self.debug_console(frames) {
                    Ok(results) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        true,
                        results,
                    ),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::DebugDebug),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::DebugGetInfo) => {
                match self.debug_get_info(frames, &arguments) {
                    Ok(results) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        true,
                        results,
                    ),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::DebugGetInfo),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::DebugSetupValue) => {
                match self.debug_setupvalue(frames, &arguments) {
                    Ok(results) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        true,
                        results,
                    ),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::DebugSetupValue),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::DebugUpvalueId) => {
                match self.debug_upvalueid(&arguments) {
                    Ok(results) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        true,
                        results,
                    ),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::DebugUpvalueId),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::DebugUpvalueJoin) => {
                match self.debug_upvaluejoin(frames, &arguments) {
                    Ok(results) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        true,
                        results,
                    ),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::DebugUpvalueJoin),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::GetFenv) => {
                // Lua's protected C-call boundary occupies one stack level
                // for numeric getfenv lookups. The VM normally represents
                // pcall as a continuation, so restore that logical level here.
                let mut protected_arguments = arguments;
                if let Some(Value::Number(level)) = protected_arguments.first_mut() {
                    *level += 1.0;
                }
                match self.get_fenv(frames, &protected_arguments) {
                    Ok(results) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        true,
                        results,
                    ),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::GetFenv),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::SetFenv) => {
                match self.set_fenv(frames, &arguments) {
                    Ok(results) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        true,
                        results,
                    ),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::SetFenv),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::StringDump) => {
                match self.string_dump(&arguments) {
                    Ok(results) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        true,
                        results,
                    ),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::StringDump),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::ToString) => {
                match self.start_tostring_protected(frames, instruction, arguments) {
                    Ok(()) => Ok(()),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::ToString),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::StringFormat) => {
                match self.start_string_format_protected(frames, instruction, arguments) {
                    Ok(()) => Ok(()),
                    Err(error) => {
                        let error = self.protected_argument_error(
                            &Value::BuiltinFunction(BuiltinFunction::StringFormat),
                            error,
                        );
                        self.write_protected_results(
                            &mut frames.at_mut(frame_index),
                            instruction.a(),
                            instruction.c(),
                            false,
                            vec![self.protected_error_object(
                                &Value::BuiltinFunction(BuiltinFunction::StringFormat),
                                error,
                            )?],
                        )
                    }
                }
            }
            Value::BuiltinFunction(BuiltinFunction::StringGsub) => {
                match self.start_string_gsub_with_completion(
                    frames,
                    GsubCompletion::Protected(CallReturn {
                        base: instruction.a(),
                        results: instruction.c(),
                    }),
                    &arguments,
                ) {
                    Ok(()) => Ok(()),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::StringGsub),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::CollectGarbage) => {
                match self.collect_garbage(frames, &arguments) {
                    Ok(CollectGarbageAction::Immediate(results)) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        true,
                        results,
                    ),
                    Ok(CollectGarbageAction::Collected(results)) => self
                        .advance_userdata_finalizers(
                            frames,
                            FinalizerCompletion::Protected {
                                target: CallReturn {
                                    base: instruction.a(),
                                    results: instruction.c(),
                                },
                                values: results,
                            },
                        )
                        .or_else(|error| {
                            self.write_protected_results(
                                &mut frames.at_mut(frame_index),
                                instruction.a(),
                                instruction.c(),
                                false,
                                vec![self.protected_error_object(
                                    &Value::BuiltinFunction(BuiltinFunction::CollectGarbage),
                                    error,
                                )?],
                            )
                        }),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::CollectGarbage),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::CoroutineCreate) => {
                match self.create_coroutine(&arguments) {
                    Ok(results) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        true,
                        results,
                    ),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::CoroutineCreate),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::CoroutineResume) => {
                let result =
                    coroutine_argument(&arguments, self.compatibility).and_then(|thread| {
                        self.resume_coroutine(frames, thread, arguments[1..].to_vec())
                    });
                match result {
                    Ok(results) => self.write_protected_results(
                        &mut frames.last_mut().expect("protected caller exists"),
                        instruction.a(),
                        instruction.c(),
                        true,
                        results,
                    ),
                    Err(error) => self.write_protected_results(
                        &mut frames.last_mut().expect("protected caller exists"),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::CoroutineResume),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::CoroutineYield)
                if self.compatibility != crate::compat::CompatibilityProfile::Lua51 =>
            {
                // A one-instruction frame stands for the native yield, so the
                // coroutine suspends with `pcall` still on its stack and the
                // resume values return through the protected target.
                let target = ReturnTarget::Protected(CallReturn {
                    base: instruction.a(),
                    results: instruction.c(),
                });
                match self.push_builtin_call_frame(
                    frames,
                    BuiltinFunction::CoroutineYield,
                    arguments,
                    target,
                ) {
                    Ok(()) => Ok(()),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![vm_error_object(&error)?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::CoroutineYield) => self
                .write_protected_results(
                    &mut frames.at_mut(frame_index),
                    instruction.a(),
                    instruction.c(),
                    false,
                    vec![match self.yield_boundary_error() {
                        RuntimeError::Raised(message) => message,
                        error => vm_error_object(&error.into())?,
                    }],
                ),
            Value::BuiltinFunction(BuiltinFunction::CoroutineStatus) => {
                match self.coroutine_status(&arguments) {
                    Ok(results) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        true,
                        results,
                    ),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::CoroutineStatus),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::CoroutineClose) => {
                match self.close_coroutine(frames, &arguments) {
                    Ok(results) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        true,
                        results,
                    ),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::CoroutineClose),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::CoroutineWrap) => {
                match self.create_coroutine(&arguments) {
                    Ok(results) => {
                        let Value::Thread(thread) = results[0] else {
                            unreachable!()
                        };
                        self.write_protected_results(
                            &mut frames.at_mut(frame_index),
                            instruction.a(),
                            instruction.c(),
                            true,
                            vec![Value::CoroutineWrapper(thread)],
                        )
                    }
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::CoroutineWrap),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::CoroutineRunning) => {
                match self.coroutine_running() {
                    Ok(results) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        true,
                        results,
                    ),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::CoroutineRunning),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::CoroutineIsYieldable) => {
                match self.coroutine_isyieldable(frames, &arguments) {
                    Ok(results) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        true,
                        results,
                    ),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::CoroutineIsYieldable),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(BuiltinFunction::DebugSetHook) => {
                self.start_protected_sethook(frames, instruction, arguments)
            }
            Value::BuiltinFunction(BuiltinFunction::DebugGetLocal) => {
                if let Some(error) = self.getlocal_argument_error(&arguments) {
                    let error = self.protected_argument_error(
                        &Value::BuiltinFunction(BuiltinFunction::DebugGetLocal),
                        error.into(),
                    );
                    return self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::DebugGetLocal),
                            error,
                        )?],
                    );
                }
                let results = self
                    .protected_local_arguments(&arguments)
                    .map_or_else(Vec::new, |arguments| {
                        self.debug_getlocal(frames, &arguments)
                    });
                self.write_protected_results(
                    &mut frames.at_mut(frame_index),
                    instruction.a(),
                    instruction.c(),
                    true,
                    results,
                )?;
                self.observe_native_call(
                    frames,
                    Value::BuiltinFunction(BuiltinFunction::DebugGetLocal),
                    arguments.len(),
                )?;
                Ok(())
            }
            Value::BuiltinFunction(BuiltinFunction::DebugSetLocal) => {
                let result = match self.protected_local_arguments(&arguments) {
                    Some(arguments) => self.debug_setlocal(frames, &arguments),
                    None => Ok(Vec::new()),
                };
                match result {
                    Ok(results) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        true,
                        results,
                    ),
                    Err(error) => self.write_protected_results(
                        &mut frames.at_mut(frame_index),
                        instruction.a(),
                        instruction.c(),
                        false,
                        vec![self.protected_error_object(
                            &Value::BuiltinFunction(BuiltinFunction::DebugSetLocal),
                            error,
                        )?],
                    ),
                }
            }
            Value::BuiltinFunction(
                builtin @ (BuiltinFunction::FileWrite
                | BuiltinFunction::FileSeek
                | BuiltinFunction::FileRead
                | BuiltinFunction::FileClose
                | BuiltinFunction::FileFlush
                | BuiltinFunction::FileSetVbuf
                | BuiltinFunction::FileLines),
            ) => match self.call_file_builtin(builtin, &arguments) {
                Ok(results) => self.write_protected_results(
                    &mut frames.at_mut(frame_index),
                    instruction.a(),
                    instruction.c(),
                    true,
                    results,
                ),
                Err(error) => self.write_protected_results(
                    &mut frames.at_mut(frame_index),
                    instruction.a(),
                    instruction.c(),
                    false,
                    vec![self.protected_error_object(&Value::BuiltinFunction(builtin), error)?],
                ),
            },
            Value::BuiltinFunction(builtin) => match self.call_builtin(builtin, &arguments) {
                Ok(results) => self.write_protected_results(
                    &mut frames.at_mut(frame_index),
                    instruction.a(),
                    instruction.c(),
                    true,
                    results,
                ),
                Err(VmError::InvalidDispatch(_)) => {
                    let target = ReturnTarget::Protected(CallReturn {
                        base: instruction.a(),
                        results: instruction.c(),
                    });
                    self.push_builtin_call_frame(frames, builtin, arguments, target)
                        .or_else(|error| {
                            self.write_protected_results(
                                &mut frames.at_mut(frame_index),
                                instruction.a(),
                                instruction.c(),
                                false,
                                vec![self.protected_error_object(
                                    &Value::BuiltinFunction(builtin),
                                    error,
                                )?],
                            )
                        })
                }
                Err(error) => self.write_protected_results(
                    &mut frames.at_mut(frame_index),
                    instruction.a(),
                    instruction.c(),
                    false,
                    vec![self.protected_error_object(&Value::BuiltinFunction(builtin), error)?],
                ),
            },
            target => self.write_protected_results(
                &mut frames.at_mut(frame_index),
                instruction.a(),
                instruction.c(),
                false,
                vec![Value::from(call_type_error_anonymous(&target).to_string())],
            ),
        }
    }

    /// `pcall(debug.sethook, ...)`, with the hook handling of a direct call:
    /// the call hook active on entry observes sethook, which then replaces or
    /// removes that hook once the call hook has returned.
    fn start_protected_sethook(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        arguments: Vec<Value>,
    ) -> VmResult<()> {
        let frame_index = frames.len() - 1;
        let call_hook_started = self.observe_native_call(
            frames,
            Value::BuiltinFunction(BuiltinFunction::DebugSetHook),
            arguments.len(),
        )?;
        let targets_current_thread = !matches!(arguments.first(), Some(Value::Thread(thread)) if Some(*thread) != self.current_coroutine);
        // A deferred sethook is validated now, so its errors stay inside
        // the protected call.
        let result = if call_hook_started {
            self.parse_debug_sethook(&arguments).map(|_| {
                self.pending_debug_sethook = Some(arguments);
                Vec::new()
            })
        } else {
            self.debug_sethook(&arguments)
        };
        let results = match result {
            Ok(results) => results,
            Err(error) => {
                return self.write_protected_results(
                    &mut frames.at_mut(frame_index),
                    instruction.a(),
                    instruction.c(),
                    false,
                    vec![self.protected_error_object(
                        &Value::BuiltinFunction(BuiltinFunction::DebugSetHook),
                        error,
                    )?],
                );
            }
        };
        self.write_protected_results(
            &mut frames.at_mut(frame_index),
            instruction.a(),
            instruction.c(),
            true,
            results,
        )?;
        if !call_hook_started && targets_current_thread {
            let frame = &mut frames[frame_index];
            let last_line = frame
                .last_interpreted_pc
                .get()
                .and_then(|pc| frame.chunk.debug.line_at(pc));
            let pc = frame.pc;
            let extras = frame.extras();
            extras.last_hook_line = last_line;
            extras.skip_hook_pc_once = Some(pc);
        }
        if !call_hook_started {
            self.fire_debug_hook(frames, DebugHookEvent::Return)?;
        }
        Ok(())
    }

    /// Arguments of a `debug.getlocal` or `debug.setlocal` called by
    /// `pcall`, with the level counted in VM frames. Lua's `pcall` is a C
    /// function at level 1, so the protected call's caller is level 2 there;
    /// the VM runs `pcall` without a frame of its own. `None` selects
    /// `pcall`'s level, which has no locals.
    fn protected_local_arguments(&self, arguments: &[Value]) -> Option<Vec<Value>> {
        let offset = match arguments.first() {
            Some(Value::Thread(thread)) if Some(*thread) != self.current_coroutine => {
                return Some(arguments.to_vec());
            }
            Some(Value::Thread(_)) => 1,
            _ => 0,
        };
        let mut arguments = arguments.to_vec();
        match arguments.get(offset).and_then(value_debug_level) {
            Some(1) => return None,
            Some(level) if level > 1 => {
                arguments[offset] = profile_usize_value(self.compatibility, level - 1).ok()?;
            }
            _ => {}
        }
        Some(arguments)
    }

    pub(crate) fn write_protected_results(
        &mut self,
        frame: &mut FrameMut<'_>,
        base: u8,
        result_count: u8,
        success: bool,
        mut results: Vec<Value>,
    ) -> VmResult<()> {
        if !success && let Some(error) = results.first_mut() {
            *error = self.normalize_nil_error_value(error.clone());
        }
        results.insert(0, Value::Boolean(success));
        write_call_results(frame, Instruction::call(base, 0, result_count), results)?;
        Ok(())
    }
}
