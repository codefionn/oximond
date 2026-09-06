//! Parked Lua calls and native continuations across modern C API yields.

use crate::bytecode::{Chunk, DebugMetadata, Instruction};
use crate::runtime::{GcRootBuilder, RuntimeError, ThreadHandle, Value};
use crate::vm::errors::{locate_runtime_error, vm_error_object};
use crate::vm::libraries::Window;
use crate::vm::registers::write_call_results;
use crate::vm::stack::CallStack;
use crate::vm::{
    CallReturn, CoroutineState, HostChild, HostResume, UnwindAction, UnwindClose, UnwindCompletion,
    Vm, VmError, VmResult,
};

impl Vm {
    /// Closes a thread and makes it ready for a new entry function.
    ///
    /// # Errors
    ///
    /// Returns an error when `thread` is not a live coroutine, when closing it
    /// fails, or when the thread's GC roots cannot be updated.
    ///
    /// # Panics
    ///
    /// Panics if the coroutine record is missing after the thread was closed.
    pub fn reset_host_coroutine(&mut self, thread: ThreadHandle) -> VmResult<Vec<Value>> {
        if self.host_coroutine_status(thread)? == crate::vm::HostCoroutineStatus::Running {
            return self.close_host_coroutine(thread);
        }
        let result = self.close_host_coroutine(thread)?;
        let coroutine = self
            .coroutines
            .get_mut(&thread)
            .expect("closed thread exists");
        coroutine.state = CoroutineState::New(None);
        coroutine.traceback = None;
        let mut roots = GcRootBuilder::new();
        roots.value(&Value::Table(coroutine.environment));
        self.runtime.set_thread_roots(thread, roots.finish())?;
        Ok(result)
    }

    /// Closes a host thread, runs its pending Lua and native close handlers,
    /// and discards parked native continuations. Results use coroutine.close's
    /// success flag, followed by an error when closing fails.
    ///
    /// # Errors
    ///
    /// Returns an error when `thread` is not a live coroutine, when a close
    /// handler frame cannot be created, or when the thread's GC roots cannot
    /// be updated. Close handler failures themselves come back in the result
    /// values, not as errors.
    pub fn close_host_coroutine(&mut self, thread: ThreadHandle) -> VmResult<Vec<Value>> {
        let coroutine = self
            .coroutines
            .get_mut(&thread)
            .ok_or_else(|| RuntimeError::message("invalid coroutine"))?;
        if matches!(coroutine.state, CoroutineState::Running) {
            return Ok(vec![
                Value::Boolean(false),
                Value::from("cannot close a running coroutine"),
            ]);
        }
        let environment = coroutine.environment;
        let state = std::mem::replace(&mut coroutine.state, CoroutineState::Dead);
        let failed = coroutine.failed.take();
        coroutine.traceback = None;
        let (mut closing, close_error) =
            failed.unwrap_or((CallStack::new(), VmError::CoroutineClosed));
        closing.for_each_mut(|mut frame| {
            frame.clear_plain_mark();
            frame.caller = None;
        });
        if let CoroutineState::Suspended {
            mut frames,
            host_resume,
            ..
        } = state
        {
            // The frames move into the closing stack; their captured locals
            // close first, in place.
            self.close_upvalues(&mut frames, 0);
            for mut frame in frames.into_owned_frames() {
                frame.caller = None;
                closing.push(frame);
            }
            let mut resume = host_resume;
            while let Some(continuation) = resume {
                if let Some(handler) = continuation.close_handler {
                    let mut frame = self.make_frame(
                        Chunk::new(1, Vec::new(), vec![Instruction::halt()]),
                        Vec::new(),
                        Vec::new(),
                        None,
                        None,
                        environment,
                        None,
                    )?;
                    frame.extras().host_close_handler = Some(handler);
                    closing.push(frame);
                }
                let Some(child) = continuation.child else {
                    break;
                };
                let mut child_frames = child.frames;
                self.close_upvalues(&mut child_frames, 0);
                for mut frame in child_frames.into_owned_frames() {
                    frame.caller = None;
                    closing.push(frame);
                }
                resume = child.host_resume;
            }
        }
        let previous_coroutine = self.current_coroutine.replace(thread);
        let previous_environment = std::mem::replace(&mut self.thread_environment, environment);
        let mut roots = GcRootBuilder::new();
        roots.value(&Value::Thread(thread));
        roots.extend(&Self::gc_roots_full_windows(environment, &closing, &[]));
        if let Ok(value) = vm_error_object(&close_error) {
            roots.value(&value);
        }
        self.suspended_roots.push(roots.finish());
        let mut active = CallStack::new();
        // coroutine.close is a non-yieldable C call, including its __close callbacks.
        self.host_yield_boundaries.push((Some(thread), false));
        let result = (|| {
            let mut action = self.advance_unwind_close(
                &mut active,
                UnwindClose {
                    error: close_error,
                    frames: closing,
                    completion: UnwindCompletion::Unprotected,
                },
            )?;
            loop {
                match action {
                    UnwindAction::Return(VmError::CoroutineClosed) => {
                        return Ok(vec![Value::Boolean(true)]);
                    }
                    UnwindAction::Return(error) => {
                        return Ok(vec![
                            Value::Boolean(false),
                            self.normalize_nil_error_value(vm_error_object(&error)?),
                        ]);
                    }
                    UnwindAction::Continue => {}
                }
                action = match self.interpret(&mut active, false) {
                    Err(error) => {
                        if active.is_empty() {
                            UnwindAction::Return(error)
                        } else if let Some(action) =
                            self.resume_unwind_handler_error(&mut active, error.clone())?
                        {
                            action
                        } else {
                            self.begin_error_unwind(&mut active, error)?
                        }
                    }
                    Ok(_) => continue,
                };
            }
        })();
        self.host_yield_boundaries.pop();
        self.current_coroutine = previous_coroutine;
        self.thread_environment = previous_environment;
        self.suspended_roots.pop();
        self.runtime
            .set_thread_roots(thread, GcRootBuilder::new().finish())?;
        result
    }

    /// Yields from a host callback and calls `continuation` on resume.
    /// A direct yield passes resume arguments unchanged. After a resumable
    /// child call finishes, arguments are its success flag followed by results.
    /// Allocator failures use integer 4 in place of false so the embedding
    /// can retain `LUA_ERRMEM` when resuming a protected C continuation.
    /// Hook errors use integer 2 so the embedding can suppress hooks while
    /// running their message handler.
    // The by-value `Value` argument is part of the public embedding API.
    #[allow(clippy::needless_pass_by_value)]
    pub fn request_host_yield_with_continuation(
        &mut self,
        values: Vec<Value>,
        continuation: Value,
    ) -> VmError {
        if let Err(error) = self.validate_host_yield() {
            return error;
        }
        let Value::NativeFunction(function) = continuation else {
            return RuntimeError::message("host continuation must be a host function").into();
        };
        if !self.is_host_function(function) {
            return RuntimeError::message("host continuation must be a host function").into();
        }
        self.pending_host_resume = Some(HostResume {
            function,
            child: self.pending_host_child.take().map(Box::new),
            close_handler: None,
        });
        VmError::Yield(values)
    }

    /// Attaches C-stack cleanup to a pending continuation. The handler receives
    /// nil for normal close, or the error that skipped the continuation.
    ///
    /// # Errors
    ///
    /// Returns an error when `handler` is not a host function, or when no host
    /// continuation is pending.
    // The by-value `Value` argument is part of the public embedding API.
    #[allow(clippy::needless_pass_by_value)]
    pub fn set_host_continuation_close_handler(&mut self, handler: Value) -> VmResult<()> {
        let Value::NativeFunction(function) = handler else {
            return Err(RuntimeError::message("host close handler must be a host function").into());
        };
        if !self.is_host_function(function) {
            return Err(RuntimeError::message("host close handler must be a host function").into());
        }
        let resume = self
            .pending_host_resume
            .as_mut()
            .ok_or_else(|| RuntimeError::message("no host continuation is pending"))?;
        resume.close_handler = Some(function);
        Ok(())
    }

    /// Calls a Lua value from a host callback while allowing modern C API
    /// suspension. On Yield, the caller must return that signal using
    /// `request_host_yield_with_continuation` to retain the native continuation.
    /// `protected` controls whether an error after resume reaches that
    /// continuation or unwinds the native caller.
    ///
    /// # Errors
    ///
    /// Returns whatever error calling `function` raises, or an error creating
    /// its call frame.
    pub fn call_value_resumable(
        &mut self,
        function: Value,
        mut arguments: Vec<Value>,
        protected: bool,
    ) -> VmResult<Vec<Value>> {
        let Some(boundary) = self.host_yield_boundaries.len().checked_sub(1) else {
            return self.call_value(function, arguments);
        };
        let chunk = Chunk::with_varargs(
            1,
            1,
            Vec::new(),
            vec![
                Instruction::call_vararg(0, 0, 0),
                Instruction::return_open(0, 0),
            ],
            Vec::new(),
            Vec::new(),
            None,
        )
        .with_debug_metadata(DebugMetadata::new(crate::vm::host::C_SOURCE, Vec::new()));
        arguments.insert(0, function);
        let frame = self.make_frame(
            chunk,
            arguments,
            Vec::new(),
            None,
            None,
            self.thread_environment,
            None,
        )?;
        self.resumable_host_boundaries.push(boundary);
        let result = self.run_resumable_host_stack(CallStack::with_frame(frame), protected, None);
        self.resumable_host_boundaries.pop();
        result
    }

    pub(crate) fn host_boundary_allows_reentrant_yield(&self, index: usize) -> bool {
        self.resumable_host_boundaries.contains(&index) && self.host_yield_boundaries[index].1
    }

    fn run_resumable_host_stack(
        &mut self,
        mut frames: CallStack,
        protected: bool,
        mut initial_error: Option<VmError>,
    ) -> VmResult<Vec<Value>> {
        loop {
            let step = initial_error
                .take()
                .map_or_else(|| self.interpret(&mut frames, false), Err);
            match step {
                Ok(Some(values)) => return Ok(values),
                Ok(None) => {
                    if let Some((yield_return, mut values, has_status)) = self.take_pending_yield()
                    {
                        if has_status {
                            values.remove(0);
                        }
                        self.pending_host_child = Some(HostChild {
                            frames,
                            yield_return,
                            host_resume: self.pending_host_resume.take(),
                            protected,
                        });
                        return Err(VmError::Yield(values));
                    }
                }
                Err(error) => {
                    let error = locate_runtime_error(&frames, self.compatibility, error);
                    let action = if let Some(action) =
                        self.resume_unwind_handler_error(&mut frames, error.clone())?
                    {
                        action
                    } else {
                        self.begin_error_unwind(&mut frames, error)?
                    };
                    if let UnwindAction::Return(error) = action {
                        return Err(error);
                    }
                }
            }
        }
    }

    /// Completes a suspended call before the outer Lua caller can continue.
    pub(crate) fn resume_host_call(
        &mut self,
        frames: &mut CallStack,
        resume: Option<HostResume>,
        arguments: Vec<Value>,
    ) -> VmResult<Vec<Value>> {
        // A plain `coroutine.yield` parks no host continuation: the resume
        // values are the call's results and nothing runs that could collect.
        if resume.is_none() {
            return Ok(arguments);
        }
        let mut roots = GcRootBuilder::new();
        roots.extend(&Self::gc_roots(self.thread_environment, frames, &arguments));
        if let Some(resume) = &resume {
            Self::root_host_resume(&mut roots, resume);
        }
        self.suspended_roots.push(roots.finish());
        let result = self.resume_host_call_inner(frames, resume, arguments);
        self.suspended_roots.pop();
        result
    }

    fn resume_host_call_inner(
        &mut self,
        frames: &mut CallStack,
        resume: Option<HostResume>,
        arguments: Vec<Value>,
    ) -> VmResult<Vec<Value>> {
        let Some(mut resume) = resume else {
            return Ok(arguments);
        };
        let arguments = if let Some(child) = resume.child.take() {
            let HostChild {
                frames: mut child_frames,
                yield_return,
                host_resume,
                protected,
            } = *child;
            let initial_error =
                match self.resume_host_call(&mut child_frames, host_resume, arguments) {
                    Ok(values) => {
                        self.write_host_resume_results(&mut child_frames, yield_return, values)?;
                        None
                    }
                    Err(VmError::Yield(values)) => {
                        resume.child = Some(Box::new(HostChild {
                            frames: child_frames,
                            yield_return,
                            protected,
                            host_resume: self.pending_host_resume.take(),
                        }));
                        self.pending_host_resume = Some(resume);
                        return Err(VmError::Yield(values));
                    }
                    Err(error) => Some(error),
                };
            let result = self.run_resumable_host_stack(child_frames, protected, initial_error);
            match result {
                Ok(mut values) => {
                    values.insert(0, Value::Boolean(true));
                    values
                }
                Err(VmError::Yield(values)) => {
                    if let Some(child) = self.pending_host_child.take() {
                        resume.child = Some(Box::new(child));
                    }
                    self.pending_host_resume = Some(resume);
                    return Err(VmError::Yield(values));
                }
                Err(error) if protected && error.exit_request().is_none() => {
                    vec![
                        if error.is_memory_error() {
                            Value::Integer(4)
                        } else if error.is_hook_error() {
                            Value::Integer(2)
                        } else {
                            Value::Boolean(false)
                        },
                        vm_error_object(&error)?,
                    ]
                }
                Err(error) => {
                    if let Some(handler) = resume.close_handler {
                        self.call_native(frames, handler, &[vm_error_object(&error)?])?;
                    }
                    return Err(error);
                }
            }
        } else {
            arguments
        };
        let values = self.call_native_yieldable(frames, resume.function, &arguments, true)?;
        if let Some(values) = self.pending_host_yield.take() {
            return Err(VmError::Yield(values));
        }
        Ok(values)
    }

    pub(crate) fn write_host_resume_results(
        &mut self,
        frames: &mut CallStack,
        target: CallReturn,
        values: Vec<Value>,
    ) -> VmResult<()> {
        let index = frames.len() - 1;
        let call = Instruction::call(target.base, 0, target.results);
        if target.results == 0 {
            return write_call_results(&mut frames.at_mut(index), call, values);
        }
        // Fixed results copy out of the vector, which then goes back to the
        // spare buffer for the next switch.
        let mut frame = frames.at_mut(index);
        frame.clear_open_results();
        crate::vm::registers::write_fixed_results(frame.registers, call, &values)?;
        self.recycle_values(values);
        Ok(())
    }

    pub(crate) fn root_host_resume(roots: &mut GcRootBuilder, resume: &HostResume) {
        roots.value(&Value::NativeFunction(resume.function));
        if let Some(handler) = resume.close_handler {
            roots.value(&Value::NativeFunction(handler));
        }
        if let Some(child) = &resume.child {
            Self::root_host_child(roots, child);
        }
    }

    pub(crate) fn root_host_child(roots: &mut GcRootBuilder, child: &HostChild) {
        // Parked host child: its frames' pcs may be stale, so root full windows.
        Self::root_call_stack(roots, &child.frames, Window::Full);
        if let Some(resume) = &child.host_resume {
            Self::root_host_resume(roots, resume);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::Constant;
    use crate::compat::CompatibilityProfile;
    use crate::runtime::BuiltinFunction;

    fn direct_yield(vm: &mut Vm, token: u64, args: &[Value]) -> VmResult<Vec<Value>> {
        let continuation = vm.register_host_function(direct_resume, token, Vec::new())?;
        Err(vm.request_host_yield_with_continuation(args.to_vec(), continuation))
    }

    fn direct_resume(vm: &mut Vm, token: u64, args: &[Value]) -> VmResult<Vec<Value>> {
        vm.runtime_mut()
            .set_global("continued", Value::Boolean(true));
        if token == 1 {
            let continuation = vm.register_host_function(direct_resume, 0, Vec::new())?;
            return Err(vm.request_host_yield_with_continuation(args.to_vec(), continuation));
        }
        Ok(args.to_vec())
    }

    fn call_child(vm: &mut Vm, token: u64, args: &[Value]) -> VmResult<Vec<Value>> {
        match vm.call_value_resumable(args[0].clone(), Vec::new(), token == 1) {
            Err(VmError::Yield(values)) => {
                let continuation = vm.register_host_function(direct_resume, 0, Vec::new())?;
                let signal = vm.request_host_yield_with_continuation(values, continuation);
                if token == 2 {
                    let close = vm.register_host_function(record_native_close, 0, Vec::new())?;
                    vm.set_host_continuation_close_handler(close)?;
                }
                Err(signal)
            }
            result => result,
        }
    }

    fn yielding_child(error: bool) -> Chunk {
        let mut code = vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::call(0, 1, 1),
            Instruction::move_(2, 0),
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 2),
            Instruction::call(0, 1, 1),
        ];
        if error {
            code.extend([
                Instruction::get_global(0, 3),
                Instruction::load_constant(1, 4),
                Instruction::call(0, 1, 1),
                Instruction::return_(0),
            ]);
        } else {
            code.extend([
                Instruction::binary(crate::bytecode::OpCode::Add, 0, 0, 2),
                Instruction::return_(0),
            ]);
        }
        Chunk::new(
            3,
            vec![
                Constant::String("yield".into()),
                Constant::Number(11.0),
                Constant::Number(22.0),
                Constant::String("error".into()),
                Constant::String("child failed".into()),
            ],
            code,
        )
    }

    fn record_native_close(vm: &mut Vm, _: u64, args: &[Value]) -> VmResult<Vec<Value>> {
        let lua_closed = vm.runtime().get_global("closed");
        vm.runtime_mut()
            .set_global("native_saw_lua_closed", lua_closed);
        vm.runtime_mut()
            .set_global("native_close_error", args[0].clone());
        vm.runtime_mut()
            .set_global("native_closed", Value::Boolean(true));
        Ok(Vec::new())
    }

    #[test]
    fn direct_host_continuation_can_yield_again() {
        let mut vm = Vm::with_profile(CompatibilityProfile::Lua54);
        let entry = vm
            .register_host_function(direct_yield, 1, Vec::new())
            .unwrap();
        let thread = vm.create_host_coroutine().unwrap();
        assert_eq!(
            vm.start_host_coroutine(thread, entry, vec![Value::Number(11.0)])
                .unwrap(),
            vec![Value::Boolean(true), Value::Number(11.0)]
        );
        assert_eq!(
            vm.resume_host_coroutine(thread, vec![Value::Number(22.0)])
                .unwrap(),
            vec![Value::Boolean(true), Value::Number(22.0)]
        );
        assert_eq!(
            vm.resume_host_coroutine(thread, vec![Value::Number(33.0)])
                .unwrap(),
            vec![Value::Boolean(true), Value::Number(33.0)]
        );
    }

    #[test]
    fn resumable_host_child_retains_lua_frames_through_repeated_yields_and_gc() {
        let mut vm = Vm::with_profile(CompatibilityProfile::Lua54);
        vm.runtime_mut().set_global(
            "yield",
            Value::BuiltinFunction(BuiltinFunction::CoroutineYield),
        );
        let entry = vm
            .register_host_function(call_child, 0, Vec::new())
            .unwrap();
        let child = vm.load_chunk(&yielding_child(false)).unwrap();
        let thread = vm.create_host_coroutine().unwrap();
        assert_eq!(
            vm.start_host_coroutine(thread, entry, vec![child]).unwrap(),
            vec![Value::Boolean(true), Value::Number(11.0)]
        );
        vm.collect_full_with_values(&[Value::Thread(thread)]);
        assert_eq!(
            vm.resume_host_coroutine(thread, vec![Value::Number(5.0)])
                .unwrap(),
            vec![Value::Boolean(true), Value::Number(22.0)]
        );
        vm.collect_full_with_values(&[Value::Thread(thread)]);
        assert_eq!(
            vm.resume_host_coroutine(thread, vec![Value::Number(7.0)])
                .unwrap(),
            vec![
                Value::Boolean(true),
                Value::Boolean(true),
                Value::Number(12.0)
            ]
        );
        assert_eq!(vm.runtime().get_global("continued"), Value::Boolean(true));
    }

    #[test]
    fn nested_resumable_host_calls_preserve_each_native_continuation() {
        let mut vm = Vm::with_profile(CompatibilityProfile::Lua54);
        vm.runtime_mut().set_global(
            "yield",
            Value::BuiltinFunction(BuiltinFunction::CoroutineYield),
        );
        let outer = vm
            .register_host_function(call_child, 0, Vec::new())
            .unwrap();
        let nested = vm
            .register_host_function(call_child, 0, Vec::new())
            .unwrap();
        vm.runtime_mut().set_global("nested", nested);
        let leaf = vm.load_chunk(&yielding_child(false)).unwrap();
        vm.runtime_mut().set_global("leaf", leaf);
        let child = vm
            .load_chunk(&Chunk::new(
                2,
                vec![
                    Constant::String("nested".into()),
                    Constant::String("leaf".into()),
                ],
                vec![
                    Instruction::get_global(0, 0),
                    Instruction::get_global(1, 1),
                    Instruction::call(0, 1, 0),
                    Instruction::return_open(0, 0),
                ],
            ))
            .unwrap();
        let thread = vm.create_host_coroutine().unwrap();
        assert_eq!(
            vm.start_host_coroutine(thread, outer, vec![child]).unwrap(),
            vec![Value::Boolean(true), Value::Number(11.0)]
        );
        vm.collect_full_with_values(&[Value::Thread(thread)]);
        assert_eq!(
            vm.resume_host_coroutine(thread, vec![Value::Number(5.0)])
                .unwrap(),
            vec![Value::Boolean(true), Value::Number(22.0)]
        );
        assert_eq!(
            vm.resume_host_coroutine(thread, vec![Value::Number(7.0)])
                .unwrap(),
            vec![
                Value::Boolean(true),
                Value::Boolean(true),
                Value::Boolean(true),
                Value::Number(12.0)
            ]
        );
    }

    #[test]
    fn resumable_host_child_errors_reach_only_protected_continuations() {
        for protected in [false, true] {
            let mut vm = Vm::with_profile(CompatibilityProfile::Lua54);
            vm.runtime_mut().set_global(
                "yield",
                Value::BuiltinFunction(BuiltinFunction::CoroutineYield),
            );
            crate::stdlib::install_base(vm.runtime_mut()).unwrap();
            let entry = vm
                .register_host_function(call_child, u64::from(protected), Vec::new())
                .unwrap();
            let child = vm.load_chunk(&yielding_child(true)).unwrap();
            let thread = vm.create_host_coroutine().unwrap();
            vm.start_host_coroutine(thread, entry, vec![child]).unwrap();
            vm.resume_host_coroutine(thread, vec![Value::Number(5.0)])
                .unwrap();
            let results = vm
                .resume_host_coroutine(thread, vec![Value::Number(7.0)])
                .unwrap();
            assert_eq!(results[0], Value::Boolean(protected));
            if protected {
                assert_eq!(results[1], Value::Boolean(false));
                assert_eq!(vm.runtime().get_global("continued"), Value::Boolean(true));
            } else {
                assert_eq!(vm.runtime().get_global("continued"), Value::Nil);
            }
            assert!(
                matches!(results.last(), Some(Value::String(message)) if message.as_bytes().ends_with(b"child failed"))
            );
        }
    }

    #[test]
    fn closing_host_coroutine_runs_lua_close_handler_and_discards_continuation() {
        let mut vm = Vm::with_profile(CompatibilityProfile::Lua54);
        let handler = vm
            .load_chunk(&Chunk::with_parts(
                3,
                2,
                vec![Constant::String("closed".into()), Constant::Boolean(true)],
                vec![
                    Instruction::load_constant(2, 1),
                    Instruction::set_global(2, 0),
                    Instruction::return_values(0, 0),
                ],
                Vec::new(),
                Vec::new(),
            ))
            .unwrap();
        let resource = vm.runtime_mut().create_table().unwrap();
        let metatable = vm.runtime_mut().create_table().unwrap();
        vm.runtime_mut()
            .table_set(metatable, &Value::from("__close"), handler)
            .unwrap();
        vm.runtime_mut()
            .raw_set_metatable(&Value::Table(resource), Some(metatable))
            .unwrap();
        vm.runtime_mut()
            .set_global("resource", Value::Table(resource));
        let host = vm
            .register_host_function(direct_yield, 0, Vec::new())
            .unwrap();
        vm.runtime_mut().set_global("host", host);
        let entry = vm
            .load_chunk(&Chunk::new(
                2,
                vec![
                    Constant::String("resource".into()),
                    Constant::String("host".into()),
                ],
                vec![
                    Instruction::get_global(0, 0),
                    Instruction::mark_close(0),
                    Instruction::get_global(1, 1),
                    Instruction::call(1, 0, 0),
                    Instruction::return_values(0, 0),
                ],
            ))
            .unwrap();
        let thread = vm.create_host_coroutine().unwrap();
        let parent = vm
            .register_host_function(call_child, 2, Vec::new())
            .unwrap();
        assert_eq!(
            vm.start_host_coroutine(thread, parent, vec![entry])
                .unwrap(),
            vec![Value::Boolean(true)]
        );
        assert_eq!(
            vm.close_host_coroutine(thread).unwrap(),
            vec![Value::Boolean(true)]
        );
        assert_eq!(vm.runtime().get_global("closed"), Value::Boolean(true));
        assert_eq!(
            vm.runtime().get_global("native_closed"),
            Value::Boolean(true)
        );
        assert_eq!(
            vm.runtime().get_global("native_saw_lua_closed"),
            Value::Boolean(true)
        );
        assert_eq!(vm.runtime().get_global("native_close_error"), Value::Nil);
        assert_eq!(vm.runtime().get_global("continued"), Value::Nil);
        assert_eq!(
            vm.host_coroutine_status(thread).unwrap(),
            crate::vm::HostCoroutineStatus::Dead
        );
        assert_eq!(
            vm.reset_host_coroutine(thread).unwrap(),
            vec![Value::Boolean(true)]
        );
        let entry = vm
            .load_chunk(&Chunk::new(
                1,
                vec![Constant::Number(42.0)],
                vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
            ))
            .unwrap();
        assert_eq!(
            vm.start_host_coroutine(thread, entry, Vec::new()).unwrap(),
            vec![Value::Boolean(true), Value::Number(42.0)]
        );
    }
    fn allocator_failure(_: &mut Vm, _: u64, _: &[Value]) -> VmResult<Vec<Value>> {
        Err(RuntimeError::memory_error().into())
    }

    #[test]
    fn native_memory_errors_retain_type_across_host_calls_and_thread_entry() {
        let mut vm = Vm::with_profile(CompatibilityProfile::Lua54);
        let native = vm
            .register_host_function(allocator_failure, 0, Vec::new())
            .unwrap();
        let error = vm.call_value(native.clone(), Vec::new()).unwrap_err();
        assert!(error.is_memory_error());
        assert_eq!(error.to_string(), "not enough memory");
        assert!(!VmError::from(RuntimeError::message("not enough memory")).is_memory_error());
        let located = VmError::LocatedRuntime {
            source: "@oom.lua".into(),
            line: 3,
            error: RuntimeError::memory_error(),
        };
        assert!(located.is_memory_error());
        assert_eq!(
            crate::vm::errors::locate_runtime_error(&CallStack::new(), vm.compatibility, error)
                .to_string(),
            "not enough memory"
        );
        let thread = vm.create_host_coroutine().unwrap();
        assert!(
            vm.start_host_coroutine(thread, native.clone(), Vec::new())
                .unwrap_err()
                .is_memory_error()
        );
        assert_eq!(
            vm.host_coroutine_status(thread).unwrap(),
            crate::vm::HostCoroutineStatus::Dead
        );
        assert!(vm.host_coroutine_failed(thread).unwrap());
        assert!(vm.current_coroutine.is_none());
        assert!(vm.resumer_stacks.is_empty());
        let Value::Thread(thread) = vm.create_coroutine(std::slice::from_ref(&native)).unwrap()[0]
        else {
            panic!()
        };
        assert_eq!(
            vm.resume_coroutine(&mut CallStack::new(), thread, Vec::new())
                .unwrap(),
            vec![Value::Boolean(false), Value::from("not enough memory")]
        );
        let protected = vm
            .call_value(
                Value::BuiltinFunction(BuiltinFunction::ProtectedCall),
                vec![native],
            )
            .unwrap();
        assert_eq!(
            protected,
            vec![Value::Boolean(false), Value::from("not enough memory")]
        );
    }

    #[test]
    fn resumed_memory_errors_keep_native_status_in_protected_and_unprotected_calls() {
        for protected in [false, true] {
            let mut vm = Vm::with_profile(CompatibilityProfile::Lua54);
            let failing = vm
                .register_host_function(allocator_failure, 0, Vec::new())
                .unwrap();
            vm.runtime_mut().set_global(
                "yield",
                Value::BuiltinFunction(BuiltinFunction::CoroutineYield),
            );
            vm.runtime_mut().set_global("error", failing);
            let entry = vm
                .register_host_function(call_child, u64::from(protected), Vec::new())
                .unwrap();
            let child = vm.load_chunk(&yielding_child(true)).unwrap();
            let thread = vm.create_host_coroutine().unwrap();
            vm.start_host_coroutine(thread, entry, vec![child]).unwrap();
            vm.resume_host_coroutine(thread, vec![Value::Number(5.0)])
                .unwrap();
            let result = vm.resume_host_coroutine(thread, vec![Value::Number(7.0)]);
            if protected {
                assert_eq!(
                    result.unwrap(),
                    vec![
                        Value::Boolean(true),
                        Value::Integer(4),
                        Value::from("not enough memory")
                    ]
                );
                assert_eq!(vm.runtime().get_global("continued"), Value::Boolean(true));
            } else {
                assert!(result.unwrap_err().is_memory_error());
                assert_eq!(vm.runtime().get_global("continued"), Value::Nil);
                assert!(vm.host_coroutine_failed(thread).unwrap());
            }
            assert!(vm.current_coroutine.is_none());
            assert!(vm.resumer_stacks.is_empty());
        }
    }
    #[test]
    fn resumed_protected_hook_errors_preserve_origin_for_native_message_handlers() {
        fn failed_hook(_: &mut Vm, _: u64, _: &[Value]) -> VmResult<Vec<Value>> {
            Err(VmError::HookError(Box::new(
                RuntimeError::Raised(Value::from("killed")).into(),
            )))
        }
        let mut vm = Vm::with_profile(CompatibilityProfile::Lua54);
        let failing = vm
            .register_host_function(failed_hook, 0, Vec::new())
            .unwrap();
        vm.runtime_mut().set_global(
            "yield",
            Value::BuiltinFunction(BuiltinFunction::CoroutineYield),
        );
        vm.runtime_mut().set_global("error", failing);
        let entry = vm
            .register_host_function(call_child, 1, Vec::new())
            .unwrap();
        let child = vm.load_chunk(&yielding_child(true)).unwrap();
        let thread = vm.create_host_coroutine().unwrap();
        vm.start_host_coroutine(thread, entry, vec![child]).unwrap();
        vm.resume_host_coroutine(thread, vec![Value::Number(5.0)])
            .unwrap();
        assert_eq!(
            vm.resume_host_coroutine(thread, vec![Value::Number(7.0)])
                .unwrap(),
            vec![
                Value::Boolean(true),
                Value::Integer(2),
                Value::from("killed")
            ]
        );
    }
}
