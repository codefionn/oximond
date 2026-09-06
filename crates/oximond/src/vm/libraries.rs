//! Resumable `collectgarbage`, `string.gsub`, and `table.sort`/`foreach` drivers and their argument helpers.

use crate::bytecode::{Instruction, RegisterSet};
use crate::vm::arith::{
    call_metamethod_key, ordered_compare, profile_integer_value, profile_usize_value,
    usize_to_number,
};
use crate::vm::builtins::builtin_table;
use crate::vm::continuations::{
    append_gsub, gsub_arguments, gsub_key, gsub_limit, gsub_string_argument, gsub_template,
    next_gsub_search, root_continuation, skip_gsub_empty_match,
};
use crate::vm::errors::{loaded_function_name, runtime_error_object, vm_error_object};
use crate::vm::registers::{set_register, write_call_results};
use crate::vm::stack::{CallStack, ensure_call_depth};
use crate::vm::{
    CallReturn, CollectGarbageAction, CoroutineState, FinalizerCompletion, ForeachCompletion,
    ForeachContinuation, ForeachMode, Frame, GsubCompletion, IndexCompletion, IndexContinuation,
    MAX_INDEX_HOPS, ReturnTarget, SortAfter, SortComparison, SortCompletion, SortContinuation,
    SortPhase, StringGsubContinuation, Vm, VmError, VmResult,
};

/// Which registers of a Lua frame's window a collection roots.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Window {
    /// Only the registers the chunk's live set holds at the frame's pc. Valid
    /// for frames whose pc is current: running, suspended in a call, or
    /// yielded.
    Live,
    /// The whole window. For frames whose pc may be stale: closing and
    /// unwinding frames (`root_unwind_close`; the error pc says nothing about what close handlers
    /// read), snapshots of frames parked under host callbacks (`host.rs` callback
    /// snapshot), host children (`root_host_child`), and any
    /// collection while a trace is in flight.
    Full,
}
use crate::compat::CompatibilityProfile;
use crate::pattern::{Match as PatternMatch, Pattern};
use crate::runtime::{
    BuiltinFunction, GcMode, GcRootBuilder, GcStepResult, NumberFormat, RuntimeError, TableHandle,
    Value, format_integer,
};
use crate::string::LuaString;

impl Vm {
    /// Replaces literal templates without constructing a resumable gsub
    /// continuation. Tables and functions retain the callback-capable path.
    pub(crate) fn try_string_gsub_direct(
        &mut self,
        arguments: &[Value],
    ) -> VmResult<Option<Vec<Value>>> {
        let (Some(Value::String(source)), Some(Value::String(pattern))) =
            (arguments.first(), arguments.get(1))
        else {
            return Ok(None);
        };
        let Some(Value::String(template)) = arguments.get(2) else {
            return Ok(None);
        };
        // Let the ordinary path produce argument and pattern errors.
        let Ok(maximum) = gsub_limit(arguments.get(3)) else {
            return Ok(None);
        };
        let Ok(pattern) = Pattern::shared(pattern.as_bytes(), false) else {
            return Ok(None);
        };
        let anchored = pattern.is_anchored();
        // Read the byte slices once; LuaString::as_bytes is not inlined across crates.
        let source_bytes = source.as_bytes();
        let source_len = source_bytes.len();
        let template_bytes = template.as_bytes();
        let literal_template = !template_bytes.contains(&b'%');
        let mut output = Vec::with_capacity(source_len);
        let mut cursor = 0;
        let mut search = 0;
        let mut count = 0_usize;
        let modern_empty_matches = self.compatibility.is_lua53_or_later();
        let mut last_match = None;
        while count < maximum && search <= source_len {
            let Some(found) = pattern.find(source_bytes, search).map_err(|error| {
                RuntimeError::message(error.lua_message(self.compatibility.is_lua51_family()))
            })?
            else {
                break;
            };
            if skip_gsub_empty_match(modern_empty_matches, &found, last_match) {
                search = found.end().saturating_add(1);
                continue;
            }
            append_gsub(&mut output, &source_bytes[cursor..found.start()])?;
            if literal_template {
                append_gsub(&mut output, template_bytes)?;
            } else {
                append_gsub(
                    &mut output,
                    &gsub_template(self.compatibility, template_bytes, source_bytes, &found)?,
                )?;
            }
            cursor = found.end();
            last_match = Some(found.end());
            search = if anchored {
                source_len.saturating_add(1)
            } else {
                next_gsub_search(modern_empty_matches, &found, source_len)
            };
            count += 1;
        }
        append_gsub(&mut output, &source_bytes[cursor..])?;
        Ok(Some(vec![
            Value::bytes(output),
            profile_usize_value(self.compatibility, count)?,
        ]))
    }

    #[allow(clippy::too_many_lines)] // Option validation and state updates form one collectgarbage transaction.
    pub(crate) fn collect_garbage(
        &mut self,
        frames: &mut CallStack,
        arguments: &[Value],
    ) -> VmResult<CollectGarbageAction> {
        let option = match arguments.first() {
            None | Some(Value::Nil) => b"collect".as_slice(),
            Some(Value::String(value)) => value.as_bytes(),
            Some(value) => {
                return Err(RuntimeError::bad_argument(1, "string", Some(value)).into());
            }
        };
        // Lua 5.1 and LuaJIT read the second argument for every option.
        if self.compatibility.is_lua51_family()
            && let Some(argument) = arguments.get(1)
            && !matches!(argument, Value::Nil | Value::Number(_) | Value::Integer(_))
        {
            return Err(RuntimeError::bad_argument(2, "number", Some(argument)).into());
        }
        match option {
            b"collect" => {
                let was_running = self.runtime.gc_is_running();
                let roots = self.gc_roots_with_coroutines(frames, arguments);
                self.runtime.collect_full(roots);
                // Lua 5.1 restarts automatic collection after a full cycle.
                // Lua 5.2 and later preserve an explicit stop.
                if self.compatibility.is_lua51_family() {
                    self.runtime.restart_gc();
                }
                self.prune_collected_coroutines();
                if !was_running && self.runtime.gc_is_running() {
                    self.pending_gc_recheck_depth = Some(frames.len());
                    self.refresh_return_watch();
                }
                Ok(CollectGarbageAction::Collected(vec![
                    profile_integer_value(self.compatibility, 0),
                ]))
            }
            b"count" => Ok(CollectGarbageAction::Immediate(vec![Value::Number(
                gc_count_number(self.runtime.gc_count_bytes())?,
            )])),
            b"stop" => {
                self.runtime.stop_gc();
                Ok(CollectGarbageAction::Immediate(Vec::new()))
            }
            b"restart" => {
                self.runtime.restart_gc();
                Ok(CollectGarbageAction::Immediate(Vec::new()))
            }
            b"isrunning" if !self.compatibility.is_lua51_family() => Ok(
                CollectGarbageAction::Immediate(vec![Value::Boolean(self.runtime.gc_is_running())]),
            ),
            b"generational"
                if matches!(
                    self.compatibility,
                    CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
                ) =>
            {
                let previous = self.runtime.set_gc_mode(GcMode::Generational);
                Ok(CollectGarbageAction::Immediate(vec![Value::from(
                    gc_mode_name(previous),
                )]))
            }
            b"incremental"
                if matches!(
                    self.compatibility,
                    CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
                ) =>
            {
                let previous = self.runtime.set_gc_mode(GcMode::Incremental);
                Ok(CollectGarbageAction::Immediate(vec![Value::from(
                    gc_mode_name(previous),
                )]))
            }
            b"step" => {
                let roots = self.gc_roots_with_coroutines(frames, arguments);
                let complete = matches!(
                    self.runtime
                        .gc_step_with_roots(roots, gc_step_size(arguments)?),
                    GcStepResult::Complete
                );
                if complete {
                    self.prune_collected_coroutines();
                }
                let results = vec![Value::Boolean(complete)];
                Ok(if complete {
                    CollectGarbageAction::Collected(results)
                } else {
                    CollectGarbageAction::Immediate(results)
                })
            }
            b"setpause" if self.compatibility != CompatibilityProfile::Lua55 => {
                let value = gc_control_value(arguments)?;
                Ok(CollectGarbageAction::Immediate(vec![Value::Number(
                    self.runtime.set_gc_pause(value),
                )]))
            }
            b"setstepmul" if self.compatibility != CompatibilityProfile::Lua55 => {
                let value = gc_control_value(arguments)?;
                Ok(CollectGarbageAction::Immediate(vec![Value::Number(
                    self.runtime.set_gc_step_multiplier(value),
                )]))
            }
            b"param" if self.compatibility == CompatibilityProfile::Lua55 => {
                let name = match arguments.get(1) {
                    Some(Value::String(name)) => name.as_bytes(),
                    Some(value) => {
                        return Err(RuntimeError::TypeError {
                            operation: "call collectgarbage with",
                            expected: "string parameter name",
                            found: value.type_name(),
                        }
                        .into());
                    }
                    None => {
                        return Err(RuntimeError::message(
                            "collectgarbage 'param' needs a parameter name",
                        )
                        .into());
                    }
                };
                let replacement = match arguments.get(2) {
                    None | Some(Value::Nil) => None,
                    Some(value) => {
                        let value = gc_control_value(&[Value::Nil, value.clone()])?;
                        if value.fract() != 0.0 || value < -1.0 || value > f64::from(i32::MAX) {
                            return Err(RuntimeError::message("GC parameter out of range").into());
                        }
                        (value >= 0.0).then(|| crate::vm::quantize_lua55_gc_parameter(value as u32))
                    }
                };
                let index = match name {
                    b"minormul" => 0,
                    b"majorminor" => 1,
                    b"minormajor" => 2,
                    b"pause" => 3,
                    b"stepmul" => 4,
                    b"stepsize" => 5,
                    _ => return Err(RuntimeError::message("invalid GC parameter").into()),
                };
                let previous = match replacement {
                    Some(value) => self.runtime.set_gc_parameter(index, value),
                    None => self.runtime.gc_parameter(index),
                }
                .expect("validated collector parameter index");
                Ok(CollectGarbageAction::Immediate(vec![
                    profile_integer_value(self.compatibility, previous as i64),
                ]))
            }
            option => Err(RuntimeError::argument_error(
                1,
                format!("invalid option '{}'", String::from_utf8_lossy(option)),
            )
            .into()),
        }
    }

    pub(crate) fn advance_userdata_finalizers(
        &mut self,
        frames: &mut CallStack,
        completion: FinalizerCompletion,
    ) -> VmResult<()> {
        while let Some(finalizer) = self.runtime.take_pending_finalizer()? {
            match self.start_userdata_finalizer(frames, finalizer, &completion) {
                // A Lua finalizer runs in a frame of its own, which resumes
                // the remaining finalizers when it returns.
                Ok(true) => return Ok(()),
                Ok(false) => {}
                Err(error) => self.absorb_finalizer_error(error)?,
            }
        }

        match completion {
            FinalizerCompletion::Automatic => {}
            FinalizerCompletion::Call { target, values } => {
                let mut caller = frames.last_mut().expect("collectgarbage caller exists");
                write_call_results(
                    &mut caller,
                    Instruction::call(target.base, 0, target.results),
                    values,
                )?;
            }
            FinalizerCompletion::Protected { target, values } => {
                let mut caller = frames
                    .last_mut()
                    .expect("protected collectgarbage caller exists");
                self.write_protected_results(
                    &mut caller,
                    target.base,
                    target.results,
                    true,
                    values,
                )?;
            }
        }
        Ok(())
    }

    /// Starts one finalizer. A native one runs to completion and returns
    /// false; a Lua one gets a frame that resumes `completion`, and returns
    /// true.
    fn start_userdata_finalizer(
        &mut self,
        frames: &mut CallStack,
        finalizer: crate::runtime::Finalizer,
        completion: &FinalizerCompletion,
    ) -> VmResult<bool> {
        let mut handler = finalizer.handler;
        let mut arguments = vec![finalizer.object];
        let mut hops = 0_u8;
        loop {
            match handler {
                Value::NativeFunction(function) => {
                    self.call_native(frames, function, &arguments)?;
                    return Ok(false);
                }
                Value::Closure(function) => {
                    let body = self.closure_body(function)?;
                    let frame = self.make_frame(
                        body.chunk(),
                        arguments,
                        body.shared_cells(),
                        Some(ReturnTarget::UserdataFinalizer(Box::new(
                            completion.clone(),
                        ))),
                        None,
                        body.environment(),
                        Some(function),
                    )?;
                    frames.push_within_limit(frame)?;
                    return Ok(true);
                }
                callee => {
                    if hops == MAX_INDEX_HOPS {
                        return Err(RuntimeError::message("metatable chain too long").into());
                    }
                    let Some(metatable) = self.runtime.raw_metatable(&callee)? else {
                        return Err(self.uncallable_finalizer(&callee));
                    };
                    let next = self
                        .runtime
                        .table_get_string_ref(metatable, call_metamethod_key())
                        .cloned()
                        .unwrap_or(Value::Nil);
                    if matches!(next, Value::Nil) {
                        return Err(self.uncallable_finalizer(&callee));
                    }
                    arguments.insert(0, callee);
                    handler = next;
                    hops += 1;
                }
            }
        }
    }

    /// The error for a `__gc` value that cannot be called, which Lua 5.4
    /// and later name as a metamethod.
    fn uncallable_finalizer(&self, callee: &Value) -> VmError {
        let suffix = if self.finalizer_errors_warn() {
            " (metamethod '__gc')"
        } else {
            ""
        };
        RuntimeError::message(format!(
            "attempt to call a {} value{suffix}",
            callee.type_name()
        ))
        .into()
    }

    /// Lua 5.4 and later turn an error in a finalizer into a warning and go
    /// on, as `GCTM` runs each finalizer in a protected call. Earlier
    /// versions pass the error to whatever triggered the collection. An
    /// `os.exit` request always goes through.
    pub(crate) fn absorb_finalizer_error(&mut self, error: VmError) -> VmResult<()> {
        if !self.finalizer_errors_warn()
            || error.exit_request().is_some()
            || matches!(error, VmError::Yield(_) | VmError::CoroutineClosed)
        {
            return Err(self.wrap_finalizer_error(error));
        }
        self.warn_finalizer_error(&error);
        Ok(())
    }

    /// Lua 5.2 and 5.3 raise an error in a finalizer again as
    /// `error in __gc metamethod (message)` (`GCTM` with `LUA_ERRGCMM`),
    /// where a non-string error object reads `no message`. A wrapped error is
    /// marked and passes on unchanged. `GCTM` wraps only a `LUA_ERRRUN`
    /// status, so a memory error passes through as it is. Other versions,
    /// `os.exit` requests, and yields pass through.
    pub(crate) fn wrap_finalizer_error(&self, error: VmError) -> VmError {
        if !self.wraps_finalizer_errors()
            || error.exit_request().is_some()
            || error.is_memory_error()
            || matches!(
                error,
                VmError::Yield(_)
                    | VmError::CoroutineClosed
                    | VmError::Runtime(RuntimeError::FinalizerWrapped(_))
            )
        {
            return error;
        }
        let message = match crate::vm::errors::vm_error_object(&error) {
            Ok(Value::String(text)) => String::from_utf8_lossy(text.as_bytes()).into_owned(),
            Ok(_) => "no message".to_owned(),
            Err(_) => return error,
        };
        RuntimeError::FinalizerWrapped(format!("error in __gc metamethod ({message})")).into()
    }

    /// Whether `GCTM` raises an error in a finalizer again wrapped and with
    /// `luaD_throw`, as Lua 5.2 and 5.3 do ([`Self::wrap_finalizer_error`]).
    pub(crate) const fn wraps_finalizer_errors(&self) -> bool {
        matches!(
            self.compatibility,
            crate::compat::CompatibilityProfile::Lua52 | crate::compat::CompatibilityProfile::Lua53
        )
    }

    pub(crate) const fn finalizer_errors_warn(&self) -> bool {
        matches!(
            self.compatibility,
            crate::compat::CompatibilityProfile::Lua54 | crate::compat::CompatibilityProfile::Lua55
        )
    }

    pub(crate) fn start_string_gsub(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        arguments: &[Value],
    ) -> VmResult<()> {
        self.start_string_gsub_with_completion(
            frames,
            GsubCompletion::Direct(CallReturn {
                base: instruction.a(),
                results: instruction.c(),
            }),
            arguments,
        )
    }

    pub(crate) fn start_string_gsub_with_completion(
        &mut self,
        frames: &mut CallStack,
        completion: GsubCompletion,
        arguments: &[Value],
    ) -> VmResult<()> {
        if let Some(results) = self.try_string_gsub_direct(arguments)? {
            let mut frame = frames.last_mut().expect("string.gsub caller exists");
            match completion {
                GsubCompletion::Direct(target) => write_call_results(
                    &mut frame,
                    Instruction::call(target.base, 0, target.results),
                    results,
                )?,
                GsubCompletion::Protected(target) => self.write_protected_results(
                    &mut frame,
                    target.base,
                    target.results,
                    true,
                    results,
                )?,
            }
            return Ok(());
        }
        let numbers = self.runtime.number_format();
        let source = gsub_string_argument(numbers, arguments.first(), 1)?;
        let pattern = gsub_string_argument(numbers, arguments.get(1), 2)?;
        let replacement = match arguments.get(2) {
            Some(
                replacement @ (Value::String(_)
                | Value::Integer(_)
                | Value::Number(_)
                | Value::Table(_)
                | Value::Closure(_)
                | Value::NativeFunction(_)
                | Value::BuiltinFunction(_)),
            ) => replacement.clone(),
            argument => {
                return Err(self
                    .runtime
                    .expected_argument_error(3, "string/function/table", argument)
                    .into());
            }
        };
        let maximum = gsub_limit(arguments.get(3))?;
        let pattern = Pattern::shared(pattern.as_bytes(), false).map_err(|error| {
            RuntimeError::message(error.lua_message(self.compatibility.is_lua51_family()))
        })?;
        self.advance_string_gsub(
            frames,
            StringGsubContinuation {
                source,
                pattern,
                replacement,
                maximum,
                output: Vec::new(),
                cursor: 0,
                search: 0,
                last_match: None,
                count: 0,
                pending: None,
                completion,
            },
        )
    }

    /// Turns an error raised while a protected `string.gsub` runs on the
    /// native's own behalf (a bad replacement value, a failing native
    /// callback or `__index`) into the `false, error` result of the `pcall`
    /// that called `gsub` directly. The continuation is already consumed by
    /// then, so unwinding cannot find it.
    pub(crate) fn guard_protected_gsub(
        &mut self,
        frames: &mut CallStack,
        completion: GsubCompletion,
        replacement: &Value,
        result: VmResult<()>,
    ) -> VmResult<()> {
        let (Err(error), GsubCompletion::Protected(target)) = (&result, completion) else {
            return result;
        };
        if error.exit_request().is_some() {
            return result;
        }
        let Err(error) = result else { unreachable!() };
        let function = match replacement {
            Value::NativeFunction(_) | Value::BuiltinFunction(_) => replacement.clone(),
            Value::Table(_) => self
                .native_gsub_index(replacement)
                .unwrap_or(Value::BuiltinFunction(BuiltinFunction::StringGsub)),
            _ => Value::BuiltinFunction(BuiltinFunction::StringGsub),
        };
        let object = self.protected_error_object(&function, error)?;
        let mut caller = frames
            .last_mut()
            .expect("protected string.gsub caller exists");
        self.write_protected_results(
            &mut caller,
            target.base,
            target.results,
            false,
            vec![object],
        )?;
        Ok(())
    }

    /// The native function a table replacement uses as `__index`, whose
    /// argument errors keep its own name.
    fn native_gsub_index(&self, table: &Value) -> Option<Value> {
        let metatable = self.runtime.raw_metatable(table).ok()??;
        match self.runtime.table_get(metatable, &Value::from("__index")) {
            Ok(index @ (Value::NativeFunction(_) | Value::BuiltinFunction(_))) => Some(index),
            _ => None,
        }
    }

    /// Names the function in an argument error raised by a library function
    /// that a native called back. Stock Lua 5.2 and later has no call site
    /// for it and finds its field in a loaded module instead.
    fn name_callback_error(&self, function: &Value, error: VmError) -> VmError {
        let VmError::Runtime(runtime_error) = &error else {
            return error;
        };
        let Some((position, detail)) = runtime_error.argument_detail() else {
            return error;
        };
        if self.compatibility.is_lua51_family() {
            return error;
        }
        let Some(name) = loaded_function_name(&self.runtime, function) else {
            return error;
        };
        RuntimeError::message(format!("bad argument #{position} to '{name}' ({detail})")).into()
    }

    pub(crate) fn advance_string_gsub(
        &mut self,
        frames: &mut CallStack,
        continuation: StringGsubContinuation,
    ) -> VmResult<()> {
        let completion = continuation.completion;
        let replacement = continuation.replacement.clone();
        let result = self.advance_string_gsub_inner(frames, continuation);
        self.guard_protected_gsub(frames, completion, &replacement, result)
    }

    #[allow(clippy::too_many_lines)]
    fn advance_string_gsub_inner(
        &mut self,
        frames: &mut CallStack,
        mut continuation: StringGsubContinuation,
    ) -> VmResult<()> {
        let modern_empty_matches = self.compatibility.is_lua53_or_later();
        let anchored = continuation.pattern.is_anchored();
        loop {
            let source_bytes = continuation.source.as_bytes();
            let source_len = source_bytes.len();
            if continuation.count >= continuation.maximum || continuation.search > source_len {
                return self.finish_string_gsub(frames, continuation);
            }
            let found = continuation
                .pattern
                .find(source_bytes, continuation.search)
                .map_err(|error| {
                    RuntimeError::message(error.lua_message(self.compatibility.is_lua51_family()))
                })?;
            let Some(found) = found else {
                return self.finish_string_gsub(frames, continuation);
            };
            if skip_gsub_empty_match(modern_empty_matches, &found, continuation.last_match) {
                continuation.search = found.end().saturating_add(1);
                continue;
            }
            append_gsub(
                &mut continuation.output,
                &source_bytes[continuation.cursor..found.start()],
            )?;
            continuation.cursor = found.end();
            continuation.last_match = Some(found.end());
            // An anchored pattern is tried once, at the start.
            continuation.search = if anchored {
                source_len.saturating_add(1)
            } else {
                next_gsub_search(modern_empty_matches, &found, source_len)
            };
            continuation.count += 1;
            match continuation.replacement.clone() {
                Value::String(template) => append_gsub(
                    &mut continuation.output,
                    &gsub_template(
                        self.compatibility,
                        template.as_bytes(),
                        continuation.source.as_bytes(),
                        &found,
                    )?,
                )?,
                Value::Integer(number) => {
                    append_gsub(&mut continuation.output, &format_integer(number))?;
                }
                Value::Number(number) => {
                    let text = self.runtime.number_format().format(number);
                    append_gsub(&mut continuation.output, &text)?;
                }
                Value::Table(table) => {
                    let key = gsub_key(self.compatibility, continuation.source.as_bytes(), &found)?;
                    let index = self.native_gsub_index(&Value::Table(table));
                    return self
                        .run_index(
                            frames,
                            IndexContinuation::get(
                                Value::Table(table),
                                key,
                                0,
                                IndexCompletion::StringGsub {
                                    continuation: Box::new(continuation),
                                    found,
                                },
                            ),
                        )
                        .map_err(|error| match &index {
                            Some(index) => self.name_callback_error(index, error),
                            None => error,
                        });
                }
                Value::NativeFunction(function) => {
                    let results = self
                        .call_native(
                            frames,
                            function,
                            &gsub_arguments(
                                self.compatibility,
                                continuation.source.as_bytes(),
                                &found,
                            )?,
                        )
                        .map_err(|error| {
                            self.name_callback_error(&Value::NativeFunction(function), error.into())
                        })?;
                    Self::append_gsub_value(
                        self.runtime.number_format(),
                        &mut continuation.output,
                        &continuation.source,
                        &found,
                        results.first().cloned().unwrap_or(Value::Nil),
                    )?;
                }
                Value::Closure(function) => {
                    continuation.pending = Some(found);
                    let body = self.closure_body(function)?;
                    let frame = self.make_frame(
                        body.chunk(),
                        gsub_arguments(
                            self.compatibility,
                            continuation.source.as_bytes(),
                            continuation.pending.as_ref().expect("pending gsub match"),
                        )?,
                        body.shared_cells(),
                        Some(ReturnTarget::StringGsub(Box::new(continuation))),
                        None,
                        body.environment(),
                        Some(function),
                    )?;
                    frames.push_within_limit(frame)?;
                    return Ok(());
                }
                Value::BuiltinFunction(builtin) => {
                    let arguments =
                        gsub_arguments(self.compatibility, continuation.source.as_bytes(), &found)?;
                    let results = match self.call_builtin(builtin, &arguments) {
                        Err(VmError::InvalidDispatch(_)) => {
                            continuation.pending = Some(found);
                            return self.push_builtin_call_frame(
                                frames,
                                builtin,
                                arguments,
                                ReturnTarget::StringGsub(Box::new(continuation)),
                            );
                        }
                        results => results.map_err(|error| {
                            self.name_callback_error(&Value::BuiltinFunction(builtin), error)
                        })?,
                    };
                    Self::append_gsub_value(
                        self.runtime.number_format(),
                        &mut continuation.output,
                        &continuation.source,
                        &found,
                        results.first().cloned().unwrap_or(Value::Nil),
                    )?;
                }
                value => {
                    return Err(RuntimeError::TypeError {
                        operation: "call string.gsub with",
                        expected: "string, number, table, or function",
                        found: value.type_name(),
                    }
                    .into());
                }
            }
        }
    }

    pub(crate) fn resume_string_gsub(
        &mut self,
        frames: &mut CallStack,
        mut continuation: StringGsubContinuation,
        results: &[Value],
    ) -> VmResult<()> {
        let found = continuation
            .pending
            .take()
            .expect("gsub callback has a pending match");
        let completion = continuation.completion;
        let replacement = continuation.replacement.clone();
        let appended = Self::append_gsub_value(
            self.runtime.number_format(),
            &mut continuation.output,
            &continuation.source,
            &found,
            results.first().cloned().unwrap_or(Value::Nil),
        );
        if appended.is_err() {
            return self.guard_protected_gsub(frames, completion, &replacement, appended);
        }
        self.advance_string_gsub(frames, continuation)
    }

    pub(crate) fn append_gsub_value(
        numbers: NumberFormat,
        output: &mut Vec<u8>,
        source: &LuaString,
        found: &PatternMatch,
        value: Value,
    ) -> VmResult<()> {
        match value {
            Value::Nil | Value::Boolean(false) => {
                append_gsub(output, &source.as_bytes()[found.start()..found.end()])
            }
            Value::String(value) => append_gsub(output, value.as_bytes()),
            Value::Integer(value) => append_gsub(output, &format_integer(value)),
            Value::Number(value) => append_gsub(output, &numbers.format(value)),
            value => Err(RuntimeError::message(format!(
                "invalid replacement value (a {})",
                value.type_name()
            ))
            .into()),
        }
    }

    pub(crate) fn finish_string_gsub(
        &mut self,
        frames: &mut CallStack,
        mut continuation: StringGsubContinuation,
    ) -> VmResult<()> {
        append_gsub(
            &mut continuation.output,
            &continuation.source.as_bytes()[continuation.cursor..],
        )?;
        let mut frame = frames.last_mut().expect("string.gsub caller exists");
        let results = vec![
            Value::bytes(continuation.output),
            profile_usize_value(self.compatibility, continuation.count)?,
        ];
        match continuation.completion {
            GsubCompletion::Direct(target) => write_call_results(
                &mut frame,
                Instruction::call(target.base, 0, target.results),
                results,
            )?,
            GsubCompletion::Protected(target) => {
                self.write_protected_results(
                    &mut frame,
                    target.base,
                    target.results,
                    true,
                    results,
                )?;
            }
        }
        Ok(())
    }

    /// Builds the roots of a collection that runs outside a native trace.
    /// It does not add the trace root bank, which is empty whenever a
    /// collection can run today. Anything that collects while a trace is in
    /// flight must go through `gc_roots_with_coroutines`.
    ///
    /// It sets the registers it does not root to nil first (see
    /// [`Self::clear_dead_registers`]), so it needs the stack mutably.
    pub(crate) fn gc_roots(
        thread_environment: TableHandle,
        frames: &mut CallStack,
        arguments: &[Value],
    ) -> crate::runtime::GcRoots {
        Self::clear_dead_registers(frames);
        let mut roots = GcRootBuilder::new();
        roots.value(&Value::Table(thread_environment));
        roots.values(arguments.iter());
        Self::root_call_stack(&mut roots, frames, Window::Live);
        roots.finish()
    }

    /// Like [`Self::gc_roots`], but roots every frame's whole register
    /// window. Frames that error unwinding is closing stopped at an error pc
    /// that no longer says what the close handlers can still read.
    pub(crate) fn gc_roots_full_windows(
        thread_environment: TableHandle,
        frames: &CallStack,
        arguments: &[Value],
    ) -> crate::runtime::GcRoots {
        let mut roots = GcRootBuilder::new();
        roots.value(&Value::Table(thread_environment));
        roots.values(arguments.iter());
        Self::root_call_stack(&mut roots, frames, Window::Full);
        roots.finish()
    }

    /// The registers of `frame`'s window a collection must root.
    ///
    /// Registers outside the chunk's live set for the frame's pc are dead
    /// temporaries (see `Chunk::live_registers`), so they are not roots. That
    /// set lies below `Chunk::live_register_count`. The fast loop and the
    /// general loop leave `pc` differently (after the instruction that ran, or
    /// at the one about to run), so the answer is the union over `pc` and the
    /// last interpreted pc, the way the op-stats retention counts do. The
    /// instruction before `pc` is not added: after a backward jump it is not
    /// a predecessor, and its stale operands would stay roots.
    ///
    /// This is where the conservative fallback applies: the whole window is
    /// rooted when `window` is [`Window::Full`] or when a pc is outside the
    /// chunk's code. Callers choose `Full` for every frame whose pc may no
    /// longer say what the frame can read, see [`Window`].
    pub(crate) fn frame_live_registers(
        frame: &Frame,
        window: Window,
        pinned: &[crate::vm::stack::OpenUpvalue],
    ) -> RegisterSet {
        if window == Window::Full {
            return RegisterSet::ALL;
        }
        let chunk = &frame.chunk;
        let len = chunk.code.len();
        let last = frame.last_interpreted_pc.get();
        if frame.pc >= len || last.is_some_and(|pc| pc >= len) {
            return RegisterSet::ALL;
        }
        let mut live = chunk.live_registers(frame.pc);
        if let Some(last) = last {
            live = live.union(chunk.live_registers(last));
        }
        // A captured register is the variable its cells stand for: clearing
        // or skipping it would change what the closures see. To-be-closed
        // registers are read by unwinding, which the chunk's code does not
        // show.
        for open in pinned {
            if let Ok(register) = u8::try_from(open.slot - frame.base) {
                live.insert(register);
            }
        }
        if let Some(extras) = frame.extras_ref() {
            for register in &extras.close_slots {
                live.insert(*register);
            }
        }
        live
    }

    /// Sets every register a [`Window::Live`] collection does not root to
    /// nil, the way stock Lua clears the dead part of a thread's stack.
    ///
    /// A dead register keeps its old value otherwise. Once the collector has
    /// freed what that value refers to, anything that still read the slot
    /// (`debug.getlocal` on a temporary, a later root scan with a wider
    /// window) would see a handle to a collected object. Every function that
    /// roots a stack with `Window::Live` calls this first, at the moment it
    /// builds those roots.
    ///
    /// It is sound for the same reason the narrow roots are: a register
    /// outside the frame's live set (see `frame_live_registers`) is written
    /// before any instruction reads it on every path from the frame's pc.
    /// Varargs and open results live in the frame record, not in registers.
    /// Captured and to-be-closed registers count as live there, so they are
    /// both kept and rooted.
    ///
    /// Collections run between instructions, after the fast loop has stored
    /// its state and returned, so no loop holds a copy of a cleared slot.
    pub(crate) fn clear_dead_registers(frames: &mut CallStack) {
        frames.for_each_mut_with_open(|frame, pinned| {
            let live = Self::frame_live_registers(frame.frame, Window::Live, pinned);
            if live == RegisterSet::ALL {
                return;
            }
            for (register, slot) in frame.registers.iter_mut().enumerate() {
                let Ok(register) = u8::try_from(register) else {
                    break;
                };
                if !live.contains(register) && !matches!(slot, Value::Nil) {
                    set_register(slot, Value::Nil);
                }
            }
        });
    }

    /// Roots the registers of `registers`, the window of `frame`, that the
    /// frame's live set contains.
    pub(crate) fn root_frame_window(
        roots: &mut GcRootBuilder,
        frame: &Frame,
        registers: &[Value],
        window: Window,
        pinned: &[crate::vm::stack::OpenUpvalue],
    ) {
        let live = Self::frame_live_registers(frame, window, pinned);
        for (register, value) in registers.iter().enumerate() {
            if u8::try_from(register).is_ok_and(|register| live.contains(register))
                // Slots past the 256 addressable registers are never dead
                // temporaries of the chunk; keep them.
                || register > 255
            {
                roots.value(value);
            }
        }
    }

    /// The one helper that roots a call stack: the value stack, with each
    /// Lua frame's window chosen by `window`, and everything each frame
    /// record keeps alive. Slots outside every frame window stay rooted.
    pub(crate) fn root_call_stack(roots: &mut GcRootBuilder, frames: &CallStack, window: Window) {
        let values = frames.values();
        let mut next = 0;
        for frame in frames {
            let base = (frame.base as usize).min(values.len());
            let top = (frame.top as usize).min(values.len()).max(base);
            roots.values(values[next.min(base)..base].iter());
            Self::root_frame_window(
                roots,
                frame,
                &values[base..top],
                window,
                frames.open_in(frame.base, frame.top),
            );
            next = next.max(top);
        }
        roots.values(values[next.min(values.len())..].iter());
        for frame in frames {
            Self::root_frame(roots, frame);
        }
        // An open upvalue reads its slot when marked, and a suspended
        // thread's cells are what keep the slot's value alive after the
        // thread's own snapshot of the stack.
        for open in frames.open_upvalues() {
            roots.cell(open.cell);
        }
    }

    /// Adds everything a suspended Lua frame record keeps alive. Its
    /// registers live on the value stack and are rooted with it.
    pub(crate) fn root_frame(roots: &mut GcRootBuilder, frame: &Frame) {
        roots.value(&Value::Table(frame.environment));
        if let Some(closure) = frame.closure {
            roots.value(&Value::Closure(closure));
        }
        // A tail call names its activation after the closure it replaced,
        // and reads the name from that closure's chunk (`FrameSite::Tail`).
        if let Some(crate::vm::FrameSite::Tail { origin, .. }) = frame.current_call_site() {
            roots.value(&Value::Closure(*origin));
        }
        if let Some(trace) = frame
            .extras_ref()
            .and_then(|extras| extras.traceback.as_ref())
        {
            roots.values(
                trace
                    .debug_frames
                    .iter()
                    .filter_map(|info| info.function.as_ref()),
            );
        }
        if let Some(handler) = frame
            .extras_ref()
            .and_then(|extras| extras.host_close_handler)
        {
            roots.value(&Value::NativeFunction(handler));
        }
        roots.values(frame.open_values().into_iter().flatten());
        for cell in frame.upvalues.iter() {
            roots.cell(*cell);
        }
        if let Some(target) = &frame.caller {
            root_continuation(roots, target);
        }
    }

    /// Rebuilds the runtime roots of every coroutine that yielded since its
    /// last snapshot. A yield only parks the stack and marks it stale; the
    /// snapshot, with the dead-register clearing that goes with it, is taken
    /// here, when a collection is about to read it. The yield also lists the
    /// thread (`Vm::stale_coroutines`), so this visits the stale stacks, not
    /// every coroutine. Every collection builds
    /// its roots through `gc_roots_with_coroutines`, and a parked stack does
    /// not change until it is resumed, so the collector sees the stack a
    /// yield-time snapshot would have shown it.
    pub(crate) fn refresh_stale_coroutine_roots(&mut self) {
        if self.stale_coroutines.is_empty() {
            return;
        }
        // The list keeps its capacity between collections.
        let mut stale = std::mem::take(&mut self.stale_coroutines);
        let mut coroutines = std::mem::take(&mut self.coroutines);
        for thread in stale.drain(..) {
            self.jit.statistics.coroutine_root_visits += 1;
            let Some(coroutine) = coroutines.get_mut(&thread) else {
                continue;
            };
            coroutine.listed = false;
            if !std::mem::take(&mut coroutine.roots_stale) {
                continue;
            }
            let environment = coroutine.environment;
            let CoroutineState::Suspended {
                frames,
                host_resume,
                ..
            } = &mut coroutine.state
            else {
                continue;
            };
            self.jit.statistics.coroutine_root_snapshots += 1;
            let mut roots = GcRootBuilder::new();
            roots.extend(&Self::gc_roots(environment, frames, &[]));
            if let Some(resume) = host_resume {
                Self::root_host_resume(&mut roots, resume);
            }
            // A thread the collector already freed has nothing to root.
            let _ = self.runtime.set_thread_roots(thread, roots.finish());
        }
        self.coroutines = coroutines;
        self.stale_coroutines = stale;
    }

    /// Builds the roots of a collection from the VM's state: `frames`, the
    /// resumer stacks, the trace root bank and the VM's own roots.
    ///
    /// While a native trace run is in flight (see
    /// `TracePolicy::run_in_flight`), every frame window is rooted whole and
    /// nothing is cleared, except when the trace is waiting for a coroutine
    /// it resumed (`Vm::trace_resumer`): only the stack the trace runs on has
    /// registers its pcs do not describe, and the others are rooted as the
    /// interpreter roots them. Otherwise it sets the registers of `frames`
    /// and the resumer stacks that it does not root to nil first (see
    /// [`Self::clear_dead_registers`]).
    pub(crate) fn gc_roots_with_coroutines(
        &mut self,
        frames: &mut CallStack,
        arguments: &[Value],
    ) -> crate::runtime::GcRoots {
        // Every caller passes these roots straight to a collection.
        #[cfg(feature = "op-stats")]
        crate::vm::op_stats::gc_roots(frames);
        self.refresh_stale_coroutine_roots();
        // A trace in flight may have written registers its frames' pcs do
        // not describe, so its collections root whole windows and clear
        // nothing.
        let in_flight = self.jit.trace.run_in_flight();
        let owned = self.trace_resumer.filter(|_| in_flight);
        let all_full = in_flight && owned.is_none();
        let window_of = |stack: Option<usize>| {
            if all_full || (stack.is_some() && stack == owned) {
                Window::Full
            } else {
                Window::Live
            }
        };
        if !all_full {
            Self::clear_dead_registers(frames);
            for (index, stack) in self.resumer_stacks.iter_mut().enumerate() {
                if Some(index) != owned {
                    Self::clear_dead_registers(stack.get_mut());
                }
            }
        }
        let mut roots = GcRootBuilder::new();
        roots.value(&Value::Table(self.thread_environment));
        if let Some(environment) = self.parked_main_environment {
            roots.value(&Value::Table(environment));
        }
        if let Some(thread) = self.main_thread {
            roots.value(&Value::Thread(thread));
        }
        for hook in self.debug_hooks.values() {
            roots.value(&hook.function);
        }
        if let Some(arguments) = &self.pending_debug_sethook {
            roots.values(arguments.iter());
        }
        roots.values(arguments.iter());
        for snapshot in &self.suspended_roots {
            roots.extend(snapshot);
        }
        for host in self.host_functions.values() {
            roots.values(host.captures.iter());
        }
        for snapshot in &self.host_snapshots {
            // A stack parked beneath a native callback. Its frame pcs are
            // the ones the callback was entered at, which may be stale, so
            // the whole windows are roots.
            if let Some(parked) = snapshot.parked_stack() {
                Self::root_call_stack(&mut roots, parked, Window::Full);
            }
            for local in snapshot.built_local_frames().unwrap_or_default() {
                // A debugger snapshot of a frame parked beneath a native
                // callback. Its pc is the one the callback was entered at and
                // the debugger may edit any local afterwards, so the whole
                // window is a root.
                roots.values(local.frame.varargs.iter());
                Self::root_frame_window(
                    &mut roots,
                    &local.frame.frame,
                    &local.frame.registers,
                    Window::Full,
                    &[],
                );
                Self::root_frame(&mut roots, &local.frame.frame);
            }
        }
        if let Some(resume) = &self.pending_host_resume {
            Self::root_host_resume(&mut roots, resume);
        }
        // What a trace's resume gave waits for the interpreter's replay of
        // the call, which may be a collection away.
        if let Some(memo) = &self.resume_memo {
            memo.root(&mut roots);
        }
        // Likewise the error a trace's host call raised.
        if let Some(VmError::Runtime(error) | VmError::LocatedRuntime { error, .. }) =
            &self.vmcall_error
        {
            match error {
                RuntimeError::Raised(value) | RuntimeError::RaisedAtLevel { value, .. } => {
                    roots.value(value);
                }
                _ => {}
            }
        }
        if let Some(child) = &self.pending_host_child {
            Self::root_host_child(&mut roots, child);
        }
        // The lanes of a trace run in flight.
        roots.values(self.jit.trace.root_values().iter());
        for (index, stack) in std::iter::once(&*frames)
            .chain(
                self.resumer_stacks
                    .iter()
                    .map(crate::vm::coroutines::ParkedStack::get),
            )
            .enumerate()
        {
            Self::root_call_stack(&mut roots, stack, window_of(index.checked_sub(1)));
        }
        for (thread, coroutine) in self.coroutines.iter() {
            if matches!(coroutine.state, CoroutineState::Running) {
                roots.value(&Value::Thread(*thread));
            }
        }
        roots.finish()
    }

    /// Reads the elements of a sorted table that has metamethods into a raw
    /// table, the way `table.sort` reads them with `lua_geti`.
    fn copy_proxy_for_sort(&mut self, proxy: &Value) -> VmResult<(TableHandle, usize)> {
        let length = self.length_with_metamethod(proxy)?;
        let length = usize::try_from(length)
            .ok()
            .filter(|length| *length < i32::MAX as usize)
            .ok_or_else(|| RuntimeError::argument_error(1, "array too big"))?;
        let copy = self.runtime.create_table()?;
        for index in 1..=length {
            let index = index as i64;
            let value = self.element(proxy, index)?;
            let key = profile_integer_value(self.compatibility, index);
            self.runtime.table_set(copy, &key, value)?;
        }
        Ok((copy, length))
    }

    /// Stores the sorted copy back through `__newindex`.
    fn store_sorted_proxy(&mut self, proxy: &Value, sort: &SortContinuation) -> VmResult<()> {
        for index in 1..=sort.length {
            let index = index as i64;
            let key = profile_integer_value(self.compatibility, index);
            let value = self.runtime.table_get(sort.table, &key)?;
            self.set_element(proxy, index, value)?;
        }
        Ok(())
    }

    pub(crate) fn start_table_sort(
        &mut self,
        frames: &mut CallStack,
        completion: SortCompletion,
        arguments: &[Value],
    ) -> VmResult<()> {
        let table = builtin_table(arguments)?;
        let comparator = match arguments.get(1) {
            None | Some(Value::Nil) => None,
            Some(
                comparator @ (Value::Closure(_)
                | Value::NativeFunction(_)
                | Value::BuiltinFunction(_)
                | Value::GmatchIterator(_)
                | Value::CoroutineWrapper(_)),
            ) => Some(comparator.clone()),
            argument => {
                return Err(RuntimeError::bad_argument(2, "function", argument).into());
            }
        };
        // Lua 5.3 and later read and write the elements through the
        // metamethods. Sort a raw copy and store it back at the end.
        let (table, proxy, length) = if self.compatibility.is_lua53_or_later()
            && self.runtime.table_needs_metamethods(table)
        {
            let original = Value::Table(table);
            let (copy, length) = self.in_c_call(|vm| vm.copy_proxy_for_sort(&original))?;
            (copy, Some(original), length)
        } else {
            (table, None, sort_length(self.runtime.table_length(table)?)?)
        };
        let continuation = SortContinuation {
            table,
            proxy,
            comparator,
            length,
            phase: SortPhase::Build {
                next_root: length / 2,
            },
            comparison: None,
            completion,
        };
        if continuation.comparator.is_none()
            && self
                .runtime
                .sort_dense_finite_number_prefix(table, length)?
        {
            #[cfg(test)]
            {
                self.table_sort_fast_path_hits += 1;
            }
            return self.finish_table_sort(frames, &continuation);
        }
        self.advance_table_sort(frames, continuation)
    }

    pub(crate) fn start_table_foreach(
        &mut self,
        frames: &mut CallStack,
        completion: ForeachCompletion,
        arguments: &[Value],
        mode: ForeachMode,
    ) -> VmResult<()> {
        let table = builtin_table(arguments)?;
        let callback = arguments.get(1).cloned().unwrap_or(Value::Nil);
        self.advance_table_foreach(
            frames,
            ForeachContinuation {
                table,
                callback,
                mode,
                previous: Value::Nil,
                index: 1,
                completion,
            },
        )
    }

    #[allow(clippy::too_many_lines)]
    pub(crate) fn advance_table_foreach(
        &mut self,
        frames: &mut CallStack,
        mut continuation: ForeachContinuation,
    ) -> VmResult<()> {
        loop {
            let (key, value) = match continuation.mode {
                ForeachMode::All => {
                    let (key, value) = self
                        .runtime
                        .table_next(continuation.table, &continuation.previous)?;
                    if matches!(key, Value::Nil) {
                        return self.finish_table_foreach(frames, continuation, Vec::new());
                    }
                    continuation.previous = key.clone();
                    (key, value)
                }
                ForeachMode::Array => {
                    let key = Value::Number(usize_to_number(continuation.index)?);
                    let value = self.runtime.table_get(continuation.table, &key)?;
                    if matches!(value, Value::Nil) {
                        return self.finish_table_foreach(frames, continuation, Vec::new());
                    }
                    continuation.index = continuation.index.checked_add(1).ok_or_else(|| {
                        RuntimeError::message("table.foreachi index is too large")
                    })?;
                    (key, value)
                }
            };
            let mut function = continuation.callback.clone();
            let mut arguments = vec![key, value];
            let mut hops = 0_u8;
            loop {
                match function {
                    Value::NativeFunction(function) => {
                        let results = self.call_native(frames, function, &arguments)?;
                        if let Some(value) =
                            results.first().filter(|value| !matches!(value, Value::Nil))
                        {
                            return self.finish_table_foreach(
                                frames,
                                continuation,
                                vec![value.clone()],
                            );
                        }
                        break;
                    }
                    Value::GmatchIterator(iterator) => {
                        let results = self.call_gmatch_iterator(iterator)?;
                        if let Some(value) =
                            results.first().filter(|value| !matches!(value, Value::Nil))
                        {
                            return self.finish_table_foreach(
                                frames,
                                continuation,
                                vec![value.clone()],
                            );
                        }
                        break;
                    }
                    Value::Closure(function) => {
                        let body = self.closure_body(function)?;
                        let frame = self.make_frame(
                            body.chunk(),
                            arguments,
                            body.shared_cells(),
                            Some(ReturnTarget::TableForeach(Box::new(continuation))),
                            None,
                            body.environment(),
                            Some(function),
                        )?;
                        frames.push_within_limit(frame)?;
                        return Ok(());
                    }
                    Value::BuiltinFunction(builtin) => {
                        let results = match self.call_builtin(builtin, &arguments) {
                            Err(VmError::InvalidDispatch(_)) => {
                                return self.push_builtin_call_frame(
                                    frames,
                                    builtin,
                                    arguments,
                                    ReturnTarget::TableForeach(Box::new(continuation)),
                                );
                            }
                            results => results?,
                        };
                        if let Some(value) =
                            results.first().filter(|value| !matches!(value, Value::Nil))
                        {
                            return self.finish_table_foreach(
                                frames,
                                continuation,
                                vec![value.clone()],
                            );
                        }
                        break;
                    }
                    callee => {
                        if hops == MAX_INDEX_HOPS {
                            return Err(RuntimeError::message("metatable chain too long").into());
                        }
                        let Some(metatable) = self.runtime.raw_metatable(&callee)? else {
                            return Err(RuntimeError::TypeError {
                                operation: "call",
                                expected: "function",
                                found: callee.type_name(),
                            }
                            .into());
                        };
                        let handler = self
                            .runtime
                            .table_get_string_ref(metatable, call_metamethod_key())
                            .cloned()
                            .unwrap_or(Value::Nil);
                        if matches!(handler, Value::Nil) {
                            return Err(RuntimeError::TypeError {
                                operation: "call",
                                expected: "function",
                                found: callee.type_name(),
                            }
                            .into());
                        }
                        arguments.insert(0, callee);
                        function = handler;
                        hops += 1;
                    }
                }
            }
        }
    }

    pub(crate) fn resume_table_foreach(
        &mut self,
        frames: &mut CallStack,
        continuation: ForeachContinuation,
        results: &[Value],
    ) -> VmResult<()> {
        if let Some(value) = results.first().filter(|value| !matches!(value, Value::Nil)) {
            return self.finish_table_foreach(frames, continuation, vec![value.clone()]);
        }
        self.advance_table_foreach(frames, continuation)
    }

    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn finish_table_foreach(
        &mut self,
        frames: &mut CallStack,
        continuation: ForeachContinuation,
        results: Vec<Value>,
    ) -> VmResult<()> {
        let mut frame = frames
            .last_mut()
            .expect("table.foreach caller frame exists");
        match continuation.completion {
            ForeachCompletion::Direct(target) => write_call_results(
                &mut frame,
                Instruction::call(target.base, 0, target.results),
                results,
            )?,
            ForeachCompletion::Protected(target) => {
                self.write_protected_results(
                    &mut frame,
                    target.base,
                    target.results,
                    true,
                    results,
                )?;
            }
        }
        Ok(())
    }

    /// Runs heap-sort steps until a Lua comparator must execute.
    ///
    /// The state machine holds no Rust references across a comparator call.
    /// Comparators may yield, run garbage collection, or change the sorted
    /// table, so every comparison and swap reads its current table entries.
    pub(crate) fn advance_table_sort(
        &mut self,
        frames: &mut CallStack,
        mut continuation: SortContinuation,
    ) -> VmResult<()> {
        loop {
            match continuation.phase {
                SortPhase::Build { next_root } => {
                    if next_root == 0 {
                        continuation.phase = SortPhase::Extract {
                            next_end: continuation.length,
                        };
                    } else {
                        continuation.phase = SortPhase::Sift {
                            root: next_root,
                            end: continuation.length,
                            after: SortAfter::Build {
                                next_root: next_root - 1,
                            },
                        };
                    }
                }
                SortPhase::Extract { next_end } => {
                    if next_end <= 1 {
                        return self.finish_table_sort(frames, &continuation);
                    }
                    self.swap_table_sequence_values(continuation.table, 1, next_end)?;
                    continuation.phase = SortPhase::Sift {
                        root: 1,
                        end: next_end - 1,
                        after: SortAfter::Extract {
                            next_end: next_end - 1,
                        },
                    };
                }
                SortPhase::Sift { root, end, after } => {
                    let child = root * 2;
                    match child.cmp(&end) {
                        std::cmp::Ordering::Greater => continuation.phase = after.into(),
                        std::cmp::Ordering::Less => {
                            let Some(next) = self.start_table_sort_comparison(
                                frames,
                                continuation,
                                child,
                                child + 1,
                                SortComparison::ChooseLargerChild {
                                    root,
                                    end,
                                    child,
                                    after,
                                },
                            )?
                            else {
                                return Ok(());
                            };
                            continuation = next;
                        }
                        std::cmp::Ordering::Equal => {
                            continuation.phase = SortPhase::CompareRoot {
                                root,
                                end,
                                child,
                                after,
                            };
                        }
                    }
                }
                SortPhase::CompareRoot {
                    root,
                    end,
                    child,
                    after,
                } => {
                    let Some(next) = self.start_table_sort_comparison(
                        frames,
                        continuation,
                        root,
                        child,
                        SortComparison::SwapRootIfLess {
                            root,
                            end,
                            child,
                            after,
                        },
                    )?
                    else {
                        return Ok(());
                    };
                    continuation = next;
                }
            }
        }
    }

    pub(crate) fn start_table_sort_comparison(
        &mut self,
        frames: &mut CallStack,
        mut continuation: SortContinuation,
        left_index: usize,
        right_index: usize,
        comparison: SortComparison,
    ) -> VmResult<Option<SortContinuation>> {
        let left = self.table_sequence_value(continuation.table, left_index)?;
        let right = self.table_sequence_value(continuation.table, right_index)?;
        continuation.comparison = Some(comparison);
        match continuation.comparator.clone() {
            None => match ordered_compare(&left, &right, "compare") {
                Ok(ordering) => {
                    self.apply_sort_comparison(
                        &mut continuation,
                        ordering.is_some_and(std::cmp::Ordering::is_lt),
                    )?;
                    Ok(Some(continuation))
                }
                Err(error) => {
                    let Some(handler) =
                        self.ordered_metamethod(&left, &right, crate::vm::arith::event::LT)?
                    else {
                        return Err(error.into());
                    };
                    self.call_table_sort_comparator(frames, continuation, handler, &[left, right])
                }
            },
            Some(comparator) => {
                self.call_table_sort_comparator(frames, continuation, comparator, &[left, right])
            }
        }
    }

    pub(crate) fn call_table_sort_comparator(
        &mut self,
        frames: &mut CallStack,
        continuation: SortContinuation,
        comparator: Value,
        arguments: &[Value],
    ) -> VmResult<Option<SortContinuation>> {
        match comparator {
            Value::NativeFunction(function) => {
                let results = self.call_native(frames, function, arguments)?;
                self.apply_table_sort_results(continuation, &results)
            }
            Value::BuiltinFunction(BuiltinFunction::ProtectedCall) => {
                self.start_protected_sort_comparator(frames, &continuation, arguments)?;
                Ok(None)
            }
            Value::BuiltinFunction(builtin) => match self.call_builtin(builtin, arguments) {
                Err(VmError::InvalidDispatch(_)) => {
                    self.push_builtin_call_frame(
                        frames,
                        builtin,
                        arguments.to_vec(),
                        ReturnTarget::TableSort(Box::new(continuation)),
                    )?;
                    Ok(None)
                }
                results => self.apply_table_sort_results(continuation, &results?),
            },
            Value::Closure(function) => {
                let body = self.closure_body(function)?;
                let frame = self.make_frame(
                    body.chunk(),
                    arguments.to_vec(),
                    body.shared_cells(),
                    Some(ReturnTarget::TableSort(Box::new(continuation))),
                    None,
                    body.environment(),
                    Some(function),
                )?;
                frames.push_within_limit(frame)?;
                Ok(None)
            }
            function => Err(RuntimeError::TypeError {
                operation: "call",
                expected: "function",
                found: function.type_name(),
            }
            .into()),
        }
    }

    pub(crate) fn start_protected_sort_comparator(
        &mut self,
        frames: &mut CallStack,
        continuation: &SortContinuation,
        arguments: &[Value],
    ) -> VmResult<()> {
        let target = arguments[0].clone();
        let call_arguments = &arguments[1..];
        match target {
            Value::NativeFunction(function) => {
                match self.call_native(frames, function, call_arguments) {
                    Ok(mut results) => {
                        results.insert(0, Value::Boolean(true));
                        self.resume_table_sort(frames, continuation.clone(), &results)
                    }
                    Err(error) => self.resume_table_sort(
                        frames,
                        continuation.clone(),
                        &[Value::Boolean(false), runtime_error_object(&error)?],
                    ),
                }
            }
            Value::Closure(_) if ensure_call_depth(frames).is_err() => self.resume_table_sort(
                frames,
                continuation.clone(),
                &[Value::Boolean(false), Value::from("stack overflow")],
            ),
            Value::Closure(function) => {
                let body = self.closure_body(function)?;
                let frame = self.make_frame(
                    body.chunk(),
                    call_arguments.to_vec(),
                    body.shared_cells(),
                    Some(ReturnTarget::ProtectedSort(Box::new(continuation.clone()))),
                    None,
                    body.environment(),
                    Some(function),
                )?;
                frames.push_within_limit(frame)?;
                Ok(())
            }
            Value::BuiltinFunction(builtin) => match self.call_builtin(builtin, call_arguments) {
                Ok(mut results) => {
                    results.insert(0, Value::Boolean(true));
                    self.resume_table_sort(frames, continuation.clone(), &results)
                }
                Err(VmError::InvalidDispatch(_)) => {
                    let target = ReturnTarget::ProtectedSort(Box::new(continuation.clone()));
                    match self.push_builtin_call_frame(
                        frames,
                        builtin,
                        call_arguments.to_vec(),
                        target,
                    ) {
                        Ok(()) => Ok(()),
                        Err(error) => self.resume_table_sort(
                            frames,
                            continuation.clone(),
                            &[Value::Boolean(false), vm_error_object(&error)?],
                        ),
                    }
                }
                Err(error) => self.resume_table_sort(
                    frames,
                    continuation.clone(),
                    &[Value::Boolean(false), vm_error_object(&error)?],
                ),
            },
            value => self.resume_table_sort(
                frames,
                continuation.clone(),
                &[
                    Value::Boolean(false),
                    Value::from(
                        RuntimeError::TypeError {
                            operation: "call",
                            expected: "function",
                            found: value.type_name(),
                        }
                        .to_string(),
                    ),
                ],
            ),
        }
    }

    pub(crate) fn resume_table_sort(
        &mut self,
        frames: &mut CallStack,
        mut continuation: SortContinuation,
        results: &[Value],
    ) -> VmResult<()> {
        self.apply_sort_comparison(
            &mut continuation,
            results.first().is_some_and(Value::is_truthy),
        )?;
        self.advance_table_sort(frames, continuation)
    }

    pub(crate) fn apply_table_sort_results(
        &mut self,
        mut continuation: SortContinuation,
        results: &[Value],
    ) -> VmResult<Option<SortContinuation>> {
        self.apply_sort_comparison(
            &mut continuation,
            results.first().is_some_and(Value::is_truthy),
        )?;
        Ok(Some(continuation))
    }

    pub(crate) fn apply_sort_comparison(
        &mut self,
        continuation: &mut SortContinuation,
        less: bool,
    ) -> VmResult<()> {
        let comparison = continuation
            .comparison
            .take()
            .expect("table.sort resumes after a comparison");
        match comparison {
            SortComparison::ChooseLargerChild {
                root,
                end,
                child,
                after,
            } => {
                continuation.phase = SortPhase::CompareRoot {
                    root,
                    end,
                    child: if less { child + 1 } else { child },
                    after,
                };
            }
            SortComparison::SwapRootIfLess {
                root,
                end,
                child,
                after,
            } => {
                if less {
                    self.swap_table_sequence_values(continuation.table, root, child)?;
                    continuation.phase = SortPhase::Sift {
                        root: child,
                        end,
                        after,
                    };
                } else {
                    continuation.phase = after.into();
                }
            }
        }
        Ok(())
    }

    pub(crate) fn finish_table_sort(
        &mut self,
        frames: &mut CallStack,
        continuation: &SortContinuation,
    ) -> VmResult<()> {
        if let Some(proxy) = &continuation.proxy {
            self.in_c_call(|vm| vm.store_sorted_proxy(proxy, continuation))?;
        }
        let mut frame = frames.last_mut().expect("table.sort caller frame exists");
        match continuation.completion {
            SortCompletion::Direct(target) => write_call_results(
                &mut frame,
                Instruction::call(target.base, 0, target.results),
                Vec::new(),
            )?,
            SortCompletion::Protected(target) => {
                self.write_protected_results(
                    &mut frame,
                    target.base,
                    target.results,
                    true,
                    Vec::new(),
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn table_sequence_value(&self, table: TableHandle, index: usize) -> VmResult<Value> {
        self.runtime
            .table_get(table, &Value::Number(usize_to_number(index)?))
            .map_err(Into::into)
    }

    pub(crate) fn table_sequence_set(
        &mut self,
        table: TableHandle,
        index: usize,
        value: Value,
    ) -> VmResult<()> {
        self.runtime
            .table_set(table, &Value::Number(usize_to_number(index)?), value)
            .map_err(Into::into)
    }

    pub(crate) fn swap_table_sequence_values(
        &mut self,
        table: TableHandle,
        left_index: usize,
        right_index: usize,
    ) -> VmResult<()> {
        let left = self.table_sequence_value(table, left_index)?;
        let right = self.table_sequence_value(table, right_index)?;
        self.table_sequence_set(table, left_index, right)?;
        self.table_sequence_set(table, right_index, left)
    }
}

pub(crate) fn gc_control_value(arguments: &[Value]) -> VmResult<f64> {
    let value = arguments.get(1).unwrap_or(&Value::Nil);
    match value {
        Value::Number(value) if value.is_finite() => Ok(*value),
        #[allow(clippy::cast_precision_loss)]
        Value::Integer(value) => Ok(*value as f64),
        value => Err(RuntimeError::bad_argument(2, "number", Some(value)).into()),
    }
}

pub(crate) fn gc_mode_name(mode: GcMode) -> &'static str {
    match mode {
        GcMode::Incremental => "incremental",
        GcMode::Generational => "generational",
    }
}

pub(crate) fn gc_step_size(arguments: &[Value]) -> VmResult<usize> {
    // Every profile reads the step size with luaL_opt*, so an absent size
    // requests a basic step.
    #[allow(clippy::cast_precision_loss)]
    let value = match arguments.get(1).unwrap_or(&Value::Nil) {
        Value::Nil => 0.0,
        Value::Number(value) => *value,
        Value::Integer(value) => *value as f64,
        value => return Err(RuntimeError::bad_argument(2, "number", Some(value)).into()),
    };
    if !value.is_finite() || value < 0.0 {
        return Err(RuntimeError::TypeError {
            operation: "step collectgarbage with",
            expected: "finite non-negative number",
            found: "number",
        }
        .into());
    }
    Ok(value
        .floor()
        .to_string()
        .parse::<usize>()
        .unwrap_or(usize::MAX))
}

pub(crate) fn gc_count_number(bytes: usize) -> VmResult<f64> {
    let bytes = bytes.to_string().parse::<f64>().map_err(|_| {
        RuntimeError::message("garbage-collector byte count exceeds Lua number range")
    })?;
    Ok(bytes / 1024.0)
}

pub(crate) fn sort_length(length: f64) -> VmResult<usize> {
    if !length.is_finite() || length < 0.0 || length.fract() != 0.0 {
        return Err(RuntimeError::message("table length is not a valid sequence size").into());
    }
    format!("{length:.0}")
        .parse()
        .map_err(|_| RuntimeError::message("table length is too large to sort").into())
}
