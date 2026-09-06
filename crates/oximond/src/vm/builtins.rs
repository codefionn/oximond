//! Builtin function dispatch, the `debug` library, `tostring`, and `string.format` drivers.

use crate::bytecode::{CallKind, Instruction, OpCode, binary};
use crate::pattern::Pattern;
use crate::runtime::{BuiltinFunction, DirectNativeKind, RuntimeError, ThreadHandle, Value};
use crate::runtime::{
    CaptureCells, FileHandle, GcRootBuilder, Runtime, RuntimeResult, TableHandle,
};
use crate::string::LuaString;
use crate::vm::arith::{number, profile_integer_value, profile_usize_value};
use crate::vm::continuations::{gsub_arguments, skip_gsub_empty_match};
use crate::vm::debug_info::{
    active_call_site, base_trace_frame, capture_traceback, capture_traceback_above,
    debug_closure_argument, debug_info_at_level, debug_level, debug_namewhat, debug_options,
    debug_short_source, debug_upvalue_index, fenv_level, format_traceback_for, frame_environment,
    hook_count, level_number, native_trace_frame, set_frame_environment, traceback_level,
    traceback_message, value_debug_level, value_index,
};
use crate::vm::errors::{file_io_results, loaded_function_name, vm_error_object};
use crate::vm::fast_path::FastExit;
use crate::vm::host::builtin_call_chunk;
use crate::vm::libraries::Window;
use crate::vm::registers::write_call_results;
use crate::vm::stack::CallStack;
use crate::vm::upvalues::replace_active_closure_cell;
use crate::vm::{
    ActiveCallSite, CallReturn, CoroutineState, CountLease, DebugHook, DebugHookEvent,
    DebugHookMask, DebugHookReturn, DebugInfo, FastHookMode, MAX_LUA_CALL_FRAMES, ReturnTarget,
    StringFormatCompletion, StringFormatContinuation, ToStringCompletion, ToStringContinuation,
    Traceback, Vm, VmError, VmResult,
};
use std::io::{Read, Write};
use std::sync::{Arc, OnceLock};

/// The most results of a `string.gmatch` step the generic `for` writes in
/// place.
const GMATCH_RESULTS_IN_PLACE: usize = 8;

const NATIVE_HANDLER_DEPTH_LIMIT: usize = 195;
/// Stock's `LUAI_MAXCCALLS`: nested C calls before "C stack overflow".
const MAX_NESTED_C_CALLS: usize = 200;

fn debug_uservalue_index(arguments: &[Value], position: usize) -> VmResult<usize> {
    let index = match arguments.get(position - 1) {
        None | Some(Value::Nil) => 1,
        Some(_) => crate::vm::table_meta::check_integer(arguments, position)?,
    };
    // Lua casts the checked integer to C int before asking for a slot. Preserve
    // that narrowing, including its wrapping behavior for oversized indices.
    #[allow(clippy::cast_possible_truncation)]
    Ok(usize::try_from(index as i32).unwrap_or(0))
}

impl Vm {
    /// Formats a callback-free modern `string.format` call without a VM
    /// continuation. Non-string `%s` arguments may have `__tostring`, so the
    /// ordinary continuation owns those calls.
    pub(crate) fn try_string_format_direct(
        &mut self,
        arguments: &[Value],
    ) -> VmResult<Option<Vec<Value>>> {
        let Some(function) = self.runtime.direct_native(DirectNativeKind::Format) else {
            return Ok(None);
        };
        let Some(Value::String(format)) = arguments.first() else {
            return Ok(None);
        };
        let mut format_index = 0;
        let mut argument_index = 1;
        while let Some((argument, next_format, next_argument)) =
            next_string_conversion(format.as_bytes(), format_index, argument_index)
        {
            let Some(value @ Value::String(_)) = arguments.get(argument) else {
                return Ok(None);
            };
            if self.tostring_metamethod(value)?.is_some() {
                return Ok(None);
            }
            format_index = next_format;
            argument_index = next_argument;
        }
        // The runtime's own formatter is never a host callback.
        Ok(Some(self.call_native(
            &mut CallStack::new(),
            function,
            arguments,
        )?))
    }

    pub(crate) fn call_builtin(
        &mut self,
        builtin: BuiltinFunction,
        arguments: &[Value],
    ) -> VmResult<Vec<Value>> {
        match builtin {
            BuiltinFunction::PackageSearchPath => self.package_searchpath(arguments),
            BuiltinFunction::PackageSourceSearcher => self.package_source_searcher(arguments),
            BuiltinFunction::ProtectedCall
            | BuiltinFunction::ExtendedProtectedCall
            | BuiltinFunction::TableSort
            | BuiltinFunction::TableForeach
            | BuiltinFunction::TableForeachi
            | BuiltinFunction::DebugGetInfo
            | BuiltinFunction::DebugTraceback
            | BuiltinFunction::StringDump
            | BuiltinFunction::ToString
            | BuiltinFunction::StringFormat
            | BuiltinFunction::StringGsub
            | BuiltinFunction::DebugSetupValue
            | BuiltinFunction::DebugSetHook
            | BuiltinFunction::DebugGetLocal
            | BuiltinFunction::DebugSetLocal
            | BuiltinFunction::DebugUpvalueId
            | BuiltinFunction::DebugUpvalueJoin
            | BuiltinFunction::GetFenv
            | BuiltinFunction::SetFenv
            | BuiltinFunction::CollectGarbage
            | BuiltinFunction::FileWrite
            | BuiltinFunction::FileSeek
            | BuiltinFunction::FileRead
            | BuiltinFunction::FileClose
            | BuiltinFunction::FileFlush
            | BuiltinFunction::FileSetVbuf
            | BuiltinFunction::FileLines
            | BuiltinFunction::Require
            | BuiltinFunction::Module
            | BuiltinFunction::LoadFile
            | BuiltinFunction::DoFile
            | BuiltinFunction::CoroutineCreate
            | BuiltinFunction::CoroutineResume
            | BuiltinFunction::CoroutineYield
            | BuiltinFunction::CoroutineStatus
            | BuiltinFunction::CoroutineClose
            | BuiltinFunction::CoroutineWrap
            | BuiltinFunction::CoroutineRunning
            | BuiltinFunction::CoroutineIsYieldable => Err(VmError::InvalidDispatch(OpCode::Call)),
            BuiltinFunction::Next => {
                let table = builtin_table(arguments)?;
                let previous = arguments.get(1).unwrap_or(&Value::Nil);
                let (key, value) = match self.runtime.table_next_fast(table, previous) {
                    Some(pair) => pair,
                    None => self.runtime.table_next(table, previous)?,
                };
                // The end of the table is one nil, as `lua_next` returning 0
                // leaves `next` one result.
                if matches!(key, Value::Nil) {
                    return Ok(vec![Value::Nil]);
                }
                Ok(vec![key, value])
            }
            BuiltinFunction::Pairs => {
                self.check_iteration_subject(arguments)?;
                let table = builtin_table(arguments)?;
                Ok(vec![
                    Value::BuiltinFunction(BuiltinFunction::Next),
                    Value::Table(table),
                    Value::Nil,
                ])
            }
            BuiltinFunction::IPairs => {
                self.check_iteration_subject(arguments)?;
                let table = builtin_table(arguments)?;
                Ok(vec![
                    Value::BuiltinFunction(BuiltinFunction::IPairsIterator),
                    Value::Table(table),
                    profile_integer_value(self.compatibility, 0),
                ])
            }
            BuiltinFunction::IPairsIterator => {
                let table = builtin_table(arguments)?;
                // Lua 5.3 and later check the control with `luaL_checkinteger`
                // and read the element with `lua_geti`, so a table with
                // `__index` takes the frame path, which runs it.
                let index = if self.compatibility.is_lua53_or_later() {
                    let control = crate::vm::table_meta::check_integer(arguments, 2)?;
                    if self.runtime.table_needs_metamethods(table) {
                        return Err(VmError::InvalidDispatch(OpCode::Call));
                    }
                    Value::Integer(control.wrapping_add(1))
                } else {
                    match arguments.get(1).unwrap_or(&Value::Nil) {
                        Value::Integer(control) => Value::Integer(control.wrapping_add(1)),
                        control => Value::Number(number(control, "iterate ipairs with")? + 1.0),
                    }
                };
                let value = self.runtime.table_get(table, &index)?;
                if matches!(value, Value::Nil) {
                    Ok(vec![Value::Nil])
                } else {
                    Ok(vec![index, value])
                }
            }
            // `load` honours the chunk name, mode and environment, and runs
            // reader functions, so it always goes through the frame stack.
            BuiltinFunction::Load | BuiltinFunction::LoadString => {
                Err(VmError::InvalidDispatch(OpCode::Call))
            }
            BuiltinFunction::DebugGetHook => Ok(self.debug_gethook(arguments)),
            BuiltinFunction::DebugGetUpvalue => self.debug_getupvalue(arguments),
            BuiltinFunction::DebugGetRegistry => Ok(self.debug_getregistry()),
            BuiltinFunction::DebugGetUserValue => self.debug_getuservalue(arguments),
            BuiltinFunction::DebugSetUserValue => self.debug_setuservalue(arguments),
            BuiltinFunction::DebugDebug => self.debug_console(&CallStack::new()),
        }
    }

    /// Starts `builtin` in a frame that calls it through the ordinary call
    /// path and returns its results to `target`, as a Lua callee would.
    /// This runs the builtins that need the frame stack, for which
    /// `call_builtin` reports `InvalidDispatch`, at call sites that
    /// otherwise have no arm for them. The frame stands for the C function
    /// the caller is running, such as `pcall` or `table.sort`, so it takes
    /// the caller's call-site name.
    pub(crate) fn push_builtin_call_frame(
        &mut self,
        frames: &mut CallStack,
        builtin: BuiltinFunction,
        arguments: Vec<Value>,
        target: ReturnTarget,
    ) -> VmResult<()> {
        if frames.len() >= MAX_LUA_CALL_FRAMES {
            return Err(RuntimeError::message("stack overflow").into());
        }
        // Each of these frames stands for a C function that a metamethod or
        // a library function called, and stock bounds those nested calls.
        if frames.len() > MAX_NESTED_C_CALLS
            && !matches!(
                self.compatibility,
                crate::compat::CompatibilityProfile::Lua51
                    | crate::compat::CompatibilityProfile::LuaJit21
            )
            && frames
                .iter()
                .filter(|frame| crate::vm::debug_info::is_c_chunk(&frame.chunk))
                .count()
                >= MAX_NESTED_C_CALLS
        {
            return Err(RuntimeError::Raised(Value::from("C stack overflow")).into());
        }
        let chunk = builtin_call_chunk(arguments.len())?;
        let caller = frames.len() - 1;
        let call_site = active_call_site(&frames[caller]);
        let frame = self.make_frame(
            chunk,
            std::iter::once(Value::BuiltinFunction(builtin)).chain(arguments),
            CaptureCells::default(),
            Some(target),
            call_site,
            self.thread_environment,
            None,
        )?;
        frames.push(frame);
        Ok(())
    }

    /// One step of a `string.gmatch` iterator for a generic `for`: the
    /// results go straight into `registers[destination..][..results]`, nil
    /// padded or cut. `false` before any change for a step the general path
    /// must take: a `utf8.codes` iterator, a pattern error, or more captures
    /// than fit [`GMATCH_RESULTS_IN_PLACE`].
    pub(crate) fn gmatch_step_in_place(
        &mut self,
        iterator: crate::runtime::GmatchIteratorHandle,
        registers: &mut [Value],
        destination: usize,
        results: usize,
    ) -> bool {
        let Some((source, pattern, search, last_match, utf8_codes)) =
            self.runtime.gmatch_iterator_view(iterator)
        else {
            return false;
        };
        if utf8_codes {
            return false;
        }
        let source = source.as_bytes();
        let mut values = [const { Value::Nil }; GMATCH_RESULTS_IN_PLACE];
        // `next_search` is where the iterator resumes; `committed` the end of
        // an accepted match under the Lua 5.3+ empty-match rule.
        let (next_search, committed) = if search > source.len() {
            (search, false)
        } else {
            let Ok(pattern) = Pattern::shared(pattern.as_bytes(), true) else {
                return false;
            };
            let modern_empty_matches = self.compatibility.is_lua53_or_later();
            let mut at = search;
            let found = loop {
                match pattern.find(source, at) {
                    Err(_) => return false,
                    Ok(Some(found))
                        if skip_gsub_empty_match(modern_empty_matches, &found, last_match) =>
                    {
                        if found.end() == source.len() {
                            break None;
                        }
                        at = found.end() + 1;
                    }
                    Ok(found) => break found,
                }
            };
            match found {
                None => (source.len().saturating_add(1), false),
                Some(found) => {
                    let captures = found.captures();
                    if captures.len() > GMATCH_RESULTS_IN_PLACE {
                        return false;
                    }
                    if captures.is_empty() {
                        values[0] = Value::bytes(&source[found.start()..found.end()]);
                    }
                    for (slot, capture) in values.iter_mut().zip(captures) {
                        *slot = if capture.is_position() {
                            profile_integer_value(self.compatibility, capture.start() as i64 + 1)
                        } else {
                            Value::bytes(&source[capture.start()..capture.end()])
                        };
                    }
                    if modern_empty_matches {
                        (found.end(), true)
                    } else if found.start() == found.end() {
                        (found.end().saturating_add(1), false)
                    } else {
                        (found.end(), false)
                    }
                }
            }
        };
        let stored = if committed {
            self.runtime
                .set_gmatch_iterator_last_match(iterator, next_search)
        } else {
            self.runtime
                .set_gmatch_iterator_search(iterator, next_search)
        };
        debug_assert!(stored.is_ok(), "the iterator was just read");
        for (position, slot) in registers[destination..destination + results]
            .iter_mut()
            .enumerate()
        {
            let value = values
                .get_mut(position)
                .map_or(Value::Nil, |value| std::mem::replace(value, Value::Nil));
            crate::vm::registers::set_register(slot, value);
        }
        true
    }

    #[allow(clippy::cast_precision_loss)] // Lua exposes byte indexes as numbers in this f64-backed runtime.
    pub(crate) fn call_gmatch_iterator(
        &mut self,
        iterator: crate::runtime::GmatchIteratorHandle,
    ) -> VmResult<Vec<Value>> {
        let snapshot = self.runtime.gmatch_iterator_snapshot(iterator)?;
        if snapshot.search() > snapshot.source().len() {
            return Ok(vec![Value::Nil]);
        }
        if snapshot.is_utf8_codes() {
            let source = std::str::from_utf8(snapshot.source().as_bytes()).map_err(|error| {
                RuntimeError::message(format!(
                    "invalid UTF-8 code at byte {}",
                    error.valid_up_to() + 1
                ))
            })?;
            let search = snapshot.search();
            if search == source.len() {
                self.runtime
                    .set_gmatch_iterator_search(iterator, search.saturating_add(1))?;
                return Ok(vec![Value::Nil]);
            }
            if !source.is_char_boundary(search) {
                return Err(RuntimeError::message("invalid UTF-8 iterator position").into());
            }
            let character = source[search..]
                .chars()
                .next()
                .expect("nonempty UTF-8 tail");
            self.runtime
                .set_gmatch_iterator_search(iterator, search + character.len_utf8())?;
            return Ok(vec![
                profile_usize_value(self.compatibility, search + 1)?,
                profile_integer_value(self.compatibility, i64::from(u32::from(character))),
            ]);
        }
        let pattern = Pattern::shared(snapshot.pattern().as_bytes(), true).map_err(|error| {
            RuntimeError::message(error.lua_message(self.compatibility.is_lua51_family()))
        })?;
        let source = snapshot.source().as_bytes();
        let modern_empty_matches = self.compatibility.is_lua53_or_later();
        let mut search = snapshot.search();
        let found = loop {
            let found = pattern.find(source, search).map_err(|error| {
                RuntimeError::message(error.lua_message(self.compatibility.is_lua51_family()))
            })?;
            match found {
                Some(found)
                    if skip_gsub_empty_match(
                        modern_empty_matches,
                        &found,
                        snapshot.last_match(),
                    ) =>
                {
                    if found.end() == source.len() {
                        break None;
                    }
                    search = found.end() + 1;
                }
                found => break found,
            }
        };
        let Some(found) = found else {
            self.runtime
                .set_gmatch_iterator_search(iterator, source.len().saturating_add(1))?;
            return Ok(vec![Value::Nil]);
        };
        if modern_empty_matches {
            self.runtime
                .set_gmatch_iterator_last_match(iterator, found.end())?;
            return gsub_arguments(self.compatibility, source, &found);
        }
        let next_search = if found.start() == found.end() {
            if found.end() == snapshot.source().len() {
                snapshot.source().len().saturating_add(1)
            } else {
                found.end() + 1
            }
        } else {
            found.end()
        };
        self.runtime
            .set_gmatch_iterator_search(iterator, next_search)?;
        gsub_arguments(self.compatibility, snapshot.source().as_bytes(), &found)
    }

    /// `file:write`'s results: the file, or Lua's failure triple.
    fn file_write_results(&mut self, file: FileHandle, values: &[Value]) -> VmResult<Vec<Value>> {
        match self.runtime.file_write(file, values) {
            Ok(()) => Ok(vec![Value::File(file)]),
            Err(error @ (RuntimeError::FileIo { .. } | RuntimeError::HostOutputIo { .. })) => {
                Ok(file_io_results(&error))
            }
            Err(error) => Err(error.into()),
        }
    }

    /// `pairs` and `ipairs` without an argument: Lua 5.3 and later and
    /// `LuaJIT` check for any value first, and Lua 5.1 and 5.2 for a table.
    fn check_iteration_subject(&self, arguments: &[Value]) -> RuntimeResult<()> {
        if arguments.is_empty()
            && !matches!(
                self.compatibility,
                crate::compat::CompatibilityProfile::Lua51
                    | crate::compat::CompatibilityProfile::Lua52
            )
        {
            return Err(RuntimeError::argument_error(1, "value expected"));
        }
        Ok(())
    }

    pub(crate) fn call_file_builtin(
        &mut self,
        builtin: BuiltinFunction,
        arguments: &[Value],
    ) -> VmResult<Vec<Value>> {
        let Some(Value::File(file)) = arguments.first() else {
            return Err(RuntimeError::bad_argument(1, "FILE*", arguments.first()).into());
        };
        match builtin {
            BuiltinFunction::FileWrite => {
                // Like Lua, write the values before a bad one, then raise
                // its error.
                let bad = arguments.iter().skip(1).position(|value| {
                    !matches!(
                        value,
                        Value::String(_) | Value::Number(_) | Value::Integer(_)
                    )
                });
                let Some(offset) = bad else {
                    return self.file_write_results(*file, &arguments[1..]);
                };
                let written = self.file_write_results(*file, &arguments[1..=offset])?;
                if !matches!(written.first(), Some(Value::File(_))) {
                    return Ok(written);
                }
                Err(
                    RuntimeError::bad_argument(offset + 2, "string", arguments.get(offset + 1))
                        .into(),
                )
            }
            BuiltinFunction::FileSeek => {
                let default_whence = Value::from("cur");
                let default_offset = Value::Number(0.0);
                let whence = arguments.get(1).unwrap_or(&default_whence);
                let offset = arguments.get(2).unwrap_or(&default_offset);
                match self.runtime.file_seek(*file, whence, offset) {
                    Ok(position) => Ok(vec![self.runtime.integral_float_value(position)]),
                    Err(
                        error @ (RuntimeError::FileIo { .. } | RuntimeError::HostOutputIo { .. }),
                    ) => Ok(file_io_results(&error)),
                    // The whence is argument 2 and the offset argument 3.
                    Err(RuntimeError::TypeError {
                        expected, found, ..
                    }) => Err(RuntimeError::BadArgument {
                        position: if expected == "string" { 2 } else { 3 },
                        expected,
                        found,
                    }
                    .into()),
                    Err(error) => Err(error.into()),
                }
            }
            BuiltinFunction::FileRead => self
                .runtime
                .file_read_values(*file, &arguments[1..])
                .map_err(|error| error.shift_argument(1).into()),
            BuiltinFunction::FileClose => match self.runtime.file_close(*file) {
                Ok(()) => Ok(vec![Value::Boolean(true)]),
                Err(error @ (RuntimeError::FileIo { .. } | RuntimeError::HostOutputIo { .. })) => {
                    Ok(file_io_results(&error))
                }
                Err(error) => Err(error.into()),
            },
            BuiltinFunction::FileFlush => match self.runtime.file_flush(*file) {
                Ok(()) => Ok(vec![Value::Boolean(true)]),
                Err(error @ (RuntimeError::FileIo { .. } | RuntimeError::HostOutputIo { .. })) => {
                    Ok(file_io_results(&error))
                }
                Err(error) => Err(error.into()),
            },
            BuiltinFunction::FileSetVbuf => {
                let mode = arguments.get(1).unwrap_or(&Value::Nil);
                self.runtime.file_setvbuf(*file, mode, arguments.get(2))?;
                Ok(vec![Value::Boolean(true)])
            }
            BuiltinFunction::FileLines => Ok(vec![self.runtime.create_file_lines_iterator(
                *file,
                false,
                &arguments[1..],
                "lines",
                1,
            )?]),
            _ => Err(VmError::InvalidDispatch(OpCode::Call)),
        }
    }

    pub(crate) fn debug_traceback(
        &self,
        frames: &CallStack,
        arguments: &[Value],
    ) -> VmResult<Vec<Value>> {
        let (thread, argument_offset) = match arguments.first() {
            Some(Value::Thread(thread)) => (Some(*thread), 1),
            _ => (None, 0),
        };
        let message = arguments.get(argument_offset).unwrap_or(&Value::Nil);
        // Lua 5.1 and LuaJIT return a message that is not a string untouched,
        // `nil` included, unless there is no message argument at all.
        if self.compatibility.is_lua51_family()
            && arguments.len() > argument_offset
            && matches!(message, Value::Nil)
        {
            return Ok(vec![Value::Nil]);
        }
        let Some(message) = traceback_message(message, self.runtime.number_format()) else {
            return Ok(vec![message.clone()]);
        };
        let level_argument = arguments.get(argument_offset + 1);
        let requested_level = level_argument.and_then(traceback_level);
        let current = thread.is_none() || thread == self.current_coroutine;
        // A negative level names no stack level.
        let negative = level_argument
            .and_then(level_number)
            .is_some_and(|level| level < 0.0 && level.is_finite());
        // debug.traceback is a VM builtin and therefore absent from the
        // captured Lua frames, so level 1 (its caller) is the innermost
        // captured frame. Another thread's traceback counts from level 0.
        let level = match requested_level {
            Some(level) if current => level.saturating_sub(1),
            Some(level) => level,
            None => 0,
        };
        let mut trace = match thread {
            Some(thread) => self.traceback_for_thread(frames, thread)?,
            None => self.live_traceback(frames),
        };
        if negative {
            trace.frames.clear();
            // Lua 5.1 reads a negative level as that many levels of tail
            // calls above level zero.
            if self.compatibility == crate::compat::CompatibilityProfile::Lua51 {
                let tails = level_argument
                    .and_then(level_number)
                    .map_or(0, |level| (-level).min(1000.0) as usize);
                let below = if current {
                    self.live_traceback(frames)
                } else {
                    self.traceback_for_thread(frames, thread.expect("another thread"))?
                };
                trace.frames = (0..tails)
                    .map(|_| crate::vm::debug_info::tail_trace_frame())
                    .collect();
                trace.frames.extend(below.frames);
                if current {
                    trace.frames.insert(
                        tails,
                        native_trace_frame(
                            Some(Value::BuiltinFunction(BuiltinFunction::DebugTraceback)),
                            Some("debug.traceback"),
                            frames.last().and_then(|frame| active_call_site(&frame)),
                        ),
                    );
                }
            }
        }
        // The VM implements debug.traceback as a builtin, so it has no normal
        // Lua frame. Level zero is that C function.
        if current && requested_level == Some(0) {
            trace.frames.insert(
                0,
                native_trace_frame(
                    Some(Value::BuiltinFunction(BuiltinFunction::DebugTraceback)),
                    Some("debug.traceback"),
                    frames.last().and_then(|frame| active_call_site(&frame)),
                ),
            );
        }
        Ok(vec![Value::from(self.format_trace(
            message.as_deref(),
            &trace,
            level,
            usize::from(current),
        ))])
    }

    #[allow(clippy::unused_self)]
    pub(crate) fn traceback_handler_result(
        &self,
        arguments: &[Value],
        traceback: Option<&Traceback>,
    ) -> Value {
        let message = arguments.first().unwrap_or(&Value::Nil);
        let Some(message) = traceback_message(message, self.runtime.number_format()) else {
            return message.clone();
        };
        if self.compatibility.is_lua51_family()
            && arguments.first().is_some_and(|v| matches!(v, Value::Nil))
        {
            return Value::Nil;
        }
        let trace = traceback.cloned().unwrap_or(Traceback {
            frames: Vec::new(),
            debug_frames: Vec::new(),
        });
        Value::from(self.format_trace(message.as_deref(), &trace, 0, 1))
    }

    /// [`format_traceback_for`] with this VM's profile and the function names
    /// of `package.loaded`.
    pub(crate) fn format_trace(
        &self,
        message: Option<&str>,
        trace: &Traceback,
        level: usize,
        first_level: usize,
    ) -> String {
        format_traceback_for(
            message,
            trace,
            level,
            first_level,
            self.compatibility,
            &|function| loaded_function_name(&self.runtime, function),
        )
    }

    /// The levels below the running function of the current thread. Inside an
    /// error handler those are the handler's own frames, then the stack the
    /// error unwound.
    fn live_traceback(&self, frames: &CallStack) -> Traceback {
        let handler = frames.iter().enumerate().rev().find_map(|(index, frame)| {
            frame
                .extras_ref()
                .and_then(|extras| extras.traceback.clone())
                .map(|captured| (index, captured))
        });
        let mut trace;
        if let Some((index, captured)) = handler {
            // Frames above the handler's boundary stay live; the captured
            // trace stands for everything below it.
            trace = capture_traceback_above(frames, index);
            trace.frames.extend(captured.frames);
            trace.debug_frames.extend(captured.debug_frames);
        } else {
            trace = self.with_base_level(capture_traceback(frames));
        }
        trace
    }

    /// `trace` with the host's call into the main thread below it, as stock
    /// Lua lists it, unless the trace is a coroutine's.
    pub(crate) fn with_base_level(&self, mut trace: Traceback) -> Traceback {
        if self.current_coroutine.is_none() {
            trace.frames.push(base_trace_frame());
        }
        trace
    }

    pub(crate) fn traceback_for_thread(
        &self,
        active_frames: &CallStack,
        thread: ThreadHandle,
    ) -> VmResult<Traceback> {
        if self.current_coroutine == Some(thread) {
            return Ok(capture_traceback(active_frames));
        }
        let Some(coroutine) = self.coroutines.get(&thread) else {
            return Err(RuntimeError::message("invalid coroutine").into());
        };
        match &coroutine.state {
            CoroutineState::Suspended { frames, .. } => {
                let mut trace = capture_traceback(frames);
                trace.frames.insert(
                    0,
                    native_trace_frame(
                        None,
                        Some("coroutine.yield"),
                        Some(
                            frames
                                .last()
                                .and_then(|frame| active_call_site(&frame))
                                .unwrap_or(ActiveCallSite {
                                    name: LuaString::from("yield"),
                                    kind: CallKind::Field,
                                }),
                        ),
                    ),
                );
                Ok(trace)
            }
            CoroutineState::Dead => Ok(coroutine.traceback.clone().unwrap_or(Traceback {
                frames: Vec::new(),
                debug_frames: Vec::new(),
            })),
            CoroutineState::New(_) | CoroutineState::Running => Ok(Traceback {
                frames: Vec::new(),
                debug_frames: Vec::new(),
            }),
        }
    }

    pub(crate) fn debug_get_info(
        &mut self,
        frames: &CallStack,
        arguments: &[Value],
    ) -> VmResult<Vec<Value>> {
        if let Some(Value::Thread(thread)) = arguments.first() {
            let tail = &arguments[1..];
            if self.current_coroutine == Some(*thread) {
                return self.debug_get_info(frames, tail);
            }
            let Some(coroutine) = self.coroutines.get(thread) else {
                return Err(RuntimeError::message("invalid coroutine").into());
            };
            if tail.first().and_then(level_number).is_some() {
                let Some(level) = tail.first().and_then(value_debug_level) else {
                    return Ok(vec![Value::Nil]);
                };
                let info = match &coroutine.state {
                    CoroutineState::Suspended { frames, .. } => debug_info_at_level(frames, level),
                    CoroutineState::Dead if coroutine.failed.is_some() => coroutine
                        .traceback
                        .as_ref()
                        .and_then(|trace| trace.debug_frames.get(level).cloned()),
                    CoroutineState::New(_) | CoroutineState::Running | CoroutineState::Dead => None,
                };
                return info
                    .map_or_else(|| Ok(vec![Value::Nil]), |info| self.debug_table(info, tail));
            }
            // Function inspection does not depend on the target thread's
            // stack. Retain the normal argument diagnostics for this case.
            return self.debug_get_info(&CallStack::new(), tail);
        }
        let Some(subject) = arguments.first() else {
            return Err(self.getinfo_subject_error(None).into());
        };
        match subject {
            Value::Closure(handle) => {
                let chunk = self.closure_body(*handle)?.chunk();
                self.debug_table(
                    DebugInfo {
                        function: Some(subject.clone()),
                        source: chunk.debug.source_name().clone(),
                        current_line: None,
                        call_site: None,
                        what: "Lua",
                        definition_line: i32::try_from(chunk.debug.definition_line()).unwrap_or(-1),
                        last_definition_line: i32::try_from(chunk.debug.last_definition_line())
                            .unwrap_or(-1),
                        upvalue_count: chunk.upvalues.len(),
                        hook_transfer: (0, 0),
                        chunk: Some(chunk),
                    },
                    arguments,
                )
            }
            Value::NativeFunction(_) | Value::BuiltinFunction(_) => self.debug_table(
                DebugInfo {
                    function: Some(subject.clone()),
                    source: LuaString::from("=[C]"),
                    current_line: None,
                    call_site: None,
                    what: "C",
                    definition_line: -1,
                    last_definition_line: -1,
                    upvalue_count: 0,
                    hook_transfer: (0, 0),
                    chunk: None,
                },
                arguments,
            ),
            Value::Number(_) | Value::Integer(_) => {
                let Some(level) = level_number(subject).and_then(debug_level) else {
                    return Ok(vec![Value::Nil]);
                };
                let Some(info) = debug_info_at_level(frames, level) else {
                    return Ok(vec![Value::Nil]);
                };
                self.debug_table(info, arguments)
            }
            value => Err(self.getinfo_subject_error(Some(value)).into()),
        }
    }

    /// `debug.getinfo`'s error for a subject that is neither a function
    /// nor a level: Lua 5.1 and `LuaJIT` say "function or level expected",
    /// and Lua 5.2 and later read it as a level.
    fn getinfo_subject_error(&self, subject: Option<&Value>) -> RuntimeError {
        if self.compatibility.is_lua51_family() {
            RuntimeError::argument_error(1, "function or level expected")
        } else {
            RuntimeError::bad_argument(1, "number", subject)
        }
    }

    pub(crate) fn debug_table(
        &mut self,
        info: DebugInfo,
        arguments: &[Value],
    ) -> VmResult<Vec<Value>> {
        let options = debug_options(arguments)?;
        let table = self.runtime.create_table()?;
        if options.contains(&b'S') {
            self.runtime.table_set(
                table,
                &Value::from("source"),
                Value::String(info.source.clone()),
            )?;
            self.runtime.table_set(
                table,
                &Value::from("short_src"),
                Value::String(debug_short_source(&info.source)),
            )?;
            self.runtime
                .table_set(table, &Value::from("what"), Value::from(info.what))?;
            self.runtime.table_set(
                table,
                &Value::from("linedefined"),
                profile_integer_value(self.compatibility, i64::from(info.definition_line)),
            )?;
            self.runtime.table_set(
                table,
                &Value::from("lastlinedefined"),
                profile_integer_value(self.compatibility, i64::from(info.last_definition_line)),
            )?;
        }
        if options.contains(&b'l') {
            self.runtime.table_set(
                table,
                &Value::from("currentline"),
                profile_integer_value(self.compatibility, info.current_line.map_or(-1, i64::from)),
            )?;
        }
        if options.contains(&b'u') {
            self.runtime.table_set(
                table,
                &Value::from("nups"),
                profile_usize_value(self.compatibility, info.upvalue_count)?,
            )?;
            // Lua 5.2 and LuaJIT expose parameter shapes; C functions report `(...)`.
            if self.compatibility != crate::compat::CompatibilityProfile::Lua51 {
                let (parameters, vararg) = info
                    .chunk
                    .as_ref()
                    .map_or((0, true), |chunk| (chunk.parameter_count, chunk.is_vararg));
                self.runtime.table_set(
                    table,
                    &Value::from("nparams"),
                    profile_integer_value(self.compatibility, i64::from(parameters)),
                )?;
                self.runtime
                    .table_set(table, &Value::from("isvararg"), Value::Boolean(vararg))?;
            }
        }
        if options.contains(&b'r')
            && matches!(
                self.compatibility,
                crate::compat::CompatibilityProfile::Lua54
                    | crate::compat::CompatibilityProfile::Lua55
            )
        {
            self.runtime.table_set(
                table,
                &Value::from("ftransfer"),
                profile_usize_value(self.compatibility, info.hook_transfer.0)?,
            )?;
            self.runtime.table_set(
                table,
                &Value::from("ntransfer"),
                profile_usize_value(self.compatibility, info.hook_transfer.1)?,
            )?;
        }
        if options.contains(&b'L')
            && let Some(chunk) = &info.chunk
        {
            let active_lines_table = self.runtime.create_table()?;
            for line in chunk.debug.active_lines() {
                self.runtime.table_set(
                    active_lines_table,
                    &Value::Number(f64::from(line)),
                    Value::Boolean(true),
                )?;
            }
            self.runtime.table_set(
                table,
                &Value::from("activelines"),
                Value::Table(active_lines_table),
            )?;
        }
        if options.contains(&b'f')
            && let Some(function) = info.function
        {
            self.runtime
                .table_set(table, &Value::from("func"), function)?;
        }
        if options.contains(&b'n') {
            if let Some(call_site) = info.call_site {
                self.runtime.table_set(
                    table,
                    &Value::from("name"),
                    Value::from(call_site.display_name(self.compatibility)),
                )?;
                self.runtime.table_set(
                    table,
                    &Value::from("namewhat"),
                    Value::from(debug_namewhat(call_site.kind)),
                )?;
            } else {
                // lua_getinfo leaves the name unset when the function has no
                // call-site identity, but still reports an empty namewhat.
                self.runtime
                    .table_set(table, &Value::from("namewhat"), Value::from(""))?;
            }
        }
        Ok(vec![Value::Table(table)])
    }

    pub(crate) fn get_fenv(&self, frames: &CallStack, arguments: &[Value]) -> VmResult<Vec<Value>> {
        let subject = match arguments.first() {
            None | Some(Value::Nil) => Value::Number(1.0),
            Some(value) => level_number(value).map_or_else(|| value.clone(), Value::Number),
        };
        let environment = match subject {
            Value::Closure(handle) => self.runtime.closure_environment(handle)?,
            Value::Thread(handle) => self
                .coroutines
                .get(&handle)
                .map(|coroutine| coroutine.environment)
                .ok_or_else(|| RuntimeError::message("invalid coroutine"))?,
            Value::Number(level) => frame_environment(frames, level, self.thread_environment)?,
            value => return Err(RuntimeError::bad_argument(1, "number", Some(&value)).into()),
        };
        Ok(vec![Value::Table(environment)])
    }

    pub(crate) fn set_fenv(
        &mut self,
        frames: &mut CallStack,
        arguments: &[Value],
    ) -> VmResult<Vec<Value>> {
        let subject = arguments.first().map_or(Value::Nil, |value| {
            level_number(value).map_or_else(|| value.clone(), Value::Number)
        });
        let Some(Value::Table(environment)) = arguments.get(1) else {
            return Err(RuntimeError::bad_argument(2, "table", arguments.get(1)).into());
        };
        let result = match subject {
            Value::Closure(handle) => {
                self.runtime.set_closure_environment(handle, *environment)?;
                for frame in frames
                    .iter_mut()
                    .filter(|frame| frame.closure == Some(handle))
                {
                    frame.environment = *environment;
                }
                Value::Closure(handle)
            }
            Value::Thread(handle) => {
                let coroutine = self
                    .coroutines
                    .get_mut(&handle)
                    .ok_or_else(|| RuntimeError::message("invalid coroutine"))?;
                coroutine.environment = *environment;
                if self.current_coroutine == Some(handle) {
                    self.thread_environment = *environment;
                }
                Value::Thread(handle)
            }
            Value::Number(level) => {
                if fenv_level(level) == Some(0) {
                    self.thread_environment = *environment;
                    if let Some(thread) = self.current_coroutine
                        && let Some(coroutine) = self.coroutines.get_mut(&thread)
                    {
                        coroutine.environment = *environment;
                    }
                    Value::Number(level)
                } else if let Some(closure) = set_frame_environment(frames, level, *environment)? {
                    self.runtime
                        .set_closure_environment(closure, *environment)?;
                    Value::Closure(closure)
                } else {
                    Value::Number(level)
                }
            }
            value => return Err(RuntimeError::bad_argument(1, "number", Some(&value)).into()),
        };
        Ok(vec![result])
    }

    pub(crate) fn debug_setupvalue(
        &mut self,
        _frames: &mut CallStack,
        arguments: &[Value],
    ) -> VmResult<Vec<Value>> {
        upvalue_arguments(arguments)?;
        if arguments.len() < 3 {
            return Err(RuntimeError::argument_error(3, "value expected").into());
        }
        let Some(Value::Closure(handle)) = arguments.first() else {
            return Ok(Vec::new());
        };
        let Some(index) = arguments.get(1).and_then(value_index) else {
            return Ok(Vec::new());
        };
        let body = self.closure_body(*handle)?;
        let chunk = body.chunk();
        let Some(descriptor) = chunk.upvalues.get(index) else {
            return Ok(Vec::new());
        };
        let Some(cell) = body.cells().get(index) else {
            return Ok(Vec::new());
        };
        let cell = *cell;
        let value = arguments.get(2).cloned().unwrap_or(Value::Nil);
        // An open cell writes the register it stands for.
        self.runtime.cell_set(cell, value)?;
        Ok(vec![Value::String(descriptor.name.clone())])
    }

    pub(crate) fn debug_getupvalue(&self, arguments: &[Value]) -> VmResult<Vec<Value>> {
        upvalue_arguments(arguments)?;
        let Some(Value::Closure(handle)) = arguments.first() else {
            return Ok(Vec::new());
        };
        let Some(index) = arguments.get(1).and_then(value_index) else {
            return Ok(Vec::new());
        };
        let body = self.closure_body(*handle)?;
        let chunk = body.chunk();
        let Some(descriptor) = chunk.upvalues.get(index) else {
            return Ok(Vec::new());
        };
        let Some(cell) = body.cells().get(index) else {
            return Ok(Vec::new());
        };
        Ok(vec![
            Value::String(descriptor.name.clone()),
            self.runtime.cell_get(*cell)?,
        ])
    }

    pub(crate) fn debug_upvalueid(&mut self, arguments: &[Value]) -> VmResult<Vec<Value>> {
        let closure = debug_closure_argument(arguments.first(), 1)?;
        let index = debug_upvalue_index(arguments.get(1), 2)?;
        let body = self.closure_body(closure)?;
        let Some(cell) = body.cells().get(index).copied() else {
            return Err(RuntimeError::message("invalid upvalue index").into());
        };
        Ok(vec![self.runtime.cell_identity(cell)?])
    }

    pub(crate) fn debug_upvaluejoin(
        &mut self,
        frames: &mut CallStack,
        arguments: &[Value],
    ) -> VmResult<Vec<Value>> {
        let destination = debug_closure_argument(arguments.first(), 1)?;
        let destination_index = debug_upvalue_index(arguments.get(1), 2)?;
        let source = debug_closure_argument(arguments.get(2), 3)?;
        let source_index = debug_upvalue_index(arguments.get(3), 4)?;

        let source_body = self.closure_body(source)?;
        let Some(source_cell) = source_body.cells().get(source_index).copied() else {
            return Err(RuntimeError::message("invalid upvalue index").into());
        };
        let destination_body = self.closure_body(destination)?;
        if destination_body.cells().get(destination_index).is_none() {
            return Err(RuntimeError::message("invalid upvalue index").into());
        }
        self.runtime
            .set_closure_cell(destination, destination_index, source_cell)?;
        // Frames parked on the Rust stack may borrow the cells just replaced;
        // `set_closure_cell` keeps the old slice alive with the closure.

        replace_active_closure_cell(frames, destination, destination_index, source_cell);
        for stack in &mut self.resumer_stacks {
            replace_active_closure_cell(
                stack.get_mut(),
                destination,
                destination_index,
                source_cell,
            );
        }
        for coroutine in self.coroutines.values_mut() {
            if let CoroutineState::Suspended { frames, .. } = &mut coroutine.state {
                replace_active_closure_cell(frames, destination, destination_index, source_cell);
            }
            if let Some((frames, _)) = &mut coroutine.failed {
                replace_active_closure_cell(frames, destination, destination_index, source_cell);
            }
        }
        Ok(Vec::new())
    }

    pub(crate) fn debug_getregistry(&self) -> Vec<Value> {
        vec![Value::Table(self.runtime.registry())]
    }

    pub(crate) fn debug_getuservalue(&self, arguments: &[Value]) -> VmResult<Vec<Value>> {
        let indexed = self.compatibility.capabilities().debug.multiple_uservalues;
        let index = if indexed {
            debug_uservalue_index(arguments, 2)?
        } else {
            1
        };
        let value = match arguments.first() {
            Some(Value::Userdata(handle)) if !self.runtime.userdata_is_light(*handle)? => {
                self.runtime.userdata_uservalue(*handle, index)?
            }
            Some(Value::File(handle)) if !indexed => Some(self.runtime.file_uservalue(*handle)?),
            _ => None,
        };
        Ok(match (indexed, value) {
            (true, Some(value)) => vec![value, Value::Boolean(true)],
            (_, Some(value)) => vec![value],
            (_, None) => vec![Value::Nil],
        })
    }

    pub(crate) fn debug_setuservalue(&mut self, arguments: &[Value]) -> VmResult<Vec<Value>> {
        let indexed = self.compatibility.capabilities().debug.multiple_uservalues;
        let index = if indexed {
            debug_uservalue_index(arguments, 3)?
        } else {
            1
        };
        if !matches!(arguments.first(), Some(Value::Userdata(_) | Value::File(_))) {
            return Err(RuntimeError::bad_argument(1, "userdata", arguments.first()).into());
        }
        if let Some(Value::Userdata(handle)) = arguments.first()
            && self.runtime.userdata_is_light(*handle)?
        {
            return Err(RuntimeError::argument_error(
                1,
                "full userdata expected, got light userdata",
            )
            .into());
        }
        let value = match arguments.get(1) {
            Some(value) => value.clone(),
            None if self.compatibility == crate::compat::CompatibilityProfile::Lua52 => Value::Nil,
            None => return Err(RuntimeError::argument_error(2, "value expected").into()),
        };
        if matches!(
            self.compatibility,
            crate::compat::CompatibilityProfile::Lua52
        ) && !matches!(value, Value::Nil | Value::Table(_))
        {
            return Err(RuntimeError::bad_argument(2, "table", Some(&value)).into());
        }
        match arguments.first() {
            Some(Value::Userdata(handle)) => {
                if self.runtime.set_userdata_uservalue(*handle, index, value)? {
                    Ok(vec![Value::Userdata(*handle)])
                } else {
                    Ok(vec![Value::Nil])
                }
            }
            Some(Value::File(handle)) if !indexed => {
                self.runtime.set_file_uservalue(*handle, value)?;
                Ok(vec![Value::File(*handle)])
            }
            Some(Value::File(_)) => Ok(vec![Value::Nil]),
            _ => unreachable!(),
        }
    }

    /// Lua's small line-at-a-time debugger. Each command is an independent
    /// protected chunk in the current global environment.
    pub(crate) fn debug_console(&mut self, frames: &CallStack) -> VmResult<Vec<Value>> {
        let mut roots = GcRootBuilder::new();
        Self::root_call_stack(&mut roots, frames, Window::Full);
        self.suspended_roots.push(roots.finish());
        let result = self.debug_console_lines();
        self.suspended_roots.pop();
        result
    }

    fn debug_console_lines(&mut self) -> VmResult<Vec<Value>> {
        let mut input = std::io::stdin();
        let mut error = std::io::stderr();
        loop {
            error
                .write_all(b"lua_debug> ")
                .and_then(|()| error.flush())
                .map_err(|cause| RuntimeError::message(cause.to_string()))?;
            let mut line = Vec::with_capacity(249);
            while line.len() < 249 {
                let mut byte = [0];
                if input
                    .read(&mut byte)
                    .map_err(|cause| RuntimeError::message(cause.to_string()))?
                    == 0
                {
                    if line.is_empty() {
                        return Ok(Vec::new());
                    }
                    break;
                }
                line.push(byte[0]);
                if byte[0] == b'\n' {
                    break;
                }
            }
            if line == b"cont\n" {
                return Ok(Vec::new());
            }
            let source = LuaString::from(line);
            let loaded = self.load_string(&source, &LuaString::from("=(debug command)"));
            let failure = match loaded.as_slice() {
                [Value::Closure(closure)] => {
                    match self.call_value(Value::Closure(*closure), Vec::new()) {
                        Ok(_) => None,
                        Err(error @ VmError::Runtime(RuntimeError::Exit(_))) => return Err(error),
                        Err(error) => Some(error.to_string()),
                    }
                }
                [Value::Nil, message, ..] => Some(match message {
                    Value::String(text) => text.to_lossy().into_owned(),
                    other => format!("{other:?}"),
                }),
                _ => None,
            };
            if let Some(message) = failure {
                error
                    .write_all(message.as_bytes())
                    .and_then(|()| error.write_all(b"\n"))
                    .map_err(|cause| RuntimeError::message(cause.to_string()))?;
            }
        }
    }

    pub(crate) fn debug_sethook(&mut self, arguments: &[Value]) -> VmResult<Vec<Value>> {
        match self.parse_debug_sethook(arguments)? {
            (thread, None) => {
                self.debug_hooks.remove(&thread);
            }
            (thread, Some(hook)) => {
                self.debug_hooks.insert(thread, hook);
            }
        }
        self.debug_hooks_changed();
        Ok(Vec::new())
    }

    /// Validates `debug.sethook` arguments without installing the hook. A
    /// `None` hook removes the thread's hook.
    pub(crate) fn parse_debug_sethook(
        &self,
        arguments: &[Value],
    ) -> VmResult<(Option<ThreadHandle>, Option<DebugHook>)> {
        let (thread, offset) = match arguments.first() {
            Some(Value::Thread(thread)) => (Some(*thread), 1),
            _ => (self.current_coroutine, 0),
        };
        if let Some(thread) = thread
            && !self.coroutines.contains_key(&thread)
        {
            return Err(RuntimeError::message("invalid coroutine").into());
        }
        let hook = arguments.get(offset).cloned().unwrap_or(Value::Nil);
        if matches!(hook, Value::Nil) {
            return Ok((thread, None));
        }
        // Like `db_sethook`, check the mask before the hook's type.
        let mask = match arguments.get(offset + 1) {
            Some(Value::String(mask)) => DebugHookMask {
                call: mask.as_bytes().contains(&b'c'),
                return_: mask.as_bytes().contains(&b'r'),
                line: mask.as_bytes().contains(&b'l'),
            },
            argument => {
                return Err(RuntimeError::bad_argument(offset + 2, "string", argument).into());
            }
        };
        if !matches!(
            hook,
            Value::Closure(_) | Value::NativeFunction(_) | Value::BuiltinFunction(_)
        ) {
            return Err(RuntimeError::bad_argument(offset + 1, "function", Some(&hook)).into());
        }
        let count = arguments
            .get(offset + 2)
            .map_or(Ok(0), |count| hook_count(count, offset + 3))?;
        Ok((
            thread,
            Some(DebugHook {
                function: hook,
                mask,
                count,
                remaining: count,
                running: false,
            }),
        ))
    }

    pub(crate) fn debug_gethook(&self, arguments: &[Value]) -> Vec<Value> {
        // Like lua_gethook's getthread, a non-thread argument selects the
        // running thread.
        let thread = match arguments.first() {
            Some(Value::Thread(thread)) => Some(*thread),
            _ => self.current_coroutine,
        };
        let Some(hook) = self.debug_hooks.get(&thread) else {
            return vec![Value::Nil];
        };
        let mut mask = String::new();
        if hook.mask.call {
            mask.push('c');
        }
        if hook.mask.return_ {
            mask.push('r');
        }
        if hook.mask.line {
            mask.push('l');
        }
        vec![
            hook.function.clone(),
            Value::from(mask),
            profile_integer_value(self.compatibility, i64::from(hook.count)),
        ]
    }

    /// `debug.getlocal`'s argument type checks: Lua 5.1 checks the level
    /// before the local's index, `LuaJIT` and Lua 5.2 and later the index
    /// first.
    pub(crate) fn getlocal_argument_error(&self, arguments: &[Value]) -> Option<RuntimeError> {
        let offset = usize::from(matches!(arguments.first(), Some(Value::Thread(_))));
        let level = arguments.get(offset);
        let index = arguments.get(offset + 1);
        let level_error = match level {
            Some(Value::Integer(_) | Value::Number(_)) => None,
            Some(Value::Closure(_) | Value::NativeFunction(_) | Value::BuiltinFunction(_))
                if !self.compatibility.is_lua51_family() =>
            {
                None
            }
            level => Some(RuntimeError::bad_argument(offset + 1, "number", level)),
        };
        let index_error = match index {
            Some(Value::Integer(_) | Value::Number(_)) => None,
            index => Some(RuntimeError::bad_argument(offset + 2, "number", index)),
        };
        if self.compatibility == crate::compat::CompatibilityProfile::Lua51 {
            level_error.or(index_error)
        } else {
            index_error.or(level_error)
        }
    }

    pub(crate) fn debug_getlocal(&self, frames: &CallStack, arguments: &[Value]) -> Vec<Value> {
        let (thread, offset) = match arguments.first() {
            Some(Value::Thread(thread)) => (Some(*thread), 1),
            _ => (None, 0),
        };
        let Some(index) = arguments
            .get(offset + 1)
            .and_then(crate::vm::host_locals::local_index)
        else {
            return Vec::new();
        };
        if !self.compatibility.is_lua51_family()
            && let Some(
                function @ (Value::Closure(_)
                | Value::NativeFunction(_)
                | Value::BuiltinFunction(_)),
            ) = arguments.get(offset)
        {
            return self
                .function_local_name(function.clone(), index)
                .ok()
                .flatten()
                .map_or_else(Vec::new, |name| vec![Value::String(name)]);
        }
        let Some(level) = arguments.get(offset).and_then(value_debug_level) else {
            return Vec::new();
        };
        if thread.is_none() && level == 0 {
            return usize::try_from(index)
                .ok()
                .and_then(|index| index.checked_sub(1))
                .and_then(|index| arguments.get(index).cloned())
                .map_or_else(Vec::new, |value| vec![Value::from("(*temporary)"), value]);
        }
        let local =
            if let Some(thread) = thread.filter(|thread| Some(*thread) != self.current_coroutine) {
                self.coroutine_get_local(thread, level, index)
                    .ok()
                    .flatten()
            } else {
                level
                    .checked_sub(1)
                    .and_then(|level| frames.nth_from_top(level))
                    .and_then(|frame| self.get_frame_local(&frame, index).ok().flatten())
            };
        local.map_or_else(Vec::new, |(name, value)| vec![Value::String(name), value])
    }

    pub(crate) fn debug_setlocal(
        &mut self,
        frames: &mut CallStack,
        arguments: &[Value],
    ) -> VmResult<Vec<Value>> {
        self.local_writes += 1;
        let (thread, offset) = match arguments.first() {
            Some(Value::Thread(thread)) => (Some(*thread), 1),
            _ => (None, 0),
        };
        let Some(level) = arguments.get(offset).and_then(value_debug_level) else {
            return Ok(Vec::new());
        };
        let Some(index) = arguments
            .get(offset + 1)
            .and_then(crate::vm::host_locals::local_index)
        else {
            return Ok(Vec::new());
        };
        let value = arguments.get(offset + 2).cloned().unwrap_or(Value::Nil);
        let name =
            if let Some(thread) = thread.filter(|thread| Some(*thread) != self.current_coroutine) {
                self.coroutine_set_local(thread, level, index, value)?
            } else {
                let Some(level) = level.checked_sub(1) else {
                    return Ok(Vec::new());
                };
                let Some(mut frame) = frames.nth_from_top_mut(level) else {
                    return Ok(Vec::new());
                };
                self.set_frame_local(&mut frame, index, value)?
            };
        Ok(name.map_or_else(Vec::new, |name| vec![Value::String(name)]))
    }

    /// Starts a hook callback, if this thread has opted into `event`. The
    /// callback is an ordinary Lua frame, which keeps `debug.getinfo(2)` and
    /// local inspection pointed at the code that raised the event.
    /// Runs the return-event hook shared by the return opcodes.
    ///
    /// Returns true when dispatch must pause and replay the return: the hook
    /// ran and asked to observe the frame again. The suppress flag and its
    /// reset live here so the four return arms stay one call each.
    pub(crate) fn return_hook_pause(&mut self, frames: &mut CallStack) -> VmResult<bool> {
        // The interrupted frame is captured up front: firing the hook pushes
        // the callback's own frame, so `frames.last()` no longer names the
        // returning activation afterwards.
        let frame_index = frames.len() - 1;
        let suppress = frames[frame_index]
            .extras_ref()
            .is_some_and(|extras| extras.suppress_return_hook);
        if suppress {
            if let Some(extras) = frames[frame_index].extras_mut_ref() {
                extras.suppress_return_hook = false;
            }
            return Ok(false);
        }
        // Lua 5.1 observes each elided tail caller before this activation
        // disappears. Keeping the frame live also gives getinfo(2) the
        // returning function during every tail-return callback.
        let tail_returns = if self.compatibility == crate::compat::CompatibilityProfile::Lua51 {
            frames[frame_index].tail_calls as usize
        } else {
            0
        };
        let depth = frames.len();
        if self.fire_debug_hook_with_tail_returns(frames, DebugHookEvent::Return, tail_returns)? {
            // Lua callbacks continue through DebugHookReturn. Native hooks
            // finish synchronously and need their remaining events here.
            if frames.len() == depth {
                for _ in 0..tail_returns {
                    self.fire_debug_hook(frames, DebugHookEvent::TailReturn)?;
                }
            }
            let frame = &mut frames[frame_index];
            frame.extras().suppress_return_hook = true;
            // The return instruction was already consumed. Run it
            // again once the callback has completed.
            frame.pc -= 1;
            return Ok(true);
        }
        Ok(false)
    }

    /// Observes a native call with the call hook, tagging the frame with the
    /// subject the hook should inspect. Returns false when no hook ran.
    pub(crate) fn observe_native_call(
        &mut self,
        frames: &mut CallStack,
        subject: Value,
        argument_count: usize,
    ) -> VmResult<bool> {
        // Most native calls have no call hook. In particular, do not create
        // FrameExtras merely to set and immediately clear the subject.
        if self.debug_hooks.is_empty()
            || self.debug_hook_depth != 0
            || !self
                .debug_hooks
                .get(&self.current_coroutine)
                .is_some_and(|hook| hook.mask.call && !hook.running)
        {
            return Ok(false);
        }
        let mut frame = frames
            .last_mut()
            .expect("a native call has an active frame");
        let extras = frame.extras();
        extras.debug_hook_subject = Some(subject);
        extras.debug_hook_transfer = (1, argument_count);
        if self.fire_debug_hook(frames, DebugHookEvent::Call)? {
            return Ok(true);
        }
        frames
            .last_mut()
            .expect("a native call has an active frame")
            .extras()
            .debug_hook_subject = None;
        if let Some(mut frame) = frames.last_mut() {
            frame.extras().debug_hook_transfer = (0, 0);
        }
        Ok(false)
    }

    pub(crate) fn fire_debug_hook(
        &mut self,
        frames: &mut CallStack,
        event: DebugHookEvent,
    ) -> VmResult<bool> {
        self.fire_debug_hook_with_tail_returns(frames, event, 0)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "Hook dispatch preserves callback and pending tail-return state together"
    )]
    pub(crate) fn fire_debug_hook_with_tail_returns(
        &mut self,
        frames: &mut CallStack,
        event: DebugHookEvent,
        remaining_tail_returns: usize,
    ) -> VmResult<bool> {
        if self.debug_hooks.is_empty() || self.debug_hook_depth != 0 {
            return Ok(false);
        }
        let thread = self.current_coroutine;
        // Most events reach here with their mask bit clear, as calls and
        // returns do under a count hook. Rule them out before the walk over
        // every frame below.
        if !self.debug_hooks.get(&thread).is_some_and(|hook| {
            !hook.running
                && match event {
                    DebugHookEvent::Call | DebugHookEvent::TailCall => hook.mask.call,
                    DebugHookEvent::Return | DebugHookEvent::TailReturn => hook.mask.return_,
                    DebugHookEvent::Line(_) => hook.mask.line,
                    DebugHookEvent::Count => hook.count != 0,
                }
        }) {
            return Ok(false);
        }
        if frames.iter().any(|frame| {
            frame
                .extras_ref()
                .is_some_and(|extras| extras.suppress_debug_hooks)
        }) {
            return Ok(false);
        }
        let function = {
            let Some(hook) = self.debug_hooks.get_mut(&thread) else {
                return Ok(false);
            };
            if hook.running {
                return Ok(false);
            }
            let enabled = match event {
                DebugHookEvent::Call | DebugHookEvent::TailCall => hook.mask.call,
                DebugHookEvent::Return | DebugHookEvent::TailReturn => hook.mask.return_,
                DebugHookEvent::Line(_) => hook.mask.line,
                DebugHookEvent::Count => hook.count != 0,
            };
            if !enabled {
                return Ok(false);
            }
            hook.running = true;
            hook.function.clone()
        };
        if matches!(
            self.compatibility,
            crate::compat::CompatibilityProfile::Lua54 | crate::compat::CompatibilityProfile::Lua55
        ) {
            let transfer = self.hook_transfer(frames, event);
            if let Some(mut frame) = frames.last_mut() {
                frame.extras().debug_hook_transfer = transfer;
            }
        } else if let Some(mut frame) = frames.last_mut() {
            frame.extras().debug_hook_transfer = (0, 0);
        }
        // Every Lua release calls the hook with the event and the current
        // line, which is nil for non-line events.
        let arguments = vec![
            self.hook_event_names[event.index()]
                .get_or_insert_with(|| Value::from(event.name()))
                .clone(),
            event.line().map_or(Value::Nil, |line| {
                profile_integer_value(self.compatibility, i64::from(line))
            }),
        ];
        self.debug_hook_depth += 1;
        // A Lua hook, or a builtin that needs the frame stack, runs in a new
        // frame that returns like a Lua hook. Other hooks finish here.
        let (chunk, arguments, cells, environment, closure) = match function {
            Value::Closure(closure) => {
                let body = self.closure_body(closure)?;
                let environment = body.environment();
                (
                    body.chunk(),
                    arguments,
                    body.shared_cells(),
                    environment,
                    Some(closure),
                )
            }
            Value::BuiltinFunction(builtin) => match self.call_builtin(builtin, &arguments) {
                Err(VmError::InvalidDispatch(_)) => (
                    Arc::new(builtin_call_chunk(arguments.len())?),
                    std::iter::once(Value::BuiltinFunction(builtin))
                        .chain(arguments)
                        .collect(),
                    CaptureCells::default(),
                    self.thread_environment,
                    None,
                ),
                result => {
                    self.finish_debug_hook(thread);
                    if let Some(mut caller) = frames.last_mut() {
                        caller.extras().debug_hook_transfer = (0, 0);
                    }
                    result.map_err(|error| VmError::HookError(Box::new(error)))?;
                    return Ok(true);
                }
            },
            Value::NativeFunction(function) => {
                let result = self
                    .call_native(frames, function, &arguments)
                    .map_err(VmError::from);
                self.finish_debug_hook(thread);
                if let Some(mut caller) = frames.last_mut() {
                    caller.extras().debug_hook_subject = None;
                    caller.extras().debug_hook_transfer = (0, 0);
                }
                result.map_err(|error| VmError::HookError(Box::new(error)))?;
                return Ok(true);
            }
            _ => unreachable!("sethook validates callable values"),
        };
        if frames.len() >= MAX_LUA_CALL_FRAMES {
            self.finish_debug_hook(thread);
            if let Some(mut caller) = frames.last_mut() {
                caller.extras().debug_hook_transfer = (0, 0);
            }
            return Err(RuntimeError::message("stack overflow").into());
        }
        let frame = self.make_frame(
            chunk,
            arguments,
            cells,
            Some(ReturnTarget::DebugHook(Box::new(DebugHookReturn {
                thread,
                remaining_tail_returns,
            }))),
            None,
            environment,
            closure,
        )?;
        frames.push(frame);
        Ok(true)
    }

    fn hook_transfer(&self, frames: &CallStack, event: DebugHookEvent) -> (usize, usize) {
        let Some(frame) = frames.last() else {
            return (0, 0);
        };
        let transfer = match event {
            DebugHookEvent::Call | DebugHookEvent::TailCall => {
                if let Some(extras) = frame
                    .extras_ref()
                    .filter(|extras| extras.debug_hook_subject.is_some())
                {
                    extras.debug_hook_transfer
                } else {
                    (1, usize::from(frame.chunk.parameter_count))
                }
            }
            DebugHookEvent::Return => {
                let Some(instruction) = frame
                    .pc
                    .checked_sub(1)
                    .and_then(|pc| frame.chunk.code.get(pc))
                else {
                    return (0, 0);
                };
                let start = usize::from(instruction.a());
                let fixed = usize::from(instruction.b());
                let count = match instruction.opcode() {
                    Ok(OpCode::Return) => {
                        if fixed == 0 {
                            frame.registers.len().saturating_sub(start)
                        } else {
                            fixed
                        }
                    }
                    Ok(OpCode::ReturnVararg) => {
                        fixed
                            + self
                                .frame_varargs(frames, frames.len() - 1)
                                .map_or(0, |values| values.len())
                    }
                    Ok(OpCode::ReturnOpen) => fixed + frame.open_values().map_or(0, <[Value]>::len),
                    Ok(OpCode::Halt) => 0,
                    _ => return (0, 0),
                };
                (start + 1, count)
            }
            DebugHookEvent::Line(_) | DebugHookEvent::Count | DebugHookEvent::TailReturn => (0, 0),
        };
        // Lua 5.4 suppresses transfer information when nothing was transferred.
        // Lua 5.5 also reports the first position for an empty call or return.
        if self.compatibility == crate::compat::CompatibilityProfile::Lua54 && transfer.1 == 0 {
            (0, 0)
        } else {
            transfer
        }
    }

    /// Called after any hook is installed or removed. A counting fast loop
    /// stops at its next counted instruction, and its count is not written
    /// back over the new hook's.
    pub(crate) fn debug_hooks_changed(&mut self) {
        if let Some(lease) = &mut self.count_lease {
            lease.invalidated = true;
            self.count_budget = 0;
        }
        self.hooks_observe_calls = self
            .debug_hooks
            .values()
            .any(|hook| hook.mask.call || hook.mask.return_ || hook.mask.line);
        self.debug_hooks_installed |= !self.debug_hooks.is_empty();
        // A hook where there was none leaves the traces that exist without
        // count accounting: they cannot run under it. Flush them, so that hot
        // loops record again with accounting.
        if self.jit.trace.note_hooks(!self.debug_hooks.is_empty()) {
            self.runtime.request_jit_flush();
        }
        self.refresh_return_watch();
    }

    /// What [`Self::plain_return_watch`] has to say, from the three things
    /// it stands for.
    #[inline]
    pub(crate) fn return_watch_due(&self) -> bool {
        self.pending_gc_recheck_depth.is_some()
            || !self.debug_hooks.is_empty()
            || self.count_lease.is_some()
    }

    /// Called after a write of the pending collection recheck, the hooks or
    /// the count lease.
    #[inline]
    pub(crate) fn refresh_return_watch(&mut self) {
        self.plain_return_watch = self.return_watch_due();
    }

    /// Whether every installed hook is a count hook that is not running: the
    /// only hooks under which traces may run, because a trace can account
    /// for instructions in bulk (see `Recorder::with_counting`) but has no
    /// way to fire call, return or line events. True when no hook is
    /// installed.
    pub(crate) fn count_only_hooks(&self) -> bool {
        self.debug_hook_depth == 0
            && self.debug_hooks.values().all(|hook| {
                !hook.mask.call
                    && !hook.mask.return_
                    && !hook.mask.line
                    && hook.count > 0
                    && !hook.running
            })
    }

    /// Whether native code may run on `frames` under the installed hooks: no
    /// hook, or count hooks only with the count applying to every frame (a
    /// frame that suppresses hooks makes the interpreter skip the count, which
    /// a trace could not).
    pub(crate) fn hooks_allow_traces(&self, frames: &CallStack) -> bool {
        self.debug_hooks.is_empty()
            || (self.count_only_hooks()
                && !frames.iter().any(|frame| {
                    frame
                        .extras_ref()
                        .is_some_and(|extras| extras.suppress_debug_hooks)
                }))
    }

    /// The running thread's count hook's live count, which the next
    /// instruction to run decrements, or `None` without a hook.
    pub(crate) fn live_count_budget(&self) -> Option<i64> {
        let hook = self.debug_hooks.get(&self.current_coroutine)?;
        Some(match &self.count_lease {
            Some(lease) if lease.thread == self.current_coroutine => {
                if lease.invalidated {
                    0
                } else {
                    i64::from(self.count_budget)
                }
            }
            _ => i64::from(hook.remaining),
        })
    }

    /// Stores the count [`Self::live_count_budget`] reads, after native code
    /// ran counted instructions.
    pub(crate) fn set_live_count_budget(&mut self, budget: i64) {
        let thread = self.current_coroutine;
        let budget = u32::try_from(budget.max(0)).unwrap_or(u32::MAX);
        match &self.count_lease {
            Some(lease) if lease.thread == thread => {
                if !lease.invalidated {
                    self.count_budget = budget;
                }
            }
            _ => {
                if let Some(hook) = self.debug_hooks.get_mut(&thread) {
                    hook.remaining = budget;
                }
            }
        }
    }

    /// How the fast loop may run under the current thread's hook: as usual
    /// when no hook event can fire, hooked when count, call, or return
    /// events can, and not at all when line events can.
    pub(crate) fn fast_hook_mode(&self, frames: &CallStack) -> FastHookMode {
        let Some(hook) = self.debug_hooks.get(&self.current_coroutine) else {
            return FastHookMode::Plain;
        };
        // Every event is ignored while a hook runs.
        if hook.running || self.debug_hook_depth != 0 {
            return FastHookMode::Plain;
        }
        if hook.mask.line {
            return FastHookMode::Off;
        }
        let count = hook.count != 0;
        let calls = hook.mask.call || hook.mask.return_;
        if !(count || calls)
            || frames.iter().any(|frame| {
                frame
                    .extras_ref()
                    .is_some_and(|extras| extras.suppress_debug_hooks)
            })
        {
            return FastHookMode::Plain;
        }
        FastHookMode::Hooked { count, calls }
    }

    /// Runs [`Self::interpret_fast`] under a hook, as [`FastHookMode::Hooked`]
    /// describes. With `calls`, every call and return goes to the dispatch
    /// loop, which fires their events, and so does every metamethod.
    ///
    /// With `count`, the current thread's count is lent to the fast loop,
    /// which counts instructions as the dispatch loop's prologue would.
    /// `prologue_counted` says whether that prologue already counted the
    /// instruction at `pc`, which the loop counts again when it fetches it.
    /// Returns the exit and whether the instruction the exit leaves at the
    /// top frame's `pc` was counted, so the dispatch loop may run it
    /// without the prologue.
    pub(crate) fn interpret_fast_hooked(
        &mut self,
        stack: &mut CallStack,
        count: bool,
        calls: bool,
        prologue_counted: bool,
    ) -> VmResult<(FastExit, bool)> {
        let saved = (self.fast_hook_count, self.fast_hook_calls);
        (self.fast_hook_count, self.fast_hook_calls) = (count, calls);
        let result = if count {
            self.interpret_fast_counted(stack, prologue_counted)
        } else {
            self.interpret_fast_with_hooks(stack)
                .map(|exit| (exit, true))
        };
        (self.fast_hook_count, self.fast_hook_calls) = saved;
        result
    }

    fn interpret_fast_counted(
        &mut self,
        stack: &mut CallStack,
        prologue_counted: bool,
    ) -> VmResult<(FastExit, bool)> {
        let thread = self.current_coroutine;
        let nested = self.count_lease.is_some();
        let saved_fetched = self.count_fetched;
        if !nested {
            let Some(hook) = self.debug_hooks.get(&thread) else {
                return Ok((FastExit::Fallback, false));
            };
            self.count_budget = hook.remaining;
            self.count_lease = Some(CountLease {
                thread,
                invalidated: false,
                count_only: !self.fast_hook_calls,
            });
            self.refresh_return_watch();
        } else if self
            .count_lease
            .is_some_and(|lease| lease.invalidated || lease.thread != thread)
        {
            return Ok((FastExit::Fallback, false));
        }
        if prologue_counted {
            self.count_budget += 1;
        }
        self.count_fetched = crate::vm::NOT_FETCHED;
        let result = self.interpret_fast_with_hooks(stack);
        let top = stack.len() - 1;
        let counted = self.count_fetched == crate::vm::fetched_mark(stack.len(), stack[top].pc);
        self.count_fetched = saved_fetched;
        if !nested {
            let lease = self.count_lease.take().expect("the lease is open");
            self.refresh_return_watch();
            if !lease.invalidated
                && let Some(hook) = self.debug_hooks.get_mut(&thread)
            {
                hook.remaining = self.count_budget;
            }
        }
        let exit = result?;
        let counted = counted
            && matches!(
                exit,
                FastExit::Fallback
                    | FastExit::IndexMiss
                    | FastExit::NewIndexMiss
                    | FastExit::Declined
            );
        Ok((exit, counted))
    }

    fn finish_debug_hook(&mut self, thread: Option<ThreadHandle>) {
        self.debug_hook_depth -= 1;
        if let Some(hook) = self.debug_hooks.get_mut(&thread) {
            hook.running = false;
        }
    }

    pub(crate) fn fire_count_hook(&mut self, frames: &mut CallStack) -> VmResult<bool> {
        if self.debug_hooks.is_empty() {
            return Ok(false);
        }
        if self.debug_hook_depth != 0
            || frames.iter().any(|frame| {
                frame
                    .extras_ref()
                    .is_some_and(|extras| extras.suppress_debug_hooks)
            })
        {
            return Ok(false);
        }
        // These register-maintenance instructions are emitted by Oximond's
        // fixed-frame compiler. Lua 5.1 performs the equivalent stack work as
        // part of surrounding opcodes, so they do not advance its count hook.
        if frames.last().is_some_and(|frame| {
            frame
                .chunk
                .code
                .get(frame.pc)
                .and_then(|instruction| instruction.opcode().ok())
                .is_some_and(|opcode| !opcode.counts_for_hooks())
        }) {
            return Ok(false);
        }
        let thread = self.current_coroutine;
        let fire = if let Some(hook) = self.debug_hooks.get_mut(&thread) {
            // While the fast loop holds the count, a nested dispatch loop on
            // the same thread counts against the loop's budget.
            let remaining = match &self.count_lease {
                Some(lease) if lease.thread == thread && !lease.invalidated => {
                    &mut self.count_budget
                }
                _ => &mut hook.remaining,
            };
            if hook.running || hook.count == 0 {
                false
            } else if *remaining <= 1 {
                *remaining = hook.count;
                true
            } else {
                *remaining -= 1;
                false
            }
        } else {
            false
        };
        if fire {
            // The hook's callback runs Lua code the recorder cannot follow.
            if self.jit.trace.recording() {
                self.jit.trace.abort_recording(
                    &mut self.jit.statistics.trace,
                    crate::vm::TraceAbortReason::Flushed,
                );
            }
            self.fire_debug_hook(frames, DebugHookEvent::Count)
        } else {
            Ok(false)
        }
    }

    pub(crate) fn string_dump(&self, arguments: &[Value]) -> VmResult<Vec<Value>> {
        let Some(Value::Closure(handle)) = arguments.first() else {
            let found = arguments.first().map_or("nil", Value::type_name);
            return Err(RuntimeError::TypeError {
                operation: "call string.dump with",
                expected: "function",
                found,
            }
            .into());
        };
        let body = self.closure_body(*handle)?;
        let bytes = binary::dump(&body.chunk())
            .map_err(|error| RuntimeError::message(error.to_string()))?;
        Ok(vec![Value::bytes(bytes)])
    }

    pub(crate) fn start_tostring(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        arguments: Vec<Value>,
    ) -> VmResult<()> {
        self.start_tostring_with_completion(
            frames,
            ToStringCompletion::Direct(CallReturn {
                base: instruction.a(),
                results: instruction.c(),
            }),
            arguments,
        )
    }

    pub(crate) fn start_tostring_protected(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        arguments: Vec<Value>,
    ) -> VmResult<()> {
        self.start_tostring_with_completion(
            frames,
            ToStringCompletion::Protected(CallReturn {
                base: instruction.a(),
                results: instruction.c(),
            }),
            arguments,
        )
    }

    pub(crate) fn start_tostring_with_completion(
        &mut self,
        frames: &mut CallStack,
        completion: ToStringCompletion,
        arguments: Vec<Value>,
    ) -> VmResult<()> {
        let value = arguments
            .into_iter()
            .next()
            .ok_or_else(|| RuntimeError::argument_error(1, "value expected"))?;
        let Some(handler) = self.tostring_metamethod(&value)? else {
            return self.finish_tostring(
                frames,
                completion,
                Value::String(self.runtime.format_value(&value)?),
            );
        };
        match handler {
            Value::NativeFunction(function) => {
                let result = self
                    .call_native(frames, function, std::slice::from_ref(&value))
                    .map_err(|error| {
                        // `tostring` called the handler from C: no call names
                        // it and the error has no position. LuaJIT names the
                        // handler by the call of `tostring` instead.
                        if self.compatibility == crate::compat::CompatibilityProfile::LuaJit21 {
                            VmError::from(error)
                        } else {
                            self.protected_argument_error(
                                &Value::NativeFunction(function),
                                error.into(),
                            )
                        }
                    })?
                    .into_iter()
                    .next()
                    .unwrap_or(Value::Nil);
                self.finish_tostring(frames, completion, result)
            }
            Value::Closure(function) => {
                let body = self.closure_body(function)?;
                let frame = self.make_frame(
                    body.chunk(),
                    vec![value.clone()],
                    body.shared_cells(),
                    Some(ReturnTarget::ToString(Box::new(ToStringContinuation {
                        completion,
                        value,
                    }))),
                    None,
                    body.environment(),
                    Some(function),
                )?;
                frames.push_within_limit(frame)?;
                Ok(())
            }
            Value::BuiltinFunction(builtin) => {
                self.check_native_handler_depth(frames)?;
                let target = |value: Value| {
                    ReturnTarget::ToString(Box::new(ToStringContinuation { completion, value }))
                };
                match self.call_builtin(builtin, std::slice::from_ref(&value)) {
                    Err(VmError::InvalidDispatch(_)) => {
                        let target = target(value.clone());
                        self.push_builtin_call_frame(frames, builtin, vec![value], target)
                    }
                    results => {
                        let result = results
                            .map_err(|error| {
                                self.protected_argument_error(
                                    &Value::BuiltinFunction(builtin),
                                    error,
                                )
                            })?
                            .into_iter()
                            .next()
                            .unwrap_or(Value::Nil);
                        self.finish_tostring(frames, completion, result)
                    }
                }
            }
            value => Err(RuntimeError::message(format!(
                "attempt to call a {} value",
                value.type_name()
            ))
            .into()),
        }
    }

    /// Stock bounds nested C calls (`LUAI_MAXCCALLS`); a `__tostring` that is
    /// a builtin reaching `tostring` again recurses through them.
    fn check_native_handler_depth(&self, frames: &CallStack) -> VmResult<()> {
        let nested = frames
            .iter()
            .filter(|frame| {
                matches!(
                    frame.caller,
                    Some(ReturnTarget::ToString(_) | ReturnTarget::StringFormat(_))
                )
            })
            .count();
        if nested >= NATIVE_HANDLER_DEPTH_LIMIT {
            let message = if self.compatibility == crate::compat::CompatibilityProfile::LuaJit21 {
                "stack overflow"
            } else {
                "C stack overflow"
            };
            return Err(RuntimeError::Raised(Value::from(message)).into());
        }
        Ok(())
    }

    /// What `__tostring` returned, as `luaL_tolstring` takes it: a string, or
    /// a number read as its string. Anything else raises. `LuaJIT`'s
    /// `string.format` formats any other result like a `%s` argument.
    pub(crate) fn string_format_tostring_result(&self, value: Value) -> VmResult<Value> {
        match value {
            Value::String(_) => Ok(value),
            Value::Integer(_) | Value::Number(_) => {
                Ok(Value::String(self.runtime.format_value(&value)?))
            }
            value if self.compatibility.is_lua51_family() => {
                Ok(Value::String(self.runtime.format_value(&value)?))
            }
            _ => Err(RuntimeError::message("'__tostring' must return a string").into()),
        }
    }

    pub(crate) fn tostring_metamethod(&self, value: &Value) -> VmResult<Option<Value>> {
        let Some(metatable) = self.runtime.raw_metatable(value)? else {
            return Ok(None);
        };
        let handler = self.runtime.table_get_string(metatable, tostring_key())?;
        Ok((!matches!(handler, Value::Nil)).then_some(handler))
    }

    /// Whether [`Self::tostring_metamethod`] finds a handler, without
    /// returning (and so copying) it.
    pub(crate) fn has_tostring_metamethod(&self, value: &Value) -> VmResult<bool> {
        let Some(metatable) = self.runtime.raw_metatable(value)? else {
            return Ok(false);
        };
        Ok(self
            .runtime
            .table_get_string_ref(metatable, tostring_key())
            .is_some())
    }

    pub(crate) fn finish_tostring(
        &mut self,
        frames: &mut CallStack,
        completion: ToStringCompletion,
        value: Value,
    ) -> VmResult<()> {
        // Lua 5.2 and later require a string (or number) from `__tostring`;
        // 5.1 and LuaJIT return whatever it returned.
        let value = if self.compatibility.is_lua51_family() {
            Ok(value)
        } else {
            match value {
                Value::String(_) => Ok(value),
                Value::Integer(_) | Value::Number(_) => {
                    Ok(Value::String(self.runtime.format_value(&value)?))
                }
                _ => Err(VmError::from(RuntimeError::message(
                    "'__tostring' must return a string",
                ))),
            }
        };
        let mut frame = frames.last_mut().expect("tostring caller exists");
        let value = match (value, completion) {
            (Ok(value), _) => value,
            (Err(error), ToStringCompletion::Protected(target)) => {
                self.write_protected_results(
                    &mut frame,
                    target.base,
                    target.results,
                    false,
                    vec![vm_error_object(&error)?],
                )?;
                return Ok(());
            }
            (Err(error), ToStringCompletion::Direct(_)) => return Err(error),
        };
        match completion {
            ToStringCompletion::Direct(target) => write_call_results(
                &mut frame,
                Instruction::call(target.base, 0, target.results),
                vec![value],
            )?,
            ToStringCompletion::Protected(target) => {
                self.write_protected_results(
                    &mut frame,
                    target.base,
                    target.results,
                    true,
                    vec![value],
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn start_string_format(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        arguments: Vec<Value>,
    ) -> VmResult<()> {
        if let Some(results) = self.try_string_format_direct(&arguments)? {
            let mut frame = frames.last_mut().expect("string.format caller exists");
            write_call_results(&mut frame, instruction, results)?;
            return Ok(());
        }
        self.advance_string_format(
            frames,
            StringFormatContinuation {
                completion: StringFormatCompletion::Direct(CallReturn {
                    base: instruction.a(),
                    results: instruction.c(),
                }),
                arguments,
                format_index: 0,
                argument_index: 1,
                converting: 0,
            },
        )
    }

    pub(crate) fn start_string_format_protected(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        arguments: Vec<Value>,
    ) -> VmResult<()> {
        if let Some(results) = self.try_string_format_direct(&arguments)? {
            let mut frame = frames.last_mut().expect("string.format caller exists");
            self.write_protected_results(
                &mut frame,
                instruction.a(),
                instruction.c(),
                true,
                results,
            )?;
            return Ok(());
        }
        self.advance_string_format(
            frames,
            StringFormatContinuation {
                completion: StringFormatCompletion::Protected(CallReturn {
                    base: instruction.a(),
                    results: instruction.c(),
                }),
                arguments,
                format_index: 0,
                argument_index: 1,
                converting: 0,
            },
        )
    }

    pub(crate) fn advance_string_format(
        &mut self,
        frames: &mut CallStack,
        mut continuation: StringFormatContinuation,
    ) -> VmResult<()> {
        loop {
            let Some(format) = string_format_source(&self.runtime, continuation.arguments.first())
            else {
                return self.finish_string_format(frames, &continuation);
            };
            let Some((argument, format_index, argument_index)) = next_string_conversion(
                &format,
                continuation.format_index,
                continuation.argument_index,
            ) else {
                return self.finish_string_format(frames, &continuation);
            };
            continuation.format_index = format_index;
            continuation.argument_index = argument_index;
            continuation.converting = argument;
            let Some(value) = continuation.arguments.get(argument).cloned() else {
                return self.finish_string_format(frames, &continuation);
            };
            let Some(handler) = self.tostring_metamethod(&value)? else {
                continuation.arguments[argument] =
                    Value::String(self.runtime.format_value(&value)?);
                continue;
            };
            match handler {
                Value::NativeFunction(function) => {
                    let converted = self
                        .call_native(frames, function, std::slice::from_ref(&value))
                        .map_err(|error| {
                            self.protected_argument_error(
                                &Value::NativeFunction(function),
                                VmError::Runtime(error),
                            )
                        })?
                        .into_iter()
                        .next()
                        .unwrap_or(Value::Nil);
                    continuation.arguments[argument] =
                        self.string_format_tostring_result(converted)?;
                }
                Value::Closure(function) => {
                    let body = self.closure_body(function)?;
                    let frame = self.make_frame(
                        body.chunk(),
                        vec![value],
                        body.shared_cells(),
                        Some(ReturnTarget::StringFormat(Box::new(continuation))),
                        None,
                        body.environment(),
                        Some(function),
                    )?;
                    frames.push_within_limit(frame)?;
                    return Ok(());
                }
                Value::BuiltinFunction(builtin) => {
                    self.check_native_handler_depth(frames)?;
                    match self.call_builtin(builtin, std::slice::from_ref(&value)) {
                        Err(VmError::InvalidDispatch(_)) => {
                            return self.push_builtin_call_frame(
                                frames,
                                builtin,
                                vec![value],
                                ReturnTarget::StringFormat(Box::new(continuation)),
                            );
                        }
                        results => {
                            let converted = results
                                .map_err(|error| {
                                    self.protected_argument_error(
                                        &Value::BuiltinFunction(builtin),
                                        error,
                                    )
                                })?
                                .into_iter()
                                .next()
                                .unwrap_or(Value::Nil);
                            continuation.arguments[argument] =
                                self.string_format_tostring_result(converted)?;
                        }
                    }
                }
                value => {
                    return Err(RuntimeError::message(format!(
                        "attempt to call a {} value",
                        value.type_name()
                    ))
                    .into());
                }
            }
        }
    }

    pub(crate) fn resume_string_format(
        &mut self,
        frames: &mut CallStack,
        mut continuation: StringFormatContinuation,
        value: Value,
    ) -> VmResult<()> {
        let completion = continuation.completion;
        let result = match self.string_format_tostring_result(value) {
            Ok(value) => {
                continuation.arguments[continuation.converting] = value;
                self.advance_string_format(frames, continuation)
            }
            Err(error) => Err(error),
        };
        match (result, completion) {
            (Err(error), StringFormatCompletion::Protected(target)) => {
                let mut frame = frames
                    .last_mut()
                    .expect("protected string.format caller exists");
                self.write_protected_results(
                    &mut frame,
                    target.base,
                    target.results,
                    false,
                    vec![vm_error_object(&error)?],
                )
            }
            (result, _) => result,
        }
    }

    pub(crate) fn finish_string_format(
        &mut self,
        frames: &mut CallStack,
        continuation: &StringFormatContinuation,
    ) -> VmResult<()> {
        let function = self
            .runtime
            .direct_native(DirectNativeKind::Format)
            .ok_or_else(|| {
                RuntimeError::message("modern string.format formatter is not installed")
            })?;
        let results = self.call_native(frames, function, &continuation.arguments)?;
        let mut frame = frames.last_mut().expect("string.format caller exists");
        match continuation.completion {
            StringFormatCompletion::Direct(target) => write_call_results(
                &mut frame,
                Instruction::call(target.base, 0, target.results),
                results,
            )?,
            StringFormatCompletion::Protected(target) => {
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
}

pub(crate) fn string_format_source(runtime: &Runtime, value: Option<&Value>) -> Option<Vec<u8>> {
    match value? {
        Value::String(value) => Some(value.as_bytes().to_vec()),
        numeric @ (Value::Integer(_) | Value::Number(_)) => runtime
            .format_value(numeric)
            .ok()
            .map(|value| value.as_bytes().to_vec()),
        _ => None,
    }
}

pub(crate) fn next_string_conversion(
    format: &[u8],
    mut format_index: usize,
    mut argument_index: usize,
) -> Option<(usize, usize, usize)> {
    while format_index < format.len() {
        if format[format_index] != b'%' {
            format_index += 1;
            continue;
        }
        format_index += 1;
        if format.get(format_index) == Some(&b'%') {
            format_index += 1;
            continue;
        }
        while matches!(
            format.get(format_index),
            Some(b'-' | b'+' | b' ' | b'#' | b'0' | b'.' | b'0'..=b'9')
        ) {
            format_index += 1;
        }
        let conversion = *format.get(format_index)?;
        format_index += 1;
        let current_argument = argument_index;
        argument_index += 1;
        if conversion == b's' {
            return Some((current_argument, format_index, argument_index));
        }
    }
    None
}

pub(crate) fn table_handle(value: &Value, operation: &'static str) -> RuntimeResult<TableHandle> {
    match value {
        Value::Table(value) => Ok(*value),
        value => Err(RuntimeError::TypeError {
            operation,
            expected: "table",
            found: value.type_name(),
        }),
    }
}

/// The table a `next`, `pairs` or `ipairs` call takes as its first argument.
pub(crate) fn builtin_table(arguments: &[Value]) -> RuntimeResult<TableHandle> {
    match arguments.first() {
        Some(Value::Table(table)) => Ok(*table),
        argument => Err(RuntimeError::bad_argument(1, "table", argument)),
    }
}

pub(crate) fn builtin_table_value(
    value: &Value,
    operation: &'static str,
) -> RuntimeResult<TableHandle> {
    table_handle(value, operation)
}

/// `debug.getupvalue` and `debug.setupvalue` take a function and a number.
fn upvalue_arguments(arguments: &[Value]) -> RuntimeResult<()> {
    match arguments.first() {
        Some(Value::Closure(_) | Value::NativeFunction(_) | Value::BuiltinFunction(_)) => {}
        function => return Err(RuntimeError::bad_argument(1, "function", function)),
    }
    match arguments.get(1) {
        Some(Value::Integer(_) | Value::Number(_)) => Ok(()),
        index => Err(RuntimeError::bad_argument(2, "number", index)),
    }
}

/// The `__tostring` key, shared: strings carry a metatable, so `tostring`
/// of any string looks it up.
fn tostring_key() -> &'static LuaString {
    static KEY: OnceLock<LuaString> = OnceLock::new();
    KEY.get_or_init(|| crate::string::shared_key(b"__tostring"))
}
