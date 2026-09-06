//! Calls from host code into a live VM.

use crate::bytecode::{Chunk, DebugMetadata, Instruction};
use crate::runtime::{
    ExitRequest, GcRootBuilder, GcRoots, NativeFunctionHandle, RuntimeError, Value,
};
use crate::vm::arith::call_metamethod_key;
use crate::vm::debug_info::{
    SuspendedFrame, debug_level, debug_namewhat, debug_short_source, level_number,
    suspended_info_at_level, suspended_level_count,
};
use crate::vm::errors::{call_argument_error, call_type_error_anonymous, raised_value};
use crate::vm::libraries::Window;
use crate::vm::registers::write_fixed_results;
use crate::vm::stack::{CallStack, FrameMut};
use crate::vm::{
    DebugInfo, HostCallback, HostFunction, MAX_INDEX_HOPS, Vm, VmError, VmResult, VmStackFrame,
};

/// Most arguments a host call the fast loop runs itself may take: the most a
/// call copies to the machine stack when the callback cannot read them in
/// the registers (see `Vm::call_host_in_registers`).
pub(crate) const HOST_FAST_ARGUMENTS: usize = 8;

/// Most emptied host call roots kept for reuse: one per level of nested
/// host calls that repeat.
const SPARE_ROOTS: usize = 8;
/// Largest roots kept for reuse, in entries. A callback for a closing frame
/// roots the whole stack, which is rare and not worth holding.
const SPARE_ROOTS_CAPACITY: usize = 64;

/// A chunk that calls the builtin in register 0 with the following
/// `argument_count` registers and returns its results. Builtins that need
/// the frame stack run through it instead of `call_builtin`. Its `=[C]`
/// source makes debug information describe the frame as the C function
/// that called the builtin.
pub(crate) fn builtin_call_chunk(argument_count: usize) -> VmResult<Chunk> {
    let count = u8::try_from(argument_count)
        .map_err(|_| RuntimeError::message("too many arguments to call"))?;
    let registers = count
        .checked_add(1)
        .ok_or_else(|| RuntimeError::message("too many arguments to call"))?;
    Ok(Chunk::with_parts(
        registers,
        registers,
        Vec::new(),
        vec![
            Instruction::call(0, count, 0),
            Instruction::return_open(0, 0),
        ],
        Vec::new(),
        Vec::new(),
    )
    .with_debug_metadata(DebugMetadata::new(C_SOURCE, Vec::new())))
}

/// The source name of C functions in debug information.
pub(crate) const C_SOURCE: &str = "=[C]";

fn host_runtime_error(error: VmError) -> RuntimeError {
    match error {
        VmError::Runtime(error) | VmError::LocatedRuntime { error, .. } => error,
        VmError::HookError(error) => RuntimeError::Hook(Box::new(host_runtime_error(*error))),
        other => RuntimeError::message(other.to_string()),
    }
}

pub(crate) fn stack_frame_info(info: DebugInfo) -> VmStackFrame {
    let (parameter_count, is_vararg) = info
        .chunk
        .as_ref()
        .map_or((0, true), |chunk| (chunk.parameter_count, chunk.is_vararg));
    let (name, name_what) = match info.call_site {
        Some(site) => (Some(site.name), debug_namewhat(site.kind)),
        None => (None, ""),
    };
    VmStackFrame {
        function: info.function,
        short_source: debug_short_source(&info.source),
        source: info.source,
        current_line: info
            .current_line
            .and_then(|line| i32::try_from(line).ok())
            .unwrap_or(-1),
        what: info.what,
        definition_line: info.definition_line,
        last_definition_line: info.last_definition_line,
        upvalue_count: info.upvalue_count,
        parameter_count,
        is_vararg,
        is_tail_call: false,
        first_transfer: info.hook_transfer.0,
        transfer_count: info.hook_transfer.1,
        name,
        name_what,
    }
}

/// Nesting limit for `table_meta` host callbacks.
const MAX_TABLE_META_DEPTH: u8 = 195;

impl Vm {
    /// Installs an embedding hook without executing Lua or firing the old hook.
    /// `None` selects the main thread. Mask bits are Lua's call, return, line,
    /// and count bits (1, 2, 4, and 8).
    ///
    /// # Errors
    ///
    /// Errors when `thread` does not name a live coroutine, or when `count`
    /// falls outside the accepted hook instruction-count range.
    pub fn set_host_debug_hook(
        &mut self,
        thread: Option<crate::runtime::ThreadHandle>,
        function: Option<Value>,
        mask: i32,
        count: i32,
    ) -> VmResult<()> {
        let mut arguments = Vec::with_capacity(4);
        if let Some(thread) = thread {
            arguments.push(Value::Thread(thread));
        }
        arguments.push(function.unwrap_or(Value::Nil));
        let mut events = String::new();
        if mask & 1 != 0 {
            events.push('c');
        }
        if mask & 2 != 0 {
            events.push('r');
        }
        if mask & 4 != 0 {
            events.push('l');
        }
        arguments.push(Value::from(events));
        arguments.push(Value::Integer(if mask & 8 != 0 {
            i64::from(count.max(0))
        } else {
            0
        }));
        let (_, hook) = self.parse_debug_sethook(&arguments)?;
        if let Some(hook) = hook {
            self.debug_hooks.insert(thread, hook);
        } else {
            self.debug_hooks.remove(&thread);
        }
        self.debug_hooks_changed();
        Ok(())
    }

    /// Calls `debug.getinfo` from a C wrapper without losing the suspended
    /// Lua stack. Lua's level includes the debug function itself, so level 2
    /// names the Lua caller of the current C callback.
    fn debug_get_info_from_host(&mut self, arguments: &[Value]) -> VmResult<Vec<Value>> {
        let Some(level) = arguments.first().and_then(level_number) else {
            return self.debug_get_info(&CallStack::new(), arguments);
        };
        let Some(level) = debug_level(level) else {
            return Ok(vec![Value::Nil]);
        };
        let Some(info) = self.host_debug_info(level - 1) else {
            return Ok(vec![Value::Nil]);
        };
        self.debug_table(info, arguments)
    }

    /// The coroutine invoking a host callback, or `None` for the main thread.
    #[must_use]
    pub const fn current_coroutine(&self) -> Option<crate::runtime::ThreadHandle> {
        self.current_coroutine
    }

    /// Environment used when loading the next top-level chunk on this thread.
    #[must_use]
    pub fn thread_environment(&self) -> crate::runtime::TableHandle {
        self.thread_environment
    }

    /// Changes the main thread's environment after validating the table handle.
    ///
    /// # Errors
    ///
    /// Returns an error when `table` is a stale table handle.
    pub fn set_thread_environment(&mut self, table: crate::runtime::TableHandle) -> VmResult<()> {
        self.runtime.table_length(table)?;
        self.thread_environment = table;
        Ok(())
    }

    /// Uses the embedding application's main-thread identity for coroutine.running.
    ///
    /// # Errors
    ///
    /// Returns an error when `thread` is not a live coroutine handle.
    pub fn set_main_thread(&mut self, thread: crate::runtime::ThreadHandle) -> VmResult<()> {
        if !self.runtime.thread_exists(thread) {
            return Err(RuntimeError::message("invalid thread").into());
        }
        self.main_thread = Some(thread);
        Ok(())
    }

    /// The environment the next top-level chunk on `thread` loads into.
    ///
    /// # Errors
    ///
    /// Returns an error when `thread` names a coroutine that is not live.
    pub fn thread_environment_for(
        &self,
        thread: Option<crate::runtime::ThreadHandle>,
    ) -> VmResult<crate::runtime::TableHandle> {
        if thread == self.current_coroutine {
            return Ok(self.thread_environment);
        }
        if let Some(thread) = thread {
            return self
                .coroutines
                .get(&thread)
                .map(|thread| thread.environment)
                .ok_or_else(|| RuntimeError::message("invalid coroutine").into());
        }
        // Each nested resumer parks its environment independently of active callbacks.
        Ok(self.main_thread_environment())
    }

    /// Sets the environment the next top-level chunk on `thread` loads into.
    ///
    /// # Errors
    ///
    /// Returns an error when `table` is a stale table handle, or when `thread`
    /// names a coroutine that is not live.
    ///
    /// # Panics
    ///
    /// Panics if the record of the current coroutine is missing from the
    /// coroutine table.
    pub fn set_thread_environment_for(
        &mut self,
        thread: Option<crate::runtime::ThreadHandle>,
        table: crate::runtime::TableHandle,
    ) -> VmResult<()> {
        self.runtime.table_length(table)?;
        if thread == self.current_coroutine {
            self.thread_environment = table;
            if let Some(thread) = thread {
                self.coroutines
                    .get_mut(&thread)
                    .expect("current coroutine exists")
                    .environment = table;
            }
        } else if let Some(thread) = thread {
            let coroutine = self
                .coroutines
                .get_mut(&thread)
                .ok_or_else(|| RuntimeError::message("invalid coroutine"))?;
            coroutine.environment = table;
            let mut roots = GcRootBuilder::new();
            roots.value(&Value::Table(table));
            if let Some((frames, error)) = &coroutine.failed {
                roots.extend(&Self::gc_roots_full_windows(table, frames, &[]));
                if let Ok(value) = crate::vm::errors::vm_error_object(error) {
                    roots.value(&value);
                }
            }
            match &mut coroutine.state {
                crate::vm::CoroutineState::New(Some(entry)) => roots.value(entry),
                crate::vm::CoroutineState::Suspended {
                    frames,
                    host_resume,
                    ..
                } => {
                    let frame_roots = Self::gc_roots(table, frames, &[]);
                    roots.extend(&frame_roots);
                    if let Some(resume) = host_resume {
                        Self::root_host_resume(&mut roots, resume);
                    }
                    self.runtime.set_thread_roots(thread, roots.finish())?;
                    coroutine.roots_stale = false;
                    return Ok(());
                }
                _ => {}
            }
            self.runtime.set_thread_roots(thread, roots.finish())?;
        } else {
            self.parked_main_environment = Some(table);
        }
        Ok(())
    }

    fn main_thread_environment(&self) -> crate::runtime::TableHandle {
        self.parked_main_environment
            .unwrap_or(self.thread_environment)
    }

    /// The current native callback is level 0; its suspended Lua caller is
    /// level 1. Older callers follow in stack order.
    #[must_use]
    pub fn host_stack_info(&self, level: usize) -> Option<VmStackFrame> {
        let info = self.host_debug_info(level)?;
        let mut info = stack_frame_info(info);
        if let Some((stack, frame)) = self.host_local_location(level) {
            info.is_tail_call = self.host_snapshots[stack].local_frames()[frame]
                .frame
                .tail_calls
                != 0;
        }
        Some(info)
    }

    /// Resolves a level of the stack seen from inside the current host
    /// callback. Suspended Lua frames were captured when each callback was
    /// entered; their debug records are built only here.
    fn host_debug_info(&self, level: usize) -> Option<DebugInfo> {
        let current = self.active_host_function?;
        if level == 0 {
            return Some(self.host_c_info(current));
        }
        let mut remaining = level - 1;
        for (depth, snapshot) in self.host_snapshots.iter().rev().enumerate() {
            let frames = snapshot.debug_frames();
            let levels = suspended_level_count(frames);
            if remaining < levels {
                return suspended_info_at_level(frames, remaining + 1);
            }
            remaining -= levels;
            if let Some(&function) = self.suspended_host_functions.iter().rev().nth(depth) {
                if remaining == 0 {
                    return Some(self.host_c_info(function));
                }
                remaining -= 1;
            }
        }
        None
    }

    pub(crate) fn host_c_info(&self, function: NativeFunctionHandle) -> DebugInfo {
        DebugInfo {
            function: Some(Value::NativeFunction(function)),
            source: crate::string::LuaString::from("=[C]"),
            current_line: None,
            call_site: None,
            what: "C",
            definition_line: -1,
            last_definition_line: -1,
            upvalue_count: self
                .host_functions
                .get(&function)
                .map_or(0, |host| host.captures.len()),
            hook_transfer: (0, 0),
            chunk: None,
        }
    }

    /// Performs a full collection with host-managed stack values as roots.
    /// This also retains suspended Lua calls and native closure captures.
    pub fn collect_full_with_values(&mut self, values: &[Value]) -> crate::runtime::GcStats {
        let roots = self.gc_roots_with_coroutines(&mut CallStack::new(), values);
        let stats = self.runtime.collect_full(roots);
        self.drain_pending_finalizers();
        stats
    }

    /// Runs the pending finalizers when a state closes, including those of
    /// objects still reachable from the global environment, in reverse
    /// order of registration. Lua 5.4 and later turn a finalizer error into
    /// a warning, `error in __gc (message)`; earlier versions ignore it.
    ///
    /// A finalizer may call `os.exit`. The last such request is returned,
    /// and one that does not close the state stops the remaining finalizers,
    /// as the process would end there.
    pub fn run_close_finalizers(&mut self) -> Option<ExitRequest> {
        self.runtime.queue_close_finalizers();
        self.drain_pending_finalizers()
    }

    fn drain_pending_finalizers(&mut self) -> Option<ExitRequest> {
        let mut exit = None;
        while let Ok(Some(finalizer)) = self.runtime.take_pending_finalizer() {
            let Err(error) = self.call_value(finalizer.handler, vec![finalizer.object]) else {
                continue;
            };
            if error.exit_request().is_none() {
                self.warn_finalizer_error(&error);
            }
            if let Some(request) = error.exit_request() {
                exit = Some(request);
                if !request.close {
                    break;
                }
            }
        }
        exit
    }

    /// Reports a failed `__gc` as Lua 5.4 does: the warning
    /// `error in __gc (message)`, which is printed only while warnings are on.
    pub(crate) fn warn_finalizer_error(&mut self, error: &VmError) {
        if !self.finalizer_errors_warn() {
            return;
        }
        let message = match crate::vm::errors::vm_error_object(error) {
            Ok(Value::String(text)) => text.as_bytes().to_vec(),
            Ok(_) => b"error object is not a string".to_vec(),
            Err(_) => return,
        };
        let runtime = &mut self.runtime;
        let _ = runtime.emit_warning(b"error in ", true);
        let _ = runtime.emit_warning(b"__gc", true);
        let _ = runtime.emit_warning(b" (", true);
        let _ = runtime.emit_warning(&message, true);
        let _ = runtime.emit_warning(b")", false);
    }

    /// Registers a VM-aware native closure. `captures` stay alive until the VM
    /// is dropped, including while Lua only retains the function indirectly.
    ///
    /// # Errors
    ///
    /// Returns an error when the native function handle space is exhausted.
    pub fn register_host_function(
        &mut self,
        callback: HostCallback,
        token: u64,
        captures: Vec<Value>,
    ) -> VmResult<Value> {
        self.register_owned_host_function(callback, token, captures, None)
    }

    /// Retains host storage for as long as this VM can call the function.
    /// The owner may release native resources when the VM is dropped, so its
    /// destructor must not access the VM being destroyed.
    ///
    /// # Errors
    ///
    /// Returns an error when the native function handle space is exhausted.
    pub fn register_owned_host_function(
        &mut self,
        callback: HostCallback,
        token: u64,
        captures: Vec<Value>,
        owner: Option<std::rc::Rc<dyn std::any::Any>>,
    ) -> VmResult<Value> {
        let handle = self.runtime.register_host_native(token)?;
        let slot = handle.index() as usize;
        if self.host_dispatch.len() <= slot {
            self.host_dispatch.resize(slot + 1, None);
        }
        self.host_dispatch[slot] = Some((callback, token));
        if let Some(direct) = self.host_direct.get_mut(slot) {
            *direct = None;
        }
        self.host_functions.insert(
            handle,
            HostFunction {
                captures,
                _owner: owner,
            },
        );
        Ok(Value::NativeFunction(handle))
    }

    /// Gives the host function `function` a [`HostDirectCallback`], which
    /// fixed calls from the interpreter's fast loop then use in place of
    /// its [`HostCallback`]. Returns `false` for a value that is no
    /// registered host function.
    ///
    /// # Safety
    ///
    /// `direct` must keep the contract of [`HostDirectCallback`] and behave
    /// as the function's registered callback does.
    #[allow(unsafe_code)]
    pub unsafe fn set_host_direct_callback(
        &mut self,
        function: &Value,
        direct: crate::vm::HostDirectCallback,
    ) -> bool {
        let Value::NativeFunction(function) = function else {
            return false;
        };
        let slot = function.index() as usize;
        if !self.is_host_function(*function) {
            return false;
        }
        if self.host_direct.len() <= slot {
            self.host_direct.resize(slot + 1, None);
        }
        self.host_direct[slot] = Some(direct);
        true
    }

    /// The direct entry of the host function `function` and its token, when
    /// it has one and a call passes `count` arguments it can read in place.
    #[inline]
    pub(crate) fn host_direct_entry(
        &self,
        function: NativeFunctionHandle,
        count: usize,
    ) -> Option<(crate::vm::HostDirectCallback, u64)> {
        if count > HOST_FAST_ARGUMENTS {
            return None;
        }
        let slot = function.index() as usize;
        let direct = (*self.host_direct.get(slot)?)?;
        let (_, token) = (*self.host_dispatch.get(slot)?)?;
        Some((direct, token))
    }

    pub(crate) fn is_host_function(&self, function: NativeFunctionHandle) -> bool {
        matches!(
            self.host_dispatch.get(function.index() as usize),
            Some(Some(_))
        )
    }

    /// Whether the fast loop may run the host callback `function` with
    /// `count` arguments through [`Self::call_host_fixed`]: the main thread,
    /// where no host callback can yield, with no debug hook and no trace
    /// recording, which watch calls through the dispatch loop.
    #[inline]
    pub(crate) fn host_call_in_fast_loop(
        &self,
        function: NativeFunctionHandle,
        count: usize,
    ) -> bool {
        count <= HOST_FAST_ARGUMENTS
            && self.current_coroutine.is_none()
            && self.debug_hooks.is_empty()
            && !self.jit.trace.recording()
            && self.is_host_function(function)
    }

    /// Calls the host callback `function` for a fixed call of the running
    /// frame, from the fast loop, with the `count` arguments in the registers
    /// from `start`, and writes its results as `call` says. This is what the
    /// dispatch loop's `call_frame` and `call_one_frame` do for a host
    /// callback, without leaving the fast loop and with the callback reading
    /// its arguments in the registers instead of in a fresh vector. `one`
    /// marks a `CallOne`,
    /// whose general path reports no call to a hook.
    ///
    /// The caller has checked [`Self::host_call_in_fast_loop`] and that the
    /// registers exist. The callback runs as not yieldable: on the main
    /// thread every yield fails as one from outside a coroutine, whatever the
    /// boundary allows, so the frame walk that decides it is skipped.
    ///
    /// Returns `false`, with nothing done, for a `string` library function
    /// that hands arithmetic on to a metamethod.
    pub(crate) fn call_host_fixed(
        &mut self,
        frames: &mut CallStack,
        function: NativeFunctionHandle,
        call: Instruction,
        (start, count): (usize, usize),
        one: bool,
    ) -> VmResult<bool> {
        debug_assert!(self.host_call_in_fast_loop(function, count));
        let frame_index = frames.len() - 1;
        if let Some(&Some(direct)) = self.host_direct.get(function.index() as usize)
            && let Some(&Some((_, token))) = self.host_dispatch.get(function.index() as usize)
            && usize::from(call.a()) + usize::from(call.c())
                <= (frames[frame_index].top - frames[frame_index].base) as usize
        {
            // No `string` library function has a direct entry, so none of
            // them hands arithmetic on from here.
            debug_assert!(self.runtime.direct_native_kind(function).is_none());
            debug_assert_ne!(call.c(), 0, "a fixed call has a result count");
            let written = self.call_host_direct(
                frames,
                function,
                (direct, token),
                (start, count),
                (usize::from(call.a()), usize::from(call.c())),
            );
            if let Err(error) = written {
                return Err(call_argument_error(
                    &frames[frame_index],
                    frame_index.checked_sub(1).map(|i| &frames[i]),
                    error,
                    self.compatibility,
                )
                .into());
            }
            debug_assert!(self.pending_host_yield.is_none());
            frames[frame_index].clear_open_results();
            // Only a hook the callback installed can observe the call.
            if !one && !self.debug_hooks.is_empty() {
                self.observe_native_call(frames, Value::NativeFunction(function), count)?;
            }
            return Ok(true);
        }
        if !one
            && self
                .string_arithmetic_delegate(
                    function,
                    &frames.at(frame_index).registers[start..start + count],
                )?
                .is_some()
        {
            return Ok(false);
        }
        let results = self
            .call_host_in_registers(frames, function, (start, count))
            .map_err(|error| {
                call_argument_error(
                    &frames[frame_index],
                    frame_index.checked_sub(1).map(|i| &frames[i]),
                    error,
                    self.compatibility,
                )
            })?;
        debug_assert!(self.pending_host_yield.is_none());
        debug_assert_ne!(call.c(), 0, "a fixed call has a result count");
        let mut frame = frames.at_mut(frame_index);
        frame.clear_open_results();
        write_fixed_results(frame.registers, call, &results)?;
        self.runtime.recycle_result_vec(results);
        if !one {
            self.observe_native_call(frames, Value::NativeFunction(function), count)?;
        }
        Ok(true)
    }

    /// [`Self::call_host_fixed`] for `return host(...)`, a fixed `TailCall`
    /// of the host callback `function` in the running frame: runs it as a
    /// call with an open result count, as the dispatch loop's tail call of a
    /// native does, and leaves its results as the frame's open results for
    /// the `ReturnOpen` that follows. The caller has checked
    /// [`Self::host_call_in_fast_loop`] and that the registers exist.
    ///
    /// Returns `false`, with nothing done, for a `string` library function
    /// that hands arithmetic on to a metamethod.
    pub(crate) fn call_host_open(
        &mut self,
        frames: &mut CallStack,
        function: NativeFunctionHandle,
        (start, count): (usize, usize),
    ) -> VmResult<bool> {
        debug_assert!(self.host_call_in_fast_loop(function, count));
        let frame_index = frames.len() - 1;
        if self
            .string_arithmetic_delegate(
                function,
                &frames.at(frame_index).registers[start..start + count],
            )?
            .is_some()
        {
            return Ok(false);
        }
        let results = self
            .call_host_in_registers(frames, function, (start, count))
            .map_err(|error| {
                call_argument_error(
                    &frames[frame_index],
                    frame_index.checked_sub(1).map(|i| &frames[i]),
                    error,
                    self.compatibility,
                )
            })?;
        debug_assert!(self.pending_host_yield.is_none());
        let spare = frames.at_mut(frame_index).replace_open_results(results);
        self.runtime.recycle_result_vec(spare);
        self.observe_native_call(frames, Value::NativeFunction(function), count)?;
        Ok(true)
    }

    /// Releases an embedding-owned registration once it can no longer be called.
    /// The embedding must keep a callback's owner alive until that callback returns.
    // The by-value `Value` argument is part of the public embedding API.
    #[allow(clippy::needless_pass_by_value)]
    pub fn unregister_host_function(&mut self, function: Value) -> bool {
        match function {
            Value::NativeFunction(function) => {
                if let Some(slot) = self.host_dispatch.get_mut(function.index() as usize) {
                    *slot = None;
                }
                if let Some(slot) = self.host_direct.get_mut(function.index() as usize) {
                    *slot = None;
                }
                self.host_functions.remove(&function).is_some()
            }
            _ => false,
        }
    }

    /// Calls a native function from code whose Lua stack is `frames`.
    ///
    /// Host callbacks may collect garbage, re-enter Lua, or inspect the stack,
    /// so the suspended frames are parked here, at the host boundary, and
    /// only when `function` really is a host callback: the collector roots
    /// the parked stack and the debugger builds its records from it when it
    /// asks (see `HostSnapshot`). The interpreter loop itself does no
    /// per-instruction work for hosts.
    pub(crate) fn call_native(
        &mut self,
        frames: &mut CallStack,
        function: NativeFunctionHandle,
        arguments: &[Value],
    ) -> Result<Vec<Value>, RuntimeError> {
        self.call_native_above(frames, None, function, arguments)
    }

    /// Like [`Self::call_native`] for a frame that has already left `frames`
    /// but is still being closed, such as a `__close` handler's owner.
    pub(crate) fn call_native_above(
        &mut self,
        frames: &mut CallStack,
        closing: Option<&mut FrameMut<'_>>,
        function: NativeFunctionHandle,
        arguments: &[Value],
    ) -> Result<Vec<Value>, RuntimeError> {
        self.call_native_boundary(frames, closing, function, arguments, false)
    }

    pub(crate) fn call_native_yieldable(
        &mut self,
        frames: &mut CallStack,
        function: NativeFunctionHandle,
        arguments: &[Value],
        allowed: bool,
    ) -> Result<Vec<Value>, RuntimeError> {
        let allowed = allowed && self.frames_allow_yield(frames);
        self.call_native_boundary(frames, None, function, arguments, allowed)
    }

    fn call_native_boundary(
        &mut self,
        frames: &mut CallStack,
        closing: Option<&mut FrameMut<'_>>,
        function: NativeFunctionHandle,
        arguments: &[Value],
        yieldable: bool,
    ) -> Result<Vec<Value>, RuntimeError> {
        let mut table_meta_call = false;
        let (callback, token) =
            if let Some(&Some(host)) = self.host_dispatch.get(function.index() as usize) {
                host
            } else {
                match self.runtime.call_native(function, arguments) {
                    // A `table` function declined a table with metamethods; run
                    // it as a host callback, which may call back into Lua.
                    Err(RuntimeError::Redispatch) => {
                        let Some(token) = self
                            .runtime
                            .direct_native_kind(function)
                            .and_then(crate::vm::table_meta::token)
                        else {
                            return Err(RuntimeError::Redispatch);
                        };
                        // Stock bounds nested C calls (`LUAI_MAXCCALLS`).
                        if self.table_meta_depth >= MAX_TABLE_META_DEPTH {
                            let message = if self.compatibility
                                == crate::compat::CompatibilityProfile::LuaJit21
                            {
                                "stack overflow"
                            } else {
                                "C stack overflow"
                            };
                            return Err(RuntimeError::Raised(Value::from(message)));
                        }
                        table_meta_call = true;
                        (crate::vm::table_meta::callback as HostCallback, token)
                    }
                    result => return result,
                }
            };
        let suppress_hooks = self.frames_suppress_hooks(frames)
            || closing.as_deref().is_some_and(|frame| {
                frame
                    .extras_ref()
                    .is_some_and(|extras| extras.suppress_debug_hooks)
            });
        let mut roots = self
            .spare_roots
            .pop()
            .map_or_else(GcRootBuilder::new, GcRoots::reuse);
        roots.value(&Value::Table(self.thread_environment));
        if let Some(closing) = closing.as_deref() {
            // A frame that is closing stays with its caller, which goes on
            // reading it: snapshot the stack now. Host callback snapshot:
            // frame pcs may be stale, so root full windows.
            Self::root_call_stack(&mut roots, frames, Window::Full);
            roots.values(closing.varargs.iter());
            Self::root_frame_window(
                &mut roots,
                closing.frame,
                closing.registers,
                Window::Full,
                &[],
            );
            Self::root_frame(&mut roots, closing.frame);
            let debug = frames
                .iter()
                .chain(std::iter::once(&*closing.frame))
                .map(SuspendedFrame::capture)
                .collect();
            let mut locals = (0..frames.len())
                .map(|index| {
                    let frame = frames.at(index);
                    crate::vm::host_locals::HostLocalFrame::capture(
                        &frame,
                        frames.open_in(frame.base, frame.top),
                    )
                })
                .collect::<Vec<_>>();
            locals.push(crate::vm::host_locals::HostLocalFrame::capture(
                &closing.as_ref(),
                &[],
            ));
            self.host_snapshots
                .push(crate::vm::host_locals::HostSnapshot::built(
                    debug,
                    locals,
                    self.current_coroutine,
                ));
        }
        // Without a closing frame the stack is lent to a snapshot where it
        // stands, by `run_parked_host`.
        roots.value(&Value::NativeFunction(function));
        roots.values(arguments.iter());
        self.suspended_roots.push(roots.finish());
        self.table_meta_depth += u8::from(table_meta_call);
        let result = self.run_parked_host(
            frames,
            closing,
            function,
            |vm| callback(vm, token, arguments),
            (yieldable, suppress_hooks),
        );
        self.table_meta_depth -= u8::from(table_meta_call);
        if let Some(roots) = self.suspended_roots.pop()
            && roots.capacity() <= SPARE_ROOTS_CAPACITY
            && self.spare_roots.len() < SPARE_ROOTS
        {
            // Emptied before it is pooled, so the pool holds no stale handles
            // that a debugger or a later reader could mistake for roots.
            self.spare_roots.push(roots.reuse().finish());
        }
        result
    }

    /// [`Self::call_native_boundary`] for the fast loop's fixed calls of the
    /// host callback `function`, with the `count` arguments in the running
    /// frame's registers from `start`, which the caller has checked exist:
    /// not yieldable, and nothing to close.
    ///
    /// The callback reads its arguments where they are, in the parked stack,
    /// unless an open upvalue may name one of them, and no suspended roots
    /// are built for the call: the parked stack roots the function and its
    /// arguments as registers of its top frame, and the thread environment
    /// is always a root.
    #[inline]
    fn call_host_in_registers(
        &mut self,
        frames: &mut CallStack,
        function: NativeFunctionHandle,
        (start, count): (usize, usize),
    ) -> Result<Vec<Value>, RuntimeError> {
        let Some(&Some((callback, token))) = self.host_dispatch.get(function.index() as usize)
        else {
            let arguments = frames.at(frames.len() - 1).registers[start..start + count].to_vec();
            return self.call_native_boundary(frames, None, function, &arguments, false);
        };
        let suppress_hooks = self.frames_suppress_hooks(frames);
        let top = frames.at(frames.len() - 1);
        let registers = &top.registers[start..start + count];
        // An open upvalue cell writes its stack slot even while the stack is
        // parked: Lua code the callback runs may assign a captured local
        // passed as an argument and drop the value the callback borrows.
        // Arguments at or above the lowest slot that could be open are
        // copied; every open cell lies below `open_top`.
        let may_be_open = frames.open_top() as usize > top.base as usize + start;
        let held: Option<[Value; HOST_FAST_ARGUMENTS]> = may_be_open
            .then(|| std::array::from_fn(|i| registers.get(i).cloned().unwrap_or(Value::Nil)));
        let in_place: *const [Value] = registers;
        let arguments = match &held {
            Some(held) => &held[..count],
            // SAFETY: the registers live in the value stack's heap buffer,
            // which moved into the snapshot with the stack and does not move
            // while it is parked. No open cell names them, so nothing writes
            // them: the collector and the debugger only read a parked stack
            // (a debugger's local edits go to the snapshot's own copies,
            // applied after the callback), and the stack comes back to
            // `frames` only after the callback has returned.
            #[allow(unsafe_code)]
            None => unsafe { &*in_place },
        };
        self.run_parked_host(
            frames,
            None,
            function,
            |vm| callback(vm, token, arguments),
            (false, suppress_hooks),
        )
    }

    /// [`Self::call_host_in_registers`] through the function's
    /// [`crate::vm::HostDirectCallback`] `direct`, which stores the `wanted`
    /// results of the call in the running frame's registers from `first`
    /// itself. The caller has checked that both register ranges exist.
    #[inline]
    pub(crate) fn call_host_direct(
        &mut self,
        frames: &mut CallStack,
        function: NativeFunctionHandle,
        (direct, token): (crate::vm::HostDirectCallback, u64),
        (start, count): (usize, usize),
        (first, wanted): (usize, usize),
    ) -> Result<(), RuntimeError> {
        let suppress_hooks = self.frames_suppress_hooks(frames);
        let index = frames.len() - 1;
        let may_be_open = {
            let top = &frames[index];
            frames.open_top() as usize > top.base as usize + start
        };
        let top = frames.at_mut(index);
        debug_assert!(start + count <= top.registers.len());
        debug_assert!(first + wanted <= top.registers.len());
        // As in `call_host_in_registers`: arguments an open cell may name
        // are copied, since Lua code the callback runs may assign them.
        let held: Option<[Value; HOST_FAST_ARGUMENTS]> = may_be_open.then(|| {
            std::array::from_fn(|i| {
                top.registers[start..start + count]
                    .get(i)
                    .cloned()
                    .unwrap_or(Value::Nil)
            })
        });
        let registers: *mut Value = top.registers.as_mut_ptr();
        let arguments: *const Value = match &held {
            Some(held) => held.as_ptr(),
            // SAFETY: inside the window, checked by the caller.
            #[allow(unsafe_code)]
            None => unsafe { registers.add(start) },
        };
        // SAFETY: the registers live in the value stack's heap buffer, which
        // stays where it is while the stack is parked, and nothing writes
        // them meanwhile (see `call_host_in_registers`). The callback reads
        // the arguments before it runs anything and stores the results
        // after its last call into the VM, as its contract says, so neither
        // range is in use while the other is written.
        #[allow(unsafe_code)]
        self.run_parked_host_storing(
            frames,
            None,
            function,
            |vm| unsafe { direct(vm, token, arguments, count, registers.add(first), wanted) },
            (false, suppress_hooks),
            first..first + wanted,
        )
    }

    /// Whether a frame of `frames` runs with hooks off, which a host
    /// callback it calls inherits. Only a failed hook marks one, so the walk
    /// is skipped until a hook has been installed.
    #[inline]
    fn frames_suppress_hooks(&self, frames: &CallStack) -> bool {
        (self.debug_hooks_installed || !self.debug_hooks.is_empty())
            && frames.iter().any(|frame| {
                frame
                    .extras_ref()
                    .is_some_and(|extras| extras.suppress_debug_hooks)
            })
    }

    /// Parks the stack for a host callback's duration. The collector roots
    /// it from here, with full windows since frame pcs may be stale, and the
    /// debugger builds its records from it on demand (see `HostSnapshot`).
    ///
    /// The stack is lent, not moved: `frames` stays with the caller, which
    /// does not touch it again before [`Self::run_parked_host`] has popped
    /// the snapshot. That function runs right after this one on every path
    /// and pops the snapshot even when the callback unwinds.
    #[inline]
    fn park_frames(&mut self, frames: &mut CallStack) {
        // SAFETY: see above; the snapshot does not outlive the call that
        // `frames` is suspended in.
        #[allow(unsafe_code)]
        let snapshot =
            unsafe { crate::vm::host_locals::HostSnapshot::parked(frames, self.current_coroutine) };
        self.host_snapshots.push(snapshot);
    }

    /// Runs the host `callback` once the caller has pushed the snapshot of
    /// `frames`, and takes the snapshot back afterwards: the second half of
    /// [`Self::call_native_boundary`].
    #[inline(always)]
    fn run_parked_host<T: Default>(
        &mut self,
        frames: &mut CallStack,
        closing: Option<&mut FrameMut<'_>>,
        function: NativeFunctionHandle,
        callback: impl FnOnce(&mut Vm) -> VmResult<T>,
        (yieldable, suppress_hooks): (bool, bool),
    ) -> Result<T, RuntimeError> {
        self.run_parked_host_storing(
            frames,
            closing,
            function,
            callback,
            (yieldable, suppress_hooks),
            0..0,
        )
    }

    /// [`Self::run_parked_host`] for a `callback` that stores its results in
    /// the registers `stored` of the top frame itself.
    #[inline(always)]
    fn run_parked_host_storing<T: Default>(
        &mut self,
        frames: &mut CallStack,
        closing: Option<&mut FrameMut<'_>>,
        function: NativeFunctionHandle,
        callback: impl FnOnce(&mut Vm) -> VmResult<T>,
        (yieldable, suppress_hooks): (bool, bool),
        stored: std::ops::Range<usize>,
    ) -> Result<T, RuntimeError> {
        // Lent here, after the caller's last use of `frames`: nothing
        // reaches the stack through `frames` again before the snapshot is
        // popped below. A closing frame's caller pushed a built snapshot.
        if closing.is_none() {
            self.park_frames(frames);
        }
        let previous_host_function = self.active_host_function.replace(function);
        if let Some(previous) = previous_host_function {
            self.suspended_host_functions.push(previous);
        }
        self.host_yield_boundaries
            .push((self.current_coroutine, yieldable));
        let previous_hook_depth = self.debug_hook_depth;
        if suppress_hooks {
            self.debug_hook_depth += 1;
        }
        // A callback that panics unwinds through here: the snapshot names
        // the caller's stack by address, so it must not stay behind.
        let result = {
            let parked = Parked(self);
            let result = callback(parked.0);
            // Returned: the snapshot is popped below, after its use.
            std::mem::forget(parked);
            result
        };
        self.debug_hook_depth = previous_hook_depth;
        let result = match result {
            Err(VmError::Yield(values)) => match self.validate_host_yield() {
                Ok(()) => {
                    self.pending_host_yield = Some(values);
                    Ok(T::default())
                }
                Err(error) => Err(error),
            },
            result => result,
        };
        let snapshot = self.host_snapshots.pop().expect("host snapshot exists");
        // A debugger that edited a local of a parked frame edited the
        // snapshot: copy its writes back before any caller writes results
        // or unwinds, including callbacks that fail or yield. Local result
        // assignments then win in the usual way when their destination is
        // itself captured.
        if let Some(locals) = snapshot.built_local_frames() {
            // A callback that failed or yielded stored nothing.
            let stored = if result.is_ok() { stored } else { 0..0 };
            // The two parallel frame stacks are indexed by the same
            // position, and `frames` bounds the loop; iterating either
            // side would change which length bounds the synchronization.
            #[allow(clippy::needless_range_loop)]
            for index in 0..frames.len() {
                if index + 1 == frames.len() {
                    locals[index].apply_outside(&mut frames.at_mut(index), stored.clone());
                } else {
                    locals[index].apply(&mut frames.at_mut(index));
                }
            }
            if let Some(frame) = closing {
                locals[frames.len()].apply(frame);
            }
        }
        self.host_yield_boundaries.pop();
        self.active_host_function = previous_host_function;
        if previous_host_function.is_some() {
            self.suspended_host_functions.pop();
        }
        result.map_err(host_runtime_error)
    }

    /// Calls a protected error handler while keeping debug hooks disabled.
    /// The caller uses this only when the original failure came from a hook.
    /// Hook state is restored on both success and failure.
    ///
    /// # Errors
    ///
    /// Returns whatever error calling `function` raises, with hook dispatch
    /// suppressed for the duration of the call.
    pub fn call_value_without_hooks(
        &mut self,
        function: Value,
        arguments: Vec<Value>,
    ) -> VmResult<Vec<Value>> {
        let previous = self.debug_hook_depth;
        self.debug_hook_depth += 1;
        let result = self.call_value(function, arguments);
        self.debug_hook_depth = previous;
        result
    }

    /// Calls a Lua or native value. A host callback may call this again on
    /// the same VM while its outer Lua invocation is suspended.
    ///
    /// # Errors
    ///
    /// Errors when the value is not callable, when its `__call` chain is
    /// longer than `MAX_INDEX_HOPS`, when a handle is stale, or with whatever
    /// error the callee raises.
    pub fn call_value(
        &mut self,
        mut function: Value,
        mut arguments: Vec<Value>,
    ) -> VmResult<Vec<Value>> {
        for _ in 0..MAX_INDEX_HOPS {
            match function {
                Value::NativeFunction(handle) => {
                    return Ok(self.call_native(&mut CallStack::new(), handle, &arguments)?);
                }
                Value::Closure(handle) => {
                    if let Some(parts) = self.runtime.closure_call_parts(handle)
                        && !parts.chunk.is_vararg
                        && parts.chunk.arg_register.is_none()
                    {
                        // A fixed-parameter body is pushed as a Lua call
                        // pushes its callee: no detached frame, and the
                        // record borrows the closure's chunk and captures.
                        let mut frames = self.spare_host_stacks.pop().unwrap_or_default();
                        // SAFETY: `parts` are `handle`'s own, and the record
                        // roots `handle` while it is live.
                        #[allow(unsafe_code)]
                        unsafe {
                            frames.push_host_activation(
                                parts.chunk,
                                parts.cells,
                                parts.environment,
                                handle,
                                arguments,
                            );
                        }
                        return self.run_pushed(frames);
                    }
                    let body = self.closure_body(handle)?;
                    let frame = self.make_frame(
                        body.chunk(),
                        arguments,
                        body.shared_cells(),
                        None,
                        None,
                        body.environment(),
                        Some(handle),
                    )?;
                    return self.run(frame);
                }
                Value::BuiltinFunction(builtin) => {
                    if builtin == crate::runtime::BuiltinFunction::DebugGetInfo
                        && self.active_host_function.is_some()
                    {
                        return self.debug_get_info_from_host(&arguments);
                    }
                    let direct = self.call_builtin(builtin, &arguments);
                    if !matches!(direct, Err(VmError::InvalidDispatch(_))) {
                        return direct;
                    }
                    let chunk = builtin_call_chunk(arguments.len())?;
                    arguments.insert(0, Value::BuiltinFunction(builtin));
                    let frame = self.make_frame(
                        chunk,
                        arguments,
                        Vec::<crate::runtime::CellHandle>::new(),
                        None,
                        None,
                        self.thread_environment,
                        None,
                    )?;
                    return self.run(frame);
                }
                Value::GmatchIterator(iterator) => return self.call_gmatch_iterator(iterator),
                Value::CoroutineWrapper(thread) => {
                    let mut results = self.resume_wrapped_coroutine_mode(
                        &mut CallStack::new(),
                        thread,
                        arguments,
                        true,
                    )?;
                    if !matches!(results.first(), Some(Value::Boolean(true))) {
                        return Err(raised_value(results.get(1).cloned().unwrap_or(Value::Nil)));
                    }
                    results.remove(0);
                    return Ok(results);
                }
                Value::Table(_) | Value::Userdata(_) => {
                    let Some(metatable) = self.runtime.raw_metatable(&function)? else {
                        return Err(call_type_error_anonymous(&function).into());
                    };
                    let handler = self
                        .runtime
                        .table_get_string_ref(metatable, call_metamethod_key())
                        .cloned()
                        .unwrap_or(Value::Nil);
                    if matches!(handler, Value::Nil) {
                        return Err(call_type_error_anonymous(&function).into());
                    }
                    arguments.insert(0, function);
                    function = handler;
                }
                _ => return Err(call_type_error_anonymous(&function).into()),
            }
        }
        Err(RuntimeError::message("metatable chain too long").into())
    }

    /// Installs a validated chunk as a callable Lua closure.
    ///
    /// # Errors
    ///
    /// Errors when `chunk` fails validation, when it uses unsupported upvalue
    /// descriptors, or when the closure cannot be installed.
    pub fn load_chunk(&mut self, chunk: &Chunk) -> VmResult<Value> {
        chunk.validate().map_err(VmError::InvalidChunk)?;
        Ok(Value::Closure(self.install_loaded_closure(chunk.clone())?))
    }

    /// [`Self::load_chunk`] for a chunk the caller no longer needs, which
    /// saves copying it.
    ///
    /// # Errors
    ///
    /// The same as [`Self::load_chunk`].
    pub fn load_owned_chunk(&mut self, chunk: Chunk) -> VmResult<Value> {
        chunk.validate().map_err(VmError::InvalidChunk)?;
        Ok(Value::Closure(self.install_loaded_closure(chunk)?))
    }

    /// Reads a field using Lua's `__index` chain.
    ///
    /// # Errors
    ///
    /// Errors when the receiver is not indexable, when its `__index` chain is
    /// longer than `MAX_INDEX_HOPS`, or with whatever a table read or an
    /// `__index` call raises.
    pub fn get_value(&mut self, mut receiver: Value, key: Value) -> VmResult<Value> {
        for _ in 0..MAX_INDEX_HOPS {
            if let Value::Table(table) = receiver {
                let value = self.runtime.table_get(table, &key)?;
                if !matches!(value, Value::Nil) {
                    return Ok(value);
                }
            }
            let Some(handler) = self.index_metamethod(&receiver, false)? else {
                return if matches!(receiver, Value::Table(_)) {
                    Ok(Value::Nil)
                } else {
                    Err(RuntimeError::message(format!(
                        "attempt to index a {} value",
                        receiver.type_name()
                    ))
                    .into())
                };
            };
            if matches!(handler, Value::Table(_)) {
                receiver = handler;
            } else {
                return Ok(self
                    .call_value(handler, vec![receiver, key])?
                    .into_iter()
                    .next()
                    .unwrap_or(Value::Nil));
            }
        }
        Err(RuntimeError::message("metatable chain too long").into())
    }

    /// Writes a field using Lua's `__newindex` chain.
    ///
    /// # Errors
    ///
    /// Errors when the receiver is not indexable, when its `__newindex` chain
    /// is longer than `MAX_INDEX_HOPS`, or with whatever a table write or a
    /// `__newindex` call raises.
    pub fn set_value(&mut self, mut receiver: Value, key: Value, value: Value) -> VmResult<()> {
        for _ in 0..MAX_INDEX_HOPS {
            if let Value::Table(table) = receiver {
                let current = self.runtime.table_get(table, &key)?;
                if !matches!(current, Value::Nil)
                    || self.index_metamethod(&receiver, true)?.is_none()
                {
                    return Ok(self.runtime.table_set(table, &key, value)?);
                }
            }
            let Some(handler) = self.index_metamethod(&receiver, true)? else {
                return Err(RuntimeError::message(format!(
                    "attempt to index a {} value",
                    receiver.type_name()
                ))
                .into());
            };
            if matches!(handler, Value::Table(_)) {
                receiver = handler;
            } else {
                self.call_value(handler, vec![receiver, key, value])?;
                return Ok(());
            }
        }
        Err(RuntimeError::message("metatable chain too long").into())
    }
}

/// Pops the snapshot of a parked stack when a host callback unwinds: the
/// snapshot names the caller's stack by address and must not stay behind.
struct Parked<'a>(&'a mut Vm);

impl Drop for Parked<'_> {
    fn drop(&mut self) {
        self.0.host_snapshots.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::{Constant, Instruction};
    use crate::runtime::{BuiltinFunction, TableHandle};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn yield_arguments(vm: &mut Vm, _: u64, arguments: &[Value]) -> VmResult<Vec<Value>> {
        Err(vm.request_host_yield(arguments.to_vec()))
    }

    #[test]
    fn host_native_entry_yields_and_returns_resume_arguments() {
        let mut vm = Vm::with_profile(crate::compat::CompatibilityProfile::Lua51);
        let callback = vm
            .register_host_function(yield_arguments, 0, Vec::new())
            .unwrap();
        let thread = vm.create_host_coroutine().unwrap();
        assert_eq!(
            vm.host_coroutine_status(thread).unwrap(),
            crate::vm::HostCoroutineStatus::New
        );
        assert_eq!(
            vm.start_host_coroutine(thread, callback, vec![Value::from("yielded")])
                .unwrap(),
            vec![Value::Boolean(true), Value::from("yielded")]
        );
        assert_eq!(
            vm.host_coroutine_status(thread).unwrap(),
            crate::vm::HostCoroutineStatus::Suspended
        );
        assert_eq!(
            vm.resume_host_coroutine(thread, vec![Value::from("resumed"), Value::Number(7.0)])
                .unwrap(),
            vec![
                Value::Boolean(true),
                Value::from("resumed"),
                Value::Number(7.0)
            ]
        );
        assert_eq!(
            vm.host_coroutine_status(thread).unwrap(),
            crate::vm::HostCoroutineStatus::Dead
        );
        assert_eq!(
            vm.resume_host_coroutine(thread, Vec::new()).unwrap()[0],
            Value::Boolean(false)
        );
    }

    #[test]
    fn host_callback_yield_resumes_the_lua_call_instruction() {
        let mut vm = Vm::with_profile(crate::compat::CompatibilityProfile::Lua51);
        let callback = vm
            .register_host_function(yield_arguments, 0, Vec::new())
            .unwrap();
        let wrapper = vm.load_chunk(&builtin_call_chunk(2).unwrap()).unwrap();
        let thread = vm.create_host_coroutine().unwrap();
        assert_eq!(
            vm.start_host_coroutine(
                thread,
                wrapper,
                vec![callback, Value::Number(1.0), Value::Number(2.0)]
            )
            .unwrap(),
            vec![Value::Boolean(true), Value::Number(1.0), Value::Number(2.0)]
        );
        assert_eq!(
            vm.resume_host_coroutine(thread, vec![Value::Number(3.0)])
                .unwrap(),
            vec![Value::Boolean(true), Value::Number(3.0)]
        );
    }

    #[test]
    fn host_callback_cannot_yield_through_protected_native_call() {
        let mut vm = Vm::with_profile(crate::compat::CompatibilityProfile::Lua51);
        let callback = vm
            .register_host_function(yield_arguments, 0, Vec::new())
            .unwrap();
        let thread = vm.create_host_coroutine().unwrap();
        let result = vm
            .start_host_coroutine(
                thread,
                Value::BuiltinFunction(BuiltinFunction::ProtectedCall),
                vec![callback],
            )
            .unwrap();
        assert_eq!(result[0], Value::Boolean(true));
        assert_eq!(result[1], Value::Boolean(false));
        assert!(
            matches!(&result[2], Value::String(message) if message.to_lossy().contains("yield across metamethod/C-call boundary"))
        );
        assert_eq!(
            vm.host_coroutine_status(thread).unwrap(),
            crate::vm::HostCoroutineStatus::Dead
        );
    }

    fn reenter_argument(vm: &mut Vm, _: u64, arguments: &[Value]) -> VmResult<Vec<Value>> {
        vm.call_value(arguments[0].clone(), Vec::new())
    }

    #[test]
    fn lua_yield_cannot_cross_a_reentrant_host_call() {
        let mut vm = Vm::with_profile(crate::compat::CompatibilityProfile::Lua51);
        let callback = vm
            .register_host_function(reenter_argument, 0, Vec::new())
            .unwrap();
        let inner = vm
            .load_chunk(&Chunk::new(
                1,
                vec![Constant::String(crate::string::LuaString::from(
                    "yield_function",
                ))],
                vec![
                    Instruction::get_global(0, 0),
                    Instruction::call(0, 0, 0),
                    Instruction::return_open(0, 0),
                ],
            ))
            .unwrap();
        vm.runtime_mut().set_global(
            "yield_function",
            Value::BuiltinFunction(BuiltinFunction::CoroutineYield),
        );
        let wrapper = vm.load_chunk(&builtin_call_chunk(1).unwrap()).unwrap();
        let thread = vm.create_host_coroutine().unwrap();
        let result = vm
            .start_host_coroutine(thread, wrapper, vec![callback, inner])
            .unwrap();
        assert_eq!(result[0], Value::Boolean(false));
        assert!(vm.host_coroutine_failed(thread).unwrap());
        assert_eq!(vm.current_coroutine(), None);
        vm.resume_host_coroutine(thread, Vec::new()).unwrap();
        assert!(vm.host_coroutine_failed(thread).unwrap());
    }

    #[test]
    fn native_entry_accepts_more_arguments_than_bytecode_registers() {
        let mut vm = Vm::with_profile(crate::compat::CompatibilityProfile::Lua51);
        let callback = vm
            .register_host_function(yield_arguments, 0, Vec::new())
            .unwrap();
        let thread = vm.create_host_coroutine().unwrap();
        let values = vec![Value::Number(1.0); 300];
        let yielded = vm
            .start_host_coroutine(thread, callback, values.clone())
            .unwrap();
        assert_eq!(&yielded[1..], values.as_slice());
        let returned = vm.resume_host_coroutine(thread, values.clone()).unwrap();
        assert_eq!(&returned[1..], values.as_slice());
    }

    static FINALIZER_CALLS: AtomicUsize = AtomicUsize::new(0);

    fn count_finalizer(_: &mut Vm, _: u64, _: &[Value]) -> VmResult<Vec<Value>> {
        FINALIZER_CALLS.fetch_add(1, Ordering::SeqCst);
        Ok(Vec::new())
    }

    fn c_debug_wrapper(vm: &mut Vm, _: u64, _: &[Value]) -> VmResult<Vec<Value>> {
        vm.call_value(
            Value::BuiltinFunction(BuiltinFunction::DebugGetInfo),
            vec![Value::Number(2.0), Value::from("n")],
        )
    }

    fn reenter_and_collect(vm: &mut Vm, token: u64, _: &[Value]) -> VmResult<Vec<Value>> {
        assert_eq!(vm.host_stack_info(0).unwrap().what, "C");
        assert_eq!(vm.host_stack_info(1).unwrap().what, "main");
        assert!(vm.host_stack_info(2).is_none());
        vm.runtime_mut().set_global("resource", Value::Nil);
        let roots = vm.gc_roots_with_coroutines(&mut CallStack::new(), &[]);
        vm.runtime_mut().collect_full(roots);
        let captured = TableHandle::from_native_identity(token);
        assert_eq!(
            vm.runtime().table_get(captured, &Value::from("marker"))?,
            Value::Number(17.0)
        );
        vm.call_value(vm.runtime().get_global("inner"), Vec::new())
    }

    #[test]
    fn host_callback_reenters_lua_and_roots_suspended_frame_and_captures() {
        let mut vm = Vm::new();
        let outer_table = vm.runtime_mut().create_table().unwrap();
        let captured = vm.runtime_mut().create_table().unwrap();
        vm.runtime_mut()
            .table_set(captured, &Value::from("marker"), Value::Number(17.0))
            .unwrap();
        vm.runtime_mut()
            .set_global("resource", Value::Table(outer_table));
        let callback = vm
            .register_host_function(
                reenter_and_collect,
                captured.native_identity(),
                vec![Value::Table(captured)],
            )
            .unwrap();
        vm.runtime_mut().set_global("host", callback);
        let inner = Chunk::new(
            1,
            vec![Constant::Number(9.0)],
            vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
        );
        let inner = vm.load_chunk(&inner).unwrap();
        vm.runtime_mut().set_global("inner", inner);
        let outer = Chunk::new(
            2,
            vec![
                Constant::String("resource".into()),
                Constant::String("host".into()),
            ],
            vec![
                Instruction::get_global(0, 0),
                Instruction::get_global(1, 1),
                Instruction::call(1, 0, 1),
                Instruction::return_values(0, 2),
            ],
        );
        assert_eq!(
            vm.execute(&outer).unwrap(),
            vec![Value::Table(outer_table), Value::Number(9.0)]
        );
    }

    #[test]
    fn direct_stateful_builtin_call_uses_vm_dispatch() {
        let mut vm = Vm::new();
        let answer = vm
            .load_chunk(&Chunk::new(
                1,
                vec![Constant::Number(42.0)],
                vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
            ))
            .unwrap();
        assert_eq!(
            vm.call_value(
                Value::BuiltinFunction(BuiltinFunction::ProtectedCall),
                vec![answer]
            )
            .unwrap(),
            vec![Value::Boolean(true), Value::Number(42.0)],
        );
        assert_eq!(
            vm.call_value(
                Value::BuiltinFunction(BuiltinFunction::CollectGarbage),
                vec![Value::from("collect")]
            )
            .unwrap(),
            vec![Value::Number(0.0)],
        );
    }

    #[test]
    fn explicit_collect_drains_userdata_finalizers() {
        FINALIZER_CALLS.store(0, Ordering::SeqCst);
        let mut vm = Vm::new();
        let userdata = vm.runtime_mut().create_userdata().unwrap();
        let metatable = vm.runtime_mut().create_table().unwrap();
        let handler = vm
            .register_host_function(count_finalizer, 0, Vec::new())
            .unwrap();
        vm.runtime_mut()
            .table_set(metatable, &Value::from("__gc"), handler)
            .unwrap();
        vm.runtime_mut()
            .raw_set_metatable(&Value::Userdata(userdata), Some(metatable))
            .unwrap();
        vm.collect_full_with_values(&[]);
        assert_eq!(FINALIZER_CALLS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn debug_getinfo_called_through_host_wrapper_sees_lua_caller() {
        let mut vm = Vm::new();
        let wrapper = vm
            .register_host_function(c_debug_wrapper, 0, Vec::new())
            .unwrap();
        vm.runtime_mut().set_global("wrapper", wrapper);
        let chunk = Chunk::new(
            1,
            vec![Constant::String("wrapper".into())],
            vec![
                Instruction::get_global(0, 0),
                Instruction::call(0, 0, 1),
                Instruction::return_(0),
            ],
        );
        let result = vm.execute(&chunk).unwrap();
        let [Value::Table(info)] = result.as_slice() else {
            panic!("debug.getinfo must return its caller table");
        };
        assert_eq!(
            vm.runtime()
                .table_get(*info, &Value::from("namewhat"))
                .unwrap(),
            Value::from(""),
        );
    }
}
