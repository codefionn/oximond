//! Host integration: globals, standard output streams, and native function registration.

use crate::runtime::error::{RuntimeError, RuntimeResult};
use crate::runtime::files::{FileBuffering, FileStorage, RuntimeFile};
use crate::runtime::helpers::host_output_io_error;
use crate::runtime::metatable::{SharedMetatable, shared_metatable_kind};
use crate::runtime::runtime::Runtime;
use crate::runtime::table_storage::key::{integer_key_from_number, integer_to_number};
use crate::runtime::table_storage::{Table, numbers_equal};
use crate::runtime::value::{
    DirectNativeKind, FileHandle, NativeCallKind, NativeCallback, NativeEntry,
    NativeFunctionHandle, NativeNumericBinary, NativeNumericUnary, NativePairCallback, NativePlan,
    NativeScalarCallback, NativeWideBinary, TableHandle, Value,
};
use crate::string::LuaString;
use std::io::Write;
use std::sync::Arc;

/// The names the installers gave natives, kept compactly: every name in one
/// buffer, ended by a NUL, and for each the handle index and where it starts.
/// Startup registers a couple of hundred natives, so a box per name would
/// show in every script's heap. Names are looked up when a recording aborts,
/// never while code runs.
#[derive(Default)]
pub(crate) struct NativeNames {
    text: String,
    entries: Vec<(u32, u32)>,
}

impl NativeNames {
    fn add(&mut self, function: u32, name: &str) {
        self.add_qualified(function, None, name);
    }

    /// Adds `library.name`, or `name` alone, without building the joined
    /// text first.
    fn add_qualified(&mut self, function: u32, library: Option<&str>, name: &str) {
        debug_assert!(!name.contains('\0'));
        if let Ok(start) = u32::try_from(self.text.len()) {
            self.entries.push((function, start));
            if let Some(library) = library {
                self.text.push_str(library);
                self.text.push('.');
            }
            self.text.push_str(name);
            self.text.push('\0');
        }
    }

    /// The first name `function` was given.
    fn get(&self, function: u32) -> Option<&str> {
        let (_, start) = self.entries.iter().find(|(known, _)| *known == function)?;
        let rest = &self.text[*start as usize..];
        Some(rest.split('\0').next().unwrap_or(rest))
    }
}

impl Runtime {
    /// Replaces the host warning callback. `None` restores standard output.
    pub fn set_warning_handler(&mut self, handler: Option<Box<crate::runtime::WarningHandler>>) {
        self.warning_handler = handler;
        self.warning_buffer.clear();
        self.warning_continues = false;
    }

    #[must_use]
    pub fn has_warning_handler(&self) -> bool {
        self.warning_handler.is_some()
    }

    /// Emits one warning fragment to the host callback or standard warning
    /// output, as `lauxlib`'s default warning function does. A message that
    /// starts with `@` and is not continued is a control message: `@on` and
    /// `@off` switch warnings, others do nothing. While warnings are on only
    /// the first piece of a message can be one; while they are off every
    /// piece is checked, so the last piece of `warn('x', '@on')` turns them
    /// on.
    ///
    /// # Errors
    /// Returns the callback's error or an output failure.
    pub fn emit_warning(&mut self, bytes: &[u8], continued: bool) -> RuntimeResult<()> {
        if let Some(handler) = &mut self.warning_handler {
            return handler(bytes, continued);
        }
        let starts = !self.warning_continues;
        self.warning_continues = continued;
        if (starts || !self.warnings_enabled)
            && !continued
            && let Some(control) = bytes.strip_prefix(b"@")
        {
            match control {
                b"on" => self.warnings_enabled = true,
                b"off" => self.warnings_enabled = false,
                _ => {}
            }
            return Ok(());
        }
        if !self.warnings_enabled {
            return Ok(());
        }
        self.warning_buffer.extend_from_slice(bytes);
        if !continued {
            let message = std::mem::take(&mut self.warning_buffer);
            self.write_error_output(b"Lua warning: ")?;
            self.write_error_output(&message)?;
            self.write_error_output(b"\n")?;
        }
        Ok(())
    }

    /// Reserves a native handle for a callback owned by the VM.
    ///
    /// # Errors
    ///
    /// Returns an error when more natives are registered than a handle index
    /// can name.
    pub fn register_host_native(&mut self, token: u64) -> RuntimeResult<NativeFunctionHandle> {
        let index = u32::try_from(self.natives.len())
            .map_err(|_| RuntimeError::message("native function handle space exhausted"))?;
        self.natives.push(NativeEntry::Host(token));
        Ok(NativeFunctionHandle(index))
    }

    /// Returns the host token if this handle belongs to a VM callback.
    #[must_use]
    pub fn host_native_token(&self, function: NativeFunctionHandle) -> Option<u64> {
        match self.natives.get(function.index() as usize)? {
            NativeEntry::Host(token) => Some(*token),
            _ => None,
        }
    }

    /// Names a native by its library-qualified name (`math.floor`,
    /// `select`), for the statistics and messages that tell natives apart. A
    /// native keeps the first name it is given.
    pub fn set_native_name(&mut self, function: NativeFunctionHandle, name: &str) {
        self.native_names.add(function.0, name);
    }

    /// Whether any object is registered for finalization or awaits its
    /// finalizer. Closing the state would run those, so a host cannot skip
    /// the close.
    #[must_use]
    pub fn has_finalizers(&self) -> bool {
        !self.finalizer_registry.is_empty()
            || !self.pending_finalizers.is_empty()
            || !self.table_finalizers.is_empty()
    }

    /// Whether dropping this runtime would do more than free memory: a file
    /// opened by the script may hold buffered writes that the drop flushes,
    /// and a pipe has a child process the drop waits for. A host that is
    /// about to exit can skip the drop when this is false.
    #[must_use]
    pub fn drop_has_effects(&self) -> bool {
        self.files.live().any(|(_, file)| {
            matches!(
                file.storage,
                FileStorage::Named { .. }
                    | FileStorage::PipeRead { .. }
                    | FileStorage::PipeWrite { .. }
            )
        })
    }

    /// Names a native `library.name`, as `set_native_name` would with the
    /// joined text. Startup names about eighty natives this way, and the
    /// joined strings were a `format!` allocation each.
    pub fn set_native_name_in_library(
        &mut self,
        function: NativeFunctionHandle,
        library: &str,
        name: &str,
    ) {
        self.native_names
            .add_qualified(function.0, Some(library), name);
    }

    /// The library-qualified name an installer gave `function`, if it did.
    #[must_use]
    pub fn native_name(&self, function: NativeFunctionHandle) -> Option<&str> {
        self.native_names.get(function.0)
    }

    /// Says that `table` is the library called `name`, so natives stored in
    /// it are named `name.field`.
    pub fn set_library_name(&mut self, table: TableHandle, name: &'static str) {
        self.library_names.push((table, name));
    }

    /// The name of the library `table` was declared to be.
    #[must_use]
    pub fn library_name(&self, table: TableHandle) -> Option<&'static str> {
        self.library_names
            .iter()
            .find_map(|(known, name)| (*known == table).then_some(*name))
    }

    /// Records the exact installed handle of a recognized library callback.
    pub fn set_direct_native_kind(
        &mut self,
        function: NativeFunctionHandle,
        kind: DirectNativeKind,
    ) {
        debug_assert_ne!(function.0, crate::runtime::runtime::NO_DIRECT_NATIVE);
        let previous = self.direct_natives[kind as usize];
        self.direct_natives[kind as usize] = function.0;
        self.index_direct_native(previous);
        self.index_direct_native(function.0);
    }

    /// Recomputes the reverse entry of raw handle `index`: the first kind
    /// recorded for it, as a scan of `direct_natives` in declaration order
    /// finds it.
    fn index_direct_native(&mut self, index: u32) {
        use crate::runtime::runtime::{NO_DIRECT_NATIVE, NO_DIRECT_NATIVE_KIND};
        if index == NO_DIRECT_NATIVE {
            return;
        }
        let kind = DirectNativeKind::ALL
            .into_iter()
            .find(|&kind| self.direct_natives[kind as usize] == index)
            .map_or(NO_DIRECT_NATIVE_KIND, |kind| kind as u8);
        let slot = index as usize;
        if slot >= self.direct_native_kinds.len() {
            if kind == NO_DIRECT_NATIVE_KIND {
                return;
            }
            self.direct_native_kinds
                .resize(slot + 1, NO_DIRECT_NATIVE_KIND);
        }
        self.direct_native_kinds[slot] = kind;
    }

    /// Gives a `math.log` native its pure form for a number and a base,
    /// which a trace calls in place of the interpreter's.
    pub fn set_native_log_base(
        &mut self,
        function: NativeFunctionHandle,
        binary: NativeNumericBinary,
    ) {
        self.native_log_bases
            .retain(|(known, _)| *known != function.0);
        self.native_log_bases.push((function.0, binary));
    }

    /// The pure number-and-base form of a `math.log` native, if it has one.
    #[must_use]
    pub fn native_log_base(&self, function: NativeFunctionHandle) -> Option<NativeNumericBinary> {
        self.native_log_bases
            .iter()
            .find_map(|(known, binary)| (*known == function.0).then_some(*binary))
    }

    /// Gives `function` a probe (see [`oximond::runtime::NativeProbe`]).
    pub fn set_native_probe(
        &mut self,
        function: NativeFunctionHandle,
        probe: crate::runtime::NativeProbe,
    ) {
        self.native_probes.retain(|(known, _)| *known != function.0);
        self.native_probes.push((function.0, probe));
    }

    /// What `function` answers for `args`, if it has a probe and the probe
    /// knows the call: nothing runs and nothing changes.
    #[must_use]
    pub fn probe_native(&self, function: NativeFunctionHandle, args: &[Value]) -> Option<Value> {
        let (_, probe) = self
            .native_probes
            .iter()
            .find(|(known, _)| *known == function.0)?;
        probe(self, args)
    }

    /// Gives `function` a probe for all of its results (see
    /// [`oximond::runtime::NativeResultsProbe`]).
    pub fn set_native_results_probe(
        &mut self,
        function: NativeFunctionHandle,
        probe: crate::runtime::NativeResultsProbe,
    ) {
        self.native_results_probes
            .retain(|(known, _)| *known != function.0);
        self.native_results_probes.push((function.0, probe));
    }

    /// Every result `function` gives for `args`, if it has a results probe
    /// and the call raises no error: nothing runs and nothing changes.
    #[must_use]
    pub fn probe_native_results(
        &self,
        function: NativeFunctionHandle,
        args: &[Value],
    ) -> Option<Vec<Value>> {
        let (_, probe) = self
            .native_results_probes
            .iter()
            .find(|(known, _)| *known == function.0)?;
        probe(self, args)
    }

    /// Gives `function` a probe for all of its results that allocates no
    /// list (see [`oximond::runtime::NativeFixedResultsProbe`]).
    pub fn set_native_fixed_results_probe(
        &mut self,
        function: NativeFunctionHandle,
        probe: crate::runtime::NativeFixedResultsProbe,
    ) {
        self.native_fixed_results_probes
            .retain(|(known, _)| *known != function.0);
        self.native_fixed_results_probes.push((function.0, probe));
    }

    /// [`Runtime::probe_native_results`] into the empty `results`: through
    /// the fixed probe of `function` when it has one, else through its
    /// results probe. `false` also when there are too many results.
    #[must_use]
    pub fn probe_native_results_fixed(
        &self,
        function: NativeFunctionHandle,
        args: &[Value],
        results: &mut crate::runtime::NativeResults,
    ) -> bool {
        if let Some((_, probe)) = self
            .native_fixed_results_probes
            .iter()
            .find(|(known, _)| *known == function.0)
        {
            return probe(self, args, results);
        }
        let Some(values) = self.probe_native_results(function, args) else {
            return false;
        };
        values.into_iter().all(|value| results.push(value).is_ok())
    }

    /// Identifies a recognized library callback by its runtime-owned handle.
    #[must_use]
    pub fn direct_native_kind(&self, function: NativeFunctionHandle) -> Option<DirectNativeKind> {
        let kind = *self.direct_native_kinds.get(function.0 as usize)?;
        DirectNativeKind::ALL.get(usize::from(kind)).copied()
    }

    /// Whether `function` is the installed handle of the recognized library
    /// callback `kind`: one comparison, for the interpreter's native call.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn is_direct_native(&self, function: NativeFunctionHandle, kind: DirectNativeKind) -> bool {
        function.0 == self.direct_natives[kind as usize]
    }

    /// The installed handle recorded for a recognized library callback.
    #[must_use]
    pub const fn direct_native(&self, kind: DirectNativeKind) -> Option<NativeFunctionHandle> {
        match self.direct_natives[kind as usize] {
            crate::runtime::runtime::NO_DIRECT_NATIVE => None,
            index => Some(NativeFunctionHandle(index)),
        }
    }

    #[must_use]
    pub fn get_global(&self, name: impl AsRef<[u8]>) -> Value {
        self.table_get(
            self.global_environment,
            &Value::String(crate::string::shared_key(name.as_ref())),
        )
        .unwrap_or(Value::Nil)
    }
    pub fn set_global(&mut self, name: impl AsRef<[u8]>, value: Value) {
        let Ok(()) = self.table_set(
            self.global_environment,
            &Value::String(LuaString::from(name.as_ref())),
            value,
        ) else {
            return;
        };
    }

    /// Sends bytes written by native functions to the configured host writer,
    /// or appends them to the host-owned output buffer.
    ///
    /// # Errors
    ///
    /// Returns an error when the capture buffer cannot grow or the host writer
    /// cannot accept output.
    pub fn write_output(&mut self, bytes: &[u8]) -> RuntimeResult<()> {
        let flush = match self.output_buffering {
            FileBuffering::No => true,
            FileBuffering::Full(_) => false,
            FileBuffering::Line(_) => bytes.contains(&b'\n'),
        };
        if let Some(writer) = &mut self.output_writer {
            writer
                .write_all(bytes)
                .map_err(|error| host_output_io_error(&error))?;
            if flush {
                writer
                    .flush()
                    .map_err(|error| host_output_io_error(&error))?;
            }
            return Ok(());
        }
        self.output
            .try_reserve(bytes.len())
            .map_err(|_| RuntimeError::message("runtime output is too large"))?;
        self.output.extend_from_slice(bytes);
        Ok(())
    }

    /// Removes and returns bytes emitted by native functions.
    #[must_use]
    pub fn take_output(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.output)
    }

    /// Routes subsequent standard-output writes to an owned host writer.
    ///
    /// A runtime captures output in memory until this method is called. The
    /// writer is retained by the runtime and receives writes as they occur.
    pub fn set_output_writer<W: Write + 'static>(&mut self, writer: W) {
        self.output_writer = Some(Box::new(writer));
    }

    /// Flushes the configured standard-output writer.
    ///
    /// Has no effect while standard output is being captured in memory.
    ///
    /// # Errors
    ///
    /// Returns the host writer's flush error.
    pub fn flush_output(&mut self) -> RuntimeResult<()> {
        match &mut self.output_writer {
            Some(writer) => writer.flush().map_err(|error| host_output_io_error(&error)),
            None => Ok(()),
        }
    }

    /// Sends bytes to the configured standard-error writer, or appends them
    /// to the runtime error sink.
    ///
    /// # Errors
    ///
    /// Returns an error when the capture buffer cannot grow or the host writer
    /// cannot accept output.
    pub fn write_error_output(&mut self, bytes: &[u8]) -> RuntimeResult<()> {
        let flush = match self.error_output_buffering {
            FileBuffering::No => true,
            FileBuffering::Full(_) => false,
            FileBuffering::Line(_) => bytes.contains(&b'\n'),
        };
        if let Some(writer) = &mut self.error_output_writer {
            writer
                .write_all(bytes)
                .map_err(|error| host_output_io_error(&error))?;
            if flush {
                writer
                    .flush()
                    .map_err(|error| host_output_io_error(&error))?;
            }
            return Ok(());
        }
        self.error_output
            .try_reserve(bytes.len())
            .map_err(|_| RuntimeError::message("runtime error output is too large"))?;
        self.error_output.extend_from_slice(bytes);
        Ok(())
    }

    /// Removes and returns bytes emitted to the runtime error sink.
    #[must_use]
    pub fn take_error_output(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.error_output)
    }

    /// Routes subsequent standard-error writes to an owned host writer.
    pub fn set_error_output_writer<W: Write + 'static>(&mut self, writer: W) {
        self.error_output_writer = Some(Box::new(writer));
    }

    /// Flushes the configured standard-error writer.
    ///
    /// Has no effect while standard error is being captured in memory.
    ///
    /// # Errors
    ///
    /// Returns the host writer's flush error.
    pub fn flush_error_output(&mut self) -> RuntimeResult<()> {
        match &mut self.error_output_writer {
            Some(writer) => writer.flush().map_err(|error| host_output_io_error(&error)),
            None => Ok(()),
        }
    }

    /// Registers a native function and returns its stable handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime has exhausted its native-function
    /// handle space.
    pub fn register_native(
        &mut self,
        callback: NativeCallback,
    ) -> RuntimeResult<NativeFunctionHandle> {
        let index = u32::try_from(self.natives.len())
            .map_err(|_| RuntimeError::message("native function handle space exhausted"))?;
        self.natives.push(NativeEntry::Multiple(callback));
        Ok(NativeFunctionHandle(index))
    }

    /// Registers a multi-result native with a fast form for the calls that
    /// return at most two values. The fast form must return what the
    /// callback would, or decline with `Ok(None)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime has exhausted its native-function
    /// handle space.
    pub fn register_native_paired(
        &mut self,
        callback: NativeCallback,
        paired: NativePairCallback,
    ) -> RuntimeResult<NativeFunctionHandle> {
        let index = u32::try_from(self.natives.len())
            .map_err(|_| RuntimeError::message("native function handle space exhausted"))?;
        self.natives.push(NativeEntry::Paired { callback, paired });
        Ok(NativeFunctionHandle(index))
    }

    /// Registers a native function that can capture state owned by the caller.
    ///
    /// # Errors
    /// Returns an error if the native-function handle space is exhausted.
    pub fn register_native_closure(
        &mut self,
        callback: impl Fn(&mut Runtime, &[Value]) -> RuntimeResult<Vec<Value>> + 'static,
    ) -> RuntimeResult<NativeFunctionHandle> {
        let index = u32::try_from(self.natives.len())
            .map_err(|_| RuntimeError::message("native function handle space exhausted"))?;
        self.natives.push(NativeEntry::Closure(Arc::new(callback)));
        Ok(NativeFunctionHandle(index))
    }

    /// Registers a single-result function that calls bypassing the VM's
    /// general call path must not run directly: [`Self::call_native_scalar`]
    /// and [`Self::call_native_unhosted`] decline it. The string
    /// metatable's arithmetic uses this, so the VM can hand a call to a
    /// second operand's metamethod as lstrlib's `trymt` does.
    ///
    /// # Errors
    /// Returns an error if the native-function handle space is exhausted.
    pub fn register_native_dispatched(
        &mut self,
        callback: NativeScalarCallback,
    ) -> RuntimeResult<NativeFunctionHandle> {
        let index = u32::try_from(self.natives.len())
            .map_err(|_| RuntimeError::message("native function handle space exhausted"))?;
        self.natives.push(NativeEntry::Dispatched(callback));
        Ok(NativeFunctionHandle(index))
    }

    /// Registers a function returning exactly one value, including nil.
    ///
    /// # Errors
    /// Returns an error if the native-function handle space is exhausted.
    pub fn register_native_scalar(
        &mut self,
        callback: NativeScalarCallback,
    ) -> RuntimeResult<NativeFunctionHandle> {
        let index = u32::try_from(self.natives.len())
            .map_err(|_| RuntimeError::message("native function handle space exhausted"))?;
        self.natives.push(NativeEntry::Scalar(callback));
        Ok(NativeFunctionHandle(index))
    }

    /// Registers a scalar callback with an opt-in pure numeric fast path.
    ///
    /// The scalar callback remains the normal Lua entry point, including
    /// coercion and error behavior. The numeric function must not mutate the
    /// runtime and may only depend on its `f64` input. Its result must match
    /// the scalar callback invoked with exactly one `Value::Number` argument.
    ///
    /// # Errors
    /// Returns an error if the native-function handle space is exhausted.
    pub fn register_native_numeric_unary(
        &mut self,
        callback: NativeScalarCallback,
        unary: NativeNumericUnary,
    ) -> RuntimeResult<NativeFunctionHandle> {
        let index = u32::try_from(self.natives.len())
            .map_err(|_| RuntimeError::message("native function handle space exhausted"))?;
        self.natives
            .push(NativeEntry::NumericUnary { callback, unary });
        Ok(NativeFunctionHandle(index))
    }

    /// Registers a scalar callback with a pure numeric path for two numbers.
    /// The binary function must match the callback for exactly two numeric
    /// arguments and must not inspect or mutate runtime state.
    ///
    /// # Errors
    /// Returns an error if the native-function handle space is exhausted.
    pub fn register_native_numeric_binary(
        &mut self,
        callback: NativeScalarCallback,
        binary: NativeNumericBinary,
    ) -> RuntimeResult<NativeFunctionHandle> {
        let index = u32::try_from(self.natives.len())
            .map_err(|_| RuntimeError::message("native function handle space exhausted"))?;
        self.natives.push(NativeEntry::NumericBinary {
            callback,
            binary,
            wide: None,
        });
        Ok(NativeFunctionHandle(index))
    }

    /// [`Self::register_native_numeric_binary`] with a pure form for other
    /// argument pairs, which the interpreter's in-window call tries before
    /// the callback. `wide` must agree with the callback wherever it answers.
    pub(crate) fn register_native_wide_binary(
        &mut self,
        callback: NativeScalarCallback,
        binary: NativeNumericBinary,
        wide: NativeWideBinary,
    ) -> RuntimeResult<NativeFunctionHandle> {
        let index = u32::try_from(self.natives.len())
            .map_err(|_| RuntimeError::message("native function handle space exhausted"))?;
        self.natives.push(NativeEntry::NumericBinary {
            callback,
            binary,
            wide: Some(wide),
        });
        Ok(NativeFunctionHandle(index))
    }

    /// The pure form a two-argument native has for argument pairs other than
    /// two numbers (see [`Self::register_native_wide_binary`]).
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub(crate) fn native_wide_binary(
        &self,
        function: NativeFunctionHandle,
    ) -> Option<NativeWideBinary> {
        match self.natives.get(function.index() as usize) {
            Some(NativeEntry::NumericBinary { wide, .. }) => *wide,
            _ => None,
        }
    }

    /// Returns a registered pure numeric unary fast path, if one exists.
    #[must_use]
    pub fn native_numeric_unary(
        &self,
        function: NativeFunctionHandle,
    ) -> Option<NativeNumericUnary> {
        match self.natives.get(function.index() as usize)? {
            NativeEntry::NumericUnary { unary, .. } => Some(*unary),
            NativeEntry::Multiple(_)
            | NativeEntry::Paired { .. }
            | NativeEntry::Closure(_)
            | NativeEntry::Host(_)
            | NativeEntry::Scalar(_)
            | NativeEntry::Dispatched(_)
            | NativeEntry::NumericBinary { .. } => None,
        }
    }

    /// Returns the opt-in pure numeric binary callback, if registered.
    #[must_use]
    pub fn native_numeric_binary(
        &self,
        function: NativeFunctionHandle,
    ) -> Option<NativeNumericBinary> {
        match self.natives.get(function.index() as usize)? {
            NativeEntry::NumericBinary { binary, .. } => Some(*binary),
            NativeEntry::Multiple(_)
            | NativeEntry::Paired { .. }
            | NativeEntry::Closure(_)
            | NativeEntry::Host(_)
            | NativeEntry::Scalar(_)
            | NativeEntry::Dispatched(_)
            | NativeEntry::NumericUnary { .. } => None,
        }
    }

    /// An empty vector for a native's results. A VM that is done with the
    /// results of a call hands the vector back through
    /// [`Self::recycle_result_vec`], so calls that return a few values
    /// allocate nothing in a steady state.
    #[inline]
    pub fn result_vec(&mut self) -> Vec<Value> {
        self.result_pool
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(4))
    }

    /// Takes back a results vector once its values have been used.
    #[inline]
    pub fn recycle_result_vec(&mut self, mut results: Vec<Value>) {
        if results.capacity() <= 16 && self.result_pool.len() < 4 {
            results.clear();
            self.result_pool.push(results);
        }
    }

    /// [`Self::native_call_kind`] with the pure numeric function of a native
    /// that has one. One table read.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn native_plan(&self, function: NativeFunctionHandle) -> NativePlan {
        match self.natives.get(function.index() as usize) {
            Some(NativeEntry::Scalar(callback)) => NativePlan::Scalar(*callback),
            Some(NativeEntry::NumericUnary { callback, unary }) => {
                NativePlan::Unary(*callback, *unary)
            }
            Some(NativeEntry::NumericBinary {
                callback, binary, ..
            }) => NativePlan::Binary(*callback, *binary),
            Some(NativeEntry::Paired { callback, paired }) => {
                NativePlan::Paired(*callback, *paired)
            }
            Some(NativeEntry::Multiple(callback)) => NativePlan::Multiple(*callback),
            _ => NativePlan::Other,
        }
    }

    /// The callback behind `function`, for a caller that runs it itself.
    /// Skips the result wrapping of [`Self::call_native_scalar`].
    #[inline]
    #[must_use]
    pub fn native_call_kind(&self, function: NativeFunctionHandle) -> NativeCallKind {
        match self.natives.get(function.index() as usize) {
            Some(
                NativeEntry::Scalar(callback)
                | NativeEntry::NumericUnary { callback, .. }
                | NativeEntry::NumericBinary { callback, .. },
            ) => NativeCallKind::Scalar(*callback),
            Some(NativeEntry::Multiple(callback) | NativeEntry::Paired { callback, .. }) => {
                NativeCallKind::Multiple(*callback)
            }
            _ => NativeCallKind::Other,
        }
    }

    /// Calls a single-result native without allocating argument or result vectors.
    ///
    /// `None` means this is an ordinary multi-result callback and it was not
    /// called. `Some(Value::Nil)` is a successful single nil result.
    ///
    /// # Errors
    /// Returns an error for an invalid handle or a failed callback.
    pub fn call_native_scalar(
        &mut self,
        function: NativeFunctionHandle,
        arguments: &[Value],
    ) -> RuntimeResult<Option<Value>> {
        match self
            .natives
            .get(function.index() as usize)
            .ok_or_else(|| RuntimeError::InvalidNativeFunction(function))?
        {
            NativeEntry::Multiple(_)
            | NativeEntry::Paired { .. }
            | NativeEntry::Closure(_)
            | NativeEntry::Host(_)
            | NativeEntry::Dispatched(_) => Ok(None),
            NativeEntry::Scalar(callback)
            | NativeEntry::NumericUnary { callback, .. }
            | NativeEntry::NumericBinary { callback, .. } => {
                let callback = *callback;
                match callback(self, arguments) {
                    Err(RuntimeError::Redispatch) => Ok(None),
                    result => result.map(Some),
                }
            }
        }
    }

    /// Registers a native function and stores it in a global name.
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime has exhausted its native-function
    /// handle space.
    pub fn register_global_native(
        &mut self,
        name: impl AsRef<[u8]>,
        callback: NativeCallback,
    ) -> RuntimeResult<NativeFunctionHandle> {
        let handle = self.register_native(callback)?;
        self.set_native_name(handle, &String::from_utf8_lossy(name.as_ref()));
        self.set_global(name, Value::NativeFunction(handle));
        Ok(handle)
    }
    /// Calls a native function the runtime can run alone, as
    /// [`Self::call_native`] does, without looking up VM host callbacks.
    ///
    /// `None` means `function` is a VM host callback and was not called.
    ///
    /// # Errors
    /// Returns an error for an invalid handle or a failed callback.
    pub fn call_native_unhosted(
        &mut self,
        function: NativeFunctionHandle,
        arguments: &[Value],
    ) -> RuntimeResult<Option<Vec<Value>>> {
        match self
            .natives
            .get(function.index() as usize)
            .ok_or_else(|| RuntimeError::InvalidNativeFunction(function))?
        {
            NativeEntry::Host(_) | NativeEntry::Dispatched(_) => Ok(None),
            NativeEntry::Multiple(callback) | NativeEntry::Paired { callback, .. } => {
                let callback = *callback;
                match callback(self, arguments) {
                    Err(RuntimeError::Redispatch) => Ok(None),
                    result => result.map(Some),
                }
            }
            NativeEntry::Closure(callback) => {
                let callback = Arc::clone(callback);
                callback(self, arguments).map(Some)
            }
            NativeEntry::Scalar(callback)
            | NativeEntry::NumericUnary { callback, .. }
            | NativeEntry::NumericBinary { callback, .. } => {
                let callback = *callback;
                match callback(self, arguments) {
                    Err(RuntimeError::Redispatch) => Ok(None),
                    result => result.map(|value| Some(vec![value])),
                }
            }
        }
    }
    /// Calls a previously registered native function.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid handle or any error returned by the
    /// native function itself.
    pub fn call_native(
        &mut self,
        function: NativeFunctionHandle,
        arguments: &[Value],
    ) -> RuntimeResult<Vec<Value>> {
        let callback = self
            .natives
            .get(function.index() as usize)
            .ok_or_else(|| RuntimeError::InvalidNativeFunction(function))?
            .clone();
        match callback {
            NativeEntry::Host(_) => Err(RuntimeError::message("host callback requires its VM")),
            NativeEntry::Multiple(callback) | NativeEntry::Paired { callback, .. } => {
                callback(self, arguments)
            }
            NativeEntry::Closure(callback) => callback(self, arguments),
            NativeEntry::Scalar(callback)
            | NativeEntry::Dispatched(callback)
            | NativeEntry::NumericUnary { callback, .. }
            | NativeEntry::NumericBinary { callback, .. } => {
                callback(self, arguments).map(|value| vec![value])
            }
        }
    }
    #[must_use]
    pub fn raw_equal(left: &Value, right: &Value) -> bool {
        match (left, right) {
            (Value::Nil, Value::Nil) => true,
            (Value::Boolean(left), Value::Boolean(right)) => left == right,
            (Value::Integer(left), Value::Integer(right)) => left == right,
            (Value::Integer(left), Value::Number(right))
            | (Value::Number(right), Value::Integer(left)) => integer_key_from_number(*right)
                .is_some_and(|integer_right| {
                    integer_right == *left && numbers_equal(integer_to_number(*left), *right)
                }),
            (Value::Number(left), Value::Number(right)) => numbers_equal(*left, *right),
            (Value::String(left), Value::String(right)) => left == right,
            (Value::Table(left), Value::Table(right)) => left == right,
            (Value::File(left), Value::File(right)) => left == right,
            (Value::Userdata(left), Value::Userdata(right)) => left == right,
            (Value::GmatchIterator(left), Value::GmatchIterator(right)) => left == right,
            (Value::Thread(left), Value::Thread(right))
            | (Value::CoroutineWrapper(left), Value::CoroutineWrapper(right)) => left == right,
            (Value::Closure(left), Value::Closure(right)) => left == right,
            (Value::NativeFunction(left), Value::NativeFunction(right)) => left == right,
            (Value::BuiltinFunction(left), Value::BuiltinFunction(right)) => left == right,
            (Value::Int64(left), Value::Int64(right)) => left == right,
            (Value::UInt64(left), Value::UInt64(right)) => left == right,
            _ => false,
        }
    }
    #[inline]
    pub(crate) fn table(&self, handle: TableHandle) -> RuntimeResult<&Table> {
        let Some(table) = self.tables.get(handle.0) else {
            return Err(RuntimeError::InvalidTable(handle));
        };
        Ok(table)
    }
    pub(crate) fn table_mut(&mut self, handle: TableHandle) -> RuntimeResult<&mut Table> {
        if self.gc_partial_table == Some(handle) {
            // A major cycle is tracing this table in chunks.
            self.gc_partial_dirty = true;
        }
        let Some(table) = self.tables.get_mut(handle.0) else {
            return Err(RuntimeError::InvalidTable(handle));
        };
        // A change may add a metamethod this table was known to lack.
        *table.absent_metamethods.get_mut() = 0;
        // Any change to a table that cached inherited reads depend on may
        // alter what they find.
        if table.watched {
            self.inherit_epoch += 1;
        }
        Ok(table)
    }

    /// The metatable all numbers share, if one is set.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn number_metatable(&self) -> Option<TableHandle> {
        self.shared_metatables[SharedMetatable::Number as usize]
    }

    /// The metatable all strings share, if one is set.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn string_metatable(&self) -> Option<TableHandle> {
        self.shared_metatables[SharedMetatable::String as usize]
    }

    /// The key `__index`, as the runtime holds it for its own lookups, so a
    /// caller that reads a metatable's `__index` makes no string.
    #[must_use]
    pub const fn index_metamethod_key(&self) -> &LuaString {
        &self.index_metamethod_key
    }

    pub(crate) fn shared_metatable(&self, value: &Value) -> Option<TableHandle> {
        self.shared_metatables[shared_metatable_kind(value) as usize]
    }
    pub(crate) fn file(&self, handle: FileHandle) -> RuntimeResult<&RuntimeFile> {
        self.files
            .get(handle.0)
            .ok_or_else(|| RuntimeError::InvalidFile(handle))
    }
    pub(crate) fn file_open(&self, handle: FileHandle) -> RuntimeResult<()> {
        let file = self.file(handle)?;
        if matches!(file.storage, FileStorage::Closed) {
            return Err(RuntimeError::message("attempt to use a closed file"));
        }
        Ok(())
    }
    pub(crate) fn file_mut_open(&mut self, handle: FileHandle) -> RuntimeResult<&mut FileStorage> {
        let file = self
            .files
            .get_mut(handle.0)
            .ok_or_else(|| RuntimeError::InvalidFile(handle))?;
        if matches!(file.storage, FileStorage::Closed) {
            return Err(RuntimeError::message("attempt to use a closed file"));
        }
        Ok(&mut file.storage)
    }
}
