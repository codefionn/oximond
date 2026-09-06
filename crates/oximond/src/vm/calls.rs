//! Closure creation and Lua/native call dispatch, including tail calls and inline fast paths.

use crate::bytecode::{Chunk, Instruction};
use crate::compat::CompatibilityProfile;
use crate::runtime::{
    BuiltinFunction, CaptureCells, CellHandle, ClosureHandle, NativeFunctionHandle, RuntimeError,
    TableHandle, Value,
};
use crate::vm::arith::{call_metamethod_key, usize_to_number};
use crate::vm::coroutines::coroutine_argument;
use crate::vm::debug_info::active_call_site;
use crate::vm::errors::{
    builtin_argument_error, call_argument_error, call_argument_error_extra, call_type_error,
    loaded_function_name, raised_value,
};
use crate::vm::frame::FrameChunk;
use crate::vm::registers::{
    call_target_and_fixed_arguments, set_register, write_call_results, write_scalar_call_result,
};
use crate::vm::stack::{ActivationArguments, CallStack, OwnedFrame, ensure_call_depth};
use crate::vm::{
    CALL_HINT_GENERAL, CALL_HINT_NUMERIC_PLAN, CALL_HINT_PLAIN, CALL_HINT_TABLE_PLAN,
    CALL_HINT_UNKNOWN, CALL_HINT_VARARG, CallReturn, CollectGarbageAction, DebugHookEvent,
    DofileCompletion, FinalizerCompletion, ForeachCompletion, ForeachMode, Frame,
    InlineCallArguments, LoadCompletion, MAX_INDEX_HOPS, ModuleCompletion, RequireCompletion,
    ReturnTarget, SortCompletion, TailCallArguments, Vm, VmError, VmResult, bump,
};
use std::sync::Arc;

/// Captures a closure gathers without allocating a list.
const INLINE_CAPTURES: usize = 8;

impl Vm {
    #[allow(
        clippy::unnecessary_lazy_evaluations,
        reason = "avoid constructing the large VmError enum on successful bytecode lookups"
    )]
    pub(crate) fn create_closure(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<()> {
        let parent_index = frames.len() - 1;
        let prototype_index = instruction.d();
        let prototype = Arc::clone(
            frames[parent_index]
                .chunk
                .prototypes
                .get(usize::from(prototype_index))
                .ok_or_else(|| VmError::MissingPrototype(prototype_index))?,
        );
        // A closure with at most eight captures gathers them on the stack: it
        // allocates only the shared slice of two or more, and nothing for
        // none or one.
        let mut buffer = [0_u64; INLINE_CAPTURES];
        let mut overflow = Vec::new();
        let count = prototype.upvalues.len();
        for (index, descriptor) in prototype.upvalues.iter().enumerate() {
            let cell = match descriptor.source {
                crate::bytecode::UpvalueSource::ParentRegister(register) => {
                    self.capture_slot(frames, parent_index, register)?
                }
                crate::bytecode::UpvalueSource::ParentUpvalue(index) => *frames[parent_index]
                    .upvalues
                    .get(usize::from(index))
                    .ok_or_else(|| VmError::RegisterRange { start: 0, count: 0 })?,
                crate::bytecode::UpvalueSource::Detached => self.create_detached_cell()?,
            };
            // The handle travels as one word, so that no narrow store has to
            // be read back wide.
            if count <= INLINE_CAPTURES {
                buffer[index] = cell.native_identity();
            } else {
                overflow.push(cell.native_identity());
            }
        }
        let cells: &[u64] = if count <= INLINE_CAPTURES {
            &buffer[..count]
        } else {
            &overflow
        };
        let environment = frames[parent_index].environment;
        // SAFETY: a prototype of a validated chunk is validated, and every
        // word is the identity of a cell handle.
        #[allow(unsafe_code)]
        let identity = unsafe {
            self.runtime
                .create_closure_from_identities(prototype, cells, environment)
        };
        if identity == crate::runtime::NO_TABLE_IDENTITY {
            return Err(self.runtime.closure_creation_error(environment).into());
        }
        let handle = ClosureHandle::from_native_identity(identity);
        #[cfg(feature = "op-stats")]
        crate::vm::op_stats::bump(crate::vm::op_stats::Counter::ClosuresCreated);
        // Written in place: a value built before the old register's drop is
        // kept on the stack across it, and loaded back with a wide load that
        // cannot forward from the narrow stores that built it.
        set_register(
            &mut frames.at_mut(parent_index).registers[usize::from(instruction.a())],
            Value::Closure(handle),
        );
        Ok(())
    }

    pub(crate) fn install_closure(
        &mut self,
        chunk: impl Into<Arc<Chunk>>,
        cells: Vec<CellHandle>,
        environment: TableHandle,
    ) -> VmResult<ClosureHandle> {
        // SAFETY: every caller passes a chunk that went through
        // `Chunk::validate`, or a prototype of one.
        #[allow(unsafe_code)]
        let handle = unsafe {
            self.runtime
                .create_closure_in_environment_unchecked(chunk.into(), cells, environment)
        };
        handle.map_err(Into::into)
    }

    pub(crate) fn call_frame(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<()> {
        let frame_index = frames.len() - 1;
        let base = usize::from(instruction.a());
        if instruction.b() == 2
            && instruction.c() == 2
            && self.try_call_ipairs_iterator_fixed(frames, base, base, 2)?
        {
            return Ok(());
        }
        let function = frames
            .at_mut(frame_index)
            .registers
            .get(base)
            .cloned()
            .ok_or_else(|| VmError::RegisterRange {
                start: instruction.a(),
                count: 1,
            })?;
        if let Value::NativeFunction(function) = function
            && self.try_call_native_scalar_range(
                frames,
                instruction,
                function,
                base + 1,
                usize::from(instruction.b()),
            )?
        {
            return Ok(());
        }
        if let Value::Closure(closure) = function {
            self.classify_closure_call(closure);
        }
        if self.try_inline_numeric_fixed_call(frames, instruction, &function)? {
            return Ok(());
        }
        if instruction.c() == 1
            && let Value::Closure(closure) = function
            && self.try_inline_table_call(
                frames,
                instruction,
                closure,
                base + 1,
                usize::from(instruction.b()),
            )?
        {
            return Ok(());
        }
        if instruction.b() == 0
            && let Value::Closure(closure) = function
        {
            return self.call_fixed_zero_argument_closure(frames, instruction, closure);
        }
        if instruction.b() == 1
            && let Value::Closure(closure) = function
        {
            let argument = frames
                .at_mut(frame_index)
                .registers
                .get(base + 1)
                .cloned()
                .ok_or_else(|| VmError::RegisterRange {
                    start: instruction.a(),
                    count: instruction.b(),
                })?;
            return self.call_fixed_one_argument_closure(frames, instruction, closure, argument);
        }
        if instruction.b() >= 2
            && let Value::Closure(closure) = function
        {
            ensure_call_depth(frames)?;
            let count = usize::from(instruction.b());
            let caller = frames.last().expect("a calling frame exists");
            if base + 1 + count > caller.registers.len() {
                return Err(VmError::RegisterRange {
                    start: instruction.a(),
                    count: instruction.b(),
                });
            }
            let start = caller.base as usize + base + 1;
            if self.push_plain_closure(
                frames,
                closure,
                CallReturn {
                    base: instruction.a(),
                    results: instruction.c(),
                },
                ActivationArguments::Stack { start, count },
            )? {
                return Ok(());
            }
            // The frame borrows the closure's chunk and cells instead of
            // counting them; it roots `closure` while it is live.
            let (chunk, cells, environment) = self.borrow_closure_call(closure)?;
            // A direct caller's frame names this call site on demand.
            let call_site = None;
            let count = usize::from(instruction.b());
            let caller = ReturnTarget::Direct(CallReturn {
                base: instruction.a(),
                results: instruction.c(),
            });
            if chunk.arg_register.is_none() {
                // Arguments past the fixed parameters stay on the stack as
                // the frame's varargs: the frame is pushed over them.
                let frame =
                    Frame::activation(chunk, cells, Some(caller), call_site, environment, closure);
                return self.push_fixed_closure_frame(frames, frame, base + 1, count, instruction);
            }
            let arguments = frames
                .at(frame_index)
                .registers
                .get(base + 1..base + 1 + count)
                .ok_or_else(|| VmError::RegisterRange {
                    start: instruction.a(),
                    count: instruction.b(),
                })?
                .to_vec();
            let frame = self.make_frame(
                chunk,
                arguments,
                cells,
                Some(caller),
                call_site,
                environment,
                Some(closure),
            )?;
            return self.enter_called_frame(frames, frame);
        }
        if self.debug_hooks.is_empty() {
            if let Value::BuiltinFunction(BuiltinFunction::ToString) = function {
                if self.try_tostring_borrowed(frames, instruction, base)? {
                    return Ok(());
                }
            } else if let Value::NativeFunction(native) = function
                && self.try_call_native_multi_borrowed(frames, instruction, native, base)?
            {
                return Ok(());
            }
        }
        let (function, arguments) =
            call_target_and_fixed_arguments(frames.at(frame_index).registers, instruction)?;
        self.dispatch_call(frames, instruction, function, arguments)
    }

    /// Calls a multi-result native function without cloning its arguments.
    ///
    /// Single-result natives already run through
    /// [`Self::try_call_native_scalar_range`]; this covers the remaining
    /// `Multiple` entries (such as `string.find`) with the registers
    /// borrowed in place. Collection only runs at safepoints, so the
    /// borrowed window cannot observe a collection.
    pub(crate) fn try_call_native_multi_borrowed(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        function: NativeFunctionHandle,
        base: usize,
    ) -> VmResult<bool> {
        if self.is_host_function(function) {
            return Ok(false);
        }
        let count = usize::from(instruction.b());
        let results = {
            let frame = frames.last().expect("native call has a frame");
            let arguments = frame
                .registers
                .get(base + 1..base + 1 + count)
                .ok_or_else(|| VmError::RegisterRange {
                    start: instruction.a(),
                    count: instruction.b(),
                })?;
            let results = self
                .runtime
                .call_native_unhosted(function, arguments)
                .map_err(|error| {
                    call_argument_error(
                        &frames.last().expect("native call has a frame"),
                        frames.nth_from_top(1).as_deref(),
                        error,
                        self.compatibility,
                    )
                })?;
            // A dispatched native takes the general call path.
            let Some(results) = results else {
                return Ok(false);
            };
            results
        };
        let frame_index = frames.len() - 1;
        write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
        Ok(true)
    }

    /// Formats one `tostring` argument without building an argument vector.
    ///
    /// Values with a `__tostring` metamethod keep the general path so
    /// handler dispatch stays in one place. Plain values format directly
    /// into the call's result window.
    pub(crate) fn try_tostring_borrowed(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        base: usize,
    ) -> VmResult<bool> {
        if instruction.b() == 0 {
            return Ok(false);
        }
        self.try_tostring_register(frames, instruction, base + 1)
    }

    /// Formats register `argument` as `tostring` would, writing the result
    /// as the call `instruction` (register `a`, result count `c`) expects.
    /// Returns `false`, having changed nothing, when the value has a
    /// `__tostring` metamethod.
    pub(crate) fn try_tostring_register(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        argument: usize,
    ) -> VmResult<bool> {
        let formatted = {
            let value = frames
                .last()
                .expect("tostring call has a frame")
                .registers
                .get(argument)
                .ok_or_else(|| VmError::RegisterRange {
                    start: instruction.a(),
                    count: instruction.b(),
                })?;
            if self.has_tostring_metamethod(value)? {
                return Ok(false);
            }
            // Numbers format infallibly; building their string here keeps
            // it out of a `Result` the caller would reload from memory.
            Value::String(match value {
                Value::Number(number) => self.runtime.number_format().format_string(*number),
                Value::String(text) => text.clone(),
                value => self.runtime.format_value(value)?,
            })
        };
        let frame_index = frames.len() - 1;
        write_scalar_call_result(&mut frames.at_mut(frame_index), instruction, formatted)?;
        Ok(true)
    }

    /// Executes one step of `for _ in ipairs(table) do`, from the iterator,
    /// table and control value in the registers at `source`. The index and
    /// element land in the registers from `destination`, padded with nil up
    /// to `results`. A call has both at its base; a `ForInCall` reads the
    /// control registers and writes the variables after them.
    ///
    /// This is intentionally narrow. A metatable, a coercible control value,
    /// open results, or hooks all retain the ordinary builtin call so their
    /// existing semantics stay in one place.
    #[inline(always)]
    pub(crate) fn try_call_ipairs_iterator_fixed(
        &mut self,
        frames: &mut CallStack,
        source: usize,
        destination: usize,
        results: usize,
    ) -> VmResult<bool> {
        if results == 0 || !self.debug_hooks.is_empty() {
            return Ok(false);
        }
        let mut frame = frames.at_mut(frames.len() - 1);
        // Read all three operands before replacing the result window. A call
        // overwrites the iterator and state registers with its results. A
        // window that does not fit takes the generic path, which reports the
        // register range.
        if destination + results.max(2) > frame.registers.len() {
            return Ok(false);
        }
        let Some(window) = frame.registers.get(source..source + 3) else {
            return Ok(false);
        };
        let [
            Value::BuiltinFunction(BuiltinFunction::IPairsIterator),
            Value::Table(table),
            control,
        ] = window
        else {
            return Ok(false);
        };
        let step = match control {
            Value::Number(control) => IpairsIndex::Number(*control + 1.0),
            Value::Integer(control) => IpairsIndex::Integer(control.wrapping_add(1)),
            _ => return Ok(false),
        };
        let table = *table;
        // The array part answers almost every step. It needs no `Value` key,
        // so nothing is staged in memory for the lookup.
        let array_hit = match step {
            IpairsIndex::Number(number) => self.runtime.table_array_hit_number(table, number),
            IpairsIndex::Integer(integer) => usize::try_from(integer)
                .ok()
                .and_then(|index| self.runtime.table_array_hit(table, index)),
        };
        if let Some(hit) = array_hit {
            frame.clear_open_results();
            Self::store_ipairs_results(frame.registers, destination, results, step, hit);
        } else {
            // A present value is the result under every profile: `__index`
            // is only consulted for absent keys. Only misses need the
            // metatable check below.
            let index = step.value();
            if let Some(hit) = self.runtime.table_get_raw_hit(table, &index) {
                frame.clear_open_results();
                Self::store_ipairs_results(frame.registers, destination, results, step, hit);
            } else {
                if self.runtime.raw_metatable(&Value::Table(table))?.is_some() {
                    return Ok(false);
                }
                let value = self.runtime.table_get(table, &index)?;
                frame.clear_open_results();
                Self::store_ipairs_results(frame.registers, destination, results, step, &value);
            }
        }
        self.jit.statistics.interpreter_ipairs_steps = self
            .jit
            .statistics
            .interpreter_ipairs_steps
            .saturating_add(1);
        Ok(true)
    }

    /// Executes one step of `for k, v in next, table do` (which is what
    /// `pairs(table)` returns), with the same register layout as
    /// [`Self::try_call_ipairs_iterator_fixed`]. The traversal is raw, so a
    /// metatable does not matter. An invalid continuation key takes the
    /// generic path, which raises the error.
    #[inline]
    pub(crate) fn try_call_next_iterator_fixed(
        &mut self,
        frames: &mut CallStack,
        source: usize,
        destination: usize,
        results: usize,
    ) -> VmResult<bool> {
        if results == 0 || !self.debug_hooks.is_empty() {
            return Ok(false);
        }
        let mut frame = frames.at_mut(frames.len() - 1);
        if destination + results.max(2) > frame.registers.len() {
            return Ok(false);
        }
        let Some(window) = frame.registers.get(source..source + 3) else {
            return Ok(false);
        };
        let [
            Value::BuiltinFunction(BuiltinFunction::Next),
            Value::Table(table),
            control,
        ] = window
        else {
            return Ok(false);
        };
        let pair = match self.runtime.table_next_fast(*table, control) {
            Some(pair) => pair,
            None => match self.runtime.table_next(*table, control) {
                Ok(pair) => pair,
                Err(_) => return Ok(false),
            },
        };
        frame.clear_open_results();
        Self::store_iteration_results(frame.registers, destination, results, pair.into());
        Ok(true)
    }

    /// The `ForInCall` step of the fast loop: one `ipairs` or `next` step
    /// from the control registers at `source`, written to the variables after
    /// them, in a frame with no captured register and no hook. Returns
    /// `false` with nothing changed whenever the general path must take the
    /// step: another iterator, a metatable consulted by `ipairs`, a control
    /// value that is not a number, or a `next` continuation the cursor
    /// cannot settle (which includes the invalid key that raises an error).
    #[inline(never)]
    pub(crate) fn for_in_builtin_step(
        &mut self,
        registers: &mut [Value; crate::vm::stack::REGISTER_WINDOW],
        source: usize,
        results: usize,
    ) -> bool {
        let destination = source + 3;
        if let Value::GmatchIterator(iterator) = registers[source] {
            return self.gmatch_step_in_place(iterator, &mut registers[..], destination, results);
        }
        let [
            Value::BuiltinFunction(function),
            Value::Table(table),
            control,
        ] = &registers[source..source + 3]
        else {
            return false;
        };
        let table = *table;
        match function {
            BuiltinFunction::IPairsIterator => {
                let step = match control {
                    Value::Number(control) => IpairsIndex::Number(*control + 1.0),
                    Value::Integer(control) => IpairsIndex::Integer(control.wrapping_add(1)),
                    _ => return false,
                };
                let array_hit = match step {
                    IpairsIndex::Number(number) => {
                        self.runtime.table_array_hit_number(table, number)
                    }
                    IpairsIndex::Integer(integer) => usize::try_from(integer)
                        .ok()
                        .and_then(|index| self.runtime.table_array_hit(table, index)),
                };
                if let Some(hit) = array_hit {
                    Self::store_ipairs_results(registers, destination, results, step, hit);
                } else {
                    let index = step.value();
                    if let Some(hit) = self.runtime.table_get_raw_hit(table, &index) {
                        Self::store_ipairs_results(registers, destination, results, step, hit);
                    } else {
                        // Absent: `__index` decides on the general path.
                        if !matches!(self.runtime.raw_metatable(&Value::Table(table)), Ok(None)) {
                            return false;
                        }
                        Self::store_ipairs_results(
                            registers,
                            destination,
                            results,
                            step,
                            &Value::Nil,
                        );
                    }
                }
                self.jit.statistics.interpreter_ipairs_steps =
                    self.jit.statistics.interpreter_ipairs_steps.wrapping_add(1);
                true
            }
            BuiltinFunction::Next => {
                let pair = match self.runtime.table_next_fast(table, control) {
                    Some(pair) => pair,
                    None => match self.runtime.table_next(table, control) {
                        Ok(pair) => pair,
                        Err(_) => return false,
                    },
                };
                Self::store_iteration_results(registers, destination, results, pair.into());
                true
            }
            _ => false,
        }
    }

    /// Writes an `ipairs` step: the index (nil when `value` is nil) and the
    /// value, nil padded or cut to `results`.
    #[inline(always)]
    pub(crate) fn store_ipairs_results(
        registers: &mut [Value],
        destination: usize,
        results: usize,
        index: IpairsIndex,
        value: &Value,
    ) {
        if matches!(value, Value::Nil) {
            crate::vm::registers::set_register(&mut registers[destination], Value::Nil);
        } else {
            match index {
                IpairsIndex::Number(number) => {
                    crate::vm::registers::store_number_wide(&mut registers[destination], number);
                }
                IpairsIndex::Integer(integer) => {
                    crate::vm::registers::store_integer_wide(&mut registers[destination], integer);
                }
            }
        }
        if results >= 2 {
            crate::vm::registers::store_clone_wide(&mut registers[destination + 1], value);
            for register in &mut registers[destination + 2..destination + results] {
                *register = Value::Nil;
            }
        }
    }

    /// Writes an iterator's two results to `registers[destination..]`, nil
    /// padded or cut to `results`.
    #[inline(always)]
    fn store_iteration_results(
        registers: &mut [Value],
        destination: usize,
        results: usize,
        pair: [Value; 2],
    ) {
        let [first, second] = pair;
        registers[destination] = first;
        if results >= 2 {
            registers[destination + 1] = second;
            for register in &mut registers[destination + 2..destination + results] {
                *register = Value::Nil;
            }
        }
    }

    /// `ForInCall`: calls the generic-for iterator at `A` with the state
    /// and control value after it. The iterator, state and control are
    /// copied to `A + 3 ..`, and the call runs from there exactly as the
    /// `Call` the compiler used to emit, so every call-related rule is the
    /// same. `ipairs` and `next` steps skip the copies and write their
    /// results in place.
    pub(crate) fn for_in_call_frame(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<()> {
        let frame_index = frames.len() - 1;
        let base = usize::from(instruction.a());
        let variables = usize::from(instruction.b());
        let window = base + 3;
        if self.try_call_ipairs_iterator_fixed(frames, base, window, variables)?
            || self.try_call_next_iterator_fixed(frames, base, window, variables)?
        {
            return Ok(());
        }
        {
            let frame = frames.at_mut(frame_index);
            if window + 3 > frame.registers.len() {
                return Err(VmError::RegisterRange {
                    start: instruction.a(),
                    count: instruction.b(),
                });
            }
            for offset in 0..3 {
                frame.registers[window + offset] = frame.registers[base + offset].clone();
            }
        }
        let call = Instruction::call(instruction.a() + 3, 2, instruction.b());
        self.call_frame(frames, call)
    }

    pub(crate) fn call_one_frame(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<()> {
        let frame_index = frames.len() - 1;
        let function = {
            let registers = frames.at(frame_index).registers;
            registers
                .get(usize::from(instruction.b()))
                .cloned()
                .ok_or_else(|| VmError::RegisterRange {
                    start: instruction.b(),
                    count: 1,
                })?
        };
        let call = Instruction::call(instruction.a(), 1, 1);
        if let Value::NativeFunction(function) = function
            && self.try_call_native_scalar_range(
                frames,
                call,
                function,
                usize::from(instruction.c()),
                1,
            )?
        {
            return Ok(());
        }
        let argument = frames
            .at_mut(frame_index)
            .registers
            .get(usize::from(instruction.c()))
            .cloned()
            .ok_or_else(|| VmError::RegisterRange {
                start: instruction.c(),
                count: 1,
            })?;
        if let Value::Closure(closure) = &function {
            self.classify_closure_call(*closure);
        }
        if self.try_inline_numeric_call(
            frames,
            instruction.a(),
            &function,
            InlineCallArguments::Values(std::slice::from_ref(&argument)),
        )? {
            return Ok(());
        }
        if let Value::Closure(closure) = function {
            return self.call_fixed_one_argument_closure(frames, call, closure, argument);
        }
        if self.debug_hooks.is_empty() {
            // `CallOne` carries its single argument directly: reuse it
            // without building a call window or argument vector.
            if let Value::BuiltinFunction(BuiltinFunction::ToString) = &function {
                if self.tostring_metamethod(&argument)?.is_none() {
                    let formatted = Value::String(self.runtime.format_value(&argument)?);
                    let frame_index = frames.len() - 1;
                    write_scalar_call_result(&mut frames.at_mut(frame_index), call, formatted)?;
                    return Ok(());
                }
            } else if let Value::NativeFunction(native) = &function {
                let results = self
                    .call_native(frames, *native, std::slice::from_ref(&argument))
                    .map_err(|error| {
                        call_argument_error(
                            &frames.last().expect("native call has a frame"),
                            frames.nth_from_top(1).as_deref(),
                            error,
                            self.compatibility,
                        )
                    })?;
                let frame_index = frames.len() - 1;
                write_call_results(&mut frames.at_mut(frame_index), call, results)?;
                return Ok(());
            }
        }
        self.dispatch_call(frames, call, function, vec![argument])
    }

    /// Handles the overwhelmingly common numeric `CallOne` without building
    /// the general argument window. This stays separate from the fixed-call
    /// helper because it receives the scalar directly from the bytecode path.
    /// Executes a scalar native directly from the caller's register window.
    /// The runtime returns `None` for ordinary native callbacks without
    /// invoking them, so those retain the generic call path below.
    pub(crate) fn try_call_native_scalar_range(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        function: NativeFunctionHandle,
        argument_start: usize,
        argument_count: usize,
    ) -> VmResult<bool> {
        let frame_index = frames.len() - 1;
        let scalar = {
            let frame = frames.at(frame_index);
            let arguments = frame
                .registers
                .get(argument_start..argument_start + argument_count)
                .ok_or_else(|| VmError::RegisterRange {
                    start: instruction.a(),
                    count: u8::try_from(argument_count).unwrap_or(u8::MAX),
                })?;
            self.runtime.call_native_scalar(function, arguments)
        }
        .map_err(|error| {
            call_argument_error(
                &frames[frame_index],
                frame_index.checked_sub(1).map(|i| &frames[i]),
                error,
                self.compatibility,
            )
        })?;
        let Some(value) = scalar else {
            return Ok(false);
        };
        write_scalar_call_result(&mut frames.at_mut(frame_index), instruction, value)?;
        self.observe_native_call(frames, Value::NativeFunction(function), argument_count)?;
        Ok(true)
    }

    /// A closure's chunk, cells and environment for a frame that will root
    /// `closure`, borrowed without reference counting.
    ///
    /// The caller must give the result to a frame whose `closure` is set to
    /// `closure` before the next collection, and must not keep it past that
    /// frame.
    #[inline(always)]
    fn borrow_closure_call(
        &self,
        closure: ClosureHandle,
    ) -> VmResult<(FrameChunk, CaptureCells, TableHandle)> {
        let parts = self
            .runtime
            .closure_call_parts(closure)
            .ok_or(VmError::MissingClosureBody(closure))?;
        // SAFETY: both come from `closure`; the caller moves them into a
        // frame that roots it, per this function's contract.
        #[allow(unsafe_code)]
        let (chunk, cells) = unsafe {
            (
                FrameChunk::borrowed(parts.chunk),
                CaptureCells::borrow_shared(parts.cells),
            )
        };
        Ok((chunk, cells, parts.environment))
    }

    pub(crate) fn call_fixed_zero_argument_closure(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        closure: ClosureHandle,
    ) -> VmResult<()> {
        ensure_call_depth(frames)?;
        let frame_index = frames.len() - 1;
        if self.push_plain_closure(
            frames,
            closure,
            CallReturn {
                base: instruction.a(),
                results: instruction.c(),
            },
            ActivationArguments::None,
        )? {
            return Ok(());
        }
        // Only vararg closures remain: they keep the general frame builder.
        let (chunk, cells, environment) = self.borrow_closure_call(closure)?;
        let call_site = active_call_site(&frames[frame_index]);
        let frame = self.make_frame(
            chunk,
            Vec::new(),
            cells,
            Some(ReturnTarget::Direct(CallReturn {
                base: instruction.a(),
                results: instruction.c(),
            })),
            call_site,
            environment,
            Some(closure),
        )?;
        self.enter_called_frame(frames, frame)
    }

    pub(crate) fn call_fixed_one_argument_closure(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        closure: ClosureHandle,
        argument: Value,
    ) -> VmResult<()> {
        ensure_call_depth(frames)?;
        let frame_index = frames.len() - 1;
        let callee_is_vararg = self
            .runtime
            .closure_call_parts(closure)
            .is_some_and(|parts| parts.chunk.is_vararg);
        if !callee_is_vararg {
            if self.push_plain_closure(
                frames,
                closure,
                CallReturn {
                    base: instruction.a(),
                    results: instruction.c(),
                },
                ActivationArguments::One(argument),
            )? {
                return Ok(());
            }
            return Err(VmError::MissingClosureBody(closure));
        }
        // A vararg callee records its call site eagerly, as other vararg
        // frames do, and keeps the general frame builder.
        let (chunk, cells, environment) = self.borrow_closure_call(closure)?;
        let call_site = active_call_site(&frames[frame_index]);
        let frame = self.make_frame(
            chunk,
            std::iter::once(argument),
            cells,
            Some(ReturnTarget::Direct(CallReturn {
                base: instruction.a(),
                results: instruction.c(),
            })),
            call_site,
            environment,
            Some(closure),
        )?;
        self.enter_called_frame(frames, frame)
    }

    pub(crate) fn call_vararg_frame(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<()> {
        let frame_index = frames.len() - 1;
        let (function, mut arguments) =
            call_target_and_fixed_arguments(frames.at(frame_index).registers, instruction)?;
        arguments.extend(self.frame_varargs(frames, frame_index)?);
        self.dispatch_call(frames, instruction, function, arguments)
    }

    pub(crate) fn frame_varargs(&self, frames: &CallStack, index: usize) -> VmResult<Vec<Value>> {
        if self.compatibility != CompatibilityProfile::Lua55 {
            return Ok(frames.varargs(index).to_vec());
        }
        let frame = frames.at(index);
        let Some(register) = frame.chunk.arg_register else {
            return Ok(frames.varargs(index).to_vec());
        };
        let Value::Table(table) =
            frame
                .registers
                .get(usize::from(register))
                .ok_or(VmError::RegisterRange {
                    start: register,
                    count: 1,
                })?
        else {
            return Err(RuntimeError::message("named vararg binding is not a table").into());
        };
        let count = self.runtime.table_get(*table, &Value::from("n"))?;
        let count = match count {
            Value::Integer(count) if (0..=1_000_000).contains(&count) => {
                usize::try_from(count).expect("bounded non-negative integer fits usize")
            }
            Value::Number(count)
                if count.is_finite()
                    && count >= 0.0
                    && count.fract() == 0.0
                    && count <= 1_000_000.0 =>
            {
                format!("{count:.0}")
                    .parse::<usize>()
                    .map_err(|_| RuntimeError::message("no proper 'n' in named vararg table"))?
            }
            _ => {
                return Err(RuntimeError::message("no proper 'n' in named vararg table").into());
            }
        };
        let mut values = Vec::with_capacity(count);
        for index in 1..=count {
            values.push(
                self.runtime
                    .table_get(*table, &Value::Number(usize_to_number(index)?))?,
            );
        }
        Ok(values)
    }

    pub(crate) fn call_open_frame(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<()> {
        let frame_index = frames.len() - 1;
        let (function, mut arguments) =
            call_target_and_fixed_arguments(frames.at(frame_index).registers, instruction)?;
        let open = frames[frame_index].take_open_results();
        arguments.extend(open);
        self.dispatch_call(frames, instruction, function, arguments)
    }

    /// Replaces a fixed closure tail call in place when no observable frame
    /// state needs the generic close-and-rebuild path.
    #[inline(never)]
    pub(crate) fn try_fixed_tail_call(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<bool> {
        let profiling = self.interpreter_profiling;
        if profiling {
            bump(&self.interpreter_counters.tail_fixed_attempts);
        }
        let frame_index = frames.len() - 1;
        // A frame with varargs stands above them on the value stack: the
        // general tail call, which rebuilds the frame, drops them.
        if !self.debug_hooks.is_empty()
            || frames[frame_index].has_vararg_state()
            || frames.open_top() > frames[frame_index].base
            || frames[frame_index]
                .extras_ref()
                .is_some_and(|extras| extras.suppress_debug_hooks || !extras.close_slots.is_empty())
        {
            if profiling {
                bump(&self.interpreter_counters.tail_fixed_miss_observed);
            }
            return Ok(false);
        }
        let base = usize::from(instruction.a());
        let argument_count = usize::from(instruction.b());
        let Some(&Value::Closure(function_handle)) = frames.at(frame_index).registers.get(base)
        else {
            if matches!(
                frames.at(frame_index).registers.get(base),
                Some(Value::NativeFunction(_))
            ) {
                // A native runs as a call with an open result count; the
                // `ReturnOpen` that follows returns its values. The caller
                // has advanced the program counter.
                if self.native_tail_call(frames, instruction)? {
                    return Ok(true);
                }
            }
            if profiling {
                bump(&self.interpreter_counters.tail_fixed_miss_callee);
            }
            return Ok(false);
        };
        if frames
            .at(frame_index)
            .registers
            .get(base + 1..base + 1 + argument_count)
            .is_none()
        {
            if profiling {
                bump(&self.interpreter_counters.tail_fixed_miss_callee);
            }
            return Ok(false);
        }
        if matches!(
            &frames.at(frame_index).caller,
            Some(ReturnTarget::Direct(CallReturn { results: 1, .. }))
        ) && self.try_inline_numeric_call(
            frames,
            instruction.a(),
            &Value::Closure(function_handle),
            InlineCallArguments::Registers {
                start: base + 1,
                count: argument_count,
            },
        )? {
            let value = frames.at_mut(frame_index).registers[usize::from(instruction.a())].clone();
            let completed = self.finish_single_return_frame(frames, value)?;
            debug_assert!(completed.is_none());
            if profiling {
                bump(&self.interpreter_counters.tail_fixed_fast);
            }
            return Ok(true);
        }
        // The replaced frame borrows the callee's chunk and cells, so a tail
        // hop does no reference counting; it roots `function_handle` once
        // `frame.closure` is set below.
        let (callee_chunk, callee_cells, callee_environment) =
            self.borrow_closure_call(function_handle)?;
        if callee_chunk.is_vararg {
            if profiling {
                bump(&self.interpreter_counters.tail_fixed_miss_vararg);
            }
            return Ok(false);
        }
        let call_site = active_call_site(&frames[frame_index]);
        let parameter_count = usize::from(callee_chunk.parameter_count);
        let required = usize::from(callee_chunk.register_count);
        {
            let frame = frames.at_mut(frame_index);
            let supplied = parameter_count.min(argument_count);
            for destination in 0..supplied {
                let source = base + 1 + destination;
                let value = std::mem::replace(&mut frame.registers[source], Value::Nil);
                frame.registers[destination] = value;
            }
            let cleared = frame.registers.len();
            for register in &mut frame.registers[supplied..cleared] {
                *register = Value::Nil;
            }
        }
        frames.resize_top(required);
        let frame = &mut frames[frame_index];
        frame.clear_plain_mark();
        frame.chunk = callee_chunk;
        frame.closure = Some(function_handle);
        frame.environment = callee_environment;
        frame.shape = crate::runtime::NO_SHAPE;
        frame.upvalues = callee_cells;
        // The fast path above refuses frames with captured or close slots,
        // so these collections are always empty here. Frames without
        // interior state skip the reset entirely: every cleared field
        // already holds its reset value.
        frame.reset_extras();
        frame.pc = 0;
        frame.last_interpreted_pc = crate::vm::frame::LastPc::NONE;
        // A site the frame already holds is overwritten in place.
        match (call_site, &mut frame.call_site) {
            (Some(site), Some(held)) => **held = crate::vm::FrameSite::Named(site),
            (site, held) => *held = site.map(crate::vm::FrameSite::boxed),
        }
        frame.tail_calls = frame.tail_calls.saturating_add(1);
        if profiling {
            bump(&self.interpreter_counters.tail_fixed_fast);
        }
        self.note_tail_landing(frame_index, frame);
        Ok(true)
    }

    /// Tells the dispatch loop, in tracing mode, that a Lua tail call just
    /// replaced the frame at `depth`, `frame`, which now stands at pc 0 of
    /// its new chunk, when that pc is a header the policy may own (see
    /// `JitState::tail_landing`): the frame landed on the chunk a few tail
    /// calls ago too, so its tail calls go round a cycle. A chain that only
    /// passes through (`return f(x)` at the end of a function, a wrapper
    /// that calls a wrapper) costs no lookup and starts no recording that
    /// would only meet the callee's return. Interpreter mode reads nothing
    /// of `frame`.
    // Out of line, interpreter mode would pay a call on every tail call.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn note_tail_landing(&mut self, depth: usize, frame: &Frame) {
        if self.jit.trace.enabled() {
            self.note_tracing_tail_landing(depth, frame.chunk.address() as usize, frame.tail_calls);
        }
    }

    /// [`Self::note_tail_landing`] in tracing mode, for the frame at `depth`
    /// that landed on the chunk at address `chunk` after `tail_calls` tail
    /// calls. A header the tracer rejected costs one compare: its landings
    /// stay out of the cycle ring and raise no flag, so a tail loop the
    /// tracer gave up on runs as it does in interpreter mode.
    #[inline]
    fn note_tracing_tail_landing(&mut self, depth: usize, chunk: usize, tail_calls: u32) {
        if self.jit.trace.tail_header_rejected(chunk) {
            return;
        }
        self.jit.tail_landing = self.jit.tail_cycles.landed(depth, chunk, tail_calls);
    }

    /// Executes the Lua-closure half of a proper tail call by replacing the
    /// active frame. Other callable kinds keep the compact fallback
    /// `ReturnOpen` emitted by the compiler, because VM-coordinated builtins
    /// can suspend through their own continuations before producing results.
    pub(crate) fn tail_call_frame(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        arguments_kind: TailCallArguments,
    ) -> VmResult<()> {
        if matches!(arguments_kind, TailCallArguments::Fixed)
            && self.try_fixed_tail_call(frames, instruction)?
        {
            return Ok(());
        }
        if self.interpreter_profiling {
            bump(&self.interpreter_counters.tail_generic_entries);
        }
        let frame_index = frames.len() - 1;
        let (mut function, mut arguments) =
            call_target_and_fixed_arguments(frames.at(frame_index).registers, instruction)?;
        match arguments_kind {
            TailCallArguments::Fixed => {}
            TailCallArguments::Vararg => {
                arguments.extend(self.frame_varargs(frames, frame_index)?);
            }
            TailCallArguments::Open => {
                let open = frames[frame_index].take_open_results();
                arguments.extend(open);
            }
        }

        let mut hops = 0_u8;
        loop {
            match function {
                Value::Closure(function_handle) => {
                    // With no observable local slots or close handlers, the
                    // resolved numeric tail callee can finish this frame
                    // directly. Other return continuations retain the normal
                    // replacement frame and open-result handling.
                    if frames[frame_index]
                        .extras_ref()
                        .is_none_or(|extras| extras.close_slots.is_empty())
                        && matches!(
                            &frames.at(frame_index).caller,
                            Some(ReturnTarget::Direct(CallReturn { results: 1, .. }))
                        )
                        && self.try_inline_numeric_call(
                            frames,
                            instruction.a(),
                            &Value::Closure(function_handle),
                            InlineCallArguments::Values(&arguments),
                        )?
                    {
                        let value = frames.at_mut(frame_index).registers
                            [usize::from(instruction.a())]
                        .clone();
                        let completed = self.finish_single_return_frame(frames, value)?;
                        debug_assert!(completed.is_none());
                        return Ok(());
                    }

                    let body = self.closure_body(function_handle)?;
                    let call_site = active_call_site(&frames[frame_index]);
                    // A proper tail call has no caller-visible pause between
                    // its old and new activations. Reuse this frame's register
                    // allocation instead of routing it through the general
                    // return pool, which otherwise scans and mutates shared
                    // pool storage for every hop in recursive tail calls.
                    // Pop before closing, as ordinary frame return does. A
                    // failing __close must unwind from the caller rather than
                    // leave this logically dead frame on the active stack.
                    // The captured locals close first: the successor starts
                    // over the registers they live in.
                    self.close_top_frame_upvalues(frames);
                    let mut previous = frames.pop().expect("tail call has an active frame");
                    let caller = previous.caller.take();
                    let tail_calls = previous.tail_calls.saturating_add(1);
                    let suppress_hooks = previous
                        .extras_ref()
                        .is_some_and(|extras| extras.suppress_debug_hooks);
                    let registers = self.close_frame_for_tail_replacement(frames, &mut previous)?;
                    let replacement = Frame::new_with_registers(
                        body.chunk(),
                        arguments,
                        body.shared_cells(),
                        caller,
                        call_site,
                        body.environment(),
                        Some(function_handle),
                        registers,
                    );
                    let mut replacement = replacement;
                    replacement.tail_calls = tail_calls;
                    if suppress_hooks {
                        replacement.extras().suppress_debug_hooks = true;
                    }
                    frames.push(replacement);
                    self.note_tail_landing(frame_index, &frames[frame_index]);
                    let event = match self.compatibility {
                        CompatibilityProfile::Lua51 | CompatibilityProfile::LuaJit21 => {
                            DebugHookEvent::Call
                        }
                        _ => DebugHookEvent::TailCall,
                    };
                    self.fire_debug_hook(frames, event)?;
                    return Ok(());
                }
                callee
                    if !matches!(
                        callee,
                        Value::NativeFunction(_)
                            | Value::CoroutineWrapper(_)
                            | Value::GmatchIterator(_)
                            | Value::BuiltinFunction(_)
                    ) =>
                {
                    if hops == MAX_INDEX_HOPS {
                        return Err(RuntimeError::message("metatable chain too long").into());
                    }
                    let Some(metatable) = self.runtime.raw_metatable(&callee)? else {
                        return self.dispatch_call(frames, instruction, callee, arguments);
                    };
                    let handler = self
                        .runtime
                        .table_get_string_ref(metatable, call_metamethod_key())
                        .cloned()
                        .unwrap_or(Value::Nil);
                    if matches!(handler, Value::Nil) {
                        return self.dispatch_call(frames, instruction, callee, arguments);
                    }
                    arguments.insert(0, callee);
                    function = handler;
                    hops += 1;
                }
                function => {
                    return self.dispatch_call_with_hops(
                        frames,
                        instruction,
                        function,
                        arguments,
                        hops,
                    );
                }
            }
        }
    }

    /// Calls `function` for the call `instruction` of the top frame. A
    /// builtin's argument error names the function after that call.
    pub(crate) fn dispatch_call(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        function: Value,
        arguments: Vec<Value>,
    ) -> VmResult<()> {
        self.dispatch_call_with_hops(frames, instruction, function, arguments, 0)
    }

    fn dispatch_call_with_hops(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        function: Value,
        arguments: Vec<Value>,
        hops: u8,
    ) -> VmResult<()> {
        let frame_index = frames.len() - 1;
        let builtin = matches!(function, Value::BuiltinFunction(_)).then(|| function.clone());
        match self.dispatch_call_value(frames, instruction, function, arguments, hops) {
            Err(error) if builtin.is_some() => {
                let function = builtin.as_ref().expect("a builtin was called");
                Err(self.builtin_call_error(frames, frame_index, function, error))
            }
            result => result,
        }
    }

    /// The error a builtin called from the frame at `frame_index` raised,
    /// worded the way its caller sees it.
    pub(crate) fn builtin_call_error(
        &self,
        frames: &CallStack,
        frame_index: usize,
        builtin: &Value,
        error: VmError,
    ) -> VmError {
        // A native continuation (a `gsub` callback, `xpcall` itself)
        // called the builtin, so no Lua call site names it. Stock
        // Lua then names it by its field in a loaded module.
        let metamethod_set = match &frames[frame_index].caller {
            Some(crate::vm::ReturnTarget::Index(continuation)) => Some(continuation.is_set()),
            Some(crate::vm::ReturnTarget::SetMetamethod) => Some(true),
            Some(crate::vm::ReturnTarget::IndexResult(_)) => Some(false),
            _ => None,
        };
        if let Some(is_set) = metamethod_set
            && let Some(parent) = frame_index.checked_sub(1).map(|i| &frames[i])
        {
            return crate::vm::errors::metamethod_argument_error(
                parent,
                is_set,
                error,
                self.compatibility,
            );
        }
        if crate::vm::debug_info::is_c_chunk(&frames[frame_index].chunk)
            && !self.compatibility.is_lua51_family()
            && let VmError::Runtime(runtime_error) = &error
            && let Some((position, detail)) = runtime_error.argument_detail()
            && let Some(name) = loaded_function_name(&self.runtime, builtin)
        {
            return RuntimeError::message(format!(
                "bad argument #{position} to '{name}' ({detail})"
            ))
            .into();
        }
        builtin_argument_error(
            &frames[frame_index],
            frame_index.checked_sub(1).map(|i| &frames[i]),
            error,
            self.compatibility,
        )
    }

    #[allow(clippy::too_many_lines)]
    fn dispatch_call_value(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        mut function: Value,
        mut arguments: Vec<Value>,
        mut hops: u8,
    ) -> VmResult<()> {
        let frame_index = frames.len() - 1;
        loop {
            if matches!(function, Value::BuiltinFunction(BuiltinFunction::Pairs))
                && !self.compatibility.is_lua51_family()
                && let Some(Value::Table(table)) = arguments.first()
                && let Some(metatable) = self.runtime.raw_metatable(&Value::Table(*table))?
            {
                let handler = self.runtime.table_get(metatable, &Value::from("__pairs"))?;
                if !matches!(handler, Value::Nil) {
                    function = handler;
                    arguments.truncate(1);
                    continue;
                }
            }
            if let Value::NativeFunction(native) = function
                && let Some(delegate) = self.string_arithmetic_delegate(native, &arguments)?
            {
                function = delegate;
                continue;
            }
            match function {
                Value::NativeFunction(function) => {
                    let results = self
                        .call_native_yieldable(frames, function, &arguments, hops == 0)
                        .map_err(|error| {
                            call_argument_error_extra(
                                &frames[frame_index],
                                frame_index.checked_sub(1).map(|i| &frames[i]),
                                error,
                                self.compatibility,
                                usize::from(hops),
                            )
                        })?;
                    if let Some(values) = self.pending_host_yield.take() {
                        self.yield_coroutine(frames, instruction, values)?;
                        return Ok(());
                    }
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                    self.observe_native_call(
                        frames,
                        Value::NativeFunction(function),
                        arguments.len(),
                    )?;
                }
                Value::CoroutineWrapper(thread) => {
                    let mut results =
                        self.resume_wrapped_coroutine_mode(frames, thread, arguments, true)?;
                    if !matches!(results.first(), Some(Value::Boolean(true))) {
                        let error = results.get(1).cloned().unwrap_or(Value::Nil);
                        // `auxwrap` prefixes a string error with the position of
                        // the Lua function that called the wrapper.
                        if matches!(error, Value::String(_)) {
                            return Err(RuntimeError::RaisedAtLevel {
                                value: error,
                                level: 1,
                            }
                            .into());
                        }
                        return Err(raised_value(error));
                    }
                    results.remove(0);
                    let frame_index = frames.len() - 1;
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                Value::GmatchIterator(iterator) => {
                    let results = self.call_gmatch_iterator(iterator)?;
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                Value::Closure(function) => {
                    ensure_call_depth(frames)?;
                    // A fixed-parameter closure takes the owned arguments
                    // straight onto the stack, with no frame built on the heap.
                    if self
                        .runtime
                        .closure_call_parts(function)
                        .is_some_and(|parts| !parts.chunk.is_vararg)
                    {
                        let caller = CallReturn {
                            base: instruction.a(),
                            results: instruction.c(),
                        };
                        let pushed = self.push_plain_closure(
                            frames,
                            function,
                            caller,
                            ActivationArguments::Values(arguments),
                        )?;
                        debug_assert!(pushed);
                        return Ok(());
                    }
                    let body = self.closure_body(function)?;
                    let call_site = active_call_site(&frames[frame_index]);
                    let frame = self.make_frame(
                        body.chunk(),
                        arguments,
                        body.shared_cells(),
                        Some(ReturnTarget::Direct(CallReturn {
                            base: instruction.a(),
                            results: instruction.c(),
                        })),
                        call_site,
                        body.environment(),
                        Some(function),
                    )?;
                    return self.enter_called_frame(frames, frame);
                }
                Value::BuiltinFunction(BuiltinFunction::ProtectedCall) => {
                    self.start_protected_call(frames, instruction, arguments)?;
                }
                Value::BuiltinFunction(BuiltinFunction::ExtendedProtectedCall) => {
                    self.start_extended_protected_call(frames, instruction, arguments)?;
                }
                Value::BuiltinFunction(BuiltinFunction::TableSort) => {
                    self.start_table_sort(frames, SortCompletion::direct(instruction), &arguments)?;
                }
                Value::BuiltinFunction(BuiltinFunction::TableForeach) => {
                    self.start_table_foreach(
                        frames,
                        ForeachCompletion::direct(instruction),
                        &arguments,
                        ForeachMode::All,
                    )?;
                }
                Value::BuiltinFunction(BuiltinFunction::TableForeachi) => {
                    self.start_table_foreach(
                        frames,
                        ForeachCompletion::direct(instruction),
                        &arguments,
                        ForeachMode::Array,
                    )?;
                }
                Value::BuiltinFunction(
                    builtin @ (BuiltinFunction::Load | BuiltinFunction::LoadString),
                ) => {
                    self.check_load_argument(builtin, &arguments)?;
                    self.start_load(frames, LoadCompletion::direct(instruction), &arguments)?;
                }
                Value::BuiltinFunction(BuiltinFunction::LoadFile) => {
                    self.start_loadfile(frames, LoadCompletion::direct(instruction), &arguments)?;
                }
                Value::BuiltinFunction(BuiltinFunction::DoFile) => {
                    self.start_dofile(frames, DofileCompletion::direct(instruction), &arguments)?;
                }
                Value::BuiltinFunction(BuiltinFunction::Require) => {
                    self.start_require(frames, RequireCompletion::direct(instruction), &arguments)?;
                }
                Value::BuiltinFunction(BuiltinFunction::Module) => {
                    self.start_module(frames, ModuleCompletion::direct(instruction), &arguments)?;
                }
                Value::BuiltinFunction(BuiltinFunction::DebugGetInfo) => {
                    let results = self.debug_get_info(frames, &arguments)?;
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                Value::BuiltinFunction(BuiltinFunction::DebugTraceback) => {
                    let results = self.debug_traceback(frames, &arguments)?;
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                Value::BuiltinFunction(BuiltinFunction::DebugSetupValue) => {
                    let results = self.debug_setupvalue(frames, &arguments)?;
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                Value::BuiltinFunction(BuiltinFunction::DebugSetHook) => {
                    // Observe the call with the hook that was active on entry.
                    // sethook may replace or remove that hook before returning.
                    let call_hook_started = self.observe_native_call(
                        frames,
                        Value::BuiltinFunction(BuiltinFunction::DebugSetHook),
                        arguments.len(),
                    )?;
                    let targets_current_thread = !matches!(arguments.first(), Some(Value::Thread(thread)) if Some(*thread) != self.current_coroutine);
                    let results = if call_hook_started {
                        self.pending_debug_sethook = Some(arguments);
                        Vec::new()
                    } else {
                        self.debug_sethook(&arguments)?
                    };
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                    if !call_hook_started && targets_current_thread {
                        let frame = &mut frames[frame_index];
                        let last_line = frame
                            .last_interpreted_pc
                            .get()
                            .and_then(|pc| frame.chunk.debug.line_at(pc));
                        let extras = frame.extras();
                        extras.last_hook_line = last_line;
                    }
                    // sethook itself is a C call. A newly installed return
                    // hook observes the completion of that call in Lua 5.1.
                    if !call_hook_started {
                        self.fire_debug_hook(frames, DebugHookEvent::Return)?;
                    }
                }
                Value::BuiltinFunction(BuiltinFunction::DebugGetHook) => {
                    let results = self.debug_gethook(&arguments);
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                Value::BuiltinFunction(BuiltinFunction::DebugGetLocal) => {
                    if let Some(error) = self.getlocal_argument_error(&arguments) {
                        return Err(error.into());
                    }
                    let results = self.debug_getlocal(frames, &arguments);
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                    self.observe_native_call(
                        frames,
                        Value::BuiltinFunction(BuiltinFunction::DebugGetLocal),
                        arguments.len(),
                    )?;
                }
                Value::BuiltinFunction(BuiltinFunction::DebugSetLocal) => {
                    let results = self.debug_setlocal(frames, &arguments)?;
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                Value::BuiltinFunction(BuiltinFunction::DebugGetUpvalue) => {
                    let results = self.debug_getupvalue(&arguments)?;
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                Value::BuiltinFunction(BuiltinFunction::DebugUpvalueId) => {
                    let results = self.debug_upvalueid(&arguments)?;
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                Value::BuiltinFunction(BuiltinFunction::DebugUpvalueJoin) => {
                    let results = self.debug_upvaluejoin(frames, &arguments)?;
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                Value::BuiltinFunction(BuiltinFunction::DebugGetRegistry) => {
                    let results = self.debug_getregistry();
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                Value::BuiltinFunction(BuiltinFunction::DebugDebug) => {
                    let results = self.debug_console(frames)?;
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                Value::BuiltinFunction(BuiltinFunction::GetFenv) => {
                    let results = self.get_fenv(frames, &arguments)?;
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                Value::BuiltinFunction(BuiltinFunction::SetFenv) => {
                    let results = self.set_fenv(frames, &arguments)?;
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                Value::BuiltinFunction(BuiltinFunction::StringDump) => {
                    let results = self.string_dump(&arguments)?;
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                Value::BuiltinFunction(BuiltinFunction::ToString) => {
                    self.start_tostring(frames, instruction, arguments)?;
                }
                Value::BuiltinFunction(BuiltinFunction::StringFormat) => {
                    self.start_string_format(frames, instruction, arguments)?;
                }
                Value::BuiltinFunction(BuiltinFunction::StringGsub) => {
                    self.start_string_gsub(frames, instruction, &arguments)?;
                }
                Value::BuiltinFunction(BuiltinFunction::CollectGarbage) => {
                    match self.collect_garbage(frames, &arguments)? {
                        CollectGarbageAction::Immediate(results) => {
                            write_call_results(
                                &mut frames.at_mut(frame_index),
                                instruction,
                                results,
                            )?;
                        }
                        CollectGarbageAction::Collected(results) => {
                            self.advance_userdata_finalizers(
                                frames,
                                FinalizerCompletion::Call {
                                    target: CallReturn {
                                        base: instruction.a(),
                                        results: instruction.c(),
                                    },
                                    values: results,
                                },
                            )?;
                        }
                    }
                }
                Value::BuiltinFunction(BuiltinFunction::CoroutineCreate) => {
                    let results = self.create_coroutine(&arguments)?;
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                Value::BuiltinFunction(BuiltinFunction::CoroutineResume) => {
                    let thread = coroutine_argument(&arguments, self.compatibility)?;
                    let results = self.resume_coroutine(frames, thread, arguments[1..].to_vec())?;
                    let frame_index = frames.len() - 1;
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                Value::BuiltinFunction(BuiltinFunction::CoroutineYield) => {
                    self.yield_coroutine(frames, instruction, arguments)?;
                }
                Value::BuiltinFunction(BuiltinFunction::CoroutineStatus) => {
                    let results = self.coroutine_status(&arguments)?;
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                Value::BuiltinFunction(BuiltinFunction::CoroutineClose) => {
                    let results = self.close_coroutine(frames, &arguments)?;
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                Value::BuiltinFunction(BuiltinFunction::CoroutineWrap) => {
                    let mut results = self.create_coroutine(&arguments)?;
                    let Value::Thread(thread) = results.remove(0) else {
                        unreachable!("coroutine.create returns a thread")
                    };
                    write_call_results(
                        &mut frames.at_mut(frame_index),
                        instruction,
                        vec![Value::CoroutineWrapper(thread)],
                    )?;
                }
                Value::BuiltinFunction(BuiltinFunction::CoroutineRunning) => {
                    let results = self.coroutine_running()?;
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                Value::BuiltinFunction(BuiltinFunction::CoroutineIsYieldable) => {
                    let results = self.coroutine_isyieldable(frames, &arguments)?;
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                Value::BuiltinFunction(
                    builtin @ (BuiltinFunction::FileWrite
                    | BuiltinFunction::FileSeek
                    | BuiltinFunction::FileRead
                    | BuiltinFunction::FileClose
                    | BuiltinFunction::FileFlush
                    | BuiltinFunction::FileSetVbuf
                    | BuiltinFunction::FileLines),
                ) => {
                    let results = self.call_file_builtin(builtin, &arguments)?;
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                // Lua 5.3 and later read the elements with `lua_geti`: a
                // table with metamethods takes the index path.
                Value::BuiltinFunction(BuiltinFunction::IPairsIterator)
                    if self.compatibility.is_lua53_or_later()
                        && matches!(
                            arguments.as_slice(),
                            [Value::Table(table), ..]
                                if self.runtime.table_needs_metamethods(*table)
                        ) =>
                {
                    let control = crate::vm::table_meta::check_integer(&arguments, 2)?;
                    let index = Value::Integer(control.wrapping_add(1));
                    let continuation = crate::vm::IndexContinuation::get(
                        arguments[0].clone(),
                        index.clone(),
                        0,
                        crate::vm::IndexCompletion::IPairs { instruction, index },
                    );
                    self.run_index(frames, continuation)?;
                }
                Value::BuiltinFunction(builtin) => {
                    let results = self.call_builtin(builtin, &arguments)?;
                    write_call_results(&mut frames.at_mut(frame_index), instruction, results)?;
                }
                callee => {
                    if hops == MAX_INDEX_HOPS {
                        return Err(RuntimeError::message("metatable chain too long").into());
                    }
                    let Some(metatable) = self.runtime.raw_metatable(&callee)? else {
                        return Err(call_type_error(
                            &frames[frame_index],
                            &callee,
                            self.compatibility,
                        )
                        .into());
                    };
                    let handler = self
                        .runtime
                        .table_get_string_ref(metatable, call_metamethod_key())
                        .cloned()
                        .unwrap_or(Value::Nil);
                    if matches!(handler, Value::Nil) {
                        return Err(call_type_error(
                            &frames[frame_index],
                            &callee,
                            self.compatibility,
                        )
                        .into());
                    }
                    arguments.insert(0, callee);
                    function = handler;
                    hops += 1;
                    continue;
                }
            }
            return Ok(());
        }
    }

    /// Records, once per closure, whether its calls can skip every
    /// interpreter inline-plan check and only push an activation. The fast
    /// loop calls closures inline only with that classification.
    #[inline]
    pub(crate) fn classify_closure_call(&mut self, closure: ClosureHandle) {
        if self
            .runtime
            .closure_call_parts(closure)
            .is_some_and(|parts| parts.call_hint == CALL_HINT_UNKNOWN)
        {
            self.classify_new_closure_call(closure);
        }
    }

    #[cold]
    #[inline(never)]
    fn classify_new_closure_call(&mut self, closure: ClosureHandle) {
        let Some((vararg, named)) = self
            .runtime
            .closure_call_parts(closure)
            .map(|parts| (parts.chunk.is_vararg, parts.chunk.arg_register.is_some()))
        else {
            return;
        };
        // A vararg body keeps its extra arguments on the value stack, which
        // the fast loop's call sets up. A named-vararg table (Lua 5.5) is
        // built by the general frame builder.
        let mut hint = if !vararg {
            CALL_HINT_PLAIN
        } else if named {
            CALL_HINT_GENERAL
        } else {
            CALL_HINT_VARARG
        };
        if self.inline_call_plan(closure).is_some() {
            hint |= CALL_HINT_NUMERIC_PLAN;
        }
        if self.inline_table_plan(closure).is_some() {
            hint |= CALL_HINT_TABLE_PLAN;
        }
        // A refusal the plan cache may retry keeps every call general. A
        // vararg body is called with a plain activation whatever its plan
        // does, so it keeps that form and has no plan to retry.
        if self.jit.inline_call_retryable_refusals.contains(&closure) {
            hint = if vararg && !named {
                CALL_HINT_VARARG
            } else {
                CALL_HINT_GENERAL
            };
        }
        self.runtime.set_closure_call_hint(closure, hint);
    }

    /// Pushes a fixed-parameter activation of `closure` directly on the value
    /// stack, reusing a retired record at that depth, and reports its call
    /// hook. Returns `false` for vararg closures, which keep the general
    /// frame builders, and for stale handles.
    #[inline]
    fn push_plain_closure(
        &mut self,
        frames: &mut CallStack,
        closure: ClosureHandle,
        caller: CallReturn,
        arguments: ActivationArguments,
    ) -> VmResult<bool> {
        let Some(parts) = self.runtime.closure_call_parts(closure) else {
            return Ok(false);
        };
        if parts.chunk.is_vararg {
            return Ok(false);
        }
        // SAFETY: `parts` are `closure`'s own, and the frame roots `closure`.
        #[allow(unsafe_code)]
        unsafe {
            frames.push_activation(
                parts.chunk,
                parts.cells,
                parts.environment,
                closure,
                caller,
                arguments,
            );
        }
        self.fire_debug_hook(frames, DebugHookEvent::Call)?;
        Ok(true)
    }

    /// Pushes a fixed-parameter Lua activation whose arguments are the
    /// caller's registers `argument_start..argument_start + argument_count`,
    /// then reports its call hook.
    #[inline(always)]
    fn push_fixed_closure_frame(
        &mut self,
        frames: &mut CallStack,
        frame: Frame,
        argument_start: usize,
        argument_count: usize,
        instruction: Instruction,
    ) -> VmResult<()> {
        let caller = frames.last().expect("a calling frame exists");
        if argument_start + argument_count > caller.registers.len() {
            return Err(VmError::RegisterRange {
                start: instruction.a(),
                count: instruction.b(),
            });
        }
        let arguments = caller.base as usize + argument_start;
        let size = usize::from(frame.chunk.register_count);
        let parameters = usize::from(frame.chunk.parameter_count);
        frames.push_with_stack_arguments(frame, size, arguments, argument_count, parameters);
        self.fire_debug_hook(frames, DebugHookEvent::Call)?;
        Ok(())
    }

    /// Pushes a new Lua activation and reports its call hook.
    #[inline(always)]
    fn enter_called_frame(&mut self, frames: &mut CallStack, frame: OwnedFrame) -> VmResult<()> {
        frames.push_within_limit(frame)?;
        self.fire_debug_hook(frames, DebugHookEvent::Call)?;
        Ok(())
    }
}

/// The next `ipairs` index, kept as a scalar so it is rebuilt into a
/// `Value` where it is used instead of being copied as bytes.
#[derive(Clone, Copy)]
pub(crate) enum IpairsIndex {
    Number(f64),
    Integer(i64),
}

impl IpairsIndex {
    #[inline(always)]
    fn value(self) -> Value {
        match self {
            Self::Number(number) => Value::Number(number),
            Self::Integer(integer) => Value::Integer(integer),
        }
    }
}
