//! Global, table, concat, and metamethod-driven instruction handlers.

use crate::bytecode::{Constant, Instruction, OpCode};
use crate::compat::CompatibilityProfile;
use crate::runtime::{
    BuiltinFunction, ClosureHandle, DirectNativeKind, NativeFunctionHandle, OperatorHandler,
    Runtime, RuntimeError, Value,
};
use crate::string::LuaString;
use crate::vm::arith::{
    bitwise_result, call_metamethod_key, coerce_lua_integer, constant_value, event, event_key,
    nonfinite_numerals, numeric_scalar_result, operator_metamethod_key, ordered_compare,
    profile_integer_value, register_form, scalar_event, scalar_string_like, table_length_integer,
};
use crate::vm::debug_info::active_diagnostic_site;
use crate::vm::errors::{
    contextual_integer_conversion_error, contextual_type_error, global_already_defined,
    index_type_error,
};
use crate::vm::registers::{global_name, jump_target, load_instruction, numeric_for_instruction};
use crate::vm::stack::{ActivationArguments, CallStack, FrameRef, HandlerKey, ensure_call_depth};
use crate::vm::{
    ConcatRangeContinuation, DebugHookEvent, Frame, IndexAction, IndexCompletion,
    IndexContinuation, MAX_INDEX_HOPS, MAX_LUA_CALL_FRAMES, OperatorMode, OperatorReturn,
    ReturnTarget, ScalarContinuation, ScalarKind, Vm, VmError, VmResult, bump,
};

/// Whether the arithmetic or bitwise `opcode` converts both operands to
/// numbers, so that no metamethod is consulted. Lua 5.4 leaves strings to
/// the string metatable's metamethods for arithmetic.
pub(crate) fn operands_coerce(
    compatibility: CompatibilityProfile,
    opcode: OpCode,
    left: &Value,
    right: &Value,
) -> bool {
    let strings_coerce = !matches!(
        compatibility,
        CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
    ) || !matches!(
        opcode,
        OpCode::Add
            | OpCode::Subtract
            | OpCode::Multiply
            | OpCode::Divide
            | OpCode::Modulo
            | OpCode::Power
            | OpCode::FloorDivide
            | OpCode::Negate
    );
    let coercible = |value: &Value| match value {
        Value::Number(_) | Value::Integer(_) => true,
        Value::String(_) => strings_coerce,
        _ => false,
    };
    coercible(left) && coercible(right)
}

/// Where a table instruction finds its key: a register, or the string
/// constant a field instruction names.
#[derive(Clone, Copy)]
enum TableKeyOperand {
    Register(u8),
    Field(u8),
}

impl TableKeyOperand {
    /// The field name as a value; `None` for a register key.
    fn field(self, frame: &FrameRef<'_>) -> Option<Value> {
        let Self::Field(index) = self else {
            return None;
        };
        match &frame.chunk.constants[usize::from(index)] {
            Constant::String(name) => Some(Value::String(name.clone())),
            _ => unreachable!("validated field names are string constants"),
        }
    }

    /// The key: the named register of `registers`, or `field` as
    /// [`Self::field`] built it.
    fn key<'a>(self, registers: &'a [Value], field: Option<&'a Value>) -> &'a Value {
        match (self, field) {
            (_, Some(field)) => field,
            (Self::Register(register), None) => &registers[usize::from(register)],
            (Self::Field(_), None) => unreachable!("field keys are built first"),
        }
    }
}

/// Finishes a test instruction at the top of a frame whose `pc` is at the
/// test's `Jump` word. A result that equals `expected` runs the jump here,
/// and any other skips it, so the next instruction is the jump's target or the
/// one after the word.
pub(crate) fn branch_after_test(frame: &mut Frame, result: bool, expected: bool) -> VmResult<()> {
    let jump_pc = frame.pc;
    if result != expected {
        frame.pc = jump_pc + 1;
        return Ok(());
    }
    let word = *frame
        .chunk
        .code
        .get(jump_pc)
        .ok_or(VmError::ProgramCounterOutOfBounds(jump_pc))?;
    if word.opcode() != Ok(OpCode::Jump) {
        // Name the opcode the word decodes to, `Jump` for an invalid one.
        return Err(VmError::InvalidDispatch(
            word.opcode().unwrap_or(OpCode::Jump),
        ));
    }
    frame.last_interpreted_pc = crate::vm::frame::LastPc::some(jump_pc);
    frame.pc = jump_target(jump_pc + 1, word.jump_offset())?;
    Ok(())
}

/// [`branch_after_test`] for the fast loop's handler return, which only asks
/// whether the branch was taken: `false`, with the frame unchanged, where
/// that routine reports an error. In line and without a result to build,
/// so the return of every `__lt`, `__le` and `__eq` handler of a test makes
/// no call.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn try_branch_after_test(frame: &mut Frame, result: bool, expected: bool) -> bool {
    let jump_pc = frame.pc;
    if result != expected {
        frame.pc = jump_pc + 1;
        return true;
    }
    let Some(&word) = frame.chunk.code.get(jump_pc) else {
        return false;
    };
    if word.opcode() != Ok(OpCode::Jump) {
        return false;
    }
    let Ok(target) = jump_target(jump_pc + 1, word.jump_offset()) else {
        return false;
    };
    frame.last_interpreted_pc = crate::vm::frame::LastPc::some(jump_pc);
    frame.pc = target;
    true
}

impl Vm {
    pub(crate) fn get_global(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<()> {
        if self.interpreter_profiling {
            bump(&self.interpreter_counters.get_global_lookups);
        }
        self.get_global_index(frames, instruction, u32::from(instruction.d()))
    }

    pub(crate) fn get_global_index(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        index: u32,
    ) -> VmResult<()> {
        let frame = frames.last().expect("global access has a frame");
        // Borrow the constant name: `table_get_string` hashes the cached
        // string hash without retaining the `Arc`, saving a retain/release
        // pair on every global read.
        let name = global_name(&frame.chunk, index)?;
        if matches!(
            self.compatibility,
            CompatibilityProfile::Lua52
                | CompatibilityProfile::Lua53
                | CompatibilityProfile::Lua54
                | CompatibilityProfile::Lua55
        ) && name.as_bytes() == b"_ENV"
        {
            let environment = frame.environment;
            frames
                .last_mut()
                .expect("global access has a frame")
                .registers[usize::from(instruction.a())] = Value::Table(environment);
            return Ok(());
        }
        let environment = frame.environment;
        // A present global resolves to its raw table value even when the
        // environment carries an `__index` metamethod, exactly as the
        // `run_index` raw-hit path below does. Short-circuit that common
        // case without building an index continuation. This deliberately
        // uses the plain raw lookup rather than `table_get_if_no_index`:
        // the missing-key fallback below is already the general path, and
        // the smaller `Value` return keeps this hot lookup tight.
        let raw = self.runtime.table_get_string(environment, name)?;
        if !matches!(raw, Value::Nil) {
            let frame = frames.last_mut().expect("global access has a frame");
            frame.registers[usize::from(instruction.a())] = raw;
            return Ok(());
        }
        let key = Value::String(name.clone());
        self.run_index(
            frames,
            IndexContinuation::get(
                Value::Table(frame.environment),
                key,
                instruction.a(),
                IndexCompletion::Frame,
            ),
        )
    }

    pub(crate) fn set_global(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<()> {
        self.set_global_index(frames, instruction, u32::from(instruction.d()))
    }

    pub(crate) fn set_global_index(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        index: u32,
    ) -> VmResult<()> {
        let frame = frames.last().expect("global access has a frame");
        let key = Value::String(global_name(&frame.chunk, index)?.clone());
        let value = frame.registers[usize::from(instruction.a())].clone();
        self.run_index(
            frames,
            IndexContinuation::set(Value::Table(frame.environment), key, value),
        )
    }

    pub(crate) fn declare_global(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<()> {
        self.declare_global_index(frames, instruction, u32::from(instruction.d()))
    }

    pub(crate) fn declare_global_index(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        index: u32,
    ) -> VmResult<()> {
        let frame = frames.last().expect("global declaration has a frame");
        let name = global_name(&frame.chunk, index)?.clone();
        let key = Value::String(name.clone());
        if !matches!(self.runtime.table_get(frame.environment, &key)?, Value::Nil) {
            return Err(global_already_defined(&name).into());
        }
        let value = frame.registers[usize::from(instruction.a())].clone();
        self.run_index(
            frames,
            IndexContinuation::set(Value::Table(frame.environment), key, value),
        )
    }

    pub(crate) fn declare_table(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<()> {
        let frame = frames.last().expect("global declaration has a frame");
        let table = match frame.registers[usize::from(instruction.a())] {
            Value::Table(table) => table,
            ref value => {
                return Err(RuntimeError::TypeError {
                    operation: "initialize global in",
                    expected: "table",
                    found: value.type_name(),
                }
                .into());
            }
        };
        let key = frame.registers[usize::from(instruction.b())].clone();
        let name = match &key {
            Value::String(name) => name.clone(),
            value => self.runtime.format_value(value)?,
        };
        if !matches!(self.runtime.table_get(table, &key)?, Value::Nil) {
            return Err(global_already_defined(&name).into());
        }
        let value = frame.registers[usize::from(instruction.c())].clone();
        self.run_index(
            frames,
            IndexContinuation::set(Value::Table(table), key, value),
        )
    }

    pub(crate) fn get_table(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<()> {
        self.get_index(
            frames,
            instruction.a(),
            instruction.b(),
            TableKeyOperand::Register(instruction.c()),
        )
    }

    pub(crate) fn get_field(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<()> {
        self.get_index(
            frames,
            instruction.a(),
            instruction.b(),
            TableKeyOperand::Field(instruction.c()),
        )
    }

    /// `SelfField`: copies register `b` to `a + 1`, then reads the field
    /// named by constant `c` from it into register `a`. The copy comes first,
    /// as Lua's SELF does it, so an `__index` function that this lookup runs
    /// finds the argument in place when it returns.
    pub(crate) fn self_field(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<()> {
        let frame = frames.last_mut().expect("method lookup has a frame");
        let receiver = frame.registers[usize::from(instruction.b())].clone();
        frame.registers[usize::from(instruction.a()) + 1] = receiver;
        self.get_index(
            frames,
            instruction.a(),
            instruction.b(),
            TableKeyOperand::Field(instruction.c()),
        )
    }

    /// `GetTabUp`: reads the field named by constant `c` from the table in
    /// upvalue `b` into register `a`. The upvalue is never copied into a
    /// register, so an `__index` function or an error leaves every register
    /// as it was.
    pub(crate) fn get_tab_up(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<()> {
        let (receiver_value, field) = {
            let frame = frames.last().expect("table access has a frame");
            let cell = *frame
                .upvalues
                .get(usize::from(instruction.b()))
                .ok_or_else(|| VmError::RegisterRange {
                    start: instruction.a(),
                    count: 1,
                })?;
            let receiver = self.runtime.cell_ref(cell)?.clone();
            (
                receiver,
                TableKeyOperand::Field(instruction.c()).field(&frame),
            )
        };
        let field = field.expect("a field operand names a constant");
        if let Value::Table(table) = receiver_value {
            let value = self.runtime.table_get_if_no_index(table, &field)?;
            if let Some(value) = value {
                let frame = frames.last_mut().expect("table access has a frame");
                frame.registers[usize::from(instruction.a())] = value;
                return Ok(());
            }
        }
        self.run_index(
            frames,
            IndexContinuation::get(
                receiver_value,
                field,
                instruction.a(),
                IndexCompletion::Frame,
            ),
        )
    }

    /// `SetTabUp`: writes register `c` to the field named by constant `b` of
    /// the table in upvalue `a`.
    pub(crate) fn set_tab_up(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<()> {
        let (receiver_value, field, value) = {
            let frame = frames.last().expect("table access has a frame");
            let cell = *frame
                .upvalues
                .get(usize::from(instruction.a()))
                .ok_or_else(|| VmError::RegisterRange {
                    start: instruction.a(),
                    count: 1,
                })?;
            let receiver = self.runtime.cell_ref(cell)?.clone();
            (
                receiver,
                TableKeyOperand::Field(instruction.b()).field(&frame),
                frame.registers[usize::from(instruction.c())].clone(),
            )
        };
        let field = field.expect("a field operand names a constant");
        if let Value::Table(table) = receiver_value
            && self
                .runtime
                .table_set_if_no_newindex(table, &field, value.clone())?
        {
            return Ok(());
        }
        self.run_index(frames, IndexContinuation::set(receiver_value, field, value))
    }

    fn get_index(
        &mut self,
        frames: &mut CallStack,
        destination: u8,
        receiver: u8,
        key: TableKeyOperand,
    ) -> VmResult<()> {
        let field = key.field(&frames.last().expect("table access has a frame"));
        {
            let frame = frames.last_mut().expect("table access has a frame");
            if let Value::Table(table) = frame.registers[usize::from(receiver)]
                && let Some(value) = self
                    .runtime
                    .table_get_raw_value(table, key.key(frame.registers, field.as_ref()))?
            {
                // Borrow from the table until the destination is written.
                // This avoids passing an owned Value through the lookup API.
                frame.registers[usize::from(destination)].clone_from(value);
                return Ok(());
            }
        }
        let raw = {
            let frame = frames.last().expect("table access has a frame");
            match frame.registers[usize::from(receiver)] {
                Value::Table(table) => self
                    .runtime
                    .table_get_if_no_index(table, key.key(frame.registers, field.as_ref()))?,
                _ => None,
            }
        };
        if let Some(value) = raw {
            let frame = frames.last_mut().expect("table access has a frame");
            frame.registers[usize::from(destination)] = value;
            return Ok(());
        }
        let (receiver_value, key, diagnostic) = {
            let frame = frames.last().expect("table access has a frame");
            let receiver_value = frame.registers[usize::from(receiver)].clone();
            // The diagnostic names the receiver in the error for indexing a
            // value that has no `__index`. `__index` chains only continue
            // through tables, so a table receiver never raises it; finding
            // the site would scan the chunk's diagnostics for nothing.
            let diagnostic = (!matches!(receiver_value, Value::Table(_)))
                .then(|| active_diagnostic_site(&frame, receiver))
                .flatten();
            (
                receiver_value,
                key.key(frame.registers, field.as_ref()).clone(),
                diagnostic,
            )
        };
        self.run_index(
            frames,
            IndexContinuation::get(receiver_value, key, destination, IndexCompletion::Frame)
                .with_diagnostic(diagnostic),
        )
    }

    pub(crate) fn set_table(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<()> {
        self.set_index(
            frames,
            instruction.a(),
            TableKeyOperand::Register(instruction.b()),
            instruction.c(),
        )
    }

    pub(crate) fn set_field(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<()> {
        self.set_index(
            frames,
            instruction.a(),
            TableKeyOperand::Field(instruction.b()),
            instruction.c(),
        )
    }

    fn set_index(
        &mut self,
        frames: &mut CallStack,
        receiver: u8,
        key: TableKeyOperand,
        source: u8,
    ) -> VmResult<()> {
        let field = key.field(&frames.last().expect("table access has a frame"));
        let written = {
            let frame = frames.last().expect("table access has a frame");
            match frame.registers[usize::from(receiver)] {
                Value::Table(table) => self.runtime.table_set_if_no_newindex(
                    table,
                    key.key(frame.registers, field.as_ref()),
                    frame.registers[usize::from(source)].clone(),
                )?,
                _ => false,
            }
        };
        if written {
            return Ok(());
        }
        let (receiver_value, key, value, diagnostic) = {
            let frame = frames.last().expect("table access has a frame");
            let receiver_value = frame.registers[usize::from(receiver)].clone();
            // As for reads, only a receiver that is not a table can raise
            // the error that names it.
            let diagnostic = (!matches!(receiver_value, Value::Table(_)))
                .then(|| active_diagnostic_site(&frame, receiver))
                .flatten();
            (
                receiver_value,
                key.key(frame.registers, field.as_ref()).clone(),
                frame.registers[usize::from(source)].clone(),
                diagnostic,
            )
        };
        self.run_index(
            frames,
            IndexContinuation::set(receiver_value, key, value).with_diagnostic(diagnostic),
        )
    }

    pub(crate) fn concat_instruction(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
    ) -> VmResult<()> {
        let frame = frames.last().expect("concatenation has a frame");
        let first = usize::from(instruction.b());
        let last = usize::from(instruction.c());
        // Only the metamethod fallback keeps the operands, so only it copies
        // them out of the registers.
        match self
            .runtime
            .concatenate_many(&frame.registers[first..=last])
        {
            Ok(value) => {
                self.finish_scalar(frames, ScalarContinuation::direct(instruction.a()), value)
            }
            Err(RuntimeError::TypeError { .. }) => {
                let values = frame.registers[first..=last].to_vec();
                let accumulator = values
                    .last()
                    .cloned()
                    .expect("validated concatenation has an operand");
                self.resume_concat_range(
                    frames,
                    ConcatRangeContinuation {
                        destination: instruction.a(),
                        first: instruction.b(),
                        last: instruction.c(),
                        remaining: values[..values.len() - 1].to_vec(),
                        accumulator,
                    },
                )
            }
            Err(error) => Err(error.into()),
        }
    }

    pub(crate) fn resume_concat_range(
        &mut self,
        frames: &mut CallStack,
        mut continuation: ConcatRangeContinuation,
    ) -> VmResult<()> {
        while let Some(left) = continuation.remaining.pop() {
            match self.runtime.concatenate(&left, &continuation.accumulator) {
                Ok(value) => continuation.accumulator = value,
                Err(error) => {
                    // Operands sit in consecutive registers from `first`.
                    // Only the first reduction's accumulator is still the
                    // last operand; later ones are results without a name.
                    let left_register = u8::try_from(continuation.remaining.len())
                        .ok()
                        .and_then(|offset| continuation.first.checked_add(offset));
                    let (bad_value, register) = if scalar_string_like(&left) {
                        let original = left_register
                            .and_then(|register| register.checked_add(1))
                            .filter(|register| *register == continuation.last);
                        (&continuation.accumulator, original)
                    } else {
                        (&left, left_register)
                    };
                    let diagnostic = register.and_then(|register| {
                        active_diagnostic_site(
                            &frames.last().expect("concatenation has a frame"),
                            register,
                        )
                    });
                    let primitive_error = contextual_type_error(
                        error,
                        bad_value,
                        diagnostic.as_ref(),
                        "concatenate",
                        self.compatibility,
                    );
                    return self.start_concat_handler(frames, continuation, left, primitive_error);
                }
            }
        }
        self.finish_scalar(
            frames,
            ScalarContinuation::direct(continuation.destination),
            continuation.accumulator,
        )
    }

    pub(crate) fn start_concat_handler(
        &mut self,
        frames: &mut CallStack,
        continuation: ConcatRangeContinuation,
        left: Value,
        primitive_error: RuntimeError,
    ) -> VmResult<()> {
        let mut arguments = vec![left, continuation.accumulator.clone()];
        let Some(mut handler) =
            self.binary_metamethod(&arguments[0], &arguments[1], event::CONCAT)?
        else {
            return Err(primitive_error.into());
        };
        let mut hops = 0_u8;
        loop {
            match handler {
                Value::NativeFunction(function) => {
                    let result = self
                        .call_native(frames, function, &arguments)?
                        .into_iter()
                        .next()
                        .unwrap_or(Value::Nil);
                    let mut continuation = continuation;
                    continuation.accumulator = result;
                    return self.resume_concat_range(frames, continuation);
                }
                Value::GmatchIterator(iterator) => {
                    let result = self
                        .call_gmatch_iterator(iterator)?
                        .into_iter()
                        .next()
                        .unwrap_or(Value::Nil);
                    let mut continuation = continuation;
                    continuation.accumulator = result;
                    return self.resume_concat_range(frames, continuation);
                }
                // A plain Lua handler reuses an activation record like an
                // ordinary call; the fast loop completes its return.
                Value::Closure(function)
                    if frames.len() < MAX_LUA_CALL_FRAMES
                        && arguments.len() == 2
                        && self.is_plain_closure(function) =>
                {
                    let mut operands = arguments.into_iter();
                    let (Some(left), Some(right)) = (operands.next(), operands.next()) else {
                        unreachable!("a concatenation handler receives two operands");
                    };
                    self.push_metamethod_activation(
                        frames,
                        function,
                        ActivationArguments::Pair(left, right),
                        ReturnTarget::ConcatRange(Box::new(continuation)),
                    )?;
                    self.fire_debug_hook(frames, DebugHookEvent::Call)?;
                    return Ok(());
                }
                Value::Closure(function) => {
                    if frames.len() >= MAX_LUA_CALL_FRAMES {
                        return Err(RuntimeError::message("stack overflow").into());
                    }
                    let body = self.closure_body(function)?;
                    let frame = self.make_frame(
                        body.chunk(),
                        arguments,
                        body.shared_cells(),
                        Some(ReturnTarget::ConcatRange(Box::new(continuation))),
                        None,
                        body.environment(),
                        Some(function),
                    )?;
                    frames.push(frame);
                    self.fire_debug_hook(frames, DebugHookEvent::Call)?;
                    return Ok(());
                }
                Value::BuiltinFunction(builtin) => {
                    let result = match self.call_builtin(builtin, &arguments) {
                        Err(VmError::InvalidDispatch(_)) => {
                            return self.push_builtin_call_frame(
                                frames,
                                builtin,
                                arguments,
                                ReturnTarget::ConcatRange(Box::new(continuation)),
                            );
                        }
                        results => results?.into_iter().next().unwrap_or(Value::Nil),
                    };
                    let mut continuation = continuation;
                    continuation.accumulator = result;
                    return self.resume_concat_range(frames, continuation);
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
                    let next = self
                        .runtime
                        .table_get_string_ref(metatable, call_metamethod_key())
                        .cloned()
                        .unwrap_or(Value::Nil);
                    if matches!(next, Value::Nil) {
                        return Err(RuntimeError::TypeError {
                            operation: "call",
                            expected: "function",
                            found: callee.type_name(),
                        }
                        .into());
                    }
                    arguments.insert(0, callee);
                    handler = next;
                    hops += 1;
                }
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    /// Converts an arithmetic operand to a number value. Lua 5.3 converts
    /// numeric strings to floats, even when the numeral parses as an integer.
    /// Lua 5.1/5.2 converts every operand to a float. Lua 5.4 leaves strings
    /// to the string metatable's arithmetic metamethods.
    fn coerce_arithmetic_operand(&self, value: &Value) -> Result<Value, RuntimeError> {
        if matches!(value, Value::String(_))
            && matches!(
                self.compatibility,
                CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
            )
        {
            Err(RuntimeError::TypeError {
                operation: "perform arithmetic",
                expected: "number",
                found: "string",
            })
        } else if self.compatibility.is_lua53_or_later() {
            let numeric = Runtime::coerce_numeric_value(value, "perform arithmetic")?;
            if matches!(value, Value::String(_)) {
                Runtime::coerce_number(&numeric, "perform arithmetic").map(Value::Number)
            } else {
                Ok(numeric)
            }
        } else {
            Runtime::coerce_number_in(
                value,
                "perform arithmetic",
                nonfinite_numerals(self.compatibility),
            )
            .map(Value::Number)
        }
    }

    /// Converts a bitwise operand to an integer. Lua 5.3 converts numeric
    /// strings; Lua 5.4 moved string coercion into the string metatable's
    /// arithmetic metamethods, which have no bitwise entries.
    fn coerce_bitwise_operand(&self, value: &Value) -> Result<i64, RuntimeError> {
        if matches!(value, Value::String(_)) && self.compatibility != CompatibilityProfile::Lua53 {
            return Err(RuntimeError::TypeError {
                operation: "perform bitwise operation",
                expected: "number",
                found: "string",
            });
        }
        coerce_lua_integer(value)
    }

    /// `Length` of a string, or of a table that has no `__len` to run: the
    /// length, stored and the frame advanced, which is all the scalar path does
    /// for it after looking for a handler. `false` when nothing was done
    /// (another operand, a handler, a recording that must see the
    /// instruction).
    #[inline(never)]
    pub(crate) fn length_in_place(
        &mut self,
        stack: &mut CallStack,
        top: usize,
        instruction: Instruction,
    ) -> bool {
        if self.jit.trace.recording() {
            return false;
        }
        let Some(frame) = stack.last() else {
            return false;
        };
        let length = match &frame.registers[usize::from(instruction.b())] {
            Value::String(text) => i64::try_from(text.len()).unwrap_or(i64::MAX),
            // Lua 5.1 and LuaJIT never consult `__len` of a table.
            &Value::Table(table) if self.compatibility.is_lua51_family() => {
                match self.runtime.table_length(table) {
                    Ok(length) => table_length_integer(length),
                    Err(_) => return false,
                }
            }
            &Value::Table(table) => match self.runtime.table_length_without_len_metamethod(table) {
                Ok(Some(length)) => table_length_integer(length),
                _ => return false,
            },
            _ => return false,
        };
        let value = profile_integer_value(self.compatibility, length);
        if let Some(frame) = stack.last_mut() {
            frame.registers[usize::from(instruction.a())] = value;
        }
        let frame = &mut stack[top];
        frame.last_interpreted_pc = crate::vm::frame::LastPc::some(frame.pc);
        frame.pc += 1;
        true
    }

    pub(crate) fn scalar_instruction(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        opcode: OpCode,
    ) -> VmResult<()> {
        // A constant-operand form behaves as its register form with the
        // constant, as `LoadConstant` loads it, in the right operand.
        let (opcode, constant) = match register_form(opcode) {
            Some(register_opcode) => {
                let frame = frames.last().expect("scalar operation has a frame");
                let constant = constant_value(
                    &frame.chunk.constants[usize::from(instruction.c())],
                    self.compatibility,
                );
                (register_opcode, Some(constant))
            }
            None => (opcode, None),
        };
        let primitive_result = {
            let frame = frames.last().expect("scalar operation has a frame");
            numeric_scalar_result(
                opcode,
                &frame.registers[usize::from(instruction.b())],
                match &constant {
                    Some(constant) => constant,
                    None => &frame.registers[usize::from(instruction.c())],
                },
                self.compatibility.float_arith(),
            )?
        };
        if let Some(result) = primitive_result {
            if self.interpreter_profiling {
                bump(&self.interpreter_counters.scalar_fast);
            }
            let frame = frames.last_mut().expect("scalar operation has a frame");
            frame.registers[usize::from(instruction.a())] = result;
            return Ok(());
        }
        if self.interpreter_profiling {
            bump(&self.interpreter_counters.scalar_slow);
        }
        if self.try_operator_metamethod(frames, instruction, opcode, constant.as_ref())? {
            return Ok(());
        }
        let frame = frames.last().expect("scalar operation has a frame");
        let left = frame.registers[usize::from(instruction.b())].clone();
        let right = if matches!(opcode, OpCode::Negate | OpCode::BitNot | OpCode::Length) {
            None
        } else if constant.is_some() {
            constant.clone()
        } else {
            Some(frame.registers[usize::from(instruction.c())].clone())
        };
        // Diagnostics name an operand in an error message. Finding one scans
        // the chunk's diagnostic sites, so only failing paths look it up.
        let diagnostic = |frames: &CallStack, register: u8| {
            active_diagnostic_site(
                &frames.last().expect("scalar operation has a frame"),
                register,
            )
        };
        let destination = instruction.a();
        match opcode {
            OpCode::Negate => match self.coerce_arithmetic_operand(&left) {
                Ok(value) => self.finish_scalar(
                    frames,
                    ScalarContinuation::direct(destination),
                    match value {
                        Value::Integer(value) => Value::Integer(value.wrapping_neg()),
                        Value::Number(value) => Value::Number(-value),
                        _ => unreachable!("arithmetic coercion yields a number"),
                    },
                ),
                Err(error) => {
                    let error = contextual_type_error(
                        error,
                        &left,
                        diagnostic(frames, instruction.b()).as_ref(),
                        "perform arithmetic on",
                        self.compatibility,
                    );
                    self.start_scalar(
                        frames,
                        ScalarContinuation::new(
                            destination,
                            ScalarKind::Value,
                            left.clone(),
                            Some(left),
                        ),
                        event::UNM,
                        Some(error),
                    )
                }
            },
            OpCode::BitNot => match self.coerce_bitwise_operand(&left) {
                Ok(value) => self.finish_scalar(
                    frames,
                    ScalarContinuation::direct(destination),
                    Value::Integer(!value),
                ),
                Err(error) => {
                    let error = contextual_integer_conversion_error(
                        error,
                        &left,
                        diagnostic(frames, instruction.b()).as_ref(),
                        "perform bitwise operation on",
                        self.compatibility,
                    );
                    self.start_scalar(
                        frames,
                        ScalarContinuation::new(destination, ScalarKind::Value, left.clone(), None),
                        event::BNOT,
                        Some(error),
                    )
                }
            },
            OpCode::Add
            | OpCode::Subtract
            | OpCode::Multiply
            | OpCode::Divide
            | OpCode::Modulo
            | OpCode::Power
            | OpCode::FloorDivide => {
                let right = right.expect("binary scalar has a right operand");
                match (
                    self.coerce_arithmetic_operand(&left),
                    self.coerce_arithmetic_operand(&right),
                ) {
                    (Ok(left_number), Ok(right_number)) => {
                        let result = numeric_scalar_result(
                            opcode,
                            &left_number,
                            &right_number,
                            self.compatibility.float_arith(),
                        )?
                        .expect("numeric operands have an arithmetic result");
                        self.finish_scalar(frames, ScalarContinuation::direct(destination), result)
                    }
                    (Err(error), _) => {
                        let error = contextual_type_error(
                            error,
                            &left,
                            diagnostic(frames, instruction.b()).as_ref(),
                            "perform arithmetic on",
                            self.compatibility,
                        );
                        self.start_scalar(
                            frames,
                            ScalarContinuation::new(
                                destination,
                                ScalarKind::Value,
                                left,
                                Some(right),
                            ),
                            scalar_event(opcode),
                            Some(error),
                        )
                    }
                    (_, Err(error)) => {
                        let error = contextual_type_error(
                            error,
                            &right,
                            if constant.is_some() {
                                None
                            } else {
                                diagnostic(frames, instruction.c())
                            }
                            .as_ref(),
                            "perform arithmetic on",
                            self.compatibility,
                        );
                        self.start_scalar(
                            frames,
                            ScalarContinuation::new(
                                destination,
                                ScalarKind::Value,
                                left,
                                Some(right),
                            ),
                            scalar_event(opcode),
                            Some(error),
                        )
                    }
                }
            }
            OpCode::BitAnd
            | OpCode::BitOr
            | OpCode::BitXor
            | OpCode::ShiftLeft
            | OpCode::ShiftRight => {
                let right = right.expect("binary scalar has a right operand");
                match (
                    self.coerce_bitwise_operand(&left),
                    self.coerce_bitwise_operand(&right),
                ) {
                    (Ok(left_integer), Ok(right_integer)) => {
                        let result = bitwise_result(opcode, left_integer, right_integer);
                        self.finish_scalar(
                            frames,
                            ScalarContinuation::direct(destination),
                            Value::Integer(result),
                        )
                    }
                    (Err(error), _) => {
                        let error = contextual_integer_conversion_error(
                            error,
                            &left,
                            diagnostic(frames, instruction.b()).as_ref(),
                            "perform bitwise operation on",
                            self.compatibility,
                        );
                        self.start_scalar(
                            frames,
                            ScalarContinuation::new(
                                destination,
                                ScalarKind::Value,
                                left,
                                Some(right),
                            ),
                            scalar_event(opcode),
                            Some(error),
                        )
                    }
                    (_, Err(error)) => {
                        let error = contextual_integer_conversion_error(
                            error,
                            &right,
                            diagnostic(frames, instruction.c()).as_ref(),
                            "perform bitwise operation on",
                            self.compatibility,
                        );
                        self.start_scalar(
                            frames,
                            ScalarContinuation::new(
                                destination,
                                ScalarKind::Value,
                                left,
                                Some(right),
                            ),
                            scalar_event(opcode),
                            Some(error),
                        )
                    }
                }
            }
            OpCode::Concat => {
                let right = right.expect("concat has a right operand");
                match self.runtime.concatenate(&left, &right) {
                    Ok(value) => {
                        self.finish_scalar(frames, ScalarContinuation::direct(destination), value)
                    }
                    Err(error) => {
                        let (bad_value, register) = if scalar_string_like(&left) {
                            (&right, instruction.c())
                        } else {
                            (&left, instruction.b())
                        };
                        let error = contextual_type_error(
                            error,
                            bad_value,
                            diagnostic(frames, register).as_ref(),
                            "concatenate",
                            self.compatibility,
                        );
                        self.start_scalar(
                            frames,
                            ScalarContinuation::new(
                                destination,
                                ScalarKind::Value,
                                left,
                                Some(right),
                            ),
                            event::CONCAT,
                            Some(error),
                        )
                    }
                }
            }
            OpCode::Equal => {
                let right = right.expect("comparison has a right operand");
                if Runtime::raw_equal(&left, &right) {
                    return self.finish_scalar(
                        frames,
                        ScalarContinuation::comparison(destination, false, left, right),
                        Value::Boolean(true),
                    );
                }
                let continuation = ScalarContinuation::comparison(destination, false, left, right);
                self.start_equal(frames, continuation)
            }
            OpCode::LessThan | OpCode::LessEqual => {
                let right = right.expect("comparison has a right operand");
                match ordered_compare(&left, &right, "compare") {
                    Ok(ordering) => {
                        let result = match opcode {
                            OpCode::LessThan => ordering.is_some_and(std::cmp::Ordering::is_lt),
                            OpCode::LessEqual => ordering.is_some_and(|value| !value.is_gt()),
                            _ => unreachable!("comparison opcode"),
                        };
                        self.finish_scalar(
                            frames,
                            ScalarContinuation::comparison(destination, false, left, right),
                            Value::Boolean(result),
                        )
                    }
                    Err(error) if opcode == OpCode::LessThan => self.start_ordered(
                        frames,
                        ScalarContinuation::comparison(destination, false, left, right),
                        event::LT,
                        error,
                    ),
                    Err(error) => self.start_less_equal(
                        frames,
                        ScalarContinuation::comparison(destination, false, left, right),
                        error,
                    ),
                }
            }
            OpCode::Length => match left {
                Value::Table(table) => {
                    let value = Value::Table(table);
                    if !self.compatibility.is_lua51_family()
                        && let Some(handler) = self.unary_metamethod(&value, event::LEN)?
                    {
                        return self.invoke_scalar_handler(
                            frames,
                            ScalarContinuation::new(
                                destination,
                                ScalarKind::Value,
                                value.clone(),
                                Some(value),
                            ),
                            handler,
                        );
                    }
                    self.finish_scalar(
                        frames,
                        ScalarContinuation::direct(destination),
                        profile_integer_value(
                            self.compatibility,
                            table_length_integer(self.runtime.table_length(table)?),
                        ),
                    )
                }
                Value::String(value) => self.finish_scalar(
                    frames,
                    ScalarContinuation::direct(destination),
                    profile_integer_value(
                        self.compatibility,
                        i64::try_from(value.len()).unwrap_or(i64::MAX),
                    ),
                ),
                value => {
                    let found = value.type_name();
                    let error = contextual_type_error(
                        RuntimeError::TypeError {
                            operation: "get length of",
                            expected: "table or string",
                            found,
                        },
                        &value,
                        diagnostic(frames, instruction.b()).as_ref(),
                        "get length of",
                        self.compatibility,
                    );
                    let right = (!self.compatibility.is_lua51_family()).then(|| value.clone());
                    self.start_scalar(
                        frames,
                        ScalarContinuation::new(destination, ScalarKind::Value, value, right),
                        event::LEN,
                        Some(error),
                    )
                }
            },
            _ => Err(VmError::InvalidDispatch(opcode)),
        }
    }

    /// Runs a compare-and-branch test: the comparison its opcode names, then
    /// the `Jump` word that follows, taken when the result equals the
    /// instruction's `expected` flag and skipped otherwise.
    ///
    /// Behaves as the comparison opcode into a temporary followed by
    /// `JumpIfFalse` (or `Not` and `JumpIfFalse`). A metamethod, including
    /// one that yields, decides the branch when it finishes.
    pub(crate) fn test_instruction(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        opcode: OpCode,
    ) -> VmResult<()> {
        let comparison = opcode
            .comparison_form()
            .ok_or(VmError::InvalidDispatch(opcode))?;
        let expected = instruction.test_expected();
        let constant_left = opcode.is_test_constant() && instruction.test_constant_left();
        // The constant, as `LoadConstant` would load it.
        let constant = if opcode.is_test_constant() {
            let frame = frames.last().expect("test has a frame");
            Some(constant_value(
                &frame.chunk.constants[usize::from(instruction.b())],
                self.compatibility,
            ))
        } else {
            None
        };
        let primitive = {
            let frame = frames.last().expect("test has a frame");
            let register = &frame.registers[usize::from(instruction.a())];
            let other = match &constant {
                Some(constant) => constant,
                None => &frame.registers[usize::from(instruction.b())],
            };
            let (left, right) = if constant_left {
                (other, register)
            } else {
                (register, other)
            };
            numeric_scalar_result(comparison, left, right, self.compatibility.float_arith())?
        };
        if let Some(result) = primitive {
            if self.interpreter_profiling {
                bump(&self.interpreter_counters.scalar_fast);
            }
            let mut frame = frames.last_mut().expect("test has a frame");
            branch_after_test(&mut frame, result.is_truthy(), expected)?;
            return Ok(());
        }
        if self.interpreter_profiling {
            bump(&self.interpreter_counters.scalar_slow);
        }
        let frame = frames.last().expect("test has a frame");
        let register = frame.registers[usize::from(instruction.a())].clone();
        let other = match constant {
            Some(constant) => constant,
            None => frame.registers[usize::from(instruction.b())].clone(),
        };
        let (left, right) = if constant_left {
            (other, register)
        } else {
            (register, other)
        };
        match comparison {
            OpCode::Equal => {
                if Runtime::raw_equal(&left, &right) {
                    return self.finish_scalar(
                        frames,
                        ScalarContinuation::branching(expected, left, right),
                        Value::Boolean(true),
                    );
                }
                self.start_equal(frames, ScalarContinuation::branching(expected, left, right))
            }
            _ => match ordered_compare(&left, &right, "compare") {
                Ok(ordering) => {
                    let result = if comparison == OpCode::LessThan {
                        ordering.is_some_and(std::cmp::Ordering::is_lt)
                    } else {
                        ordering.is_some_and(|value| !value.is_gt())
                    };
                    self.finish_scalar(
                        frames,
                        ScalarContinuation::branching(expected, left, right),
                        Value::Boolean(result),
                    )
                }
                Err(error) if comparison == OpCode::LessThan => self.start_ordered(
                    frames,
                    ScalarContinuation::branching(expected, left, right),
                    event::LT,
                    error,
                ),
                Err(error) => self.start_less_equal(
                    frames,
                    ScalarContinuation::branching(expected, left, right),
                    error,
                ),
            },
        }
    }

    /// Fuses the arithmetic-and-latch shape emitted for the common
    /// `sum = sum + i; for ...` loop body. It removes one complete trip
    /// through `step` per iteration while retaining the normal scalar path
    /// for coercions and metamethods.
    pub(crate) fn try_fused_add_numeric_for_loop(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        opcode: OpCode,
    ) -> VmResult<bool> {
        if !self.debug_hooks.is_empty() {
            return Ok(false);
        }
        let mut frame = frames.last_mut().expect("addition has an active frame");
        let mut latch_pc = frame.pc;
        let Some(mut latch) = frame.chunk.code.get(latch_pc).copied() else {
            return Ok(false);
        };
        let mut clear = None;
        if latch.opcode().ok() == Some(OpCode::ClearRange) {
            let next_pc = latch_pc + 1;
            let Some(next) = frame.chunk.code.get(next_pc).copied() else {
                return Ok(false);
            };
            if next.opcode().ok() != Some(OpCode::NumericForLoop) {
                return Ok(false);
            }
            clear = Some(latch);
            latch_pc = next_pc;
            latch = next;
        } else if latch.opcode().ok() != Some(OpCode::NumericForLoop) {
            return Ok(false);
        }
        let left = &frame.registers[usize::from(instruction.b())];
        // `AddK` behaves as `Add` with the constant as `LoadConstant` loads
        // it in the right operand.
        let constant = (opcode == OpCode::AddK).then(|| {
            constant_value(
                &frame.chunk.constants[usize::from(instruction.c())],
                self.compatibility,
            )
        });
        let right = match &constant {
            Some(constant) => constant,
            None => &frame.registers[usize::from(instruction.c())],
        };
        let Some(value) =
            numeric_scalar_result(OpCode::Add, left, right, self.compatibility.float_arith())?
        else {
            return Ok(false);
        };
        frame.registers[usize::from(instruction.a())] = value;
        if let Some(clear) = clear {
            // The compiler clears dead temporaries before each latch. Keep
            // that release point while dispatching the whole suffix together.
            load_instruction(
                &frame.frame.chunk,
                frame.registers,
                clear,
                OpCode::ClearRange,
                self.compatibility,
            )?;
        }
        // `step` has already advanced past Add. Consume the optional clear
        // and latch exactly as their standalone dispatch would, including the
        // latch's relative target.
        frame.pc = latch_pc + 1;
        frame.last_interpreted_pc = crate::vm::frame::LastPc::some(latch_pc);
        numeric_for_instruction(
            frame.registers,
            &mut frame.frame.pc,
            latch,
            OpCode::NumericForLoop,
            self.compatibility,
        )?;
        Ok(true)
    }

    pub(crate) fn start_equal(
        &mut self,
        frames: &mut CallStack,
        continuation: ScalarContinuation,
    ) -> VmResult<()> {
        let right = continuation.right.as_ref().expect("equality right operand");
        // Equality events apply only to two tables or two userdata values.
        let eligible = matches!(
            (&continuation.left, right),
            (Value::Table(_), Value::Table(_))
                | (
                    Value::File(_) | Value::Userdata(_),
                    Value::File(_) | Value::Userdata(_)
                )
        );
        let handler = if !eligible {
            None
        } else if self.compatibility.is_lua53_or_later() {
            self.first_comparison_metamethod(&continuation.left, right, event::EQ)?
        } else {
            self.shared_metamethod(&continuation.left, right, event::EQ)?
        };
        let Some(handler) = handler else {
            return self.finish_scalar(frames, continuation, Value::Boolean(false));
        };
        self.invoke_scalar_handler(frames, continuation, handler)
    }

    pub(crate) fn start_less_equal(
        &mut self,
        frames: &mut CallStack,
        continuation: ScalarContinuation,
        error: RuntimeError,
    ) -> VmResult<()> {
        if let Some(handler) = self.ordered_metamethod(
            &continuation.left,
            continuation
                .right
                .as_ref()
                .expect("comparison right operand"),
            event::LE,
        )? {
            return self.invoke_scalar_handler(frames, continuation, handler);
        }
        if self.compatibility == CompatibilityProfile::Lua55 {
            return Err(error.into());
        }
        let mut fallback = continuation;
        fallback.reverse_comparison();
        fallback.invert = true;
        self.start_ordered(frames, fallback, event::LT, error)
    }

    pub(crate) fn start_ordered(
        &mut self,
        frames: &mut CallStack,
        continuation: ScalarContinuation,
        event: u8,
        primitive_error: RuntimeError,
    ) -> VmResult<()> {
        let right = continuation
            .right
            .as_ref()
            .expect("ordered comparison has a right operand");
        let Some(handler) = self.ordered_metamethod(&continuation.left, right, event)? else {
            return Err(primitive_error.into());
        };
        self.invoke_scalar_handler(frames, continuation, handler)
    }

    /// Dispatches an arithmetic or bitwise instruction with an operand that
    /// can never coerce to a number (a table, userdata, function, ...)
    /// straight to its metamethod. The general path reaches the same
    /// handler, but only after formatting the error it raises without one.
    /// Returns `false`, having changed nothing, when that path must run.
    fn try_operator_metamethod(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        opcode: OpCode,
        constant: Option<&Value>,
    ) -> VmResult<bool> {
        let Some(key) = operator_metamethod_key(opcode) else {
            return Ok(false);
        };
        let frame = frames.last().expect("scalar operation has a frame");
        let left = &frame.registers[usize::from(instruction.b())];
        // `__unm` receives its operand twice, as Lua passes it.
        let right = if opcode == OpCode::Negate {
            left
        } else if let Some(constant) = constant {
            constant
        } else {
            &frame.registers[usize::from(instruction.c())]
        };
        if operands_coerce(self.compatibility, opcode, left, right) {
            return Ok(false);
        }
        let handler = match self.metamethod_by_key(left, key)? {
            Some(Value::NativeFunction(handler)) if self.defers_to(handler, right) => self
                .metamethod_by_key(right, key)?
                .unwrap_or(Value::NativeFunction(handler)),
            Some(handler) => handler,
            None if opcode == OpCode::Negate => return Ok(false),
            None => match self.metamethod_by_key(right, key)? {
                Some(handler) => handler,
                None => return Ok(false),
            },
        };
        if let Value::NativeFunction(native) = handler
            && let Some(value) = self.string_arithmetic_result(native, opcode, left, right)?
        {
            self.finish_scalar(frames, ScalarContinuation::direct(instruction.a()), value)?;
            return Ok(true);
        }
        let continuation = ScalarContinuation::new(
            instruction.a(),
            ScalarKind::Value,
            left.clone(),
            Some(right.clone()),
        );
        self.invoke_scalar_handler(frames, continuation, handler)?;
        Ok(true)
    }

    /// The result the string metatable's arithmetic `native` gives for
    /// operands that both convert to numbers, computed without calling it.
    /// The native's own float modulo differs from the VM's at infinity, so
    /// `%` still calls it; so does any operand that does not convert,
    /// for the native's error.
    fn string_arithmetic_result(
        &self,
        native: NativeFunctionHandle,
        opcode: OpCode,
        left: &Value,
        right: &Value,
    ) -> VmResult<Option<Value>> {
        let kind = match opcode {
            OpCode::Add => DirectNativeKind::StringAdd,
            OpCode::Subtract => DirectNativeKind::StringSub,
            OpCode::Multiply => DirectNativeKind::StringMul,
            OpCode::Divide => DirectNativeKind::StringDiv,
            OpCode::Power => DirectNativeKind::StringPow,
            OpCode::Negate => DirectNativeKind::StringUnm,
            OpCode::FloorDivide => DirectNativeKind::StringFloorDiv,
            _ => return Ok(None),
        };
        if self.runtime.direct_native(kind) != Some(native) {
            return Ok(None);
        }
        let convert = |value: &Value| match value {
            Value::Integer(_) | Value::Number(_) => Some(value.clone()),
            Value::String(_) => Runtime::coerce_numeric_value(value, "").ok(),
            _ => None,
        };
        let (Some(left), Some(right)) = (convert(left), convert(right)) else {
            return Ok(None);
        };
        match numeric_scalar_result(opcode, &left, &right, self.compatibility.float_arith()) {
            // Like `lua_arith` in the native, errors carry no position.
            Err(VmError::Runtime(RuntimeError::Message(message))) => {
                Err(RuntimeError::Raised(Value::from(message)).into())
            }
            result => result,
        }
    }

    /// Pushes a fixed-parameter Lua `closure` as the handler of
    /// `continuation`, reusing the call stack's activation records like an
    /// ordinary call; the fast loop completes its return. The continuation
    /// comes back unused when a vararg closure needs the general frame
    /// builder.
    fn push_scalar_closure(
        &mut self,
        frames: &mut CallStack,
        closure: ClosureHandle,
        continuation: ScalarContinuation,
    ) -> VmResult<Option<ScalarContinuation>> {
        if !self.is_plain_closure(closure) {
            return Ok(Some(continuation));
        }
        // The operands, as `ScalarContinuation::arguments` lists them.
        let arguments = match &continuation.right {
            Some(right) => ActivationArguments::Pair(continuation.left.clone(), right.clone()),
            None => ActivationArguments::One(continuation.left.clone()),
        };
        // The continuation target also marks the metamethod boundary that
        // a yield may not cross. Operands the handler's arguments and the
        // caller's registers hold need no box.
        let target = match OperatorReturn::of(&continuation) {
            Some(operator) => ReturnTarget::Operator(operator),
            None => ReturnTarget::Scalar(Box::new(continuation)),
        };
        self.push_metamethod_activation(frames, closure, arguments, target)?;
        self.fire_debug_hook(frames, DebugHookEvent::Call)?;
        Ok(None)
    }

    /// Whether the native handler `function` can run on its operands in
    /// place: not an embedder's host function, and not one that declines
    /// operands needing the VM (`table` functions, `print`) and so runs as a
    /// host callback.
    #[inline]
    pub(crate) fn native_runs_in_place(&self, function: NativeFunctionHandle) -> bool {
        !self.is_host_function(function)
            && self
                .runtime
                .direct_native_kind(function)
                .and_then(crate::vm::table_meta::token)
                .is_none()
    }

    /// The first result of the runtime native `function`, through its
    /// scalar entry point when it has one.
    pub(crate) fn call_runtime_native_first(
        &mut self,
        function: NativeFunctionHandle,
        arguments: &[Value],
    ) -> VmResult<Value> {
        if let Some(value) = self.runtime.call_native_scalar(function, arguments)? {
            return Ok(value);
        }
        Ok(self
            .runtime
            .call_native(function, arguments)?
            .into_iter()
            .next()
            .unwrap_or(Value::Nil))
    }

    /// Whether `closure` is a live fixed-parameter Lua function, which
    /// [`Self::push_metamethod_activation`] can enter.
    #[inline]
    pub(crate) fn is_plain_closure(&self, closure: ClosureHandle) -> bool {
        self.runtime
            .closure_call_parts(closure)
            .is_some_and(|parts| !parts.chunk.is_vararg)
    }

    /// Enters the plain Lua metamethod `closure`, returning to `target`, by
    /// reusing the call stack's activation records the way an ordinary
    /// call does instead of building a detached frame. The caller's
    /// captured locals are written back first, as for any call.
    #[inline]
    pub(crate) fn push_metamethod_activation(
        &mut self,
        frames: &mut CallStack,
        closure: ClosureHandle,
        arguments: ActivationArguments,
        target: ReturnTarget,
    ) -> VmResult<()> {
        ensure_call_depth(frames)?;
        let parts = self
            .runtime
            .closure_call_parts(closure)
            .expect("a plain closure is live");
        // SAFETY: `parts` are `closure`'s own, and the frame roots `closure`.
        #[allow(unsafe_code)]
        unsafe {
            frames.push_activation_to(
                parts.chunk,
                parts.cells,
                parts.environment,
                closure,
                target,
                arguments,
            );
        }
        Ok(())
    }

    /// [`Self::unary_metamethod`] with a prebuilt key.
    fn metamethod_by_key(&self, value: &Value, key: &LuaString) -> VmResult<Option<Value>> {
        let Some(metatable) = self.runtime.raw_metatable(value)? else {
            return Ok(None);
        };
        let handler = self.runtime.table_get_string(metatable, key)?;
        Ok((!matches!(handler, Value::Nil)).then_some(handler))
    }

    pub(crate) fn start_scalar(
        &mut self,
        frames: &mut CallStack,
        continuation: ScalarContinuation,
        event: u8,
        primitive_error: Option<RuntimeError>,
    ) -> VmResult<()> {
        let handler = match continuation.right.as_ref() {
            Some(right) => self.binary_metamethod(&continuation.left, right, event)?,
            None => self.unary_metamethod(&continuation.left, event)?,
        };
        let Some(handler) = handler else {
            return Err(primitive_error
                .expect("scalar fallback has a primitive error")
                .into());
        };
        self.invoke_scalar_handler(frames, continuation, handler)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "Callable kinds share operand storage and the scalar return continuation"
    )]
    pub(crate) fn invoke_scalar_handler(
        &mut self,
        frames: &mut CallStack,
        continuation: ScalarContinuation,
        mut handler: Value,
    ) -> VmResult<()> {
        if let Value::NativeFunction(function) = handler
            && self
                .runtime
                .direct_native_kind(function)
                .is_some_and(DirectNativeKind::is_string_arithmetic)
            && let Some(delegate) =
                self.string_arithmetic_delegate(function, &continuation.arguments())?
        {
            handler = delegate;
        }
        let continuation = match handler {
            Value::Closure(function) if frames.len() < MAX_LUA_CALL_FRAMES => {
                match self.push_scalar_closure(frames, function, continuation)? {
                    Some(continuation) => continuation,
                    None => return Ok(()),
                }
            }
            // A runtime native (not an embedder's host function, which needs
            // the VM's roots and frames) runs on the operands in place.
            Value::NativeFunction(function) if self.native_runs_in_place(function) => {
                let result = match &continuation.right {
                    Some(right) => self.call_runtime_native_first(
                        function,
                        &[continuation.left.clone(), right.clone()],
                    )?,
                    None => self.call_runtime_native_first(
                        function,
                        std::slice::from_ref(&continuation.left),
                    )?,
                };
                return self.finish_scalar(frames, continuation, result);
            }
            _ => continuation,
        };
        let mut arguments = continuation.arguments();
        let mut hops = 0_u8;
        loop {
            match handler {
                Value::NativeFunction(function) => {
                    let result = self
                        .call_native(frames, function, &arguments)?
                        .into_iter()
                        .next()
                        .unwrap_or(Value::Nil);
                    return self.finish_scalar(frames, continuation, result);
                }
                Value::GmatchIterator(iterator) => {
                    let result = self
                        .call_gmatch_iterator(iterator)?
                        .into_iter()
                        .next()
                        .unwrap_or(Value::Nil);
                    return self.finish_scalar(frames, continuation, result);
                }
                Value::Closure(function) => {
                    if frames.len() >= MAX_LUA_CALL_FRAMES {
                        return Err(RuntimeError::message("stack overflow").into());
                    }
                    let body = self.closure_body(function)?;
                    let frame = self.make_frame(
                        body.chunk(),
                        arguments,
                        body.shared_cells(),
                        Some(ReturnTarget::Scalar(Box::new(continuation))),
                        None,
                        body.environment(),
                        Some(function),
                    )?;
                    frames.push(frame);
                    self.fire_debug_hook(frames, DebugHookEvent::Call)?;
                    return Ok(());
                }
                Value::BuiltinFunction(builtin) => {
                    let result = match self.call_builtin(builtin, &arguments) {
                        Err(VmError::InvalidDispatch(_)) => {
                            return self.push_builtin_call_frame(
                                frames,
                                builtin,
                                arguments,
                                ReturnTarget::Scalar(Box::new(continuation)),
                            );
                        }
                        results => results?.into_iter().next().unwrap_or(Value::Nil),
                    };
                    return self.finish_scalar(frames, continuation, result);
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
                    let next = self
                        .runtime
                        .table_get_string_ref(metatable, call_metamethod_key())
                        .cloned()
                        .unwrap_or(Value::Nil);
                    if matches!(next, Value::Nil) {
                        return Err(RuntimeError::TypeError {
                            operation: "call",
                            expected: "function",
                            found: callee.type_name(),
                        }
                        .into());
                    }
                    arguments.insert(0, callee);
                    handler = next;
                    hops += 1;
                }
            }
        }
    }

    // Kept as a method beside the continuation finishers that do use `self`.
    #[allow(clippy::unused_self)]
    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn finish_scalar(
        &mut self,
        frames: &mut CallStack,
        continuation: ScalarContinuation,
        result: Value,
    ) -> VmResult<()> {
        let mut frame = frames.last_mut().expect("scalar caller exists");
        if let Some(expected) = continuation.test {
            // A test instruction: the result picks the branch.
            let truth = result.is_truthy() != continuation.invert;
            branch_after_test(&mut frame, truth, expected)?;
            return Ok(());
        }
        frame.registers[usize::from(continuation.destination)] =
            if continuation.kind == ScalarKind::Comparison {
                Value::Boolean(result.is_truthy() != continuation.invert)
            } else {
                result
            };
        Ok(())
    }

    /// The handler `value`'s metatable has for the operator metamethod
    /// numbered `event` (see [`oximond::vm::arith::event`]).
    #[inline]
    pub(crate) fn unary_metamethod(&self, value: &Value, event: u8) -> VmResult<Option<Value>> {
        self.metamethod_by_key(value, event_key(event))
    }

    pub(crate) fn binary_metamethod(
        &self,
        left: &Value,
        right: &Value,
        event: u8,
    ) -> VmResult<Option<Value>> {
        // The right operand is consulted only when the left has no handler,
        // or when the left's handler defers to it.
        match self.unary_metamethod(left, event)? {
            Some(Value::NativeFunction(handler)) if self.defers_to(handler, right) => Ok(Some(
                self.unary_metamethod(right, event)?
                    .unwrap_or(Value::NativeFunction(handler)),
            )),
            Some(handler) => Ok(Some(handler)),
            None => self.unary_metamethod(right, event),
        }
    }

    /// The metamethod a direct call of the string metatable's arithmetic
    /// `function` runs instead, as lstrlib's `trymt` does: when the
    /// operands do not both convert to numbers and the second, which is
    /// not a string, has a metamethod for the event of its own.
    pub(crate) fn string_arithmetic_delegate(
        &self,
        function: NativeFunctionHandle,
        arguments: &[Value],
    ) -> VmResult<Option<Value>> {
        let Some(event) = self
            .runtime
            .direct_native_kind(function)
            .and_then(string_arithmetic_event)
        else {
            return Ok(None);
        };
        let converts = |value: &Value| match value {
            Value::Integer(_) | Value::Number(_) => true,
            Value::String(_) => Runtime::coerce_numeric_value(value, "").is_ok(),
            _ => false,
        };
        let first = arguments.first().unwrap_or(&Value::Nil);
        // With one argument, `tonum` reads the first operand's conversion
        // back as the second.
        let second = match arguments.get(1) {
            Some(second) => second,
            None if converts(first) => return Ok(None),
            None => &Value::Nil,
        };
        if (converts(first) && converts(second)) || matches!(second, Value::String(_)) {
            return Ok(None);
        }
        // A second operand sharing this very handler would only call it
        // again; the native then raises trymt's error.
        Ok(self
            .unary_metamethod(second, event)?
            .filter(|delegate| !Runtime::raw_equal(delegate, &Value::NativeFunction(function))))
    }

    /// Whether the left operand's `handler` passes the operation to a
    /// `right` operand's own metamethod, as lstrlib's `trymt` does for
    /// the string metatable's arithmetic when `right` is not a string.
    fn defers_to(&self, handler: NativeFunctionHandle, right: &Value) -> bool {
        !matches!(right, Value::String(_))
            && self
                .runtime
                .direct_native_kind(handler)
                .is_some_and(DirectNativeKind::is_string_arithmetic)
    }

    pub(crate) fn shared_metamethod(
        &self,
        left: &Value,
        right: &Value,
        event: u8,
    ) -> VmResult<Option<Value>> {
        let Some(left) = self.unary_metamethod(left, event)? else {
            return Ok(None);
        };
        let Some(right) = self.unary_metamethod(right, event)? else {
            return Ok(None);
        };
        Ok(Runtime::raw_equal(&left, &right).then_some(left))
    }

    /// Lua 5.1 requires matching operand types and handlers for ordering.
    /// Lua 5.2 and later choose the left handler, then the right one.
    pub(crate) fn ordered_metamethod(
        &self,
        left: &Value,
        right: &Value,
        event: u8,
    ) -> VmResult<Option<Value>> {
        if self.compatibility.is_lua51_family() {
            if left.type_name() != right.type_name() {
                return Ok(None);
            }
            self.shared_metamethod(left, right, event)
        } else {
            self.first_comparison_metamethod(left, right, event)
        }
    }

    // Comparison lookup chooses a handler without arithmetic's string
    // delegation. A native function used here performs its own dispatch.
    fn first_comparison_metamethod(
        &self,
        left: &Value,
        right: &Value,
        event: u8,
    ) -> VmResult<Option<Value>> {
        match self.unary_metamethod(left, event)? {
            Some(handler) => Ok(Some(handler)),
            None => self.unary_metamethod(right, event),
        }
    }

    #[allow(clippy::too_many_lines)]
    pub(crate) fn run_index(
        &mut self,
        frames: &mut CallStack,
        mut continuation: IndexContinuation,
    ) -> VmResult<()> {
        loop {
            if continuation.hops == MAX_INDEX_HOPS {
                return Err(RuntimeError::message("metatable chain too long").into());
            }
            let raw = match continuation.receiver {
                Value::Table(table) => Some(self.runtime.table_get(table, &continuation.key)?),
                _ => None,
            };
            match continuation.action {
                IndexAction::Get { .. } if matches!(raw, Some(ref value) if !matches!(value, Value::Nil)) =>
                {
                    return self.complete_index(
                        frames,
                        continuation,
                        raw.expect("non-nil raw value"),
                    );
                }
                IndexAction::Set { ref value } if matches!(raw, Some(ref value) if !matches!(value, Value::Nil)) =>
                {
                    let Value::Table(table) = continuation.receiver else {
                        unreachable!("only tables have raw values");
                    };
                    self.runtime
                        .table_set(table, &continuation.key, value.clone())?;
                    return self.complete_index(frames, continuation, Value::Nil);
                }
                _ => {}
            }

            let metamethod =
                self.index_metamethod(&continuation.receiver, continuation.is_set())?;
            let Some(metamethod) = metamethod else {
                return match continuation.action {
                    IndexAction::Get { .. } => {
                        if let Value::Table(_) = continuation.receiver {
                            self.complete_index(frames, continuation, Value::Nil)
                        } else if let Value::File(file) = continuation.receiver {
                            let value = self.runtime.file_method(file, &continuation.key)?;
                            self.complete_index(frames, continuation, value)
                        } else {
                            Err(index_type_error(
                                &continuation.receiver,
                                continuation.diagnostic.as_ref(),
                                self.compatibility,
                            )
                            .into())
                        }
                    }
                    IndexAction::Set { .. } => {
                        if let Value::Table(table) = continuation.receiver {
                            let value = continuation.value().clone();
                            self.runtime.table_set(table, &continuation.key, value)?;
                            self.complete_index(frames, continuation, Value::Nil)
                        } else {
                            Err(index_type_error(
                                &continuation.receiver,
                                continuation.diagnostic.as_ref(),
                                self.compatibility,
                            )
                            .into())
                        }
                    }
                };
            };
            match metamethod {
                Value::Table(table) => {
                    continuation.receiver = Value::Table(table);
                    continuation.hops += 1;
                }
                // A runtime native reading a field runs on the operands in
                // place; embedders' host functions need the VM's roots.
                Value::NativeFunction(function)
                    if matches!(continuation.action, IndexAction::Get { .. })
                        && self.native_runs_in_place(function) =>
                {
                    let result = self.call_runtime_native_first(
                        function,
                        &[continuation.receiver.clone(), continuation.key.clone()],
                    )?;
                    return self.complete_index(frames, continuation, result);
                }
                Value::NativeFunction(function) => {
                    let results =
                        self.call_native(frames, function, &continuation.callback_arguments())?;
                    return self.complete_index(
                        frames,
                        continuation,
                        results.into_iter().next().unwrap_or(Value::Nil),
                    );
                }
                Value::Closure(function)
                    if matches!(continuation.action, IndexAction::Get { .. })
                        && self.is_plain_closure(function) =>
                {
                    let arguments = ActivationArguments::Pair(
                        continuation.receiver.clone(),
                        continuation.key.clone(),
                    );
                    let target = match (&continuation.action, &continuation.completion) {
                        (IndexAction::Get { destination }, IndexCompletion::Frame) => {
                            ReturnTarget::IndexResult(*destination)
                        }
                        _ => ReturnTarget::Index(Box::new(continuation)),
                    };
                    self.push_metamethod_activation(frames, function, arguments, target)?;
                    return Ok(());
                }
                Value::Closure(function) => {
                    let body = self.closure_body(function)?;
                    let arguments = continuation.callback_arguments();
                    let target = match (&continuation.action, &continuation.completion) {
                        (IndexAction::Set { .. }, IndexCompletion::Frame) => {
                            ReturnTarget::SetMetamethod
                        }
                        _ => ReturnTarget::Index(Box::new(continuation)),
                    };
                    let frame = self.make_frame(
                        body.chunk(),
                        arguments,
                        body.shared_cells(),
                        Some(target),
                        None,
                        body.environment(),
                        Some(function),
                    )?;
                    frames.push_within_limit(frame)?;
                    return Ok(());
                }
                Value::BuiltinFunction(BuiltinFunction::ProtectedCall) => {
                    return self.start_protected_index(frames, continuation);
                }
                Value::BuiltinFunction(builtin) => {
                    let arguments = continuation.callback_arguments();
                    match self.call_builtin(builtin, &arguments) {
                        Err(VmError::InvalidDispatch(_)) => {
                            return self.push_builtin_call_frame(
                                frames,
                                builtin,
                                arguments,
                                ReturnTarget::Index(Box::new(continuation)),
                            );
                        }
                        results => {
                            let result = results
                                .map_err(|error| {
                                    let caller = frames.last().expect("metamethod caller exists");
                                    crate::vm::errors::metamethod_argument_error(
                                        &caller,
                                        continuation.is_set(),
                                        error,
                                        self.compatibility,
                                    )
                                })?
                                .into_iter()
                                .next()
                                .unwrap_or(Value::Nil);
                            return self.complete_index(frames, continuation, result);
                        }
                    }
                }
                value => {
                    return Err(RuntimeError::TypeError {
                        operation: if continuation.is_set() {
                            "perform newindex with"
                        } else {
                            "perform index with"
                        },
                        expected: "table or function",
                        found: value.type_name(),
                    }
                    .into());
                }
            }
        }
    }

    /// The native function that the metamethod instruction `frames` ends in
    /// calls directly as its handler, so a traceback can show its `[C]` level.
    /// Covers the table accesses and the arithmetic and bitwise operators; a
    /// handler reached any other way has no level.
    pub(crate) fn native_metamethod_handler(&self, frames: &CallStack) -> Option<Value> {
        let frame = frames.last()?;
        let pc = frame
            .last_interpreted_pc
            .get()
            .or_else(|| frame.pc.checked_sub(1))?;
        let instruction = *frame.chunk.code.get(pc)?;
        let register = |index: u8| frame.registers.get(usize::from(index));
        let mut opcode = instruction.opcode().ok()?;
        let mut constant = None;
        if let Some(register_opcode) = register_form(opcode) {
            constant = Some(constant_value(
                frame.chunk.constants.get(usize::from(instruction.c()))?,
                self.compatibility,
            ));
            opcode = register_opcode;
        }
        let handler = match opcode {
            OpCode::GetTable
            | OpCode::GetField
            | OpCode::SelfField
            | OpCode::SetTable
            | OpCode::SetField
            | OpCode::GetTabUp
            | OpCode::SetTabUp => {
                let (event, object) = if matches!(
                    opcode,
                    OpCode::GetTable | OpCode::GetField | OpCode::SelfField | OpCode::GetTabUp
                ) {
                    (event::INDEX, instruction.b())
                } else {
                    (event::NEWINDEX, instruction.a())
                };
                let mut value = if matches!(opcode, OpCode::GetTabUp | OpCode::SetTabUp) {
                    let cell = *frame.upvalues.get(usize::from(object))?;
                    self.runtime.cell_ref(cell).ok()?.clone()
                } else {
                    register(object)?.clone()
                };
                // A table handler repeats the access on that table.
                for _ in 0..100 {
                    let handler = self.unary_metamethod(&value, event).ok()??;
                    if matches!(handler, Value::Table(_)) {
                        value = handler;
                    } else {
                        return Some(handler).filter(Self::is_native_value);
                    }
                }
                return None;
            }
            OpCode::Negate | OpCode::BitNot | OpCode::Length => {
                let event = match opcode {
                    OpCode::Negate => event::UNM,
                    OpCode::BitNot => event::BNOT,
                    _ => event::LEN,
                };
                self.unary_metamethod(register(instruction.b())?, event)
                    .ok()??
            }
            OpCode::Add
            | OpCode::Subtract
            | OpCode::Multiply
            | OpCode::Divide
            | OpCode::Modulo
            | OpCode::Power
            | OpCode::FloorDivide
            | OpCode::BitAnd
            | OpCode::BitOr
            | OpCode::BitXor
            | OpCode::ShiftLeft
            | OpCode::ShiftRight => self
                .binary_metamethod(
                    register(instruction.b())?,
                    match &constant {
                        Some(constant) => constant,
                        None => register(instruction.c())?,
                    },
                    scalar_event(opcode),
                )
                .ok()??,
            _ => return None,
        };
        Some(handler).filter(Self::is_native_value)
    }

    fn is_native_value(value: &Value) -> bool {
        matches!(
            value,
            Value::BuiltinFunction(_) | Value::NativeFunction(_) | Value::CoroutineWrapper(_)
        )
    }

    pub(crate) fn index_metamethod(&self, value: &Value, set: bool) -> VmResult<Option<Value>> {
        self.unary_metamethod(value, if set { event::NEWINDEX } else { event::INDEX })
    }

    pub(crate) fn complete_index(
        &mut self,
        frames: &mut CallStack,
        continuation: IndexContinuation,
        result: Value,
    ) -> VmResult<()> {
        match continuation.completion {
            IndexCompletion::IPairs { instruction, index } => {
                let mut frame = frames.last_mut().expect("metamethod caller exists");
                let results = if matches!(result, Value::Nil) {
                    vec![Value::Nil]
                } else {
                    vec![index, result]
                };
                crate::vm::registers::write_call_results(&mut frame, instruction, results)?;
                Ok(())
            }
            IndexCompletion::Frame => {
                let frame = frames.last_mut().expect("metamethod caller exists");
                if let IndexAction::Get { destination } = continuation.action {
                    frame.registers[usize::from(destination)] = result;
                }
                Ok(())
            }
            IndexCompletion::StringGsub {
                continuation,
                found,
            } => {
                let mut continuation = *continuation;
                let appended = Self::append_gsub_value(
                    self.runtime.number_format(),
                    &mut continuation.output,
                    &continuation.source,
                    &found,
                    result,
                );
                if appended.is_err() {
                    let completion = continuation.completion;
                    let replacement = continuation.replacement.clone();
                    return self.guard_protected_gsub(frames, completion, &replacement, appended);
                }
                self.advance_string_gsub(frames, continuation)
            }
        }
    }

    pub(crate) fn resume_index(
        &mut self,
        frames: &mut CallStack,
        continuation: IndexContinuation,
        values: &[Value],
    ) -> VmResult<()> {
        self.complete_index(
            frames,
            continuation,
            values.first().cloned().unwrap_or(Value::Nil),
        )
    }

    /// Runs `pcall` when it is itself an index metamethod.  A Lua closure
    /// cannot be called synchronously here, so its completion is retained on
    /// the frame stack just like an ordinary protected call.
    pub(crate) fn start_protected_index(
        &mut self,
        frames: &mut CallStack,
        continuation: IndexContinuation,
    ) -> VmResult<()> {
        let mut target = continuation.receiver.clone();
        let mut arguments = continuation.callback_arguments();
        arguments.remove(0);
        let mut hops = 0_u8;
        loop {
            match target {
                Value::NativeFunction(function) => {
                    let result = match self.call_native(frames, function, &arguments) {
                        Ok(_) => Value::Boolean(true),
                        Err(error @ RuntimeError::Exit(_)) => return Err(error.into()),
                        Err(_) => Value::Boolean(false),
                    };
                    return self.complete_index(frames, continuation, result);
                }
                Value::GmatchIterator(iterator) => {
                    let result = match self.call_gmatch_iterator(iterator) {
                        Ok(_) => Value::Boolean(true),
                        Err(_) => Value::Boolean(false),
                    };
                    return self.complete_index(frames, continuation, result);
                }
                Value::Closure(_) if ensure_call_depth(frames).is_err() => {
                    return self.complete_index(frames, continuation, Value::Boolean(false));
                }
                Value::Closure(function) => {
                    let body = self.closure_body(function)?;
                    let frame = self.make_frame(
                        body.chunk(),
                        arguments,
                        body.shared_cells(),
                        Some(ReturnTarget::ProtectedIndex(Box::new(continuation))),
                        None,
                        body.environment(),
                        Some(function),
                    )?;
                    frames.push_within_limit(frame)?;
                    return Ok(());
                }
                Value::BuiltinFunction(builtin) => {
                    let result = match self.call_builtin(builtin, &arguments) {
                        Ok(_) => Value::Boolean(true),
                        Err(VmError::InvalidDispatch(_)) => {
                            return self.push_builtin_call_frame(
                                frames,
                                builtin,
                                arguments,
                                ReturnTarget::ProtectedIndex(Box::new(continuation)),
                            );
                        }
                        Err(error) if error.exit_request().is_some() => return Err(error),
                        Err(_) => Value::Boolean(false),
                    };
                    return self.complete_index(frames, continuation, result);
                }
                callee => {
                    if hops == MAX_INDEX_HOPS {
                        return Err(RuntimeError::message("metatable chain too long").into());
                    }
                    let Some(metatable) = self.runtime.raw_metatable(&callee)? else {
                        return self.complete_index(frames, continuation, Value::Boolean(false));
                    };
                    let handler = self
                        .runtime
                        .table_get_string_ref(metatable, call_metamethod_key())
                        .cloned()
                        .unwrap_or(Value::Nil);
                    if matches!(handler, Value::Nil) {
                        return self.complete_index(frames, continuation, Value::Boolean(false));
                    }
                    arguments.insert(0, callee);
                    target = handler;
                    hops += 1;
                }
            }
        }
    }
}

/// The metamethod name of a string metatable arithmetic native.
const fn string_arithmetic_event(kind: DirectNativeKind) -> Option<u8> {
    Some(match kind {
        DirectNativeKind::StringAdd => event::ADD,
        DirectNativeKind::StringSub => event::SUB,
        DirectNativeKind::StringMul => event::MUL,
        DirectNativeKind::StringDiv => event::DIV,
        DirectNativeKind::StringMod => event::MOD,
        DirectNativeKind::StringPow => event::POW,
        DirectNativeKind::StringUnm => event::UNM,
        DirectNativeKind::StringFloorDiv => event::IDIV,
        _ => return None,
    })
}

/// What a value's metatable holds for an operator event, as far as entering
/// a plain Lua handler directly is concerned.
enum Pick {
    Closure(ClosureHandle),
    Absent,
    /// A handler of another kind, or a stale handle: the general path decides.
    Decline,
}

/// The handler the operands of an operator instruction choose, by the rules
/// of `binary_metamethod`, `first_comparison_metamethod` and
/// `shared_metamethod`: `Some(None)` for no handler, `None` for a handler
/// that is not a closure.
type Chosen = Option<Option<ClosureHandle>>;

impl Vm {
    fn operator_closure(&mut self, value: &Value, event: u8) -> Pick {
        let metatable = match self.runtime.raw_metatable(value) {
            Ok(Some(metatable)) => metatable,
            Ok(None) => return Pick::Absent,
            Err(_) => return Pick::Decline,
        };
        match self
            .runtime
            .operator_handler(metatable, event, event_key(event))
        {
            OperatorHandler::Absent => Pick::Absent,
            OperatorHandler::Closure(closure) => Pick::Closure(closure),
            OperatorHandler::Other => Pick::Decline,
        }
    }

    /// The left operand's handler, else the right's.
    fn first_operator_closure(&mut self, left: &Value, right: &Value, event: u8) -> Chosen {
        match self.operator_closure(left, event) {
            Pick::Closure(closure) => Some(Some(closure)),
            Pick::Decline => None,
            Pick::Absent => match self.operator_closure(right, event) {
                Pick::Closure(closure) => Some(Some(closure)),
                Pick::Decline => None,
                Pick::Absent => Some(None),
            },
        }
    }

    /// The handler both operands have, for the profiles that need them to
    /// match.
    fn shared_operator_closure(&mut self, left: &Value, right: &Value, event: u8) -> Chosen {
        let Pick::Closure(left_handler) = (match self.operator_closure(left, event) {
            Pick::Absent => return Some(None),
            pick => pick,
        }) else {
            return None;
        };
        match self.operator_closure(right, event) {
            Pick::Absent => Some(None),
            Pick::Closure(right_handler) => {
                Some((left_handler == right_handler).then_some(left_handler))
            }
            Pick::Decline => None,
        }
    }

    /// `ordered_metamethod` for closures.
    fn ordered_operator_closure(&mut self, left: &Value, right: &Value, event: u8) -> Chosen {
        if self.compatibility.is_lua51_family() {
            if left.type_name() != right.type_name() {
                return Some(None);
            }
            self.shared_operator_closure(left, right, event)
        } else {
            self.first_operator_closure(left, right, event)
        }
    }

    /// Enters the plain Lua handler of the operator instruction at the top
    /// frame's `pc`, which the fast loop could not finish, without building
    /// a continuation: the handler's activation is pushed over the frame,
    /// returning through a [`ReturnTarget::Operator`]. Covers arithmetic,
    /// bitwise operations, unary minus, comparisons and tests on tables,
    /// the length of a table, and the concatenation of two operands, with
    /// the handler choices of the general path. Returns `false` before
    /// anything changes for every other case, which that path runs.
    #[allow(unsafe_code, clippy::too_many_lines)]
    pub(crate) fn operator_miss_hot(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
        opcode: OpCode,
    ) -> bool {
        let depth = stack.len();
        // SAFETY: a frame runs, so the top record is the one at `depth - 1`.
        let (frame, window) = unsafe {
            let (frame, values) = stack.top_raw();
            (frame, values.add((*frame).base as usize))
        };
        // SAFETY: the record is live, and its `pc` is the instruction.
        let (chunk, pc) = unsafe { (&*(*frame).chunk, (*frame).pc) };
        // A register of the frame's window (the chunk is validated). The
        // handler's window lies above it, so the operands stay put.
        // SAFETY: the operands are in the frame's window.
        let register = |index: u8| -> *const Value { unsafe { window.add(usize::from(index)) } };
        // The constant operand of a `K` form, read as `LoadConstant` loads it.
        let constant;
        let compatibility = self.compatibility;
        let destination = instruction.a();
        let (first, second, closure, mode) = match opcode {
            OpCode::Add
            | OpCode::Subtract
            | OpCode::Multiply
            | OpCode::Divide
            | OpCode::Modulo
            | OpCode::Power
            | OpCode::FloorDivide
            | OpCode::BitAnd
            | OpCode::BitOr
            | OpCode::BitXor
            | OpCode::ShiftLeft
            | OpCode::ShiftRight
            | OpCode::AddK
            | OpCode::SubtractK
            | OpCode::MultiplyK
            | OpCode::DivideK
            | OpCode::ModuloK
            | OpCode::Negate => {
                let base = register_form(opcode).unwrap_or(opcode);
                let left = register(instruction.b());
                let right = if base == OpCode::Negate {
                    left
                } else if base != opcode {
                    let Some(value) = chunk.constants.get(usize::from(instruction.c())) else {
                        return false;
                    };
                    constant = constant_value(value, compatibility);
                    &raw const constant
                } else {
                    register(instruction.c())
                };
                // SAFETY: both point at live values.
                let (left_value, right_value) = unsafe { (&*left, &*right) };
                if operands_coerce(compatibility, base, left_value, right_value) {
                    return false;
                }
                let event = if base == OpCode::Negate {
                    event::UNM
                } else {
                    scalar_event(base)
                };
                let closure = match self.operator_closure(left_value, event) {
                    Pick::Closure(closure) => closure,
                    Pick::Decline => return false,
                    Pick::Absent if base == OpCode::Negate => return false,
                    Pick::Absent => match self.operator_closure(right_value, event) {
                        Pick::Closure(closure) => closure,
                        _ => return false,
                    },
                };
                (left, right, closure, OperatorMode::Value)
            }
            OpCode::Length => {
                let left = register(instruction.b());
                // SAFETY: a register of the window.
                let left_value = unsafe { &*left };
                if !matches!(left_value, Value::Table(_)) || compatibility.is_lua51_family() {
                    return false;
                }
                let Pick::Closure(closure) = self.operator_closure(left_value, event::LEN) else {
                    return false;
                };
                // `__len` receives its operand twice, as the general path
                // passes it.
                (left, left, closure, OperatorMode::Value)
            }
            OpCode::Concat => {
                if instruction.c() != instruction.b().wrapping_add(1) {
                    return false;
                }
                let (left, right) = (register(instruction.b()), register(instruction.c()));
                // SAFETY: registers of the window.
                let (left_value, right_value) = unsafe { (&*left, &*right) };
                if !matches!(left_value, Value::Table(_)) && !matches!(right_value, Value::Table(_))
                {
                    return false;
                }
                let Some(Some(closure)) =
                    self.first_operator_closure(left_value, right_value, event::CONCAT)
                else {
                    return false;
                };
                (left, right, closure, OperatorMode::Value)
            }
            OpCode::Equal
            | OpCode::LessThan
            | OpCode::LessEqual
            | OpCode::TestEqual
            | OpCode::TestLessThan
            | OpCode::TestLessEqual
            | OpCode::TestEqualK
            | OpCode::TestLessThanK
            | OpCode::TestLessEqualK => {
                let comparison = opcode.comparison_form().unwrap_or(opcode);
                let test = comparison != opcode;
                let (left, right) = if !test {
                    (register(instruction.b()), register(instruction.c()))
                } else if opcode.is_test_constant() {
                    let Some(value) = chunk.constants.get(usize::from(instruction.b())) else {
                        return false;
                    };
                    constant = constant_value(value, compatibility);
                    let (register, other) = (register(instruction.a()), &raw const constant);
                    if instruction.test_constant_left() {
                        (other, register)
                    } else {
                        (register, other)
                    }
                } else {
                    (register(instruction.a()), register(instruction.b()))
                };
                // SAFETY: registers of the window, or the constant.
                let (left_value, right_value) = unsafe { (&*left, &*right) };
                let handler = if comparison == OpCode::Equal {
                    // Only two distinct tables consult `__eq` here.
                    let (Value::Table(left_table), Value::Table(right_table)) =
                        (left_value, right_value)
                    else {
                        return false;
                    };
                    if left_table == right_table {
                        return false;
                    }
                    if compatibility.is_lua53_or_later() {
                        self.first_operator_closure(left_value, right_value, event::EQ)
                    } else {
                        self.shared_operator_closure(left_value, right_value, event::EQ)
                    }
                } else {
                    // A table never orders primitively.
                    if !matches!(left_value, Value::Table(_))
                        && !matches!(right_value, Value::Table(_))
                    {
                        return false;
                    }
                    if comparison == OpCode::LessThan {
                        self.ordered_operator_closure(left_value, right_value, event::LT)
                    } else {
                        self.ordered_operator_closure(left_value, right_value, event::LE)
                    }
                };
                let expected = instruction.test_expected();
                let mode = |invert: bool| {
                    if test {
                        OperatorMode::Branch { expected, invert }
                    } else {
                        OperatorMode::Compare { invert }
                    }
                };
                match handler {
                    Some(Some(closure)) => (left, right, closure, mode(false)),
                    // `__le` falls back to the negated `__lt` of the swapped
                    // operands, except in Lua 5.5.
                    Some(None)
                        if comparison == OpCode::LessEqual
                            && compatibility != CompatibilityProfile::Lua55 =>
                    {
                        match self.ordered_operator_closure(right_value, left_value, event::LT) {
                            Some(Some(closure)) => (right, left, closure, mode(true)),
                            _ => return false,
                        }
                    }
                    _ => return false,
                }
            }
            _ => return false,
        };
        let target = ReturnTarget::Operator(OperatorReturn { destination, mode });
        // SAFETY: `frame` is the live top record, whose `pc` is the
        // instruction, `window` its first register and `depth` the live
        // count. The operands lie in the window or in `constant`, outside
        // the handler's window, and the pushed record roots `closure`.
        let pushed = unsafe {
            stack.push_handler(
                &self.runtime,
                closure,
                target,
                first,
                HandlerKey::Register(second),
                std::ptr::null(),
                frame,
                window,
                depth,
                pc,
            )
        };
        let entered = pushed.is_some();
        #[cfg(test)]
        if entered {
            self.operator_handler_entries += 1;
        }
        entered
    }

    /// What the operator cache knows the table `value` to hold under
    /// `event`: `Absent` for a table without a metatable, the cache's entry
    /// for any other, and `None` when `value` is no live table or the cache
    /// has no valid entry. Reads what [`Self::operator_closure`] would find,
    /// without its calls.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn cached_table_operator(&self, value: &Value, event: u8) -> Option<OperatorHandler> {
        let Value::Table(table) = value else {
            return None;
        };
        match self.runtime.table(*table).ok()?.metatable {
            None => Some(OperatorHandler::Absent),
            Some(metatable) => self.runtime.operator_handler_cached(metatable, event),
        }
    }

    /// [`Self::operator_miss_hot`] for the operators of a class, decided
    /// from the operator cache alone: `+ - * / %` whose left operand is a
    /// table that has the handler (or has none, and the right one has it),
    /// and a comparison of two tables that share a metatable. Both pick the
    /// handler every profile picks: the left operand's comes first for
    /// arithmetic, and two operands of one metatable have one handler, which
    /// is what the profiles that want them to match ask for. Returns `false`
    /// before anything changes for every other case, an entry the cache
    /// lacks included, which [`Self::operator_miss_hot`] then decides.
    ///
    /// That routine decodes every operator it covers and looks each handler
    /// up through two calls. An object's operators come here on every use,
    /// so the usual ones are answered first, with a compare and a load.
    #[allow(unsafe_code)]
    pub(crate) fn operator_miss_cached(
        &mut self,
        stack: &mut CallStack,
        instruction: Instruction,
        opcode: OpCode,
    ) -> bool {
        let depth = stack.len();
        // SAFETY: a frame runs, so the top record is the one at `depth - 1`.
        let (frame, window) = unsafe {
            let (frame, values) = stack.top_raw();
            (frame, values.add((*frame).base as usize))
        };
        // SAFETY: the operands are in the frame's window (the chunk is
        // validated), below the handler's.
        let register = |index: u8| -> *const Value { unsafe { window.add(usize::from(index)) } };
        let constant;
        let (first, second, closure, mode) = match opcode {
            OpCode::Add | OpCode::Subtract | OpCode::Multiply | OpCode::Divide | OpCode::Modulo => {
                let (left, right) = (register(instruction.b()), register(instruction.c()));
                let event = scalar_event(opcode);
                // SAFETY: registers of the window.
                let (left_value, right_value) = unsafe { (&*left, &*right) };
                let closure = match self.cached_table_operator(left_value, event) {
                    Some(OperatorHandler::Closure(closure)) => closure,
                    Some(OperatorHandler::Absent) => {
                        match self.cached_table_operator(right_value, event) {
                            Some(OperatorHandler::Closure(closure)) => closure,
                            _ => return false,
                        }
                    }
                    _ => return false,
                };
                (left, right, closure, OperatorMode::Value)
            }
            OpCode::AddK
            | OpCode::SubtractK
            | OpCode::MultiplyK
            | OpCode::DivideK
            | OpCode::ModuloK => {
                let Some(base) = register_form(opcode) else {
                    return false;
                };
                let left = register(instruction.b());
                // SAFETY: a register of the window.
                let Some(OperatorHandler::Closure(closure)) =
                    self.cached_table_operator(unsafe { &*left }, scalar_event(base))
                else {
                    return false;
                };
                // SAFETY: the record is live.
                let chunk = unsafe { &*(*frame).chunk };
                let Some(value) = chunk.constants.get(usize::from(instruction.c())) else {
                    return false;
                };
                constant = constant_value(value, self.compatibility);
                (left, &raw const constant, closure, OperatorMode::Value)
            }
            OpCode::Equal
            | OpCode::LessThan
            | OpCode::LessEqual
            | OpCode::TestEqual
            | OpCode::TestLessThan
            | OpCode::TestLessEqual => {
                let comparison = opcode.comparison_form().unwrap_or(opcode);
                let test = comparison != opcode;
                let (left, right) = if test {
                    (register(instruction.a()), register(instruction.b()))
                } else {
                    (register(instruction.b()), register(instruction.c()))
                };
                // SAFETY: registers of the window.
                let (Value::Table(left_table), Value::Table(right_table)) =
                    (unsafe { &*left }, unsafe { &*right })
                else {
                    return false;
                };
                // One table equals itself without `__eq`.
                if comparison == OpCode::Equal && left_table == right_table {
                    return false;
                }
                let (Ok(left_body), Ok(right_body)) = (
                    self.runtime.table(*left_table),
                    self.runtime.table(*right_table),
                ) else {
                    return false;
                };
                let Some(metatable) = left_body.metatable else {
                    return false;
                };
                if right_body.metatable != Some(metatable) {
                    return false;
                }
                let event = match comparison {
                    OpCode::Equal => event::EQ,
                    OpCode::LessThan => event::LT,
                    _ => event::LE,
                };
                // An absent `__le` falls back to `__lt` in most profiles,
                // which the general routine works out.
                let Some(OperatorHandler::Closure(closure)) =
                    self.runtime.operator_handler_cached(metatable, event)
                else {
                    return false;
                };
                let mode = if test {
                    OperatorMode::Branch {
                        expected: instruction.test_expected(),
                        invert: false,
                    }
                } else {
                    OperatorMode::Compare { invert: false }
                };
                (left, right, closure, mode)
            }
            _ => return false,
        };
        let target = ReturnTarget::Operator(OperatorReturn {
            destination: instruction.a(),
            mode,
        });
        // SAFETY: as in `operator_miss_hot`.
        let pushed = unsafe {
            stack.push_handler(
                &self.runtime,
                closure,
                target,
                first,
                HandlerKey::Register(second),
                std::ptr::null(),
                frame,
                window,
                depth,
                (*frame).pc,
            )
        };
        let entered = pushed.is_some();
        #[cfg(test)]
        if entered {
            self.operator_handler_entries += 1;
        }
        entered
    }
}
