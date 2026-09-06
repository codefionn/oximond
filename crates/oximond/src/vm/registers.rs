//! Register-file helpers: frame value slices, varargs, result writes, loads, branches, and loops.

use crate::bytecode::{Chunk, Constant, Instruction, OpCode, ValidationError};
use crate::compat::CompatibilityProfile;
use crate::runtime::{Runtime, RuntimeError, RuntimeResult, TableHandle, TableShape, Value};
use crate::string::LuaString;
use crate::vm::arith::{constant_value, nonfinite_numerals, strict_number, usize_to_number};
use crate::vm::builtins::table_handle;
use crate::vm::errors::numeric_for_error;
use crate::vm::stack::{FrameMut, FrameRef};
use crate::vm::{Frame, VmError, VmResult};

pub(crate) fn frame_values(frame: &FrameRef<'_>, start: u8, count: u8) -> VmResult<Vec<Value>> {
    register_values(frame.registers, start, count)
}

pub(crate) fn fixed_frame_values(
    frame: &FrameRef<'_>,
    start: u8,
    count: u8,
) -> VmResult<Vec<Value>> {
    if count == 0 {
        let index = usize::from(start);
        if index < frame.registers.len() {
            Ok(Vec::new())
        } else {
            Err(VmError::RegisterRange { start, count })
        }
    } else {
        register_values(frame.registers, start, count)
    }
}

pub(crate) fn call_target_and_fixed_arguments(
    registers: &[Value],
    instruction: Instruction,
) -> VmResult<(Value, Vec<Value>)> {
    let base = usize::from(instruction.a());
    let function = registers
        .get(base)
        .cloned()
        .ok_or_else(|| VmError::RegisterRange {
            start: instruction.a(),
            count: 1,
        })?;
    let count = usize::from(instruction.b());
    let arguments = registers
        .get(base + 1..base + 1 + count)
        .map(<[Value]>::to_vec)
        .ok_or_else(|| VmError::RegisterRange {
            start: instruction.a(),
            count: instruction.b(),
        })?;
    Ok((function, arguments))
}

pub(crate) fn vararg_values_instruction(
    registers: &mut [Value],
    instruction: Instruction,
    varargs: &[Value],
) -> VmResult<()> {
    let start = usize::from(instruction.a());
    let count = usize::from(instruction.b());
    let destination =
        registers
            .get_mut(start..start + count)
            .ok_or_else(|| VmError::RegisterRange {
                start: instruction.a(),
                count: instruction.b(),
            })?;
    for (index, slot) in destination.iter_mut().enumerate() {
        *slot = varargs.get(index).cloned().unwrap_or(Value::Nil);
    }
    Ok(())
}

pub(crate) fn table_vararg_instruction(
    runtime: &mut Runtime,
    frame: &FrameRef<'_>,
    instruction: Instruction,
    varargs: &[Value],
) -> VmResult<()> {
    let table = table_handle(
        frame
            .registers
            .get(usize::from(instruction.a()))
            .ok_or_else(|| VmError::RegisterRange {
                start: instruction.a(),
                count: 1,
            })?,
        "append varargs to",
    )?;
    for (offset, value) in varargs.iter().enumerate() {
        let index = vararg_index(instruction.d(), offset)?;
        runtime.table_set(table, &Value::Number(index), value.clone())?;
    }
    Ok(())
}

pub(crate) fn table_open_instruction(
    runtime: &mut Runtime,
    frame: &mut FrameMut<'_>,
    instruction: Instruction,
) -> VmResult<()> {
    let table = table_handle(
        frame
            .registers
            .get(usize::from(instruction.a()))
            .ok_or_else(|| VmError::RegisterRange {
                start: instruction.a(),
                count: 1,
            })?,
        "append call results to",
    )?;
    for (offset, value) in frame.take_open_results().into_iter().enumerate() {
        let index = vararg_index(instruction.d(), offset)?;
        runtime.table_set(table, &Value::Number(index), value)?;
    }
    Ok(())
}

pub(crate) fn vararg_index(start: u16, offset: usize) -> VmResult<f64> {
    let index = usize::from(start)
        .checked_add(offset)
        .ok_or_else(|| RuntimeError::message("vararg table index overflow"))?;
    usize_to_number(index)
}

pub(crate) fn write_call_results(
    frame: &mut FrameMut<'_>,
    instruction: Instruction,
    results: Vec<Value>,
) -> VmResult<()> {
    if instruction.c() == 0 {
        frame.set_open_results(results);
        return Ok(());
    }
    frame.clear_open_results();
    write_fixed_results(frame.registers, instruction, &results)
}

pub(crate) fn write_scalar_call_result(
    frame: &mut FrameMut<'_>,
    instruction: Instruction,
    value: Value,
) -> VmResult<()> {
    if instruction.c() == 0 {
        frame.open_buffer().push(value);
        return Ok(());
    }
    frame.clear_open_results();
    let base = usize::from(instruction.a());
    let count = usize::from(instruction.c());
    let target =
        frame
            .registers
            .get_mut(base..base + count)
            .ok_or_else(|| VmError::RegisterRange {
                start: instruction.a(),
                count: instruction.c(),
            })?;
    let (first, rest) = target
        .split_first_mut()
        .expect("a fixed call result has at least one destination");
    *first = value;
    rest.fill(Value::Nil);
    Ok(())
}

pub(crate) fn write_fixed_results(
    registers: &mut [Value],
    instruction: Instruction,
    results: &[Value],
) -> VmResult<()> {
    let base = usize::from(instruction.a());
    let count = usize::from(instruction.c());
    let target = registers
        .get_mut(base..base + count)
        .ok_or_else(|| VmError::RegisterRange {
            start: instruction.a(),
            count: instruction.c(),
        })?;
    for (index, slot) in target.iter_mut().enumerate() {
        *slot = results.get(index).cloned().unwrap_or(Value::Nil);
    }
    Ok(())
}

/// Clones a register value without routing through the generic `Clone`
/// implementation. Every variant except `String` is a plain bitwise copy,
/// so spelling the match out lets hot register fills inline to a
/// discriminant check plus a 16-byte store instead of an out-of-line clone
/// call whose result round-trips through the stack.
#[inline(always)]
pub(crate) fn clone_register_value(value: &Value) -> Value {
    match value {
        Value::Nil => Value::Nil,
        Value::Boolean(flag) => Value::Boolean(*flag),
        Value::Integer(number) => Value::Integer(*number),
        Value::Number(number) => Value::Number(*number),
        Value::String(text) => Value::String(text.clone()),
        Value::Table(handle) => Value::Table(*handle),
        Value::File(handle) => Value::File(*handle),
        Value::Userdata(handle) => Value::Userdata(*handle),
        Value::GmatchIterator(handle) => Value::GmatchIterator(*handle),
        Value::Thread(handle) => Value::Thread(*handle),
        Value::CoroutineWrapper(handle) => Value::CoroutineWrapper(*handle),
        Value::Closure(handle) => Value::Closure(*handle),
        Value::NativeFunction(handle) => Value::NativeFunction(*handle),
        Value::BuiltinFunction(builtin) => Value::BuiltinFunction(*builtin),
        Value::Int64(value) => Value::Int64(*value),
        Value::UInt64(value) => Value::UInt64(*value),
    }
}

/// Stores `value` in a register slot.
///
/// Only strings need their previous value dropped. Skipping the drop call
/// for every other variant lets the new value go straight from machine
/// registers into the slot. Otherwise it is staged on the stack around the
/// possible drop and copied with a wide load that cannot forward from the
/// narrow stores that built it.
#[inline(always)]
pub(crate) fn set_register(slot: &mut Value, value: Value) {
    if matches!(slot, Value::String(_)) {
        *slot = value;
    } else {
        // No other variant owns a resource, so there is nothing to drop.
        std::mem::forget(std::mem::replace(slot, value));
    }
}

/// Stores a native's `result` in a register slot. `false` leaves an error in
/// `result` for the caller to take. Otherwise the caller must `forget` the
/// result: a variant with an eight-byte payload moved out bit for bit, and
/// every other variant owns no resource.
///
/// The result stays where the callee wrote it. Passing it by value copied it
/// with wide loads that could not forward from the callee's narrow tag and
/// payload stores, and the copy stalled.
#[inline(always)]
#[allow(unsafe_code)]
pub(crate) fn store_result<E>(slot: &mut Value, result: &mut Result<Value, E>) -> bool {
    match result {
        // Numbers, 64-bit integers and strings carry an eight-byte payload.
        // The tag and the payload are read as narrow as the callee stored
        // them and written back with one 16-byte store, which the register
        // move that usually follows can forward from.
        Ok(value @ Value::Number(_)) => {
            if matches!(slot, Value::String(_)) {
                *slot = Value::Nil;
            }
            // SAFETY: `slot` holds no resource after the release above, and
            // the caller forgets the source.
            unsafe { write_words(std::ptr::from_mut(slot), value) };
        }
        // A string's payload is a pointer, eight bytes only on 64-bit targets.
        #[cfg(not(target_pointer_width = "64"))]
        Ok(value @ Value::String(_)) => {
            // SAFETY: the caller forgets the source, so the string moves.
            *slot = unsafe { std::ptr::read(value) };
        }
        #[cfg(not(target_pointer_width = "64"))]
        Ok(value @ (Value::Integer(_) | Value::Int64(_) | Value::UInt64(_))) => {
            if matches!(slot, Value::String(_)) {
                *slot = Value::Nil;
            }
            // SAFETY: `slot` holds no resource after the release above, and
            // the caller forgets the source.
            unsafe { write_words(std::ptr::from_mut(slot), value) };
        }
        #[cfg(target_pointer_width = "64")]
        Ok(value @ (Value::Integer(_) | Value::Int64(_) | Value::UInt64(_) | Value::String(_))) => {
            if matches!(slot, Value::String(_)) {
                *slot = Value::Nil;
            }
            // SAFETY: `slot` holds no resource after the release above, and
            // the caller forgets the source.
            unsafe { write_words(std::ptr::from_mut(slot), value) };
        }
        Ok(value) => store_clone(slot, value),
        Err(_) => return false,
    }
    true
}

/// Copies a value whose payload is eight bytes (`Number`, `Integer`,
/// `Int64`, `UInt64`, `String`) to `destination` with one 16-byte store. `Value` is
/// `repr(C, u8)`: the tag is byte 0 and the payload is the word at offset 8.
///
/// # Safety
///
/// `destination` must be valid for a 16-byte write and hold no live
/// resource: it is overwritten without a drop.
#[inline(always)]
#[allow(unsafe_code)]
unsafe fn write_words(destination: *mut Value, value: &Value) {
    let source = std::ptr::from_ref(value);
    // SAFETY: the caller passes a variant with an eight-byte payload, so
    // byte 0 and the word at offset 8 are initialized. The destination's
    // padding bytes carry no meaning.
    unsafe {
        let tag = source.cast::<u8>().read();
        let payload = source.cast::<u8>().add(8).cast::<u64>().read();
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{__m128i, _mm_set_epi64x, _mm_storeu_si128};
            _mm_storeu_si128(
                destination.cast::<__m128i>(),
                _mm_set_epi64x(payload.cast_signed(), i64::from(tag)),
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            destination.cast::<u8>().write(tag);
            destination.cast::<u8>().add(8).cast::<u64>().write(payload);
        }
    }
}

/// Stores a clone of `source` in `slot`. Variants with an eight-byte payload
/// are written as one 16-byte store, so a later register move (a 16-byte
/// load) forwards from it instead of stalling on separate tag and payload
/// stores.
#[inline(always)]
#[allow(unsafe_code)]
pub(crate) fn store_clone_wide(slot: &mut Value, source: &Value) {
    if matches!(
        source,
        Value::Number(_) | Value::Integer(_) | Value::Int64(_) | Value::UInt64(_)
    ) {
        if matches!(slot, Value::String(_)) {
            *slot = Value::Nil;
        }
        // SAFETY: the source variant has an eight-byte payload, and `slot`
        // holds no resource after the release above (every other variant
        // owns none), so overwriting it without a drop leaks nothing.
        unsafe { write_words(std::ptr::from_mut(slot), source) };
    } else {
        store_clone(slot, source);
    }
}

/// Stores `Value::Number(number)` in `slot` as one 16-byte store built in
/// registers. [`store_clone_wide`] of a `Value` built on the stack costs a
/// tag store, a payload store and two reloads on the path to that store.
#[inline(always)]
#[allow(unsafe_code)]
pub(crate) fn store_number_wide(slot: &mut Value, number: f64) {
    store_word_wide(slot, &Value::Number(0.0), number.to_bits());
}

/// [`store_number_wide`] for `Value::Integer`.
#[inline(always)]
#[allow(unsafe_code)]
pub(crate) fn store_integer_wide(slot: &mut Value, integer: i64) {
    store_word_wide(slot, &Value::Integer(0), integer.cast_unsigned());
}

/// Writes the variant of `proto` (one with an eight-byte payload) with
/// `payload` into `slot`. `proto` is a constant, so its tag folds away.
#[inline(always)]
#[allow(unsafe_code)]
fn store_word_wide(slot: &mut Value, proto: &Value, payload: u64) {
    if matches!(slot, Value::String(_)) {
        *slot = Value::Nil;
    }
    // SAFETY: `proto` is a `Value` with an eight-byte payload, so its byte 0
    // is the initialized tag. `slot` holds no resource after the release
    // above (every other variant owns none), so overwriting it without a
    // drop leaks nothing, and the 16-byte store stays inside it. The padding
    // bytes carry no meaning.
    unsafe {
        let tag = std::ptr::from_ref(proto).cast::<u8>().read();
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{__m128i, _mm_set_epi64x, _mm_storeu_si128};
            _mm_storeu_si128(
                std::ptr::from_mut(slot).cast::<__m128i>(),
                _mm_set_epi64x(payload.cast_signed(), i64::from(tag)),
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let destination = std::ptr::from_mut(slot);
            destination.cast::<u8>().write(tag);
            destination.cast::<u8>().add(8).cast::<u64>().write(payload);
        }
    }
}

#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<LuaString>() == 8);

/// [`store_result`] for a new element at the end of `buffer`. Pushing the
/// value instead staged it on the stack for an out-of-line `push`, and the
/// 16-byte reload stalled like the by-value copy does.
#[inline(always)]
#[allow(unsafe_code)]
pub(crate) fn push_result<E>(buffer: &mut Vec<Value>, result: &mut Result<Value, E>) -> bool {
    let Ok(value) = result else {
        return false;
    };
    buffer.reserve(1);
    let slot = buffer.spare_capacity_mut()[0].as_mut_ptr();
    // SAFETY: `slot` is unused capacity, valid for a write of one `Value`
    // and owning nothing. The length grows once it holds a value.
    unsafe {
        match value {
            Value::Number(_)
            | Value::Integer(_)
            | Value::Int64(_)
            | Value::UInt64(_)
            | Value::String(_) => write_words(slot, value),
            _ => slot.write(clone_register_value(value)),
        }
        buffer.set_len(buffer.len() + 1);
    }
    true
}

/// Stores a clone of `source` in `slot`, releasing an overwritten string
/// first so that no call separates building the clone from storing it.
#[inline(always)]
pub(crate) fn store_clone(slot: &mut Value, source: &Value) {
    if matches!(slot, Value::String(_)) {
        *slot = Value::Nil;
    }
    let value = clone_register_value(source);
    std::mem::forget(std::mem::replace(slot, value));
}

/// [`store_clone`] for a source that is usually not a string: a bitwise copy
/// of any other variant is one 16-byte move, where the general clone
/// dispatches on all sixteen variants.
#[inline(always)]
#[allow(unsafe_code)]
pub(crate) fn store_clone_scalar(slot: &mut Value, source: &Value) {
    if matches!(slot, Value::String(_)) {
        *slot = Value::Nil;
    }
    if matches!(source, Value::String(_)) {
        std::mem::forget(std::mem::replace(slot, source.clone()));
    } else {
        // SAFETY: a variant other than a string owns no heap data, so a
        // bitwise copy is a valid second value, and `slot` holds no string
        // to leak after the release above.
        unsafe { copy_words(slot, source) };
    }
}

/// The closure handle of a register that holds a closure, loaded as the one
/// word it is. Matching on the value loads its two halves one after the
/// other, and a half of a word stored whole (a register written by a move)
/// does not forward from the store buffer.
#[inline(always)]
#[allow(unsafe_code)]
pub(crate) fn closure_handle(slot: &Value) -> Option<crate::runtime::ClosureHandle> {
    if !matches!(slot, Value::Closure(_)) {
        return None;
    }
    // SAFETY: the slot holds a closure, whose payload is the handle, in the
    // word at offset 8 of a `repr(C, u8)` value: its slot in the low half and
    // its generation in the high half (`native_identity` is that word; the
    // test `closure_handle_is_the_payload_word` checks it).
    let identity = unsafe {
        std::ptr::from_ref(slot)
            .cast::<u8>()
            .add(8)
            .cast::<u64>()
            .read()
    };
    Some(crate::runtime::ClosureHandle::from_native_identity(
        identity,
    ))
}

/// Copies register `source` into `destination`, like `Move`.
#[inline(always)]
#[allow(unsafe_code)]
pub(crate) fn copy_register(registers: &mut [Value], destination: usize, source: usize) {
    assert!(destination < registers.len() && source < registers.len());
    let base = registers.as_mut_ptr();
    // SAFETY: both indices were bounds-checked above.
    let (to, from) = unsafe { (base.add(destination), base.add(source)) };
    // SAFETY: both pointers are valid, aligned `Value`s inside `registers`.
    // When neither is a string the copy needs no drop or clone; `ptr::copy`
    // tolerates `to == from`. Strings take the checked path below.
    unsafe {
        if matches!(*to, Value::String(_)) | matches!(*from, Value::String(_)) {
            copy_register_string(to, from);
        } else {
            std::ptr::copy(from, to, 1);
        }
    }
}

/// [`copy_register`] as a tag byte and a payload word, for a source that a
/// narrow store wrote just before: a table handle a call or an arithmetic
/// instruction left in a register. The 16-byte load of the plain copy cannot
/// forward from those stores and waits until they retire.
#[inline(always)]
#[allow(unsafe_code, clippy::inline_always)]
pub(crate) fn copy_register_words(registers: &mut [Value], destination: usize, source: usize) {
    assert!(destination < registers.len() && source < registers.len());
    let base = registers.as_mut_ptr();
    // SAFETY: both indices were bounds-checked above.
    let (to, from) = unsafe { (base.add(destination), base.add(source)) };
    // SAFETY: both pointers are valid, aligned `Value`s inside `registers`.
    // When neither is a string the copy needs no drop or clone, and
    // `copy_words` tolerates `to == from`. Strings take the checked path.
    unsafe {
        if matches!(*to, Value::String(_)) | matches!(*from, Value::String(_)) {
            copy_register_string(to, from);
        } else {
            copy_words(to, from);
        }
    }
}

/// Moves `from` into `slot` and leaves nil behind, as a tag byte and a
/// payload word (see [`copy_words`]). `*slot = mem::replace(from, Nil)`
/// compiles to a copy of bytes 1 to 15 in two overlapping eight-byte loads,
/// and the one at offset 7 straddles the padding and the payload word that
/// narrow stores wrote, so it waits for them to retire.
#[inline(always)]
#[allow(unsafe_code)]
pub(crate) fn move_value(slot: &mut Value, from: &mut Value) {
    if matches!(*slot, Value::String(_)) | matches!(*from, Value::String(_)) {
        *slot = std::mem::replace(from, Value::Nil);
        return;
    }
    // SAFETY: both are valid, aligned `Value`s and neither is a string, so
    // `slot` owns nothing to drop and the bits moved out of `from` are not
    // dropped (nil replaces them without a drop).
    unsafe {
        copy_words(slot, from);
        std::ptr::write(from, Value::Nil);
    }
}

/// Copies the value at `from` over the one at `to` as a one-byte tag and an
/// eight-byte payload, each loaded on its own and stored on its own. One
/// 16-byte load cannot forward from the tag and payload stores that built
/// most registers (an arithmetic result, a call's returned value), and
/// stalls until they leave the store buffer; two narrow loads forward from
/// those stores, and from a 16-byte store as well.
///
/// # Safety
///
/// `from` and `to` are valid, aligned `Value`s, neither a string.
#[inline(always)]
#[allow(unsafe_code)]
unsafe fn copy_words(to: *mut Value, from: *const Value) {
    // SAFETY: the contract. `Value` is `repr(C, u8)`: the tag is byte 0 and
    // the payload, which may be partly uninitialised, the word at offset 8.
    unsafe {
        let (to, from) = (to.cast::<u8>(), from.cast::<u8>());
        let tag = from.read();
        let payload = from.add(8).cast::<std::mem::MaybeUninit<u64>>().read();
        to.write(tag);
        to.add(8)
            .cast::<std::mem::MaybeUninit<u64>>()
            .write(payload);
    }
}

/// Move where a string is involved. Out of line: it holds the Arc
/// increment and the old-string release.
#[inline(never)]
#[allow(unsafe_code)]
fn copy_register_string(to: *mut Value, from: *const Value) {
    if std::ptr::eq(to, from) {
        return;
    }
    // SAFETY: the caller passes distinct, valid, aligned pointers into one
    // register slice, so a shared and a unique reference do not alias.
    unsafe { *to = clone_register_value(&*from) };
}

#[inline(always)]
pub(crate) fn load_instruction(
    chunk: &Chunk,
    registers: &mut [Value],
    instruction: Instruction,
    opcode: OpCode,
    compatibility: CompatibilityProfile,
) -> VmResult<()> {
    let destination = usize::from(instruction.a());
    match opcode {
        OpCode::LoadNil => registers[destination] = Value::Nil,
        OpCode::ClearRange => {
            let count = instruction.b();
            let last = destination.saturating_add(usize::from(count));
            let registers =
                registers
                    .get_mut(destination..last)
                    .ok_or_else(|| VmError::RegisterRange {
                        start: instruction.a(),
                        count,
                    })?;
            // `Value::Nil` is a zero-payload enum variant. A direct loop lets
            // LLVM specialize the replacement and old-value drop for this
            // concrete type; slice `fill` otherwise retains the generic
            // clone-and-fill path in the interpreter's hot temporary clears.
            for register in registers {
                *register = Value::Nil;
            }
        }
        OpCode::LoadConstant => {
            registers[destination] = constant_value(
                &chunk.constants[usize::from(instruction.d())],
                compatibility,
            );
        }
        OpCode::Move => registers[destination] = registers[usize::from(instruction.b())].clone(),
        _ => return Err(VmError::InvalidDispatch(opcode)),
    }
    Ok(())
}

pub(crate) fn consume_wide_index(frame: &mut Frame) -> VmResult<u32> {
    let pc = frame.pc;
    let payload = *frame
        .chunk
        .code
        .get(pc)
        .ok_or(VmError::ProgramCounterOutOfBounds(pc))?;
    if payload.opcode().ok() != Some(OpCode::ExtraArgument) {
        return Err(VmError::InvalidChunk(
            ValidationError::MissingExtraArgument {
                pc: pc.saturating_sub(1),
                opcode: frame.chunk.code[pc.saturating_sub(1)]
                    .opcode()
                    .expect("executed instruction has a decoded opcode"),
            },
        ));
    }
    frame.pc += 1;
    Ok(payload.extra_argument_index())
}

pub(crate) fn load_wide_constant(
    chunk: &Chunk,
    registers: &mut [Value],
    instruction: Instruction,
    index: u32,
    compatibility: CompatibilityProfile,
) -> VmResult<()> {
    let index = usize::try_from(index).expect("u32 fits usize on supported targets");
    let constant = chunk.constants.get(index).ok_or_else(|| {
        VmError::InvalidChunk(ValidationError::ConstantOutOfBounds {
            pc: 0,
            index: u32::try_from(index).expect("wide constant index fits u32"),
            constant_count: chunk.constants.len(),
        })
    })?;
    registers[usize::from(instruction.a())] = constant_value(constant, compatibility);
    Ok(())
}

pub(crate) fn numeric_for_instruction(
    registers: &mut [Value],
    pc: &mut usize,
    instruction: Instruction,
    opcode: OpCode,
    compatibility: CompatibilityProfile,
) -> VmResult<()> {
    let base = usize::from(instruction.a());
    let offset = i32::from(i16::from_ne_bytes(instruction.d().to_ne_bytes()));
    match opcode {
        OpCode::NumericForPrep => {
            // Integer initial and step values select an integer loop even
            // when its limit is a fractional number or a numeric string.
            if compatibility.is_lua53_or_later()
                && !matches!(registers[base + 1], Value::Integer(_))
                && let (Value::Integer(_), Value::Integer(step)) =
                    (&registers[base], &registers[base + 2])
            {
                let step = *step;
                if let Ok(limit) =
                    Runtime::coerce_numeric_value(&registers[base + 1], "perform numeric for")
                {
                    let limit = match limit {
                        Value::Integer(limit) => limit,
                        Value::Number(limit) => {
                            let rounded = if step < 0 {
                                limit.ceil()
                            } else {
                                limit.floor()
                            };
                            let beyond_high = rounded >= 9_223_372_036_854_775_808.0;
                            let beyond_low =
                                rounded < -9_223_372_036_854_775_808.0 || rounded.is_nan();
                            if (beyond_high && step < 0) || (beyond_low && step > 0) {
                                *pc = jump_target(*pc, offset)?
                                    .checked_add(1)
                                    .ok_or(VmError::ProgramCounterOutOfBounds(*pc))?;
                                return Ok(());
                            }
                            if rounded.is_nan() {
                                i64::MIN
                            } else {
                                // Rust saturates out-of-range float casts, as
                                // Lua's integer loop limit clamping requires.
                                #[allow(clippy::cast_possible_truncation)]
                                let limit = rounded as i64;
                                limit
                            }
                        }
                        _ => unreachable!("numeric coercion yields a number"),
                    };
                    registers[base + 1] = Value::Integer(limit);
                }
            }
            if let (Value::Integer(control), Value::Integer(limit), Value::Integer(step)) =
                (&registers[base], &registers[base + 1], &registers[base + 2])
            {
                if *step == 0 {
                    return Err(RuntimeError::message("'for' step is zero").into());
                }
                let (control, limit, step) = (*control, *limit, *step);
                let continues = if step > 0 {
                    control <= limit
                } else {
                    control >= limit
                };
                registers[base] = Value::Integer(control);
                registers[base + 1] = Value::Integer(limit);
                registers[base + 2] = Value::Integer(step);
                if continues {
                    // The program counter already points at the body. Enter
                    // it once before its latch increments the control value.
                    // This avoids an `initial - step` sentinel, which is
                    // ambiguous at the integer bounds.
                    registers[base + 3] = Value::Integer(control);
                } else {
                    // NumericForPrep's jump target is the latch. Skip it
                    // too, otherwise an empty range could wrap its control.
                    *pc = jump_target(*pc, offset)?
                        .checked_add(1)
                        .ok_or(VmError::ProgramCounterOutOfBounds(*pc))?;
                }
                return Ok(());
            }
            let nonfinite = nonfinite_numerals(compatibility);
            let number = |index: usize, what: &str| {
                let value = &registers[base + index];
                Runtime::coerce_number_in(value, "perform numeric for", nonfinite)
                    .map_err(|_| numeric_for_error(compatibility, what, value))
            };
            // Lua 5.4 converts the limit and the step before the initial value.
            let (control, limit, step) = if matches!(
                compatibility,
                CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
            ) {
                let limit = number(1, "limit")?;
                let step = number(2, "step")?;
                (number(0, "initial value")?, limit, step)
            } else {
                (
                    number(0, "initial value")?,
                    number(1, "limit")?,
                    number(2, "step")?,
                )
            };
            registers[base] = Value::Number(control - step);
            registers[base + 1] = Value::Number(limit);
            registers[base + 2] = Value::Number(step);
            *pc = jump_target(*pc, offset)?;
        }
        OpCode::NumericForLoop => {
            if let (Value::Integer(control), Value::Integer(limit), Value::Integer(step)) =
                (&registers[base], &registers[base + 1], &registers[base + 2])
            {
                let (limit, step) = (*limit, *step);
                // Lua integer loops stop at the edge of the integer domain.
                // Wrapping here turns a final `MAX - 1, MAX` iteration into
                // `MIN <= MAX`, which starts an effectively endless loop.
                let Some(control) = control.checked_add(step) else {
                    return Ok(());
                };
                let continues = if step > 0 {
                    control <= limit
                } else {
                    control >= limit
                };
                registers[base] = Value::Integer(control);
                if continues {
                    registers[base + 3] = Value::Integer(control);
                    *pc = jump_target(*pc, offset)?;
                }
                return Ok(());
            }
            let control = strict_number(&registers[base], "perform numeric for")?;
            let limit = strict_number(&registers[base + 1], "perform numeric for")?;
            let step = strict_number(&registers[base + 2], "perform numeric for")?;
            let control = control + step;
            let continues = if step > 0.0 {
                control <= limit
            } else {
                control >= limit
            };
            registers[base] = Value::Number(control);
            if continues {
                registers[base + 3] = Value::Number(control);
                *pc = jump_target(*pc, offset)?;
            }
        }
        _ => return Err(VmError::InvalidDispatch(opcode)),
    }
    Ok(())
}

pub(crate) fn branch_instruction(
    registers: &[Value],
    pc: &mut usize,
    instruction: Instruction,
    opcode: OpCode,
) -> VmResult<()> {
    match opcode {
        OpCode::Jump => *pc = jump_target(*pc, instruction.jump_offset())?,
        OpCode::JumpIfFalse if !registers[usize::from(instruction.a())].is_truthy() => {
            *pc = jump_target(
                *pc,
                i32::from(i16::from_ne_bytes(instruction.d().to_ne_bytes())),
            )?;
        }
        OpCode::JumpIfFalse => {}
        _ => return Err(VmError::InvalidDispatch(opcode)),
    }
    Ok(())
}

/// `ForInLoop`: when the first loop variable is not nil, it becomes the
/// control value and execution jumps to the body. Only nil ends the loop.
pub(crate) fn for_in_loop_instruction(
    registers: &mut [Value],
    pc: &mut usize,
    instruction: Instruction,
) -> VmResult<()> {
    let base = usize::from(instruction.a());
    let Some(window) = registers.get_mut(base + 2..base + 4) else {
        return Err(VmError::RegisterRange {
            start: instruction.a(),
            count: 4,
        });
    };
    if !matches!(window[1], Value::Nil) {
        window[0] = window[1].clone();
        *pc = jump_target(
            *pc,
            i32::from(i16::from_ne_bytes(instruction.d().to_ne_bytes())),
        )?;
    }
    Ok(())
}

/// The shape shared by every table the `NewTable` at `pc` makes, if the
/// constructor stores two to eight distinct string keys right after it.
#[inline]
fn constructor_shape(chunk: &Chunk, pc: usize) -> Option<&std::sync::Arc<TableShape>> {
    chunk.constructor_shape(pc)?.shared(interned_shape).as_ref()
}

/// The live shapes by their sorted key lists. Constructor sites that name the
/// same keys share one shape, and so one id for the interpreter's inline
/// caches: an access site that sees tables from several sites stays
/// monomorphic.
struct InternedShapes {
    shapes: std::collections::HashMap<Vec<u8>, std::sync::Weak<TableShape>>,
    /// Size of the map when dead entries were last dropped.
    pruned_at: usize,
}

static INTERNED_SHAPES: std::sync::Mutex<Option<InternedShapes>> = std::sync::Mutex::new(None);

#[cold]
#[inline(never)]
fn interned_shape(keys: &[LuaString]) -> Option<std::sync::Arc<TableShape>> {
    let mut sorted: Vec<&[u8]> = keys.iter().map(LuaString::as_bytes).collect();
    sorted.sort_unstable();
    sorted.dedup();
    let mut identity = Vec::new();
    for key in &sorted {
        identity.extend_from_slice(&key.len().to_le_bytes());
        identity.extend_from_slice(key);
    }
    let mut guard = INTERNED_SHAPES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let interned = guard.get_or_insert_with(|| InternedShapes {
        shapes: std::collections::HashMap::new(),
        pruned_at: 0,
    });
    if let Some(shape) = interned
        .shapes
        .get(&identity)
        .and_then(std::sync::Weak::upgrade)
    {
        return Some(shape);
    }
    let shape = TableShape::new(keys.iter().cloned())?;
    // Tables made from the site's shape borrow one count instead of
    // taking their own.
    shape.pin();
    if interned.shapes.len() >= interned.pruned_at.max(64) * 2 {
        interned.shapes.retain(|_, shape| shape.strong_count() > 0);
        interned.pruned_at = interned.shapes.len();
    }
    interned
        .shapes
        .insert(identity, std::sync::Arc::downgrade(&shape));
    Some(shape)
}

/// Creates the table for the `NewTable` at `pc`, with the site's shared key
/// list when it has one so the constructor's field stores clone no key.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn create_constructor_table(
    runtime: &mut Runtime,
    chunk: &Chunk,
    pc: usize,
) -> RuntimeResult<TableHandle> {
    match constructor_shape(chunk, pc) {
        Some(shape) => runtime.create_table_with_shape(shape),
        // The compiler counts the positional items the constructor stores.
        None => runtime.create_table_sized(array_hint(chunk, pc)),
    }
}

#[inline]
fn array_hint(chunk: &Chunk, pc: usize) -> usize {
    chunk.code.get(pc).map_or(0, |at| usize::from(at.b()))
}

/// [`create_constructor_table`] that returns the table's native identity, or
/// `None` when no table could be made, without a `Result` in memory.
#[inline]
pub(crate) fn create_constructor_table_fast(
    runtime: &mut Runtime,
    chunk: &Chunk,
    pc: usize,
) -> Option<TableHandle> {
    let identity = match constructor_shape(chunk, pc) {
        Some(shape) => runtime.create_table_with_shape_identity(shape),
        None => runtime.create_table_sized_identity(array_hint(chunk, pc)),
    };
    (identity != crate::runtime::NO_TABLE_IDENTITY)
        .then(|| TableHandle::from_native_identity(identity))
}

pub(crate) fn table_instruction(
    runtime: &mut Runtime,
    registers: &mut [Value],
    chunk: &Chunk,
    pc: usize,
    instruction: Instruction,
    opcode: OpCode,
) -> VmResult<()> {
    match opcode {
        OpCode::NewTable => {
            registers[usize::from(instruction.a())] =
                Value::Table(create_constructor_table(runtime, chunk, pc)?);
        }
        OpCode::GetTable => {
            let receiver = &registers[usize::from(instruction.b())];
            let key = &registers[usize::from(instruction.c())];
            registers[usize::from(instruction.a())] = match receiver {
                Value::Table(table) => runtime.table_get(*table, key)?,
                Value::String(_) => {
                    let string_table = table_handle(&runtime.get_global("string"), "index")?;
                    runtime.table_get(string_table, key)?
                }
                Value::File(file) => runtime.file_method(*file, key)?,
                receiver => {
                    return Err(RuntimeError::TypeError {
                        operation: "index",
                        expected: "table",
                        found: receiver.type_name(),
                    }
                    .into());
                }
            };
        }
        OpCode::SetTable => {
            let table = table_handle(&registers[usize::from(instruction.a())], "index")?;
            let key = registers[usize::from(instruction.b())].clone();
            let value = registers[usize::from(instruction.c())].clone();
            runtime.table_set(table, &key, value)?;
        }
        _ => return Err(VmError::ProgramCounterOutOfBounds(0)),
    }
    Ok(())
}

pub(crate) fn global_name(chunk: &Chunk, index: u32) -> VmResult<&LuaString> {
    match chunk
        .constants
        .get(usize::try_from(index).expect("u32 fits usize on supported targets"))
    {
        Some(Constant::String(value)) => Ok(value),
        _ => Err(VmError::InvalidGlobalName(index)),
    }
}

pub(crate) fn jump_target(next_pc: usize, offset: i32) -> VmResult<usize> {
    usize::try_from(i64::try_from(next_pc).unwrap_or(i64::MAX) + i64::from(offset))
        .map_err(|_| VmError::ProgramCounterOutOfBounds(next_pc))
}

pub(crate) fn register_values(registers: &[Value], start: u8, count: u8) -> VmResult<Vec<Value>> {
    let first = usize::from(start);
    let last = if count == 0 {
        registers.len()
    } else {
        first + usize::from(count)
    };
    registers
        .get(first..last)
        .map(<[Value]>::to_vec)
        .ok_or(VmError::RegisterRange { start, count })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::string::LuaString;

    #[test]
    fn moving_a_string_over_a_string_releases_the_old_one_and_shares_the_new() {
        let first = LuaString::from("first");
        let first_weak = first.downgrade();
        let second = LuaString::from("second");
        let second_weak = second.downgrade();
        let mut registers = vec![Value::String(first), Value::String(second), Value::Nil];
        copy_register(&mut registers, 1, 0);
        assert!(
            !second_weak.is_alive(),
            "the overwritten string is released"
        );
        assert_eq!(registers[0], registers[1]);
        // Two registers share one allocation: dropping one keeps it alive.
        registers[0] = Value::Nil;
        assert!(first_weak.is_alive());
        registers[1] = Value::Nil;
        assert!(!first_weak.is_alive(), "no reference leaked");
    }

    #[test]
    fn move_between_strings_and_other_values() {
        let text = LuaString::from("text");
        let weak = text.downgrade();
        let mut registers = vec![Value::String(text), Value::Number(2.5), Value::Nil];
        copy_register(&mut registers, 2, 0); // string over nil
        copy_register(&mut registers, 0, 1); // number over string
        assert_eq!(registers[0], Value::Number(2.5));
        assert!(weak.is_alive(), "the copy in register 2 still holds it");
        copy_register(&mut registers, 2, 2); // string over itself
        assert!(weak.is_alive());
        copy_register(&mut registers, 2, 0); // number over string
        assert!(!weak.is_alive());
        copy_register(&mut registers, 1, 1);
        assert_eq!(registers[1], Value::Number(2.5));
    }

    #[test]
    fn closure_handle_is_the_payload_word() {
        let mut vm = crate::vm::Vm::new();
        let chunk = crate::bytecode::Chunk::new(
            1,
            Vec::new(),
            vec![crate::bytecode::Instruction::return_values(0, 1)],
        );
        // Several closures, so slots and generations differ.
        let mut handles = Vec::new();
        for _ in 0..5 {
            handles.push(
                vm.runtime_mut()
                    .create_closure(std::sync::Arc::new(chunk.clone()), Vec::new())
                    .unwrap(),
            );
        }
        for handle in handles {
            let value = Value::Closure(handle);
            assert_eq!(closure_handle(&value), Some(handle));
            assert_eq!(
                closure_handle(&value).map(crate::runtime::ClosureHandle::native_identity),
                Some(handle.native_identity())
            );
        }
        assert_eq!(closure_handle(&Value::Nil), None);
        assert_eq!(closure_handle(&Value::Number(1.0)), None);
        assert_eq!(
            closure_handle(&Value::Table(
                crate::runtime::TableHandle::from_native_identity(3)
            )),
            None
        );
    }

    /// A move copies the tag and the payload word each on its own. Every
    /// variant that owns nothing must come out equal, over a register of
    /// any variant, and a register moved onto itself must stay as it was.
    #[test]
    fn move_copies_every_variant_without_a_string() {
        let cases = [
            Value::Nil,
            Value::Boolean(true),
            Value::Boolean(false),
            Value::Integer(-5),
            Value::Number(2.5),
            Value::Number(f64::NAN),
            Value::Table(crate::runtime::TableHandle::from_native_identity(9)),
            Value::Int64(i64::MIN),
            Value::UInt64(u64::MAX),
        ];
        // `NaN != NaN`, so numbers are compared by their bits.
        let same = |left: &Value, right: &Value| match (left, right) {
            (Value::Number(left), Value::Number(right)) => left.to_bits() == right.to_bits(),
            _ => left == right,
        };
        for from in &cases {
            for to in &cases {
                let mut registers = vec![to.clone(), from.clone()];
                copy_register(&mut registers, 0, 1);
                assert!(same(&registers[0], from), "{to:?} <- {from:?}");
                assert!(same(&registers[1], from));
                copy_register(&mut registers, 0, 0);
                assert!(same(&registers[0], from), "{from:?} onto itself");
            }
        }
    }

    /// `move_value` takes the word path unless a string is on either side,
    /// and must leave nil behind, release a string it overwrites and keep a
    /// moved string alive.
    #[test]
    fn move_value_leaves_nil_and_drops_what_it_overwrites() {
        let old = LuaString::from("old");
        let old_weak = old.downgrade();
        let mut slot = Value::String(old);
        let mut from = Value::Integer(7);
        move_value(&mut slot, &mut from);
        assert_eq!(slot, Value::Integer(7));
        assert_eq!(from, Value::Nil);
        assert!(!old_weak.is_alive(), "the overwritten string is released");

        // A non-string over an integer.
        let mut from = Value::Boolean(true);
        move_value(&mut slot, &mut from);
        assert_eq!(slot, Value::Boolean(true));
        assert_eq!(from, Value::Nil);

        // A string into an integer slot.
        let mut slot = Value::Integer(3);
        let moved = LuaString::from("moved");
        let moved_weak = moved.downgrade();
        let mut from = Value::String(moved);
        move_value(&mut slot, &mut from);
        assert_eq!(slot, Value::from("moved"));
        assert_eq!(from, Value::Nil);
        assert!(moved_weak.is_alive(), "the slot holds the moved string");

        let mut from = Value::Number(1.5);
        move_value(&mut slot, &mut from);
        assert_eq!(slot, Value::Number(1.5));
        assert_eq!(from, Value::Nil);
        assert!(!moved_weak.is_alive());

        // Handle slots own nothing, so overwriting one is a word copy.
        let mut vm = crate::vm::Vm::new();
        let thread = vm.runtime_mut().create_thread().unwrap();
        let table = crate::runtime::TableHandle::from_native_identity(9);
        let mut slot = Value::Table(table);
        let mut from = Value::Thread(thread);
        move_value(&mut slot, &mut from);
        assert_eq!(slot, Value::Thread(thread));
        assert_eq!(from, Value::Nil);
        let mut from = Value::Table(table);
        move_value(&mut slot, &mut from);
        assert_eq!(slot, Value::Table(table));
        assert_eq!(from, Value::Nil);
        let mut from = Value::Integer(-1);
        move_value(&mut slot, &mut from);
        assert_eq!(slot, Value::Integer(-1));
    }

    #[test]
    fn store_result_moves_every_variant_and_releases_the_old_string() {
        let text = LuaString::from("result");
        let text_weak = text.downgrade();
        let cases = [
            Value::Nil,
            Value::Boolean(true),
            Value::Integer(-5),
            Value::Number(2.5),
            Value::String(text),
            Value::Table(crate::runtime::TableHandle::from_native_identity(9)),
            Value::Int64(i64::MIN),
            Value::UInt64(u64::MAX),
        ];
        for case in cases {
            let old = LuaString::from("old");
            let old_weak = old.downgrade();
            for start in [Value::String(old), Value::Number(1.0)] {
                let released = matches!(start, Value::String(_));
                let mut slot = start;
                let mut result: Result<Value, ()> = Ok(case.clone());
                assert!(store_result(&mut slot, &mut result));
                std::mem::forget(result);
                assert_eq!(slot, case);
                if released {
                    assert!(!old_weak.is_alive());
                }
            }
        }
        // The slot owns the only reference to the string now.
        assert!(!text_weak.is_alive());
    }

    #[test]
    fn store_result_leaves_the_slot_and_hands_back_an_error() {
        let old = LuaString::from("old");
        let old_weak = old.downgrade();
        let mut slot = Value::String(old);
        let mut result: Result<Value, &str> = Err("boom");
        assert!(!store_result(&mut slot, &mut result));
        assert_eq!(result, Err("boom"));
        assert!(old_weak.is_alive());
    }

    #[test]
    fn push_result_appends_without_touching_stale_capacity() {
        let stale = LuaString::from("stale");
        let stale_weak = stale.downgrade();
        let mut buffer = vec![Value::String(stale)];
        buffer.clear();
        assert!(!stale_weak.is_alive());
        for case in [Value::Number(0.5), Value::Int64(3), Value::Boolean(false)] {
            let mut result: Result<Value, ()> = Ok(case.clone());
            assert!(push_result(&mut buffer, &mut result));
            std::mem::forget(result);
            assert_eq!(buffer.last(), Some(&case));
        }
        let text = LuaString::from("pushed");
        let mut result: Result<Value, ()> = Ok(Value::String(text.clone()));
        assert!(push_result(&mut buffer, &mut result));
        std::mem::forget(result);
        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer[3], Value::String(text));
        let mut failed: Result<Value, ()> = Err(());
        assert!(!push_result(&mut buffer, &mut failed));
        assert_eq!(buffer.len(), 4);
    }

    #[test]
    fn store_clone_and_set_register_account_for_strings() {
        let a = LuaString::from("a");
        let a_weak = a.downgrade();
        let b = LuaString::from("b");
        let b_weak = b.downgrade();
        let source = Value::String(a);
        let mut slot = Value::String(b);
        store_clone(&mut slot, &source);
        assert!(!b_weak.is_alive());
        assert_eq!(slot, source);
        store_clone(&mut slot, &Value::Boolean(true));
        assert_eq!(slot, Value::Boolean(true));
        assert!(a_weak.is_alive());
        store_clone(&mut slot, &source);
        set_register(&mut slot, Value::Integer(7));
        drop(source);
        assert!(!a_weak.is_alive());
        assert_eq!(slot, Value::Integer(7));
        assert_eq!(
            clone_register_value(&Value::Number(1.5)),
            Value::Number(1.5)
        );
    }

    #[test]
    fn store_clone_scalar_copies_scalars_and_clones_and_releases_strings() {
        let a = LuaString::from("a");
        let a_weak = a.downgrade();
        let b = LuaString::from("b");
        let b_weak = b.downgrade();
        let source = Value::String(a);
        let mut slot = Value::String(b);
        store_clone_scalar(&mut slot, &source);
        assert!(!b_weak.is_alive());
        assert_eq!(slot, source);
        store_clone_scalar(&mut slot, &Value::Number(2.5));
        assert_eq!(slot, Value::Number(2.5));
        assert!(a_weak.is_alive());
        store_clone_scalar(&mut slot, &Value::Boolean(false));
        store_clone_scalar(&mut slot, &Value::Integer(-4));
        assert_eq!(slot, Value::Integer(-4));
        drop(source);
        assert!(!a_weak.is_alive());
    }

    #[test]
    fn constructor_sites_with_the_same_keys_share_one_shape() {
        let keys = |names: &[&str]| {
            names
                .iter()
                .map(|name| LuaString::from(*name))
                .collect::<Vec<_>>()
        };
        let first = interned_shape(&keys(&["y", "x", "z"])).expect("three keys make a shape");
        let second = interned_shape(&keys(&["z", "y", "x", "x"])).expect("the same keys");
        assert!(std::sync::Arc::ptr_eq(&first, &second));
        let other = interned_shape(&keys(&["x", "y"])).expect("two keys make a shape");
        assert!(!std::sync::Arc::ptr_eq(&first, &other));
        // The registry keeps no shape alive, except the first few that
        // `TableShape::pin` keeps for good.
        let weak = std::sync::Arc::downgrade(&other);
        let pinned = other.is_pinned();
        drop(other);
        assert_eq!(weak.upgrade().is_some(), pinned);
        let rebuilt = interned_shape(&keys(&["y", "x"])).expect("rebuilt");
        assert_eq!(rebuilt.len(), 2);
        assert_eq!(first.len(), 3);
    }

    #[test]
    fn only_the_first_shapes_are_pinned_and_the_rest_are_freed_with_their_tables() {
        let mut freed = 0;
        for index in 0..1500 {
            let keys = [
                LuaString::from(format!("pin_a_{index}").as_str()),
                LuaString::from(format!("pin_b_{index}").as_str()),
            ];
            let shape = interned_shape(&keys).expect("two keys make a shape");
            let weak = std::sync::Arc::downgrade(&shape);
            let pinned = shape.is_pinned();
            drop(shape);
            assert_eq!(weak.upgrade().is_some(), pinned, "shape {index}");
            freed += usize::from(!pinned);
        }
        assert!(freed > 0, "the pin limit never bit");
    }
}
