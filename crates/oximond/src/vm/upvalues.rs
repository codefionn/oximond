//! Captured register slots, closure cells, and upvalue reads and writes.

use crate::bytecode::Instruction;
use crate::runtime::{CellHandle, ClosureHandle, ClosureSnapshot, RuntimeError, Value};
use crate::vm::registers::clone_register_value;
use crate::vm::stack::{CallStack, FrameMut, FrameRef};
use crate::vm::{ReturnTarget, Vm, VmError, VmResult, bump};

impl Vm {
    // Inlined so the snapshot is built in the caller's registers: returned
    // through memory, its copy stalled on store forwarding in every call.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn closure_body(&self, handle: ClosureHandle) -> VmResult<ClosureSnapshot> {
        if self.interpreter_profiling {
            bump(&self.interpreter_counters.closure_body_calls);
        }
        self.runtime
            .closure_snapshot(handle)
            .map_err(|_| VmError::MissingClosureBody(handle))
    }

    /// The cell for register `register` of frame `frame_index`: the open
    /// upvalue of that stack slot if a closure already captured it, else a
    /// new one. The variable stays in its register.
    #[allow(unsafe_code)]
    pub(crate) fn capture_slot(
        &mut self,
        frames: &mut CallStack,
        frame_index: usize,
        register: u8,
    ) -> VmResult<CellHandle> {
        let frame = &frames[frame_index];
        let slot = frame.base + u32::from(register);
        if slot >= frame.top {
            return Err(VmError::RegisterRange {
                start: register,
                count: 1,
            });
        }
        if let Some(cell) = frames.open_cell_at(slot) {
            return Ok(cell);
        }
        let anchor = frames.anchor();
        // SAFETY: the slot lies in the live window of `frames`' value stack
        // and the anchor names that stack. Every removal of the slot from
        // the stack closes the cell first (`close_upvalues`).
        let cell = unsafe { self.runtime.create_open_cell(anchor, slot) }?;
        frames.push_open(slot, cell);
        Ok(cell)
    }

    /// Closes the open upvalues of every slot at or above absolute stack
    /// index `level`: each variable's value moves into its cell, which
    /// stands alone afterwards. Must run before those slots are reused,
    /// moved or dropped.
    #[inline(always)]
    pub(crate) fn close_upvalues(&mut self, frames: &mut CallStack, level: u32) {
        if frames.open_top() > level {
            self.close_upvalues_cold(frames, level);
        }
    }

    #[cold]
    #[inline(never)]
    #[allow(unsafe_code)]
    pub(crate) fn close_upvalues_cold(&mut self, frames: &mut CallStack, level: u32) {
        while let Some(open) = frames.pop_open(level) {
            // SAFETY: the slot is still in the stack, which has not been
            // touched since the cell was opened.
            unsafe { self.runtime.close_open_cell(open.cell) };
        }
    }

    /// Closes the upvalues of the top frame's registers, before the frame
    /// leaves the stack.
    #[inline(always)]
    pub(crate) fn close_top_frame_upvalues(&mut self, frames: &mut CallStack) {
        if frames.open_top() != 0 && !frames.is_empty() {
            let base = frames.top_base();
            self.close_upvalues(frames, base);
        }
    }

    pub(crate) fn create_detached_cell(&mut self) -> VmResult<CellHandle> {
        self.runtime.create_cell(Value::Nil).map_err(Into::into)
    }

    /// `CloseUpvalue`: the variable in register `register` of frame
    /// `frame_index` goes out of scope. A closure that captured it keeps
    /// the value; the register starts afresh.
    #[allow(unsafe_code)]
    pub(crate) fn close_captured_slot(
        &mut self,
        frames: &mut CallStack,
        frame_index: usize,
        register: u8,
    ) -> VmResult<()> {
        let frame = &frames[frame_index];
        let slot = frame.base + u32::from(register);
        if slot >= frame.top {
            return Err(VmError::RegisterRange {
                start: register,
                count: 1,
            });
        }
        if let Some(open) = frames.remove_open(slot) {
            // SAFETY: the slot is in the live window.
            unsafe { self.runtime.close_open_cell(open.cell) };
        }
        Ok(())
    }

    pub(crate) fn mark_close_slot(
        &mut self,
        frame: &mut FrameMut<'_>,
        register: u8,
    ) -> VmResult<()> {
        let value =
            frame
                .registers
                .get(usize::from(register))
                .cloned()
                .ok_or(VmError::RegisterRange {
                    start: register,
                    count: 1,
                })?;
        if matches!(value, Value::Nil) {
            return Ok(());
        }
        let Some(metatable) = self.runtime.raw_metatable(&value)? else {
            return Err(RuntimeError::message("variable got a non-closable value").into());
        };
        if matches!(
            self.runtime.table_get(metatable, &Value::from("__close"))?,
            Value::Nil
        ) {
            return Err(RuntimeError::message("variable got a non-closable value").into());
        }
        frame.extras().close_slots.push(register);
        Ok(())
    }

    pub(crate) fn close_local_slot(
        &mut self,
        frames: &mut CallStack,
        frame: &mut FrameMut<'_>,
        register: u8,
        error: Value,
    ) -> VmResult<()> {
        let Some(extras) = frame.extras_mut_ref() else {
            return Ok(());
        };
        let Some(index) = extras
            .close_slots
            .iter()
            .rposition(|slot| *slot == register)
        else {
            return Ok(());
        };
        extras.close_slots.remove(index);
        let value = frame.registers[usize::from(register)].clone();
        if matches!(value, Value::Nil) {
            return Ok(());
        }
        let metatable = self
            .runtime
            .raw_metatable(&value)?
            .ok_or_else(|| RuntimeError::message("variable got a non-closable value"))?;
        let handler = self.runtime.table_get(metatable, &Value::from("__close"))?;
        match handler {
            Value::NativeFunction(function) => {
                self.call_native_above(frames, Some(frame), function, &[value, error])?;
                Ok(())
            }
            _ => Err(
                RuntimeError::message("__close must be a native callable in this VM path").into(),
            ),
        }
    }

    pub(crate) fn start_close_local(
        &mut self,
        frames: &mut CallStack,
        register: u8,
        error: Value,
    ) -> VmResult<()> {
        let frame_index = frames.len() - 1;
        let mut frame = frames.at_mut(frame_index);
        let Some(extras) = frame.extras_mut_ref() else {
            return Ok(());
        };
        let Some(index) = extras
            .close_slots
            .iter()
            .rposition(|slot| *slot == register)
        else {
            return Ok(());
        };
        extras.close_slots.remove(index);
        let value = frame.registers[usize::from(register)].clone();
        if matches!(value, Value::Nil) {
            return Ok(());
        }
        let metatable = self
            .runtime
            .raw_metatable(&value)?
            .ok_or_else(|| RuntimeError::message("variable got a non-closable value"))?;
        let handler = self.runtime.table_get(metatable, &Value::from("__close"))?;
        match handler {
            Value::NativeFunction(function) => self
                .call_native(frames, function, &[value, error])
                .map(|_| ())
                .map_err(Into::into),
            Value::Closure(closure) => {
                let body = self.closure_body(closure)?;
                let child = self.make_frame(
                    body.chunk(),
                    vec![value, error],
                    body.shared_cells(),
                    Some(ReturnTarget::CloseLocal),
                    None,
                    body.environment(),
                    Some(closure),
                )?;
                frames.push_within_limit(child)?;
                Ok(())
            }
            Value::BuiltinFunction(builtin) => {
                let arguments = vec![value, error];
                match self.call_builtin(builtin, &arguments) {
                    Ok(_) => Ok(()),
                    Err(VmError::InvalidDispatch(_)) => self.push_builtin_call_frame(
                        frames,
                        builtin,
                        arguments,
                        ReturnTarget::CloseLocal,
                    ),
                    Err(error) => Err(error),
                }
            }
            _ => Err(RuntimeError::message("__close must be callable").into()),
        }
    }

    pub(crate) fn get_upvalue(
        &self,
        frame: &mut FrameMut<'_>,
        instruction: Instruction,
    ) -> VmResult<()> {
        if self.interpreter_profiling {
            bump(&self.interpreter_counters.get_upvalue_reads);
        }
        // The error below is cold: build it lazily so a hot upvalue read
        // does not construct and immediately drop a `VmError` every step.
        let cell = *frame
            .upvalues
            .get(usize::from(instruction.d()))
            .ok_or_else(|| VmError::RegisterRange {
                start: instruction.a(),
                count: 1,
            })?;
        let value = self.runtime.cell_ref(cell)?;
        frame.registers[usize::from(instruction.a())] = clone_register_value(value);
        Ok(())
    }

    pub(crate) fn set_upvalue(
        &mut self,
        frame: &FrameRef<'_>,
        instruction: Instruction,
    ) -> VmResult<()> {
        let cell = *frame
            .upvalues
            .get(usize::from(instruction.d()))
            .ok_or_else(|| VmError::RegisterRange {
                start: instruction.a(),
                count: 1,
            })?;
        self.runtime
            .cell_set(cell, frame.registers[usize::from(instruction.a())].clone())
            .map_err(Into::into)
    }
}

pub(crate) fn replace_active_closure_cell(
    frames: &mut CallStack,
    closure: ClosureHandle,
    index: usize,
    cell: CellHandle,
) {
    for frame in frames {
        if frame.closure == Some(closure) {
            // The captures may be the frame's own now.
            frame.clear_plain_mark();
            frame.upvalues.replace(index, cell);
        }
    }
}
