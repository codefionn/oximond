//! Safe debugger access to register owners parked beneath native callbacks.

use crate::compat::CompatibilityProfile;
use crate::runtime::{CaptureCells, Runtime, ThreadHandle, Value};
use crate::string::LuaString;
use crate::vm::debug_info::{SuspendedFrame, debug_local_slot};
use crate::vm::stack::{CallStack, FrameMut, FrameRef, OwnedFrame};
use crate::vm::{CoroutineState, Frame, FrameExtras, HostResume, Vm, VmResult};
use std::cell::OnceCell;
use std::collections::BTreeSet;

/// What the VM keeps of the Lua stack parked beneath one active host
/// callback, for the debugger (`debug.getinfo`, `debug.getlocal` and
/// [`Vm::host_stack_info`] from inside the callback) and the collector.
///
/// A plain callback parks the stack itself here for the callback's
/// duration, and the debug records and local snapshots are built from it
/// only when something asks for them. A host such as Luanti calls a C
/// function per node or vector, which never asks: building them at every
/// entry walked every frame of the Lua stack, three times, per call. The
/// parked stack does not change while it waits (re-entrant Lua runs on a
/// stack of its own, and the only writes it can make, through captured
/// variables, go to slots the snapshots read through their cells), so a
/// record built late is the one an entry would have built.
pub(crate) struct HostSnapshot {
    /// The parked stack, when the callback has one to itself. A callback
    /// run for a frame that is closing keeps the stack with its caller,
    /// and its records are built at entry.
    ///
    /// The stack stays where its caller holds it and is lent here by
    /// address: moving it in and out cost each host call two copies of the
    /// whole `CallStack` and the drop of the empty one left behind. The
    /// caller is suspended in the callback for as long as the snapshot
    /// exists (see `Vm::park_frames`), so nothing else reads or writes the
    /// stack through its own name meanwhile.
    parked: Option<std::ptr::NonNull<CallStack>>,
    pub(crate) thread: Option<ThreadHandle>,
    debug: OnceCell<Vec<SuspendedFrame>>,
    locals: OnceCell<Vec<HostLocalFrame>>,
}

impl HostSnapshot {
    /// The snapshot of `frames`, parked for a callback of `thread`.
    ///
    /// # Safety
    ///
    /// `frames` must stay where it is, unused through any other name, until
    /// the snapshot is dropped.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn parked(frames: &mut CallStack, thread: Option<ThreadHandle>) -> Self {
        Self {
            parked: Some(std::ptr::NonNull::from(frames)),
            thread,
            debug: OnceCell::new(),
            locals: OnceCell::new(),
        }
    }

    /// A snapshot whose records were built at entry.
    pub(crate) fn built(
        debug: Vec<SuspendedFrame>,
        locals: Vec<HostLocalFrame>,
        thread: Option<ThreadHandle>,
    ) -> Self {
        Self {
            parked: None,
            thread,
            debug: OnceCell::from(debug),
            locals: OnceCell::from(locals),
        }
    }

    /// The parked stack, if the callback has one to itself.
    #[inline]
    pub(crate) fn parked_stack(&self) -> Option<&CallStack> {
        // SAFETY: `Self::parked`'s caller keeps the stack in place and
        // unused for the snapshot's life.
        #[allow(unsafe_code)]
        self.parked.map(|stack| unsafe { &*stack.as_ptr() })
    }

    /// The debug records of the parked frames, bottom first.
    pub(crate) fn debug_frames(&self) -> &[SuspendedFrame] {
        self.debug.get_or_init(|| {
            self.parked_stack()
                .map(|frames| frames.iter().map(SuspendedFrame::capture).collect())
                .unwrap_or_default()
        })
    }

    /// The local snapshots of the parked frames, bottom first.
    pub(crate) fn local_frames(&self) -> &[HostLocalFrame] {
        self.locals.get_or_init(|| {
            self.parked_stack()
                .map(|frames| {
                    (0..frames.len())
                        .map(|index| {
                            let frame = frames.at(index);
                            HostLocalFrame::capture(&frame, frames.open_in(frame.base, frame.top))
                        })
                        .collect()
                })
                .unwrap_or_default()
        })
    }

    pub(crate) fn local_frames_mut(&mut self) -> &mut [HostLocalFrame] {
        self.local_frames();
        self.locals
            .get_mut()
            .expect("local snapshots were just built")
    }

    /// The local snapshots, if anything built them.
    pub(crate) fn built_local_frames(&self) -> Option<&[HostLocalFrame]> {
        self.locals.get().map(Vec::as_slice)
    }
}

pub(crate) struct HostLocalFrame {
    pub(crate) frame: OwnedFrame,
    register_writes: BTreeSet<usize>,
    vararg_writes: BTreeSet<usize>,
    /// The registers of the live frame that closures captured, with their
    /// cells. The snapshot's copy of such a register goes stale as soon as a
    /// closure writes it, so reads and writes go to the cell, which stands
    /// for the live register.
    captured: Vec<(usize, crate::runtime::CellHandle)>,
}

impl HostLocalFrame {
    pub(crate) fn capture(source: &FrameRef<'_>, open: &[crate::vm::stack::OpenUpvalue]) -> Self {
        let extras = FrameExtras::default();
        let captured = open
            .iter()
            .map(|open| ((open.slot - source.base) as usize, open.cell))
            .collect();
        Self {
            frame: OwnedFrame {
                frame: Frame {
                    chunk: source.chunk.as_arc().clone().into(),
                    closure: source.closure,
                    environment: source.environment,
                    base: 0,
                    top: source.top - source.base,
                    upvalues: CaptureCells::default(),
                    pc: source.pc,
                    last_interpreted_pc: source.last_interpreted_pc,
                    caller: None,
                    call_site: None,
                    tail_calls: source.tail_calls,
                    vararg_word: 0,
                    shape: crate::runtime::NO_SHAPE,
                    open: None,
                    extras: Some(Box::new(extras)),
                },
                registers: source.registers.to_vec(),
                varargs: source.varargs.to_vec(),
            },
            register_writes: BTreeSet::new(),
            vararg_writes: BTreeSet::new(),
            captured,
        }
    }

    pub(crate) fn apply(&self, frame: &mut FrameMut<'_>) {
        self.apply_outside(frame, 0..0);
    }

    /// [`Self::apply`] that leaves the registers in `results` alone: a
    /// callback that stored its results there itself did so after every
    /// edit, and a result assignment wins over an edit of its destination.
    pub(crate) fn apply_outside(&self, frame: &mut FrameMut<'_>, results: std::ops::Range<usize>) {
        for &register in &self.register_writes {
            if results.contains(&register) {
                continue;
            }
            frame.registers[register] = self.frame.registers[register].clone();
        }
        for &index in &self.vararg_writes {
            frame.varargs[index] = self.frame.varargs[index].clone();
        }
    }
}

fn local_slot(frame: &FrameRef<'_>, index: i32) -> Option<(LuaString, usize, bool)> {
    if index < 0 {
        let slot = usize::try_from(index.checked_neg()?.checked_sub(1)?).ok()?;
        return Some((LuaString::from("(*vararg)"), slot, true));
    }
    let index = usize::try_from(index).ok()?.checked_sub(1)?;
    let (name, register) = debug_local_slot(frame, index)?;
    Some((
        name.unwrap_or_else(|| LuaString::from("(*temporary)")),
        register,
        false,
    ))
}

pub(crate) fn local_index(value: &Value) -> Option<i32> {
    match value {
        Value::Integer(index) => i32::try_from(*index).ok(),
        Value::Number(index)
            if index.is_finite()
                && index.fract() == 0.0
                && *index >= f64::from(i32::MIN)
                && *index <= f64::from(i32::MAX) =>
        {
            Some(*index as i32)
        }
        _ => None,
    }
}

/// The cell of register `register` among a snapshot's captured registers.
fn captured_cell(
    captured: &[(usize, crate::runtime::CellHandle)],
    register: usize,
) -> Option<crate::runtime::CellHandle> {
    captured
        .iter()
        .find(|(captured, _)| *captured == register)
        .map(|(_, cell)| *cell)
}

fn get_local(
    runtime: &Runtime,
    compatibility: CompatibilityProfile,
    frame: &FrameRef<'_>,
    captured: &[(usize, crate::runtime::CellHandle)],
    index: i32,
) -> VmResult<Option<(LuaString, Value)>> {
    let Some((name, slot, vararg)) = local_slot(frame, index) else {
        return Ok(None);
    };
    let value = if vararg {
        if compatibility.is_lua51_family() {
            return Ok(None);
        }
        if compatibility == CompatibilityProfile::Lua55
            && let Some(register) = frame.chunk.arg_register
        {
            let Some(Value::Table(table)) = frame.registers.get(usize::from(register)) else {
                return Ok(None);
            };
            let count = runtime.table_get(*table, &Value::from("n"))?;
            let count = match count {
                Value::Integer(count) => count as f64,
                Value::Number(count) => count,
                _ => return Ok(None),
            };
            if slot as f64 >= count {
                return Ok(None);
            }
            runtime.table_get(*table, &Value::Integer((slot + 1) as i64))?
        } else {
            let Some(value) = frame.varargs.get(slot) else {
                return Ok(None);
            };
            value.clone()
        }
    } else if let Some(cell) = captured_cell(captured, slot) {
        runtime.cell_get(cell)?
    } else {
        frame.registers[slot].clone()
    };
    Ok(Some((name, value)))
}

fn set_local(
    runtime: &mut Runtime,
    compatibility: CompatibilityProfile,
    frame: &mut FrameMut<'_>,
    captured: &[(usize, crate::runtime::CellHandle)],
    index: i32,
    value: Value,
) -> VmResult<Option<(LuaString, usize, bool, bool)>> {
    if get_local(runtime, compatibility, &frame.as_ref(), captured, index)?.is_none() {
        return Ok(None);
    }
    let (name, slot, vararg) =
        local_slot(&frame.as_ref(), index).expect("existing local has a slot");
    let deferred = if vararg {
        if compatibility == CompatibilityProfile::Lua55
            && let Some(register) = frame.chunk.arg_register
        {
            let Value::Table(table) = frame.registers[usize::from(register)] else {
                unreachable!()
            };
            runtime.table_set(table, &Value::Integer((slot + 1) as i64), value)?;
            false
        } else {
            frame.varargs[slot] = value;
            true
        }
    } else if let Some(cell) = captured_cell(captured, slot) {
        runtime.cell_set(cell, value.clone())?;
        frame.registers[slot] = value;
        false
    } else {
        frame.registers[slot] = value;
        true
    };
    Ok(Some((name, slot, vararg, deferred)))
}

impl Vm {
    pub(crate) fn get_frame_local(
        &self,
        frame: &FrameRef<'_>,
        index: i32,
    ) -> VmResult<Option<(LuaString, Value)>> {
        get_local(&self.runtime, self.compatibility, frame, &[], index)
    }

    pub(crate) fn set_frame_local(
        &mut self,
        frame: &mut FrameMut<'_>,
        index: i32,
        value: Value,
    ) -> VmResult<Option<LuaString>> {
        let name = set_local(
            &mut self.runtime,
            self.compatibility,
            frame,
            &[],
            index,
            value,
        )?
        .map(|(name, _, _, _)| name);
        if name.is_some() {
            self.local_mutation_epoch = self.local_mutation_epoch.saturating_add(1);
        }
        Ok(name)
    }

    pub(crate) fn host_local_location(&self, mut level: usize) -> Option<(usize, usize)> {
        if level == 0 {
            return None;
        }
        for stack_index in (0..self.host_snapshots.len()).rev() {
            let stack = &self.host_snapshots[stack_index];
            if stack.thread != self.current_coroutine {
                continue;
            }
            for (index, frame) in stack.local_frames().iter().enumerate().rev() {
                if level == 1 {
                    return Some((stack_index, index));
                }
                level -= 1;
                let tails = frame.frame.tail_calls as usize;
                if level <= tails {
                    return None;
                }
                level -= tails;
            }
            if level == 1 {
                return None;
            }
            level -= 1; // The older parked native callback.
        }
        None
    }

    /// Reads a local of the Lua stack beneath the active native callback.
    /// Level 1 is its Lua caller; positive indices name locals and negative
    /// indices name varargs. Native level 0 belongs to the embedding stack.
    ///
    /// # Errors
    ///
    /// Returns an error when a captured-register cell read or a parked vararg
    /// table read fails, such as on a stale handle.
    pub fn host_get_local(&self, level: usize, index: i32) -> VmResult<Option<(LuaString, Value)>> {
        let Some((stack, frame)) = self.host_local_location(level) else {
            return Ok(None);
        };
        get_local(
            &self.runtime,
            self.compatibility,
            &self.host_snapshots[stack].local_frames()[frame]
                .frame
                .view_ref(),
            &self.host_snapshots[stack].local_frames()[frame].captured,
            index,
        )
    }

    /// Writes a parked local. Captured locals change their shared cell at once;
    /// other registers and varargs copy back before the native callback exits.
    ///
    /// # Errors
    ///
    /// Returns an error when a captured-register cell write or a parked vararg
    /// table write fails, such as on a stale handle.
    pub fn host_set_local(
        &mut self,
        level: usize,
        index: i32,
        value: Value,
    ) -> VmResult<Option<LuaString>> {
        self.local_writes += 1;
        let Some((stack, frame)) = self.host_local_location(level) else {
            return Ok(None);
        };
        let snapshot = &mut self.host_snapshots[stack].local_frames_mut()[frame];
        let captured = std::mem::take(&mut snapshot.captured);
        let result = set_local(
            &mut self.runtime,
            self.compatibility,
            &mut snapshot.frame.view(),
            &captured,
            index,
            value,
        );
        snapshot.captured = captured;
        let Some((name, slot, vararg, deferred)) = result? else {
            return Ok(None);
        };
        if deferred {
            if vararg {
                snapshot.vararg_writes.insert(slot);
            } else {
                snapshot.register_writes.insert(slot);
            }
        }
        self.local_mutation_epoch = self.local_mutation_epoch.saturating_add(1);
        Ok(Some(name))
    }

    /// Returns a source local's name before a Lua function starts running.
    ///
    /// # Errors
    ///
    /// Returns `MissingClosureBody` when `function` is a closure whose body is
    /// no longer live.
    // The by-value `Value` argument is part of the public debugging API.
    #[allow(clippy::needless_pass_by_value)]
    pub fn function_local_name(&self, function: Value, index: i32) -> VmResult<Option<LuaString>> {
        let Value::Closure(closure) = function else {
            return Ok(None);
        };
        let Some(index) = usize::try_from(index)
            .ok()
            .and_then(|index| index.checked_sub(1))
        else {
            return Ok(None);
        };
        let body = self.closure_body(closure)?;
        Ok(body
            .chunk()
            .debug
            .locals_at(0)
            .nth(index)
            .map(|local| local.name().clone()))
    }

    /// Returns a stack frame of a suspended coroutine, including its retained
    /// native yield frame at level zero and parked child calls.
    pub fn coroutine_stack_info(
        &self,
        thread: ThreadHandle,
        level: usize,
    ) -> Option<crate::vm::VmStackFrame> {
        let coroutine = self.coroutines.get(&thread)?;
        if coroutine.failed.is_some() {
            let info = coroutine
                .traceback
                .as_ref()?
                .debug_frames
                .get(level)?
                .clone();
            return Some(crate::vm::host::stack_frame_info(info));
        }
        let CoroutineState::Suspended {
            frames,
            host_resume,
            ..
        } = &coroutine.state
        else {
            return None;
        };
        let locations = coroutine_locations(frames, host_resume);
        let location = locations.iter().rev().nth(level)?;
        let info = match location {
            CoroutineLocation::Frame { depth, index } => {
                let stack = stack_at_depth(frames, host_resume, *depth)?;
                let level = 1 + stack
                    .iter()
                    .skip(index + 1)
                    .map(|frame| 1 + frame.tail_calls as usize)
                    .sum::<usize>();
                crate::vm::debug_info::debug_info_at_level(stack, level)?
            }
            CoroutineLocation::Native(function) => match function {
                Value::NativeFunction(function) => self.host_c_info(*function),
                function => crate::vm::debug_info::c_function_info(Some(function.clone())),
            },
            CoroutineLocation::TailCall => crate::vm::debug_info::tail_call_info(),
        };
        let mut info = crate::vm::host::stack_frame_info(info);
        if let CoroutineLocation::Frame { depth, index } = location {
            info.is_tail_call =
                stack_at_depth(frames, host_resume, *depth)?[*index].tail_calls != 0;
        }
        Some(info)
    }

    /// Reads a parked thread. Level zero is its retained C yield frame;
    /// local-bearing Lua frames follow at the same levels as `stack_info`.
    ///
    /// # Errors
    ///
    /// Returns an error when a captured-register cell read or a parked vararg
    /// table read fails, such as on a stale handle.
    pub fn coroutine_get_local(
        &self,
        thread: ThreadHandle,
        level: usize,
        index: i32,
    ) -> VmResult<Option<(LuaString, Value)>> {
        if let Some(coroutine) = self.coroutines.get(&thread)
            && let Some((frames, _)) = &coroutine.failed
        {
            let Some(frame_index) = failed_frame_index(coroutine, level) else {
                return Ok(None);
            };
            return get_local(
                &self.runtime,
                self.compatibility,
                &frames.at(frame_index),
                &[],
                index,
            );
        }
        let Some(CoroutineState::Suspended {
            frames,
            host_resume,
            ..
        }) = self.coroutines.get(&thread).map(|thread| &thread.state)
        else {
            return Ok(None);
        };
        let locations = coroutine_locations(frames, host_resume);
        let Some(CoroutineLocation::Frame {
            depth,
            index: frame_index,
        }) = locations.iter().rev().nth(level)
        else {
            return Ok(None);
        };
        let Some(stack) = stack_at_depth(frames, host_resume, *depth) else {
            return Ok(None);
        };
        get_local(
            &self.runtime,
            self.compatibility,
            &stack.at(*frame_index),
            &[],
            index,
        )
    }

    /// Writes a local of a parked thread at the same levels as
    /// `coroutine_get_local`.
    ///
    /// # Errors
    ///
    /// Returns an error when a captured-register cell write or a parked vararg
    /// table write fails, or when the thread's GC roots cannot be updated.
    ///
    /// # Panics
    ///
    /// Panics if the thread's recorded state is inconsistent: a failed thread
    /// without its saved stack, a debug location without a matching stack, or
    /// a missing thread record.
    pub fn coroutine_set_local(
        &mut self,
        thread: ThreadHandle,
        level: usize,
        index: i32,
        value: Value,
    ) -> VmResult<Option<LuaString>> {
        if let Some(coroutine) = self.coroutines.get(&thread)
            && coroutine.failed.is_some()
        {
            let Some(frame_index) = failed_frame_index(coroutine, level) else {
                return Ok(None);
            };
            let coroutine = self
                .coroutines
                .get_mut(&thread)
                .expect("failed thread exists");
            let (frames, error) = coroutine.failed.as_mut().expect("failed stack exists");
            let name = set_local(
                &mut self.runtime,
                self.compatibility,
                &mut frames.at_mut(frame_index),
                &[],
                index,
                value,
            )?
            .map(|(name, _, _, _)| name);
            if name.is_some() {
                self.local_mutation_epoch = self.local_mutation_epoch.saturating_add(1);
                let mut roots = crate::runtime::GcRootBuilder::new();
                roots.extend(&Self::gc_roots_full_windows(
                    coroutine.environment,
                    frames,
                    &[],
                ));
                if let Ok(value) = crate::vm::errors::vm_error_object(error) {
                    roots.value(&value);
                }
                self.runtime.set_thread_roots(thread, roots.finish())?;
            }
            return Ok(name);
        }
        let Some(CoroutineState::Suspended {
            frames,
            host_resume,
            ..
        }) = self
            .coroutines
            .get_mut(&thread)
            .map(|thread| &mut thread.state)
        else {
            return Ok(None);
        };
        let locations = coroutine_locations(frames, host_resume);
        let Some(CoroutineLocation::Frame {
            depth,
            index: frame_index,
        }) = locations.iter().rev().nth(level)
        else {
            return Ok(None);
        };
        let stack =
            stack_at_depth_mut(frames, host_resume, *depth).expect("debug location has a stack");
        let mut frame = stack.at_mut(*frame_index);
        let name = set_local(
            &mut self.runtime,
            self.compatibility,
            &mut frame,
            &[],
            index,
            value,
        )?
        .map(|(name, _, _, _)| name);
        if name.is_some() {
            self.local_mutation_epoch = self.local_mutation_epoch.saturating_add(1);
            let thread_state = &self.coroutines[&thread];
            let CoroutineState::Suspended {
                frames,
                host_resume,
                ..
            } = &thread_state.state
            else {
                unreachable!()
            };
            let mut roots = crate::runtime::GcRootBuilder::new();
            // The debugger may have written a "(temporary)" slot outside the
            // live set of the frame's pc. Live roots would skip that value
            // and a collection would free it under the register, so root
            // the whole stack, as for a failed one.
            roots.extend(&Self::gc_roots_full_windows(
                thread_state.environment,
                frames,
                &[],
            ));
            if let Some(resume) = host_resume {
                Self::root_host_resume(&mut roots, resume);
            }
            self.runtime.set_thread_roots(thread, roots.finish())?;
            self.coroutines
                .get_mut(&thread)
                .expect("edited thread exists")
                .roots_stale = false;
        }
        Ok(name)
    }
}

/// Failed stacks retain the native error level in their saved traceback.
/// Native levels have no Lua register owner, so skip them before locating a local.
fn failed_frame_index(coroutine: &crate::vm::Coroutine, level: usize) -> Option<usize> {
    let (frames, _) = coroutine.failed.as_ref()?;
    let native_levels = coroutine
        .traceback
        .as_ref()?
        .debug_frames
        .len()
        .saturating_sub(
            crate::vm::debug_info::capture_traceback(frames)
                .debug_frames
                .len(),
        );
    let mut remaining = level.checked_sub(native_levels)?;
    for (index, frame) in frames.iter().enumerate().rev() {
        if remaining == 0 {
            return Some(index);
        }
        remaining -= 1;
        remaining = remaining.checked_sub(frame.tail_calls as usize)?;
    }
    None
}

enum CoroutineLocation {
    Frame { depth: usize, index: usize },
    Native(Value),
    TailCall,
}

// The pair mirrors `HostResume`'s `child` fields, whose `host_resume` is an
// `Option`; taking it by reference keeps the traversal call sites uniform.
#[allow(clippy::ref_option)]
fn coroutine_locations(
    frames: &crate::vm::stack::CallStack,
    resume: &Option<HostResume>,
) -> Vec<CoroutineLocation> {
    // Same `&Option<HostResume>` shape as the enclosing traversal.
    #[allow(clippy::ref_option)]
    fn add(
        frames: &crate::vm::stack::CallStack,
        resume: &Option<HostResume>,
        depth: usize,
        output: &mut Vec<CoroutineLocation>,
    ) {
        for (index, frame) in frames.iter().enumerate() {
            if frame.chunk.debug.source_name().as_bytes() == crate::vm::host::C_SOURCE.as_bytes() {
                continue;
            }
            for _ in 0..frame.tail_calls.min(64) {
                output.push(CoroutineLocation::TailCall);
            }
            output.push(CoroutineLocation::Frame { depth, index });
        }
        if let Some(resume) = resume {
            output.push(CoroutineLocation::Native(Value::NativeFunction(
                resume.function,
            )));
            if let Some(child) = &resume.child {
                add(&child.frames, &child.host_resume, depth + 1, output);
            }
        } else {
            output.push(CoroutineLocation::Native(Value::BuiltinFunction(
                crate::runtime::BuiltinFunction::CoroutineYield,
            )));
        }
    }
    let mut output = Vec::new();
    add(frames, resume, 0, &mut output);
    output
}

// Same `&Option<HostResume>` shape as `coroutine_locations`; `stack_at_depth_mut`
// below takes the matching mutable form.
#[allow(clippy::ref_option)]
fn stack_at_depth<'a>(
    frames: &'a crate::vm::stack::CallStack,
    resume: &'a Option<HostResume>,
    depth: usize,
) -> Option<&'a crate::vm::stack::CallStack> {
    if depth == 0 {
        return Some(frames);
    }
    let child = resume.as_ref()?.child.as_ref()?;
    stack_at_depth(&child.frames, &child.host_resume, depth - 1)
}

fn stack_at_depth_mut<'a>(
    frames: &'a mut crate::vm::stack::CallStack,
    resume: &'a mut Option<HostResume>,
    depth: usize,
) -> Option<&'a mut crate::vm::stack::CallStack> {
    if depth == 0 {
        return Some(frames);
    }
    let child = resume.as_mut()?.child.as_mut()?;
    stack_at_depth_mut(&mut child.frames, &mut child.host_resume, depth - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::{
        Chunk, Constant, DebugMetadata, Instruction, LocalVariable, OpCode, UpvalueDescriptor,
        UpvalueSource,
    };
    use crate::runtime::{BuiltinFunction, RuntimeError};

    fn increment(register: u8) -> Chunk {
        Chunk::with_parts(
            2,
            0,
            vec![Constant::Number(1.0)],
            vec![
                Instruction::get_upvalue(0, 0),
                Instruction::load_constant(1, 0),
                Instruction::binary(OpCode::Add, 0, 0, 1),
                Instruction::set_upvalue(0, 0),
                Instruction::return_(0),
            ],
            Vec::new(),
            vec![UpvalueDescriptor {
                name: "captured".into(),
                source: UpvalueSource::ParentRegister(register),
            }],
        )
    }

    fn edit_locals(vm: &mut Vm, token: u64, args: &[Value]) -> VmResult<Vec<Value>> {
        assert_eq!(
            vm.host_get_local(1, 1)?,
            Some(("plain".into(), Value::Number(11.0)))
        );
        assert_eq!(
            vm.host_get_local(1, 2)?,
            Some(("captured".into(), Value::Number(22.0)))
        );
        assert_eq!(
            vm.host_set_local(1, 1, Value::Number(111.0))?,
            Some("plain".into())
        );
        assert_eq!(
            vm.host_set_local(1, 2, Value::Number(222.0))?,
            Some("captured".into())
        );
        assert_eq!(
            vm.call_value(args[0].clone(), Vec::new())?,
            vec![Value::Number(223.0)]
        );
        assert_eq!(
            vm.host_get_local(1, 2)?,
            Some(("captured".into(), Value::Number(223.0)))
        );
        assert_eq!(
            vm.host_get_local(1, 1)?,
            Some(("plain".into(), Value::Number(111.0)))
        );
        match token {
            1 => Err(RuntimeError::message("edited before error").into()),
            2 => Err(vm.request_host_yield(vec![Value::Number(333.0)])),
            _ => Ok(Vec::new()),
        }
    }

    fn local_caller(protected: bool) -> Chunk {
        let code = if protected {
            vec![
                Instruction::load_constant(0, 0),
                Instruction::load_constant(1, 1),
                Instruction::closure(2, 0),
                Instruction::get_global(3, 3),
                Instruction::get_global(4, 2),
                Instruction::move_(5, 2),
                Instruction::call(3, 2, 2),
                Instruction::return_values(0, 3),
            ]
        } else {
            vec![
                Instruction::load_constant(0, 0),
                Instruction::load_constant(1, 1),
                Instruction::closure(2, 0),
                Instruction::get_global(3, 2),
                Instruction::move_(4, 2),
                Instruction::call(3, 1, 1),
                Instruction::return_values(0, 3),
            ]
        };
        let length = code.len();
        Chunk::with_parts(
            6,
            0,
            vec![
                Constant::Number(11.0),
                Constant::Number(22.0),
                Constant::String("edit".into()),
                Constant::String("pcall".into()),
            ],
            code,
            vec![increment(1)],
            Vec::new(),
        )
        .with_debug_metadata(
            DebugMetadata::new("@locals.lua", Vec::new()).with_local_variables(vec![
                LocalVariable::new("plain", 0, 0, length),
                LocalVariable::new("captured", 1, 0, length),
                LocalVariable::new("callback", 2, 0, length),
            ]),
        )
    }

    #[test]
    fn host_debug_local_edits_survive_normal_return_error_and_yield() {
        for token in 0..=2 {
            let mut vm = Vm::with_profile(CompatibilityProfile::Lua54);
            let host = vm
                .register_host_function(edit_locals, token, Vec::new())
                .unwrap();
            vm.runtime_mut().set_global("edit", host);
            vm.runtime_mut().set_global(
                "pcall",
                Value::BuiltinFunction(BuiltinFunction::ProtectedCall),
            );
            let chunk = local_caller(token == 1);
            let results = if token == 2 {
                let entry = vm.load_chunk(&chunk).unwrap();
                let thread = vm.create_host_coroutine().unwrap();
                assert_eq!(
                    vm.start_host_coroutine(thread, entry, Vec::new()).unwrap(),
                    vec![Value::Boolean(true), Value::Number(333.0)]
                );
                let mut returned = vm.resume_host_coroutine(thread, Vec::new()).unwrap();
                returned.remove(0);
                returned
            } else {
                vm.execute(&chunk).unwrap()
            };
            assert_eq!(results[0], Value::Number(111.0));
            assert_eq!(results[1], Value::Number(223.0));
            assert_eq!(
                vm.call_value(results[2].clone(), Vec::new()).unwrap(),
                vec![Value::Number(224.0)]
            );
        }
    }

    fn edit_varargs(vm: &mut Vm, _: u64, _: &[Value]) -> VmResult<Vec<Value>> {
        assert_eq!(
            vm.host_get_local(1, -1)?,
            Some(("(*vararg)".into(), Value::Number(31.0)))
        );
        assert_eq!(
            vm.host_get_local(1, -2)?,
            Some(("(*vararg)".into(), Value::Number(32.0)))
        );
        assert_eq!(vm.host_get_local(1, -3)?, None);
        vm.host_set_local(1, -1, Value::Number(41.0))?;
        vm.host_set_local(1, -2, Value::Number(42.0))?;
        Ok(Vec::new())
    }

    #[test]
    fn host_debug_negative_indices_edit_varargs() {
        let mut vm = Vm::with_profile(CompatibilityProfile::Lua54);
        let host = vm
            .register_host_function(edit_varargs, 0, Vec::new())
            .unwrap();
        vm.runtime_mut().set_global("edit", host);
        let chunk = Chunk::with_varargs(
            4,
            1,
            vec![Constant::String("edit".into())],
            vec![
                Instruction::get_global(1, 0),
                Instruction::call(1, 0, 0),
                Instruction::vararg(2, 2),
                Instruction::return_values(2, 2),
            ],
            Vec::new(),
            Vec::new(),
            None,
        );
        let entry = vm.load_chunk(&chunk).unwrap();
        assert_eq!(
            vm.call_value(
                entry,
                vec![Value::Number(0.0), Value::Number(31.0), Value::Number(32.0)]
            )
            .unwrap(),
            vec![Value::Number(41.0), Value::Number(42.0)]
        );
    }

    fn check_unwound_locals(vm: &mut Vm, _: u64, _: &[Value]) -> VmResult<Vec<Value>> {
        assert_eq!(
            vm.host_get_local(1, 1)?,
            Some(("plain".into(), Value::Number(111.0)))
        );
        assert_eq!(
            vm.host_get_local(1, 2)?,
            Some(("captured".into(), Value::Number(223.0)))
        );
        vm.host_set_local(1, 1, Value::Number(112.0))?;
        assert_eq!(
            vm.host_get_local(1, 1)?,
            Some(("plain".into(), Value::Number(112.0)))
        );
        vm.runtime_mut()
            .set_global("unwind_debug_checked", Value::Boolean(true));
        Ok(Vec::new())
    }

    #[test]
    fn host_debug_edits_are_visible_to_popped_owner_close_handler_on_unwind() {
        let mut vm = Vm::with_profile(CompatibilityProfile::Lua54);
        let edit = vm
            .register_host_function(edit_locals, 1, Vec::new())
            .unwrap();
        let close = vm
            .register_host_function(check_unwound_locals, 0, Vec::new())
            .unwrap();
        let resource = vm.runtime_mut().create_table().unwrap();
        let metatable = vm.runtime_mut().create_table().unwrap();
        vm.runtime_mut()
            .table_set(metatable, &Value::from("__close"), close)
            .unwrap();
        vm.runtime_mut()
            .raw_set_metatable(&Value::Table(resource), Some(metatable))
            .unwrap();
        vm.runtime_mut()
            .set_global("resource", Value::Table(resource));
        vm.runtime_mut().set_global("edit", edit);
        let mut chunk = local_caller(false);
        chunk.constants.push(Constant::String("resource".into()));
        chunk.code.splice(
            3..3,
            [Instruction::get_global(5, 4), Instruction::mark_close(5)],
        );
        chunk.debug = DebugMetadata::new("@unwind.lua", Vec::new()).with_local_variables(vec![
            LocalVariable::new("plain", 0, 0, 9),
            LocalVariable::new("captured", 1, 0, 9),
            LocalVariable::new("callback", 2, 0, 9),
        ]);
        assert!(vm.execute(&chunk).is_err());
        assert_eq!(
            vm.runtime().get_global("unwind_debug_checked"),
            Value::Boolean(true)
        );
    }

    #[test]
    fn coroutine_debug_edits_scalar_capture_and_varargs_survive_gc_and_resume() {
        let mut vm = Vm::with_profile(CompatibilityProfile::Lua54);
        vm.runtime_mut().set_global(
            "yield",
            Value::BuiltinFunction(BuiltinFunction::CoroutineYield),
        );
        let chunk = Chunk::with_varargs(
            6,
            2,
            vec![Constant::String("yield".into())],
            vec![
                Instruction::closure(2, 0),
                Instruction::get_global(3, 0),
                Instruction::call(3, 0, 1),
                Instruction::vararg(4, 2),
                Instruction::return_values(0, 6),
            ],
            vec![increment(1)],
            Vec::new(),
            None,
        )
        .with_debug_metadata(
            DebugMetadata::new("@parked.lua", Vec::new()).with_local_variables(vec![
                LocalVariable::new("plain", 0, 0, 5),
                LocalVariable::new("captured", 1, 0, 5),
                LocalVariable::new("callback", 2, 0, 5),
            ]),
        );
        let entry = vm.load_chunk(&chunk).unwrap();
        assert_eq!(
            vm.function_local_name(entry.clone(), 1).unwrap(),
            Some("plain".into())
        );
        assert_eq!(vm.function_local_name(entry.clone(), -1).unwrap(), None);
        let thread = vm.create_host_coroutine().unwrap();
        vm.start_host_coroutine(
            thread,
            entry,
            vec![
                Value::Number(10.0),
                Value::Number(20.0),
                Value::Number(21.0),
                Value::Number(22.0),
            ],
        )
        .unwrap();
        assert_eq!(vm.coroutine_stack_info(thread, 0).unwrap().what, "C");
        assert_eq!(
            vm.coroutine_stack_info(thread, 1).unwrap().source,
            LuaString::from("@parked.lua")
        );
        assert_eq!(vm.coroutine_get_local(thread, 0, 1).unwrap(), None);
        assert_eq!(
            vm.coroutine_get_local(thread, 1, 1).unwrap(),
            Some(("plain".into(), Value::Number(10.0)))
        );
        let table = vm.runtime_mut().create_table().unwrap();
        vm.runtime_mut()
            .table_set(table, &Value::from("marker"), Value::Number(17.0))
            .unwrap();
        vm.coroutine_set_local(thread, 1, 1, Value::Table(table))
            .unwrap();
        vm.coroutine_set_local(thread, 1, 2, Value::Number(30.0))
            .unwrap();
        vm.coroutine_set_local(thread, 1, -1, Value::Number(40.0))
            .unwrap();
        vm.coroutine_set_local(thread, 1, -2, Value::Number(50.0))
            .unwrap();
        let (_, callback) = vm.coroutine_get_local(thread, 1, 3).unwrap().unwrap();
        assert_eq!(
            vm.call_value(callback, Vec::new()).unwrap(),
            vec![Value::Number(31.0)]
        );
        vm.collect_full_with_values(&[Value::Thread(thread)]);
        let results = vm.resume_host_coroutine(thread, Vec::new()).unwrap();
        assert_eq!(results[1], Value::Table(table));
        assert_eq!(results[2], Value::Number(31.0));
        assert_eq!(&results[5..], &[Value::Number(40.0), Value::Number(50.0)]);
        assert_eq!(
            vm.runtime()
                .table_get(table, &Value::from("marker"))
                .unwrap(),
            Value::Number(17.0)
        );
    }
}
