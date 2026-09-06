//! Call-site, traceback, stack-level, and `debug` library introspection helpers.

use crate::bytecode::{CallKind, Chunk, OpCode, RegisterSet};
use crate::compat::CompatibilityProfile;
use crate::runtime::{ClosureHandle, NumberFormat, RuntimeError, TableHandle, Value};
use crate::string::LuaString;
use crate::vm::frame::FrameChunk;
use crate::vm::host::C_SOURCE;
use crate::vm::libraries::Window;
use crate::vm::stack::{CallStack, FrameRef};
use crate::vm::{
    ActiveCallSite, DebugInfo, Frame, FrameSite, LogicalFrame, TraceExtra, TraceFrame, TraceKind,
    Traceback, VmResult,
};
use std::fmt::Write;
use std::sync::Arc;

pub(crate) fn active_call_site(frame: &Frame) -> Option<ActiveCallSite> {
    call_site_at(&frame.chunk, frame.pc)
}

/// The named call expression that `pc` has just executed past.
pub(crate) fn call_site_at(chunk: &Chunk, pc: usize) -> Option<ActiveCallSite> {
    let pc = pc.checked_sub(1)?;
    let call_site = chunk.debug.call_site_at(pc)?;
    let kind = call_site.kind()?;
    let name = match call_site.name() {
        Some(name) => name.clone(),
        // The compiler stores no name for the generic-`for` iterator call.
        None if kind == CallKind::ForIterator => LuaString::from("for iterator"),
        None => return None,
    };
    Some(ActiveCallSite { name, kind })
}

/// Names the activation of a tail call after the call at `pc - 1` of the
/// activation it replaces, which runs `closure` and `chunk`, reusing the box
/// `slot` already holds, or `spare`. The name is looked up when something
/// asks (see [`FrameSite::Tail`]), so the tail call does no lookup and takes
/// no reference count.
///
/// The replaced closure stays rooted through the site, as long as it lives.
#[inline(always)]
pub(crate) fn set_call_site(
    slot: &mut Option<Box<FrameSite>>,
    spare: &mut Option<Box<FrameSite>>,
    closure: Option<ClosureHandle>,
    chunk: &FrameChunk,
    pc: usize,
) {
    // SAFETY: `closure` runs `chunk` and is rooted with the frame until the
    // frame's closure is replaced. `Vm::root_frame` roots the closure of the
    // site too, so the copy of the borrowed chunk stays valid for as long as
    // the site is read.
    let tail = closure.and_then(|origin| {
        #[allow(unsafe_code)]
        let chunk = unsafe { chunk.borrowed_copy() }?;
        Some(FrameSite::Tail {
            origin,
            chunk,
            // Validation keeps every chunk's code within `u32`.
            pc: u32::try_from(pc).ok()?,
        })
    });
    let Some(site) = tail else {
        set_call_site_named(slot, spare, chunk, pc);
        return;
    };
    match slot {
        Some(held) => replace_site(held, site),
        None => match spare.take() {
            Some(mut held) => {
                replace_site(&mut held, site);
                // SAFETY: `slot` holds `None`, which owns nothing to drop.
                #[allow(unsafe_code)]
                unsafe {
                    std::ptr::write(slot, Some(held));
                }
            }
            None => *slot = Some(Box::new(site)),
        },
    }
}

/// [`set_call_site`] for an activation of `origin` that borrows its chunk,
/// at an instruction whose index fits `u32` (validation keeps every chunk's
/// code within it): a tail site, with no test for the other cases.
///
/// # Safety
///
/// `origin` runs `chunk` and is rooted with the frame, `chunk` is a borrowed
/// view ([`FrameChunk::is_owned`] is false), and `pc` fits `u32`.
#[allow(unsafe_code, clippy::cast_possible_truncation)]
#[inline(always)]
pub(crate) unsafe fn set_tail_site(
    slot: &mut Option<Box<FrameSite>>,
    spare: &mut Option<Box<FrameSite>>,
    origin: ClosureHandle,
    chunk: &FrameChunk,
    pc: usize,
) {
    debug_assert!(u32::try_from(pc).is_ok());
    let site = FrameSite::Tail {
        origin,
        // SAFETY: the contract.
        chunk: unsafe { chunk.borrowed_copy_of_borrowed() },
        pc: pc as u32,
    };
    match slot {
        Some(held) => replace_site(held, site),
        None => match spare.take() {
            Some(mut held) => {
                replace_site(&mut held, site);
                // SAFETY: `slot` holds `None`, which owns nothing to drop.
                unsafe {
                    std::ptr::write(slot, Some(held));
                }
            }
            None => *slot = Some(Box::new(site)),
        },
    }
}

/// `*held = site`, with no drop call when `held` is a tail site, which owns
/// nothing.
#[inline(always)]
fn replace_site(held: &mut FrameSite, site: FrameSite) {
    if matches!(held, FrameSite::Named(_)) {
        *held = site;
    } else {
        // SAFETY: a tail site owns nothing: its chunk is a borrowed view
        // (`FrameChunk::borrowed_copy`), which drops as a no-op.
        #[allow(unsafe_code)]
        unsafe {
            std::ptr::write(held, site);
        }
    }
}

/// [`set_call_site`] for an activation that cannot hand its closure on:
/// looks the name up now.
#[cold]
#[inline(never)]
fn set_call_site_named(
    slot: &mut Option<Box<FrameSite>>,
    spare: &mut Option<Box<FrameSite>>,
    chunk: &Chunk,
    pc: usize,
) {
    let site = call_site_at(chunk, pc).map(FrameSite::boxed);
    match (site, slot) {
        (Some(site), Some(held)) => **held = *site,
        (Some(site), slot) => match spare.take() {
            Some(mut held) => {
                *held = *site;
                *slot = Some(held);
            }
            None => *slot = Some(site),
        },
        (None, slot) => *slot = None,
    }
}

fn frame_call_site(frames: &CallStack, index: usize) -> Option<ActiveCallSite> {
    let frame = &frames[index];
    if let Some(call_site) = frame.current_call_site().and_then(FrameSite::resolve) {
        Some(call_site)
    } else if frame.tail_calls == 0
        && matches!(frame.caller, Some(crate::vm::ReturnTarget::Direct(_)))
    {
        index
            .checked_sub(1)
            .and_then(|parent| active_call_site(&frames[parent]))
    } else {
        None
    }
}

pub(crate) fn active_diagnostic_site(frame: &Frame, register: u8) -> Option<ActiveCallSite> {
    let pc = frame.pc.checked_sub(1)?;
    let site = frame
        .chunk
        .debug
        .diagnostic_sites()
        .iter()
        .find(|site| site.pc() == pc && site.register() == register)?;
    Some(ActiveCallSite {
        name: site.name().clone(),
        kind: site.kind(),
    })
}

pub(crate) fn debug_level(value: f64) -> Option<usize> {
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
        return None;
    }
    format!("{value:.0}").parse().ok()
}

/// Numeric level or count argument. Lua 5.3+ profiles pass these as
/// integers; Lua 5.1 profiles pass floats.
#[allow(clippy::cast_precision_loss)]
pub(crate) const fn level_number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(value) => Some(*value),
        Value::Integer(value) => Some(*value as f64),
        _ => None,
    }
}

pub(crate) fn value_debug_level(value: &Value) -> Option<usize> {
    let value = level_number(value)?;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return None;
    }
    format!("{value:.0}").parse().ok()
}

/// A `debug.sethook` count at argument `position`.
pub(crate) fn hook_count(value: &Value, position: usize) -> VmResult<u32> {
    let Some(value) = level_number(value) else {
        return Err(RuntimeError::bad_argument(position, "number", Some(value)).into());
    };
    if !value.is_finite()
        || value < 0.0
        || value.fract() != 0.0
        || value > f64::from(0x00ff_ffff_u32)
    {
        return Err(RuntimeError::argument_error(position, "count out of range").into());
    }
    format!("{value:.0}")
        .parse()
        .map_err(|_| RuntimeError::argument_error(position, "count out of range").into())
}

pub(crate) fn traceback_level(value: &Value) -> Option<usize> {
    match level_number(value) {
        Some(value) if value.is_finite() && value >= 0.0 && value.fract() == 0.0 => {
            format!("{value:.0}").parse().ok()
        }
        _ => None,
    }
}

#[allow(clippy::option_option)]
pub(crate) fn traceback_message(value: &Value, numbers: NumberFormat) -> Option<Option<String>> {
    match value {
        Value::Nil => Some(None),
        Value::String(value) => Some(Some(value.to_lossy().into_owned())),
        Value::Number(value) => Some(Some(
            String::from_utf8_lossy(&numbers.format(*value)).into_owned(),
        )),
        Value::Integer(value) => Some(Some(value.to_string())),
        _ => None,
    }
}

/// The event name (`index`, `add`, ...) of the metamethod that the
/// instruction `frame` is executing can call, or `None` for any other.
pub(crate) fn metamethod_event(frame: &Frame) -> Option<&'static str> {
    let pc = frame
        .last_interpreted_pc
        .get()
        .or_else(|| frame.pc.checked_sub(1))?;
    let opcode = frame.chunk.code.get(pc)?.opcode().ok()?;
    Some(match opcode {
        OpCode::GetTable
        | OpCode::GetField
        | OpCode::SelfField
        | OpCode::GetTabUp
        | OpCode::GetGlobal
        | OpCode::GetGlobalWide => "index",
        OpCode::SetTable
        | OpCode::SetField
        | OpCode::SetTabUp
        | OpCode::SetGlobal
        | OpCode::SetGlobalWide => "newindex",
        OpCode::Add | OpCode::AddK => "add",
        OpCode::Subtract | OpCode::SubtractK => "sub",
        OpCode::Multiply | OpCode::MultiplyK => "mul",
        OpCode::Divide | OpCode::DivideK => "div",
        OpCode::Modulo | OpCode::ModuloK => "mod",
        OpCode::Power => "pow",
        OpCode::FloorDivide => "idiv",
        OpCode::BitAnd => "band",
        OpCode::BitOr => "bor",
        OpCode::BitXor => "bxor",
        OpCode::ShiftLeft => "shl",
        OpCode::ShiftRight => "shr",
        OpCode::Negate => "unm",
        OpCode::BitNot => "bnot",
        OpCode::Length => "len",
        OpCode::Concat => "concat",
        OpCode::Equal | OpCode::TestEqual | OpCode::TestEqualK => "eq",
        OpCode::LessThan | OpCode::TestLessThan | OpCode::TestLessThanK => "lt",
        OpCode::LessEqual | OpCode::TestLessEqual | OpCode::TestLessEqualK => "le",
        _ => return None,
    })
}

/// The qualified name of the native function that a frame's caller target
/// stands for, for the `[C]` level between that frame and the one below.
fn native_caller_name(target: &crate::vm::ReturnTarget) -> Option<&'static str> {
    use crate::vm::ReturnTarget as Target;
    Some(match target {
        Target::Protected(_) => "pcall",
        Target::ExtendedProtectedTarget(_) => "xpcall",
        Target::TableSort(_) | Target::ProtectedSort(_) => "table.sort",
        Target::TableForeach(_) => "table.foreach",
        Target::StringGsub(_) => "string.gsub",
        Target::Require(_) | Target::RequireSearch(_) => "require",
        Target::Module(_) => "module",
        Target::Dofile(_) => "dofile",
        Target::LoadReader(_) => "load",
        Target::StringFormat(_) => "string.format",
        Target::ToString(_) => "tostring",
        _ => return None,
    })
}

/// The traceback level of a Lua frame: its position, its name where the
/// caller's instruction gives one, and the tail calls that replaced its
/// caller.
fn lua_trace_frame(
    frames: &CallStack,
    index: usize,
    frame: &Frame,
    call_site: Option<ActiveCallSite>,
) -> TraceFrame {
    let metamethod = match frame.caller {
        Some(
            crate::vm::ReturnTarget::Index(_)
            | crate::vm::ReturnTarget::ProtectedIndex(_)
            | crate::vm::ReturnTarget::SetMetamethod
            | crate::vm::ReturnTarget::IndexResult(_)
            | crate::vm::ReturnTarget::Scalar(_)
            | crate::vm::ReturnTarget::Operator(_)
            | crate::vm::ReturnTarget::ConcatRange(_),
        ) => index
            .checked_sub(1)
            .and_then(|parent| metamethod_event(&frames[parent])),
        _ => None,
    };
    TraceFrame {
        source: frame.chunk.debug.source_name().clone(),
        line: frame
            .last_interpreted_pc
            .get()
            .or_else(|| frame.pc.checked_sub(1))
            .and_then(|pc| frame.chunk.debug.line_at(pc)),
        call_site,
        is_main: frame.chunk.debug.definition_line() == 0 && !is_c_chunk(&frame.chunk),
        extra: TraceExtra {
            kind: TraceKind::Lua,
            function: frame.closure.map(Value::Closure),
            definition_line: Some(frame.chunk.debug.definition_line()),
            metamethod,
            native_name: None,
            tail_called: frame.tail_calls > 0,
            caller_site: (frame.tail_calls > 0
                && matches!(frame.caller, Some(crate::vm::ReturnTarget::Direct(_)))
                || matches!(frame.caller, Some(crate::vm::ReturnTarget::ToString(_))))
            .then(|| {
                index
                    .checked_sub(1)
                    .filter(|parent| !crate::vm::errors::is_tail_call(&frames[*parent]))
                    .and_then(|parent| active_call_site(&frames[parent]))
            })
            .flatten(),
            in_tail_call: crate::vm::errors::is_tail_call(frame),
            luajit_tail: None,
        },
    }
}

/// A native level, named by the call that reached it.
pub(crate) fn native_trace_frame(
    function: Option<Value>,
    native_name: Option<&'static str>,
    call_site: Option<ActiveCallSite>,
) -> TraceFrame {
    TraceFrame {
        source: LuaString::from("=[C]"),
        line: None,
        call_site,
        is_main: false,
        extra: TraceExtra {
            kind: TraceKind::Native,
            function,
            definition_line: None,
            metamethod: None,
            native_name,
            tail_called: false,
            caller_site: None,
            in_tail_call: false,
            luajit_tail: None,
        },
    }
}

/// A level that tail calls replaced.
pub(crate) fn tail_trace_frame() -> TraceFrame {
    TraceFrame {
        source: LuaString::from("=(tail call)"),
        line: None,
        call_site: None,
        is_main: false,
        extra: TraceExtra {
            kind: TraceKind::Tail,
            ..TraceExtra::default()
        },
    }
}

/// The level below the main chunk, where the host called into Lua.
pub(crate) fn base_trace_frame() -> TraceFrame {
    TraceFrame {
        source: LuaString::from("=[C]"),
        line: None,
        call_site: None,
        is_main: false,
        extra: TraceExtra {
            kind: TraceKind::Base,
            ..TraceExtra::default()
        },
    }
}

pub(crate) fn capture_traceback(frames: &CallStack) -> Traceback {
    capture_traceback_above(frames, 0)
}

/// [`capture_traceback`] for the frames from index `lowest` up.
pub(crate) fn capture_traceback_above(frames: &CallStack, lowest: usize) -> Traceback {
    let mut trace_frames = Vec::new();
    let mut debug_frames = Vec::new();
    for (index, frame) in frames
        .iter()
        .enumerate()
        .rev()
        .take_while(|(index, _)| *index >= lowest)
    {
        debug_frames.push(
            frame
                .extras_ref()
                .and_then(|extras| extras.debug_hook_subject.as_ref())
                .map_or_else(
                    || {
                        lua_frame_info(
                            &frame.chunk.as_arc(),
                            frame.closure,
                            current_line_pc(frame).and_then(|pc| frame.chunk.debug.line_at(pc)),
                            frame_call_site(frames, index),
                        )
                    },
                    |subject| hook_subject_info(subject.clone()),
                ),
        );
        let call_site = frame_call_site(frames, index);
        if is_c_chunk(&frame.chunk) {
            trace_frames.push(native_trace_frame(None, None, call_site));
        } else {
            trace_frames.push(lua_trace_frame(frames, index, frame, call_site));
        }
        for _ in 0..frame.tail_calls {
            debug_frames.push(tail_call_info());
            trace_frames.push(tail_trace_frame());
        }
        // A function that a native called has that native as its caller, a
        // level of its own between it and the Lua frame below.
        if let Some(name) = frame
            .caller
            .as_ref()
            .and_then(native_caller_name)
            .filter(|_| !is_c_chunk(&frame.chunk))
        {
            let call_site = index
                .checked_sub(1)
                .and_then(|parent| active_call_site(&frames[parent]));
            trace_frames.push(native_trace_frame(None, Some(name), call_site));
        }
    }
    Traceback {
        debug_frames,
        frames: trace_frames,
    }
}

pub(crate) fn debug_info_at_level(frames: &CallStack, level: usize) -> Option<DebugInfo> {
    let mut remaining = level;
    for frame in frames.iter().rev() {
        if remaining == 1 {
            break;
        }
        remaining = remaining.checked_sub(1)?;
        if remaining <= frame.tail_calls as usize {
            return Some(tail_call_info());
        }
        remaining -= frame.tail_calls as usize;
        if let Some(traceback) = frame
            .extras_ref()
            .and_then(|extras| extras.traceback.as_ref())
        {
            return traceback
                .debug_frames
                .get(remaining.checked_sub(1)?)
                .cloned();
        }
    }
    let (index, on_frame) = locate_level(
        frames.len(),
        |index| frames[index].tail_calls as usize,
        level,
    )?;
    if !on_frame {
        return Some(tail_call_info());
    }
    let frame = &frames[index];
    if let Some(subject) = frame
        .extras_ref()
        .and_then(|extras| extras.debug_hook_subject.as_ref())
    {
        let mut info = hook_subject_info(subject.clone());
        info.hook_transfer = frame
            .extras_ref()
            .map_or((0, 0), |extras| extras.debug_hook_transfer);
        return Some(info);
    }
    let mut info = lua_frame_info(
        &frame.chunk.as_arc(),
        frame.closure,
        current_line_pc(frame).and_then(|pc| frame.chunk.debug.line_at(pc)),
        frame_call_site(frames, index),
    );
    info.hook_transfer = frame
        .extras_ref()
        .map_or((0, 0), |extras| extras.debug_hook_transfer);
    Some(info)
}

/// Finds the frame holding stack `level` (1 is the innermost). The flag is
/// false when the level names one of the tail calls that frame replaced.
fn locate_level(
    count: usize,
    tail_calls: impl Fn(usize) -> usize,
    mut level: usize,
) -> Option<(usize, bool)> {
    for index in (0..count).rev() {
        if level == 1 {
            return Some((index, true));
        }
        level -= 1;
        let tail_calls = tail_calls(index);
        if level <= tail_calls {
            return Some((index, false));
        }
        level -= tail_calls;
    }
    None
}

/// The instruction whose line a frame reports. A line hook paused before
/// `pc` runs reports `pc` itself; otherwise the last executed instruction.
fn current_line_pc(frame: &Frame) -> Option<usize> {
    if frame
        .extras_ref()
        .is_some_and(|extras| extras.skip_hook_pc_once == Some(frame.pc))
    {
        Some(frame.pc)
    } else {
        frame.pc.checked_sub(1)
    }
}

/// Whether `chunk` stands in for a C function; see `builtin_call_chunk`.
pub(crate) fn is_c_chunk(chunk: &Chunk) -> bool {
    chunk.debug.source_name().as_bytes() == C_SOURCE.as_bytes()
}

fn lua_frame_info(
    chunk: &Arc<Chunk>,
    closure: Option<ClosureHandle>,
    current_line: Option<u32>,
    call_site: Option<ActiveCallSite>,
) -> DebugInfo {
    if is_c_chunk(chunk) {
        return DebugInfo {
            call_site,
            ..c_function_info(None)
        };
    }
    DebugInfo {
        function: closure.map(Value::Closure),
        source: chunk.debug.source_name().clone(),
        current_line,
        call_site,
        what: if chunk.debug.definition_line() == 0 {
            "main"
        } else {
            "Lua"
        },
        definition_line: i32::try_from(chunk.debug.definition_line()).unwrap_or(-1),
        last_definition_line: i32::try_from(chunk.debug.last_definition_line()).unwrap_or(-1),
        upvalue_count: chunk.upvalues.len(),
        hook_transfer: (0, 0),
        chunk: Some(chunk.clone()),
    }
}

fn hook_subject_info(subject: Value) -> DebugInfo {
    c_function_info(Some(subject))
}

pub(crate) fn c_function_info(function: Option<Value>) -> DebugInfo {
    DebugInfo {
        function,
        source: LuaString::from(C_SOURCE),
        current_line: None,
        call_site: None,
        what: "C",
        definition_line: -1,
        last_definition_line: -1,
        upvalue_count: 0,
        hook_transfer: (0, 0),
        chunk: None,
    }
}

pub(crate) fn tail_call_info() -> DebugInfo {
    DebugInfo {
        function: None,
        source: LuaString::from("=(tail call)"),
        current_line: None,
        call_site: None,
        what: "tail",
        definition_line: -1,
        last_definition_line: -1,
        upvalue_count: 0,
        hook_transfer: (0, 0),
        chunk: None,
    }
}

/// A Lua frame suspended under a host callback, as `debug.getinfo` and
/// [`oximond::vm::Vm::host_stack_info`] see it from inside the callback.
///
/// Capturing clones a few handles per frame. Source names, line numbers and
/// call-site names are resolved only when the host actually asks, so calling
/// a host function costs no debug-info work.
#[derive(Clone)]
pub(crate) struct SuspendedFrame {
    chunk: Arc<Chunk>,
    closure: Option<ClosureHandle>,
    pc: usize,
    line_pc: Option<usize>,
    tail_calls: usize,
    /// The call site recorded on the frame itself, if any.
    call_site: Option<ActiveCallSite>,
    /// Whether the frame is named by its caller's active call instruction.
    named_by_caller: bool,
    hook_subject: Option<Value>,
    hook_transfer: (usize, usize),
}

impl SuspendedFrame {
    pub(crate) fn capture(frame: &Frame) -> Self {
        let extras = frame.extras_ref();
        Self {
            chunk: frame.chunk.as_arc().clone(),
            closure: frame.closure,
            pc: frame.pc,
            line_pc: current_line_pc(frame),
            tail_calls: frame.tail_calls as usize,
            call_site: frame.current_call_site().and_then(FrameSite::resolve),
            named_by_caller: frame.tail_calls == 0
                && matches!(frame.caller, Some(crate::vm::ReturnTarget::Direct(_))),
            hook_subject: extras.and_then(|extras| extras.debug_hook_subject.clone()),
            hook_transfer: extras.map_or((0, 0), |extras| extras.debug_hook_transfer),
        }
    }
}

/// The number of stack levels a suspended stack exposes. Like a live
/// snapshot, long tail-call chains are cut off 64 levels past the frames.
pub(crate) fn suspended_level_count(frames: &[SuspendedFrame]) -> usize {
    frames
        .iter()
        .map(|frame| frame.tail_calls.saturating_add(1))
        .fold(0, usize::saturating_add)
        .min(frames.len().saturating_add(64))
}

pub(crate) fn suspended_info_at_level(
    frames: &[SuspendedFrame],
    level: usize,
) -> Option<DebugInfo> {
    let (index, on_frame) = locate_level(frames.len(), |index| frames[index].tail_calls, level)?;
    if !on_frame {
        return Some(tail_call_info());
    }
    let frame = &frames[index];
    if let Some(subject) = &frame.hook_subject {
        let mut info = hook_subject_info(subject.clone());
        info.hook_transfer = frame.hook_transfer;
        return Some(info);
    }
    let call_site = if frame.call_site.is_some() {
        frame.call_site.clone()
    } else if frame.named_by_caller {
        index.checked_sub(1).and_then(|parent| {
            let parent = &frames[parent];
            call_site_at(&parent.chunk, parent.pc)
        })
    } else {
        None
    };
    let mut info = lua_frame_info(
        &frame.chunk,
        frame.closure,
        frame.line_pc.and_then(|pc| frame.chunk.debug.line_at(pc)),
        call_site,
    );
    info.hook_transfer = frame.hook_transfer;
    Some(info)
}

/// [`format_traceback_for`] in the Lua 5.1 style, without global function
/// names.
#[cfg(test)]
pub(crate) fn format_traceback(message: Option<&str>, trace: &Traceback, level: usize) -> String {
    format_traceback_for(
        message,
        trace,
        level,
        1,
        CompatibilityProfile::Lua51,
        &|_: &Value| None,
    )
}

/// The `stack traceback:` text of `trace`, as the stock library of
/// `profile` writes it, skipping the innermost `level` levels. The first level
/// of `trace` is stack level `first_level` (1 below the running function, 0
/// for another thread). `global_name` gives the name Lua 5.2 and later find for a function in
/// `package.loaded`.
pub(crate) fn format_traceback_for(
    message: Option<&str>,
    trace: &Traceback,
    level: usize,
    first_level: usize,
    profile: CompatibilityProfile,
    global_name: &dyn Fn(&Value) -> Option<String>,
) -> String {
    let mut output = String::new();
    if let Some(message) = message {
        output.push_str(message);
        output.push('\n');
    }
    output.push_str("stack traceback:");
    // Only Lua 5.1 lists the levels that tail calls replaced.
    let lists_tails = profile == CompatibilityProfile::Lua51;
    let levels = trace
        .frames
        .iter()
        .filter(|frame| lists_tails || frame.extra.kind != TraceKind::Tail)
        // LuaJIT reuses the frame of a function that tail-calls a native, and
        // calls `tostring`'s `__tostring` without a native level.
        .filter(|frame| {
            profile != CompatibilityProfile::LuaJit21
                || !(frame.extra.in_tail_call
                    || frame.extra.kind == TraceKind::Native
                        && frame.extra.native_name == Some("tostring"))
        })
        .skip(level)
        .collect::<Vec<_>>();
    // `luaL_traceback` cuts a long stack in the middle. Lua 5.1 and LuaJIT
    // count from the first stack level, so they keep one level less of a
    // traceback that starts at level 1; Lua 5.4 and later count from the
    // first level shown.
    let count = levels.len();
    let (head_end, resume, hidden_message) = match profile {
        CompatibilityProfile::Lua51 | CompatibilityProfile::LuaJit21 => {
            let start = first_level + level.min(trace.frames.len());
            let head = 12usize.saturating_sub(start);
            if head + 11 <= count.saturating_sub(1) {
                (head, count - 10, None)
            } else {
                (count, count, None)
            }
        }
        CompatibilityProfile::Lua52 | CompatibilityProfile::Lua53 => {
            // Marks absolute level 12 when the stack has more than 22 levels,
            // then resumes 10 levels above the last one.
            let start = first_level + level.min(trace.frames.len());
            let last = first_level + trace.frames.len().saturating_sub(1);
            if last > 22 && start <= 11 {
                (
                    11 - start,
                    (last - 10).saturating_sub(start).max(11 - start),
                    None,
                )
            } else {
                (count, count, None)
            }
        }
        _ => {
            if count > 22 {
                (10, count - 11, Some(count - 22))
            } else {
                (count, count, None)
            }
        }
    };
    for (index, frame) in levels.iter().enumerate() {
        if index >= head_end && index < resume {
            if index == head_end {
                match hidden_message {
                    Some(skipped) => {
                        write!(output, "\n\t...\t(skipping {skipped} levels)")
                            .expect("formatting into a String cannot fail");
                    }
                    None => output.push_str("\n\t..."),
                }
            }
            continue;
        }
        output.push_str("\n\t");
        output.push_str(&trace_level_text(frame, profile, global_name));
        if frame.extra.tail_called && !lists_tails && profile != CompatibilityProfile::LuaJit21 {
            output.push_str("\n\t(...tail calls...)");
        }
    }
    output
}

/// One traceback level, without its leading tab.
fn trace_level_text(
    frame: &TraceFrame,
    profile: CompatibilityProfile,
    global_name: &dyn Fn(&Value) -> Option<String>,
) -> String {
    let modern = !profile.is_lua51_family();
    let luajit = profile == CompatibilityProfile::LuaJit21;
    // Lua 5.2 calls every name from code a function.
    let plain_names = profile == CompatibilityProfile::Lua52;
    let mut output = String::new();
    match frame.extra.kind {
        TraceKind::Tail => return "(tail call): ?".to_owned(),
        TraceKind::Base => {
            return if luajit {
                "[C]: at 0x0".to_owned()
            } else if modern {
                "[C]: in ?".to_owned()
            } else {
                "[C]: ?".to_owned()
            };
        }
        TraceKind::Native => {
            output.push_str("[C]: ");
            let global = frame.extra.function.as_ref().and_then(global_name);
            let site = frame.call_site.as_ref();
            if luajit && let Some(tail) = &frame.extra.luajit_tail {
                match tail {
                    crate::vm::LuajitTailName::Event(event) => {
                        write!(output, "in function '__{event}'")
                            .expect("formatting into a String cannot fail");
                    }
                    crate::vm::LuajitTailName::Site(site) => {
                        write!(output, "in function '{}'", site.display_name(profile))
                            .expect("formatting into a String cannot fail");
                    }
                    crate::vm::LuajitTailName::Unnamed => output.push_str("at 0x0"),
                }
                return output;
            }
            if let Some(event) = frame.extra.metamethod {
                // A native that is itself the handler: only the event names it.
                // Lua 5.4 still prefers its `package.loaded` name.
                if luajit || plain_names {
                    write!(output, "in function '__{event}'")
                        .expect("formatting into a String cannot fail");
                } else if modern {
                    match global.filter(|_| profile == CompatibilityProfile::Lua54) {
                        Some(name) => write!(output, "in function '{name}'")
                            .expect("formatting into a String cannot fail"),
                        None => write!(output, "in metamethod '{event}'")
                            .expect("formatting into a String cannot fail"),
                    }
                } else {
                    output.push('?');
                }
                return output;
            }
            if modern {
                // Lua 5.5 prefers the name the call gives over the global one.
                let global = global.or_else(|| frame.extra.native_name.map(str::to_owned));
                let global =
                    global.filter(|_| !(site.is_some() && profile == CompatibilityProfile::Lua55));
                if let Some(name) = global {
                    write!(output, "in function '{name}'")
                        .expect("formatting into a String cannot fail");
                } else if let Some(site) = site {
                    // Lua 5.4 finds a global's function in `package.loaded._G`
                    // before it looks at the call.
                    let what = if plain_names
                        || profile == CompatibilityProfile::Lua54 && site.kind == CallKind::Global
                    {
                        "function"
                    } else {
                        debug_namewhat(site.kind)
                    };
                    write!(output, "in {what} '{}'", site.display_name(profile))
                        .expect("formatting into a String cannot fail");
                } else {
                    output.push_str("in ?");
                }
            } else if let Some(site) = site {
                write!(output, "in function '{}'", site.display_name(profile))
                    .expect("formatting into a String cannot fail");
            } else if let Some(name) = frame.extra.native_name {
                let short = name.rsplit('.').next().unwrap_or(name);
                write!(output, "in function '{short}'")
                    .expect("formatting into a String cannot fail");
            } else if luajit {
                output.push_str("at 0x0");
            } else {
                output.push('?');
            }
            return output;
        }
        TraceKind::Lua => {}
    }
    let source = debug_short_source(&frame.source).to_lossy().into_owned();
    output.push_str(&source);
    if let Some(line) = frame.line {
        output.push(':');
        output.push_str(&line.to_string());
    }
    output.push_str(": in ");
    // Tail calls replaced the caller, so no call names the function, except
    // in LuaJIT, which names the frame after the call that entered it.
    let site = if luajit && frame.extra.tail_called {
        frame.extra.caller_site.as_ref()
    } else if !frame.extra.tail_called {
        frame.call_site.as_ref()
    } else {
        None
    };
    // LuaJIT calls `__tostring` in place of `tostring`'s own frame.
    let site = site.or(frame.extra.caller_site.as_ref().filter(|_| luajit));
    let by_site_first = profile == CompatibilityProfile::Lua55 && site.is_some();
    if modern
        && !plain_names
        && !by_site_first
        && let Some(name) = frame.extra.function.as_ref().and_then(global_name)
    {
        write!(output, "function '{name}'").expect("formatting into a String cannot fail");
    } else if modern
        && !plain_names
        && !frame.extra.tail_called
        && let Some(event) = frame.extra.metamethod
    {
        write!(output, "metamethod '{event}'").expect("formatting into a String cannot fail");
    } else if (luajit || plain_names)
        && let Some(event) = frame.extra.metamethod
    {
        write!(output, "function '__{event}'").expect("formatting into a String cannot fail");
    } else if let Some(site) = site {
        if modern && !plain_names {
            write!(
                output,
                "{} '{}'",
                debug_namewhat(site.kind),
                site.display_name(profile)
            )
            .expect("formatting into a String cannot fail");
        } else {
            write!(output, "function '{}'", site.display_name(profile))
                .expect("formatting into a String cannot fail");
        }
    } else if frame.is_main {
        output.push_str("main chunk");
    } else {
        let line = frame
            .extra
            .definition_line
            .filter(|line| *line > 0)
            .or(frame.line)
            .unwrap_or(0);
        write!(output, "function <{source}:{line}>").expect("formatting into a String cannot fail");
    }
    output
}

pub(crate) fn frame_environment(
    frames: &CallStack,
    level: f64,
    global: TableHandle,
) -> VmResult<TableHandle> {
    let Some(level) = fenv_level(level) else {
        return Err(RuntimeError::argument_error(1, "invalid level").into());
    };
    if level == 0 {
        return Ok(global);
    }
    match logical_frame_at(frames, level) {
        LogicalFrame::Frame(frame) => Ok(frame.environment),
        LogicalFrame::TailCall | LogicalFrame::OutOfRange => {
            Err(RuntimeError::argument_error(1, "level out of range").into())
        }
    }
}

pub(crate) fn logical_frame_at(frames: &CallStack, mut level: usize) -> LogicalFrame<'_> {
    for frame in frames.iter().rev() {
        if level == 1 {
            return LogicalFrame::Frame(frame);
        }
        level -= 1;
        for _ in 0..frame.tail_calls {
            if level == 1 {
                return LogicalFrame::TailCall;
            }
            level -= 1;
        }
    }
    LogicalFrame::OutOfRange
}

pub(crate) fn set_frame_environment(
    frames: &mut CallStack,
    level: f64,
    environment: TableHandle,
) -> VmResult<Option<ClosureHandle>> {
    let Some(level) = fenv_level(level) else {
        return Err(RuntimeError::argument_error(1, "invalid level").into());
    };
    if level == 0 {
        return Err(RuntimeError::message("cannot change the environment of a thread").into());
    }
    let frame_index = logical_frame_index(frames, level)
        .ok_or_else(|| RuntimeError::argument_error(1, "level out of range"))?;
    let frame = &mut frames[frame_index];
    frame.environment = environment;
    Ok(frame.closure)
}

/// Resolves a Lua stack level to its physical frame. Proper tail calls leave a
/// logical placeholder in the stack, but that placeholder has no function
/// environment for `setfenv` to modify.
pub(crate) fn logical_frame_index(frames: &CallStack, mut level: usize) -> Option<usize> {
    for (index, frame) in frames.iter().enumerate().rev() {
        if level == 1 {
            return Some(index);
        }
        level -= 1;
        for _ in 0..frame.tail_calls {
            if level == 1 {
                return None;
            }
            level -= 1;
        }
    }
    None
}

pub(crate) fn fenv_level(value: f64) -> Option<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return None;
    }
    format!("{value:.0}").parse().ok()
}

pub(crate) fn value_index(value: &Value) -> Option<usize> {
    match value {
        Value::Integer(value) => usize::try_from(*value).ok()?.checked_sub(1),
        Value::Number(value) => {
            if !value.is_finite() || *value < 1.0 || value.fract() != 0.0 {
                return None;
            }
            format!("{value:.0}").parse::<usize>().ok()?.checked_sub(1)
        }
        _ => None,
    }
}

/// The Lua function at argument `position` of `debug.upvalueid` or
/// `debug.upvaluejoin`.
pub(crate) fn debug_closure_argument(
    value: Option<&Value>,
    position: usize,
) -> VmResult<ClosureHandle> {
    match value {
        Some(Value::Closure(closure)) => Ok(*closure),
        Some(Value::NativeFunction(_) | Value::BuiltinFunction(_)) => {
            Err(RuntimeError::argument_error(position, "Lua function expected").into())
        }
        value => Err(RuntimeError::bad_argument(position, "function", value).into()),
    }
}

/// The upvalue index at argument `position`.
pub(crate) fn debug_upvalue_index(value: Option<&Value>, position: usize) -> VmResult<usize> {
    value
        .and_then(value_index)
        .ok_or_else(|| RuntimeError::argument_error(position, "invalid upvalue index").into())
}

/// Lua exposes source locals first, then the frame registers the compiler did
/// not associate with a name. The latter are intentionally visible as
/// `(*temporary)` because debuggers use them to inspect expression results.
pub(crate) fn debug_local_slot(
    frame: &impl crate::vm::stack::FrameView,
    index: usize,
) -> Option<(Option<LuaString>, usize)> {
    let frame = frame.frame_ref();
    let pc = frame.last_interpreted_pc.unwrap_or(frame.pc);
    let locals = frame.chunk.debug.locals_at(pc).collect::<Vec<_>>();
    if let Some(local) = locals.get(index) {
        return Some((Some(local.name().clone()), usize::from(local.register())));
    }
    let temporary_index = index.checked_sub(locals.len())?;
    let named = locals
        .iter()
        .map(|local| usize::from(local.register()))
        .collect::<Vec<_>>();
    let temporary_limit = debug_temporary_register_limit(&frame, pc);
    let live = debug_live_temporaries(frame.frame, pc);
    frame
        .registers
        .iter()
        .enumerate()
        .take(temporary_limit)
        .filter(|(register, _)| !named.contains(register))
        .filter(|(register, _)| {
            u8::try_from(*register).is_ok_and(|register| live.contains(register))
        })
        .nth(temporary_index)
        .map(|(register, _)| (None, register))
}

/// The unnamed registers a debugger sees at `pc`, the instruction the frame
/// stopped at: those the chunk's control-flow liveness says are read after
/// `pc` (see `Chunk::live_registers`), and that a collection now roots. The
/// collector, the JIT and `debug.getlocal` then agree on which temporaries
/// still hold a value, and a debugger never sees a slot the collector
/// skipped and cleared.
fn debug_live_temporaries(frame: &Frame, pc: usize) -> RegisterSet {
    frame
        .chunk
        .live_registers(pc.saturating_add(1))
        .intersection(crate::vm::Vm::frame_live_registers(
            frame,
            Window::Live,
            &[],
        ))
}

pub(crate) fn debug_temporary_register_limit(frame: &FrameRef<'_>, pc: usize) -> usize {
    let Some(instruction) = frame.chunk.code.get(pc) else {
        return frame.registers.len();
    };
    let Ok(opcode) = instruction.opcode() else {
        return frame.registers.len();
    };
    match opcode {
        OpCode::Call
        | OpCode::CallVararg
        | OpCode::CallOpen
        | OpCode::TailCall
        | OpCode::TailCallVararg
        | OpCode::TailCallOpen => usize::from(instruction.a()),
        // The compact call form has no contiguous base. All three operands
        // are live call machinery, so temporaries stop before the earliest.
        OpCode::CallOne => usize::from(instruction.a().min(instruction.b()).min(instruction.c())),
        // The call window starts after the three control registers.
        OpCode::ForInCall => usize::from(instruction.a()) + 3,
        _ => frame.registers.len(),
    }
}

pub(crate) fn debug_options(arguments: &[Value]) -> VmResult<&[u8]> {
    match arguments.get(1) {
        None | Some(Value::Nil) => Ok(b"flnSu"),
        Some(Value::String(options)) => Ok(options.as_bytes()),
        Some(value) => Err(RuntimeError::bad_argument(2, "string", Some(value)).into()),
    }
}

pub(crate) fn debug_short_source(source: &LuaString) -> LuaString {
    // Lua 5.1's LUA_IDSIZE includes the trailing NUL byte. Keep the same
    // 59-byte visible limit and the same chunk-id rules as luaO_chunkid.
    pub(crate) const SHORT_SOURCE_LIMIT: usize = 60;
    let bytes = source.as_bytes();
    match bytes.first() {
        Some(b'=') => LuaString::from(
            bytes[1..]
                .iter()
                .copied()
                .take_while(|byte| *byte != 0)
                .take(SHORT_SOURCE_LIMIT - 1)
                .collect::<Vec<_>>(),
        ),
        Some(b'@') => {
            let path_tail = &bytes[1..];
            let path_length = path_tail
                .iter()
                .position(|byte| *byte == 0)
                .unwrap_or(path_tail.len());
            let path = &path_tail[..path_length];
            // sizeof("...") in Lua's C implementation includes its NUL.
            let limit = SHORT_SOURCE_LIMIT - 4;
            if path.len() <= limit {
                return LuaString::from(path.to_vec());
            }
            let mut short = b"...".to_vec();
            short.extend_from_slice(&path[path.len() - limit..]);
            LuaString::from(short)
        }
        _ => {
            let source_length = bytes
                .iter()
                .position(|byte| *byte == 0)
                .unwrap_or(bytes.len());
            let first_line_length = bytes[..source_length]
                .iter()
                .position(|byte| matches!(*byte, b'\n' | b'\r'))
                .unwrap_or(source_length);
            // sizeof("[string \"...\"]") includes its NUL.
            let limit = SHORT_SOURCE_LIMIT - 15;
            let retained = first_line_length.min(limit);
            let mut short = b"[string \"".to_vec();
            short.extend_from_slice(&bytes[..retained]);
            if retained < source_length {
                short.extend_from_slice(b"...");
            }
            short.extend_from_slice(b"\"]");
            LuaString::from(short)
        }
    }
}

pub(crate) const fn debug_namewhat(kind: CallKind) -> &'static str {
    match kind {
        CallKind::Global => "global",
        CallKind::Local => "local",
        CallKind::Upvalue => "upvalue",
        CallKind::Field => "field",
        CallKind::Method => "method",
        CallKind::Constant => "constant",
        CallKind::ForIterator => "for iterator",
    }
}

/// Error handlers inspect the failed call before Lua unwinds it. Builtins
/// execute without an activation, so retain their missing C frame too.
pub(crate) fn capture_error_traceback(
    frames: &CallStack,
    metamethod_handler: Option<Value>,
) -> Traceback {
    let mut trace = capture_traceback(frames);
    // Native levels added to `debug_frames` alone, which `getinfo` counts.
    let mut natives = 0;
    if let Some(frame) = frames.last() {
        let instruction = frame
            .pc
            .checked_sub(1)
            .and_then(|pc| frame.chunk.code.get(pc));
        let called = instruction
            .filter(|instruction| {
                matches!(
                    instruction.opcode(),
                    Ok(OpCode::Call
                        | OpCode::ForInCall
                        | OpCode::CallOne
                        | OpCode::CallVararg
                        | OpCode::CallOpen
                        | OpCode::TailCall
                        | OpCode::TailCallVararg
                        | OpCode::TailCallOpen)
                )
            })
            .and_then(|instruction| {
                frame.registers.get(match instruction.opcode() {
                    Ok(OpCode::CallOne) => usize::from(instruction.b()),
                    // The iterator copy in the call window.
                    Ok(OpCode::ForInCall) => usize::from(instruction.a()) + 3,
                    _ => usize::from(instruction.a()),
                })
            })
            .filter(|function| {
                matches!(
                    function,
                    Value::BuiltinFunction(_)
                        | Value::NativeFunction(_)
                        | Value::CoroutineWrapper(_)
                )
            });
        if let Some(function) = called {
            let mut level =
                native_trace_frame(Some(function.clone()), None, active_call_site(&frame));
            // LuaJIT names a tail-called native after the call (or the
            // metamethod) that reached the tail-calling function.
            if crate::vm::errors::is_tail_call(&frame) {
                let parent = frames.len().checked_sub(2).map(|index| &frames[index]);
                level.extra.luajit_tail =
                    Some(crate::vm::errors::luajit_tail_native_name(&frame, parent));
            }
            trace
                .debug_frames
                .insert(0, c_function_info(Some(function.clone())));
            trace.frames.insert(0, level);
            natives += 1;
        } else if let (Some(function), Some(event)) = (metamethod_handler, metamethod_event(&frame))
        {
            // A native called directly as the metamethod handler.
            let mut level = native_trace_frame(Some(function.clone()), None, None);
            level.extra.metamethod = Some(event);
            trace
                .debug_frames
                .insert(0, c_function_info(Some(function)));
            trace.frames.insert(0, level);
            natives += 1;
        }
    }
    let mut offset = 0;
    for frame in frames.iter().rev() {
        offset += 1 + frame.tail_calls as usize;
        if matches!(
            frame.caller,
            Some(crate::vm::ReturnTarget::ExtendedProtectedTarget(_))
        ) {
            trace.debug_frames.insert(
                offset + natives,
                c_function_info(Some(Value::BuiltinFunction(
                    crate::runtime::BuiltinFunction::ExtendedProtectedCall,
                ))),
            );
            break;
        }
    }
    trace
}
