//! Construction and location of Lua runtime errors and error values.

use crate::bytecode::{CallKind, OpCode};
use crate::compat::CompatibilityProfile;
use crate::runtime::{Runtime, RuntimeError, Value};
use crate::string::LuaString;
use crate::vm::debug_info::{active_call_site, debug_namewhat, metamethod_event};
use crate::vm::stack::{CallStack, FrameRef};
use crate::vm::{ActiveCallSite, Frame, VmError, VmResult};

pub(crate) fn call_type_error(
    frame: &Frame,
    value: &Value,
    compatibility: CompatibilityProfile,
) -> RuntimeError {
    let Some(call_site) = active_call_site(frame) else {
        return call_type_error_anonymous(value);
    };
    named_type_error("call", value, &call_site, compatibility)
}

pub(crate) fn call_type_error_anonymous(value: &Value) -> RuntimeError {
    RuntimeError::message(format!("attempt to call a {} value", value.type_name()))
}

/// Whether the call `frame` has just executed was a `return f(...)`.
pub(crate) fn is_tail_call(frame: &Frame) -> bool {
    frame
        .pc
        .checked_sub(1)
        .and_then(|pc| frame.chunk.code.get(pc))
        .and_then(|instruction| instruction.opcode().ok())
        .is_some_and(|opcode| {
            matches!(
                opcode,
                OpCode::TailCall | OpCode::TailCallVararg | OpCode::TailCallOpen
            )
        })
}

fn argument_message_named(name: &str, position: usize, detail: &str) -> String {
    if position == 0 {
        format!("calling '{name}' on bad self ({detail})")
    } else {
        format!("bad argument #{position} to '{name}' ({detail})")
    }
}

/// The `luaL_argerror` text for argument `position` of the function called
/// by `site`: a method call does not count its receiver.
fn argument_message(
    site: Option<&ActiveCallSite>,
    position: usize,
    detail: &str,
    compatibility: CompatibilityProfile,
) -> String {
    argument_message_extra(site, position, detail, compatibility, 0)
}

/// [`argument_message`] for a function that `extra` `__call` hops reached.
/// Lua 5.5 does not count those hops' arguments, and says so when one of
/// them is at fault.
pub(crate) fn argument_message_extra(
    site: Option<&ActiveCallSite>,
    position: usize,
    detail: &str,
    compatibility: CompatibilityProfile,
    extra: usize,
) -> String {
    let extra = if compatibility == CompatibilityProfile::Lua55 {
        extra
    } else {
        0
    };
    if position >= 1 && position <= extra {
        let name = site.map_or_else(|| "?".to_owned(), |site| site.display_name(compatibility));
        return format!("bad extra argument #{position} to '{name}' ({detail})");
    }
    let position = position - extra;
    let (name, position) = match site {
        Some(site) => (
            site.display_name(compatibility),
            if site.kind == CallKind::Method {
                position.saturating_sub(1)
            } else {
                position
            },
        ),
        None => ("?".to_owned(), position),
    };
    if position == 0 {
        format!("calling '{name}' on bad self ({detail})")
    } else {
        format!("bad argument #{position} to '{name}' ({detail})")
    }
}

/// The name stock Lua gives a native function that `frame`'s current
/// instruction called as a metamethod: `index` in Lua 5.4 and later,
/// `__index` in `LuaJIT`. Lua 5.1 names nothing.
fn metamethod_name(frame: &Frame, compatibility: CompatibilityProfile) -> Option<String> {
    if compatibility.is_lua51_family() && compatibility != CompatibilityProfile::LuaJit21 {
        return None;
    }
    let event = metamethod_event(frame)?;
    Some(if compatibility == CompatibilityProfile::LuaJit21 {
        format!("__{event}")
    } else {
        event.to_owned()
    })
}

/// The frame whose call `LuaJIT` names a native after, when `frame` tail-called
/// it. A metamethod handler's tail call is named after the metamethod and
/// located at the instruction that triggered it. A `__tostring` handler is
/// named after `tostring`, unless that call was itself a tail call (then
/// stock names nothing). `None` names nothing.
pub(crate) fn luajit_tail_caller<'a>(
    frame: &Frame,
    parent: Option<&'a Frame>,
) -> Option<&'a Frame> {
    let from_metamethod = matches!(
        frame.caller,
        Some(
            crate::vm::ReturnTarget::Index(_)
                | crate::vm::ReturnTarget::ProtectedIndex(_)
                | crate::vm::ReturnTarget::SetMetamethod
                | crate::vm::ReturnTarget::IndexResult(_)
                | crate::vm::ReturnTarget::Scalar(_)
                | crate::vm::ReturnTarget::Operator(_)
                | crate::vm::ReturnTarget::ConcatRange(_)
        )
    );
    let from_tostring = matches!(frame.caller, Some(crate::vm::ReturnTarget::ToString(_)));
    parent.filter(|parent| {
        from_metamethod && metamethod_event(parent).is_some()
            || matches!(frame.caller, Some(crate::vm::ReturnTarget::Direct(_)))
                && !called_by_native(frame, CompatibilityProfile::LuaJit21)
            || from_tostring && !is_tail_call(parent)
    })
}

/// How `LuaJIT`'s traceback names a native that `frame` tail-called.
pub(crate) fn luajit_tail_native_name(
    frame: &Frame,
    parent: Option<&Frame>,
) -> crate::vm::LuajitTailName {
    let Some(caller) = luajit_tail_caller(frame, parent) else {
        return crate::vm::LuajitTailName::Unnamed;
    };
    let from_metamethod = matches!(
        frame.caller,
        Some(
            crate::vm::ReturnTarget::Index(_)
                | crate::vm::ReturnTarget::ProtectedIndex(_)
                | crate::vm::ReturnTarget::SetMetamethod
                | crate::vm::ReturnTarget::IndexResult(_)
                | crate::vm::ReturnTarget::Scalar(_)
                | crate::vm::ReturnTarget::Operator(_)
                | crate::vm::ReturnTarget::ConcatRange(_)
        )
    );
    match metamethod_event(caller).filter(|_| from_metamethod) {
        Some(event) => crate::vm::LuajitTailName::Event(event),
        None => match active_call_site(caller) {
            Some(site) => crate::vm::LuajitTailName::Site(site),
            None => crate::vm::LuajitTailName::Unnamed,
        },
    }
}

/// The argument error of a native that `LuaJIT`'s function `frame` tail-called.
/// The tail call replaced `frame`'s record, so `LuaJIT` names the function from
/// the call in `parent`, the Lua frame that called `frame`, and locates the
/// error at that call. With no Lua caller (`pcall`, a coroutine body, the
/// top level) it names nothing and adds no position.
fn luajit_tail_call_error(
    frame: &Frame,
    parent: Option<&Frame>,
    compatibility: CompatibilityProfile,
    position: usize,
    detail: &str,
) -> RuntimeError {
    let from_metamethod = matches!(
        frame.caller,
        Some(
            crate::vm::ReturnTarget::Index(_)
                | crate::vm::ReturnTarget::ProtectedIndex(_)
                | crate::vm::ReturnTarget::SetMetamethod
                | crate::vm::ReturnTarget::IndexResult(_)
                | crate::vm::ReturnTarget::Scalar(_)
                | crate::vm::ReturnTarget::Operator(_)
                | crate::vm::ReturnTarget::ConcatRange(_)
        )
    );
    let caller = luajit_tail_caller(frame, parent);
    let Some(caller) = caller else {
        return RuntimeError::Raised(Value::from(argument_message(
            None,
            position,
            detail,
            compatibility,
        )));
    };
    let message = match metamethod_name(caller, compatibility).filter(|_| from_metamethod) {
        Some(name) => argument_message_named(&name, position, detail),
        None => argument_message(
            active_call_site(caller).as_ref(),
            position,
            detail,
            compatibility,
        ),
    };
    let line = caller
        .last_interpreted_pc
        .get()
        .or_else(|| caller.pc.checked_sub(1))
        .and_then(|pc| caller.chunk.debug.line_at(pc));
    RuntimeError::Raised(Value::from(match line {
        Some(line) => format!(
            "{}:{line}: {message}",
            display_chunk_source(caller.chunk.debug.source_name())
        ),
        None => message,
    }))
}

/// The argument error of a builtin that runs as an `__index` or `__newindex`
/// handler for the instruction `parent` is executing. Stock names the
/// function after the event ('index', '__index' in `LuaJIT`, '?' in Lua 5.1) and
/// locates the error at that instruction. Anything but an argument error
/// passes through.
pub(crate) fn metamethod_argument_error(
    parent: &Frame,
    is_set: bool,
    error: VmError,
    compatibility: CompatibilityProfile,
) -> VmError {
    let VmError::Runtime(error) = error else {
        return error;
    };
    let (position, detail) = if let Some(found) = error.argument_detail() {
        found
    } else if let RuntimeError::TypeError {
        expected, found, ..
    } = &error
    {
        (1, format!("{expected} expected, got {found}"))
    } else {
        return error.into();
    };
    let name = match (compatibility, is_set) {
        (CompatibilityProfile::LuaJit21, false) => "__index",
        (CompatibilityProfile::LuaJit21, true) => "__newindex",
        _ if compatibility.is_lua51_family() => "?",
        (_, false) => "index",
        (_, true) => "newindex",
    };
    let message = if position == 0 {
        format!("calling '{name}' on bad self ({detail})")
    } else {
        format!("bad argument #{position} to '{name}' ({detail})")
    };
    let line = parent
        .last_interpreted_pc
        .get()
        .or_else(|| parent.pc.checked_sub(1))
        .and_then(|pc| parent.chunk.debug.line_at(pc));
    RuntimeError::Raised(Value::from(match line {
        Some(line) => format!(
            "{}:{line}: {message}",
            display_chunk_source(parent.chunk.debug.source_name())
        ),
        None => message,
    }))
    .into()
}

/// Names the function in a library argument error from its call, as
/// `luaL_argerror` does. A method call does not count its receiver. `parent`
/// is the frame below `frame`. `LuaJIT` names the function from `parent`'s call
/// in a tail call (see [`luajit_tail_call_error`]).
pub(crate) fn call_argument_error(
    frame: &Frame,
    parent: Option<&Frame>,
    error: RuntimeError,
    compatibility: CompatibilityProfile,
) -> RuntimeError {
    call_argument_error_extra(frame, parent, error, compatibility, 0)
}

/// [`call_argument_error`] for a call that went through `extra` `__call`
/// hops.
pub(crate) fn call_argument_error_extra(
    frame: &Frame,
    parent: Option<&Frame>,
    error: RuntimeError,
    compatibility: CompatibilityProfile,
    extra: usize,
) -> RuntimeError {
    let luajit_tail = compatibility == CompatibilityProfile::LuaJit21 && is_tail_call(frame);
    if let Some((position, detail)) = error.argument_detail() {
        if luajit_tail {
            return luajit_tail_call_error(frame, parent, compatibility, position, &detail);
        }
        let Some(call_site) = active_call_site(frame) else {
            return error;
        };
        return RuntimeError::message(argument_message_extra(
            Some(&call_site),
            position,
            &detail,
            compatibility,
            extra,
        ));
    }
    let RuntimeError::TypeError {
        operation,
        expected,
        found,
    } = error
    else {
        return error;
    };
    if luajit_tail {
        return luajit_tail_call_error(
            frame,
            parent,
            compatibility,
            1,
            &format!("{expected} expected, got {found}"),
        );
    }
    let Some(call_site) = active_call_site(frame) else {
        return RuntimeError::TypeError {
            operation,
            expected,
            found,
        };
    };
    RuntimeError::message(format!(
        "bad argument #1 to '{}' ({expected} expected, got {found})",
        call_site.display_name(compatibility)
    ))
}

/// The error for reading or writing a field of a value that cannot be
/// indexed, naming the value when debug information knows it.
/// [`call_argument_error`] for a builtin called from `frame`, which fails
/// with a [`VmError`].
pub(crate) fn builtin_argument_error(
    frame: &Frame,
    parent: Option<&Frame>,
    error: VmError,
    compatibility: CompatibilityProfile,
) -> VmError {
    match error {
        VmError::Runtime(error) if error.argument_detail().is_some() => {
            call_argument_error(frame, parent, error, compatibility).into()
        }
        error => error,
    }
}

/// The name `pushglobalfuncname` gives `function`: its field in a module
/// of `package.loaded`, such as `string.rep`, or a bare name for `_G`.
pub(crate) fn loaded_function_name(runtime: &Runtime, function: &Value) -> Option<String> {
    let package = runtime.package_require_table()?;
    let Ok(Value::Table(loaded)) = runtime.table_get(package, &Value::from("loaded")) else {
        return None;
    };
    let mut module = Value::Nil;
    loop {
        let (key, value) = runtime.table_next(loaded, &module).ok()?;
        if matches!(key, Value::Nil) {
            return None;
        }
        if let (Value::String(module_name), Value::Table(table)) = (&key, &value) {
            let mut field = Value::Nil;
            loop {
                let (name, member) = runtime.table_next(*table, &field).ok()?;
                match name {
                    Value::Nil => break,
                    Value::String(ref name) if Runtime::raw_equal(&member, function) => {
                        let module_name = module_name.to_lossy();
                        return Some(if module_name == "_G" {
                            name.to_lossy().into_owned()
                        } else {
                            format!("{module_name}.{}", name.to_lossy())
                        });
                    }
                    _ => {}
                }
                field = name;
            }
        }
        module = key;
    }
}

pub(crate) fn index_type_error(
    value: &Value,
    diagnostic: Option<&ActiveCallSite>,
    compatibility: CompatibilityProfile,
) -> RuntimeError {
    match diagnostic {
        Some(diagnostic) => named_type_error("index", value, diagnostic, compatibility),
        None => anonymous_type_error("index", value),
    }
}

/// Spells a type error that names its operand in the profile's layout:
/// Lua 5.1 and 5.2 write "attempt to index local 't' (a nil value)", and
/// Lua 5.3 and later "attempt to index a nil value (local 't')".
pub(crate) fn named_type_error(
    operation: &str,
    value: &Value,
    diagnostic: &ActiveCallSite,
    compatibility: CompatibilityProfile,
) -> RuntimeError {
    let kind = debug_namewhat(diagnostic.kind);
    let name = diagnostic.name.to_lossy();
    let found = value.type_name();
    // Lua 5.1 and LuaJIT do not name the generic-`for` iterator in a call error.
    if diagnostic.kind == CallKind::ForIterator && compatibility.is_lua51_family() {
        return anonymous_type_error(operation, value);
    }
    RuntimeError::message(if compatibility.is_lua53_or_later() {
        format!("attempt to {operation} a {found} value ({kind} '{name}')")
    } else {
        format!("attempt to {operation} {kind} '{name}' (a {found} value)")
    })
}

fn anonymous_type_error(operation: &str, value: &Value) -> RuntimeError {
    RuntimeError::message(format!(
        "attempt to {operation} a {} value",
        value.type_name()
    ))
}

/// Names the offending operand when debug information knows it, and
/// otherwise spells a primitive type error as Lua does, without the
/// expected-type detail of [`RuntimeError::TypeError`].
pub(crate) fn contextual_type_error(
    fallback: RuntimeError,
    value: &Value,
    diagnostic: Option<&ActiveCallSite>,
    operation: &str,
    compatibility: CompatibilityProfile,
) -> RuntimeError {
    match diagnostic {
        Some(diagnostic) => named_type_error(operation, value, diagnostic, compatibility),
        None if matches!(fallback, RuntimeError::TypeError { .. }) => {
            anonymous_type_error(operation, value)
        }
        None => fallback,
    }
}

/// Lua's error for a numeric `for` initial value, limit or step that is not
/// a number.
pub(crate) fn numeric_for_error(
    compatibility: CompatibilityProfile,
    what: &str,
    value: &Value,
) -> RuntimeError {
    let lua54 = matches!(
        compatibility,
        CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
    );
    RuntimeError::message(if lua54 {
        format!(
            "bad 'for' {what} (number expected, got {})",
            value.type_name()
        )
    } else {
        format!("'for' {what} must be a number")
    })
}

/// Lua's error for ordering two values that have no primitive order.
pub(crate) fn compare_type_error(operation: &str, left: &Value, right: &Value) -> RuntimeError {
    let (left, right) = (left.type_name(), right.type_name());
    RuntimeError::message(if left == right {
        format!("attempt to {operation} two {left} values")
    } else {
        format!("attempt to {operation} {left} with {right}")
    })
}

pub(crate) fn contextual_integer_conversion_error(
    fallback: RuntimeError,
    value: &Value,
    diagnostic: Option<&ActiveCallSite>,
    operation: &str,
    compatibility: CompatibilityProfile,
) -> RuntimeError {
    if matches!(
        &fallback,
        RuntimeError::Message(message) if message == "number has no integer representation"
    ) && let Some(diagnostic) = diagnostic
    {
        return RuntimeError::message(format!(
            "number ({} '{}') has no integer representation",
            debug_namewhat(diagnostic.kind),
            diagnostic.name.to_lossy()
        ));
    }
    contextual_type_error(fallback, value, diagnostic, operation, compatibility)
}

/// The Lua value a protected call, coroutine, or `__close` handler receives
/// for a caught `error`.
///
/// # Errors
///
/// An `os.exit` request is not a Lua error: it comes back unchanged so the
/// catching site keeps unwinding instead of handing Lua a value.
pub(crate) fn runtime_error_object(error: &RuntimeError) -> VmResult<Value> {
    Ok(match error {
        RuntimeError::Hook(error) => return runtime_error_object(error),
        RuntimeError::RaisedNil => Value::Nil,
        RuntimeError::Raised(value) | RuntimeError::RaisedAtLevel { value, .. } => value.clone(),
        RuntimeError::Exit(_) => return Err(error.clone().into()),
        _ => Value::from(error.to_string()),
    })
}

impl crate::vm::Vm {
    /// Lua 5.5 substitutes a message only after the message handler has run.
    pub(crate) fn normalize_nil_error_value(&self, value: Value) -> Value {
        if self.compatibility == CompatibilityProfile::Lua55 && matches!(value, Value::Nil) {
            Value::from("<no error object>")
        } else {
            value
        }
    }

    pub(crate) fn normalize_nil_error(&self, error: VmError) -> VmError {
        if self.compatibility == CompatibilityProfile::Lua55
            && !matches!(error, VmError::CoroutineClosed)
            && matches!(vm_error_object(&error), Ok(Value::Nil))
        {
            let replacement = RuntimeError::Raised(Value::from("<no error object>")).into();
            if error.is_hook_error() {
                VmError::HookError(Box::new(replacement))
            } else {
                replacement
            }
        } else {
            error
        }
    }
}

pub(crate) fn raised_value(value: Value) -> VmError {
    match value {
        Value::Nil => RuntimeError::RaisedNil.into(),
        value => RuntimeError::Raised(value).into(),
    }
}

/// Like [`runtime_error_object`], for any VM error.
///
/// # Errors
///
/// Returns an `os.exit` request unchanged.
pub(crate) fn vm_error_object(error: &VmError) -> VmResult<Value> {
    match error {
        VmError::HookError(error) => vm_error_object(error),
        VmError::CoroutineClosed => Ok(Value::Nil),
        VmError::Runtime(error) => runtime_error_object(error),
        _ if error.exit_request().is_some() => Err(error.clone()),
        _ => Ok(Value::from(error.to_string())),
    }
}

pub(crate) fn locate_runtime_error(
    frames: &CallStack,
    compatibility: CompatibilityProfile,
    error: VmError,
) -> VmError {
    let error = match error {
        VmError::HookError(error) => {
            return VmError::HookError(Box::new(locate_runtime_error(
                frames,
                compatibility,
                *error,
            )));
        }
        error => error,
    };
    let VmError::Runtime(runtime_error) = error else {
        return error;
    };
    match runtime_error {
        RuntimeError::Raised(_)
        | RuntimeError::RaisedNil
        | RuntimeError::Exit(_)
        | RuntimeError::Memory => {
            return VmError::Runtime(runtime_error);
        }
        RuntimeError::RaisedAtLevel { value, level } => {
            if level == 0 || !matches!(value, Value::String(_)) {
                return VmError::Runtime(RuntimeError::Raised(value));
            }
            let Some(frame) = frame_at_error_level(frames, compatibility, level) else {
                return VmError::Runtime(RuntimeError::Raised(value));
            };
            let Some(line) = frame
                .last_interpreted_pc
                .get()
                .and_then(|pc| frame.chunk.debug.line_at(pc))
            else {
                return VmError::Runtime(RuntimeError::Raised(value));
            };
            return VmError::Runtime(RuntimeError::Raised(Value::from(format!(
                "{}:{line}: {}",
                display_chunk_source(frame.chunk.debug.source_name()),
                value_to_error_text(&value)
            ))));
        }
        _ => {}
    }
    let Some(frame) = frames.last() else {
        return VmError::Runtime(runtime_error);
    };
    // A native called as a metamethod: stock names it after the event.
    let runtime_error = match runtime_error.argument_detail() {
        Some((position, detail)) => match metamethod_name(&frame, compatibility) {
            Some(name) => RuntimeError::message(argument_message_named(&name, position, &detail)),
            None => runtime_error,
        },
        None => runtime_error,
    };
    let Some(line) = frame
        .last_interpreted_pc
        .get()
        .and_then(|pc| frame.chunk.debug.line_at(pc))
    else {
        return VmError::Runtime(runtime_error);
    };
    // LuaJIT's tail call replaces the Lua frame, so a library error raised by
    // the callee has no Lua caller to give it a position. Errors of the
    // call instruction itself ("attempt to call ...") keep theirs.
    if compatibility == CompatibilityProfile::LuaJit21
        && is_tail_call(&frame)
        && frame.last_interpreted_pc.get() == frame.pc.checked_sub(1)
        && matches!(&runtime_error, RuntimeError::Message(text) if !text.starts_with("attempt to "))
    {
        return VmError::Runtime(runtime_error);
    }
    VmError::LocatedRuntime {
        source: frame.chunk.debug.source_name().clone(),
        line,
        error: runtime_error,
    }
}

/// Whether the function running in a frame was called by a native function
/// (`pcall`, `xpcall`, `table.sort`, ...), which Lua counts as a level of its
/// own that has no source position.
fn called_by_native(frame: &crate::vm::Frame, compatibility: CompatibilityProfile) -> bool {
    use crate::vm::{FinalizerCompletion, ReturnTarget};
    // A finalizer run by an explicit `collectgarbage` has that native function
    // as its caller; an automatic one runs under the current Lua frame.
    if let Some(ReturnTarget::UserdataFinalizer(completion)) = &frame.caller {
        return matches!(
            **completion,
            FinalizerCompletion::Call { .. } | FinalizerCompletion::Protected { .. }
        );
    }
    matches!(
        frame.caller,
        Some(
            ReturnTarget::Protected(_)
                | ReturnTarget::ExtendedProtectedTarget(_)
                | ReturnTarget::TableSort(_)
                | ReturnTarget::ProtectedSort(_)
                | ReturnTarget::TableForeach(_)
                | ReturnTarget::StringGsub(_)
                | ReturnTarget::Require(_)
                | ReturnTarget::RequireSearch(_)
                | ReturnTarget::Module(_)
                | ReturnTarget::Dofile(_)
                | ReturnTarget::LoadReader(_)
                | ReturnTarget::StringFormat(_)
        )
    ) || (compatibility != CompatibilityProfile::LuaJit21
        && matches!(frame.caller, Some(ReturnTarget::ToString(_))))
}

/// The Lua function at `error` level `level` (1 is the function that called
/// `error`), or `None` when that level is a native function or does not
/// exist. Native callers occupy a level between the Lua frames, as in Lua.
fn frame_at_error_level(
    frames: &CallStack,
    compatibility: CompatibilityProfile,
    level: usize,
) -> Option<FrameRef<'_>> {
    let mut depth = 0;
    let mut current = 1;
    loop {
        let frame = frames.nth_from_top(depth)?;
        if current == level {
            return Some(frame);
        }
        // Lua 5.1 counts each tail call a frame replaced as a level without
        // a position; later profiles reuse the call record.
        if compatibility == CompatibilityProfile::Lua51 {
            let tails = frame.tail_calls as usize;
            if level <= current + tails {
                return None;
            }
            current += tails;
        }
        if called_by_native(&frame, compatibility) {
            current += 1;
            if current == level {
                return None;
            }
        }
        current += 1;
        depth += 1;
    }
}

/// Locates an `error` raised by a native function that `pcall` or `xpcall`
/// called directly. The protected-call boundary is level 1 and carries no
/// position, so the Lua frames start at level 2.
pub(crate) fn locate_error_below_protected_call(
    frames: &CallStack,
    compatibility: CompatibilityProfile,
    error: RuntimeError,
) -> RuntimeError {
    match error {
        RuntimeError::RaisedAtLevel { value, level } if level <= 1 => RuntimeError::Raised(value),
        RuntimeError::RaisedAtLevel { value, level } => {
            match locate_runtime_error(
                frames,
                compatibility,
                RuntimeError::RaisedAtLevel {
                    value,
                    level: level - 1,
                }
                .into(),
            ) {
                VmError::Runtime(error) => error,
                other => RuntimeError::message(other.to_string()),
            }
        }
        error => error,
    }
}

pub(crate) fn value_to_error_text(value: &Value) -> String {
    match value {
        Value::String(value) => value.to_lossy().into_owned(),
        _ => value.type_name().to_owned(),
    }
}

pub(crate) fn display_chunk_source(source: &LuaString) -> String {
    let bytes = source.as_bytes();
    match bytes.first() {
        Some(b'@' | b'=') => String::from_utf8_lossy(&bytes[1..]).into_owned(),
        _ => format!("[string \"{}\"]", String::from_utf8_lossy(bytes)),
    }
}

pub(crate) fn file_io_results(error: &RuntimeError) -> Vec<Value> {
    let code = match error {
        RuntimeError::FileIo { code, .. } | RuntimeError::HostOutputIo { code, .. } => *code,
        _ => 1,
    };
    vec![
        Value::Nil,
        Value::from(error.to_string()),
        Value::Number(f64::from(code)),
    ]
}

pub(crate) fn global_already_defined(name: &LuaString) -> RuntimeError {
    RuntimeError::message(format!(
        "global '{}' already defined",
        String::from_utf8_lossy(name.as_bytes())
    ))
}
