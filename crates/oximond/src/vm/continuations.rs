//! Continuation state for resumable library calls, loads, and protected unwinding.

use crate::bytecode::Instruction;
use crate::bytecode::load::LoadPoll;
use crate::compat::CompatibilityProfile;
use crate::pattern::{Match as PatternMatch, Pattern};
use crate::runtime::{GcRootBuilder, NumberFormat, Runtime, RuntimeError, Value, format_integer};
use crate::string::LuaString;
use crate::vm::arith::{profile_usize_value, usize_to_number};
use crate::vm::libraries::Window;
use crate::vm::{
    ActiveCallSite, CallReturn, DebugHookEvent, DofileCompletion, ForeachCompletion, IndexAction,
    IndexCompletion, IndexContinuation, LoadCompletion, LoadPipeline, ModuleCompletion,
    OperatorMode, OperatorReturn, PipelinePoll, RequireCompletion, ReturnTarget,
    ScalarContinuation, ScalarKind, SortAfter, SortCompletion, SortPhase, UnwindClose,
    UnwindCompletion, Vm, VmResult,
};

impl IndexContinuation {
    pub(crate) fn get(
        receiver: Value,
        key: Value,
        destination: u8,
        completion: IndexCompletion,
    ) -> Self {
        Self {
            receiver,
            key,
            action: IndexAction::Get { destination },
            completion,
            hops: 0,
            diagnostic: None,
        }
    }

    pub(crate) fn set(receiver: Value, key: Value, value: Value) -> Self {
        Self {
            receiver,
            key,
            action: IndexAction::Set { value },
            completion: IndexCompletion::Frame,
            hops: 0,
            diagnostic: None,
        }
    }

    pub(crate) fn with_diagnostic(mut self, diagnostic: Option<ActiveCallSite>) -> Self {
        self.diagnostic = diagnostic;
        self
    }

    pub(crate) const fn is_set(&self) -> bool {
        matches!(self.action, IndexAction::Set { .. })
    }

    pub(crate) fn value(&self) -> &Value {
        match &self.action {
            IndexAction::Set { value } => value,
            IndexAction::Get { .. } => unreachable!("get has no assigned value"),
        }
    }

    pub(crate) fn callback_arguments(&self) -> Vec<Value> {
        let mut arguments = vec![self.receiver.clone(), self.key.clone()];
        if let IndexAction::Set { value } = &self.action {
            arguments.push(value.clone());
        }
        arguments
    }
}

impl OperatorReturn {
    /// The plain form of `continuation`, when its operands are not needed
    /// after the handler is entered: not a string left operand, which marks
    /// a call from Lua 5.4's string arithmetic (see
    /// `Vm::is_string_arithmetic_callback`).
    pub(crate) fn of(continuation: &ScalarContinuation) -> Option<Self> {
        if matches!(continuation.left, Value::String(_)) {
            return None;
        }
        let mode = match (continuation.test, continuation.kind) {
            (Some(expected), _) => OperatorMode::Branch {
                expected,
                invert: continuation.invert,
            },
            (None, ScalarKind::Comparison) => OperatorMode::Compare {
                invert: continuation.invert,
            },
            (None, ScalarKind::Value) => OperatorMode::Value,
        };
        Some(Self {
            destination: continuation.destination,
            mode,
        })
    }

    /// The continuation `finish_scalar` completes, without operands.
    pub(crate) fn continuation(self) -> ScalarContinuation {
        let mut continuation = ScalarContinuation::direct(self.destination);
        match self.mode {
            OperatorMode::Value => {}
            OperatorMode::Compare { invert } => {
                continuation.kind = ScalarKind::Comparison;
                continuation.invert = invert;
            }
            OperatorMode::Branch { expected, invert } => {
                continuation.kind = ScalarKind::Comparison;
                continuation.invert = invert;
                continuation.test = Some(expected);
            }
        }
        continuation
    }
}

impl ScalarContinuation {
    pub(crate) const fn direct(destination: u8) -> Self {
        Self {
            destination,
            kind: ScalarKind::Value,
            left: Value::Nil,
            right: None,
            invert: false,
            test: None,
        }
    }

    pub(crate) fn new(
        destination: u8,
        kind: ScalarKind,
        left: Value,
        right: Option<Value>,
    ) -> Self {
        Self {
            destination,
            kind,
            left,
            right,
            invert: false,
            test: None,
        }
    }

    pub(crate) fn comparison(destination: u8, invert: bool, left: Value, right: Value) -> Self {
        Self {
            destination,
            kind: ScalarKind::Comparison,
            left,
            right: Some(right),
            invert,
            test: None,
        }
    }

    /// A comparison that branches instead of storing a boolean: the branch
    /// of a test instruction whose `expected` flag is given.
    pub(crate) fn branching(expected: bool, left: Value, right: Value) -> Self {
        Self {
            test: Some(expected),
            ..Self::comparison(0, false, left, right)
        }
    }

    pub(crate) fn arguments(&self) -> Vec<Value> {
        let mut arguments = vec![self.left.clone()];
        if let Some(right) = &self.right {
            arguments.push(right.clone());
        }
        arguments
    }

    pub(crate) fn reverse_comparison(&mut self) {
        let right = self.right.take().expect("comparison has right operand");
        self.right = Some(self.left.clone());
        self.left = right;
    }
}

pub(crate) fn root_continuation(roots: &mut GcRootBuilder, target: &ReturnTarget) {
    match target {
        ReturnTarget::TableSort(continuation) | ReturnTarget::ProtectedSort(continuation) => {
            roots.value(&Value::Table(continuation.table));
            if let Some(proxy) = &continuation.proxy {
                roots.value(proxy);
            }
            if let Some(comparator) = &continuation.comparator {
                roots.value(comparator);
            }
        }
        ReturnTarget::TableForeach(continuation) => {
            roots.value(&Value::Table(continuation.table));
            roots.value(&continuation.callback);
            roots.value(&continuation.previous);
        }
        ReturnTarget::LoadReader(continuation) => roots.value(&continuation.reader),
        ReturnTarget::Require(continuation) => {
            roots.value(&Value::String(continuation.name.clone()));
            roots.value(&Value::Table(continuation.loaded));
            roots.value(&Value::Table(continuation.sentinel));
            roots.value(&continuation.loader_data);
        }
        ReturnTarget::RequireSearch(search) => {
            roots.value(&Value::String(search.name.clone()));
            roots.value(&Value::Table(search.loaded));
            roots.value(&Value::Table(search.loaders));
        }
        ReturnTarget::Module(continuation) => {
            roots.value(&Value::Table(continuation.module));
            for option in &continuation.options[continuation.next_option..] {
                roots.value(option);
            }
        }
        ReturnTarget::StringGsub(continuation) => {
            roots.value(&Value::String(continuation.source.clone()));
            roots.value(&continuation.replacement);
        }
        ReturnTarget::Index(continuation) | ReturnTarget::ProtectedIndex(continuation) => {
            roots.value(&continuation.receiver);
            roots.value(&continuation.key);
            if let IndexAction::Set { value } = &continuation.action {
                roots.value(value);
            }
            if let IndexCompletion::StringGsub { continuation, .. } = &continuation.completion {
                roots.value(&Value::String(continuation.source.clone()));
                roots.value(&continuation.replacement);
            }
        }
        ReturnTarget::ToString(continuation) => roots.value(&continuation.value),
        ReturnTarget::StringFormat(continuation) => {
            for value in &continuation.arguments {
                roots.value(value);
            }
        }
        ReturnTarget::Scalar(continuation) => {
            roots.value(&continuation.left);
            if let Some(right) = &continuation.right {
                roots.value(right);
            }
        }
        ReturnTarget::ConcatRange(continuation) => {
            roots.value(&continuation.accumulator);
            for value in &continuation.remaining {
                roots.value(value);
            }
        }
        ReturnTarget::UnwindClose(continuation) => root_unwind_close(roots, continuation),
        ReturnTarget::ExtendedProtectedTarget(continuation)
        | ReturnTarget::ExtendedProtectedHandler(continuation) => {
            roots.value(&continuation.handler);
            if let Some(pending) = &continuation.pending_unwind
                && let Some(pending) = pending.lock().expect("pending unwind lock").as_ref()
            {
                root_unwind_close(roots, pending);
            }
        }
        ReturnTarget::CloseLocal
        | ReturnTarget::Direct(_)
        | ReturnTarget::Protected(_)
        | ReturnTarget::Dofile(_)
        | ReturnTarget::UserdataFinalizer(_)
        | ReturnTarget::SetMetamethod
        | ReturnTarget::IndexResult(_)
        | ReturnTarget::Operator(_)
        | ReturnTarget::DebugHook(_) => {}
    }
}

pub(crate) fn root_unwind_close(roots: &mut GcRootBuilder, continuation: &UnwindClose) {
    if let Ok(error) = crate::vm::errors::vm_error_object(&continuation.error) {
        roots.value(&error);
    }
    Vm::root_call_stack(roots, &continuation.frames, Window::Full);
    match &continuation.completion {
        UnwindCompletion::Unprotected => {}
        UnwindCompletion::ExtendedHandled { continuation, .. } => {
            root_continuation(
                roots,
                &ReturnTarget::ExtendedProtectedHandler(continuation.clone()),
            );
        }
        UnwindCompletion::Protected { target, traceback } => {
            root_continuation(roots, target);
            roots.values(
                traceback
                    .debug_frames
                    .iter()
                    .filter_map(|info| info.function.as_ref()),
            );
        }
        UnwindCompletion::Resume(next) => root_unwind_close(roots, next),
    }
}

/// A `string.gsub` string at argument `position`.
pub(crate) fn gsub_string_argument(
    numbers: NumberFormat,
    argument: Option<&Value>,
    position: usize,
) -> VmResult<LuaString> {
    match argument {
        Some(Value::String(value)) => Ok(value.clone()),
        Some(Value::Integer(value)) => Ok(LuaString::from(format_integer(*value).as_bytes())),
        Some(Value::Number(value)) => Ok(LuaString::from(numbers.format(*value).as_bytes())),
        argument => Err(RuntimeError::bad_argument(position, "string", argument).into()),
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub(crate) fn gsub_limit(argument: Option<&Value>) -> VmResult<usize> {
    let Some(argument) = argument else {
        return Ok(usize::MAX);
    };
    if matches!(argument, Value::Nil) {
        return Ok(usize::MAX);
    }
    let number = Runtime::coerce_number(argument, "")
        .map_err(|_| RuntimeError::bad_argument(4, "number", Some(argument)))?;
    if !number.is_finite() {
        return Err(RuntimeError::message("string.gsub replacement limit is too large").into());
    }
    if number <= 0.0 {
        return Ok(0);
    }
    if number >= usize::MAX as f64 {
        return Ok(usize::MAX);
    }
    Ok(number.trunc() as usize)
}

pub(crate) fn gsub_arguments(
    compatibility: CompatibilityProfile,
    source: &[u8],
    found: &PatternMatch,
) -> VmResult<Vec<Value>> {
    if found.captures().is_empty() {
        return Ok(vec![Value::bytes(&source[found.start()..found.end()])]);
    }
    found
        .captures()
        .iter()
        .map(|capture| {
            if capture.is_position() {
                profile_usize_value(compatibility, capture.start() + 1)
            } else {
                Ok(Value::bytes(&source[capture.start()..capture.end()]))
            }
        })
        .collect()
}

pub(crate) fn gsub_key(
    compatibility: CompatibilityProfile,
    source: &[u8],
    found: &PatternMatch,
) -> VmResult<Value> {
    if let Some(capture) = found.captures().first() {
        if capture.is_position() {
            profile_usize_value(compatibility, capture.start() + 1)
        } else {
            Ok(Value::bytes(&source[capture.start()..capture.end()]))
        }
    } else {
        Ok(Value::bytes(&source[found.start()..found.end()]))
    }
}

pub(crate) fn gsub_template(
    compatibility: CompatibilityProfile,
    template: &[u8],
    source: &[u8],
    found: &PatternMatch,
) -> VmResult<Vec<u8>> {
    let mut output = Vec::new();
    let mut index = 0;
    while index < template.len() {
        if template[index] != b'%' {
            append_gsub(&mut output, &template[index..=index])?;
            index += 1;
            continue;
        }
        let Some(&next) = template.get(index + 1) else {
            // Lua 5.1 and LuaJIT copy the terminating NUL after a final `%`.
            if compatibility.is_lua51_family() {
                append_gsub(&mut output, b"\0")?;
                break;
            }
            return Err(RuntimeError::message("invalid use of '%' in replacement string").into());
        };
        match next {
            b'%' => append_gsub(&mut output, b"%")?,
            b'0' => append_gsub(&mut output, &source[found.start()..found.end()])?,
            b'1'..=b'9' => {
                // Lua 5.1 uses `%1` for the whole match if no capture exists.
                if found.captures().is_empty() && next == b'1' {
                    append_gsub(&mut output, &source[found.start()..found.end()])?;
                    index += 2;
                    continue;
                }
                let capture = found
                    .captures()
                    .get(usize::from(next - b'1'))
                    .ok_or_else(|| {
                        // Lua 5.3 and later name the capture.
                        if compatibility.is_lua53_or_later() {
                            RuntimeError::message(format!("invalid capture index %{}", next - b'0'))
                        } else {
                            RuntimeError::message("invalid capture index")
                        }
                    })?;
                if capture.is_position() {
                    append_gsub(
                        &mut output,
                        usize_to_number(capture.start() + 1)?.to_string().as_bytes(),
                    )?;
                } else {
                    append_gsub(&mut output, &source[capture.start()..capture.end()])?;
                }
            }
            // Lua 5.1 and LuaJIT copy any other character after a `%`.
            _ if compatibility.is_lua51_family() => append_gsub(&mut output, &[next])?,
            _ => {
                return Err(
                    RuntimeError::message("invalid use of '%' in replacement string").into(),
                );
            }
        }
        index += 2;
    }
    Ok(output)
}

/// `string.gsub` of `source` with a replacement that is literal text (no `%`),
/// at most `maximum` times: the new text and the number of replacements.
/// `None` for a pattern the matcher stops on, text too large to build, or a
/// result longer than `largest`, which the caller leaves to the general path
/// to answer or report. The match is read where the matcher holds it, so no
/// match is allocated.
pub(crate) fn gsub_literal(
    source: &[u8],
    pattern: &Pattern,
    template: &[u8],
    (maximum, modern, largest): (usize, bool, usize),
) -> Option<(Vec<u8>, usize)> {
    let length = source.len();
    let anchored = pattern.is_anchored();
    let mut output = Vec::new();
    output.try_reserve(length).ok()?;
    let mut cursor = 0;
    let mut search = 0;
    let mut count = 0_usize;
    let mut last_match = None;
    while count < maximum && search <= length {
        let found = pattern
            .find_with(source, search, |found| (found.start(), found.end()))
            .ok()?;
        let Some((start, end)) = found else {
            break;
        };
        // As `skip_gsub_empty_match` and `next_gsub_search` do.
        if modern && start == end && last_match == Some(end) {
            search = end.saturating_add(1);
            continue;
        }
        output.try_reserve(start - cursor + template.len()).ok()?;
        output.extend_from_slice(&source[cursor..start]);
        output.extend_from_slice(template);
        if output.len() > largest {
            return None;
        }
        cursor = end;
        last_match = Some(end);
        search = if anchored {
            length.saturating_add(1)
        } else if start != end || modern {
            end
        } else if end == length {
            length.saturating_add(1)
        } else {
            end + 1
        };
        count += 1;
    }
    output.try_reserve(length - cursor).ok()?;
    output.extend_from_slice(&source[cursor..]);
    Some((output, count))
}

pub(crate) fn append_gsub(output: &mut Vec<u8>, bytes: &[u8]) -> VmResult<()> {
    output
        .try_reserve(bytes.len())
        .map_err(|_| RuntimeError::message("string.gsub result is too large"))?;
    output.extend_from_slice(bytes);
    Ok(())
}

impl LoadPipeline {
    pub(crate) fn new(source_name: LuaString) -> Self {
        Self::Undecided {
            source_name,
            bytes: Vec::new(),
        }
    }
}

pub(crate) fn map_load_poll(poll: LoadPoll) -> PipelinePoll {
    match poll {
        LoadPoll::NeedInput(session) => PipelinePoll::NeedInput(LoadPipeline::Text(session)),
        LoadPoll::Loaded(chunk) => PipelinePoll::Loaded(chunk),
        LoadPoll::Failed(error) => PipelinePoll::Failed(error),
    }
}

impl RequireCompletion {
    pub(crate) const fn direct(instruction: Instruction) -> Self {
        Self::Direct(CallReturn {
            base: instruction.a(),
            results: instruction.c(),
        })
    }

    pub(crate) const fn protected(instruction: Instruction) -> Self {
        Self::Protected(CallReturn {
            base: instruction.a(),
            results: instruction.c(),
        })
    }
}

impl ModuleCompletion {
    pub(crate) const fn direct(instruction: Instruction) -> Self {
        Self::Direct(CallReturn {
            base: instruction.a(),
            results: instruction.c(),
        })
    }

    pub(crate) const fn protected(instruction: Instruction) -> Self {
        Self::Protected(CallReturn {
            base: instruction.a(),
            results: instruction.c(),
        })
    }
}

impl LoadCompletion {
    pub(crate) const fn direct(instruction: Instruction) -> Self {
        Self::Direct(CallReturn {
            base: instruction.a(),
            results: instruction.c(),
        })
    }

    pub(crate) const fn protected(instruction: Instruction) -> Self {
        Self::Protected(CallReturn {
            base: instruction.a(),
            results: instruction.c(),
        })
    }
}

impl DofileCompletion {
    pub(crate) const fn direct(instruction: Instruction) -> Self {
        Self::Direct(CallReturn {
            base: instruction.a(),
            results: instruction.c(),
        })
    }

    pub(crate) const fn protected(instruction: Instruction) -> Self {
        Self::Protected(CallReturn {
            base: instruction.a(),
            results: instruction.c(),
        })
    }
}

impl ForeachCompletion {
    pub(crate) const fn direct(instruction: Instruction) -> Self {
        Self::Direct(CallReturn {
            base: instruction.a(),
            results: instruction.c(),
        })
    }

    pub(crate) const fn protected(instruction: Instruction) -> Self {
        Self::Protected(CallReturn {
            base: instruction.a(),
            results: instruction.c(),
        })
    }
}

impl From<SortAfter> for SortPhase {
    fn from(after: SortAfter) -> Self {
        match after {
            SortAfter::Build { next_root } => Self::Build { next_root },
            SortAfter::Extract { next_end } => Self::Extract { next_end },
        }
    }
}

impl SortCompletion {
    pub(crate) const fn direct(instruction: Instruction) -> Self {
        Self::Direct(CallReturn {
            base: instruction.a(),
            results: instruction.c(),
        })
    }

    pub(crate) const fn protected(instruction: Instruction) -> Self {
        Self::Protected(CallReturn {
            base: instruction.a(),
            results: instruction.c(),
        })
    }
}

impl DebugHookEvent {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Call => "call",
            Self::TailCall => "tail call",
            Self::Return => "return",
            Self::TailReturn => "tail return",
            Self::Line(_) => "line",
            Self::Count => "count",
        }
    }

    /// The event's slot in `Vm::hook_event_names`.
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::Call => 0,
            Self::Return => 1,
            Self::TailReturn => 2,
            Self::Line(_) => 3,
            Self::Count => 4,
            Self::TailCall => 5,
        }
    }

    pub(crate) const fn line(self) -> Option<u32> {
        match self {
            Self::Line(line) => Some(line),
            Self::Call | Self::TailCall | Self::Return | Self::TailReturn | Self::Count => None,
        }
    }
}

pub(crate) fn reset_registers(registers: &mut Vec<Value>, required: usize) {
    if registers.len() != required {
        registers.clear();
        registers.resize(required, Value::Nil);
    }
}

/// Lua 5.3 and later skip an empty match that ends where the previous match
/// ended; earlier versions accept it.
#[inline(always)]
pub(crate) fn skip_gsub_empty_match(
    modern: bool,
    found: &PatternMatch,
    last_match: Option<usize>,
) -> bool {
    modern && found.start() == found.end() && last_match == Some(found.end())
}

/// Where `string.gsub` searches after an accepted match. Lua 5.3 and later
/// resume at its end, earlier versions step over a byte after an empty match.
#[inline(always)]
pub(crate) fn next_gsub_search(modern: bool, found: &PatternMatch, length: usize) -> usize {
    if found.start() != found.end() || modern {
        found.end()
    } else if found.end() == length {
        length.saturating_add(1)
    } else {
        found.end() + 1
    }
}
