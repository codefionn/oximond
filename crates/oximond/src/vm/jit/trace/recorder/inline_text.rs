//! `utf8.char`, `utf8.codepoint`, `utf8.len`, `utf8.offset`, `utf8.codes` and
//! the loops over its iterator, `string.pack`, `string.unpack` and
//! `string.packsize`, as calls of helpers.
//!
//! The recorder runs the native's probe on the live arguments (the
//! interpreter is about to make the same call) to learn how the call ends:
//! how many code points `utf8.codepoint` gives, whether `utf8.len` counts or
//! fails, the kinds of the values `string.unpack` gives. The recording calls
//! the helper of that shape (see
//! [`oximond::jit_ir::trace::HelperId::utf8_codepoint`]) and the helper misses
//! when a later call ends another way. A miss leaves through the `Before`
//! snapshot of the call, the interpreter makes it, and the guard takes a side
//! trace that records the call as it now ends. A call that raises an error
//! has no recording: the probe has no answer for it.
//!
//! The format of `string.pack` and `string.unpack` is an operand like any
//! other, which the native parses on each call. A trace never needs the
//! format to be the recorded one: a call whose values have other kinds
//! misses.
//!
//! `utf8.codes (s)` gives the step as its first result, a native that holds
//! no state, and the step is a call of that native with the string and the
//! position of the last character. Its helper gives the position and the code
//! point, or ends the loop.

use super::{
    Flow, HookPoint, Operand, RecKind, Recorder, SnapshotPhase, TraceAbortReason, TraceScalar,
    TraceType,
};
use crate::jit_ir::trace::{
    HelperId, MAX_HELPER_ARGS, MAX_PACK_VALUES, MAX_UNPACK_VALUES, MAX_UTF8_CODEPOINTS,
    pinned_native_identity,
};
use crate::runtime::{DirectNativeKind, NativeFunctionHandle, Value};

/// The guest instructions a call of these natives counts for, on top of the
/// call of a helper, when the policy weighs the work of a loop.
const TEXT_GUEST_WORK: u32 = 40;

impl Recorder {
    /// Whether `kind` is a function of this module.
    pub(super) const fn is_text_kind(kind: DirectNativeKind) -> bool {
        matches!(
            kind,
            DirectNativeKind::Utf8Char
                | DirectNativeKind::Utf8Codepoint
                | DirectNativeKind::Utf8Len
                | DirectNativeKind::Utf8Offset
                | DirectNativeKind::Utf8Codes
                | DirectNativeKind::Utf8Next
                | DirectNativeKind::Utf8Next53
                | DirectNativeKind::Utf8NextLax
                | DirectNativeKind::StringPack
                | DirectNativeKind::StringUnpack
                | DirectNativeKind::StringPacksize
        )
    }

    /// A constant integer operation.
    fn integer_constant(&mut self, pc: usize, value: i64) -> u32 {
        self.emit(
            pc,
            RecKind::Constant(TraceScalar::Integer(value)),
            Some(TraceType::Integer),
        )
    }

    /// A call of one of the natives of this module.
    #[allow(clippy::too_many_lines)]
    pub(super) fn record_text_native(
        &mut self,
        at: &HookPoint<'_>,
        (kind, function): (DirectNativeKind, NativeFunctionHandle),
        callee: Operand,
        (arguments, first_argument): (&[(Operand, TraceType)], usize),
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let text = |ty: TraceType| matches!(ty, TraceType::String | TraceType::StringRef);
        let scalar = |ty: TraceType| {
            matches!(
                ty,
                TraceType::Integer
                    | TraceType::Float
                    | TraceType::Boolean
                    | TraceType::Nil
                    | TraceType::String
                    | TraceType::StringRef
            )
        };
        let (wanted, source_first) = match kind {
            DirectNativeKind::Utf8Char => (0..=MAX_HELPER_ARGS - 1, false),
            DirectNativeKind::Utf8Codepoint | DirectNativeKind::Utf8Len => (1..=4, true),
            DirectNativeKind::Utf8Codes => (1..=2, true),
            DirectNativeKind::Utf8Next
            | DirectNativeKind::Utf8Next53
            | DirectNativeKind::Utf8NextLax => (2..=2, true),
            DirectNativeKind::StringPack => (1..=MAX_PACK_VALUES + 1, true),
            DirectNativeKind::Utf8Offset | DirectNativeKind::StringUnpack => (2..=3, true),
            _ => (1..=1, true),
        };
        if !wanted.contains(&arguments.len())
            || (source_first && !text(arguments[0].1))
            || !arguments.iter().all(|(_, ty)| scalar(*ty))
        {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        // `string.unpack (format, s)` reads the string second.
        if kind == DirectNativeKind::StringUnpack && !text(arguments[1].1) {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let runtime = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?
            .runtime;
        let live = at
            .registers
            .get(first_argument..first_argument + arguments.len())
            .ok_or(TraceAbortReason::Divergence)?;
        // The call that raises an error has no recording.
        let answer = if kind == DirectNativeKind::Utf8Codes {
            Vec::new()
        } else {
            runtime
                .probe_native_results(function, live)
                .ok_or(TraceAbortReason::UnsupportedOperand)?
        };
        self.host_work = self.host_work.saturating_add(TEXT_GUEST_WORK);
        let lane = self.pin(pinned_native_identity(function.index()))?;
        self.guard_callee(pc, callee, lane);
        let integers = |answer: &[Value]| {
            answer
                .iter()
                .all(|value| matches!(value, Value::Integer(_)))
        };
        let values = match kind {
            DirectNativeKind::Utf8Char | DirectNativeKind::StringPack => {
                if !matches!(answer.as_slice(), [Value::String(_)]) {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                let helper = if kind == DirectNativeKind::Utf8Char {
                    HelperId::UTF8_CHAR
                } else {
                    HelperId::STRING_PACK
                };
                vec![self.emit_helper_call(pc, helper, arguments)]
            }
            DirectNativeKind::StringPacksize => {
                if !matches!(answer.as_slice(), [Value::Integer(_)]) {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                vec![self.emit_helper_call(pc, HelperId::STRING_PACKSIZE, arguments)]
            }
            DirectNativeKind::Utf8Codepoint => {
                if answer.len() > MAX_UTF8_CODEPOINTS || !integers(&answer) {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                let helper = HelperId::utf8_codepoint(answer.len());
                let call = self.emit_helper_call(pc, helper, arguments);
                self.pattern_results(pc, call, vec![TraceType::Integer; answer.len()])?
            }
            DirectNativeKind::Utf8Len => match answer.as_slice() {
                [Value::Integer(_)] => {
                    vec![self.emit_helper_call(pc, HelperId::UTF8_LEN, arguments)]
                }
                [Value::Nil, Value::Integer(_)] => {
                    let position = self.emit_helper_call(pc, HelperId::UTF8_LEN_FAIL, arguments);
                    let nil = self.nil_constant(pc);
                    vec![nil, position]
                }
                _ => return Err(TraceAbortReason::UnsupportedOperand),
            },
            DirectNativeKind::Utf8Offset => match answer.as_slice() {
                [Value::Integer(_)] => {
                    vec![self.emit_helper_call(pc, HelperId::UTF8_OFFSET, arguments)]
                }
                [Value::Integer(_), Value::Integer(_)] => {
                    let call = self.emit_helper_call(pc, HelperId::UTF8_OFFSET_END, arguments);
                    self.pattern_results(pc, call, [TraceType::Integer; 2])?
                }
                [Value::Nil] => {
                    self.emit_helper_call(pc, HelperId::UTF8_OFFSET_NIL, arguments);
                    vec![self.nil_constant(pc)]
                }
                _ => return Err(TraceAbortReason::UnsupportedOperand),
            },
            DirectNativeKind::Utf8Next
            | DirectNativeKind::Utf8Next53
            | DirectNativeKind::Utf8NextLax => {
                let mode = match kind {
                    DirectNativeKind::Utf8Next => 0,
                    DirectNativeKind::Utf8NextLax => 1,
                    _ => 2,
                };
                let mode = self.integer_constant(pc, mode);
                let given = [
                    (Operand::Op(mode), TraceType::Integer),
                    arguments[0],
                    arguments[1],
                ];
                if answer.is_empty() {
                    self.emit_helper_call(pc, HelperId::UTF8_NEXT_END, &given);
                    Vec::new()
                } else if answer.len() == 2 && integers(&answer) {
                    let call = self.emit_helper_call(pc, HelperId::UTF8_NEXT, &given);
                    self.pattern_results(pc, call, [TraceType::Integer; 2])?
                } else {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
            }
            DirectNativeKind::StringUnpack => {
                let count = answer.len();
                if count == 0 || count > MAX_UNPACK_VALUES + 1 {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                let mut types = Vec::with_capacity(count);
                let mut kinds = 0;
                let mut weight = 1;
                for value in &answer[..count - 1] {
                    let (ty, digit) = match value {
                        Value::Integer(_) => (TraceType::Integer, 0),
                        Value::Number(_) => (TraceType::Float, 1),
                        Value::String(_) => (TraceType::String, 2),
                        _ => return Err(TraceAbortReason::UnsupportedOperand),
                    };
                    types.push(ty);
                    kinds += digit * weight;
                    weight *= 3;
                }
                if !matches!(answer[count - 1], Value::Integer(_)) {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                types.push(TraceType::Integer);
                let helper = HelperId::string_unpack(count - 1, kinds);
                let call = self.emit_helper_call(pc, helper, arguments);
                self.pattern_results(pc, call, types)?
            }
            DirectNativeKind::Utf8Codes => {
                return self.record_codes(at, pc, arguments, live, (base, results));
            }
            _ => return Err(TraceAbortReason::UnsupportedOperand),
        };
        Ok(self.native_results(pc, &values, (base, results)))
    }

    /// A constant nil operation.
    fn nil_constant(&mut self, pc: usize) -> u32 {
        self.emit(
            pc,
            RecKind::Constant(TraceScalar::Nil),
            Some(TraceType::Nil),
        )
    }

    /// `utf8.codes (s [, lax])`: the step, the string and the position 0,
    /// after the check the call makes of the string. The callee is guarded
    /// already.
    fn record_codes(
        &mut self,
        at: &HookPoint<'_>,
        pc: usize,
        arguments: &[(Operand, TraceType)],
        live: &[Value],
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        let runtime = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?
            .runtime;
        // Lua 5.3's call takes no second argument and accepts any string.
        let legacy = runtime
            .direct_native(DirectNativeKind::Utf8Next53)
            .is_some();
        let step = if legacy {
            DirectNativeKind::Utf8Next53
        } else {
            // Whether the call asks for the lax step is the truth of its
            // second argument: its type says it, but for a boolean, which the
            // trace guards.
            let lax = match arguments.get(1) {
                None | Some((_, TraceType::Nil)) => false,
                Some((operand, TraceType::Boolean)) => {
                    let truthy = live[1].is_truthy();
                    let before = self.snapshot(SnapshotPhase::Before { pc }, &[]);
                    self.emit(
                        pc,
                        RecKind::GuardBranch {
                            condition: *operand,
                            expected: truthy,
                            snapshot: before,
                        },
                        None,
                    );
                    truthy
                }
                Some(_) => true,
            };
            self.emit_helper_call(pc, HelperId::UTF8_CODES_CHECK, &arguments[..1]);
            if lax {
                DirectNativeKind::Utf8NextLax
            } else {
                DirectNativeKind::Utf8Next
            }
        };
        let step = runtime
            .direct_native(step)
            .ok_or(TraceAbortReason::UnsupportedOperand)?;
        let lane = self.pin(pinned_native_identity(step.index()))?;
        let iterator = self.emit(
            pc,
            RecKind::Constant(TraceScalar::Closure(u64::from(lane))),
            Some(TraceType::Closure),
        );
        let (source, ty) = arguments[0];
        let source = self.emit(pc, RecKind::Move(source), Some(ty));
        let start = self.integer_constant(pc, 0);
        Ok(self.native_results(pc, &[iterator, source, start], (base, results)))
    }
}
