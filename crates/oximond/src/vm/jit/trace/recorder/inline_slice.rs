//! `string.sub` of a bound string whose result only `string.byte`, `#` and
//! `==` against a string constant read.
//!
//! `local slice = string.sub (source, i, j)` followed by
//! `string.byte (slice, 1, 4)` makes a string the program never keeps. The
//! recorder emits the call as [`HelperId::STRING_SLICE`], behind guards that
//! keep both indices in `-2^31 <= x < 2^31`, and notes it. A later reader of
//! that result is recorded against the source instead, behind the guards
//! `i >= 1` and `j <= #source`, so the substring is exactly `source[i..j]`:
//!
//! - `string.byte` with constant positions reads each byte as a
//!   [`RecKind::StringByte`] at `i + p - 1`, with `i + b - 1 <= j` guarded.
//! - `#slice` is `j - i + 1`, with `i <= j` guarded.
//! - `slice == "text"` guards `j - i == #text - 1`, then each byte against
//!   the text, up to the first byte the recording found different. Only the
//!   path the recording took is native; the other outcomes exit.
//!
//! The call stays in the trace. When nothing but snapshots read its result
//! the optimizer drops it, and the exits make the string from a `Slice`
//! entry (see [`oximond::jit_ir::trace::SliceRecipe`]). A run the readers do
//! not take (an empty, clipped or negative range) keeps the call and the
//! interpreter's rules.
//!
//! Every profile records this. A profile without integers truncates a float
//! index toward zero, whatever its size inside the 32-bit range. The call
//! does that itself, and so does an exit's `Slice` entry, but the readers do
//! arithmetic on the raw indices: `#slice` and the length compare of `==`
//! guard both indices whole first, and a byte read leaves on a fractional
//! key. A profile with integers takes only integer indices here, which need
//! no conversion. Neither raises an error for a number inside the range.

use super::{
    ArithmeticOp, BindingKind, Flow, HookPoint, Operand, RecKind, Recorder, SnapshotPhase, Sym,
    TraceAbortReason, TraceScalar, TraceType, is_number,
};
use crate::bytecode::{Constant, Instruction, OpCode};
use crate::jit_ir::Comparison;
use crate::jit_ir::trace::{HelperId, NumericOp, pinned_native_identity};
use crate::runtime::{NativeFunctionHandle, Value};

/// The most bytes of a substring one `string.byte` call reads natively.
const MAX_SLICE_BYTES: i64 = 16;

/// A `string.sub` call the recorder emitted as [`HelperId::STRING_SLICE`], so
/// that a later `string.byte` of its result can read the source instead.
#[derive(Clone, Copy, Debug)]
pub(super) struct SliceNote {
    /// The operation holding the call's result.
    call: u32,
    /// The window register of the bound source string.
    window: usize,
    /// The index operands of the call.
    first: Operand,
    last: Operand,
    /// The indices and the source length of the recorded run.
    live_first: i64,
    live_last: i64,
    length: i64,
}

impl Recorder {
    /// A call of `string.sub (s, i, j)` on a string bound at entry and two
    /// numbers of the profile's number type, or `None` for a call of another
    /// shape.
    pub(super) fn record_slice_sub(
        &mut self,
        at: &HookPoint<'_>,
        (callee, function): (Operand, NativeFunctionHandle),
        (arguments, first_argument): (&[(Operand, TraceType)], usize),
        (base, results): (usize, usize),
    ) -> Result<Option<Flow>, TraceAbortReason> {
        let pc = at.pc;
        let number = self.number_type();
        let [
            (string, TraceType::String),
            (first, first_type),
            (last, last_type),
        ] = arguments
        else {
            return Ok(None);
        };
        if *first_type != number || *last_type != number {
            return Ok(None);
        }
        let Ok(window) = self.bound_string(*string) else {
            return Ok(None);
        };
        let live = at
            .registers
            .get(first_argument..first_argument + 3)
            .ok_or(TraceAbortReason::Divergence)?;
        let (Value::String(text), Some(live_first), Some(live_last)) = (
            &live[0],
            Self::live_integer(&live[1]),
            Self::live_integer(&live[2]),
        ) else {
            return Ok(None);
        };
        let limit = 1_i64 << 31;
        if live_first.abs() >= limit || live_last.abs() >= limit {
            return Ok(None);
        }
        let length = i64::try_from(text.len()).map_err(|_| TraceAbortReason::Divergence)?;
        let lane = self.pin(pinned_native_identity(function.index()))?;
        self.guard_callee(pc, callee, lane);
        self.guard_small_index(pc, *first);
        self.guard_small_index(pc, *last);
        let call = self.emit_helper_call(pc, HelperId::STRING_SLICE, arguments);
        self.slices.push(SliceNote {
            call,
            window,
            first: *first,
            last: *last,
            live_first,
            live_last,
            length,
        });
        Ok(Some(self.native_result(pc, call, (base, results))))
    }

    /// Guards `-2^31 <= index < 2^31` for a number, which NaN fails.
    fn guard_small_index(&mut self, pc: usize, index: Operand) {
        let below = Operand::Op(self.number_scalar(pc, -(1_i64 << 31)));
        let above = Operand::Op(self.number_scalar(pc, 1_i64 << 31));
        let low = self.emit(
            pc,
            RecKind::Compare(Comparison::LessEqual, below, index),
            Some(TraceType::Boolean),
        );
        self.guard_flag(pc, Operand::Op(low), true);
        let high = self.emit(
            pc,
            RecKind::Compare(Comparison::LessThan, index, above),
            Some(TraceType::Boolean),
        );
        self.guard_flag(pc, Operand::Op(high), true);
    }

    /// The integer a constant operand stands for.
    fn constant_position(&self, operand: Operand) -> Option<i64> {
        let Operand::Op(index) = self.origin(operand) else {
            return None;
        };
        match self.ops[index as usize].kind {
            #[allow(clippy::cast_possible_truncation, reason = "checked to be small")]
            RecKind::Constant(TraceScalar::Float(number))
                if number.fract() == 0.0 && number.abs() < 1.0e9 =>
            {
                Some(number as i64)
            }
            RecKind::Constant(TraceScalar::Integer(number)) => Some(number),
            _ => None,
        }
    }

    /// A call of `string.byte (slice, a, b)` on the result of a noted
    /// `string.sub`, with constant positions the recording found inside the
    /// substring. `None` for any other call.
    pub(super) fn record_slice_byte(
        &mut self,
        at: &HookPoint<'_>,
        (callee, function): (Operand, NativeFunctionHandle),
        arguments: &[(Operand, TraceType)],
        (base, results): (usize, usize),
    ) -> Result<Option<Flow>, TraceAbortReason> {
        let pc = at.pc;
        if self.slices.is_empty() {
            return Ok(None);
        }
        let Some(&(string, TraceType::String)) = arguments.first() else {
            return Ok(None);
        };
        let Operand::Op(made) = self.origin(string) else {
            return Ok(None);
        };
        let Some(note) = self.slices.iter().find(|note| note.call == made).copied() else {
            return Ok(None);
        };
        if arguments.len() > 3 || !arguments[1..].iter().all(|(_, ty)| is_number(*ty)) {
            return Ok(None);
        }
        let position = |at: usize, default: Option<i64>| match arguments.get(at) {
            Some((operand, _)) => self.constant_position(*operand),
            None => default,
        };
        let Some(from) = position(1, Some(1)) else {
            return Ok(None);
        };
        let Some(to) = position(2, Some(from)) else {
            return Ok(None);
        };
        // Only a run inside the substring: the interpreter makes every call
        // that is empty, negative or clipped.
        let reaches = note.live_first.saturating_add(to - 1);
        if from < 1
            || to < from
            || to - from >= MAX_SLICE_BYTES
            || note.live_first < 1
            || reaches > note.live_last
            || reaches > note.length
        {
            return Ok(None);
        }
        let lane = self.pin(pinned_native_identity(function.index()))?;
        self.guard_callee(pc, callee, lane);
        let number = self.number_type();
        let binding = self.binding(note.window, BindingKind::Bytes { ty: number });
        let at_offset = |this: &mut Self, offset: i64| this.slice_offset(pc, note, offset);
        // `i >= 1`: a smaller start is clipped to the first byte.
        let one = Operand::Op(self.number_scalar(pc, 1));
        let positive = self.emit(
            pc,
            RecKind::Compare(Comparison::LessEqual, one, note.first),
            Some(TraceType::Boolean),
        );
        self.guard_flag(pc, Operand::Op(positive), true);
        // `i + to - 1 <= j`: the substring reaches the last byte read.
        let last = at_offset(self, to - 1);
        let covers = self.emit(
            pc,
            RecKind::Compare(Comparison::LessEqual, last, note.last),
            Some(TraceType::Boolean),
        );
        self.guard_flag(pc, Operand::Op(covers), true);
        let mut bytes = Vec::new();
        for offset in from - 1..to {
            let key = at_offset(self, offset);
            let snapshot = self.snapshot(SnapshotPhase::Before { pc }, &[]);
            bytes.push(self.emit(
                pc,
                RecKind::StringByte {
                    binding,
                    string: Operand::Carried(note.window),
                    key,
                    snapshot,
                },
                Some(number),
            ));
        }
        Ok(Some(self.native_results(pc, &bytes, (base, results))))
    }

    /// The source position `first + offset`.
    fn slice_offset(&mut self, pc: usize, note: SliceNote, offset: i64) -> Operand {
        if offset == 0 {
            return note.first;
        }
        let number = self.number_type();
        let step = Operand::Op(self.number_scalar(pc, offset));
        Operand::Op(self.emit(
            pc,
            RecKind::Arithmetic(ArithmeticOp::Add, note.first, step),
            Some(number),
        ))
    }

    /// The note of the call that made the substring in `register`, when the
    /// recording took the path the readers handle: a range inside the source
    /// (`1 <= i` and `j <= #source`), which makes the substring exactly
    /// `source[i..j]`.
    fn inside_slice(&self, register: usize) -> Option<SliceNote> {
        if self.slices.is_empty() {
            return None;
        }
        let Sym::Op(index) = self.sym(register) else {
            return None;
        };
        let Operand::Op(made) = self.origin(Operand::Op(index)) else {
            return None;
        };
        let note = self.slices.iter().find(|note| note.call == made).copied()?;
        (note.live_first >= 1 && note.live_last <= note.length).then_some(note)
    }

    /// Guards `1 <= i` and `j <= #source`, the two facts that make a noted
    /// substring `source[i..j]`, and in a profile without integers that both
    /// indices are whole: the call truncates a fractional one, and `j - i`
    /// of the raw floats would not be the substring's span.
    fn guard_slice_inside(&mut self, pc: usize, note: SliceNote) {
        if !self.integers {
            self.guard_whole(pc, note.first);
            self.guard_whole(pc, note.last);
        }
        let number = self.number_type();
        let one = Operand::Op(self.number_scalar(pc, 1));
        let positive = self.emit(
            pc,
            RecKind::Compare(Comparison::LessEqual, one, note.first),
            Some(TraceType::Boolean),
        );
        self.guard_flag(pc, Operand::Op(positive), true);
        let binding = self.binding(note.window, BindingKind::Bytes { ty: number });
        let length = self.emit(
            pc,
            RecKind::StringLength {
                binding,
                string: Operand::Carried(note.window),
            },
            Some(number),
        );
        let covered = self.emit(
            pc,
            RecKind::Compare(Comparison::LessEqual, note.last, Operand::Op(length)),
            Some(TraceType::Boolean),
        );
        self.guard_flag(pc, Operand::Op(covered), true);
    }

    /// Guards `floor(x) == x` for the float `index`, which the guards of
    /// [`Self::guard_small_index`] already keep finite.
    fn guard_whole(&mut self, pc: usize, index: Operand) {
        let floor = self.emit(
            pc,
            RecKind::Numeric(NumericOp::Floor, index),
            Some(TraceType::Float),
        );
        let whole = self.emit(
            pc,
            RecKind::Compare(Comparison::Equal, Operand::Op(floor), index),
            Some(TraceType::Boolean),
        );
        self.guard_flag(pc, Operand::Op(whole), true);
    }

    /// `j - i`, the substring's length less one when the range is not empty.
    fn slice_span(&mut self, pc: usize, note: SliceNote) -> u32 {
        let number = self.number_type();
        self.emit(
            pc,
            RecKind::Arithmetic(ArithmeticOp::Subtract, note.last, note.first),
            Some(number),
        )
    }

    /// `#slice` for the substring in register `source` of the instruction
    /// at `at`, written to register `target`: `j - i + 1`, for a range the
    /// recording found inside the source and not empty. `None` for any other
    /// length, which the call's result answers.
    pub(super) fn record_slice_length(
        &mut self,
        at: &HookPoint<'_>,
        (target, source): (usize, usize),
    ) -> Result<Option<usize>, TraceAbortReason> {
        let pc = at.pc;
        let Some(note) = self.inside_slice(source) else {
            return Ok(None);
        };
        if note.live_first > note.live_last {
            return Ok(None);
        }
        self.guard_slice_inside(pc, note);
        let ordered = self.emit(
            pc,
            RecKind::Compare(Comparison::LessEqual, note.first, note.last),
            Some(TraceType::Boolean),
        );
        self.guard_flag(pc, Operand::Op(ordered), true);
        let number = self.number_type();
        let span = self.slice_span(pc, note);
        let one = Operand::Op(self.number_scalar(pc, 1));
        let length = self.emit(
            pc,
            RecKind::Arithmetic(ArithmeticOp::Add, Operand::Op(span), one),
            Some(number),
        );
        self.write(target, length);
        Ok(Some(pc + 1))
    }

    /// `slice == "text"` for the substring in register `register` and a
    /// string constant, as the outcome the recording saw. The guards leave
    /// before the comparison unless the length and the bytes up to the first
    /// difference go the recorded way. `None` for a comparison this does not
    /// handle (no noted substring, a range outside the source, an empty or
    /// long text).
    pub(super) fn record_slice_equality(
        &mut self,
        at: &HookPoint<'_>,
        register: usize,
        text: &[u8],
    ) -> Result<Option<bool>, TraceAbortReason> {
        let pc = at.pc;
        let Ok(count) = i64::try_from(text.len()) else {
            return Ok(None);
        };
        if !(1..=MAX_SLICE_BYTES).contains(&count) {
            return Ok(None);
        }
        let Some(note) = self.inside_slice(register) else {
            return Ok(None);
        };
        let Some(Value::String(live)) = at.registers.get(register) else {
            return Ok(None);
        };
        let live = live.as_bytes();
        let outcome = live == text;
        self.guard_slice_inside(pc, note);
        // `j - i + 1 == #text`. A range that is empty or reversed gives a
        // smaller number, so the length compare alone tells.
        let span = self.slice_span(pc, note);
        let wanted = Operand::Op(self.number_scalar(pc, count - 1));
        let same_length = self.emit(
            pc,
            RecKind::Compare(Comparison::Equal, Operand::Op(span), wanted),
            Some(TraceType::Boolean),
        );
        let live_same_length = note.live_last.saturating_sub(note.live_first) == count - 1;
        self.guard_flag(pc, Operand::Op(same_length), live_same_length);
        if !live_same_length {
            return Ok(Some(false));
        }
        let number = self.number_type();
        let binding = self.binding(note.window, BindingKind::Bytes { ty: number });
        for (position, (&byte, &live_byte)) in text.iter().zip(live).enumerate() {
            let key = self.slice_offset(pc, note, i64::try_from(position).unwrap_or(0));
            let snapshot = self.snapshot(SnapshotPhase::Before { pc }, &[]);
            let read = self.emit(
                pc,
                RecKind::StringByte {
                    binding,
                    string: Operand::Carried(note.window),
                    key,
                    snapshot,
                },
                Some(number),
            );
            let want = Operand::Op(self.number_scalar(pc, i64::from(byte)));
            let equal = self.emit(
                pc,
                RecKind::Compare(Comparison::Equal, Operand::Op(read), want),
                Some(TraceType::Boolean),
            );
            self.guard_flag(pc, Operand::Op(equal), byte == live_byte);
            if byte != live_byte {
                break;
            }
        }
        Ok(Some(outcome))
    }

    /// [`Self::record_slice_equality`] for two registers, one of them a
    /// noted substring and the other a string constant.
    ///
    /// A non-constant string on the other side keeps the call: the byte
    /// guards need the text at record time, and a runtime string's length
    /// and bytes are unbounded trace inputs. `sink_slices` cannot do it
    /// later, since the equality reads the call's result.
    pub(super) fn slice_equals_register(
        &mut self,
        at: &HookPoint<'_>,
        left: usize,
        right: usize,
    ) -> Result<Option<bool>, TraceAbortReason> {
        if self.slices.is_empty() {
            return Ok(None);
        }
        let (register, text) = if let Some(text) = self.constant_text(at.registers, right) {
            (left, text.to_vec())
        } else if let Some(text) = self.constant_text(at.registers, left) {
            (right, text.to_vec())
        } else {
            return Ok(None);
        };
        self.record_slice_equality(at, register, &text)
    }

    /// [`Self::record_slice_equality`] for an equality test word: a register
    /// against a constant of the chunk, or against another register.
    pub(super) fn slice_equals_test(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        opcode: OpCode,
    ) -> Result<Option<bool>, TraceAbortReason> {
        if self.slices.is_empty() {
            return Ok(None);
        }
        let a = usize::from(instruction.a());
        if opcode.is_test_constant() {
            let Some(Constant::String(text)) = at.chunk.constants.get(usize::from(instruction.b()))
            else {
                return Ok(None);
            };
            return self.record_slice_equality(at, a, text.as_bytes());
        }
        self.slice_equals_register(at, a, usize::from(instruction.b()))
    }
}
