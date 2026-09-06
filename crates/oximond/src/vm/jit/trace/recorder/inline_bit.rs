//! The `bit` and `bit32` libraries on plain numbers, as integer operations of
//! the trace.
//!
//! Every operand becomes a 32-bit word held in an integer operation (a value
//! in `0..2^32`), the operation works on those words with the profile-free
//! integer operations (`& | ~ << >>`, which follow Lua's integer semantics),
//! and the result converts back to what the library gives: a number of the
//! profile for `bit` (the word as a signed 32-bit integer, always a float) and
//! for `bit32` on a profile without integers (the unsigned word), an integer
//! for `bit32` on a profile with them.
//!
//! How an operand becomes a word is the library's own rule:
//!
//! - `bit` (`LuaJIT`'s `BitOp`) keeps the low 32 bits of an integer, and
//!   reads a float with the `2^52 + 2^51` rounding trick for every float
//!   ([`NumericOp::TrickBits`]).
//! - `bit32` on a profile without integers reads a float exactly as the
//!   trick does below `2^51` in magnitude and by its mathematical value
//!   above, so a guard sends a larger float, and a NaN, to the interpreter.
//! - `bit32` with integers takes an integer's low 32 bits and a float that is
//!   an integral value in the range of an integer; any other float is an error
//!   the interpreter raises, so a guard leaves for it.
//!
//! A shift count of `bit` is taken modulo 32. A count of `bit32` is an
//! integer of any size: the integer shift treats a count of 32 or more places
//! in either direction as clearing the word, which is `bit32`'s rule, and a
//! negative count shifts the other way. `arshift` and the field of `extract`
//! and `replace` are narrower. The recording guards `arshift`'s count into
//! `0..32` and the field and width of `extract` and `replace` against the ones
//! it saw, and the interpreter makes any call outside them, errors included.
//!
//! A call with a 64-bit integer argument (`bit` on `LuaJIT`) is not here: see
//! [`Recorder::record_wide_native`].

use super::{
    ArithmeticOp, Flow, HookPoint, Operand, RecKind, Recorder, TraceAbortReason, TraceScalar,
    TraceType, is_number,
};
use crate::jit_ir::trace::{NumericOp, pinned_native_identity};
use crate::runtime::{DirectNativeKind, NativeFunctionHandle};

/// The most operands a `band`, `bor`, `bxor` or `btest` call may fold.
const MAX_FOLDED_WORDS: usize = 4;

/// 2^51 as a float: below it the rounding trick is the exact rounding.
const TWO_51: f64 = 2_251_799_813_685_248.0;

/// How a library reads its operands and gives its result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Words {
    /// `bit`: the trick for a float, the low word of an integer, a float
    /// result holding the signed word.
    BitOp,
    /// `bit32` on a profile with floats only: the unsigned word as a float.
    Bit32Floats,
    /// `bit32` on a profile with integers: the unsigned word as an integer.
    Bit32Integers,
}

impl Recorder {
    /// Whether `kind` is a `bit` or `bit32` function this module records.
    pub(super) const fn is_bit_kind(kind: DirectNativeKind) -> bool {
        matches!(
            kind,
            DirectNativeKind::BitTobit
                | DirectNativeKind::BitBnot
                | DirectNativeKind::BitBand
                | DirectNativeKind::BitBor
                | DirectNativeKind::BitBxor
                | DirectNativeKind::BitLshift
                | DirectNativeKind::BitRshift
                | DirectNativeKind::BitArshift
                | DirectNativeKind::BitRol
                | DirectNativeKind::BitRor
                | DirectNativeKind::Bit32Band
                | DirectNativeKind::Bit32Bor
                | DirectNativeKind::Bit32Bxor
                | DirectNativeKind::Bit32Bnot
                | DirectNativeKind::Bit32Lshift
                | DirectNativeKind::Bit32Rshift
                | DirectNativeKind::Bit32Arshift
                | DirectNativeKind::Bit32Lrotate
                | DirectNativeKind::Bit32Rrotate
                | DirectNativeKind::Bit32Btest
                | DirectNativeKind::Bit32Extract
                | DirectNativeKind::Bit32Replace
        )
    }

    /// Records a call of a `bit` or `bit32` function on numbers, behind the
    /// callee guard. A call that takes open results (`hoisted`) may not
    /// emit a guard of its own: it has no snapshot to leave through.
    pub(super) fn record_bit_native(
        &mut self,
        at: &HookPoint<'_>,
        kind: DirectNativeKind,
        (callee, function): (Operand, NativeFunctionHandle),
        (arguments, first_argument): (&[(Operand, TraceType)], usize),
        (base, results): (usize, usize),
        hoisted: bool,
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        if arguments.is_empty()
            || arguments
                .iter()
                .any(|(_, ty)| !is_number(*ty) || ty.is_wide())
        {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let lane = self.pin(pinned_native_identity(function.index()))?;
        if hoisted {
            if !self.take_hoisted(callee) {
                return Err(TraceAbortReason::Divergence);
            }
        } else {
            self.guard_callee(pc, callee, lane);
        }
        let first_op = self.ops.len();
        let value = self.record_bit_operation(at, kind, arguments, first_argument)?;
        if hoisted
            && self.ops[first_op..]
                .iter()
                .any(|op| matches!(op.kind, RecKind::GuardBranch { .. }))
        {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        Ok(self.native_result(pc, value, (base, results)))
    }

    fn int_const(&mut self, pc: usize, value: i64) -> Operand {
        Operand::Op(self.scalar(pc, TraceScalar::Integer(value)))
    }

    fn int_op(&mut self, pc: usize, op: ArithmeticOp, left: Operand, right: Operand) -> Operand {
        Operand::Op(self.emit(
            pc,
            RecKind::Arithmetic(op, left, right),
            Some(TraceType::Integer),
        ))
    }

    /// The word `value` keeps: its low 32 bits.
    fn low_word(&mut self, pc: usize, value: Operand) -> Operand {
        let mask = self.int_const(pc, 0xffff_ffff);
        self.int_op(pc, ArithmeticOp::BitAnd, value, mask)
    }

    /// Guards that the float `value` is an integral number (not NaN or an
    /// infinity) in the range of an integer, and converts it to one.
    fn integral_float(&mut self, pc: usize, value: Operand) -> Operand {
        let floor = self.emit(
            pc,
            RecKind::Numeric(NumericOp::Floor, value),
            Some(TraceType::Float),
        );
        let integral = self.emit(
            pc,
            RecKind::Compare(super::Comparison::Equal, Operand::Op(floor), value),
            Some(TraceType::Boolean),
        );
        self.guard_flag(pc, Operand::Op(integral), true);
        self.guard_integer_range(pc, value);
        Operand::Op(self.emit(
            pc,
            RecKind::Numeric(NumericOp::FloatToInteger, value),
            Some(TraceType::Integer),
        ))
    }

    /// The 32-bit word of the number `(operand, ty)` as `words` reads it.
    fn bit_word(
        &mut self,
        pc: usize,
        words: Words,
        (operand, ty): (Operand, TraceType),
    ) -> Result<Operand, TraceAbortReason> {
        match (words, ty) {
            (Words::BitOp | Words::Bit32Integers, TraceType::Integer) => {
                Ok(self.low_word(pc, operand))
            }
            (Words::BitOp, TraceType::Float) => Ok(Operand::Op(self.emit(
                pc,
                RecKind::Numeric(NumericOp::TrickBits, operand),
                Some(TraceType::Integer),
            ))),
            (Words::Bit32Floats, TraceType::Float) => {
                // Below 2^51 the trick is the exact rounding. A NaN fails the
                // comparison too.
                let magnitude = self.emit(
                    pc,
                    RecKind::Numeric(NumericOp::Abs, operand),
                    Some(TraceType::Float),
                );
                let limit = self.scalar(pc, TraceScalar::Float(TWO_51));
                let small = self.emit(
                    pc,
                    RecKind::Compare(
                        super::Comparison::LessThan,
                        Operand::Op(magnitude),
                        Operand::Op(limit),
                    ),
                    Some(TraceType::Boolean),
                );
                self.guard_flag(pc, Operand::Op(small), true);
                Ok(Operand::Op(self.emit(
                    pc,
                    RecKind::Numeric(NumericOp::TrickBits, operand),
                    Some(TraceType::Integer),
                )))
            }
            (Words::Bit32Integers, TraceType::Float) => {
                let integer = self.integral_float(pc, operand);
                Ok(self.low_word(pc, integer))
            }
            _ => Err(TraceAbortReason::UnsupportedOperand),
        }
    }

    /// The count of a `bit32` shift or rotation, as an integer of any size.
    fn bit32_count(
        &mut self,
        pc: usize,
        words: Words,
        (operand, ty): (Operand, TraceType),
    ) -> Result<Operand, TraceAbortReason> {
        match (words, ty) {
            (Words::Bit32Integers, TraceType::Integer) => Ok(operand),
            (Words::Bit32Integers, TraceType::Float) => Ok(self.integral_float(pc, operand)),
            (Words::Bit32Floats, TraceType::Float) => {
                // A float count truncates toward zero; one out of range is an
                // edge case the interpreter keeps.
                self.guard_integer_range(pc, operand);
                Ok(Operand::Op(self.emit(
                    pc,
                    RecKind::Numeric(NumericOp::FloatToInteger, operand),
                    Some(TraceType::Integer),
                )))
            }
            _ => Err(TraceAbortReason::UnsupportedOperand),
        }
    }

    /// The word `value` (in `0..2^32`) as the signed 32-bit integer `bit`
    /// gives.
    fn signed_word(&mut self, pc: usize, value: Operand) -> Operand {
        let sign = self.int_const(pc, 0x8000_0000);
        let flipped = self.int_op(pc, ArithmeticOp::BitXor, value, sign);
        self.int_op(pc, ArithmeticOp::Subtract, flipped, sign)
    }

    /// The operation holding the result of the library function: the word
    /// `result` as the library gives it.
    fn bit_value(&mut self, pc: usize, words: Words, result: Operand) -> u32 {
        let to_float = |this: &mut Self, operand: Operand| {
            this.emit(pc, RecKind::IntToFloat(operand), Some(TraceType::Float))
        };
        match words {
            Words::BitOp => {
                let signed = self.signed_word(pc, result);
                to_float(self, signed)
            }
            Words::Bit32Floats => to_float(self, result),
            Words::Bit32Integers => self.emit(pc, RecKind::Move(result), Some(TraceType::Integer)),
        }
    }

    /// `word` rotated left by `count` (below 32).
    fn rotate_left_word(&mut self, pc: usize, word: Operand, count: Operand) -> Operand {
        let width = self.int_const(pc, 32);
        let back = self.int_op(pc, ArithmeticOp::Subtract, width, count);
        let high = self.int_op(pc, ArithmeticOp::ShiftLeft, word, count);
        let low = self.int_op(pc, ArithmeticOp::ShiftRight, word, back);
        let both = self.int_op(pc, ArithmeticOp::BitOr, high, low);
        self.low_word(pc, both)
    }

    /// `word` rotated right by `count` (below 32).
    fn rotate_right_word(&mut self, pc: usize, word: Operand, count: Operand) -> Operand {
        let width = self.int_const(pc, 32);
        let back = self.int_op(pc, ArithmeticOp::Subtract, width, count);
        let low = self.int_op(pc, ArithmeticOp::ShiftRight, word, count);
        let high = self.int_op(pc, ArithmeticOp::ShiftLeft, word, back);
        let both = self.int_op(pc, ArithmeticOp::BitOr, high, low);
        self.low_word(pc, both)
    }

    /// `word` shifted right with its sign bit copied in, by `count` in
    /// `0..32`.
    fn arithmetic_shift_word(&mut self, pc: usize, word: Operand, count: Operand) -> Operand {
        let top = self.int_const(pc, 31);
        let sign = self.int_op(pc, ArithmeticOp::ShiftRight, word, top);
        let fill = Operand::Op(self.emit(pc, RecKind::Negate(sign), Some(TraceType::Integer)));
        let width = self.int_const(pc, 32);
        let back = self.int_op(pc, ArithmeticOp::Subtract, width, count);
        let copies = self.int_op(pc, ArithmeticOp::ShiftLeft, fill, back);
        let low = self.int_op(pc, ArithmeticOp::ShiftRight, word, count);
        let both = self.int_op(pc, ArithmeticOp::BitOr, low, copies);
        self.low_word(pc, both)
    }

    /// The operation for one call of `kind` on `arguments` (all numbers).
    #[allow(clippy::too_many_lines)]
    fn record_bit_operation(
        &mut self,
        at: &HookPoint<'_>,
        kind: DirectNativeKind,
        arguments: &[(Operand, TraceType)],
        first_argument: usize,
    ) -> Result<u32, TraceAbortReason> {
        let pc = at.pc;
        let words = match kind {
            DirectNativeKind::BitTobit
            | DirectNativeKind::BitBnot
            | DirectNativeKind::BitBand
            | DirectNativeKind::BitBor
            | DirectNativeKind::BitBxor
            | DirectNativeKind::BitLshift
            | DirectNativeKind::BitRshift
            | DirectNativeKind::BitArshift
            | DirectNativeKind::BitRol
            | DirectNativeKind::BitRor => Words::BitOp,
            _ if self.integers => Words::Bit32Integers,
            _ => Words::Bit32Floats,
        };
        let count = arguments.len();
        let exactly = |wanted: usize| {
            if count == wanted {
                Ok(())
            } else {
                Err(TraceAbortReason::UnsupportedOperand)
            }
        };
        let mask = self.int_const(pc, 0xffff_ffff);
        let result = match kind {
            DirectNativeKind::BitTobit => {
                exactly(1)?;
                self.bit_word(pc, words, arguments[0])?
            }
            DirectNativeKind::BitBnot | DirectNativeKind::Bit32Bnot => {
                exactly(1)?;
                let word = self.bit_word(pc, words, arguments[0])?;
                self.int_op(pc, ArithmeticOp::BitXor, word, mask)
            }
            DirectNativeKind::BitBand
            | DirectNativeKind::BitBor
            | DirectNativeKind::BitBxor
            | DirectNativeKind::Bit32Band
            | DirectNativeKind::Bit32Bor
            | DirectNativeKind::Bit32Bxor
            | DirectNativeKind::Bit32Btest => {
                if count > MAX_FOLDED_WORDS {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                let op = match kind {
                    DirectNativeKind::BitBor | DirectNativeKind::Bit32Bor => ArithmeticOp::BitOr,
                    DirectNativeKind::BitBxor | DirectNativeKind::Bit32Bxor => ArithmeticOp::BitXor,
                    _ => ArithmeticOp::BitAnd,
                };
                let mut folded = self.bit_word(pc, words, arguments[0])?;
                for argument in &arguments[1..] {
                    let word = self.bit_word(pc, words, *argument)?;
                    folded = self.int_op(pc, op, folded, word);
                }
                if kind == DirectNativeKind::Bit32Btest {
                    let zero = self.int_const(pc, 0);
                    let none = self.emit(
                        pc,
                        RecKind::Compare(super::Comparison::Equal, folded, zero),
                        Some(TraceType::Boolean),
                    );
                    return Ok(self.emit(
                        pc,
                        RecKind::Not(Operand::Op(none)),
                        Some(TraceType::Boolean),
                    ));
                }
                folded
            }
            DirectNativeKind::BitLshift
            | DirectNativeKind::BitRshift
            | DirectNativeKind::BitArshift
            | DirectNativeKind::BitRol
            | DirectNativeKind::BitRor => {
                exactly(2)?;
                let word = self.bit_word(pc, words, arguments[0])?;
                let places = self.bit_word(pc, words, arguments[1])?;
                let thirty_one = self.int_const(pc, 31);
                let places = self.int_op(pc, ArithmeticOp::BitAnd, places, thirty_one);
                match kind {
                    DirectNativeKind::BitLshift => {
                        let shifted = self.int_op(pc, ArithmeticOp::ShiftLeft, word, places);
                        self.low_word(pc, shifted)
                    }
                    DirectNativeKind::BitRshift => {
                        self.int_op(pc, ArithmeticOp::ShiftRight, word, places)
                    }
                    DirectNativeKind::BitArshift => self.arithmetic_shift_word(pc, word, places),
                    DirectNativeKind::BitRol => self.rotate_left_word(pc, word, places),
                    _ => self.rotate_right_word(pc, word, places),
                }
            }
            DirectNativeKind::Bit32Lshift
            | DirectNativeKind::Bit32Rshift
            | DirectNativeKind::Bit32Arshift
            | DirectNativeKind::Bit32Lrotate
            | DirectNativeKind::Bit32Rrotate => {
                exactly(2)?;
                let word = self.bit_word(pc, words, arguments[0])?;
                let places = self.bit32_count(pc, words, arguments[1])?;
                match kind {
                    DirectNativeKind::Bit32Lshift => {
                        let shifted = self.int_op(pc, ArithmeticOp::ShiftLeft, word, places);
                        self.low_word(pc, shifted)
                    }
                    DirectNativeKind::Bit32Rshift => {
                        let shifted = self.int_op(pc, ArithmeticOp::ShiftRight, word, places);
                        self.low_word(pc, shifted)
                    }
                    DirectNativeKind::Bit32Arshift => {
                        // Only a count in 0..32; any other the interpreter
                        // makes. A negative count is a huge one unsigned.
                        let five = self.int_const(pc, 5);
                        let high = self.int_op(pc, ArithmeticOp::ShiftRight, places, five);
                        let zero = self.int_const(pc, 0);
                        let inside = self.emit(
                            pc,
                            RecKind::Compare(super::Comparison::Equal, high, zero),
                            Some(TraceType::Boolean),
                        );
                        self.guard_flag(pc, Operand::Op(inside), true);
                        self.arithmetic_shift_word(pc, word, places)
                    }
                    kind => {
                        let thirty_one = self.int_const(pc, 31);
                        let places = self.int_op(pc, ArithmeticOp::BitAnd, places, thirty_one);
                        if kind == DirectNativeKind::Bit32Lrotate {
                            self.rotate_left_word(pc, word, places)
                        } else {
                            self.rotate_right_word(pc, word, places)
                        }
                    }
                }
            }
            DirectNativeKind::Bit32Extract | DirectNativeKind::Bit32Replace => {
                let replacing = kind == DirectNativeKind::Bit32Replace;
                let fixed = if replacing { 3 } else { 2 };
                if count != fixed && count != fixed + 1 {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                // The field and the width are guarded against the ones seen,
                // so the shifts below are constants and the interpreter
                // raises the error for any other.
                let live = |index: usize| {
                    at.registers
                        .get(first_argument + index)
                        .and_then(Self::live_integer)
                        .ok_or(TraceAbortReason::UnsupportedOperand)
                };
                let field = live(fixed - 1)?;
                let width = if count > fixed { live(fixed)? } else { 1 };
                if field < 0 || width <= 0 || field >= 32 || width > 32 - field {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                self.guard_bit_field(pc, arguments[fixed - 1].0, field);
                if count > fixed {
                    self.guard_bit_field(pc, arguments[fixed].0, width);
                }
                let word = self.bit_word(pc, words, arguments[0])?;
                let field_const = self.int_const(pc, field);
                let width_mask = self.int_const(pc, (1_i64 << width) - 1);
                if replacing {
                    let replacement = self.bit_word(pc, words, arguments[1])?;
                    let field_mask =
                        self.int_op(pc, ArithmeticOp::ShiftLeft, width_mask, field_const);
                    let cleared = {
                        let inverse = self.int_op(pc, ArithmeticOp::BitXor, field_mask, mask);
                        self.int_op(pc, ArithmeticOp::BitAnd, word, inverse)
                    };
                    let kept = self.int_op(pc, ArithmeticOp::BitAnd, replacement, width_mask);
                    let moved = self.int_op(pc, ArithmeticOp::ShiftLeft, kept, field_const);
                    self.int_op(pc, ArithmeticOp::BitOr, cleared, moved)
                } else {
                    let moved = self.int_op(pc, ArithmeticOp::ShiftRight, word, field_const);
                    self.int_op(pc, ArithmeticOp::BitAnd, moved, width_mask)
                }
            }
            _ => return Err(TraceAbortReason::UnsupportedOperand),
        };
        Ok(self.bit_value(pc, words, result))
    }

    /// Guards that the number `operand` is the integral `value`.
    fn guard_bit_field(&mut self, pc: usize, operand: Operand, value: i64) {
        let expected = Operand::Op(self.number_scalar(pc, value));
        let same = self.emit(
            pc,
            RecKind::Compare(super::Comparison::Equal, operand, expected),
            Some(TraceType::Boolean),
        );
        self.guard_flag(pc, Operand::Op(same), true);
    }
}
